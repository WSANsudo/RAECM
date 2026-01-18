"""
模型评估模块
Model evaluation module for Network Device Analyzer Training System
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from .config import (
    ModelConfig,
    InferenceConfig,
    load_config,
    get_model_config,
    get_inference_config
)

# 设置日志 - 同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 同时将 print 输出重定向到日志文件
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

import sys
_log_file = open('log.txt', 'a', encoding='utf-8')
sys.stdout = TeeOutput(sys.__stdout__, _log_file)


@dataclass
class EvaluationMetrics:
    """评估指标"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    support: int = 0
    confusion_matrix: Optional[List[List[int]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'support': self.support,
            'confusion_matrix': self.confusion_matrix
        }


@dataclass
class EvaluationReport:
    """评估报告"""
    overall_accuracy: float = 0.0  # 主要字段准确率
    field_metrics: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    inference_speed: float = 0.0  # tokens/second
    total_samples: int = 0
    valid_samples: int = 0  # 实际参与准确率计算的样本数
    parse_failures: int = 0  # JSON解析失败数
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'primary_accuracy': self.overall_accuracy,  # 主要字段准确率
            'field_metrics': {k: v.to_dict() for k, v in self.field_metrics.items()},
            'inference_speed': self.inference_speed,
            'total_samples': self.total_samples,
            'valid_samples': self.valid_samples,  # 实际参与准确率计算的样本数
            'parse_failures': self.parse_failures,  # JSON解析失败数
            'total_time': self.total_time
        }


def calculate_metrics(
    predictions: List[str],
    labels: List[str],
    field_name: str = ""
) -> EvaluationMetrics:
    """
    计算评估指标
    
    Args:
        predictions: 预测值列表
        labels: 真实标签列表
        field_name: 字段名称
        
    Returns:
        评估指标
    """
    # 处理None值
    predictions = [str(p) if p is not None else 'null' for p in predictions]
    labels = [str(l) if l is not None else 'null' for l in labels]
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算精确率、召回率、F1（使用weighted平均处理多分类）
    try:
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    except Exception:
        precision = recall = f1 = 0.0
    
    # 计算混淆矩阵
    try:
        cm = confusion_matrix(labels, predictions)
        cm_list = cm.tolist()
    except Exception:
        cm_list = None
    
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        support=len(labels),
        confusion_matrix=cm_list
    )


class ModelEvaluator:
    """模型评估器"""
    
    # 评估字段 - 每种专业模型只评估对应的唯一字段
    # vendor 模型 -> ['vendor']
    # os 模型 -> ['os']  
    # devicetype 模型 -> ['type']
    # 由 evaluate.py 在运行时根据 --mt 参数设置
    EVAL_FIELDS = ['vendor']  # 默认值，会被 evaluate.py 覆盖
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict] = None,
        base_model_path: Optional[str] = None
    ):
        """
        初始化评估器
        
        Args:
            model_path: 训练好的模型路径（LoRA权重或完整模型）
            config: 配置字典
            base_model_path: 基础模型路径（用于LoRA）
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.config = config or {}
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # 模型配置
        self.max_length = self.config.get('model', {}).get('max_length', 1024)
        
        # 推理配置
        self.inference_config = InferenceConfig(
            temperature=self.config.get('inference', {}).get('temperature', 0.1),
            top_p=self.config.get('inference', {}).get('top_p', 0.9),
            max_new_tokens=self.config.get('inference', {}).get('max_new_tokens', 512),
            do_sample=self.config.get('inference', {}).get('do_sample', False)
        )
    
    def _build_prompt(self, system_prompt: str, user_content: str) -> str:
        """根据模型类型构建 prompt"""
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'phi' in model_name:
            # Phi 格式
            return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_content}<|end|>\n<|assistant|>\n"
        elif 'llama' in model_name:
            # Llama 格式
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif 'qwen3' in model_name or 'qwen/qwen3' in model_name:
            # Qwen3 格式 - 禁用思考模式
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Qwen/ChatML 格式
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    def _is_local_path(self, path: str) -> bool:
        """检查是否是本地路径（必须存在）"""
        # 只有路径真正存在时才认为是本地路径
        # 这样可以避免 local_files_only=True 但路径不存在的错误
        return os.path.exists(path)
    
    def _resolve_local_model_path(self, model_path: str) -> str:
        """
        解析模型路径，优先使用本地模型
        
        检查顺序:
        1. 原始路径
        2. models/ 目录下
        3. models/{vendor,os,devicetype}/ 子目录下
        4. models/ 目录下（替换 / 为 -）
        """
        # 1. 原始路径存在
        if os.path.exists(model_path):
            return model_path
        
        # 2. 检查 models/ 目录
        models_dir = "./models"
        
        # 提取模型名称
        if '/' in model_path:
            model_name = model_path.split('/')[-1]
        elif '\\' in model_path:
            model_name = model_path.split('\\')[-1]
        else:
            model_name = model_path
        
        # 2.1 直接在 models/ 下查找
        local_path = os.path.join(models_dir, model_name)
        if os.path.exists(local_path):
            logger.info(f"找到本地模型: {local_path}")
            return local_path
        
        # 2.2 在 models/ 子目录下查找 (vendor, os, devicetype)
        for subdir in ['vendor', 'os', 'devicetype']:
            local_path = os.path.join(models_dir, subdir, model_name)
            if os.path.exists(local_path):
                logger.info(f"找到本地模型: {local_path}")
                return local_path
        
        # 3. 尝试完整路径替换 / 为 -
        if '/' in model_path:
            local_path = os.path.join(models_dir, model_path.replace('/', '-'))
            if os.path.exists(local_path):
                logger.info(f"找到本地模型: {local_path}")
                return local_path
            
            # 尝试组织名/模型名格式
            local_path = os.path.join(models_dir, model_path)
            if os.path.exists(local_path):
                logger.info(f"找到本地模型: {local_path}")
                return local_path
        
        # 4. 遍历 models/ 下所有子目录查找
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for d in dirs:
                    if d == model_name:
                        found_path = os.path.join(root, d)
                        logger.info(f"找到本地模型: {found_path}")
                        return found_path
        
        # 未找到本地模型，返回原始路径（可能需要网络下载）
        logger.warning(f"未找到本地模型 '{model_path}'，将尝试从网络加载")
        return model_path
    
    def load_model(self) -> None:
        """加载模型用于评估"""
        logger.info(f"加载模型: {self.model_path}")
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 检查是否是LoRA模型
        adapter_config_path = os.path.join(self.model_path, 'adapter_config.json')
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora:
            logger.info("检测到LoRA模型")
            
            # 确定基础模型路径
            if self.base_model_path is None:
                # 尝试从adapter_config读取
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                self.base_model_path = adapter_config.get('base_model_name_or_path')
            
            if self.base_model_path is None:
                raise ValueError("LoRA模型需要指定base_model_path")
            
            # 尝试解析本地模型路径
            local_base_path = self._resolve_local_model_path(self.base_model_path)
            is_local = os.path.exists(local_base_path)
            logger.info(f"基础模型: {local_base_path} (本地: {is_local})")
            
            if not is_local:
                logger.error(f"基础模型路径不存在: {local_base_path}")
                logger.info("请检查模型是否已下载到 models/ 目录")
                raise FileNotFoundError(f"基础模型不存在: {local_base_path}")
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                local_base_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None,
                local_files_only=True
            )
            
            # 加载LoRA权重
            logger.info(f"加载LoRA权重: {self.model_path}")
            peft_model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # 合并LoRA权重到基础模型（推理时更高效）
            logger.info("合并LoRA权重...")
            self.model = peft_model.merge_and_unload()
            logger.info("LoRA权重合并完成")
            
            # 调试：测试基础模型是否正常
            print(f"【调试】模型类型: {type(self.model)}")
            print(f"【调试】模型设备: {next(self.model.parameters()).device}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_base_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 打印模型信息
            logger.info(f"模型参数量: {self.model.num_parameters() / 1e6:.1f}M")
        else:
            logger.info("加载完整模型")
            
            local_model_path = self._resolve_local_model_path(self.model_path)
            is_local = os.path.exists(local_model_path)
            logger.info(f"模型路径: {local_model_path} (本地: {is_local})")
            
            if not is_local:
                logger.error(f"模型路径不存在: {local_model_path}")
                raise FileNotFoundError(f"模型不存在: {local_model_path}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None,
                local_files_only=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("模型加载完成")
    
    def inference_single(self, evidence: List[Dict], services: List[str] = None) -> Dict:
        """
        对单条数据进行推理
        
        Args:
            evidence: 证据列表
            services: 检测到的服务列表
            
        Returns:
            分析结果
        """
        # 构建输入
        from .data_processor import SYSTEM_PROMPT, DataProcessor
        
        processor = DataProcessor()
        user_content = processor.format_evidence(evidence, services)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # 尝试使用 chat_template，如果不存在则使用默认格式
        try:
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                raise ValueError("No chat template")
        except (ValueError, AttributeError):
            # 根据模型类型选择格式
            prompt = self._build_prompt(SYSTEM_PROMPT, user_content)
        
        # 分词
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # 生成
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.inference_config.max_new_tokens,
                "do_sample": self.inference_config.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            # 只在采样模式下添加 temperature 和 top_p
            if self.inference_config.do_sample:
                gen_kwargs["temperature"] = self.inference_config.temperature
                gen_kwargs["top_p"] = self.inference_config.top_p
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # 解码
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # 解析JSON
        try:
            # 尝试提取JSON部分
            response = response.strip()
            if response.startswith('{'):
                end_idx = response.rfind('}') + 1
                response = response[:end_idx]
            result = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"无法解析响应为JSON: {response[:100]}...")
            result = {
                'vendor': None,
                'model': None,
                'os': None,
                'type': None,
                'primary_usage': 'unknown',
                'industry': None,
                '_raw_response': response
            }
        
        return result
    
    def evaluate_field(
        self,
        predictions: List[str],
        labels: List[str],
        field_name: str
    ) -> EvaluationMetrics:
        """
        评估单个字段的性能
        
        Args:
            predictions: 预测值列表
            labels: 真实标签列表
            field_name: 字段名称
            
        Returns:
            评估指标
        """
        return calculate_metrics(predictions, labels, field_name)
    
    def _parse_json_response(self, response: str) -> Dict:
        """改进的 JSON 解析，处理各种格式问题，始终返回字典"""
        import re
        
        def ensure_dict(obj):
            """确保返回字典类型"""
            if isinstance(obj, dict):
                return obj
            return {}
        
        response = response.strip()
        
        # 0. 处理 Qwen3 的思考模式输出 (移除 <think>...</think> 部分)
        if '<think>' in response:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # 0.1 移除特殊 token 和前缀文本
        response = re.sub(r'<\|.*?\|>', '', response).strip()
        
        # 1. 直接尝试解析
        try:
            result = json.loads(response)
            return ensure_dict(result)
        except:
            pass
        
        # 2. 去除 markdown 代码块
        if '```json' in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                    return ensure_dict(result)
                except:
                    pass
        
        if '```' in response:
            match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                    return ensure_dict(result)
                except:
                    pass
        
        # 3. 提取嵌套 JSON（处理多层括号）
        start = response.find('{')
        if start != -1:
            depth = 0
            for i, c in enumerate(response[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = response[start:i+1]
                        try:
                            result = json.loads(json_str)
                            return ensure_dict(result)
                        except:
                            # 尝试修复常见问题
                            fixed = self._fix_json_string(json_str)
                            try:
                                result = json.loads(fixed)
                                return ensure_dict(result)
                            except:
                                pass
                        break
        
        # 4. 尝试提取简单 JSON 块
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return ensure_dict(result)
            except:
                # 尝试修复后解析
                fixed = self._fix_json_string(json_match.group())
                try:
                    result = json.loads(fixed)
                    return ensure_dict(result)
                except:
                    pass
        
        # 5. 尝试从关键字提取字段值
        result = self._extract_fields_from_text(response)
        if result:
            return result
        
        # 6. 返回空结果
        return {}
    
    def _fix_json_string(self, json_str: str) -> str:
        """尝试修复常见的 JSON 格式问题"""
        import re
        
        # 移除尾部多余逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 修复单引号
        json_str = json_str.replace("'", '"')
        
        # 修复没有引号的键
        json_str = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1"\2":', json_str)
        
        # 修复 None -> null
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        json_str = re.sub(r'\bNULL\b', 'null', json_str)
        
        # 修复 True/False
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        
        # 修复没有引号的值 (简单情况)
        # 例如: {"vendor":MikroTik} -> {"vendor":"MikroTik"}
        json_str = re.sub(r':(\s*)([A-Za-z][A-Za-z0-9_-]*)\s*([,}])', r':"\2"\3', json_str)
        
        return json_str
    
    def _extract_fields_from_text(self, text: str) -> Dict:
        """从非 JSON 文本中尝试提取字段值"""
        import re
        
        result = {}
        
        # 定义字段模式
        patterns = {
            'vendor': r'vendor["\s:]+["\']?([^"\'}\n,]+)',
            'model': r'model["\s:]+["\']?([^"\'}\n,]+)',
            'os': r'(?:^|["\s])os["\s:]+["\']?([^"\'}\n,]+)',
            'type': r'type["\s:]+["\']?([^"\'}\n,]+)',
            'primary_usage': r'primary_usage["\s:]+["\']?([^"\'}\n,]+)',
            'industry': r'industry["\s:]+["\']?([^"\'}\n,]+)',
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip().strip('"\'')
                if value.lower() in ('null', 'none', ''):
                    result[field] = None
                else:
                    result[field] = value
        
        return result if result else None
    
    def _normalize_value(self, value, field_name: str) -> str:
        """标准化字段值用于比较"""
        import re
        
        if value is None:
            return 'null'
        
        # 处理 model 字段的数组格式
        if field_name == 'model':
            if isinstance(value, list) and len(value) > 0:
                value = value[0][0] if isinstance(value[0], list) else value[0]
        
        # 转为小写字符串
        normalized = str(value).lower().strip()
        
        # 厂商名称标准化映射
        if field_name == 'vendor':
            vendor_mapping = {
                'juniper networks': 'juniper',
                'juniper network': 'juniper',
                'cisco systems': 'cisco',
                'cisco system': 'cisco',
                'huawei technologies': 'huawei',
                'mikrotik': 'mikrotik',
                'routeros': 'mikrotik',
                'fortinet': 'fortinet',
                'fortigate': 'fortinet',
                'ubiquiti': 'ubiquiti',
                'ubnt': 'ubiquiti',
                'unifi': 'ubiquiti',
            }
            normalized = vendor_mapping.get(normalized, normalized)
        
        # OS 字段：提取核心 OS 名称，忽略版本号，支持别名
        elif field_name == 'os':
            # 先移除版本号 (数字、点、连字符组成的版本串)
            # 例如: "RouterOS 6.49.10" -> "RouterOS"
            # 例如: "KeeneticOS 4.03.C.6.3-9" -> "KeeneticOS"
            os_name = re.sub(r'\s*[\d]+[\d.\-a-zA-Z]*$', '', normalized).strip()
            os_name = re.sub(r'\s*v[\d]+[\d.\-]*$', '', os_name).strip()  # 处理 v6.49 格式
            
            # OS 别名映射 - 统一到标准名称
            os_mapping = {
                # MikroTik RouterOS 系列
                'routeros': 'routeros',
                'mikrotik': 'routeros',
                'mikrotik routeros': 'routeros',
                'ros': 'routeros',
                
                # Cisco IOS 系列
                'ios': 'ios',
                'cisco ios': 'ios',
                'cisco': 'ios',
                'ios-xe': 'ios',
                'ios xe': 'ios',
                'iosxe': 'ios',
                'ios-xr': 'ios',
                'ios xr': 'ios',
                'iosxr': 'ios',
                'nx-os': 'nxos',
                'nxos': 'nxos',
                'nexus': 'nxos',
                
                # Juniper JunOS 系列
                'junos': 'junos',
                'juniper': 'junos',
                'juniper junos': 'junos',
                
                # Huawei VRP 系列
                'vrp': 'vrp',
                'huawei vrp': 'vrp',
                'huawei': 'vrp',
                'versatile routing platform': 'vrp',
                
                # Fortinet FortiOS 系列
                'fortios': 'fortios',
                'fortigate': 'fortios',
                'fortinet': 'fortios',
                
                # Keenetic 系列
                'keeneticos': 'keeneticos',
                'keenetic': 'keeneticos',
                
                # ZyXEL 系列
                'zynos': 'zynos',
                'zyxel': 'zynos',
                
                # Ubiquiti 系列
                'edgeos': 'edgeos',
                'ubiquiti': 'edgeos',
                'ubnt': 'edgeos',
                'unifi': 'edgeos',
                'edgemax': 'edgeos',
                
                # Synology 系列
                'dsm': 'dsm',
                'synology': 'dsm',
                'diskstation': 'dsm',
                
                # QNAP 系列
                'qts': 'qts',
                'qnap': 'qts',
                
                # Linux 系列
                'linux': 'linux',
                'ubuntu': 'linux',
                'debian': 'linux',
                'centos': 'linux',
                'redhat': 'linux',
                'rhel': 'linux',
                'fedora': 'linux',
                
                # Windows 系列
                'windows': 'windows',
                'windows server': 'windows',
                'win': 'windows',
                
                # FreeBSD 系列
                'freebsd': 'freebsd',
                'bsd': 'freebsd',
                
                # PAN-OS (Palo Alto)
                'pan-os': 'panos',
                'panos': 'panos',
                'palo alto': 'panos',
                
                # Arista EOS
                'eos': 'eos',
                'arista': 'eos',
                'arista eos': 'eos',
                
                # HP/Aruba
                'arubaos': 'arubaos',
                'aruba': 'arubaos',
                'procurve': 'arubaos',
                
                # Dell/EMC
                'dell os': 'dellos',
                'dellos': 'dellos',
                'force10': 'dellos',
                
                # Yamaha
                'yamaha': 'yamaha',
                'rtx': 'yamaha',
            }
            
            normalized = os_mapping.get(os_name, os_name)
        
        return normalized
    
    def evaluate(self, test_path: str, batch_size: int = 4) -> EvaluationReport:
        """
        在测试集上进行完整评估（优化版，支持批处理）
        
        Args:
            test_path: 测试数据路径
            batch_size: 批处理大小
            
        Returns:
            评估报告
        """
        logger.info(f"开始评估: {test_path}")
        
        # 加载测试数据
        test_data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
        
        logger.info(f"测试样本数: {len(test_data)}")
        
        # 预处理：提取所有 prompts 和 labels
        # 使用测试数据中的 system prompt，而不是导入的全局变量
        
        prompts = []
        labels = []
        valid_indices = []
        
        for i, sample in enumerate(test_data):
            messages = sample['messages']
            label_str = messages[-1]['content']
            try:
                label = json.loads(label_str)
            except:
                continue
            
            # 从测试数据中获取 system prompt 和 user content
            system_prompt = messages[0]['content']
            user_content = messages[1]['content']
            
            # 构建 prompt - 禁用 Qwen3 思考模式
            try:
                if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_content}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False  # 禁用思考模式
                        )
                    except TypeError:
                        # 不支持 enable_thinking 参数
                        prompt = self.tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_content}],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                else:
                    raise ValueError("No chat template")
            except (ValueError, AttributeError):
                prompt = self._build_prompt(system_prompt, user_content)
            
            prompts.append(prompt)
            labels.append(label)
            valid_indices.append(i)
        
        logger.info(f"有效样本数: {len(prompts)}")
        
        # 调试：打印第一个样本的信息
        if prompts:
            print("=" * 60)
            print("【调试】第一个样本示例:")
            print(f"Prompt 长度: {len(prompts[0])} 字符")
            print(f"Prompt 内容:\n{prompts[0][:800]}")
            print(f"\n期望标签: {labels[0]}")
            print("=" * 60)
        
        # 收集预测和标签
        all_predictions = {field: [] for field in self.EVAL_FIELDS}
        all_labels = {field: [] for field in self.EVAL_FIELDS}
        
        total_tokens = 0
        start_time = time.time()
        parse_failures = 0
        
        # 批处理推理
        from tqdm import tqdm
        
        # 减少 max_new_tokens 加速推理
        max_new_tokens = min(self.inference_config.max_new_tokens, 256)
        
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="评估进度", unit="batch"):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end]
            
            # 批量分词（左填充用于生成）
            self.tokenizer.padding_side = 'left'
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # 批量生成
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,  # 评估用贪婪解码
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "num_beams": 1,  # 贪婪解码
                    "repetition_penalty": 1.5,  # 防止重复输出
                }
                
                # 调试：打印第一个 batch 的输入信息
                if batch_start == 0:
                    print(f"\n【调试】输入 shape: {inputs['input_ids'].shape}")
                    print(f"【调试】max_new_tokens: {max_new_tokens}")
                
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # 调试：打印输出信息
                if batch_start == 0:
                    print(f"【调试】输出 shape: {outputs.shape}")
            
            total_tokens += outputs.numel()
            
            # 解码每个样本
            for j, (output, label) in enumerate(zip(outputs, batch_labels)):
                # 提取生成的部分
                # 左填充时，输入长度就是 input_ids 的总长度
                input_len = inputs['input_ids'].shape[1]
                generated = output[input_len:]
                
                # 调试：打印生成的 token ID
                if batch_start == 0 and j < 2:
                    print(f"\n【调试】样本 {j+1}:")
                    print(f"  生成的 token IDs (前20个): {generated[:20].tolist()}")
                    print(f"  pad_token_id: {self.tokenizer.pad_token_id}")
                    print(f"  eos_token_id: {self.tokenizer.eos_token_id}")
                
                response = self.tokenizer.decode(generated, skip_special_tokens=True)
                
                # 解析 JSON
                prediction = self._parse_json_response(response)
                
                # 调试：打印前几个样本的实际输出
                if batch_start == 0 and j < 2:
                    print(f"\n【调试】样本 {j+1} 模型输出:")
                    print(f"原始响应: {response[:500]}")
                    print(f"解析结果: {prediction}")
                    print(f"期望标签: {batch_labels[j]}")
                
                # 确保 prediction 是字典
                is_parse_failure = False
                if not isinstance(prediction, dict):
                    prediction = {}
                    is_parse_failure = True
                    parse_failures += 1
                    if parse_failures <= 5:
                        print(f"【警告】JSON解析返回非字典类型: {type(prediction).__name__}")
                        print(f"  原始响应: {response[:300]}...")
                elif not prediction:
                    is_parse_failure = True
                    parse_failures += 1
                    if parse_failures <= 5:  # 只打印前几个失败案例
                        print(f"【警告】JSON解析失败，响应内容:")
                        print(f"  {response[:300]}...")
                
                # 只有解析成功的样本才纳入准确率计算
                if not is_parse_failure:
                    for field in self.EVAL_FIELDS:
                        pred_val = self._normalize_value(prediction.get(field), field)
                        label_val = self._normalize_value(label.get(field), field)
                        all_predictions[field].append(pred_val)
                        all_labels[field].append(label_val)
        
        # 恢复 padding_side
        self.tokenizer.padding_side = 'right'
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算实际参与评估的样本数
        valid_samples = len(prompts) - parse_failures
        logger.info(f"JSON解析失败数: {parse_failures}/{len(prompts)} ({100*parse_failures/len(prompts):.1f}%)")
        logger.info(f"实际参与准确率计算的样本数: {valid_samples}")
        
        # 计算各字段指标
        field_metrics = {}
        
        for field in self.EVAL_FIELDS:
            metrics = self.evaluate_field(
                all_predictions[field],
                all_labels[field],
                field
            )
            field_metrics[field] = metrics
            logger.info(f"{field}: Accuracy={metrics.accuracy:.4f}, F1={metrics.f1_score:.4f}")
        
        # 计算推理速度
        inference_speed = total_tokens / total_time if total_time > 0 else 0
        
        # 使用主要评估字段的准确率作为 overall_accuracy
        primary_field = self.EVAL_FIELDS[0] if self.EVAL_FIELDS else 'vendor'
        primary_accuracy = field_metrics[primary_field].accuracy if primary_field in field_metrics else 0.0
        
        report = EvaluationReport(
            overall_accuracy=primary_accuracy,
            field_metrics=field_metrics,
            inference_speed=inference_speed,
            total_samples=len(test_data),
            valid_samples=valid_samples,
            parse_failures=parse_failures,
            total_time=total_time
        )
        
        logger.info(f"\n评估完成!")
        logger.info(f"{primary_field} 准确率: {primary_accuracy:.4f} (基于 {valid_samples} 个有效样本)")
        logger.info(f"推理速度: {inference_speed:.1f} tokens/s")
        logger.info(f"总耗时: {total_time:.1f}s")
        
        return report
    
    def generate_report(
        self,
        report: EvaluationReport,
        output_path: str
    ) -> None:
        """
        生成评估报告
        
        Args:
            report: 评估报告
            output_path: 报告输出路径
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # 保存JSON格式
        json_path = output_path if output_path.endswith('.json') else output_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 生成Markdown格式
        md_path = output_path.replace('.json', '.md') if output_path.endswith('.json') else output_path + '.md'
        
        # 获取主要评估字段
        primary_field = list(report.field_metrics.keys())[0] if report.field_metrics else 'unknown'
        
        md_content = f"""# 模型评估报告

## 概览

| 指标 | 值 |
|------|-----|
| 评估字段 | {primary_field} |
| 准确率 | {report.overall_accuracy:.4f} |
| 测试样本数 | {report.total_samples} |
| 推理速度 | {report.inference_speed:.1f} tokens/s |
| 总耗时 | {report.total_time:.1f}s |

## 评估结果

| 字段 | 准确率 | 精确率 | 召回率 | F1分数 | 样本数 |
|------|--------|--------|--------|--------|--------|
"""
        
        for field, metrics in report.field_metrics.items():
            md_content += f"| {field} | {metrics.accuracy:.4f} | {metrics.precision:.4f} | {metrics.recall:.4f} | {metrics.f1_score:.4f} | {metrics.support} |\n"
        
        md_content += "\n## 指标说明\n\n"
        md_content += "- **准确率**: 预测正确的样本比例\n"
        md_content += "- **精确率/召回率/F1**: 使用weighted平均计算\n"
        md_content += "- **推理速度**: 每秒生成的token数量\n"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"报告已保存: {json_path}, {md_path}")


def evaluate_model(
    model_path: str,
    test_path: str,
    config_path: str = "config.yaml",
    base_model_path: str = None,
    output_path: str = None
) -> EvaluationReport:
    """
    评估模型的便捷函数
    
    Args:
        model_path: 模型路径
        test_path: 测试数据路径
        config_path: 配置文件路径
        base_model_path: 基础模型路径
        output_path: 报告输出路径
        
    Returns:
        评估报告
    """
    # 加载配置
    try:
        config = load_config(config_path)
    except:
        config = {}
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path, config, base_model_path)
    
    # 加载模型
    evaluator.load_model()
    
    # 评估
    report = evaluator.evaluate(test_path)
    
    # 生成报告
    if output_path:
        evaluator.generate_report(report, output_path)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='模型评估工具')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--test', type=str, required=True, help='测试数据路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--base-model', type=str, help='基础模型路径（LoRA模型需要）')
    parser.add_argument('--output', type=str, default='evaluation_report', help='报告输出路径')
    
    args = parser.parse_args()
    
    evaluate_model(
        args.model,
        args.test,
        args.config,
        args.base_model,
        args.output
    )
