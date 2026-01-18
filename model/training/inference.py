"""
推理服务模块
Inference service module for Network Device Analyzer Training System
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None

from .config import InferenceConfig, load_config, get_inference_config
from .data_processor import SYSTEM_PROMPT, DataProcessor
from .model_manager import ModelManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceService:
    """推理服务"""
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict] = None,
        base_model_path: Optional[str] = None
    ):
        """
        初始化推理服务
        
        Args:
            model_path: 训练好的模型路径
            config: 配置字典
            base_model_path: 基础模型路径（用于LoRA）
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.config = config or {}
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # 推理配置
        infer_cfg = self.config.get('inference', {})
        self.inference_config = InferenceConfig(
            temperature=infer_cfg.get('temperature', 0.1),
            top_p=infer_cfg.get('top_p', 0.9),
            max_new_tokens=infer_cfg.get('max_new_tokens', 512),
            do_sample=infer_cfg.get('do_sample', False)
        )
        
        self.data_processor = DataProcessor()
    
    def load_model(self) -> None:
        """加载模型"""
        logger.info(f"加载模型: {self.model_path}")
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 使用模型管理器确保模型可用
        model_manager = ModelManager()
        
        # 检查是否是LoRA模型
        adapter_config_path = os.path.join(self.model_path, 'adapter_config.json')
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora:
            logger.info("检测到LoRA模型")
            
            # 确定基础模型路径
            if self.base_model_path is None:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                self.base_model_path = adapter_config.get('base_model_name_or_path')
            
            if self.base_model_path is None:
                raise ValueError("LoRA模型需要指定base_model_path")
            
            # 确保基础模型可用
            self.base_model_path = model_manager.ensure_model(self.base_model_path)
            logger.info(f"基础模型: {self.base_model_path}")
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None
            )
            
            # 加载LoRA权重
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
        else:
            logger.info("加载完整模型")
            
            # 确保模型可用
            resolved_path = model_manager.ensure_model(self.model_path)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("模型加载完成")
    
    def predict(
        self,
        evidence: List[Dict],
        services: List[str] = None,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None
    ) -> Dict:
        """
        执行单条预测
        
        Args:
            evidence: 证据列表
            services: 检测到的服务列表
            temperature: 生成温度（覆盖默认值）
            top_p: top-p采样参数（覆盖默认值）
            max_new_tokens: 最大生成token数（覆盖默认值）
            
        Returns:
            预测结果
        """
        # 使用传入参数或默认配置
        temperature = temperature if temperature is not None else self.inference_config.temperature
        top_p = top_p if top_p is not None else self.inference_config.top_p
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        
        # 构建输入
        user_content = self.data_processor.format_evidence(evidence, services)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # 应用chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        
        # 分词
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # 准备停止标记 - 支持多种模型
        stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        
        special_tokens = ['<|im_end|>', '<|endoftext|>', '\n', '<|end|>', '<|eot_id|>']
        for token in special_tokens:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id and token_id not in stop_token_ids:
                    stop_token_ids.append(token_id)
            except:
                pass
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=self.inference_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=stop_token_ids if stop_token_ids else self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # 清理预测结果 - 移除所有可能的特殊标记
        for token in ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '<|startoftext|>', '<|end|>', '<|eot_id|>', '<|im', '<|']:
            response = response.replace(token, '')
        
        # 解析JSON
        try:
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
    
    def batch_predict(
        self,
        evidence_list: List[List[Dict]],
        services_list: List[List[str]] = None,
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        批量预测
        
        Args:
            evidence_list: 证据列表的列表
            services_list: 服务列表的列表
            batch_size: 批次大小（当前实现为逐条处理）
            show_progress: 是否显示进度条
            
        Returns:
            预测结果列表
        """
        results = []
        
        if services_list is None:
            services_list = [None] * len(evidence_list)
        
        iterator = zip(evidence_list, services_list)
        if show_progress:
            iterator = tqdm(list(iterator), desc="推理中")
        
        for evidence, services in iterator:
            result = self.predict(evidence, services)
            results.append(result)
        
        return results
    
    def predict_from_raw(self, raw_data: Dict) -> Dict:
        """
        从原始数据格式进行预测
        
        Args:
            raw_data: 原始数据记录（包含evidence和services_detected字段）
            
        Returns:
            预测结果
        """
        evidence = raw_data.get('evidence', [])
        usage_evidence = raw_data.get('usage_evidence', [])
        services = raw_data.get('services_detected', [])
        
        # 合并证据
        all_evidence = evidence + [e for e in usage_evidence if e not in evidence]
        
        return self.predict(all_evidence, services)


def create_inference_service(
    model_path: str,
    config_path: str = "config.yaml",
    base_model_path: str = None
) -> InferenceService:
    """
    创建推理服务的便捷函数
    
    Args:
        model_path: 模型路径
        config_path: 配置文件路径
        base_model_path: 基础模型路径
        
    Returns:
        推理服务实例
    """
    try:
        config = load_config(config_path)
    except:
        config = {}
    
    service = InferenceService(model_path, config, base_model_path)
    service.load_model()
    
    return service


def interactive_inference(service: InferenceService) -> None:
    """
    交互式推理模式
    
    Args:
        service: 推理服务实例
    """
    print("\n" + "="*50)
    print("交互式推理模式")
    print("输入证据信息进行分析，输入 'quit' 退出")
    print("="*50 + "\n")
    
    while True:
        print("\n请输入证据信息（JSON格式的证据列表）:")
        print("示例: [{\"src\": \"ftp-21\", \"val\": \"220 MikroTik FTP server ready\", \"weight\": 0.9}]")
        
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("退出交互模式")
            break
        
        try:
            evidence = json.loads(user_input)
            if not isinstance(evidence, list):
                evidence = [evidence]
            
            print("\n正在分析...")
            result = service.predict(evidence)
            
            print("\n分析结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        except json.JSONDecodeError:
            print("错误: 输入不是有效的JSON格式")
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='推理服务')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--base-model', type=str, help='基础模型路径')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--input', type=str, help='输入文件路径（JSONL格式）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 创建服务
    service = create_inference_service(args.model, args.config, args.base_model)
    
    if args.interactive:
        interactive_inference(service)
    elif args.input:
        # 批量推理
        print(f"从文件加载数据: {args.input}")
        
        data = []
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        print(f"共 {len(data)} 条数据")
        
        results = []
        for item in tqdm(data, desc="推理中"):
            result = service.predict_from_raw(item)
            result['ip'] = item.get('ip', '')
            results.append(result)
        
        # 保存结果
        output_path = args.output or 'inference_results.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"结果已保存到: {output_path}")
    else:
        print("请指定 --interactive 或 --input 参数")
