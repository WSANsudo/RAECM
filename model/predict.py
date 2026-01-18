#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预测脚本
Model Prediction Script for Network Device Analyzer

使用方法:
    # 使用默认测试数据进行预测
    python predict.py --mt vd --model output/simple/vendor/Qwen2.5-1.5B-Instruct/best_model
    
    # 指定输入文件
    python predict.py --mt vd --model output/simple/vendor/Qwen2.5-1.5B-Instruct/best_model --input data/vendor/test.jsonl
    
    # 快速测试
    python predict.py --mt vd --max 50
"""

import argparse
import os
import sys
import json
import re
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.config import load_config, get_model_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 模型类型映射
MODEL_TYPE_MAP = {
    'vd': {
        'name': 'vendor',
        'description': '厂商识别模型',
        'output_subdir': 'vendor',
    },
    'os': {
        'name': 'os',
        'description': '操作系统识别模型',
        'output_subdir': 'os',
    },
    'dt': {
        'name': 'devicetype',
        'description': '设备类型识别模型',
        'output_subdir': 'devicetype',
    }
}


class Predictor:
    """模型预测器"""
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict] = None,
        base_model_path: Optional[str] = None,
        mode: str = 'simple'
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.config = config or {}
        self.mode = mode  # 保存训练模式
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # 推理配置 - simple 模式只输出标签，使用较小的 max_new_tokens
        self.max_length = self.config.get('model', {}).get('max_length', 1024)
        self.max_new_tokens = 10  # simple 模式只输出标签
    
    def _is_local_path(self, path: str) -> bool:
        """检查是否是本地路径"""
        return os.path.exists(path)
    
    def _resolve_local_model_path(self, model_path: str) -> str:
        """解析模型路径，优先使用本地模型"""
        if os.path.exists(model_path):
            return model_path
        
        models_dir = "./models"
        if '/' in model_path:
            model_name = model_path.split('/')[-1]
        else:
            model_name = model_path
        
        # 在 models/ 下查找
        local_path = os.path.join(models_dir, model_name)
        if os.path.exists(local_path):
            return local_path
        
        # 在子目录下查找
        for subdir in ['vendor', 'os', 'devicetype']:
            local_path = os.path.join(models_dir, subdir, model_name)
            if os.path.exists(local_path):
                return local_path
        
        return model_path
    
    def _build_prompt(self, system_prompt: str, user_content: str) -> str:
        """根据模型类型构建 prompt"""
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'llama' in model_name:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Qwen/ChatML 格式
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    def load_model(self) -> None:
        """加载模型"""
        logger.info(f"加载模型: {self.model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 检查是否是 LoRA 模型
        adapter_config_path = os.path.join(self.model_path, 'adapter_config.json')
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora:
            logger.info("检测到 LoRA 模型")
            
            if self.base_model_path is None:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                self.base_model_path = adapter_config.get('base_model_name_or_path')
            
            if self.base_model_path is None:
                raise ValueError("LoRA 模型需要指定 base_model_path")
            
            local_base_path = self._resolve_local_model_path(self.base_model_path)
            logger.info(f"基础模型: {local_base_path}")
            
            if not os.path.exists(local_base_path):
                raise FileNotFoundError(f"基础模型不存在: {local_base_path}")
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                local_base_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None,
                local_files_only=True
            )
            
            # 加载 LoRA 权重并合并
            logger.info(f"加载 LoRA 权重: {self.model_path}")
            peft_model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = peft_model.merge_and_unload()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_base_path,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            logger.info("加载完整模型")
            local_model_path = self._resolve_local_model_path(self.model_path)
            
            if not os.path.exists(local_model_path):
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
        logger.info(f"模型加载完成，参数量: {self.model.num_parameters() / 1e6:.1f}M")

    
    def _parse_json_response(self, response: str) -> Dict:
        """解析 JSON 响应"""
        response = response.strip()
        
        # 移除 Qwen3 思考模式输出
        if '<think>' in response:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # 移除特殊 token
        response = re.sub(r'<\|.*?\|>', '', response).strip()
        
        # 直接解析
        try:
            return json.loads(response)
        except:
            pass
        
        # 提取 JSON 块
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
                            return json.loads(json_str)
                        except:
                            pass
                        break
        
        return {"_raw_response": response, "_parse_error": True}
    
    def predict_batch(
        self,
        samples: List[Dict],
        batch_size: int = 4
    ) -> List[Dict]:
        """批量预测"""
        results = []
        
        # 预处理所有 prompts
        prompts = []
        for sample in samples:
            messages = sample['messages']
            system_prompt = messages[0]['content']
            user_content = messages[1]['content']
            
            try:
                if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_content}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                    except TypeError:
                        prompt = self.tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_content}],
                            tokenize=False,
                            add_generation_prompt=True
                        )
                else:
                    prompt = self._build_prompt(system_prompt, user_content)
            except:
                prompt = self._build_prompt(system_prompt, user_content)
            
            prompts.append(prompt)
        
        # 批量推理
        self.tokenizer.padding_side = 'left'
        
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="预测进度"):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_samples = samples[batch_start:batch_end]
            
            # 分词
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
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
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=stop_token_ids if stop_token_ids else self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # 与 evaluate.py 保持一致
                )
            
            # 解码
            input_len = inputs['input_ids'].shape[1]
            for j, (output, sample) in enumerate(zip(outputs, batch_samples)):
                generated = output[input_len:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                
                # 清理预测结果 - 移除所有可能的特殊标记
                for token in ['<|end_of_text|>', '<|im_end|>', '<|im_start|>', '<|endoftext|>', '<|startoftext|>', '<|end|>', '<|eot_id|>', '<|im', '<|']:
                    response = response.replace(token, '')
                response = response.split('\n')[0].strip()
                
                # 解析预测结果
                prediction = self._parse_json_response(response)
                
                # 获取原始标签
                label = {}
                try:
                    label = json.loads(sample['messages'][-1]['content'])
                except:
                    pass
                
                # 构建结果
                result = {
                    "system_prompt": sample['messages'][0]['content'],
                    "user_input": sample['messages'][1]['content'],
                    "prediction": prediction,
                    "label": label,
                    "raw_response": response[:500] if len(response) > 500 else response
                }
                results.append(result)
        
        self.tokenizer.padding_side = 'right'
        return results



def get_default_model_path(config: dict, model_type: str) -> str:
    """获取默认模型路径，优先使用 best_model"""
    type_info = MODEL_TYPE_MAP.get(model_type, MODEL_TYPE_MAP['vd'])
    
    # 简化训练模式的路径
    simple_base = os.path.join('./output', 'simple', type_info['name'])
    if os.path.exists(simple_base):
        # 查找最新的模型
        for model_name in os.listdir(simple_base):
            model_dir = os.path.join(simple_base, model_name)
            if os.path.isdir(model_dir):
                for subdir in ['best_model', 'final_model']:
                    candidate = os.path.join(model_dir, subdir)
                    if os.path.exists(candidate):
                        return candidate
    
    # 回退到旧路径
    train_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    
    base_output = train_cfg.get('output_dir', './output')
    model_path = model_cfg.get('name_or_path', './models/Qwen3-32B')
    model_name = os.path.basename(model_path.rstrip('/\\'))
    
    base_dir = os.path.join(base_output, type_info['output_subdir'], model_name)
    
    for subdir in ['best_model', 'best_model_sft', 'final_model', 'sft_final']:
        candidate = os.path.join(base_dir, subdir)
        if os.path.exists(candidate):
            return candidate
    
    return os.path.join(base_dir, 'final_model')


def get_model_output_dir(model_path: str) -> str:
    """
    从模型路径获取模型输出目录
    
    例如:
        model_path = './output/simple/vendor/Qwen2.5-1.5B-Instruct/best_model'
        返回: './output/simple/vendor/Qwen2.5-1.5B-Instruct'
    """
    if model_path.endswith('best_model') or model_path.endswith('final_model'):
        return os.path.dirname(model_path)
    return model_path


def list_available_models(config: dict, model_type: str) -> list:
    """列出可用模型"""
    type_info = MODEL_TYPE_MAP.get(model_type, MODEL_TYPE_MAP['vd'])
    
    models = []
    
    # 简化训练模式的路径
    simple_base = os.path.join('./output', 'simple', type_info['name'])
    if os.path.exists(simple_base):
        for model_name in os.listdir(simple_base):
            model_dir = os.path.join(simple_base, model_name)
            if os.path.isdir(model_dir):
                for subdir in ['best_model', 'final_model']:
                    model_path = os.path.join(model_dir, subdir)
                    if os.path.exists(model_path):
                        models.append((model_name, model_path, subdir))
    
    # 旧路径（兼容性）
    train_cfg = config.get('training', {})
    base_output = train_cfg.get('output_dir', './output')
    task_dir = os.path.join(base_output, type_info['output_subdir'])
    
    if os.path.exists(task_dir):
        for item in os.listdir(task_dir):
            item_path = os.path.join(task_dir, item)
            if os.path.isdir(item_path):
                for subdir in ['best_model', 'best_model_sft', 'final_model', 'sft_final']:
                    model_path = os.path.join(item_path, subdir)
                    if os.path.exists(model_path):
                        # 避免重复
                        if not any(m[1] == model_path for m in models):
                            models.append((item, model_path, subdir))
                        break
    
    return models


def select_model_interactive(config: dict, model_type: str) -> str:
    """交互式选择模型"""
    models = list_available_models(config, model_type)
    
    if not models:
        return None
    
    print("\n可用模型:")
    print("-" * 50)
    for i, (name, path, mtype) in enumerate(models, 1):
        print(f"  [{i}] {name} ({mtype})")
    print("-" * 50)
    
    while True:
        try:
            choice = input(f"请选择模型 [1-{len(models)}]: ").strip()
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx][1]
        except (ValueError, KeyboardInterrupt):
            return None


def main():
    parser = argparse.ArgumentParser(
        description='网络设备分析模型预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py --mt vd
  python predict.py --mt vd --model output/simple/vendor/Qwen2.5-1.5B-Instruct/best_model
  python predict.py --mt vd --input data/vendor/test.jsonl
  python predict.py --mt vd --max 50
        """
    )
    
    parser.add_argument('--mt', type=str, default='vd',
                        choices=['vd', 'os', 'dt'],
                        help='模型类型: vd=厂商, os=操作系统, dt=设备类型')
    parser.add_argument('--mode', type=str, default='simple',
                        choices=['simple'],
                        help='训练模式: simple=简化分类 (default: simple)')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径')
    parser.add_argument('--input', type=str, default=None,
                        help='输入数据路径 (默认: data/{task}/simple_test.jsonl)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径 (默认: 模型专用文件夹/prediction_{timestamp}.json)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--base-model', type=str, default=None,
                        help='基础模型路径 (LoRA 模型需要)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批处理大小')
    parser.add_argument('--max', '-m', type=int, default=None,
                        dest='max_samples',
                        help='最大预测样本数')
    
    args = parser.parse_args()
    
    type_info = MODEL_TYPE_MAP.get(args.mt, MODEL_TYPE_MAP['vd'])
    is_simple_mode = args.mode == 'simple'
    
    logger.info("=" * 60)
    logger.info(f"预测任务: {type_info['description']}")
    logger.info(f"训练模式: {args.mode}")
    logger.info("=" * 60)
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"配置文件 {args.config} 不存在，使用默认配置")
        config = {}
    
    # 确定模型路径
    model_path = args.model or get_default_model_path(config, args.mt)
    
    if not os.path.exists(model_path):
        logger.warning(f"默认模型路径不存在: {model_path}")
        model_path = select_model_interactive(config, args.mt)
        if model_path is None:
            logger.error("未选择模型，退出")
            sys.exit(1)
    
    # 从模型路径提取模型名
    model_output_dir = get_model_output_dir(model_path)
    path_parts = model_output_dir.replace('\\', '/').split('/')
    model_name = path_parts[-1] if path_parts else None
    
    # 确定输入路径 - 优先使用模型专用数据目录
    if args.input:
        input_path = args.input
    else:
        base_dir = config.get('data', {}).get('output_dir', './data')
        
        # 优先使用模型专用数据目录
        if model_name:
            model_task_dir = os.path.join(base_dir, model_name, type_info['name'])
            for filename in ['simple_test.jsonl', 'test.jsonl']:
                candidate = os.path.join(model_task_dir, filename)
                if os.path.exists(candidate):
                    input_path = candidate
                    break
            else:
                input_path = None
        else:
            input_path = None
        
        # 回退到旧格式目录
        if not input_path:
            task_dir = os.path.join(base_dir, type_info['name'])
            for filename in ['simple_test.jsonl', 'test.jsonl']:
                candidate = os.path.join(task_dir, filename)
                if os.path.exists(candidate):
                    input_path = candidate
                    break
            else:
                input_path = os.path.join(base_dir, 'test.jsonl')
    
    if not os.path.exists(input_path):
        logger.error(f"输入数据不存在: {input_path}")
        if model_name:
            logger.info(f"请确保数据在: data/{model_name}/{type_info['name']}/test.jsonl")
        sys.exit(1)
    
    logger.info(f"模型路径: {model_path}")
    logger.info(f"输入数据: {input_path}")
    
    # 获取模型输出目录（用于保存转换数据和预测结果）
    model_output_dir = get_model_output_dir(model_path)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 确定输出路径 - 默认保存到模型专用文件夹
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(model_output_dir, f"prediction_{timestamp}.json")
    
    logger.info(f"输出文件: {output_path}")
    logger.info("-" * 60)
    
    # 检查并转换测试数据格式（与 evaluate.py 保持一致）
    def convert_test_data_if_needed(test_data_path: str, task_type: str, model_output_dir: str) -> str:
        """检查测试数据格式，如果是原始格式则转换为简化格式
        
        Args:
            test_data_path: 原始测试数据路径
            task_type: 任务类型 (vendor/os/devicetype)
            model_output_dir: 模型输出目录
            
        Returns:
            转换后的数据路径
        """
        # 检查第一条数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return test_data_path
            
            item = json.loads(first_line)
            messages = item.get('messages', [])
            if len(messages) < 3:
                return test_data_path
            
            # 检查 assistant 的内容是否是 JSON
            assistant_content = messages[-1].get('content', '')
            try:
                json.loads(assistant_content)
                # 是 JSON 格式，需要转换
                logger.info("检测到原始格式测试数据，转换为简化格式...")
                
                from training.simple_classifier import convert_to_simple_format
                
                # 保存到模型输出目录
                converted_path = os.path.join(model_output_dir, 'predict_data.jsonl')
                
                # 转换
                convert_to_simple_format(test_data_path, converted_path, task_type)
                logger.info(f"✓ 已转换为简化格式: {converted_path}")
                return converted_path
            except (json.JSONDecodeError, KeyError):
                # 已经是简化格式
                return test_data_path
    
    # 转换数据格式
    converted_input_path = convert_test_data_if_needed(input_path, type_info['name'], model_output_dir)
    if converted_input_path != input_path:
        logger.info(f"使用转换后的测试数据: {converted_input_path}")
        input_path = converted_input_path
    
    # 加载数据
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    logger.info(f"加载样本数: {len(samples)}")
    
    # 创建预测器
    base_model_path = args.base_model
    predictor = Predictor(model_path, config, base_model_path, mode=args.mode)
    predictor.load_model()
    
    # 执行预测
    start_time = time.time()
    raw_results = predictor.predict_batch(samples, batch_size=args.batch_size)
    elapsed = time.time() - start_time
    
    # 处理结果，计算指标
    predictions = []
    labels = []
    predictions_by_ip = {}
    errors = []
    
    for i, (result, sample) in enumerate(zip(raw_results, samples)):
        # 提取预测标签
        pred_output = result['prediction']
        if isinstance(pred_output, dict) and not pred_output.get('_parse_error'):
            # JSON 格式输出（蒸馏模式）
            if type_info['name'] == 'vendor':
                predicted = pred_output.get('vendor', 'null')
            elif type_info['name'] == 'os':
                predicted = pred_output.get('os', 'null')
            else:
                predicted = pred_output.get('type', 'null')
            predicted = str(predicted) if predicted else 'null'
        else:
            # 简化格式输出（直接是标签）
            raw_resp = result.get('raw_response', '').strip()
            # 清理特殊字符
            predicted = re.sub(r'<\|.*?\|>', '', raw_resp).strip()
            if not predicted:
                predicted = 'null'
        
        # 提取真实标签
        messages = sample.get('messages', [])
        if len(messages) >= 3:
            raw_expected = messages[-1].get('content', 'null')
            # 尝试从 JSON 中提取标签（适用于所有模式）
            try:
                expected_json = json.loads(raw_expected)
                if type_info['name'] == 'vendor':
                    expected = expected_json.get('vendor', 'null')
                elif type_info['name'] == 'os':
                    expected = expected_json.get('os', 'null')
                else:
                    expected = expected_json.get('devicetype') or expected_json.get('type', 'null')
                expected = str(expected) if expected else 'null'
            except (json.JSONDecodeError, TypeError):
                # 如果不是 JSON，直接使用原始内容
                expected = raw_expected
        else:
            expected = 'null'
        
        # 标准化比较
        pred_lower = predicted.lower()
        exp_lower = expected.lower()
        is_correct = pred_lower == exp_lower
        
        predictions.append(pred_lower)
        labels.append(exp_lower)
        
        # 提取 IP（从用户输入中尝试提取）
        user_input = messages[1].get('content', '') if len(messages) >= 2 else ''
        ip_match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', user_input)
        ip = ip_match.group(1) if ip_match else f"sample_{i+1}"
        
        # 按 IP 整合
        predictions_by_ip[ip] = {
            "ip": ip,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "input_evidence": user_input[:500] if len(user_input) > 500 else user_input
        }
        
        # 收集错误
        if not is_correct:
            errors.append({
                "ip": ip,
                "expected": expected,
                "predicted": predicted,
                "input_evidence": user_input[:300] if len(user_input) > 300 else user_input
            })
    
    # 计算指标
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0
    
    try:
        from sklearn.metrics import f1_score, classification_report
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # 每个类别的指标
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        per_class = {k: {
            "precision": v['precision'],
            "recall": v['recall'],
            "f1": v['f1-score'],
            "support": int(v['support'])
        } for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
    except ImportError:
        logger.warning("sklearn 未安装，跳过 F1 计算")
        f1_macro = 0
        f1_weighted = 0
        per_class = {}
    
    # 错误分布统计
    error_distribution = Counter()
    for err in errors:
        key = f"{err['expected']}->{err['predicted']}"
        error_distribution[key] += 1
    
    # 构建输出结果
    output_data = {
        "metadata": {
            "model_path": model_path,
            "input_file": input_path,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(samples),
            "task_type": type_info['name'],
            "elapsed_seconds": round(elapsed, 2),
            "samples_per_second": round(len(samples) / elapsed, 2) if elapsed > 0 else 0
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "correct": correct,
            "total": len(labels)
        },
        "per_class_metrics": per_class,
        "predictions_by_ip": predictions_by_ip,
        "error_analysis": {
            "total_errors": len(errors),
            "error_distribution": dict(error_distribution.most_common(20)),
            "errors": errors[:100]  # 只保存前 100 个错误
        }
    }
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info("预测完成")
    logger.info("=" * 60)
    logger.info(f"总样本数: {len(samples)}")
    logger.info(f"准确率: {accuracy*100:.2f}% ({correct}/{len(labels)})")
    logger.info(f"F1 (Macro): {f1_macro:.4f}")
    logger.info(f"F1 (Weighted): {f1_weighted:.4f}")
    logger.info(f"错误数: {len(errors)}")
    logger.info(f"耗时: {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/s)")
    logger.info(f"输出文件: {output_path}")
    logger.info("=" * 60)
    
    # 打印错误分布 Top 5
    if error_distribution:
        logger.info("\n错误分布 (Top 5):")
        for pattern, count in error_distribution.most_common(5):
            logger.info(f"  {pattern}: {count}")


if __name__ == "__main__":
    main()
