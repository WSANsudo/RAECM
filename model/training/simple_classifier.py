"""
简化分类训练模块 - 方案A
只输出标签，不输出JSON

特点：
1. 输出极简：只输出标签名称（如 "Juniper", "MikroTik"）
2. 学习难度低：小模型（0.6B-3B）完全能胜任
3. 推理速度快：输出 token 数仅 1-3 个
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ============================================================
# 简化版 System Prompt
# ============================================================

SIMPLE_PROMPTS = {
    "vendor": """Identify the device vendor from network scan evidence.
Output ONLY the vendor name, nothing else.

Vendor list: MikroTik, Cisco, Juniper, Huawei, Fortinet, Synology, QNAP, Ubiquiti, HPE/Aruba, ZyXEL, D-Link, Netgear, TP-Link, ASUS, Nokia, Palo Alto Networks, Check Point, Arista, Extreme Networks, Ruckus, Allied Telesis, Yamaha, NEC, Keenetic, Lancom, SonicWall, Brocade, DD-WRT, Tomato, ZTE, Linksys, Enterasys, Maipu, Ruijie, IBM, Apple, Microsoft, Google

If unknown, output: null""",

    "os": """Identify the operating system from network scan evidence.
Output ONLY the OS name, nothing else.

OS list: RouterOS, IOS, IOS-XE, IOS-XR, NX-OS, JunOS, VRP, FortiOS, DSM, QTS, UniFi OS, ArubaOS, PAN-OS, Linux, Windows, FreeBSD, OpenBSD

If unknown, output: null""",

    "devicetype": """Identify the device type from network scan evidence.
Output ONLY the device type, nothing else.

Type list: router, switch, firewall, server, camera, nas, printer, iot, appliance

If unknown, output: null"""
}


@dataclass
class SimpleClassifierConfig:
    """简化分类器配置"""
    model_path: str = "./models/Qwen2.5-3B-Instruct"
    output_dir: str = "./output/simple"
    task_type: str = "vendor"  # vendor, os, devicetype
    
    # 训练参数
    num_epochs: int = 15  # 增加默认轮数
    batch_size: int = 16
    learning_rate: float = 3e-5  # 降低学习率，配合余弦退火
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # LoRA 参数
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 早停配置
    early_stopping: bool = True
    early_stopping_patience: int = 3
    min_epochs: int = 5  # 最小训练轮数，早停只在此之后生效
    
    # 生成参数
    max_new_tokens: int = 50  # 生成的最大token数
    
    # 评估配置
    eval_samples: int = 300  # 训练中评估的样本数
    eval_every_n_epochs: float = 0.2  # 每N轮评估一次（支持0.2表示每0.2轮评估）
    
    # 学习率调度器
    lr_scheduler_type: str = "cosine"  # linear, cosine, cosine_with_restarts
    
    # 精度
    bf16: bool = True
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


def convert_to_simple_format(
    input_path: str,
    output_path: str,
    task_type: str = "vendor"
) -> int:
    """
    将原始数据转换为简化格式
    
    原始格式:
    {
        "messages": [
            {"role": "system", "content": "...复杂提示词..."},
            {"role": "user", "content": "Network scan evidence:\n[ftp-21] ..."},
            {"role": "assistant", "content": "{\"ip\": \"...\", \"vendor\": \"Juniper\", ...}"}
        ]
    }
    
    简化格式:
    {
        "messages": [
            {"role": "system", "content": "简化提示词"},
            {"role": "user", "content": "[ftp-21] 220 ... FTP server ready\n[ssh-22] SSH-2.0-..."},
            {"role": "assistant", "content": "Juniper"}
        ]
    }
    """
    # 字段映射
    field_map = {
        "vendor": "vendor",
        "os": "os", 
        "devicetype": "devicetype"
    }
    field_name = field_map.get(task_type, "vendor")
    logger.info(f"转换任务类型: {task_type}, 字段名: {field_name}")
    
    system_prompt = SIMPLE_PROMPTS.get(task_type, SIMPLE_PROMPTS["vendor"])
    
    converted = []
    skipped = 0
    null_count = 0
    non_null_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                messages = item.get('messages', [])
                
                if len(messages) < 3:
                    skipped += 1
                    continue
                
                # 提取用户输入（去掉 "Network scan evidence:\n\n" 前缀）
                user_content = messages[1].get('content', '')
                if user_content.startswith('Network scan evidence:'):
                    user_content = user_content.replace('Network scan evidence:\n\n', '')
                    user_content = user_content.replace('Network scan evidence:\n', '')
                
                # 提取标签
                assistant_content = messages[2].get('content', '')
                try:
                    output = json.loads(assistant_content)
                    label = output.get(field_name)
                    if label is None:
                        label = "null"
                        null_count += 1
                        # 调试：打印前几个 null 样本
                        if null_count <= 3:
                            logger.warning(f"样本 {line_num} 标签为 null, output keys: {list(output.keys())}")
                    else:
                        non_null_count += 1
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                
                # 构建简化格式
                simple_item = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": str(label)}
                    ]
                }
                converted.append(simple_item)
                
            except Exception as e:
                skipped += 1
                continue
    
    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"转换完成: {len(converted)} 条, 跳过: {skipped} 条")
    logger.info(f"  - 非 null 标签: {non_null_count}")
    logger.info(f"  - null 标签: {null_count}")
    logger.info(f"输出文件: {output_path}")
    
    return len(converted)


class SimpleClassifierDataset(Dataset):
    """简化分类数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        task_type: str = "vendor"
    ):
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.data = self._load_data(data_path)
        
        # 统计标签分布
        self._log_label_stats()
    
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _log_label_stats(self):
        """统计标签分布"""
        from collections import Counter
        labels = []
        for item in self.data:
            messages = item.get('messages', [])
            if len(messages) >= 3:
                labels.append(messages[2].get('content', 'unknown'))
        
        counter = Counter(labels)
        logger.info(f"标签分布 (Top 10):")
        for label, count in counter.most_common(10):
            logger.info(f"  {label}: {count}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        messages = item['messages']
        
        # 分离 prompt 和 response
        prompt_messages = messages[:-1]
        label = messages[-1]['content']
        
        # 构建 prompt
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_text = self._build_prompt(prompt_messages)
        
        # 构建完整文本 - 只包含标签，不包含结束符
        # 方法：prompt + 标签（不使用 apply_chat_template 生成完整文本，避免自动添加结束符）
        full_text = prompt_text + label
        
        # Tokenize
        prompt_enc = self.tokenizer(prompt_text, truncation=False, padding=False, return_tensors='pt')
        prompt_length = prompt_enc['input_ids'].shape[1]
        
        full_enc = self.tokenizer(full_text, truncation=False, padding=False, return_tensors='pt')
        input_ids = full_enc['input_ids'].squeeze(0)
        attention_mask = full_enc['attention_mask'].squeeze(0)
        
        # 添加 EOS token 到末尾（让模型学会在标签后停止）
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
            attention_mask = torch.cat([attention_mask, torch.tensor([1])])
        
        # Labels: 只在 response 部分（标签 + EOS）计算 loss
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def _build_prompt(self, messages: List[Dict]) -> str:
        """构建 prompt（备用）"""
        parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return '\n'.join(parts)
    
    def _build_full_text(self, messages: List[Dict]) -> str:
        """构建完整文本（备用）"""
        parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return '\n'.join(parts)
    
    def get_collate_fn(self):
        """获取 collate 函数"""
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        def collate_fn(batch):
            input_ids = pad_sequence(
                [item['input_ids'] for item in batch],
                batch_first=True, padding_value=pad_token_id
            )
            attention_mask = pad_sequence(
                [item['attention_mask'] for item in batch],
                batch_first=True, padding_value=0
            )
            labels = pad_sequence(
                [item['labels'] for item in batch],
                batch_first=True, padding_value=-100
            )
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        
        return collate_fn


def clean_prediction(predicted_label: str) -> str:
    """
    清理模型预测输出，移除特殊标记和重复内容
    
    处理情况：
    - ios<|im_end|>ios<| -> ios
    - routerosiosios-xeios-rxjunosnetgear -> routeros (取第一个有效标签)
    - junosjunosjunosjunos -> junos
    """
    # 1. 移除所有特殊标记
    special_tokens = ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '<|startoftext|>', 
                      '<|end|>', '<|eot_id|>', '<|im', '<|end_of_text|>', '<|begin_of_text|>',
                      '<|start_header_id|>', '<|end_header_id|>', '<|']
    for token in special_tokens:
        predicted_label = predicted_label.replace(token, '')
    
    # 2. 取第一行
    predicted_label = predicted_label.split('\n')[0].strip()
    
    # 3. 如果为空，返回
    if not predicted_label:
        return predicted_label
    
    # 4. 检测并处理重复模式（如 iosiosios -> ios）
    # 尝试找到最短的重复单元
    for length in range(1, len(predicted_label) // 2 + 1):
        unit = predicted_label[:length]
        # 检查是否整个字符串都是这个单元的重复
        if predicted_label == unit * (len(predicted_label) // length) + unit[:len(predicted_label) % length]:
            # 如果是纯重复，返回单元
            if len(predicted_label) % length == 0:
                return unit
    
    # 5. 处理混合重复（如 routerosiosios-xe... -> 取第一个词）
    # 常见标签列表
    known_labels = [
        # OS
        'routeros', 'ios', 'ios-xe', 'ios-xr', 'nx-os', 'junos', 'vrp', 'fortios',
        'dsm', 'qts', 'unifi', 'arubaos', 'pan-os', 'linux', 'windows', 'freebsd', 'openbsd',
        # Vendor  
        'mikrotik', 'cisco', 'juniper', 'huawei', 'fortinet', 'synology', 'qnap', 'ubiquiti',
        'hpe', 'aruba', 'zyxel', 'd-link', 'netgear', 'tp-link', 'asus', 'nokia',
        'palo alto', 'check point', 'arista', 'extreme', 'ruckus', 'allied telesis',
        'yamaha', 'nec', 'keenetic', 'lancom', 'sonicwall', 'brocade', 'dd-wrt',
        # Device type
        'router', 'switch', 'firewall', 'server', 'camera', 'nas', 'printer', 'iot', 'appliance',
        'null'
    ]
    
    # 尝试从开头匹配已知标签
    predicted_lower = predicted_label.lower()
    for label in sorted(known_labels, key=len, reverse=True):  # 优先匹配长标签
        if predicted_lower.startswith(label):
            return predicted_label[:len(label)]
    
    # 6. 如果没有匹配到已知标签，返回原始清理结果
    return predicted_label


def evaluate_simple_classifier(
    model,
    tokenizer: PreTrainedTokenizer,
    eval_data_path: str,
    task_type: str = "vendor",
    device: torch.device = None
) -> Dict:
    """
    评估简化分类器
    
    Returns:
        {
            'accuracy': float,
            'total': int,
            'correct': int,
            'predictions': List[Dict]
        }
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 加载数据
    data = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    correct = 0
    total = 0
    predictions = []
    
    # 准备停止标记
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    special_tokens = ['<|im_end|>', '<|endoftext|>', '\n', '<|end|>', '<|eot_id|>']
    for token in special_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id and token_id not in stop_token_ids:
                stop_token_ids.append(token_id)
        except:
            pass
    
    if not stop_token_ids:
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else None
    
    # 常见厂商名列表（用于提取）
    vendors = ['Cisco', 'Juniper', 'MikroTik', 'Huawei', 'Fortinet', 'Ubiquiti', 
              'HPE', 'Aruba', 'ZyXEL', 'Zyxel', 'QNAP', 'Synology', 'Keenetic', 'Yamaha',
              'NEC Platforms', 'Linux', 'ASUS', 'Allied Telesis', 'H3C', 'DD-WRT',
              'Google', 'HPE/Aruba', 'Lancom', 'Ruijie', 'Rad Data']
    
    for item in data:
        messages = item['messages']
        prompt_messages = messages[:-1]
        expected_label = messages[-1]['content']
        
        # 构建 prompt
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_text = '\n'.join([
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                for m in prompt_messages
            ]) + "\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
                repetition_penalty=1.2,
            )
        
        # Decode
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # 使用统一的清理函数
        predicted_label = clean_prediction(raw_output)
        
        # 比较
        is_correct = predicted_label.lower() == expected_label.lower()
        if is_correct:
            correct += 1
        total += 1
        
        predictions.append({
            'expected': expected_label,
            'predicted': predicted_label,
            'correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'predictions': predictions
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='简化分类数据转换')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--task', type=str, default='vendor', choices=['vendor', 'os', 'devicetype'])
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    convert_to_simple_format(args.input, args.output, args.task)
