"""
数据集模块 - 蒸馏训练和DPO数据集
支持变长序列，不截断不填充
"""

import json
import logging
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def dynamic_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    动态 collate 函数，支持变长序列的批处理
    在 batch 内动态填充到最长序列长度，而非固定 max_length
    
    Args:
        batch: 数据样本列表
        pad_token_id: 填充 token ID
    
    Returns:
        批处理后的字典，包含填充后的张量
    """
    # 提取各字段
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    difficulty_list = [item['difficulty'] for item in batch]
    
    # 动态填充到 batch 内最长序列
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    # difficulty 直接堆叠
    difficulty = torch.stack(difficulty_list)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'difficulty': difficulty,
    }


def dpo_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    DPO 数据集的动态 collate 函数
    
    Args:
        batch: 数据样本列表
        pad_token_id: 填充 token ID
    
    Returns:
        批处理后的字典
    """
    chosen_input_ids_list = [item['chosen_input_ids'] for item in batch]
    chosen_attention_mask_list = [item['chosen_attention_mask'] for item in batch]
    rejected_input_ids_list = [item['rejected_input_ids'] for item in batch]
    rejected_attention_mask_list = [item['rejected_attention_mask'] for item in batch]
    prompt_length_list = [item['prompt_length'] for item in batch]
    
    # 动态填充
    chosen_input_ids = pad_sequence(chosen_input_ids_list, batch_first=True, padding_value=pad_token_id)
    chosen_attention_mask = pad_sequence(chosen_attention_mask_list, batch_first=True, padding_value=0)
    rejected_input_ids = pad_sequence(rejected_input_ids_list, batch_first=True, padding_value=pad_token_id)
    rejected_attention_mask = pad_sequence(rejected_attention_mask_list, batch_first=True, padding_value=0)
    prompt_length = torch.stack(prompt_length_list)
    
    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'rejected_input_ids': rejected_input_ids,
        'rejected_attention_mask': rejected_attention_mask,
        'prompt_length': prompt_length,
    }


class DistillationDataset(Dataset):
    """
    蒸馏训练数据集，支持多上下文变体
    不截断、不固定填充，保留完整序列
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = None,  # 仅用于警告，不截断
        num_context_variants: int = 3,
        context_dropout_rate: float = 0.1
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length  # 仅用于统计和警告
        self.num_context_variants = num_context_variants
        self.context_dropout_rate = context_dropout_rate
        self.data = self._load_data(data_path)
        
        # 统计序列长度
        self._log_length_stats()
        
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item['difficulty'] = self._compute_difficulty(item)
                    data.append(item)
        return data
    
    def _log_length_stats(self):
        """统计并记录序列长度分布"""
        if len(self.data) == 0:
            return
        
        # 采样统计（避免全量计算太慢）
        sample_size = min(100, len(self.data))
        sample_indices = torch.randperm(len(self.data))[:sample_size].tolist()
        
        lengths = []
        for idx in sample_indices:
            item = self.data[idx]
            messages = item['messages']
            
            # 构建文本
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                try:
                    full_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                        enable_thinking=False
                    )
                except (TypeError, Exception):
                    full_text = self._build_chat_text(messages)
            else:
                full_text = self._build_chat_text(messages)
            
            # 计算长度
            tokens = self.tokenizer(full_text, add_special_tokens=False)
            lengths.append(len(tokens['input_ids']))
        
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            max_len = max(lengths)
            min_len = min(lengths)
            logger.info(f"序列长度统计 (采样 {sample_size} 条):")
            logger.info(f"  平均: {avg_len:.0f}, 最小: {min_len}, 最大: {max_len}")
            
            if self.max_length and max_len > self.max_length:
                over_count = sum(1 for l in lengths if l > self.max_length)
                logger.warning(f"  ⚠ {over_count}/{sample_size} 条样本超过 max_length={self.max_length}")
    
    def _compute_difficulty(self, item: Dict) -> float:
        """计算样本难度分数 (0-1, 越高越难)"""
        messages = item.get('messages', [])
        difficulty = 0.0
        
        for msg in messages:
            if msg.get('role') == 'assistant':
                try:
                    output = json.loads(msg['content'])
                    null_count = sum(1 for v in output.values() if v is None)
                    difficulty += null_count * 0.1
                except:
                    difficulty += 0.3
            elif msg.get('role') == 'user':
                content = msg.get('content', '')
                evidence_count = content.count('权重')
                if evidence_count < 3:
                    difficulty += 0.2
                elif evidence_count > 6:
                    difficulty -= 0.1
        
        return max(0.0, min(1.0, difficulty))
    
    def _apply_context_dropout(self, text: str) -> str:
        """对上下文应用dropout，模拟检索变体"""
        if torch.rand(1).item() > self.context_dropout_rate:
            return text
        
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            if '权重' in line and torch.rand(1).item() < self.context_dropout_rate:
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        messages = item['messages']
        difficulty = item.get('difficulty', 0.5)
        
        # 分离 prompt 和 response
        prompt_messages = messages[:-1]  # system + user
        
        # 构建 prompt 部分
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_text = self._build_prompt_text(prompt_messages)
            except Exception:
                prompt_text = self._build_prompt_text(prompt_messages)
        else:
            prompt_text = self._build_prompt_text(prompt_messages)
        
        # 构建完整文本
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False
                )
            except TypeError:
                try:
                    full_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    full_text = self._build_chat_text(messages)
            except Exception:
                full_text = self._build_chat_text(messages)
        else:
            full_text = self._build_chat_text(messages)
        
        # 分词 prompt（不截断）
        prompt_encodings = self.tokenizer(
            prompt_text,
            truncation=False,  # 不截断
            padding=False,     # 不填充
            return_tensors='pt'
        )
        prompt_length = prompt_encodings['input_ids'].shape[1]
        
        # 分词完整文本（不截断、不填充）
        encodings = self.tokenizer(
            full_text,
            truncation=False,  # 不截断
            padding=False,     # 不填充
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = encodings['attention_mask'].squeeze(0)  # [seq_len]
        
        # 创建 labels：只在 response 部分计算 loss
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # prompt 部分不计算 loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'difficulty': torch.tensor(difficulty, dtype=torch.float),
        }
    
    def _build_prompt_text(self, messages: List[Dict]) -> str:
        """构建 prompt 文本"""
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'llama' in model_name:
            text_parts = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == 'user':
                    text_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            text_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "<|begin_of_text|>" + "".join(text_parts)
        else:
            text_parts = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == 'user':
                    text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            text_parts.append("<|im_start|>assistant\n")
            return '\n'.join(text_parts)
    
    def _build_chat_text(self, messages: List[Dict]) -> str:
        """手动构建对话文本"""
        text_parts = []
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'llama' in model_name:
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == 'user':
                    text_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == 'assistant':
                    text_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
            return "<|begin_of_text|>" + "".join(text_parts)
        else:
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == 'user':
                    text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == 'assistant':
                    text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            return '\n'.join(text_parts)
    
    def get_collate_fn(self):
        """获取配套的 collate 函数"""
        pad_token_id = self.tokenizer.pad_token_id or 0
        return lambda batch: dynamic_collate_fn(batch, pad_token_id)


class DPODataset(Dataset):
    """
    DPO 偏好学习数据集
    不截断、不固定填充
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = None  # 仅用于警告，不截断
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        prompt_messages = item.get('prompt', [])
        chosen_text = item.get('chosen', '')
        rejected_text = item.get('rejected', '')
        
        prompt_text = self._build_chat_text(prompt_messages)
        
        chosen_full = prompt_text + self._format_assistant(chosen_text)
        rejected_full = prompt_text + self._format_assistant(rejected_text)
        
        # 分词（不截断、不填充）
        chosen_encodings = self.tokenizer(
            chosen_full,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        
        rejected_encodings = self.tokenizer(
            rejected_full,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        
        prompt_encodings = self.tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        prompt_length = prompt_encodings['input_ids'].shape[1]
        
        return {
            'chosen_input_ids': chosen_encodings['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encodings['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encodings['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encodings['attention_mask'].squeeze(0),
            'prompt_length': torch.tensor(prompt_length, dtype=torch.long)
        }
    
    def _build_chat_text(self, messages: List[Dict]) -> str:
        """构建对话文本"""
        text_parts = []
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'llama' in model_name:
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == 'user':
                    text_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            return "<|begin_of_text|>" + "".join(text_parts)
        else:
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == 'user':
                    text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            return '\n'.join(text_parts)
    
    def _format_assistant(self, content: str) -> str:
        """格式化 assistant 回复"""
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'llama' in model_name:
            return f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        else:
            return f"\n<|im_start|>assistant\n{content}<|im_end|>"
    
    def get_collate_fn(self):
        """获取配套的 collate 函数"""
        pad_token_id = self.tokenizer.pad_token_id or 0
        return lambda batch: dpo_collate_fn(batch, pad_token_id)
