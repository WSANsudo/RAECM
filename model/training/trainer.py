"""
模型训练模块
Model training module for Network Device Analyzer Training System
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer
)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None

from .config import (
    ModelConfig, 
    TrainingConfig, 
    ModelRegistry,
    load_config,
    get_model_config,
    get_training_config
)
from .model_manager import ModelManager


class EarlyStopping:
    """早停机制，防止过拟合"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        Args:
            patience: 容忍多少次评估没有改善
            min_delta: 最小改善阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 当前验证集 loss
            
        Returns:
            是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            # 有改善，重置计数器
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # 没有改善
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkDeviceDataset(Dataset):
    """网络设备分析训练数据集"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.data[idx]
        messages = item['messages']
        
        # 分离 prompt 和 response
        prompt_messages = messages[:-1]  # system + user
        response_content = messages[-1]['content']  # assistant 回复
        
        # 构建 prompt 部分（不包含 assistant 回复，但包含开始标记）
        # 注意：Qwen3 需要禁用思考模式
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False  # 禁用思考模式
                )
            except TypeError:
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_messages, 
                        tokenize=False, 
                        add_generation_prompt=True
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
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False,
                    enable_thinking=False  # 禁用思考模式
                )
            except TypeError:
                try:
                    full_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                except Exception:
                    full_text = self._build_chat_text(messages)
            except Exception:
                full_text = self._build_chat_text(messages)
        else:
            full_text = self._build_chat_text(messages)
        
        # 分词 prompt 以获取其长度
        prompt_encodings = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encodings['input_ids'].shape[1]
        
        # 分词完整文本 - 不padding，不截断
        encodings = self.tokenizer(
            full_text,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # 创建 labels：只在 response 部分计算 loss
        labels = input_ids.clone()
        # prompt 部分设为 -100（不计算 loss）
        labels[:prompt_length] = -100
        # padding 部分也设为 -100（只处理 attention_mask 为 0 的位置）
        # 注意：不能直接用 pad_token_id 判断，因为可能和 eos_token_id 相同
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _build_prompt_text(self, messages: List[Dict]) -> str:
        """构建 prompt 文本（不包含 assistant 回复，但包含 assistant 开始标记）"""
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'phi' in model_name:
            text_parts = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|system|>\n{content}<|end|>")
                elif role == 'user':
                    text_parts.append(f"<|user|>\n{content}<|end|>")
            text_parts.append("<|assistant|>\n")
            return "\n".join(text_parts)
        elif 'llama' in model_name:
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
            # Qwen/默认格式
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
        """手动构建对话文本（备用方法，支持多种格式）"""
        text_parts = []
        
        # 检测模型类型
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()
        
        if 'phi' in model_name:
            # Phi 格式
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    text_parts.append(f"<|system|>\n{content}<|end|>")
                elif role == 'user':
                    text_parts.append(f"<|user|>\n{content}<|end|>")
                elif role == 'assistant':
                    text_parts.append(f"<|assistant|>\n{content}<|end|>")
            return "\n".join(text_parts)
        elif 'llama' in model_name:
            # Llama 3 格式
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
            # Qwen/默认格式
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


class NetworkDeviceTrainer:
    """网络设备分析模型训练器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig
    ):
        """
        初始化训练器
        
        Args:
            model_config: 模型配置
            training_config: 训练配置
        """
        self.model_config = model_config
        self.training_config = training_config
        
        self.device = self.setup_device()
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_device(self) -> torch.device:
        """
        设置训练设备（自动检测GPU/CPU）
        
        Returns:
            训练设备
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"检测到GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info("将使用GPU进行训练")
        else:
            device = torch.device('cpu')
            logger.info("未检测到GPU，将使用CPU进行训练")
            logger.warning("CPU训练速度较慢，建议使用GPU")
        
        return device
    
    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        加载模型和分词器
        
        Returns:
            (模型, 分词器)
        """
        model_path = self.model_config.model_name_or_path
        logger.info(f"正在加载模型: {model_path}")
        
        # 使用模型管理器确保模型可用（本地优先，否则自动下载）
        model_manager = ModelManager()
        model_path = model_manager.ensure_model(model_path)
        logger.info(f"解析后的模型路径: {model_path}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.model_config.trust_remote_code,
                padding_side='right'
            )
            
            # 确保有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            # 根据设备和配置选择加载方式
            if self.device.type == 'cuda':
                # 确定数据类型
                if self.training_config.bf16:
                    torch_dtype = torch.bfloat16
                elif self.training_config.fp16:
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=self.model_config.trust_remote_code,
                    torch_dtype=torch_dtype,
                    device_map='auto'
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=self.model_config.trust_remote_code,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
            
            logger.info(f"模型加载成功")
            logger.info(f"模型参数量: {self.model.num_parameters() / 1e6:.1f}M")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error("请检查:")
            logger.error(f"  1. 模型路径是否正确: {model_path}")
            logger.error("  2. 模型文件是否完整")
            logger.error("  3. 是否有足够的内存/显存")
            raise
    
    def setup_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        配置LoRA微调
        
        Args:
            model: 基础模型
            
        Returns:
            配置LoRA后的模型
        """
        if not self.model_config.use_lora:
            logger.info("使用全参数微调模式")
            return model
        
        if not PEFT_AVAILABLE:
            raise ImportError(
                "LoRA微调需要安装peft库。请运行: pip install peft>=0.7.0"
            )
        
        logger.info("配置LoRA微调...")
        
        # 获取模型类型对应的目标模块
        target_modules = self.model_config.lora_target_modules
        if target_modules is None:
            registry_config = ModelRegistry.get_model_config(
                self.model_config.model_name_or_path
            )
            target_modules = registry_config.get('lora_target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ])
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        
        # 启用输入梯度（gradient checkpointing需要）
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA配置完成")
        logger.info(f"可训练参数: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
        
        return model
    
    def prepare_dataset(
        self,
        train_path: str,
        valid_path: str
    ) -> Tuple[Dataset, Dataset]:
        """
        准备训练和验证数据集
        
        Args:
            train_path: 训练数据路径
            valid_path: 验证数据路径
            
        Returns:
            (训练数据集, 验证数据集)
        """
        logger.info("准备数据集...")
        
        train_dataset = NetworkDeviceDataset(
            train_path,
            self.tokenizer,
            self.model_config.max_length
        )
        
        valid_dataset = NetworkDeviceDataset(
            valid_path,
            self.tokenizer,
            self.model_config.max_length
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(valid_dataset)}")
        
        return train_dataset, valid_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.001
    ) -> None:
        """
        执行训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            early_stopping_patience: 早停耐心值
            early_stopping_min_delta: 早停最小改善阈值
        """
        logger.info("开始训练...")
        
        # 创建输出目录
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            fp16=self.training_config.fp16 and self.device.type == 'cuda',
            bf16=self.training_config.bf16 and self.device.type == 'cuda',
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to="none",
            remove_unused_columns=False,
            weight_decay=getattr(self.training_config, 'weight_decay', 0.01),
            max_grad_norm=getattr(self.training_config, 'max_grad_norm', 1.0),
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # 早停回调
        from transformers import EarlyStoppingCallback
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_min_delta
            )
        ]
        
        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # 断点续训
        resume_from = self.training_config.resume_from_checkpoint
        if resume_from and os.path.exists(resume_from):
            logger.info(f"从检查点恢复训练: {resume_from}")
        else:
            resume_from = None
        
        # 开始训练
        logger.info(f"早停配置: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        self.trainer.train(resume_from_checkpoint=resume_from)
        
        logger.info("训练完成!")
    
    def save_model(self, output_path: str) -> None:
        """
        保存训练好的模型
        
        Args:
            output_path: 输出路径
        """
        logger.info(f"保存模型到: {output_path}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # 保存模型
        if isinstance(self.model, PeftModel):
            # 保存LoRA权重
            self.model.save_pretrained(output_path)
        else:
            # 保存完整模型
            self.model.save_pretrained(output_path)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_path)
        
        # 保存配置
        config_dict = {
            'model_config': asdict(self.model_config),
            'training_config': asdict(self.training_config)
        }
        with open(os.path.join(output_path, 'training_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info("模型保存完成")


def train_model(
    config_path: str = "config.yaml",
    train_path: str = None,
    valid_path: str = None
) -> None:
    """
    训练模型的便捷函数
    
    Args:
        config_path: 配置文件路径
        train_path: 训练数据路径
        valid_path: 验证数据路径
    """
    # 加载配置
    config = load_config(config_path)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    
    # 默认数据路径
    data_config = config.get('data', {})
    train_path = train_path or os.path.join(data_config.get('output_dir', './data'), 'train.jsonl')
    valid_path = valid_path or os.path.join(data_config.get('output_dir', './data'), 'valid.jsonl')
    
    # 创建训练器
    trainer = NetworkDeviceTrainer(model_config, training_config)
    
    # 加载模型
    trainer.load_model_and_tokenizer()
    
    # 配置LoRA
    trainer.model = trainer.setup_lora(trainer.model)
    
    # 准备数据集
    train_dataset, valid_dataset = trainer.prepare_dataset(train_path, valid_path)
    
    # 训练
    trainer.train(train_dataset, valid_dataset)
    
    # 保存模型
    final_output = os.path.join(training_config.output_dir, 'final_model')
    trainer.save_model(final_output)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='模型训练工具')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--train', type=str, help='训练数据路径')
    parser.add_argument('--valid', type=str, help='验证数据路径')
    
    args = parser.parse_args()
    
    train_model(args.config, args.train, args.valid)
