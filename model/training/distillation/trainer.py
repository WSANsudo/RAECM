"""
蒸馏训练器模块
"""

import os
import json
import logging
import time
import platform
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None

from ..config import ModelConfig
from .config import DistillationConfig
from .memory import MemoryMonitor
from .utils import EarlyStopping, ErrorAnalyzer
from .datasets import DistillationDataset
from .losses import DistillationLoss
from .schedulers import CurriculumScheduler, ConfidenceCalibrator

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """增强版蒸馏训练器 - 支持多阶段训练、早停和错误分析"""
    
    def __init__(
        self,
        config: DistillationConfig,
        model_config: Optional[ModelConfig] = None
    ):
        self.config = config
        self.model_config = model_config or ModelConfig()
        
        self.device = self._setup_device()
        self.student_model = None
        self.teacher_model = None
        self.tokenizer = None
        self.ref_model = None  # DPO参考模型
        
        # 显存监控器
        self.memory_monitor = MemoryMonitor(self.device)
        
        self.loss_fn = DistillationLoss(config)
        self.curriculum = CurriculumScheduler(
            start_epoch=config.curriculum_start_epoch,
            end_epoch=config.curriculum_end_epoch,
            use_loss_based=config.use_loss_based_difficulty
        ) if config.use_curriculum else None
        self.calibrator = ConfidenceCalibrator(config.calibration_temperature)
        
        # 早停和错误分析
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        ) if config.early_stopping else None
        
        self.error_analyzer = ErrorAnalyzer(
            output_path=config.error_analysis_path or os.path.join(config.output_dir, 'error_analysis.jsonl')
        ) if config.collect_errors else None
        
        self.feature_projector = None
        self.current_epoch = 0
        self.current_stage = 'sft'  # 'sft' or 'dpo'
    
    def _is_local_path(self, path: str) -> bool:
        """检查是否是本地路径（必须存在）"""
        return os.path.exists(path)
        
    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"使用GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU训练")
        return device
    
    def load_models(self) -> None:
        """加载student模型和tokenizer"""
        logger.info(f"加载Student模型: {self.config.student_model_path}")
        
        is_local = self._is_local_path(self.config.student_model_path) or os.path.exists(self.config.student_model_path)
        logger.info(f"模型路径类型: {'本地' if is_local else '远程'}")
        
        self.memory_monitor.checkpoint("before_load")
        self.memory_monitor.log_memory_status("模型加载前 - ")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_path,
            trust_remote_code=True,
            padding_side='right',
            local_files_only=is_local
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 确定数据类型
        if self.config.bf16 and self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("使用 bfloat16 精度")
        elif self.config.fp16 and self.device.type == 'cuda':
            dtype = torch.float16
            logger.info("使用 float16 精度")
        else:
            dtype = torch.float32
            logger.info("使用 float32 精度")
        
        # 检查显存大小
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU显存: {gpu_mem:.1f}GB ({gpu_name})")
        else:
            gpu_mem = 0
            gpu_name = "CPU"
        
        # 根据显存大小选择加载策略
        if gpu_mem < 8:
            logger.info("检测到小显存GPU，启用显存优化模式")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto',
                low_cpu_mem_usage=True,
                max_memory={0: f"{int(gpu_mem * 0.85)}GB"},
                local_files_only=is_local
            )
        elif gpu_mem >= 48:
            logger.info("检测到大显存GPU，启用高性能模式")
            logger.info("注意力实现: sdpa (PyTorch 内置)")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto',
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                local_files_only=is_local
            )
        else:
            logger.info("检测到中等显存GPU")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto',
                low_cpu_mem_usage=True,
                local_files_only=is_local
            )
        
        # 启用gradient checkpointing节省显存
        if hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()
            logger.info("已启用 Gradient Checkpointing")
        
        if self.device.type == 'cpu':
            self.student_model = self.student_model.to(self.device)
        
        self.memory_monitor.checkpoint("after_model_load")
        self.memory_monitor.log_memory_status("模型加载后 - ")
        self.memory_monitor.log_model_memory(self.student_model, "Student模型")
        
        logger.info(f"Student模型参数量: {self.student_model.num_parameters() / 1e6:.1f}M")
        
        # 如果使用DPO，保存参考模型的初始状态
        if self.config.use_dpo:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_mem < 8:
                    logger.warning("小显存模式: 跳过DPO参考模型，使用简化DPO")
                    self.ref_model = None
                else:
                    self._create_reference_model()
            else:
                self._create_reference_model()
    
    def _create_reference_model(self) -> None:
        """创建DPO参考模型（冻结的初始模型副本）"""
        logger.info("创建DPO参考模型...")
        
        is_local = self._is_local_path(self.config.student_model_path) or os.path.exists(self.config.student_model_path)
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            gpu_mem = 0
        
        if gpu_mem >= 48:
            logger.info("大显存模式: 参考模型加载到 GPU")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto',
                low_cpu_mem_usage=True,
                local_files_only=is_local
            )
        else:
            logger.info("参考模型加载到 CPU (按需移动到 GPU)")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='cpu',
                low_cpu_mem_usage=True,
                local_files_only=is_local
            )
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.ref_model.eval()
        logger.info("DPO参考模型已加载")
    
    def setup_lora(self) -> None:
        """配置LoRA"""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT不可用，使用全参数微调")
            return
        
        self.memory_monitor.checkpoint("before_lora")
        
        # 调试：打印 target_modules
        target_modules = self.model_config.lora_target_modules
        logger.info(f"LoRA target_modules: {target_modules}")
        
        if not target_modules or len(target_modules) < 3:
            logger.warning(f"target_modules 数量不足 ({len(target_modules) if target_modules else 0})，使用默认值")
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        self.student_model = get_peft_model(self.student_model, lora_config)
        
        if hasattr(self.student_model, 'enable_input_require_grads'):
            self.student_model.enable_input_require_grads()
        
        trainable = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student_model.parameters())
        trainable_size = sum(p.element_size() * p.numel() for p in self.student_model.parameters() if p.requires_grad) / 1024**2
        
        logger.info(f"LoRA配置完成:")
        logger.info(f"  可训练参数: {trainable/1e6:.2f}M ({100*trainable/total:.2f}%)")
        logger.info(f"  可训练参数显存: {trainable_size:.2f} MB")
        
        self.memory_monitor.checkpoint("after_lora")
        self.memory_monitor.log_memory_status("LoRA配置后 - ")
    
    def prepare_datasets(
        self,
        train_path: str,
        valid_path: str
    ) -> Tuple[Dataset, Dataset]:
        """准备数据集"""
        train_dataset = DistillationDataset(
            train_path,
            self.tokenizer,
            max_length=self.config.max_length,
            num_context_variants=self.config.num_context_variants,
            context_dropout_rate=self.config.context_dropout_rate
        )
        
        valid_dataset = DistillationDataset(
            valid_path,
            self.tokenizer,
            max_length=self.config.max_length,
            num_context_variants=1,
            context_dropout_rate=0.0
        )
        
        logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}")
        return train_dataset, valid_dataset
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算复合蒸馏损失"""
        input_ids = batch['input_ids']
        labels = batch['labels']
        difficulty = batch['difficulty']
        
        # 计算样本权重（课程学习）
        if self.curriculum is not None:
            sample_weights = self.curriculum.get_sample_weight(difficulty, self.current_epoch)
        else:
            sample_weights = torch.ones_like(difficulty)
        
        sample_weights = sample_weights.to(self.device)
        
        # 结构化输出损失
        struct_loss = self.loss_fn.compute_struct_loss(
            outputs.logits, labels, sample_weights
        )
        
        total_loss = self.config.lambda_struct * struct_loss
        loss_dict = {'struct_loss': struct_loss.item()}
        
        # 一致性损失（如果有多个上下文变体）
        if hasattr(batch, 'variant_outputs') and len(batch.variant_outputs) > 1:
            cons_loss = self.loss_fn.compute_consistency_loss(
                batch.variant_outputs, labels
            )
            total_loss += self.config.lambda_cons * cons_loss
            loss_dict['cons_loss'] = cons_loss.item()
        
        return total_loss, loss_dict

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        sample_indices: Optional[list] = None
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.student_model.train()
        
        total_loss = 0.0
        loss_components = defaultdict(float)
        num_batches = 0
        
        epoch_start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if TQDM_AVAILABLE:
            pbar = tqdm(
                train_dataloader, 
                desc=f"Epoch {self.current_epoch + 1} ({self.current_stage.upper()})",
                ncols=120,
                leave=True
            )
        else:
            pbar = train_dataloader
            logger.info(f"提示: 安装tqdm可获得更好的进度显示 (pip install tqdm)")
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            difficulty = batch['difficulty'].to(self.device)
            
            start_idx = batch_idx * train_dataloader.batch_size
            end_idx = min(start_idx + train_dataloader.batch_size, len(train_dataloader.dataset))
            current_indices = list(range(start_idx, end_idx))
            
            try:
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("=" * 70)
                    logger.error("OOM错误! 显存不足")
                    logger.error("=" * 70)
                    self.memory_monitor.log_memory_status("OOM时 - ")
                    logger.error(f"当前Batch信息:")
                    logger.error(f"  - input_ids shape: {input_ids.shape}")
                    logger.error(f"  - batch_idx: {batch_idx}")
                    self.memory_monitor.log_tensor_memory("input_ids", input_ids)
                    self.memory_monitor.log_tensor_memory("attention_mask", attention_mask)
                    self.memory_monitor.log_tensor_memory("labels", labels)
                    logger.error("建议: 减小batch_size或max_length，或启用gradient_checkpointing")
                    self.memory_monitor.clear_cache()
                raise
            
            batch_data = {
                'input_ids': input_ids,
                'labels': labels,
                'difficulty': difficulty,
                'sample_indices': current_indices
            }
            loss, loss_dict = self.compute_loss(batch_data, outputs)
            
            if self.curriculum and self.config.use_loss_based_difficulty:
                per_sample_loss = loss_dict.get('per_sample_loss', [])
                for i, idx in enumerate(current_indices):
                    if i < len(per_sample_loss):
                        self.curriculum.update_sample_loss(idx, per_sample_loss[i])
            
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            for k, v in loss_dict.items():
                if k != 'per_sample_loss':
                    loss_components[k] += v
            num_batches += 1
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    clear_interval = 4 if gpu_mem < 24 else 16
                    if (batch_idx + 1) % (self.config.gradient_accumulation_steps * clear_interval) == 0:
                        torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.config.learning_rate
            mem_str = self.memory_monitor.get_memory_str()
            
            if TQDM_AVAILABLE:
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'mem': mem_str
                })
            else:
                if (batch_idx + 1) % 10 == 0:
                    progress = (batch_idx + 1) / len(train_dataloader) * 100
                    logger.info(
                        f"[{progress:5.1f}%] Batch {batch_idx+1}/{len(train_dataloader)} | "
                        f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                        f"Mem: {mem_str}"
                    )
        
        if self.curriculum and self.config.use_loss_based_difficulty:
            self.curriculum.update_percentiles()
        
        epoch_time = time.time() - epoch_start_time
        samples_per_sec = len(train_dataloader.dataset) / epoch_time
        mem_stats = self.memory_monitor.get_memory_stats()
        
        logger.info(f"Epoch {self.current_epoch + 1} 完成:")
        logger.info(f"  - 总耗时: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
        logger.info(f"  - 吞吐量: {samples_per_sec:.1f} samples/s")
        logger.info(f"  - 平均Loss: {total_loss / num_batches:.4f}")
        logger.info(f"  - 显存峰值: {mem_stats['peak']:.2f}GB / {mem_stats['total']:.1f}GB")
        
        metrics = {'loss': total_loss / num_batches}
        for k, v in loss_components.items():
            metrics[k] = v / num_batches
        
        return metrics
    
    def evaluate(
        self,
        eval_dataloader: DataLoader
    ) -> Dict[str, float]:
        """评估模型"""
        self.student_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if TQDM_AVAILABLE:
            pbar = tqdm(eval_dataloader, desc="评估中", ncols=100, leave=False)
        else:
            pbar = eval_dataloader
        
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                        outputs = self.student_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
                del outputs, input_ids, attention_mask, labels
                
                if TQDM_AVAILABLE:
                    mem_str = self.memory_monitor.get_memory_str()
                    pbar.set_postfix({'eval_loss': f'{total_loss / num_batches:.4f}', 'mem': mem_str})
        
        self.memory_monitor.clear_cache()
        
        return {'eval_loss': total_loss / num_batches}

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        dpo_train_dataset: Optional[Dataset] = None,
        dpo_eval_dataset: Optional[Dataset] = None
    ) -> None:
        """
        执行完整训练 - 支持多阶段训练
        
        阶段1: SFT (监督微调)
        阶段2: DPO (偏好学习) - 如果提供了DPO数据
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Windows兼容
        if platform.system() == 'Windows':
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 2 if self.device.type == 'cuda' else 0
            pin_memory = self.device.type == 'cuda'
        
        # 获取动态 collate 函数（支持变长序列）
        train_collate_fn = train_dataset.get_collate_fn() if hasattr(train_dataset, 'get_collate_fn') else None
        eval_collate_fn = eval_dataset.get_collate_fn() if hasattr(eval_dataset, 'get_collate_fn') else None
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=train_collate_fn  # 动态填充
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=eval_collate_fn  # 动态填充
        )
        
        if self.config.multi_stage_training and dpo_train_dataset is not None:
            sft_epochs = self.config.sft_epochs
            dpo_epochs = self.config.dpo_epochs
            total_epochs = sft_epochs + dpo_epochs
            logger.info(f"多阶段训练: SFT({sft_epochs} epochs) → DPO({dpo_epochs} epochs)")
        else:
            sft_epochs = self.config.num_train_epochs
            dpo_epochs = 0
            total_epochs = sft_epochs
        
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_dataloader) * total_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.memory_monitor.checkpoint("after_optimizer")
        self.memory_monitor.log_optimizer_memory(optimizer, "AdamW优化器")
        
        logger.info("=" * 70)
        logger.info("蒸馏训练开始")
        logger.info("=" * 70)
        logger.info(f"训练配置:")
        logger.info(f"  - 训练集大小: {len(train_dataset)}")
        logger.info(f"  - 验证集大小: {len(eval_dataset)}")
        logger.info(f"  - Batch Size: {self.config.per_device_train_batch_size}")
        logger.info(f"  - 有效Batch Size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  - 总Epochs: {total_epochs} (SFT: {sft_epochs}, DPO: {dpo_epochs})")
        logger.info(f"  - 学习率: {self.config.learning_rate}")
        logger.info(f"  - 调度器: warmup + linear decay")
        logger.info(f"  - 课程学习: {'启用 (基于Loss)' if self.config.use_curriculum else '禁用'}")
        logger.info(f"  - 早停: {'启用 (patience=' + str(self.config.early_stopping_patience) + ')' if self.config.early_stopping else '禁用'}")
        
        logger.info("-" * 70)
        logger.info("显存占用分析:")
        self.memory_monitor.log_memory_status("  训练开始前 - ")
        
        # 显存估算（max_length 可能为 None，使用默认值）
        est_max_length = self.config.max_length or 2048
        batch_mem_estimate = (
            self.config.per_device_train_batch_size * 
            est_max_length * 4 * 2
        ) / 1024**3
        logger.info(f"  预估单batch输入: {batch_mem_estimate*1000:.1f} MB (基于 max_length={est_max_length})")
        logger.info("=" * 70)
        
        best_eval_loss = float('inf')
        training_start_time = time.time()
        global_epoch = 0
        
        # ========== 阶段1: SFT ==========
        self.current_stage = 'sft'
        logger.info("")
        logger.info("=" * 70)
        logger.info("阶段1: SFT (监督微调)")
        logger.info("=" * 70)
        
        if self.early_stopping:
            self.early_stopping.reset()
        
        for epoch in range(sft_epochs):
            self.current_epoch = global_epoch
            global_epoch += 1
            
            logger.info("")
            logger.info(f"[SFT] Epoch {epoch + 1}/{sft_epochs}")
            
            if self.curriculum is not None:
                if self.config.use_loss_based_difficulty:
                    logger.info(f"  课程学习: 基于Loss的动态难度")
                else:
                    difficulty_threshold = self._get_difficulty_threshold(epoch)
                    logger.info(f"  课程学习难度阈值: {difficulty_threshold:.2f}")
            
            train_metrics = self.train_epoch(train_dataloader, optimizer, scheduler)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("正在评估...")
            eval_metrics = self.evaluate(eval_dataloader)
            
            self._log_epoch_summary(epoch + 1, train_metrics, eval_metrics, training_start_time, sft_epochs)
            
            if eval_metrics['eval_loss'] < best_eval_loss:
                improvement = best_eval_loss - eval_metrics['eval_loss']
                best_eval_loss = eval_metrics['eval_loss']
                self.save_model(os.path.join(self.config.output_dir, 'best_model_sft'))
                logger.info(f"  ✓ 新的最佳SFT模型! (提升: {improvement:.4f})")
            
            if self.early_stopping:
                if self.early_stopping(eval_metrics['eval_loss']):
                    logger.info(f"  ⚠ 早停触发! 验证Loss连续{self.config.early_stopping_patience}个epoch未改善")
                    break
        
        self.save_model(os.path.join(self.config.output_dir, 'sft_final'))
        logger.info(f"SFT阶段完成，模型已保存")
        
        # ========== 阶段2: DPO ==========
        if dpo_epochs > 0 and dpo_train_dataset is not None:
            self.current_stage = 'dpo'
            logger.info("")
            logger.info("=" * 70)
            logger.info("阶段2: DPO (偏好学习)")
            logger.info("=" * 70)
            
            # 获取 DPO 数据集的 collate 函数
            dpo_collate_fn = dpo_train_dataset.get_collate_fn() if hasattr(dpo_train_dataset, 'get_collate_fn') else None
            
            dpo_train_loader = DataLoader(
                dpo_train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=dpo_collate_fn  # 动态填充
            )
            
            if self.early_stopping:
                self.early_stopping.reset()
            
            dpo_lr = self.config.learning_rate * 0.5
            dpo_optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=dpo_lr
            )
            
            dpo_steps = len(dpo_train_loader) * dpo_epochs // self.config.gradient_accumulation_steps
            dpo_scheduler = get_linear_schedule_with_warmup(
                dpo_optimizer,
                num_warmup_steps=int(dpo_steps * 0.1),
                num_training_steps=dpo_steps
            )
            
            for epoch in range(dpo_epochs):
                self.current_epoch = global_epoch
                global_epoch += 1
                
                logger.info("")
                logger.info(f"[DPO] Epoch {epoch + 1}/{dpo_epochs}")
                
                train_metrics = self.train_dpo_epoch(dpo_train_loader, dpo_optimizer, dpo_scheduler)
                eval_metrics = self.evaluate(eval_dataloader)
                
                self._log_epoch_summary(epoch + 1, train_metrics, eval_metrics, training_start_time, dpo_epochs)
                
                if eval_metrics['eval_loss'] < best_eval_loss:
                    improvement = best_eval_loss - eval_metrics['eval_loss']
                    best_eval_loss = eval_metrics['eval_loss']
                    self.save_model(os.path.join(self.config.output_dir, 'best_model_dpo'))
                    logger.info(f"  ✓ 新的最佳DPO模型! (提升: {improvement:.4f})")
                
                if self.early_stopping and self.early_stopping(eval_metrics['eval_loss']):
                    logger.info(f"  ⚠ DPO早停触发!")
                    break
        
        if self.error_analyzer:
            self.error_analyzer.save()
            summary = self.error_analyzer.get_summary()
            logger.info(f"错误分析: {summary}")
        
        final_path = os.path.join(self.config.output_dir, 'final_model')
        self.save_model(final_path)
        
        total_time = time.time() - training_start_time
        mem_stats = self.memory_monitor.get_memory_stats()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("训练完成!")
        logger.info("=" * 70)
        logger.info(f"总训练时间: {total_time/60:.1f}min ({total_time/3600:.2f}h)")
        logger.info(f"最佳验证Loss: {best_eval_loss:.4f}")
        logger.info(f"显存峰值: {mem_stats['peak']:.2f}GB / {mem_stats['total']:.1f}GB ({100*mem_stats['peak']/mem_stats['total']:.1f}%)")
        logger.info(f"最终模型: {final_path}")
        logger.info("=" * 70)
    
    def _get_difficulty_threshold(self, epoch: int) -> float:
        """获取当前epoch的难度阈值"""
        if epoch < self.curriculum.start_epoch:
            return self.curriculum.initial_threshold
        elif epoch >= self.curriculum.end_epoch:
            return self.curriculum.final_threshold
        else:
            progress = (epoch - self.curriculum.start_epoch) / (self.curriculum.end_epoch - self.curriculum.start_epoch)
            return self.curriculum.initial_threshold + progress * (self.curriculum.final_threshold - self.curriculum.initial_threshold)
    
    def _log_epoch_summary(
        self, 
        epoch: int, 
        train_metrics: Dict, 
        eval_metrics: Dict, 
        start_time: float,
        total_epochs: int
    ):
        """打印epoch总结"""
        logger.info("-" * 50)
        logger.info(f"Epoch {epoch} 总结:")
        logger.info(f"  训练Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  验证Loss: {eval_metrics['eval_loss']:.4f}")
        
        elapsed = time.time() - start_time
        avg_time = elapsed / epoch
        remaining = avg_time * (total_epochs - epoch)
        logger.info(f"  已用时间: {elapsed/60:.1f}min | 预计剩余: {remaining/60:.1f}min")

    def train_dpo_epoch(
        self,
        dpo_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> Dict[str, float]:
        """DPO训练一个epoch"""
        self.student_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        if TQDM_AVAILABLE:
            pbar = tqdm(dpo_dataloader, desc=f"DPO Epoch", ncols=120)
        else:
            pbar = dpo_dataloader
        
        for batch_idx, batch in enumerate(pbar):
            chosen_ids = batch['chosen_input_ids'].to(self.device)
            chosen_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_ids = batch['rejected_input_ids'].to(self.device)
            rejected_mask = batch['rejected_attention_mask'].to(self.device)
            
            loss = self._compute_dpo_loss(
                chosen_ids, chosen_mask,
                rejected_ids, rejected_mask
            )
            
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if (batch_idx + 1) % (self.config.gradient_accumulation_steps * 4) == 0:
                    self.memory_monitor.clear_cache()
            
            if TQDM_AVAILABLE:
                mem_str = self.memory_monitor.get_memory_str()
                pbar.set_postfix({'dpo_loss': f'{total_loss/num_batches:.4f}', 'mem': mem_str})
        
        return {'loss': total_loss / num_batches, 'dpo_loss': total_loss / num_batches}
    
    def _compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算DPO损失"""
        beta = self.config.dpo_beta
        
        chosen_outputs = self.student_model(chosen_ids, attention_mask=chosen_mask)
        chosen_logprob = self._get_sequence_logprob(chosen_outputs.logits, chosen_ids)
        
        rejected_outputs = self.student_model(rejected_ids, attention_mask=rejected_mask)
        rejected_logprob = self._get_sequence_logprob(rejected_outputs.logits, rejected_ids)
        
        loss = -F.logsigmoid(beta * (chosen_logprob - rejected_logprob)).mean()
        
        return loss
    
    def _get_sequence_logprob(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算序列的log概率"""
        log_probs = F.log_softmax(logits, dim=-1)
        shift_logprobs = log_probs[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        token_logprobs = torch.gather(
            shift_logprobs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        sequence_logprob = (token_logprobs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return sequence_logprob
    
    def save_model(self, output_path: str) -> None:
        """保存模型"""
        os.makedirs(output_path, exist_ok=True)
        
        if PEFT_AVAILABLE and isinstance(self.student_model, PeftModel):
            self.student_model.save_pretrained(output_path)
        else:
            self.student_model.save_pretrained(output_path)
        
        self.tokenizer.save_pretrained(output_path)
        
        config_dict = {
            'distillation_config': {
                'lambda_struct': self.config.lambda_struct,
                'lambda_cons': self.config.lambda_cons,
                'lambda_pref': self.config.lambda_pref,
                'dpo_beta': self.config.dpo_beta,
                'calibration_temperature': self.calibrator.temperature,
                'confidence_threshold': self.config.confidence_threshold
            }
        }
        
        with open(os.path.join(output_path, 'distillation_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)


def train_with_distillation(
    config_path: str = "config.yaml",
    train_path: str = None,
    valid_path: str = None,
    distillation_config: Optional[DistillationConfig] = None
) -> None:
    """蒸馏训练便捷函数"""
    from ..config import load_config, get_model_config
    
    config = load_config(config_path)
    model_config = get_model_config(config)
    
    if distillation_config is None:
        distillation_config = DistillationConfig(
            student_model_path=model_config.model_name_or_path,
            max_length=model_config.max_length
        )
    
    data_config = config.get('data', {})
    train_path = train_path or os.path.join(data_config.get('output_dir', './data'), 'train.jsonl')
    valid_path = valid_path or os.path.join(data_config.get('output_dir', './data'), 'valid.jsonl')
    
    trainer = DistillationTrainer(distillation_config, model_config)
    
    trainer.load_models()
    
    if model_config.use_lora:
        trainer.setup_lora()
    
    train_dataset, valid_dataset = trainer.prepare_datasets(train_path, valid_path)
    
    trainer.train(train_dataset, valid_dataset)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='蒸馏训练工具')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--train', type=str, help='训练数据路径')
    parser.add_argument('--valid', type=str, help='验证数据路径')
    parser.add_argument('--lambda-struct', type=float, default=1.0, help='结构化损失权重')
    parser.add_argument('--lambda-cons', type=float, default=0.3, help='一致性损失权重')
    parser.add_argument('--lambda-pref', type=float, default=0.2, help='偏好学习损失权重')
    parser.add_argument('--use-dpo', action='store_true', help='使用DPO')
    parser.add_argument('--use-curriculum', action='store_true', help='使用课程学习')
    
    args = parser.parse_args()
    
    distill_config = DistillationConfig(
        lambda_struct=args.lambda_struct,
        lambda_cons=args.lambda_cons,
        lambda_pref=args.lambda_pref,
        use_dpo=args.use_dpo,
        use_curriculum=args.use_curriculum
    )
    
    train_with_distillation(
        args.config,
        args.train,
        args.valid,
        distill_config
    )
