"""
调度器模块 - 课程学习和置信度校准
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """课程学习调度器 - 支持基于Loss的动态难度评估"""
    
    def __init__(
        self,
        start_epoch: int = 0,
        end_epoch: int = 2,
        initial_difficulty_threshold: float = 0.3,
        final_difficulty_threshold: float = 1.0,
        use_loss_based: bool = True
    ):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.initial_threshold = initial_difficulty_threshold
        self.final_threshold = final_difficulty_threshold
        self.use_loss_based = use_loss_based
        
        # 存储样本的历史Loss，用于动态难度评估
        self.sample_losses: Dict[int, List[float]] = defaultdict(list)
        self.loss_percentiles: Optional[Tuple[float, float]] = None
    
    def update_sample_loss(self, sample_idx: int, loss: float):
        """更新样本的Loss历史"""
        self.sample_losses[sample_idx].append(loss)
        # 只保留最近5个
        if len(self.sample_losses[sample_idx]) > 5:
            self.sample_losses[sample_idx] = self.sample_losses[sample_idx][-5:]
    
    def compute_loss_based_difficulty(self, sample_idx: int) -> float:
        """基于历史Loss计算样本难度"""
        if sample_idx not in self.sample_losses or not self.sample_losses[sample_idx]:
            return 0.5  # 默认中等难度
        
        avg_loss = sum(self.sample_losses[sample_idx]) / len(self.sample_losses[sample_idx])
        
        # 归一化到 0-1
        if self.loss_percentiles:
            low, high = self.loss_percentiles
            if high > low:
                difficulty = (avg_loss - low) / (high - low)
                return max(0.0, min(1.0, difficulty))
        
        return min(1.0, avg_loss / 2.0)  # 简单归一化
    
    def update_percentiles(self):
        """更新Loss百分位数"""
        all_losses = []
        for losses in self.sample_losses.values():
            if losses:
                all_losses.append(sum(losses) / len(losses))
        
        if len(all_losses) > 10:
            sorted_losses = sorted(all_losses)
            self.loss_percentiles = (
                sorted_losses[int(len(sorted_losses) * 0.1)],  # 10th percentile
                sorted_losses[int(len(sorted_losses) * 0.9)]   # 90th percentile
            )
        
    def get_sample_weight(
        self,
        difficulty: torch.Tensor,
        current_epoch: int,
        sample_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """根据当前epoch和样本难度计算权重"""
        # 如果使用基于Loss的难度，更新难度值
        if self.use_loss_based and sample_indices is not None:
            loss_difficulties = []
            for idx in sample_indices:
                loss_difficulties.append(self.compute_loss_based_difficulty(idx))
            difficulty = torch.tensor(loss_difficulties, device=difficulty.device, dtype=difficulty.dtype)
        
        if current_epoch < self.start_epoch:
            threshold = self.initial_threshold
        elif current_epoch >= self.end_epoch:
            return torch.ones_like(difficulty)
        else:
            progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            threshold = self.initial_threshold + progress * (self.final_threshold - self.initial_threshold)
        
        weight = torch.where(
            difficulty <= threshold,
            torch.ones_like(difficulty),
            torch.exp(-2 * (difficulty - threshold))
        )
        
        return weight


class ConfidenceCalibrator:
    """置信度校准器"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._calibrated = False
        
    def calibrate(
        self,
        model: nn.Module,
        val_dataloader: DataLoader,
        device: torch.device
    ) -> float:
        """在验证集上学习最优温度"""
        model.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels.cpu())
        
        # 简单的温度搜索
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            nll = self._compute_nll(all_logits, all_labels, temp)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
        self._calibrated = True
        logger.info(f"校准完成，最优温度: {best_temp:.2f}")
        
        return best_temp
    
    def _compute_nll(
        self,
        logits_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        temperature: float
    ) -> float:
        """计算给定温度下的负对数似然"""
        total_nll = 0.0
        total_count = 0
        
        for logits, labels in zip(logits_list, labels_list):
            scaled_logits = logits / temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            
            # Shift for causal LM
            shift_logprobs = log_probs[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            mask = (shift_labels != -100)
            
            token_logprobs = torch.gather(
                shift_logprobs,
                dim=-1,
                index=shift_labels.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            
            nll = -(token_logprobs * mask.float()).sum()
            total_nll += nll.item()
            total_count += mask.sum().item()
        
        return total_nll / max(total_count, 1)
    
    def get_calibrated_confidence(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """获取校准后的置信度"""
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return confidence
