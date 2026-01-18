"""
损失函数模块 - 蒸馏损失和特征投影
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistillationConfig


class FeatureProjector(nn.Module):
    """特征投影层，用于对齐teacher和student的隐藏表示"""
    
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.projector = nn.Linear(student_dim, teacher_dim)
        
    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        return self.projector(student_features)


class DistillationLoss(nn.Module):
    """复合蒸馏损失函数"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        # 添加 label smoothing 防止过拟合
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none', label_smoothing=0.1)
        
    def compute_struct_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor
    ) -> torch.Tensor:
        """计算加权结构化输出损失"""
        # [batch, seq, vocab] -> [batch * seq, vocab]
        batch_size, seq_len, vocab_size = student_logits.shape
        logits_flat = student_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # 计算每个token的损失
        token_loss = self.ce_loss(logits_flat, labels_flat)
        token_loss = token_loss.view(batch_size, seq_len)
        
        # 对每个样本求平均，然后应用样本权重
        sample_loss = token_loss.sum(dim=1) / (labels != -100).sum(dim=1).float().clamp(min=1)
        weighted_loss = (sample_loss * sample_weights).mean()
        
        return weighted_loss
    
    def compute_consistency_loss(
        self,
        outputs_list: List[torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算检索一致性损失 - 只在预测不一致时惩罚"""
        if len(outputs_list) < 2:
            return torch.tensor(0.0, device=outputs_list[0].device)
        
        total_kl = torch.tensor(0.0, device=outputs_list[0].device)
        count = 0
        
        for i in range(len(outputs_list)):
            for j in range(i + 1, len(outputs_list)):
                # 获取预测分布
                p_i = F.softmax(outputs_list[i], dim=-1)
                p_j = F.softmax(outputs_list[j], dim=-1)
                
                # 获取预测的token
                pred_i = outputs_list[i].argmax(dim=-1)
                pred_j = outputs_list[j].argmax(dim=-1)
                
                # 只在预测不一致的位置计算KL散度
                mask = (pred_i != pred_j) & (labels != -100)
                
                if mask.sum() > 0:
                    kl_div = F.kl_div(
                        F.log_softmax(outputs_list[i], dim=-1),
                        p_j,
                        reduction='none'
                    ).sum(dim=-1)
                    
                    masked_kl = (kl_div * mask.float()).sum() / mask.sum().float()
                    total_kl += masked_kl
                    count += 1
        
        return total_kl / max(count, 1)
    
    def compute_dpo_loss(
        self,
        model: nn.Module,
        preferred_ids: torch.Tensor,
        dispreferred_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """计算DPO偏好学习损失"""
        beta = self.config.dpo_beta
        
        # 计算当前模型的log概率
        with torch.no_grad() if ref_model is None else torch.enable_grad():
            preferred_logits = model(preferred_ids, attention_mask=attention_mask).logits
            dispreferred_logits = model(dispreferred_ids, attention_mask=attention_mask).logits
        
        # 计算序列log概率
        preferred_logprob = self._compute_sequence_logprob(preferred_logits, preferred_ids)
        dispreferred_logprob = self._compute_sequence_logprob(dispreferred_logits, dispreferred_ids)
        
        # 如果有参考模型，计算参考模型的log概率
        if ref_model is not None:
            with torch.no_grad():
                ref_preferred_logits = ref_model(preferred_ids, attention_mask=attention_mask).logits
                ref_dispreferred_logits = ref_model(dispreferred_ids, attention_mask=attention_mask).logits
            
            ref_preferred_logprob = self._compute_sequence_logprob(ref_preferred_logits, preferred_ids)
            ref_dispreferred_logprob = self._compute_sequence_logprob(ref_dispreferred_logits, dispreferred_ids)
            
            # DPO损失
            preferred_ratio = preferred_logprob - ref_preferred_logprob
            dispreferred_ratio = dispreferred_logprob - ref_dispreferred_logprob
        else:
            preferred_ratio = preferred_logprob
            dispreferred_ratio = dispreferred_logprob
        
        loss = -F.logsigmoid(beta * (preferred_ratio - dispreferred_ratio)).mean()
        return loss
    
    def _compute_sequence_logprob(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算序列的log概率"""
        log_probs = F.log_softmax(logits, dim=-1)
        # Shift for causal LM
        shift_logprobs = log_probs[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        # Gather log probs for actual tokens
        token_logprobs = torch.gather(
            shift_logprobs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        mask = (shift_labels != -100).float()
        sequence_logprob = (token_logprobs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return sequence_logprob
    
    def compute_feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        projector: FeatureProjector
    ) -> torch.Tensor:
        """计算特征级蒸馏损失"""
        projected_student = projector(student_features)
        return F.mse_loss(projected_student, teacher_features)
