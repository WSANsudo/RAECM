"""
蒸馏训练模块 - 增强版
Enhanced Distillation Training Module for Network Device Analyzer

实现的改进:
1. 渐进式蒸馏 (Progressive Distillation)
2. 特征级蒸馏 (Feature-level Distillation)
3. 动态样本权重与课程学习 (Curriculum Learning) - 基于Loss的难度评估
4. 检索一致性损失 (Retrieval Consistency Loss)
5. DPO偏好学习 (Direct Preference Optimization)
6. 多阶段训练 (Multi-stage Training): SFT → DPO
7. 早停策略 (Early Stopping)
8. 错误分析收集 (Error Analysis)
9. 显存监控 (Memory Monitoring)
"""

from .config import DistillationConfig
from .memory import MemoryMonitor
from .utils import EarlyStopping, ErrorAnalyzer
from .datasets import DistillationDataset, DPODataset, dynamic_collate_fn, dpo_collate_fn
from .losses import DistillationLoss, FeatureProjector
from .schedulers import CurriculumScheduler, ConfidenceCalibrator
from .trainer import DistillationTrainer, train_with_distillation

__all__ = [
    # 配置
    'DistillationConfig',
    
    # 显存监控
    'MemoryMonitor',
    
    # 工具类
    'EarlyStopping',
    'ErrorAnalyzer',
    
    # 数据集
    'DistillationDataset',
    'DPODataset',
    'dynamic_collate_fn',
    'dpo_collate_fn',
    
    # 损失函数
    'DistillationLoss',
    'FeatureProjector',
    
    # 调度器
    'CurriculumScheduler',
    'ConfidenceCalibrator',
    
    # 训练器
    'DistillationTrainer',
    'train_with_distillation',
]
