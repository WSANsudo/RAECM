"""
蒸馏训练配置模块
"""

from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """蒸馏训练配置"""
    # 基础配置
    teacher_model_path: str = ""
    student_model_path: str = "./models/Meta-Llama-3-8B-Instruct"
    output_dir: str = "./output/distillation"
    
    # 蒸馏损失权重
    lambda_struct: float = 1.0          # 结构化输出损失权重
    lambda_cons: float = 0.3            # 一致性损失权重
    lambda_pref: float = 0.2            # 偏好学习损失权重
    lambda_feat: float = 0.1            # 特征蒸馏损失权重
    
    # DPO配置
    dpo_beta: float = 0.1               # DPO温度参数
    use_dpo: bool = True                # 启用DPO
    
    # 课程学习配置 - 基于Loss的动态难度
    use_curriculum: bool = True         # 启用课程学习
    curriculum_start_epoch: int = 0     # 课程学习开始epoch
    curriculum_end_epoch: int = 3       # 课程学习结束epoch
    use_loss_based_difficulty: bool = True  # 使用Loss作为难度指标
    
    # 一致性训练配置
    num_context_variants: int = 3       # 多变体增强
    context_dropout_rate: float = 0.15  # 上下文dropout率
    
    # 置信度校准
    calibration_temperature: float = 1.0
    confidence_threshold: float = 0.7
    
    # 多阶段训练配置
    multi_stage_training: bool = True   # 启用多阶段训练
    sft_epochs: int = 8                 # SFT阶段epoch数
    dpo_epochs: int = 2                 # DPO阶段epoch数
    
    # 早停配置
    early_stopping: bool = True         # 启用早停
    early_stopping_patience: int = 3    # 早停耐心值
    early_stopping_min_delta: float = 0.001  # 最小改进阈值
    
    # 错误分析
    collect_errors: bool = True         # 收集错误样本
    error_analysis_path: str = ""       # 错误分析输出路径
    
    # 训练配置
    max_length: int = None  # 不限制长度，保留完整序列（仅用于统计警告）
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5         # 降低学习率，更稳定
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1           # 权重衰减（正则化）
    max_grad_norm: float = 0.5          # 梯度裁剪阈值
    fp16: bool = False
    bf16: bool = True
    
    # LoRA 配置优化
    lora_rank: int = 32                 # 降低 rank
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # DPO 数据路径
    dpo_train_path: str = ""
    dpo_valid_path: str = ""
