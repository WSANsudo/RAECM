"""
配置加载模块
Configuration loading module for Network Device Analyzer Training System
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """模型配置类"""
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_type: str = "llama"
    use_lora: bool = True
    lora_rank: int = 64                # A800 80GB 最佳配置
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    max_length: Optional[int] = None  # 不限制长度
    trust_remote_code: bool = True


@dataclass
class TrainingConfig:
    """训练配置类"""
    output_dir: str = "./output"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8    # A800 80GB
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2    # 有效 batch = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = False
    bf16: bool = True                       # A800 使用 bf16
    gradient_checkpointing: bool = False    # A800 显存充足，关闭以加速
    resume_from_checkpoint: Optional[str] = None


@dataclass
class DataConfig:
    """数据配置类"""
    input_path: str = "./input/vendor_model_train.jsonl"
    output_dir: str = "./data"
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    augment_ratio: float = 0.3  # 数据增强比例


@dataclass
class InferenceConfig:
    """推理配置类"""
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 512
    do_sample: bool = False


@dataclass
class DistillationConfig:
    """蒸馏训练配置类 - A800 80GB 最佳配置"""
    # 损失权重 - 全部启用
    lambda_struct: float = 1.0
    lambda_cons: float = 0.3
    lambda_pref: float = 0.2
    lambda_feat: float = 0.1
    
    # DPO配置 - 启用
    use_dpo: bool = True
    dpo_beta: float = 0.1
    
    # 课程学习配置 - 启用
    use_curriculum: bool = True
    curriculum_start_epoch: int = 0
    curriculum_end_epoch: int = 3
    use_loss_based_difficulty: bool = True  # 使用Loss作为难度指标
    
    # 一致性训练配置 - 完整配置
    num_context_variants: int = 3
    context_dropout_rate: float = 0.15
    
    # 置信度校准
    calibration_temperature: float = 1.0
    confidence_threshold: float = 0.7
    
    # 多阶段训练配置
    multi_stage_training: bool = True
    sft_epochs: int = 3
    dpo_epochs: int = 2
    
    # 早停配置
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    
    # 错误分析
    collect_errors: bool = True


class ModelRegistry:
    """模型注册表，管理支持的模型配置"""
    
    SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
        "qwen2.5-0.5b": {
            "model_type": "qwen2",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen"
        },
        "qwen2.5-1.5b": {
            "model_type": "qwen2",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen"
        },
        "qwen2.5-3b": {
            "model_type": "qwen2",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen"
        },
        "qwen3-0.6b": {
            "model_type": "qwen3",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen3"
        },
        "qwen3-1.7b": {
            "model_type": "qwen3",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen3"
        },
        "qwen3-4b": {
            "model_type": "qwen3",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen3"
        },
        "qwen3-8b": {
            "model_type": "qwen3",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen3"
        },
        "qwen3-14b": {
            "model_type": "qwen3",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen3"
        },
        "qwen3-30b-a3b": {
            "model_type": "qwen3_moe",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "qwen3"
        },
        "llama3": {
            "model_type": "llama",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "llama3"
        },
        "mistral": {
            "model_type": "mistral",
            "lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "chat_template": "mistral"
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """
        获取模型配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置字典
            
        Raises:
            ValueError: 如果模型不支持
        """
        model_key = model_name.lower()
        for key in cls.SUPPORTED_MODELS:
            if key in model_key:
                return cls.SUPPORTED_MODELS[key].copy()
        
        # 默认返回qwen2配置
        return cls.SUPPORTED_MODELS["qwen2.5-0.5b"].copy()
    
    @classmethod
    def register_model(cls, model_name: str, config: Dict[str, Any]) -> None:
        """
        注册新模型
        
        Args:
            model_name: 模型名称
            config: 模型配置字典
        """
        cls.SUPPORTED_MODELS[model_name.lower()] = config
    
    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有支持的模型"""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """检查模型是否支持"""
        model_key = model_name.lower()
        return any(key in model_key for key in cls.SUPPORTED_MODELS)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果YAML解析失败
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_config(config: Dict[str, Any]) -> ModelConfig:
    """
    从配置字典创建ModelConfig对象
    
    Args:
        config: 配置字典
        
    Returns:
        ModelConfig对象
    """
    model_cfg = config.get('model', {})
    
    # 获取模型注册表中的默认配置
    model_name = model_cfg.get('name_or_path', '')
    registry_config = ModelRegistry.get_model_config(model_name)
    
    # 默认的LoRA目标模块
    default_lora_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    return ModelConfig(
        model_name_or_path=model_cfg.get('name_or_path', './models/Qwen2.5-0.5B-Instruct'),
        model_type=model_cfg.get('type', registry_config.get('model_type', 'qwen2')),
        use_lora=model_cfg.get('use_lora', True),
        lora_rank=model_cfg.get('lora_rank', 8),
        lora_alpha=model_cfg.get('lora_alpha', 16),
        lora_dropout=model_cfg.get('lora_dropout', 0.05),
        lora_target_modules=model_cfg.get(
            'lora_target_modules', 
            registry_config.get('lora_target_modules', default_lora_modules)
        ),
        max_length=model_cfg.get('max_length', None),  # 默认不限制长度
        trust_remote_code=model_cfg.get('trust_remote_code', True)
    )


def get_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """
    从配置字典创建TrainingConfig对象
    
    Args:
        config: 配置字典
        
    Returns:
        TrainingConfig对象
    """
    train_cfg = config.get('training', {})
    
    return TrainingConfig(
        output_dir=train_cfg.get('output_dir', './output'),
        num_train_epochs=train_cfg.get('num_train_epochs', 3),
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=train_cfg.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 4),
        learning_rate=train_cfg.get('learning_rate', 2e-4),
        warmup_ratio=train_cfg.get('warmup_ratio', 0.1),
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'cosine'),
        logging_steps=train_cfg.get('logging_steps', 10),
        save_steps=train_cfg.get('save_steps', 500),
        eval_steps=train_cfg.get('eval_steps', 500),
        fp16=train_cfg.get('fp16', True),
        bf16=train_cfg.get('bf16', False),
        gradient_checkpointing=train_cfg.get('gradient_checkpointing', True),
        resume_from_checkpoint=train_cfg.get('resume_from_checkpoint', None)
    )


def get_data_config(config: Dict[str, Any]) -> DataConfig:
    """
    从配置字典创建DataConfig对象
    
    Args:
        config: 配置字典
        
    Returns:
        DataConfig对象
    """
    data_cfg = config.get('data', {})
    
    return DataConfig(
        input_path=data_cfg.get('input_path', './input/vendor_model_train.jsonl'),
        output_dir=data_cfg.get('output_dir', './data'),
        train_ratio=data_cfg.get('train_ratio', 0.8),
        valid_ratio=data_cfg.get('valid_ratio', 0.1),
        test_ratio=data_cfg.get('test_ratio', 0.1),
        augment_ratio=data_cfg.get('augment_ratio', 0.3)
    )


def get_inference_config(config: Dict[str, Any]) -> InferenceConfig:
    """
    从配置字典创建InferenceConfig对象
    
    Args:
        config: 配置字典
        
    Returns:
        InferenceConfig对象
    """
    infer_cfg = config.get('inference', {})
    
    return InferenceConfig(
        temperature=infer_cfg.get('temperature', 0.1),
        top_p=infer_cfg.get('top_p', 0.9),
        max_new_tokens=infer_cfg.get('max_new_tokens', 512),
        do_sample=infer_cfg.get('do_sample', False)
    )


def get_distillation_config(config: Dict[str, Any]) -> DistillationConfig:
    """
    从配置字典创建DistillationConfig对象
    
    Args:
        config: 配置字典
        
    Returns:
        DistillationConfig对象
    """
    distill_cfg = config.get('distillation', {})
    
    return DistillationConfig(
        lambda_struct=distill_cfg.get('lambda_struct', 1.0),
        lambda_cons=distill_cfg.get('lambda_cons', 0.3),
        lambda_pref=distill_cfg.get('lambda_pref', 0.2),
        lambda_feat=distill_cfg.get('lambda_feat', 0.1),
        use_dpo=distill_cfg.get('use_dpo', True),
        dpo_beta=distill_cfg.get('dpo_beta', 0.1),
        use_curriculum=distill_cfg.get('use_curriculum', True),
        curriculum_start_epoch=distill_cfg.get('curriculum_start_epoch', 0),
        curriculum_end_epoch=distill_cfg.get('curriculum_end_epoch', 3),
        use_loss_based_difficulty=distill_cfg.get('use_loss_based_difficulty', True),
        num_context_variants=distill_cfg.get('num_context_variants', 3),
        context_dropout_rate=distill_cfg.get('context_dropout_rate', 0.15),
        calibration_temperature=distill_cfg.get('calibration_temperature', 1.0),
        confidence_threshold=distill_cfg.get('confidence_threshold', 0.7),
        multi_stage_training=distill_cfg.get('multi_stage_training', True),
        sft_epochs=distill_cfg.get('sft_epochs', 3),
        dpo_epochs=distill_cfg.get('dpo_epochs', 2),
        early_stopping=distill_cfg.get('early_stopping', True),
        early_stopping_patience=distill_cfg.get('early_stopping_patience', 3),
        early_stopping_min_delta=distill_cfg.get('early_stopping_min_delta', 0.001),
        collect_errors=distill_cfg.get('collect_errors', True)
    )


if __name__ == "__main__":
    # 测试配置加载
    try:
        config = load_config("config.yaml")
        print("配置加载成功!")
        
        model_config = get_model_config(config)
        print(f"模型配置: {model_config}")
        
        training_config = get_training_config(config)
        print(f"训练配置: {training_config}")
        
        data_config = get_data_config(config)
        print(f"数据配置: {data_config}")
        
        inference_config = get_inference_config(config)
        print(f"推理配置: {inference_config}")
        
        print(f"\n支持的模型: {ModelRegistry.list_models()}")
    except Exception as e:
        print(f"错误: {e}")
