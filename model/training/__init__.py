# Network Device Analyzer Training System
# 网络设备分析模型训练系统

__version__ = "1.0.0"

from .config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,
    DistillationConfig,
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    get_inference_config,
    get_distillation_config,
    ModelRegistry
)

from .data_processor import DataProcessor, SYSTEM_PROMPT
from .trainer import NetworkDeviceTrainer, NetworkDeviceDataset, train_model
from .evaluator import ModelEvaluator
from .inference import InferenceService, create_inference_service, interactive_inference
from .model_manager import ModelManager, SUPPORTED_MODELS, list_available_models

# 蒸馏训练模块 (已拆分为子模块)
from .distillation import (
    DistillationTrainer,
    DistillationDataset,
    DPODataset,
    DistillationLoss,
    CurriculumScheduler,
    ConfidenceCalibrator,
    FeatureProjector,
    train_with_distillation,
    DistillationConfig as DistillConfig,  # 避免与config中的重名
    MemoryMonitor,
    EarlyStopping,
    ErrorAnalyzer,
)

# 简化分类训练模块 (方案A)
from .simple_classifier import (
    SimpleClassifierConfig,
    SimpleClassifierDataset,
    convert_to_simple_format,
    evaluate_simple_classifier,
    SIMPLE_PROMPTS,
)

# 指标记录模块
from .metrics_recorder import MetricsRecorder, METRICS_DESCRIPTIONS

__all__ = [
    # 配置
    'ModelConfig',
    'TrainingConfig', 
    'DataConfig',
    'InferenceConfig',
    'DistillationConfig',
    'load_config',
    'get_model_config',
    'get_training_config',
    'get_data_config',
    'get_inference_config',
    'get_distillation_config',
    'ModelRegistry',
    
    # 数据处理
    'DataProcessor',
    'SYSTEM_PROMPT',
    
    # 训练
    'NetworkDeviceTrainer',
    'NetworkDeviceDataset',
    'train_model',
    
    # 评估
    'ModelEvaluator',
    
    # 推理
    'InferenceService',
    'create_inference_service',
    'interactive_inference',
    
    # 模型管理
    'ModelManager',
    'SUPPORTED_MODELS',
    'list_available_models',
    
    # 蒸馏
    'DistillationTrainer',
    'DistillationDataset',
    'DPODataset',
    'DistillationLoss',
    'CurriculumScheduler',
    'ConfidenceCalibrator',
    'FeatureProjector',
    'train_with_distillation',
    'MemoryMonitor',
    'EarlyStopping',
    'ErrorAnalyzer',
    
    # 简化分类 (方案A)
    'SimpleClassifierConfig',
    'SimpleClassifierDataset',
    'convert_to_simple_format',
    'evaluate_simple_classifier',
    'SIMPLE_PROMPTS',
    
    # 指标记录
    'MetricsRecorder',
    'METRICS_DESCRIPTIONS',
]
