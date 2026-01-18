"""
模型预设配置
Model Presets for quick switching between different models
支持 RTX 4090 (24GB) 和 A800 (80GB) 配置
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelPreset:
    """模型预设配置"""
    name: str                          # 预设名称
    model_path: str                    # 模型路径或 HuggingFace ID
    model_type: str                    # 模型类型 (qwen, llama, phi, etc.)
    
    # LoRA 配置
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    
    # 训练配置
    max_length: int = 2048
    learning_rate: float = 2e-5
    batch_size: int = 2
    gradient_accumulation: int = 8
    
    # 显存估算 (GB) - 基于 24GB 显卡
    vram_estimate: float = 16.0
    
    # Chat template 格式
    chat_format: str = "chatml"        # chatml, llama, phi, alpaca
    
    # 备注
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_vram(
    model_size_b: float,
    lora_rank: int,
    batch_size: int,
    seq_length: int,
    num_lora_modules: int = 7,
    use_gradient_checkpointing: bool = True,
    dtype: str = "bf16"
) -> Dict[str, float]:
    """
    估算训练显存需求
    
    Args:
        model_size_b: 模型参数量 (十亿)
        lora_rank: LoRA rank
        batch_size: 批次大小
        seq_length: 序列长度
        num_lora_modules: LoRA 模块数量
        use_gradient_checkpointing: 是否使用梯度检查点
        dtype: 数据类型 (bf16/fp16/fp32)
        
    Returns:
        显存估算字典
    """
    # 每参数字节数
    bytes_per_param = 2 if dtype in ["bf16", "fp16"] else 4
    
    # 模型权重显存
    model_vram = model_size_b * bytes_per_param  # GB
    
    # LoRA 参数显存 (估算)
    # 每个模块: 2 * hidden_dim * rank * 2 (A和B矩阵)
    hidden_dim = int(model_size_b * 500)  # 粗略估算
    lora_params = num_lora_modules * 2 * hidden_dim * lora_rank * 2
    lora_vram = lora_params * bytes_per_param / 1e9
    
    # 优化器状态 (AdamW: 2x 参数量)
    optimizer_vram = lora_vram * 2
    
    # 梯度显存
    gradient_vram = lora_vram
    
    # 激活值显存 (粗略估算)
    # 与 batch_size * seq_length * hidden_dim 成正比
    activation_base = model_size_b * 0.5  # 基础激活值
    activation_vram = activation_base * batch_size * (seq_length / 2048)
    
    if use_gradient_checkpointing:
        activation_vram *= 0.3  # 梯度检查点减少约70%
    
    total_vram = model_vram + lora_vram + optimizer_vram + gradient_vram + activation_vram
    
    return {
        "model": round(model_vram, 1),
        "lora": round(lora_vram, 2),
        "optimizer": round(optimizer_vram, 2),
        "gradient": round(gradient_vram, 2),
        "activation": round(activation_vram, 1),
        "total": round(total_vram, 1),
        "recommended_vram": round(total_vram * 1.15, 0)  # 15% 安全余量
    }


def get_optimal_config(vram_gb: float, model_size_b: float) -> Dict[str, Any]:
    """
    根据显存大小获取最优配置
    
    Args:
        vram_gb: 可用显存 (GB)
        model_size_b: 模型大小 (十亿参数)
        
    Returns:
        最优配置字典
    """
    if vram_gb >= 80:
        # A800/A100 80GB - 最大性能
        return {
            "lora_rank": 64,
            "batch_size": 8,
            "max_length": 2048,
            "gradient_checkpointing": False,
            "num_lora_modules": 7,
            "description": "A800/A100 80GB 最大性能配置"
        }
    elif vram_gb >= 48:
        # A40/A6000 48GB
        return {
            "lora_rank": 32,
            "batch_size": 4,
            "max_length": 2048,
            "gradient_checkpointing": False,
            "num_lora_modules": 7,
            "description": "48GB 高性能配置"
        }
    elif vram_gb >= 24:
        # RTX 4090/3090 24GB
        return {
            "lora_rank": 16,
            "batch_size": 1,
            "max_length": 1024,
            "gradient_checkpointing": True,
            "num_lora_modules": 4,  # 只训练注意力层
            "description": "RTX 4090 24GB 优化配置"
        }
    else:
        # RTX 3080/4080 16GB
        return {
            "lora_rank": 8,
            "batch_size": 1,
            "max_length": 512,
            "gradient_checkpointing": True,
            "num_lora_modules": 4,
            "description": "16GB 显存受限配置"
        }


# ============================================================
# 预设配置注册表 - 基于 24GB 显卡 (RTX 4090)
# ============================================================

PRESETS: Dict[str, ModelPreset] = {
    # ==================== Qwen 系列 ====================
    "qwen2.5-0.5b": ModelPreset(
        name="Qwen2.5-0.5B-Instruct",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        model_type="qwen",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=5e-5,
        batch_size=8,
        gradient_accumulation=2,
        vram_estimate=4.0,
        chat_format="chatml",
        description="最小模型，适合测试"
    ),
    
    "qwen2.5-1.5b": ModelPreset(
        name="Qwen2.5-1.5B-Instruct",
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        model_type="qwen",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=3e-5,
        batch_size=4,
        gradient_accumulation=4,
        vram_estimate=6.0,
        chat_format="chatml",
        description="小型高效模型"
    ),
    
    "qwen2.5-3b": ModelPreset(
        name="Qwen2.5-3B-Instruct",
        model_path="Qwen/Qwen2.5-3B-Instruct",
        model_type="qwen",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation=4,
        vram_estimate=10.0,
        chat_format="chatml",
        description="平衡性能与效率"
    ),
    
    "qwen2.5-7b": ModelPreset(
        name="Qwen2.5-7B-Instruct",
        model_path="Qwen/Qwen2.5-7B-Instruct",
        model_type="qwen",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=1e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=18.0,
        chat_format="chatml",
        description="高性能 Qwen 模型"
    ),
    
    "qwen3-4b": ModelPreset(
        name="Qwen3-4B",
        model_path="Qwen/Qwen3-4B",
        model_type="qwen3",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation=4,
        vram_estimate=12.0,
        chat_format="chatml",
        description="Qwen3 4B - 24GB 显存优化"
    ),
    
    "qwen3-8b": ModelPreset(
        name="Qwen3-8B",
        model_path="Qwen/Qwen3-8B",
        model_type="qwen3",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=1e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=20.0,
        chat_format="chatml",
        description="Qwen3 8B - 推理能力强"
    ),
    
    # ==================== Llama 系列 ====================
    "llama3-8b": ModelPreset(
        name="Llama-3-8B-Instruct",
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        model_type="llama",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=20.0,
        chat_format="llama",
        description="Meta Llama 3 8B (24GB)"
    ),
    
    "llama3-8b-a800": ModelPreset(
        name="Llama-3-8B-Instruct-A800",
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        model_type="llama",
        lora_rank=64,
        lora_alpha=128,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=8,
        gradient_accumulation=2,
        vram_estimate=60.0,
        chat_format="llama",
        description="Meta Llama 3 8B (A800 80GB 最佳配置)"
    ),
    
    "llama3.1-8b": ModelPreset(
        name="Llama-3.1-8B-Instruct",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        model_type="llama",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=20.0,
        chat_format="llama",
        description="Llama 3.1 扩展上下文"
    ),
    
    "llama3.2-3b": ModelPreset(
        name="Llama-3.2-3B-Instruct",
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        model_type="llama",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation=4,
        vram_estimate=10.0,
        chat_format="llama",
        description="紧凑型 Llama 3.2"
    ),
    
    # ==================== Phi 系列 ====================
    "phi-3-mini": ModelPreset(
        name="Phi-3-mini-4k-instruct",
        model_path="microsoft/Phi-3-mini-4k-instruct",
        model_type="phi",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation=4,
        vram_estimate=10.0,
        chat_format="phi",
        description="Microsoft Phi-3 mini (3.8B)"
    ),
    
    "phi-3.5-mini": ModelPreset(
        name="Phi-3.5-mini-instruct",
        model_path="microsoft/Phi-3.5-mini-instruct",
        model_type="phi",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation=4,
        vram_estimate=10.0,
        chat_format="phi",
        description="Phi-3.5 mini 性能提升版"
    ),
    
    # ==================== Gemma 系列 ====================
    "gemma2-2b": ModelPreset(
        name="Gemma-2-2B-it",
        model_path="google/gemma-2-2b-it",
        model_type="gemma",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation=4,
        vram_estimate=8.0,
        chat_format="gemma",
        description="Google Gemma 2 2B"
    ),
    
    "gemma2-9b": ModelPreset(
        name="Gemma-2-9B-it",
        model_path="google/gemma-2-9b-it",
        model_type="gemma",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=1e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=22.0,
        chat_format="gemma",
        description="Google Gemma 2 9B"
    ),
    
    # ==================== Mistral 系列 ====================
    "mistral-7b": ModelPreset(
        name="Mistral-7B-Instruct-v0.3",
        model_path="mistralai/Mistral-7B-Instruct-v0.3",
        model_type="mistral",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=1e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=18.0,
        chat_format="mistral",
        description="Mistral 7B v0.3"
    ),
    
    # ==================== DeepSeek 系列 ====================
    "deepseek-r1-qwen-7b": ModelPreset(
        name="DeepSeek-R1-Distill-Qwen-7B",
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        model_type="qwen",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=1e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=18.0,
        chat_format="chatml",
        description="DeepSeek R1 蒸馏版，推理能力强"
    ),
    
    # ==================== Yi 系列 ====================
    "yi-1.5-6b": ModelPreset(
        name="Yi-1.5-6B-Chat",
        model_path="01-ai/Yi-1.5-6B-Chat",
        model_type="yi",
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=2e-5,
        batch_size=2,
        gradient_accumulation=8,
        vram_estimate=14.0,
        chat_format="chatml",
        description="Yi 1.5 6B 中文支持好"
    ),
    
    "yi-1.5-9b": ModelPreset(
        name="Yi-1.5-9B-Chat",
        model_path="01-ai/Yi-1.5-9B-Chat",
        model_type="yi",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_length=2048,
        learning_rate=1e-5,
        batch_size=1,
        gradient_accumulation=16,
        vram_estimate=22.0,
        chat_format="chatml",
        description="Yi 1.5 9B"
    ),
}


def get_preset(name: str) -> Optional[ModelPreset]:
    """获取预设配置"""
    return PRESETS.get(name.lower())


def list_presets(vram_limit: float = None) -> None:
    """列出所有预设"""
    print("\n" + "=" * 80)
    print("可用模型预设 (Available Model Presets)")
    print("=" * 80)
    
    # 按显存需求分组
    groups = {
        "小型 (< 8GB)": [],
        "中型 (8-16GB)": [],
        "大型 (16-24GB)": [],
        "超大 (> 24GB, 需要 A800/A100)": []
    }
    
    for key, preset in PRESETS.items():
        if preset.vram_estimate < 8:
            groups["小型 (< 8GB)"].append((key, preset))
        elif preset.vram_estimate < 16:
            groups["中型 (8-16GB)"].append((key, preset))
        elif preset.vram_estimate <= 24:
            groups["大型 (16-24GB)"].append((key, preset))
        else:
            groups["超大 (> 24GB, 需要 A800/A100)"].append((key, preset))
    
    for group_name, presets in groups.items():
        if not presets:
            continue
        
        print(f"\n{group_name}")
        print("-" * 60)
        
        for key, preset in presets:
            if vram_limit and preset.vram_estimate > vram_limit:
                status = "❌"
            else:
                status = "✓"
            
            print(f"  {status} {key:20} ~{preset.vram_estimate:>4.0f}GB  {preset.description}")
    
    print("\n" + "=" * 80)
    print("使用方法: python main.py train --preset llama3-8b")
    print("A800 用户: python main.py train --preset llama3-8b-a800")
    print("=" * 80 + "\n")


def apply_preset_to_config(preset: ModelPreset, config: dict) -> dict:
    """将预设应用到配置字典"""
    if 'model' not in config:
        config['model'] = {}
    if 'training' not in config:
        config['training'] = {}
    
    config['model']['name_or_path'] = preset.model_path
    config['model']['type'] = preset.model_type
    config['model']['lora_rank'] = preset.lora_rank
    config['model']['lora_alpha'] = preset.lora_alpha
    config['model']['lora_target_modules'] = list(preset.lora_target_modules)
    config['model']['max_length'] = preset.max_length
    
    config['training']['learning_rate'] = preset.learning_rate
    config['training']['per_device_train_batch_size'] = preset.batch_size
    config['training']['gradient_accumulation_steps'] = preset.gradient_accumulation
    
    # 根据显存估算设置 gradient_checkpointing
    if preset.vram_estimate > 40:
        config['training']['gradient_checkpointing'] = False
    else:
        config['training']['gradient_checkpointing'] = True
    
    return config


def print_vram_estimate(
    model_name: str = "llama3-8b",
    vram_gb: float = 80,
    custom_config: Dict = None
) -> None:
    """打印显存估算"""
    preset = get_preset(model_name)
    if not preset:
        print(f"未找到预设: {model_name}")
        return
    
    # 获取最优配置
    optimal = get_optimal_config(vram_gb, 8.0)
    
    config = custom_config or {
        "lora_rank": optimal["lora_rank"],
        "batch_size": optimal["batch_size"],
        "max_length": 2048,
        "num_lora_modules": optimal["num_lora_modules"],
        "gradient_checkpointing": optimal["gradient_checkpointing"]
    }
    
    estimate = estimate_vram(
        model_size_b=8.0,
        lora_rank=config["lora_rank"],
        batch_size=config["batch_size"],
        seq_length=config["max_length"],
        num_lora_modules=config["num_lora_modules"],
        use_gradient_checkpointing=config["gradient_checkpointing"]
    )
    
    print(f"\n{'='*50}")
    print(f"显存估算: {model_name} on {vram_gb}GB GPU")
    print(f"{'='*50}")
    print(f"配置: {optimal['description']}")
    print(f"  - LoRA Rank: {config['lora_rank']}")
    print(f"  - Batch Size: {config['batch_size']}")
    print(f"  - Max Length: {config['max_length']}")
    print(f"  - Gradient Checkpointing: {config['gradient_checkpointing']}")
    print(f"\n显存分布:")
    print(f"  - 模型权重: {estimate['model']:.1f} GB")
    print(f"  - LoRA 参数: {estimate['lora']:.2f} GB")
    print(f"  - 优化器状态: {estimate['optimizer']:.2f} GB")
    print(f"  - 梯度: {estimate['gradient']:.2f} GB")
    print(f"  - 激活值: {estimate['activation']:.1f} GB")
    print(f"  - 总计: {estimate['total']:.1f} GB")
    print(f"  - 推荐显存: {estimate['recommended_vram']:.0f} GB")
    print(f"{'='*50}\n")


# Chat format templates
CHAT_FORMATS = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>",
        "generation_prompt": "<|im_start|>assistant\n"
    },
    "llama": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "generation_prompt": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "bos": "<|begin_of_text|>"
    },
    "phi": {
        "system": "<|system|>\n{content}<|end|>\n",
        "user": "<|user|>\n{content}<|end|>\n",
        "assistant": "<|assistant|>\n{content}<|end|>",
        "generation_prompt": "<|assistant|>\n"
    },
    "gemma": {
        "system": "",
        "user": "<start_of_turn>user\n{content}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n{content}<end_of_turn>",
        "generation_prompt": "<start_of_turn>model\n"
    },
    "mistral": {
        "system": "",
        "user": "[INST] {content} [/INST]",
        "assistant": "{content}</s>",
        "generation_prompt": ""
    },
    "alpaca": {
        "system": "### Instruction:\n{content}\n\n",
        "user": "### Input:\n{content}\n\n",
        "assistant": "### Response:\n{content}",
        "generation_prompt": "### Response:\n"
    }
}


def build_prompt(messages: list, chat_format: str = "chatml") -> str:
    """根据 chat format 构建 prompt"""
    fmt = CHAT_FORMATS.get(chat_format, CHAT_FORMATS["chatml"])
    
    parts = []
    if "bos" in fmt:
        parts.append(fmt["bos"])
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system" and fmt["system"]:
            parts.append(fmt["system"].format(content=content))
        elif role == "user":
            parts.append(fmt["user"].format(content=content))
        elif role == "assistant":
            parts.append(fmt["assistant"].format(content=content))
    
    if messages[-1]["role"] != "assistant":
        parts.append(fmt["generation_prompt"])
    
    return "".join(parts)


if __name__ == "__main__":
    # 测试显存估算
    print_vram_estimate("llama3-8b", vram_gb=24)
    print_vram_estimate("llama3-8b", vram_gb=80)
    
    # 列出所有预设
    list_presets(vram_limit=80)
