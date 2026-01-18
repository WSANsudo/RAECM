#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU检查模块
GPU availability check module
"""

import sys


def check_gpu_availability(require_gpu: bool = True, verbose: bool = True) -> dict:
    """
    检查GPU是否可用
    
    Args:
        require_gpu: 是否强制要求GPU，如果为True且无GPU则退出程序
        verbose: 是否打印详细信息
    
    Returns:
        dict: GPU信息字典
    """
    result = {
        'cuda_available': False,
        'device_count': 0,
        'devices': [],
        'current_device': None,
        'torch_version': None,
        'cuda_version': None,
    }
    
    try:
        import torch
        result['torch_version'] = torch.__version__
        result['cuda_available'] = torch.cuda.is_available()
        
        if result['cuda_available']:
            result['device_count'] = torch.cuda.device_count()
            result['current_device'] = torch.cuda.current_device()
            result['cuda_version'] = torch.version.cuda
            
            for i in range(result['device_count']):
                props = torch.cuda.get_device_properties(i)
                # 获取当前显存使用情况
                free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
                total_mem = props.total_memory / (1024**3)
                device_info = {
                    'index': i,
                    'name': props.name,
                    'total_memory_gb': round(total_mem, 2),
                    'free_memory_gb': round(free_mem, 2),
                    'compute_capability': f"{props.major}.{props.minor}",
                }
                result['devices'].append(device_info)
    except ImportError:
        if verbose:
            print("错误: PyTorch未安装")
        if require_gpu:
            sys.exit(1)
        return result
    
    if verbose:
        print_gpu_info(result)
    
    if require_gpu and not result['cuda_available']:
        print("\n错误: 未检测到可用的GPU!")
        print("请确保:")
        print("  1. 已安装NVIDIA显卡驱动")
        print("  2. 已安装支持CUDA的PyTorch版本")
        print("  3. 运行 'pip install torch --index-url https://download.pytorch.org/whl/cu118' 安装GPU版本")
        print("\n如需使用CPU训练，请添加 --cpu 参数")
        sys.exit(1)
    
    return result


def print_gpu_info(info: dict) -> None:
    """打印GPU信息"""
    print("\n" + "="*50)
    print("GPU 环境检查")
    print("="*50)
    print(f"PyTorch版本: {info['torch_version']}")
    print(f"CUDA可用: {'是' if info['cuda_available'] else '否'}")
    
    if info['cuda_available']:
        print(f"CUDA版本: {info['cuda_version']}")
        print(f"GPU数量: {info['device_count']}")
        print("-"*50)
        for device in info['devices']:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      总显存: {device['total_memory_gb']} GB")
            print(f"      可用显存: {device['free_memory_gb']} GB")
            print(f"      计算能力: {device['compute_capability']}")
        
        # 显存建议
        total_vram = sum(d['total_memory_gb'] for d in info['devices'])
        print("-"*50)
        print("训练建议:")
        if total_vram >= 48:
            print("  ✓ 大显存GPU，可训练大模型 (32B+)")
            print("  ✓ 建议: batch_size=4-8, 使用 SDPA 加速")
            print("  ✓ 可使用 bf16 精度，更稳定")
        elif total_vram >= 24:
            print("  ✓ 显存充足，可使用较大batch_size和更长序列")
            print("  ✓ 建议: batch_size=2-4, LoRA rank=32-64")
        elif total_vram >= 16:
            print("  ✓ 显存足够，建议使用LoRA + gradient_checkpointing")
        elif total_vram >= 8:
            print("  ⚠ 显存有限，建议使用QLoRA + 小batch_size")
        else:
            print("  ⚠ 显存较小，建议使用更小的模型或CPU训练")
    else:
        print("\n⚠ 警告: 将使用CPU运行，速度会很慢!")
    print("="*50 + "\n")


def estimate_training_memory(model_params_b: float, use_lora: bool = True, 
                             batch_size: int = 1, seq_length: int = 1024) -> dict:
    """
    估算训练所需显存
    
    Args:
        model_params_b: 模型参数量（十亿）
        use_lora: 是否使用LoRA
        batch_size: 批次大小
        seq_length: 序列长度
    
    Returns:
        dict: 显存估算信息
    """
    # 基础模型权重 (bf16: 2 bytes per param)
    model_memory = model_params_b * 2  # GB
    
    if use_lora:
        # LoRA只训练少量参数，优化器状态小很多
        optimizer_memory = model_params_b * 0.1  # 约10%
        gradient_memory = model_params_b * 0.05  # 约5%
    else:
        # 全参数微调需要更多显存
        optimizer_memory = model_params_b * 8  # AdamW: 8 bytes per param
        gradient_memory = model_params_b * 2
    
    # KV Cache和激活值
    activation_memory = batch_size * seq_length * 0.001  # 粗略估算
    
    total = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    return {
        'model_memory_gb': round(model_memory, 2),
        'optimizer_memory_gb': round(optimizer_memory, 2),
        'gradient_memory_gb': round(gradient_memory, 2),
        'activation_memory_gb': round(activation_memory, 2),
        'total_estimated_gb': round(total * 1.2, 2),  # 加20%余量
    }


def get_device(prefer_gpu: bool = True) -> str:
    """
    获取推荐的设备
    
    Args:
        prefer_gpu: 是否优先使用GPU
    
    Returns:
        str: 'cuda' 或 'cpu'
    """
    import torch
    if prefer_gpu and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
