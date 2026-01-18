#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练入口脚本
Training entry script for Network Device Analyzer Model
"""

import argparse
import os
import sys

from training.config import (
    load_config,
    get_model_config,
    get_training_config,
    get_data_config
)
from training.trainer import NetworkDeviceTrainer
from training.gpu_check import check_gpu_availability


def main():
    parser = argparse.ArgumentParser(
        description='网络设备分析模型训练工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置训练
  python train.py
  
  # 指定配置文件
  python train.py --config my_config.yaml
  
  # 指定数据路径
  python train.py --train data/train.jsonl --valid data/valid.jsonl
  
  # 从检查点恢复训练
  python train.py --resume output/checkpoint-1000
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    parser.add_argument(
        '--train', 
        type=str,
        help='训练数据路径 (覆盖配置文件)'
    )
    parser.add_argument(
        '--valid', 
        type=str,
        help='验证数据路径 (覆盖配置文件)'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='输出目录 (覆盖配置文件)'
    )
    parser.add_argument(
        '--resume', 
        type=str,
        help='从检查点恢复训练'
    )
    parser.add_argument(
        '--epochs', 
        type=int,
        help='训练轮数 (覆盖配置文件)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int,
        help='批次大小 (覆盖配置文件)'
    )
    parser.add_argument(
        '--lr', 
        type=float,
        help='学习率 (覆盖配置文件)'
    )
    parser.add_argument(
        '--no-lora', 
        action='store_true',
        help='禁用LoRA，使用全参数微调'
    )
    parser.add_argument(
        '--cpu', 
        action='store_true',
        help='强制使用CPU训练'
    )
    
    args = parser.parse_args()
    
    # GPU检查（除非指定--cpu）
    if not args.cpu:
        check_gpu_availability(require_gpu=True)
    else:
        check_gpu_availability(require_gpu=False)
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    
    # 应用命令行参数覆盖
    if args.output:
        training_config.output_dir = args.output
    if args.resume:
        training_config.resume_from_checkpoint = args.resume
    if args.epochs:
        training_config.num_train_epochs = args.epochs
    if args.batch_size:
        training_config.per_device_train_batch_size = args.batch_size
    if args.lr:
        training_config.learning_rate = args.lr
    if args.no_lora:
        model_config.use_lora = False
    if args.cpu:
        # 禁用FP16以支持CPU
        training_config.fp16 = False
        training_config.bf16 = False
    
    # 确定数据路径
    train_path = args.train or os.path.join(data_config.output_dir, 'train.jsonl')
    valid_path = args.valid or os.path.join(data_config.output_dir, 'valid.jsonl')
    
    # 检查数据文件
    if not os.path.exists(train_path):
        print(f"错误: 训练数据文件不存在: {train_path}")
        print("请先运行数据处理: python -m training.data_processor")
        sys.exit(1)
    if not os.path.exists(valid_path):
        print(f"错误: 验证数据文件不存在: {valid_path}")
        sys.exit(1)
    
    # 打印配置信息
    print("\n" + "="*50)
    print("训练配置")
    print("="*50)
    print(f"模型路径: {model_config.model_name_or_path}")
    print(f"使用LoRA: {model_config.use_lora}")
    if model_config.use_lora:
        print(f"  - LoRA Rank: {model_config.lora_rank}")
        print(f"  - LoRA Alpha: {model_config.lora_alpha}")
    print(f"最大序列长度: {model_config.max_length}")
    print(f"训练轮数: {training_config.num_train_epochs}")
    print(f"批次大小: {training_config.per_device_train_batch_size}")
    print(f"梯度累积: {training_config.gradient_accumulation_steps}")
    print(f"学习率: {training_config.learning_rate}")
    print(f"FP16: {training_config.fp16}")
    print(f"输出目录: {training_config.output_dir}")
    print(f"训练数据: {train_path}")
    print(f"验证数据: {valid_path}")
    print("="*50 + "\n")
    
    # 创建训练器
    trainer = NetworkDeviceTrainer(model_config, training_config)
    
    # 加载模型
    print("正在加载模型...")
    trainer.load_model_and_tokenizer()
    
    # 配置LoRA
    if model_config.use_lora:
        print("配置LoRA...")
        trainer.model = trainer.setup_lora(trainer.model)
    
    # 准备数据集
    print("准备数据集...")
    train_dataset, valid_dataset = trainer.prepare_dataset(train_path, valid_path)
    
    # 开始训练
    print("\n开始训练...")
    
    # 获取早停配置 - 优先使用 simple 配置
    simple_config = config.get('simple', {})
    early_stopping_patience = simple_config.get('early_stopping_patience', 3)
    early_stopping_min_delta = simple_config.get('early_stopping_min_delta', 0.001)
    
    trainer.train(
        train_dataset, 
        valid_dataset,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta
    )
    
    # 保存最终模型
    final_output = os.path.join(training_config.output_dir, 'final_model')
    print(f"\n保存最终模型到: {final_output}")
    trainer.save_model(final_output)
    
    print("\n训练完成!")
    print(f"模型已保存到: {final_output}")


if __name__ == "__main__":
    main()
