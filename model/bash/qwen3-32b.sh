#!/bin/bash
# Qwen3-32B 训练脚本 - A800 80GB 优化
set -e
cd "$(dirname "$0")/.."

# GPU 设置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# 性能优化
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export TF32_OVERRIDE=1

# 启动训练
python train_distill.py --mt "${1:-vd}" --config configs/qwen3-32b.yaml
