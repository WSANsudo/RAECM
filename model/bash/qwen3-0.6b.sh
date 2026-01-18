#!/bin/bash
# Qwen3-0.6B 训练脚本
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

python train_distill.py --mt "${1:-vd}" --config configs/qwen3-0.6b.yaml
