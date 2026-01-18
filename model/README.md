# RAECM - Student Model Distillation

**Student Model Component** of RAECM framework for high-throughput router attribute identification. This module distills knowledge from the teacher-side LLM pipeline into efficient task-specific models, achieving 86.43% accuracy while enabling scalable deployment.

## 📦 Module Overview

The student model component addresses the throughput-reliability trade-off in Internet-scale identification:

- **High Throughput**: Local inference without API dependencies
- **Cost Effective**: Eliminates per-request API costs
- **Strong Accuracy**: Maintains strong performance on benchmark dataset
- **Scalable Deployment**: Suitable for continuous monitoring and large-scale analysis

## 🎯 Key Features

### Knowledge Distillation

- **Evidence-Grounded Training**: Learn from teacher's structured outputs with explicit evidence
- **Task-Specific Optimization**: Specialized models for vendor/OS/device type identification
- **Retrieval-Augmented**: Maintains RAG capability for long-tail cases
- **Independent Deployment**: Can run independently without teacher model dependency

### Model Architecture

- **Base Models**: Qwen2.5/Qwen3/Llama3 series
- **LoRA Fine-tuning**: Parameter-efficient adaptation
- **Multi-Task Support**: Vendor, OS, and device type identification
- **Configurable Size**: From 1.5B to 8B parameters

### Deployment Strategy

- **Independent Inference**: Student model handles identification tasks independently
- **Batch Processing**: Efficient parallel inference
- **Quality Control**: Confidence assessment and result verification
- **Large-Scale Deployment**: Suitable for continuous monitoring scenarios

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory (depends on model size)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Main dependencies:
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets
- accelerate
- peft
- bitsandbytes

### Download Pre-trained Models

```bash
# Using ModelScope (recommended for China)
pip install modelscope
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./models/Qwen2.5-3B-Instruct

# Or using Hugging Face
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
```

### Training

```bash
# Train vendor identification model (recommended: Qwen2.5-3B)
python train.py --mt vd --model qwen2.5-3b

# Train OS identification model
python train.py --mt os --model qwen2.5-3b

# Train device type identification model
python train.py --mt dt --model qwen2.5-3b

# Advanced training options
python train.py --mt vd --model qwen2.5-3b \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 2048
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --mt vd

# Specify model path
python evaluate.py --mt vd --model output/vendor/Qwen2.5-3B/best_model

# Quick evaluation (50 samples)
python evaluate.py --mt vd --max-samples 50
```

### Prediction

```bash
# Single prediction
python predict.py --mt vd \
  --model output/vendor/Qwen2.5-3B/best_model \
  --input "Port 8291 (Winbox), HTTP banner: RouterOS"

# Batch prediction
python predict.py --mt vd \
  --model output/vendor/Qwen2.5-3B/best_model \
  --input-file test_data.jsonl \
  --output-file predictions.jsonl
```

## 📁 Project Structure

### Core Files

```
model/
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
├── predict.py                   # Prediction entry point
├── full_run.py                  # Complete pipeline (train + evaluate)
├── config.yaml                  # Global configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

### Training Core

```
model/training/
├── config.py                    # Configuration management
├── data_processor.py            # Data processing pipeline
├── trainer.py                   # Standard trainer
├── evaluator.py                 # Model evaluation
├── inference.py                 # Inference service
├── simple_classifier.py         # Simplified classification training
├── train_evaluator.py           # Training evaluator
├── metrics_recorder.py          # Metrics recording
├── model_manager.py             # Model management
├── model_presets.py             # Model presets
├── gpu_check.py                 # GPU checking
├── data_pipeline_v2.py          # Data pipeline v2
├── evaluation_v2.py             # Evaluation v2
│
└── distillation/                # Distillation training module
    ├── config.py               # Distillation configuration
    ├── trainer.py              # Distillation trainer
    ├── datasets.py             # Dataset handling
    ├── losses.py               # Loss functions
    ├── schedulers.py           # Learning rate schedulers
    ├── memory.py               # Memory management
    └── utils.py                # Utility functions
```

### Model Configurations

```
model/configs/
├── Qwen2.5-1.5B-Instruct.yaml   # Qwen2.5-1.5B config
├── Qwen2.5-3B-Instruct.yaml     # Qwen2.5-3B config (recommended)
├── Qwen2.5-7B-Instruct.yaml     # Qwen2.5-7B config
├── Qwen3-0.6B.yaml              # Qwen3-0.6B config
├── Qwen3-4B.yaml                # Qwen3-4B config
├── Qwen3-8B-Base.yaml           # Qwen3-8B config
├── Llama-3-8B-Instruct.yaml     # Llama3-8B config
└── ...                          # Other model configs
```

### Training Data

```
model/input/
├── example_train.jsonl          # Example training data
├── vendor_model_train.jsonl     # Vendor identification data
├── os_model_train.jsonl         # OS identification data
└── devicetype_model_train.jsonl # Device type identification data
```

### Prompt Templates

```
model/prompt/
├── product_prompts.json         # Product identification prompts
├── check_prompts.json           # Verification prompts
├── student.json                 # Student model prompts
└── new_prompt.json              # New prompts
```

### Training Scripts

```
model/bash/
├── qwen3-0.6b.sh                # Qwen3-0.6B training script
├── qwen3-8b.sh                  # Qwen3-8B training script
├── qwen3-32b.sh                 # Qwen3-32B training script
└── llama3-8b.sh                 # Llama3-8B training script
```

### Runtime Directories

```
model/
├── models/                      # Pre-trained models (download required)
│   ├── Qwen2.5-3B-Instruct/
│   └── ...
│
├── output/                      # Training outputs
│   ├── vendor/                 # Vendor identification models
│   ├── os/                     # OS identification models
│   └── devicetype/             # Device type identification models
│
├── result/                      # Evaluation results
│   ├── evaluation_report_vd.json
│   ├── evaluation_report_vd.md
│   └── ...
│
└── logs/                        # Training logs
```

## 🎯 Task Types

| Parameter | Task | Description | Example Labels |
|-----------|------|-------------|----------------|
| `vd` | Vendor Identification | Identify device vendor | MikroTik, Cisco, Juniper |
| `os` | OS Identification | Identify operating system | RouterOS, IOS, JunOS |
| `dt` | Device Type | Identify device type | router, switch, firewall |

## 💻 Recommended Model Configurations

### Qwen2.5 Series (Recommended) ⭐

| Model | Parameters | Memory | Speed | Use Case |
|-------|-----------|--------|-------|----------|
| qwen2.5-1.5b | 1.5B | ~8GB | Fastest | Quick testing |
| qwen2.5-3b | 3B | ~12GB | Fast | **Production (recommended)** |
| qwen2.5-7b | 7B | ~20GB | Medium | High accuracy requirements |

### Qwen3 Series

| Model | Parameters | Memory | Speed | Use Case |
|-------|-----------|--------|-------|----------|
| qwen3-0.6b | 0.6B | ~6GB | Fastest | Rapid prototyping |
| qwen3-4b | 4B | ~14GB | Fast | Balanced performance |
| qwen3-8b | 8B | ~22GB | Medium | High accuracy |

### Llama3 Series

| Model | Parameters | Memory | Speed | Use Case |
|-------|-----------|--------|-------|----------|
| llama3-8b | 8B | ~22GB | Medium | Comparative experiments |

## 📊 Training Parameters

### Basic Configuration

```bash
python train.py \
  --mt vd \                    # Task type (vd/os/dt)
  --model qwen2.5-3b \         # Model name
  --epochs 3 \                 # Training epochs
  --batch-size 16 \            # Batch size
  --learning-rate 2e-5 \       # Learning rate
  --max-length 2048            # Maximum sequence length
```

### Advanced Configuration

```bash
python train.py \
  --mt vd \
  --model qwen2.5-3b \
  --lora-r 8 \                 # LoRA rank
  --lora-alpha 16 \            # LoRA alpha
  --lora-dropout 0.05 \        # LoRA dropout
  --warmup-ratio 0.1 \         # Warmup ratio
  --weight-decay 0.01 \        # Weight decay
  --gradient-accumulation 4    # Gradient accumulation steps
```

## 📈 Evaluation

### Basic Evaluation

```bash
# Evaluate vendor identification model
python evaluate.py --mt vd

# Evaluate OS identification model
python evaluate.py --mt os

# Evaluate device type identification model
python evaluate.py --mt dt
```

### Specify Model Path

```bash
python evaluate.py \
  --mt vd \
  --model output/vendor/Qwen2.5-3B/best_model
```

### Evaluation Output

Generates comprehensive reports:
- `result/evaluation_report_vd.json` - JSON format detailed report
- `result/evaluation_report_vd.md` - Markdown format report

Report includes:
- Overall metrics (accuracy, F1 score)
- Per-class metrics (precision, recall, F1)
- Confusion matrix
- Label distribution
- Error analysis

## 🔮 Model Prediction

### Single Prediction

```bash
python predict.py \
  --mt vd \
  --model output/vendor/Qwen2.5-3B/best_model \
  --input "Port 8291 (Winbox), HTTP banner: RouterOS"
```

### Batch Prediction

```bash
python predict.py \
  --mt vd \
  --model output/vendor/Qwen2.5-3B/best_model \
  --input-file test_data.jsonl \
  --output-file predictions.jsonl
```

## 🎛️ Configuration Files

### Global Configuration (config.yaml)

```yaml
# Training configuration
training:
  epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  max_length: 2048
  
# LoRA configuration
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  
# Data configuration
data:
  train_ratio: 0.8
  valid_ratio: 0.1
  test_ratio: 0.1
```

### Model Configuration (configs/*.yaml)

Each model has its own configuration file containing:
- Model path
- Tokenizer configuration
- Training parameters
- LoRA parameters

## 🚀 Complete Training Pipeline

Use `full_run.py` for end-to-end training-evaluation-prediction:

```bash
python full_run.py --mt vd --model qwen2.5-3b
```

Pipeline includes:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Report generation
5. Best model saving

## 📊 Data Formats

### Training Data (JSONL)

```json
{
  "input": "Port 8291 (Winbox), HTTP banner: RouterOS v6.49",
  "output": "MikroTik",
  "Services": [...]
}
```

### Prediction Input (JSONL)

```json
{
  "IP Index": "192.168.1.1",
  "Services": [
    {
      "Port": 8291,
      "Protocol": "Winbox",
      "Banner": "RouterOS v6.49"
    }
  ]
}
```

### Prediction Output (JSONL)

```json
{
  "IP Index": "192.168.1.1",
  "Vendor": "MikroTik",
  "Confidence": 0.95,
  "Services": [...]
}
```

## 📊 Data Processing Pipeline

```
Labeled Data (input/*.jsonl)
    ↓
Data Processing (data_processor.py)
    ├─ Format conversion
    ├─ Train/valid/test split
    └─ Tokenization
    ↓
Model Training (trainer.py)
    ├─ LoRA fine-tuning
    ├─ Gradient accumulation
    └─ Checkpoint saving
    ↓
Model Evaluation (evaluator.py)
    ├─ Accuracy calculation
    ├─ Per-class metrics
    └─ Confusion matrix
    ↓
Trained Model (output/*/best_model)
```

## 🎯 Use Cases

### Scenario 1: High-Throughput Identification

```
Scanning Data → Student Model → Rapid Classification
```

1. Deploy trained student model
2. Process observations in batches
3. Achieve high throughput with maintained accuracy

### Scenario 2: Continuous Monitoring

```
Periodic Scanning → Student Model → Longitudinal Analysis
```

1. Deploy student model for continuous monitoring
2. Track attribute changes over time
3. Cost-effective large-scale analysis

## ❓ Frequently Asked Questions

### Q1: How much training data is needed?

Recommended data volume:
- **Minimum**: Sufficient samples per class for model convergence
- **Recommended**: Adequate samples per class for robust performance
- **Ideal**: Rich dataset with diverse samples per class

### Q2: How long does training take?

Training time depends on:
- Model size (smaller models train faster)
- Data volume (more data requires longer training)
- GPU performance (better GPUs reduce training time)

### Q3: How to handle GPU memory issues?

Solutions:
- Reduce `batch_size` (e.g., from 16 to 8)
- Reduce `max_length` (e.g., from 2048 to 1024)
- Use smaller model (e.g., qwen2.5-1.5b)
- Enable gradient accumulation

### Q4: How to improve model accuracy?

Strategies:
- Increase training data
- Increase training epochs
- Adjust learning rate
- Use larger model
- Tune LoRA parameters

### Q5: How to optimize inference speed?

Optimizations:
- Use smaller model (qwen2.5-1.5b)
- Increase batch size
- Use GPU inference
- Enable mixed precision (fp16/bf16)

### Q6: What is the accuracy-cost trade-off?

| Strategy | Accuracy | Cost | Throughput |
|----------|----------|------|------------|
| Teacher | Highest | High | Low |
| Student | Strong | Low | High |

### Q7: What is the relationship between student model and teacher pipeline?

The student model learns from the teacher pipeline through knowledge distillation:
- Compatible input/output formats
- Can be deployed independently
- Shared task definitions and label systems
- Consistent evaluation metrics

## 🔬 Performance Metrics

Based on benchmark evaluation:

| Metric | Teacher | Student |
|--------|---------|---------|
| Accuracy | Highest | Strong |
| Throughput | Low | High |
| Cost | High | Low |
| Latency | High | Low |

## 💡 Best Practices

### Model Selection

- **Quick testing**: qwen2.5-1.5b or qwen3-0.6b
- **Production deployment**: qwen2.5-3b (recommended)
- **High accuracy requirements**: qwen2.5-7b or qwen3-8b

### Training Parameters

- **Learning rate**: Typically in the range of 1e-5 to 2e-5
- **Batch size**: Start with 16 (increase if memory allows)
- **Training epochs**: Usually 3-5 epochs (adjust based on validation performance)

### Data Preparation

- Ensure data quality (clean, deduplicated)
- Balance class distribution
- Split train/valid/test appropriately

### Evaluation Metrics

- Focus on F1 score (balances precision and recall)
- Check confusion matrix (identify confusable classes)
- Analyze error samples (improve data or model)

## 📚 Related Documentation

- [training/config.py](training/config.py) - Training configuration
- [training/trainer.py](training/trainer.py) - Trainer implementation
- [training/evaluator.py](training/evaluator.py) - Evaluator implementation
- [../README.md](../README.md) - Project overview
- [../recog/README.md](../recog/README.md) - Teacher pipeline


