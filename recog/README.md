# RAECM - Evidence-Centric Router Asset Identification

**RAECM** (Router Asset Evidence-Centric Multi-agent) is an autonomous framework for Internet-scale IPv6 router attribute identification. It leverages Large Language Models (LLMs) to transform noisy network measurements from lightweight multi-port probing into structured, verifiable, and traceable asset labels.

## 📦 System Overview

RAECM addresses the persistent challenge of converting heterogeneous and noisy service artifacts into fine-grained, auditable semantic labels at Internet scale. Through an extensible multi-agent analysis pipeline, RAECM achieves:

- **High Accuracy**: Strong identification performance on ground-truth benchmark
- **Cost Efficiency**: Substantial reduction in overall cost compared to unconstrained direct inference
- **Improved Reliability**: Significant accuracy improvement over direct LLM inference
- **Scalable Deployment**: Distilled student model maintains strong accuracy for high-throughput scenarios

## 🎯 Key Features

### Evidence-Centric Multi-Agent Framework

- **Specialized Analysts**: Decompose semantic labeling into extensible specialized agents
- **Explicit Evidence**: Each prediction is grounded in verifiable evidence with reliability weights
- **Post-hoc Verification**: CheckAnalyst performs consistency validation and conservative correction
- **Retrieval-Augmented Generation**: External knowledge base supports evidence-grounded reasoning

### Internet-Scale Optimization

- **Content Hashing**: Deduplication and cache reuse for repeated observations
- **Entropy-Guided Sorting**: Prioritize high-signal observations for efficient processing
- **Teacher-Student Architecture**: Distilled student model handles routine cases, teacher focuses on complex scenarios
- **Lightweight Probing**: Efficient multi-port scanning with minimal overhead

### Downstream Applications

- **Fingerprint Construction**: Automated generation from evidence-linked structured outputs
- **Longitudinal Monitoring**: Support for drift detection and temporal analysis
- **Auditable Outputs**: Traceable and maintainable identification results

## 🚀 Quick Start

This module implements the **teacher-side** LLM-based identification pipeline of RAECM.

### Prerequisites

- Python 3.8+
- OpenAI-compatible API access (GPT-4, Claude, Gemini, DeepSeek, etc.)
- Network scanning data (Nmap, Masscan, or similar tools)

### Installation

```bash
# Install dependencies
pip install openai requests
```

### Configuration

Edit `6Analyst/config.py` to set your API credentials:

```python
# API Configuration
API_KEY = "your-api-key"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4"

# Processing Parameters
BATCH_SIZE = 3              # Batch size for LLM inference
MAX_INPUT_TOKENS = 4096     # Maximum input tokens
SPEED_LEVEL = 6             # Speed level (1-6)

# Entropy Filtering
DEFAULT_ENTROPY_RATIO = 1.0  # Entropy-based filtering ratio
```

**Supported APIs**:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude series)
- Google (Gemini series)
- DeepSeek (DeepSeek-V3, etc.)
- Other OpenAI-compatible endpoints

### Data Preparation

Place your scanning data in `6Analyst/data/input/`:

```
6Analyst/data/input/
├── example_input.jsonl        # Example data (included)
├── vendor_input_data.jsonl    # Vendor identification data
├── os_input_data.jsonl        # OS identification data
└── devicetype_input_data.jsonl # Device type identification data
```

**Input Format** (JSONL):

```json
{
  "IP Index": "192.168.1.1",
  "Timestamp": "2024-01-01 00:00:00",
  "Services": [
    {
      "Port": 80,
      "Protocol": "HTTP",
      "Banner": "Server: nginx/1.18.0",
      "Body": "<!DOCTYPE html>..."
    }
  ]
}
```

### Running Identification

```bash
# Basic usage
python run_6analyst.py

# Specify task type
python run_6analyst.py --task vd    # Vendor identification
python run_6analyst.py --task os    # OS identification
python run_6analyst.py --task dt    # Device type identification

# Advanced options
python run_6analyst.py --max-records 1000 --speed 6 --model gpt-4
```

### Output Format

The system generates structured labels with explicit evidence:

```json
{
  "IP Index": "192.168.1.1",
  "Vendor": "MikroTik",
  "OS": "RouterOS",
  "Device Type": "router",
  "Confidence": "high",
  "Evidence": [
    "Port 8291 (Winbox) - MikroTik proprietary protocol",
    "HTTP banner contains 'RouterOS'",
    "SSH banner: 'SSH-2.0-ROSSSH'"
  ],
  "Reliability": 0.95,
  "Services": [...]
}
```

## 📁 Project Structure

### Core Files

```
recog/
├── run_6analyst.py              # Main entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

### 6Analyst Package

```
recog/6Analyst/
├── config.py                    # Configuration management
├── run.py                       # Main execution logic
├── run_config.py                # Runtime configuration
│
├── data_cleaner.py              # Data cleaning and normalization
├── product_analyst.py           # Product identification agent
├── check_analyst.py             # Verification and consistency checking
├── entropy_sorter.py            # Entropy-guided sorting
├── base_analyst.py              # Base agent class
│
├── accuracy_calculator.py       # Accuracy metrics
├── accuracy_evaluator.py        # Evaluation framework
├── cost_calculator.py           # Cost tracking
├── multi_thread_runner.py       # Parallel processing
│
├── data/                        # Data directory
│   ├── input/                   # Input data
│   │   ├── example_input.jsonl # Example data
│   │   ├── vendor_input_data.jsonl
│   │   ├── os_input_data.jsonl
│   │   └── devicetype_input_data.jsonl
│   └── output/                  # Output results (generated)
│
├── prompts/                     # Prompt templates
│   ├── product_prompts.json    # Product identification prompts
│   └── check_prompts.json      # Verification prompts
│
├── utils/                       # Utility modules
│   ├── logger.py               # Logging utilities
│   ├── token_counter.py        # Token counting
│   ├── error_logger.py         # Error logging
│   ├── html_extractor.py       # HTML extraction
│   └── common.py               # Common utilities
│
└── docs/                        # Documentation
    └── entropy_sampling_guide.md
```

## 🔧 Configuration Guide

### API Configuration

```python
# config.py

# API Settings
API_KEY = "your-api-key"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4"

# Model Pricing (per 1K tokens)
MODEL_PRICES = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    # ... more models
}
```

### Processing Parameters

```python
# Processing Configuration
MAX_RECORDS = None              # Process all records (None) or limit
BATCH_SIZE = 3                  # Batch size for LLM inference
MAX_INPUT_TOKENS = 4096         # Maximum input tokens
DEBUG_MODE = False              # Enable debug output

# Speed Levels
# Lower values: Slower with delays
# Higher values: Faster with minimal delays
DEFAULT_SPEED_LEVEL = 6
```

### Data Cleaning Rules

```python
# Fields to remove (avoid data leakage)
REMOVE_TOP_FIELDS = {"IP Index", "Timestamp", "OS", "Vendor", "Device Type"}
REMOVE_SERVICE_FIELDS = {"Body sha256"}

# HTTP error patterns (filter noisy responses)
HTTP_ERROR_PATTERNS = [
    (r'400\s*Bad\s*Request', '400 Bad Request'),
    (r'404(\s*Not\s*Found)?', '404 Not Found'),
    (r'500\s*Internal\s*Server\s*Error', '500 Internal Server Error'),
    # ... more patterns
]
```

### Entropy-Based Filtering

```python
# Entropy Configuration
DEFAULT_ENTROPY_RATIO = 1.0     # Keep all data (no filtering)
                                # Reduce to filter low-entropy observations

# Difficulty-based Sampling
DIFFICULTY_VENDORS = {
    'easy': ['MikroTik', 'Keenetic'],
    'normal': ['Cisco', 'Juniper'],
    'hard': []  # Other vendors
}

DEFAULT_DIFFICULTY_RATIOS = {
    'easy': 0.7,    # Majority easy vendors
    'normal': 0.15, # Some normal vendors
    'hard': 0.15    # Some hard vendors
}
```

## 📊 Data Processing Pipeline

```
Raw Scanning Data (input/*.jsonl)
    ↓
Data Cleaning (data_cleaner.py)
    ├─ Remove sensitive fields
    ├─ Filter HTTP errors
    ├─ Normalize SSH algorithms
    └─ Calculate information entropy
    ↓
Product Identification (product_analyst.py)
    ├─ Batch LLM inference
    ├─ Extract vendor/OS/device type
    └─ Generate evidence chains
    ↓
Consistency Checking (check_analyst.py)
    ├─ Validate cross-field consistency
    ├─ Verify evidence sufficiency
    └─ Conservative correction
    ↓
Entropy-Guided Sorting (entropy_sorter.py)
    ├─ Rank by information entropy
    └─ Filter low-quality observations
    ↓
Final Structured Output (final_analysis.jsonl)
```

## 🔑 Key Components

### Multi-Agent Architecture

| Component | Function | Description |
|-----------|----------|-------------|
| `data_cleaner.py` | Data Normalization | Remove sensitive fields, filter noise, calculate entropy |
| `product_analyst.py` | Product Identification | LLM-based vendor/OS/device type inference |
| `check_analyst.py` | Verification | Consistency validation and conservative correction |
| `entropy_sorter.py` | Quality Control | Entropy-guided filtering and ranking |

### Evidence-Centric Design

Each identification result includes:
- **Structured Labels**: Vendor, OS, Device Type
- **Explicit Evidence**: Observable artifacts supporting each label
- **Reliability Scores**: Confidence weights for each prediction
- **Provenance**: Traceable reasoning chains

### Retrieval-Augmented Generation (RAG)

- External knowledge base of router documentation
- On-demand retrieval of relevant technical specifications
- Evidence grounding for both inference and verification
- Support for long-tail models and specialized terminology

## 🎯 Use Cases

### Scenario 1: Internet-Scale Asset Discovery

```
Network Scanning → RAECM Identification → Asset Inventory
```

1. Perform lightweight multi-port probing
2. Run RAECM identification pipeline
3. Generate structured asset inventory with evidence

### Scenario 2: Fingerprint Construction

```
RAECM Outputs → Evidence Clustering → Automated Fingerprints
```

1. Collect evidence-linked structured outputs
2. Cluster by evidence patterns
3. Generate maintainable fingerprint rules

### Scenario 3: Longitudinal Monitoring

```
Periodic Scanning → RAECM Analysis → Drift Detection
```

1. Continuous observation of target networks
2. Track attribute changes over time
3. Detect configuration drift and updates

## ❓ Frequently Asked Questions

### Q1: What makes RAECM different from traditional fingerprinting?

RAECM combines the strengths of LLM-based semantic understanding with explicit evidence grounding:
- **Adaptability**: Handles unseen models and evolving firmware
- **Cross-Port Reasoning**: Integrates evidence from multiple services
- **Auditability**: Every prediction is backed by verifiable evidence
- **Maintainability**: Reduces manual signature engineering effort

### Q2: How does RAECM handle sparse or noisy observations?

- **Conservative Abstention**: Defaults to "unknown" when evidence is insufficient
- **Entropy-Guided Filtering**: Prioritizes high-information observations
- **Cross-Field Validation**: Detects and corrects inconsistencies
- **Evidence Weighting**: Assigns reliability scores to each piece of evidence

### Q3: What are the supported identification tasks?

- **Vendor Identification**: MikroTik, Cisco, Juniper, Huawei, etc.
- **OS Identification**: RouterOS, IOS, JunOS, VRP, etc.
- **Device Type**: router, switch, firewall, gateway, etc.

### Q4: How to optimize for cost and throughput?

- **Batch Processing**: Increase `BATCH_SIZE` for better throughput
- **Speed Level**: Set `SPEED_LEVEL` to higher values for maximum throughput
- **Model Selection**: Use cost-effective models for large-scale processing
- **Entropy Filtering**: Adjust `DEFAULT_ENTROPY_RATIO` to filter low-quality data
- **Student Model**: Use distilled model for routine cases (see ../model/)

### Q5: How to integrate with existing workflows?

RAECM outputs are designed for downstream integration:
- **Structured JSON**: Easy to parse and process
- **Evidence Chains**: Support for audit and verification
- **Fingerprint Generation**: Automated rule extraction
- **API-Compatible**: Works with standard scanning tools

## 🔬 Performance Metrics

Based on ground-truth benchmark evaluation:

| Metric | Description |
|--------|-------------|
| Overall Accuracy | High accuracy on benchmark dataset |
| Vendor Accuracy | Strong vendor identification performance |
| OS Accuracy | Strong OS identification performance |
| Device Type Accuracy | Strong device type classification |
| Cost Reduction | Substantial cost savings |
| Accuracy Improvement | Significant gains vs. direct inference |

## 📚 Related Documentation

- [config.py](6Analyst/config.py) - Complete configuration reference
- [data_cleaner.py](6Analyst/data_cleaner.py) - Data cleaning rules
- [product_analyst.py](6Analyst/product_analyst.py) - Identification logic
- [../README.md](../README.md) - Project overview
- [../model/README.md](../model/README.md) - Student model training


