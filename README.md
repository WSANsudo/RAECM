# RAECM: Internet-Scale Router Attribute Identification

**RAECM** (Router Asset Evidence-Centric Multi-agent) is an autonomous framework for Internet-scale router attribute identification via an evidence-centric multi-agent approach.
**The complete code will be open-sourced after passing peer review**

## 📦 Overview

RAECM addresses the persistent challenge of converting heterogeneous and noisy service artifacts into fine-grained, auditable semantic labels at Internet scale. It leverages Large Language Models (LLMs) to transform lightweight multi-port probing measurements into structured, verifiable, and traceable asset labels.

### Key Achievements

- **High Accuracy**: Achieves strong performance on ground-truth benchmark dataset
- **Significant Improvement**: Substantial accuracy gains over unconstrained direct LLM inference
- **Cost Reduction**: Significantly lower cost compared to direct inference approach
- **Efficient Student Model**: Distilled model maintains strong accuracy for high-throughput scenarios

## 🎯 Core Innovations

### 1. Evidence-Centric Multi-Agent Framework

- **Specialized Analysts**: Decompose semantic labeling into extensible specialized agents
- **Explicit Evidence**: Each prediction grounded in verifiable evidence with reliability weights
- **Post-hoc Verification**: CheckAnalyst performs consistency validation and conservative correction
- **Retrieval-Augmented Generation**: External knowledge base supports evidence-grounded reasoning

### 2. Internet-Scale Optimization

- **Content Hashing**: Deduplication and cache reuse for repeated observations
- **Entropy-Guided Sorting**: Prioritize high-signal observations for efficient processing
- **Teacher-Student Architecture**: Distilled student model handles routine cases
- **Lightweight Probing**: Efficient multi-port scanning with minimal overhead

### 3. Downstream Applications

- **Fingerprint Construction**: Automated generation from evidence-linked structured outputs
- **Longitudinal Monitoring**: Support for drift detection and temporal analysis
- **Auditable Outputs**: Traceable and maintainable identification results

## 📁 Project Structure

```
Analyst-master/
├── README.md              # This file - Project overview
│
├── recog/                 # Teacher-side LLM identification pipeline
│   ├── Analyst/         # Core implementation
│   ├── run_analyst.py   # Main entry point
│   └── README.md         # Complete documentation
│
└── model/                 # Student model distillation
    ├── training/         # Training core
    ├── configs/          # Model configurations
    ├── input/            # Training data
    ├── train.py          # Training entry point
    ├── evaluate.py       # Evaluation entry point
    └── README.md         # Complete documentation
```

## 🚀 Quick Start

### Teacher Pipeline (recog)

High-accuracy identification using LLMs:

```bash
cd recog
pip install openai requests
# Configure API in Analyst/config.py
python run_analyst.py
```

**See [recog/README.md](recog/README.md) for complete documentation.**

### Student Model (model)

High-throughput identification using distilled models:

```bash
cd model
pip install -r requirements.txt
# Download model
python train.py --mt vd --model qwen3-4b
python evaluate.py --mt vd
```

**See [model/README.md](model/README.md) for complete documentation.**

## 🎯 System Architecture

### Multi-Agent Pipeline

```
Raw Scanning Data
    ↓
Data Cleaning & Normalization
    ├─ Remove sensitive fields
    ├─ Filter noise
    └─ Calculate entropy
    ↓
Product Identification (Specialized Analysts)
    ├─ Vendor identification
    ├─ OS identification
    └─ Device type identification
    ↓
Consistency Checking (CheckAnalyst)
    ├─ Cross-field validation
    ├─ Evidence sufficiency check
    └─ Conservative correction
    ↓
Structured Output with Evidence
```

### Teacher-Student Architecture

```
Teacher Pipeline (LLM-based)
    ├─ High accuracy
    ├─ Evidence generation
    └─ Complex case handling
    ↓
Knowledge Distillation
    ↓
Student Model (Distilled)
    ├─ High throughput
    ├─ Cost effective
    └─ Large-scale deployment
```

## 📊 Use Cases

### Scenario 1: Internet-Scale Asset Discovery

```
Network Scanning → RAECM Teacher → Asset Inventory
```

- Perform lightweight multi-port probing
- Run RAECM identification pipeline
- Generate structured asset inventory with evidence

### Scenario 2: High-Throughput Monitoring

```
Continuous Scanning → RAECM Student → Rapid Classification
```

- Deploy distilled student model
- Process large-scale observations
- Achieve high throughput with maintained accuracy

### Scenario 3: Fingerprint Construction

```
RAECM Outputs → Evidence Clustering → Automated Fingerprints
```

- Collect evidence-linked structured outputs
- Cluster by evidence patterns
- Generate maintainable fingerprint rules

## 🔧 System Requirements

### Teacher Pipeline (recog)

- Python 3.8+
- OpenAI-compatible API access
- Network scanning data

### Student Model (model)

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU training)
- 8GB+ GPU memory

## 📚 Documentation

Each module contains complete, standalone documentation:

- **[README.md](README.md)** (this file) - Project overview
- **[recog/README.md](recog/README.md)** - Teacher pipeline complete guide
  - Quick start and installation
  - Configuration reference
  - Data formats and processing
  - API integration
  - Performance optimization
  - FAQ and troubleshooting
  
- **[model/README.md](model/README.md)** - Student model complete guide
  - Training procedures
  - Model configurations
  - Evaluation metrics
  - Deployment strategies
  - Performance tuning
  - FAQ and best practices

## ❓ Frequently Asked Questions

### Q1: Which module should I use?

- **Teacher (recog)**: High-accuracy identification, evidence generation, complex case handling
- **Student (model)**: High-throughput processing, cost-effective deployment, large-scale scenarios

### Q2: What makes RAECM different?

| Feature | Traditional Fingerprinting | RAECM |
|---------|---------------------------|-------|
| Adaptability | Low (manual updates) | High (LLM-based) |
| Evidence | Implicit | Explicit and traceable |
| Cross-Port Reasoning | Limited | Comprehensive |
| Maintenance | High manual effort | Automated |
| Auditability | Limited | Full provenance |

### Q3: How to get training data?

- Use teacher pipeline (recog) to generate labeled data
- Manual annotation of scanning results
- Existing labeled datasets

### Q4: What are the performance trade-offs?

| Metric | Teacher | Student |
|--------|---------|---------|
| Accuracy | Highest | Strong |
| Throughput | Low | High |
| Cost | High | Low |
| Latency | High | Low |

### Q5: How to optimize for my use case?

**For accuracy**: Use teacher pipeline with high-quality models (GPT-4, Claude)

**For throughput**: Deploy student model with batch processing

**For cost**: Use student model for large-scale deployment

**For auditability**: Use teacher pipeline to enable evidence generation and verification

## 🔬 Performance Metrics

Based on ground-truth benchmark evaluation:

### Overall Performance

| Metric | Description |
|--------|-------------|
| Teacher Accuracy | High accuracy on benchmark dataset |
| Student Accuracy | Strong accuracy with efficient inference |
| Accuracy Improvement | Significant gains vs. direct inference |
| Cost Reduction | Substantial cost savings |

### Task-Specific Performance

| Task | Teacher | Student |
|------|---------|---------|
| Vendor Identification | High accuracy | Strong accuracy |
| OS Identification | High accuracy | Strong accuracy |
| Device Type | High accuracy | Strong accuracy |

## 🌟 Key Advantages

### Compared to Traditional Fingerprinting

- **Adaptability**: Handles unseen models and evolving firmware
- **Cross-Port Reasoning**: Integrates evidence from multiple services
- **Reduced Maintenance**: Automated fingerprint generation
- **Evidence Grounding**: Every prediction backed by verifiable evidence

### Compared to Direct LLM Inference

- **Higher Accuracy**: Significant improvement through multi-agent framework
- **Lower Cost**: Substantial cost reduction through optimization
- **Better Reliability**: Conservative abstention on insufficient evidence
- **Auditability**: Explicit evidence chains and provenance

