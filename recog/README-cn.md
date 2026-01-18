# RAECM - 以证据为中心的路由器资产识别

**RAECM**（Router Asset Evidence-Centric Multi-agent，路由器资产证据中心多智能体系统）是一个面向互联网规模IPv6路由器属性识别的自主框架。它利用大语言模型（LLM）将轻量级多端口探测产生的含噪网络测量数据转化为结构化、可验证且可追溯的资产标签。

## 📦 系统概述

RAECM解决了将异构且含噪的服务数据大规模转换为细粒度、可审计的语义标签这一持续性挑战。通过可扩展的多智能体分析流水线，RAECM实现了：

- **高准确率**：在ground-truth基准数据集上实现强劲的识别性能
- **成本效益**：相比无约束的直接推理大幅降低总体成本
- **可靠性提升**：相比直接LLM推理显著提升准确率
- **可扩展部署**：蒸馏学生模型在高吞吐量场景下保持强劲准确率

## 🎯 核心特性

### 以证据为中心的多智能体框架

- **专业化分析员**：将语义标注分解为可扩展的专业化智能体
- **显式证据**：每个预测都基于可验证的证据，并附带可靠性权重
- **事后验证**：CheckAnalyst执行一致性验证和保守修正
- **检索增强生成**：外部知识库支持基于证据的推理

### 互联网规模优化

- **内容哈希**：去重和缓存复用，处理重复观测
- **熵引导排序**：优先处理高信号观测，提升效率
- **教师-学生架构**：蒸馏学生模型处理常规案例，教师专注于复杂场景
- **轻量级探测**：高效的多端口扫描，最小化开销

### 下游应用

- **指纹构建**：从证据关联的结构化输出自动生成指纹
- **纵向监测**：支持漂移检测和时序分析
- **可审计输出**：可追溯且可维护的识别结果

## 🚀 快速开始

本模块实现了RAECM的**教师端**基于LLM的识别流水线。

### 前置要求

- Python 3.8+
- OpenAI兼容API访问（GPT-4、Claude、Gemini、DeepSeek等）
- 网络扫描数据（Nmap、Masscan或类似工具）

### 安装

```bash
# 安装依赖
pip install openai requests
```

### 配置

编辑 `6Analyst/config.py` 设置您的API凭证：

```python
# API配置
API_KEY = "your-api-key"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4"

# 处理参数
BATCH_SIZE = 3              # LLM推理批大小
MAX_INPUT_TOKENS = 4096     # 最大输入token数
SPEED_LEVEL = 6             # 速度级别（1-6）

# 熵过滤
DEFAULT_ENTROPY_RATIO = 1.0  # 基于熵的过滤比例
```

**支持的API**：
- OpenAI（GPT-4、GPT-3.5等）
- Anthropic（Claude系列）
- Google（Gemini系列）
- DeepSeek（DeepSeek-V3等）
- 其他OpenAI兼容端点

### 数据准备

将扫描数据放置在 `6Analyst/data/input/` 目录：

```
6Analyst/data/input/
├── example_input.jsonl        # 示例数据（已包含）
├── vendor_input_data.jsonl    # 厂商识别数据
├── os_input_data.jsonl        # 操作系统识别数据
└── devicetype_input_data.jsonl # 设备类型识别数据
```

**输入格式**（JSONL）：

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

### 运行识别

```bash
# 基本用法
python run_6analyst.py

# 指定任务类型
python run_6analyst.py --task vd    # 厂商识别
python run_6analyst.py --task os    # 操作系统识别
python run_6analyst.py --task dt    # 设备类型识别

# 高级选项
python run_6analyst.py --max-records 1000 --speed 6 --model gpt-4
```

### 输出格式

系统生成带有显式证据的结构化标签：

```json
{
  "IP Index": "192.168.1.1",
  "Vendor": "MikroTik",
  "OS": "RouterOS",
  "Device Type": "router",
  "Confidence": "high",
  "Evidence": [
    "Port 8291 (Winbox) - MikroTik专有协议",
    "HTTP banner包含'RouterOS'",
    "SSH banner: 'SSH-2.0-ROSSSH'"
  ],
  "Reliability": 0.95,
  "Services": [...]
}
```

## 📁 项目结构

### 核心文件

```
recog/
├── run_6analyst.py              # 主入口
├── requirements.txt             # 依赖项
├── README.md                    # 英文文档
└── README-cn.md                 # 中文文档（本文件）
```

### 6Analyst包

```
recog/6Analyst/
├── config.py                    # 配置管理
├── run.py                       # 主执行逻辑
├── run_config.py                # 运行时配置
│
├── data_cleaner.py              # 数据清洗和规范化
├── product_analyst.py           # 产品识别智能体
├── check_analyst.py             # 验证和一致性检查
├── entropy_sorter.py            # 熵引导排序
├── base_analyst.py              # 基础智能体类
│
├── accuracy_calculator.py       # 准确率指标
├── accuracy_evaluator.py        # 评估框架
├── cost_calculator.py           # 成本跟踪
├── multi_thread_runner.py       # 并行处理
│
├── data/                        # 数据目录
│   ├── input/                   # 输入数据
│   │   ├── example_input.jsonl # 示例数据
│   │   ├── vendor_input_data.jsonl
│   │   ├── os_input_data.jsonl
│   │   └── devicetype_input_data.jsonl
│   └── output/                  # 输出结果（生成）
│
├── prompts/                     # 提示词模板
│   ├── product_prompts.json    # 产品识别提示词
│   └── check_prompts.json      # 验证提示词
│
├── utils/                       # 工具模块
│   ├── logger.py               # 日志工具
│   ├── token_counter.py        # Token计数
│   ├── error_logger.py         # 错误日志
│   ├── html_extractor.py       # HTML提取
│   └── common.py               # 通用工具
│
└── docs/                        # 文档
    └── entropy_sampling_guide.md
```

## 🔧 配置指南

### API配置

```python
# config.py

# API设置
API_KEY = "your-api-key"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4"

# 模型定价（每1K tokens）
MODEL_PRICES = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    # ... 更多模型
}
```

### 处理参数

```python
# 处理配置
MAX_RECORDS = None              # 处理所有记录（None）或限制数量
BATCH_SIZE = 3                  # LLM推理批大小
MAX_INPUT_TOKENS = 4096         # 最大输入token数
DEBUG_MODE = False              # 启用调试输出

# 速度级别
# 较低值：较慢，有延迟
# 较高值：较快，最小延迟
DEFAULT_SPEED_LEVEL = 6
```

### 数据清洗规则

```python
# 要移除的字段（避免数据泄露）
REMOVE_TOP_FIELDS = {"IP Index", "Timestamp", "OS", "Vendor", "Device Type"}
REMOVE_SERVICE_FIELDS = {"Body sha256"}

# HTTP错误模式（过滤噪声响应）
HTTP_ERROR_PATTERNS = [
    (r'400\s*Bad\s*Request', '400 Bad Request'),
    (r'404(\s*Not\s*Found)?', '404 Not Found'),
    (r'500\s*Internal\s*Server\s*Error', '500 Internal Server Error'),
    # ... 更多模式
]
```

### 基于熵的过滤

```python
# 熵配置
DEFAULT_ENTROPY_RATIO = 1.0     # 保留所有数据（无过滤）
                                # 降低该值以过滤低熵观测

# 基于难度的采样
DIFFICULTY_VENDORS = {
    'easy': ['MikroTik', 'Keenetic'],
    'normal': ['Cisco', 'Juniper'],
    'hard': []  # 其他厂商
}

DEFAULT_DIFFICULTY_RATIOS = {
    'easy': 0.7,    # 大部分简单厂商
    'normal': 0.15, # 部分普通厂商
    'hard': 0.15    # 部分困难厂商
}
```

## 📊 数据处理流水线

```
原始扫描数据（input/*.jsonl）
    ↓
数据清洗（data_cleaner.py）
    ├─ 移除敏感字段
    ├─ 过滤HTTP错误
    ├─ 规范化SSH算法
    └─ 计算信息熵
    ↓
产品识别（product_analyst.py）
    ├─ 批量LLM推理
    ├─ 提取厂商/操作系统/设备类型
    └─ 生成证据链
    ↓
一致性检查（check_analyst.py）
    ├─ 验证跨字段一致性
    ├─ 验证证据充分性
    └─ 保守修正
    ↓
熵引导排序（entropy_sorter.py）
    ├─ 按信息熵排序
    └─ 过滤低质量观测
    ↓
最终结构化输出（final_analysis.jsonl）
```

## 🔑 核心组件

### 多智能体架构

| 组件 | 功能 | 描述 |
|------|------|------|
| `data_cleaner.py` | 数据规范化 | 移除敏感字段、过滤噪声、计算熵 |
| `product_analyst.py` | 产品识别 | 基于LLM的厂商/操作系统/设备类型推理 |
| `check_analyst.py` | 验证 | 一致性验证和保守修正 |
| `entropy_sorter.py` | 质量控制 | 熵引导过滤和排序 |

### 以证据为中心的设计

每个识别结果包含：
- **结构化标签**：厂商、操作系统、设备类型
- **显式证据**：支持每个标签的可观测数据
- **可靠性评分**：每个预测的置信度权重
- **溯源**：可追溯的推理链

### 检索增强生成（RAG）

- 路由器文档的外部知识库
- 按需检索相关技术规范
- 推理和验证的证据基础
- 支持长尾型号和专业术语

## 🎯 使用场景

### 场景1：互联网规模资产发现

```
网络扫描 → RAECM识别 → 资产清单
```

1. 执行轻量级多端口探测
2. 运行RAECM识别流水线
3. 生成带证据的结构化资产清单

### 场景2：指纹构建

```
RAECM输出 → 证据聚类 → 自动化指纹
```

1. 收集证据关联的结构化输出
2. 按证据模式聚类
3. 生成可维护的指纹规则

### 场景3：纵向监测

```
定期扫描 → RAECM分析 → 漂移检测
```

1. 持续观测目标网络
2. 跟踪属性随时间的变化
3. 检测配置漂移和更新

## ❓ 常见问题

### Q1：RAECM与传统指纹识别有什么不同？

RAECM结合了基于LLM的语义理解与显式证据基础的优势：
- **适应性**：处理未见过的型号和演进的固件
- **跨端口推理**：整合来自多个服务的证据
- **可审计性**：每个预测都有可验证的证据支撑
- **可维护性**：减少手动签名工程工作

### Q2：RAECM如何处理稀疏或含噪的观测？

- **保守弃权**：证据不足时默认为"未知"
- **熵引导过滤**：优先处理高信息量观测
- **跨字段验证**：检测和修正不一致性
- **证据加权**：为每条证据分配可靠性评分

### Q3：支持哪些识别任务？

- **厂商识别**：MikroTik、Cisco、Juniper、Huawei等
- **操作系统识别**：RouterOS、IOS、JunOS、VRP等
- **设备类型**：router、switch、firewall、gateway等

### Q4：如何优化成本和吞吐量？

- **批处理**：增加 `BATCH_SIZE` 以提高吞吐量
- **速度级别**：设置 `SPEED_LEVEL` 为更高值以获得最大吞吐量
- **模型选择**：使用成本效益模型进行大规模处理
- **熵过滤**：调整 `DEFAULT_ENTROPY_RATIO` 过滤低质量数据
- **学生模型**：使用蒸馏模型处理常规案例（见 ../model/）

### Q5：如何与现有工作流集成？

RAECM输出设计用于下游集成：
- **结构化JSON**：易于解析和处理
- **证据链**：支持审计和验证
- **指纹生成**：自动化规则提取
- **API兼容**：与标准扫描工具配合使用

## 🔬 性能指标

基于ground-truth基准评估：

| 指标 | 描述 |
|------|------|
| 总体准确率 | 基准数据集上的高准确率 |
| 厂商准确率 | 强劲的厂商识别性能 |
| 操作系统准确率 | 强劲的操作系统识别性能 |
| 设备类型准确率 | 强劲的设备类型分类 |
| 成本降低 | 大幅成本节省 |
| 准确率提升 | 相比直接推理显著提升 |

## 📚 相关文档

- [config.py](6Analyst/config.py) - 完整配置参考
- [data_cleaner.py](6Analyst/data_cleaner.py) - 数据清洗规则
- [product_analyst.py](6Analyst/product_analyst.py) - 识别逻辑
- [../README-cn.md](../README-cn.md) - 项目总览
- [../model/README-cn.md](../model/README-cn.md) - 学生模型训练


