"""
数据处理模块
Data processing module for Network Device Analyzer Training System
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass

from .config import DataConfig, load_config, get_data_config


# Default System prompt - vendor模型默认提示词
DEFAULT_SYSTEM_PROMPT = """Identify device vendor from network scan evidence.

=== OUTPUT FORMAT ===
{
  "ip": "device IP address",
  "vendor": "manufacturer name or null",
  "result_type": "direct or inferred",
  "confidence": 0.0-1.0,
  "evidence": [{"src": "source", "val": "value", "weight": 0.0-1.0}],
  "evidence_quality": "strong/moderate/weak/insufficient"
}

=== RULES ===
- vendor: Device manufacturer (MikroTik/Cisco/Juniper/Huawei/etc.) or null
- result_type: "direct" if explicit in banner, "inferred" if deduced
- confidence: 0.9+ for direct, 0.7-0.9 for inference
- evidence_quality: strong/moderate/weak/insufficient

Output JSON only. No explanation."""

# 当前使用的 prompt（可通过 load_prompt 切换）
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

# 当前模型类型和对应的输出字段
CURRENT_MODEL_TYPE = "vendor"  # vendor, os, devicetype
MODEL_OUTPUT_FIELDS = {
    "vendor": ["ip", "vendor", "result_type", "confidence", "evidence", "evidence_quality"],
    "os": ["ip", "os", "result_type", "confidence", "evidence", "evidence_quality"],
    "devicetype": ["ip", "devicetype", "result_type", "confidence", "evidence", "evidence_quality"]
}

# 评估字段 - 根据模型类型确定
EVAL_FIELDS = ['vendor']


# ============================================================
# 标签规范化映射表
# ============================================================

# DeviceType 标签映射（根据提示词规则）
# 标准类型: router | firewall | server | camera | nas | printer | iot | appliance
DEVICETYPE_LABEL_MAPPING = {
    # → router (根据提示词: management_interface/gateway/access_point/broadband_router/network_device → router)
    'switch': 'router',
    'gateway': 'router',
    'network_device': 'router',
    'modem': 'router',
    'access_point': 'router',
    'wireless_controller': 'router',
    'wireless_access_point': 'router',
    'wireless_ap': 'router',
    'management-interface': 'router',
    'broadband_router': 'router',
    
    # → server (根据提示词: web_interface → server)
    'mail_server': 'server',
    'web_server': 'server',
    'web-server': 'server',
    'web_interface': 'server',
    'ftp_server': 'server',
    'vpn_server': 'server',
    'streaming_server': 'server',
    'database': 'server',
    'application_server': 'server',
    'ilo': 'server',
    'BMC/management': 'server',
    'management-controller': 'server',
    'management_controller': 'server',
    
    # → appliance (网络设备/控制器)
    'network_management': 'appliance',
    'network_controller': 'appliance',
    'controller': 'appliance',
    'load_balancer': 'appliance',
    'proxy': 'appliance',
    'cdn_server': 'appliance',
    'cdn': 'appliance',
    'orchestrator': 'appliance',
    'infrastructure': 'appliance',
    'service': 'appliance',
    
    # → firewall
    'security_appliance': 'firewall',
    'security': 'firewall',
    
    # → camera
    'nvr': 'camera',
    'dvr': 'camera',
    
    # → iot
    'voip': 'iot',
    'phone': 'iot',
    'weather_station': 'iot',
    'embedded_device': 'iot',
    'embedded': 'iot',
}

# Vendor 标签映射（统一大小写和别名）
VENDOR_LABEL_MAPPING = {
    'Juniper Networks': 'Juniper',
    'ZyXEL': 'Zyxel',
    'ZYXEL': 'Zyxel',
    'zyxel': 'Zyxel',
    'Ubiquiti Inc.': 'Ubiquiti',
    'UPVEL': 'Upvel',
    'HPE': 'Hewlett Packard Enterprise',
    'HP': 'Hewlett Packard Enterprise',
    'Keenetic Ltd.': 'Keenetic',
    'Keenetic Ltd': 'Keenetic',
    'YAMAHA': 'Yamaha',
    'NEC': 'NEC Platforms',
    'Netgear': 'NETGEAR',
    'Port25': 'Port25 Solutions',
    'Apache': 'Apache Software Foundation',
    'Fedora Project': 'Fedora',
    'Nginx Inc.': 'nginx',
}


def normalize_label(label: str, model_type: str) -> str:
    """
    根据模型类型规范化标签
    
    Args:
        label: 原始标签
        model_type: 模型类型 (vendor/os/devicetype)
        
    Returns:
        规范化后的标签
    """
    if not label:
        return label
    
    if model_type == 'vendor':
        return VENDOR_LABEL_MAPPING.get(label, label)
    elif model_type == 'devicetype':
        return DEVICETYPE_LABEL_MAPPING.get(label, label)
    else:
        # os 类型暂不做映射
        return label


def load_prompt(prompt_file: str = "prompt/student.json", prompt_id: str = "vendor") -> str:
    """
    从 JSON 文件加载指定的提示词，并设置对应的模型类型
    
    Args:
        prompt_file: 提示词 JSON 文件路径
        prompt_id: 提示词 ID (vendor/os/devicetype)
        
    Returns:
        提示词内容
    """
    global SYSTEM_PROMPT, CURRENT_MODEL_TYPE, EVAL_FIELDS
    
    # 设置模型类型
    if prompt_id in ["vendor", "os", "devicetype"]:
        CURRENT_MODEL_TYPE = prompt_id
        if prompt_id == "vendor":
            EVAL_FIELDS = ['vendor']
        elif prompt_id == "os":
            EVAL_FIELDS = ['os']
        elif prompt_id == "devicetype":
            EVAL_FIELDS = ['devicetype']
        print(f"模型类型设置为: {CURRENT_MODEL_TYPE}, 评估字段: {EVAL_FIELDS}")
    
    if not os.path.exists(prompt_file):
        print(f"警告: 提示词文件不存在 {prompt_file}，使用默认提示词")
        return DEFAULT_SYSTEM_PROMPT
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = data.get('prompts', [])
    
    for prompt in prompts:
        if prompt.get('id') == prompt_id:
            system_prompt = prompt.get('system_prompt')
            if system_prompt:
                SYSTEM_PROMPT = system_prompt
                print(f"已加载提示词: [{prompt_id}] {prompt.get('name', '')}")
                return system_prompt
    
    print(f"警告: 未找到提示词 ID '{prompt_id}'，使用默认提示词")
    return DEFAULT_SYSTEM_PROMPT


def list_prompts(prompt_file: str = "prompt/student.json") -> List[Dict]:
    """
    列出所有可用的提示词
    
    Args:
        prompt_file: 提示词 JSON 文件路径
        
    Returns:
        提示词列表
    """
    if not os.path.exists(prompt_file):
        print(f"提示词文件不存在: {prompt_file}")
        return []
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = data.get('prompts', [])
    
    print("\n可用的提示词:")
    print("-" * 60)
    for p in prompts:
        prompt_text = p.get('system_prompt', '')
        token_estimate = len(prompt_text) // 3  # 粗略估算 token 数
        print(f"  [{p.get('id')}] {p.get('name', '未命名')}")
        print(f"      {p.get('description', '')}")
        print(f"      约 {token_estimate} tokens")
        print()
    
    return prompts


@dataclass
class ProcessingStats:
    """数据处理统计信息"""
    total_records: int = 0
    valid_records: int = 0
    negative_records: int = 0  # 负样本数量
    filtered_by_status: int = 0
    filtered_by_quality: int = 0
    truncated_samples: int = 0
    dpo_pairs: int = 0  # DPO配对数量
    augmented_samples: int = 0  # 数据增强样本数


class DataAugmenter:
    """数据增强器 - Banner扰动和证据丢弃"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def augment_banner(self, banner: str) -> str:
        """对 Banner 文本进行轻微扰动"""
        if not banner or random.random() > 0.5:
            return banner
        
        augmented = banner
        aug_type = random.choice(['case', 'space', 'truncate'])
        
        if aug_type == 'case':
            # 随机改变部分字符的大小写
            chars = list(augmented)
            for i in range(len(chars)):
                if random.random() < 0.1 and chars[i].isalpha():
                    chars[i] = chars[i].swapcase()
            augmented = ''.join(chars)
        elif aug_type == 'space':
            # 随机调整空格
            if '  ' in augmented:
                augmented = augmented.replace('  ', ' ')
        elif aug_type == 'truncate':
            # 轻微截断末尾
            if len(augmented) > 20:
                truncate_len = random.randint(1, min(5, len(augmented) // 10))
                augmented = augmented[:-truncate_len]
        
        return augmented
    
    def drop_evidence(self, original_data: Dict, drop_rate: float = 0.2) -> Dict:
        """随机丢弃部分证据，训练模型在信息不完整时的鲁棒性"""
        if 'Services' not in original_data:
            return original_data
        
        services = original_data.get('Services', {})
        if len(services) <= 1:
            return original_data
        
        augmented_data = {'Services': {}}
        for port_service, data in services.items():
            if random.random() > drop_rate:
                augmented_data['Services'][port_service] = data.copy()
        
        # 确保至少保留一个服务
        if not augmented_data['Services']:
            key = random.choice(list(services.keys()))
            augmented_data['Services'][key] = services[key].copy()
        
        return augmented_data
    
    def augment_record(self, record: Dict) -> Dict:
        """对单条记录进行数据增强"""
        import copy
        augmented = copy.deepcopy(record)
        augmented['_augmented'] = True
        
        original_data = augmented.get('original_data', {})
        aug_type = random.choice(['banner', 'drop', 'both'])
        
        if aug_type in ['banner', 'both']:
            # Banner 扰动
            for port_service in original_data.get('Services', {}):
                if 'Banner' in original_data['Services'][port_service]:
                    original_data['Services'][port_service]['Banner'] = \
                        self.augment_banner(original_data['Services'][port_service]['Banner'])
        
        if aug_type in ['drop', 'both']:
            # 证据丢弃
            augmented['original_data'] = self.drop_evidence(original_data)
        
        return augmented


class DataProcessor:
    """数据处理器，负责数据清洗、转换和划分"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        初始化数据处理器
        
        Args:
            config: 数据配置对象
        """
        self.config = config or DataConfig()
        self.stats = ProcessingStats()
        self._tokenizer = None
        self.augmenter = DataAugmenter()
    
    def augment_dataset(self, records: List[Dict], augment_ratio: float = 0.3) -> List[Dict]:
        """
        对数据集进行增强
        
        Args:
            records: 原始记录列表
            augment_ratio: 增强比例
            
        Returns:
            增强后的记录列表
        """
        augmented_records = list(records)
        num_to_augment = int(len(records) * augment_ratio)
        
        for _ in range(num_to_augment):
            original = random.choice(records)
            augmented = self.augmenter.augment_record(original)
            augmented_records.append(augmented)
            self.stats.augmented_samples += 1
        
        return augmented_records
    
    def find_similar_positive(
        self, 
        negative_record: Dict, 
        positive_samples: List[Dict]
    ) -> Optional[Dict]:
        """
        为负样本找到相似的正样本
        
        基于证据来源的相似度匹配
        
        Args:
            negative_record: 负样本
            positive_samples: 正样本列表
            
        Returns:
            最相似的正样本
        """
        if not positive_samples:
            return None
        
        neg_services = set(negative_record.get('original_data', {}).get('Services', {}).keys())
        neg_teacher = negative_record.get('teacher_output', {})
        neg_vendor = neg_teacher.get('vendor')
        
        best_match = None
        best_score = -1
        
        for pos_record in positive_samples:
            score = 0
            pos_services = set(pos_record.get('original_data', {}).get('Services', {}).keys())
            pos_teacher = pos_record.get('teacher_output', {})
            
            # 服务端口相似度
            if neg_services and pos_services:
                overlap = len(neg_services & pos_services)
                score += overlap * 2
            
            # 如果厂商相同（但负样本判断错误），给予更高分数
            # 这样可以学习"相似输入但不同输出"的模式
            if neg_vendor and pos_teacher.get('vendor') == neg_vendor:
                score += 5
            
            if score > best_score:
                best_score = score
                best_match = pos_record
        
        return best_match or random.choice(positive_samples)
    
    def load_data(self, file_path: str) -> List[Dict]:
        """
        加载JSONL数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            解析后的记录列表
            
        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
        
        self.stats.total_records = len(records)
        
        # 标签规范化处理
        records = self.normalize_labels(records)
        
        return records
    
    def normalize_labels(self, records: List[Dict]) -> List[Dict]:
        """
        标签规范化处理 - 统一标签格式和别名
        
        Args:
            records: 原始记录列表
            
        Returns:
            规范化后的记录列表
        """
        normalized_count = 0
        true_label_normalized = 0
        
        for record in records:
            teacher_output = record.get('teacher_output', {})
            true_label = record.get('true_label', {})
            
            # 根据当前模型类型获取对应字段
            if CURRENT_MODEL_TYPE == 'vendor':
                field = 'vendor'
            elif CURRENT_MODEL_TYPE == 'os':
                field = 'os'
            elif CURRENT_MODEL_TYPE == 'devicetype':
                # true_label 和 teacher_output 都使用 'devicetype'
                field = 'devicetype'
                teacher_field = 'devicetype'
            else:
                continue
            
            # 规范化 teacher_output 中的标签
            if teacher_output:
                tf = teacher_field if CURRENT_MODEL_TYPE == 'devicetype' else field
                original_label = teacher_output.get(tf)
                if original_label:
                    normalized_label = normalize_label(original_label, CURRENT_MODEL_TYPE)
                    if normalized_label != original_label:
                        teacher_output[tf] = normalized_label
                        normalized_count += 1
            
            # 规范化 true_label 中的标签
            if true_label:
                original_true = true_label.get(field)
                if original_true:
                    normalized_true = normalize_label(original_true, CURRENT_MODEL_TYPE)
                    if normalized_true != original_true:
                        true_label[field] = normalized_true
                        true_label_normalized += 1
        
        if normalized_count > 0:
            print(f"标签规范化 (teacher_output): 已处理 {normalized_count} 条记录")
        if true_label_normalized > 0:
            print(f"标签规范化 (true_label): 已处理 {true_label_normalized} 条记录")
        
        return records
    
    def clean_data(self, records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        清洗数据，分离正样本和负样本
        
        数据结构:
        - ip: IP地址
        - teacher_output: 老师模型输出 (包含 vendor, os, devicetype, confidence, evidence 等)
        - original_data: 原始扫描数据
        - is_correct: 正确性判断 (yes/no)
        - true_label: 真实标签 (用于评估)
          - vendor_model_train.jsonl: {"vendor": "xxx"}
          - os_model_train.jsonl: {"os": "xxx"}
          - devicetype_model_train.jsonl: {"devicetype": "xxx"}
        
        正负样本判断:
        - is_correct == 'yes': 正样本 (teacher_output 的预测正确)
        - is_correct == 'no': 负样本 (teacher_output 的预测错误，true_label 是正确答案)
        
        Args:
            records: 原始记录列表
            
        Returns:
            (正样本列表, 负样本列表)
        """
        positive_samples = []
        negative_samples = []
        
        # 根据模型类型确定 true_label 中的字段名
        if CURRENT_MODEL_TYPE == 'vendor':
            true_label_field = 'vendor'
        elif CURRENT_MODEL_TYPE == 'os':
            true_label_field = 'os'
        elif CURRENT_MODEL_TYPE == 'devicetype':
            true_label_field = 'devicetype'
        else:
            true_label_field = 'vendor'
        
        for record in records:
            # 检查必需字段
            if 'teacher_output' not in record or 'original_data' not in record:
                self.stats.filtered_by_status += 1
                continue
            
            teacher_output = record.get('teacher_output', {})
            
            # 检查 teacher_output 中是否有 evidence
            if 'evidence' not in teacher_output:
                self.stats.filtered_by_status += 1
                continue
            
            # 检查 evidence_quality - 对于正负样本都过滤掉 insufficient
            quality = teacher_output.get('evidence_quality', '')
            if quality == 'insufficient':
                self.stats.filtered_by_quality += 1
                continue
            
            # 检查 true_label 是否存在且有对应字段
            true_label = record.get('true_label', {})
            if not true_label or true_label_field not in true_label:
                # 没有 true_label，使用 is_correct 判断
                is_correct = record.get('is_correct', '')
                if is_correct == 'yes':
                    positive_samples.append(record)
                elif is_correct == 'no':
                    negative_samples.append(record)
                else:
                    self.stats.filtered_by_status += 1
                continue
            
            # 有 true_label，根据 is_correct 分类
            is_correct = record.get('is_correct', '')
            if is_correct == 'yes':
                positive_samples.append(record)
            elif is_correct == 'no':
                negative_samples.append(record)
            else:
                # 未知状态，跳过
                self.stats.filtered_by_status += 1
                continue
        
        self.stats.valid_records = len(positive_samples)
        self.stats.negative_records = len(negative_samples)
        return positive_samples, negative_samples
    
    def format_original_data(self, original_data: Dict) -> str:
        """
        格式化原始扫描数据为文本描述
        
        Args:
            original_data: 原始扫描数据 (包含 Services 等)
            
        Returns:
            格式化的文本
        """
        lines = ["Network scan evidence:", ""]
        
        services = original_data.get('Services', {})
        
        for port_service, data in services.items():
            banner = data.get('Banner', '')
            if banner:
                # 截断过长的 banner
                if len(banner) > 200:
                    banner = banner[:200] + "..."
                lines.append(f"[{port_service}] {banner}")
        
        return "\n".join(lines)
    
    def format_evidence(self, evidence: List[Dict], services: List[str] = None) -> str:
        """
        格式化证据为文本描述
        
        Args:
            evidence: 证据列表
            services: 检测到的服务列表
            
        Returns:
            格式化的证据文本
        """
        lines = ["请分析以下网络设备信息：", "", "证据列表："]
        
        # 按权重排序证据
        sorted_evidence = sorted(evidence, key=lambda x: x.get('weight', 0), reverse=True)
        
        for i, ev in enumerate(sorted_evidence, 1):
            src = ev.get('src', 'unknown')
            val = ev.get('val', '')
            weight = ev.get('weight', 0)
            lines.append(f"{i}. [{src}] {val} (权重: {weight})")
        
        if services:
            lines.append("")
            lines.append(f"检测到的服务: {', '.join(services)}")
        
        return "\n".join(lines)
    
    def format_output(self, record: Dict) -> str:
        """
        格式化输出为JSON字符串 - 从 teacher_output 中提取对应字段
        
        Args:
            record: 原始记录 (包含 teacher_output)
            
        Returns:
            JSON格式的输出字符串
        """
        teacher_output = record.get('teacher_output', {})
        ip = record.get('ip', '')
        
        # 获取证据列表
        evidence = teacher_output.get('evidence', [])
        # 确保证据格式正确 (只保留 src, val, weight)
        formatted_evidence = []
        for ev in evidence:
            formatted_evidence.append({
                "src": ev.get('src', ''),
                "val": ev.get('val', ''),
                "weight": ev.get('weight', 0.5)
            })
        
        # 根据模型类型构建输出
        if CURRENT_MODEL_TYPE == "vendor":
            output = {
                "ip": ip,
                "vendor": teacher_output.get('vendor'),
                "result_type": teacher_output.get('result_type', 'inferred'),
                "confidence": teacher_output.get('confidence', 0.5),
                "evidence": formatted_evidence,
                "evidence_quality": teacher_output.get('evidence_quality', 'moderate')
            }
        elif CURRENT_MODEL_TYPE == "os":
            output = {
                "ip": ip,
                "os": teacher_output.get('os'),
                "result_type": teacher_output.get('result_type', 'inferred'),
                "confidence": teacher_output.get('confidence', 0.5),
                "evidence": formatted_evidence,
                "evidence_quality": teacher_output.get('evidence_quality', 'moderate')
            }
        elif CURRENT_MODEL_TYPE == "devicetype":
            output = {
                "ip": ip,
                "devicetype": teacher_output.get('devicetype'),
                "result_type": teacher_output.get('result_type', 'inferred'),
                "confidence": teacher_output.get('confidence', 0.5),
                "evidence": formatted_evidence,
                "evidence_quality": teacher_output.get('evidence_quality', 'moderate')
            }
        else:
            # 默认 vendor 模型
            output = {
                "ip": ip,
                "vendor": teacher_output.get('vendor'),
                "result_type": teacher_output.get('result_type', 'inferred'),
                "confidence": teacher_output.get('confidence', 0.5),
                "evidence": formatted_evidence,
                "evidence_quality": teacher_output.get('evidence_quality', 'moderate')
            }
        
        return json.dumps(output, ensure_ascii=False)
    
    def convert_to_training_format(self, record: Dict) -> Dict:
        """
        将单条记录转换为训练格式
        
        新数据结构:
        - ip: IP地址
        - teacher_output: 老师模型输出
        - original_data: 原始扫描数据
        
        Args:
            record: 原始数据记录
            
        Returns:
            指令微调格式的训练样本
        """
        original_data = record.get('original_data', {})
        
        # 使用原始扫描数据作为用户输入
        user_content = self.format_original_data(original_data)
        
        # 使用老师输出作为助手回复（标签）
        assistant_content = self.format_output(record)
        
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    
    def convert_negative_to_training_format(self, record: Dict) -> Dict:
        """
        将负样本转换为训练格式
        
        负样本是 is_correct=no 的记录，表示 teacher_output 的预测错误。
        使用 true_label 中的正确标签作为训练目标。
        
        Args:
            record: 负样本记录
            
        Returns:
            指令微调格式的训练样本
        """
        original_data = record.get('original_data', {})
        teacher_output = record.get('teacher_output', {})
        true_label = record.get('true_label', {})
        ip = record.get('ip', '')
        
        # 使用原始扫描数据作为用户输入
        user_content = self.format_original_data(original_data)
        
        # 获取证据
        evidence = teacher_output.get('evidence', [])
        formatted_evidence = []
        for ev in evidence[:5]:  # 最多保留5条证据
            formatted_evidence.append({
                "src": ev.get('src', ''),
                "val": ev.get('val', ''),
                "weight": ev.get('weight', 0.5)
            })
        
        # 根据模型类型从 true_label 获取正确标签
        if CURRENT_MODEL_TYPE == "vendor":
            correct_label = true_label.get('vendor')
            output = {
                "ip": ip,
                "vendor": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        elif CURRENT_MODEL_TYPE == "os":
            correct_label = true_label.get('os')
            output = {
                "ip": ip,
                "os": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        elif CURRENT_MODEL_TYPE == "devicetype":
            correct_label = true_label.get('devicetype')
            output = {
                "ip": ip,
                "devicetype": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        else:
            correct_label = true_label.get('vendor')
            output = {
                "ip": ip,
                "vendor": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        
        assistant_content = json.dumps(output, ensure_ascii=False)
        
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "_is_negative_sample": True,  # 标记为负样本转换
            "_true_label": true_label  # 保留真实标签用于评估
        }
    
    def convert_to_dpo_format(
        self, 
        positive_record: Optional[Dict], 
        negative_record: Dict
    ) -> Dict:
        """
        将正负样本配对转换为 DPO 训练格式
        
        改进：使用相似正样本的输出模式生成更好的 chosen
        
        Args:
            positive_record: 相似的正样本记录
            negative_record: 负样本记录 (is_correct=no)
            
        Returns:
            DPO 训练格式的样本
        """
        original_data = negative_record.get('original_data', {})
        user_content = self.format_original_data(original_data)
        
        # 负样本的输出作为 rejected
        rejected_content = self.format_output(negative_record)
        
        # 使用改进的 chosen 生成
        chosen_content = self._generate_improved_chosen(negative_record, positive_record)
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "chosen": chosen_content,
            "rejected": rejected_content
        }
    
    def _generate_improved_chosen(
        self, 
        negative_record: Dict, 
        reference_positive: Optional[Dict]
    ) -> str:
        """
        基于 true_label 生成正确的 chosen 输出
        
        Args:
            negative_record: 负样本记录 (包含 true_label)
            reference_positive: 参考的正样本记录 (未使用)
            
        Returns:
            正确的 chosen 输出
        """
        neg_teacher = negative_record.get('teacher_output', {})
        true_label = negative_record.get('true_label', {})
        ip = negative_record.get('ip', '')
        
        # 获取证据
        evidence = neg_teacher.get('evidence', [])
        formatted_evidence = []
        for ev in evidence:
            formatted_evidence.append({
                "src": ev.get('src', ''),
                "val": ev.get('val', ''),
                "weight": ev.get('weight', 0.5)
            })
        
        # 根据模型类型从 true_label 获取正确标签
        if CURRENT_MODEL_TYPE == "vendor":
            correct_label = true_label.get('vendor')
            output = {
                "ip": ip,
                "vendor": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        elif CURRENT_MODEL_TYPE == "os":
            correct_label = true_label.get('os')
            output = {
                "ip": ip,
                "os": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        elif CURRENT_MODEL_TYPE == "devicetype":
            correct_label = true_label.get('devicetype')
            output = {
                "ip": ip,
                "devicetype": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        else:
            correct_label = true_label.get('vendor')
            output = {
                "ip": ip,
                "vendor": correct_label,
                "result_type": "inferred",
                "confidence": 0.8 if correct_label else 0.2,
                "evidence": formatted_evidence,
                "evidence_quality": "moderate" if correct_label else "insufficient"
            }
        
        return json.dumps(output, ensure_ascii=False)
    
    def _generate_conservative_output(self, record: Dict) -> str:
        """
        为负样本生成保守的输出（承认不确定性）
        
        当老师模型判断错误时，正确的做法通常是：
        1. 降低置信度
        2. 或者返回 null 表示无法确定
        
        Args:
            record: 负样本记录
            
        Returns:
            保守的 JSON 输出
        """
        teacher_output = record.get('teacher_output', {})
        ip = record.get('ip', '')
        
        # 获取证据
        evidence = teacher_output.get('evidence', [])
        formatted_evidence = []
        for ev in evidence:
            formatted_evidence.append({
                "src": ev.get('src', ''),
                "val": ev.get('val', ''),
                "weight": ev.get('weight', 0.3)  # 降低权重
            })
        
        # 根据模型类型构建保守输出
        if CURRENT_MODEL_TYPE == "vendor":
            output = {
                "ip": ip,
                "vendor": None,  # 无法确定
                "result_type": "inferred",
                "confidence": 0.3,  # 低置信度
                "evidence": formatted_evidence,
                "evidence_quality": "weak"
            }
        elif CURRENT_MODEL_TYPE == "os":
            output = {
                "ip": ip,
                "os": None,
                "result_type": "inferred",
                "confidence": 0.3,
                "evidence": formatted_evidence,
                "evidence_quality": "weak"
            }
        elif CURRENT_MODEL_TYPE == "devicetype":
            output = {
                "ip": ip,
                "devicetype": None,
                "result_type": "inferred",
                "confidence": 0.3,
                "evidence": formatted_evidence,
                "evidence_quality": "weak"
            }
        else:
            output = {
                "ip": ip,
                "vendor": None,
                "result_type": "inferred",
                "confidence": 0.3,
                "evidence": formatted_evidence,
                "evidence_quality": "weak"
            }
        
        return json.dumps(output, ensure_ascii=False)
    
    def create_dpo_pairs(
        self, 
        positive_samples: List[Dict], 
        negative_samples: List[Dict],
        max_pairs: Optional[int] = None
    ) -> List[Dict]:
        """
        创建 DPO 训练配对
        
        改进策略：
        1. 为每个负样本找到相似的正样本
        2. 使用改进的 chosen 生成方法
        
        Args:
            positive_samples: 正样本列表
            negative_samples: 负样本列表
            max_pairs: 最大配对数量
            
        Returns:
            DPO 配对列表
        """
        if not negative_samples:
            print("警告: 没有负样本，无法创建 DPO 配对")
            return []
        
        dpo_pairs = []
        
        for neg_record in negative_samples:
            # 找到相似的正样本
            similar_pos = self.find_similar_positive(neg_record, positive_samples)
            
            dpo_pair = self.convert_to_dpo_format(similar_pos, neg_record)
            dpo_pairs.append(dpo_pair)
            
            if max_pairs and len(dpo_pairs) >= max_pairs:
                break
        
        self.stats.dpo_pairs = len(dpo_pairs)
        return dpo_pairs
    
    def truncate_evidence(
        self, 
        evidence: List[Dict], 
        max_evidence: int = 10
    ) -> List[Dict]:
        """
        截断证据列表，保留高权重证据
        
        Args:
            evidence: 证据列表
            max_evidence: 最大证据数量
            
        Returns:
            截断后的证据列表
        """
        if evidence is None:
            return []
        
        if len(evidence) <= max_evidence:
            return evidence
        
        # 按权重排序，保留最高权重的证据
        sorted_evidence = sorted(
            evidence, 
            key=lambda x: x.get('weight', 0), 
            reverse=True
        )
        
        self.stats.truncated_samples += 1
        return sorted_evidence[:max_evidence]
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量（简单估算）
        
        Args:
            text: 输入文本
            
        Returns:
            估算的token数量
        """
        # 简单估算：中文约1.5字符/token，英文约4字符/token
        # 这里使用简单的字符计数作为估算
        return len(text) // 2
    
    def split_dataset(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        use_difficulty_split: bool = True,
        holdout_rare_labels: bool = True,
        rare_label_threshold: int = 15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        划分数据集（改进版：基于难度分层 + 留出少数类别）
        
        方案 A+C 组合策略：
        1. 基于难度分层：确保困难样本在测试集中占比更高
        2. 留出少数类别：选择部分少数类别完全留给测试集（Zero-shot评估）
        
        难度定义：
        - 简单样本：result_type == "direct" 且 confidence >= 0.9
        - 中等样本：result_type == "inferred" 且 confidence >= 0.8
        - 困难样本：confidence < 0.8 或 evidence_quality == "moderate/weak"
        
        Args:
            samples: 训练样本列表
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            use_difficulty_split: 是否使用基于难度的分层划分
            holdout_rare_labels: 是否留出少数类别给测试集
            rare_label_threshold: 少数类别阈值（样本数小于此值的类别）
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        random.seed(seed)
        
        if not use_difficulty_split:
            # 使用原始的简单划分
            return self._simple_split(samples, train_ratio, valid_ratio, test_ratio, seed)
        
        # ============================================================
        # 步骤1：分析样本难度和标签分布
        # ============================================================
        
        # 按标签分组
        label_groups: Dict[str, List[Dict]] = {}
        for sample in samples:
            label = self._extract_label(sample)
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        
        # 统计标签分布
        label_counts = [(label, len(group)) for label, group in label_groups.items()]
        label_counts.sort(key=lambda x: x[1], reverse=True)
        total_samples = len(samples)
        
        print(f"\n{'='*60}")
        print("数据集划分分析（方案 A+C：难度分层 + 留出少数类别）")
        print(f"{'='*60}")
        print(f"\n标签分布统计 (共{len(label_counts)}种):")
        for label, count in label_counts[:10]:
            print(f"  {label}: {count} ({count/total_samples*100:.1f}%)")
        if len(label_counts) > 10:
            print(f"  ... 还有 {len(label_counts)-10} 种标签")
        
        # ============================================================
        # 步骤2：识别少数类别（方案C）
        # ============================================================
        
        rare_labels = []
        normal_labels = []
        
        if holdout_rare_labels:
            for label, count in label_counts:
                if count <= rare_label_threshold and count >= 2:
                    # 样本数在 2-threshold 之间的类别作为少数类别
                    rare_labels.append(label)
                else:
                    normal_labels.append(label)
            
            # 最多留出 5 个少数类别
            if len(rare_labels) > 5:
                random.shuffle(rare_labels)
                rare_labels = rare_labels[:5]
            
            print(f"\n少数类别留出策略:")
            print(f"  阈值: 样本数 <= {rare_label_threshold}")
            print(f"  留出类别数: {len(rare_labels)}")
            if rare_labels:
                for label in rare_labels:
                    print(f"    - {label}: {len(label_groups[label])} 样本 (全部放入测试集)")
        else:
            normal_labels = [label for label, _ in label_counts]
        
        # ============================================================
        # 步骤3：对正常类别按难度分层（方案A）
        # ============================================================
        
        # 分析每个样本的难度
        easy_samples = []    # 简单样本
        medium_samples = []  # 中等样本
        hard_samples = []    # 困难样本
        
        for label in normal_labels:
            for sample in label_groups[label]:
                difficulty = self._assess_sample_difficulty(sample)
                if difficulty == 'easy':
                    easy_samples.append(sample)
                elif difficulty == 'medium':
                    medium_samples.append(sample)
                else:
                    hard_samples.append(sample)
        
        # 打乱各难度组
        random.shuffle(easy_samples)
        random.shuffle(medium_samples)
        random.shuffle(hard_samples)
        
        print(f"\n难度分布统计:")
        print(f"  简单样本: {len(easy_samples)} ({len(easy_samples)/total_samples*100:.1f}%)")
        print(f"  中等样本: {len(medium_samples)} ({len(medium_samples)/total_samples*100:.1f}%)")
        print(f"  困难样本: {len(hard_samples)} ({len(hard_samples)/total_samples*100:.1f}%)")
        
        # ============================================================
        # 步骤4：按难度分层划分数据集
        # ============================================================
        
        test_set = []
        valid_set = []
        train_set = []
        
        # 4.1 首先将少数类别全部放入测试集
        rare_samples_count = 0
        if holdout_rare_labels and rare_labels:
            for label in rare_labels:
                test_set.extend(label_groups[label])
                rare_samples_count += len(label_groups[label])
        
        # 4.2 计算剩余目标大小
        remaining_total = total_samples - rare_samples_count
        target_test_from_normal = int(total_samples * test_ratio) - rare_samples_count
        target_valid_size = int(total_samples * valid_ratio)
        
        # 确保目标值合理
        target_test_from_normal = max(0, target_test_from_normal)
        
        # 4.3 按难度分层采样测试集（动态比例策略）
        # 目标比例：困难 40%，中等 30%，简单 30%
        # 但如果某类样本不足，则动态调整
        if target_test_from_normal > 0:
            # 计算各难度可用于测试集的最大数量
            # 困难样本：最多取 60%（保留 40% 给训练集学习困难模式）
            hard_available = min(len(hard_samples) * 6 // 10, len(hard_samples))
            # 中等样本：最多取 40%
            medium_available = min(len(medium_samples) * 4 // 10, len(medium_samples))
            # 简单样本：最多取 25%
            easy_available = min(len(easy_samples) // 4, len(easy_samples))
            
            total_available = hard_available + medium_available + easy_available
            
            # 动态计算实际采样数量
            if total_available <= target_test_from_normal:
                # 可用样本不足，全部使用
                hard_for_test = hard_available
                medium_for_test = medium_available
                easy_for_test = easy_available
                print(f"\n⚠ 困难/中等样本不足，使用全部可用样本")
            else:
                # 按优先级分配：优先困难 > 中等 > 简单
                remaining_target = target_test_from_normal
                
                # 1. 先尽量满足困难样本目标（40%）
                hard_target = int(target_test_from_normal * 0.4)
                hard_for_test = min(hard_target, hard_available)
                remaining_target -= hard_for_test
                
                # 2. 如果困难样本不足目标，将差额分配给中等样本
                hard_shortfall = hard_target - hard_for_test
                medium_target = int(target_test_from_normal * 0.3) + hard_shortfall
                medium_for_test = min(medium_target, medium_available)
                remaining_target -= medium_for_test
                
                # 3. 剩余目标由简单样本填充
                medium_shortfall = medium_target - medium_for_test
                easy_target = remaining_target + medium_shortfall
                easy_for_test = min(easy_target, easy_available)
            
            # 执行采样
            test_set.extend(hard_samples[:hard_for_test])
            hard_samples = hard_samples[hard_for_test:]
            
            test_set.extend(medium_samples[:medium_for_test])
            medium_samples = medium_samples[medium_for_test:]
            
            test_set.extend(easy_samples[:easy_for_test])
            easy_samples = easy_samples[easy_for_test:]
            
            print(f"\n测试集采样详情:")
            print(f"  困难样本: {hard_for_test} (目标40%, 实际{hard_for_test/target_test_from_normal*100:.1f}%)")
            print(f"  中等样本: {medium_for_test} (目标30%, 实际{medium_for_test/target_test_from_normal*100:.1f}%)")
            print(f"  简单样本: {easy_for_test} (目标30%, 实际{easy_for_test/target_test_from_normal*100:.1f}%)")
        
        # 4.4 合并剩余样本
        remaining_samples = easy_samples + medium_samples + hard_samples
        random.shuffle(remaining_samples)
        
        # 4.5 从剩余样本中采样验证集
        actual_valid_size = min(target_valid_size, len(remaining_samples) // 5)
        valid_set = remaining_samples[:actual_valid_size]
        train_set = remaining_samples[actual_valid_size:]
        
        # 打乱各数据集
        random.shuffle(test_set)
        random.shuffle(valid_set)
        random.shuffle(train_set)
        
        # ============================================================
        # 步骤5：输出划分结果统计
        # ============================================================
        
        print(f"\n数据集划分结果:")
        print(f"  训练集: {len(train_set)} ({len(train_set)/total_samples*100:.1f}%)")
        print(f"  验证集: {len(valid_set)} ({len(valid_set)/total_samples*100:.1f}%)")
        print(f"  测试集: {len(test_set)} ({len(test_set)/total_samples*100:.1f}%)")
        
        # 分析测试集组成
        test_difficulty_stats = {'easy': 0, 'medium': 0, 'hard': 0, 'rare': rare_samples_count}
        for sample in test_set:
            if sample in label_groups.get(rare_labels[0], []) if rare_labels else False:
                continue  # 已统计
            difficulty = self._assess_sample_difficulty(sample)
            test_difficulty_stats[difficulty] = test_difficulty_stats.get(difficulty, 0) + 1
        
        print(f"\n测试集组成分析:")
        print(f"  少数类别样本: {rare_samples_count} ({rare_samples_count/len(test_set)*100:.1f}%)")
        print(f"  困难样本: {test_difficulty_stats.get('hard', 0)}")
        print(f"  中等样本: {test_difficulty_stats.get('medium', 0)}")
        print(f"  简单样本: {test_difficulty_stats.get('easy', 0)}")
        
        # 分析训练集标签覆盖
        train_labels = set(self._extract_label(s) for s in train_set)
        test_labels = set(self._extract_label(s) for s in test_set)
        unseen_in_train = test_labels - train_labels
        
        print(f"\n标签覆盖分析:")
        print(f"  训练集标签数: {len(train_labels)}")
        print(f"  测试集标签数: {len(test_labels)}")
        print(f"  测试集中训练集未见标签: {len(unseen_in_train)}")
        if unseen_in_train:
            print(f"    未见标签: {list(unseen_in_train)[:5]}{'...' if len(unseen_in_train) > 5 else ''}")
        
        print(f"{'='*60}\n")
        
        return train_set, valid_set, test_set
    
    def _assess_sample_difficulty(self, sample: Dict) -> str:
        """
        评估样本难度
        
        难度定义：
        - easy: result_type == "direct" 且 confidence >= 0.9
        - medium: result_type == "inferred" 且 confidence >= 0.8
        - hard: confidence < 0.8 或 evidence_quality == "moderate/weak"
        
        Args:
            sample: 训练样本
            
        Returns:
            难度等级: 'easy', 'medium', 'hard'
        """
        try:
            messages = sample.get('messages', [])
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    output = json.loads(content)
                    
                    result_type = output.get('result_type', 'inferred')
                    confidence = output.get('confidence', 0.5)
                    evidence_quality = output.get('evidence_quality', 'moderate')
                    
                    # 困难样本判断
                    if confidence < 0.8:
                        return 'hard'
                    if evidence_quality in ['weak', 'insufficient']:
                        return 'hard'
                    
                    # 简单样本判断
                    if result_type == 'direct' and confidence >= 0.9:
                        return 'easy'
                    
                    # 中等样本
                    return 'medium'
        except:
            pass
        
        return 'medium'  # 默认中等难度
    
    def _simple_split(
        self,
        samples: List[Dict],
        train_ratio: float,
        valid_ratio: float,
        test_ratio: float,
        seed: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """简单随机划分（原始方法）"""
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_end = int(total * train_ratio)
        valid_end = train_end + int(total * valid_ratio)
        
        train_set = shuffled[:train_end]
        valid_set = shuffled[train_end:valid_end]
        test_set = shuffled[valid_end:]
        
        print(f"\n数据集划分结果 (简单随机划分):")
        print(f"  训练集: {len(train_set)} ({len(train_set)/total*100:.1f}%)")
        print(f"  验证集: {len(valid_set)} ({len(valid_set)/total*100:.1f}%)")
        print(f"  测试集: {len(test_set)} ({len(test_set)/total*100:.1f}%)")
        
        return train_set, valid_set, test_set
    
    def _extract_label(self, sample: Dict) -> str:
        """从训练样本中提取标签值（根据当前模型类型）"""
        try:
            messages = sample.get('messages', [])
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    output = json.loads(content)
                    # 根据模型类型提取对应字段
                    if CURRENT_MODEL_TYPE == 'vendor':
                        return output.get('vendor') or 'null'
                    elif CURRENT_MODEL_TYPE == 'os':
                        return output.get('os') or 'null'
                    elif CURRENT_MODEL_TYPE == 'devicetype':
                        return output.get('devicetype') or 'null'
                    else:
                        return output.get('vendor') or 'null'
        except:
            pass
        return 'unknown'
    
    def save_dataset(self, samples: List[Dict], output_path: str) -> None:
        """
        保存数据集到JSONL文件
        
        Args:
            samples: 样本列表
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    def process(
        self, 
        input_path: Optional[str] = None, 
        output_dir: Optional[str] = None,
        include_dpo: bool = True,
        augment_ratio: float = 0.3
    ) -> Dict[str, str]:
        """
        执行完整的数据处理流程
        
        Args:
            input_path: 输入数据文件路径
            output_dir: 输出目录路径
            include_dpo: 是否生成 DPO 训练数据
            augment_ratio: 数据增强比例
            
        Returns:
            包含各数据集路径的字典
        """
        input_path = input_path or self.config.input_path
        output_dir = output_dir or self.config.output_dir
        
        print(f"开始处理数据: {input_path}")
        
        # 1. 加载数据
        records = self.load_data(input_path)
        print(f"加载记录数: {len(records)}")
        
        # 2. 清洗数据，分离正负样本
        positive_records, negative_records = self.clean_data(records)
        print(f"正样本数: {len(positive_records)}")
        print(f"负样本数: {len(negative_records)}")
        print(f"  - 因字段缺失过滤: {self.stats.filtered_by_status}")
        print(f"  - 因quality过滤: {self.stats.filtered_by_quality}")
        
        # 3. 数据增强
        if augment_ratio > 0:
            print(f"\n应用数据增强 (比例: {augment_ratio})...")
            positive_records = self.augment_dataset(positive_records, augment_ratio)
            print(f"增强后正样本数: {len(positive_records)}")
            print(f"  - 新增增强样本: {self.stats.augmented_samples}")
        
        # 4. 转换正样本为训练格式
        samples = []
        for record in positive_records:
            if 'teacher_output' in record:
                teacher_output = record['teacher_output']
                teacher_output['evidence'] = self.truncate_evidence(
                    teacher_output.get('evidence', [])
                )
            
            sample = self.convert_to_training_format(record)
            samples.append(sample)
        
        print(f"正样本训练样本数: {len(samples)}")
        
        # 4.1 转换负样本为训练格式（输出 null）
        negative_samples_converted = 0
        if negative_records:
            print(f"\n利用负样本 (转换为 null 输出)...")
            for record in negative_records:
                sample = self.convert_negative_to_training_format(record)
                samples.append(sample)
                negative_samples_converted += 1
            print(f"  - 负样本转换数: {negative_samples_converted}")
        
        print(f"总训练样本数: {len(samples)}")
        if self.stats.truncated_samples > 0:
            print(f"  - 截断的样本数: {self.stats.truncated_samples}")
        
        # 5. 划分数据集（使用改进的难度分层 + 留出少数类别策略）
        train_set, valid_set, test_set = self.split_dataset(
            samples,
            train_ratio=self.config.train_ratio,
            valid_ratio=self.config.valid_ratio,
            test_ratio=self.config.test_ratio,
            use_difficulty_split=True,      # 启用基于难度的分层划分
            holdout_rare_labels=True,       # 启用留出少数类别
            rare_label_threshold=15         # 样本数 <= 15 的类别作为少数类别
        )
        
        print(f"数据集划分:")
        print(f"  - 训练集: {len(train_set)}")
        print(f"  - 验证集: {len(valid_set)}")
        print(f"  - 测试集: {len(test_set)}")
        
        # 6. 保存数据集
        paths = {
            'train': os.path.join(output_dir, 'train.jsonl'),
            'valid': os.path.join(output_dir, 'valid.jsonl'),
            'test': os.path.join(output_dir, 'test.jsonl')
        }
        
        self.save_dataset(train_set, paths['train'])
        self.save_dataset(valid_set, paths['valid'])
        self.save_dataset(test_set, paths['test'])
        
        # 7. 生成 DPO 训练数据
        if include_dpo and negative_records:
            print(f"\n生成 DPO 训练数据...")
            # 使用原始正样本（非增强）来找相似样本
            original_positive = [r for r in positive_records if not r.get('_augmented')]
            dpo_pairs = self.create_dpo_pairs(original_positive, negative_records)
            
            if dpo_pairs:
                random.shuffle(dpo_pairs)
                dpo_train_size = int(len(dpo_pairs) * 0.9)
                dpo_train = dpo_pairs[:dpo_train_size]
                dpo_valid = dpo_pairs[dpo_train_size:]
                
                paths['dpo_train'] = os.path.join(output_dir, 'dpo_train.jsonl')
                paths['dpo_valid'] = os.path.join(output_dir, 'dpo_valid.jsonl')
                
                self.save_dataset(dpo_train, paths['dpo_train'])
                self.save_dataset(dpo_valid, paths['dpo_valid'])
                
                print(f"DPO 数据集:")
                print(f"  - DPO训练集: {len(dpo_train)}")
                print(f"  - DPO验证集: {len(dpo_valid)}")
        
        # 8. 保存负样本（用于错误分析）
        if negative_records:
            negative_samples = []
            for record in negative_records:
                if 'teacher_output' in record:
                    teacher_output = record['teacher_output']
                    teacher_output['evidence'] = self.truncate_evidence(
                        teacher_output.get('evidence', [])
                    )
                sample = self.convert_to_training_format(record)
                sample['is_negative'] = True
                # 添加错误分析信息
                sample['error_info'] = {
                    'original_vendor': record.get('teacher_output', {}).get('vendor'),
                    'original_confidence': record.get('teacher_output', {}).get('confidence'),
                    'evidence_quality': record.get('teacher_output', {}).get('evidence_quality')
                }
                negative_samples.append(sample)
            
            paths['negative'] = os.path.join(output_dir, 'negative.jsonl')
            self.save_dataset(negative_samples, paths['negative'])
            print(f"  - 负样本集(含错误分析): {len(negative_samples)}")
        
        print(f"\n数据集已保存到: {output_dir}")
        self._print_vendor_stats(train_set, valid_set, test_set)
        
        return paths
    
    def _print_vendor_stats(
        self, 
        train_set: List[Dict], 
        valid_set: List[Dict], 
        test_set: List[Dict]
    ) -> None:
        """打印厂商分布统计"""
        def count_vendors(samples):
            vendors = [self._extract_label(s) for s in samples]
            return Counter(vendors)
        
        train_vendors = count_vendors(train_set)
        valid_vendors = count_vendors(valid_set)
        test_vendors = count_vendors(test_set)
        
        print("\n标签分布统计:")
        all_vendors = set(train_vendors.keys()) | set(valid_vendors.keys()) | set(test_vendors.keys())
        
        for vendor in sorted(all_vendors):
            train_count = train_vendors.get(vendor, 0)
            valid_count = valid_vendors.get(vendor, 0)
            test_count = test_vendors.get(vendor, 0)
            total = train_count + valid_count + test_count
            print(f"  {vendor}: 训练={train_count}, 验证={valid_count}, 测试={test_count}, 总计={total}")


def balance_dataset_labels(
    samples: List[Dict],
    major_ratio: float = 0.85,
    task_type: str = "vendor",
    seed: int = 42
) -> List[Dict]:
    """
    平衡数据集中的标签分布
    
    将最多的标签占比调整为 major_ratio，其他标签占比为 1-major_ratio
    如果其他标签样本不足，则重复采样补足
    
    Args:
        samples: 原始样本列表
        major_ratio: 最多标签的目标占比 (default: 0.85)
        task_type: 任务类型 (vendor/os/devicetype)
        seed: 随机种子
        
    Returns:
        平衡后的样本列表
    """
    if not samples:
        return samples
    
    random.seed(seed)
    
    # 字段映射
    field_map = {
        "vendor": "vendor",
        "os": "os",
        "devicetype": "devicetype"
    }
    field_name = field_map.get(task_type, "vendor")
    
    # 提取标签
    def extract_label(sample: Dict) -> str:
        messages = sample.get('messages', [])
        if len(messages) < 3:
            return 'unknown'
        assistant_content = messages[-1].get('content', '')
        try:
            output = json.loads(assistant_content)
            if output is None:
                return 'null'
            if isinstance(output, dict):
                label = output.get(field_name)
                return str(label) if label else 'null'
            else:
                # 如果不是字典，直接返回字符串形式
                return str(output).strip() or 'null'
        except (json.JSONDecodeError, TypeError):
            # 不是 JSON，可能是简单格式（直接是标签）
            return assistant_content.strip() or 'null'
    
    # 按标签分组
    label_to_samples = {}
    for sample in samples:
        label = extract_label(sample)
        if label not in label_to_samples:
            label_to_samples[label] = []
        label_to_samples[label].append(sample)
    
    # 统计标签分布
    label_counts = {label: len(samps) for label, samps in label_to_samples.items()}
    total_samples = sum(label_counts.values())
    
    # 找出最多的标签
    major_label = max(label_counts, key=label_counts.get)
    major_count = label_counts[major_label]
    other_labels = [l for l in label_counts if l != major_label]
    other_count = sum(label_counts[l] for l in other_labels)
    
    print(f"  原始分布: {major_label}={major_count} ({100*major_count/total_samples:.1f}%), 其他={other_count} ({100*other_count/total_samples:.1f}%)")
    
    # 计算目标数量
    # 设 major_target = x, other_target = y
    # x / (x + y) = major_ratio
    # 保持总样本数不变: x + y = total_samples
    # 解得: x = total_samples * major_ratio
    major_target = int(total_samples * major_ratio)
    other_target = total_samples - major_target
    
    # 采样最多标签
    major_samples = label_to_samples[major_label]
    if len(major_samples) >= major_target:
        # 随机采样
        balanced_major = random.sample(major_samples, major_target)
    else:
        # 不足则重复采样
        balanced_major = major_samples.copy()
        while len(balanced_major) < major_target:
            balanced_major.append(random.choice(major_samples))
    
    # 采样其他标签
    other_samples = []
    for label in other_labels:
        other_samples.extend(label_to_samples[label])
    
    if len(other_samples) >= other_target:
        # 随机采样
        balanced_other = random.sample(other_samples, other_target)
    else:
        # 不足则重复采样
        balanced_other = other_samples.copy()
        while len(balanced_other) < other_target:
            balanced_other.append(random.choice(other_samples))
    
    # 合并并打乱
    balanced_samples = balanced_major + balanced_other
    random.shuffle(balanced_samples)
    
    # 统计平衡后的分布
    balanced_label_counts = {}
    for sample in balanced_samples:
        label = extract_label(sample)
        balanced_label_counts[label] = balanced_label_counts.get(label, 0) + 1
    
    new_major_count = balanced_label_counts.get(major_label, 0)
    new_other_count = sum(v for k, v in balanced_label_counts.items() if k != major_label)
    new_total = new_major_count + new_other_count
    
    print(f"  平衡后: {major_label}={new_major_count} ({100*new_major_count/new_total:.1f}%), 其他={new_other_count} ({100*new_other_count/new_total:.1f}%)")
    
    return balanced_samples


def balance_data_files(
    data_dir: str,
    task_type: str = "vendor",
    major_ratio: float = 0.85,
    train_ratio: float = None,
    seed: int = 42
) -> None:
    """
    平衡数据目录中的数据集标签分布
    
    Args:
        data_dir: 数据目录路径
        task_type: 任务类型 (vendor/os/devicetype)
        major_ratio: 验证集/测试集最多标签的目标占比 (default: 0.85)
        train_ratio: 训练集最多标签的目标占比 (default: None, 不平衡训练集)
        seed: 随机种子
    """
    print(f"\n平衡数据集标签分布: {data_dir}")
    print(f"  任务类型: {task_type}")
    if train_ratio is not None:
        print(f"  训练集: 最多标签 {train_ratio*100:.0f}%, 其他 {(1-train_ratio)*100:.0f}%")
    if major_ratio is not None:
        print(f"  验证/测试集: 最多标签 {major_ratio*100:.0f}%, 其他 {(1-major_ratio)*100:.0f}%")
    print("-" * 50)
    
    # 处理训练集
    if train_ratio is not None:
        for filename in ['simple_train.jsonl']:
            file_path = os.path.join(data_dir, filename)
            if not os.path.exists(file_path):
                print(f"  跳过 {filename}: 文件不存在")
                continue
            
            print(f"\n处理训练集 {filename}:")
            
            # 读取数据
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            
            if not samples:
                print(f"  跳过: 文件为空")
                continue
            
            print(f"  原始样本数: {len(samples)}")
            
            # 平衡标签
            balanced_samples = balance_dataset_labels(
                samples, 
                major_ratio=train_ratio,
                task_type=task_type,
                seed=seed
            )
            
            # 保存（覆盖原文件）
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in balanced_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"  ✓ 已保存: {file_path} ({len(balanced_samples)} 条)")
    
    # 处理验证集和测试集
    if major_ratio is not None:
        for filename in ['valid.jsonl', 'test.jsonl', 'simple_valid.jsonl']:
            file_path = os.path.join(data_dir, filename)
            if not os.path.exists(file_path):
                continue
            
            print(f"\n处理 {filename}:")
            
            # 读取数据
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            
            if not samples:
                print(f"  跳过: 文件为空")
                continue
            
            print(f"  原始样本数: {len(samples)}")
            
            # 平衡标签
            balanced_samples = balance_dataset_labels(
                samples, 
                major_ratio=major_ratio,
                task_type=task_type,
                seed=seed
            )
            
            # 保存（覆盖原文件）
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in balanced_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"  ✓ 已保存: {file_path} ({len(balanced_samples)} 条)")
    
    print("-" * 50)
    print("标签平衡完成")


def balance_data_files_v2(
    data_dir: str,
    task_type: str = "vendor",
    train_ratio: float = None,
    valid_ratio: float = None,
    test_ratio: float = None,
    seed: int = 42
) -> None:
    """
    平衡数据目录中的数据集标签分布（v2版本，支持分别配置训练/验证/测试集）
    
    Args:
        data_dir: 数据目录路径
        task_type: 任务类型 (vendor/os/devicetype)
        train_ratio: 训练集最多标签的目标占比 (default: None, 不平衡)
        valid_ratio: 验证集最多标签的目标占比 (default: None, 不平衡)
        test_ratio: 测试集最多标签的目标占比 (default: None, 不平衡)
        seed: 随机种子
    """
    print(f"\n平衡数据集标签分布: {data_dir}")
    print(f"  任务类型: {task_type}")
    if train_ratio is not None:
        print(f"  训练集: 最多标签 {train_ratio*100:.0f}%")
    if valid_ratio is not None:
        print(f"  验证集: 最多标签 {valid_ratio*100:.0f}%")
    if test_ratio is not None:
        print(f"  测试集: 最多标签 {test_ratio*100:.0f}%")
    print("-" * 50)
    
    # 文件和对应的平衡比例
    file_configs = [
        # (文件名列表, 平衡比例, 描述)
        (['simple_train.jsonl'], train_ratio, '训练集'),
        (['simple_valid.jsonl', 'valid.jsonl'], valid_ratio, '验证集'),
        (['test.jsonl'], test_ratio, '测试集'),
    ]
    
    for filenames, ratio, desc in file_configs:
        if ratio is None:
            continue
        
        for filename in filenames:
            file_path = os.path.join(data_dir, filename)
            if not os.path.exists(file_path):
                continue
            
            print(f"\n处理{desc} {filename}:")
            
            # 读取数据
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            
            if not samples:
                print(f"  跳过: 文件为空")
                continue
            
            print(f"  原始样本数: {len(samples)}")
            
            # 平衡标签
            balanced_samples = balance_dataset_labels(
                samples, 
                major_ratio=ratio,
                task_type=task_type,
                seed=seed
            )
            
            # 保存（覆盖原文件）
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in balanced_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"  ✓ 已保存: {file_path} ({len(balanced_samples)} 条)")
    
    print("-" * 50)
    print("标签平衡完成")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据处理工具')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--input', type=str, help='输入数据文件路径')
    parser.add_argument('--output', type=str, help='输出目录路径')
    parser.add_argument('--prompt-id', type=str, default='vendor', 
                        choices=['vendor', 'os', 'devicetype'],
                        help='模型类型/提示词ID (vendor/os/devicetype)')
    parser.add_argument('--prompt-file', type=str, default='prompt/student.json',
                        help='提示词文件路径')
    parser.add_argument('--no-dpo', action='store_true',
                        help='不生成 DPO 训练数据')
    parser.add_argument('--augment-ratio', type=float, default=0.3,
                        help='数据增强比例 (default: 0.3)')
    parser.add_argument('--no-augment', action='store_true',
                        help='禁用数据增强')
    
    args = parser.parse_args()
    
    # 加载提示词并设置模型类型
    load_prompt(args.prompt_file, args.prompt_id)
    print(f"当前模型类型: {CURRENT_MODEL_TYPE}")
    
    # 加载配置
    try:
        config = load_config(args.config)
        data_config = get_data_config(config)
    except FileNotFoundError:
        print(f"配置文件不存在，使用默认配置")
        data_config = DataConfig()
    
    # 覆盖命令行参数
    if args.input:
        data_config.input_path = args.input
    if args.output:
        data_config.output_dir = args.output
    
    # 执行处理
    processor = DataProcessor(data_config)
    augment_ratio = 0 if args.no_augment else args.augment_ratio
    processor.process(include_dpo=not args.no_dpo, augment_ratio=augment_ratio)


if __name__ == "__main__":
    main()
