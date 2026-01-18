#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理管道 V2 - 优化版本

主要优化：
1. 按 IP 分组划分数据集（防止数据泄露）
2. 输入文本脱敏（防止过拟合）
3. 严格样本筛选（提升训练纯度）
4. 结构化输入构造（提升特征质量）
5. 训练集重采样权重（解决长尾问题）
"""

import os
import re
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass, field


# ============================================================
# 配置
# ============================================================

@dataclass
class PipelineConfig:
    """数据处理管道配置"""
    task_type: str = "vendor"  # vendor, os, devicetype
    
    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    
    # 样本筛选
    strict_mode: bool = True           # 严格模式：只用 true_label
    min_confidence: float = 0.8        # 弱标注最低置信度
    allowed_evidence_quality: List[str] = field(
        default_factory=lambda: ['strong', 'moderate']
    )

    # 输入构造
    max_total_chars: int = 2500        # 总输入最大字符数
    max_service_chars: int = 1000      # 单个服务最大字符数
    head_chars: int = 800              # 截断时保留头部字符数
    tail_chars: int = 200              # 截断时保留尾部字符数
    
    # 验证集平衡（增加评估难度）
    val_balance_ratio: float = 0.4     # 验证集头部标签占比（默认40%，增加难度）
    
    # 脱敏
    sanitize_ip: bool = True
    sanitize_domain: bool = True
    sanitize_email: bool = True
    sanitize_sensitive: bool = True    # UUID, MAC, 长hex等
    
    # 输出
    output_dir: str = "./data"
    include_unknown: bool = True       # 是否包含 Unknown 类别


# ============================================================
# 标签白名单
# ============================================================

LABEL_WHITELIST = {
    'vendor': {
        'MikroTik', 'Cisco', 'Juniper', 'Huawei', 'Fortinet', 'Synology', 'QNAP',
        'Ubiquiti', 'HPE', 'Aruba', 'Zyxel', 'D-Link', 'Netgear', 'TP-Link', 'ASUS',
        'Nokia', 'Palo Alto', 'Check Point', 'Arista', 'Extreme', 'Ruckus',
        'Allied Telesis', 'Yamaha', 'NEC', 'Keenetic', 'Lancom', 'SonicWall',
        'Brocade', 'DD-WRT', 'ZTE', 'Linksys', 'Ruijie', 'IBM', 'Apple', 'Microsoft',
        'Google', 'Dell', 'Hewlett Packard Enterprise', 'NEC Platforms', 'Upvel'
    },
    'os': {
        'routeros', 'ios', 'ios-xe', 'ios-xr', 'nx-os', 'junos', 'vrp', 'fortios',
        'dsm', 'qts', 'unifi', 'arubaos', 'pan-os', 'linux', 'windows', 'freebsd',
        'openbsd', 'ilo', 'keenetic', 'edgeos', 'zyxel', 'univerge', 'openwrt',
        'airos', 'vyos', 'pfsense', 'opnsense'
    },
    'devicetype': {
        'router', 'server', 'firewall', 'camera', 'nas', 'printer', 'iot', 'appliance'
    }
}


# ============================================================
# 标签规范化映射
# ============================================================

LABEL_NORMALIZE_MAP = {
    'vendor': {
        'Juniper Networks': 'Juniper',
        'ZyXEL': 'Zyxel', 'ZYXEL': 'Zyxel', 'zyxel': 'Zyxel',
        'Ubiquiti Inc.': 'Ubiquiti', 'Ubiquiti Inc': 'Ubiquiti',
        'HPE': 'Hewlett Packard Enterprise', 'HP': 'Hewlett Packard Enterprise',
        'HPE/Aruba': 'Aruba',
        'Keenetic Ltd.': 'Keenetic', 'Keenetic Ltd': 'Keenetic',
        'YAMAHA': 'Yamaha',
        'NEC': 'NEC Platforms',
        'Netgear': 'Netgear', 'NETGEAR': 'Netgear',
        'UPVEL': 'Upvel',
    },
    'os': {
        'RouterOS': 'routeros', 'ROUTEROS': 'routeros',
        'IOS': 'ios', 'Cisco IOS': 'ios',
        'IOS-XE': 'ios-xe', 'IOS XE': 'ios-xe',
        'JunOS': 'junos', 'JUNOS': 'junos', 'Junos': 'junos',
        'Linux': 'linux', 'LINUX': 'linux',
        'Windows': 'windows', 'WINDOWS': 'windows',
        'iLO': 'ilo', 'ILO': 'ilo', 'HP iLO': 'ilo',
        'EdgeOS': 'edgeos',
        'OpenWrt': 'openwrt', 'OpenWRT': 'openwrt',
        'AirOS': 'airos',
    },
    'devicetype': {
        'Router': 'router', 'ROUTER': 'router',
        'Server': 'server', 'SERVER': 'server',
        'Firewall': 'firewall', 'FIREWALL': 'firewall',
        'Camera': 'camera', 'CAMERA': 'camera', 'IP Camera': 'camera',
        'NAS': 'nas',
        'Printer': 'printer', 'PRINTER': 'printer',
        'IoT': 'iot', 'IOT': 'iot',
        'Appliance': 'appliance', 'APPLIANCE': 'appliance',
        # 合并映射
        'switch': 'router', 'Switch': 'router',
        'gateway': 'router', 'Gateway': 'router',
        'access_point': 'router', 'AP': 'router',
        'web_server': 'server', 'mail_server': 'server',
        'security_appliance': 'firewall',
        'nvr': 'camera', 'dvr': 'camera',
        'voip': 'iot', 'phone': 'iot',
    }
}


# 服务优先级（指纹强度，数字越小优先级越高）
SERVICE_PRIORITY = {
    'ssh': 1,
    'snmp': 1,
    'http': 2,
    'https': 2,
    'ftp': 3,
    'telnet': 4,
    'smb': 3,
    'smtp': 4,
    'imap': 4,
    'pop3': 4,
}


# ============================================================
# 脱敏函数
# ============================================================

def sanitize_text(text: str, config: PipelineConfig = None) -> str:
    """
    脱敏处理
    - IP 地址 → <IP>
    - 域名 → <DOMAIN>
    - Email → <EMAIL>
    - UUID/MAC/长hex → <ID>
    """
    if config is None:
        config = PipelineConfig()
    
    if config.sanitize_ip:
        # IPv4
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', text)
        # IPv6 (简化版)
        text = re.sub(r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b', '<IP>', text)
    
    if config.sanitize_email:
        # Email（先于域名处理）
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '<EMAIL>', text)
    
    if config.sanitize_domain:
        # 域名（保留 TLD 类型信息）
        text = re.sub(
            r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.){1,}(com|net|org|io|cn|ru|de|uk|jp|kr|edu|gov)\b',
            r'<DOMAIN>',
            text,
            flags=re.IGNORECASE
        )
    
    if config.sanitize_sensitive:
        # UUID
        text = re.sub(
            r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
            '<UUID>',
            text
        )
        # MAC 地址
        text = re.sub(r'\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b', '<MAC>', text)
        # 长 hex 串 (32位以上)
        text = re.sub(r'\b[0-9a-fA-F]{32,}\b', '<HEX>', text)
        # 序列号模式 (字母数字混合长串)
        text = re.sub(r'\b[A-Z0-9]{12,}\b', '<SERIAL>', text)
    
    return text


# ============================================================
# 标签处理函数
# ============================================================

def normalize_label(label: str, task_type: str) -> str:
    """规范化标签"""
    if not label or str(label).lower() in ['null', 'none', '', 'unknown']:
        return 'Unknown'
    
    label = str(label).strip()
    mapping = LABEL_NORMALIZE_MAP.get(task_type, {})
    return mapping.get(label, label)


def validate_label(label: str, task_type: str) -> bool:
    """验证标签是否在白名单中"""
    if label == 'Unknown':
        return True
    whitelist = LABEL_WHITELIST.get(task_type, set())
    # 大小写不敏感匹配
    return label in whitelist or label.lower() in {l.lower() for l in whitelist}


def normalize_label_strict(label: str, task_type: str) -> str:
    """严格规范化：不在白名单的返回 Unknown"""
    normalized = normalize_label(label, task_type)
    if not validate_label(normalized, task_type):
        return 'Unknown'
    return normalized


# ============================================================
# 样本筛选函数
# ============================================================

def select_valid_samples(
    records: List[Dict],
    task_type: str,
    config: PipelineConfig = None
) -> List[Dict]:
    """
    选择可用样本（保证训练纯度）
    
    规则 A（强真值）：is_correct == "yes" 且 true_label 包含目标字段
    规则 B（弱标注）：teacher_output 满足高置信度条件（非严格模式）
    
    Returns:
        List of {'record': Dict, 'label': str, 'label_source': str, 'weak_label': bool}
    """
    if config is None:
        config = PipelineConfig(task_type=task_type)
    
    valid_samples = []
    stats = {'total': 0, 'rule_a': 0, 'rule_b': 0, 'skipped': 0}
    
    for r in records:
        stats['total'] += 1
        teacher = r.get('teacher_output', {})
        true_label = r.get('true_label', {})
        is_correct = r.get('is_correct', '')
        
        # 必须有 original_data
        if 'original_data' not in r:
            stats['skipped'] += 1
            continue
        
        # 规则 A：强真值
        if is_correct == 'yes' and true_label.get(task_type):
            label = normalize_label_strict(true_label[task_type], task_type)
            valid_samples.append({
                'record': r,
                'label': label,
                'label_source': 'true_label',
                'weak_label': False
            })
            stats['rule_a'] += 1
            continue
        
        # 规则 B：弱标注（严格模式下跳过）
        if config.strict_mode:
            stats['skipped'] += 1
            continue
        
        # 检查 teacher_output 条件
        confidence = teacher.get('confidence', 0)
        evidence_quality = teacher.get('evidence_quality', '')
        teacher_label = teacher.get(task_type)
        
        if (confidence >= config.min_confidence and
            evidence_quality in config.allowed_evidence_quality and
            teacher_label):
            label = normalize_label_strict(teacher_label, task_type)
            valid_samples.append({
                'record': r,
                'label': label,
                'label_source': 'teacher_output',
                'weak_label': True
            })
            stats['rule_b'] += 1
        else:
            stats['skipped'] += 1
    
    print(f"样本筛选统计:")
    print(f"  总样本: {stats['total']}")
    print(f"  规则A (强真值): {stats['rule_a']}")
    print(f"  规则B (弱标注): {stats['rule_b']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"  有效样本: {len(valid_samples)}")
    
    return valid_samples


# ============================================================
# 输入文本构造函数
# ============================================================

def get_service_priority(port_service: str) -> int:
    """获取服务优先级"""
    # 提取服务名（如 "http-80" → "http"）
    service = port_service.split('-')[0].lower()
    return SERVICE_PRIORITY.get(service, 5)


def format_input_structured(
    original_data: Dict,
    config: PipelineConfig = None
) -> str:
    """
    结构化输入构造
    - 按指纹强度排序
    - 分块截断（保留头尾）
    - 脱敏处理
    """
    if config is None:
        config = PipelineConfig()
    
    services = original_data.get('Services', {})
    if not services:
        return ""
    
    # 按优先级排序
    sorted_services = sorted(
        services.items(),
        key=lambda x: (get_service_priority(x[0]), x[0])
    )
    
    blocks = []
    total_chars = 0
    
    for port_service, data in sorted_services:
        banner = data.get('Banner', '')
        if not banner:
            continue
        
        # 脱敏
        banner = sanitize_text(banner, config)
        
        # 分块截断：保留头部 + 尾部
        if len(banner) > config.max_service_chars:
            head = banner[:config.head_chars]
            tail = banner[-config.tail_chars:]
            banner = f"{head}\n...[truncated]...\n{tail}"
        
        block = f"[{port_service}]\n{banner}"
        
        # 检查总长度
        if total_chars + len(block) + 2 > config.max_total_chars:
            # 如果还没有任何内容，至少加入一个截断的块
            if not blocks:
                remaining = config.max_total_chars - 50
                blocks.append(block[:remaining] + "\n...[truncated]...")
            break
        
        blocks.append(block)
        total_chars += len(block) + 2  # +2 for \n\n
    
    return '\n\n'.join(blocks)


# ============================================================
# 数据集划分函数（按 IP 分组）
# ============================================================

def split_by_ip_group(
    samples: List[Dict],
    config: PipelineConfig = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按 IP 分组划分数据集
    确保同一 IP 只出现在一个集合中
    
    Args:
        samples: 样本列表，每个元素包含 'record' 字段
        config: 配置
        
    Returns:
        (train_set, val_set, test_set)
    """
    if config is None:
        config = PipelineConfig()
    
    random.seed(config.seed)
    
    # 按 IP 分组
    ip_to_samples: Dict[str, List[Dict]] = {}
    for sample in samples:
        ip = sample['record'].get('ip', f"unknown_{id(sample)}")
        if ip not in ip_to_samples:
            ip_to_samples[ip] = []
        ip_to_samples[ip].append(sample)
    
    # 获取所有 IP 并打乱
    all_ips = list(ip_to_samples.keys())
    random.shuffle(all_ips)
    
    # 计算划分点
    n_ips = len(all_ips)
    train_end = int(n_ips * config.train_ratio)
    val_end = train_end + int(n_ips * config.val_ratio)
    
    # 划分 IP
    train_ips = set(all_ips[:train_end])
    val_ips = set(all_ips[train_end:val_end])
    test_ips = set(all_ips[val_end:])
    
    # 根据 IP 分配样本
    train_set = []
    val_set = []
    test_set = []
    
    for ip, ip_samples in ip_to_samples.items():
        if ip in train_ips:
            train_set.extend(ip_samples)
        elif ip in val_ips:
            val_set.extend(ip_samples)
        else:
            test_set.extend(ip_samples)
    
    # 打乱各集合
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    # 验证无泄露
    train_ip_set = {s['record'].get('ip') for s in train_set}
    val_ip_set = {s['record'].get('ip') for s in val_set}
    test_ip_set = {s['record'].get('ip') for s in test_set}
    
    assert len(train_ip_set & val_ip_set) == 0, "IP 泄露: train & val"
    assert len(train_ip_set & test_ip_set) == 0, "IP 泄露: train & test"
    assert len(val_ip_set & test_ip_set) == 0, "IP 泄露: val & test"
    
    print(f"\n数据集划分 (按 IP 分组):")
    print(f"  总 IP 数: {n_ips}")
    print(f"  训练集: {len(train_set)} 样本, {len(train_ips)} IPs")
    print(f"  验证集: {len(val_set)} 样本, {len(val_ips)} IPs")
    print(f"  测试集: {len(test_set)} 样本, {len(test_ips)} IPs")
    
    return train_set, val_set, test_set


# ============================================================
# 分层划分（保持标签分布）
# ============================================================

def split_by_ip_stratified(
    samples: List[Dict],
    config: PipelineConfig = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按 IP 分组 + 分层划分（尽量保持标签分布）
    
    策略：
    1. 按标签分组 IP
    2. 对每个标签的 IP 集合分别划分
    3. 合并结果
    """
    if config is None:
        config = PipelineConfig()
    
    random.seed(config.seed)
    
    # 按 IP 分组，并记录每个 IP 的主标签
    ip_to_samples: Dict[str, List[Dict]] = {}
    ip_to_label: Dict[str, str] = {}
    
    for sample in samples:
        ip = sample['record'].get('ip', f"unknown_{id(sample)}")
        label = sample['label']
        
        if ip not in ip_to_samples:
            ip_to_samples[ip] = []
            ip_to_label[ip] = label
        ip_to_samples[ip].append(sample)
    
    # 按标签分组 IP
    label_to_ips: Dict[str, List[str]] = {}
    for ip, label in ip_to_label.items():
        if label not in label_to_ips:
            label_to_ips[label] = []
        label_to_ips[label].append(ip)
    
    # 对每个标签的 IP 进行划分
    train_ips = set()
    val_ips = set()
    test_ips = set()
    
    for label, ips in label_to_ips.items():
        random.shuffle(ips)
        n = len(ips)
        
        if n == 1:
            # 只有一个 IP，放入训练集
            train_ips.add(ips[0])
        elif n == 2:
            # 两个 IP，一个训练一个测试
            train_ips.add(ips[0])
            test_ips.add(ips[1])
        else:
            # 正常划分
            train_end = max(1, int(n * config.train_ratio))
            val_end = train_end + max(1, int(n * config.val_ratio))
            
            train_ips.update(ips[:train_end])
            val_ips.update(ips[train_end:val_end])
            test_ips.update(ips[val_end:])
    
    # 根据 IP 分配样本
    train_set = []
    val_set = []
    test_set = []
    
    for ip, ip_samples in ip_to_samples.items():
        if ip in train_ips:
            train_set.extend(ip_samples)
        elif ip in val_ips:
            val_set.extend(ip_samples)
        else:
            test_set.extend(ip_samples)
    
    # 打乱
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    # 统计标签分布
    def count_labels(dataset):
        return Counter(s['label'] for s in dataset)
    
    print(f"\n数据集划分 (按 IP 分组 + 分层):")
    print(f"  训练集: {len(train_set)} 样本, {len(train_ips)} IPs")
    print(f"  验证集: {len(val_set)} 样本, {len(val_ips)} IPs")
    print(f"  测试集: {len(test_set)} 样本, {len(test_ips)} IPs")
    
    train_labels = count_labels(train_set)
    test_labels = count_labels(test_set)
    print(f"\n  训练集标签分布 (Top 5):")
    for label, count in train_labels.most_common(5):
        print(f"    {label}: {count} ({count/len(train_set)*100:.1f}%)")
    print(f"\n  测试集标签分布 (Top 5):")
    for label, count in test_labels.most_common(5):
        print(f"    {label}: {count} ({count/len(test_set)*100:.1f}%)")
    
    return train_set, val_set, test_set


# ============================================================
# 训练集重采样权重
# ============================================================

def compute_sample_weights(labels: List[str]) -> List[float]:
    """
    计算样本权重（逆频率平方根）
    w(label) = 1 / sqrt(freq)
    
    用于 WeightedRandomSampler
    """
    counter = Counter(labels)
    total = len(labels)
    
    weights = []
    for label in labels:
        freq = counter[label] / total
        weight = 1.0 / math.sqrt(freq)
        weights.append(weight)
    
    # 归一化到 [0, 1]
    max_weight = max(weights)
    weights = [w / max_weight for w in weights]
    
    return weights


def compute_class_weights(labels: List[str]) -> Dict[str, float]:
    """
    计算类别权重（用于损失函数加权）
    """
    counter = Counter(labels)
    total = len(labels)
    n_classes = len(counter)
    
    # 使用 sklearn 的 balanced 策略: n_samples / (n_classes * n_samples_per_class)
    class_weights = {}
    for label, count in counter.items():
        class_weights[label] = total / (n_classes * count)
    
    return class_weights


def balance_dataset_for_eval(
    samples: List[Dict],
    task_type: str,
    major_ratio: float = 0.4,
    seed: int = 42
) -> List[Dict]:
    """
    平衡数据集用于评估（增加评估难度）
    
    将头部标签占比降低到 major_ratio，增加小类样本比例
    
    Args:
        samples: 样本列表，每个元素包含 'label' 字段
        task_type: 任务类型
        major_ratio: 头部标签目标占比（默认 0.4 = 40%）
        seed: 随机种子
        
    Returns:
        平衡后的样本列表
    """
    if not samples or major_ratio >= 1.0:
        return samples
    
    random.seed(seed)
    
    # 按标签分组
    label_to_samples: Dict[str, List[Dict]] = {}
    for s in samples:
        label = s['label']
        if label not in label_to_samples:
            label_to_samples[label] = []
        label_to_samples[label].append(s)
    
    # 统计
    label_counts = {l: len(s) for l, s in label_to_samples.items()}
    total = sum(label_counts.values())
    
    # 找出头部标签（占比最高的）
    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])
    major_label = sorted_labels[0][0]
    major_count = sorted_labels[0][1]
    other_count = total - major_count
    
    current_major_ratio = major_count / total
    
    print(f"  原始分布: {major_label}={major_count} ({current_major_ratio*100:.1f}%), 其他={other_count}")
    
    if current_major_ratio <= major_ratio:
        print(f"  头部标签占比已 <= {major_ratio*100:.0f}%，无需平衡")
        return samples
    
    # 计算目标数量
    # 设头部标签目标数量为 x，其他标签保持不变
    # x / (x + other_count) = major_ratio
    # x = major_ratio * other_count / (1 - major_ratio)
    target_major = int(major_ratio * other_count / (1 - major_ratio))
    target_major = max(target_major, 1)  # 至少保留 1 个
    
    # 下采样头部标签
    major_samples = label_to_samples[major_label]
    if len(major_samples) > target_major:
        balanced_major = random.sample(major_samples, target_major)
    else:
        balanced_major = major_samples
    
    # 合并结果
    balanced_samples = balanced_major.copy()
    for label, samps in label_to_samples.items():
        if label != major_label:
            balanced_samples.extend(samps)
    
    random.shuffle(balanced_samples)
    
    # 统计平衡后
    new_major_count = len(balanced_major)
    new_total = len(balanced_samples)
    new_other_count = new_total - new_major_count
    
    print(f"  平衡后: {major_label}={new_major_count} ({new_major_count/new_total*100:.1f}%), 其他={new_other_count} ({new_other_count/new_total*100:.1f}%)")
    print(f"  样本数: {total} → {new_total}")
    
    return balanced_samples


# ============================================================
# 训练样本格式转换
# ============================================================

# System Prompts
SYSTEM_PROMPTS = {
    "vendor": """Identify the device vendor from network scan evidence.
Output ONLY the vendor name, nothing else.
If evidence is insufficient, output: Unknown""",

    "os": """Identify the operating system from network scan evidence.
Output ONLY the OS name, nothing else.
If evidence is insufficient, output: Unknown""",

    "devicetype": """Identify the device type from network scan evidence.
Output ONLY the device type, nothing else.
Valid types: router, server, firewall, camera, nas, printer, iot, appliance
If evidence is insufficient, output: Unknown"""
}


def convert_to_training_format(
    sample: Dict,
    task_type: str,
    config: PipelineConfig = None
) -> Dict:
    """
    转换为训练格式
    
    输出格式:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "标签"}
        ]
    }
    """
    if config is None:
        config = PipelineConfig(task_type=task_type)
    
    record = sample['record']
    label = sample['label']
    original_data = record.get('original_data', {})
    
    # 构造输入
    user_content = format_input_structured(original_data, config)
    
    # 获取 system prompt
    system_prompt = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS['vendor'])
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label}
        ]
    }


# ============================================================
# 主处理管道
# ============================================================

def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_jsonl(data: List[Dict], file_path: str):
    """保存 JSONL 文件"""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_pipeline(
    input_path: str,
    output_dir: str,
    task_type: str,
    config: PipelineConfig = None,
    use_stratified: bool = True
) -> Dict:
    """
    完整数据处理管道
    
    流程:
    1. 加载原始数据
    2. 样本筛选
    3. 按 IP 分组划分
    4. 转换为训练格式
    5. 计算采样权重
    6. 保存输出
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        task_type: 任务类型 (vendor/os/devicetype)
        config: 配置
        use_stratified: 是否使用分层划分
        
    Returns:
        处理统计信息
    """
    if config is None:
        config = PipelineConfig(task_type=task_type)
    config.task_type = task_type
    config.output_dir = output_dir
    
    print("=" * 60)
    print(f"数据处理管道 V2 - {task_type.upper()}")
    print("=" * 60)
    
    # 1. 加载数据
    print(f"\n[1/6] 加载数据: {input_path}")
    records = load_jsonl(input_path)
    print(f"  加载 {len(records)} 条记录")
    
    # 2. 样本筛选
    print(f"\n[2/6] 样本筛选 (strict={config.strict_mode})")
    valid_samples = select_valid_samples(records, task_type, config)
    
    if not valid_samples:
        print("  错误: 没有有效样本!")
        return {'error': 'no_valid_samples'}
    
    # 统计标签分布
    label_counts = Counter(s['label'] for s in valid_samples)
    print(f"\n  标签分布 ({len(label_counts)} 类):")
    for label, count in label_counts.most_common(10):
        print(f"    {label}: {count} ({count/len(valid_samples)*100:.1f}%)")
    if len(label_counts) > 10:
        print(f"    ... 还有 {len(label_counts)-10} 类")
    
    # 3. 数据集划分
    print(f"\n[3/6] 数据集划分 (按 IP 分组)")
    if use_stratified:
        train_set, val_set, test_set = split_by_ip_stratified(valid_samples, config)
    else:
        train_set, val_set, test_set = split_by_ip_group(valid_samples, config)
    
    # 3.5 平衡验证集（增加评估难度）
    val_balance_ratio = getattr(config, 'val_balance_ratio', 0.4)
    if val_balance_ratio and val_balance_ratio < 1.0:
        print(f"\n[3.5/6] 平衡验证集 (头部标签占比: {val_balance_ratio*100:.0f}%)")
        val_set = balance_dataset_for_eval(val_set, task_type, val_balance_ratio)
    
    # 4. 转换为训练格式
    print(f"\n[4/6] 转换为训练格式")
    train_data = [convert_to_training_format(s, task_type, config) for s in train_set]
    val_data = [convert_to_training_format(s, task_type, config) for s in val_set]
    test_data = [convert_to_training_format(s, task_type, config) for s in test_set]
    
    # 5. 计算采样权重
    print(f"\n[5/6] 计算采样权重")
    train_labels = [s['label'] for s in train_set]
    sample_weights = compute_sample_weights(train_labels)
    class_weights = compute_class_weights(train_labels)
    
    print(f"  类别权重 (Top 5):")
    sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
    for label, weight in sorted_weights[:5]:
        print(f"    {label}: {weight:.2f}")
    
    # 6. 保存输出
    print(f"\n[6/6] 保存输出到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练数据
    save_jsonl(train_data, os.path.join(output_dir, 'train.jsonl'))
    save_jsonl(val_data, os.path.join(output_dir, 'val.jsonl'))
    save_jsonl(test_data, os.path.join(output_dir, 'test.jsonl'))
    
    # 采样权重
    weights_path = os.path.join(output_dir, 'sample_weights.json')
    with open(weights_path, 'w') as f:
        json.dump({
            'sample_weights': sample_weights,
            'class_weights': class_weights
        }, f, indent=2)
    
    # 标签列表
    labels_path = os.path.join(output_dir, f'{task_type}_labels.txt')
    all_labels = sorted(set(train_labels))
    with open(labels_path, 'w') as f:
        for label in all_labels:
            f.write(label + '\n')
    
    # IP 划分记录
    split_info = {
        'train_ips': list(set(s['record'].get('ip') for s in train_set)),
        'val_ips': list(set(s['record'].get('ip') for s in val_set)),
        'test_ips': list(set(s['record'].get('ip') for s in test_set)),
    }
    split_path = os.path.join(output_dir, 'ip_split.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n  ✓ train.jsonl: {len(train_data)} 条")
    print(f"  ✓ val.jsonl: {len(val_data)} 条")
    print(f"  ✓ test.jsonl: {len(test_data)} 条")
    print(f"  ✓ sample_weights.json")
    print(f"  ✓ {task_type}_labels.txt: {len(all_labels)} 类")
    print(f"  ✓ ip_split.json")
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    
    return {
        'total_records': len(records),
        'valid_samples': len(valid_samples),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'n_labels': len(all_labels),
        'label_counts': dict(label_counts),
        'class_weights': class_weights
    }


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='数据处理管道 V2 - 优化版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理 vendor 数据
  python -m training.data_pipeline_v2 --task vendor --input input/vendor_model_train.jsonl
  
  # 处理 os 数据（非严格模式）
  python -m training.data_pipeline_v2 --task os --input input/os_model_train.jsonl --no-strict
  
  # 处理 devicetype 数据
  python -m training.data_pipeline_v2 --task devicetype --input input/devicetype_model_train.jsonl
        """
    )
    
    parser.add_argument('--task', type=str, required=True,
                        choices=['vendor', 'os', 'devicetype'],
                        help='任务类型')
    parser.add_argument('--input', type=str, required=True,
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: ./data/{task}_v2)')
    parser.add_argument('--no-strict', action='store_true',
                        help='非严格模式（允许使用弱标注）')
    parser.add_argument('--min-confidence', type=float, default=0.8,
                        help='弱标注最低置信度 (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例 (default: 0.8)')
    parser.add_argument('--no-sanitize', action='store_true',
                        help='禁用脱敏处理')
    
    args = parser.parse_args()
    
    # 构建配置
    config = PipelineConfig(
        task_type=args.task,
        strict_mode=not args.no_strict,
        min_confidence=args.min_confidence,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=(1 - args.train_ratio) / 2,
        test_ratio=(1 - args.train_ratio) / 2,
        sanitize_ip=not args.no_sanitize,
        sanitize_domain=not args.no_sanitize,
        sanitize_email=not args.no_sanitize,
        sanitize_sensitive=not args.no_sanitize,
    )
    
    # 输出目录
    output_dir = args.output or f'./data/{args.task}_v2'
    
    # 执行处理
    result = process_pipeline(
        input_path=args.input,
        output_dir=output_dir,
        task_type=args.task,
        config=config
    )
    
    print(f"\n处理结果: {json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
