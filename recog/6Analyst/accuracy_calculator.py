"""
准确率计算模块

读取输入数据中的标签和输出结果，计算准确率并生成报告
根据输入数据中实际存在的标签字段动态决定评估哪些指标

新准确率计算规则（2025-01-10更新）：
- 低置信度数据（<0.6）视为错误，纳入统计
- 预测值为空视为错误，纳入统计
- 输入标签为空但输出预测不为空视为错误，纳入统计
- 预测值与标签不匹配视为错误

支持多候选输出格式：[[name, conf], ...]
"""

# 置信度阈值，低于此值视为低置信度数据（错误）
MIN_CONFIDENCE_THRESHOLD = 0.6

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Union


def parse_multi_candidate(value: Union[str, List, None]) -> List[Tuple[str, float]]:
    """
    解析多候选格式的字段值
    
    支持的格式：
    1. 字符串: "router" -> [("router", 1.0)]
    2. 列表: [["router", 0.9], ["firewall", 0.7]] -> [("router", 0.9), ("firewall", 0.7)]
    3. None: -> []
    
    Returns:
        [(candidate_name, confidence), ...] 按置信度降序排列
    """
    if value is None:
        return []
    
    if isinstance(value, str):
        # 单值字符串，默认置信度 1.0
        return [(value, 1.0)] if value else []
    
    if isinstance(value, list):
        candidates = []
        for item in value:
            if isinstance(item, list) and len(item) >= 2:
                # [[name, conf], ...] 格式
                name, conf = item[0], item[1]
                if name and isinstance(conf, (int, float)):
                    candidates.append((str(name), float(conf)))
            elif isinstance(item, str):
                # ["name1", "name2"] 格式（兼容旧格式）
                candidates.append((item, 1.0))
        # 按置信度降序排列
        candidates.sort(key=lambda x: -x[1])
        return candidates
    
    return []


def get_primary_value(value: Union[str, List, None]) -> Optional[str]:
    """
    获取主要值（置信度最高的候选）
    
    用于兼容旧代码，返回第一个候选的名称
    """
    candidates = parse_multi_candidate(value)
    return candidates[0][0] if candidates else None


def detect_available_labels(input_path: str, sample_size: int = 100) -> Set[str]:
    """
    检测输入数据中存在哪些标签字段
    
    Args:
        input_path: 输入文件或目录路径
        sample_size: 采样数量（用于快速检测）
        
    Returns:
        存在的标签字段集合: {'vendor', 'os', 'device_type'}
    """
    available_labels = set()
    count = 0
    
    # 获取输入文件列表
    if os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        input_files = sorted(json_files + jsonl_files)
    elif os.path.isfile(input_path):
        input_files = [input_path]
    else:
        return available_labels
    
    for filepath in input_files:
        if count >= sample_size:
            break
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= sample_size:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    obj = json.loads(line)
                    ip = next(iter(obj.keys()))
                    data = obj[ip]
                    
                    # 检测 Vendor 字段
                    if 'Vendor' in data and data['Vendor']:
                        available_labels.add('vendor')
                    
                    # 检测 OS 字段（包含 OS 和 Device Type）
                    if 'OS' in data and isinstance(data['OS'], dict):
                        os_data = data['OS']
                        if os_data.get('OS'):
                            available_labels.add('os')
                        if os_data.get('Device Type'):
                            available_labels.add('device_type')
                    
                    count += 1
                        
                except (json.JSONDecodeError, StopIteration):
                    continue
    
    return available_labels


def load_input_labels(input_path: str, max_records: int = None, 
                      sampled_ips: Set[str] = None) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    从输入数据中加载标签信息，并返回存在的标签类型
    
    Args:
        input_path: 输入文件或目录路径
        max_records: 最大记录数（当sampled_ips为None时使用）
        sampled_ips: 采样的IP集合（如果提供，只加载这些IP的标签）
        
    Returns:
        ({ip: {'device_type': ..., 'os': ..., 'vendor': ...}, ...}, 存在的标签集合)
    """
    labels = {}
    available_labels = set()
    count = 0
    
    # 获取输入文件列表
    if os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        input_files = sorted(json_files + jsonl_files)
    elif os.path.isfile(input_path):
        input_files = [input_path]
    else:
        print(f"[ERROR] 输入路径不存在: {input_path}")
        return labels, available_labels
    
    for filepath in input_files:
        # 如果使用sampled_ips，不限制max_records
        if sampled_ips is None and max_records and count >= max_records:
            break
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 如果使用sampled_ips，不限制max_records
                if sampled_ips is None and max_records and count >= max_records:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    obj = json.loads(line)
                    ip = next(iter(obj.keys()))
                    
                    # 如果提供了sampled_ips，只加载这些IP的标签
                    if sampled_ips is not None and ip not in sampled_ips:
                        continue
                    
                    data = obj[ip]
                    
                    # 提取标签信息
                    label_info = {}
                    
                    # 从OS字段提取
                    if 'OS' in data and isinstance(data['OS'], dict):
                        os_data = data['OS']
                        if os_data.get('Device Type'):
                            label_info['device_type'] = os_data.get('Device Type', '')
                            available_labels.add('device_type')
                        if os_data.get('OS'):
                            label_info['os'] = os_data.get('OS', '')
                            available_labels.add('os')
                    
                    # 从Vendor字段提取（如果存在）
                    if 'Vendor' in data and data['Vendor']:
                        label_info['vendor'] = data['Vendor']
                        available_labels.add('vendor')
                    elif '_vendor' in data and data['_vendor']:
                        label_info['vendor'] = data['_vendor']
                        available_labels.add('vendor')
                    
                    # 只保存有标签的记录
                    if any(label_info.values()):
                        labels[ip] = label_info
                        count += 1
                        
                except (json.JSONDecodeError, StopIteration):
                    continue
    
    return labels, available_labels


def is_valid_prediction(result: Dict) -> bool:
    """
    判断预测结果是否有效（非空）
    
    如果 vendor、type、os 全部为空，则视为无效数据
    支持多候选格式
    """
    vendor = result.get('vendor')
    device_type = result.get('type')
    os_val = result.get('os')
    
    def is_empty(val):
        if val is None or val == '' or val == '-' or val == 'null' or val == 'unknown':
            return True
        # 支持多候选格式：空列表也视为空
        if isinstance(val, list) and len(val) == 0:
            return True
        return False
    
    return not (is_empty(vendor) and is_empty(device_type) and is_empty(os_val))


def load_output_results(output_path: str, skip_invalid: bool = False, 
                        skip_low_confidence: bool = False) -> Tuple[Dict[str, Dict], int, int]:
    """
    从输出结果中加载预测信息
    
    Args:
        output_path: 输出文件路径
        skip_invalid: 是否跳过无效数据（预测结果全为空）- 默认False，纳入统计
        skip_low_confidence: 是否跳过低置信度数据 - 默认False，纳入统计
        
    Returns:
        ({ip: {'vendor': ..., 'type': ..., 'os': ...}, ...}, 跳过的无效数据数量, 跳过的低置信度数量)
    """
    results = {}
    skipped_invalid = 0
    skipped_low_conf = 0
    
    if not os.path.exists(output_path):
        print(f"[ERROR] 输出文件不存在: {output_path}")
        return results, skipped_invalid, skipped_low_conf
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                ip = record.get('ip')
                if ip:
                    result_data = {
                        'vendor': record.get('vendor'),
                        'type': record.get('type'),
                        'os': record.get('os'),
                        'model': record.get('model'),
                        'confidence': record.get('confidence', 0),
                        'status': record.get('status')
                    }
                    
                    # 统计无效数据（但不跳过，纳入统计）
                    if not is_valid_prediction(result_data):
                        skipped_invalid += 1
                        if skip_invalid:
                            continue
                    
                    # 统计低置信度数据（但不跳过，纳入统计）
                    if result_data['confidence'] < MIN_CONFIDENCE_THRESHOLD:
                        skipped_low_conf += 1
                        if skip_low_confidence:
                            continue
                    
                    results[ip] = result_data
            except json.JSONDecodeError:
                continue
    
    return results, skipped_invalid, skipped_low_conf


def normalize_vendor(vendor: str) -> str:
    """标准化厂商名称用于比较"""
    if not vendor:
        return ''
    
    vendor = vendor.lower().strip()
    
    # 常见厂商别名映射
    aliases = {
        'mikrotik': ['mikrotik', 'routeros', 'routerboard'],
        'cisco': ['cisco', 'cisco systems', 'ciscosystems'],
        'juniper': ['juniper', 'juniper networks', 'junos'],
        'huawei': ['huawei', 'hw'],
        'hp': ['hp', 'hewlett packard', 'hewlett-packard', 'hpe', 'aruba'],
        'fortinet': ['fortinet', 'fortigate', 'fortios'],
        'ubiquiti': ['ubiquiti', 'ubnt', 'unifi', 'edgeos'],
    }
    
    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            if alias in vendor:
                return canonical
    
    return vendor


def compare_vendor(label_vendor: str, pred_vendor: str) -> bool:
    """比较厂商是否匹配"""
    if not label_vendor or not pred_vendor:
        return False
    
    norm_label = normalize_vendor(label_vendor)
    norm_pred = normalize_vendor(pred_vendor)
    
    if not norm_label or not norm_pred:
        return False
    
    # 完全匹配或互相包含
    return norm_label == norm_pred or norm_label in norm_pred or norm_pred in norm_label


def compare_device_type(label_type: str, pred_type: Union[str, List, None]) -> bool:
    """
    比较设备类型是否匹配
    
    支持：
    - 多值标签（用|分隔）
    - 多候选预测格式：[[name, conf], ...]
    - 任一候选匹配即视为正确
    """
    if not label_type:
        return False
    
    # 解析预测值（支持多候选格式）
    pred_candidates = parse_multi_candidate(pred_type)
    if not pred_candidates:
        return False
    
    # 设备类型别名映射（扩展版）
    type_aliases = {
        'router': ['router', 'routeros', 'routing', 'broadband router', 'broadband_router', 
                   'soho router', 'home router', 'gateway'],
        'switch': ['switch', 'switching', 'layer 2', 'layer 3'],
        'firewall': ['firewall', 'fw', 'security-misc', 'utm', 'security appliance'],
        'server': ['server', 'srv', 'general purpose', 'mail server', 'mail_server', 
                   'web server', 'ftp server', 'mail transfer agent'],
        'ap': ['ap', 'access point', 'wireless', 'wap', 'wifi'],
        'storage': ['storage', 'nas', 'san', 'storage-misc'],
        'printer': ['printer', 'print server'],
        'phone': ['phone', 'voip', 'ip phone', 'pbx'],
        'camera': ['camera', 'webcam', 'ip camera', 'surveillance'],
        'appliance': ['appliance', 'embedded system', 'iot'],
    }
    
    # 特殊规则：已移除（不再将 Broadband Router 预测为 firewall 视为正确）
    special_matches = []
    
    # 处理多值标签（用|分隔）
    label_types = [t.strip().lower() for t in label_type.split('|')]
    
    # 遍历所有预测候选
    for pred_name, pred_conf in pred_candidates:
        pred_lower = pred_name.lower().strip()
        
        for label_lower in label_types:
            # 直接匹配
            if label_lower == pred_lower or label_lower in pred_lower or pred_lower in label_lower:
                return True
            
            # 别名匹配
            for canonical, alias_list in type_aliases.items():
                label_match = any(alias in label_lower for alias in alias_list)
                pred_match = any(alias in pred_lower for alias in alias_list)
                if label_match and pred_match:
                    return True
            
            # 特殊规则匹配
            for label_patterns, pred_patterns in special_matches:
                if any(lp in label_lower for lp in label_patterns):
                    if any(pp in pred_lower for pp in pred_patterns):
                        return True
    
    return False


def compare_os(label_os: str, pred_os: Union[str, List, None]) -> bool:
    """
    比较操作系统是否匹配（增强模糊匹配）
    
    支持：
    - 多候选预测格式：[[name, conf], ...]
    - 任一候选匹配即视为正确
    
    匹配规则：
    1. 提取OS核心名称（去除版本号）进行匹配
    2. 使用别名映射进行归一化匹配
    3. 只要核心OS类型匹配即可，不要求版本号完全一致
    """
    if not label_os:
        return False
    
    # 解析预测值（支持多候选格式）
    pred_candidates = parse_multi_candidate(pred_os)
    if not pred_candidates:
        return False
    
    label_lower = label_os.lower().strip()
    
    # OS 核心名称提取（去除版本号）
    os_core_patterns = [
        'keeneticos', 'keenetic',
        'routeros', 'mikrotik',
        'cisco ios xr', 'cisco ios xe', 'cisco ios', 'ios',
        'junos', 'juniper',
        'fortios', 'fortigate', 'fortinet',
        'panos', 'pan-os', 'palo alto',
        'linux', 'ubuntu', 'debian', 'centos', 'redhat', 'rhel',
        'windows server', 'windows',
        'freebsd', 'openbsd', 'netbsd',
        'vyos', 'vyatta',
        'openwrt', 'dd-wrt',
        'airos', 'ubiquiti',
        'edgeos', 'edgemax',
        'arubaos', 'aruba',
        'comware', 'h3c',
        'huawei vrp', 'vrp', 'huawei',
        'nxos', 'nx-os',
        'eos', 'arista',
        'sros', 'nokia',
        'exos', 'extreme',
        'dnos', 'dell',
        'sonicos', 'sonicwall',
        'screenos', 'netscreen',
        'asyncos', 'ironport',
        'big-ip', 'f5',
        'netscaler', 'citrix',
        'checkpoint', 'gaia',
        'sophos', 'cyberoam',
        'pfsense', 'opnsense',
        'univergeix', 'univerge', 'nec',
        'yamaha',
        'zyxel', 'zynos',
        'draytek',
        'netgear',
        'tp-link', 'tplink',
        'd-link', 'dlink',
        'asus', 'asuswrt',
        'synology', 'dsm',
        'qnap', 'qts',
        'xfinity', 'comcast',
        'boss', 'sagem',
        'powermta',
    ]
    
    def extract_os_core(os_str: str) -> str:
        os_lower = os_str.lower()
        for pattern in os_core_patterns:
            if pattern in os_lower:
                return pattern
        import re
        match = re.match(r'^([a-zA-Z][a-zA-Z\-]*)', os_str)
        if match:
            return match.group(1).lower()
        return os_lower
    
    # OS 别名映射
    os_aliases = {
        'mikrotik': ['routeros', 'mikrotik routeros', 'mikrotik', 'ros'],
        'nec': ['univergeix', 'univerge ix', 'univerge', 'nec', 'ix series', 'ix software'],
        'zyxel': ['zynos', 'zld', 'zywall', 'zyxel', 'zywall os'],
        'allied': ['alliedware', 'allied telesyn', 'allied telesis', 'at-', 'atr-'],
        'yamaha': ['yamaha', 'rt series', 'rtx', 'yamaha embedded'],
        'cisco': ['ios', 'cisco ios', 'cisco ios xe', 'cisco ios xr', 'nxos', 'nx-os', 'cisco'],
        'juniper': ['junos', 'juniper junos', 'juniper'],
        'keenetic': ['keeneticos', 'keenetic'],
        'linux': ['linux', 'ubuntu', 'debian', 'centos', 'redhat', 'rhel', 'fedora'],
        'windows': ['windows', 'windows server', 'win', 'microsoft'],
        'freebsd': ['freebsd', 'bsd'],
        'fortinet': ['fortios', 'fortigate', 'fortinet'],
        'paloalto': ['panos', 'pan-os', 'palo alto'],
        'huawei': ['huawei vrp', 'vrp', 'huawei', 'comware', 'h3c'],
        'ubiquiti': ['airos', 'ubiquiti', 'edgeos', 'edgemax', 'unifi', 'ubnt'],
        'arista': ['eos', 'arista eos', 'arista'],
        'nokia': ['sros', 'nokia sr os', 'nokia', 'timos'],
        'vyos': ['vyos', 'vyatta'],
        'aruba': ['arubaos', 'aruba'],
        'extreme': ['exos', 'extremexos', 'extreme'],
        'dell': ['dnos', 'ftos', 'dell', 'force10'],
        'sonicwall': ['sonicos', 'sonicwall'],
        'checkpoint': ['checkpoint', 'gaia', 'check point'],
        'f5': ['big-ip', 'f5', 'bigip'],
        'citrix': ['netscaler', 'citrix'],
        'comcast': ['xfinity', 'comcast'],
        'netgear': ['netgear', 'netgear embedded'],
        'draytek': ['draytek', 'vigor'],
        'tplink': ['tp-link', 'tplink'],
        'dlink': ['d-link', 'dlink'],
    }
    
    label_core = extract_os_core(label_lower)
    
    # 遍历所有预测候选
    for pred_name, pred_conf in pred_candidates:
        pred_lower = pred_name.lower().strip()
        pred_core = extract_os_core(pred_lower)
        
        # 核心名称直接匹配
        if label_core == pred_core:
            return True
        if label_core in pred_core or pred_core in label_core:
            return True
        
        # 别名匹配
        for canonical, alias_list in os_aliases.items():
            label_match = any(alias in label_lower for alias in alias_list)
            pred_match = any(alias in pred_lower for alias in alias_list)
            if label_match and pred_match:
                return True
        
        # 简单包含匹配
        if label_lower in pred_lower or pred_lower in label_lower:
            return True
    
    return False


def calculate_accuracy(labels: Dict[str, Dict], results: Dict[str, Dict], 
                       available_labels: Set[str] = None,
                       total_records: int = None) -> Dict:
    """
    计算准确率，根据输入数据中存在的标签动态评估
    
    新准确率计算规则（2025-01-10更新）：
    - 低置信度数据（<0.6）视为错误，纳入统计
    - 预测值为空视为错误，纳入统计
    - 输入标签为空但输出预测不为空视为错误，纳入统计
    - 预测值与标签不匹配视为错误
    
    覆盖率和正确率计算（2025-01-17新增）：
    - 覆盖率 = 输出结果中预测不为null的数量 / 总记录数
    - 正确率 = 正确的结果 / 输出结果中预测不为null的数量
    
    Args:
        labels: 标签数据
        results: 预测结果
        available_labels: 输入数据中存在的标签类型集合
        total_records: 总记录数（用于计算覆盖率）
        
    Returns:
        {
            'total_labels': 有标签的记录数,
            'total_results': 有结果的记录数,
            'matched_count': 匹配的记录数,
            'available_labels': 评估的标签类型列表,
            'vendor_accuracy': {'correct': n, 'total': n, 'rate': 0.xx, 'errors': {...}},
            'type_accuracy': {'correct': n, 'total': n, 'rate': 0.xx, 'errors': {...}},
            'os_accuracy': {'correct': n, 'total': n, 'rate': 0.xx, 'errors': {...}},
            'coverage': {'vendor': 0.xx, 'os': 0.xx, 'device_type': 0.xx},  # 覆盖率
            'precision': {'vendor': 0.xx, 'os': 0.xx, 'device_type': 0.xx},  # 正确率
            'details': [详细匹配信息列表]
        }
    """
    # 默认评估所有标签
    if available_labels is None:
        available_labels = {'vendor', 'device_type', 'os'}
    
    stats = {
        'total_labels': len(labels),
        'total_results': len(results),
        'matched_count': 0,
        'available_labels': list(available_labels),
        'total_records': total_records or len(results),  # 总记录数
        'coverage': {},  # 覆盖率
        'precision': {},  # 正确率
        'details': []
    }
    
    # 辅助函数：判断值是否为空
    def is_empty(val):
        if val is None or val == '' or val == '-' or val == 'null' or val == 'unknown':
            return True
        if isinstance(val, list) and len(val) == 0:
            return True
        return False
    
    # 根据可用标签初始化准确率统计（包含错误分类）
    if 'vendor' in available_labels:
        stats['vendor_accuracy'] = {
            'correct': 0, 'total': 0, 'rate': 0.0,
            'labeled_count': 0,  # 输入数据中有标签的记录数
            'predicted_count': 0,  # 输出结果中有预测的记录数（用于计算覆盖率和正确率）
            'errors': {
                'low_confidence': 0,      # 低置信度错误
                'empty_prediction': 0,    # 预测为空错误
                'label_empty_pred_not': 0, # 标签为空但预测不为空
                'mismatch': 0             # 不匹配错误
            }
        }
    if 'device_type' in available_labels:
        stats['type_accuracy'] = {
            'correct': 0, 'total': 0, 'rate': 0.0,
            'labeled_count': 0,
            'predicted_count': 0,
            'errors': {
                'low_confidence': 0,
                'empty_prediction': 0,
                'label_empty_pred_not': 0,
                'mismatch': 0
            }
        }
    if 'os' in available_labels:
        stats['os_accuracy'] = {
            'correct': 0, 'total': 0, 'rate': 0.0,
            'labeled_count': 0,
            'predicted_count': 0,
            'errors': {
                'low_confidence': 0,
                'empty_prediction': 0,
                'label_empty_pred_not': 0,
                'mismatch': 0
            }
        }
    
    # 找出同时有标签和结果的IP
    common_ips = set(labels.keys()) & set(results.keys())
    stats['matched_count'] = len(common_ips)
    
    for ip in common_ips:
        label = labels[ip]
        result = results[ip]
        confidence = result.get('confidence', 0)
        is_low_conf = confidence < MIN_CONFIDENCE_THRESHOLD
        
        detail = {
            'ip': ip,
            'confidence': confidence,
            'is_low_confidence': is_low_conf,
        }
        
        # 厂商匹配（仅当输入数据有vendor标签时评估）
        if 'vendor' in available_labels:
            label_vendor = label.get('vendor', '')
            pred_vendor = result.get('vendor', '')
            
            detail['label_vendor'] = label_vendor
            detail['pred_vendor'] = pred_vendor
            detail['vendor_match'] = False
            detail['vendor_error_type'] = None
            
            # 统计输入数据中有标签的记录数
            if not is_empty(label_vendor):
                stats['vendor_accuracy']['labeled_count'] += 1
            
            # 统计输出结果中有预测的记录数（用于计算覆盖率）
            if not is_empty(pred_vendor):
                stats['vendor_accuracy']['predicted_count'] += 1
            
            # 纳入统计（所有有标签或有预测的记录都计入总数）
            if label_vendor or pred_vendor:
                stats['vendor_accuracy']['total'] += 1
                
                # 判断是否正确（调整判断顺序：先判断预测是否为空，再判断置信度）
                if is_empty(pred_vendor):
                    # 预测为空 -> 错误
                    detail['vendor_error_type'] = 'empty_prediction'
                    stats['vendor_accuracy']['errors']['empty_prediction'] += 1
                elif is_low_conf:
                    # 低置信度（但有预测）-> 错误
                    detail['vendor_error_type'] = 'low_confidence'
                    stats['vendor_accuracy']['errors']['low_confidence'] += 1
                elif is_empty(label_vendor) and not is_empty(pred_vendor):
                    # 标签为空但预测不为空 -> 错误
                    detail['vendor_error_type'] = 'label_empty_pred_not'
                    stats['vendor_accuracy']['errors']['label_empty_pred_not'] += 1
                elif label_vendor and pred_vendor and compare_vendor(label_vendor, pred_vendor):
                    # 匹配成功 -> 正确
                    stats['vendor_accuracy']['correct'] += 1
                    detail['vendor_match'] = True
                else:
                    # 不匹配 -> 错误
                    detail['vendor_error_type'] = 'mismatch'
                    stats['vendor_accuracy']['errors']['mismatch'] += 1
        
        # 设备类型匹配（仅当输入数据有device_type标签时评估）
        if 'device_type' in available_labels:
            label_type = label.get('device_type', '')
            pred_type = result.get('type', '')
            
            detail['label_type'] = label_type
            detail['pred_type'] = pred_type
            detail['type_match'] = False
            detail['type_error_type'] = None
            
            # 统计有标签的记录数
            if not is_empty(label_type):
                stats['type_accuracy']['labeled_count'] += 1
            
            # 统计有预测的记录数
            if not is_empty(pred_type):
                stats['type_accuracy']['predicted_count'] += 1
            
            # 纳入统计
            if label_type or pred_type:
                stats['type_accuracy']['total'] += 1
                
                # 判断是否正确（调整判断顺序：先判断预测是否为空，再判断置信度）
                if is_empty(pred_type):
                    detail['type_error_type'] = 'empty_prediction'
                    stats['type_accuracy']['errors']['empty_prediction'] += 1
                elif is_low_conf:
                    detail['type_error_type'] = 'low_confidence'
                    stats['type_accuracy']['errors']['low_confidence'] += 1
                elif is_empty(label_type) and not is_empty(pred_type):
                    detail['type_error_type'] = 'label_empty_pred_not'
                    stats['type_accuracy']['errors']['label_empty_pred_not'] += 1
                elif label_type and pred_type and compare_device_type(label_type, pred_type):
                    stats['type_accuracy']['correct'] += 1
                    detail['type_match'] = True
                else:
                    detail['type_error_type'] = 'mismatch'
                    stats['type_accuracy']['errors']['mismatch'] += 1
        
        # OS匹配（仅当输入数据有os标签时评估）
        if 'os' in available_labels:
            label_os = label.get('os', '')
            pred_os = result.get('os', '')
            
            detail['label_os'] = label_os
            detail['pred_os'] = pred_os
            detail['os_match'] = False
            detail['os_error_type'] = None
            
            # 统计有标签的记录数
            if not is_empty(label_os):
                stats['os_accuracy']['labeled_count'] += 1
            
            # 统计有预测的记录数
            if not is_empty(pred_os):
                stats['os_accuracy']['predicted_count'] += 1
            
            # 纳入统计
            if label_os or pred_os:
                stats['os_accuracy']['total'] += 1
                
                # 判断是否正确（调整判断顺序：先判断预测是否为空，再判断置信度）
                if is_empty(pred_os):
                    detail['os_error_type'] = 'empty_prediction'
                    stats['os_accuracy']['errors']['empty_prediction'] += 1
                elif is_low_conf:
                    detail['os_error_type'] = 'low_confidence'
                    stats['os_accuracy']['errors']['low_confidence'] += 1
                elif is_empty(label_os) and not is_empty(pred_os):
                    detail['os_error_type'] = 'label_empty_pred_not'
                    stats['os_accuracy']['errors']['label_empty_pred_not'] += 1
                elif label_os and pred_os and compare_os(label_os, pred_os):
                    stats['os_accuracy']['correct'] += 1
                    detail['os_match'] = True
                else:
                    detail['os_error_type'] = 'mismatch'
                    stats['os_accuracy']['errors']['mismatch'] += 1
        
        stats['details'].append(detail)
    
    # 计算准确率
    if 'vendor' in available_labels and stats['vendor_accuracy']['total'] > 0:
        stats['vendor_accuracy']['rate'] = stats['vendor_accuracy']['correct'] / stats['vendor_accuracy']['total']
    
    if 'device_type' in available_labels and stats['type_accuracy']['total'] > 0:
        stats['type_accuracy']['rate'] = stats['type_accuracy']['correct'] / stats['type_accuracy']['total']
    
    if 'os' in available_labels and stats['os_accuracy']['total'] > 0:
        stats['os_accuracy']['rate'] = stats['os_accuracy']['correct'] / stats['os_accuracy']['total']
    
    # 计算覆盖率和正确率
    total_records = stats['total_records']
    
    if 'vendor' in available_labels:
        predicted_count = stats['vendor_accuracy']['predicted_count']
        correct_count = stats['vendor_accuracy']['correct']
        # 覆盖率 = 输出结果中有预测的记录数 / 总记录数
        stats['coverage']['vendor'] = predicted_count / total_records if total_records > 0 else 0.0
        # 正确率 = 正确的结果 / 输出结果中有预测的记录数
        stats['precision']['vendor'] = correct_count / predicted_count if predicted_count > 0 else 0.0
    
    if 'device_type' in available_labels:
        predicted_count = stats['type_accuracy']['predicted_count']
        correct_count = stats['type_accuracy']['correct']
        stats['coverage']['device_type'] = predicted_count / total_records if total_records > 0 else 0.0
        stats['precision']['device_type'] = correct_count / predicted_count if predicted_count > 0 else 0.0
    
    if 'os' in available_labels:
        predicted_count = stats['os_accuracy']['predicted_count']
        correct_count = stats['os_accuracy']['correct']
        stats['coverage']['os'] = predicted_count / total_records if total_records > 0 else 0.0
        stats['precision']['os'] = correct_count / predicted_count if predicted_count > 0 else 0.0
    
    return stats


def generate_accuracy_report(stats: Dict, output_path: str = 'accuracy_report.md') -> str:
    """
    生成准确率报告，根据可用标签动态生成内容
    
    新准确率计算规则（2025-01-10更新）：
    - 低置信度数据（<0.6）视为错误，纳入统计
    - 预测值为空视为错误，纳入统计
    - 输入标签为空但输出预测不为空视为错误，纳入统计
    - 预测值与标签不匹配视为错误
    
    覆盖率和正确率计算（2025-01-17新增）：
    - 覆盖率 = 标签不为null的数量 / 总记录数
    - 正确率 = 正确的结果 / 标签不为null的数量
    
    Args:
        stats: 准确率统计结果
        output_path: 报告输出路径
        
    Returns:
        报告文件路径
    """
    available_labels = set(stats.get('available_labels', ['vendor', 'device_type', 'os']))
    total_records = stats.get('total_records', stats.get('total_results', 0))
    
    report_lines = [
        "# 准确率评估报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 概览",
        "",
        f"- 总记录数: {total_records}",
        f"- 有标签的记录数: {stats['total_labels']}",
        f"- 有结果的记录数: {stats['total_results']}",
        f"- 匹配的记录数: {stats['matched_count']}",
        f"- 评估的标签类型: {', '.join(available_labels)}",
        "",
        "## 指标说明",
        "",
        "- **准确率**: 正确预测数 / 总预测数（包含标签为空但预测不为空的情况）",
        "- **覆盖率**: 输出结果中有预测的记录数 / 总记录数",
        "- **正确率**: 正确预测数 / 输出结果中有预测的记录数",
        "",
        "## 准确率计算规则",
        "",
        "以下情况视为错误：",
        f"1. 低置信度（<{MIN_CONFIDENCE_THRESHOLD}）",
        "2. 预测值为空",
        "3. 输入标签为空但输出预测不为空",
        "4. 预测值与标签不匹配",
        "",
        "## 准确率统计",
        "",
    ]
    
    # 辅助函数：生成错误分类统计
    def format_error_stats(accuracy_data: Dict, label_name: str) -> List[str]:
        lines = []
        errors = accuracy_data.get('errors', {})
        total_errors = accuracy_data['total'] - accuracy_data['correct']
        if total_errors > 0:
            lines.append(f"  错误分类统计（共{total_errors}条）：")
            if errors.get('low_confidence', 0) > 0:
                lines.append(f"    - 低置信度: {errors['low_confidence']}")
            if errors.get('empty_prediction', 0) > 0:
                lines.append(f"    - 预测为空: {errors['empty_prediction']}")
            if errors.get('label_empty_pred_not', 0) > 0:
                lines.append(f"    - 标签空但预测不空: {errors['label_empty_pred_not']}")
            if errors.get('mismatch', 0) > 0:
                lines.append(f"    - 不匹配: {errors['mismatch']}")
        return lines
    
    # 根据可用标签添加准确率统计
    if 'vendor' in available_labels and 'vendor_accuracy' in stats:
        predicted_count = stats['vendor_accuracy'].get('predicted_count', 0)
        coverage = stats.get('coverage', {}).get('vendor', 0)
        precision = stats.get('precision', {}).get('vendor', 0)
        report_lines.extend([
            "### 厂商识别 (Vendor)",
            "",
            f"- 正确: {stats['vendor_accuracy']['correct']}",
            f"- 总数: {stats['vendor_accuracy']['total']}",
            f"- 准确率: **{stats['vendor_accuracy']['rate']*100:.2f}%**",
            f"- 有预测记录数: {predicted_count}",
            f"- 覆盖率: **{coverage*100:.2f}%** ({predicted_count}/{total_records})",
            f"- 正确率: **{precision*100:.2f}%** ({stats['vendor_accuracy']['correct']}/{predicted_count})",
        ])
        report_lines.extend(format_error_stats(stats['vendor_accuracy'], 'vendor'))
        report_lines.append("")
    
    if 'os' in available_labels and 'os_accuracy' in stats:
        predicted_count = stats['os_accuracy'].get('predicted_count', 0)
        coverage = stats.get('coverage', {}).get('os', 0)
        precision = stats.get('precision', {}).get('os', 0)
        report_lines.extend([
            "### 操作系统识别 (OS)",
            "",
            f"- 正确: {stats['os_accuracy']['correct']}",
            f"- 总数: {stats['os_accuracy']['total']}",
            f"- 准确率: **{stats['os_accuracy']['rate']*100:.2f}%**",
            f"- 有预测记录数: {predicted_count}",
            f"- 覆盖率: **{coverage*100:.2f}%** ({predicted_count}/{total_records})",
            f"- 正确率: **{precision*100:.2f}%** ({stats['os_accuracy']['correct']}/{predicted_count})",
        ])
        report_lines.extend(format_error_stats(stats['os_accuracy'], 'os'))
        report_lines.append("")
    
    if 'device_type' in available_labels and 'type_accuracy' in stats:
        predicted_count = stats['type_accuracy'].get('predicted_count', 0)
        coverage = stats.get('coverage', {}).get('device_type', 0)
        precision = stats.get('precision', {}).get('device_type', 0)
        report_lines.extend([
            "### 设备类型识别 (Device Type)",
            "",
            f"- 正确: {stats['type_accuracy']['correct']}",
            f"- 总数: {stats['type_accuracy']['total']}",
            f"- 准确率: **{stats['type_accuracy']['rate']*100:.2f}%**",
            f"- 有预测记录数: {predicted_count}",
            f"- 覆盖率: **{coverage*100:.2f}%** ({predicted_count}/{total_records})",
            f"- 正确率: **{precision*100:.2f}%** ({stats['type_accuracy']['correct']}/{predicted_count})",
        ])
        report_lines.extend(format_error_stats(stats['type_accuracy'], 'device_type'))
        report_lines.append("")
    
    # 添加详细结果（前20条）
    if stats['details']:
        report_lines.extend([
            "## 详细结果（前20条）",
            "",
        ])
        
        # 动态生成表头
        headers = ["IP"]
        if 'vendor' in available_labels:
            headers.extend(["标签厂商", "预测厂商", "厂商匹配"])
        if 'os' in available_labels:
            headers.extend(["标签OS", "预测OS", "OS匹配"])
        if 'device_type' in available_labels:
            headers.extend(["标签类型", "预测类型", "类型匹配"])
        headers.append("置信度")
        
        report_lines.append("| " + " | ".join(headers) + " |")
        report_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        for detail in stats['details'][:20]:
            row = [detail['ip'][:20]]
            
            if 'vendor' in available_labels:
                vendor_match = "[OK]" if detail.get('vendor_match') else "[FAIL]"
                label_vendor = (detail.get('label_vendor') or '-')[:15]
                pred_vendor = (detail.get('pred_vendor') or '-')[:15]
                row.extend([label_vendor, pred_vendor, vendor_match])
            
            if 'os' in available_labels:
                os_match = "[OK]" if detail.get('os_match') else "[FAIL]"
                label_os = (detail.get('label_os') or '-')[:15]
                pred_os = (detail.get('pred_os') or '-')[:15]
                row.extend([label_os, pred_os, os_match])
            
            if 'device_type' in available_labels:
                type_match = "[OK]" if detail.get('type_match') else "[FAIL]"
                label_type = (detail.get('label_type') or '-')[:15]
                pred_type = (detail.get('pred_type') or '-')[:15]
                row.extend([label_type, pred_type, type_match])
            
            row.append(f"{detail['confidence']:.2f}")
            report_lines.append("| " + " | ".join(row) + " |")
        
        report_lines.append("")
    
    # 添加错误分析
    if 'vendor' in available_labels:
        errors = [d for d in stats['details'] if not d.get('vendor_match') and d.get('label_vendor')]
        if errors:
            report_lines.extend([
                "## 厂商识别错误分析（前10条）",
                "",
            ])
            
            for detail in errors[:10]:
                report_lines.append(
                    f"- **{detail['ip']}**: 标签=`{detail.get('label_vendor') or '-'}`, "
                    f"预测=`{detail.get('pred_vendor') or '-'}`"
                )
            
            report_lines.append("")
    
    if 'device_type' in available_labels:
        errors = [d for d in stats['details'] if not d.get('type_match') and d.get('label_type')]
        if errors:
            report_lines.extend([
                "## 设备类型识别错误分析（前10条）",
                "",
            ])
            
            for detail in errors[:10]:
                report_lines.append(
                    f"- **{detail['ip']}**: 标签=`{detail.get('label_type') or '-'}`, "
                    f"预测=`{detail.get('pred_type') or '-'}`"
                )
            
            report_lines.append("")
    
    # 写入文件
    report_content = '\n'.join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return output_path


def load_sampled_ips(sampled_ips_path: str) -> Optional[Set[str]]:
    """
    从中间文件加载采样的IP列表
    
    Args:
        sampled_ips_path: 采样IP列表文件路径
        
    Returns:
        采样的IP集合，如果文件不存在则返回None
    """
    if not os.path.exists(sampled_ips_path):
        return None
    
    try:
        with open(sampled_ips_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ips = data.get('ips', [])
            return set(ips) if ips else None
    except (json.JSONDecodeError, IOError) as e:
        print(f"[WARN] 读取采样IP列表失败: {e}")
        return None


def run_accuracy_calculation(input_path: str, output_path: str, 
                             report_path: str = 'accuracy_report.md',
                             max_records: int = None,
                             sampled_ips_path: str = None) -> Dict:
    """
    运行准确率计算，根据输入数据中存在的标签动态评估
    
    新准确率计算规则（2025-01-10更新）：
    - 低置信度数据（<0.6）视为错误，纳入统计
    - 预测值为空视为错误，纳入统计
    - 输入标签为空但输出预测不为空视为错误，纳入统计
    - 预测值与标签不匹配视为错误
    
    覆盖率和正确率计算（2025-01-17新增）：
    - 覆盖率 = 标签不为null的数量 / 总记录数
    - 正确率 = 正确的结果 / 标签不为null的数量
    
    Args:
        input_path: 输入数据路径
        output_path: 输出结果路径
        report_path: 报告输出路径
        max_records: 最大记录数（当sampled_ips_path为None时使用）
        sampled_ips_path: 采样IP列表文件路径（如果提供，只计算这些IP的准确率）
        
    Returns:
        准确率统计结果
    """
    print("\n" + "=" * 60)
    print("准确率评估（新规则：低置信度/空值/不匹配均视为错误）")
    print("=" * 60)
    
    # 尝试加载采样IP列表
    sampled_ips = None
    if sampled_ips_path:
        sampled_ips = load_sampled_ips(sampled_ips_path)
        if sampled_ips:
            print(f"\n加载采样IP列表: {sampled_ips_path}")
            print(f"  采样IP数量: {len(sampled_ips)}")
    
    # 加载标签（同时返回可用标签类型）
    print(f"\n加载输入标签: {input_path}")
    if sampled_ips:
        # 使用采样IP列表过滤标签
        labels, available_labels = load_input_labels(input_path, sampled_ips=sampled_ips)
    else:
        # 使用max_records限制
        labels, available_labels = load_input_labels(input_path, max_records)
    print(f"  找到 {len(labels)} 条有标签的记录")
    print(f"  检测到的标签类型: {', '.join(available_labels) if available_labels else '无'}")
    
    if not available_labels:
        print("\n[WARN] 输入数据中未检测到任何标签字段，无法进行准确率评估")
        return {'total_labels': 0, 'total_results': 0, 'matched_count': 0, 'available_labels': []}
    
    # 加载结果（不跳过任何数据，全部纳入统计）
    print(f"\n加载输出结果: {output_path}")
    results, invalid_count, low_conf_count = load_output_results(output_path, skip_invalid=False, skip_low_confidence=False)
    print(f"  找到 {len(results)} 条结果记录")
    if invalid_count > 0:
        print(f"  其中 {invalid_count} 条预测为空（将视为错误）")
    if low_conf_count > 0:
        print(f"  其中 {low_conf_count} 条低置信度（<{MIN_CONFIDENCE_THRESHOLD}，将视为错误）")
    
    # 计算准确率（传入可用标签和总记录数）
    print("\n计算准确率...")
    print(f"  错误判定规则：")
    print(f"    1. 置信度 < {MIN_CONFIDENCE_THRESHOLD} -> 错误")
    print(f"    2. 预测值为空 -> 错误")
    print(f"    3. 标签为空但预测不为空 -> 错误")
    print(f"    4. 预测与标签不匹配 -> 错误")
    
    # 传入总记录数用于计算覆盖率
    total_records = len(results)
    stats = calculate_accuracy(labels, results, available_labels, total_records)
    stats['invalid_count'] = invalid_count
    stats['low_conf_count'] = low_conf_count
    
    # 辅助函数：显示错误分类
    def print_error_breakdown(accuracy_data: Dict, label_name: str):
        errors = accuracy_data.get('errors', {})
        total_errors = accuracy_data['total'] - accuracy_data['correct']
        if total_errors > 0:
            print(f"  错误分类（共{total_errors}条）：")
            if errors.get('low_confidence', 0) > 0:
                print(f"    - 低置信度: {errors['low_confidence']}")
            if errors.get('empty_prediction', 0) > 0:
                print(f"    - 预测为空: {errors['empty_prediction']}")
            if errors.get('label_empty_pred_not', 0) > 0:
                print(f"    - 标签空但预测不空: {errors['label_empty_pred_not']}")
            if errors.get('mismatch', 0) > 0:
                print(f"    - 不匹配: {errors['mismatch']}")
    
    # 显示结果
    print(f"\n匹配记录数: {stats['matched_count']}")
    print(f"总记录数: {total_records}")
    
    if 'vendor' in available_labels and 'vendor_accuracy' in stats:
        print(f"\n厂商识别 (Vendor):")
        print(f"  准确率: {stats['vendor_accuracy']['correct']}/{stats['vendor_accuracy']['total']} = {stats['vendor_accuracy']['rate']*100:.2f}%")
        predicted_count = stats['vendor_accuracy'].get('predicted_count', 0)
        print(f"  覆盖率: {predicted_count}/{total_records} = {stats['coverage'].get('vendor', 0)*100:.2f}%")
        print(f"  正确率: {stats['vendor_accuracy']['correct']}/{predicted_count} = {stats['precision'].get('vendor', 0)*100:.2f}%")
        print_error_breakdown(stats['vendor_accuracy'], 'vendor')
    
    if 'os' in available_labels and 'os_accuracy' in stats:
        print(f"\n操作系统识别 (OS):")
        print(f"  准确率: {stats['os_accuracy']['correct']}/{stats['os_accuracy']['total']} = {stats['os_accuracy']['rate']*100:.2f}%")
        predicted_count = stats['os_accuracy'].get('predicted_count', 0)
        print(f"  覆盖率: {predicted_count}/{total_records} = {stats['coverage'].get('os', 0)*100:.2f}%")
        print(f"  正确率: {stats['os_accuracy']['correct']}/{predicted_count} = {stats['precision'].get('os', 0)*100:.2f}%")
        print_error_breakdown(stats['os_accuracy'], 'os')
    
    if 'device_type' in available_labels and 'type_accuracy' in stats:
        print(f"\n设备类型识别 (Device Type):")
        print(f"  准确率: {stats['type_accuracy']['correct']}/{stats['type_accuracy']['total']} = {stats['type_accuracy']['rate']*100:.2f}%")
        predicted_count = stats['type_accuracy'].get('predicted_count', 0)
        print(f"  覆盖率: {predicted_count}/{total_records} = {stats['coverage'].get('device_type', 0)*100:.2f}%")
        print(f"  正确率: {stats['type_accuracy']['correct']}/{predicted_count} = {stats['precision'].get('device_type', 0)*100:.2f}%")
        print_error_breakdown(stats['type_accuracy'], 'device_type')
    
    # 生成报告
    print(f"\n生成报告: {report_path}")
    generate_accuracy_report(stats, report_path)
    print(f"  报告已保存")
    
    print("\n" + "=" * 60)
    
    return stats
