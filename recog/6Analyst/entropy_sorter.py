"""
信息熵排序模块

根据数据的信息熵对数据集进行排序和筛选
"""

from typing import List, Dict, Optional, Tuple
from .exp.data_extractor import calculate_richness_score


def calculate_entropy_for_dataset(dataset: List[Dict]) -> List[Tuple[Dict, float]]:
    """
    为数据集中的每条数据计算信息熵
    
    Args:
        dataset: 数据集列表，每个元素是一个字典
                 支持两种格式：
                 1. {ip: data} 格式（原始数据）
                 2. {'Services': ...} 格式（直接数据）
        
    Returns:
        包含(数据, 信息熵)元组的列表
    """
    entropy_data = []
    
    for data in dataset:
        # 检测数据格式：如果只有一个key且看起来像IP，则提取内部数据
        actual_data = data
        if len(data) == 1:
            key = next(iter(data.keys()))
            # 检查是否是IP格式的key（包含.或:）
            if isinstance(key, str) and ('.' in key or ':' in key):
                actual_data = data[key]
        
        # 计算信息熵
        entropy = calculate_richness_score(actual_data)
        entropy_data.append((data, entropy))
    
    return entropy_data


def isort(dataset: List[Dict], ratio: Optional[float] = None) -> List[Dict]:
    """
    根据信息熵对数据集进行排序和筛选
    
    Args:
        dataset: 输入数据集，每个元素是一个字典
        ratio: 可选，保留数据的比例（0-1之间）
               如果指定，则只返回信息熵最高的前ratio比例的数据
               如果为None，则返回全部数据（按信息熵从高到低排序）
    
    Returns:
        排序后（并可能筛选过）的数据集
        
    Examples:
        >>> data = [{'services': 'xxx'}, {'services': 'yyy'}]
        >>> sorted_data = isort(data)  # 返回全部数据，按熵从高到低排序
        >>> top_80_data = isort(data, 0.8)  # 返回信息熵最高的前80%数据
    """
    if not dataset:
        return []
    
    # 验证ratio参数
    if ratio is not None:
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio必须在(0, 1]区间内，当前值: {ratio}")
    
    # 计算每条数据的信息熵
    entropy_data = calculate_entropy_for_dataset(dataset)
    
    # 按信息熵从高到低排序
    sorted_data = sorted(entropy_data, key=lambda x: x[1], reverse=True)
    
    # 如果指定了ratio，则只保留前ratio比例的数据
    if ratio is not None:
        keep_count = max(1, int(len(sorted_data) * ratio))  # 至少保留1条
        sorted_data = sorted_data[:keep_count]
    
    # 只返回数据部分（不包含熵值）
    result = [data for data, entropy in sorted_data]
    
    return result


def isort_with_entropy(dataset: List[Dict], ratio: Optional[float] = None) -> List[Tuple[Dict, float]]:
    """
    根据信息熵对数据集进行排序和筛选，同时返回信息熵值
    
    Args:
        dataset: 输入数据集，每个元素是一个字典
        ratio: 可选，保留数据的比例（0-1之间）
    
    Returns:
        包含(数据, 信息熵)元组的列表，按信息熵从高到低排序
        
    Examples:
        >>> data = [{'services': 'xxx'}, {'services': 'yyy'}]
        >>> sorted_data = isort_with_entropy(data, 0.8)
        >>> for item, entropy in sorted_data:
        ...     print(f"Entropy: {entropy:.2f}")
    """
    if not dataset:
        return []
    
    # 验证ratio参数
    if ratio is not None:
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio必须在(0, 1]区间内，当前值: {ratio}")
    
    # 计算每条数据的信息熵
    entropy_data = calculate_entropy_for_dataset(dataset)
    
    # 按信息熵从高到低排序
    sorted_data = sorted(entropy_data, key=lambda x: x[1], reverse=True)
    
    # 如果指定了ratio，则只保留前ratio比例的数据
    if ratio is not None:
        keep_count = max(1, int(len(sorted_data) * ratio))
        sorted_data = sorted_data[:keep_count]
    
    return sorted_data


def get_entropy_statistics(dataset: List[Dict]) -> Dict:
    """
    获取数据集的信息熵统计信息
    
    Args:
        dataset: 输入数据集
        
    Returns:
        包含统计信息的字典：
        {
            'count': 数据总数,
            'min': 最小熵值,
            'max': 最大熵值,
            'mean': 平均熵值,
            'median': 中位数熵值
        }
    """
    if not dataset:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0
        }
    
    # 计算信息熵
    entropy_values = [calculate_richness_score(data) for data in dataset]
    entropy_values.sort()
    
    count = len(entropy_values)
    min_entropy = entropy_values[0]
    max_entropy = entropy_values[-1]
    mean_entropy = sum(entropy_values) / count
    
    # 计算中位数
    if count % 2 == 0:
        median_entropy = (entropy_values[count // 2 - 1] + entropy_values[count // 2]) / 2
    else:
        median_entropy = entropy_values[count // 2]
    
    return {
        'count': count,
        'min': min_entropy,
        'max': max_entropy,
        'mean': mean_entropy,
        'median': median_entropy
    }


# 主流厂商列表（用于厂商平衡采样）- 已废弃，改为动态识别
# MAJOR_VENDORS = ['Juniper', 'Cisco', 'MikroTik', 'Huawei', 'Fortinet']

# 厂商别名映射表：将各种变体名称统一归类到标准厂商名
# 格式：关键词（小写） -> 标准厂商名
VENDOR_ALIASES = {
    # Cisco 系列
    'cisco': 'Cisco',
    'cisco ios': 'Cisco',
    'cisco ios xr': 'Cisco',
    'cisco ios xe': 'Cisco',
    'cisco nx-os': 'Cisco',
    'cisco asa': 'Cisco',
    'cisco aironet': 'Cisco',
    'cisco catalyst': 'Cisco',
    'cisco nexus': 'Cisco',
    'cisco meraki': 'Cisco',
    'ciscosystems': 'Cisco',
    
    # Juniper 系列
    'juniper': 'Juniper',
    'juniper networks': 'Juniper',
    'junos': 'Juniper',
    'juniper networks, inc.': 'Juniper',
    'juniper srx': 'Juniper',
    'juniper mx': 'Juniper',
    'juniper ex': 'Juniper',
    
    # MikroTik 系列
    'mikrotik': 'MikroTik',
    'routeros': 'MikroTik',
    'mikrotik routeros': 'MikroTik',
    
    # Huawei 系列
    'huawei': 'Huawei',
    'huawei vrp': 'Huawei',
    'huawei versatile routing platform': 'Huawei',
    'h3c': 'Huawei',  # H3C 是华为与3Com合资，后独立但技术同源
    'hpe comware': 'Huawei',
    
    # Fortinet 系列
    'fortinet': 'Fortinet',
    'fortigate': 'Fortinet',
    'fortios': 'Fortinet',
    
    # Arista 系列
    'arista': 'Arista',
    'arista eos': 'Arista',
    'arista networks': 'Arista',
    
    # Palo Alto 系列
    'palo alto': 'Palo Alto',
    'palo alto networks': 'Palo Alto',
    'pan-os': 'Palo Alto',
    
    # Nokia/Alcatel-Lucent 系列
    'nokia': 'Nokia',
    'alcatel': 'Nokia',
    'alcatel-lucent': 'Nokia',
    'nokia sr os': 'Nokia',
    'timos': 'Nokia',
    
    # Extreme Networks 系列
    'extreme': 'Extreme',
    'extreme networks': 'Extreme',
    'extremexos': 'Extreme',
    
    # Dell/Force10 系列
    'dell': 'Dell',
    'dell emc': 'Dell',
    'dell networking': 'Dell',
    'force10': 'Dell',
    'ftos': 'Dell',
    
    # HP/HPE/Aruba 系列
    'hp': 'HPE',
    'hpe': 'HPE',
    'hewlett packard': 'HPE',
    'hewlett-packard': 'HPE',
    'aruba': 'HPE',
    'procurve': 'HPE',
    
    # Brocade 系列
    'brocade': 'Brocade',
    'brocade communications': 'Brocade',
    
    # Ubiquiti 系列
    'ubiquiti': 'Ubiquiti',
    'ubnt': 'Ubiquiti',
    'edgeos': 'Ubiquiti',
    'unifi': 'Ubiquiti',
    
    # ZTE 系列
    'zte': 'ZTE',
    'zte corporation': 'ZTE',
    
    # Ericsson 系列
    'ericsson': 'Ericsson',
    'redback': 'Ericsson',
    
    # F5 系列
    'f5': 'F5',
    'f5 networks': 'F5',
    'big-ip': 'F5',
    
    # Check Point 系列
    'check point': 'Check Point',
    'checkpoint': 'Check Point',
    'gaia': 'Check Point',
    
    # Sophos 系列
    'sophos': 'Sophos',
    'sophos xg': 'Sophos',
    'cyberoam': 'Sophos',
    
    # pfSense/Netgate 系列
    'pfsense': 'pfSense',
    'netgate': 'pfSense',
    
    # VyOS/Vyatta 系列
    'vyos': 'VyOS',
    'vyatta': 'VyOS',
    
    # Linux 路由相关
    'linux': 'Linux',
    'openwrt': 'Linux',
    'dd-wrt': 'Linux',
    
    # FreeBSD 系列
    'freebsd': 'FreeBSD',
    'the freebsd project': 'FreeBSD',
    
    # Netgear 系列
    'netgear': 'Netgear',
    
    # TP-Link 系列
    'tp-link': 'TP-Link',
    'tplink': 'TP-Link',
    
    # D-Link 系列
    'd-link': 'D-Link',
    'dlink': 'D-Link',
    
    # ASUS 系列
    'asus': 'ASUS',
    'asuswrt': 'ASUS',
}


def _normalize_vendor(vendor_str: Optional[str]) -> Optional[str]:
    """
    将厂商字符串标准化为统一的厂商名
    
    Args:
        vendor_str: 原始厂商字符串
        
    Returns:
        标准化后的厂商名，如果无法识别则返回原始值
    """
    if not vendor_str:
        return None
    
    vendor_lower = vendor_str.lower().strip()
    
    # 精确匹配
    if vendor_lower in VENDOR_ALIASES:
        return VENDOR_ALIASES[vendor_lower]
    
    # 模糊匹配：检查是否包含已知厂商关键词
    # 按关键词长度降序排列，优先匹配更具体的名称
    sorted_aliases = sorted(VENDOR_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if alias in vendor_lower:
            return VENDOR_ALIASES[alias]
    
    # 无法识别，返回原始值（首字母大写）
    return vendor_str.strip()


def _get_vendor_from_record(record: Dict) -> Optional[str]:
    """
    从记录中提取厂商信息，并标准化为统一的厂商名
    优先使用 Vendor 字段，若无则从 OS 字段中提取
    
    Args:
        record: 数据记录（包含IP为key的字典或直接的数据字典）
        
    Returns:
        标准化后的厂商名称或None
    """
    # 获取实际数据（可能是 {ip: data} 格式或直接的 data 格式）
    if len(record) == 1:
        key = next(iter(record.keys()))
        # 检查是否是IP格式的key（简单判断）
        if '.' in str(key) or ':' in str(key):
            data = record[key]
        else:
            data = record
    else:
        data = record
    
    # 优先使用 Vendor 字段
    vendor = data.get('Vendor')
    if vendor:
        return _normalize_vendor(vendor)
    
    # 若无 Vendor，尝试从 OS 字段提取
    os_info = data.get('OS')
    if os_info and isinstance(os_info, dict):
        os_value = os_info.get('OS', '')
        if os_value:
            return _normalize_vendor(os_value)
    
    return None


def _analyze_vendor_distribution(dataset: List[Dict]) -> Dict[str, List[Tuple[Dict, float]]]:
    """
    分析数据集的厂商分布，计算每条数据的信息熵
    
    Returns:
        {vendor_name: [(data, entropy), ...], ...}
    """
    vendor_data = {}
    
    for data in dataset:
        entropy = calculate_richness_score(data)
        vendor = _get_vendor_from_record(data)
        vendor_key = vendor if vendor else '__unknown__'
        
        if vendor_key not in vendor_data:
            vendor_data[vendor_key] = []
        vendor_data[vendor_key].append((data, entropy))
    
    # 每个厂商内部按熵从高到低排序
    for vendor in vendor_data:
        vendor_data[vendor].sort(key=lambda x: x[1], reverse=True)
    
    return vendor_data


def isort_with_vendor_balance(
    dataset: List[Dict],
    target_count: int,
    major_ratio: float = 0.90,
    max_single_vendor_ratio: float = 0.90,
    min_single_vendor_ratio: float = 0.10,
    top_n_vendors: int = 5
) -> Tuple[List[Dict], Dict]:
    """
    厂商平衡采样：按信息熵排序，同时保证厂商分布平衡
    
    采样规则：
    1. 动态识别数据集中排名前 top_n_vendors 的厂商作为主流厂商
    2. 如果其他厂商占比 < 10%，则：其他厂商全采样，主流厂商目标 = 其他厂商数量 * 9
    3. 否则：其他厂商目标 = 总目标 * 10%，主流厂商目标 = 总目标 * 90%
    4. 每个主流厂商最低采样 10%（占主流厂商目标），最高 40%（占总目标）
    5. 必须保证每个主流厂商都有数据被采样
    6. 采样顺序按信息熵从高到低
    
    Args:
        dataset: 输入数据集
        target_count: 目标采样数量
        major_ratio: 主流厂商数据占比（默认0.9，即90%）
        max_single_vendor_ratio: 单个主流厂商最大占比（默认0.90，占总目标）
        min_single_vendor_ratio: 单个主流厂商最小占比（默认0.10，占主流厂商目标）
        top_n_vendors: 取前N个厂商作为主流厂商（默认5）
        
    Returns:
        (采样后的数据列表, 统计信息字典)
    """
    if not dataset:
        return [], {'total': 0, 'sampled': 0, 'vendor_distribution': {}}
    
    # 第一步：分析厂商分布
    vendor_data = _analyze_vendor_distribution(dataset)
    
    # 按数量排序，确定主流厂商（前N个）
    vendor_counts = [(v, len(data_list)) for v, data_list in vendor_data.items()]
    vendor_counts.sort(key=lambda x: x[1], reverse=True)
    
    # 排除 unknown，确定主流厂商
    known_vendors = [(v, c) for v, c in vendor_counts if v != '__unknown__']
    major_vendors = [v for v, c in known_vendors[:top_n_vendors]]
    
    # 计算主流厂商和其他厂商的原始数量
    major_total = sum(len(vendor_data[v]) for v in major_vendors if v in vendor_data)
    other_vendors = [v for v in vendor_data.keys() if v not in major_vendors]
    other_total = sum(len(vendor_data[v]) for v in other_vendors)
    total_data = len(dataset)
    
    print(f"\n  [厂商分布分析]")
    print(f"    总数据量: {total_data}")
    print(f"    主流厂商 (Top {len(major_vendors)}): {major_vendors}")
    print(f"    主流厂商数据: {major_total} 条 ({major_total/total_data*100:.1f}%)")
    print(f"    其他厂商数据: {other_total} 条 ({other_total/total_data*100:.1f}%)")
    
    # 第二步：计算采样目标
    other_ratio_in_data = other_total / total_data if total_data > 0 else 0
    
    if other_ratio_in_data < 0.10:
        # 其他厂商占比不足10%，优先保证其他厂商全采样
        # 但总数不能超过 target_count
        other_target = min(other_total, int(target_count * (1 - major_ratio)))
        # 如果其他厂商数量很少，可以全采样，剩余给主流厂商
        if other_total <= int(target_count * (1 - major_ratio)):
            other_target = other_total
            major_target = min(target_count - other_target, major_total)
        else:
            other_target = int(target_count * (1 - major_ratio))
            major_target = min(int(target_count * major_ratio), major_total)
        print(f"    [调整] 其他厂商占比不足10%，优先采样其他厂商")
        print(f"    调整后目标: 主流 {major_target} 条, 其他 {other_target} 条")
    else:
        # 正常比例
        other_target = min(int(target_count * (1 - major_ratio)), other_total)
        major_target = min(int(target_count * major_ratio), major_total)
    
    # 第三步：计算每个主流厂商的采样配额
    # 最低: 主流目标的 10%，最高: 总目标的 40%
    final_target = major_target + other_target
    min_per_vendor = max(1, int(major_target * min_single_vendor_ratio))  # 至少1条
    max_per_vendor = int(final_target * max_single_vendor_ratio)
    
    print(f"\n  [采样配额计算]")
    print(f"    主流厂商目标: {major_target} 条")
    print(f"    其他厂商目标: {other_target} 条")
    print(f"    单厂商最低: {min_per_vendor} 条 (主流目标的10%)")
    print(f"    单厂商最高: {max_per_vendor} 条 (总目标的40%)")
    
    # 计算每个主流厂商的初始配额（按比例分配）
    vendor_quotas = {}
    
    # 首先计算每个厂商的理想配额（按数据量比例）
    for vendor in major_vendors:
        available = len(vendor_data.get(vendor, []))
        # 按比例计算理想配额
        if major_total > 0:
            ideal_quota = int(major_target * available / major_total)
        else:
            ideal_quota = 0
        # 限制在 [min, max] 范围内，且不超过可用数量
        quota = max(min_per_vendor, min(ideal_quota, max_per_vendor, available))
        vendor_quotas[vendor] = {
            'available': available,
            'quota': quota,
            'min': min_per_vendor,
            'max': max_per_vendor
        }
    
    # 检查配额总和是否达到主流目标，进行调整
    total_quota = sum(v['quota'] for v in vendor_quotas.values())
    
    if total_quota < major_target:
        # 配额不足，需要补齐（按比例从有余量的厂商补）
        shortfall = major_target - total_quota
        while shortfall > 0:
            added = False
            for vendor in major_vendors:
                if shortfall <= 0:
                    break
                info = vendor_quotas[vendor]
                can_add = min(info['available'] - info['quota'], info['max'] - info['quota'], 1)
                if can_add > 0:
                    info['quota'] += can_add
                    shortfall -= can_add
                    added = True
            if not added:
                break  # 无法再添加
    elif total_quota > major_target:
        # 配额超出，需要削减（从配额最多的厂商削减，但保证最低）
        excess = total_quota - major_target
        while excess > 0:
            reduced = False
            sorted_vendors = sorted(major_vendors, key=lambda v: vendor_quotas[v]['quota'], reverse=True)
            for vendor in sorted_vendors:
                if excess <= 0:
                    break
                info = vendor_quotas[vendor]
                can_reduce = info['quota'] - info['min']
                if can_reduce > 0:
                    info['quota'] -= 1
                    excess -= 1
                    reduced = True
                    break  # 每次只削减一个，重新排序
            if not reduced:
                break  # 无法再削减
    
    print(f"\n  [最终配额]")
    for vendor in major_vendors:
        info = vendor_quotas[vendor]
        print(f"    {vendor}: {info['quota']} 条 (可用 {info['available']})")
    
    # 第四步：执行采样
    # 先采样其他厂商（按熵从高到低）
    sampled_other = []
    for vendor in other_vendors:
        data_list = vendor_data.get(vendor, [])
        for data, entropy in data_list:
            if len(sampled_other) >= other_target:
                break
            sampled_other.append(data)
        if len(sampled_other) >= other_target:
            break
    
    # 再采样主流厂商（按配额）
    sampled_major = []
    vendor_sampled_counts = {v: 0 for v in major_vendors}
    
    for vendor in major_vendors:
        quota = vendor_quotas[vendor]['quota']
        data_list = vendor_data.get(vendor, [])
        count = 0
        for data, entropy in data_list:
            if count >= quota:
                break
            sampled_major.append(data)
            count += 1
        vendor_sampled_counts[vendor] = count
    
    # 合并结果
    result = sampled_other + sampled_major
    
    # 统计信息
    stats = {
        'total': len(dataset),
        'sampled': len(result),
        'major_count': len(sampled_major),
        'other_count': len(sampled_other),
        'major_vendors': major_vendors,
        'other_vendors': other_vendors,
        'vendor_distribution': vendor_sampled_counts,
        'vendor_quotas': {v: vendor_quotas[v]['quota'] for v in major_vendors},
        'original_distribution': {v: len(vendor_data.get(v, [])) for v in vendor_data.keys()}
    }
    
    return result, stats


def uniform_sample(dataset: List[Dict], target_count: int, verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """
    均匀采样：按固定间隔从数据集中采样
    
    采样规则：
    1. 计算采样间隔 = 数据总量 / 目标采样数（向下取整）
    2. 从第1条开始，每隔 interval 条采样一条
    3. 如果目标数量 >= 数据总量，返回全部数据
    
    Args:
        dataset: 输入数据集（已按信息熵排序）
        target_count: 目标采样数量
        verbose: 是否打印详细信息（默认True）
        
    Returns:
        (采样后的数据列表, 统计信息字典)
        
    Examples:
        >>> data = [...]  # 13000条数据
        >>> sampled, stats = uniform_sample(data, 2000)
        >>> # 间隔 = 13000 // 2000 = 6
        >>> # 采样第1, 7, 13, 19, ... 条
    """
    if not dataset:
        return [], {'total': 0, 'sampled': 0, 'interval': 0}
    
    total = len(dataset)
    
    # 如果目标数量 >= 数据总量，返回全部数据
    if target_count >= total:
        return dataset, {
            'total': total,
            'sampled': total,
            'interval': 1,
            'target': target_count,
            'note': '目标数量>=数据总量，返回全部数据'
        }
    
    # 计算采样间隔（向下取整）
    interval = total // target_count
    if interval < 1:
        interval = 1
    
    # 执行均匀采样：从索引0开始，每隔interval取一条
    sampled = []
    index = 0
    while index < total and len(sampled) < target_count:
        sampled.append(dataset[index])
        index += interval
    
    stats = {
        'total': total,
        'sampled': len(sampled),
        'interval': interval,
        'target': target_count,
        'actual_ratio': len(sampled) / total if total > 0 else 0
    }
    
    if verbose:
        print(f"\n[均匀采样]")
        print(f"  数据总量: {total}")
        print(f"  目标采样: {target_count}")
        print(f"  采样间隔: {interval}")
        print(f"  实际采样: {len(sampled)} 条 ({stats['actual_ratio']*100:.1f}%)")
    
    return sampled, stats


def presample_by_vendor(
    input_path: str,
    target_count: int,
    major_ratio: float = 0.9,
    max_single_vendor_ratio: float = 0.90,
    top_n_vendors: int = 5
) -> Tuple[List[str], Dict]:
    """
    根据标签中的厂商信息，从原始数据文件中按比例预采样
    
    在数据清洗之前执行，确保采样后的数据保持厂商比例。
    
    流程：
    1. 扫描原始数据，按厂商分类（使用标签中的 Vendor 或 OS 字段）
    2. 动态识别前 top_n_vendors 个厂商作为主流厂商
    3. 按 major_ratio 比例采样主流厂商和其他厂商（默认90%主流+10%其他）
    4. 单个厂商最多占总目标的 max_single_vendor_ratio（默认90%）
    5. 返回采样后的原始数据行（未清洗）
    
    Args:
        input_path: 输入文件或目录路径
        target_count: 目标采样数量
        major_ratio: 主流厂商占比（默认0.9，即90%）
        max_single_vendor_ratio: 单个厂商最大占比（默认0.90，占总目标）
        top_n_vendors: 取前N个厂商作为主流厂商（默认5）
        
    Returns:
        (采样后的原始数据行列表, 统计信息字典)
    """
    import os
    import glob
    import json
    
    # 获取所有输入文件
    if os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        input_files = sorted(json_files + jsonl_files)
    elif os.path.isfile(input_path):
        input_files = [input_path]
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    
    if not input_files:
        return [], {'error': '未找到输入文件'}
    
    print(f"\n[预采样] 扫描 {len(input_files)} 个输入文件...")
    
    # 第一步：扫描所有数据，按厂商分类
    # 存储格式：{vendor: [(file_path, line_number, raw_line), ...]}
    vendor_lines = {}
    total_scanned = 0
    
    for filepath in input_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_scanned += 1
                    
                    try:
                        record = json.loads(line)
                        # 提取厂商信息
                        vendor = _get_vendor_from_record(record)
                        vendor_key = vendor if vendor else '__unknown__'
                        
                        if vendor_key not in vendor_lines:
                            vendor_lines[vendor_key] = []
                        vendor_lines[vendor_key].append((filepath, line_num, line))
                        
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  警告: 读取文件 {filepath} 失败: {e}")
            continue
    
    print(f"  扫描完成: {total_scanned} 条记录")
    
    # 第二步：确定主流厂商（按数量排序，取前N个）
    vendor_counts = [(v, len(lines)) for v, lines in vendor_lines.items()]
    vendor_counts.sort(key=lambda x: x[1], reverse=True)
    
    # 排除 unknown，确定主流厂商
    known_vendors = [(v, c) for v, c in vendor_counts if v != '__unknown__']
    major_vendors = [v for v, c in known_vendors[:top_n_vendors]]
    
    # 计算各类数量
    major_total = sum(len(vendor_lines[v]) for v in major_vendors if v in vendor_lines)
    other_vendors = [v for v in vendor_lines.keys() if v not in major_vendors]
    other_total = sum(len(vendor_lines[v]) for v in other_vendors)
    
    print(f"\n[预采样] 厂商分布分析:")
    print(f"  总数据量: {total_scanned}")
    print(f"  主流厂商 (Top {len(major_vendors)}): {major_vendors}")
    print(f"  主流厂商数据: {major_total} 条 ({major_total/max(total_scanned,1)*100:.1f}%)")
    print(f"  其他厂商数据: {other_total} 条 ({other_total/max(total_scanned,1)*100:.1f}%)")
    
    # 第三步：计算采样目标
    major_target = int(target_count * major_ratio)
    other_target = target_count - major_target
    
    # 调整目标（如果数据不足）
    if major_total < major_target:
        print(f"  [调整] 主流厂商数据不足: 需要 {major_target}，可用 {major_total}")
        major_target = major_total
        other_target = min(target_count - major_target, other_total)
    
    if other_total < other_target:
        print(f"  [调整] 其他厂商数据不足: 需要 {other_target}，可用 {other_total}")
        other_target = other_total
        major_target = min(target_count - other_target, major_total)
    
    final_target = major_target + other_target
    print(f"\n[预采样] 采样目标:")
    print(f"  主流厂商: {major_target} 条 ({major_target/max(final_target,1)*100:.1f}%)")
    print(f"  其他厂商: {other_target} 条 ({other_target/max(final_target,1)*100:.1f}%)")
    
    # 第四步：按比例分配主流厂商配额（带单厂商上限）
    max_per_vendor = int(target_count * max_single_vendor_ratio)  # 单厂商上限
    vendor_quotas = {}
    if major_total > 0:
        for vendor in major_vendors:
            available = len(vendor_lines.get(vendor, []))
            # 按比例分配
            quota = int(major_target * available / major_total)
            # 确保至少有1条（如果有数据的话）
            if available > 0 and quota == 0:
                quota = 1
            # 应用单厂商上限
            quota = min(quota, available, max_per_vendor)
            vendor_quotas[vendor] = quota
    
    # 调整配额总和
    total_quota = sum(vendor_quotas.values())
    if total_quota < major_target:
        # 补齐差额（但不超过单厂商上限）
        shortfall = major_target - total_quota
        for vendor in major_vendors:
            if shortfall <= 0:
                break
            available = len(vendor_lines.get(vendor, []))
            current_quota = vendor_quotas.get(vendor, 0)
            # 可添加量 = min(可用量 - 当前配额, 上限 - 当前配额)
            can_add = min(available - current_quota, max_per_vendor - current_quota)
            if can_add > 0:
                add = min(can_add, shortfall)
                vendor_quotas[vendor] = current_quota + add
                shortfall -= add
    
    print(f"\n[预采样] 主流厂商配额 (单厂商上限: {max_per_vendor}):")
    for vendor in major_vendors:
        quota = vendor_quotas.get(vendor, 0)
        available = len(vendor_lines.get(vendor, []))
        print(f"    {vendor}: {quota} 条 (可用 {available})")
    
    # 第五步：执行采样
    sampled_lines = []
    vendor_sampled = {v: 0 for v in major_vendors}
    
    # 采样主流厂商
    for vendor in major_vendors:
        quota = vendor_quotas.get(vendor, 0)
        lines = vendor_lines.get(vendor, [])
        for i, (filepath, line_num, raw_line) in enumerate(lines):
            if i >= quota:
                break
            sampled_lines.append(raw_line)
            vendor_sampled[vendor] += 1
    
    # 采样其他厂商
    other_sampled = 0
    for vendor in other_vendors:
        if other_sampled >= other_target:
            break
        lines = vendor_lines.get(vendor, [])
        for filepath, line_num, raw_line in lines:
            if other_sampled >= other_target:
                break
            sampled_lines.append(raw_line)
            other_sampled += 1
    
    # 统计信息
    stats = {
        'total_scanned': total_scanned,
        'sampled': len(sampled_lines),
        'major_target': major_target,
        'other_target': other_target,
        'major_sampled': sum(vendor_sampled.values()),
        'other_sampled': other_sampled,
        'major_vendors': major_vendors,
        'vendor_distribution': vendor_sampled,
        'original_distribution': {v: len(vendor_lines.get(v, [])) for v in vendor_lines.keys()}
    }
    
    print(f"\n[预采样] 采样完成:")
    print(f"  总采样: {len(sampled_lines)} 条")
    print(f"  主流厂商: {stats['major_sampled']} 条")
    print(f"  其他厂商: {stats['other_sampled']} 条")
    
    return sampled_lines, stats



def presample_by_difficulty(
    input_path: str,
    target_count: int,
    difficulty_vendors: Dict[str, List[str]] = None,
    difficulty_ratios: Dict[str, float] = None,
    max_single_vendor_ratio: float = 0.6
) -> Tuple[List[str], Dict]:
    """
    按难度分级采样：根据厂商识别难度进行分层采样
    
    采样规则：
    1. 将厂商分为三个难度级别：easy（易识别）、normal（较易识别）、hard（难识别）
    2. 按指定比例从各难度级别采样（默认 easy:normal:hard = 80:10:10）
    3. 在每个难度级别内部，单个厂商不超过该级别的指定比例（默认60%）
    
    Args:
        input_path: 输入文件或目录路径
        target_count: 目标采样数量
        difficulty_vendors: 难度分级厂商字典，格式：
            {
                'easy': ['MikroTik', 'Keenetic'],
                'normal': ['Cisco', 'Juniper'],
                'hard': []  # 空列表表示其他所有厂商
            }
        difficulty_ratios: 难度比例字典，格式：
            {
                'easy': 0.8,    # 80%
                'normal': 0.1,  # 10%
                'hard': 0.1     # 10%
            }
        max_single_vendor_ratio: 类别内单厂商上限（默认0.6，即60%）
        
    Returns:
        (采样后的原始数据行列表, 统计信息字典)
        
    Examples:
        >>> # 采样100条，easy:normal:hard = 80:10:10，单厂商上限60%
        >>> sampled, stats = presample_by_difficulty(
        ...     'input.jsonl',
        ...     target_count=100,
        ...     max_single_vendor_ratio=0.6
        ... )
        >>> # easy: 80条（MikroTik≤48, Keenetic≤48）
        >>> # normal: 10条（Cisco≤6, Juniper≤6）
        >>> # hard: 10条（其他厂商分配）
    """
    import os
    import glob
    import json
    
    # 使用默认配置
    if difficulty_vendors is None:
        from .config import DIFFICULTY_VENDORS
        difficulty_vendors = DIFFICULTY_VENDORS.copy()
    
    if difficulty_ratios is None:
        from .config import DEFAULT_DIFFICULTY_RATIOS
        difficulty_ratios = DEFAULT_DIFFICULTY_RATIOS.copy()
    
    # 验证比例总和为1
    ratio_sum = sum(difficulty_ratios.values())
    if abs(ratio_sum - 1.0) > 0.001:
        raise ValueError(f"难度比例总和必须为1.0，当前为{ratio_sum}")
    
    # 获取所有输入文件
    if os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        input_files = sorted(json_files + jsonl_files)
    elif os.path.isfile(input_path):
        input_files = [input_path]
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    
    if not input_files:
        return [], {'error': '未找到输入文件'}
    
    print(f"\n[难度分级采样] 扫描 {len(input_files)} 个输入文件...")
    
    # 第一步：扫描所有数据，按厂商分类
    vendor_lines = {}
    total_scanned = 0
    
    for filepath in input_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_scanned += 1
                    
                    try:
                        record = json.loads(line)
                        vendor = _get_vendor_from_record(record)
                        vendor_key = vendor if vendor else '__unknown__'
                        
                        if vendor_key not in vendor_lines:
                            vendor_lines[vendor_key] = []
                        vendor_lines[vendor_key].append((filepath, line_num, line))
                        
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  警告: 读取文件 {filepath} 失败: {e}")
            continue
    
    print(f"  扫描完成: {total_scanned} 条记录")
    
    # 第二步：将厂商分配到难度级别
    difficulty_groups = {
        'easy': {},
        'normal': {},
        'hard': {}
    }
    
    for vendor, lines in vendor_lines.items():
        if vendor == '__unknown__':
            difficulty_groups['hard'][vendor] = lines
        elif vendor in difficulty_vendors['easy']:
            difficulty_groups['easy'][vendor] = lines
        elif vendor in difficulty_vendors['normal']:
            difficulty_groups['normal'][vendor] = lines
        else:
            difficulty_groups['hard'][vendor] = lines
    
    # 统计各难度级别的数据量
    print(f"\n[难度分级采样] 数据分布分析:")
    print(f"  总数据量: {total_scanned}")
    for difficulty in ['easy', 'normal', 'hard']:
        group = difficulty_groups[difficulty]
        group_total = sum(len(lines) for lines in group.values())
        vendors = [v for v in group.keys() if v != '__unknown__']
        print(f"  {difficulty.upper()}: {group_total} 条 ({group_total/max(total_scanned,1)*100:.1f}%) - 厂商: {vendors[:5]}")
    
    # 第三步：计算各难度级别的采样目标
    targets = {}
    for difficulty, ratio in difficulty_ratios.items():
        targets[difficulty] = int(target_count * ratio)
    
    # 调整目标以确保总和等于target_count
    total_target = sum(targets.values())
    if total_target < target_count:
        # 补齐差额到easy
        targets['easy'] += (target_count - total_target)
    
    print(f"\n[难度分级采样] 采样目标:")
    for difficulty in ['easy', 'normal', 'hard']:
        target = targets[difficulty]
        ratio = difficulty_ratios[difficulty]
        print(f"  {difficulty.upper()}: {target} 条 ({ratio*100:.0f}%)")
    
    # 第四步：为每个难度级别分配厂商配额
    all_quotas = {}
    
    for difficulty in ['easy', 'normal', 'hard']:
        group = difficulty_groups[difficulty]
        target = targets[difficulty]
        
        if not group or target == 0:
            all_quotas[difficulty] = {}
            continue
        
        # 计算该难度级别的总数据量
        group_total = sum(len(lines) for lines in group.values())
        
        # 计算单厂商上限
        max_per_vendor = int(target * max_single_vendor_ratio)
        
        # 按比例分配配额
        vendor_quotas = {}
        for vendor, lines in group.items():
            available = len(lines)
            # 按比例计算
            if group_total > 0:
                quota = int(target * available / group_total)
            else:
                quota = 0
            # 确保至少1条（如果有数据）
            if available > 0 and quota == 0:
                quota = 1
            # 应用单厂商上限
            quota = min(quota, available, max_per_vendor)
            vendor_quotas[vendor] = quota
        
        # 调整配额总和到目标值
        total_quota = sum(vendor_quotas.values())
        if total_quota < target:
            # 补齐差额（但不超过单厂商上限）
            shortfall = target - total_quota
            # 按可用量排序，优先给数据多的厂商增加配额
            sorted_vendors = sorted(group.keys(), key=lambda v: len(group[v]), reverse=True)
            for vendor in sorted_vendors:
                if shortfall <= 0:
                    break
                available = len(group[vendor])
                current_quota = vendor_quotas.get(vendor, 0)
                can_add = min(available - current_quota, max_per_vendor - current_quota)
                if can_add > 0:
                    add = min(can_add, shortfall)
                    vendor_quotas[vendor] = current_quota + add
                    shortfall -= add
        elif total_quota > target:
            # 削减超出部分
            excess = total_quota - target
            # 优先削减配额为1的厂商（移除小厂商）
            while excess > 0:
                reduced = False
                # 先尝试削减配额为1的厂商
                vendors_with_1 = [v for v, q in vendor_quotas.items() if q == 1]
                if vendors_with_1 and excess > 0:
                    # 按可用量排序，优先削减数据少的厂商
                    vendors_with_1.sort(key=lambda v: len(group[v]))
                    for vendor in vendors_with_1:
                        if excess <= 0:
                            break
                        vendor_quotas[vendor] = 0
                        excess -= 1
                        reduced = True
                
                # 如果还有超出，从配额大的厂商削减
                if excess > 0:
                    sorted_vendors = sorted(
                        [v for v in vendor_quotas.keys() if vendor_quotas[v] > 0],
                        key=lambda v: vendor_quotas[v],
                        reverse=True
                    )
                    for vendor in sorted_vendors:
                        if excess <= 0:
                            break
                        if vendor_quotas[vendor] > 0:
                            vendor_quotas[vendor] -= 1
                            excess -= 1
                            reduced = True
                            break
                
                if not reduced:
                    break
        
        # 移除配额为0的厂商
        vendor_quotas = {v: q for v, q in vendor_quotas.items() if q > 0}
        
        all_quotas[difficulty] = vendor_quotas
        
        print(f"\n[难度分级采样] {difficulty.upper()} 配额 (单厂商上限: {max_per_vendor}):")
        if vendor_quotas:
            for vendor, quota in sorted(vendor_quotas.items(), key=lambda x: -x[1]):
                available = len(group[vendor])
                print(f"    {vendor}: {quota} 条 (可用 {available})")
        else:
            print(f"    (无配额)")
    
    # 第四步补充：全局配额调整，确保总和等于target_count
    actual_total = sum(sum(quotas.values()) for quotas in all_quotas.values())
    if actual_total != target_count:
        print(f"\n[难度分级采样] 全局配额调整: 当前总和 {actual_total}，目标 {target_count}")
        
        if actual_total > target_count:
            # 总配额超出，需要削减
            excess = actual_total - target_count
            print(f"  需要削减 {excess} 条")
            
            # 优先从配额最多的难度级别削减，但保持比例
            while excess > 0:
                reduced = False
                # 按配额总和排序（从多到少）
                sorted_difficulties = sorted(
                    all_quotas.keys(),
                    key=lambda d: sum(all_quotas[d].values()),
                    reverse=True
                )
                
                for difficulty in sorted_difficulties:
                    if excess <= 0:
                        break
                    quotas = all_quotas[difficulty]
                    if not quotas:
                        continue
                    
                    # 从该难度级别中配额最多的厂商削减
                    sorted_vendors = sorted(quotas.keys(), key=lambda v: quotas[v], reverse=True)
                    for vendor in sorted_vendors:
                        if excess <= 0:
                            break
                        if quotas[vendor] > 0:  # 可以削减到0
                            quotas[vendor] -= 1
                            excess -= 1
                            reduced = True
                            break
                
                if not reduced:
                    # 无法再削减，强制削减（即使为0）
                    for difficulty in sorted_difficulties:
                        if excess <= 0:
                            break
                        quotas = all_quotas[difficulty]
                        for vendor in list(quotas.keys()):
                            if excess <= 0:
                                break
                            if quotas[vendor] > 0:
                                quotas[vendor] -= 1
                                excess -= 1
                                break
                    break
        
        elif actual_total < target_count:
            # 总配额不足，需要补齐
            shortfall = target_count - actual_total
            print(f"  需要补齐 {shortfall} 条")
            
            # 优先补齐到easy，然后normal，最后hard
            for difficulty in ['easy', 'normal', 'hard']:
                if shortfall <= 0:
                    break
                quotas = all_quotas.get(difficulty, {})
                if not quotas:
                    continue
                
                group = difficulty_groups[difficulty]
                target = targets[difficulty]
                max_per_vendor = int(target * max_single_vendor_ratio)
                
                # 尝试给每个厂商增加配额
                for vendor in sorted(quotas.keys()):
                    if shortfall <= 0:
                        break
                    available = len(group[vendor])
                    current_quota = quotas[vendor]
                    # 可添加量：不超过可用量，不超过单厂商上限
                    can_add = min(available - current_quota, max_per_vendor - current_quota)
                    if can_add > 0:
                        add = min(can_add, shortfall)
                        quotas[vendor] += add
                        shortfall -= add
        
        # 重新计算总和
        actual_total = sum(sum(quotas.values()) for quotas in all_quotas.values())
        print(f"  调整后总和: {actual_total}")
        
        # 如果还是不等于target_count，说明数据不足或配额分配有问题
        if actual_total != target_count:
            print(f"  [警告] 无法精确达到目标数量，实际: {actual_total}，目标: {target_count}")
    
    # 第五步：执行采样
    sampled_lines = []
    difficulty_sampled = {d: {} for d in ['easy', 'normal', 'hard']}
    
    for difficulty in ['easy', 'normal', 'hard']:
        group = difficulty_groups[difficulty]
        quotas = all_quotas.get(difficulty, {})
        
        for vendor, quota in quotas.items():
            lines = group.get(vendor, [])
            count = 0
            for i, (filepath, line_num, raw_line) in enumerate(lines):
                if count >= quota:
                    break
                sampled_lines.append(raw_line)
                count += 1
            difficulty_sampled[difficulty][vendor] = count
    
    # 统计信息
    stats = {
        'total_scanned': total_scanned,
        'sampled': len(sampled_lines),
        'difficulty_targets': targets,
        'difficulty_sampled': {
            d: sum(difficulty_sampled[d].values()) 
            for d in ['easy', 'normal', 'hard']
        },
        'vendor_distribution': {
            d: difficulty_sampled[d] 
            for d in ['easy', 'normal', 'hard']
        },
        'difficulty_ratios': difficulty_ratios,
        'max_single_vendor_ratio': max_single_vendor_ratio
    }
    
    print(f"\n[难度分级采样] 采样完成:")
    print(f"  总采样: {len(sampled_lines)} 条")
    for difficulty in ['easy', 'normal', 'hard']:
        sampled = stats['difficulty_sampled'][difficulty]
        target = targets[difficulty]
        print(f"  {difficulty.upper()}: {sampled} 条 (目标 {target})")
    
    return sampled_lines, stats
