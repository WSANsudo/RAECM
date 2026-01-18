"""
公共工具模块
提供数据验证、结果标准化、文件操作等通用功能
避免代码重复，提高可维护性
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union


# ========= 数据验证 =========

class ValidationError(Exception):
    """数据验证错误"""
    pass


def validate_record(record: Dict[str, Any], required_fields: List[str] = None) -> Tuple[bool, str]:
    """
    验证单条记录的完整性
    
    Args:
        record: 要验证的记录
        required_fields: 必需字段列表，默认检查 'ip'
        
    Returns:
        (是否有效, 错误信息)
    """
    if not isinstance(record, dict):
        return False, "记录必须是字典类型"
    
    if required_fields is None:
        required_fields = ['ip']
    
    for field in required_fields:
        if field not in record:
            return False, f"缺少必需字段: {field}"
        if record[field] is None or record[field] == '':
            return False, f"字段 {field} 不能为空"
    
    return True, ""


def validate_json_line(line: str) -> Tuple[bool, Optional[Dict], str]:
    """
    验证并解析JSON行
    
    Args:
        line: JSON字符串
        
    Returns:
        (是否有效, 解析后的对象或None, 错误信息)
    """
    line = line.strip()
    if not line:
        return False, None, "空行"
    
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            return False, None, "JSON必须是对象类型"
        return True, obj, ""
    except json.JSONDecodeError as e:
        return False, None, f"JSON解析错误: {str(e)}"


def validate_ip_record(obj: Dict[str, Any]) -> Tuple[bool, str, Dict, str]:
    """
    验证IP记录格式（{ip: data} 格式）
    
    Args:
        obj: JSON对象
        
    Returns:
        (是否有效, IP地址, 数据, 错误信息)
    """
    if not obj:
        return False, "", {}, "空对象"
    
    try:
        ip = next(iter(obj.keys()))
        data = obj[ip]
        if not isinstance(data, dict):
            return False, ip, {}, "IP数据必须是字典类型"
        return True, ip, data, ""
    except StopIteration:
        return False, "", {}, "对象没有键"


# ========= 结果标准化 =========

def normalize_result_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化结果字段值：
    - 所有属性字段未知时统一为 null
    - model字段为列表格式 [[name, conf], ...]
    - 说明性字段（evidence, conclusion等）：空值 -> null
    
    Args:
        record: 要标准化的记录
        
    Returns:
        标准化后的记录
    """
    # 关键结果字段（字符串）：空字符串/"unknown" -> null
    key_str_fields = ['vendor', 'os', 'firmware', 'type', 'result_type']
    for field in key_str_fields:
        val = record.get(field)
        if val is None or val == '' or val == 'null' or val == 'unknown':
            record[field] = None
    
    # model字段特殊处理：应为列表格式
    model_val = record.get('model')
    if model_val is None or model_val == '' or model_val == 'unknown':
        record['model'] = None
    elif isinstance(model_val, str):
        # 兼容旧格式：字符串转为列表
        record['model'] = [[model_val, 0.5]]
    elif isinstance(model_val, list):
        # 验证列表格式
        if len(model_val) == 0:
            record['model'] = None
        else:
            # 确保每个元素是 [name, conf] 格式
            normalized = []
            for item in model_val:
                if isinstance(item, list) and len(item) >= 2:
                    normalized.append([str(item[0]), float(item[1])])
                elif isinstance(item, str):
                    normalized.append([item, 0.5])
            record['model'] = normalized if normalized else None
    
    # 说明性字段（字符串）：空字符串/"unknown" -> null
    desc_str_fields = ['conclusion']
    for field in desc_str_fields:
        val = record.get(field)
        if val == '' or val == 'unknown':
            record[field] = None
    
    # 说明性字段（数组）：空数组或None -> null
    desc_arr_fields = ['evidence']
    for field in desc_arr_fields:
        val = record.get(field)
        if val is None or (isinstance(val, list) and len(val) == 0):
            record[field] = None
    
    return record


def merge_analysis_results(ip: str, product: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并产品分析结果
    
    Args:
        ip: IP地址
        product: 产品分析结果
        
    Returns:
        合并后的记录
    """
    merged = {
        'ip': ip,
        # 产品信息
        'vendor': product.get('vendor'),
        'model': product.get('model'),
        'os': product.get('os'),
        'firmware': product.get('firmware'),
        'type': product.get('type'),
        'result_type': product.get('result_type'),
        'confidence': product.get('confidence', 0),
        'evidence': product.get('evidence', []),
        'conclusion': product.get('conclusion', ''),
    }
    
    return normalize_result_fields(merged)


# ========= 安全文件操作 =========

def safe_read_jsonl(file_path: str, max_records: int = None) -> Tuple[List[Dict], List[str]]:
    """
    安全读取JSONL文件
    
    Args:
        file_path: 文件路径
        max_records: 最大读取条数
        
    Returns:
        (成功解析的记录列表, 错误信息列表)
    """
    records = []
    errors = []
    
    if not os.path.exists(file_path):
        errors.append(f"文件不存在: {file_path}")
        return records, errors
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_records is not None and len(records) >= max_records:
                    break
                
                valid, obj, err = validate_json_line(line)
                if valid:
                    records.append(obj)
                elif err != "空行":
                    errors.append(f"第{i+1}行: {err}")
    except IOError as e:
        errors.append(f"读取文件失败: {str(e)}")
    except Exception as e:
        errors.append(f"未知错误: {str(e)}")
    
    return records, errors


def safe_write_jsonl(file_path: str, records: List[Dict], append: bool = False) -> Tuple[bool, str]:
    """
    安全写入JSONL文件
    
    Args:
        file_path: 文件路径
        records: 要写入的记录列表
        append: 是否追加模式
        
    Returns:
        (是否成功, 错误信息)
    """
    try:
        # 确保目录存在
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        
        return True, ""
    except IOError as e:
        return False, f"写入文件失败: {str(e)}"
    except Exception as e:
        return False, f"未知错误: {str(e)}"


def safe_append_record(file_path: str, record: Dict) -> Tuple[bool, str]:
    """
    安全追加单条记录到文件
    
    Args:
        file_path: 文件路径
        record: 要追加的记录
        
    Returns:
        (是否成功, 错误信息)
    """
    return safe_write_jsonl(file_path, [record], append=True)


# ========= 置信度分类 =========

def classify_confidence(confidence: float) -> str:
    """
    根据置信度分类
    
    Args:
        confidence: 置信度值 (0-1)
        
    Returns:
        分类: 'high', 'mid', 'low'
    """
    if confidence >= 0.8:
        return 'high'
    elif confidence >= 0.6:
        return 'mid'
    return 'low'


def is_invalid_record(record: Dict[str, Any]) -> bool:
    """
    判断记录是否为无效数据
    无效条件：所有置信度为0，且除ip/status/status_detail外所有值为null/空字符串/空列表
    
    Args:
        record: 要检查的记录
        
    Returns:
        是否无效
    """
    # 检查置信度
    confidence = record.get('confidence', 0)
    
    if confidence != 0:
        return False
    
    # 需要检查的字段
    check_fields = ['vendor', 'model', 'type', 'result_type', 'conclusion']
    list_fields = ['evidence']
    
    # 检查普通字段
    for field in check_fields:
        val = record.get(field)
        if val is not None and val != '':
            return False
    
    # 检查列表字段
    for field in list_fields:
        val = record.get(field, [])
        if val and len(val) > 0:
            return False
    
    return True
