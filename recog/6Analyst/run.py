"""
运行控制器
管理整个分析流程，支持命令行参数
支持多种速度等级，包括并行模式
"""

import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from .config import (
    INPUT_DIR, TEST_INPUT_DIR, CLEANED_DATA_PATH,
    PRODUCT_OUTPUT_PATH, MERGED_OUTPUT_PATH,
    CHECK_OUTPUT_PATH, FINAL_OUTPUT_PATH, MAX_RECORDS, BATCH_SIZE,
    RUN_STATE_PATH, MODEL_NAME, MODEL_PRICES, SPEED_LEVELS, DEFAULT_SPEED_LEVEL,
    TEST_CLEANED_DATA_PATH, TEST_PRODUCT_OUTPUT_PATH,
    TEST_MERGED_OUTPUT_PATH, TEST_CHECK_OUTPUT_PATH, TEST_FINAL_OUTPUT_PATH,
    TEST_RUN_STATE_PATH, TEST_OUTPUT_DIR, LOG_DIR, TEST_LOG_DIR,
    TASK_TYPES, get_task_paths
)
from .data_cleaner import DataCleaner
from .product_analyst import ProductAnalyst
from .check_analyst import CheckAnalyst


# 全局路径变量（根据test模式或任务类型动态设置）
_cleaned_data_path = CLEANED_DATA_PATH
_product_output_path = PRODUCT_OUTPUT_PATH
_merged_output_path = MERGED_OUTPUT_PATH
_check_output_path = CHECK_OUTPUT_PATH
_final_output_path = FINAL_OUTPUT_PATH
_run_state_path = RUN_STATE_PATH
_log_dir = LOG_DIR
_is_test_mode = False
_current_task_type = None  # 当前任务类型 (os/vd/dt)


def set_task_type(task_type: str) -> None:
    """
    设置任务类型，切换到对应的专属输出路径
    
    Args:
        task_type: 任务类型 ('os', 'vd', 'dt')
    """
    global _cleaned_data_path, _product_output_path
    global _merged_output_path, _check_output_path, _final_output_path
    global _run_state_path, _log_dir, _current_task_type
    
    if task_type not in TASK_TYPES:
        raise ValueError(f"未知的任务类型: {task_type}，支持的类型: {list(TASK_TYPES.keys())}")
    
    _current_task_type = task_type
    paths = get_task_paths(task_type)
    
    _cleaned_data_path = paths['cleaned_data']
    _product_output_path = paths['product_output']
    _merged_output_path = paths['merged_output']
    _check_output_path = paths['check_output']
    _final_output_path = paths['final_output']
    _run_state_path = paths['run_state']
    _log_dir = paths['log_dir']
    
    # 确保输出目录存在
    os.makedirs(paths['temp_dir'], exist_ok=True)
    os.makedirs(paths['final_dir'], exist_ok=True)
    os.makedirs(paths['log_dir'], exist_ok=True)
    
    print(f"[INFO] 任务类型: {task_type} ({paths['task_name']})")
    print(f"  输入文件: {paths['input_path']}")
    print(f"  中间输出: {paths['temp_dir']}")
    print(f"  最终输出: {paths['final_dir']}")


def get_current_task_type() -> Optional[str]:
    """获取当前任务类型"""
    return _current_task_type


def set_test_mode(is_test: bool) -> None:
    """设置测试模式，切换输出路径"""
    global _cleaned_data_path, _product_output_path
    global _merged_output_path, _check_output_path, _final_output_path
    global _run_state_path, _log_dir, _is_test_mode
    
    _is_test_mode = is_test
    
    if is_test:
        _cleaned_data_path = TEST_CLEANED_DATA_PATH
        _product_output_path = TEST_PRODUCT_OUTPUT_PATH
        _merged_output_path = TEST_MERGED_OUTPUT_PATH
        _check_output_path = TEST_CHECK_OUTPUT_PATH
        _final_output_path = TEST_FINAL_OUTPUT_PATH
        _run_state_path = TEST_RUN_STATE_PATH
        _log_dir = TEST_LOG_DIR
        
        # 确保测试输出目录存在
        import os
        if not os.path.exists(TEST_OUTPUT_DIR):
            os.makedirs(TEST_OUTPUT_DIR)
    else:
        _cleaned_data_path = CLEANED_DATA_PATH
        _product_output_path = PRODUCT_OUTPUT_PATH
        _merged_output_path = MERGED_OUTPUT_PATH
        _check_output_path = CHECK_OUTPUT_PATH
        _final_output_path = FINAL_OUTPUT_PATH
        _run_state_path = RUN_STATE_PATH
        _log_dir = LOG_DIR


def get_output_paths() -> dict:
    """获取当前模式下的所有输出路径"""
    # 如果设置了任务类型，使用任务专属的 temp_dir
    if _current_task_type and _current_task_type in TASK_TYPES:
        task_paths = get_task_paths(_current_task_type)
        temp_dir = task_paths['temp_dir']
    else:
        # 默认情况：使用 cleaned_data 的父目录
        temp_dir = os.path.dirname(_cleaned_data_path) if _cleaned_data_path else None
    
    return {
        'cleaned_data': _cleaned_data_path,
        'product_output': _product_output_path,
        'merged_output': _merged_output_path,
        'check_output': _check_output_path,
        'final_output': _final_output_path,
        'run_state': _run_state_path,
        'log_dir': _log_dir,
        'temp_dir': temp_dir,
        'is_test_mode': _is_test_mode
    }


# 全局速度等级控制
_current_speed_level: Union[int, str] = DEFAULT_SPEED_LEVEL
_speed_level_lock = threading.Lock()


def get_speed_config() -> Dict:
    """获取当前速度等级配置"""
    with _speed_level_lock:
        return SPEED_LEVELS.get(_current_speed_level, SPEED_LEVELS[DEFAULT_SPEED_LEVEL])


def get_current_speed_level() -> Union[int, str]:
    """获取当前速度等级"""
    with _speed_level_lock:
        return _current_speed_level


def set_speed_level(level: Union[int, str]) -> None:
    """设置速度等级"""
    global _current_speed_level
    with _speed_level_lock:
        if level in SPEED_LEVELS:
            _current_speed_level = level


def downgrade_speed_level() -> bool:
    """
    降低速度等级一级
    返回是否成功降级（如果已经是最低等级则返回False）
    """
    global _current_speed_level
    with _speed_level_lock:
        if _current_speed_level == 's':
            _current_speed_level = 6
            return True
        elif isinstance(_current_speed_level, int) and _current_speed_level > 1:
            _current_speed_level -= 1
            return True
        return False


def get_agent_delay() -> float:
    """获取当前agent间隔时间"""
    return get_speed_config()['delay']


def is_parallel_mode() -> bool:
    """是否为并行模式"""
    return get_speed_config()['parallel']


def handle_rate_limit(analyst, batch: List, agent_name: str, log_info_func, log_error_func, 
                      skip_downgrade: bool = False) -> Tuple[List, Dict]:
    """
    处理API调用，包含增强的并发限制重试逻辑
    - 首次限制等待30分钟
    - 后续每次翻倍（60分钟、120分钟...）
    - 触发限制后自动降低速度等级（除非skip_downgrade=True）
    
    返回: (results, batch_stats)
    注意: 
    - 余额不足时返回带有 insufficient_balance=True 的 batch_stats，由调用方处理退出
    - 触发限制时返回带有 triggered_rate_limit=True 的 batch_stats，供调用方判断是否需要降级
    """
    retry_count = 0
    base_wait_minutes = 30
    
    while True:
        results, batch_stats = analyst.process_batch(batch)
        
        # 检查余额不足 - 返回给调用方处理，不在这里退出（避免子线程问题）
        if batch_stats.get('insufficient_balance'):
            log_error_func("余额不足")
            return results, batch_stats
        
        # 检查并发限制或安全限制
        if batch_stats.get('rate_limited') or batch_stats.get('security_limited'):
            retry_count += 1
            wait_minutes = base_wait_minutes * (2 ** (retry_count - 1))  # 30, 60, 120, 240...
            wait_seconds = int(wait_minutes * 60)
            
            # 标记触发了限制（供调用方判断）
            batch_stats['triggered_rate_limit'] = True
            
            # 降低速度等级（仅在非跳过模式下）
            if not skip_downgrade:
                old_level = get_current_speed_level()
                if downgrade_speed_level():
                    new_level = get_current_speed_level()
                    log_info_func(f"触发限制，速度等级从 {old_level} 降至 {new_level}")
            
            limit_type = "安全限制" if batch_stats.get('security_limited') else "并发限制"
            log_info_func(f"检测到{limit_type}，第{retry_count}次重试，等待{wait_minutes}分钟...")
            
            # 显示倒计时等待（包含提示信息，全程在同一行）
            _wait_with_countdown(wait_seconds, f"API{limit_type} 第{retry_count}次重试")
            
            log_info_func(f"重试{agent_name}...")
            continue
        
        # 成功返回
        return results, batch_stats


def _wait_with_countdown(total_seconds: int, reason: str = "等待中"):
    """
    带倒计时显示的等待函数（始终在同一行显示）
    
    Args:
        total_seconds: 总等待秒数
        reason: 等待原因（显示在倒计时前）
    """
    import sys
    
    end_time = time.time() + total_seconds
    
    while True:
        remaining = int(end_time - time.time())
        if remaining <= 0:
            # 清除倒计时行，显示完成信息
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.write(f'[{reason}] 等待完成，继续执行...\n')
            sys.stdout.flush()
            break
        
        # 格式化剩余时间
        hours = remaining // 3600
        minutes = (remaining % 3600) // 60
        seconds = remaining % 60
        
        if hours > 0:
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = f"{minutes:02d}:{seconds:02d}"
        
        # 显示倒计时（使用\r覆盖当前行，不换行）
        sys.stdout.write(f'\r[{reason}] 剩余时间: {time_str}   ')
        sys.stdout.flush()
        
        time.sleep(1)


# 全局任务ID
_current_task_id: int = 1
_task_id_lock = threading.Lock()


def get_current_task_id() -> int:
    """获取当前任务ID"""
    with _task_id_lock:
        return _current_task_id


def set_task_id(task_id: int) -> None:
    """设置任务ID"""
    global _current_task_id
    with _task_id_lock:
        _current_task_id = task_id
    # 同步设置日志任务ID
    from .utils.logger import set_log_task_id
    set_log_task_id(task_id)


def save_run_state(stats: Dict, start_time: float) -> None:
    """保存运行状态到文件"""
    # 获取当前运行配置
    from .run_config import get_current_run_config
    run_config = get_current_run_config()
    
    state = {
        'task_id': get_current_task_id(),
        'last_update': datetime.now().isoformat(),
        'start_time': datetime.fromtimestamp(start_time).isoformat() if start_time else datetime.now().isoformat(),
        'elapsed_seconds': time.time() - start_time if start_time else 0,
        'stats': stats,
        'run_config': run_config  # 保存完整的运行配置
    }
    output_dir = os.path.dirname(_run_state_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(_run_state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def load_run_state() -> Tuple[Dict, float, int]:
    """
    加载上次运行状态
    返回: (stats字典, 已用时间秒数, 任务ID)
    """
    if not os.path.exists(_run_state_path):
        return None, 0, 0
    try:
        with open(_run_state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        return state.get('stats'), state.get('elapsed_seconds', 0), state.get('task_id', 1)
    except (json.JSONDecodeError, IOError):
        return None, 0, 0


def load_run_state_full() -> Optional[Dict]:
    """
    加载完整的运行状态（包括run_config）
    返回: 完整的状态字典，如果不存在则返回None
    """
    if not os.path.exists(_run_state_path):
        return None
    try:
        with open(_run_state_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_run_config(run_config: Dict) -> None:
    """
    保存运行配置到run_state.json
    用于retry功能
    """
    state = load_run_state_full() or {}
    state['run_config'] = run_config
    state['last_update'] = datetime.now().isoformat()
    
    output_dir = os.path.dirname(_run_state_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(_run_state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def extract_failed_ips(input_path: str, final_output_path: str, retry_level: int = 3) -> Tuple[List[str], Dict]:
    """
    提取需要重试的IP列表
    
    重试等级：
    - level=1: 缺失 + 失败
    - level=2: 缺失 + 失败 + 标签不完整
    - level=3: 缺失 + 失败 + 标签不完整 + 不可信
    
    Args:
        input_path: 输入文件或目录路径
        final_output_path: 最终结果文件路径
        retry_level: 重试等级 (1-3)，默认3
    
    Returns:
        (需要重试的IP列表, 统计信息字典)
    """
    # 标签字段映射：输入数据字段 -> 输出结果字段
    LABEL_FIELD_MAPPING = {
        'Device Type': 'type',      # 设备类型
        'OS': 'os',                 # 操作系统
        'Vendor': 'vendor',         # 厂商
    }
    
    # 处理输入路径：如果是目录，查找所有jsonl文件
    input_files = []
    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.endswith('.jsonl'):
                input_files.append(os.path.join(input_path, fname))
    else:
        input_files = [input_path]
    
    # 读取输入文件，同时检测标签字段
    input_ips = set()
    detected_label_fields = set()  # 检测到的输入标签字段
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ip = next(iter(obj.keys()))
                    data = obj[ip]
                    input_ips.add(ip)
                    
                    # 检测标签字段（从OS字段或Vendor字段）
                    if 'OS' in data and isinstance(data['OS'], dict):
                        os_info = data['OS']
                        for field in ['Device Type', 'OS']:
                            if field in os_info and os_info[field]:
                                detected_label_fields.add(field)
                    
                    if 'Vendor' in data:
                        vendor_info = data['Vendor']
                        if isinstance(vendor_info, dict):
                            if vendor_info.get('Vendor'):
                                detected_label_fields.add('Vendor')
                        elif vendor_info:  # 字符串形式
                            detected_label_fields.add('Vendor')
                            
                except (json.JSONDecodeError, StopIteration):
                    continue
    
    # 确定需要检查的输出字段
    required_output_fields = []
    for input_field in detected_label_fields:
        if input_field in LABEL_FIELD_MAPPING:
            required_output_fields.append(LABEL_FIELD_MAPPING[input_field])
    
    # 读取结果文件
    result_ips = {}  # ip -> record
    if os.path.exists(final_output_path):
        with open(final_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    ip = record.get('ip')
                    if ip:
                        result_ips[ip] = record
                except json.JSONDecodeError:
                    continue
    
    # 统计信息
    stats = {
        'retry_level': retry_level,
        'total_input': len(input_ips),
        'total_results': len(result_ips),
        'detected_label_fields': list(detected_label_fields),
        'required_output_fields': required_output_fields,
        'missing': 0,
        'failed': 0,
        'incomplete': 0,  # 标签字段有空值
        'untrusted': 0,
        'total_retry': 0,
    }
    
    # 提取需要重试的IP
    retry_ips = []
    
    for ip in input_ips:
        need_retry = False
        
        if ip not in result_ips:
            # 情况1：不在结果文件中（所有等级都重试）
            need_retry = True
            stats['missing'] += 1
        else:
            record = result_ips[ip]
            status = record.get('status', '')
            confidence = record.get('confidence', 0)
            
            if 'failed' in status.lower():
                # 情况2：状态为failed（所有等级都重试）
                need_retry = True
                stats['failed'] += 1
            else:
                # 独立检查每个条件
                
                # 检查标签字段是否有空值
                is_incomplete = False
                for field in required_output_fields:
                    value = record.get(field)
                    if value is None or value == '' or value == 'null':
                        is_incomplete = True
                        break
                
                # 检查置信度是否不可信
                is_untrusted = confidence < 0.6
                
                # 独立统计（每个level独立计数）
                if is_incomplete:
                    stats['incomplete'] += 1
                if is_untrusted:
                    stats['untrusted'] += 1
                
                # 根据retry_level决定是否重试（或关系）
                if (is_incomplete and retry_level >= 2) or (is_untrusted and retry_level >= 3):
                    need_retry = True
        
        if need_retry:
            retry_ips.append(ip)
    
    stats['total_retry'] = len(retry_ips)
    
    return retry_ips, stats


def clear_run_state() -> None:
    """清除运行状态文件"""
    if os.path.exists(_run_state_path):
        os.remove(_run_state_path)


def update_run_state_from_results() -> Dict:
    """
    根据结果文件更新run_state.json中的统计信息
    
    Returns:
        更新后的统计信息
    """
    # 从结果文件统计
    result_stats = load_stats_from_results(_final_output_path)
    
    # 统计结果文件中的记录数
    processed_count = 0
    if os.path.exists(_final_output_path):
        with open(_final_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    processed_count += 1
    
    # 读取现有的run_state
    run_state = {}
    if os.path.exists(_run_state_path):
        with open(_run_state_path, 'r', encoding='utf-8') as f:
            run_state = json.load(f)
    
    # 更新stats
    if 'stats' not in run_state:
        run_state['stats'] = {}
    
    run_state['stats'].update({
        'processed_records': processed_count,
        'high_conf': result_stats['high_conf'],
        'mid_conf': result_stats['mid_conf'],
        'low_conf': result_stats['low_conf'],
        'error_count': result_stats['error_count'],
        'verified_count': result_stats['verified_count'],
        'adjusted_count': result_stats['adjusted_count'],
        'rejected_count': result_stats['rejected_count'],
    })
    
    # 更新时间戳
    run_state['last_update'] = datetime.now().isoformat()
    
    # 保存
    with open(_run_state_path, 'w', encoding='utf-8') as f:
        json.dump(run_state, f, ensure_ascii=False, indent=2)
    
    return run_state['stats']


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str = None) -> float:
    """
    根据模型价格计算费用（单位：元）
    价格单位：元/1K token
    
    Args:
        input_tokens: 输入token数
        output_tokens: 输出token数
        model_name: 模型名称，如果为None则使用默认MODEL_NAME
    """
    actual_model = model_name if model_name else MODEL_NAME
    prices = MODEL_PRICES.get(actual_model, MODEL_PRICES.get('default'))
    input_cost = input_tokens * prices['input'] / 1000
    output_cost = output_tokens * prices['output'] / 1000
    return input_cost + output_cost


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='6Analyst - 网络资产数据分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python run_6analyst.py               # 默认执行完整流程，处理input中全部数据
  python run_6analyst.py --test        # 使用test文件夹中的测试数据
  python run_6analyst.py --all          # 执行完整流程
  python run_6analyst.py --clean-only   # 仅执行数据清洗（简称 --cdo）
  python run_6analyst.py --product-only # 仅执行产品分析
  python run_6analyst.py --check-only   # 仅执行结果校验
  python run_6analyst.py --no-check     # 跳过校验步骤
  python run_6analyst.py --max-records 10  # 限制处理10条记录
  python run_6analyst.py --extract-device # 提取含device/OS的数据
  python run_6analyst.py --show         # 生成HTML报告并打开浏览器
  python run_6analyst.py --show_raw_data # 生成原始数据统计报告（raw_data.md）
  python run_6analyst.py --clean-log    # 清理旧日志，合并最近任务日志
  python run_6analyst.py --speed-level 3 # 使用中速模式（间隔1秒）
  python run_6analyst.py --speed-level 6 # 使用极速模式（无间隔，默认）
  python run_6analyst.py --calculate-cost # 计算Token和费用开销
  python run_6analyst.py --calculate-cost --batch-size 5 # 使用5条/批计算费用
  python run_6analyst.py --calculate-cost --datanum 100000 # 估算10万条数据的开销
  python run_6analyst.py --calculate-cost --file-cost # 基于原始输入文件计算开销
  python run_6analyst.py --concat       # 合并分析结果与原始数据，输出model_train.jsonl

数据清洗流程:
  默认流程: 原始数据 -> 清洗内容 -> 信息熵排序 -> 取前75%% -> cleaned_data.jsonl
  python run_6analyst.py --entropy-ratio 0.8  # 保留前80%%高信息熵数据
  python run_6analyst.py --no-entropy         # 跳过信息熵筛选，保留全部数据
  
  厂商平衡采样（在--max-records下保持厂商比例）:
  python run_6analyst.py --max-records 1000 -s  # 难度分级采样（Easy 80% + Normal 10% + Hard 10%）
  python run_6analyst.py --max-records 1000 -s --difficulty-ratio 0.7,0.2,0.1  # 自定义比例（70:20:10）
  python run_6analyst.py --max-records 1000 -s --max-vendor-ratio 0.5  # 类别内单厂商上限50%
  
  均匀采样（在熵排序后按固定间隔采样）:
  python run_6analyst.py --max-records 2000 -u  # 熵排序后均匀采样2000条（间隔=总量/2000）
  python run_6analyst.py --max-records 1000 -u --entropy-ratio 0.9  # 先保留前90%%高熵数据，再均匀采样1000条

重试模式 (--retry):
  python run_6analyst.py --retry        # 默认等级3，重试所有问题IP
  python run_6analyst.py --retry 1      # 等级1: 仅重试缺失+失败
  python run_6analyst.py --retry 2      # 等级2: 重试缺失+失败+标签不完整
  python run_6analyst.py --retry 3      # 等级3: 重试所有问题IP（默认）
  python run_6analyst.py --retry -y     # 自动确认，跳过交互式确认
  
  重试等级说明:
  - 等级1 (最低): 缺失 + 失败
  - 等级2 (中等): 缺失 + 失败 + 标签不完整
  - 等级3 (高强度): 缺失 + 失败 + 标签不完整 + 不可信

信息熵排序与采样 (--isort):
  python run_6analyst.py --isort 0.8                  # 按信息熵排序，保留前80%数据
  python run_6analyst.py --isort 0.8 -u --max-records 1000 # 熵排序后均匀采样1000条
  
  注意：-s 厂商平衡采样会自动跳过信息熵筛选，保留100%数据

提示词管理 (--prompt):
  python run_6analyst.py --prompt --list              # 列出所有可用提示词及费用
  python run_6analyst.py --prompt --update            # 更新所有提示词的费用信息
  python run_6analyst.py --prompt -p p1               # 产品Agent使用p1提示词
  python run_6analyst.py --prompt -c c1               # 校验Agent使用c1提示词
  python run_6analyst.py --prompt -p p3 -c c3         # 全部使用v3版本
  python run_6analyst.py --prompt --expected 35       # 计算35元预算下的推荐提示词长度
  python run_6analyst.py --prompt --expected 50 --multi  # 多模型对比报告

配置中心:
  python run_6analyst.py --config                     # 打开6Analyst配置中心（可视化配置）

模型配置:
  python run_6analyst.py --model deepseek-v3.2       # 所有Agent使用同一模型
  python run_6analyst.py --p-model gpt-4o-mini --c-model deepseek-v3.2
                                                      # 为不同Agent配置不同模型
  python run_6analyst.py --prompt -p p1 --p-model gpt-4o-mini  # 组合提示词和模型配置

提示词ID说明:
  default - 使用程序内置的默认提示词
  p1/p2/p3 - 产品Agent提示词 (精简/详细/知识增强)
  c1/c2/c3 - 校验Agent提示词 (精简/详细/严格)

速度等级说明:
  1 - 最慢模式，agent间隔10秒
  2 - 慢速模式，agent间隔3秒
  3 - 中速模式，agent间隔1秒
  4 - 快速模式，agent间隔0.5秒
  5 - 高速模式，agent间隔0.1秒
  6 - 极速模式，无间隔
  s - 并行模式，产品/用途agent并行执行（默认，最高速）

多线程模式:
  python run_6analyst.py -t 10              # 使用10个线程并行处理
  python run_6analyst.py -t 4 --speed-level 6  # 4线程，每线程内部极速模式
  python run_6analyst.py -t 2 --speed-level 3  # 2线程，每线程内部中速模式
  
  多线程模式特点:
  - 多个工作线程同时处理不同批次，大幅提升吞吐量
  - 每个线程内部仍支持speed-level配置（包括并行agent模式）
  - 线程安全的任务分配、日志记录、统计计算
  - 支持断点续传，自动跳过已处理的记录
  - 智能限流处理，触发API限制时全局等待

任务类型模式 (--mt):
  python run_6analyst.py --mt os -t 10      # 处理OS任务，10线程
  python run_6analyst.py --mt vd -t 10      # 处理Vendor任务，10线程
  python run_6analyst.py --mt dt -t 10      # 处理DeviceType任务，10线程
  
  任务类型说明:
  - os: 操作系统识别任务，输入 os_input_data.jsonl，输出到 output/os/
  - vd: 厂商识别任务，输入 vendor_input_data.jsonl，输出到 output/vendor/
  - dt: 设备类型识别任务，输入 devicetype_input_data.jsonl，输出到 output/devicetype/
  
  输出目录结构:
  - output/{task}/temp/  中间文件（cleaned_data, product_analysis等）
  - output/{task}/final/ 最终结果（final_analysis.jsonl）

默认配置:
  - 输入文件夹: 6Analyst/data/input/（自动读取所有json文件）
  - 测试文件夹: 6Analyst/data/input/test/（--test模式使用）
  - 中间结果: 6Analyst/data/output/ 文件夹
  - 最终结果: final_analysis.jsonl（当前目录）
  - 批处理大小: 每次向大模型提交3条数据
  - 默认速度等级: s（并行模式）
  - 遇到并发限制自动降级并等待重试
        '''
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', default=True,
                       help='执行完整流程（清洗+产品分析+用途分析+汇总）[默认]')
    group.add_argument('--clean-only', '--cdo', action='store_true',
                       help='仅执行数据清洗')
    group.add_argument('--product-only', action='store_true',
                       help='仅执行产品分析（需要已有清洗数据）')
    group.add_argument('--check-only', action='store_true',
                       help='仅执行结果校验（需要已有汇总结果）')
    
    parser.add_argument('--test', action='store_true',
                        help='使用测试数据集（读取test文件夹）')
    parser.add_argument('--max-records', type=int, default=None,
                        help='最大处理条数（默认: 全部）')
    parser.add_argument('--input', type=str, default=None,
                        help=f'输入路径，可以是文件夹或文件（默认: {INPUT_DIR}）')
    parser.add_argument('--no-check', action='store_true',
                        help='跳过结果校验步骤')
    parser.add_argument('--entropy-ratio', type=float, default=None,
                        help='数据清洗时的信息熵筛选比例（默认: 0.75，即保留前75%%高信息熵数据；设为1.0则不筛选）')
    parser.add_argument('--no-entropy', action='store_true',
                        help='跳过信息熵筛选，保留全部清洗后的数据')
    parser.add_argument('--extract-device', action='store_true',
                        help='提取含有nmap识别的device/OS数据到单独文件')
    parser.add_argument('--isort', type=float, default=None,
                        help='信息熵排序筛选：按信息熵排序，保留前N%%数据（如 --isort 0.8 保留前80%%）')
    parser.add_argument('-s', '--vendor-balance', action='store_true', dest='vendor_balance',
                        help='难度分级采样模式（与--max-records配合）：Easy 80%% + Normal 10%% + Hard 10%%，类别内单厂商上限60%%，自动跳过信息熵筛选')
    parser.add_argument('--difficulty-ratio', type=str, default=None, dest='difficulty_ratio',
                        help='自定义难度比例，格式: easy,normal,hard（如 0.7,0.2,0.1 表示70%%:20%%:10%%）')
    parser.add_argument('--max-vendor-ratio', type=float, default=None, dest='max_vendor_ratio',
                        help='类别内单厂商上限（默认0.6，即60%%）')
    parser.add_argument('-u', '--uniform-sample', action='store_true', dest='uniform_sample',
                        help='均匀采样模式：在熵排序后按固定间隔采样（与--max-records配合使用）')
    parser.add_argument('--acc', action='store_true',
                        help='计算准确率：读取输入标签和输出结果，计算并生成准确率报告')
    parser.add_argument('--float', action='store_true',
                        help='准确率浮动分析：对比高信息熵（前1%%）和低信息熵（后1%%）数据的准确率差异')
    parser.add_argument('--float-h', '-fh', action='store_true', dest='float_high',
                        help='仅处理高信息熵组（前1%%）数据')
    parser.add_argument('--float-l', '-fl', action='store_true', dest='float_low',
                        help='仅处理低信息熵组（后1%%）数据')
    parser.add_argument('--float-acc', action='store_true', dest='float_acc',
                        help='计算浮动分析结果的准确率：读取已有的高/低信息熵组结果，计算并对比准确率')
    parser.add_argument('--e-information', action='store_true', dest='e_information',
                        help='信息熵分析：计算所有数据的信息熵，输出排序结果和分布柱状图')
    parser.add_argument('--no-acc', action='store_true',
                        help='跳过自动准确率计算（默认完整运行后会自动计算）')
    parser.add_argument('--concat', action='store_true',
                        help='合并分析结果与原始数据，输出到model_train.jsonl（用于模型训练）')
    parser.add_argument('-d', '--separate-labels', action='store_true', dest='separate_labels',
                        help='分离OS和Device Type标签，生成单独的训练数据（与--concat配合使用）')
    parser.add_argument('--restart', action='store_true',
                        help='清除已有结果，重新开始处理（禁用断点续传）')
    parser.add_argument('--retry', type=int, nargs='?', const=3, default=None,
                        metavar='LEVEL', choices=[1, 2, 3],
                        help='重试模式：1=缺失+失败, 2=1+标签不完整, 3=2+不可信(默认)')
    parser.add_argument('-y', '--yes', action='store_true', dest='auto_confirm',
                        help='自动确认，跳过交互式确认（与--retry配合使用）')
    parser.add_argument('--show', action='store_true',
                        help='生成HTML报告并在浏览器中打开')
    parser.add_argument('--show_raw_data', action='store_true',
                        help='生成原始数据统计报告（raw_data.md），包括字段组成和有效值统计')
    parser.add_argument('--update-state', action='store_true',
                        help='根据结果文件更新run_state.json中的统计信息')
    parser.add_argument('--clean-polluted', action='store_true',
                        help='清除结果文件中被标签数据污染的记录')
    parser.add_argument('--calculate-cost', action='store_true',
                        help='计算Token和费用开销')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='每批次包含的数据条数（用于费用计算，默认: 3）')
    parser.add_argument('--datanum', type=int, default=None,
                        help='估算指定数据量的开销（如 --datanum 100000）')
    parser.add_argument('--file-cost', action='store_true',
                        help='基于原始输入文件计算开销（与--calculate-cost配合使用）')
    
    # 提示词管理参数组 (--prompt 为入口)
    parser.add_argument('--prompt', action='store_true',
                        help='提示词管理模式（配合 --list/--update/-p/-u/-c/--expected 使用）')
    parser.add_argument('--list', action='store_true', dest='list_prompts',
                        help='列出所有可用的提示词及费用信息（与--prompt配合）')
    parser.add_argument('--update', action='store_true', dest='update_prompts',
                        help='更新所有提示词的费用信息（与--prompt配合）')
    parser.add_argument('-p', type=str, default='default', dest='product_prompt',
                        help='产品Agent提示词ID: default/p1/p2/p3')
    parser.add_argument('-c', type=str, default='default', dest='check_prompt',
                        help='校验Agent提示词ID: default/c1/c2/c3')
    parser.add_argument('--expected', type=float, default=None,
                        help='预算金额（元），计算推荐提示词长度（与--prompt配合）')
    parser.add_argument('--precision', type=int, default=3,
                        help='精度（10^-n元），与--prompt --expected配合，默认: 3')
    parser.add_argument('--multi', action='store_true',
                        help='多模型对比模式（与--prompt --expected配合）')
    parser.add_argument('--config', action='store_true', dest='config_center',
                        help='打开6Analyst配置中心（单任务流程可视化配置）')
    parser.add_argument('--mt-config', action='store_true', dest='mt_config_center',
                        help='打开多任务配置中心（多任务模式可视化配置）')
    
    # 模型配置参数
    parser.add_argument('--model', type=str, default=None,
                        help='全局模型名称（所有Agent使用同一模型）')
    parser.add_argument('--p-model', type=str, default=None, dest='product_model',
                        help='产品Agent使用的模型名称')
    parser.add_argument('--c-model', type=str, default=None, dest='check_model',
                        help='校验Agent使用的模型名称')
    
    parser.add_argument('--clean-log', action='store_true',
                        help='清理旧日志，保留并合并最近一次任务的日志')
    parser.add_argument('-f', '--force-clean-log', action='store_true',
                        help='强制合并所有日志并清理（与--clean-log配合使用）')
    parser.add_argument('--speed-level', type=str, default=str(DEFAULT_SPEED_LEVEL),
                        help='速度等级: 1(最慢)-6(极速)。默认: 6')
    parser.add_argument('-t', type=int, default=1, dest='num_threads',
                        help='线程数（如 -t 10 使用10个线程并行处理）。默认: 1（单线程）')
    parser.add_argument('--mt', type=str, default=None, choices=['os', 'vd', 'dt', 'mg', 'all'],
                        help='任务类型: os(操作系统), vd(厂商), dt(设备类型), mg(融合标签), all(并行执行全部)。使用专属输入和输出路径')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式：输出详细的解析失败信息')
    parser.add_argument('--log-level', type=str, default='warning',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='控制台日志级别（默认: warning）。也可通过环境变量 LOG_LEVEL 设置')
    
    # ===== 实验模块参数 =====
    parser.add_argument('--exp', type=int, default=None,
                        help='运行实验模块（如 --exp 1 运行实验1配置页面）')
    parser.add_argument('--run', action='store_true', dest='exp_run',
                        help='直接运行实验（与--exp配合使用）')
    parser.add_argument('--eval', action='store_true', dest='exp_eval',
                        help='评估实验准确率（与--exp配合使用）')
    parser.add_argument('--exp-show', action='store_true', dest='exp_show',
                        help='生成实验HTML报告并在浏览器中打开（与--exp配合使用）')
    parser.add_argument('--exp-restart', action='store_true', dest='exp_restart',
                        help='从头开始运行实验，清除已有结果（与--exp --run配合使用）')
    parser.add_argument('--save', type=str, default=None, dest='exp_save',
                        help='保存代码为zip（--save main 保存主项目，--save exp1 保存实验1）')
    parser.add_argument('--model1', type=str, default=None,
                        help='实验组1使用的模型')
    parser.add_argument('--model2', type=str, default=None,
                        help='实验组2使用的模型')
    parser.add_argument('--model3', type=str, default=None,
                        help='实验组3使用的模型')
    parser.add_argument('-et', '--exp-threads', type=int, default=4, dest='exp_threads',
                        help='实验模块每组使用的线程数（默认: 4）')
    
    # exp2专用参数
    parser.add_argument('--cln', action='store_true', dest='exp2_cln',
                        help='仅运行清洗数据组（与--exp 2配合使用）')
    parser.add_argument('--raw', action='store_true', dest='exp2_raw',
                        help='仅运行原始数据组（与--exp 2配合使用）')
    
    return parser.parse_args()


def is_invalid_record(record: Dict) -> bool:
    """
    判断记录是否为无效数据
    无效条件：所有置信度为0，且除ip/status/status_detail外所有值为null/空字符串/空列表
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


def clean_invalid_data() -> Dict:
    """
    清除中间结果和最终结果中的无效数据
    """
    print("\n" + "=" * 60)
    print("清理无效数据")
    print("=" * 60)
    
    stats = {
        'files_processed': 0,
        'total_records': 0,
        'invalid_records': 0,
        'valid_records': 0
    }
    
    # 需要清理的文件列表
    files_to_clean = [
        _product_output_path,
        _merged_output_path,
        _check_output_path,
        _final_output_path
    ]
    
    for file_path in files_to_clean:
        if not os.path.exists(file_path):
            print(f"  跳过（不存在）: {file_path}")
            continue
        
        valid_records = []
        invalid_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    stats['total_records'] += 1
                    
                    if is_invalid_record(record):
                        invalid_count += 1
                        stats['invalid_records'] += 1
                    else:
                        valid_records.append(record)
                        stats['valid_records'] += 1
                except json.JSONDecodeError:
                    continue
        
        # 重写文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in valid_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        stats['files_processed'] += 1
        print(f"  {file_path}: 删除 {invalid_count} 条无效数据")
    
    print(f"\n清理完成:")
    print(f"  处理文件: {stats['files_processed']}")
    print(f"  总记录: {stats['total_records']}")
    print(f"  无效记录: {stats['invalid_records']}")
    print(f"  有效记录: {stats['valid_records']}")
    
    return stats


def extract_device_data(input_path: str = None, max_records: int = None) -> Dict:
    """
    提取含有nmap识别的device/OS数据到单独文件
    用于后续对比实验
    """
    import glob
    
    print("\n" + "=" * 60)
    print("提取 Device/OS 数据")
    print("=" * 60)
    
    input_path = input_path or INPUT_DIR
    output_path = "6Analyst/data/output/device_os_data.jsonl"
    
    stats = {
        'total_records': 0,
        'extracted_records': 0,
        'input_files': 0
    }
    
    # 获取输入文件
    if os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        input_files = sorted(json_files + jsonl_files)
    elif os.path.isfile(input_path):
        input_files = [input_path]
    else:
        print(f"[ERROR] 输入路径不存在: {input_path}")
        return stats
    
    stats['input_files'] = len(input_files)
    print(f"发现 {len(input_files)} 个输入文件")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    extracted = []
    current_count = 0
    
    for filepath in input_files:
        if max_records is not None and current_count >= max_records:
            break
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if max_records is not None and current_count >= max_records:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                stats['total_records'] += 1
                current_count += 1
                
                try:
                    obj = json.loads(line)
                    ip = next(iter(obj.keys()))
                    data = obj[ip]
                    
                    # 检查是否有OS字段
                    if 'OS' in data:
                        os_info = data['OS']
                        record = {
                            'ip': ip,
                            'device_type': os_info.get('Device Type'),
                            'os': os_info.get('OS')
                        }
                        extracted.append(record)
                        stats['extracted_records'] += 1
                        
                except (json.JSONDecodeError, StopIteration):
                    continue
    
    # 保存提取结果
    if extracted:
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in extracted:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n提取完成:")
    print(f"  总记录: {stats['total_records']}")
    print(f"  含Device/OS: {stats['extracted_records']} ({stats['extracted_records']/max(stats['total_records'],1)*100:.1f}%)")
    print(f"  输出文件: {output_path}")
    
    return stats


def run_cleaner(input_path: str = None, max_records: int = None, entropy_ratio: float = None, 
                vendor_balance: bool = False, uniform_sample: bool = False,
                difficulty_ratios: Dict[str, float] = None,
                max_vendor_ratio: float = None) -> Dict:
    """
    执行数据清洗（包含信息熵排序和筛选）
    
    默认流程：
    1. 读取原始数据（input目录下的jsonl文件）
    2. 清洗数据内容，删除无用数据
    3. 按信息熵排序
    4. 取前75%数据（可通过entropy_ratio参数调整）
    5. 输出到cleaned_data.jsonl
    
    难度分级采样流程（当 vendor_balance=True 且 max_records 指定时）：
    1. 先根据标签中的厂商信息进行难度分级预采样
       - Easy（易识别）: MikroTik, Keenetic - 默认80%
       - Normal（较易识别）: Cisco, Juniper - 默认10%
       - Hard（难识别）: 其他厂商 - 默认10%
    2. 在每个难度级别内部，单个厂商不超过该级别的60%（可配置）
    3. 自动将 entropy_ratio 设为 1.0，跳过信息熵筛选
    4. 对采样后的数据进行清洗
    5. 输出全部清洗后的数据
    
    均匀采样流程（当 uniform_sample=True 且 max_records 指定时）：
    1. 清洗数据
    2. 按信息熵排序
    3. 计算采样间隔 = 数据总量 / max_records（向下取整）
    4. 从第1条开始，每隔 interval 条采样一条
    
    Args:
        input_path: 输入路径，默认为INPUT_DIR
        max_records: 最大处理条数，None表示全部
        entropy_ratio: 信息熵筛选比例，None使用默认值(0.75)，设为1.0则不筛选
        vendor_balance: 是否启用难度分级采样（仅在指定max_records时生效）
        uniform_sample: 是否启用均匀采样（仅在指定max_records时生效，在熵排序后执行）
        difficulty_ratios: 难度比例字典，如 {'easy': 0.8, 'normal': 0.1, 'hard': 0.1}
        max_vendor_ratio: 类别内单厂商上限（默认0.6，即60%）
    """
    from .utils.logger import log_info
    from .entropy_sorter import isort, presample_by_difficulty, uniform_sample as do_uniform_sample
    from .config import DEFAULT_ENTROPY_RATIO, DEFAULT_MAX_VENDOR_RATIO_PER_CATEGORY
    
    input_path = input_path or INPUT_DIR
    
    # 保存原始输入路径（用于准确率计算，因为预采样文件可能没有标签）
    original_input_path = input_path
    
    # 保存原始的max_records值（用于均匀采样）
    original_max_records = max_records
    
    # 确定信息熵筛选比例
    if entropy_ratio is None:
        entropy_ratio = DEFAULT_ENTROPY_RATIO
    
    # 确定单厂商上限
    if max_vendor_ratio is None:
        max_vendor_ratio = DEFAULT_MAX_VENDOR_RATIO_PER_CATEGORY
    
    print("\n" + "=" * 60)
    print("步骤 1: 数据清洗")
    print("=" * 60)
    
    # 获取临时文件目录（使用任务专属的temp目录）
    temp_dir = os.path.dirname(_cleaned_data_path)
    if temp_dir and not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # === 新增：难度分级预采样 ===
    temp_presampled_path = None
    if vendor_balance and max_records is not None:
        # 显示配置信息
        if difficulty_ratios is None:
            from .config import DEFAULT_DIFFICULTY_RATIOS
            difficulty_ratios = DEFAULT_DIFFICULTY_RATIOS
        
        easy_pct = difficulty_ratios.get('easy', 0.8) * 100
        normal_pct = difficulty_ratios.get('normal', 0.1) * 100
        hard_pct = difficulty_ratios.get('hard', 0.1) * 100
        
        print(f"  [难度分级采样] 目标: {max_records}条")
        print(f"    Easy(易识别): {easy_pct:.0f}% | Normal(较易): {normal_pct:.0f}% | Hard(难识别): {hard_pct:.0f}%")
        print(f"    类别内单厂商上限: {max_vendor_ratio*100:.0f}%")
        print(f"  [难度分级采样] 自动跳过信息熵筛选，保留100%数据")
        
        # 难度分级模式下，强制跳过信息熵筛选
        entropy_ratio = 1.0
        
        # 执行预采样
        sampled_lines, presample_stats = presample_by_difficulty(
            input_path=input_path,
            target_count=max_records,
            difficulty_ratios=difficulty_ratios,
            max_single_vendor_ratio=max_vendor_ratio
        )
        
        if not sampled_lines:
            print(f"  [警告] 预采样失败，回退到普通采样模式")
        else:
            # 将预采样结果写入临时文件
            temp_presampled_path = os.path.join(temp_dir, 'temp_presampled_data.jsonl')
            with open(temp_presampled_path, 'w', encoding='utf-8') as f:
                for line in sampled_lines:
                    f.write(line + '\n')
                f.flush()
                os.fsync(f.fileno())
            
            # 更新input_path为预采样文件
            input_path = temp_presampled_path
            # 清除max_records限制（因为已经采样过了）
            max_records = None
            
            print(f"  [难度分级采样] 完成: {len(sampled_lines)}条")
            for difficulty in ['easy', 'normal', 'hard']:
                count = presample_stats['difficulty_sampled'].get(difficulty, 0)
                print(f"    {difficulty.upper()}: {count}条")
    
    # 如果不需要信息熵筛选，直接清洗输出
    if entropy_ratio is None or entropy_ratio >= 1.0:
        cleaner = DataCleaner(input_path, _cleaned_data_path)
        stats = cleaner.run(max_records)
        log_info(f"数据清洗完成: {stats}")
        print(f"  清洗完成: {stats.get('processed_records', 0)}条, 压缩率{stats.get('compression_ratio', 0)*100:.1f}%")
        
        # 保存原始输入路径（用于准确率计算）
        stats['original_input_path'] = original_input_path
        
        # 清理临时预采样文件
        if temp_presampled_path and os.path.exists(temp_presampled_path):
            os.remove(temp_presampled_path)
        
        return stats
    
    # 需要信息熵筛选的流程
    print(f"  输入路径: {input_path}")
    print(f"  信息熵筛选: 保留前{entropy_ratio*100:.0f}%数据")
    
    # 计算总步骤数（根据是否启用均匀采样）
    # 使用original_max_records判断，因为max_records可能被厂商平衡采样修改
    total_steps = 4 if (uniform_sample and original_max_records is not None) else 3
    step_num = 0
    
    # 步骤1: 清洗数据到临时文件（使用任务专属的temp目录）
    step_num += 1
    temp_cleaned_path = os.path.join(temp_dir, 'temp_cleaned_data.jsonl')
    cleaner = DataCleaner(input_path, temp_cleaned_path)
    clean_stats = cleaner.run(max_records if not uniform_sample else None)  # 均匀采样时不限制清洗数量
    print(f"  [{step_num}/{total_steps}] 清洗完成: {clean_stats.get('processed_records', 0)}条, 压缩率{clean_stats.get('compression_ratio', 0)*100:.1f}%")
    
    # 步骤2: 读取清洗后的数据
    step_num += 1
    cleaned_records = []
    with open(temp_cleaned_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cleaned_records.append(record)
            except json.JSONDecodeError:
                continue
    
    original_count = len(cleaned_records)
    print(f"  [{step_num}/{total_steps}] 信息熵排序: {original_count}条数据")
    
    # 步骤3: 按信息熵排序并筛选
    step_num += 1
    sorted_records = isort(cleaned_records, ratio=entropy_ratio)
    filtered_count = len(sorted_records)
    print(f"  [{step_num}/{total_steps}] 筛选完成: 保留{filtered_count}条 ({filtered_count/max(original_count,1)*100:.1f}%)")
    
    # 步骤4: 均匀采样（如果启用且指定了max_records）
    # 使用original_max_records，因为max_records可能被厂商平衡采样修改为None
    uniform_stats = None
    sampled_ips_path = None
    if uniform_sample and original_max_records is not None:
        step_num += 1
        if original_max_records < filtered_count:
            sorted_records, uniform_stats = do_uniform_sample(sorted_records, original_max_records, verbose=False)
            sampled_count = len(sorted_records)
            interval = uniform_stats.get('interval', 1)
            print(f"  [{step_num}/{total_steps}] 均匀采样: {sampled_count}条 (间隔{interval}, 目标{original_max_records})")
            filtered_count = sampled_count
            
            # 保存采样的IP列表到中间文件
            sampled_ips = []
            for record in sorted_records:
                if len(record) == 1:
                    ip = next(iter(record.keys()))
                    sampled_ips.append(ip)
            
            sampled_ips_path = os.path.join(temp_dir, 'sampled_ips.json')
            with open(sampled_ips_path, 'w', encoding='utf-8') as f:
                json.dump({'ips': sampled_ips, 'count': len(sampled_ips)}, f, ensure_ascii=False)
            print(f"  采样IP列表已保存: {sampled_ips_path}")
        else:
            print(f"  [{step_num}/{total_steps}] 均匀采样: 跳过 (目标{original_max_records} >= 数据量{filtered_count})")
    
    # 写入最终文件
    with open(_cleaned_data_path, 'w', encoding='utf-8') as f:
        for record in sorted_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    
    # 删除临时文件
    if os.path.exists(temp_cleaned_path):
        os.remove(temp_cleaned_path)
    if temp_presampled_path and os.path.exists(temp_presampled_path):
        os.remove(temp_presampled_path)
    
    # 更新统计信息
    stats = clean_stats.copy()
    stats['entropy_filtered'] = True
    stats['entropy_ratio'] = entropy_ratio
    stats['original_count'] = original_count
    stats['filtered_count'] = filtered_count
    stats['processed_records'] = filtered_count  # 更新为筛选后的数量
    
    # 保存原始输入路径（用于准确率计算）
    stats['original_input_path'] = original_input_path
    
    # 添加均匀采样统计
    if uniform_stats:
        stats['sampled_ips_path'] = sampled_ips_path
        stats['uniform_sample'] = uniform_stats
    
    log_info(f"数据清洗完成（含信息熵筛选）: {stats}")
    print(f"\n数据清洗完成: {filtered_count}条 (原{original_count}条, 保留{entropy_ratio*100:.0f}%)")
    
    return stats


def run_product_analyst(max_records: int = None, model_name: str = None) -> Dict:
    """执行产品分析（串行模式）"""
    print("\n" + "=" * 60)
    print("步骤 2: 产品型号分析")
    print("=" * 60)
    
    analyst = ProductAnalyst(_cleaned_data_path, _product_output_path, model_name)
    stats = analyst.run(max_records)
    
    print(f"\n产品分析完成:")
    print(f"  成功: {stats['successful_records']}/{stats['processed_records']}")
    print(f"  Token超限批次: {stats['token_exceeded_batches']}")
    print(f"  输出文件: {_product_output_path}")
    
    return stats


def load_processed_ips(file_path: str) -> set:
    """从结果文件中加载已处理的IP列表"""
    processed = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ip = obj.get('ip')
                    if ip:
                        processed.add(ip)
                except json.JSONDecodeError:
                    continue
    return processed


def load_stats_from_results(file_path: str) -> Dict:
    """
    从结果文件中统计置信度分类和校验分类
    返回统计字典
    """
    stats = {
        'high_conf': 0, 'mid_conf': 0, 'low_conf': 0,
        'error_count': 0,
        'verified_count': 0, 'adjusted_count': 0, 'rejected_count': 0
    }
    
    if not os.path.exists(file_path):
        return stats
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                status = obj.get('status', '')
                
                # 错误状态计入error_count
                if status in ('failed', 'check_failed'):
                    stats['error_count'] += 1
                    continue
                
                # 置信度分类
                conf = obj.get('confidence', 0)
                if conf >= 0.8:
                    stats['high_conf'] += 1
                elif conf >= 0.6:
                    stats['mid_conf'] += 1
                else:
                    stats['low_conf'] += 1
                
                # 校验分类
                validation_status = obj.get('validation_status', '')
                if validation_status == 'verified':
                    stats['verified_count'] += 1
                elif validation_status == 'adjusted':
                    stats['adjusted_count'] += 1
                elif validation_status == 'rejected':
                    stats['rejected_count'] += 1
                    
            except json.JSONDecodeError:
                continue
    
    return stats


def append_result(file_path: str, result: Dict) -> None:
    """追加单条结果到文件，确保立即写入磁盘"""
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def normalize_result_fields(record: Dict) -> Dict:
    """
    标准化结果字段值：
    - 所有属性字段未知时统一为 null
    - 说明性字段（evidence, conclusion）：空值 -> null
    """
    # 关键结果字段：空字符串/"unknown" -> null
    key_fields = ['vendor', 'model', 'os', 'type', 'result_type']
    for field in key_fields:
        val = record.get(field)
        if val is None or val == '' or val == 'null' or val == 'unknown':
            record[field] = None
    
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


def run_pipeline(max_records: int = None, skip_check: bool = False,
                 product_model: str = None, check_model: str = None) -> Dict:
    """
    流水线处理：按batch为单位，依次执行产品分析、校验，实时写入结果
    支持断点续传：跳过已处理的IP，恢复上次统计数据
    支持多种速度等级
    
    Args:
        max_records: 最大处理记录数
        skip_check: 是否跳过校验步骤
        product_model: 产品Agent使用的模型
        check_model: 校验Agent使用的模型
    """
    from .utils.logger import log_info, log_debug, log_error, log_exception
    
    start_time = time.time()
    
    # 显示当前速度等级
    speed_level = get_current_speed_level()
    speed_config = get_speed_config()
    log_info(f"速度等级: {speed_level} ({speed_config['desc']})")
    print(f"[INFO] 速度等级: {speed_level} ({speed_config['desc']})")
    
    log_info("=" * 60)
    log_info("开始流水线分析")
    log_info("=" * 60)
    
    # 初始化分析器
    product_analyst = ProductAnalyst(_cleaned_data_path, _product_output_path, product_model)
    check_analyst = CheckAnalyst(_merged_output_path, _check_output_path, _final_output_path, check_model)
    
    # 加载已处理的IP（断点续传）
    processed_ips = load_processed_ips(_final_output_path)
    
    # 尝试加载上次运行状态
    prev_stats, prev_elapsed, prev_task_id = load_run_state()
    
    # 设置任务ID（如果有上次的任务ID则继续使用）
    if prev_task_id > 0:
        set_task_id(prev_task_id)
    
    task_id = get_current_task_id()
    log_info(f"任务ID: {task_id}")
    
    if processed_ips:
        log_info(f"[断点续传] 已处理 {len(processed_ips)} 条记录")
        if prev_stats:
            log_info(f"[断点续传] 恢复上次统计数据，已用时 {prev_elapsed:.1f}s")
    
    # 加载数据
    all_records = product_analyst.load_records(max_records)
    
    # 过滤已处理的记录
    records = [r for r in all_records if r['ip'] not in processed_ips]
    total_records = len(records)
    skipped_records = len(all_records) - total_records
    
    log_info(f"总记录: {len(all_records)}, 跳过: {skipped_records}, 待处理: {total_records}")
    
    if total_records == 0:
        print(f"所有记录已处理完成！（跳过 {skipped_records} 条已处理数据）")
        # 清除运行状态文件
        clear_run_state()
        return {
            'total_records': len(all_records), 'skipped_records': skipped_records,
            'processed_records': 0, 'successful_records': 0, 'execution_time_seconds': 0
        }
    
    # 统计信息：从结果文件实时统计，不依赖状态文件
    if skipped_records > 0:
        # 从结果文件统计置信度分类和校验分类
        result_stats = load_stats_from_results(_final_output_path)
        stats = {
            'total_records': len(all_records),
            'skipped_records': skipped_records,
            'processed_records': skipped_records,  # 以结果文件为准
            'error_count': result_stats['error_count'],
            'high_conf': result_stats['high_conf'],
            'mid_conf': result_stats['mid_conf'],
            'low_conf': result_stats['low_conf'],
            'input_tokens': 0,  # 累计token（从状态文件恢复）
            'output_tokens': 0,
            'this_run_input_tokens': 0,  # 本次运行token
            'this_run_output_tokens': 0,
            'realtime_avg_input': 0,  # 实时平均input token/条
            'realtime_avg_output': 0,  # 实时平均output token/条
            'verified_count': result_stats['verified_count'],
            'adjusted_count': result_stats['adjusted_count'],
            'rejected_count': result_stats['rejected_count'],
            'execution_time_seconds': 0
        }
        # 尝试从状态文件恢复累计token、实时平均和用时
        if prev_stats:
            accumulated_time = prev_elapsed
            stats['input_tokens'] = prev_stats.get('input_tokens', 0)
            stats['output_tokens'] = prev_stats.get('output_tokens', 0)
            stats['realtime_avg_input'] = prev_stats.get('realtime_avg_input', 0)
            stats['realtime_avg_output'] = prev_stats.get('realtime_avg_output', 0)
        else:
            accumulated_time = 0
        prev_processed_count = skipped_records
        
        # 使用上次的实时平均计算总估费用（完成全部数据所需的总开销估计）
        total_estimated_input = stats['realtime_avg_input'] * stats['total_records']
        total_estimated_output = stats['realtime_avg_output'] * stats['total_records']
        # 使用 product_model 计算费用（如果指定了的话）
        cost = calculate_cost(stats['input_tokens'], stats['output_tokens'], product_model)
        total_estimated_cost = calculate_cost(total_estimated_input, total_estimated_output, product_model)
        
        print(f"[断点续传] 恢复统计: 已处理{skipped_records}条, "
              f"高置信度{stats['high_conf']} 中置信度{stats['mid_conf']} 不可信{stats['low_conf']} 错误{stats['error_count']}")
        print(f"[断点续传] 累计Token: {stats['input_tokens']}/{stats['output_tokens']} 费用: CNY{cost:.4f} | "
              f"上次实时平均: {stats['realtime_avg_input']:.0f}/{stats['realtime_avg_output']:.0f} 总估: CNY{total_estimated_cost:.4f}")
    else:
        stats = {
            'total_records': len(all_records), 'skipped_records': skipped_records,
            'processed_records': 0, 'error_count': 0,
            'high_conf': 0, 'mid_conf': 0, 'low_conf': 0,
            'input_tokens': 0, 'output_tokens': 0,  # 累计token
            'this_run_input_tokens': 0, 'this_run_output_tokens': 0,  # 本次运行token
            'realtime_avg_input': 0, 'realtime_avg_output': 0,  # 实时平均token/条
            'verified_count': 0, 'adjusted_count': 0, 'rejected_count': 0,
            'execution_time_seconds': 0
        }
        accumulated_time = 0
        prev_processed_count = 0
    
    # 确保输出目录存在
    for path in [_product_output_path, _merged_output_path, 
                 _check_output_path, _final_output_path]:
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    total_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    
    print()  # 空行分隔
    
    # 分批处理
    for i in range(0, total_records, BATCH_SIZE):
        batch = records[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        batch_start = time.time()
        
        log_debug(f"批次 {batch_num}/{total_batches}: {[r['ip'] for r in batch]}")
        
        try:
            # === 步骤1: 执行产品分析 ===
            log_debug("执行产品分析...")
            product_results, product_batch_stats = handle_rate_limit(
                product_analyst, batch, "产品分析", log_info, log_error)
            
            # 检查余额不足
            if product_batch_stats.get('insufficient_balance'):
                log_error("余额不足，保存状态后退出")
                print("\n[ERROR] 余额不足，正在保存当前状态...")
                save_run_state(stats, start_time)
                print("[INFO] 状态已保存，程序终止。请充值后重新运行以继续处理。")
                sys.exit(1)
            
            stats['input_tokens'] += product_batch_stats.get('input_token_count', 0)
            stats['output_tokens'] += product_batch_stats.get('output_token_count', 0)
            stats['this_run_input_tokens'] += product_batch_stats.get('input_token_count', 0)
            stats['this_run_output_tokens'] += product_batch_stats.get('output_token_count', 0)
            
            log_debug(f"产品分析: {len(product_results) if product_results else 0} 条, Token: {product_batch_stats.get('token_count', 0)}")
            
            # 保存产品结果
            if product_results:
                for res in product_results:
                    append_result(_product_output_path, res)
                    model_val = res.get('model')
                    if isinstance(model_val, list) and len(model_val) > 0:
                        top = model_val[0]
                        model_str = f"{top[0]}({top[1]:.1f})" if isinstance(top, list) and len(top) >= 2 else str(top)
                        if len(model_val) > 1:
                            model_str += f"+{len(model_val)-1}"
                    else:
                        model_str = model_val
                    log_debug(f"产品: {res.get('ip')} -> {res.get('vendor')} {model_str} (conf:{res.get('confidence', 0)})")
            
            # === 步骤2: 构建合并结果（仅产品分析） ===
            product_map = {r.get('ip'): r for r in (product_results or [])}
            
            merged_results = []
            failed_ips = []  # 记录前置agent失败的IP
            for r in batch:
                ip = r['ip']
                product = product_map.get(ip, {})
                
                # 检查产品分析是否有结果
                if not product:
                    # 产品分析失败
                    log_error(f"IP {ip} 产品分析失败")
                    failed_result = {
                        'ip': ip,
                        'status': 'failed',
                        'status_detail': 'product_agent_failed'
                    }
                    append_result(_final_output_path, failed_result)
                    failed_ips.append(ip)
                    stats['error_count'] += 1
                    continue
                
                merged_record = {
                    'ip': ip,
                    'vendor': product.get('vendor'), 'model': product.get('model'),
                    'os': product.get('os'),
                    'type': product.get('type'), 'result_type': product.get('result_type'),
                    'confidence': product.get('confidence', 0), 'evidence': product.get('evidence', []),
                    'conclusion': product.get('conclusion', '')
                }
                # 标准化字段值
                merged_record = normalize_result_fields(merged_record)
                merged_results.append(merged_record)
                append_result(_merged_output_path, merged_record)
            
            # === 步骤3: 校验 ===
            if not skip_check:
                # 按速度等级等待（并行模式无间隔）
                delay = get_agent_delay()
                if delay > 0:
                    time.sleep(delay)
                
                log_debug("执行校验...")
                check_batch = [{'ip': r['ip'], 'raw': json.dumps(r, ensure_ascii=False), 'data': r} 
                              for r in merged_results]
                check_results, check_batch_stats = handle_rate_limit(
                    check_analyst, check_batch, "校验分析", log_info, log_error)
                
                # 检查余额不足
                if check_batch_stats.get('insufficient_balance'):
                    log_error("余额不足，保存状态后退出")
                    print("\n[ERROR] 余额不足，正在保存当前状态...")
                    save_run_state(stats, start_time)
                    print("[INFO] 状态已保存，程序终止。请充值后重新运行以继续处理。")
                    sys.exit(1)
                
                stats['input_tokens'] += check_batch_stats.get('input_token_count', 0)
                stats['output_tokens'] += check_batch_stats.get('output_token_count', 0)
                stats['this_run_input_tokens'] += check_batch_stats.get('input_token_count', 0)
                stats['this_run_output_tokens'] += check_batch_stats.get('output_token_count', 0)
                
                if check_results:
                    # 创建IP到校验结果的映射
                    check_map = {r.get('ip'): r for r in check_results}
                    
                    for merged in merged_results:
                        ip = merged['ip']
                        check_res = check_map.get(ip)
                        
                        if check_res:
                            validation_status = check_res.get('validation_status', 'unknown')
                            
                            if validation_status == 'verified':
                                stats['verified_count'] += 1
                            elif validation_status == 'adjusted':
                                stats['adjusted_count'] += 1
                            elif validation_status == 'rejected':
                                stats['rejected_count'] += 1
                            
                            append_result(_check_output_path, check_res)
                            final_result = check_analyst._build_final_result(check_res, merged)
                            
                            # 数据完整，状态为done
                            final_result['status'] = 'done'
                            final_result['status_detail'] = 'all_agents_completed'
                            
                            # 统计置信度
                            conf = final_result.get('confidence', 0)
                            if conf >= 0.8:
                                stats['high_conf'] += 1
                            elif conf >= 0.6:
                                stats['mid_conf'] += 1
                            else:
                                stats['low_conf'] += 1
                            
                            append_result(_final_output_path, final_result)
                            log_debug(f"校验: {ip} -> {validation_status}, conf: {conf}")
                        else:
                            # 该IP的校验结果解析失败
                            log_error(f"IP {ip} 校验结果解析失败")
                            print(f"IP {ip} 校验结果解析失败")
                            
                            # DEBUG模式：输出详细信息
                            from .config import DEBUG_MODE
                            if DEBUG_MODE:
                                print(f"[DEBUG] IP {ip} 校验结果解析失败")
                                print(f"[DEBUG] 期望的IP列表: {[m['ip'] for m in merged_results]}")
                                print(f"[DEBUG] 实际解析到的IP列表: {list(check_map.keys())}")
                                if check_results:
                                    print(f"[DEBUG] 解析到的校验结果数量: {len(check_results)}")
                                    for i, res in enumerate(check_results):
                                        print(f"[DEBUG] 结果{i}: {json.dumps(res, ensure_ascii=False)[:200]}...")
                                else:
                                    print(f"[DEBUG] 校验结果为空或解析完全失败")
                            
                            merged['validation_status'] = 'parse_error'
                            merged['status'] = 'check_failed'
                            merged['status_detail'] = 'check_result_parse_error'
                            append_result(_final_output_path, merged)
                            
                            # 解析失败计入错误
                            stats['error_count'] += 1
                else:
                    log_error(f"批次 {batch_num} 校验全部失败")
                    for merged in merged_results:
                        merged['validation_status'] = 'check_error'
                        merged['status'] = 'check_failed'
                        merged['status_detail'] = 'check_batch_failed'
                        append_result(_final_output_path, merged)
                    
                    # 批次全部失败，按实际条数计入错误
                    stats['error_count'] += len(merged_results)
            else:
                for merged in merged_results:
                    merged['validation_status'] = 'skipped'
                    merged['status'] = 'done'
                    merged['status_detail'] = 'check_skipped'
                    
                    # 统计置信度
                    conf = merged.get('confidence', 0)
                    if conf >= 0.8:
                        stats['high_conf'] += 1
                    elif conf >= 0.6:
                        stats['mid_conf'] += 1
                    else:
                        stats['low_conf'] += 1
                    append_result(_final_output_path, merged)
            
            stats['processed_records'] += len(batch)
            
        except Exception as e:
            log_exception(f"批次 {batch_num} 处理出错: {e}")
            # 批次处理异常，为每条记录添加错误状态
            for r in batch:
                error_result = {
                    'ip': r['ip'],
                    'validation_status': 'exception',
                    'status': 'failed',
                    'status_detail': f'batch_exception: {str(e)}'
                }
                append_result(_final_output_path, error_result)
            stats['error_count'] += len(batch)
        
        # 计算并打印进度
        batch_time = time.time() - batch_start
        # total_processed 就是 processed_records（已包含断点续传恢复的数量）
        total_processed = stats['processed_records']
        progress = total_processed / stats['total_records'] * 100
        
        # 本次运行处理的条数
        this_run_processed = stats['processed_records'] - prev_processed_count
        
        # 计算实时平均token（基于本次运行的数据）
        realtime_avg_input = stats['this_run_input_tokens'] / max(this_run_processed, 1)
        realtime_avg_output = stats['this_run_output_tokens'] / max(this_run_processed, 1)
        
        # 更新实时平均到stats（用于保存到状态文件）
        stats['realtime_avg_input'] = realtime_avg_input
        stats['realtime_avg_output'] = realtime_avg_output
        
        elapsed = time.time() - start_time  # 本次运行用时
        
        # 计算费用（基于累计token，使用 product_model）
        cost = calculate_cost(stats['input_tokens'], stats['output_tokens'], product_model)
        
        # 计算总估费用：实时平均 × 数据总数（完成全部数据所需的总开销估计）
        total_estimated_input = realtime_avg_input * stats['total_records']
        total_estimated_output = realtime_avg_output * stats['total_records']
        total_estimated_cost = calculate_cost(total_estimated_input, total_estimated_output, product_model)
        
        # 计算剩余时间（基于本次运行的平均速度）
        remaining_records = stats['total_records'] - total_processed
        if this_run_processed > 0:
            avg_time_per_record = elapsed / this_run_processed
            remaining_time = avg_time_per_record * remaining_records
        else:
            remaining_time = 0
        
        # 格式化时间显示
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        print(f"进度: {progress:5.1f}% {total_processed:>4}/{stats['total_records']:<4}  "
              f"高置信度: {stats['high_conf']:<3} 中置信度: {stats['mid_conf']:<3} 不可信: {stats['low_conf']:<3} 错误: {stats['error_count']:<3} | "
              f"Verified: {stats['verified_count']:<2} Adjust: {stats['adjusted_count']:<2} Reject: {stats['rejected_count']:<2} | "
              f"Tok/条: {realtime_avg_input:.0f}/{realtime_avg_output:.0f}  CNY{cost:.4f}(总估CNY{total_estimated_cost:.4f})  用时: {format_time(elapsed)} / 剩余: {format_time(remaining_time)}")
        
        # 每批次后保存运行状态（保存累计时间和token用于统计）
        stats['execution_time_seconds'] = elapsed + accumulated_time
        save_run_state(stats, start_time)
    
    # 本次运行用时
    this_run_time = time.time() - start_time
    stats['execution_time_seconds'] = this_run_time + accumulated_time  # 累计总时间保存到stats
    
    # 最终统计
    print("-" * 100)
    # 本次运行处理的条数
    final_this_run_processed = stats['processed_records'] - prev_processed_count
    # 计算实时平均token（基于本次运行的数据）
    realtime_avg_input = stats['this_run_input_tokens'] / max(final_this_run_processed, 1)
    realtime_avg_output = stats['this_run_output_tokens'] / max(final_this_run_processed, 1)
    
    # 更新实时平均到stats（用于保存到状态文件）
    stats['realtime_avg_input'] = realtime_avg_input
    stats['realtime_avg_output'] = realtime_avg_output
    
    # 计算总费用（基于累计token，使用 product_model）
    total_cost = calculate_cost(stats['input_tokens'], stats['output_tokens'], product_model)
    
    # 计算总估费用：实时平均 × 数据总数（完成全部数据所需的总开销估计）
    total_estimated_input = realtime_avg_input * stats['total_records']
    total_estimated_output = realtime_avg_output * stats['total_records']
    total_estimated_cost = calculate_cost(total_estimated_input, total_estimated_output, product_model)
    
    # 格式化本次用时
    if this_run_time < 60:
        time_str = f"{this_run_time:.1f}s"
    elif this_run_time < 3600:
        time_str = f"{this_run_time/60:.1f}m"
    else:
        time_str = f"{this_run_time/3600:.1f}h"
    
    print(f"完成: {stats['processed_records']}条(本次{final_this_run_processed}条) | 高置信度: {stats['high_conf']} 中置信度: {stats['mid_conf']} "
          f"不可信: {stats['low_conf']} 错误: {stats['error_count']} | "
          f"Verified: {stats['verified_count']} Adjust: {stats['adjusted_count']} Reject: {stats['rejected_count']}")
    print(f"Tok/条: {realtime_avg_input:.0f}/{realtime_avg_output:.0f} | "
          f"费用: CNY{total_cost:.4f} (总估: CNY{total_estimated_cost:.4f}) | 总用时: {time_str}")
    
    log_info(f"流水线完成: {stats}")
    
    # 任务完成后清除运行状态文件
    if stats['processed_records'] + stats['skipped_records'] >= stats['total_records']:
        clear_run_state()
    
    return stats


def check_cleaned_data_exists() -> bool:
    """检查清洗数据是否存在"""
    return os.path.exists(_cleaned_data_path)


def check_final_output_exists() -> bool:
    """检查汇总结果是否存在"""
    return os.path.exists(_merged_output_path)


def run_check_analyst(max_records: int = None, model_name: str = None) -> Dict:
    """执行结果校验"""
    print("\n" + "=" * 60)
    print("步骤 4: 结果校验与修正")
    print("=" * 60)
    
    analyst = CheckAnalyst(_merged_output_path, _check_output_path, _final_output_path, model_name)
    stats, eval_stats = analyst.run(max_records)
    
    # 打印评估报告
    analyst.print_evaluation_report()
    
    print(f"\n校验完成:")
    print(f"  校验: {stats['successful_records']}/{stats['processed_records']}")
    print(f"  Token超限批次: {stats['token_exceeded_batches']}")
    print(f"  校验详情: {_check_output_path}")
    print(f"  最终结果: {_final_output_path}")
    
    # 合并统计信息
    stats['evaluation'] = eval_stats
    
    return stats


def merge_results(product_path: str, output_path: str) -> Dict:
    """
    处理产品分析结果，准备给check agent
    """
    print("\n" + "=" * 60)
    print("步骤 2: 结果汇总")
    print("=" * 60)
    
    stats = {
        'product_records': 0,
        'merged_records': 0,
    }
    
    # 加载产品分析结果
    product_results = {}
    if os.path.exists(product_path):
        with open(product_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    ip = record.get('ip')
                    if ip:
                        product_results[ip] = record
                        stats['product_records'] += 1
                except json.JSONDecodeError:
                    continue
    
    print(f"加载产品分析: {stats['product_records']} 条")
    
    # 处理结果
    merged = []
    for ip, product in product_results.items():
        merged_record = {
            'ip': ip,
            # Product分析字段
            'vendor': product.get('vendor'),
            'model': product.get('model'),
            'os': product.get('os'),
            'type': product.get('type'),
            'result_type': product.get('result_type'),
            'confidence': product.get('confidence', 0),
            'evidence': product.get('evidence', []),
            'conclusion': product.get('conclusion', ''),
        }
        
        # 标准化字段值
        merged_record = normalize_result_fields(merged_record)
        merged.append(merged_record)
        stats['merged_records'] += 1
    
    # 保存汇总结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"汇总完成: {stats['merged_records']} 条")
    print(f"输出文件: {output_path}")
    
    return stats


def print_summary(all_stats: Dict) -> None:
    """打印执行摘要（简洁版）"""
    from .utils.logger import log_info
    log_info(f"执行摘要: {all_stats}")
    
    # 生成统计报告
    try:
        from .tools.generate_stats_report import generate_stats_report
        
        # 确定任务类型和输入路径
        task_type = get_current_task_type()
        
        # 确定输入路径（用于准确率计算）
        if task_type:
            from .config import get_task_paths
            paths = get_task_paths(task_type)
            input_path = paths['input_path']
        else:
            input_path = INPUT_DIR
        
        # 确定模型名称（优先从run_config获取，其次从pipeline获取）
        model_name = (
            all_stats.get('run_config', {}).get('product_model') or
            all_stats.get('pipeline', {}).get('model_name') or
            MODEL_NAME
        )
        
        # 生成报告
        generate_stats_report(
            task_type=task_type,
            input_path=input_path,
            output_dir=None,  # 使用默认输出目录（项目根目录）
            model_name=model_name
        )
    except Exception as e:
        print(f"\n[警告] 生成统计报告失败: {e}")
        log_info(f"生成统计报告失败: {e}")


def generate_html_report() -> str:
    """生成HTML报告并返回index.html路径"""
    from .tools.generate_html_report import (
        load_final_results, load_original_data, load_cleaned_data,
        generate_index_page, generate_detail_page,
        OUTPUT_DIR, CLEANED_DATA_PATH
    )
    from .accuracy_calculator import (
        load_input_labels, load_output_results, calculate_accuracy
    )
    
    print("\n" + "=" * 50)
    print("生成HTML报告")
    print("=" * 50)
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 加载数据
    results = load_final_results(FINAL_OUTPUT_PATH)
    if not results:
        print(f"[ERROR] 无分析结果: {FINAL_OUTPUT_PATH}")
        return None
    
    cleaned_data = load_cleaned_data(CLEANED_DATA_PATH)
    original_data = load_original_data(INPUT_DIR)
    print(f"加载 {len(results)} 条分析结果, {len(cleaned_data)} 条清洗数据, {len(original_data)} 条原始数据")
    
    # 计算准确率
    accuracy_stats = None
    try:
        print("计算准确率...")
        # load_input_labels 返回 (labels, available_labels) 元组
        labels, available_labels = load_input_labels(INPUT_DIR)
        # load_output_results 返回 (results, skipped_invalid, skipped_low_conf) 元组
        output_results, _, _ = load_output_results(FINAL_OUTPUT_PATH)
        if labels and output_results:
            accuracy_stats = calculate_accuracy(labels, output_results, available_labels)
            if 'vendor_accuracy' in accuracy_stats:
                vendor_rate = accuracy_stats['vendor_accuracy']['rate'] * 100
                print(f"  厂商准确率: {vendor_rate:.1f}% ({accuracy_stats['vendor_accuracy']['correct']}/{accuracy_stats['vendor_accuracy']['total']})")
            if 'type_accuracy' in accuracy_stats:
                type_rate = accuracy_stats['type_accuracy']['rate'] * 100
                print(f"  类型准确率: {type_rate:.1f}% ({accuracy_stats['type_accuracy']['correct']}/{accuracy_stats['type_accuracy']['total']})")
            if 'os_accuracy' in accuracy_stats:
                os_rate = accuracy_stats['os_accuracy']['rate'] * 100
                print(f"  OS准确率: {os_rate:.1f}% ({accuracy_stats['os_accuracy']['correct']}/{accuracy_stats['os_accuracy']['total']})")
        else:
            print("  [跳过] 无标签数据或输出结果")
    except Exception as e:
        print(f"  [警告] 准确率计算失败: {e}")
    
    # 生成页面
    index_path = generate_index_page(results, OUTPUT_DIR, accuracy_stats=accuracy_stats)
    for r in results:
        ip = r.get('ip', 'unknown')
        cleaned = cleaned_data.get(ip, {})
        original = original_data.get(ip, {})
        generate_detail_page(ip, r, cleaned, original, OUTPUT_DIR)
    
    print(f"报告已生成: {index_path}")
    return index_path


def run_all_tasks_parallel(args):
    """
    并行执行所有任务类型（os/vd/dt）
    
    每个任务使用独立的进程，使用各自专属的提示词和输出路径。
    共享参数：--model, -t, --restart, --speed-level 等
    
    Args:
        args: 命令行参数
    """
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .config import SINGLE_TASK_TYPES, TASK_TYPES, get_task_paths
    
    print("\n" + "=" * 60)
    print("ALL 模式：并行执行所有任务类型")
    print("=" * 60)
    
    # 检查各任务的输入文件
    task_info = []
    for task_type in SINGLE_TASK_TYPES:
        paths = get_task_paths(task_type)
        task_name = paths['task_name']  # 任务全称（vendor/os/devicetype）
        input_path = paths['input_path']
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            task_info.append((task_type, task_name, count, input_path))
            print(f"  [{task_name.upper()}] {count:,} 条数据 - {input_path}")
        else:
            print(f"  [{task_name.upper()}] [警告] 输入文件不存在: {input_path}")
    
    if not task_info:
        print("\n[错误] 没有可执行的任务，请检查输入文件")
        return
    
    total_count = sum(info[2] for info in task_info)
    print(f"\n  总计: {total_count:,} 条数据，{len(task_info)} 个任务")
    print("=" * 60)
    
    # 构建各任务的命令行参数
    def build_task_command(task_type: str, task_name: str) -> list:
        """构建单个任务的命令行参数"""
        cmd = [sys.executable, '-m', '6Analyst.run', '--mt', task_type]
        
        # 提示词：使用任务全称对应的专属提示词
        cmd.extend(['-p', task_name, '-c', task_name])
        
        # 共享参数
        if args.model:
            cmd.extend(['--model', args.model])
        if args.product_model:
            cmd.extend(['--p-model', args.product_model])
        if args.check_model:
            cmd.extend(['--c-model', args.check_model])
        if args.num_threads and args.num_threads > 1:
            cmd.extend(['-t', str(args.num_threads)])
        if args.speed_level:
            cmd.extend(['--speed-level', str(args.speed_level)])
        if args.max_records:
            cmd.extend(['--max-records', str(args.max_records)])
        if args.restart:
            cmd.append('--restart')
        if args.no_check:
            cmd.append('--no-check')
        if args.debug:
            cmd.append('--debug')
        if hasattr(args, 'log_level') and args.log_level:
            cmd.extend(['--log-level', args.log_level])
        
        return cmd
    
    # 执行单个任务
    def run_task(task_type: str, task_name: str) -> tuple:
        """执行单个任务，返回 (task_name, success, message)"""
        cmd = build_task_command(task_type, task_name)
        print(f"\n[{task_name.upper()}] 启动任务: {' '.join(cmd)}")
        
        try:
            # 设置子进程环境变量，确保 UTF-8 编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # 使用 subprocess 执行，实时输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            # 实时输出带任务标签的日志
            for line in process.stdout:
                print(f"[{task_name.upper()}] {line}", end='')
            
            process.wait()
            
            if process.returncode == 0:
                return (task_name, True, "完成")
            else:
                return (task_name, False, f"退出码: {process.returncode}")
                
        except Exception as e:
            return (task_name, False, str(e))
    
    # 并行执行所有任务
    print(f"\n[INFO] 启动 {len(task_info)} 个并行任务...")
    
    results = {}
    with ThreadPoolExecutor(max_workers=len(task_info)) as executor:
        futures = {executor.submit(run_task, info[0], info[1]): info[1] for info in task_info}
        
        for future in as_completed(futures):
            task_name, success, message = future.result()
            results[task_name] = (success, message)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("ALL 模式执行结果汇总")
    print("=" * 60)
    
    success_count = 0
    for task_type, task_name, count, _ in task_info:
        if task_name in results:
            success, message = results[task_name]
            status = "成功" if success else "失败"
            print(f"  [{task_name.upper()}] {status} - {message}")
            if success:
                success_count += 1
    
    print(f"\n  总计: {success_count}/{len(task_info)} 个任务成功")
    print("=" * 60)


def main():
    """主函数"""
    from .utils.logger import setup_logger, log_info, set_console_log_level
    from .utils.error_logger import clear_error_log, get_error_log_path
    from . import config
    import webbrowser
    
    args = parse_args()
    
    print("6Analyst - 网络资产数据分析工具")
    
    # 打开6Analyst配置中心
    if args.config_center:
        from .tools.generate_config_center import generate_config_center_page
        generate_config_center_page()
        return
    
    # 打开多任务配置中心
    if args.mt_config_center:
        from .tools.generate_multi_task_config import generate_config_center_page
        generate_config_center_page()
        return
    
    # ===== 保存实验数据 =====
    if args.exp_save:
        from .exp.exp_saver import save_experiment, save_main
        # 解析保存目标
        exp_save = args.exp_save.lower().strip()
        if exp_save == 'exp1' or exp_save == '1':
            save_experiment(1)
        elif exp_save == 'main':
            save_main()
        else:
            print(f"[错误] 未知的保存目标: {args.exp_save}")
            print("目前支持: --save exp1, --save main")
            sys.exit(1)
        return
    
    # ===== 实验模块入口 =====
    if args.exp is not None:
        if args.exp == 1:
            if args.exp_eval:
                # 评估实验准确率
                from .exp.accuracy_evaluator import evaluate_all_groups
                print("\n开始评估实验准确率...")
                evaluate_all_groups()
                return
            elif args.exp_show:
                # 生成HTML报告并打开
                from .exp.exp1_html_report import open_html_report
                open_html_report()
                return
            elif args.exp_run:
                # 运行实验1 - 使用复用主项目模块的运行器
                if not args.model1 or not args.model2 or not args.model3:
                    print("[错误] 运行实验需要指定所有三个模型参数: --model1, --model2, --model3")
                    print("示例: python run.py --exp 1 --run --model1 gpt-4o --model2 deepseek-v3 --model3 claude-3-5-sonnet-20241022")
                    sys.exit(1)
                
                from .exp.exp1_main_runner import Exp1MainRunner
                from .exp.exp_config import Exp1Config
                
                config = Exp1Config(num_threads=args.exp_threads)
                runner = Exp1MainRunner(
                    args.model1, 
                    args.model2, 
                    args.model3, 
                    config,
                    restart=args.exp_restart
                )
                runner.run()
            else:
                # 打开实验配置页面
                from .exp.config_page_generator import Exp1ConfigPageGenerator
                generator = Exp1ConfigPageGenerator()
                generator.save_and_open()
            return
        elif args.exp == 2:
            # exp2: 数据清洗效果对比实验
            
            # 检查 --cln 和 --raw 互斥
            if args.exp2_cln and args.exp2_raw:
                print("[错误] --cln 和 --raw 不能同时使用")
                print("使用 --cln 仅运行清洗数据组，使用 --raw 仅运行原始数据组")
                sys.exit(1)
            
            from .exp.exp2_main_runner import run_exp2
            
            # 确定运行哪些组
            run_cleaned = not args.exp2_raw  # 如果指定--raw则不运行cleaned
            run_raw = not args.exp2_cln      # 如果指定--cln则不运行raw
            
            # 获取模型参数（优先使用--model，否则使用默认值）
            model = args.model if args.model else "deepseek-v3.2"
            
            # 获取线程数（优先使用-t，否则使用--exp-threads，否则使用默认值24）
            num_threads = args.num_threads if args.num_threads else (args.exp_threads if args.exp_threads else 24)
            
            # 获取最大记录数
            max_records = args.max_records if args.max_records else 5000
            
            run_exp2(
                model=model,
                max_records=max_records,
                num_threads=num_threads,
                restart=args.exp_restart or args.restart,
                run_cleaned=run_cleaned,
                run_raw=run_raw,
                vendor_balance=args.vendor_balance,
            )
            return
        else:
            print(f"[错误] 未知的实验编号: {args.exp}")
            print("目前支持的实验: --exp 1, --exp 2")
            sys.exit(1)
    
    # 提示词管理模式 (--prompt)
    if args.prompt:
        # 更新提示词费用信息
        if args.update_prompts:
            from .prompts import update_all_prompt_costs, print_update_result, print_prompts_list
            print("\n正在计算所有提示词的费用信息...")
            stats = update_all_prompt_costs(batch_size=args.batch_size)
            print_update_result(stats)
            print("\n更新后的提示词列表:")
            print_prompts_list()
            return
        
        # 列出可用提示词
        if args.list_prompts:
            from .prompts import print_prompts_list
            print_prompts_list()
            return
        
        # 如果指定了提示词ID但没有其他操作，设置提示词并继续执行
        has_prompt_selection = (args.product_prompt != 'default' or 
                               args.usage_prompt != 'default' or 
                               args.check_prompt != 'default')
        
        # 提示词长度计算
        if args.expected is not None:
            if args.multi:
                # 多模型对比模式
                from .cost_calculator import run_multi_model_prompt_calculation
                # 确定输入路径
                if args.input:
                    input_path = args.input
                elif args.test:
                    input_path = TEST_INPUT_DIR
                else:
                    input_path = INPUT_DIR
                run_multi_model_prompt_calculation(
                    budget=args.expected,
                    precision=args.precision,
                    batch_size=args.batch_size,
                    input_path=input_path,
                    generate_report=True
                )
            else:
                # 单模型模式
                from .cost_calculator import run_prompt_length_calculation
                if args.input:
                    input_path = args.input
                elif args.test:
                    input_path = TEST_INPUT_DIR
                else:
                    input_path = INPUT_DIR
                run_prompt_length_calculation(
                    budget=args.expected,
                    precision=args.precision,
                    batch_size=args.batch_size,
                    input_path=input_path
                )
            return
        
        # 如果只是选择提示词，不退出，继续执行分析
        if not has_prompt_selection:
            # 没有指定任何操作，显示帮助
            from .prompts import print_prompts_list
            print_prompts_list()
            return
    
    # 设置各Agent的提示词ID
    from .product_analyst import ProductAnalyst
    from .check_analyst import CheckAnalyst
    
    ProductAnalyst.set_prompt_id(args.product_prompt)
    CheckAnalyst.set_prompt_id(args.check_prompt)
    
    # 显示当前使用的提示词
    if args.product_prompt != 'default' or args.check_prompt != 'default':
        print(f"[提示词配置] Product: {args.product_prompt}, Check: {args.check_prompt}")
    
    # 处理模型配置
    # 优先级: --p-model/--c-model > --model > 默认配置
    product_model = args.product_model or args.model or None
    check_model = args.check_model or args.model or None
    
    # 显示当前使用的模型
    if product_model or check_model:
        from .config import MODEL_NAME
        p_model_display = product_model or MODEL_NAME
        c_model_display = check_model or MODEL_NAME
        print(f"[模型配置] Product: {p_model_display}, Check: {c_model_display}")
    
    # 将模型配置存储到 args 中供后续使用
    args._product_model = product_model
    args._check_model = check_model
    
    # 清理日志（独立功能，不初始化日志系统）
    if args.clean_log:
        from .tools.clean_log import clean_logs
        clean_logs(force=args.force_clean_log)
        return
    
    # 设置调试模式
    if args.debug:
        config.DEBUG_MODE = True
        print("[DEBUG] 调试模式已启用")
    
    # 设置控制台日志级别
    log_level = args.log_level
    if args.debug and log_level == 'warning':
        # 调试模式下默认使用 debug 级别
        log_level = 'debug'
    set_console_log_level(log_level)
    
    # 初始化日志（仅在非clean-log模式下）
    logger = setup_logger(console_level=log_level)
    
    # 初始化错误日志（每次运行清空）
    if not args.restart:
        # 非重启模式下保留错误日志（断点续传）
        pass
    else:
        # 重启模式下清空错误日志
        clear_error_log()
        log_info(f"错误日志已清空: {get_error_log_path()}")
    
    log_info("程序启动")
    log_info(f"参数: {vars(args)}")
    log_info(f"错误日志路径: {get_error_log_path()}")
    log_info(f"控制台日志级别: {log_level}")
    if args.debug:
        log_info("调试模式已启用")
    
    # 设置速度等级
    speed_level_str = args.speed_level
    if speed_level_str.isdigit():
        level = int(speed_level_str)
        if level in SPEED_LEVELS:
            set_speed_level(level)
        else:
            print(f"[WARN] 无效的速度等级 {level}，使用默认等级 {DEFAULT_SPEED_LEVEL}")
            set_speed_level(DEFAULT_SPEED_LEVEL)
    else:
        print(f"[WARN] 无效的速度等级 '{speed_level_str}'，使用默认等级 {DEFAULT_SPEED_LEVEL}")
        set_speed_level(DEFAULT_SPEED_LEVEL)
    
    log_info(f"速度等级: {get_current_speed_level()} ({get_speed_config()['desc']})")
    
    # 确定输入路径和输出模式
    # 优先级: --mt > --input > --test > 默认
    if args.mt:
        # 任务类型模式
        if args.mt == 'all':
            # all 模式：并行执行 os/vd/dt 三个任务
            run_all_tasks_parallel(args)
            return
        elif args.mt == 'mg':
            # mg 模式：融合三个标签文件
            from .tools.merge_labels import merge_label_files
            
            print("\n" + "=" * 60)
            print("融合标签模式 (--mt mg)")
            print("=" * 60)
            
            input_dir = args.input or INPUT_DIR
            os_file = os.path.join(input_dir, 'os_input_data.jsonl')
            vendor_file = os.path.join(input_dir, 'vendor_input_data.jsonl')
            devicetype_file = os.path.join(input_dir, 'devicetype_input_data.jsonl')
            output_file = os.path.join(input_dir, 'merged_input_data.jsonl')
            
            # 检查输入文件
            missing_files = []
            for fname, fpath in [('OS', os_file), ('Vendor', vendor_file), ('DeviceType', devicetype_file)]:
                if not os.path.exists(fpath):
                    missing_files.append(fname)
            
            if missing_files:
                print(f"\n[警告] 以下文件不存在: {', '.join(missing_files)}")
                print("将使用存在的文件进行融合")
            
            # 执行融合
            stats = merge_label_files(os_file, vendor_file, devicetype_file, output_file)
            
            print(f"\n融合完成！输出文件: {output_file}")
            print(f"总计: {stats['merged_count']} 条融合记录")
            
            return
        else:
            # 单任务模式：使用专属的输入和输出路径
            set_task_type(args.mt)
            task_paths = get_task_paths(args.mt)
            input_path = task_paths['input_path']
            
            # 检查输入文件是否存在
            if not os.path.exists(input_path):
                print(f"[错误] 任务输入文件不存在: {input_path}")
                return
    elif args.input:
        input_path = args.input
        set_test_mode(False)
    elif args.test:
        input_path = TEST_INPUT_DIR
        set_test_mode(True)
        print(f"[测试模式] 使用测试数据: {input_path}")
        print(f"[测试模式] 输出目录: {get_output_paths()['final_output']}")
    else:
        input_path = INPUT_DIR
        set_test_mode(False)
    
    # 确定处理条数（默认全部，可通过--max-records限制）
    max_records = args.max_records  # None表示全部
    
    # 确定信息熵筛选比例
    if args.no_entropy:
        entropy_ratio = 1.0  # 不筛选
    elif args.entropy_ratio is not None:
        entropy_ratio = args.entropy_ratio
    else:
        entropy_ratio = None  # 使用默认值(0.75)
    
    # 解析难度比例参数
    difficulty_ratios = None
    if args.difficulty_ratio:
        try:
            parts = [float(x.strip()) for x in args.difficulty_ratio.split(',')]
            if len(parts) != 3:
                print(f"[错误] --difficulty-ratio 需要3个值（easy,normal,hard），当前: {len(parts)}个")
                sys.exit(1)
            if abs(sum(parts) - 1.0) > 0.001:
                print(f"[错误] --difficulty-ratio 三个值的和必须为1.0，当前: {sum(parts)}")
                sys.exit(1)
            difficulty_ratios = {
                'easy': parts[0],
                'normal': parts[1],
                'hard': parts[2]
            }
            print(f"[INFO] 使用自定义难度比例: Easy {parts[0]*100:.0f}% | Normal {parts[1]*100:.0f}% | Hard {parts[2]*100:.0f}%")
        except ValueError as e:
            print(f"[错误] --difficulty-ratio 格式错误: {e}")
            print("正确格式: --difficulty-ratio 0.8,0.1,0.1")
            sys.exit(1)
    
    # 获取单厂商上限
    max_vendor_ratio = args.max_vendor_ratio
    
    # 费用计算（独立功能）
    if args.calculate_cost:
        from .cost_calculator import run_cost_calculation
        run_cost_calculation(
            batch_size=args.batch_size, 
            datanum=args.datanum,
            file_cost=args.file_cost,
            input_path=input_path
        )
        return
    
    # 生成HTML报告（独立功能）
    if args.show:
        index_path = generate_html_report()
        if index_path:
            abs_path = os.path.abspath(index_path)
            print(f"正在打开浏览器: {abs_path}")
            webbrowser.open(f"file://{abs_path}")
        return
    
    # 生成原始数据统计报告（独立功能）
    if args.show_raw_data:
        from .tools.generate_raw_data_report import generate_raw_data_report
        generate_raw_data_report(input_path, 'raw_data.md')
        return
    
    # 根据结果文件更新run_state（独立功能）
    if args.update_state:
        print("\n" + "=" * 50)
        print("根据结果文件更新 run_state.json")
        print("=" * 50)
        
        stats = update_run_state_from_results()
        
        print(f"\n更新完成:")
        print(f"  已处理记录: {stats.get('processed_records', 0)}")
        print(f"  高置信度: {stats.get('high_conf', 0)}")
        print(f"  中置信度: {stats.get('mid_conf', 0)}")
        print(f"  低置信度: {stats.get('low_conf', 0)}")
        print(f"  错误: {stats.get('error_count', 0)}")
        print(f"  Verified: {stats.get('verified_count', 0)}")
        print(f"  Adjusted: {stats.get('adjusted_count', 0)}")
        print(f"  Rejected: {stats.get('rejected_count', 0)}")
        print(f"\n已保存到: {_run_state_path}")
        return
    
    # 准确率计算（独立功能）
    if args.acc:
        from .accuracy_calculator import run_accuracy_calculation
        run_accuracy_calculation(
            input_path=input_path,
            output_path=_final_output_path,
            report_path='accuracy_report.md',
            max_records=max_records
        )
        return
    
    # 准确率浮动分析（独立功能）
    if args.float or args.float_high or args.float_low:
        from .float_analysis import run_float_analysis
        # 确定模型
        model = args.model or "deepseek-v3.2"
        
        # 确定处理模式
        if args.float_high:
            mode = 'high'
        elif args.float_low:
            mode = 'low'
        else:
            mode = 'both'
        
        run_float_analysis(
            input_path=input_path,
            model=model,
            num_threads=args.num_threads if args.num_threads > 1 else 24,
            high_ratio=0.01,
            low_ratio=0.01,
            mode=mode
        )
        return
    
    # 浮动分析准确率计算（独立功能）
    if args.float_acc:
        from .float_analysis import run_float_accuracy
        run_float_accuracy()
        return
    
    # 信息熵分析（独立功能）
    if args.e_information:
        from .tools.entropy_analysis import run_entropy_analysis
        run_entropy_analysis(input_path=input_path)
        return
    
    # ===== 设置当前运行配置（用于统计报告生成） =====
    from .run_config import set_current_run_config
    current_config = {
        'input_path': input_path,
        'max_records': max_records,
        'batch_size': args.batch_size,
        'num_threads': args.num_threads,
        'speed_level': get_current_speed_level(),
        'skip_check': args.no_check,
        'entropy_ratio': entropy_ratio,
        'product_prompt': args.product_prompt,
        'check_prompt': args.check_prompt,
        'product_model': product_model,
        'check_model': check_model,
        'test_mode': _is_test_mode,
        'task_type': get_current_task_type(),
        'restart': args.restart,
        'retry_mode': args.retry is not None,
        'retry_level': args.retry if args.retry is not None else None,
        'debug': args.debug,
        'log_level': log_level
    }
    set_current_run_config(current_config)
    
    # 清除被污染的数据（独立功能）
    if args.clean_polluted:
        from .tools.clean_polluted_data import clean_polluted_data
        print("\n" + "=" * 50)
        print("清除被标签数据污染的记录")
        print("=" * 50)
        print(f"输入文件: {_final_output_path}")
        print(f"检测条件: evidence.src 包含 'OS.Device Type', 'OS.OS', 'OS', 'Device Type', 'Vendor'")
        
        total, polluted, kept = clean_polluted_data(_final_output_path)
        
        if polluted > 0:
            # 同时更新 run_state
            print("\n更新 run_state.json...")
            update_run_state_from_results()
        return
    
    # 数据合并（独立功能）
    if args.concat:
        from .tools.concat_data import run_concat
        run_concat(
            input_path=input_path,
            final_path=_final_output_path,
            output_path='model_train.jsonl',
            max_records=max_records,
            separate_labels=args.separate_labels
        )
        return
    
    all_stats = {}
    
    # 提取device/OS数据（独立功能）
    if args.extract_device:
        extract_device_data(input_path, max_records)
        return
    
    # ===== RETRY模式 =====
    if args.retry is not None:
        retry_level = args.retry
        
        # 等级描述
        level_desc = {
            1: "最低强度 (缺失 + 失败)",
            2: "中等强度 (缺失 + 失败 + 标签不完整)",
            3: "高强度 (缺失 + 失败 + 标签不完整 + 不可信)"
        }
        
        print("\n" + "=" * 60)
        print(f"RETRY模式 - 等级 {retry_level}: {level_desc.get(retry_level, '未知')}")
        print("=" * 60)
        
        # 加载上次运行配置
        prev_state = load_run_state_full()
        if not prev_state:
            print("[ERROR] 未找到上次运行状态，无法执行retry")
            print("请先运行一次完整的分析任务")
            sys.exit(1)
        
        run_config = prev_state.get('run_config')
        if not run_config:
            print("[ERROR] 上次运行状态中没有保存运行配置")
            print("请使用最新版本重新运行一次完整的分析任务")
            sys.exit(1)
        
        # 显示上次运行配置
        print(f"\n上次运行配置:")
        print(f"  模型: {run_config.get('model', 'default')}")
        print(f"  Product模型: {run_config.get('product_model', 'default')}")
        print(f"  Check模型: {run_config.get('check_model', 'default')}")
        print(f"  线程数: {run_config.get('num_threads', 1)}")
        print(f"  速度等级: {run_config.get('speed_level', '6')}")
        print(f"  输入路径: {run_config.get('input_path', 'default')}")
        
        # 使用原始输入文件路径（从run_config获取，或使用默认路径）
        retry_input_path = run_config.get('input_path') or input_path
        
        # 提取需要重试的IP（根据等级）- 使用原始输入文件
        retry_ips, retry_stats = extract_failed_ips(retry_input_path, _final_output_path, retry_level)
        
        print(f"\n检测到的标签字段: {retry_stats.get('detected_label_fields', [])}")
        print(f"对应的输出字段: {retry_stats.get('required_output_fields', [])}")
        
        print(f"\n数据统计 (等级 {retry_level}):")
        print(f"  输入总数: {retry_stats['total_input']}")
        print(f"  结果总数: {retry_stats['total_results']}")
        print(f"  缺失: {retry_stats['missing']} {'[重试]' if retry_level >= 1 else ''}")
        print(f"  失败: {retry_stats['failed']} {'[重试]' if retry_level >= 1 else ''}")
        print(f"  标签不完整: {retry_stats.get('incomplete', 0)} {'[重试]' if retry_level >= 2 else '[跳过]'}")
        print(f"  不可信: {retry_stats['untrusted']} {'[重试]' if retry_level >= 3 else '[跳过]'}")
        print(f"  本次重试: {retry_stats['total_retry']}")
        
        if retry_stats['total_retry'] == 0:
            print("\n[INFO] 没有需要重试的IP，所有数据处理成功！")
            return
        
        # 从上次运行状态获取平均token消耗，估算本次重试开销
        prev_stats = prev_state.get('stats', {})
        
        # 优先使用 realtime_avg，如果为0则从累计值计算
        avg_input_tokens = prev_stats.get('realtime_avg_input', 0)
        avg_output_tokens = prev_stats.get('realtime_avg_output', 0)
        
        if avg_input_tokens == 0 or avg_output_tokens == 0:
            # 从累计token和已处理数计算平均值
            total_input = prev_stats.get('input_tokens', 0)
            total_output = prev_stats.get('output_tokens', 0)
            processed = prev_stats.get('processed_records', 0)
            if processed > 0:
                avg_input_tokens = total_input / processed
                avg_output_tokens = total_output / processed
            else:
                # 使用默认值
                avg_input_tokens = 800
                avg_output_tokens = 350
        
        # 从配置中恢复参数
        retry_product_model = run_config.get('product_model')
        retry_check_model = run_config.get('check_model')
        
        # 估算开销
        retry_count = retry_stats['total_retry']
        estimated_input_tokens = int(avg_input_tokens * retry_count)
        estimated_output_tokens = int(avg_output_tokens * retry_count)
        estimated_cost = calculate_cost(estimated_input_tokens, estimated_output_tokens, retry_product_model)
        
        print(f"\n" + "=" * 60)
        print(f"重试开销估算")
        print(f"=" * 60)
        print(f"  重试IP数量: {retry_count}")
        print(f"  平均Token/条: {avg_input_tokens:.0f} (输入) / {avg_output_tokens:.0f} (输出)")
        print(f"  预计总Token: {estimated_input_tokens:,} (输入) / {estimated_output_tokens:,} (输出)")
        print(f"  使用模型: {retry_product_model or 'default'}")
        print(f"  预计费用: CNY{estimated_cost:.4f}")
        print(f"=" * 60)
        
        # 询问用户是否继续（如果不是自动模式）
        if not args.auto_confirm:
            print(f"\n是否继续执行重试? (按 Enter 继续，Ctrl+C 取消)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n[INFO] 用户取消重试")
                return
        else:
            print(f"\n[INFO] 自动确认模式，开始执行重试...")
        
        # 使用上次的配置运行重试
        from .multi_thread_runner import run_multi_thread_pipeline
        
        retry_num_threads = run_config.get('num_threads', 1)
        retry_speed_level = run_config.get('speed_level', '6')
        
        # 设置提示词
        retry_product_prompt = run_config.get('product_prompt', 'default')
        retry_check_prompt = run_config.get('check_prompt', 'default')
        ProductAnalyst.set_prompt_id(retry_product_prompt)
        CheckAnalyst.set_prompt_id(retry_check_prompt)
        
        print(f"\n开始重试 {retry_stats['total_retry']} 个IP...")
        
        # 获取当前任务ID
        _, _, prev_task_id = load_run_state()
        current_task_id = prev_task_id if prev_task_id > 0 else 1
        set_task_id(current_task_id)
        
        # RETRY模式标记（简化：所有统计从0开始）
        retry_stats_adjustment = {'is_retry': True}
        
        pipeline_stats = run_multi_thread_pipeline(
            input_path=input_path,
            max_records=None,
            num_workers=retry_num_threads,
            speed_level=retry_speed_level,
            skip_check=False,
            skip_clean=False,  # 不跳过清洗
            restart=False,
            task_id=current_task_id,
            product_model=retry_product_model,
            check_model=retry_check_model,
            run_config=run_config,
            retry_ips=retry_ips,
            retry_stats_adjustment=retry_stats_adjustment,
            entropy_ratio=entropy_ratio
        )
        
        all_stats['retry'] = {
            'retry_stats': retry_stats,
            'pipeline_stats': pipeline_stats
        }
        
        # 将run_config存储到all_stats中，供后续生成报告使用
        all_stats['run_config'] = run_config
        
        print_summary(all_stats)
        log_info("RETRY模式完成")
        return
    
    # 确定执行模式
    if args.clean_only:
        all_stats['cleaner'] = run_cleaner(input_path, max_records, entropy_ratio, args.vendor_balance, args.uniform_sample, difficulty_ratios, max_vendor_ratio)
    
    elif args.product_only:
        if not check_cleaned_data_exists():
            all_stats['cleaner'] = run_cleaner(input_path, max_records, entropy_ratio, args.vendor_balance, args.uniform_sample, difficulty_ratios, max_vendor_ratio)
        all_stats['product'] = run_product_analyst(max_records, args._product_model)
    
    elif args.check_only:
        if not check_final_output_exists():
            print(f"[ERROR] 汇总结果不存在: {_merged_output_path}")
            sys.exit(1)
        all_stats['check'] = run_check_analyst(max_records, args._check_model)
    
    else:  # --all 或默认：使用流水线模式
        # 获取或设置任务ID
        if args.restart:
            # 获取上次任务ID，新任务ID = 上次ID + 1
            _, _, prev_task_id = load_run_state()
            new_task_id = prev_task_id + 1 if prev_task_id > 0 else 1
            set_task_id(new_task_id)
            
            log_info(f"清除已有结果文件和运行状态，开始新任务 (ID: {new_task_id})")
            print(f"[INFO] 开始新任务 (ID: {new_task_id})")
            
            for path in [_product_output_path, _merged_output_path,
                        _check_output_path, _final_output_path]:
                if os.path.exists(path):
                    os.remove(path)
            # 清除运行状态文件
            clear_run_state()
        else:
            # 断点续传：尝试从状态文件恢复任务ID
            _, _, prev_task_id = load_run_state()
            if prev_task_id > 0:
                set_task_id(prev_task_id)
            else:
                set_task_id(1)
        
        # 获取当前任务ID
        current_task_id = get_current_task_id()
        
        # 构建运行配置（用于retry功能）
        run_config = {
            'start_time': datetime.now().isoformat(),
            'input_path': input_path,
            'max_records': max_records,
            'model': args.model,
            'product_model': args._product_model,
            'check_model': args._check_model,
            'product_prompt': args.product_prompt,
            'check_prompt': args.check_prompt,
            'num_threads': args.num_threads,
            'speed_level': args.speed_level,
            'skip_check': args.no_check,
            'test_mode': args.test,
            'vendor_balance': args.vendor_balance,
            'difficulty_ratios': difficulty_ratios,
            'max_vendor_ratio': max_vendor_ratio
        }
        
        # 将run_config存储到all_stats中，供后续生成报告使用
        all_stats['run_config'] = run_config
        
        # 检查是否使用多线程模式
        if args.num_threads and args.num_threads > 1:
            # 多线程模式
            from .multi_thread_runner import run_multi_thread_pipeline
            
            log_info(f"使用多线程模式: {args.num_threads} 个工作线程, 任务ID: {current_task_id}")
            print(f"[INFO] 多线程模式: {args.num_threads} 个工作线程, 速度等级: {args.speed_level}, 任务ID: {current_task_id}")
            
            pipeline_stats = run_multi_thread_pipeline(
                input_path=input_path,
                max_records=max_records,
                num_workers=args.num_threads,
                speed_level=args.speed_level,
                skip_check=args.no_check,
                restart=args.restart,
                task_id=current_task_id,
                product_model=args._product_model,
                check_model=args._check_model,
                run_config=run_config,
                entropy_ratio=entropy_ratio,
                uniform_sample=args.uniform_sample
            )
            all_stats['pipeline'] = pipeline_stats
        else:
            # 信息熵排序筛选（如果指定了--isort参数）
            if args.isort is not None:
                from .entropy_sorter import isort, isort_with_vendor_balance, presample_by_vendor
                from .data_cleaner import DataCleaner
                from .utils.logger import log_info
                
                # 判断是否为厂商平衡采样模式
                is_vendor_balance_mode = args.vendor_balance
                ratio = args.isort
                
                if is_vendor_balance_mode:
                    # 厂商平衡采样模式
                    log_info("开始厂商平衡采样（90%主流厂商 + 10%其他）")
                    print(f"\n[INFO] 厂商平衡采样模式")
                    print(f"  主流厂商: 动态识别数据集中排名前5的厂商")
                    print(f"  采样比例: 90%主流厂商 + 10%其他厂商")
                    print(f"  单厂商: 最低10%, 最高25%")
                    
                    # 新流程：先按厂商比例预采样，再清洗和熵排序
                    if max_records:
                        # 有 max_records 限制时，先按厂商比例预采样原始数据
                        print(f"\n[INFO] 步骤1: 按厂商比例预采样 {max_records} 条原始数据")
                        sampled_lines, presample_stats = presample_by_vendor(
                            input_path,
                            target_count=max_records,
                            major_ratio=0.8,
                            top_n_vendors=5
                        )
                        
                        # 将预采样的数据写入临时文件
                        output_dir = os.path.dirname(_cleaned_data_path)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        temp_presampled_path = os.path.join(output_dir, 'temp_presampled.jsonl')
                        
                        with open(temp_presampled_path, 'w', encoding='utf-8') as f:
                            for line in sampled_lines:
                                f.write(line + '\n')
                        
                        print(f"  预采样完成: {len(sampled_lines)} 条")
                        
                        # 步骤2：清洗预采样的数据
                        print(f"\n[INFO] 步骤2: 清洗数据（保留标签字段）")
                        temp_cleaned_path = os.path.join(output_dir, 'temp_cleaned_with_labels.jsonl')
                        
                        cleaner = DataCleaner(temp_presampled_path, temp_cleaned_path, keep_labels=True)
                        clean_stats = cleaner.run()  # 不限制数量，处理全部预采样数据
                        print(f"  清洗完成: {clean_stats['processed_records']} 条, 压缩率 {clean_stats.get('compression_ratio', 0)*100:.1f}%")
                        
                        # 删除预采样临时文件
                        if os.path.exists(temp_presampled_path):
                            os.remove(temp_presampled_path)
                        
                        # 记录预采样统计
                        all_stats['presample'] = presample_stats
                    else:
                        # 无 max_records 限制时，使用原流程
                        print(f"\n[INFO] 步骤1: 清洗数据（保留标签字段）")
                        output_dir = os.path.dirname(_cleaned_data_path)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        temp_cleaned_path = os.path.join(output_dir, 'temp_cleaned_with_labels.jsonl')
                        
                        cleaner = DataCleaner(input_path, temp_cleaned_path, keep_labels=True)
                        clean_stats = cleaner.run(max_records)
                        print(f"  清洗完成: {clean_stats['processed_records']} 条, 压缩率 {clean_stats.get('compression_ratio', 0)*100:.1f}%")
                else:
                    log_info(f"开始信息熵排序筛选，保留前{ratio*100:.0f}%数据")
                    print(f"\n[INFO] 信息熵排序筛选: 保留前{ratio*100:.0f}%数据")
                    
                    # 第一步：清洗数据但保留标签
                    print(f"[INFO] 步骤1: 清洗数据（保留标签字段）")
                    output_dir = os.path.dirname(_cleaned_data_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    temp_cleaned_path = os.path.join(output_dir, 'temp_cleaned_with_labels.jsonl')
                    
                    cleaner = DataCleaner(input_path, temp_cleaned_path, keep_labels=True)
                    clean_stats = cleaner.run(max_records)
                    print(f"  清洗完成: {clean_stats['processed_records']} 条, 压缩率 {clean_stats.get('compression_ratio', 0)*100:.1f}%")
                
                # 读取清洗后的数据并进行熵排序
                step_num = 3 if (is_vendor_balance_mode and max_records) else 2
                print(f"[INFO] 步骤{step_num}: 信息熵排序")
                cleaned_records = []
                with open(temp_cleaned_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            cleaned_records.append(record)
                        except json.JSONDecodeError:
                            continue
                
                original_count = len(cleaned_records)
                
                # 进行信息熵排序和筛选
                if is_vendor_balance_mode:
                    if max_records:
                        # 已经预采样过，直接按熵排序（不再按厂商比例筛选）
                        from .entropy_sorter import isort
                        sorted_records = isort(cleaned_records, ratio=ratio)
                        filtered_count = len(sorted_records)
                        
                        log_info(f"熵排序完成: {filtered_count} 条")
                        print(f"  清洗后数据: {original_count} 条")
                        print(f"  熵排序后: {filtered_count} 条 (保留前{ratio*100:.0f}%)")
                        
                        # 使用预采样的统计信息
                        vendor_stats = all_stats.get('presample', {})
                        
                        # 记录统计信息
                        all_stats['isort'] = {
                            'mode': 'vendor_balance_presample',
                            'original_count': original_count,
                            'filtered_count': filtered_count,
                            'presample_stats': vendor_stats
                        }
                    else:
                        # 无 max_records，使用原有的厂商平衡采样
                        target_count = int(original_count * ratio)
                        sorted_records, vendor_stats = isort_with_vendor_balance(
                            cleaned_records, 
                            target_count=target_count,
                            major_ratio=0.8,
                            max_single_vendor_ratio=0.25
                        )
                        filtered_count = len(sorted_records)
                        
                        log_info(f"厂商平衡采样完成: {filtered_count} 条")
                        print(f"  清洗后数据: {original_count} 条")
                        print(f"  采样后: {filtered_count} 条")
                        print(f"  主流厂商数据: {vendor_stats['major_count']} 条 ({vendor_stats['major_count']/max(filtered_count,1)*100:.1f}%)")
                        print(f"  其他厂商数据: {vendor_stats['other_count']} 条 ({vendor_stats['other_count']/max(filtered_count,1)*100:.1f}%)")
                        print(f"  厂商分布:")
                        for vendor, count in vendor_stats['vendor_distribution'].items():
                            if count > 0:
                                print(f"    - {vendor}: {count} 条 ({count/max(filtered_count,1)*100:.1f}%)")
                        
                        # 记录统计信息
                        all_stats['isort'] = {
                            'mode': 'vendor_balance',
                            'original_count': original_count,
                            'filtered_count': filtered_count,
                            'vendor_stats': vendor_stats
                        }
                else:
                    # 原有的比例筛选
                    sorted_records = isort(cleaned_records, ratio=ratio)
                    filtered_count = len(sorted_records)
                    
                    log_info(f"筛选后数据数量: {filtered_count} (保留{filtered_count/max(original_count,1)*100:.1f}%)")
                    print(f"  清洗后数据: {original_count} 条")
                    print(f"  筛选后: {filtered_count} 条 (保留{filtered_count/max(original_count,1)*100:.1f}%)")
                    
                    # 记录统计信息
                    all_stats['isort'] = {
                        'mode': 'ratio',
                        'original_count': original_count,
                        'filtered_count': filtered_count,
                        'ratio': ratio
                    }
                
                # 生成最终文件 isort_data.jsonl
                isort_output_path = os.path.join(output_dir, 'isort_data.jsonl')
                with open(isort_output_path, 'w', encoding='utf-8') as f:
                    for record in sorted_records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                # 删除临时文件
                if os.path.exists(temp_cleaned_path):
                    os.remove(temp_cleaned_path)
                
                log_info(f"信息熵排序结果已保存: {isort_output_path}")
                print(f"  输出文件: {isort_output_path}")
                
                all_stats['cleaner'] = clean_stats
                all_stats['isort']['output_path'] = isort_output_path
                
                # isort模式下只进行数据处理，不运行agent
                print("\n" + "=" * 60)
                if is_vendor_balance_mode:
                    print("厂商平衡采样完成！")
                else:
                    print("信息熵排序完成！")
                print("=" * 60)
                print(f"筛选后的数据已保存到: {isort_output_path}")
                print(f"数据已清洗并保留了标签字段（OS, Vendor）")
                print(f"如需继续处理，请使用该文件作为输入运行agent")
                print("=" * 60)
                
                # 打印统计摘要并退出
                print_summary(all_stats)
                log_info("程序结束（isort模式）")
                return
            
            # 单线程模式（原有逻辑）
            all_stats['cleaner'] = run_cleaner(input_path, max_records, entropy_ratio, args.vendor_balance, args.uniform_sample, difficulty_ratios, max_vendor_ratio)
            
            pipeline_stats = run_pipeline(
                max_records, 
                skip_check=args.no_check,
                product_model=args._product_model,
                check_model=args._check_model
            )
            all_stats['pipeline'] = pipeline_stats
    
    # 自动准确率计算（除非指定了--no-acc）
    if not args.no_acc and 'pipeline' in all_stats:
        try:
            from .accuracy_calculator import run_accuracy_calculation
            
            # 检查是否有采样IP列表文件
            sampled_ips_path = None
            if 'cleaner' in all_stats and all_stats['cleaner'].get('sampled_ips_path'):
                sampled_ips_path = all_stats['cleaner']['sampled_ips_path']
            else:
                # 尝试从默认位置读取
                temp_dir = os.path.dirname(_cleaned_data_path)
                default_sampled_ips_path = os.path.join(temp_dir, 'sampled_ips.json')
                if os.path.exists(default_sampled_ips_path):
                    sampled_ips_path = default_sampled_ips_path
            
            # 使用原始输入路径（带标签的数据）进行准确率计算
            # 如果cleaner保存了原始输入路径，优先使用它
            accuracy_input_path = input_path
            if 'cleaner' in all_stats and all_stats['cleaner'].get('original_input_path'):
                accuracy_input_path = all_stats['cleaner']['original_input_path']
            
            acc_stats = run_accuracy_calculation(
                input_path=accuracy_input_path,
                output_path=_final_output_path,
                report_path='accuracy_report.md',
                max_records=max_records,
                sampled_ips_path=sampled_ips_path
            )
            all_stats['accuracy'] = acc_stats
        except Exception as e:
            print(f"\n[警告] 准确率计算失败: {e}")
            log_info(f"准确率计算失败: {e}")
    
    print_summary(all_stats)
    log_info("程序结束")


if __name__ == "__main__":
    main()
