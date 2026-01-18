"""
错误日志模块
专门用于记录程序运行中的错误信息，包括解析错误、API错误等
记录完整的提示词输入和原始返回结果，便于调试和调整代码
"""

import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..config import LOG_DIR


# 错误日志文件路径
ERROR_LOG_PATH = os.path.join(LOG_DIR, "error_log.txt")

# 线程锁，确保多线程写入安全
_error_log_lock = threading.Lock()

# 批次上下文存储（用于记录整个批次的处理过程）
_batch_context: Dict[int, Dict] = {}
_batch_context_lock = threading.Lock()


def _ensure_log_dir():
    """确保日志目录存在"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


def _format_timestamp() -> str:
    """格式化时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _write_to_error_log(content: str):
    """写入错误日志文件"""
    _ensure_log_dir()
    with _error_log_lock:
        with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())


def start_batch_context(batch_id: int, batch_ips: List[str], agent_name: str):
    """
    开始记录批次上下文
    
    Args:
        batch_id: 批次ID（通常是批次序号）
        batch_ips: 批次中的IP列表
        agent_name: Agent名称（如 ProductAnalyst, CheckAnalyst）
    """
    with _batch_context_lock:
        _batch_context[batch_id] = {
            'batch_id': batch_id,
            'agent_name': agent_name,
            'batch_ips': batch_ips,
            'start_time': _format_timestamp(),
            'steps': [],  # 记录处理步骤
            'prompt_input': None,  # 完整的提示词输入
            'raw_response': None,  # 原始返回结果
            'error': None  # 错误信息
        }


def record_batch_step(batch_id: int, step_name: str, details: Any = None):
    """
    记录批次处理步骤
    
    Args:
        batch_id: 批次ID
        step_name: 步骤名称
        details: 步骤详情（可选）
    """
    with _batch_context_lock:
        if batch_id in _batch_context:
            step = {
                'time': _format_timestamp(),
                'step': step_name
            }
            if details is not None:
                step['details'] = details
            _batch_context[batch_id]['steps'].append(step)


def record_prompt_input(batch_id: int, messages: List[Dict]):
    """
    记录完整的提示词输入（填充数据后的输入）
    
    Args:
        batch_id: 批次ID
        messages: 发送给API的完整消息列表
    """
    with _batch_context_lock:
        if batch_id in _batch_context:
            _batch_context[batch_id]['prompt_input'] = messages


def record_raw_response(batch_id: int, response: str):
    """
    记录API的原始返回结果
    
    Args:
        batch_id: 批次ID
        response: API返回的原始字符串
    """
    with _batch_context_lock:
        if batch_id in _batch_context:
            _batch_context[batch_id]['raw_response'] = response


def end_batch_context(batch_id: int, success: bool = True):
    """
    结束批次上下文记录
    如果成功则清除上下文，如果失败则保留用于错误日志
    
    Args:
        batch_id: 批次ID
        success: 是否成功完成
    """
    if success:
        with _batch_context_lock:
            if batch_id in _batch_context:
                del _batch_context[batch_id]


def get_batch_context(batch_id: int) -> Optional[Dict]:
    """获取批次上下文"""
    with _batch_context_lock:
        return _batch_context.get(batch_id, {}).copy() if batch_id in _batch_context else None


def log_parse_error(
    batch_id: int,
    error_type: str,
    error_message: str,
    additional_info: Dict = None
):
    """
    记录解析错误
    
    Args:
        batch_id: 批次ID
        error_type: 错误类型（如 json_parse_error, response_format_error）
        error_message: 错误消息
        additional_info: 额外信息
    """
    context = get_batch_context(batch_id)
    
    log_entry = []
    log_entry.append("\n" + "=" * 80)
    log_entry.append(f"[ERROR] {_format_timestamp()}")
    log_entry.append(f"错误类型: {error_type}")
    log_entry.append(f"错误消息: {error_message}")
    log_entry.append("=" * 80)
    
    if context:
        log_entry.append(f"\n--- 批次信息 ---")
        log_entry.append(f"批次ID: {context.get('batch_id')}")
        log_entry.append(f"Agent: {context.get('agent_name')}")
        log_entry.append(f"开始时间: {context.get('start_time')}")
        log_entry.append(f"IP列表: {context.get('batch_ips')}")
        
        # 记录处理步骤
        steps = context.get('steps', [])
        if steps:
            log_entry.append(f"\n--- 处理步骤 ({len(steps)}步) ---")
            for i, step in enumerate(steps, 1):
                log_entry.append(f"  {i}. [{step.get('time')}] {step.get('step')}")
                if 'details' in step:
                    details = step['details']
                    if isinstance(details, str) and len(details) > 200:
                        log_entry.append(f"      详情: {details[:200]}...")
                    else:
                        log_entry.append(f"      详情: {details}")
        
        # 记录完整的提示词输入
        prompt_input = context.get('prompt_input')
        if prompt_input:
            log_entry.append(f"\n--- 完整提示词输入 ---")
            for msg in prompt_input:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                log_entry.append(f"[{role}]:")
                log_entry.append(content)
                log_entry.append("")  # 空行分隔
        
        # 记录原始返回结果
        raw_response = context.get('raw_response')
        if raw_response:
            log_entry.append(f"\n--- 原始返回结果 ---")
            log_entry.append(raw_response)
    
    if additional_info:
        log_entry.append(f"\n--- 额外信息 ---")
        for key, value in additional_info.items():
            # 不截断，记录完整内容
            log_entry.append(f"{key}:")
            log_entry.append(str(value))
    
    log_entry.append("\n" + "-" * 80 + "\n")
    
    _write_to_error_log("\n".join(log_entry))
    
    # 清除批次上下文
    end_batch_context(batch_id, success=False)


def log_api_error(
    batch_id: int,
    error_type: str,
    error_message: str,
    exception: Exception = None
):
    """
    记录API调用错误
    
    Args:
        batch_id: 批次ID
        error_type: 错误类型（如 rate_limit, security_limit, insufficient_balance）
        error_message: 错误消息
        exception: 异常对象（可选）
    """
    context = get_batch_context(batch_id)
    
    log_entry = []
    log_entry.append("\n" + "=" * 80)
    log_entry.append(f"[API ERROR] {_format_timestamp()}")
    log_entry.append(f"错误类型: {error_type}")
    log_entry.append(f"错误消息: {error_message}")
    
    if exception:
        log_entry.append(f"异常类型: {type(exception).__name__}")
        log_entry.append(f"异常详情: {str(exception)}")
    
    log_entry.append("=" * 80)
    
    if context:
        log_entry.append(f"\n--- 批次信息 ---")
        log_entry.append(f"批次ID: {context.get('batch_id')}")
        log_entry.append(f"Agent: {context.get('agent_name')}")
        log_entry.append(f"开始时间: {context.get('start_time')}")
        log_entry.append(f"IP列表: {context.get('batch_ips')}")
        
        # 记录处理步骤
        steps = context.get('steps', [])
        if steps:
            log_entry.append(f"\n--- 处理步骤 ({len(steps)}步) ---")
            for i, step in enumerate(steps, 1):
                log_entry.append(f"  {i}. [{step.get('time')}] {step.get('step')}")
        
        # 记录完整的提示词输入
        prompt_input = context.get('prompt_input')
        if prompt_input:
            log_entry.append(f"\n--- 完整提示词输入 ---")
            for msg in prompt_input:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                log_entry.append(f"[{role}]:")
                log_entry.append(content)
                log_entry.append("")
    
    log_entry.append("\n" + "-" * 80 + "\n")
    
    _write_to_error_log("\n".join(log_entry))


def log_batch_exception(
    batch_id: int,
    exception: Exception,
    stage: str = "unknown"
):
    """
    记录批次处理异常
    
    Args:
        batch_id: 批次ID
        exception: 异常对象
        stage: 发生异常的阶段
    """
    import traceback
    
    context = get_batch_context(batch_id)
    
    log_entry = []
    log_entry.append("\n" + "=" * 80)
    log_entry.append(f"[EXCEPTION] {_format_timestamp()}")
    log_entry.append(f"阶段: {stage}")
    log_entry.append(f"异常类型: {type(exception).__name__}")
    log_entry.append(f"异常消息: {str(exception)}")
    log_entry.append("=" * 80)
    
    # 记录堆栈跟踪
    log_entry.append(f"\n--- 堆栈跟踪 ---")
    log_entry.append(traceback.format_exc())
    
    if context:
        log_entry.append(f"\n--- 批次信息 ---")
        log_entry.append(f"批次ID: {context.get('batch_id')}")
        log_entry.append(f"Agent: {context.get('agent_name')}")
        log_entry.append(f"开始时间: {context.get('start_time')}")
        log_entry.append(f"IP列表: {context.get('batch_ips')}")
        
        # 记录处理步骤
        steps = context.get('steps', [])
        if steps:
            log_entry.append(f"\n--- 处理步骤 ({len(steps)}步) ---")
            for i, step in enumerate(steps, 1):
                log_entry.append(f"  {i}. [{step.get('time')}] {step.get('step')}")
        
        # 记录完整的提示词输入
        prompt_input = context.get('prompt_input')
        if prompt_input:
            log_entry.append(f"\n--- 完整提示词输入 ---")
            for msg in prompt_input:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                log_entry.append(f"[{role}]:")
                log_entry.append(content)
                log_entry.append("")
        
        # 记录原始返回结果
        raw_response = context.get('raw_response')
        if raw_response:
            log_entry.append(f"\n--- 原始返回结果 ---")
            log_entry.append(raw_response)
    
    log_entry.append("\n" + "-" * 80 + "\n")
    
    _write_to_error_log("\n".join(log_entry))
    
    # 清除批次上下文
    end_batch_context(batch_id, success=False)


def clear_error_log():
    """清空错误日志文件"""
    _ensure_log_dir()
    with _error_log_lock:
        with open(ERROR_LOG_PATH, 'w', encoding='utf-8') as f:
            f.write(f"# 错误日志 - 创建于 {_format_timestamp()}\n")
            f.write("# 本文件记录程序运行中的错误信息，包括解析错误、API错误等\n")
            f.write("# 每条错误记录包含完整的提示词输入和原始返回结果\n")
            f.write("\n")


def get_error_log_path() -> str:
    """获取错误日志文件路径"""
    return ERROR_LOG_PATH
