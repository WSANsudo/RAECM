"""
日志工具模块
提供文件日志记录和简洁的命令行输出
支持通过命令行参数或环境变量控制日志级别
"""

import logging
import os
from datetime import datetime
from typing import Optional

from ..config import LOG_DIR


# 全局任务ID（由run.py设置）
_log_task_id: int = 0

# 日志级别映射
LOG_LEVEL_MAP = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# 默认控制台日志级别
_console_log_level: int = logging.WARNING


def set_console_log_level(level: str) -> bool:
    """
    设置控制台日志级别
    
    Args:
        level: 日志级别字符串 ('debug', 'info', 'warning', 'error', 'critical')
        
    Returns:
        是否设置成功
    """
    global _console_log_level
    level_lower = level.lower()
    if level_lower in LOG_LEVEL_MAP:
        _console_log_level = LOG_LEVEL_MAP[level_lower]
        # 如果logger已初始化，更新控制台handler的级别
        if _logger is not None:
            for handler in _logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(_console_log_level)
        return True
    return False


def get_console_log_level() -> str:
    """获取当前控制台日志级别名称"""
    for name, level in LOG_LEVEL_MAP.items():
        if level == _console_log_level:
            return name
    return 'warning'


def set_log_task_id(task_id: int) -> None:
    """设置日志任务ID，并写入日志文件"""
    global _log_task_id
    _log_task_id = task_id
    # 如果logger已初始化，记录任务ID
    if _logger is not None:
        _logger.info(f"任务ID: {task_id}")


def get_log_task_id() -> int:
    """获取日志任务ID"""
    return _log_task_id


def setup_logger(log_dir: str = None, task_id: int = 0, console_level: str = None) -> logging.Logger:
    """
    设置日志记录器
    - 文件日志：详细记录所有信息和错误
    - 控制台：根据设置的级别输出（默认WARNING以上）
    
    Args:
        log_dir: 日志目录路径
        task_id: 任务ID
        console_level: 控制台日志级别 ('debug', 'info', 'warning', 'error', 'critical')
    """
    global _log_task_id, _console_log_level
    if task_id > 0:
        _log_task_id = task_id
    
    # 设置控制台日志级别
    if console_level:
        set_console_log_level(console_level)
    
    # 检查环境变量
    env_level = os.environ.get('LOG_LEVEL', '').lower()
    if env_level and env_level in LOG_LEVEL_MAP:
        _console_log_level = LOG_LEVEL_MAP[env_level]
    
    # 使用config中的LOG_DIR作为默认值
    if log_dir is None:
        log_dir = LOG_DIR
    
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志文件名（按时间）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{timestamp}.txt")
    
    # 创建logger
    logger = logging.getLogger("6Analyst")
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 文件handler - 详细日志
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 控制台handler - 根据设置的级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_console_log_level)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    logger.info(f"控制台日志级别: {get_console_log_level()}")
    if _log_task_id > 0:
        logger.info(f"任务ID: {_log_task_id}")
    
    return logger


# 全局logger实例
_logger = None


def get_logger() -> logging.Logger:
    """获取全局logger实例"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def log_debug(msg: str):
    get_logger().debug(msg)


def log_info(msg: str):
    get_logger().info(msg)


def log_warning(msg: str):
    get_logger().warning(msg)


def log_error(msg: str):
    get_logger().error(msg)


def log_exception(msg: str):
    """记录异常信息（包含堆栈）"""
    get_logger().exception(msg)
