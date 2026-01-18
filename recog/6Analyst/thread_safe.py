"""
线程安全模块
提供多线程执行所需的线程安全组件：
- ThreadSafeStats: 线程安全的统计管理器
- ThreadSafeFileWriter: 线程安全的文件写入器
- TaskManager: 任务分配和状态管理器
- ThreadSafeLogger: 线程安全的日志管理器
"""

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from queue import Queue, Empty
import logging


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 待处理
    IN_PROGRESS = "in_progress"  # 处理中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    SKIPPED = "skipped"          # 跳过（已处理过）


@dataclass
class TaskItem:
    """任务项"""
    batch_id: int                    # 批次ID
    records: List[Dict]              # 记录列表
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[int] = None  # 处理该任务的工作线程ID
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    error_msg: Optional[str] = None
    
    @property
    def ips(self) -> List[str]:
        return [r['ip'] for r in self.records]


class TaskManager:
    """
    线程安全的任务管理器
    负责任务分配、状态跟踪、避免重复执行
    """
    
    def __init__(self, max_retries: int = 3):
        self._lock = threading.RLock()
        self._tasks: Dict[int, TaskItem] = {}  # batch_id -> TaskItem
        self._pending_queue: Queue = Queue()
        self._processed_ips: Set[str] = set()  # 已处理的IP集合
        self._in_progress_ips: Set[str] = set()  # 正在处理的IP集合
        self._active_workers: Set[int] = set()  # 活跃的工作线程ID集合
        self._max_retries = max_retries
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._shutdown = False
    
    def load_processed_ips(self, file_path: str) -> int:
        """从结果文件加载已处理的IP"""
        count = 0
        if os.path.exists(file_path):
            with self._lock:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            ip = obj.get('ip')
                            if ip:
                                self._processed_ips.add(ip)
                                count += 1
                        except json.JSONDecodeError:
                            continue
        return count
    
    def add_tasks(self, all_records: List[Dict], batch_size: int) -> int:
        """
        添加任务，自动过滤已处理的记录
        返回实际添加的任务数
        """
        with self._lock:
            # 过滤已处理的记录
            pending_records = [r for r in all_records 
                             if r['ip'] not in self._processed_ips]
            
            # 分批创建任务
            batch_id = 0
            for i in range(0, len(pending_records), batch_size):
                batch = pending_records[i:i+batch_size]
                task = TaskItem(batch_id=batch_id, records=batch)
                self._tasks[batch_id] = task
                self._pending_queue.put(batch_id)
                batch_id += 1
            
            self._total_tasks = batch_id
            return batch_id
    
    def get_next_task(self, worker_id: int, timeout: float = 1.0) -> Optional[TaskItem]:
        """
        获取下一个待处理任务
        返回None表示没有更多任务或已关闭
        """
        if self._shutdown:
            return None
        
        try:
            batch_id = self._pending_queue.get(timeout=timeout)
        except Empty:
            return None
        
        with self._lock:
            if batch_id not in self._tasks:
                return None
            
            task = self._tasks[batch_id]
            
            # 检查是否有IP正在被其他线程处理（避免冲突）
            for ip in task.ips:
                if ip in self._in_progress_ips:
                    # 放回队列稍后重试
                    self._pending_queue.put(batch_id)
                    return None
            
            # 标记为处理中
            task.status = TaskStatus.IN_PROGRESS
            task.worker_id = worker_id
            task.start_time = time.time()
            
            # 标记IP为处理中
            for ip in task.ips:
                self._in_progress_ips.add(ip)
            
            # 标记工作线程为活跃
            self._active_workers.add(worker_id)
            
            return task
    
    def complete_task(self, batch_id: int, success: bool, error_msg: str = None) -> None:
        """标记任务完成"""
        with self._lock:
            if batch_id not in self._tasks:
                return
            
            task = self._tasks[batch_id]
            task.end_time = time.time()
            
            # 从处理中移除IP
            for ip in task.ips:
                self._in_progress_ips.discard(ip)
            
            # 从活跃线程中移除
            if task.worker_id is not None:
                self._active_workers.discard(task.worker_id)
            
            if success:
                task.status = TaskStatus.COMPLETED
                self._completed_tasks += 1
                # 标记IP为已处理
                for ip in task.ips:
                    self._processed_ips.add(ip)
            else:
                task.retry_count += 1
                task.error_msg = error_msg
                
                if task.retry_count < self._max_retries:
                    # 重新加入队列
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    self._pending_queue.put(batch_id)
                else:
                    task.status = TaskStatus.FAILED
                    self._failed_tasks += 1
    
    def mark_ip_processed(self, ip: str) -> None:
        """标记单个IP为已处理"""
        with self._lock:
            self._processed_ips.add(ip)
    
    def is_ip_processed(self, ip: str) -> bool:
        """检查IP是否已处理"""
        with self._lock:
            return ip in self._processed_ips
    
    def shutdown(self) -> None:
        """关闭任务管理器"""
        self._shutdown = True
    
    def get_stats(self) -> Dict:
        """获取任务统计"""
        with self._lock:
            return {
                'total_tasks': self._total_tasks,
                'completed_tasks': self._completed_tasks,
                'failed_tasks': self._failed_tasks,
                'pending_tasks': self._pending_queue.qsize(),
                'processed_ips': len(self._processed_ips),
                'in_progress_ips': len(self._in_progress_ips),
                'active_workers': len(self._active_workers)
            }
    
    @property
    def is_all_done(self) -> bool:
        """检查是否所有任务都已完成"""
        with self._lock:
            return (self._pending_queue.empty() and 
                    len(self._in_progress_ips) == 0 and
                    self._completed_tasks + self._failed_tasks >= self._total_tasks)


class ThreadSafeStats:
    """
    线程安全的统计管理器
    负责计数、token统计、费用计算
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._stats = {
            'total_records': 0,
            'skipped_records': 0,
            'processed_records': 0,
            'error_count': 0,
            'high_conf': 0,
            'mid_conf': 0,
            'low_conf': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'this_run_input_tokens': 0,
            'this_run_output_tokens': 0,
            'verified_count': 0,
            'adjusted_count': 0,
            'rejected_count': 0,
            'execution_time_seconds': 0,
        }
        # 每个工作线程的独立统计
        self._worker_stats: Dict[int, Dict] = defaultdict(lambda: {
            'processed_records': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'error_count': 0,
            'high_conf': 0,
            'mid_conf': 0,
            'low_conf': 0,
        })
        self._start_time = time.time()
    
    def set_total(self, total: int, skipped: int = 0) -> None:
        """设置总记录数和已跳过（已处理）的记录数"""
        with self._lock:
            self._stats['total_records'] = total
            self._stats['skipped_records'] = skipped
            # 已跳过的记录就是已处理的记录（断点续传）
            self._stats['processed_records'] = skipped
    
    def add_processed(self, worker_id: int, count: int = 1) -> None:
        """增加已处理记录数"""
        with self._lock:
            self._stats['processed_records'] += count
            self._worker_stats[worker_id]['processed_records'] += count
    
    def add_tokens(self, worker_id: int, input_tokens: int, output_tokens: int) -> None:
        """增加token计数"""
        with self._lock:
            self._stats['input_tokens'] += input_tokens
            self._stats['output_tokens'] += output_tokens
            self._stats['this_run_input_tokens'] += input_tokens
            self._stats['this_run_output_tokens'] += output_tokens
            self._worker_stats[worker_id]['input_tokens'] += input_tokens
            self._worker_stats[worker_id]['output_tokens'] += output_tokens
    
    def add_confidence(self, worker_id: int, confidence: float) -> None:
        """根据置信度分类计数"""
        with self._lock:
            if confidence >= 0.8:
                self._stats['high_conf'] += 1
                self._worker_stats[worker_id]['high_conf'] += 1
            elif confidence >= 0.6:
                self._stats['mid_conf'] += 1
                self._worker_stats[worker_id]['mid_conf'] += 1
            else:
                self._stats['low_conf'] += 1
                self._worker_stats[worker_id]['low_conf'] += 1
    
    def add_validation(self, status: str) -> None:
        """增加校验状态计数"""
        with self._lock:
            if status == 'verified':
                self._stats['verified_count'] += 1
            elif status == 'adjusted':
                self._stats['adjusted_count'] += 1
            elif status == 'rejected':
                self._stats['rejected_count'] += 1
    
    def add_error(self, worker_id: int, count: int = 1) -> None:
        """增加错误计数"""
        with self._lock:
            self._stats['error_count'] += count
            self._worker_stats[worker_id]['error_count'] += count
    
    def restore_from_state(self, prev_stats: Dict) -> None:
        """从之前的状态恢复统计"""
        with self._lock:
            # 恢复累计统计（不包括本次运行的统计）
            for key in ['input_tokens', 'output_tokens', 'high_conf', 'mid_conf', 
                       'low_conf', 'verified_count', 'adjusted_count', 'rejected_count',
                       'error_count', 'processed_records', 'skipped_records']:
                if key in prev_stats:
                    self._stats[key] = prev_stats[key]
            # 注意：this_run_* 不恢复，因为是本次运行的统计
    
    def get_stats(self) -> Dict:
        """获取当前统计（线程安全的副本）"""
        with self._lock:
            stats = self._stats.copy()
            stats['execution_time_seconds'] = time.time() - self._start_time
            
            # 计算实时平均
            processed = stats['this_run_input_tokens']
            this_run_processed = stats['processed_records'] - stats['skipped_records']
            if this_run_processed > 0:
                stats['realtime_avg_input'] = stats['this_run_input_tokens'] / this_run_processed
                stats['realtime_avg_output'] = stats['this_run_output_tokens'] / this_run_processed
            else:
                stats['realtime_avg_input'] = 0
                stats['realtime_avg_output'] = 0
            
            return stats
    
    def get_worker_stats(self, worker_id: int) -> Dict:
        """获取指定工作线程的统计"""
        with self._lock:
            return dict(self._worker_stats[worker_id])
    
    def get_progress(self) -> Tuple[int, int, float]:
        """获取进度信息: (已处理, 总数, 百分比)"""
        with self._lock:
            processed = self._stats['processed_records']
            total = self._stats['total_records']
            pct = processed / total * 100 if total > 0 else 0
            return processed, total, pct


class ThreadSafeFileWriter:
    """
    线程安全的文件写入器
    避免并发写入冲突
    """
    
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
    
    def _get_lock(self, file_path: str) -> threading.Lock:
        """获取文件对应的锁"""
        with self._global_lock:
            if file_path not in self._locks:
                self._locks[file_path] = threading.Lock()
            return self._locks[file_path]
    
    def append_json(self, file_path: str, data: Dict) -> bool:
        """追加JSON记录到文件"""
        lock = self._get_lock(file_path)
        with lock:
            try:
                # 确保目录存在
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception as e:
                # 记录错误信息
                print(f"[ERROR] 文件写入失败 {file_path}: {e}")
                return False
    
    def write_json(self, file_path: str, data: Any) -> bool:
        """写入JSON文件（覆盖）"""
        lock = self._get_lock(file_path)
        with lock:
            try:
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception:
                return False
    
    def append_batch(self, file_path: str, records: List[Dict]) -> bool:
        """批量追加多条记录"""
        lock = self._get_lock(file_path)
        with lock:
            try:
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with open(file_path, 'a', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception:
                return False


class ThreadSafeLogger:
    """
    线程安全的日志管理器
    每个工作线程有独立的日志文件，同时汇总到主日志
    日志文件名包含任务ID，格式: log_task{task_id}_{timestamp}_main.txt
    """
    
    def __init__(self, log_dir: str, task_id: int = 0):
        self._log_dir = log_dir
        self._task_id = task_id
        self._lock = threading.RLock()  # 使用可重入锁
        self._worker_loggers: Dict[int, logging.Logger] = {}
        self._main_logger: Optional[logging.Logger] = None
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._closed = False  # 标记是否已关闭
        
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 初始化主日志
        self._setup_main_logger()
    
    def _setup_main_logger(self) -> None:
        """设置主日志记录器"""
        # 日志文件名包含任务ID
        log_file = os.path.join(self._log_dir, f"log_task{self._task_id}_{self._timestamp}_main.txt")
        
        logger = logging.getLogger(f"6Analyst_task{self._task_id}_main_{self._timestamp}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False  # 禁止传播到根logger
        
        # 文件handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        self._main_logger = logger
        self._safe_log(self._main_logger, 'info', f"=" * 60)
        self._safe_log(self._main_logger, 'info', f"多线程日志系统初始化")
        self._safe_log(self._main_logger, 'info', f"任务ID: {self._task_id}")
        self._safe_log(self._main_logger, 'info', f"=" * 60)
    
    def _safe_log(self, logger: logging.Logger, level: str, msg: str) -> bool:
        """
        安全地写入日志，处理文件句柄失效的情况
        
        Returns:
            是否成功写入
        """
        if self._closed or logger is None:
            return False
        
        try:
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(msg)
            return True
        except (OSError, IOError, ValueError) as e:
            # 文件句柄失效，尝试重建
            if "Bad file descriptor" in str(e) or "I/O operation" in str(e):
                try:
                    self._rebuild_logger_handlers(logger)
                    log_func = getattr(logger, level.lower(), logger.info)
                    log_func(msg)
                    return True
                except Exception:
                    pass
            return False
        except Exception:
            return False
    
    def _rebuild_logger_handlers(self, logger: logging.Logger) -> None:
        """重建logger的文件handlers"""
        with self._lock:
            # 关闭并移除所有旧handlers
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
                logger.removeHandler(handler)
            
            # 确定日志文件路径
            logger_name = logger.name
            if '_main_' in logger_name:
                log_file = os.path.join(self._log_dir, f"log_task{self._task_id}_{self._timestamp}_main.txt")
            else:
                # 从logger名称中提取worker_id
                import re
                match = re.search(r'worker(\d+)', logger_name)
                if match:
                    worker_id = match.group(1)
                    log_file = os.path.join(
                        self._log_dir, 
                        f"log_task{self._task_id}_{self._timestamp}_worker{worker_id}.txt"
                    )
                else:
                    return  # 无法确定文件路径
            
            # 创建新的handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-7s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
    
    def get_worker_logger(self, worker_id: int) -> logging.Logger:
        """获取工作线程的日志记录器"""
        with self._lock:
            if worker_id not in self._worker_loggers:
                # 日志文件名包含任务ID
                log_file = os.path.join(
                    self._log_dir, 
                    f"log_task{self._task_id}_{self._timestamp}_worker{worker_id}.txt"
                )
                
                logger = logging.getLogger(f"6Analyst_task{self._task_id}_worker{worker_id}_{self._timestamp}")
                logger.setLevel(logging.DEBUG)
                logger.handlers.clear()
                logger.propagate = False  # 禁止传播到根logger
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_format = logging.Formatter(
                    '%(asctime)s | %(levelname)-7s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_format)
                logger.addHandler(file_handler)
                
                self._worker_loggers[worker_id] = logger
                self._safe_log(logger, 'info', f"=" * 60)
                self._safe_log(logger, 'info', f"工作线程 {worker_id} 日志初始化")
                self._safe_log(logger, 'info', f"任务ID: {self._task_id}")
                self._safe_log(logger, 'info', f"=" * 60)
            
            return self._worker_loggers[worker_id]
    
    def log_main(self, level: str, msg: str) -> None:
        """记录到主日志"""
        if self._main_logger and not self._closed:
            self._safe_log(self._main_logger, level, msg)
    
    def log_worker(self, worker_id: int, level: str, msg: str) -> None:
        """记录到工作线程日志，同时汇总到主日志"""
        if self._closed:
            return
        
        with self._lock:
            logger = self.get_worker_logger(worker_id)
            self._safe_log(logger, level, msg)
            
            # 同时记录到主日志（带工作线程标识）
            self.log_main(level, f"[W{worker_id}] {msg}")
    
    def info(self, worker_id: int, msg: str) -> None:
        self.log_worker(worker_id, 'info', msg)
    
    def debug(self, worker_id: int, msg: str) -> None:
        self.log_worker(worker_id, 'debug', msg)
    
    def warning(self, worker_id: int, msg: str) -> None:
        self.log_worker(worker_id, 'warning', msg)
    
    def error(self, worker_id: int, msg: str) -> None:
        self.log_worker(worker_id, 'error', msg)
    
    def exception(self, worker_id: int, msg: str) -> None:
        if self._closed:
            return
        with self._lock:
            logger = self.get_worker_logger(worker_id)
            try:
                logger.exception(msg)
            except (OSError, IOError):
                pass
            self.log_main('error', f"[W{worker_id}] {msg}")
    
    def close(self) -> None:
        """关闭所有日志handlers"""
        with self._lock:
            self._closed = True
            
            # 关闭主日志
            if self._main_logger:
                for handler in self._main_logger.handlers[:]:
                    try:
                        handler.close()
                    except Exception:
                        pass
                    self._main_logger.removeHandler(handler)
            
            # 关闭所有worker日志
            for logger in self._worker_loggers.values():
                for handler in logger.handlers[:]:
                    try:
                        handler.close()
                    except Exception:
                        pass
                    logger.removeHandler(handler)
            
            self._worker_loggers.clear()


# 全局实例（延迟初始化）
_file_writer: Optional[ThreadSafeFileWriter] = None
_file_writer_lock = threading.Lock()


def get_file_writer() -> ThreadSafeFileWriter:
    """获取全局文件写入器实例"""
    global _file_writer
    with _file_writer_lock:
        if _file_writer is None:
            _file_writer = ThreadSafeFileWriter()
        return _file_writer
