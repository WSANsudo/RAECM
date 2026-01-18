"""
多线程多批次执行器
支持配置多个工作线程同时处理多个批次任务
每个工作线程内部仍支持speed level配置和并行agent模式
"""

import json
import os
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any

from .config import (
    BATCH_SIZE, MODEL_NAME, MODEL_PRICES,
    SPEED_LEVELS, DEFAULT_SPEED_LEVEL, DEBUG_MODE
)
from .run import get_output_paths
from .thread_safe import (
    TaskManager, TaskItem, TaskStatus,
    ThreadSafeStats, ThreadSafeFileWriter, ThreadSafeLogger,
    get_file_writer
)
from .product_analyst import ProductAnalyst
from .check_analyst import CheckAnalyst


@dataclass
class MultiThreadConfig:
    """多线程配置"""
    num_workers: int = 2              # 工作线程数
    speed_level: str = 's'            # 每个线程内部的速度等级
    batch_size: int = BATCH_SIZE      # 批次大小
    max_retries: int = 3              # 最大重试次数
    skip_check: bool = False          # 是否跳过校验
    rate_limit_wait: int = 30         # 触发限制后等待分钟数
    rate_limit_backoff: float = 2.0   # 等待时间倍增系数


class WorkerContext:
    """工作线程上下文，包含该线程专用的分析器实例"""
    
    def __init__(self, worker_id: int, config: MultiThreadConfig,
                 logger: ThreadSafeLogger, stats: ThreadSafeStats,
                 file_writer: ThreadSafeFileWriter,
                 product_model: str = None, check_model: str = None):
        self.worker_id = worker_id
        self.config = config
        self.logger = logger
        self.stats = stats
        self.file_writer = file_writer
        
        # 获取当前模式下的路径
        paths = get_output_paths()
        cleaned_data_path = paths['cleaned_data']
        product_output_path = paths['product_output']
        merged_output_path = paths['merged_output']
        check_output_path = paths['check_output']
        final_output_path = paths['final_output']
        
        # 每个工作线程创建独立的分析器实例（避免共享状态）
        self.product_analyst = ProductAnalyst(cleaned_data_path, product_output_path, product_model)
        self.check_analyst = CheckAnalyst(merged_output_path, check_output_path, final_output_path, check_model)
        
        # 速度配置
        self.speed_config = SPEED_LEVELS.get(config.speed_level, SPEED_LEVELS[DEFAULT_SPEED_LEVEL])
        
        # 限流状态
        self.rate_limit_count = 0
        self.last_rate_limit_time = 0
    
    def log_info(self, msg: str):
        self.logger.info(self.worker_id, msg)
    
    def log_debug(self, msg: str):
        self.logger.debug(self.worker_id, msg)
    
    def log_error(self, msg: str):
        self.logger.error(self.worker_id, msg)
    
    def log_warning(self, msg: str):
        self.logger.warning(self.worker_id, msg)
    
    def get_delay(self) -> float:
        """获取agent间隔时间"""
        return self.speed_config['delay']
    
    def is_parallel_mode(self) -> bool:
        """是否为并行模式"""
        return self.speed_config['parallel']
    
    def handle_rate_limit(self) -> int:
        """
        处理限流，返回需要等待的秒数
        使用指数退避策略
        """
        self.rate_limit_count += 1
        wait_minutes = self.config.rate_limit_wait * (
            self.config.rate_limit_backoff ** (self.rate_limit_count - 1)
        )
        self.last_rate_limit_time = time.time()
        return int(wait_minutes * 60)


class MultiThreadRunner:
    """
    多线程多批次执行器
    """
    
    def __init__(self, config: MultiThreadConfig = None,
                 product_model: str = None, check_model: str = None):
        self.config = config or MultiThreadConfig()
        
        # 模型配置
        self.product_model = product_model
        self.check_model = check_model
        
        # 获取当前模式下的路径
        paths = get_output_paths()
        self._cleaned_data_path = paths['cleaned_data']
        self._product_output_path = paths['product_output']
        self._merged_output_path = paths['merged_output']
        self._check_output_path = paths['check_output']
        self._final_output_path = paths['final_output']
        self._run_state_path = paths['run_state']
        self._log_dir = paths['log_dir']
        
        # 初始化线程安全组件
        self.task_manager = TaskManager(max_retries=self.config.max_retries)
        self.stats = ThreadSafeStats()
        self.file_writer = get_file_writer()
        self.logger: Optional[ThreadSafeLogger] = None
        
        # 运行状态
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 默认不暂停
        
        # 进度回调
        self._progress_callback: Optional[Callable] = None
        
        # 全局限流控制
        self._global_rate_limit_lock = threading.Lock()
        self._global_rate_limit_until = 0
        self._global_rate_limit_reason = ""  # 限流原因
        
        # 等待状态（用于进度打印线程判断是否显示倒计时）
        self._waiting_lock = threading.Lock()
        self._is_waiting = False
        self._wait_end_time = 0
        self._wait_reason = ""
    
    def set_progress_callback(self, callback: Callable[[Dict], None]) -> None:
        """设置进度回调函数"""
        self._progress_callback = callback
    
    def _notify_progress(self) -> None:
        """通知进度更新"""
        if self._progress_callback:
            stats = self.stats.get_stats()
            task_stats = self.task_manager.get_stats()
            stats.update(task_stats)
            self._progress_callback(stats)
    
    def _safe_write_json(self, file_path: str, data: Dict, ctx: 'WorkerContext' = None, 
                         file_type: str = "output") -> bool:
        """
        安全地写入JSON记录到文件，带日志记录
        
        Args:
            file_path: 目标文件路径
            data: 要写入的数据
            ctx: 工作线程上下文（用于日志）
            file_type: 文件类型描述（用于日志）
        
        Returns:
            是否写入成功
        """
        success = self.file_writer.append_json(file_path, data)
        if not success:
            ip = data.get('ip', 'unknown')
            error_msg = f"写入{file_type}文件失败: {file_path}, IP: {ip}"
            if ctx:
                ctx.log_error(error_msg)
            else:
                print(f"[ERROR] {error_msg}")
        return success
    
    def _check_global_rate_limit(self) -> int:
        """
        检查全局限流状态
        返回需要等待的秒数，0表示不需要等待
        """
        with self._global_rate_limit_lock:
            now = time.time()
            if now < self._global_rate_limit_until:
                return int(self._global_rate_limit_until - now)
            return 0
    
    def _set_global_rate_limit(self, wait_seconds: int, reason: str = "API限制") -> None:
        """设置全局限流"""
        with self._global_rate_limit_lock:
            new_until = time.time() + wait_seconds
            # 只有新的限流时间更长才更新
            if new_until > self._global_rate_limit_until:
                self._global_rate_limit_until = new_until
                self._global_rate_limit_reason = reason
        
        # 设置等待状态（用于进度打印线程）
        with self._waiting_lock:
            self._is_waiting = True
            self._wait_end_time = new_until
            self._wait_reason = reason
    
    def _clear_waiting_state(self) -> None:
        """清除等待状态"""
        with self._waiting_lock:
            self._is_waiting = False
            self._wait_end_time = 0
            self._wait_reason = ""
    
    def _get_waiting_state(self) -> Tuple[bool, int, str]:
        """
        获取等待状态
        返回: (是否在等待, 剩余秒数, 等待原因)
        """
        with self._waiting_lock:
            if not self._is_waiting:
                return False, 0, ""
            remaining = int(self._wait_end_time - time.time())
            if remaining <= 0:
                self._is_waiting = False
                return False, 0, ""
            return True, remaining, self._wait_reason
    
    def _process_batch_with_retry(self, ctx: WorkerContext, task: TaskItem) -> Tuple[bool, str]:
        """
        处理单个批次，包含重试逻辑
        返回: (是否成功, 错误信息)
        """
        batch = task.records
        batch_id = task.batch_id
        
        ctx.log_info(f"开始处理批次 {batch_id}, IPs: {task.ips}")
        
        try:
            # 检查全局限流
            wait_time = self._check_global_rate_limit()
            if wait_time > 0:
                ctx.log_warning(f"全局限流中，等待 {wait_time}s")
                # 更新等待状态（用于进度打印线程显示倒计时）
                with self._waiting_lock:
                    if not self._is_waiting or self._wait_end_time < time.time() + wait_time:
                        self._is_waiting = True
                        self._wait_end_time = time.time() + wait_time
                        self._wait_reason = self._global_rate_limit_reason or "API限制"
                time.sleep(wait_time)
                # 清除等待状态
                self._clear_waiting_state()
            
            # === 步骤1: 产品分析 ===
            product_results, product_stats = ctx.product_analyst.process_batch(batch)
            
            # 检查是否触发限流或余额不足
            if product_stats.get('insufficient_balance'):
                ctx.log_error("余额不足")
                self._shutdown_event.set()
                return False, "insufficient_balance"
            
            if product_stats.get('rate_limited') or product_stats.get('security_limited'):
                limit_type = "安全限制" if product_stats.get('security_limited') else "并发限制"
                wait_seconds = ctx.handle_rate_limit()
                self._set_global_rate_limit(wait_seconds, f"API{limit_type}")
                ctx.log_warning(f"触发{limit_type}，全局等待 {wait_seconds}s")
                return False, "rate_limited"
            
            # 更新token统计
            input_tokens = product_stats.get('input_token_count', 0)
            output_tokens = product_stats.get('output_token_count', 0)
            ctx.stats.add_tokens(ctx.worker_id, input_tokens, output_tokens)
            
            # 保存产品结果（实时写入）
            if product_results:
                for res in product_results:
                    self._safe_write_json(self._product_output_path, res, ctx, "product_output")
            
            # === 步骤2: 构建合并结果（仅产品分析） ===
            product_map = {r.get('ip'): r for r in (product_results or [])}
            
            merged_results = []
            for r in batch:
                ip = r['ip']
                product = product_map.get(ip, {})
                
                # 检查产品分析是否有结果
                if not product:
                    ctx.log_error(f"IP {ip} 产品分析失败")
                    failed_result = {
                        'ip': ip,
                        'status': 'failed',
                        'status_detail': 'product_agent_failed'
                    }
                    self._safe_write_json(self._final_output_path, failed_result, ctx, "final_output")
                    ctx.stats.add_error(ctx.worker_id)
                    continue
                
                merged_record = self._merge_results(ip, product)
                merged_results.append(merged_record)
                # 实时写入合并结果
                self._safe_write_json(self._merged_output_path, merged_record, ctx, "merged_output")
            
            # === 步骤3: 校验 ===
            if not self.config.skip_check and merged_results:
                delay = ctx.get_delay()
                if delay > 0:
                    time.sleep(delay)
                
                check_batch = [{'ip': r['ip'], 'raw': json.dumps(r, ensure_ascii=False), 'data': r}
                              for r in merged_results]
                check_results, check_stats = ctx.check_analyst.process_batch(check_batch)
                
                # 检查限流
                if check_stats.get('insufficient_balance'):
                    ctx.log_error("余额不足")
                    self._shutdown_event.set()
                    return False, "insufficient_balance"
                
                if check_stats.get('rate_limited') or check_stats.get('security_limited'):
                    limit_type = "安全限制" if check_stats.get('security_limited') else "并发限制"
                    wait_seconds = ctx.handle_rate_limit()
                    self._set_global_rate_limit(wait_seconds, f"API{limit_type}")
                    return False, "rate_limited"
                
                # 更新token统计
                ctx.stats.add_tokens(ctx.worker_id, 
                                    check_stats.get('input_token_count', 0),
                                    check_stats.get('output_token_count', 0))
                
                # 处理校验结果
                self._process_check_results(ctx, check_results, merged_results)
            else:
                # 跳过校验
                for merged in merged_results:
                    merged['validation_status'] = 'skipped'
                    merged['status'] = 'done'
                    merged['status_detail'] = 'check_skipped'
                    
                    conf = merged.get('confidence', 0)
                    ctx.stats.add_confidence(ctx.worker_id, conf)
                    # 实时写入最终结果
                    self._safe_write_json(self._final_output_path, merged, ctx, "final_output")
            
            # 更新处理计数
            ctx.stats.add_processed(ctx.worker_id, len(batch))
            ctx.log_info(f"批次 {batch_id} 处理完成")
            
            return True, ""
            
        except Exception as e:
            ctx.log_error(f"批次 {batch_id} 处理异常: {e}")
            # 为每条记录添加错误状态
            for r in batch:
                error_result = {
                    'ip': r['ip'],
                    'validation_status': 'exception',
                    'status': 'failed',
                    'status_detail': f'batch_exception: {str(e)}'
                }
                # 实时写入错误结果
                self._safe_write_json(self._final_output_path, error_result, ctx, "final_output")
            ctx.stats.add_error(ctx.worker_id, len(batch))
            return False, str(e)
    
    def _merge_results(self, ip: str, product: Dict) -> Dict:
        """构建产品分析结果"""
        merged = {
            'ip': ip,
            'vendor': product.get('vendor'),
            'model': product.get('model'),
            'os': product.get('os'),
            'firmware': product.get('firmware'),
            'type': product.get('type'),
            'result_type': product.get('result_type'),
            'confidence': product.get('confidence', 0),
            'evidence': product.get('evidence', []),
            'conclusion': product.get('conclusion', '')
        }
        # 标准化字段值
        return self._normalize_fields(merged)
    
    def _normalize_fields(self, record: Dict) -> Dict:
        """
        标准化字段值：
        - 所有属性字段未知时统一为 null
        - model字段为列表格式 [[name, conf], ...]
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
    
    def _process_check_results(self, ctx: WorkerContext, 
                               check_results: List[Dict], 
                               merged_results: List[Dict]) -> None:
        """处理校验结果"""
        from .config import DEBUG_MODE
        
        # 记录输入和输出IP用于调试
        input_ips = [r['ip'] for r in merged_results]
        output_ips = [r.get('ip') for r in (check_results or [])]
        
        ctx.log_debug(f"Check输入IP数量: {len(input_ips)}")
        ctx.log_debug(f"Check输出IP数量: {len(output_ips)}")
        ctx.log_debug(f"Check输入IP样本: {input_ips[:3]}")
        ctx.log_debug(f"Check输出IP样本: {output_ips[:3]}")
        
        # 计算匹配率
        input_set = set(input_ips)
        output_set = set(output_ips)
        match_count = len(input_set & output_set)
        match_rate = match_count / len(input_set) if input_set else 0
        
        ctx.log_info(f"Check IP匹配率: {match_rate*100:.1f}% ({match_count}/{len(input_set)})")
        
        if match_rate < 0.5:
            ctx.log_error(f"[警告] Check批次混淆: 匹配率仅{match_rate*100:.1f}%")
            # 记录不匹配的IP样本
            only_in_input = list(input_set - output_set)[:5]
            only_in_output = list(output_set - input_set)[:5]
            ctx.log_error(f"仅在输入中的IP样本: {only_in_input}")
            ctx.log_error(f"仅在输出中的IP样本: {only_in_output}")
        
        check_map = {r.get('ip'): r for r in (check_results or [])}
        merged_map = {r['ip']: r for r in merged_results}
        
        for merged in merged_results:
            ip = merged['ip']
            check_res = check_map.get(ip)
            
            if check_res:
                validation_status = check_res.get('validation_status', 'unknown')
                ctx.stats.add_validation(validation_status)
                
                # 实时写入校验结果
                self._safe_write_json(self._check_output_path, check_res, ctx, "check_output")
                final_result = ctx.check_analyst._build_final_result(check_res, merged)
                final_result['status'] = 'done'
                final_result['status_detail'] = 'all_agents_completed'
                
                conf = final_result.get('confidence', 0)
                ctx.stats.add_confidence(ctx.worker_id, conf)
                # 实时写入最终结果
                self._safe_write_json(self._final_output_path, final_result, ctx, "final_output")
                
                ctx.log_debug(f"校验: {ip} -> {validation_status}, conf: {conf}")
            else:
                ctx.log_error(f"IP {ip} 校验结果解析失败")
                print(f"\nIP {ip} 校验结果解析失败")
                
                # DEBUG模式：输出详细信息
                if DEBUG_MODE:
                    print(f"[DEBUG] IP {ip} 校验结果解析失败")
                    print(f"[DEBUG] 期望的IP列表: {list(merged_map.keys())}")
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
                # 实时写入失败结果
                self._safe_write_json(self._final_output_path, merged, ctx, "final_output")
                ctx.stats.add_error(ctx.worker_id)
    
    def _worker_loop(self, worker_id: int) -> None:
        """工作线程主循环"""
        ctx = WorkerContext(
            worker_id=worker_id,
            config=self.config,
            logger=self.logger,
            stats=self.stats,
            file_writer=self.file_writer,
            product_model=self.product_model,
            check_model=self.check_model
        )
        
        ctx.log_info(f"工作线程 {worker_id} 启动, 速度等级: {self.config.speed_level}")
        
        while not self._shutdown_event.is_set():
            # 检查暂停
            self._pause_event.wait()
            
            # 获取任务
            task = self.task_manager.get_next_task(worker_id)
            if task is None:
                if self.task_manager.is_all_done:
                    break
                continue
            
            # 处理任务
            success, error_msg = self._process_batch_with_retry(ctx, task)
            
            # 标记任务完成
            self.task_manager.complete_task(task.batch_id, success, error_msg)
            
            # 通知进度
            self._notify_progress()
            
            # 如果是余额不足，停止所有线程
            if error_msg == "insufficient_balance":
                self._shutdown_event.set()
                break
        
        ctx.log_info(f"工作线程 {worker_id} 结束")

    def _progress_printer(self) -> None:
        """进度打印线程"""
        import sys
        
        last_print_time = 0
        print_interval = 2.0  # 每2秒打印一次进度
        countdown_interval = 1.0  # 每1秒更新一次倒计时
        last_countdown_time = 0
        was_waiting = False  # 记录上一次是否在等待状态
        
        while not self._shutdown_event.is_set() and not self.task_manager.is_all_done:
            now = time.time()
            
            # 检查是否在等待状态
            is_waiting, remaining, reason = self._get_waiting_state()
            
            if is_waiting:
                # 显示倒计时（在同一行更新）
                if now - last_countdown_time >= countdown_interval:
                    self._print_countdown(remaining, reason)
                    last_countdown_time = now
                was_waiting = True
                time.sleep(0.5)
            else:
                # 如果刚从等待状态恢复，先清除倒计时行并打印恢复信息
                if was_waiting:
                    sys.stdout.write('\r' + ' ' * 100 + '\r')
                    sys.stdout.write(f'[等待结束] 继续执行...\n')
                    sys.stdout.flush()
                    was_waiting = False
                
                # 显示正常进度
                if now - last_print_time >= print_interval:
                    self._print_progress()
                    last_print_time = now
                time.sleep(0.5)
        
        # 最终打印
        if was_waiting:
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.flush()
        self._print_progress()
    
    def _print_countdown(self, remaining_seconds: int, reason: str) -> None:
        """打印倒计时（始终在同一行显示，不换行）"""
        import sys
        
        # 格式化剩余时间
        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        seconds = remaining_seconds % 60
        
        if hours > 0:
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = f"{minutes:02d}:{seconds:02d}"
        
        # 显示倒计时（使用\r覆盖当前行，不换行）
        sys.stdout.write(f'\r[{reason}] 剩余等待时间: {time_str}   ')
        sys.stdout.flush()
    
    def _print_progress(self) -> None:
        """打印当前进度"""
        stats = self.stats.get_stats()
        task_stats = self.task_manager.get_stats()
        
        processed, total, pct = self.stats.get_progress()
        
        # 计算费用
        cost = self._calculate_cost(stats['input_tokens'], stats['output_tokens'])
        
        # 计算实时平均（考虑正在处理中的批次，避免多线程启动时估算偏高）
        this_run_processed = processed - stats['skipped_records']
        in_progress_count = task_stats.get('in_progress_ips', 0)
        
        # 有效处理数 = 已完成数 + 正在处理数（正在处理的批次已经消耗了token但还没计入processed）
        # 这样可以避免多线程启动时，token已累加但processed还很少导致的估算偏高
        effective_processed = this_run_processed + in_progress_count
        
        if effective_processed > 0:
            avg_input = stats['this_run_input_tokens'] / effective_processed
            avg_output = stats['this_run_output_tokens'] / effective_processed
        else:
            avg_input = avg_output = 0
        
        # 估算总费用
        if avg_input > 0 or avg_output > 0:
            total_est_input = avg_input * total
            total_est_output = avg_output * total
            total_est_cost = self._calculate_cost(total_est_input, total_est_output)
        else:
            total_est_cost = 0
        
        # 计算剩余时间
        elapsed = stats['execution_time_seconds']
        if this_run_processed > 0:
            avg_time = elapsed / this_run_processed
            remaining = avg_time * (total - processed)
        else:
            remaining = 0
        
        # 格式化时间
        def fmt_time(s):
            if s < 60:
                return f"{s:.0f}s"
            elif s < 3600:
                return f"{s/60:.1f}m"
            else:
                return f"{s/3600:.1f}h"
        
        # 打印进度行（与单线程模式格式一致）
        # 使用 CNY 替代 ¥ 符号，避免 Windows GBK 编码问题
        print(f"进度: {pct:5.1f}% {processed:>4}/{total:<4}  "
              f"高置信度: {stats['high_conf']:<3} 中置信度: {stats['mid_conf']:<3} 不可信: {stats['low_conf']:<3} 错误: {stats['error_count']:<3} | "
              f"Verified: {stats['verified_count']:<2} Adjust: {stats['adjusted_count']:<2} Reject: {stats['rejected_count']:<2} | "
              f"Tok/条: {avg_input:.0f}/{avg_output:.0f}  CNY{cost:.4f}(总估CNY{total_est_cost:.4f})  用时: {fmt_time(elapsed)} / 剩余: {fmt_time(remaining)} | "
              f"线程: {task_stats['active_workers']}活跃")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """计算费用，使用 product_model 的价格（如果指定了的话）"""
        actual_model = self.product_model if self.product_model else MODEL_NAME
        prices = MODEL_PRICES.get(actual_model, MODEL_PRICES.get('default'))
        input_cost = input_tokens * prices['input'] / 1000
        output_cost = output_tokens * prices['output'] / 1000
        return input_cost + output_cost
    
    def _save_run_state(self, run_config: Dict = None) -> None:
        """保存运行状态"""
        stats = self.stats.get_stats()
        task_stats = self.task_manager.get_stats()
        
        state = {
            'task_id': getattr(self, '_task_id', 1),
            'last_update': datetime.now().isoformat(),
            'start_time': datetime.fromtimestamp(self._start_time).isoformat() if self._start_time else datetime.now().isoformat(),
            'elapsed_seconds': time.time() - self._start_time,
            'stats': stats,
            'task_stats': task_stats,
            'config': {
                'num_workers': self.config.num_workers,
                'speed_level': self.config.speed_level,
                'batch_size': self.config.batch_size
            }
        }
        
        # 保存完整的运行配置（用于retry功能）
        if run_config:
            state['run_config'] = run_config
        elif hasattr(self, '_run_config'):
            state['run_config'] = self._run_config
        
        self.file_writer.write_json(self._run_state_path, state)
    
    def _load_run_state(self) -> Tuple[Optional[Dict], float, int]:
        """加载运行状态"""
        if not os.path.exists(self._run_state_path):
            return None, 0, 0
        
        try:
            with open(self._run_state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            return state.get('stats'), state.get('elapsed_seconds', 0), state.get('task_id', 1)
        except (json.JSONDecodeError, IOError):
            return None, 0, 0
    
    def run(self, records: List[Dict], task_id: int = 1) -> Dict:
        """
        运行多线程处理
        
        Args:
            records: 待处理的记录列表
            task_id: 任务ID
        
        Returns:
            统计信息字典
        """
        from .run import load_stats_from_results
        
        self._task_id = task_id
        self._start_time = time.time()
        
        # 初始化日志
        self.logger = ThreadSafeLogger(self._log_dir, task_id)
        self.logger.log_main('info', f"多线程执行器启动 | 工作线程数: {self.config.num_workers} | "
                            f"速度等级: {self.config.speed_level} | 批次大小: {self.config.batch_size}")
        
        # 加载已处理的IP（断点续传）
        processed_count = self.task_manager.load_processed_ips(self._final_output_path)
        
        # 尝试加载上次运行状态
        prev_stats, prev_elapsed, prev_task_id = self._load_run_state()
        
        total_records = len(records)
        skipped_records = processed_count
        
        # 检查是否是retry模式
        retry_adjustment = getattr(self, '_retry_stats_adjustment', None)
        
        if retry_adjustment:
            # RETRY模式：所有统计从0开始，进度为 0/重试总数
            self.stats.set_total(total_records, 0)
            skipped_records = 0
            self.logger.log_main('info', f"RETRY模式: 重试 {total_records} 条")
            print(f"[RETRY模式] 重试 {total_records} 条，统计从0开始")
        elif skipped_records > 0:
            # 断点续传模式：从结果文件恢复统计
            result_stats = load_stats_from_results(self._final_output_path)
            
            with self.stats._lock:
                self.stats._stats['total_records'] = total_records
                self.stats._stats['skipped_records'] = skipped_records
                self.stats._stats['processed_records'] = skipped_records
                self.stats._stats['error_count'] = result_stats['error_count']
                self.stats._stats['high_conf'] = result_stats['high_conf']
                self.stats._stats['mid_conf'] = result_stats['mid_conf']
                self.stats._stats['low_conf'] = result_stats['low_conf']
                self.stats._stats['verified_count'] = result_stats['verified_count']
                self.stats._stats['adjusted_count'] = result_stats['adjusted_count']
                self.stats._stats['rejected_count'] = result_stats['rejected_count']
                
                if prev_stats:
                    self.stats._stats['input_tokens'] = prev_stats.get('input_tokens', 0)
                    self.stats._stats['output_tokens'] = prev_stats.get('output_tokens', 0)
                    self.stats._stats['realtime_avg_input'] = prev_stats.get('realtime_avg_input', 0)
                    self.stats._stats['realtime_avg_output'] = prev_stats.get('realtime_avg_output', 0)
            
            self.logger.log_main('info', f"断点续传: 已处理 {skipped_records} 条")
            print(f"[断点续传] 恢复统计: 已处理{skipped_records}条, "
                  f"高置信度{result_stats['high_conf']} 中置信度{result_stats['mid_conf']} "
                  f"不可信{result_stats['low_conf']} 错误{result_stats['error_count']}")
        else:
            # 新任务：所有统计从0开始
            self.stats.set_total(total_records, 0)
        
        # 添加任务（会自动过滤已处理的IP）
        task_count = self.task_manager.add_tasks(records, self.config.batch_size)
        
        if task_count == 0:
            print(f"所有记录已处理完成！（跳过 {skipped_records} 条已处理数据）")
            return self.stats.get_stats()
        
        self.logger.log_main('info', f"创建 {task_count} 个批次任务, 待处理: {total_records - skipped_records} 条")
        print(f"\n[多线程模式] 工作线程: {self.config.num_workers} | 批次数: {task_count} | "
              f"待处理: {total_records - skipped_records} 条")
        print("-" * 100)
        
        # 确保输出目录存在
        for path in [self._product_output_path, self._merged_output_path,
                    self._check_output_path, self._final_output_path]:
            output_dir = os.path.dirname(path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # 启动进度打印线程
        progress_thread = threading.Thread(target=self._progress_printer, daemon=True)
        progress_thread.start()
        
        # 启动工作线程
        threads = []
        for i in range(self.config.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,))
            t.start()
            threads.append(t)
        
        # 定期保存状态
        insufficient_balance_detected = False
        try:
            while not self.task_manager.is_all_done and not self._shutdown_event.is_set():
                time.sleep(5)
                self._save_run_state()
                # 检查是否因余额不足而停止
                if self._shutdown_event.is_set():
                    # 检查是否有任务因余额不足失败
                    for task in self._tasks.values() if hasattr(self, '_tasks') else []:
                        if getattr(task, 'error_msg', None) == "insufficient_balance":
                            insufficient_balance_detected = True
                            break
        except KeyboardInterrupt:
            print("\n\n[中断] 正在保存状态...")
            self._shutdown_event.set()
        
        # 等待所有工作线程结束
        for t in threads:
            t.join(timeout=10)
        
        # 最终保存状态
        self._save_run_state()
        
        # 检查是否因余额不足退出
        task_stats = self.task_manager.get_stats()
        if self._shutdown_event.is_set() and task_stats.get('failed_tasks', 0) > 0:
            # 检查失败任务是否因余额不足
            for batch_id, task in self.task_manager._tasks.items():
                if task.error_msg == "insufficient_balance":
                    insufficient_balance_detected = True
                    break
        
        if insufficient_balance_detected:
            print("\n[ERROR] 余额不足，当前状态已保存。")
            print("[INFO] 请充值后重新运行以继续处理。")
        
        # 打印最终统计
        print("\n" + "-" * 100)
        self._print_final_stats()
        
        self.logger.log_main('info', f"多线程执行完成")
        
        # 关闭日志handlers，避免文件句柄泄漏
        if self.logger:
            self.logger.close()
        
        return self.stats.get_stats()
    
    def _print_final_stats(self) -> None:
        """打印最终统计（与单线程模式格式一致）"""
        stats = self.stats.get_stats()
        task_stats = self.task_manager.get_stats()
        
        elapsed = stats['execution_time_seconds']
        cost = self._calculate_cost(stats['input_tokens'], stats['output_tokens'])
        
        # 计算实时平均
        this_run_processed = stats['processed_records'] - stats['skipped_records']
        if this_run_processed > 0:
            avg_input = stats['this_run_input_tokens'] / this_run_processed
            avg_output = stats['this_run_output_tokens'] / this_run_processed
            total_est_cost = self._calculate_cost(
                avg_input * stats['total_records'],
                avg_output * stats['total_records']
            )
        else:
            avg_input = avg_output = total_est_cost = 0
        
        # 格式化时间
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}m"
        else:
            time_str = f"{elapsed/3600:.1f}h"
        
        print(f"完成: {stats['processed_records']}条(本次{this_run_processed}条) | 高置信度: {stats['high_conf']} 中置信度: {stats['mid_conf']} "
              f"不可信: {stats['low_conf']} 错误: {stats['error_count']} | "
              f"Verified: {stats['verified_count']} Adjust: {stats['adjusted_count']} Reject: {stats['rejected_count']}")
        # 使用 CNY 替代 ¥ 符号，避免 Windows GBK 编码问题
        print(f"Tok/条: {avg_input:.0f}/{avg_output:.0f} | "
              f"费用: CNY{cost:.4f} (总估: CNY{total_est_cost:.4f}) | 总用时: {time_str} | "
              f"任务: 完成{task_stats['completed_tasks']} 失败{task_stats['failed_tasks']}")
    
    def pause(self) -> None:
        """暂停处理"""
        self._pause_event.clear()
        self.logger.log_main('info', "处理已暂停")
    
    def resume(self) -> None:
        """恢复处理"""
        self._pause_event.set()
        self.logger.log_main('info', "处理已恢复")
    
    def shutdown(self) -> None:
        """关闭执行器"""
        self._shutdown_event.set()
        self.task_manager.shutdown()
        if self.logger:
            self.logger.log_main('info', "执行器已关闭")
            self.logger.close()  # 关闭日志handlers
    
    def set_run_config(self, run_config: Dict) -> None:
        """设置运行配置（用于retry功能）"""
        self._run_config = run_config
    
    def set_retry_stats_adjustment(self, adjustment: Dict) -> None:
        """
        设置retry模式的统计调整信息
        
        Args:
            adjustment: 包含需要从统计中减去的数量
                - retry_count: 重试的IP数量
                - subtract_low_conf: 需要从low_conf减去的数量
                - subtract_error: 需要从error_count减去的数量
        """
        self._retry_stats_adjustment = adjustment


def run_multi_thread_pipeline(
    input_path: str = None,
    max_records: int = None,
    num_workers: int = 2,
    speed_level: str = 's',
    skip_check: bool = False,
    skip_clean: bool = False,
    restart: bool = False,
    task_id: int = None,
    product_model: str = None,
    check_model: str = None,
    run_config: Dict = None,
    retry_ips: List[str] = None,
    retry_stats_adjustment: Dict = None,
    entropy_ratio: float = None,
    uniform_sample: bool = False
) -> Dict:
    """
    多线程流水线入口函数
    
    Args:
        input_path: 输入文件路径
        max_records: 最大处理记录数
        num_workers: 工作线程数
        speed_level: 速度等级
        skip_check: 是否跳过校验
        skip_clean: 是否跳过数据清洗（当输入已经是清洗后的数据时使用）
        restart: 是否重新开始（清除已有结果）
        task_id: 任务ID（由调用方传入，确保与主程序同步）
        product_model: 产品Agent使用的模型
        check_model: 校验Agent使用的模型
        run_config: 运行配置（用于保存到run_state.json，支持retry功能）
        retry_ips: 需要重试的IP列表（retry模式使用）
        retry_stats_adjustment: retry模式的统计调整信息
        entropy_ratio: 信息熵筛选比例，None使用默认值(0.75)，设为1.0则不筛选
        uniform_sample: 是否启用均匀采样（在熵排序后执行）
    
    Returns:
        统计信息字典
    """
    from .data_cleaner import DataCleaner
    from .config import INPUT_DIR, DEFAULT_ENTROPY_RATIO
    from .entropy_sorter import isort, uniform_sample as do_uniform_sample
    from .run import get_current_task_type
    
    # 获取当前模式下的路径
    paths = get_output_paths()
    cleaned_data_path = paths['cleaned_data']
    product_output_path = paths['product_output']
    merged_output_path = paths['merged_output']
    check_output_path = paths['check_output']
    final_output_path = paths['final_output']
    run_state_path = paths['run_state']
    temp_dir = paths['temp_dir']  # 临时文件目录
    
    input_path = input_path or INPUT_DIR
    
    # 确定信息熵筛选比例
    if entropy_ratio is None:
        entropy_ratio = DEFAULT_ENTROPY_RATIO
    
    # 专属任务模式：自动清除该任务的专属中间文件（避免断点续传干扰）
    current_task_type = get_current_task_type()
    if current_task_type and not retry_ips:
        # 在专属任务模式下（os/vd/dt），默认清除该任务的所有中间文件
        files_to_clear = [
            cleaned_data_path,
            product_output_path,
            merged_output_path,
            check_output_path,
            final_output_path,
            run_state_path
        ]
        cleared_count = 0
        for path in files_to_clear:
            if os.path.exists(path):
                os.remove(path)
                cleared_count += 1
        
        if cleared_count > 0:
            print(f"[INFO] 专属任务模式 ({current_task_type}): 已清除 {cleared_count} 个中间文件，从头开始处理")
    
    # 专属任务模式：自动清除该任务的专属中间文件（避免断点续传干扰）
    # 必须在创建 MultiThreadRunner 之前清除，因为 runner.run() 会加载这些文件
    current_task_type = get_current_task_type()
    if current_task_type and not retry_ips:
        # 在专属任务模式下（os/vd/dt），默认清除该任务的所有中间文件
        files_to_clear = [
            cleaned_data_path,
            product_output_path,
            merged_output_path,
            check_output_path,
            final_output_path,
            run_state_path
        ]
        cleared_count = 0
        for path in files_to_clear:
            if os.path.exists(path):
                os.remove(path)
                cleared_count += 1
        
        if cleared_count > 0:
            print(f"[INFO] 专属任务模式 ({current_task_type}): 已清除 {cleared_count} 个中间文件，从头开始处理")
    
    # retry模式提示
    if retry_ips:
        print(f"\n[RETRY模式] 需要重试 {len(retry_ips)} 个IP")
    
    # 数据清洗（可跳过）
    if not skip_clean:
        # 直接调用 run_cleaner 函数，确保使用统一的清洗逻辑（包括难度分级采样）
        from .run import run_cleaner
        
        # 注意：run_config 中可能包含 vendor_balance, difficulty_ratios, max_vendor_ratio 等参数
        vendor_balance = run_config.get('vendor_balance', False) if run_config else False
        difficulty_ratios = run_config.get('difficulty_ratios') if run_config else None
        max_vendor_ratio = run_config.get('max_vendor_ratio') if run_config else None
        
        clean_stats = run_cleaner(
            input_path=input_path,
            max_records=max_records,
            entropy_ratio=entropy_ratio,
            vendor_balance=vendor_balance,
            uniform_sample=uniform_sample,
            difficulty_ratios=difficulty_ratios,
            max_vendor_ratio=max_vendor_ratio
        )
    else:
        print("\n" + "=" * 60)
        print("步骤 1: 跳过数据清洗（使用已清洗的数据）")
        print("=" * 60)
        print(f"输入文件: {cleaned_data_path}")
    
    # 加载清洗后的数据
    records = []
    with open(cleaned_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ip = next(iter(obj.keys()))
                records.append({'ip': ip, 'raw': line, 'data': obj})
            except (json.JSONDecodeError, StopIteration):
                continue
    
    # retry模式：只保留需要重试的IP
    if retry_ips:
        retry_ip_set = set(retry_ips)
        records = [r for r in records if r['ip'] in retry_ip_set]
        print(f"[RETRY模式] 从清洗数据中筛选出 {len(records)} 条需要重试的记录")
    elif max_records:
        records = records[:max_records]
    
    # 非专属任务模式下，如果指定了 restart 参数，清除已有结果（retry模式不清除）
    # 注意：专属任务模式已经在函数开头自动清除了
    if restart and not retry_ips and not current_task_type:
        for path in [product_output_path, merged_output_path,
                    check_output_path, final_output_path, run_state_path]:
            if os.path.exists(path):
                os.remove(path)
        print("[INFO] 已清除之前的结果文件")
    
    # 使用传入的任务ID，如果没有传入则从状态文件获取
    if task_id is None:
        task_id = 1
        if os.path.exists(run_state_path):
            try:
                with open(run_state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # 断点续传时使用相同的任务ID
                    task_id = state.get('task_id', 1)
            except:
                pass
    
    print(f"[INFO] 任务ID: {task_id}")
    
    # 创建配置
    config = MultiThreadConfig(
        num_workers=num_workers,
        speed_level=speed_level,
        skip_check=skip_check
    )
    
    # 运行多线程处理
    print("\n" + "=" * 60)
    if retry_ips:
        print("步骤 2-3: 多线程重试流水线 (RETRY模式)")
    else:
        print("步骤 2-3: 多线程分析流水线")
    print("=" * 60)
    
    runner = MultiThreadRunner(config, product_model, check_model)
    
    # 设置运行配置（用于保存到run_state.json）
    if run_config:
        runner.set_run_config(run_config)
    
    # 设置retry统计调整
    if retry_stats_adjustment:
        runner.set_retry_stats_adjustment(retry_stats_adjustment)
    
    # retry模式：需要特殊处理，先删除这些IP的旧结果
    if retry_ips:
        _remove_ips_from_results(retry_ips, [
            product_output_path, merged_output_path,
            check_output_path, final_output_path
        ])
        # 清除任务管理器中这些IP的已处理状态
        runner.task_manager._processed_ips -= set(retry_ips)
    
    stats = runner.run(records, task_id)
    
    return stats


def _remove_ips_from_results(ips_to_remove: List[str], file_paths: List[str]) -> None:
    """
    从结果文件中删除指定IP的记录
    
    Args:
        ips_to_remove: 需要删除的IP列表
        file_paths: 结果文件路径列表
    """
    ip_set = set(ips_to_remove)
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        
        # 读取所有记录，过滤掉需要删除的IP
        kept_records = []
        removed_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    ip = record.get('ip')
                    if ip and ip in ip_set:
                        removed_count += 1
                        continue
                    kept_records.append(line)
                except json.JSONDecodeError:
                    kept_records.append(line)
        
        # 重写文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in kept_records:
                f.write(line + '\n')
        
        if removed_count > 0:
            print(f"  从 {os.path.basename(file_path)} 中删除 {removed_count} 条旧记录")
