"""
分析Agent基类
提供通用的数据加载、Token计算、批处理和API调用功能
"""

import json
import re
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

from openai import OpenAI

from .config import (
    API_KEY, BASE_URL, MODEL_NAME,
    BATCH_SIZE, MAX_INPUT_TOKENS, MAX_RECORDS
)
from .utils.token_counter import get_tokenizer, count_tokens, count_message_tokens, count_output_tokens


# 全局批次ID计数器
_batch_id_counter = 0
_batch_id_lock = threading.Lock()


def _get_next_batch_id() -> int:
    """获取下一个批次ID"""
    global _batch_id_counter
    with _batch_id_lock:
        _batch_id_counter += 1
        return _batch_id_counter


class BaseAnalyst(ABC):
    """分析Agent基类"""
    
    SYSTEM_PROMPT: str = ""
    AGENT_NAME: str = "BaseAnalyst"  # 子类应覆盖此属性
    
    # 安全限制容错阈值：连续多少次检测到安全限制才真正认为触发
    SECURITY_LIMIT_THRESHOLD = 5
    
    def __init__(self, input_path: str, output_path: str, model_name: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.tokenizer = get_tokenizer()
        # 使用自定义模型或默认模型
        self.model_name = model_name if model_name else MODEL_NAME
        # 安全限制连续计数器
        self._consecutive_security_count = 0
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'successful_records': 0,
            'failed_records': 0,
            'token_exceeded_batches': 0,
            'total_tokens_used': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'execution_time_seconds': 0
        }
    
    def load_records(self, max_count: int = None) -> List[Dict]:
        """加载清洗后的数据"""
        records = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_count is not None and i >= max_count:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ip = next(iter(obj.keys()))
                    records.append({'ip': ip, 'raw': line, 'data': obj})
                except (json.JSONDecodeError, StopIteration):
                    continue
        return records
    
    def count_tokens(self, messages: List[Dict]) -> int:
        return count_message_tokens(messages, self.tokenizer)
    
    @abstractmethod
    def build_prompt(self, batch: List[Dict], batch_id: int = None) -> List[Dict]:
        pass
    
    @abstractmethod
    def parse_response(self, response: str, batch_id: int = None, expected_ips: List[str] = None) -> List[Dict]:
        pass
    
    def process_batch(self, batch: List[Dict]) -> Tuple[List[Dict], Dict]:
        """处理单个批次"""
        from .utils.logger import log_debug, log_error, log_exception
        from .utils.error_logger import (
            start_batch_context, record_batch_step, record_prompt_input,
            record_raw_response, end_batch_context, log_parse_error,
            log_api_error, log_batch_exception
        )
        
        # 获取批次ID
        batch_id = _get_next_batch_id()
        batch_ips = [r['ip'] for r in batch]
        
        # 开始记录批次上下文
        start_batch_context(batch_id, batch_ips, self.AGENT_NAME)
        record_batch_step(batch_id, "开始处理批次", f"包含 {len(batch)} 条记录")
        
        # 错误关键词定义
        error_keywords_rate_limit = ['rate limit', 'too many requests', 
                                     '请求过于频繁', '并发', 'concurrent', '频率限制',
                                     'rate_limit', 'ratelimit', '429']
        error_keywords_security = ['安全策略', '拦截', '安全限制', 'security', 'blocked',
                                   '风控', '异常请求', 'forbidden', '403']
        error_keywords_balance = ['余额不足', 'insufficient balance', 'quota exceeded', 
                                  '额度不足', '账户余额', 'billing', '欠费']
        
        batch_stats = {
            'input_token_count': 0,
            'output_token_count': 0,
            'token_exceeded': False,
            'success': False,
            'error': None,
            'rate_limited': False,  # 是否触发并发限制
            'security_limited': False,  # 是否触发安全限制
            'insufficient_balance': False,  # 是否余额不足
            'batch_id': batch_id,  # 记录批次ID用于错误追踪
        }
        
        record_batch_step(batch_id, "构建提示词")
        messages = self.build_prompt(batch, batch_id)
        
        # 记录完整的提示词输入
        record_prompt_input(batch_id, messages)
        
        input_token_count = self.count_tokens(messages)
        batch_stats['input_token_count'] = input_token_count
        record_batch_step(batch_id, "计算Token", f"input_tokens={input_token_count}")
        
        if input_token_count > MAX_INPUT_TOKENS:
            log_debug(f"Token超限: {input_token_count} > {MAX_INPUT_TOKENS}")
            record_batch_step(batch_id, "Token超限", f"{input_token_count} > {MAX_INPUT_TOKENS}")
            batch_stats['token_exceeded'] = True
            self.stats['token_exceeded_batches'] += 1
        
        try:
            log_debug(f"调用API, IPs: {batch_ips}")
            record_batch_step(batch_id, "调用API", f"model={self.model_name}")
            
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
            )
            reply = resp.choices[0].message.content.strip() if resp.choices else ""
            
            # 记录原始返回结果
            record_raw_response(batch_id, reply)
            record_batch_step(batch_id, "收到API响应", f"长度={len(reply)}")
            
            # 判断是否为正常的JSON响应
            # 检查多个常规字段，任意一个存在即认为是正常数据（不区分大小写）
            reply_lower_for_check = reply.lower()
            normal_fields = ['"ip"', '"confidence"', '"vendor"', '"model"', 
                           '"validation_status"', '"evidence"',
                           '"firmware"', '"os"', '"type"', '"device type"',
                           '"services"', '"country"', '"asn"', '"org"']
            is_normal_response = any(field in reply_lower_for_check for field in normal_fields)
            
            # 额外检查：如果响应以 [ 或 { 开头，且包含多个JSON特征，也认为是正常响应
            reply_stripped = reply.strip()
            if not is_normal_response:
                json_indicators = ['":', '",', '{}', '[]', 'null', 'true', 'false']
                json_indicator_count = sum(1 for ind in json_indicators if ind in reply)
                if (reply_stripped.startswith('[') or reply_stripped.startswith('{')) and json_indicator_count >= 3:
                    is_normal_response = True
            
            # 只有非正常响应才检查错误关键词
            if not is_normal_response:
                reply_lower = reply.lower()
                
                # 检查余额不足
                for keyword in error_keywords_balance:
                    if keyword.lower() in reply_lower:
                        log_error(f"检测到余额不足 [关键词: {keyword}]: {reply[:500]}")
                        log_api_error(batch_id, "insufficient_balance", 
                                     f"检测到余额不足 [关键词: {keyword}]")
                        batch_stats['insufficient_balance'] = True
                        batch_stats['error'] = f"余额不足 [关键词: {keyword}]: {reply[:500]}"
                        return [], batch_stats
                
                # 检查并发限制
                for keyword in error_keywords_rate_limit:
                    if keyword.lower() in reply_lower:
                        log_error(f"检测到并发限制 [关键词: {keyword}]: {reply[:500]}")
                        log_api_error(batch_id, "rate_limit", 
                                     f"检测到并发限制 [关键词: {keyword}]")
                        batch_stats['rate_limited'] = True
                        batch_stats['error'] = f"并发限制 [关键词: {keyword}]: {reply[:500]}"
                        return [], batch_stats
                
                # 检查安全限制（需要连续多次才真正触发）
                for keyword in error_keywords_security:
                    if keyword.lower() in reply_lower:
                        self._consecutive_security_count += 1
                        log_debug(f"检测到安全限制关键词 [{keyword}], 连续计数: {self._consecutive_security_count}/{self.SECURITY_LIMIT_THRESHOLD}")
                        
                        if self._consecutive_security_count >= self.SECURITY_LIMIT_THRESHOLD:
                            log_error(f"连续 {self._consecutive_security_count} 次检测到安全限制 [关键词: {keyword}]: {reply[:500]}")
                            log_api_error(batch_id, "security_limit", 
                                         f"连续 {self._consecutive_security_count} 次检测到安全限制 [关键词: {keyword}]")
                            batch_stats['security_limited'] = True
                            batch_stats['error'] = f"安全限制 [关键词: {keyword}]: {reply[:500]}"
                            return [], batch_stats
                        else:
                            # 未达到阈值，记录警告但继续处理
                            log_debug(f"安全限制警告 ({self._consecutive_security_count}/{self.SECURITY_LIMIT_THRESHOLD}): {reply[:200]}")
                            batch_stats['error'] = f"安全限制警告 ({self._consecutive_security_count}/{self.SECURITY_LIMIT_THRESHOLD})"
                            return [], batch_stats
            
            # 计算output token
            output_token_count = count_output_tokens(reply, self.tokenizer)
            batch_stats['output_token_count'] = output_token_count
            
            # 更新统计
            self.stats['total_input_tokens'] += input_token_count
            self.stats['total_output_tokens'] += output_token_count
            self.stats['total_tokens_used'] += input_token_count + output_token_count
            
            log_debug(f"API响应长度: {len(reply)}, output tokens: {output_token_count}")
            
        except Exception as e:
            error_str = str(e).lower()
            record_batch_step(batch_id, "API调用异常", str(e))
            
            # 检查异常信息中是否包含余额不足
            for keyword in error_keywords_balance:
                if keyword.lower() in error_str:
                    log_error(f"API异常-余额不足: {e}")
                    log_api_error(batch_id, "insufficient_balance", f"API异常-余额不足", e)
                    batch_stats['insufficient_balance'] = True
                    batch_stats['error'] = f"余额不足: {e}"
                    return [], batch_stats
            
            # 检查异常信息中是否包含并发限制
            for keyword in error_keywords_rate_limit:
                if keyword.lower() in error_str:
                    log_error(f"API异常-并发限制: {e}")
                    log_api_error(batch_id, "rate_limit", f"API异常-并发限制", e)
                    batch_stats['rate_limited'] = True
                    batch_stats['error'] = f"并发限制: {e}"
                    return [], batch_stats
            
            # 检查异常信息中是否包含安全限制（需要连续多次才真正触发）
            for keyword in error_keywords_security:
                if keyword.lower() in error_str:
                    self._consecutive_security_count += 1
                    log_debug(f"API异常-安全限制关键词 [{keyword}], 连续计数: {self._consecutive_security_count}/{self.SECURITY_LIMIT_THRESHOLD}")
                    
                    if self._consecutive_security_count >= self.SECURITY_LIMIT_THRESHOLD:
                        log_error(f"连续 {self._consecutive_security_count} 次API异常-安全限制: {e}")
                        log_api_error(batch_id, "security_limit", f"连续 {self._consecutive_security_count} 次API异常-安全限制", e)
                        batch_stats['security_limited'] = True
                        batch_stats['error'] = f"安全限制: {e}"
                        return [], batch_stats
                    else:
                        # 未达到阈值，记录警告但继续处理
                        log_debug(f"API异常-安全限制警告 ({self._consecutive_security_count}/{self.SECURITY_LIMIT_THRESHOLD}): {e}")
                        batch_stats['error'] = f"安全限制警告 ({self._consecutive_security_count}/{self.SECURITY_LIMIT_THRESHOLD}): {e}"
                        return [], batch_stats
            
            log_exception(f"API调用失败: {e}")
            log_batch_exception(batch_id, e, "API调用")
            batch_stats['error'] = str(e)
            return [], batch_stats
        
        # 解析响应
        record_batch_step(batch_id, "开始解析响应")
        # 提取期望的IP列表用于验证
        expected_ips = [r['ip'] for r in batch]
        results = self.parse_response(reply, batch_id, expected_ips)
        
        if results:
            batch_stats['success'] = True
            # 成功处理，重置安全限制连续计数器
            self._consecutive_security_count = 0
            record_batch_step(batch_id, "解析成功", f"解析到 {len(results)} 条结果")
            # 成功完成，清除批次上下文
            end_batch_context(batch_id, success=True)
        else:
            record_batch_step(batch_id, "解析失败", "未能解析出有效结果")
            # 解析失败，记录错误（如果try_parse_json中没有记录的话）
            # 注意：try_parse_json中已经调用了log_parse_error，这里不再重复调用
            # 但需要确保上下文被清除
            end_batch_context(batch_id, success=False)
        
        return results, batch_stats
    
    def run(self, max_records: int = None) -> Dict:
        """执行分析主循环"""
        start_time = time.time()
        
        if max_records is None:
            max_records = MAX_RECORDS
        records = self.load_records(max_records)
        self.stats['total_records'] = len(records)
        
        all_results = []
        
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            results, batch_stats = self.process_batch(batch)
            
            if results:
                for res in results:
                    all_results.append(res)
                    self.stats['successful_records'] += 1
            else:
                self.stats['failed_records'] += len(batch)
            
            self.stats['processed_records'] += len(batch)
            
            # 实时输出平均token统计
            self._print_token_stats()
        
        if all_results:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        self.stats['execution_time_seconds'] = time.time() - start_time
        return self.stats
    
    def _print_token_stats(self) -> None:
        """实时输出平均input/output token统计"""
        processed = self.stats['processed_records']
        if processed > 0:
            avg_input = self.stats['total_input_tokens'] / processed
            avg_output = self.stats['total_output_tokens'] / processed
            print(f"  [Token] 已处理 {processed} 条 | 平均input: {avg_input:.2f} tokens/条 | 平均output: {avg_output:.2f} tokens/条")
    
    def _print_result(self, result: Dict) -> None:
        ip = result.get('ip', 'unknown')
        confidence = result.get('confidence', 0)
        print(f"  [OK] {ip}: conf={confidence}")
    
    @staticmethod
    def clean_json_response(reply: str) -> str:
        """清理API响应中的markdown标记"""
        reply = re.sub(r'^```\w*\n?', '', reply)
        reply = re.sub(r'\n?```$', '', reply)
        reply = reply.strip()
        
        # 提取JSON
        start_bracket = reply.find('[')
        start_brace = reply.find('{')
        
        if start_bracket == -1 and start_brace == -1:
            return reply
        
        if start_bracket == -1:
            start = start_brace
        elif start_brace == -1:
            start = start_bracket
        else:
            start = min(start_bracket, start_brace)
        
        if reply[start] == '[':
            end = reply.rfind(']')
        else:
            end = reply.rfind('}')
        
        if end > start:
            reply = reply[start:end+1]
        
        return reply.strip()
    
    @staticmethod
    def try_parse_json(reply: str, batch_id: int = None) -> List[Dict]:
        """尝试多种方式解析JSON"""
        from .utils.logger import log_debug, log_error
        from .utils.error_logger import log_parse_error, record_batch_step
        from .config import DEBUG_MODE
        
        original_reply = reply  # 保存原始响应用于debug输出
        reply = BaseAnalyst.clean_json_response(reply)
        
        if batch_id:
            record_batch_step(batch_id, "清理JSON响应", f"原始长度={len(original_reply)}, 清理后={len(reply)}")
        
        # 方法1: 直接解析
        try:
            results = json.loads(reply)
            if not isinstance(results, list):
                results = [results]
            if batch_id:
                record_batch_step(batch_id, "JSON直接解析成功", f"解析到 {len(results)} 条")
            return results
        except json.JSONDecodeError as e:
            log_debug(f"直接解析失败: {e}")
            if batch_id:
                record_batch_step(batch_id, "JSON直接解析失败", str(e))
        
        # 方法2: 修复尾部逗号
        try:
            fixed = re.sub(r',\s*([}\]])', r'\1', reply)
            results = json.loads(fixed)
            if not isinstance(results, list):
                results = [results]
            log_debug("修复尾部逗号后解析成功")
            if batch_id:
                record_batch_step(batch_id, "修复尾部逗号后解析成功", f"解析到 {len(results)} 条")
            return results
        except json.JSONDecodeError as e:
            if batch_id:
                record_batch_step(batch_id, "修复尾部逗号后仍失败", str(e))
        
        # 方法3: 按行解析
        try:
            results = []
            for line in reply.split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    results.append(json.loads(line))
            if results:
                log_debug(f"按行解析成功: {len(results)} 条")
                if batch_id:
                    record_batch_step(batch_id, "按行解析成功", f"解析到 {len(results)} 条")
                return results
        except json.JSONDecodeError as e:
            if batch_id:
                record_batch_step(batch_id, "按行解析失败", str(e))
        
        # 方法4: 正则提取
        try:
            results = []
            pattern = r'\{[^{}]*\}'
            matches = re.findall(pattern, reply)
            for match in matches:
                try:
                    results.append(json.loads(match))
                except:
                    pass
            if results:
                log_debug(f"正则提取成功: {len(results)} 条")
                if batch_id:
                    record_batch_step(batch_id, "正则提取成功", f"解析到 {len(results)} 条")
                return results
        except Exception as e:
            if batch_id:
                record_batch_step(batch_id, "正则提取失败", str(e))
        
        # 解析失败
        log_error(f"JSON解析失败, 响应前500字符: {reply[:500]}")
        
        # 记录详细错误日志（包含完整的原始响应）
        if batch_id:
            log_parse_error(
                batch_id,
                "json_parse_error",
                "所有JSON解析方法均失败",
                {
                    "原始响应长度": len(original_reply),
                    "清理后长度": len(reply),
                    "完整原始响应": original_reply,
                    "清理后响应": reply
                }
            )
        
        # DEBUG模式：输出完整的原始响应
        if DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"[DEBUG] JSON解析失败 - 完整响应内容:")
            print(f"{'='*60}")
            print(f"原始响应长度: {len(original_reply)} 字符")
            print(f"清理后长度: {len(reply)} 字符")
            print(f"{'-'*60}")
            print(original_reply)
            print(f"{'='*60}\n")
        return []
