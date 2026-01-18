# 6Analyst Utils Module

from .token_counter import get_tokenizer, count_tokens, count_message_tokens
from .html_extractor import HTMLTextExtractor, simplify_html
from .error_logger import (
    start_batch_context, record_batch_step, record_prompt_input,
    record_raw_response, end_batch_context, log_parse_error,
    log_api_error, log_batch_exception, clear_error_log, get_error_log_path
)
from .common import (
    ValidationError,
    validate_record, validate_json_line, validate_ip_record,
    normalize_result_fields, merge_analysis_results,
    safe_read_jsonl, safe_write_jsonl, safe_append_record,
    classify_confidence, is_invalid_record
)
from .logger import (
    setup_logger, get_logger, set_log_task_id, get_log_task_id,
    set_console_log_level, get_console_log_level, LOG_LEVEL_MAP,
    log_debug, log_info, log_warning, log_error, log_exception
)

__all__ = [
    'get_tokenizer',
    'count_tokens', 
    'count_message_tokens',
    'HTMLTextExtractor',
    'simplify_html',
    # Error logger
    'start_batch_context',
    'record_batch_step',
    'record_prompt_input',
    'record_raw_response',
    'end_batch_context',
    'log_parse_error',
    'log_api_error',
    'log_batch_exception',
    'clear_error_log',
    'get_error_log_path',
    # Common utilities
    'ValidationError',
    'validate_record',
    'validate_json_line',
    'validate_ip_record',
    'normalize_result_fields',
    'merge_analysis_results',
    'safe_read_jsonl',
    'safe_write_jsonl',
    'safe_append_record',
    'classify_confidence',
    'is_invalid_record',
    # Logger utilities
    'setup_logger',
    'get_logger',
    'set_log_task_id',
    'get_log_task_id',
    'set_console_log_level',
    'get_console_log_level',
    'LOG_LEVEL_MAP',
    'log_debug',
    'log_info',
    'log_warning',
    'log_error',
    'log_exception',
]
