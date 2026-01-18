"""
Token和费用计算模块
根据API定价计算各Agent的输入输出token开销和费用
"""

import glob
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional

from .config import (
    CLEANED_DATA_PATH, MERGED_OUTPUT_PATH, CHECK_OUTPUT_PATH,
    PRODUCT_OUTPUT_PATH, MODEL_NAME, INPUT_DIR
)
from .utils.token_counter import get_tokenizer, count_tokens, count_message_tokens
from .product_analyst import ProductAnalyst
from .check_analyst import CheckAnalyst
from .data_cleaner import DataCleaner


# 模型定价表 (单位: 元/1K Tokens) - 基于api-README.md
# 注意：此表与 config.py 中的 MODEL_PRICES 保持同步
MODEL_PRICING = {
    # ===== Gemini 系列 =====
    "gemini-2.5-pro": {"input": 0.007, "output": 0.04},
    "gemini-2.5-flash": {"input": 0.0012, "output": 0.01},
    "gemini-2.5-flash-nothinking": {"input": 0.0012, "output": 0.01},
    "gemini-2.5-flash-lite": {"input": 0.0004, "output": 0.0016},
    "gemini-2.5-flash-lite-preview-06-17": {"input": 0.0004, "output": 0.0016},  # 别名
    "gemini-2.5-flash-image-preview": {"input": 0.0015, "output": 0.15},
    "gemini-3-pro-preview": {"input": 0.008, "output": 0.048},
    "gemini-3-flash-preview": {"input": 0.002, "output": 0.012},
    "gemini-3-flash-preview-nothinking": {"input": 0.002, "output": 0.012},
    
    # ===== DeepSeek 系列 =====
    "deepseek-v3": {"input": 0.0012, "output": 0.0048},
    "deepseek-chat": {"input": 0.0012, "output": 0.0048},
    "deepseek-v3-2-exp": {"input": 0.0012, "output": 0.0018},
    "deepseek-v3.2": {"input": 0.0012, "output": 0.0018},
    "deepseek-v3.2-thinking": {"input": 0.0012, "output": 0.0018},
    "deepseek-v3.1-250821": {"input": 0.0024, "output": 0.0072},
    "deepseek-v3.1-think-250821": {"input": 0.0024, "output": 0.0072},
    "deepseek-r1": {"input": 0.0024, "output": 0.0096},
    "deepseek-reasoner": {"input": 0.0024, "output": 0.0096},
    "deepseek-r1-250528": {"input": 0.0024, "output": 0.0096},
    
    # ===== GPT-5.2 系列 =====
    "gpt-5.2": {"input": 0.01225, "output": 0.098},
    "gpt-5.2-2025-12-11": {"input": 0.01225, "output": 0.098},
    "gpt-5.2-chat-latest": {"input": 0.01225, "output": 0.098},
    "gpt-5.2-pro": {"input": 0.147, "output": 1.176},
    "gpt-5.2-pro-2025-12-11": {"input": 0.147, "output": 1.176},
    "gpt-5.2-ca": {"input": 0.007, "output": 0.056},
    "gpt-5.2-chat-latest-ca": {"input": 0.007, "output": 0.056},
    
    # ===== GPT-5.1 系列 =====
    "gpt-5.1": {"input": 0.00875, "output": 0.07},
    "gpt-5.1-2025-11-13": {"input": 0.00875, "output": 0.07},
    "gpt-5.1-chat-latest": {"input": 0.00875, "output": 0.07},
    "gpt-5.1-codex": {"input": 0.00875, "output": 0.07},
    "gpt-5.1-ca": {"input": 0.005, "output": 0.04},
    "gpt-5.1-chat-latest-ca": {"input": 0.005, "output": 0.04},
    
    # ===== GPT-5 系列 =====
    "gpt-5": {"input": 0.00875, "output": 0.07},
    "gpt-5-codex": {"input": 0.00875, "output": 0.07},
    "gpt-5-chat-latest": {"input": 0.00875, "output": 0.07},
    "gpt-5-search-api": {"input": 0.00875, "output": 0.07},
    "gpt-5-mini": {"input": 0.00175, "output": 0.014},
    "gpt-5-nano": {"input": 0.00035, "output": 0.0028},
    "gpt-5-pro": {"input": 0.105, "output": 0.84},
    "gpt-5-ca": {"input": 0.005, "output": 0.04},
    "gpt-5-mini-ca": {"input": 0.001, "output": 0.008},
    "gpt-5-nano-ca": {"input": 0.0002, "output": 0.0016},
    "gpt-5-chat-latest-ca": {"input": 0.005, "output": 0.04},
    
    # ===== GPT-4.1 系列 =====
    "gpt-4.1": {"input": 0.014, "output": 0.056},
    "gpt-4.1-2025-04-14": {"input": 0.014, "output": 0.056},
    "gpt-4.1-mini": {"input": 0.0028, "output": 0.0112},
    "gpt-4.1-mini-2025-04-14": {"input": 0.0028, "output": 0.0112},
    "gpt-4.1-nano": {"input": 0.0007, "output": 0.0028},
    "gpt-4.1-nano-2025-04-14": {"input": 0.0007, "output": 0.0028},
    "gpt-4.1-ca": {"input": 0.008, "output": 0.032},
    "gpt-4.1-mini-ca": {"input": 0.0016, "output": 0.0064},
    "gpt-4.1-nano-ca": {"input": 0.0004, "output": 0.003},
    
    # ===== GPT-4o 系列 =====
    "gpt-4o": {"input": 0.0175, "output": 0.07},
    "gpt-4o-2024-11-20": {"input": 0.0175, "output": 0.07},
    "gpt-4o-2024-08-06": {"input": 0.0175, "output": 0.07},
    "gpt-4o-2024-05-13": {"input": 0.035, "output": 0.105},
    "gpt-4o-mini": {"input": 0.00105, "output": 0.0042},
    "chatgpt-4o-latest": {"input": 0.035, "output": 0.105},
    "gpt-4o-search-preview": {"input": 0.0175, "output": 0.07},
    "gpt-4o-search-preview-2025-03-11": {"input": 0.0175, "output": 0.07},
    "gpt-4o-mini-search-preview": {"input": 0.00105, "output": 0.0042},
    "gpt-4o-mini-search-preview-2025-03-11": {"input": 0.00105, "output": 0.0042},
    "gpt-4o-ca": {"input": 0.01, "output": 0.04},
    "gpt-4o-mini-ca": {"input": 0.00075, "output": 0.003},
    
    # ===== GPT-4 系列 =====
    "gpt-4": {"input": 0.21, "output": 0.42},
    "gpt-4-0613": {"input": 0.21, "output": 0.42},
    "gpt-4-turbo": {"input": 0.07, "output": 0.21},
    "gpt-4-turbo-preview": {"input": 0.07, "output": 0.21},
    "gpt-4-0125-preview": {"input": 0.07, "output": 0.21},
    "gpt-4-1106-preview": {"input": 0.07, "output": 0.21},
    "gpt-4-vision-preview": {"input": 0.07, "output": 0.21},
    "gpt-4-turbo-2024-04-09": {"input": 0.07, "output": 0.21},
    "gpt-4-ca": {"input": 0.12, "output": 0.24},
    
    # ===== GPT-3.5 系列 =====
    "gpt-3.5-turbo": {"input": 0.0035, "output": 0.0105},
    "gpt-3.5-turbo-0125": {"input": 0.0035, "output": 0.0105},
    "gpt-3.5-turbo-1106": {"input": 0.007, "output": 0.014},
    "gpt-3.5-turbo-16k": {"input": 0.021, "output": 0.028},
    "gpt-3.5-turbo-instruct": {"input": 0.0105, "output": 0.014},
    
    # ===== GPT OSS 系列 =====
    "gpt-oss-20b": {"input": 0.0008, "output": 0.0032},
    "gpt-oss-120b": {"input": 0.0044, "output": 0.0176},
    
    # ===== o1/o3/o4 系列 =====
    "o1": {"input": 0.12, "output": 0.48},
    "o1-mini": {"input": 0.0088, "output": 0.0352},
    "o1-preview": {"input": 0.105, "output": 0.42},
    "o3": {"input": 0.014, "output": 0.056},
    "o3-2025-04-16": {"input": 0.014, "output": 0.056},
    "o3-mini": {"input": 0.0088, "output": 0.0352},
    "o4-mini": {"input": 0.0088, "output": 0.0352},
    "o4-mini-2025-04-16": {"input": 0.0088, "output": 0.0352},
    "o1-mini-ca": {"input": 0.012, "output": 0.048},
    "o1-preview-ca": {"input": 0.06, "output": 0.24},
    
    # ===== Claude 系列 =====
    "claude-3-5-sonnet-20240620": {"input": 0.015, "output": 0.075},
    "claude-3-5-sonnet-20241022": {"input": 0.015, "output": 0.075},
    "claude-3-5-haiku-20241022": {"input": 0.005, "output": 0.025},
    "claude-3-7-sonnet-20250219": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514-thinking": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-5-20250929": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-5-20250929-thinking": {"input": 0.015, "output": 0.075},
    "claude-opus-4-20250514": {"input": 0.075, "output": 0.375},
    "claude-opus-4-20250514-thinking": {"input": 0.075, "output": 0.375},
    "claude-opus-4-1-20250805": {"input": 0.075, "output": 0.375},
    "claude-opus-4-1-20250805-thinking": {"input": 0.075, "output": 0.375},
    "claude-opus-4-5-20251101": {"input": 0.025, "output": 0.125},
    "claude-opus-4-5-20251101-thinking": {"input": 0.025, "output": 0.125},
    "claude-haiku-4-5-20251001": {"input": 0.005, "output": 0.025},
    "claude-haiku-4-5-20251001-thinking": {"input": 0.005, "output": 0.025},
    
    # ===== Grok 系列 =====
    "grok-4": {"input": 0.012, "output": 0.06},
    "grok-4-fast": {"input": 0.0008, "output": 0.002},
    
    # ===== Qwen 系列 =====
    "qwen3-235b-a22b": {"input": 0.0014, "output": 0.0056},
    "qwen3-235b-a22b-instruct-2507": {"input": 0.0014, "output": 0.0056},
    "qwen3-coder-plus": {"input": 0.0028, "output": 0.0112},
    "qwen3-coder-480b-a35b-instruct": {"input": 0.0042, "output": 0.0168},
    
    # ===== Kimi 系列 =====
    "kimi-k2-0711-preview": {"input": 0.0028, "output": 0.0112},
    "kimi-k2-0905-preview": {"input": 0.0028, "output": 0.0112},
    "kimi-k2-thinking": {"input": 0.0028, "output": 0.0112},
    "kimi-k2-thinking-turbo": {"input": 0.0056, "output": 0.0406},
}


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """获取模型定价，如果找不到则返回默认值"""
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    # 尝试匹配前缀
    for key in MODEL_PRICING:
        if model_name.startswith(key):
            return MODEL_PRICING[key]
    # 默认使用gpt-3.5-turbo定价
    return MODEL_PRICING.get("gpt-3.5-turbo", {"input": 0.0035, "output": 0.0105})


class CostCalculator:
    """Token和费用计算器"""
    
    def __init__(self, batch_size: int = 3):
        """
        初始化计算器
        
        Args:
            batch_size: 每批次包含的数据条数，默认为3
        """
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer()
        self.model_name = MODEL_NAME
        self.pricing = get_model_pricing(MODEL_NAME)
        self._temp_files = []  # 临时文件列表，用于清理
    
    def clean_raw_input_data(self, input_path: str = None) -> Tuple[List[Dict], int]:
        """
        清洗原始输入文件数据，返回清洗后的数据
        
        Args:
            input_path: 输入路径
            
        Returns:
            (清洗后的数据列表, 原始记录数)
        """
        import gc
        import tempfile
        input_path = input_path or INPUT_DIR
        
        # 获取模块所在目录作为基准
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # 项目根目录是模块目录的父目录（6Analyst包的父目录）
        project_root = os.path.dirname(module_dir)
        
        # 使用项目内的临时目录存储临时文件
        temp_dir = os.path.join(module_dir, 'data', 'output', 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_filename = 'temp_cleaned_data.jsonl'
        temp_path = os.path.join(temp_dir, temp_filename)
        self._temp_files.append(temp_path)
        
        # 处理输入路径 - 转换为绝对路径
        if not os.path.isabs(input_path):
            input_path = os.path.join(project_root, input_path)
        input_path = os.path.normpath(input_path)
        
        print(f"[DEBUG] 输入路径: {input_path}")
        print(f"[DEBUG] 临时文件路径: {temp_path}")
        
        # 使用DataCleaner清洗数据
        cleaner = DataCleaner(input_path, temp_path)
        stats = cleaner.run(max_records=None)
        
        original_count = stats.get('total_records', 0)
        processed_count = stats.get('processed_records', 0)
        
        print(f"[DEBUG] 清洗统计: 原始={original_count}, 处理={processed_count}")
        
        # 强制垃圾回收，确保文件句柄释放
        del cleaner
        gc.collect()
        
        # 读取清洗后的数据
        records = []
        
        # 检查文件是否存在
        if not os.path.exists(temp_path):
            print(f"[WARN] 临时文件不存在: {temp_path}")
            return records, original_count
        
        file_size = os.path.getsize(temp_path)
        print(f"[DEBUG] 临时文件大小: {file_size} bytes")
        
        if file_size == 0:
            print(f"[WARN] 临时文件为空")
            return records, original_count
        
        try:
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            print(f"[DEBUG] 成功读取 {len(records)} 条记录")
        except Exception as e:
            print(f"[WARN] 读取临时文件失败: {type(e).__name__}: {e}")
        
        return records, original_count
    def cleanup_temp_files(self):
        """清理临时文件 - 保留临时文件，每次运行时覆盖"""
        # 不再删除临时文件，保留以便调试
        # 每次运行时会自动覆盖
        self._temp_files = []
    
    def load_raw_input_data(self, input_path: str = None) -> List[Dict]:
        """加载原始输入文件数据（未清洗）"""
        input_path = input_path or INPUT_DIR
        records = []
        
        if os.path.isdir(input_path):
            json_files = glob.glob(os.path.join(input_path, "*.json"))
            jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
            input_files = sorted(json_files + jsonl_files)
        elif os.path.isfile(input_path):
            input_files = [input_path]
        else:
            return records
        
        for filepath in input_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return records
        
    def load_cleaned_data(self) -> List[Dict]:
        """加载清洗后的数据"""
        records = []
        if not os.path.exists(CLEANED_DATA_PATH):
            return records
        with open(CLEANED_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
    
    def load_merged_data(self) -> List[Dict]:
        """加载合并后的分析结果（check agent的输入）"""
        records = []
        if not os.path.exists(MERGED_OUTPUT_PATH):
            return records
        with open(MERGED_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
    
    def load_output_data(self, path: str) -> List[Dict]:
        """加载输出数据"""
        records = []
        if not os.path.exists(path):
            return records
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def calc_product_input_tokens(self, records: List[Dict]) -> Tuple[int, float]:
        """
        计算产品分析Agent的输入token
        
        Returns:
            (总token数, 平均每条数据token数)
        """
        if not records:
            return 0, 0.0
        
        # 构建批次并计算token
        total_tokens = 0
        total_records = len(records)
        
        # 将记录转换为analyst需要的格式
        formatted_records = []
        for r in records:
            ip = next(iter(r.keys()))
            formatted_records.append({'ip': ip, 'raw': json.dumps(r, ensure_ascii=False)})
        
        # 按批次计算
        for i in range(0, len(formatted_records), self.batch_size):
            batch = formatted_records[i:i+self.batch_size]
            input_text = "\n".join([r["raw"] for r in batch])
            messages = [
                {"role": "system", "content": ProductAnalyst.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze these {len(batch)} records:\n{input_text}"},
            ]
            total_tokens += count_message_tokens(messages, self.tokenizer)
        
        avg_per_record = total_tokens / total_records if total_records > 0 else 0
        return total_tokens, avg_per_record
    
    def calc_usage_input_tokens(self, records: List[Dict]) -> Tuple[int, float]:
        """
        计算用途分析Agent的输入token
        
        Returns:
            (总token数, 平均每条数据token数)
        """
        if not records:
            return 0, 0.0
        
        total_tokens = 0
        total_records = len(records)
        
        formatted_records = []
        for r in records:
            ip = next(iter(r.keys()))
            formatted_records.append({'ip': ip, 'raw': json.dumps(r, ensure_ascii=False)})
        
        # 注意：此方法已废弃，usage_analyst已移除
        # 保留此方法仅为兼容性，返回0
        return 0, 0.0

    def calc_check_input_tokens(self, merged_records: List[Dict]) -> Tuple[int, float]:
        """
        计算校验Agent的输入token
        
        Args:
            merged_records: 合并后的分析结果
            
        Returns:
            (总token数, 平均每条数据token数)
        """
        if not merged_records:
            return 0, 0.0
        
        total_tokens = 0
        total_records = len(merged_records)
        
        formatted_records = []
        for r in merged_records:
            formatted_records.append({
                'ip': r.get('ip', 'unknown'),
                'raw': json.dumps(r, ensure_ascii=False)
            })
        
        for i in range(0, len(formatted_records), self.batch_size):
            batch = formatted_records[i:i+self.batch_size]
            input_text = "\n".join([r["raw"] for r in batch])
            messages = [
                {"role": "system", "content": CheckAnalyst.SYSTEM_PROMPT},
                {"role": "user", "content": f"Validate and correct these {len(batch)} analysis results:\n{input_text}"},
            ]
            total_tokens += count_message_tokens(messages, self.tokenizer)
        
        avg_per_record = total_tokens / total_records if total_records > 0 else 0
        return total_tokens, avg_per_record
    
    def calc_output_tokens(self, output_records: List[Dict]) -> Tuple[int, float]:
        """
        计算输出token
        
        Args:
            output_records: 输出记录列表
            
        Returns:
            (总token数, 平均每条数据token数)
        """
        if not output_records:
            return 0, 0.0
        
        total_tokens = 0
        for r in output_records:
            text = json.dumps(r, ensure_ascii=False)
            total_tokens += count_tokens(text, self.tokenizer)
        
        avg_per_record = total_tokens / len(output_records) if output_records else 0
        return total_tokens, avg_per_record
    
    def calc_cost(self, input_tokens: int, output_tokens: int) -> Tuple[float, float, float]:
        """
        计算费用
        
        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数
            
        Returns:
            (输入费用, 输出费用, 总费用) 单位：元
        """
        input_cost = (input_tokens / 1000) * self.pricing["input"]
        output_cost = (output_tokens / 1000) * self.pricing["output"]
        return input_cost, output_cost, input_cost + output_cost
    
    def calculate_all(self, datanum: int = None, file_cost: bool = False, input_path: str = None) -> Dict:
        """
        计算所有Agent的token和费用
        
        Args:
            datanum: 要估算的数据量，None表示使用实际数据量
            file_cost: 是否基于原始输入文件计算（产品和用途Agent的输入）
            input_path: 输入文件路径（file_cost=True时使用）
            
        Returns:
            包含所有统计信息的字典
        """
        # 加载数据
        cleaned_data = self.load_cleaned_data()
        merged_data = self.load_merged_data()
        product_output = self.load_output_data(PRODUCT_OUTPUT_PATH)
        check_output = self.load_output_data(CHECK_OUTPUT_PATH)
        
        # 如果使用file_cost模式，清洗原始输入文件
        file_cost_cleaned_data = []
        raw_input_count = 0
        if file_cost:
            print("正在清洗原始输入文件...")
            file_cost_cleaned_data, raw_input_count = self.clean_raw_input_data(input_path)
            print(f"清洗完成: 原始 {raw_input_count} 条 -> 清洗后 {len(file_cost_cleaned_data)} 条")
        
        # 确定实际记录数和目标记录数
        if file_cost and file_cost_cleaned_data:
            actual_records = len(file_cost_cleaned_data)
        else:
            actual_records = len(cleaned_data)
        
        is_estimation = datanum is not None and datanum != actual_records
        target_records = datanum if datanum is not None else actual_records
        
        # 计算各Agent的输入token
        if file_cost and file_cost_cleaned_data:
            # 基于清洗后的原始输入文件计算产品Agent的输入token
            product_input_tokens, product_input_avg = self.calc_product_input_tokens(file_cost_cleaned_data)
        else:
            # 基于已有清洗数据计算
            product_input_tokens, product_input_avg = self.calc_product_input_tokens(cleaned_data)
        
        # check agent的输入基于merged数据计算平均值
        check_input_tokens, check_input_avg = self.calc_check_input_tokens(merged_data)
        
        # 计算各Agent的输出token（基于实际输出数据的平均值）
        product_output_tokens, product_output_avg = self.calc_output_tokens(product_output)
        check_output_tokens, check_output_avg = self.calc_output_tokens(check_output)
        
        # 如果是file_cost模式或估算模式，需要根据目标数据量估算token
        need_estimate = file_cost or is_estimation
        
        if need_estimate:
            # 计算系统提示词的token开销（每批次固定开销）
            product_sys_tokens = count_message_tokens([
                {"role": "system", "content": ProductAnalyst.SYSTEM_PROMPT}
            ], self.tokenizer)
            check_sys_tokens = count_message_tokens([
                {"role": "system", "content": CheckAnalyst.SYSTEM_PROMPT}
            ], self.tokenizer)
            
            # 如果是估算模式（datanum指定），需要重新计算输入token
            if is_estimation:
                # 计算每条数据的平均token（不含系统提示词）
                num_batches_actual = (actual_records + self.batch_size - 1) // self.batch_size
                product_data_avg = (product_input_avg * actual_records - product_sys_tokens * num_batches_actual) / actual_records if actual_records > 0 else 0
                
                # 估算目标数据量的输入token
                num_batches_target = (target_records + self.batch_size - 1) // self.batch_size
                product_input_tokens = int(product_data_avg * target_records + product_sys_tokens * num_batches_target)
            
            # check agent输入token估算（基于merged数据的平均值）
            merged_records = len(merged_data)
            if merged_records > 0 and check_input_avg > 0:
                num_batches_merged = (merged_records + self.batch_size - 1) // self.batch_size
                check_data_avg = (check_input_avg * merged_records - check_sys_tokens * num_batches_merged) / merged_records
            else:
                # 如果没有merged数据，使用默认估算值
                check_data_avg = 450  # 默认每条check输入约450 tokens
            
            num_batches_target = (target_records + self.batch_size - 1) // self.batch_size
            check_input_tokens = int(check_data_avg * target_records + check_sys_tokens * num_batches_target)
            
            # 估算输出token：根据已有输出数据的平均值 * 目标数据量
            # 如果没有已有输出数据，使用默认估算值
            if product_output_avg == 0:
                product_output_avg = 150  # 默认每条产品分析输出约150 tokens
            if check_output_avg == 0:
                check_output_avg = 120  # 默认每条校验输出约120 tokens
            
            product_output_tokens = int(product_output_avg * target_records)
            check_output_tokens = int(check_output_avg * target_records)
        
        # 计算费用
        product_input_cost, product_output_cost, product_total_cost = self.calc_cost(
            product_input_tokens, product_output_tokens)
        check_input_cost, check_output_cost, check_total_cost = self.calc_cost(
            check_input_tokens, check_output_tokens)
        
        # 汇总
        total_input_tokens = product_input_tokens + check_input_tokens
        total_output_tokens = product_output_tokens + check_output_tokens
        total_input_cost = product_input_cost + check_input_cost
        total_output_cost = product_output_cost + check_output_cost
        total_cost = total_input_cost + total_output_cost

        return {
            "model": self.model_name,
            "pricing": self.pricing,
            "batch_size": self.batch_size,
            "is_estimation": is_estimation,
            "file_cost": file_cost,
            "target_records": target_records,
            "actual_records": actual_records,
            "raw_input_records": raw_input_count if file_cost else 0,
            "file_cost_cleaned_records": len(file_cost_cleaned_data) if file_cost else 0,
            "cleaned_records": len(cleaned_data),
            "merged_records": len(merged_data),
            
            "product_agent": {
                "input_tokens": product_input_tokens,
                "input_avg_per_record": product_input_avg,
                "output_tokens": product_output_tokens,
                "output_avg_per_record": product_output_avg,
                "input_cost": product_input_cost,
                "output_cost": product_output_cost,
                "total_cost": product_total_cost,
            },
            "check_agent": {
                "input_tokens": check_input_tokens,
                "input_avg_per_record": check_input_avg,
                "output_tokens": check_output_tokens,
                "output_avg_per_record": check_output_avg,
                "input_cost": check_input_cost,
                "output_cost": check_output_cost,
                "total_cost": check_total_cost,
            },
            "summary": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "avg_input_per_record": total_input_tokens / target_records if target_records > 0 else 0,
                "avg_output_per_record": total_output_tokens / target_records if target_records > 0 else 0,
                "avg_total_per_record": (total_input_tokens + total_output_tokens) / target_records if target_records > 0 else 0,
                "total_input_cost": total_input_cost,
                "total_output_cost": total_output_cost,
                "total_cost": total_cost,
                "cost_per_record": total_cost / target_records if target_records > 0 else 0,
            }
        }


def print_cost_report(stats: Dict) -> None:
    """打印费用报告"""
    print("\n" + "=" * 70)
    title_parts = ["Token 和费用"]
    if stats.get('file_cost'):
        title_parts.append("(基于原始文件)")
    if stats.get('is_estimation'):
        title_parts.append(f"估算报告 (目标: {stats['target_records']:,} 条)")
    else:
        title_parts.append("计算报告")
    print(" ".join(title_parts))
    print("=" * 70)
    
    print(f"\n模型: {stats['model']}")
    print(f"定价: 输入 {stats['pricing']['input']} 元/1K tokens, 输出 {stats['pricing']['output']} 元/1K tokens")
    print(f"批次大小: {stats['batch_size']} 条/批")
    
    if stats.get('file_cost'):
        print(f"数据来源: 原始 {stats['raw_input_records']} 条 -> 清洗后 {stats['file_cost_cleaned_records']} 条")
        print(f"参考数据: 合并数据 {stats['merged_records']} 条 (用于check agent和输出估算)")
    elif stats.get('is_estimation'):
        print(f"样本数据: 清洗 {stats['cleaned_records']} 条, 合并 {stats['merged_records']} 条")
        print(f"估算目标: {stats['target_records']:,} 条")
    else:
        print(f"数据量: 清洗数据 {stats['cleaned_records']} 条, 合并数据 {stats['merged_records']} 条")
    
    print("\n" + "-" * 70)
    print("各Agent详情")
    print("-" * 70)
    
    for agent_name, agent_key in [("产品分析Agent", "product_agent"), 
                                   ("校验Agent", "check_agent")]:
        agent = stats[agent_key]
        print(f"\n{agent_name}:")
        print(f"  输入: {agent['input_tokens']:>12,} tokens (平均 {agent['input_avg_per_record']:.1f} tokens/条)")
        print(f"  输出: {agent['output_tokens']:>12,} tokens (平均 {agent['output_avg_per_record']:.1f} tokens/条)")
        print(f"  费用: 输入 {agent['input_cost']:.4f} 元, 输出 {agent['output_cost']:.4f} 元, 合计 {agent['total_cost']:.4f} 元")
    
    print("\n" + "-" * 70)
    print("汇总统计")
    print("-" * 70)
    
    summary = stats["summary"]
    print(f"\n总Token数:")
    print(f"  输入: {summary['total_input_tokens']:>14,} tokens (平均 {summary['avg_input_per_record']:.1f} tokens/条)")
    print(f"  输出: {summary['total_output_tokens']:>14,} tokens (平均 {summary['avg_output_per_record']:.1f} tokens/条)")
    print(f"  合计: {summary['total_tokens']:>14,} tokens (平均 {summary['avg_total_per_record']:.1f} tokens/条)")
    
    print(f"\n总费用:")
    print(f"  输入费用: {summary['total_input_cost']:>10.4f} 元")
    print(f"  输出费用: {summary['total_output_cost']:>10.4f} 元")
    print(f"  总费用:   {summary['total_cost']:>10.4f} 元")
    print(f"  单条成本: {summary['cost_per_record']:.6f} 元/条")
    
    if stats.get('is_estimation'):
        print(f"\n注: 以上为基于 {stats['actual_records']} 条样本数据的估算值")
    if stats.get('file_cost'):
        print(f"\n注: 产品/用途Agent输入基于原始文件, check agent输入和所有输出基于已有中间结果估算")
    
    print("\n" + "=" * 70)


def run_cost_calculation(batch_size: int = 3, datanum: int = None, file_cost: bool = False, input_path: str = None) -> Dict:
    """
    运行费用计算
    
    Args:
        batch_size: 每批次包含的数据条数
        datanum: 要估算的数据量，None表示使用实际数据量
        file_cost: 是否基于原始输入文件计算
        input_path: 输入文件路径
        
    Returns:
        统计结果字典
    """
    calculator = CostCalculator(batch_size=batch_size)
    try:
        stats = calculator.calculate_all(datanum=datanum, file_cost=file_cost, input_path=input_path)
        print_cost_report(stats)
        
        # 自动生成cost_report.md
        try:
            from .tools.generate_cost_report import generate_report
            print("\n正在生成开销评估报告...")
            generate_report(input_path or INPUT_DIR, batch_size=batch_size)
        except Exception as e:
            print(f"[WARN] 生成报告失败: {e}")
        
        return stats
    finally:
        # 清理临时文件
        calculator.cleanup_temp_files()


class PromptLengthCalculator:
    """
    提示词长度计算器
    使用缩放法迭代计算，在给定预算下找到合适的提示词长度
    """
    
    def __init__(self, batch_size: int = 3):
        """
        初始化计算器
        
        Args:
            batch_size: 每批次包含的数据条数
        """
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer()
        self.model_name = MODEL_NAME
        self.pricing = get_model_pricing(MODEL_NAME)
        
        # 获取当前各Agent的提示词（字符数和token数）
        self.product_prompt_chars = len(ProductAnalyst.SYSTEM_PROMPT)
        self.check_prompt_chars = len(CheckAnalyst.SYSTEM_PROMPT)
        
        self.product_prompt_tokens = count_message_tokens([
            {"role": "system", "content": ProductAnalyst.SYSTEM_PROMPT}
        ], self.tokenizer)
        self.check_prompt_tokens = count_message_tokens([
            {"role": "system", "content": CheckAnalyst.SYSTEM_PROMPT}
        ], self.tokenizer)
        
        # 当前总提示词长度
        self.current_prompt_chars = (
            self.product_prompt_chars + 
            self.check_prompt_chars
        )
        self.current_prompt_tokens = (
            self.product_prompt_tokens + 
            self.check_prompt_tokens
        )
        
        # 字符/token比例（用于从token数估算字符数）
        self.chars_per_token = self.current_prompt_chars / self.current_prompt_tokens if self.current_prompt_tokens > 0 else 3.6
    
    def estimate_cost_with_prompt_length(self, 
                                         data_count: int, 
                                         prompt_tokens: int,
                                         avg_data_tokens: float,
                                         avg_output_tokens: float) -> float:
        """
        估算给定提示词长度下的总开销
        
        Args:
            data_count: 数据条数
            prompt_tokens: 总提示词token数（三个Agent的系统提示词之和）
            avg_data_tokens: 每条数据的平均token数（不含系统提示词）
            avg_output_tokens: 每条数据的平均输出token数
            
        Returns:
            估算的总费用（元）
        """
        # 计算批次数
        num_batches = (data_count + self.batch_size - 1) // self.batch_size
        
        # 按比例分配提示词长度到各Agent
        total_current = self.current_prompt_tokens
        product_ratio = self.product_prompt_tokens / total_current
        check_ratio = self.check_prompt_tokens / total_current
        
        product_prompt = prompt_tokens * product_ratio
        check_prompt = prompt_tokens * check_ratio
        
        # 计算输入token
        # Product Agent: 系统提示词 * 批次数 + 数据token * 数据条数
        product_input = product_prompt * num_batches + avg_data_tokens * data_count
        
        # Check Agent: 系统提示词 * 批次数 + 合并结果token * 数据条数
        check_data_tokens_per_record = 375  # 合并后每条数据的平均token数
        check_input = check_prompt * num_batches + check_data_tokens_per_record * data_count
        
        total_input_tokens = product_input + check_input
        
        # 总输出token（两个Agent的输出）
        total_output_tokens = avg_output_tokens * data_count * 2
        
        # 计算费用
        input_cost = (total_input_tokens / 1000) * self.pricing["input"]
        output_cost = (total_output_tokens / 1000) * self.pricing["output"]
        
        return input_cost + output_cost
    
    def calculate_prompt_length_for_budget(self,
                                           budget: float,
                                           data_count: int,
                                           precision: int = 3,
                                           avg_data_tokens: float = None,
                                           avg_output_tokens: float = None) -> Dict:
        """
        使用缩放法计算给定预算下的最大提示词长度
        
        算法：
        1. 使用当前提示词长度计算估计总开销
        2. 若不满足精度要求，将提示词长度乘以 (预算/当前开销) 的倍率
        3. 重复直到精度小于10^-precision且估计开销小于预算
        
        Args:
            budget: 预算（元）
            data_count: 数据条数
            precision: 精度（10^-precision元）
            avg_data_tokens: 每条数据的平均token数（可选，默认使用估算值）
            avg_output_tokens: 每条数据的平均输出token数（可选，默认使用估算值）
            
        Returns:
            包含计算结果的字典
        """
        # 基于实际测量的估算值（来自--calculate-cost的结果）
        # Product: 输入547.9 tokens/条（含系统提示词分摊），输出179.6 tokens/条
        # Usage: 输入494.3 tokens/条（含系统提示词分摊），输出200.7 tokens/条
        # Check: 输入564.1 tokens/条（含系统提示词分摊），输出83.2 tokens/条
        
        # 计算批次数
        num_batches = (data_count + self.batch_size - 1) // self.batch_size
        
        # 计算每条数据的纯数据token（不含系统提示词）
        # 系统提示词每批次发送一次，所以要从平均值中扣除
        product_sys_per_record = self.product_prompt_tokens / self.batch_size
        usage_sys_per_record = self.usage_prompt_tokens / self.batch_size
        check_sys_per_record = self.check_prompt_tokens / self.batch_size
        
        # 默认估算值（基于实际测量）
        if avg_data_tokens is None:
            # 纯数据token = 总输入平均 - 系统提示词分摊
            # Product: 547.9 - 660/3 ≈ 328
            # Usage: 494.3 - 496/3 ≈ 329
            avg_data_tokens = 330  # 每条数据平均330 tokens（不含系统提示词）
        
        if avg_output_tokens is None:
            # 三个Agent的平均输出: (179.6 + 200.7 + 83.2) / 3 ≈ 154.5
            avg_output_tokens = 155  # 每条输出平均155 tokens
        
        # Check agent的输入数据token（合并后的结果）
        # Check输入564.1 - 570/3 ≈ 374
        check_data_tokens_per_record = 375
        
        precision_threshold = 10 ** (-precision)
        
        # 初始提示词长度
        prompt_tokens = float(self.current_prompt_tokens)
        iteration = 0
        max_iterations = 100  # 防止无限循环
        
        history = []  # 记录迭代历史
        
        # 计算固定开销（数据token和输出token的费用，与提示词长度无关）
        # 数据输入token（product + usage，每条数据处理两次）
        data_input_tokens = avg_data_tokens * data_count * 2
        # check agent的数据输入
        check_data_input_tokens = check_data_tokens_per_record * data_count
        # 总输出token（三个Agent的输出）
        total_output_tokens = avg_output_tokens * data_count * 3
        
        fixed_input_cost = ((data_input_tokens + check_data_input_tokens) / 1000) * self.pricing["input"]
        fixed_output_cost = (total_output_tokens / 1000) * self.pricing["output"]
        fixed_cost = fixed_input_cost + fixed_output_cost
        
        # 可用于提示词的预算
        prompt_budget = budget - fixed_cost
        
        if prompt_budget <= 0:
            # 预算不足以覆盖固定开销
            return {
                'budget': budget,
                'precision': precision,
                'precision_threshold': precision_threshold,
                'data_count': data_count,
                'iterations': 0,
                'converged': False,
                'error': '预算不足以覆盖数据处理的固定开销',
                'fixed_cost': fixed_cost,
                'current_prompt_tokens': self.current_prompt_tokens,
                'recommended_prompt_tokens': 0,
                'estimated_cost': fixed_cost,
                'cost_diff': abs(fixed_cost - budget),
                'model': self.model_name,
                'pricing': self.pricing,
                'batch_size': self.batch_size,
            }
        
        while iteration < max_iterations:
            iteration += 1
            
            # 计算当前提示词长度下的估计开销
            estimated_cost = self.estimate_cost_with_prompt_length(
                data_count, prompt_tokens, avg_data_tokens, avg_output_tokens
            )
            
            # 计算与预算的差值
            diff = abs(estimated_cost - budget)
            
            # 记录历史
            history.append({
                'iteration': iteration,
                'prompt_tokens': int(prompt_tokens),
                'estimated_cost': estimated_cost,
                'diff': diff
            })
            
            # 检查是否满足精度要求且开销小于等于预算
            if diff < precision_threshold and estimated_cost <= budget:
                break
            
            # 计算提示词部分的开销
            prompt_cost = estimated_cost - fixed_cost
            
            if prompt_cost > 0:
                # 计算新的提示词长度
                # 提示词开销 = (prompt_tokens * num_batches / 1000) * input_price
                # 目标: prompt_cost_new = prompt_budget
                # prompt_tokens_new = prompt_tokens * (prompt_budget / prompt_cost)
                scale = prompt_budget / prompt_cost
                new_prompt_tokens = prompt_tokens * scale
                
                # 限制变化幅度，避免震荡
                if new_prompt_tokens > prompt_tokens * 2:
                    new_prompt_tokens = prompt_tokens * 2
                elif new_prompt_tokens < prompt_tokens * 0.5:
                    new_prompt_tokens = prompt_tokens * 0.5
                
                prompt_tokens = new_prompt_tokens
            else:
                break
        
        # 计算最终结果
        final_prompt_tokens = int(prompt_tokens)
        final_cost = self.estimate_cost_with_prompt_length(
            data_count, final_prompt_tokens, avg_data_tokens, avg_output_tokens
        )
        
        # 计算各Agent的推荐长度（token和字符）
        total_current = self.current_prompt_tokens
        product_ratio = self.product_prompt_tokens / total_current
        usage_ratio = self.usage_prompt_tokens / total_current
        check_ratio = self.check_prompt_tokens / total_current
        
        # 推荐的字符长度（基于缩放比例）
        prompt_scale = final_prompt_tokens / self.current_prompt_tokens
        recommended_prompt_chars = int(self.current_prompt_chars * prompt_scale)
        
        # 计算每次提交的平均字符长度
        # 每条数据的平均字符数（基于token数估算，使用chars_per_token比例）
        avg_data_chars = avg_data_tokens * self.chars_per_token
        check_data_chars = check_data_tokens_per_record * self.chars_per_token
        
        # 当前每次提交的平均字符长度 = 系统提示词 + 数据(batch_size条)
        current_avg_per_request = {
            'product_analyst': self.product_prompt_chars + int(avg_data_chars * self.batch_size),
            'check_analyst': self.check_prompt_chars + int(check_data_chars * self.batch_size),
        }
        
        # 推荐每次提交的平均字符长度
        recommended_avg_per_request = {
            'product_analyst': int(self.product_prompt_chars * prompt_scale) + int(avg_data_chars * self.batch_size),
            'check_analyst': int(self.check_prompt_chars * prompt_scale) + int(check_data_chars * self.batch_size),
        }
        
        return {
            'budget': budget,
            'precision': precision,
            'precision_threshold': precision_threshold,
            'data_count': data_count,
            'iterations': iteration,
            'converged': iteration < max_iterations,
            'current_prompt_tokens': self.current_prompt_tokens,
            'current_prompt_chars': self.current_prompt_chars,
            'recommended_prompt_tokens': final_prompt_tokens,
            'recommended_prompt_chars': recommended_prompt_chars,
            'prompt_scale': prompt_scale,
            'chars_per_token': self.chars_per_token,
            'avg_data_chars': avg_data_chars,
            'check_data_chars': check_data_chars,
            'fixed_cost': fixed_cost,
            'prompt_budget': prompt_budget,
            'estimated_cost': final_cost,
            'cost_diff': abs(final_cost - budget),
            'model': self.model_name,
            'pricing': self.pricing,
            'batch_size': self.batch_size,
            'prompt_breakdown': {
                'product_analyst': {
                    'tokens': int(final_prompt_tokens * product_ratio),
                    'chars': int(self.product_prompt_chars * prompt_scale),
                },
                'check_analyst': {
                    'tokens': int(final_prompt_tokens * check_ratio),
                    'chars': int(self.check_prompt_chars * prompt_scale),
                },
            },
            'current_breakdown': {
                'product_analyst': {
                    'tokens': self.product_prompt_tokens,
                    'chars': self.product_prompt_chars,
                },
                'check_analyst': {
                    'tokens': self.check_prompt_tokens,
                    'chars': self.check_prompt_chars,
                },
            },
            'current_avg_per_request': current_avg_per_request,
            'recommended_avg_per_request': recommended_avg_per_request,
            'history': history[-10:]  # 只保留最后10次迭代
        }


def print_prompt_length_report(result: Dict) -> None:
    """打印提示词长度计算报告"""
    print("\n" + "=" * 70)
    print("提示词长度计算报告（缩放法）")
    print("=" * 70)
    
    # 检查是否有错误
    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        print(f"  固定开销: {result['fixed_cost']:.4f} 元")
        print(f"  预算: {result['budget']:.4f} 元")
        print("\n" + "=" * 70)
        return
    
    print(f"\n输入参数:")
    print(f"  预算: {result['budget']:.2f} 元")
    print(f"  精度: 10^-{result['precision']} = {result['precision_threshold']:.6f} 元")
    print(f"  数据量: {result['data_count']:,} 条")
    print(f"  批次大小: {result['batch_size']} 条/批")
    
    print(f"\n模型信息:")
    print(f"  模型: {result['model']}")
    print(f"  定价: 输入 {result['pricing']['input']} 元/1K tokens, 输出 {result['pricing']['output']} 元/1K tokens")
    
    print(f"\n费用分解:")
    print(f"  固定开销（数据+输出）: {result['fixed_cost']:.4f} 元")
    print(f"  提示词预算: {result['prompt_budget']:.4f} 元")
    
    print(f"\n计算结果:")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  是否收敛: {'是' if result['converged'] else '否'}")
    print(f"  缩放比例: {result['prompt_scale']:.4f}")
    
    print(f"\n提示词长度:")
    print(f"  当前: {result['current_prompt_chars']:,} 字符 / {result['current_prompt_tokens']:,} tokens")
    print(f"  推荐: {result['recommended_prompt_chars']:,} 字符 / {result['recommended_prompt_tokens']:,} tokens")
    change_chars = result['recommended_prompt_chars'] - result['current_prompt_chars']
    change_pct = (result['prompt_scale'] - 1) * 100
    print(f"  变化: {change_chars:+,} 字符 ({change_pct:+.1f}%)")
    
    print(f"\n各Agent当前/推荐长度:")
    current = result.get('current_breakdown', {})
    recommended = result['prompt_breakdown']
    for agent, key in [("ProductAnalyst", "product_analyst"), 
                       ("CheckAnalyst", "check_analyst")]:
        curr = current.get(key, {})
        rec = recommended.get(key, {})
        print(f"  {agent}: {curr.get('chars', 0):,}字符 -> {rec.get('chars', 0):,}字符")
    
    print(f"\n费用估算:")
    print(f"  预算: {result['budget']:.4f} 元")
    print(f"  估算开销: {result['estimated_cost']:.4f} 元")
    print(f"  差值: {result['cost_diff']:.6f} 元")
    
    if result.get('history'):
        print(f"\n迭代历史（最后{len(result['history'])}次）:")
        for h in result['history']:
            print(f"  第{h['iteration']:2d}次: tokens={h['prompt_tokens']:,}, cost={h['estimated_cost']:.4f}, diff={h['diff']:.6f}")
    
    print("\n" + "=" * 70)


def run_prompt_length_calculation(budget: float, 
                                   precision: int = 3, 
                                   batch_size: int = 3,
                                   input_path: str = None) -> Dict:
    """
    运行提示词长度计算
    
    Args:
        budget: 预算（元）
        precision: 精度（10^-precision元）
        batch_size: 每批次包含的数据条数
        input_path: 输入文件路径（用于获取数据量）
        
    Returns:
        计算结果字典
    """
    # 获取数据量
    calculator = CostCalculator(batch_size=batch_size)
    try:
        cleaned_data, raw_count = calculator.clean_raw_input_data(input_path)
        data_count = len(cleaned_data) if cleaned_data else raw_count
        
        if data_count == 0:
            print("[ERROR] 无法获取数据量，请检查输入路径")
            return {}
        
        print(f"[INFO] 数据量: {data_count} 条")
        
        # 计算提示词长度
        prompt_calculator = PromptLengthCalculator(batch_size=batch_size)
        result = prompt_calculator.calculate_prompt_length_for_budget(
            budget=budget,
            data_count=data_count,
            precision=precision
        )
        
        print_prompt_length_report(result)
        return result
        
    finally:
        calculator.cleanup_temp_files()


def run_multi_model_prompt_calculation(budget: float,
                                        precision: int = 3,
                                        batch_size: int = 3,
                                        input_path: str = None,
                                        generate_report: bool = True) -> Dict:
    """
    运行多模型提示词长度计算，并生成报告
    
    Args:
        budget: 预算（元）
        precision: 精度（10^-precision元）
        batch_size: 每批次包含的数据条数
        input_path: 输入文件路径
        generate_report: 是否生成Markdown报告
        
    Returns:
        包含所有模型计算结果的字典
    """
    # 获取数据量
    calculator = CostCalculator(batch_size=batch_size)
    try:
        cleaned_data, raw_count = calculator.clean_raw_input_data(input_path)
        data_count = len(cleaned_data) if cleaned_data else raw_count
        
        if data_count == 0:
            print("[ERROR] 无法获取数据量，请检查输入路径")
            return {}
        
        print(f"[INFO] 数据量: {data_count} 条")
        print(f"[INFO] 预算: {budget} 元")
        print(f"[INFO] 精度: 10^-{precision} 元")
        
        # 要比较的模型列表
        models_to_compare = [
            ("deepseek-v3.2", {"input": 0.0012, "output": 0.0018}, "⭐⭐⭐⭐⭐ 最佳性价比"),
            ("deepseek-v3", {"input": 0.0012, "output": 0.0048}, "⭐⭐⭐⭐⭐ 高性价比"),
            ("gemini-2.5-flash-lite", {"input": 0.0004, "output": 0.0016}, "⭐⭐⭐⭐ 最便宜"),
            ("gemini-2.5-flash", {"input": 0.0006, "output": 0.014}, "⭐⭐⭐⭐ 性价比高"),
            ("gpt-4.1-nano", {"input": 0.0007, "output": 0.0028}, "⭐⭐⭐⭐ 官方稳定"),
            ("gpt-4o-mini-ca", {"input": 0.00075, "output": 0.003}, "⭐⭐⭐⭐ 第三方便宜"),
            ("gpt-4o-mini", {"input": 0.00105, "output": 0.0042}, "⭐⭐⭐⭐ 官方稳定"),
            ("qwen3-235b-a22b", {"input": 0.0014, "output": 0.0056}, "⭐⭐⭐⭐ 国产高性能"),
            ("gpt-3.5-turbo", {"input": 0.0035, "output": 0.0105}, "⭐⭐⭐ 基准模型"),
            ("gpt-5-mini", {"input": 0.00175, "output": 0.014}, "⭐⭐⭐ 新一代mini"),
            ("gpt-4.1-mini", {"input": 0.0028, "output": 0.0112}, "⭐⭐⭐ 官方中端"),
            ("kimi-k2-0711-preview", {"input": 0.0028, "output": 0.0112}, "⭐⭐⭐ 国产"),
            ("gpt-5", {"input": 0.00875, "output": 0.07}, "⭐⭐ 旗舰但贵"),
            ("gpt-4.1", {"input": 0.014, "output": 0.056}, "⭐⭐ 高端"),
            ("claude-3-5-haiku", {"input": 0.005, "output": 0.025}, "⭐⭐ Claude入门"),
            ("grok-4-fast", {"input": 0.0008, "output": 0.002}, "⭐⭐⭐⭐ 极速便宜"),
        ]
        
        results = {
            'budget': budget,
            'precision': precision,
            'data_count': data_count,
            'batch_size': batch_size,
            'input_path': input_path,
            'models': []
        }
        
        print(f"\n正在计算 {len(models_to_compare)} 个模型的推荐提示词长度...")
        
        for model_name, pricing, rating in models_to_compare:
            # 创建计算器并设置模型定价
            prompt_calc = PromptLengthCalculator(batch_size=batch_size)
            prompt_calc.model_name = model_name
            prompt_calc.pricing = pricing
            
            # 计算该模型的推荐提示词长度
            model_result = prompt_calc.calculate_prompt_length_for_budget(
                budget=budget,
                data_count=data_count,
                precision=precision
            )
            
            model_result['model'] = model_name
            model_result['rating'] = rating
            
            # 计算性价比指标（推荐提示词长度 / 预算）
            if model_result.get('recommended_prompt_tokens', 0) > 0:
                model_result['cost_efficiency'] = model_result['recommended_prompt_tokens'] / budget
            else:
                model_result['cost_efficiency'] = 0
            
            results['models'].append(model_result)
            
            # 简要输出
            if 'error' in model_result:
                print(f"  {model_name}: 预算不足")
            else:
                print(f"  {model_name}: {model_result['recommended_prompt_tokens']:,} tokens (scale: {model_result['prompt_scale']:.2f}x)")
        
        # 按推荐提示词长度排序（从大到小，长度越大说明预算越充裕）
        results['models'].sort(key=lambda x: x.get('recommended_prompt_tokens', 0), reverse=True)
        
        # 打印汇总报告
        print_multi_model_prompt_report(results)
        
        # 生成Markdown报告
        if generate_report:
            report_path = generate_prompt_length_report(results, input_path)
            print(f"\n报告已保存到: {report_path}")
        
        return results
        
    finally:
        calculator.cleanup_temp_files()


def print_multi_model_prompt_report(results: Dict) -> None:
    """打印多模型提示词长度计算汇总报告"""
    print("\n" + "=" * 110)
    print("多模型提示词长度计算报告")
    print("=" * 110)
    
    print(f"\n输入参数:")
    print(f"  预算: {results['budget']:.2f} 元")
    print(f"  精度: 10^-{results['precision']} 元")
    print(f"  数据量: {results['data_count']:,} 条")
    print(f"  批次大小: {results['batch_size']} 条/批")
    
    # 显示当前各Agent的提示词长度
    print(f"\n当前提示词长度:")
    print(f"  ProductAnalyst: {len(ProductAnalyst.SYSTEM_PROMPT):,} 字符")
    print(f"  CheckAnalyst: {len(CheckAnalyst.SYSTEM_PROMPT):,} 字符")
    print(f"  合计: {len(ProductAnalyst.SYSTEM_PROMPT) + len(CheckAnalyst.SYSTEM_PROMPT):,} 字符")
    
    # 计算批次数
    num_batches = (results['data_count'] + results['batch_size'] - 1) // results['batch_size']
    
    print(f"\n" + "-" * 110)
    print(f"{'模型':<30} {'推荐平均长度':>14} {'缩放比例':>10} {'估算开销':>12} {'性价比(元/万字符)':>20} {'评价'}")
    print("-" * 110)
    
    for model in results['models']:
        model_name = model.get('model', 'unknown')
        if 'error' in model:
            print(f"{model_name:<30} {'预算不足':>14} {'-':>10} {model.get('fixed_cost', 0):>11.2f}元 {'-':>20} {model.get('rating', '')}")
        else:
            # 计算每次提交的平均字符长度（三个Agent的平均）
            avg_per_req = model.get('recommended_avg_per_request', {})
            if avg_per_req:
                avg_chars = (avg_per_req.get('product_analyst', 0) + 
                            avg_per_req.get('check_analyst', 0)) // 2
            else:
                avg_chars = model.get('recommended_prompt_chars', 0)
            
            scale = model.get('prompt_scale', 0)
            cost = model.get('estimated_cost', 0)
            # 性价比 = 总开销 / 总字符数（元/字符）
            # 总字符数 = 推荐平均长度 × 总批次数 × 2个Agent
            total_chars = avg_chars * num_batches * 2
            cost_per_char = cost / total_chars * 10000 if total_chars > 0 else 0  # 转换为元/万字符
            rating = model.get('rating', '')
            print(f"{model_name:<30} {avg_chars:>12,}字符 {scale:>9.2f}x {cost:>11.2f}元 {cost_per_char:>18.4f} {rating}")
    
    print("-" * 110)
    
    # 找出最佳模型
    valid_models = [m for m in results['models'] if 'error' not in m]
    if valid_models:
        # 按推荐平均长度排序
        def get_avg_chars(m):
            avg_per_req = m.get('recommended_avg_per_request', {})
            if avg_per_req:
                return (avg_per_req.get('product_analyst', 0) + 
                       avg_per_req.get('check_analyst', 0)) // 2
            return m.get('recommended_prompt_chars', 0)
        
        best_chars = max(valid_models, key=get_avg_chars)
        best_avg = get_avg_chars(best_chars)
        
        print(f"\n推荐:")
        print(f"  最大平均提交长度: {best_chars['model']} ({best_avg:,} 字符/次)")
    
    print("\n" + "=" * 110)


def generate_prompt_length_report(results: Dict, input_path: str = None) -> str:
    """
    生成提示词长度计算的Markdown报告
    
    Args:
        results: 多模型计算结果
        input_path: 输入文件路径
        
    Returns:
        报告文件路径
    """
    from datetime import datetime
    import os
    
    # 获取当前提示词信息
    tokenizer = get_tokenizer()
    product_prompt = ProductAnalyst.SYSTEM_PROMPT
    check_prompt = CheckAnalyst.SYSTEM_PROMPT
    
    product_tokens = count_message_tokens([{"role": "system", "content": product_prompt}], tokenizer)
    check_tokens = count_message_tokens([{"role": "system", "content": check_prompt}], tokenizer)
    total_current_tokens = product_tokens + check_tokens
    total_current_chars = len(product_prompt) + len(check_prompt)
    
    lines = []
    lines.append("# 6Analyst 提示词长度计算报告")
    lines.append("")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 一、计算参数
    lines.append("## 一、计算参数")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 预算 | **{results['budget']:.2f} 元** |")
    lines.append(f"| 精度 | 10^-{results['precision']} = {10**(-results['precision']):.6f} 元 |")
    lines.append(f"| 数据量 | {results['data_count']:,} 条 |")
    lines.append(f"| 批次大小 | {results['batch_size']} 条/批 |")
    lines.append(f"| 总批次数 | {(results['data_count'] + results['batch_size'] - 1) // results['batch_size']:,} 批 |")
    if input_path:
        lines.append(f"| 输入路径 | `{input_path}` |")
    lines.append("")
    
    # 二、当前提示词配置
    lines.append("## 二、当前提示词配置")
    lines.append("")
    lines.append("| Agent | 字符数 | Token数 |")
    lines.append("|-------|--------|---------|")
    lines.append(f"| ProductAnalyst | {len(product_prompt):,} | {product_tokens:,} |")
    lines.append(f"| CheckAnalyst | {len(check_prompt):,} | {check_tokens:,} |")
    lines.append(f"| **合计** | **{total_current_chars:,}** | **{total_current_tokens:,}** |")
    lines.append("")
    
    # 三、多模型推荐提示词长度
    lines.append("## 三、多模型推荐提示词长度")
    lines.append("")
    lines.append("*按推荐平均提交长度从大到小排序（每次提交给大模型的平均字符长度 = 系统提示词 + 批次数据）*")
    lines.append("")
    lines.append("| 模型 | 输入价格 | 输出价格 | 推荐平均长度(字符/次) | 缩放比例 | 估算开销 | 性价比(元/万字符) | 评价 |")
    lines.append("|------|----------|----------|----------------------|----------|----------|-------------------|------|")
    
    # 计算每个模型的平均提交长度并排序
    def get_avg_per_request(m):
        if 'error' in m:
            return 0
        avg_per_req = m.get('recommended_avg_per_request', {})
        if avg_per_req:
            return (avg_per_req.get('product_analyst', 0) + 
                   avg_per_req.get('check_analyst', 0)) // 2
        return m.get('recommended_prompt_chars', 0)
    
    sorted_models = sorted(results['models'], key=get_avg_per_request, reverse=True)
    
    # 计算批次数
    num_batches = (results['data_count'] + results['batch_size'] - 1) // results['batch_size']
    
    for model in sorted_models:
        model_name = model.get('model', 'unknown')
        pricing = model.get('pricing', {})
        rating = model.get('rating', '')
        
        if 'error' in model:
            lines.append(f"| {model_name} | {pricing.get('input', 0):.4f} | {pricing.get('output', 0):.4f} | [失败] 预算不足 | - | {model.get('fixed_cost', 0):.2f}元 | - | {rating} |")
        else:
            # 计算每次提交的平均字符长度
            avg_chars = get_avg_per_request(model)
            scale = model.get('prompt_scale', 0)
            cost = model.get('estimated_cost', 0)
            # 性价比 = 总开销 / 总字符数（元/字符）
            # 总字符数 = 推荐平均长度 × 总批次数 × 2个Agent
            total_chars = avg_chars * num_batches * 2
            cost_per_char = cost / total_chars * 10000 if total_chars > 0 else 0  # 转换为元/万字符
            
            # 根据缩放比例添加标记
            if scale >= 2:
                scale_mark = "🟢"  # 可以扩展
            elif scale >= 1:
                scale_mark = "🟡"  # 刚好够用
            else:
                scale_mark = "🔴"  # 需要缩减
            
            lines.append(f"| {model_name} | {pricing.get('input', 0):.4f} | {pricing.get('output', 0):.4f} | **{avg_chars:,}** | {scale_mark} {scale:.2f}x | {cost:.2f}元 | {cost_per_char:.4f} | {rating} |")
    
    lines.append("")
    
    # 四、各模型详细分解
    lines.append("## 四、各模型详细分解")
    lines.append("")
    lines.append("*推荐平均长度 = 系统提示词 + 批次数据（每次提交给大模型的平均字符长度）*")
    lines.append("")
    
    valid_models = [m for m in sorted_models if 'error' not in m]
    
    for model in valid_models[:5]:  # 只展示前5个
        model_name = model.get('model', 'unknown')
        lines.append(f"### {model_name}")
        lines.append("")
        
        current_avg = model.get('current_avg_per_request', {})
        recommended_avg = model.get('recommended_avg_per_request', {})
        breakdown = model.get('prompt_breakdown', {})
        current = model.get('current_breakdown', {})
        
        lines.append("| Agent | 当前提示词(字符) | 推荐提示词(字符) | 当前平均长度(字符/次) | 推荐平均长度(字符/次) |")
        lines.append("|-------|------------------|------------------|----------------------|----------------------|")
        
        for agent, key in [("ProductAnalyst", "product_analyst"), 
                           ("CheckAnalyst", "check_analyst")]:
            curr_data = current.get(key, {})
            rec_data = breakdown.get(key, {})
            curr_prompt = curr_data.get('chars', 0) if isinstance(curr_data, dict) else curr_data
            rec_prompt = rec_data.get('chars', 0) if isinstance(rec_data, dict) else rec_data
            curr_avg_val = current_avg.get(key, 0)
            rec_avg_val = recommended_avg.get(key, 0)
            lines.append(f"| {agent} | {curr_prompt:,} | {rec_prompt:,} | {curr_avg_val:,} | {rec_avg_val:,} |")
        
        # 计算总计/平均
        total_curr_prompt = sum(v.get('chars', 0) if isinstance(v, dict) else v for v in current.values())
        total_rec_prompt = sum(v.get('chars', 0) if isinstance(v, dict) else v for v in breakdown.values())
        avg_curr = sum(current_avg.values()) // 2 if current_avg else 0
        avg_rec = sum(recommended_avg.values()) // 2 if recommended_avg else 0
        lines.append(f"| **合计/平均** | **{total_curr_prompt:,}** | **{total_rec_prompt:,}** | **{avg_curr:,}** | **{avg_rec:,}** |")
        lines.append("")
        
        # 变化说明
        prompt_change = total_rec_prompt - total_curr_prompt
        prompt_pct = (prompt_change / total_curr_prompt * 100) if total_curr_prompt > 0 else 0
        lines.append(f"- 提示词变化: {prompt_change:+,} 字符 ({prompt_pct:+.1f}%)")
        lines.append(f"- 固定开销: {model.get('fixed_cost', 0):.4f} 元")
        lines.append(f"- 提示词预算: {model.get('prompt_budget', 0):.4f} 元")
        lines.append(f"- 估算总开销: {model.get('estimated_cost', 0):.4f} 元")
        lines.append("")
    
    # 五、推荐方案
    lines.append("## 五、推荐方案")
    lines.append("")
    
    if valid_models:
        # 最大平均提交长度
        best_avg_model = max(valid_models, key=get_avg_per_request)
        best_avg_chars = get_avg_per_request(best_avg_model)
        
        # 缩放比例最接近1的（刚好够用）
        models_with_scale = [m for m in valid_models if m.get('prompt_scale', 0) >= 0.8]
        if models_with_scale:
            best_fit = min(models_with_scale, key=lambda x: abs(x.get('prompt_scale', 1) - 1))
        else:
            best_fit = None
        
        lines.append("### 🏆 最大平均提交长度")
        lines.append("")
        lines.append(f"**{best_avg_model['model']}**")
        lines.append("")
        lines.append(f"- 推荐平均长度: **{best_avg_chars:,} 字符/次**")
        lines.append(f"- 缩放比例: {best_avg_model['prompt_scale']:.2f}x")
        lines.append(f"- 估算开销: {best_avg_model['estimated_cost']:.2f} 元")
        lines.append(f"- 评价: {best_avg_model['rating']}")
        lines.append("")
        
        if best_fit:
            best_fit_avg = get_avg_per_request(best_fit)
            lines.append("### ⚖️ 最佳匹配（缩放比例接近1）")
            lines.append("")
            lines.append(f"**{best_fit['model']}**")
            lines.append("")
            lines.append(f"- 推荐平均长度: **{best_fit_avg:,} 字符/次**")
            lines.append(f"- 缩放比例: {best_fit['prompt_scale']:.2f}x（当前提示词长度{'刚好' if 0.9 <= best_fit['prompt_scale'] <= 1.1 else '需要调整'}）")
            lines.append(f"- 估算开销: {best_fit['estimated_cost']:.2f} 元")
            lines.append(f"- 评价: {best_fit['rating']}")
            lines.append("")
    
    # 六、使用建议
    lines.append("## 六、使用建议")
    lines.append("")
    lines.append("### 缩放比例说明")
    lines.append("")
    lines.append("| 缩放比例 | 含义 | 建议 |")
    lines.append("|----------|------|------|")
    lines.append("| 🟢 ≥2.0x | 预算充裕 | 可以大幅扩展提示词，增加更多示例和说明 |")
    lines.append("| 🟡 1.0-2.0x | 预算适中 | 当前提示词长度合适，可适当优化 |")
    lines.append("| 🔴 <1.0x | 预算紧张 | 需要精简提示词，或增加预算 |")
    lines.append("")
    
    lines.append("### 如何调整提示词")
    lines.append("")
    lines.append("1. **扩展提示词**（缩放比例 > 1）:")
    lines.append("   - 增加更多示例")
    lines.append("   - 添加边界情况说明")
    lines.append("   - 补充领域知识")
    lines.append("")
    lines.append("2. **精简提示词**（缩放比例 < 1）:")
    lines.append("   - 删除冗余说明")
    lines.append("   - 合并相似规则")
    lines.append("   - 使用更简洁的表达")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*本报告基于缩放法迭代计算生成，实际费用可能因API调用情况略有差异*")
    
    # 保存报告
    report_content = "\n".join(lines)
    
    # 获取项目根目录
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    report_path = os.path.join(project_root, 'prompt_length_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path
