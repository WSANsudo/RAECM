"""
6Analyst 配置模块
集中管理所有配置项，包括API配置、文件路径、处理参数和数据清洗规则
"""

import os

# ========= 基础路径配置 =========
# 获取6Analyst包的根目录（config.py所在目录）
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（6Analyst的父目录）
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)

def _pkg_path(*paths):
    """生成相对于6Analyst包目录的路径"""
    return os.path.join(_PACKAGE_DIR, *paths)

def _proj_path(*paths):
    """生成相对于项目根目录的路径"""
    return os.path.join(_PROJECT_ROOT, *paths)

# ========= API配置 =========
# API密钥和基础URL配置
# 文档参考: 文档/聊天接口 - ChatAnywhere API 帮助文档.md
API_KEY = "sk-UJY9AdodDkq2mALVD0vBIrfUuKDuGLkxllgINwbbgJlcOQT8"
BASE_URL = "https://api.chatanywhere.tech/v1"
MODEL_NAME = "gemini-2.5-flash-lite"  # 使用小容量模型，突出数据压缩优势

# ========= 文件路径配置 =========
INPUT_DIR = _pkg_path("data", "input")                              # 输入文件夹（存放原始json文件）
TEST_INPUT_DIR = _pkg_path("data", "input_test")                    # 测试数据文件夹

# 常规模式输出路径
OUTPUT_DIR = _pkg_path("data", "output")                            # 常规输出目录
CLEANED_DATA_PATH = _pkg_path("data", "output", "cleaned_data.jsonl")       # 清洗后数据
PRODUCT_OUTPUT_PATH = _pkg_path("data", "output", "product_analysis.jsonl") # 产品分析结果
MERGED_OUTPUT_PATH = _pkg_path("data", "output", "merged_analysis.jsonl")   # 汇总中间结果
CHECK_OUTPUT_PATH = _pkg_path("data", "output", "check_details.jsonl")      # 校验详情
FINAL_OUTPUT_PATH = _proj_path("final_analysis.jsonl")              # 最终修正结果（项目根目录）
RUN_STATE_PATH = _pkg_path("data", "output", "run_state.json")      # 运行状态记录文件
LOG_DIR = _pkg_path("data", "output", "log")                        # 日志目录

# 测试模式输出路径
TEST_OUTPUT_DIR = _pkg_path("data", "output", "test_output")        # 测试输出目录
TEST_CLEANED_DATA_PATH = _pkg_path("data", "output", "test_output", "cleaned_data.jsonl")
TEST_PRODUCT_OUTPUT_PATH = _pkg_path("data", "output", "test_output", "product_analysis.jsonl")
TEST_MERGED_OUTPUT_PATH = _pkg_path("data", "output", "test_output", "merged_analysis.jsonl")
TEST_CHECK_OUTPUT_PATH = _pkg_path("data", "output", "test_output", "check_details.jsonl")
TEST_FINAL_OUTPUT_PATH = _pkg_path("data", "output", "test_output", "final_analysis.jsonl")
TEST_RUN_STATE_PATH = _pkg_path("data", "output", "test_output", "run_state.json")
TEST_LOG_DIR = _pkg_path("data", "output", "test_output", "log")

# ========= 任务类型配置 =========
# 任务类型映射：命令行参数 -> (任务名称, 输入文件名)
# 命令行使用简称 os/vd/dt/mg，内部使用全称
TASK_TYPES = {
    'os': ('os', 'os_input_data.jsonl'),
    'vd': ('vendor', 'vendor_input_data.jsonl'),
    'dt': ('devicetype', 'devicetype_input_data.jsonl'),
    'mg': ('merged', 'merged_input_data.jsonl'),  # 融合标签任务
    'all': ('all', None),  # all 任务不直接使用输入文件，而是并行执行 os/vd/dt
}

# 单独任务类型（不包含 all 和 mg）
SINGLE_TASK_TYPES = ['os', 'vd', 'dt']

def get_task_paths(task_type: str) -> dict:
    """
    获取指定任务类型的所有路径配置
    
    Args:
        task_type: 任务类型 ('os', 'vd', 'dt', 'all')
        
    Returns:
        包含所有路径的字典
        
    目录结构:
        输入: 6Analyst/data/input/{task}_input_data.jsonl
        中间输出: 6Analyst/data/output/{task_name}/temp/
        最终输出: 6Analyst/data/output/{task_name}/final/
        
    注意: 'all' 任务类型不直接使用，而是并行执行 os/vd/dt
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"未知的任务类型: {task_type}，支持的类型: {list(TASK_TYPES.keys())}")
    
    # all 任务类型特殊处理
    if task_type == 'all':
        return {
            'task_name': 'all',
            'task_type': 'all',
            'input_path': None,
            'input_filename': None,
            'temp_dir': None,
            'final_dir': None,
            'sub_tasks': SINGLE_TASK_TYPES,  # 包含的子任务
        }
    
    task_name, input_filename = TASK_TYPES[task_type]
    
    # 输入路径: 6Analyst/data/input/
    input_path = _pkg_path("data", "input", input_filename) if input_filename else None
    
    # 输出目录结构: 6Analyst/data/output/{task_name}/temp 和 final
    # 使用任务全称作为目录名（vendor/os/devicetype）
    temp_dir = _pkg_path("data", "output", task_name, "temp")
    final_dir = _pkg_path("data", "output", task_name, "final")
    
    return {
        'task_name': task_name,
        'task_type': task_type,
        'input_path': input_path,
        'input_filename': input_filename,
        'temp_dir': temp_dir,
        'final_dir': final_dir,
        # 中间文件路径（temp目录）
        'cleaned_data': os.path.join(temp_dir, "cleaned_data.jsonl"),
        'product_output': os.path.join(temp_dir, "product_analysis.jsonl"),
        'merged_output': os.path.join(temp_dir, "merged_analysis.jsonl"),
        'check_output': os.path.join(temp_dir, "check_details.jsonl"),
        'run_state': os.path.join(temp_dir, "run_state.json"),
        'log_dir': os.path.join(temp_dir, "log"),
        # 最终输出路径（final目录）
        'final_output': os.path.join(final_dir, "final_analysis.jsonl"),
    }


def get_all_task_data_counts() -> dict:
    """
    获取所有任务类型的数据量统计
    
    Returns:
        {task_type: count} 字典，包含 os/vd/dt/all 的数据量
    """
    counts = {}
    total = 0
    
    for task_type in SINGLE_TASK_TYPES:
        paths = get_task_paths(task_type)
        input_path = paths['input_path']
        if input_path and os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            counts[task_type] = count
            total += count
        else:
            counts[task_type] = 0
    
    counts['all'] = total
    return counts

# ========= 处理参数 =========
MAX_RECORDS = None  # 默认处理全部记录，设置数值则限制处理条数
BATCH_SIZE = 3      # 每批向大模型提交3条数据
MAX_INPUT_TOKENS = 4096  # 输入token上限
DEBUG_MODE = False  # 调试模式：输出详细的解析失败信息

# ========= 数据清洗参数 =========
# 默认信息熵筛选比例：清洗后按信息熵排序，保留前N%的数据
# 设置为 None 表示不进行信息熵筛选，保留全部数据
DEFAULT_ENTROPY_RATIO = 1.0  # 默认保留前100%的数据（不筛选）

# ========= 难度分级采样配置 =========
# 厂商难度分级：根据识别难度将厂商分为三类
DIFFICULTY_VENDORS = {
    'easy': ['MikroTik', 'Keenetic'],           # 易识别厂商（特征明显）
    'normal': ['Cisco', 'Juniper'],             # 较易识别厂商（特征较明显）
    'hard': []                                   # 难识别厂商（其他所有厂商，自动识别）
}

# 难度分级采样比例：各难度级别在采样中的占比
DEFAULT_DIFFICULTY_RATIOS = {
    'easy': 0.7,       # 易识别厂商占70%
    'normal': 0.15,    # 较易识别厂商占15%
    'hard': 0.15       # 难识别厂商占15%
}

# 类别内单厂商上限：在每个难度类别内部，单个厂商最多占该类别的比例
DEFAULT_MAX_VENDOR_RATIO_PER_CATEGORY = 0.6  # 默认60%

# ========= 速度等级配置 =========
# 等级定义：数字越大速度越快，'s'为最高等级（并行模式，当前已停用）
SPEED_LEVELS = {
    1: {'delay': 10.0, 'parallel': False, 'desc': '最慢 - 间隔10秒'},
    2: {'delay': 3.0, 'parallel': False, 'desc': '慢速 - 间隔3秒'},
    3: {'delay': 1.0, 'parallel': False, 'desc': '中速 - 间隔1秒'},
    4: {'delay': 0.5, 'parallel': False, 'desc': '快速 - 间隔0.5秒'},
    5: {'delay': 0.1, 'parallel': False, 'desc': '高速 - 间隔0.1秒'},
    6: {'delay': 0.0, 'parallel': False, 'desc': '极速 - 无间隔'},
    # 's' 模式已停用（当前没有可并行的Agent），保留代码以便后续重新启用
    # 's': {'delay': 0.0, 'parallel': True, 'desc': '最高速 - 并行模式'},
}
DEFAULT_SPEED_LEVEL = 6  # 默认速度等级（极速模式）

# ========= 模型价格配置（单位：元/1K token）=========
# 价格来源：api-README.md
# 注意：此表与 cost_calculator.py 中的 MODEL_PRICING 保持同步
MODEL_PRICES = {
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
    
    # ===== 默认价格 =====
    "default": {"input": 0.01, "output": 0.04}
}

# ========= 默认运行模式 =========
DEFAULT_MODE = "all"  # 默认开启全部功能（清洗+产品分析+用途分析+汇总）

# ========= 数据清洗规则 =========
# 需要删除的顶层字段（包含标签信息，避免数据泄露）
REMOVE_TOP_FIELDS = {"IP Index", "Timestamp", "OS", "Vendor", "Device Type"}

# 需要删除的服务字段
REMOVE_SERVICE_FIELDS = {"Body sha256"}

# HTTP 错误响应模式（匹配则过滤Body）
HTTP_ERROR_PATTERNS = [
    (r'400\s*Bad\s*Request', '400 Bad Request'),
    (r'401\s*(Unauthorized|Authorization\s*Required)', '401 Unauthorized'),
    (r'403\s*Forbidden', '403 Forbidden'),
    (r'404(\s*Not\s*Found)?', '404 Not Found'),
    (r'406\s*Not\s*Acceptable', '406 Not Acceptable'),
    (r'415\s*Unsupported\s*Media\s*Type', '415 Unsupported Media Type'),
    (r'421\s*Misdirected\s*Request', '421 Misdirected Request'),
    (r'500\s*Internal\s*Server\s*Error', '500 Internal Server Error'),
    (r'502\s*Bad\s*Gateway', '502 Bad Gateway'),
    (r'503\s*Service\s*(Unavailable|Temporarily)', '503 Service Unavailable'),
]

# 默认页面模式（匹配则标签化）
DEFAULT_PAGE_PATTERNS = [
    (r'web\s*host\s*not\s*found', '[DEFAULT:web-host-not-found]'),
    (r"Web\s*Server's\s*Default\s*Page", '[DEFAULT:plesk-default]'),
    (r'Default\s*(Parallels\s*)?Plesk', '[DEFAULT:plesk-default]'),
    (r'Plesk\s*(Onyx|Obsidian|Panel)?\s*\d+', '[DEFAULT:plesk-panel]'),
    (r'Apache2?\s*(Debian|Ubuntu|CentOS)?\s*Default.*It\s*works', '[DEFAULT:apache-default]'),
    (r'Apache\s*HTTP\s*Server\s*Test\s*Page', '[DEFAULT:apache-test]'),
    (r'Test\s*Page.*Apache', '[DEFAULT:apache-test]'),
    (r'Welcome\s*to\s*nginx', '[DEFAULT:nginx-default]'),
    (r'Geen\s*webhosting\s*actief', '[DEFAULT:no-webhosting-nl]'),
    (r'Domain\s*(Default|nicht\s*verfügbar|not\s*available)', '[DEFAULT:domain-unavailable]'),
    (r'No\s*title\s*found', '[DEFAULT:no-title]'),
    (r'Shared\s*IP', '[DEFAULT:shared-ip]'),
    (r'Directadmin\s*Default', '[DEFAULT:directadmin-default]'),
    (r'HTTP\s*Server\s*Test\s*Page.*CentOS', '[DEFAULT:centos-test]'),
    (r'Server\s*error', '[DEFAULT:server-error]'),
]


# ========= SSH算法压缩配置 =========
# 常见密钥交换算法
COMMON_KEX = {
    "curve25519-sha256", 
    "curve25519-sha256@libssh.org", 
    "ecdh-sha2-nistp256", 
    "ecdh-sha2-nistp384", 
    "ecdh-sha2-nistp521", 
    "diffie-hellman-group-exchange-sha256",
    "diffie-hellman-group16-sha512", 
    "diffie-hellman-group18-sha512",
    "diffie-hellman-group14-sha256", 
    "diffie-hellman-group14-sha1"
}

# 常见主机密钥算法
COMMON_HOST_KEY = {
    "ssh-rsa", 
    "rsa-sha2-512", 
    "rsa-sha2-256", 
    "ecdsa-sha2-nistp256", 
    "ssh-ed25519"
}

# 常见加密算法
COMMON_ENCRYPTION = {
    "chacha20-poly1305@openssh.com", 
    "aes128-ctr", 
    "aes192-ctr", 
    "aes256-ctr",
    "aes128-gcm@openssh.com", 
    "aes256-gcm@openssh.com"
}

# 常见MAC算法
COMMON_MAC = {
    "umac-64-etm@openssh.com", 
    "umac-128-etm@openssh.com", 
    "hmac-sha2-256-etm@openssh.com",
    "hmac-sha2-512-etm@openssh.com", 
    "hmac-sha1-etm@openssh.com", 
    "umac-64@openssh.com",
    "umac-128@openssh.com", 
    "hmac-sha2-256", 
    "hmac-sha2-512", 
    "hmac-sha1"
}

# 特殊算法（需要特别标注）
SPECIAL_ALGORITHMS = {
    "sntrup761x25519-sha512@openssh.com",  # 后量子算法
    "kex-strict-s-v00@openssh.com",         # Terrapin修复
    "diffie-hellman-group-exchange-sha1",   # 遗留DH
    "diffie-hellman-group1-sha1",           # 遗留DH
    "ssh-dss",                               # DSS算法
    "3des-cbc",                              # 3DES
    "aes128-cbc",                            # CBC模式
    "aes256-cbc"                             # CBC模式
}

# ========= Banner清洗正则模式 =========
# TLS Banner需要删除的模式
TLS_REMOVE_PATTERNS = [
    r"^\s*modulus:.*", 
    r"^\s*exponent:.*", 
    r"^Signature Value:.*",
    r"^\s*Subject Key Identifier:.*", 
    r"^\s*Authority Key Identifier:.*"
]

# MySQL Banner需要删除的模式
MYSQL_REMOVE_PATTERNS = [
    r"Capability Flags:[\s\S]*?(?=SSL Certificate:|$)"
]

# SSH Banner需要删除的模式
SSH_REMOVE_PATTERNS = [
    r"^Public Key:.*", 
    r"^Fingerprint_sha256:.*"
]
