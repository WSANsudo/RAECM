"""
提示词管理模块
支持从JSON文件加载不同版本的提示词
提示词ID格式: p1/p2/p3 (product), u1/u2/u3 (usage), c1/c2/c3 (check)
"""

import json
import os
from typing import Dict, Optional, List

# 提示词文件目录
_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Agent类型映射
AGENT_TYPE_MAP = {
    'p': 'product',
    'u': 'usage', 
    'c': 'check',
    'product': 'product',
    'usage': 'usage',
    'check': 'check'
}

# 用于费用计算的常用模型列表（按类别和价格排序）
COST_CALC_MODELS = [
    # 性价比之选（推荐）
    'deepseek-v3.2',           # DeepSeek最新版，极致性价比
    'gemini-2.5-flash-lite-preview-06-17',  # 最便宜
    'gpt-5-nano-ca',           # GPT系列最便宜
    'gemini-2.5-flash',        # Google性价比之选
    'grok-4-fast',             # Grok快速版
    'gpt-4.1-nano',            # GPT-4.1最便宜
    # 主流模型
    'gpt-4o-mini',             # OpenAI性价比之选
    'gpt-4o-mini-ca',          # 第三方渠道
    'gpt-5-mini',              # GPT-5轻量版
    'claude-3-5-haiku-20241022', # Claude轻量版
    'qwen3-235b-a22b',         # 通义千问
    'kimi-k2-0711-preview',    # Kimi
    # 高端模型
    'gpt-4o',                  # OpenAI主力
    'gpt-5',                   # GPT-5
    'gemini-2.5-pro',          # Google高端
    'claude-3-5-sonnet-20241022', # Claude主力
    'grok-4',                  # Grok
    'deepseek-r1',             # DeepSeek推理
]


def _get_prompt_file(agent_type: str) -> str:
    """获取提示词文件路径"""
    return os.path.join(_PROMPTS_DIR, f"{agent_type}_prompts.json")


def load_prompts(agent_type: str) -> Dict:
    """
    加载指定Agent类型的所有提示词
    
    Args:
        agent_type: 'product', 'usage', 或 'check'
    
    Returns:
        提示词配置字典
    """
    file_path = _get_prompt_file(agent_type)
    if not os.path.exists(file_path):
        return {"prompts": []}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_prompts(agent_type: str, config: Dict) -> None:
    """
    保存提示词配置到文件
    
    Args:
        agent_type: 'product', 'usage', 或 'check'
        config: 提示词配置字典
    """
    file_path = _get_prompt_file(agent_type)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_prompt_by_id(agent_type: str, prompt_id: str) -> Optional[str]:
    """
    根据ID获取指定的提示词
    
    Args:
        agent_type: 'product', 'usage', 或 'check'
        prompt_id: 提示词ID（如 'p1', 'u2', 'c3' 或 'default'）
    
    Returns:
        提示词内容，如果是'default'或找不到则返回None
    """
    if prompt_id == 'default':
        return None
    
    config = load_prompts(agent_type)
    for prompt in config.get('prompts', []):
        if prompt.get('id') == prompt_id:
            return prompt.get('system_prompt')
    
    return None


def get_prompt_info(agent_type: str, prompt_id: str) -> Optional[Dict]:
    """
    获取提示词的完整信息（包括费用信息）
    
    Args:
        agent_type: 'product', 'usage', 或 'check'
        prompt_id: 提示词ID
    
    Returns:
        提示词完整信息字典
    """
    config = load_prompts(agent_type)
    for prompt in config.get('prompts', []):
        if prompt.get('id') == prompt_id:
            return prompt
    return None


def list_available_prompts(agent_type: str) -> List[Dict]:
    """
    列出指定Agent类型的所有可用提示词
    
    Args:
        agent_type: 'product', 'usage', 或 'check'
    
    Returns:
        提示词列表，每项包含 id, name, description, cost_info
    """
    config = load_prompts(agent_type)
    return [
        {
            'id': p.get('id'),
            'name': p.get('name'),
            'description': p.get('description'),
            'cost_info': p.get('cost_info', {})
        }
        for p in config.get('prompts', [])
    ]


def get_all_prompt_ids() -> Dict[str, List[str]]:
    """
    获取所有Agent类型的提示词ID列表
    
    Returns:
        {agent_type: [prompt_ids]}
    """
    result = {}
    for agent_type in ['product', 'usage', 'check']:
        prompts = list_available_prompts(agent_type)
        result[agent_type] = [p['id'] for p in prompts]
    return result


def parse_prompt_id(prompt_id: str) -> tuple:
    """
    解析提示词ID，返回(agent_type, id)
    
    Args:
        prompt_id: 如 'p1', 'u2', 'c3'
    
    Returns:
        (agent_type, prompt_id) 如 ('product', 'p1')
    """
    if not prompt_id or prompt_id == 'default':
        return None, 'default'
    
    prefix = prompt_id[0].lower()
    if prefix in AGENT_TYPE_MAP:
        return AGENT_TYPE_MAP[prefix], prompt_id
    
    return None, prompt_id


def load_model_pricing_from_csv() -> Dict[str, Dict[str, float]]:
    """
    从CSV文件加载所有模型定价
    
    Returns:
        {model_name: {'input': input_cost, 'output': output_cost}}
    """
    import csv
    csv_path = os.path.join(os.path.dirname(_PROMPTS_DIR), 'data', 'model_pricing.csv')
    
    pricing = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get('model', '').strip()
                if model:
                    try:
                        pricing[model] = {
                            'input': float(row.get('input_cost', 0)),
                            'output': float(row.get('output_cost', 0))
                        }
                    except (ValueError, TypeError):
                        continue
    return pricing


def calculate_prompt_cost(system_prompt: str, batch_size: int = 3, 
                          avg_data_tokens: float = 800,
                          avg_output_tokens: float = 150,
                          use_all_models: bool = False) -> Dict:
    """
    计算单个提示词的费用信息（基于估算的平均数据token）
    
    Args:
        system_prompt: 系统提示词内容
        batch_size: 批次大小
        avg_data_tokens: 每条数据的平均token数
        avg_output_tokens: 每条数据的平均输出token数
        use_all_models: 是否计算所有模型的开销（从CSV加载）
    
    Returns:
        费用信息字典
    """
    from ..utils.token_counter import get_tokenizer, count_message_tokens
    from ..cost_calculator import MODEL_PRICING
    
    tokenizer = get_tokenizer()
    
    # 计算提示词的token数
    prompt_tokens = count_message_tokens([
        {"role": "system", "content": system_prompt}
    ], tokenizer)
    
    char_count = len(system_prompt)
    
    # 计算每1K条数据的开销
    # 每批次的输入token = 提示词token + 数据token * batch_size
    # 每1K条数据的批次数 = 1000 / batch_size
    batches_per_1k = 1000 / batch_size
    input_tokens_per_1k = batches_per_1k * prompt_tokens + 1000 * avg_data_tokens
    output_tokens_per_1k = 1000 * avg_output_tokens
    
    # 选择使用哪个定价数据源
    if use_all_models:
        # 从CSV加载所有模型定价
        all_pricing = load_model_pricing_from_csv()
        # 合并内置定价（CSV优先）
        pricing_source = {**MODEL_PRICING, **all_pricing}
        models_to_calc = list(pricing_source.keys())
    else:
        pricing_source = MODEL_PRICING
        models_to_calc = COST_CALC_MODELS
    
    cost_per_1k = {}
    for model in models_to_calc:
        if model in pricing_source:
            pricing = pricing_source[model]
            input_cost = (input_tokens_per_1k / 1000) * pricing['input']
            output_cost = (output_tokens_per_1k / 1000) * pricing['output']
            cost_per_1k[model] = round(input_cost + output_cost, 4)
    
    return {
        'char_count': char_count,
        'token_count': prompt_tokens,
        'cost_per_1k_records': cost_per_1k
    }


def calculate_prompt_cost_from_files(agent_type: str, system_prompt: str, batch_size: int = 3) -> Dict:
    """
    基于现有文件计算提示词的实际费用（输入+输出）
    
    Args:
        agent_type: 'product', 'usage', 或 'check'
        system_prompt: 系统提示词内容
        batch_size: 批次大小
    
    Returns:
        费用信息字典，包含总开销
    """
    import json
    from ..utils.token_counter import get_tokenizer, count_tokens, count_message_tokens
    from ..cost_calculator import MODEL_PRICING
    from ..config import CLEANED_DATA_PATH, MERGED_OUTPUT_PATH, PRODUCT_OUTPUT_PATH, CHECK_OUTPUT_PATH
    
    tokenizer = get_tokenizer()
    
    # 计算提示词的token数
    prompt_tokens = count_message_tokens([
        {"role": "system", "content": system_prompt}
    ], tokenizer)
    
    char_count = len(system_prompt)
    
    # 根据agent类型确定输入和输出文件
    if agent_type == 'product':
        input_file = CLEANED_DATA_PATH
        output_file = PRODUCT_OUTPUT_PATH
    elif agent_type == 'check':
        input_file = MERGED_OUTPUT_PATH
        output_file = CHECK_OUTPUT_PATH
    else:
        return {'char_count': char_count, 'token_count': prompt_tokens, 'error': 'unknown agent type'}
    
    # 读取输入文件计算平均数据token
    input_records = []
    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    input_records.append(line)
    
    # 读取输出文件计算平均输出token
    output_records = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    output_records.append(line)
    
    record_count = len(input_records)
    if record_count == 0:
        return {
            'char_count': char_count,
            'token_count': prompt_tokens,
            'error': 'no input data',
            'note': '无输入数据，请先运行分析'
        }
    
    # 计算输入token（采样计算，避免太慢）
    sample_size = min(100, record_count)
    sample_indices = list(range(0, record_count, max(1, record_count // sample_size)))[:sample_size]
    
    total_data_tokens = 0
    for idx in sample_indices:
        total_data_tokens += count_tokens(input_records[idx], tokenizer)
    avg_data_tokens = total_data_tokens / len(sample_indices)
    
    # 计算输出token
    total_output_tokens = 0
    output_count = len(output_records)
    if output_count > 0:
        sample_size_out = min(100, output_count)
        sample_indices_out = list(range(0, output_count, max(1, output_count // sample_size_out)))[:sample_size_out]
        for idx in sample_indices_out:
            total_output_tokens += count_tokens(output_records[idx], tokenizer)
        avg_output_tokens = total_output_tokens / len(sample_indices_out)
    else:
        # 默认估算
        avg_output_tokens = 150 if agent_type == 'product' else (180 if agent_type == 'usage' else 120)
    
    # 计算总开销
    # 输入token = 批次数 * 提示词token + 记录数 * 平均数据token
    num_batches = (record_count + batch_size - 1) // batch_size
    total_input_tokens = num_batches * prompt_tokens + record_count * avg_data_tokens
    total_output_tokens_all = record_count * avg_output_tokens
    
    # 计算各模型的总费用
    total_cost = {}
    cost_per_1k = {}
    for model in COST_CALC_MODELS:
        if model in MODEL_PRICING:
            pricing = MODEL_PRICING[model]
            input_cost = (total_input_tokens / 1000) * pricing['input']
            output_cost = (total_output_tokens_all / 1000) * pricing['output']
            total_cost[model] = round(input_cost + output_cost, 4)
            # 每1K条的开销
            cost_per_1k[model] = round((input_cost + output_cost) / record_count * 1000, 4)
    
    return {
        'char_count': char_count,
        'token_count': prompt_tokens,
        'record_count': record_count,
        'avg_input_tokens': round(avg_data_tokens, 1),
        'avg_output_tokens': round(avg_output_tokens, 1),
        'total_cost': total_cost,
        'cost_per_1k_records': cost_per_1k
    }


def update_all_prompt_costs(batch_size: int = 3) -> Dict:
    """
    更新所有提示词的费用信息并保存到JSON文件
    基于现有文件计算实际的输入输出开销
    
    Args:
        batch_size: 批次大小
    
    Returns:
        更新统计信息
    """
    from ..product_analyst import ProductAnalyst
    from ..check_analyst import CheckAnalyst
    from ..utils.token_counter import get_tokenizer, count_message_tokens
    
    tokenizer = get_tokenizer()
    stats = {'updated': 0, 'skipped': 0, 'errors': 0}
    
    # Agent配置
    agent_configs = {
        'product': ProductAnalyst,
        'check': CheckAnalyst
    }
    
    for agent_type, agent_class in agent_configs.items():
        config = load_prompts(agent_type)
        updated = False
        
        for prompt in config.get('prompts', []):
            prompt_id = prompt.get('id')
            system_prompt = prompt.get('system_prompt')
            
            # default提示词使用内置的
            if prompt_id == 'default' or system_prompt is None:
                default_prompt = agent_class.DEFAULT_SYSTEM_PROMPT
                cost_info = calculate_prompt_cost_from_files(agent_type, default_prompt, batch_size)
                cost_info['note'] = '程序内置默认提示词'
                prompt['cost_info'] = cost_info
                updated = True
                stats['updated'] += 1
                continue
            
            try:
                cost_info = calculate_prompt_cost_from_files(agent_type, system_prompt, batch_size)
                prompt['cost_info'] = cost_info
                updated = True
                stats['updated'] += 1
            except Exception as e:
                print(f"[ERROR] 计算 {agent_type}/{prompt_id} 费用失败: {e}")
                stats['errors'] += 1
        
        if updated:
            save_prompts(agent_type, config)
    
    return stats


def print_prompts_list():
    """打印所有可用提示词的详细列表"""
    print("\n可用的提示词配置:")
    print("=" * 80)
    
    for agent_type, prefix in [('product', 'p'), ('usage', 'u'), ('check', 'c')]:
        print(f"\n【{agent_type.upper()} Agent】 (使用 -{prefix} 指定)")
        print("-" * 70)
        prompts = list_available_prompts(agent_type)
        for p in prompts:
            cost_info = p.get('cost_info', {})
            char_count = cost_info.get('char_count', 0)
            token_count = cost_info.get('token_count', 0)
            record_count = cost_info.get('record_count', 0)
            
            print(f"  {p['id']:8} - {p['name']} ({char_count}字符/{token_count}tokens)")
            
            if cost_info.get('note'):
                print(f"             {cost_info['note']}")
            elif p.get('description'):
                print(f"             {p['description']}")
            
            # 显示总开销
            total_cost = cost_info.get('total_cost', {})
            if total_cost and record_count > 0:
                costs = [f"{model}: CNY{cost:.4f}" for model, cost in total_cost.items()]
                print(f"             总开销({record_count}条): {', '.join(costs)}")
            
            # 显示每1K条开销
            cost_per_1k = cost_info.get('cost_per_1k_records', {})
            if cost_per_1k:
                costs = [f"{model}: CNY{cost:.4f}" for model, cost in cost_per_1k.items()]
                print(f"             每1K条开销: {', '.join(costs)}")
            
            # 显示平均token
            avg_input = cost_info.get('avg_input_tokens', 0)
            avg_output = cost_info.get('avg_output_tokens', 0)
            if avg_input > 0 or avg_output > 0:
                print(f"             平均token: 输入{avg_input:.0f}/输出{avg_output:.0f}")
    
    print("\n" + "=" * 80)
    print("使用示例:")
    print("  python run_6analyst.py --prompt -p p1           # 产品Agent使用p1提示词")
    print("  python run_6analyst.py --prompt -c c1           # 校验Agent使用c1提示词")
    print("  python run_6analyst.py --prompt -p p3 -c c3     # 全部使用v3版本")
    print("  python run_6analyst.py --prompt --update        # 更新所有提示词的费用信息")
    print("=" * 80)


def print_update_result(stats: Dict):
    """打印更新结果"""
    print("\n提示词费用信息更新完成:")
    print(f"  更新: {stats['updated']} 个")
    print(f"  跳过: {stats['skipped']} 个")
    print(f"  错误: {stats['errors']} 个")
