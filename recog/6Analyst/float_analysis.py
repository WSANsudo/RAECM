"""
准确率浮动分析模块

功能：
1. 读取原始数据，按信息熵排序
2. 提取前20%高信息熵数据和后20%低信息熵数据
3. 分别执行主项目流程
4. 在结果中记录每条数据的信息熵值
"""

import os
import json
import sys
from typing import List, Dict, Tuple
from datetime import datetime

from .config import INPUT_DIR, _pkg_path
from .entropy_sorter import isort_with_entropy


# 浮动分析输出目录
FLOAT_OUTPUT_DIR = _pkg_path("data", "output", "float")
HIGH_ENTROPY_DIR = os.path.join(FLOAT_OUTPUT_DIR, "high_entropy")
LOW_ENTROPY_DIR = os.path.join(FLOAT_OUTPUT_DIR, "low_entropy")


def ensure_dirs():
    """确保输出目录存在"""
    for base_dir in [HIGH_ENTROPY_DIR, LOW_ENTROPY_DIR]:
        for sub_dir in ["input", "temp", "final"]:
            dir_path = os.path.join(base_dir, sub_dir)
            os.makedirs(dir_path, exist_ok=True)


def load_raw_data(input_path: str) -> List[Dict]:
    """
    加载原始数据
    
    Args:
        input_path: 输入目录或文件路径
        
    Returns:
        数据列表
    """
    import glob
    
    records = []
    
    if os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        jsonl_files = glob.glob(os.path.join(input_path, "*.jsonl"))
        input_files = sorted(json_files + jsonl_files)
    elif os.path.isfile(input_path):
        input_files = [input_path]
    else:
        print(f"[ERROR] 输入路径不存在: {input_path}")
        return records
    
    for filepath in input_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    
    return records


def extract_entropy_groups(
    input_path: str,
    high_ratio: float = 0.1,
    low_ratio: float = 0.1
) -> Tuple[List[Tuple[Dict, float]], List[Tuple[Dict, float]]]:
    """
    按信息熵排序并提取高/低信息熵数据组
    
    Args:
        input_path: 输入数据路径
        high_ratio: 高信息熵组比例（前N%）
        low_ratio: 低信息熵组比例（后N%）
        
    Returns:
        (高信息熵组, 低信息熵组)，每组包含 (数据, 信息熵值) 元组
    """
    print(f"\n加载原始数据: {input_path}")
    records = load_raw_data(input_path)
    total_count = len(records)
    print(f"  总数据量: {total_count}")
    
    if total_count == 0:
        return [], []
    
    # 按信息熵排序（返回带熵值的数据）
    print(f"\n按信息熵排序...")
    sorted_data = isort_with_entropy(records, ratio=1.0)  # 获取全部数据的熵值
    
    # 计算分组数量
    high_count = max(1, int(total_count * high_ratio))
    low_count = max(1, int(total_count * low_ratio))
    
    # 提取高信息熵组（前N%）
    high_entropy_group = sorted_data[:high_count]
    
    # 提取低信息熵组（后N%）
    low_entropy_group = sorted_data[-low_count:]
    
    print(f"\n数据分组:")
    print(f"  高信息熵组（前{high_ratio*100:.0f}%）: {len(high_entropy_group)} 条")
    if high_entropy_group:
        print(f"    熵值范围: {high_entropy_group[-1][1]:.4f} ~ {high_entropy_group[0][1]:.4f}")
    print(f"  低信息熵组（后{low_ratio*100:.0f}%）: {len(low_entropy_group)} 条")
    if low_entropy_group:
        print(f"    熵值范围: {low_entropy_group[-1][1]:.4f} ~ {low_entropy_group[0][1]:.4f}")
    
    return high_entropy_group, low_entropy_group


def save_group_data(
    group_data: List[Tuple[Dict, float]],
    output_dir: str,
    group_name: str
) -> Tuple[str, Dict[str, float]]:
    """
    保存分组数据到输入文件，并记录每条数据的信息熵值
    
    Args:
        group_data: 带熵值的数据列表
        output_dir: 输出目录
        group_name: 组名（用于日志）
        
    Returns:
        (输入文件路径, IP到熵值的映射)
    """
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    input_file = os.path.join(input_dir, "input_data.jsonl")
    entropy_map = {}  # IP -> 熵值
    
    with open(input_file, 'w', encoding='utf-8') as f:
        for record, entropy in group_data:
            # 获取IP
            ip = next(iter(record.keys()))
            entropy_map[ip] = entropy
            
            # 写入原始数据
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"  {group_name}数据已保存: {input_file} ({len(group_data)} 条)")
    
    # 保存熵值映射（供后续合并使用）
    entropy_file = os.path.join(output_dir, "entropy_map.json")
    with open(entropy_file, 'w', encoding='utf-8') as f:
        json.dump(entropy_map, f, ensure_ascii=False, indent=2)
    
    return input_file, entropy_map


def run_main_pipeline(
    input_file: str,
    output_dir: str,
    group_name: str,
    model: str = "deepseek-v3.2",
    num_threads: int = 24
) -> bool:
    """
    调用主程序执行分析流程
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        group_name: 组名
        model: 使用的模型
        num_threads: 线程数
        
    Returns:
        是否成功
    """
    temp_dir = os.path.join(output_dir, "temp")
    final_dir = os.path.join(output_dir, "final")
    final_output = os.path.join(final_dir, "final_analysis.jsonl")
    
    # 确保目录存在
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"执行 {group_name} 分析")
    print(f"{'='*60}")
    print(f"输入: {input_file}")
    print(f"输出: {final_output}")
    print(f"模型: {model}")
    print(f"线程数: {num_threads}")
    
    try:
        # 直接调用内部模块进行分析
        from .data_cleaner import DataCleaner
        from .multi_thread_runner import MultiThreadRunner, MultiThreadConfig
        from .run_config import RunConfig, apply_config_to_globals, restore_config_from_backup
        
        # 步骤1: 数据清洗（不进行信息熵筛选）
        print(f"\n[1/2] 数据清洗...")
        cleaned_data_path = os.path.join(temp_dir, "cleaned_data.jsonl")
        cleaner = DataCleaner(input_file, cleaned_data_path, keep_labels=True)
        clean_stats = cleaner.run(max_records=None)
        print(f"清洗完成: {clean_stats.get('processed_records', 0)} 条")
        
        # 步骤2: 多线程分析
        print(f"\n[2/2] 执行分析 ({num_threads} 线程)...")
        
        # 配置输出路径
        product_output = os.path.join(temp_dir, "product_analysis.jsonl")
        merged_output = os.path.join(temp_dir, "merged_analysis.jsonl")
        check_output = os.path.join(temp_dir, "check_details.jsonl")
        run_state = os.path.join(temp_dir, "run_state.json")
        log_dir = os.path.join(temp_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        # 清除已有结果（restart模式）
        for f in [product_output, merged_output, check_output, final_output, run_state]:
            if os.path.exists(f):
                os.remove(f)
        
        # 创建运行配置并应用到全局变量
        run_config = RunConfig(
            input_path=input_file,
            cleaned_data_path=cleaned_data_path,
            product_output_path=product_output,
            merged_output_path=merged_output,
            check_output_path=check_output,
            final_output_path=final_output,
            run_state_path=run_state,
            log_dir=log_dir,
            product_model=model,
            check_model=model,
            num_threads=num_threads,
            speed_level='6',
            batch_size=3,
            skip_check=False,
            restart=True,
        )
        orig_config = apply_config_to_globals(run_config)
        
        # 创建多线程配置
        mt_config = MultiThreadConfig(
            num_workers=num_threads,
            speed_level='6',
            batch_size=3,
            skip_check=False
        )
        
        # 创建多线程运行器
        runner = MultiThreadRunner(
            config=mt_config,
            product_model=model,
            check_model=model
        )
        
        # 加载清洗后的数据并转换格式
        # 清洗后的格式是 {ip: data}，需要转换为 {'ip': ip, ...} 格式
        records = []
        with open(cleaned_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        # 转换格式：{ip: data} -> {'ip': ip, 'raw': json_str, 'data': data}
                        ip = next(iter(obj.keys()))
                        records.append({
                            'ip': ip,
                            'raw': line,
                            'data': obj[ip]
                        })
                    except (json.JSONDecodeError, StopIteration):
                        continue
        
        # 执行分析
        stats = runner.run(records)
        
        # 恢复原始配置
        restore_config_from_backup(orig_config)
        
        print(f"\n[成功] {group_name}分析完成")
        print(f"  处理记录: {stats.get('processed_records', 0)}")
        print(f"  高置信度: {stats.get('high_conf', 0)}")
        print(f"  中置信度: {stats.get('mid_conf', 0)}")
        print(f"  低置信度: {stats.get('low_conf', 0)}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"\n[错误] {group_name}分析出错: {e}")
        traceback.print_exc()
        return False


def add_entropy_to_results(output_dir: str, entropy_map: Dict[str, float]):
    """
    将信息熵值添加到结果文件中
    
    Args:
        output_dir: 输出目录
        entropy_map: IP到熵值的映射
    """
    final_file = os.path.join(output_dir, "final", "final_analysis.jsonl")
    
    if not os.path.exists(final_file):
        print(f"  [WARN] 结果文件不存在: {final_file}")
        return
    
    # 读取结果
    results = []
    with open(final_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # 添加信息熵值
                ip = record.get('ip')
                if ip and ip in entropy_map:
                    record['entropy'] = entropy_map[ip]
                results.append(record)
            except json.JSONDecodeError:
                continue
    
    # 重写结果文件
    with open(final_file, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"  已添加信息熵值到: {final_file}")


def generate_float_report(high_dir: str, low_dir: str):
    """
    生成浮动分析报告
    
    Args:
        high_dir: 高信息熵组目录
        low_dir: 低信息熵组目录
    """
    from .accuracy_calculator import (
        load_input_labels, load_output_results, calculate_accuracy
    )
    
    report_lines = [
        "# 准确率浮动分析报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 概览",
        "",
        "本报告对比高信息熵数据（前20%）和低信息熵数据（后20%）的识别准确率差异。",
        "",
    ]
    
    # 分析两组数据
    for group_name, group_dir in [("高信息熵组", high_dir), ("低信息熵组", low_dir)]:
        input_file = os.path.join(group_dir, "input", "input_data.jsonl")
        final_file = os.path.join(group_dir, "final", "final_analysis.jsonl")
        
        if not os.path.exists(final_file):
            report_lines.append(f"### {group_name}")
            report_lines.append("")
            report_lines.append("结果文件不存在，跳过分析。")
            report_lines.append("")
            continue
        
        # 加载数据（不跳过任何数据，全部纳入统计）
        labels, available_labels = load_input_labels(input_file)
        results, _, _ = load_output_results(final_file, skip_invalid=False, skip_low_confidence=False)
        
        # 计算准确率（新规则：低置信度/空值/不匹配均视为错误）
        stats = calculate_accuracy(labels, results, available_labels)
        
        report_lines.append(f"### {group_name}")
        report_lines.append("")
        report_lines.append(f"- 数据量: {stats['total_labels']}")
        report_lines.append(f"- 匹配记录: {stats['matched_count']}")
        
        if 'vendor_accuracy' in stats:
            rate = stats['vendor_accuracy']['rate'] * 100
            report_lines.append(f"- Vendor准确率: **{rate:.2f}%**")
        if 'os_accuracy' in stats:
            rate = stats['os_accuracy']['rate'] * 100
            report_lines.append(f"- OS准确率: **{rate:.2f}%**")
        if 'type_accuracy' in stats:
            rate = stats['type_accuracy']['rate'] * 100
            report_lines.append(f"- DeviceType准确率: **{rate:.2f}%**")
        
        report_lines.append("")
    
    # 保存报告
    report_path = os.path.join(FLOAT_OUTPUT_DIR, "float_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n浮动分析报告已生成: {report_path}")


def run_float_analysis(
    input_path: str = None,
    model: str = "deepseek-v3.2",
    num_threads: int = 24,
    high_ratio: float = 0.01,
    low_ratio: float = 0.01,
    mode: str = 'both'
):
    """
    执行准确率浮动分析
    
    Args:
        input_path: 输入数据路径，默认为 INPUT_DIR
        model: 使用的模型
        num_threads: 线程数
        high_ratio: 高信息熵组比例
        low_ratio: 低信息熵组比例
        mode: 处理模式 - 'both'(两组都处理), 'high'(仅高熵组), 'low'(仅低熵组)
    """
    input_path = input_path or INPUT_DIR
    
    mode_desc = {
        'both': '高熵组 + 低熵组',
        'high': '仅高熵组',
        'low': '仅低熵组'
    }
    
    print("\n" + "=" * 60)
    print(f"准确率浮动分析 (--float)")
    print("=" * 60)
    print(f"输入路径: {input_path}")
    print(f"模型: {model}")
    print(f"线程数: {num_threads}")
    print(f"处理模式: {mode_desc.get(mode, mode)}")
    if mode in ['both', 'high']:
        print(f"高信息熵组比例: {high_ratio*100:.0f}%")
    if mode in ['both', 'low']:
        print(f"低信息熵组比例: {low_ratio*100:.0f}%")
    
    # 确保目录存在
    ensure_dirs()
    
    # 提取高/低信息熵数据组
    high_group, low_group = extract_entropy_groups(input_path, high_ratio, low_ratio)
    
    if mode == 'high' and not high_group:
        print("\n[ERROR] 高信息熵数据提取失败")
        return
    if mode == 'low' and not low_group:
        print("\n[ERROR] 低信息熵数据提取失败")
        return
    if mode == 'both' and (not high_group or not low_group):
        print("\n[ERROR] 数据提取失败")
        return
    
    # 保存分组数据
    print("\n保存分组数据...")
    high_input, high_entropy_map = None, None
    low_input, low_entropy_map = None, None
    
    if mode in ['both', 'high']:
        high_input, high_entropy_map = save_group_data(high_group, HIGH_ENTROPY_DIR, "高信息熵组")
    if mode in ['both', 'low']:
        low_input, low_entropy_map = save_group_data(low_group, LOW_ENTROPY_DIR, "低信息熵组")
    
    # 执行分析流程
    print("\n开始执行分析流程...")
    
    high_success = False
    low_success = False
    
    # 执行高信息熵组分析
    if mode in ['both', 'high']:
        high_success = run_main_pipeline(
            high_input, HIGH_ENTROPY_DIR, "高信息熵组",
            model=model, num_threads=num_threads
        )
        # 添加信息熵值到结果
        if high_success:
            add_entropy_to_results(HIGH_ENTROPY_DIR, high_entropy_map)
    
    # 执行低信息熵组分析
    if mode in ['both', 'low']:
        low_success = run_main_pipeline(
            low_input, LOW_ENTROPY_DIR, "低信息熵组",
            model=model, num_threads=num_threads
        )
        # 添加信息熵值到结果
        if low_success:
            add_entropy_to_results(LOW_ENTROPY_DIR, low_entropy_map)
    
    # 生成浮动分析报告（仅当两组都完成时）
    if mode == 'both' and high_success and low_success:
        generate_float_report(HIGH_ENTROPY_DIR, LOW_ENTROPY_DIR)
    
    print("\n" + "=" * 60)
    print("浮动分析完成")
    print("=" * 60)
    if mode in ['both', 'high']:
        print(f"高信息熵组结果: {os.path.join(HIGH_ENTROPY_DIR, 'final', 'final_analysis.jsonl')}")
    if mode in ['both', 'low']:
        print(f"低信息熵组结果: {os.path.join(LOW_ENTROPY_DIR, 'final', 'final_analysis.jsonl')}")
    if mode == 'both':
        print(f"分析报告: {os.path.join(FLOAT_OUTPUT_DIR, 'float_analysis_report.md')}")


def run_float_accuracy():
    """
    计算高信息熵组和低信息熵组的准确率（--float-acc）
    
    读取已有的分析结果，计算并对比两组的准确率差异。
    """
    from .accuracy_calculator import (
        load_input_labels, load_output_results, calculate_accuracy,
        MIN_CONFIDENCE_THRESHOLD
    )
    
    print("\n" + "=" * 70)
    print("浮动分析准确率计算 (--float-acc)")
    print("=" * 70)
    
    results_summary = {}
    
    for group_name, group_dir in [("高信息熵组", HIGH_ENTROPY_DIR), ("低信息熵组", LOW_ENTROPY_DIR)]:
        input_file = os.path.join(group_dir, "input", "input_data.jsonl")
        final_file = os.path.join(group_dir, "final", "final_analysis.jsonl")
        
        print(f"\n{'='*60}")
        print(f"{group_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(input_file):
            print(f"[ERROR] 输入文件不存在: {input_file}")
            print(f"  请先运行 --float 生成分组数据")
            continue
            
        if not os.path.exists(final_file):
            print(f"[ERROR] 结果文件不存在: {final_file}")
            print(f"  请先运行 --float 执行分析流程")
            continue
        
        # 加载标签
        print(f"\n加载输入标签: {input_file}")
        labels, available_labels = load_input_labels(input_file)
        print(f"  找到 {len(labels)} 条有标签的记录")
        print(f"  检测到的标签类型: {', '.join(available_labels) if available_labels else '无'}")
        
        if not available_labels:
            print(f"[WARN] 未检测到标签字段，跳过准确率计算")
            continue
        
        # 加载结果（不跳过任何数据，全部纳入统计）
        print(f"\n加载输出结果: {final_file}")
        results, invalid_count, low_conf_count = load_output_results(
            final_file, skip_invalid=False, skip_low_confidence=False
        )
        print(f"  找到 {len(results)} 条结果记录")
        if invalid_count > 0:
            print(f"  其中 {invalid_count} 条预测为空（视为错误）")
        if low_conf_count > 0:
            print(f"  其中 {low_conf_count} 条低置信度（<{MIN_CONFIDENCE_THRESHOLD}，视为错误）")
        
        # 计算准确率
        print(f"\n计算准确率...")
        stats = calculate_accuracy(labels, results, available_labels)
        
        # 辅助函数：显示错误分类
        def print_error_breakdown(accuracy_data: Dict, label_name: str):
            errors = accuracy_data.get('errors', {})
            total_errors = accuracy_data['total'] - accuracy_data['correct']
            if total_errors > 0:
                print(f"  错误分类（共{total_errors}条）：")
                if errors.get('low_confidence', 0) > 0:
                    print(f"    - 低置信度: {errors['low_confidence']}")
                if errors.get('empty_prediction', 0) > 0:
                    print(f"    - 预测为空: {errors['empty_prediction']}")
                if errors.get('label_empty_pred_not', 0) > 0:
                    print(f"    - 标签空但预测不空: {errors['label_empty_pred_not']}")
                if errors.get('mismatch', 0) > 0:
                    print(f"    - 不匹配: {errors['mismatch']}")
        
        # 显示结果
        print(f"\n匹配记录数: {stats['matched_count']}")
        
        group_stats = {'matched_count': stats['matched_count']}
        
        if 'vendor' in available_labels and 'vendor_accuracy' in stats:
            print(f"\n厂商识别准确率 (Vendor):")
            print(f"  正确: {stats['vendor_accuracy']['correct']}/{stats['vendor_accuracy']['total']}")
            print(f"  准确率: {stats['vendor_accuracy']['rate']*100:.2f}%")
            print_error_breakdown(stats['vendor_accuracy'], 'vendor')
            group_stats['vendor'] = stats['vendor_accuracy']['rate']
        
        if 'os' in available_labels and 'os_accuracy' in stats:
            print(f"\n操作系统识别准确率 (OS):")
            print(f"  正确: {stats['os_accuracy']['correct']}/{stats['os_accuracy']['total']}")
            print(f"  准确率: {stats['os_accuracy']['rate']*100:.2f}%")
            print_error_breakdown(stats['os_accuracy'], 'os')
            group_stats['os'] = stats['os_accuracy']['rate']
        
        if 'device_type' in available_labels and 'type_accuracy' in stats:
            print(f"\n设备类型识别准确率 (Device Type):")
            print(f"  正确: {stats['type_accuracy']['correct']}/{stats['type_accuracy']['total']}")
            print(f"  准确率: {stats['type_accuracy']['rate']*100:.2f}%")
            print_error_breakdown(stats['type_accuracy'], 'device_type')
            group_stats['device_type'] = stats['type_accuracy']['rate']
        
        results_summary[group_name] = group_stats
    
    # 对比分析
    if len(results_summary) == 2:
        print("\n" + "=" * 70)
        print("准确率对比分析")
        print("=" * 70)
        
        high_stats = results_summary.get("高信息熵组", {})
        low_stats = results_summary.get("低信息熵组", {})
        
        print(f"\n{'指标':<20} {'高信息熵组':>15} {'低信息熵组':>15} {'差异':>15}")
        print("-" * 70)
        
        for metric, metric_name in [('vendor', 'Vendor准确率'), 
                                     ('os', 'OS准确率'), 
                                     ('device_type', 'DeviceType准确率')]:
            if metric in high_stats and metric in low_stats:
                high_rate = high_stats[metric] * 100
                low_rate = low_stats[metric] * 100
                diff = high_rate - low_rate
                diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
                print(f"{metric_name:<20} {high_rate:>14.2f}% {low_rate:>14.2f}% {diff_str:>15}")
        
        print("-" * 70)
        print("\n结论：")
        
        # 计算平均差异
        diffs = []
        for metric in ['vendor', 'os', 'device_type']:
            if metric in high_stats and metric in low_stats:
                diffs.append(high_stats[metric] - low_stats[metric])
        
        if diffs:
            avg_diff = sum(diffs) / len(diffs) * 100
            if avg_diff > 5:
                print(f"  高信息熵数据的平均准确率比低信息熵数据高 {avg_diff:.2f}%")
                print(f"  信息熵与识别准确率呈正相关")
            elif avg_diff < -5:
                print(f"  低信息熵数据的平均准确率比高信息熵数据高 {abs(avg_diff):.2f}%")
                print(f"  信息熵与识别准确率呈负相关（异常情况）")
            else:
                print(f"  高低信息熵数据的准确率差异不大（平均差异 {avg_diff:.2f}%）")
                print(f"  信息熵对识别准确率影响有限")
    
    print("\n" + "=" * 70)
    print("浮动分析准确率计算完成")
    print("=" * 70)
