#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型专项训练组脚本
Full Training Pipeline for a Specific Model

执行指定模型的完整训练流程：
    训练 vendor → 评估 vendor → 预测 vendor
    训练 os → 评估 os → 预测 os
    训练 devicetype → 评估 devicetype → 预测 devicetype

用法:
    python full_run.py
"""

import argparse
import gc
import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 run.py 中的函数
from run import (
    run_train,
    run_evaluate,
    run_predict,
    get_config_path,
    setup_environment,
    get_available_models,
    select_model_interactive,
    select_models_interactive
)

import yaml


def load_data_balance_config(config_path: str) -> Dict:
    """
    加载数据平衡配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"数据配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_balance_ratios_from_config(config: Dict, task_type: str) -> Dict:
    """
    从配置中获取指定任务的平衡比例
    
    Args:
        config: 配置字典
        task_type: 任务类型 (vendor/os/devicetype)
        
    Returns:
        {'train': float|None, 'valid': float|None, 'test': float|None}
    """
    result = {'train': None, 'valid': None, 'test': None}
    
    # 先获取默认配置
    default_config = config.get('default', {})
    for split in ['train', 'valid', 'test']:
        split_config = default_config.get(split, {})
        if split_config:
            result[split] = split_config.get('major_ratio')
    
    # 再获取任务特定配置（覆盖默认）
    tasks_config = config.get('tasks', {})
    task_config = tasks_config.get(task_type, {})
    for split in ['train', 'valid', 'test']:
        split_config = task_config.get(split, {})
        if split_config and 'major_ratio' in split_config:
            result[split] = split_config.get('major_ratio')
    
    return result


# 重试配置
RETRY_WAIT_SECONDS = 300  # 5分钟
MAX_RETRIES = 1  # 最多重试1次（总共执行2次）


def clear_gpu_memory():
    """
    强制清理 GPU 显存
    在每个任务完成后调用，确保显存被释放
    """
    try:
        import torch
        if torch.cuda.is_available():
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 同步 CUDA 操作
            torch.cuda.synchronize()
            # 强制垃圾回收
            gc.collect()
            # 再次清空缓存
            torch.cuda.empty_cache()
            
            # 打印当前显存使用情况
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
    except Exception as e:
        print(f"  清理 GPU 显存时出错: {e}")


def setup_logging(model_name: str) -> logging.Logger:
    """设置日志"""
    # 创建日志目录
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"full_run_{model_name}_{timestamp}.log")
    
    # 配置日志
    logger = logging.getLogger(f"full_run_{model_name}")
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


def run_with_retry(
    func,
    func_name: str,
    logger: logging.Logger,
    *args,
    **kwargs
) -> bool:
    """
    执行函数并在失败时重试
    
    Args:
        func: 要执行的函数
        func_name: 函数名称（用于日志）
        logger: 日志记录器
        *args, **kwargs: 传递给函数的参数
    
    Returns:
        是否成功
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"{'重试' if attempt > 0 else '开始'} {func_name}...")
            success = func(*args, **kwargs)
            
            # 任务完成后清理 GPU 显存
            logger.info(f"清理 GPU 显存...")
            clear_gpu_memory()
            
            if success:
                logger.info(f"✓ {func_name} 成功")
                return True
            else:
                logger.warning(f"✗ {func_name} 失败（返回 False）")
                
                if attempt < MAX_RETRIES:
                    logger.info(f"等待 {RETRY_WAIT_SECONDS} 秒后重试...")
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    logger.error(f"✗ {func_name} 最终失败，已达到最大重试次数")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ {func_name} 异常: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 异常后也要清理 GPU 显存
            logger.info(f"清理 GPU 显存...")
            clear_gpu_memory()
            
            if attempt < MAX_RETRIES:
                logger.info(f"等待 {RETRY_WAIT_SECONDS} 秒后重试...")
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                logger.error(f"✗ {func_name} 最终失败，已达到最大重试次数")
                return False
    
    return False


def run_task_pipeline(
    model_name: str,
    task_type: str,
    logger: logging.Logger,
    mode: str = 'simple',
    subtasks: List[str] = None
) -> Dict[str, bool]:
    """
    执行单个任务的完整流程：训练 -> 评估 -> 预测
    每个步骤都有重试机制，每个步骤完成后清理 GPU 显存
    
    Args:
        model_name: 模型名称
        task_type: 任务类型 (vd/os/dt)
        logger: 日志记录器
        mode: 训练模式
        subtasks: 要执行的子任务列表 (train/evaluate/predict)
    
    Returns:
        {'train': bool, 'evaluate': bool, 'predict': bool}
    """
    if subtasks is None:
        subtasks = ['train', 'evaluate', 'predict']
    
    task_names = {"vd": "厂商识别", "os": "操作系统识别", "dt": "设备类型识别"}
    task_name = task_names.get(task_type, task_type)
    
    results = {'train': False, 'evaluate': False, 'predict': False}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"开始任务: {task_name} ({task_type})")
    logger.info(f"子任务: {', '.join(subtasks)}")
    logger.info(f"{'='*70}")
    
    # 任务开始前先清理一次显存
    logger.info("任务开始前清理 GPU 显存...")
    clear_gpu_memory()
    
    subtask_count = len(subtasks)
    current_step = 0
    
    # 1. 训练（带重试）
    if 'train' in subtasks:
        current_step += 1
        logger.info(f"\n[{current_step}/{subtask_count}] 训练 {task_name}...")
        results['train'] = run_with_retry(
            run_train,
            f"训练 {task_name}",
            logger,
            model_name, task_type, mode
        )
    else:
        logger.info(f"\n跳过训练 {task_name}")
        results['train'] = None  # 标记为跳过
    
    # 2. 评估（带重试，即使训练失败也尝试）
    if 'evaluate' in subtasks:
        current_step += 1
        logger.info(f"\n[{current_step}/{subtask_count}] 评估 {task_name}...")
        results['evaluate'] = run_with_retry(
            run_evaluate,
            f"评估 {task_name}",
            logger,
            model_name, task_type, mode
        )
    else:
        logger.info(f"\n跳过评估 {task_name}")
        results['evaluate'] = None  # 标记为跳过
    
    # 3. 预测（带重试）
    if 'predict' in subtasks:
        current_step += 1
        logger.info(f"\n[{current_step}/{subtask_count}] 预测 {task_name}...")
        results['predict'] = run_with_retry(
            run_predict,
            f"预测 {task_name}",
            logger,
            model_name, task_type, None, 4, mode
        )
    else:
        logger.info(f"\n跳过预测 {task_name}")
        results['predict'] = None  # 标记为跳过
    
    # 任务完成后再清理一次显存
    logger.info(f"\n任务 {task_name} 完成，最终清理 GPU 显存...")
    clear_gpu_memory()
    
    return results


def generate_execution_plan(model_names: List[str], tasks: List[str], subtasks: List[str], args) -> List[Dict]:
    """
    生成执行计划（按实际执行顺序）
    
    实际执行顺序：
    - 模型1: 数据处理 → 训练/评估/预测 (vd) → 训练/评估/预测 (os) → 训练/评估/预测 (dt)
    - 模型2: 数据处理 → 训练/评估/预测 (vd) → 训练/评估/预测 (os) → 训练/评估/预测 (dt)
    - ...
    
    Returns:
        执行计划列表，每个元素包含 {model, task, subtask, description}
    """
    task_names = {"vd": "vendor", "os": "os", "dt": "devicetype"}
    task_desc = {"vd": "厂商识别", "os": "操作系统识别", "dt": "设备类型识别"}
    subtask_desc = {"train": "训练", "evaluate": "评估", "predict": "预测"}
    
    plan = []
    
    # 按模型顺序生成计划（与实际执行顺序一致）
    for model_name in model_names:
        # 1. 先处理该模型的所有数据
        for task_type in tasks:
            plan.append({
                'model': model_name,
                'task': task_type,
                'subtask': 'data_process',
                'description': f"处理数据 - {task_desc[task_type]}"
            })
        
        # 2. 然后执行该模型的所有任务
        for task_type in tasks:
            for subtask in subtasks:
                plan.append({
                    'model': model_name,
                    'task': task_type,
                    'subtask': subtask,
                    'description': f"{subtask_desc[subtask]} - {task_desc[task_type]}"
                })
    
    return plan


def display_execution_plan(plan: List[Dict], args) -> bool:
    """
    显示执行计划并请求用户确认
    
    Returns:
        用户是否确认执行
    """
    print()
    print("=" * 70)
    print("执行计划")
    print("=" * 70)
    print()
    print(f"训练模式: {args.mode}")
    print(f"任务类型: {', '.join(args.tasks)}")
    print(f"子任务: {', '.join(args.subtasks)}")
    
    # 显示标签平衡配置
    if args.data_config:
        print(f"数据配置文件: {args.data_config}")
    else:
        if args.balance_ratio is not None:
            print(f"验证/测试集标签平衡: 最多标签 {args.balance_ratio*100:.0f}%")
        if args.train_vd is not None:
            print(f"vendor训练集标签平衡: 最多标签 {args.train_vd*100:.0f}%")
        if args.train_os is not None:
            print(f"os训练集标签平衡: 最多标签 {args.train_os*100:.0f}%")
        if args.train_dt is not None:
            print(f"devicetype训练集标签平衡: 最多标签 {args.train_dt*100:.0f}%")
    
    print()
    print(f"总计 {len(plan)} 个步骤:")
    print("-" * 70)
    
    # 按模型分组显示
    current_model = None
    step_num = 0
    for item in plan:
        if item['model'] != current_model:
            current_model = item['model']
            print(f"\n【模型: {current_model}】")
        step_num += 1
        print(f"  {step_num:2d}. {item['description']}")
    
    print()
    print("=" * 70)
    
    # 请求确认
    while True:
        response = input("\n是否继续执行? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            return True
        elif response in ['n', 'no', '否']:
            return False
        else:
            print("请输入 y 或 n")


def main():
    parser = argparse.ArgumentParser(
        description='模型专项训练组 - 执行指定模型的完整训练流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python full_run.py
  python full_run.py --tasks vd os
  python full_run.py --skip os dt              # 只执行 vd 任务组
  python full_run.py --skip dt                 # 跳过 devicetype，执行 vd 和 os
  python full_run.py --subtasks predict        # 只执行预测，跳过训练和评估
  python full_run.py --subtasks train evaluate # 只执行训练和评估，跳过预测
  python full_run.py --tasks vd --subtasks predict  # 只对 vendor 执行预测
  
  # 标签平衡示例（命令行参数）
  python full_run.py --balance-ratio 0.4       # 验证/测试集最多标签占40%
  python full_run.py --train-os 0.6 --train-dt 0.7 --train-vd 0.8  # 训练集标签平衡
  python full_run.py --balance-ratio 0.4 --train-os 0.6 --train-vd 0.8  # 同时平衡训练集和验证/测试集
  
  # 使用配置文件（推荐）
  python full_run.py --data-config configs/data_balance.yaml  # 使用配置文件定义平衡比例
  
  # 使用优化后的 V2 数据管道（推荐）
  python full_run.py --pipeline-v2                    # 按IP分组、脱敏（devicetype自动用非严格模式）
  python full_run.py --pipeline-v2 --no-strict        # 所有任务都用非严格模式
  python full_run.py --pipeline-v2 --models           # V2管道 + 多模型选择

执行流程:
  1. 显示执行计划，等待用户确认
  2. 重新处理所有数据（确保可复现，使用固定随机种子）
  3. 按顺序执行所有任务
  4. 生成完整日志文件

日志输出:
  logs/full_run_{timestamp}.log
        """
    )
    
    parser.add_argument('--mode', type=str, default='simple',
                        choices=['simple'],
                        help='训练模式 (default: simple)')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['vd', 'os', 'dt'],
                        choices=['vd', 'os', 'dt'],
                        help='要执行的任务列表 (default: vd os dt)')
    parser.add_argument('--skip', type=str, nargs='+',
                        default=[],
                        choices=['vd', 'os', 'dt'],
                        help='要跳过的任务组 (例如: --skip os dt)')
    parser.add_argument('--subtasks', type=str, nargs='+',
                        default=['train', 'evaluate', 'predict'],
                        choices=['train', 'evaluate', 'predict'],
                        help='要执行的子任务 (default: train evaluate predict)')
    parser.add_argument('--balance-ratio', type=float, default=None,
                        help='平衡验证集/测试集标签分布，指定最多标签的占比 (例如: 0.85 表示最多标签占85%%)')
    parser.add_argument('--train-vd', type=float, default=None,
                        help='vendor训练集最多标签占比 (例如: 0.8 表示最多标签占80%%)')
    parser.add_argument('--train-os', type=float, default=None,
                        help='os训练集最多标签占比 (例如: 0.6 表示最多标签占60%%)')
    parser.add_argument('--train-dt', type=float, default=None,
                        help='devicetype训练集最多标签占比 (例如: 0.7 表示最多标签占70%%)')
    parser.add_argument('--data-config', type=str, default=None,
                        help='数据平衡配置文件路径 (例如: configs/data_balance.yaml)')
    parser.add_argument('--pipeline-v2', action='store_true',
                        help='使用优化后的数据处理管道 V2（按IP分组、脱敏、严格筛选）')
    parser.add_argument('--no-strict', action='store_true',
                        help='V2管道：所有任务使用非严格模式（devicetype 默认已是非严格模式）')
    parser.add_argument('--models', action='store_true',
                        help='启用多模型选择模式（可以选择多个模型依次执行）')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='跳过确认，直接执行')
    
    args = parser.parse_args()
    
    # 从任务列表中移除要跳过的任务
    if args.skip:
        args.tasks = [t for t in args.tasks if t not in args.skip]
        if not args.tasks:
            print("错误: 所有任务都被跳过了")
            sys.exit(1)
    
    # 验证子任务列表
    if not args.subtasks:
        print("错误: 至少需要指定一个子任务")
        sys.exit(1)
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print()
    print("=" * 60)
    print("    模型专项训练组")
    print("=" * 60)
    print()
    
    # 获取可用模型列表
    available_models = get_available_models()
    
    if not available_models:
        print("错误: models/ 目录下没有找到可用模型")
        print("请将模型文件夹放入 models/ 目录")
        sys.exit(1)
    
    # 根据 --models 参数选择单个或多个模型
    if args.models:
        # 多模型模式
        model_names = select_models_interactive(available_models, "选择要训练的模型")
        if not model_names:
            print("已取消")
            sys.exit(0)
    else:
        # 单模型模式
        model_name = select_model_interactive(available_models, "选择要训练的模型")
        if not model_name:
            print("已取消")
            sys.exit(0)
        model_names = [model_name]
    
    # 验证所有模型的配置文件
    valid_models = []
    for model_name in model_names:
        try:
            config_path = get_config_path(model_name)
            valid_models.append(model_name)
        except FileNotFoundError as e:
            print(f"警告: {model_name} - {e}")
            print(f"  跳过此模型")
    
    if not valid_models:
        print("\n错误: 没有有效的模型配置")
        print("\n可用配置文件:")
        configs_dir = "configs"
        if os.path.exists(configs_dir):
            for f in sorted(os.listdir(configs_dir)):
                if f.endswith('.yaml'):
                    print(f"  - {f}")
        sys.exit(1)
    
    model_names = valid_models
    
    # 生成执行计划
    execution_plan = generate_execution_plan(model_names, args.tasks, args.subtasks, args)
    
    # 显示执行计划并请求确认
    if not args.yes:
        if not display_execution_plan(execution_plan, args):
            print("\n已取消执行")
            sys.exit(0)
    
    # 创建统一的日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"full_run_{timestamp}.log")
    
    # 配置统一日志
    main_logger = logging.getLogger("full_run_main")
    main_logger.setLevel(logging.INFO)
    main_logger.handlers.clear()
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    main_logger.addHandler(file_handler)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    main_logger.addHandler(console_handler)
    
    main_logger.info("=" * 70)
    main_logger.info("模型专项训练组启动")
    main_logger.info("=" * 70)
    main_logger.info(f"日志文件: {log_file}")
    main_logger.info(f"模型列表: {', '.join(model_names)}")
    main_logger.info(f"任务列表: {', '.join(args.tasks)}")
    main_logger.info(f"子任务: {', '.join(args.subtasks)}")
    if args.data_config:
        main_logger.info(f"数据配置文件: {args.data_config}")
    else:
        if args.balance_ratio is not None:
            main_logger.info(f"验证/测试集标签平衡: {args.balance_ratio*100:.0f}%")
        if args.train_vd is not None:
            main_logger.info(f"vendor训练集标签平衡: {args.train_vd*100:.0f}%")
        if args.train_os is not None:
            main_logger.info(f"os训练集标签平衡: {args.train_os*100:.0f}%")
        if args.train_dt is not None:
            main_logger.info(f"devicetype训练集标签平衡: {args.train_dt*100:.0f}%")
    main_logger.info(f"总步骤数: {len(execution_plan)}")
    main_logger.info("=" * 70)
    
    # 开始执行
    start_time = datetime.now()
    
    # 记录每个模型的执行结果
    model_results = {}
    
    # 执行所有选中的模型
    for model_idx, model_name in enumerate(model_names, 1):
        main_logger.info("")
        main_logger.info("=" * 70)
        main_logger.info(f"开始处理模型 [{model_idx}/{len(model_names)}]: {model_name}")
        main_logger.info("=" * 70)
        
        config_path = get_config_path(model_name)
        
        # 执行单个模型的完整流程
        success = run_single_model(model_name, config_path, args, main_logger)
        model_results[model_name] = success
    
    # 所有模型执行完成
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 统计结果
    total_models = len(model_names)
    success_models = sum(1 for s in model_results.values() if s)
    failed_models = total_models - success_models
    
    main_logger.info("")
    main_logger.info("=" * 70)
    main_logger.info("所有任务执行完成")
    main_logger.info("=" * 70)
    main_logger.info(f"执行模型数: {total_models}")
    main_logger.info(f"成功: {success_models}, 失败: {failed_models}")
    main_logger.info(f"总耗时: {duration}")
    main_logger.info(f"日志文件: {log_file}")
    main_logger.info("")
    
    # 显示每个模型的结果
    for model_name, success in model_results.items():
        status = "✓" if success else "✗"
        main_logger.info(f"  {status} {model_name}")
    
    main_logger.info("=" * 70)
    
    # 返回码
    if failed_models == 0:
        main_logger.info("✓ 所有模型执行成功！")
        sys.exit(0)
    else:
        main_logger.warning(f"✗ {failed_models} 个模型执行失败")
        sys.exit(1)


def run_single_model(model_name: str, config_path: str, args, main_logger=None):
    """执行单个模型的完整训练流程"""
    
    # 使用传入的主日志或创建新日志
    if main_logger is None:
        logger = setup_logging(model_name)
    else:
        logger = main_logger
    
    # 开始时间
    start_time = datetime.now()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"模型: {model_name}")
    logger.info("=" * 70)
    logger.info(f"配置: {config_path}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"任务: {', '.join(args.tasks)}")
    if args.skip:
        logger.info(f"跳过: {', '.join(args.skip)}")
    logger.info(f"子任务: {', '.join(args.subtasks)}")
    logger.info(f"数据处理: 强制重新处理（使用固定随机种子确保可复现）")
    if hasattr(args, 'data_config') and args.data_config:
        logger.info(f"数据配置文件: {args.data_config}")
    else:
        if args.balance_ratio is not None:
            logger.info(f"验证/测试集标签平衡: 最多标签占比 {args.balance_ratio*100:.0f}%")
        if args.train_vd is not None:
            logger.info(f"vendor训练集标签平衡: 最多标签占比 {args.train_vd*100:.0f}%")
        if args.train_os is not None:
            logger.info(f"os训练集标签平衡: 最多标签占比 {args.train_os*100:.0f}%")
        if args.train_dt is not None:
            logger.info(f"devicetype训练集标签平衡: 最多标签占比 {args.train_dt*100:.0f}%")
    logger.info(f"重试机制: 失败后等待 {RETRY_WAIT_SECONDS}秒，最多重试 {MAX_RETRIES} 次")
    logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # 设置环境
    setup_environment(model_name)
    
    # 强制重新处理所有数据（确保可复现）
    logger.info("\n" + "=" * 70)
    logger.info("数据处理阶段")
    logger.info("=" * 70)
    
    task_type_names = {"vd": "vendor", "os": "os", "dt": "devicetype"}
    
    # 检查是否使用 V2 管道
    use_pipeline_v2 = hasattr(args, 'pipeline_v2') and args.pipeline_v2
    
    if use_pipeline_v2:
        # 使用优化后的 V2 管道
        from training.data_pipeline_v2 import process_pipeline, PipelineConfig
        
        for task_type in args.tasks:
            task_name = task_type_names[task_type]
            logger.info(f"\n处理 {task_name} 数据 (V2 管道)...")
            
            # 数据目录 - 按模型名隔离
            data_dir = os.path.join('./data', model_name, task_name)
            input_path = f'./input/{task_name}_model_train.jsonl'
            
            if not os.path.exists(input_path):
                logger.warning(f"  输入数据不存在: {input_path}，跳过")
                continue
            
            # 配置严格模式
            # devicetype 默认非严格模式（因为 true_label 中 router 占比过高）
            # vendor 和 os 默认严格模式
            if hasattr(args, 'no_strict') and args.no_strict:
                # 用户明确指定非严格模式
                strict_mode = False
            elif task_name == 'devicetype':
                # devicetype 默认非严格模式
                strict_mode = False
                logger.info(f"  [自动] devicetype 使用非严格模式（允许弱标注）")
            else:
                # vendor/os 默认严格模式
                strict_mode = True
            
            pipeline_config = PipelineConfig(
                task_type=task_name,
                strict_mode=strict_mode,
                seed=42
            )
            
            try:
                result = process_pipeline(
                    input_path=input_path,
                    output_dir=data_dir,
                    task_type=task_name,
                    config=pipeline_config
                )
                logger.info(f"  ✓ {task_name} V2 数据处理完成")
                logger.info(f"    训练集: {result['train_size']}, 验证集: {result['val_size']}, 测试集: {result['test_size']}")
                
                # V2 管道生成的文件名与旧版不同，需要重命名以兼容训练代码
                # train.jsonl -> simple_train.jsonl
                # val.jsonl -> simple_valid.jsonl
                import shutil
                train_src = os.path.join(data_dir, 'train.jsonl')
                train_dst = os.path.join(data_dir, 'simple_train.jsonl')
                val_src = os.path.join(data_dir, 'val.jsonl')
                val_dst = os.path.join(data_dir, 'simple_valid.jsonl')
                
                if os.path.exists(train_src):
                    shutil.copy(train_src, train_dst)
                if os.path.exists(val_src):
                    shutil.copy(val_src, val_dst)
                
            except Exception as e:
                logger.error(f"  ✗ {task_name} V2 数据处理失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
    else:
        # 使用原有管道
        from training.data_processor import DataProcessor, load_prompt
        from training.config import DataConfig
        
        for task_type in args.tasks:
            task_name = task_type_names[task_type]
            logger.info(f"\n处理 {task_name} 数据...")
            
            # 设置模型类型和加载 prompt
            load_prompt(prompt_file='./prompt/student.json', prompt_id=task_name)
            
            # 数据目录 - 按模型名隔离
            data_dir = os.path.join('./data', model_name, task_name)
            os.makedirs(data_dir, exist_ok=True)
            
            # 输入数据路径
            input_path = f'./input/{task_name}_model_train.jsonl'
            
            if not os.path.exists(input_path):
                logger.warning(f"  输入数据不存在: {input_path}，跳过")
                continue
            
            # 删除旧的处理数据
            for filename in ['train.jsonl', 'valid.jsonl', 'test.jsonl', 
                            'simple_train.jsonl', 'simple_valid.jsonl']:
                file_path = os.path.join(data_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"  删除旧数据: {file_path}")
            
            # 创建 DataConfig
            data_cfg = DataConfig(
                input_path=input_path,
                output_dir=data_dir,
                train_ratio=0.8,
                valid_ratio=0.1,
                test_ratio=0.1,
                augment_ratio=0.0  # 简化训练默认不增强
            )
            
            # 处理数据（DataProcessor 内部会使用固定随机种子）
            try:
                processor = DataProcessor(data_cfg)
                processor.process(
                    input_path=input_path,
                    output_dir=data_dir,
                    include_dpo=False,  # 简化训练不需要 DPO
                    augment_ratio=0.0
                )
                logger.info(f"  ✓ {task_name} 数据处理完成")
                
                # 转换为 simple 格式（只输出标签）
                from training.simple_classifier import convert_to_simple_format
                
                train_file = os.path.join(data_dir, 'train.jsonl')
                valid_file = os.path.join(data_dir, 'valid.jsonl')
                simple_train = os.path.join(data_dir, 'simple_train.jsonl')
                simple_valid = os.path.join(data_dir, 'simple_valid.jsonl')
                
                if os.path.exists(train_file):
                    logger.info(f"  转换训练数据为 simple 格式...")
                    convert_to_simple_format(train_file, simple_train, task_name)
                    logger.info(f"  ✓ {simple_train}")
                
                if os.path.exists(valid_file):
                    logger.info(f"  转换验证数据为 simple 格式...")
                    convert_to_simple_format(valid_file, simple_valid, task_name)
                    logger.info(f"  ✓ {simple_valid}")
                
                # 获取该任务的平衡比例
                train_balance_ratio = None
                valid_balance_ratio = None
                test_balance_ratio = None
                
                # 优先使用配置文件
                if hasattr(args, 'data_config') and args.data_config:
                    try:
                        data_config = load_data_balance_config(args.data_config)
                        ratios = get_balance_ratios_from_config(data_config, task_name)
                        train_balance_ratio = ratios.get('train')
                        valid_balance_ratio = ratios.get('valid')
                        test_balance_ratio = ratios.get('test')
                        logger.info(f"  从配置文件加载平衡比例: train={train_balance_ratio}, valid={valid_balance_ratio}, test={test_balance_ratio}")
                    except Exception as e:
                        logger.warning(f"  加载数据配置文件失败: {e}，使用命令行参数")
                
                # 如果没有配置文件或加载失败，使用命令行参数
                if train_balance_ratio is None:
                    train_ratio_map = {
                        'vendor': args.train_vd,
                        'os': args.train_os,
                        'devicetype': args.train_dt
                    }
                    train_balance_ratio = train_ratio_map.get(task_name)
                
                if valid_balance_ratio is None:
                    valid_balance_ratio = args.balance_ratio
                if test_balance_ratio is None:
                    test_balance_ratio = args.balance_ratio
                
                # 平衡数据集标签分布
                need_balance = (train_balance_ratio is not None or 
                              valid_balance_ratio is not None or 
                              test_balance_ratio is not None)
                
                if need_balance:
                    from training.data_processor import balance_data_files_v2
                    
                    if train_balance_ratio is not None:
                        logger.info(f"  平衡训练集标签分布 (最多标签占比: {train_balance_ratio*100:.0f}%)...")
                    if valid_balance_ratio is not None:
                        logger.info(f"  平衡验证集标签分布 (最多标签占比: {valid_balance_ratio*100:.0f}%)...")
                    if test_balance_ratio is not None:
                        logger.info(f"  平衡测试集标签分布 (最多标签占比: {test_balance_ratio*100:.0f}%)...")
                    
                    balance_data_files_v2(
                        data_dir=data_dir,
                        task_type=task_name,
                        train_ratio=train_balance_ratio,
                        valid_ratio=valid_balance_ratio,
                        test_ratio=test_balance_ratio,
                        seed=42
                    )
                    logger.info(f"  ✓ 标签平衡完成")
                    
            except Exception as e:
                logger.error(f"  ✗ {task_name} 数据处理失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 70)
    logger.info("数据处理完成，开始训练流程")
    logger.info("=" * 70 + "\n")
    
    # 执行所有任务
    all_results = {}
    task_names = {"vd": "厂商识别", "os": "操作系统识别", "dt": "设备类型识别"}
    
    for task_type in args.tasks:
        results = run_task_pipeline(model_name, task_type, logger, args.mode, args.subtasks)
        all_results[task_type] = results
    
    # 结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 打印总结
    logger.info("\n" + "=" * 70)
    logger.info("执行总结")
    logger.info("=" * 70)
    logger.info(f"模型: {model_name}")
    logger.info(f"总耗时: {duration}")
    logger.info("")
    
    # 统计
    total_success = 0
    total_tasks = 0
    
    for task_type, results in all_results.items():
        task_name = task_names.get(task_type, task_type)
        
        # 只统计实际执行的子任务
        train_status = "✓" if results['train'] is True else ("✗" if results['train'] is False else "-")
        eval_status = "✓" if results['evaluate'] is True else ("✗" if results['evaluate'] is False else "-")
        pred_status = "✓" if results['predict'] is True else ("✗" if results['predict'] is False else "-")
        
        logger.info(f"{task_name}:")
        logger.info(f"  训练: {train_status}  评估: {eval_status}  预测: {pred_status}")
        
        # 统计实际执行的任务数
        for subtask in ['train', 'evaluate', 'predict']:
            if results[subtask] is not None:  # 不是跳过的
                total_tasks += 1
                if results[subtask]:  # 成功
                    total_success += 1
    
    logger.info("")
    logger.info(f"成功率: {total_success}/{total_tasks} ({100*total_success/total_tasks:.1f}%)")
    logger.info("=" * 70)
    
    # 返回成功状态，不在这里退出
    success = (total_success == total_tasks)
    if success:
        logger.info("✓ 模型所有任务执行成功")
    else:
        logger.warning(f"✗ 模型部分任务失败 ({total_tasks - total_success} 个)")
    
    return success


if __name__ == "__main__":
    main()
