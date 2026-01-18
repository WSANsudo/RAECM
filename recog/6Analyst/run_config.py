"""
运行配置模块

支持从 JSON 配置文件读取所有运行参数，包括：
- 输入/输出路径
- 模型配置
- 处理参数（线程数、批次大小等）
- 运行模式

exp 模块可以通过生成配置文件来调用主项目的完整流程。
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any


# 全局当前运行配置
_current_run_config: Optional[Dict[str, Any]] = None


def set_current_run_config(config: Dict[str, Any]) -> None:
    """设置当前运行配置"""
    global _current_run_config
    _current_run_config = config


def get_current_run_config() -> Dict[str, Any]:
    """获取当前运行配置"""
    return _current_run_config or {}


@dataclass
class RunConfig:
    """运行配置类"""
    
    # ===== 路径配置 =====
    input_path: Optional[str] = None          # 输入路径（文件或目录）
    cleaned_data_path: Optional[str] = None   # 清洗后数据路径
    product_output_path: Optional[str] = None # 产品分析输出路径
    merged_output_path: Optional[str] = None  # 合并结果路径
    check_output_path: Optional[str] = None   # 校验详情路径
    final_output_path: Optional[str] = None   # 最终结果路径
    run_state_path: Optional[str] = None      # 运行状态路径
    log_dir: Optional[str] = None             # 日志目录
    
    # ===== 模型配置 =====
    model: Optional[str] = None               # 全局模型（所有Agent使用）
    product_model: Optional[str] = None       # 产品Agent模型
    check_model: Optional[str] = None         # 校验Agent模型
    
    # ===== 处理参数 =====
    max_records: Optional[int] = None         # 最大处理条数
    batch_size: int = 3                       # 批次大小
    num_threads: int = 1                      # 线程数
    speed_level: str = '6'                    # 速度等级
    
    # ===== 运行模式 =====
    mode: str = 'all'                         # 运行模式: all/clean-only/product-only/check-only
    skip_check: bool = False                  # 跳过校验
    skip_clean: bool = False                  # 跳过数据清洗（当输入已是清洗后的数据时使用）
    restart: bool = False                     # 重新开始（禁用断点续传）
    test_mode: bool = False                   # 测试模式
    debug: bool = False                       # 调试模式
    
    # ===== 提示词配置 =====
    product_prompt: str = 'default'           # 产品Agent提示词ID
    check_prompt: str = 'default'             # 校验Agent提示词ID
    
    # ===== 日志配置 =====
    log_level: str = 'warning'                # 日志级别
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """保存配置到JSON文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RunConfig':
        """从JSON文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunConfig':
        """从字典创建配置"""
        # 过滤掉不存在的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def get_effective_model(self, agent_type: str) -> Optional[str]:
        """
        获取指定Agent的有效模型
        
        Args:
            agent_type: 'product' 或 'check'
            
        Returns:
            模型名称
        """
        if agent_type == 'product':
            return self.product_model or self.model
        elif agent_type == 'check':
            return self.check_model or self.model
        return self.model


def apply_config_to_globals(config: RunConfig) -> Dict[str, Any]:
    """
    将配置应用到主项目的全局变量
    
    Args:
        config: 运行配置
        
    Returns:
        原始配置（用于恢复）
    """
    from . import config as main_config
    from . import run as main_run
    
    # 保存原始配置
    orig = {
        'CLEANED_DATA_PATH': main_config.CLEANED_DATA_PATH,
        'PRODUCT_OUTPUT_PATH': main_config.PRODUCT_OUTPUT_PATH,
        'MERGED_OUTPUT_PATH': main_config.MERGED_OUTPUT_PATH,
        'CHECK_OUTPUT_PATH': main_config.CHECK_OUTPUT_PATH,
        'FINAL_OUTPUT_PATH': main_config.FINAL_OUTPUT_PATH,
        'RUN_STATE_PATH': main_config.RUN_STATE_PATH,
        'LOG_DIR': main_config.LOG_DIR,
        'MODEL_NAME': main_config.MODEL_NAME,
        'BATCH_SIZE': main_config.BATCH_SIZE,
        'DEBUG_MODE': main_config.DEBUG_MODE,
        # run模块的全局变量
        '_cleaned_data_path': main_run._cleaned_data_path,
        '_product_output_path': main_run._product_output_path,
        '_merged_output_path': main_run._merged_output_path,
        '_check_output_path': main_run._check_output_path,
        '_final_output_path': main_run._final_output_path,
        '_run_state_path': main_run._run_state_path,
        '_log_dir': main_run._log_dir,
    }
    
    # 应用新配置（仅当配置值不为None时）
    if config.cleaned_data_path:
        main_config.CLEANED_DATA_PATH = config.cleaned_data_path
        main_run._cleaned_data_path = config.cleaned_data_path
    
    if config.product_output_path:
        main_config.PRODUCT_OUTPUT_PATH = config.product_output_path
        main_run._product_output_path = config.product_output_path
    
    if config.merged_output_path:
        main_config.MERGED_OUTPUT_PATH = config.merged_output_path
        main_run._merged_output_path = config.merged_output_path
    
    if config.check_output_path:
        main_config.CHECK_OUTPUT_PATH = config.check_output_path
        main_run._check_output_path = config.check_output_path
    
    if config.final_output_path:
        main_config.FINAL_OUTPUT_PATH = config.final_output_path
        main_run._final_output_path = config.final_output_path
    
    if config.run_state_path:
        main_config.RUN_STATE_PATH = config.run_state_path
        main_run._run_state_path = config.run_state_path
    
    if config.log_dir:
        main_config.LOG_DIR = config.log_dir
        main_run._log_dir = config.log_dir
    
    # 设置提示词ID
    from .product_analyst import ProductAnalyst
    from .check_analyst import CheckAnalyst
    ProductAnalyst.set_prompt_id(config.product_prompt)
    CheckAnalyst.set_prompt_id(config.check_prompt)
    
    if config.model:
        main_config.MODEL_NAME = config.model
    
    if config.batch_size:
        main_config.BATCH_SIZE = config.batch_size
    
    if config.debug:
        main_config.DEBUG_MODE = config.debug
    
    # 设置速度等级
    main_run.set_speed_level(config.speed_level)
    
    return orig


def restore_config_from_backup(orig: Dict[str, Any]) -> None:
    """
    从备份恢复原始配置
    
    Args:
        orig: apply_config_to_globals 返回的原始配置
    """
    from . import config as main_config
    from . import run as main_run
    
    main_config.CLEANED_DATA_PATH = orig['CLEANED_DATA_PATH']
    main_config.PRODUCT_OUTPUT_PATH = orig['PRODUCT_OUTPUT_PATH']
    main_config.MERGED_OUTPUT_PATH = orig['MERGED_OUTPUT_PATH']
    main_config.CHECK_OUTPUT_PATH = orig['CHECK_OUTPUT_PATH']
    main_config.FINAL_OUTPUT_PATH = orig['FINAL_OUTPUT_PATH']
    main_config.RUN_STATE_PATH = orig['RUN_STATE_PATH']
    main_config.LOG_DIR = orig['LOG_DIR']
    main_config.MODEL_NAME = orig['MODEL_NAME']
    main_config.BATCH_SIZE = orig['BATCH_SIZE']
    main_config.DEBUG_MODE = orig['DEBUG_MODE']
    
    main_run._cleaned_data_path = orig['_cleaned_data_path']
    main_run._product_output_path = orig['_product_output_path']
    main_run._merged_output_path = orig['_merged_output_path']
    main_run._check_output_path = orig['_check_output_path']
    main_run._final_output_path = orig['_final_output_path']
    main_run._run_state_path = orig['_run_state_path']
    main_run._log_dir = orig['_log_dir']


def run_with_config(config: RunConfig) -> Dict[str, Any]:
    """
    使用指定配置运行主项目流程
    
    这是 exp 模块调用主项目的主要接口。
    
    Args:
        config: 运行配置
        
    Returns:
        运行统计信息
    """
    from . import run as main_run
    
    # 应用配置
    orig = apply_config_to_globals(config)
    
    try:
        # 清除运行状态（如果是重启模式）
        if config.restart:
            main_run.clear_run_state()
        
        # 根据模式执行不同流程
        if config.mode == 'clean-only':
            # 仅清洗
            stats = main_run.run_cleaner(config.input_path, config.max_records)
            return {'cleaner': stats}
        
        elif config.mode == 'product-only':
            # 仅产品分析
            stats = main_run.run_product_analyst(
                config.max_records, 
                config.get_effective_model('product')
            )
            return {'product': stats}
        
        elif config.mode == 'check-only':
            # 仅校验
            from .check_analyst import CheckAnalyst
            analyst = CheckAnalyst(
                main_run._merged_output_path,
                main_run._check_output_path,
                main_run._final_output_path,
                config.get_effective_model('check')
            )
            stats, eval_stats = analyst.run(config.max_records)
            return {'check': stats, 'evaluation': eval_stats}
        
        else:
            # 完整流程
            if config.num_threads > 1:
                # 多线程模式
                from .multi_thread_runner import run_multi_thread_pipeline
                stats = run_multi_thread_pipeline(
                    input_path=config.input_path,
                    max_records=config.max_records,
                    num_workers=config.num_threads,
                    speed_level=config.speed_level,
                    skip_check=config.skip_check,
                    skip_clean=config.skip_clean,
                    restart=config.restart,
                    product_model=config.get_effective_model('product'),
                    check_model=config.get_effective_model('check')
                )
            else:
                # 单线程模式
                # 先执行数据清洗（除非跳过）
                if not config.skip_clean:
                    main_run.run_cleaner(config.input_path, config.max_records)
                
                stats = main_run.run_pipeline(
                    max_records=config.max_records,
                    skip_check=config.skip_check,
                    product_model=config.get_effective_model('product'),
                    check_model=config.get_effective_model('check')
                )
            return {'pipeline': stats}
    
    finally:
        # 恢复原始配置
        restore_config_from_backup(orig)


def run_with_config_file(config_path: str) -> Dict[str, Any]:
    """
    从配置文件运行主项目流程
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        运行统计信息
    """
    config = RunConfig.load(config_path)
    return run_with_config(config)


def create_config_for_exp(
    group_id: int,
    output_dir: str,
    input_file: str,
    model_name: str,
    num_threads: int = 4,
    max_records: Optional[int] = None,
    restart: bool = False
) -> RunConfig:
    """
    为实验组创建运行配置
    
    Args:
        group_id: 实验组ID
        output_dir: 输出目录
        input_file: 输入文件路径
        model_name: 使用的模型
        num_threads: 线程数
        max_records: 最大处理条数
        restart: 是否重新开始
        
    Returns:
        运行配置
    """
    output_path = Path(output_dir)
    
    return RunConfig(
        # 路径配置
        input_path=input_file,
        cleaned_data_path=input_file,  # exp模块已经清洗过数据
        product_output_path=str(output_path / "product_analysis.jsonl"),
        merged_output_path=str(output_path / "merged_analysis.jsonl"),
        check_output_path=str(output_path / "check_details.jsonl"),
        final_output_path=str(output_path / "model_results.jsonl"),
        run_state_path=str(output_path / "run_state.json"),
        log_dir=str(output_path / "logs"),
        
        # 模型配置
        model=model_name,
        
        # 处理参数
        max_records=max_records,
        num_threads=num_threads,
        speed_level='6',
        
        # 运行模式
        mode='all',
        skip_clean=True,  # exp模块数据已经清洗过
        restart=restart,
    )
