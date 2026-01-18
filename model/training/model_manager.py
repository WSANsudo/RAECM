"""
模型管理模块
Model management module for downloading and switching models
"""

import os
import logging
from typing import Optional, Dict, List
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi

logger = logging.getLogger(__name__)

# 默认模型目录
DEFAULT_MODEL_DIR = "./models"

# 支持的模型列表
SUPPORTED_MODELS = {
    # Qwen2.5 系列
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    
    # Qwen3 系列
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    
    # Microsoft Phi 系列
    "phi-4": "microsoft/phi-4",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
}


class ModelManager:
    """模型管理器，负责模型的下载、切换和管理"""
    
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型存储目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def list_local_models(self) -> List[str]:
        """
        列出本地已下载的模型
        
        Returns:
            本地模型名称列表
        """
        models = []
        if self.model_dir.exists():
            for item in self.model_dir.iterdir():
                if item.is_dir():
                    # 检查是否是有效的模型目录（包含config.json）
                    if (item / "config.json").exists():
                        models.append(item.name)
        return sorted(models)
    
    def get_local_model_path(self, model_name: str) -> Optional[str]:
        """
        获取本地模型路径
        
        Args:
            model_name: 模型名称（简称或完整名称）
            
        Returns:
            本地模型路径，如果不存在返回 None
        """
        # 1. 直接检查是否是完整路径
        if os.path.exists(model_name):
            config_path = os.path.join(model_name, "config.json")
            if os.path.exists(config_path):
                return model_name
        
        # 2. 检查 models 目录下的路径
        model_path = self.model_dir / model_name
        if model_path.exists() and (model_path / "config.json").exists():
            return str(model_path)
        
        # 3. 如果路径包含 models/，去掉前缀再检查
        if model_name.startswith("models/"):
            relative_path = model_name[7:]  # 去掉 "models/"
            model_path = self.model_dir / relative_path
            if model_path.exists() and (model_path / "config.json").exists():
                return str(model_path)
        
        # 4. 尝试从简称映射
        if model_name.lower() in SUPPORTED_MODELS:
            hub_name = SUPPORTED_MODELS[model_name.lower()]
            # 转换为本地目录名
            local_name = hub_name.replace("/", "-")
            model_path = self.model_dir / local_name
            if model_path.exists() and (model_path / "config.json").exists():
                return str(model_path)
            
            # 也检查保留 / 的路径
            model_path = self.model_dir / hub_name
            if model_path.exists() and (model_path / "config.json").exists():
                return str(model_path)
        
        return None
    
    def resolve_model_path(self, model_name_or_path: str) -> str:
        """
        解析模型路径，优先使用本地模型，否则返回 HuggingFace Hub 名称
        
        Args:
            model_name_or_path: 模型名称或路径
            
        Returns:
            解析后的模型路径
        """
        # 1. 如果是绝对路径且存在，直接返回
        if os.path.isabs(model_name_or_path) and os.path.exists(model_name_or_path):
            return model_name_or_path
        
        # 2. 检查本地模型目录
        local_path = self.get_local_model_path(model_name_or_path)
        if local_path:
            logger.info(f"使用本地模型: {local_path}")
            return local_path
        
        # 3. 如果是简称，转换为 HuggingFace Hub 名称
        if model_name_or_path.lower() in SUPPORTED_MODELS:
            hub_name = SUPPORTED_MODELS[model_name_or_path.lower()]
            logger.info(f"模型 '{model_name_or_path}' 映射到 '{hub_name}'")
            return hub_name
        
        # 4. 假设是 HuggingFace Hub 名称
        return model_name_or_path
    
    def download_model(
        self,
        model_name: str,
        force: bool = False
    ) -> str:
        """
        下载模型到本地
        
        Args:
            model_name: 模型名称（简称或 HuggingFace Hub 名称）
            force: 是否强制重新下载
            
        Returns:
            本地模型路径
        """
        # 解析 Hub 名称
        if model_name.lower() in SUPPORTED_MODELS:
            hub_name = SUPPORTED_MODELS[model_name.lower()]
        else:
            hub_name = model_name
        
        # 本地目录名
        local_name = hub_name.replace("/", "-")
        local_path = self.model_dir / local_name
        
        # 检查是否已存在
        if local_path.exists() and not force:
            if (local_path / "config.json").exists():
                logger.info(f"模型已存在: {local_path}")
                return str(local_path)
        
        logger.info(f"开始下载模型: {hub_name}")
        logger.info(f"保存到: {local_path}")
        
        try:
            snapshot_download(
                repo_id=hub_name,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"模型下载完成: {local_path}")
            return str(local_path)
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            raise
    
    def ensure_model(self, model_name_or_path: str) -> str:
        """
        确保模型可用（本地存在或自动下载）
        
        Args:
            model_name_or_path: 模型名称或路径
            
        Returns:
            可用的模型路径
        """
        # 1. 如果已经是本地路径且存在，直接返回
        if os.path.exists(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
            if os.path.exists(config_path):
                logger.info(f"使用本地模型: {model_name_or_path}")
                return model_name_or_path
        
        # 2. 检查本地模型目录
        local_path = self.get_local_model_path(model_name_or_path)
        if local_path:
            logger.info(f"使用本地模型: {local_path}")
            return local_path
        
        # 3. 如果是 HuggingFace Hub 名称，尝试下载
        if "/" in model_name_or_path and not model_name_or_path.startswith("models/"):
            logger.info(f"本地未找到模型，尝试下载: {model_name_or_path}")
            return self.download_model(model_name_or_path)
        
        if model_name_or_path.lower() in SUPPORTED_MODELS:
            logger.info(f"本地未找到模型，尝试下载: {model_name_or_path}")
            return self.download_model(model_name_or_path)
        
        # 4. 返回原始路径（可能是 Hub 名称，让 transformers 处理）
        return model_name_or_path
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典
        """
        info = {
            "name": model_name,
            "local_path": None,
            "hub_name": None,
            "is_local": False,
            "is_supported": False
        }
        
        # 检查本地
        local_path = self.get_local_model_path(model_name)
        if local_path:
            info["local_path"] = local_path
            info["is_local"] = True
        
        # 检查是否支持
        if model_name.lower() in SUPPORTED_MODELS:
            info["hub_name"] = SUPPORTED_MODELS[model_name.lower()]
            info["is_supported"] = True
        elif "/" in model_name:
            info["hub_name"] = model_name
            info["is_supported"] = True
        
        return info


def list_available_models() -> None:
    """打印可用模型列表"""
    manager = ModelManager()
    local_models = manager.list_local_models()
    
    print("\n" + "=" * 60)
    print("可用模型列表")
    print("=" * 60)
    
    print("\n【支持的模型】")
    print("-" * 40)
    for short_name, hub_name in SUPPORTED_MODELS.items():
        local_path = manager.get_local_model_path(short_name)
        status = "✓ 已下载" if local_path else "○ 未下载"
        print(f"  {short_name:20} {status}")
        print(f"    └─ {hub_name}")
    
    if local_models:
        print("\n【本地模型】")
        print("-" * 40)
        for model in local_models:
            print(f"  ✓ {model}")
    
    print("\n" + "=" * 60)


def download_model_cli(model_name: str, force: bool = False) -> None:
    """命令行下载模型"""
    manager = ModelManager()
    
    print(f"\n下载模型: {model_name}")
    print("-" * 40)
    
    try:
        path = manager.download_model(model_name, force=force)
        print(f"\n✓ 下载完成: {path}")
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型管理工具")
    subparsers = parser.add_subparsers(dest="command")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出可用模型")
    
    # download 命令
    download_parser = subparsers.add_parser("download", help="下载模型")
    download_parser.add_argument("model", help="模型名称")
    download_parser.add_argument("--force", action="store_true", help="强制重新下载")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_available_models()
    elif args.command == "download":
        download_model_cli(args.model, args.force)
    else:
        parser.print_help()
