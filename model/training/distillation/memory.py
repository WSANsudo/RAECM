"""
显存监控模块
"""

import gc
import logging
from typing import Dict
from collections import defaultdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """显存监控工具类"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.is_cuda = self.device.type == 'cuda'
        self.peak_memory = 0
        self.checkpoints: Dict[str, float] = {}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取当前显存统计信息"""
        if not self.is_cuda:
            return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        free = total - reserved
        
        self.peak_memory = max(self.peak_memory, allocated)
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'peak': self.peak_memory
        }
    
    def get_memory_str(self) -> str:
        """获取显存使用的简短字符串"""
        stats = self.get_memory_stats()
        if not self.is_cuda:
            return "CPU"
        return f"{stats['allocated']:.2f}GB/{stats['total']:.1f}GB"
    
    def checkpoint(self, name: str) -> float:
        """记录检查点的显存使用"""
        if not self.is_cuda:
            return 0
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        self.checkpoints[name] = allocated
        return allocated
    
    def log_memory_status(self, prefix: str = ""):
        """输出详细的显存状态"""
        if not self.is_cuda:
            logger.info(f"{prefix}运行在CPU模式")
            return
        
        stats = self.get_memory_stats()
        logger.info(f"{prefix}显存状态:")
        logger.info(f"  已分配: {stats['allocated']:.3f} GB")
        logger.info(f"  已预留: {stats['reserved']:.3f} GB")
        logger.info(f"  可用: {stats['free']:.3f} GB")
        logger.info(f"  总量: {stats['total']:.1f} GB")
        logger.info(f"  峰值: {stats['peak']:.3f} GB")
    
    def log_tensor_memory(self, name: str, tensor: torch.Tensor):
        """输出张量的显存占用"""
        if tensor is None:
            return
        size_mb = tensor.element_size() * tensor.nelement() / 1024**2
        logger.info(f"  {name}: {list(tensor.shape)} -> {size_mb:.2f} MB")
    
    def log_model_memory(self, model: nn.Module, name: str = "Model"):
        """输出模型各部分的显存占用"""
        if not self.is_cuda:
            return
        
        total_params = 0
        total_size = 0
        layer_sizes = defaultdict(float)
        
        for param_name, param in model.named_parameters():
            param_size = param.element_size() * param.nelement() / 1024**2
            total_params += param.nelement()
            total_size += param_size
            
            # 按层分组
            layer_name = param_name.split('.')[0]
            layer_sizes[layer_name] += param_size
        
        logger.info(f"{name} 显存占用:")
        logger.info(f"  总参数量: {total_params / 1e6:.2f}M")
        logger.info(f"  参数显存: {total_size:.2f} MB")
        
        # 输出主要层的占用
        sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"  主要层占用:")
        for layer_name, size in sorted_layers:
            logger.info(f"    {layer_name}: {size:.2f} MB")
    
    def log_optimizer_memory(self, optimizer: torch.optim.Optimizer, name: str = "Optimizer"):
        """输出优化器状态的显存占用"""
        if not self.is_cuda:
            return
        
        total_size = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for key, val in state.items():
                        if isinstance(val, torch.Tensor):
                            total_size += val.element_size() * val.nelement()
        
        logger.info(f"{name} 状态显存: {total_size / 1024**2:.2f} MB")
    
    def clear_cache(self):
        """清理显存缓存"""
        if self.is_cuda:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
