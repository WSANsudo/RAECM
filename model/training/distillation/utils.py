"""
工具类模块 - 早停和错误分析
"""

import os
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
    
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False


class ErrorAnalyzer:
    """错误分析收集器"""
    
    def __init__(self, output_path: str = ""):
        self.errors: List[Dict] = []
        self.output_path = output_path
    
    def add_error(
        self, 
        input_text: str, 
        predicted: str, 
        expected: str, 
        loss: float,
        metadata: Optional[Dict] = None
    ):
        """添加错误样本"""
        error = {
            'input': input_text[:500],  # 截断
            'predicted': predicted,
            'expected': expected,
            'loss': loss,
            'metadata': metadata or {}
        }
        self.errors.append(error)
    
    def save(self, path: Optional[str] = None):
        """保存错误分析结果"""
        save_path = path or self.output_path
        if not save_path:
            save_path = 'error_analysis.jsonl'
        
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for error in self.errors:
                f.write(json.dumps(error, ensure_ascii=False) + '\n')
        
        logger.info(f"错误分析已保存: {save_path} ({len(self.errors)} 条)")
    
    def get_summary(self) -> Dict:
        """获取错误摘要"""
        if not self.errors:
            return {'total_errors': 0}
        
        losses = [e['loss'] for e in self.errors]
        return {
            'total_errors': len(self.errors),
            'avg_loss': sum(losses) / len(losses),
            'max_loss': max(losses),
            'min_loss': min(losses)
        }
