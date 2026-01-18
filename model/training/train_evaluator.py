#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练评估记录器
用于记录训练过程中的详细评估指标，支持 sub-epoch 评估频率
输出 train_evaluate.json 文件，便于后续作图分析
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter


class TrainEvaluator:
    """训练评估记录器 - 生成 train_evaluate.json"""
    
    def __init__(
        self,
        output_dir: str,
        model_name: str,
        task_type: str,
        eval_every_n_epochs: float = 0.2,
        class_names: List[str] = None
    ):
        """
        初始化训练评估记录器
        
        Args:
            output_dir: 输出目录
            model_name: 模型名称
            task_type: 任务类型 (vendor/os/devicetype)
            eval_every_n_epochs: 评估频率（epoch 单位，如 0.2 表示每 0.2 epoch 评估一次）
            class_names: 类别名称列表
        """
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, 'train_evaluate.json')
        self.eval_every_n_epochs = eval_every_n_epochs
        
        # 初始化记录结构
        self.records = {
            "metadata": {
                "model": model_name,
                "task": task_type,
                "eval_frequency": eval_every_n_epochs,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "class_names": class_names or []
            },
            "evaluations": []
        }
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def record_evaluation(
        self,
        epoch: float,
        step: int,
        train_loss: float,
        eval_loss: float,
        predictions: List[str],
        labels: List[str],
        learning_rate: float = None
    ) -> Dict:
        """
        记录一次评估结果
        
        Args:
            epoch: 当前 epoch 进度（如 1.2 表示第 1 个 epoch 的 20% 处）
            step: 当前训练步数
            train_loss: 当前训练损失（累计平均）
            eval_loss: 验证集损失
            predictions: 预测标签列表
            labels: 真实标签列表
            learning_rate: 当前学习率
            
        Returns:
            评估记录字典
        """
        # 计算基础指标
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(labels) if labels else 0
        
        # 计算详细指标
        try:
            from sklearn.metrics import (
                f1_score, precision_score, recall_score,
                classification_report
            )
            
            f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
            precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
            recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
            
            # 计算每个类别的指标
            report = classification_report(labels, predictions, output_dict=True, zero_division=0)
            per_class = {}
            for class_name, metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                per_class[class_name] = {
                    "precision": round(metrics['precision'], 4),
                    "recall": round(metrics['recall'], 4),
                    "f1": round(metrics['f1-score'], 4),
                    "support": int(metrics['support'])
                }
            
        except ImportError:
            f1_macro = 0
            f1_weighted = 0
            precision_macro = 0
            recall_macro = 0
            per_class = {}
        
        # 构建评估记录
        evaluation = {
            "epoch": round(epoch, 2),
            "step": step,
            "train_loss": round(train_loss, 6),
            "eval_loss": round(eval_loss, 6) if eval_loss else None,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_f1": round(f1_weighted, 4),
            "macro_precision": round(precision_macro, 4),
            "macro_recall": round(recall_macro, 4),
            "learning_rate": learning_rate,
            "total_samples": len(labels),
            "correct_samples": correct,
            "per_class": per_class
        }
        
        self.records["evaluations"].append(evaluation)
        
        # 实时保存
        self._save()
        
        return evaluation
    
    def finalize(self):
        """完成记录"""
        self.records["metadata"]["end_time"] = datetime.now().isoformat()
        self.records["metadata"]["total_evaluations"] = len(self.records["evaluations"])
        
        # 计算最佳指标
        if self.records["evaluations"]:
            best_eval = max(self.records["evaluations"], key=lambda x: x["macro_f1"])
            self.records["metadata"]["best_epoch"] = best_eval["epoch"]
            self.records["metadata"]["best_macro_f1"] = best_eval["macro_f1"]
            self.records["metadata"]["best_accuracy"] = best_eval["accuracy"]
        
        self._save()
    
    def _save(self):
        """保存记录到文件"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
    
    def get_evaluations(self) -> List[Dict]:
        """获取所有评估记录"""
        return self.records["evaluations"]
