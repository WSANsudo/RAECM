#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练指标记录器
记录训练过程中的所有详细指标，并保存为JSON格式
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch


# 指标字段说明
METRICS_DESCRIPTIONS = {
    # 元数据
    "metadata": {
        "description": "训练元数据信息",
        "fields": {
            "task_type": "任务类型 (vendor/os/devicetype)",
            "model_name": "模型名称",
            "model_path": "模型路径",
            "start_time": "训练开始时间 (ISO格式)",
            "end_time": "训练结束时间 (ISO格式)",
            "total_duration_seconds": "总训练时长 (秒)",
            "total_epochs": "总训练轮数",
            "completed_epochs": "完成的轮数",
            "early_stopped": "是否早停",
            "best_epoch": "最佳模型所在轮次",
            "train_samples": "训练样本数",
            "eval_samples": "验证样本数",
            "batch_size": "批次大小",
            "learning_rate": "初始学习率",
            "num_classes": "类别数量",
            "class_names": "类别名称列表"
        }
    },
    
    # 基础训练指标
    "basic_training": {
        "description": "基础训练指标",
        "fields": {
            "epoch": "当前训练轮次 (从1开始)",
            "train_loss": "训练集平均损失",
            "eval_loss": "验证集平均损失",
            "learning_rate": "当前学习率",
            "train_time_seconds": "本轮训练耗时 (秒)",
            "eval_time_seconds": "本轮评估耗时 (秒)",
            "total_time_seconds": "本轮总耗时 (秒)"
        }
    },
    
    # 分类性能指标
    "classification_metrics": {
        "description": "分类性能指标",
        "fields": {
            "accuracy": "准确率 (正确预测数/总预测数)",
            "f1_macro": "F1宏平均 (各类别F1的算术平均，不考虑类别样本数)",
            "f1_weighted": "F1加权平均 (按类别样本数加权的F1平均)",
            "f1_micro": "F1微平均 (全局计算精确率和召回率后计算F1)",
            "precision_macro": "精确率宏平均 (各类别精确率的算术平均)",
            "precision_weighted": "精确率加权平均 (按类别样本数加权)",
            "precision_micro": "精确率微平均 (全局TP/(TP+FP))",
            "recall_macro": "召回率宏平均 (各类别召回率的算术平均)",
            "recall_weighted": "召回率加权平均 (按类别样本数加权)",
            "recall_micro": "召回率微平均 (全局TP/(TP+FN))",
            "cohen_kappa": "Cohen's Kappa系数 (考虑随机一致性的分类准确度)",
            "matthews_corrcoef": "Matthews相关系数 (适用于不平衡数据的分类指标)"
        }
    },
    
    # 类别级别指标
    "per_class_metrics": {
        "description": "每个类别的详细指标",
        "fields": {
            "class_name": "类别名称",
            "f1": "该类别的F1分数",
            "precision": "该类别的精确率 (TP/(TP+FP))",
            "recall": "该类别的召回率 (TP/(TP+FN))",
            "support": "该类别在验证集中的样本数",
            "predicted_count": "模型预测为该类别的次数",
            "correct_count": "正确预测该类别的次数"
        }
    },
    
    # 训练过程指标
    "training_process": {
        "description": "训练过程监控指标",
        "fields": {
            "gradient_norm": "梯度L2范数 (监控梯度爆炸/消失)",
            "gradient_max": "梯度最大绝对值",
            "param_norm": "模型参数L2范数",
            "lr_scheduler_step": "学习率调度器当前步数",
            "samples_per_second": "训练吞吐量 (样本/秒)",
            "batches_per_second": "批次处理速度 (批次/秒)",
            "loss_per_batch": "每个批次的损失列表 (可选，用于详细分析)"
        }
    },
    
    # 硬件资源指标
    "hardware_metrics": {
        "description": "硬件资源使用情况",
        "fields": {
            "gpu_memory_used_gb": "GPU显存使用量 (GB)",
            "gpu_memory_total_gb": "GPU总显存 (GB)",
            "gpu_memory_percent": "GPU显存使用率 (%)",
            "gpu_memory_peak_gb": "GPU显存峰值 (GB)",
            "gpu_utilization_percent": "GPU计算利用率 (%, 如果可用)",
            "cpu_percent": "CPU使用率 (%)",
            "ram_used_gb": "内存使用量 (GB)",
            "ram_percent": "内存使用率 (%)"
        }
    },
    
    # 模型状态指标
    "model_state": {
        "description": "模型训练状态",
        "fields": {
            "best_f1_macro": "历史最佳F1 Macro",
            "best_accuracy": "历史最佳准确率",
            "best_eval_loss": "历史最佳验证损失",
            "patience_counter": "早停计数器 (连续未提升的轮数)",
            "is_best_model": "本轮是否为最佳模型",
            "model_saved": "本轮是否保存了模型",
            "improvement": "相比上一最佳的提升幅度"
        }
    },
    
    # 预测分析指标
    "prediction_analysis": {
        "description": "预测结果分析",
        "fields": {
            "prediction_distribution": "预测标签分布 (各类别预测次数)",
            "label_distribution": "真实标签分布 (各类别实际次数)",
            "prediction_diversity": "预测多样性 (预测了多少个不同类别)",
            "most_confused_pairs": "最容易混淆的类别对 (错误预测最多的组合)",
            "rare_class_accuracy": "少数类别平均准确率 (样本数<10的类别)",
            "major_class_accuracy": "主要类别平均准确率 (样本数>=10的类别)"
        }
    },
    
    # 稳定性指标
    "stability_metrics": {
        "description": "训练稳定性指标",
        "fields": {
            "loss_change": "损失变化 (本轮-上轮)",
            "loss_change_percent": "损失变化率 (%)",
            "f1_change": "F1变化 (本轮-上轮)",
            "accuracy_change": "准确率变化 (本轮-上轮)",
            "loss_variance_last3": "最近3轮损失方差",
            "f1_variance_last3": "最近3轮F1方差",
            "is_converging": "是否在收敛 (损失持续下降)",
            "is_overfitting": "是否过拟合 (训练损失下降但验证损失上升)"
        }
    },
    
    # 混淆矩阵
    "confusion_matrix": {
        "description": "混淆矩阵 (每5轮记录一次完整矩阵)",
        "fields": {
            "matrix": "混淆矩阵 (二维数组，行为真实标签，列为预测标签)",
            "labels": "标签顺序 (对应矩阵的行列顺序)",
            "normalized_matrix": "归一化混淆矩阵 (按行归一化)"
        }
    }
}


class MetricsRecorder:
    """训练指标记录器"""
    
    def __init__(
        self,
        output_dir: str,
        task_type: str,
        model_name: str,
        model_path: str,
        train_samples: int,
        eval_samples: int,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        class_names: List[str] = None
    ):
        """
        初始化指标记录器
        
        Args:
            output_dir: 输出目录
            task_type: 任务类型
            model_name: 模型名称
            model_path: 模型路径
            train_samples: 训练样本数
            eval_samples: 验证样本数
            batch_size: 批次大小
            learning_rate: 学习率
            num_epochs: 总轮数
            class_names: 类别名称列表
        """
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, 'data_records.json')
        
        # 初始化记录结构
        self.records = {
            "field_descriptions": METRICS_DESCRIPTIONS,
            "metadata": {
                "task_type": task_type,
                "model_name": model_name,
                "model_path": model_path,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_duration_seconds": None,
                "total_epochs": num_epochs,
                "completed_epochs": 0,
                "early_stopped": False,
                "best_epoch": None,
                "train_samples": train_samples,
                "eval_samples": eval_samples,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_classes": len(class_names) if class_names else None,
                "class_names": class_names
            },
            "epochs": {}
        }
        
        # 历史记录（用于计算稳定性指标）
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "f1_macro": [],
            "accuracy": []
        }
        
        # 最佳指标
        self.best_f1_macro = 0.0
        self.best_accuracy = 0.0
        self.best_eval_loss = 999999.0  # 使用大数代替 inf，避免 JSON 序列化问题
        self.best_epoch = None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        eval_loss: float,
        learning_rate: float,
        train_time: float,
        eval_time: float,
        predictions: List[str],
        labels: List[str],
        gradient_norm: float = None,
        gradient_max: float = None,
        param_norm: float = None,
        lr_step: int = None,
        samples_per_second: float = None,
        is_best: bool = False,
        model_saved: bool = False,
        patience_counter: int = 0,
        record_confusion_matrix: bool = False
    ):
        """
        记录一个epoch的所有指标
        
        Args:
            epoch: 轮次
            train_loss: 训练损失
            eval_loss: 验证损失
            learning_rate: 当前学习率
            train_time: 训练耗时
            eval_time: 评估耗时
            predictions: 预测标签列表
            labels: 真实标签列表
            gradient_norm: 梯度范数
            gradient_max: 梯度最大值
            param_norm: 参数范数
            lr_step: 学习率调度器步数
            samples_per_second: 吞吐量
            is_best: 是否最佳模型
            model_saved: 是否保存模型
            patience_counter: 早停计数
            record_confusion_matrix: 是否记录混淆矩阵
        """
        epoch_record = {}
        
        # 1. 基础训练指标
        epoch_record["basic_training"] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "eval_loss": round(eval_loss, 6),
            "learning_rate": learning_rate,
            "train_time_seconds": round(train_time, 2),
            "eval_time_seconds": round(eval_time, 2),
            "total_time_seconds": round(train_time + eval_time, 2)
        }
        
        # 2. 分类性能指标
        classification_metrics = self._compute_classification_metrics(predictions, labels)
        epoch_record["classification_metrics"] = classification_metrics
        
        # 3. 类别级别指标
        per_class_metrics = self._compute_per_class_metrics(predictions, labels)
        epoch_record["per_class_metrics"] = per_class_metrics
        
        # 4. 训练过程指标
        epoch_record["training_process"] = {
            "gradient_norm": round(gradient_norm, 6) if gradient_norm else None,
            "gradient_max": round(gradient_max, 6) if gradient_max else None,
            "param_norm": round(param_norm, 6) if param_norm else None,
            "lr_scheduler_step": lr_step,
            "samples_per_second": round(samples_per_second, 2) if samples_per_second else None,
            "batches_per_second": None  # 可选
        }
        
        # 5. 硬件资源指标
        epoch_record["hardware_metrics"] = self._get_hardware_metrics()
        
        # 6. 模型状态指标
        improvement = None
        if is_best:
            improvement = classification_metrics["f1_macro"] - self.best_f1_macro
            self.best_f1_macro = classification_metrics["f1_macro"]
            self.best_accuracy = classification_metrics["accuracy"]
            self.best_eval_loss = eval_loss
            self.best_epoch = epoch
        
        epoch_record["model_state"] = {
            "best_f1_macro": round(self.best_f1_macro, 6),
            "best_accuracy": round(self.best_accuracy, 6),
            "best_eval_loss": round(self.best_eval_loss, 6),
            "patience_counter": patience_counter,
            "is_best_model": is_best,
            "model_saved": model_saved,
            "improvement": round(improvement, 6) if improvement else None
        }
        
        # 7. 预测分析指标
        epoch_record["prediction_analysis"] = self._compute_prediction_analysis(predictions, labels)
        
        # 8. 稳定性指标
        self.history["train_loss"].append(train_loss)
        self.history["eval_loss"].append(eval_loss)
        self.history["f1_macro"].append(classification_metrics["f1_macro"])
        self.history["accuracy"].append(classification_metrics["accuracy"])
        
        epoch_record["stability_metrics"] = self._compute_stability_metrics(epoch)
        
        # 9. 混淆矩阵（每5轮或最后一轮记录）
        if record_confusion_matrix or epoch % 5 == 0:
            epoch_record["confusion_matrix"] = self._compute_confusion_matrix(predictions, labels)
        
        # 保存到记录
        self.records["epochs"][str(epoch)] = epoch_record
        self.records["metadata"]["completed_epochs"] = epoch
        self.records["metadata"]["best_epoch"] = self.best_epoch
        
        # 实时保存
        self._save()
        
        return epoch_record
    
    def _compute_classification_metrics(self, predictions: List[str], labels: List[str]) -> Dict:
        """计算分类性能指标"""
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                cohen_kappa_score, matthews_corrcoef
            )
            
            accuracy = accuracy_score(labels, predictions)
            
            metrics = {
                "accuracy": round(accuracy, 6),
                "f1_macro": round(f1_score(labels, predictions, average='macro', zero_division=0), 6),
                "f1_weighted": round(f1_score(labels, predictions, average='weighted', zero_division=0), 6),
                "f1_micro": round(f1_score(labels, predictions, average='micro', zero_division=0), 6),
                "precision_macro": round(precision_score(labels, predictions, average='macro', zero_division=0), 6),
                "precision_weighted": round(precision_score(labels, predictions, average='weighted', zero_division=0), 6),
                "precision_micro": round(precision_score(labels, predictions, average='micro', zero_division=0), 6),
                "recall_macro": round(recall_score(labels, predictions, average='macro', zero_division=0), 6),
                "recall_weighted": round(recall_score(labels, predictions, average='weighted', zero_division=0), 6),
                "recall_micro": round(recall_score(labels, predictions, average='micro', zero_division=0), 6),
                "cohen_kappa": round(cohen_kappa_score(labels, predictions), 6),
                "matthews_corrcoef": round(matthews_corrcoef(labels, predictions), 6)
            }
            
        except ImportError:
            # sklearn不可用时的简化计算
            correct = sum(1 for p, l in zip(predictions, labels) if p == l)
            accuracy = correct / len(labels) if labels else 0
            
            metrics = {
                "accuracy": round(accuracy, 6),
                "f1_macro": None,
                "f1_weighted": None,
                "f1_micro": None,
                "precision_macro": None,
                "precision_weighted": None,
                "precision_micro": None,
                "recall_macro": None,
                "recall_weighted": None,
                "recall_micro": None,
                "cohen_kappa": None,
                "matthews_corrcoef": None
            }
        
        return metrics
    
    def _compute_per_class_metrics(self, predictions: List[str], labels: List[str]) -> Dict:
        """计算每个类别的指标"""
        try:
            from sklearn.metrics import classification_report
            
            report = classification_report(labels, predictions, output_dict=True, zero_division=0)
            
            per_class = {}
            for class_name, metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                
                # 计算预测次数和正确次数
                predicted_count = sum(1 for p in predictions if p == class_name)
                correct_count = sum(1 for p, l in zip(predictions, labels) if p == l and l == class_name)
                
                per_class[class_name] = {
                    "f1": round(metrics['f1-score'], 6),
                    "precision": round(metrics['precision'], 6),
                    "recall": round(metrics['recall'], 6),
                    "support": int(metrics['support']),
                    "predicted_count": predicted_count,
                    "correct_count": correct_count
                }
            
            return per_class
            
        except ImportError:
            return {}
    
    def _compute_prediction_analysis(self, predictions: List[str], labels: List[str]) -> Dict:
        """计算预测分析指标"""
        from collections import Counter
        
        pred_dist = dict(Counter(predictions))
        label_dist = dict(Counter(labels))
        
        # 预测多样性
        prediction_diversity = len(set(predictions))
        
        # 混淆对分析
        confused_pairs = defaultdict(int)
        for pred, label in zip(predictions, labels):
            if pred != label:
                pair = f"{label} -> {pred}"
                confused_pairs[pair] += 1
        
        # 取前10个最混淆的对
        most_confused = dict(sorted(confused_pairs.items(), key=lambda x: -x[1])[:10])
        
        # 少数类别和主要类别性能
        rare_correct = 0
        rare_total = 0
        major_correct = 0
        major_total = 0
        
        for label in set(labels):
            count = label_dist.get(label, 0)
            label_preds = [p for p, l in zip(predictions, labels) if l == label]
            correct = sum(1 for p in label_preds if p == label)
            
            if count < 10:
                rare_correct += correct
                rare_total += len(label_preds)
            else:
                major_correct += correct
                major_total += len(label_preds)
        
        rare_accuracy = rare_correct / rare_total if rare_total > 0 else None
        major_accuracy = major_correct / major_total if major_total > 0 else None
        
        return {
            "prediction_distribution": pred_dist,
            "label_distribution": label_dist,
            "prediction_diversity": prediction_diversity,
            "most_confused_pairs": most_confused,
            "rare_class_accuracy": round(rare_accuracy, 6) if rare_accuracy else None,
            "major_class_accuracy": round(major_accuracy, 6) if major_accuracy else None
        }
    
    def _compute_stability_metrics(self, epoch: int) -> Dict:
        """计算稳定性指标"""
        metrics = {
            "loss_change": None,
            "loss_change_percent": None,
            "f1_change": None,
            "accuracy_change": None,
            "loss_variance_last3": None,
            "f1_variance_last3": None,
            "is_converging": None,
            "is_overfitting": None
        }
        
        if len(self.history["eval_loss"]) >= 2:
            prev_loss = self.history["eval_loss"][-2]
            curr_loss = self.history["eval_loss"][-1]
            metrics["loss_change"] = round(curr_loss - prev_loss, 6)
            metrics["loss_change_percent"] = round((curr_loss - prev_loss) / prev_loss * 100, 2) if prev_loss != 0 else None
            
            prev_f1 = self.history["f1_macro"][-2]
            curr_f1 = self.history["f1_macro"][-1]
            metrics["f1_change"] = round(curr_f1 - prev_f1, 6)
            
            prev_acc = self.history["accuracy"][-2]
            curr_acc = self.history["accuracy"][-1]
            metrics["accuracy_change"] = round(curr_acc - prev_acc, 6)
            
            # 判断是否收敛
            metrics["is_converging"] = curr_loss < prev_loss
            
            # 判断是否过拟合
            train_improving = self.history["train_loss"][-1] < self.history["train_loss"][-2]
            eval_worsening = curr_loss > prev_loss
            metrics["is_overfitting"] = train_improving and eval_worsening
        
        if len(self.history["eval_loss"]) >= 3:
            last3_loss = self.history["eval_loss"][-3:]
            last3_f1 = self.history["f1_macro"][-3:]
            
            mean_loss = sum(last3_loss) / 3
            mean_f1 = sum(last3_f1) / 3
            
            metrics["loss_variance_last3"] = round(sum((x - mean_loss) ** 2 for x in last3_loss) / 3, 8)
            metrics["f1_variance_last3"] = round(sum((x - mean_f1) ** 2 for x in last3_f1) / 3, 8)
        
        return metrics
    
    def _compute_confusion_matrix(self, predictions: List[str], labels: List[str]) -> Dict:
        """计算混淆矩阵"""
        try:
            from sklearn.metrics import confusion_matrix
            import numpy as np
            
            # 获取所有唯一标签
            all_labels = sorted(set(labels) | set(predictions))
            
            cm = confusion_matrix(labels, predictions, labels=all_labels)
            
            # 归一化（按行）
            row_sums = cm.sum(axis=1, keepdims=True)
            normalized_cm = np.divide(cm, row_sums, where=row_sums != 0)
            
            return {
                "matrix": cm.tolist(),
                "labels": all_labels,
                "normalized_matrix": [[round(x, 4) for x in row] for row in normalized_cm.tolist()]
            }
            
        except ImportError:
            return None
    
    def _get_hardware_metrics(self) -> Dict:
        """获取硬件资源指标"""
        metrics = {
            "gpu_memory_used_gb": None,
            "gpu_memory_total_gb": None,
            "gpu_memory_percent": None,
            "gpu_memory_peak_gb": None,
            "gpu_utilization_percent": None,
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_percent": None
        }
        
        # GPU指标
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                
                metrics["gpu_memory_used_gb"] = round(allocated, 2)
                metrics["gpu_memory_total_gb"] = round(total, 2)
                metrics["gpu_memory_percent"] = round(allocated / total * 100, 1)
                metrics["gpu_memory_peak_gb"] = round(peak, 2)
            except:
                pass
        
        # CPU和内存指标
        try:
            import psutil
            metrics["cpu_percent"] = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            metrics["ram_used_gb"] = round(mem.used / 1024**3, 2)
            metrics["ram_percent"] = mem.percent
        except ImportError:
            pass
        
        return metrics
    
    def finalize(self, early_stopped: bool = False):
        """完成记录，保存最终结果"""
        self.records["metadata"]["end_time"] = datetime.now().isoformat()
        
        start = datetime.fromisoformat(self.records["metadata"]["start_time"])
        end = datetime.fromisoformat(self.records["metadata"]["end_time"])
        self.records["metadata"]["total_duration_seconds"] = round((end - start).total_seconds(), 2)
        self.records["metadata"]["early_stopped"] = early_stopped
        
        self._save()
    
    def _save(self):
        """保存记录到文件"""
        # 清理特殊浮点值（Infinity, -Infinity, NaN）
        def clean_value(obj):
            if isinstance(obj, float):
                if obj == float('inf'):
                    return "Infinity"  # 转为字符串
                elif obj == float('-inf'):
                    return "-Infinity"
                elif obj != obj:  # NaN check
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: clean_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_value(item) for item in obj]
            return obj
        
        cleaned_records = clean_value(self.records)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            "completed_epochs": self.records["metadata"]["completed_epochs"],
            "best_epoch": self.best_epoch,
            "best_f1_macro": self.best_f1_macro,
            "best_accuracy": self.best_accuracy,
            "best_eval_loss": self.best_eval_loss,
            "early_stopped": self.records["metadata"]["early_stopped"],
            "output_file": self.output_file
        }
