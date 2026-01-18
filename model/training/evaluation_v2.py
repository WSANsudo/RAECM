#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估模块 V2 - 完整的分类评估

主要功能：
1. Macro-F1（主指标，适合长尾分布）
2. 每类 Precision/Recall/F1
3. 混淆矩阵
4. 零召回类检测
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter


def clean_prediction(pred: str) -> str:
    """
    清洗模型预测输出
    - 去除空白
    - 去除特殊 token
    - 只取第一行
    - 去除常见前缀
    """
    if not pred:
        return 'Unknown'
    
    # 去除特殊 token
    special_tokens = [
        '<|im_end|>', '<|im_start|>', '<|endoftext|>',
        '<|eot_id|>', '<|end|>', '</s>', '<s>',
        '[/INST]', '[INST]', '###'
    ]
    for token in special_tokens:
        pred = pred.replace(token, '')
    
    # 去除空白，只取第一行
    pred = pred.strip()
    if '\n' in pred:
        pred = pred.split('\n')[0].strip()
    
    # 去除常见前缀
    prefixes = [
        'Vendor:', 'vendor:', 'VENDOR:',
        'OS:', 'os:', 'Operating System:',
        'Device Type:', 'devicetype:', 'Type:',
        'Answer:', 'Output:', 'Result:',
    ]
    for prefix in prefixes:
        if pred.startswith(prefix):
            pred = pred[len(prefix):].strip()
    
    # 去除括号内的解释
    pred = re.sub(r'\s*\([^)]*\)\s*$', '', pred)
    pred = re.sub(r'\s*\[[^\]]*\]\s*$', '', pred)
    
    # 去除尾部标点
    pred = pred.rstrip('.,;:!?')
    
    return pred.strip() or 'Unknown'


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> Dict:
    """
    计算完整的分类指标
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        labels: 标签列表（可选，用于固定顺序）
        
    Returns:
        {
            'accuracy': float,
            'macro_f1': float,
            'weighted_f1': float,
            'micro_f1': float,
            'per_class': {label: {'precision', 'recall', 'f1', 'support'}},
            'zero_recall_classes': [labels with recall=0],
            'confusion_matrix': {...}
        }
    """
    assert len(y_true) == len(y_pred), "长度不匹配"
    
    n = len(y_true)
    if n == 0:
        return {'error': 'empty_data'}
    
    # 获取所有标签
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    # 基本统计
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / n
    
    # 每类统计
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Macro-F1（所有类别平均，不考虑样本数）
    f1_scores = [m['f1'] for m in per_class.values() if m['support'] > 0]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    # Weighted-F1（按样本数加权）
    weighted_f1 = sum(
        m['f1'] * m['support'] for m in per_class.values()
    ) / n if n > 0 else 0.0
    
    # Micro-F1（全局 TP/FP/FN）
    total_tp = sum(m['tp'] for m in per_class.values())
    total_fp = sum(m['fp'] for m in per_class.values())
    total_fn = sum(m['fn'] for m in per_class.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0.0
    
    # 零召回类
    zero_recall_classes = [
        label for label, m in per_class.items()
        if m['support'] > 0 and m['recall'] == 0
    ]
    
    # 混淆矩阵
    confusion = {}
    for t, p in zip(y_true, y_pred):
        if t not in confusion:
            confusion[t] = {}
        confusion[t][p] = confusion[t].get(p, 0) + 1
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'micro_f1': micro_f1,
        'total_samples': n,
        'correct': correct,
        'n_classes': len([l for l in labels if per_class.get(l, {}).get('support', 0) > 0]),
        'per_class': per_class,
        'zero_recall_classes': zero_recall_classes,
        'confusion_matrix': confusion
    }


def print_evaluation_report(metrics: Dict, top_n: int = 15):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("评估报告")
    print("=" * 70)
    
    print(f"\n【总体指标】")
    print(f"  样本数: {metrics['total_samples']}")
    print(f"  类别数: {metrics['n_classes']}")
    print(f"  正确数: {metrics['correct']}")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Macro-F1: {metrics['macro_f1']*100:.2f}%  ← 主指标")
    print(f"  Weighted-F1: {metrics['weighted_f1']*100:.2f}%")
    print(f"  Micro-F1: {metrics['micro_f1']*100:.2f}%")
    
    # 零召回类警告
    if metrics['zero_recall_classes']:
        print(f"\n⚠️ 【零召回类别】(模型完全无法识别)")
        for label in metrics['zero_recall_classes']:
            support = metrics['per_class'][label]['support']
            print(f"  - {label} (样本数: {support})")
    
    # 每类指标
    print(f"\n【每类指标】(按 F1 排序, Top {top_n})")
    print(f"  {'类别':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 62)
    
    sorted_classes = sorted(
        metrics['per_class'].items(),
        key=lambda x: (-x[1]['f1'], -x[1]['support'])
    )
    
    for label, m in sorted_classes[:top_n]:
        if m['support'] == 0:
            continue
        print(f"  {label:<20} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% "
              f"{m['f1']*100:>9.1f}% {m['support']:>10}")
    
    if len(sorted_classes) > top_n:
        print(f"  ... 还有 {len(sorted_classes) - top_n} 类")
    
    # 主要错误
    print(f"\n【主要错误】(Top 10)")
    errors = []
    for true_label, preds in metrics['confusion_matrix'].items():
        for pred_label, count in preds.items():
            if true_label != pred_label:
                errors.append((true_label, pred_label, count))
    
    errors.sort(key=lambda x: -x[2])
    for true_label, pred_label, count in errors[:10]:
        print(f"  {true_label} → {pred_label}: {count}")
    
    print("\n" + "=" * 70)


def evaluate_predictions(
    predictions: List[Dict],
    task_type: str,
    label_field: str = None
) -> Dict:
    """
    评估预测结果
    
    Args:
        predictions: 预测结果列表，每个元素包含 'true_label' 和 'prediction'
        task_type: 任务类型
        label_field: 标签字段名（默认与 task_type 相同）
        
    Returns:
        评估指标
    """
    if label_field is None:
        label_field = task_type
    
    y_true = []
    y_pred = []
    
    for item in predictions:
        true_label = item.get('true_label', item.get('label', ''))
        pred_label = item.get('prediction', item.get('pred', ''))
        
        # 清洗预测
        pred_label = clean_prediction(pred_label)
        
        y_true.append(str(true_label))
        y_pred.append(str(pred_label))
    
    return compute_metrics(y_true, y_pred)


def save_evaluation_report(metrics: Dict, output_path: str):
    """保存评估报告为 JSON"""
    # 转换为可序列化格式
    report = {
        'summary': {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'micro_f1': metrics['micro_f1'],
            'total_samples': metrics['total_samples'],
            'n_classes': metrics['n_classes'],
            'zero_recall_classes': metrics['zero_recall_classes']
        },
        'per_class': metrics['per_class'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"评估报告已保存: {output_path}")
