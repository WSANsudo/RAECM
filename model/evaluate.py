#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化分类模型评估脚本
Simple Classifier Model Evaluation Script

使用方法:
    # 评估厂商识别模型（自动查找最新模型）
    python evaluate.py --mt vd
    
    # 评估操作系统识别模型
    python evaluate.py --mt os
    
    # 评估设备类型识别模型
    python evaluate.py --mt dt
    
    # 指定模型路径
    python evaluate.py --mt vd --model output/simple/vendor/qwen2.5-3b/best_model
    
    # 快速测试（限制样本数）
    python evaluate.py --mt vd --max-samples 50
    
    # 指定测试数据
    python evaluate.py --mt vd --test data/vendor/test.jsonl
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.simple_classifier import evaluate_simple_classifier, clean_prediction
from training.gpu_check import check_gpu_availability

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 模型类型映射
MODEL_TYPE_MAP = {
    'vd': {
        'name': 'vendor',
        'description': '厂商识别',
        'field': 'vendor'
    },
    'os': {
        'name': 'os',
        'description': '操作系统识别',
        'field': 'os'
    },
    'dt': {
        'name': 'devicetype',
        'description': '设备类型识别',
        'field': 'type'
    }
}


def find_latest_model(task_type: str, mode: str = 'simple') -> str:
    """
    查找最新的训练模型
    
    Args:
        task_type: 任务类型 (vendor/os/devicetype)
        mode: 训练模式 (simple)
    
    Returns:
        模型路径，如果找不到返回 None
    """
    base_dir = os.path.join('./output', mode, task_type)
    
    if not os.path.exists(base_dir):
        return None
    
    # 查找所有模型目录
    models = []
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        # 优先使用 best_model（基于准确率的最佳模型）
        best_model = os.path.join(model_dir, 'best_model')
        if os.path.exists(best_model):
            # 获取修改时间
            mtime = os.path.getmtime(best_model)
            models.append((mtime, best_model, model_name, 'best_model'))
            continue  # 找到 best_model 就不再查找其他
        
        # 其次使用 final_model
        final_model = os.path.join(model_dir, 'final_model')
        if os.path.exists(final_model):
            mtime = os.path.getmtime(final_model)
            models.append((mtime, final_model, model_name, 'final_model'))
    
    if not models:
        return None
    
    # 返回最新的模型（优先 best_model）
    models.sort(reverse=True)
    return models[0][1]


def list_available_models(task_type: str, mode: str = 'simple') -> list:
    """
    列出所有可用模型
    
    Returns:
        [(model_name, model_path, model_type), ...]
    """
    base_dir = os.path.join('./output', mode, task_type)
    
    if not os.path.exists(base_dir):
        return []
    
    models = []
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        # 检查 best_model
        best_model = os.path.join(model_dir, 'best_model')
        if os.path.exists(best_model):
            models.append((model_name, best_model, 'best_model'))
        
        # 检查 final_model
        final_model = os.path.join(model_dir, 'final_model')
        if os.path.exists(final_model):
            models.append((model_name, final_model, 'final_model'))
    
    return models


def select_model_interactive(task_type: str, mode: str = 'simple') -> str:
    """交互式选择模型"""
    models = list_available_models(task_type, mode)
    
    if not models:
        return None
    
    print()
    print(f"找到以下 {task_type} 模型:")
    print("-" * 60)
    for i, (name, path, mtype) in enumerate(models, 1):
        print(f"  [{i}] {name} ({mtype})")
    print("-" * 60)
    
    while True:
        try:
            choice = input(f"选择模型 [1-{len(models)}, 回车=取消]: ").strip()
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"✓ 已选择: {selected[0]} ({selected[2]})")
                return selected[1]
            else:
                print(f"请输入 1-{len(models)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n已取消")
            return None


def get_test_data_path(task_type: str, model_name: str = None) -> str:
    """获取测试数据路径
    
    Args:
        task_type: 任务类型 (vendor/os/devicetype)
        model_name: 模型名称，用于查找模型专用数据目录
    """
    # 优先使用模型专用数据目录
    if model_name:
        model_test = os.path.join('./data', model_name, task_type, 'test.jsonl')
        if os.path.exists(model_test):
            return model_test
        model_simple_test = os.path.join('./data', model_name, task_type, 'simple_test.jsonl')
        if os.path.exists(model_simple_test):
            return model_simple_test
    
    # 回退到任务类型子目录（旧格式兼容）
    task_test = os.path.join('./data', task_type, 'test.jsonl')
    if os.path.exists(task_test):
        return task_test
    
    # 简化格式测试数据
    simple_test = os.path.join('./data', task_type, 'simple_test.jsonl')
    if os.path.exists(simple_test):
        return simple_test
    
    # 回退到根目录
    root_test = os.path.join('./data', 'test.jsonl')
    if os.path.exists(root_test):
        return root_test
    
    return None


def convert_test_data_if_needed(test_data_path: str, task_type: str) -> str:
    """
    检查测试数据格式，如果是原始格式则转换为简化格式
    
    Returns:
        简化格式的测试数据路径
    """
    # 检查第一条数据
    with open(test_data_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            return test_data_path
        
        item = json.loads(first_line)
        messages = item.get('messages', [])
        if len(messages) < 3:
            return test_data_path
        
        # 检查 assistant 的内容是否是 JSON
        assistant_content = messages[-1].get('content', '')
        try:
            json.loads(assistant_content)
            # 是 JSON 格式，需要转换
            logger.info("检测到原始格式测试数据，转换为简化格式...")
            
            from training.simple_classifier import convert_to_simple_format
            import tempfile
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # 转换
            convert_to_simple_format(test_data_path, temp_path, task_type)
            logger.info(f"✓ 已转换为简化格式: {temp_path}")
            return temp_path
        except (json.JSONDecodeError, KeyError):
            # 已经是简化格式
            return test_data_path


def evaluate_detailed(
    model,
    tokenizer,
    test_data_path: str,
    task_type: str,
    device: torch.device,
    max_samples: int = None
) -> dict:
    """
    详细评估，包含每个样本的预测结果
    
    Returns:
        {
            'accuracy': float,
            'f1_macro': float,
            'f1_weighted': float,
            'total': int,
            'correct': int,
            'predictions': [{'expected': str, 'predicted': str, 'correct': bool}, ...],
            'confusion_matrix': dict,
            'per_class_metrics': dict
        }
    """
    from collections import Counter
    
    model.eval()
    
    # 加载数据
    data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    
    predictions = []
    labels = []
    details = []
    
    logger.info(f"开始评估 {len(data)} 个样本...")
    
    # 调试：输出前3个样本的详细信息
    debug_samples = min(3, len(data))
    
    for i, item in enumerate(data):
        if (i + 1) % 50 == 0:
            logger.info(f"  进度: {i+1}/{len(data)}")
        
        messages = item['messages']
        prompt_messages = messages[:-1]
        raw_expected_label = messages[-1]['content']
        
        # 从 JSON 中提取标签（如果是 JSON 格式）
        try:
            expected_json = json.loads(raw_expected_label)
            # 根据任务类型提取对应字段
            if 'vendor' in expected_json:
                expected_label = expected_json.get('vendor', 'null')
            elif 'os' in expected_json:
                expected_label = expected_json.get('os', 'null')
            elif 'devicetype' in expected_json:
                expected_label = expected_json.get('devicetype', 'null')
            elif 'type' in expected_json:
                expected_label = expected_json.get('type', 'null')
            else:
                expected_label = raw_expected_label
            expected_label = str(expected_label) if expected_label else 'null'
        except (json.JSONDecodeError, TypeError):
            # 如果不是 JSON，直接使用原始内容
            expected_label = raw_expected_label
        
        # 构建 prompt
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_text = '\n'.join([
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                for m in prompt_messages
            ]) + "\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
        
        # 准备停止标记 - 支持多种模型
        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)
        
        # 添加各种可能的停止标记
        special_tokens = ['<|im_end|>', '<|endoftext|>', '<|end|>', '<|eot_id|>', '\n']
        for token in special_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id and token_id not in stop_token_ids:
                    stop_token_ids.append(token_id)
            except:
                pass
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids if stop_token_ids else tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        
        # Decode
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # 手动移除可能残留的特殊标记
        for token in ['<|end_of_text|>', '<|endoftext|>', '<|im_end|>', '<|eot_id|>', '<|end|>']:
            raw_output = raw_output.replace(token, '')
        raw_output = raw_output.strip()
        
        # 使用统一的清理函数
        predicted_label = clean_prediction(raw_output)
        
        # 调试输出
        if i < debug_samples:
            logger.info(f"\n【调试样本 {i+1}】")
            logger.info(f"  期望标签: {expected_label}")
            logger.info(f"  预测标签: {predicted_label}")
            logger.info(f"  原始输出: {repr(tokenizer.decode(generated, skip_special_tokens=False))}")
        
        # 标准化（转小写比较）
        pred_lower = predicted_label.lower()
        exp_lower = expected_label.lower()
        
        is_correct = pred_lower == exp_lower
        
        predictions.append(pred_lower)
        labels.append(exp_lower)
        details.append({
            'expected': expected_label,
            'predicted': predicted_label,
            'correct': is_correct
        })
    
    # 计算准确率
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0
    
    # 计算 F1
    try:
        from sklearn.metrics import f1_score, confusion_matrix, classification_report
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # 混淆矩阵
        unique_labels = sorted(set(labels + predictions))
        cm = confusion_matrix(labels, predictions, labels=unique_labels)
        confusion_dict = {}
        for i, true_label in enumerate(unique_labels):
            confusion_dict[true_label] = {}
            for j, pred_label in enumerate(unique_labels):
                confusion_dict[true_label][pred_label] = int(cm[i][j])
        
        # 每个类别的指标
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        per_class = {k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
        
    except ImportError:
        logger.warning("sklearn 未安装，跳过 F1 和混淆矩阵计算")
        f1_macro = 0
        f1_weighted = 0
        confusion_dict = {}
        per_class = {}
    
    # 统计标签分布
    label_dist = Counter(labels)
    pred_dist = Counter(predictions)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'total': len(labels),
        'correct': correct,
        'predictions': details,
        'confusion_matrix': confusion_dict,
        'per_class_metrics': per_class,
        'label_distribution': dict(label_dist),
        'prediction_distribution': dict(pred_dist)
    }


def get_model_output_dir(model_path: str) -> str:
    """
    从模型路径获取模型输出目录
    
    例如:
        model_path = './output/simple/vendor/Qwen2.5-1.5B-Instruct/best_model'
        返回: './output/simple/vendor/Qwen2.5-1.5B-Instruct'
    """
    # 如果路径以 best_model 或 final_model 结尾，取父目录
    if model_path.endswith('best_model') or model_path.endswith('final_model'):
        return os.path.dirname(model_path)
    return model_path


def save_report(results: dict, output_path: str, task_info: dict, model_path: str, test_path: str):
    """保存评估报告"""
    
    # JSON 报告
    json_path = output_path + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'task': task_info['description'],
            'model_path': model_path,
            'test_data': test_path,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ JSON 报告已保存: {json_path}")
    
    # Markdown 报告
    md_path = output_path + '.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {task_info['description']}评估报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**模型路径**: `{model_path}`\n\n")
        f.write(f"**测试数据**: `{test_path}`\n\n")
        f.write("---\n\n")
        
        f.write("## 总体指标\n\n")
        f.write(f"- **准确率**: {results['accuracy']*100:.2f}%\n")
        f.write(f"- **F1 (Macro)**: {results['f1_macro']:.4f}\n")
        f.write(f"- **F1 (Weighted)**: {results['f1_weighted']:.4f}\n")
        f.write(f"- **正确数**: {results['correct']}/{results['total']}\n\n")
        
        if results['per_class_metrics']:
            f.write("## 每个类别的指标\n\n")
            f.write("| 类别 | 精确率 | 召回率 | F1 分数 | 样本数 |\n")
            f.write("|------|--------|--------|---------|--------|\n")
            for label, metrics in sorted(results['per_class_metrics'].items()):
                f.write(f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                       f"{metrics['f1-score']:.4f} | {int(metrics['support'])} |\n")
            f.write("\n")
        
        if results['confusion_matrix']:
            f.write("## 混淆矩阵\n\n")
            labels_list = sorted(results['confusion_matrix'].keys())
            f.write("| 真实\\预测 | " + " | ".join(labels_list) + " |\n")
            f.write("|" + "---|" * (len(labels_list) + 1) + "\n")
            for true_label in labels_list:
                row = [true_label]
                for pred_label in labels_list:
                    count = results['confusion_matrix'][true_label].get(pred_label, 0)
                    row.append(str(count))
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
        
        f.write("## 标签分布\n\n")
        f.write("### 真实标签分布\n\n")
        for label, count in sorted(results['label_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"- {label}: {count}\n")
        f.write("\n")
        
        f.write("### 预测标签分布\n\n")
        for label, count in sorted(results['prediction_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"- {label}: {count}\n")
        f.write("\n")
        
        # 错误样本
        errors = [p for p in results['predictions'] if not p['correct']]
        if errors:
            f.write(f"## 错误样本 ({len(errors)} 个)\n\n")
            for i, err in enumerate(errors[:20], 1):  # 只显示前 20 个
                f.write(f"### 错误 {i}\n\n")
                f.write(f"- **期望**: {err['expected']}\n")
                f.write(f"- **预测**: {err['predicted']}\n\n")
            if len(errors) > 20:
                f.write(f"\n*（还有 {len(errors)-20} 个错误未显示）*\n\n")
    
    logger.info(f"✓ Markdown 报告已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description='简化分类模型评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估厂商识别模型（自动查找）
  python evaluate.py --mt vd
  
  # 评估操作系统识别模型
  python evaluate.py --mt os
  
  # 指定模型路径
  python evaluate.py --mt vd --model output/simple/vendor/qwen2.5-3b/best_model
  
  # 快速测试
  python evaluate.py --mt vd --max-samples 50
  
  # 指定测试数据
  python evaluate.py --mt vd --test data/vendor/simple_test.jsonl

模型类型:
  vd  - 厂商识别
  os  - 操作系统识别
  dt  - 设备类型识别
        """
    )
    
    parser.add_argument('--mt', type=str, default='vd',
                        choices=['vd', 'os', 'dt'],
                        help='模型类型 (default: vd)')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径（默认自动查找最新模型）')
    parser.add_argument('--test', type=str, default=None,
                        help='测试数据路径（默认自动查找）')
    parser.add_argument('--output', type=str, default=None,
                        help='报告输出路径（默认: evaluation_report_<mt>）')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大评估样本数（用于快速测试）')
    parser.add_argument('--mode', type=str, default='simple',
                        choices=['simple'],
                        help='训练模式 (default: simple)')
    
    args = parser.parse_args()
    
    # 获取任务信息
    type_info = MODEL_TYPE_MAP[args.mt]
    
    print()
    print("=" * 60)
    print(f"  {type_info['description']}模型评估")
    print("=" * 60)
    print()
    
    # 确定模型路径
    model_path = args.model
    if not model_path:
        model_path = find_latest_model(type_info['name'], args.mode)
        if not model_path:
            logger.warning(f"未找到 {type_info['name']} 模型")
            model_path = select_model_interactive(type_info['name'], args.mode)
            if not model_path:
                logger.error("未选择模型，退出")
                sys.exit(1)
        else:
            logger.info(f"✓ 自动找到模型: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        sys.exit(1)
    
    # 确定测试数据路径 - 从模型路径提取模型名
    model_name = None
    model_output_dir = get_model_output_dir(model_path)
    # 从路径中提取模型名，如 ./output/simple/vendor/Qwen2.5-1.5B-Instruct -> Qwen2.5-1.5B-Instruct
    path_parts = model_output_dir.replace('\\', '/').split('/')
    if len(path_parts) >= 1:
        model_name = path_parts[-1]
    
    test_path = args.test
    if not test_path:
        test_path = get_test_data_path(type_info['name'], model_name)
        if not test_path:
            logger.error(f"未找到测试数据")
            logger.info(f"请将测试数据放在: data/{model_name}/{type_info['name']}/test.jsonl 或 data/{type_info['name']}/test.jsonl")
            sys.exit(1)
        logger.info(f"✓ 使用测试数据: {test_path}")
    
    if not os.path.exists(test_path):
        logger.error(f"测试数据不存在: {test_path}")
        sys.exit(1)
    
    # 检查并转换测试数据格式
    converted_test_path = convert_test_data_if_needed(test_path, type_info['name'])
    if converted_test_path != test_path:
        logger.info(f"使用转换后的测试数据: {converted_test_path}")
        test_path = converted_test_path
    
    # 确定输出路径 - 默认保存到模型专用文件夹
    if args.output:
        output_path = args.output
    else:
        # 获取模型输出目录
        model_output_dir = get_model_output_dir(model_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(model_output_dir, f"evaluation_{timestamp}")
        
        # 确保目录存在
        os.makedirs(model_output_dir, exist_ok=True)
    
    # 打印配置
    print(f"任务类型: {type_info['description']}")
    print(f"模型路径: {model_path}")
    print(f"测试数据: {test_path}")
    print(f"输出路径: {output_path}")
    if args.max_samples:
        print(f"最大样本数: {args.max_samples}")
    print("-" * 60)
    print()
    
    # GPU 检查
    check_gpu_availability(require_gpu=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    logger.info("加载模型...")
    try:
        # 检查是否是 LoRA 模型
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora:
            logger.info("检测到 LoRA 模型，加载并合并...")
            
            # 读取 adapter_config 获取基础模型路径
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get('base_model_name_or_path')
            
            if not base_model_path or not os.path.exists(base_model_path):
                logger.error(f"基础模型路径不存在: {base_model_path}")
                logger.info("请确保基础模型在正确的位置")
                sys.exit(1)
            
            logger.info(f"基础模型: {base_model_path}")
            
            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载基础模型
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto' if torch.cuda.is_available() else None
            )
            
            # 加载 LoRA adapter 并合并
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, model_path)
                logger.info("合并 LoRA 权重...")
                model = model.merge_and_unload()
                logger.info("✓ LoRA 模型加载并合并成功")
            except ImportError:
                logger.error("peft 库未安装，无法加载 LoRA 模型")
                logger.info("请安装: pip install peft")
                sys.exit(1)
        else:
            # 完整模型，直接加载
            logger.info("加载完整模型...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map='auto' if torch.cuda.is_available() else None
            )
            logger.info("✓ 模型加载成功")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 评估
    logger.info("开始评估...")
    results = evaluate_detailed(
        model, tokenizer, test_path, type_info['name'], device, args.max_samples
    )
    
    # 保存报告
    save_report(results, output_path, type_info, model_path, test_path)
    
    # 打印摘要
    print()
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"准确率: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"F1 (Macro): {results['f1_macro']:.4f}")
    print(f"F1 (Weighted): {results['f1_weighted']:.4f}")
    print("=" * 60)
    print(f"详细报告: {output_path}.json, {output_path}.md")
    print()


if __name__ == "__main__":
    main()
