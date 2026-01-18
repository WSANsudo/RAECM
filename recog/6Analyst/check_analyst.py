"""
结果校验Agent
对产品分析和用途分析的结果进行检查、校验和修正
验证置信度、证据可靠性、结果是否符合常识和事实
输出修正后的最终结果和评估统计报告
"""

import json
import time
from typing import Dict, List, Tuple, Optional

from .base_analyst import BaseAnalyst
from .config import BATCH_SIZE, MAX_INPUT_TOKENS
from .prompts import get_prompt_by_id


class CheckAnalyst(BaseAnalyst):
    """结果校验Agent"""
    
    # 默认提示词ID（可通过命令行参数覆盖）
    _prompt_id: str = "default"
    
    DEFAULT_SYSTEM_PROMPT = """Validate and correct network asset identification results.

CRITICAL: Return ONLY valid JSON array. No markdown, no explanation.
IMPORTANT: Each result MUST contain the EXACT "ip" from input. Return exactly N results for N input records.

Format: [{"ip": str, "validation_status": "verified|adjusted|rejected", "evidence_quality": "strong|moderate|weak|insufficient", "issues_found": [str], "original_confidence": float, "validated_confidence": float, "reasoning": str|null, "adjustments": {}}, ...]

=== ACCURACY EVALUATION LABELS (VALIDATE WITH EXTRA CARE) ===
These 3 fields are KEY METRICS - errors here directly impact accuracy score:
- vendor: Must be manufacturer name, NOT OS name
- os: Must include version when available, must be OS name NOT vendor
- type: Must be consistent with OS (RouterOS→router, JunOS→router, FortiOS→firewall)

=== LABEL FIELD COMMON MISTAKES TO CORRECT ===
[失败] vendor: "RouterOS" → WRONG (RouterOS is OS, vendor should be "MikroTik")
[失败] os: "MikroTik" → WRONG (MikroTik is vendor, os should be "RouterOS X.X.X")
[失败] os: "Juniper" → WRONG (Juniper is vendor, os should be "JunOS X.X")
[失败] os: "RouterOS" without version → Should include version if available
[失败] type: null when os is "RouterOS" → Should be "router"
[失败] type: "server" when os is "RouterOS" → Should be "router"
[OK] vendor: "MikroTik", os: "RouterOS 6.49.10", type: "router" → CORRECT

=== OTHER FIELD RULES ===
- model: [[name, conf], ...] or null (hardware SKU only)
- firmware: Full firmware name with version or null
- issues_found: [] if no issues

=== ADJUSTMENTS ===
- verified: adjustments = {}
- adjusted: {"os": "RouterOS 6.49.10", "type": "router", ...}
- rejected: {"vendor": null, "model": null, "confidence": <0.3}
"""
    
    AGENT_NAME = "CheckAnalyst"
    
    @classmethod
    def set_prompt_id(cls, prompt_id: str) -> None:
        """设置使用的提示词ID"""
        cls._prompt_id = prompt_id
    
    @property
    def SYSTEM_PROMPT(self) -> str:
        """获取当前使用的系统提示词"""
        # 优先使用实验模块设置的提示词
        if hasattr(self, '_exp_prompt') and self._exp_prompt:
            return self._exp_prompt
        if self._prompt_id == "default":
            return self.DEFAULT_SYSTEM_PROMPT
        custom_prompt = get_prompt_by_id("check", self._prompt_id)
        return custom_prompt if custom_prompt else self.DEFAULT_SYSTEM_PROMPT
    
    def __init__(self, input_path: str, output_path: str, final_output_path: str = None, model_name: str = None):
        """
        初始化校验器
        
        Args:
            input_path: 输入文件路径（汇总后的分析结果JSONL）
            output_path: 输出文件路径（校验详情JSONL）
            final_output_path: 最终修正结果路径
            model_name: 使用的模型名称（可选）
        """
        super().__init__(input_path, output_path, model_name)
        self.final_output_path = final_output_path
        self.evaluation_stats = {
            'total_records': 0,
            'verified_count': 0,
            'adjusted_count': 0,
            'rejected_count': 0,
            'evidence_quality': {
                'strong': 0,
                'moderate': 0,
                'weak': 0,
                'insufficient': 0
            },
            'avg_original_confidence': 0,
            'avg_validated_confidence': 0,
            'confidence_increased': 0,
            'confidence_decreased': 0,
            'confidence_unchanged': 0,
            'common_issues': {}
        }
    
    def load_records(self, max_count: int = None) -> List[Dict]:
        """加载汇总后的分析结果"""
        records = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_count is not None and i >= max_count:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    ip = obj.get('ip', f'unknown_{i}')
                    records.append({
                        'ip': ip,
                        'raw': line,
                        'data': obj
                    })
                except json.JSONDecodeError:
                    continue
        
        return records
    
    def build_prompt(self, batch: List[Dict], batch_id: int = None) -> List[Dict]:
        """构建校验提示"""
        input_text = "\n".join([r["raw"] for r in batch])
        
        # 添加批次ID到prompt中，要求返回时也包含批次ID
        batch_id_str = f"[BATCH_ID: {batch_id}]" if batch_id else ""
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"{batch_id_str}\nValidate and correct these {len(batch)} analysis results and return exactly {len(batch)} results:\n{input_text}"},
        ]
        
        return messages
    
    def parse_response(self, response: str, batch_id: int = None, expected_ips: List[str] = None) -> List[Dict]:
        """解析API响应并验证IP匹配"""
        results = self.try_parse_json(response, batch_id)
        
        # 验证返回的IP是否匹配输入
        if results and expected_ips:
            returned_ips = {r.get('ip') for r in results if r.get('ip')}
            expected_set = set(expected_ips)
            
            if returned_ips and expected_set:
                match_count = len(returned_ips & expected_set)
                match_rate = match_count / len(expected_set)
                
                from .utils.logger import log_error, log_warning, log_info
                
                # IP修正逻辑：当batch大小为3且恰好有1条IP不匹配时，自动修正
                if len(expected_ips) == 3 and len(results) == 3 and match_count == 2:
                    only_in_expected = list(expected_set - returned_ips)
                    only_in_returned = list(returned_ips - expected_set)
                    
                    if len(only_in_expected) == 1 and len(only_in_returned) == 1:
                        wrong_ip = only_in_returned[0]
                        correct_ip = only_in_expected[0]
                        
                        # 找到并修正错误的IP
                        for r in results:
                            if r.get('ip') == wrong_ip:
                                r['ip'] = correct_ip
                                log_info(f"[CheckAnalyst] 批次{batch_id}: 自动修正IP {wrong_ip} -> {correct_ip}")
                                break
                        
                        return results
                
                if match_rate < 0.5:  # 匹配率低于50%
                    log_error(f"[CheckAnalyst] 批次{batch_id}: IP匹配率仅{match_rate*100:.1f}% ({match_count}/{len(expected_set)}), 可能是批次混淆")
                    log_error(f"  期望IP样本: {list(expected_set)[:3]}")
                    log_error(f"  返回IP样本: {list(returned_ips)[:3]}")
                    # 记录不匹配的IP
                    only_in_expected = list(expected_set - returned_ips)[:5]
                    only_in_returned = list(returned_ips - expected_set)[:5]
                    if only_in_expected:
                        log_error(f"  仅在期望中: {only_in_expected}")
                    if only_in_returned:
                        log_error(f"  仅在返回中: {only_in_returned}")
                    return []  # 拒绝使用错误的结果
                elif match_rate < 0.9:  # 匹配率低于90%
                    log_warning(f"[CheckAnalyst] 批次{batch_id}: IP匹配率{match_rate*100:.1f}% ({match_count}/{len(expected_set)}), 部分IP不匹配")
        
        return results
    
    def _update_evaluation_stats(self, result: Dict, original_data: Dict) -> None:
        """更新评估统计"""
        self.evaluation_stats['total_records'] += 1
        
        # 统计校验状态
        status = result.get('validation_status', 'unknown')
        if status == 'verified':
            self.evaluation_stats['verified_count'] += 1
        elif status == 'adjusted':
            self.evaluation_stats['adjusted_count'] += 1
        elif status == 'rejected':
            self.evaluation_stats['rejected_count'] += 1
        
        # 统计证据质量
        quality = result.get('evidence_quality', 'unknown')
        if quality in self.evaluation_stats['evidence_quality']:
            self.evaluation_stats['evidence_quality'][quality] += 1
        
        # 统计置信度变化
        orig_conf = result.get('original_confidence', 0)
        valid_conf = result.get('validated_confidence', 0)
        
        self.evaluation_stats['avg_original_confidence'] += orig_conf
        self.evaluation_stats['avg_validated_confidence'] += valid_conf
        
        if valid_conf > orig_conf + 0.01:
            self.evaluation_stats['confidence_increased'] += 1
        elif valid_conf < orig_conf - 0.01:
            self.evaluation_stats['confidence_decreased'] += 1
        else:
            self.evaluation_stats['confidence_unchanged'] += 1
        
        # 统计常见问题
        for issue in result.get('issues_found', []):
            issue_key = issue[:50]  # 截断长问题
            self.evaluation_stats['common_issues'][issue_key] = \
                self.evaluation_stats['common_issues'].get(issue_key, 0) + 1
    
    def _build_final_result(self, result: Dict, original_data: Dict) -> Dict:
        """构建最终修正结果，将adjustments应用到原始数据"""
        adjustments = result.get('adjustments', {})
        
        # 从原始数据构建基础结果
        final_result = {
            'ip': result.get('ip') or original_data.get('ip'),
            # 产品信息
            'vendor': original_data.get('vendor'),
            'model': original_data.get('model'),
            'os': original_data.get('os'),
            'firmware': original_data.get('firmware'),
            'type': original_data.get('type'),
            'result_type': original_data.get('result_type'),
            'confidence': original_data.get('confidence', 0),
            'evidence': original_data.get('evidence', []),
            # 校验元数据
            'validation_status': result.get('validation_status'),
            'evidence_quality': result.get('evidence_quality')
        }
        
        # 应用adjustments中的修改
        for key, value in adjustments.items():
            if key in final_result:
                final_result[key] = value
        
        # 标准化字段值
        final_result = self._normalize_fields(final_result)
        
        return final_result
    
    def _normalize_fields(self, record: Dict) -> Dict:
        """
        标准化字段值：
        - 所有属性字段未知时统一为 null
        - model字段为列表格式 [[name, conf], ...]
        """
        # 关键结果字段（字符串）：空字符串/"unknown" -> null
        key_str_fields = ['vendor', 'os', 'firmware', 'type', 'result_type']
        for field in key_str_fields:
            val = record.get(field)
            if val is None or val == '' or val == 'null' or val == 'unknown':
                record[field] = None
        
        # model字段特殊处理：应为列表格式
        model_val = record.get('model')
        if model_val is None or model_val == '' or model_val == 'unknown':
            record['model'] = None
        elif isinstance(model_val, str):
            # 兼容旧格式：字符串转为列表
            record['model'] = [[model_val, 0.5]]
        elif isinstance(model_val, list):
            # 验证列表格式
            if len(model_val) == 0:
                record['model'] = None
            else:
                # 确保每个元素是 [name, conf] 格式
                normalized = []
                for item in model_val:
                    if isinstance(item, list) and len(item) >= 2:
                        normalized.append([str(item[0]), float(item[1])])
                    elif isinstance(item, str):
                        normalized.append([item, 0.5])
                record['model'] = normalized if normalized else None
        
        # 说明性字段（字符串）：空字符串/"unknown" -> null
        desc_str_fields = ['conclusion']
        for field in desc_str_fields:
            val = record.get(field)
            if val == '' or val == 'unknown':
                record[field] = None
        
        # 说明性字段（数组）：空数组 -> null
        desc_arr_fields = ['evidence']
        for field in desc_arr_fields:
            val = record.get(field)
            if val is None or (isinstance(val, list) and len(val) == 0):
                record[field] = None
        
        return record
    
    def _print_result(self, result: Dict) -> None:
        """打印单条校验结果"""
        ip = result.get('ip', 'unknown')
        status = result.get('validation_status', 'unknown')
        orig_conf = result.get('original_confidence', 0)
        valid_conf = result.get('validated_confidence', 0)
        quality = result.get('evidence_quality', 'unknown')
        issues = result.get('issues_found', [])
        
        status_icon = {'verified': '[OK]', 'adjusted': '~', 'rejected': '[FAIL]'}.get(status, '?')
        
        print(f"  [{status_icon}] {ip}: {status} (置信度: {orig_conf:.2f} → {valid_conf:.2f})")
        print(f"       证据质量: {quality}")
        if issues:
            print(f"       问题: {', '.join(issues[:3])}")
    
    def run(self, max_records: int = None) -> Tuple[Dict, Dict]:
        """
        执行校验主循环
        
        Returns:
            (统计信息, 评估统计)
        """
        from .config import MAX_RECORDS
        start_time = time.time()
        
        # 加载数据
        if max_records is None:
            max_records = MAX_RECORDS
        records = self.load_records(max_records)
        self.stats['total_records'] = len(records)
        
        print(f"加载 {len(records)} 条记录进行校验")
        
        all_check_results = []
        all_final_results = []
        
        # 创建IP到原始数据的映射
        ip_to_original = {r['ip']: r['data'] for r in records}
        
        # 分批处理
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            print(f"\n=== 校验批次 {batch_num}: 处理 {len(batch)} 条 ===")
            for r in batch:
                print(f"  - {r['ip']}")
            
            results, batch_stats = self.process_batch(batch)
            
            print(f"  Token数: {batch_stats.get('input_token_count', 0)}/{MAX_INPUT_TOKENS}")
            
            if results:
                for res in results:
                    ip = res.get('ip')
                    original_data = ip_to_original.get(ip, {})
                    
                    # 更新评估统计
                    self._update_evaluation_stats(res, original_data)
                    
                    # 保存校验详情
                    all_check_results.append(res)
                    
                    # 构建最终结果
                    final_result = self._build_final_result(res, original_data)
                    all_final_results.append(final_result)
                    
                    self.stats['successful_records'] += 1
                    self._print_result(res)
            else:
                self.stats['failed_records'] += len(batch)
            
            self.stats['processed_records'] += len(batch)
        
        # 计算平均置信度
        if self.evaluation_stats['total_records'] > 0:
            self.evaluation_stats['avg_original_confidence'] /= self.evaluation_stats['total_records']
            self.evaluation_stats['avg_validated_confidence'] /= self.evaluation_stats['total_records']
        
        # 保存校验详情
        if all_check_results:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for r in all_check_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"\n校验详情已保存: {self.output_path}")
        
        # 保存最终修正结果
        if all_final_results and self.final_output_path:
            with open(self.final_output_path, 'w', encoding='utf-8') as f:
                for r in all_final_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"最终结果已保存: {self.final_output_path}")
        
        self.stats['execution_time_seconds'] = time.time() - start_time
        
        return self.stats, self.evaluation_stats
    
    def print_evaluation_report(self) -> None:
        """打印评估统计报告"""
        stats = self.evaluation_stats
        total = stats['total_records']
        
        if total == 0:
            print("无校验记录")
            return
        
        print("\n" + "=" * 60)
        print("评估统计报告")
        print("=" * 60)
        
        print(f"\n校验状态分布:")
        print(f"  [OK] 验证通过: {stats['verified_count']} ({stats['verified_count']/total*100:.1f}%)")
        print(f"  ~ 已调整:   {stats['adjusted_count']} ({stats['adjusted_count']/total*100:.1f}%)")
        print(f"  [FAIL] 已拒绝:   {stats['rejected_count']} ({stats['rejected_count']/total*100:.1f}%)")
        
        print(f"\n证据质量分布:")
        for quality, count in stats['evidence_quality'].items():
            print(f"  {quality}: {count} ({count/total*100:.1f}%)")
        
        print(f"\n置信度变化:")
        print(f"  平均原始置信度: {stats['avg_original_confidence']:.3f}")
        print(f"  平均校验置信度: {stats['avg_validated_confidence']:.3f}")
        print(f"  置信度提升: {stats['confidence_increased']} 条")
        print(f"  置信度降低: {stats['confidence_decreased']} 条")
        print(f"  置信度不变: {stats['confidence_unchanged']} 条")
        
        if stats['common_issues']:
            print(f"\n常见问题 (Top 5):")
            sorted_issues = sorted(stats['common_issues'].items(), key=lambda x: -x[1])[:5]
            for issue, count in sorted_issues:
                print(f"  - {issue}: {count} 次")
