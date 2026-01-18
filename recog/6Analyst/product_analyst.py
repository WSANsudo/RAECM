"""
产品型号分析Agent
识别设备厂商和型号
"""

import json
from typing import Dict, List, Optional

from .base_analyst import BaseAnalyst
from .prompts import get_prompt_by_id


class ProductAnalyst(BaseAnalyst):
    """产品型号分析Agent"""
    
    # 默认提示词ID（可通过命令行参数覆盖）
    _prompt_id: str = "default"
    
    DEFAULT_SYSTEM_PROMPT = """Analyze network scan records to identify device vendor, model, OS and firmware.

CRITICAL: Return ONLY valid JSON array. No markdown, no explanation.
IMPORTANT: Each result MUST contain the EXACT "ip" from input. Return exactly N results for N input records.

Format: [{"ip": str, "vendor": str|null, "model": [[str,float],...]|null, "os": str|null, "firmware": str|null, "type": str|null, "result_type": "direct"|"inferred"|null, "confidence": 0-1, "evidence": [{"src": str, "val": str, "weight": 0-1}]|null}, ...]

=== ACCURACY EVALUATION LABELS (HIGHEST PRIORITY) ===
These 3 fields are KEY METRICS for accuracy evaluation - extract with extra care:
- vendor: Hardware manufacturer (e.g., "MikroTik", "Cisco", "Juniper", "Huawei")
- os: Operating system WITH version (e.g., "RouterOS 6.49.10", "JunOS 21.4R1", "IOS-XE 17.3")
- type: Device type - MUST be one of: router|switch|server|firewall|camera|nas|printer|iot|appliance|null

=== DEVICE TYPE STANDARDIZATION (IMPORTANT) ===
Use ONLY standard device types. Map ambiguous types to standard ones:
- management_interface → router
- management → router  
- web_interface → server
- network_device → router (if routing features) or switch
- gateway → router
- access_point/ap/wap → router
- broadband_router → router

=== OTHER FIELDS ===
- model: Hardware model as [[name, confidence], ...] or null (SKU only, NOT os/firmware)
- firmware: Full firmware name with version (e.g., "MikroTik 7.16.1")
- evidence: [{src, val, weight}] - source field, value, confidence weight

=== EXTRACTION RULES ===
- os: MUST include version when available (e.g., "RouterOS 6.49.10" not just "RouterOS")
- vendor: Manufacturer name only, NOT OS name (MikroTik is vendor, RouterOS is OS)
- type: Infer from OS/services (RouterOS→router, JunOS→router, FortiOS→firewall)
- model: Hardware SKU only, NOT os/firmware
- Use null for unknown fields, never "unknown"
"""
    
    AGENT_NAME = "ProductAnalyst"
    
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
        custom_prompt = get_prompt_by_id("product", self._prompt_id)
        return custom_prompt if custom_prompt else self.DEFAULT_SYSTEM_PROMPT
    
    def build_prompt(self, batch: List[Dict], batch_id: int = None) -> List[Dict]:
        """构建分析提示"""
        input_text = "\n".join([r["raw"] for r in batch])
        
        # 添加批次ID到prompt中，要求返回时也包含批次ID
        batch_id_str = f"[BATCH_ID: {batch_id}]" if batch_id else ""
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"{batch_id_str}\nAnalyze these {len(batch)} records and return exactly {len(batch)} results:\n{input_text}"},
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
                                log_info(f"[ProductAnalyst] 批次{batch_id}: 自动修正IP {wrong_ip} -> {correct_ip}")
                                break
                        
                        return results
                
                if match_rate < 0.5:  # 匹配率低于50%
                    log_error(f"[ProductAnalyst] 批次{batch_id}: IP匹配率仅{match_rate*100:.1f}% ({match_count}/{len(expected_set)}), 可能是批次混淆")
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
                    log_warning(f"[ProductAnalyst] 批次{batch_id}: IP匹配率{match_rate*100:.1f}% ({match_count}/{len(expected_set)}), 部分IP不匹配")
        
        return results
    
    def _print_result(self, result: Dict) -> None:
        """打印单条结果"""
        ip = result.get('ip', 'unknown')
        vendor = result.get('vendor') or '未知'
        model_list = result.get('model')
        os_info = result.get('os') or ''
        firmware = result.get('firmware') or ''
        device_type = result.get('type', 'unknown')
        confidence = result.get('confidence', 0)
        
        # 格式化model列表
        if model_list and isinstance(model_list, list) and len(model_list) > 0:
            top_model = model_list[0]
            if isinstance(top_model, list) and len(top_model) >= 2:
                model_str = f"{top_model[0]}({top_model[1]:.1f})"
                if len(model_list) > 1:
                    model_str += f" +{len(model_list)-1}"
            else:
                model_str = str(top_model)
        else:
            model_str = '未知'
        
        ver_str = f" [{os_info}]" if os_info else ""
        if firmware:
            ver_str += f" fw:{firmware}"
        print(f"  [OK] {ip}: {vendor} {model_str}{ver_str} ({device_type}) 置信度:{confidence}")
