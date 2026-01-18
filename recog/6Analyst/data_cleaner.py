"""
数据清洗模块
负责过滤无用字段、压缩数据、减少Token消耗
"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional

from .config import (
    REMOVE_TOP_FIELDS, REMOVE_SERVICE_FIELDS,
    HTTP_ERROR_PATTERNS, DEFAULT_PAGE_PATTERNS,
    COMMON_KEX, COMMON_HOST_KEY, COMMON_ENCRYPTION, COMMON_MAC,
    SPECIAL_ALGORITHMS, TLS_REMOVE_PATTERNS, MYSQL_REMOVE_PATTERNS,
    SSH_REMOVE_PATTERNS
)
from .utils.html_extractor import simplify_html


# ========= HTTP Banner 清洗 =========

# CSS 属性模式（需要删除）
CSS_PROPERTY_PATTERNS = [
    r'color\s*:\s*[^;}\s]+',
    r'font-family\s*:\s*[^;}\s]+',
    r'font-size\s*:\s*[^;}\s]+',
    r'background\s*:\s*[^;}\s]+',
    r'background-color\s*:\s*[^;}\s]+',
    r'background-image\s*:\s*[^;}\s]+',
    r'border\s*:\s*[^;}\s]+',
    r'margin\s*:\s*[^;}\s]+',
    r'padding\s*:\s*[^;}\s]+',
    r'width\s*:\s*[^;}\s]+',
    r'height\s*:\s*[^;}\s]+',
    r'display\s*:\s*[^;}\s]+',
    r'position\s*:\s*[^;}\s]+',
    r'float\s*:\s*[^;}\s]+',
    r'text-align\s*:\s*[^;}\s]+',
    r'line-height\s*:\s*[^;}\s]+',
    r'overflow\s*:\s*[^;}\s]+',
]

# 十六进制颜色值模式
HEX_COLOR_PATTERN = r'#[0-9a-fA-F]{3,8}\b'

# CSS 类名和 ID 引用模式（在 JS/CSS 中）
CSS_CLASS_ID_PATTERN = r'\.[\w-]+\s*\{|#[\w-]+\s*\{'

# 网络设备厂商关键词（用于从 HTML 中提取）
VENDOR_KEYWORDS = [
    'cisco', 'huawei', 'h3c', 'zte', 'tp-link', 'tplink', 'd-link', 'dlink',
    'netgear', 'asus', 'linksys', 'mikrotik', 'routeros', 'juniper', 'fortinet',
    'ubiquiti', 'aruba', 'ruckus', 'meraki', 'paloalto', 'sonicwall', 'watchguard',
    'sophos', 'barracuda', 'f5', 'a10', 'radware', 'citrix', 'vmware', 'dell',
    'hp', 'hpe', 'ibm', 'lenovo', 'supermicro', 'hikvision', 'dahua', 'uniview',
    'ruijie', 'sangfor', 'hillstone', 'venustech', 'nsfocus', 'dptech', 'maipu',
    'bdcom', 'raisecom', 'fiberhome', 'ericsson', 'nokia', 'alcatel', 'siemens',
    'schneider', 'rockwell', 'honeywell', 'emerson', 'abb', 'ge', 'mitsubishi',
    'keenetic', 'openwrt', 'dd-wrt', 'tomato', 'asuswrt', 'padavan', 'lede',
    'synology', 'qnap', 'buffalo', 'western digital', 'seagate', 'drobo'
]


def extract_vendor_from_html(html_content: str) -> List[str]:
    """
    从 HTML 内容中提取厂商关键词
    检查 href, src, alt, title 等属性中的厂商信息
    """
    found_vendors = []
    html_lower = html_content.lower()
    
    for vendor in VENDOR_KEYWORDS:
        if vendor in html_lower:
            # 检查是否在有意义的上下文中（不是在 UA 检测代码中）
            # 查找所有出现位置
            pos = 0
            while True:
                pos = html_lower.find(vendor, pos)
                if pos == -1:
                    break
                # 获取上下文
                context_start = max(0, pos - 30)
                context_end = min(len(html_lower), pos + len(vendor) + 30)
                context = html_lower[context_start:context_end]
                
                # 排除 UA 检测代码中的出现
                if 'useragent' not in context and 'navigator' not in context and '.test(' not in context:
                    # 检查是否在 href, src, title, alt 等属性中
                    if any(attr in context for attr in ['href=', 'src=', 'title=', 'alt=', 'content=', 'name=']):
                        found_vendors.append(vendor)
                        break
                    # 检查是否在文本内容中（标题、版权等）
                    if any(marker in context for marker in ['<title', 'copyright', '©', 'powered by', 'welcome to']):
                        found_vendors.append(vendor)
                        break
                pos += 1
    
    return list(set(found_vendors))


def extract_js_string_literals(js_content: str) -> List[str]:
    """
    从 JavaScript 代码中提取字符串字面量
    可能包含设备指纹信息（如型号、版本等）
    """
    literals = []
    
    # 匹配双引号字符串
    double_quote_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
    # 匹配单引号字符串
    single_quote_pattern = r"'([^'\\]*(?:\\.[^'\\]*)*)'"
    
    for pattern in [double_quote_pattern, single_quote_pattern]:
        matches = re.findall(pattern, js_content)
        for match in matches:
            # 过滤掉太短或纯数字/符号的字符串
            if len(match) >= 3 and not re.match(r'^[\d\s\.\,\-\+\*\/\=\|\&\^\%\$\#\@\!\~\`\(\)\[\]\{\}\<\>\;]+$', match):
                # 过滤掉常见的无意义字符串
                if match.lower() not in {'true', 'false', 'null', 'undefined', 'none', 'click', 'submit', 'button'}:
                    # 过滤掉 CSS 选择器和 DOM 操作
                    if not re.match(r'^[\.\#\[\]]', match) and not match.startswith('data-'):
                        # 保留可能包含设备信息的字符串
                        if len(match) <= 100:
                            literals.append(match)
    
    # 去重并限制数量
    unique_literals = list(set(literals))[:20]
    return unique_literals


def clean_http_banner(banner: str) -> str:
    """
    清洗 HTTP Banner 中的 HTML/CSS/JavaScript 内容
    
    处理策略：
    1. 清洗前先提取所有字符串中的厂商关键词
    2. 分离 HTTP 头部和 Body
    3. 保留有价值的 HTTP 头部（Server, X-Powered-By 等）
    4. 删除 <style>...</style> 块
    5. 从 <script>...</script> 块中提取字符串字面量
    6. 删除 CSS 属性、十六进制颜色值
    7. 删除 HTML 结构标签
    8. 清理多余空白字符
    """
    if not banner or not isinstance(banner, str):
        return banner
    
    # 检测是否包含 HTTP 响应（包含 HTML 内容）
    if '<html' not in banner.lower() and '<script' not in banner.lower() and '<style' not in banner.lower():
        # 不是 HTML 内容，只做基本清理
        return banner
    
    result_parts = []
    
    # 0. 清洗前先提取所有字符串中的厂商关键词
    vendors = extract_vendor_from_html(banner)
    if vendors:
        result_parts.append(f"vendors:{','.join(vendors)}")
    
    # 1. 尝试分离 HTTP 头部和 Body
    # HTTP 头部和 Body 之间通常有空行或 <!DOCTYPE 或 <html
    http_header = ""
    html_body = banner
    
    # 查找 HTML 开始位置
    html_start_patterns = [r'<!DOCTYPE', r'<html', r'<HTML']
    html_start_pos = len(banner)
    for pattern in html_start_patterns:
        match = re.search(pattern, banner, re.IGNORECASE)
        if match and match.start() < html_start_pos:
            html_start_pos = match.start()
    
    if html_start_pos > 0 and html_start_pos < len(banner):
        http_header = banner[:html_start_pos].strip()
        html_body = banner[html_start_pos:]
    
    # 2. 保留有价值的 HTTP 头部
    if http_header:
        valuable_headers = []
        header_lines = http_header.split('\n')
        for line in header_lines:
            line = line.strip()
            if not line:
                continue
            # 保留状态行
            if line.startswith('HTTP/'):
                valuable_headers.append(line)
            # 保留有价值的头部
            elif any(h in line.lower() for h in ['server:', 'x-powered-by:', 'x-aspnet', 'www-authenticate:', 
                                                   'set-cookie:', 'x-frame-options:', 'content-security-policy:']):
                valuable_headers.append(line)
        if valuable_headers:
            result_parts.append(' '.join(valuable_headers))
    
    # 3. 提取 title（在删除 head 之前）
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_body, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        if title and len(title) > 2:
            result_parts.append(f"title:{title}")
    
    # 4. 从 <script>...</script> 块中提取字符串字面量（在删除之前）
    script_pattern = r'<script[^>]*>(.*?)</script>'
    script_matches = re.findall(script_pattern, html_body, flags=re.DOTALL | re.IGNORECASE)
    
    js_literals = []
    for script_content in script_matches:
        literals = extract_js_string_literals(script_content)
        js_literals.extend(literals)
    
    # 5. 删除 <head>...</head> 块（包含 style, script, meta, link 等）
    html_body = re.sub(r'<head[^>]*>.*?</head>', '', html_body, flags=re.DOTALL | re.IGNORECASE)
    
    # 6. 删除 <style>...</style> 块（CSS 无指纹价值）- 可能在 body 中也有
    html_body = re.sub(r'<style[^>]*>.*?</style>', '', html_body, flags=re.DOTALL | re.IGNORECASE)
    
    # 7. 删除 <script>...</script> 块
    html_body = re.sub(r'<script[^>]*>.*?</script>', '', html_body, flags=re.DOTALL | re.IGNORECASE)
    
    # 8. 删除 CSS 属性（可能在 style 属性中）
    for pattern in CSS_PROPERTY_PATTERNS:
        html_body = re.sub(pattern, '', html_body, flags=re.IGNORECASE)
    
    # 删除十六进制颜色值
    html_body = re.sub(HEX_COLOR_PATTERN, '', html_body)
    
    # 删除 CSS 类名和 ID 定义
    html_body = re.sub(CSS_CLASS_ID_PATTERN, '', html_body)
    
    # 9. 删除 meta 标签
    html_body = re.sub(r'<meta[^>]*/?>', '', html_body, flags=re.IGNORECASE)
    
    # 删除 link 标签
    html_body = re.sub(r'<link[^>]*/?>', '', html_body, flags=re.IGNORECASE)
    
    # 删除常见结构标签（保留内容）
    structure_tags = ['html', 'body', 'div', 'span', 'head', 'header', 'footer', 'nav', 
                      'section', 'article', 'aside', 'main', 'table', 'tr', 'td', 'th',
                      'thead', 'tbody', 'ul', 'ol', 'li', 'form', 'fieldset', 'legend',
                      'label', 'input', 'select', 'option', 'textarea', 'button', 'iframe',
                      'noscript', 'br', 'hr', 'p', 'a', 'img', 'i', 'b', 'strong', 'em',
                      'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    for tag in structure_tags:
        html_body = re.sub(rf'<{tag}[^>]*>', '', html_body, flags=re.IGNORECASE)
        html_body = re.sub(rf'</{tag}>', '', html_body, flags=re.IGNORECASE)
    
    # 删除注释
    html_body = re.sub(r'<!--.*?-->', '', html_body, flags=re.DOTALL)
    html_body = re.sub(r'/\*.*?\*/', '', html_body, flags=re.DOTALL)
    html_body = re.sub(r'//[^\n]*', '', html_body)
    
    # 删除 DOCTYPE
    html_body = re.sub(r'<!DOCTYPE[^>]*>', '', html_body, flags=re.IGNORECASE)
    
    # 10. 删除残留的 CSS/JS 片段（花括号块、圆括号块等）
    html_body = re.sub(r'\{[^}]*\}', '', html_body)
    
    # 11. 清理多余空白字符
    html_body = html_body.replace('\t', ' ')
    html_body = html_body.replace('\r\n', ' ')
    html_body = html_body.replace('\r', ' ')
    html_body = html_body.replace('\n', ' ')
    # 合并多个空格
    while '  ' in html_body:
        html_body = html_body.replace('  ', ' ')
    html_body = html_body.strip()
    
    # 12. 添加 JS 字符串字面量（可能包含设备信息）
    if js_literals:
        # 过滤并限制
        filtered_literals = [l for l in js_literals if len(l) >= 4 and len(l) <= 50][:10]
        if filtered_literals:
            result_parts.append(f"js_strings:{','.join(filtered_literals)}")
    
    # 13. 添加清理后的 HTML 内容（如果有价值）
    if html_body and len(html_body) > 10:
        # 再次清理空白
        while '  ' in html_body:
            html_body = html_body.replace('  ', ' ')
        html_body = html_body.strip()
        
        if html_body and len(html_body) > 10:
            # 截断过长内容
            if len(html_body) > 500:
                html_body = html_body[:500] + "..."
            result_parts.append(f"content:{html_body}")
    
    # 组合结果
    if result_parts:
        return ' | '.join(result_parts)
    
    # 如果清理后没有内容，返回原始 banner 的前 200 字符
    return banner[:200] if len(banner) > 200 else banner


# ========= 字符串清理 =========

# 恶意脚本/加密货币挖矿相关的关键词（会触发Windows Defender）
MALWARE_KEYWORDS = [
    'coinhive', 'CoinHive', 'cryptominer', 'cryptonight',
    'minero', 'coin-hive', 'jsecoin', 'cryptoloot',
    'webminer', 'deepminer', 'coinimp'
]


def sanitize_malware_content(s: str) -> str:
    """
    清理可能触发安全软件的恶意内容
    将恶意脚本关键词替换为安全标记
    """
    if not isinstance(s, str):
        return s
    
    result = s
    for keyword in MALWARE_KEYWORDS:
        if keyword in result:
            # 替换为安全标记，保留长度信息以便分析
            result = result.replace(keyword, f'[FILTERED:{len(keyword)}]')
    
    return result


def clean_string(s: str) -> str:
    """
    清理字符串中的多余空白字符和特殊Unicode字符
    - 移除Unicode行分隔符(LS U+2028)和段落分隔符(PS U+2029)
    - 将换行符替换为空格
    - 合并连续空格为单个空格
    - 过滤恶意脚本内容
    """
    if not isinstance(s, str):
        return s
    # 先过滤恶意内容
    s = sanitize_malware_content(s)
    # 移除Unicode行分隔符和段落分隔符（会导致JSONL解析问题）
    s = s.replace('\u2028', ' ')  # Line Separator
    s = s.replace('\u2029', ' ')  # Paragraph Separator
    # 移除其他可能导致问题的控制字符
    s = s.replace('\x00', '')     # NUL
    s = s.replace('\x0b', ' ')    # Vertical Tab
    s = s.replace('\x0c', ' ')    # Form Feed
    s = s.replace('\x85', ' ')    # Next Line (NEL)
    # 替换换行符
    s = s.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s.strip()


def clean_value(value: Any) -> Any:
    """递归清理值中的多余空白字符"""
    if isinstance(value, str):
        return clean_string(value)
    elif isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_value(item) for item in value]
    return value


# ========= HTTP错误和默认页面检测 =========

def is_http_error(body: str, title: str = '') -> Optional[str]:
    """
    检测是否为HTTP错误响应
    返回错误类型名称，如果不是错误则返回None
    """
    text = f"{title} {body}"
    for pattern, name in HTTP_ERROR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return None


def get_default_page_type(title: str, body: str = '') -> Optional[str]:
    """
    检测是否为默认页面
    返回默认页面标签，如果不是默认页面则返回None
    """
    text = f"{title} {body[:500]}"
    for pattern, label in DEFAULT_PAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


# ========= Body清洗 =========

def is_already_simplified(body: str) -> bool:
    """
    检测Body是否已经是简化后的格式
    简化格式特征：
    1. 包含 "title:" 或 "img:" 或 "links:" 或 "forms:" 或 "text:" 等标记
    2. 不包含HTML标签（如 <html>, <body>, <div> 等）
    """
    if not body or not isinstance(body, str):
        return False
    
    # 检查是否包含简化格式的标记
    simplified_markers = ['title:', 'img:', 'links:', 'forms:', 'text:', 'description:', 'generator:']
    has_marker = any(marker in body for marker in simplified_markers)
    
    # 检查是否包含HTML标签（原始HTML的特征）
    html_patterns = ['<html', '<body', '<div', '<head', '<script', '<style', '<!DOCTYPE', '<meta']
    has_html = any(pattern.lower() in body.lower() for pattern in html_patterns)
    
    # 如果有简化标记且没有HTML标签，认为是已简化的格式
    if has_marker and not has_html:
        return True
    
    # 如果是默认页面标签格式
    if body.startswith('[DEFAULT:') and body.endswith(']'):
        return True
    
    return False


def clean_body(body: str, title: str = '') -> Optional[str]:
    """
    清洗Body内容
    - 已简化的内容直接返回
    - HTTP错误响应返回None（过滤）
    - 默认页面返回标签字符串
    - 有价值的HTML返回简化后的紧凑字符串
    - 无价值内容返回None
    """
    if not body or len(body) < 20:
        return None
    
    # 检测是否已经是简化后的格式，如果是则直接返回
    if is_already_simplified(body):
        return body
    
    # 检测HTTP错误
    if is_http_error(body, title):
        return None
    
    # 检测默认页面
    default_type = get_default_page_type(title, body)
    if default_type:
        return default_type
    
    # 简化HTML内容（返回紧凑字符串格式）
    simplified = simplify_html(body)
    
    # 如果简化后无内容，返回None
    if not simplified:
        return None
    
    # 如果有title参数且simplified中没有title信息，添加到开头
    if title and 'title:' not in simplified:
        simplified = f"title:{title} ; {simplified}"
    
    return simplified


# ========= SSH Banner压缩 =========

def compress_algorithms(algo_line: str, common_set: set) -> Tuple[bool, Optional[str]]:
    """
    压缩算法列表
    返回: (是否为常见配置, 特殊算法标记)
    """
    algos = set(a.strip() for a in algo_line.split(",") if a.strip())
    missing_common = common_set - algos
    extra_algos = algos - common_set
    special_found = extra_algos & SPECIAL_ALGORITHMS
    is_common = len(missing_common) <= 2
    
    special_parts = []
    if special_found:
        for alg in special_found:
            if "sntrup761" in alg:
                special_parts.append("post-quantum")
            elif "kex-strict" in alg:
                special_parts.append("terrapin-fix")
            elif "group1-sha1" in alg or "group-exchange-sha1" in alg:
                special_parts.append("legacy-dh")
            elif "ssh-dss" in alg:
                special_parts.append("dss")
            elif "3des" in alg:
                special_parts.append("3des")
            elif "-cbc" in alg:
                special_parts.append("cbc-mode")
    
    if not is_common and missing_common:
        if "curve25519-sha256" in missing_common:
            special_parts.append("no-curve25519")
        if "chacha20-poly1305@openssh.com" in missing_common:
            special_parts.append("no-chacha20")
        if "ssh-ed25519" in missing_common:
            special_parts.append("no-ed25519")
    
    return is_common, ",".join(special_parts) if special_parts else None


def clean_ssh_banner(banner: str) -> Tuple[str, Optional[Dict]]:
    """
    清洗SSH Banner，压缩算法列表
    返回: (清洗后的Banner, 算法摘要字典)
    """
    lines = banner.split("\n")
    cleaned_lines = []
    algo_summary = {}
    
    for line in lines:
        # 跳过需要删除的行
        skip = any(re.match(p, line) for p in SSH_REMOVE_PATTERNS)
        if skip:
            continue
        
        # 处理各类算法行
        if line.startswith("Kex Algorithms:"):
            is_common, special = compress_algorithms(
                line.replace("Kex Algorithms:", "").strip(), COMMON_KEX
            )
            algo_summary["kex_common"] = is_common
            if special:
                algo_summary["kex_special"] = special
            continue
            
        if line.startswith("Server Host Key Algorithms:"):
            is_common, special = compress_algorithms(
                line.replace("Server Host Key Algorithms:", "").strip(), COMMON_HOST_KEY
            )
            algo_summary["hostkey_common"] = is_common
            if special:
                algo_summary["hostkey_special"] = special
            continue
            
        if line.startswith("Encryption Algorithms:"):
            is_common, special = compress_algorithms(
                line.replace("Encryption Algorithms:", "").strip(), COMMON_ENCRYPTION
            )
            algo_summary["enc_common"] = is_common
            if special:
                algo_summary["enc_special"] = special
            continue
            
        if line.startswith("MAC Algorithms:"):
            is_common, special = compress_algorithms(
                line.replace("MAC Algorithms:", "").strip(), COMMON_MAC
            )
            algo_summary["mac_common"] = is_common
            if special:
                algo_summary["mac_special"] = special
            continue
            
        if line.startswith("Compression Algorithms:"):
            continue
            
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines), algo_summary if algo_summary else None


def clean_tls_banner(banner: str) -> str:
    """清洗TLS Banner，删除modulus、exponent等冗余信息"""
    lines = banner.split("\n")
    cleaned_lines = []
    skip_until_next = False
    
    for line in lines:
        skip = any(re.match(p, line) for p in TLS_REMOVE_PATTERNS)
        
        if "modulus:" in line:
            skip_until_next = True
            skip = True
        elif skip_until_next:
            if line.strip() and not line.startswith(" ") and ":" in line:
                skip_until_next = False
            else:
                skip = True
        
        if not skip:
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)


def clean_mysql_banner(banner: str) -> str:
    """清洗MySQL Banner，删除Capability Flags块"""
    result = banner
    for pattern in MYSQL_REMOVE_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.MULTILINE)
    return result.strip()


# ========= 服务和记录清洗 =========

def is_service_already_cleaned(service_data: Dict) -> bool:
    """
    检测服务数据是否已经被压缩过
    通过多种特征判断：
    1. SSH服务：存在 SSH_Algorithms 字段
    2. HTTP服务：Body 已经是简化格式（包含 title:, img:, links: 等标记，且不含HTML标签）
    3. 通用：不存在原始HTML标签（如 <html>, <body>, <div> 等）
    """
    # 检查 SSH_Algorithms 字段（SSH服务的压缩标志）
    if "SSH_Algorithms" in service_data:
        return True
    
    # 检查 Body 是否已经是简化格式
    body = service_data.get("Body") or service_data.get("Html Body")
    if body and isinstance(body, str):
        # 如果 Body 已经是简化格式，认为数据已压缩
        if is_already_simplified(body):
            return True
    
    return False


def clean_service(service_name: str, service_data: Dict) -> Dict:
    """清洗单个服务数据"""
    cleaned = {}
    
    # 检测数据是否已经被压缩过
    is_already_cleaned = is_service_already_cleaned(service_data)
    
    # 如果数据已经被压缩过，直接返回原数据（不做任何处理）
    if is_already_cleaned:
        return service_data.copy()
    
    # 以下是对未压缩数据的处理逻辑
    
    # 处理Body（可能是 "Body" 或 "Html Body"）
    body = service_data.get("Body") or service_data.get("Html Body")
    body_title = service_data.get("Body Title", "") or service_data.get("Html Body Title", "")
    
    # 处理Body
    if body:
        cleaned_body = clean_body(body, body_title)
        if cleaned_body is not None:
            cleaned["Body"] = cleaned_body
    
    # 处理其他字段
    for key, value in service_data.items():
        # 跳过已处理的Body相关字段
        if key in REMOVE_SERVICE_FIELDS or key in ("Body", "Body Title", "Html Body", "Html Body Title"):
            continue
        
        # 清洗Banner
        if key == "Banner" and isinstance(value, str):
            if "ssh" in service_name.lower():
                banner_text, algo_summary = clean_ssh_banner(value)
                value = banner_text
                if algo_summary:
                    cleaned["SSH_Algorithms"] = algo_summary
            elif "mysql" in service_name.lower():
                value = clean_mysql_banner(value)
            elif "http" in service_name.lower():
                # HTTP Banner 清洗：处理 HTML/CSS/JavaScript 内容
                value = clean_http_banner(value)
        
        # 清洗TLS Banner
        if key == "TLS Banner" and isinstance(value, str):
            value = clean_tls_banner(value)
        
        cleaned[key] = value
    
    return cleaned


def is_record_already_cleaned(data: Dict) -> bool:
    """
    检测整条记录是否已经被压缩过
    通过检查多种特征来判断：
    1. Services 中的服务数据是否已被压缩
    2. 是否缺少原始数据的特征字段（如 IP Index, Timestamp）
    3. 是否存在压缩后的特征字段（如简化的 Body 格式）
    """
    services = data.get("Services", {})
    
    # 检查任意一个服务是否已被压缩
    for svc_name, svc_data in services.items():
        if isinstance(svc_data, dict) and is_service_already_cleaned(svc_data):
            return True
    
    # 如果没有 IP Index 和 Timestamp 字段，但有 Services 字段，
    # 需要检查是否有原始 HTML（在 Body 或 Banner 字段中）
    if "IP Index" not in data and "Timestamp" not in data and services:
        # 检查是否有任何服务包含原始 HTML
        has_raw_html = False
        for svc_name, svc_data in services.items():
            if isinstance(svc_data, dict):
                # 检查 Body 字段
                body = svc_data.get("Body") or svc_data.get("Html Body") or ""
                if isinstance(body, str):
                    html_patterns = ['<html', '<body', '<div', '<head', '<script', '<style', '<!DOCTYPE']
                    if any(pattern.lower() in body.lower() for pattern in html_patterns):
                        has_raw_html = True
                        break
                
                # 也检查 Banner 字段（HTTP 响应可能包含 HTML）
                banner = svc_data.get("Banner") or ""
                if isinstance(banner, str):
                    html_patterns = ['<html', '<body', '<div', '<head', '<script', '<style', '<!DOCTYPE']
                    if any(pattern.lower() in banner.lower() for pattern in html_patterns):
                        has_raw_html = True
                        break
        
        # 如果没有原始 HTML，认为数据已经被压缩过
        if not has_raw_html:
            return True
    
    return False


def clean_record(ip: str, data: Dict, keep_labels: bool = False) -> Dict:
    """
    清洗单条记录
    
    Args:
        ip: IP地址
        data: 原始数据
        keep_labels: 是否保留标签字段（OS, Vendor, Device Type等）
    """
    # 检测数据是否已经被压缩过
    if is_record_already_cleaned(data):
        # 已压缩的数据，但仍需要删除标签字段（如果不保留）
        if not keep_labels:
            # 删除标签字段（OS, Vendor, Device Type）
            data = {k: v for k, v in data.items() if k not in {"OS", "Vendor", "Device Type"}}
        return clean_value({ip: data})
    
    cleaned = {}
    
    # 确定要删除的字段
    fields_to_remove = REMOVE_TOP_FIELDS
    if keep_labels:
        # 保留标签字段，只删除非标签的顶层字段
        fields_to_remove = {"IP Index", "Timestamp"}
    
    for key, value in data.items():
        # 删除顶层字段
        if key in fields_to_remove:
            continue
        
        # 处理Services
        if key == "Services" and isinstance(value, dict):
            cleaned_services = {}
            for svc_name, svc_data in value.items():
                if isinstance(svc_data, dict):
                    cleaned_services[svc_name] = clean_service(svc_name, svc_data)
                else:
                    cleaned_services[svc_name] = svc_data
            cleaned[key] = cleaned_services
        else:
            cleaned[key] = value
    
    return clean_value({ip: cleaned})


# ========= DataCleaner类 =========

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, input_path: str, output_path: str, keep_labels: bool = False):
        """
        初始化清洗器
        
        Args:
            input_path: 输入路径（可以是文件夹或单个文件）
            output_path: 输出文件路径（清洗后JSONL）
            keep_labels: 是否保留标签字段（OS, Vendor等），默认False
        """
        self.input_path = input_path
        self.output_path = output_path
        self.keep_labels = keep_labels
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'failed_records': 0,
            'original_size': 0,
            'cleaned_size': 0,
            'input_files': 0
        }
    
    def _get_input_files(self) -> list:
        """
        获取所有输入文件
        如果input_path是文件夹，返回其中所有json文件
        如果是单个文件，返回该文件
        """
        import os
        import glob
        
        if os.path.isdir(self.input_path):
            # 获取文件夹中所有json文件
            json_files = glob.glob(os.path.join(self.input_path, "*.json"))
            jsonl_files = glob.glob(os.path.join(self.input_path, "*.jsonl"))
            all_files = sorted(json_files + jsonl_files)
            return all_files
        elif os.path.isfile(self.input_path):
            return [self.input_path]
        else:
            raise FileNotFoundError(f"输入路径不存在: {self.input_path}")
    
    def _process_file(self, filepath: str, outfile, max_records: int, current_count: int) -> int:
        """
        处理单个文件
        
        Args:
            filepath: 输入文件路径
            outfile: 输出文件对象
            max_records: 最大处理条数
            current_count: 当前已处理条数
            
        Returns:
            处理后的总条数
        """
        import os
        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                if max_records is not None and current_count >= max_records:
                    return current_count
                
                line = line.strip()
                if not line:
                    continue
                
                self.stats['total_records'] += 1
                self.stats['original_size'] += len(line)
                
                try:
                    obj = json.loads(line)
                    ip = next(iter(obj.keys()))
                    
                    # 清洗数据
                    cleaned = clean_record(ip, obj[ip], keep_labels=self.keep_labels)
                    cleaned_json = json.dumps(cleaned, ensure_ascii=False)
                    
                    # 写入输出
                    outfile.write(cleaned_json + '\n')
                    
                    self.stats['processed_records'] += 1
                    self.stats['cleaned_size'] += len(cleaned_json)
                    current_count += 1
                    
                except (json.JSONDecodeError, StopIteration) as e:
                    self.stats['failed_records'] += 1
                    print(f"[WARN] {filename} 第{i+1}行: {e}")
                    continue
        
        return current_count
    
    def run(self, max_records: int = None) -> Dict:
        """
        执行清洗，返回统计信息
        
        Args:
            max_records: 最大处理条数，None表示处理全部
            
        Returns:
            统计信息字典
        """
        import time
        import os
        start_time = time.time()
        
        # 获取所有输入文件
        input_files = self._get_input_files()
        self.stats['input_files'] = len(input_files)
        
        if not input_files:
            print(f"[WARN] 未找到输入文件: {self.input_path}")
            return self.stats
        
        print(f"发现 {len(input_files)} 个输入文件:")
        for f in input_files:
            print(f"  - {os.path.basename(f)}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 处理所有文件
        current_count = 0
        with open(self.output_path, 'w', encoding='utf-8') as outfile:
            for filepath in input_files:
                if max_records is not None and current_count >= max_records:
                    break
                current_count = self._process_file(filepath, outfile, max_records, current_count)
            # 确保数据写入磁盘
            outfile.flush()
            os.fsync(outfile.fileno())
        
        # 计算压缩率
        if self.stats['original_size'] > 0:
            self.stats['compression_ratio'] = 1 - (
                self.stats['cleaned_size'] / self.stats['original_size']
            )
        else:
            self.stats['compression_ratio'] = 0
        
        self.stats['execution_time_seconds'] = time.time() - start_time
        
        return self.stats
