"""
HTML解析工具模块
提供HTML内容解析和简化功能，提取关键信息用于分析
"""

import re
from html.parser import HTMLParser
from typing import Dict, List, Any, Optional, Set


class HTMLTextExtractor(HTMLParser):
    """
    HTML文本提取器
    从HTML内容中提取title、关键文本、链接、表单等信息
    """
    
    def __init__(self):
        super().__init__()
        self.title = ''
        self.texts: Set[str] = set()  # 使用set自动去重
        self.links: Set[str] = set()
        self.images: Set[str] = set()
        self.forms: List[str] = []
        self.meta_info: Dict[str, str] = {}
        
        self.current_tag = ''
        self.in_script = False
        self.in_style = False
        self.in_noscript = False
        self.tag_stack = []
    
    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        """处理开始标签"""
        self.current_tag = tag
        self.tag_stack.append(tag)
        attrs_dict = dict(attrs)
        
        # 跳过script、style、noscript标签内容
        if tag == 'script':
            self.in_script = True
        elif tag == 'style':
            self.in_style = True
        elif tag == 'noscript':
            self.in_noscript = True
        
        # 提取meta信息
        elif tag == 'meta':
            name = attrs_dict.get('name', attrs_dict.get('property', '')).lower()
            content = attrs_dict.get('content', '')
            if name in ('description', 'generator', 'author') and content:
                self.meta_info[name] = content[:100]
        
        # 提取链接
        elif tag == 'a':
            href = attrs_dict.get('href', '')
            if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                self._add_link(href)
        
        # 提取图片
        elif tag == 'img':
            src = attrs_dict.get('src', '')
            alt = attrs_dict.get('alt', '')
            if src:
                self._add_image(src)
            if alt and len(alt) > 2:
                self.texts.add(alt[:50])
        
        # 提取表单
        elif tag == 'form':
            action = attrs_dict.get('action', '')
            if action:
                self.forms.append(f"form:{action[:50]}")
        
        # 提取输入字段名
        elif tag == 'input':
            name = attrs_dict.get('name', attrs_dict.get('id', ''))
            input_type = attrs_dict.get('type', 'text')
            if name and input_type in ('text', 'password', 'email', 'search', 'hidden'):
                self.forms.append(f"{input_type}:{name[:20]}")
    
    def _add_link(self, href: str) -> None:
        """添加链接，提取关键部分"""
        href = re.sub(r'^https?://', '', href)
        if len(href) < 100:
            self.links.add(href)
    
    def _add_image(self, src: str) -> None:
        """添加图片，提取文件名"""
        match = re.search(r'([^/]+\.(png|jpg|jpeg|gif|svg|ico|webp))', src, re.I)
        if match:
            self.images.add(match.group(1))
        elif len(src) < 50:
            self.images.add(src)
    
    def handle_endtag(self, tag: str) -> None:
        """处理结束标签"""
        if tag == 'script':
            self.in_script = False
        elif tag == 'style':
            self.in_style = False
        elif tag == 'noscript':
            self.in_noscript = False
        
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        self.current_tag = self.tag_stack[-1] if self.tag_stack else ''
    
    def handle_data(self, data: str) -> None:
        """处理文本数据"""
        if self.in_script or self.in_style or self.in_noscript:
            return
        
        text = data.strip()
        if not text or len(text) < 2:
            return
        
        if self._is_noise(text):
            return
        
        if self.current_tag == 'title':
            self.title = text[:100]
        elif self.current_tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.texts.add(text[:60])
        elif len(text) > 3 and len(text) < 100:
            self.texts.add(text[:60])
    
    def _is_noise(self, text: str) -> bool:
        """判断是否为噪音文本"""
        # 纯符号或数字
        noise_pattern = r'^[\d\s\.\,\-\+\*\/\=\|\&\^\%\$\#\@\!\~\`\(\)\[\]\{\}\<\>\;]+$'
        if re.match(noise_pattern, text):
            return True
        # 常见无意义词
        noise_words = {'login', 'password', 'submit', 'cancel', 'ok', 'yes', 'no',
                       'click', 'here', 'more', 'loading', 'please', 'wait'}
        if text.lower() in noise_words:
            return True
        return False
    
    def error(self, message: str) -> None:
        pass


def simplify_html(html_content: str) -> Optional[str]:
    """
    简化HTML内容，提取关键信息并返回紧凑的字符串格式
    
    Args:
        html_content: 原始HTML内容
        
    Returns:
        紧凑的关键信息字符串，或None（无有价值内容）
    """
    if not html_content or len(html_content) < 20:
        return None
    
    try:
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        
        parts = []
        
        if parser.title:
            parts.append(f"title:{parser.title}")
        
        for k, v in parser.meta_info.items():
            parts.append(f"{k}:{v}")
        
        if parser.images:
            imgs = list(parser.images)[:3]
            parts.append(f"img:{','.join(imgs)}")
        
        if parser.links:
            links = list(parser.links)[:3]
            parts.append(f"links:{','.join(links)}")
        
        if parser.forms:
            forms = list(set(parser.forms))[:3]
            parts.append(f"forms:{','.join(forms)}")
        
        if parser.texts:
            texts = list(parser.texts)[:5]
            parts.append(f"text:{' | '.join(texts)}")
        
        if not parts:
            return None
        
        return ' ; '.join(parts)
    
    except Exception:
        clean_text = re.sub(r'<[^>]+>', ' ', html_content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if clean_text and len(clean_text) > 10:
            return clean_text[:200]
        return None
