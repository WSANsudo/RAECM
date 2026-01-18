"""
Token计算工具模块
提供基于tiktoken的Token计算功能，支持降级到估算模式
"""

from typing import List, Dict, Optional, Any

# 尝试导入tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False


def get_tokenizer() -> Optional[Any]:
    """
    获取tokenizer实例
    
    Returns:
        tiktoken编码器实例，如果tiktoken不可用则返回None
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    计算文本的Token数量
    
    Args:
        text: 要计算的文本
        tokenizer: tiktoken编码器实例，如果为None则使用估算模式
        
    Returns:
        Token数量
    """
    if not text:
        return 0
    
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    
    # 降级到估算模式：约3个字符一个token
    return len(text) // 3


def count_message_tokens(messages: List[Dict[str, str]], tokenizer: Optional[Any] = None) -> int:
    """
    计算消息列表的总Token数量
    
    遵循OpenAI的消息格式计算规则：
    - 每条消息有4个额外token（用于消息格式）
    - 最后有3个额外token（用于回复格式）
    
    Args:
        messages: 消息列表，每条消息包含role和content
        tokenizer: tiktoken编码器实例
        
    Returns:
        总Token数量
    """
    total = 0
    for msg in messages:
        # 每条消息的格式开销
        total += 4
        # 消息内容的token数
        content = msg.get("content", "")
        total += count_tokens(content, tokenizer)
    
    # 回复格式的开销
    total += 3
    return total


def count_output_tokens(response_text: str, tokenizer: Optional[Any] = None) -> int:
    """
    计算大模型输出响应的Token数量
    
    Args:
        response_text: 大模型返回的响应文本
        tokenizer: tiktoken编码器实例
        
    Returns:
        输出Token数量
    """
    return count_tokens(response_text, tokenizer)
