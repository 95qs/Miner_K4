"""
日志预处理：变量归一化

将日志中的变量部分替换为统一占位符，减少词法多样性、
提升嵌入质量。此模块直接复用 K4/preprocess.py 的策略。

此文件独立维护，不依赖原始 K4 目录。
"""

from __future__ import annotations

import re


def normalize_log(text: str) -> str:
    """
    变量归一化：将日志中的变量部分替换为占位符。

    归一化策略：
      - IP 地址及端口  -> <IP>
      - 十六进制数      -> <HEX>
      - 超长十六进制串  -> <HEX>
      - 纯数字          -> <NUM>
      - UUID            -> <UUID>
      - BGL 节点名      -> <NODE>

    执行顺序很重要（先长后短、先特殊后通用），避免模式互相覆盖。
    """
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?', '<IP>', text)
    text = re.sub(r'0x[0-9A-Fa-f]+', '<HEX>', text)
    text = re.sub(r'\b[0-9A-Fa-f]{8,}\b', '<HEX>', text)
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    text = re.sub(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        '<UUID>',
        text,
    )
    text = re.sub(r'R\d{2}-[A-Z]\d-\w+-[CI]?:J\d{2}-U\d{2}', '<NODE>', text)
    return text
