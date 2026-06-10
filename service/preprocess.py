"""
日志预处理工具：变量归一化

将日志中的变量部分替换为占位符，以提升异常检测的泛化能力。
此实现独立于原始 K4 代码库，由 K4-service 自行维护。
"""

from __future__ import annotations

import re


def normalize_log(text: str) -> str:
    """
    变量归一化：将日志中的变量部分替换为占位符。

    归一化策略：
    - IP 地址 -> <IP>
    - 十六进制数 -> <HEX>
    - 十进制数 -> <NUM>
    - UUID -> <UUID>
    - 节点名称(BGL格式) -> <NODE>
    """
    text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?", "<IP>", text)
    text = re.sub(r"0x[0-9A-Fa-f]+", "<HEX>", text)
    text = re.sub(r"\b[0-9A-Fa-f]{8,}\b", "<HEX>", text)
    text = re.sub(r"\b\d+\b", "<NUM>", text)
    text = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "<UUID>",
        text,
    )
    text = re.sub(r"R\d{2}-[A-Z]\d-\w+-[CI]?:J\d{2}-U\d{2}", "<NODE>", text)
    return text
