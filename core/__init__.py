"""H3 Manga Studio —— 核心纯逻辑层。

分层原则（决定了测试策略）：

    core/  ← 不 import comfy（h3_api 除外，且是延迟导入）
             ↑ 可被纯 Python 单测完整覆盖
    nodes/ ← 薄壳：只做输入归一化 + 调用 core + 返回 io.NodeOutput

``h3_api`` 是唯一的例外：它必须调 comfy 才能复用官方 H3 节点，
但它把所有 comfy 导入都放在**函数内部**，所以 ``import core`` 本身仍然安全。
"""

from __future__ import annotations

from . import (  # noqa: F401
    types, script_parse, prompt_build, panel_split, refs, plan, image_ops, mem,
    ui, vlm,
)

__all__ = [
    "types", "script_parse", "prompt_build", "panel_split", "refs", "plan",
    "image_ops", "mem", "ui", "vlm",
]
