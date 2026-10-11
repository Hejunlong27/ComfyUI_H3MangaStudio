"""自定义 IO 类型的模块级单例。

``io.Custom(io_type)`` 每次调用都会**新建一个类**，所以必须在这里建一次、
全局复用，否则同一个类型名在不同节点里会拿到不同的 Python 类对象。
（ComfyUI 的类型匹配最终走 ``io_type`` 字符串，但保持一致能省掉一类诡异问题。）
"""

from __future__ import annotations

from comfy_api.latest import io

try:
    from ..core.types import TYPE_BOARD, TYPE_PLAN, TYPE_REFS, TYPE_VLM
except ImportError:  # pragma: no cover
    from core.types import TYPE_BOARD, TYPE_PLAN, TYPE_REFS, TYPE_VLM

RefsType = io.Custom(TYPE_REFS)
PlanType = io.Custom(TYPE_PLAN)
BoardType = io.Custom(TYPE_BOARD)
VLMType = io.Custom(TYPE_VLM)


def vlm_in(name: str = "vlm", **kw):
    return VLMType.Input(name, **kw)


def vlm_out(**kw):
    return VLMType.Output(**kw)


def refs_in(name: str = "refs", **kw):
    return RefsType.Input(name, **kw)


def refs_out(**kw):
    return RefsType.Output(**kw)


def plan_in(name: str = "plan", **kw):
    return PlanType.Input(name, **kw)


def plan_out(**kw):
    return PlanType.Output(**kw)


def board_in(name: str = "board", **kw):
    return BoardType.Input(name, **kw)


def board_out(**kw):
    return BoardType.Output(**kw)
