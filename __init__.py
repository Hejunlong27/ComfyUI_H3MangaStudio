"""H3 Manga Studio —— 漫改动画批量量产插件（基于 MiniMax H3 多参考视频生成）。

ComfyUI 自定义节点包入口。
"""

from __future__ import annotations

import logging
import os

LOG = logging.getLogger("H3MangaStudio")

_WEB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js")
WEB_DIRECTORY = _WEB_DIR if os.path.isdir(_WEB_DIR) else None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from comfy_api.latest import ComfyExtension, io
    from .nodes import (
        H3MangaPanelSplitter,
        H3MangaRefPack,
        H3MangaScriptLoader,
        H3MangaShotBinder,
        H3MangaPromptComposer,
        H3MangaConsistencyAudit,
        H3MangaPlanPreview,
        H3MangaShotConditioning,
        H3MangaEpisodeRender,
        H3MangaFreeMemory,
        H3MangaShotOut,
        H3MangaShotCond,
        H3MangaVLMScribe,
        H3MangaVLMPlan,
    )

    _NODES = [
        H3MangaPanelSplitter,
        H3MangaRefPack,
        H3MangaScriptLoader,
        H3MangaShotBinder,
        H3MangaPromptComposer,
        # ★ v0.79：VLM 层（看图 → 结构化镜头表 → 分镜计划）
        H3MangaVLMScribe,
        H3MangaVLMPlan,
        # ★ v0.7：前置出栈（插件到这两个节点为止，采样交回工作流）
        H3MangaShotOut,
        H3MangaShotCond,
        H3MangaShotConditioning,
        H3MangaEpisodeRender,
        H3MangaConsistencyAudit,
        H3MangaPlanPreview,
        H3MangaFreeMemory,
    ]

    class H3MangaStudioExtension(ComfyExtension):
        async def get_node_list(self):
            return list(_NODES)

    async def comfy_entrypoint() -> H3MangaStudioExtension:
        return H3MangaStudioExtension()

    # 同时提供 V1 风格的映射表：有些工具/前端会直接读它，
    # 与 V3 的 comfy_entrypoint 并存不冲突。
    for _cls in _NODES:
        try:
            _schema = _cls.define_schema()
            NODE_CLASS_MAPPINGS[_schema.node_id] = _cls
            NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = (
                _schema.display_name or _schema.node_id
            )
        except Exception as e:  # pragma: no cover
            LOG.warning("H3MangaStudio: 无法读取 %s 的 schema：%s", _cls.__name__, e)

except Exception as e:  # pragma: no cover
    # 导入失败绝不能让整个 ComfyUI 起不来
    LOG.error("H3MangaStudio 加载失败：%s: %s", type(e).__name__, e, exc_info=True)


def _startup_capability_check() -> None:
    """启动时打一行 H3 能力自检，缺依赖时用户能在日志第一屏就看到。"""
    try:
        from .core.h3_api import capability_report
    except Exception:
        return
    try:
        rep = capability_report()
    except Exception as e:  # pragma: no cover
        LOG.warning("H3MangaStudio: 能力自检异常：%s", e)
        return
    if rep.get("ok"):
        LOG.info("H3MangaStudio: 官方 H3 能力齐备（%d 项）", len(rep.get("found", [])))
    else:
        LOG.warning(
            "H3MangaStudio: 官方 H3 能力不完整，缺失：%s。"
            "整集渲染节点将不可用。", ", ".join(rep.get("missing", []))
        )


_startup_capability_check()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "comfy_entrypoint",
    "H3MangaStudioExtension",
]
