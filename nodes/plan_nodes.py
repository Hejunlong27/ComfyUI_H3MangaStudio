"""编排层节点：清单加载 / 分镜绑定 / 提示词装配 / 一致性审计 / 预览。"""

from __future__ import annotations

import json
from typing import List

from comfy_api.latest import ComfyExtension, io

try:
    from ._io_types import board_in, plan_in, plan_out, refs_in
except ImportError:  # pragma: no cover
    from _io_types import board_in, plan_in, plan_out, refs_in

try:
    from ..core import plan as pl
    from ..core import prompt_build as pb
    from ..core import script_parse as sp
    from ..core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from ..core.ui import safe_display_name as _inp, label_input as _L
except ImportError:  # pragma: no cover
    from core import plan as pl
    from core import prompt_build as pb
    from core import script_parse as sp
    from core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from core.ui import safe_display_name as _inp, label_input as _L


# ==========================================================================
# 3. 分镜清单加载
# ==========================================================================
class H3MangaScriptLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaScriptLoader",
            display_name=_NL["H3MangaScriptLoader"],
            category="H3 Manga/编排",
            description=(
                "加载分镜清单（JSON 文件 / 手写块文本），自动做 17n+5 帧对齐、"
                "32 倍数画幅对齐，并输出结构化的分镜计划。\n"
                "块文本格式见插件 templates/shot_list_template.txt。"
            ),
            inputs=[
                _inp(io.Combo.Input, "source", _L("source"),
                     options=["file", "inline", "auto"], default="file"),
                _inp(io.String.Input, "file_path", _L("file_path"),
                     default="", tooltip="分镜清单绝对路径（.json 或 .txt）"),
                _inp(io.String.Input, "inline_text", _L("inline_text"),
                     default="", multiline=True,
                     tooltip="直接粘贴分镜内容。JSON 或块文本都行。"),
            ],
            outputs=[
                plan_out(display_name=_OL["plan"]),
                io.String.Output(display_name=_OL["report"]),
            ],
        )

    @classmethod
    def execute(cls, source="file", file_path="", inline_text="") -> io.NodeOutput:
        plan = sp.load_plan(source=source, file_path=file_path, inline_text=inline_text)
        report = plan.summary()
        if plan.shots:
            report += "\n" + "-" * 52 + "\n镜头明细：\n"
            for s in plan.shots:
                report += (
                    f"  [{s.index:02d}] {s.id:<12} {s.duration:>5.1f}s "
                    f"{s.width}x{s.height} panel={s.panel_index} "
                    f"角色={','.join(s.characters) or '-'}\n"
                )
        return io.NodeOutput(plan, report, ui={"h3manga_panel": [report]})


# ==========================================================================
# 4. 分镜绑定
# ==========================================================================
class H3MangaShotBinder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaShotBinder",
            display_name=_NL["H3MangaShotBinder"],
            category="H3 Manga/编排",
            description=(
                "把分镜计划与切好的分镜格图绑定。\n"
                "by_panel_index=用分镜里的 panel 字段精确指定（推荐）；"
                "sequential=按镜头顺序依次取格。"
            ),
            inputs=[
                plan_in("plan", display_name=_L("plan")),
                board_in("board", display_name=_L("board"), optional=True),
                refs_in("refs", display_name=_L("refs"), optional=True,
                         tooltip="接上参考包可让绑定阶段就报出「角色缺图」等问题"),
                _inp(io.Combo.Input, "bind_mode", _L("bind_mode"),
                     options=["by_panel_index", "sequential", "none"],
                     default="by_panel_index"),
                _inp(io.Int.Input, "panel_offset", _L("panel_offset"),
                     default=0, min=-999, max=999, step=1),
                _inp(io.Boolean.Input, "allow_reuse", _L("allow_reuse"), default=True),
            ],
            outputs=[
                plan_out(display_name=_OL["plan"]),
                io.String.Output(display_name=_OL["report"]),
            ],
        )

    @classmethod
    def execute(cls, plan, board=None, refs=None, bind_mode="by_panel_index",
                panel_offset=0, allow_reuse=True) -> io.NodeOutput:
        if plan is None:
            raise ValueError("分镜绑定需要上游的分镜计划（plan）")
        out = plan.clone()
        pl.bind_board(out, board, mode=bind_mode, offset=int(panel_offset),
                      allow_reuse=bool(allow_reuse))

        lines = [f"分镜绑定报告（mode={bind_mode}, offset={panel_offset}）", "=" * 52]
        bound = sum(1 for s in out.shots if s.board_image_index is not None)
        lines.append(f"  已绑定 {bound} / {len(out.shots)} 镜")
        for s in out.shots:
            tag = f"panel#{s.board_image_index}" if s.board_image_index is not None else "无格图"
            lines.append(f"  [{s.index:02d}] {s.id:<14} → {tag}")

        if refs is not None:
            aud = pl.audit(out, refs, strict=False)
            lines.append("")
            lines.append(pl.audit_report(aud))
        if out.warnings:
            lines.append("")
            lines.append("警告：")
            for w in out.warnings:
                lines.append(f"  ! {w}")
        body = "\n".join(lines)
        return io.NodeOutput(out, body, ui={"h3manga_panel": [body]})


# ==========================================================================
# 5. 提示词装配
# ==========================================================================
class H3MangaPromptComposer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaPromptComposer",
            display_name=_NL["H3MangaPromptComposer"],
            category="H3 Manga/编排",
            description=(
                "把分镜字段编译成 H3 提示词：@角色id → <Picture i>、运镜短名展开、"
                "风格注入、台词包裹、角色一致性锁定。\n"
                "占位符：{refs} {action} {camera} {style} {dialogue} {extra} {speaker}"
            ),
            inputs=[
                plan_in("plan", display_name=_L("plan")),
                refs_in("refs", display_name=_L("refs"), optional=True),
                _inp(io.String.Input, "template", _L("template"),
                     default="", multiline=True,
                     tooltip="留空用内置模板。可用 {refs}{action}{camera}{style}{dialogue}{extra}"),
                _inp(io.String.Input, "style_prefix", _L("style_prefix"), default=""),
                _inp(io.String.Input, "style_suffix", _L("style_suffix"), default=""),
                _inp(io.String.Input, "negative", _L("negative"), default="", multiline=True),
                _inp(io.String.Input, "camera_preset", _L("camera_preset"),
                     default="static",
                     tooltip="分镜没写 camera 时用这个。可选值见 core/prompt_build.py 的 CAMERA_PRESETS"),
                _inp(io.Combo.Input, "dialogue_mode", _L("dialogue_mode"),
                     options=pb.DIALOGUE_MODES, default="spoken",
                     tooltip="spoken=台词交给 H3 生成语音；none=不带台词（后期配音）"),
                _inp(io.Combo.Input, "negative_mode", _L("negative_mode"),
                     options=pb.NEGATIVE_MODES, default="append",
                     tooltip="H3 官方 R2V 没有 negative 输入，所以负向词只能以自然语言\n"
                             "追加进正向提示词。append=追加 / ignore=只记报告不写进提示词。"),
                _inp(io.Boolean.Input, "lock_consistency", _L("lock_consistency"), default=True),
                _inp(io.Boolean.Input, "include_describe", _L("include_describe"), default=True),
                _inp(io.String.Input, "camera_json", _L("camera_json"),
                     default="", tooltip="自定义运镜表 JSON 文件路径（可留空）"),
                _inp(io.String.Input, "style_json", _L("style_json"),
                     default="", tooltip="自定义风格表 JSON 文件路径（可留空）"),
            ],
            outputs=[
                plan_out(display_name=_OL["plan"]),
                io.String.Output(display_name=_OL["prompts"]),
            ],
        )

    @classmethod
    def execute(cls, plan, refs=None, template="", style_prefix="", style_suffix="",
                negative="", camera_preset="static", dialogue_mode="spoken",
                negative_mode="append", lock_consistency=True, include_describe=True,
                camera_json="", style_json="") -> io.NodeOutput:
        if plan is None:
            raise ValueError("提示词装配需要上游的分镜计划（plan）")
        out = plan.clone()
        cam_extra = pb.load_camera_presets(camera_json)
        style_extra = pb.load_style_presets(style_json)

        pb.compose_plan(
            out, refs=refs, template=template,
            style_prefix=style_prefix, style_suffix=style_suffix,
            negative=negative, camera_preset=camera_preset,
            dialogue_mode=dialogue_mode, lock_consistency=bool(lock_consistency),
            extra_camera_presets=cam_extra, extra_style_presets=style_extra,
            include_describe=bool(include_describe), negative_mode=negative_mode,
        )
        body = pb.preview_all(out)
        # ui 载荷 → 前端面板直接显示「机器实际拿到的提示词」
        return io.NodeOutput(out, body, ui={"h3manga_panel": [body]})


# ==========================================================================
# 6. 一致性审计
# ==========================================================================
class H3MangaConsistencyAudit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaConsistencyAudit",
            display_name=_NL["H3MangaConsistencyAudit"],
            category="H3 Manga/校验",
            description=(
                "跑整集之前的一致性体检：角色引用缺失、参考图过少、"
                "序号溢出、槽位漂移、无锚点镜头。\n"
                "strict=True 时 P1 也会导致不通过。"
            ),
            inputs=[
                plan_in("plan", display_name=_L("plan")),
                refs_in("refs", display_name=_L("refs"), optional=True),
                _inp(io.Boolean.Input, "strict", _L("strict"), default=False),
            ],
            outputs=[
                io.String.Output(display_name=_OL["report"]),
                io.Boolean.Output(display_name=_OL["passed"]),
            ],
        )

    @classmethod
    def execute(cls, plan, refs=None, strict=False) -> io.NodeOutput:
        if plan is None:
            raise ValueError("一致性审计需要上游的分镜计划（plan）")
        result = pl.audit(plan, refs, strict=bool(strict))
        rep = pl.audit_report(result)
        return io.NodeOutput(rep, bool(result["passed"]), ui={"h3manga_panel": [rep]})


# ==========================================================================
# 7. 计划预览
# ==========================================================================
class H3MangaPlanPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaPlanPreview",
            display_name=_NL["H3MangaPlanPreview"],
            category="H3 Manga/校验",
            description=(
                "只读预览：每镜的最终提示词 + 参考序号映射 + 警告。"
                "节点上会挂一个前端面板直接显示（排查角色串号最快的手段）。"
            ),
            inputs=[
                plan_in("plan", display_name=_L("plan")),
                refs_in("refs", display_name=_L("refs"), optional=True),
                _inp(io.String.Input, "text", _L("text"), default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name=_OL["text"]),
                io.String.Output(display_name=_OL["meta"]),
            ],
        )

    @classmethod
    def execute(cls, plan, refs=None, text="") -> io.NodeOutput:
        parts: List[str] = []
        if text.strip():
            parts.append(text)
        if plan is not None:
            parts.append(pb.preview_all(plan))
            if refs is not None:
                parts.append(refs.describe_table())
        if not parts:
            parts.append("（没有接到分镜计划，也没有文本输入）")

        body = "\n\n".join(parts)
        meta = json.dumps(
            {
                "plan": plan.to_meta() if plan is not None else None,
                "refs": refs.to_meta() if refs is not None else None,
            },
            ensure_ascii=False, indent=1,
        )
        return io.NodeOutput(body, meta, ui={"h3manga_panel": [body]})
