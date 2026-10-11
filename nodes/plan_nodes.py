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
    from ..core import types as _T
    from ..core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from ..core.ui import safe_display_name as _inp, label_input as _L
except ImportError:  # pragma: no cover
    from core import plan as pl
    from core import prompt_build as pb
    from core import script_parse as sp
    from core import types as _T
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
                "生成 / 加载分镜清单，自动做 17n+5 帧对齐、32 倍数画幅对齐，"
                "输出结构化的分镜计划。\n"
                "★ source=board：**从 ① 的分镜格自动出镜**，不用再手写全表；"
                "source=hybrid：board 自动稿为底、文本只改要改的那几镜。\n"
                "★ **一格多镜**：镜头数 ≠ 格数。一格画了两个时间相时，"
                "用 `auto_shots_per_panel`（整卷批量）或文本里的 `shots: N`（逐格）"
                "把该格展开成多个镜头 —— 它们共用同一张格图当参考锚点。\n"
                "画幅可按**官方 adapt_canvas 规则**选比例（16:9 → 1344x768 等）。\n"
                "块文本字段见 templates/shot_list_template.txt。"
            ),
            inputs=[
                _inp(io.Combo.Input, "source", _L("source"),
                     options=["auto", "board", "hybrid", "inline", "file"],
                     default="auto",
                     tooltip="board  = 从 ① 的分镜格自动出镜（一格一镜）\n"
                             "hybrid = board 为底，文本按 panel 覆盖（推荐）\n"
                             "inline = 只用粘贴的文本 / file = 读文件\n"
                             "auto   = 有文本就用文本，没有才从 board 出"),
                _inp(io.String.Input, "file_path", _L("file_path"),
                     default="", tooltip="分镜清单绝对路径（.json 或 .txt）"),
                _inp(io.String.Input, "inline_text", _L("inline_text"),
                     default="", multiline=True,
                     tooltip="分镜文本。每镜可填的字段：\n"
                             "  panel: 分镜格号（对应 ① 的第几格；hybrid 靠它覆盖）\n"
                             "  ★ shots: N —— 这一格**展开成 N 个镜头**（一格多镜）。\n"
                             "     action / content / dialogue 用 `A | B | C` 分段对应各镜。\n"
                             "  duration: 秒（**每镜**的时长）—— 会被 17n+5 对齐\n"
                             "  content: 画面内容（人填，后续可自动填）\n"
                             "  action: 动作  dialogue: 台词  speaker: 说话人\n"
                             "  characters: 角色（逗号分隔）  camera: 运镜\n"
                             "  size: 画幅（1344x768 或 16:9）—— 逐镜覆盖\n"
                             "  extra / negative / seed / style\n"
                             "格式：`=== s01 ===` 分段头 + 每行 `key: value`"),
                board_in(name="board", display_name="分镜格", optional=True,
                         tooltip="① 的分镜格。source=board / hybrid 必需；"
                                 "size_mode=page 也用它取原页比例。"),
                # ---- v0.7 新增：默认时长与画幅（追加在末尾，旧工作流不受影响）----
                _inp(io.Float.Input, "default_duration", _L("default_duration"),
                     default=5.0, min=0.2, max=60.0, step=0.1,
                     tooltip="文本没写 duration 的镜头用这个秒数。\n"
                             "同样会被 17n+5 对齐（5 → 124 帧 = 5.167s）。"),
                _inp(io.Combo.Input, "size_mode", _L("size_mode"),
                     options=list(_T.H3_SIZE_MODES), default=_T.SIZE_MODE_PRESET,
                     tooltip="preset = 用比例预设（**对齐官方 adapt_canvas 规则**）\n"
                             "page   = 跟随原漫画页比例（需要接 board）\n"
                             "text   = 用文本里的 default_size / 逐镜 size"),
                _inp(io.Combo.Input, "size_preset", _L("size_preset"),
                     options=list(_T.H3_SIZE_PRESETS.keys()),
                     default=_T.SIZE_PRESET_DEFAULT,
                     tooltip="官方规范画幅（短边 768、面积 ≤768×1344、32 倍数）：\n"
                             "  16:9 → 1344x768     4:3 → 1024x768     1:1 → 768x768\n"
                             "  9:16 → 768x1344     3:4 → 768x1024     2:3 → 768x1152"),
                _inp(io.Float.Input, "size_scale", _L("size_scale"),
                     default=_T.DEFAULT_SIZE_SCALE, min=0.1, max=1.0, step=0.05,
                     tooltip="在官方规范画幅上再按比例缩（显存不够就调小）。\n"
                             "16:9 档位：0.5→672x384 · 0.6→800x448 · 0.75→1008x576 · 1.0→1344x768\n"
                             "8GB 卡建议 0.5~0.6。"),
                # ---- v0.71 新增：一格多镜（追加在末尾，旧工作流不受影响）----
                _inp(io.Int.Input, "auto_shots_per_panel",
                     _L("auto_shots_per_panel"), default=1, min=1, max=8, step=1,
                     tooltip="board 自动出镜时**每格展开几镜**（默认 1 = 一格一镜）。\n"
                             "「镜头数 ≠ 格数」：一个分镜格画了两个时间相（大格先全景再特写、\n"
                             "格内插格）时，整卷都这样就把这里设成 2，比逐格手写省事。\n"
                             "展开出的镜共用同一张格图当参考锚点，报告里标为 `panel x/N`。\n"
                             "逐格差异：在文本里对该格写 `shots: N` 覆盖（hybrid 模式）。"),
            ],
            outputs=[
                plan_out(display_name=_OL["plan"]),
                io.String.Output(display_name=_OL["report"]),
            ],
        )

    @classmethod
    def execute(cls, source="auto", file_path="", inline_text="", board=None,
                default_duration=5.0, size_mode=_T.SIZE_MODE_PRESET,
                size_preset=_T.SIZE_PRESET_DEFAULT,
                size_scale=_T.DEFAULT_SIZE_SCALE,
                auto_shots_per_panel=1) -> io.NodeOutput:
        extra_warns: List[str] = []
        mode = str(size_mode or _T.SIZE_MODE_PRESET).lower()

        # ---- 画幅：text 模式交给文本（build_plan 会读 meta.default_size）----
        size_override = None
        if mode == _T.SIZE_MODE_TEXT:
            size_note = "文本里的 default_size / 逐镜 size"
        else:
            page_size = tuple(getattr(board, "page_size", (0, 0)) or (0, 0))
            if mode == _T.SIZE_MODE_PAGE and int(page_size[0]) <= 0:
                extra_warns.append(
                    "size_mode=page 但 board 没有页尺寸（① 未接 / 旧工作流）—— 已回退比例预设"
                )
                mode = _T.SIZE_MODE_PRESET
            w, h, size_note = _T.resolve_canvas(
                size_mode=mode, preset=str(size_preset),
                page_size=page_size if int(page_size[0]) > 0 else None,
                scale=float(size_scale),
            )
            size_override = (w, h, size_note)

        plan = sp.load_plan(
            source=str(source or "auto"), file_path=file_path,
            inline_text=inline_text, default_duration=float(default_duration),
            size_override=size_override, board=board,
            shots_per_panel=int(auto_shots_per_panel),
        )
        for m in extra_warns:
            plan.warnings.append(m)

        report = plan.summary()
        report += "\n" + "-" * 52
        report += f"\n画幅来源：{size_note}"
        report += f"\n镜头明细（共 {len(plan.shots)} 镜）：\n"
        report += plan.shot_table()
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
