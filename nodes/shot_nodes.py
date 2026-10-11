"""前置出栈节点：把「分镜计划」变成 **官方多参视频节点** 能直接吃的输入。

定位（v0.7 起）
--------------
插件到这两个节点为止 —— **采样、放大、二采、解码全部交回工作流**。
官方 ``MiniMaxH3ReferenceToVideo`` 自己就输出 ``positive(CONDITIONING) + Latent``，
所以「插件做前置」这条路在官方口径下本来就成立，不需要插件碰采样链。

两个节点，按需要挑一个：

* **⑫-A `H3MangaShotOut`（9 槽出栈）**
  产物 = 提示词 + 9 个独立参考图槽 + 宽高帧数 + 槽位对照表。
  用户把它们一一接进官方节点（``ref_image_1..9`` / ``prompt`` / ``width`` 等），
  官方节点**在图里看得见**，后面接什么采样链完全自由。

* **⑫-B `H3MangaShotCond`（条件直出）**
  内部就是调用官方 ``MiniMaxH3ReferenceToVideo``，直接吐 ``positive`` + ``latent``。
  少接 9 根线、零接错风险；代价是官方节点不显示在图上。

为什么 A 必须给 **9 个口**而不是 1 个批量口
-------------------------------------------
官方 ``ref_images`` 是 ``Autogrow(prefix="ref_image_", max=9)`` —— **一个槽 = 一张 IMAGE**，
批量张量在官方口径下没有对应接口。而且官方执行体里是

    for img in (ref_images or {}).values():
        if img is None:
            continue

**空槽会被自动跳过**，所以用不到的槽返回 ``None`` 是安全的：
它既不占 ``<Picture i>`` 的序号，也不会污染提示词里的标签。
"""

from __future__ import annotations

from typing import Any, List, Optional

from comfy_api.latest import ComfyExtension, io

try:
    from ._io_types import board_in, plan_in, refs_in
except ImportError:  # pragma: no cover
    from _io_types import board_in, plan_in, refs_in

try:
    from ..core import h3_api as api
    from ..core import plan as pl
    from ..core import refs as rf
    from ..core import types as _T
    from ..core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from ..core.ui import safe_display_name as _inp, label_input as _L
except ImportError:  # pragma: no cover
    from core import h3_api as api
    from core import plan as pl
    from core import refs as rf
    from core import types as _T
    from core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from core.ui import safe_display_name as _inp, label_input as _L

MAX_SLOTS = int(_T.H3_MAX_REF_IMAGES)      # 官方上限 9


def _panel_tensor(board, idx) -> Any:
    """从 board 取第 idx 张格图（对应 ``shot.board_image_index``）。"""
    if board is None or idx is None:
        return None
    imgs = getattr(board, "images", None) or []
    try:
        return imgs[int(idx)]
    except (IndexError, TypeError, ValueError):
        return None


def _pick_shot(plan, shot_index: int):
    """★ 1-based（对齐 ``<Picture i>`` 的编号习惯）。

    plan 缺失时**不抛异常**：返回 ``(None, 0, 0, 说明)``，由调用方输出空结果 +
    一条可读报告。理由 —— 这个节点常常被先摆到画布上再接线，硬报错会让整个
    工作流无法运行，用户还看不到「该接哪根线」。
    """
    if plan is None or not getattr(plan, "shots", None):
        return None, 0, 0, (
            "还没有分镜计划 —— 请把 ③「分镜清单加载」的 `分镜计划` 输出接到本节点的 "
            "`plan` 输入（③ 可用 source=board 从 ① 的分镜格自动出镜）。"
        )
    n = len(plan.shots)
    idx = int(shot_index)
    note = ""
    if idx < 1:
        idx = 1
        note = "shot_index 从 1 起，已按第 1 镜处理"
    elif idx > n:
        note = f"shot_index={idx} 超出范围（本计划 {n} 镜），已夹到第 {n} 镜"
        idx = n
    return plan.shots[idx - 1], idx, n, note


def _build_layout(plan, refs, board, shot, include_anchor: bool):
    """统一的参考图布局（★ 唯一出参考图的地方，序号契约在此）。"""
    pack = refs or _T.RefPack()
    style_ids = [e.id for e in pack.ordered() if e.kind == "style"]
    panel = _panel_tensor(board, shot.board_image_index) if include_anchor else None
    layout = rf.shot_ref_layout(
        pack, characters=list(shot.characters),
        panel_image=panel,
        has_panel=bool(include_anchor and shot.board_image_index is not None),
        style_ids=style_ids,
    )
    return layout


def _slots_report(shot, idx, total, layout, note: str) -> str:
    """槽位对照表 —— 用户核对「第 N 槽是谁」的唯一依据。

    `<Picture i>` 接错**不会报错**，只会脸不对，所以这张表必须写成人能核对的形式。
    """
    ref_images = layout.get("ref_images") or {}
    used = sorted(ref_images.keys())
    lines = [
        f"第 {idx} / {total} 镜：{shot.id}",
    ]
    # ★ 这一条是「为什么只出了一个镜头」的现场答案。
    #   ComfyUI 的工作流本身**没有循环**，所以本节点一次运行只处理一镜；
    #   想按顺序跑完全部镜头，必须交给内部循环的 ⑦。
    if total > 1:
        lines.append(
            f"★ 本节点**一次只输出 1 个镜头**（ComfyUI 工作流没有循环）："
            f"本计划共 {total} 镜，本次是第 {idx} 镜。\n"
            f"   · 想逐镜出片：改 `shot_index`（1~{total}）再跑一次\n"
            f"   · 想**一次跑完全部镜头**：把 ④ 的 `分镜计划` 接到 "
            f"⑦「整集渲染 / 批量出镜」（`max_shots=0`），它内部逐镜循环"
        )
    lines += [
        f"  画幅 {shot.width}x{shot.height}  帧数 {shot.length}"
        f"（{shot.length / 24.0:.2f}s @24fps）",
        f"  本镜参考图 {len(used)} 张，占前 {len(used)} 个槽；"
        f"其余 {max(0, MAX_SLOTS - len(used))} 槽返回空（不接即可，官方会跳过）",
        "  ★ 接线对照：输出「参考图 N」→ 官方输入 ref_image_(N-1)",
        "    （官方 Autogrow 槽名 **0 起**，而提示词里的 <Picture i> 是 **1 起**，差 1）",
        "",
        rf.layout_table(layout),
    ]
    if note:
        lines.insert(0, f"! {note}")
    if not (shot.prompt or "").strip():
        lines.append("")
        lines.append(
            "! 该镜的提示词是空的 —— 请确认 ⑤ 提示词装配 接在 ③ 之后，"
            "否则参考图没有 <Picture i> 标签去引用（画面会退化成随机）。"
        )
    return "\n".join(lines)


# ==========================================================================
# ⑫-A 镜头出栈（9 槽）
# ==========================================================================
class H3MangaShotOut(io.ComfyNode):
    """把一个镜头拆成官方多参节点能直接吃的 9 个参考图槽 + 提示词 + 宽高帧数。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaShotOut",
            display_name=_NL["H3MangaShotOut"],
            category="H3 Manga/前置",
            description=(
                "★ **官方多参视频节点的前置接口**：取计划里的第 N 镜，输出\n"
                "  · `prompt`（@角色 已换成 <Picture i> 的最终提示词）\n"
                "  · `参考图 1..9`（**按槽位序号**，用不到的槽返回空）\n"
                "  · `宽` / `高` / `帧数`（可直连官方节点同名输入）\n"
                "  · `槽位对照表`（第 N 槽是谁，接错时核对用）\n"
                "把有图的槽按序号接进官方 `MiniMaxH3ReferenceToVideo` 即可；\n"
                "采样器 / LoRA / 放大 / 二采 / 解码全在你自己的图里。"
            ),
            inputs=[
                plan_in("plan", display_name="分镜计划"),
                refs_in(name="refs", display_name="参考包", optional=True),
                board_in(name="board", display_name="分镜格", optional=True),
                _inp(io.Int.Input, "shot_index", _L("shot_index"),
                     default=1, min=1, max=9999, step=1,
                     tooltip="取计划里的第几镜（**从 1 起**，对齐 <Picture i> 的编号习惯）。\n"
                             "越界会自动夹到最后一镜并在报告里说明。\n"
                             "★ 本节点**一次运行只输出 1 个镜头** —— ComfyUI 的工作流\n"
                             "   没有循环，所以「一次跑完所有镜头」得用 ⑦ 整集渲染\n"
                             "   （它内部逐镜循环，max_shots=0 即全部）。\n"
                             "   逐镜出片就改这个值再跑一次。"),
                _inp(io.Boolean.Input, "include_anchor", _L("include_anchor"),
                     default=True,
                     tooltip="把该镜的分镜格图作为**第 1 张**参考图（构图锚点）。\n"
                             "关掉它就只是纯角色参考 + 纯文本构图。"),
                _inp(io.Combo.Input, "ref_image_size", _L("ref_image_size"),
                     options=["match", "max"], default="match",
                     tooltip="给官方节点的建议值（若你把本节点输出接到官方节点）：\n"
                             "match=按生成画幅缩放参考图（快）/ max=2048 短边（最像但慢数倍）"),
            ],
            outputs=[
                io.String.Output(display_name=_OL["prompt"]),
                # ★ 显示名里直接写出官方的槽名，避免「第1张图该接 ref_image_0」这个错位坑：
                #   官方 Autogrow 的槽名是 **0 起**（ref_image_0..8），
                #   而提示词里的 <Picture i> 是 **1 起**，两者差 1。
                *[io.Image.Output(display_name=f"参考图 {i} → 官方 ref_image_{i - 1}")
                  for i in range(1, MAX_SLOTS + 1)],
                io.Int.Output(display_name="宽"),
                io.Int.Output(display_name="高"),
                io.Int.Output(display_name="帧数"),
                io.String.Output(display_name=_OL["slots"]),
            ],
        )

    @classmethod
    def execute(cls, plan, refs=None, board=None, shot_index=1,
                include_anchor=True, ref_image_size="match") -> io.NodeOutput:
        shot, idx, total, note = _pick_shot(plan, shot_index)
        if shot is None:
            return io.NodeOutput("", *([None] * MAX_SLOTS), 0, 0, 0, note,
                                 ui={"h3manga_panel": [note]})
        layout = _build_layout(plan, refs, board, shot, bool(include_anchor))

        w, h = _T.align_canvas(shot.width, shot.height)
        length = _T.align_frame_count(shot.length)

        ref_images = layout.get("ref_images") or {}
        slots: List[Any] = []
        for i in range(MAX_SLOTS):
            slots.append(ref_images.get(f"ref_image_{i}"))

        slots_txt = _slots_report(shot, idx, total, layout, note)
        slots_txt += f"\n  参考图尺寸建议：ref_image_size={ref_image_size}"
        return io.NodeOutput(shot.prompt or "", *slots, int(w), int(h), int(length),
                             slots_txt, ui={"h3manga_panel": [slots_txt]})


# ==========================================================================
# ⑫-B 单镜条件直出
# ==========================================================================
class H3MangaShotCond(io.ComfyNode):
    """取第 N 镜，直接产出官方条件的 ``positive`` + ``latent``（少接 9 根线）。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaShotCond",
            display_name=_NL["H3MangaShotCond"],
            category="H3 Manga/前置",
            description=(
                "⑫-B **省事版前置**：内部调用官方 `MiniMaxH3ReferenceToVideo`，\n"
                "直接输出 `positive`(CONDITIONING) + `latent`(LATENT)。\n"
                "接你自己的采样链（官方五件套 / KSampler 任意）+ 官方解码即可。\n"
                "与 ⑫-A 的区别：官方节点不出现在图里，但不用手接 9 根参考图线。"
            ),
            inputs=[
                plan_in("plan", display_name="分镜计划"),
                refs_in(name="refs", display_name="参考包", optional=True),
                board_in(name="board", display_name="分镜格", optional=True),
                io.Clip.Input("clip", display_name="文本编码器 CLIP"),
                _inp(io.Int.Input, "shot_index", _L("shot_index"),
                     default=1, min=1, max=9999, step=1,
                     tooltip="取计划里的第几镜（**从 1 起**）。\n"
                             "★ 本节点一次运行只输出 1 个镜头；要一次跑完全部用 ⑦。"),
                _inp(io.Boolean.Input, "include_anchor", _L("include_anchor"),
                     default=True,
                     tooltip="把该镜的分镜格图作为第 1 张参考图（构图锚点）"),
                io.Vae.Input("vae", display_name="视频 VAE", optional=True,
                             tooltip="视频 VAE。不给它，参考图只进文本编码器（不建议）"),
                io.Vae.Input("audio_vae", display_name="音频 VAE", optional=True,
                             tooltip="音频 VAE。给了才启用参考音频"),
                _inp(io.Combo.Input, "ref_image_size", _L("ref_image_size"),
                     options=["match", "max"], default="match",
                     tooltip="match=按生成画幅缩放参考图（快）/ max=2048 短边（最像但慢数倍）"),
            ],
            outputs=[
                io.Conditioning.Output(display_name=_OL["positive"]),
                io.Latent.Output(display_name=_OL["latent"]),
                io.String.Output(display_name=_OL["slots"]),
            ],
        )

    @classmethod
    def execute(cls, plan, clip, refs=None, board=None, shot_index=1,
                include_anchor=True, vae=None, audio_vae=None,
                ref_image_size="match") -> io.NodeOutput:
        if clip is None:
            raise ValueError("⑫-B 需要 clip（Qwen3-VL 文本编码器）")
        shot, idx, total, note = _pick_shot(plan, shot_index)
        if shot is None:
            return io.NodeOutput(None, None, note, ui={"h3manga_panel": [note]})
        if clip is None:
            msg = ("⑫-B 需要 clip（Qwen3-VL 文本编码器）—— 本次未产出条件。"
                   "请把 CLIPLoader 接到本节点的 `clip` 输入。")
            return io.NodeOutput(None, None, msg, ui={"h3manga_panel": [msg]})
        layout = _build_layout(plan, refs, board, shot, bool(include_anchor))

        w, h = _T.align_canvas(shot.width, shot.height)
        length = _T.align_frame_count(shot.length)

        positive, latent = api.encode_reference_to_video(
            clip=clip, prompt=shot.prompt, width=w, height=h, length=length,
            ref_image_size=str(ref_image_size), vae=vae, audio_vae=audio_vae,
            ref_images=layout["ref_images"],
            ref_videos=layout["ref_videos"],
            ref_video_audios=layout["ref_video_audios"],
            ref_audios=layout["ref_audios"],
        )
        slots_txt = _slots_report(shot, idx, total, layout, note)
        return io.NodeOutput(positive, latent, slots_txt,
                             ui={"h3manga_panel": [slots_txt]})


# ==========================================================================
# 注册
# ==========================================================================
class ShotNodesExtension(ComfyExtension):
    @classmethod
    async def get_node_list(cls) -> List[type]:
        return [H3MangaShotOut, H3MangaShotCond]


__all__ = ["H3MangaShotOut", "H3MangaShotCond", "ShotNodesExtension"]
