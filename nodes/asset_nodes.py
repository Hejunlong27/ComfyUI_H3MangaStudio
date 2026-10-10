"""素材层节点：漫画页切格 + 参考包打包。"""

from __future__ import annotations

from typing import Any, List

from comfy_api.latest import ComfyExtension, io

try:
    from ._io_types import board_out, refs_out
except ImportError:  # pragma: no cover
    from _io_types import board_out, refs_out

try:
    from ..core import image_ops as iops
    from ..core import panel_split as ps
    from ..core import refs as rf
    from ..core.types import (
        Board, Panel, H3_MAX_REF_IMAGES, H3_MAX_REF_VIDEOS, H3_MAX_REF_AUDIOS,
    )
    from ..core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from ..core.ui import safe_display_name as _inp, label_input as _L
except ImportError:  # pragma: no cover
    from core import image_ops as iops
    from core import panel_split as ps
    from core import refs as rf
    from core.types import (
        Board, Panel, H3_MAX_REF_IMAGES, H3_MAX_REF_VIDEOS, H3_MAX_REF_AUDIOS,
    )
    from core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from core.ui import safe_display_name as _inp, label_input as _L


def _torch():
    import torch
    return torch


# ==========================================================================
# 1. 漫画页切分镜格
# ==========================================================================
class H3MangaPanelSplitter(io.ComfyNode):
    """整页漫画 → 分镜格。两条通道：自动切格 + 手动框覆盖。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaPanelSplitter",
            display_name=_NL["H3MangaPanelSplitter"],
            category="H3 Manga/素材",
            description=(
                "把整页漫画切成分镜格。自动模式默认走**非白连通域法**"
                "（frame=component，一个非纯白连通块算一格），"
                "对「格内空白被当成装订线」这类过切免疫；"
                "frame=projection 是旧的投影递归切分，留作兼容与降级。\n"
                "手动模式用你标好的框，hybrid 两者叠加。"
                "出口统一做一遍去重叠，相邻框不会互相压着。\n"
                "输出格图批次 + 结构化 board（供分镜绑定节点使用）。"
            ),
            inputs=[
                _inp(io.Image.Input, "image", _L("image")),
                _inp(io.Combo.Input, "mode", _L("mode"),
                     options=["auto", "hybrid", "manual"], default="auto",
                     tooltip="auto=纯自动 / hybrid=自动+手动框叠加 / manual=只用手动框"),
                _inp(io.String.Input, "manual_boxes", _L("manual_boxes"),
                     multiline=True, default="",
                     tooltip="每行一个分镜格，坐标基于原图。**矩形与多边形可混用**：\n"
                             "  4 个数 = 矩形          x,y,w,h\n"
                             "  8 个数 = 四边形（斜切格）x1,y1,x2,y2,x3,y3,x4,y4\n"
                             "  6/10/… 个数 = 任意多边形（≥3 个顶点，偶数个数）\n"
                             "前缀 page0: 只作用于第 0 页；不写前缀 = 套用到所有页。\n"
                             "也可贴 JSON：[[x,y,w,h],...] 或 [[x,y],[x,y],[x,y],[x,y]]。\n"
                             "★ 斜切版面（斜的四边形格）**只能**用多边形写法 —— 矩形切不出来。\n"
                             "坐标可以用 ① 的「网格预览」输出对着数。"),
                _inp(io.Int.Input, "ink_threshold", _L("ink_threshold"),
                     default=200, min=0, max=255, step=1,
                     tooltip="灰度低于此值算「墨迹」。漫画扫描件一般 180~210。"),
                _inp(io.Float.Input, "gutter_ratio", _L("gutter_ratio"),
                     default=0.06, min=0.0, max=0.5, step=0.005,
                     tooltip="一行/列的墨迹占比低于此值就算装订线。调大=更容易切开。"),
                _inp(io.Int.Input, "min_panel_px", _L("min_panel_px"),
                     default=80, min=8, max=2048, step=1),
                _inp(io.Int.Input, "min_gutter_px", _L("min_gutter_px"),
                     default=8, min=1, max=512, step=1),
                _inp(io.Float.Input, "min_panel_ratio", _L("min_panel_ratio"),
                     default=0.01, min=0.0, max=0.9, step=0.001,
                     tooltip="面积占比低于此值的碎片会被并进邻居"),
                _inp(io.Combo.Input, "reading_order", _L("reading_order"),
                     options=["rtl", "ltr"], default="rtl",
                     tooltip="rtl=日漫（右→左）/ ltr=美漫条漫（左→右）"),
                _inp(io.Int.Input, "padding_px", _L("padding_px"),
                     default=4, min=0, max=256, step=1,
                     tooltip="裁切时向外扩的像素，避免切掉格边框"),
                _inp(io.Float.Input, "panel_scale", _L("panel_scale"),
                     default=1.0, min=0.1, max=1.0, step=0.05,
                     tooltip="先缩放再切格（加速），框会自动映射回原图坐标"),
                _inp(io.Float.Input, "merge_tolerance", _L("merge_tolerance"),
                     default=0.0, min=0.0, max=0.5, step=0.001,
                     tooltip=">0 时把面积小于此值的框再合并一次"),
                # ---- v0.4.0 新增：切格框架与去重叠（追加在末尾以兼容旧工作流）----
                _inp(io.Combo.Input, "frame", _L("frame"),
                     options=["component", "projection"], default="component",
                     tooltip="component=非白连通域法（默认，推荐）：一个「非纯白」连通块算一格。\n"
                             "  格内的背景留白天然与画格同块 → 不会把格内空白当成装订线。\n"
                             "projection=整行/整列投影递归切分（旧算法）。\n"
                             "  格内空白（竖排文字栏间距、对话框内部、天空）会被误当成 gutter，\n"
                             "  实测 097 页 12 刀里 8 刀砍在格内部。"),
                _inp(io.Int.Input, "white_threshold", _L("white_threshold"),
                     default=245, min=1, max=255, step=1,
                     tooltip="仅 frame=component 有效。灰度 ≥ 此值算「纸白」。\n"
                             "调高=只有更白的算背景（格内容更完整）；调低=更多区域算背景。"),
                _inp(io.Float.Input, "component_min_area", _L("component_min_area"),
                     default=0.015, min=0.0, max=0.5, step=0.001,
                     tooltip="仅 frame=component 有效。小于整页此比例的块丢弃。"),
                _inp(io.Int.Input, "component_close", _L("component_close"),
                     default=1, min=0, max=4, step=1,
                     tooltip="仅 frame=component 有效。3x3 闭运算迭代数，弥合 1~2px 的断缝。"),
                _inp(io.Boolean.Input, "deoverlap", _L("deoverlap"), default=True,
                     tooltip="把互相重叠的分镜框沿中线分开（默认开）。\n"
                             "重叠来自「向外扩 padding」+「碎片合并取并集」，会让同一批像素\n"
                             "被裁进两个镜头。实测 19 页有 46 对重叠，关闭可复现。"),
                _inp(io.Int.Input, "component_open", _L("component_open"),
                     default=0, min=0, max=4, step=1,
                     tooltip="仅 frame=component 有效。连通域之前先做一次开运算，\n"
                             "用**打断相邻格之间的连接**（共用框线 / 大块黑底）的办法分格。\n"
                             "0=交给自动（见下）。手动指定时会跳过自动。\n"
                             "实测：同一页需要的半径不同（_081 要 2、_135 用 2 反而变差）。"),
                _inp(io.Boolean.Input, "component_auto_open", _L("component_auto_open"),
                     default=True,
                     tooltip="★ 默认开：**逐页**自动把开运算半径试到「漏切迹象消失」为止（0~3）。\n"
                             "只在检测到「过大连通块」或「整页只切出 1 格」时才动手，"
                             "并且**不许减少格数**（防止靠删格把告警消掉）。\n"
                             "实测：19 页平均偏差 1.74 → 1.63；第10卷_081 由 3 格 → 5 格。"),
                # ---- v0.4.3 新增：多边形分镜格 + 坐标网格（同样追加在末尾）----
                _inp(io.Combo.Input, "panel_polygon", _L("panel_polygon"),
                     options=["auto", "off"], default="auto",
                     tooltip="★ 斜切版面（斜的四边形分镜格）用**多边形**表示。\n"
                             "auto（默认）—— 自动通道里，轮廓明显不是矩形的块自动配多边形；\n"
                             "  裁图时把多边形以外填成纸白。矩形格行为与旧版完全一致。\n"
                             "off —— 只用矩形（等价于旧行为）。\n"
                             "手填时**永远**可以写多边形，与本开关无关。"),
                _inp(io.Boolean.Input, "grid_preview", _L("grid_preview"),
                     default=False,
                     tooltip="★ 打开后多出一个「网格预览」输出：在原图上画坐标刻度（每 100px\n"
                             "带数字）+ 当前所有分镜格的编号与轮廓。\n"
                             "用途：手填分镜格时对着它数坐标（斜切格要填 4 个角）。"),
            ],
            outputs=[
                board_out(display_name=_OL["board"]),
                io.Image.Output(display_name=_OL["images"]),
                io.String.Output(display_name=_OL["report"]),
                io.Image.Output(display_name=_OL["grid_preview"]),
            ],
        )

    @classmethod
    def execute(cls, image, mode="auto", manual_boxes="", ink_threshold=200,
                gutter_ratio=0.06, min_panel_px=80, min_gutter_px=8,
                min_panel_ratio=0.01, reading_order="rtl", padding_px=4,
                panel_scale=1.0, merge_tolerance=0.0,
                frame="component", white_threshold=245, component_min_area=0.015,
                component_close=1, deoverlap=True, component_open=0,
                component_auto_open=True, panel_polygon="auto",
                grid_preview=False) -> io.NodeOutput:
        import numpy as np

        if image is None:
            raise ValueError("切格节点需要输入图像（漫画页）")

        arr = image
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[None, ...]          # 单页 → [1,H,W,C]

        # ★ 用 shapes 版解析器：同一行既可以是 4 个数的矩形，也可以是
        #   8 个数的四边形（斜切格），或任意 ≥3 顶点的多边形。
        shapes_manual, warns = ps.parse_manual_shapes(manual_boxes)
        warns = list(warns)

        all_panels: List[Panel] = []
        all_imgs: List[Any] = []
        grid_imgs: List[Any] = []
        idx = 0

        for page_i in range(arr.shape[0]):
            page = arr[page_i]
            H, W = page.shape[:2]

            work = page
            scale = 1.0
            if panel_scale < 0.999:
                scale = float(panel_scale)
                th, tw = max(8, int(H * scale)), max(8, int(W * scale))
                work = _resize_np(page, tw, th)

            shapes, w = ps.split_page_shapes(
                work, mode=mode,
                ink_threshold=int(ink_threshold),
                gutter_ratio=float(gutter_ratio),
                min_panel_px=int(min_panel_px),
                min_gutter_px=int(min_gutter_px),
                min_panel_ratio=float(min_panel_ratio),
                reading_order=reading_order,
                padding_px=int(padding_px),
                manual_boxes=None,
                manual_shapes=shapes_manual,
                merge_tolerance=float(merge_tolerance),
                page_index=page_i,
                frame=str(frame),
                white_threshold=int(white_threshold),
                component_min_area=float(component_min_area),
                component_close=int(component_close),
                deoverlap=bool(deoverlap),
                component_open=int(component_open),
                component_auto_open=bool(component_auto_open),
                panel_polygon=str(panel_polygon),
            )
            for msg in w:
                warns.append(f"[page{page_i}] {msg}")

            if scale < 0.999:
                shapes = ps.upscale_shapes(shapes, 1.0 / scale)

            # ★ crop_shapes：多边形格按掩膜裁（多边形以外填纸白），
            #   矩形格走原来的切片路径 —— 两者混在一批里也没问题。
            crops = ps.crop_shapes(page, shapes)
            page_area = float(H * W)
            n_poly = 0
            for shape, crop in zip(shapes, crops):
                box = [int(v) for v in shape["box"]]
                poly = shape.get("poly")
                if poly:
                    n_poly += 1
                all_panels.append(Panel(
                    index=idx, box=box, page_index=page_i,
                    page_name=f"page{page_i}", area_ratio=(box[2] * box[3]) / page_area,
                    order_hint=reading_order,
                    poly=[[int(p[0]), int(p[1])] for p in poly] if poly else None,
                ))
                all_imgs.append(crop)
                idx += 1
            if n_poly:
                warns.append(
                    f"[page{page_i}] {n_poly} 个分镜格按**多边形**裁剪（斜切版面）"
                )

            if grid_preview:
                grid_imgs.append(_np_to_tensor(_grid_overlay(page, shapes)))

        board = Board(panels=all_panels, images=[], pages=int(arr.shape[0]),
                      source="splitter")
        grid_batch = (iops.stack_images(grid_imgs) if grid_imgs
                      else iops.empty_image_batch())

        if not all_imgs:
            warns.append("没有切出任何分镜格。请检查 mode / 墨迹阈值 / 最小格尺寸。")
            report = board.report() + "\n" + "\n".join(warns)
            return io.NodeOutput(board, iops.empty_image_batch(), report, grid_batch,
                                 ui={"h3manga_panel": [report]})

        tensors = [_np_to_tensor(c) for c in all_imgs]
        board.images = list(tensors)
        batch = iops.stack_images(tensors)

        report = board.report()
        if warns:
            report += "\n" + "-" * 52 + "\n警告：\n" + "\n".join(f"  ! {w}" for w in warns)
        return io.NodeOutput(board, batch, report, grid_batch,
                             ui={"h3manga_panel": [report]})


def _resize_np(img, tw: int, th: int):
    import numpy as np
    if ps.HAS_CV2:
        import cv2
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    # 最近邻降级
    H, W = img.shape[:2]
    ys = (np.arange(th) * H / th).astype(int).clip(0, H - 1)
    xs = (np.arange(tw) * W / tw).astype(int).clip(0, W - 1)
    return img[ys][:, xs]


def _grid_overlay(page, shapes) -> Any:
    """在原图上画坐标刻度 + 分镜格编号与轮廓，供手填坐标时对着数。

    斜切版面要填 4 个角，没有刻度基本没法数 —— 这就是「看得见」那一步。
    """
    import numpy as np

    a = np.asarray(page)
    if a.dtype.kind == "f":
        a = (a * 255.0).clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        vis = np.stack([a] * 3, axis=-1).copy()
    else:
        vis = a[..., :3].copy()
    if not ps.HAS_CV2:
        return vis
    import cv2

    H, W = vis.shape[:2]
    for x in range(0, W, 100):
        cv2.line(vis, (x, 0), (x, H), (200, 150, 60), 1)
        cv2.putText(vis, str(x), (x + 3, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (30, 30, 220), 2)
    for y in range(0, H, 100):
        cv2.line(vis, (0, y), (W, y), (200, 150, 60), 1)
        cv2.putText(vis, str(y), (4, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (30, 30, 220), 2)
    for i, s in enumerate(shapes):
        poly = s.get("poly")
        if poly:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 3)
            x, y = int(poly[0][0]), int(poly[0][1])
        else:
            x, y, w, h = (int(v) for v in s["box"])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(vis, f"#{i}", (x + 8, y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return vis


def _np_to_tensor(arr):
    import numpy as np
    torch = _torch()
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32) / 255.0
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    elif a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    return torch.from_numpy(np.ascontiguousarray(a))


# ==========================================================================
# 2. 参考包（角色 / 风格 / 音频 / 视频）
# ==========================================================================
class H3MangaRefPack(io.ComfyNode):
    """多参数组织中枢：把杂乱的参考素材变成有序参考包 + 序号映射表。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaRefPack",
            display_name=_NL["H3MangaRefPack"],
            category="H3 Manga/素材",
            description=(
                "把角色定妆图、风格基准图、参考音频/视频打包成有序参考包，"
                "并自动算出 <Picture i> / <Video k> / <Audio j> 序号映射表。\n"
                "核心约定：**角色名列表与图槽一一对应**——第一个名字就是第一张图；"
                "同名重复 = 同一角色的多张定妆图（正面/侧面/全身）。"
            ),
            inputs=[
                _inp(io.String.Input, "character_names", _L("character_names"),
                     default="", multiline=True,
                     tooltip="逗号或换行分隔。第 N 个名字 ↔ 第 N 个图槽。\n"
                             "名字会成为分镜里 @名字 的引用目标。"),
                io.Autogrow.Input(
                    "character_images",
                    optional=True,
                    display_name=_L("character_images"),
                    tooltip="第 N 个图槽 ↔ character_names 里的第 N 个名字。\n"
                            "同名重复 = 同一角色的多张定妆图（正面/侧面/全身）。\n"
                            "接批次张量时会按帧拆开，每帧算一张参考图。",
                    template=io.Autogrow.TemplatePrefix(
                        input=_inp(io.Image.Input, "character_image", "角色定妆图"),
                        prefix="character_", min=0, max=H3_MAX_REF_IMAGES,
                    ),
                ),
                _inp(io.String.Input, "character_describes", _L("character_describes"),
                     default="", multiline=True,
                     tooltip="每行一个角色的外观锁定描述（与角色顺序对应）。\n"
                             "比纯标签更稳，例如：棕发橙眼，黑色学生制服，左手中指戴戒指"),
                _inp(io.String.Input, "style_name", _L("style_name"), default="style_main"),
                _inp(io.Image.Input, "style_images", _L("style_images"), optional=True,
                     tooltip="风格基准图。会出现在**每个镜头**的参考里，占 1 个 <Picture i>。"),
                _inp(io.String.Input, "audio_names", _L("audio_names"), default="",
                     tooltip="逗号分隔，与下方音频槽一一对应。分镜里可用 @名字 引用成 <Audio j>。"),
                io.Autogrow.Input(
                    "audios",
                    optional=True,
                    display_name=_L("audios"),
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="BGM / 音效 / 配音样本"),
                        prefix="ref_audio_", min=0, max=H3_MAX_REF_AUDIOS,
                    ),
                ),
                _inp(io.String.Input, "video_names", _L("video_names"), default="",
                     tooltip="逗号分隔，与下方视频槽一一对应。分镜里可用 @名字 引用成 <Video k>。"),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    display_name=_L("videos"),
                    tooltip="**一个槽位 = 一段多帧视频**（不拆帧）。至少 5 帧，官方上限 3 条。",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="参考视频帧，24fps，2~15 秒"),
                        prefix="ref_video_", min=0, max=H3_MAX_REF_VIDEOS,
                    ),
                ),
                io.Autogrow.Input(
                    "video_audios",
                    optional=True,
                    display_name=_L("video_audios"),
                    tooltip="与上方视频**按下标一一对应**的配套音轨（第 1 个音轨属于第 1 段视频）。",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="该段参考视频的原声"),
                        prefix="ref_video_audio_", min=0, max=H3_MAX_REF_VIDEOS,
                    ),
                ),
                _inp(io.Int.Input, "max_ref_images", _L("max_ref_images"),
                     default=H3_MAX_REF_IMAGES, min=1, max=H3_MAX_REF_IMAGES, step=1,
                     tooltip="官方 R2V 上限 9 张。降低可省显存与时间。"),
                _inp(io.Combo.Input, "name_source", _L("name_source"),
                     options=["slot", "index"], default="slot",
                     tooltip="slot=用你填的名字做 id；index=强制 ref1/ref2（名字只做显示）"),
            ],
            outputs=[
                refs_out(display_name=_OL["refs"]),
                io.String.Output(display_name=_OL["table"]),
            ],
        )

    @classmethod
    def execute(cls, character_names="", character_images=None, character_describes="",
                style_name="style_main", style_images=None, audio_names="", audios=None,
                video_names="", videos=None, video_audios=None,
                max_ref_images=H3_MAX_REF_IMAGES, name_source="slot") -> io.NodeOutput:
        pack = rf.build_ref_pack(
            character_names=character_names,
            character_images=character_images,
            character_describes=character_describes,
            style_name=style_name,
            style_images=style_images,
            audio_names=audio_names,
            audios=audios,
            video_names=video_names,
            videos=videos,
            video_audios=video_audios,
            max_ref_images=int(max_ref_images),
            name_source=name_source,
        )
        table = pack.describe_table()
        return io.NodeOutput(pack, table, ui={"h3manga_panel": [table]})
