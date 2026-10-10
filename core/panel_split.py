"""漫画页 → 分镜格：纯 numpy / OpenCV 实现，**不 import comfy / torch**。

为什么抽出来
------------
切格是「静默失败」的重灾区：切歪了不会报错，只会让你拿错图去生成。
把它做成不依赖 comfy 的纯模块，就能用真 torch/numpy 写单元测试把
「已知边界」固化成断言（见 tests/test_pure_logic.py）。

两条通道（``split_page_boxes(frame=...)``）
------------------------------------------
**frame="component"（默认）—— 非白连通域法**
    一个「非纯白」连通块 = 一格。格内的背景留白天然和画格处在同一个连通块里，
    因此不会像投影法那样把格内的空白当成装订线。

    为什么换掉投影法：投影法把「某一行/列的墨迹少」当格边界，但**格内空白**
    （竖排文字的栏间距、对话框内部、天空、页边）同样满足、**而且常常更干净**
    —— 097 页实测真 gutter 行墨迹 0.027~0.054，而文字栏间的缝只有 0.009~0.020，
    共 12 刀里有 8 刀砍在格内部。19 页人工标注：平均绝对偏差 2.89 → **1.74**；
    64 页跨 8 卷抽样：「整页退化成 1 格」27% → **12%**。

**frame="projection" —— 递归投影分割（recursive gutter split）**
    1. 灰度 → 二值化墨迹（漫画是黑线白纸，投影非常干净）
    2. 在区域内找「整行/整列墨迹占比低于阈值」的空白带 = 装订线（gutter）
    3. 选**最宽且居中**的那条带切一刀，递归处理两侧
    4. 递归终止：区域小于最小格尺寸 / 找不到有效 gutter
    5. 合并：面积过小或被 gutter 误切的碎片并回邻居
    6. 排序：日漫右→左、上→下（rtl）；美漫/条漫左→右（ltr）

    注意：这批扫描件的框线是「按栏目分段」的（097 页 y=528 有 1042/1164 宽的
    横线，但 102 页只有 582/1164）—— 所以整行投影既看不到这些线（漏切），
    又会被格内空白骗到（过切）。留着它只为兼容与降级。

去重叠
------
两条通道出口都会跑一遍 ``deoverlap_boxes()``：``_tighten()`` 的外扩
和碎片合并取并集会让相邻框 bbox 互相侵入（实测 19 页 46 对重叠，
最严重的侵入比 0.996）。去重叠零精度代价。

已知边界（故意固化在测试里）
----------------------------
* 跨页大图（出血格）没有 gutter，会被当成一整格 —— 这是对的。
* 斜切分镜 + 大面积黑底（扉页）在两种框架下都失效：投影法无贯穿 gutter，
  连通域法会把多格并成一块。**必须用手动框**。
* 连通域法会把「描边相接 / 网点连片」的相邻格并成一块（→ 用 count_contained 检测）。
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except Exception:  # pragma: no cover
    HAS_CV2 = False

# 手动框的 page 字段用 -1 表示「套用到所有页」
PAGE_ANY = -1


# --------------------------------------------------------------------------
# 图像工具（纯 numpy，cv2 只用于缩放/形态学，缺失时降级）
# --------------------------------------------------------------------------
def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """接受 [H,W] / [H,W,1] / [H,W,3] / [H,W,4] 的 float(0-1) 或 uint8，统一成 [H,W] uint8。"""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if a.ndim == 3:
        if a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[-1] >= 3:
            a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.uint8)
        else:
            a = a[..., 0]
    elif a.ndim == 2:
        pass
    else:
        raise ValueError(f"不支持的图像形状：{a.shape}")
    return a


def ink_mask(gray: np.ndarray, ink_threshold: int = 200) -> np.ndarray:
    """墨迹掩码：True = 有内容（黑）。"""
    return gray < int(ink_threshold)


def _projection(mask: np.ndarray, axis: int) -> np.ndarray:
    """沿 axis 求墨迹占比（0~1）。axis=1 → 每行的占比；axis=0 → 每列的占比。"""
    return mask.mean(axis=axis)


def _find_bands(profile: np.ndarray, low: float, min_width: int) -> List[Tuple[int, int]]:
    """在 1D 投影里找「连续低于 low」的带，返回 [(start, end)]（半开区间）。"""
    empty = profile < low
    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start >= min_width:
                bands.append((start, i))
            start = None
    if start is not None and len(empty) - start >= min_width:
        bands.append((start, len(empty)))
    return bands


def _best_cut(profile: np.ndarray, low: float, min_width: int) -> Optional[Tuple[int, int]]:
    """挑最宽的一条 gutter；同宽取最居中的（避免偏向页边空白）。"""
    bands = _find_bands(profile, low, min_width)
    if not bands:
        return None
    n = len(profile)
    center = n / 2.0

    def score(b: Tuple[int, int]) -> Tuple[float, float]:
        width = b[1] - b[0]
        # 排除贴边的空白（那是页边距，不是 gutter）
        if b[0] <= 1 or b[1] >= n - 1:
            return (-1.0, 0.0)
        return (width, -abs((b[0] + b[1]) / 2.0 - center))

    bands.sort(key=score, reverse=True)
    best = bands[0]
    if score(best)[0] <= 0:
        return None
    return best


# --------------------------------------------------------------------------
# 递归切分
# --------------------------------------------------------------------------
def _recursive_split(
    mask: np.ndarray,
    box: Tuple[int, int, int, int],
    gutter_ratio: float,
    min_panel_px: int,
    min_gutter_px: int,
    depth: int = 0,
    max_depth: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """返回 (x, y, w, h) 列表。"""
    x, y, w, h = box
    if depth >= max_depth or w < min_panel_px * 2 or h < min_panel_px * 2:
        return [box]

    region = mask[y:y + h, x:x + w]
    if region.size == 0:
        return [box]

    row_prof = _projection(region, axis=1)
    col_prof = _projection(region, axis=0)

    row_cut = _best_cut(row_prof, gutter_ratio, min_gutter_px)
    col_cut = _best_cut(col_prof, gutter_ratio, min_gutter_px)

    # 哪条 gutter 更宽就用哪条；同宽优先横切（漫画多为横向排格）
    candidates: List[Tuple[float, str, Tuple[int, int]]] = []
    if row_cut:
        candidates.append((row_cut[1] - row_cut[0], "row", row_cut))
    if col_cut:
        candidates.append((col_cut[1] - col_cut[0] + 0.5, "col", col_cut))
    if not candidates:
        return [box]

    candidates.sort(reverse=True)
    _, kind, (s, e) = candidates[0]

    if kind == "row":
        cut = y + (s + e) // 2
        if cut - y < min_panel_px or (y + h) - cut < min_panel_px:
            return [box]
        top = _recursive_split(mask, (x, y, w, cut - y), gutter_ratio,
                               min_panel_px, min_gutter_px, depth + 1, max_depth)
        bottom = _recursive_split(mask, (x, cut, w, y + h - cut), gutter_ratio,
                                  min_panel_px, min_gutter_px, depth + 1, max_depth)
        return top + bottom
    else:
        cut = x + (s + e) // 2
        if cut - x < min_panel_px or (x + w) - cut < min_panel_px:
            return [box]
        left = _recursive_split(mask, (x, y, cut - x, h), gutter_ratio,
                                min_panel_px, min_gutter_px, depth + 1, max_depth)
        right = _recursive_split(mask, (cut, y, x + w - cut, h), gutter_ratio,
                                 min_panel_px, min_gutter_px, depth + 1, max_depth)
        return left + right


def _tighten(mask: np.ndarray, box: Tuple[int, int, int, int], pad: int) -> Tuple[int, int, int, int]:
    """把框收紧到实际墨迹范围（去掉切分后残留的白边）。"""
    x, y, w, h = box
    region = mask[y:y + h, x:x + w]
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return box
    y0, y1 = y + int(rows[0]), y + int(rows[-1]) + 1
    x0, x1 = x + int(cols[0]), x + int(cols[-1]) + 1
    H, W = mask.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)
    return (x0, y0, x1 - x0, y1 - y0)


def _merge_fragments(
    boxes: Sequence[Tuple[int, int, int, int]],
    min_area_ratio: float,
    page_area: int,
) -> List[Tuple[int, int, int, int]]:
    """把面积过小的碎片并进「重叠最多 / 最近」的邻居。"""
    min_area = page_area * min_area_ratio
    boxes = list(boxes)
    keep: List[Tuple[int, int, int, int]] = []
    small: List[Tuple[int, int, int, int]] = []

    for b in boxes:
        if b[2] * b[3] < min_area:
            small.append(b)
        else:
            keep.append(b)

    if not keep:                     # 全都很小 → 保留最大的那个，别输出空
        if small:
            small.sort(key=lambda b: b[2] * b[3], reverse=True)
            return [small[0]]
        return []

    for s in small:
        sx, sy, sw, sh = s
        scx, scy = sx + sw / 2, sy + sh / 2
        best_i, best_d = 0, float("inf")
        for i, k in enumerate(keep):
            kx, ky, kw, kh = k
            # 若与某格有交叠，直接并进去
            if not (sx + sw <= kx or kx + kw <= sx or sy + sh <= ky or ky + kh <= sy):
                best_i, best_d = i, -1.0
                break
            d = math.hypot(scx - (kx + kw / 2), scy - (ky + kh / 2))
            if d < best_d:
                best_i, best_d = i, d
        kx, ky, kw, kh = keep[best_i]
        nx0, ny0 = min(sx, kx), min(sy, ky)
        nx1, ny1 = max(sx + sw, kx + kw), max(sy + sh, ky + kh)
        keep[best_i] = (nx0, ny0, nx1 - nx0, ny1 - ny0)

    return keep


def _reading_order(
    boxes: Sequence[Tuple[int, int, int, int]],
    direction: str = "rtl",
    row_tolerance: float = 0.45,
) -> List[int]:
    """按阅读顺序返回 boxes 的索引。

    rtl（日漫）：从上到下，同一「行」内从右到左。
    行的判定用**垂直重叠比例**，比纯 y 排序稳得多（漫画格常常错位排布）。
    """
    n = len(boxes)
    if n <= 1:
        return list(range(n))

    remaining = list(range(n))
    ordered: List[int] = []

    while remaining:
        # 取当前最靠上的一格作为「行锚」
        anchor = min(remaining, key=lambda i: (boxes[i][1], boxes[i][0]))
        ax, ay, aw, ah = boxes[anchor]
        row: List[int] = []
        for i in list(remaining):
            x, y, w, h = boxes[i]
            overlap = min(ay + ah, y + h) - max(ay, y)
            if overlap > 0 and overlap / max(1.0, min(ah, h)) >= row_tolerance:
                row.append(i)
        if not row:
            row = [anchor]
        # 行内排序
        row.sort(key=lambda i: boxes[i][0], reverse=(direction == "rtl"))
        ordered.extend(row)
        for i in row:
            remaining.remove(i)

    return ordered


# --------------------------------------------------------------------------
# 非白连通域法（frame="component"）
# --------------------------------------------------------------------------
DEFAULT_WHITE_THRESHOLD = 245      # 灰度 ≥ 此值算「纸白」
DEFAULT_COMPONENT_MIN_AREA = 0.015  # 小于整页此比例的块丢弃
DEFAULT_COMPONENT_CLOSE = 1         # 3x3 闭运算迭代数（弥合 1~2px 断缝）
DEFAULT_COMPONENT_OPEN = 0          # 开运算半径（0=关；>0 可打断细框线）
DEFAULT_AUTO_OPEN_MAX = 3           # 自动开运算的最大半径（实测 4 会过度侵蚀）


def nonwhite_mask(gray: np.ndarray, white_threshold: int = DEFAULT_WHITE_THRESHOLD):
    """「非纯白」掩膜。True = 有内容（线、网点、灰底）。

    用 245（而不是墨迹阈值 200）是刻意的：只有**纸白**才算背景，
    这样格的**背景留白**会留在块内，不会把格切开。
    """
    return gray < int(white_threshold)


def _component_shapes(
    img: np.ndarray,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    min_area_ratio: float = DEFAULT_COMPONENT_MIN_AREA,
    close: int = DEFAULT_COMPONENT_CLOSE,
    open_k: int = DEFAULT_COMPONENT_OPEN,
    ink_threshold: int = 200,
    padding_px: int = 0,
    with_poly: bool = False,
) -> Tuple[List[Any], List[str]]:
    """非白连通域法：一个「非纯白」连通块 = 一个分镜格。

    为什么需要它
    ------------
    投影法把「某一行/列的墨迹少」当成格边界。但**格内空白**（竖排文字的栏间距、
    对话框内部、天空、页边留白）同样满足这个条件，**而且常常比真 gutter 更干净**
    （097 页实测：真 gutter 行墨迹 0.027~0.054，文字栏间的缝只有 0.009~0.020）
    → 一个格被切成多个。097 页共 12 刀，其中 8 刀砍在格内部。

    连通域法换判据：**看画面连不连通**。格内空白天然和画格处在同一个连通块里，
    因此不会制造假切点。

    实测（19 页逐格人工标注 / 64 页跨 8 卷抽样）
    ---------------------------------------------
    * 平均绝对偏差：投影法 2.89 → **1.74**
    * 「整页退化成 1 格」的页面：27% → **12%**
    * 「切得很碎（≥8 格）」的页面：22% → **3%**

    已知弱项
    --------
    * 相邻格若描边相接 / 网点连片，会被并成一块（斜切格 + 大面积黑底扉页）
    * 小块可能落在另一块的 bbox 内部（「框套框」）—— 调用方可用
      ``count_contained()`` 检测；不要简单删掉内层，内层往往才是真格

    参数
    ----
    white_threshold : 灰度 ≥ 此值算纸白（默认 245）
    min_area_ratio  : 小于整页此比例的块丢弃（默认 0.015）
    close           : 3x3 闭运算迭代数，弥合 1~2px 断缝（0 = 关闭）
    padding_px      : 向外扩的像素（默认 0，连通块 bbox 本身已贴合）
    """
    warns: List[str] = []
    gray = to_gray_u8(img)
    white_threshold = int(min(255, max(1, white_threshold)))

    if not HAS_CV2:
        warns.append(
            "缺少 OpenCV，连通域法不可用 —— 已自动回退到投影法（frame=projection）"
        )
        nw, w2 = _projection_boxes(
            gray, ink_threshold=ink_threshold, gutter_ratio=0.06,
            min_panel_px=80, min_gutter_px=8, min_panel_ratio=0.01, padding_px=4,
        )
        return ([{"box": [int(v) for v in b], "poly": None, "page": PAGE_ANY}
                 for b in nw], warns + w2)

    import cv2

    nw = nonwhite_mask(gray, white_threshold).astype(np.uint8)
    if int(open_k) > 0:
        # 开运算 = 先腐蚀后膨胀。它会**打断细框线**（1~2px 的格边框），
        # 而保留较粗的内容（人物线条、网点块）—— 于是被框线焊在一起的相邻格能分开。
        # 代价：格内容会被轻微侵蚀（调用方用 padding_px 补回来），且过强的核会过切。
        k = 2 * int(open_k) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        nw = cv2.morphologyEx(nw, cv2.MORPH_OPEN, ker)
    if int(close) > 0:
        nw = cv2.morphologyEx(
            nw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=int(close)
        )

    H, W = gray.shape
    page_area = float(H * W)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(nw, 4)

    boxes: List[Tuple[int, int, int, int]] = []
    shapes: List[Any] = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < page_area * float(min_area_ratio):
            continue
        if w <= 0 or h <= 0:
            continue
        pad = int(padding_px)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        box = (x0, y0, x2 - x0, y2 - y0)
        boxes.append(box)
        shapes.append({
            "box": [int(v) for v in box],
            "poly": (_component_poly(_lab, i, (x, y, w, h), W, H,
                                     restore=int(open_k)) if with_poly else None),
            "page": PAGE_ANY,
        })

    if not boxes:
        warns.append(
            "非白连通域法没找到任何块（整页近乎纯白？）—— 已退化为整页 1 格"
        )
        boxes = [(0, 0, W, H)]
        shapes = [{"box": [0, 0, W, H], "poly": None, "page": PAGE_ANY}]

    # ---- 「过大连通块」告警：把静默漏切变成显式提示 ----
    reason = merge_suspect(boxes, W, H, page_area)
    if reason:
        warns.append(
            f"{reason}—— 相邻分镜格可能通过相接/共用的黑框线（或大块黑底）连成了一块，"
            "**多半是漏切**。建议：① 该页改用手填（斜切版面请用**多边形**写法："
            "每行 8 个数 x1,y1,x2,y2,x3,y3,x4,y4）；"
            "② 或把 component_open 调大（用开运算打断连接）后重试"
        )
    return shapes, warns


def _component_poly(
    lab: np.ndarray, idx: int, bbox: Sequence[int], W: int, H: int,
    eps_ratio: float = 0.02, near_rect: float = 0.92, restore: int = 0,
) -> Optional[List[List[int]]]:
    """从连通块的轮廓提取多边形顶点（斜切格用）。

    轮廓用 ``approxPolyDP`` 逼近；若逼近结果基本就是个矩形
    （面积 / bbox 面积 ≥ ``near_rect``）就返回 None —— 那种情况用 box 裁更省事，
    也避免给每一格都挂上 4 个「其实等于矩形」的顶点。

    ``restore`` 是开运算半径：轮廓取自**开运算之后**的掩膜，边界被侵蚀了
    ``k`` 像素，所以先把它膨胀回去再取轮廓 —— 否则多边形会比真实格小一圈
    （box 侧已经有 padding 补回来了，poly 侧必须自己补）。
    """
    if not HAS_CV2:
        return None
    import cv2

    x, y, w, h = (int(v) for v in bbox)
    if w <= 2 or h <= 2:
        return None
    sub = (lab[y:y + h, x:x + w] == idx).astype(np.uint8)
    if int(restore) > 0:
        k = 2 * int(restore) + 1
        sub = cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    if peri <= 0:
        return None
    ap = cv2.approxPolyDP(c, float(eps_ratio) * peri, True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    if polygon_area(ap) / float(max(1, w * h)) >= float(near_rect):
        return None
    return [[int(px) + x, int(py) + y] for px, py in ap]


def split_by_components(
    img: np.ndarray,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    min_area_ratio: float = DEFAULT_COMPONENT_MIN_AREA,
    close: int = DEFAULT_COMPONENT_CLOSE,
    open_k: int = DEFAULT_COMPONENT_OPEN,
    ink_threshold: int = 200,
    padding_px: int = 0,
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """矩形版入口（向后兼容）。要多边形用 ``split_by_components_shapes``。"""
    shapes, warns = _component_shapes(
        img, white_threshold=white_threshold, min_area_ratio=min_area_ratio,
        close=close, open_k=open_k, ink_threshold=ink_threshold,
        padding_px=padding_px, with_poly=False,
    )
    return [tuple(int(v) for v in s["box"]) for s in shapes], warns


def split_by_components_shapes(
    img: np.ndarray,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    min_area_ratio: float = DEFAULT_COMPONENT_MIN_AREA,
    close: int = DEFAULT_COMPONENT_CLOSE,
    open_k: int = DEFAULT_COMPONENT_OPEN,
    ink_threshold: int = 200,
    padding_px: int = 0,
) -> Tuple[List[Any], List[str]]:
    """带多边形的连通域法：形状 = ``{"box": [...], "poly": [[x,y],...] | None}``。"""
    return _component_shapes(
        img, white_threshold=white_threshold, min_area_ratio=min_area_ratio,
        close=close, open_k=open_k, ink_threshold=ink_threshold,
        padding_px=padding_px, with_poly=True,
    )


def merge_suspect(
    boxes: Sequence[Sequence[int]], W: int, H: int, page_area: float
) -> Optional[str]:
    """检测「多半是漏切」的迹象，返回原因描述；正常则返回 None。

    判据刻意用**版面跨度**，不用「比别的格大」—— 后者会把真的大格误报
    （第一版写成 `≥2× 中位面积`，被合成测试立刻抓出假阳性）。

    真实事故 Reborn_10_081：一个大块 bbox 占页面 71%（宽 94% × 高 76%），
    而它自己的像素只占 38% —— 是一张细线网，实为 3 格焊在一起。
    """
    if not boxes:
        return None
    if len(boxes) == 1:
        b = boxes[0]
        cover = b[2] * b[3] / max(1.0, page_area)
        if cover >= 0.95:
            return f"整页只切出 1 格且占满整页（{cover * 100:.0f}%）"
        return None
    worst = None
    for b in boxes:
        bw = b[2] / float(W)
        bh = b[3] / float(H)
        if (bw >= 0.85 and bh >= 0.45) or (bh >= 0.85 and bw >= 0.45):
            cov = b[2] * b[3] / max(1.0, page_area)
            if worst is None or cov > worst[0]:
                worst = (cov, bw, bh)
    if worst is None:
        return None
    return (f"检测到「过大连通块」，最大者 bbox 占页面 {worst[0] * 100:.0f}%"
            f"（宽 {worst[1] * 100:.0f}% × 高 {worst[2] * 100:.0f}%）")


def _projection_boxes(
    gray: np.ndarray,
    ink_threshold: int = 200,
    gutter_ratio: float = 0.06,
    min_panel_px: int = 80,
    min_gutter_px: int = 8,
    min_panel_ratio: float = 0.01,
    padding_px: int = 4,
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """投影法（原来的 auto 通道），抽成函数以便被两处调用。"""
    warns: List[str] = []
    H, W = gray.shape
    mask = ink_mask(gray, ink_threshold)
    raw = _recursive_split(
        mask, (0, 0, W, H), gutter_ratio, min_panel_px, min_gutter_px
    )
    tightened = [_tighten(mask, b, padding_px) for b in raw]
    boxes = _merge_fragments(tightened, min_panel_ratio, float(H * W))
    if len(boxes) == 1 and len(raw) > 1:
        warns.append("所有碎片被合并成一格——请调低 min_panel_ratio 或 min_gutter_px")
    return boxes, warns


# --------------------------------------------------------------------------
# 去重叠 / 框套框检测
# --------------------------------------------------------------------------
def _rect_overlap(b1, b2) -> Tuple[int, int]:
    """两框交集的 (宽, 高)；不相交则为 0。"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ox = min(x1 + w1, x2 + w2) - max(x1, x2)
    oy = min(y1 + h1, y2 + h2) - max(y1, y2)
    return (max(0, ox), max(0, oy))


def count_overlaps(boxes: Sequence[Sequence[int]]) -> int:
    """统计互相重叠（交集非空）的框对数。"""
    n = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ox, oy = _rect_overlap(boxes[i], boxes[j])
            if ox > 0 and oy > 0:
                n += 1
    return n


def count_contained(boxes: Sequence[Sequence[int]], thr: float = 0.85) -> int:
    """统计「小框 ≥thr 落在另一个框内」的对数（框套框）。

    注意：这**不是**简单的冗余。内层往往才是真分镜格，
    错的是外层那个「把好几格并进去」的超集框 → 正确处理是**切开外层**。
    """
    n = 0
    for j in range(len(boxes)):
        x2, y2, w2, h2 = boxes[j]
        a2 = float(w2 * h2)
        for i in range(len(boxes)):
            if i == j:
                continue
            x1, y1, w1, h1 = boxes[i]
            ox, oy = _rect_overlap(boxes[i], boxes[j])
            if ox > 0 and oy > 0 and (ox * oy) / max(1.0, a2) >= thr \
                    and a2 < float(w1 * h1) * 0.98:
                n += 1
    return n


def deoverlap_boxes(
    boxes: Sequence[Sequence[int]], iters: int = 12
) -> List[Tuple[int, int, int, int]]:
    """把互相重叠的矩形沿「重叠较小的那根轴」推到中线，直到互不重叠。

    重叠从哪来
    ----------
    `_tighten()` 会把框向外放几像素、碎片合并取并集 → 相邻框的 bbox 互相侵入，
    同一批像素被裁进两个镜头。实测现状算法 19 页有 46 对重叠，
    最严重的「侵入比」达 0.996（一个框几乎完全被另一个框盖住）。

    做法：只动**被侵入的那条边**，框数与阅读顺序都不变。
    实测：19 页重叠对数 46 → 0（投影法）/ 26 → 0（连通域法），
    平均绝对偏差完全不变（零精度代价）。
    """
    b = [list(map(int, x)) for x in boxes]
    for _ in range(int(iters)):
        changed = False
        for i in range(len(b)):
            for j in range(len(b)):
                if i == j:
                    continue
                x1, y1, w1, h1 = b[i]
                x2, y2, w2, h2 = b[j]
                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if ox <= 0 or oy <= 0:
                    continue
                if ox <= oy:
                    mid = (max(x1, x2) + min(x1 + w1, x2 + w2)) // 2
                    if x1 < x2:                       # i 在左，j 在右
                        b[i][2] = max(1, mid - x1)
                        dx = mid - x2
                        b[j][0] = mid
                        b[j][2] = max(1, w2 - dx)
                    else:
                        b[j][2] = max(1, mid - x2)
                        dx = mid - x1
                        b[i][0] = mid
                        b[i][2] = max(1, w1 - dx)
                else:
                    mid = (max(y1, y2) + min(y1 + h1, y2 + h2)) // 2
                    if y1 < y2:                       # i 在上，j 在下
                        b[i][3] = max(1, mid - y1)
                        dy = mid - y2
                        b[j][1] = mid
                        b[j][3] = max(1, h2 - dy)
                    else:
                        b[j][3] = max(1, mid - y2)
                        dy = mid - y1
                        b[i][1] = mid
                        b[i][3] = max(1, h1 - dy)
                changed = True
        if not changed:
            break
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in b]


# --------------------------------------------------------------------------
# 对外主入口
# --------------------------------------------------------------------------
def split_page_shapes(
    img: np.ndarray,
    mode: str = "auto",
    ink_threshold: int = 200,
    gutter_ratio: float = 0.06,
    min_panel_px: int = 80,
    min_gutter_px: int = 8,
    min_panel_ratio: float = 0.01,
    reading_order: str = "rtl",
    padding_px: int = 4,
    manual_boxes: Optional[Sequence[Sequence[int]]] = None,
    merge_tolerance: float = 0.0,
    page_index: int = PAGE_ANY,
    frame: str = "component",
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    component_min_area: float = DEFAULT_COMPONENT_MIN_AREA,
    component_close: int = DEFAULT_COMPONENT_CLOSE,
    component_open: int = DEFAULT_COMPONENT_OPEN,
    component_auto_open: bool = True,
    deoverlap: bool = True,
    manual_shapes: Optional[Sequence[dict]] = None,
    panel_polygon: str = "auto",
) -> Tuple[List[Any], List[str]]:
    """切一页，返回 **(形状列表, warnings)**。

    形状 = ``{"page": p, "box": [x, y, w, h], "poly": [[x, y], ...] | None}``。

    ``poly`` 非 None 表示这是一个**非矩形分镜格**（斜切版面）。裁图时必须按
    多边形掩膜（``crop_shapes`` / ``crop_polygons``），否则 bbox 一定会把邻居
    的内容带进来 —— 斜格的 bbox 与邻格的 bbox 相交是几何必然，不是算法问题。
    矩形格 ``poly`` 为 None，行为与旧版完全一致。

    mode / frame / deoverlap 的语义同 ``split_page_boxes``；

    panel_polygon:
      ``auto``（默认）—— 连通域法里自动给「明显不是矩形」的块配多边形
      ``off``         —— 完全不用多边形（等价于旧行为）

    manual_shapes 优先于 manual_boxes；两者都支持 ``page = -1``（套用到所有页）。
    """
    warns: List[str] = []
    gray = to_gray_u8(img)
    H, W = gray.shape[:2]
    page_area = H * W

    want_poly = str(panel_polygon).lower() != "off"
    shapes: List[Any] = []

    if mode in ("auto", "hybrid"):
        # 页级体检：两种 frame 都会受影响，所以放在分支之前统一报
        ink_ratio = float(ink_mask(gray, ink_threshold).mean())
        if ink_ratio > 0.85:
            warns.append(
                "整页墨迹占比过高（>85%），自动切格可能失效——"
                "建议改用手动框或降低 ink_threshold"
            )

        if str(frame).lower() == "projection":
            pb, w = _projection_boxes(
                gray, ink_threshold=ink_threshold, gutter_ratio=gutter_ratio,
                min_panel_px=min_panel_px, min_gutter_px=min_gutter_px,
                min_panel_ratio=min_panel_ratio, padding_px=padding_px,
            )
            shapes = [{"page": PAGE_ANY, "box": [int(v) for v in b], "poly": None}
                      for b in pb]
            warns.extend(w)
        else:
            k_fixed = int(component_open)
            if k_fixed > 0 or not component_auto_open:
                shapes, w = _component_shapes(
                    gray, white_threshold=white_threshold,
                    min_area_ratio=component_min_area, close=component_close,
                    open_k=k_fixed, ink_threshold=ink_threshold,
                    padding_px=0, with_poly=want_poly,
                )
                warns.extend(w)
            else:
                # ★ 自动开运算：**逐页**把半径调到「漏切迹象消失」为止。
                # 为什么必须自动：实测 Reborn_10_081 的连接组织在 k=1 时还在
                #（最大块 798469→757747 几乎没变），要到 k=2 才断开；而别的页
                # 用 k=2 会过切（19 页基准 1.74→3.05）。所以半径只能逐页定，
                # 而「漏切检测器」正好可以当这个调优的目标函数。
                shapes, k_used, reason = [], 0, None
                clean: List[str] = []
                cands = []
                for kk in range(0, int(DEFAULT_AUTO_OPEN_MAX) + 1):
                    sx, bwx = _component_shapes(
                        gray, white_threshold=white_threshold,
                        min_area_ratio=component_min_area, close=component_close,
                        open_k=kk, ink_threshold=ink_threshold,
                        padding_px=kk,          # 补回开运算侵蚀掉的边界像素
                        with_poly=want_poly,
                    )
                    bb = [(s["box"][0], s["box"][1], s["box"][2], s["box"][3])
                          for s in sx]
                    cands.append((kk, sx, bwx,
                                  merge_suspect(bb, W, H, float(page_area))))
                    if kk == 0 and cands[0][3] is None:
                        break                   # 本来就正常 → 不动它
                # ★ 守卫：只认「框数不少于基准」的候选 —— 开运算只许**细分**，
                # 不许吞掉原来的格。否则会出现退化解：加大半径把内容细的小格
                # 整个腐蚀掉，「漏切告警消失」其实是被删格删出来的
                #（实测 108 页 k=1 时 4 格→1 格；那一格仍是「整页 1 格」的
                #  可疑态，于是被这条守卫 + 可疑判据一起挡回 k=0）。
                base_n = len(cands[0][1])
                safe = [c for c in cands if len(c[1]) >= base_n]
                pick = next((c for c in safe if c[3] is None), None)
                if pick is None and safe:
                    pick = max(safe, key=lambda c: (len(c[1]), -c[0]))
                if pick is None:
                    pick = cands[0]
                k_used, shapes, bwx, reason = pick
                if reason is None:
                    # 只保留与本轮调优无关的 warning（如「近乎纯白」），
                    # 避免把调优过程中的中间态告警混进最终报告
                    clean = [x for x in bwx
                             if "过大连通块" not in x and "整页只切出 1 格" not in x]
                warns.extend(clean)
                if k_used > 0:
                    warns.append(
                        f"自动开运算：component_open={k_used}"
                        "（检测到漏切迹象后逐级尝试，用于打断相邻格之间的连接）"
                    )
                if reason is not None:
                    warns.append(
                        f"{reason}—— 已试到 component_open={k_used} 仍未消除，"
                        "该页多半需要手填分镜格（斜切版面用**多边形**写法："
                        "每行 8 个数 x1,y1,x2,y2,x3,y3,x4,y4）"
                    )

    if mode in ("manual", "hybrid"):
        # 只取「本页」的形状：page == -1 表示通用。
        # manual_shapes（可带多边形）优先；否则退回 manual_boxes（纯矩形）。
        cand: List[Any] = []
        for s in (manual_shapes or []):
            p = int(s.get("page", PAGE_ANY))
            if p == PAGE_ANY or page_index == PAGE_ANY or p == page_index:
                cand.append(s)
        if not cand and manual_boxes:
            cand = [
                {"page": int(b[0]), "box": [int(v) for v in b[1:]], "poly": None}
                for b in manual_boxes
                if len(b) == 5 and (int(b[0]) == PAGE_ANY
                                    or page_index == PAGE_ANY
                                    or int(b[0]) == page_index)
            ]

        if cand:
            for i, s in enumerate(cand):
                x, y, w, h = (int(v) for v in s["box"])
                if w <= 0 or h <= 0:
                    warns.append(f"手动分镜格 #{i} 宽高非正，已跳过")
                    continue
                x = max(0, min(x, W - 1))
                y = max(0, min(y, H - 1))
                w = max(1, min(w, W - x))
                h = max(1, min(h, H - y))
                poly = ([[int(p[0]), int(p[1])] for p in s["poly"]]
                        if s.get("poly") else None)
                if poly:
                    poly = clamp_poly(poly, W, H)
                    x, y, w, h = poly_to_box(poly)
                if mode == "hybrid":
                    # 保留与手动形状**不相交**的自动形状（相交的被手动的取代）。
                    # 注意：下面是「分离」判定，所以命中即保留 —— 早先版本写成
                    # `if not (...)` 把逻辑取反了，会把不相关的框一起删掉。
                    shapes = [
                        t for t in shapes
                        if (x + w <= t["box"][0] or t["box"][0] + t["box"][2] <= x
                            or y + h <= t["box"][1] or t["box"][1] + t["box"][3] <= y)
                    ]
                shapes.append({"page": int(s.get("page", PAGE_ANY)),
                               "box": [x, y, w, h], "poly": poly})
        elif mode == "manual":
            warns.append("mode=manual 但本页没有可用的手动分镜格，返回空结果")

    if not shapes:
        return [], warns

    if merge_tolerance > 0:
        if all(s.get("poly") is None for s in shapes):
            bb = [tuple(int(v) for v in s["box"]) for s in shapes]
            merged = [list(b) for b in _merge_fragments(bb, merge_tolerance, page_area)]
            if len(merged) == len(shapes):
                for s, b in zip(shapes, merged):
                    s["box"] = [int(v) for v in b]
            else:
                shapes = [{"page": PAGE_ANY, "box": b, "poly": None} for b in merged]
        else:
            warns.append("merge_tolerance 与多边形分镜格同时使用时已跳过合并")

    if deoverlap:
        # 只对**矩形**格去重叠：多边形格用自己的掩膜裁，bbox 相交无害；
        # 若把它一起推开，box 就不再是它的 bbox 了（报告 / 阅读顺序会失真）。
        idx = [i for i, s in enumerate(shapes) if s.get("poly") is None]
        if idx:
            rb = [shapes[i]["box"] for i in idx]
            n_ov = count_overlaps(rb)
            if n_ov:
                nb = deoverlap_boxes(rb)
                for i, b in zip(idx, nb):
                    shapes[i]["box"] = [int(v) for v in b]
                warns.append(f"检测到 {n_ov} 对分镜格重叠，已自动分离（可关掉 deoverlap）")

    order = _reading_order([s["box"] for s in shapes], reading_order)
    return [shapes[i] for i in order], warns


def split_page_boxes(
    img: np.ndarray,
    mode: str = "auto",
    ink_threshold: int = 200,
    gutter_ratio: float = 0.06,
    min_panel_px: int = 80,
    min_gutter_px: int = 8,
    min_panel_ratio: float = 0.01,
    reading_order: str = "rtl",
    padding_px: int = 4,
    manual_boxes: Optional[Sequence[Sequence[int]]] = None,
    merge_tolerance: float = 0.0,
    page_index: int = PAGE_ANY,
    frame: str = "component",
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    component_min_area: float = DEFAULT_COMPONENT_MIN_AREA,
    component_close: int = DEFAULT_COMPONENT_CLOSE,
    component_open: int = DEFAULT_COMPONENT_OPEN,
    component_auto_open: bool = True,
    deoverlap: bool = True,
    manual_shapes: Optional[Sequence[dict]] = None,
    panel_polygon: str = "off",
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """切一页。返回 (boxes, warnings)。

    **矩形版入口**（向后兼容）：即使某个格子被识别为多边形，这里也只返回它的
    bbox。需要真实形状请用 ``split_page_shapes``（或用 ``shapes_from_boxes``）。

    mode:
      auto   —— 纯自动
      manual —— 只用手动分镜格（自动结果丢弃）
      hybrid —— 自动为主，手动分镜格作为「强制追加/替换」的补充

    frame:
      ``component``（默认）—— **非白连通域法**。一个「非纯白」连通块 = 一格。
                 对「格内空白被当成装订线」这类过切免疫（见 split_by_components）。
      ``projection``       —— 原来的整行/整列投影递归切分。留给「格与格之间
                 完全没有背景留白」的页面，或作为连通域法不可用时的降级。

    deoverlap:
      切完之后把互相重叠的框沿中线分开（`_tighten` 的外扩 + 碎片合并取并集
      会让相邻框 bbox 互相侵入）。默认开，实测零精度代价。

    manual_boxes 每项是 ``[page, x, y, w, h]``（``parse_manual_boxes`` 的产物）；
    ``page = -1`` 的项套用到所有页，否则只在该页生效。
    """
    shapes, warns = split_page_shapes(
        img, mode=mode, ink_threshold=ink_threshold, gutter_ratio=gutter_ratio,
        min_panel_px=min_panel_px, min_gutter_px=min_gutter_px,
        min_panel_ratio=min_panel_ratio, reading_order=reading_order,
        padding_px=padding_px, manual_boxes=manual_boxes,
        merge_tolerance=merge_tolerance, page_index=page_index, frame=frame,
        white_threshold=white_threshold, component_min_area=component_min_area,
        component_close=component_close, component_open=component_open,
        component_auto_open=component_auto_open, deoverlap=deoverlap,
        manual_shapes=manual_shapes, panel_polygon=panel_polygon,
    )
    return [tuple(int(v) for v in s["box"]) for s in shapes], warns


def shapes_from_boxes(boxes: Sequence[Sequence[int]],
                      page: int = PAGE_ANY) -> List[dict]:
    """把纯矩形框列表包成形状列表（便于统一走 ``crop_shapes``）。"""
    return [{"page": int(page), "box": [int(v) for v in b], "poly": None}
            for b in boxes]


def crop_boxes(img: np.ndarray, boxes: Sequence[Sequence[int]]) -> List[np.ndarray]:
    """按框裁剪（纯 numpy 视图拷贝）。"""
    a = np.asarray(img)
    out = []
    H, W = a.shape[:2]
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, x0 + int(w)), min(H, y0 + int(h))
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(a[y0:y1, x0:x1].copy())
    return out


def _norm_box_list(data, warns: List[str], page: int = PAGE_ANY) -> List[List[int]]:
    """把 JSON 解析出来的东西归一成 [[page, x, y, w, h], ...]。"""
    out: List[List[int]] = []
    if not isinstance(data, (list, tuple)):
        return out

    # 单个框 [x,y,w,h]
    if len(data) == 4 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in data
    ):
        return [[page] + [int(v) for v in data]]

    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 4 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in item
        ):
            out.append([page] + [int(v) for v in item])
        else:
            warns.append(f"手动框里有无法解析的条目，已跳过：{str(item)[:40]}")
    return out


def parse_manual_boxes(text: str) -> Tuple[List[List[int]], List[str]]:
    """解析手动框文本。每项返回 ``[page, x, y, w, h]``，``page = -1`` 表示套用到所有页。

    支持四种写法混用：

        x,y,w,h                     # 每行一个（套用到所有页）
        page0: x,y,w,h              # 只套用到第 0 页（多页输入时必须用它）
        [x,y,w,h]                   # 每行一个 JSON 数组
        [[x,y,w,h],[x,y,w,h]]       # 单行多个 / 整个文本就是一个 JSON 数组

    解析顺序：先整体 JSON → 再逐行 JSON → 再逐行纯数字。
    任何一种失败都只记 warning，不影响其它行。
    """
    shapes, w = parse_manual_shapes(text)
    # 老 API 只返回矩形（``[page, x, y, w, h]``）；多边形项返回它的 bbox，
    # 需要真实顶点请改用 parse_manual_shapes。
    return [[s["page"]] + [int(v) for v in s["box"]] for s in shapes], w


def upscale_boxes(boxes: Sequence[Sequence[int]], factor: float) -> List[List[int]]:
    """把框按比例放大（用于「在缩略图上标框、在原图上切」的场景）。"""
    f = float(factor)
    return [[int(round(v * f)) for v in b] for b in boxes]


def upscale_shapes(shapes: Sequence[dict], factor: float) -> List[dict]:
    """按比例放大形状（矩形与多边形都处理；box 永远与 poly 保持一致）。"""
    f = float(factor)
    out: List[dict] = []
    for s in shapes:
        poly = s.get("poly")
        if poly:
            pts = [[int(round(px * f)), int(round(py * f))] for px, py in poly]
            out.append({"page": s.get("page", PAGE_ANY), "poly": pts,
                        "box": poly_to_box(pts)})
        else:
            box = [int(round(v * f)) for v in s["box"]]
            out.append({"page": s.get("page", PAGE_ANY), "box": box, "poly": None})
    return out


# --------------------------------------------------------------------------
# 多边形分镜格（斜切版面）
# --------------------------------------------------------------------------
# 为什么必须有它
# --------------
# 斜切分镜（格子是斜的四边形）**在矩形里无解**，这不是算法不够好，是几何：
# 一个斜格的 bbox 必然与相邻格的 bbox 相交 —— 裁哪边都只能二选一，
# 要么漏掉自己的一角，要么把邻居的内容裁进来。
#
# 真实样本 Reborn_10_134：左上「手臂/海面」格与右上「臉」格之间是一条
# **51° 的粗黑斜带**（实测 (239,0)→(761,644)，长 829px，全页最长的直线）。
# 任何轴对齐矩形都切不出这两格。而且连「手填框」这条退路也走不通 ——
# 老版手动框解析器只收 4 个数（x,y,w,h），8 个数会被当成语法错误丢掉。
#
# 所以这里把「形状」升级为可选的任意多边形：
#   * 形状 = {"page": p, "box": [x,y,w,h], "poly": [[x,y], ...] | None}
#   * 裁剪时把多边形以外的像素填成纸白 → **输出仍是矩形图像**
#   * 因此下游（参考图槽位 / `<Picture i>` 序号契约）一行都不用改


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    """鞋带公式。"""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def poly_to_box(poly: Sequence[Sequence[int]]) -> List[int]:
    """多边形的包围盒 ``[x, y, w, h]``（至少 1x1）。"""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def clamp_poly(poly: Sequence[Sequence[int]], W: int, H: int) -> List[List[int]]:
    """把顶点夹进页面范围。

    上界用 ``W`` / ``H``（页边）而不是 ``W-1`` / ``H-1``：一个分镜格完全可能
    一直画到页边（出血），夹成 ``W-1`` 会让裁出来的图**少一行一列**，
    于是「多边形裁剪的输出尺寸」就不再等于它的 bbox 尺寸了。
    掩膜填充（fillPoly）对越界顶点本身是安全的。
    """
    return [[max(0, min(int(x), W)), max(0, min(int(y), H))] for x, y in poly]


def _fill_poly_mask(H: int, W: int, pts: np.ndarray) -> np.ndarray:
    """多边形内部掩膜。有 cv2 用 fillPoly，否则退回扫描线（偶奇规则）。"""
    if HAS_CV2:
        import cv2
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [pts.reshape(-1, 1, 2).astype(np.int32)], 1)
        return m
    m = np.zeros((H, W), dtype=np.uint8)
    n = len(pts)
    for y in range(int(pts[:, 1].min()), int(pts[:, 1].max()) + 1):
        xs = []
        for i in range(n):
            x1, y1 = float(pts[i][0]), float(pts[i][1])
            x2, y2 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
            if (y1 <= y < y2) or (y2 <= y < y1):
                xs.append(x1 + (y - y1) / (y2 - y1) * (x2 - x1))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            a0 = max(0, int(math.ceil(xs[i])))
            a1 = min(W, int(math.floor(xs[i + 1])) + 1)
            if a1 > a0:
                m[y, a0:a1] = 1
    return m


def crop_polygons(
    img: np.ndarray, polys: Sequence[Sequence[Sequence[int]]], fill: int = 255
) -> List[np.ndarray]:
    """按多边形裁剪：多边形以外填成纸白，再裁到它的 bbox。

    ``fill`` 对 float 输入按 0~1 处理，对整型输入按 0~255 处理。
    """
    a = np.asarray(img)
    H, W = a.shape[:2]
    is_float = a.dtype.kind == "f"
    fv = (float(fill) / 255.0) if is_float else fill
    out: List[np.ndarray] = []
    for poly in polys:
        pts = np.asarray(clamp_poly(poly, W, H), dtype=np.float64)
        m = _fill_poly_mask(H, W, pts).astype(bool)
        x, y, w, h = poly_to_box(pts)
        sub = a[y:y + h, x:x + w].copy()
        msk = m[y:y + h, x:x + w]
        if sub.ndim == 3:
            msk = msk[..., None]
        out.append(np.where(msk, sub, fv).astype(a.dtype))
    return out


def crop_shapes(img: np.ndarray, shapes: Sequence[dict], fill: int = 255) -> List[np.ndarray]:
    """按形状列表裁剪（矩形走切片，多边形走掩膜）。与 ``shapes`` 一一对应。"""
    if not shapes:
        return []
    if all(s.get("poly") is None for s in shapes):
        return crop_boxes(img, [s["box"] for s in shapes])
    a = np.asarray(img)
    H, W = a.shape[:2]
    out: List[np.ndarray] = []
    for s in shapes:
        poly = s.get("poly")
        if poly:
            out.extend(crop_polygons(a, [poly], fill=fill))
        else:
            x, y, w, h = (int(v) for v in s["box"])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x0 + w), min(H, y0 + h)
            if x1 <= x0 or y1 <= y0:
                continue
            out.append(a[y0:y1, x0:x1].copy())
    return out


def shape_key(shape: dict) -> str:
    """形状的稳定指纹（写报告 / 排查用）。"""
    if shape.get("poly"):
        return "poly:" + " ".join(f"{int(p[0])},{int(p[1])}" for p in shape["poly"])
    x, y, w, h = (int(v) for v in shape["box"])
    return f"rect:{x},{y},{w},{h}"


def _shape_from_nums(page: int, nums: Sequence[int], warns: List[str], src: str) -> Optional[dict]:
    """4 个数 = 矩形；≥6 且为偶数 = 多边形（每两点一个顶点）。"""
    n = len(nums)
    if n == 4:
        return {"page": page, "box": [int(v) for v in nums], "poly": None}
    if n >= 6 and n % 2 == 0:
        pts = [[int(nums[i]), int(nums[i + 1])] for i in range(0, n, 2)]
        return {"page": page, "box": poly_to_box(pts), "poly": pts}
    warns.append(
        f"手动分镜格「{src[:40]}」需要 4 个数（矩形 x,y,w,h）"
        f"或 ≥6 个偶数个数（多边形 x1,y1,x2,y2,...），实得 {n} 个数，已跳过"
    )
    return None


def _shape_from_json(item, warns: List[str], page: int = PAGE_ANY) -> Optional[dict]:
    """JSON 条目：``[x,y,w,h]`` 或 ``[[x,y],[x,y],...]``。"""
    if not isinstance(item, (list, tuple)):
        return None
    # 点列表
    if len(item) >= 3 and all(
        isinstance(p, (list, tuple)) and len(p) == 2 for p in item
    ):
        pts = [[int(p[0]), int(p[1])] for p in item]
        return {"page": page, "box": poly_to_box(pts), "poly": pts}
    # 扁平数字
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in item):
        return _shape_from_nums(page, [int(v) for v in item], warns, str(item))
    return None


def _json_to_shapes(data, warns: List[str], page: int = PAGE_ANY) -> List[dict]:
    """JSON → 形状列表。必须区分两种「嵌套列表」：

    * ``[[x,y],[x,y],[x,y]]`` —— **一个**多边形（每项是长度 2 的点）
    * ``[[x,y,w,h],[x,y,w,h]]`` —— **多个**矩形

    早先只看 ``data[0] 是不是 list`` 就当成多项，于是多边形被拆成 3 个
    「长度 2 的矩形」并全部丢掉（这条是写测试时才发现的）。
    """
    if isinstance(data, (list, tuple)) and len(data) >= 3 and all(
        isinstance(p, (list, tuple)) and len(p) == 2 for p in data
    ):
        s = _shape_from_json(data, warns, page)
        return [s] if s else []
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        return [s for s in (_shape_from_json(it, warns, page) for it in data) if s]
    s = _shape_from_json(data, warns, page)
    return [s] if s else []


def parse_manual_shapes(text: str) -> Tuple[List[dict], List[str]]:
    """解析手动分镜格文本，**矩形与任意多边形可混用**。

    每项返回 ``{"page": p, "box": [x, y, w, h], "poly": [[x, y], ...] | None}``。
    ``page = -1`` 表示套用到所有页。支持的写法：

        10,10,100,100                            # 矩形 x,y,w,h
        240,0,761,644,700,780,240,780            # 四边形（斜切格），4 个顶点、8 个数
        100,50,300,50,300,200                    # 任意多边形（≥3 个顶点，个数为偶数）
        page0: 240,0,761,644,700,780,240,780     # 只作用于第 0 页
        [10,10,100,100]                          # JSON 矩形
        [[240,0],[761,644],[700,780],[240,780]]  # JSON 多边形

    顶点顺序不限（顺/逆时针都行）。
    """
    import json
    import re

    warns: List[str] = []
    text = (text or "").strip()
    if not text:
        return [], warns

    def _split_page(line: str) -> Tuple[int, str]:
        m = re.match(r"^\s*(?:page|页)\s*[:：]?\s*(-?\d+)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            return int(m.group(1)), m.group(2).strip()
        return PAGE_ANY, line

    # 1) 整体就是一个 JSON（无 page 前缀）
    if text.startswith("["):
        try:
            data = json.loads(text)
            got = _json_to_shapes(data, warns)
            if got:
                return got, warns
        except Exception:
            pass

    shapes: List[dict] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().rstrip(",").strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        page, body = _split_page(line)

        if body.startswith("["):
            try:
                data = json.loads(body)
                got = _json_to_shapes(data, warns, page)
                if got:
                    shapes.extend(got)
                    continue
            except Exception:
                pass

        nums = re.findall(r"-?\d+", body)
        s = _shape_from_nums(page, nums, warns, line)
        if s is not None:
            shapes.append(s)

    return shapes, warns


def shape_to_panel_kwargs(shape: dict) -> dict:
    """给 ``types.Panel`` 用的字段（box + poly）。"""
    return {"box": [int(v) for v in shape["box"]],
            "poly": [[int(p[0]), int(p[1])] for p in shape["poly"]] if shape.get("poly") else None}

