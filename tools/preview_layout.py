"""把工作流 JSON 画成布局示意图（PNG）。

用途
----
改了 ``tools/make_example_workflow.py`` 的矩阵布局参数后，**不用重启 ComfyUI**
就能先看一眼排布对不对：group 是否落成网格、有没有重叠、节点有没有越出框。

刻意做得很轻：只读 JSON、只画方框与文字（PIL），不依赖 ComfyUI。

用法
----
    <ComfyUI>/<standalone-env>/python.exe tools/preview_layout.py --out-dir <目录>

默认输出 ``layout_<工作流名>.png``。窗口宽 ``--width``（默认 1900）。
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
TEMPLATES = os.path.join(PKG, "templates")

def _discover_workflows():
    """自动发现 templates/ 下的**所有**工作流 json。

    ★ 不写死列表：写死的话每加一版工作流都要记得回来改（与「白名单」同类问题，
      漏改的后果是「新版本一直没被预览过」这种静默缺失）。
      用内容探测排除 camera_presets.json / shot_list_template 这类数据文件。
    """
    import json
    out = []
    for f in sorted(os.listdir(TEMPLATES)):
        if not f.endswith(".json"):
            continue
        p = os.path.join(TEMPLATES, f)
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and d.get("nodes") and "links" in d:
            out.append((f, f))
    return out


WORKFLOWS = _discover_workflows()

PAD = 46
BG = (250, 251, 252)
NODE_FILL = (255, 255, 255)
NODE_EDGE = (176, 182, 190)
NODE_TEXT = (52, 58, 66)
GRID_LINE = (226, 230, 235)


def _font(size: int):
    from PIL import ImageFont
    for name in ("msyh.ttc", "simhei.ttf", "Deng.ttf", "simsun.ttc"):
        p = os.path.join(r"C:\Windows\Fonts", name)
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _hex_rgb(h: str):
    h = (h or "#888888").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (136, 136, 136)


def _tint(rgb, ratio: float):
    """把颜色往白里调（ratio=0 原色，1 全白）。"""
    return tuple(int(c + (255 - c) * ratio) for c in rgb)


def draw_workflow(path: str, title: str, width: int, dpi_scale: float = 1.0):
    from PIL import Image, ImageDraw

    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    nodes, groups = wf["nodes"], wf.get("groups", [])

    xs0 = min(n["pos"][0] for n in nodes)
    ys0 = min(n["pos"][1] for n in nodes)
    xs1 = max(n["pos"][0] + n["size"][0] for n in nodes)
    ys1 = max(n["pos"][1] + n["size"][1] for n in nodes)
    # group 框可能比节点更外扩
    for g in groups:
        gx, gy, gw, gh = g["bounding"]
        xs0, ys0 = min(xs0, gx), min(ys0, gy)
        xs1, ys1 = max(xs1, gx + gw), max(ys1, gy + gh)

    span_w, span_h = xs1 - xs0, ys1 - ys0
    scale = (width - PAD * 2) / max(1.0, span_w)
    out_w = int(width)
    out_h = int(span_h * scale + PAD * 2 + 54)

    img = Image.new("RGB", (out_w, out_h), BG)
    d = ImageDraw.Draw(img, "RGBA")

    f_head = _font(int(30 * dpi_scale))
    f_gt = _font(int(21 * dpi_scale))
    f_nt = _font(int(15 * dpi_scale))
    f_sm = _font(int(12 * dpi_scale))

    d.text((PAD, 16), f"{title}　·　{len(nodes)} 节点 / {len(groups)} 组　"
                      f"跨度 {int(span_w)}×{int(span_h)}",
           font=f_head, fill=(28, 32, 38))

    def tx(v):
        return PAD + (v - xs0) * scale

    def ty(v):
        return 54 + PAD + (v - ys0) * scale

    # ---- group 框（底层） ----
    for gi, g in enumerate(groups, start=1):
        gx, gy, gw, gh = g["bounding"]
        x0, y0, x1, y1 = tx(gx), ty(gy), tx(gx + gw), ty(gy + gh)
        rgb = _hex_rgb(g.get("color", "#888888"))
        d.rectangle([x0, y0, x1, y1], fill=_tint(rgb, 0.90),
                    outline=_tint(rgb, 0.30), width=3)
        d.text((x0 + 12, y0 + 8), f"{g['title']}", font=f_gt,
               fill=_tint(rgb, 0.05))
        d.text((x0 + 12, y0 + 8 + int(24 * dpi_scale)),
               f"{int(gw)}×{int(gh)}", font=f_sm, fill=_tint(rgb, 0.45))

    # ---- 节点（上层） ----
    for n in nodes:
        x, y = n["pos"]
        w, h = n["size"]
        x0, y0, x1, y1 = tx(x), ty(y), tx(x + w), ty(y + h)
        d.rectangle([x0, y0, x1, y1], fill=NODE_FILL, outline=NODE_EDGE, width=2)

        label = n.get("title") or n["type"]
        # 手动折行（节点框小，标题长）
        max_chars = max(6, int((x1 - x0) / (7.6 * dpi_scale)))
        lines, cur = [], ""
        for ch in label:
            cur += ch
            if len(cur) >= max_chars:
                lines.append(cur)
                cur = ""
                if len(lines) >= 3:
                    break
        if cur and len(lines) < 3:
            lines.append(cur)

        ty_ = y0 + 7
        for i, ln in enumerate(lines):
            d.text((x0 + 8, ty_), ln, font=f_nt if i == 0 else f_sm,
                   fill=NODE_TEXT if i == 0 else (120, 126, 134))
            ty_ += int((19 if i == 0 else 16) * dpi_scale)
        d.text((x0 + 8, y1 - int(17 * dpi_scale)), n["type"], font=f_sm,
               fill=(158, 164, 172))

    # ---- 右侧标尺：把每行的 y 画出参考线，便于确认「同行顶对齐」 ----
    for gy in sorted({g["bounding"][1] for g in groups}):
        yy = ty(gy)
        d.line([PAD, yy, out_w - PAD, yy], fill=GRID_LINE, width=1)

    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="工作流布局示意图")
    ap.add_argument("--out-dir", default=os.getcwd(), help="输出目录")
    ap.add_argument("--width", type=int, default=1900, help="输出宽度（像素）")
    ap.add_argument("--only", default="", help="只看某个文件名关键字")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    made = 0
    for label, fn in WORKFLOWS:
        if args.only and args.only not in fn:
            continue
        p = os.path.join(TEMPLATES, fn)
        if not os.path.isfile(p):
            print(f"！缺文件：{p}（先跑 tools/make_example_workflow.py）")
            continue
        img = draw_workflow(p, label, args.width)
        out = os.path.join(args.out_dir, f"layout_{os.path.splitext(fn)[0]}.png")
        img.save(out)
        print(f"已生成 {out}  ({img.size[0]}×{img.size[1]})")
        made += 1
    if not made:
        print("没有生成任何图。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
