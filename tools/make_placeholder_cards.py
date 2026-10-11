"""为示例工作流生成「四视图 / 角色设定卡」占位图。

背景
----
示例工作流引用了 ``input/char_*_card.png``。这些图**不属于插件仓库**
（用户必须替换成自己的角色设定卡），但缺了会让工作流卡在 LoadImage ——
所以这里生成一张**明确标注为占位**的卡片：能跑通、又不会被误当成真素材。

一角色一图约定
-------------
每个角色只锁 **一张** 参考图，用「四视图 / 角色设定卡」把正/侧/背/表情并到同一张。
多张独立定妆图会在服装、发型、角度之间产生歧义，而且白占官方 9 张参考图额度。

用法
----
    <ComfyUI>/<standalone-env>/python.exe tools/make_placeholder_cards.py [--force]

不带参数时只补**缺失**的文件；带 ``--force`` 覆盖重建。
真实素材请直接覆盖 ``input/char_tsuna_card.png`` 等同名文件。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)
INPUT_DIR = os.path.join(COMFY_ROOT, "input")

# 与 tools/make_example_workflow.py 的 CHAR_CARDS 保持一致
CARDS = [
    ("纲吉", "char_tsuna_card.png", (255, 140, 40), "橙"),
    ("Reborn", "char_reborn_card.png", (60, 70, 90), "黑"),
]

CELLS = ["正面 FRONT", "侧面 SIDE", "背面 BACK", "表情 EXPRESSION"]
W, H = 1024, 768
TITLE_H = 88


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


def make_card(display_name: str, path: str, accent) -> bool:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (246, 247, 249))
    d = ImageDraw.Draw(img)

    f_title = _font(40)
    f_cell = _font(24)
    f_hint = _font(30)
    f_small = _font(20)

    # 标题带
    d.rectangle([0, 0, W, TITLE_H], fill=accent)
    d.text((28, 22), f"角色设定卡（四视图）· {display_name}", font=f_title,
           fill=(255, 255, 255))
    d.text((W - 250, 34), "占位图 · 请替换", font=f_cell, fill=(255, 235, 225))

    # 2×2 四视图格
    pad, gap = 26, 18
    gw = (W - pad * 2 - gap) // 2
    gh = (H - TITLE_H - pad * 2 - gap) // 2
    for i, label in enumerate(CELLS):
        r, c = divmod(i, 2)
        x0 = pad + c * (gw + gap)
        y0 = TITLE_H + pad + r * (gh + gap)
        d.rectangle([x0, y0, x0 + gw, y0 + gh], fill=(255, 255, 255),
                    outline=(205, 209, 214), width=2)
        # 中心提示
        cx, cy = x0 + gw // 2, y0 + gh // 2
        d.text((cx, cy), "此处放该视角图", font=f_hint, fill=(196, 202, 210),
               anchor="mm")
        # 角标
        d.rectangle([x0, y0, x0 + 210, y0 + 42], fill=(240, 242, 245))
        d.text((x0 + 12, y0 + 10), label, font=f_cell, fill=(70, 76, 86))

    d.text((pad, H - 26), "替换方法：把这个文件换成你的真实设定卡（同名覆盖）",
           font=f_small, fill=(150, 156, 166))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return True


def main() -> int:
    force = "--force" in sys.argv
    if not os.path.isdir(INPUT_DIR):
        print(f"！找不到 ComfyUI 的 input 目录：{INPUT_DIR}")
        print("  请从 ComfyUI 根目录下运行，或确认插件放在 custom_nodes/ 里。")
        return 1

    print(f"input 目录：{INPUT_DIR}")
    made = skipped = 0
    for display_name, fname, accent, _c in CARDS:
        path = os.path.join(INPUT_DIR, fname)
        if os.path.isfile(path) and not force:
            print(f"  跳过（已存在）  {fname}")
            skipped += 1
            continue
        make_card(display_name, path, accent)
        print(f"  已生成         {fname}  ({display_name})")
        made += 1

    page = os.path.join(INPUT_DIR, "manga_page.png")
    print()
    print(f"漫画页 {os.path.basename(page)}：",
          "已存在" if os.path.isfile(page) else
          "**缺失** —— 示例工作流会卡在 LoadImage，请自己放一张漫画页并命名为 manga_page.png")

    print()
    print(f"完成：新建 {made} 张 / 跳过 {skipped} 张。"
          f"{'' if force else '（要覆盖请加 --force）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
