"""生成一份可直接拖进 ComfyUI 的示例工作流 JSON。

为什么用脚本生成而不是手写
--------------------------
`widgets_values` 是**按位置存的**：连线型参数不占槽位、Autogrow 会展开成
`输入名.插槽名`、个别节点前端还会额外插控件（LoadImage 的上传按钮、
INT 的 control_after_generate）。手写必错，而且错了不报错、只是参数整体串位。

所以本脚本直接问真实 ComfyUI 加载器要 `INPUT_TYPES()`，自己算
「哪些是 widget、哪些是连线、默认值是多少、节点该多高」，再拼 flat graph。

用法（cwd 为 ComfyUI 根目录）：
    .venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tools/make_example_workflow.py

生成后用 `tools/verify_workflow.py` 校验（会逐节点核对 widgets 数量与连线槽位）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)

os.chdir(COMFY_ROOT)
for p in (COMFY_ROOT, CUSTOM_NODES):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.argv = [sys.argv[0], "--cpu"]

import nodes  # noqa: E402

# 连线型类型 ID（不占 widgets 槽位）
LINK_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK", "AUDIO",
    "VIDEO", "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET", "CLIP_VISION",
    "STYLE_MODEL", "UPSCALE_MODEL", "GLIGEN", "HOOKS", "WEBCAM", "MODEL_PATCH",
    "LORA_MODEL", "ANY", "*",
    "H3_MANGA_REFS", "H3_MANGA_PLAN", "H3_MANGA_BOARD",
}

# 前端会额外插入的控件（定义里看不到）
KNOWN_EXTRA_WIDGETS = {
    "LoadImage": [("image", "image")],           # 上传按钮对应的值
}

# 需要特殊处理的下拉默认值
WIDGET_OVERRIDE = {
    "CLIPLoader": {"type": "minimax"},
    "SaveVideo": {"format": "auto", "codec": "auto"},
    "CreateVideo": {"bit_depth": "8", "color_space": "sRGB", "codec": "none"},
}

# 示例用到的模型文件（**逐字来自本机 models/ 目录**）
UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

PAGE_IMAGE = "manga_page.png"
CHAR_IMAGES = ["char_tsuna_front.png", "char_tsuna_side.png", "char_reborn.png"]

INLINE_SCRIPT = """# episode: Reborn 第30卷 目标283
# style: manga_anime
# default_size: 1344x768
=== s01 ===
panel: 0
duration: 4
camera: tilt_up
action: 黄昏的住宅区街景，电线杆与交错的电线横过天空
=== s02 ===
panel: 1
duration: 5
characters: 纲吉
camera: slow_push_in
action: @纲吉 从门后走出，抬眼看向镜头
speaker: 纲吉
dialogue: 这一刻终于还是到来了。
"""

README_NOTE = """# H3 Manga Studio · 漫改动画量产示例

## 三步跑通
1. 把 4 个 **LoadImage** 的文件名换成你自己的图（1 张漫画页 + 3 张角色定妆图）
2. 改 **② 参考包** 的 `character_names`（与图槽一一对应；同名重复 = 同一角色多张图）
3. 改 **③ 分镜清单** 的 `inline_text`

## 首次运行建议
* 先把 **⑦ 整集渲染** 的 `max_shots` 设成 `1`，确认画面 OK 再改成 `0` 跑整集
* 想看机器实际拿到的提示词 → 点 **⑨ 计划预览** 或 **⑤ 提示词装配** 节点上的面板
* 跑之前可以单独跑 **⑧ 一致性审计** 看有没有角色缺图

## 接线顺序（重要）
分镜清单 → **分镜绑定** → 提示词装配 → 整集渲染
必须让「分镜绑定」接在「提示词装配」之前，否则提示词里没有分镜格锚点，槽位会对不上。

## 参数速查
| 想要 | 调什么 |
|---|---|
| 更快 | 挂 turbo LoRA + steps 降到 8；或降 max_ref_images |
| 更像原角色 | ref_image_size 改 max（慢数倍）；多给参考图 |
| 画面更贴分镜 | 确认 board 接到 ⑦；运镜改 static |
| 单镜失败不中断 | on_error 保持 skip（默认），失败原因会写进报告 |
"""


# 前端虚拟节点：后端 NODE_CLASS_MAPPINGS 里没有，需要手写规格
# (widgets, links, autogrow) —— outputs 由「没有后端类」自然推导为空
VIRTUAL_NODES = {
    "MarkdownNote": ([("text", "", True)], [], {}),
    "Note": ([("text", "", True)], [], {}),
}


def describe(node_id: str):
    """返回 (widgets, links, autogrow)。

    widgets: [(name, default, multiline)]
    links:   [(name, type)]
    autogrow:{name: {"prefix","min","max","inner_type"}}
    """
    if node_id in VIRTUAL_NODES:
        return VIRTUAL_NODES[node_id]

    cls = nodes.NODE_CLASS_MAPPINGS[node_id]
    it = cls.INPUT_TYPES()
    widgets = []
    links = []
    autogrow = {}

    for sect in ("required", "optional"):
        for name, v in (it.get(sect) or {}).items():
            t = v[0] if isinstance(v, (list, tuple)) and v else v
            meta = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else {}

            if isinstance(t, str) and t == "COMFY_AUTOGROW_V3":
                tpl = meta.get("template") or {}
                inner = (tpl.get("input") or {}).get("required") or {}
                itype = "IMAGE"
                if inner:
                    ispec = next(iter(inner.values()))
                    itype = ispec[0] if isinstance(ispec, (list, tuple)) else ispec
                autogrow[name] = {
                    "prefix": tpl.get("prefix") or "",
                    "min": tpl.get("min") or 0,
                    "max": tpl.get("max") or 0,
                    "inner_type": itype,
                }
                continue

            if isinstance(t, str) and t in LINK_TYPES:
                links.append((name, t))
                continue

            default = meta.get("default")
            if default is None and isinstance(t, list) and t:
                default = t[0]
            widgets.append((name, default, bool(meta.get("multiline"))))
            if meta.get("control_after_generate"):
                widgets.append(("__control_after_generate__", "randomize", False))

    for extra_name, extra_val in KNOWN_EXTRA_WIDGETS.get(node_id, []):
        widgets.append((extra_name, extra_val, False))

    return widgets, links, autogrow


def estimate_height(widgets, links, used_slots) -> int:
    """粗略估算节点高度（ComfyUI 的节点尺寸是存下来的，给个合理值即可）。"""
    h = 64
    for name, _default, multiline in widgets:
        if name == "__control_after_generate__":
            continue
        h += 92 if multiline else 26
    h += 26 * (len(links) + len(used_slots))
    return max(120, min(900, h))


# --------------------------------------------------------------------------
# 图定义：按列（层）组织，列内自上而下
# --------------------------------------------------------------------------
NODES = [
    ("unet", "UNETLoader", "UNET · Ref2VA（r2v 专用）", {"unet_name": UNET}, "模型"),
    ("clip", "CLIPLoader", "CLIP · type 必须是 minimax", {"clip_name": CLIP}, "模型"),
    ("vae_v", "VAELoader", "VAE · 视频", {"vae_name": VAE_VIDEO}, "模型"),
    ("vae_a", "VAELoader", "VAE · 音频", {"vae_name": VAE_AUDIO}, "模型"),

    ("img_pg", "LoadImage", "输入 · 漫画页", {"image": PAGE_IMAGE}, "输入图"),
    ("img_c1", "LoadImage", "输入 · 纲吉 正面", {"image": CHAR_IMAGES[0]}, "输入图"),
    ("img_c2", "LoadImage", "输入 · 纲吉 侧面", {"image": CHAR_IMAGES[1]}, "输入图"),
    ("img_c3", "LoadImage", "输入 · Reborn", {"image": CHAR_IMAGES[2]}, "输入图"),

    ("split", "H3MangaPanelSplitter", "① 漫画页切分镜格", {}, "素材"),
    ("refs", "H3MangaRefPack", "② 参考包", {
        "character_names": "纲吉, 纲吉, Reborn",
        "character_describes": "棕发橙眼，黑色并盛中学学生制服，左手中指戴彭格列戒指\n"
                               "黑西装婴儿，黑色卷鬓角，礼帽上有橙色变色龙",
    }, "素材"),

    ("script", "H3MangaScriptLoader", "③ 分镜清单加载", {
        "source": "inline", "inline_text": INLINE_SCRIPT}, "编排"),
    ("bind", "H3MangaShotBinder", "④ 分镜绑定", {}, "编排"),
    ("comp", "H3MangaPromptComposer", "⑤ 提示词装配", {
        "negative": "flickering, identity drift, extra fingers, deformed hands"},
        "编排"),
    ("audit", "H3MangaConsistencyAudit", "⑧ 一致性审计", {}, "编排"),

    ("render", "H3MangaEpisodeRender", "⑦ 整集渲染 ★", {
        "steps": 8, "lora_name": TURBO_LORA, "max_shots": 1}, "执行"),

    ("vid", "CreateVideo", "合成视频（24fps）", {"fps": 24}, "输出"),
    ("save", "SaveVideo", "保存整集视频", {"filename_prefix": "H3MangaStudio/episode"},
     "输出"),
    ("prev", "PreviewAny", "预览渲染报告", {}, "输出"),

    ("note", "MarkdownNote", "使用说明", {"text": README_NOTE}, "说明"),
]

# (源变量, 源输出序号, 目标变量, 目标输入名)
LINKS = [
    ("unet", 0, "render", "model"),
    ("clip", 0, "render", "clip"),
    ("vae_v", 0, "render", "vae"),
    ("vae_a", 0, "render", "audio_vae"),

    ("img_pg", 0, "split", "image"),

    ("img_c1", 0, "refs", "character_images.character_0"),
    ("img_c2", 0, "refs", "character_images.character_1"),
    ("img_c3", 0, "refs", "character_images.character_2"),

    ("script", 0, "bind", "plan"),
    ("split", 0, "bind", "board"),
    ("refs", 0, "bind", "refs"),

    ("bind", 0, "comp", "plan"),
    ("refs", 0, "comp", "refs"),

    ("comp", 0, "audit", "plan"),
    ("refs", 0, "audit", "refs"),

    ("comp", 0, "render", "plan"),
    ("refs", 0, "render", "refs"),
    ("split", 0, "render", "board"),

    ("render", 0, "vid", "images"),
    ("render", 1, "vid", "audio"),
    ("vid", 0, "save", "video"),
    ("render", 2, "prev", "source"),
]

COLUMNS = ["模型", "输入图", "素材", "编排", "执行", "输出", "说明"]

COL_COLOR = {
    "模型": "#3f789e",
    "输入图": "#88A",
    "素材": "#a1309b",
    "编排": "#b58b2a",
    "执行": "#444",
    "输出": "#b06634",
    "说明": "#666",
}

COL_X0 = -2020
COL_STEP = 500
COL_GAP = 70
COL_WIDTH = 430


def _cnr_id(node_type: str) -> str:
    """标注来源，方便前端显示归属（自定义节点 vs 官方）。"""
    if node_type.startswith("H3Manga"):
        return "ComfyUI_H3MangaStudio"
    return "comfy-core"


def build(sparse_override: dict | None = None):
    # sparse_override：按「输入名 → 值」覆盖 ⑦ 整集渲染 的 widget
    # （tools/sparse_ab.py 用它生成「dense vs 稀疏」两份只差一个开关的工作流）
    over_of = {var: dict(over or {}) for var, _t, _ti, over, _c in NODES}
    if sparse_override:
        over_of.setdefault("render", {}).update(sparse_override)

    # comfy_extras 的 600 个节点（含 CreateVideo / SaveVideo / PreviewAny）
    # 必须先异步加载，否则 NODE_CLASS_MAPPINGS 里只有 65 个基础节点
    asyncio.run(nodes.init_builtin_extra_nodes())
    asyncio.run(nodes.load_custom_node(PKG))

    specs = {var: describe(ntype) for var, ntype, _t, _o, _c in NODES}

    used_slots = {}
    for _s, _si, dvar, dname in LINKS:
        if "." in dname:
            used_slots.setdefault(dvar, []).append(dname)

    # ---- 尺寸与坐标：按列自上而下堆叠，保证不重叠、列对齐 ----
    size_of, pos_of, col_y = {}, {}, {}
    for var, ntype, _title, _over, col in NODES:
        widgets, links, autogrow = specs[var]
        h = estimate_height(widgets, links, used_slots.get(var, []))
        size_of[var] = [COL_WIDTH, h]
        y = col_y.get(col, 0)
        pos_of[var] = y
        col_y[col] = y + h + COL_GAP

    col_index = {c: i for i, c in enumerate(COLUMNS)}
    pos_of = {v: (COL_X0 + col_index[c] * COL_STEP, y) for v, y in pos_of.items()
              for c in [next(c for vv, _t, _ti, _o, c in NODES if vv == v)]}

    id_of = {var: i + 1 for i, (var, *_r) in enumerate(NODES)}
    var_of_id = {i: v for v, i in id_of.items()}

    # ---- 连线 ----
    link_records = []
    incoming = {}
    next_link = 1
    for svar, sidx, dvar, dname in LINKS:
        link_records.append({
            "id": next_link, "origin_id": id_of[svar], "origin_slot": sidx,
            "target_id": id_of[dvar], "target_name": dname, "target_slot": None,
            "type": "*",
        })
        incoming[(dvar, dname)] = next_link
        next_link += 1

    # ---- 节点 ----
    graph_nodes = []
    for var, ntype, title, over, col in NODES:
        widgets, links, autogrow = specs[var]

        inputs = []
        for name, t in links:
            inputs.append({"name": name, "type": t, "link": incoming.get((var, name))})
        for aname, info in autogrow.items():
            for full in used_slots.get(var, []):
                if full.startswith(aname + "."):
                    inputs.append({
                        "name": full, "type": info["inner_type"],
                        "link": incoming.get((var, full)), "shape": 7,
                    })

        wv = []
        for wname, wdefault, _ml in widgets:
            if wname in over_of.get(var, {}):
                wv.append(over_of[var][wname])
            elif wname == "__control_after_generate__":
                wv.append("randomize")
            else:
                wv.append(wdefault)

        for i, inp in enumerate(inputs):
            if inp["link"] is not None:
                for rec in link_records:
                    if rec["id"] == inp["link"]:
                        rec["target_slot"] = i

        cls = nodes.NODE_CLASS_MAPPINGS.get(ntype)
        rtypes = list(getattr(cls, "RETURN_TYPES", []) or []) if cls else []
        rnames = list(getattr(cls, "RETURN_NAMES", []) or []) if cls else []
        outputs = []
        for i, t in enumerate(rtypes):
            outputs.append({
                "name": rnames[i] if i < len(rnames) else t,
                "type": t, "links": [], "slot_index": i,
            })

        graph_nodes.append({
            "id": id_of[var], "type": ntype, "title": title,
            "pos": list(pos_of[var]), "size": list(size_of[var]),
            "flags": {}, "order": 0, "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": ntype, "cnr_id": _cnr_id(ntype)},
            "widgets_values": wv,
            "_var": var, "_col": col,
        })

    # ---- 输出 links 回填 + 连线类型 ----
    node_of_id = {n["id"]: n for n in graph_nodes}
    type_of_out = {}
    for n in graph_nodes:
        for o in n["outputs"]:
            type_of_out[(n["_var"], o["slot_index"])] = o["type"]

    for rec in link_records:
        rec["type"] = type_of_out.get(
            (var_of_id[rec["origin_id"]], rec["origin_slot"]), "*")
        node_of_id[rec["origin_id"]]["outputs"][rec["origin_slot"]]["links"].append(rec["id"])

    # ---- 拓扑排序 ----
    indeg = {n["id"]: 0 for n in graph_nodes}
    adj = {n["id"]: [] for n in graph_nodes}
    for rec in link_records:
        adj[rec["origin_id"]].append(rec["target_id"])
        indeg[rec["target_id"]] += 1
    queue = sorted(i for i, d in indeg.items() if d == 0)
    order = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in sorted(adj[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    for i, nid in enumerate(order):
        node_of_id[nid]["order"] = i

    # ---- 分组：每列一个框，纵向包住该列全部节点 ----
    groups = []
    for gi, col in enumerate(COLUMNS, start=1):
        members = [n for n in graph_nodes if n["_col"] == col]
        if not members:
            continue
        x0 = min(n["pos"][0] for n in members) - 26
        y0 = min(n["pos"][1] for n in members) - 54
        x1 = max(n["pos"][0] + n["size"][0] for n in members) + 26
        y1 = max(n["pos"][1] + n["size"][1] for n in members) + 26
        groups.append({
            "id": gi, "title": col,
            "bounding": [x0, y0, x1 - x0, y1 - y0],
            "color": COL_COLOR.get(col, "#88A"), "font_size": 26, "flags": {},
        })

    for n in graph_nodes:
        n.pop("_var", None)
        n.pop("_col", None)

    return {
        "id": "h3-manga-studio-example",
        "revision": 0,
        "last_node_id": max(id_of.values()),
        "last_link_id": next_link - 1,
        "nodes": graph_nodes,
        "links": [[r["id"], r["origin_id"], r["origin_slot"], r["target_id"],
                   r["target_slot"], r["type"]] for r in link_records],
        "groups": groups,
        "config": {},
        "extra": {
            "ds": {"scale": 0.42, "offset": [2400, 120]},
            "note": "由 ComfyUI_H3MangaStudio/tools/make_example_workflow.py 生成；"
                    "用 tools/verify_workflow.py 校验",
        },
        "version": 0.4,
    }


if __name__ == "__main__":
    wf = build()
    out = os.path.join(PKG, "templates", "H3MangaStudio_example_workflow.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=1)
    print(f"已生成 {out}")
    print(f"  节点 {len(wf['nodes'])} / 连线 {len(wf['links'])} / 分组 {len(wf['groups'])}")
    for n in wf["nodes"]:
        print(f"  #{n['id']:>3} {n['type']:<26} pos={str(n['pos']):<16} "
              f"size={str(n['size']):<12} widgets={len(n['widgets_values'])} "
              f"inputs={len(n['inputs'])}  「{n['title']}」")
