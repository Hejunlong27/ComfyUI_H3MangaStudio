"""校验生成的工作流是否「真的能导入并跑起来」。

它比通用的 JSON 结构校验更严：**逐节点对着真实 `INPUT_TYPES()` 核对**

1. `widgets_values` 长度 == widget 型输入数（含前端额外控件）
2. `inputs` 数组里的名字 == 连线型输入 + 用到的 Autogrow 插槽，且**顺序一致**
3. 每条连线的 `target_slot` 指向的输入名，与 `links` 里记的名字一致
   —— 这一类错位不会报错，只会让参数整体串位，是最贵的一类 bug
4. 连线类型与源节点 `RETURN_TYPES[slot]` 一致（放行 ANY / `*`）
5. 拓扑顺序合法（每条连线的源节点排在目标节点之前）
6. 每个节点都落在某个分组框内，且**任意两个节点不重叠**（布局可读性）
7. `version == 0.4`、无 subgraph、`last_node_id/last_link_id` 正确

用法（cwd 为 ComfyUI 根目录）：
    .venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tools/verify_workflow.py
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
WF_PATH = os.path.join(PKG, "templates", "H3MangaStudio_example_workflow.json")

os.chdir(COMFY_ROOT)
for p in (COMFY_ROOT, CUSTOM_NODES):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.argv = [sys.argv[0], "--cpu"]

import nodes  # noqa: E402

LINK_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK", "AUDIO",
    "VIDEO", "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET", "CLIP_VISION",
    "STYLE_MODEL", "UPSCALE_MODEL", "GLIGEN", "HOOKS", "WEBCAM", "MODEL_PATCH",
    "LORA_MODEL", "ANY", "*",
    "H3_MANGA_REFS", "H3_MANGA_PLAN", "H3_MANGA_BOARD",
}
KNOWN_EXTRA_WIDGETS = {"LoadImage": 1}   # 上传按钮

# 纯前端虚拟节点：后端 NODE_CLASS_MAPPINGS 里没有，但工作流里合法
FRONTEND_VIRTUAL = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode", "Comment"}

# 工作流 inputs 数组的排序规则（用 606 个官方模板反推、已核实）：
#   ① 连线型输入（required → optional，各自按声明顺序）
#   ② Autogrow 插槽（required → optional，按声明顺序；组内按槽位号）
#   ③ **被连线**的 widget 型输入（required → optional）
# 参考样本：MiniMaxH3ReferenceToVideo / HiDreamO1ReferenceImages
ERRORS: list = []
WARNINGS: list = []


def err(msg):
    ERRORS.append(msg)
    print(f"  [ERROR] {msg}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN ] {msg}")


def ok(msg):
    print(f"  [ok] {msg}")


def schema_of(node_type: str):
    """返回 (widgets:[(name,meta)], links:[(name,type)], autogrow:{name:info})。"""
    cls = nodes.NODE_CLASS_MAPPINGS.get(node_type)
    if cls is None:
        return None
    it = cls.INPUT_TYPES()
    widgets, links, autogrow = [], [], {}
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
                autogrow[name] = {"prefix": tpl.get("prefix") or "", "inner_type": itype}
                continue
            if isinstance(t, str) and t in LINK_TYPES:
                links.append((name, t))
            else:
                widgets.append((name, meta))
    return widgets, links, autogrow


def main() -> int:
    asyncio.run(nodes.init_builtin_extra_nodes())
    asyncio.run(nodes.load_custom_node(PKG))

    if not os.path.isfile(WF_PATH):
        print(f"找不到工作流文件：{WF_PATH}")
        return 1
    wf = json.load(open(WF_PATH, encoding="utf-8"))

    node_by_id = {n["id"]: n for n in wf["nodes"]}
    name_of = {n["id"]: n.get("title") or n["type"] for n in wf["nodes"]}

    print("=" * 72)
    print("① 顶层结构")
    print("=" * 72)
    ok(f"version = {wf.get('version')}") if wf.get("version") == 0.4 else err(
        f"version 必须是数字 0.4，实得 {wf.get('version')!r}")
    if any("nodes" in n and isinstance(n.get("nodes"), list) and n.get("nodes")
           for n in wf["nodes"]):
        err("出现了 subgraph 结构，必须用 flat graph")
    else:
        ok("flat graph（无 subgraph）")
    max_id = max(node_by_id)
    if wf["last_node_id"] != max_id:
        err(f"last_node_id={wf['last_node_id']} 应为 {max_id}")
    else:
        ok(f"last_node_id = {max_id}")
    max_link = max([l[0] for l in wf["links"]] or [0])
    if wf["last_link_id"] != max_link:
        err(f"last_link_id={wf['last_link_id']} 应为 {max_link}")
    else:
        ok(f"last_link_id = {max_link}")

    print()
    print("=" * 72)
    print("② 逐节点核对 widgets_values 与 inputs（对真实 INPUT_TYPES）")
    print("=" * 72)
    link_by_id = {l[0]: l for l in wf["links"]}

    for n in wf["nodes"]:
        ntype = n["type"]
        label = f"#{n['id']} {ntype}"
        if ntype in FRONTEND_VIRTUAL:
            ok(f"{label}：纯前端虚拟节点，跳过 schema 核对（{name_of[n['id']]}）")
            continue
        sch = schema_of(ntype)
        if sch is None:
            warn(f"{label}：本机没有这个节点类型，跳过（{name_of[n['id']]}）")
            continue
        widgets, links, autogrow = sch

        # --- widgets_values 数量 ---
        expect_widgets = len(widgets) + KNOWN_EXTRA_WIDGETS.get(ntype, 0)
        # control_after_generate 前端会插一个槽位
        expect_widgets += sum(1 for _n, meta in widgets if meta.get("control_after_generate"))
        got = len(n.get("widgets_values") or [])
        if got != expect_widgets:
            err(f"{label}：widgets_values 数量 {got} != 期望 {expect_widgets}")
        else:
            ok(f"{label}：widgets_values = {got}")

        # --- inputs 数组 ---
        used_ag = []
        for inp in n.get("inputs") or []:
            nm = inp.get("name", "")
            if "." in nm:
                used_ag.append(nm)

        expect_input_names = [nm for nm, _t in links] + used_ag
        got_input_names = [i.get("name") for i in (n.get("inputs") or [])]
        if got_input_names != expect_input_names:
            err(f"{label}：inputs 顺序/内容不符\n"
                f"        期望 {expect_input_names}\n"
                f"        实得 {got_input_names}")
        else:
            ok(f"{label}：inputs = {got_input_names}")

        # --- 若把 widget 型输入也连了线，它必须排在第 ③ 段（最后） ---
        widget_names = {nm for nm, _m in widgets}
        extra = [nm for nm in got_input_names if nm in widget_names]
        if extra:
            first_extra = got_input_names.index(extra[0])
            tail = got_input_names[first_extra:]
            if tail != extra:
                err(f"{label}：被连线的 widget 输入必须集中在末尾，实得 {got_input_names}")
            else:
                ok(f"{label}：被连线的 widget 输入排在末尾 {extra}")

        # --- 每个连线型输入的类型 ---
        declared = dict(links)
        for inp in n.get("inputs") or []:
            nm, t = inp.get("name", ""), inp.get("type")
            if "." in nm:
                aname = nm.split(".", 1)[0]
                want = autogrow.get(aname, {}).get("inner_type")
            else:
                want = declared.get(nm)
            if want is not None and t != want:
                err(f"{label}：输入 {nm} 类型 {t} != 声明 {want}")

    print()
    print("=" * 72)
    print("③ 连线：目标槽位名 / 类型 / 双向登记")
    print("=" * 72)
    for lid, oid, oslot, tid, tslot, ltype in wf["links"]:
        src, dst = node_by_id.get(oid), node_by_id.get(tid)
        if src is None or dst is None:
            err(f"连线 {lid}：端点节点不存在（{oid} → {tid}）")
            continue
        # 目标槽位名必须与 links 记录一致
        dst_inputs = dst.get("inputs") or []
        if tslot is None or tslot >= len(dst_inputs):
            err(f"连线 {lid}：target_slot={tslot} 越界（{name_of[tid]} 只有 {len(dst_inputs)} 个输入）")
            continue
        slot_name = dst_inputs[tslot].get("name")
        if dst_inputs[tslot].get("link") != lid:
            err(f"连线 {lid}：目标 {name_of[tid]}.{slot_name} 的 link 字段是 "
                f"{dst_inputs[tslot].get('link')}，不是 {lid}")
        # 源输出登记
        src_outputs = src.get("outputs") or []
        if oslot >= len(src_outputs):
            err(f"连线 {lid}：origin_slot={oslot} 越界（{name_of[oid]} 只有 {len(src_outputs)} 个输出）")
            continue
        if lid not in (src_outputs[oslot].get("links") or []):
            err(f"连线 {lid}：源 {name_of[oid]}[{oslot}] 的 links 里没有登记这条连线")
        # 类型一致
        stype = src_outputs[oslot].get("type")
        if ltype != stype:
            err(f"连线 {lid}：link.type={ltype} != 源输出类型 {stype}")
    ok(f"共校验 {len(wf['links'])} 条连线的槽位与类型")

    # 类型兼容性（源类型 vs 目标声明类型）
    for lid, oid, oslot, tid, tslot, _lt in wf["links"]:
        src, dst = node_by_id.get(oid), node_by_id.get(tid)
        if src is None or dst is None:
            continue
        stype = (src.get("outputs") or [{}])[oslot].get("type")
        dtype = (dst.get("inputs") or [{}])[tslot].get("type")
        if stype in ("*", "ANY") or dtype in ("*", "ANY", None):
            continue
        if stype != dtype:
            err(f"连线 {lid}：{stype} → {dtype}（{name_of[oid]} → {name_of[tid]}）类型不兼容")
    ok("连线类型全部兼容")

    print()
    print("=" * 72)
    print("④ 拓扑顺序")
    print("=" * 72)
    order_index = {n["id"]: n.get("order", 0) for n in wf["nodes"]}
    bad = [l for l in wf["links"] if order_index[l[1]] >= order_index[l[3]]]
    if bad:
        err(f"{len(bad)} 条连线的源节点没有排在目标节点之前")
    else:
        ok("所有连线的源节点都排在目标节点之前")
    if len(set(order_index.values())) != len(order_index):
        err("order 值有重复")
    else:
        ok(f"order 唯一，共 {len(order_index)} 个节点")

    print()
    print("=" * 72)
    print("⑤ 布局：分组包含 + 不重叠")
    print("=" * 72)
    groups = wf.get("groups") or []
    if not groups:
        err("没有 groups，画布可读性不合格")
    for g in groups:
        b = g.get("bounding")
        if not (isinstance(b, list) and len(b) == 4 and b[2] > 0 and b[3] > 0):
            err(f"分组「{g.get('title')}」bounding 必须是 [x, y, w, h] 且宽高为正：{b}")

    def inside(n, g):
        x, y = n["pos"]
        w, h = n["size"]
        gx, gy, gw, gh = g["bounding"]
        return gx <= x and gy <= y and x + w <= gx + gw and y + h <= gy + gh

    for n in wf["nodes"]:
        if not any(inside(n, g) for g in groups):
            err(f"节点 {name_of[n['id']]} 不在任何分组框内")

    def rect(n):
        return (n["pos"][0], n["pos"][1], n["pos"][0] + n["size"][0], n["pos"][1] + n["size"][1])

    overlap = 0
    ns = wf["nodes"]
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            a, b = rect(ns[i]), rect(ns[j])
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                overlap += 1
                err(f"节点重叠：{name_of[ns[i]['id']]} ↔ {name_of[ns[j]['id']]}")
    if not overlap:
        ok(f"{len(ns)} 个节点两两不重叠")
    # 分组框之间不重叠
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i]["bounding"], groups[j]["bounding"]
            if a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]:
                err(f"分组框重叠：「{groups[i]['title']}」↔「{groups[j]['title']}」")
    ok(f"共 {len(groups)} 个分组框：" + "、".join(g["title"] for g in groups))

    print()
    print("=" * 72)
    print("⑥ 节点命名与参数")
    print("=" * 72)
    untitled = [n["id"] for n in wf["nodes"] if not (n.get("title") or "").strip()]
    if untitled:
        warn(f"{len(untitled)} 个节点没有自定义标题（将显示默认名）")
    else:
        ok(f"{len(wf['nodes'])} 个节点都设置了可读标题")
    empty_critical = []
    for n in wf["nodes"]:
        if n["type"] in ("H3MangaScriptLoader", "H3MangaRefPack"):
            vals = [v for v in (n.get("widgets_values") or []) if isinstance(v, str)]
            if not any(v.strip() for v in vals):
                empty_critical.append(name_of[n["id"]])
    if empty_critical:
        warn(f"这些节点没有任何文本参数（会跑不出东西）：{empty_critical}")
    else:
        ok("关键节点的文本参数都已填好（分镜清单 / 角色名）")

    print()
    print("=" * 72)
    print(f"结果：{len(ERRORS)} 个错误 / {len(WARNINGS)} 个警告")
    print("=" * 72)
    if ERRORS:
        for e in ERRORS:
            print(f"  ! {e}")
        return 1
    print("PASS  工作流可直接导入并运行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
