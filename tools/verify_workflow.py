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
    .venv/Scripts/python.exe .../verify_workflow.py <工作流.json> [更多.json ...]

不给参数时校验 ``templates/`` 下的**全部** json。
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

# ★ 必须在改写 sys.argv **之前**把自己的参数摘出来。
#   下面那句 `sys.argv = [sys.argv[0], "--cpu"]` 是喂给 ComfyUI 节点加载器的，
#   它会把真正的命令行参数**抹掉** —— 曾经因为这一点，本工具一直只校验
#   WF_PATH 硬编码的那一个文件，「官方节点接线版」从未被检查过，
#   它的接线错误（widget 型输入连不上）因此全部逃过校验。
_CLI_WORKFLOWS = [a for a in sys.argv[1:] if not a.startswith("-")]

os.chdir(COMFY_ROOT)
for p in (COMFY_ROOT, CUSTOM_NODES):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.argv = [sys.argv[0], "--cpu"]

import nodes  # noqa: E402

# 连线 / widget 的判定见 `_is_widget_type()`（与生成器同一规则）。
# 这里曾经是一份「连线类型白名单」，已废弃：白名单每新增一个自定义类型都要
# 同步维护，两边不同步还会造成「双双自洽的真错」。
LINK_TYPES = set()          # 保留名字仅为兼容旧引用
# 前端会额外插入的控件：这些**不在 INPUT_TYPES 里**，但会占据 widgets_values 的槽位。
# ★ 必须带**名字**：否则「widgets_values_named 键集合」会少一项，误报键不匹配。
EXTRA_WIDGETS = {
    "LoadImage": [("upload", "image")],      # 上传按钮，值恒为 "image"
}

WIDGET_PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _is_widget_type(t) -> bool:
    """这个输入在 ComfyUI 里是 widget 还是连线？（与生成器同一规则）

    下拉框有两种形态：旧式 ``list`` 与 V3 式 ``"COMBO"`` —— 都要算 widget。
    """
    if isinstance(t, list):
        return True
    return isinstance(t, str) and t in WIDGET_PRIMITIVES


# --------------------------------------------------------------------------
# ★ 领域规则：这类错误「选项合法、后端不报错、画面却全错」，只能靠语义兜住。
# 每项：node 节点类型 / when 生效条件 / field 字段 / expect 期望值 / why 说明
# --------------------------------------------------------------------------
SEMANTIC_RULES = [
    {
        "node": "CLIPLoader",
        "when": lambda wm: any(k in str(wm.get("clip_name", "")).lower()
                               for k in ("minimax_h3", "qwen3vl")),
        "field": "type", "expect": "minimax",
        "why": "H3 的文本编码器是 Qwen3VL，type 必须是 minimax；"
               "写成 stable_diffusion 虽然选项合法、后端也不报错，"
               "但 CLIP 会按 SD 的 CLIP 结构去解析 Qwen3VL 权重 → "
               "prompt 根本没法被正确编码 → 出片与文本/参考图毫无关联",
    },
]

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
    """返回 (widgets:[(name, meta, wtype, options)], links:[(name,type)], autogrow:{name:info})。

    widgets 的每一项带 **类型** 与 **下拉选项**：只比数量是不够的 ——
    «CLIPLoader.type» 写成 ``stable_diffusion`` 数量对、选项也合法，
    但语义完全错（H3 的文本编码器是 Qwen3VL），必须靠选项表 + 领域规则兜住。
    """
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
            # ★ 与生成器用**同一条本质规则**：只有原始类型 / 选项列表是 widget，
            #   其余（含 H3_MANGA_* 自定义类型）都是连线。用白名单会两边不同步 ——
            #   生成器把连线写成 widget、校验器也认为是 widget，就变成「双双自洽的真错」。
            if not _is_widget_type(t):
                links.append((name, t))
                continue
            # widget 型：类型名 + 下拉选项
            if isinstance(t, list):
                wtype, options = "COMBO", list(t)
            else:
                wtype, options = str(t), None
            if options is None:
                mo = meta.get("options")
                if isinstance(mo, list):
                    options = list(mo)
            widgets.append((name, meta, wtype, options))
    return widgets, links, autogrow


def verify_one(wf_path: str) -> int:
    """校验单个工作流，返回错误数。"""
    global ERRORS, WARNINGS
    ERRORS, WARNINGS = [], []

    if not os.path.isfile(wf_path):
        print(f"找不到工作流文件：{wf_path}")
        return 1
    wf = json.load(open(wf_path, encoding="utf-8"))
    print("#" * 72)
    print(f"# {os.path.basename(wf_path)}")
    print("#" * 72)

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

        # ★★ 重建**完整的 widget 序列** —— 它就是 widgets_values 的语义顺序：
        #     ① INPUT_TYPES 里声明的 widget
        #     ② seed 后面紧跟的 control_after_generate（前端插的）
        #     ③ 前端额外控件（LoadImage 的 upload）
        #   为什么必须重建：只按 ① 去「按位置取值」，从 seed 之后每一项都会错位
        #   ——早先正是这样把 H3MangaEpisodeRender 的 max_shots 读成了 'randomize'，
        #   而旧代码只比「数量」，数量对得上，于是这批错位全部逃过校验。
        wseq = []          # [(name, wtype, options, meta)]
        for nm, meta, wtype, opts in widgets:
            wseq.append((nm, wtype, opts, meta))
            if meta.get("control_after_generate"):
                wseq.append(("control_after_generate", "COMBO", None, {}))
        for xname, _xval in EXTRA_WIDGETS.get(ntype, []):
            wseq.append((xname, "STRING", None, {}))

        # --- widgets_values 数量 ---
        expect_widgets = len(wseq)
        got = len(n.get("widgets_values") or [])
        if got != expect_widgets:
            names_seq = [w[0] for w in wseq]
            err(f"{label}：widgets_values 数量 {got} != 期望 {expect_widgets}\n"
                f"        期望顺序 {names_seq}")
        else:
            ok(f"{label}：widgets_values = {got}")

        # --- inputs 数组：三段式 ---
        #   ① 连线型输入（required → optional 声明顺序）
        #   ② Autogrow 插槽
        #   ③ **被连线**的 widget 型输入（前端「widget → input」的序列化形式）
        used_ag = []
        for inp in n.get("inputs") or []:
            nm = inp.get("name", "")
            if "." in nm:
                used_ag.append(nm)

        base_expect = [nm for nm, _t in links] + used_ag
        got_input_names = [i.get("name") for i in (n.get("inputs") or [])]
        widget_names = {w[0] for w in wseq}

        n_base = len(base_expect)
        if got_input_names[:n_base] != base_expect:
            err(f"{label}：inputs 前段不符\n"
                f"        期望 {base_expect}\n"
                f"        实得 {got_input_names[:n_base]}")
        else:
            ok(f"{label}：inputs = {base_expect}")

        tail = got_input_names[n_base:]
        bad_tail = [nm for nm in tail if nm not in widget_names]
        if bad_tail:
            err(f"{label}：inputs 末尾出现既非连线型输入、也非 widget 的名字：{bad_tail}")
        dupes = sorted({nm for nm in got_input_names
                        if got_input_names.count(nm) > 1})
        if dupes:
            err(f"{label}：inputs 有重复条目 {dupes}")
        if tail and not bad_tail:
            ok(f"{label}：被连线的 widget 型输入（末尾）= {tail}")

        # --- ★ 每个「被连线的 widget」必须带 widget 标记 ---
        # 少了它 → ComfyUI 不认这条连线 → links 里 target_slot 变成 null → 线被丢弃。
        # 表现：「界面上这条线连不上」，而参数静默退回 widget 默认值
        #（实测事故：官方多参节点的 prompt 没接上 → 空 prompt 出片，画面与漫画无关）。
        for inp in n.get("inputs") or []:
            nm = inp.get("name", "")
            if nm in widget_names and (inp.get("widget") or {}).get("name") != nm:
                err(f"{label}：widget 输入 {nm} 缺少 widget 标记 —— "
                    f"ComfyUI 会丢弃这条连线（target_slot 会变成 null）")

        # --- ★ 独立守卫：带 widget 标记的条目，其 type 必须真的是 widget 型 ---
        # 这条**不依赖**上面的分类函数（分类错了它照样能抓），是最后一道网。
        # 反例（v0.79 实测）：⑭ 的 `vlm` 是 H3_MANGA_VLM 连线型，却被序列化成
        #   {"name":"vlm","type":"H3_MANGA_VLM","widget":{"name":"vlm"}}
        # 前端会把它当「被连线的 widget」渲染成一个空控件，契约就错了。
        for inp in n.get("inputs") or []:
            if not inp.get("widget"):
                continue
            it_type = inp.get("type")
            if isinstance(it_type, str) and it_type not in WIDGET_PRIMITIVES:
                err(f"{label}：输入 {inp.get('name')} 带 widget 标记，但类型是 "
                    f"{it_type!r} —— 这是**连线型**输入，不该被写成 widget-input。"
                    f"（生成器的输入分类规则有问题）")

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

        # --- ★★ widget 值的【内容】校验 ---
        #     数量对了不代表值对。早先这里只查数量与顺序，于是
        #     ``CLIPLoader.type = stable_diffusion``（数量对、选项合法、语义全错）
        #     一路溜到出片现场 —— 那正是「生成的视频与漫画毫无关联」的一个真因。
        wv = n.get("widgets_values") or []
        wmap = {}
        for i, w in enumerate(wseq):
            if i < len(wv):
                wmap[w[0]] = wv[i]

        # ① 值类型：INT/FLOAT 不能是字符串、BOOLEAN 必须是布尔
        for i, (nm, wtype, _opts, _meta) in enumerate(wseq):
            if i >= len(wv) or wv[i] is None:
                continue
            v = wv[i]
            if wtype in ("INT", "FLOAT") and isinstance(v, str):
                err(f"{label}：{nm} 是 {wtype}，值却是字符串 {v!r}"
                    f"（widgets_values 顺序与真实 widget 不一致？）")
            elif wtype == "STRING" and not isinstance(v, str):
                warn(f"{label}：{nm} 是 STRING，值却是 {type(v).__name__} {v!r}")
            elif wtype == "BOOLEAN" and not isinstance(v, bool):
                err(f"{label}：{nm} 是 BOOLEAN，值却是 {type(v).__name__} {v!r}")

        # ② 下拉值必须在选项表内（防「值非法 → 静默取默认」）
        for i, (nm, wtype, opts, _meta) in enumerate(wseq):
            if i >= len(wv) or wtype != "COMBO" or not opts or wv[i] is None:
                continue
            try:
                bad = not any(wv[i] == o for o in opts)
            except Exception:                       # 选项里有非标量，放弃比较
                bad = False
            if bad:
                err(f"{label}：{nm} = {wv[i]!r} 不在选项表内（合法值如 {opts[:6]}…）")

        # ②' 值为 null：后端虽能用 default 兜住，但前端下拉/输入框会显示为空，
        #     用户看不出「这里到底会用什么」，属于该显式写清的状态。
        nulls = [wseq[i][0] for i in range(min(len(wv), len(wseq))) if wv[i] is None]
        if nulls:
            warn(f"{label}：这些 widget 值为 null（界面显示为空、实际走默认值）：{nulls}")

        # ③ ★ 领域规则
        for rule in SEMANTIC_RULES:
            if rule["node"] != ntype:
                continue
            try:
                hit = bool(rule["when"](wmap))
            except Exception:
                hit = False
            if not hit:
                continue
            got_v = wmap.get(rule["field"])
            if got_v != rule["expect"]:
                err(f"{label}：{rule['field']} = {got_v!r}，应为 {rule['expect']!r}"
                    f" —— {rule['why']}")
            else:
                ok(f"{label}：{rule['field']} = {got_v!r} 符合领域规则")

        # ④ ★ widgets_values_named：当前前端（1.49+）导出的标配字段，
        #    按名字恢复 widget 值；缺失会退化成按位置恢复，widget 被转成
        #    input 之后最容易错位（官方模板 25/29 个节点都带它）。
        wnamed = n.get("widgets_values_named")
        if wseq:
            if wnamed is None:
                warn(f"{label}：缺 widgets_values_named（前端 1.49+ 的标配字段）")
            else:
                want_keys = {w[0] for w in wseq}
                got_keys = set(wnamed.keys())
                if got_keys != want_keys:
                    err(f"{label}：widgets_values_named 键不匹配\n"
                        f"        期望 {sorted(want_keys)}\n"
                        f"        实得 {sorted(got_keys)}")
                else:
                    names_ordered = [w[0] for w in wseq]
                    for kk in sorted(got_keys):
                        i = names_ordered.index(kk)
                        if i < len(wv) and wnamed[kk] != wv[i]:
                            err(f"{label}：{kk} 的 named 值 {wnamed[kk]!r} "
                                f"!= widgets_values 里的 {wv[i]!r}")

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
        if tslot is None:
            err(f"连线 {lid}：target_slot 为 null —— "
                f"{name_of[oid]}[{oslot}] → {name_of[tid]} 接不上"
                f"（目标节点没有这个输入槽）→ ComfyUI 会**丢弃**这条线")
            continue
        if tslot >= len(dst_inputs):
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
    else:
        print("PASS  工作流可直接导入并运行")
    return len(ERRORS)


def _is_workflow(path: str) -> bool:
    """templates/ 下还放着 camera_presets.json 这类**数据**文件，别把它们当工作流。"""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return False
    return isinstance(d, dict) and isinstance(d.get("nodes"), list) and bool(d["nodes"])


def main() -> int:
    """校验命令行指定的工作流；不给参数时校验 templates/ 下的**全部**工作流 json。"""
    asyncio.run(nodes.init_builtin_extra_nodes())
    asyncio.run(nodes.load_custom_node(PKG))

    tdir = os.path.join(PKG, "templates")
    if _CLI_WORKFLOWS:
        paths = list(_CLI_WORKFLOWS)
    else:
        paths = [os.path.join(tdir, f) for f in sorted(os.listdir(tdir))
                 if f.endswith(".json") and _is_workflow(os.path.join(tdir, f))]
    if not paths:
        print("没有要校验的工作流。")
        return 1

    total, failed = 0, []
    for i, p in enumerate(paths):
        if i:
            print()
        n = verify_one(p)
        total += n
        if n:
            failed.append(os.path.basename(p))

    if len(paths) > 1:
        print()
        print("=" * 72)
        print(f"合计：{len(paths)} 个工作流，{len(failed)} 个不通过"
              + (f"（{', '.join(failed)}）" if failed else "（全部通过）"))
        print("=" * 72)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
