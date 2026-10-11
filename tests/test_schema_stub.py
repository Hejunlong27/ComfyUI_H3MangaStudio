"""第 2 层：schema 桩测试（不需要 ComfyUI / 显卡）。

跑法（standalone-env 即可，不需要 CUDA 版 torch）：
    "<安装根>/standalone-env/python.exe" tests/test_schema_stub.py

这一层抓的是**最贵的一类 bug**：`define_schema()` 里改了输入名，
`execute()` 形参没跟着改 —— ComfyUI 只会在执行时抛 TypeError，白等一轮采样。
"""

from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_package  # noqa: E402

PASS = 0
FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------
# schema 归一化：桩里 inputs 是 dict，真实 API 里是 io.Input 对象 —— 两种都要吃
# --------------------------------------------------------------------------
def norm_inputs(schema):
    out = []
    for i in schema.inputs:
        if isinstance(i, dict):
            out.append({
                "name": i.get("name"),
                "kw": dict(i.get("kw") or {}),
                "kind": i.get("kind"),
                "template": None,
            })
            continue
        kind = type(i).__name__.lstrip("_")
        if kind == "AutogrowInput" or hasattr(i, "template"):
            kind = "AUTOGROW"
        elif kind in ("Input", "Port"):
            kind = getattr(getattr(i, "Parent", None), "io_type", kind)
        out.append({
            "name": getattr(i, "id", None),
            "kw": {
                "optional": bool(getattr(i, "optional", False)),
                "display_name": getattr(i, "display_name", None),
                "tooltip": getattr(i, "tooltip", None),
            },
            "kind": kind,
            "template": getattr(i, "template", None),
        })
    return out


def norm_outputs(schema):
    out = []
    for o in schema.outputs:
        if isinstance(o, dict):
            out.append({"name": o.get("name")})
        else:
            out.append({"name": getattr(o, "display_name", None) or getattr(o, "id", None)})
    return out


pkg = load_package(with_stub=True)
mapping = getattr(pkg, "NODE_CLASS_MAPPINGS", {})
schemas = {nid: cls.define_schema() for nid, cls in mapping.items()}
INPUTS = {nid: norm_inputs(s) for nid, s in schemas.items()}
OUTPUTS = {nid: norm_outputs(s) for nid, s in schemas.items()}

print(f"载入 {len(schemas)} 个节点：{', '.join(sorted(schemas))}")

# --------------------------------------------------------------------------
section("节点元信息")
# --------------------------------------------------------------------------
ok(len(schemas) == 14, f"节点数量为 14（实得 {len(schemas)}）")
ok(all(s.category for s in schemas.values()), "所有节点都有 category")
ok(all(s.display_name for s in schemas.values()), "所有节点都有 display_name")
ok(len(set(schemas)) == len(schemas), "node_id 全局唯一")
ok(all("H3Manga" in nid for nid in schemas),
   "所有 node_id 带 H3Manga 前缀（与 ComfyUI_MiniMaxH3_Director 的 MiniMaxH3Director* 刻意区分）")

# --------------------------------------------------------------------------
section("输入名 ↔ execute 形参对齐")
# --------------------------------------------------------------------------
for node_id, schema in schemas.items():
    cls = mapping[node_id]
    sig = inspect.signature(cls.execute)
    params = [p for p in sig.parameters if p != "self"]
    schema_in = [i["name"] for i in INPUTS[node_id]]
    missing = [n for n in schema_in if n not in params]
    extra = [p for p in params if p not in schema_in]
    ok(not missing, f"{node_id}: schema 输入都在 execute 形参里" +
       (f"（缺 {missing}）" if missing else ""))
    ok(not extra, f"{node_id}: execute 没有多余形参" +
       (f"（多 {extra}）" if extra else ""))

# --------------------------------------------------------------------------
section("optional / default 落位")
# --------------------------------------------------------------------------
for node_id, schema in schemas.items():
    cls = mapping[node_id]
    sig = inspect.signature(cls.execute)
    problems = []
    for inp in INPUTS[node_id]:
        name, kw = inp["name"], inp["kw"]
        if name not in sig.parameters:
            continue
        # 判定条件必须是 (default in kw) or kw.get("optional")，
        # 只判 "default" in kw 会把 optional=True 的输入误判成必填
        has_default_semantics = ("default" in kw) or bool(kw.get("optional"))
        has_py_default = sig.parameters[name].default is not inspect.Parameter.empty
        if has_default_semantics and not has_py_default:
            problems.append(f"{name}(schema 可选但形参无默认值)")
        if has_py_default and not has_default_semantics:
            problems.append(f"{name}(形参有默认值但 schema 是必填)")
    ok(not problems, f"{node_id}: optional/default 与形参一致" +
       (f"（{problems}）" if problems else ""))

# --------------------------------------------------------------------------
section("Autogrow 必须同时给 min=0 和 optional=True")
# --------------------------------------------------------------------------
# 实测坑：min=0 只管「组内插槽数量」，管不了「这个输入整体是否必填」。
# 少写 optional=True，io.Autogrow.Input 会落到 required，
# 于是「纯文生视频」那条路直接被堵死。
autogrow_count = 0
for node_id, schema in schemas.items():
    for inp in INPUTS[node_id]:
        if inp["kind"] != "AUTOGROW":
            continue
        autogrow_count += 1
        ok(bool(inp["kw"].get("optional")),
           f"{node_id}.{inp['name']}: Autogrow 显式写了 optional=True")
        tpl = inp["template"]
        minv = getattr(tpl, "min", 0)
        ok(minv == 0, f"{node_id}.{inp['name']}: Autogrow min=0（当前 {minv}）")
ok(autogrow_count >= 2, f"确实存在 Autogrow 输入（共 {autogrow_count} 个）")

# --------------------------------------------------------------------------
section("Combo 的 default 必须在 options 里")
# --------------------------------------------------------------------------
for node_id, schema in schemas.items():
    for inp in INPUTS[node_id]:
        if inp["kind"] != "Combo":
            continue
        opts = inp["kw"].get("options")
        default = inp["kw"].get("default")
        if opts is None:
            continue
        ok(default in opts,
           f"{node_id}.{inp['name']}: default={default!r} 在 options 内（共 {len(opts)} 项）")

# --------------------------------------------------------------------------
section("中文标签：输入名保持 ASCII，标签走 display_name")
# --------------------------------------------------------------------------
for node_id, schema in schemas.items():
    inputs = INPUTS[node_id]
    non_ascii = [i["name"] for i in inputs if not str(i["name"]).isascii()]
    ok(not non_ascii,
       f"{node_id}: 输入名全为 ASCII（避免污染 API 键名）" +
       (f"（违规 {non_ascii}）" if non_ascii else ""))
    unlabeled = [i["name"] for i in inputs
                 if not str(i["kw"].get("display_name") or "").strip()]
    ok(not unlabeled,
       f"{node_id}: 所有输入都带了 display_name（{len(inputs) - len(unlabeled)}/{len(inputs)}）" +
       (f"（缺 {unlabeled}）" if unlabeled else ""))

# --------------------------------------------------------------------------
section("输出名唯一 + 数量")
# --------------------------------------------------------------------------
for node_id, schema in schemas.items():
    names = [o["name"] for o in OUTPUTS[node_id]]
    ok(len(names) == len(set(names)), f"{node_id}: 输出名唯一")
    ok(len(names) >= 1, f"{node_id}: 至少有一个输出")

# --------------------------------------------------------------------------
section("纯计算节点：用 schema 默认值直接跑通")
# --------------------------------------------------------------------------
RUNTIME_NEEDED = {
    "H3MangaPanelSplitter", "H3MangaScriptLoader", "H3MangaShotBinder",
    "H3MangaPromptComposer", "H3MangaShotConditioning", "H3MangaEpisodeRender",
    "H3MangaConsistencyAudit", "H3MangaPlanPreview",
    # v0.7 前置出栈：需要真实 plan / clip 才有意义（缺输入时会优雅返回空结果，
    # 由第 4 层用桩对象覆盖真实路径）
    "H3MangaShotOut", "H3MangaShotCond",
    # v0.79 VLM 层：⑬ 要调外部 API（缺 key 必须报错，不能静默），
    # ⑭ 要上游的 VLM 结果 —— 都由第 4 层用桩覆盖。
    "H3MangaVLMScribe", "H3MangaVLMPlan",
}
ran = 0
for node_id, schema in schemas.items():
    if node_id in RUNTIME_NEEDED:
        print(f"  [skip] {node_id}：需要真实输入（图 / 计划 / 模型），由第 4 层覆盖")
        continue
    cls = mapping[node_id]
    kwargs = {i["name"]: i["kw"].get("default") for i in INPUTS[node_id]}
    try:
        out = cls.execute(**kwargs)
        n_ret = len(getattr(out, "result", ()))
        ran += 1
        ok(n_ret == len(OUTPUTS[node_id]),
           f"{node_id}: 默认值跑通，返回 {n_ret} 项 == outputs {len(OUTPUTS[node_id])}")
    except Exception as e:
        ok(False, f"{node_id}: 默认值跑通失败 —— {type(e).__name__}: {e}")
ok(ran >= 1, "至少有一个节点做了「默认值跑通」验证")

# --------------------------------------------------------------------------
section("负向守卫：不要用 validate_inputs")
# --------------------------------------------------------------------------
# ComfyUI 的 validate_inputs 有两个硬伤（execution.py 源码已核实）：
#   ① 校验阶段 execution_list 是 None，所有**连线输入**被标成 missing → 接好线也误报
#   ② 一条校验失败会为每个输入各生成一条错误 → 29 个输入显示 29 条同样的话
# 所以本包统一把检查放进 execute() 开头的 _preflight()。
for node_id, cls in mapping.items():
    src = inspect.getsource(cls)
    ok("def validate_inputs" not in src, f"{node_id}: 没有实现 validate_inputs")

# --------------------------------------------------------------------------
section("前端资源")
# --------------------------------------------------------------------------
web = getattr(pkg, "WEB_DIRECTORY", None)
ok(web is not None and os.path.isdir(web), f"WEB_DIRECTORY 指向真实目录：{web}")
js_files = []
if web and os.path.isdir(web):
    js_files = sorted(f for f in os.listdir(web) if f.endswith(".js"))
ok("h3manga_preview.js" in js_files, f"前端入口存在（目录内 .js：{js_files}）")

print(f"\n{'=' * 60}\n第 2 层 schema 桩：{PASS} 通过 / {FAIL} 失败\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
