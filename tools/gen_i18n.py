"""从节点 schema 生成 i18n 文件与前端 labels 模块。

为什么需要这个脚本
------------------
中文文案有三处落点，必须同源：

  1. 输入级 ``display_name``（后端直接给，最优先）
  2. ``locales/zh/nodeDefs.json``（前端语言切到中文时生效）
  3. ``js/h3manga_labels.js``（前端 ``widget.label`` 兜底，一定生效）

改一处忘两处 = 界面上半中半英。所以文案只在 ``core/ui.py`` 里写一次，
由本脚本生成 2 和 3。

用法（必须用 ComfyUI 自带的 python，且 cwd 为 ComfyUI 根目录）：

    cd <ComfyUI根>
    .venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tools/gen_i18n.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)
PKG_NAME = os.path.basename(PKG)


def _get_schemas():
    """用真实 ComfyUI 加载器拿到每个节点的 define_schema()。"""
    os.chdir(COMFY_ROOT)
    if COMFY_ROOT not in sys.path:
        sys.path.insert(0, COMFY_ROOT)
    if CUSTOM_NODES not in sys.path:
        sys.path.insert(0, CUSTOM_NODES)
    # 必须在 import comfy 之前：comfy/model_management.py 在 import 时就会碰 cuda
    sys.argv = [sys.argv[0], "--cpu"]

    import importlib
    pkg = importlib.import_module(PKG_NAME)
    mapping = getattr(pkg, "NODE_CLASS_MAPPINGS", {})
    out = {}
    for node_id, cls in mapping.items():
        try:
            out[node_id] = cls.define_schema()
        except Exception as e:
            print(f"  [skip] {node_id}: {e}")
    return out


def _name_of(obj) -> str:
    """真实 API 下 inputs/outputs 是 io.Input / io.Output 对象，取 .id / .display_name。"""
    return getattr(obj, "id", None) or getattr(obj, "display_name", None) or ""


def _label_of(obj) -> str:
    return getattr(obj, "display_name", None) or _name_of(obj)


def main() -> int:
    schemas = _get_schemas()
    if not schemas:
        print("没有拿到任何节点 schema —— 请确认 cwd 是 ComfyUI 根目录，且用 .venv 的 python。")
        return 1

    node_defs = {}
    js_labels = {}

    for node_id, schema in schemas.items():
        entry = {
            "display_name": getattr(schema, "display_name", node_id) or node_id,
            "inputs": {},
            "outputs": {},
        }
        for inp in (getattr(schema, "inputs", []) or []):
            name = _name_of(inp)
            label = _label_of(inp)
            if name:
                entry["inputs"][name] = {"name": label}
        for i, out in enumerate(getattr(schema, "outputs", []) or []):
            label = _label_of(out)
            entry["outputs"][str(i)] = {"name": label}

        node_defs[node_id] = entry
        js_labels[node_id] = {
            "display_name": entry["display_name"],
            "inputs": {k: v["name"] for k, v in entry["inputs"].items()},
            "outputs": entry["outputs"],
        }

    # ---- locales/zh/nodeDefs.json（扁平结构，不要外层 nodeDefs 包裹）----
    loc_dir = os.path.join(PKG, "locales", "zh")
    os.makedirs(loc_dir, exist_ok=True)
    loc_path = os.path.join(loc_dir, "nodeDefs.json")
    with open(loc_path, "w", encoding="utf-8") as f:
        json.dump(node_defs, f, ensure_ascii=False, indent=2)
    print(f"已写入 {loc_path}（{len(node_defs)} 个节点）")

    # ---- js/h3manga_labels.js ----
    js_path = os.path.join(PKG, "js", "h3manga_labels.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("// 本文件由 tools/gen_i18n.py 自动生成，请勿手改。\n")
        f.write("// 真源：core/ui.py + 各节点的 define_schema()\n\n")
        f.write("export const NODE_LABELS = ")
        json.dump(js_labels, f, ensure_ascii=False, indent=2)
        f.write(";\n\nexport default NODE_LABELS;\n")
    print(f"已写入 {js_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
