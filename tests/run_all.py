"""一键跑全部测试（四层 + 工作流自检）。

用法：
    "<安装根>/standalone-env/python.exe" custom_nodes/ComfyUI_H3MangaStudio/tests/run_all.py

第 1 / 2 / 4 层用 standalone-env（CPU torch 足够，更快更安全）；
第 3 层需要 **CUDA 版 torch**（要 import ComfyUI 的 nodes），
第 5 层需要真实节点定义（同样要 import nodes），
所以这两层会用 `<ComfyUI>/.venv/Scripts/python.exe` 起子进程跑。
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)
INSTALL_ROOT = os.path.dirname(COMFY_ROOT)

STANDALONE = os.path.join(INSTALL_ROOT, "standalone-env", "python.exe")
VENV = os.path.join(COMFY_ROOT, ".venv", "Scripts", "python.exe")


def run(title, script, python_exe, cwd):
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)
    if not os.path.isfile(python_exe):
        print(f"  [skip] 找不到 python：{python_exe}")
        return None
    r = subprocess.run([python_exe, script], cwd=cwd)
    return r.returncode


def main() -> int:
    results = {}

    results["第1层 纯逻辑"] = run(
        "第 1 层 · 纯逻辑（帧对齐 / 解析 / 逐镜重排 / 切格）",
        os.path.join(HERE, "test_pure_logic.py"), STANDALONE, PKG)

    results["第2层 schema桩"] = run(
        "第 2 层 · schema 桩（输入名 ↔ execute 形参）",
        os.path.join(HERE, "test_schema_stub.py"), STANDALONE, PKG)

    results["第4层 契约"] = run(
        "第 4 层 · 契约（桩替换 + 真跑 execute + 反向验证）",
        os.path.join(HERE, "test_render_contract.py"), STANDALONE, PKG)

    results["第3层 真实加载"] = run(
        "第 3 层 · 真实加载（ComfyUI 自己的加载器）",
        os.path.join(HERE, "test_real_load.py"), VENV, COMFY_ROOT)

    # ★ 第 5 层：工作流自检。把「模板 ↔ 真实 INPUT_TYPES」的一致性检查
    #   焊进回归 —— 否则模板里的错（widget 值错位 / 连线接不上 / 参数取错默认值）
    #   照样能一路溜到出片现场：早先 CLIPLoader.type 一直是默认的
    #   stable_diffusion（应为 minimax），任何测试都没拦住。
    results["第5层 工作流自检"] = run(
        "第 5 层 · 工作流自检（templates/*.json ↔ 真实 INPUT_TYPES + 领域规则）",
        os.path.join(PKG, "tools", "verify_workflow.py"), VENV, COMFY_ROOT)

    print("\n" + "=" * 70)
    bad = 0
    for k, v in results.items():
        state = "SKIP" if v is None else ("PASS" if v == 0 else "FAIL")
        if v not in (0, None):
            bad += 1
        print(f"  {state:<5} {k}")
    print("=" * 70)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
