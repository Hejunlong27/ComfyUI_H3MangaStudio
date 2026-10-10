"""测试用：把节点包当普通包 import 进来（第 2 / 4 层用）。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent
CUSTOM_NODES = PKG_DIR.parent
PKG_NAME = PKG_DIR.name

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))


def load_package(with_stub: bool = True):
    """import 节点包；with_stub=True 时先注入假 comfy_api。"""
    if with_stub:
        import _stub  # noqa: WPS433
        _stub.install()

    if str(CUSTOM_NODES) not in sys.path:
        sys.path.insert(0, str(CUSTOM_NODES))

    # 清掉可能残留的半成品模块
    for k in list(sys.modules):
        if k == PKG_NAME or k.startswith(PKG_NAME + "."):
            del sys.modules[k]

    return importlib.import_module(PKG_NAME)


def core():
    """只拿 core（不需要 comfy_api 桩）。"""
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    import core as core_pkg  # noqa: WPS433
    return core_pkg
