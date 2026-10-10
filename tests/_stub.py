"""第 2 层测试用的 comfy_api 桩。

目标：在没有 ComfyUI、没有显卡的情况下，验证
  * ``define_schema()`` 的输入名与 ``execute()`` 形参是否对齐
  * optional / default 是否落对
  * Combo 的 default 是否在 options 里
  * RETURN_TYPES / RETURN_NAMES 长度与 outputs 一致

桩要**故意宽容**（不校验业务逻辑），把失败留给断言 —— 报错才指向病因。
"""

from __future__ import annotations

import sys
import types


class _Port:
    """io.Xxx 的替身：既能 .Input(...) 也能 .Output(...)。"""

    def __init__(self, kind: str):
        self.kind = kind

    def Input(self, name, **kw):
        return {"io": "in", "kind": self.kind, "name": name, "kw": kw}

    def Output(self, **kw):
        return {"io": "out", "kind": self.kind, "name": kw.get("display_name") or self.kind, "kw": kw}


class _CustomPort(_Port):
    def __init__(self, io_type: str):
        super().__init__("CUSTOM")
        self.io_type = io_type


class _AutogrowInput:
    def __init__(self, id, template=None, optional=False, display_name=None, tooltip=None,
                 lazy=None, extra_dict=None):
        self.id = id
        self.template = template
        self.optional = optional
        self.display_name = display_name
        self.tooltip = tooltip
        self.kind = "AUTOGROW"


class _Autogrow:
    Type = dict

    class TemplatePrefix:
        def __init__(self, input=None, prefix="", min=0, max=10):
            self.input = input
            self.prefix = prefix
            self.min = min
            self.max = max

    class TemplateNames:
        def __init__(self, input=None, names=None, min=0):
            self.input = input
            self.names = names or []
            self.min = min

    Input = _AutogrowInput


class _Schema:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.node_id = kw.get("node_id")
        self.display_name = kw.get("display_name")
        self.category = kw.get("category", "")
        self.description = kw.get("description", "")
        self.inputs = list(kw.get("inputs") or [])
        self.outputs = list(kw.get("outputs") or [])
        self.hidden = list(kw.get("hidden") or [])
        self.is_output_node = bool(kw.get("is_output_node", False))


class _NodeOutput:
    def __init__(self, *args, ui=None, **kw):
        self.result = tuple(args)
        self.ui = ui


def install() -> None:
    """把假 comfy_api 注入 sys.modules（幂等）。"""
    if "comfy_api.latest" in sys.modules and getattr(
        sys.modules["comfy_api.latest"], "_H3MANGA_STUB", False
    ):
        return

    io_ns = types.ModuleType("io")

    io_ns.ComfyNode = type("ComfyNode", (), {})
    io_ns.Schema = _Schema
    io_ns.NodeOutput = _NodeOutput
    io_ns.Custom = lambda io_type: _CustomPort(io_type)
    io_ns.Autogrow = _Autogrow
    io_ns.Hidden = type("Hidden", (), {"unique_id": "UNIQUE_ID", "prompt": "PROMPT",
                                       "extra_pnginfo": "EXTRA_PNGINFO"})

    for _n in ("Model", "Clip", "Vae", "Image", "Mask", "Latent", "String", "Int",
               "Float", "Boolean", "Combo", "Conditioning", "Audio", "Video",
               "Guider", "Noise", "Sampler", "Sigmas", "AnyType", "MultiType",
               "ModelPatch", "Hooks", "LoraModel", "StyleModel"):
        setattr(io_ns, _n, _Port(_n))

    latest = types.ModuleType("comfy_api.latest")
    latest._H3MANGA_STUB = True
    latest.io = io_ns
    latest.ui = types.ModuleType("ui")
    latest.ui.PreviewImage = type("PreviewImage", (), {"__init__": lambda s, *a, **k: None})
    latest.ComfyExtension = type("ComfyExtension", (), {})

    root = types.ModuleType("comfy_api")
    root.latest = latest

    sys.modules["comfy_api"] = root
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.io"] = io_ns
