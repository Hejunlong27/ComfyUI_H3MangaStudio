"""资源层节点：显存 / 系统内存释放。

本机的硬约束：R2V UNET 19.53 GB + 文本编码器 13.93 GB = 33.5 GB 权重，
而物理内存只有 16 GB、显存只有 8 GB。三个阶段的模型同时驻留必然爆 ——
所以需要一个「用完就卸」的显式手段。

用法（两种）
------------
1. **自动**：整集渲染节点的 ``unload_stages`` 打开后，
   会在「编码 → 采样 → 解码」三个阶段的边界自动卸载，通常不需要本节点。
2. **手动**：把本节点接在任意位置，把要释放的模型对象接进来。
   ``trigger`` 输入用于**保证执行顺序** —— ComfyUI 只按依赖关系决定执行次序，
   一个没有任何输入连线的节点可能被排到最前面执行，所以想让它在渲染之后跑，
   就把渲染节点的 ``report`` 输出接到 ``trigger`` 上。
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

try:
    from ..core import mem as memmod
    from ..core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from ..core.ui import safe_display_name as _inp, label_input as _L
except ImportError:  # pragma: no cover
    from core import mem as memmod
    from core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from core.ui import safe_display_name as _inp, label_input as _L


class H3MangaFreeMemory(io.ComfyNode):
    """卸载模型，释放显存与系统内存。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaFreeMemory",
            display_name=_NL["H3MangaFreeMemory"],
            category="H3 Manga/资源",
            description=(
                "把接进来的模型从显存与内存里卸载掉，输出释放前后的用量对比。\n"
                "本机权重总量 33.5GB（R2V UNET 19.5GB + 文本编码器 13.9GB）"
                "而内存只有 16GB —— 串行加载、用完即卸是唯一能跑通的方式。\n"
                "提示：卸载是安全的，ComfyUI 下次用到该模型会自动重新加载。\n"
                "提示：把上游节点的输出接到 trigger，可保证本节点在它之后执行。"
            ),
            inputs=[
                _inp(io.String.Input, "trigger", _L("trigger"),
                     force_input=True, optional=True,
                     tooltip="接任意上游节点的输出（如渲染节点的 report），仅用于保证执行顺序"),
                _inp(io.Model.Input, "model", _L("model"), optional=True),
                _inp(io.Clip.Input, "clip", _L("clip"), optional=True),
                _inp(io.Vae.Input, "vae", _L("vae"), optional=True),
                _inp(io.Vae.Input, "audio_vae", _L("audio_vae"), optional=True),
                _inp(io.Boolean.Input, "unload_all", _L("unload_all"), default=False,
                     tooltip="勾上 = 卸载全部已加载模型（不只是接进来的这几个）"),
                _inp(io.Boolean.Input, "run_gc", _L("run_gc"), default=True,
                     tooltip="额外做一次 gc.collect()，释放 Python 侧引用，"
                             "让 mmap 的权重页可被系统回收"),
                _inp(io.String.Input, "label", _L("label"), default="",
                     tooltip="给这次释放起个名字，会出现在报告里，便于对照流水线的哪一步"),
            ],
            outputs=[
                io.String.Output(display_name=_OL["report"]),
            ],
            # 终末节点：即使下游没人接，它也一定会被执行
            is_output_node=True,
        )

    @classmethod
    def execute(cls, trigger=None, model=None, clip=None, vae=None, audio_vae=None,
                unload_all=False, run_gc=True, label="") -> io.NodeOutput:
        objs = [model, clip, vae, audio_vae]
        res = memmod.unload(objs, unload_all=bool(unload_all),
                            run_gc=bool(run_gc), label=str(label or ""))

        report = memmod.format_unload_report(res)
        if all(o is None for o in objs) and not unload_all:
            report += ("\n\n注意：没有接任何模型，也没有勾选「全部卸载」—— 本次什么都没做。\n"
                       "     请把要释放的 model / clip / vae 接进来，或勾上 unload_all。")
        return io.NodeOutput(report, ui={"h3manga_panel": [report]})


NODES = [H3MangaFreeMemory]
