"""第 3 层：用 ComfyUI 自己的加载器做真实加载测试。

跑法（**必须**用 ComfyUI 安装根下带 CUDA torch 的那套 python）：
    cd <ComfyUI根>
    .venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tests/test_real_load.py

为什么不能只用桩
----------------
桩验不了「schema 写法在真实 API 上语义不同」。实测案例：
`io.Autogrow.Input(..., min=0)` 在桩里看不出任何问题，
真实环境里却因为落在 required 而**直接堵死纯文生图**。

本层还顺带做**官方节点签名漂移探针** —— ComfyUI 升级后
`MiniMaxH3ReferenceToVideo` 的输入名若被改掉，这里会立刻变红。
"""

from __future__ import annotations

import asyncio
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG_DIR)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)

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
# 必须在 import nodes 之前设置
# --------------------------------------------------------------------------
os.chdir(COMFY_ROOT)
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)
if CUSTOM_NODES not in sys.path:
    sys.path.insert(0, CUSTOM_NODES)
# comfy/model_management.py 在 import 时就会碰 cuda —— 无显卡 / CPU 版 torch 会直接崩
sys.argv = [sys.argv[0], "--cpu"]

print(f"ComfyUI 根目录：{COMFY_ROOT}")

import nodes  # noqa: E402

section("加载自定义节点包")
loaded = asyncio.run(nodes.load_custom_node(PKG_DIR))
ok(loaded is True, "load_custom_node() 返回 True")

EXPECTED = [
    "H3MangaPanelSplitter",
    "H3MangaRefPack",
    "H3MangaScriptLoader",
    "H3MangaShotBinder",
    "H3MangaPromptComposer",
    "H3MangaShotConditioning",
    "H3MangaEpisodeRender",
    "H3MangaConsistencyAudit",
    "H3MangaPlanPreview",
]
mapping = nodes.NODE_CLASS_MAPPINGS
missing = [n for n in EXPECTED if n not in mapping]
ok(not missing, f"9 个 node_id 全部进入 NODE_CLASS_MAPPINGS" + (f"（缺 {missing}）" if missing else ""))

# 与已有插件的前缀冲突检查
others = [k for k in mapping if k.startswith("MiniMaxH3Director")]
ok(not [k for k in EXPECTED if k in others],
   f"与已装的 ComfyUI_MiniMaxH3_Director 无 id 冲突（对方有 {len(others)} 个节点）")

# --------------------------------------------------------------------------
section("INPUT_TYPES() 的真实划分（required / optional）")
# --------------------------------------------------------------------------
# 桩里看不到的东西全在这一层。特别是 Autogrow 必须落在 optional。
for node_id in EXPECTED:
    cls = mapping[node_id]
    try:
        it = cls.INPUT_TYPES()
    except Exception as e:
        ok(False, f"{node_id}: INPUT_TYPES() 可调用 —— {type(e).__name__}: {e}")
        continue
    req = it.get("required", {}) or {}
    opt = it.get("optional", {}) or {}
    all_names = set(req) | set(opt)

    schema = cls.define_schema()
    schema_names = {getattr(i, "id", None) for i in schema.inputs}
    ok(schema_names == all_names,
       f"{node_id}: define_schema 输入名 == INPUT_TYPES 输入名"
       + (f"（差 {schema_names ^ all_names}）" if schema_names != all_names else ""))

    for i in schema.inputs:
        name = getattr(i, "id", None)
        want_optional = bool(getattr(i, "optional", False))
        if want_optional:
            ok(name in opt, f"{node_id}.{name}: 真的落在 optional 里")
        else:
            ok(name in req, f"{node_id}.{name}: 真的落在 required 里")

# --------------------------------------------------------------------------
section("Autogrow 的真实形态（COMFY_AUTOGROW_V3 + template）")
# --------------------------------------------------------------------------
# 实测：Autogrow 在 INPUT_TYPES() 里是
#   ("COMFY_AUTOGROW_V3", {"template": {"prefix": ..., "min": 0, "max": 9}})
# 而不是「输入名.插槽名」展开 —— 展开是前端拿到 template 之后自己做的。
# 所以这一层要断言的是 template 的 prefix/min/max 有没有正确传到前端。
it = mapping["H3MangaRefPack"].INPUT_TYPES()
req = it.get("required") or {}
opt = it.get("optional") or {}

for name, want_prefix, want_max in (
    ("character_images", "character_", 9),
    ("audios", "ref_audio_", 3),
):
    entry = opt.get(name)
    ok(entry is not None, f"{name} 落在 optional（少写 optional 会堵死纯文生图）")
    ok(name not in req, f"{name} 不在 required")
    if entry:
        t = entry[0]
        meta = entry[1] if len(entry) > 1 else {}
        ok(t == "COMFY_AUTOGROW_V3", f"{name} 类型是 COMFY_AUTOGROW_V3（实得 {t}）")
        tpl = (meta or {}).get("template") or {}
        ok(tpl.get("prefix") == want_prefix,
           f"{name} template.prefix={tpl.get('prefix')!r}（期望 {want_prefix!r}）")
        ok(tpl.get("min") == 0, f"{name} template.min=0（实得 {tpl.get('min')}）")
        ok(tpl.get("max") == want_max,
           f"{name} template.max={want_max}（实得 {tpl.get('max')}）")
        ok(bool((meta or {}).get("display_name")), f"{name} 带了中文 display_name")
        inner = (tpl.get("input") or {})
        ok(bool(inner), f"{name} template.input 存在（前端据此生成插槽）")

# 交叉核对：官方 R2V 的 ref_images 上限也应该是 9
try:
    from comfy_extras import nodes_minimax_h3 as _h3
    _it = _h3.MiniMaxH3ReferenceToVideo.INPUT_TYPES()
    _e = (_it.get("optional") or {}).get("ref_images") or (_it.get("required") or {}).get("ref_images")
    _tpl = (_e[1] or {}).get("template") or {} if _e else {}
    ok(_tpl.get("max") == 9,
       f"官方 R2V 的 ref_images 上限仍是 9（实得 {_tpl.get('max')}）—— 与本插件常量一致")
    ok(_tpl.get("prefix") == "ref_image_",
       f"官方 R2V 插槽前缀仍是 ref_image_（实得 {_tpl.get('prefix')!r}）")
except Exception as e:
    ok(False, f"官方 Autogrow 形态探针失败：{type(e).__name__}: {e}")

# --------------------------------------------------------------------------
section("RETURN_TYPES / RETURN_NAMES")
# --------------------------------------------------------------------------
for node_id in EXPECTED:
    cls = mapping[node_id]
    rt = list(getattr(cls, "RETURN_TYPES", []) or [])
    rn = list(getattr(cls, "RETURN_NAMES", []) or [])
    schema = cls.define_schema()
    n_out = len(schema.outputs)

    ok(len(rt) == n_out, f"{node_id}: RETURN_TYPES 长度 {len(rt)} == outputs {n_out}")
    ok(len(rn) == n_out, f"{node_id}: RETURN_NAMES 长度 {len(rn)} == outputs {n_out}")
    ok(all(str(t).isascii() for t in rt),
       f"{node_id}: RETURN_TYPES 全为 ASCII（中文只能是 RETURN_NAMES）"
       + (f"（违规 {[t for t in rt if not str(t).isascii()]}）"
          if any(not str(t).isascii() for t in rt) else ""))
    # 注意：真实加载器下 RETURN_NAMES 装的是 display_name，不是逻辑名
    ids = [getattr(o, "id", None) for o in schema.outputs]
    ok(all(i is None or str(i).isascii() for i in ids),
       f"{node_id}: 输出逻辑 id 为 ASCII 或未设置")

# --------------------------------------------------------------------------
section("自定义类型的类型 ID")
# --------------------------------------------------------------------------
for node_id in ("H3MangaRefPack", "H3MangaScriptLoader", "H3MangaEpisodeRender"):
    cls = mapping[node_id]
    rt = list(getattr(cls, "RETURN_TYPES", []) or [])
    print(f"  {node_id} RETURN_TYPES = {rt}")
ok("H3_MANGA_REFS" in list(getattr(mapping["H3MangaRefPack"], "RETURN_TYPES", [])),
   "参考包输出类型为 H3_MANGA_REFS")
ok("H3_MANGA_PLAN" in list(getattr(mapping["H3MangaScriptLoader"], "RETURN_TYPES", [])),
   "分镜计划输出类型为 H3_MANGA_PLAN")
ok("H3_MANGA_BOARD" in list(getattr(mapping["H3MangaPanelSplitter"], "RETURN_TYPES", [])),
   "分镜格输出类型为 H3_MANGA_BOARD")

# --------------------------------------------------------------------------
section("官方 H3 节点签名漂移探针")
# --------------------------------------------------------------------------
try:
    from comfy_extras import nodes_minimax_h3 as h3

    r2v = h3.MiniMaxH3ReferenceToVideo
    names = {getattr(i, "id", None) for i in r2v.define_schema().inputs}
    need = {"clip", "prompt", "width", "height", "length", "ref_image_size",
            "vae", "audio_vae", "ref_images", "ref_videos",
            "ref_video_audios", "ref_audios"}
    ok(need <= names, f"MiniMaxH3ReferenceToVideo 输入未漂移"
       + (f"（缺 {need - names}）" if not need <= names else ""))

    outs = [getattr(o, "display_name", None) or getattr(o, "id", None)
            for o in r2v.define_schema().outputs]
    ok(len(outs) == 2, f"R2V 仍是 2 个输出（positive / LATENT）：{outs}")

    i2v = h3.MiniMaxH3ImageToVideo
    n2 = {getattr(i, "id", None) for i in i2v.define_schema().inputs}
    ok({"clip", "vae", "prompt", "width", "height", "length",
        "first_frame", "last_frame"} <= n2, "MiniMaxH3ImageToVideo 输入未漂移")

    ok(hasattr(h3, "MiniMaxH3SigmaShift"), "MiniMaxH3SigmaShift 存在")
    ok(hasattr(h3, "MiniMaxH3AddGuide"), "MiniMaxH3AddGuide 存在")

    # 帧对齐常量没变
    ok(h3.align_frame_count(121) == 124, "官方 align_frame_count 行为一致（121→124）")
    ok(h3.FPS == 24, "官方 FPS 仍是 24")
    ok(h3.CANVAS_MULTIPLE == 32, "官方画幅倍数仍是 32")
    ok(h3.REF_IMAGE_SHORT_EDGE == 2048, "官方参考图短边仍是 2048")
    ok(getattr(h3.MiniMaxH3ReferenceToVideo, "__doc__", "") is not None, "R2V 文档字符串存在")
except Exception as e:
    ok(False, f"官方 H3 节点探针失败：{type(e).__name__}: {e}")

try:
    from comfy_extras import nodes_custom_sampler as ncs
    for n in ("KSamplerSelect", "BasicScheduler", "RandomNoise", "BasicGuider",
              "SamplerCustomAdvanced"):
        ok(hasattr(ncs, n), f"官方采样五件套：{n} 存在")
except Exception as e:
    ok(False, f"官方采样节点探针失败：{type(e).__name__}: {e}")

try:
    import nodes as _n
    ok(hasattr(_n, "VAEDecode") and hasattr(_n, "LoraLoaderModelOnly"),
       "内置 VAEDecode / LoraLoaderModelOnly 存在")
except Exception as e:
    ok(False, f"内置节点探针失败：{e}")

# --------------------------------------------------------------------------
section("官方 execute() 可直接调用（不经过框架包装）")
# --------------------------------------------------------------------------
import inspect  # noqa: E402

for cls, meth in ((h3.MiniMaxH3ReferenceToVideo, "execute"),
                  (h3.MiniMaxH3SigmaShift, "execute")):
    fn = getattr(cls, meth, None)
    ok(fn is not None and inspect.isfunction(fn) or inspect.ismethod(fn),
       f"{cls.__name__}.{meth} 是可直接调用的 classmethod")

sig = inspect.signature(ncs.SamplerCustomAdvanced.execute)
ok(list(sig.parameters) == ["noise", "guider", "sampler", "sigmas", "latent_image"],
   f"SamplerCustomAdvanced.execute 形参未变：{list(sig.parameters)}")

sig2 = inspect.signature(h3.MiniMaxH3ReferenceToVideo.execute)
ok("ref_images" in sig2.parameters and "ref_image_size" in sig2.parameters,
   "R2V.execute 仍接受 ref_images / ref_image_size 关键字")

# --------------------------------------------------------------------------
section("WEB_DIRECTORY")
# --------------------------------------------------------------------------
import importlib  # noqa: E402

mod = importlib.import_module(os.path.basename(PKG_DIR))
web = getattr(mod, "WEB_DIRECTORY", None)
ok(web and os.path.isdir(web), f"WEB_DIRECTORY 存在且是目录：{web}")

# --------------------------------------------------------------------------
section("★ 真实 common_upscale 下的参考图维度契约")
# --------------------------------------------------------------------------
# 线上报错就抛在 comfy/utils.py::common_upscale 里：
#   RuntimeError: shape '[1, -1, 3, 1152, 32]' is invalid for input of size 36864
# 这里用**真实函数**（不是测试里复刻的版本）把「4 维能过 / 3 维必炸」钉死。
import math  # noqa: E402
import torch  # noqa: E402
import comfy.utils as _cu  # noqa: E402

if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
import core.image_ops as _iops  # noqa: E402


def _official_ref_resize(img, width=1344, height=768):
    """复刻官方 MiniMaxH3ReferenceToVideo 对一张参考图做的 resize。"""
    h, w = int(img.shape[1]), int(img.shape[2])
    scale = min(1.0, math.sqrt((width * height) / float(w * h)))
    tw = max(32, int(round(w * scale / 32)) * 32)
    th = max(32, int(round(h * scale / 32)) * 32)
    samples = img[:1][..., :3].movedim(-1, 1)
    out = _cu.common_upscale(samples, tw, th, "lanczos", "disabled")
    return tuple(out.movedim(1, -1).shape)


panel3 = torch.zeros((768, 1152, 3), dtype=torch.float32)   # 插件内部约定：一张图 = [H,W,C]
_e = None
try:
    _official_ref_resize(panel3)
except Exception as e:      # noqa: BLE001
    _e = e
ok(_e is not None and "shape '[1, -1, 3, 1152, 32]'" in str(_e),
   f"★ 3 维参考图在真实 common_upscale 里复现线上报错（{_e}）")

panel4 = _iops.as_ref_image(panel3)
ok(tuple(panel4.shape) == (1, 768, 1152, 3),
   f"as_ref_image 升成 4 维：{tuple(panel4.shape)}")
ok(_official_ref_resize(panel4) == (1, 768, 1152, 3),
   f"★ 升维后真实 common_upscale 通过，目标尺寸正常：{_official_ref_resize(panel4)}")

# --------------------------------------------------------------------------
section("★ 端到端：插件产出的 ref_images 直接喂给官方 R2V（真 _resize）")
# --------------------------------------------------------------------------
# vae=None 时官方 R2V 唯一的重活就是参考图 resize —— 也就是线上炸掉的那一段。
# 用假 clip 把文本编码器摘掉，就能在 CPU 上端到端验证「插件的张量契约 ✓ 官方肯收」。
import core.refs as _refs  # noqa: E402


class _FakeClip:
    """只记下官方交给 tokenizer 的 ref_items（不需要 32B 文本编码器）。"""

    def __init__(self):
        self.items = []

    def tokenize(self, prompt, minimax_ref_items=None, **kw):
        self.items = list(minimax_ref_items or [])
        return {"qwen3vl_32b": [[]]}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 1, 4), {}]]


# 真实 ComfyUI 里参考图一定是 torch 张量（IMAGE 类型），所以这里也用 torch
_panel = torch.full((768, 1152, 3), 0.5, dtype=torch.float32)
_pack = _refs.build_ref_pack(
    character_names="纲吉",
    character_images=[torch.full((512, 512, 3), 0.3, dtype=torch.float32)],
    style_name="风格", style_images=[torch.full((256, 256, 3), 0.2, dtype=torch.float32)],
)
_layout = _refs.shot_ref_layout(_pack, characters=["纲吉"], panel_image=_panel, has_panel=True)
_clip = _FakeClip()


def _call_official(ref_images, clip):
    return h3.MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt="<Picture 1> follow the panel", width=1344, height=768,
        length=107, ref_image_size="match", vae=None, audio_vae=None,
        ref_images=ref_images, ref_videos={}, ref_video_audios={}, ref_audios={},
    )


_ok_out = None
_err_e2e = None
try:
    _ok_out = _call_official(_layout["ref_images"], _clip)
except Exception as e:      # noqa: BLE001
    _err_e2e = e
ok(_ok_out is not None,
   f"★ 官方 R2V 吃下插件产出的 ref_images（3 张：格图 + 纲吉 + 风格）：{_err_e2e}")
ok(len(_clip.items) == 3,
   f"官方按顺序收到 3 个参考项，<Picture i> 不会错位（实得 {len(_clip.items)}）")

# 反向验证：把其中一张退回 3 维（= 修复前的样子）→ 必须复现线上报错
_bad = dict(_layout["ref_images"])
_bad["ref_image_0"] = _panel          # 3 维 [H,W,C]
_err_back = None
try:
    _call_official(_bad, _FakeClip())
except Exception as e:      # noqa: BLE001
    _err_back = e
ok(_err_back is not None and "shape '[1, -1, 3, 1152, 32]'" in str(_err_back),
   f"★ 反向验证：退回 3 维后官方 R2V 立刻复现线上报错（{_err_back}）")

print(f"\n{'=' * 60}\n第 3 层真实加载：{PASS} 通过 / {FAIL} 失败\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
