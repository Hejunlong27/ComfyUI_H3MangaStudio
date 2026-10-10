"""官方 H3 能力适配层：**只调官方节点，绝不自己拼 conditioning**。

为什么必须这样
--------------
H3 的参考素材靠 conditioning 里的 ``minimax_refs`` / ``minimax_keyframes`` 传递，
而提示词侧的 ``<Picture i>`` 标签依赖 tokenizer 给出的 ``minimax_ref_items``
序列位置。这套口径**跟 ComfyUI 版本强相关**，自己拼必然跟丢，
而且升级后会**静默失效**（不报错，只是角色串号）。

所以本模块的所有函数都是「官方节点的薄包装」：
用 **特征探测**（不是版本号）判断官方节点是否存在，缺了就抛中文错误。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# 官方节点缺失时的统一中文提示
_MISSING_HINT = (
    "当前 ComfyUI 缺少官方 MiniMax H3 节点（{names}）。\n"
    "本插件不自己实现 H3 conditioning，必须复用官方节点。\n"
    "请把 ComfyUI 升级到包含 comfy_extras/nodes_minimax_h3.py 的版本"
    "（实测 0.38.2 已内置），或检查 comfy_extras 是否被裁剪。"
)


class H3ApiError(RuntimeError):
    """H3 官方能力不可用 / 调用失败。消息已是中文，可直接展示给用户。"""


# --------------------------------------------------------------------------
# 特征探测（延迟导入，不在模块顶层 import comfy）
# --------------------------------------------------------------------------
def get_h3_classes(required: Tuple[str, ...] = ("MiniMaxH3ReferenceToVideo",)) -> Dict[str, Any]:
    """返回 ``{类名: 类对象}``。缺失时抛带中文说明的 H3ApiError。"""
    out: Dict[str, Any] = {}
    missing: List[str] = []

    try:
        from comfy_extras import nodes_minimax_h3 as h3  # 延迟导入
    except Exception as e:  # pragma: no cover
        raise H3ApiError(_MISSING_HINT.format(names="nodes_minimax_h3 模块")) from e

    for name in required:
        cls = getattr(h3, name, None)
        if cls is None:
            missing.append(name)
        else:
            out[name] = cls

    if missing:
        raise H3ApiError(_MISSING_HINT.format(names=", ".join(missing)))
    return out


def _result_of(node_output: Any) -> Tuple[Any, ...]:
    """兼容取 NodeOutput 的返回值：V3 的 .result 是 args 元组。"""
    if node_output is None:
        return ()
    res = getattr(node_output, "result", None)
    if res is not None:
        return tuple(res) if isinstance(res, (list, tuple)) else (res,)
    if isinstance(node_output, tuple):
        return node_output
    return (node_output,)


# --------------------------------------------------------------------------
# 出站守卫：官方 R2V 只吃 4 维 [B,H,W,C]
# --------------------------------------------------------------------------
def _normalize_media_refs(ref_images: Dict[str, Any], ref_videos: Dict[str, Any]):
    """把要交给官方节点的参考图/视频规整成官方要求的维度。

    ★ 这是最后一道网。正常情况下 ``core/refs.py::shot_ref_layout()`` 已经升过维了，
    这里只是防止「有人绕过它自己拼 ref_images」。

    症状提醒（真实事故）：3 维 ``[H,W,C]`` 会让官方按 ``shape[1]/shape[2]`` 读成
    (宽, 通道数)，最后炸在 ``comfy/utils.py::common_upscale`` 里，报错长得像
    ``shape '[1, -1, 3, 1152, 32]' is invalid for input of size 36864`` ——
    与形状毫无字面关联，所以这里宁可自己修一遍，也不要把它放过去。

    规整不了（不是张量 / 维度莫名其妙）时抛带槽位名的中文错误。
    """
    from .image_ops import as_ref_image, as_ref_video  # 延迟导入，保持本模块的 comfy 边界干净

    imgs: Dict[str, Any] = {}
    for k, v in (ref_images or {}).items():
        if v is None:
            continue
        try:
            t = as_ref_image(v, None, f"槽位 {k}")
        except Exception as e:
            raise H3ApiError(
                f"参考图槽位 {k} 的形状 {getattr(v, 'shape', None)} 无法规整成官方要求的 "
                f"[1,H,W,C]：{e}"
            ) from e
        if t is None:
            raise H3ApiError(f"参考图槽位 {k} 拿到的不是图像张量（{type(v).__name__}）")
        imgs[k] = t

    vids: Dict[str, Any] = {}
    for k, v in (ref_videos or {}).items():
        if v is None:
            continue
        try:
            t = as_ref_video(v, None, f"槽位 {k}")
        except Exception as e:
            raise H3ApiError(
                f"参考视频槽位 {k} 的形状 {getattr(v, 'shape', None)} 无法规整成官方要求的 "
                f"[T,H,W,C]：{e}"
            ) from e
        if t is not None:
            vids[k] = t
    return imgs, vids


# --------------------------------------------------------------------------
# 条件编码
# --------------------------------------------------------------------------
def encode_reference_to_video(
    clip: Any,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_size: str = "match",
    vae: Any = None,
    audio_vae: Any = None,
    ref_images: Optional[Dict[str, Any]] = None,
    ref_videos: Optional[Dict[str, Any]] = None,
    ref_video_audios: Optional[Dict[str, Any]] = None,
    ref_audios: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """包装官方 ``MiniMaxH3ReferenceToVideo``。返回 (positive, latent)。

    ref_images 的键必须是 ``ref_image_0`` .. ``ref_image_8``（官方 Autogrow 展开名）。
    空 dict / None 都会被官方当成「无参考图」，安全。

    两件事在这里收口（别再往外散）：
    1. **维度守卫**：官方的 ``h, w = img.shape[1], img.shape[2]`` 是按 ``[B,H,W,C]`` 读的，
       所以出站前统一规整成 4 维（见 ``_normalize_media_refs``）。
    2. **空提示词**：官方收到空 prompt 只会出无意义画面且不报错，这里直接拦住。
    """
    cls = get_h3_classes(("MiniMaxH3ReferenceToVideo",))["MiniMaxH3ReferenceToVideo"]

    if not (prompt or "").strip():
        raise H3ApiError(
            "提示词是空的。MiniMax H3 收到空 prompt 不会报错，只会出无意义画面 —— "
            "请确认「提示词装配」节点的输出接到了渲染节点（而不是「分镜清单加载」的原始输出）。"
        )

    ref_images, ref_videos = _normalize_media_refs(
        ref_images or {}, ref_videos or {}
    )

    try:
        out = cls.execute(
            clip=clip,
            prompt=prompt,
            width=int(width),
            height=int(height),
            length=int(length),
            ref_image_size=ref_image_size,
            vae=vae,
            audio_vae=audio_vae,
            ref_images=ref_images or {},
            ref_videos=ref_videos or {},
            ref_video_audios=ref_video_audios or {},
            ref_audios=ref_audios or {},
        )
    except H3ApiError:
        raise
    except Exception as e:
        shapes = ", ".join(
            f"{k}={tuple(getattr(v, 'shape', ()))}" for k, v in ref_images.items()
        ) or "（无参考图）"
        raise H3ApiError(
            f"官方 MiniMaxH3ReferenceToVideo 执行失败：{type(e).__name__}: {e}\n"
            f"常见原因：参考图数量超限（图≤9/视频≤3/音频≤3）、"
            f"帧数未按 17n+5 对齐、分辨率不是 32 的倍数。\n"
            f"当前参数：{width}x{height}, length={length}, "
            f"ref_images={len(ref_images)}\n"
            f"参考图张量：{shapes}\n"
            f"若报错里出现 `shape '[...]' is invalid for input of size ...`，"
            f"那是**张量维度**问题（官方要求 4 维 [B,H,W,C]）—— 本版本已在出站前自动规整，"
            f"若仍出现请把该图的形状贴出来。"
        ) from e
    res = _result_of(out)
    if len(res) < 2:
        raise H3ApiError(f"官方 R2V 返回了 {len(res)} 个结果，期望 2 个（positive, LATENT）")
    return res[0], res[1]


def encode_image_to_video(
    clip: Any,
    vae: Any,
    prompt: str,
    width: int,
    height: int,
    length: int,
    first_frame: Any = None,
    last_frame: Any = None,
) -> Tuple[Any, Any]:
    """包装官方 ``MiniMaxH3ImageToVideo``（t2va / fl2va）。"""
    from .image_ops import as_ref_image

    cls = get_h3_classes(("MiniMaxH3ImageToVideo",))["MiniMaxH3ImageToVideo"]
    # 首/尾帧与参考图走同一套维度契约（官方同样按 [B,H,W,C] 读 shape[1]/shape[2]）
    if first_frame is not None:
        first_frame = as_ref_image(first_frame, None, "首帧")
    if last_frame is not None:
        last_frame = as_ref_image(last_frame, None, "尾帧")
    try:
        out = cls.execute(
            clip=clip, vae=vae, prompt=prompt,
            width=int(width), height=int(height), length=int(length),
            first_frame=first_frame, last_frame=last_frame,
        )
    except Exception as e:
        raise H3ApiError(f"官方 MiniMaxH3ImageToVideo 执行失败：{type(e).__name__}: {e}") from e
    res = _result_of(out)
    return res[0], res[1]


def add_guide(
    positive: Any,
    latent: Any,
    frame_idx: int = 0,
    vae: Any = None,
    audio_vae: Any = None,
    image: Any = None,
    audio: Any = None,
) -> Any:
    """包装官方 ``MiniMaxH3AddGuide``：在任意帧锚定一张图 / 一段音频。"""
    cls = get_h3_classes(("MiniMaxH3AddGuide",))["MiniMaxH3AddGuide"]
    try:
        out = cls.execute(
            positive=positive, latent=latent, frame_idx=int(frame_idx),
            vae=vae, audio_vae=audio_vae, image=image, audio=audio,
        )
    except Exception as e:
        raise H3ApiError(f"官方 MiniMaxH3AddGuide 执行失败：{type(e).__name__}: {e}") from e
    return _result_of(out)[0]


# --------------------------------------------------------------------------
# 模型补丁
# --------------------------------------------------------------------------
def apply_sigma_shift(model: Any, shift_video: float = 12.0, shift_audio: float = 3.0) -> Any:
    """包装官方 ``MiniMaxH3SigmaShift``（display_name: ModelSamplingMiniMaxH3）。"""
    cls = get_h3_classes(("MiniMaxH3SigmaShift",))["MiniMaxH3SigmaShift"]
    try:
        out = cls.execute(model=model, shift_video=float(shift_video), shift_audio=float(shift_audio))
    except Exception as e:
        raise H3ApiError(f"官方 MiniMaxH3SigmaShift 执行失败：{type(e).__name__}: {e}") from e
    return _result_of(out)[0]


def apply_lora(model: Any, lora_name: str, strength: float = 1.0) -> Any:
    """包装内置 ``LoraLoaderModelOnly``。lora_name 为空时原样返回模型。"""
    if not lora_name or lora_name in ("none", "None", ""):
        return model
    import nodes as comfy_nodes
    try:
        node = comfy_nodes.LoraLoaderModelOnly()
        out = node.load_lora_model_only(model, lora_name, float(strength))
    except Exception as e:
        raise H3ApiError(
            f"加载 LoRA「{lora_name}」失败：{type(e).__name__}: {e}\n"
            f"请确认 models/loras/ 下存在该文件。"
        ) from e
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------
# 稀疏注意力（可选加速，包装官方 BlockSparseAttention）
# --------------------------------------------------------------------------
SPARSE_METHODS = ("none", "sol-attn", "sla", "vsa")
SPARSE_SINK_MODES = ("exact_kv_and_rows", "exact_kv", "off")

# 官方节点里 sol-attn 的 tau 与保留比例大致对应关系（来自其 tooltip）：
#   tau=1.0 保留 ~16% key block；1.5 ~7%；2.0 ~2.7%
SPARSE_DEFAULT_TAU = 1.3
SPARSE_DEFAULT_KEEP = 10.0
SPARSE_DEFAULT_MIN_TOKENS = 12288


def sparse_capability() -> Dict[str, Any]:
    """探测稀疏注意力在本机是否可用。

    返回 ``{"available": bool, "kernel": bool|None, "reason": str}``。

    ``kernel`` 指 comfy_kitchen 的 ``sol_attn`` 内核 ——
    官方 ``h3_eligible()`` 里**连 vsa 模式也要先过这道门**，
    没有它稀疏路径会整体退化为 dense。
    """
    try:
        from comfy_extras import nodes_sparse_attention as nsa  # 延迟导入
    except Exception as e:  # noqa: BLE001
        return {"available": False, "kernel": None,
                "reason": f"导入 nodes_sparse_attention 失败：{type(e).__name__}: {e}"}

    if getattr(nsa, "BlockSparseAttention", None) is None:
        return {"available": False, "kernel": None,
                "reason": "官方没有 BlockSparseAttention 节点（需要 ComfyUI ≥ 0.37）"}

    kernel: Optional[bool] = None
    try:
        import comfy_kitchen as ck
        import torch as _t
        kernel = bool(ck.sol_attn_is_available(_t.device("cuda")))
    except Exception:  # noqa: BLE001
        kernel = None
    return {"available": True, "kernel": kernel, "reason": ""}


def apply_sparse_attention(
    model: Any,
    method: str = "none",
    tau: float = SPARSE_DEFAULT_TAU,
    keep_percent: float = SPARSE_DEFAULT_KEEP,
    start_percent: float = 0.2,
    min_tokens: int = SPARSE_DEFAULT_MIN_TOKENS,
    sink_conditioning: str = "exact_kv_and_rows",
    extra_tokens: int = 256,
    verbose: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """包装官方 ``BlockSparseAttention``（category model/patch）。返回 ``(model, info)``。

    **加速失败绝不能让渲染挂掉** —— 任何异常都收敛进 ``info["reason"]`` 并原样返回 model。
    这与 ``apply_sigma_shift`` 那种「失败即抛错」的策略刻意不同：稀疏注意力是可选优化，
    不是正确性依赖。

    ``sink_conditioning`` 对**本插件特别重要**：官方默认 ``exact_kv_and_rows``
    会让每个 query 精确注意到 packed 序列里的文本 / 音频 / **参考图**行
    （官方注释：约 3% 额外开销）。也就是说**角色参考图不会被稀疏化** ——
    这正是我们敢在漫改链路上开稀疏的原因。改成 ``off`` 会更快，但参考图影响会变弱。
    """
    info: Dict[str, Any] = {
        "applied": False, "method": method, "kernel": None, "reason": "",
        "tau": float(tau), "keep_percent": float(keep_percent),
        "start_percent": float(start_percent), "min_tokens": int(min_tokens),
        "sink_conditioning": sink_conditioning,
    }

    if not method or method == "none":
        info["reason"] = "未启用（method=none）"
        return model, info

    if method not in SPARSE_METHODS:
        info["reason"] = f"未知的稀疏方式「{method}」，可选：{', '.join(SPARSE_METHODS)}"
        return model, info

    cap = sparse_capability()
    info["kernel"] = cap.get("kernel")
    if not cap["available"]:
        info["reason"] = f"官方 BlockSparseAttention 不可用：{cap['reason']}"
        return model, info
    if cap.get("kernel") is False:
        info["reason"] = ("本机没有可用的 sol_attn 内核（稀疏路径会退化为 dense）—— "
                          "已原样返回模型")
        return model, info

    try:
        from comfy_extras import nodes_sparse_attention as nsa
        cls = nsa.BlockSparseAttention
    except Exception as e:  # noqa: BLE001
        info["reason"] = f"取不到 BlockSparseAttention：{type(e).__name__}: {e}"
        return model, info

    # 官方 selection 是 DynamicCombo：sol-attn 用 tau，sla/vsa 用 keep_percent
    if method == "sol-attn":
        selection: Dict[str, Any] = {"selection": "sol-attn", "tau": float(tau)}
    else:
        selection = {"selection": method, "keep_percent": float(keep_percent)}

    try:
        out = cls.execute(
            model=model, selection=selection,
            start_percent=float(start_percent), end_percent=1.0,
            dense_blocks="", min_tokens=int(min_tokens),
            extra_tokens=int(extra_tokens),
            sink_conditioning=sink_conditioning, verbose=bool(verbose),
        )
    except Exception as e:  # noqa: BLE001
        info["reason"] = f"官方 BlockSparseAttention 执行失败：{type(e).__name__}: {e}"
        return model, info

    res = _result_of(out)
    if not res or res[0] is None:
        info["reason"] = "官方 BlockSparseAttention 没有返回模型"
        return model, info

    info["applied"] = True
    info["reason"] = "已应用"
    return res[0], info


# --------------------------------------------------------------------------
# 采样（官方五件套）
# --------------------------------------------------------------------------
def sample(
    model: Any,
    positive: Any,
    latent: Any,
    sampler_name: str = "res_multistep",
    scheduler: str = "simple",
    steps: int = 20,
    denoise: float = 1.0,
    seed: int = 0,
    progress: Any = None,
) -> Any:
    """跑官方采样链：KSamplerSelect + BasicScheduler + RandomNoise + BasicGuider
    → SamplerCustomAdvanced。返回 LATENT（dict，samples 是 NestedTensor）。

    **不复刻内部逻辑**，逐字调官方 execute()，这样 ComfyUI 升级时自动跟随。
    """
    from comfy_extras import nodes_custom_sampler as ncs

    try:
        noise = _result_of(ncs.RandomNoise.execute(noise_seed=int(seed) & 0xFFFFFFFFFFFFFFFF))[0]
        sampler = _result_of(ncs.KSamplerSelect.execute(sampler_name=sampler_name))[0]
        sigmas = _result_of(ncs.BasicScheduler.execute(
            model=model, scheduler=scheduler, steps=int(steps), denoise=float(denoise)))[0]
        guider = _result_of(ncs.BasicGuider.execute(model=model, conditioning=positive))[0]
        out = _result_of(ncs.SamplerCustomAdvanced.execute(
            noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent))
    except Exception as e:
        raise H3ApiError(
            f"官方采样链执行失败：{type(e).__name__}: {e}\n"
            f"参数：sampler={sampler_name}, scheduler={scheduler}, steps={steps}, "
            f"denoise={denoise}, seed={seed}"
        ) from e

    return out[0]


def decode_video(vae: Any, latent: Any):
    """包装内置 ``VAEDecode``（它内部对 NestedTensor 取 unbind()[0]，即视频流）。"""
    import nodes as comfy_nodes
    try:
        return comfy_nodes.VAEDecode().decode(vae, latent)[0]
    except Exception as e:
        raise H3ApiError(f"VAEDecode 失败：{type(e).__name__}: {e}") from e


def decode_audio(audio_vae: Any, latent: Any):
    """包装官方 ``VAEDecodeAudio``。"""
    from comfy_extras import nodes_audio as na
    try:
        return _result_of(na.VAEDecodeAudio.execute(vae=audio_vae, samples=latent))[0]
    except Exception as e:
        raise H3ApiError(f"VAEDecodeAudio 失败：{type(e).__name__}: {e}") from e


# --------------------------------------------------------------------------
# 能力自检（节点 INPUT_TYPES / 启动时可调）
# --------------------------------------------------------------------------
def capability_report() -> Dict[str, Any]:
    """探测本机 H3 官方能力是否齐备。**不抛异常**，返回结构化结果。"""
    report: Dict[str, Any] = {"ok": True, "missing": [], "found": []}
    checks = {
        "MiniMaxH3ReferenceToVideo": "参考生视频 R2V（漫改动画主链路）",
        "MiniMaxH3ImageToVideo": "图生视频 I2V / 首尾帧",
        "MiniMaxH3SigmaShift": "视频/音频 shift 调优",
        "MiniMaxH3AddGuide": "任意帧锚定（可选）",
    }
    try:
        from comfy_extras import nodes_minimax_h3 as h3
    except Exception as e:
        report["ok"] = False
        report["missing"] = list(checks.keys())
        report["error"] = str(e)
        return report

    for name, desc in checks.items():
        if hasattr(h3, name):
            report["found"].append({"name": name, "desc": desc})
        else:
            report["missing"].append(name)

    try:
        from comfy_extras import nodes_custom_sampler as ncs
        for n in ("KSamplerSelect", "BasicScheduler", "RandomNoise", "BasicGuider",
                  "SamplerCustomAdvanced"):
            if not hasattr(ncs, n):
                report["missing"].append(n)
    except Exception as e:
        report["missing"].append(f"nodes_custom_sampler({e})")

    # 稀疏注意力（可选加速，不影响 ok —— 它是优化不是依赖）
    try:
        report["sparse_attention"] = sparse_capability()
    except Exception as e:  # noqa: BLE001
        report["sparse_attention"] = {
            "available": False, "kernel": None,
            "reason": f"{type(e).__name__}: {e}",
        }

    report["ok"] = len([m for m in report["missing"] if "MiniMaxH3" in m]) == 0
    return report
