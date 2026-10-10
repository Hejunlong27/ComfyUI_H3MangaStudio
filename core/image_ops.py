"""图像 / 音频张量拼接工具。

**不 import comfy**，只依赖 torch —— 所以能脱离 ComfyUI 用真 torch 单测覆盖。
（这类代码最容易出静默错：形状对了、内容错位，肉眼在成品上很难发现。）

为什么要有这个模块
------------------
早先 ``_pad_stack``（切格节点）与 ``_concat_frames``（整集渲染节点）
各写了一遍「尺寸不一 → pad 到最大 → 拼批」，两处都可能漏 ``device=``，
在 ``--gpu-only`` 下（VAE 输出留在 GPU）会 device mismatch。
现在收口成一处，只需要修一个地方。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def empty_image_batch(h: int = 64, w: int = 64, c: int = 3):
    """空 IMAGE 批次（下游 SaveImage/CreateVideo 至少不会拿到 None）。"""
    if not HAS_TORCH:
        return None
    return torch.zeros((0, h, w, c), dtype=torch.float32)


# --------------------------------------------------------------------------
# ★ 官方节点的「参考素材张量契约」—— 本插件踩过一次最贵的线上事故
# --------------------------------------------------------------------------
REF_CHANNELS = 3


def _shape_of(t):
    try:
        return tuple(int(v) for v in t.shape)
    except Exception:
        return None


def _expand_channels(x, n: int):
    """把单通道图扩成 n 通道。torch 与 numpy 都吃 ——

    ``np.ndarray.repeat(3)`` 不带 axis 会**展平**整个数组（而 ``torch.repeat(1,1,1,3)``
    不会），这类差异在纯逻辑测试里会变成「本地过了、ComfyUI 里炸」。
    所以统一走高级索引（两个库语义一致）。
    """
    n = int(n)
    try:
        return x[..., [0] * n]
    except Exception:  # pragma: no cover
        return x.repeat(1, 1, 1, n)


def as_ref_image(t, notes: Optional[List[str]] = None, where: str = ""):
    """把一张参考图规整成官方 ``MiniMaxH3ReferenceToVideo`` 要求的 4 维批次 ``[1, H, W, C]``。

    为什么必须做这一步（真实事故复盘）
    ----------------------------------
    官方节点读尺寸的写法是::

        h, w = img.shape[1], img.shape[2]        # 它是按 [B, H, W, C] 读的

    而本插件内部统一用 3 维 ``[H, W, C]``（「一个槽位 = 一张图」）。
    3 维直接喂进去，h 会取到**图宽**、w 会取到**通道数 3**：

    1. 于是参考图被 resize 成畸形尺寸 —— 真实案例：漫画页宽 1152 的分镜格被算成
       ``th=1152, tw=32``（因为 ``round(3/32)*32`` 落到 0，被 ``max(32, ...)`` 兜成 32）；
    2. ``comfy.utils.common_upscale`` 因此走进 ``len(orig_shape) != 4`` 分支，
       在最后一行 ``out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))``
       抛 ``RuntimeError: shape '[1, -1, 3, 1152, 32]' is invalid for input of size 36864``。

    这句报错文本**完全看不出是形状问题**（既不是 3 维也不是 [B,H,W,C]），
    于是会把人带去查帧数对齐、分辨率倍数、参考图数量 —— 全是错的。
    所以在这里一次性升维，并且把「非例行修正」变成可见警告。

    归一化规则
    ----------
    * ``[B,H,W,C]`` 4 维   → 取第 1 张（一个槽位 = 一张图，否则 ``<Picture i>`` 会错位）
    * ``[H,W,C]`` 3 维     → 补 batch 维（**插件的常规路径，不告警**）
    * ``[H,W]`` 2 维       → 补 batch + 复制成 3 通道（灰度）
    * 5 维及以上           → 先剥掉值为 1 的前导维；剥不动就报错
    * 通道维只接受 1 / 3 / 4（1 → 复制、4 → 丢 alpha）；其它值报错 ——
      与其静默把 ``[24,48,84]`` 这种潜张量当图片塞进模型，不如当场说清楚。

    ``notes`` 传一个 list 时，会把「做了什么修正」追加进去（供 warnings 用）。
    """
    if t is None:
        return None

    shape = _shape_of(t)
    if shape is None:
        if notes is not None:
            notes.append(f"{where}：参考图不是张量（{type(t).__name__}），已丢弃")
        return None

    x = t
    # 剥掉值为 1 的前导维（[1,1,H,W,C] 这种误加维很常见）
    while len(shape) > 4 and shape[0] == 1:
        x = x[0]
        shape = _shape_of(x)

    if len(shape) == 2:
        x = _expand_channels(x[:, :, None], REF_CHANNELS)
        if notes is not None:
            notes.append(f"{where}：参考图是 2 维 {shape}（灰度），已按 3 通道补齐")
        shape = _shape_of(x)
    elif len(shape) not in (3, 4):
        raise ValueError(
            f"参考图维度不支持：{shape}。官方 MiniMaxH3ReferenceToVideo 只接受 "
            f"[H,W,C] 或 [B,H,W,C]（4 维 [B,H,W,C] 是最终交付形态）。"
        )

    if len(shape) == 3:                 # 常规路径：插件内部就是 3 维，不告警
        x = x[None]
        shape = _shape_of(x)

    c = shape[-1]
    if c not in (1, REF_CHANNELS, 4):
        raise ValueError(
            f"参考图最后一维是 {c}，不像通道数（{shape}）。"
            f"官方要求 [B,H,W,C] 且 C={REF_CHANNELS}。"
        )
    x, shape = _fix_channels(x, shape, notes, where)

    if shape[0] != 1:
        x = x[:1]
        if notes is not None:
            notes.append(
                f"{where}：一个槽位收到 {shape[0]} 张图，只取第 1 张"
                f"（一个槽位 = 一张图，否则 <Picture i> 序号会整体错位）"
            )
    return x


def as_ref_video(t, notes: Optional[List[str]] = None, where: str = ""):
    """参考视频槽位：官方要求 ``[T, H, W, C]``（T = 帧数，≥5 帧）。

    与参考图**共享通道规则**，但 **T 维绝不能当成 batch 截掉** ——
    视频的第 0 维是帧数，截成 1 帧会让官方直接报「至少 5 帧」。
    另外 3 维且最后一维不像通道数时，按「无通道维的帧序列 ``[T,H,W]``」处理并补 3 通道。
    """
    if t is None:
        return None
    shape = _shape_of(t)
    if shape is None:
        if notes is not None:
            notes.append(f"{where}：参考视频不是张量（{type(t).__name__}），已丢弃")
        return None

    x = t
    while len(shape) > 4 and shape[0] == 1:      # 剥掉值为 1 的前导维
        x = x[0]
        shape = _shape_of(x)

    if len(shape) == 3:
        if shape[-1] not in (1, REF_CHANNELS, 4):
            if notes is not None:
                notes.append(f"{where}：参考视频形状 {shape} 看着是 [T,H,W]（无通道维），已补 3 通道")
            x = _expand_channels(x[..., None], REF_CHANNELS)
        else:
            if notes is not None:
                notes.append(f"{where}：参考视频只有 1 帧 {shape} —— 官方要求 ≥5 帧")
            x = x[None]
        shape = _shape_of(x)

    if len(shape) != 4:
        raise ValueError(
            f"参考视频维度不支持：{shape}。官方 MiniMaxH3ReferenceToVideo 要求 [T,H,W,C]"
            f"（一个槽位 = 一段多帧视频，T≥5）。"
        )

    x, shape = _fix_channels(x, shape, notes, where)
    return x


def _fix_channels(x, shape, notes, where):
    """把最后一维（通道）规整成 3。返回 (张量, 新形状)。"""
    c = shape[-1]
    if c == 1:
        x = _expand_channels(x, REF_CHANNELS)
        if notes is not None:
            notes.append(f"{where}：参考图/视频只有 1 个通道，已复制成 {REF_CHANNELS} 通道")
    elif c == 4:
        x = x[..., :REF_CHANNELS]
        if notes is not None:
            notes.append(f"{where}：参考图/视频有 4 个通道（含 alpha），已丢弃 alpha")
    elif c != REF_CHANNELS:
        raise ValueError(
            f"参考图/视频最后一维是 {c}，不像通道数（{shape}）。"
            f"官方要求 [B,H,W,C] / [T,H,W,C] 且 C={REF_CHANNELS}。"
        )
    return x, _shape_of(x)


def pad_spatial(t, max_h: int, max_w: int):
    """把 ``[..., H, W, C]`` 的最后两个空间维 pad 到 (max_h, max_w)。

    ``device`` 与 ``dtype`` 都跟着源张量走 —— 这是关键：
    早先版本只写了 ``dtype``，在 GPU 上会把 CPU 的零张量和 CUDA 张量相加。
    """
    h, w = int(t.shape[-3]), int(t.shape[-2])
    if h == max_h and w == max_w:
        return t
    shape = list(t.shape[:-3]) + [max_h, max_w, int(t.shape[-1])]
    canvas = torch.zeros(shape, dtype=t.dtype, device=t.device)
    canvas[..., :h, :w, :] = t
    return canvas


def _max_hw(tensors: Sequence[Any]):
    return (max(int(t.shape[-3]) for t in tensors),
            max(int(t.shape[-2]) for t in tensors))


def stack_images(tensors: Sequence[Any]):
    """把一组 ``[H, W, C]`` 堆成 ``[B, maxH, maxW, C]``（尺寸不一自动 pad）。"""
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return empty_image_batch()
    max_h, max_w = _max_hw(tensors)
    if len(tensors) == 1:
        return pad_spatial(tensors[0], max_h, max_w)[None, ...]
    return torch.stack([pad_spatial(t, max_h, max_w) for t in tensors], dim=0)


def concat_frames(frames_list: Sequence[Any]):
    """把多镜的 ``[T, H, W, C]`` 帧序列沿时间轴拼成 ``[ΣT, maxH, maxW, C]``。

    尺寸不一致时 pad 到最大（各镜原始尺寸已在渲染报告里记录，需要时可按帧切回）。
    """
    frames_list = [f for f in frames_list if f is not None]
    if not frames_list:
        return empty_image_batch()
    max_h, max_w = _max_hw(frames_list)
    if len(frames_list) == 1:
        return pad_spatial(frames_list[0], max_h, max_w)
    return torch.cat([pad_spatial(f, max_h, max_w) for f in frames_list], dim=0)


def concat_audio(waveforms: Sequence[Any], sample_rate: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """把多镜音轨按时间顺序拼成一条 ``{"waveform": [B, C, L], "sample_rate": int}``。

    失败返回 ``None``，并把原因通过返回值里的 ``"error"`` 带出去 ——
    调用方负责记日志。早先版本在这里裸 ``except`` 直接 return None，
    会**静默丢掉整条音轨**，用户只知道「视频没声音」。
    """
    waveforms = [w for w in waveforms if w is not None]
    if not waveforms:
        return None
    try:
        norm: List[Any] = []
        for w in waveforms:
            if w.dim() == 3:
                norm.append(w)
            elif w.dim() == 2:
                norm.append(w.unsqueeze(0))
            elif w.dim() == 1:
                norm.append(w.view(1, 1, -1))
            else:
                return {"waveform": None, "sample_rate": int(sample_rate or 32000),
                        "error": f"不支持的音轨维度：{w.dim()}（期望 1~3）"}
        if not norm:
            return None

        max_c = max(int(w.shape[1]) for w in norm)
        fixed = []
        for w in norm:
            if int(w.shape[1]) < max_c:
                pad = torch.zeros((int(w.shape[0]), max_c - int(w.shape[1]), int(w.shape[2])),
                                  dtype=w.dtype, device=w.device)
                w = torch.cat([w, pad], dim=1)
            fixed.append(w)
        return {"waveform": torch.cat(fixed, dim=-1),
                "sample_rate": int(sample_rate or 32000)}
    except Exception as e:  # pragma: no cover
        return {"waveform": None, "sample_rate": int(sample_rate or 32000),
                "error": f"{type(e).__name__}: {e}"}
