"""显存 / 系统内存管理：快照 + 卸载原语。

为什么需要这个模块
------------------
本机实测：H3 的 R2V UNET **19.53 GB** + 文本编码器 **13.93 GB** = 权重总量 33.5 GB，
而物理内存只有 **16 GB**、显存只有 **8 GB**。若三个阶段（编码 / 采样 / 解码）
的模型同时驻留，内存必然耗尽 —— 表现为 ComfyUI 被 OOM Killer 杀掉后自动重启，
日志里连一条 Python traceback 都没有。

隔离手段是**串行化**：用完一个模型立刻把它从显存与内存里卸掉，
下一个模型再加载。本模块提供这套卸载原语。

为什么卸载是安全的（已核实源码）
--------------------------------
``comfy/sd.py``：

* ``CLIP`` 有 ``self.patcher``（ModelPatcher），``VAE`` 也有 ``self.patcher``（L1120）
* ``CLIP.encode_from_tokens`` → ``self.load_model(tokens)`` → ``load_models_gpu([self.patcher])``
* ``VAE.decode`` 各分支同样先 ``load_models_gpu([self.patcher])``

也就是说 **卸载之后下次再用会自动重新加载**，不会报「权重在 meta 设备上」。
所以「分阶段卸载」在 ComfyUI 的口径下是安全的。

设计约束
--------
* **绝不抛异常**。内存管理失败不能把渲染搞挂 —— 所有失败都收敛进返回的 dict。
* **comfy 延迟导入**。本模块可以在没有 ComfyUI 的环境里被 import（纯逻辑测试）。
* 只依赖标准库 + 可选 torch；RAM 读数是 Windows/Linux 双实现。
"""

from __future__ import annotations

import ctypes
import gc
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

GIB = 1024 ** 3
MIB = 1024 ** 2


# --------------------------------------------------------------------------
# 环境探测
# --------------------------------------------------------------------------
def _torch():
    """拿到 torch；拿不到返回 None（不抛）。"""
    try:
        import torch  # noqa: WPS433

        return torch
    except Exception:  # noqa: BLE001
        return None


def _mm():
    """拿到 comfy.model_management；拿不到返回 None（不抛）。

    单元测试里没有真实的 comfy 包，这里会返回 None，
    调用方据此降级 —— 这正是本模块「绝不抛异常」的第一道保障。
    """
    try:
        import comfy.model_management as mm  # noqa: WPS433

        return mm
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# 快照
# --------------------------------------------------------------------------
def vram_snapshot() -> Dict[str, float]:
    """显存快照（MiB）。无 CUDA / 无 torch 时返回 {}。"""
    torch = _torch()
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        free_b, total_b = torch.cuda.mem_get_info()
        out = {
            "free_mib": free_b / MIB,
            "total_mib": total_b / MIB,
            "used_mib": (total_b - free_b) / MIB,
        }
        try:
            out["allocated_mib"] = torch.cuda.memory_allocated() / MIB
            out["reserved_mib"] = torch.cuda.memory_reserved() / MIB
            out["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception:  # noqa: BLE001
        return {}


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def ram_snapshot() -> Dict[str, float]:
    """系统内存快照（GB）。Windows 用 GlobalMemoryStatusEx，Linux 读 /proc/meminfo。"""
    if os.name == "nt":
        try:
            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return {
                    "used_gb": (st.ullTotalPhys - st.ullAvailPhys) / GIB,
                    "total_gb": st.ullTotalPhys / GIB,
                    "pct": float(st.dwMemoryLoad),
                    "free_gb": st.ullAvailPhys / GIB,
                    "pagefile_free_gb": st.ullAvailPageFile / GIB,
                    "pagefile_total_gb": st.ullTotalPageFile / GIB,
                }
        except Exception:  # noqa: BLE001
            return {}

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            info = {}
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.split()[0]) * 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        if not total:
            return {}
        return {
            "used_gb": (total - avail) / GIB,
            "total_gb": total / GIB,
            "pct": (total - avail) / total * 100.0,
            "free_gb": avail / GIB,
        }
    except Exception:  # noqa: BLE001
        return {}


def snapshot() -> Dict[str, Any]:
    """一次拿全：{'vram': {...}, 'ram': {...}}。"""
    return {"vram": vram_snapshot(), "ram": ram_snapshot()}


# --------------------------------------------------------------------------
# 对象 → ModelPatcher
# --------------------------------------------------------------------------
def patcher_of(obj: Any) -> Optional[Any]:
    """从 MODEL / CLIP / VAE 里取出 ModelPatcher。

    * MODEL 本身就是 ModelPatcher
    * CLIP / VAE 有 ``.patcher``
    拿不到返回 None（例如测试里传的是字符串桩）。
    """
    if obj is None:
        return None
    # CLIP / VAE
    p = getattr(obj, "patcher", None)
    if p is not None and hasattr(p, "model_unload"):
        return p
    # MODEL（ModelPatcher 自身）
    if hasattr(obj, "model_unload") and hasattr(obj, "model"):
        return obj
    return None


def resolve_patchers(objs: Sequence[Any]) -> Tuple[List[Any], List[str]]:
    """把一堆输入对象解析成 patcher 列表，同时收集无法解析的名字。"""
    found: List[Any] = []
    unknown: List[str] = []
    for o in objs or ():
        if o is None:
            continue
        p = patcher_of(o)
        if p is None:
            unknown.append(type(o).__name__)
        elif not any(p is f for f in found):
            found.append(p)
    return found, unknown


def _patchers_match(targets: Sequence[Any], patcher: Any) -> bool:
    """判断一个已加载的 patcher 是否属于要卸载的目标。

    优先用 ``clone_base_uuid`` —— LoRA / sigma-shift 会 clone 出新的
    ModelPatcher，只有 uuid 能把它和原始模型关联起来；退化到身份比较。
    """
    for t in targets:
        if t is patcher:
            return True
        tu = getattr(t, "clone_base_uuid", None)
        pu = getattr(patcher, "clone_base_uuid", None)
        if tu is not None and pu is not None and tu == pu:
            return True
    return False


# --------------------------------------------------------------------------
# 卸载
# --------------------------------------------------------------------------
def unload(
    objs: Sequence[Any] = (),
    unload_all: bool = False,
    run_gc: bool = True,
    label: str = "",
) -> Dict[str, Any]:
    """卸载模型。**绝不抛异常**。

    参数
    ----
    objs
        要卸载的对象（MODEL / CLIP / VAE 都行）。为空且 ``unload_all=False``
        时什么都不做 —— 这是刻意的：避免「参数传错」变成「把所有模型都卸了」。
    unload_all
        True = 卸载**全部**已加载模型（等价官方 ``unload_all_models()``）。
    run_gc
        是否额外 ``gc.collect()``。释放 Python 侧引用，让 mmap 的权重页可被回收。
    label
        仅用于报告里标明这是流水线的哪一步。

    返回
    ----
    ``{"ok", "mode", "label", "targets", "unknown", "freed", "error",
       "vram_before", "vram_after", "ram_before", "ram_after"}``
    """
    before = snapshot()
    out: Dict[str, Any] = {
        "ok": False,
        "mode": "none",
        "label": label,
        "targets": [],
        "unknown": [],
        "api_errors": [],
        "vram_before": before.get("vram") or {},
        "vram_after": {},
        "ram_before": before.get("ram") or {},
        "ram_after": {},
    }

    targets, unknown = resolve_patchers(objs)
    out["unknown"] = unknown
    out["targets"] = [type(t.model).__name__ if getattr(t, "model", None) is not None
                      else type(t).__name__ for t in targets]

    mm = _mm()

    # mode 反映的是「打算做什么」，不因环境缺 comfy 而改变 ——
    # 否则报告会说不清「本来要卸什么、为什么没卸成」。
    if unload_all:
        out["mode"] = "all"
    elif targets:
        out["mode"] = "selected"
    else:
        out["mode"] = "none"
        out["api_errors"].append("没有可解析的模型对象，未执行卸载")

    if mm is None:
        out["api_errors"].append("comfy.model_management 不可用（非 ComfyUI 环境？）—— 已跳过卸载")
    elif out["mode"] == "all":
        try:
            mm.unload_all_models()
            out["ok"] = True
        except Exception as e:  # noqa: BLE001
            out["api_errors"].append(f"unload_all_models: {type(e).__name__}: {e}")
    elif out["mode"] == "selected":
        try:
            devices = _all_devices(mm)
            # keep_loaded 收的是 current_loaded_models 里的条目，不是 patcher
            loaded = list(getattr(mm, "current_loaded_models", []) or [])
            keep = [lm for lm in loaded
                    if getattr(lm, "model", None) is not None
                    and not _patchers_match(targets, lm.model)]
            for dev in devices:
                # memory_required 给超大值 = 「把该设备上不在 keep 里的都卸掉」
                mm.free_memory(1e30, dev, keep_loaded=keep)
            out["ok"] = True
        except Exception as e:  # noqa: BLE001
            out["api_errors"].append(f"free_memory: {type(e).__name__}: {e}")

    if mm is not None:
        try:
            mm.soft_empty_cache(force=True)
        except Exception as e:  # noqa: BLE001
            out["api_errors"].append(f"soft_empty_cache: {type(e).__name__}: {e}")

    if run_gc:
        try:
            gc.collect()
        except Exception:  # noqa: BLE001
            pass

    after = snapshot()
    out["vram_after"] = after.get("vram") or {}
    out["ram_after"] = after.get("ram") or {}
    return out


def _all_devices(mm: Any) -> List[Any]:
    """尽力拿到全部 torch 设备；拿不到就退化成 [None]（free_memory 支持 device=None）。"""
    try:
        devs = list(mm.get_all_torch_devices())
        return devs if devs else [None]
    except Exception:  # noqa: BLE001
        return [None]


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def _fmt_vram(v: Dict[str, float]) -> str:
    if not v:
        return "无 CUDA"
    used = v.get("used_mib")
    total = v.get("total_mib")
    free = v.get("free_mib")
    if used is None or total is None:
        return "无 CUDA"
    return f"{used:.0f}/{total:.0f} MiB 用（空闲 {free:.0f}）"


def _fmt_ram(r: Dict[str, float]) -> str:
    if not r:
        return "读数不可用"
    s = f"{r.get('used_gb', 0):.2f}/{r.get('total_gb', 0):.2f} GB（{r.get('pct', 0):.0f}%）"
    if "pagefile_free_gb" in r:
        s += f" 页文件余 {r['pagefile_free_gb']:.2f} GB"
    return s


def format_stage_line(label: str, snap: Dict[str, Any]) -> str:
    """把一次 snapshot() 渲染成一行：用于流水线报告。"""
    return (f"{label:<22} 显存 {_fmt_vram(snap.get('vram') or {}):<34} "
            f"内存 {_fmt_ram(snap.get('ram') or {})}")


def format_unload_report(res: Dict[str, Any]) -> str:
    """把 unload() 的返回值渲染成人类可读报告。"""
    lines = [
        f"显存/内存释放报告{('：' + res['label']) if res.get('label') else ''}",
        "=" * 52,
        f"  模式      : {res.get('mode')}",
        f"  目标模型  : {', '.join(res.get('targets') or []) or '（无）'}",
    ]
    if res.get("unknown"):
        lines.append(f"  无法识别  : {', '.join(res['unknown'])}（已跳过）")
    lines.append(f"  释放前    : {_fmt_vram(res.get('vram_before') or {})}")
    lines.append(f"  释放后    : {_fmt_vram(res.get('vram_after') or {})}")

    vb = (res.get("vram_before") or {}).get("used_mib")
    va = (res.get("vram_after") or {}).get("used_mib")
    if vb is not None and va is not None:
        lines.append(f"  显存回收  : {vb - va:+.0f} MiB")

    rb = (res.get("ram_before") or {}).get("used_gb")
    ra = (res.get("ram_after") or {}).get("used_gb")
    lines.append(f"  内存      : {_fmt_ram(res.get('ram_before') or {})}"
                 f"  ->  {_fmt_ram(res.get('ram_after') or {})}")
    if rb is not None and ra is not None:
        lines.append(f"  内存回收  : {rb - ra:+.2f} GB")

    if res.get("api_errors"):
        lines.append("  提示      :")
        for e in res["api_errors"]:
            lines.append(f"    ! {e}")
    if not res.get("ok"):
        lines.append("  结果      : 未实际卸载（原因见上）")
    else:
        lines.append("  结果      : 已执行卸载")
    return "\n".join(lines)
