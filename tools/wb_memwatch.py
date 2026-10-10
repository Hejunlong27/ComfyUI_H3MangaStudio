"""H3MangaStudio 资源监视器 —— 显存 / 系统内存 / 页面文件 / 进程 双曲线采样。

为什么需要它
------------
"ComfyUI 自动重启"有三种可能，必须分开，否则优化方向会打偏：

  1. 纯 VRAM 不足        -> 日志里有 torch.cuda.OutOfMemoryError，**进程不重启**
  2. 系统 RAM 耗尽       -> 被 Windows OOM Killer 杀进程，**没有 Python traceback**
  3. CUDA 驱动级失败     -> 日志里有 CUDA error: illegal memory access / device lost

本工具同时采 GPU 与 RAM 两条曲线，正是区分这三者的唯一手段。
本机实测：H3 的 R2V UNET 19.53GB + 文本编码器 13.93GB = 33.5GB 权重，
而物理内存只有 16GB —— 案例 2 的概率很高。

用法
----
  # 单次快照（看当前状态）
  python wb_memwatch.py

  # 采样 600 秒（跑工作流前启动，跑完停），写 CSV
  python wb_memwatch.py --watch 600 --csv vram_log.csv

  # 指定要重点跟踪的进程名（默认自动找 python/comfy）
  python wb_memwatch.py --watch 600 --track python.exe

输出
----
  CSV 列：ts, gpu_mem_used_mib, gpu_mem_total_mib, gpu_util_pct,
          ram_used_gb, ram_total_gb, ram_pct, pagefile_free_gb, tracked_rss_gb
  结束时打印峰值摘要与判定建议。

零依赖：只用标准库 + nvidia-smi（驱动的命令行工具）。psutil 可选。
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime

GIB = 1024 ** 3
MIB = 1024 ** 2
_CREATE_NO_WINDOW = 0x08000000


# --------------------------------------------------------------------------
# 采集原语
# --------------------------------------------------------------------------
def _run(cmd: list[str]) -> str:
    """跑一个命令行工具并返回 stdout（Windows 中文环境用 GBK 兜底）。"""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, FileNotFoundError):
        return ""
    if p.returncode != 0 or not p.stdout:
        # 某些驱动在中文 Windows 上输出 GBK
        try:
            p2 = subprocess.run(
                cmd,
                capture_output=True,
                creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            return p2.stdout.decode("gbk", errors="replace")
        except Exception:
            return ""
    return p.stdout


def gpu_snapshot() -> dict:
    """显存与利用率。取不到 nvidia-smi 时返回空 dict（不抛）。"""
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    line = (out or "").strip().splitlines()
    if not line:
        return {}
    parts = [x.strip() for x in line[0].split(",")]
    if len(parts) < 5:
        return {}
    try:
        return {
            "gpu_name": parts[0],
            "gpu_mem_total_mib": float(parts[1]),
            "gpu_mem_used_mib": float(parts[2]),
            "gpu_mem_free_mib": float(parts[3]),
            "gpu_util_pct": float(parts[4]),
        }
    except ValueError:
        return {}


def gpu_processes() -> list[dict]:
    """占用显存的进程（需要管理员权限才完整）。"""
    out = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    rows: list[dict] = []
    for line in (out or "").strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append({
                "pid": int(parts[0]),
                "name": parts[1],
                "used_mib": float(parts[2]),
            })
        except ValueError:
            continue
    return sorted(rows, key=lambda r: -r["used_mib"])


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


def ram_snapshot() -> dict:
    """物理内存 + 页面文件。当前只支持 Windows。"""
    if os.name != "nt":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                info = {
                    k.strip(): int(v.split()[0]) * 1024
                    for k, v in (ln.split(":", 1) for ln in f if ":" in ln)
                }
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", 0)
            return {
                "ram_total_gb": total / GIB,
                "ram_used_gb": (total - avail) / GIB,
                "ram_pct": (total - avail) / total * 100 if total else 0.0,
                "pagefile_free_gb": 0.0,
            }
        except Exception:
            return {}

    st = _MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return {}
    return {
        "ram_total_gb": st.ullTotalPhys / GIB,
        "ram_used_gb": (st.ullTotalPhys - st.ullAvailPhys) / GIB,
        "ram_pct": float(st.dwMemoryLoad),
        "pagefile_free_gb": st.ullAvailPageFile / GIB,
        "pagefile_total_gb": st.ullTotalPageFile / GIB,
    }


def top_processes(n: int = 12) -> list[dict]:
    """按工作集排序的进程列表。优先 psutil，否则退到 tasklist。"""
    rows: list[dict] = []
    try:
        import psutil  # 可选

        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                rows.append({
                    "pid": p.info["pid"],
                    "name": p.info["name"] or "?",
                    "rss": p.info["memory_info"].rss,
                })
            except Exception:
                continue
        rows.sort(key=lambda r: -r["rss"])
        return rows[:n]
    except ImportError:
        pass

    out = _run(["tasklist", "/FO", "CSV", "/NH"])
    for line in (out or "").splitlines():
        parts = [x.strip().strip('"') for x in line.split('","')]
        if len(parts) < 5:
            continue
        try:
            kb = int(parts[4].replace(",", "").replace(" K", "").replace("K", "").strip())
        except ValueError:
            continue
        rows.append({"pid": int(parts[1]), "name": parts[0], "rss": kb * 1024})
    rows.sort(key=lambda r: -r["rss"])
    return rows[:n]


def find_process(names: tuple[str, ...], rows: list[dict]) -> dict | None:
    """在进程列表里找第一个名字命中的（不分大小写）。"""
    for want in names:
        low = want.lower()
        for r in rows:
            if low in r["name"].lower():
                return r
    return None


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def print_snapshot(rows: list[dict] | None = None) -> dict:
    gpu = gpu_snapshot()
    ram = ram_snapshot()
    rows = rows if rows is not None else top_processes(12)

    print("=" * 74)
    print(f"H3MangaStudio 资源快照  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    print("\n[GPU]")
    if gpu:
        used, total = gpu["gpu_mem_used_mib"], gpu["gpu_mem_total_mib"]
        pct = used / total * 100 if total else 0
        print(f"  {gpu['gpu_name']}")
        print(f"  显存  {used:8.0f} / {total:.0f} MiB  ({pct:.1f}%)")
        print(f"  利用率 {gpu['gpu_util_pct']:.0f}%")
        gp = gpu_processes()
        if gp:
            print("  占用显存的进程：")
            for r in gp[:8]:
                print(f"    {r['used_mib']:8.0f} MiB  pid={r['pid']:<7} {r['name']}")
        else:
            print("  （无法列出进程，通常需要管理员权限）")
    else:
        print("  取不到 nvidia-smi 输出")

    print("\n[系统内存]")
    if ram:
        print(f"  物理  {ram['ram_used_gb']:.2f} / {ram['ram_total_gb']:.2f} GB  ({ram['ram_pct']:.0f}%)")
        if "pagefile_total_gb" in ram:
            print(f"  页面文件 可用 {ram['pagefile_free_gb']:.2f} / {ram['pagefile_total_gb']:.2f} GB")

    print(f"\n[内存占用 TOP {len(rows)}]")
    for r in rows:
        print(f"  {r['rss'] / GIB:7.2f} GB  pid={r['pid']:<8} {r['name']}")

    print("\n[判定]")
    verdict = []
    if ram:
        if ram["ram_pct"] >= 90:
            verdict.append("!! 系统内存 >=90%：随时可能被 OOM Killer 杀进程（表现为「自动重启」）")
        elif ram["ram_pct"] >= 80:
            verdict.append("!  系统内存 >=80%：接近危险区")
        if ram.get("pagefile_free_gb", 99) < 4:
            verdict.append("!! 页面文件可用 <4GB：换页压力极大，H3 的 19.5GB 权重无法驻留")
    if gpu:
        free = gpu["gpu_mem_free_mib"]
        if free < 512:
            verdict.append("!! 显存可用 <512MiB：下一次分配就可能 CUDA OOM")
        elif free < 1500:
            verdict.append("!  显存可用 <1.5GB：余量偏紧")
    for v in verdict:
        print("  " + v)
    if not verdict:
        print("  当前无异常")

    return {"gpu": gpu, "ram": ram}


def watch(seconds: int, csv_path: str, track: tuple[str, ...], interval: float = 1.0) -> None:
    rows = top_processes(15)
    print(f"采样 {seconds}s -> {csv_path}（Ctrl+C 可提前结束）")
    fields = [
        "ts", "gpu_mem_used_mib", "gpu_mem_total_mib", "gpu_util_pct",
        "ram_used_gb", "ram_total_gb", "ram_pct", "pagefile_free_gb", "tracked_rss_gb",
    ]
    peak = {"gpu_mem_used_mib": 0.0, "ram_pct": 0.0, "tracked_rss_gb": 0.0}
    t0 = time.time()
    n = 0
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            while time.time() - t0 < seconds:
                gpu = gpu_snapshot()
                ram = ram_snapshot()
                procs = top_processes(20)
                hit = find_process(track, procs) if track else None
                row = {
                    "ts": f"{time.time():.1f}",
                    "gpu_mem_used_mib": round(gpu.get("gpu_mem_used_mib", 0), 1),
                    "gpu_mem_total_mib": round(gpu.get("gpu_mem_total_mib", 0), 1),
                    "gpu_util_pct": round(gpu.get("gpu_util_pct", 0), 1),
                    "ram_used_gb": round(ram.get("ram_used_gb", 0), 3),
                    "ram_total_gb": round(ram.get("ram_total_gb", 0), 3),
                    "ram_pct": round(ram.get("ram_pct", 0), 1),
                    "pagefile_free_gb": round(ram.get("pagefile_free_gb", 0), 3),
                    "tracked_rss_gb": round(hit["rss"] / GIB, 3) if hit else "",
                }
                w.writerow(row)
                f.flush()
                n += 1
                peak["gpu_mem_used_mib"] = max(peak["gpu_mem_used_mib"], row["gpu_mem_used_mib"])
                peak["ram_pct"] = max(peak["ram_pct"], row["ram_pct"])
                if hit:
                    peak["tracked_rss_gb"] = max(peak["tracked_rss_gb"], row["tracked_rss_gb"])
                if n % 10 == 0:
                    print(f"  t={int(time.time() - t0):4d}s  "
                          f"VRAM {row['gpu_mem_used_mib']:.0f}MiB  "
                          f"GPU {row['gpu_util_pct']:.0f}%  "
                          f"RAM {row['ram_pct']:.0f}%  "
                          f"页文件余 {row['pagefile_free_gb']:.2f}GB")
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n（用户中断）")

    print("\n" + "=" * 74)
    print(f"采样结束：{n} 个点 -> {csv_path}")
    print(f"  峰值显存   : {peak['gpu_mem_used_mib']:.0f} MiB")
    print(f"  峰值内存占用: {peak['ram_pct']:.0f}%")
    if peak["tracked_rss_gb"]:
        print(f"  被跟踪进程峰值 RSS: {peak['tracked_rss_gb']:.2f} GB")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="H3MangaStudio 显存/内存双曲线监视器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="采样秒数；0 = 只打一次快照（默认）")
    ap.add_argument("--csv", default="vram_log.csv", help="CSV 输出路径")
    ap.add_argument("--track", default="python,comfy",
                    help="重点跟踪的进程名关键字（逗号分隔），默认 python,comfy")
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔秒，默认 1.0")
    args = ap.parse_args()

    print_snapshot()
    if args.watch > 0:
        print()
        track = tuple(s.strip() for s in args.track.split(",") if s.strip())
        watch(args.watch, args.csv, track, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
