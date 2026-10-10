"""稀疏注意力 A/B 对比工具（dense vs sol-attn）。

为什么要这个工具
----------------
`sparse_attn` 打开之后有两件事必须用**实测**回答，不能靠理论：

  1. **到底生效了吗？** 官方 `BlockSparseAttention` 有 `min_tokens` 门槛，
     序列太短会**静默**回退 dense。渲染报告里的 `sparse_attention.applied`
     和 `preflight.warnings` 会说实话。
  2. **画质掉了吗？** 稀疏注意力是近似计算，官方标着 experimental。
     漫画线稿对碎纹理敏感，必须并排看。

用法（需要 ComfyUI 环境，建议用 `.venv` 的 python）
--------------------------------------------------
  # 1) 生成两份只差一个开关的工作流（同 seed，其它参数完全一致）
  <ComfyUI根>/.venv/Scripts/python.exe tools/sparse_ab.py make \\
      --out-dir D:/h3ab --method sol-attn --tau 1.3 --min-tokens 8192

  # 2) 两份分别跑一遍，把 ⑦ 渲染节点的 report 输出各存成一个 .txt
  #    然后对比耗时 / 显存 / 是否真的生效
  ... tools/sparse_ab.py compare --off off_report.txt --on on_report.txt

  # 3) 并排抽帧看画质（用 ComfyUI 自带的 ffmpeg + cv2）
  ... tools/sparse_ab.py frames --off off.mp4 --on on.mp4 --out-dir D:/h3ab
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)

SPARSE_WIDGETS = (
    "sparse_attn", "sparse_tau", "sparse_keep_percent",
    "sparse_start_percent", "sparse_min_tokens", "sparse_sink",
)


# --------------------------------------------------------------------------
# 中文路径安全的图像 IO
# --------------------------------------------------------------------------
# cv2.imread / cv2.imwrite 在 Windows 上**不支持非 ASCII 路径**，而且失败时
# 不抛异常（imread 返回 None、imwrite 返回 False）→ 静默出错。
# 本项目的漫画素材路径几乎必然含中文，所以一律走 fromfile / imencode。
def _imread(path: str):
    """读图为 BGR ndarray；失败返回 None。"""
    try:
        import cv2
        import numpy as np
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite(path: str, img) -> bool:
    """写图。返回是否成功（cv2.imwrite 在中文路径下会静默失败）。"""
    try:
        import cv2
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    except Exception:
        return False
    if not ok:
        return False
    buf.tofile(path)
    return True


# --------------------------------------------------------------------------
# 1) 生成 A/B 工作流
# --------------------------------------------------------------------------
def cmd_make(args) -> int:
    if PKG not in sys.path:
        sys.path.insert(0, PKG)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if COMFY_ROOT not in sys.path:
        sys.path.insert(0, COMFY_ROOT)
    os.chdir(COMFY_ROOT)

    try:
        from make_example_workflow import build  # 同目录
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 导入 make_example_workflow 失败：{type(e).__name__}: {e}")
        print("       必须在 ComfyUI 根目录下、并用 .venv 的 python 运行。")
        return 2

    off_override = {"sparse_attn": "none"}
    on_override = {
        "sparse_attn": args.method,
        "sparse_tau": float(args.tau),
        "sparse_keep_percent": float(args.keep),
        "sparse_start_percent": float(args.start),
        "sparse_min_tokens": int(args.min_tokens),
        "sparse_sink": args.sink,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = [
        ("ab_off_dense", off_override, "A/B · baseline（dense，未启用稀疏）"),
        (f"ab_on_{args.method.replace('-', '_')}", on_override,
         f"A/B · 稀疏注意力（{args.method}）"),
    ]

    for name, override, title in pairs:
        try:
            wf = build(override)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] 构建 {name} 失败：{type(e).__name__}: {e}")
            return 2
        # 把标记写进节点标题，导入后一眼能分清哪份是哪份
        for n in wf["nodes"]:
            if n["type"] == "H3MangaEpisodeRender":
                n["title"] = title
        path = os.path.join(args.out_dir, name + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=1)
        print(f"  已写出 {path}")

    print()
    print("下一步：")
    print("  1) 把两份工作流分别拖进 ComfyUI 各跑一次（**同 seed**，不要改其它参数）")
    print("  2) 把 ⑦整集渲染 的 report 输出各存成一个 .txt")
    print("  3) 本工具 compare 子命令对比耗时/显存；frames 子命令抽帧并排看画质")
    return 0


# --------------------------------------------------------------------------
# 2) 对比两份渲染报告
# --------------------------------------------------------------------------
def _load_report(path: str) -> dict:
    """渲染报告是 JSON + 可选的「失败镜头」尾巴，这里容错解析。"""
    raw = open(path, "r", encoding="utf-8-sig").read()
    head = raw.split("\n\n失败镜头")[0]
    return json.loads(head)


def _peak_vram(rep: dict):
    vals = [s.get("vram_used_mib") for s in (rep.get("memory", {}).get("stages") or [])
            if s.get("vram_used_mib") is not None]
    return max(vals) if vals else None


def _peak_ram(rep: dict):
    vals = [s.get("ram_pct") for s in (rep.get("memory", {}).get("stages") or [])
            if s.get("ram_pct") is not None]
    return max(vals) if vals else None


def cmd_compare(args) -> int:
    try:
        off = _load_report(args.off)
        on = _load_report(args.on)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 解析报告失败：{type(e).__name__}: {e}")
        return 2

    def row(label, a, b, fmt="{}", better="low"):
        sa, sb = fmt.format(a), fmt.format(b)
        delta = ""
        try:
            fa, fb = float(a), float(b)
            if fa:
                pct = (fb - fa) / fa * 100.0
                arrow = "↓" if (fb < fa) == (better == "low") else "↑"
                delta = f"{arrow} {abs(pct):.1f}%"
        except (TypeError, ValueError):
            pass
        print(f"  {label:<22} {sa:>16} {sb:>16}   {delta}")

    print("=" * 78)
    print(f"稀疏注意力 A/B 对比：{os.path.basename(args.off)}  vs  {os.path.basename(args.on)}")
    print("=" * 78)
    print(f"  {'指标':<22} {'baseline(dense)':>16} {'稀疏':>16}   {'变化':<10}")
    print("  " + "-" * 74)

    row("总耗时(s)", off.get("elapsed_sec"), on.get("elapsed_sec"), "{:.1f}")
    row("单镜均时(s)", off.get("per_shot_avg_sec"), on.get("per_shot_avg_sec"), "{:.1f}")
    row("成功镜数", off.get("done_shots"), on.get("done_shots"), "{}", better="high")
    row("输出帧数", off.get("total_frames_out"), on.get("total_frames_out"), "{}", "high")
    row("峰值显存(MiB)", _peak_vram(off), _peak_vram(on), "{:.0f}")
    row("峰值内存(%)", _peak_ram(off), _peak_ram(on), "{:.0f}")

    print()
    print("=" * 78)
    print("稀疏注意力是否真的生效")
    print("=" * 78)
    for tag, rep in (("baseline", off), ("稀疏", on)):
        sp = rep.get("sparse_attention") or {}
        print(f"  [{tag}] method={sp.get('method')}  applied={sp.get('applied')}")
        if sp.get("reason") and sp.get("reason") != "已应用":
            print(f"          原因：{sp['reason']}")
        if sp.get("kernel") is not None:
            print(f"          sol_attn 内核可用：{sp['kernel']}")

    warn = (on.get("preflight") or {}).get("warnings") or []
    hits = [w for w in warn if "稀疏" in w]
    if hits:
        print()
        print("  ⚠ 预检里与稀疏相关的警告：")
        for w in hits:
            print(f"    - {w}")

    print()
    print("=" * 78)
    print("判定")
    print("=" * 78)
    sp_on = (on.get("sparse_attention") or {})
    if not sp_on.get("applied"):
        print("  ❌ 稀疏**没有生效** —— 这次对比反映的是同一条 dense 路径。")
        print("     常见原因：序列 token 数低于 sparse_min_tokens；或本机没有 sol_attn 内核。")
        print("     建议把 sparse_min_tokens 降到 8192，或把画幅提到 960×544 再试。")
    else:
        try:
            a, b = float(off.get("elapsed_sec")), float(on.get("elapsed_sec"))
            pct = (b - a) / a * 100.0 if a else 0.0
            if pct < -5:
                print(f"  ✅ 稀疏生效，总耗时下降 {abs(pct):.1f}%")
            elif pct > 5:
                print(f"  ⚠ 稀疏生效，但总耗时反而上升 {pct:.1f}% —— "
                      f"序列可能太短，稀疏的调度开销盖过了收益。建议关掉。")
            else:
                print(f"  ⚠ 稀疏生效，但耗时几乎没变（{pct:+.1f}%）—— "
                      f"说明当前序列长度下稀疏收益很小。")
        except (TypeError, ValueError):
            pass
        print("  画质请用 frames 子命令抽帧并排看，或直接看两份视频。")
    return 0


# --------------------------------------------------------------------------
# 3) 抽帧并排（用 ComfyUI 自带的 ffmpeg + cv2）
# --------------------------------------------------------------------------
def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg  # ComfyUI 自带
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def _extract(ffmpeg: str, video: str, t: float, out_png: str) -> bool:
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
           "-i", video, "-frames:v", "1", out_png]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and os.path.isfile(out_png)
    except Exception:  # noqa: BLE001
        return False


def cmd_frames(args) -> int:
    ff = _ffmpeg()
    if not ff:
        print("[FAIL] 找不到 ffmpeg（ComfyUI 一般自带 imageio_ffmpeg）。")
        print("       可手动执行：")
        print(f"         ffmpeg -ss 1 -i {args.off} -frames:v 1 a.png")
        print(f"         ffmpeg -ss 1 -i {args.on}  -frames:v 1 b.png")
        return 2

    try:
        import cv2
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 需要 cv2/numpy（ComfyUI 环境本来就有）：{type(e).__name__}: {e}")
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    times = [float(x) for x in args.at.split(",") if x.strip()]
    rows = []

    with tempfile.TemporaryDirectory() as td:
        for tag, video in (("dense", args.off), ("sparse", args.on)):
            if not os.path.isfile(video):
                print(f"[FAIL] 找不到视频：{video}")
                return 2
            imgs = []
            for i, t in enumerate(times):
                png = os.path.join(td, f"{tag}_{i}.png")
                if _extract(ff, video, t, png):
                    im = _imread(png)
                    if im is not None:
                        imgs.append(im)
            if not imgs:
                print(f"[FAIL] 从 {video} 抽帧失败")
                return 2
            rows.append((tag, imgs))

        # 统一高度，横向拼接：上排 dense，下排 sparse
        h = min(im.shape[0] for _t, ims in rows for im in ims)
        strips = []
        for tag, ims in rows:
            resized = []
            for im in ims:
                s = h / im.shape[0]
                resized.append(cv2.resize(im, (max(1, int(im.shape[1] * s)), h)))
            strip = np.hstack(resized)
            cv2.putText(strip, tag, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(strip, tag, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 0), 1, cv2.LINE_AA)
            strips.append(strip)
        w = min(s.shape[1] for s in strips)
        sheet = np.vstack([s[:, :w] for s in strips])

    out = os.path.join(args.out_dir, "ab_sheet.png")
    _imwrite(out, sheet)
    print(f"已写出并排对比图：{out}")
    print(f"  上排 = dense（{os.path.basename(args.off)}）")
    print(f"  下排 = 稀疏（{os.path.basename(args.on)}）")
    print(f"  抽帧时间点：{times}")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="稀疏注意力 A/B 对比（dense vs sol-attn）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="生成两份只差稀疏开关的工作流")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--method", default="sol-attn",
                   choices=["sol-attn", "sla", "vsa"])
    m.add_argument("--tau", type=float, default=1.3, help="仅 sol-attn")
    m.add_argument("--keep", type=float, default=10.0, help="仅 sla/vsa")
    m.add_argument("--start", type=float, default=0.2)
    m.add_argument("--min-tokens", type=int, default=8192,
                   help="默认给 8192（比官方 12288 低），否则 768x448 的 5s 镜头刚好被挡在门外")
    m.add_argument("--sink", default="exact_kv_and_rows",
                   choices=["exact_kv_and_rows", "exact_kv", "off"])
    m.set_defaults(fn=cmd_make)

    c = sub.add_parser("compare", help="对比两份渲染报告的耗时/显存/是否生效")
    c.add_argument("--off", required=True, help="baseline（dense）的 report")
    c.add_argument("--on", required=True, help="稀疏版的 report")
    c.set_defaults(fn=cmd_compare)

    f = sub.add_parser("frames", help="从两份视频抽出同一时间点的帧并排成图")
    f.add_argument("--off", required=True)
    f.add_argument("--on", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--at", default="0.5,2.5,4.5", help="抽帧时间点（秒，逗号分隔）")
    f.set_defaults(fn=cmd_frames)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
