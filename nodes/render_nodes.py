"""执行层节点：单镜条件（调试）+ 整集渲染（批量量产核心）。

设计要点
--------
* **复用官方节点**：conditioning 走 ``MiniMaxH3ReferenceToVideo``，
  采样走官方五件套，解码走 ``VAEDecode`` / ``VAEDecodeAudio``。
  本插件一行 H3 内部逻辑都不复刻（见 core/h3_api.py 的说明）。
* **循环写在节点里**：ComfyUI 工作流没有 for 循环，
  所以「一个节点出一整集」是唯一能真正量产的形态。
* **能降级就降级**：单镜失败默认跳过并记进报告，
  不因为第 7 镜炸掉就让前面 6 镜白算。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from comfy_api.latest import ComfyExtension, io

try:
    from ._io_types import board_in, plan_in, refs_in
except ImportError:  # pragma: no cover
    from _io_types import board_in, plan_in, refs_in

try:
    from ..core import h3_api as api
    from ..core import image_ops as iops
    from ..core import mem as memmod
    from ..core import plan as pl
    from ..core import refs as rf
    from ..core.types import (
        Plan, RefPack, Board,
        align_canvas, align_frame_count, estimate_cost, estimate_video_tokens,
    )
    from ..core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from ..core.ui import safe_display_name as _inp, label_input as _L
except ImportError:  # pragma: no cover
    from core import h3_api as api
    from core import image_ops as iops
    from core import mem as memmod
    from core import plan as pl
    from core import refs as rf
    from core.types import (
        Plan, RefPack, Board,
        align_canvas, align_frame_count, estimate_cost, estimate_video_tokens,
    )
    from core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from core.ui import safe_display_name as _inp, label_input as _L


def _sampler_options() -> List[str]:
    """采样器选项直接取官方列表 —— 早先硬编码 4 个，凭空砍掉了用户的选择。

    ``comfy.samplers`` 在 ComfyUI 启动时就已加载，所以这里不会扫目录、不慢。
    拿不到时退回到「只保证 H3 官方模板用的那个」。
    """
    try:
        import comfy.samplers
        return list(comfy.samplers.SAMPLER_NAMES)
    except Exception:  # pragma: no cover
        return ["res_multistep"]


def _scheduler_options() -> List[str]:
    try:
        import comfy.samplers
        return list(comfy.samplers.SCHEDULER_NAMES)
    except Exception:  # pragma: no cover
        return ["simple"]


def _panel_tensor(board: Optional[Board], index: Optional[int]):
    if board is None or index is None:
        return None
    try:
        return board.images[int(index)]
    except (IndexError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# LoRA 下拉：现算 + 短 TTL 缓存
# /object_info 会为**所有**节点各调一次 INPUT_TYPES，所以不能每次都扫目录。
# 缓存 15 秒，用户丢进新 LoRA 后重取就能看到（配合前端的 refreshComboInNodes）。
# --------------------------------------------------------------------------
_LORA_CACHE: Dict[str, Any] = {"t": -1e9, "names": ["none"]}
_LORA_TTL = 15.0


def _h3_lora_options() -> List[str]:
    import os
    import time as _time

    now = _time.time()
    if now - _LORA_CACHE["t"] < _LORA_TTL:
        return list(_LORA_CACHE["names"])

    names = ["none"]
    try:
        import folder_paths
        for n in folder_paths.get_filename_list("loras"):
            low = os.path.basename(n).lower()
            # 只列 H3 相关的，避免下拉里塞满 Qwen/SD 的 LoRA
            if ("h3" in low or "minimax" in low) and low.endswith((".safetensors", ".ckpt", ".pt")):
                names.append(n)
    except Exception:
        pass

    _LORA_CACHE["t"] = now
    _LORA_CACHE["names"] = names
    return list(names)



# ==========================================================================
# 8. 单镜条件（调试用）
# ==========================================================================
class H3MangaShotConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaShotConditioning",
            display_name=_NL["H3MangaShotConditioning"],
            category="H3 Manga/执行",
            description=(
                "只构建**单个镜头**的 conditioning + latent，方便接到官方采样五件套"
                "做单镜调试。整集量产请用「H3 整集渲染」节点。\n"
                "输出的 positive/latent 与整集渲染内部用的是同一套逻辑，"
                "所以调好的参数可以直接搬过去。"
            ),
            inputs=[
                _inp(io.Clip.Input, "clip", _L("clip")),
                _inp(io.Vae.Input, "vae", _L("vae"), optional=True),
                _inp(io.Vae.Input, "audio_vae", _L("audio_vae"), optional=True),
                plan_in("plan", display_name=_L("plan")),
                refs_in("refs", display_name=_L("refs"), optional=True),
                board_in("board", display_name=_L("board"), optional=True),
                _inp(io.Int.Input, "shot_index", _L("shot_index"),
                     default=0, min=0, max=999, step=1),
                _inp(io.Combo.Input, "ref_image_size", _L("ref_image_size"),
                     options=["match", "max"], default="match",
                     tooltip="match=按生成画幅缩放参考图（快）/ max=2048短边（最像但慢数倍）"),
            ],
            outputs=[
                io.Conditioning.Output(display_name=_OL["positive"]),
                io.Latent.Output(display_name=_OL["latent"]),
                io.String.Output(display_name=_OL["report"]),
            ],
        )

    @classmethod
    def execute(cls, clip, plan, refs=None, board=None, shot_index=0,
                vae=None, audio_vae=None, ref_image_size="match") -> io.NodeOutput:
        if plan is None:
            raise ValueError("单镜条件需要上游的分镜计划（plan）")
        if not plan.shots:
            raise ValueError("分镜计划里没有镜头")
        idx = max(0, min(int(shot_index), len(plan.shots) - 1))
        shot = plan.shots[idx]

        pack = refs or RefPack()
        style_ids = [e.id for e in pack.ordered() if e.kind == "style"]
        panel = _panel_tensor(board, shot.board_image_index)

        layout = rf.shot_ref_layout(
            pack, characters=list(shot.characters),
            panel_image=panel, has_panel=shot.board_image_index is not None,
            style_ids=style_ids,
        )

        w, h = align_canvas(shot.width, shot.height)
        length = align_frame_count(shot.length)

        positive, latent = api.encode_reference_to_video(
            clip=clip, prompt=shot.prompt, width=w, height=h, length=length,
            ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
            ref_images=layout["ref_images"],
            ref_videos=layout["ref_videos"],
            ref_video_audios=layout["ref_video_audios"],
            ref_audios=layout["ref_audios"],
        )

        # seed：shot.seed 优先，否则按渲染节点的同一套规则派生（base_seed=0），
        # 这样报告里显示的值与整集渲染真正用的值口径一致（早先直接打印 0）。
        seed = shot.seed if shot.seed is not None else pl.derive_seed(0, idx, shot.id)

        report = "\n".join([
            f"单镜条件：{shot.id}（index={idx}）",
            "=" * 52,
            f"  画幅     : {w}x{h}",
            f"  帧数     : {length}（{length / 24.0:.2f}s @24fps，已对齐 17n+5）",
            f"  seed     : {seed}（渲染节点的 base_seed 会覆盖它）",
            f"  参考图尺寸: {ref_image_size}",
            rf.layout_table(layout),
            "",
            "提示词：",
            shot.prompt or "(空)",
        ])
        return io.NodeOutput(positive, latent, report, ui={"h3manga_panel": [report]})


# ==========================================================================
# 9. 整集渲染 ★
# ==========================================================================
class H3MangaEpisodeRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaEpisodeRender",
            display_name=_NL["H3MangaEpisodeRender"],
            category="H3 Manga/执行",
            description=(
                "**批量量产核心**：吃完整分镜计划，内部逐镜循环——\n"
                "  逐镜构建 R2V conditioning → 采样 → 解码 → 拼接。\n"
                "工作流画布上只有几根线，一键跑完整集。\n"
                "单镜失败默认跳过并记进报告，不会让整集白跑。"
            ),
            inputs=[
                _inp(io.Model.Input, "model", _L("model")),
                _inp(io.Clip.Input, "clip", _L("clip")),
                _inp(io.Vae.Input, "vae", _L("vae"), tooltip="视频 VAE（minimax_h3_video_vae_*）"),
                _inp(io.Vae.Input, "audio_vae", _L("audio_vae"), optional=True,
                             tooltip="音频 VAE（minimax_h3_audio_vae_*）。不接则不出音频。"),
                plan_in("plan", display_name=_L("plan")),
                refs_in("refs", display_name=_L("refs"), optional=True),
                board_in("board", display_name=_L("board"), optional=True,
                         tooltip="接了分镜格后，每镜会把对应的格图作为 <Picture 1> 视觉锚点"),
                # --- 采样 ---
                _inp(io.Combo.Input, "sampler_name", _L("sampler_name"),
                     options=_sampler_options(), default="res_multistep"),
                _inp(io.Combo.Input, "scheduler", _L("scheduler"),
                     options=_scheduler_options(), default="simple"),
                _inp(io.Int.Input, "steps", _L("steps"), default=20, min=1, max=200, step=1,
                     tooltip="无 Turbo LoRA 用 20；挂了 turbo 4step 用 4~8"),
                _inp(io.Float.Input, "shift_video", _L("shift_video"),
                     default=12.0, min=0.01, max=100.0, step=0.01),
                _inp(io.Float.Input, "shift_audio", _L("shift_audio"),
                     default=3.0, min=0.01, max=100.0, step=0.01),
                _inp(io.Combo.Input, "ref_image_size", _L("ref_image_size"),
                     options=["match", "max"], default="match"),
                # --- LoRA ---
                _inp(io.Combo.Input, "lora_name", _L("lora_name"),
                     options=_h3_lora_options(), default="none",
                     tooltip="H3 turbo LoRA（可选）。挂了之后 steps 可降到 4~8。"
                             "只列文件名含 h3/minimax 的 LoRA。"),
                _inp(io.Float.Input, "lora_strength", _L("lora_strength"),
                     default=1.0, min=-10.0, max=10.0, step=0.01),
                # --- 批量控制 ---
                _inp(io.Int.Input, "base_seed", _L("base_seed"), default=0,
                     min=0, max=0xFFFFFFFFFFFFFFFF, step=1, control_after_generate=True),
                _inp(io.Int.Input, "max_shots", _L("max_shots"),
                     default=0, min=0, max=999, step=1,
                     tooltip="0 = 全部。调试时先跑 1~2 镜确认参数。"),
                _inp(io.String.Input, "skip_done_ids", _L("skip_done_ids"),
                     default="", tooltip="逗号分隔的镜头 id，断点续跑时跳过"),
                _inp(io.Combo.Input, "on_error", _L("on_error"),
                     options=["skip", "abort"], default="skip",
                     tooltip="skip=单镜失败继续跑后面的 / abort=立刻中断"),
                # --- 资源 ---
                _inp(io.Boolean.Input, "unload_stages", _L("unload_stages"), default=True,
                     tooltip="分阶段卸载：编码后卸文本编码器、采样后卸视频模型、解码后卸 VAE。\n"
                             "本机权重总量 33.5GB（R2V UNET 19.5GB + 文本编码器 13.9GB）"
                             "而内存只有 16GB —— 三个模型同时驻留必然爆，强烈建议保持开启。"),
                # --- 稀疏注意力（可选加速，包装官方 BlockSparseAttention） ---
                _inp(io.Combo.Input, "sparse_attn", _L("sparse_attn"),
                     options=["none", "sol-attn", "sla", "vsa"], default="none",
                     tooltip="稀疏注意力（官方 BlockSparseAttention，实验特性）。\n"
                             "none = 不启用（默认，行为与以前完全一致）\n"
                             "sol-attn = 免训练自适应阈值（推荐先试，不需要新权重）\n"
                             "vsa = FastVideo 的 Video Sparse Attention（需 FastH3 权重才发挥）\n"
                             "sla = 需配套 LoRA\n"
                             "⚠ 官方有 min_tokens 门槛：768×432 的 5 秒镜头约 11.5k token，"
                             "低于默认 12288 会走 dense。"),
                _inp(io.Float.Input, "sparse_tau", _L("sparse_tau"),
                     default=1.3, min=0.0, max=4.0, step=0.05,
                     tooltip="仅 sol-attn：分数分布阈值，越大越稀疏"
                             "（1.0≈保留 16% key block，1.5≈7%，2.0≈2.7%）"),
                _inp(io.Float.Input, "sparse_keep_percent", _L("sparse_keep_percent"),
                     default=10.0, min=0.5, max=95.0, step=0.5,
                     tooltip="仅 sla / vsa：每个 query 块精确保留的 key 块百分比。"
                             "FastH3-VSA 权重是在 10% 上训练的"),
                _inp(io.Float.Input, "sparse_start_percent", _L("sparse_start_percent"),
                     default=0.2, min=0.0, max=1.0, step=0.01,
                     tooltip="采样进度到百分之几才开始稀疏；之前保持 dense，保护构图成型阶段"),
                _inp(io.Int.Input, "sparse_min_tokens", _L("sparse_min_tokens"),
                     default=12288, min=0, max=1 << 20, step=512,
                     tooltip="序列短于这个 token 数就保持 dense（官方默认 12288）。"
                             "768×432 的 5 秒镜头约 11.5k → 调到 8192 才会生效"),
                _inp(io.Combo.Input, "sparse_sink", _L("sparse_sink"),
                     options=["exact_kv_and_rows", "exact_kv", "off"],
                     default="exact_kv_and_rows",
                     tooltip="H3 专用：让每个 query 精确注意到文本 / 音频 / 参考图行（约 3% 开销）。\n"
                             "exact_kv_and_rows = 默认，参考图与生成音频都保持精确（推荐）\n"
                             "exact_kv = 只精确保留参考行\n"
                             "off = 全稀疏，最快但角色参考的影响会变弱"),
            ],
            outputs=[
                io.Image.Output(display_name=_OL["frames"]),
                io.Audio.Output(display_name=_OL["audio"]),
                io.String.Output(display_name=_OL["report"]),
            ],
            is_output_node=False,
        )

    @classmethod
    def execute(cls, model, clip, vae, plan, refs=None, board=None,
                audio_vae=None, sampler_name="res_multistep", scheduler="simple",
                steps=20, shift_video=12.0, shift_audio=3.0, ref_image_size="match",
                lora_name="none", lora_strength=1.0, base_seed=0, max_shots=0,
                skip_done_ids="", on_error="skip", unload_stages=True,
                sparse_attn="none", sparse_tau=1.3, sparse_keep_percent=10.0,
                sparse_start_percent=0.2, sparse_min_tokens=12288,
                sparse_sink="exact_kv_and_rows") -> io.NodeOutput:
        import torch

        if plan is None:
            raise ValueError("整集渲染需要上游的分镜计划（plan）")
        if not plan.shots:
            raise ValueError("分镜计划里没有镜头，无法渲染")
        if model is None or clip is None or vae is None:
            raise ValueError("整集渲染需要 model / clip / vae 三个必需输入")

        # ★ 先 clone：预检会修正 length/画幅，build_render_tasks 会派生 seed。
        #   不改动上游传进来的 plan，否则同一个 plan 接两个渲染节点时会互相污染。
        plan = plan.clone()

        pack = refs or RefPack()
        skip = [s.strip() for s in (skip_done_ids or "").split(",") if s.strip()]

        # ---- 预检：把「模型侧硬约束」在采样之前一次性报清楚 ----
        preflight = cls._preflight(plan, pack, board,
                                   unload_stages=bool(unload_stages))

        tasks = pl.build_render_tasks(
            plan, pack, base_seed=int(base_seed),
            max_shots=int(max_shots), skip_ids=skip,
        )
        if not tasks:
            raise ValueError("展开后没有任何待渲染镜头（可能被 max_shots 或 skip_done_ids 全过滤了）")

        # ★ 报告口径：让 max_shots 的效果**显式可见**。
        #   真实事故：用户看到计划写「9.6s」、出片却只有 4s —— 因为本节点
        #   max_shots=1 只渲了第 1 镜（计划两镜 107+124=231 帧=9.625s）。
        if int(max_shots) and len(tasks) < len(plan.shots):
            preflight["warnings"].append(
                f"★ max_shots={int(max_shots)}：本次**只渲染前 {len(tasks)} 镜**，"
                f"计划共 {len(plan.shots)} 镜。想要整集请把 max_shots 设为 0。"
            )

        # ---- 模型补丁只做一次（shift + LoRA），避免每镜 clone ----
        patched = api.apply_sigma_shift(model, float(shift_video), float(shift_audio))
        if lora_name and lora_name != "none":
            try:
                patched = api.apply_lora(patched, lora_name, float(lora_strength))
            except api.H3ApiError as e:
                # 降级：LoRA 加载失败不该让整集跑不了。
                # 用户从别人那拷来的工作流里，下拉值很可能指向一个不存在的文件。
                preflight["warnings"].append(
                    f"LoRA「{lora_name}」加载失败，已跳过（本次按无 LoRA 渲染）：{e}"
                )
                preflight["ok"] = False

        # ---- 稀疏注意力（可选加速；失败只记录，绝不影响渲染） ----
        # 官方 BlockSparseAttention 有个 min_tokens 门槛，短序列会**静默**回退到 dense，
        # 所以这里先按 H3 的 latent 几何估一次 token 数，低于门槛就提前说清楚。
        sparse_info: Dict[str, Any] = {
            "applied": False, "method": str(sparse_attn or "none"),
            "reason": "未启用（method=none）",
        }
        if sparse_attn and str(sparse_attn) != "none":
            est = max((estimate_video_tokens(s.width, s.height, s.length)
                       for s in plan.shots), default=0)
            if est and est < int(sparse_min_tokens):
                preflight["warnings"].append(
                    f"稀疏注意力：本集最大镜头的视频 token 估算 ≈{est}（下界），"
                    f"低于 sparse_min_tokens={int(sparse_min_tokens)} —— "
                    f"官方会走 dense 回退，开了也不会有加速。"
                    f"可把画幅提到 960×544，或把 sparse_min_tokens 降到 8192。"
                )
            patched, sparse_info = api.apply_sparse_attention(
                patched, method=str(sparse_attn), tau=float(sparse_tau),
                keep_percent=float(sparse_keep_percent),
                start_percent=float(sparse_start_percent),
                min_tokens=int(sparse_min_tokens),
                sink_conditioning=str(sparse_sink),
            )
            if not sparse_info.get("applied"):
                preflight["warnings"].append(
                    f"稀疏注意力未生效（{sparse_info.get('reason')}）—— 本次按 dense 渲染"
                )

        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(len(tasks))
        except Exception:  # pragma: no cover
            pbar = None

        # ★ 分阶段（stage-major）流水线
        #   编码 / 采样 / 解码各自成段，每段结束按需卸载该段的模型。
        #   本机实测：R2V UNET 19.53GB + 文本编码器 13.93GB = 33.5GB 权重，
        #   而物理内存只有 16GB —— 三个模型同时驻留必然被 OOM Killer 杀掉
        #   （表现为 ComfyUI 自动重启，日志里连一条 traceback 都没有）。
        #   串行加载 + 用完即卸是唯一能跑通的形态。
        #   安全性依据：comfy/sd.py 里 CLIP.encode / VAE.decode 内部都会调
        #   load_models_gpu([self.patcher])，所以卸载后下次使用会自动重新加载。
        stages: List[Dict[str, Any]] = []

        def _record(label: str, snap: Dict[str, Any],
                    freed_mib: Optional[float] = None,
                    api_ok: Optional[bool] = None,
                    errors: Optional[List[str]] = None) -> None:
            stages.append({
                "label": label,
                "line": memmod.format_stage_line(label, snap),
                "vram_used_mib": (snap.get("vram") or {}).get("used_mib"),
                "vram_free_mib": (snap.get("vram") or {}).get("free_mib"),
                "ram_used_gb": (snap.get("ram") or {}).get("used_gb"),
                "ram_pct": (snap.get("ram") or {}).get("pct"),
                "pagefile_free_gb": (snap.get("ram") or {}).get("pagefile_free_gb"),
                "freed_vram_mib": freed_mib,
                "unload_ok": api_ok,
                "errors": list(errors or []),
            })

        def _unload_stage(objs: List[Any], label: str) -> None:
            """卸载某阶段用完的模型。unload_stages 关闭时只记快照，不卸载。"""
            if not unload_stages:
                _record(label + "（未卸载）", memmod.snapshot())
                return
            res = memmod.unload(objs, label=label)
            vb = (res.get("vram_before") or {}).get("used_mib")
            va = (res.get("vram_after") or {}).get("used_mib")
            freed = (vb - va) if (vb is not None and va is not None) else None
            _record(label,
                    {"vram": res.get("vram_after") or {},
                     "ram": res.get("ram_after") or {}},
                    freed_mib=freed, api_ok=res.get("ok"),
                    errors=res.get("api_errors") or [])

        all_frames: List[Any] = []
        all_waveforms: List[Any] = []
        sample_rate = None
        done: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        aborted = False
        t0 = time.time()

        _record("起点", memmod.snapshot())

        def _fail(task: Dict[str, Any], stage: str, e: BaseException) -> None:
            failed.append({
                "id": task["id"], "index": task["index"], "stage": stage,
                "error": f"{type(e).__name__}: {e}",
            })

        # ---------------- 阶段 A：逐镜 layout + 条件编码（占用 clip） ----------------
        encoded: List[Dict[str, Any]] = []
        for task in tasks:
            if aborted:
                break
            try:
                shot = plan.shots[task["index"]]
                panel = _panel_tensor(board, task["board_image_index"])
                has_panel = task["board_image_index"] is not None
                # 计划里绑了分镜格、但 board 没接（或索引越界）→ 明确报错。
                # 不能默默当成「无格图」：那会让提示词里的 <Picture 1> 指到别的图上（静默串号）。
                if has_panel and panel is None:
                    raise RuntimeError(
                        f"镜头 {task['id']} 绑定了分镜格 #{task['board_image_index']}，"
                        f"但本节点没接到可用的 board 输入（未连线或索引越界）。\n"
                        f"请把「分镜格」接到本节点的 board 输入。"
                    )

                layout = rf.shot_ref_layout(
                    pack, characters=list(shot.characters),
                    panel_image=panel,
                    has_panel=has_panel,
                    style_ids=[e.id for e in pack.ordered() if e.kind == "style"],
                )
                # 用逐镜布局**再校验一次**：装配时算的槽位键序列必须与现在一致
                if list(layout["ref_image_keys"]) != list(task["ref_image_keys"]):
                    raise RuntimeError(
                        f"参考图槽位在装配后被改动过（装配={task['ref_image_keys']}，"
                        f"现在={list(layout['ref_image_keys'])}）。"
                        f"请重新跑一次「提示词装配」再渲染。"
                    )

                positive, latent = api.encode_reference_to_video(
                    clip=clip, prompt=task["prompt"],
                    width=task["width"], height=task["height"], length=task["length"],
                    ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
                    ref_images=layout["ref_images"],
                    ref_videos=layout["ref_videos"],
                    ref_video_audios=layout["ref_video_audios"],
                    ref_audios=layout["ref_audios"],
                )
                encoded.append({"task": task, "positive": positive, "latent": latent})
            except Exception as e:
                _fail(task, "encode", e)
                if on_error == "abort":
                    aborted = True
            finally:
                if pbar is not None:
                    try:
                        pbar.update(1)
                    except Exception:
                        pass

        _unload_stage([clip], "阶段A后 · 文本编码器")

        # ---------------- 阶段 B：逐镜采样（占用 model） ----------------
        sampled: List[Dict[str, Any]] = []
        for item in encoded:
            if aborted:
                break
            task = item["task"]
            try:
                s = api.sample(
                    model=patched, positive=item["positive"], latent=item["latent"],
                    sampler_name=sampler_name, scheduler=scheduler,
                    steps=int(steps), denoise=1.0, seed=int(task["seed"]),
                )
                sampled.append({"task": task, "sampled": s})
            except Exception as e:
                _fail(task, "sample", e)
                if on_error == "abort":
                    aborted = True

        _unload_stage([patched, model], "阶段B后 · 视频模型")

        # ---------------- 阶段 C：逐镜解码（占用 vae / audio_vae） ----------------
        # 注意：这里**不因 aborted 而跳过** —— 采样（最贵的一步）已经完成了，
        # 把已经算出来的潜空间解码出来给用户，比丢掉更合理。
        for item in sampled:
            task = item["task"]
            try:
                frames = api.decode_video(vae, item["sampled"])
                if frames is not None and len(frames) > 0:
                    all_frames.append(frames)

                if audio_vae is not None:
                    aud = api.decode_audio(audio_vae, item["sampled"])
                    wf = (aud or {}).get("waveform")
                    if wf is not None and wf.shape[-1] > 0:
                        all_waveforms.append(wf)
                        sample_rate = (aud or {}).get("sample_rate", sample_rate)

                done.append({
                    "id": task["id"], "index": task["index"], "seed": task["seed"],
                    "frames": int(frames.shape[0]) if frames is not None else 0,
                    "width": task["width"], "height": task["height"],
                    "length": task["length"],
                    "picture_tags": task["picture_tags"],
                })
            except Exception as e:
                _fail(task, "decode", e)
                if on_error == "abort":
                    aborted = True

        _unload_stage([vae, audio_vae], "阶段C后 · 视频/音频 VAE")

        _record("结束", memmod.snapshot())

        # ---- 拼接 ----
        if not all_frames:
            first = failed[0] if failed else {"id": "-", "error": "未知原因"}
            raise api.H3ApiError(
                f"整集没有任何镜头渲染成功（计划 {len(tasks)} 镜，失败 {len(failed)} 镜）。\n"
                f"首个失败原因：[{first['id']}] {first['error']}\n"
                f"排查提示：\n"
                f"  1) 确认「分镜绑定」接在「提示词装配」之前 —— "
                f"否则提示词里没有分镜格锚点，参考图槽位会对不上而被守卫拦下。\n"
                f"  2) 若错误信息里有 `shape '[1, -1, 3, ...]' is invalid for input of size ...`，"
                f"那是**参考图张量维度**不对（官方要求 4 维 [B,H,W,C]）。"
                f"本版本已在 shot_ref_layout 与 h3_api 两道关口自动升维；"
                f"若仍报错，请输出该图的 shape 以便定位。"
            )

        frames_out = iops.concat_frames(all_frames)
        audio_out = iops.concat_audio(all_waveforms, sample_rate)
        if audio_vae is not None and audio_out and audio_out.get("waveform") is None:
            preflight["warnings"].append(
                f"音轨拼接失败，本次不输出音频：{audio_out.get('error')}"
            )

        elapsed = time.time() - t0
        report_obj = {
            "episode": plan.episode,
            "preflight": preflight,
            "scheduled": len(tasks),
            # ★ 渲染范围：解释「计划 9.6s 却只出 4s」这类困惑的唯一地方
            "scope": {
                "plan_shots": len(plan.shots),
                "rendered_shots": len(tasks),
                "max_shots": int(max_shots),
                "skipped_count": len(plan.shots) - len(tasks),
                "planned_frames": plan.total_frames(),
                "planned_seconds": round(plan.total_seconds(), 3),
            },
            # 计数与明细分开命名：同名键会被后面的覆盖，
            # 早期版本把 done 写成 len(done) 又被明细覆盖，报告里就只剩列表了。
            "done_shots": len(done),
            "failed_shots": len(failed),
            "elapsed_sec": round(elapsed, 1),
            "per_shot_avg_sec": round(elapsed / max(1, len(done)), 1),
            "total_frames_out": int(frames_out.shape[0]) if frames_out is not None else 0,
            "output_duration_sec": round(
                (int(frames_out.shape[0]) / 24.0) if frames_out is not None else 0.0, 3
            ),
            "cost": estimate_cost(plan, int(steps)),
            "params": {
                "sampler": sampler_name, "scheduler": scheduler, "steps": int(steps),
                "shift_video": shift_video, "shift_audio": shift_audio,
                "lora": lora_name, "lora_strength": lora_strength,
                "ref_image_size": ref_image_size, "base_seed": int(base_seed),
                "unload_stages": bool(unload_stages),
            },
            # ★ 稀疏注意力的实际生效情况（没生效时 reason 会说明原因）
            "sparse_attention": sparse_info,
            # ★ 各阶段的显存/内存曲线。本机 16GB 内存对 33.5GB 权重，
            #   这一节是判断「有没有真的把上一个模型卸掉」的唯一依据。
            "memory": {
                "unload_stages": bool(unload_stages),
                # 人读的一行式曲线；放在 JSON 内部，避免破坏
                # callers 里 `report.split("\n\n失败镜头")[0]` 的 JSON 解析
                "lines": [s["line"] for s in stages],
                "stages": stages,
            },
            "done": done,
            "failed": failed,
        }
        report = json.dumps(report_obj, ensure_ascii=False, indent=1)
        if failed:
            report += "\n\n失败镜头：\n" + "\n".join(
                f"  [{f['id']}]({f.get('stage', '?')}) {f['error']}" for f in failed
            )

        return io.NodeOutput(frames_out, audio_out, report, ui={"h3manga_panel": [report]})

    # ------------------------------------------------------------------
    @staticmethod
    def _preflight(plan: Plan, refs: RefPack, board: Optional[Board],
                   unload_stages: bool = True) -> Dict[str, Any]:
        """采样之前的静态预检。返回结构化结果，不抛异常（除非彻底不可用）。

        把问题在**花钱之前**报清楚：帧数对齐、分辨率倍数、参考图溢出、缺图。
        """
        result: Dict[str, Any] = {"warnings": [], "fixed": [], "ok": True}

        empty_prompts: List[str] = []
        for s in plan.shots:
            if s.length % 17 != 5:
                fixed = align_frame_count(s.length)
                result["fixed"].append(
                    f"[{s.id}] length {s.length} → {fixed}（H3 要求 17n+5）"
                )
                s.length = fixed
            if s.width % 32 or s.height % 32:
                w, h = align_canvas(s.width, s.height)
                result["fixed"].append(f"[{s.id}] 画幅 {s.width}x{s.height} → {w}x{h}（须 32 倍数）")
                s.width, s.height = w, h
            if s.length > 362:
                result["warnings"].append(
                    f"[{s.id}] length={s.length} 超出官方训练区间（≈362），效果不保证"
                )
            # ★ 提示词为空是最常见的接线错误（漏接「提示词装配」），
            #   而模型收到空 prompt 只会出无意义画面且不报错 —— 必须在采样前拦住。
            if not (s.prompt or "").strip():
                empty_prompts.append(s.id)

        if empty_prompts:
            result["ok"] = False
            result["warnings"].append(
                f"{len(empty_prompts)} 个镜头的提示词是空的：{', '.join(empty_prompts[:8])}"
                + ("..." if len(empty_prompts) > 8 else "")
                + "。请确认「提示词装配」节点已接到本节点（plan 走的是装配后的输出，"
                  "而不是「分镜清单加载」的原始输出）。"
            )

        if board is None:
            result["warnings"].append("没有接分镜格（board）——所有镜头都是纯参考图 + 文字生成")
        else:
            missing = [s.id for s in plan.shots
                       if s.board_image_index is None]
            if missing:
                result["warnings"].append(
                    f"{len(missing)} 个镜头没有绑定分镜格：{', '.join(missing[:8])}"
                    + ("..." if len(missing) > 8 else "")
                )

            # ★ 碎条分镜格：切格偶尔会切出 1~几像素高的条带（封面横幅、装订线边角）。
            #   它作为 <Picture 1> 锚点毫无画面信息，而且尺寸极小 ——
            #   真实事故里正是这样一个 1x1152 的碎条把参考图算成了畸形目标尺寸。
            #   采样前就说清楚，别等用户对着一堆废画面猜。
            thin = []
            for p in board.panels:
                box = list(p.box) + [0, 0, 0, 0]
                w, h = int(box[2]), int(box[3])
                if min(w, h) < 8:
                    thin.append(f"#{p.index}({w}x{h})")
            if thin:
                result["warnings"].append(
                    f"{len(thin)} 个分镜格是碎条（{', '.join(thin[:6])}"
                    + ("..." if len(thin) > 6 else "") + "）—— "
                    "作为 <Picture 1> 锚点没有画面信息，建议改用切格节点的 manual_boxes 手填框，"
                    "或把这些镜头绑到别的格。"
                )

        known = {e.id for e in refs.entries}
        unknown: List[str] = []
        for s in plan.shots:
            for c in s.characters:
                if c not in known and c not in unknown:
                    unknown.append(c)
        if unknown:
            result["warnings"].append(
                f"参考包里没有这些角色：{', '.join(unknown)} —— 提示词里已移除对它们的引用，"
                f"画面不会保持这些角色的形象"
            )

        for w in refs.resolve_slots()["warnings"]:
            result["warnings"].append(w)

        # ★ 资源预检（本机专属的现实约束）
        #   H3 权重总量 33.5GB（R2V UNET 19.53 + 文本编码器 13.93）
        #   而物理内存只有 16GB —— 内存接近打满时进程会被系统杀掉，
        #   表现为「ComfyUI 自动重启」，日志里没有任何 traceback。
        ram = memmod.ram_snapshot()
        if ram:
            pct = ram.get("pct", 0)
            if pct >= 90:
                result["warnings"].append(
                    f"系统内存已占用 {pct:.0f}%（{ram.get('used_gb', 0):.1f}/"
                    f"{ram.get('total_gb', 0):.1f} GB）。H3 权重总量 33.5GB，"
                    f"当前余量很可能不够 —— 建议先关掉浏览器等占内存的程序，"
                    f"并降低画幅，否则可能被系统杀掉进程（表现为 ComfyUI 自动重启）。"
                )
            elif pct >= 80:
                result["warnings"].append(
                    f"系统内存已占用 {pct:.0f}%，接近危险区（权重 33.5GB / 内存 "
                    f"{ram.get('total_gb', 0):.1f} GB）。"
                )
            pf = ram.get("pagefile_free_gb")
            if pf is not None and pf < 4:
                result["warnings"].append(
                    f"页面文件可用仅 {pf:.2f} GB —— 换页压力很大，"
                    f"会显著拖慢权重加载甚至失败。"
                )

        if not unload_stages:
            result["warnings"].append(
                "「分阶段卸载模型」已关闭。本机权重 33.5GB / 内存 16GB，"
                "三个模型同时驻留几乎必然爆内存 —— 除非你已升级内存，否则建议打开。"
            )

        return result
