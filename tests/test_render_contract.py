"""第 4 层：契约测试（桩替换 comfy 依赖 + 真跑 execute() + 反向验证）。

跑法（standalone-env 即可，CPU torch 够用）：
    "<安装根>/standalone-env/python.exe" tests/test_render_contract.py

这一层抓的是**静默失效**：功能看着跑完了，其实悄悄传错了东西。
本插件最贵的 bug 就是「提示词里写 <Picture 2>，但实际传进去的第 2 张图是别人」——
不报错、不崩，只是角色串号，肉眼在成品上很难定位。

做法：把 `core.h3_api` 的四个重活函数换成 recorder，真跑
`H3MangaEpisodeRender.execute()`，然后**反查**：
    提示词里的 <Picture i>  →  实际 ref_images 的第 i 个位置  →  那张图的指纹
指纹必须等于该角色自己的定妆图。只要序号规则被改坏，这条立刻变红。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_package  # noqa: E402

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


pkg = load_package(with_stub=True)
import importlib  # noqa: E402

core = importlib.import_module(pkg.__name__ + ".core")
types_ = core.types
sp = core.script_parse
pb = core.prompt_build
rf = core.refs
pl = core.plan
api = core.h3_api
render_mod = importlib.import_module(pkg.__name__ + ".nodes.render_nodes")

import torch  # noqa: E402


# --------------------------------------------------------------------------
# 指纹图像：每张图用一个唯一数值填充，便于反查「传进去的到底是哪张」
# --------------------------------------------------------------------------
def make_img(value: float):
    return torch.full((4, 4, 3), float(value), dtype=torch.float32)


def img_fingerprint(t) -> int:
    return int(round(float(t.flatten()[0].item())))


CHAR_A_IMGS = [101, 102]      # 纲吉：两张定妆图
CHAR_B_IMGS = [201]           # Reborn：一张
STYLE_IMG = 301
PANEL_IMGS = [11, 22, 33]     # 三个分镜格

# --------------------------------------------------------------------------
# 构造参考包 / 分镜格 / 计划
# --------------------------------------------------------------------------
# ★ 这一层**故意关掉「一角色一图」**（one_image_per_character=False），
#   让纲吉吃掉 2 个图槽 —— 多槽位是暴露「逐镜重排 / 串号」最灵敏的构造：
#   少放一张就会让后面所有角色整体错位。一角色一图下的序号契约在第 1 层覆盖。
pack = rf.build_ref_pack(
    character_names="纲吉, 纲吉, Reborn",
    character_images=[make_img(v) for v in CHAR_A_IMGS + CHAR_B_IMGS],
    character_describes="棕发橙眼\n黑西装婴儿",
    style_name="基准风格",
    style_images=make_img(STYLE_IMG),
    one_image_per_character=False,
)
char_ids = [e.id for e in pack.entries if e.kind == "character"]
A, B = char_ids[0], char_ids[1]

board = types_.Board(
    panels=[types_.Panel(index=i, box=[0, 0, 100, 100], page_index=0)
            for i in range(len(PANEL_IMGS))],
    images=[make_img(v) for v in PANEL_IMGS],
    pages=1,
)

SCRIPT = """
# episode: 契约测试
# style: manga_anime
=== s01 ===
panel: 0
duration: 5
characters: 纲吉
camera: slow_push_in
action: @纲吉 从门后走出
speaker: 纲吉
dialogue: 第一句。

=== s02 ===
panel: 1
duration: 5
characters: Reborn
camera: close_up
action: @Reborn 抬头
speaker: Reborn
dialogue: 第二句。

=== s03 ===
panel: 2
duration: 5
characters: 纲吉, Reborn
camera: static
action: @纲吉 和 @Reborn 同框
"""
plan = sp.parse_text(SCRIPT)
pl.bind_board(plan, board, mode="by_panel_index")
pb.compose_plan(plan, pack, dialogue_mode="spoken", lock_consistency=True)

# --------------------------------------------------------------------------
# 装 recorder
# --------------------------------------------------------------------------
CALLS = {"encode": [], "sample": [], "decode_v": [], "decode_a": [], "shift": [], "lora": []}


def rec_encode(**kw):
    CALLS["encode"].append(kw)
    return (f"POS::{len(CALLS['encode'])}", {"samples": f"LAT::{len(CALLS['encode'])}"})


def rec_shift(model, shift_video=12.0, shift_audio=3.0):
    CALLS["shift"].append({"shift_video": shift_video, "shift_audio": shift_audio})
    return model


def rec_lora(model, name, strength=1.0):
    CALLS["lora"].append({"name": name, "strength": strength})
    return model


def rec_sample(**kw):
    CALLS["sample"].append(kw)
    return kw["latent"]


def rec_decode_video(vae, latent):
    CALLS["decode_v"].append({"latent": latent})
    return torch.zeros((5, 8, 8, 3), dtype=torch.float32)   # 每镜 5 帧


def rec_decode_audio(vae, latent):
    CALLS["decode_a"].append({"latent": latent})
    return {"waveform": torch.zeros((1, 2, 40), dtype=torch.float32), "sample_rate": 32000}


api.encode_reference_to_video = rec_encode
api.apply_sigma_shift = rec_shift
api.apply_lora = rec_lora
api.sample = rec_sample
api.decode_video = rec_decode_video
api.decode_audio = rec_decode_audio

# --------------------------------------------------------------------------
# 契约检查器：提示词的 <Picture i> ↔ 实际 ref_images 第 i 个位置
# --------------------------------------------------------------------------
def check_contract(shots, encode_calls):
    """返回违规列表。空列表 = 序号契约成立。"""
    violations = []
    for shot, call in zip(shots, encode_calls):
        ref_images = call.get("ref_images") or {}
        ordered = list(ref_images.values())          # ← 官方就是按这个顺序编 <Picture i>
        prompt = call.get("prompt") or ""

        def slot_of(tag: str):
            try:
                return int(tag.strip("<>").split()[1]) - 1
            except Exception:
                return None

        # 1) 场景锚点：<Picture 1> 必须是分镜格图
        if shot.board_image_index is not None:
            expect = PANEL_IMGS[shot.board_image_index]
            got = img_fingerprint(ordered[0]) if ordered else None
            if got != expect:
                violations.append(
                    f"[{shot.id}] <Picture 1> 应是分镜格#{shot.board_image_index}"
                    f"（指纹 {expect}），实际是 {got}"
                )
            if "follow <Picture 1>" not in prompt:
                violations.append(f"[{shot.id}] 提示词里没有引用分镜格锚点")

        # 2) 每个角色：它的 <Picture i> 必须指向它自己的定妆图
        expect_imgs = {A: CHAR_A_IMGS, B: CHAR_B_IMGS}
        for cid, tag in (shot.picture_tags or {}).items():
            idx = slot_of(tag)
            if idx is None:
                violations.append(f"[{shot.id}] 无法解析标签 {tag}")
                continue
            if idx >= len(ordered):
                violations.append(
                    f"[{shot.id}] {cid} 的 {tag} 越界（实际只传了 {len(ordered)} 张）"
                )
                continue
            got = img_fingerprint(ordered[idx])
            if got not in expect_imgs.get(cid, []):
                violations.append(
                    f"[{shot.id}] {cid} 的 {tag} 指向指纹 {got}，"
                    f"但它的定妆图是 {expect_imgs.get(cid)}"
                )
            # 标签必须真的出现在提示词里
            if tag not in prompt:
                violations.append(f"[{shot.id}] {cid} 的 {tag} 没写进提示词")

        # 3) 没出场的角色不能被传进去（per_shot 策略）
        for cid in (A, B):
            if cid not in (shot.picture_tags or {}):
                for v in expect_imgs[cid]:
                    if any(img_fingerprint(t) == v for t in ordered):
                        violations.append(
                            f"[{shot.id}] 没出场的角色 {cid} 的图（指纹 {v}）却被传进去了"
                        )
    return violations


# --------------------------------------------------------------------------
# 张量形状契约检查器：官方 R2V 按 [B,H,W,C] 读尺寸，3 维必炸
# --------------------------------------------------------------------------
def check_tensor_contract(encode_calls):
    """返回违规列表。空列表 = 交给官方节点的张量维度正确。

    官方源码按 ``h, w = img.shape[1], img.shape[2]`` 读参考图尺寸，
    也就是说它**假定** ``[B,H,W,C]``。3 维 ``[H,W,C]`` 会被读成
    「h = 宽、w = 通道数」，继而 resize 出畸形尺寸并在
    ``comfy/utils.py::common_upscale`` 里抛 reshape 错误（见文件末尾的复现）。
    """
    violations = []
    for call in encode_calls:
        for key, t in (call.get("ref_images") or {}).items():
            shape = tuple(getattr(t, "shape", ()))
            if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3:
                violations.append(f"参考图 {key} 形状 {shape}（官方要求 [1,H,W,3]）")
        for key, t in (call.get("ref_videos") or {}).items():
            shape = tuple(getattr(t, "shape", ()))
            if len(shape) != 4 or shape[-1] != 3:
                violations.append(f"参考视频 {key} 形状 {shape}（官方要求 [T,H,W,3]）")
    return violations


# --------------------------------------------------------------------------
section("正向：整集渲染（真跑 execute）")
# --------------------------------------------------------------------------
CALLS["encode"].clear()
out = render_mod.H3MangaEpisodeRender.execute(
    model="FAKE_MODEL", clip="FAKE_CLIP", vae="FAKE_VAE",
    plan=plan, refs=pack, board=board, audio_vae="FAKE_AUDIO_VAE",
    sampler_name="res_multistep", scheduler="simple", steps=8,
    shift_video=12.0, shift_audio=3.0, ref_image_size="match",
    lora_name="none", lora_strength=1.0, base_seed=1234,
    max_shots=0, skip_done_ids="", on_error="skip",
)
frames_out, audio_out, report = out.result

ok(len(CALLS["encode"]) == 3, f"逐镜调用官方 R2V 共 3 次（实得 {len(CALLS['encode'])}）")
ok(len(CALLS["sample"]) == 3, f"逐镜采样 3 次（实得 {len(CALLS['sample'])}）")
ok(len(CALLS["shift"]) == 1, "shift 补丁只做一次（不在循环里反复 clone）")

viol = check_contract(plan.shots, CALLS["encode"])
ok(not viol, f"★ 序号契约成立：提示词的 <Picture i> 指向正确的图（违规 {len(viol)} 条）")
for v in viol:
    print(f"      ! {v}")

tviol = check_tensor_contract(CALLS["encode"])
ok(not tviol,
   f"★ 张量形状契约成立：参考图/视频都是 4 维 [B,H,W,C]（违规 {len(tviol)} 条）")
for v in tviol:
    print(f"      ! {v}")

# 逐镜细节
for i, (shot, call) in enumerate(zip(plan.shots, CALLS["encode"])):
    keys = list((call.get("ref_images") or {}).keys())
    print(f"  [{shot.id}] ref_images={keys}  tags={shot.picture_tags}")

ok(list((CALLS["encode"][0].get("ref_images") or {}).keys())
   == [f"ref_image_{i}" for i in range(4)],
   "s01：分镜格 + 纲吉的 2 张图 + 风格图（连续键名）")
ok(list((CALLS["encode"][1].get("ref_images") or {}).keys())
   == ["ref_image_0", "ref_image_1", "ref_image_2"],
   "s02：Reborn 那一镜仍然连续（面板 + Reborn + 风格），说明重排生效")

# 参数传递
ok(all(c["width"] % 32 == 0 and c["height"] % 32 == 0 for c in CALLS["encode"]),
   "所有镜头画幅都是 32 的倍数")
ok(all(c["length"] % 17 == 5 for c in CALLS["encode"]),
   "所有镜头帧数都是 17n+5")
ok(all(c["ref_image_size"] == "match" for c in CALLS["encode"]), "ref_image_size 透传")
ok(CALLS["encode"][0]["prompt"].strip() != "", "提示词非空透传")

# seed 必须等于任务展开时算出来的 seed
tasks = pl.build_render_tasks(plan, pack, base_seed=1234)
ok([c["seed"] for c in CALLS["sample"]] == [t["seed"] for t in tasks],
   "采样 seed 与任务计划的 seed 完全一致")

# 输出拼接
ok(int(frames_out.shape[0]) == 15, f"帧序列拼接为 3 镜 × 5 帧 = 15（实得 {frames_out.shape[0]}）")
ok(audio_out is not None and int(audio_out["waveform"].shape[-1]) == 120,
   f"音频拼接为 3 × 40 = 120（实得 {None if audio_out is None else audio_out['waveform'].shape[-1]}）")
ok(audio_out and audio_out["sample_rate"] == 32000, "音频采样率透传")

rep = json.loads(report.split("\n\n失败镜头")[0])
ok(rep["done_shots"] == 3 and rep["failed_shots"] == 0,
   f"渲染报告：3 成功 0 失败（实得 {rep['done_shots']}/{rep['failed_shots']}）")
ok(len(rep["done"][0]["picture_tags"]) >= 1, "报告里带每镜的参考标签（可追溯）")
ok(rep["preflight"]["ok"] is True, "预检通过")

# --------------------------------------------------------------------------
section("单镜失败降级（on_error=skip）")
# --------------------------------------------------------------------------
orig_encode = api.encode_reference_to_video
_flaky_counter = {"n": 0}


def flaky_encode(**kw):
    _flaky_counter["n"] += 1
    if _flaky_counter["n"] == 2:      # 第 2 镜（s02）必定失败，确定性
        raise api.H3ApiError("模拟：这一镜采样崩了")
    return orig_encode(**kw)


api.encode_reference_to_video = flaky_encode
out2 = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae=None, steps=4, base_seed=1, max_shots=0, skip_done_ids="",
    on_error="skip",
)
frames2, audio2, report2 = out2.result
rep2 = json.loads(report2.split("\n\n失败镜头")[0])
ok(rep2["done_shots"] == 2 and rep2["failed_shots"] == 1,
   f"单镜失败被跳过，其余继续跑（done={rep2['done_shots']} failed={rep2['failed_shots']}）")
ok([f["id"] for f in rep2["failed"]] == ["s02"], "报告指明是哪一镜失败")
ok(int(frames2.shape[0]) == 10, f"失败镜不产生帧（10 帧，实得 {frames2.shape[0]}）")
ok("模拟：这一镜采样崩了" in report2, "失败原因写进报告（用户可定位）")
api.encode_reference_to_video = orig_encode

# --------------------------------------------------------------------------
section("断点续跑：skip_done_ids")
# --------------------------------------------------------------------------
CALLS["encode"].clear()
out3 = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae=None, steps=4, base_seed=1234, max_shots=0,
    skip_done_ids="s02", on_error="skip",
)
ok(len(CALLS["encode"]) == 2, f"跳过 1 镜后只渲染 2 镜（实得 {len(CALLS['encode'])}）")
rep3 = json.loads(out3.result[2].split("\n\n失败镜头")[0])
ok(rep3["scheduled"] == 2, "报告里的 scheduled 反映了跳过")

# --------------------------------------------------------------------------
section("★ 分阶段卸载：编码 / 采样 / 解码各自成段，段末卸载对应模型")
# --------------------------------------------------------------------------
# 本机权重总量 33.5GB（R2V UNET 19.53 + 文本编码器 13.93）而内存只有 16GB，
# 三个模型同时驻留必然被 OOM Killer 杀掉（表现为 ComfyUI 自动重启）。
# 所以「分阶段 + 用完即卸」不是优化，是能不能跑通的前提。
CALLS["encode"].clear()
out_mem = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae="A", steps=4, base_seed=1234, max_shots=0, skip_done_ids="",
    on_error="skip", unload_stages=True,
)
rep_mem = json.loads(out_mem.result[2].split("\n\n失败镜头")[0])
stages_mem = rep_mem["memory"]["stages"]
labels_on = [s["label"] for s in stages_mem]
ok(rep_mem["memory"]["unload_stages"] is True, "报告里记录了 unload_stages=True")
ok(labels_on and labels_on[0] == "起点" and labels_on[-1] == "结束",
   f"资源曲线首尾是起点/结束：{labels_on}")
ok(any("阶段A后" in l for l in labels_on), f"阶段A（编码）后走了卸载分支：{labels_on}")
ok(any("阶段B后" in l for l in labels_on), "阶段B（采样）后走了卸载分支")
ok(any("阶段C后" in l for l in labels_on), "阶段C（解码）后走了卸载分支")
ok(rep_mem["memory"]["lines"] and all(isinstance(x, str) and x for x in rep_mem["memory"]["lines"]),
   "每条曲线都有可读文本（节点面板直接显示）")
ok(rep_mem["params"]["unload_stages"] is True, "参数回写进报告，便于比对两次运行")

# 反向验证：关掉开关后必须**看不到卸载**，否则说明这个开关是摆设
out_nomem = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae="A", steps=4, base_seed=1234, max_shots=0, skip_done_ids="",
    on_error="skip", unload_stages=False,
)
rep_off = json.loads(out_nomem.result[2].split("\n\n失败镜头")[0])
labels_off = [s["label"] for s in rep_off["memory"]["stages"]]
ok(rep_off["memory"]["unload_stages"] is False, "关闭后报告状态为 False")
_stage_off = [l for l in labels_off if "阶段" in l]
ok(_stage_off and all("未卸载" in l for l in _stage_off),
   f"★ 反向验证：关掉开关后各阶段标注「未卸载」而不是假装卸了：{_stage_off}")
ok(any("已关闭" in w for w in rep_off["preflight"]["warnings"]),
   "预检提示了「关闭分阶段卸载」的风险")

# --------------------------------------------------------------------------
section("★ 稀疏注意力：只有显式开启才应用，参数如实透传，失败必须降级")
# --------------------------------------------------------------------------
# 官方 BlockSparseAttention 的三件事必须被锁住：
#   ① 默认不能偷偷启用（行为要与以前完全一致）
#   ② 参数要如实透传（尤其 sink_conditioning —— 它决定参考图会不会被稀疏化）
#   ③ 失败只能降级，绝不能把渲染搞挂（它是可选优化，不是正确性依赖）
SPARSE_CALLS = []


def rec_sparse(model, method="none", **kw):
    SPARSE_CALLS.append({"model": model, "method": method, **kw})
    return model, {"applied": method != "none", "method": method, "reason": "已应用"}


_orig_sparse = api.apply_sparse_attention
api.apply_sparse_attention = rec_sparse

SPARSE_CALLS.clear()
out_sp0 = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae=None, steps=4, base_seed=1234, max_shots=1, on_error="skip",
)
ok(len(SPARSE_CALLS) == 0, f"默认 sparse_attn=none 时完全不调用（实得 {len(SPARSE_CALLS)}）")
rep_sp0 = json.loads(out_sp0.result[2].split("\n\n失败镜头")[0])
ok(rep_sp0["sparse_attention"]["applied"] is False, "报告里如实记为未应用")

SPARSE_CALLS.clear()
out_sp1 = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae=None, steps=4, base_seed=1234, max_shots=1, on_error="skip",
    sparse_attn="sol-attn", sparse_tau=1.5, sparse_keep_percent=7.0,
    sparse_start_percent=0.3, sparse_min_tokens=8192, sparse_sink="exact_kv",
)
ok(len(SPARSE_CALLS) == 1, f"开启后恰好调用一次（实得 {len(SPARSE_CALLS)}）")
_sc = SPARSE_CALLS[0]
ok(_sc["method"] == "sol-attn", "method 透传")
ok(abs(_sc["tau"] - 1.5) < 1e-9, "tau 透传")
ok(int(_sc["min_tokens"]) == 8192, "min_tokens 透传")
ok(abs(_sc["start_percent"] - 0.3) < 1e-9, "start_percent 透传")
ok(_sc["sink_conditioning"] == "exact_kv",
   "★ sink_conditioning 透传（它决定参考图行会不会被稀疏化）")
rep_sp1 = json.loads(out_sp1.result[2].split("\n\n失败镜头")[0])
ok(rep_sp1["sparse_attention"]["method"] == "sol-attn", "报告记录方式")
ok(rep_sp1["sparse_attention"]["applied"] is True, "报告记录已应用")


def rec_sparse_fail(model, method="none", **kw):
    return model, {"applied": False, "method": method, "reason": "模拟：内核不可用"}


api.apply_sparse_attention = rec_sparse_fail
out_sp2 = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae=None, steps=4, base_seed=1234, max_shots=1, on_error="skip",
    sparse_attn="vsa",
)
rep_sp2 = json.loads(out_sp2.result[2].split("\n\n失败镜头")[0])
ok(rep_sp2["sparse_attention"]["applied"] is False, "失败时报告如实记为未应用")
ok(any("稀疏注意力未生效" in w for w in rep_sp2["preflight"]["warnings"]),
   "★ 失败时预检给出警告（而不是静默降级让用户以为开了）")
ok(len(CALLS["sample"]) > 0, "★ 稀疏失败后采样照常进行（可选优化不该阻断渲染）")

# ★ min_tokens 门槛：704x384 的 5 秒镜头只有 ≈9768 token，低于官方默认 12288，
#   官方会**静默**走 dense —— 必须提前提醒，否则用户会以为「开了没用」是插件的问题。
#   （注意不能用 768x432：432 不是 32 的倍数，会被 align_canvas 吸附成 448，
#     而 768x448 的 5 秒镜头是 12432，刚好过线、反而不会触发提醒。）
_plan_small = sp.parse_text(
    "=== s1 ===\npanel: 0\nduration: 5\nsize: 704x384\n"
    "characters: 纲吉\naction: @纲吉 走路\n"
)
pl.bind_board(_plan_small, board, mode="by_panel_index")
pb.compose_plan(_plan_small, pack)
out_sp3 = render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=_plan_small, refs=pack, board=board,
    audio_vae=None, steps=4, base_seed=1234, max_shots=1, on_error="skip",
    sparse_attn="sol-attn",
)
rep_sp3 = json.loads(out_sp3.result[2].split("\n\n失败镜头")[0])
ok(any("低于 sparse_min_tokens" in w for w in rep_sp3["preflight"]["warnings"]),
   "★ 预检提醒「token 数低于 min_tokens 门槛，开了也不加速」")
ok(any("960×544" in w for w in rep_sp3["preflight"]["warnings"]),
   "并给出可执行的建议（提到画幅或 min_tokens）")

api.apply_sparse_attention = _orig_sparse

# --------------------------------------------------------------------------
section("★ 反向验证：把序号规则改坏，契约检查必须变红")
# --------------------------------------------------------------------------
# 如果这条测试不变红，说明上面的 check_contract 是空转的 —— 测试无效。
CALLS["encode"].clear()

# 临时把 shot_ref_layout 换成「全局池顺序」的错误实现：
# 这正是「以为角色 A 全局是 <Picture 3>」那种天真写法。
_orig_layout = rf.shot_ref_layout
_global_slots = pack.resolve_slots()
_global_order = [it["id"] for it in _global_slots["flat"]]


def broken_layout(refs, characters=(), panel_image=None, style_ids=None,
                  max_images=0, policy="per_shot", panel_id="", has_panel=False):
    """错误实现：不管本镜出场谁，都按全局池顺序塞进去。"""
    ref_images = {}
    picture_map = {}
    order = []
    slot = 0
    pool = {e.id: e for e in refs.entries}
    for eid in _global_order:
        imgs = [panel_image] if eid == refs.panel_id else list(pool[eid].images) if eid in pool else []
        for img in imgs:
            if img is None or slot >= 9:
                continue
            ref_images[f"ref_image_{slot}"] = img
            slot += 1
            picture_map.setdefault(eid, []).append(slot)
            order.append({"slot": slot - 1, "tag": f"<Picture {slot}>", "id": eid, "kind": "x"})
    return {"ref_images": ref_images, "ref_image_keys": list(ref_images.keys()),
            "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {},
            "picture_map": picture_map, "video_map": {}, "audio_map": {}, "order": order,
            "dropped": [], "warnings": [], "resolved_characters": list(characters),
            "unknown_characters": []}


# B1：内置守卫必须拦下（现在会直接抛错，而不是返回空批次）
rf.shot_ref_layout = broken_layout
raised = None
try:
    render_mod.H3MangaEpisodeRender.execute(
        model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
        audio_vae=None, steps=4, base_seed=1234, max_shots=0, skip_done_ids="",
        on_error="skip",
    )
except Exception as e:
    raised = e
finally:
    rf.shot_ref_layout = _orig_layout

ok(raised is not None, "★ 序号规则被改坏后，渲染节点**直接抛错**（不再返回空批次）")
ok(raised is not None and "装配后被改动过" in str(raised),
   "★ 守卫给出的原因可读（提示重新跑提示词装配）")
ok(raised is not None and "没有任何镜头渲染成功" in str(raised),
   "★ 全失败时给出的是「整集无成功镜头」而不是下游的莫名其妙报错")

# B2：契约检查器自检 —— 直接把一次调用的 ref_images 顺序打乱，必须报违规。
#     没有这一步，无法区分「测试通过」和「测试在空转」。
CALLS["encode"].clear()
render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
    audio_vae=None, steps=4, base_seed=1234, max_shots=0, skip_done_ids="",
    on_error="skip",
)
ok(not check_contract(plan.shots, CALLS["encode"]), "正确布局下契约检查 0 违规")

tampered = [dict(c) for c in CALLS["encode"]]
tampered[0]["ref_images"] = dict(reversed(list(tampered[0]["ref_images"].items())))
viol_self = check_contract(plan.shots, tampered)
ok(len(viol_self) > 0,
   f"★ 契约检查器自检：故意打乱 ref_images 顺序必须报违规（实得 {len(viol_self)} 条）")
print(f"      {viol_self[0] if viol_self else ''}")

# --------------------------------------------------------------------------
section("单镜条件节点（调试路径）")
# --------------------------------------------------------------------------
CALLS["encode"].clear()
sc = render_mod.H3MangaShotConditioning.execute(
    clip="C", plan=plan, refs=pack, board=board, shot_index=2,
    vae="V", audio_vae=None, ref_image_size="match",
)
pos, lat, rep_txt = sc.result
ok(len(CALLS["encode"]) == 1, "单镜条件只调用一次 R2V")
ok("单镜条件" in rep_txt and "17n+5" in rep_txt, "单镜报告含关键信息")
ok("本镜参考图布局" in rep_txt, "单镜报告含逐镜参考布局表")
ok("seed" in rep_txt and "渲染节点的 base_seed 会覆盖它" in rep_txt,
   "单镜报告的 seed 不再固定显示 0")

# --------------------------------------------------------------------------
section("参考视频 / 参考音频真的送到官方节点")
# --------------------------------------------------------------------------
# 早先 audios / videos 是死输入：节点上有，但 shot_ref_layout 不带，
# 渲染节点也不传，等于白接。这条测试锁住「接了就一定传下去」。
aud = {"waveform": torch.zeros((1, 2, 64), dtype=torch.float32), "sample_rate": 32000}
video_clip = torch.zeros((8, 8, 8, 3), dtype=torch.float32)   # 8 帧
pack_av = rf.build_ref_pack(
    character_names="纲吉", character_images=[make_img(CHAR_A_IMGS[0])],
    audio_names="bgm", audios=[aud],
    video_names="动作参考", videos=[video_clip], video_audios=[aud],
)
plan_av = sp.parse_text("=== s1 ===\npanel: 0\nduration: 5\ncharacters: 纲吉\n"
                        "action: @纲吉 走路\n")
pl.bind_board(plan_av, board, mode="by_panel_index")
pb.compose_plan(plan_av, pack_av)

CALLS["encode"].clear()
render_mod.H3MangaEpisodeRender.execute(
    model="M", clip="C", vae="V", plan=plan_av, refs=pack_av, board=board,
    audio_vae=None, steps=4, base_seed=7, max_shots=1, on_error="skip",
)
c = CALLS["encode"][0]
ok(list(c["ref_videos"].keys()) == ["ref_video_0"], "参考视频传到了官方 R2V")
ok(list(c["ref_video_audios"].keys()) == ["ref_video_audio_0"], "视频配套音轨同号传到")
ok(list(c["ref_audios"].keys()) == ["ref_audio_0"], "独立参考音频传到")
ok(c["ref_videos"]["ref_video_0"].shape[0] == 8,
   "视频整段传入（8 帧，没有被拆成 8 个单帧槽位）")
ok(list(c["ref_images"].keys()) == ["ref_image_0", "ref_image_1"],
   f"图片槽位不受视频/音频影响：{list(c['ref_images'].keys())}")

# --------------------------------------------------------------------------
section("预检拦截：提示词为空")
# --------------------------------------------------------------------------
plan_empty = sp.parse_text("=== s1 ===\npanel: 0\nduration: 5\ncharacters: 纲吉\naction: 走路\n")
pl.bind_board(plan_empty, board, mode="by_panel_index")
# 故意不跑 compose_plan → prompt 为空（等价于用户漏接「提示词装配」）
pf = render_mod.H3MangaEpisodeRender._preflight(plan_empty, pack, board)
ok(pf["ok"] is False, "预检判定不通过")
ok(any("提示词是空的" in w for w in pf["warnings"]),
   "预检报出「提示词是空的」，并提示检查装配节点接线")

# 碎条分镜格（切格偶尔切出 1~几像素条带）：作为 <Picture 1> 没有画面信息，要提前说
board_thin = types_.Board(
    panels=[types_.Panel(index=0, box=[0, 0, 1152, 1], page_index=0),
            types_.Panel(index=1, box=[0, 0, 100, 100], page_index=0)],
    images=[make_img(1), make_img(2)], pages=1,
)
pf_thin = render_mod.H3MangaEpisodeRender._preflight(plan, pack, board_thin)
ok(any("碎条" in w for w in pf_thin["warnings"]),
   f"预检报出碎条分镜格：{[w for w in pf_thin['warnings'] if '碎条' in w][:1]}")

# --------------------------------------------------------------------------
section("★ 复现线上事故：3 维参考图 → common_upscale 里的畸形 reshape")
# --------------------------------------------------------------------------
# 事故原文（ComfyUI 0.38.2，节点 15 H3MangaEpisodeRender）：
#   H3ApiError: 官方 MiniMaxH3ReferenceToVideo 执行失败：
#   RuntimeError: shape '[1, -1, 3, 1152, 32]' is invalid for input of size 36864
# 根因：分镜格是 3 维 [H,W,C]（漫画页宽 1152），官方按 [B,H,W,C] 读成
#       h=1152（其实是宽）、w=3（其实是通道数）→ 目标尺寸算成 1152x32。
# 下面把这条链路**逐字复刻**，让它在测试里可以被断言（而不是靠读日志猜）。
import math  # noqa: E402


def official_ref_resize(img, width, height, ref_image_size="match"):
    """复刻官方 R2V 对一张参考图的**形状**处理（不做真实插值，只做维度记账）。

    刻意保留两处真实的坑：
      1. ``h, w = img.shape[1], img.shape[2]`` —— 写死了 [B,H,W,C] 的前提
      2. ``comfy/utils.py::common_upscale`` 在 ``len(orig_shape) != 4`` 时走
         ``out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))``
    形状对不上时抛 torch 的真实 RuntimeError（文案与线上完全一致）。
    """
    h, w = int(img.shape[1]), int(img.shape[2])
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / float(w * h)))
    else:
        scale = min(1.0, 2048.0 / float(min(w, h)))
    tw = max(32, int(round(w * scale / 32.0)) * 32)
    th = max(32, int(round(h * scale / 32.0)) * 32)

    samples = img[:1][..., :3].movedim(-1, 1)             # _resize(img[:1], ...) 第一步
    orig_shape = tuple(int(v) for v in samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(orig_shape[0], orig_shape[1], -1,
                                  orig_shape[-2], orig_shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])

    # lanczos 的形状效果：逐张 PIL resize 后 stack（通道维跟着首维走）
    if samples.dim() == 4:
        out = torch.zeros(samples.shape[0], samples.shape[1], th, tw)
    elif samples.dim() == 3:
        out = torch.zeros(samples.shape[0], th, tw)
    else:
        out = torch.zeros(th, tw)

    if len(orig_shape) == 4:
        return tuple(out.shape)
    return tuple(out.reshape((orig_shape[0], -1, orig_shape[1]) + (th, tw)).shape)


# 事故里那张图：一张很普通的 3 维分镜格 [H,W,3]（漫画页宽 1152）。
# 官方 `_resize` 会先取 img[:1]，所以首维变成 1 → 报错文本里的 36864 = 1×1152×32。
panel_3d = torch.zeros((768, 1152, 3), dtype=torch.float32)
_err = None
try:
    official_ref_resize(panel_3d, 1344, 768)
except Exception as e:      # noqa: BLE001
    _err = e
ok(_err is not None
   and "shape '[1, -1, 3, 1152, 32]'" in str(_err)
   and "36864" in str(_err),
   f"★ 复刻出线上原文（3 维 → {_err}）")

# 同一类错：任何 3 维 [H,W,C] 都会炸（只是数字不同），所以必须统一升维
_err_narrow = None
try:
    official_ref_resize(torch.zeros((64, 512, 3), dtype=torch.float32), 1344, 768)
except Exception as e:      # noqa: BLE001
    _err_narrow = e
ok(_err_narrow is not None and "is invalid for input of size" in str(_err_narrow),
   f"★ 同一类错：其它尺寸的 3 维图也会炸（{_err_narrow}）")

# 修复后的路径：shot_ref_layout / h3_api 已升成 4 维 → 目标尺寸按真实 H/W 走
fixed = rf.as_ref_image(panel_3d)
ok(tuple(fixed.shape) == (1, 768, 1152, 3),
   f"3 维 [H,W,C] → 4 维 [1,H,W,C]（实得 {tuple(fixed.shape)}）")
ok(official_ref_resize(fixed, 1344, 768) == (1, 3, 768, 1152),
   "升维后官方算出的目标是 768x1152（不再是 1152x32 这种畸形值）")

# 反向验证：把升维关掉（= 修复前的行为），契约检查必须变红 ——
# 没有这一步，无法区分「测试通过」和「测试在空转」。
_orig_as_ref_image = rf.as_ref_image
_orig_as_ref_video = rf.as_ref_video
rf.as_ref_image = lambda t, notes=None, where="": t
rf.as_ref_video = lambda t, notes=None, where="": t
try:
    CALLS["encode"].clear()
    render_mod.H3MangaEpisodeRender.execute(
        model="M", clip="C", vae="V", plan=plan, refs=pack, board=board,
        audio_vae=None, steps=4, base_seed=1234, max_shots=1,
        skip_done_ids="", on_error="skip",
    )
    _bad_calls = [dict(c) for c in CALLS["encode"]]
finally:
    rf.as_ref_image = _orig_as_ref_image
    rf.as_ref_video = _orig_as_ref_video

tv_bad = check_tensor_contract(_bad_calls)
ok(len(tv_bad) > 0,
   f"★ 反向验证：关掉升维后张量契约必须报违规（实得 {len(tv_bad)} 条）")
if tv_bad:
    print(f"      {tv_bad[0]}")

_first_bad = list(_bad_calls[0]["ref_images"].values())[0]
_err2 = None
try:
    official_ref_resize(_first_bad, 1344, 768)
except Exception as e:      # noqa: BLE001
    _err2 = e
ok(_err2 is not None and "is invalid for input of size" in str(_err2),
   f"★ 反向验证：未升维的参考图喂给官方逻辑仍然抛 reshape 错（{_err2}）")

# 单镜调试节点也吃同一套契约（它走的是同一个 api.encode_reference_to_video）
CALLS["encode"].clear()
render_mod.H3MangaShotConditioning.execute(
    clip="C", plan=plan, refs=pack, board=board, shot_index=0,
    vae="V", audio_vae=None, ref_image_size="match",
)
ok(not check_tensor_contract(CALLS["encode"]),
   "单镜条件节点的参考图同样是 4 维")

# ==========================================================================
section("★ VLM 层：⑬/⑭ 端到端契约（格号必须与 ① 的 board 全局号一致）")
# ==========================================================================
# 这一层抓的是**静默错**：① 的 Panel.index 是整批全局递增的，④ 用
# panel_index 去索引 board.panels。若 ⑬ 只按「本页」编号（0..k-1），
# 第 2 页起就会绑到错误的格图 —— 不报错，只是画面不对。
# 所以这里直接**反查 VLM 收到的提示词**：第 2 页的格号必须是 3..8 而不是 0..5。
import numpy as _np  # noqa: E402

vlm_mod = importlib.import_module(pkg.__name__ + ".core.vlm")
vlm_nodes = importlib.import_module(pkg.__name__ + ".nodes.vlm_nodes")
types_mod = core.types

PROMPTS = []
BOX_IDS = []


def _fake_call_vlm(opener, base_url, api_key, model, sys_prompt, user_text,
                   b64, mime, timeout=180, retries=3, temperature=0.2,
                   max_tokens=4096):
    """桩：不联网，按提示词里给的格号造结果（故意倒序，验证代码会排序）。"""
    import re as _re
    PROMPTS.append(user_text)
    ids = [int(m) for m in _re.findall(r"格 (\d+):", user_text)]
    BOX_IDS.append(ids)
    panels = []
    for i in ids:
        panels.append({
            "panel": i,
            # 只让 panel 0 判成 2 镜（便于精确核对总镜数：9 格 + 1 = 10 镜）
            "shots": 2 if i == 0 else 1,
            "camera": "static", "content": f"格{i}的画面内容",
            "duration_hint": 8, "characters": [],
        })
    obj = {"page_kind": "narrative", "panels": list(reversed(panels)),
           "summary": "summary"}
    return json.dumps(obj, ensure_ascii=False), {"usage": {"total_tokens": 10}}


_orig_call_vlm = vlm_mod.call_vlm
vlm_mod.call_vlm = _fake_call_vlm

# ---- 造 board：第 0 页 3 格、第 1 页 6 格（全局号 0..8）----
_bd_panels, _gi = [], 0
for _pi, _n in enumerate((3, 6)):
    for _k in range(_n):
        _bd_panels.append(types_mod.Panel(index=_gi, box=[0, 0, 10, 10],
                                          page_index=_pi))
        _gi += 1
_vlm_board = types_mod.Board(panels=_bd_panels, pages=2, page_size=(100, 100))
ALL_IDS = set(p.index for p in _vlm_board.panels)
ok(ALL_IDS == set(range(9)), f"board 全局格号 0..8（实得 {sorted(ALL_IDS)}）")

_img2 = _np.full((2, 100, 100, 3), 0.5, dtype=_np.float32)
_vout = vlm_nodes.H3MangaVLMScribe.execute(
    image=_img2, board=_vlm_board, characters="纲吉", char_alias="",
    base_url="http://stub/v1", model="stub-model", api_key="k",
    api_key_env="X", max_edge=0, temperature=0.2, timeout=10, max_pages=0,
    strict_coverage=True, inspection_fallback=False, use_proxy=False)
_vres, _vrep = _vout.result[0], _vout.result[1]

ok(len(PROMPTS) == 2, f"两页各调一次 VLM（实得 {len(PROMPTS)}）")
ok(BOX_IDS[0] == [0, 1, 2],
   f"第 1 页提示词里的格号 = 0,1,2（实得 {BOX_IDS[0]}）")
ok(BOX_IDS[1] == [3, 4, 5, 6, 7, 8],
   f"★ 第 2 页提示词里是**全局号** 3..8（实得 {BOX_IDS[1]}）"
   " —— 若为 0..5 说明没换算，会绑错格图")
ok("共 6 格" in PROMPTS[1] or "共 3 格" in PROMPTS[0], "提示词里写了本页格数")
ok("纲吉" in PROMPTS[0], "角色闭集写进了提示词")

ok([p["panel"] for p in _vres.pages[0]["panels"]] == [0, 1, 2],
   "第 1 页结果按 panel 排序（桩故意倒序）")
ok([p["panel"] for p in _vres.pages[1]["panels"]] == [3, 4, 5, 6, 7, 8],
   "★ 第 2 页结果的 panel 号 = 3..8（全局号）")
_used = set(p["panel"] for pg in _vres.pages for p in pg["panels"])
ok(_used <= ALL_IDS,
   "★ 端到端契约：⑬ 产出的 panel 号全部落在 board.panels 的 index 集合内"
   "（这是 ④ 能正确绑定的前提）")
ok("格号来自 ① 的 board（全局号）" in _vrep, "报告写明格号来源")
ok("narrative" in _vrep, "报告含页型分布")

# ---- ⑭ → Plan 契约 ----
_vout2 = vlm_nodes.H3MangaVLMPlan.execute(
    vlm=_vres, include_non_narrative=False, force_shots=0, duration_scale=1.0,
    default_duration=5.0, size_mode="preset", size_preset="16:9 横屏",
    size_scale=0.5, episode="e", style="manga_anime")
_plan = _vout2.result[0]
_ptext, _prep = _vout2.result[1], _vout2.result[2]
ok(len(_plan.shots) == 10,
   f"⑭ → Plan：9 格 + 1 个多镜展开 = 10 镜（实得 {len(_plan.shots)}）")
ok(all(s.length % 17 == 5 for s in _plan.shots),
   "帧数满足 17n+5（官方硬约束）")
ok(all(s.width % 32 == 0 and s.height % 32 == 0 for s in _plan.shots),
   "画幅是 32 的倍数")
ok((_plan.shots[0].width, _plan.shots[0].height) == (672, 384),
   f"画幅 = 16:9 官方规范 × 0.5 = 672x384（实得 "
   f"{_plan.shots[0].width}x{_plan.shots[0].height}）")
ok(all(s.panel_index in ALL_IDS for s in _plan.shots),
   "★ 每个 shot 的 panel_index 都能在 board 里找到（④ 绑定不会退化）")
_double = [s for s in _plan.shots if s.panel_index == 0]
ok(len(_double) == 2 and len(set(s.panel_index for s in _double)) == 1,
   "★ 一格多镜：展开出的两镜共用同一 panel_index（绑同一张格图当锚点）")
ok("=== page000_00 ===" in _ptext and "shots: 2" in _ptext,
   "⑭ 的第 2 个输出是块文本，shots 已写入")

# force_shots 覆盖
_vout3 = vlm_nodes.H3MangaVLMPlan.execute(
    vlm=_vres, include_non_narrative=False, force_shots=1, duration_scale=0.5,
    default_duration=5.0, size_mode="preset", size_preset="16:9 横屏",
    size_scale=0.5, episode="e", style="manga_anime")
ok(len(_vout3.result[0].shots) == 9,
   f"★ force_shots=1：从 10 镜回到 9 镜（实得 {len(_vout3.result[0].shots)}）")
ok(all(s.sub_total == 1 for s in _vout3.result[0].shots),
   "force_shots=1 后没有一格多镜")
ok(abs(_vout3.result[0].shots[0].duration - 4.46) < 0.3,
   f"★ duration_scale=0.5：8s → 对齐后 "
   f"{_vout3.result[0].shots[0].duration:.2f}s（应约 4.46s）")

# ---- 不接 board：自切格，编号从 0 连续 ----
_page2x2 = _np.full((600, 400, 3), 1.0, dtype=_np.float32)
_page2x2[20:270, 20:180] = 0.0
_page2x2[20:270, 220:380] = 0.0
_page2x2[330:580, 20:180] = 0.0
_page2x2[330:580, 220:380] = 0.0
BOX_IDS.clear()
_vout4 = vlm_nodes.H3MangaVLMScribe.execute(
    image=_page2x2[None, ...], board=None, characters="", char_alias="",
    base_url="http://stub/v1", model="stub-model", api_key="k",
    api_key_env="X", max_edge=0, temperature=0.2, timeout=10, max_pages=0,
    strict_coverage=True, inspection_fallback=False, use_proxy=False)
ok(BOX_IDS and BOX_IDS[0] == [0, 1, 2, 3],
   f"不接 board → 自切格 4 格，编号 0..3（实得 {BOX_IDS[0] if BOX_IDS else None}）")
ok("格号由本节点自切格算出（未接 board）" in _vout4.result[1],
   "报告写明未接 board 时的格号来源")

# ---- 缺 API key 必须报错（不能静默跑）----
_err_key = None
try:
    vlm_nodes.H3MangaVLMScribe.execute(
        image=_img2, board=_vlm_board, characters="", char_alias="",
        base_url="http://stub/v1", model="m", api_key="",
        api_key_env="__NO_SUCH_ENV_VAR__", max_edge=0, temperature=0.2,
        timeout=10, max_pages=1, strict_coverage=True,
        inspection_fallback=False, use_proxy=False)
except Exception as e:      # noqa: BLE001
    _err_key = e
ok(_err_key is not None and "缺 API key" in str(_err_key),
   "★ 反向验证：缺 API key 时报明确错误（不静默重试）")

vlm_mod.call_vlm = _orig_call_vlm

print(f"\n{'=' * 60}\n第 4 层契约测试：{PASS} 通过 / {FAIL} 失败\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
