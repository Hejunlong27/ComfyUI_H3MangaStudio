"""提示词装配：把「分镜字段 + 参考包序号」编译成 H3 可直接吃的提示词。

H3 提示词结构（官方 h3-prompt-writing 思路）：

    [主体/参考锁定] + [动作/状态] + [镜头语言] + [光影氛围] + [风格/画质] + [对白]

本模块的三条硬规则
------------------
1. **序号唯一真源**：``@角色id`` 只在**这里**被替换成 ``<Picture i>``。
   节点层不拼标签、不猜序号，全部走 ``RefPack.resolve_slots()``。
2. **未命中的引用必须可见**：分镜里写了 ``@tsuna`` 但参考包里没有 tsuna，
   不能静默留成字面量 ``@tsuna``（模型会把它当普通文本，结果不可控）。
   这里会替换成**最近似的角色标签**或删掉，并记一条 warning。
3. **中文台词不翻译**：H3 原生支持中文对白，直接引号包裹交给模型，
   不要机器翻译（会毁掉口型与语气）。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

try:
    from .types import Plan, TAG_PICTURE, TAG_VIDEO, TAG_AUDIO, RefPack
    from .refs import shot_ref_layout, resolve_character_ids
except ImportError:  # pragma: no cover
    from types import Plan, TAG_PICTURE, TAG_VIDEO, TAG_AUDIO, RefPack
    from refs import shot_ref_layout, resolve_character_ids


# --------------------------------------------------------------------------
# 运镜预设：分镜里写短名，这里展开成 H3 吃得懂的镜头语言
# 用户可在 templates/camera_presets.json 里追加/覆盖
# --------------------------------------------------------------------------
CAMERA_PRESETS: Dict[str, str] = {
    "static": "static locked-off shot, camera does not move",
    "slow_push_in": "slow push in, camera dollies toward the subject",
    "push_in": "push in toward the subject",
    "pull_back": "slow pull back, revealing the surroundings",
    "pan_left": "smooth pan to the left",
    "pan_right": "smooth pan to the right",
    "tilt_up": "slow tilt up",
    "tilt_down": "slow tilt down",
    "tracking": "tracking shot following the subject",
    "orbit": "slow orbit around the subject",
    "handheld": "subtle handheld camera, slight natural shake",
    "crane_up": "crane shot rising upward",
    "dolly_zoom": "dolly zoom, background perspective shifts",
    "whip_pan": "fast whip pan",
    "over_shoulder": "over-the-shoulder shot",
    "close_up": "close-up shot, shallow depth of field",
    "extreme_close_up": "extreme close-up, macro detail",
    "medium_shot": "medium shot, waist up",
    "full_shot": "full shot, whole body in frame",
    "wide_shot": "wide establishing shot",
    "low_angle": "low angle looking up, dramatic",
    "high_angle": "high angle looking down",
    "dutch_angle": "dutch angle, tilted horizon",
    "pov": "first-person point of view shot",
}

# --------------------------------------------------------------------------
# 风格预设：漫改动画是主场景，同时给真人/赛璐璐等备选
# --------------------------------------------------------------------------
STYLE_PRESETS: Dict[str, str] = {
    "manga_anime": (
        "Japanese TV anime look, clean cel-shaded animation, crisp line art, "
        "two-tone shading, vivid saturated colors, 2000s shonen anime aesthetic"
    ),
    "manga_anime_modern": (
        "modern Japanese anime, crisp digital cel shading, soft rim light, "
        "detailed background art, cinematic composition, 24fps animation feel"
    ),
    "manga_anime_film": (
        "anime film look, painterly backgrounds, film grain, dramatic lighting, "
        "Studio-quality feature animation, shallow depth of field"
    ),
    "manga_motion_comic": (
        "motion comic style, original manga line art preserved, subtle parallax "
        "camera move, limited animation, speech-bubble-free"
    ),
    "live_action": (
        "photorealistic live-action, real human skin texture with pores, "
        "natural facial anatomy, cinematic lighting, 85mm lens"
    ),
    "ink_mono": (
        "black and white ink illustration, heavy linework, screentone shading, "
        "manga panel aesthetic, high contrast"
    ),
}

# 一致性锁定的默认措辞：告诉模型「这几张参考图是同一个角色」
CONSISTENCY_TEMPLATE = (
    "The character in {tags} must stay exactly the same person as in the "
    "reference: identical face shape, hairstyle, hair color, eye color, "
    "outfit design and color scheme. Keep the character's identity consistent."
)

DEFAULT_NEGATIVE = (
    "lowres, blurry, jpeg artifacts, watermark, text, signature, "
    "extra fingers, missing fingers, deformed hands, bad anatomy, "
    "flickering, identity drift, inconsistent face, morphing"
)

# 台词排版模式（节点 schema 也引用这一份，避免两处硬编码）
DIALOGUE_MODES = ["spoken", "none"]
# 负向词处理方式：H3 官方 R2V 没有 negative 输入，所以负向只能以自然语言进正向提示词
NEGATIVE_MODES = ["append", "ignore"]
NEGATIVE_TEMPLATE = "Avoid these problems: {}."


# --------------------------------------------------------------------------
# 角色引用替换
# --------------------------------------------------------------------------
_REF_RE = re.compile(r"@([A-Za-z0-9_\u4e00-\u9fff\-]+)")


def substitute_refs(
    text: str,
    picture_map: Dict[str, List[int]],
    name_map: Optional[Dict[str, str]] = None,
    video_map: Optional[Dict[str, List[int]]] = None,
    audio_map: Optional[Dict[str, List[int]]] = None,
) -> Tuple[str, List[str], List[str]]:
    """把 ``@id`` / ``@名字`` 替换成对应的官方标签。

    查找顺序：图片 ``<Picture i>`` → 视频 ``<Video k>`` → 音频 ``<Audio j>``。
    多张/多条时用 ``and`` 连接（``<Picture 2> and <Picture 3>``）。

    返回 (新文本, 用到的条目 id 列表, 警告列表)。
    """
    name_map = name_map or {}
    video_map = video_map or {}
    audio_map = audio_map or {}
    used: List[str] = []
    warns: List[str] = []

    # 名字 → id 的反查（允许用中文名写分镜）
    alias = {v: k for k, v in name_map.items() if v}

    def _join(tags: List[str]) -> str:
        return " and ".join(tags) if len(tags) > 1 else tags[0]

    def _repl(m: re.Match) -> str:
        token = m.group(1)
        eid = token if token in picture_map else alias.get(token, token)

        for table, tag in ((picture_map, TAG_PICTURE),
                           (video_map, TAG_VIDEO),
                           (audio_map, TAG_AUDIO)):
            if eid in table and table[eid]:
                if eid not in used:
                    used.append(eid)
                return _join([tag.format(i) for i in table[eid]])

        warns.append(f"引用了 @{token}，但参考包里没有这个条目（已从提示词中移除）")
        return ""

    out = _REF_RE.sub(_repl, text)
    # 清理替换后留下的多余空格 / 悬空连接词
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"^\s*[,，;；]\s*", "", out)
    out = re.sub(r"\s*[,，;；]\s*$", "", out)
    return out.strip(), used, warns


def resolve_camera(camera: str, extra_presets: Optional[Dict[str, str]] = None) -> str:
    """运镜短名 → 镜头语言。未知短名原样返回（允许用户直接写英文句子）。"""
    c = (camera or "static").strip()
    if extra_presets and c in extra_presets:
        return extra_presets[c]
    if c in CAMERA_PRESETS:
        return CAMERA_PRESETS[c]
    return c


def resolve_style(style: str, extra_presets: Optional[Dict[str, str]] = None) -> str:
    s = (style or "").strip()
    if not s:
        return ""
    if extra_presets and s in extra_presets:
        return extra_presets[s]
    if s in STYLE_PRESETS:
        return STYLE_PRESETS[s]
    return s


# --------------------------------------------------------------------------
# 主装配函数
# --------------------------------------------------------------------------
def compose_plan(
    plan: Plan,
    refs: Optional[RefPack] = None,
    template: str = "",
    style_prefix: str = "",
    style_suffix: str = "",
    negative: str = "",
    camera_preset: str = "",
    dialogue_mode: str = "spoken",
    lock_consistency: bool = True,
    extra_camera_presets: Optional[Dict[str, str]] = None,
    extra_style_presets: Optional[Dict[str, str]] = None,
    include_describe: bool = True,
    ref_policy: str = "per_shot",
    max_ref_images: int = 0,
    negative_mode: str = "append",
) -> Plan:
    """把 Plan 里每个 Shot 装配成最终提示词（原地修改并返回同一个 Plan）。

    ★ 关键：序号是**逐镜重排**的。
    对每个镜头调用 ``refs.shot_ref_layout()``，按
    「分镜格锚点 → 本镜出场角色 → 风格图」重新从 <Picture 1> 开始编号，
    并把 ``@角色id`` 替换成**本镜的**标签。
    这样即使某镜少了某个角色，也不会串号（官方节点的序号来自 ref_items 追加顺序）。

    template 支持占位符（缺省用内置模板）：
        {scene}    分镜格锚点句
        {refs}     角色参考锁定句
        {action}   分镜 action 字段（@引用已替换）
        {camera}   运镜语言
        {style}    风格语言
        {dialogue} 对白句
        {extra}    分镜 extra 字段
        {speaker}  说话人 <Picture i>（若有）
        {negative} 负向句（negative_mode=append 时非空）

    negative_mode：H3 官方 R2V **没有 negative 输入**（实测输入列表里没有），
    BasicGuider 也没有负向分支。所以负向词只有两条路：
      "append" —— 以自然语言追加进正向提示词（`Avoid these problems: …`），默认；
      "ignore" —— 只记进报告，不写进提示词。
    早先版本两者都不做，negative 参数纯装饰，会让用户以为它在起作用。
    """
    refs = refs or RefPack()
    global_slots = refs.resolve_slots()
    name_map = {e.id: (e.name or e.id) for e in refs.entries}
    describe_map = {e.id: e.describe for e in refs.entries}
    style_ids = [e.id for e in refs.ordered() if e.kind == "style"]

    plan.negative = negative or plan.negative or DEFAULT_NEGATIVE
    plan.warnings.extend(global_slots["warnings"])

    for shot in plan.shots:
        warns: List[str] = []

        # ---- 0. 本镜的密集参考布局（序号唯一权威）----
        has_panel = shot.board_image_index is not None
        layout = shot_ref_layout(
            refs,
            characters=list(shot.characters),
            has_panel=has_panel,
            style_ids=style_ids,
            max_images=max_ref_images,
            policy=ref_policy,
        )
        picture_map = layout["picture_map"]
        video_map = layout["video_map"]
        audio_map = layout["audio_map"]
        warns.extend(layout["warnings"])

        # ★ 分镜里写的是人话（纲吉 / Reborn），这里统一解析成参考包 id。
        #   不解析 = 角色被静默忽略（画面随机且不报错）。
        chars = list(layout["resolved_characters"])
        for u in layout["unknown_characters"]:
            warns.append(f"引用了角色「{u}」，但参考包里没有这个条目（已忽略）")

        used_ids: List[str] = [c for c in chars if c in picture_map]

        # ---- 1. 场景锚点（分镜格）----
        scene_sentence = ""
        if has_panel:
            tag = TAG_PICTURE.format(picture_map[refs.panel_id][0])
            scene_sentence = (
                f"The composition, staging and framing follow {tag} "
                f"(the original manga panel). Animate it; do not redraw the layout."
            )

        # ---- 2. 角色参考锁定 ----
        ref_tags: List[str] = []
        for cid in chars:
            for i in picture_map.get(cid, []):
                tag = TAG_PICTURE.format(i)
                if tag not in ref_tags:
                    ref_tags.append(tag)

        ref_sentence = ""
        if ref_tags and lock_consistency:
            ref_sentence = CONSISTENCY_TEMPLATE.format(tags=" and ".join(ref_tags))

        if include_describe:
            descs = [describe_map.get(c, "") for c in chars]
            descs = [d for d in descs if d]
            if descs:
                ref_sentence = (ref_sentence + " " + " ".join(descs)).strip()

        # ---- 3. 主体动作（替换 @引用，用本镜的 picture_map）----
        action, a_used, a_warns = substitute_refs(
            shot.action, picture_map, name_map,
            video_map=video_map, audio_map=audio_map,
        )
        warns.extend(a_warns)
        for u in a_used:
            if u not in used_ids:
                used_ids.append(u)

        # ---- 4. 运镜 ----
        cam_src = shot.camera or camera_preset or "static"
        cam_text = resolve_camera(cam_src, extra_camera_presets)

        # ---- 5. 风格 ----
        style_id = shot.style or plan.style
        style_text = resolve_style(style_id, extra_style_presets) or plan.style_prompt
        if style_prefix:
            style_text = f"{style_prefix} {style_text}".strip()
        if style_suffix:
            style_text = f"{style_text} {style_suffix}".strip()

        # ---- 6. 对白 ----
        dialogue_text = ""
        if dialogue_mode == "spoken" and shot.dialogue.strip():
            line = shot.dialogue.strip().strip('"').strip("「」")
            speaker_tag = ""
            sp_ids, _ = resolve_character_ids(refs, [shot.speaker] if shot.speaker else [])
            if sp_ids and picture_map.get(sp_ids[0]):
                speaker_tag = TAG_PICTURE.format(picture_map[sp_ids[0]][0])
            if speaker_tag:
                dialogue_text = f'{speaker_tag} says: "{line}"'
            else:
                dialogue_text = f'She/he says: "{line}"'

        # ---- 7. extra（同样替换 @引用）----
        extra, e_used, e_warns = substitute_refs(
            shot.extra, picture_map, name_map,
            video_map=video_map, audio_map=audio_map,
        )
        warns.extend(e_warns)
        for u in e_used:
            if u not in used_ids:
                used_ids.append(u)

        # ---- 7.5 负向词（H3 无 negative 输入，只能以自然语言进正向）----
        neg_text = ""
        shot_neg = (shot.negative or plan.negative or "").strip()
        if negative_mode == "append" and shot_neg:
            neg_text = NEGATIVE_TEMPLATE.format(shot_neg)

        # ---- 8. 组装 ----
        if template.strip():
            prompt = template.format(
                scene=scene_sentence, refs=ref_sentence, action=action,
                camera=cam_text, style=style_text, dialogue=dialogue_text,
                extra=extra, speaker=shot.speaker or "", negative=neg_text,
            )
        else:
            parts = [
                scene_sentence,
                ref_sentence,
                action,
                extra,
                cam_text,
                style_text,
                dialogue_text,
                neg_text,
                plan.global_prompt,
            ]
            prompt = ", ".join(p.strip().rstrip(",") for p in parts if p and p.strip())

        prompt = re.sub(r"\s{2,}", " ", prompt).strip().strip(",").strip()
        if not prompt:
            warns.append("装配后提示词为空——请检查分镜的 action / dialogue 字段")

        shot.prompt = prompt
        shot.picture_tags = {
            cid: TAG_PICTURE.format(picture_map[cid][0])
            for cid in used_ids if picture_map.get(cid)
        }
        shot.ref_image_keys = list(layout["ref_image_keys"])
        shot.warnings.extend(warns)
        shot.negative = shot.negative or plan.negative

    return plan



def preview_all(plan: Plan) -> str:
    """整集提示词预览（喂给预览节点 / 前端面板）。

    明确标出「序号是逐镜重排的」——这是本插件最容易被误解的一点。
    """
    lines = [
        f"提示词预览：{plan.episode}（{len(plan.shots)} 镜）",
        "=" * 60,
        "注：<Picture i> 序号是**逐镜重排**的（按本镜实际传入的参考图顺序），",
        "    不要跨镜套用同一个序号。",
    ]
    for s in plan.shots:
        lines.append("")
        lines.append(
            f"--- [{s.index:02d}] {s.id}  |  {s.duration:.1f}s / {s.length}帧  |  "
            f"{s.width}x{s.height}  |  seed={s.resolved_seed}"
        )
        if s.picture_tags:
            lines.append("    本镜参考: " + ", ".join(f"{k}→{v}" for k, v in s.picture_tags.items()))
        lines.append("    提示词: " + (s.prompt or "(空)"))
        if s.warnings:
            for w in s.warnings:
                lines.append(f"    ! {w}")
    if plan.warnings:
        lines.append("")
        lines.append("-" * 60)
        lines.append("计划级警告：")
        for w in plan.warnings:
            lines.append(f"  ! {w}")
    return "\n".join(lines)


def load_camera_presets(path: str) -> Dict[str, str]:
    """从 JSON 文件读用户自定义运镜预设（失败返回空字典，不影响主流程）。"""
    import json
    import os

    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def load_style_presets(path: str) -> Dict[str, str]:
    return load_camera_presets(path)
