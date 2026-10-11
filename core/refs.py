"""参考包装配：把节点上的各种输入形态归一化成 RefPack。

输入形态可能五花八门（Autogrow 的 dict / 单张张量 / 列表 / None），
这里统一收口，业务逻辑就永远只面对一种结构。
—— 对齐插件开发规范：「execute() 里请把收到的值归一化（dict / 单张张量 / 列表都吃）」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .types import (
        RefEntry, RefPack, H3_MAX_REF_IMAGES, H3_MAX_REF_AUDIOS, H3_MAX_REF_VIDEOS,
        TAG_PICTURE, TAG_VIDEO, TAG_AUDIO,
        REF_IMAGE_KEY, REF_VIDEO_KEY, REF_VIDEO_AUDIO_KEY, REF_AUDIO_KEY,
    )
    from .image_ops import as_ref_image, as_ref_video
except ImportError:  # pragma: no cover
    from types import (
        RefEntry, RefPack, H3_MAX_REF_IMAGES, H3_MAX_REF_AUDIOS, H3_MAX_REF_VIDEOS,
        TAG_PICTURE, TAG_VIDEO, TAG_AUDIO,
        REF_IMAGE_KEY, REF_VIDEO_KEY, REF_VIDEO_AUDIO_KEY, REF_AUDIO_KEY,
    )
    from image_ops import as_ref_image, as_ref_video


def _is_media_dict(value: Any) -> bool:
    """判断一个 dict 是「媒体对象」而不是「Autogrow 容器」。

    ⚠️ AUDIO 在 ComfyUI 里本身就是 ``{"waveform": ..., "sample_rate": ...}``。
    早先 ``normalize_images`` 见 dict 就当成 Autogrow 拆包，
    结果音频被拆成 ``[waveform 张量, 32000]`` 这种垃圾 —— 音频输入一直是坏的，
    而且不会报错，只是「参考音频没生效」。
    """
    return isinstance(value, dict) and ("waveform" in value or "sample_rate" in value)


def normalize_images(value: Any, split_batch: bool = True) -> List[Any]:
    """把任意输入形态归一成「张量 / 媒体对象列表」。

    吃得下：
      * None                      → []
      * 媒体对象 dict（AUDIO）    → [该对象]（**不拆包**）
      * 单个张量 [H,W,C]          → [t]
      * 批次张量 [B,H,W,C]        → [t[0], t[1], ...]（按 batch 拆开当多张参考图）
      * list / tuple              → 递归
      * dict（Autogrow）          → 按 key 排序后递归（保证顺序稳定！）

    split_batch=False 用于**参考视频**：官方 ``ref_videos`` 的一个槽位
    就是「一段多帧视频」``[T,H,W,C]``，拆帧会让每段只剩 1 帧，
    直接触发官方的「reference videos need at least 5 frames」。
    """
    if value is None:
        return []
    if _is_media_dict(value):
        return [value]
    if isinstance(value, dict):
        # Autogrow 的 key 形如 ref_image_0 / image_1 / ref_image_10
        def _k(k: str):
            tail = str(k).rsplit("_", 1)[-1]
            return (0, int(tail)) if tail.isdigit() else (1, 0)
        out: List[Any] = []
        for k in sorted(value.keys(), key=_k):
            out.extend(normalize_images(value[k], split_batch=split_batch))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(normalize_images(v, split_batch=split_batch))
        return out
    # 单个张量
    try:
        ndim = getattr(value, "ndim", None)
        if ndim == 4:
            if split_batch:                # [B,H,W,C] → 拆批
                return [value[i] for i in range(int(value.shape[0]))]
            return [value]                 # 整段视频，不拆
        if ndim == 3:                      # [H,W,C] 或 [T,H,W]
            return [value]
    except Exception:
        pass
    return [value]


def parse_name_list(text: str, count: int, fallback_prefix: str = "ref") -> List[str]:
    """把逗号分隔的名字列表补齐/截断到 count 个。

    用户在节点上填 ``泽田纲吉, Reborn``，而图槽有 3 个 →
    第 3 个自动叫 ``ref3``，并保持 id 唯一。
    """
    import re

    names = [n.strip() for n in re.split(r"[,，;；\n]+", text or "") if n.strip()]
    out: List[str] = []
    for i in range(count):
        if i < len(names):
            out.append(names[i])
        else:
            out.append(f"{fallback_prefix}{i + 1}")
    return out


def slugify(name: str, index: int) -> str:
    """把角色名变成可用作 ``@引用`` 的稳定 id。

    **中文原样保留** —— id 只用于 ``@引用`` 与内部映射，不进 COMBO，
    所以没有理由把它变成 ``u7eb2u5409`` 这种没人能读的东西。
    只做三件事：去掉标点、空格转下划线、空名兜底。
    """
    n = (name or "").strip()
    if not n:
        return f"ref{index + 1}"

    out = []
    for ch in n:
        if ch.isalnum() or ch in "_-":
            out.append(ch)          # isalnum() 对中文返回 True
        elif ch.isspace():
            out.append("_")
        # 其余标点直接丢弃
    s = "".join(out).strip("_")
    return s or f"ref{index + 1}"


def build_ref_pack(
    character_names: str = "",
    character_images: Any = None,
    character_describes: str = "",
    style_name: str = "",
    style_images: Any = None,
    audio_names: str = "",
    audios: Any = None,
    video_names: str = "",
    videos: Any = None,
    video_audios: Any = None,
    max_ref_images: int = H3_MAX_REF_IMAGES,
    name_source: str = "slot",
    one_image_per_character: bool = True,
) -> RefPack:
    """从节点输入装配 RefPack。

    参数
    ----
    character_names   : 逗号分隔的角色名，**与 character_images 的槽位一一对应**
                        （即「第一个名字 = 第一个图槽」）。这是本插件最核心的约定：
                        用户不需要记 <Picture i> 序号，只需要把「角色名 → 它的定妆图」
                        按顺序填好。
    character_images  : Autogrow 图槽（dict / list / 张量）。批次会**拆帧**当多张参考图。
    character_describes: 每行一个角色外观锁定描述，与角色顺序对应
                        （比纯标签更稳的一致性手段）
    style_name/images : 全局风格基准图（所有镜头都会带上）
    audio_names/audios: 独立参考音频（BGM / 音效 / 配音样本），≤3
    video_names/videos: 参考视频片段，≤3。**一个槽位 = 一段多帧视频**（不拆帧）
    video_audios      : 与 videos **按下标一一对应**的配套音轨
    name_source       : "slot" = 用用户填的名字；"index" = 强制用 ref1/ref2
    one_image_per_character :
                        **默认 True —— 一个角色只锁一张参考图**。同名重复写只保留
                        第一张，其余忽略并发 warning。约定用「四视图 / 角色设定卡」
                        把多视角合成到一张图里，避免多图之间的服装/角度歧义。
                        置 False 可恢复旧行为（同名重复 = 多张定妆图，一对一吃槽）。
    """
    pack = RefPack(max_ref_images=int(max_ref_images or H3_MAX_REF_IMAGES))

    char_imgs = normalize_images(character_images)
    char_names = parse_name_list(character_names, len(char_imgs), "ref")
    describes = [d.strip() for d in (character_describes or "").splitlines() if d.strip()]

    provided_names = [n for n in
                      (x.strip() for x in (character_names or "").replace("，", ",").split(","))
                      if n]
    notes: List[str] = []
    if char_imgs and len(provided_names) < len(char_imgs):
        auto_named = "/".join(char_names[len(provided_names):])
        if one_image_per_character:
            notes.append(
                f"图槽有 {len(char_imgs)} 张，但只填了 {len(provided_names)} 个名字；"
                f"多出来的图会被当成独立角色（{auto_named}）—— **一角色一图**约定下，"
                f"请给每个角色填一个名字 + 一张「四视图 / 角色设定卡」，"
                f"不要用重复名字来塞多张图。"
            )
        else:
            notes.append(
                f"图槽有 {len(char_imgs)} 张，但只填了 {len(provided_names)} 个名字；"
                f"多出来的图被当成独立角色（{auto_named}）。"
                f"若它们是某个角色的第二、第三张定妆图，请把名字重复写一遍 —— "
                f"例如「纲吉, 纲吉, Reborn」表示纲吉 2 张 + Reborn 1 张。"
            )

    # ★ 角色参考图：**默认一个角色只锁一张图**。
    #   为什么默认改成一图：
    #     ① 歧义 —— 同一个角色的正面/侧面/全身图之间若发型、服装、画风有差异，
    #        模型会在它们之间摇摆，反而不如一张清晰的四视图稳定；
    #     ② 稀缺 —— 官方参考图上限 9 张，还要跟分镜格锚点、风格基准图共享，
    #        同一个角色吃掉 2~3 张会挤掉别的角色。
    #   约定：用一张「四视图 / 角色设定卡」把正/侧/背/表情并到一张图里。
    #   确实需要多图（例如角色换装、多形态）时把 one_image_per_character 关掉，
    #   行为与旧版完全一致（同名重复 = 按顺序吃掉多个图槽）。
    groups: List[Tuple[str, List[Any]]] = []
    index_of: Dict[str, int] = {}
    dropped: List[str] = []
    for slot_i, (name, img) in enumerate(zip(char_names, char_imgs)):
        if name in index_of:
            if one_image_per_character:
                dropped.append(f"槽{slot_i + 1}「{name}」")
                continue
            groups[index_of[name]][1].append(img)
        else:
            index_of[name] = len(groups)
            groups.append((name, [img]))

    if dropped:
        notes.append(
            f"一角色一图（默认）：以下同名重复的图槽被忽略 —— {', '.join(dropped)}。"
            f"每个角色只保留**第一张**图（其余不进参考包，也不会占 <Picture i> 序号）。"
            f"请改用「四视图 / 角色设定卡」把多视角合成一张；"
            f"确实需要多图时，把 ② 的「一角色一图」关掉。"
        )
    pack.source_note = " ".join(notes)

    used_ids: set = set()
    for i, (raw_name, imgs) in enumerate(groups):
        eid = raw_name if name_source == "slot" else f"ref{i + 1}"
        eid = slugify(eid, i)
        base = eid
        k = 2
        while eid in used_ids:
            eid = f"{base}_{k}"
            k += 1
        used_ids.add(eid)

        pack.entries.append(RefEntry(
            id=eid,
            name=raw_name,
            kind="character",
            images=list(imgs),
            priority=i,
            lock=True,
            describe=describes[i] if i < len(describes) else "",
        ))

    style_imgs = normalize_images(style_images)
    if style_imgs:
        sname = (style_name or "style_main").strip() or "style_main"
        pack.entries.append(RefEntry(
            id=slugify(sname, 900),
            name=sname,
            kind="style",
            images=list(style_imgs),
            priority=900,
            lock=True,
            describe="match the art style and color grading of the style reference",
        ))

    # 参考视频：**不拆批** —— 一个槽位就是一段多帧视频
    vid_list = normalize_images(videos, split_batch=False)
    vid_list = [v for v in vid_list if v is not None]
    vid_audio_list = normalize_images(video_audios, split_batch=False)
    vid_audio_list = [a for a in vid_audio_list if a is not None]
    if vid_list:
        vnames = parse_name_list(video_names, len(vid_list), "clip")
        for i, v in enumerate(vid_list):
            pack.entries.append(RefEntry(
                id=slugify(vnames[i], 500 + i), name=vnames[i], kind="video",
                videos=[v], priority=500 + i,
                # 官方按「键名尾号」配对音轨：ref_video_audio_N ↔ ref_video_N，
                # 所以这里按下标一一对应即可
                video_audio=vid_audio_list[i] if i < len(vid_audio_list) else None,
            ))
        if vid_audio_list and len(vid_audio_list) != len(vid_list):
            pack.source_note = (
                f"参考视频 {len(vid_list)} 条、配套音轨 {len(vid_audio_list)} 条，"
                f"数量不一致；只有前 {min(len(vid_list), len(vid_audio_list))} 条会被配对。"
            )

    audio_list = normalize_images(audios)
    audio_list = [a for a in audio_list if a is not None]
    if audio_list:
        anames = parse_name_list(audio_names, len(audio_list), "audio")
        for i, a in enumerate(audio_list):
            pack.entries.append(RefEntry(
                id=slugify(anames[i], 700 + i), name=anames[i], kind="audio",
                audio=a, priority=700 + i,
            ))

    if not pack.entries:
        pack.source_note = "空参考包（没有任何参考素材）"
    return pack


# ==========================================================================
# 角色名 → 参考包 id 解析
# ==========================================================================
def _alias_table(refs: RefPack) -> Dict[str, str]:
    """构造 别名 → 正式 id 的查找表。

    分镜里编剧写的是**人话**（``纲吉`` / ``Reborn`` / ``阿纲``），
    而参考包里的 id 是 slugify 之后的（``纲吉`` / ``Reborn``）。
    两者不一定相等（名字带空格、标点、别名时），所以必须统一在这里解析 ——
    否则 ``shot.characters`` 会查不到 ``picture_map``，
    表现为「角色被静默忽略、画面随机」，而且**不报错**。
    """
    table: Dict[str, str] = {}
    for e in refs.entries:
        table.setdefault(e.id, e.id)
        if e.name:
            table.setdefault(e.name, e.id)
            table.setdefault(slugify(e.name, 0), e.id)
    return table


def resolve_character_ids(refs: RefPack, tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    """把分镜里的角色写法解析成参考包 id。返回 (已解析 id 列表, 未命中的原文列表)。

    保持输入顺序、自动去重。
    """
    table = _alias_table(refs)
    ids: List[str] = []
    unknown: List[str] = []
    for t in tokens or ():
        raw = str(t).strip()
        if not raw:
            continue
        eid = table.get(raw) or table.get(slugify(raw, 0))
        if eid:
            if eid not in ids:
                ids.append(eid)
        elif raw not in unknown:
            unknown.append(raw)
    return ids, unknown


# ==========================================================================
# ★ 每镜密集重排：<Picture i> 序号唯一权威来源
# ==========================================================================
def shot_ref_layout(
    refs: RefPack,
    characters: Sequence[str] = (),
    panel_image: Any = None,
    style_ids: Optional[Sequence[str]] = None,
    max_images: int = 0,
    policy: str = "per_shot",
    panel_id: str = "",
    has_panel: bool = False,
) -> Dict[str, Any]:
    """为一个镜头生成**密集编号**的参考图布局。

    为什么必须密集重排
    ------------------
    官方源码：

        for img in (ref_images or {}).values():
            ref_items.append({"type": "image", "data": resized})

    ``<Picture i>`` 的序号 = ``ref_items`` 的**追加顺序**，与 dict 的键名无关。
    所以「角色 A 全局是 <Picture 3>」这种想法是错的 —— 一旦某镜没有角色 B，
    序号会整体前移，而提示词里的 ``<Picture 3>`` 就会指到别人身上（**静默串号**）。

    本函数保证：**序号只由本镜实际传进去的图的顺序决定**，
    顺序固定为「分镜格锚点 → 本镜出场角色（按 plan 里的出场顺序）→ 风格图」。

    参数
    ----
    characters : 本镜出场的角色 id（顺序即编号顺序）
    panel_image: 本镜的分镜格图（作为 <Picture 1> 的视觉锚点）。None 则不留位。
    style_ids  : 要带上的风格条目 id，默认取参考包里所有 kind == "style" 的条目
    max_images : 上限，0 表示取 refs.max_ref_images
    policy     : "per_shot"（只带本镜用到的）/ "all"（全量带，编号与全局一致但更慢）

    返回
    ----
    {
      "ref_images": {"ref_image_0": t, ...},   # 4 维 [1,H,W,C]，可直接喂官方 execute()
      "picture_map": {"tsuna": [2], ...},      # ★本镜的序号，提示词必须用这个
      "ref_videos": {...}, "ref_video_audios": {...}, "ref_audios": {...},
      "video_map": {...}, "audio_map": {...},
      "order": [ {"id","kind","tag"} ... ],    # 有序清单（前端面板展示）
      "dropped": [ ... ],                       # 被丢弃的（溢出/缺图）
      "warnings": [ ... ],
    }

    张量契约（★ 别绕过这里自己拼 ref_images）
    -----------------------------------------
    本函数是**唯一**出参考图的地方，同时负责把它们规整成官方节点要求的 4 维
    ``[B,H,W,C]``：官方按 ``h, w = img.shape[1], img.shape[2]`` 读尺寸，3 维
    ``[H,W,C]`` 会被读成「宽 = h、通道数 = w」，继而 resize 出畸形尺寸并炸在
    ``comfy/utils.py::common_upscale`` 里（报错文本与形状毫无字面关联）。
    细节与真实事故复盘见 ``core/image_ops.py::as_ref_image``。

    关于视频/音频
    -------------
    它们**不参与逐镜重排**（是全局素材，每个镜头都带上），但仍按官方契约
    分池编号：图片池 → 视频池 → 音频池，各自 1-based。
    视频的配套音轨必须与其 ``<Video k>`` **同号**（官方按键名尾号配对）。
    """
    limit = int(max_images or refs.max_ref_images or H3_MAX_REF_IMAGES)
    limit = min(limit, H3_MAX_REF_IMAGES)
    panel_id = panel_id or refs.panel_id

    pool = {e.id: e for e in refs.entries}
    if style_ids is None:
        style_ids = [e.id for e in refs.ordered() if e.kind == "style"]
    else:
        style_ids, _ = resolve_character_ids(refs, list(style_ids))

    # ★ 角色名 → id 解析（分镜写人话，这里统一翻译）
    resolved_chars, unknown_chars = resolve_character_ids(refs, list(characters or ()))
    style_ids = [s for s in style_ids if s in pool]

    plan_order: List[Tuple[str, str]] = []   # (id, kind)

    if panel_image is not None or has_panel:
        plan_order.append((panel_id, "panel"))

    if policy == "all":
        for e in refs.ordered():
            if e.kind in ("character", "style", "prop"):
                plan_order.append((e.id, e.kind))
    else:
        for cid in resolved_chars:
            if cid in pool:
                plan_order.append((cid, pool[cid].kind))
        for sid in style_ids:
            if sid in pool and (sid, pool[sid].kind) not in plan_order:
                plan_order.append((sid, pool[sid].kind))

    ref_images: Dict[str, Any] = {}
    ref_image_keys: List[str] = []   # ★ 槽位键序列：**总是**含分镜格位，与张量是否就绪无关
    picture_map: Dict[str, List[int]] = {}
    order: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    warnings: List[str] = []
    overflow_warned = False

    slot = 0
    for eid, kind in plan_order:
        if kind == "panel":
            images = [panel_image]
            name = "分镜格锚点"
        else:
            entry = pool.get(eid)
            if entry is None:
                continue
            images = list(entry.images)
            name = entry.name or eid
            if not images:
                dropped.append({"id": eid, "reason": "没有参考图"})
                continue

        for img in images:
            if slot >= limit:
                dropped.append({"id": eid, "reason": f"超出 {limit} 张上限"})
                if not overflow_warned:
                    overflow_warned = True
                    warnings.append(
                        f"参考图超过 {limit} 张上限，已丢弃部分图片"
                        f"（本镜需要 {len(plan_order)} 个条目）"
                    )
                continue

            # 序号照样占用（提示词里的 <Picture i> 是按**位次**算的），
            # 只是「装配/审计」阶段拿不到分镜格张量时，ref_images 里不放 None。
            # 真正喂给官方节点时（渲染节点）张量一定就绪。
            key = REF_IMAGE_KEY.format(slot)
            ref_image_keys.append(key)
            if img is not None:
                # ★ 4 维化：官方 R2V 按 [B,H,W,C] 读 shape[1]/shape[2]，
                #   3 维 [H,W,C] 会被读成 (宽, 通道数) → 畸形 resize →
                #   comfy/utils.py::common_upscale 里抛 "shape '[1, -1, 3, 1152, 32]' is invalid..."。
                fixed = as_ref_image(img, warnings, f"[{eid}] {key}")
                if fixed is None:
                    # 不能「跳过这一槽」—— ref_images 是按 .values() 顺序编 <Picture i> 的，
                    # 少放一张会让后面所有序号整体前移（静默串号）。直接报错更好查。
                    raise ValueError(
                        f"参考图槽位 {key}（条目「{eid}」）拿到的是 "
                        f"{type(img).__name__}，不是图像张量。"
                        f"请检查该槽位接的是不是 IMAGE。"
                    )
                ref_images[key] = fixed
            slot += 1
            picture_map.setdefault(eid, []).append(slot)
            order.append({
                "slot": slot - 1,
                "tag": TAG_PICTURE.format(slot),
                "id": eid,
                "name": name,
                "kind": kind,
            })

    # ---- 视频池 / 音频池：全局素材，不参与逐镜重排，但按官方契约编号 ----
    ref_videos: Dict[str, Any] = {}
    ref_video_audios: Dict[str, Any] = {}
    ref_audios: Dict[str, Any] = {}
    video_map: Dict[str, List[int]] = {}
    audio_map: Dict[str, List[int]] = {}

    vid_slot = 0
    for entry in refs.ordered():
        if entry.kind != "video":
            continue
        for vf in entry.videos:
            if vf is None or vid_slot >= H3_MAX_REF_VIDEOS:
                dropped.append({"id": entry.id, "reason": f"视频超出 {H3_MAX_REF_VIDEOS} 条上限"})
                continue
            vkey = REF_VIDEO_KEY.format(vid_slot)
            fixed_v = as_ref_video(vf, warnings, f"[{entry.id}] {vkey}")
            if fixed_v is None:
                raise ValueError(
                    f"参考视频槽位 {vkey}（条目「{entry.id}」）不是视频张量。"
                    f"官方要求 [T,H,W,C]（T 为帧数，≥5 帧）。"
                )
            ref_videos[vkey] = fixed_v
            if entry.video_audio is not None:
                ref_video_audios[REF_VIDEO_AUDIO_KEY.format(vid_slot)] = entry.video_audio
            vid_slot += 1
            video_map.setdefault(entry.id, []).append(vid_slot)
            order.append({
                "slot": None, "tag": TAG_VIDEO.format(vid_slot), "id": entry.id,
                "name": entry.name or entry.id, "kind": "video",
            })

    aud_slot = 0
    for entry in refs.ordered():
        if entry.kind != "audio" or entry.audio is None:
            continue
        if aud_slot >= H3_MAX_REF_AUDIOS:
            dropped.append({"id": entry.id, "reason": f"音频超出 {H3_MAX_REF_AUDIOS} 条上限"})
            continue
        ref_audios[REF_AUDIO_KEY.format(aud_slot)] = entry.audio
        aud_slot += 1
        audio_map.setdefault(entry.id, []).append(aud_slot)
        order.append({
            "slot": None, "tag": TAG_AUDIO.format(aud_slot), "id": entry.id,
            "name": entry.name or entry.id, "kind": "audio",
        })

    return {
        "ref_images": ref_images,
        "ref_image_keys": ref_image_keys,
        "ref_videos": ref_videos,
        "ref_video_audios": ref_video_audios,
        "ref_audios": ref_audios,
        "picture_map": picture_map,
        "video_map": video_map,
        "audio_map": audio_map,
        "order": order,
        "dropped": dropped,
        "warnings": warnings,
        "resolved_characters": resolved_chars,
        "unknown_characters": unknown_chars,
    }


def layout_table(layout: Dict[str, Any], title: str = "本镜参考图布局") -> str:
    lines = [f"{title}（共 {len(layout['order'])} 张）", "-" * 52]
    if not layout["order"]:
        lines.append("  （无参考图）")
    for item in layout["order"]:
        lines.append(f"  {item['tag']:<12} {item['kind']:<9} {item['id']}  ({item['name']})")
    for d in layout.get("dropped", []):
        lines.append(f"  [丢弃] {d['id']} —— {d['reason']}")
    for w in layout.get("warnings", []):
        lines.append(f"  ! {w}")
    return "\n".join(lines)

