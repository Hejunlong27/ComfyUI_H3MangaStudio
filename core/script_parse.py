"""分镜清单解析：把「编剧写的分镜文本 / JSON」变成 Plan 对象。

支持三种来源，共用一套字段语义：

1. **JSON 文件** —— 推荐，结构化、可被外部工具生成
2. **块文本** —— 编剧手写，`=== shot_id ===` 分段 + `key: value` 行
3. **内联文本** —— 直接粘在节点上

块文本格式（对编剧友好，容错解析）：

    # episode: Reborn 第30卷 目标283
    # style: manga_anime
    === s01 ===
    panel: 0
    duration: 5
    characters: tsuna, reborn
    camera: slow_push_in
    action: 从门后走出，抬眼看向镜头
    dialogue: 这一刻终于还是到来了。
    speaker: tsuna
    extra: 窗外有乌鸦飞过

容错原则：**不因为一行写错就整集失败**。
解析器把问题记进 Plan.warnings，让用户在预览节点上看见，
而不是抛异常断掉工作流（对齐插件开发规范「能降级就降级」）。
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:  # 包内相对导入 / 测试脚本顶层导入 双模式
    from .types import (
        Plan, Shot, frames_for_seconds, align_canvas, official_canvas,
        resolve_canvas, DEFAULT_SIZE_SCALE, SIZE_MODE_PRESET,
        SIZE_PRESET_DEFAULT, H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT,
    )
except ImportError:  # pragma: no cover
    from types import (
        Plan, Shot, frames_for_seconds, align_canvas, official_canvas,
        resolve_canvas, DEFAULT_SIZE_SCALE, SIZE_MODE_PRESET,
        SIZE_PRESET_DEFAULT, H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT,
    )

# 分镜字段的别名表：编剧怎么写都能吃进来
_FIELD_ALIASES: Dict[str, str] = {
    "panel": "panel_index", "panels": "panel_index", "格": "panel_index",
    "面板": "panel_index", "分镜格": "panel_index", "格号": "panel_index",
    "dur": "duration", "time": "duration", "秒": "duration", "时长": "duration",
    "chars": "characters", "cast": "characters", "角色": "characters",
    "出场": "characters", "人物": "characters",
    "cam": "camera", "运镜": "camera", "镜头": "camera", "机位": "camera",
    "act": "action", "画面": "action", "描述": "action", "desc": "action",
    "动作": "action",
    "line": "dialogue", "台词": "dialogue", "对白": "dialogue", "说话": "dialogue",
    "who": "speaker", "说话人": "speaker", "说话者": "speaker",
    "style": "style", "风格": "style",
    "size": "size", "尺寸": "size", "分辨率": "size", "画幅": "size",
    "seed": "seed", "种子": "seed", "neg": "negative", "负向": "negative",
    "负面": "negative",
    "extra": "extra", "补充": "extra", "追加": "extra",
    "content": "content", "内容": "content", "画面内容": "content",
    "台词内容": "content", "summary": "content",
    "shots": "shots", "镜数": "shots", "镜头数": "shots", "格内镜头数": "shots",
    "id": "id", "编号": "id",
}

_INT_FIELDS = {"panel_index", "seed", "shots"}

# ★ 一格最多拆几个镜头。没有官方硬上限，纯粹是防手滑（写 100 会生成 100 个任务）。
MAX_SHOTS_PER_PANEL = 8


def _coerce(key: str, raw: str) -> Any:
    raw = raw.strip()
    if key in _INT_FIELDS:
        try:
            return int(float(raw))
        except ValueError:
            return None
    if key == "duration":
        try:
            return float(raw)
        except ValueError:
            return None
    if key == "characters":
        return [c.strip() for c in re.split(r"[,，;；\s]+", raw) if c.strip()]
    return raw


def _norm_key(key: str) -> str:
    k = key.strip().lower().replace(" ", "_").replace("-", "_")
    return _FIELD_ALIASES.get(k, k)


def _split_multi(value: str, n: int) -> List[str]:
    """把 ``A | B | C`` 按 ``|`` 切成 ``n`` 段（用于一格多镜）。

    **只在 ``shots > 1`` 时才会被调用** —— 单镜时绝不切分，所以动作描述里
    出现 ``|`` 也不会被误伤（完全向后兼容）。

    段数不足 → 后面的镜头重复最后一段（并不会有内容丢失）；
    段数多余 → 多余的并回最后一段（在最后一段里用 ``|`` 连接），
    这样「写多了一段」不会静默丢掉用户写的字。
    """
    if n <= 1:
        return [value]
    parts = [p.strip() for p in re.split(r"[|｜]", value or "")]
    parts = [p for p in parts if p] or [""]
    if len(parts) == 1:
        return [parts[0]] * n
    if len(parts) >= n:
        return parts[:n - 1] + [" | ".join(parts[n - 1:])]
    return parts + [parts[-1]] * (n - len(parts))


# --------------------------------------------------------------------------
# 块文本解析
# --------------------------------------------------------------------------
def parse_block_text(text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """解析块文本 → (meta, shot_dicts, warnings)。"""
    warnings: List[str] = []
    meta: Dict[str, Any] = {}
    shots: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        # 注释行：# key: value（只取第一段注释块作为元信息）
        if line.startswith("#") or line.startswith("//"):
            body = line.lstrip("#/ ").strip()
            if ":" in body:
                k, v = body.split(":", 1)
                nk = _norm_key(k)
                if nk in ("episode", "style", "negative", "default_size", "language",
                          "style_prompt", "global_prompt"):
                    meta[nk] = v.strip()
            continue

        # 分段头：=== s01 ===  /  --- s01 ---
        m = re.match(r"^[=\-*]{2,}\s*(.+?)\s*[=\-*]{2,}$", line)
        if m:
            if current is not None:
                shots.append(current)
            current = {"id": m.group(1).strip()}
            continue

        if ":" not in line and "：" not in line:
            warnings.append(f"第 {lineno} 行无法解析（既不是分段头也不是 key: value）：{line[:40]}")
            continue

        k, v = re.split(r"[:：]", line, maxsplit=1)
        nk = _norm_key(k)
        if nk not in _FIELD_ALIASES.values() and nk not in (
            "episode", "style", "negative", "default_size", "language",
            "style_prompt", "global_prompt",
        ):
            warnings.append(f"第 {lineno} 行未知字段「{k.strip()}」，已忽略")
            continue

        if current is None:
            # 还没进入分段，当作全局元信息
            if nk in ("episode", "style", "negative", "default_size", "language",
                      "style_prompt", "global_prompt"):
                meta[nk] = v.strip()
            else:
                warnings.append(f"第 {lineno} 行出现在任何分段之前，已忽略：{line[:40]}")
            continue

        current[nk] = _coerce(nk, v)

    if current is not None:
        shots.append(current)

    return meta, shots, warnings


# --------------------------------------------------------------------------
# JSON 解析
# --------------------------------------------------------------------------
def parse_json_obj(obj: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """解析 JSON 对象 / 数组 → (meta, shot_dicts, warnings)。"""
    warnings: List[str] = []

    if isinstance(obj, list):
        return {}, list(obj), warnings

    if not isinstance(obj, dict):
        raise ValueError("分镜 JSON 顶层必须是对象或数组")

    # 允许 {shots: [...]} / {episode: {...}, shots: [...]} 两种形态
    shots = obj.get("shots") or obj.get("镜头") or obj.get("分镜") or []
    if not isinstance(shots, list):
        raise ValueError("shots 字段必须是数组")

    meta: Dict[str, Any] = {}
    for k in ("episode", "style", "negative", "default_size", "language",
              "style_prompt", "global_prompt"):
        if k in obj and obj[k] is not None:
            meta[k] = obj[k]

    norm_shots: List[Dict[str, Any]] = []
    for i, s in enumerate(shots):
        if not isinstance(s, dict):
            warnings.append(f"第 {i} 个镜头不是对象，已跳过")
            continue
        norm: Dict[str, Any] = {}
        for k, v in s.items():
            norm[_norm_key(str(k))] = v
        norm_shots.append(norm)

    return meta, norm_shots, warnings


# --------------------------------------------------------------------------
# 组装 Plan
# --------------------------------------------------------------------------
def build_plan(
    meta: Dict[str, Any],
    shot_dicts: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
    source_note: str = "",
    default_duration: Optional[float] = None,
    size_override: Optional[Tuple[int, int, str]] = None,
    mark_missing_size: bool = False,
) -> Plan:
    """把解析结果组装成 Plan（含 17n+5 帧对齐与 32 倍数画幅对齐）。

    default_duration : 文本没写 duration 时的默认秒数（None = 用 Shot 自带默认 5s）
    size_override    : ``(width, height, note)``，由画幅选择器算好，**优先于**
                       meta 里的 ``default_size``（③ 的 preset / page 模式走这条）
    mark_missing_size: 文本没写逐镜 `size:` 时把 ``shot.size`` 留空 —— 只有
                       hybrid 合并需要它来区分「文本写了没写」
    """
    warnings = list(warnings or [])
    plan = Plan(source_note=source_note)

    if "episode" in meta:
        plan.episode = str(meta["episode"])
    if "style" in meta:
        plan.style = str(meta["style"])
    if "style_prompt" in meta:
        plan.style_prompt = str(meta["style_prompt"])
    if "negative" in meta:
        plan.negative = str(meta["negative"])
    if "default_size" in meta:
        plan.default_size = str(meta["default_size"])
    if "language" in meta:
        plan.language = str(meta["language"])
    if "global_prompt" in meta:
        plan.global_prompt = str(meta["global_prompt"])

    # 默认画幅：画幅选择器 > 文本 default_size
    if size_override:
        dw, dh = int(size_override[0]), int(size_override[1])
        size_note = str(size_override[2])
        plan.default_size = f"{dw}x{dh}"
    else:
        try:
            dw, dh = _parse_size(plan.default_size)
            size_note = f"文本默认 {plan.default_size}"
        except ValueError:
            warnings.append(f"默认画幅「{plan.default_size}」无法解析，回退 1344x768")
            dw, dh = H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT
            plan.default_size = f"{dw}x{dh}"
            size_note = "回退默认"
        dw, dh = align_canvas(int(dw), int(dh))

    seen_ids: set = set()
    for i, sd in enumerate(shot_dicts):
        sid = str(sd.get("id") or f"shot{i + 1:02d}")
        if sid in seen_ids:
            warnings.append(f"镜头 id 重复「{sid}」，已自动加后缀")
            sid = f"{sid}_{i}"
        seen_ids.add(sid)

        shot = Shot(id=sid, index=i)

        if sd.get("panel_index") is not None:
            try:
                shot.panel_index = int(sd["panel_index"])
            except (TypeError, ValueError):
                warnings.append(f"[{sid}] panel 值无法转成整数，已忽略")

        # ---- 时长：文本值 > default_duration > Shot 默认 ----
        raw_dur = sd.get("duration")
        if raw_dur is not None:
            try:
                shot.requested_seconds = float(raw_dur)
            except (TypeError, ValueError):
                warnings.append(f"[{sid}] duration 值无法转成数字，回退 5 秒")
        elif default_duration is not None:
            shot.requested_seconds = float(default_duration)
        if shot.requested_seconds > 0:
            shot.duration = shot.requested_seconds

        shot.action = str(sd.get("action") or "")
        shot.content = str(sd.get("content") or "")
        shot.dialogue = str(sd.get("dialogue") or "")
        shot.speaker = str(sd.get("speaker") or "")
        shot.camera = str(sd.get("camera") or "static")
        shot.style = str(sd.get("style") or "")
        shot.extra = str(sd.get("extra") or "")
        shot.negative = str(sd.get("negative") or "")

        chars = sd.get("characters")
        if isinstance(chars, str):
            chars = [c.strip() for c in re.split(r"[,，;；\s]+", chars) if c.strip()]
        shot.characters = list(chars or [])

        # 说话人自动补进角色列表（他一定出现在画面里，或者至少需要参考图）
        if shot.speaker and shot.speaker not in shot.characters:
            shot.characters.append(shot.speaker)

        # ---- 画幅：逐镜 size > 选择器/默认 ----
        size_str = str(sd.get("size") or "").strip()
        if size_str:
            try:
                shot.width, shot.height = _parse_size(size_str)
                shot.size = size_str
                shot.size_note = f"本镜文本指定 {size_str}"
                shot.size_was_given = True
            except ValueError:
                warnings.append(f"[{sid}] 画幅「{size_str}」无法解析，用计划默认值")
                shot.width, shot.height = dw, dh
                shot.size = f"{dw}x{dh}"
                shot.size_note = size_note
        else:
            shot.width, shot.height = dw, dh
            shot.size = "" if mark_missing_size else f"{dw}x{dh}"
            shot.size_note = size_note
        shot.width, shot.height = align_canvas(shot.width, shot.height)

        # 帧数：duration 是权威，length 由它推出来并对齐 17n+5
        shot.length = frames_for_seconds(shot.duration)
        shot.duration = round(shot.length / 24.0, 3)

        if sd.get("seed") is not None:
            try:
                shot.seed = int(sd["seed"])
            except (TypeError, ValueError):
                warnings.append(f"[{sid}] seed 无法转成整数，改用派生 seed")

        if not shot.action and not shot.dialogue and not shot.content:
            warnings.append(
                f"[{sid}] 既没有 content / action / dialogue，生成的画面可能完全随机"
            )

        # ---- ★ 一格多镜：`shots: N` 把这一条展开成 N 个 Shot ----
        #   场景：一个分镜格里画了两个时间相（大格先全景再特写、格内多时相）。
        #   展开出来的镜共用同一个 panel_index（绑同一张格图当参考锚点）；
        #   **第 1 个沿用原 id** → 单镜时 id / seed 完全不变（向后兼容）。
        n_sub = 1
        raw_n = sd.get("shots")
        if raw_n is not None:
            try:
                n_sub = int(float(raw_n))
            except (TypeError, ValueError):
                warnings.append(f"[{sid}] shots 值无法转成整数，按 1 个镜头处理")
                n_sub = 1
        if n_sub < 1:
            warnings.append(f"[{sid}] shots={n_sub} 非法（需 ≥1），按 1 个镜头处理")
            n_sub = 1
        elif n_sub > MAX_SHOTS_PER_PANEL:
            warnings.append(f"[{sid}] shots={n_sub} 超过上限 {MAX_SHOTS_PER_PANEL}，已截断")
            n_sub = MAX_SHOTS_PER_PANEL

        group = [shot]
        if n_sub > 1:
            acts = _split_multi(shot.action, n_sub)
            conts = _split_multi(shot.content, n_sub)
            dlogs = _split_multi(shot.dialogue, n_sub)
            group = []
            for k in range(n_sub):
                sub = shot if k == 0 else copy.deepcopy(shot)
                if k > 0:
                    sub.id = f"{sid}_{k + 1}"
                sub.sub_index, sub.sub_total = k, n_sub
                sub.action, sub.content, sub.dialogue = acts[k], conts[k], dlogs[k]
                group.append(sub)
            warnings.append(
                f"[{sid}] shots={n_sub}：该格展开为 {n_sub} 个镜头 —— "
                f"{group[0].id} … {group[-1].id}，每镜 {shot.requested_seconds:g}s"
                f"（该格合计 {shot.requested_seconds * n_sub:g}s）。"
                "各镜的 action / content / dialogue 可用 `A | B | C` 分段一一对应。"
            )
        plan.shots.extend(group)

    # 展开之后重排序号；顺带保证 id 唯一（`_2` 后缀可能与用户自己写的 id 撞）
    used_ids: set = set()
    for i, s in enumerate(plan.shots):
        s.index = i
        if s.id in used_ids:
            base, k = s.id, 2
            while f"{base}#{k}" in used_ids:
                k += 1
            s.id = f"{base}#{k}"
        used_ids.add(s.id)

    if not plan.shots:
        warnings.append("分镜清单里没有任何镜头")

    plan.warnings.extend(warnings)
    return plan


# --------------------------------------------------------------------------
# 从 board 自动出镜（③ 的 source=board / hybrid）
# --------------------------------------------------------------------------
def from_board(
    board: Any,
    default_duration: float = 5.0,
    size_override: Optional[Tuple[int, int, str]] = None,
    episode: str = "",
    shots_per_panel: int = 1,
) -> Plan:
    """★ 一个分镜格 = 一个镜头，自动生成镜头表初稿。

    为什么需要它：③ 原本是**纯文本**节点，用户必须从零手写；而「有几镜」这件事
    其实 ① 已经算出来了（board 里有几个格）。自动出稿之后，用户只改时长和内容。

    ⚠️ **镜头数 ≠ 格数**：一格想拍多个镜头（大格先全景再特写、格内多时相）时，
    用 ``shots_per_panel`` 批量设（整卷都习惯一格两镜就设 2），
    或改用 ``source=hybrid`` 在文本里对该格写 ``shots: N`` 逐格覆盖。
    展开出来的镜共用同一个 ``panel_index``（绑同一张格图当锚点）。
    """
    if board is None or not getattr(board, "panels", None):
        raise ValueError(
            "source=board / hybrid 需要上游的 board（① 漫画页切分镜格）。"
            "请把 ① 的 `分镜格` 输出接到本节点的 `board` 输入。"
        )
    if size_override:
        dw, dh, note = int(size_override[0]), int(size_override[1]), str(size_override[2])
    else:
        dw, dh, note = H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT, "回退默认 1344x768"

    n_sub = max(1, min(int(shots_per_panel or 1), MAX_SHOTS_PER_PANEL))

    plan = Plan(source_note="board", episode=episode or "untitled")
    plan.default_size = f"{dw}x{dh}"

    for p in board.panels:
        sid = f"p{int(getattr(p, 'page_index', 0)):02d}_{int(p.index):02d}"
        for k in range(n_sub):
            shot = Shot(
                id=sid if k == 0 else f"{sid}_{k + 1}",
                index=len(plan.shots),
                panel_index=int(p.index),
                duration=float(default_duration),
                requested_seconds=float(default_duration),
                sub_index=k,
                sub_total=n_sub,
            )
            shot.width, shot.height = dw, dh
            shot.size = f"{dw}x{dh}"
            shot.size_note = note
            shot.length = frames_for_seconds(shot.duration)
            shot.duration = round(shot.length / 24.0, 3)
            plan.shots.append(shot)

    if not plan.shots:
        plan.warnings.append("board 里没有任何分镜格，自动出镜得到 0 镜")
    else:
        plan.warnings.append(
            f"已从 board 自动生成 {len(plan.shots)} 个镜头"
            f"（{len(board.panels)} 格"
            + (f"，每格 {n_sub} 镜" if n_sub > 1 else "，一格一镜")
            + f"，{float(default_duration):g}s/镜）—— 请逐镜补 `content:`（画面内容）。"
            "一格想拍更多/更少镜头：改 `auto_shots_per_panel`，"
            "或在文本里对该格写 `shots: N`。"
        )
    return plan


def merge_plan_with_text(base: Plan, text_plan: Plan) -> Plan:
    """hybrid：以 board 自动稿为底，文本**按 panel 覆盖**；文本里没写 panel 的追加。

    文本没写的字段（duration / size / content）沿用自动稿的值 ——
    这样用户只需要在文本里写「要改的那几行」。

    ★ 按 **panel 分组**处理：同一个 panel 在文本里出现多行时（`shots: N` 展开出来的），
    要**整组替换**掉自动稿里该 panel 的所有镜头。若按行逐条替换，
    后一行会把前一行刚放进去的挤掉 —— 「一格多镜」在 hybrid 下就永远只剩 1 镜。
    """
    out = base.clone()

    panel_to_idx: Dict[int, List[int]] = {}
    for i, s in enumerate(out.shots):
        if s.panel_index is not None:
            panel_to_idx.setdefault(int(s.panel_index), []).append(i)

    # 文本侧按 panel 分组：连续的同一个 panel 归一组；没写 panel 的各自成组
    groups: List[Tuple[Optional[int], List[Any]]] = []
    for ts in text_plan.shots:
        key = int(ts.panel_index) if ts.panel_index is not None else None
        if key is not None and groups and groups[-1][0] == key:
            groups[-1][1].append(ts)
        else:
            groups.append((key, [ts]))

    replaced, appended = 0, 0
    for key, group in groups:
        idxs = panel_to_idx.get(key) if key is not None else None
        if idxs:
            old = out.shots[idxs[0]]
            for ts in group:
                # 文本里没写的字段沿用自动稿
                if ts.requested_seconds <= 0:
                    ts.requested_seconds = old.requested_seconds
                    ts.duration = old.duration
                    ts.length = old.length
                if not ts.size_was_given:
                    ts.width, ts.height = old.width, old.height
                    ts.size = old.size
                    ts.size_note = old.size_note
                if not ts.content:
                    ts.content = old.content
            replaced += len(group)
        else:
            appended += len(group)

    # 重建镜头列表：被覆盖的 panel 位置换成**整个文本组**（保持原顺序）
    by_key = {k: g for k, g in groups if k is not None}
    new_shots: List[Any] = []
    emitted: set = set()
    for s in out.shots:
        key = int(s.panel_index) if s.panel_index is not None else None
        if key is not None and key in by_key:
            if key not in emitted:
                new_shots.extend(by_key[key])
                emitted.add(key)
            continue
        new_shots.append(s)
    # 文本里没命中任何 panel 的组 + 无 panel 的行 → 追加
    for key, group in groups:
        if key is None or key not in emitted:
            new_shots.extend(group)

    out.shots = new_shots
    for i, s in enumerate(out.shots):
        s.index = i

    used: set = set()
    for s in out.shots:
        if s.id in used:
            b, k = s.id, 2
            while f"{b}#{k}" in used:
                k += 1
            s.id = f"{b}#{k}"
        used.add(s.id)

    out.source_note = "board+text"
    out.warnings.extend(text_plan.warnings)
    out.warnings.append(
        f"hybrid 合并：board 自动稿 {len(base.shots)} 镜，"
        f"文本覆盖 {replaced} 镜、追加 {appended} 镜"
    )
    if isinstance(text_plan.episode, str) and text_plan.episode and text_plan.episode != "untitled":
        out.episode = text_plan.episode
    if text_plan.style:
        out.style = text_plan.style
    return out


def _parse_size(s: str) -> Tuple[int, int]:
    s = str(s).strip().lower().replace(" ", "")
    m = re.match(r"^(\d+)\s*[x×*]\s*(\d+)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if ":" in s:  # 比例写法 16:9 —— 走官方 adapt_canvas 规则（16:9 → 1344x768）
        a, b = s.split(":", 1)
        return official_canvas(float(a), float(b))
    raise ValueError(f"无法解析画幅：{s}")


# --------------------------------------------------------------------------
# 对外主入口
# --------------------------------------------------------------------------
def _read_text(source: str, file_path: str, inline_text: str) -> Tuple[str, str]:
    """按 source 读文本。返回 ``(raw, note)``。保留原文的错误提示措辞。"""
    if source == "file":
        path = (file_path or "").strip().strip('"').strip("'")
        if not path:
            raise ValueError("source=file 但 file_path 为空。请填写分镜清单的绝对路径。")
        if not os.path.isabs(path):
            raise ValueError(f"分镜清单路径必须是绝对路径：{path}")
        if not os.path.isfile(path):
            raise ValueError(
                f"找不到分镜清单文件：{path}\n"
                f"请检查路径，或改用 source=inline 直接粘贴分镜文本。"
            )
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read(), f"file://{path}"
    raw = inline_text or ""
    if not raw.strip():
        raise ValueError("source=inline 但内联分镜文本为空。")
    return raw, "inline"


def load_plan(source: str = "file", file_path: str = "", inline_text: str = "",
              default_duration: Optional[float] = None,
              size_override: Optional[Tuple[int, int, str]] = None,
              board: Any = None, episode: str = "",
              shots_per_panel: int = 1) -> Plan:
    """统一入口：按 source 选择来源，返回 Plan。

    source:
      ``board``   —— ★ **从 board 自动出镜**（一格一镜），完全不读文本。
                     解决「③ 只解析手写文本，用户看不到图里有几镜」这个缺口。
      ``hybrid``  —— board 自动稿为底，文本按 ``panel: N`` 覆盖（没写的字段沿用自动稿）
      ``file`` / ``inline`` / ``auto`` —— 纯文本（旧行为，auto 在没有文本时才走 board）

    ``size_override`` ``(width, height, note)`` 与 ``default_duration``
    由 ③ 的画幅选择器 / 默认时长传入。

    ``shots_per_panel``：board 自动出镜时**每格展开几镜**（默认 1 = 一格一镜）。
    整卷都习惯「一格两镜」时设 2 最省事；逐格差异用文本里的 ``shots: N`` 覆盖。
    """
    source = (source or "auto").lower()
    dur = 5.0 if default_duration is None else float(default_duration)
    spp = max(1, int(shots_per_panel or 1))

    if source == "board":
        return from_board(board, default_duration=dur,
                          size_override=size_override, episode=episode,
                          shots_per_panel=spp)

    if source == "hybrid":
        base = from_board(board, default_duration=dur,
                          size_override=size_override, episode=episode,
                          shots_per_panel=spp)
        if not ((file_path or "").strip() or (inline_text or "").strip()):
            base.warnings.append("source=hybrid 但文本为空 —— 本次只用 board 自动稿")
            return base
        sub = "file" if (file_path or "").strip() else "inline"
        raw, note = _read_text(sub, file_path, inline_text)
        text_plan = parse_text(raw, note=note, default_duration=None,
                               size_override=size_override, mark_missing_size=True)
        return merge_plan_with_text(base, text_plan)

    if source == "auto":
        if (file_path or "").strip():
            source = "file"
        elif (inline_text or "").strip():
            source = "inline"
        else:
            return from_board(board, default_duration=dur,
                              size_override=size_override, episode=episode,
                              shots_per_panel=spp)

    raw, note = _read_text(source, file_path, inline_text)
    return parse_text(raw, note=note, default_duration=default_duration,
                      size_override=size_override)


def parse_text(raw: str, note: str = "",
               default_duration: Optional[float] = None,
               size_override: Optional[Tuple[int, int, str]] = None,
               mark_missing_size: bool = False) -> Plan:
    """自动判别 JSON / 块文本并解析。"""
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"分镜 JSON 解析失败（第 {e.lineno} 行第 {e.colno} 列）：{e.msg}\n"
                f"如果是手写块文本，请确认首行不是 '{{'。"
            ) from e
        meta, shots, warns = parse_json_obj(obj)
    else:
        meta, shots, warns = parse_block_text(raw)

    return build_plan(meta, shots, warns, source_note=note,
                      default_duration=default_duration,
                      size_override=size_override,
                      mark_missing_size=mark_missing_size)
