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

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:  # 包内相对导入 / 测试脚本顶层导入 双模式
    from .types import Plan, Shot, frames_for_seconds, align_canvas, H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT
except ImportError:  # pragma: no cover
    from types import Plan, Shot, frames_for_seconds, align_canvas, H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT

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
    "id": "id", "编号": "id",
}

_INT_FIELDS = {"panel_index", "seed"}


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
) -> Plan:
    """把解析结果组装成 Plan（含 17n+5 帧对齐与 32 倍数画幅对齐）。"""
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

    # 默认画幅解析
    try:
        dw, dh = _parse_size(plan.default_size)
    except ValueError:
        warnings.append(f"默认画幅「{plan.default_size}」无法解析，回退 1344x768")
        dw, dh = H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT
        plan.default_size = f"{dw}x{dh}"

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
        if sd.get("duration") is not None:
            try:
                shot.duration = float(sd["duration"])
            except (TypeError, ValueError):
                warnings.append(f"[{sid}] duration 值无法转成数字，回退 5 秒")
        shot.action = str(sd.get("action") or "")
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

        size_str = str(sd.get("size") or "").strip()
        if size_str:
            try:
                shot.width, shot.height = _parse_size(size_str)
            except ValueError:
                warnings.append(f"[{sid}] 画幅「{size_str}」无法解析，用计划默认值")
                shot.width, shot.height = dw, dh
            shot.size = size_str
        else:
            shot.width, shot.height = dw, dh
            shot.size = f"{dw}x{dh}"

        shot.width, shot.height = align_canvas(shot.width, shot.height)

        # 帧数：duration 是权威，length 由它推出来并对齐 17n+5
        shot.length = frames_for_seconds(shot.duration)
        shot.duration = round(shot.length / 24.0, 3)

        if sd.get("seed") is not None:
            try:
                shot.seed = int(sd["seed"])
            except (TypeError, ValueError):
                warnings.append(f"[{sid}] seed 无法转成整数，改用派生 seed")

        if not shot.action and not shot.dialogue:
            warnings.append(f"[{sid}] 既没有 action 也没有 dialogue，生成的画面可能完全随机")

        plan.shots.append(shot)

    if not plan.shots:
        warnings.append("分镜清单里没有任何镜头")

    plan.warnings.extend(warnings)
    return plan


def _parse_size(s: str) -> Tuple[int, int]:
    s = str(s).strip().lower().replace(" ", "")
    m = re.match(r"^(\d+)\s*[x×*]\s*(\d+)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if ":" in s:  # 比例写法 16:9
        a, b = s.split(":", 1)
        ratio = float(a) / float(b)
        h = 768
        w = int(round(h * ratio / 32) * 32)
        return w, h
    raise ValueError(f"无法解析画幅：{s}")


# --------------------------------------------------------------------------
# 对外主入口
# --------------------------------------------------------------------------
def load_plan(source: str, file_path: str = "", inline_text: str = "") -> Plan:
    """统一入口：按 source 选择来源，返回 Plan。

    source: "file" | "inline" | "auto"
    """
    source = (source or "auto").lower()

    if source == "auto":
        source = "file" if (file_path or "").strip() else "inline"

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
            raw = f.read()
        note = f"file://{path}"
    else:
        raw = inline_text or ""
        note = "inline"
        if not raw.strip():
            raise ValueError("source=inline 但内联分镜文本为空。")

    return parse_text(raw, note=note)


def parse_text(raw: str, note: str = "") -> Plan:
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

    return build_plan(meta, shots, warns, source_note=note)
