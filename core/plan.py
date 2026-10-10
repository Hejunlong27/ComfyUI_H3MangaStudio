"""计划编排：绑定分镜格、派生 seed、一致性审计、展开成可执行任务列表。

这一层是「批量生产与任务编排」的大脑，**不 import comfy**，
所以能被完整单测覆盖。节点层只是它的壳。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

try:
    from .types import (
        Board, Plan, RefPack,
        align_canvas, align_frame_count,
        TAG_PICTURE,
    )
    from .refs import shot_ref_layout, resolve_character_ids
except ImportError:  # pragma: no cover
    from types import (
        Board, Plan, RefPack,
        align_canvas, align_frame_count,
        TAG_PICTURE,
    )
    from refs import shot_ref_layout, resolve_character_ids


# --------------------------------------------------------------------------
# seed 派生
# --------------------------------------------------------------------------
SEED_MAX = 0xFFFFFFFFFFFFFFFF


def derive_seed(base_seed: int, index: int, shot_id: str = "") -> int:
    """从 base_seed 稳定派生每镜 seed。

    为什么不用 base_seed + index：这样「插入一镜」会让后面所有镜头全部换 seed，
    已经调好的镜头白费。改用 shot_id 哈希混合 —— **index 只作为无 id 时的兜底**，
    所以插入 / 删除 / 重排镜头都不会改变其它镜头的 seed，
    断点续跑（skip_done_ids）才真正成立。
    """
    base = int(base_seed) & SEED_MAX
    if not shot_id:
        return (base + index * 0x9E3779B97F4A7C15) & SEED_MAX
    h = hashlib.blake2b(shot_id.encode("utf-8"), digest_size=8).digest()
    return (base ^ int.from_bytes(h, "little")) & SEED_MAX


def assign_seeds(plan: Plan, base_seed: int) -> Plan:
    for i, s in enumerate(plan.shots):
        s.resolved_seed = int(s.seed) if s.seed is not None else derive_seed(base_seed, i, s.id)
    return plan


# --------------------------------------------------------------------------
# 分镜格绑定
# --------------------------------------------------------------------------
def bind_board(
    plan: Plan,
    board: Optional[Board],
    mode: str = "by_panel_index",
    offset: int = 0,
    allow_reuse: bool = True,
) -> Plan:
    """把 board 里的分镜格图绑到镜头上。

    mode:
      by_panel_index —— 用 shot.panel_index 精确指定（推荐，最可控）
      sequential     —— 按镜头顺序依次取格（漫画改编最常见：一格一镜）
      none           —— 不绑图（纯文生视频 / 只用角色参考图）
    """
    if board is None or mode == "none":
        for s in plan.shots:
            s.board_image_index = None
        if mode != "none":
            plan.warnings.append("未连接分镜格（board），本次按「无格图」处理")
        return plan

    n_panels = len(board)
    if n_panels == 0:
        plan.warnings.append("分镜格为空（切格结果 0 格），请检查切格参数")
        for s in plan.shots:
            s.board_image_index = None
        return plan

    used: set = set()
    for i, s in enumerate(plan.shots):
        if mode == "sequential":
            idx = i + offset
        else:
            if s.panel_index is None:
                plan.warnings.append(f"[{s.id}] 没写 panel，已按顺序回退到第 {i + offset} 格")
                idx = i + offset
            else:
                idx = int(s.panel_index) + offset

        if idx < 0 or idx >= n_panels:
            plan.warnings.append(
                f"[{s.id}] 指定的分镜格 #{idx} 超出范围（共 {n_panels} 格），"
                f"该镜将退化为纯文生视频（仅用角色参考图）"
            )
            s.board_image_index = None
            continue

        if not allow_reuse and idx in used:
            plan.warnings.append(f"[{s.id}] 分镜格 #{idx} 已被前面的镜头占用，已解绑")
            s.board_image_index = None
            continue

        s.board_image_index = idx
        used.add(idx)

    return plan


# --------------------------------------------------------------------------
# 一致性审计
# --------------------------------------------------------------------------
def audit(plan: Plan, refs: Optional[RefPack] = None, strict: bool = False) -> Dict[str, Any]:
    """审计「一致性」相关的五类问题，返回结构化报告。

    检查项：
      A. 分镜引用了参考包里不存在的角色（提示词里会被移除 → 画面随机）
      B. 参考包里存在但**从未被任何镜头引用**的角色（可能是分镜漏写）
      C. 角色参考图数量过少（<2 张）—— 一致性风险高
      D. 参考图溢出（>9）—— 官方节点会静默丢弃
      E. **逐镜布局校验**：重算每镜的密集布局，与提示词里实际写入的
         ``picture_tags`` 比对。这是防「角色串号」的回归守卫 ——
         一旦序号规则被改坏，这条会立刻变红。
    """
    refs = refs or RefPack()
    slots = refs.resolve_slots()
    global_map = slots["picture_map"]
    entries = {e.id: e for e in refs.entries}
    style_ids = [e.id for e in refs.ordered() if e.kind == "style"]

    issues: List[Dict[str, str]] = []

    referenced: set = set()
    for s in plan.shots:
        resolved, unknown = resolve_character_ids(refs, list(s.characters))
        for cid in resolved:
            referenced.add(cid)
        for u in unknown:
            issues.append({
                "level": "P0",
                "shot": s.id,
                "kind": "missing_ref",
                "msg": f"镜头引用了角色「{u}」，但参考包里没有这个条目 —— "
                       f"模型不会知道这个角色长什么样",
            })
        if s.board_image_index is None and not resolved:
            issues.append({
                "level": "P1",
                "shot": s.id,
                "kind": "no_anchor",
                "msg": "既没有分镜格图、也没有角色参考图，本镜完全靠文字生成（结果不可复现）",
            })

        # ---- E. 逐镜布局校验（防串号回归守卫）----
        layout = shot_ref_layout(
            refs, characters=list(s.characters),
            has_panel=s.board_image_index is not None,
            style_ids=style_ids,
        )
        expected = {
            cid: TAG_PICTURE.format(layout["picture_map"][cid][0])
            for cid in layout["resolved_characters"] if layout["picture_map"].get(cid)
        }
        if s.picture_tags and expected != s.picture_tags:
            issues.append({
                "level": "P0",
                "shot": s.id,
                "kind": "slot_mismatch",
                "msg": f"提示词里写的参考序号 {s.picture_tags} 与本镜重算的布局 {expected} "
                       f"不一致 —— 角色会串号。请重新跑一次「提示词装配」。",
            })
        if s.ref_image_keys:
            actual_keys = list(layout["ref_image_keys"])
            if actual_keys != s.ref_image_keys:
                issues.append({
                    "level": "P0",
                    "shot": s.id,
                    "kind": "slot_layout_drift",
                    "msg": f"参考图槽位布局漂移：提示词装配时是 {s.ref_image_keys}，"
                           f"现在是 {actual_keys}。参考包在装配之后被改动过。",
                })

    for eid, entry in entries.items():
        if entry.kind == "character" and eid not in referenced:
            issues.append({
                "level": "P2", "shot": "-", "kind": "unused_ref",
                "msg": f"角色「{eid}」在参考包里，但没有任何镜头引用它（分镜是否漏写？）",
            })
        if entry.kind == "character" and len(entry.images) == 0:
            issues.append({
                "level": "P0", "shot": "-", "kind": "empty_ref",
                "msg": f"角色「{eid}」没有任何参考图",
            })
        elif entry.kind == "character" and len(entry.images) < 2:
            issues.append({
                "level": "P1", "shot": "-", "kind": "thin_ref",
                "msg": f"角色「{eid}」只有 {len(entry.images)} 张参考图，"
                       f"跨镜一致性风险高（建议 ≥3 张：正面 / 侧面 / 全身）",
            })

    for w in slots["warnings"]:
        issues.append({"level": "P0", "shot": "-", "kind": "overflow", "msg": w})

    p0 = [i for i in issues if i["level"] == "P0"]
    p1 = [i for i in issues if i["level"] == "P1"]
    p2 = [i for i in issues if i["level"] == "P2"]
    passed = (len(p0) == 0) and (not strict or len(p1) == 0)

    return {
        "passed": passed,
        "strict": strict,
        "counts": {"P0": len(p0), "P1": len(p1), "P2": len(p2)},
        "issues": issues,
        "picture_map": global_map,
        "picture_pool": {k: len(v) for k, v in slots["picture_pool"].items()},
    }



def audit_report(result: Dict[str, Any]) -> str:
    lines = ["一致性审计报告", "=" * 60]
    status = "通过" if result["passed"] else "未通过"
    if result["strict"]:
        status += "（严格模式：P0/P1 都必须为 0）"
    lines.append(f"结论：{status}")
    c = result["counts"]
    lines.append(f"  P0 阻断 {c['P0']} 项 / P1 高风险 {c['P1']} 项 / P2 提示 {c['P2']} 项")
    if result["issues"]:
        lines.append("-" * 60)
        for i in result["issues"]:
            lines.append(f"  [{i['level']}] [{i['shot']}] {i['msg']}")
    if result["picture_map"]:
        lines.append("-" * 60)
        lines.append("角色 → 序号映射：")
        for eid, imgs in result["picture_map"].items():
            tags = " ".join(TAG_PICTURE.format(i) for i in imgs)
            lines.append(f"  {eid:<16} {tags}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 展开成可执行任务
# --------------------------------------------------------------------------
def build_render_tasks(
    plan: Plan,
    refs: Optional[RefPack] = None,
    base_seed: int = 0,
    max_shots: int = 0,
    skip_ids: Optional[Sequence[str]] = None,
    ref_policy: str = "per_shot",
    max_ref_images: int = 0,
    mutate: bool = False,
) -> List[Dict[str, Any]]:
    """把 Plan 展开成**逐镜任务列表**，供整集渲染节点逐个消费。

    每个 task 是纯字典（除 ref_images 里的张量），可以被打印、写盘、断点续跑比对。
    ★ ``ref_image_keys`` 与 ``picture_tags`` 都来自**逐镜密集重排**，
    与提示词里的标签严格一致（这是不串号的保证）。

    参数
    ----
    mutate : 默认 ``False`` —— **不改动传入的 plan**（内部先 clone）。
             早先版本会直接写回 ``shot.ref_image_keys`` / ``shot.picture_tags`` /
             ``shot.warnings``，导致 ``plan_diff()`` 这类「只想读一下」的调用
             把已经装配好的计划改坏（实测：picture_tags 被清空）。
             需要写回时显式传 ``mutate=True``。
    """
    if not mutate:
        plan = plan.clone()

    refs = refs or RefPack()
    style_ids = [e.id for e in refs.ordered() if e.kind == "style"]
    skip = set(skip_ids or [])

    assign_seeds(plan, base_seed)

    tasks: List[Dict[str, Any]] = []
    for s in plan.shots:
        if s.id in skip:
            continue
        w, h = align_canvas(s.width, s.height)
        length = align_frame_count(s.length)

        layout = shot_ref_layout(
            refs, characters=list(s.characters),
            has_panel=s.board_image_index is not None,
            style_ids=style_ids,
            max_images=max_ref_images,
            policy=ref_policy,
        )
        picture_tags = {
            cid: TAG_PICTURE.format(layout["picture_map"][cid][0])
            for cid in layout["resolved_characters"] if layout["picture_map"].get(cid)
        }
        # 注意：这些结果**只放进 task**，不写回 Shot。
        # 渲染节点读的是 task，不依赖 Shot 被改写。
        if mutate:
            s.ref_image_keys = list(layout["ref_image_keys"])
            s.picture_tags = dict(picture_tags)
            for wmsg in layout.get("warnings", []):
                if wmsg not in s.warnings:
                    s.warnings.append(wmsg)

        tasks.append({
            "id": s.id,
            "index": s.index,
            "prompt": s.prompt,
            "negative": s.negative,
            "width": w,
            "height": h,
            "length": length,
            "duration": round(length / 24.0, 3),
            "seed": s.resolved_seed,
            "ref_image_keys": list(layout["ref_image_keys"]),
            "picture_tags": dict(picture_tags),
            "layout_order": list(layout["order"]),
            "board_image_index": s.board_image_index,
            "characters": list(s.characters),
            "camera": s.camera,
            "dialogue": s.dialogue,
            "warnings": list(layout.get("warnings", [])),
        })

    if max_shots and max_shots > 0:
        tasks = tasks[:max_shots]

    return tasks


def plan_diff(
    old_meta: Dict[str, Any],
    plan: Plan,
    refs: Optional[RefPack] = None,
) -> str:
    """断点续跑用：对比「上次渲染的镜头指纹」与当前计划，报出新增/变更/删除。

    指纹 = (id, prompt, length, width, height, seed)。
    只有指纹一致的镜头才可跳过，避免「改了提示词却复用旧视频」。

    指纹**不含参考图槽位**（那是装配阶段的产物），所以 refs 可以省略；
    传进来只是为了顺带核对槽位是否漂移。本函数**不会改动传入的 plan**
    （``build_render_tasks`` 默认 mutate=False）。
    """
    def fp(t: Dict[str, Any]) -> str:
        raw = "|".join(str(t.get(k, "")) for k in
                       ("id", "prompt", "length", "width", "height", "seed"))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    old = {d.get("id"): fp(d) for d in (old_meta.get("done") or [])}
    new = {t["id"]: fp(t) for t in build_render_tasks(
        plan, refs=refs, base_seed=int(old_meta.get("base_seed", 0) or 0))}


    added = [k for k in new if k not in old]
    changed = [k for k in new if k in old and old[k] != new[k]]
    removed = [k for k in old if k not in new]
    reusable = [k for k in new if k in old and old[k] == new[k]]

    return (
        f"断点续跑对比：可复用 {len(reusable)} 镜 / 新增 {len(added)} / "
        f"已变更 {len(changed)} / 已删除 {len(removed)}\n"
        f"  可复用: {reusable}\n  新增: {added}\n  变更: {changed}\n  删除: {removed}"
    )
