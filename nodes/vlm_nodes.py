"""VLM 层节点：把「看图说人话」接进 ComfyUI 工作流。

链路定位
--------
::

    漫画页 IMAGE ─┬─► ① 切分镜格 ─► board ─┐
                  │                        ├─► ⑬ VLM 漫画解读 ─► ⑭ VLM 转分镜计划 ─► ④ ▶ ⑤ ▶ ⑫ ▶ 官方多参节点
                  └────────────────────────┘

为什么拆成两个节点
------------------
**VLM 调用贵且慢（实测约 3k tokens/页、11s/页），而「结果 → 分镜计划」的转换是纯本地计算。**
拆开之后：

* 一次解读，反复调参（画幅 / 时长缩放 / 是否强制 shots / 是否保留非叙事页）
  —— 不用重新花钱；
* VLM 原始结果可以存档、人工核对、给别的节点复用。

为什么 VLM 只填字段、不编号
---------------------------
H3 的 ``<Picture i>`` 按**追加顺序**编号，错一位 = 整片串脸，而且**不报错**。
所以序号由 ⑤ 提示词装配在装配时注入，VLM **只许产出结构化字段**。
``core.vlm.validate_page`` 会二次校验，出现 ``<Picture i>`` 标签的整页拒收。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from comfy_api.latest import io

try:
    from ._io_types import board_in, plan_out, vlm_in, vlm_out
except ImportError:  # pragma: no cover
    from _io_types import board_in, plan_out, vlm_in, vlm_out

try:
    from ..core import script_parse as sp
    from ..core import types as _T
    from ..core import vlm as _V
    from ..core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from ..core.ui import safe_display_name as _inp, label_input as _L
except ImportError:  # pragma: no cover
    from core import script_parse as sp
    from core import types as _T
    from core import vlm as _V
    from core.ui import NODE_LABELS as _NL, OUTPUT_LABELS as _OL
    from core.ui import safe_display_name as _inp, label_input as _L


# ==========================================================================
# 内部工具
# ==========================================================================
def _to_np_batch(image) -> Any:
    """``IMAGE`` → ``[B,H,W,C]`` 的 numpy 数组（float 0~1 或 uint8 原样保留）。"""
    import numpy as np

    a = image
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[None, ...]
    if a.ndim != 4:
        raise ValueError(f"IMAGE 输入应为 [B,H,W,C] 或 [H,W,C]，实得 {a.shape}")
    return a


def _page_boxes_from_board(board: Any) -> Dict[int, List[Tuple[int, List[int]]]]:
    """``board`` → ``{page_index: [(全局格号, [x,y,w,h]), ...]}``。

    ★ 用 ``Panel.index`` 作格号 —— 它是**整批全局递增**的，正是 ④ 绑定用的号。
    直接把全局号告诉 VLM（写进提示词的格坐标表），就省掉了「事后重编号」
    这一步；而事后换算一旦漏做，就是「绑错格图」且**不报错**。
    """
    out: Dict[int, List[Tuple[int, List[int]]]] = {}
    for p in (getattr(board, "panels", None) or []):
        pi = int(getattr(p, "page_index", 0) or 0)
        out.setdefault(pi, []).append((int(p.index), list(p.box)))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return out


def _self_split_page(page: Any) -> List[List[int]]:
    """没接 board 时自己切格（与 ① 默认参数一致），返回本页框列表。"""
    import numpy as np

    from ..core import panel_split as _ps
    a = np.asarray(page)
    if a.dtype != np.uint8:
        a = np.clip(a.astype("float32") * 255.0, 0, 255).astype("uint8")
    gray = _ps.to_gray_u8(a)
    boxes, _warns = _ps.split_page_boxes(
        gray, mode="auto", reading_order="rtl",
        frame="component", deoverlap=True)
    return [list(b) for b in boxes]


# ==========================================================================
# ⑬ VLM 漫画解读
# ==========================================================================
class H3MangaVLMScribe(io.ComfyNode):
    """把漫画页交给视觉语言模型，逐格读出「画面内容 / 角色 / 台词 / 一格几镜」。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaVLMScribe",
            display_name=_NL["H3MangaVLMScribe"],
            category="H3 Manga/编排",
            description=(
                "★ 把漫画页交给 **VLM（视觉语言模型）**，逐格读出结构化数据：\n"
                "  画面内容 / 动作 / 运镜 / 出场角色 / 台词草稿 / 一格几镜 / 页型。\n"
                "这补上了整条链路里**唯一还靠人肉**的一环（③ 原本要从零手写全表）。\n\n"
                "三个设计要点：\n"
                "① **VLM 只填字段、不编号** —— `<Picture i>` 由 ⑤ 装配时注入，\n"
                "   出现该标签的整页会被拒收（防串脸）。\n"
                "② **格号用 ① 的全局号**（直接把编号写进提示词），天然与 ④ 对齐。\n"
                "③ **字段分级采信** —— 格数/内容/角色/页型跨模型稳定；\n"
                "   `shots`（一格几镜）不稳定（同一页可能跑出不同镜数），\n"
                "   所以 ⑭ 提供 `force_shots` 统一覆盖。\n\n"
                "免费拿到的一个信号：**page_kind**（叙事/封面/设定页），\n"
                "几何切格完全看不出这一点，而它能让设定页不出镜头、不白烧显存。\n\n"
                "API 兼容 OpenAI 协议（Qwen-VL / GLM-4V / GPT-4o 均可）。"
            ),
            inputs=[
                io.Image.Input("image", display_name=_L("image"),
                               tooltip="漫画页（可多页批次）。会按页逐张送给 VLM。"),
                board_in(name="board", display_name="分镜格", optional=True,
                         tooltip="① 的分镜格。**强烈建议接上**：\n"
                                 "接了就用 ① 的格坐标 + 全局格号（与 ④ 绑定一致）；\n"
                                 "不接则本节点自己按同一套参数切格，"
                                 "但两处参数不一致时会出现格号错位。"),
                _inp(io.String.Input, "characters", _L("characters"),
                     default="", multiline=True,
                     tooltip="角色闭集（逗号/换行分隔）。\n"
                             "★ VLM 的 characters 只能从这里选，表外名字会被剔除。\n"
                             "★ 写**简称**即可（闭集里有「纲吉」就写「纲吉」）；\n"
                             "   模型若输出全名「泽田纲吉」，代码会做子串兜底匹配。"),
                _inp(io.String.Input, "char_alias", _L("char_alias"),
                     default="",
                     tooltip="别名表，应对模型输出繁体或异体写法。\n"
                             "格式：`繁体或异体=闭集里的名字`，逗号分隔。\n"
                             "例：`綱吉=纲吉,獄寺=狱寺`\n"
                             "（代码已内置常见繁简转换，这里只补特殊情况）"),
                _inp(io.String.Input, "base_url", _L("base_url"),
                     default=_V.DEFAULT_BASE_URL,
                     tooltip="OpenAI 兼容端点。默认阿里云百炼。\n"
                             "★ 必须用国内站（国际站会 invalid_api_key）。\n"
                             "Ollama 本地：http://127.0.0.1:11434/v1"),
                _inp(io.String.Input, "model", _L("model"),
                     default=_V.DEFAULT_MODEL,
                     tooltip="视觉模型名。实测首选 qwen3.5-omni-plus\n"
                             "（比 qwen3-vl-plus 快约 3 倍且同样准）。\n"
                             "⚠ qwen3.8-max / deepseek-v4.1-flash **没有视觉能力**，\n"
                             "   传图会返回空 —— 它们只能做纯文本任务。"),
                _inp(io.String.Input, "api_key", _L("api_key"), default="",
                     tooltip="留空则读环境变量（见下一项）。\n"
                             "★ 别把 key 写进工作流文件 —— 会被提交进 git。"),
                _inp(io.String.Input, "api_key_env", _L("api_key_env"),
                     default="VLM_API_KEY",
                     tooltip="从哪个环境变量读 key（api_key 非空时以它为准）"),
                _inp(io.Int.Input, "max_edge", _L("max_edge"),
                     default=1600, min=0, max=4096, step=32,
                     tooltip="发送前把长边缩到此值（0=不缩）。\n"
                             "1600 是精度与花费的平衡点；调到 1200 以下会伤竖排小字\n"
                             "的识别，需抽样验证。"),
                _inp(io.Float.Input, "temperature", _L("temperature"),
                     default=0.2, min=0.0, max=2.0, step=0.05,
                     tooltip="采样温度。★ 实测降到 0 也**稳不住 shots**\n"
                             "（那是判据歧义，不是采样随机性），所以别指望这个。"),
                _inp(io.Int.Input, "timeout", _L("timeout"),
                     default=180, min=10, max=900, step=10,
                     tooltip="单次请求超时（秒）"),
                _inp(io.Int.Input, "max_pages", _L("max_pages"),
                     default=0, min=0, max=9999, step=1,
                     tooltip="只处理前 N 页（0=全部）。先设 1~3 验证连通性与效果。"),
                _inp(io.Boolean.Input, "strict_coverage", _L("strict_coverage"),
                     default=True,
                     tooltip="严格覆盖：有格没被解读就**整页拒收**（默认开）。\n"
                             "★ 实测有模型读了 8 格里的 4 格就交结果（还编出 10 镜），\n"
                             "   这种静默漏读必须拦住。调试时可关掉。"),
                _inp(io.Boolean.Input, "inspection_fallback",
                     _L("inspection_fallback"), default=True,
                     tooltip="内容审核回退阶梯（默认开）。\n"
                             "战斗类页面会被云服务判定为不当内容而拒绝，且判定\n"
                             "与模型/分辨率相关、非确定 —— 开启后会自动\n"
                             "「缩图 → 换模型」重试，实测能救回大部分被拦页。"),
                _inp(io.Boolean.Input, "use_proxy", _L("use_proxy"), default=False,
                     tooltip="走系统代理（默认不走）。\n"
                             "★ 代理会把「连不上」伪装成 502，排查时容易被误导。"),
            ],
            outputs=[
                vlm_out(display_name=_OL["vlm"]),
                io.String.Output(display_name=_OL["report"]),
            ],
        )

    # ★ 加缓存：同一张图 + 同一组参数不重复调 API（实测 ~3k tokens/页、11s/页）。
    @classmethod
    def IS_CHANGED(cls, image=None, characters="", char_alias="", base_url="",
                   model="", max_edge=1600, temperature=0.2, max_pages=0,
                   strict_coverage=True, **kw):
        import hashlib
        h = hashlib.sha256()
        try:
            import numpy as np
            a = image
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            a = np.ascontiguousarray(np.asarray(a))
            h.update(str(a.shape).encode())
            h.update(a.tobytes()[:4_000_000])      # 前 4MB 足以区分页面
        except Exception:  # noqa: BLE001
            h.update(repr(type(image)).encode())
        for v in (characters, char_alias, base_url, model, max_edge,
                  temperature, max_pages, strict_coverage):
            h.update(repr(v).encode())
        return h.hexdigest()

    @classmethod
    def execute(cls, image, board=None, characters="", char_alias="",
                base_url=_V.DEFAULT_BASE_URL, model=_V.DEFAULT_MODEL,
                api_key="", api_key_env="VLM_API_KEY", max_edge=1600,
                temperature=0.2, timeout=180, max_pages=0,
                strict_coverage=True, inspection_fallback=True,
                use_proxy=False) -> io.NodeOutput:
        key = (api_key or "").strip() or os.environ.get(
            (api_key_env or "VLM_API_KEY").strip(), "")
        if not key:
            raise ValueError(
                "缺 API key。二选一：\n"
                "  ① 把 api_key 填上（不建议 —— 会写进工作流文件里）；\n"
                f"  ② 设环境变量 {api_key_env or 'VLM_API_KEY'} 后重启 ComfyUI。\n"
                "（ComfyUI 在启动时读环境变量，改完必须重启。）"
            )
        if not (model or "").strip():
            raise ValueError("model 不能为空。")

        chars = _V.parse_names(characters)
        aliases = _V.parse_aliases(char_alias)
        arr = _to_np_batch(image)
        n_pages_total = int(arr.shape[0])
        n_pages = n_pages_total if int(max_pages) <= 0 else min(
            n_pages_total, int(max_pages))

        # ---- 每页的格（格号 + 框）----
        warns: List[str] = []
        by_page: Dict[int, List[Tuple[int, List[int]]]] = {}
        offset_note = ""
        if board is not None and getattr(board, "panels", None):
            by_page = _page_boxes_from_board(board)
            n_board_pages = int(getattr(board, "pages", 0) or 0)
            offset_note = "格号来自 ① 的 board（全局号）"
            if n_board_pages and n_board_pages != n_pages_total:
                warns.append(
                    f"board 有 {n_board_pages} 页，而 image 有 {n_pages_total} 页 "
                    "—— 不一致，可能接了不同的上游")

        pages: List[Dict[str, Any]] = []
        errors: List[str] = []
        all_warns: List[str] = list(warns)
        failed: List[str] = []
        usage_tot: Dict[str, int] = {}

        opener = _V.make_opener(bool(use_proxy))
        t0 = time.time()
        global_acc = 0            # 未接 board 时的全局格号累加器

        for pi in range(n_pages):
            name = f"page{pi:03d}"
            page = arr[pi]

            if by_page:
                pairs = by_page.get(pi) or []
                ids = [t[0] for t in pairs]
                boxes = [t[1] for t in pairs]
                if not boxes:
                    warns.append(f"{name}：board 里这一页没有分镜格，已跳过")
                    continue
            else:
                boxes = _self_split_page(page)
                if not boxes:
                    failed.append(name)
                    errors.append(f"{name}：没切出任何分镜格")
                    continue
                # 自切格：累计全局偏移（与 ① 的 Panel.index 语义一致）
                ids = list(range(global_acc, global_acc + len(boxes)))
                global_acc += len(boxes)
                offset_note = "格号由本节点自切格算出（未接 board）"

            try:
                b64, mime, W, H = _V.tensor_to_b64(page, max_edge=int(max_edge))

                def _reencode(mult: float, _page=page, _edge=int(max_edge)):
                    e = int(_edge * mult)
                    if e <= 0:
                        raise RuntimeError("max_edge=0，无法用缩图做审核回退")
                    return _V.tensor_to_b64(_page, max_edge=e)

                res = _V.call_with_fallback(
                    b64, mime, W, H, boxes, chars, name,
                    opener=opener, base_url=base_url, api_key=key,
                    model=model, ids=ids, timeout=int(timeout),
                    temperature=float(temperature),
                    reencode=_reencode,
                    allow_inspection_fallback=bool(inspection_fallback),
                    log=lambda m: None,
                )
                for n in res["notes"]:
                    all_warns.append(f"{name}：{n}")
                u = res.get("usage") or {}
                for k, v in u.items():
                    if isinstance(v, int):
                        usage_tot[k] = usage_tot.get(k, 0) + v

                panels, errs, w2 = _V.validate_page(
                    res["obj"], ids, chars, name, aliases=aliases,
                    strict_coverage=bool(strict_coverage))
                all_warns.extend(w2)
                if errs:
                    all_warns.extend(f"{name}：{e}" for e in errs)
                    errors.extend(errs)
                    failed.append(name)
                    continue

                pages.append({
                    "page": len(pages), "page_name": name,
                    "page_size": [W, H],
                    "page_kind": str(res["obj"].get("page_kind") or "").strip(),
                    "summary": str(res["obj"].get("summary") or "").strip(),
                    "panels": panels,
                    "model_used": res["model"],
                    "edge_mult": res["edge_mult"],
                })
            except Exception as e:  # noqa: BLE001
                failed.append(name)
                errors.append(f"{name}：{type(e).__name__}: {e}")

        elapsed = time.time() - t0
        result = _T.VLMResult(
            pages=pages, model=str(model), chars=chars, aliases=aliases,
            warnings=all_warns, errors=errors, failed_pages=failed,
            usage=usage_tot, seconds=elapsed,
            source_note="⑬ VLM 漫画解读",
        )
        report = _V.vlm_report(
            pages, model=str(model), usage=usage_tot, seconds=elapsed,
            warnings=all_warns, errors=errors, failed_pages=failed)
        report += (f"\n\n处理范围: 前 {n_pages} / {n_pages_total} 页"
                   f"（max_pages={int(max_pages)}）")
        report += f"\n{offset_note}"
        report += ("\n\n下一步：接 ⑭「VLM 转分镜计划」→ ④ → ⑤ → ⑫ → 官方多参节点。")
        return io.NodeOutput(result, report, ui={"h3manga_panel": [report]})


# ==========================================================================
# ⑭ VLM 转分镜计划
# ==========================================================================
class H3MangaVLMPlan(io.ComfyNode):
    """把 VLM 解读结果转成 ④/⑤ 能直接吃的分镜计划（纯本地计算，不花钱）。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MangaVLMPlan",
            display_name=_NL["H3MangaVLMPlan"],
            category="H3 Manga/编排",
            description=(
                "把 ⑬ 的 VLM 结果转成分镜计划（`H3_MANGA_PLAN`），可直接接 ④ 绑定。\n\n"
                "★ **这一步不调用 API**，所以可以反复调参而不花钱：\n"
                "  画幅 / 默认时长 / 时长缩放 / 是否保留非叙事页 / 强制镜数。\n\n"
                "输出里给一份 `分镜文本`（块文本），既能贴回 ③ 手改，\n"
                "也能存档 —— 它是「VLM 判断 + 你的调整」的最终快照。"
            ),
            inputs=[
                vlm_in("vlm", display_name=_OL["vlm"]),
                _inp(io.Boolean.Input, "include_non_narrative",
                     _L("include_non_narrative"), default=False,
                     tooltip="是否把封面/设定页/目录页也做成镜头（默认否）。\n"
                             "★ 实测一卷 207 页里有 16 页是这类，跳过可省 64 格\n"
                             "   的渲染时间，而且它们本来就不该动起来。"),
                _inp(io.Int.Input, "force_shots", _L("force_shots"),
                     default=0, min=0, max=8, step=1,
                     tooltip="0 = 保留 VLM 判断的「一格几镜」（默认）；\n"
                             "≥1 = **全部强制**成该值（想省时间就设 1）。\n"
                             "★ 该字段跨次运行不稳定，强制统一是让它可复现的最省事办法。"),
                _inp(io.Float.Input, "duration_scale", _L("duration_scale"),
                     default=1.0, min=0.2, max=3.0, step=0.05,
                     tooltip="把 VLM 给的时长建议整体缩放。\n"
                             "★ 实测 VLM 偏长（一卷给到 89 分钟），出片前压到\n"
                             "   0.5~0.6 可以显著缩短渲染时间。\n"
                             "   缩放后仍会被 17n+5 对齐。"),
                _inp(io.Float.Input, "default_duration", _L("default_duration"),
                     default=5.0, min=0.2, max=60.0, step=0.1,
                     tooltip="VLM 没给时长的格用这个（一般用不到，VLM 总会给）。"),
                _inp(io.Combo.Input, "size_mode", _L("size_mode"),
                     options=list(_T.H3_SIZE_MODES), default=_T.SIZE_MODE_PRESET,
                     tooltip="preset = 按官方 adapt_canvas 规则选比例（推荐）\n"
                             "page   = 跟随原漫画页比例（需要接 board，本节点未接\n"
                             "         时会退回 preset）\n"
                             "text   = 用文本里的 default_size"),
                _inp(io.Combo.Input, "size_preset", _L("size_preset"),
                     options=list(_T.H3_SIZE_PRESETS.keys()),
                     default=_T.SIZE_PRESET_DEFAULT,
                     tooltip="官方规范画幅：16:9 → 1344x768，4:3 → 1024x768 …"),
                _inp(io.Float.Input, "size_scale", _L("size_scale"),
                     default=_T.DEFAULT_SIZE_SCALE, min=0.1, max=1.0, step=0.05,
                     tooltip="在官方规范画幅上再按比例缩。8GB 卡建议 0.5（672x384）。"),
                _inp(io.String.Input, "episode", _L("episode"), default="",
                     tooltip="剧集名，只影响报告与文本头部"),
                _inp(io.String.Input, "style", _L("style"),
                     default="manga_anime",
                     tooltip="全局风格 id（写进文本的 # style 头）"),
            ],
            outputs=[
                plan_out(display_name=_OL["plan"]),
                io.String.Output(display_name=_OL["shot_text"]),
                io.String.Output(display_name=_OL["report"]),
            ],
        )

    @classmethod
    def execute(cls, vlm, include_non_narrative=False, force_shots=0,
                duration_scale=1.0, default_duration=5.0,
                size_mode=_T.SIZE_MODE_PRESET,
                size_preset=_T.SIZE_PRESET_DEFAULT,
                size_scale=_T.DEFAULT_SIZE_SCALE,
                episode="", style="manga_anime") -> io.NodeOutput:
        if vlm is None or not getattr(vlm, "pages", None):
            raise ValueError(
                "需要 ⑬「VLM 漫画解读」的输出。请把它的 `VLM 结果` 接过来。")

        pages = _V.transform_pages(
            vlm.pages, force_shots=int(force_shots),
            duration_scale=float(duration_scale))
        text = _V.to_block_text(
            pages, episode=str(episode or vlm.episode or ""),
            style=str(style or ""),
            include_non_narrative=bool(include_non_narrative),
            source_note="⑬ VLM 漫画解读")

        size_override = None
        mode = str(size_mode or _T.SIZE_MODE_PRESET).lower()
        if mode != _T.SIZE_MODE_TEXT:
            try:
                size_override = _T.resolve_canvas(
                    size_mode=(_T.SIZE_MODE_PRESET
                               if mode == _T.SIZE_MODE_PAGE else mode),
                    preset=str(size_preset), scale=float(size_scale))
            except Exception as e:  # noqa: BLE001
                size_override = None
                text = f"# 画幅解析失败（{e}），用文本里的 default_size\n" + text

        plan = sp.parse_text(text, note="vlm",
                             default_duration=float(default_duration),
                             size_override=size_override)
        plan.source_note = "VLM → Plan"

        st = _V.count_pages(pages,
                            include_non_narrative=bool(include_non_narrative))
        report = [
            f"VLM → 分镜计划：{len(plan.shots)} 镜",
            "=" * 52,
            f"  来自        : {len(pages)} 页 VLM 结果"
            f"（跳过非叙事页 {st['skipped_pages']} 页）",
            f"  格 / 镜     : {st['panels']} 格 / {st['shots']} 镜",
            f"  时长        : {plan.total_seconds() / 60:.1f} 分钟"
            f"（{plan.total_frames()} 帧 @ {_T.H3_FPS}fps）",
            f"  画幅        : "
            + (f"{size_override[0]}x{size_override[1]}（{size_override[2]}）"
               if size_override else "按文本 default_size"),
            f"  强制镜数    : {int(force_shots) or '保留 VLM 判断'}",
            f"  时长缩放    : ×{float(duration_scale):g}",
            "",
            "前 8 镜：",
        ]
        for s in plan.shots[:8]:
            report.append(
                f"  [{s.index:>3}] panel={s.panel_index} "
                f"{s.duration:.2f}s {s.width}x{s.height} "
                f"{s.camera:<14} {','.join(s.characters) or '-':<12} "
                f"{(s.content or s.action or '')[:26]}")
        if len(plan.shots) > 8:
            report.append(f"  …共 {len(plan.shots)} 镜（见 ⑨ 计划预览）")
        if plan.warnings:
            report.append("")
            report.append(f"警告 {len(plan.warnings)} 条：")
            for w in plan.warnings[:10]:
                report.append(f"  ! {w}")
        report.append("")
        report.append("下一步：接 ④ 分镜绑定 → ⑤ 提示词装配 → ⑫ → 官方多参节点。")
        rep = "\n".join(report)
        return io.NodeOutput(plan, text, rep, ui={"h3manga_panel": [rep]})
