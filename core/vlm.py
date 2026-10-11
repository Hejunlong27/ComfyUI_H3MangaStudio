"""VLM 页级解读：提示词 / 调用 / 校验 / 输出转换。

**这个模块是唯一真源**：离线批处理工具 ``tools/vlm_shotlist.py`` 与
ComfyUI 节点 ``nodes/vlm_nodes.py`` 都从这里取逻辑 —— 两份实现必然漂移，
而这类漂移是「静默错」（比如一边做了繁简归一化、另一边没做），所以只留一份。

不 import comfy / torch，可以脱离 ComfyUI 单测（第 1 层）。

设计要点
--------
1. **三条硬约束由代码二次校验，不信任模型**：
   禁 ``<Picture i>`` 标签 / 角色必须在闭集内 / panel 必须在切格范围内。
   不合规**整页拒收**，绝不静默放过 —— H3 的序号是「静默失败」型的。
2. ★ **直接把「全局格号」告诉 VLM**：① 的 ``Panel.index`` 是整批全局递增的，
   而 ④ 绑定用的是全局号。所以提示词里就写全局号，模型返回的自然就是全局号，
   **不需要事后换算**（离线工具早期版本按本页 0..k-1 问、再 retxt 重编号，
   这里从源头消除该风险）。
3. **按字段分级采信**（实测结论）：``page_kind`` / 格数 / ``content`` /
   ``characters`` / ``camera`` 跨模型稳定；``shots``（一格几镜）与
   ``confidence`` 不稳定（同页 3 次跑出 3/4/5 镜，降 temperature 到 0 无效
   —— 那是判据歧义，不是采样随机性）。所以要为 ``shots`` 提供强制覆盖开关。
4. **内容审核回退**：云服务会拦战斗页（``data_inspection_failed``），且
   与模型/分辨率相关、非确定 —— 给一条阶梯而不是丢掉这页。
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 异常
# --------------------------------------------------------------------------
class VLMError(RuntimeError):
    """VLM 调用/校验的总异常。"""


class VLMInspectionError(VLMError):
    """云服务端内容审核拒绝 —— 同一张图必然再被拦，重试无意义。"""


class VLMFatalError(VLMError):
    """客户端错误（模型名错 / key 无效 / 参数错）—— 重试无意义。"""


# --------------------------------------------------------------------------
# 提示词（三条硬约束同时写进提示词 + 代码二次校验）
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一名专业的漫画分镜分析师，为「漫画改编动画」项目服务。

我会给你一张漫画页，以及这一页已经切好的分镜格坐标（像素，左上角为原点，
x 向右、y 向下）。请你**逐格**理解画面，输出可直接用于视频生成的结构化数据。

# 输出字段（每个分镜格一条）
- panel         : 整数，必须原样使用我给你的格子编号（不要自己重新编号）
- shots         : 整数，这一格应该拆成几个镜头（见下方判据）
- content       : 一句话描述这一格「画了什么」（画面内容，客观描述，不要加评论）
- action        : 这一格里的动作或镜头运动，动词开头
- camera        : 运镜建议，只能从这个表里选：
                  static / slow_push_in / push_in / pull_out / pan_left / pan_right
                  / tilt_up / tilt_down / close_up / medium_shot / wide_shot
- characters    : 出场角色名数组，**只能从我给你的角色表里选**；表里没有的留空数组。
                  ★ 一律写成**简体中文**，不要写繁体（读图读到的繁体名要转成简体再比对）
                  ★ 用**角色表里写的那个名字**，不要写全名（表里写「纲吉」就写「纲吉」，
                    不要写「泽田纲吉」）
- dialogue_draft: 台词草稿数组（你有把握才写；读不清就给空数组，不要编）。
                  ★ 台词**照抄画面原文**（原文是繁体就写繁体），不要翻译、不要改写
- duration_hint : 建议时长（秒，整数 3~10）—— **每一镜**的时长
- confidence    : 你对这一格理解的把握，0~1

# ★ shots 大于 1 时，必须分段（否则这一格只会生成两个一模一样的镜头）
当 shots = N（N ≥ 2）时，`content`、`action`、`dialogue_draft` 都要按镜头
**用 ` | ` 分成 N 段，第 1 段对应第 1 个镜头**。例如 shots=2 时：

    "content": "左侧：少女仰头特写 | 右侧：餐桌旁两人争吵",
    "action":  "镜头贴近少女脸部 | 后退拉出餐桌全景",
    "dialogue_draft": ["好吃！", "今天也谢谢你。"]   ← 数组按顺序切分即可，不必加竖线

段落数必须等于 shots。做不到就说明这一格其实不该拆 —— 那就写 shots = 1。

# shots 的判据（最容易分歧的一项）
- 一格内如果**只有单一视点、单一动作** → shots = 1
- 一格内如果有**两个以上互不相连的视点**（例如左半是人物特写、右半是另一场景），
  或者同一个格子里画了**连续动作的几个时间相** → shots 按视点/时间相数量给
- 判断依据是**画面结构**，不是台词数量。同一场景里多人对话仍然算 1 镜。
- 拿不准就写 1（保守优于过度拆分）

# 硬性禁止
- 禁止输出任何 <Picture i> / <Video k> / <Audio j> 标签
- 禁止输出 Markdown 代码块以外的解释文字
- 禁止编造画面里没有的内容

# 输出格式
严格输出一个 JSON 对象，不要有任何额外文字：
{
  "page_kind": "narrative | cover | info",
  "panels": [ { ...字段如上... } ],
  "summary": "这一页在剧情里起什么作用，一句话"
}
其中 page_kind：叙事页 = narrative；扉页/封面 = cover；设定页/资料页 = info。
"""

USER_TMPL = """这一页的文件名：{name}
图片尺寸：宽 {W} × 高 {H}

这一页已经切好的分镜格（共 {n} 格，按阅读顺序；请**原样使用**这些编号）：
{boxes}

可用角色表（characters 只能从中选）：{chars}
"""

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
BANNED_TAG = re.compile(r"<\s*(Picture|Video|Audio)\b", re.I)

CAMERAS = ("static", "slow_push_in", "push_in", "pull_out", "pan_left",
           "pan_right", "tilt_up", "tilt_down", "close_up", "medium_shot",
           "wide_shot")

MAX_SHOTS_PER_PANEL = 8

# ★ 内容审核回退阶梯：(模型或 None=沿用主模型, 长边倍率)
#
# 实测（某战斗页）：主模型在 edge=1600/900 都被拦，但**换 qwen3-vl-plus
# 保持 1600 就过**，而主模型缩到 600 也过。说明审核判定与模型、分辨率都相关，
# 且非确定 —— 所以给一条阶梯，而不是把被拦的页直接丢掉。
INSPECTION_FALLBACKS: Tuple[Tuple[Optional[str], float], ...] = (
    (None, 0.4),                 # 沿用主模型，缩到长边 40%
    ("qwen3-vl-plus", 1.0),      # 换模型，保持全分辨率
)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"

# 时长硬夹（秒）—— 与提示词里的 3~10 一致
DURATION_MIN, DURATION_MAX = 3, 10

# --------------------------------------------------------------------------
# 角色名归一化
# --------------------------------------------------------------------------
# ★ 繁体 → 简体 归一化表。
#
# 为什么必须有：实测某个便宜模型会把角色名写成繁体（`綱吉 / 獄寺 / 藍波`），
# 而角色闭集是简体 → **名字全被剔除**，参考图槽位随之丢失，而且只在
# warning 里出现、不报错。本机没有 zhconv/opencc（零第三方依赖约束），
# 所以内置一张小表兜底，同时：提示词里强制简体、允许用户传别名表。
TRAD2SIMP = str.maketrans({
    "綱": "纲", "獄": "狱", "藍": "蓝", "優": "优", "雲": "云",
    "澤": "泽", "風": "风", "廣": "广", "島": "岛", "國": "国", "學": "学",
    "會": "会", "體": "体", "來": "来", "個": "个", "這": "这", "們": "们",
    "無": "无", "辦": "办", "贏": "赢", "嗎": "吗", "麼": "么", "買": "买",
    "賣": "卖", "師": "师", "復": "复", "對": "对", "決": "决", "戰": "战",
    "鬥": "斗", "爭": "争", "轉": "转", "過": "过", "還": "还", "讓": "让",
    "說": "说", "話": "话", "語": "语", "讀": "读", "書": "书", "寫": "写",
    "見": "见", "覺": "觉", "現": "现", "發": "发", "經": "经", "頭": "头",
    "長": "长", "門": "门", "開": "开", "關": "关", "間": "间", "問": "问",
    "際": "际", "陳": "陈", "陽": "阳", "葉": "叶", "華": "华", "蘇": "苏",
    "萬": "万", "與": "与", "從": "从", "愛": "爱", "樂": "乐", "歲": "岁",
    "廳": "厅", "舊": "旧", "燈": "灯", "點": "点", "線": "线", "練": "练",
    "級": "级", "紅": "红", "綠": "绿", "鐵": "铁", "銀": "银", "馬": "马",
    "鳥": "鸟", "魚": "鱼", "龍": "龙", "鳳": "凤", "龜": "龟", "羅": "罗",
    "爾": "尔", "納": "纳", "諾": "诺", "蘭": "兰", "骷": "骷", "髏": "髅",
    "奈": "奈", "浦": "浦", "後": "后", "樣": "样", "集": "集", "活": "活",
})


def norm_name(s: Any) -> str:
    """繁→简 + 去空白。用于角色闭集匹配（不用于展示）。"""
    return str(s or "").strip().translate(TRAD2SIMP)


def parse_aliases(text: str) -> Dict[str, str]:
    """``"綱吉=纲吉,藍波=蓝波"`` → ``{"綱吉": "纲吉", "藍波": "蓝波"}``。"""
    out: Dict[str, str] = {}
    for part in (text or "").split(","):
        if "=" in part:
            a, b = part.split("=", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                out[a] = b
    return out


def parse_names(text: str) -> List[str]:
    """逗号/分号/换行分隔的名字列表（去重保序）。"""
    parts = re.split(r"[,，;；\n]+", text or "")
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# --------------------------------------------------------------------------
# 网络
# --------------------------------------------------------------------------
def make_opener(use_proxy: bool = False):
    """默认**不走代理**。

    ★ 实测踩过：系统代理会把「服务没起来」伪装成
    ``502 Tunnel connection failed``，排查时会被误导到错误方向。
    """
    if use_proxy:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def extract_text(resp: Dict[str, Any]) -> str:
    try:
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        c = msg.get("content")
        if isinstance(c, list):          # 有的网关返回分段内容
            return "".join(x.get("text", "") for x in c if isinstance(x, dict))
        if isinstance(c, str):
            return c
        return ch.get("text", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def extract_json(text: str):
    """从模型回复里抠出 JSON。容忍 ```json 包裹与前后废话。"""
    if not text:
        raise VLMError("回复为空")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise VLMError("回复里找不到 JSON 对象")
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError as e:
        raise VLMError(f"回复里的 JSON 解析失败：{e}") from e


def call_vlm(opener, base_url: str, api_key: str, model: str,
             sys_prompt: str, user_text: str, b64: str, mime: str,
             timeout: int = 180, retries: int = 3, temperature: float = 0.2,
             max_tokens: int = 4096) -> Tuple[str, Dict[str, Any]]:
    """调一次 OpenAI 兼容的 ``chat/completions``，返回 (文本, 原始响应)。

    ★ 内容审核拒绝与 4xx **不重试**：同一张图/同一个错必然再犯，
      重试只是白等（实测一次白等 90 秒 → 改成 12 秒返回明确原因）。
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(1, max(1, int(retries)) + 1):
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST")
        try:
            with opener.open(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return extract_text(data), data
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            last = f"HTTP {e.code}: {detail}"
            low = detail.lower()
            if "data_inspection_failed" in low or "inappropriatecontent" in low:
                raise VLMInspectionError(
                    "内容审核拒绝（云服务端判定图片含不当内容）—— 原始响应："
                    + detail[:200]
                ) from e
            if e.code in (400, 401, 403, 404, 422):
                raise VLMFatalError(f"不可重试的请求错误 —— {last}") from e
        except VLMError:
            raise
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise VLMError(f"调用失败（{retries} 次）：{last}")


# --------------------------------------------------------------------------
# 图片编码
# --------------------------------------------------------------------------
def ndarray_to_b64(arr, max_edge: int = 1600,
                   jpeg_quality: int = 88) -> Tuple[str, str, int, int]:
    """``uint8`` 的 **RGB** 数组 → ``(base64, mime, 宽, 高)``。

    ★ 颜色顺序：ComfyUI 的 IMAGE 张量是 **RGB**，而 ``cv2.imencode`` 按
    **BGR** 编码 —— 不转换就会红蓝颠倒送给模型，直接影响识别质量。
    离线工具那边用 ``imdecode`` 拿到的是 BGR，所以它的调用方不要传 RGB。
    """
    import cv2
    import numpy as np

    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        raise ValueError(f"期望 uint8 数组，实得 {a.dtype}")
    h, w = a.shape[:2]
    if max_edge and max(h, w) > int(max_edge):
        s = float(max_edge) / float(max(h, w))
        a = cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
        h, w = a.shape[:2]
    bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY,
                                         int(jpeg_quality)])
    if not ok:
        raise ValueError("cv2.imencode 失败")
    return base64.b64encode(buf.tobytes()).decode("ascii"), "image/jpeg", w, h


def tensor_to_b64(img, max_edge: int = 1600,
                  jpeg_quality: int = 88) -> Tuple[str, str, int, int]:
    """ComfyUI 的 ``IMAGE`` 张量 → ``(base64, mime, 宽, 高)``。

    接受 ``[H,W,C]`` 或 ``[B,H,W,C]``（取第 1 张）；float 0~1 或 uint8。
    """
    import numpy as np

    a = img
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"IMAGE 张量应为 [H,W,C] 或 [B,H,W,C]，实得 shape={a.shape}")
    if a.shape[2] == 4:                      # RGBA → RGB
        a = a[:, :, :3]
    if a.shape[2] != 3:
        raise ValueError(f"只支持 3 通道 RGB，实得 {a.shape[2]} 通道")
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    return ndarray_to_b64(a, max_edge=max_edge, jpeg_quality=jpeg_quality)


def image_bytes_to_b64(raw: bytes, max_edge: int = 1600,
                       jpeg_quality: int = 88) -> Tuple[str, str, int, int]:
    """字节流（PNG/JPG/...）→ ``(base64, mime, 宽, 高)``。

    ★ 用 ``imdecode(np.frombuffer(...))`` 而不是 ``cv2.imread`` ——
    后者在 Windows 上**遇到非 ASCII 路径会静默返回 None**（不抛异常），
    而本项目的漫画素材路径几乎必然含中文。
    """
    import cv2
    import numpy as np

    mime = "image/png"
    head = raw[:16].lower()
    if head.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    elif head.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        mime = "image/webp"

    if not max_edge:
        h = w = 0
        return base64.b64encode(raw).decode("ascii"), mime, w, h

    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # 解不开就原样发（至少不会静默丢）
        return base64.b64encode(raw).decode("ascii"), mime, 0, 0
    h, w = img.shape[:2]
    if max(h, w) > int(max_edge):
        s = float(max_edge) / float(max(h, w))
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,
                                             int(jpeg_quality)])
        if not ok:
            return base64.b64encode(raw).decode("ascii"), mime, 0, 0
        return base64.b64encode(buf.tobytes()).decode("ascii"), "image/jpeg", \
            img.shape[1], img.shape[0]
    h, w = img.shape[:2]
    return base64.b64encode(raw).decode("ascii"), mime, w, h


# --------------------------------------------------------------------------
# 提示词素材
# --------------------------------------------------------------------------
def boxes_text(boxes: Sequence[Sequence[int]],
               ids: Optional[Sequence[int]] = None) -> str:
    """把切格结果写成提示词里的坐标表。

    ★ ``ids`` 用来传**全局格号** —— 让模型直接沿用，省掉事后重编号
    （事后换算一旦漏做就是「绑错格图」且不报错）。
    """
    ids = list(range(len(boxes))) if ids is None else list(ids)
    return "\n".join(
        f"  格 {ids[i]}: x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}"
        for i, (x, y, w, h) in enumerate(boxes))


def build_user_prompt(name, W, H, boxes, chars, ids=None) -> str:
    return USER_TMPL.format(
        name=name, W=W, H=H, n=len(boxes),
        boxes=boxes_text(boxes, ids),
        chars=", ".join(sorted(chars)) if chars else "(无)")


# --------------------------------------------------------------------------
# 一次「带审核回退」的读取
# --------------------------------------------------------------------------
def call_with_fallback(b64: str, mime: str, W: int, H: int,
                       boxes: Sequence[Sequence[int]],
                       chars: Sequence[str], name: str,
                       opener, base_url: str, api_key: str, model: str,
                       ids: Optional[Sequence[int]] = None,
                       timeout: int = 180, temperature: float = 0.2,
                       fallbacks: Optional[Sequence[Sequence[Any]]] = None,
                       reencode=None,
                       allow_inspection_fallback: bool = True,
                       log=None) -> Dict[str, Any]:
    """读一页，内容审核被拦时按阶梯重试。

    返回 ``{"obj", "model", "edge_mult", "notes", "usage", "attempts"}``。

    ``reencode(mult) -> (b64, mime, w, h)``：审核回退要缩图时由调用方重新编码
    （节点侧是张量重编码、离线工具是文件重编码 —— 用回调解耦）。
    """
    say = log or (lambda _m: None)
    attempts: List[Tuple[str, float]] = [(model, 1.0)]
    if allow_inspection_fallback:
        for mb, mult in (fallbacks if fallbacks is not None
                         else INSPECTION_FALLBACKS):
            attempts.append((model if mb is None else str(mb), float(mult)))

    notes: List[str] = []
    last: Optional[Exception] = None
    for i, (mdl, mult) in enumerate(attempts):
        cur_b64, cur_mime = b64, mime
        if mult != 1.0 and reencode is not None:
            try:
                r = reencode(mult)
                cur_b64, cur_mime = r[0], r[1]
            except Exception as e:  # noqa: BLE001
                notes.append(f"第 {i + 1} 次缩图失败（×{mult}）：{e}")
                continue
        user = build_user_prompt(name, W, H, boxes, chars, ids)
        try:
            text, resp = call_vlm(opener, base_url, api_key, mdl,
                                  SYSTEM_PROMPT, user, cur_b64, cur_mime,
                                  timeout=timeout, temperature=temperature)
            obj = extract_json(text)
            if i > 0:
                notes.append(f"内容审核回退成功：第 {i + 1} 次尝试"
                             f"（模型 {mdl}，长边×{mult}）")
            return {"obj": obj, "model": mdl, "edge_mult": mult,
                    "notes": notes, "usage": (resp.get("usage") or {}),
                    "attempts": i + 1}
        except VLMInspectionError as e:
            last = e
            say(f"第 {i + 1} 次被内容审核拦下（模型 {mdl}，长边×{mult}）")
            notes.append(f"第 {i + 1} 次被内容审核拦下（模型 {mdl}，长边×{mult}）")
            continue
        except Exception as e:  # noqa: BLE001
            last = e
            say(f"第 {i + 1} 次失败（模型 {mdl}，长边×{mult}）：{e}")
            notes.append(f"第 {i + 1} 次失败（模型 {mdl}，长边×{mult}）：{e}")
            continue
    raise VLMError(f"所有 {len(attempts)} 次尝试都失败：{' | '.join(notes)}") \
        from last


# --------------------------------------------------------------------------
# 校验（三条硬约束）
# --------------------------------------------------------------------------
def validate_page(obj: Dict[str, Any], panel_ids: Sequence[int],
                  chars: Sequence[str], page_label: str,
                  aliases: Optional[Dict[str, str]] = None,
                  strict_coverage: bool = True
                  ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """返回 ``(干净的 panel 列表, 错误, 警告)``。有不合规**整页拒收**。

    ``strict_coverage=True`` 时「有格没被解读」升级为**错误**。
    ★ 实测某个便宜模型在读 8 格的页时只解读了 4 格就把结果交出来
      （还编出 10 镜）—— 这种「静默漏读」必须拦住。
    """
    errs: List[str] = []
    warns: List[str] = []
    panels = obj.get("panels")
    if not isinstance(panels, list) or not panels:
        return [], [f"{page_label}：没有 panels 数组"], []

    idset = set(int(x) for x in panel_ids) if panel_ids else None
    out: List[Dict[str, Any]] = []
    seen = set()
    for k, p in enumerate(panels):
        if not isinstance(p, dict):
            errs.append(f"{page_label}：panels[{k}] 不是对象")
            continue
        pid = p.get("panel")
        if not isinstance(pid, int) or isinstance(pid, bool):
            errs.append(f"{page_label}：panels[{k}] 缺 panel 编号（须为整数）")
            continue
        if idset is not None and pid not in idset:
            errs.append(f"{page_label}：panel={pid} 不在本页切出的格号里 "
                        f"{sorted(idset)}")
            continue
        if pid in seen:
            errs.append(f"{page_label}：panel={pid} 重复出现")
            continue
        seen.add(pid)

        # ★ 硬约束 1：不得出现 <Picture i> 等标签
        whole = json.dumps(p, ensure_ascii=False)
        if BANNED_TAG.search(whole):
            errs.append(f"{page_label}：panel={pid} 出现了 <Picture/Video/Audio> "
                        "标签 —— 序号必须由 ⑤ 提示词装配注入，VLM 不得生成")
            continue

        shots = p.get("shots", 1)
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
            warns.append(f"{page_label}：panel={pid} 的 shots={shots!r} 非法，按 1 处理")
            shots = 1
        if shots > MAX_SHOTS_PER_PANEL:
            warns.append(f"{page_label}：panel={pid} 的 shots={shots} 超过上限 "
                         f"{MAX_SHOTS_PER_PANEL}，已截断")
            shots = MAX_SHOTS_PER_PANEL

        # ★ 硬约束 2：角色必须在闭集里（先繁→简 + 别名 + 子串兜底）
        cs = p.get("characters") or []
        if not isinstance(cs, list):
            cs = []
        raw_cs = [str(c).strip() for c in cs if str(c).strip()]
        fixed, bad = [], []
        for c in raw_cs:
            if (not chars) or c in chars:
                fixed.append(c)
                continue
            n = norm_name(c)
            hit = (aliases or {}).get(c) or (aliases or {}).get(n)
            if hit is None and chars:
                hit = next((ok for ok in chars if norm_name(ok) == n), None)
            if hit is None and chars:
                # ★ 子串兜底：模型常写**全名**而闭集是**简称**。
                #   实测一整卷里 7+ 处写成「泽田纲吉」而闭集只有「纲吉」→
                #   主角参考图被整批剔除（静默）。取最长匹配以避免误配。
                cand = [ok for ok in chars
                        if norm_name(ok) and norm_name(ok) in n]
                if cand:
                    hit = max(cand, key=lambda ok: len(norm_name(ok)))
            if hit:
                if hit != c:
                    warns.append(f"{page_label}：panel={pid} 角色 {c!r} 经"
                                 f"繁简/别名归一为 {hit!r}")
                fixed.append(hit)
            else:
                bad.append(c)
        if bad:
            warns.append(f"{page_label}：panel={pid} 出现角色表外的名字 "
                         f"{bad}，已剔除")
        seen_c, cs2 = set(), []
        for c in fixed:
            if c not in seen_c:
                seen_c.add(c)
                cs2.append(c)

        # ★ shots>1 必须分段，否则展开出来的 N 个镜头内容完全一样（白烧渲染时间）。
        #   分段契约：`content` / `action` 用 ` | ` 分 N 段；`dialogue_draft`
        #   数组按顺序切。这里只警告不拒收 —— 内容仍可用，只是不够精确。
        if shots > 1:
            short = []
            for key in ("content", "action"):
                v = str(p.get(key) or "")
                if len([x for x in v.split(" | ") if x.strip()]) < shots:
                    short.append(key)
            if short:
                warns.append(
                    f"{page_label}：panel={pid} shots={shots}，但 "
                    f"{'/'.join(short)} 只有 1 段（应 {shots} 段）—— "
                    "展开后这些镜头会用同一段内容，等于重复。"
                    "分段写法：`左半是… | 右半是…`（建议抽查该格）")

        cam = p.get("camera", "static")
        if cam not in CAMERAS:
            warns.append(f"{page_label}：panel={pid} 的 camera={cam!r} 不在预设表里，"
                         "改成 static")
            cam = "static"

        dh = p.get("duration_hint", 5)
        try:
            dh = int(round(float(dh)))
        except (TypeError, ValueError):
            dh = 5
        dh = max(DURATION_MIN, min(DURATION_MAX, dh))

        try:
            conf = float(p.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0

        out.append({
            "panel": int(pid), "shots": int(shots),
            "content": str(p.get("content") or "").strip(),
            "action": str(p.get("action") or "").strip(),
            "camera": cam, "characters": cs2,
            "dialogue_draft": [str(x).strip()
                               for x in (p.get("dialogue_draft") or [])
                               if str(x).strip()],
            "duration_hint": dh,
            "confidence": conf,
        })

    missing = sorted(set(idset) - seen) if idset else []
    if missing:
        msg = (f"{page_label}：这些格没有被解读：{missing}"
               f"（共 {len(idset)} 格只解读了 {len(seen)} 格）")
        if strict_coverage:
            errs.append(msg + " —— 静默漏读，整页拒收"
                             "（可关掉 strict_coverage 放宽为警告）")
        else:
            warns.append(msg)
    out.sort(key=lambda x: x["panel"])
    return out, errs, warns


# --------------------------------------------------------------------------
# 结果变换（VLM → 分镜计划之间的可调旋钮）
# --------------------------------------------------------------------------
def transform_pages(pages: Sequence[Dict[str, Any]], force_shots: int = 0,
                    duration_scale: float = 1.0,
                    strict_duration: bool = True) -> List[Dict[str, Any]]:
    """对页结果做纯数据变换（不调 API）。用于 ⑭ 的「便宜旋钮」。

    ``force_shots``   ：0 = 保留 VLM 判断；≥1 = **全部强制**成该值。
                        ★ 实测 `shots` 跨次运行不稳定（同页 3/4/5），
                        而它直接决定出片镜数 —— 想省时间就设 1。
    ``duration_scale``：VLM 的时长建议偏长（整卷 89 分钟），可整体缩放。
                        缩放后仍会被 17n+5 对齐。
    """
    fs = int(force_shots or 0)
    ds = float(duration_scale or 1.0)
    out = []
    for pg in pages:
        pg2 = dict(pg)
        panels = []
        for p in (pg.get("panels") or []):
            q = dict(p)
            if fs >= 1:
                q["shots"] = min(fs, MAX_SHOTS_PER_PANEL)
            if abs(ds - 1.0) > 1e-6:
                dh = float(q.get("duration_hint") or 5) * ds
                dh = int(round(dh))
                if strict_duration:
                    dh = max(DURATION_MIN, min(DURATION_MAX, dh))
                else:
                    dh = max(1, dh)
                q["duration_hint"] = dh
            panels.append(q)
        pg2["panels"] = panels
        out.append(pg2)
    return out


NON_NARRATIVE_KINDS = ("cover", "info")


def count_pages(pages: Sequence[Dict[str, Any]],
                include_non_narrative: bool = False) -> Dict[str, int]:
    """统计页/格/镜/非叙事页（报告与省钱预估用）。"""
    n_pages = n_panel = n_shot = n_skip = 0
    for pg in pages:
        kind = str(pg.get("page_kind") or "").strip().lower()
        if not include_non_narrative and kind in NON_NARRATIVE_KINDS:
            n_skip += 1
            continue
        n_pages += 1
        for p in (pg.get("panels") or []):
            n_panel += 1
            n_shot += int(p.get("shots") or 1)
    return {"pages": n_pages, "panels": n_panel, "shots": n_shot,
            "skipped_pages": n_skip}


# --------------------------------------------------------------------------
# 输出：③ 能直接吃的块文本
# --------------------------------------------------------------------------
def to_block_text(pages: Sequence[Dict[str, Any]], episode: str = "",
                  style: str = "manga_anime",
                  include_non_narrative: bool = False,
                  source_note: str = "VLM 页级解读") -> str:
    """把页级结果拼成 ③ 能直接吃的块文本。

    ★ 默认**跳过非叙事页**（封面 / 设定页 / 目录页）。实测一批里这类占 42%
    （某卷 16 页），它们**不该变成镜头**（白烧显存与时间）。几何切格完全看不出
    这一点，而 VLM 免费给了这个信号。

    ★ ``shots > 1`` 的格子前面加一行注释说明 —— 该字段跨次运行不稳定，
      所以「保留判断」的同时提示抽查。
    """
    L: List[str] = []
    if episode:
        L.append(f"# episode: {episode}")
    if style:
        L.append(f"# style: {style}")
    if source_note:
        L.append(f"# 由 {source_note} 生成；dialogue 为 VLM 草稿，请人工核对")
    L.append("# 带 shots: 的格是 VLM 判的「一格多镜」—— 已保留该判断；"
             "该项跨次运行不稳定，建议抽查")
    skipped: List[str] = []
    for pg in pages:
        kind = str(pg.get("page_kind") or "").strip().lower()
        if not include_non_narrative and kind in NON_NARRATIVE_KINDS:
            skipped.append(f"{pg.get('page_name')}({kind})")
            continue
        pname = str(pg.get("page_name") or "page")
        if "." in pname:
            pname = pname.rsplit(".", 1)[0]
        for p in (pg.get("panels") or []):
            L.append(f"=== {pname}_{int(p['panel']):02d} ===")
            L.append(f"panel: {int(p['panel'])}")
            if int(p.get("shots") or 1) > 1:
                L.append(f"# VLM 判为一格 {p['shots']} 镜"
                         f"（把握 {float(p.get('confidence') or 0):.2f}）"
                         "—— 已保留；跨次运行不稳定，建议抽查")
                L.append(f"shots: {int(p['shots'])}")
            L.append(f"duration: {int(p.get('duration_hint') or 5)}")
            L.append(f"camera: {p.get('camera') or 'static'}")
            if p.get("content"):
                L.append(f"content: {p['content']}")
            if p.get("action"):
                L.append(f"action: {p['action']}")
            if p.get("characters"):
                L.append("characters: " + ", ".join(p["characters"]))
            if p.get("dialogue_draft"):
                L.append("dialogue: " + " | ".join(p["dialogue_draft"]))
            L.append("")
    if skipped:
        L.append(f"# 已跳过 {len(skipped)} 个非叙事页（封面/设定/目录）：")
        L.append("#   " + ", ".join(skipped))
        L.append("# 想保留它们请在 ⑭ 打开 include_non_narrative")
    return "\n".join(L)


def to_json_obj(pages: Sequence[Dict[str, Any]], model: str = "",
                episode: str = "", chars: Sequence[str] = (),
                aliases: Optional[Dict[str, str]] = None,
                warnings: Optional[Sequence[str]] = None,
                errors: Optional[Sequence[str]] = None,
                failed_pages: Optional[Sequence[str]] = None,
                usage: Optional[Dict[str, Any]] = None,
                seconds: float = 0.0) -> Dict[str, Any]:
    """结构化结果（供 JSON 输出 / 人工核对 / 下游复用）。"""
    st = count_pages(pages, include_non_narrative=True)
    shot_all = sum(int(p.get("shots") or 1)
                   for pg in pages for p in (pg.get("panels") or []))
    narr = count_pages(pages, include_non_narrative=False)
    return {
        "schema": "h3manga.vlm_result/1",
        "model": model,
        "episode": episode,
        "character_closed_set": sorted(chars),
        "aliases": dict(aliases or {}),
        "stats": {
            "pages_ok": len(pages),
            "pages_failed": len(list(failed_pages or [])),
            "panels": st["panels"],
            "shots": shot_all,
            "pages_non_narrative": narr["skipped_pages"],
            "shots_over_one": sum(1 for pg in pages
                                  for p in (pg.get("panels") or [])
                                  if int(p.get("shots") or 1) > 1),
            "seconds": round(float(seconds), 2),
            **{k: v for k, v in (usage or {}).items()
               if k in ("total_tokens", "prompt_tokens", "completion_tokens")},
        },
        "warnings": list(warnings or []),
        "errors": list(errors or []),
        "failed_pages": list(failed_pages or []),
        "pages": list(pages),
    }


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def vlm_report(pages: Sequence[Dict[str, Any]], model: str = "",
               usage: Optional[Dict[str, Any]] = None, seconds: float = 0.0,
               warnings: Optional[Sequence[str]] = None,
               errors: Optional[Sequence[str]] = None,
               failed_pages: Optional[Sequence[str]] = None,
               include_non_narrative: bool = False,
               max_warn_lines: int = 30) -> str:
    """人读报告（节点输出用）。"""
    narr = count_pages(pages, include_non_narrative=include_non_narrative)
    st = count_pages(pages, include_non_narrative=True)
    L = [f"VLM 页级解读报告：{len(pages)} 页成功"
         f"{f' / {len(failed_pages)} 页失败' if failed_pages else ''}",
         "=" * 56,
         f"  模型        : {model}",
         f"  格 / 镜     : {st['panels']} 格 / {narr['shots']} 镜"
         f"（非叙事页已跳过 {narr['skipped_pages']} 页 / {st['panels'] - narr['panels']} 格）"
         if not include_non_narrative else
         f"  格 / 镜     : {st['panels']} 格 / {narr['shots']} 镜",
         f"  一格多镜的格: {sum(1 for pg in pages for p in (pg.get('panels') or [])
                                if int(p.get('shots') or 1) > 1)}"
         f"  ← 该字段跨次运行不稳定，建议抽查",
         f"  耗时        : {seconds:.1f}s"
         + (f"（{seconds / max(1, len(pages)):.1f}s/页）" if pages else "")]
    if usage:
        L.append(f"  token       : {usage.get('total_tokens', '?')}"
                 f"（prompt {usage.get('prompt_tokens', '?')}"
                 f" + completion {usage.get('completion_tokens', '?')}）")
    kinds: Dict[str, int] = {}
    for pg in pages:
        k = str(pg.get("page_kind") or "(空)").strip() or "(空)"
        kinds[k] = kinds.get(k, 0) + 1
    if kinds:
        L.append("  页型分布    : " + "  ".join(f"{k}×{v}"
                                                for k, v in sorted(kinds.items())))
    if failed_pages:
        L.append("")
        L.append(f"失败页（{len(failed_pages)}）：")
        for x in failed_pages:
            L.append(f"  ! {x}")
    if errors:
        L.append("")
        L.append(f"不合规记录（这些页没有写进清单，共 {len(errors)} 条）：")
        for x in errors[:max_warn_lines]:
            L.append(f"  ! {x}")
        if len(errors) > max_warn_lines:
            L.append(f"  …还有 {len(errors) - max_warn_lines} 条（见 JSON 的 errors）")
    if warnings:
        L.append("")
        L.append(f"警告（清单已自动修正，共 {len(warnings)} 条）：")
        for x in warnings[:max_warn_lines]:
            L.append(f"  - {x}")
        if len(warnings) > max_warn_lines:
            L.append(f"  …还有 {len(warnings) - max_warn_lines} 条（见 JSON 的 warnings）")
    L.append("")
    L.append("提示：shots 字段跨次运行不稳定（同一页可能跑出不同镜数），"
             "已按「保留 VLM 判断」处理；不认同可用 ⑭ 的 force_shots 统一覆盖。")
    return "\n".join(L)
