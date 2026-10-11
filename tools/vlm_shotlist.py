"""VLM 页级解读 → 分镜清单（单卷试点工具）。

设计要点
--------
1. **零第三方依赖** —— 只用标准库 urllib / base64 / json，与插件其余工具一致。
2. **OpenAI 兼容协议** —— 一个 base_url + 一个 model 就能接：
   Qwen-VL（dashscope 兼容模式）、GLM-4V、GPT-4o、Claude 网关、本地 vLLM/Ollama。
3. **中文路径安全** —— 图片用 ``open(path, "rb")`` 二进制读（``cv2.imread`` 在
   非 ASCII 路径下会**静默返回 None**，本项目素材路径必然含中文）。
4. **代理可控** —— 走系统代理常把「服务没起来」伪装成 502，本工具默认直连、
   需要时才开代理。
5. **产出前先校验** —— 三条硬约束（禁 ``<Picture i>`` / 角色闭集 / panel 范围）
   在写文件之前就要过，不合格就整页拒收并记错误，**绝不静默放过**。
6. **只产出「分镜清单」，不碰序号** —— ``<Picture i>`` 一律由 ⑤ 提示词装配
   在运行时注入（架构红线，见 MEMORY.md 硬约定 12）。

用法（cwd = ComfyUI 根目录，用 standalone-env 的 python）
--------------------------------------------------------
    # 0) 看配置对不对、要发多少 token、估算花费（不发请求）
    .../vlm_shotlist.py dry  --pages "D:/剧本/漫改/.../Reborn_30_096.png"

    # 1) 连通性 + 返回格式自检（只跑 1 页）
    set VLM_API_KEY=sk-xxxx
    .../vlm_shotlist.py probe --pages "D:/.../Reborn_30_096.png"

    # 2) 批量跑一整卷 → 分镜清单 + 对照报告
    .../vlm_shotlist.py run --dir "D:/剧本/漫改/.../家庭教师 第30卷" ^
        --out-dir D:/h3vlm --model qwen-vl-max --base-url https://dashscope.aliyuncs.com/compatible-mode/v1

    # 3) 把 VLM 的 shotlist 与 ① 的几何切格结果对照
    .../vlm_shotlist.py compare --shotlist D:/h3vlm/shotlist.json --dir "D:/.../第30卷"

环境变量
--------
    VLM_API_KEY   必填（也可用 --api-key 传，但**不要写进工作流文件**）
    VLM_BASE_URL  可选，默认 OpenAI 官方
    VLM_MODEL     可选，默认 gpt-4o-mini
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)
if PKG not in sys.path:
    sys.path.insert(0, PKG)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# ★ 三条硬约束写进提示词，同时由代码二次校验（不信任模型）
SYSTEM_PROMPT = """你是一名专业的漫画分镜分析师，为「漫画改编动画」项目服务。

我会给你一张漫画页，以及这一页已经切好的分镜格坐标（像素，左上角为原点，
x 向右、y 向下）。请你**逐格**理解画面，输出可直接用于视频生成的结构化数据。

# 输出字段（每个分镜格一条）
- panel         : 整数，必须使用我给你的格子编号
- shots         : 整数，这一格应该拆成几个镜头（见下方判据）
- content       : 一句话描述这一格「画了什么」（画面内容，客观描述，不要加评论）
- action        : 这一格里的动作或镜头运动，动词开头
- camera        : 运镜建议，只能从这个表里选：
                  static / slow_push_in / push_in / pull_out / pan_left / pan_right
                  / tilt_up / tilt_down / close_up / medium_shot / wide_shot
- characters    : 出场角色名数组，**只能从我给你的角色表里选**；表里没有的留空数组。
                  ★ 一律写成**简体中文**，不要写繁体（读图读到的繁体名要转成简体再比对）
- dialogue_draft: 台词草稿数组（你有把握才写；读不清就给空数组，不要编）
- duration_hint : 建议时长（秒，整数 3~10）
- confidence    : 你对这一格理解的把握，0~1

# shots 的判据（最重要）
- 一格内如果**只有单一视点、单一动作** → shots = 1
- 一格内如果有**两个以上互不相连的视点**（例如左半是人物特写、右半是另一场景），
  或者同一个格子里画了**连续动作的几个时间相** → shots 按视点/时间相数量给
- 判断依据是**画面结构**，不是台词数量。同一场景里多人对话仍然算 1 镜。

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

这一页已经切好的分镜格（共 {n} 格，按阅读顺序）：
{boxes}

可用角色表（characters 只能从中选）：{chars}
"""


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def err(msg):
    print(f"  [ERROR] {msg}", flush=True)


def warn(msg):
    print(f"  [WARN ] {msg}", flush=True)


def read_image_b64(path: str, max_edge: int = 1600):
    """返回 (base64字符串, mime, 宽, 高)。

    ★ 用二进制读 + cv2 解码，**避开 cv2.imread 在中文路径下静默返回 None 的坑**。
    """
    with open(path, "rb") as f:            # 二进制读不受中文路径影响
        raw = f.read()
    mime = "image/png"
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    if max_edge:
        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode 返回 None")
            h, w = img.shape[:2]
            if max(h, w) > max_edge:
                s = max_edge / float(max(h, w))
                img = cv2.resize(img, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise ValueError("cv2.imencode 失败")
                raw = buf.tobytes()
                mime = "image/jpeg"
            h, w = img.shape[:2]
            return base64.b64encode(raw).decode("ascii"), mime, w, h
        except Exception as e:  # noqa: BLE001
            warn(f"缩放失败（{type(e).__name__}: {e}），按原图发送")

    return base64.b64encode(raw).decode("ascii"), mime, 0, 0


def make_opener(use_proxy: bool):
    if use_proxy:
        return urllib.request.build_opener()
    # ★ 空 ProxyHandler = 明确不走代理。之前的实测里，系统代理会把
    #   「本机服务没起来」伪装成 502 Tunnel connection failed。
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call_vlm(opener, base_url, api_key, model, sys_prompt, user_text,
             b64, mime, timeout=180, retries=3, temperature=0.2):
    """调一次 OpenAI 兼容的 chat/completions，返回 (文本, 原始响应dict)。

    ★ 内容审核拒绝（`data_inspection_failed`）**不重试**：同一张图必然再被拦，
      重试只是白等。实测漫画战斗页会被阿里云判为 inappropriate content。
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
        "temperature": temperature,
        "max_tokens": 4096,
    }
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(1, retries + 1):
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
            if "data_inspection_failed" in detail or "InappropriateContent" in detail:
                raise RuntimeError(
                    "内容审核拒绝（云服务端判定图片含不当内容）—— "
                    "重试无用，需人工处理或换模型/换区域。原始响应：" + detail[:200]
                ) from e
            if e.code in (400, 401, 403, 404, 422):
                # 客户端错误（模型名错 / key 无效 / 参数错）→ 重试没意义
                raise RuntimeError(f"不可重试的请求错误 —— {last}") from e
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            wait = 2 ** attempt
            warn(f"第 {attempt} 次失败（{last}），{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"调用失败（{retries} 次）：{last}")


def extract_text(resp: dict) -> str:
    try:
        ch = resp["choices"][0]
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
        raise ValueError("回复为空")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("回复里找不到 JSON 对象")
    return json.loads(text[i:j + 1])


# --------------------------------------------------------------------------
# 校验（三条硬约束）
# --------------------------------------------------------------------------
BANNED_TAG = re.compile(r"<\s*(Picture|Video|Audio)\b", re.I)
CAMERAS = {"static", "slow_push_in", "push_in", "pull_out", "pan_left",
           "pan_right", "tilt_up", "tilt_down", "close_up", "medium_shot",
           "wide_shot"}
MAX_SHOTS_PER_PANEL = 8

# ★ 内容审核回退阶梯：(模型或 None=沿用主模型, 长边倍率)
#
# 实测（Reborn_30_020 战斗页）：主模型在 edge=1600/900 都被拦，但
# **换 qwen3-vl-plus 保持 1600 就过**，而主模型缩到 600 也过。
# 说明审核判定与模型、分辨率都相关，且非确定 —— 所以给一条阶梯，
# 而不是把被拦的页丢掉。
INSPECTION_FALLBACKS = [(None, 0.4), ("qwen3-vl-plus", 1.0)]

# ★ 繁体 → 简体 归一化表。
#
# 为什么必须有：实测 `qwen3.5-omni-flash` 会把角色名写成繁体
#（`綱吉 / 獄寺 / 藍波 / 優尼`），而角色闭集是简体 → **名字全被剔除**，
# 参考图槽位随之丢失，而且只在 warning 里出现、不报错。
# 无第三方库（zhconv/opencc 在本机都没有）→ 内置一张小表兜底，
# 同时提示词里强制简体、并允许用 `--char-alias` 由用户补充。
TRAD2SIMP = str.maketrans({
    "綱": "纲", "獄": "狱", "藍": "蓝", "優": "优", "雲": "云", "雀": "雀",
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
    "爾": "尔", "納": "纳", "諾": "诺", "蘭": "兰", "迪": "迪", "骷": "骷",
    "髏": "髅", "涅": "涅", "留": "留", "史": "史", "帕": "帕", "正": "正",
    "江": "江",
    "奈": "奈", "太": "太", "浦": "浦", "春": "春", "京": "京", "碧": "碧",
    "洋": "洋", "琪": "琪", "山": "山", "本": "本", "武": "武", "了": "了",
    "平": "平", "白": "白", "十": "十", "年": "年", "後": "后", "模": "模",
    "樣": "样", "轉": "转", "集": "集", "師": "师", "活": "活", "田": "田",
})


def norm_name(s: str) -> str:
    """繁→简 + 去空白。用于角色闭集匹配（不用于展示）。"""
    return str(s).strip().translate(TRAD2SIMP)


def parse_aliases(text: str):
    """`--char-alias "綱吉=纲吉,藍波=蓝波"` → {繁体写法: 闭集里的名字}。"""
    out = {}
    for part in (text or "").split(","):
        if "=" in part:
            a, b = part.split("=", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                out[a] = b
    return out


def validate_page(obj: dict, panel_ids, chars, page_label: str, aliases=None,
                  strict_coverage: bool = True):
    """返回 (干净的panel列表, 错误列表, 警告列表)。不合规就整页拒收。

    参数
    ----
    chars          : 角色闭集（简体）
    aliases        : 额外别名映射 {写法: 闭集名}
    strict_coverage: True 时「有格没被解读」升级为**错误**（整页拒收）。
                     ★ 实测 `qwen3.5-omni-flash` 在 018 页只解读了 8 格里的 4 格
                     就把结果交出来（并编出 10 镜）→ 这种「静默漏读」必须拦住。
    """
    errs, warns = [], []
    panels = obj.get("panels")
    if not isinstance(panels, list) or not panels:
        return [], [f"{page_label}: 没有 panels 数组"], []

    out = []
    seen = set()
    for k, p in enumerate(panels):
        if not isinstance(p, dict):
            errs.append(f"{page_label}: panels[{k}] 不是对象")
            continue
        pid = p.get("panel")
        if not isinstance(pid, int):
            errs.append(f"{page_label}: panels[{k}] 缺 panel 编号")
            continue
        if panel_ids and pid not in panel_ids:
            errs.append(f"{page_label}: panel={pid} 不在本页切出的格号里 "
                        f"{sorted(panel_ids)}")
            continue
        if pid in seen:
            errs.append(f"{page_label}: panel={pid} 重复出现")
            continue
        seen.add(pid)

        # ★ 硬约束 1：不得出现 <Picture i> 等标签
        whole = json.dumps(p, ensure_ascii=False)
        if BANNED_TAG.search(whole):
            errs.append(f"{page_label}: panel={pid} 出现了 <Picture/Video/Audio> "
                        f"标签 —— 序号必须由 ⑤ 提示词装配注入，VLM 不得生成")
            continue

        shots = p.get("shots", 1)
        if not isinstance(shots, int) or shots < 1:
            warns.append(f"{page_label}: panel={pid} 的 shots={shots!r} 非法，按 1 处理")
            shots = 1
        if shots > MAX_SHOTS_PER_PANEL:
            warns.append(f"{page_label}: panel={pid} 的 shots={shots} 超过上限 "
                         f"{MAX_SHOTS_PER_PANEL}，已截断")
            shots = MAX_SHOTS_PER_PANEL

        # ★ 硬约束 2：角色必须在闭集里
        #   先繁→简归一化 + 查别名表，再做闭集匹配 —— 否则模型写繁体名会被整批剔除。
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
                # 归一化后再查一次（繁→简）
                for ok_name in chars:
                    if norm_name(ok_name) == n:
                        hit = ok_name
                        break
            if hit is None and chars:
                # ★ 子串兜底：模型常写**全名**而闭集是**简称**
                #   实测第30卷有 7+ 处写成「泽田纲吉」而闭集只有「纲吉」→
                #   主角参考图被整批剔除（静默）。取最长的匹配以避免误配。
                #   `n` 已归一化；闭集侧也归一化后比对。
                cand = [ok for ok in chars
                        if norm_name(ok) and norm_name(ok) in n]
                if cand:
                    hit = max(cand, key=lambda ok: len(norm_name(ok)))
            if hit:
                if hit != c:
                    warns.append(f"{page_label}: panel={pid} 角色 {c!r} 经繁简/别名"
                                 f"归一为 {hit!r}")
                fixed.append(hit)
            else:
                bad.append(c)
        if bad:
            warns.append(f"{page_label}: panel={pid} 出现角色表外的名字 "
                         f"{bad}，已剔除")
        # 去重保序
        seen_c, cs = set(), []
        for c in fixed:
            if c not in seen_c:
                seen_c.add(c)
                cs.append(c)

        cam = p.get("camera", "static")
        if cam not in CAMERAS:
            warns.append(f"{page_label}: panel={pid} 的 camera={cam!r} 不在预设表里，"
                         f"改成 static")
            cam = "static"

        dh = p.get("duration_hint", 5)
        try:
            dh = int(round(float(dh)))
        except (TypeError, ValueError):
            dh = 5
        dh = max(3, min(10, dh))

        out.append({
            "panel": pid, "shots": shots,
            "content": str(p.get("content") or "").strip(),
            "action": str(p.get("action") or "").strip(),
            "camera": cam, "characters": cs,
            "dialogue_draft": [str(x).strip() for x in (p.get("dialogue_draft") or [])
                               if str(x).strip()],
            "duration_hint": dh,
            "confidence": float(p.get("confidence") or 0) or 0.0,
        })

    missing = sorted(set(panel_ids) - seen)
    if missing:
        msg = (f"{page_label}: 这些格没有被解读：{missing}"
               f"（共 {len(panel_ids)} 格只解读了 {len(seen)} 格）")
        if strict_coverage:
            errs.append(msg + " —— 静默漏读，整页拒收（可用 --allow-partial 放宽）")
        else:
            warns.append(msg)
    out.sort(key=lambda x: x["panel"])
    return out, errs, warns


# --------------------------------------------------------------------------
# 切格（复用插件自己的实现，保证与 ① 一致）
# --------------------------------------------------------------------------
def split_page(path: str, frame="component", deoverlap=True):
    import numpy as np
    import cv2
    from core import panel_split as ps
    with open(path, "rb") as f:
        raw = f.read()
    g = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"读不到图：{path}")
    boxes, warns = ps.split_page_boxes(
        g, mode="auto", reading_order="rtl", frame=frame, deoverlap=deoverlap)
    return boxes, warns, (g.shape[1], g.shape[0])


def panel_offsets(paths):
    """★ 算出每一页的「全局格号偏移」。

    为什么必须做：① `H3MangaPanelSplitter` 的 `Panel.index` 是**整批全局递增**的
    （`idx = 0` 写在页循环之外，跨页累加），而 VLM 只按**本页**编号 0..k-1。
    ④ `bind_board(mode="by_panel_index")` 用的是 `panel_index + offset` 去索引
    `board.panels` —— 所以第 2 页起若直接照抄本页编号，就会**绑到错误的格图**
    （或超范围 → 静默退化成纯文生视频）。两类都是不报错的错。

    切格是确定性的，所以这里能把偏移算准（前提：与 ① 用同一套切格参数）。
    """
    off, acc = {}, 0
    for p in paths:
        off[p] = acc
        try:
            acc += len(split_page(p)[0])
        except Exception:  # noqa: BLE001
            pass
    return off


def boxes_text(boxes):
    return "\n".join(
        f"  格 {i}: x={x}, y={y}, w={w}, h={h}"
        for i, (x, y, w, h) in enumerate(boxes))


# --------------------------------------------------------------------------
# 输出：③ 能直接吃的块文本
# --------------------------------------------------------------------------
def to_block_text(pages, episode="", style="manga_anime",
                  include_non_narrative=False):
    """把页级结果拼成 ③ 能直接吃的块文本。

    ★ 默认**跳过非叙事页**（封面 / 设定页 / 目录页）。理由：实测这批页里
    `page_kind=cover/info` 的占 42%，它们**不该变成镜头**（会白烧显存与时间）。
    几何切格完全看不出这一点，而 VLM 免费给了这个信号。

    ★ `shots > 1` 的格子前面加一行注释提醒 —— 实测跨模型/跨次运行不稳定
    （同一页 3 次跑出 3/4/5 镜），是**唯一需要人工确认**的字段。
    """
    L = []
    if episode:
        L.append(f"# episode: {episode}")
    if style:
        L.append(f"# style: {style}")
    L.append("# 由 tools/vlm_shotlist.py 生成；dialogue 为 VLM 草稿，请人工核对")
    L.append("# ★ 带 `shots:` 的格是 VLM 判的「一格多镜」—— 已保留该判断；"
             "该项跨次运行不稳定，建议抽查")
    skipped = []
    for pg in pages:
        kind = (pg.get("page_kind") or "").strip().lower()
        if not include_non_narrative and kind in ("cover", "info"):
            skipped.append(f"{pg.get('page_name')}({kind})")
            continue
        pname = os.path.splitext(pg.get("page_name", ""))[0]
        for p in pg["panels"]:
            L.append(f"=== {pname}_{p['panel']:02d} ===")
            L.append(f"panel: {p['panel']}")
            if p["shots"] > 1:
                L.append(f"# VLM 判为一格 {p['shots']} 镜"
                         f"（把握 {p.get('confidence', 0):.2f}）"
                         f"—— 已保留；跨次运行不稳定，建议抽查")
                L.append(f"shots: {p['shots']}")
            L.append(f"duration: {p['duration_hint']}")
            L.append(f"camera: {p['camera']}")
            if p.get("content"):
                L.append(f"content: {p['content']}")
            if p.get("action"):
                L.append(f"action: {p['action']}")
            if p.get("characters"):
                L.append(f"characters: {', '.join(p['characters'])}")
            if p.get("dialogue_draft"):
                L.append(f"dialogue: {' | '.join(p['dialogue_draft'])}")
            L.append("")
    if skipped:
        L.append(f"# 已跳过 {len(skipped)} 个非叙事页（封面/设定/目录）：")
        L.append("#   " + ", ".join(skipped))
        L.append("# 想保留它们请加 --include-non-narrative")
    return "\n".join(L)


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------
def _natural_key(name: str):
    """自然排序键：把文件名里的数字段当**数字**比。

    ★ 必须这么做 —— 直接 sorted() 会得到 001, 010, 011, 002, ... 的顺序，
    阅读顺序一乱，分镜清单里的 panel 号就跟着全错，而且**不会报错**。
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_pages(dirpath):
    fs = [f for f in os.listdir(dirpath) if f.lower().endswith(IMAGE_EXT)]
    fs.sort(key=_natural_key)
    return [os.path.join(dirpath, f) for f in fs]


def is_spread(W: int, H: int) -> bool:
    """宽高比过大 → 多半是跨页扫描（双页拼成一张）。"""
    return H > 0 and (W / float(H)) > 2.2


def cmd_dry(a):
    paths = [a.pages] if a.pages else list_pages(a.dir)
    if not paths:
        err("没有找到图片")
        return 1
    n = min(len(paths), a.limit or len(paths))
    log(f"将要处理 {n} / {len(paths)} 页（dry-run，不发请求）")
    log("")
    log(f"  {'#':>3} {'文件':<26}{'原图':>12}{'发送':>12}{'格数':>5}{'~图token':>9}  备注")
    log("  " + "-" * 84)
    tot_tok, flagged = 0, 0
    for i, p in enumerate(paths[:n], 1):
        name = os.path.basename(p)
        b64, mime, w, h = read_image_b64(p, a.max_edge)
        boxes, warns, (W, H) = split_page(p)
        # ★ image token 按**分辨率**估，不是按文件体积：
        #   OpenAI 量级 ≈ (w*h)/750。用体积换算会得出「单页 13 万 token」这种荒谬值。
        itok = int((w or W) * (h or H) / 750)
        tot_tok += itok
        note = []
        if is_spread(W, H):
            note.append(f"疑似跨页 {W / float(H):.1f}:1")
        if len(boxes) <= 1:
            note.append("★只切出1格")
        if any("过大连通块" in x for x in warns):
            note.append("★几何失效")
        if note:
            flagged += 1
        orig = f"{W}x{H}"
        sent = f"{w or W}x{h or H}"
        log(f"  {i:>3} {name[:26]:<26}{orig:>12}{sent:>12}{len(boxes):>5}{itok:>9}  "
            + " ".join(note))
        for x in warns:
            if "重叠" in x or "开运算" in x:
                continue
            warn(f"      {name}: {x[:110]}")
    log("  " + "-" * 84)
    log("")
    per_page = (tot_tok + n * 1200) / max(1, n)
    log("成本量级估算：")
    log(f"  图片部分：平均约 {tot_tok // max(1, n)} tokens/页，小计 ~{tot_tok / 1000:.0f}k")
    log(f"  文本部分：提示词 + 输出约 1200 tokens/页，小计 ~{n * 1200 / 1000:.0f}k")
    log(f"  → {n} 页合计约 **{per_page * n / 1000:.0f}k tokens**")
    log(f"  → 整卷外推（{len(paths)} 页）约 **{per_page * len(paths) / 1000:.0f}k tokens**")
    log("  ★ 单价按你选的模型算；这里只判断量级。")
    log("")
    if flagged:
        log(f"  ★ {flagged} 页出现「几何失效 / 只切出 1 格 / 疑似跨页」——")
        log("    这几页恰恰是 VLM 最能补位的地方（它看得出「这里其实有 2 格」），")
        log("    但最终边界仍建议人工确认（`manual_boxes` 支持多边形写法）。")
    return 0


def cmd_probe(a):
    paths = [a.pages] if a.pages else list_pages(a.dir)
    if not paths:
        err("没有找到图片")
        return 1
    p = paths[0]
    boxes, _w, (W, H) = split_page(p)
    b64, mime, w, h = read_image_b64(p, a.max_edge)
    user = USER_TMPL.format(name=os.path.basename(p), W=w or W, H=h or H,
                            n=len(boxes), boxes=boxes_text(boxes),
                            chars=", ".join(a.chars.split(",")) if a.chars else "(无)")
    log(f"probe: {os.path.basename(p)}  切出 {len(boxes)} 格")
    log(f"  POST {a.base_url}/chat/completions  model={a.model}")
    opener = make_opener(a.use_proxy)
    t0 = time.time()
    try:
        text, raw = call_vlm(opener, a.base_url, a.api_key, a.model,
                             SYSTEM_PROMPT, user, b64, mime, a.timeout)
    except Exception as e:  # noqa: BLE001
        err(f"调用失败：{e}")
        return 1
    log(f"  OK  {time.time() - t0:.1f}s，回复 {len(text)} 字符")
    try:
        obj = extract_json(text)
    except Exception as e:  # noqa: BLE001
        err(f"JSON 解析失败：{e}")
        log("  原始回复前 800 字：")
        log(text[:800])
        return 1
    panels, errs, warns = validate_page(obj, list(range(len(boxes))),
                                        set(c.strip() for c in a.chars.split(",") if c.strip()),
                                        "probe",
                                        aliases=parse_aliases(getattr(a, "char_alias", "")))
    for x in warns:
        warn(x)
    for x in errs:
        err(x)
    log(f"  校验：{len(panels)} 格通过 / {len(errs)} 个错误"
        f"   page_kind={obj.get('page_kind')}")
    for p_ in panels:
        log(f"    格 {p_['panel']}: shots={p_['shots']} camera={p_['camera']} "
            f"角色={','.join(p_['characters']) or '-'} 时长={p_['duration_hint']}s "
            f"把握={p_['confidence']:.2f}")
        log(f"      内容: {p_['content'][:70]}")
    return 0 if not errs else 1


def cmd_run(a):
    all_paths = [a.pages] if a.pages else list_pages(a.dir)
    if not all_paths:
        err("没有找到图片")
        return 1
    # ★ 全局格号偏移**必须按「全量」算**（不能按 --limit 截断后的列表），
    #   否则某页的偏移会偏小 → 绑错格图。
    offs = panel_offsets(all_paths)
    extra = int(getattr(a, "panel_offset", 0))
    local_mode = bool(getattr(a, "local_panel_index", False))
    paths = all_paths[:a.limit] if a.limit else all_paths
    if not paths:
        err("没有找到图片")
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    opener = make_opener(a.use_proxy)
    chars = set(c.strip() for c in a.chars.split(",") if c.strip())
    aliases = parse_aliases(getattr(a, "char_alias", ""))

    pages, all_errs, all_warns, failed = [], [], [], []
    tok_total = tok_img = 0
    nonnarr = 0
    review = []
    t_start = time.time()
    log(f"共 {len(paths)} 页 → {a.out_dir}")
    log(f"  模型 {a.model}  角色闭集 {len(chars)} 个"
        + (f"  别名 {len(aliases)} 条" if aliases else "")
        + ("" if not getattr(a, "allow_partial", False)
           else "  ⚠ --allow-partial：允许漏读整格"))
    if local_mode:
        warn("--local-panel-index：panel 用**本页**编号。"
             "若 ① 吃的是多页批次，格号会错位（④ 的 panel_index 是全局索引）")
    else:
        log(f"  格号换算：全局偏移（① 的 Panel.index 跨页累加）"
            f"{'，额外 +' + str(extra) if extra else ''}；"
            f"全量 {len(all_paths)} 页共 {offs.get(all_paths[-1], 0) + len(split_page(all_paths[-1])[0])} 格")
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)
        tag = f"[{i}/{len(paths)}] {name}"
        try:
            boxes, _w, (W, H) = split_page(p)
        except Exception as e:  # noqa: BLE001
            err(f"{tag} 切格失败：{e}")
            failed.append(name)
            continue

        # ── 调用（含内容审核回退阶梯）────────────────────────────────
        attempts = [(a.model, 1.0)]
        if not getattr(a, "no_inspection_fallback", False):
            attempts += INSPECTION_FALLBACKS
        obj = _raw = None
        last_err = None
        for att_i, (mdl, fac) in enumerate(attempts):
            use_model = mdl or a.model
            edge = a.max_edge if fac == 1.0 else int(a.max_edge * fac)
            b64, mime, w, h = read_image_b64(p, edge)
            user = USER_TMPL.format(
                name=name, W=w or W, H=h or H, n=len(boxes),
                boxes=boxes_text(boxes),
                chars=", ".join(sorted(chars)) if chars else "(无)")
            try:
                t, _raw = call_vlm(opener, a.base_url, a.api_key, use_model,
                                   SYSTEM_PROMPT, user, b64, mime, a.timeout)
                obj = extract_json(t)
                if att_i:
                    log(f"{tag}   ↳ 审核回退成功：model={use_model} edge={edge}")
                    all_warns.append(
                        f"{name}: 被内容审核拦下，已用 model={use_model} "
                        f"edge={edge} 重跑通过")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if "内容审核" in str(e) and att_i + 1 < len(attempts):
                    nxt = attempts[att_i + 1]
                    warn(f"{tag} 被内容审核拦下 → 换 "
                         f"{nxt[0] or a.model}(edge×{nxt[1]}) 重试")
                    continue
                break
        if obj is None:
            err(f"{tag} 调用/解析失败：{last_err}")
            failed.append(name)
            continue

        u = _raw.get("usage") or {}
        tok_total += int(u.get("total_tokens") or 0)
        tok_img += int((u.get("prompt_tokens_details") or {}).get("image_tokens") or 0)

        panels, errs, warns = validate_page(
            obj, list(range(len(boxes))), chars, name,
            aliases=aliases, strict_coverage=not a.allow_partial)
        for x in warns:
            all_warns.append(x)
        if errs:
            for x in errs:
                all_errs.append(x)
            err(f"{tag} 有 {len(errs)} 处不合规 → 整页拒收")
            failed.append(name)
            continue

        # ★ 把「本页编号」换算成「全卷全局编号」（见 panel_offsets 的说明）。
        #   panel_local 保留原值，便于回溯与 compare。
        base = 0 if local_mode else offs.get(p, 0) + extra
        for p_ in panels:
            p_["panel_local"] = p_["panel"]
            p_["panel"] = p_["panel_local"] + base
        pages.append({"page": len(pages), "page_name": name,
                      "page_size": [W, H], "page_kind": obj.get("page_kind", ""),
                      "panel_offset": base,
                      "panels": panels, "summary": obj.get("summary", "")})
        shots = sum(x["shots"] for x in panels)
        kind = (obj.get("page_kind") or "").lower()
        if kind in ("cover", "info"):
            nonnarr += 1
        for p_ in panels:
            if p_["shots"] > 1:
                review.append((name, p_["panel"], p_["shots"],
                               p_.get("confidence", 0),
                               (p_.get("content") or "")[:34]))
        log(f"{tag}  OK  {len(panels)} 格 / {shots} 镜"
            + (f"  [{kind}]" if kind else ""))

        if a.sleep:
            time.sleep(a.sleep)

    elapsed = time.time() - t_start
    keep_non_narr = bool(getattr(a, "include_non_narrative", False))
    n_shot_total = sum(sum(x["shots"] for x in pg["panels"]) for pg in pages)
    n_panel_total = sum(len(pg["panels"]) for pg in pages)
    stats = {
        "pages_ok": len(pages), "pages_failed": len(failed),
        "panels": n_panel_total, "shots": n_shot_total,
        "pages_non_narrative": nonnarr, "non_narrative_kept": keep_non_narr,
        "panels_pending_review": len(review),
        "tokens_total": tok_total, "tokens_image": tok_img,
        "seconds": round(elapsed, 1),
    }
    shotlist = {"schema": "h3manga.vlm_shotlist/1",
                "model": a.model,
                "base_url": a.base_url,
                "character_closed_set": sorted(chars),
                "aliases": aliases,
                "stats": stats,
                "failed_pages": list(failed),
                # ★ 全量警告入 JSON（原来只往控制台打前 10 条 → 事后无法分析）
                "warnings": list(all_warns),
                "errors": list(all_errs),
                "review_required": [
                    {"page": n, "panel": p, "shots": s, "confidence": c,
                     "content": t} for n, p, s, c, t in review],
                "pages": pages}
    jp = os.path.join(a.out_dir, "shotlist.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(shotlist, f, ensure_ascii=False, indent=1)
    tp = os.path.join(a.out_dir, "shotlist.txt")
    with open(tp, "w", encoding="utf-8") as f:
        f.write(to_block_text(pages,
                              episode=a.episode or os.path.basename(a.dir or ""),
                              include_non_narrative=keep_non_narr))
    rp = os.path.join(a.out_dir, "review.txt")
    with open(rp, "w", encoding="utf-8") as f:
        if review:
            f.write("以下格子被 VLM 判为「一格多镜」（shots>1）"
                    "—— 已按项目决定**保留该判断**，会原样写进 shotlist.txt。\n"
                    "★ 为什么仍列出来：实测这一项跨模型、跨次运行**不稳定**\n"
                    "  （同一页可跑出 3/4/5 镜），所以是**建议抽查**，不是必须逐格确认。\n"
                    "  其余字段（格数 / 内容 / 角色 / 台词 / 页型）跨模型一致，可直接用。\n"
                    "  若不认同某格的拆分 → 把该条的 `shots: N` 删掉（等于 1 镜）即可。\n\n")
            f.write(f"{'页':<24}{'格':>4}{'镜数':>6}{'把握':>7}  内容\n")
            f.write("-" * 92 + "\n")
            for n, p, s, c, t in review:
                f.write(f"{n:<24}{p:>4}{s:>6}{c:>7.2f}  {t}\n")
        else:
            f.write("没有 shots>1 的格子 —— 全部是「一格一镜」，无需人工确认。\n")

    log("")
    log("=" * 72)
    log(f"完成：{len(pages)} 页成功 / {len(failed)} 页失败   用时 {elapsed:.0f}s")
    log(f"  合计 {n_panel_total} 格 → {n_shot_total} 镜")
    if nonnarr:
        log(f"  其中非叙事页 {nonnarr} 页（封面/设定/目录）"
            + ("—— 已保留" if keep_non_narr
               else "—— **已从清单里跳过**（加 --include-non-narrative 可保留）"))
    log(f"  token 用量：{tok_total:,}（其中图片 {tok_img:,}）"
        f"，平均 {tok_total // max(1, len(pages)):,}/页")
    log(f"  分镜清单(JSON) : {jp}")
    log(f"  分镜清单(块文本): {tp}   ← 直接贴进 ③ 的 inline_text")
    log(f"  待人工确认清单 : {rp}"
        + (f"   ★ {len(review)} 格是 shots>1" if review else ""))
    if failed:
        log(f"  失败页：{failed}")
    if all_errs:
        log("  不合规记录（这些页没有写进清单）：")
        for x in all_errs:
            log(f"    ! {x}")
    if all_warns:
        log(f"  警告 {len(all_warns)} 条（清单已自动修正），前 10 条：")
        for x in all_warns[:10]:
            log(f"    - {x}")
    log("=" * 72)
    log("")
    log("下一步：")
    log("  1) 只看 review.txt 里那几格（shots>1 是唯一不可靠的字段）")
    log("  2) 把 shotlist.txt 贴进 ③ 分镜清单的 inline_text，source=inline")
    log("  3) 确认 ① 的切格数 == 清单里的 panel 数")
    return 0 if pages else 1


def cmd_retxt(a):
    """把已有 shotlist.json 的 `panel` 换算成**全卷全局编号**并重写 txt/review。

    **不调用任何 API** —— 因为编号换算纯属本地计算。用途：
    早期版本按「本页编号」写出的 JSON，可以用它免费修正，不必重跑（重跑要花钱）。
    """
    jp = a.shotlist
    if not os.path.isfile(jp):
        err(f"找不到 {jp}")
        return 1
    data = json.load(open(jp, encoding="utf-8"))
    pages = data.get("pages") or []
    if not pages:
        err("shotlist.json 里没有 pages")
        return 1

    base_of = {}
    if a.already_global:
        for pg in pages:
            base_of[pg["page"]] = 0
    elif a.dir:
        paths = list_pages(a.dir)
        offs = panel_offsets(paths)
        by_name = {os.path.basename(x): x for x in paths}
        miss = 0
        for pg in pages:
            pth = by_name.get(pg.get("page_name"))
            if pth is None:
                miss += 1
            base_of[pg["page"]] = offs.get(pth, 0) if pth else 0
        if miss:
            warn(f"{miss} 页在 --dir 里找不到同名文件，其偏移按 0 处理")
    else:
        acc = 0
        for pg in pages:
            base_of[pg["page"]] = acc
            acc += len(pg.get("panels") or [])
        log("按 JSON 内页序累加偏移（假设 JSON 从第 1 页起连续）——"
            "若只跑了部分页，请加 --dir 让偏移更准")

    extra = int(getattr(a, "panel_offset", 0))
    changed = 0
    review = []
    for pg in pages:
        b = base_of.get(pg["page"], 0) + extra
        pg["panel_offset"] = b
        for p in pg.get("panels") or []:
            loc = p.get("panel_local", p["panel"])
            p["panel_local"] = loc
            if p["panel"] != loc + b:
                changed += 1
            p["panel"] = loc + b
            if p["shots"] > 1:
                review.append((pg["page_name"], p["panel"], p["shots"],
                               p.get("confidence", 0),
                               (p.get("content") or "")[:34]))

    out_dir = a.out_dir or os.path.dirname(os.path.abspath(jp))
    os.makedirs(out_dir, exist_ok=True)
    jout = os.path.join(out_dir, "shotlist.json")
    with open(jout, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    keep = bool(getattr(a, "include_non_narrative", False))
    tout = os.path.join(out_dir, "shotlist.txt")
    with open(tout, "w", encoding="utf-8") as f:
        f.write(to_block_text(
            pages, episode=a.episode or os.path.basename(a.dir or ""),
            include_non_narrative=keep))
    rout = os.path.join(out_dir, "review.txt")
    with open(rout, "w", encoding="utf-8") as f:
        if review:
            f.write("以下格子被 VLM 判为「一格多镜」（shots>1）"
                    "—— 已保留该判断，会原样写进 shotlist.txt。\n"
                    "★ 这一项跨模型/跨次运行不稳定，所以是**建议抽查**。\n"
                    "  不认同某格的拆分 → 把该条的 `shots: N` 删掉即可。\n\n")
            f.write(f"{'页':<24}{'全局格号':>8}{'镜数':>6}{'把握':>7}  内容\n")
            f.write("-" * 92 + "\n")
            for n_, p_, s_, c_, t_ in review:
                f.write(f"{n_:<24}{p_:>8}{s_:>6}{c_:>7.2f}  {t_}\n")
        else:
            f.write("没有 shots>1 的格子。\n")

    total = sum(len(pg.get("panels") or []) for pg in pages)
    log("=" * 72)
    log(f"retxt 完成（未调用 API）")
    log(f"  页 {len(pages)}  格 {total}  重编号 {changed} 处")
    log(f"  格号范围 {min((p['panel'] for pg in pages for p in pg.get('panels') or []), default=-1)}"
        f" ~ {max((p['panel'] for pg in pages for p in pg.get('panels') or []), default=-1)}")
    log(f"  {jout}")
    log(f"  {tout}")
    log(f"  {rout}")
    log("=" * 72)
    return 0


def cmd_compare(a):
    data = json.load(open(a.shotlist, encoding="utf-8"))
    paths = [a.pages] if a.pages else list_pages(a.dir)
    by_name = {os.path.basename(p): p for p in paths}
    log(f"{'页面':<26}{'几何格数':>8}{'VLM镜数':>9}{'VLM格数':>9}   结论")
    log("-" * 78)
    tot_g = tot_v = 0
    for pg in data.get("pages", []):
        name = pg["page_name"]
        p = by_name.get(name)
        if not p:
            log(f"{name:<26}{'—':>8}{sum(x['shots'] for x in pg['panels']):>9}"
                f"{len(pg['panels']):>9}   （找不到原图，跳过几何对照）")
            continue
        boxes, _w, _s = split_page(p)
        g, vshot, vpanel = len(boxes), sum(x["shots"] for x in pg["panels"]), len(pg["panels"])
        tot_g += g
        tot_v += vshot
        if vpanel != g:
            note = f"★ 格数不一致（VLM {vpanel} vs 几何 {g}）"
        elif vshot != g:
            note = f"格数一致，但 VLM 拆出 {vshot} 镜（多 {vshot - g}）"
        else:
            note = "完全一致"
        log(f"{name:<26}{g:>8}{vshot:>9}{vpanel:>9}   {note}")
    log("-" * 78)
    log(f"几何格数合计 {tot_g} / VLM 镜数合计 {tot_v}（差 {tot_v - tot_g}）")
    log("")
    log("★ 差异集中在哪：「格数不一致」说明几何切错；「格数一致但多镜」")
    log("  说明几何没错、只是看不到格内的时间相 —— 后者正是 VLM 的价值。")
    return 0


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="VLM 页级解读 → 分镜清单（单卷试点）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pages", default="", help="单页路径（与 --dir 二选一）")
        p.add_argument("--dir", default="", help="整卷目录")
        p.add_argument("--chars", default="", help="角色闭集，逗号分隔")
        p.add_argument("--char-alias", default="",
                       help="角色别名/繁简映射，如 \"綱吉=纲吉,藍波=蓝波\"")
        p.add_argument("--limit", type=int, default=0, help="只处理前 N 页")
        p.add_argument("--max-edge", type=int, default=1600,
                       help="发送前把长边缩到此值（0=不缩，默认 1600）")
        p.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
        p.add_argument("--base-url",
                       default=os.environ.get("VLM_BASE_URL",
                                              "https://api.openai.com/v1"))
        p.add_argument("--model", default=os.environ.get("VLM_MODEL",
                                                         "qwen3.5-omni-plus"))
        p.add_argument("--timeout", type=int, default=180)
        p.add_argument("--use-proxy", action="store_true",
                       help="走系统代理（默认直连；代理常把连不上伪装成 502）")

    d = sub.add_parser("dry", help="不发请求，只检查配置与成本量级")
    common(d)
    d.set_defaults(func=cmd_dry)

    pr = sub.add_parser("probe", help="只跑 1 页，验证连通性与返回格式")
    common(pr)
    pr.set_defaults(func=cmd_probe)

    r = sub.add_parser("run", help="批量跑，产出 shotlist.json / shotlist.txt")
    common(r)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--episode", default="")
    r.add_argument("--sleep", type=float, default=0.0, help="每页之间的间隔秒")
    r.add_argument("--include-non-narrative", action="store_true",
                   help="非叙事页（封面/设定/目录）也写进清单。默认跳过")
    r.add_argument("--allow-partial", action="store_true",
                   help="允许「有格没被解读」的页通过。默认拒收（防静默漏读）")
    r.add_argument("--no-inspection-fallback", action="store_true",
                   help="关掉「内容审核被拦 → 换模型/缩图重试」的回退阶梯")
    r.add_argument("--panel-offset", type=int, default=0,
                   help="在自动算出的全局格号上再加此偏移（① 只吃部分页时用）")
    r.add_argument("--local-panel-index", action="store_true",
                   help="panel 用**本页**编号（默认换算成全卷全局编号）")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="VLM 结果 vs 几何切格的对照报告")
    common(c)
    c.add_argument("--shotlist", required=True)
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("retxt",
                       help="把已有 shotlist.json 的格号换算成全卷全局编号并重写 "
                            "txt/review（**不调 API**，可免费修正旧结果）")
    t.add_argument("--shotlist", required=True)
    t.add_argument("--dir", default="",
                   help="按此目录重算偏移（不给我则按 JSON 内页序累加）")
    t.add_argument("--out-dir", default="")
    t.add_argument("--episode", default="")
    t.add_argument("--include-non-narrative", action="store_true")
    t.add_argument("--panel-offset", type=int, default=0)
    t.add_argument("--already-global", action="store_true",
                   help="JSON 里已经是全局编号，只重写 txt")
    t.set_defaults(func=cmd_retxt)

    a = ap.parse_args()
    if a.cmd in ("probe", "run") and not a.api_key:
        err("缺 API key：设环境变量 VLM_API_KEY，或用 --api-key 传（别写进工作流文件）")
        return 2
    if a.cmd in ("probe", "run", "compare") and not a.pages and not a.dir:
        err("必须给 --pages 或 --dir")
        return 2
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
