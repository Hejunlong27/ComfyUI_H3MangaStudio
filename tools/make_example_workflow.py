"""生成一份可直接拖进 ComfyUI 的示例工作流 JSON。

为什么用脚本生成而不是手写
--------------------------
`widgets_values` 是**按位置存的**：连线型参数不占槽位、Autogrow 会展开成
`输入名.插槽名`、个别节点前端还会额外插控件（LoadImage 的上传按钮、
INT 的 control_after_generate）。手写必错，而且错了不报错、只是参数整体串位。

所以本脚本直接问真实 ComfyUI 加载器要 `INPUT_TYPES()`，自己算
「哪些是 widget、哪些是连线、默认值是多少、节点该多高」，再拼 flat graph。

用法（cwd 为 ComfyUI 根目录）：
    .venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tools/make_example_workflow.py

生成后用 `tools/verify_workflow.py` 校验（会逐节点核对 widgets 数量与连线槽位）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PKG)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)

os.chdir(COMFY_ROOT)
for p in (COMFY_ROOT, CUSTOM_NODES):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.argv = [sys.argv[0], "--cpu"]

import nodes  # noqa: E402

# 连线型类型 ID（不占 widgets 槽位）
# 连线 / widget 的判定见 `_is_widget_type()`。
# 这里曾经是一份「连线类型白名单」（LINK_TYPES），已废弃：白名单每新增一个
# 自定义类型（如 H3_MANGA_VLM）都要同步维护，漏改就把连线输入写成 widget。
LINK_TYPES = set()          # 保留名字仅为兼容旧引用；判定请用 _is_widget_type()

# 前端会额外插入的控件（定义里看不到）
# 元素是 (名字, 默认值, 是否多行, 类型) —— 类型用于「widget 转 input」的序列化
KNOWN_EXTRA_WIDGETS = {
    # ★ 名字必须是 ``upload``（前端给 LoadImage 插的上传控件），值恒为字符串 "image"。
    #   早先误写成 ("image", "image")：**与真实 widget ``image`` 重名**，
    #   于是 over 里给的文件名把两个槽位都填成了同一个值 →
    #   生成 ``["xx.png", "xx.png"]``（第二项本该是 "image"），
    #   前端恢复时第二个控件拿到文件名，图片加载行为不可预期。
    "LoadImage": [("upload", "image", "STRING")],
}

# 需要特殊处理的下拉默认值：这些项**不显式写就会取到错的默认值**，
# 而且取错了**不会报错**，只会静默出坏结果。
# ★★ 必须在 build() 里真正合并进 over_of。早先版本只定义了这个常量却没在任何地方
#    引用它（全文只出现 1 次），后果是 ``CLIPLoader.type`` 一直是默认的
#    ``stable_diffusion`` —— H3 的文本编码器是 Qwen3VL，按 SD 的 CLIP 结构去解析，
#    prompt 根本没法被正确编码，出片自然与漫画内容毫无关联。
WIDGET_OVERRIDE = {
    "CLIPLoader": {"type": "minimax"},
    # SaveVideo 的 format / codec 留 null 会让前端下拉空白（后端虽能用 default 兜住，
    # 但界面显示为空、用户无法确认），显式写 "auto"。
    "SaveVideo": {"format": "auto", "codec": "auto"},
}

# 示例用到的模型文件（**逐字来自本机 models/ 目录**）
UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

PAGE_IMAGE = "manga_page.png"

# ★ 一角色一图：一个角色 = 一张「四视图 / 角色设定卡」
#   为什么不给每个角色多张独立定妆图：
#     ① 歧义 —— 正面/侧面/全身之间只要发型、服装或画风有细微差异，模型就会摇摆；
#     ② 稀缺 —— 官方参考图上限 9 张，还要与分镜格锚点、风格基准图共享额度。
#   一张设定卡（正/侧/背/表情并排）既把视角信息给全，又只占 1 个槽。
CHAR_CARDS = [
    ("纲吉", "char_tsuna_card.png"),
    ("Reborn", "char_reborn_card.png"),
]

INLINE_SCRIPT = """# episode: Reborn 第30卷 目标283
# style: manga_anime
# default_size: 1344x768
=== s01 ===
panel: 0
duration: 4
camera: tilt_up
action: 黄昏的住宅区街景，电线杆与交错的电线横过天空
=== s02 ===
panel: 1
shots: 2
duration: 5
characters: 纲吉
camera: slow_push_in
content: 纲吉从门后走出 | 抬眼看向镜头
action: @纲吉 推开半掩的门，从门后走出 | @纲吉 抬眼看向镜头，瞳孔收紧
speaker: 纲吉
dialogue: 门轴吱呀一声。 | 这一刻终于还是到来了。
"""

README_NOTE = """# H3 Manga Studio · 漫改动画量产示例

## 三步跑通
1. 把 3 个 **LoadImage** 的文件名换成你自己的图（1 张漫画页 + **每个角色 1 张设定卡**）
2. 改 **② 参考包** 的 `character_names`（与图槽一一对应，**一个角色一个名字**）
3. 改 **③ 分镜清单**（`source=hybrid` 时：先由 ① 的分镜格自动出镜，文本只覆盖要改的镜）

## ★ 一角色一图（默认）
每个角色只锁 **一张** 参考图，用「**四视图 / 角色设定卡**」把正/侧/背/表情
并到同一张图里。理由：
* 多张独立定妆图之间只要有发型/服装/角度的细微差异，模型就会摇摆；
* 官方参考图上限 9 张，还要跟分镜格锚点、风格基准图共享额度 —— 一个角色吃 3 张会挤掉别人。

确实需要多图（角色换装 / 多形态）时，把 ② 的「一角色一图」关掉即可。

## 首次运行建议
* 先把 **⑦ 整集渲染** 的 `max_shots` 设成 `1`，确认画面 OK 再改成 `0` 跑整集
* 想看机器实际拿到的提示词 → 点 **⑨ 计划预览** 或 **⑤ 提示词装配** 节点上的面板
* 跑之前可以单独跑 **⑧ 一致性审计** 看有没有角色缺图

## 接线顺序（重要）
分镜清单 → **分镜绑定** → 提示词装配 → 整集渲染
必须让「分镜绑定」接在「提示词装配」之前，否则提示词里没有分镜格锚点，槽位会对不上。

## 参数速查
| 想要 | 调什么 |
|---|---|
| 更快 | 挂 turbo LoRA + steps 降到 8；或降 max_ref_images |
| 更像原角色 | 换成更清晰的设定卡 + 补 character_describes 的锁定句 |
| 画面更贴分镜 | 确认 board 接到 ⑦；运镜改 static |
| 单镜失败不中断 | on_error 保持 skip（默认），失败原因会写进报告 |
"""


# 前端虚拟节点：后端 NODE_CLASS_MAPPINGS 里没有，需要手写规格
# (widgets, links, autogrow) —— outputs 由「没有后端类」自然推导为空
VIRTUAL_NODES = {
    "MarkdownNote": ([("text", "", True, "STRING")], [], {}),
    "Note": ([("text", "", True, "STRING")], [], {}),
}


WIDGET_PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _is_widget_type(t) -> bool:
    """这个输入在 ComfyUI 里是 widget 还是连线？

    widget 只有五类：四个原始类型 + 下拉框；下拉框有两种形态 ——
      旧式 ``list``（如 ``CLIPLoader.clip_name``）
      V3 式 ``"COMBO"``（如 ``io.Combo.Input`` 产出的 ``size_mode``）
    （Autogrow 另行处理。）**其余一律是连线** —— MODEL / IMAGE / LATENT /
    CONDITIONING / CLIP / VAE 这些内置类型，以及本插件的 H3_MANGA_* 自定义类型。

    ★ 用这个规则而不是白名单：白名单每加一个自定义类型都要同步维护，
      漏改的代价是「连线输入被写成 widget，值变 null」这种静默错。
    """
    if isinstance(t, list):          # 旧式下拉框
        return True
    return isinstance(t, str) and t in WIDGET_PRIMITIVES


def _widget_type(t) -> str:
    """把 INPUT_TYPES 里的类型描述归一成前端认识的类型名。

    这个类型**只用于**「widget 转 input」的序列化：当一条连线接到某个带 default
    的输入上时，节点的 ``inputs`` 数组里要写 ``"type": "INT"`` 这样的名字。
    """
    if isinstance(t, str):
        return t
    if isinstance(t, list):      # 下拉框
        return "COMBO"
    return "STRING"


def describe(node_id: str):
    """返回 (widgets, links, autogrow)。

    widgets: [(name, default, multiline, type)]

    ★ 为什么要带 ``type``：``width`` / ``height`` / ``length`` / ``prompt`` 这类
      **带 default 的输入在 ComfyUI 里是 widget**。想让一条连线接到它上面，
      节点的 ``inputs`` 里必须有
      ``{"name":.., "type":.., "widget":{"name":..}}``
      —— 这就是前端「把 widget 转成 input」的序列化形式
      （权威样例：官方模板 ``video_minimax_h3_r2v.json``）。
      少了它，连线在 ``links`` 里存在但 ``target_slot`` 是 null，**ComfyUI 会丢掉它**，
      表现就是「界面上这条线连不上」。
    """
    if node_id in VIRTUAL_NODES:
        return VIRTUAL_NODES[node_id]

    cls = nodes.NODE_CLASS_MAPPINGS[node_id]
    it = cls.INPUT_TYPES()
    widgets = []
    links = []
    autogrow = {}

    for sect in ("required", "optional"):
        for name, v in (it.get(sect) or {}).items():
            t = v[0] if isinstance(v, (list, tuple)) and v else v
            meta = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else {}

            if isinstance(t, str) and t == "COMFY_AUTOGROW_V3":
                tpl = meta.get("template") or {}
                inner = (tpl.get("input") or {}).get("required") or {}
                itype = "IMAGE"
                if inner:
                    ispec = next(iter(inner.values()))
                    itype = ispec[0] if isinstance(ispec, (list, tuple)) else ispec
                autogrow[name] = {
                    "prefix": tpl.get("prefix") or "",
                    "min": tpl.get("min") or 0,
                    "max": tpl.get("max") or 0,
                    "inner_type": itype,
                }
                continue

            # ★ 分类规则（本质规则，不是白名单）：
            #   原始类型 INT/FLOAT/STRING/BOOLEAN 或「选项列表」（COMBO）→ widget；
            #   其余一切（MODEL/IMAGE/H3_MANGA_* 等自定义类型）→ 连线。
            #   曾经的写法是 `t in LINK_TYPES`（白名单），**每新增一个自定义类型
            #   都要记得改两处白名单**，漏了就把连线输入误判成 widget
            #   （实测：v0.79 新增 H3_MANGA_VLM 时，⑭ 的 vlm 被写成 widget-input，
            #    widgets_values 首位变 null）。改成本质规则后不会再复发。
            if not _is_widget_type(t):
                links.append((name, t))
                continue

            default = meta.get("default")
            if default is None and isinstance(t, list) and t:
                default = t[0]
            wtype = _widget_type(t)
            if default is None and wtype == "STRING":
                # ★ 别写 null：官方 execute 会拿到 None 而不是空串
                #（实测官方多参节点的 prompt 位置写成 null 时行为异常）
                default = ""
            widgets.append((name, default, bool(meta.get("multiline")), wtype))
            if meta.get("control_after_generate"):
                widgets.append(("control_after_generate", "randomize", False, "STRING"))

    for extra in KNOWN_EXTRA_WIDGETS.get(node_id, []):
        ename, eval_ = extra[0], extra[1]
        etype = extra[2] if len(extra) > 2 else "STRING"
        widgets.append((ename, eval_, False, etype))

    return widgets, links, autogrow


def estimate_height(widgets, links, used_slots) -> int:
    """粗略估算节点高度（ComfyUI 的节点尺寸是存下来的，给个合理值即可）。"""
    h = 64
    for w in widgets:
        if w[0] == "control_after_generate":
            continue
        h += 92 if w[2] else 26
    h += 26 * (len(links) + len(used_slots))
    return max(120, min(900, h))


# --------------------------------------------------------------------------
# 图定义：按列（层）组织，列内自上而下
# --------------------------------------------------------------------------
NODES = [
    ("unet", "UNETLoader", "UNET · Ref2VA（r2v 专用）", {"unet_name": UNET}, "模型"),
    ("clip", "CLIPLoader", "CLIP · type 必须是 minimax", {"clip_name": CLIP}, "模型"),
    ("vae_v", "VAELoader", "VAE · 视频", {"vae_name": VAE_VIDEO}, "模型"),
    ("vae_a", "VAELoader", "VAE · 音频", {"vae_name": VAE_AUDIO}, "模型"),

    # ---- LoRA 区：加速 LoRA → 轮盘位 ----
    # ★ 为什么是两个 LoraLoaderModelOnly 串联、而不是「轮盘节点」：
    #   本机 rgthree 的 Power Lora Loader **未注册**（rgthree-comfy 没成功加载），
    #   efficiency 的 LoRA Stacker 只被它自己的大 Loader 消费 —— 可用的只有核心的
    #   LoraLoaderModelOnly。于是串两级：第 1 级放加速 LoRA（固定不动），
    #   第 2 级是「轮盘位」——**默认处于旁路(bypass)状态**（ComfyUI 里 mode=4 = 不执行、
#   输入直通输出）。想加风格/角色 LoRA：在界面上选中该节点按 Ctrl+B 解除旁路，
#   再把 lora_name 换成你的文件。★ 不要试图把文件名填成 "none" 来关闭 ——
#   核心节点的下拉里没有这个值，会取不到。
    ("lora_fast", "LoraLoaderModelOnly", "LoRA · 加速（turbo）", {
        "lora_name": TURBO_LORA, "strength_model": 1.0}, "LoRA"),
    # 默认 **旁路**（节点上按 Ctrl+B 可解除）—— 想加风格/角色 LoRA 时：
    # 解除旁路 → 把 lora_name 换成你的文件 → 调 strength。
    ("lora_alt", "LoraLoaderModelOnly", "LoRA · 轮盘位（默认旁路；Ctrl+B 解除）", {
        "lora_name": TURBO_LORA, "strength_model": 0.0}, "LoRA"),

    ("img_pg", "LoadImage", "输入 · 漫画页", {"image": PAGE_IMAGE}, "输入图"),
    ("img_c1", "LoadImage", f"输入 · {CHAR_CARDS[0][0]} 设定卡（四视图）",
     {"image": CHAR_CARDS[0][1]}, "输入图"),
    ("img_c2", "LoadImage", f"输入 · {CHAR_CARDS[1][0]} 设定卡",
     {"image": CHAR_CARDS[1][1]}, "输入图"),

    ("split", "H3MangaPanelSplitter", "① 漫画页切分镜格", {}, "素材"),

    # ---- 预览区：把 ① 的三个输出显式接出来 ----
    # ★ 以前这三个输出全悬空 → 切完看不见、错了也不知道。
    #   PreviewImage = 节点上即时看（存 temp 目录，随时可删）
    #   SaveImage    = 落盘到 <ComfyUI>/output/h3manga/（可长期留档）
    #   PreviewAny   = 看文本报告（格数、warning、「过大连通块」告警都在这里）
    ("prev_panels", "PreviewImage", "预览 · 切出的分镜格", {}, "预览"),
    ("prev_grid", "PreviewImage", "预览 · 坐标网格（手填框用）", {}, "预览"),
    ("prev_report", "PreviewAny", "预览 · ① 切格报告（格数/警告）", {}, "预览"),
    ("save_panels", "SaveImage", "落盘 · 分镜格 → output/h3manga", {
        "filename_prefix": "h3manga/panels"}, "预览"),
    ("refs", "H3MangaRefPack", "② 参考包", {
        "character_names": ", ".join(n for n, _f in CHAR_CARDS),
        "character_describes": "棕发橙眼，黑色并盛中学学生制服，左手中指戴彭格列戒指\n"
                               "黑西装婴儿，黑色卷鬓角，礼帽上有橙色变色龙",
    }, "素材"),

    ("script", "H3MangaScriptLoader", "③ 分镜清单（board 自动出镜 + 文本覆盖）", {
        "source": "hybrid", "inline_text": INLINE_SCRIPT,
        "default_duration": 5.0, "size_mode": "preset",
        "size_preset": "16:9 横屏", "size_scale": 0.5}, "编排"),
    ("bind", "H3MangaShotBinder", "④ 分镜绑定", {}, "编排"),
    ("comp", "H3MangaPromptComposer", "⑤ 提示词装配", {
        "negative": "flickering, identity drift, extra fingers, deformed hands"},
        "编排"),
    ("audit", "H3MangaConsistencyAudit", "⑧ 一致性审计", {}, "编排"),

    ("render", "H3MangaEpisodeRender", "⑦ 整集渲染 ★（max_shots=0 即整集）", {
        "steps": 8,
        # ★ LoRA 改由外置「LoRA 区」统一管理（UNETLoader → 加速 → 轮盘位 → 这里），
        #   所以设 none。本节点内部的 lora_name 仍可用，但那会与 LoRA 区叠加。
        "lora_name": "none", "max_shots": 0}, "执行"),

    ("vid", "CreateVideo", "合成视频（24fps）", {"fps": 24}, "输出"),
    ("save", "SaveVideo", "保存整集视频", {"filename_prefix": "H3MangaStudio/episode"},
     "输出"),
    ("prev", "PreviewAny", "预览渲染报告", {}, "输出"),

    ("note", "MarkdownNote", "使用说明", {"text": README_NOTE}, "说明"),
]

# (源变量, 源输出序号, 目标变量, 目标输入名)
LINKS = [
    # ★ 模型链：UNETLoader → LoRA 加速 → LoRA 轮盘位 → ⑦
    ("unet", 0, "lora_fast", "model"),
    ("lora_fast", 0, "lora_alt", "model"),
    ("lora_alt", 0, "render", "model"),
    ("clip", 0, "render", "clip"),
    ("vae_v", 0, "render", "vae"),
    ("vae_a", 0, "render", "audio_vae"),

    ("img_pg", 0, "split", "image"),

    ("img_c1", 0, "refs", "character_images.character_0"),
    ("img_c2", 0, "refs", "character_images.character_1"),

    # ★ 预览：把 ① 的三个输出显式接出来（以前全悬空 → 切完看不见）
    ("split", 1, "prev_panels", "images"),
    ("split", 3, "prev_grid", "images"),
    ("split", 2, "prev_report", "source"),
    ("split", 1, "save_panels", "images"),

    ("split", 0, "script", "board"),      # ★ v0.7：③ 从 board 自动出镜

    ("script", 0, "bind", "plan"),
    ("split", 0, "bind", "board"),
    ("refs", 0, "bind", "refs"),

    ("bind", 0, "comp", "plan"),
    ("refs", 0, "comp", "refs"),

    ("comp", 0, "audit", "plan"),
    ("refs", 0, "audit", "refs"),

    ("comp", 0, "render", "plan"),
    ("refs", 0, "render", "refs"),
    ("split", 0, "render", "board"),

    ("render", 0, "vid", "images"),
    ("render", 1, "vid", "audio"),
    ("vid", 0, "save", "video"),
    ("render", 2, "prev", "source"),
]

COLUMNS = ["模型", "LoRA", "输入图", "素材", "预览", "编排", "执行", "输出", "说明"]

COL_COLOR = {
    "模型": "#3f789e",
    "LoRA": "#0f766e",
    "输入图": "#88A",
    "素材": "#a1309b",
    "预览": "#4d7c0f",
    "编排": "#b58b2a",
    "执行": "#444",
    "输出": "#b06634",
    "说明": "#666",
}

# --------------------------------------------------------------------------
# 矩阵式布局参数
# --------------------------------------------------------------------------
# 两层矩阵，规则统一：
#   · 组内矩阵 —— 每个 group 内部的节点按 INNER_COLS 列折行；行高 = 该行最高节点，
#                 同行节点顶对齐。所以「同一列的第 N 行」永远在同一水平线上。
#   · 组间矩阵 —— 每个 group 视为网格中的一个单元，按 OUTER_COLS 列折行；
#                 同列的 group 左对齐（列宽取该列最宽者），同行的 group 顶对齐。
# 之所以做成两层而不是一层「按列纵向堆叠」：层与层之间的从属关系要靠
# 位置关系的规整性读出来 —— 矩阵比长条更容易一眼看出「哪个节点属于哪一层」。
# ★ 「旁路(bypass)」的节点：ComfyUI 里 mode=4 表示不执行、输入直通输出。
#   核心的 LoraLoaderModelOnly 的 lora_name 下拉**没有 "none" 选项**，
#   所以「这个 LoRA 槽先关着」只能靠旁路表达，不能填假名字。
BYPASS_VARS = {"lora_alt"}

COL_WIDTH = 430          # 节点统一宽度（所有节点同宽 → 网格才成立）
INNER_COLS = 2           # 组内每行放几个节点
INNER_GAP_X = 56         # 组内水平间距
INNER_GAP_Y = 40         # 组内垂直间距
OUTER_COLS = 4           # 组间每行放几个 group
OUTER_GAP_X = 88         # group 间水平间距
OUTER_GAP_Y = 96         # group 间垂直间距
GROUP_PAD_X = 30         # group 框左右留白
GROUP_PAD_TOP = 58       # group 框顶部（让出标题带）
GROUP_PAD_BOTTOM = 30    # group 框底部留白
CANVAS_SCALE = 0.30      # 工作流默认缩放（让整张矩阵图一屏可见）


# ==========================================================================
# 第二版：官方节点接线版（插件只做前置，采样/解码在你自己的图里）
# ==========================================================================
README_NOTE_OFFICIAL = """\
# H3 Manga Studio · 官方节点接线版（v0.7）

这一版演示 **「插件只做前置」**：采样 / 放大 / 二采 / 解码全部由你自己的图负责。

数据流：
  ① 切分镜格 → ② 参考包 → ③ 分镜清单（可从 board 自动出镜）→ ④ 绑定 → ⑤ 提示词装配
  → ⑫-A 镜头出栈（prompt + 9 个参考图槽 + 宽高帧数）
  → 官方 MiniMax H3 多参视频节点
  → 你自己的采样链（这里用官方五件套；换成 KSampler / 加 LoRA / 加放大都随意）
  → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo

## ★ 接线要点

* 输出「参考图 N」→ 官方输入 `ref_image_(N-1)`
  （官方 Autogrow 槽名是 **0 起**，提示词里的 `<Picture i>` 是 **1 起**，两者差 1）
* 用不到的槽**不要接**：留空即可，官方会自动跳过，不影响序号
* ⑫-A 的 `shot_index` **从 1 起**；要渲第 2 镜就再复制一份 ⑫-A 并改成 2
* 想少接 9 根线：把「⑫-A + 官方节点」整段换成 **⑫-B 单镜条件（直出）**，
  它直接吐 `positive` + `latent`，接 BasicGuider / SamplerCustomAdvanced 即可

## 分辨率

这里的宽/高/帧数由 **③ 的画幅选择器**算好（对齐官方 adapt_canvas 规则）后经 ⑫-A 传来。
想手动改：直接改官方节点上的 width/height（把连线断开即回到手动值）。
"""

NODES_OFFICIAL = [
    ("unet", "UNETLoader", "UNET · Ref2VA（r2v 专用）", {"unet_name": UNET}, "模型"),
    ("clip", "CLIPLoader", "CLIP · type 必须是 minimax", {"clip_name": CLIP}, "模型"),
    ("vae_v", "VAELoader", "VAE · 视频", {"vae_name": VAE_VIDEO}, "模型"),
    ("vae_a", "VAELoader", "VAE · 音频", {"vae_name": VAE_AUDIO}, "模型"),

    # ---- LoRA 区：加速 LoRA → 轮盘位 ----
    # ★ 为什么是两个 LoraLoaderModelOnly 串联、而不是「轮盘节点」：
    #   本机 rgthree 的 Power Lora Loader **未注册**（rgthree-comfy 没成功加载），
    #   efficiency 的 LoRA Stacker 只被它自己的大 Loader 消费 —— 可用的只有核心的
    #   LoraLoaderModelOnly。于是串两级：第 1 级放加速 LoRA（固定不动），
    #   第 2 级是「轮盘位」——**默认处于旁路(bypass)状态**（ComfyUI 里 mode=4 = 不执行、
#   输入直通输出）。想加风格/角色 LoRA：在界面上选中该节点按 Ctrl+B 解除旁路，
#   再把 lora_name 换成你的文件。★ 不要试图把文件名填成 "none" 来关闭 ——
#   核心节点的下拉里没有这个值，会取不到。
    ("lora_fast", "LoraLoaderModelOnly", "LoRA · 加速（turbo）", {
        "lora_name": TURBO_LORA, "strength_model": 1.0}, "LoRA"),
    # 默认 **旁路**（节点上按 Ctrl+B 可解除）—— 想加风格/角色 LoRA 时：
    # 解除旁路 → 把 lora_name 换成你的文件 → 调 strength。
    ("lora_alt", "LoraLoaderModelOnly", "LoRA · 轮盘位（默认旁路；Ctrl+B 解除）", {
        "lora_name": TURBO_LORA, "strength_model": 0.0}, "LoRA"),

    ("img_pg", "LoadImage", "输入 · 漫画页", {"image": PAGE_IMAGE}, "输入图"),
    ("img_c1", "LoadImage", f"输入 · {CHAR_CARDS[0][0]} 设定卡（四视图）",
     {"image": CHAR_CARDS[0][1]}, "输入图"),
    ("img_c2", "LoadImage", f"输入 · {CHAR_CARDS[1][0]} 设定卡",
     {"image": CHAR_CARDS[1][1]}, "输入图"),

    ("split", "H3MangaPanelSplitter", "① 漫画页切分镜格", {}, "素材"),

    # ---- 预览区：把 ① 的三个输出显式接出来 ----
    # ★ 以前这三个输出全悬空 → 切完看不见、错了也不知道。
    #   PreviewImage = 节点上即时看（存 temp 目录，随时可删）
    #   SaveImage    = 落盘到 <ComfyUI>/output/h3manga/（可长期留档）
    #   PreviewAny   = 看文本报告（格数、warning、「过大连通块」告警都在这里）
    ("prev_panels", "PreviewImage", "预览 · 切出的分镜格", {}, "预览"),
    ("prev_grid", "PreviewImage", "预览 · 坐标网格（手填框用）", {}, "预览"),
    ("prev_report", "PreviewAny", "预览 · ① 切格报告（格数/警告）", {}, "预览"),
    ("save_panels", "SaveImage", "落盘 · 分镜格 → output/h3manga", {
        "filename_prefix": "h3manga/panels"}, "预览"),
    ("refs", "H3MangaRefPack", "② 参考包", {
        "character_names": ", ".join(n for n, _f in CHAR_CARDS),
        "character_describes": "棕发橙眼，黑色并盛中学学生制服，左手中指戴彭格列戒指\n"
                               "黑西装婴儿，黑色卷鬓角，礼帽上有橙色变色龙",
    }, "素材"),

    ("script", "H3MangaScriptLoader", "③ 分镜清单（board 自动出镜 + 文本覆盖）", {
        "source": "hybrid", "inline_text": INLINE_SCRIPT,
        "default_duration": 5.0, "size_mode": "preset",
        "size_preset": "16:9 横屏", "size_scale": 0.5}, "编排"),
    ("bind", "H3MangaShotBinder", "④ 分镜绑定", {}, "编排"),
    ("comp", "H3MangaPromptComposer", "⑤ 提示词装配", {
        "negative": "flickering, identity drift, extra fingers, deformed hands"},
        "编排"),

    ("shot", "H3MangaShotOut", "⑫-A 镜头出栈（喂官方多参节点）", {
        "shot_index": 1, "include_anchor": True, "ref_image_size": "match"}, "前置"),

    ("mmh3", "MiniMaxH3ReferenceToVideo", "官方 · MiniMax H3 多参视频节点", {
        "ref_image_size": "match"}, "采样"),
    ("ksel", "KSamplerSelect", "采样器", {"sampler_name": "res_multistep"}, "采样"),
    ("sched", "BasicScheduler", "调度器（steps 在这里改）", {
        "scheduler": "simple", "steps": 8}, "采样"),
    ("noise", "RandomNoise", "噪声", {}, "采样"),
    ("guider", "BasicGuider", "引导（conditioning 在这里进）", {}, "采样"),
    ("samp", "SamplerCustomAdvanced", "采样 · 官方五件套", {}, "采样"),
    ("dec_v", "VAEDecode", "解码 · 视频", {}, "采样"),
    ("dec_a", "VAEDecodeAudio", "解码 · 音频", {}, "采样"),

    ("vid", "CreateVideo", "合成视频（24fps）", {"fps": 24}, "输出"),
    ("save", "SaveVideo", "保存单镜", {"filename_prefix": "H3MangaStudio/shot01"}, "输出"),
    ("prev", "PreviewAny", "预览槽位对照表", {}, "输出"),

    # ---- 批量出镜：⑦ 内部逐镜循环 ----
    # ★ ComfyUI 的工作流**没有循环**，所以「一次顺序跑完全部镜头」只能由节点内部循环
    #   实现 —— 这就是 ⑦。它与上面的「单镜」分支并列：单镜用来调参看效果，
    #   批量用来出全片。max_shots=0 表示不限，跑完计划里的每一镜。
    ("render", "H3MangaEpisodeRender", "⑦ 批量出镜 ★｜顺序跑完全部镜头", {
        "steps": 8, "lora_name": "none", "max_shots": 0}, "批量"),
    ("vid2", "CreateVideo", "合成 · 整页（24fps）", {"fps": 24}, "批量"),
    ("save2", "SaveVideo", "保存 · 整页视频", {"filename_prefix": "H3MangaStudio/page"},
     "批量"),
    ("prev_batch", "PreviewAny", "预览 · 批量渲染报告", {}, "批量"),

    ("note", "MarkdownNote", "使用说明（官方接线版）",
     {"text": README_NOTE_OFFICIAL}, "说明"),
]

LINKS_OFFICIAL = [
    ("clip", 0, "mmh3", "clip"),
    ("vae_v", 0, "mmh3", "vae"),
    ("vae_a", 0, "mmh3", "audio_vae"),

    ("img_pg", 0, "split", "image"),
    ("img_c1", 0, "refs", "character_images.character_0"),
    ("img_c2", 0, "refs", "character_images.character_1"),

    # ★ 预览：把 ① 的三个输出显式接出来（以前全悬空 → 切完看不见）
    ("split", 1, "prev_panels", "images"),
    ("split", 3, "prev_grid", "images"),
    ("split", 2, "prev_report", "source"),
    ("split", 1, "save_panels", "images"),

    ("split", 0, "script", "board"),
    ("script", 0, "bind", "plan"),
    ("split", 0, "bind", "board"),
    ("refs", 0, "bind", "refs"),
    ("bind", 0, "comp", "plan"),
    ("refs", 0, "comp", "refs"),

    ("comp", 0, "shot", "plan"),
    ("refs", 0, "shot", "refs"),
    ("split", 0, "shot", "board"),

    # ⑫-A → 官方节点：输出「参考图 N」接 ref_image_(N-1)（★ 官方槽名 0 起）
    # ★ 9 个参考图槽**全部连上**：用不到的槽 ⑫-A 会输出 None、官方会自动跳过，
    #   所以连满是无害的；好处是以后要多加参考图**不用改接线**。
    ("shot", 0, "mmh3", "prompt"),
    ("shot", 1, "mmh3", "ref_images.ref_image_0"),
    ("shot", 2, "mmh3", "ref_images.ref_image_1"),
    ("shot", 3, "mmh3", "ref_images.ref_image_2"),
    ("shot", 4, "mmh3", "ref_images.ref_image_3"),
    ("shot", 5, "mmh3", "ref_images.ref_image_4"),
    ("shot", 6, "mmh3", "ref_images.ref_image_5"),
    ("shot", 7, "mmh3", "ref_images.ref_image_6"),
    ("shot", 8, "mmh3", "ref_images.ref_image_7"),
    ("shot", 9, "mmh3", "ref_images.ref_image_8"),
    ("shot", 10, "mmh3", "width"),
    ("shot", 11, "mmh3", "height"),
    ("shot", 12, "mmh3", "length"),

    # 官方 conditioning → 引导；latent → 采样输入
    ("mmh3", 0, "guider", "conditioning"),
    ("unet", 0, "lora_fast", "model"),
    ("lora_fast", 0, "lora_alt", "model"),
    ("lora_alt", 0, "guider", "model"),
    ("lora_alt", 0, "sched", "model"),
    ("ksel", 0, "samp", "sampler"),
    ("sched", 0, "samp", "sigmas"),
    ("noise", 0, "samp", "noise"),
    ("guider", 0, "samp", "guider"),
    ("mmh3", 1, "samp", "latent_image"),

    # 解码 → 合成
    ("samp", 0, "dec_v", "samples"),
    ("vae_v", 0, "dec_v", "vae"),
    ("samp", 0, "dec_a", "samples"),
    ("vae_a", 0, "dec_a", "vae"),
    ("dec_v", 0, "vid", "images"),
    ("dec_a", 0, "vid", "audio"),
    ("vid", 0, "save", "video"),
    ("shot", 13, "prev", "source"),

    # ★ 批量分支：⑦ 内部循环 → 一次输出**全部镜头**（与「单镜」并列）
    ("lora_alt", 0, "render", "model"),
    ("clip", 0, "render", "clip"),
    ("vae_v", 0, "render", "vae"),
    ("vae_a", 0, "render", "audio_vae"),
    ("comp", 0, "render", "plan"),
    ("refs", 0, "render", "refs"),
    ("split", 0, "render", "board"),
    ("render", 0, "vid2", "images"),
    ("render", 1, "vid2", "audio"),
    ("vid2", 0, "save2", "video"),
    ("render", 2, "prev_batch", "source"),
]

COLUMNS_OFFICIAL = ["模型", "LoRA", "输入图", "素材", "预览", "编排", "前置", "采样", "批量", "输出", "说明"]

COL_COLOR_OFFICIAL = {
    "模型": "#3f789e",
    "LoRA": "#0f766e",
    "输入图": "#88A",
    "素材": "#a1309b",
    "预览": "#4d7c0f",
    "编排": "#b58b2a",
    "前置": "#2f8f6f",
    "采样": "#8a4fbf",
    "批量": "#7c2d12",
    "输出": "#b06634",
    "说明": "#666",
}


# ==========================================================================
# 第三版：VLM 全自动版（漫画图 → VLM 读图 → 分镜计划 → 官方多参节点）
# ==========================================================================
README_NOTE_VLM = """\
# H3 Manga Studio · VLM 全自动版（v0.79）

这一版把「**看漫画 → 出分镜**」也交给模型 —— 补上整条链路里唯一还靠人肉的一环。

数据流：
  LoadImage(漫画页) ─┬─► ① 切分镜格 ──board────┐
                     │                         ├─► ⑬ VLM 漫画解读（逐格读画面）
                     └──整页原图───────────────┘          │
  LoadImage(角色卡) ─► ② 参考包                          ▼
                                            ⑭ VLM 转分镜计划（纯本地，不花钱）
                                                        │
                      ④ 分镜绑定 → ⑤ 提示词装配 → ⑫-A → 官方多参节点
                                        → 你的采样链 → 解码 → CreateVideo → SaveVideo

## ★ 第一次用必读：配 API key

⑬ 要访问多模态大模型（OpenAI 兼容协议）。**别把 key 填进节点** ——
它会被写进工作流文件、进而被提交进 git。用环境变量：

    set VLM_API_KEY=sk-xxxxxxxx
    （然后**重启 ComfyUI**，环境变量只在启动时读）

临时调试也可以直接填 ⑬ 的 `api_key`，但别保存/分享那份工作流。

## 推荐模型（实测）

| 模型 | 速度 | 说明 |
|---|---|---|
| **qwen3.5-omni-plus** ★ | 6~10 s/页 | 首选。格数判断与几何切格 12/12 一致 |
| qwen3-vl-plus | ~20 s/页 | 备选，也是内容审核被拦时的回退模型 |
| ⚠ qwen3.8-max / deepseek-v4.1-flash | — | **没有视觉能力**，传图会返回空 |

默认端点是阿里云百炼。本地 Ollama 可填 `http://127.0.0.1:11434/v1`。

## 先小批验证，再全量

⑬ 的 `max_pages` 默认 **1**（只读第 1 页）。确认效果与花费都 OK 后改 **0** 跑全部。
量级参考：约 **3,000 tokens/页**、**6~11 秒/页**。

## 哪些字段可信

* **格数 / 画面内容 / 角色 / 页型** —— 跨模型稳定，可以直接用；
* **`shots`（一格几镜）** —— **跨次运行不稳定**（同一页可能跑出不同镜数），
  已按「保留 VLM 判断」处理。不认同就在 ⑭ 把 `force_shots` 设成 1 统一压平。
* 封面 / 设定页 / 目录页会被**自动跳过**（不出镜头）—— 这是几何切格给不出的信号。
  想保留就在 ⑭ 打开 `include_non_narrative`。

## 想改哪里

* 画幅 / 时长 / 镜数 → 改 **⑭**（不调 API，随便改、不花钱）
* 想手工润色分镜 → 把 **⑭ 的「分镜文本」**复制出来改完，贴进 **③**
  （把 ④ 的 `plan` 改接 ③，③ 用 `source=inline`）
* 想换采样 / 加放大 / 二采 → 从 ⑫-A 之后随意改（和官方接线版共用后半段）
"""

NODES_VLM = [
    ("unet", "UNETLoader", "UNET · Ref2VA（r2v 专用）", {"unet_name": UNET}, "模型"),
    ("clip", "CLIPLoader", "CLIP · type 必须是 minimax", {"clip_name": CLIP}, "模型"),
    ("vae_v", "VAELoader", "VAE · 视频", {"vae_name": VAE_VIDEO}, "模型"),
    ("vae_a", "VAELoader", "VAE · 音频", {"vae_name": VAE_AUDIO}, "模型"),

    # ---- LoRA 区：加速 LoRA → 轮盘位 ----
    # ★ 为什么是两个 LoraLoaderModelOnly 串联、而不是「轮盘节点」：
    #   本机 rgthree 的 Power Lora Loader **未注册**（rgthree-comfy 没成功加载），
    #   efficiency 的 LoRA Stacker 只被它自己的大 Loader 消费 —— 可用的只有核心的
    #   LoraLoaderModelOnly。于是串两级：第 1 级放加速 LoRA（固定不动），
    #   第 2 级是「轮盘位」——**默认处于旁路(bypass)状态**（ComfyUI 里 mode=4 = 不执行、
#   输入直通输出）。想加风格/角色 LoRA：在界面上选中该节点按 Ctrl+B 解除旁路，
#   再把 lora_name 换成你的文件。★ 不要试图把文件名填成 "none" 来关闭 ——
#   核心节点的下拉里没有这个值，会取不到。
    ("lora_fast", "LoraLoaderModelOnly", "LoRA · 加速（turbo）", {
        "lora_name": TURBO_LORA, "strength_model": 1.0}, "LoRA"),
    # 默认 **旁路**（节点上按 Ctrl+B 可解除）—— 想加风格/角色 LoRA 时：
    # 解除旁路 → 把 lora_name 换成你的文件 → 调 strength。
    ("lora_alt", "LoraLoaderModelOnly", "LoRA · 轮盘位（默认旁路；Ctrl+B 解除）", {
        "lora_name": TURBO_LORA, "strength_model": 0.0}, "LoRA"),

    ("img_pg", "LoadImage", "输入 · 漫画页", {"image": PAGE_IMAGE}, "输入图"),
    ("img_c1", "LoadImage", f"输入 · {CHAR_CARDS[0][0]} 设定卡（四视图）",
     {"image": CHAR_CARDS[0][1]}, "输入图"),
    ("img_c2", "LoadImage", f"输入 · {CHAR_CARDS[1][0]} 设定卡",
     {"image": CHAR_CARDS[1][1]}, "输入图"),

    ("split", "H3MangaPanelSplitter", "① 漫画页切分镜格", {}, "素材"),

    # ---- 预览区：把 ① 的三个输出显式接出来 ----
    # ★ 以前这三个输出全悬空 → 切完看不见、错了也不知道。
    #   PreviewImage = 节点上即时看（存 temp 目录，随时可删）
    #   SaveImage    = 落盘到 <ComfyUI>/output/h3manga/（可长期留档）
    #   PreviewAny   = 看文本报告（格数、warning、「过大连通块」告警都在这里）
    ("prev_panels", "PreviewImage", "预览 · 切出的分镜格", {}, "预览"),
    ("prev_grid", "PreviewImage", "预览 · 坐标网格（手填框用）", {}, "预览"),
    ("prev_report", "PreviewAny", "预览 · ① 切格报告（格数/警告）", {}, "预览"),
    ("save_panels", "SaveImage", "落盘 · 分镜格 → output/h3manga", {
        "filename_prefix": "h3manga/panels"}, "预览"),
    ("refs", "H3MangaRefPack", "② 参考包", {
        "character_names": ", ".join(n for n, _f in CHAR_CARDS),
        "character_describes": "棕发橙眼，黑色并盛中学学生制服，左手中指戴彭格列戒指\n"
                               "黑西装婴儿，黑色卷鬓角，礼帽上有橙色变色龙",
    }, "素材"),

    ("vlmscribe", "H3MangaVLMScribe", "⑬ VLM 漫画解读（看图出分镜）", {
        "characters": ", ".join(n for n, _f in CHAR_CARDS),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.5-omni-plus",
        "max_pages": 1,          # ★ 先只读 1 页；确认后改 0 跑全部
    }, "VLM"),
    ("vlmplan", "H3MangaVLMPlan", "⑭ VLM 转分镜计划（不花钱，随便调）", {
        "size_preset": "16:9 横屏", "size_scale": 0.5,
        "force_shots": 0, "duration_scale": 1.0, "episode": "Reborn_30",
    }, "VLM"),

    ("bind", "H3MangaShotBinder", "④ 分镜绑定", {}, "编排"),
    ("comp", "H3MangaPromptComposer", "⑤ 提示词装配", {
        "negative": "flickering, identity drift, extra fingers, deformed hands"},
        "编排"),

    ("shot", "H3MangaShotOut", "⑫-A 镜头出栈（喂官方多参节点）", {
        "shot_index": 1, "include_anchor": True, "ref_image_size": "match"}, "前置"),

    ("mmh3", "MiniMaxH3ReferenceToVideo", "官方 · MiniMax H3 多参视频节点", {
        "ref_image_size": "match"}, "采样"),
    ("ksel", "KSamplerSelect", "采样器", {"sampler_name": "res_multistep"}, "采样"),
    ("sched", "BasicScheduler", "调度器（steps 在这里改）", {
        "scheduler": "simple", "steps": 8}, "采样"),
    ("noise", "RandomNoise", "噪声", {}, "采样"),
    ("guider", "BasicGuider", "引导（conditioning 在这里进）", {}, "采样"),
    ("samp", "SamplerCustomAdvanced", "采样 · 官方五件套", {}, "采样"),
    ("dec_v", "VAEDecode", "解码 · 视频", {}, "采样"),
    ("dec_a", "VAEDecodeAudio", "解码 · 音频", {}, "采样"),

    ("vid", "CreateVideo", "合成视频（24fps）", {"fps": 24}, "输出"),
    ("save", "SaveVideo", "保存单镜", {"filename_prefix": "H3MangaStudio/vlm01"}, "输出"),
    ("prev_vlm", "PreviewAny", "预览 · VLM 解读报告（页型/格数/花费）", {}, "输出"),
    ("prev_txt", "PreviewAny", "预览 · VLM 分镜文本（可贴回 ③）", {}, "输出"),
    ("prev", "PreviewAny", "预览 · 槽位对照表", {}, "输出"),

    # ---- 批量出镜：⑦ 内部逐镜循环 ----
    # ★ ComfyUI 的工作流**没有循环**，所以「一次顺序跑完全部镜头」只能由节点内部循环
    #   实现 —— 这就是 ⑦。它与上面的「单镜」分支并列：单镜用来调参看效果，
    #   批量用来出全片。max_shots=0 表示不限，跑完计划里的每一镜。
    ("render", "H3MangaEpisodeRender", "⑦ 批量出镜 ★｜顺序跑完全部镜头", {
        "steps": 8, "lora_name": "none", "max_shots": 0}, "批量"),
    ("vid2", "CreateVideo", "合成 · 整页（24fps）", {"fps": 24}, "批量"),
    ("save2", "SaveVideo", "保存 · 整页视频", {"filename_prefix": "H3MangaStudio/page"},
     "批量"),
    ("prev_batch", "PreviewAny", "预览 · 批量渲染报告", {}, "批量"),

    ("note", "MarkdownNote", "使用说明（VLM 全自动版）",
     {"text": README_NOTE_VLM}, "说明"),
]

LINKS_VLM = [
    ("clip", 0, "mmh3", "clip"),
    ("vae_v", 0, "mmh3", "vae"),
    ("vae_a", 0, "mmh3", "audio_vae"),

    ("img_pg", 0, "split", "image"),
    ("img_c1", 0, "refs", "character_images.character_0"),
    ("img_c2", 0, "refs", "character_images.character_1"),

    # ★ 预览：把 ① 的三个输出显式接出来（以前全悬空 → 切完看不见）
    ("split", 1, "prev_panels", "images"),
    ("split", 3, "prev_grid", "images"),
    ("split", 2, "prev_report", "source"),
    ("split", 1, "save_panels", "images"),

    # ★ ⑬ 要的是**整页原图**（不是 ① 切出来的格图），所以同一个 LoadImage 分两路；
    #   同时把 ① 的 board 接过去 —— 这样 VLM 用的是 ① 的全局格号，
    #   与 ④ 的绑定口径天然一致（少一次「事后重编号」的出错机会）。
    ("img_pg", 0, "vlmscribe", "image"),
    ("split", 0, "vlmscribe", "board"),
    ("vlmscribe", 0, "vlmplan", "vlm"),

    ("vlmplan", 0, "bind", "plan"),
    ("split", 0, "bind", "board"),
    ("refs", 0, "bind", "refs"),
    ("bind", 0, "comp", "plan"),
    ("refs", 0, "comp", "refs"),

    ("comp", 0, "shot", "plan"),
    ("refs", 0, "shot", "refs"),
    ("split", 0, "shot", "board"),

    # ⑫-A → 官方节点：输出「参考图 N」接 ref_image_(N-1)（★ 官方槽名 0 起）
    ("shot", 0, "mmh3", "prompt"),
    ("shot", 1, "mmh3", "ref_images.ref_image_0"),
    ("shot", 2, "mmh3", "ref_images.ref_image_1"),
    ("shot", 3, "mmh3", "ref_images.ref_image_2"),
    ("shot", 4, "mmh3", "ref_images.ref_image_3"),
    ("shot", 5, "mmh3", "ref_images.ref_image_4"),
    ("shot", 6, "mmh3", "ref_images.ref_image_5"),
    ("shot", 7, "mmh3", "ref_images.ref_image_6"),
    ("shot", 8, "mmh3", "ref_images.ref_image_7"),
    ("shot", 9, "mmh3", "ref_images.ref_image_8"),
    ("shot", 10, "mmh3", "width"),
    ("shot", 11, "mmh3", "height"),
    ("shot", 12, "mmh3", "length"),

    ("mmh3", 0, "guider", "conditioning"),
    ("unet", 0, "lora_fast", "model"),
    ("lora_fast", 0, "lora_alt", "model"),
    ("lora_alt", 0, "guider", "model"),
    ("lora_alt", 0, "sched", "model"),
    ("ksel", 0, "samp", "sampler"),
    ("sched", 0, "samp", "sigmas"),
    ("noise", 0, "samp", "noise"),
    ("guider", 0, "samp", "guider"),
    ("mmh3", 1, "samp", "latent_image"),

    ("samp", 0, "dec_v", "samples"),
    ("vae_v", 0, "dec_v", "vae"),
    ("samp", 0, "dec_a", "samples"),
    ("vae_a", 0, "dec_a", "vae"),
    ("dec_v", 0, "vid", "images"),
    ("dec_a", 0, "vid", "audio"),
    ("vid", 0, "save", "video"),

    ("vlmscribe", 1, "prev_vlm", "source"),
    ("vlmplan", 1, "prev_txt", "source"),
    ("shot", 13, "prev", "source"),

    # ★ 批量分支：⑦ 内部循环 → 一次输出**全部镜头**（与「单镜」并列）
    ("lora_alt", 0, "render", "model"),
    ("clip", 0, "render", "clip"),
    ("vae_v", 0, "render", "vae"),
    ("vae_a", 0, "render", "audio_vae"),
    ("comp", 0, "render", "plan"),
    ("refs", 0, "render", "refs"),
    ("split", 0, "render", "board"),
    ("render", 0, "vid2", "images"),
    ("render", 1, "vid2", "audio"),
    ("vid2", 0, "save2", "video"),
    ("render", 2, "prev_batch", "source"),
]

COLUMNS_VLM = ["模型", "LoRA", "输入图", "素材", "VLM", "预览", "编排", "前置", "采样", "批量", "输出", "说明"]

COL_COLOR_VLM = {
    "模型": "#3f789e",
    "LoRA": "#0f766e",
    "输入图": "#88A",
    "素材": "#a1309b",
    "VLM": "#c05621",
    "预览": "#4d7c0f",
    "编排": "#b58b2a",
    "前置": "#2f8f6f",
    "采样": "#8a4fbf",
    "批量": "#7c2d12",
    "输出": "#b06634",
    "说明": "#666",
}


def _cnr_id(node_type: str) -> str:
    """标注来源，方便前端显示归属（自定义节点 vs 官方）。"""
    if node_type.startswith("H3Manga"):
        return "ComfyUI_H3MangaStudio"
    return "comfy-core"


def build(sparse_override: dict | None = None):
    # sparse_override：按「输入名 → 值」覆盖 ⑦ 整集渲染 的 widget
    # （tools/sparse_ab.py 用它生成「dense vs 稀疏」两份只差一个开关的工作流）
    over_of = {var: dict(over or {}) for var, _t, _ti, over, _c in NODES}
    # ★ 合并 WIDGET_OVERRIDE —— 这些是「漏写就静默取错默认值」的项。
    #   setdefault：显式写在 NODES 里的 over 优先。
    for var, ntype, _ti, _over, _c in NODES:
        for k, v in (WIDGET_OVERRIDE.get(ntype) or {}).items():
            over_of[var].setdefault(k, v)
    if sparse_override:
        over_of.setdefault("render", {}).update(sparse_override)

    # comfy_extras 的 600 个节点（含 CreateVideo / SaveVideo / PreviewAny）
    # 必须先异步加载，否则 NODE_CLASS_MAPPINGS 里只有 65 个基础节点
    asyncio.run(nodes.init_builtin_extra_nodes())
    asyncio.run(nodes.load_custom_node(PKG))

    specs = {var: describe(ntype) for var, ntype, _t, _o, _c in NODES}

    used_slots = {}
    for _s, _si, dvar, dname in LINKS:
        if "." in dname:
            used_slots.setdefault(dvar, []).append(dname)

    # ---- ★ 哪些连线要接到「widget（带 default 的输入）」上 ----
    # width / height / length / prompt 这类带 default 的输入在 ComfyUI 里是 widget。
    # 想让连线接上去，目标节点的 inputs 里必须补一条
    #   {"name":.., "type":.., "link":.., "widget": {"name":..}}
    # —— 这是前端「把 widget 转成 input」的序列化形式。
    # 不做这一步的后果（实测踩过）：links 里线存在、target_slot 却是 null，
    # ComfyUI 直接丢弃 → 界面上「这条线连不上」，官方节点用空 prompt 出片。
    widget_inputs: dict = {}
    for _s, _si, dvar, dname in LINKS:
        _w, _lks, _ag = specs[dvar]
        if dname in {n for n, _ in _lks}:
            continue                                    # 普通 input，已有槽
        if dname.split(".")[0] in _ag:
            continue                                    # Autogrow 槽，已有槽
        if dname in {w[0] for w in _w}:
            lst = widget_inputs.setdefault(dvar, [])
            if dname not in lst:
                lst.append(dname)

    # ---- 尺寸：所有节点同宽（网格成立的前提），高度按 widget / 槽位估算 ----
    size_of = {}
    for var, ntype, _title, _over, _col in NODES:
        widgets, links, autogrow = specs[var]
        h = estimate_height(widgets, links, used_slots.get(var, []))
        size_of[var] = [COL_WIDTH, h]

    # ---- 组内矩阵：每个 group 内按 INNER_COLS 折行 ----
    members_of = {}
    for var, _t, _ti, _o, col in NODES:
        members_of.setdefault(col, []).append(var)

    inner_pos, group_w, group_h = {}, {}, {}
    for col in COLUMNS:
        members = members_of.get(col) or []
        if not members:
            continue
        rows = [members[i:i + INNER_COLS]
                for i in range(0, len(members), INNER_COLS)]
        inner_pos[col] = {}
        y = 0
        for r in rows:
            row_h = max(size_of[v][1] for v in r)
            for ci, v in enumerate(r):
                inner_pos[col][v] = (ci * (COL_WIDTH + INNER_GAP_X), y)
            y += row_h + INNER_GAP_Y
        group_h[col] = y - INNER_GAP_Y
        ncols = max(len(r) for r in rows)
        group_w[col] = COL_WIDTH * ncols + INNER_GAP_X * (ncols - 1)

    # ---- 组间矩阵：按 OUTER_COLS 折行，同列左对齐、同行顶对齐 ----
    # ★ 蛇形（boustrophedon）折行：第 1 行左→右，第 2 行右→左，第 3 行再左→右。
    #   为什么不用「每行都从左开始」：数据流是 模型→输入图→素材→编排→执行→输出，
    #   若每行都左对齐，第 1 行末尾（编排）与第 2 行开头（执行）会落在对角两端，
    #   视觉上读不出流程的连续性。蛇形让「上一行的最后一格」紧邻「下一行的第一格」，
    #   流程像 Z 字一样连贯，一眼就能顺着走完。
    filled = [c for c in COLUMNS if c in group_w]
    outer_rows = [filled[i:i + OUTER_COLS]
                  for i in range(0, len(filled), OUTER_COLS)]
    n_outer = max((len(r) for r in outer_rows), default=0)

    placements = []          # (网格行, 网格列, group 名)
    for ri, r in enumerate(outer_rows):
        if ri % 2 == 0:
            placements += [(ri, ci, c) for ci, c in enumerate(r)]
        else:
            placements += [(ri, n_outer - 1 - k, c) for k, c in enumerate(r)]

    col_extent = [0] * n_outer
    for _ri, ci, c in placements:
        col_extent[ci] = max(col_extent[ci], group_w[c])

    x_at, x = [], 0
    for ci in range(n_outer):
        x_at.append(x)
        x += col_extent[ci] + OUTER_GAP_X

    row_extent = [max(group_h[c] for c in r) for r in outer_rows]
    y_at, y = [], 0
    for ri in range(len(outer_rows)):
        y_at.append(y)
        y += row_extent[ri] + OUTER_GAP_Y

    pos_of, group_origin = {}, {}
    for ri, ci, col in placements:
        gx, gy = x_at[ci], y_at[ri]
        group_origin[col] = (gx, gy)
        for v in members_of[col]:
            rx, ry = inner_pos[col][v]
            pos_of[v] = (gx + rx, gy + ry)

    id_of = {var: i + 1 for i, (var, *_r) in enumerate(NODES)}
    var_of_id = {i: v for v, i in id_of.items()}

    # ---- 连线 ----
    link_records = []
    incoming = {}
    next_link = 1
    for svar, sidx, dvar, dname in LINKS:
        link_records.append({
            "id": next_link, "origin_id": id_of[svar], "origin_slot": sidx,
            "target_id": id_of[dvar], "target_name": dname, "target_slot": None,
            "type": "*",
        })
        incoming[(dvar, dname)] = next_link
        next_link += 1

    # ---- 节点 ----
    graph_nodes = []
    for var, ntype, title, over, col in NODES:
        widgets, links, autogrow = specs[var]

        inputs = []
        for name, t in links:
            inputs.append({"name": name, "type": t, "link": incoming.get((var, name))})
        for aname, info in autogrow.items():
            for full in used_slots.get(var, []):
                if full.startswith(aname + "."):
                    inputs.append({
                        "name": full, "type": info["inner_type"],
                        "link": incoming.get((var, full)), "shape": 7,
                        # ★ label 是 Autogrow 槽的显示名（官方模板里是去掉前缀的槽名）
                        "label": full.split(".", 1)[1],
                    })
        # ★ widget 转 input 的条目一律放在**最后**（与官方模板的顺序一致：
        #   clip/vae/.../ref_images 在前，转出来的 prompt/width/height/length 在后）
        wtype_of = {w[0]: w[3] for w in widgets}
        for wname in widget_inputs.get(var, []):
            inputs.append({
                "name": wname, "type": wtype_of.get(wname, "*"),
                "link": incoming.get((var, wname)),
                "widget": {"name": wname},
            })

        wv = []
        for w in widgets:
            wname, wdefault = w[0], w[1]
            if wname in over_of.get(var, {}):
                wv.append(over_of[var][wname])
            elif wname == "control_after_generate":
                wv.append("randomize")
            else:
                wv.append(wdefault)

        for i, inp in enumerate(inputs):
            if inp["link"] is not None:
                for rec in link_records:
                    if rec["id"] == inp["link"]:
                        rec["target_slot"] = i

        cls = nodes.NODE_CLASS_MAPPINGS.get(ntype)
        rtypes = list(getattr(cls, "RETURN_TYPES", []) or []) if cls else []
        rnames = list(getattr(cls, "RETURN_NAMES", []) or []) if cls else []
        outputs = []
        for i, t in enumerate(rtypes):
            # ★ 不写 slot_index：官方模板的 outputs 里没有它（前端会自己算），
            #   多写的字段一旦与实际不符反而会误导前端。
            outputs.append({
                "name": rnames[i] if i < len(rnames) else t,
                "type": t, "links": [],
            })

        node = {
            "id": id_of[var], "type": ntype, "title": title,
            "pos": list(pos_of[var]), "size": list(size_of[var]),
            "flags": {}, "order": 0,
            # ★ mode=4 是 ComfyUI 的「**旁路（bypass）**」：节点不执行，输入直通输出。
            #   为什么需要它：核心的 LoraLoaderModelOnly 的 lora_name 下拉里**没有
            #   "none" 选项**（只有 loras 目录里的文件名），所以「关闭这个 LoRA 槽」
            #   在 ComfyUI 里的标准做法就是旁路 —— 而不是填一个假名字。
            "mode": 4 if var in BYPASS_VARS else 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": ntype, "cnr_id": _cnr_id(ntype)},
            "_var": var, "_col": col,
        }
        if widgets:
            node["widgets_values"] = wv
            # ★ widgets_values_named —— 当前前端（官方模板 export 的 frontendVersion
            #   是 1.49.6）导出的**标配字段**：按名字记录 widget 值，加载时优先用它
            #   恢复。这正是「widget 被转成 input 之后位置会错位」的官方解法。
            #   官方模板里凡有 widgets_values 的节点都同时带它（25/29），
            #   缺失会与官方导出格式产生实质差异 → 前端只能退化成按位置恢复。
            node["widgets_values_named"] = {w[0]: wv[i] for i, w in enumerate(widgets)}
        graph_nodes.append(node)

    # ---- 输出 links 回填 + 连线类型 ----
    node_of_id = {n["id"]: n for n in graph_nodes}
    type_of_out = {}
    for n in graph_nodes:
        for i, o in enumerate(n["outputs"]):
            type_of_out[(n["_var"], i)] = o["type"]

    for rec in link_records:
        rec["type"] = type_of_out.get(
            (var_of_id[rec["origin_id"]], rec["origin_slot"]), "*")
        node_of_id[rec["origin_id"]]["outputs"][rec["origin_slot"]]["links"].append(rec["id"])

    # ---- 拓扑排序 ----
    indeg = {n["id"]: 0 for n in graph_nodes}
    adj = {n["id"]: [] for n in graph_nodes}
    for rec in link_records:
        adj[rec["origin_id"]].append(rec["target_id"])
        indeg[rec["target_id"]] += 1
    queue = sorted(i for i, d in indeg.items() if d == 0)
    order = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in sorted(adj[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    for i, nid in enumerate(order):
        node_of_id[nid]["order"] = i

    # ---- 分组：每个 group 的框 = 它在组间矩阵里占的单元（含标题带与留白） ----
    groups = []
    for gi, col in enumerate(filled, start=1):
        gx, gy = group_origin[col]
        members = members_of[col]
        # 用实际节点包围盒兜底，避免估算高度与实际渲染高度有出入时框吃亏
        bx0 = min(pos_of[v][0] for v in members) - GROUP_PAD_X
        by0 = min(pos_of[v][1] for v in members) - GROUP_PAD_TOP
        bx1 = max(pos_of[v][0] + size_of[v][0] for v in members) + GROUP_PAD_X
        by1 = max(pos_of[v][1] + size_of[v][1] for v in members) + GROUP_PAD_BOTTOM
        groups.append({
            "id": gi, "title": col,
            "bounding": [bx0, by0, bx1 - bx0, by1 - by0],
            "color": COL_COLOR.get(col, "#88A"), "font_size": 26, "flags": {},
        })

    for n in graph_nodes:
        n.pop("_var", None)
        n.pop("_col", None)

    # ---- 画布视野：让整张矩阵图一屏可见（从左上空隙起算） ----
    if graph_nodes:
        span_x0 = min(n["pos"][0] for n in graph_nodes) - 40
        span_y0 = min(n["pos"][1] for n in graph_nodes) - 70
    else:
        span_x0 = span_y0 = 0
    ds_offset = [int(-span_x0 * CANVAS_SCALE + 60),
                 int(-span_y0 * CANVAS_SCALE + 60)]

    return {
        "id": "h3-manga-studio-example",
        "revision": 0,
        "last_node_id": max(id_of.values()),
        "last_link_id": next_link - 1,
        "nodes": graph_nodes,
        "links": [[r["id"], r["origin_id"], r["origin_slot"], r["target_id"],
                   r["target_slot"], r["type"]] for r in link_records],
        "groups": groups,
        "config": {},
        "extra": {
            "ds": {"scale": CANVAS_SCALE, "offset": ds_offset},
            "note": "由 ComfyUI_H3MangaStudio/tools/make_example_workflow.py 生成；"
                    "用 tools/verify_workflow.py 校验",
        },
        "version": 0.4,
    }


if __name__ == "__main__":
    # build() 读的是**模块级全局** NODES / LINKS / COLUMNS，
    # 所以「换一份图定义」= 改这几个全局再调用一次。
    JOBS = [
        (NODES, LINKS, COLUMNS, COL_COLOR,
         "H3MangaStudio_example_workflow.json",
         "整集渲染版（⑦ 一键跑完，max_shots=0）"),
        (NODES_OFFICIAL, LINKS_OFFICIAL, COLUMNS_OFFICIAL, COL_COLOR_OFFICIAL,
         "H3MangaStudio_official_nodes_workflow.json",
         "官方节点接线版（⑫-A → 官方多参节点 → 你自己的采样链）"),
        (NODES_VLM, LINKS_VLM, COLUMNS_VLM, COL_COLOR_VLM,
         "H3MangaStudio_vlm_auto_workflow.json",
         "VLM 全自动版（漫画图 → ⑬ 读图 → ⑭ 出分镜 → 官方多参节点）"),
    ]
    for nodes_spec, links_spec, cols_spec, col_color, fname, label in JOBS:
        globals()["NODES"] = nodes_spec
        globals()["LINKS"] = links_spec
        globals()["COLUMNS"] = cols_spec
        globals()["COL_COLOR"] = col_color
        wf = build()
        out = os.path.join(PKG, "templates", fname)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=1)
        print(f"已生成 {out}")
        print(f"  —— {label}")
        print(f"  节点 {len(wf['nodes'])} / 连线 {len(wf['links'])} / 分组 {len(wf['groups'])}")
        for n in wf["nodes"]:
            # ★ 用 .get：没有 widget 的节点现在**不写** widgets_values（与官方导出一致），
            #   硬下标会 KeyError —— 一旦抛出，后面那个工作流根本生成不出来。
            nw = len(n.get("widgets_values") or [])
            nn = len(n.get("widgets_values_named") or {})
            print(f"  #{n['id']:>3} {n['type']:<28} widgets={nw} named={nn} "
                  f"inputs={len(n['inputs'])}  「{n['title']}」")
        print()
