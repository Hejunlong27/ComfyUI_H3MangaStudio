"""H3 Manga Studio —— 自定义数据类型与核心数据结构。

设计边界（重要）
----------------
本模块**只定义数据结构与常量**，不 import comfy / torch / cv2。
所有「多参数怎么组织」的规则都集中在这里，节点层与前端都从这里取唯一真源。

关于 MiniMax H3 的参考图契约（来自 ComfyUI 官方 comfy_extras/nodes_minimax_h3.py 实测）
------------------------------------------------------------------------------------
官方节点 ``MiniMaxH3ReferenceToVideo`` 的参考素材进入「呈现序列」的顺序是**固定**的：

    图片(<Picture 1..N>)  →  视频(<Video 1..M>，每个视频的音轨作为 <Audio j> 排在它之前)
                          →  独立音频(<Audio j>)

序号是 **按类型各数、1-based**：``<Picture i>`` / ``<Video k>`` / ``<Audio j>``。
提示词里用这三类标签指代素材，写错序号 = 角色串号，而且**不会报错**。

本插件的全部价值，就是把这套「序号契约」自动化：
用户按**角色名**写分镜，插件负责拍平成有序槽位、算出序号、把提示词里的
``@角色id`` 替换成正确的 ``<Picture i>``。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# ComfyUI 自定义类型 ID
# 命名与已装插件刻意区分：本包统一用 H3_MANGA_* / H3Manga* 前缀，
# 避免与 ComfyUI_MiniMaxH3_Director 的 MiniMaxH3Director* 互相顶替。
# --------------------------------------------------------------------------
TYPE_REFS = "H3_MANGA_REFS"
TYPE_PLAN = "H3_MANGA_PLAN"
TYPE_BOARD = "H3_MANGA_BOARD"

# --------------------------------------------------------------------------
# H3 模型侧硬约束（来自 comfy_extras/nodes_minimax_h3.py 源码常量）
# --------------------------------------------------------------------------
H3_FPS = 24
H3_CANVAS_MULTIPLE = 32          # 分辨率须为 32 的倍数
H3_MAX_REF_IMAGES = 9            # ref_images autogrow 上限
H3_MAX_REF_VIDEOS = 3
H3_MAX_REF_AUDIOS = 3
H3_REF_IMAGE_SHORT_EDGE = 2048   # ref_image_size="max" 时的短边
H3_MAX_PIXELS = 768 * 1344       # 单帧像素上限
H3_DEFAULT_WIDTH = 1344
H3_DEFAULT_HEIGHT = 768
H3_DEFAULT_LENGTH = 124          # ≈5s
H3_MAX_LENGTH = 362              # 官方训练区间上限（~15s）

# H3 视频 VAE 的空间压缩比（comfy/ldm/minimax/model.py：`height = video.shape[3] * 16`）
H3_LATENT_SPATIAL = 16
# 每 5 个 latent 帧覆盖 17 个像素帧（FRAME_PER_TOKEN = (1,4,4,4,4)，和 = 17）
H3_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

# 提示词标签：与官方一致，改这里就是改契约
TAG_PICTURE = "<Picture {}>"
TAG_VIDEO = "<Video {}>"
TAG_AUDIO = "<Audio {}>"

# 参考槽位在 autogrow 里的键名前缀（官方实测：ref_images.ref_image_0 .. _8）
REF_IMAGE_KEY = "ref_image_{}"
REF_VIDEO_KEY = "ref_video_{}"
REF_VIDEO_AUDIO_KEY = "ref_video_audio_{}"
REF_AUDIO_KEY = "ref_audio_{}"


def align_frame_count(length: int) -> int:
    """把帧数对齐到 H3 的 17n+5 网格（官方 align_frame_count 的纯函数版）。

    与官方实现逐字等价：向上找第一个满足 n % 17 == 5 的值。
    5 秒 ≈ 121→124 帧；本函数在插件里做「提交前预检」，
    避免把不对齐的 length 喂进节点后才在采样阶段炸掉。
    """
    n = max(5, int(length))
    while n % 17 != 5:
        n += 1
    return n


def frames_for_seconds(seconds: float) -> int:
    """秒 → 对齐后的帧数。等价官方模板的 ComfyMathExpression：
    max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17
    """
    raw = max(5, int(round(float(seconds) * H3_FPS)))
    return raw + (5 - (raw % 17)) % 17


def seconds_for_frames(length: int) -> float:
    return align_frame_count(length) / float(H3_FPS)


def latent_frames_for(length: int) -> int:
    """像素帧数 → 视频 latent 帧数。

    H3 的 latent 时间轴不是均匀的：``FRAME_PER_TOKEN = (1,4,4,4,4)``，
    第 k 个 latent 帧覆盖的像素帧数是 ``H3_FRAME_PER_TOKEN[k % 5]``
    （官方 ``comfy/ldm/minimax/model.py::_video_t_spans``）。

    所以这里直接按定义累加求解 —— 帧数上限 3600，循环代价可忽略。
    """
    target = int(length)
    if target < 5:
        return 0
    covered = 0
    for k in range(target):          # 上界足够：每帧至少贡献 1
        covered += H3_FRAME_PER_TOKEN[k % 5]
        if covered >= target:
            return k + 1
    return 0


def estimate_video_tokens(width: int, height: int, length: int) -> int:
    """估算 H3 单个镜头在 packed 序列里占用的**视频行** token 数。

    口径来自 ``comfy_extras/nodes_sparse_attention.py::vsa_plan``::

        grid = (latent_t, latent_h // 2, latent_w // 2)

    其中 ``latent_h = height // 16``、``latent_w = width // 16``
    （H3 视频 VAE 是 16× 空间压缩），DiT 用 2×2 patch，
    所以每个 latent 帧贡献 ``(latent_h//2) × (latent_w//2)`` 行。

    用途：判断稀疏注意力会不会落到官方 ``min_tokens``（默认 12288）阈值以下。
    注意这是**下界** —— 真实 packed 序列还要加上文本 / 参考图 / 音频行。
    """
    lh = max(1, int(height) // H3_LATENT_SPATIAL)
    lw = max(1, int(width) // H3_LATENT_SPATIAL)
    lt = latent_frames_for(int(length))
    return int(lt) * max(1, lh // 2) * max(1, lw // 2)


def align_canvas(width: int, height: int) -> "tuple[int, int]":
    """分辨率对齐到 32 的倍数，并夹在像素上限内（等比缩放）。"""
    import math

    w = max(H3_CANVAS_MULTIPLE, int(round(int(width) / H3_CANVAS_MULTIPLE)) * H3_CANVAS_MULTIPLE)
    h = max(H3_CANVAS_MULTIPLE, int(round(int(height) / H3_CANVAS_MULTIPLE)) * H3_CANVAS_MULTIPLE)
    if w * h > H3_MAX_PIXELS:
        s = math.sqrt(H3_MAX_PIXELS / float(w * h))
        w = max(H3_CANVAS_MULTIPLE, int(round(w * s / H3_CANVAS_MULTIPLE)) * H3_CANVAS_MULTIPLE)
        h = max(H3_CANVAS_MULTIPLE, int(round(h * s / H3_CANVAS_MULTIPLE)) * H3_CANVAS_MULTIPLE)
    return w, h


# --------------------------------------------------------------------------
# 参考素材包（多参数组织的中枢）
# --------------------------------------------------------------------------
@dataclass
class RefEntry:
    """一个参考条目。角色 / 风格 / 视频 / 音频共用同一结构，靠 kind 区分。"""

    id: str                       # 稳定标识，分镜里用 @id 引用
    name: str = ""                # 显示名（中文可）
    kind: str = "character"       # character | style | prop | video | audio
    images: List[Any] = field(default_factory=list)   # torch.Tensor [H,W,C] 列表
    audio: Any = None             # AUDIO dict（{"waveform","sample_rate"}）
    videos: List[Any] = field(default_factory=list)   # 视频帧张量 [T,H,W,C]（T 为帧数）
    video_audio: Any = None       # 与视频配套的音轨（kind=="video" 时用，≤3）
    lock: bool = True             # 是否参与「一致性锁定」（只影响提示词措辞）
    priority: int = 0             # 拍平顺序（小的排前面）
    describe: str = ""            # 角色外观描述，用于提示词锁定
    slot_pictures: List[int] = field(default_factory=list)  # 计算后回填：占用的 <Picture i>
    slot_videos: List[int] = field(default_factory=list)
    slot_audios: List[int] = field(default_factory=list)


@dataclass
class RefPack:
    """``H3_MANGA_REFS`` 的载体。节点间靠它传递全部参考素材 + 序号映射表。"""

    entries: List[RefEntry] = field(default_factory=list)
    max_ref_images: int = H3_MAX_REF_IMAGES
    source_note: str = ""
    # 分镜格锚点在映射表里的 id（提示词里可用 @panel 引用它）
    panel_id: str = "panel"

    # ---- 拍平与序号计算 ------------------------------------------------
    def ordered(self) -> List[RefEntry]:
        """拍平顺序：先按 kind 分组（character → style → prop → video → audio），
        组内按 priority 升序、再按声明顺序稳定排序。
        这个顺序**必须**与官方 tokenizer 的呈现顺序一致，否则 <Picture i> 全错。
        """
        kind_rank = {"character": 0, "style": 1, "prop": 2, "video": 3, "audio": 4}
        return sorted(
            self.entries,
            key=lambda e: (kind_rank.get(e.kind, 9), e.priority),
        )

    def resolve_slots(self) -> Dict[str, Any]:
        """把 entries 拍平成官方槽位池，并给出**全局展示用**的序号。

        ⚠️ 重要：这里的序号**只用于展示与审计**。
        官方节点里 ``<Picture i>`` 的真实序号来自 ``ref_items`` 的**追加顺序**
        （源码：``for img in (ref_images or {}).values(): ref_items.append(...)``），
        也就是「本镜实际传进去的那几张图的顺序」，与键名无关。

        所以真正喂给模型时必须用 ``refs.shot_ref_layout()`` 做**密集重排**，
        每个镜头按「分镜格锚点 → 本镜出场角色 → 风格图」重新从 1 开始编号。
        全局序号只用来回答「这个角色一共有几张图、有没有溢出」。

        返回：
            {
              "ref_images": {...},          # 全量槽位池（顺序 = 全局展示序）
              "ref_videos": {...},          # 视频参考池（键 ref_video_0..2）
              "ref_video_audios": {...},    # 视频配套音轨（键与视频同号）
              "ref_audios": {...},          # 独立音频池（键 ref_audio_0..2）
              "picture_pool": {id: [tensor,...]},   # 角色/风格 → 它的图（权威数据）
              "video_pool":   {id: [frames,...]},
              "audio_pool":   {id: [audio,...]},
              "picture_map": {id: [全局序号]},      # 仅展示
              "video_map":   {id: [全局序号]},
              "audio_map":   {id: [全局序号]},
              "flat": [...], "warnings": [...], "overflow": [...],
            }
        """
        ref_images: Dict[str, Any] = {}
        ref_videos: Dict[str, Any] = {}
        ref_video_audios: Dict[str, Any] = {}
        ref_audios: Dict[str, Any] = {}
        picture_pool: Dict[str, List[Any]] = {}
        audio_pool: Dict[str, List[Any]] = {}
        video_pool: Dict[str, List[Any]] = {}
        picture_map: Dict[str, List[int]] = {}
        audio_map: Dict[str, List[int]] = {}
        video_map: Dict[str, List[int]] = {}
        flat: List[Dict[str, Any]] = []
        warnings: List[str] = []
        overflow: List[Dict[str, Any]] = []

        # 上限在循环外算一次（早先在循环体内反复求值）
        img_limit = min(int(self.max_ref_images or H3_MAX_REF_IMAGES), H3_MAX_REF_IMAGES)
        img_slot = 0
        vid_slot = 0
        aud_slot = 0

        for entry in self.ordered():
            # ---- 视频参考（占 <Video k>，其配套音轨占同号的 <Audio j>）----
            # 官方源码按「键名尾号」配对：ref_video_audio_N ↔ ref_video_N，
            # 所以配套音轨必须用**同一个 vid_slot**，不能另起计数。
            for vf in entry.videos:
                if vid_slot >= H3_MAX_REF_VIDEOS:
                    overflow.append({"id": entry.id, "dropped_videos": 1})
                    warnings.append(
                        f"参考视频超过 {H3_MAX_REF_VIDEOS} 条上限，已丢弃 {entry.id} 的一条"
                    )
                    continue
                ref_videos[REF_VIDEO_KEY.format(vid_slot)] = vf
                if entry.video_audio is not None:
                    ref_video_audios[REF_VIDEO_AUDIO_KEY.format(vid_slot)] = entry.video_audio
                vid_slot += 1
                video_pool.setdefault(entry.id, []).append(vf)
                video_map.setdefault(entry.id, []).append(vid_slot)

            # ---- 图片参考（角色图 / 风格图 / 道具图，共用 <Picture i> 序号池）----
            for img in entry.images:
                if img_slot >= img_limit:
                    overflow.append({"id": entry.id, "dropped_images": 1})
                    warnings.append(
                        f"参考图超过 {img_limit} 张上限，"
                        f"已丢弃 {entry.id} 的图片（角色一致性可能下降）"
                    )
                    continue
                ref_images[REF_IMAGE_KEY.format(img_slot)] = img
                img_slot += 1
                picture_pool.setdefault(entry.id, []).append(img)
                picture_map.setdefault(entry.id, []).append(img_slot)
                flat.append({
                    "slot": img_slot - 1,
                    "tag": TAG_PICTURE.format(img_slot),
                    "id": entry.id,
                    "name": entry.name or entry.id,
                    "kind": entry.kind,
                })

            # ---- 独立音频 ----
            if entry.audio is not None and entry.kind == "audio":
                if aud_slot >= H3_MAX_REF_AUDIOS:
                    warnings.append(f"参考音频超过 {H3_MAX_REF_AUDIOS} 条上限，已丢弃 {entry.id}")
                    continue
                ref_audios[REF_AUDIO_KEY.format(aud_slot)] = entry.audio
                aud_slot += 1
                audio_pool.setdefault(entry.id, []).append(entry.audio)
                audio_map.setdefault(entry.id, []).append(aud_slot)

        # 回填到 entries，方便下游直接读
        for entry in self.entries:
            entry.slot_pictures = picture_map.get(entry.id, [])
            entry.slot_videos = video_map.get(entry.id, [])
            entry.slot_audios = audio_map.get(entry.id, [])

        return {
            "ref_images": ref_images,
            "ref_videos": ref_videos,
            "ref_video_audios": ref_video_audios,
            "ref_audios": ref_audios,
            "picture_pool": picture_pool,
            "video_pool": video_pool,
            "audio_pool": audio_pool,
            "picture_map": picture_map,
            "video_map": video_map,
            "audio_map": audio_map,
            "flat": flat,
            "warnings": warnings,
            "overflow": overflow,
        }


    def describe_table(self) -> str:
        """人类可读的参考包全量清单（节点输出 + 前端面板用）。

        ⚠️ 这里的序号是**全局池序号，仅供核对「谁有几张图」**。
        真正喂给模型的 ``<Picture i>`` 是**逐镜重排**的，
        以「提示词装配」节点输出的预览为准。
        """
        res = self.resolve_slots()
        lines = [
            "参考包全量清单",
            "=" * 52,
            "（下表的序号是全局池编号，仅用于核对素材是否齐全；",
            "  实际提示词里的 <Picture i> 是逐镜重排的，见提示词预览）",
            "-" * 52,
        ]
        if not res["flat"]:
            lines.append("（无参考图）")
        for item in res["flat"]:
            lines.append(f"  {item['tag']:<12} ← {item['kind']:<9} {item['id']}  ({item['name']})")
        if res["video_map"]:
            lines.append("-" * 52)
            for eid, slots in res["video_map"].items():
                for s in slots:
                    lines.append(f"  {TAG_VIDEO.format(s):<12} ← video     {eid}")
            if res["ref_video_audios"]:
                lines.append(f"  （含 {len(res['ref_video_audios'])} 条配套音轨，随 <Video k> 同号）")
        if res["audio_map"]:
            lines.append("-" * 52)
            for eid, slots in res["audio_map"].items():
                for s in slots:
                    lines.append(f"  {TAG_AUDIO.format(s):<12} ← audio     {eid}")
        if res["warnings"]:
            lines.append("-" * 52)
            lines.append("警告：")
            for w in res["warnings"]:
                lines.append(f"  ! {w}")
        if self.source_note:
            lines.append("-" * 52)
            lines.append(f"注意：{self.source_note}")
        return "\n".join(lines)

    def to_meta(self) -> Dict[str, Any]:
        """不含张量的元信息快照（可 JSON 序列化，用于报告与前端面板）。"""
        return {
            "entries": [
                {
                    "id": e.id, "name": e.name, "kind": e.kind,
                    "images": len(e.images), "videos": len(e.videos),
                    "has_audio": e.audio is not None,
                    "lock": e.lock, "priority": e.priority,
                    "describe": e.describe,
                }
                for e in self.entries
            ],
            "max_ref_images": self.max_ref_images,
        }


# --------------------------------------------------------------------------
# 分镜格（漫画页切分产物）
# --------------------------------------------------------------------------
@dataclass
class Panel:
    index: int                    # 阅读顺序序号（0-based）
    box: List[int]                # [x, y, w, h] 像素
    page_index: int = 0           # 来自第几页
    page_name: str = ""
    area_ratio: float = 0.0       # 占整页面积比
    order_hint: str = ""          # "rtl" / "ltr" / "manual"
    poly: Optional[List[List[int]]] = None
    """真实形状的顶点 ``[[x, y], ...]``；``None`` = 普通矩形（用 box 裁）。

    斜切分镜（格子是斜的四边形）**用矩形表达不了**：一个斜格的 bbox 必然与
    相邻格的 bbox 相交，裁哪边都会带进邻居的内容。所以这类格子记下顶点，
    裁剪时把多边形以外填成纸白（见 ``panel_split.crop_polygons``）。
    box 始终是它的 bbox，供阅读顺序 / 报告 / 旧代码使用。
    """


@dataclass
class Board:
    """``H3_MANGA_BOARD`` 的载体。"""

    panels: List[Panel] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)   # 与 panels 同序的张量
    pages: int = 0
    source: str = ""

    def __len__(self) -> int:
        return len(self.panels)

    def report(self) -> str:
        lines = [f"分镜格切分报告：{len(self.panels)} 格 / {self.pages} 页", "=" * 52]
        for p in self.panels:
            x, y, w, h = p.box
            shape = f"  形状={len(p.poly)}边形" if p.poly else ""
            lines.append(
                f"  #{p.index:02d}  page{p.page_index}  "
                f"box=({x},{y},{w},{h})  area={p.area_ratio * 100:.1f}%" + shape
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 镜头与整集计划
# --------------------------------------------------------------------------
@dataclass
class Shot:
    """单个镜头。字段设计对齐「漫改动画」的真实制作要素。"""

    id: str
    index: int = 0
    panel_index: Optional[int] = None     # 绑定到第几格（board.panels 的 index）
    duration: float = 5.0                 # 秒
    action: str = ""                      # 画面动作描述（中文可）
    dialogue: str = ""                    # 台词
    speaker: str = ""                     # 说话人（角色 id）
    camera: str = "static"                # 运镜预设名
    characters: List[str] = field(default_factory=list)   # 出场角色 id（决定参考图槽位）
    style: str = ""                       # 覆盖全局风格 id
    size: str = ""                        # "1344x768"，空则用计划默认
    seed: Optional[int] = None            # 空则由 base_seed 派生
    negative: str = ""                    # 单镜负向补充
    extra: str = ""                       # 追加提示词（自由文本）
    # ---- 运行时解析结果 ----
    prompt: str = ""                      # 装配后的最终提示词
    width: int = H3_DEFAULT_WIDTH
    height: int = H3_DEFAULT_HEIGHT
    length: int = H3_DEFAULT_LENGTH
    resolved_seed: int = 0
    picture_tags: Dict[str, str] = field(default_factory=dict)  # 角色 id → "<Picture 1>"
    ref_image_keys: List[str] = field(default_factory=list)     # 本镜实际用到的槽位键
    board_image_index: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def meta(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Plan:
    """``H3_MANGA_PLAN`` 的载体：一整集的可执行分镜计划。"""

    episode: str = "untitled"
    shots: List[Shot] = field(default_factory=list)
    style: str = "manga_anime"
    style_prompt: str = ""
    negative: str = ""
    default_size: str = f"{H3_DEFAULT_WIDTH}x{H3_DEFAULT_HEIGHT}"
    fps: int = H3_FPS
    language: str = "zh"
    global_prompt: str = ""
    source_note: str = ""
    warnings: List[str] = field(default_factory=list)

    def total_seconds(self) -> float:
        return sum(seconds_for_frames(s.length) for s in self.shots)

    def total_frames(self) -> int:
        return sum(s.length for s in self.shots)

    def clone(self) -> "Plan":
        return copy.deepcopy(self)

    def summary(self) -> str:
        lines = [
            f"整集计划：{self.episode}",
            "=" * 52,
            f"  镜头数     : {len(self.shots)}",
            f"  总时长     : {self.total_seconds():.1f} s（{self.total_frames()} 帧 @ {self.fps}fps）",
            f"  默认画幅   : {self.default_size}",
            f"  全局风格   : {self.style}",
        ]
        if self.warnings:
            lines.append("-" * 52)
            lines.append("警告：")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "style": self.style,
            "default_size": self.default_size,
            "fps": self.fps,
            "language": self.language,
            "total_seconds": round(self.total_seconds(), 2),
            "total_frames": self.total_frames(),
            "warnings": list(self.warnings),
            "shots": [s.meta() for s in self.shots],
        }


def estimate_cost(plan: Plan, steps: int) -> Dict[str, Any]:
    """粗算成本：帧数 × 步数 的相对算力单位。

    不做绝对时间承诺（跟显卡强相关），只给**相对量级**，
    让用户在跑整集之前知道「这一集比上一集重多少倍」。
    """
    units = 0
    per_shot = []
    for s in plan.shots:
        w, h = align_canvas(s.width, s.height)
        u = (s.length / 124.0) * (w * h) / float(H3_DEFAULT_WIDTH * H3_DEFAULT_HEIGHT) * max(1, steps)
        units += u
        per_shot.append({"id": s.id, "units": round(u, 2), "frames": s.length})
    return {
        "total_units": round(units, 2),
        "steps": steps,
        "per_shot": per_shot,
        "note": "相对算力单位（以 1344x768 / 124帧 / 该步数为 1.0 基准），仅用于横向比较",
    }
