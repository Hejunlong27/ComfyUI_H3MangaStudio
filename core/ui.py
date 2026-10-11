"""中文文案唯一真源。

三处文案集中在这里，避免「改一处忘两处」：
  1. 输入级 display_name（三重保险之一）
  2. locales/zh/nodeDefs.json（由 scripts/gen_i18n.py 生成）
  3. js/h3manga_labels.js（由同一脚本生成，前端 widget.label 兜底）

**输入名一律 ASCII** —— 中文只出现在 label 里。
把输入名写成中文会污染 API 格式的键名，发布后极难改回。
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 节点级
# --------------------------------------------------------------------------
NODE_LABELS = {
    "H3MangaPanelSplitter": "H3 漫画页切分镜格",
    "H3MangaRefPack": "H3 参考包（角色/风格/音频）",
    "H3MangaScriptLoader": "H3 分镜清单加载",
    "H3MangaShotBinder": "H3 分镜绑定",
    "H3MangaPromptComposer": "H3 提示词装配",
    "H3MangaShotConditioning": "H3 单镜条件（调试用）",
    "H3MangaEpisodeRender": "H3 整集渲染 ★",
    "H3MangaConsistencyAudit": "H3 一致性审计",
    "H3MangaPlanPreview": "H3 计划预览",
    "H3MangaFreeMemory": "H3 显存/内存释放",
    "H3MangaShotOut": "⑫-A 镜头出栈（喂官方多参节点）",
    "H3MangaShotCond": "⑫-B 单镜条件（直出）",
    "H3MangaVLMScribe": "⑬ VLM 漫画解读 ★",
    "H3MangaVLMPlan": "⑭ VLM 转分镜计划",
}

NODE_DESCRIPTIONS = {
    "H3MangaPanelSplitter": "把整页漫画按装订线递归切成分镜格，支持手动框覆盖。输出格图批次 + 结构化 board。",
    "H3MangaRefPack": "把「角色定妆图 / 风格基准图 / 参考音频」打包成有序参考包，并算出 <Picture i> 序号映射表。",
    "H3MangaScriptLoader": "从 JSON 文件或手写块文本加载分镜清单，自动做 17n+5 帧对齐与 32 倍数画幅对齐。",
    "H3MangaShotBinder": "把分镜计划与分镜格图绑定（按 panel 序号 / 按顺序 / 不绑），并做绑定合法性检查。",
    "H3MangaPromptComposer": "把分镜字段编译成 H3 提示词：@角色id → <Picture i>、运镜展开、风格注入、台词包裹。",
    "H3MangaShotConditioning": "只构建单个镜头的 conditioning + latent，方便接官方采样五件套做单镜调试。",
    "H3MangaEpisodeRender": "整集渲染：内部逐镜循环（官方 R2V + 官方采样五件套），输出整集帧序列 + 音频 + 渲染报告。",
    "H3MangaConsistencyAudit": "审计角色引用缺失、参考图过少、序号溢出、槽位漂移等一致性问题，输出分层报告。",
    "H3MangaPlanPreview": "只读预览：显示每镜的最终提示词、参考序号映射与警告，排查串号最快的手段。",
    "H3MangaFreeMemory": "把模型从显存与内存里卸载，输出释放前后用量。本机 33.5GB 权重对 16GB 内存，串行加载才能跑通。",
    "H3MangaShotOut": "★ 官方多参节点的前置接口：出 prompt + 9 个参考图槽（按序号）+ 宽高帧数 + 槽位对照表。采样/放大/解码全在你的图里。",
    "H3MangaShotCond": "省事版前置：内部调用官方多参节点，直接出 positive + latent，少接 9 根参考图线。",
    "H3MangaVLMScribe": "把漫画页交给视觉语言模型，逐格读出画面内容/动作/运镜/角色/台词/一格几镜/页型 —— 补上链路里唯一还靠人肉的一环。",
    "H3MangaVLMPlan": "把 VLM 结果转成分镜计划。纯本地计算、不花钱，可反复调画幅/时长/镜数。",
}

# --------------------------------------------------------------------------
# 输入级（key = 输入名）
# --------------------------------------------------------------------------
INPUT_LABELS = {
    # 通用
    "image": "输入图像",
    "mode": "模式",
    "source": "来源",
    "file_path": "文件路径",
    "inline_text": "内联文本",
    "template": "提示词模板",
    "negative": "负向提示词",
    "base_seed": "基准种子",
    "steps": "采样步数",
    "strict": "严格模式",
    "text": "文本",
    "plan": "分镜计划",
    "refs": "参考包",
    "board": "分镜格",
    "shot_index": "镜头序号",
    "max_shots": "最多渲染镜数",
    "lora_name": "LoRA",
    "lora_strength": "LoRA 权重",
    "sampler_name": "采样器",
    "scheduler": "调度器",
    "shift_video": "视频 shift",
    "shift_audio": "音频 shift",
    "ref_image_size": "参考图尺寸策略",
    "clip": "文本编码器",
    "vae": "视频 VAE",
    "audio_vae": "音频 VAE",
    "model": "模型",
    # 切格
    "ink_threshold": "墨迹阈值",
    "gutter_ratio": "装订线空白阈值",
    "min_panel_px": "最小格边长(px)",
    "min_gutter_px": "最小装订线宽(px)",
    "min_panel_ratio": "最小格面积占比",
    "reading_order": "阅读顺序",
    "padding_px": "裁切外扩(px)",
    "manual_boxes": "手动框",
    "merge_tolerance": "碎片合并阈值",
    "panel_scale": "切格缩放比",
    "frame": "切格框架",
    "white_threshold": "纸白阈值",
    "component_min_area": "最小块面积占比",
    "component_close": "闭运算迭代数",
    "component_open": "开运算半径(0=自动)",
    "component_auto_open": "自动调开运算",
    "deoverlap": "自动分离重叠框",
    # 参考包
    "character_names": "角色名（逗号分隔，与图槽同序）",
    "character_images": "角色定妆图",
    "character_describes": "角色外观锁定描述（每行一个）",
    "style_name": "风格名",
    "style_images": "风格基准图",
    "audio_names": "音频名",
    "audios": "参考音频",
    "video_names": "视频名",
    "videos": "参考视频",
    "max_ref_images": "参考图上限",
    "name_source": "id 生成方式",
    "one_image_per_character": "一角色一图（默认开）",
    # 绑定
    "bind_mode": "绑定方式",
    "panel_offset": "格序号偏移",
    "allow_reuse": "允许重复用格",
    # 提示词
    "style_prefix": "风格前缀",
    "style_suffix": "风格后缀",
    "camera_preset": "默认运镜",
    "dialogue_mode": "台词模式",
    "lock_consistency": "锁定角色一致性",
    "include_describe": "拼入外观描述",
    # 审计 / 预览
    "skip_done_ids": "跳过镜头 id",
    "on_error": "单镜失败处理",
    "negative_mode": "负向词处理",
    "video_names": "视频名",
    "videos": "参考视频",
    "video_audios": "视频配套音轨",
    "camera_json": "运镜表 JSON",
    "style_json": "风格表 JSON",
    # 资源释放
    "trigger": "上游触发（仅顺序用）",
    "unload_all": "全部卸载",
    "run_gc": "额外 GC",
    "label": "阶段标注",
    # 整集渲染
    "unload_stages": "分阶段卸载模型",
    "audio_crossfade_ms": "音频交叉淡化(ms)",
    # 稀疏注意力
    "sparse_attn": "稀疏注意力（可选加速）",
    "sparse_tau": "稀疏阈值 τ（sol-attn）",
    "sparse_keep_percent": "保留 key 块 %（sla/vsa）",
    "sparse_start_percent": "稀疏起始进度",
    "sparse_min_tokens": "稀疏最小 token 数",
    "sparse_sink": "参考行精确注意（H3）",
    # 多边形分镜格 / 坐标网格
    "panel_polygon": "斜切格用多边形",
    "grid_preview": "输出坐标网格预览",
    # 分镜清单：默认时长与画幅（v0.7）
    "default_duration": "默认时长(秒)",
    "size_mode": "画幅模式",
    "size_preset": "画幅比例(官方规范)",
    "size_scale": "画幅缩放",
    "auto_shots_per_panel": "每格自动出镜数",
    # 前置出栈（v0.7）
    "shot_index": "取第几镜(从 1 起)",
    "include_anchor": "把分镜格当锚点",
    # VLM 层（v0.79）
    "characters": "角色闭集",
    "char_alias": "角色别名表",
    "base_url": "API 端点",
    "api_key": "API Key(留空读环境变量)",
    "api_key_env": "API Key 环境变量名",
    "max_edge": "发送长边上限",
    "temperature": "采样温度",
    "timeout": "超时(秒)",
    "max_pages": "只处理前 N 页",
    "strict_coverage": "严格覆盖(漏读即拒收)",
    "inspection_fallback": "内容审核回退",
    "use_proxy": "走系统代理",
    "include_non_narrative": "保留封面/设定页",
    "force_shots": "强制每格镜数",
    "duration_scale": "时长缩放",
    "episode": "剧集名",
    "style": "全局风格",
}

# --------------------------------------------------------------------------
# 输出级（key = 输出逻辑名；中文直接作为 RETURN_NAMES 是安全的）
# --------------------------------------------------------------------------
OUTPUT_LABELS = {
    "board": "分镜格",
    "images": "图像",
    "grid_preview": "坐标网格预览",
    "prompt": "提示词",
    "slots": "槽位对照表",
    "aspect": "画幅",
    "report": "报告",
    "refs": "参考包",
    "plan": "分镜计划",
    "prompts": "提示词（整套）",
    "positive": "正向条件",
    "latent": "潜空间",
    "frames": "帧序列",
    "audio": "音频",
    "passed": "是否通过",
    "meta": "元数据",
    "table": "序号映射表",
    "text": "文本",
    "vlm": "VLM 结果",
    "shot_text": "分镜文本（可贴回 ③）",
}


def label_input(name: str, default: str = "") -> str:
    return INPUT_LABELS.get(name, default or name)




def safe_display_name(factory, name: str, label: str = None, **kw):
    """给 io.Xxx.Input 加 display_name，遇到不支持的版本自动降级。

    绝不因为一个标签把节点注册搞挂。
    """
    if label:
        try:
            return factory(name, display_name=label, **kw)
        except TypeError:
            pass
    return factory(name, **kw)
