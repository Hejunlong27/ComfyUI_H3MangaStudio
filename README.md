# H3 Manga Studio

**漫改动画批量量产插件** —— 基于 MiniMax H3 的多参考视频生成能力（Ref2VA），
把「整页漫画」变成「一整集动画」。

- 位置：`<ComfyUI根>/custom_nodes/ComfyUI_H3MangaStudio/`
- 节点前缀：`H3Manga*`（与已装的 `ComfyUI_MiniMaxH3_Director` 的 `MiniMaxH3Director*` 刻意区分）
- 自定义类型：`H3_MANGA_REFS` / `H3_MANGA_PLAN` / `H3_MANGA_BOARD`
- 依赖：**无**（只用 ComfyUI 自带的 torch / numpy / cv2 / aiohttp）
- 本机实测环境：ComfyUI 0.38.2 + MiniMax H3 全套模型

---

## 0. 一句话说清它在解决什么

H3 的 Ref2VA 靠提示词里的 `<Picture 1>` / `<Video 1>` / `<Audio 1>` 指代参考素材。
官方源码（`comfy_extras/nodes_minimax_h3.py`）：

```python
for img in (ref_images or {}).values():
    ref_items.append({"type": "image", "data": resized})
```

**序号 = 追加顺序，不是键名。** 也就是说：

> 某镜少了一个角色，后面所有角色的 `<Picture i>` 会整体前移。
> 提示词还写着老序号 → 角色串号，**不报错、不崩，只是脸不对**。

漫改动画一集几十镜、每镜出场角色不同 —— 手工维护这个序号是必错的。

**本插件的全部价值：把这套序号契约自动化。** 编剧只写 `@纲吉`，
插件负责逐镜重排、算出正确的 `<Picture i>`、并把一致性检查做成可执行的断言。

---

## 1. 核心功能与节点划分

10 个节点，分五层。数据沿 `REFS / PLAN / BOARD` 三个自定义类型单向流动。

```
素材层                编排层                        执行层            资源层 / 输出
──────────────────────────────────────────────────────────────────────────────────
漫画页 ─► ①切分镜格 ─┐
                    ├─► ④分镜绑定 ─► ⑤提示词装配 ─┐
参考图 ─► ②参考包 ──┘                              ├─► ⑦整集渲染 ─┬─► CreateVideo ─► SaveVideo
                    └─► ③分镜清单加载 ────────────┘        │      └─► ⑩显存/内存释放
                                                          └─► ⑧一致性审计（可前置当体检）
                     ⑥单镜条件（调试）      ⑨计划预览（只读）
```

| # | 节点 | 作用 | 关键输出 |
|---|---|---|---|
| ① | **H3 漫画页切分镜格** | **非白连通域法**切格（可切回投影法）+ 自动 / 手动框 / 混合三种模式 + 出口去重叠 + **多边形分镜格（斜切版面）** | `H3_MANGA_BOARD` + 格图批次 + 报告 + **坐标网格预览** |
| ② | **H3 参考包** | 角色定妆图 / 风格基准图 / 参考音频 → 有序参考包 | `H3_MANGA_REFS` + 全量清单 |
| ③ | **H3 分镜清单加载** | JSON 文件 / 手写块文本 → 结构化计划，自动 17n+5 帧对齐 | `H3_MANGA_PLAN` + 报告 |
| ④ | **H3 分镜绑定** | 计划 × 分镜格绑定（按 panel 号 / 按顺序 / 不绑） | 已绑定计划 + 绑定报告 |
| ⑤ | **H3 提示词装配** | `@角色` → `<Picture i>`、运镜展开、风格注入、台词包裹 | 带最终提示词的计划 + 预览 |
| ⑥ | **H3 单镜条件**（调试） | 只构建单镜 conditioning + latent，接官方五件套做单镜调参 | `CONDITIONING` + `LATENT` |
| ⑦ | **H3 整集渲染 ★** | **批量量产核心**：内部逐镜循环，跑完一整集 | 帧序列 + 音频 + 渲染报告 |
| ⑧ | **H3 一致性审计** | 跑之前的体检：缺图、参考图太少、序号溢出、槽位漂移 | 分层报告 + 是否通过 |
| ⑨ | **H3 计划预览** | 只读预览，节点上挂前端面板显示「机器实际拿到的提示词」 | 文本 + 元数据 JSON |
| ⑩ | **H3 显存/内存释放** | 把模型从显存与内存里卸载掉，输出释放前后用量对比 | 释放报告 |

**为什么是「一个节点出一整集」（⑦）而不是工作流层循环**
ComfyUI 的工作流本身没有 for 循环。可行的落地方式就是**循环写在节点内部**，
工作流层面只呈现为几根线 —— 既满足「工作流模式」的心智模型，
又不用依赖执行图展开或子图。

---

## 2. 所需输入参数

### 2.1 参考图（`H3_MANGA_REFS`）

| 输入 | 说明 |
|---|---|
| `character_names` | 逗号/换行分隔。**第 N 个名字 ↔ 第 N 个图槽**。同名重复 = 同一角色的多张定妆图 |
| `character_images` | Autogrow 图槽（≤9）。例：`纲吉, 纲吉, Reborn` + 3 张图 → 纲吉 2 张、Reborn 1 张 |
| `character_describes` | 每行一个角色的外观锁定描述（行序 = 角色首次出现顺序）。**比纯标签稳得多** |
| `style_name` / `style_images` | 风格基准图，会出现在每个镜头的参考里（占 1 个 `<Picture i>`） |
| `audio_names` / `audios` | 独立参考音频（BGM / 音效 / 配音样本，≤3）。分镜里可用 `@名字` 引用成 `<Audio j>` |
| `video_names` / `videos` | 参考视频（≤3）。**一个槽位 = 一段多帧视频**，不拆帧；官方要求 ≥5 帧 |
| `video_audios` | 与 `videos` **按下标一一对应**的配套音轨（官方按键名尾号配对） |
| `max_ref_images` | 参考图上限，默认 9（官方硬上限）。降低可省显存与时间 |
| `name_source` | `slot` = 用你填的名字做 id / `index` = 强制 `ref1`/`ref2` |

### 2.2 分镜（`H3_MANGA_PLAN`）

两种来源，同一套字段语义。**块文本**（对编剧友好）：

```
# episode: Reborn 第30卷 目标283
# style: manga_anime
# default_size: 1344x768
=== s02 ===
panel: 1
duration: 5
characters: 纲吉, Reborn
camera: slow_push_in
action: @纲吉 从门后走出，@Reborn 坐在他肩上
speaker: 纲吉
dialogue: 这一刻终于还是到来了。
```

| 字段 | 含义 |
|---|---|
| `panel` | 绑定第几格（对应 board.panels 的 index） |
| `duration` | 秒。**会自动对齐到 H3 的 17n+5 帧网格**（5s → 124 帧） |
| `characters` | 出场角色（决定参考图槽位） |
| `camera` | 运镜短名，展开成镜头语言 |
| `action` | 画面动作。**用 `@角色id` 引用角色**，插件负责替换成 `<Picture i>` |
| `speaker` / `dialogue` | 说话人与台词（H3 原生支持中文对白，直接引号包裹，**不翻译**） |
| `style` | 覆盖全局风格 |
| `size` | 覆盖画幅（自动对齐 32 倍数） |
| `seed` | 覆盖派生 seed |
| `negative` / `extra` | 单镜负向 / 追加提示词 |

字段名支持中英混写与别名（`面板` / `时长` / `角色` / `运镜` / `画面` / `台词` / `说话人`…），
未知字段只警告不报错 —— 不会因为一行写错就让整集失败。

### 2.3 风格与角色一致性设置

| 手段 | 参数 | 说明 |
|---|---|---|
| **角色一致性** | `lock_consistency` | 自动生成「这几张参考图必须是同一个人」的锁定句 |
| | `include_describe` | 把 `character_describes` 的外观描述拼进提示词（**比纯标签稳**） |
| | 参考图数量 | 审计会警告 <2 张的角色（建议 ≥3：正面 / 侧面 / 全身） |
| **风格一致性** | `style`（全局）+ 每镜 `style` | 内置 6 个预设，可被 `style_json` 覆盖 |
| | `style_images` | 风格基准图，参与每个镜头的参考 |
| | `style_prefix` / `style_suffix` | 手工微调风格措辞 |
| **镜头语言** | `camera` | 内置 24 个运镜预设，可被 `camera_json` 扩展 |
| **负面** | `negative` + `negative_mode` | ⚠️ H3 官方 R2V **没有 negative 输入**（实测输入列表里没有），`BasicGuider` 也没有负向分支。所以负向词只有两条路：`append`（默认，以 `Avoid these problems: …` 追加进正向提示词）/ `ignore`（只记报告不写进提示词） |

---

## 3. 多参数如何组织与传递

这是本插件的技术核心，分四步。

### 第 1 步：打包 —— 从节点输入到 `RefPack`

```
character_names="纲吉, 纲吉, Reborn"
character_images=[img1, img2, img3]
        ↓  同名分组
RefPack.entries = [
  RefEntry(id="纲吉",  kind="character", images=[img1, img2], priority=0, describe="棕发橙眼…"),
  RefEntry(id="Reborn", kind="character", images=[img3],      priority=1, describe="黑西装婴儿…"),
  RefEntry(id="基准风格", kind="style", images=[style],        priority=900),
]
```

输入形态归一化吃得下：`None` / 单张张量 / 批次张量（自动拆批）/ list / dict（Autogrow，按槽位号排序）。

### 第 2 步：**逐镜密集重排** —— 序号唯一权威来源

`core/refs.py::shot_ref_layout()`。对**每个镜头**重新编号，顺序固定为：

```
[分镜格锚点] → [本镜出场角色（按分镜里的出场顺序）] → [风格图]
```

```python
# s01：分镜格 + 纲吉(2张) + 风格
ref_images = {ref_image_0: 格图, ref_image_1: 纲吉正面, ref_image_2: 纲吉侧面, ref_image_3: 风格}
picture_map = {"panel": [1], "纲吉": [2, 3], "基准风格": [4]}

# s02：只有 Reborn（少了一个角色）
ref_images = {ref_image_0: 格图, ref_image_1: Reborn, ref_image_2: 风格}
picture_map = {"panel": [1], "Reborn": [2], "基准风格": [3]}   # ← 整体前移，这才是对的
```

键名一定是**连续的** `ref_image_0..N`，因为官方是按 `.values()` 的迭代顺序取序号的。

### 第 3 步：引用替换 —— `@角色` → `<Picture i>`

`core/prompt_build.py::substitute_refs()`。正则匹配 `@纲吉`，查**本镜的** `picture_map`，
替换成 `<Picture 2>`。多图角色替换成 `<Picture 2> and <Picture 3>`。

- 角色名 → id 有别名表兜底（`resolve_character_ids`），写 `纲吉` / `泽田纲吉` / 带空格都能解析
- **未命中的引用必须可见**：写成 `@路人甲` 而参考包里没有 → 移除并记 warning，
  绝不留下字面量 `@路人甲`（模型会当普通文本，结果不可控）

### 第 4 步：传递 —— 复用官方节点，不自己拼 conditioning

`core/h3_api.py` 全部是**官方节点的薄包装**：

| 本插件函数 | 包装的官方节点 |
|---|---|
| `encode_reference_to_video()` | `MiniMaxH3ReferenceToVideo` |
| `encode_image_to_video()` | `MiniMaxH3ImageToVideo` |
| `add_guide()` | `MiniMaxH3AddGuide` |
| `apply_sigma_shift()` | `MiniMaxH3SigmaShift` |
| `sample()` | `KSamplerSelect` + `BasicScheduler` + `RandomNoise` + `BasicGuider` → `SamplerCustomAdvanced` |
| `decode_video()` / `decode_audio()` | `VAEDecode` / `VAEDecodeAudio` |

**为什么不自己拼**：H3 的参考素材靠 conditioning 里的 `minimax_refs` / `minimax_keyframes` 传递，
提示词侧的标签依赖 tokenizer 给出的 `minimax_ref_items` 序列位置。
这套口径**跟 ComfyUI 版本强相关**，自己拼必然跟丢，而且升级后会**静默失效**。
用**特征探测**（不是版本号）判断官方节点是否存在，缺了抛中文错误。

#### ★ 参考素材的张量契约：交付前必须是 4 维 `[B,H,W,C]`

官方 R2V 读参考图尺寸的写法是 `h, w = img.shape[1], img.shape[2]` —— 它**假定**输入是
`[B,H,W,C]`。插件内部为了「一个槽位 = 一张图」统一用 3 维 `[H,W,C]`，两者必须在边界上对齐：

| 环节 | 形态 | 谁负责 |
|---|---|---|
| 节点输入 / RefPack / 分镜格 | 3 维 `[H,W,C]`（一张图一个槽位） | `normalize_images()` |
| 交给官方 `execute()` | **4 维 `[1,H,W,C]`** | `shot_ref_layout()`（主）+ `h3_api` 出站守卫（兜底） |

**不对齐的后果（真实事故，v0.2.1 修）**：3 维 `[768,1152,3]` 会被读成
「h = 宽 1152、w = 通道数 3」，于是参考图被 resize 成畸形的 1152×32，
最后炸在 `comfy/utils.py::common_upscale` 里：

```
RuntimeError: shape '[1, -1, 3, 1152, 32]' is invalid for input of size 36864
```

这句报错**与形状毫无字面关联**（既不是 3 维也不是 `[B,H,W,C]`），
会把人带去查帧数对齐 / 分辨率倍数 / 参考图数量 —— 全是错的。
所以：`core/image_ops.py::as_ref_image()` 负责升维（并把非例行修正变成 warnings），
`tests/test_real_load.py` 用**真实的** `common_upscale` 与**真实的官方 R2V 节点**
把这条契约锁死（含反向验证）。

### 两道防线

1. **渲染前自校验**：⑦渲染节点会重算一次逐镜布局，与⑤装配时记录的
   `ref_image_keys` 比对，不一致直接报「参考包在装配之后被改动过，请重新跑提示词装配」
2. **可执行的回归守卫**：⑧审计节点重算布局并与 `picture_map` 比对 ——
   序号规则一旦被改坏，测试立刻变红（见 `tests/test_render_contract.py` 的反向验证）

---

## 4. 工作流的可配置与可扩展方式

| 想改什么 | 改哪里 | 不用改代码？ |
|---|---|---|
| 分镜内容 | 节点上的 `inline_text`，或外部 JSON/文本文件（`file_path`） | ✅ |
| 角色档案 | ②参考包节点的 `character_names` / `character_describes` | ✅ |
| 提示词结构 | ⑤的 `template`（占位符 `{scene}{refs}{action}{camera}{style}{dialogue}{extra}`） | ✅ |
| 运镜词库 | `templates/camera_presets.json`（填路径到 `camera_json`） | ✅ |
| 风格词库 | `templates/style_presets.json`（填路径到 `style_json`） | ✅ |
| 切格行为 | ①的 11 个参数；切不准就用 `manual_boxes` 手填框 | ✅ |
| 绑定策略 | ④的 `bind_mode`（精确 / 顺序 / 不绑）+ `panel_offset` | ✅ |
| 采样与加速 | ⑦的 `sampler_name` / `scheduler` / `steps` / `shift_video` / `shift_audio` / `lora_name` | ✅ |
| 输出画幅与时长 | 分镜里的 `size` / `duration`（自动对齐） | ✅ |
| 新增节点类型 | `nodes/*.py` + `core/*.py` | ❌ 写代码 |

**扩展的三条约定**（保证改动不破坏一致性）：

1. **序号规则只在 `core/refs.py::shot_ref_layout()` 一处**。
   任何新节点要拿参考图，都必须走它 —— 不要在别处拼 `ref_image_i`。
2. **文案只在 `core/ui.py` 一处**。改完跑
   `tools/gen_i18n.py` 重新生成 `locales/zh/nodeDefs.json` 与 `js/h3manga_labels.js`。
3. **输入名一律 ASCII**，中文只出现在 `display_name`。
   把输入名写成中文会污染 API 格式的键名，发布后极难改回。

---

## 5. 批量生产与任务编排的模块与接口

### 5.1 模块职责

| 模块 | 职责 | 依赖 comfy？ |
|---|---|---|
| `core/types.py` | 数据结构 + H3 硬约束常量（17n+5、32 倍数、9/3/3 上限）+ 帧数/画幅对齐 | ❌ |
| `core/script_parse.py` | 分镜清单解析（JSON / 块文本 / 别名 / 容错） | ❌ |
| `core/panel_split.py` | 漫画页切格（**连通域法 + 投影法 + 去重叠 + 多边形分镜格** + 手动框/多边形 + 阅读顺序） | ❌ |
| `core/refs.py` | 参考包装配 + **逐镜密集重排** + 名字解析 | ❌ |
| `core/prompt_build.py` | 提示词装配（引用替换 / 运镜 / 风格 / 台词 / 负向） | ❌ |
| `core/plan.py` | 绑定 / seed 派生 / 审计 / **任务展开** / 断点续跑对比 | ❌ |
| `core/image_ops.py` | 张量 pad / 拼批 / 音轨拼接（`device` 跟着源张量走） | ❌ |
| `core/h3_api.py` | 官方 H3 能力适配（唯一的 comfy 边界，且是延迟导入） | ⚠️ 延迟 |
| `core/ui.py` | 中文文案唯一真源（`NODE_LABELS` / `INPUT_LABELS` / `OUTPUT_LABELS`） | ❌ |
| `nodes/*.py` | 薄壳：输入归一化 → 调 core → 返回 `io.NodeOutput` | ✅ |

> `core/` 里除 `h3_api` 外**不 import comfy**，所以 94 项纯逻辑断言可以脱离 ComfyUI 跑。

### 5.2 批量编排的关键接口

```python
# ① 展开成可执行任务列表（纯字典，可打印/写盘/比对）
tasks = plan.build_render_tasks(plan, refs, board,
                                base_seed=1234, max_shots=0, steps=20,
                                skip_ids=["s03"], ref_policy="per_shot")
# → [{"id","index","prompt","width","height","length","seed",
#     "ref_image_keys","picture_tags","layout_order","board_image_index", ...}]

# ② seed 派生：shot_id 哈希，不是 index 累加
seed = plan.derive_seed(base_seed=1234, index=7, shot_id="s07")
# 插入 / 删除 / 重排镜头都不会改变其它镜头的 seed → 断点续跑才成立

# ③ 成本估算（相对算力单位，跑之前就知道这一集比上一集重几倍）
cost = types.estimate_cost(plan, steps=8)

# ④ 断点续跑对比（指纹 = id+prompt+length+w+h+seed）
print(plan.plan_diff(previous_report, plan))
# → 可复用 N 镜 / 新增 N / 已变更 N / 已删除 N

# ⑤ 整集渲染
frames, audio, report_json = H3MangaEpisodeRender.execute(...).result
```

### 5.3 批量相关的节点参数（⑦整集渲染）

| 参数 | 作用 |
|---|---|
| `max_shots` | `0` = 全部。**调试时先跑 1~2 镜确认参数**，别一上来跑整集 |
| `base_seed` | 基准种子（`control_after_generate` 可 randomize） |
| `skip_done_ids` | 逗号分隔的镜头 id，断点续跑时跳过 |
| `on_error` | `skip`（默认，单镜失败继续）/ `abort`（立刻中断） |
| **`unload_stages`** | **默认开**。编码后卸文本编码器、采样后卸视频模型、解码后卸 VAE。本机 33.5GB 权重对 16GB 内存，关掉几乎必然爆内存 |
| `sparse_attn` | `none`（默认）/ `sol-attn` / `sla` / `vsa` —— 稀疏注意力（见 §5.5） |
| `sparse_tau` / `sparse_keep_percent` | `sol-attn` 用 τ（默认 1.3）；`sla`/`vsa` 用保留百分比（默认 10%） |
| `sparse_start_percent` | 采样进度到百分之几才开始稀疏（默认 0.2，之前保持 dense） |
| `sparse_min_tokens` | 序列短于这个 token 数保持 dense（默认 12288） |
| `sparse_sink` | H3 专用：让 query 精确注意参考图行。默认 `exact_kv_and_rows`（**参考图不会被稀疏化**） |
| `lora_name` / `lora_strength` | H3 turbo LoRA。下拉只列文件名含 `h3`/`minimax` 的，**加载失败自动降级为无 LoRA 并记警告** |

**降级原则**：单镜失败默认跳过并把原因写进报告 ——
不因为第 7 镜炸掉就让前面 6 镜白算。`on_error=abort` 才中断。

**预检（`_preflight`）**：采样之前一次性报清 ——
帧数对齐修正、画幅倍数修正、超长镜头、未绑定格图的镜头、参考包里没有的角色、参考图溢出。
问题在**花钱之前**暴露。

### 5.4 ★ 低显存运行（8GB 显存 / 16GB 内存）

先看两组实测数字，它们决定了整套运行方式：

| 项 | 值 |
|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（R2V 主链路 UNET） | **19.53 GB** |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors`（文本编码器） | **13.93 GB** |
| 权重总量 | **33.5 GB** |
| 物理内存 / 显存 | **16 GB / 8 GB** |

**结论**：任何时刻都装不下两个模型。所以本插件把渲染拆成**三个阶段**，
每个阶段只用**一个**模型，阶段结束立刻卸载：

```
阶段 A  逐镜 layout + 条件编码   ── 用 clip  ──▶ 卸载文本编码器（放掉 ~14GB）
阶段 B  逐镜采样                 ── 用 model ──▶ 卸载视频模型（放掉 ~19.5GB）
阶段 C  逐镜解码                 ── 用 vae   ──▶ 卸载 VAE
```

**为什么卸载是安全的**：`comfy/sd.py` 里 `CLIP.encode_from_tokens` 会调
`self.load_model()` → `load_models_gpu([self.patcher])`，`VAE.decode` 各分支同样先
`load_models_gpu([self.patcher])`（源码已核实）。**卸载之后下次用到会自动重新加载**，
不会报「权重在 meta 设备上」。

渲染报告里的 `memory.stages` 会记下每个阶段的显存/内存读数，
用来验证「上一个模型是不是真的卸掉了」：

```json
"memory": {
  "unload_stages": true,
  "lines": [
    "起点                  显存 673/8188 MiB 用（空闲 7515）  内存 13.90/15.23 GB（91%） 页文件余 3.6 GB",
    "阶段A后 · 文本编码器    显存 210/8188 MiB 用（空闲 7978）  内存 11.20/15.23 GB（74%） 页文件余 6.1 GB"
  ]
}
```

#### 本机推荐配置

| 项 | 推荐值 | 理由 |
|---|---|---|
| ComfyUI 启动参数 | `--lowvram --reserve-vram 0.8` | **不要用 `--novram`**：16GB 内存装不下 19.5GB 权重，会疯狂换页 |
| 画幅 | **768×448**（先跑通）→ 960×544 | 1344×768 单段解码帧就 4.5GB，8GB 卡撑不住 |
| `unload_stages` | **保持开启** | 见上 |
| `ref_image_size` | `match` | `max` 会用 2048 短边，慢数倍且多占显存 |
| `max_ref_images` | 4~5 | 参考图 latent 每步都要过一遍 |
| `steps` | 4~8 + turbo LoRA | 无 LoRA 用 20 |
| `audio_vae` | 首轮先不接 | 少一个 VAE 常驻 |
| `max_shots` | 调试时设 1，跑通再设 0 | 单镜出片后再铺开 |

#### 诊断工具

`tools/wb_memwatch.py`（零依赖）同时采 GPU 与系统内存两条曲线。
「ComfyUI 自动重启」有三种可能（显存不足 / 内存被 OOM Killer 杀 / CUDA 驱动失败），
只有拿到双曲线才能区分：

```bash
# 单次快照
<安装根>/standalone-env/python.exe custom_nodes/ComfyUI_H3MangaStudio/tools/wb_memwatch.py
# 跑工作流前启动，跑完停 —— 得到重启瞬间的曲线
<安装根>/standalone-env/python.exe .../wb_memwatch.py --watch 600 --csv vram_log.csv
```

> ⚠️ **画幅必须是 32 的倍数**：`432` **不是**（432/32 = 13.5），写 `768x432` 会被
> `align_canvas()` 吸附成 **768×448**。想精确控制请用真正合法的值：
> `768×448`、`960×544`（30×32 与 17×32，恰好合法）、`1344×768`。

> 已有实测：本机 GPU 利用率 100%、显存 7284/8188 MiB、系统内存 13.86/15.23 GB（91%）、
> 页面文件可用仅 3.64 GB。**「自动重启」的真因大概率是内存而不是显存。**

### 5.5 稀疏注意力（可选加速，默认关）

ComfyUI **已内置**官方 `BlockSparseAttention`（category `model/patch`，标着 experimental），
本插件把它封成了 ⑦ 的一个参数，**不需要在画布上多接节点**。

| method | 说明 | 需要新权重吗 |
|---|---|---|
| `none` | **默认**，行为与以前完全一致 | — |
| **`sol-attn`** | Sparsifying Online Attention，**免训练**自适应阈值（`sparse_tau`，默认 1.3 ≈ 保留 11~12% key block） | 不需要 |
| `vsa` | FastVideo 的 Video Sparse Attention（3D 立方体切块 + 学出来的粗分支） | 官方注释说需要 FastH3 权重才发挥 |
| `sla` | Sparse-Linear Attention | 需配套 LoRA |

#### 为什么敢在漫改链路上开它

官方 `sink_conditioning` 的默认值 `exact_kv_and_rows` 会让**每个 query 精确注意到
packed 序列里的文本 / 音频 / 参考图行**（官方注释：约 3% 额外开销）。
也就是说 —— **角色的参考图不会被稀疏化**，一致性契约不受影响。

#### ⚠️ 一个必踩的坑：`min_tokens` 门槛

官方序列短于 `min_tokens`（默认 **12288**）会**静默**回退 dense。本插件按 H3 的
latent 几何（16× 空间压缩、`FRAME_PER_TOKEN=(1,4,4,4,4)`）提前估算并**在预检里警告**：

| 画幅 | 5s（124 帧）token 估算 | 默认门槛 12288 |
|---|---|---|
| 704×384 | 9,768 | ❌ 走 dense |
| **768×448** | **12,432** | ✅ 刚过线（只高 1%） |
| 960×544 | 18,870 | ✅ |
| 1344×768 | 37,296 | ✅ |

> 所以要稳，要么画幅 ≥960×544，要么把 `sparse_min_tokens` 降到 **8192**。

#### A/B 怎么测

`tools/sparse_ab.py` 一次生成两份**只差一个开关**的工作流（同 seed），跑完再对比：

```bash
P=<ComfyUI根>/.venv/Scripts/python.exe
# 1) 生成 ab_off_dense.json / ab_on_sol_attn.json
$P custom_nodes/ComfyUI_H3MangaStudio/tools/sparse_ab.py make --out-dir D:/h3ab
# 2) 两份各跑一次，把 ⑦ 的 report 输出各存成 .txt
$P .../tools/sparse_ab.py compare --off off.txt --on on.txt
# 3) 抽同一时间点的帧并排成图
$P .../tools/sparse_ab.py frames --off off.mp4 --on on.mp4 --out-dir D:/h3ab
```

`compare` 会明确告诉你三件事：**耗时变化多少、峰值显存变化多少、稀疏到底生效没有**
（没生效会直接说原因，而不是让你误以为"开了没用"是插件的锅）。

---

## 6. 安装与使用

### 6.1 安装

已装在本机：

```
<ComfyUI根>/custom_nodes/ComfyUI_H3MangaStudio/
```

**改完 Python 代码必须重启 ComfyUI**（刷新网页不会重载 Python 模块），
并删掉包内 `__pycache__` —— 残留的 `.pyc` 会让旧代码继续生效。

启动日志里会有一行能力自检：

```
H3MangaStudio: 官方 H3 能力齐备（4 项）
```

### 6.2 最小可用流程

1. 把 `templates/H3MangaStudio_example_workflow.json` 拖进 ComfyUI 画布
2. 改 4 个 `LoadImage` 的文件名（1 张漫画页 + 3 张角色定妆图）
3. 改②参考包的 `character_names` / `character_describes`
4. 改③的 `inline_text`（你的分镜）
5. 先把⑦的 `max_shots` 设成 `1` 跑通，再设 `0` 跑整集

### 6.3 本机模型文件（已在示例工作流里填好）

| 用途 | 文件 |
|---|---|
| UNET（Ref2VA，r2v 专用） | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| CLIP（type 必须是 `minimax`） | `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| 加速 LoRA（可选） | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` |

> ⚠️ 官方模板里写的是 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`，
> 本机没有这个文件，示例工作流已换成实际存在的 `int4_convrot` 版本。
> **H3 的 UNET 变体不能混**：`t2v/i2v/flf2v` 用 FL2VA，`r2v` 用 Ref2VA。

### 6.4 调参速查

| 症状 | 调什么 |
|---|---|
| 角色不像 | 加参考图（≥3 张）；`ref_image_size` 改 `max`（最像但慢数倍）；写 `character_describes` |
| 画面不跟分镜 | 确认 `board` 接到了⑦；`ref_image_size` 用 `match`；提示词里已有 `follow <Picture 1>` |
| 动作太乱 | 降 `duration`（15 秒内不要塞超过 2~3 个动作节点）；运镜换 `static` |
| 太慢 | 挂 turbo LoRA + `steps` 降到 4~8；降 `max_ref_images`；分辨率降到 768×432 |
| 台词口型不对 | 台词必须原样保留中文，别翻译；`speaker` 填对（它会成为 `<Picture i> says:`） |
| 想省显存 | 用 `max_shots` 分批跑 + `skip_done_ids` 续跑；确认 `unload_stages` 是开的 |
| 想省内存 | 保持 `unload_stages` 开启（编码/采样/解码各自成段、段末卸载）；画幅降到 768×448；关掉浏览器；页面文件留足空间 |
| ComfyUI 自己重启了（没有报错） | 大概率是**内存**被系统回收而不是显存 —— 用 `tools/wb_memwatch.py` 采双曲线确认 |
| 想再快一点 | `sparse_attn=sol-attn` + `sparse_min_tokens=8192`（序列短的话默认 12288 会挡住）；**先按 §5.5 做 A/B 确认画质** |
| 开了稀疏但速度没变 | 序列 token 数低于 `sparse_min_tokens`，官方静默走了 dense —— 预检里会有警告，降到 8192 或把画幅提到 960×544 |
| 报 `shape '[...]' is invalid for input of size ...` | **参考图/视频张量维度**问题（官方要求 4 维 `[B,H,W,C]`）。v0.2.1 起 `shot_ref_layout` 与 `h3_api` 两道关口会自动升维；若仍出现，把该图的 `shape` 贴出来 |
| 几个格被并成一整格 | 报告里会报「**过大连通块**」：这些格是**共用黑框线/大块黑底**连在一起的（第10卷 _081 实测：大块 bbox 占 71% 但像素只占 38%）。**默认已自动处理**（`component_auto_open` 逐页试开运算半径），报告会写明用了哪个半径。若报告说「已试到 N 仍未消除」→ 该页用 `manual_boxes` 手填；**斜切版面请写多边形**（每行 8 个数 `x1,y1,x2,y2,x3,y3,x4,y4`），坐标对着 ① 的 `grid_preview` 输出数 |
| 报告说「自动开运算 component_open=N」 | 正常现象：这一页检测到漏切迹象，自动把开运算半径提到了 N。想关掉自动就设 `component_auto_open=False`（然后自己填 `component_open`） |
| 整页只切出 1 格 | 报告会提示「请确认是否整版大图」。扉页/跨页/出血格**并成 1 格是对的**；若是正文页则是漏切 → `manual_boxes` |
| 一个格被切成好几个 | 确认 ① 的 `frame=component`（默认）。`projection` 会把格内空白（竖排文字栏间距、对话框内部、天空）当成装订线 |
| 切得太碎 / 有小碎格 | ① 调高 `component_min_area`（默认 0.015 → 0.03） |
| 相邻格有重叠区域 | ① 的 `deoverlap` 保持开（默认开）。关掉是为了复现问题 |
| 几个格被并成一整格 | 该页很可能是**扉页/斜切格/大面积黑底**（连通域法会把它们并成一块）→ 用 `manual_boxes` 手填；斜切格写**多边形**（8 个数一行） |
| 想对照旧行为 | ① 把 `frame` 切回 `projection`（v0.4.0 之前的算法） |

---

## 7. 测试（四层，全绿）

```bash
cd <ComfyUI根>
.venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tests/run_all.py
```

| 层 | 文件 | 验什么 | 结果 |
|---|---|---|---|
| 1 纯逻辑 | `test_pure_logic.py` | 帧对齐 / 清单解析 / **逐镜重排** / 提示词装配 / 绑定 / 审计 / **切格（两种框架 + 格内白带 + 去重叠 + 框线焊接告警 + 自动开运算与守卫 + 多边形分镜格）** / 参考素材张量契约 / 资源管理与卸载原语 | 189/189 |
| 2 schema 桩 | `test_schema_stub.py` | 输入名 ↔ `execute` 形参、optional 落位、Combo default、中文标签 | 116/116 |
| 3 真实加载 | `test_real_load.py` | ComfyUI 自己的加载器 + Autogrow 真实形态 + **官方节点签名漂移探针** + **真实 `common_upscale` / 真实官方 R2V 端到端** | 191/191 |
| 4 契约 | `test_render_contract.py` | **桩替换 + 真跑 `execute()` + 反向验证 + 检查器自检** + **分阶段卸载** | 73/73 |

**第 4 层为什么最重要**：它把每张参考图用一个唯一数值填充，
然后反查「提示词里的 `<Picture i>` → 实际传进去的第 i 张图 → 指纹」是否对得上。
只要序号规则被改坏，这条立刻变红。

并且做了**两层反向验证**：
1. 故意把 `shot_ref_layout` 换成「全局池顺序」的错误实现 → 必须看到渲染节点抛错；
2. 直接打乱一次调用的 `ref_images` 顺序 → 契约检查器必须报违规。

第 2 条是「检查器自检」——没有它，你无法区分「测试通过」和「测试在空转」。

**张量形状契约也有三层守卫**（v0.2.1）：
1. 第 1 层断言 `shot_ref_layout` 产出的都是 `[1,H,W,C]`；
2. 第 4 层把「3 维参考图 → `common_upscale` 畸形 reshape」逐字复刻成可断言的用例
   （含反向验证：关掉升维必须变红）；
3. 第 3 层用**真实的** `comfy.utils.common_upscale` 与**真实的官方
   `MiniMaxH3ReferenceToVideo.execute`**（假 clip，只需 CPU）端到端验证，
   退回 3 维时能复现线上那句原文。

### 工作流校验（另一条独立链路）

```bash
.venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tools/verify_workflow.py
```

它比通用 JSON 结构校验更严：**逐节点对着真实 `INPUT_TYPES()` 核对**
`widgets_values` 数量、`inputs` 顺序、每条连线的目标槽位名与类型、
拓扑顺序、分组包含、节点两两不重叠。生成工作流后必跑。

---

## 8. 已知边界（故意固化在测试里）

| 边界 | 说明 |
|---|---|
| 斜切分镜 + 大面积黑底扉页 | **两种框架都失效**：投影法没有贯穿 gutter；连通域法会把多格并成一块 → 必须 `manual_boxes` |
| 相邻格描边相接 / 共用框线 | **连通域法的固有弱项**：两格共用一条黑框线 → 连成一个连通块 → 漏切。实测 bbox 占页面 71% 的块，像素只占 38%（就是一张细线网）。报告会报「过大连通块」，`component_open=1~2` 可分化，但**没有普适参数** → 靠 manual_boxes / 编辑器 |
| 相邻格网点连片 | 同上，会并成一块。用 `count_contained()` 可检「框套框」，但**不要直接删内层**（内层往往才是真格，错的是外层超集框） |
| 格内空白（文字栏间距 / 对话框 / 天空） | v0.4.0 起由**连通域法**解决（`frame=component`）。切回 `frame=projection` 会复现过切 |
| 相邻框重叠 | v0.4.0 起出口自动去重叠（`deoverlap=True`）。关掉可复现（实测 19 页 46 对重叠，最严重侵入比 0.996） |
| 墨迹占比 >85%（整页全黑） | 两种框架都失效，会给出警告 → 用 `manual_boxes` |
| 跨页出血格 | 没有装订线，会被当成一整格（**这是对的**） |
| 参考图上限 | 9 张（官方硬上限）。超出会被静默丢弃 → 审计报 P0 |
| 时长上限 | 官方训练区间 ≈362 帧（15s）。超出会警告，效果不保证 |
| 参考视频 | ≤3 条，且每条 ≥5 帧（官方硬约束）。插件按「一个槽位 = 一段多帧视频」处理，**不拆帧** |
| 参考音频 | ≤3 条独立音频；另有 ≤3 条与视频配对的音轨（按键名尾号配对） |
| 负向提示词 | H3 无 negative 输入，只能以自然语言进正向（`negative_mode=append`） |
| 参考素材维度 | 官方只吃 4 维 `[B,H,W,C]`（视频 `[T,H,W,C]`）。插件内部是 3 维，**边界统一升维**；通道数不是 1/3/4、或剥不掉的 ≥5 维会**直接报错**（不静默改） |
| 碎条分镜格 | 切格切出的 <8px 条带会被当参考图用，但没有画面信息 → 预检给警告，建议用 `manual_boxes` 手填框 |

---

## 9. 目录结构

```
ComfyUI_H3MangaStudio/
├── __init__.py                 # 注册入口 + WEB_DIRECTORY + 启动能力自检
├── pyproject.toml / requirements.txt
├── README.md                   # 本文件
├── core/                       # 纯逻辑（不 import comfy，可脱离 ComfyUI 测试）
│   ├── types.py                # 数据结构 + H3 硬约束 + 帧数/画幅对齐
│   ├── script_parse.py         # 分镜清单解析
│   ├── panel_split.py          # 漫画页切格
│   ├── refs.py                 # 参考包 + ★逐镜密集重排
│   ├── prompt_build.py         # 提示词装配
│   ├── plan.py                 # 绑定 / seed / 审计 / 任务展开
│   ├── image_ops.py            # 张量 pad / 拼批 / 音轨拼接（device 安全）
│   ├── mem.py                  # ★ 显存/内存快照 + 卸载原语（绝不抛异常）
│   ├── h3_api.py               # 官方 H3 能力适配（唯一 comfy 边界）
│   └── ui.py                   # 中文文案唯一真源
├── nodes/
│   ├── _io_types.py            # 自定义 IO 类型单例
│   ├── asset_nodes.py          # ①②
│   ├── plan_nodes.py           # ③④⑤⑧⑨
│   ├── mem_nodes.py            # ⑩
│   └── render_nodes.py         # ⑥⑦
├── js/h3manga_preview.js       # 轻量只读预览面板（serialize: false）
├── locales/zh/nodeDefs.json    # i18n（由 tools/gen_i18n.py 生成）
├── templates/                  # 示例工作流 / 分镜模板 / 运镜与风格词库
├── data/characters_example.json
├── tools/
│   ├── gen_i18n.py             # 从 schema 生成 i18n 与前端 labels
│   ├── make_example_workflow.py# 从真实 INPUT_TYPES 生成示例工作流
│   ├── wb_memwatch.py          # ★ 显存/内存双曲线监视（诊断「自动重启」）
│   ├── sparse_ab.py            # ★ 稀疏注意力 A/B（生成对比工作流 + 对比报告 + 抽帧并排）
│   └── verify_workflow.py      # 逐节点核对工作流（widgets / 槽位 / 类型 / 布局）
└── tests/                      # 四层测试 + run_all.py
```

## 10. 变更记录

### v0.4.3 —— 多边形分镜格：斜切版面终于能表达（本轮）

**问题**：`Reborn_10_134` 的正确分区是**斜的四边形**（用户用绿线标了出来），
但插件从数据模型到裁图全是**轴对齐矩形**，连「手填框」这条退路都走不通
—— 解析器只收 4 个数 `x,y,w,h`，写 8 个数会被当语法错误丢掉。

**量化根因**：
* 该页整页是**一个**非白连通块（418 万像素里 53.9% 有内容、46.1% 是空白），
  格与格之间是**51° 的粗黑斜带**（Hough 实测 `(239,0)→(761,644)`，长 829px，
  是全页最长直线）。
* 斜格的 bbox **必然**与邻格 bbox 相交 —— 矩形裁只能二选一：漏掉自己一角，
  或把邻居的内容带进来。**这是几何必然，不是算法不够好。**
* 097 实测：`#0` 格的多边形面积只占它 bbox 的 **49.3%** → 矩形裁把一半的
  框内像素换成了别人的东西（其中大部分是纸白，真正有害的是邻居格的像素）。

**改动**：
* `core/panel_split.py`
  * 形状升级为 `{"page", "box", "poly"}`；`poly=None` = 普通矩形（旧行为不变）
  * 新增 `polygon_area` / `poly_to_box` / `clamp_poly` / `crop_polygons` /
    `crop_shapes` / `upscale_shapes` / `shape_key` / `shapes_from_boxes`
  * 新增 `parse_manual_shapes()`：**4 个数 = 矩形 / 8 个数 = 四边形 /
    任意 ≥3 顶点多边形**，两种可混写；`parse_manual_boxes()` 降为薄包装（向后兼容）
  * 新增 `split_page_shapes()`（返回形状）与内部 `_component_shapes()`；
    `split_page_boxes()` 变成矩形版包装，**签名与行为完全不变**
  * 自动多边形 `_component_poly()`：`findContours + approxPolyDP` 取轮廓。
    面积 / bbox ≥ 0.92 判为矩形返回 `None`（不给每个矩形格挂 4 个多余顶点）；
    开运算半径会先膨胀回去（`restore`），否则多边形会比真实格小一圈
  * 去重叠只作用于**矩形**格（多边形格用自己的掩膜裁，bbox 相交无害；
    把它一起推开反而会让 box 不再是它的 bbox）
* `core/types.py`：`Panel.poly` 字段；报告显示「形状=N边形」
* ① 新增 `panel_polygon`（auto/off）与 `grid_preview`，并多出第 4 个输出
  **坐标网格预览**（每 100px 一条刻度 + 编号 + 轮廓）——
  斜切格要填 4 个角，**没有刻度根本没法数**，这就是「看得见」那一步
* ① 的 `manual_boxes` 现在支持多边形写法（widget 名不变，旧工作流不受影响）

**关键回归测试**（`test_pure_logic.py`）：
> 两个三角格共用一条对角线、bbox 各占整页 → 断言「矩形裁的角落取到的是
> **邻居**的像素（120）」，而「多边形裁的同一位置是纸白（255）」，多边形内部像素不变。
> 反向验证：`panel_polygon=off` 必须退回纯矩形。

四层 **189 / 116 / 73 / 191 全绿**（原 173 / 115 / 73 / 189）；① 现 **20 个 widget**。

**仍然做不到的**：斜切格若**被粗黑带焊成一整块**（134 的 `#2` 就是），
自动通道给不出正确**格数** —— 它给的多边形**形状是对的**（134 `#0` 的红多边形
严格沿斜带走），只是块本身是并起来的。⇒ 这类页走**手填多边形**；
自动的下一步是「在已识别的多边形**内部**，再按内部斜带细分」。

### v0.4.2 —— 开运算「逐页自动」调优（上一轮）

**背景**：用户指出 v0.4.1 展示的那张 `component_open=1` 图里，box 0 仍然是**三格焊在一起**。
复核确认：**`component_open=1` 对那一页根本没解决**（报告里仍报「过大连通块」），
是我上一轮展示时没把这点标注出来。

**实测那条「连接组织」到底多厚**（第10卷 _081，逐半径扫）：

| 开运算半径 | 最大连通块像素 | 组件数（≥1.5%） |
|---|---|---|
| 0 | 798 469 | 3 |
| **1** | **757 747**（几乎没变！） | 4 |
| **2** | **175 941**（断开了） | 5 |
| 3 | 166 258 | 5 |
| 4 | 163 331 | **2**（过度侵蚀） |

→ 需要的半径**逐页不同**（_081 要 2；_135 用 2 反而从 5 格掉到 3 格）。
**所以不该让用户猜半径。**

**改了什么**：

* 新增 `component_auto_open`（**默认开**）：**逐页**把半径从 0 试到 3，
  取第一个「漏切迹象消失」的候选。已经正常的页**一步都不动**。
* ★ **守卫：开运算只许细分、不许吞格**（框数不得少于半径 0 时的结果）。
  这条是踩出来的 —— 没有守卫时会出现**退化解**：加大半径把内容细的小格
  **整个腐蚀掉**，「告警消失」其实是被删格删出来的（实测 108 页 4 格 → 1 格）。
  加入守卫后，这类页被正确挡回半径 0。

**实测**：

| | 19 页平均偏差 | 完全命中 | 触发自动的页数 |
|---|---|---|---|
| 关掉自动 | 1.74 | 6/19 | 0/19 |
| **自动（新默认）** | **1.63** | 6/19 | 5/19 |

用户报的页：`_081` **3 → 5 格**（上排拆成 獄寺/学校，中排拆成左右）；
`_134` 1 → 3 格（仍报漏切 → 提示手填）；`_153` / `_135` 无变化（本来正常）。

**测试**：四层 **173 / 115 / 73 / 189 全绿**。
新增断言：默认必须自动分开焊住的格、报告必须写明采用了哪个半径、
反向验证（关掉自动必须重新焊上并重新告警）、守卫（自动不得减少格数）。

**仍然存在**：`_134` 这类「整页被一张大图/重墨连成一片」的页，
试到半径 3 也分不开 → 报告会明确说「该页多半需要 manual_boxes 手填」。
**几何层的极限到此为止，编辑器（人工确认）仍是最终解法。**

### v0.4.1 —— 相邻格「共用框线」漏切：告警 + 开运算手轮（上一轮）

**背景**：用户看对比图发现「有部分没有区分出来的分镜区域」。实测复现（第10卷 _081）。

**根因（已用像素数据确证）**：`Reborn_10_081` 的那个大块 —— bbox 占页面 **71.46%**，
但它自己的**像素只占 38.11%** → 说明它不是一块实心画面，而是**一张细框线织成的网**。
看图就明白：上格与中格**共用一条横框线**、中格左右两格**共用一条竖框线**，
三条线连成一片 → 三格被焊成一块。
另一项佐证：该页**行方向最小非白率只有 0.097**（p5 = 0.247）——**整页找不到一条横向白缝**。

> ⚠️ 顺带纠正上一轮的一个误判：我当时测「块内是否有贯穿线」，要求线段跨 **80% 的块**，
> 而共用框线只跨**它自己那一格**，所以测出来是 0，于是错判成「不是框线问题」。

**改了什么**：

* **新增「过大连通块」告警**（默认开，纯确定性）：一个块同时在两个方向占掉大半页
  （宽 ≥85% 且高 ≥45%，或转置）就告警，并直接给出处置建议。
  判据刻意用**版面跨度**而不是「比别的格大」—— 后者会把真的大格误报。
  实测在 `_081`、`_134` 上准确触发。
* **新增 `component_open`（开运算半径，默认 0）**：连通域之前先做一次开运算，
  用**打断细框线**的办法把焊住的格分开。实测 `_081`：0→3 格、1→4 格、2→**5 格**。
* `len(boxes)==1` 且几乎占满整页时，改为提示「请确认是否真是一张整版大图」。

**⚠️ 为什么 `component_open` 默认 0，而不是直接设 2**：它是个**按卷调的手轮**，不是普适参数。

| 用法 | 第30卷 19 页平均偏差 | _081 | _135 |
|---|---|---|---|
| 0（默认） | **1.74** | 3 格 | 5 格 |
| 1 | 1.79 | 4 格 | 5 格 |
| 2 | **3.05**（过切） | 5 格 ✓ | **3 格**（回退） |

**所以：先看报告里的告警，只对报出来的那几页调。**

**测试**：四层 170 / 115 / 73 / 188 全绿。
新增关键回归：合成一张「两块内容靠 2px 细线相连」的页 —— 断言默认必须报「过大连通块」、
`component_open=1` 必须把它打断、再关回去必须重新焊上（反向验证）。

**仍然存在**：共用框线漏切**没有普适解**。几何层到这里的极限就是「告警 + 手轮」，
真正解决要靠**编辑器**（人看一眼就改完了）。

### v0.4.0 —— 切格换框架：连通域法 + 去重叠（上一轮）

**背景**：用户反馈 097 页「台词区域导致一个格被识别成多个」「有格子划分出重叠区域」。
在 19 页逐格人工标注 + **64 页跨 8 卷抽样**上量了五种方案后换掉了主算法。

**根因**：投影法把「某一行/列的墨迹少」当格边界，但**格内空白**
（竖排文字的栏间距、对话框内部、天空、页边）同样满足、**而且常常更干净** ——
097 页实测真 gutter 行墨迹 0.027~0.054，文字栏间的缝只有 **0.009~0.020**。
那页共切 12 刀，**8 刀砍在格内部**。

**换了什么**：

| 方案 | 平均绝对偏差（19 页） | 「整页 1 格」比例（64 页） | 过切比例 |
|---|---|---|---|
| 投影法（旧） | 2.89 | 27% | 22% |
| **连通域法（新默认）** | **1.74** | **12%** | **3%** |

* `frame="component"`（**新默认**）：一个「非纯白」连通块 = 一格。
  格内空白天然与画格同块 → 不会制造假切点。`frame="projection"` 保留作兼容与降级。
* `deoverlap=True`（新默认）：出口把互相重叠的框沿中线分开。
  重叠来自「向外扩 padding」+「碎片合并取并集」—— 实测旧算法 19 页有 **46 对重叠**，
  最严重的「侵入比」达 **0.996**（一个框几乎完全被另一个框盖住）。去重叠**零精度代价**。
* 新增 `count_overlaps()` / `count_contained()`（检测「框套框」）两个纯函数。
* ① 节点新增 5 个参数：`frame` / `white_threshold` / `component_min_area` /
  `component_close` / `deoverlap`，**追加在 widget 末尾**以兼容已有的工作流。

**也在本轮试过但失败的**（留档避免重复投入）：框线检测（任何长线条都被当框线，
爆到 20~60 格）、下刀门禁（实线度 / 长游程 / 2D 走廊贯通度三个度量都分不开真刀假刀）、
连通域+局部投影细分（1.74 → 2.16）。

**仍然存在**：斜切格 + 大面积黑底扉页两种框架都失效（必须 `manual_boxes`）；
连通域法会把描边相接的相邻格并成一块；「看得到 + 改得动」的编辑器仍是 P0。

**测试**：四层 165 / 115 / 73 / 187 全绿（原 150 / 114 / 73 / 182）。
新增的关键回归：把「格内贯穿白带 → 投影法切 2 格 / 连通域法保持 1 格」固化成断言，
并给去重叠加了反向验证（`iters=0` 必须复现重叠）。

### v0.3.1 —— 稀疏注意力接入（上一轮）

**背景**：FastVideo / Hao AI Lab 发布了 **FastH3**（H3 蒸馏到 8 步 + VSA 稀疏注意力）。
核实后发现：**ComfyUI 已内置官方 `BlockSparseAttention` 节点**，而且它的
`sol-attn` 模式是**免训练**的 —— 也就是说**不需要下载任何新模型**，
直接作用在现有的 `ref2va_pruned_int8` 权重上就能吃到稀疏收益。
（FastH3 的 V2/Trim 权重只有 t2v/i2v、**不含 Ref2VA**，本插件的角色一致性链路用不了。）

| 改动 | 文件 | 说明 |
|---|---|---|
| 新增 `sparse_attn` 等 6 个参数 | `nodes/render_nodes.py` | 把官方 `BlockSparseAttention` 封成 ⑦ 的参数，**不需要在画布上多接节点**。默认 `none`，行为与以前完全一致 |
| 新增 `apply_sparse_attention()` | `core/h3_api.py` | 薄包装官方节点。**失败只降级不抛错**（稀疏是可选优化，不是正确性依赖） |
| 新增 `sparse_capability()` | `core/h3_api.py` | 探测官方节点是否存在 + `comfy_kitchen.sol_attn_is_available()`。启动自检里会带出来 |
| ★ `estimate_video_tokens()` | `core/types.py` | 按 H3 的 latent 几何提前估算 token 数，**在预检里警告「低于 min_tokens 门槛，开了也不加速」** |
| 新增 `tools/sparse_ab.py` | `tools/` | 生成两份只差稀疏开关的工作流（同 seed）+ 对比耗时/显存/是否生效 + 抽帧并排成图 |
| `build(sparse_override=…)` | `tools/make_example_workflow.py` | 支持 widget 覆盖，供 A/B 工具复用 |
| 示例工作流重生成 | `templates/` | ⑦ 的 widget 从 14 个变 20 个，`verify_workflow.py` 校验 0 错误 0 警告 |

**勘误**：`432` 不是 32 的倍数，`768x432` 会被吸附成 **768×448**。
之前文档里的「768×432 省显存基线」应当读作 **768×448**。

**测试增量**：第 1 层 131→150、第 2 层 112→114、第 3 层 176→182、第 4 层 58→73（全绿）。
第 4 层新增：默认不调用 / 参数如实透传 / **失败必须降级且渲染继续** / min_tokens 门槛提醒。

### v0.3.0 —— 低显存运行能力

**目标**：让「漫画 → 5 秒动画片段」这条核心链路在 **8GB 显存 / 16GB 内存** 上真正跑通。

| 改动 | 文件 | 说明 |
|---|---|---|
| **分阶段（stage-major）流水线** ★ | `nodes/render_nodes.py` | 原来的逐镜循环是「每镜：编码→采样→解码」，三个阶段**同时**占用三个模型；现在改为「**先全部编码 → 再全部采样 → 再全部解码**」。这是低内存运行的前提：任何时刻只有一个大模型驻留 |
| **`unload_stages` 参数**（默认开） | `nodes/render_nodes.py` | 每个阶段结束卸载该阶段的模型。依据：`CLIP.encode` / `VAE.decode` 内部都会 `load_models_gpu([self.patcher])`，卸载后下次使用会自动重载，所以是安全的 |
| **新增 `core/mem.py`** | `core/mem.py` | 显存/内存快照（torch + Windows `GlobalMemoryStatusEx`）、对象→`ModelPatcher` 解析、选择性卸载（`free_memory(keep_loaded=…)`）、`gc.collect()`。**绝不抛异常**，拿不到 comfy 就降级 |
| **新增 ⑩ `H3MangaFreeMemory`** | `nodes/mem_nodes.py` | 手动释放节点。`trigger` 输入用于保证执行顺序（ComfyUI 只按依赖决定次序，无输入的节点可能被排到最前面） |
| **渲染报告加 `memory` 段** | `nodes/render_nodes.py` | 每阶段的显存/内存读数与人读曲线，用来验证「上一个模型是不是真卸了」 |
| **预检加资源检查** | `nodes/render_nodes.py` | 内存 ≥90% / 页面文件 <4GB / 关掉分阶段卸载 → 采样前就警告 |
| **新增 `tools/wb_memwatch.py`** | `tools/` | 零依赖双曲线监视器，用于区分「显存不足 / 内存被杀 / CUDA 驱动失败」三种「自动重启」 |

**测试增量**：第 1 层 108→131、第 2 层 103→112、第 3 层 175→176、第 4 层 48→58（全绿）。
第 4 层新增「分阶段卸载」正向断言 + **反向验证**（关掉开关后各阶段必须标注「未卸载」）。

**实测环境约束**（决定了上面的设计）：

| 项 | 实测 |
|---|---|
| R2V UNET / 文本编码器 | 19.53 GB / 13.93 GB（合计 33.5 GB） |
| 物理内存 / 显存 | 16 GB / 8 GB |
| 运行时占用（启动后未跑图） | 内存 91%、显存 7284/8188 MiB、GPU 100% |

### v0.2.1 —— 首跑崩溃修复

**现象**：节点 15 `H3MangaEpisodeRender` 首次运行就抛

```
H3ApiError: 整集没有任何镜头渲染成功（计划 1 镜，失败 1 镜）。
首个失败原因：[s01] H3ApiError: 官方 MiniMaxH3ReferenceToVideo 执行失败：
RuntimeError: shape '[1, -1, 3, 1152, 32]' is invalid for input of size 36864
```

**根因**（已用真实 `comfy.utils.common_upscale` + 真实官方节点双向确认）：
参考图交给官方 R2V 时是 **3 维 `[H,W,C]`**，而官方按 `h, w = img.shape[1], img.shape[2]`
读 —— 它假定 `[B,H,W,C]`。于是 `[768,1152,3]` 被读成「h=1152(宽)、w=3(通道数)」：

| 环节 | 算出来是什么 |
|---|---|
| `scale = min(1, sqrt(1344*768/(3*1152)))` | `1.0` |
| `tw = max(32, round(3/32)*32)` | `32`（畸形！） |
| `th = max(32, round(1152/32)*32)` | `1152` |
| `_resize(img[:1], 32, 1152)` 后 | `[1, 1152, 32]` = 36864 个元素 |
| `common_upscale` 末尾 `reshape((1,-1,3,1152,32))` | 需要 110592 个 → **炸** |

**修法**（三处，都补齐了测试）：

| 改动 | 文件 | 说明 |
|---|---|---|
| 新增 `as_ref_image()` / `as_ref_video()` | `core/image_ops.py` | 把任意形态升成官方要求的 4 维；非例行修正（灰度 / RGBA / 多张 / 多余维）记入 warnings |
| `shot_ref_layout` 出图即升维 | `core/refs.py` | 唯一出参考图的地方，面板图 / 角色图 / 风格图 / 参考视频全部过一遍；非张量直接报错（**不静默丢槽位**，否则 `<Picture i>` 会整体前移） |
| 出站守卫 + 报错补形状诊断 | `core/h3_api.py` | 防止有人绕过 `shot_ref_layout`；错误信息里列出每个槽位的实际 shape，并点明 reshape 报错是维度问题 |
| 空提示词提前拦 | `core/h3_api.py` | 官方收到空 prompt 不报错、只出无意义画面，改成明确报错 |
| 碎条分镜格预检警告 | `nodes/render_nodes.py` | 切格切出 `min(w,h) < 8px` 的条带时提前警告（作为 `<Picture 1>` 没有画面信息） |

**测试增量**：第 1 层 94→108、第 3 层 169→175（新增真实 `common_upscale` 与真实官方 R2V
端到端）、第 4 层 39→48（新增线上报错逐字复刻 + 反向验证）。

### v0.2.0 —— 代码审核修复（上一轮）

按三个维度（质量/兼容性/冗余）自审后修的项，全部有测试锁住：

**必须修**
| 项 | 问题 | 修法 |
|---|---|---|
| `plan_diff()` 破坏性副作用 | 内部调 `build_render_tasks` 没传 refs，会把已装配好的 `shot.picture_tags` **清空**、`ref_image_keys` 截断（实测） | `build_render_tasks` 默认 `mutate=False`（内部 clone），需要写回才显式传 `mutate=True` |
| 整集全失败返回空批次 | 下游 CreateVideo 报的错与真实原因无关 | 直接抛中文错误 + 带首条失败原因 + 接线排查提示 |
| `requires-comfyui = ">=0.3.60"` | 明显偏低（实际要 H3 节点 + `io.Custom`/`io.Autogrow`） | 改 `>=0.38.0`；`requires-python` 对齐 ComfyUI 下限 `>=3.9` |
| `negative` 全链路无效 | 官方 R2V **没有 negative 输入**，字段纯装饰 | 新增 `negative_mode`：`append`（默认，拼进正向）/ `ignore` |

**建议修**
| 项 | 问题 | 修法 |
|---|---|---|
| `build_render_tasks(board=…)` | 死参数（函数体只读 `s.board_image_index`） | 删参数 |
| 不检查空提示词 | 漏接「提示词装配」时静默出废画面 | `_preflight` 报错并提示检查接线 |
| `torch.zeros` 没传 `device` | `--gpu-only` 下 device mismatch | 收口到 `core/image_ops.py`，`device` 跟源张量 |
| `_concat_audio` 裸 except | **静默丢掉整条音轨** | 把原因带出去并写进预检警告 |
| 采样器/调度器白名单硬编码 | 凭空砍掉用户的选择 | 改用 `comfy.samplers.SAMPLER_NAMES` / `SCHEDULER_NAMES` |
| 中文标签双轨 | `core/ui.py` 的标签表基本没被用，`on_error` 缺标签显示英文 | 节点统一走 `_NL` / `_OL` / `_L`；补 `on_error` / `negative_mode` |
| `_pad_stack` 与 `_concat_frames` 重复 | 同一套 padding 逻辑写两遍 | 抽 `core/image_ops.py`，两处共用 |
| 手动框套用到所有页 | 多页输入时坐标错位 | 支持 `pageN: x,y,w,h` 前缀 |
| **`normalize_images` 把 AUDIO 当容器拆包** | AUDIO 本身是 dict，被当成 Autogrow 拆成 `[waveform, 32000]` → **参考音频一直是坏的** | 识别媒体 dict，不拆包 |
| `audios` / `videos` 是死输入 | 节点上有，但布局不带、渲染不传 | 视频/音频池进入 `shot_ref_layout` 并真的传给官方节点 |

**冗余清理**：`merge_packs` / `reserve_leading`（上一版错误设计遗留）/ `render_report` /
`DIALOGUE_MODES` 硬编码 / `RefEntry.image_count` / `ui.label_node|label_output` 全部删除；
`nodes/*.py` 的 20+ 未使用导入清空；`from ._io_types import *` 改成显式导入。

## 11. 来源声明

参考模式与标签契约（`<Picture i>` / `<Video k>` / `<Audio j>`、17n+5 帧网格、
9/3/3 参考上限、`ref_image_size` 语义）来自 ComfyUI 官方
`comfy_extras/nodes_minimax_h3.py` 与本机实测；本插件为独立实现，
**不复制任何官方文件正文**，全部通过调用官方节点的 `execute()` 复用其能力。
