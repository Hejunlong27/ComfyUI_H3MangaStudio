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

**14 个节点**，分六层。数据沿 `REFS / PLAN / BOARD / VLM` 四个自定义类型单向流动。

```
素材层              语义层(VLM)         编排层                        前置层          采样/输出
───────────────────────────────────────────────────────────────────────────────────────────
漫画页 ─► ①切分镜格 ─┬─────────────────────────────────────────────┐
                    │  ┌─► ⑬ VLM 漫画解读 ─► ⑭ VLM 转分镜计划 ─┐  │
                    └──┤        （看图出分镜，可跳过）           │  │
参考图 ─► ②参考包 ─────┴────────────────────────────────────────┘  │
                       └─► ③分镜清单加载 ─► ④分镜绑定 ─► ⑤提示词装配 ┤
                                                                   ├─► ⑫-A 镜头出栈 ─► 官方多参节点 ─► 你的采样链
                                                                   ├─► ⑫-B 单镜条件（直出 positive/latent）
                                                                   └─► ⑦整集渲染 ─► CreateVideo ─► SaveVideo
       ⑥单镜条件（调试）   ⑧一致性审计（体检）   ⑨计划预览（只读）   ⑩显存/内存释放
```

| # | 节点 | 作用 | 关键输出 |
|---|---|---|---|
| ① | **H3 漫画页切分镜格** | **非白连通域法**切格（可切回投影法）+ 自动 / 手动框 / 混合三种模式 + 出口去重叠 + **多边形分镜格（斜切版面）** | `H3_MANGA_BOARD` + 格图批次 + 报告 + **坐标网格预览**（★ 四路输出在示例工作流里已分别接 `PreviewImage`×2 / `PreviewAny` / `SaveImage`） |
| ② | **H3 参考包** | 角色设定卡 / 风格基准图 / 参考音频 → 有序参考包（**一角色一图**） | `H3_MANGA_REFS` + 全量清单 |
| ③ | **H3 分镜清单加载** | JSON 文件 / 手写块文本 → 结构化计划；也可 `source=board/hybrid` **从分镜格自动出镜** | `H3_MANGA_PLAN` + 报告 |
| ④ | **H3 分镜绑定** | 计划 × 分镜格绑定（按 panel 号 / 按顺序 / 不绑） | 已绑定计划 + 绑定报告 |
| ⑤ | **H3 提示词装配** | `@角色` → `<Picture i>`、运镜展开、风格注入、台词包裹 | 带最终提示词的计划 + 预览 |
| ⑥ | **H3 单镜条件**（调试） | 只构建单镜 conditioning + latent，接官方五件套做单镜调参 | `CONDITIONING` + `LATENT` |
| ⑦ | **H3 整集渲染 ★** | **批量量产核心**：内部逐镜循环，跑完一整集 | 帧序列 + 音频 + 渲染报告 |
| ⑧ | **H3 一致性审计** | 跑之前的体检：缺图、参考图太少、序号溢出、槽位漂移 | 分层报告 + 是否通过 |
| ⑨ | **H3 计划预览** | 只读预览，节点上挂前端面板显示「机器实际拿到的提示词」 | 文本 + 元数据 JSON |
| ⑩ | **H3 显存/内存释放** | 把模型从显存与内存里卸载掉，输出释放前后用量对比 | 释放报告 |
| ⑫-A | **H3 镜头出栈 ★** | **插件的终点**：把某一镜拆成 `prompt` + 9 个参考图槽 + 宽/高/帧数，直接喂**官方多参节点** | `STRING` + `IMAGE×9` + `INT×3` + 槽位对照 |
| ⑫-B | **H3 单镜条件（直出）** | 省事版：输入同 A，再给 `clip/vae/audio_vae`，直接出 `positive` + `latent` | `CONDITIONING` + `LATENT` + 槽位对照 |
| ⑬ | **H3 VLM 漫画解读 ★** | **看图出分镜**：一页漫画 → 页型 / 格数 / 每格内容 / 角色 / 台词 / 运镜 / 时长，**不需要手写 ③** | `H3_MANGA_VLM` + 分镜文本 + 报告 |
| ⑭ | **H3 VLM 转分镜计划** | 把 ⑬ 的结果转成 `PLAN`（**纯本地、不调 API**，画幅/时长/镜数可反复调参不花钱） | `H3_MANGA_PLAN` + 分镜文本 + 报告 |

**三个自定义类型之外再加一个 `H3_MANGA_VLM`** —— 它是 ⑬→⑭ 之间的中间产物（未落盘的解读结果）。

**为什么是「一个节点出一整集」（⑦）而不是工作流层循环**
ComfyUI 的工作流本身没有 for 循环。可行的落地方式就是**循环写在节点内部**，
工作流层面只呈现为几根线 —— 既满足「工作流模式」的心智模型，
又不用依赖执行图展开或子图。

**⑫-A 与 ⑦ 的区别（重要）**：⑫-A 是「**前置接口**」—— 插件只负责把漫画变成
官方节点要的输入，采样/解码交回你自己的工作流；⑦ 是「**一键整集**」—— 循环写在
插件内部，方便但不透明。**三版示例工作流**分别对应不同路线（见 §4）。

---

## 2. 所需输入参数

### 2.1 参考图（`H3_MANGA_REFS`）

| 输入 | 说明 |
|---|---|
| `character_names` | 逗号/换行分隔。**第 N 个名字 ↔ 第 N 个图槽**。一角色一图（默认）时每个角色填**一个**名字 |
| `character_images` | Autogrow 图槽（≤9）。**一角色一图**：每个角色一张「四视图 / 角色设定卡」。例：`纲吉, Reborn` + 2 张卡 |
| `character_describes` | 每行一个角色的外观锁定描述（行序 = 角色首次出现顺序）。**比纯标签稳得多** |
| `style_name` / `style_images` | 风格基准图，会出现在每个镜头的参考里（占 1 个 `<Picture i>`） |
| `audio_names` / `audios` | 独立参考音频（BGM / 音效 / 配音样本，≤3）。分镜里可用 `@名字` 引用成 `<Audio j>` |
| `video_names` / `videos` | 参考视频（≤3）。**一个槽位 = 一段多帧视频**，不拆帧；官方要求 ≥5 帧 |
| `video_audios` | 与 `videos` **按下标一一对应**的配套音轨（官方按键名尾号配对） |
| `max_ref_images` | 参考图上限，默认 9（官方硬上限）。降低可省显存与时间 |
| `name_source` | `slot` = 用你填的名字做 id / `index` = 强制 `ref1`/`ref2` |
| `one_image_per_character` | **默认开** —— 一个角色只锁一张参考图；同名重复写只保留第一张，其余忽略 + warning。关掉 = 旧行为（同名重复按顺序吃多个图槽，用于角色换装/多形态） |

### 2.2 分镜（`H3_MANGA_PLAN`）

两种来源，同一套字段语义。**块文本**（对编剧友好）：

```
# episode: Reborn 第30卷 目标283
# style: manga_anime
# default_size: 1344x768
=== s01 ===
panel: 0
duration: 4
camera: tilt_up
action: 黄昏的住宅区街景，电线杆与交错的电线横过天空
=== s02 ===
panel: 1
shots: 2                       ← ★ 这一格展开成 2 个镜头（一格多镜）
duration: 5                    ← 是**每镜**的秒数（该格合计 10s），不是整格平分
characters: 纲吉
camera: slow_push_in
content: 纲吉从门后走出 | 抬眼看向镜头
action: @纲吉 推开半掩的门，从门后走出 | @纲吉 抬眼看向镜头，瞳孔收紧
speaker: 纲吉
dialogue: 门轴吱呀一声。 | 这一刻终于还是到来了。
```

| 字段 | 含义 |
|---|---|
| `panel` | 绑定第几格（对应 board.panels 的 index） |
| `shots` | ★ **这一格展开成 N 个镜头**（默认 1）。见下方「一格多镜」 |
| `duration` | 秒。**会自动对齐到 H3 的 17n+5 帧网格**（5s → 124 帧）。`shots: N` 时是**每镜**的时长 |
| `characters` | 出场角色（决定参考图槽位） |
| `camera` | 运镜短名，展开成镜头语言 |
| `action` | 画面动作。**用 `@角色id` 引用角色**，插件负责替换成 `<Picture i>` |
| `speaker` / `dialogue` | 说话人与台词（H3 原生支持中文对白，直接引号包裹，**不翻译**） |
| `style` | 覆盖全局风格 |
| `size` | 覆盖画幅（自动对齐 32 倍数） |
| `seed` | 覆盖派生 seed |
| `negative` / `extra` | 单镜负向 / 追加提示词 |

字段名支持中英混写与别名（`面板` / `时长` / `角色` / `运镜` / `画面` / `台词` / `说话人` /
`镜头数`…），未知字段只警告不报错 —— 不会因为一行写错就让整集失败。

#### ★ 一格多镜（`shots: N`）—— 镜头数 ≠ 格数

漫画的「格」是版面单位，「镜头」是时间单位。一个分镜格里画了两个时间相
（大格先全景再特写、格内插格）时，就是「**1 格 = N 镜**」。

两条路径，可混用：

| 路径 | 用法 | 场景 |
|---|---|---|
| 批量 | ③ 的 `auto_shots_per_panel`（默认 1） | 整卷都习惯「一格两镜」 |
| 逐格 | 文本里的 `shots: N` | 只有某几格需要拆 |

- 展开出的镜**共用同一个 `panel_index`** → 绑同一张格图当参考锚点，
  各自独立出 `<Picture i>` 序号（**没有串号风险**，序号仍由 `shot_ref_layout()` 唯一生成）
- `action` / `content` / `dialogue` 用 `A | B | C` **分段一一对应**各镜；
  段数不足 → 重复最后一段；段数超出 → 并回最后一段（**不丢你写的字**）
- 第 1 个镜沿用原 id（`s02`），后续是 `s02_2` / `s02_3`
  → **不写 `shots` 时 id / seed 与旧版逐字段一致**（可放心断点续跑）
- `shots: 0` / 非数字 → 按 1 镜 + warning；`shots: 99` → 截断到 8 + warning

`shot_table()` 会把它们标成 `panel 1 1/2`、`panel 1 2/2`，并给一行汇总。

### 2.3 风格与角色一致性设置

| 手段 | 参数 | 说明 |
|---|---|---|
| **角色一致性** | `lock_consistency` | 自动生成「这几张参考图必须是同一个人」的锁定句 |
| | `include_describe` | 把 `character_describes` 的外观描述拼进提示词（**比纯标签稳**） |
| | 参考图数量 | **一角色一图（默认）**：每个角色 1 张「四视图 / 角色设定卡」。审计提示角色有没有参考图 |
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
character_names="纲吉, Reborn"
character_images=[card_纲吉, card_Reborn]      ← 每个角色一张「四视图 / 角色设定卡」
        ↓  建条目
RefPack.entries = [
  RefEntry(id="纲吉",  kind="character", images=[card_纲吉],   priority=0, describe="棕发橙眼…"),
  RefEntry(id="Reborn", kind="character", images=[card_Reborn], priority=1, describe="黑西装婴儿…"),
  RefEntry(id="基准风格", kind="style", images=[style],        priority=900),
]
```

> **★ 一角色一图（默认）**：每个角色只锁 **一张** 参考图，用「四视图 / 角色设定卡」
> 把正/侧/背/表情并到同一张图里。两个理由：
> ① **歧义** —— 多张独立定妆图之间只要有发型/服装/角度的细微差异，模型就会摇摆；
> ② **稀缺** —— 官方参考图上限 9 张，还要与分镜格锚点、风格基准图共享额度。
> 同名重复写只保留第一张，其余忽略并给 warning（**不静默丢**）。
> 确实需要多图（角色换装 / 多形态）时，把 `one_image_per_character` 关掉。

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
| 角色档案 | ②参考包的 `character_names` / `character_describes` / `one_image_per_character` | ✅ |
| 提示词结构 | ⑤的 `template`（占位符 `{scene}{refs}{action}{camera}{style}{dialogue}{extra}`） | ✅ |
| 运镜词库 | `templates/camera_presets.json`（填路径到 `camera_json`） | ✅ |
| 风格词库 | `templates/style_presets.json`（填路径到 `style_json`） | ✅ |
| 切格行为 | ①的 11 个参数；切不准就用 `manual_boxes` 手填框 | ✅ |
| 绑定策略 | ④的 `bind_mode`（精确 / 顺序 / 不绑）+ `panel_offset` | ✅ |
| 采样与加速 | ⑦的 `sampler_name` / `scheduler` / `steps` / `shift_video` / `shift_audio` / `lora_name` | ✅ |
| 输出画幅与时长 | 分镜里的 `size` / `duration`（自动对齐） | ✅ |
| **VLM 页级解读** | ⑬的 `base_url` / `model` / `characters`（角色闭集）/ `max_edge` / `strict_coverage` / `inspection_fallback` | ✅ |
| **VLM 结果 → 分镜计划** | ⑭的 `size_mode` / `size_preset` / `size_scale` / `duration_scale` / `force_shots` / `include_non_narrative`（**不调 API，随便调**） | ✅ |
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

1. 从 `user/default/workflows/` 打开**三版之一**：

   | 文件 | 路线 | 适合 |
   |---|---|---|
   | `H3MangaStudio_①整集渲染版` | ⑦ 一键跑完 | 想省事，循环在插件内部 |
   | `H3MangaStudio_②官方节点接线版` | ③ 手写清单 → ⑫-A → 官方多参节点 | 要自己掌控采样链 |
   | **`H3MangaStudio_③VLM全自动版`** | **① → ⑬ → ⑭ → ④ → ⑤ → ⑫-A → 官方多参节点** | **从漫画图直接跑到动画，不手写分镜** |

   —— 源文件在 `templates/`，用 `tools/make_example_workflow.py` 重新生成
2. 改 3 个 `LoadImage` 的文件名（1 张漫画页 + **每个角色 1 张设定卡**）
   —— 例子里那两张 `char_*_card.png` 是**占位图**，用 `tools/make_placeholder_cards.py` 生成，请换成你自己的
3. 改②参考包的 `character_names` / `character_describes`（**一个角色一个名字**）
4. **③版额外一步**：给 ⑬ 填 `api_key_env`（默认读环境变量 `VLM_API_KEY`）与 `base_url`/`model`；
   ⑬ 不接 ① 的 `board` 也能跑（会自切格）
5. ③/② 版：把 ③ 或 ⑭ 的输出接 ④；若用 ③版，第 4 步起不用手写分镜
6. 先把采样链的 `steps` 调低（或挂 turbo LoRA）跑通**单镜**，再放量

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
| **出片跟漫画/参考图毫无关联** | 先查**官方节点的 `prompt` 是不是真的接上了**。它带 `default=` → 在 ComfyUI 里是 **widget**，要被连线必须在 `node.inputs` 里带 `widget` 标记（见 §8）。prompt 没接上 → 官方拿到空 prompt → 没有文本指令，画面就是随机的。跑 `tools/verify_workflow.py` 会直接点名这类断线 |
| 界面上某条线拖不到官方节点上 / 拖完又弹回 | 同上：**目标端是 widget 型输入**，缺 `widget` 标记就没有可接的槽。**请重新打开 `templates/` 里的最新工作流**，不要手工改造旧版本 |
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

## 7. 测试（四层 + 工作流自检，全绿）

```bash
cd <ComfyUI根>
.venv/Scripts/python.exe custom_nodes/ComfyUI_H3MangaStudio/tests/run_all.py
```

| 层 | 文件 | 验什么 | 结果 |
|---|---|---|---|
| 1 纯逻辑 | `test_pure_logic.py` | 帧对齐 / 清单解析 / **逐镜重排** / 提示词装配 / 绑定 / 审计 / **切格（两种框架 + 格内白带 + 去重叠 + 框线焊接告警 + 自动开运算与守卫 + 多边形分镜格）** / **官方画幅规则 + 从 board 自动出镜 + hybrid 合并 + 一格多镜 `shots: N`** / **一角色一图（含反向验证）** / **工作流布局矩阵契约** / 参考素材张量契约 / 资源管理与卸载原语 / **VLM 三条硬约束 + 繁简归一化 + 子串兜底 + 格号换算 + 分段契约 + 块文本渲染** | 330/330 |
| 2 schema 桩 | `test_schema_stub.py` | 输入名 ↔ `execute` 形参、optional 落位、Combo default、中文标签、返回项数 == 输出数 | 154/154 |
| 3 真实加载 | `test_real_load.py` | ComfyUI 自己的加载器 + Autogrow 真实形态 + **官方节点签名漂移探针** + **真实 `common_upscale` / 真实官方 R2V 端到端** | 198/198 |
| 4 契约 | `test_render_contract.py` | **桩替换 + 真跑 `execute()` + 反向验证 + 检查器自检** + **分阶段卸载** + **VLM 全局格号 / 分段 / 禁标签 / 越界拒收** | 97/97 |
| **5 工作流自检** | `tools/verify_workflow.py` | `templates/*.json` ↔ **真实 `INPUT_TYPES()`**：widget 序列 / 值类型 / 下拉选项 / **领域规则**（如 `CLIPLoader.type` 必须 `minimax`）/ `widgets_values_named` / 连线 `target_slot` / **widget-input 类型自洽** | 3 个工作流全通过 |

**第 5 层为什么必须存在**：`widgets_values` 的数量对、下拉值也合法，
但**语义完全错**（`CLIPLoader.type = stable_diffusion`）时，
前四层一层都拦不住 —— 这类错误只在出片现场表现为「画面与漫画毫无关联」。
第 5 层把「模板 ↔ 真实节点定义」的一致性焊进回归，才真正堵住这一类。

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
| ★ **widget ≠ input**（ComfyUI 序列化契约） | 带 `default=` 的输入在 ComfyUI 里是 **widget**，界面上的连接受 `node.inputs` 数组控制。要让连线接上去，`inputs` 里必须有一条 `{"name":X,"type":T,"link":N,"widget":{"name":X}}`；且**被转换的 widget 仍在 `widgets_values` 里占位**（顺序不变）、转换条目排在**所有非 widget input 之后**（三段的第 ③ 段）。权威样例：官方模板 `video_minimax_h3_r2v.json`。漏掉它 → `links` 里有线但 `target_slot` 为 `null` → **ComfyUI 静默丢弃**，参数退回 widget 默认值（v0.73 事故：官方多参节点的 `prompt` 没接上 → 空 prompt → 出片与漫画毫无关联）。`tools/make_example_workflow.py` 已自动转换；`tools/verify_workflow.py` 已加硬检查 + 反向验证 |
| 空槽返回 `None` 穿 Autogrow | 官方 `execute` 里是 `if img is None: continue` → 逻辑上安全（9 图槽连满时用不到的槽就这么处理）。**但真实运行尚未实测**；若卡在这里，退路是「空槽输出 1×1 白图，且必须排在最后」（否则序号会错） |

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
│   ├── vlm.py                  # ★ VLM 逻辑唯一真源（解码→校验→变换→块文本）
│   ├── h3_api.py               # 官方 H3 能力适配（唯一 comfy 边界）
│   └── ui.py                   # 中文文案唯一真源
├── nodes/
│   ├── _io_types.py            # 自定义 IO 类型单例
│   ├── asset_nodes.py          # ①②
│   ├── plan_nodes.py           # ③④⑤⑧⑨
│   ├── mem_nodes.py            # ⑩
│   ├── shot_nodes.py           # ⑫-A / ⑫-B（前置出栈：9 槽 或 直出 CONDITIONING）
│   ├── vlm_nodes.py            # ★ ⑬⑭（VLM 漫画解读 → 分镜计划）
│   └── render_nodes.py         # ⑥⑦
├── js/h3manga_preview.js       # 轻量只读预览面板（serialize: false）
├── locales/zh/nodeDefs.json    # i18n（由 tools/gen_i18n.py 生成）
├── templates/                  # 示例工作流 / 分镜模板 / 运镜与风格词库
├── data/characters_example.json
├── tools/
│   ├── gen_i18n.py             # 从 schema 生成 i18n 与前端 labels
│   ├── make_example_workflow.py# 从真实 INPUT_TYPES 生成三版示例工作流
│   ├── vlm_shotlist.py         # ★ 离线批量跑 VLM（dry/probe/run/compare/retxt），逻辑来自 core/vlm.py
│   ├── wb_memwatch.py          # ★ 显存/内存双曲线监视（诊断「自动重启」）
│   ├── sparse_ab.py            # ★ 稀疏注意力 A/B（生成对比工作流 + 对比报告 + 抽帧并排）
│   ├── verify_workflow.py      # 逐节点核对工作流（widgets / 槽位 / 类型 / 值 / 领域规则）
│   ├── preview_layout.py       # ★ 把工作流画成布局示意图（改布局后先看图，不必重启 ComfyUI）
│   └── make_placeholder_cards.py # ★ 生成「四视图 / 角色设定卡」占位图（示例工作流依赖）
└── tests/                      # 五层测试（4 层断言 + 工作流自检）+ run_all.py
```

## 10. 变更记录

> **版本号规则**：自 v0.71 起，每次改动以 **0.01 为步长**递增（v0.71 → v0.72 → …）。
> 历史上的 v0.3.0 / v0.4.x / v0.7.0 沿用当时的语义化写法，不再回溯改名。

### v0.80 —— 切格预览接线 + LoRA 区 + 「只出一个镜头」根因（本轮）

**1. 切格结果现在看得见了** —— 根因是 ① 的三个输出**一直悬空**（`links=[]`）。
① 本来就有 4 个输出，只是工作流没接。三版统一新增**预览组**：

| 节点 | 接 ① 的 | 呈现方式 / 位置 |
|---|---|---|
| `PreviewImage` | `图像` | **节点上即时看**切出的每个分镜格（存 `<ComfyUI>/temp/`，随时可删） |
| `PreviewImage` | `坐标网格预览` | 手填 `manual_boxes` 时按它数坐标（每 100px 一条刻度） |
| `PreviewAny` | `报告` | **格数与全部 warning 在这里** —— 排查「为什么只切出 1 格」的入口 |
| `SaveImage` | `图像` | **落盘**：`<ComfyUI>/output/h3manga/panels_00001_.png` …（可长期留档） |

**2. 新增 LoRA 区**：`UNETLoader → LoRA 加速 → LoRA 轮盘位 → 消费者`

| 节点 | 默认值 | 说明 |
|---|---|---|
| `LoraLoaderModelOnly`「LoRA · 加速（turbo）」 | `minimax_h3_fl2v_turbo_8step_v1.0` @1.0 | 加速 LoRA，固定 |
| `LoraLoaderModelOnly`「LoRA · 轮盘位」 | 同上 @0.0，**旁路(mode=4)** | 想加风格/角色 LoRA 就**按 Ctrl+B 解除旁路**再换文件 |

★ **为什么不是「轮盘节点」**：本机 **rgthree-comfy 没有注册任何节点**（Power Lora Loader
不可用），efficiency 的 `LoRA Stacker` 产出 `LORA_STACK` 只被它自己的大 `Efficient Loader`
消费 —— 可用的只有核心的 `LoraLoaderModelOnly`，所以用两级串联实现「加速 + 可换」。

★ **为什么用旁路而不是填 `none`**：核心 `LoraLoaderModelOnly` 的 `lora_name` 下拉
**没有 `"none"` 这个值**（只有 loras 目录里的文件名），填进去会被校验器判为
「不在选项表内」。ComfyUI 里「关掉一个 LoRA 槽」的**唯一标准做法就是旁路**。

★ 同时把 ⑦ 的 `lora_name` 改为 `none` —— 否则会「外置 LoRA 区 + ⑦ 内部 LoRA」双份叠加。

**3. 「只输出一个镜头」的根因（两条叠加）**

| # | 原因 |
|---|---|
| 1 | ②/③ 版里 `⑫-A` 的 `shot_index` **固定为 1** → 每次只出第 1 镜 |
| 2 | **ComfyUI 的执行缓存** —— 参数没变时整图走缓存，第二次运行结果与第一次**逐字节相同** |

★ **本质**：ComfyUI 的工作流**没有循环**；官方多参节点一次只吃一镜
（`prompt`/`width`/`height`/`length`/`ref_images` 都是单镜的）。`dynamic_prompts`
只是**前端提示**，不是批量机制。

**解**：②/③ 版新增**批量组**（与「单镜」分支并列）：
`④ → ⑦ 批量出镜（max_shots=0）→ CreateVideo → SaveVideo` —— ⑦ 内部逐镜循环，
**一次顺序跑完全部镜头**。另外 `⑫-A/⑫-B` 的报告顶部现在会显式写
「★ 本节点一次只输出 1 个镜头 / 本计划共 N 镜 / 本次第 k 镜 / 想一次跑完用 ⑦」。

**产物**：① 24 节点/9 组 · ② 35 节点/11 组 · ③ 38 节点/12 组；
三版 `verify_workflow` **0 错 0 警**；五层回归 **335 / 154 / 97 / 198 + 3 工作流自检** 全绿。

### v0.79 —— VLM 入库：把「看图出分镜」做成 ComfyUI 节点（上一轮）

**目标修正**：上一轮我把 VLM 当成离线工具去产某几卷成品；**这一轮把它做成插件节点** ——
目标是在 **ComfyUI 工作流里**从漫画图直接跑到动画，而不是产出某一卷的视频。

| 新增 | 内容 |
|---|---|
| `core/vlm.py` | VLM 逻辑的**唯一真源**：解码 → 校验 → 变换 → 块文本。**离线工具与节点共用同一份**（避免两份实现漂移） |
| `nodes/vlm_nodes.py` | ⑬ `H3MangaVLMScribe`（看图出分镜）+ ⑭ `H3MangaVLMPlan`（转 `PLAN`，**不调 API**） |
| `H3_MANGA_VLM` 类型 | ⑬→⑭ 的中间产物类型 |
| 第三个示例工作流 | `H3MangaStudio_vlm_auto_workflow.json`（28 节点 / 48 连线）：**漫画图 → ① → ⑬ → ⑭ → ④ → ⑤ → ⑫-A → 官方多参节点 → 采样 → 出片** |
| `tools/vlm_shotlist.py` 瘦身 | 逻辑改为 `from core.vlm import ...`，不再自带一份 |

**关键设计**：
1. **⑬ 可接 `board`** —— 接上就用 ① 的**全局格号**（与 ④ 的索引语义一致），
   不接就自切格并自算偏移。★ 实测验证：两页跑出 `0,1,2` / `3..8` 连续递增。
2. **⑭ 不花钱** —— 画幅 / 时长缩放 / 强制镜数 / 是否保留非叙事页，都可以反复调。
3. **三条硬约束由代码二次校验**（不信模型）：禁 `<Picture i>` 标签 / 角色闭集 / panel 在范围内；
   **不合规整页拒收**。
4. **内容审核回退阶梯**（`INSPECTION_FALLBACKS`）：被云审核拦下时换模型或缩图，实测救回 2/3。

**顺带修掉一个「双向自洽的真错」**：生成器与校验器各有一份**连线类型白名单**，
新增 `H3_MANGA_VLM` 时两边都漏改 → ⑭ 的 `vlm` 被序列化成 widget-input（值 null）。
现改为**本质规则**（`_is_widget_type()`：只有 INT/FLOAT/STRING/BOOLEAN/COMBO 是 widget，
其余全是连线），并删掉两份白名单；另加一条**不依赖分类函数**的独立守卫
（带 widget 标记的条目，其类型必须是 widget 型）—— 已反向验证能精确检出。

五层 **330 / 154 / 97 / 198 + 3 个工作流自检** 全绿。i18n **14 个节点**。

### v0.78 —— 第30卷全卷实跑：206/207 页，端到端验证通过（上一轮）

用 `qwen3.5-omni-plus` 跑完整第30卷（207 页），**206 成功 / 1 失败**，
用时 **38 分 24 秒**，**634,701 tokens**。

**结果**：873 格（① 全卷应切 878，缺口正好是失败页的 5 格）→ 叙事页 190 页 / 809 格 / **1009 镜**，
VLM 给的总时长 **89.0 分钟**。**16 个非叙事页自动跳过**（省 64 格）。

**★ 端到端验证通过**：用 ③ 的真实解析器吃 271 KB 的 `shotlist.txt` →
**解析出 1009 镜 = JSON 叙事页镜数**，panel 唯一号 809 = 叙事页格数。
**逐页对号 206/206，0 页错**。

**★ 内容审核战绩**：3 页被拦（048 / 052 / 112），回退阶梯救回 2 页，
**1 页最终失败（0.5%）**，全部有记录、不静默。

**★★ 生产现实（必须知道）**：1009 镜 × 672×384（0.36 GB/镜，内存安全）
→ 按 3~5 min/镜，**出片要 50~84 小时**。瓶颈在渲染，不在 VLM。
→ **整卷全量出片不现实，必须按叙事价值筛选**（这正是 `page_kind` + `content` 的用途）。

**本轮追加修的三处**：

1. ★ **角色名「全名 vs 简称」子串兜底** —— 模型常写全名（`泽田纲吉`）而闭集是简称（`纲吉`）
   → 主角参考图被**整批剔除**（本卷 ≥7 次）。改为取「最长的子串匹配」。
2. ★ **全量警告写入 JSON**（`warnings` / `errors` / `failed_pages`）——
   原来只往控制台打前 10 条，**事后无法分析**（本卷 22 条只能看到 10 条）。
3. ★ 新增**免 API 的 `retxt` 子命令**（见 v0.77）—— 已用它把 873 格中的 870 格重编号为全局号。

**★ 用 `source=file` 而不是 `inline`**：271 KB 贴进 widget 容易出问题；
`source=file` + 绝对路径更稳、可重复加载。已实测该文件能被 ③ 正确解析。

**额度**：该模型累计约 **71%**（710k / 1M），再跑一整卷需先增购或降字段。

### v0.77 —— VLM 实测：字段分级采信 + 三个生产级修补（上一轮）

用阿里云百炼（`qwen3.5-omni-plus`，国内站 `dashscope.aliyuncs.com/compatible-mode/v1`）
对第30卷 12 页跑了完整实测，含 **7 模型横向对照 + 12 次稳定性实验**。
详细报告在仓库外：`H3MangaStudio_VLM实测报告_v0.77.md`。

**★ 最重要的一条：VLM 输出必须按字段分级采信**

| 可信 | 不可信 |
|---|---|
| `page_kind` 页型（12/12 跨模型一致） | **`shots` 一格几镜**（同页 3 次跑出 3/4/5 镜） |
| **格数**（与几何切格 **12/12 一致**） | `confidence` 把握度（判错时也报 0.90+） |
| `content` / `dialogue` / `characters` / `camera` | — |

- ★ **降 temperature 到 0 无效** → 不稳定来自**判据歧义**，不是采样随机性
- ★ **几何切格在这批页上与 VLM 完全一致** → 推翻了「几何 37% 漏切」的旧结论（那只出现在极端页）
- ★ 因此 `shots>1` 单独进 `review.txt`：**51 格里人工只需看 5 格**

**★ 三个生产级修补**：

1. **内容审核回退阶梯** —— 战斗页会被云服务拦（`DataInspectionFailed`）。
   实测**与模型/分辨率相关且非确定**（主模型 1600/900 被拦、600 通过；换 `qwen3-vl-plus`
   保持 1600 也通过）→ 被拦时自动换模型/缩图重试，并把「哪页走了回退」写进警告。
2. **4xx 不重试** —— 原来审核拦截会白等 3 次共 90 秒，现在 12 秒返回明确原因。
3. **繁简归一化 + 严格覆盖度** —— `omni-flash` 会写繁体名（`綱吉/獄寺/藍波`）导致
   角色被整批剔除；还会**静默漏读格**（018 页只读 8 格里的 4 格就交结果）。
   新增 `norm_name()` / `--char-alias` / 默认拒收漏读页（`--allow-partial` 可放宽）。

**★ `page_kind` 免费解决 L0 页型准入**：12 页里 4 页是封面/设定/目录页，
默认**跳过不出镜头**（几何完全看不出这一点）。注意 `010` 那类「扉页+小格」混合页有边界误判。

**成本实测**：**2,819 tokens/页**（图片 1,449 + 文本/输出 1,370）、**10.5 s/页**。
→ **1M 额度 ≈ 355 页 ≈ 1.7 卷**；全 31 卷（6449 页）≈ **18.2M tokens**。

**工具新增 CLI**：`--char-alias` / `--include-non-narrative` / `--allow-partial` /
`--no-inspection-fallback`；默认模型改为 `qwen3.5-omni-plus`。

**★ 用户拍板两项（同日）**：

1. **额度按模型独立计算** —— 每个大模型各有 1M token，**分开算**。
   → 主模型跑完一整卷（207 页 ≈ 583k）后累计约 **66%**，单卷可行；
   但**双跑投票**（用一致性稳住 `shots`）会超池，故维持单跑。
2. **`shots`（一格几镜）保留 VLM 的判断** —— 用户判断「是帮我」。
   → `shots: N` 照写进 `shotlist.txt`（工具默认行为，**未改代码逻辑**）；
   `review.txt` 保留但**降级为「建议抽查」**（不再称「必须确认」）；
   不认同某格时删掉那一行 `shots:` 即可。
   ★ `<Picture i>` 序号**仍只由 ⑤ 注入**，与 `shots` 无关（红线不变）。

**★★ 追加修复：格号必须换算成「全卷全局编号」（第二个「静默错」级问题）**

`①` 的 `Panel.index` 是**整批全局递增**的（`idx = 0` 在页循环**之外**），
`④ bind_board` 用 `panel_index + offset` 去索引 `board.panels`；
而 VLM 只按**本页**编号 0..k-1 ⇒ 第 2 页起直接照抄就会**绑错格图**
（或超范围 → 静默退化成纯文生视频）。**两类都不报错。**

- 新增 `panel_offsets()`：算每页全局偏移（切格确定，可精确算出）→ `panel` 用全局号，
  JSON 同时留 `panel_local`
- 新增 `--panel-offset`（① 只吃部分页时用）/ `--local-panel-index`（回旧行为）
- ★ 新增**免 API 子命令 `retxt`**：把已有 JSON 重编号并重写 txt/review，**零成本**修正旧结果
  （已验证**幂等**）
- ⚠️ 前提：工具与 ① 必须用**同一套切格参数**（默认都是 `frame=component` + `deoverlap=True`）

### v0.76 —— LLM/VLM 优先架构：单卷试点工具（上一轮）

**用户拍板**：走**云端**多模态 API、**接受漫画图上传**；其余按建议 ——
切格**不交给 VLM**（只做「格数 / 一格几镜」修正）、`dialogue` 由用户把关、
先做**单卷试点**、节点精简留待后续。

> 架构判断的依据见 `H3MangaStudio_LLM优先架构评估_v0.75.md`：
> 用 096/097 两页真实漫画做的对照实验 —— **几何管「框」，VLM 管「语义」**。
> 两条红线：① **`<Picture i>` 序号绝不能由 LLM 生成**；② **切格不交给 VLM**
>（不是能力问题，是它不稳定/不可复现，而锚点要求同页两次结果一致）。

**新增 `tools/vlm_shotlist.py`**（零第三方依赖，OpenAI 兼容协议）：

| 子命令 | 作用 |
|---|---|
| `dry` | **不发请求**：切格 / 图片编码 / token 量级 / 风险页一览 |
| `probe` | 只跑 1 页，验证连通性与返回格式 |
| `run` | 批量跑 → `shotlist.json` + `shotlist.txt`（**直接贴进 ③**） |
| `compare` | VLM 结果 **vs** 几何切格的对照报告 |

**三条硬约束由代码二次校验**（不信任模型输出）：
① 不得出现 `<Picture i>` / `<Video k>` / `<Audio j>` 标签（序号由 ⑤ 提示词装配注入）；
② `characters` 必须落在**角色闭集**内；③ `panel` 必须在 ① 实际切出的格号范围内。
**不合规就整页拒收**并记录错误，绝不静默放过。

**新增 `templates/vlm_shotlist_sample.json`** —— 黄金样例（096/097 的真实读图结果），
既是工具的 few-shot 样例，也是字段契约的验收标准。

**实测（第30卷 前 14 页）**：

- 207 页/卷；整卷外推 **~695k tokens**（图片 ~2156/页 + 文本 ~1200/页）
- 14 页里 **4 页被标记风险**（几何失效 / 只切出 1 格 / 疑似跨页）→ 正是 VLM 的补位点
- 对照报告：`096`「格数一致，但 VLM 拆出 4 镜（多 1）」、`097`「完全一致」

### v0.74 —— 全盘排查：堵住「值取错但不报错」的一整类缺陷（上一轮）

**用户复报**：官方节点的 `prompt` 仍然没连上，要求全盘检查。

**排查方法**：不再采信上一轮的结论，改为与**官方模板逐字段 diff** ＋
**本机真实 `INPUT_TYPES()` 交叉验证**。

★ 结论：**widget-input 的写法与官方模板完全等价**（prompt 线在 JSON 层面是通的，
索引差异只是因为本插件连满了 9 个参考图槽）。但 diff 顺带暴露出一批
**更隐蔽、危害更大**的问题 —— 它们全都「不报错」：

| # | 缺陷 | 实际值 | 应为 |
|---|---|---|---|
| 1 | ★★ **`CLIPLoader.type`** | **`stable_diffusion`** | `minimax` |
| 2 | `SaveVideo.format` / `codec` | `null` / `null` | `auto` / `auto` |
| 3 | `LoadImage` 第二个控件名 | `image`（**与真实 widget 重名**） | `upload`（值恒为 `"image"`） |
| 4 | `widgets_values_named` | 全缺 | 前端 1.49+ 的标配字段 |
| 5 | `control_after_generate` | 自造名 `__control_after_generate__` | `control_after_generate` |
| 6 | Autogrow 槽 `label` | 缺 | 补上 |
| 7 | `outputs.slot_index` | 多写 | 官方没有此字段 |

★ **第 1 条是头号问题**：H3 的文本编码器是 **Qwen3VL**，`type=stable_diffusion`
会让 CLIP 按 SD 的结构去解析 Qwen3VL 权重 → **prompt 根本没法被正确编码** →
**这一条单独就足以造成「出片与漫画毫无关联」，而且不报错**。
根因很直白：`WIDGET_OVERRIDE` 常量**定义了却从未被任何代码引用**，
而那个节点的标题还写着「CLIP · type 必须是 minimax」—— 标题在提醒，代码没执行。

另外两个「生成器自己崩」的问题（互为因果）：CLI 打印用硬下标 `n['widgets_values']`
→ `KeyError` → **第二个工作流根本生成不出来**；删 `slot_index` 后 `o["slot_index"]`
残留引用 → 又崩一次。

**校验工具加三道网（这才是「不再出现类似问题」的关键）**

旧版只查结构不查值。而 `stable_diffusion` 数量对、选项也合法，
于是**一路溜到出片现场，任何测试都没拦住**。

| 网 | 抓什么 |
|---|---|
| 重建**完整 widget 序列** | `control_after_generate` + `upload` 若不计入，从 seed 之后**按位置取值全部错位**（实跑立刻抓出 `max_shots 值却是 'randomize'`） |
| **值内容**校验 | 类型 / 下拉选项合法性 / 值为 `null` |
| **领域规则表** `SEMANTIC_RULES` | 「值合法、后端不报错、画面却全错」—— 首条：`CLIPLoader.type` 必须 `minimax` |

**接入回归**：`tests/run_all.py` 新增 **第 5 层 · 工作流自检**
（`templates/*.json` ↔ 真实 `INPUT_TYPES` + 领域规则）。

**反向验证**（证明网不是摆设）：人为注入四类缺陷，`verify` 全部精确点名，
其中包括一直追的那条 ——
`连线 16：target_slot 为 null —— ⑫-A[0] → 官方多参节点 接不上 → ComfyUI 会丢弃这条线`。

**结果**：两版 `verify` **0 错 0 警**；五层回归 **292 / 136 / 73 / 198 + 工作流自检** 全绿。

> 完整分析（含 diff 表、反向验证证据、复现命令、仍未确证的点）
> 见 `H3MangaStudio_官方接线全盘排查_v0.74.md`。

### v0.73 —— 官方接线版断线修复（上一轮）

**用户实测报告**：⑫-A 的宽/高/帧数连不到官方节点；勉强连上后跑了一条，**画面与漫画毫无关联**。

**三条根因**（完整分析见 `H3MangaStudio_官方接线断线诊断与修复_v0.73.md`）：

1. **widget ≠ input**。官方 `MiniMaxH3ReferenceToVideo` 的 `prompt` / `width` / `height` / `length`
   都带 `default=` → 在 ComfyUI 里它们是 **widget**。要让连线接上去，节点的 `inputs` 里必须补一条
   `{"name":"width","type":"INT","link":N,"widget":{"name":"width"}}`
   —— 这是前端「widget → input」的序列化形式（权威样例：官方模板 `video_minimax_h3_r2v.json`）。
   生成器没有这个机制 → 四条线的 `target_slot` 是 `null` → **ComfyUI 直接丢弃**。
2. **`prompt` 的默认值被写成了 `null`**（官方模板是 `""`）。prompt 又接不上 → 官方拿到空 prompt
   → 无文本指令 → **画面随机**。这就是「与漫画毫无关联」的直接原因。
   ★ ⑫-A 侧的 prompt 实测**完全正常**（分镜格锚点 / 角色锁定句 / 运镜 / 台词 / 负向词都在），
   差的只是**接线**这一步。
3. **校验工具从未检查过这个文件**：`verify_workflow.py` 的路径是硬编码的，
   且 `sys.argv = [sys.argv[0], "--cpu"]` 会把命令行参数**抹掉** → 它永远只校验整集渲染版。
   「官方节点接线版」的 0 错误 0 警告，其实是整集渲染版的成绩单。

**修复**

| 文件 | 改动 |
|---|---|
| `tools/make_example_workflow.py` | widget 元素带上**类型**；STRING 型默认值 `null` → `""`；新增 `widget_inputs` 计算与生成（转换条目排在**最后**，与官方模板一致）；`widgets_values` 保持完整 |
| `tools/verify_workflow.py` | 支持命令行指定文件（不给参数 = 校验 `templates/` 下**全部**工作流，用内容探测排除数据文件）；`inputs` 改**三段式**校验；★ 新增硬检查「被连线的 widget 必须带 `widget` 标记」；`target_slot is None` 单独报错并写明后果 |
| 模板 | 9 个参考图槽**连满**（空槽 ⑫-A 输出 `None`、官方会跳过，**不影响 `<Picture i>` 序号**；以后加图不用改接线） |

**验证**
- 两版工作流 `verify_workflow.py` **0 错误 / 0 警告**（现在会同时校验两个文件）
- ★ **反向验证**：人为把修复回退后，校验精确点名 4 条断线
  （`target_slot 为 null —— … 接不上 → ComfyUI 会丢弃这条线`）
- 四层 **292 / 136 / 73 / 198 全绿**

### v0.72 —— 一格多镜：`shots: N`（上一轮）

**解决的问题**：用户实测 096 页「③ 只显示 2 个镜头，实际是 4 个」——
根因是「**镜头数 ≠ 格数**」：一个分镜格里可能画了两个时间相。
之前只能靠 hybrid 手写多行，但同一 panel 的多行会被 merge **逐行覆盖**，实际永远只剩 1 镜。

**新增两条路径**（可混用）

| 路径 | 用法 | 场景 |
|---|---|---|
| 批量 | ③ 的 `auto_shots_per_panel`（默认 1） | 整卷都习惯「一格两镜」 |
| 逐格 | 文本里的 `shots: N` | 只有某几格需要拆 |

- 展开出的镜**共用同一个 `panel_index`** → 绑同一张格图当参考锚点
- `action` / `content` / `dialogue` 可用 `A | B | C` **分段一一对应**各镜
- **`duration` 语义 = 每个镜头各自的秒数**（该格合计 N × duration），不是整格平分
- 第 1 个镜沿用原 id（`s02`），后续 `s02_2` / `s02_3` → **不写 `shots` 时 id / seed 完全不变**（向后兼容）

**修掉一个真 bug**：`merge_plan_with_text` 改为**按 panel 分组整组替换**。
原来按行逐条替换，同一 panel 的第二行会把第一行刚放进去的挤掉 —— hybrid 下「一格多镜」永远只剩 1 镜。

**新增字段**：`Shot.sub_index` / `Shot.sub_total`（`shot_table()` 显示 `panel x/N` 并给汇总行）。

**容错**：`shots: 0` / 非数字 → 按 1 镜 + warning；`shots: 99` → 截断到
`MAX_SHOTS_PER_PANEL = 8` + warning；分段不足 → 重复最后一段；分段超出 → 并回最后一段（**不丢字**）。

**实测**（真实 096 页，切出 3 格）：

```
  #     panel             时长（填 → 帧 → 实际）        画幅  内容
  0         0         5s → 124 帧 → 5.17s     672x384  黄昏街景，电线与电线杆
  1     1 1/2         4s → 107 帧 → 4.46s     672x384  短发女生笑脸
  2     1 2/2         4s → 107 帧 → 4.46s     672x384  饭桌上的对话
  3         2         4s → 107 帧 → 4.46s     672x384  早餐特写
★ 其中 2 镜来自「一格多镜」（1 个分镜格被拆成多镜，标为 `panel x/N`）。这些镜共用同一张格图当参考锚点。
```

总镜数 3 → **4**，总时长 15s → **18.54s**。

**测试**：四层 **292 / 136 / 73 / 198 全绿**。新增 24 条断言，含
**反向验证**（不写 `shots` 时 id / index / sub 与旧版逐字段一致）+ hybrid 的 4 个场景。

### v0.71 —— 矩阵式工作流布局 + 一角色一图（上一轮）

**① 工作流模板改成矩阵式布局**（`tools/make_example_workflow.py`）

从「按列纵向堆叠」换成**两层矩阵**：

| 层 | 规则 |
|---|---|
| **组内矩阵** | 每个 group 内的节点按 `INNER_COLS=2` 列折行；行高 = 该行最高节点，同行节点**顶对齐** → 同一列的第 N 行永远在同一水平线上 |
| **组间矩阵** | 每个 group 视为网格中的一个单元，按 `OUTER_COLS=4` 折行；**同列左对齐**（列宽取该列最宽者）、同行顶对齐 |

★ **组间用蛇形（boustrophedon）折行**：第 1 行左→右，第 2 行右→左。
为什么不用「每行都从左开始」：数据流是 `模型→输入图→素材→编排→执行→输出`，
若每行都左对齐，第 1 行末尾（编排）与第 2 行开头（执行）会落在**对角两端**，
读不出流程的连续性。蛇形让「上一行的最后一格」紧邻「下一行的第一格」，
流程像 Z 字一样连贯 —— **这才是「一眼看出层级与关联关系」的关键**。

实测（两版工作流）：组间 **2 行 × 4 列**网格、节点两两重叠 **0** 对、
group 框重叠 **0** 对、孤儿节点 **0** 个、节点归属冲突 **0** 个。

**② 角色参考图：一角色一图（默认）**

`core/refs.py::build_ref_pack(one_image_per_character=True)`：同名重复写**只保留第一张**，
其余忽略 + warning（**不静默丢**）。约定每个角色用一张「**四视图 / 角色设定卡**」。

两个理由：
1. **歧义** —— 多张独立定妆图之间只要有发型/服装/角度的细微差异，模型就会摇摆，反而不如一张清晰四视图稳；
2. **稀缺** —— 官方参考图上限 9 张，还要与分镜格锚点、风格基准图共享，一个角色吃 3 张会挤掉别人。

**能力保留**：`one_image_per_character=False` 完整回到旧行为（同名重复按顺序吃槽，用于角色换装/多形态）。

**新增工具**
- `tools/preview_layout.py` —— 把工作流画成布局示意图 PNG（改布局后**不用重启 ComfyUI** 就能先看对不对）
- `tools/make_placeholder_cards.py` —— 生成两张「四视图设定卡」**占位图**放进 `input/`（示例工作流依赖，且不属于插件仓库）

**测试**：四层 **266 / 136 / 73 / 197 全绿**（原 228/136/73/196）。
新增：一角色一图的建包 + 序号断言、**反向验证**（关掉开关必须回到多图）、
以及**布局矩阵契约**（组间二维网格 / 单元唯一 / 组内多列 / 无重叠 / 无孤儿）。

### v0.7.0 —— 退回「官方多参节点前置接口」（上一轮）

**定位调整**：插件不再以「内部跑完采样」为主路径，改成
**官方 `MiniMaxH3ReferenceToVideo` 的前置接口** —— 插件负责
「漫画图 + 人物参考图 → 分镜 → 参考图对齐 → H3 提示词 → 出栈」，
**采样 / 放大 / 二采 / 解码全部交回工作流**。

**这条路为什么成立**（核对 0.38.2 源码）：官方多参节点**自己就输出**
`positive(CONDITIONING) + Latent`；它的 `ref_images` 是
`Autogrow(prefix="ref_image_", max=9)`（每槽一张 IMAGE），且执行体里是
`if img is None: continue` —— **空槽会被自动跳过**，所以「按序号出 9 个口、
用不到的口返回 None」是安全的。

**新增节点**

| 节点 | 作用 |
|---|---|
| **⑫-A `H3MangaShotOut`** | 单镜出栈：`prompt` + **9 个参考图槽**（按序号）+ `宽/高/帧数` + `槽位对照表`。显示名直接写明「参考图 N → 官方 `ref_image_(N-1)`」 |
| **⑫-B `H3MangaShotCond`** | 省事版：内部调用官方节点，直出 `positive` + `latent`，少接 9 根线 |

**修掉的两个 bug**

1. **③ 只显示 2 镜 / 没有可手填的时长画幅**：③ 原本是**纯文本**节点，压根不读图。
   现在新增 `board` 输入与 `source=board / hybrid` —— **从 ① 的分镜格自动出镜**
   （一格一镜），文本只改要改的那几镜；新增 `default_duration` 与
   `size_mode` / `size_preset` / `size_scale`（**手动选画幅**）；文本新增 `content:`。
2. **出片 4s ≠ 报告 9.6s**：报告里的 9.6s 是**计划总时长**，
   而示例工作流 ⑦ 的 **`max_shots=1`** 只渲了第 1 镜。现在 ⑦ 的报告显式写出
   「只渲染前 N 镜 / 计划共 M 镜」与「实际输出 = X 帧 = Y 秒」（`scope` +
   `output_duration_sec`），③ 的镜头表也把「**填的秒数 → 帧 → 实际秒数**」三段链显示出来：

   ```
   duration: 4  →  96 帧  →  对齐 17n+5  →  107 帧 = 4.458 s
   duration: 5  →  120 帧 →  对齐         →  124 帧 = 5.167 s
   合计 231 帧 = 9.625 s     ← 这就是报告里的「9.6 s」
   ```

**画幅：严格对齐官方 `adapt_canvas`**（短边 768 / 面积 ≤ 768×1344 / 逐轴 32 倍数）

| 比例 | 官方规范画幅 | ×0.5（8GB 机先跑通） |
|---|---|---|
| 16:9 横屏 | 1344×768 | 672×384 |
| 4:3 横屏 | 1024×768 | 512×384 |
| 1:1 方形 | 768×768 | 384×384 |
| 9:16 竖屏 | 768×1344 | 384×672 |

另修一处真 bug：官方是**逐轴独立** round 到 32，极端比例下两轴同时向上取整会让面积
**超过**上限（实测 2.39:1 → 1568×672 = 1,053,696 > 1,032,192），已补面积硬约束。

**两版示例工作流**（`templates/`，`verify_workflow` 均 **0 错误 / 0 警告**）

| 文件 | 用途 |
|---|---|
| `H3MangaStudio_example_workflow.json` | **整集渲染版**：⑦ 一键跑完（`max_shots=0`） |
| `H3MangaStudio_official_nodes_workflow.json` | **官方节点接线版**：⑫-A → 官方多参节点 → 你自己的采样链（26 节点 / 41 连线） |

四层 **228 / 136 / 73 / 196 全绿**（原 189 / 116 / 73 / 191）；节点数 **10 → 12**。

**本期没做**（刻意留到后续）：`shots: N`（一格多镜）、`⑭ H3MangaPrepPack`（①+②合体）。

### v0.4.3 —— 多边形分镜格：斜切版面终于能表达（上一轮）

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
