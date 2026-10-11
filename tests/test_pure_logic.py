"""第 1 层：纯逻辑测试（不需要 ComfyUI，只需要 numpy）。

跑法（用 ComfyUI 自带的独立运行时，自带 numpy / cv2）：
    "<安装根>/standalone-env/python.exe" tests/test_pure_logic.py

覆盖：
  * H3 帧数 17n+5 对齐、画幅 32 倍数对齐
  * 分镜清单解析（块文本 / JSON / 别名 / 容错）
  * ★ 参考图**逐镜密集重排**（本插件最核心的正确性保证）
  * 提示词装配（@角色 → <Picture i>、场景锚点、台词）
  * 计划编排（seed 派生稳定性、绑定、审计）
  * 漫画页切格（合成页 + 已知边界）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import core, PKG_DIR  # noqa: E402

C = core()
T = C.types
PB = C.prompt_build
SP = C.script_parse
RF = C.refs
PL = C.plan
PS = C.panel_split

PASS = 0
FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------
section("H3 帧数 / 画幅对齐")
# --------------------------------------------------------------------------
ok(T.align_frame_count(5) == 5, "align(5) == 5")
ok(T.align_frame_count(121) == 124, "align(121) == 124（5 秒）")
ok(T.align_frame_count(124) == 124, "align(124) 保持不变")
ok(T.frames_for_seconds(5) == 124, "5 秒 → 124 帧")
ok(T.frames_for_seconds(15) == 362, "15 秒 → 362 帧")
ok(T.frames_for_seconds(0.1) == 5, "极短时长兜底到 5 帧")
ok(all(T.align_frame_count(n) % 17 == 5 for n in range(5, 400)),
   "任意长度对齐后都满足 17n+5")
ok(T.align_canvas(1344, 768) == (1344, 768), "1344x768 已是 32 倍数")
w, h = T.align_canvas(1000, 1000)
ok(w % 32 == 0 and h % 32 == 0, "非 32 倍数画幅被对齐")
w, h = T.align_canvas(4096, 4096)
ok(w * h <= T.H3_MAX_PIXELS * 1.05, "超像素上限时被等比压回")

# --------------------------------------------------------------------------
section("分镜清单解析")
# --------------------------------------------------------------------------
BLOCK = """
# episode: 测试集
# style: manga_anime
=== s01 ===
panel: 0
duration: 5
characters: tsuna, reborn
camera: slow_push_in
action: @tsuna 从门后走出
dialogue: 我回来了。
speaker: tsuna

=== s02 ===
面板: 1
时长: 3
角色: reborn
运镜: close_up
画面: @reborn 抬头
台词: 是9代首领寄来的。
"""
plan = SP.parse_text(BLOCK)
ok(plan.episode == "测试集", "块文本 meta 解析")
ok(len(plan.shots) == 2, f"块文本解析出 2 镜（实得 {len(plan.shots)}）")
ok(plan.shots[0].length == 124, "5 秒镜头 length=124")
ok(plan.shots[1].length == 73, "3 秒镜头 length=73（17n+5）")
ok(plan.shots[1].panel_index == 1, "中文别名「面板」被识别")
ok(plan.shots[1].camera == "close_up", "中文别名「运镜」被识别")
ok("reborn" in plan.shots[1].characters, "中文别名「角色」被识别")
ok(plan.shots[0].speaker in plan.shots[0].characters,
   "说话人自动补进角色列表")

JSON_SCRIPT = '{"episode":"J","shots":[{"id":"a","duration":5,"panel":0},' \
              '{"id":"a","duration":5,"panel":1}]}'
plan_j = SP.parse_text(JSON_SCRIPT)
ok(len(plan_j.shots) == 2, "JSON 清单解析")
ok(plan_j.shots[1].id != plan_j.shots[0].id, "重复 id 被自动去重")

plan_bad = SP.parse_text("# episode: E\n=== s1 ===\n不存在的字段: 1\nduration: 5\n")
ok(len(plan_bad.shots) == 1 and any("未知字段" in w for w in plan_bad.warnings),
   "未知字段只警告不报错（容错）")

# --------------------------------------------------------------------------
section("★ 参考图逐镜密集重排（防串号的核心）")
# --------------------------------------------------------------------------
import numpy as np  # noqa: E402

def fake_img(tag):
    return np.full((4, 4, 3), tag, dtype=np.float32)

# ---- ★ 一角色一图（默认）----
# 每个角色只锁**一张**参考图：同名重复写只保留第一张，其余忽略 + warning。
# 约定用「四视图 / 角色设定卡」把正/侧/背/表情并到一张图，避免多图之间的歧义。
pack_single = RF.build_ref_pack(
    character_names="纲吉, 纲吉, Reborn",
    character_images=[fake_img(1), fake_img(2), fake_img(3)],
    character_describes="棕发橙眼，黑色学生制服\n黑西装婴儿，卷鬓角",
    style_name="基准风格",
    style_images=fake_img(9),
)
ok(len(pack_single.entries) == 3,
   f"参考包：2 角色 + 1 风格（实得 {len(pack_single.entries)}）")
ok(len([e for e in pack_single.entries if e.kind == "character"]) == 2,
   "同名重复被合并成一个角色")
ok(len(pack_single.entries[0].images) == 1,
   f"★ 一角色一图：纲吉只保留 1 张（实得 {len(pack_single.entries[0].images)}）")
ok("一角色一图" in pack_single.source_note,
   f"★ 被忽略的重复图槽必须发 warning，不能静默丢（实得 {pack_single.source_note[:48]!r}）")
ok(pack_single.entries[0].id == "纲吉",
   f"中文 id 原样保留（实得 {pack_single.entries[0].id}）")
ok(pack_single.entries[2].kind == "style", "风格条目排在最后")

# ★ 反向验证：关掉开关必须完整回到旧行为（多图各自吃槽）
pack_multi = RF.build_ref_pack(
    character_names="纲吉, 纲吉, Reborn",
    character_images=[fake_img(1), fake_img(2), fake_img(3)],
    one_image_per_character=False,
)
ok(len(pack_multi.entries[0].images) == 2,
   f"★ 反向验证：关掉开关后纲吉拿回 2 张（实得 {len(pack_multi.entries[0].images)}）")
ok(pack_multi.source_note == "",
   f"关掉开关后不该再报「一角色一图」（实得 {pack_multi.source_note!r}）")

# 名字少于图槽时必须给出可操作提示（两种约定下的提示不同）
pack_mismatch = RF.build_ref_pack(character_names="纲吉",
                                  character_images=[fake_img(1), fake_img(2)])
ok("四视图" in pack_mismatch.source_note and "不要用重复名字" in pack_mismatch.source_note,
   f"一角色一图：数量不匹配时提示改用四视图（实得 {pack_mismatch.source_note[:60]!r}）")
pack_mismatch_multi = RF.build_ref_pack(character_names="纲吉",
                                        character_images=[fake_img(1), fake_img(2)],
                                        one_image_per_character=False)
ok("重复写一遍" in pack_mismatch_multi.source_note,
   "关掉开关后提示改回「把名字重复写一遍」")

# 一角色一图下的序号：分镜格 <Picture 1>、纲吉 <Picture 2>、Reborn <Picture 3>
single_ids = [e.id for e in pack_single.entries if e.kind == "character"]
single_style = [e.id for e in pack_single.entries if e.kind == "style"][0]
lay_single = RF.shot_ref_layout(pack_single, characters=single_ids,
                                has_panel=True, style_ids=[single_style])
ok(lay_single["picture_map"][single_ids[0]] == [2],
   f"一角色一图：纲吉占 1 个槽 <Picture 2>（实得 {lay_single['picture_map'][single_ids[0]]}）")
ok(lay_single["picture_map"][single_ids[1]] == [3],
   f"一角色一图：Reborn 紧接其后 <Picture 3>（实得 {lay_single['picture_map'][single_ids[1]]}）")
ok(len(lay_single["ref_image_keys"]) == 4,
   f"一角色一图省下槽位：格+2角色+风格 = 4 槽（实得 {len(lay_single['ref_image_keys'])}）")

# ---- 多图路径（关掉开关）继续用于验证「逐镜密集重排」----
pack = RF.build_ref_pack(
    character_names="纲吉, 纲吉, Reborn",
    character_images=[fake_img(1), fake_img(2), fake_img(3)],
    character_describes="棕发橙眼，黑色学生制服\n黑西装婴儿，卷鬓角",
    style_name="基准风格",
    style_images=fake_img(9),
    one_image_per_character=False,
)
ok(len(pack.entries) == 3, f"参考包：2 角色 + 1 风格（实得 {len(pack.entries)}）")
ok(len(pack.entries[0].images) == 2, "（多图路径）纲吉拿到 2 张定妆图")
ok(pack.entries[0].id == "纲吉", f"中文 id 原样保留（实得 {pack.entries[0].id}）")
ok(pack.entries[2].kind == "style", "风格条目排在最后")

# 角色名 → id 用 unicode 转义，保证 @引用可用
ids = [e.id for e in pack.entries]
ok(all(" " not in i for i in ids), f"id 不含空格：{ids}")

char_ids = [e.id for e in pack.entries if e.kind == "character"]
style_id = [e.id for e in pack.entries if e.kind == "style"][0]

# 场景 A：镜 1 出场「纲吉 + Reborn」，有分镜格
layoutA = RF.shot_ref_layout(pack, characters=[char_ids[0], char_ids[1]],
                             has_panel=True, style_ids=[style_id])
# 分镜格 + 纲吉 2 张 + Reborn 1 张 + 风格 1 张 = 5 个槽位
ok(list(layoutA["ref_image_keys"]) == [f"ref_image_{i}" for i in range(5)],
   f"槽位键序列连续（含分镜格位）：{list(layoutA['ref_image_keys'])}")
# 只传 has_panel（没有张量）时，ref_images 里不放 None，但槽位照占 ——
# 这是「装配/审计阶段」与「渲染阶段」能对齐键序列的关键。
ok(list(layoutA["ref_images"].keys()) == [f"ref_image_{i}" for i in range(1, 5)],
   "无分镜格张量时 ref_images 跳过该槽位，但 ref_image_keys 仍保留它")
ok(layoutA["picture_map"][char_ids[0]] == [2, 3], "分镜格占 <Picture 1>，纲吉的两张图是 <Picture 2>/<Picture 3>")
ok(layoutA["picture_map"][char_ids[1]] == [4], "第二个角色是 <Picture 4>")

# 场景 B：镜 2 只出场「Reborn」，有分镜格 —— 序号必须重排！
layoutB = RF.shot_ref_layout(pack, characters=[char_ids[1]],
                             has_panel=True, style_ids=[style_id])
ok(layoutB["picture_map"][char_ids[1]] == [2],
   "同一角色在只出场自己的镜头里变成 <Picture 2>（重排生效）")

# 反向验证：如果不重排（用全局池），序号会不同 —— 这正是会串号的写法
global_slots = pack.resolve_slots()
ok(global_slots["picture_map"][char_ids[1]][0] != 2,
   "全局池序号与逐镜序号确实不同（证明重排是必要的，不是同义反复）")

# 无分镜格时，第一个角色回到 <Picture 1>
layoutC = RF.shot_ref_layout(pack, characters=[char_ids[0]], has_panel=False,
                             style_ids=[style_id])
ok(layoutC["picture_map"][char_ids[0]] == [1, 2], "无格图时第一个角色从 <Picture 1> 起排")

# 溢出保护
many = RF.build_ref_pack(character_names=",".join(f"c{i}" for i in range(20)),
                         character_images=[fake_img(i) for i in range(20)])
many_slots = many.resolve_slots()
ok(len(many_slots["ref_images"]) <= T.H3_MAX_REF_IMAGES,
   f"参考图被截到 ≤{T.H3_MAX_REF_IMAGES} 张")
ok(any("上限" in w for w in many_slots["warnings"]), "溢出时给出警告")

# --------------------------------------------------------------------------
section("提示词装配")
# --------------------------------------------------------------------------
plan2 = SP.parse_text("""
# episode: 装配测试
# style: manga_anime
=== s01 ===
panel: 0
duration: 4
characters: 纲吉, Reborn
camera: slow_push_in
action: @纲吉 从门后走出，@Reborn 坐在他肩上
speaker: 纲吉
dialogue: 这一刻终于还是到来了。
""")
# 绑一个分镜格，让「场景锚点」这条断言真正生效（不绑的话 <Picture 1> 会是角色）
board2 = T.Board(
    panels=[T.Panel(index=0, box=[0, 0, 100, 100], page_index=0)],
    images=[fake_img(7)], pages=1,
)
PL.bind_board(plan2, board2, mode="by_panel_index")

PB.compose_plan(plan2, pack, dialogue_mode="spoken", lock_consistency=True)
sh = plan2.shots[0]
ok(sh.prompt, "提示词非空")
ok("follow <Picture 1>" in sh.prompt, "场景锚点句引用了 <Picture 1>（分镜格）")
ok("@纲吉" not in sh.prompt and "@Reborn" not in sh.prompt,
   "@引用已被替换（不留字面量）")
ok('"这一刻终于还是到来了。"' in sh.prompt, "中文台词原样引号包裹，未翻译")
ok(sh.picture_tags.get(char_ids[0]) == "<Picture 2>",
   f"角色标签与逐镜布局一致：{sh.picture_tags}")
ok(sh.picture_tags.get(char_ids[1]) == "<Picture 4>",
   "第二个角色的标签把纲吉的两张图算进去了（<Picture 4>）")
ok("slow push in" in sh.prompt, "运镜短名被展开成镜头语言")

# 未命中引用必须留痕
plan3 = SP.parse_text("=== s1 ===\nduration: 5\ncharacters: 纲吉\naction: @不存在的角色 走过来\n")
PB.compose_plan(plan3, pack)
ok("@不存在的角色" not in plan3.shots[0].prompt, "未命中的 @引用被移除")
ok(any("没有这个条目" in w for w in plan3.shots[0].warnings),
   "未命中的 @引用产生警告")

# 自定义模板
plan4 = SP.parse_text("=== s1 ===\nduration: 5\ncharacters: 纲吉\naction: 走路\n")
PB.compose_plan(plan4, pack, template="STYLE[{style}] ACTION[{action}] CAM[{camera}]")
ok(plan4.shots[0].prompt.startswith("STYLE["), "自定义模板生效")

# --------------------------------------------------------------------------
section("计划编排")
# --------------------------------------------------------------------------
ok(PL.derive_seed(42, 0, "s01") == PL.derive_seed(42, 0, "s01"), "seed 派生稳定")
ok(PL.derive_seed(42, 0, "s01") != PL.derive_seed(42, 1, "s02"), "不同镜头 seed 不同")
ok(PL.derive_seed(42, 0, "s01") == PL.derive_seed(42, 9, "s01"),
   "★ 插入镜头（改 index）不影响同一 shot_id 的 seed —— 断点续跑才成立")

board = T.Board(
    panels=[T.Panel(index=i, box=[0, 0, 100, 100], page_index=0) for i in range(3)],
    images=[fake_img(i) for i in range(3)], pages=1,
)
plan5 = SP.parse_text("""
=== a ===
panel: 2
duration: 5
=== b ===
panel: 0
duration: 5
=== c ===
duration: 5
""")
PL.bind_board(plan5, board, mode="by_panel_index")
ok(plan5.shots[0].board_image_index == 2, "按 panel 精确绑定")
ok(plan5.shots[1].board_image_index == 0, "panel 可以乱序（不强制递增）")
ok(plan5.shots[2].board_image_index == 2, "没写 panel 时按顺序回退")

plan6 = SP.parse_text("=== a ===\npanel: 99\nduration: 5\n")
PL.bind_board(plan6, board, mode="by_panel_index")
ok(plan6.shots[0].board_image_index is None, "越界 panel 被解绑")
ok(any("超出范围" in w for w in plan6.warnings), "越界 panel 产生警告")

# 审计：故意制造序号不一致
plan7 = SP.parse_text("=== s1 ===\nduration: 5\ncharacters: 纲吉\naction: 走路\n")
PB.compose_plan(plan7, pack)
plan7.shots[0].picture_tags = {char_ids[0]: "<Picture 9>"}   # 人为破坏
aud = PL.audit(plan7, pack)
ok(not aud["passed"], "序号不一致 → 审计不通过")
ok(any(i["kind"] == "slot_mismatch" for i in aud["issues"]),
   "审计报出 slot_mismatch")

# 审计：缺图角色
plan8 = SP.parse_text("=== s1 ===\nduration: 5\ncharacters: 路人甲\naction: 走路\n")
PB.compose_plan(plan8, pack)
aud8 = PL.audit(plan8, pack)
ok(any(i["kind"] == "missing_ref" for i in aud8["issues"]), "审计报出 missing_ref")

# 任务展开
tasks = PL.build_render_tasks(plan2, pack, base_seed=7)
ok(len(tasks) == 1, "任务展开数量正确")
ok(tasks[0]["length"] == T.align_frame_count(tasks[0]["length"]), "任务帧数已对齐")
ok(tasks[0]["ref_image_keys"] == [f"ref_image_{i}" for i in range(5)],
   f"任务参考键密集：{tasks[0]['ref_image_keys']}")

# --------------------------------------------------------------------------
section("漫画页切格（合成页）")
# --------------------------------------------------------------------------
page = np.full((600, 400, 3), 255, dtype=np.float32)
# 2x2 四格：黑框白心，中间留白 gutter
page[20:270, 20:180] = 0.0
page[20:270, 220:380] = 0.0
page[330:580, 20:180] = 0.0
page[330:580, 220:380] = 0.0

boxes, warns = PS.split_page_boxes(page, mode="auto", reading_order="rtl")
ok(len(boxes) == 4, f"2x2 合成页切出 4 格（实得 {len(boxes)}）")
ok(boxes[0][0] > boxes[1][0], "rtl 顺序：第一格在右侧（日漫阅读序）")
ok(boxes[0][1] < boxes[2][1], "rtl 顺序：同列自上而下")

boxes_ltr, _ = PS.split_page_boxes(page, mode="auto", reading_order="ltr")
ok(boxes_ltr[0][0] < boxes_ltr[1][0], "ltr 顺序：第一格在左侧")

# 手动框通道
mb, mw = PS.parse_manual_boxes("10,10,100,100\n[200,10,100,100]\n[[10,200,100,100],[200,200,100,100]]")
ok(len(mb) == 4, f"三种手动框写法混用被解析（实得 {len(mb)}）")
ok(all(len(b) == 5 and b[0] == PS.PAGE_ANY for b in mb),
   "手动框归一成 [page, x, y, w, h]，无 page 前缀时 page=-1（套用到所有页）")
boxes_m, _ = PS.split_page_boxes(page, mode="manual", manual_boxes=mb)
ok(len(boxes_m) == 4, "manual 模式只用手动框")

boxes_h, _ = PS.split_page_boxes(page, mode="hybrid", manual_boxes=[[300, 300, 80, 80]])
ok(len(boxes_h) >= 4, "hybrid 模式叠加自动 + 手动")

# 已知边界（故意固化）：整页全黑 → 投影法失效，必须给出警告而不是静默出错
black = np.zeros((400, 400, 3), dtype=np.float32)
bb, bw = PS.split_page_boxes(black, mode="auto")
ok(any("墨迹占比过高" in w for w in bw), "整页全黑时给出「投影法可能失效」警告")

# 已知边界：纯白页 → 无墨迹，退化成 1 格（不是 0 格，避免下游空批次崩溃）
white = np.full((400, 400, 3), 255, dtype=np.float32)
wb, _ = PS.split_page_boxes(white, mode="auto")
ok(len(wb) <= 1, f"纯白页不会切出大量碎片（实得 {len(wb)}）")

# 整页全黑在两种框架下都要给警告（连通域法也会失效）
bb2, bw2 = PS.split_page_boxes(black, mode="auto", frame="component")
ok(any("墨迹占比过高" in w for w in bw2), "整页全黑：连通域法也给出警告")

# --------------------------------------------------------------------------
section("★ 关键回归：格内「贯穿白带」不该被当成装订线")
# --------------------------------------------------------------------------
# 构造一个格：黑框 + 两块内容，两块之间一条贯穿内宽的白带 ——
# 就是「竖排文字栏间距 / 天空 / 两个对话框之间」那种**格内空白**。
# 左侧留一条 20px 的黑「脊」把上下两块连起来（真分镜格的边框就是这样连的）。
inner = np.full((600, 400, 3), 255, dtype=np.float32)
inner[20:580, 20:380] = 0.0
inner[250:350, 40:380] = 255.0        # 白带：x 从 40 到 380，右边到底

b_proj, _ = PS.split_page_boxes(inner, mode="auto", frame="projection")
b_comp, _ = PS.split_page_boxes(inner, mode="auto", frame="component")
ok(len(b_proj) >= 2,
   f"投影法把这条格内白带当成了装订线（实得 {len(b_proj)} 格）—— 这就是已知过切")
ok(len(b_comp) == 1,
   f"★ 连通域法把它保持为 1 格（实得 {len(b_comp)}）—— 格内空白与画格同块")

# 反过来说：格与格之间的空白，两种框架都要能切开（不能矫枉过正）
two = np.full((600, 400, 3), 255, dtype=np.float32)
two[20:280, 20:380] = 0.0
two[340:580, 20:380] = 0.0            # 中间 60px 全宽白 = 真装订线
for fr in ("component", "projection"):
    bb, _ = PS.split_page_boxes(two, mode="auto", frame=fr)
    ok(len(bb) == 2, f"frame={fr}：真装订线仍被正确切开（实得 {len(bb)}）")

# --------------------------------------------------------------------------
section("分镜框去重叠（相邻框不得互相压着）")
# --------------------------------------------------------------------------
ov = [(0, 0, 100, 100), (90, 0, 100, 100)]
ok(PS.count_overlaps(ov) == 1, "重叠检测：构造的两个框被判为重叠")
fixed = PS.deoverlap_boxes(ov)
ok(PS.count_overlaps(fixed) == 0, f"去重叠后互不重叠（实得 {fixed}）")
ok(fixed[0][2] + fixed[0][0] <= fixed[1][0],
   "去重叠是沿中线分开：左框右边界 ≤ 右框左边界")
ok(PS.count_overlaps(PS.deoverlap_boxes(ov, iters=0)) == 1,
   "★ 反向验证：iters=0（等价于关掉去重叠）时重叠仍在")

# 出口保证：两条框架的自动结果都不能带重叠
for fr in ("component", "projection"):
    bb, _ = PS.split_page_boxes(page, mode="auto", frame=fr)
    ok(PS.count_overlaps(bb) == 0, f"frame={fr} 自动切格出口无重叠框")

# 制造重叠后再验证开关真的起作用（padding 放大 → 相邻框互相侵入）
bb_off, _ = PS.split_page_boxes(page, mode="auto", frame="projection",
                                padding_px=60, deoverlap=False)
bb_on, _ = PS.split_page_boxes(page, mode="auto", frame="projection",
                               padding_px=60, deoverlap=True)
ok(PS.count_overlaps(bb_off) >= 1,
   f"★ 反向验证：padding 放大且关掉去重叠 → 检出重叠（实得 {PS.count_overlaps(bb_off)}）")
ok(PS.count_overlaps(bb_on) == 0,
   "padding 放大但开着去重叠 → 仍无重叠")

# 框套框检测（不是重叠，是超集框）
ok(PS.count_contained([(0, 0, 200, 200), (50, 50, 50, 50)]) == 1,
   "框套框检测：小框完全落在大框内")
ok(PS.count_contained([(0, 0, 100, 100), (200, 0, 100, 100)]) == 0,
   "并排的框不算框套框")

# --------------------------------------------------------------------------
section("★ 关键回归：相邻格被细框线焊在一起 → 必须告警 + 可用开运算救")
# --------------------------------------------------------------------------
# 真实事故（Reborn_10_081）：一个大块 bbox 占页面 71%，但**像素只占 38%** ——
# 说明它是一张「细框线 + 重墨」织成的网，把好几个格连成了一块。
# 该页行方向最小非白率 0.097（整页找不到一条横向白缝），几何上无解。
# 这里用合成页复现：两块内容靠一条 2px 细线相连。
weld = np.full((800, 600, 3), 255, dtype=np.float32)
weld[20:300, 20:560] = 0.0            # 块 A
weld[400:700, 20:560] = 0.0           # 块 B
weld[20:700, 20:22] = 0.0             # ★ 2px 细线，把 A 和 B 焊在一起
for x0 in (40, 220, 420):             # 三个独立小格（把中位面积拉低）
    weld[720:790, x0:x0 + 130] = 0.0

wb_weld, ww_weld = PS.split_page_boxes(weld, mode="auto")
ok(len(wb_weld) == 5,
   f"★ 自动开运算逐级试到能分开（5 格：A、B、三个小格），实得 {len(wb_weld)}")
ok(not any("过大连通块" in w for w in ww_weld),
   f"自动调完不再报「过大连通块」（warns={ww_weld}）")
ok(any("自动开运算" in w for w in ww_weld), "报告里说明自动采用了哪个半径")

wb_open, ww_open = PS.split_page_boxes(weld, mode="auto", component_open=1)
ok(len(wb_open) == 5,
   f"component_open=1 用开运算打断细框线 → A/B 分开，共 5 格（实得 {len(wb_open)}）")
ok(not any("过大连通块" in w for w in ww_open),
   f"打断之后不再报「过大连通块」（warns={ww_open}）")

# 反向验证：关掉自动 + 半径 0 → 必须重新焊上并重新告警
wb_back, ww_back = PS.split_page_boxes(weld, mode="auto", component_auto_open=False)
ok(len(wb_back) == 4, f"★ 反向验证：关掉自动时重新焊成 4 格（实得 {len(wb_back)}）")
ok(any("过大连通块" in w for w in ww_back), "★ 反向验证：关掉自动时告警重新出现")

# 守卫：自动开运算不得靠「删格」来消除告警
# （真实事故 Reborn_10_134 / _108：半径一大，格被整个腐蚀掉，反而只剩 1 格）
wb_guard, _ = PS.split_page_boxes(weld, mode="auto")
wb_guard_off, _ = PS.split_page_boxes(weld, mode="auto", component_auto_open=False)
ok(len(wb_guard) >= len(wb_guard_off),
   f"★ 守卫：开运算只许细分、不许吞格（auto={len(wb_guard)} ≥ off={len(wb_guard_off)}）")

# --------------------------------------------------------------------------
section("成本估算 / 报告")
# --------------------------------------------------------------------------
cost = T.estimate_cost(plan2, 20)
ok(cost["total_units"] > 0 and len(cost["per_shot"]) == len(plan2.shots),
   "成本估算结构正确")
ok("相对算力单位" in cost["note"], "成本口径说明存在")

cost2 = T.estimate_cost(plan2, 20)
ok(cost2["per_shot"][0]["id"] == plan2.shots[0].id, "成本估算按镜头给出明细")

# --------------------------------------------------------------------------
section("修复回归守卫（本次审核发现的问题）")
# --------------------------------------------------------------------------
# ① plan_diff 不得改动传入的 plan
planA = SP.parse_text("=== s1 ===\npanel: 0\nduration: 5\ncharacters: 纲吉\naction: @纲吉 走路\n")
PL.bind_board(planA, board2, mode="by_panel_index")
PB.compose_plan(planA, pack)
before_tags = dict(planA.shots[0].picture_tags)
before_keys = list(planA.shots[0].ref_image_keys)
PL.plan_diff({"done": [], "base_seed": 0}, planA)
ok(planA.shots[0].picture_tags == before_tags,
   f"plan_diff 不再清空 picture_tags（{before_tags}）")
ok(planA.shots[0].ref_image_keys == before_keys,
   f"plan_diff 不再截断 ref_image_keys（{before_keys}）")

# ② build_render_tasks 默认不改动 plan
planB = SP.parse_text("=== s1 ===\npanel: 0\nduration: 5\ncharacters: 纲吉\naction: 走路\n")
PL.bind_board(planB, board2, mode="by_panel_index")
PB.compose_plan(planB, pack)
snap = (dict(planB.shots[0].picture_tags), planB.shots[0].resolved_seed)
PL.build_render_tasks(planB, pack, base_seed=99)
ok((dict(planB.shots[0].picture_tags), planB.shots[0].resolved_seed) == snap,
   "build_render_tasks 默认 mutate=False，不动传入的 plan")
PL.build_render_tasks(planB, pack, base_seed=99, mutate=True)
ok(planB.shots[0].resolved_seed != snap[1], "显式 mutate=True 时才写回 seed")

# ③ negative_mode：append 才进提示词，且 H3 没有 negative 输入
planN = SP.parse_text("=== s1 ===\nduration: 5\ncharacters: 纲吉\naction: 走路\n"
                      "negative: 画面抖动, 脸变形\n")
PB.compose_plan(planN.clone(), pack, negative_mode="append")
p_append = PB.compose_plan(planN, pack, negative_mode="append").shots[0].prompt
ok("Avoid these problems:" in p_append, "negative_mode=append 把负向词写进正向提示词")
ok("画面抖动" in p_append, "负向词内容被保留（不翻译）")

planN2 = SP.parse_text("=== s1 ===\nduration: 5\ncharacters: 纲吉\naction: 走路\n"
                       "negative: 画面抖动\n")
p_ignore = PB.compose_plan(planN2, pack, negative_mode="ignore").shots[0].prompt
ok("Avoid these problems:" not in p_ignore, "negative_mode=ignore 不写进提示词")

# ④ 手动框分页：page0 只作用于第 0 页
mb2, _ = PS.parse_manual_boxes("page0: 10,10,100,100\n200,10,100,100")
ok(len(mb2) == 2 and mb2[0][0] == 0 and mb2[1][0] == PS.PAGE_ANY,
   "手动框支持 pageN: 前缀，未标 page 的套用到所有页")
b_p0, _ = PS.split_page_boxes(page, mode="manual", manual_boxes=mb2, page_index=0)
b_p1, _ = PS.split_page_boxes(page, mode="manual", manual_boxes=mb2, page_index=1)
ok(len(b_p0) == 2, f"第 0 页拿到 page0 + 通用框（实得 {len(b_p0)}）")
ok(len(b_p1) == 1, f"第 1 页只拿到通用框（实得 {len(b_p1)}）")

# ⑤ 视频 / 音频池真的进布局（此前是死字段）
pack_av = RF.build_ref_pack(
    character_names="纲吉", character_images=[fake_img(1)],
    audio_names="bgm", audios=[{"waveform": np.zeros((1, 2, 32), dtype=np.float32),
                                "sample_rate": 32000}],
    video_names="动作参考", videos=[fake_img(5)[None].repeat(8, axis=0)],
    video_audios=[{"waveform": np.zeros((1, 2, 32), dtype=np.float32), "sample_rate": 32000}],
)
lay_av = RF.shot_ref_layout(pack_av, characters=["纲吉"], has_panel=True)
ok(list(lay_av["ref_videos"].keys()) == ["ref_video_0"], "视频进布局池（ref_video_0）")
ok(list(lay_av["ref_video_audios"].keys()) == ["ref_video_audio_0"],
   "视频配套音轨与视频同号（ref_video_audio_0）")
ok(list(lay_av["ref_audios"].keys()) == ["ref_audio_0"], "独立音频进布局池（ref_audio_0）")
ok(RF.normalize_images([fake_img(5)[None].repeat(8, axis=0)], split_batch=False)[0].shape[0] == 8,
   "参考视频不拆帧（整段 8 帧）")
ok(len(RF.normalize_images(fake_img(5)[None].repeat(8, axis=0))) == 8,
   "参考图批次仍会拆帧（8 张）")

# ⑥ 图像运算抽到 image_ops 后行为不变
import torch as _torch
st = C.image_ops.stack_images([_torch.zeros((4, 4, 3)), _torch.zeros((8, 6, 3))])
ok(tuple(st.shape) == (2, 8, 6, 3), f"stack_images 尺寸不一时 pad 到最大：{tuple(st.shape)}")
cc = C.image_ops.concat_frames([_torch.zeros((5, 8, 8, 3)), _torch.zeros((5, 4, 4, 3))])
ok(tuple(cc.shape) == (10, 8, 8, 3), f"concat_frames 沿时间轴拼：{tuple(cc.shape)}")
ok(C.image_ops.empty_image_batch().shape[0] == 0, "empty_image_batch 是 0 长度批次")
au = C.image_ops.concat_audio([_torch.zeros((1, 2, 40))], 32000)
ok(au and au["waveform"].shape[-1] == 40 and "error" not in au, "concat_audio 正常路径")
au_bad = C.image_ops.concat_audio([_torch.zeros((1, 2, 40, 5))], 32000)
ok(au_bad and "error" in au_bad, "concat_audio 异常时把原因带出去（不再静默丢音轨）")

# --------------------------------------------------------------------------
section("★ 参考素材张量契约：交付给官方节点的必须是 4 维 [B,H,W,C]")
# --------------------------------------------------------------------------
# 线上事故根因：插件内部统一用 3 维 [H,W,C]（一个槽位 = 一张图），
# 而官方 R2V 按 `h, w = img.shape[1], img.shape[2]` 读 —— 它假定 [B,H,W,C]。
# 3 维会被读成「h = 宽、w = 通道数 3」，于是：
#   * 参考图被 resize 成畸形尺寸（漫画页宽 1152 的分镜格 → 目标 1152x32）
#   * comfy/utils.py::common_upscale 抛
#     RuntimeError: shape '[1, -1, 3, 1152, 32]' is invalid for input of size 36864
# 报错文本与形状毫无字面关联，排查成本极高。这里把「升维 + 顺序不许变」锁死。
IO = C.image_ops

lay_panel = RF.shot_ref_layout(pack, characters=[char_ids[0]],
                               panel_image=fake_img(7), has_panel=True,
                               style_ids=[style_id])
ok(all(len(t.shape) == 4 and t.shape[0] == 1 and t.shape[-1] == 3
       for t in lay_panel["ref_images"].values()),
   "layout 产出的所有参考图都是 4 维 [1,H,W,C]")
ok(all(isinstance(t, np.ndarray) and t.shape[1:] == (4, 4, 3)
       for t in lay_panel["ref_images"].values()),
   "3 维 [H,W,C] 只补 batch 维，H/W/通道顺序不许变")

# 官方读尺寸的那两行：升维后 shape[1]/shape[2] 必须等于原始 (H, W)
_p = np.zeros((768, 1152, 3), dtype=np.float32)
ok(IO.as_ref_image(_p).shape[1:3] == (768, 1152),
   "升维后 img.shape[1:3] == (H, W)（官方正是靠这两维算目标尺寸）")

# 灰度 / 通道异常
ok(IO.as_ref_image(np.zeros((16, 24), dtype=np.float32)).shape == (1, 16, 24, 3),
   "2 维灰度图 → [1,H,W,3]（通道复制 3 份，不是把数组展平）")
ok(IO.as_ref_image(np.zeros((16, 24, 1), dtype=np.float32)).shape == (1, 16, 24, 3),
   "单通道 → 3 通道")
ok(IO.as_ref_image(np.zeros((16, 24, 4), dtype=np.float32)).shape == (1, 16, 24, 3),
   "RGBA → 丢 alpha")
ok(IO.as_ref_image(np.zeros((1, 1, 16, 24, 3), dtype=np.float32)).shape == (1, 16, 24, 3),
   "多余的前导 1 维被剥掉")
_notes = []
_batch4 = IO.as_ref_image(np.zeros((3, 16, 24, 3), dtype=np.float32), _notes, "test")
ok(_batch4.shape == (1, 16, 24, 3) and any("只取第 1 张" in n for n in _notes),
   "一个槽位收到多张图 → 只取第 1 张并告警（否则 <Picture i> 会整体错位）")
_bad = None
try:
    IO.as_ref_image(np.zeros((16, 24, 5), dtype=np.float32))
except ValueError as e:
    _bad = str(e)
ok(_bad is not None and "通道" in _bad,
   f"通道数莫名其妙时报错（而不是静默塞进模型）：{_bad}")

# 参考视频：4 维 [T,H,W,C]，T 不能被当成 batch 截掉
_vid = IO.as_ref_video(np.zeros((8, 16, 24, 3), dtype=np.float32))
ok(_vid.shape == (8, 16, 24, 3), f"参考视频保持 [T,H,W,C]（实得 {_vid.shape}）")
ok(IO.as_ref_video(np.zeros((8, 16, 24), dtype=np.float32)).shape == (8, 16, 24, 3),
   "无通道维的 [T,H,W] 视频补 3 通道（而不是当成 1 张图截成 1 帧）")

# 非张量塞进图片槽位 → 直接报错，绝不静默丢槽位（丢槽位 = 后面的 <Picture i> 全体前移）
_raised = None
try:
    RF.shot_ref_layout(pack, characters=[char_ids[0]],
                       panel_image={"oops": 1}, has_panel=True)
except ValueError as e:
    _raised = str(e)
ok(_raised is not None and "不是图像张量" in _raised,
   "非张量塞进参考图槽位 → 报错（不静默丢槽位导致串号）")

# 出站守卫：h3_api 自己也会再升一次维（防止有人绕过 shot_ref_layout 拼 ref_images）
import core.h3_api as _h3api  # noqa: E402  （core 包已在 sys.path 上，comfy 导入都是延迟的）
_imgs, _vids = _h3api._normalize_media_refs(
    {"ref_image_0": np.zeros((768, 1152, 3), dtype=np.float32)},
    {"ref_video_0": np.zeros((8, 16, 24, 3), dtype=np.float32)},
)
ok(_imgs["ref_image_0"].shape == (1, 768, 1152, 3), "h3_api 出站守卫把参考图升成 4 维")
ok(_vids["ref_video_0"].shape == (8, 16, 24, 3), "h3_api 出站守卫保持参考视频 [T,H,W,C]")

# --------------------------------------------------------------------------
section("资源管理：快照 / 对象解析 / 卸载原语（绝不抛异常）")
# --------------------------------------------------------------------------
ME = C.mem

_ram = ME.ram_snapshot()
ok(bool(_ram) and _ram.get("total_gb", 0) > 0,
   f"读到系统内存总量：{_ram.get('total_gb', 0):.2f} GB")
ok(0 <= _ram.get("pct", -1) <= 100, f"内存占用百分比在合理范围：{_ram.get('pct')}%")

_snap = ME.snapshot()
ok("vram" in _snap and "ram" in _snap, "snapshot() 同时给出显存与内存两块")
ok(isinstance(_snap["vram"], dict), "显存块是 dict（无 CUDA 时为空 dict，不报错）")


class _FakePatcher:
    """ModelPatcher 的最小替身：既有 model 又有 model_unload。"""

    def __init__(self, uuid=None):
        self.model = object()
        self.clone_base_uuid = uuid

    def model_unload(self, memory_to_free):
        return True

    def is_dead(self):
        return False


class _FakeClip:
    def __init__(self):
        self.patcher = _FakePatcher("clip-uuid")


ok(ME.patcher_of(None) is None, "patcher_of(None) → None")
ok(ME.patcher_of("FAKE_CLIP") is None, "patcher_of(字符串桩) → None（测试环境不会误判）")
ok(ME.patcher_of(_FakePatcher()) is not None, "MODEL 本身就是 ModelPatcher，能识别")
_c = _FakeClip()
ok(ME.patcher_of(_c) is _c.patcher, "CLIP 走 .patcher 取到 ModelPatcher")

_found, _unknown = ME.resolve_patchers([_c, _c, "oops", None])
ok(len(_found) == 1, f"重复对象只算一次（实得 {len(_found)}）")
ok(_unknown == ["str"], f"无法识别的对象被记下来而不是静默丢弃：{_unknown}")

# ★ 关键行为：什么都不传 = 什么都不做。
# 否则「参数接错」会变成「把用户的模型全卸了」，那是最糟的失败模式。
_none_res = ME.unload([], unload_all=False)
ok(_none_res["mode"] == "none" and _none_res["ok"] is False,
   "不传对象也不勾「全部卸载」→ 一个模型都不动（防止误卸）")
ok(any("没有可解析的模型对象" in e for e in _none_res["api_errors"]),
   "并且明确说明原因")

# 测试环境没有 comfy.model_management —— 这正是「绝不抛异常」的验证点
_sel = ME.unload([_c], unload_all=False)
ok(_sel["mode"] == "selected" and _sel["ok"] is False,
   "有对象但拿不到 model_management 时：不抛异常，返回 ok=False")
ok(any("comfy.model_management 不可用" in e for e in _sel["api_errors"]),
   "并给出可读原因（ComfyUI 外部环境）")

_all = ME.unload([], unload_all=True)
ok(_all["mode"] == "all", "unload_all=True 走「全部卸载」分支")

_txt = ME.format_unload_report(_sel)
for _kw in ("显存/内存释放报告", "模式", "释放前", "释放后", "结果"):
    ok(_kw in _txt, f"释放报告含「{_kw}」")
ok("未实际卸载" in _txt, "未真正卸载时报告如实说明（不谎报成功）")

_line = ME.format_stage_line(
    "起点", {"vram": {"used_mib": 100.0, "total_mib": 8188.0, "free_mib": 8088.0},
             "ram": {"used_gb": 13.9, "total_gb": 15.2, "pct": 91.0,
                     "pagefile_free_gb": 3.6}})
ok("起点" in _line and "8188" in _line and "91%" in _line,
   f"阶段曲线一行里有标签/显存/内存占用：{_line}")

_line_nocuda = ME.format_stage_line("起点", {"vram": {}, "ram": {}})
ok("无 CUDA" in _line_nocuda, "无 CUDA 环境下降级为「无 CUDA」而不是崩")

# --------------------------------------------------------------------------
section("H3 latent 几何：帧数 → latent 帧 → 视频 token（稀疏注意力的门槛判断）")
# --------------------------------------------------------------------------
# 官方 FRAME_PER_TOKEN = (1,4,4,4,4)：每 5 个 latent 帧覆盖 17 个像素帧。
# H3 的帧数是 17n+5，对应的 latent_t = 5n+2（124 帧 → 37）。
ok(T.latent_frames_for(124) == 37, f"124 帧 → latent_t 37（实得 {T.latent_frames_for(124)}）")
ok(T.latent_frames_for(243) == 72, f"243 帧 → latent_t 72（实得 {T.latent_frames_for(243)}）")
ok(T.latent_frames_for(362) == 107, f"362 帧 → latent_t 107（实得 {T.latent_frames_for(362)}）")
ok(T.latent_frames_for(4) == 0, "小于 5 帧返回 0（不瞎猜）")

# 自洽性：latent_t 覆盖的像素帧数必须 >= 目标，且少一帧就不够
for _f in (124, 243, 362, 736, 1450):
    _lt = T.latent_frames_for(_f)
    _cov = sum(T.H3_FRAME_PER_TOKEN[k % 5] for k in range(_lt))
    _cov_prev = sum(T.H3_FRAME_PER_TOKEN[k % 5] for k in range(_lt - 1))
    ok(_cov >= _f > _cov_prev, f"{_f} 帧：latent_t {_lt} 覆盖 {_cov}（恰为最小可行值）")

# ★ 画幅必须吸附到 32 的倍数 —— 432 不是 32 的倍数（13.5×32），会被吸附到 448。
#   这条很反直觉：文档里写「768×432」是错的落值，真正生效的是 768×448。
ok(T.align_canvas(768, 432) == (768, 448),
   f"768x432 不是合法画幅，会被吸附成 768x448（实得 {T.align_canvas(768, 432)}）")
ok(T.align_canvas(960, 544) == (960, 544),
   "960x544 本身就是合法的 32 倍数（30×32 与 17×32），不会被改")

# token 估算：官方 vsa_plan 用 grid = (latent_t, latent_h//2, latent_w//2)，16× 空间压缩
ok(T.estimate_video_tokens(768, 448, 124) == 12432,
   f"768x448 × 124 帧 ≈ 12432 token（实得 {T.estimate_video_tokens(768, 448, 124)}）")
ok(T.estimate_video_tokens(960, 544, 124) == 18870,
   f"960x544 × 124 帧 ≈ 18870 token（实得 {T.estimate_video_tokens(960, 544, 124)}）")
ok(T.estimate_video_tokens(1344, 768, 124) == 37296,
   f"1344x768 × 124 帧 ≈ 37296 token（实得 {T.estimate_video_tokens(1344, 768, 124)}）")

# 单调性：画幅越大 / 时长越长，token 只增不减
ok(T.estimate_video_tokens(960, 544, 124) > T.estimate_video_tokens(768, 448, 124),
   "画幅变大 → token 变多")
ok(T.estimate_video_tokens(768, 448, 362) > T.estimate_video_tokens(768, 448, 124),
   "时长变长 → token 变多")

# ★ 官方 min_tokens 默认 12288：
#   768×448×5s 是 12432（刚过线 → 稀疏能生效，但只比门槛高 1%），
#   再小一档（704×384）就掉到 9768 → 会**静默**走 dense，必须提醒用户。
_DEFAULT_MIN_TOKENS = 12288
ok(T.estimate_video_tokens(768, 448, 124) > _DEFAULT_MIN_TOKENS,
   "768x448 × 5s 刚过官方阈值（12432 > 12288）→ 稀疏能生效")
ok(T.estimate_video_tokens(704, 384, 124) < _DEFAULT_MIN_TOKENS,
   "★ 704x384 × 5s 只有 9768 < 12288 → 走 dense，预检必须提醒")
ok(T.estimate_video_tokens(960, 544, 124) > _DEFAULT_MIN_TOKENS,
   "960x544 × 5s 明显超过阈值 → 稀疏稳定生效")

# --------------------------------------------------------------------------
section("★ 多边形分镜格（斜切版面）")
# --------------------------------------------------------------------------
# 为什么需要：斜切分镜里，一个斜格的 bbox 必然与相邻格的 bbox 相交 ——
# 这是几何必然，不是算法不够好。而且旧版**手填框也只收 4 个数**，
# 连退路都没有。真实样本 Reborn_10_134（51° 粗黑斜带分隔，整页 1 个连通块）。
_sh, _sw = PS.parse_manual_shapes(
    "10,10,100,100\n"
    "240,0,761,644,700,780,240,780\n"
    "page0: 100,50,300,50,300,200\n"
    "[[1,2],[10,2],[10,9],[1,9]]"
)
ok(len(_sh) == 4, f"手动分镜格支持矩形/四边形/多边形/JSON 混写（实得 {len(_sh)}）")
ok(_sh[0]["poly"] is None and _sh[0]["box"] == [10, 10, 100, 100],
   "4 个数 = 矩形，poly 为 None")
ok(_sh[1]["poly"] == [[240, 0], [761, 644], [700, 780], [240, 780]]
   and _sh[1]["box"] == [240, 0, 521, 780],
   f"8 个数 = 四边形，box 自动取 bbox（实得 {_sh[1]}）")
ok(_sh[2]["page"] == 0 and len(_sh[2]["poly"]) == 3,
   "page0: 前缀 + 三角形（3 个顶点）")
ok(len(_sh[3]["poly"]) == 4,
   "★ JSON 点列表 [[x,y],...] = **一个**多边形，不是 4 个矩形")
ok(not _sw, f"合法写法不应有警告（实得 {_sw}）")

ok(PS.polygon_area([[0, 0], [10, 0], [10, 10], [0, 10]]) == 100.0, "鞋带公式面积")
ok(PS.poly_to_box([[5, 7], [50, 7], [50, 30]]) == [5, 7, 45, 23], "多边形 → bbox")
_up = PS.upscale_shapes(
    [{"page": -1, "box": [0, 0, 10, 10], "poly": [[0, 0], [10, 0], [10, 10]]}], 2.0)
ok(_up[0]["poly"] == [[0, 0], [20, 0], [20, 20]] and _up[0]["box"] == [0, 0, 20, 20],
   "多边形缩放后 box 与 poly 始终一致")

# 斜切版面：两个三角格共用一条对角线，两者 bbox 完全重合（各占整页）
_diag = np.full((200, 200, 3), 255, dtype=np.uint8)
_yy, _xx = np.mgrid[0:200, 0:200]
_diag[_xx + _yy < 200] = 40          # 左上三角 = 格 A
_diag[_xx + _yy >= 200] = 120        # 右下三角 = 格 B
_A = [[0, 0], [200, 0], [0, 200]]
_rect_crop = PS.crop_boxes(_diag, [[0, 0, 200, 200]])[0]
_poly_crop = PS.crop_polygons(_diag, [_A])[0]
ok(_rect_crop.shape == _poly_crop.shape,
   "多边形裁剪输出仍是 bbox 尺寸的矩形图像 → 下游契约不变")
ok(int(_rect_crop[195, 195, 0]) == 120,
   f"★ 矩形裁：右下角是**邻居格 B** 的像素（实得 {int(_rect_crop[195, 195, 0])}）")
ok(int(_poly_crop[195, 195, 0]) == 255,
   f"★ 多边形裁：同一位置已填成纸白（实得 {int(_poly_crop[195, 195, 0])}）")
ok(int(_poly_crop[5, 5, 0]) == 40, "多边形内部像素保持原样")

# 反向验证：只有「明明不是矩形」的块才配多边形；关掉开关必须退回纯矩形
_tri = np.full((200, 200, 3), 255, dtype=np.uint8)
_tri[_xx + _yy < 200] = 0
_s_auto, _ = PS.split_page_shapes(_tri, mode="auto", panel_polygon="auto")
_s_off, _ = PS.split_page_shapes(_tri, mode="auto", panel_polygon="off")
ok(len(_s_auto) == 1 and _s_auto[0]["poly"] and len(_s_auto[0]["poly"]) >= 3,
   f"panel_polygon=auto：三角形的块自动配多边形（实得 {_s_auto[0].get('poly')}）")
ok(_s_off[0]["poly"] is None, "★ 反向验证：panel_polygon=off 时退回纯矩形")
ok(PS.split_page_boxes(_tri, mode="auto")[0][0] == tuple(_s_off[0]["box"]),
   "split_page_boxes（矩形 API）与 shapes 版的 box 一致")

# --------------------------------------------------------------------------
section("★ 画幅：严格对齐官方 adapt_canvas 规则")
# --------------------------------------------------------------------------
# 官方 comfy_extras/nodes_minimax_h3.py::adapt_canvas：
#   短边 768 / 面积 ≤ 768×1344 / 逐轴 round 到 32
ok(T.official_canvas(16, 9) == (1344, 768),
   f"16:9 → 1344x768（实得 {T.official_canvas(16, 9)}）")
ok(T.official_canvas(9, 16) == (768, 1344), "9:16 → 768x1344")
ok(T.official_canvas(4, 3) == (1024, 768), "4:3 → 1024x768")
ok(T.official_canvas(1, 1) == (768, 768), "1:1 → 768x768")
for _rw, _rh in [(16, 9), (4, 3), (1, 1), (3, 4), (9, 16), (21, 9), (2.39, 1)]:
    _w, _h = T.official_canvas(_rw, _rh)
    ok(_w % 32 == 0 and _h % 32 == 0, f"official_canvas({_rw}:{_rh}) = {_w}x{_h} 是 32 倍数")
    ok(_w * _h <= T.H3_MAX_PIXELS,
       f"official_canvas({_rw}:{_rh}) 面积 {_w * _h} ≤ 官方上限 {T.H3_MAX_PIXELS}")
ok(SP._parse_size("16:9") == (1344, 768), "文本里的 16:9 也走官方规则（不是 1376x768）")
ok(T.resolve_canvas("preset", "16:9 横屏", None, 1.0)[:2] == (1344, 768), "preset ×1.0")
ok(T.resolve_canvas("preset", "16:9 横屏", None, 0.5)[:2] == (672, 384), "preset ×0.5")
ok(T.resolve_canvas("page", page_size=(1164, 1800), scale=1.0)[:2] == (768, 1184),
   f"跟随原页比例 1164x1800 → 768x1184（实得 {T.resolve_canvas('page', page_size=(1164, 1800))[:2]}）")
ok(T.resolve_canvas("text", text_size=(960, 544))[0] == 960, "text 模式用文本值")
# ★ 关键：缩放必须在「官方规范之后」做，否则 official_canvas 会把短边拉回 768、缩放白做
ok(T.resolve_canvas("preset", "16:9 横屏", None, 0.25)[:2] == (320, 192),
   f"preset ×0.25 → 320x192（实得 {T.resolve_canvas('preset', '16:9 横屏', None, 0.25)[:2]}）")

# --------------------------------------------------------------------------
section("★ 从 board 自动出镜 + hybrid 合并")
# --------------------------------------------------------------------------
_bd = T.Board(panels=[T.Panel(index=i, box=[0, 0, 10, 10], page_index=0)
                      for i in range(3)], pages=1, page_size=(1164, 1800))
_so = T.resolve_canvas("preset", "16:9 横屏", None, 0.5)
_pb = SP.from_board(_bd, default_duration=5.0, size_override=_so)
ok(len(_pb.shots) == 3, f"一个分镜格 = 一个镜头（3 格 → 实得 {len(_pb.shots)} 镜）")
ok(_pb.shots[0].panel_index == 0 and _pb.shots[2].panel_index == 2, "panel 索引按 board 顺序填好")
ok(_pb.shots[0].requested_seconds == 5.0 and _pb.shots[0].length == 124,
   "时长链：填 5s → 124 帧")
ok(abs(_pb.shots[0].duration - 5.167) < 0.01,
   f"实际时长 5.167s 可追溯（实得 {_pb.shots[0].duration}）")
ok((_pb.shots[0].width, _pb.shots[0].height) == (672, 384), "画幅用画幅选择器的值")
ok(any("自动生成" in w for w in _pb.warnings), "报告里说明「这是自动生成的初稿」")

# 反向验证：没有 board 时必须明确报错，而不是静默给出 0 镜
try:
    SP.from_board(None)
    ok(False, "★ 反向验证：缺 board 时应当报错")
except ValueError as _e:
    ok("board" in str(_e), f"★ 反向验证：缺 board 时明确报错（{str(_e)[:30]}…）")

_txt = "=== s01 ===\npanel: 1\nduration: 4\ncontent: 饭桌上的对话\n"
_tp = SP.parse_text(_txt, default_duration=None, size_override=_so, mark_missing_size=True)
_mg = SP.merge_plan_with_text(_pb, _tp)
ok(len(_mg.shots) == 3, f"hybrid 覆盖不改变镜数（实得 {len(_mg.shots)}）")
ok(_mg.shots[1].length == 107 and _mg.shots[1].content == "饭桌上的对话",
   f"panel 1 被文本覆盖：4s → 107 帧（实得 {_mg.shots[1].length}）")
ok(_mg.shots[1].requested_seconds == 4.0, "原始秒数 4 被保留（不是被对齐后的 4.458）")
ok(_mg.shots[0].length == 124 and _mg.shots[0].content == "",
   "没被覆盖的镜沿用 board 自动稿")
ok(any("hybrid 合并" in w for w in _mg.warnings), "报告写明覆盖/追加了几镜")

# --------------------------------------------------------------------------
section("★ 报告口径：『填 4 会变成 4.46』必须可见")
# --------------------------------------------------------------------------
_tbl = _mg.shot_table()
ok("→ 107 帧 →" in _tbl, "镜头表显示「填的秒数 → 帧 → 实际秒数」三段链")
ok("4.46s" in _tbl, f"表格里能看到 4 → 107 帧 → 4.46s（表：{_tbl[:80]}…）")
ok("计划值" in _mg.summary(), "摘要明确标注是「计划值」，并提示 max_shots 的影响")

# --------------------------------------------------------------------------
section("★ 工作流模板的矩阵式布局（组间网格 + 组内网格）")
# --------------------------------------------------------------------------
# 契约（与 tools/make_example_workflow.py 的 INNER_COLS / OUTER_COLS 对应）：
#   · 组间：group 的左上角落在二维网格上 —— 至少 2 行 2 列，且无两 group 占同一单元
#   · 组内：每个 group 内节点按 INNER_COLS 列铺开（不是旧的单列堆叠），
#           同一列的节点 x 相同 → 列对齐
# 这些断言会在改了布局算法却改坏规整性时立刻变红。
import itertools as _it  # noqa: E402
import json as _json  # noqa: E402

_INNER_COLS = 2
_T_FILES = [
    ("整集渲染版", "H3MangaStudio_example_workflow.json"),
    ("官方接线版", "H3MangaStudio_official_nodes_workflow.json"),
]

for _label, _fn in _T_FILES:
    _path = os.path.join(PKG_DIR, "templates", _fn)
    if not os.path.isfile(_path):
        ok(False, f"[{_label}] {_fn} 不存在（先跑 tools/make_example_workflow.py）")
        continue
    with open(_path, encoding="utf-8") as _f:
        _wf = _json.load(_f)
    _nodes, _groups = _wf["nodes"], _wf["groups"]

    _ov = 0
    for _a, _b in _it.combinations(_nodes, 2):
        _ax, _ay = _a["pos"]
        _bx, _by = _b["pos"]
        if (min(_ax + _a["size"][0], _bx + _b["size"][0]) - max(_ax, _bx) > 0
                and min(_ay + _a["size"][1], _by + _b["size"][1]) - max(_ay, _by) > 0):
            _ov += 1
    ok(_ov == 0, f"[{_label}] 节点两两不重叠（实得 {_ov} 对）")

    _gov = 0
    for _a, _b in _it.combinations(_groups, 2):
        _ax, _ay, _aw, _ah = _a["bounding"]
        _bx, _by, _bw, _bh = _b["bounding"]
        if (min(_ax + _aw, _bx + _bw) - max(_ax, _bx) > 0
                and min(_ay + _ah, _by + _bh) - max(_ay, _by) > 0):
            _gov += 1
    ok(_gov == 0, f"[{_label}] group 框两两不重叠（实得 {_gov} 对）")

    _gx = sorted({g["bounding"][0] for g in _groups})
    _gy = sorted({g["bounding"][1] for g in _groups})
    ok(len(_gx) >= 2 and len(_gy) >= 2,
       f"[{_label}] 组间是二维网格而非单行长条"
       f"（实得 {len(_gx)} 列 × {len(_gy)} 行，共 {len(_groups)} 个 group）")
    _cells = [(g["bounding"][0], g["bounding"][1]) for g in _groups]
    ok(len(set(_cells)) == len(_cells),
       f"[{_label}] 组间无两个 group 占同一网格单元"
       f"（冲突 {len(_cells) - len(set(_cells))} 处）")

    for _g in _groups:
        _gx0, _gy0, _gw, _gh = _g["bounding"]
        _mem = [n for n in _nodes
                if _gx0 <= n["pos"][0] + n["size"][0] / 2 <= _gx0 + _gw
                and _gy0 <= n["pos"][1] + n["size"][1] / 2 <= _gy0 + _gh]
        _xs = sorted({n["pos"][0] for n in _mem})
        _ys = sorted({n["pos"][1] for n in _mem})
        _want_rows = (len(_mem) + _INNER_COLS - 1) // _INNER_COLS
        if len(_mem) >= 2 and len(_xs) == 1:
            ok(False, f"[{_label}]「{_g['title']}」组内是单列堆叠，未铺成矩阵")
        elif len(_ys) > _want_rows:
            ok(False, f"[{_label}]「{_g['title']}」组内行数 {len(_ys)} 超过"
                      f"{_INNER_COLS} 列布局应有的 {_want_rows} 行")
        else:
            ok(True, f"[{_label}]「{_g['title']}」组内矩阵 {len(_xs)} 列 × {len(_ys)} 行"
                     f"（{len(_mem)} 个节点）")

    _owner = []
    for _n in _nodes:
        _cx = _n["pos"][0] + _n["size"][0] / 2
        _cy = _n["pos"][1] + _n["size"][1] / 2
        _owner.append(sum(
            1 for g in _groups
            if g["bounding"][0] <= _cx <= g["bounding"][0] + g["bounding"][2]
            and g["bounding"][1] <= _cy <= g["bounding"][1] + g["bounding"][3]))
    ok(_owner.count(0) == 0, f"[{_label}] 无孤儿节点（实得 {_owner.count(0)} 个）")
    ok(max(_owner) <= 1, f"[{_label}] 无节点被多个 group 框住（实得 {_owner.count(2)} 个）")


# --------------------------------------------------------------------------
section("★ 一格多镜（shots: N）—— 「镜头数 ≠ 格数」的落地")
# --------------------------------------------------------------------------
_MULTI_TXT = """=== s01 ===
panel: 0
duration: 4
action: 街景推近
=== s02 ===
panel: 1
shots: 2
duration: 5
content: 短发女生笑脸 | 饭桌对话
action: 少女特写 | 两人对坐
dialogue: 好吃！ | 今天也谢谢你
speaker: 纲吉
=== s03 ===
panel: 2
duration: 3
action: 收尾
"""
_m = SP.parse_text(_MULTI_TXT)
ok(len(_m.shots) == 4, f"shots=2 的那格展开后总镜数 3→4（实得 {len(_m.shots)}）")
ok([s.id for s in _m.shots] == ["s01", "s02", "s02_2", "s03"],
   f"展开的镜 id 用 _2 后缀且顺序不变（实得 {[s.id for s in _m.shots]}）")
_s02, _s02b = _m.shots[1], _m.shots[2]
ok(_s02.panel_index == 1 and _s02b.panel_index == 1,
   "展开出的两镜绑**同一个** panel（共用一张格图当锚点）")
ok((_s02.sub_index, _s02.sub_total) == (0, 2)
   and (_s02b.sub_index, _s02b.sub_total) == (1, 2),
   f"sub_index/sub_total 正确（实得 {(_s02.sub_index, _s02.sub_total)} / "
   f"{(_s02b.sub_index, _s02b.sub_total)}）")
ok(_s02.content == "短发女生笑脸" and _s02b.content == "饭桌对话",
   f"content 按 `|` 分段对应（实得 {_s02.content!r} / {_s02b.content!r}）")
ok(_s02.action == "少女特写" and _s02b.action == "两人对坐",
   "action 也按 `|` 分段")
ok(_s02.dialogue == "好吃！" and _s02b.dialogue == "今天也谢谢你",
   "dialogue 也按 `|` 分段")
ok(_s02.requested_seconds == 5 and _s02b.requested_seconds == 5,
   "★ duration 语义 = **每个镜头**各自 5s（该格合计 10s），不是整格平分")
ok(_s02.speaker == "纲吉" and _s02b.speaker == "纲吉", "speaker 不切分，两镜都继承")
ok([s.index for s in _m.shots] == [0, 1, 2, 3], "展开后 index 重新连续编号")
ok(any("一格" in w or "shots=2" in w for w in _m.warnings),
   "展开时给出可读 warning（让人知道镜头数为什么变多）")
ok("panel x/N" in _m.shot_table() or "1/2" in _m.shot_table(),
   "shot_table 标出「第几镜 / 共几镜」")

# ★ 反向验证：不写 shots 时行为必须**完全不变**（id / index / sub_total）
_back = SP.parse_text("=== a ===\npanel: 0\nduration: 5\naction: x\n")
ok(len(_back.shots) == 1 and _back.shots[0].id == "a"
   and _back.shots[0].sub_total == 1 and _back.shots[0].sub_index == 0
   and _back.shots[0].index == 0,
   "★ 反向验证：缺省（不写 shots）时 id/index/sub 与旧版逐字段一致")

# 分段容错：不足 → 重复最后一段；超出 → 并回最后一段（不丢字）
_short = SP.parse_text("=== b ===\npanel: 0\nshots: 3\nduration: 5\ncontent: 只有一段\n")
ok([s.content for s in _short.shots] == ["只有一段"] * 3,
   f"段数不足时重复最后一段（实得 {[s.content for s in _short.shots]}）")
_long = SP.parse_text("=== c ===\npanel: 0\nshots: 2\nduration: 5\ncontent: 一 | 二 | 三\n")
ok([s.content for s in _long.shots] == ["一", "二 | 三"],
   f"段数超出时并回最后一段、不丢内容（实得 {[s.content for s in _long.shots]}）")

# 非法值 / 上限
_bad = SP.parse_text("=== d ===\npanel: 0\nshots: 0\nduration: 5\ncontent: x\n")
ok(len(_bad.shots) == 1 and any("非法" in w for w in _bad.warnings),
   "shots=0 按 1 镜处理并给 warning")
_cap = SP.parse_text("=== e ===\npanel: 0\nshots: 99\nduration: 5\ncontent: x\n")
ok(len(_cap.shots) == SP.MAX_SHOTS_PER_PANEL,
   f"shots 超上限截断到 {SP.MAX_SHOTS_PER_PANEL}（实得 {len(_cap.shots)}）")
ok(any("上限" in w for w in _cap.warnings), "截断时给出 warning")

# hybrid：文本按 panel **整组**替换（否则一格多镜在 hybrid 下永远只剩 1 镜）
_bd3 = T.Board(panels=[T.Panel(index=i, box=[0, 0, 100, 100], page_index=0)
                       for i in range(3)], pages=1)
_base3 = SP.from_board(_bd3, default_duration=5.0)
ok(len(_base3.shots) == 3 and all(s.sub_total == 1 for s in _base3.shots),
   "from_board 默认一格一镜（3 格 → 3 镜）")

_txt_over = SP.parse_text("=== s02 ===\npanel: 1\nshots: 2\nduration: 4\n"
                          "content: 左半格特写 | 右半格对话\n")
_mg = SP.merge_plan_with_text(_base3, _txt_over)
ok(len([s for s in _mg.shots if s.panel_index == 1]) == 2,
   f"★ hybrid 下 shots=2 生效：panel 1 变 2 镜（实得 "
   f"{len([s for s in _mg.shots if s.panel_index == 1])}）")
ok(len(_mg.shots) == 4, f"合并后总镜数 3→4（实得 {len(_mg.shots)}）")
ok([s.index for s in _mg.shots] == [0, 1, 2, 3], "合并后 index 连续")
ok(len([s for s in _mg.shots if s.panel_index == 0]) == 1
   and len([s for s in _mg.shots if s.panel_index == 2]) == 1,
   "未覆盖的 panel 保持原样（各 1 镜）")

_base6 = SP.from_board(_bd3, default_duration=5.0, shots_per_panel=2)
ok(len(_base6.shots) == 6 and all(s.sub_total == 2 for s in _base6.shots),
   f"from_board(shots_per_panel=2)：3 格 → 6 镜（实得 {len(_base6.shots)}）")
_mg2 = SP.merge_plan_with_text(
    _base6, SP.parse_text("=== q ===\npanel: 0\nduration: 9\ncontent: 改了第一格\n"))
ok(len([s for s in _mg2.shots if s.panel_index == 1]) == 2,
   "只有 panel 0 被文本覆盖，panel 1 仍保留自动稿的 2 镜")
ok(len([s for s in _mg2.shots if s.panel_index == 0]) == 1,
   "★ 文本显式声明了该格 → 以文本为准（panel 0 从 2 镜回到 1 镜）")

# ==========================================================================
section("★ VLM 层：三条硬约束 / 繁简与子串兜底 / 分段契约 / 结果转换")
# ==========================================================================
VLM = C.vlm

# ---- 角色名归一化 ----
ok(VLM.norm_name("綱吉") == "纲吉", f"繁→简：綱吉 → 纲吉（实得 {VLM.norm_name('綱吉')}）")
ok(VLM.norm_name("獄寺") == "狱寺" and VLM.norm_name("藍波") == "蓝波",
   "繁→简：獄寺 / 藍波")
ok(VLM.norm_name("  纲吉  ") == "纲吉", "去空白")
ok(VLM.parse_aliases("綱吉=纲吉, 藍波=蓝波,BAD") ==
   {"綱吉": "纲吉", "藍波": "蓝波"}, "别名表解析（忽略无 = 的项）")
ok(VLM.parse_names("纲吉, Reborn；京子\n纲吉") == ["纲吉", "Reborn", "京子"],
   f"角色名列表去重保序（实得 {VLM.parse_names('纲吉, Reborn；京子\n纲吉')}）")

_CHARS = {"纲吉", "狱寺", "蓝波", "Reborn"}

# ---- 正常路径：繁简归一 + 闭集剔除 + camera 纠正 + shots 截断 ----
_ob = {"page_kind": "narrative", "panels": [
    {"panel": 0, "shots": 1, "characters": ["綱吉", "獄寺", "蓝波", "優尼", "查无此人"],
     "camera": "static", "content": "街景", "confidence": 0.9},
    {"panel": 1, "shots": 99, "camera": "zoom_x", "content": "特写"},
]}
_p, _e, _w = VLM.validate_page(_ob, [0, 1], _CHARS, "p")
ok(_e == [], f"正常内容不报错（实得 {_e}）")
ok(_p[0]["characters"] == ["纲吉", "狱寺", "蓝波"],
   f"繁简归一 + 闭集剔除：{_p[0]['characters']}")
ok(_p[1]["shots"] == VLM.MAX_SHOTS_PER_PANEL,
   f"shots 超上限截断到 {VLM.MAX_SHOTS_PER_PANEL}（实得 {_p[1]['shots']}）")
ok(_p[1]["camera"] == "static", "camera 越表 → static")
ok(any("角色表外的名字" in x for x in _w), "闭集剔除留警告（不静默）")

# ---- 子串兜底：模型写全名而闭集是简称 ----
# ★ 实测一整卷里 7+ 处写成「泽田纲吉」→ 主角参考图被整批剔除且不报错。
_p2, _e2, _w2 = VLM.validate_page(
    {"panels": [{"panel": 0, "camera": "static", "content": "x",
                 "characters": ["澤田綱吉"]}]},
    [0], {"纲吉", "狱寺"}, "p")
ok(_p2[0]["characters"] == ["纲吉"],
   f"★ 子串兜底：澤田綱吉 → 纲吉（实得 {_p2[0]['characters']}）")
ok(any("归一为" in x for x in _w2), "归一化留警告")

# 反向验证：闭集里没有匹配项时必须剔除，绝不能瞎配
_p3, _e3, _w3 = VLM.validate_page(
    {"panels": [{"panel": 0, "camera": "static", "content": "x",
                 "characters": ["完全无关的名字"]}]}, [0], {"纲吉"}, "p")
ok(_p3[0]["characters"] == [], "★ 反向验证：闭集里无匹配 → 必须剔除（不瞎配）")

# ---- 硬约束 1：禁 <Picture i> 标签 → 整页拒收 ----
_p4, _e4, _ = VLM.validate_page(
    {"panels": [{"panel": 0, "camera": "static", "content": "看 <Picture 1> 的人"}]},
    [0], _CHARS, "p")
ok(len(_e4) == 1 and "Picture" in _e4[0],
   f"★ 出现 <Picture i> → 整页拒收（实得 {_e4}）")

# ---- 硬约束 3：panel 越界 ----
_p5, _e5, _ = VLM.validate_page(
    {"panels": [{"panel": 42, "camera": "static", "content": "x"}]}, [0, 1], _CHARS, "p")
ok(any("不在本页切出的格号里" in x for x in _e5),
   f"panel 越界 → 拒收（实得 {_e5}）")
# 越界那一格被剔除后，0/1 自然也没被解读 → 会连带报漏读。两条都对。
ok(any("没有被解读" in x for x in _e5), "越界连带触发漏读（一致的行为）")

# ---- 严格覆盖：漏读整格 ----
_ob_miss = {"panels": [{"panel": 0, "camera": "static", "content": "z"}]}
_, _e6, _ = VLM.validate_page(_ob_miss, [0, 1, 2], _CHARS, "p", strict_coverage=True)
ok(len(_e6) == 1 and "没有被解读" in _e6[0], "★ 漏读整格 → 默认拒收")
_, _e7, _w7 = VLM.validate_page(_ob_miss, [0, 1, 2], _CHARS, "p",
                                strict_coverage=False)
ok(_e7 == [] and any("没有被解读" in x for x in _w7), "放宽后只警告不拒收")

# ---- ★ shots>1 的分段契约 ----
_p8, _e8, _w8 = VLM.validate_page(
    {"panels": [{"panel": 0, "shots": 2, "camera": "static",
                 "content": "只有一段内容"}]}, [0], _CHARS, "p")
ok(any("应 2 段" in x for x in _w8),
   f"★ 反向验证：shots=2 但只有 1 段 → 必须警告（实得 {_w8}）")
_p9, _e9, _w9 = VLM.validate_page(
    {"panels": [{"panel": 0, "shots": 2, "camera": "static",
                 "content": "左半是特写 | 右半是全景",
                 "action": "推近 | 拉远"}]}, [0], _CHARS, "p")
ok(not any("应 2 段" in x for x in _w9),
   f"分段齐了就不警告（实得 {_w9}）")

# ---- extract_json 容错 ----
ok(VLM.extract_json('```json\n{"a": 1}\n```')["a"] == 1, "剥 ```json 包裹")
ok(VLM.extract_json('废话 {"a": 2} 更多废话')["a"] == 2, "容忍前后废话")
try:
    VLM.extract_json("没有 JSON")
    ok(False, "无 JSON 时应抛错")
except VLM.VLMError:
    ok(True, "无 JSON 时抛 VLMError")

# ---- transform_pages ----
_pages = [{"page_name": "p.png", "page_kind": "narrative", "panels": [
    {"panel": 0, "shots": 3, "duration_hint": 10, "camera": "static",
     "content": "a", "characters": []},
    {"panel": 1, "shots": 1, "duration_hint": 6, "camera": "static",
     "content": "b", "characters": []}]}]
_t1 = VLM.transform_pages(_pages, force_shots=1, duration_scale=0.5)
ok([x["shots"] for x in _t1[0]["panels"]] == [1, 1],
   f"force_shots=1 全部压成 1 镜（实得 {[x['shots'] for x in _t1[0]['panels']]}）")
ok([x["duration_hint"] for x in _t1[0]["panels"]] == [5, 3],
   f"duration_scale=0.5：10→5、6→3（实得 "
   f"{[x['duration_hint'] for x in _t1[0]['panels']]}）")
_t2 = VLM.transform_pages(_pages)
ok([x["shots"] for x in _t2[0]["panels"]] == [3, 1],
   "不传参数时原样保留（默认不过度干预）")
ok(_pages[0]["panels"][0]["shots"] == 3, "transform_pages 不修改入参（纯函数）")

# ---- to_block_text ----
_bt = VLM.to_block_text(
    [{"page_name": "page001.png", "page_kind": "cover",
      "panels": [{"panel": 0, "shots": 1, "camera": "static",
                  "content": "封面", "duration_hint": 3}]}] + _pages,
    episode="t")
ok("=== page001" not in _bt,
   "★ 非叙事页（cover）默认跳过 —— 不生成它的分段头（文件名只出现在「已跳过」说明里）")
ok("=== p_00 ===" in _bt and "panel: 0" in _bt, "块文本分段头正确")
ok("shots: 3" in _bt and "一格 3 镜" in _bt, "shots>1 写进文本并带抽查注释")
ok("已跳过 1 个非叙事页" in _bt, "跳过说明写进文本尾部")
_cover = {"page_name": "page001.png", "page_kind": "cover",
          "panels": [{"panel": 0, "shots": 1, "camera": "static",
                      "content": "封面", "duration_hint": 3}]}
_bt2 = VLM.to_block_text([_cover] + _pages, episode="t",
                         include_non_narrative=True)
ok("=== page001_00 ===" in _bt2,
   "★ 正向对照：include_non_narrative=True 时非叙事页也出镜头（开关真的有效）")

# ---- count_pages / VLMResult ----
_cp = VLM.count_pages(_pages, include_non_narrative=False)
ok(_cp["panels"] == 2 and _cp["shots"] == 4,
   f"count_pages：2 格 / 4 镜（实得 {_cp}）")
_vr = T.VLMResult(pages=_pages, model="m", seconds=1.5)
_st = _vr.stats()
ok(_st["panels"] == 2 and _st["shots"] == 4 and _st["shots_over_one"] == 1,
   f"VLMResult.stats（实得 {_st}）")
ok("VLM 解读结果" in _vr.report(), "VLMResult.report 可读")

# ---- tensor_to_b64：★ RGB 顺序（不转换就会红蓝颠倒送给模型）----
import numpy as _np  # noqa: E402
_red = _np.zeros((300, 600, 3), dtype=_np.float32)
_red[:, :, 0] = 1.0                                # RGB 里的红
_b64, _mime, _w, _h = VLM.tensor_to_b64(_red, max_edge=300)
ok((_w, _h) == (300, 150), f"长边缩放到 300（实得 {_w}x{_h}）")
ok(_mime == "image/jpeg", f"缩过就是 jpeg（实得 {_mime}）")
import base64 as _b64mod  # noqa: E402
import cv2 as _cv2  # noqa: E402
_im = _cv2.imdecode(_np.frombuffer(_b64mod.b64decode(_b64), dtype=_np.uint8),
                   _cv2.IMREAD_COLOR)              # cv2 解出来是 BGR
_px = _im[_im.shape[0] // 2, _im.shape[1] // 2].tolist()
ok(_px[2] > 200 and _px[0] < 60,
   f"★ RGB 张量编码后解码仍是红（BGR={_px}，应为 [0,0,255]）—— 不转换就会红蓝颠倒")

print(f"\n{'=' * 60}\n第 1 层纯逻辑：{PASS} 通过 / {FAIL} 失败\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
