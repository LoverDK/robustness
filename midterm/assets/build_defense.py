# -*- coding: utf-8 -*-
"""Build midterm defense slides, script, Q&A, and form-field JSON.

Reads the college template and writes only under midterm/. Does not modify
the template or any other project file.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(r"D:\study\科研拔尖项目")
MID = ROOT / "midterm"
ASSETS = MID / "assets"
TEMPLATE = next(ROOT.glob("*.pptx"))
OUT_PPT = MID / "中期答辩.pptx"
FONT_FILE = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_NAME = "微软雅黑"

NAVY = "1B4F8A"
BLUE = "4874CB"
INK = "243044"
MUTED = "5C6B7A"
PALE = "F4F7FB"
ORANGE = "C05621"
GREEN = "1F7A4D"
RED = "A33B3B"

# Spoken script, one block per slide. Target 450-550 non-whitespace characters.
SPEECH = [
    "各位老师好。我是邱添，汇报的题目是「结构鲁棒性对大模型推理能力提升的探究」，指导教师是滕佳烨。",
    "大模型做对一道题，不等于换一种等价写法也做得对。Chen等人在ICML 2024发现，前提顺序一变，准确率可以下降百分之三十以上。我们要分开看两件事：同一张依赖图上，换呈现会不会翻；换一张没见过的图，能力还在不在。",
    "前期做了文献和形式化，也跑过一次真实试测。Qwen2.5-1.5B，二百条题，四个格子准确率都是0.76，正好等于多数类基线。当时只允许生成四个词，模型几乎全答yes。所以这不能说明已经具备结构鲁棒性。中期起改为方法加诊断，用负担引导挖掘，检验均匀随机重排是不是最好。",
    "做法是把潜在任务和文本呈现拆开。评估用见过或没见过的图族，乘以规范或打乱的呈现。训练时按回读次数、相关前提距离和活动依赖前沿，挑选更难的等价呈现，并和随机重排在同样预算下比较。",
    "健全性实验正在运行，正式指标还没落盘，这里不报小样本数字。达标之后才做五种等预算对照。算力是一块8GB的笔记本显卡。",
    "如果挖掘打不赢随机重排，就改写成诊断论文，不把没出现的提升写成结论。表格10月31日前提交，答辩在11月上旬。预期是一篇中英论文和开源代码。",
    "谢谢各位老师，请批评指正。",
]

SLIDE_TITLES = [
    "封面",
    "顺序一变，推理就可能翻",
    "前期工作，和一次没有信号的试测",
    "把稳定性与迁移拆开",
    "实验还在第一道关",
    "风险、截止时间和后续安排",
    "致谢",
]


def nchars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def setup_matplotlib() -> None:
    if FONT_FILE.exists():
        font_manager.fontManager.addfont(str(FONT_FILE))
        # The family name inside msyh.ttc is Microsoft YaHei, not the Chinese alias.
        name = font_manager.FontProperties(fname=str(FONT_FILE)).get_name()
        plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def style_run(run, size: float, bold: bool, color: str) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    run.font.name = FONT_NAME
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        node = rpr.find(qn(tag))
        if node is None:
            node = etree.SubElement(rpr, qn(tag))
        node.set("typeface", FONT_NAME)


def write_lines(shape, lines, size, color, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE) -> None:
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(2)
        p.line_spacing = 1.0
        run = p.add_run()
        run.text = line
        style_run(run, size, bold, color)


def add_textbox(slide, left, top, width, height, lines, size, color, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(left, top, width, height)
    write_lines(box, lines, size, color, bold, align, anchor)
    return box


def add_round(slide, left, top, width, height, fill: str):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background()
    # Less rounded than the default.
    try:
        shape.adjustments[0] = 0.08
    except Exception:
        pass
    return shape


def delete_slide(prs: Presentation, index: int) -> None:
    sld_id_lst = prs.slides._sldIdLst
    sld_id = list(sld_id_lst)[index]
    prs.part.drop_rel(sld_id.get(qn("r:id")))
    sld_id_lst.remove(sld_id)


def strip_placeholders(slide) -> None:
    for shape in list(slide.shapes):
        if shape.is_placeholder:
            element = shape._element
            element.getparent().remove(element)


def place_figure(slide, path: Path, left, top, max_w, max_h):
    """Top-align the figure and return its bottom edge, so the footer can start immediately under it."""
    with Image.open(path) as im:
        aspect = im.size[0] / im.size[1]
    width = int(max_w)
    height = int(width / aspect)
    if height > max_h:
        height = int(max_h)
        width = int(height * aspect)
        left = int(left + (max_w - width) / 2)
    slide.shapes.add_picture(str(path), int(left), int(top), width, height)
    return int(top) + height


def box(ax, x, y, w, h, text, fc, ec, tc="#243044", size=12, weight="medium"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=tc, fontweight=weight)


def save_fig(fig, name: str) -> Path:
    path = ASSETS / name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fig_reorder() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 4.45))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.45)
    ax.axis("off")
    ax.text(6.2, 4.18, "同一道题的两种等价写法", ha="center", va="center", fontsize=16, color="#1B4F8A", fontweight="bold")
    box(ax, 0.35, 0.85, 4.7, 3.05, "", "#F4F7FB", "#4874CB")
    box(ax, 7.35, 0.85, 4.7, 3.05, "", "#FDF6F1", "#C05621")
    ax.text(2.7, 3.55, "证明顺序", ha="center", fontsize=14, color="#1B4F8A", fontweight="bold")
    ax.text(9.7, 3.55, "打乱顺序", ha="center", fontsize=14, color="#C05621", fontweight="bold")
    left_lines = ["若 A 则 B", "若 B 则 C", "A 为真", "问：C 是否成立？"]
    right_lines = ["A 为真", "若 B 则 C", "若 A 则 B", "问：C 是否成立？"]
    for i, (a, b) in enumerate(zip(left_lines, right_lines)):
        y = 2.85 - i * 0.55
        box(ax, 0.7, y, 4.0, 0.46, a, "white", "#C5D4EE", size=12)
        box(ax, 7.7, y, 4.0, 0.46, b, "white", "#F0D2C2", size=12)
    ax.text(2.7, 0.55, "与证明一致，较容易做对", ha="center", fontsize=11, color="#1F7A4D")
    ax.text(9.7, 0.55, "答案不该变，但模型可能翻", ha="center", fontsize=11, color="#A33B3B")
    ax.annotate("", xy=(7.15, 2.3), xytext=(5.25, 2.3), arrowprops=dict(arrowstyle="-|>", color="#1B4F8A", lw=2.0))
    ax.text(6.2, 2.55, "只改顺序", ha="center", fontsize=11, color="#1B4F8A")
    fig.tight_layout(pad=0.2)
    return save_fig(fig, "fig_reorder.png")


def fig_pilot() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 4.45))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.45)
    ax.axis("off")
    ax.text(3.55, 4.15, "四格准确率都是 0.76", ha="center", fontsize=16, color="#1B4F8A", fontweight="bold")
    headers = ["规范呈现", "打乱呈现"]
    rows = ["见过的图", "没见过的图"]
    for j, h in enumerate(headers):
        ax.text(2.15 + j * 2.15, 3.45, h, ha="center", fontsize=12, color="#5C6B7A")
    for i, r in enumerate(rows):
        ax.text(0.95, 2.55 - i * 1.15, r, ha="center", va="center", fontsize=12, color="#243044")
        for j in range(2):
            box(ax, 1.25 + j * 2.15, 2.05 - i * 1.15, 1.85, 0.95, "0.76", "#E7EEF8", "#4874CB", tc="#1B4F8A", size=20, weight="bold")
    # Side facts from the saved pilot, not estimates.
    facts = [
        "等于多数类基线",
        "每格 38 个 yes、12 个 no",
        "稳定性 SS = 1.00",
        "翻转率 SF = 0",
        "换图差距 = 0",
        "最多生成 4 个词，几乎全答 yes",
    ]
    box(ax, 6.55, 0.55, 5.45, 3.35, "", "#F7F9FC", "#D5DEE8")
    ax.text(9.25, 3.5, "这次试测没有结构信号", ha="center", fontsize=13, color="#A33B3B", fontweight="bold")
    for i, fact in enumerate(facts):
        ax.text(6.85, 3.0 - i * 0.38, fact, ha="left", va="center", fontsize=12, color="#243044")
    ax.text(6.2, 0.22, "Qwen2.5-1.5B-Instruct，200 条。来源：experiments/results/mvp_qwen25_1.5b/metrics.json 与 run_metadata.json", ha="center", fontsize=8.5, color="#5C6B7A")
    fig.tight_layout(pad=0.15)
    return save_fig(fig, "fig_pilot.png")


def fig_method() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 4.45))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.45)
    ax.axis("off")
    ax.text(3.15, 4.18, "2×2：先分开，再解释", ha="center", fontsize=14, color="#1B4F8A", fontweight="bold")
    ax.text(9.15, 4.18, "负担引导：挑难的等价呈现", ha="center", fontsize=14, color="#1B4F8A", fontweight="bold")
    ax.text(1.55, 3.45, "规范呈现", ha="center", fontsize=11, color="#5C6B7A")
    ax.text(4.55, 3.45, "打乱呈现", ha="center", fontsize=11, color="#5C6B7A")
    ax.text(0.15, 2.55, "见过", ha="left", va="center", fontsize=11, color="#243044")
    ax.text(0.15, 1.35, "没见过", ha="left", va="center", fontsize=11, color="#243044")
    cells = [
        (0.95, 2.05, "会不会做这道题", "#E7EEF8"),
        (3.15, 2.05, "同图稳不稳\nSS / SF", "#E7EEF8"),
        (0.95, 0.85, "换图还行不行", "#E7EEF8"),
        (3.15, 0.85, "两者的交互", "#FDF6F1"),
    ]
    for x, y, text, fc in cells:
        box(ax, x, y, 2.05, 0.95, text, fc, "#4874CB", size=11)
    steps = ["潜在任务", "等价呈现", "负担 + 翻转", "挑出难的", "对照随机重排"]
    for i, step in enumerate(steps):
        box(ax, 6.55, 3.15 - i * 0.62, 2.55, 0.5, step, "white", "#4874CB", tc="#1B4F8A", size=11, weight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(7.82, 3.12 - i * 0.62), xytext=(7.82, 3.15 - i * 0.62 + 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#4874CB", lw=1.4))
            ax.plot([7.82, 7.82], [3.08 - i * 0.62, 3.15 - (i + 1) * 0.62 + 0.5], color="#4874CB", lw=1.4)
    box(ax, 9.3, 1.55, 2.85, 1.7, "回读次数\n前提距离\n活动前沿", "#F4F7FB", "#1B4F8A", tc="#1B4F8A", size=12, weight="bold")
    ax.text(10.72, 1.25, "三个量都只看相关规则\n在文本里的位置", ha="center", fontsize=9.5, color="#5C6B7A")
    fig.tight_layout(pad=0.15)
    return save_fig(fig, "fig_method.png")


def fig_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 4.45))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.45)
    ax.axis("off")
    stages = [
        ("E0  健全性", "正在运行", "无训练\n开思维链\n每格 ≥ 200", "#C05621", "#FDF6F1"),
        ("E1  主实验", "未开始", "五种条件\n严格等预算\n先看能否迁移", "#1B4F8A", "#E7EEF8"),
        ("E2  消融", "未开始", "去掉负担\n去掉错误引导\n去掉一致性", "#5C6B7A", "#F4F7FB"),
        ("E3  边界", "未开始", "换到算术图\n算力不够\n可以后做", "#5C6B7A", "#F4F7FB"),
    ]
    for i, (title, state, body, ec, fc) in enumerate(stages):
        x = 0.35 + i * 3.05
        box(ax, x, 1.35, 2.8, 2.55, "", fc, ec)
        ax.text(x + 1.4, 3.5, title, ha="center", fontsize=13, color=ec, fontweight="bold")
        ax.text(x + 1.4, 3.05, state, ha="center", fontsize=12, color=ec, fontweight="bold")
        ax.text(x + 1.4, 2.15, body, ha="center", va="center", fontsize=11, color="#243044")
        if i < 3:
            ax.annotate("", xy=(x + 2.95, 2.6), xytext=(x + 2.82, 2.6), arrowprops=dict(arrowstyle="-|>", color="#8AA0B8", lw=1.5))
    ax.text(6.2, 0.85, "E1 的五个条件：不训练 · 规范微调 · 均匀随机重排 · 再加一致性 · 负担引导", ha="center", fontsize=12, color="#1B4F8A")
    ax.text(6.2, 0.4, "判据和效应方向等实验落盘后再写。现在没有 round1 的 metrics.json。", ha="center", fontsize=11, color="#5C6B7A")
    fig.tight_layout(pad=0.15)
    return save_fig(fig, "fig_pipeline.png")


def fig_gantt() -> Path:
    fig, ax = plt.subplots(figsize=(12.4, 4.45))
    # Months from 2026-05 to 2027-02 inclusive = 10 months. x in month index.
    labels = [
        "文献、形式化、试测、重定位",
        "审核 E0，交中期表",
        "中期答辩",
        "E1 等预算主实验",
        "E2 消融 / E3 边界",
        "中英论文与代码",
    ]
    # (start_month_index, duration) May=0 ... Feb=9
    spans = [
        (0, 5.5, "#4874CB"),   # May to mid-Oct
        (5.5, 0.7, "#C05621"),  # late Oct, through 10/31
        (6.0, 0.4, "#A33B3B"),  # early Nov, date unset
        (6.2, 1.0, "#8AA4C8"),  # November
        (7.2, 1.0, "#8AA4C8"),  # December
        (8.0, 2.0, "#8AA4C8"),  # Jan-Feb
    ]
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.55, len(labels) - 0.15)
    for i, ((start, dur, color), label) in enumerate(zip(spans, labels)):
        y = len(labels) - 1 - i
        ax.barh(y, dur, left=start, height=0.55, color=color, zorder=2)
        ax.text(-0.15, y, label, ha="right", va="center", fontsize=11, color="#243044")
    months = ["2026\n5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月", "2027\n1月", "2月"]
    ax.set_xticks([i + 0.5 for i in range(10)])
    ax.set_xticklabels(months, fontsize=10, color="#5C6B7A")
    ax.axvline(5.95, color="#A33B3B", ls="--", lw=1.0, zorder=1)
    ax.text(5.95, len(labels) - 0.15, "10月31日交表", ha="center", va="bottom", fontsize=9, color="#A33B3B")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout(pad=0.35)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.94, bottom=0.16)
    return save_fig(fig, "fig_gantt.png")


def prepare_template_images() -> tuple[Path, Path]:
    inspect = MID / "_inspect" / "media"
    bg_src = inspect / "image6.png"
    logo_src = inspect / "image7.png"
    bg = ASSETS / "template_bg.png"
    logo = ASSETS / "template_header.png"
    shutil.copyfile(bg_src, bg)
    shutil.copyfile(logo_src, logo)
    return bg, logo


def find_text_shape(slide, predicate):
    for shape in slide.shapes:
        if shape.has_text_frame and predicate(shape.text_frame.text or ""):
            return shape
    return None


def content_slide(prs, blank_layout, bg: Path, logo: Path, title: str, lines: list[str], figure: Path, speech: str):
    slide = prs.slides.add_slide(blank_layout)
    strip_placeholders(slide)
    slide.shapes.add_picture(str(bg), Emu(0), Emu(0), prs.slide_width, prs.slide_height)
    # Header logo is a wide transparent PNG. Match the template's aspect (about 2.33).
    slide.shapes.add_picture(str(logo), Inches(0.18), Inches(0.02), Inches(3.55), Inches(1.05))
    add_textbox(
        slide, Inches(3.85), Inches(0.18), Inches(9.1), Inches(0.72),
        [title], 26, NAVY, bold=True, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE,
    )
    # Keep body text to 4 lines. Title is separate and short.
    if len(lines) > 4:
        raise RuntimeError(f"too many body lines on {title}")
    fig_bottom = place_figure(slide, figure, Inches(0.32), Inches(1.02), Inches(12.7), Inches(4.35))
    # Cover everything under the figure, including network lines and the corner watermark.
    bar_top = fig_bottom - int(Inches(0.06))
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), bar_top, prs.slide_width, prs.slide_height - bar_top)
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")
    bar.line.fill.background()
    add_textbox(
        slide, Inches(0.42), bar_top + int(Inches(0.08)), Inches(12.5), Inches(1.55),
        lines, 16, INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
    )
    notes = slide.notes_slide
    notes.notes_text_frame.text = speech
    return slide


def edit_cover(slide, speech: str) -> None:
    date = find_text_shape(slide, lambda t: "月" in t and "日" in t and len(t) < 16)
    person = find_text_shape(slide, lambda t: "汇报人" in t)
    school = find_text_shape(slide, lambda t: "学院" in t)
    if date is None or person is None or school is None:
        raise RuntimeError(f"cover placeholders missing: date={date} person={person} school={school}")
    write_lines(date, ["11月上旬"], 14, NAVY, bold=True, align=PP_ALIGN.CENTER)
    write_lines(person, ["汇报人：邱添"], 14, NAVY, bold=True, align=PP_ALIGN.CENTER)
    write_lines(school, ["统计与管理学院"], 14, NAVY, bold=True, align=PP_ALIGN.CENTER)
    # Title sits on the lower part of the building photo, above the three pills.
    band = add_round(slide, Inches(0.7), Inches(4.15), Inches(11.9), Inches(1.22), NAVY)
    try:
        band.adjustments[0] = 0.15
    except Exception:
        pass
    add_textbox(
        slide, Inches(0.85), Inches(4.2), Inches(11.6), Inches(1.12),
        ["结构鲁棒性对大模型推理能力提升的探究", "拔尖型科研训练项目 · 中期答辩"],
        26, "FFFFFF", bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
    )
    # Second line slightly smaller: rewrite with two runs via paragraphs of different sizes.
    # The helper uses one size. Replace with mixed sizes.
    # Find the textbox we just added: it is the last shape.
    title_box = slide.shapes[-1]
    tf = title_box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = "结构鲁棒性对大模型推理能力提升的探究"
    style_run(r1, 26, True, "FFFFFF")
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "拔尖型科研训练项目 · 中期答辩"
    style_run(r2, 16, False, "E7EEF8")
    slide.notes_slide.notes_text_frame.text = speech


def edit_thanks(slide, speech: str) -> None:
    date = find_text_shape(slide, lambda t: "月" in t and "日" in t and len(t) < 16)
    person = find_text_shape(slide, lambda t: "汇报人" in t)
    school = find_text_shape(slide, lambda t: "学院" in t)
    thanks = find_text_shape(slide, lambda t: "感谢" in t)
    if date is None or person is None or school is None or thanks is None:
        raise RuntimeError("thanks placeholders missing")
    write_lines(date, ["11月上旬"], 14, NAVY, bold=True, align=PP_ALIGN.CENTER)
    write_lines(person, ["汇报人：邱添"], 14, NAVY, bold=True, align=PP_ALIGN.CENTER)
    write_lines(school, ["统计与管理学院"], 14, NAVY, bold=True, align=PP_ALIGN.CENTER)
    # Light plate so the closing line stays readable on the building photo.
    add_round(slide, Inches(2.6), Inches(2.95), Inches(8.1), Inches(1.7), "FFFFFF")
    element = thanks._element
    element.getparent().remove(element)
    slide.shapes._spTree.append(element)
    tf = thanks.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = "感谢聆听"
    style_run(r1, 40, True, NAVY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "欢迎提问"
    style_run(r2, 22, False, BLUE)
    slide.notes_slide.notes_text_frame.text = speech


def build_ppt(bg: Path, logo: Path) -> None:
    prs = Presentation(str(TEMPLATE))
    if len(prs.slides) < 12:
        raise RuntimeError(f"unexpected template slide count: {len(prs.slides)}")
    # Keep the designed cover (0) and the closing slide that mirrors it (original 11).
    for index in range(10, 0, -1):
        delete_slide(prs, index)
    if len(prs.slides) != 2:
        raise RuntimeError(f"expected cover+thanks, got {len(prs.slides)}")
    blank = prs.slide_layouts[6]
    bodies = [
        [
            "等价的前提重排，不该改变答案；高分不等于结构上稳健。",
            "Chen 等，ICML 2024：顺序一变，准确率可下降 30% 以上。这是文献结果。",
            "目标：分开测量同图稳定性，以及换到未见依赖结构后的迁移。",
        ],
        [
            "文献和测量框架已完成；论文改为方法加诊断，不再以基准本身为贡献。",
            "试测四格都是 0.76，等于多数类；当时最多只生成 4 个词。",
            "E0 正在运行。正式指标还没落盘，小样本数字不往这里写。",
        ],
        [
            "潜在任务和文本呈现分开。见过或没见过，乘以规范或打乱。",
            "按回读次数、相关前提距离、活动依赖前沿，挑选困难呈现。",
            "和均匀随机重排做严格等预算对照。打不赢也如实报告。",
        ],
        [
            "E0 无训练：开思维链，比较证明顺序、随机和逆序，每格至少 200 条。",
            "达标之后才做五种对照。现在 round1 还没有 metrics.json。",
            "主模型是 Qwen2.5-1.5B。算力是一块 8GB 的笔记本显卡。",
        ],
        [
            "若打不赢随机重排，就改写诊断论文，不把没出现的提升写成结论。",
            "任务目前是合成规则推理，外部效度有限。10 月 31 日前提交表格。",
            "答辩在 11 月上旬，具体日期未定。预期是中英论文和开源代码。",
        ],
    ]
    figures = [
        ASSETS / "fig_reorder.png",
        ASSETS / "fig_pilot.png",
        ASSETS / "fig_method.png",
        ASSETS / "fig_pipeline.png",
        ASSETS / "fig_gantt.png",
    ]
    for title, lines, figure, speech in zip(SLIDE_TITLES[1:6], bodies, figures, SPEECH[1:6]):
        content_slide(prs, blank, bg, logo, title, lines, figure, speech)
    # New slides were appended after the thanks slide. Move thanks to the end.
    sld_id_lst = prs.slides._sldIdLst
    thanks_node = list(sld_id_lst)[1]
    sld_id_lst.remove(thanks_node)
    sld_id_lst.append(thanks_node)
    edit_cover(prs.slides[0], SPEECH[0])
    edit_thanks(prs.slides[6], SPEECH[6])
    prs.core_properties.title = "结构鲁棒性对大模型推理能力提升的探究"
    prs.core_properties.author = "邱添"
    prs.core_properties.subject = "拔尖型科研训练项目中期答辩"
    prs.save(str(OUT_PPT))


def write_script() -> int:
    total = nchars("".join(SPEECH))
    parts = ["# 中期答辩讲稿", "", "自述按 3 分钟控制。下面只计口播正文，不计本页标题。", ""]
    parts.append(f"口播正文合计 {total} 字（去掉空白后的字符数）。")
    parts.append("")
    for i, (title, speech) in enumerate(zip(SLIDE_TITLES, SPEECH), start=1):
        parts.append(f"## 第 {i} 页　{title}")
        parts.append("")
        parts.append(speech)
        parts.append("")
        parts.append(f"（本页 {nchars(speech)} 字）")
        parts.append("")
    parts.append("## 数字出处")
    parts.append("")
    parts.append("- 四格准确率 0.76、SS = 1.00、SF = 0、换图差距 = 0、每格 38 个 yes 与 12 个 no、共 200 条：`experiments/results/mvp_qwen25_1.5b/metrics.json`。")
    parts.append("- 最多生成 4 个词：`experiments/results/mvp_qwen25_1.5b/run_metadata.json` 的 `max_new_tokens`。同一解释也写在 `experiments/mvp_pilot_report.md`。")
    parts.append("- 准确率可下降 30% 以上：文献结果，Chen 等，ICML 2024，见 `docs/01_literature_review.md`。不是本项目的实测。")
    parts.append("- E0 round1 截至写稿时只有 `build/e0/round1/generations.jsonl`，没有 `metrics.json`。口播和幻灯片都只说正在运行。")
    parts.append("")
    (MID / "讲稿.md").write_text("\n".join(parts), encoding="utf-8")
    return total


def write_qa() -> None:
    items = [
        (
            "和 Chen 等人的 Premise Order Matters（ICML 2024）有什么区别？",
            "他们证明前提顺序会改变准确率，下降可以超过 30%，并把训练留作以后的工作。我们做的是训练干预，而且要把同图翻转和换图失败分开测。30% 是他们的文献结果，不是我们这次测到的。",
        ),
        (
            "和 He 等人（EMNLP 2025）、MEND（ACL Findings 2025）的重排微调有什么区别？",
            "他们已经用随机重排做监督微调。He 等人还报告均匀随机重排在各档上都最好。我们不把重排本身当作新贡献，而是在同样预算下，看按依赖负担挑选的呈现，能不能比随机重排更能迁到没见过的图。",
        ),
        (
            "为什么不说「首次」？",
            "文献综述里核对过的工作，没有把同图稳定性和新图迁移分开评估。这是「据目前检索」，不是已经穷尽全部文献之后的首次声明。如果后面发现已有工作做了这个区分，这个说法要改。",
        ),
        (
            "只用 1.5B，结论能推广到更大的模型吗？",
            "不能直接推广。现在的正式试测只有 Qwen2.5-1.5B，而且那次没有信号。更大模型还没有进入已经落盘的实验。论文里会把适用范围写成我们实际跑过的模型。",
        ),
        (
            "如果负担引导并不比随机重排好，怎么办？",
            "就改成以诊断为主的分析论文，报告随机重排到底改善的是同图稳定性，还是也能迁到新图。不会把没出现的提升写成结论。这个退路在开题后的方案里已经写明，需要的话再请导师确认。",
        ),
        (
            "四格都是 0.76，是不是说明模型已经对顺序不敏感？",
            "不是。每格 38 个 yes、12 个 no，0.76 正好等于多数类。模型几乎全答 yes，所以稳定性等于 1 只说明输出没有翻，不说明它会推理。当时最多只生成 4 个词，没有思维链。",
        ),
        (
            "E0 跑了这么久，为什么幻灯片上没有新数字？",
            "round1 还没有写出 metrics.json。生成过程中的半成品，以及更早的小样本试跑，都不能当成正式结果。所以这里只写正在运行。",
        ),
        (
            "同图稳定性和新图迁移有什么不一样？",
            "同图稳定性看的是同一张依赖图，换成另一种合法呈现之后会不会答翻。新图迁移看的是换一套没见过的依赖结构，规范呈现下还会不会做。只在见过的图上把打乱版本做稳，不能叫学会了推理。",
        ),
        (
            "负担分数会不会只是在挑更长、更深的题？",
            "这三个量只看相关规则在文本里的位置：回读多少、证明相邻步离多远、读的时候要挂起多少条规则。深度用来分层，不放进分数里代替呈现难度。这是设计，还没有被实验证实。",
        ),
        (
            "合成规则题能说明真实的数学推理吗？",
            "不能直接说明。它的好处是答案由求解器核对，呈现变化不会偷偷改答案。外部效度是明确的风险。R-GSM 没有找到公开文件，所以不会写成我们已经评估过的数据。",
        ),
        (
            "等预算具体卡的是什么？",
            "几种训练看的是同一批题，每种挑选呈现的方式不同，但训练词数、步数、LoRA 的秩和更新的层保持一致。避免有的条件只是因为多看了数据才更高。",
        ),
        (
            "一块 8GB 的显卡做不完主实验怎么办？",
            "优先顺序是健全性、主实验、消融、边界实验。种子数或训练题量如果必须减少，会在论文里写明，不把缩小后的实验说成原计划已经做完。",
        ),
        (
            "现在有论文或获奖吗？准备投哪里？",
            "截至这个报告，没有已发表论文、专利、奖励或专著。投稿会议等主实验有结果、也和导师确认之后再定，今天不预告接收或截稿。",
        ),
        (
            "表格里有些格子是空的，还标了黄，是漏了吗？",
            "是故意留空的。项目编号、报告时间、联系电话、项目组成员我没有得到，所以不编。原表没有经费栏，因此没有填金额。指导老师意见、专家意见和学院意见留给导师和评审。",
        ),
    ]
    lines = ["# 评委可能的提问", "", "问答环节大约 5 分钟。每条回答按两三句话来讲，不展开实验细节。", ""]
    for i, (q, a) in enumerate(items, start=1):
        lines.append(f"## {i}. {q}")
        lines.append("")
        lines.append(a)
        lines.append("")
    (MID / "评委问答.md").write_text("\n".join(lines), encoding="utf-8")


def write_form_json() -> dict[str, int]:
    progress = (
        "项目自2026年5月起进行，至今未结束。研究的是：模型在一种写法上做对，并不等于等价的结构变化下仍然做对。"
        "Chen等（ICML 2024）报告，前提顺序偏离证明顺序时，准确率可下降30%以上。本项目把同图上的呈现稳定性，和换到未见依赖结构后的迁移分开测量，并考察训练能否提升这种结构鲁棒性。\n"
        "已完成四项工作。第一，文献调研。已核对的近邻包括Chen等（ICML 2024）、DyVal（ICLR 2024）、GSM-∞（ICML 2025）、iGSM（ICLR 2025），以及He等（EMNLP 2025）和MEND（ACL Findings 2025）的重排微调。"
        "He等报告均匀随机重排在各档上都最好。据目前检索，这些工作没有把同图稳定性与新图迁移分开评估。"
        "第二，测量框架：潜在任务与文本呈现分离，评估为见过或未见图族乘以规范或打乱呈现。"
        "第三，一次真实试测。模型为Qwen2.5-1.5B-Instruct，共200条，四格准确率都是0.76，稳定性为1.00，翻转率为0，换图差距为0。"
        "每格38个yes、12个no，模型几乎全答yes，0.76等于多数类基线。当时最多生成4个词，没有思维链，因此不能说明模型已经具备结构鲁棒性。"
        "数字来自experiments/results/mvp_qwen25_1.5b/metrics.json，生成长度来自同目录run_metadata.json。"
        "第四，据此把论文从以基准为主改为方法与诊断。方法收束为负担引导挖掘：按回读次数、相关前提距离和活动依赖前沿挑选训练呈现，并与均匀随机重排做严格等预算对照。"
        "数据生成与统计代码已实现。健全性实验E0正在运行，正式指标尚未落盘，本报告不使用小样本数字。"
    )
    outcome = (
        "（项目发表论文、获得专利、获得奖励、出版专著、媒体报道等，附成果或证书原件，按照所获成果级别给予一次性奖励）\n"
        "截至2026年10月12日，尚无已发表论文、专利、奖励、专著或媒体报道。已有的是文献综述、测量框架、一次没有信号的真实试测，以及正在运行的健全性实验。"
    )
    problems = (
        "算力目前是一台8GB显存的笔记本GPU，已经落盘的模型实验只有Qwen2.5-1.5B。"
        "负担引导挖掘未必优于均匀随机重排，He等（EMNLP 2025）的结果提示了这种可能。若主实验不支持，将改为以诊断为主的分析论文，不把未观察到的提升写成结论。"
        "当前任务是求解器可验证的合成规则推理，外部效度有限。R-GSM没有找到公开文件，不能写成已评估数据。"
        "E0仍在运行，难度是否达标尚未可知。"
    )
    change = (
        "（项目组成员变更、指导教师变更、项目实施方案变更等）\n"
        "指导教师未变更，仍为滕佳烨。项目组成员信息尚未提供，此处不填变更名单。"
        "研究方案有调整，尚未形成论文：由以合成基准为中心，改为方法与诊断相结合；训练上改为负担引导的困难呈现选择，并用均匀随机重排做等预算对照。"
    )
    later = (
        "中期表格于2026年10月31日前提交。答辩在11月上旬，具体日期未定。"
        "10月下旬审核正在运行的健全性实验；若无信号，只调整难度或呈现，最多三轮。"
        "11月在达标后做等预算主实验，对照为不训练、规范顺序微调、均匀随机重排、随机重排加一致性项，以及负担引导挖掘。"
        "12月视结果做消融，并在算术图上做边界检查；算力不足时优先保证主实验。"
        "2027年1月至2月撰写中英论文并整理开源代码。投稿会议等实验结果和导师确认后再定，此处不预写。"
        "预期成果是一篇技术论文和可复现代码。若方法不优于随机重排，则改为报告诊断结果。"
    )
    cells = [
        {"r": 2, "c": 2, "text": "结构鲁棒性对大模型推理能力提升的探究", "size": 14, "yellow": False},
        {"r": 3, "c": 2, "text": "邱添；学号2024111054", "size": 12, "yellow": True},
        {"r": 4, "c": 2, "text": "滕佳烨", "size": 14, "yellow": False},
        {"r": 5, "c": 2, "text": progress, "size": 12, "yellow": False},
        {"r": 6, "c": 2, "text": outcome, "size": 12, "yellow": False},
        {"r": 7, "c": 2, "text": problems, "size": 12, "yellow": False},
        {"r": 8, "c": 2, "text": change, "size": 12, "yellow": False},
        {"r": 9, "c": 2, "text": later, "size": 12, "yellow": False},
    ]
    yellow_only = [[1, 2], [1, 4], [3, 4], [10, 2], [11, 2], [12, 2], [13, 2]]
    payload = {"cells": cells, "yellow_only": yellow_only}
    (ASSETS / "form_fields.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    counts = {f"R{c['r']}C{c['c']}": nchars(c["text"]) for c in cells}
    return counts


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    fig_reorder()
    fig_pilot()
    fig_method()
    fig_pipeline()
    fig_gantt()
    bg, logo = prepare_template_images()
    total = write_script()
    write_qa()
    counts = write_form_json()
    build_ppt(bg, logo)
    print("SPEECH_CHARS", total)
    print("FORM_CHARS", json.dumps(counts, ensure_ascii=False))
    print("PPT", OUT_PPT, OUT_PPT.stat().st_size)


if __name__ == "__main__":
    main()
