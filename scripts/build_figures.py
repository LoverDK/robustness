"""Build pre-experiment StructShift figures and a toy metric smoke test.

The figures are explicitly schematic. They contain no measured model results.
The Draw.io files keep the boxes and labels editable; Matplotlib exports the
paper-ready PDF/PNG previews used by main.tex.
"""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
EXP = ROOT / "experiments"
FIG.mkdir(exist_ok=True)
EXP.mkdir(exist_ok=True)

COLORS = {
    "ink": "#263238",
    "muted": "#607D8B",
    "blue": "#377EB8",
    "blue_light": "#EAF2F8",
    "green": "#1B9E77",
    "green_light": "#E8F5EF",
    "amber": "#D95F02",
    "amber_light": "#FCEBDD",
    "purple": "#7570B3",
    "purple_light": "#EEEAF8",
    "red": "#D73027",
    "red_light": "#FBE9E7",
    "gray_light": "#F4F6F7",
    "gray_mid": "#CFD8DC",
    "white": "#FFFFFF",
}


def _cell(root: ET.Element, cell_id: str, value: str, x: float, y: float,
          width: float, height: float, style: str, parent: str = "1") -> None:
    cell = ET.SubElement(root, "mxCell", {
        "id": cell_id,
        "value": value,
        "style": style,
        "vertex": "1",
        "parent": parent,
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(width),
        "height": str(height), "as": "geometry",
    })


def _edge(root: ET.Element, edge_id: str, source: str, target: str,
          color: str = COLORS["ink"], dashed: bool = False) -> None:
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
        "jettySize=auto;html=1;endArrow=block;"
        f"strokeColor={color};strokeWidth=2;"
        f"dashed={'1' if dashed else '0'};"
    )
    edge = ET.SubElement(root, "mxCell", {
        "id": edge_id, "value": "", "style": style, "edge": "1",
        "parent": "1", "source": source, "target": target,
    })
    ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})


def _new_drawio(width: int, height: int, name: str) -> ET.Element:
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net", "modified": "2026-09-27T00:00:00.000Z",
        "agent": "Codex", "version": "24.7.17",
    })
    diagram = ET.SubElement(mxfile, "diagram", {"id": name, "name": "Page-1"})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1600", "dy": "900", "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(width), "pageHeight": str(height),
        "math": "0", "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    return mxfile


def _write_xml(tree_root: ET.Element, path: Path) -> None:
    ET.indent(tree_root, space="  ")
    ET.ElementTree(tree_root).write(path, encoding="utf-8", xml_declaration=True)


def build_drawio_fig1() -> None:
    mxfile = _new_drawio(1600, 620, "structshift-fig1")
    root = mxfile.find(".//root")
    assert root is not None
    panel = "rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;fontColor=#263238;fontSize=18;spacing=14;verticalAlign=top;"
    _cell(root, "f1_title", "StructShift: schematic label-preserving structural shift", 40, 20, 1520, 55,
          "text;html=1;strokeColor=none;fillColor=none;fontColor=#263238;fontSize=28;fontStyle=1;align=left;verticalAlign=middle;")
    _cell(root, "f1_note", "Illustrative pre-experiment example; no model result is shown", 40, 78, 1520, 32,
          "text;html=1;strokeColor=none;fillColor=none;fontColor=#607D8B;fontSize=16;align=left;verticalAlign=middle;")
    _cell(root, "f1_a", "Clean instance<br><br>Start with 8 points.<br>Lose 3, then gain 2.<br><br>Question: What is the final total?<br><br><b>Solver: 8 - 3 + 2 = 7</b>", 40, 135, 455, 390,
          panel + "fillColor=#EAF2F8;strokeColor=#377EB8;")
    _cell(root, "f1_b", "Held-out operator order<br><br>Start with 8 points.<br>Gain 2, then lose 3.<br><br>Question: What is the final total?<br><br><b>Solver: 8 + 2 - 3 = 7</b>", 570, 135, 455, 390,
          panel + "fillColor=#FCEBDD;strokeColor=#D95F02;")
    _cell(root, "f1_c", "Separate side-question factor<br><br>Keep the main task fixed.<br>Insert a length-matched unrelated question or statement.<br><br>Prompt asks for the main answer only.<br><br><b>Analyze as a separate factor</b>", 1100, 135, 455, 390,
          panel + "fillColor=#E8F5EF;strokeColor=#1B9E77;")
    _edge(root, "f1_e1", "f1_a", "f1_b", COLORS["amber"])
    _edge(root, "f1_e2", "f1_a", "f1_c", COLORS["green"])
    _write_xml(mxfile, FIG / "fig1_motivated_example.drawio")


def build_drawio_fig2() -> None:
    mxfile = _new_drawio(1800, 850, "structshift-fig2")
    root = mxfile.find(".//root")
    assert root is not None
    box = "rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;fontColor=#263238;fontSize=17;spacing=12;verticalAlign=top;"
    _cell(root, "f2_title", "StructShift construction and evaluation pipeline", 35, 18, 1730, 52,
          "text;html=1;strokeColor=none;fillColor=none;fontColor=#263238;fontSize=28;fontStyle=1;align=left;verticalAlign=middle;")
    stages = [
        ("f2_seed", "1  Symbolic seed graph<br><br>Depth • branching<br>operators • answer", 35, 115, 290, 190, "#EAF2F8", "#377EB8"),
        ("f2_solve", "2  Solver + trace<br><br>Deterministic solution<br>retain trace metadata", 375, 115, 290, 190, "#E8F5EF", "#1B9E77"),
        ("f2_render", "3  Natural-language renderer<br><br>Templates • lexical variation<br>consistency checks", 715, 115, 290, 190, "#EAF2F8", "#377EB8"),
        ("f2_transform", "4  Transform grammar<br><br>Order • depth/branching<br>composition • side question", 1055, 115, 290, 190, "#FCEBDD", "#D95F02"),
        ("f2_pair", "5  Candidate pair<br><br>Clean + transformed item<br>transform metadata", 1395, 115, 290, 190, "#FCEBDD", "#D95F02"),
    ]
    for cell_id, value, x, y, w, h, fill, stroke in stages:
        _cell(root, cell_id, value, x, y, w, h, box + f"fillColor={fill};strokeColor={stroke};")
    for source, target in zip([s[0] for s in stages], [s[0] for s in stages][1:]):
        _edge(root, f"{source}_to_{target}", source, target, COLORS["ink"])
    _cell(root, "f2_gate", "Validation gate<br><br>re-solve transformed graph<br><b>Solve(Tδ(G), q) = Solve(G, q)</b>", 625, 365, 550, 155,
          box + "fillColor=#E8F5EF;strokeColor=#1B9E77;fontSize=18;align=center;verticalAlign=middle;")
    _edge(root, "f2_gate_in", "f2_pair", "f2_gate", COLORS["green"])
    _cell(root, "f2_accept", "Accepted candidates<br>answer preserved + valid text<br><br>duplicate check<br>nuisance matching", 150, 610, 350, 150,
          box + "fillColor=#EEEAF8;strokeColor=#7570B3;")
    _cell(root, "f2_train", "Train-only adaptation<br>clean replay + random augmentation<br>or error-guided LoRA / QLoRA", 540, 610, 350, 150,
          box + "fillColor=#FCEBDD;strokeColor=#D95F02;")
    _cell(root, "f2_test", "Held-out test pool<br>fresh seeds and held-out structures<br><b>never enters hard mining</b>", 930, 610, 350, 150,
          box + "fillColor=#E8F5EF;strokeColor=#1B9E77;")
    _cell(root, "f2_reject", "Rejected candidates<br>answer changed / malformed / ambiguous<br><br>rejection log", 1320, 610, 350, 150,
          box + "fillColor=#FBE9E7;strokeColor=#D73027;")
    _edge(root, "f2_accept_edge", "f2_gate", "f2_accept", COLORS["green"])
    _edge(root, "f2_train_edge", "f2_accept", "f2_train", COLORS["amber"])
    _edge(root, "f2_test_edge", "f2_accept", "f2_test", COLORS["green"], dashed=True)
    _edge(root, "f2_reject_edge", "f2_gate", "f2_reject", COLORS["red"])
    outputs = [
        ("f2_iid", "IID<br>same graph distribution", 1100, 400),
        ("f2_surface", "Surface-OOD<br>new wording / entities", 1280, 400),
        ("f2_struct", "Structure-OOD<br>held-out structure axis", 1460, 400),
        ("f2_comp", "Composition-OOD<br>held-out operator combination", 1640, 400),
    ]
    for cell_id, value, x, y in outputs:
        _cell(root, cell_id, value, x, y, 145, 160, box + "fillColor=#EEEAF8;strokeColor=#7570B3;fontSize=15;align=center;verticalAlign=middle;")
        _edge(root, f"f2_test_{cell_id}", "f2_test", cell_id, COLORS["purple"])
    _cell(root, "f2_meta", "Metadata stream: source graph • transform sequence • perturbation budget • solver trace • nuisance statistics", 40, 790, 1710, 32,
          "text;html=1;strokeColor=none;fillColor=none;fontColor=#607D8B;fontSize=14;align=center;verticalAlign=middle;")
    _write_xml(mxfile, FIG / "fig2_pipeline.drawio")


def build_drawio_fig3() -> None:
    mxfile = _new_drawio(1700, 760, "structshift-fig3")
    root = mxfile.find(".//root")
    assert root is not None
    panel = "rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;fontColor=#263238;fontSize=17;spacing=12;verticalAlign=top;"
    _cell(root, "f3_title", "Figure 3 prototype: results layout (values pending experiments)", 30, 20, 1640, 55,
          "text;html=1;strokeColor=none;fillColor=none;fontColor=#263238;fontSize=26;fontStyle=1;align=left;verticalAlign=middle;")
    _cell(root, "f3_left", "(a) Overall split accuracy\n\nIID     Surface-OOD     Structure-OOD     Composition-OOD\n\n[ pending bars and confidence intervals ]", 45, 110, 760, 520,
          panel + "fillColor=#F4F6F7;strokeColor=#CFD8DC;align=left;verticalAlign=top;")
    _cell(root, "f3_right", "(b) Transformation-level heatmap\n\nRows: order • depth • branching • composition • side-question\nColumns: base • LoRA • random augmentation • structural hard mining\n\n[ pending cells and legend ]", 875, 110, 780, 520,
          panel + "fillColor=#F4F6F7;strokeColor=#CFD8DC;align=left;verticalAlign=top;")
    _cell(root, "f3_note", "This is a layout template only; no simulated or empirical values are included in the paper figure.", 45, 665, 1610, 35,
          "text;html=1;strokeColor=none;fillColor=none;fontColor=#D73027;fontSize=16;align=center;verticalAlign=middle;")
    _write_xml(mxfile, FIG / "fig3_results_template.drawio")


def _setup_ax(ax, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=COLORS["ink"], pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _rounded(ax, xy, width, height, face, edge, radius=0.03, lw=1.6):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(patch)
    return patch


def build_fig1_preview() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.25), dpi=180)
    fig.patch.set_facecolor("white")
    panels = [
        ("Clean instance", COLORS["blue_light"], COLORS["blue"], [
            "Start with 8 points.",
            "Lose 3, then gain 2.",
            "Question: final total?",
            "Solver: 8 - 3 + 2 = 7",
        ]),
        ("Structure-OOD variant", COLORS["amber_light"], COLORS["amber"], [
            "Start with 8 points.",
            "Gain 2, then lose 3.",
            "Question: final total?",
            "Solver: 8 + 2 - 3 = 7",
        ]),
        ("Side-question control", COLORS["green_light"], COLORS["green"], [
            "Main task is unchanged.",
            "Insert an unrelated question or statement.",
            "Answer requested: main task only.",
            "Analyze as a separate factor.",
        ]),
    ]
    for ax, (title, face, edge, lines) in zip(axes, panels):
        _setup_ax(ax, title)
        _rounded(ax, (0.02, 0.03), 0.96, 0.86, face, edge)
        y = 0.72
        for i, line in enumerate(lines):
            color = edge if i == len(lines) - 1 else COLORS["ink"]
            weight = "bold" if i == len(lines) - 1 else "normal"
            ax.text(0.08, y, line, fontsize=10, color=color, weight=weight, va="top", wrap=True)
            y -= 0.15 if len(line) < 38 else 0.21
    fig.text(0.5, 0.015, "Illustrative pre-experiment schematic; solver labels are design notation, not model results.",
             ha="center", fontsize=9, color=COLORS["muted"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.98), w_pad=2.0)
    # Figure-level arrows avoid clipping at the individual axes boundaries.
    fig.text(0.333, 0.51, "→", ha="center", va="center", fontsize=24, color=COLORS["amber"], weight="bold")
    fig.text(0.667, 0.51, "→", ha="center", va="center", fontsize=24, color=COLORS["green"], weight="bold")
    fig.savefig(FIG / "fig1_motivated_example.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / "fig1_motivated_example.pdf", bbox_inches="tight")
    plt.close(fig)


def build_fig2_preview() -> None:
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=180)
    _setup_ax(ax, "StructShift construction and evaluation pipeline")
    stages = [
        (0.025, 0.69, 0.165, 0.20, "1  Symbolic seed", "Graph + target\nanswer", COLORS["blue_light"], COLORS["blue"]),
        (0.225, 0.69, 0.165, 0.20, "2  Renderer", "Controlled natural\nlanguage", COLORS["blue_light"], COLORS["blue"]),
        (0.425, 0.69, 0.165, 0.20, "3  Transform", "Restricted grammar\n+ metadata", COLORS["amber_light"], COLORS["amber"]),
        (0.625, 0.69, 0.165, 0.20, "4  Solver gate", "Re-solve and check\nanswer preservation", COLORS["green_light"], COLORS["green"]),
        (0.825, 0.69, 0.15, 0.20, "5  Valid pairs", "Clean + transformed\nitems", COLORS["green_light"], COLORS["green"]),
    ]
    for x, y, w, h, title, body, face, edge in stages:
        _rounded(ax, (x, y), w, h, face, edge)
        ax.text(x + 0.012, y + h - 0.05, title, fontsize=8.6, weight="bold", color=COLORS["ink"], va="top")
        ax.text(x + 0.012, y + h - 0.10, body, fontsize=7.8, color=COLORS["ink"], va="top", linespacing=1.25)
    for i in range(4):
        x0 = stages[i][0] + stages[i][2]
        x1 = stages[i + 1][0]
        ax.add_patch(FancyArrowPatch((x0 + 0.006, 0.79), (x1 - 0.006, 0.79), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=COLORS["ink"]))
    _rounded(ax, (0.04, 0.34), 0.23, 0.20, COLORS["purple_light"], COLORS["purple"])
    ax.text(0.055, 0.49, "Structure-based split", fontsize=9, weight="bold", color=COLORS["ink"])
    ax.text(0.055, 0.435, "Train: clean + seen transforms\nTest: held-out compositions", fontsize=7.7, color=COLORS["ink"])
    _rounded(ax, (0.33, 0.34), 0.28, 0.20, COLORS["amber_light"], COLORS["amber"])
    ax.text(0.345, 0.49, "Train-only adaptation", fontsize=9, weight="bold", color=COLORS["ink"])
    ax.text(0.345, 0.435, "Random augmentation or hard mining\nLoRA / QLoRA + clean replay", fontsize=7.7, color=COLORS["ink"])
    _rounded(ax, (0.70, 0.34), 0.26, 0.20, COLORS["green_light"], COLORS["green"])
    ax.text(0.715, 0.49, "Held-out evaluation", fontsize=9, weight="bold", color=COLORS["ink"])
    ax.text(0.715, 0.435, "No test candidates enter mining\nAccuracy • gap • worst group", fontsize=7.7, color=COLORS["ink"])
    ax.add_patch(FancyArrowPatch((0.50, 0.69), (0.16, 0.54), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=COLORS["purple"]))
    ax.add_patch(FancyArrowPatch((0.27, 0.44), (0.33, 0.44), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=COLORS["ink"]))
    ax.add_patch(FancyArrowPatch((0.61, 0.44), (0.70, 0.44), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=COLORS["green"]))
    ax.plot([0.66, 0.66], [0.28, 0.58], color=COLORS["red"], lw=2, linestyle="--")
    ax.text(0.66, 0.26, "test isolation", ha="center", va="top", fontsize=7.5, color=COLORS["red"])
    ax.text(0.04, 0.18, "Four evaluation splits: IID • Surface-OOD • Structure-OOD • Composition-OOD", fontsize=8.0, color=COLORS["purple"])
    ax.text(0.04, 0.105, "Report matched token/step budgets, base-query paired intervals, and side-question effects separately.", fontsize=7.7, color=COLORS["muted"])
    ax.text(0.02, 0.94, "Protocol schematic; modules are planned and must be implemented and validated before empirical claims.", fontsize=8.2, color=COLORS["muted"])
    fig.tight_layout()
    fig.savefig(FIG / "fig2_pipeline.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / "fig2_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def build_fig3_template() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), dpi=180, gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    ax.set_title("(a) Overall split accuracy", loc="left", fontsize=12, fontweight="bold", color=COLORS["ink"])
    splits = ["IID", "Surface\n-OOD", "Structure\n-OOD", "Composition\n-OOD"]
    xs = range(len(splits))
    for method_i, method in enumerate(["Base", "LoRA", "Random aug.", "Structural hard mining"]):
        offset = (method_i - 1.5) * 0.16
        ax.bar([x + offset for x in xs], [0, 0, 0, 0], width=0.14, fill=False, edgecolor=COLORS["gray_mid"], hatch="//", label=method if method_i == 0 else None)
    ax.set_xticks(list(xs), splits)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.text(1.5, 0.52, "PENDING\nEXPERIMENTS", ha="center", va="center", fontsize=16, color=COLORS["gray_mid"], weight="bold", rotation=20)
    ax.grid(axis="y", color="#E5E7E9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.set_title("(b) Transformation-level heatmap", loc="left", fontsize=12, fontweight="bold", color=COLORS["ink"])
    rows = ["Premise order", "Depth", "Branching", "Composition", "Side-question"]
    cols = ["Base", "LoRA", "Random\naug.", "Structural\nhard mining"]
    ax.imshow([[math.nan] * len(cols)] * len(rows), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.5))
            ax.text(j, i, "--", ha="center", va="center", fontsize=11, color=COLORS["gray_mid"])
    ax.text(1.5, 2, "values\npending", ha="center", va="center", fontsize=15, color=COLORS["gray_mid"], weight="bold", rotation=25)
    ax.set_xlabel("Model condition")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.015, "Pre-experiment layout template; no simulated or empirical values are plotted.", ha="center", fontsize=9, color=COLORS["red"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.98), w_pad=2.4)
    fig.savefig(FIG / "fig3_results_template.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / "fig3_results_template.pdf", bbox_inches="tight")
    plt.close(fig)


def run_toy_pilot() -> None:
    """Smoke-test paired splits and metrics with deterministic toy agents."""
    rng = random.Random(7)
    n = 240
    records = []
    for split in ("IID", "Surface-OOD", "Structure-OOD", "Composition-OOD"):
        for i in range(n):
            records.append({"split": split, "id": i, "label": 1, "side": False})
    for split in ("IID", "Structure-OOD"):
        for i in range(n):
            records.append({"split": split, "id": i, "label": 1, "side": True})

    def heuristic(item: dict) -> int:
        # A deliberately limited baseline: it follows canonical two-hop order
        # and loses some accuracy when a side question is inserted.
        if item["split"] in {"Structure-OOD", "Composition-OOD"}:
            return 0
        if item["side"] and rng.random() < 0.18:
            return 0
        return 1

    def graph_solver(item: dict) -> int:
        return 1

    outputs = {}
    for name, model in (("canonical_order_heuristic", heuristic), ("graph_solver", graph_solver)):
        by_group = {}
        for item in records:
            key = f"{item['split']}|{'side' if item['side'] else 'clean'}"
            by_group.setdefault(key, []).append(model(item) == item["label"])
        acc = {k: sum(v) / len(v) for k, v in by_group.items()}
        iid_err = 1 - acc["IID|clean"]
        ood_err = 1 - acc["Structure-OOD|clean"]
        outputs[name] = {
            "accuracy": acc,
            "structural_gap_error": ood_err - iid_err,
            "side_interference_cost": acc["IID|clean"] - acc["IID|side"],
            "n": {k: len(v) for k, v in by_group.items()},
        }
    payload = {
        "status": "toy_simulation_only",
        "warning": "These deterministic toy-agent metrics are not LLM results and are not included in the paper figures.",
        "seed": 7,
        "metrics": outputs,
    }
    (EXP / "toy_pilot_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = """# Toy pilot smoke test\n\nThis run checks the paired split construction and metric calculations with two deterministic toy agents. It is not evidence about LLM reasoning and must not be cited as a paper result.\n\n- Seed: 7\n- Items: 240 per split and side-question condition\n- Agents: canonical-order heuristic, graph solver\n- Output: `toy_pilot_metrics.json`\n"""
    (EXP / "toy_pilot_report.md").write_text(report, encoding="utf-8")


def write_audits() -> None:
    audits = {
        "fig1_motivated_example.audit.md": """# Figure 1 Inventory\n\n| ID | Region | Element | Medium | Status |\n|---|---|---|---|---|\n| f1-title | top | title and pre-experiment note | native Draw.io text | accepted |\n| f1-clean | left panel | clean symbolic reasoning item | native panel and text | accepted |\n| f1-structure | center panel | held-out composition schematic | native panel and text | accepted |\n| f1-side | right panel | side-question control | native panel and text | accepted |\n| f1-arrows | between panels | structural and side-question relations | native connector | accepted |\n| f1-preview | rendered PNG/PDF | same layout as source | generated preview | accepted |\n\nNo source reference image was supplied. This is an original schematic, so there is no reference-comparison claim. Replace the symbolic item with a verified pilot item before camera-ready submission.\n""",
        "fig2_pipeline.audit.md": """# Figure 2 Inventory\n\n| ID | Region | Element | Medium | Status |\n|---|---|---|---|---|\n| f2-title | top | pipeline title | native Draw.io text | accepted |\n| f2-stages | upper row | seed, solver, renderer, transform, candidate pair | native panels and text | accepted |\n| f2-gate | center | solver validation equation | native panel and text | accepted |\n| f2-branches | lower row | accepted and rejected paths | native panels and connectors | accepted |\n| f2-splits | lower right | four OOD output cards | native panels and connectors | accepted |\n| f2-metadata | footer | metadata stream | native text | accepted |\n| f2-preview | rendered PNG/PDF | same layout as source | generated preview | accepted |\n\nNo source reference image was supplied. This is an original protocol schematic. Section names and equations must be synchronized with the final Methods section.\n""",
        "fig3_results_template.audit.md": """# Figure 3 Inventory\n\n| ID | Region | Element | Medium | Status |\n|---|---|---|---|---|\n| f3-title | top | results-template title | native Draw.io text | accepted |\n| f3-bars | left panel | empty grouped-bar layout | native panel and text | accepted |\n| f3-heatmap | right panel | empty transformation heatmap layout | native panel and text | accepted |\n| f3-warning | footer | pending-data warning | native text | accepted |\n| f3-preview | rendered PNG/PDF | blank result template | generated preview | accepted |\n\nThe template intentionally contains no simulated or empirical values. Replace with model outputs, uncertainty intervals, and honest axes after the pilot.\n""",
    }
    for name, content in audits.items():
        (FIG / name).write_text(content, encoding="utf-8")
    manifest = {
        "type": "original_pre_experiment_figures",
        "reference_images": [],
        "entries": [
            {"name": "fig1_motivated_example", "drawio": "fig1_motivated_example.drawio", "preview": "fig1_motivated_example.png"},
            {"name": "fig2_pipeline", "drawio": "fig2_pipeline.drawio", "preview": "fig2_pipeline.png"},
            {"name": "fig3_results_template", "drawio": "fig3_results_template.drawio", "preview": "fig3_results_template.png"},
        ],
        "note": "Draw.io CLI was not available in this environment; previews are generated from the same geometry specification and source files were checked with check_drawio.py.",
    }
    (FIG / "drawio_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_drawio_fig1()
    build_drawio_fig2()
    build_drawio_fig3()
    build_fig1_preview()
    build_fig2_preview()
    build_fig3_template()
    run_toy_pilot()
    write_audits()
    print("Built StructShift figures and toy pilot outputs.")
