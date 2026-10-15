"""CPU rescore of saved E1 logic answers.

Reads existing logic.jsonl files, rebuilds test metadata, and writes
experiments/e1/rescore/. It does not load a model, train, or generate.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .burden import burden
from .e1 import PROJECT, build_eval_items, build_pools, graph_hash, read_jsonl
from .logic import render
from .prompts import extract_yes_no
from .stats import conditional_flip_ci, group_mean_diff_ci, mean_ci

PRESENTATIONS = ("aligned", "reversed")
FEATURES = ("back_reads", "d_rel", "f_active")
CELLS = ("seen_aligned", "seen_shifted", "unseen_aligned", "unseen_shifted")
CELL_SPEC = {
    "seen_aligned": ("seen", "aligned"),
    "seen_shifted": ("seen", "reversed"),
    "unseen_aligned": ("unseen", "aligned"),
    "unseen_shifted": ("unseen", "reversed"),
}
N_BOOT = 2000
BOOT_SEED = 0
ALPHA = 0.05
ACC_TOL = 1e-12
OUT = PROJECT / "experiments" / "e1" / "rescore"
ROUNDS = (
    {
        "name": "pilot",
        "has_b0": True,
        "conditions": ("b0", "b1", "b2", "b3", "ours"),
        "logic_root": PROJECT / "build" / "e1" / "pilot",
        "metrics_root": PROJECT / "experiments" / "e1" / "pilot",
    },
    {
        "name": "pilot_mix15",
        "has_b0": False,
        "conditions": ("b1", "b2", "b3", "ours"),
        "logic_root": PROJECT / "build" / "e1" / "pilot_mix15",
        "metrics_root": PROJECT / "experiments" / "e1" / "pilot_mix15",
    },
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z0-9_])")


def gap_and_interaction(seen, unseen, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = ALPHA):
    """Δ_T(p) = acc(seen, p) - acc(unseen, p), with independent pool resamples.

    The interaction Δ_T(reversed) - Δ_T(aligned) uses that same pair of
    resamples. Same estimator as transfer.gap_and_interaction, restricted to
    aligned and reversed.
    """
    seen_v = {name: np.asarray(seen[name], dtype=float) for name in PRESENTATIONS}
    unseen_v = {name: np.asarray(unseen[name], dtype=float) for name in PRESENTATIONS}
    n_seen = int(seen_v["aligned"].size)
    n_unseen = int(unseen_v["aligned"].size)
    if any(seen_v[name].size != n_seen for name in PRESENTATIONS):
        raise ValueError("seen presentations must share one latent-task order")
    if any(unseen_v[name].size != n_unseen for name in PRESENTATIONS):
        raise ValueError("unseen presentations must share one latent-task order")
    if n_seen == 0 or n_unseen == 0:
        return {name: None for name in PRESENTATIONS}, {"reversed": None}
    point = {
        name: float(seen_v[name].mean() - unseen_v[name].mean())
        for name in PRESENTATIONS
    }
    rng = np.random.default_rng(seed)
    idx_seen = rng.integers(0, n_seen, size=(n_boot, n_seen))
    idx_unseen = rng.integers(0, n_unseen, size=(n_boot, n_unseen))
    boots = {
        name: seen_v[name][idx_seen].mean(axis=1) - unseen_v[name][idx_unseen].mean(axis=1)
        for name in PRESENTATIONS
    }
    delta = {
        name: [point[name], float(np.quantile(boots[name], alpha / 2)), float(np.quantile(boots[name], 1 - alpha / 2))]
        for name in PRESENTATIONS
    }
    gap = boots["reversed"] - boots["aligned"]
    interaction_point = point["reversed"] - point["aligned"]
    interaction = {
        "reversed": [
            interaction_point,
            float(np.quantile(gap, alpha / 2)),
            float(np.quantile(gap, 1 - alpha / 2)),
        ]
    }
    return delta, interaction


def _interval(triple):
    if triple[0] is None:
        return None
    return [float(triple[0]), float(triple[1]), float(triple[2])]


def accuracy_interval(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    return _interval(mean_ci(values, n_boot=n_boot, seed=seed))


def sf_interval(aligned, reversed_, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    return _interval(conditional_flip_ci(aligned, reversed_, n_boot=n_boot, seed=seed))


def sf_count(aligned, reversed_) -> float | None:
    """SF = count(aligned correct and reversed wrong) / count(aligned correct)."""
    canon = np.asarray(aligned, dtype=float)
    shifted = np.asarray(reversed_, dtype=float)
    denom = int((canon == 1).sum())
    if denom == 0:
        return None
    return float(((canon == 1) & (shifted == 0)).sum() / denom)


def _stop(message: str) -> None:
    print(message, flush=True)
    raise SystemExit(2)


def _py(obj):
    if isinstance(obj, dict):
        return {str(key): _py(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(value) for value in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _feature_summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "constant": False, "value": None}
    constant = bool(float(arr.max()) - float(arr.min()) <= ACC_TOL)
    mean = float(arr.mean())
    return {
        "n": int(arr.size),
        "mean": mean,
        "constant": constant,
        "value": mean if constant else None,
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(values.mean())


def _relation(left, right) -> str:
    if left[2] < right[1] or right[2] < left[1]:
        return "disjoint"
    return "overlap"


def _read_generation_file(path: Path) -> dict:
    """Last line for a uid wins, matching e1.read_jsonl."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row)
    identical = 0
    text_conflict = 0
    answer_conflict = 0
    for group in grouped.values():
        if len(group) == 1:
            continue
        texts = [item.get("generation") for item in group]
        answers = [extract_yes_no(text) if isinstance(text, str) else None for text in texts]
        if len(set(texts)) == 1:
            identical += 1
        else:
            text_conflict += 1
        if len(set(answers)) > 1:
            answer_conflict += 1
    gens = {uid: group[-1] for uid, group in grouped.items()}
    if gens != read_jsonl(path):
        _stop(f"最后一行与 read_jsonl 不一致: {path}")
    return {
        "gens": gens,
        "n_lines": len(rows),
        "n_uid": len(grouped),
        "n_duplicate_uid": identical + text_conflict,
        "n_identical_duplicate_uid": identical,
        "n_text_conflict_uid": text_conflict,
        "n_answer_conflict_uid": answer_conflict,
    }


def _mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def _expected_quant(condition: str) -> str:
    if condition == "b0":
        return "bf16-eval"
    return "nf4"


def _build_metadata():
    if "torch" in sys.modules:
        _stop("torch is already imported; refusing to continue")
    try:
        pools, summary = build_pools(seed=0)
    except RuntimeError as exc:
        _stop("池隔离失败: " + str(exc))
    if not summary.get("ok"):
        _stop("池隔离失败: " + json.dumps(summary.get("overlaps", {}), ensure_ascii=False))
    meta = {}
    latents = []
    for split, pool_name in (("seen", "test_seen"), ("unseen", "test_unseen")):
        _items, part = build_eval_items(pools[pool_name], split)
        meta.update(part)
        for lat in pools[pool_name]:
            digest = graph_hash(lat)
            features = {}
            for presentation in PRESENTATIONS:
                rendered = render(lat, presentation)
                raw = burden(lat, rendered["core_positions"])
                features[presentation] = {name: float(raw[name]) for name in FEATURES}
                features[presentation]["n_core"] = int(raw["n_core"])
            latents.append({
                "split": split,
                "family": lat.family,
                "graph_hash": digest,
                "label": lat.label,
                "features": features,
            })
    for row in meta.values():
        if row["cell"] not in CELL_SPEC:
            _stop(f"unexpected cell {row['cell']}")
        split, presentation = CELL_SPEC[row["cell"]]
        if row["presentation"] != presentation:
            _stop(f"{row['uid']} presentation is {row['presentation']}, expected {presentation}")
        if row["uid"] != f"{row['cell']}|{row['graph_hash']}":
            _stop(f"uid does not match cell|graph_hash: {row['uid']}")
        if not row["cell"].startswith(split + "_"):
            _stop(f"cell split mismatch: {row['cell']}")
    by_hash = {(row["split"], row["graph_hash"]): row for row in latents}
    if len(by_hash) != len(latents):
        _stop("duplicate graph hash inside a test pool")
    for row in meta.values():
        split, presentation = CELL_SPEC[row["cell"]]
        lat = by_hash.get((split, row["graph_hash"]))
        if lat is None:
            _stop(f"metadata hash missing from pool: {row['uid']}")
        if lat["label"] != row["label"]:
            _stop(f"label mismatch for {row['uid']}")
        if lat["features"][presentation]["back_reads"] != float(row["back_reads"]):
            _stop(f"back_reads mismatch for {row['uid']}")
        if lat["features"][presentation]["n_core"] != int(row["n_core"]):
            _stop(f"n_core mismatch for {row['uid']}")
    meta_hashes = {(CELL_SPEC[row["cell"]][0], row["graph_hash"]) for row in meta.values()}
    if meta_hashes != set(by_hash):
        _stop("test metadata does not cover the rebuilt pools")
    return pools, summary, meta, latents


def _score_condition(meta: dict, gens: dict) -> dict:
    missing = [uid for uid in meta if uid not in gens]
    if missing:
        preview = ", ".join(missing[:5])
        _stop(f"缺生成行 {len(missing)} 条，例如 {preview}")
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    for uid, row in meta.items():
        generated = gens[uid].get("generation")
        if not isinstance(generated, str):
            _stop(f"generation 字段缺失: {uid}")
        pred = extract_yes_no(generated)
        correct = int(pred == row["label"])
        split, presentation = CELL_SPEC[row["cell"]]
        grouped[row["cell"]].append({
            "correct": correct,
            "label": row["label"],
        })
        slot = paired[(split, row["graph_hash"])]
        if presentation in slot:
            _stop(f"duplicate presentation for {row['graph_hash']}")
        slot[presentation] = correct
    cells = {}
    for cell in CELLS:
        rows = grouped[cell]
        acc = float(np.mean([item["correct"] for item in rows]))
        cells[cell] = {
            "acc": acc,
            "n": len(rows),
            "n_yes": int(sum(item["label"] == "yes" for item in rows)),
            "n_no": int(sum(item["label"] == "no" for item in rows)),
        }
    return {"cells": cells, "paired": paired}


def _vectors(paired: dict, latents: list[dict], split: str):
    hashes = sorted(row["graph_hash"] for row in latents if row["split"] == split)
    if len(hashes) != len(set(hashes)):
        _stop(f"duplicate hash in {split}")
    aligned, reversed_ = [], []
    for digest in hashes:
        slot = paired.get((split, digest))
        if slot is None or set(slot) != set(PRESENTATIONS):
            _stop(f"潜在任务配对不完整: {split} {digest}")
        aligned.append(slot["aligned"])
        reversed_.append(slot["reversed"])
    return hashes, np.asarray(aligned, dtype=float), np.asarray(reversed_, dtype=float)


def _condition_estimates(paired: dict, latents: list[dict]) -> dict:
    vectors = {}
    acc = {}
    sf = {}
    counts = {}
    for split in ("seen", "unseen"):
        _hashes, aligned, reversed_ = _vectors(paired, latents, split)
        vectors[split] = {"aligned": aligned, "reversed": reversed_}
        acc[split] = {
            "aligned": accuracy_interval(aligned),
            "reversed": accuracy_interval(reversed_),
        }
        sf_hat = sf_interval(aligned, reversed_)
        counted = sf_count(aligned, reversed_)
        if sf_hat is None or counted is None or abs(sf_hat[0] - counted) > ACC_TOL:
            _stop(f"{split} SF 点估计与计数公式不一致")
        sf[split] = sf_hat
        n_aligned_correct = int((aligned == 1).sum())
        n_flip = int(((aligned == 1) & (reversed_ == 0)).sum())
        counts[split] = {
            "n": int(aligned.size),
            "n_aligned_correct": n_aligned_correct,
            "n_flip": n_flip,
            "n_stay": n_aligned_correct - n_flip,
        }
        for presentation, values in (("aligned", aligned), ("reversed", reversed_)):
            if abs(acc[split][presentation][0] - float(values.mean())) > ACC_TOL:
                _stop(f"{split} {presentation} 准确率区间点估计不是均值")
    delta, interaction = gap_and_interaction(vectors["seen"], vectors["unseen"])
    for presentation in PRESENTATIONS:
        expected = float(vectors["seen"][presentation].mean() - vectors["unseen"][presentation].mean())
        if abs(delta[presentation][0] - expected) > ACC_TOL:
            _stop(f"Δ_T({presentation}) 点估计与计数公式不一致")
    expected_interaction = delta["reversed"][0] - delta["aligned"][0]
    if abs(interaction["reversed"][0] - expected_interaction) > ACC_TOL:
        _stop("交互点估计不是同一次定义下的差值")
    return {
        "acc": acc,
        "SF": sf,
        "delta_T": delta,
        "interaction": interaction,
        "counts": counts,
        "vectors": vectors,
    }


def _burden_tables(latents: list[dict]) -> dict:
    pools = {}
    for split in ("seen", "unseen"):
        group = [row for row in latents if row["split"] == split]
        families = sorted({row["family"] for row in group})
        presentations = {}
        for presentation in PRESENTATIONS:
            presentations[presentation] = {
                feature: _feature_summary([row["features"][presentation][feature] for row in group])
                for feature in FEATURES
            }
        n_core = sorted({row["features"]["aligned"]["n_core"] for row in group})
        pools[split] = {
            "n": len(group),
            "families": families,
            "n_core_values": n_core,
            "presentations": presentations,
        }
    families = {}
    for row in latents:
        families.setdefault(row["family"], []).append(row)
    family_out = {}
    for family in ("chain", "conj_fact", "tree", "diamond"):
        group = families.get(family, [])
        if not group:
            _stop(f"图族为空: {family}")
        splits = {row["split"] for row in group}
        if len(splits) != 1:
            _stop(f"图族跨 seen/unseen: {family}")
        n_core = sorted({row["features"]["aligned"]["n_core"] for row in group})
        presentations = {}
        for presentation in PRESENTATIONS:
            presentations[presentation] = {
                feature: _feature_summary([row["features"][presentation][feature] for row in group])
                for feature in FEATURES
            }
        family_out[family] = {
            "pool": next(iter(splits)),
            "n": len(group),
            "n_core_values": n_core,
            "presentations": presentations,
        }
    return {"pools": pools, "families": family_out}


def _flip_tables(latents: list[dict], estimates: dict, burden_info: dict) -> dict:
    by_family = defaultdict(list)
    for row in latents:
        by_family[row["family"]].append(row)
    out = {}
    for family, group in by_family.items():
        group = sorted(group, key=lambda row: row["graph_hash"])
        hashes = [row["graph_hash"] for row in group]
        split = group[0]["split"]
        index = {digest: i for i, digest in enumerate(estimates["_hashes"][split])}
        positions = [index[digest] for digest in hashes]
        aligned = estimates["vectors"][split]["aligned"][positions]
        reversed_ = estimates["vectors"][split]["reversed"][positions]
        chosen = aligned == 1
        flipped = chosen & (reversed_ == 0)
        stayed = chosen & (reversed_ == 1)
        feature_means = {}
        for feature in FEATURES:
            values = np.asarray([row["features"]["reversed"][feature] for row in group], dtype=float)
            summary = burden_info["families"][family]["presentations"]["reversed"][feature]
            flip_mean = _mean_or_none(values[flipped])
            stay_mean = _mean_or_none(values[stayed])
            if summary["constant"]:
                for mean in (flip_mean, stay_mean):
                    if mean is not None and abs(mean - summary["value"]) > ACC_TOL:
                        _stop(f"{family} {feature} 常数与分组均值不一致")
            feature_means[feature] = {
                "constant_in_family": bool(summary["constant"]),
                "flip_mean": flip_mean,
                "stay_mean": stay_mean,
            }
        out[family] = {
            "pool": split,
            "n_aligned_correct": int(chosen.sum()),
            "n_flip": int(flipped.sum()),
            "n_stay": int(stayed.sum()),
            "reversed_burden_mean": feature_means,
        }
    return out


def _check_cells(scored: dict, original: dict) -> list[dict]:
    rows = []
    original_cells = original["cells"]
    for cell in CELLS:
        got = scored["cells"][cell]
        src = original_cells[cell]
        diff = abs(got["acc"] - float(src["acc"]))
        match = (
            diff <= ACC_TOL
            and got["n"] == int(src["n"])
            and got["n_yes"] == int(src["n_yes"])
            and got["n_no"] == int(src["n_no"])
        )
        rows.append({
            "cell": cell,
            "acc": got["acc"],
            "original_acc": float(src["acc"]),
            "abs_diff": diff,
            "n": got["n"],
            "n_yes": got["n_yes"],
            "n_no": got["n_no"],
            "original_n": int(src["n"]),
            "original_n_yes": int(src["n_yes"]),
            "original_n_no": int(src["n_no"]),
            "match": match,
        })
    return rows


def _attach_hashes(estimates: dict, latents: list[dict]) -> None:
    estimates["_hashes"] = {}
    for split in ("seen", "unseen"):
        estimates["_hashes"][split] = sorted(row["graph_hash"] for row in latents if row["split"] == split)


def _protected_paths() -> list[Path]:
    paths = []
    for spec in ROUNDS:
        for condition in spec["conditions"]:
            paths.append(spec["logic_root"] / condition / "logic.jsonl")
            paths.append(spec["metrics_root"] / condition / "metrics.json")
    return paths


def _fmt(value) -> str:
    if isinstance(value, bool):
        raise TypeError("refusing to format a boolean as a report number")
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return json.dumps(float(value))
    raise TypeError(type(value))


def _fmt_interval(values) -> str:
    if values is None:
        return "null"
    return "[" + ", ".join(_fmt(item) for item in values) + "]"


def _assert_report_numbers(report: str, json_text: str) -> None:
    missing = []
    for token in _NUMBER.findall(report):
        if token not in json_text:
            missing.append(token)
    if missing:
        unique = list(dict.fromkeys(missing))
        _stop("报告里的数字没有写进 json: " + ", ".join(unique[:40]))


def _render_report(payload: dict) -> str:
    lines = [
        "# E1 已有答卷的 CPU 重算",
        "",
        "本轮只读已有 `logic.jsonl`，按种子 "
        + _fmt(payload["config"]["seed"])
        + " 重建测试池后重新计分。没有加载模型，没有训练，没有新的生成。",
        "",
        "B0 的 `quant` 是 " + payload["quant"]["b0"] + "。B1、B2、B3、Ours 的 `quant` 是 "
        + payload["quant"]["trained"] + "。`pilot_mix15` 没有 B0。",
        "",
        "区间写成 [point, low, high]。准确率用 `stats.mean_ci`，SF 用 `stats.conditional_flip_ci`，"
        "Δ_T 与交互按 `transfer.gap_and_interaction` 的写法：两个池独立重采样，交互用同一次重采样。"
        "重采样次数为 " + _fmt(payload["config"]["n_boot"])
        + "，种子为 " + _fmt(payload["config"]["seed"])
        + "，alpha 为 " + _fmt(payload["config"]["alpha"]) + "。",
        "",
        "SF = P(reversed 答错 | aligned 答对)。Δ_T(p) = acc(seen, p) - acc(unseen, p)。"
        "交互 = Δ_T(reversed) - Δ_T(aligned)。潜在任务按图哈希排序后配对。shifted 的 presentation 是 reversed。",
        "",
        "字段路径：",
        "",
        "- 准确率：`rounds.<轮次>.conditions.<条件>.acc.<seen|unseen>.<aligned|reversed>`",
        "- SF：`rounds.<轮次>.conditions.<条件>.SF.<seen|unseen>`",
        "- Δ_T：`rounds.<轮次>.conditions.<条件>.delta_T.<aligned|reversed>`",
        "- 交互：`rounds.<轮次>.conditions.<条件>.interaction.reversed`",
        "- 四格核对：`cell_check.rows`",
        "- 池负担：`burden.pools.<seen|unseen>.presentations.<aligned|reversed>.<特征>`",
        "- 图族负担：`burden.families.<图族>.presentations.<aligned|reversed>.<特征>`",
        "- 改口分组均值：`burden.by_condition.<轮次>.<条件>.<图族>.reversed_burden_mean.<特征>`",
        "",
        "## 四格核对",
        "",
    ]
    if payload["cell_check"]["all_match"]:
        lines.append(
            "四个格子的准确率、n、n_yes、n_no 与原 metrics 的 cells 一致。"
            "准确率最大绝对差为 " + _fmt(payload["cell_check"]["max_abs_acc_diff"])
            + "，容许 " + _fmt(payload["cell_check"]["tolerance"]) + "。"
        )
    else:
        lines.append("四格核对没有通过。")
    lines.append("")
    uniform = payload["cell_check"]["uniform_counts"]
    if uniform["same"]:
        lines.append(
            "每个格子的 n 为 " + _fmt(uniform["n"])
            + "，n_yes 为 " + _fmt(uniform["n_yes"])
            + "，n_no 为 " + _fmt(uniform["n_no"]) + "。"
        )
    lines.append("")
    lines.append("## 准确率、SF、Δ_T、交互")
    lines.append("")
    for round_name, round_payload in payload["rounds"].items():
        lines.append("### " + round_name)
        lines.append("")
        if not round_payload["has_b0"]:
            lines.append("这一轮没有 B0。")
            lines.append("")
        for condition, block in round_payload["conditions"].items():
            lines.append("#### " + condition)
            lines.append("")
            lines.append("quant 为 `" + block["quant"] + "`。")
            if block["n_duplicate_uid"]:
                lines.append(
                    "logic.jsonl 有 " + _fmt(block["n_logic_lines"])
                    + " 行、" + _fmt(block["n_logic_uid"])
                    + " 个 uid，重复 uid 有 " + _fmt(block["n_duplicate_uid"])
                    + " 个；其中生成文本相同的有 " + _fmt(block["n_identical_duplicate_uid"])
                    + " 个，文本不同的有 " + _fmt(block["n_text_conflict_uid"])
                    + " 个，抽出的答案不同的有 " + _fmt(block["n_answer_conflict_uid"])
                    + " 个。计分保留每个 uid 的最后一行。"
                )
            lines.append("")
            for split in ("seen", "unseen"):
                for presentation in PRESENTATIONS:
                    lines.append(
                        split + " " + presentation + " 准确率 "
                        + _fmt_interval(block["acc"][split][presentation])
                    )
                lines.append("SF " + split + " " + _fmt_interval(block["SF"][split]))
                counts = block["counts"][split]
                lines.append(
                    split + " 的 n 为 " + _fmt(counts["n"])
                    + "，aligned 做对 " + _fmt(counts["n_aligned_correct"])
                    + "，其中改口 " + _fmt(counts["n_flip"])
                    + "，未改口 " + _fmt(counts["n_stay"]) + "。"
                )
            for presentation in PRESENTATIONS:
                lines.append(
                    "Δ_T(" + presentation + ") " + _fmt_interval(block["delta_T"][presentation])
                )
            lines.append("交互 " + _fmt_interval(block["interaction"]["reversed"]))
            lines.append("")
    lines.extend([
        "## 负担",
        "",
        "负担由 `burden.burden` 在每个潜在任务的 aligned 与 reversed 上计算。"
        "seen 与 unseen 分开。改口分组只在同一图族内比较，图族是 chain、conj_fact、tree、diamond。"
        "分组限定为 aligned 做对的题，比较的是 reversed 的 back_reads、d_rel、f_active。",
        "",
        "### seen 与 unseen",
        "",
    ])
    for split, block in payload["burden"]["pools"].items():
        lines.append(
            split + " 有 " + _fmt(block["n"]) + " 个潜在任务，图族为 "
            + "、".join(block["families"]) + "，n_core 取值为 "
            + "、".join(_fmt(value) for value in block["n_core_values"]) + "。"
        )
        for presentation in PRESENTATIONS:
            for feature in FEATURES:
                summary = block["presentations"][presentation][feature]
                if summary["constant"]:
                    lines.append(
                        split + " 的 " + presentation + " " + feature
                        + " 是常数 " + _fmt(summary["value"])
                        + "（n 为 " + _fmt(summary["n"]) + "）。"
                    )
                else:
                    lines.append(
                        split + " 的 " + presentation + " " + feature
                        + " 均值是 " + _fmt(summary["mean"])
                        + "（n 为 " + _fmt(summary["n"]) + "），在该池内不是常数。"
                    )
        lines.append("")
    lines.append("### 图族内的常数")
    lines.append("")
    for family, block in payload["burden"]["families"].items():
        lines.append(
            family + " 属于 " + block["pool"] + "，n 为 " + _fmt(block["n"])
            + "，n_core 取值为 " + "、".join(_fmt(value) for value in block["n_core_values"]) + "。"
        )
        for presentation in PRESENTATIONS:
            for feature in FEATURES:
                summary = block["presentations"][presentation][feature]
                if summary["constant"]:
                    lines.append(
                        family + " 的 " + presentation + " " + feature
                        + " 在该图族内是常数 " + _fmt(summary["value"]) + "。"
                    )
                else:
                    lines.append(
                        family + " 的 " + presentation + " " + feature
                        + " 均值是 " + _fmt(summary["mean"]) + "，在该图族内不是常数。"
                    )
        lines.append("")
    lines.append("### aligned 做对时，reversed 负担在改口与未改口两组的均值")
    lines.append("")
    for family, block in payload["burden"]["families"].items():
        for feature in FEATURES:
            summary = block["presentations"]["reversed"][feature]
            if summary["constant"]:
                lines.append(
                    family + " 的 reversed " + feature + " 在该图族内是常数 "
                    + _fmt(summary["value"]) + "，不能区分改口。"
                )
    lines.append("")
    for round_name, by_condition in payload["burden"]["by_condition"].items():
        for condition, families in by_condition.items():
            for family, block in families.items():
                varying = []
                for feature in FEATURES:
                    feature_block = block["reversed_burden_mean"][feature]
                    family_summary = payload["burden"]["families"][family]["presentations"]["reversed"][feature]
                    if family_summary["constant"]:
                        continue
                    varying.append(feature)
                    flip_mean = feature_block["flip_mean"]
                    stay_mean = feature_block["stay_mean"]
                    lines.append(
                        round_name + " " + condition + " " + family + " 的 reversed " + feature
                        + "：改口组 n 为 " + _fmt(block["n_flip"])
                        + "，均值 " + ("null" if flip_mean is None else _fmt(flip_mean))
                        + "；未改口组 n 为 " + _fmt(block["n_stay"])
                        + "，均值 " + ("null" if stay_mean is None else _fmt(stay_mean))
                        + "。aligned 做对的题有 " + _fmt(block["n_aligned_correct"]) + "。"
                    )
                if not varying:
                    lines.append(
                        round_name + " " + condition + " " + family
                        + " 的三个 reversed 负担都是该图族内的常数，两组均值没有额外信息。"
                        " aligned 做对 " + _fmt(block["n_aligned_correct"])
                        + "，改口 " + _fmt(block["n_flip"])
                        + "，未改口 " + _fmt(block["n_stay"]) + "。"
                    )
    lines.extend([
        "",
        "## 区间关系",
        "",
        "这里只比较 unseen reversed 准确率的区间是否重叠。这是单次试跑、按潜在任务重采样的区间。",
        "",
    ])
    for round_name, block in payload["c2_interval_relation"].items():
        lines.append(
            round_name + " 的 Ours 区间为 " + _fmt_interval(block["ours"])
            + "，B2 区间为 " + _fmt_interval(block["b2"])
            + "，二者" + _relation_zh(block["ours_and_b2"]) + "。"
        )
        lines.append(
            round_name + " 的 Ours 区间为 " + _fmt_interval(block["ours"])
            + "，B3 区间为 " + _fmt_interval(block["b3"])
            + "，二者" + _relation_zh(block["ours_and_b3"]) + "。"
        )
    lines.extend([
        "",
        "产出文件是 `experiments/e1/rescore/metrics.json` 与 `experiments/e1/rescore/report.md`。",
        "",
    ])
    return "\n".join(lines)


def _relation_zh(name: str) -> str:
    if name == "overlap":
        return "重叠"
    if name == "disjoint":
        return "不相交"
    raise ValueError(name)


CONTRAST_FEATURES = ("d_rel", "f_active")


def _same_number(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= ACC_TOL


def _paired_estimates(paired: dict, latents: list[dict]) -> dict:
    vectors = {}
    hashes = {}
    for split in ("seen", "unseen"):
        split_hashes, aligned, reversed_ = _vectors(paired, latents, split)
        vectors[split] = {"aligned": aligned, "reversed": reversed_}
        hashes[split] = split_hashes
    return {"vectors": vectors, "_hashes": hashes}


def _aligned_feature_split(latents: list[dict], estimates: dict) -> dict:
    """Aligned-correct latent tasks in each family, split by reversed wrong vs right."""
    by_family = defaultdict(list)
    for row in latents:
        by_family[row["family"]].append(row)
    out = {}
    for family in ("chain", "conj_fact", "tree", "diamond"):
        group = by_family.get(family, [])
        if not group:
            _stop(f"图族为空: {family}")
        group = sorted(group, key=lambda row: row["graph_hash"])
        splits = {row["split"] for row in group}
        if len(splits) != 1:
            _stop(f"图族跨 seen/unseen: {family}")
        split = group[0]["split"]
        hashes = [row["graph_hash"] for row in group]
        index = {digest: i for i, digest in enumerate(estimates["_hashes"][split])}
        positions = [index[digest] for digest in hashes]
        aligned = estimates["vectors"][split]["aligned"][positions]
        reversed_ = estimates["vectors"][split]["reversed"][positions]
        chosen = aligned == 1
        flipped = chosen & (reversed_ == 0)
        stayed = chosen & (reversed_ == 1)
        if int(flipped.sum() + stayed.sum()) != int(chosen.sum()):
            _stop(f"{family} 的 aligned 做对没有被改口与未改口分完")
        features = {}
        for feature in FEATURES:
            values = np.asarray(
                [row["features"]["reversed"][feature] for row in group],
                dtype=float,
            )
            features[feature] = {"flip": values[flipped], "stay": values[stayed]}
        out[family] = {
            "pool": split,
            "n_aligned_correct": int(chosen.sum()),
            "n_flip": int(flipped.sum()),
            "n_stay": int(stayed.sum()),
            "features": features,
        }
    return out


def _feature_contrast(flip_values: np.ndarray, stay_values: np.ndarray) -> dict:
    """mean(changed) - mean(unchanged). Empty group => not computable, no invented interval."""
    flip_mean = _mean_or_none(flip_values)
    stay_mean = _mean_or_none(stay_values)
    n_flip = int(np.asarray(flip_values).size)
    n_stay = int(np.asarray(stay_values).size)
    if n_flip == 0 or n_stay == 0:
        reasons = []
        if n_flip == 0:
            reasons.append("改口组人数为 0")
        if n_stay == 0:
            reasons.append("未改口组人数为 0")
        return {
            "computable": False,
            "reason": "；".join(reasons) + "，算不了",
            "reading": "算不了",
            "flip_mean": flip_mean,
            "stay_mean": stay_mean,
            "diff": None,
            "contains_0": None,
        }
    got = group_mean_diff_ci(
        flip_values, stay_values, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA,
    )
    if got is None:
        _stop("两组都非空，但差值区间为空")
    point, low, high = got
    expected = float(flip_mean - stay_mean)
    if abs(point - expected) > ACC_TOL:
        _stop("点估计不是两组均值之差")
    contains = bool(low <= 0.0 <= high)
    return {
        "computable": True,
        "reason": None,
        "reading": "不能区分改口" if contains else "区间不含 0",
        "flip_mean": flip_mean,
        "stay_mean": stay_mean,
        "diff": [point, low, high],
        "contains_0": contains,
    }


def _compare_reversed_means(rounds: dict, existing: dict) -> dict:
    stored = existing["burden"]["by_condition"]
    mismatches = []
    n_compared = 0
    for round_name, conditions in rounds.items():
        for condition, families in conditions.items():
            for family, block in families.items():
                src = stored[round_name][condition][family]
                for field in ("pool", "n_aligned_correct", "n_flip", "n_stay"):
                    n_compared += 1
                    if block[field] != src[field]:
                        mismatches.append({
                            "round": round_name,
                            "condition": condition,
                            "family": family,
                            "field": field,
                            "recomputed": block[field],
                            "existing": src[field],
                        })
                for feature in FEATURES:
                    feature_src = src["reversed_burden_mean"][feature]
                    for field, key in (("flip_mean", "flip"), ("stay_mean", "stay")):
                        n_compared += 1
                        recomputed = _mean_or_none(block["features"][feature][key])
                        previous = feature_src[field]
                        if not _same_number(recomputed, previous):
                            mismatches.append({
                                "round": round_name,
                                "condition": condition,
                                "family": family,
                                "feature": feature,
                                "field": field,
                                "recomputed": recomputed,
                                "existing": previous,
                            })
    return {
        "compared_to": "experiments/e1/rescore/metrics.json burden.by_condition reversed_burden_mean",
        "tolerance": ACC_TOL,
        "n_compared": n_compared,
        "n_mismatch": len(mismatches),
        "mismatches": mismatches,
    }


def _public_family(block: dict) -> dict:
    public = {
        "pool": block["pool"],
        "n_aligned_correct": block["n_aligned_correct"],
        "n_flip": block["n_flip"],
        "n_stay": block["n_stay"],
    }
    for feature in CONTRAST_FEATURES:
        public[feature] = _feature_contrast(
            block["features"][feature]["flip"],
            block["features"][feature]["stay"],
        )
    return public


def write_burden_contrast() -> None:
    """Rebuild pairs and write only experiments/e1/rescore/burden_contrast.json."""
    if "torch" in sys.modules:
        _stop("torch is already imported; refusing to continue")
    protected = _protected_paths()
    for path in protected:
        if not path.exists():
            _stop(f"缺少文件: {path}")
    watched = protected + [OUT / "metrics.json", OUT / "report.md"]
    for path in watched:
        if not path.exists():
            _stop(f"缺少文件: {path}")
    before = {str(path): _mtime_ns(path) for path in watched}
    existing = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    _pools, _summary, meta, latents = _build_metadata()
    if "torch" in sys.modules:
        _stop("重建元数据时加载了 torch")
    raw_rounds = {}
    public_rounds = {}
    for spec in ROUNDS:
        raw_conditions = {}
        public_conditions = {}
        for condition in spec["conditions"]:
            logic_path = spec["logic_root"] / condition / "logic.jsonl"
            loaded = _read_generation_file(logic_path)
            gens = loaded["gens"]
            extra = [uid for uid in gens if uid not in meta]
            if extra:
                _stop(f"{spec['name']} {condition} 有 {len(extra)} 条生成 uid 不在重建元数据中")
            scored = _score_condition(meta, gens)
            estimates = _paired_estimates(scored["paired"], latents)
            groups = _aligned_feature_split(latents, estimates)
            raw_conditions[condition] = groups
            public_conditions[condition] = {
                family: _public_family(block) for family, block in groups.items()
            }
            print(f"scored {spec['name']} {condition}", flush=True)
        raw_rounds[spec["name"]] = raw_conditions
        public_rounds[spec["name"]] = public_conditions
    mean_check = _compare_reversed_means(raw_rounds, existing)
    payload = {
        "config": {
            "seed": BOOT_SEED,
            "n_boot": N_BOOT,
            "alpha": ALPHA,
            "interval": "[point, low, high]",
            "contrast": "mean(changed) - mean(unchanged)",
            "changed": "aligned correct and reversed wrong",
            "unchanged": "aligned correct and reversed correct",
            "subset": "aligned-correct latent tasks within one graph family",
            "features": list(CONTRAST_FEATURES),
            "families": ["chain", "conj_fact", "tree", "diamond"],
            "presentation": "reversed",
            "bootstrap": (
                "Each group is resampled independently. "
                "The unit is one aligned-correct latent task. "
                "The changed group is drawn first from Generator(seed)."
            ),
            "back_reads": "not contrasted; constant within each family",
            "loaded_model": False,
            "training": False,
        },
        "mean_check": mean_check,
        "rounds": public_rounds,
    }
    payload = _py(payload)
    if "torch" in sys.modules:
        _stop("计分过程中加载了 torch，没有写入结果")
    mid = {str(path): _mtime_ns(path) for path in watched}
    if mid != before:
        _stop("原文件或已有 rescore 的时间戳变了，没有写入 burden_contrast")
    out_path = OUT / "burden_contrast.json"
    if out_path.name != "burden_contrast.json" or out_path.parent.resolve() != OUT.resolve():
        _stop("拒绝写入 burden_contrast.json 以外的路径")
    for banned_root in (
        PROJECT / "experiments" / "e1" / "pilot",
        PROJECT / "experiments" / "e1" / "pilot_mix15",
        PROJECT / "build" / "e1",
    ):
        if banned_root.resolve() in out_path.resolve().parents:
            _stop("拒绝写入原实验目录")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(text, encoding="utf-8")
    end = {str(path): _mtime_ns(path) for path in watched}
    if end != before:
        _stop("写入 burden_contrast 之后原文件或已有 rescore 的时间戳变了")
    print(f"wrote {out_path}", flush=True)
    print(f"mean_check.n_mismatch {mean_check['n_mismatch']}", flush=True)


def main() -> None:
    protected = _protected_paths()
    for path in protected:
        if not path.exists():
            _stop(f"缺少文件: {path}")
    before = {str(path): _mtime_ns(path) for path in protected}
    _pools, summary, meta, latents = _build_metadata()
    if "torch" in sys.modules:
        _stop("重建元数据时加载了 torch")
    burden_info = _burden_tables(latents)
    cell_rows = []
    rounds = {}
    by_condition = {}
    for spec in ROUNDS:
        conditions = {}
        flip_round = {}
        for condition in spec["conditions"]:
            logic_path = spec["logic_root"] / condition / "logic.jsonl"
            metrics_path = spec["metrics_root"] / condition / "metrics.json"
            original = json.loads(metrics_path.read_text(encoding="utf-8"))
            quant = original["quant"]
            if quant != _expected_quant(condition):
                _stop(f"{spec['name']} {condition} quant 为 {quant}，与预期不符")
            loaded = _read_generation_file(logic_path)
            gens = loaded["gens"]
            extra = [uid for uid in gens if uid not in meta]
            if extra:
                _stop(f"{spec['name']} {condition} 有 {len(extra)} 条生成 uid 不在重建元数据中")
            scored = _score_condition(meta, gens)
            checked = _check_cells(scored, original)
            for row in checked:
                row["round"] = spec["name"]
                row["condition"] = condition
            cell_rows.extend(checked)
            if any(not row["match"] for row in checked):
                bad = [row for row in checked if not row["match"]]
                print(json.dumps(bad, ensure_ascii=False, indent=2), flush=True)
                _stop("四格与原 metrics 不一致，已停止，没有写入 rescore 结果")
            estimates = _condition_estimates(scored["paired"], latents)
            _attach_hashes(estimates, latents)
            for split in ("seen", "unseen"):
                for presentation, cell in (
                    ("aligned", f"{split}_aligned"),
                    ("reversed", f"{split}_shifted"),
                ):
                    point = estimates["acc"][split][presentation][0]
                    cell_acc = scored["cells"][cell]["acc"]
                    if abs(point - cell_acc) > ACC_TOL:
                        _stop(f"{condition} {cell} 的配对准确率与四格准确率不一致")
            flip_round[condition] = _flip_tables(latents, estimates, burden_info)
            conditions[condition] = {
                "quant": quant,
                "logic": str(logic_path.relative_to(PROJECT)).replace("\\", "/"),
                "metrics": str(metrics_path.relative_to(PROJECT)).replace("\\", "/"),
                "n_logic_lines": loaded["n_lines"],
                "n_logic_uid": loaded["n_uid"],
                "n_duplicate_uid": loaded["n_duplicate_uid"],
                "n_identical_duplicate_uid": loaded["n_identical_duplicate_uid"],
                "n_text_conflict_uid": loaded["n_text_conflict_uid"],
                "n_answer_conflict_uid": loaded["n_answer_conflict_uid"],
                "acc": estimates["acc"],
                "SF": estimates["SF"],
                "delta_T": estimates["delta_T"],
                "interaction": estimates["interaction"],
                "counts": estimates["counts"],
            }
        rounds[spec["name"]] = {"has_b0": spec["has_b0"], "conditions": conditions}
        by_condition[spec["name"]] = flip_round
    mismatches = [row for row in cell_rows if not row["match"]]
    if mismatches:
        _stop("四格与原 metrics 不一致，已停止，没有写入 rescore 结果")
    count_keys = ("n", "n_yes", "n_no")
    uniform_values = {key: {row[key] for row in cell_rows} for key in count_keys}
    uniform = {"same": all(len(values) == 1 for values in uniform_values.values())}
    if uniform["same"]:
        uniform.update({key: next(iter(values)) for key, values in uniform_values.items()})
    diffs = [row["abs_diff"] for row in cell_rows]
    c2 = {}
    for round_name, round_payload in rounds.items():
        ours = round_payload["conditions"]["ours"]["acc"]["unseen"]["reversed"]
        b2 = round_payload["conditions"]["b2"]["acc"]["unseen"]["reversed"]
        b3 = round_payload["conditions"]["b3"]["acc"]["unseen"]["reversed"]
        c2[round_name] = {
            "cell": "unseen_reversed",
            "ours": ours,
            "b2": b2,
            "b3": b3,
            "ours_and_b2": _relation(ours, b2),
            "ours_and_b3": _relation(ours, b3),
        }
    isolation = {
        "ok": bool(summary["ok"]),
        "overlaps": {key: int(value) for key, value in summary["overlaps"].items()},
        "shifted_presentation": summary["shifted_presentation"],
        "test_seen_n": int(summary["pools"]["test_seen"]["n"]),
        "test_unseen_n": int(summary["pools"]["test_unseen"]["n"]),
        "test_seen_n_yes": int(summary["pools"]["test_seen"]["n_yes"]),
        "test_seen_n_no": int(summary["pools"]["test_seen"]["n_no"]),
        "test_unseen_n_yes": int(summary["pools"]["test_unseen"]["n_yes"]),
        "test_unseen_n_no": int(summary["pools"]["test_unseen"]["n_no"]),
    }
    if isolation["shifted_presentation"] != "reversed":
        _stop("shifted presentation 不是 reversed")
    if any(value != 0 for value in isolation["overlaps"].values()):
        _stop("池隔离失败")
    payload = {
        "config": {
            "seed": BOOT_SEED,
            "n_boot": N_BOOT,
            "alpha": ALPHA,
            "tolerance_acc": ACC_TOL,
            "pair_key": "graph_hash",
            "latent_order": "sorted graph_hash",
            "interval": "[point, low, high]",
            "training": False,
            "loaded_model": False,
        },
        "quant": {"b0": "bf16-eval", "trained": "nf4", "pilot_mix15_has_b0": False},
        "definitions": {
            "SF": "P(reversed wrong | aligned right)",
            "delta_T": "acc(seen, p) - acc(unseen, p); pools resampled independently",
            "interaction": "delta_T(reversed) - delta_T(aligned) on the same bootstrap draw",
        },
        "isolation": isolation,
        "cell_check": {
            "all_match": True,
            "n_cells": len(cell_rows),
            "max_abs_acc_diff": float(max(diffs)),
            "tolerance": ACC_TOL,
            "uniform_counts": uniform,
            "rows": cell_rows,
        },
        "rounds": rounds,
        "burden": {
            "features": list(FEATURES),
            "pools": burden_info["pools"],
            "families": burden_info["families"],
            "by_condition": by_condition,
        },
        "c2_interval_relation": c2,
    }
    payload = _py(payload)
    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    report = _render_report(payload)
    _assert_report_numbers(report, json_text)
    if "torch" in sys.modules:
        _stop("计分过程中加载了 torch，没有写入结果")
    after = {str(path): _mtime_ns(path) for path in protected}
    if after != before:
        _stop("原 metrics 或 logic.jsonl 的时间戳变了，没有写入 rescore")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / "metrics.json"
    report_path = OUT / "report.md"
    allowed = {metrics_path.resolve(), report_path.resolve()}
    if metrics_path.resolve() not in allowed or report_path.resolve() not in allowed:
        _stop("输出路径异常")
    for banned_root in (
        PROJECT / "experiments" / "e1" / "pilot",
        PROJECT / "experiments" / "e1" / "pilot_mix15",
        PROJECT / "build" / "e1",
    ):
        if banned_root.resolve() in metrics_path.resolve().parents or banned_root.resolve() == metrics_path.resolve().parent:
            _stop("拒绝写入原实验目录")
    metrics_path.write_text(json_text, encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    end = {str(path): _mtime_ns(path) for path in protected}
    if end != before:
        _stop("写入 rescore 之后原文件时间戳变了")
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    print("cell_check.all_match true", flush=True)
    print("max_abs_acc_diff " + _fmt(payload["cell_check"]["max_abs_acc_diff"]), flush=True)


if __name__ == "__main__":
    main()
