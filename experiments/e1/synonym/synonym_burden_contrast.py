"""Aligned-order burden contrast for synonym flips.

Among items answered correctly in the canonical aligned presentation, compare
d_rel and f_active between synonym flips and stays. Burden uses aligned core
positions. No training and no model load.

Writes only experiments/e1/synonym/synonym_burden_contrast.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from structrobust.burden import burden
from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl
from structrobust.logic import render
from structrobust.prompts import extract_yes_no
from structrobust.stats import group_mean_diff_ci
from structrobust.wording import render_condition

N_BOOT = 2000
SEED = 0
ALPHA = 0.05
TOL = 1e-12
FEATURES = ("d_rel", "f_active")
FAMILIES = ("chain", "conj_fact", "tree", "diamond")
POOL_OF = {
    "chain": "seen",
    "conj_fact": "seen",
    "tree": "unseen",
    "diamond": "unseen",
}
CONDITIONS = (
    ("b0", "build/e1/synonym/b0/generations.jsonl", "build/e1/pilot/b0/logic.jsonl"),
    ("b1", "build/e1/synonym/b1/generations.jsonl", "build/e1/pilot/b1/logic.jsonl"),
    ("b2", "build/e1/synonym/b2/generations.jsonl", "build/e1/pilot/b2/logic.jsonl"),
    ("b3", "build/e1/synonym/b3/generations.jsonl", "build/e1/pilot/b3/logic.jsonl"),
    ("ours", "build/e1/synonym/ours/generations.jsonl", "build/e1/pilot/ours/logic.jsonl"),
    ("mix15_b1", "build/e1/synonym/mix15_b1/generations.jsonl", "build/e1/pilot_mix15/b1/logic.jsonl"),
    ("mix15_b2", "build/e1/synonym/mix15_b2/generations.jsonl", "build/e1/pilot_mix15/b2/logic.jsonl"),
    ("mix15_b3", "build/e1/synonym/mix15_b3/generations.jsonl", "build/e1/pilot_mix15/b3/logic.jsonl"),
    ("mix15_ours", "build/e1/synonym/mix15_ours/generations.jsonl", "build/e1/pilot_mix15/ours/logic.jsonl"),
)
OUT = PROJECT / "experiments" / "e1" / "synonym" / "synonym_burden_contrast.json"
# Published mix15_ours seen cell: 11 both-flipped + 40 only-synonym.
SANITY_SEEN_ALIGNED = 179
SANITY_SEEN_FLIP = 51
SANITY_BOTH = 11
SANITY_ONLY_SYNONYM = 40


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _toy_point_check() -> None:
    changed = [1.0, 2.0, 9.0]
    unchanged = [1.0, 3.0]
    result = group_mean_diff_ci(changed, unchanged, n_boot=N_BOOT, seed=SEED, alpha=ALPHA)
    if result is None:
        _stop("toy contrast returned None")
    point = result[0]
    expected = float(np.mean(changed) - np.mean(unchanged))
    if point != expected:
        _stop(f"toy point {point} != difference of means {expected}")


def _constant(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return bool(float(values.max()) - float(values.min()) <= TOL)


def _mean_or_none(values: np.ndarray):
    if values.size == 0:
        return None
    return float(values.mean())


def _feature_span(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "min": None, "max": None, "zero_variance": False, "value": None}
    constant = _constant(values)
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "zero_variance": constant,
        "value": float(values.mean()) if constant else None,
    }


def _load_items() -> list[dict]:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    items = []
    for split, pool_name in (("seen", "test_seen"), ("unseen", "test_unseen")):
        for lat in pools[pool_name]:
            if lat.family not in POOL_OF or POOL_OF[lat.family] != split:
                _stop(f"{lat.family} is not in the {split} pool")
            digest = graph_hash(lat)
            aligned = render(lat, "aligned")
            synonym = render_condition(lat, "synonym")
            if list(synonym["core_positions"]) != list(aligned["core_positions"]):
                _stop(f"{digest} synonym core positions differ from aligned")
            if list(synonym["rule_seq"]) != list(aligned["rule_seq"]):
                _stop(f"{digest} synonym rule order differs from aligned")
            raw = burden(lat, list(aligned["core_positions"]))
            items.append({
                "split": split,
                "family": lat.family,
                "graph_hash": digest,
                "label": lat.label,
                "features": {name: float(raw[name]) for name in ("back_reads",) + FEATURES},
            })
    return items


def _family_burden(items: list[dict]) -> dict:
    out = {}
    for family in FAMILIES:
        group = [row for row in items if row["family"] == family]
        if not group:
            _stop(f"empty family {family}")
        splits = {row["split"] for row in group}
        if splits != {POOL_OF[family]}:
            _stop(f"{family} is not confined to {POOL_OF[family]}")
        block = {"pool": POOL_OF[family], "n": len(group)}
        for feature in ("back_reads",) + FEATURES:
            values = np.asarray([row["features"][feature] for row in group], dtype=float)
            block[feature] = _feature_span(values)
        out[family] = block
    return out


def _score(items: list[dict], synonym_gens: dict, logic_gens: dict) -> list[dict]:
    rows = []
    missing = []
    for item in items:
        digest = item["graph_hash"]
        split = item["split"]
        aligned_uid = f"{split}_aligned|{digest}"
        synonym_uid = f"{split}_synonym|{digest}"
        shifted_uid = f"{split}_shifted|{digest}"
        if aligned_uid not in logic_gens or synonym_uid not in synonym_gens or shifted_uid not in logic_gens:
            missing.append(aligned_uid)
            continue
        aligned_pred = extract_yes_no(logic_gens[aligned_uid]["generation"])
        synonym_pred = extract_yes_no(synonym_gens[synonym_uid]["generation"])
        reversed_pred = extract_yes_no(logic_gens[shifted_uid]["generation"])
        rows.append({
            **item,
            "aligned_correct": int(aligned_pred == item["label"]),
            "synonym_correct": int(synonym_pred == item["label"]),
            "reversed_correct": int(reversed_pred == item["label"]),
        })
    if missing:
        _stop(f"missing {len(missing)} paired rows, e.g. {missing[:3]}")
    if len(rows) != len(items):
        _stop("scored row count does not match the pool")
    return rows


def _contrast(flip_values: np.ndarray, stay_values: np.ndarray, zero_variance: bool) -> dict:
    flip_mean = _mean_or_none(flip_values)
    stay_mean = _mean_or_none(stay_values)
    n_flip = int(flip_values.size)
    n_stay = int(stay_values.size)
    base = {
        "zero_variance": bool(zero_variance),
        "flip_mean": flip_mean,
        "stay_mean": stay_mean,
    }
    if n_flip == 0 or n_stay == 0:
        reasons = []
        if n_flip == 0:
            reasons.append("flip group is empty")
        if n_stay == 0:
            reasons.append("stay group is empty")
        base.update({
            "computable": False,
            "reason": "; ".join(reasons),
            "diff": None,
            "contains_0": None,
        })
        return base
    if zero_variance:
        base.update({
            "computable": False,
            "reason": "zero variance; no difference to test",
            "diff": None,
            "contains_0": None,
        })
        return base
    got = group_mean_diff_ci(flip_values, stay_values, n_boot=N_BOOT, seed=SEED, alpha=ALPHA)
    if got is None:
        _stop("both groups are nonempty but the interval is missing")
    point, low, high = got
    expected = float(flip_mean - stay_mean)
    if abs(point - expected) > TOL:
        _stop(f"point {point} is not the difference of means {expected}")
    base.update({
        "computable": True,
        "reason": None,
        "diff": [point, low, high],
        "contains_0": bool(low <= 0.0 <= high),
    })
    return base


def _family_block(rows: list[dict], family: str, family_constant: dict) -> dict:
    group = sorted((row for row in rows if row["family"] == family), key=lambda row: row["graph_hash"])
    aligned = np.asarray([row["aligned_correct"] for row in group], dtype=float)
    synonym = np.asarray([row["synonym_correct"] for row in group], dtype=float)
    chosen = aligned == 1
    flipped = chosen & (synonym == 0)
    stayed = chosen & (synonym == 1)
    if int(flipped.sum() + stayed.sum()) != int(chosen.sum()):
        _stop(f"{family} aligned-correct items are not split into flip and stay")
    block = {
        "pool": POOL_OF[family],
        "n_aligned_correct": int(chosen.sum()),
        "n_flip": int(flipped.sum()),
        "n_stay": int(stayed.sum()),
    }
    for feature in FEATURES:
        values = np.asarray([row["features"][feature] for row in group], dtype=float)
        eligible = values[chosen]
        subset_constant = bool(eligible.size) and _constant(eligible)
        zero_variance = bool(family_constant[feature] or subset_constant)
        block[feature] = _contrast(values[flipped], values[stayed], zero_variance)
        block[feature]["zero_variance_in_family"] = bool(family_constant[feature])
        block[feature]["zero_variance_among_aligned_correct"] = bool(subset_constant)
    return block


def _seen_partition(rows: list[dict]) -> dict:
    both = only_synonym = aligned_correct = synonym_flip = 0
    for row in rows:
        if row["split"] != "seen" or row["aligned_correct"] != 1:
            continue
        aligned_correct += 1
        synonym_wrong = row["synonym_correct"] == 0
        reversed_wrong = row["reversed_correct"] == 0
        if synonym_wrong:
            synonym_flip += 1
        if synonym_wrong and reversed_wrong:
            both += 1
        elif synonym_wrong and not reversed_wrong:
            only_synonym += 1
    return {
        "n_aligned_correct": aligned_correct,
        "n_flip": synonym_flip,
        "both_flipped": both,
        "only_synonym_flipped": only_synonym,
    }


def _guard_out(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OUT.resolve():
        _stop(f"refusing to write {resolved}")
    if "rescore" in resolved.parts:
        _stop("refusing to write into rescore")
    banned_names = {"generations.jsonl", "logic.jsonl"}
    if resolved.name in banned_names:
        _stop(f"refusing to write {resolved.name}")
    return resolved


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    _toy_point_check()
    items = _load_items()
    family_burden = _family_burden(items)
    family_constant = {
        family: {feature: bool(family_burden[family][feature]["zero_variance"]) for feature in FEATURES}
        for family in FAMILIES
    }
    back_reads_zero = all(
        family_burden[family]["back_reads"]["zero_variance"]
        and family_burden[family]["back_reads"]["value"] == 0.0
        for family in FAMILIES
    )
    conditions = {}
    mix_rows = None
    for name, gen_rel, logic_rel in CONDITIONS:
        gen_path = PROJECT / gen_rel
        logic_path = PROJECT / logic_rel
        if not gen_path.is_file() or not logic_path.is_file():
            _stop(f"missing input for {name}")
        rows = _score(items, read_jsonl(gen_path), read_jsonl(logic_path))
        conditions[name] = {
            family: _family_block(rows, family, family_constant[family]) for family in FAMILIES
        }
        if name == "mix15_ours":
            mix_rows = rows
    if mix_rows is None:
        _stop("mix15_ours was not scored")
    sanity = _seen_partition(mix_rows)
    if sanity["n_aligned_correct"] != SANITY_SEEN_ALIGNED or sanity["n_flip"] != SANITY_SEEN_FLIP:
        _stop(
            "mix15_ours seen sanity failed: "
            f"n_aligned_correct={sanity['n_aligned_correct']} n_flip={sanity['n_flip']}"
        )
    if sanity["both_flipped"] != SANITY_BOTH or sanity["only_synonym_flipped"] != SANITY_ONLY_SYNONYM:
        _stop(
            "mix15_ours seen partition failed: "
            f"both={sanity['both_flipped']} only_synonym={sanity['only_synonym_flipped']}"
        )
    n_computable = 0
    n_exclude = 0
    signs = []
    for name, families in conditions.items():
        for family, block in families.items():
            for feature in FEATURES:
                cell = block[feature]
                if not cell["computable"]:
                    continue
                n_computable += 1
                if cell["contains_0"]:
                    continue
                n_exclude += 1
                point = cell["diff"][0]
                if point > 0:
                    signs.append(1)
                elif point < 0:
                    signs.append(-1)
                else:
                    signs.append(0)
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    payload = {
        "config": {
            "seed": SEED,
            "n_boot": N_BOOT,
            "alpha": ALPHA,
            "interval": "[point, low, high]",
            "contrast": "mean(flip) - mean(stay)",
            "changed": "aligned correct and synonym wrong",
            "unchanged": "aligned correct and synonym correct",
            "subset": "aligned-correct latent tasks within one graph family",
            "features": list(FEATURES),
            "families": list(FAMILIES),
            "presentation": "aligned",
            "wording": "synonym",
            "burden_positions": "aligned core_positions; synonym keeps the aligned rule order and rewords proof step 0",
            "bootstrap": (
                "Each group is resampled independently. "
                "The unit is one aligned-correct latent task, sorted by graph hash. "
                "The changed group is drawn first from Generator(seed)."
            ),
            "back_reads": (
                "not contrasted; 0 for every test item on the aligned order"
                if back_reads_zero
                else "not contrasted; aligned back_reads was not identically 0"
            ),
            "zero_variance": (
                "A feature with no variation in the family, or among that condition's "
                "aligned-correct items, is recorded and is not given a difference interval."
            ),
            "interpretation": (
                "Descriptive contrast only. A nonzero difference is not evidence that "
                "higher burden causes synonym flips."
            ),
            "loaded_model": False,
            "training": False,
            "sanity_mix15_ours_seen": sanity,
        },
        "aligned_family_burden": family_burden,
        "conditions": conditions,
        "summary": {
            "n_computable": n_computable,
            "n_exclude_0": n_exclude,
            "exclude_0_signs": signs,
            "exclude_0_signs_agree": bool(signs) and len(set(signs)) == 1,
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(f"n_computable {n_computable}")
    print(f"n_exclude_0 {n_exclude}")
    print(f"signs {signs}")
    print(f"signs_agree {payload['summary']['exclude_0_signs_agree']}")
    for family in FAMILIES:
        for feature in ("back_reads",) + FEATURES:
            span = family_burden[family][feature]
            if span["zero_variance"]:
                print(f"zero_variance {family} {feature} value={span['value']}")


if __name__ == "__main__":
    main()
