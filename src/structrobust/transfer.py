"""One untrained comparison of seen and unseen dependency graphs at depth 2.

Seen families are chain and conj_fact. Unseen families are tree and diamond.
Each latent task is shown in canonical order, reverse order, and a synonym
rewrite of proof step 0. The synonym comes from wording.synonym_sentence and
is not added to logic.PRESENTATIONS.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import logic, wording
from .prompts import extract_yes_no, logic_prompt
from .stats import conditional_flip_ci, mean_ci

SEEN_FAMILIES = logic.SEEN_FAMILIES
UNSEEN_FAMILIES = logic.UNSEEN_FAMILIES
DEPTH = 2
N_PER_FAMILY = 100
SEED = 2
TAG = "transfer"
VOCAB = "eval"
CONDITIONS = ("aligned", "reversed", "synonym")
SF_TARGETS = ("reversed", "synonym")
N_BOOT = 2000
BOOT_SEED = 2
_BANNED_TAILS = (
    ("build", "e0", "round1"),
    ("build", "e0", "wording"),
    ("experiments", "e1"),
)


def render_condition(lat: logic.Latent, condition: str) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return wording.render_condition(lat, condition)


def build_pool() -> list[logic.Latent]:
    seen = logic.make_pool(SEEN_FAMILIES, [DEPTH], N_PER_FAMILY, SEED, VOCAB, TAG)
    unseen = logic.make_pool(UNSEEN_FAMILIES, [DEPTH], N_PER_FAMILY, SEED, VOCAB, TAG)
    return seen + unseen


def _pool_name(family: str) -> str:
    if family in SEEN_FAMILIES:
        return "seen"
    if family in UNSEEN_FAMILIES:
        return "unseen"
    raise ValueError(family)


def build_items(pool: list[logic.Latent] | None = None):
    if pool is None:
        pool = build_pool()
    items, meta = [], {}
    for lat in pool:
        pool_name = _pool_name(lat.family)
        for condition in CONDITIONS:
            rendered = render_condition(lat, condition)
            uid = f"{lat.task_id}|{condition}"
            items.append({"uid": uid, "prompt": logic_prompt(rendered["text"])})
            meta[uid] = {
                "task_id": lat.task_id,
                "depth": lat.depth,
                "family": lat.family,
                "pool": pool_name,
                "label": lat.label,
                "condition": condition,
            }
    return items, meta


def _triple(values) -> list | None:
    if values[0] is None:
        return None
    return [float(x) for x in values]


def _ci(point: float, boots: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def gap_and_interaction(
    seen: dict[str, np.ndarray],
    unseen: dict[str, np.ndarray],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
):
    """Δ_T(p) = acc_seen(p) - acc_unseen(p), with independent pool resamples.

    Int(p) = Δ_T(p) - Δ_T(aligned) is computed on that same pair of resamples.
    """
    seen_v = {condition: np.asarray(seen[condition], dtype=float) for condition in CONDITIONS}
    unseen_v = {condition: np.asarray(unseen[condition], dtype=float) for condition in CONDITIONS}
    n_seen = int(seen_v["aligned"].size)
    n_unseen = int(unseen_v["aligned"].size)
    if any(seen_v[condition].size != n_seen for condition in CONDITIONS):
        raise ValueError("seen presentations must share one latent-task order")
    if any(unseen_v[condition].size != n_unseen for condition in CONDITIONS):
        raise ValueError("unseen presentations must share one latent-task order")
    if n_seen == 0 or n_unseen == 0:
        return {condition: None for condition in CONDITIONS}, {condition: None for condition in SF_TARGETS}
    point = {
        condition: float(seen_v[condition].mean() - unseen_v[condition].mean())
        for condition in CONDITIONS
    }
    rng = np.random.default_rng(seed)
    idx_seen = rng.integers(0, n_seen, size=(n_boot, n_seen))
    idx_unseen = rng.integers(0, n_unseen, size=(n_boot, n_unseen))
    boots = {
        condition: seen_v[condition][idx_seen].mean(axis=1) - unseen_v[condition][idx_unseen].mean(axis=1)
        for condition in CONDITIONS
    }
    delta = {condition: _triple(_ci(point[condition], boots[condition], alpha)) for condition in CONDITIONS}
    interaction = {}
    for condition in SF_TARGETS:
        gap = boots[condition] - boots["aligned"]
        interaction[condition] = _triple(_ci(point[condition] - point["aligned"], gap, alpha))
    return delta, interaction


def analyze(meta: dict, gens: dict, boot_seed: int = BOOT_SEED, n_boot: int = N_BOOT):
    by = {"seen": defaultdict(dict), "unseen": defaultdict(dict)}
    rows = []
    for uid, row in meta.items():
        generated = gens.get(uid)
        if generated is None:
            continue
        pred = extract_yes_no(generated["generation"])
        correct = int(pred == row["label"])
        by[row["pool"]][row["task_id"]][row["condition"]] = correct
        rows.append({**row, "uid": uid, "pred": pred, "correct": correct})
    task_ids = {}
    vec = {}
    for pool_name in ("seen", "unseen"):
        ids = sorted(
            task_id
            for task_id, conds in by[pool_name].items()
            if set(CONDITIONS) <= set(conds)
        )
        task_ids[pool_name] = ids
        vec[pool_name] = {
            condition: np.array([by[pool_name][task_id][condition] for task_id in ids], dtype=float)
            for condition in CONDITIONS
        }
    acc, sf = {}, {}
    for pool_name in ("seen", "unseen"):
        acc[pool_name] = {
            condition: _triple(mean_ci(vec[pool_name][condition], n_boot=n_boot, seed=boot_seed))
            for condition in CONDITIONS
        }
        sf[pool_name] = {
            target: _triple(
                conditional_flip_ci(
                    vec[pool_name]["aligned"], vec[pool_name][target], n_boot=n_boot, seed=boot_seed
                )
            )
            for target in SF_TARGETS
        }
    delta, interaction = gap_and_interaction(
        vec["seen"], vec["unseen"], n_boot=n_boot, seed=boot_seed
    )
    complete = {pool_name: set(ids) for pool_name, ids in task_ids.items()}
    n_yes, n_no = {}, {}
    extract_fail, pred_yes = {}, {}
    for pool_name in ("seen", "unseen"):
        labels = [
            row["label"]
            for row in rows
            if row["pool"] == pool_name
            and row["condition"] == "aligned"
            and row["task_id"] in complete[pool_name]
        ]
        n_yes[pool_name] = labels.count("yes")
        n_no[pool_name] = labels.count("no")
        extract_fail[pool_name] = {}
        pred_yes[pool_name] = {}
        for condition in CONDITIONS:
            subset = [
                row
                for row in rows
                if row["pool"] == pool_name
                and row["condition"] == condition
                and row["task_id"] in complete[pool_name]
            ]
            extract_fail[pool_name][condition] = (
                float(np.mean([row["pred"] == "" for row in subset])) if subset else None
            )
            pred_yes[pool_name][condition] = (
                float(np.mean([row["pred"] == "yes" for row in subset])) if subset else None
            )
    out = {
        "n_latent": {pool_name: len(task_ids[pool_name]) for pool_name in ("seen", "unseen")},
        "n_yes": n_yes,
        "n_no": n_no,
        "paired_unit": "latent_task",
        "n_boot": n_boot,
        "acc": acc,
        "SF": sf,
        "delta_T": delta,
        "interaction": interaction,
        "SF_definition": "P(target wrong | aligned right), latent-task bootstrap within the pool",
        "delta_T_definition": (
            "acc(seen, p) - acc(unseen, p); latent tasks are resampled independently in each pool"
        ),
        "interaction_definition": "delta_T(p) - delta_T(aligned) on the same bootstrap draw",
        "extract_fail_rate": extract_fail,
        "pred_yes_rate": pred_yes,
    }
    return out, rows


def _guard_out(out: Path) -> Path:
    resolved = out.resolve()
    parts = tuple(part.lower() for part in resolved.parts)
    for tail in _BANNED_TAILS:
        width = len(tail)
        for start in range(len(parts) - width + 1):
            if parts[start:start + width] == tail:
                banned = "/".join(tail)
                raise SystemExit(f"refusing to write into {banned}")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("build/e0/transfer"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    out = _guard_out(args.out)
    items, meta = build_items()
    expected = (len(SEEN_FAMILIES) + len(UNSEEN_FAMILIES)) * N_PER_FAMILY * len(CONDITIONS)
    if len(items) != expected:
        raise SystemExit(f"expected {expected} items, got {len(items)}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[transfer] items={len(items)} out={out}", flush=True)
    gen_path = out / "generations.jsonl"
    if not args.analyze_only:
        from .infer import generate, load_model

        tok, model = load_model(args.model)
        generate(
            tok,
            model,
            items,
            gen_path,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            shuffle_seed=SEED,
        )
    gens = {}
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            gens[record["uid"]] = record
    result, rows = analyze(meta, gens)
    result["config"] = {
        "model": args.model,
        "adapter": None,
        "depth": DEPTH,
        "seen_families": list(SEEN_FAMILIES),
        "unseen_families": list(UNSEEN_FAMILIES),
        "n_per_family": N_PER_FAMILY,
        "vocab": VOCAB,
        "tag": TAG,
        "seed": SEED,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "decoding": "greedy",
        "cot": True,
        "bootstrap_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "paired_unit": "latent_task",
        "conditions": list(CONDITIONS),
        "sf_pairs": ["aligned->reversed", "aligned->synonym"],
        "training": False,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
