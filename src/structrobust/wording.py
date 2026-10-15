"""One untrained comparison of four surface wordings on the same depth-2 tasks.

Canonical order and reverse order use the existing generator. The synonym and
ambiguous wordings rewrite only the surface sentence of proof step 0 and leave
every other rule in the aligned order. They are not added to logic.PRESENTATIONS.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import logic
from .prompts import extract_yes_no, logic_prompt
from .stats import conditional_flip_ci, mean_ci

FAMILIES = ("chain", "conj_fact")
DEPTH = 2
N_PER_FAMILY = 100
SEED = 1
TAG = "wording"
VOCAB = "eval"
CONDITIONS = ("aligned", "reversed", "synonym", "ambiguous")
SF_TARGETS = ("reversed", "synonym")
BOOT_SEED = 2
_BANNED_OUT = (Path("build/e0/round1"), Path("experiments/e1"))


def synonym_sentence(entity: str, rule: logic.Rule) -> str:
    premises, conclusion = rule
    if len(premises) == 1:
        return f"{entity} is always {conclusion} whenever {entity} is {premises[0]}."
    if len(premises) == 2:
        return (
            f"{entity} is always {conclusion} whenever {entity} is {premises[0]} "
            f"and {entity} is {premises[1]}."
        )
    raise ValueError(len(premises))


def ambiguous_sentence(entity: str, rule: logic.Rule) -> str:
    _premises, conclusion = rule
    return f"{entity} likes {conclusion}."


def _splice_step0(lat: logic.Latent, sentence: str, condition: str) -> dict:
    base = logic.render(lat, "aligned")
    head, rest = base["text"].split("\n\nRules:\n", 1)
    rules_block, question = rest.split("\n\nQuestion:", 1)
    lines = rules_block.splitlines()
    seq = base["rule_seq"]
    if len(lines) != len(seq):
        raise RuntimeError("rule lines do not match the aligned sequence")
    replaced = False
    for pos, (kind, index) in enumerate(seq):
        if kind == "core" and index == 0:
            lines[pos] = sentence
            replaced = True
            break
    if not replaced:
        raise RuntimeError("proof step 0 is missing from the aligned render")
    out = dict(base)
    out["text"] = head + "\n\nRules:\n" + "\n".join(lines) + "\n\nQuestion:" + question
    out["presentation"] = condition
    return out


def render_condition(lat: logic.Latent, condition: str) -> dict:
    if condition == "aligned":
        return logic.render(lat, "aligned")
    if condition == "reversed":
        return logic.render(lat, "reversed")
    if condition == "synonym":
        return _splice_step0(lat, synonym_sentence(lat.entity, lat.core_rules[0]), condition)
    if condition == "ambiguous":
        return _splice_step0(lat, ambiguous_sentence(lat.entity, lat.core_rules[0]), condition)
    raise ValueError(condition)


def build_pool() -> list[logic.Latent]:
    return logic.make_pool(FAMILIES, [DEPTH], N_PER_FAMILY, SEED, VOCAB, TAG)


def build_items(pool: list[logic.Latent] | None = None):
    if pool is None:
        pool = build_pool()
    items, meta = [], {}
    for lat in pool:
        for condition in CONDITIONS:
            rendered = render_condition(lat, condition)
            uid = f"{lat.task_id}|{condition}"
            items.append({"uid": uid, "prompt": logic_prompt(rendered["text"])})
            meta[uid] = {
                "task_id": lat.task_id,
                "depth": lat.depth,
                "family": lat.family,
                "label": lat.label,
                "condition": condition,
                "entailment_broken": condition == "ambiguous",
            }
    return items, meta


def _triple(values) -> list | None:
    if values[0] is None:
        return None
    return [float(x) for x in values]


def analyze(meta: dict, gens: dict, boot_seed: int = BOOT_SEED):
    by = defaultdict(dict)
    rows = []
    for uid, row in meta.items():
        generated = gens.get(uid)
        if generated is None:
            continue
        pred = extract_yes_no(generated["generation"])
        correct = int(pred == row["label"])
        by[row["task_id"]][row["condition"]] = {"correct": correct, "pred": pred}
        rows.append({**row, "uid": uid, "pred": pred, "correct": correct})
    task_ids = sorted(
        task_id for task_id, conds in by.items() if set(CONDITIONS) <= set(conds)
    )
    vec = {
        condition: np.array([by[task_id][condition]["correct"] for task_id in task_ids], dtype=float)
        for condition in CONDITIONS
    }
    acc = {condition: _triple(mean_ci(vec[condition], seed=boot_seed)) for condition in CONDITIONS}
    sf = {
        target: _triple(conditional_flip_ci(vec["aligned"], vec[target], seed=boot_seed))
        for target in SF_TARGETS
    }
    complete = set(task_ids)
    extract_fail, pred_yes = {}, {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition and row["task_id"] in complete]
        extract_fail[condition] = float(np.mean([row["pred"] == "" for row in subset])) if subset else None
        pred_yes[condition] = float(np.mean([row["pred"] == "yes" for row in subset])) if subset else None
    labels = [row["label"] for row in rows if row["condition"] == "aligned" and row["task_id"] in complete]
    out = {
        "n_latent": len(task_ids),
        "n_yes": labels.count("yes"),
        "n_no": labels.count("no"),
        "paired_unit": "latent_task",
        "acc": acc,
        "SF": sf,
        "SF_definition": "P(target wrong | aligned right), latent-task bootstrap; ambiguous excluded",
        "ambiguous": {
            "acc": acc["ambiguous"],
            "entailment_broken": True,
            "scored_against": "original_solver_label",
        },
        "extract_fail_rate": extract_fail,
        "pred_yes_rate": pred_yes,
    }
    return out, rows


def _guard_out(out: Path) -> Path:
    resolved = out.resolve()
    for banned in _BANNED_OUT:
        banned_resolved = banned.resolve()
        if resolved == banned_resolved or banned_resolved in resolved.parents:
            raise SystemExit(f"refusing to write into {banned.as_posix()}")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("build/e0/wording"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    out = _guard_out(args.out)
    items, meta = build_items()
    out.mkdir(parents=True, exist_ok=True)
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
        "depth": DEPTH,
        "families": list(FAMILIES),
        "n_per_family": N_PER_FAMILY,
        "vocab": VOCAB,
        "tag": TAG,
        "seed": SEED,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "decoding": "greedy",
        "cot": True,
        "bootstrap_seed": BOOT_SEED,
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
