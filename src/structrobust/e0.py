"""E0 sanity gate: no training. CoT evaluation across difficulty levels and presentations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import logic
from .burden import burden
from .prompts import extract_yes_no, logic_prompt
from .stats import conditional_flip_ci, mean_ci, paired_diff_ci

SHIFTS = ("reversed", "random", "random_distract")


def build_items(depths, n, seed, families=logic.SEEN_FAMILIES, vocab="eval"):
    items, meta = [], {}
    per_fam = n // len(families)
    for d in depths:
        pool = logic.make_pool(families, [d], per_fam, seed, vocab, tag="e0")
        for lat in pool:
            for p in logic.PRESENTATIONS:
                r = logic.render(lat, p)
                uid = f"{lat.task_id}|{p}"
                items.append({"uid": uid, "prompt": logic_prompt(r["text"])})
                meta[uid] = {"task_id": lat.task_id, "depth": d, "family": lat.family,
                             "label": lat.label, "presentation": p,
                             **burden(lat, r["core_positions"])}
    return items, meta


def analyze(meta: dict, gens: dict, gate_lo=0.55, gate_hi=0.90, sf_min=0.10) -> dict:
    by = defaultdict(dict)
    rows = []
    for uid, m in meta.items():
        g = gens.get(uid)
        if g is None:
            continue
        pred = extract_yes_no(g["generation"])
        correct = int(pred == m["label"])
        by[(m["depth"], m["task_id"])][m["presentation"]] = correct
        rows.append({**m, "uid": uid, "pred": pred, "correct": correct})
    depths = sorted({m["depth"] for m in meta.values()})
    out = {"levels": {}, "gate": {}}
    for d in depths:
        tids = sorted(t for (dd, t) in by if dd == d and len(by[(dd, t)]) == len(logic.PRESENTATIONS))
        lev = {"n_latent": len(tids), "acc": {}, "diff_vs_aligned": {}, "SF": {}}
        vec = {p: np.array([by[(d, t)][p] for t in tids]) for p in logic.PRESENTATIONS}
        for p, v in vec.items():
            lev["acc"][p] = mean_ci(v, seed=d)
        for p in SHIFTS:
            lev["diff_vs_aligned"][p] = paired_diff_ci(vec["aligned"], vec[p], seed=d)
            lev["SF"][p] = conditional_flip_ci(vec["aligned"], vec[p], seed=d)
        rr = [r for r in rows if r["depth"] == d]
        lev["extract_fail_rate"] = float(np.mean([r["pred"] == "" for r in rr]))
        lev["pred_yes_rate"] = float(np.mean([r["pred"] == "yes" for r in rr]))
        acc_a, lo_a, hi_a = lev["acc"]["aligned"]
        sf, sf_lo, _ = lev["SF"]["random"]
        passed = (acc_a is not None and sf is not None and gate_lo <= acc_a <= gate_hi and lo_a > 0.5
                  and sf >= sf_min and sf_lo > 0)
        lev["gate_pass"] = bool(passed)
        out["levels"][str(d)] = lev
    out["gate"]["any_level_pass"] = any(v["gate_pass"] for v in out["levels"].values())
    out["gate"]["criteria"] = (f"aligned acc in [{gate_lo},{gate_hi}], aligned CI lower > 0.5 (majority baseline), "
                               f"SF(aligned->random) >= {sf_min} with CI lower > 0")
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("build/e0/round1"))
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    items, meta = build_items(args.depths, args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    gen_path = args.out / "generations.jsonl"
    if not args.analyze_only:
        from .infer import generate, load_model

        tok, model = load_model(args.model)
        generate(tok, model, items, gen_path, max_new_tokens=args.max_new_tokens,
                 batch_size=args.batch_size, shuffle_seed=args.seed)
    gens = {}
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gens[r["uid"]] = r
    res, rows = analyze(meta, gens)
    res["config"] = {"model": args.model, "depths": args.depths, "n_per_level": args.n,
                     "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                     "families": list(logic.SEEN_FAMILIES), "vocab": "eval", "decoding": "greedy"}
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
