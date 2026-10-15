"""Paired synonym-error difference on common items with a same-premise decoy.

Reads saved generations only. No training and no model load. A common item
has gold label yes, a same-premise decoy, aligned extracts of yes from both
conditions, and synonym extracts of yes or no from both. An empty synonym
extract on either side stays out.

The wrong indicator is 1 when the synonym extract is no, and 0 otherwise.
The difference is the left indicator minus the right indicator. A larger
difference means more synonym extracts of no. It is not an accuracy win,
and it is not evidence that reasoning improved.

Writes only experiments/e1/synonym/yes_decoy_paired.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl
from structrobust.prompts import extract_yes_no
from structrobust.stats import paired_diff_ci

SEED = 0
N_BOOT = 2000
SPLITS = ("seen", "unseen")
SEEN_FAMILIES = ("chain", "conj_fact")
UNSEEN_FAMILIES = ("tree", "diamond")
FAMILIES = {"seen": SEEN_FAMILIES, "unseen": UNSEEN_FAMILIES}
POOL_N = 200
NEEDED = ("ours", "b2", "mix15_ours", "mix15_b2")
PAIRS = (
    ("seen", "ours_minus_b2", "ours", "b2"),
    ("seen", "mix15_ours_minus_mix15_b2", "mix15_ours", "mix15_b2"),
    ("unseen", "ours_minus_b2", "ours", "b2"),
    ("unseen", "mix15_ours_minus_mix15_b2", "mix15_ours", "mix15_b2"),
)
GATES = {
    ("seen", "ours", "b2"): {"n": 48, "left_wrong": 39, "right_wrong": 27},
    ("seen", "mix15_ours", "mix15_b2"): {"n": 48, "left_wrong": 43, "right_wrong": 29},
}
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_decoy_paired.json"
BANNED_NAMES = {
    "generations.jsonl",
    "logic.jsonl",
    "metrics.json",
    "label_paired.json",
    "label_sf.json",
    "label_sf_reversed.json",
    "yes_item_overlap.json",
    "yes_decoy_availability.json",
    "yes_flip_trace.json",
    "yes_flip_rule_source.json",
    "yes_flip_aligned_trace.json",
    "yes_flip_first_since.json",
    "yes_flip_first_since_unseen.json",
    "yes_flip_unseen_later_core.json",
    "yes_flip_unseen_aligned_on_later.json",
    "yes_correct_since_seen.json",
    "yes_correct_since_unseen.json",
    "synonym_burden_contrast.json",
    "burden_contrast.json",
    "report.md",
    "b0_metrics.json",
    "b1_metrics.json",
    "b2_metrics.json",
    "b3_metrics.json",
    "ours_metrics.json",
    "mix15_b1_metrics.json",
    "mix15_b2_metrics.json",
    "mix15_b3_metrics.json",
    "mix15_ours_metrics.json",
}


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _load_trace():
    path = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_trace.py"
    spec = importlib.util.spec_from_file_location("yes_flip_trace_decoy_paired", path)
    if spec is None or spec.loader is None:
        _stop(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_same_premise_decoy(lat) -> bool:
    """True when some other rule shares the step-0 premise set. Latent only."""
    if not lat.core_rules:
        _stop("a latent has no core rules")
    step0 = frozenset(str(premise).lower() for premise in lat.core_rules[0][0])
    for premises, _conclusion in lat.other_rules:
        if frozenset(str(premise).lower() for premise in premises) == step0:
            return True
    return False


def _generation(row: dict, uid: str) -> str:
    text = row.get("generation")
    if not isinstance(text, str):
        _stop(f"{uid} has no generation text")
    return text


def _guard_out(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OUT.resolve():
        _stop(f"refusing to write {resolved}")
    if resolved.name in BANNED_NAMES or "rescore" in resolved.parts or "paper" in resolved.parts:
        _stop(f"refusing to write {resolved}")
    if "build" in resolved.parts:
        _stop(f"refusing to write {resolved}")
    return resolved


def _self_check() -> None:
    present = SimpleNamespace(
        core_rules=((("vivid",), "silver"),),
        other_rules=((("vivid",), "windy"),),
    )
    if not has_same_premise_decoy(present):
        _stop("a same-premise other rule was missed")
    absent = SimpleNamespace(
        core_rules=((("vivid",), "silver"),),
        other_rules=((("other",), "silver"),),
    )
    if has_same_premise_decoy(absent):
        _stop("a different-premise other rule counted as a decoy")
    swapped = SimpleNamespace(
        core_rules=((("a", "b"), "step0"),),
        other_rules=((("b", "a"), "windy"),),
    )
    if not has_same_premise_decoy(swapped):
        _stop("frozenset equality missed swapped premises")
    empty = SimpleNamespace(
        core_rules=((("vivid",), "silver"),),
        other_rules=(),
    )
    if has_same_premise_decoy(empty):
        _stop("an empty other_rules list counted as a decoy")
    core_only = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("vivid",), "windy")),
        other_rules=((("x",), "amber"),),
    )
    if has_same_premise_decoy(core_only):
        _stop("a later core rule counted as an other-rule decoy")
    cased = SimpleNamespace(
        core_rules=((("Vivid",), "silver"),),
        other_rules=((("vivid",), "windy"),),
    )
    if not has_same_premise_decoy(cased):
        _stop("premise equality missed a case difference")
    if extract_yes_no("Answer: yes") != "yes" or extract_yes_no("Answer: no") != "no":
        _stop("extract self-check failed")
    if extract_yes_no("the derivation stops here") != "":
        _stop("an empty extract was not empty")
    point, low, high = paired_diff_ci([1, 0, 1, 0], [0, 0, 1, 1], n_boot=50, seed=SEED)
    manual = sum((1, 0, 0, -1)) / 4
    if point != manual:
        _stop(f"paired point {point} is not the mean of indicator differences {manual}")
    if low is None or high is None or not (low <= high):
        _stop("paired self-check returned an empty interval")


def _index(items, families: tuple[str, ...], split: str) -> dict:
    found = {}
    for lat in items:
        digest = graph_hash(lat)
        if digest in found:
            _stop(f"duplicate {split} graph hash {digest}")
        if lat.family not in families:
            _stop(f"{digest} family {lat.family} is outside {split}")
        if not lat.core_rules:
            _stop(f"{digest} has no core rules")
        found[digest] = lat
    if len(found) != POOL_N:
        _stop(f"{split} pool has {len(found)} graphs")
    present = {lat.family for lat in found.values()}
    if present != set(families):
        _stop(f"{split} families {sorted(present)} are not {families}")
    return found


def _load_condition(name: str, gen_rel: str, logic_rel: str, pools: dict) -> dict:
    gen_path = PROJECT / gen_rel
    logic_path = PROJECT / logic_rel
    if not gen_path.is_file() or not logic_path.is_file():
        _stop(f"missing input for {name}: {gen_rel} or {logic_rel}")
    synonym = read_jsonl(gen_path)
    logic = read_jsonl(logic_path)
    missing_synonym = []
    missing_aligned = []
    for split, pool in pools.items():
        for digest in pool:
            synonym_uid = f"{split}_synonym|{digest}"
            aligned_uid = f"{split}_aligned|{digest}"
            if synonym_uid not in synonym or not isinstance(synonym[synonym_uid].get("generation"), str):
                missing_synonym.append(synonym_uid)
            if aligned_uid not in logic or not isinstance(logic[aligned_uid].get("generation"), str):
                missing_aligned.append(aligned_uid)
    if missing_synonym:
        listed = "\n".join(missing_synonym)
        _stop(f"{name} synonym generations missing {len(missing_synonym)} uid(s):\n{listed}")
    if missing_aligned:
        listed = "\n".join(missing_aligned)
        _stop(f"{name} aligned generations missing {len(missing_aligned)} uid(s):\n{listed}")
    return {"name": name, "synonym": synonym, "logic": logic, "gen_rel": gen_rel, "logic_rel": logic_rel}


def _extract(bundle: dict, kind: str, split: str, digest: str) -> str:
    if kind == "aligned":
        uid = f"{split}_aligned|{digest}"
        text = _generation(bundle["logic"][uid], uid)
    elif kind == "synonym":
        uid = f"{split}_synonym|{digest}"
        text = _generation(bundle["synonym"][uid], uid)
    else:
        _stop(f"unknown extract kind {kind}")
    got = extract_yes_no(text)
    if got not in {"yes", "no", ""}:
        _stop(f"{bundle['name']} {uid} extract {got!r} is not yes, no, or empty")
    return got


def _common_indicators(left: dict, right: dict, pool: dict, split: str) -> tuple[list[int], list[int]]:
    left_ind: list[int] = []
    right_ind: list[int] = []
    for digest in sorted(pool):
        lat = pool[digest]
        if lat.label != "yes" or not has_same_premise_decoy(lat):
            continue
        if _extract(left, "aligned", split, digest) != "yes":
            continue
        if _extract(right, "aligned", split, digest) != "yes":
            continue
        left_syn = _extract(left, "synonym", split, digest)
        right_syn = _extract(right, "synonym", split, digest)
        if left_syn not in {"yes", "no"} or right_syn not in {"yes", "no"}:
            continue
        left_ind.append(1 if left_syn == "no" else 0)
        right_ind.append(1 if right_syn == "no" else 0)
    return left_ind, right_ind


def _counts(left_ind: list[int], right_ind: list[int]) -> dict:
    if len(left_ind) != len(right_ind):
        _stop("paired indicator lists have different lengths")
    return {
        "n": len(left_ind),
        "left_wrong": int(sum(left_ind)),
        "right_wrong": int(sum(right_ind)),
    }


def _diff_block(left_ind: list[int], right_ind: list[int], counts: dict) -> dict:
    n = counts["n"]
    if n == 0:
        _stop("a pair has no common items, so the interval was not invented")
    point, low, high = paired_diff_ci(left_ind, right_ind, n_boot=N_BOOT, seed=SEED)
    if point is None or low is None or high is None:
        _stop("paired_diff_ci returned an empty interval on a nonempty pair")
    manual = sum(a - b for a, b in zip(left_ind, right_ind)) / n
    if point != manual:
        _stop(
            f"point {point} is not the mean of the indicator differences {manual}; "
            f"n={n} left_wrong={counts['left_wrong']} right_wrong={counts['right_wrong']}"
        )
    return {
        "n": n,
        "left_wrong": counts["left_wrong"],
        "right_wrong": counts["right_wrong"],
        "diff": [point, low, high],
        "contains_0": bool(low <= 0.0 <= high),
    }


def _gate_report(observed: dict) -> dict:
    cells = {}
    ok = True
    for (split, left_name, right_name), expected in GATES.items():
        key = f"{left_name}_minus_{right_name}"
        got = observed[split][key]
        cell_ok = {field: got[field] for field in ("n", "left_wrong", "right_wrong")} == expected
        ok = ok and cell_ok
        cells[key] = {"ok": cell_ok, "observed": {field: got[field] for field in expected}, "expected": dict(expected)}
    return {"ok": ok, "seen": cells}


def _print_pairs(pairs: dict) -> None:
    for split in SPLITS:
        for key, block in pairs[split].items():
            point, low, high = block["diff"]
            print(
                f"{split} {key} n={block['n']} left_wrong={block['left_wrong']} "
                f"right_wrong={block['right_wrong']} diff={point} "
                f"p2.5={low} p97.5={high} contains_0={block['contains_0']}"
            )


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    _self_check()
    trace = _load_trace()
    if len(trace.CONDITIONS) != 9:
        _stop(f"expected 9 conditions, found {len(trace.CONDITIONS)}")
    paths = {name: (gen_rel, logic_rel) for name, gen_rel, logic_rel in trace.CONDITIONS}
    for name in NEEDED:
        if name not in paths:
            _stop(f"{name} is missing from yes_flip_trace.CONDITIONS")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    indexed = {
        "seen": _index(pools["test_seen"], SEEN_FAMILIES, "seen"),
        "unseen": _index(pools["test_unseen"], UNSEEN_FAMILIES, "unseen"),
    }
    overlap = set(indexed["seen"]) & set(indexed["unseen"])
    if overlap:
        _stop(f"seen and unseen share {len(overlap)} graph hashes")
    bundles = {}
    for name in NEEDED:
        gen_rel, logic_rel = paths[name]
        bundles[name] = _load_condition(name, gen_rel, logic_rel, indexed)
    if "torch" in sys.modules:
        _stop("torch was imported while reading generations")
    raw = {split: {} for split in SPLITS}
    for split, key, left_name, right_name in PAIRS:
        left_ind, right_ind = _common_indicators(
            bundles[left_name], bundles[right_name], indexed[split], split,
        )
        raw[split][key] = {
            "left": left_name,
            "right": right_name,
            "left_ind": left_ind,
            "right_ind": right_ind,
            **_counts(left_ind, right_ind),
        }
    gates = _gate_report(raw)
    for key, cell in gates["seen"].items():
        observed = " ".join(f"{field}={cell['observed'][field]}" for field in ("n", "left_wrong", "right_wrong"))
        print(f"gate seen {key} ok={cell['ok']} {observed}")
    if not gates["ok"]:
        print("STOP gate mismatch; counts were not adjusted and the json was not written")
        for key, cell in gates["seen"].items():
            if not cell["ok"]:
                print(f"expected seen {key} {cell['expected']}")
        raise SystemExit(1)
    public = {split: {} for split in SPLITS}
    for split, key, left_name, right_name in PAIRS:
        block = raw[split][key]
        scored = _diff_block(block["left_ind"], block["right_ind"], block)
        public[split][key] = {
            "left": left_name,
            "right": right_name,
            **scored,
        }
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    payload = {
        "config": {
            "seed": SEED,
            "n_boot": N_BOOT,
            "alpha": 0.05,
            "interval": "[point, 2.5%, 97.5%]",
            "statistic": "paired_diff_ci(left_wrong_indicator, right_wrong_indicator)",
            "point_identity": (
                "diff[0] equals the mean of left indicator minus right indicator. "
                "A mismatch stops the script before it writes this file."
            ),
            "contains_0_rule": "true when the closed interval includes 0, including when an endpoint is 0",
            "difference": "left minus right",
            "wrong_indicator": "1 when the synonym extract is no, otherwise 0",
            "hash_order": "sorted graph_hash within the pair",
            "pools": {
                "seen": "test_seen from build_pools(seed=0)",
                "unseen": "test_unseen from build_pools(seed=0)",
            },
            "population": (
                "gold label yes, a same-premise decoy, both aligned extracts yes, "
                "and both synonym extracts yes or no. An empty synonym extract on "
                "either side stays out."
            ),
            "aligned_generation": (
                "{seen|unseen}_aligned|{graph_hash} in each condition's logic.jsonl. "
                "The row is not regenerated."
            ),
            "synonym_generation": (
                "{seen|unseen}_synonym|{graph_hash} in each condition's generations.jsonl"
            ),
            "decoy": (
                "some other_rules entry has a premise frozenset equal to "
                "core_rules[0]. The comparison uses the latent only."
            ),
            "premise_equality": "frozenset of the premise atoms, lowercased; order is ignored",
            "paths": {
                name: {"synonym": bundles[name]["gen_rel"], "aligned": bundles[name]["logic_rel"]}
                for name in NEEDED
            },
            "pairs": [
                {"split": split, "key": key, "left": left_name, "right": right_name}
                for split, key, left_name, right_name in PAIRS
            ],
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "gates": (
                "seen ours minus b2: n=48, left_wrong=39, right_wrong=27. "
                "seen mix15_ours minus mix15_b2: n=48, left_wrong=43, right_wrong=29. "
                "A mismatch stops the script before it writes this file. "
                "Counts are not adjusted to meet the gate."
            ),
            "loaded_model": False,
            "training": False,
            "gpu": False,
            "reading": (
                "On common items where both canonical extracts are yes and the latent "
                "has a same-premise decoy, this is how often the synonym extract is no. "
                "A larger difference means more synonym extracts of no. "
                "It is not an accuracy win, it is not evidence that reasoning improved, "
                "and it is not a win for any training condition."
            ),
        },
        "gates": gates,
        "pairs": public,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    _print_pairs(public)
    print(f"gates_ok {gates['ok']}")


if __name__ == "__main__":
    main()
