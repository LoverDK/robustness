"""Cross gold-yes canonical-yes items by a same-premise decoy and the synonym extract.

Reads saved generations only. No training and no model load. A same-premise
decoy is present when some other_rules entry has a premise frozenset equal to
core_rules[0]. That flag uses the latent only. The synonym generation is used
only for extract_yes_no.

right is a synonym extract of yes. wrong is a synonym extract of no. An empty
synonym extract is n_empty and stays outside the four cells.

These counts are not evidence that reasoning improved, and they are not a
win for any training condition.

Writes only experiments/e1/synonym/yes_decoy_availability.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl
from structrobust.prompts import extract_yes_no

SEED = 0
SPLITS = ("seen", "unseen")
SEEN_FAMILIES = ("chain", "conj_fact")
UNSEEN_FAMILIES = ("tree", "diamond")
FAMILIES = {"seen": SEEN_FAMILIES, "unseen": UNSEEN_FAMILIES}
POOL_N = 200
CELLS = (
    "wrong_with_decoy",
    "wrong_without_decoy",
    "right_with_decoy",
    "right_without_decoy",
)
GATES = {
    "seen": {
        "ours": {
            "wrong_with_decoy": 39,
            "wrong_without_decoy": 3,
            "right_with_decoy": 9,
            "right_without_decoy": 26,
        },
        "b2": {
            "wrong_with_decoy": 36,
            "wrong_without_decoy": 1,
            "right_with_decoy": 25,
            "right_without_decoy": 36,
        },
    }
}
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_decoy_availability.json"
BANNED_NAMES = {
    "generations.jsonl",
    "logic.jsonl",
    "metrics.json",
    "label_paired.json",
    "label_sf.json",
    "label_sf_reversed.json",
    "yes_item_overlap.json",
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
    spec = importlib.util.spec_from_file_location("yes_flip_trace_decoy_availability", path)
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


def _blank() -> dict:
    return {name: 0 for name in CELLS} | {"n_empty": 0, "n": 0}


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
    return {"name": name, "synonym": synonym, "logic": logic}


def _check_partition(label: str, block: dict) -> None:
    total = sum(block[name] for name in CELLS) + block["n_empty"]
    if total != block["n"]:
        _stop(
            f"{label} partition failed: cells+empty={total} n={block['n']} "
            f"{ {name: block[name] for name in CELLS} } empty={block['n_empty']}"
        )


def _score_split(bundle: dict, pool: dict, split: str) -> dict:
    block = _blank()
    for digest, lat in pool.items():
        if lat.label != "yes":
            continue
        aligned_uid = f"{split}_aligned|{digest}"
        synonym_uid = f"{split}_synonym|{digest}"
        aligned_text = _generation(bundle["logic"][aligned_uid], aligned_uid)
        if extract_yes_no(aligned_text) != "yes":
            continue
        block["n"] += 1
        synonym_text = _generation(bundle["synonym"][synonym_uid], synonym_uid)
        synonym = extract_yes_no(synonym_text)
        decoy = has_same_premise_decoy(lat)
        if synonym == "":
            block["n_empty"] += 1
        elif synonym == "yes":
            key = "right_with_decoy" if decoy else "right_without_decoy"
            block[key] += 1
        elif synonym == "no":
            key = "wrong_with_decoy" if decoy else "wrong_without_decoy"
            block[key] += 1
        else:
            _stop(f"{bundle['name']} {synonym_uid} extract {synonym!r} is not yes, no, or empty")
    _check_partition(f"{bundle['name']}.{split}", block)
    return block


def _public_block(block: dict) -> dict:
    return {
        "n": block["n"],
        **{name: block[name] for name in CELLS},
        "n_empty": block["n_empty"],
    }


def _gate_report(splits: dict) -> dict:
    cells = {}
    ok = True
    for name, expected in GATES["seen"].items():
        observed = {key: splits["seen"][name][key] for key in CELLS}
        cell_ok = observed == expected
        ok = ok and cell_ok
        cells[name] = {"ok": cell_ok, "observed": observed, "expected": dict(expected)}
    return {"ok": ok, "seen": cells}


def _print_splits(splits: dict) -> None:
    for split in SPLITS:
        for name, block in splits[split].items():
            fields = " ".join(f"{key}={block[key]}" for key in (*CELLS, "n_empty", "n"))
            print(f"{split} {name} {fields}")


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    _self_check()
    trace = _load_trace()
    if len(trace.CONDITIONS) != 9:
        _stop(f"expected 9 conditions, found {len(trace.CONDITIONS)}")
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
    splits = {split: {} for split in SPLITS}
    for name, gen_rel, logic_rel in trace.CONDITIONS:
        bundle = _load_condition(name, gen_rel, logic_rel, indexed)
        for split in SPLITS:
            splits[split][name] = _score_split(bundle, indexed[split], split)
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    gates = _gate_report(splits)
    public = {
        split: {name: _public_block(block) for name, block in splits[split].items()}
        for split in SPLITS
    }
    for name, cell in gates["seen"].items():
        observed = " ".join(f"{key}={cell['observed'][key]}" for key in CELLS)
        print(f"gate seen {name} ok={cell['ok']} {observed}")
    if not gates["ok"]:
        print("STOP gate mismatch; counts were not adjusted and the json was not written")
        for name, cell in gates["seen"].items():
            if not cell["ok"]:
                print(f"expected seen {name} {cell['expected']}")
        _print_splits(public)
        raise SystemExit(1)
    payload = {
        "config": {
            "seed": SEED,
            "pools": {
                "seen": "test_seen from build_pools(seed=0)",
                "unseen": "test_unseen from build_pools(seed=0)",
            },
            "population": (
                "latent label yes and aligned extract yes. "
                "n is that count. The synonym extract then splits n into the four cells "
                "and n_empty."
            ),
            "aligned_generation": (
                "{seen|unseen}_aligned|{graph_hash} in each condition's logic.jsonl. "
                "The row is not regenerated. It is used only to require extract yes."
            ),
            "synonym_generation": (
                "{seen|unseen}_synonym|{graph_hash} in each condition's generations.jsonl"
            ),
            "decoy": (
                "some other_rules entry has a premise frozenset equal to "
                "core_rules[0]. The comparison uses the latent only. Generation text "
                "is not read for this flag."
            ),
            "premise_equality": "frozenset of the premise atoms, lowercased; order is ignored",
            "cells": {
                "wrong_with_decoy": "synonym extract no, and the latent has a same-premise decoy",
                "wrong_without_decoy": "synonym extract no, and the latent has no same-premise decoy",
                "right_with_decoy": "synonym extract yes, and the latent has a same-premise decoy",
                "right_without_decoy": "synonym extract yes, and the latent has no same-premise decoy",
            },
            "n_empty": (
                "synonym extract is empty. These items stay in n and stay out of the four cells."
            ),
            "partition": "wrong_with_decoy + wrong_without_decoy + right_with_decoy + right_without_decoy + n_empty = n",
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "conditions": [name for name, _gen, _logic in trace.CONDITIONS],
            "gates": (
                "seen pilot ours: wrong_with_decoy 39, wrong_without_decoy 3, "
                "right_with_decoy 9, right_without_decoy 26. "
                "seen pilot b2: wrong_with_decoy 36, wrong_without_decoy 1, "
                "right_with_decoy 25, right_without_decoy 36. "
                "A mismatch stops the script before it writes this file. "
                "Counts are not adjusted to meet the gate."
            ),
            "loaded_model": False,
            "training": False,
            "reading": (
                "A cross-tab of whether the latent has a same-premise decoy and "
                "whether the saved synonym extract is yes or no. "
                "These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
            ),
        },
        "gates": gates,
        "splits": public,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    _print_splits(public)
    print(f"gates_ok {gates['ok']}")


if __name__ == "__main__":
    main()
