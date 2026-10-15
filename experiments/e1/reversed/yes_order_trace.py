"""Since-chain trace when a reversed answer flips gold yes to no.

Reads saved logic.jsonl rows only. No training and no model load. This is
the reversed presentation, not a synonym rewrite, and this file does not
open generations.jsonl. The population is a seen or unseen graph, latent
label yes, aligned extract yes, and shifted extract no. An empty shifted
extract is counted separately and is not classified.

Since atoms are yes_flip_trace.since_atoms and keep that function's order.
The first atom uses yes_flip_first_since.first_since_class. Premise equality
uses frozenset. has_query and has_step0 ask whether any reversed Since atom
is the query or the step-0 conclusion. Those two counts may overlap and are
not part of the seven-class partition. The seven classes sum to n.

These counts are not evidence that reasoning improved, and they are not a
win for any training condition.

Writes only experiments/e1/reversed/yes_order_trace.json.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl
from structrobust.prompts import extract_yes_no

SEED = 0
POOL_N = 200
CLASSES = (
    "step0",
    "query",
    "same_premise_decoy",
    "later_core",
    "other_decoy",
    "other",
    "none",
)
SPLITS = (
    ("seen", "test_seen", ("chain", "conj_fact")),
    ("unseen", "test_unseen", ("tree", "diamond")),
)
CONDITIONS = (
    ("b0", "build/e1/pilot/b0/logic.jsonl"),
    ("b1", "build/e1/pilot/b1/logic.jsonl"),
    ("b2", "build/e1/pilot/b2/logic.jsonl"),
    ("b3", "build/e1/pilot/b3/logic.jsonl"),
    ("ours", "build/e1/pilot/ours/logic.jsonl"),
    ("mix15_b1", "build/e1/pilot_mix15/b1/logic.jsonl"),
    ("mix15_b2", "build/e1/pilot_mix15/b2/logic.jsonl"),
    ("mix15_b3", "build/e1/pilot_mix15/b3/logic.jsonl"),
    ("mix15_ours", "build/e1/pilot_mix15/ours/logic.jsonl"),
)
GATES = {
    "ours": {
        "seen": {
            "n": 7,
            "has_query": 0,
            "has_step0": 6,
            "first": {
                "step0": 4,
                "query": 0,
                "same_premise_decoy": 3,
                "later_core": 0,
                "other_decoy": 0,
                "other": 0,
                "none": 0,
            },
        }
    }
}
OUT = PROJECT / "experiments" / "e1" / "reversed" / "yes_order_trace.json"
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
    "yes_decoy_availability.json",
    "yes_decoy_paired.json",
    "synonym_burden_contrast.json",
    "burden_contrast.json",
    "report.md",
}


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _load_module(rel: str, module_name: str):
    path = PROJECT / rel
    if "synonym" not in path.parts or path.suffix != ".py":
        _stop(f"refusing to load {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _stop(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blank_first() -> dict:
    return {name: 0 for name in CLASSES}


def population_slot(label: str, aligned_pred: str, shifted_pred: str) -> str:
    """flip is classified; empty is counted aside; everything else is out."""
    if label != "yes" or aligned_pred != "yes":
        return "out"
    if shifted_pred == "":
        return "empty"
    if shifted_pred == "no":
        return "flip"
    return "out"


def chain_flags(atoms: list[str], lat, first_since_class) -> dict:
    """Query and step-0 presence on the whole chain, plus the first-atom class."""
    kind = first_since_class(atoms, lat)
    if kind not in CLASSES:
        _stop(f"unknown first-atom class {kind}")
    found = {str(atom).lower() for atom in atoms}
    step0 = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    return {
        "first": kind,
        "has_query": query in found,
        "has_step0": step0 in found,
    }


def _generation(row: dict, uid: str) -> str:
    text = row.get("generation")
    if not isinstance(text, str):
        _stop(f"{uid} has no generation text")
    return text


def _guard_out(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OUT.resolve():
        _stop(f"refusing to write {resolved}")
    if resolved.name in BANNED_NAMES or "synonym" in resolved.parts or "rescore" in resolved.parts or "paper" in resolved.parts:
        _stop(f"refusing to write {resolved}")
    return resolved


def _guard_logic(rel: str) -> Path:
    path = (PROJECT / rel).resolve()
    if path.name != "logic.jsonl":
        _stop(f"refusing to read {path}")
    if "synonym" in path.parts or path.name == "generations.jsonl":
        _stop(f"refusing to read {path}")
    if not path.is_file():
        _stop(f"missing {rel}")
    return path


def _self_check(since_atoms, first_since_class) -> None:
    if "frozenset" not in inspect.getsource(first_since_class):
        _stop("first_since_class does not compare premises with frozenset")
    if "cannot be proved" not in inspect.getsource(since_atoms):
        _stop("since_atoms does not skip a cannot-be-proved line")
    sample = (
        "Known facts: Tina is vivid.\n"
        "Since Tina is vivid, Tina is windy.\n"
        "Nothing else can be derived, so Tina is snowy cannot be proved.\n"
        "Answer: no"
    )
    if since_atoms(sample, "Tina") != ["windy"]:
        _stop("regex self-check did not keep the intermediate atom only")
    proved = (
        "Since Tina is vivid, Tina is snowy cannot be proved.\n"
        "Since Tina is vivid, Tina is windy."
    )
    if since_atoms(proved, "Tina") != ["windy"]:
        _stop("regex self-check let a cannot-be-proved line become the first atom")
    ordered = (
        "Since Tina is vivid, Tina is windy. "
        "Since Tina is windy, Tina is silver."
    )
    if since_atoms(ordered, "Tina") != ["windy", "silver"]:
        _stop("regex self-check did not keep Since atoms in order")

    chain = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "snowy"), (("snowy",), "muddy")),
        other_rules=((("vivid",), "windy"), (("otherprem",), "muddy"), (("x",), "amber")),
        query="snowy",
        entity="Tina",
    )
    empty = chain_flags([], chain, first_since_class)
    if empty["first"] != "none" or empty["has_query"] or empty["has_step0"]:
        _stop("no Since sentence was not none")
    step0_first = chain_flags(["silver", "windy"], chain, first_since_class)
    if step0_first["first"] != "step0" or not step0_first["has_step0"] or step0_first["has_query"]:
        _stop("the step-0 conclusion was not step0")
    query_first = chain_flags(["snowy"], chain, first_since_class)
    if query_first["first"] != "query" or not query_first["has_query"] or query_first["has_step0"]:
        _stop("the query atom was not query")
    decoy = chain_flags(["windy", "silver", "snowy"], chain, first_since_class)
    if decoy["first"] != "same_premise_decoy" or not decoy["has_step0"] or not decoy["has_query"]:
        _stop("a same-premise decoy did not keep later step 0 and the query")
    if chain_flags(["amber"], chain, first_since_class)["first"] != "other_decoy":
        _stop("a different-premise decoy was not other_decoy")
    if chain_flags(["invented"], chain, first_since_class)["first"] != "other":
        _stop("an atom outside the rules was not other")
    later = chain_flags(["muddy"], chain, first_since_class)
    if later["first"] != "later_core" or later["has_query"] or later["has_step0"]:
        _stop("a later core conclusion was not later_core")

    query_is_step0 = SimpleNamespace(
        core_rules=((("vivid",), "snowy"),),
        other_rules=((("vivid",), "snowy"),),
        query="snowy",
        entity="Tina",
    )
    both = chain_flags(["snowy"], query_is_step0, first_since_class)
    if both["first"] != "step0" or not both["has_query"] or not both["has_step0"]:
        _stop("a query that is the step-0 conclusion was not step0")

    swapped = SimpleNamespace(
        core_rules=((("a", "b"), "step0"), (("step0",), "windy")),
        other_rules=((("b", "a"), "windy"),),
        query="later",
        entity="Tina",
    )
    if chain_flags(["windy"], swapped, first_since_class)["first"] != "same_premise_decoy":
        _stop("frozenset equality missed swapped premises, or later core won")

    shared = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "muddy")),
        other_rules=((("otherprem",), "muddy"),),
        query="silver",
        entity="Tina",
    )
    if chain_flags(["muddy"], shared, first_since_class)["first"] != "later_core":
        _stop("a later core conclusion lost to a different-premise decoy")
    if chain_flags(["silver"], shared, first_since_class)["first"] != "step0":
        _stop("the step-0 conclusion lost to the query")

    if population_slot("yes", "yes", "no") != "flip":
        _stop("a reversed no was not classified")
    if population_slot("yes", "yes", "") != "empty":
        _stop("an empty reversed extract entered classification")
    if population_slot("yes", "yes", "yes") != "out":
        _stop("a reversed yes entered the flip population")
    if population_slot("no", "yes", "no") != "out":
        _stop("a gold-no item entered the population")
    if population_slot("yes", "no", "no") != "out":
        _stop("an aligned no entered the population")
    if population_slot("yes", "", "no") != "out":
        _stop("an empty aligned extract entered the population")


def _index_pool(pool, families: tuple[str, ...], split: str) -> dict:
    indexed = {}
    for lat in pool:
        digest = graph_hash(lat)
        if digest in indexed:
            _stop(f"duplicate {split} graph hash {digest}")
        if lat.family not in families:
            _stop(f"{digest} family {lat.family} is outside the {split} families")
        if not lat.core_rules:
            _stop(f"{digest} has no core rules")
        indexed[digest] = lat
    if len(indexed) != POOL_N:
        _stop(f"{split} pool has {len(indexed)} graphs")
    return indexed


def _require_rows(name: str, logic: dict, pools: dict) -> None:
    missing = []
    for split, _pool_name, _families in SPLITS:
        for digest in pools[split]:
            for suffix in ("aligned", "shifted"):
                uid = f"{split}_{suffix}|{digest}"
                row = logic.get(uid)
                if not isinstance(row, dict) or not isinstance(row.get("generation"), str):
                    missing.append(uid)
    if missing:
        listed = "\n".join(missing[:20])
        _stop(f"{name} logic.jsonl missing {len(missing)} uid(s):\n{listed}")


def _score_split(name: str, split: str, logic: dict, pool: dict, since_atoms, first_since_class) -> dict:
    n = 0
    n_empty = 0
    has_query = 0
    has_step0 = 0
    first = _blank_first()
    for digest, lat in pool.items():
        aligned_uid = f"{split}_aligned|{digest}"
        shifted_uid = f"{split}_shifted|{digest}"
        aligned_pred = extract_yes_no(_generation(logic[aligned_uid], aligned_uid))
        shifted_text = _generation(logic[shifted_uid], shifted_uid)
        shifted_pred = extract_yes_no(shifted_text)
        slot = population_slot(lat.label, aligned_pred, shifted_pred)
        if slot == "out":
            continue
        if slot == "empty":
            n_empty += 1
            continue
        atoms = since_atoms(shifted_text, lat.entity)
        flags = chain_flags(atoms, lat, first_since_class)
        first[flags["first"]] += 1
        if flags["has_query"]:
            has_query += 1
        if flags["has_step0"]:
            has_step0 += 1
        n += 1
    total = sum(first.values())
    if total != n:
        _stop(f"{name}.{split} classes {total} != n {n} {first}")
    if has_query > n or has_step0 > n:
        _stop(f"{name}.{split} presence counts exceed n={n}")
    if has_step0 < first["step0"]:
        _stop(f"{name}.{split} has_step0 is below first step0")
    if has_query < first["query"]:
        _stop(f"{name}.{split} has_query is below first query")
    return {
        "n": n,
        "n_empty_extract": n_empty,
        "has_query": has_query,
        "has_step0": has_step0,
        "first": first,
    }


def _gate_cell(observed: dict, expected: dict) -> dict:
    first_ok = observed["first"] == expected["first"]
    if sum(expected["first"].values()) != expected["n"]:
        _stop(f"gate table does not partition n={expected['n']}")
    ok = (
        observed["n"] == expected["n"]
        and observed["has_query"] == expected["has_query"]
        and observed["has_step0"] == expected["has_step0"]
        and first_ok
    )
    return {
        "ok": ok,
        "observed": {
            "n": observed["n"],
            "has_query": observed["has_query"],
            "has_step0": observed["has_step0"],
            "first": observed["first"],
        },
        "expected": expected,
    }


def _gate_report(conditions: dict) -> dict:
    report = {}
    ok = True
    for name, splits in GATES.items():
        for split, expected in splits.items():
            cell = _gate_cell(conditions[name][split], expected)
            report[f"{name}_{split}"] = cell
            ok = ok and cell["ok"]
    report["ok"] = ok
    return report


def _print_split(name: str, split: str, block: dict) -> None:
    nonzero = [f"{key}={block['first'][key]}" for key in CLASSES if block["first"][key]]
    shown = " ".join(nonzero) if nonzero else "none"
    print(
        f"{name} {split} n={block['n']} empty={block['n_empty_extract']} "
        f"has_query={block['has_query']} has_step0={block['has_step0']} first({shown})"
    )


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_module("experiments/e1/synonym/yes_flip_trace.py", "yes_flip_trace_order")
    first_mod = _load_module(
        "experiments/e1/synonym/yes_flip_first_since.py",
        "yes_flip_first_since_order",
    )
    since_atoms = trace.since_atoms
    first_since_class = first_mod.first_since_class
    _self_check(since_atoms, first_since_class)
    if len(CONDITIONS) != 9:
        _stop(f"expected 9 conditions, found {len(CONDITIONS)}")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    indexed = {
        split: _index_pool(pools[pool_name], families, split)
        for split, pool_name, families in SPLITS
    }
    conditions = {}
    for name, rel in CONDITIONS:
        logic = read_jsonl(_guard_logic(rel))
        _require_rows(name, logic, indexed)
        conditions[name] = {
            split: _score_split(name, split, logic, indexed[split], since_atoms, first_since_class)
            for split, _pool_name, _families in SPLITS
        }
        for split, _pool_name, _families in SPLITS:
            total = sum(conditions[name][split]["first"].values())
            if total != conditions[name][split]["n"]:
                _stop(f"{name}.{split} classes {total} != n {conditions[name][split]['n']}")
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    gates = _gate_report(conditions)
    for key, cell in gates.items():
        if key == "ok":
            continue
        print(
            f"gate {key} ok={cell['ok']} n={cell['observed']['n']} "
            f"expected_n={cell['expected']['n']} "
            f"has_query={cell['observed']['has_query']} "
            f"has_step0={cell['observed']['has_step0']}"
        )
    if not gates["ok"]:
        print("STOP gate mismatch; counts were not adjusted and the json was not written")
        for key, cell in gates.items():
            if key == "ok":
                continue
            print(f"observed {key} {cell['observed']}")
            print(f"expected {key} {cell['expected']}")
        raise SystemExit(1)
    payload = {
        "config": {
            "seed": SEED,
            "pools": "test_seen and test_unseen from build_pools(seed=0)",
            "splits": {
                "seen": "test_seen; families chain and conj_fact; uid seen_aligned|{hash} and seen_shifted|{hash}",
                "unseen": "test_unseen; families tree and diamond; uid unseen_aligned|{hash} and unseen_shifted|{hash}",
            },
            "presentation": "shifted rows are the generator's reversed presentation, not a synonym rewrite",
            "population": (
                "latent label yes, aligned extract yes, shifted extract no. "
                "The Since chain is the shifted generation."
            ),
            "empty_extract": (
                "Gold-yes items whose aligned extract is yes and whose shifted "
                "extract is empty. They are counted in n_empty_extract and are not "
                "placed in the seven classes. n counts shifted extract no."
            ),
            "partition": "On each split, the seven first classes sum to n.",
            "has_query": "a shifted Since atom equals lat.query",
            "has_step0": "a shifted Since atom equals core_rules[0]'s conclusion",
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "yes_flip_trace.since_atoms. Order is the order of matches. "
                "A line containing 'cannot be proved' contributes no atom."
            ),
            "first_atom": "atoms[0] when Since atoms exist; otherwise the class is none",
            "classes": {
                "step0": "the first Since atom equals core_rules[0]'s conclusion",
                "query": "the first Since atom equals lat.query and is not the step-0 conclusion",
                "same_premise_decoy": (
                    "the first Since atom is the conclusion of some other_rules entry "
                    "whose premise frozenset equals core_rules[0], and it is not step0 or query"
                ),
                "later_core": (
                    "the first Since atom is a conclusion of core_rules[1:], "
                    "and it is not step0, query, or same_premise_decoy"
                ),
                "other_decoy": "the first Since atom is a conclusion of a remaining other_rules entry",
                "other": "the generation has a first Since atom and it matches none of the above",
                "none": "the generation has no Since sentence",
            },
            "priority": "yes_flip_first_since.first_since_class; earlier classes win",
            "premise_equality": "frozenset of the premise atoms; order is ignored",
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "inputs": [rel for _name, rel in CONDITIONS],
            "conditions": [name for name, _rel in CONDITIONS],
            "gates": (
                "pilot ours seen: n=7, has_query=0, has_step0=6, first step0=4, "
                "same_premise_decoy=3, and the other five first classes are 0. "
                "A mismatch stops the script before it writes this file."
            ),
            "loaded_model": False,
            "training": False,
            "not_read": "synonym generations.jsonl",
            "reading": (
                "Counts of Since atoms in saved reversed generations. "
                "These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
            ),
        },
        "gates": {
            "ok": gates["ok"],
            "pilot_ours_seen": gates["ours_seen"],
        },
        "conditions": conditions,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    print(f"gates_ok {gates['ok']}")
    for name, _rel in CONDITIONS:
        for split, _pool_name, _families in SPLITS:
            _print_split(name, split, conditions[name][split])


if __name__ == "__main__":
    main()
