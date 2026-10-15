"""Aligned-order Since trace for items whose synonym answer flips gold yes to no.

Reads saved generations only. No training and no model load. The population is
the same as yes_flip_trace: a seen graph, latent label yes, aligned extract
yes, synonym extract no. Since atoms on the aligned generation use
yes_flip_trace.since_atoms. A same-premise decoy atom is a conclusion of some
other_rules entry whose premise frozenset equals core_rules[0].

The listed aligned counts are headcounts and may overlap. The synonym-text
"is always" count is separate and does not move those counts. This file does
not say that reasoning improved, and it does not treat any condition as a
method win.

Writes only experiments/e1/synonym/yes_flip_aligned_trace.json.
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
SPLIT = "seen"
FAMILIES = ("chain", "conj_fact")
COUNTS = (
    "n",
    "aligned_since_step0",
    "aligned_since_query",
    "aligned_since_same_premise_decoy",
    "aligned_no_since",
    "aligned_step0_and_query",
    "aligned_decoy_without_step0",
)
IS_ALWAYS = "is always"
OURS_OTHER = 42
OURS_SAME = 38
IS_ALWAYS_CELLS = (
    ("b0", 13, 12),
    ("mix15_b2", 44, 1),
    ("b2", None, 0),
    ("mix15_ours", None, 0),
)
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_aligned_trace.json"
TRACE_JSON = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_trace.json"
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
    "synonym_burden_contrast.json",
    "burden_contrast.json",
    "report.md",
}


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _load_module(name: str, filename: str):
    path = PROJECT / "experiments" / "e1" / "synonym" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _stop(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_premise_conclusions(lat) -> set[str]:
    """Conclusions of other_rules whose premise set equals core_rules[0]."""
    step0 = frozenset(str(premise).lower() for premise in lat.core_rules[0][0])
    found = set()
    for premises, conclusion in lat.other_rules:
        premises_key = frozenset(str(premise).lower() for premise in premises)
        if premises_key == step0:
            found.add(str(conclusion).lower())
    return found


def aligned_flags(atoms: list[str], step0: str, query: str, decoys: set[str]) -> dict:
    found = {atom.lower() for atom in atoms}
    has_step0 = step0 in found
    has_query = query in found
    has_decoy = bool(found & decoys)
    return {
        "aligned_since_step0": has_step0,
        "aligned_since_query": has_query,
        "aligned_since_same_premise_decoy": has_decoy,
        "aligned_no_since": not atoms,
        "aligned_step0_and_query": has_step0 and has_query,
        "aligned_decoy_without_step0": has_decoy and not has_step0,
    }


def _blank() -> dict:
    return {name: 0 for name in COUNTS}


def _add(block: dict, flags: dict) -> None:
    block["n"] += 1
    for name, hit in flags.items():
        if hit:
            block[name] += 1


def _check_block(label: str, block: dict) -> None:
    n = block["n"]
    for name in COUNTS:
        if block[name] > n:
            _stop(f"{label} {name} {block[name]} exceeds n {n}")
    if block["aligned_step0_and_query"] > block["aligned_since_step0"]:
        _stop(f"{label} step0-and-query exceeds step0")
    if block["aligned_step0_and_query"] > block["aligned_since_query"]:
        _stop(f"{label} step0-and-query exceeds query")
    if block["aligned_decoy_without_step0"] > block["aligned_since_same_premise_decoy"]:
        _stop(f"{label} decoy-without-step0 exceeds decoy")
    disjoint = (
        ("aligned_no_since", "aligned_since_step0"),
        ("aligned_no_since", "aligned_since_query"),
        ("aligned_no_since", "aligned_since_same_premise_decoy"),
        ("aligned_no_since", "aligned_step0_and_query"),
        ("aligned_decoy_without_step0", "aligned_since_step0"),
        ("aligned_decoy_without_step0", "aligned_step0_and_query"),
    )
    for left, right in disjoint:
        if block[left] + block[right] > n:
            _stop(f"{label} {left} and {right} overlap")


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
    return resolved


def _self_check(since_atoms, item_source) -> None:
    sample = (
        "Known facts: Tina is vivid.\n"
        "Since Tina is vivid, Tina is windy.\n"
        "Nothing else can be derived, so Tina is snowy cannot be proved.\n"
        "Answer: no"
    )
    if since_atoms(sample, "Tina") != ["windy"]:
        _stop("regex self-check did not keep the intermediate atom only")
    proved = "Since Tina is vivid, Tina is snowy cannot be proved."
    if since_atoms(proved, "Tina"):
        _stop("regex self-check counted a cannot-be-proved line")
    folded = "SINCE tina is vivid, tina is Windy."
    if since_atoms(folded, "Tina") != ["windy"]:
        _stop("regex self-check missed a case-insensitive Since sentence")
    copied = "Since Tina is vivid, Tina is always windy."
    if since_atoms(copied, "Tina") != ["always"]:
        _stop("regex self-check did not capture always from a copied synonym sentence")
    if IS_ALWAYS not in copied:
        _stop("is-always self-check missed the copied synonym sentence")

    lat = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "snowy")),
        other_rules=((("vivid",), "windy"), (("silver",), "muddy"), (("b", "a"), "amber")),
        facts=("vivid",),
        query="snowy",
        entity="Tina",
        family="chain",
        label="yes",
    )
    decoys = same_premise_conclusions(lat)
    if decoys != {"windy"}:
        _stop("same-premise conclusions included a different premise or a core conclusion")
    if item_source(["windy"], lat, "frozenset") != "same_premise_decoy":
        _stop("gate source class missed a same-premise decoy")
    if item_source(["muddy"], lat, "frozenset") == "same_premise_decoy":
        _stop("gate source class treated a different premise as the same set")
    swapped = SimpleNamespace(
        core_rules=((("a", "b"), "step0"),),
        other_rules=((("b", "a"), "windy"),),
        facts=(),
        query="step0",
        entity="Tina",
    )
    if same_premise_conclusions(swapped) != {"windy"}:
        _stop("frozenset equality missed swapped premises")
    if item_source(["windy"], swapped, "frozenset") != "same_premise_decoy":
        _stop("gate source class missed swapped premises")

    both = (
        "Since Tina is vivid, Tina is silver.\n"
        "Since Tina is silver, Tina is snowy.\n"
        "Since Tina is vivid, Tina is windy.\n"
    )
    flags = aligned_flags(since_atoms(both, "Tina"), "silver", "snowy", decoys)
    if flags != {
        "aligned_since_step0": True,
        "aligned_since_query": True,
        "aligned_since_same_premise_decoy": True,
        "aligned_no_since": False,
        "aligned_step0_and_query": True,
        "aligned_decoy_without_step0": False,
    }:
        _stop(f"overlap flags failed for a chain that also names a decoy: {flags}")
    decoy_only = "Since Tina is vivid, Tina is windy.\n"
    flags = aligned_flags(since_atoms(decoy_only, "Tina"), "silver", "snowy", decoys)
    if not (
        flags["aligned_decoy_without_step0"]
        and flags["aligned_since_same_premise_decoy"]
        and not flags["aligned_since_step0"]
        and not flags["aligned_no_since"]
    ):
        _stop("decoy-without-step0 flag failed")
    listed = "1. Tina is always windy whenever Tina is vivid.\nAnswer: no"
    if since_atoms(listed, "Tina") or IS_ALWAYS not in listed:
        _stop("a numbered is-always line was treated as a Since sentence")
    flags = aligned_flags(since_atoms(listed, "Tina"), "silver", "snowy", decoys)
    if not flags["aligned_no_since"] or any(
        flags[name]
        for name in (
            "aligned_since_step0",
            "aligned_since_query",
            "aligned_since_same_premise_decoy",
            "aligned_step0_and_query",
            "aligned_decoy_without_step0",
        )
    ):
        _stop("is always changed a Since flag")


def _index_seen(pools: dict) -> dict:
    seen = {}
    for lat in pools["test_seen"]:
        digest = graph_hash(lat)
        if digest in seen:
            _stop(f"duplicate seen graph hash {digest}")
        if lat.family not in FAMILIES:
            _stop(f"{digest} family {lat.family} is outside the seen families")
        if not lat.core_rules:
            _stop(f"{digest} has no core rules")
        seen[digest] = lat
    if len(seen) != 200:
        _stop(f"seen pool has {len(seen)} graphs")
    return seen


def _load_condition(name: str, gen_rel: str, logic_rel: str, seen: dict) -> dict:
    gen_path = PROJECT / gen_rel
    logic_path = PROJECT / logic_rel
    if not gen_path.is_file() or not logic_path.is_file():
        _stop(f"missing input for {name}: {gen_rel} or {logic_rel}")
    synonym = read_jsonl(gen_path)
    logic = read_jsonl(logic_path)
    missing_synonym = []
    missing_aligned = []
    for digest in seen:
        synonym_uid = f"{SPLIT}_synonym|{digest}"
        aligned_uid = f"{SPLIT}_aligned|{digest}"
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


def _score_condition(bundle: dict, seen: dict, since_atoms, bucket_of, item_source) -> dict:
    total = _blank()
    families = {family: _blank() for family in FAMILIES}
    is_always = 0
    family_always = {family: 0 for family in FAMILIES}
    ours_gate = None
    if bundle["name"] == "ours":
        ours_gate = {"other_derived": 0, "same_premise_decoy": 0, "sources": {}}
    for digest, lat in seen.items():
        if lat.label != "yes":
            continue
        aligned_uid = f"{SPLIT}_aligned|{digest}"
        synonym_uid = f"{SPLIT}_synonym|{digest}"
        aligned_text = _generation(bundle["logic"][aligned_uid], aligned_uid)
        synonym_text = _generation(bundle["synonym"][synonym_uid], synonym_uid)
        if extract_yes_no(aligned_text) != "yes":
            continue
        if extract_yes_no(synonym_text) != "no":
            continue
        step0 = str(lat.core_rules[0][1]).lower()
        query = str(lat.query).lower()
        atoms = since_atoms(aligned_text, lat.entity)
        flags = aligned_flags(atoms, step0, query, same_premise_conclusions(lat))
        family = lat.family
        _add(total, flags)
        _add(families[family], flags)
        if IS_ALWAYS in synonym_text:
            is_always += 1
            family_always[family] += 1
        if ours_gate is not None:
            synonym_atoms = since_atoms(synonym_text, lat.entity)
            bucket = bucket_of(synonym_atoms, step0, query)
            if bucket == "other_derived":
                source = item_source(synonym_atoms, lat, "frozenset")
                ours_gate["other_derived"] += 1
                ours_gate["sources"][source] = ours_gate["sources"].get(source, 0) + 1
                if source == "same_premise_decoy":
                    ours_gate["same_premise_decoy"] += 1
    _check_block(bundle["name"], total)
    for family in FAMILIES:
        _check_block(f"{bundle['name']}.{family}", families[family])
    for name in COUNTS:
        summed = sum(families[family][name] for family in FAMILIES)
        if summed != total[name]:
            _stop(f"{bundle['name']} family sum {name} {summed} != {total[name]}")
    if sum(family_always.values()) != is_always:
        _stop(f"{bundle['name']} family is-always sum failed")
    if is_always > total["n"]:
        _stop(f"{bundle['name']} is always {is_always} exceeds n {total['n']}")
    block = {
        **total,
        "synonym_is_always": is_always,
        "families": {
            family: {**families[family], "synonym_is_always": family_always[family]}
            for family in FAMILIES
        },
    }
    return block, ours_gate


def _check_population(conditions: dict) -> None:
    if not TRACE_JSON.is_file():
        _stop(f"missing {TRACE_JSON}")
    previous = json.loads(TRACE_JSON.read_text(encoding="utf-8"))
    for name, block in conditions.items():
        old = previous["conditions"][name]
        if block["n"] != old["n"]:
            _stop(f"{name} n {block['n']} != yes_flip_trace {old['n']}")
        for family in FAMILIES:
            old_n = old["families"][family]["n"]
            new_n = block["families"][family]["n"]
            if new_n != old_n:
                _stop(f"{name}.{family} n {new_n} != yes_flip_trace {old_n}")


def _is_always_gate(conditions: dict) -> dict:
    cells = []
    ok = True
    for name, expected_n, expected_count in IS_ALWAYS_CELLS:
        block = conditions[name]
        observed_n = block["n"]
        observed_count = block["synonym_is_always"]
        cell_ok = observed_count == expected_count and (
            expected_n is None or observed_n == expected_n
        )
        ok = ok and cell_ok
        cells.append({
            "condition": name,
            "expected_n": expected_n,
            "expected_synonym_is_always": expected_count,
            "n": observed_n,
            "synonym_is_always": observed_count,
            "ok": cell_ok,
        })
    return {
        "match": "exact substring 'is always' in the synonym generation; Since flags are unchanged",
        "cells": cells,
        "ok": ok,
    }


def _ours_gate(ours_gate: dict) -> dict:
    other = ours_gate["other_derived"]
    same = ours_gate["same_premise_decoy"]
    return {
        "condition": "ours",
        "side": "synonym",
        "equality": "frozenset",
        "other_derived": other,
        "same_premise_decoy": same,
        "expected_other_derived": OURS_OTHER,
        "expected_same_premise_decoy": OURS_SAME,
        "sources": ours_gate["sources"],
        "ok": other == OURS_OTHER and same == OURS_SAME,
    }


def _public_block(block: dict) -> dict:
    return {
        **{name: block[name] for name in COUNTS},
        "synonym_is_always": block["synonym_is_always"],
        "families": {
            family: {
                **{name: block["families"][family][name] for name in COUNTS},
                "synonym_is_always": block["families"][family]["synonym_is_always"],
            }
            for family in FAMILIES
        },
    }


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_module("yes_flip_trace_aligned", "yes_flip_trace.py")
    rules = _load_module("yes_flip_rule_source_aligned", "yes_flip_rule_source.py")
    _self_check(trace.since_atoms, rules.item_source)
    if tuple(trace.FAMILIES) != FAMILIES:
        _stop("seen families drifted from yes_flip_trace")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    seen = _index_seen(pools)
    conditions = {}
    ours_gate = None
    for name, gen_rel, logic_rel in trace.CONDITIONS:
        bundle = _load_condition(name, gen_rel, logic_rel, seen)
        block, gate = _score_condition(
            bundle, seen, trace.since_atoms, trace.bucket_of, rules.item_source,
        )
        conditions[name] = block
        if gate is not None:
            ours_gate = gate
    if ours_gate is None:
        _stop("pilot ours was not scored")
    _check_population(conditions)
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    public = {name: _public_block(block) for name, block in conditions.items()}
    gates = {
        "ours_synonym": _ours_gate(ours_gate),
        "is_always": _is_always_gate(public),
    }
    gates["ok"] = gates["ours_synonym"]["ok"] and gates["is_always"]["ok"]
    payload = {
        "config": {
            "seed": SEED,
            "split": SPLIT,
            "pool": "test_seen from build_pools(seed=0)",
            "population": (
                "seen graph, latent label yes, aligned extract yes, synonym extract no"
            ),
            "aligned_generation": (
                "seen_aligned|{graph_hash} in each condition's logic.jsonl. "
                "The row is not regenerated."
            ),
            "synonym_generation": "seen_synonym|{graph_hash} in each condition's generations.jsonl",
            "overlap": (
                "The aligned counts are item headcounts and may overlap. "
                "They are not a partition of n."
            ),
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "yes_flip_trace.since_atoms on the aligned generation. "
                "A line containing 'cannot be proved' contributes no derived atom."
            ),
            "counts": {
                "n": "items in the population",
                "aligned_since_step0": "a Since sentence derives core_rules[0]'s conclusion",
                "aligned_since_query": "a Since sentence derives the query atom",
                "aligned_since_same_premise_decoy": (
                    "a Since atom is the conclusion of some other_rules entry whose "
                    "premise frozenset equals core_rules[0]"
                ),
                "aligned_no_since": "the aligned generation has no Since sentence",
                "aligned_step0_and_query": "Since sentences derive both the step-0 conclusion and the query",
                "aligned_decoy_without_step0": (
                    "a Since atom is a same-premise decoy conclusion and none derives "
                    "the step-0 conclusion"
                ),
            },
            "synonym_is_always": (
                "The synonym generation contains the exact substring 'is always'. "
                "This count does not change any Since flag. Family blocks record the "
                "same substring inside that family."
            ),
            "premise_equality": "frozenset of the premise atoms; order is ignored",
            "ours_synonym_gate": (
                "On pilot ours, synonym-side other_derived and the frozenset "
                "same_premise_decoy source class. Expected 42 and 38."
            ),
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "loaded_model": False,
            "training": False,
            "reading": (
                "Counts of Since sentences in saved aligned generations for items "
                "whose synonym answer flipped from yes to no. These counts are not "
                "evidence that reasoning improved, and they are not a win for any "
                "training condition."
            ),
        },
        "gates": gates,
        "conditions": public,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    print(
        f"gates_ok {gates['ok']} ours_synonym_ok {gates['ours_synonym']['ok']} "
        f"is_always_ok {gates['is_always']['ok']}"
    )
    ours = gates["ours_synonym"]
    print(
        "ours synonym "
        f"other_derived={ours['other_derived']} "
        f"same_premise_decoy={ours['same_premise_decoy']} "
        f"sources={ours['sources']}"
    )
    for cell in gates["is_always"]["cells"]:
        print(
            f"is_always {cell['condition']} {cell['synonym_is_always']}/{cell['n']} "
            f"expected_count={cell['expected_synonym_is_always']} "
            f"expected_n={cell['expected_n']} ok={cell['ok']}"
        )
    for name, block in public.items():
        print(
            f"{name} n={block['n']} "
            + " ".join(f"{count}={block[count]}" for count in COUNTS if count != "n")
            + f" synonym_is_always={block['synonym_is_always']}"
        )
        for family in FAMILIES:
            fam = block["families"][family]
            print(
                f"  {family} n={fam['n']} "
                + " ".join(f"{count}={fam[count]}" for count in COUNTS if count != "n")
                + f" synonym_is_always={fam['synonym_is_always']}"
            )
    if not gates["ok"]:
        print("STOP gate mismatch; counts were not adjusted")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
