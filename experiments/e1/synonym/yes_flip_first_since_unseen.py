"""First Since-atom source when a synonym answer flips gold yes to no.

Reads saved generations only. No training and no model load. The population
is an unseen graph, latent label yes, aligned extract yes, synonym extract
no. Since atoms use yes_flip_trace.since_atoms and keep that function's
order. Only the first atom is classified, with the same mutually exclusive
rules as yes_flip_first_since.py. Premise equality uses frozenset.

When that first atom is a same-premise decoy, two further headcounts ask
whether a later Since atom is the step-0 conclusion or the query. Those two
counts stay inside that subset and may overlap. This file does not say that
reasoning improved, and it does not treat any condition as a method win.

Writes only experiments/e1/synonym/yes_flip_first_since_unseen.json.
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
SPLIT = "unseen"
PREFERRED_FAMILIES = ("tree", "diamond")
CLASSES = (
    "step0",
    "query",
    "same_premise_decoy",
    "later_core",
    "other_decoy",
    "other",
    "none",
)
LATER_STEP0 = "same_premise_first_later_step0"
LATER_QUERY = "same_premise_first_later_query"
GATES = {
    "ours": {
        "n": 28,
        "aligned": {
            "step0": 13,
            "same_premise_decoy": 6,
            "later_core": 5,
            "other_decoy": 3,
            "other": 1,
        },
        "synonym": {
            "later_core": 13,
            "same_premise_decoy": 9,
            "other_decoy": 4,
            "step0": 2,
        },
        "later": {
            "aligned": {LATER_STEP0: 6, LATER_QUERY: 6},
            "synonym": {LATER_STEP0: 2, LATER_QUERY: 1},
        },
    },
    "b2": {
        "n": 25,
        "aligned": {"step0": 16, "later_core": 4, "same_premise_decoy": 5},
        "synonym": {
            "later_core": 15,
            "same_premise_decoy": 8,
            "other": 1,
            "other_decoy": 1,
        },
        "later": {
            "aligned": {LATER_STEP0: 5, LATER_QUERY: 5},
            "synonym": {LATER_STEP0: 0, LATER_QUERY: 0},
        },
    },
}
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_first_since_unseen.json"
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
    spec = importlib.util.spec_from_file_location("yes_flip_trace_unseen_first", path)
    if spec is None or spec.loader is None:
        _stop(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_since_class(atoms: list[str], lat) -> str:
    """Class of the first Since atom. Earlier classes win; the classes partition."""
    if not atoms:
        return "none"
    atom = str(atoms[0]).lower()
    step0 = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    if atom == step0:
        return "step0"
    if atom == query:
        return "query"
    step0_premises = frozenset(str(premise).lower() for premise in lat.core_rules[0][0])
    matched_other = False
    same = False
    for premises, conclusion in lat.other_rules:
        if str(conclusion).lower() != atom:
            continue
        matched_other = True
        if frozenset(str(premise).lower() for premise in premises) == step0_premises:
            same = True
    if same:
        return "same_premise_decoy"
    if any(str(conclusion).lower() == atom for _premises, conclusion in lat.core_rules[1:]):
        return "later_core"
    if matched_other:
        return "other_decoy"
    return "other"


def later_step0_query(atoms: list[str], lat, kind: str) -> tuple[bool, bool]:
    """Later step-0 and query hits, and only after a same-premise first atom."""
    if kind != "same_premise_decoy":
        return False, False
    step0 = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    rest = {str(atom).lower() for atom in atoms[1:]}
    return step0 in rest, query in rest


def _blank_classes() -> dict:
    return {name: 0 for name in CLASSES}


def _blank_side() -> dict:
    return {**_blank_classes(), LATER_STEP0: 0, LATER_QUERY: 0}


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


def _expected_side(partial: dict, n: int) -> dict:
    full = {name: partial.get(name, 0) for name in CLASSES}
    if sum(full.values()) != n:
        _stop(f"gate table does not partition n={n}: {full}")
    return full


def _self_check(since_atoms) -> None:
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
    if first_since_class([], chain) != "none":
        _stop("no Since sentence was not none")
    if first_since_class(["silver", "windy"], chain) != "step0":
        _stop("the step-0 conclusion was not step0")
    if first_since_class(["snowy"], chain) != "query":
        _stop("the query atom was not query")
    if first_since_class(["windy", "silver", "snowy"], chain) != "same_premise_decoy":
        _stop("a same-premise decoy was not same_premise_decoy")
    if first_since_class(["amber"], chain) != "other_decoy":
        _stop("a different-premise decoy was not other_decoy")
    if first_since_class(["invented"], chain) != "other":
        _stop("an atom outside the rules was not other")
    if later_step0_query(["silver", "snowy"], chain, "step0") != (False, False):
        _stop("later counts were taken outside the same-premise subset")
    if later_step0_query(["windy", "silver", "snowy"], chain, "same_premise_decoy") != (True, True):
        _stop("later step-0 and query were missed after a same-premise first atom")
    if later_step0_query(["windy", "snowy"], chain, "same_premise_decoy") != (False, True):
        _stop("a later query was missed when step 0 was absent")
    if later_step0_query(["windy", "windy"], chain, "same_premise_decoy") != (False, False):
        _stop("a repeated decoy atom was treated as step 0 or the query")

    query_is_step0 = SimpleNamespace(
        core_rules=((("vivid",), "snowy"),),
        other_rules=((("vivid",), "snowy"),),
        query="snowy",
        entity="Tina",
    )
    if first_since_class(["snowy"], query_is_step0) != "step0":
        _stop("a query that is the step-0 conclusion was not step0")

    swapped = SimpleNamespace(
        core_rules=((("a", "b"), "step0"), (("step0",), "windy")),
        other_rules=((("b", "a"), "windy"),),
        query="later",
        entity="Tina",
    )
    if first_since_class(["windy"], swapped) != "same_premise_decoy":
        _stop("frozenset equality missed swapped premises, or later core won")

    shared = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "muddy")),
        other_rules=((("otherprem",), "muddy"),),
        query="silver",
        entity="Tina",
    )
    if first_since_class(["muddy"], shared) != "later_core":
        _stop("a later core conclusion lost to a different-premise decoy")
    if first_since_class(["silver"], shared) != "step0":
        _stop("the step-0 conclusion lost to the query")


def _family_order(present: set[str]) -> tuple[str, ...]:
    ordered = [name for name in PREFERRED_FAMILIES if name in present]
    extras = sorted(name for name in present if name not in PREFERRED_FAMILIES)
    return tuple(ordered + extras)


def _index_unseen(pools: dict) -> tuple[dict, tuple[str, ...]]:
    unseen = {}
    for lat in pools["test_unseen"]:
        digest = graph_hash(lat)
        if digest in unseen:
            _stop(f"duplicate unseen graph hash {digest}")
        if not lat.core_rules:
            _stop(f"{digest} has no core rules")
        unseen[digest] = lat
    if len(unseen) != 200:
        _stop(f"unseen pool has {len(unseen)} graphs")
    families = _family_order({lat.family for lat in unseen.values()})
    if not families:
        _stop("unseen pool has no families")
    return unseen, families


def _load_condition(name: str, gen_rel: str, logic_rel: str, unseen: dict) -> dict:
    gen_path = PROJECT / gen_rel
    logic_path = PROJECT / logic_rel
    if not gen_path.is_file() or not logic_path.is_file():
        _stop(f"missing input for {name}: {gen_rel} or {logic_rel}")
    synonym = read_jsonl(gen_path)
    logic = read_jsonl(logic_path)
    missing_synonym = []
    missing_aligned = []
    for digest in unseen:
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


def _check_side(label: str, side: dict, n: int) -> None:
    total = sum(side[name] for name in CLASSES)
    if total != n:
        _stop(f"{label} classes {total} != n {n} { {name: side[name] for name in CLASSES} }")
    if side[LATER_STEP0] > side["same_premise_decoy"]:
        _stop(f"{label} later step0 exceeds same_premise_decoy")
    if side[LATER_QUERY] > side["same_premise_decoy"]:
        _stop(f"{label} later query exceeds same_premise_decoy")


def _score_condition(bundle: dict, unseen: dict, families: tuple[str, ...], since_atoms) -> dict:
    n = 0
    sides = {"aligned": _blank_side(), "synonym": _blank_side()}
    family_blocks = {
        family: {"n": 0, "aligned": _blank_classes(), "synonym": _blank_classes()}
        for family in families
    }
    for digest, lat in unseen.items():
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
        n += 1
        family = lat.family
        if family not in family_blocks:
            _stop(f"{digest} family {family} is outside the unseen pool families")
        family_blocks[family]["n"] += 1
        texts = {"aligned": aligned_text, "synonym": synonym_text}
        for side, text in texts.items():
            atoms = since_atoms(text, lat.entity)
            kind = first_since_class(atoms, lat)
            if kind not in CLASSES:
                _stop(f"{bundle['name']} {digest} {side} class {kind} is unknown")
            sides[side][kind] += 1
            family_blocks[family][side][kind] += 1
            has_step0, has_query = later_step0_query(atoms, lat, kind)
            if has_step0:
                sides[side][LATER_STEP0] += 1
            if has_query:
                sides[side][LATER_QUERY] += 1
    for side, block in sides.items():
        _check_side(f"{bundle['name']}.{side}", block, n)
    if sum(family_blocks[family]["n"] for family in families) != n:
        _stop(f"{bundle['name']} family n does not sum to {n}")
    for family in families:
        fam_n = family_blocks[family]["n"]
        for side in ("aligned", "synonym"):
            total = sum(family_blocks[family][side][name] for name in CLASSES)
            if total != fam_n:
                _stop(f"{bundle['name']}.{family}.{side} classes {total} != n {fam_n}")
    for side in ("aligned", "synonym"):
        for name in CLASSES:
            summed = sum(family_blocks[family][side][name] for family in families)
            if summed != sides[side][name]:
                _stop(f"{bundle['name']}.{side} family sum {name} {summed} != {sides[side][name]}")
    return {
        "n": n,
        "aligned": sides["aligned"],
        "synonym": sides["synonym"],
        "families": family_blocks,
    }


def _gate_report(conditions: dict) -> dict:
    report = {}
    ok = True
    for name, expected in GATES.items():
        block = conditions[name]
        cell = {"n": block["n"], "expected_n": expected["n"], "ok": block["n"] == expected["n"]}
        for side in ("aligned", "synonym"):
            observed = {key: block[side][key] for key in CLASSES}
            target = _expected_side(expected[side], expected["n"])
            side_ok = observed == target
            later_observed = {
                LATER_STEP0: block[side][LATER_STEP0],
                LATER_QUERY: block[side][LATER_QUERY],
            }
            later_expected = {
                LATER_STEP0: expected["later"][side][LATER_STEP0],
                LATER_QUERY: expected["later"][side][LATER_QUERY],
            }
            later_ok = later_observed == later_expected
            cell[side] = {
                "observed": observed,
                "expected": target,
                "ok": side_ok,
                "later_observed": later_observed,
                "later_expected": later_expected,
                "later_ok": later_ok,
            }
            cell["ok"] = cell["ok"] and side_ok and later_ok
        ok = ok and cell["ok"]
        report[name] = cell
    report["ok"] = ok
    return report


def _public_side(side: dict) -> dict:
    return {
        **{name: side[name] for name in CLASSES},
        LATER_STEP0: side[LATER_STEP0],
        LATER_QUERY: side[LATER_QUERY],
    }


def _public_condition(block: dict) -> dict:
    return {
        "n": block["n"],
        "aligned": _public_side(block["aligned"]),
        "synonym": _public_side(block["synonym"]),
        "families": {
            family: {
                "n": fam["n"],
                "aligned": {name: fam["aligned"][name] for name in CLASSES},
                "synonym": {name: fam["synonym"][name] for name in CLASSES},
            }
            for family, fam in block["families"].items()
        },
    }


def _nonzero(side: dict) -> str:
    parts = [f"{name}={side[name]}" for name in CLASSES if side[name]]
    if not parts:
        return "none_of_seven"
    return " ".join(parts)


def _print_side(prefix: str, side: dict) -> None:
    classes = " ".join(f"{name}={side[name]}" for name in CLASSES)
    print(
        f"{prefix} {classes} "
        f"later_step0={side[LATER_STEP0]} later_query={side[LATER_QUERY]}"
    )
    print(
        f"{prefix} nonzero {_nonzero(side)} "
        f"later_step0={side[LATER_STEP0]} later_query={side[LATER_QUERY]}"
    )


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_trace()
    _self_check(trace.since_atoms)
    if len(trace.CONDITIONS) != 9:
        _stop(f"expected 9 conditions, found {len(trace.CONDITIONS)}")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    unseen, families = _index_unseen(pools)
    conditions = {}
    for name, gen_rel, logic_rel in trace.CONDITIONS:
        bundle = _load_condition(name, gen_rel, logic_rel, unseen)
        conditions[name] = _score_condition(bundle, unseen, families, trace.since_atoms)
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    gates = _gate_report(conditions)
    public = {name: _public_condition(block) for name, block in conditions.items()}
    for name in ("ours", "b2"):
        cell = gates[name]
        print(
            f"gate {name} ok={cell['ok']} n={cell['n']} expected_n={cell['expected_n']} "
            f"aligned_ok={cell['aligned']['ok']} synonym_ok={cell['synonym']['ok']} "
            f"aligned_later_ok={cell['aligned']['later_ok']} "
            f"synonym_later_ok={cell['synonym']['later_ok']}"
        )
    if not gates["ok"]:
        print("STOP gate mismatch; counts were not adjusted and the json was not written")
        for name in ("ours", "b2"):
            for side in ("aligned", "synonym"):
                _print_side(f"observed {name} {side}", public[name][side])
                print(f"expected {name} {side} {gates[name][side]['expected']}")
                print(f"expected {name} {side} later {gates[name][side]['later_expected']}")
        raise SystemExit(1)
    payload = {
        "config": {
            "seed": SEED,
            "split": SPLIT,
            "pool": "test_unseen from build_pools(seed=0)",
            "population": (
                "unseen graph, latent label yes, aligned extract yes, synonym extract no"
            ),
            "aligned_generation": (
                "unseen_aligned|{graph_hash} in each condition's logic.jsonl. "
                "The row is not regenerated."
            ),
            "synonym_generation": (
                "unseen_synonym|{graph_hash} in each condition's generations.jsonl"
            ),
            "partition": (
                "On each side, the seven classes sum to n. Family blocks repeat "
                "those seven classes and sum to the condition. The two later "
                "counts are not part of that partition."
            ),
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "yes_flip_trace.since_atoms. Order is the order of matches. "
                "A line containing 'cannot be proved' contributes no atom."
            ),
            "first_atom": "atoms[0] when Since atoms exist; otherwise the class is none",
            "classes": {
                "step0": "the first Since atom equals core_rules[0]'s conclusion",
                "query": (
                    "the first Since atom equals lat.query and is not the step-0 conclusion"
                ),
                "same_premise_decoy": (
                    "the first Since atom is the conclusion of some other_rules entry "
                    "whose premise frozenset equals core_rules[0], and it is not step0 or query"
                ),
                "later_core": (
                    "the first Since atom is a conclusion of core_rules[1:], "
                    "and it is not step0, query, or same_premise_decoy"
                ),
                "other_decoy": (
                    "the first Since atom is a conclusion of a remaining other_rules entry"
                ),
                "other": "the generation has a first Since atom and it matches none of the above",
                "none": "the generation has no Since sentence",
            },
            "premise_equality": "frozenset of the premise atoms; order is ignored",
            "same_premise_first_later_step0": (
                "Among items whose first Since atom is same_premise_decoy, a later "
                "Since atom equals core_rules[0]'s conclusion. The first atom is not "
                "included. This count is not one of the seven classes."
            ),
            "same_premise_first_later_query": (
                "Among items whose first Since atom is same_premise_decoy, a later "
                "Since atom equals lat.query. The first atom is not included. "
                "This count may overlap later step 0 and is not one of the seven classes."
            ),
            "families": list(families),
            "family_rule": (
                "Families are those present in test_unseen. tree and diamond are "
                "written when present, in that order. A family absent from the pool "
                "is omitted. Any other family present in the pool is written under "
                "its own name. Each family records the seven classes on both sides."
            ),
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "conditions": [name for name, _gen, _logic in trace.CONDITIONS],
            "gates": (
                "pilot ours n=28. aligned first atom: step0 13, same_premise_decoy 6, "
                "later_core 5, other_decoy 3, other 1. synonym first atom: later_core 13, "
                "same_premise_decoy 9, other_decoy 4, step0 2. After a same-premise first "
                "atom, aligned later step0 6 and query 6; synonym later step0 2 and query 1. "
                "pilot b2 n=25. aligned first atom: step0 16, later_core 4, "
                "same_premise_decoy 5. synonym first atom: later_core 15, "
                "same_premise_decoy 8, other 1, other_decoy 1. After a same-premise first "
                "atom, aligned later step0 5 and query 5; synonym later step0 0 and query 0. "
                "Unlisted classes are 0 because those cells sum to n. "
                "A mismatch stops the script before it writes this file."
            ),
            "loaded_model": False,
            "training": False,
            "reading": (
                "Counts of which rule the first Since atom matches in saved generations "
                "on unseen graphs. These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
            ),
        },
        "gates": {
            "ok": gates["ok"],
            "pilot_ours": gates["ours"],
            "pilot_b2": gates["b2"],
        },
        "conditions": public,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    print(f"families {' '.join(families)}")
    print(f"gates_ok {gates['ok']}")
    for name, block in public.items():
        print(f"{name} n={block['n']}")
        _print_side("  aligned", block["aligned"])
        _print_side("  synonym", block["synonym"])
        for family, fam in block["families"].items():
            aligned = " ".join(f"{key}={fam['aligned'][key]}" for key in CLASSES)
            synonym = " ".join(f"{key}={fam['synonym'][key]}" for key in CLASSES)
            print(f"  {family} n={fam['n']} aligned({aligned}) synonym({synonym})")


if __name__ == "__main__":
    main()
