"""Since atoms when an unseen synonym answer stays gold yes.

Reads saved generations only. No training and no model load. The population
is an unseen graph, latent label yes, aligned extract yes, and synonym
extract yes. These items were answered yes. They are not the flip population.

Since atoms are yes_flip_trace.since_atoms and keep that function's order.
The first atom uses yes_flip_first_since.first_since_class. Priority is
step0, then query, then same_premise_decoy, then later_core. The collapse
keeps step0, same_premise_decoy, later_core, and none, and puts query,
other_decoy, and other into first_other. has_query and has_step0 ask whether
any synonym Since atom is the query or the step-0 conclusion. Those two
counts may overlap and are not part of the five-way first-atom partition.

These counts are not evidence that reasoning improved, and they are not a
win for any training condition.

Writes only experiments/e1/synonym/yes_correct_since_unseen.json.
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
FAMILIES = ("tree", "diamond")
COUNT_KEYS = (
    "n",
    "has_query",
    "has_step0",
    "first_step0",
    "first_same_premise_decoy",
    "first_later_core",
    "first_other",
    "first_none",
)
FIRST_KEYS = (
    "first_step0",
    "first_same_premise_decoy",
    "first_later_core",
    "first_other",
    "first_none",
)
GATE_KEYS = (
    "n",
    "has_query",
    "has_step0",
)
GATES = {
    "ours": {
        "n": 21,
        "has_query": 21,
        "has_step0": 15,
    }
}
COLLAPSE = {
    "step0": "first_step0",
    "query": "first_other",
    "same_premise_decoy": "first_same_premise_decoy",
    "later_core": "first_later_core",
    "other_decoy": "first_other",
    "other": "first_other",
    "none": "first_none",
}
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_correct_since_unseen.json"
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


def _load_module(filename: str, module_name: str):
    path = PROJECT / "experiments" / "e1" / "synonym" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _stop(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blank() -> dict:
    return {key: 0 for key in COUNT_KEYS}


def collapse_first(kind: str) -> str:
    """Map the seven-class first atom onto the five classes in this file."""
    if kind not in COLLAPSE:
        _stop(f"unknown first-atom class {kind}")
    return COLLAPSE[kind]


def item_flags(atoms: list[str], lat, first_since_class) -> dict:
    """Presence of query and step 0, plus the collapsed class of atoms[0]."""
    kind = collapse_first(first_since_class(atoms, lat))
    step0 = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    found = {str(atom).lower() for atom in atoms}
    return {
        "has_query": query in found,
        "has_step0": step0 in found,
        "first": kind,
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
    if resolved.name in BANNED_NAMES or "rescore" in resolved.parts or "paper" in resolved.parts:
        _stop(f"refusing to write {resolved}")
    return resolved


def _self_check(since_atoms, first_since_class) -> None:
    sample = (
        "Known facts: Tina is vivid.\n"
        "Since Tina is vivid, Tina is windy.\n"
        "Nothing else can be derived, so Tina is snowy cannot be proved.\n"
        "Answer: yes"
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
    empty = item_flags([], chain, first_since_class)
    if empty["first"] != "first_none" or empty["has_query"] or empty["has_step0"]:
        _stop("no Since sentence was not first_none")
    step0_only = item_flags(["silver", "windy"], chain, first_since_class)
    if step0_only["first"] != "first_step0" or not step0_only["has_step0"] or step0_only["has_query"]:
        _stop("the step-0 conclusion was not first_step0")
    query_first = item_flags(["snowy"], chain, first_since_class)
    if query_first["first"] != "first_other" or not query_first["has_query"] or query_first["has_step0"]:
        _stop("a query-first atom was not first_other")
    decoy = item_flags(["windy", "silver", "snowy"], chain, first_since_class)
    if (
        decoy["first"] != "first_same_premise_decoy"
        or not decoy["has_step0"]
        or not decoy["has_query"]
    ):
        _stop("a same-premise decoy was not first_same_premise_decoy")
    if item_flags(["amber"], chain, first_since_class)["first"] != "first_other":
        _stop("a different-premise decoy was not first_other")
    if item_flags(["invented"], chain, first_since_class)["first"] != "first_other":
        _stop("an atom outside the rules was not first_other")
    later = item_flags(["muddy"], chain, first_since_class)
    if later["first"] != "first_later_core" or later["has_query"] or later["has_step0"]:
        _stop("a later core conclusion was not first_later_core")

    query_is_step0 = SimpleNamespace(
        core_rules=((("vivid",), "snowy"),),
        other_rules=((("vivid",), "snowy"),),
        query="snowy",
        entity="Tina",
    )
    both = item_flags(["snowy"], query_is_step0, first_since_class)
    if both["first"] != "first_step0" or not both["has_query"] or not both["has_step0"]:
        _stop("a query that is the step-0 conclusion was not first_step0")

    query_decoy = SimpleNamespace(
        core_rules=((("vivid",), "silver"),),
        other_rules=((("vivid",), "snowy"),),
        query="snowy",
        entity="Tina",
    )
    hidden = item_flags(["snowy"], query_decoy, first_since_class)
    if hidden["first"] != "first_other" or not hidden["has_query"] or hidden["has_step0"]:
        _stop("a query with the step-0 premises was counted as a same-premise decoy")

    query_later = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "muddy")),
        other_rules=(),
        query="muddy",
        entity="Tina",
    )
    claimed = item_flags(["muddy"], query_later, first_since_class)
    if claimed["first"] != "first_other" or not claimed["has_query"] or claimed["has_step0"]:
        _stop("a query that is also a later core conclusion was not taken by query")

    same_later = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "windy")),
        other_rules=((("vivid",), "windy"),),
        query="snowy",
        entity="Tina",
    )
    if item_flags(["windy"], same_later, first_since_class)["first"] != "first_same_premise_decoy":
        _stop("a same-premise decoy lost to a later core conclusion")

    swapped = SimpleNamespace(
        core_rules=((("a", "b"), "step0"), (("step0",), "windy")),
        other_rules=((("b", "a"), "windy"),),
        query="later",
        entity="Tina",
    )
    if item_flags(["windy"], swapped, first_since_class)["first"] != "first_same_premise_decoy":
        _stop("frozenset equality missed swapped premises")


def _index_unseen(pools: dict) -> dict:
    unseen = {}
    for lat in pools["test_unseen"]:
        digest = graph_hash(lat)
        if digest in unseen:
            _stop(f"duplicate unseen graph hash {digest}")
        if lat.family not in FAMILIES:
            _stop(f"{digest} family {lat.family} is outside tree and diamond")
        if not lat.core_rules:
            _stop(f"{digest} has no core rules")
        unseen[digest] = lat
    if len(unseen) != 200:
        _stop(f"unseen pool has {len(unseen)} graphs")
    present = {lat.family for lat in unseen.values()}
    if present != set(FAMILIES):
        _stop(f"unseen families {sorted(present)} are not tree and diamond")
    return unseen


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
        synonym_uid = f"unseen_synonym|{digest}"
        aligned_uid = f"unseen_aligned|{digest}"
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


def _check_block(label: str, block: dict) -> None:
    total = sum(block[name] for name in FIRST_KEYS)
    if total != block["n"]:
        _stop(
            f"{label} first classes {total} != n {block['n']} "
            f"{ {name: block[name] for name in FIRST_KEYS} }"
        )
    for key in ("has_query", "has_step0"):
        if block[key] > block["n"]:
            _stop(f"{label} {key} {block[key]} exceeds n {block['n']}")


def _score_condition(bundle: dict, unseen: dict, since_atoms, first_since_class) -> dict:
    block = _blank()
    families = {family: _blank() for family in FAMILIES}
    for digest, lat in unseen.items():
        if lat.label != "yes":
            continue
        aligned_uid = f"unseen_aligned|{digest}"
        synonym_uid = f"unseen_synonym|{digest}"
        aligned_text = _generation(bundle["logic"][aligned_uid], aligned_uid)
        synonym_text = _generation(bundle["synonym"][synonym_uid], synonym_uid)
        if extract_yes_no(aligned_text) != "yes":
            continue
        if extract_yes_no(synonym_text) != "yes":
            continue
        atoms = since_atoms(synonym_text, lat.entity)
        flags = item_flags(atoms, lat, first_since_class)
        family = lat.family
        block["n"] += 1
        families[family]["n"] += 1
        if flags["has_query"]:
            block["has_query"] += 1
            families[family]["has_query"] += 1
        if flags["has_step0"]:
            block["has_step0"] += 1
            families[family]["has_step0"] += 1
        kind = flags["first"]
        if kind not in FIRST_KEYS:
            _stop(f"{bundle['name']} {digest} class {kind} is unknown")
        block[kind] += 1
        families[family][kind] += 1
    _check_block(bundle["name"], block)
    if sum(families[family]["n"] for family in FAMILIES) != block["n"]:
        _stop(f"{bundle['name']} family n does not sum to {block['n']}")
    for family in FAMILIES:
        _check_block(f"{bundle['name']}.{family}", families[family])
    for key in COUNT_KEYS:
        summed = sum(families[family][key] for family in FAMILIES)
        if summed != block[key]:
            _stop(f"{bundle['name']} family sum {key} {summed} != {block[key]}")
    return {**block, "families": families}


def _gate_report(conditions: dict) -> dict:
    expected = GATES["ours"]
    observed = {key: conditions["ours"][key] for key in GATE_KEYS}
    return {
        "ok": observed == expected,
        "pilot_ours": {
            "ok": observed == expected,
            "observed": observed,
            "expected": {key: expected[key] for key in GATE_KEYS},
        },
    }


def _public_condition(block: dict) -> dict:
    return {
        **{key: block[key] for key in COUNT_KEYS},
        "families": {
            family: {key: block["families"][family][key] for key in COUNT_KEYS}
            for family in FAMILIES
        },
    }


def _print_block(prefix: str, block: dict) -> None:
    fields = " ".join(f"{key}={block[key]}" for key in COUNT_KEYS)
    print(f"{prefix} {fields}")


def _print_public(public: dict) -> None:
    for name, block in public.items():
        _print_block(name, block)
        for family in FAMILIES:
            _print_block(f"  {family}", block["families"][family])


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_module("yes_flip_trace.py", "yes_flip_trace_correct_unseen")
    prior = _load_module("yes_flip_first_since.py", "yes_flip_first_since_correct_unseen")
    _self_check(trace.since_atoms, prior.first_since_class)
    if len(trace.CONDITIONS) != 9:
        _stop(f"expected 9 conditions, found {len(trace.CONDITIONS)}")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    unseen = _index_unseen(pools)
    conditions = {}
    for name, gen_rel, logic_rel in trace.CONDITIONS:
        bundle = _load_condition(name, gen_rel, logic_rel, unseen)
        conditions[name] = _score_condition(
            bundle, unseen, trace.since_atoms, prior.first_since_class
        )
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    gates = _gate_report(conditions)
    public = {name: _public_condition(block) for name, block in conditions.items()}
    count_cell = gates["pilot_ours"]
    print(
        f"gate ours ok={count_cell['ok']} "
        + " ".join(f"{key}={count_cell['observed'][key]}" for key in GATE_KEYS)
    )
    if not count_cell["ok"]:
        print("STOP gate mismatch; counts were not adjusted and the json was not written")
        print(f"expected ours {count_cell['expected']}")
        _print_public(public)
        raise SystemExit(1)
    payload = {
        "config": {
            "seed": SEED,
            "split": "unseen",
            "pool": "test_unseen from build_pools(seed=0)",
            "population": (
                "unseen graph, latent label yes, aligned extract yes, synonym extract yes. "
                "These items were answered yes on the synonym presentation. "
                "They are not the flip population."
            ),
            "aligned_generation": (
                "unseen_aligned|{graph_hash} in each condition's logic.jsonl. "
                "The row is not regenerated. It is used only to require extract yes."
            ),
            "synonym_generation": "unseen_synonym|{graph_hash} in each condition's generations.jsonl",
            "partition": (
                "first_step0 + first_same_premise_decoy + first_later_core + "
                "first_other + first_none = n. "
                "has_query and has_step0 may overlap and are not in that sum. "
                "tree and diamond repeat the same counts and sum to the condition."
            ),
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "yes_flip_trace.since_atoms. Order is the order of matches. "
                "A line containing 'cannot be proved' contributes no atom."
            ),
            "has_query": "some synonym Since atom equals lat.query",
            "has_step0": "some synonym Since atom equals core_rules[0]'s conclusion",
            "first_atom": "atoms[0] when Since atoms exist; otherwise first_none",
            "priority": "step0, then query, then same_premise_decoy, then later_core",
            "first_classes": {
                "first_step0": "the first Since atom equals core_rules[0]'s conclusion",
                "first_same_premise_decoy": (
                    "the first Since atom is the conclusion of some other_rules entry "
                    "whose premise frozenset equals core_rules[0], and step0 and query "
                    "did not claim it"
                ),
                "first_later_core": (
                    "the first Since atom is a conclusion of core_rules[1:], and step0, "
                    "query, and same_premise_decoy did not claim it"
                ),
                "first_other": (
                    "the generation has a Since sentence, and the first atom is neither "
                    "first_step0, nor first_same_premise_decoy, nor first_later_core"
                ),
                "first_none": "the generation has no Since sentence",
            },
            "premise_equality": "frozenset of the premise atoms; order is ignored",
            "class_source": (
                "yes_flip_first_since.first_since_class. step0, same_premise_decoy, "
                "later_core, and none keep their own first classes. query, other_decoy, "
                "and other count as first_other."
            ),
            "families": "tree and diamond",
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "conditions": [name for name, _gen, _logic in trace.CONDITIONS],
            "gates": (
                "pilot ours: n=21, has_query=21, has_step0=15. A mismatch stops the "
                "script before it writes this file. Counts are not adjusted to meet the gate."
            ),
            "loaded_model": False,
            "training": False,
            "reading": (
                "Counts of Since atoms in saved generations on items answered yes. "
                "These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
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
    _print_public(public)
    print(f"gates_ok {gates['ok']}")


if __name__ == "__main__":
    main()
