"""Aligned Since atoms when an unseen synonym flip starts on a later core rule.

Reads saved generations only. No training and no model load. The population
is the same as yes_flip_unseen_later_core.py: an unseen graph, latent label
yes, aligned extract yes, synonym extract no, whose first synonym Since atom
falls in later_core. The class order is yes_flip_first_since_unseen.first_since_class:
step0, then query, then same_premise_decoy, then later_core. Since atoms are
yes_flip_trace.since_atoms and keep that function's order.

The later core rule is the single core_rules[1:] entry with that conclusion.
Two or more core rules with the same conclusion stop the script.

On those items the aligned generation is scored by presence, not by the first
atom. The five headcounts may overlap. aligned_has_same_later_atom is true
when any aligned Since atom equals that later-core conclusion.

These counts are not evidence that reasoning improved, and they are not a
win for any training condition.

Writes only experiments/e1/synonym/yes_flip_unseen_aligned_on_later.json.
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
FAMILIES = ("tree", "diamond")
COUNT_KEYS = (
    "n",
    "aligned_has_step0",
    "aligned_has_query",
    "aligned_has_same_later_atom",
    "aligned_has_step0_and_query",
)
GATE_KEYS = (
    "n",
    "aligned_has_step0",
    "aligned_has_query",
    "aligned_has_same_later_atom",
)
GATES = {
    "ours": {
        "n": 13,
        "aligned_has_step0": 12,
        "aligned_has_query": 13,
        "aligned_has_same_later_atom": 13,
    }
}
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_unseen_aligned_on_later.json"
PRIOR = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_unseen_later_core.json"
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


def _add(block: dict, flags: dict) -> None:
    block["n"] += 1
    for key, value in flags.items():
        if value:
            block[key] += 1


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


def aligned_presence(atoms: list[str], lat, conclusion: str) -> dict:
    """Whether any aligned Since atom is step 0, the query, or this later conclusion."""
    found = {str(atom).lower() for atom in atoms}
    step0 = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    has_step0 = step0 in found
    has_query = query in found
    return {
        "aligned_has_step0": has_step0,
        "aligned_has_query": has_query,
        "aligned_has_same_later_atom": str(conclusion).lower() in found,
        "aligned_has_step0_and_query": has_step0 and has_query,
    }


def _check_block(label: str, block: dict) -> None:
    n = block["n"]
    step0 = block["aligned_has_step0"]
    query = block["aligned_has_query"]
    both = block["aligned_has_step0_and_query"]
    for key in COUNT_KEYS:
        if key == "n":
            continue
        if block[key] > n:
            _stop(f"{label} {key}={block[key]} exceeds n={n}")
    if both > step0 or both > query:
        _stop(f"{label} joint {both} exceeds step0={step0} or query={query}")
    if both < step0 + query - n:
        _stop(
            f"{label} joint {both} is below step0+query-n "
            f"({step0}+{query}-{n})"
        )


def _self_check(since_atoms, first_since_class, rule_error) -> None:
    sample = (
        "Known facts: Tina is vivid.\n"
        "Since Tina is vivid, Tina is windy.\n"
        "Nothing else can be derived, so Tina is snowy cannot be proved.\n"
        "Answer: no"
    )
    if since_atoms(sample, "Tina") != ["windy"]:
        _stop("regex self-check did not keep the intermediate atom only")
    ordered = (
        "Since Tina is vivid, Tina is windy. "
        "Since Tina is windy, Tina is silver."
    )
    if since_atoms(ordered, "Tina") != ["windy", "silver"]:
        _stop("regex self-check did not keep Since atoms in order")

    chain = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("bright",), "muddy"), (("silver",), "snowy")),
        other_rules=((("vivid",), "windy"), (("otherprem",), "amber"), (("x",), "muddy")),
        query="snowy",
        facts=("vivid", "bright"),
        entity="Tina",
        family="tree",
        label="yes",
    )
    if first_since_class(["silver"], chain) != "step0":
        _stop("the step-0 conclusion was not step0")
    if first_since_class(["snowy"], chain) != "query":
        _stop("the query was not preferred to a later core rule")
    if first_since_class(["windy"], chain) != "same_premise_decoy":
        _stop("a same-premise decoy was not preferred to later core")
    if first_since_class(["muddy"], chain) != "later_core":
        _stop("a later core conclusion lost to a different-premise decoy")
    if rule_error("muddy", chain, "self-check") is not None:
        _stop("the unique later core rule was rejected")

    full = aligned_presence(["windy", "silver", "snowy", "muddy"], chain, "muddy")
    if full != {
        "aligned_has_step0": True,
        "aligned_has_query": True,
        "aligned_has_same_later_atom": True,
        "aligned_has_step0_and_query": True,
    }:
        _stop(f"aligned presence missed atoms that are not first: {full}")
    if aligned_presence(["muddy"], chain, "muddy")["aligned_has_same_later_atom"] is not True:
        _stop("the later conclusion was missed when it was the only aligned atom")
    only_later = aligned_presence(["muddy"], chain, "muddy")
    if only_later["aligned_has_step0"] or only_later["aligned_has_query"]:
        _stop("a later conclusion was counted as step 0 or the query")
    middle = aligned_presence(["windy", "muddy", "amber"], chain, "muddy")
    if not middle["aligned_has_same_later_atom"]:
        _stop("a later conclusion in the middle of the aligned chain was missed")
    if middle["aligned_has_step0"] or middle["aligned_has_query"] or middle["aligned_has_step0_and_query"]:
        _stop("decoy atoms were counted as step 0 or the query")
    both = aligned_presence(["snowy", "silver"], chain, "muddy")
    if not (both["aligned_has_step0"] and both["aligned_has_query"] and both["aligned_has_step0_and_query"]):
        _stop("step 0 and the query were missed when neither was first")
    if both["aligned_has_same_later_atom"]:
        _stop("a missing later conclusion was counted")
    step_only = aligned_presence(["silver"], chain, "muddy")
    if not step_only["aligned_has_step0"] or step_only["aligned_has_query"] or step_only["aligned_has_step0_and_query"]:
        _stop("step 0 alone did not stay separate from the query")
    if aligned_presence([], chain, "muddy") != {
        "aligned_has_step0": False,
        "aligned_has_query": False,
        "aligned_has_same_later_atom": False,
        "aligned_has_step0_and_query": False,
    }:
        _stop("a missing aligned Since sentence was counted")

    decoy_and_core = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("bright",), "muddy")),
        other_rules=((("vivid",), "muddy"),),
        query="snowy",
        facts=("vivid",),
        entity="Tina",
    )
    if first_since_class(["muddy"], decoy_and_core) != "same_premise_decoy":
        _stop("a same-premise decoy that is also a later core conclusion entered later_core")

    shared = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("bright",), "muddy"), (("dim",), "muddy")),
        other_rules=(),
        query="snowy",
        facts=("vivid", "bright", "dim"),
        entity="Tina",
    )
    message = rule_error("muddy", shared, "self-check hash")
    if message is None or "muddy" not in message or message.count("core_rules[") != 2:
        _stop(f"two core rules for one atom did not stop:\n{message}")


def _index_unseen(pools: dict) -> dict:
    unseen = {}
    for lat in pools["test_unseen"]:
        digest = graph_hash(lat)
        if digest in unseen:
            _stop(f"duplicate unseen graph hash {digest}")
        if not lat.core_rules:
            _stop(f"{digest} has no core rules")
        if lat.family not in FAMILIES:
            _stop(f"{digest} family {lat.family} is outside tree and diamond")
        unseen[digest] = lat
    if len(unseen) != 200:
        _stop(f"unseen pool has {len(unseen)} graphs")
    present = {lat.family for lat in unseen.values()}
    if present != set(FAMILIES):
        _stop(f"unseen families are {sorted(present)}")
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


def _score_condition(bundle: dict, unseen: dict, since_atoms, first_since_class, core_matches, rule_error) -> tuple[dict, list[str]]:
    block = _blank()
    families = {family: _blank() for family in FAMILIES}
    problems = []
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
        synonym_atoms = since_atoms(synonym_text, lat.entity)
        if first_since_class(synonym_atoms, lat) != "later_core":
            continue
        where = f"{bundle['name']} {digest}"
        problem = rule_error(synonym_atoms[0], lat, where)
        if problem is not None:
            problems.append(problem)
            continue
        _index, _premises, conclusion = core_matches(synonym_atoms[0], lat)[0]
        if str(conclusion).lower() != str(synonym_atoms[0]).lower():
            _stop(f"{where} resolved conclusion does not equal the first Since atom")
        aligned_atoms = since_atoms(aligned_text, lat.entity)
        flags = aligned_presence(aligned_atoms, lat, conclusion)
        _add(block, flags)
        if lat.family not in families:
            _stop(f"{digest} family {lat.family} is outside tree and diamond")
        _add(families[lat.family], flags)
    if not problems:
        _check_block(bundle["name"], block)
        for family, fam in families.items():
            _check_block(f"{bundle['name']}.{family}", fam)
        for key in COUNT_KEYS:
            summed = sum(families[family][key] for family in FAMILIES)
            if summed != block[key]:
                _stop(f"{bundle['name']} family sum {key} {summed} != {block[key]}")
    return {"name_counts": block, "families": families}, problems


def _lock_prior(public: dict) -> None:
    if not PRIOR.is_file():
        _stop(f"missing prior population {PRIOR}")
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    conditions = prior.get("conditions")
    if not isinstance(conditions, dict):
        _stop("prior population has no conditions")
    for name, block in public.items():
        if name not in conditions:
            _stop(f"prior population has no {name}")
        expected = conditions[name]["n_later_first"]
        if block["n"] != expected:
            _stop(f"{name} n {block['n']} != prior n_later_first {expected}")
        for family in FAMILIES:
            prev = conditions[name]["families"][family]["n_later_first"]
            got = block["families"][family]["n"]
            if got != prev:
                _stop(f"{name}.{family} n {got} != prior n_later_first {prev}")


def _gate_report(public: dict) -> dict:
    report = {}
    ok = True
    for name, expected in GATES.items():
        block = public[name]
        observed = {key: block[key] for key in GATE_KEYS}
        cell_ok = observed == expected
        report[name] = {"observed": observed, "expected": expected, "ok": cell_ok}
        ok = ok and cell_ok
    report["ok"] = ok
    return report


def _public_block(block: dict) -> dict:
    return {key: block[key] for key in COUNT_KEYS}


def _print_block(prefix: str, block: dict) -> None:
    parts = " ".join(f"{key}={block[key]}" for key in COUNT_KEYS)
    print(f"{prefix} {parts}")


def _print_public(public: dict) -> None:
    for name, block in public.items():
        _print_block(name, block)
        for family in FAMILIES:
            _print_block(f"  {family}", block["families"][family])


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_module("yes_flip_trace.py", "yes_flip_trace_aligned_on_later")
    prior_mod = _load_module(
        "yes_flip_first_since_unseen.py",
        "yes_flip_first_since_unseen_aligned_on_later",
    )
    later = _load_module(
        "yes_flip_unseen_later_core.py",
        "yes_flip_unseen_later_core_aligned_on_later",
    )
    _self_check(trace.since_atoms, prior_mod.first_since_class, later.rule_error)
    if len(trace.CONDITIONS) != 9:
        _stop(f"expected 9 conditions, found {len(trace.CONDITIONS)}")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    unseen = _index_unseen(pools)
    public = {}
    problems = []
    for name, gen_rel, logic_rel in trace.CONDITIONS:
        bundle = _load_condition(name, gen_rel, logic_rel, unseen)
        scored, found = _score_condition(
            bundle,
            unseen,
            trace.since_atoms,
            prior_mod.first_since_class,
            later.core_matches,
            later.rule_error,
        )
        problems.extend(found)
        public[name] = {
            **_public_block(scored["name_counts"]),
            "families": {
                family: _public_block(scored["families"][family]) for family in FAMILIES
            },
        }
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    if problems:
        print("STOP one atom matches more than one core rule; counts were not written")
        print("\n".join(problems))
        raise SystemExit(1)
    _lock_prior(public)
    gates = _gate_report(public)
    for name in GATES:
        cell = gates[name]
        print(f"gate {name} ok={cell['ok']}")
        print(f"  observed {cell['observed']}")
        print(f"  expected {cell['expected']}")
    if not gates["ok"]:
        print("STOP gate mismatch; counts were not adjusted and the json was not written")
        _print_public(public)
        raise SystemExit(1)
    payload = {
        "config": {
            "seed": SEED,
            "split": SPLIT,
            "pool": "test_unseen from build_pools(seed=0)",
            "population": (
                "unseen graph, latent label yes, aligned extract yes, synonym extract no, "
                "and the first synonym Since atom is later_core. Same population as "
                "yes_flip_unseen_later_core.py."
            ),
            "aligned_generation": (
                "unseen_aligned|{graph_hash} in each condition's logic.jsonl. "
                "The row is not regenerated."
            ),
            "synonym_generation": (
                "unseen_synonym|{graph_hash} in each condition's generations.jsonl"
            ),
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "yes_flip_trace.since_atoms. Order is the order of matches. "
                "A line containing 'cannot be proved' contributes no atom."
            ),
            "class_priority": (
                "yes_flip_first_since_unseen.first_since_class. step0, then query, "
                "then same_premise_decoy, then later_core, then other_decoy, then other. "
                "Only later_core enters this file."
            ),
            "later_core_rule": (
                "The first synonym Since atom equals the conclusion of exactly one "
                "entry of core_rules[1:]. If that atom is the conclusion of more than "
                "one core rule, the script stops and writes nothing. The check is "
                "yes_flip_unseen_later_core.rule_error."
            ),
            "aligned_has_step0": (
                "On the same items, any Since atom in the aligned generation equals "
                "core_rules[0]'s conclusion. The atom need not be first."
            ),
            "aligned_has_query": (
                "On the same items, any Since atom in the aligned generation equals "
                "lat.query. The atom need not be first. This count may overlap "
                "aligned_has_step0."
            ),
            "aligned_has_same_later_atom": (
                "On the same items, any Since atom in the aligned generation equals "
                "the conclusion of the same later core rule. The atom need not be "
                "first. This count may overlap the other aligned counts."
            ),
            "aligned_has_step0_and_query": (
                "On the same items, the aligned generation has both a Since atom "
                "equal to the step-0 conclusion and a Since atom equal to the query. "
                "One atom can satisfy both when those two strings are equal."
            ),
            "overlap": (
                "The five headcounts may overlap. They are not a partition of n. "
                "Each of the four flags is at most n, and "
                "aligned_has_step0_and_query is at most each of its parts and at "
                "least aligned_has_step0 + aligned_has_query - n."
            ),
            "families": list(FAMILIES),
            "family_rule": (
                "tree and diamond each repeat the same five counts. Their sums equal "
                "the condition."
            ),
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "conditions": [name for name, _gen, _logic in trace.CONDITIONS],
            "prior_lock": (
                "n, including tree and diamond, must equal n_later_first in "
                "experiments/e1/synonym/yes_flip_unseen_later_core.json. That file "
                "is only read."
            ),
            "gates": (
                "pilot ours n=13, aligned_has_step0=12, aligned_has_query=13, "
                "aligned_has_same_later_atom=13. A mismatch stops the script before "
                "it writes this file. Counts are not edited to meet the gate."
            ),
            "loaded_model": False,
            "training": False,
            "reading": (
                "On unseen graphs whose synonym yes-to-no flip opens on a later core "
                "rule, counts of whether the same model's aligned Since atoms include "
                "the step-0 conclusion, the query, and that same later conclusion. "
                "These counts are not evidence that reasoning improved, and they are "
                "not a win for any training condition."
            ),
        },
        "gates": {
            "ok": gates["ok"],
            "pilot_ours": gates["ours"],
        },
        "prior_lock": {
            "ok": True,
            "file": "experiments/e1/synonym/yes_flip_unseen_later_core.json",
            "compared": "n equals n_later_first at the condition and at tree and diamond",
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
    print("gates_ok True")
    print("prior_lock_ok True")
    _print_public(public)


if __name__ == "__main__":
    main()
