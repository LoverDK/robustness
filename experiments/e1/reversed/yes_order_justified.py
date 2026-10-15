"""Whether a Since step adds the query only after its premises are known.

Reads saved logic.jsonl rows only. No training and no model load. This is
the reversed presentation, not a synonym rewrite, and this file does not
open generations.jsonl. The population matches yes_order_trace: a seen or
unseen graph, latent label yes, aligned extract yes, and shifted extract no.
An empty shifted extract is counted separately and does not enter the four
classes.

A generation is split into lines. A line whose lowercase text contains
"cannot be proved" produces no step. If that line matches
"{entity} is {query} cannot be proved", the generation is marked
cannot_query. On every other line that contains the word since, matches of
"{entity} is ([a-z]+)" are read in order. The last atom is the conclusion
and the earlier atoms are the premises. The known set starts as the
lowercase atoms in lat.facts. A step is justified when every premise is
already known, and only then is the conclusion added. The query is
justified when a justified step adds it at least once; otherwise early when
the query was some step's conclusion; otherwise never when there is at least
one step; otherwise no_since. The four classes are exclusive and sum to n.
cannot_query is not one of the four classes.

Aligned and shifted generations are parsed separately. Three crosses are
counted only inside this population. These counts are not evidence that
reasoning improved, and they are not a win for any training condition.

Writes only experiments/e1/reversed/yes_order_justified.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl
from structrobust.prompts import extract_yes_no

SEED = 0
POOL_N = 200
CLASSES = ("justified", "early", "never", "no_since")
CROSS_KEYS = (
    "aligned_justified_reversed_never",
    "aligned_justified_reversed_early",
    "aligned_justified_reversed_justified",
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
            "reversed": {
                "justified": 0,
                "early": 0,
                "never": 7,
                "no_since": 0,
                "cannot_query": 7,
            },
            "aligned": {"justified": 7, "early": 0, "never": 0, "no_since": 0},
            "cross": {"aligned_justified_reversed_never": 7},
        },
        "unseen": {
            "n": 8,
            "reversed": {
                "justified": 0,
                "early": 4,
                "never": 4,
                "no_since": 0,
                "cannot_query": 8,
            },
            "aligned": {"justified": 7, "early": 1, "never": 0, "no_since": 0},
        },
    },
    "b2": {
        "seen": {
            "n": 15,
            "reversed": {
                "justified": 0,
                "early": 0,
                "never": 15,
                "no_since": 0,
                "cannot_query": 15,
            },
            "aligned": {"justified": 15, "early": 0, "never": 0, "no_since": 0},
            "cross": {"aligned_justified_reversed_never": 15},
        },
        "unseen": {
            "n": 12,
            "reversed": {
                "justified": 0,
                "early": 0,
                "never": 12,
                "no_since": 0,
                "cannot_query": 12,
            },
            "aligned": {"justified": 10, "early": 1, "never": 1, "no_since": 0},
        },
    },
    "b1": {
        "seen": {
            "n": 48,
            "reversed": {
                "justified": 0,
                "early": 0,
                "never": 48,
                "no_since": 0,
                "cannot_query": 48,
            },
            "aligned": {"justified": 47, "early": 1, "never": 0, "no_since": 0},
        },
        "unseen": {
            "n": 58,
            "reversed": {
                "justified": 1,
                "early": 0,
                "never": 57,
                "no_since": 0,
                "cannot_query": 58,
            },
            "aligned": {"justified": 53, "early": 5, "never": 0, "no_since": 0},
        },
    },
}
OUT = PROJECT / "experiments" / "e1" / "reversed" / "yes_order_justified.json"
TRACE_JSON = PROJECT / "experiments" / "e1" / "reversed" / "yes_order_trace.json"
PREFIX_JSON = PROJECT / "experiments" / "e1" / "reversed" / "yes_order_prefix.json"
SINCE_WORD = re.compile(r"\bsince\b", re.IGNORECASE)
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
    "yes_order_trace.json",
    "yes_order_prefix.json",
}


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _blank_classes() -> dict:
    return {name: 0 for name in CLASSES}


def _blank_cross() -> dict:
    return {name: 0 for name in CROSS_KEYS}


def population_slot(label: str, aligned_pred: str, shifted_pred: str) -> str:
    """flip is classified; empty is counted aside; everything else is out."""
    if label != "yes" or aligned_pred != "yes":
        return "out"
    if shifted_pred == "":
        return "empty"
    if shifted_pred == "no":
        return "flip"
    return "out"


def parse_generation(text: str, entity: str, query: str, facts) -> dict:
    """One generation: exclusive class, plus whether the query was called unproved."""
    atom_re = re.compile(rf"{re.escape(entity)} is ([a-z]+)")
    cannot_re = re.compile(
        rf"{re.escape(entity)} is {re.escape(str(query))} cannot be proved",
        re.IGNORECASE,
    )
    query_l = str(query).lower()
    known = {str(fact).lower() for fact in facts}
    cannot_query = False
    steps: list[tuple[list[str], str]] = []
    for line in text.splitlines():
        if "cannot be proved" in line.lower():
            if cannot_re.search(line):
                cannot_query = True
            continue
        if SINCE_WORD.search(line) is None:
            continue
        atoms = [match.group(1).lower() for match in atom_re.finditer(line)]
        if not atoms:
            continue
        steps.append((atoms[:-1], atoms[-1]))
    justified_add = False
    was_conclusion = False
    for premises, conclusion in steps:
        if all(premise in known for premise in premises):
            known.add(conclusion)
            if conclusion == query_l:
                justified_add = True
        if conclusion == query_l:
            was_conclusion = True
    if justified_add:
        kind = "justified"
    elif was_conclusion:
        kind = "early"
    elif steps:
        kind = "never"
    else:
        kind = "no_since"
    if kind not in CLASSES:
        _stop(f"unknown class {kind}")
    return {"class": kind, "cannot_query": cannot_query}


def _generation(row: dict, uid: str) -> str:
    text = row.get("generation")
    if not isinstance(text, str):
        _stop(f"{uid} has no generation text")
    return text


def _guard_out(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OUT.resolve() or resolved.name != "yes_order_justified.json":
        _stop(f"refusing to write {resolved}")
    if resolved in {TRACE_JSON.resolve(), PREFIX_JSON.resolve()}:
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


def _expect(got, wanted, msg: str) -> None:
    if got != wanted:
        _stop(f"{msg}: got {got}")


def _self_check() -> None:
    facts = ("vivid",)
    chain = (
        "Known facts: Tina is vivid.\n"
        "Since Tina is vivid, Tina is silver.\n"
        "Since Tina is silver, Tina is snowy.\n"
        "So Tina is snowy is proved.\n"
        "Answer: yes"
    )
    proved = parse_generation(chain, "Tina", "snowy", facts)
    _expect(proved["class"], "justified", "a premise-ready chain was not justified")
    _expect(proved["cannot_query"], False, "a proved chain was marked cannot_query")

    early = parse_generation(
        "Since Tina is silver, Tina is snowy.\n",
        "Tina",
        "snowy",
        facts,
    )
    _expect(early["class"], "early", "a query concluded before its premise was not early")
    _expect(early["cannot_query"], False, "an early line was marked cannot_query")

    late_premise = (
        "Since Tina is silver, Tina is snowy.\n"
        "Since Tina is vivid, Tina is silver.\n"
    )
    _expect(
        parse_generation(late_premise, "Tina", "snowy", facts)["class"],
        "early",
        "a later premise rescued an earlier query step",
    )
    rescued = (
        "Since Tina is missing, Tina is snowy.\n"
        "Since Tina is vivid, Tina is snowy.\n"
    )
    _expect(
        parse_generation(rescued, "Tina", "snowy", facts)["class"],
        "justified",
        "a later justified query step stayed early",
    )

    never = (
        "Since Tina is vivid, Tina is windy.\n"
        "Nothing else can be derived, so Tina is snowy cannot be proved.\n"
        "Answer: no"
    )
    never_parsed = parse_generation(never, "Tina", "snowy", facts)
    _expect(never_parsed["class"], "never", "a chain that misses the query was not never")
    _expect(never_parsed["cannot_query"], True, "the query-cannot line was missed")

    both = (
        "Since Tina is vivid, Tina is snowy.\n"
        "Nothing else can be derived, so Tina is snowy cannot be proved.\n"
    )
    both_parsed = parse_generation(both, "Tina", "snowy", facts)
    _expect(both_parsed["class"], "justified", "a cannot line erased a justified step")
    _expect(both_parsed["cannot_query"], True, "a justified generation lost cannot_query")

    blocked = "Since Tina is vivid, Tina is snowy cannot be proved.\n"
    blocked_parsed = parse_generation(blocked, "Tina", "snowy", facts)
    _expect(blocked_parsed["class"], "no_since", "a cannot line produced a step")
    _expect(blocked_parsed["cannot_query"], True, "a since-and-cannot line missed cannot_query")

    bare = parse_generation("Answer: no\n", "Tina", "snowy", facts)
    _expect(bare["class"], "no_since", "a generation without Since was not no_since")
    _expect(bare["cannot_query"], False, "Answer: no was marked cannot_query")

    other_atom = parse_generation(
        "Tina is windy cannot be proved.\n",
        "Tina",
        "snowy",
        facts,
    )
    _expect(other_atom["cannot_query"], False, "a different atom set cannot_query")
    _expect(other_atom["class"], "no_since", "a cannot line about another atom produced a step")

    shouted = parse_generation(
        "NOTHING ELSE, SO TINA IS SNOWY CANNOT BE PROVED.\n",
        "Tina",
        "snowy",
        facts,
    )
    _expect(shouted["cannot_query"], True, "cannot_query was case sensitive")
    _expect(shouted["class"], "no_since", "an uppercase cannot line produced a step")

    _expect(
        parse_generation("consince Tina is vivid, Tina is snowy.\n", "Tina", "snowy", facts)["class"],
        "no_since",
        "a substring of since produced a step",
    )
    _expect(
        parse_generation("SINCE Tina is vivid, Tina is snowy.\n", "Tina", "snowy", facts)["class"],
        "justified",
        "the word Since was missed",
    )

    invented = (
        "Known facts: Tina is vivid.\n"
        "Since Tina is vivid, Tina is snowy.\n"
    )
    _expect(
        parse_generation(invented, "Tina", "snowy", ())["class"],
        "early",
        "the Known-facts line was copied into the known set",
    )

    one_premise = parse_generation(
        "Since Tina is vivid and Tina is bright, Tina is snowy.\n",
        "Tina",
        "snowy",
        facts,
    )
    _expect(one_premise["class"], "early", "a missing second premise was ignored")
    both_premises = parse_generation(
        "Since Tina is vivid and Tina is bright, Tina is snowy.\n",
        "Tina",
        "snowy",
        ("vivid", "bright"),
    )
    _expect(both_premises["class"], "justified", "two known premises were not enough")

    packed = "Since Tina is vivid, Tina is silver, Tina is snowy.\n"
    _expect(
        parse_generation(packed, "Tina", "snowy", facts)["class"],
        "early",
        "atoms before the last is were treated as their own steps",
    )
    _expect(
        parse_generation("Since the above, Tina is snowy.\n", "Tina", "snowy", ())["class"],
        "justified",
        "an empty premise list was not vacuous",
    )
    _expect(
        parse_generation("Since Tina is snowy, Tina is windy.\n", "Tina", "snowy", facts)["class"],
        "never",
        "a query used only as a premise was called a conclusion",
    )
    _expect(
        parse_generation("Since Tina is Vivid, Tina is Snowy.\n", "Tina", "snowy", facts)["class"],
        "no_since",
        "the atom pattern accepted uppercase",
    )

    dotted = parse_generation(
        "A.B is snowy cannot be proved.\nAXB is snowy cannot be proved.\n",
        "A.B",
        "snowy",
        (),
    )
    _expect(dotted["cannot_query"], True, "re.escape missed the entity")
    escaped = parse_generation("AXB is snowy cannot be proved.\n", "A.B", "snowy", ())
    _expect(escaped["cannot_query"], False, "re.escape let a dot match another character")

    already = parse_generation(
        "Since Tina is vivid, Tina is windy.\n",
        "Tina",
        "snowy",
        ("snowy", "vivid"),
    )
    _expect(already["class"], "never", "a query that was only a fact was called justified")
    again = parse_generation(
        "Since Tina is vivid, Tina is snowy.\n",
        "Tina",
        "snowy",
        ("Vivid",),
    )
    _expect(again["class"], "justified", "a fact was not lowercased before the premise check")

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

    for name, splits in GATES.items():
        for split, expected in splits.items():
            rev = expected["reversed"]
            ali = expected["aligned"]
            if sum(rev[key] for key in CLASSES) != expected["n"]:
                _stop(f"gate {name}.{split} reversed classes do not sum to n")
            if sum(ali[key] for key in CLASSES) != expected["n"]:
                _stop(f"gate {name}.{split} aligned classes do not sum to n")
            if rev["cannot_query"] > expected["n"]:
                _stop(f"gate {name}.{split} cannot_query exceeds n")
            for key, value in expected.get("cross", {}).items():
                if key not in CROSS_KEYS or value > expected["n"]:
                    _stop(f"gate {name}.{split} has a bad cross {key}")


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


def _check_partition(name: str, split: str, block: dict) -> None:
    n = block["n"]
    rev = block["reversed"]
    ali = block["aligned"]
    cross = block["cross"]
    if sum(rev[key] for key in CLASSES) != n:
        _stop(f"{name}.{split} reversed classes {rev} do not sum to n {n}")
    if sum(ali[key] for key in CLASSES) != n:
        _stop(f"{name}.{split} aligned classes {ali} do not sum to n {n}")
    if rev["cannot_query"] > n:
        _stop(f"{name}.{split} cannot_query {rev['cannot_query']} exceeds n {n}")
    if cross["aligned_justified_reversed_never"] > min(ali["justified"], rev["never"]):
        _stop(f"{name}.{split} cross never exceeds its margins")
    if cross["aligned_justified_reversed_early"] > min(ali["justified"], rev["early"]):
        _stop(f"{name}.{split} cross early exceeds its margins")
    if cross["aligned_justified_reversed_justified"] > min(ali["justified"], rev["justified"]):
        _stop(f"{name}.{split} cross justified exceeds its margins")
    covered = sum(cross.values())
    if covered > ali["justified"]:
        _stop(f"{name}.{split} crosses {covered} exceed aligned justified {ali['justified']}")
    if rev["never"] == n and cross["aligned_justified_reversed_never"] != ali["justified"]:
        _stop(f"{name}.{split} every reversed item is never, but the never cross disagrees")
    if rev["never"] == n and (
        cross["aligned_justified_reversed_early"] or cross["aligned_justified_reversed_justified"]
    ):
        _stop(f"{name}.{split} a non-never cross survived an all-never reversed split")


def _score_split(name: str, split: str, logic: dict, pool: dict) -> dict:
    n = 0
    n_empty = 0
    reversed_counts = _blank_classes()
    aligned_counts = _blank_classes()
    cannot_query = 0
    cross = _blank_cross()
    for digest, lat in pool.items():
        aligned_uid = f"{split}_aligned|{digest}"
        shifted_uid = f"{split}_shifted|{digest}"
        aligned_text = _generation(logic[aligned_uid], aligned_uid)
        shifted_text = _generation(logic[shifted_uid], shifted_uid)
        slot = population_slot(
            lat.label,
            extract_yes_no(aligned_text),
            extract_yes_no(shifted_text),
        )
        if slot == "out":
            continue
        if slot == "empty":
            n_empty += 1
            continue
        reversed_parsed = parse_generation(shifted_text, lat.entity, lat.query, lat.facts)
        aligned_parsed = parse_generation(aligned_text, lat.entity, lat.query, lat.facts)
        reversed_counts[reversed_parsed["class"]] += 1
        aligned_counts[aligned_parsed["class"]] += 1
        if reversed_parsed["cannot_query"]:
            cannot_query += 1
        if aligned_parsed["class"] == "justified" and reversed_parsed["class"] == "never":
            cross["aligned_justified_reversed_never"] += 1
        elif aligned_parsed["class"] == "justified" and reversed_parsed["class"] == "early":
            cross["aligned_justified_reversed_early"] += 1
        elif aligned_parsed["class"] == "justified" and reversed_parsed["class"] == "justified":
            cross["aligned_justified_reversed_justified"] += 1
        n += 1
    block = {
        "n": n,
        "n_empty_extract": n_empty,
        "reversed": {**reversed_counts, "cannot_query": cannot_query},
        "aligned": aligned_counts,
        "cross": cross,
    }
    _check_partition(name, split, block)
    return block


def _gate_cell(observed: dict, expected: dict) -> dict:
    rev_ok = all(observed["reversed"][key] == expected["reversed"][key] for key in (*CLASSES, "cannot_query"))
    ali_ok = all(observed["aligned"][key] == expected["aligned"][key] for key in CLASSES)
    cross_ok = all(observed["cross"][key] == value for key, value in expected.get("cross", {}).items())
    ok = observed["n"] == expected["n"] and rev_ok and ali_ok and cross_ok
    shown = {
        "n": observed["n"],
        "reversed": observed["reversed"],
        "aligned": observed["aligned"],
        "cross": observed["cross"],
    }
    return {"ok": ok, "observed": shown, "expected": expected}


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


def _lock_population(conditions: dict) -> None:
    if not TRACE_JSON.is_file():
        _stop(f"missing {TRACE_JSON}")
    if TRACE_JSON.resolve() in {OUT.resolve(), PREFIX_JSON.resolve()}:
        _stop("population lock points at an output file")
    previous = json.loads(TRACE_JSON.read_text(encoding="utf-8"))
    for name, _rel in CONDITIONS:
        for split, _pool_name, _families in SPLITS:
            old = previous["conditions"][name][split]
            new = conditions[name][split]
            if new["n"] != old["n"]:
                _stop(f"{name}.{split} n {new['n']} != yes_order_trace {old['n']}")
            if new["n_empty_extract"] != old["n_empty_extract"]:
                _stop(
                    f"{name}.{split} empty {new['n_empty_extract']} != "
                    f"yes_order_trace {old['n_empty_extract']}"
                )


def _print_split(name: str, split: str, block: dict) -> None:
    rev = block["reversed"]
    ali = block["aligned"]
    nonzero = [f"{key}={block['cross'][key]}" for key in CROSS_KEYS if block["cross"][key]]
    shown = " ".join(nonzero) if nonzero else "none"
    print(
        f"{name} {split} n={block['n']} empty={block['n_empty_extract']} "
        f"reversed justified={rev['justified']} early={rev['early']} "
        f"never={rev['never']} no_since={rev['no_since']} cannot_query={rev['cannot_query']} "
        f"aligned justified={ali['justified']} early={ali['early']} "
        f"never={ali['never']} no_since={ali['no_since']} cross({shown})"
    )


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    _self_check()
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
            split: _score_split(name, split, logic, indexed[split])
            for split, _pool_name, _families in SPLITS
        }
        for split, _pool_name, _families in SPLITS:
            _check_partition(name, split, conditions[name][split])
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    _lock_population(conditions)
    gates = _gate_report(conditions)
    for key, cell in gates.items():
        if key == "ok":
            continue
        print(
            f"gate {key} ok={cell['ok']} n={cell['observed']['n']} "
            f"expected_n={cell['expected']['n']}"
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
                "The population matches yes_order_trace."
            ),
            "empty_extract": (
                "Gold-yes items whose aligned extract is yes and whose shifted "
                "extract is empty. They are counted in n_empty_extract and do not "
                "enter the four classes. n counts shifted extract no."
            ),
            "line": "The generation is split on lines. A line is one step or none.",
            "cannot_line": (
                "A line whose lowercase text contains 'cannot be proved' produces "
                "no step. If it matches '{entity} is {query} cannot be proved' "
                "with re.IGNORECASE and re.escape on entity and query, the "
                "generation is cannot_query. cannot_query is not one of the four classes."
            ),
            "since_line": (
                "Any other line that contains the word since, case insensitive. "
                "Atoms are matches of '{entity} is ([a-z]+)' in order. The last "
                "atom is the conclusion and the earlier atoms are the premises. "
                "A since line with no atom produces no step."
            ),
            "known_set": (
                "Starts as the lowercase atoms in lat.facts. A step is justified "
                "when every premise is already in that set. Only a justified step "
                "adds its conclusion."
            ),
            "classes": {
                "justified": "a justified step adds the query at least once",
                "early": "the query is some step's conclusion, and no justified step adds it",
                "never": "there is at least one step, and the query is never a conclusion",
                "no_since": "there is no step",
            },
            "partition": "On each side, the four classes are exclusive and sum to n.",
            "sides": "aligned is the canonical generation; shifted is the reversed generation",
            "cross": {
                "aligned_justified_reversed_never": "aligned class justified and reversed class never",
                "aligned_justified_reversed_early": "aligned class justified and reversed class early",
                "aligned_justified_reversed_justified": "aligned class justified and reversed class justified",
            },
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "population_lock": (
                "n and n_empty_extract match experiments/e1/reversed/yes_order_trace.json"
            ),
            "inputs": [rel for _name, rel in CONDITIONS],
            "conditions": [name for name, _rel in CONDITIONS],
            "gates": (
                "pilot ours seen: n=7, reversed never 7, cannot_query 7, aligned justified 7, "
                "cross aligned-justified and reversed-never 7. "
                "pilot ours unseen: n=8, reversed early 4, never 4, cannot_query 8, "
                "aligned justified 7, early 1. "
                "b2 seen: n=15, reversed never 15, cannot_query 15, aligned justified 15, "
                "cross never 15. "
                "b2 unseen: n=12, reversed never 12, cannot_query 12, aligned justified 10, "
                "early 1, never 1. "
                "b1 seen: n=48, reversed never 48, cannot_query 48, aligned justified 47, early 1. "
                "b1 unseen: n=58, reversed never 57, justified 1, cannot_query 58, "
                "aligned justified 53, early 5. "
                "A mismatch stops the script before it writes this file."
            ),
            "loaded_model": False,
            "training": False,
            "not_read": "synonym generations.jsonl",
            "reading": (
                "Counts of whether a Since step adds the query after its premises "
                "are known. These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
            ),
        },
        "gates": {
            "ok": gates["ok"],
            "population_matches_yes_order_trace": True,
            "pilot_ours_seen": gates["ours_seen"],
            "pilot_ours_unseen": gates["ours_unseen"],
            "b2_seen": gates["b2_seen"],
            "b2_unseen": gates["b2_unseen"],
            "b1_seen": gates["b1_seen"],
            "b1_unseen": gates["b1_unseen"],
        },
        "conditions": conditions,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.resolve() in {TRACE_JSON.resolve(), PREFIX_JSON.resolve()} or tmp.name in BANNED_NAMES:
        _stop(f"refusing to write {tmp}")
    if "synonym" in tmp.parts or "paper" in tmp.parts:
        _stop(f"refusing to write {tmp}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    print(f"gates_ok {gates['ok']}")
    for name, _rel in CONDITIONS:
        for split, _pool_name, _families in SPLITS:
            _print_split(name, split, conditions[name][split])


if __name__ == "__main__":
    main()
