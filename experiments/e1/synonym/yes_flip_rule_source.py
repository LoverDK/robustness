"""Rule source of Since atoms when a synonym answer flips gold yes to no.

Reads saved generations only. No training and no model load. Since sentences
use yes_flip_trace.since_atoms: a case-insensitive match of

    since\\b[^\\.\\n]*,\\s*{entity} is ([a-z]+)

on a line that does not contain "cannot be proved".

Among other_derived items, each captured atom is given one source. Premise
equality for the reported counts is frozenset equality with core_rules[0];
order does not matter. Tuple equality is computed only for the gate. These
counts are not evidence that reasoning improved, and they are not a win for
any training condition.

Writes only experiments/e1/synonym/yes_flip_rule_source.json.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl
from structrobust.prompts import extract_yes_no

SEED = 0
SPLIT = "seen"
BUCKETS = ("query_derived", "step0_not_query", "other_derived", "no_since")
FAMILIES = ("chain", "conj_fact")
SOURCE_NAMES = (
    "same_premise_decoy",
    "other_decoy",
    "later_core",
    "unknown",
    "fact",
    "mixed",
)
LEADER_KEYS = (
    "other_derived",
    "same_premise_decoy",
    "other_decoy",
    "later_core",
    "unknown",
    "mixed",
    "pair_n",
)
LEADER_TUPLE = {
    "condition": "ours",
    "equality": "tuple",
    "other_derived": 42,
    "same_premise_decoy": 38,
    "other_decoy": 3,
    "later_core": 0,
    "unknown": 0,
    "mixed": 1,
    "pair_n": 18,
}
SANITY_HASH = "099cd1bf088e74592cba2fd9c8b6303c1b89361f8fe3f673939bfb6394ab30ad"
SANITY_STEP0 = (("vivid",), "silver")
SANITY_QUERY = "snowy"
SANITY_DECOY = (("vivid",), "windy")
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_rule_source.json"
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
    "synonym_burden_contrast.json",
    "burden_contrast.json",
    "report.md",
}


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _load_trace():
    path = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_trace.py"
    spec = importlib.util.spec_from_file_location("yes_flip_trace_rules", path)
    if spec is None or spec.loader is None:
        _stop(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def premise_key(premises, mode: str):
    atoms = tuple(str(premise).lower() for premise in premises)
    if mode == "tuple":
        return atoms
    if mode == "frozenset":
        return frozenset(atoms)
    _stop(f"unknown premise mode {mode}")
    return atoms


def atom_source(atom: str, lat, mode: str) -> str:
    """One class for one atom. An other_rule match precedes later core and fact."""
    atom = atom.lower()
    step0 = premise_key(lat.core_rules[0][0], mode)
    matched_other = False
    same = False
    for premises, conclusion in lat.other_rules:
        if str(conclusion).lower() != atom:
            continue
        matched_other = True
        if premise_key(premises, mode) == step0:
            same = True
    if same:
        return "same_premise_decoy"
    if matched_other:
        return "other_decoy"
    for _premises, conclusion in lat.core_rules[1:]:
        if str(conclusion).lower() == atom:
            return "later_core"
    if atom in {str(fact).lower() for fact in lat.facts}:
        return "fact"
    return "unknown"


def item_source(atoms: list[str], lat, mode: str) -> str:
    labels = {atom_source(atom, lat, mode) for atom in atoms}
    if not labels:
        _stop("item_source called with no Since atoms")
    if len(labels) == 1:
        return next(iter(labels))
    return "mixed"


def contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def _blank_sources() -> dict:
    return {name: 0 for name in SOURCE_NAMES}


def _blank_block() -> dict:
    block = {
        "n": 0,
        "n_empty_extract": 0,
        **{name: 0 for name in BUCKETS},
        "sources": _blank_sources(),
        "step0_word_in_generation": 0,
    }
    return block


def _check_block(label: str, block: dict) -> None:
    buckets = sum(block[name] for name in BUCKETS)
    if buckets != block["n"]:
        _stop(f"{label} buckets {buckets} != n {block['n']}")
    sources = sum(block["sources"].values())
    if sources != block["other_derived"]:
        _stop(
            f"{label} sources {sources} != other_derived {block['other_derived']} "
            f"{block['sources']}"
        )
    if block["step0_word_in_generation"] > block["other_derived"]:
        _stop(f"{label} step0 word count exceeds other_derived")


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


def _self_check(since_atoms) -> None:
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

    lat = SimpleNamespace(
        core_rules=((("a", "b"), "step0"), (("step0",), "later")),
        other_rules=((("b", "a"), "windy"), (("c",), "amber")),
        facts=("a", "factatom"),
        query="later",
        entity="Tina",
    )
    if atom_source("windy", lat, "tuple") != "other_decoy":
        _stop("tuple equality treated swapped premises as the same set")
    if atom_source("windy", lat, "frozenset") != "same_premise_decoy":
        _stop("frozenset equality missed swapped premises")
    if atom_source("amber", lat, "frozenset") != "other_decoy":
        _stop("a different-premise decoy was not other_decoy")
    if atom_source("later", lat, "frozenset") != "later_core":
        _stop("a later core conclusion was not later_core")
    if atom_source("factatom", lat, "frozenset") != "fact":
        _stop("a fact atom was not fact")
    if atom_source("invented", lat, "frozenset") != "unknown":
        _stop("an atom outside the rules and facts was not unknown")
    if item_source(["windy", "windy"], lat, "frozenset") != "same_premise_decoy":
        _stop("repeated atoms of one class were not counted as that class")
    if item_source(["amber", "later"], lat, "tuple") != "mixed":
        _stop("two source classes were not mixed")
    if item_source(["factatom", "amber"], lat, "tuple") != "mixed":
        _stop("a fact atom together with another class was not mixed")
    if not contains_word("Tina is Silver.", "silver"):
        _stop("whole-word check missed a case-insensitive atom")
    if contains_word("silverware", "silver"):
        _stop("whole-word check matched inside a longer word")


def _index_seen(pools: dict, trace) -> dict:
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
    if trace.FAMILIES != FAMILIES or trace.CONDITIONS is None:
        _stop("yes_flip_trace conditions could not be read")
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


def _flip_rows(bundle: dict, seen: dict, since_atoms, bucket_of) -> list[dict]:
    rows = []
    for digest, lat in seen.items():
        if lat.label != "yes":
            continue
        aligned_uid = f"{SPLIT}_aligned|{digest}"
        synonym_uid = f"{SPLIT}_synonym|{digest}"
        aligned_pred = extract_yes_no(_generation(bundle["logic"][aligned_uid], aligned_uid))
        text = _generation(bundle["synonym"][synonym_uid], synonym_uid)
        synonym_pred = extract_yes_no(text)
        if aligned_pred != "yes":
            continue
        if synonym_pred == "":
            rows.append({
                "digest": digest,
                "family": lat.family,
                "lat": lat,
                "text": text,
                "empty": True,
                "bucket": None,
                "atoms": [],
            })
            continue
        if synonym_pred != "no":
            continue
        step0 = str(lat.core_rules[0][1]).lower()
        query = str(lat.query).lower()
        atoms = since_atoms(text, lat.entity)
        bucket = bucket_of(atoms, step0, query)
        rows.append({
            "digest": digest,
            "family": lat.family,
            "lat": lat,
            "text": text,
            "empty": False,
            "bucket": bucket,
            "atoms": atoms,
            "source_frozenset": item_source(atoms, lat, "frozenset") if bucket == "other_derived" else None,
            "source_tuple": item_source(atoms, lat, "tuple") if bucket == "other_derived" else None,
            "step0_word": contains_word(text, step0) if bucket == "other_derived" else False,
        })
    return rows


def _add_row(total: dict, family_blocks: dict, row: dict) -> None:
    family = family_blocks[row["family"]]
    if row["empty"]:
        total["n_empty_extract"] += 1
        family["n_empty_extract"] += 1
        return
    bucket = row["bucket"]
    total[bucket] += 1
    family[bucket] += 1
    total["n"] += 1
    family["n"] += 1
    if bucket != "other_derived":
        return
    source = row["source_frozenset"]
    total["sources"][source] += 1
    family["sources"][source] += 1
    if row["step0_word"]:
        total["step0_word_in_generation"] += 1
        family["step0_word_in_generation"] += 1


def _score_condition(rows: list[dict]) -> dict:
    total = _blank_block()
    families = {family: _blank_block() for family in FAMILIES}
    for row in rows:
        _add_row(total, families, row)
    total["families"] = families
    _check_block("condition", total)
    for family in FAMILIES:
        _check_block(family, families[family])
    summed = _blank_block()
    for family in FAMILIES:
        block = families[family]
        summed["n"] += block["n"]
        summed["n_empty_extract"] += block["n_empty_extract"]
        summed["step0_word_in_generation"] += block["step0_word_in_generation"]
        for name in BUCKETS:
            summed[name] += block[name]
        for name in SOURCE_NAMES:
            summed["sources"][name] += block["sources"][name]
    for name in ("n", "n_empty_extract", "step0_word_in_generation", *BUCKETS):
        if summed[name] != total[name]:
            _stop(f"family sum {name} {summed[name]} != {total[name]}")
    for name in SOURCE_NAMES:
        if summed["sources"][name] != total["sources"][name]:
            _stop(f"family source sum {name} failed")
    return total


def _score_pair(left_rows: list[dict], right_name: str, right_bundle: dict, seen: dict, since_atoms) -> dict:
    left_by = {row["digest"]: row for row in left_rows}
    n = 0
    left = _blank_block()
    right = {
        "condition": right_name,
        "n_since_step0_conclusion": 0,
        "n_since_query": 0,
        "n_since_same_premise_decoy": 0,
    }
    for digest, lat in seen.items():
        if lat.label != "yes":
            continue
        aligned_uid = f"{SPLIT}_aligned|{digest}"
        synonym_uid = f"{SPLIT}_synonym|{digest}"
        right_aligned = extract_yes_no(_generation(right_bundle["logic"][aligned_uid], aligned_uid))
        right_text = _generation(right_bundle["synonym"][synonym_uid], synonym_uid)
        right_pred = extract_yes_no(right_text)
        if right_aligned != "yes" or right_pred != "yes":
            continue
        left_row = left_by.get(digest)
        if left_row is None or left_row["empty"]:
            continue
        step0 = str(lat.core_rules[0][1]).lower()
        query = str(lat.query).lower()
        atoms = since_atoms(right_text, lat.entity)
        found = {atom.lower() for atom in atoms}
        if step0 in found:
            right["n_since_step0_conclusion"] += 1
        if query in found:
            right["n_since_query"] += 1
        if any(atom_source(atom, lat, "frozenset") == "same_premise_decoy" for atom in found):
            right["n_since_same_premise_decoy"] += 1
        bucket = left_row["bucket"]
        left[bucket] += 1
        left["n"] += 1
        if bucket == "other_derived":
            left["sources"][left_row["source_frozenset"]] += 1
            if left_row["step0_word"]:
                left["step0_word_in_generation"] += 1
        n += 1
    _check_block("pair-left", left)
    for key in ("n_since_step0_conclusion", "n_since_query", "n_since_same_premise_decoy"):
        if right[key] > n:
            _stop(f"pair right {key} {right[key]} exceeds n {n}")
    return {"n": n, "left": left, "right": right}


def _gate_view(block: dict, pair_n: int, mode: str) -> dict:
    return {
        "condition": "ours",
        "equality": mode,
        "other_derived": block["other_derived"],
        "same_premise_decoy": block["sources"]["same_premise_decoy"],
        "other_decoy": block["sources"]["other_decoy"],
        "later_core": block["sources"]["later_core"],
        "unknown": block["sources"]["unknown"],
        "fact": block["sources"]["fact"],
        "mixed": block["sources"]["mixed"],
        "pair_n": pair_n,
    }


def _leader_view(view: dict) -> dict:
    return {key: view[key] for key in LEADER_KEYS}


def _gates(ours_rows: list[dict], ours_block: dict, pair_n: int) -> dict:
    tuple_sources = _blank_sources()
    frozen_sources = _blank_sources()
    differ = []
    other_n = 0
    for row in ours_rows:
        if row["empty"] or row["bucket"] != "other_derived":
            continue
        other_n += 1
        tuple_sources[row["source_tuple"]] += 1
        frozen_sources[row["source_frozenset"]] += 1
        if row["source_tuple"] != row["source_frozenset"]:
            differ.append(row["digest"])
    if other_n != ours_block["other_derived"]:
        _stop("gate other_derived recount failed")
    if frozen_sources != ours_block["sources"]:
        _stop("gate frozenset recount does not match the condition block")
    tuple_block = {
        "other_derived": other_n,
        "sources": tuple_sources,
    }
    frozen_view = _gate_view(ours_block, pair_n, "frozenset")
    tuple_view = _gate_view(tuple_block, pair_n, "tuple")
    count_keys = (
        "other_derived",
        "same_premise_decoy",
        "other_decoy",
        "later_core",
        "unknown",
        "fact",
        "mixed",
        "pair_n",
    )
    tuple_match = all(frozen_view[key] == tuple_view[key] for key in count_keys) and not differ
    return {
        "tuple_match": tuple_match,
        "n_items_differ": len(differ),
        "differing_graph_hashes": sorted(differ),
        "leader": LEADER_TUPLE,
        "matches_leader": _leader_view(tuple_view) == _leader_view(LEADER_TUPLE) and pair_n == LEADER_TUPLE["pair_n"],
        "tuple": tuple_view,
        "frozenset": frozen_view,
    }


def _find_decoy(lat) -> dict | None:
    found = None
    for premises, conclusion in lat.other_rules:
        if (
            tuple(str(premise).lower() for premise in premises) == SANITY_DECOY[0]
            and str(conclusion).lower() == SANITY_DECOY[1]
        ):
            found = {"premises": list(SANITY_DECOY[0]), "conclusion": SANITY_DECOY[1]}
    return found


def _sanity(rows: list[dict], seen: dict) -> dict:
    lat = seen.get(SANITY_HASH)
    row = next((item for item in rows if item["digest"] == SANITY_HASH), None)
    payload = {
        "graph_hash": SANITY_HASH,
        "condition": "ours",
        "in_population": False,
        "entity": None if lat is None else lat.entity,
        "family": None if lat is None else lat.family,
        "label": None if lat is None else lat.label,
        "step0_premises": None,
        "step0_conclusion": None,
        "query": None,
        "same_premise_decoy_rule": None,
        "derived_atoms": None,
        "bucket": None,
        "source": None,
        "sanity_ok": False,
    }
    if lat is None or row is None or row["empty"]:
        return payload
    step0_premises = [str(premise).lower() for premise in lat.core_rules[0][0]]
    step0_conclusion = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    decoy = _find_decoy(lat)
    payload.update({
        "in_population": True,
        "step0_premises": step0_premises,
        "step0_conclusion": step0_conclusion,
        "query": query,
        "same_premise_decoy_rule": decoy,
        "derived_atoms": row["atoms"],
        "bucket": row["bucket"],
        "source": row["source_frozenset"],
    })
    structure_ok = (
        lat.label == "yes"
        and tuple(step0_premises) == SANITY_STEP0[0]
        and step0_conclusion == SANITY_STEP0[1]
        and query == SANITY_QUERY
        and decoy is not None
        and frozenset(decoy["premises"]) == frozenset(SANITY_STEP0[0])
    )
    payload["sanity_ok"] = bool(
        structure_ok
        and row["bucket"] == "other_derived"
        and row["source_frozenset"] == "same_premise_decoy"
    )
    return payload


def _check_trace(conditions: dict, pairs: dict) -> None:
    if not TRACE_JSON.is_file():
        _stop(f"missing {TRACE_JSON}")
    previous = json.loads(TRACE_JSON.read_text(encoding="utf-8"))
    for name, block in conditions.items():
        old = previous["conditions"][name]
        for key in ("n", "n_empty_extract", *BUCKETS):
            if block[key] != old[key]:
                _stop(f"{name} {key} {block[key]} != yes_flip_trace {old[key]}")
        for family in FAMILIES:
            old_family = old["families"][family]
            new_family = block["families"][family]
            for key in ("n", "n_empty_extract", *BUCKETS):
                if new_family[key] != old_family[key]:
                    _stop(
                        f"{name}.{family} {key} {new_family[key]} != "
                        f"yes_flip_trace {old_family[key]}"
                    )
    for name, block in pairs.items():
        old = previous["pairs"][name]
        if block["n"] != old["n"]:
            _stop(f"{name} n {block['n']} != yes_flip_trace {old['n']}")
        for key in ("n_empty_extract", *BUCKETS):
            if block["left"][key] != old["left"][key]:
                _stop(f"{name} left {key} {block['left'][key]} != yes_flip_trace {old['left'][key]}")
        if block["right"]["n_since_query"] != old["right"]["query_derived"]:
            _stop(
                f"{name} right query count {block['right']['n_since_query']} != "
                f"yes_flip_trace query_derived {old['right']['query_derived']}"
            )


def _public_block(block: dict) -> dict:
    return {
        "n": block["n"],
        "n_empty_extract": block["n_empty_extract"],
        **{name: block[name] for name in BUCKETS},
        "sources": block["sources"],
        "step0_word_in_generation": block["step0_word_in_generation"],
        "families": {
            family: {
                "n": block["families"][family]["n"],
                "n_empty_extract": block["families"][family]["n_empty_extract"],
                **{name: block["families"][family][name] for name in BUCKETS},
                "sources": block["families"][family]["sources"],
                "step0_word_in_generation": block["families"][family]["step0_word_in_generation"],
            }
            for family in FAMILIES
        },
    }


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_trace()
    _self_check(trace.since_atoms)
    if list(trace.BUCKETS) != list(BUCKETS):
        _stop("bucket names drifted from yes_flip_trace")
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    seen = _index_seen(pools, trace)
    loaded = {}
    row_sets = {}
    conditions = {}
    for name, gen_rel, logic_rel in trace.CONDITIONS:
        bundle = _load_condition(name, gen_rel, logic_rel, seen)
        loaded[name] = bundle
        row_sets[name] = _flip_rows(bundle, seen, trace.since_atoms, trace.bucket_of)
        conditions[name] = _score_condition(row_sets[name])
    pairs = {}
    for pair_name, left_name, right_name in trace.PAIRS:
        scored = _score_pair(
            row_sets[left_name], right_name, loaded[right_name], seen, trace.since_atoms,
        )
        scored["left"] = {"condition": left_name, **scored["left"]}
        pairs[pair_name] = scored
    gates = _gates(row_sets["ours"], conditions["ours"], pairs["pilot_ours_wrong_b2_right"]["n"])
    sanity = _sanity(row_sets["ours"], seen)
    public_conditions = {name: _public_block(block) for name, block in conditions.items()}
    _check_trace(public_conditions, pairs)
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    payload = {
        "config": {
            "seed": SEED,
            "split": SPLIT,
            "pool": "test_seen from build_pools(seed=0)",
            "population": (
                "seen graph, latent label yes, aligned extract yes, synonym extract no"
            ),
            "empty_extract": (
                "Seen gold-yes items whose aligned extract is yes and whose synonym "
                "extract is empty. They are counted in n_empty_extract and are not "
                "placed in the four buckets or the source classes. n counts synonym extract no."
            ),
            "partition": (
                "The four buckets sum to n. The source classes, including fact and mixed, "
                "sum to other_derived. step0_word_in_generation is a subset of other_derived "
                "and does not move an item between classes."
            ),
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "yes_flip_trace.since_atoms. A line containing 'cannot be proved' "
                "contributes no derived atom."
            ),
            "buckets": {
                "query_derived": "a Since sentence derives the query atom",
                "step0_not_query": (
                    "a Since sentence derives core_rules[0]'s conclusion and none derives the query"
                ),
                "other_derived": (
                    "at least one Since sentence, and neither the step-0 conclusion nor the query is derived"
                ),
                "no_since": "no Since sentence",
            },
            "premise_equality": "frozenset of the premise atoms; order is ignored",
            "tuple_equality": (
                "Ordered premise tuples. Used only to compare with the leader's recount. "
                "Reported source counts use frozenset."
            ),
            "atom_source": {
                "same_premise_decoy": (
                    "conclusion of some other_rule whose premise set equals core_rules[0]"
                ),
                "other_decoy": (
                    "conclusion of some other_rule, and no such rule has the step-0 premise set"
                ),
                "later_core": "conclusion of core_rules[1:], and not a conclusion of other_rules",
                "fact": "in lat.facts, and not a conclusion of other_rules or core_rules[1:]",
                "unknown": "not a conclusion of those rules and not a fact",
                "mixed": (
                    "the item's Since atoms fall in more than one class. "
                    "A fact atom together with another class is mixed."
                ),
            },
            "step0_word_in_generation": (
                "Among other_derived items, the full generation contains core_rules[0]'s "
                "conclusion as a whole word, ignoring case. This count does not change the bucket "
                "or the source class."
            ),
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "pairs": {
                "pilot_ours_wrong_b2_right": (
                    "seen, gold yes, pilot ours and pilot b2 aligned extracts are yes, "
                    "ours synonym extract is no, b2 synonym extract is yes"
                ),
                "mix15_ours_wrong_b2_right": (
                    "seen, gold yes, mix15 ours and mix15 b2 aligned extracts are yes, "
                    "mix15 ours synonym extract is no, mix15 b2 synonym extract is yes"
                ),
            },
            "pair_sides": (
                "Left is ours and receives the same source classification as other_derived items. "
                "Right is b2 and is not restricted to extract no. Its three counts are headcounts "
                "of Since atoms and may overlap: step-0 conclusion, query, and a same-premise decoy conclusion."
            ),
            "loaded_model": False,
            "training": False,
            "reading": (
                "Counts of which rules the Since atoms match in saved generations. "
                "These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
            ),
        },
        "gates": gates,
        "sanity": sanity,
        "conditions": public_conditions,
        "pairs": {
            name: {
                "n": block["n"],
                "left": {
                    "condition": block["left"]["condition"],
                    "n_empty_extract": block["left"]["n_empty_extract"],
                    **{bucket: block["left"][bucket] for bucket in BUCKETS},
                    "sources": block["left"]["sources"],
                    "step0_word_in_generation": block["left"]["step0_word_in_generation"],
                },
                "right": block["right"],
            }
            for name, block in pairs.items()
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")
    print(
        f"tuple_match {gates['tuple_match']} n_items_differ {gates['n_items_differ']} "
        f"matches_leader {gates['matches_leader']} sanity_ok {sanity['sanity_ok']}"
    )
    print(
        "sanity "
        f"bucket={sanity['bucket']} source={sanity['source']} "
        f"atoms={sanity['derived_atoms']} step0={sanity['step0_premises']}->{sanity['step0_conclusion']} "
        f"query={sanity['query']} decoy={sanity['same_premise_decoy_rule']}"
    )
    for name, block in public_conditions.items():
        sources = block["sources"]
        print(
            f"{name} n={block['n']} other={block['other_derived']} "
            + " ".join(f"{key}={sources[key]}" for key in SOURCE_NAMES)
            + f" step0_word={block['step0_word_in_generation']}"
        )
        for family in FAMILIES:
            fam = block["families"][family]["sources"]
            print(
                f"  {family} "
                + " ".join(f"{key}={fam[key]}" for key in SOURCE_NAMES)
                + f" step0_word={block['families'][family]['step0_word_in_generation']}"
            )
    for name, block in payload["pairs"].items():
        left = block["left"]["sources"]
        right = block["right"]
        print(
            f"{name} n={block['n']} left "
            + " ".join(f"{key}={left[key]}" for key in SOURCE_NAMES)
            + f" step0_word={block['left']['step0_word_in_generation']} right "
            + f"step0={right['n_since_step0_conclusion']} "
            + f"query={right['n_since_query']} "
            + f"same_premise={right['n_since_same_premise_decoy']}"
        )


if __name__ == "__main__":
    main()
