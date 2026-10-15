"""Since-sentence trace for synonym answers that flip gold yes to no.

Reads saved generations only. No training and no model load. A Since sentence
is a case-insensitive match of

    since\\b[^\\.\\n]*,\\s*{entity} is ([a-z]+)

on a line that does not contain "cannot be proved". The captured atom is what
that sentence derives. This file records those counts. It does not say that
reasoning improved, and it does not treat any condition as a method win.

Writes only experiments/e1/synonym/yes_flip_trace.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl
from structrobust.prompts import extract_yes_no

SEED = 0
SPLIT = "seen"
BUCKETS = ("query_derived", "step0_not_query", "other_derived", "no_since")
FAMILIES = ("chain", "conj_fact")
SANITY_HASH = "099cd1bf088e74592cba2fd9c8b6303c1b89361f8fe3f673939bfb6394ab30ad"
SANITY_CHARS = 138
CONDITIONS = (
    ("b0", "build/e1/synonym/b0/generations.jsonl", "build/e1/pilot/b0/logic.jsonl"),
    ("b1", "build/e1/synonym/b1/generations.jsonl", "build/e1/pilot/b1/logic.jsonl"),
    ("b2", "build/e1/synonym/b2/generations.jsonl", "build/e1/pilot/b2/logic.jsonl"),
    ("b3", "build/e1/synonym/b3/generations.jsonl", "build/e1/pilot/b3/logic.jsonl"),
    ("ours", "build/e1/synonym/ours/generations.jsonl", "build/e1/pilot/ours/logic.jsonl"),
    (
        "mix15_b1",
        "build/e1/synonym/mix15_b1/generations.jsonl",
        "build/e1/pilot_mix15/b1/logic.jsonl",
    ),
    (
        "mix15_b2",
        "build/e1/synonym/mix15_b2/generations.jsonl",
        "build/e1/pilot_mix15/b2/logic.jsonl",
    ),
    (
        "mix15_b3",
        "build/e1/synonym/mix15_b3/generations.jsonl",
        "build/e1/pilot_mix15/b3/logic.jsonl",
    ),
    (
        "mix15_ours",
        "build/e1/synonym/mix15_ours/generations.jsonl",
        "build/e1/pilot_mix15/ours/logic.jsonl",
    ),
)
PAIRS = (
    ("pilot_ours_wrong_b2_right", "ours", "b2"),
    ("mix15_ours_wrong_b2_right", "mix15_ours", "mix15_b2"),
)
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_trace.json"
BANNED_NAMES = {
    "generations.jsonl",
    "logic.jsonl",
    "metrics.json",
    "label_paired.json",
    "label_sf.json",
    "label_sf_reversed.json",
    "yes_item_overlap.json",
    "synonym_burden_contrast.json",
    "burden_contrast.json",
    "report.md",
}


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _since_pattern(entity: str) -> re.Pattern[str]:
    return re.compile(
        rf"since\b[^\.\n]*,\s*{re.escape(entity)} is ([a-z]+)",
        re.IGNORECASE,
    )


def since_atoms(generation: str, entity: str) -> list[str]:
    """Atoms derived by Since sentences. A cannot-be-proved line derives nothing."""
    pattern = _since_pattern(entity)
    atoms: list[str] = []
    for line in generation.splitlines():
        if "cannot be proved" in line.lower():
            continue
        atoms.extend(match.group(1).lower() for match in pattern.finditer(line))
    return atoms


def bucket_of(atoms: list[str], step0: str, query: str) -> str:
    found = set(atoms)
    if query in found:
        return "query_derived"
    if step0 in found:
        return "step0_not_query"
    if atoms:
        return "other_derived"
    return "no_since"


def _blank() -> dict:
    return {name: 0 for name in BUCKETS}


def _mean(values: list[int]):
    if not values:
        return None
    return float(sum(values) / len(values))


def _block(n: int, n_empty: int, counts: dict, since_counts: list[int]) -> dict:
    return {
        "n": n,
        "n_empty_extract": n_empty,
        **counts,
        "since_count_mean": _mean(since_counts),
    }


def _check_partition(label: str, block: dict) -> None:
    total = sum(block[name] for name in BUCKETS) + block["n_empty_extract"]
    if total != block["n"]:
        _stop(
            f"{label} partition failed: buckets+empty={total} "
            f"n_extract_no={block['n']} counts={ {name: block[name] for name in BUCKETS} } "
            f"empty={block['n_empty_extract']}"
        )


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


def _self_check() -> None:
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
    if bucket_of(["windy"], "windy", "snowy") != "step0_not_query":
        _stop("bucket self-check failed for step 0")
    if bucket_of(["windy", "snowy"], "windy", "snowy") != "query_derived":
        _stop("bucket self-check failed when the query is also derived")
    if bucket_of(["muddy"], "windy", "snowy") != "other_derived":
        _stop("bucket self-check failed for another atom")
    if bucket_of([], "windy", "snowy") != "no_since":
        _stop("bucket self-check failed for no Since sentence")


def _index_seen(pools: dict) -> dict:
    seen = {}
    for lat in pools["test_seen"]:
        digest = graph_hash(lat)
        if digest in seen:
            _stop(f"duplicate seen graph hash {digest}")
        if lat.family not in FAMILIES:
            _stop(f"{digest} family {lat.family} is outside the seen families")
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
        if synonym_uid not in synonym:
            missing_synonym.append(synonym_uid)
        elif not isinstance(synonym[synonym_uid].get("generation"), str):
            missing_synonym.append(synonym_uid)
        if aligned_uid not in logic:
            missing_aligned.append(aligned_uid)
        elif not isinstance(logic[aligned_uid].get("generation"), str):
            missing_aligned.append(aligned_uid)
    if missing_synonym:
        listed = "\n".join(missing_synonym)
        _stop(f"{name} synonym generations missing {len(missing_synonym)} uid(s):\n{listed}")
    if missing_aligned:
        listed = "\n".join(missing_aligned)
        _stop(f"{name} aligned generations missing {len(missing_aligned)} uid(s):\n{listed}")
    return {"name": name, "synonym": synonym, "logic": logic}


def _score_condition(bundle: dict, seen: dict) -> dict:
    counts = _blank()
    family_counts = {family: _blank() for family in FAMILIES}
    since_counts: list[int] = []
    family_since = {family: [] for family in FAMILIES}
    n = 0
    n_empty = 0
    family_n = {family: 0 for family in FAMILIES}
    family_empty = {family: 0 for family in FAMILIES}
    for digest, lat in seen.items():
        if lat.label != "yes":
            continue
        aligned_uid = f"{SPLIT}_aligned|{digest}"
        synonym_uid = f"{SPLIT}_synonym|{digest}"
        aligned_pred = extract_yes_no(_generation(bundle["logic"][aligned_uid], aligned_uid))
        synonym_text = _generation(bundle["synonym"][synonym_uid], synonym_uid)
        synonym_pred = extract_yes_no(synonym_text)
        if aligned_pred != "yes":
            continue
        family = lat.family
        if synonym_pred == "":
            n_empty += 1
            family_empty[family] += 1
            continue
        if synonym_pred != "no":
            continue
        step0 = str(lat.core_rules[0][1]).lower()
        query = str(lat.query).lower()
        atoms = since_atoms(synonym_text, lat.entity)
        bucket = bucket_of(atoms, step0, query)
        counts[bucket] += 1
        family_counts[family][bucket] += 1
        since_counts.append(len(atoms))
        family_since[family].append(len(atoms))
        n += 1
        family_n[family] += 1
    block = _block(n, n_empty, counts, since_counts)
    block["families"] = {
        family: _block(family_n[family], family_empty[family], family_counts[family], family_since[family])
        for family in FAMILIES
    }
    _check_partition(bundle["name"], block)
    for family in FAMILIES:
        _check_partition(f"{bundle['name']}.{family}", block["families"][family])
    return block


def _score_pair(left_name: str, right_name: str, loaded: dict, seen: dict) -> dict:
    left = loaded[left_name]
    right = loaded[right_name]
    n = 0
    sides = {
        "left": {"counts": _blank(), "since": [], "empty": 0},
        "right": {"counts": _blank(), "since": [], "empty": 0},
    }
    hashes = []
    for digest, lat in seen.items():
        if lat.label != "yes":
            continue
        preds = {}
        texts = {}
        ok = True
        for name, bundle in ((left_name, left), (right_name, right)):
            aligned_uid = f"{SPLIT}_aligned|{digest}"
            synonym_uid = f"{SPLIT}_synonym|{digest}"
            aligned_pred = extract_yes_no(_generation(bundle["logic"][aligned_uid], aligned_uid))
            synonym_text = _generation(bundle["synonym"][synonym_uid], synonym_uid)
            synonym_pred = extract_yes_no(synonym_text)
            if aligned_pred != "yes":
                ok = False
                break
            preds[name] = synonym_pred
            texts[name] = synonym_text
        if not ok:
            continue
        if preds[left_name] == "yes" or preds[right_name] != "yes":
            continue
        step0 = str(lat.core_rules[0][1]).lower()
        query = str(lat.query).lower()
        for side, name in (("left", left_name), ("right", right_name)):
            pred = preds[name]
            if pred == "":
                sides[side]["empty"] += 1
                continue
            atoms = since_atoms(texts[name], lat.entity)
            bucket = bucket_of(atoms, step0, query)
            sides[side]["counts"][bucket] += 1
            sides[side]["since"].append(len(atoms))
        n += 1
        hashes.append(digest)
    out = {"n": n, "graph_hashes": hashes}
    for side, name in (("left", left_name), ("right", right_name)):
        counts = sides[side]["counts"]
        empty = sides[side]["empty"]
        block = {
            "condition": name,
            "n_empty_extract": empty,
            **counts,
            "since_count_mean": _mean(sides[side]["since"]),
        }
        total = sum(counts.values()) + empty
        if total != n:
            _stop(f"pair {left_name} vs {right_name} {side} partition failed: {total} != {n}")
        out[side] = block
    return out


def _sanity(loaded: dict, seen: dict, pairs: dict) -> dict:
    pair_name = "pilot_ours_wrong_b2_right"
    in_pair = SANITY_HASH in pairs[pair_name]["graph_hashes"]
    lat = seen.get(SANITY_HASH)
    if lat is None:
        return {
            "graph_hash": SANITY_HASH,
            "pair": pair_name,
            "in_pair": False,
            "generation_chars": None,
            "ends_with_answer_no": False,
            "n_since": None,
            "derived_atoms": None,
            "step0_conclusion": None,
            "query": None,
            "entity": None,
            "bucket": None,
            "sanity_ok": False,
        }
    uid = f"{SPLIT}_synonym|{SANITY_HASH}"
    text = _generation(loaded["ours"]["synonym"][uid], uid)
    step0 = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    atoms = since_atoms(text, lat.entity)
    bucket = bucket_of(atoms, step0, query)
    ends = text.strip().endswith("Answer: no")
    chars_ok = abs(len(text) - SANITY_CHARS) <= 8
    one_intermediate = len(atoms) == 1 and atoms[0] == step0 and step0 != query
    sanity_ok = bool(in_pair and ends and chars_ok and one_intermediate and bucket == "step0_not_query")
    return {
        "graph_hash": SANITY_HASH,
        "pair": pair_name,
        "in_pair": in_pair,
        "generation_chars": len(text),
        "ends_with_answer_no": ends,
        "n_since": len(atoms),
        "derived_atoms": atoms,
        "step0_conclusion": step0,
        "query": query,
        "entity": lat.entity,
        "bucket": bucket,
        "sanity_ok": sanity_ok,
    }


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    _self_check()
    pools, summary = build_pools(seed=SEED)
    if not summary.get("ok"):
        _stop("pool isolation failed")
    seen = _index_seen(pools)
    loaded = {}
    conditions = {}
    for name, gen_rel, logic_rel in CONDITIONS:
        bundle = _load_condition(name, gen_rel, logic_rel, seen)
        loaded[name] = bundle
        conditions[name] = _score_condition(bundle, seen)
    pairs = {}
    for pair_name, left_name, right_name in PAIRS:
        pairs[pair_name] = _score_pair(left_name, right_name, loaded, seen)
    sanity = _sanity(loaded, seen, pairs)
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
                "placed in the four buckets. n counts synonym extract no."
            ),
            "partition": "sum of the four buckets plus n_empty_extract equals n",
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "A match of the regex on one generation line. A line containing "
                "'cannot be proved' contributes neither a derived atom nor a Since count."
            ),
            "buckets": {
                "query_derived": "a Since sentence derives the query atom",
                "step0_not_query": "a Since sentence derives core_rules[0]'s conclusion and none derives the query",
                "other_derived": "at least one Since sentence, and neither the step-0 conclusion nor the query is derived",
                "no_since": "no Since sentence",
            },
            "step0_conclusion": "conclusion atom of lat.core_rules[0]",
            "query": "lat.query",
            "entity": "lat.entity",
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "pairs": {
                "pilot_ours_wrong_b2_right": (
                    "seen, gold yes, pilot ours and pilot b2 aligned extracts are yes, "
                    "ours synonym extract is not yes, b2 synonym extract is yes"
                ),
                "mix15_ours_wrong_b2_right": (
                    "seen, gold yes, mix15 ours and mix15 b2 aligned extracts are yes, "
                    "mix15 ours synonym extract is not yes, mix15 b2 synonym extract is yes"
                ),
            },
            "pair_sides": "left is ours, right is b2; the right side is not restricted to extract no",
            "loaded_model": False,
            "training": False,
            "reading": (
                "Counts of Since sentences in saved generations. "
                "These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
            ),
        },
        "sanity": sanity,
        "conditions": conditions,
        "pairs": {
            name: {key: value for key, value in block.items() if key != "graph_hashes"}
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
    print(f"sanity_ok {sanity['sanity_ok']}")
    print(
        "sanity "
        f"in_pair={sanity['in_pair']} chars={sanity['generation_chars']} "
        f"atoms={sanity['derived_atoms']} step0={sanity['step0_conclusion']} "
        f"query={sanity['query']} bucket={sanity['bucket']}"
    )
    for name, block in conditions.items():
        print(
            f"{name} n={block['n']} empty={block['n_empty_extract']} "
            + " ".join(f"{bucket}={block[bucket]}" for bucket in BUCKETS)
            + f" since_mean={block['since_count_mean']}"
        )
    for name, block in payload["pairs"].items():
        left = block["left"]
        right = block["right"]
        print(
            f"{name} n={block['n']} "
            f"left({' '.join(f'{bucket}={left[bucket]}' for bucket in BUCKETS)}) "
            f"right({' '.join(f'{bucket}={right[bucket]}' for bucket in BUCKETS)})"
        )


if __name__ == "__main__":
    main()
