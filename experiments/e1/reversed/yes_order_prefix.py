"""How far a reversed Since chain follows the core proof before it stops.

Reads saved logic.jsonl rows only. No training and no model load. This is
the reversed presentation, not a synonym rewrite, and this file does not
open generations.jsonl. The population matches yes_order_trace: a seen or
unseen graph, latent label yes, aligned extract yes, and shifted extract no.
An empty shifted extract is counted separately and does not enter the prefix.

Since atoms are yes_flip_trace.since_atoms and keep that function's order.
The first atom uses yes_flip_first_since.first_since_class. Premise equality
uses frozenset. Core conclusions stay in proof order as c0, c1, .... The
prefix length is the largest r such that c0 through c_{r-1} all occur in the
Since atoms. Comparison is lowercase set membership, so the Since sentences
themselves need not follow the proof order. The first missing core conclusion
truncates the prefix.

skipped_query counts items whose query occurs in the Since atoms while the
prefix is shorter than the core chain. Inside the same-premise-decoy first
atom subset, two further counts ask whether a later atom is the step-0
conclusion or the query. Those two counts stay inside that subset and may
overlap.

If any classified item's query is not the last core conclusion, the script
stops and names that uid. These counts are not evidence that reasoning
improved, and they are not a win for any training condition.

Writes only experiments/e1/reversed/yes_order_prefix.json.
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
            "core_chain_length": {"2": 7},
            "prefix": {"0": 1, "1": 6},
            "skipped_query": 0,
            "same_premise_first": 3,
            "same_premise_later_step0": 2,
            "same_premise_later_query": 0,
        },
        "unseen": {
            "n": 8,
            "core_chain_length": {"3": 8},
            "prefix": {"0": 1, "2": 3, "3": 4},
            "skipped_query": 0,
            "same_premise_first": 1,
            "same_premise_later_step0": 1,
            "same_premise_later_query": 1,
        },
    }
}
GATE_KEYS = (
    "n",
    "core_chain_length",
    "prefix",
    "skipped_query",
    "same_premise_first",
    "same_premise_later_step0",
    "same_premise_later_query",
)
OUT = PROJECT / "experiments" / "e1" / "reversed" / "yes_order_prefix.json"
TRACE_JSON = PROJECT / "experiments" / "e1" / "reversed" / "yes_order_trace.json"
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


def population_slot(label: str, aligned_pred: str, shifted_pred: str) -> str:
    """flip is classified; empty is counted aside; everything else is out."""
    if label != "yes" or aligned_pred != "yes":
        return "out"
    if shifted_pred == "":
        return "empty"
    if shifted_pred == "no":
        return "flip"
    return "out"


def core_conclusions(lat) -> list[str]:
    """Core-rule conclusions in proof order, lowercased."""
    return [str(conclusion).lower() for _premises, conclusion in lat.core_rules]


def query_is_last(lat) -> bool:
    """True when the query equals the last core conclusion."""
    conclusions = core_conclusions(lat)
    return bool(conclusions) and conclusions[-1] == str(lat.query).lower()


def prefix_length(atoms: list[str], conclusions: list[str]) -> int:
    """Largest r such that c0..c_{r-1} are all in the Since-atom set."""
    found = {str(atom).lower() for atom in atoms}
    length = 0
    for conclusion in conclusions:
        if str(conclusion).lower() not in found:
            break
        length += 1
    return length


def query_was_skipped(atoms: list[str], conclusions: list[str], query: str) -> bool:
    """Query is written, but an earlier core conclusion is missing."""
    found = {str(atom).lower() for atom in atoms}
    return str(query).lower() in found and prefix_length(atoms, conclusions) < len(conclusions)


def _hist(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    ordered = {key: counts[key] for key in sorted(counts, key=int)}
    if sum(ordered.values()) != len(values):
        _stop(f"histogram {ordered} does not sum to {len(values)}")
    return ordered


def _generation(row: dict, uid: str) -> str:
    text = row.get("generation")
    if not isinstance(text, str):
        _stop(f"{uid} has no generation text")
    return text


def _guard_out(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OUT.resolve() or resolved.name != "yes_order_prefix.json":
        _stop(f"refusing to write {resolved}")
    if resolved == TRACE_JSON.resolve():
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


def _self_check(since_atoms, first_since_class, later_step0_query) -> None:
    if "frozenset" not in inspect.getsource(first_since_class):
        _stop("first_since_class does not compare premises with frozenset")
    if "cannot be proved" not in inspect.getsource(since_atoms):
        _stop("since_atoms does not skip a cannot-be-proved line")
    if "atoms[1:]" not in inspect.getsource(later_step0_query):
        _stop("later counts do not use atoms after the first")
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

    chain = ["silver", "snowy", "muddy"]
    if prefix_length(["silver", "muddy"], chain) != 1:
        _stop("a missing middle conclusion did not truncate the prefix")
    if prefix_length(["muddy", "snowy", "silver"], chain) != 3:
        _stop("set membership did not ignore Since order")
    if prefix_length(["snowy", "muddy"], chain) != 0:
        _stop("a missing step-0 conclusion did not truncate at 0")
    if prefix_length([], chain) != 0:
        _stop("an empty Since chain did not have prefix 0")
    if prefix_length(["Silver"], ["silver", "snowy"]) != 1:
        _stop("prefix comparison was not lowercase")
    if prefix_length(["silver"], ["silver", "silver", "snowy"]) != 2:
        _stop("a repeated conclusion string was not satisfied by one atom")
    if not query_was_skipped(["silver", "muddy"], chain, "muddy"):
        _stop("a written query after a hole was not skipped_query")
    if query_was_skipped(["silver", "snowy", "muddy"], chain, "muddy"):
        _stop("a full prefix was counted as skipped_query")
    if query_was_skipped(["silver", "snowy"], ["silver", "snowy", "muddy"], "muddy"):
        _stop("a missing query was counted as skipped_query")
    if _hist([2, 2, 3, 2]) != {"2": 3, "3": 1}:
        _stop("histogram keys were not sorted numeric strings")
    if sum(_hist([0, 1, 1]).values()) != 3:
        _stop("histogram values did not sum to the number of items")

    last = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "snowy")),
        other_rules=((("vivid",), "windy"), (("b", "a"), "amber")),
        query="snowy",
        entity="Tina",
    )
    if not query_is_last(last):
        _stop("the query was not the last core conclusion")
    if core_conclusions(last) != ["silver", "snowy"]:
        _stop("core conclusions left proof order")
    if first_since_class(["windy", "silver", "snowy"], last) != "same_premise_decoy":
        _stop("a same-premise decoy was not same_premise_decoy")
    if later_step0_query(["windy", "silver"], last, "same_premise_decoy") != (True, False):
        _stop("a later step-0 atom was missed after a same-premise first atom")
    if later_step0_query(["windy", "silver", "snowy"], last, "same_premise_decoy") != (True, True):
        _stop("later step-0 and query were missed after a same-premise first atom")
    if later_step0_query(["windy", "snowy"], last, "same_premise_decoy") != (False, True):
        _stop("a later query was missed when step 0 was absent")
    if later_step0_query(["silver", "snowy"], last, "step0") != (False, False):
        _stop("later counts were taken outside the same-premise subset")
    if later_step0_query(["windy", "windy"], last, "same_premise_decoy") != (False, False):
        _stop("a repeated decoy atom was treated as step 0 or the query")
    if prefix_length(["windy", "snowy"], core_conclusions(last)) != 0:
        _stop("a decoy-plus-query chain did not truncate before step 0")
    if not query_was_skipped(["windy", "snowy"], core_conclusions(last), "snowy"):
        _stop("query written without step 0 was not skipped_query")
    if query_was_skipped(["windy", "silver"], core_conclusions(last), "snowy"):
        _stop("step 0 without the query was counted as skipped_query")

    swapped = SimpleNamespace(
        core_rules=((("a", "b"), "step0"), (("step0",), "query")),
        other_rules=((("b", "a"), "windy"),),
        query="query",
        entity="Tina",
    )
    if first_since_class(["windy"], swapped) != "same_premise_decoy":
        _stop("frozenset equality missed swapped premises, or later core won")
    if not query_is_last(swapped):
        _stop("swapped-premise chain lost the last-core query")

    not_last = SimpleNamespace(
        core_rules=((("vivid",), "silver"), (("silver",), "muddy")),
        other_rules=(),
        query="snowy",
        entity="Tina",
    )
    if query_is_last(not_last):
        _stop("a query that is not the last core conclusion passed")

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


def _score_split(name: str, split: str, logic: dict, pool: dict, since_atoms, first_since_class, later_step0_query) -> tuple[dict, list[str]]:
    n = 0
    n_empty = 0
    skipped = 0
    same_first = 0
    later_step0 = 0
    later_query = 0
    lengths: list[int] = []
    prefixes: list[int] = []
    bad_uids: list[str] = []
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
        if not query_is_last(lat):
            bad_uids.append(shifted_uid)
            continue
        conclusions = core_conclusions(lat)
        atoms = since_atoms(shifted_text, lat.entity)
        plen = prefix_length(atoms, conclusions)
        if plen > len(conclusions):
            _stop(f"{shifted_uid} prefix {plen} exceeds core length {len(conclusions)}")
        found = {str(atom).lower() for atom in atoms}
        query = str(lat.query).lower()
        if plen == len(conclusions) and query not in found:
            _stop(f"{shifted_uid} full prefix does not contain the query")
        kind = first_since_class(atoms, lat)
        has_step0, has_query = later_step0_query(atoms, lat, kind)
        if kind == "same_premise_decoy":
            same_first += 1
            if has_step0:
                later_step0 += 1
            if has_query:
                later_query += 1
        elif has_step0 or has_query:
            _stop(f"{shifted_uid} later counts escaped the same-premise subset")
        if query_was_skipped(atoms, conclusions, query):
            skipped += 1
        lengths.append(len(conclusions))
        prefixes.append(plen)
        n += 1
    core_hist = _hist(lengths)
    prefix_hist = _hist(prefixes)
    if sum(core_hist.values()) != n:
        _stop(f"{name}.{split} core lengths {sum(core_hist.values())} != n {n}")
    if sum(prefix_hist.values()) != n:
        _stop(f"{name}.{split} prefix histogram {sum(prefix_hist.values())} != n {n} {prefix_hist}")
    if later_step0 > same_first or later_query > same_first:
        _stop(f"{name}.{split} later counts exceed same_premise_first {same_first}")
    if skipped > n:
        _stop(f"{name}.{split} skipped_query {skipped} exceeds n {n}")
    return {
        "n": n,
        "n_empty_extract": n_empty,
        "core_chain_length": core_hist,
        "prefix": prefix_hist,
        "skipped_query": skipped,
        "same_premise_first": same_first,
        "same_premise_later_step0": later_step0,
        "same_premise_later_query": later_query,
    }, bad_uids


def _gate_cell(observed: dict, expected: dict) -> dict:
    if sum(expected["prefix"].values()) != expected["n"]:
        _stop(f"gate prefix histogram does not sum to n={expected['n']}")
    if sum(expected["core_chain_length"].values()) != expected["n"]:
        _stop(f"gate core lengths do not sum to n={expected['n']}")
    if expected["same_premise_later_step0"] > expected["same_premise_first"]:
        _stop("gate later step0 exceeds same_premise_first")
    if expected["same_premise_later_query"] > expected["same_premise_first"]:
        _stop("gate later query exceeds same_premise_first")
    obs = {key: observed[key] for key in GATE_KEYS}
    return {"ok": obs == expected, "observed": obs, "expected": expected}


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
    if TRACE_JSON.resolve() == OUT.resolve():
        _stop("population lock points at the output file")
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
            old_same = old["first"]["same_premise_decoy"]
            if new["same_premise_first"] != old_same:
                _stop(
                    f"{name}.{split} same_premise_first {new['same_premise_first']} != "
                    f"yes_order_trace {old_same}"
                )


def _fmt_hist(hist: dict) -> str:
    parts = ", ".join(f"{int(key)}: {hist[key]}" for key in hist)
    return "{" + parts + "}"


def _print_split(name: str, split: str, block: dict) -> None:
    print(
        f"{name} {split} n={block['n']} empty={block['n_empty_extract']} "
        f"core_len={_fmt_hist(block['core_chain_length'])} "
        f"prefix={_fmt_hist(block['prefix'])} skipped_query={block['skipped_query']} "
        f"same_premise_first={block['same_premise_first']} "
        f"later_step0={block['same_premise_later_step0']} "
        f"later_query={block['same_premise_later_query']}"
    )


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_module("experiments/e1/synonym/yes_flip_trace.py", "yes_flip_trace_prefix")
    first_mod = _load_module(
        "experiments/e1/synonym/yes_flip_first_since.py",
        "yes_flip_first_since_prefix",
    )
    since_atoms = trace.since_atoms
    first_since_class = first_mod.first_since_class
    later_step0_query = first_mod.later_step0_query
    _self_check(since_atoms, first_since_class, later_step0_query)
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
    bad_uids: list[str] = []
    for name, rel in CONDITIONS:
        logic = read_jsonl(_guard_logic(rel))
        _require_rows(name, logic, indexed)
        conditions[name] = {}
        for split, _pool_name, _families in SPLITS:
            block, bad = _score_split(
                name,
                split,
                logic,
                indexed[split],
                since_atoms,
                first_since_class,
                later_step0_query,
            )
            conditions[name][split] = block
            bad_uids.extend(f"{name} {uid}" for uid in bad)
            if sum(block["prefix"].values()) != block["n"]:
                _stop(f"{name}.{split} prefix histogram does not sum to n")
    if bad_uids:
        print("STOP query is not the last core conclusion; counts were not adjusted and the json was not written")
        for uid in bad_uids:
            print(f"uid {uid}")
        raise SystemExit(1)
    if "torch" in sys.modules:
        _stop("torch was imported during scoring")
    _lock_population(conditions)
    gates = _gate_report(conditions)
    for key, cell in gates.items():
        if key == "ok":
            continue
        print(
            f"gate {key} ok={cell['ok']} n={cell['observed']['n']} "
            f"expected_n={cell['expected']['n']} "
            f"prefix={_fmt_hist(cell['observed']['prefix'])} "
            f"skipped_query={cell['observed']['skipped_query']} "
            f"same_premise_first={cell['observed']['same_premise_first']} "
            f"later_step0={cell['observed']['same_premise_later_step0']} "
            f"later_query={cell['observed']['same_premise_later_query']}"
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
                "The Since chain is the shifted generation. The population matches yes_order_trace."
            ),
            "empty_extract": (
                "Gold-yes items whose aligned extract is yes and whose shifted "
                "extract is empty. They are counted in n_empty_extract and do not "
                "enter the prefix histogram. n counts shifted extract no."
            ),
            "core_chain": (
                "c0, c1, ... are the conclusions of core_rules in proof order. "
                "core_chain_length is the histogram of that length. Its values sum to n."
            ),
            "prefix": (
                "The largest r such that c0 through c_{r-1} all occur in the shifted "
                "Since atoms. Comparison is lowercase set membership. Since sentences "
                "need not follow proof order. The first missing core conclusion truncates "
                "the prefix. Prefix histogram values sum to n."
            ),
            "query_is_last": (
                "Every classified item has lat.query equal to the last core conclusion. "
                "Otherwise the script stops and names the shifted uid."
            ),
            "skipped_query": (
                "The query occurs in the Since atoms and the prefix is shorter than "
                "the core chain. A full prefix that includes the query is not counted."
            ),
            "same_premise_first": (
                "yes_flip_first_since.first_since_class returns same_premise_decoy. "
                "Premise equality is frozenset."
            ),
            "same_premise_later_step0": (
                "Among same_premise_first items, an atom after the first equals "
                "core_rules[0]'s conclusion. The first atom is not included."
            ),
            "same_premise_later_query": (
                "Among same_premise_first items, an atom after the first equals lat.query. "
                "The first atom is not included. This count may overlap later step 0."
            ),
            "regex": r"since\b[^\.\n]*,\s*{entity} is ([a-z]+)",
            "regex_flags": "IGNORECASE",
            "since_sentence": (
                "yes_flip_trace.since_atoms. Order is the order of matches. "
                "A line containing 'cannot be proved' contributes no atom."
            ),
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "population_lock": (
                "n, n_empty_extract, and same_premise_first match "
                "experiments/e1/reversed/yes_order_trace.json"
            ),
            "inputs": [rel for _name, rel in CONDITIONS],
            "conditions": [name for name, _rel in CONDITIONS],
            "gates": (
                "pilot ours seen: n=7, core_chain_length {2: 7}, prefix {0: 1, 1: 6}, "
                "skipped_query=0, same_premise_first=3, later_step0=2, later_query=0. "
                "pilot ours unseen: n=8, core_chain_length {3: 8}, prefix {0: 1, 2: 3, 3: 4}, "
                "skipped_query=0, same_premise_first=1, later_step0=1, later_query=1. "
                "A mismatch stops the script before it writes this file."
            ),
            "loaded_model": False,
            "training": False,
            "not_read": "synonym generations.jsonl",
            "reading": (
                "Counts of how far reversed Since atoms follow the core proof. "
                "These counts are not evidence that reasoning improved, "
                "and they are not a win for any training condition."
            ),
        },
        "gates": {
            "ok": gates["ok"],
            "population_matches_yes_order_trace": True,
            "pilot_ours_seen": gates["ours_seen"],
            "pilot_ours_unseen": gates["ours_unseen"],
        },
        "conditions": conditions,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = _guard_out(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.resolve() == TRACE_JSON.resolve() or "synonym" in tmp.parts:
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
