"""Whether a correct reversed answer adds the query only after its premises are known.

Reads saved logic.jsonl rows only. No training and no model load. This is
the reversed presentation, not a synonym rewrite, and this file does not
open generations.jsonl. The population is a seen or unseen graph whose latent
label is yes, whose aligned extract is yes, and whose shifted extract is yes.
An empty shifted extract is counted separately and does not enter the four
classes. A shifted extract of no is outside this population.

Parsing is yes_order_justified.parse_generation. This file does not redefine it.
Aligned and shifted generations are parsed separately. The only cross is
aligned justified and reversed justified. These counts are not evidence that
reasoning improved, and they are not a win for C2 or any training condition.

Writes only experiments/e1/reversed/yes_order_correct.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from yes_order_justified import (  # noqa: E402
    CLASSES,
    CONDITIONS,
    POOL_N,
    SEED,
    SPLITS,
    parse_generation,
)
from structrobust.e1 import PROJECT, build_pools, graph_hash, read_jsonl  # noqa: E402
from structrobust.prompts import extract_yes_no  # noqa: E402

CROSS_KEY = "aligned_justified_reversed_justified"
EXPECTED_CONDITIONS = (
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
EXPECTED_SPLITS = (
    ("seen", "test_seen", ("chain", "conj_fact")),
    ("unseen", "test_unseen", ("tree", "diamond")),
)
GATES = {
    "ours": {
        "seen": {
            "n": 70,
            "reversed": {
                "justified": 70,
                "early": 0,
                "never": 0,
                "no_since": 0,
                "cannot_query": 0,
            },
            "aligned": {"justified": 70, "early": 0, "never": 0, "no_since": 0},
            "cross": {CROSS_KEY: 70},
        },
        "unseen": {
            "n": 41,
            "reversed": {
                "justified": 37,
                "early": 4,
                "never": 0,
                "no_since": 0,
                "cannot_query": 0,
            },
            "aligned": {"justified": 39, "early": 2, "never": 0, "no_since": 0},
            "cross": {CROSS_KEY: 37},
        },
    },
    "b2": {
        "seen": {
            "n": 83,
            "reversed": {
                "justified": 81,
                "early": 0,
                "never": 2,
                "no_since": 0,
                "cannot_query": 0,
            },
            "aligned": {"justified": 81, "early": 0, "never": 2, "no_since": 0},
            "cross": {CROSS_KEY: 80},
        },
        "unseen": {
            "n": 71,
            "reversed": {
                "justified": 60,
                "early": 9,
                "never": 2,
                "no_since": 0,
                "cannot_query": 0,
            },
            "aligned": {"justified": 57, "early": 11, "never": 3, "no_since": 0},
            "cross": {CROSS_KEY: 53},
        },
    },
    "b1": {
        "seen": {
            "n": 48,
            "reversed": {
                "justified": 48,
                "early": 0,
                "never": 0,
                "no_since": 0,
                "cannot_query": 0,
            },
            "aligned": {"justified": 48, "early": 0, "never": 0, "no_since": 0},
            "cross": {CROSS_KEY: 48},
        },
        "unseen": {
            "n": 36,
            "reversed": {
                "justified": 35,
                "early": 1,
                "never": 0,
                "no_since": 0,
                "cannot_query": 0,
            },
            "aligned": {"justified": 34, "early": 2, "never": 0, "no_since": 0},
            "cross": {CROSS_KEY: 34},
        },
    },
}
OUT = PROJECT / "experiments" / "e1" / "reversed" / "yes_order_correct.json"
PROTECTED = {
    PROJECT / "experiments" / "e1" / "reversed" / "yes_order_justified.json",
    PROJECT / "experiments" / "e1" / "reversed" / "yes_order_justified.py",
    PROJECT / "experiments" / "e1" / "reversed" / "yes_order_prefix.json",
    PROJECT / "experiments" / "e1" / "reversed" / "yes_order_prefix.py",
    PROJECT / "experiments" / "e1" / "reversed" / "yes_order_trace.json",
    PROJECT / "experiments" / "e1" / "reversed" / "yes_order_trace.py",
}
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
    "yes_order_justified.json",
    "yes_order_trace.py",
    "yes_order_prefix.py",
    "yes_order_justified.py",
}


def _stop(msg: str) -> None:
    raise SystemExit(msg)


def _blank_classes() -> dict:
    return {name: 0 for name in CLASSES}


def population_slot(label: str, aligned_pred: str, shifted_pred: str) -> str:
    """correct is classified; empty is counted aside; reversed no is out."""
    if label != "yes" or aligned_pred != "yes":
        return "out"
    if shifted_pred == "":
        return "empty"
    if shifted_pred == "yes":
        return "correct"
    return "out"


def _generation(row: dict, uid: str) -> str:
    text = row.get("generation")
    if not isinstance(text, str):
        _stop(f"{uid} has no generation text")
    return text


def _guard_out(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OUT.resolve() or resolved.name != "yes_order_correct.json":
        _stop(f"refusing to write {resolved}")
    if resolved in {item.resolve() for item in PROTECTED}:
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
    if parse_generation.__module__ != "yes_order_justified":
        _stop(f"parse_generation came from {parse_generation.__module__}")
    proved = parse_generation(
        "Since Tina is vivid, Tina is snowy.\n",
        "Tina",
        "snowy",
        ("vivid",),
    )
    _expect(proved["class"], "justified", "the imported parser missed a ready premise")
    _expect(proved["cannot_query"], False, "the imported parser marked a proved line cannot_query")
    if set(proved) != {"class", "cannot_query"}:
        _stop(f"parse_generation returned {sorted(proved)}")

    if population_slot("yes", "yes", "yes") != "correct":
        _stop("a reversed yes was not classified")
    if population_slot("yes", "yes", "") != "empty":
        _stop("an empty reversed extract entered classification")
    if population_slot("yes", "yes", "no") != "out":
        _stop("a reversed no entered the correct population")
    if population_slot("no", "yes", "yes") != "out":
        _stop("a gold-no item entered the population")
    if population_slot("yes", "no", "yes") != "out":
        _stop("an aligned no entered the population")
    if population_slot("yes", "", "yes") != "out":
        _stop("an empty aligned extract entered the population")

    if CONDITIONS != EXPECTED_CONDITIONS:
        _stop("conditions differ from yes_order_justified.py")
    if SPLITS != EXPECTED_SPLITS:
        _stop("splits differ from yes_order_justified.py")
    if SEED != 0 or POOL_N != 200:
        _stop("seed or pool size differs from yes_order_justified.py")
    if CLASSES != ("justified", "early", "never", "no_since"):
        _stop("classes differ from yes_order_justified.py")
    if len(CONDITIONS) != 9:
        _stop(f"expected 9 conditions, found {len(CONDITIONS)}")

    for name, splits in GATES.items():
        if name not in {item[0] for item in CONDITIONS}:
            _stop(f"gate names an unknown condition {name}")
        for split, expected in splits.items():
            if split not in {item[0] for item in SPLITS}:
                _stop(f"gate names an unknown split {split}")
            rev = expected["reversed"]
            ali = expected["aligned"]
            if sum(rev[key] for key in CLASSES) != expected["n"]:
                _stop(f"gate {name}.{split} reversed classes do not sum to n")
            if sum(ali[key] for key in CLASSES) != expected["n"]:
                _stop(f"gate {name}.{split} aligned classes do not sum to n")
            if rev["cannot_query"] > expected["n"]:
                _stop(f"gate {name}.{split} cannot_query exceeds n")
            both = expected["cross"][CROSS_KEY]
            if both > min(ali["justified"], rev["justified"]):
                _stop(f"gate {name}.{split} both-justified exceeds its margins")
            if set(expected["cross"]) != {CROSS_KEY}:
                _stop(f"gate {name}.{split} records more than one cross")


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
    if set(cross) != {CROSS_KEY}:
        _stop(f"{name}.{split} records more than one cross")
    if sum(rev[key] for key in CLASSES) != n:
        _stop(f"{name}.{split} reversed classes {rev} do not sum to n {n}")
    if sum(ali[key] for key in CLASSES) != n:
        _stop(f"{name}.{split} aligned classes {ali} do not sum to n {n}")
    if rev["cannot_query"] > n:
        _stop(f"{name}.{split} cannot_query {rev['cannot_query']} exceeds n {n}")
    both = cross[CROSS_KEY]
    if both > min(ali["justified"], rev["justified"]):
        _stop(f"{name}.{split} both-justified {both} exceeds its margins")
    if rev["justified"] == n and ali["justified"] == n and both != n:
        _stop(f"{name}.{split} both sides are justified, but the cross is {both}")


def _score_split(name: str, split: str, logic: dict, pool: dict) -> dict:
    n = 0
    n_empty = 0
    reversed_counts = _blank_classes()
    aligned_counts = _blank_classes()
    cannot_query = 0
    both_justified = 0
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
        if aligned_parsed["class"] == "justified" and reversed_parsed["class"] == "justified":
            both_justified += 1
        n += 1
    block = {
        "n": n,
        "n_empty_extract": n_empty,
        "reversed": {**reversed_counts, "cannot_query": cannot_query},
        "aligned": aligned_counts,
        "cross": {CROSS_KEY: both_justified},
    }
    _check_partition(name, split, block)
    return block


def _gate_cell(observed: dict, expected: dict) -> dict:
    rev_ok = all(observed["reversed"][key] == expected["reversed"][key] for key in (*CLASSES, "cannot_query"))
    ali_ok = all(observed["aligned"][key] == expected["aligned"][key] for key in CLASSES)
    cross_ok = observed["cross"][CROSS_KEY] == expected["cross"][CROSS_KEY]
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


def _print_split(name: str, split: str, block: dict) -> None:
    rev = block["reversed"]
    ali = block["aligned"]
    both = block["cross"][CROSS_KEY]
    print(
        f"{name} {split} n={block['n']} empty={block['n_empty_extract']} "
        f"reversed justified={rev['justified']} early={rev['early']} "
        f"never={rev['never']} no_since={rev['no_since']} cannot_query={rev['cannot_query']} "
        f"aligned justified={ali['justified']} early={ali['early']} "
        f"never={ali['never']} no_since={ali['no_since']} both_justified={both}"
    )


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    _self_check()
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
                "latent label yes, aligned extract yes, shifted extract yes. "
                "This is the reversed-correct population. It is not the reversed-no "
                "population in yes_order_justified, and it is not a synonym rewrite."
            ),
            "empty_extract": (
                "Gold-yes items whose aligned extract is yes and whose shifted "
                "extract is empty. They are counted in n_empty_extract and do not "
                "enter the four classes. n counts shifted extract yes."
            ),
            "parser": "yes_order_justified.parse_generation",
            "line": "The generation is split on lines. A line is one step or none.",
            "cannot_line": (
                "A line whose lowercase text contains 'cannot be proved' produces "
                "no step. If it matches '{entity} is {query} cannot be proved' "
                "with re.IGNORECASE and re.escape on entity and query, the "
                "generation is cannot_query. cannot_query is counted on the reversed "
                "generation only and is not one of the four classes."
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
                CROSS_KEY: "aligned class justified and reversed class justified",
            },
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "inputs": [rel for _name, rel in CONDITIONS],
            "conditions": [name for name, _rel in CONDITIONS],
            "gates": (
                "pilot ours seen: n=70, reversed justified 70, cannot_query 0, "
                "aligned justified 70, both justified 70. "
                "pilot ours unseen: n=41, reversed justified 37, early 4, cannot_query 0, "
                "aligned justified 39, early 2, both justified 37. "
                "b2 seen: n=83, reversed justified 81, never 2, cannot_query 0, "
                "aligned justified 81, never 2, both justified 80. "
                "b2 unseen: n=71, reversed justified 60, early 9, never 2, cannot_query 0, "
                "aligned justified 57, early 11, never 3, both justified 53. "
                "b1 seen: n=48, reversed justified 48, cannot_query 0, aligned justified 48, "
                "both justified 48. "
                "b1 unseen: n=36, reversed justified 35, early 1, cannot_query 0, "
                "aligned justified 34, early 2, both justified 34. "
                "Classes not listed in a cell are 0 because the listed classes already sum to n. "
                "A mismatch stops the script before it writes this file."
            ),
            "loaded_model": False,
            "training": False,
            "not_read": "synonym generations.jsonl",
            "reading": (
                "Counts of whether a Since step adds the query after its premises "
                "are known, among items whose reversed extract is yes, and whether "
                "that reversed generation says the query cannot be proved. These "
                "counts are not evidence that reasoning improved, and they are not "
                "a win for C2 or any training condition."
            ),
        },
        "gates": {
            "ok": gates["ok"],
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
    if tmp.resolve() in {item.resolve() for item in PROTECTED} or tmp.name in BANNED_NAMES:
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
