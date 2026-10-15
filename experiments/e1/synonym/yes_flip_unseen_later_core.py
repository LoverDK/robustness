"""Premises and later Since atoms when an unseen synonym flip starts on a later core rule.

Reads saved generations only. No training and no model load. The population
is an unseen graph, latent label yes, aligned extract yes, synonym extract
no, whose first synonym Since atom falls in later_core. The class order is
yes_flip_first_since_unseen.first_since_class: step0, then query, then
same_premise_decoy, then later_core. Since atoms are
yes_flip_trace.since_atoms and keep that function's order.

The later core rule is the single core_rules[1:] entry with that conclusion.
Two or more core rules with the same conclusion stop the script. All of that
rule's premises must sit in lat.facts to count as premises_subset_of_facts.
Step 0 and the query are sought only in Since atoms after the first.

On the same items, the aligned generation adds two overlapping headcounts:
whether its first Since atom is the step-0 conclusion, and whether that atom
is this same later-core conclusion. Those two counts are not a partition.

These counts are not evidence that reasoning improved, and they are not a
win for any training condition.

Writes only experiments/e1/synonym/yes_flip_unseen_later_core.json.
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
    "n_later_first",
    "premises_subset_of_facts",
    "later_has_step0",
    "later_has_query",
    "aligned_first_step0",
    "aligned_first_same_later",
)
GATE_KEYS = (
    "n_later_first",
    "premises_subset_of_facts",
    "later_has_step0",
    "later_has_query",
)
GATES = {
    "ours": {
        "n_later_first": 13,
        "premises_subset_of_facts": 13,
        "later_has_step0": 3,
        "later_has_query": 0,
    }
}
OUT = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_unseen_later_core.json"
PRIOR = PROJECT / "experiments" / "e1" / "synonym" / "yes_flip_first_since_unseen.json"
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
    block["n_later_first"] += 1
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


def core_matches(atom: str, lat) -> list[tuple[int, tuple, str]]:
    """Every core rule whose conclusion equals atom, in core_rules order."""
    wanted = str(atom).lower()
    found = []
    for index, (premises, conclusion) in enumerate(lat.core_rules):
        if str(conclusion).lower() == wanted:
            found.append((index, tuple(premises), str(conclusion)))
    return found


def rule_error(atom: str, lat, where: str) -> str | None:
    """Error text when the later-core conclusion is not exactly one later core rule."""
    matches = core_matches(atom, lat)
    later = [item for item in matches if item[0] >= 1]
    if len(matches) == 1 and len(later) == 1:
        return None
    lines = [
        (
            f"{where} atom {str(atom).lower()} matches {len(matches)} core rule(s), "
            f"{len(later)} after step 0."
        )
    ]
    if not matches:
        lines.append("  no core rule has this conclusion")
    for index, premises, conclusion in matches:
        shown = tuple(str(premise) for premise in premises)
        lines.append(f"  core_rules[{index}] premises={shown} conclusion={conclusion}")
    return "\n".join(lines)


def premises_in_facts(premises, lat) -> bool:
    facts = {str(fact).lower() for fact in lat.facts}
    return all(str(premise).lower() in facts for premise in premises)


def later_step0_query(atoms: list[str], lat) -> tuple[bool, bool]:
    """Step 0 and the query among Since atoms after the first. The first atom is excluded."""
    step0 = str(lat.core_rules[0][1]).lower()
    query = str(lat.query).lower()
    rest = {str(atom).lower() for atom in atoms[1:]}
    return step0 in rest, query in rest


def aligned_flags(atoms: list[str], lat, conclusion: str) -> tuple[bool, bool]:
    """Whether the first aligned atom is step 0, and whether it is this later conclusion."""
    if not atoms:
        return False, False
    first = str(atoms[0]).lower()
    step0 = str(lat.core_rules[0][1]).lower()
    return first == step0, first == str(conclusion).lower()


def _check_block(label: str, block: dict) -> None:
    n = block["n_later_first"]
    for key in COUNT_KEYS:
        if key == "n_later_first":
            continue
        if block[key] > n:
            _stop(f"{label} {key}={block[key]} exceeds n_later_first={n}")


def _self_check(since_atoms, first_since_class) -> None:
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
    if first_since_class(["amber"], chain) != "other_decoy":
        _stop("a different-premise decoy was not other_decoy")
    if rule_error("muddy", chain, "self-check") is not None:
        _stop("the unique later core rule was rejected")
    if not premises_in_facts(("bright",), chain):
        _stop("a premise that is a fact was missed")
    if premises_in_facts(("silver",), chain):
        _stop("a derived premise was treated as a fact")
    if not premises_in_facts(("bright", "vivid"), chain):
        _stop("a two-premise rule whose premises are facts was missed")
    if premises_in_facts(("vivid", "silver"), chain):
        _stop("a two-premise rule with one missing premise was accepted")
    if later_step0_query(["muddy", "silver", "snowy"], chain) != (True, True):
        _stop("later step 0 and query were missed")
    if later_step0_query(["muddy"], chain) != (False, False):
        _stop("an absent later atom was counted")
    if later_step0_query(["silver", "snowy"], chain) != (False, True):
        _stop("the first atom was counted as a later step-0 conclusion")
    if aligned_flags(["silver", "muddy"], chain, "muddy") != (True, False):
        _stop("aligned step 0 was missed")
    if aligned_flags(["muddy"], chain, "muddy") != (False, True):
        _stop("the same later-core conclusion was missed on the aligned side")
    if aligned_flags([], chain, "muddy") != (False, False):
        _stop("a missing aligned Since sentence was counted")
    if aligned_flags(["windy"], chain, "muddy") != (False, False):
        _stop("a decoy first atom was counted as step 0 or the later conclusion")
    both = SimpleNamespace(
        core_rules=((("vivid",), "silver"),),
        other_rules=(),
        query="later",
        facts=("vivid",),
        entity="Tina",
    )
    if aligned_flags(["silver"], both, "silver") != (True, True):
        _stop("overlapping aligned headcounts did not both fire")

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
    if "('bright',)" not in message or "('dim',)" not in message:
        _stop(f"the ambiguous rules were not named:\n{message}")


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


def _score_condition(bundle: dict, unseen: dict, since_atoms, first_since_class) -> tuple[dict, list[str]]:
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
        _index, premises, conclusion = core_matches(synonym_atoms[0], lat)[0]
        if str(conclusion).lower() != str(synonym_atoms[0]).lower():
            _stop(f"{where} resolved conclusion does not equal the first Since atom")
        has_step0, has_query = later_step0_query(synonym_atoms, lat)
        aligned_atoms = since_atoms(aligned_text, lat.entity)
        is_step0, is_same = aligned_flags(aligned_atoms, lat, conclusion)
        flags = {
            "premises_subset_of_facts": premises_in_facts(premises, lat),
            "later_has_step0": has_step0,
            "later_has_query": has_query,
            "aligned_first_step0": is_step0,
            "aligned_first_same_later": is_same,
        }
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
        _stop(f"missing prior classification {PRIOR}")
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    conditions = prior.get("conditions")
    if not isinstance(conditions, dict):
        _stop("prior classification has no conditions")
    for name, block in public.items():
        if name not in conditions:
            _stop(f"prior classification has no {name}")
        expected = conditions[name]["synonym"]["later_core"]
        if block["n_later_first"] != expected:
            _stop(
                f"{name} n_later_first {block['n_later_first']} != "
                f"prior synonym later_core {expected}"
            )
        for family in FAMILIES:
            prev = conditions[name]["families"][family]["synonym"]["later_core"]
            got = block["families"][family]["n_later_first"]
            if got != prev:
                _stop(f"{name}.{family} n_later_first {got} != prior synonym later_core {prev}")


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


def main() -> None:
    if "torch" in sys.modules:
        _stop("torch is already imported")
    trace = _load_module("yes_flip_trace.py", "yes_flip_trace_later_core")
    prior_mod = _load_module(
        "yes_flip_first_since_unseen.py",
        "yes_flip_first_since_unseen_later_core",
    )
    _self_check(trace.since_atoms, prior_mod.first_since_class)
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
            bundle, unseen, trace.since_atoms, prior_mod.first_since_class
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
        for name in public:
            _print_block(name, public[name])
            for family in FAMILIES:
                _print_block(f"  {family}", public[name]["families"][family])
        raise SystemExit(1)
    payload = {
        "config": {
            "seed": SEED,
            "split": SPLIT,
            "pool": "test_unseen from build_pools(seed=0)",
            "population": (
                "unseen graph, latent label yes, aligned extract yes, synonym extract no, "
                "and the first synonym Since atom is later_core"
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
                "one core rule, the script stops and writes nothing."
            ),
            "premises_subset_of_facts": (
                "Every premise of that unique later core rule is a member of lat.facts. "
                "Comparison is case-insensitive. Premise order is ignored because this "
                "is membership, not tuple equality."
            ),
            "later_has_step0": (
                "A Since atom after the first equals core_rules[0]'s conclusion. "
                "The first atom is not included."
            ),
            "later_has_query": (
                "A Since atom after the first equals lat.query. The first atom is not "
                "included. This count may overlap later_has_step0."
            ),
            "aligned_first_step0": (
                "On the same items, the aligned generation's first Since atom equals "
                "core_rules[0]'s conclusion."
            ),
            "aligned_first_same_later": (
                "On the same items, the aligned generation's first Since atom equals "
                "the conclusion of the same later core rule. This count may overlap "
                "aligned_first_step0. The two aligned counts are not a partition of "
                "n_later_first."
            ),
            "families": list(FAMILIES),
            "family_rule": (
                "tree and diamond each repeat the same six counts. Their sums equal "
                "the condition."
            ),
            "answer_extract": "structrobust.prompts.extract_yes_no on the generation only",
            "duplicate_uids": "e1.read_jsonl keeps the last row for a repeated uid",
            "conditions": [name for name, _gen, _logic in trace.CONDITIONS],
            "prior_lock": (
                "n_later_first, including tree and diamond, must equal synonym "
                "later_core in experiments/e1/synonym/yes_flip_first_since_unseen.json. "
                "That file is only read."
            ),
            "gates": (
                "pilot ours n_later_first=13, premises_subset_of_facts=13, "
                "later_has_step0=3, later_has_query=0. A mismatch stops the script "
                "before it writes this file. Counts are not edited to meet the gate."
            ),
            "loaded_model": False,
            "training": False,
            "reading": (
                "Counts, on unseen graphs, of whether a later core rule that opens "
                "a synonym yes-to-no flip has its premises among the known facts, "
                "and whether a later Since atom is the step-0 conclusion or the query. "
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
            "file": "experiments/e1/synonym/yes_flip_first_since_unseen.json",
            "compared": "synonym later_core at the condition and at tree and diamond",
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
    for name, block in public.items():
        _print_block(name, block)
        for family in FAMILIES:
            _print_block(f"  {family}", block["families"][family])


if __name__ == "__main__":
    main()
