"""Surface rewrites of proof step 0. These stay out of the global presentation list."""

from __future__ import annotations

import re

import numpy as np

from structrobust import logic
from structrobust.prompts import LOGIC_INSTRUCTION
from structrobust.stats import conditional_flip_ci
from structrobust import wording


_CONNECTIVE = re.compile(r"\b(?:if|then|whenever)\b", re.I)
_SYN_BINARY = re.compile(
    r"^(.+) is always (\S+) whenever \1 is (\S+) and \1 is (\S+)\.$"
)
_SYN_UNARY = re.compile(r"^(.+) is always (\S+) whenever \1 is (\S+)\.$")


def _rule_lines(text: str) -> list[str]:
    rules = text.split("Rules:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    return [line for line in rules.splitlines() if line]


def _parse_synonym(sentence: str):
    binary = _SYN_BINARY.match(sentence)
    if binary:
        _entity, conclusion, left, right = binary.groups()
        return (left, right), conclusion
    unary = _SYN_UNARY.match(sentence)
    assert unary, sentence
    _entity, conclusion, premise = unary.groups()
    return (premise,), conclusion


def _depth2(family: str, label: str, index: int) -> logic.Latent:
    seed = logic.stable_seed(wording.TAG, wording.SEED, family, wording.DEPTH, index)
    return logic.make_latent(family, wording.DEPTH, label, seed, wording.VOCAB)


def test_presentations_are_not_extended():
    assert logic.PRESENTATIONS == ("aligned", "reversed", "random", "random_distract")
    assert "synonym" not in logic.PRESENTATIONS
    assert "ambiguous" not in logic.PRESENTATIONS


def test_synonym_templates_keep_both_arities():
    unary = wording.synonym_sentence("Kara", (("amber",), "coral"))
    binary = wording.synonym_sentence("Kara", (("amber", "azure"), "coral"))
    assert unary == "Kara is always coral whenever Kara is amber."
    assert binary == "Kara is always coral whenever Kara is amber and Kara is azure."
    assert _parse_synonym(unary) == (("amber",), "coral")
    assert _parse_synonym(binary) == (("amber", "azure"), "coral")


def test_ambiguous_templates_drop_the_implication():
    for rule in ((("amber",), "coral"), (("amber", "azure"), "coral")):
        sentence = wording.ambiguous_sentence("Kara", rule)
        assert sentence == "Kara likes coral."
        assert _CONNECTIVE.search(sentence) is None
        assert "amber" not in sentence
        assert "azure" not in sentence


def test_synonym_keeps_solver_answer_and_symbol_rules():
    for family in ("chain", "conj_fact"):
        for index, label in ((0, "yes"), (1, "no")):
            lat = _depth2(family, label, index)
            rules = lat.core_rules
            others = lat.other_rules
            facts = lat.facts
            query = lat.query
            before = logic.solve(set(lat.facts), lat.all_rules(False), lat.query)

            aligned = wording.render_condition(lat, "aligned")
            synonym = wording.render_condition(lat, "synonym")
            assert aligned["text"] == logic.render(lat, "aligned")["text"]
            assert wording.render_condition(lat, "reversed")["text"] == logic.render(lat, "reversed")["text"]

            assert lat.core_rules is rules
            assert lat.other_rules is others
            assert lat.facts is facts
            assert lat.query is query
            after = logic.solve(set(lat.facts), lat.all_rules(False), lat.query)
            assert before is after is (label == "yes")

            aligned_lines = _rule_lines(aligned["text"])
            synonym_lines = _rule_lines(synonym["text"])
            step0 = aligned["rule_seq"].index(("core", 0))
            changed = [i for i, (a, b) in enumerate(zip(aligned_lines, synonym_lines)) if a != b]
            assert changed == [step0]
            assert len(synonym_lines) == len(aligned_lines)
            assert _parse_synonym(synonym_lines[step0]) == rules[0]
            assert synonym_lines[step0] == wording.synonym_sentence(lat.entity, rules[0])
            # Later core rules, including a dual-premise step, stay in the aligned If/then form.
            assert synonym_lines[aligned["rule_seq"].index(("core", 1))] == aligned_lines[
                aligned["rule_seq"].index(("core", 1))
            ]


def test_ambiguous_sentence_has_no_conditional_words():
    for family in ("chain", "conj_fact"):
        for index, label in ((0, "yes"), (1, "no")):
            lat = _depth2(family, label, index)
            rules = lat.core_rules
            aligned = wording.render_condition(lat, "aligned")
            ambiguous = wording.render_condition(lat, "ambiguous")
            aligned_lines = _rule_lines(aligned["text"])
            ambiguous_lines = _rule_lines(ambiguous["text"])
            step0 = aligned["rule_seq"].index(("core", 0))
            changed = [i for i, (a, b) in enumerate(zip(aligned_lines, ambiguous_lines)) if a != b]
            assert changed == [step0]
            sentence = ambiguous_lines[step0]
            assert sentence == f"{lat.entity} likes {rules[0][1]}."
            assert _CONNECTIVE.search(sentence) is None
            assert lat.core_rules is rules
            assert logic.solve(set(lat.facts), lat.all_rules(False), lat.query) is (label == "yes")


def test_ambiguous_is_excluded_from_structural_flip(monkeypatch):
    calls = []
    real = conditional_flip_ci

    def wrapped(canon, shifted, **kwargs):
        calls.append(np.asarray(shifted, dtype=float).tolist())
        return real(canon, shifted, **kwargs)

    monkeypatch.setattr(wording, "conditional_flip_ci", wrapped)
    labels = ["yes", "yes", "yes", "yes"]
    # aligned all right; reversed flips 2 of 4; synonym flips 1 of 4; ambiguous all wrong.
    preds = {
        "aligned": ["yes", "yes", "yes", "yes"],
        "reversed": ["yes", "no", "yes", "no"],
        "synonym": ["yes", "yes", "no", "yes"],
        "ambiguous": ["no", "no", "no", "no"],
    }
    meta, gens = {}, {}
    for i, label in enumerate(labels):
        task_id = f"t{i}"
        for condition in wording.CONDITIONS:
            uid = f"{task_id}|{condition}"
            meta[uid] = {
                "task_id": task_id,
                "label": label,
                "condition": condition,
                "family": "chain",
                "depth": 2,
            }
            gens[uid] = {"uid": uid, "generation": f"Answer: {preds[condition][i]}"}
    out, _rows = wording.analyze(meta, gens)
    assert set(out["SF"]) == {"reversed", "synonym"}
    assert "ambiguous" not in out["SF"]
    assert len(calls) == 2
    assert out["SF"]["reversed"][0] == 0.5
    assert out["SF"]["synonym"][0] == 0.25
    assert out["ambiguous"]["entailment_broken"] is True
    assert out["ambiguous"]["scored_against"] == "original_solver_label"
    assert out["acc"]["ambiguous"][0] == 0.0
    assert out["n_latent"] == 4
    assert out["paired_unit"] == "latent_task"
    expected_reversed = real(
        np.array([1, 1, 1, 1], dtype=float),
        np.array([1, 0, 1, 0], dtype=float),
        seed=wording.BOOT_SEED,
    )
    expected_synonym = real(
        np.array([1, 1, 1, 1], dtype=float),
        np.array([1, 1, 0, 1], dtype=float),
        seed=wording.BOOT_SEED,
    )
    assert out["SF"]["reversed"] == [float(x) for x in expected_reversed]
    assert out["SF"]["synonym"] == [float(x) for x in expected_synonym]


def test_pool_is_the_wording_split_not_the_e0_pool():
    pool = wording.build_pool()
    assert wording.TAG == "wording"
    assert wording.SEED == 1
    assert wording.FAMILIES == ("chain", "conj_fact")
    assert wording.DEPTH == 2
    assert wording.N_PER_FAMILY == 100
    assert wording.VOCAB == "eval"
    assert len(pool) == 200
    assert {lat.family for lat in pool} == {"chain", "conj_fact"}
    assert {lat.depth for lat in pool} == {2}
    assert {lat.vocab for lat in pool} == {"eval"}
    assert sum(lat.label == "yes" for lat in pool) == 100
    assert sum(lat.label == "no" for lat in pool) == 100
    for family in wording.FAMILIES:
        group = [lat for lat in pool if lat.family == family]
        assert len(group) == 100
        assert sum(lat.label == "yes" for lat in group) == 50
    old = logic.make_pool(logic.SEEN_FAMILIES, [2], 100, 0, "eval", "e0")
    seed0 = logic.make_pool(wording.FAMILIES, [2], 100, 0, "eval", "wording")
    assert {lat.task_id for lat in pool}.isdisjoint({lat.task_id for lat in old})
    assert {lat.task_id for lat in pool}.isdisjoint({lat.task_id for lat in seed0})
    items, _meta = wording.build_items(pool[:1])
    assert len(items) == 4
    assert all(item["prompt"].startswith(LOGIC_INSTRUCTION) for item in items)
    assert "Reason step by step" in items[0]["prompt"]
