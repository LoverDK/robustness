"""Solver, presentation equivalence, burden features, and pool isolation."""

from __future__ import annotations

import itertools

import pytest

from structrobust import logic
from structrobust.burden import burden, burden_score
from structrobust.prompts import extract_number, extract_yes_no
from structrobust.stats import conditional_flip_ci, mean_ci, paired_diff_ci


FAMILIES = logic.SEEN_FAMILIES + logic.UNSEEN_FAMILIES
DEPTHS = (2, 3, 4, 6, 8)


def _signature(lat: logic.Latent):
    rules = []
    for group in (lat.core_rules, lat.other_rules, lat.extra_distractors):
        for premises, conclusion in group:
            rules.append((tuple(premises), conclusion))
    return (lat.entity, tuple(lat.facts), tuple(sorted(rules)), lat.query)


def _atoms(lat: logic.Latent) -> set[str]:
    atoms = set(lat.facts) | {lat.query}
    for group in (lat.core_rules, lat.other_rules, lat.extra_distractors):
        for premises, conclusion in group:
            atoms.update(premises)
            atoms.add(conclusion)
    return atoms


def _rule_lines(lat: logic.Latent, presentation: str) -> list[str]:
    text = logic.render(lat, presentation)["text"]
    rules = text.split("Rules:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    return [line for line in rules.splitlines() if line]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("label", ("yes", "no"))
def test_solver_matches_declared_label(family, depth, label):
    lat = logic.make_latent(family, depth, label, seed=logic.stable_seed(family, depth, label))
    got = logic.solve(set(lat.facts), lat.all_rules(False), lat.query)
    got_extra = logic.solve(
        set(lat.facts) | set(lat.meta["extra_facts"]),
        lat.all_rules(True),
        lat.query,
    )
    assert got is (label == "yes")
    assert got_extra is got
    assert lat.query not in lat.facts


def test_yes_and_no_keep_the_same_rule_count():
    for family, depth in itertools.product(FAMILIES, (2, 4, 8)):
        yes = logic.make_latent(family, depth, "yes", seed=11)
        no = logic.make_latent(family, depth, "no", seed=11)
        assert len(yes.all_rules(False)) == len(no.all_rules(False))
        assert len(yes.facts) == len(no.facts)


def test_equivalent_presentations_do_not_change_the_answer_or_the_rule_multiset():
    for family, depth, label, i in itertools.product(FAMILIES, (2, 5, 8), ("yes", "no"), range(4)):
        lat = logic.make_latent(family, depth, label, seed=logic.stable_seed("eq", family, depth, label, i))
        base = None
        for presentation in ("aligned", "reversed", "random"):
            lines = _rule_lines(lat, presentation)
            bag = tuple(sorted(lines))
            if base is None:
                base = bag
            else:
                assert bag == base
            rendered = logic.render(lat, presentation)
            assert lat.label in {"yes", "no"}
            # The question atom is the solver label, independent of order.
            assert logic.solve(set(lat.facts), lat.all_rules(False), lat.query) is (lat.label == "yes")
            assert rendered["n_rules"] == len(lines)
        distract = _rule_lines(lat, "random_distract")
        assert set(base) < set(distract)
        assert len(distract) == len(base) + len(lat.extra_distractors)


def test_burden_on_a_yes_chain():
    lat = logic.make_latent("chain", 6, "yes", seed=3)
    aligned = logic.render(lat, "aligned")["core_positions"]
    reversed_pos = logic.render(lat, "reversed")["core_positions"]
    m = len(lat.core_rules)
    b_al = burden(lat, aligned)
    b_re = burden(lat, reversed_pos)
    assert b_al["back_reads"] == 0
    assert b_re["back_reads"] == m * (m - 1) / 2
    # A proof-aligned chain discharges every rule as it is read, so the
    # unresolved frontier stays empty. The reversed chain holds every later
    # step until the first one is read, so the peak frontier is m - 1.
    assert b_al["f_active"] == 0
    assert b_re["f_active"] == m - 1
    assert b_al["d_rel"] >= 1
    assert burden_score(b_al) < burden_score(b_re)
    # Random order is a permutation of the same core rules.
    random_pos = logic.render(lat, "random")["core_positions"]
    assert sorted(random_pos) == sorted(aligned)


def test_stability_and_transfer_pools_do_not_overlap():
    depths = [2, 4, 6]
    stability = logic.make_pool(logic.SEEN_FAMILIES, depths, n_per_cell=8, seed=0, vocab="eval", tag="stability")
    transfer = logic.make_pool(logic.UNSEEN_FAMILIES, depths, n_per_cell=8, seed=0, vocab="eval", tag="transfer")
    assert logic.SEEN_FAMILIES and logic.UNSEEN_FAMILIES
    assert set(logic.SEEN_FAMILIES).isdisjoint(logic.UNSEEN_FAMILIES)
    assert set(ADJ for ADJ in logic.ADJ_TRAIN).isdisjoint(logic.ADJ_EVAL)
    assert set(logic.NAMES_TRAIN).isdisjoint(logic.NAMES_EVAL)

    stab_ids = {lat.task_id for lat in stability}
    trans_ids = {lat.task_id for lat in transfer}
    assert len(stab_ids) == len(stability)
    assert stab_ids.isdisjoint(trans_ids)

    stab_fam = {lat.family for lat in stability}
    trans_fam = {lat.family for lat in transfer}
    assert stab_fam == set(logic.SEEN_FAMILIES)
    assert trans_fam == set(logic.UNSEEN_FAMILIES)
    assert stab_fam.isdisjoint(trans_fam)

    stab_sig = {_signature(lat) for lat in stability}
    trans_sig = {_signature(lat) for lat in transfer}
    assert len(stab_sig) == len(stability)
    assert stab_sig.isdisjoint(trans_sig)

    # Train and eval inventories cannot leak into each other.
    train_pool = logic.make_pool(logic.SEEN_FAMILIES, [3], n_per_cell=4, seed=2, vocab="train", tag="train")
    eval_pool = logic.make_pool(logic.UNSEEN_FAMILIES, [3], n_per_cell=4, seed=2, vocab="eval", tag="eval")
    train_atoms = set().union(*(_atoms(lat) for lat in train_pool))
    eval_atoms = set().union(*(_atoms(lat) for lat in eval_pool))
    train_names = {lat.entity for lat in train_pool}
    eval_names = {lat.entity for lat in eval_pool}
    assert train_atoms.isdisjoint(eval_atoms)
    assert train_names.isdisjoint(eval_names)
    assert {_signature(lat) for lat in train_pool}.isdisjoint({_signature(lat) for lat in eval_pool})


def test_pool_labels_are_balanced():
    pool = logic.make_pool(logic.SEEN_FAMILIES, [4], n_per_cell=10, seed=5, vocab="eval", tag="bal")
    labels = [lat.label for lat in pool]
    assert labels.count("yes") == labels.count("no") == 10


def test_answer_extraction_and_paired_intervals():
    assert extract_yes_no("because yes is tempting\nAnswer: no") == "no"
    assert extract_yes_no("Answer: **true**") == "yes"
    assert extract_yes_no("the derivation is unfinished") == ""
    assert extract_number("total is 3\nAnswer: 42") == "42"
    mean, lo, hi = mean_ci([1, 1, 1, 1], n_boot=200, seed=0)
    assert mean == 1 and lo == 1 and hi == 1
    diff, dlo, dhi = paired_diff_ci([1, 0, 1, 0], [1, 0, 1, 0], n_boot=200, seed=0)
    assert diff == 0 and dlo == 0 and dhi == 0
    sf, sf_lo, sf_hi = conditional_flip_ci([1, 1, 1, 1], [1, 0, 1, 0], n_boot=400, seed=0)
    assert sf == 0.5
    assert sf_lo < sf < sf_hi
