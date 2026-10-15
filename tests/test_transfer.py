"""Seen and unseen depth-2 graphs, without an ambiguous wording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from structrobust import logic, transfer, wording
from structrobust.prompts import LOGIC_INSTRUCTION
from structrobust.stats import conditional_flip_ci, mean_ci, paired_diff_ci


def _rule_lines(text: str) -> list[str]:
    rules = text.split("Rules:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    return [line for line in rules.splitlines() if line]


def _expected_gaps(seen: dict[str, np.ndarray], unseen: dict[str, np.ndarray], n_boot: int = 2000,
                   seed: int = 2, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n_seen = len(seen["aligned"])
    n_unseen = len(unseen["aligned"])
    idx_seen = rng.integers(0, n_seen, size=(n_boot, n_seen))
    idx_unseen = rng.integers(0, n_unseen, size=(n_boot, n_unseen))
    delta, boots = {}, {}
    for condition in transfer.CONDITIONS:
        left = np.asarray(seen[condition], dtype=float)
        right = np.asarray(unseen[condition], dtype=float)
        draw = left[idx_seen].mean(axis=1) - right[idx_unseen].mean(axis=1)
        point = float(left.mean() - right.mean())
        delta[condition] = [point, float(np.quantile(draw, alpha / 2)), float(np.quantile(draw, 1 - alpha / 2))]
        boots[condition] = draw
    interaction = {}
    for condition in ("reversed", "synonym"):
        draw = boots[condition] - boots["aligned"]
        point = delta[condition][0] - delta["aligned"][0]
        interaction[condition] = [
            point, float(np.quantile(draw, alpha / 2)), float(np.quantile(draw, 1 - alpha / 2))
        ]
    return delta, interaction


def _pack(pool_name: str, family: str, correct: dict[str, list[int]]):
    meta, gens = {}, {}
    n = len(correct["aligned"])
    for condition in transfer.CONDITIONS:
        assert len(correct[condition]) == n
    for i in range(n):
        task_id = f"{pool_name}{i}"
        for condition in transfer.CONDITIONS:
            uid = f"{task_id}|{condition}"
            label = "yes"
            pred = "yes" if correct[condition][i] else "no"
            meta[uid] = {
                "task_id": task_id,
                "label": label,
                "condition": condition,
                "family": family,
                "pool": pool_name,
                "depth": 2,
            }
            gens[uid] = {"uid": uid, "generation": f"Answer: {pred}"}
    return meta, gens


def test_seen_and_unseen_families_do_not_overlap():
    assert set(transfer.SEEN_FAMILIES).isdisjoint(transfer.UNSEEN_FAMILIES)
    assert transfer.SEEN_FAMILIES == ("chain", "conj_fact")
    assert transfer.UNSEEN_FAMILIES == ("tree", "diamond")
    assert transfer.TAG == "transfer"
    assert transfer.SEED == 2
    assert transfer.DEPTH == 2
    assert transfer.N_PER_FAMILY == 100
    assert transfer.VOCAB == "eval"
    assert transfer.CONDITIONS == ("aligned", "reversed", "synonym")
    assert "ambiguous" not in transfer.CONDITIONS
    assert logic.PRESENTATIONS == ("aligned", "reversed", "random", "random_distract")
    assert "synonym" not in logic.PRESENTATIONS
    pool = transfer.build_pool()
    seen = [lat for lat in pool if lat.family in transfer.SEEN_FAMILIES]
    unseen = [lat for lat in pool if lat.family in transfer.UNSEEN_FAMILIES]
    assert len(pool) == 400
    assert len(seen) == len(unseen) == 200
    assert {lat.family for lat in seen}.isdisjoint({lat.family for lat in unseen})
    assert {lat.task_id for lat in seen}.isdisjoint({lat.task_id for lat in unseen})
    assert {lat.depth for lat in pool} == {2}
    assert {lat.vocab for lat in pool} == {"eval"}
    for family in (*transfer.SEEN_FAMILIES, *transfer.UNSEEN_FAMILIES):
        group = [lat for lat in pool if lat.family == family]
        assert len(group) == 100
        assert sum(lat.label == "yes" for lat in group) == 50
        assert sum(lat.label == "no" for lat in group) == 50
    ids = {lat.task_id for lat in pool}
    for seed in (0, 1):
        for tag in ("e0", "wording"):
            old_seen = logic.make_pool(logic.SEEN_FAMILIES, [2], 100, seed, "eval", tag)
            old_unseen = logic.make_pool(logic.UNSEEN_FAMILIES, [2], 100, seed, "eval", tag)
            assert ids.isdisjoint({lat.task_id for lat in old_seen})
            assert ids.isdisjoint({lat.task_id for lat in old_unseen})
    sample = [next(lat for lat in pool if lat.family == family) for family in ("chain", "tree")]
    items, meta = transfer.build_items(sample)
    assert len(items) == 6
    assert {row["pool"] for row in meta.values()} == {"seen", "unseen"}
    assert {row["condition"] for row in meta.values()} == set(transfer.CONDITIONS)
    assert all(item["prompt"].startswith(LOGIC_INSTRUCTION) for item in items)
    assert "Reason step by step" in items[0]["prompt"]
    assert all(" likes " not in item["prompt"] for item in items)


def test_synonym_rewrite_keeps_solver_answer():
    for family in ("chain", "conj_fact", "tree", "diamond"):
        for index, label in ((0, "yes"), (1, "no")):
            seed = logic.stable_seed(transfer.TAG, transfer.SEED, family, transfer.DEPTH, index)
            lat = logic.make_latent(family, transfer.DEPTH, label, seed, transfer.VOCAB)
            rules = lat.core_rules
            others = lat.other_rules
            facts = lat.facts
            query = lat.query
            before = logic.solve(set(lat.facts), lat.all_rules(False), lat.query)

            aligned = transfer.render_condition(lat, "aligned")
            synonym = transfer.render_condition(lat, "synonym")
            assert aligned["text"] == logic.render(lat, "aligned")["text"]
            assert transfer.render_condition(lat, "reversed")["text"] == logic.render(lat, "reversed")["text"]
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
            assert synonym_lines[step0] == wording.synonym_sentence(lat.entity, rules[0])
            premises, conclusion = rules[0]
            for atom in list(premises) + [conclusion]:
                assert atom in synonym_lines[step0]
            assert " likes " not in synonym_lines[step0]
    assert logic.PRESENTATIONS == ("aligned", "reversed", "random", "random_distract")


def test_fake_vectors_give_sf_gap_and_interaction(monkeypatch):
    calls = []
    real_sf = conditional_flip_ci

    def wrapped(canon, shifted, **kwargs):
        calls.append(np.asarray(shifted, dtype=float).tolist())
        return real_sf(canon, shifted, **kwargs)

    monkeypatch.setattr(transfer, "conditional_flip_ci", wrapped)

    # Seen all right, unseen all wrong: the aligned transfer gap is +1.
    seen_right = {condition: [1, 1, 1, 1] for condition in transfer.CONDITIONS}
    unseen_wrong = {condition: [0, 0] for condition in transfer.CONDITIONS}
    meta, gens = _pack("seen", "chain", seen_right)
    extra_meta, extra_gens = _pack("unseen", "tree", unseen_wrong)
    meta.update(extra_meta)
    gens.update(extra_gens)
    out, _rows = transfer.analyze(meta, gens)
    assert out["n_latent"] == {"seen": 4, "unseen": 2}
    assert out["paired_unit"] == "latent_task"
    assert out["acc"]["seen"]["aligned"] == [1.0, 1.0, 1.0]
    assert out["acc"]["unseen"]["aligned"] == [0.0, 0.0, 0.0]
    assert out["delta_T"]["aligned"][0] == 1.0
    assert out["delta_T"]["reversed"][0] == 1.0
    assert out["delta_T"]["synonym"][0] == 1.0
    assert out["interaction"]["reversed"][0] == 0.0
    assert out["interaction"]["synonym"][0] == 0.0
    assert out["SF"]["seen"]["reversed"][0] == 0.0
    assert out["SF"]["seen"]["synonym"][0] == 0.0
    assert out["SF"]["unseen"]["reversed"] is None
    assert out["SF"]["unseen"]["synonym"] is None
    with pytest.raises(ValueError):
        paired_diff_ci(np.ones(4), np.zeros(2), seed=transfer.BOOT_SEED)

    # Nonzero flips and a negative interaction, so the subtraction direction is fixed.
    mixed_seen = {
        "aligned": [1, 1, 1, 1],
        "reversed": [1, 1, 0, 0],
        "synonym": [1, 1, 1, 0],
    }
    mixed_unseen = {
        "aligned": [1, 1, 0, 0],
        "reversed": [0, 1, 0, 0],
        "synonym": [1, 0, 0, 0],
    }
    meta, gens = _pack("seen", "chain", mixed_seen)
    extra_meta, extra_gens = _pack("unseen", "tree", mixed_unseen)
    meta.update(extra_meta)
    gens.update(extra_gens)
    out, _rows = transfer.analyze(meta, gens)
    expected_sf = {
        "seen": {
            "reversed": real_sf(np.array([1, 1, 1, 1], float), np.array([1, 1, 0, 0], float), seed=transfer.BOOT_SEED),
            "synonym": real_sf(np.array([1, 1, 1, 1], float), np.array([1, 1, 1, 0], float), seed=transfer.BOOT_SEED),
        },
        "unseen": {
            "reversed": real_sf(np.array([1, 1, 0, 0], float), np.array([0, 1, 0, 0], float), seed=transfer.BOOT_SEED),
            "synonym": real_sf(np.array([1, 1, 0, 0], float), np.array([1, 0, 0, 0], float), seed=transfer.BOOT_SEED),
        },
    }
    for pool_name in ("seen", "unseen"):
        for target in ("reversed", "synonym"):
            assert out["SF"][pool_name][target] == [float(x) for x in expected_sf[pool_name][target]]
    assert out["SF"]["seen"]["reversed"][0] == 0.5
    assert out["SF"]["seen"]["synonym"][0] == 0.25
    assert out["SF"]["unseen"]["reversed"][0] == 0.5
    assert out["SF"]["unseen"]["synonym"][0] == 0.5
    expected_delta, expected_interaction = _expected_gaps(
        {key: np.array(val, float) for key, val in mixed_seen.items()},
        {key: np.array(val, float) for key, val in mixed_unseen.items()},
        seed=transfer.BOOT_SEED,
    )
    assert out["delta_T"]["aligned"][0] == 0.5
    assert out["delta_T"]["reversed"][0] == 0.25
    assert out["delta_T"]["synonym"][0] == 0.5
    assert out["interaction"]["reversed"][0] == -0.25
    assert out["interaction"]["synonym"][0] == 0.0
    assert out["interaction"]["reversed"][0] == out["delta_T"]["reversed"][0] - out["delta_T"]["aligned"][0]
    for condition in transfer.CONDITIONS:
        assert out["delta_T"][condition] == expected_delta[condition]
        acc_seen = mean_ci(np.array(mixed_seen[condition], float), seed=transfer.BOOT_SEED)
        acc_unseen = mean_ci(np.array(mixed_unseen[condition], float), seed=transfer.BOOT_SEED)
        assert out["acc"]["seen"][condition] == [float(x) for x in acc_seen]
        assert out["acc"]["unseen"][condition] == [float(x) for x in acc_unseen]
    for condition in ("reversed", "synonym"):
        assert out["interaction"][condition] == expected_interaction[condition]
    assert len(calls) == 8


def test_refuses_banned_output_directories():
    for banned in (
        "build/e0/round1",
        "build/e0/wording",
        "build/e0/wording/nested",
        "experiments/e1",
        "experiments/e1/pilot",
    ):
        with pytest.raises(SystemExit):
            transfer._guard_out(Path(banned))
    allowed = transfer._guard_out(Path("build/e0/transfer"))
    assert allowed.name.lower() == "transfer"
