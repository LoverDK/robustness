"""CPU checks for the B1 synonym control. No model load."""

from __future__ import annotations

import sys

import pytest

from structrobust.e1_synonym import interval_relation, sf_interval
from structrobust.e1_synonym_b1 import (
    B1_SANITY,
    GEN_PATH,
    LOG_PATH,
    METRICS_PATH,
    PROJECT,
    assert_b1_control_numbers,
    family_conditional_sf,
    guard_b1_output,
    reversed_control_record,
)


def test_import_does_not_load_torch():
    assert "torch" not in sys.modules


def test_reversed_control_record_counts_flips():
    record = reversed_control_record([1, 1, 1, 0], [0, 1, 0, 1])
    assert record["n"] == 4
    assert record["n_aligned_correct"] == 3
    assert record["n_flip"] == 2
    assert record["n_stay"] == 1
    assert record["aligned_acc"][0] == pytest.approx(0.75)
    assert record["reversed_SF"][0] == pytest.approx(2 / 3)
    assert record["reversed_SF"] == sf_interval([1, 1, 1, 0], [0, 1, 0, 1])


def _matching_control():
    block = {"quant": "nf4", "SF": {}, "acc": {}, "counts": {}}
    recomputed = {}
    for split, sanity in B1_SANITY.items():
        aligned = [sanity["aligned_acc"], 0.0, 1.0]
        reversed_acc = [0.5, 0.4, 0.6]
        reversed_sf = list(sanity["reversed_SF"])
        block["SF"][split] = list(reversed_sf)
        block["acc"][split] = {"aligned": list(aligned), "reversed": list(reversed_acc)}
        block["counts"][split] = {
            "n": 200,
            "n_aligned_correct": sanity["n_aligned_correct"],
            "n_flip": sanity["n_flip"],
            "n_stay": sanity["n_aligned_correct"] - sanity["n_flip"],
        }
        recomputed[split] = {
            "aligned_acc": list(aligned),
            "reversed_acc": list(reversed_acc),
            "reversed_SF": list(reversed_sf),
            "n_aligned_correct": sanity["n_aligned_correct"],
            "n_flip": sanity["n_flip"],
            "n_stay": sanity["n_aligned_correct"] - sanity["n_flip"],
        }
    return recomputed, block


def test_control_numbers_accept_the_stated_b1_values():
    recomputed, block = _matching_control()
    assert_b1_control_numbers(recomputed, block)


def test_control_numbers_stop_when_aligned_accuracy_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["aligned_acc"][0] = 0.96
    with pytest.raises(RuntimeError, match="aligned acc"):
        assert_b1_control_numbers(recomputed, block)


def test_control_numbers_stop_when_rescore_sf_differs():
    recomputed, block = _matching_control()
    block["SF"]["unseen"] = [0.5, 0.4, 0.6]
    with pytest.raises(RuntimeError):
        assert_b1_control_numbers(recomputed, block)


def _family_rows():
    specs = {
        "chain": ("seen", [1, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]),
        "conj_fact": ("seen", [1, 0, 1], [1, 0, 0], [0, 1, 1]),
        "tree": ("unseen", [1, 1, 0], [0, 1, 1], [1, 0, 0]),
        "diamond": ("unseen", [1, 1], [1, 0], [0, 1]),
    }
    rows = []
    for family, (split, aligned, synonym, reversed_) in specs.items():
        for index, (a_ok, s_ok, r_ok) in enumerate(zip(aligned, synonym, reversed_)):
            rows.append({
                "split": split,
                "family": family,
                "graph_hash": f"{family}-{index:02d}",
                "aligned_correct": a_ok,
                "synonym_correct": s_ok,
                "reversed_correct": r_ok,
                "synonym_extract_fail": int(index == 0 and family == "diamond"),
            })
    return rows


def test_family_sf_bootstraps_in_graph_hash_order():
    rows = list(reversed(_family_rows()))
    got = family_conditional_sf(rows)
    chain = sorted(
        (row for row in rows if row["family"] == "chain"),
        key=lambda row: row["graph_hash"],
    )
    aligned = [row["aligned_correct"] for row in chain]
    synonym = [row["synonym_correct"] for row in chain]
    reversed_ = [row["reversed_correct"] for row in chain]
    assert got["chain"]["synonym_SF"] == sf_interval(aligned, synonym)
    assert got["chain"]["reversed_SF"] == sf_interval(aligned, reversed_)
    assert got["chain"]["synonym_SF"][0] == pytest.approx(2 / 3)
    assert got["chain"]["pool"] == "seen"
    assert got["tree"]["pool"] == "unseen"
    assert got["diamond"]["pool"] == "unseen"
    relation = interval_relation(got["chain"]["synonym_SF"], got["chain"]["reversed_SF"])
    assert got["chain"]["interval_relation"] == relation
    assert got["chain"]["overlaps_reversed_SF"] is (relation == "重叠")
    assert got["diamond"]["synonym_extract_fail"] == 1
    assert set(got) == {"chain", "conj_fact", "tree", "diamond"}


def test_family_sf_rejects_missing_or_crossed_families():
    missing = [row for row in _family_rows() if row["family"] != "diamond"]
    with pytest.raises(RuntimeError, match="empty family"):
        family_conditional_sf(missing)
    crossed = _family_rows()
    crossed[0]["split"] = "unseen"
    with pytest.raises(RuntimeError, match="spans"):
        family_conditional_sf(crossed)
    unknown = _family_rows()
    unknown.append(dict(unknown[-1], family="other", graph_hash="other-00"))
    with pytest.raises(ValueError):
        family_conditional_sf(unknown)


def test_guard_allows_only_b1_outputs():
    assert "torch" not in sys.modules
    guard_b1_output(GEN_PATH)
    guard_b1_output(METRICS_PATH)
    guard_b1_output(LOG_PATH)
    guard_b1_output(METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".tmp"))
    banned = [
        PROJECT / "build" / "e1" / "synonym" / "b0" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b1" / "rows.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b1" / "run.json",
        PROJECT / "experiments" / "e1" / "synonym" / "metrics.json",
        PROJECT / "build" / "e1" / "pilot" / "b1" / "logic.jsonl",
        PROJECT / "experiments" / "e1" / "pilot" / "b1" / "metrics.json",
        PROJECT / "experiments" / "e1" / "rescore" / "metrics.json",
        PROJECT / "build" / "logs" / "e1_synonym.log",
    ]
    for path in banned:
        with pytest.raises(SystemExit):
            guard_b1_output(path)
