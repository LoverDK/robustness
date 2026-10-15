"""CPU checks for the mix15 B1 synonym control. No model load."""

from __future__ import annotations

import sys

import pytest

from structrobust.e1_synonym_mix15_b1 import (
    GEN_PATH,
    LOG_PATH,
    METRICS_PATH,
    MIX15_B1_SANITY,
    PROJECT,
    assert_mix15_b1_control_numbers,
    flip_partition,
    guard_mix15_b1_output,
)


def test_import_does_not_load_torch():
    assert "torch" not in sys.modules


def _matching_control():
    block = {"quant": "nf4", "SF": {}, "acc": {}, "counts": {}}
    recomputed = {}
    for split, sanity in MIX15_B1_SANITY.items():
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


def test_control_numbers_accept_the_stated_mix15_b1_values():
    recomputed, block = _matching_control()
    assert_mix15_b1_control_numbers(recomputed, block)


def test_control_numbers_stop_when_aligned_accuracy_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["aligned_acc"][0] = 0.96
    with pytest.raises(RuntimeError, match="aligned acc"):
        assert_mix15_b1_control_numbers(recomputed, block)


def test_control_numbers_stop_when_rescore_sf_differs():
    recomputed, block = _matching_control()
    block["SF"]["unseen"] = [0.5, 0.4, 0.6]
    with pytest.raises(RuntimeError):
        assert_mix15_b1_control_numbers(recomputed, block)


def test_control_numbers_stop_when_flip_count_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["n_flip"] = 44
    with pytest.raises(RuntimeError, match="counts"):
        assert_mix15_b1_control_numbers(recomputed, block)


def test_stated_sanity_matches_the_mix15_b1_rescore_record():
    assert MIX15_B1_SANITY["seen"]["aligned_acc"] == 0.965
    assert MIX15_B1_SANITY["unseen"]["aligned_acc"] == 0.8
    assert MIX15_B1_SANITY["seen"]["reversed_SF"] == (
        0.22279792746113988,
        0.16666666666666666,
        0.28350515463917525,
    )
    assert MIX15_B1_SANITY["unseen"]["reversed_SF"] == (
        0.325,
        0.25308641975308643,
        0.4024409418622488,
    )
    assert MIX15_B1_SANITY["seen"]["n_aligned_correct"] == 193
    assert MIX15_B1_SANITY["seen"]["n_flip"] == 43
    assert MIX15_B1_SANITY["unseen"]["n_aligned_correct"] == 160
    assert MIX15_B1_SANITY["unseen"]["n_flip"] == 52


def test_flip_partition_counts_four_cells_among_aligned_correct():
    cells = flip_partition(
        [1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1],
    )
    assert cells == {
        "both_flipped": 1,
        "only_reversed_flipped": 1,
        "only_synonym_flipped": 1,
        "neither_flipped": 1,
    }
    assert sum(cells.values()) == 4


def test_flip_partition_rejects_nonbinary_values():
    with pytest.raises(ValueError, match="0/1"):
        flip_partition([1], [0.5], [0])


def test_guard_allows_only_mix15_b1_outputs():
    assert "torch" not in sys.modules
    guard_mix15_b1_output(GEN_PATH)
    guard_mix15_b1_output(METRICS_PATH)
    guard_mix15_b1_output(LOG_PATH)
    guard_mix15_b1_output(METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".tmp"))
    banned = [
        PROJECT / "build" / "e1" / "synonym" / "b0" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b1" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b1" / "rows.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b1" / "run.json",
        PROJECT / "experiments" / "e1" / "synonym" / "metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "b1_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "mix15_b2_metrics.json",
        PROJECT / "build" / "e1" / "pilot" / "b1" / "logic.jsonl",
        PROJECT / "build" / "e1" / "pilot_mix15" / "b1" / "logic.jsonl",
        PROJECT / "experiments" / "e1" / "pilot" / "b1" / "metrics.json",
        PROJECT / "experiments" / "e1" / "pilot_mix15" / "b1" / "metrics.json",
        PROJECT / "experiments" / "e1" / "rescore" / "metrics.json",
        PROJECT / "build" / "logs" / "e1_synonym.log",
        PROJECT / "build" / "logs" / "e1_synonym_b1.log",
        PROJECT / "build" / "logs" / "e1_synonym_mix15_b2.log",
    ]
    for path in banned:
        with pytest.raises(SystemExit):
            guard_mix15_b1_output(path)
