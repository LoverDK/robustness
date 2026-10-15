"""CPU checks for the mix15 ours synonym control. No model load."""

from __future__ import annotations

import sys

import pytest

from structrobust.e1_synonym_mix15_ours import (
    GEN_PATH,
    LOG_PATH,
    METRICS_PATH,
    MIX15_OURS_SANITY,
    PROJECT,
    RESCORE_KEY,
    assert_mix15_ours_control_numbers,
    guard_mix15_ours_output,
)


def test_import_does_not_load_torch():
    assert "torch" not in sys.modules


def _matching_control():
    block = {"quant": "nf4", "SF": {}, "acc": {}, "counts": {}}
    recomputed = {}
    for split, sanity in MIX15_OURS_SANITY.items():
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


def test_control_numbers_accept_the_stated_mix15_ours_values():
    recomputed, block = _matching_control()
    assert_mix15_ours_control_numbers(recomputed, block)


def test_control_numbers_stop_when_aligned_accuracy_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["aligned_acc"][0] = 0.89
    with pytest.raises(RuntimeError, match="aligned acc"):
        assert_mix15_ours_control_numbers(recomputed, block)


def test_control_numbers_stop_when_rescore_sf_differs():
    recomputed, block = _matching_control()
    block["SF"]["unseen"] = [0.5, 0.4, 0.6]
    with pytest.raises(RuntimeError):
        assert_mix15_ours_control_numbers(recomputed, block)


def test_control_numbers_stop_when_flip_count_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["n_flip"] = 19
    with pytest.raises(RuntimeError, match="counts"):
        assert_mix15_ours_control_numbers(recomputed, block)


def test_control_numbers_stop_when_the_interval_is_dropped():
    recomputed, block = _matching_control()
    recomputed["seen"]["reversed_SF"] = []
    with pytest.raises(RuntimeError, match="dropped"):
        assert_mix15_ours_control_numbers(recomputed, block)


def test_stated_sanity_matches_the_mix15_ours_record():
    assert RESCORE_KEY == ("rounds", "pilot_mix15", "conditions", "ours")
    assert MIX15_OURS_SANITY["seen"]["aligned_acc"] == 0.895
    assert MIX15_OURS_SANITY["unseen"]["aligned_acc"] == 0.785
    assert MIX15_OURS_SANITY["seen"]["reversed_SF"] == (
        0.1005586592178771,
        0.05714285714285714,
        0.14606741573033707,
    )
    assert MIX15_OURS_SANITY["unseen"]["reversed_SF"] == (
        0.20382165605095542,
        0.14093959731543623,
        0.2709677419354839,
    )
    assert MIX15_OURS_SANITY["seen"]["n_aligned_correct"] == 179
    assert MIX15_OURS_SANITY["seen"]["n_flip"] == 18
    assert MIX15_OURS_SANITY["unseen"]["n_aligned_correct"] == 157
    assert MIX15_OURS_SANITY["unseen"]["n_flip"] == 32


def test_guard_allows_only_mix15_ours_outputs():
    assert "torch" not in sys.modules
    guard_mix15_ours_output(GEN_PATH)
    guard_mix15_ours_output(METRICS_PATH)
    guard_mix15_ours_output(LOG_PATH)
    guard_mix15_ours_output(METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".tmp"))
    banned = [
        PROJECT / "build" / "e1" / "synonym" / "b0" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b1" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b3" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b1" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b3" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "ours" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_ours" / "rows.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_ours" / "run.json",
        PROJECT / "experiments" / "e1" / "synonym" / "metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "b1_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "b3_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "ours_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "mix15_b1_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "mix15_b2_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "mix15_b3_metrics.json",
        PROJECT / "build" / "e1" / "pilot" / "ours" / "logic.jsonl",
        PROJECT / "build" / "e1" / "pilot_mix15" / "ours" / "logic.jsonl",
        PROJECT / "experiments" / "e1" / "rescore" / "metrics.json",
        PROJECT / "build" / "logs" / "e1_synonym.log",
        PROJECT / "build" / "logs" / "e1_synonym_ours.log",
        PROJECT / "build" / "logs" / "e1_synonym_mix15_b1.log",
    ]
    for path in banned:
        with pytest.raises(SystemExit):
            guard_mix15_ours_output(path)
