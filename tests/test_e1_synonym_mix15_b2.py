"""CPU checks for the mix15 B2 synonym control. No model load."""

from __future__ import annotations

import sys

import pytest

from structrobust.e1_synonym_mix15_b2 import (
    GEN_PATH,
    LOG_PATH,
    METRICS_PATH,
    MIX15_B2_SANITY,
    PROJECT,
    assert_mix15_b2_control_numbers,
    guard_mix15_b2_output,
)


def test_import_does_not_load_torch():
    assert "torch" not in sys.modules


def _matching_control():
    block = {"quant": "nf4", "SF": {}, "acc": {}, "counts": {}}
    recomputed = {}
    for split, sanity in MIX15_B2_SANITY.items():
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


def test_control_numbers_accept_the_stated_mix15_b2_values():
    recomputed, block = _matching_control()
    assert_mix15_b2_control_numbers(recomputed, block)


def test_control_numbers_stop_when_aligned_accuracy_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["aligned_acc"][0] = 0.96
    with pytest.raises(RuntimeError, match="aligned acc"):
        assert_mix15_b2_control_numbers(recomputed, block)


def test_control_numbers_stop_when_rescore_sf_differs():
    recomputed, block = _matching_control()
    block["SF"]["unseen"] = [0.5, 0.4, 0.6]
    with pytest.raises(RuntimeError):
        assert_mix15_b2_control_numbers(recomputed, block)


def test_control_numbers_stop_when_flip_count_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["n_flip"] = 16
    with pytest.raises(RuntimeError, match="counts"):
        assert_mix15_b2_control_numbers(recomputed, block)


def test_stated_sanity_matches_the_mix15_rescore_record():
    assert MIX15_B2_SANITY["seen"]["aligned_acc"] == 0.965
    assert MIX15_B2_SANITY["unseen"]["aligned_acc"] == 0.85
    assert MIX15_B2_SANITY["seen"]["reversed_SF"] == (
        0.07772020725388601,
        0.041666666666666664,
        0.11641084521795177,
    )
    assert MIX15_B2_SANITY["unseen"]["reversed_SF"] == (
        0.2235294117647059,
        0.16666666666666666,
        0.28651685393258425,
    )
    assert MIX15_B2_SANITY["seen"]["n_aligned_correct"] == 193
    assert MIX15_B2_SANITY["seen"]["n_flip"] == 15
    assert MIX15_B2_SANITY["unseen"]["n_aligned_correct"] == 170
    assert MIX15_B2_SANITY["unseen"]["n_flip"] == 38


def test_guard_allows_only_mix15_b2_outputs():
    assert "torch" not in sys.modules
    guard_mix15_b2_output(GEN_PATH)
    guard_mix15_b2_output(METRICS_PATH)
    guard_mix15_b2_output(LOG_PATH)
    guard_mix15_b2_output(METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".tmp"))
    banned = [
        PROJECT / "build" / "e1" / "synonym" / "b0" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b1" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b2" / "rows.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b2" / "run.json",
        PROJECT / "experiments" / "e1" / "synonym" / "metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "b1_metrics.json",
        PROJECT / "build" / "e1" / "pilot" / "b2" / "logic.jsonl",
        PROJECT / "build" / "e1" / "pilot_mix15" / "b2" / "logic.jsonl",
        PROJECT / "experiments" / "e1" / "pilot" / "b2" / "metrics.json",
        PROJECT / "experiments" / "e1" / "pilot_mix15" / "b2" / "metrics.json",
        PROJECT / "experiments" / "e1" / "rescore" / "metrics.json",
        PROJECT / "build" / "logs" / "e1_synonym.log",
        PROJECT / "build" / "logs" / "e1_synonym_b1.log",
    ]
    for path in banned:
        with pytest.raises(SystemExit):
            guard_mix15_b2_output(path)
