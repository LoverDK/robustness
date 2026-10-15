"""CPU checks for the mix15 B3 synonym control. No model load."""

from __future__ import annotations

import sys

import pytest

from structrobust.e1_synonym_mix15_b3 import (
    GEN_PATH,
    LOG_PATH,
    METRICS_PATH,
    MIX15_B3_SANITY,
    PROJECT,
    RESCORE_KEY,
    assert_mix15_b3_control_numbers,
    guard_mix15_b3_output,
)


def test_import_does_not_load_torch():
    assert "torch" not in sys.modules


def _matching_control():
    block = {"quant": "nf4", "SF": {}, "acc": {}, "counts": {}}
    recomputed = {}
    for split, sanity in MIX15_B3_SANITY.items():
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


def test_control_numbers_accept_the_stated_mix15_b3_values():
    recomputed, block = _matching_control()
    assert_mix15_b3_control_numbers(recomputed, block)


def test_control_numbers_stop_when_aligned_accuracy_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["aligned_acc"][0] = 0.97
    with pytest.raises(RuntimeError, match="aligned acc"):
        assert_mix15_b3_control_numbers(recomputed, block)


def test_control_numbers_stop_when_rescore_sf_differs():
    recomputed, block = _matching_control()
    block["SF"]["unseen"] = [0.5, 0.4, 0.6]
    with pytest.raises(RuntimeError):
        assert_mix15_b3_control_numbers(recomputed, block)


def test_control_numbers_stop_when_flip_count_differs():
    recomputed, block = _matching_control()
    recomputed["seen"]["n_flip"] = 25
    with pytest.raises(RuntimeError, match="counts"):
        assert_mix15_b3_control_numbers(recomputed, block)


def test_stated_sanity_matches_the_mix15_b3_record():
    assert RESCORE_KEY == ("rounds", "pilot_mix15", "conditions", "b3")
    assert MIX15_B3_SANITY["seen"]["aligned_acc"] == 0.975
    assert MIX15_B3_SANITY["unseen"]["aligned_acc"] == 0.79
    assert MIX15_B3_SANITY["seen"]["reversed_SF"] == (
        0.12307692307692308,
        0.08121827411167512,
        0.1717171717171717,
    )
    assert MIX15_B3_SANITY["unseen"]["reversed_SF"] == (
        0.25316455696202533,
        0.1875,
        0.3231707317073171,
    )
    assert MIX15_B3_SANITY["seen"]["n_aligned_correct"] == 195
    assert MIX15_B3_SANITY["seen"]["n_flip"] == 24
    assert MIX15_B3_SANITY["unseen"]["n_aligned_correct"] == 158
    assert MIX15_B3_SANITY["unseen"]["n_flip"] == 40


def test_guard_allows_only_mix15_b3_outputs():
    assert "torch" not in sys.modules
    guard_mix15_b3_output(GEN_PATH)
    guard_mix15_b3_output(METRICS_PATH)
    guard_mix15_b3_output(LOG_PATH)
    guard_mix15_b3_output(METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".tmp"))
    banned = [
        PROJECT / "build" / "e1" / "synonym" / "b0" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b1" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "b3" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b1" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b2" / "generations.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b3" / "rows.jsonl",
        PROJECT / "build" / "e1" / "synonym" / "mix15_b3" / "run.json",
        PROJECT / "experiments" / "e1" / "synonym" / "metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "b1_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "b3_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "mix15_b1_metrics.json",
        PROJECT / "experiments" / "e1" / "synonym" / "mix15_b2_metrics.json",
        PROJECT / "build" / "e1" / "pilot" / "b3" / "logic.jsonl",
        PROJECT / "build" / "e1" / "pilot_mix15" / "b3" / "logic.jsonl",
        PROJECT / "experiments" / "e1" / "rescore" / "metrics.json",
        PROJECT / "build" / "logs" / "e1_synonym.log",
        PROJECT / "build" / "logs" / "e1_synonym_b3.log",
        PROJECT / "build" / "logs" / "e1_synonym_mix15_b1.log",
        PROJECT / "build" / "logs" / "e1_synonym_mix15_b2.log",
    ]
    for path in banned:
        with pytest.raises(SystemExit):
            guard_mix15_b3_output(path)
