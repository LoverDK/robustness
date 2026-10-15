"""Counting formulas for SF and delta_T. No saved generations are read."""

from __future__ import annotations

import numpy as np
import pytest

from structrobust.rescore import gap_and_interaction, sf_count
from structrobust.stats import conditional_flip_ci


def _count_sf(aligned, reversed_):
    aligned = list(aligned)
    reversed_ = list(reversed_)
    assert len(aligned) == len(reversed_)
    num = 0
    den = 0
    for canon, shifted in zip(aligned, reversed_):
        if canon == 1:
            den += 1
            if shifted == 0:
                num += 1
    if den == 0:
        return None
    return num / den


def test_sf_count_is_flips_over_aligned_correct():
    aligned = [1, 1, 0, 1, 0, 1]
    reversed_ = [0, 1, 1, 0, 0, 1]
    assert _count_sf(aligned, reversed_) == 2 / 4
    assert sf_count(aligned, reversed_) == 2 / 4
    point, low, high = conditional_flip_ci(aligned, reversed_, n_boot=2000, seed=0)
    assert point == 2 / 4
    assert low <= point <= high


def test_sf_ignores_aligned_wrong_and_is_empty_without_aligned_correct():
    assert _count_sf([1, 0, 0, 0], [0, 1, 1, 1]) == 1 / 1
    assert sf_count([1, 0, 0, 0], [0, 1, 1, 1]) == 1.0
    point, _low, _high = conditional_flip_ci([1, 0, 0, 0], [0, 1, 1, 1], n_boot=2000, seed=0)
    assert point == 1.0
    assert _count_sf([0, 0], [1, 0]) is None
    assert sf_count([0, 0], [1, 0]) is None
    assert conditional_flip_ci([0, 0], [1, 0], n_boot=20, seed=0)[0] is None


def test_delta_t_is_unpooled_mean_difference_and_interaction_subtracts_aligned():
    seen = {
        "aligned": np.array([1, 1, 0, 0], dtype=float),
        "reversed": np.array([1, 0, 0, 1], dtype=float),
    }
    unseen = {
        "aligned": np.array([1, 0, 0], dtype=float),
        "reversed": np.array([0, 0, 1], dtype=float),
    }
    delta, interaction = gap_and_interaction(seen, unseen, n_boot=2000, seed=0)
    for presentation in ("aligned", "reversed"):
        expected = float(seen[presentation].mean() - unseen[presentation].mean())
        assert delta[presentation][0] == expected
        assert delta[presentation][1] <= delta[presentation][0] <= delta[presentation][2]
    assert interaction["reversed"][0] == delta["reversed"][0] - delta["aligned"][0]
    assert seen["aligned"].mean() - unseen["aligned"].mean() != (
        float(np.sum(seen["aligned"]) + np.sum(unseen["aligned"])) / (len(seen["aligned"]) + len(unseen["aligned"]))
    )


def test_delta_t_subtracts_unseen_from_seen():
    seen = {"aligned": [0, 0], "reversed": [1, 1]}
    unseen = {"aligned": [1, 1, 1, 1], "reversed": [0, 0, 0, 0]}
    delta, interaction = gap_and_interaction(seen, unseen, n_boot=2000, seed=0)
    assert delta["aligned"][0] == -1.0
    assert delta["reversed"][0] == 1.0
    assert interaction["reversed"][0] == 2.0


def test_presentations_must_share_one_latent_order():
    seen = {"aligned": [1, 0], "reversed": [1]}
    unseen = {"aligned": [1, 0], "reversed": [1, 0]}
    with pytest.raises(ValueError):
        gap_and_interaction(seen, unseen, n_boot=10, seed=0)
