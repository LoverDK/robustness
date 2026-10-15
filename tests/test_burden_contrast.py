"""Two-group mean difference. No saved generations are read."""

from __future__ import annotations

import numpy as np

from structrobust.stats import group_mean_diff_ci


def test_bootstrap_diff_point_equals_difference_of_means():
    changed = [1.0, 2.0, 9.0]
    unchanged = [1.0, 3.0]
    result = group_mean_diff_ci(changed, unchanged, n_boot=2000, seed=0)
    assert result is not None
    point = result[0]
    expected = float(np.mean(changed) - np.mean(unchanged))
    assert point == expected
