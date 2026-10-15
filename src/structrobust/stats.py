"""Paired bootstrap over latent tasks."""

from __future__ import annotations

import numpy as np


def mean_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def group_mean_diff_ci(changed, unchanged, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """CI of mean(changed) - mean(unchanged).

    Each group is resampled on its own. The unit is one latent task.
    The changed group is drawn first. Returns None if either group is empty.
    The point is the difference of the two sample means, not the bootstrap mean.
    """
    changed_v = np.asarray(changed, dtype=float)
    unchanged_v = np.asarray(unchanged, dtype=float)
    if changed_v.size == 0 or unchanged_v.size == 0:
        return None
    point = float(changed_v.mean() - unchanged_v.mean())
    rng = np.random.default_rng(seed)
    idx_changed = rng.integers(0, changed_v.size, size=(n_boot, changed_v.size))
    idx_unchanged = rng.integers(0, unchanged_v.size, size=(n_boot, unchanged_v.size))
    boots = changed_v[idx_changed].mean(axis=1) - unchanged_v[idx_unchanged].mean(axis=1)
    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1 - alpha / 2))
    return point, low, high


def paired_diff_ci(a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """CI of mean(a - b), resampling paired units (latent tasks)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return mean_ci(a - b, n_boot=n_boot, seed=seed, alpha=alpha)


def conditional_flip_ci(canon, shifted, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """SF = P(shifted wrong | canonical right), bootstrapping latent tasks."""
    c = np.asarray(canon, dtype=float)
    s = np.asarray(shifted, dtype=float)
    if c.sum() == 0:
        return None, None, None
    point = float(((c == 1) & (s == 0)).sum() / (c == 1).sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.size, size=(n_boot, c.size))
    cc, ss = c[idx], s[idx]
    num = ((cc == 1) & (ss == 0)).sum(axis=1)
    den = np.maximum((cc == 1).sum(axis=1), 1)
    boots = num / den
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def multi_seed_diff_ci(per_seed_a: list, per_seed_b: list, n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05):
    """Hierarchical bootstrap: resample seeds, then latent tasks within seed.

    per_seed_a[k] and per_seed_b[k] are paired arrays (same latent tasks) for seed k.
    """
    rng = np.random.default_rng(seed)
    diffs = [np.asarray(a, float) - np.asarray(b, float) for a, b in zip(per_seed_a, per_seed_b)]
    point = float(np.mean([d.mean() for d in diffs]))
    boots = np.empty(n_boot)
    k = len(diffs)
    for t in range(n_boot):
        chosen = rng.integers(0, k, size=k)
        vals = []
        for c in chosen:
            d = diffs[c]
            vals.append(d[rng.integers(0, d.size, size=d.size)].mean())
        boots[t] = np.mean(vals)
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
