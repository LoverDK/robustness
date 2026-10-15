"""Dependency-burden features of a presentation.

All features are computed from the positions of the relevant (core) rules in the
text, relative to the proof order of the latent task:

- back_reads: inversions between proof order and text order (pairs of proof
  steps i < j whose rules appear in the text with pos(i) > pos(j)).
- d_rel: mean absolute text distance, in rule sentences, between consecutive
  proof steps.
- f_active: peak number of relevant rules a single left-to-right reader must hold
  in memory because their premises are not yet established when they are read.
"""

from __future__ import annotations

from .logic import Latent


def burden(lat: Latent, core_positions: list[int]) -> dict[str, float]:
    m = len(core_positions)
    inv = sum(1 for i in range(m) for j in range(i + 1, m) if core_positions[i] > core_positions[j])
    if m > 1:
        d_rel = sum(abs(core_positions[i] - core_positions[i - 1]) for i in range(1, m)) / (m - 1)
    else:
        d_rel = 0.0
    known = set(lat.facts)
    pending: list[int] = []
    peak = 0
    text_order = sorted(range(m), key=lambda i: core_positions[i])
    for i in text_order:
        pending.append(i)
        progress = True
        while progress:
            progress = False
            for k in list(pending):
                s, t = lat.core_rules[k]
                if all(x in known for x in s):
                    known.add(t)
                    pending.remove(k)
                    progress = True
        peak = max(peak, len(pending))
    max_inv = m * (m - 1) / 2 if m > 1 else 1.0
    return {
        "back_reads": float(inv),
        "back_reads_norm": inv / max_inv,
        "d_rel": d_rel,
        "f_active": float(peak),
        "n_core": float(m),
    }


def burden_score(b: dict[str, float]) -> float:
    """Scalar burden used for mining: normalised inversions plus frontier share."""
    m = max(1.0, b["n_core"])
    return b["back_reads_norm"] + b["f_active"] / m + b["d_rel"] / (m + 1)
