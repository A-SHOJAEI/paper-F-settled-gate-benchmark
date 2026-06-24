"""Pointing-performance metrics over a per-step pointing-error trajectory.

The legacy "gate" in ``simulation/basilisk/attitude_env.py`` is the **transient
minimum** (``best_pointing_deg``): the episode "succeeds" if the pointing error
*touches* the threshold once, even mid-swing. That overstates capability — a real
ADCS science gate requires the attitude to *settle and hold*. This module provides
the honest **settled** criterion (held within threshold for the final dwell window)
plus the transient-min as a reported secondary, so every controller is scored on
both and the difference is explicit.

All functions are pure (operate on a sequence of per-step pointing-error degrees),
so they need no Basilisk and are unit-testable directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# dt = 0.5 s/step; 20 steps = 10 s held within spec — the default settling window.
DEFAULT_DWELL_STEPS = 20
SCIENCE_DEG = 0.2
OPERATIONAL_DEG = 5.0


def _arr(trace: Sequence[float]) -> np.ndarray:
    return np.asarray(trace, dtype=float)


def transient_min_deg(trace: Sequence[float]) -> float:
    """Minimum pointing error over the trajectory (the legacy 'best')."""
    a = _arr(trace)
    return float(a.min()) if a.size else float("inf")


def transient_min_success(trace: Sequence[float], thresh: float = SCIENCE_DEG) -> bool:
    """Legacy gate: pointing error touches ``thresh`` at least once (secondary metric)."""
    return transient_min_deg(trace) <= thresh


def settled_success(
    trace: Sequence[float], thresh: float = SCIENCE_DEG, dwell: int = DEFAULT_DWELL_STEPS
) -> bool:
    """Honest gate: the final ``dwell`` steps are ALL within ``thresh`` (settled & held)."""
    a = _arr(trace)
    if a.size < dwell:
        return False
    return bool(np.all(a[-dwell:] <= thresh))


def settle_step(
    trace: Sequence[float], thresh: float = SCIENCE_DEG, dwell: int = DEFAULT_DWELL_STEPS
) -> int | None:
    """First step index from which the trajectory stays <= ``thresh`` through the end.

    This is the permanent settle point (and the recovery time, when the trajectory
    starts faulted). Returns None if it never settles for the final ``dwell`` window.
    """
    a = _arr(trace)
    if not settled_success(a, thresh, dwell):
        return None
    # walk backward to the earliest index that stays within threshold to the end
    s = int(a.size)
    while s > 0 and a[s - 1] <= thresh:
        s -= 1
    return s


def final_deg(trace: Sequence[float], dwell: int = DEFAULT_DWELL_STEPS) -> float:
    """Mean pointing error over the final dwell window (the held value)."""
    a = _arr(trace)
    if a.size == 0:
        return float("inf")
    w = a[-min(dwell, a.size) :]
    return float(w.mean())


def summarize(
    trace: Sequence[float],
    dt_s: float = 0.5,
    science_deg: float = SCIENCE_DEG,
    op_deg: float = OPERATIONAL_DEG,
    dwell: int = DEFAULT_DWELL_STEPS,
) -> dict:
    """All metrics for one trajectory: settled (science + operational), transient-min,
    settle/recovery time, and the held final value."""
    st = settle_step(trace, science_deg, dwell)
    return {
        "settled_science": settled_success(trace, science_deg, dwell),
        "settled_operational": settled_success(trace, op_deg, dwell),
        "transient_min_science": transient_min_success(trace, science_deg),
        "transient_min_deg": round(transient_min_deg(trace), 4),
        "final_deg": round(final_deg(trace, dwell), 4),
        "settle_step": st,
        "settle_time_s": None if st is None else round(st * dt_s, 3),
        "n_steps": int(_arr(trace).size),
    }
