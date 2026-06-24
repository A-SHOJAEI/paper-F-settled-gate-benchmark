"""Seed-swept evaluation harness with honest uncertainty.

Every controller campaign routes through ``evaluate`` so results are comparable and
carry confidence intervals. The harness is decoupled from Basilisk: the caller
supplies ``rollout_fn(fault, seed) -> trace`` returning the per-step pointing-error
trajectory (degrees); the harness applies the settled/transient metrics
(``program.metrics``) over the full faults x seeds grid and aggregates rates with
Wilson intervals (``program.stats``).

This replaces the prior per-script eval that reported a single rate on a single
seed over as few as 4 fault patterns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from program import metrics
from program.stats import wilson

# rollout_fn(fault, seed) -> sequence of per-step pointing-error degrees
RolloutFn = Callable[[Any, int], Sequence[float]]


def _rate(k: int, n: int) -> dict:
    return {"rate": round(k / n, 4) if n else None, "k": k, "n": n, "wilson95": wilson(k, n)}


def _median_iqr(xs: list[float]) -> dict:
    if not xs:
        return {"median": None, "iqr": None, "n": 0}
    a = np.asarray(xs, dtype=float)
    q1, med, q3 = (float(np.percentile(a, p)) for p in (25, 50, 75))
    return {"median": round(med, 4), "iqr": [round(q1, 4), round(q3, 4)], "n": int(a.size)}


def evaluate(
    rollout_fn: RolloutFn,
    faults: Sequence[Any],
    seeds: Sequence[int],
    *,
    dt_s: float = 0.5,
    dwell: int = metrics.DEFAULT_DWELL_STEPS,
    science_deg: float = metrics.SCIENCE_DEG,
    op_deg: float = metrics.OPERATIONAL_DEG,
    label: str = "",
) -> dict:
    """Run ``rollout_fn`` over every (fault, seed) and aggregate honest metrics.

    Returns rates with Wilson 95% CIs for the settled-science (primary), settled-
    operational, and transient-min (legacy/secondary) gates, plus settle-time and
    pointing-error distributions. ``n_trials = len(faults) * len(seeds)``.
    """
    settled_sci = settled_op = transient = errors = 0
    settle_times: list[float] = []
    min_degs: list[float] = []
    final_degs: list[float] = []
    n = 0

    for fault in faults:
        for seed in seeds:
            n += 1
            try:
                trace = rollout_fn(fault, seed)
            except Exception:  # noqa: BLE001 - an errored rollout counts as a failure
                errors += 1
                continue
            s = metrics.summarize(
                trace, dt_s=dt_s, science_deg=science_deg, op_deg=op_deg, dwell=dwell
            )
            settled_sci += int(s["settled_science"])
            settled_op += int(s["settled_operational"])
            transient += int(s["transient_min_science"])
            min_degs.append(s["transient_min_deg"])
            final_degs.append(s["final_deg"])
            if s["settle_time_s"] is not None:
                settle_times.append(s["settle_time_s"])

    return {
        "label": label,
        "n_trials": n,
        "n_faults": len(faults),
        "n_seeds": len(seeds),
        "errors": errors,
        "settled_science": _rate(settled_sci, n),  # PRIMARY honest gate
        "settled_operational": _rate(settled_op, n),
        "transient_min_science": _rate(transient, n),  # legacy 'best' gate (secondary)
        "settle_time_s": _median_iqr(settle_times),
        "transient_min_deg": _median_iqr(min_degs),
        "final_deg": _median_iqr(final_degs),
        "config": {
            "dt_s": dt_s,
            "dwell_steps": dwell,
            "science_deg": science_deg,
            "op_deg": op_deg,
        },
    }
