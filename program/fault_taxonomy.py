"""Structurally held-out actuator-fault taxonomy.

The audit found the controller's "held-out continuous faults" were an i.i.d. resample
of the *training* distribution (same generator, same range) — in-distribution, not
held out. This module defines train vs test fault sets that are **disjoint by
construction**: the test set lives in regions of (inertia, gain magnitude, sign
pattern, bias) the controller never trains on, so a high test score is genuine
generalization.

A fault is ``applied torque = command * g + b`` on a body with inertia scaled by ``f``.

Splits (disjoint by construction):
  - inertia f:   train [0.8, 2.0]            test [0.7, 0.8) U (2.0, 2.3]   (extrapolation)
  - gain |g|:    train [0.5, 1.5]            test [0.3, 0.5) U (1.5, 2.0]   (severe/over)
  - sign pattern: train <=1 reversed axis    test >=2 reversed axes          (unseen combos)
  - bias b:      train [-0.2, 0.2]           test [-0.3,-0.2) U (0.2, 0.3]
  - total loss (one axis g=0): out-of-distribution for every class -> shield-only regime.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum

import numpy as np


class FaultClass(str, Enum):
    SIGN = "sign"  # g in {+-1}, b = 0
    GAIN = "gain"  # continuous effectiveness, b = 0
    GAIN_BIAS = "gain_bias"  # continuous gain + additive bias
    TOTAL_LOSS = "total_loss"  # one axis g = 0 (under-actuated)


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass(frozen=True)
class Fault:
    """Per-axis actuator fault: applied = command * g + b, inertia scaled by f."""

    f: float
    g: tuple[float, float, float]
    b: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def g_arr(self) -> np.ndarray:
        return np.asarray(self.g, dtype=np.float32)

    def b_arr(self) -> np.ndarray:
        return np.asarray(self.b, dtype=np.float32)


# --- range constants ---------------------------------------------------------
F_TRAIN = (0.8, 2.0)
F_TEST = ((0.7, 0.8), (2.0, 2.3))
GMAG_TRAIN = (0.5, 1.5)
GMAG_TEST = ((0.3, 0.5), (1.5, 2.0))
BIAS_TRAIN = (-0.2, 0.2)
BIAS_TEST = ((-0.3, -0.2), (0.2, 0.3))

# sign patterns split by number of reversed (-1) axes
_ALL_SIGNS = list(itertools.product((1, -1), repeat=3))
SIGN_TRAIN = tuple(s for s in _ALL_SIGNS if sum(1 for v in s if v < 0) <= 1)  # 4 combos
SIGN_TEST = tuple(s for s in _ALL_SIGNS if sum(1 for v in s if v < 0) >= 2)  # 4 combos


def _draw_inertia(rng: np.random.Generator, split: Split) -> float:
    if split is Split.TRAIN:
        return float(rng.uniform(*F_TRAIN))
    lo, hi = F_TEST[rng.integers(len(F_TEST))]
    return float(rng.uniform(lo, hi))


def _draw_gmag(rng: np.random.Generator, split: Split) -> float:
    if split is Split.TRAIN:
        return float(rng.uniform(*GMAG_TRAIN))
    lo, hi = GMAG_TEST[rng.integers(len(GMAG_TEST))]
    return float(rng.uniform(lo, hi))


def _draw_bias(rng: np.random.Generator, split: Split) -> float:
    if split is Split.TRAIN:
        return float(rng.uniform(*BIAS_TRAIN))
    lo, hi = BIAS_TEST[rng.integers(len(BIAS_TEST))]
    return float(rng.uniform(lo, hi))


def _draw_signs(rng: np.random.Generator, split: Split) -> tuple[int, int, int]:
    pool = SIGN_TRAIN if split is Split.TRAIN else SIGN_TEST
    return tuple(pool[rng.integers(len(pool))])  # type: ignore[return-value]


def sample_fault(rng: np.random.Generator, fclass: FaultClass, split: Split) -> Fault:
    """Draw one fault of the given class from the train or test region."""
    f = _draw_inertia(rng, split)
    signs = _draw_signs(rng, split)
    if fclass is FaultClass.SIGN:
        g = tuple(float(s) for s in signs)
        return Fault(f, g)  # type: ignore[arg-type]
    if fclass is FaultClass.GAIN:
        g = tuple(float(s) * _draw_gmag(rng, split) for s in signs)
        return Fault(f, g)  # type: ignore[arg-type]
    if fclass is FaultClass.GAIN_BIAS:
        g = tuple(float(s) * _draw_gmag(rng, split) for s in signs)
        b = tuple(_draw_bias(rng, split) for _ in range(3))
        return Fault(f, g, b)  # type: ignore[arg-type]
    if fclass is FaultClass.TOTAL_LOSS:
        g_list = [float(s) * _draw_gmag(rng, split) for s in signs]
        g_list[int(rng.integers(3))] = 0.0  # one axis fully lost
        return Fault(f, tuple(g_list))  # type: ignore[arg-type]
    raise ValueError(f"unknown fault class {fclass}")


def heldout_test_faults(fclass: FaultClass, n: int = 200, seed: int = 7_000) -> list[Fault]:
    """Deterministic canonical held-out test battery for a class (test split only)."""
    rng = np.random.default_rng(seed)
    return [sample_fault(rng, fclass, Split.TEST) for _ in range(n)]


def train_faults(fclass: FaultClass, n: int = 200, seed: int = 1_000) -> list[Fault]:
    """Deterministic training-region fault draws (train split only)."""
    rng = np.random.default_rng(seed)
    return [sample_fault(rng, fclass, Split.TRAIN) for _ in range(n)]
