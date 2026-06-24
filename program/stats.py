"""Canonical small-sample binomial statistics for the evaluation harness.

Distribution-free / exact interval helpers shared by every campaign so that
reported rates carry honest uncertainty. These match the inline copies in
``program/wp8_adversarial_eval.py`` (``_wilson``/``_cp_upper``); those call sites
should migrate to import from here to avoid divergence.
"""

from __future__ import annotations


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval (default 95%) for a binomial proportion k/n."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1.0 + z * z / n
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    center = (p + z * z / (2 * n)) / d
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial rate k/n.

    Used to bound a *failure* fraction (e.g. shield-caught / exceed-limit) from
    above: with k failures in n trials, the true rate is ``<= cp_upper`` at the
    given confidence. Exact (Beta-quantile); for k=0 uses the rule-of-three form.
    """
    if n == 0:
        return 1.0
    if k == 0:
        return float(round(1.0 - (1.0 - conf) ** (1.0 / n), 5))
    if k >= n:
        return 1.0
    from scipy.stats import beta  # type: ignore[import-untyped]

    return round(float(beta.ppf(conf, k + 1, n - k)), 5)  # type: ignore[no-any-return]


def cp_interval(k: int, n: int, conf: float = 0.95) -> list[float]:
    """Two-sided Clopper-Pearson interval for k/n at the given confidence."""
    if n == 0:
        return [0.0, 1.0]
    from scipy.stats import beta

    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 5), round(hi, 5)]
