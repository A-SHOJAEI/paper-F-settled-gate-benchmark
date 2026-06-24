"""
Runtime LTL monitor for the AMPLE-GNC shield subsystem.

Samples telemetry every 100 ms and predicts constraint violations within a
configurable horizon (default 1 s).  The monitor instantiates all nine
invariants (I1--I9) and aggregates their check / predict results into a
single :class:`ShieldDecision` that downstream consumers (safe-hold
sequencer, commander, scheduler) can act on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from shield.monitors.invariants import (
    ALL_INVARIANT_CLASSES,
    InvariantBase,
    InvariantThresholds,
)


# ---------------------------------------------------------------------------
# Shield decision data structures
# ---------------------------------------------------------------------------
class Severity(Enum):
    """Severity of a shield violation or warning."""

    NOMINAL = auto()
    WARNING = auto()       # predicted violation within horizon
    VIOLATION = auto()     # invariant already breached
    CRITICAL = auto()      # multiple invariants breached simultaneously


@dataclass(frozen=True)
class InvariantResult:
    """Result of evaluating a single invariant."""

    invariant_id: str
    description: str
    satisfied: bool
    time_to_violation: Optional[float]  # seconds, None if safe


@dataclass(frozen=True)
class ShieldDecision:
    """Aggregate output from one monitor evaluation cycle."""

    timestamp_s: float
    severity: Severity
    results: tuple[InvariantResult, ...]
    violated_ids: tuple[str, ...]
    warning_ids: tuple[str, ...]

    @property
    def is_safe(self) -> bool:
        return self.severity == Severity.NOMINAL

    @property
    def min_time_to_violation(self) -> Optional[float]:
        """Shortest predicted time-to-violation across all invariants."""
        ttv_values = [
            r.time_to_violation for r in self.results if r.time_to_violation is not None
        ]
        return min(ttv_values) if ttv_values else None


# ---------------------------------------------------------------------------
# LTL Monitor
# ---------------------------------------------------------------------------
class LTLMonitor:
    """Runtime monitor that evaluates all safety invariants.

    Parameters
    ----------
    thresholds
        Mission-configurable limits for each invariant.
    horizon_s
        Look-ahead window for violation prediction (default 1.0 s).
    sample_interval_s
        Nominal sampling period (default 0.1 s = 100 ms).
    """

    SAMPLE_INTERVAL_S: float = 0.100  # 100 ms

    def __init__(
        self,
        thresholds: Optional[InvariantThresholds] = None,
        horizon_s: float = 1.0,
        sample_interval_s: float = 0.100,
    ) -> None:
        self.thresholds = thresholds or InvariantThresholds()
        self.horizon_s = horizon_s
        self.sample_interval_s = sample_interval_s

        # Instantiate every invariant
        self._invariants: List[InvariantBase] = [
            cls(self.thresholds) for cls in ALL_INVARIANT_CLASSES
        ]

        # Bookkeeping
        self._last_sample_time: float = 0.0
        self._evaluation_count: int = 0
        self._history: List[ShieldDecision] = []
        self._max_history: int = 1000  # ring-buffer depth

    # -- public API ---------------------------------------------------------

    @property
    def invariants(self) -> List[InvariantBase]:
        return list(self._invariants)

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    def evaluate(self, state: Dict[str, Any]) -> ShieldDecision:
        """Run all invariants against *state* and return a :class:`ShieldDecision`.

        This is the main entry point called every sample interval.
        """
        now = time.monotonic()
        results: List[InvariantResult] = []
        violated: List[str] = []
        warned: List[str] = []

        for inv in self._invariants:
            satisfied = inv.check(state)
            ttv = inv.predict_violation(state, horizon_s=self.horizon_s)

            results.append(
                InvariantResult(
                    invariant_id=inv.id,
                    description=inv.description,
                    satisfied=satisfied,
                    time_to_violation=ttv,
                )
            )

            if not satisfied:
                violated.append(inv.id)
            elif ttv is not None:
                warned.append(inv.id)

        severity = self._classify_severity(violated, warned)

        decision = ShieldDecision(
            timestamp_s=now,
            severity=severity,
            results=tuple(results),
            violated_ids=tuple(violated),
            warning_ids=tuple(warned),
        )

        self._record(decision, now)
        return decision

    def should_sample(self) -> bool:
        """Return True if enough time has elapsed since the last sample."""
        return (time.monotonic() - self._last_sample_time) >= self.sample_interval_s

    def get_history(self, n: int = 10) -> List[ShieldDecision]:
        """Return the last *n* decisions (most recent last)."""
        return self._history[-n:]

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _classify_severity(
        violated: List[str], warned: List[str]
    ) -> Severity:
        if len(violated) >= 2:
            return Severity.CRITICAL
        if violated:
            return Severity.VIOLATION
        if warned:
            return Severity.WARNING
        return Severity.NOMINAL

    def _record(self, decision: ShieldDecision, now: float) -> None:
        self._last_sample_time = now
        self._evaluation_count += 1
        self._history.append(decision)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_monitor(
    thresholds: Optional[InvariantThresholds] = None,
    horizon_s: float = 1.0,
) -> LTLMonitor:
    """Create an :class:`LTLMonitor` with optional custom thresholds."""
    return LTLMonitor(thresholds=thresholds, horizon_s=horizon_s)
