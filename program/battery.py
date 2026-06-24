"""Common stress battery — the single source of truth for the whole program.

Every work package draws its perturbations from here so results compose into one
attribution story (RESEARCH_PROGRAM.md §1). Three perturbation families:

  - dynamics tasks: inertia scaling × unobserved actuator-sign fault (the G2
    family) — the controller-reliability stressor.
  - subsystem upsets: scenarios that drive a specific LTL invariant toward
    violation (battery drain, wheel saturation, thermal runaway, pointing loss)
    — the shield stressor.
  - SEU/TID: RP4-calibrated single-event upsets (NOMINAL / SPE_STRESS).

Seeds are explicit and versioned so the battery is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from controller.maml.meta_imitation_basilisk import TaskSpec

# --- dynamics task family (controller reliability stressor) -----------------
# Nominal + single- and two-axis actuator-sign faults across an inertia sweep.
DYNAMICS_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(1.0, (1, 1, 1)),  # nominal
    TaskSpec(1.3, (-1, 1, 1)),  # 1-axis flip
    TaskSpec(1.6, (1, 1, -1)),  # 1-axis flip
    TaskSpec(0.8, (1, -1, 1)),  # 1-axis flip
    TaskSpec(2.2, (-1, -1, 1)),  # 2-axis flip + extrapolated inertia (hardest)
)

# Tasks WITH an unobserved fault (the regime where the controller is unreliable).
FAULTED_TASKS: tuple[TaskSpec, ...] = tuple(t for t in DYNAMICS_TASKS if t.fault != (1, 1, 1))


@dataclass(frozen=True)
class SubsystemUpset:
    """A scenario that drives one invariant toward violation."""

    name: str
    invariant: str  # the LTL invariant it stresses (I1..I9)
    channel: str  # the shield-state channel it corrupts
    magnitude: float  # how far past the threshold it pushes


SUBSYSTEM_UPSETS: tuple[SubsystemUpset, ...] = (
    SubsystemUpset("battery_drain", "I1", "battery_soc", -0.45),
    SubsystemUpset("wheel_saturation", "I2", "wheel_momentum_frac", +0.55),
    SubsystemUpset("pointing_loss", "I4", "pointing_error_deg", +8.0),
    SubsystemUpset("thermal_runaway", "I8", "avionics_temp_c", +30.0),
)


@dataclass(frozen=True)
class Battery:
    """The versioned common stress battery."""

    version: str = "1.0"
    inertia_range: tuple[float, float] = (0.7, 2.3)
    sensor_noise_std: float = 0.01
    seu_modes: tuple[str, ...] = ("NOMINAL", "SPE_STRESS")
    base_seed: int = 1234
    dynamics_tasks: tuple[TaskSpec, ...] = field(default_factory=lambda: DYNAMICS_TASKS)
    faulted_tasks: tuple[TaskSpec, ...] = field(default_factory=lambda: FAULTED_TASKS)
    subsystem_upsets: tuple[SubsystemUpset, ...] = field(default_factory=lambda: SUBSYSTEM_UPSETS)


BATTERY = Battery()
