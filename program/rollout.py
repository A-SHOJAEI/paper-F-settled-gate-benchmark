"""Basilisk rollout adapter: ``(Fault, policy, seed) -> per-step pointing trace``.

This is the WS0->WS1 bridge. WS0's honest-evaluation spine
(``program.eval_harness`` / ``program.metrics``) is deliberately Basilisk-free: it
consumes a per-step pointing-error trajectory and scores the *settled* gate (held
within threshold over a dwell window) rather than the legacy transient
"touch-once" minimum that the audit flagged as overstating capability. This module
produces those trajectories from the **real** Basilisk 6-DOF attitude dynamics.

Given a structurally-held-out actuator ``Fault`` (``program.fault_taxonomy``) and a
controller policy, ``rollout`` runs one episode in ``BasiliskAttitudeEnv`` with the
hub inertia scaled by ``f`` and the actuator fault applied in the loop
(``applied command = command * g + b``), and returns the per-step pointing error in
degrees for ``metrics.summarize`` / ``eval_harness.evaluate``.

Design notes
------------
* The policies are the **audited** control laws, reused verbatim from
  ``controller.rma.rma_attitude`` (fault-unaware PD, the privileged analytic
  teacher, and the deployed GRU student), so the eval scores the very artifacts the
  audit examined — no re-implementation drift.
* The fault is applied to the *normalised* command before the env scales by the
  actuator authority, matching ``rma_attitude.gate_eval`` (``env.step(a * g)``); the
  additive bias ``b`` generalises it to the GAIN_BIAS / total-loss classes
  (``b = 0`` for SIGN/GAIN, so the GAIN eval reduces exactly to the audited loop).
* Basilisk is imported **lazily** (only when an episode actually runs), so this
  module and its pure trajectory logic import and unit-test without Basilisk; the
  real-dynamics path is covered by a Basilisk-gated smoke test. A custom
  ``env_factory`` can be injected for tests.

Run (requires Basilisk on PATH — see HANDOFF.md §6):
    python -m program.rollout --faults 50 --seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from program.fault_taxonomy import Fault, FaultClass, heldout_test_faults

logger = logging.getLogger(__name__)

# A controller policy maps an observation [sigma(3), omega(3)] to a normalised
# action in [-1, 1]^3 (the actuator authority is applied by the env). This is the
# exact signature produced by controller.rma.rma_attitude.{pd,teacher,rma}_policy.
PolicyFn = Callable[[np.ndarray], np.ndarray]
# make_policy(fault) -> a FRESH policy for one (fault, seed) trial. Fresh so any
# recurrent state (the GRU student's hidden state) resets per episode, matching
# rma_attitude.gate_eval's fresh-per-task contract.
MakePolicy = Callable[[Fault], PolicyFn]
# env_factory(fault, seed, cfg) -> a gym-style env exposing reset()/step()/close().
# Typed Any because the real env wraps Basilisk SWIG objects (codebase convention).
EnvFactory = Callable[[Fault, int, Any], Any]
# obs_corruptor(obs, step) -> the observation the POLICY sees (sensor fault/noise
# model); the scored trace stays the TRUE simulator state. Built fresh per
# (fault, seed) by a MakeObsCorruptor so stochastic corruptors are seedable.
ObsCorruptor = Callable[[np.ndarray, int], np.ndarray]
MakeObsCorruptor = Callable[[Fault, int], ObsCorruptor]

HELDOUT_GAIN_OUT = Path("evidence/program/rma_heldout_gain.json")


def pointing_deg(sigma: np.ndarray) -> float:
    """Pointing error (deg) of an MRP attitude set from identity: phi = 4*atan(|s|).

    phi is mathematically bounded by 360 deg (atan -> pi/2). A controller that drives
    the body to tumble (the fault-unaware PD on a reversed-sign fault) can overflow the
    float32 MRP to +/-inf/nan upstream; we treat any non-finite attitude as the maximum
    180 deg pointing error so the trace stays finite. This only ever raises a value on
    an already-diverged trajectory (it cannot lower a settled one), so it cannot inflate
    any settled-gate result -- it just keeps the metrics (and the evidence JSON) finite
    instead of producing NaN medians for the diverged baseline.
    """
    n = float(np.linalg.norm(sigma))
    if not math.isfinite(n):
        return 180.0
    return float(math.degrees(4.0 * math.atan(n)))


def default_cfg() -> Any:
    """The WP0-testbed-matched controller config (ep_len=400 -> 200 s, 0.2 N·m).

    Matches ``controller.rma.rma_attitude.env_cfg`` so the deployed RMA policy plugs
    in unchanged.
    """
    from controller.maml.meta_imitation_basilisk import MetaConfig

    return MetaConfig(ep_len=400, max_torque_nm=0.2)


def default_env_factory(fault: Fault, seed: int, cfg: Any) -> Any:
    """Build a real ``BasiliskAttitudeEnv`` with hub inertia scaled by ``fault.f``.

    Mirrors ``controller.maml.meta_imitation_basilisk._task_env`` (same base inertia
    and scaling) so the dynamics match the audited training/eval testbed exactly.
    """
    from controller.maml.meta_imitation_basilisk import BASE_INERTIA
    from simulation.basilisk.attitude_env import AttitudeEnvConfig, BasiliskAttitudeEnv

    inertia = tuple(base * fault.f for base in BASE_INERTIA)
    return BasiliskAttitudeEnv(
        AttitudeEnvConfig(
            episode_length=int(cfg.ep_len),
            max_torque_nm=float(cfg.max_torque_nm),
            inertia_diag=inertia,  # type: ignore[arg-type]
            seed=seed,
        )
    )


def rw_env_factory(fault: Fault, seed: int, cfg: Any) -> Any:
    """Build a REACTION-WHEEL-MODEL ``BasiliskAttitudeEnv`` (Paper C, C-4).

    Identical inertia scaling to ``default_env_factory`` (hub inertia scaled by
    ``fault.f``) but with ``use_reaction_wheels=True``: three body-axis wheels
    produce the body torque via real wheel dynamics (momentum, saturation), so the
    per-axis actuator fault (applied = action*g + b, in ``rollout``) now acts on the
    WHEEL torque command. Opt-in only; the default path is unchanged.
    """
    from controller.maml.meta_imitation_basilisk import BASE_INERTIA
    from simulation.basilisk.attitude_env import AttitudeEnvConfig, BasiliskAttitudeEnv

    inertia = tuple(base * fault.f for base in BASE_INERTIA)
    return BasiliskAttitudeEnv(
        AttitudeEnvConfig(
            episode_length=int(cfg.ep_len),
            max_torque_nm=float(cfg.max_torque_nm),
            inertia_diag=inertia,  # type: ignore[arg-type]
            seed=seed,
            use_reaction_wheels=True,
        )
    )


def rollout(
    policy: PolicyFn,
    fault: Fault,
    seed: int,
    *,
    cfg: Any | None = None,
    env_factory: EnvFactory = default_env_factory,
    obs_corruptor: ObsCorruptor | None = None,
) -> list[float]:
    """Run one episode; return the per-step pointing-error trace (degrees).

    The actuator fault is applied to the normalised command each step
    (``applied = action * g + b``); the env clips to its authority, so an
    over-effective or biased command saturates physically (matching the real
    actuator). The recorded value each step is the post-step pointing error, so the
    trace has up to ``cfg.ep_len`` samples (fewer if the episode truncates early).

    ``obs_corruptor(obs, step) -> obs'`` models a SENSOR fault/noise: it corrupts
    only what the POLICY sees. The recorded trace is always the TRUE simulator
    pointing error — scoring on corrupted measurements would let a controller
    "succeed" by being lied to. ``None`` (default) is the audited byte-identical
    path.
    """
    resolved_cfg = default_cfg() if cfg is None else cfg
    g = fault.g_arr()
    b = fault.b_arr()
    env = env_factory(fault, seed, resolved_cfg)
    trace: list[float] = []
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(int(resolved_cfg.ep_len)):
            seen = obs if obs_corruptor is None else obs_corruptor(obs, step)
            a = np.asarray(policy(seen), dtype=np.float32)
            applied = (a * g + b).astype(np.float32)
            obs, _, _, truncated, _ = env.step(applied)
            trace.append(pointing_deg(np.asarray(obs[:3])))
            if truncated:
                break
    finally:
        env.close()
    return trace


def make_rollout_fn(
    make_policy: MakePolicy,
    *,
    cfg: Any | None = None,
    env_factory: EnvFactory = default_env_factory,
    make_obs_corruptor: MakeObsCorruptor | None = None,
) -> Callable[[Fault, int], list[float]]:
    """Adapt a controller into an ``eval_harness.evaluate`` ``rollout_fn(fault, seed)``.

    A fresh policy (and, if given, a fresh seeded obs corruptor) is built per
    (fault, seed) so recurrent adaptation state and noise streams reset each episode.
    """
    resolved_cfg = default_cfg() if cfg is None else cfg

    def rollout_fn(fault: Fault, seed: int) -> list[float]:
        corruptor = None if make_obs_corruptor is None else make_obs_corruptor(fault, seed)
        return rollout(
            make_policy(fault),
            fault,
            seed,
            cfg=resolved_cfg,
            env_factory=env_factory,
            obs_corruptor=corruptor,
        )

    return rollout_fn


# --------------------------------------------------------------------------- #
# Controller makers (the audited control laws, reused verbatim)
# --------------------------------------------------------------------------- #
def pd_maker(cfg: Any) -> MakePolicy:
    """Fault-UNAWARE PD baseline — the honest lower bound."""
    from controller.rma.rma_attitude import pd_policy

    def make(_fault: Fault) -> PolicyFn:
        return pd_policy(cfg)

    return make


def teacher_maker(cfg: Any) -> MakePolicy:
    """Privileged analytic teacher fed the TRUE fault z — the capability upper bound."""
    from controller.rma.rma_attitude import teacher_true_policy

    def make(fault: Fault) -> PolicyFn:
        return teacher_true_policy(fault.f, fault.g_arr(), cfg)

    return make


def teacher_bias_maker(cfg: Any) -> MakePolicy:
    """Privileged teacher with the TRUE constant-bias feedforward (Paper C, C-1 oracle).

    The capability upper bound for GAIN_BIAS: cancels the additive actuator bias exactly.
    Used to validate the augmented law oracle-first before deploying an online bias
    observer (PLAN_C C-1: the oracle must settle GAIN_BIAS, else the law is wrong)."""
    from controller.rma.rma_attitude import teacher_bias_true_policy

    def make(fault: Fault) -> PolicyFn:
        return teacher_bias_true_policy(fault.f, fault.g_arr(), fault.b_arr(), cfg)

    return make


def rma_student_maker(
    student: Any,
    cfg: Any,
    *,
    kd_scale: float = 1.0,
    g_margin: float = 0.0,
    latch_below_deg: float = 0.0,
) -> MakePolicy:
    """The DEPLOYED fault-adaptive policy: GRU student infers z online (no privilege).

    Optional knobs (default no-op) robustify steady-state holding against the limit-cycle
    failure mode diagnosed on held-out faults: ``latch_below_deg`` freezes ẑ once pointing
    converges; ``kd_scale``/``g_margin`` adjust the control law (see rma_policy).
    """
    from controller.rma.rma_attitude import rma_policy

    def make(_fault: Fault) -> PolicyFn:
        return rma_policy(
            student, cfg, kd_scale=kd_scale, g_margin=g_margin, latch_below_deg=latch_below_deg
        )

    return make


def load_rma_student(path: str | Path = "checkpoints/rma_student.pt") -> Any:
    """Load the trained RMA adaptation module (lazy torch import)."""
    import torch

    from controller.rma.rma_attitude import RMAStudent

    student = RMAStudent()
    student.load_state_dict(torch.load(Path(path), map_location="cpu"))
    student.eval()
    return student


def basilisk_available() -> bool:
    """True iff the Basilisk package is importable."""
    import importlib.util

    return importlib.util.find_spec("Basilisk") is not None


# --------------------------------------------------------------------------- #
# First honest RMA eval: held-out GAIN faults, >=10 seeds, settled gate
# --------------------------------------------------------------------------- #
def run_heldout_gain_eval(
    n_faults: int = 50,
    n_seeds: int = 10,
    *,
    student_ckpt: str | Path = "checkpoints/rma_student.pt",
    latch_below_deg: float = 3.0,
    out_path: Path = HELDOUT_GAIN_OUT,
    env_factory: EnvFactory = default_env_factory,
) -> dict:
    """Score PD, the deployed RMA student (baseline and estimate-latched variants), and
    the privileged teacher on the structurally-held-out GAIN battery through
    ``eval_harness.evaluate`` (settled gate, Wilson 95% CIs), and write the evidence JSON.

    ``rma_student_latched`` freezes the online fault estimate inside the pointing basin
    (``latch_below_deg``), which removes the low-excitation estimate jitter that drives
    the steady-state limit cycle; the threshold is tuned on TRAIN faults only (no test
    leakage). Fails fast if Basilisk is absent rather than letting every rollout error
    out and silently reporting a misleading 0% (the harness counts an errored rollout as
    a failure) — see the Prime Directive in HANDOFF.md §1.
    """
    if env_factory is default_env_factory and not basilisk_available():
        raise ModuleNotFoundError(
            "Basilisk is required for the real rollout but is not installed. "
            "Build it (HANDOFF.md §6) and put it on PATH, then re-run.",
            name="Basilisk",
        )

    from program import determinism, eval_harness

    determinism.set_global_determinism(7)
    cfg = default_cfg()
    faults = heldout_test_faults(FaultClass.GAIN, n=n_faults, seed=7_000)
    seeds = list(range(n_seeds))

    student = load_rma_student(student_ckpt)
    controllers: dict[str, MakePolicy] = {
        "pd_fault_unaware": pd_maker(cfg),
        "rma_student_deployed": rma_student_maker(student, cfg),
        "rma_student_latched": rma_student_maker(student, cfg, latch_below_deg=latch_below_deg),
        "teacher_privileged": teacher_maker(cfg),
    }
    results = {
        name: eval_harness.evaluate(
            make_rollout_fn(mk, cfg=cfg, env_factory=env_factory),
            faults,
            seeds,
            label=f"heldout_gain::{name}",
        )
        for name, mk in controllers.items()
    }
    payload = {
        "experiment": "rma_heldout_gain",
        "fault_class": "GAIN",
        "split": "test (structurally held out by construction)",
        "n_faults": n_faults,
        "n_seeds": n_seeds,
        "primary_gate": "settled_science (0.2 deg held over final dwell window)",
        "latch_below_deg": latch_below_deg,
        "latch_provenance": "tuned on TRAIN GAIN faults (seed 1000); argmax over {0.5,1,2,3,5} deg",
        "controllers": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_path)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="First honest RMA eval (held-out GAIN, settled gate).")
    p.add_argument("--faults", type=int, default=50, help="held-out GAIN faults to draw")
    p.add_argument("--seeds", type=int, default=10, help="seeds per fault (>=10)")
    p.add_argument("--student-ckpt", default="checkpoints/rma_student.pt")
    p.add_argument("--latch", type=float, default=3.0, help="latch_below_deg (train-tuned)")
    p.add_argument("--out", default=str(HELDOUT_GAIN_OUT))
    args = p.parse_args()

    payload = run_heldout_gain_eval(
        n_faults=args.faults,
        n_seeds=args.seeds,
        student_ckpt=args.student_ckpt,
        latch_below_deg=args.latch,
        out_path=Path(args.out),
    )
    for name, r in payload["controllers"].items():
        s = r["settled_science"]
        print(
            f"{name:24s} settled-science {s['rate']}  "
            f"Wilson95 {s['wilson95']}  (n={r['n_trials']}, errors={r['errors']})"
        )


if __name__ == "__main__":
    main()
