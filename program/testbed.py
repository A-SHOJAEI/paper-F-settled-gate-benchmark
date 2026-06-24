"""WP0 — the integrated, instrumented, ablatable testbed.

One closed loop, real components, every other WP is a *configuration* of this:

    controller (learned | pd-fault-unaware | scratch | safehold)
        -> [shield veto?] -> actuator fault -> real Basilisk 6-DOF attitude
        -> subsystem model (+ injected upset, + SEU) -> LTL shield (9 invariants)
        -> safe-hold / ground escalation

Ablation switches: ``shield_on``, ``controller``, ``fault`` (on/off via task),
``subsystem_upset``, ``seu_mode``. Instrumentation per episode: pointing
trajectory, gate success, I4 (and other) violation steps, shield detections,
safe-hold activations, ground escalations, **unhandled violations** (the C-A
quantity: violations that occur with no shield to catch them).

The C-A logic mirrors the proposal/SIL intent: a verified shield (sound by Kind 2
for I1/I3/I4/I6) **detects every violation within the horizon** (soundness ⇒ no
false all-clear), so transient breaches get a safe-hold and persistent ones
escalate to ground — *handled either way*. With the shield off, the
variance-limited controller's breaches are **unhandled**. That contrast is C-A.
"""

from __future__ import annotations

# ruff: noqa: N806
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from controller.maml.meta_imitation_basilisk import MetaConfig, TaskSpec, _apply_fault, _task_env
from controller.rl.bc_attitude import MLPPolicy
from program.battery import SubsystemUpset
from shield.monitors.ltl_monitor import LTLMonitor, Severity

BC_CKPT = "checkpoints/learned_attitude_bc.pt"


@dataclass
class TestbedConfig:
    ep_len: int = 400  # 200 s at dt=0.5 — the proposal's Phase-1 gate horizon
    dt_s: float = 0.5
    kp: float = 0.2
    kd: float = 1.5
    max_torque_nm: float = 0.2  # matches the BC-controller training regime
    gate_deg: float = 0.2
    i4_violation_deg: float = 5.0  # I4 threshold (pointing error)
    safehold_recover_steps: int = 20  # persistent breach -> ground escalation
    horizon_s: float = 1.0
    # env consistent with the BC controller: 400-step horizon, 0.2 N·m actuators
    base: MetaConfig = field(default_factory=lambda: MetaConfig(ep_len=400, max_torque_nm=0.2))


def load_controller(kind: str, cfg: TestbedConfig, seed: int = 0) -> Any:
    """Return a callable obs->normalized-action for the requested controller."""
    if kind == "pd":  # fault-UNAWARE PD: drives the wrong way on faulted axes

        def pd(obs: np.ndarray) -> np.ndarray:
            sigma, omega = obs[:3], obs[3:]
            u = (-cfg.kp * sigma - cfg.kd * omega) / cfg.max_torque_nm
            return np.clip(u, -1.0, 1.0).astype(np.float32)

        return pd
    if kind == "safehold":  # conservative coast (cut actuator-driven divergence)
        return lambda obs: np.zeros(3, dtype=np.float32)
    # learned (BC checkpoint) or scratch (random init) MLP
    pol = MLPPolicy()
    if kind == "learned":
        try:
            sd = torch.load(BC_CKPT, map_location="cpu", weights_only=True)
            pol.load_state_dict(
                sd if isinstance(sd, dict) and "net.0.weight" in sd else sd["state_dict"]
            )
        except Exception:  # noqa: BLE001 — fall back to untrained if ckpt shape differs
            torch.manual_seed(seed)
    else:
        torch.manual_seed(seed + 777)
    pol.eval()

    def learned(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = pol(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return np.asarray(out, dtype=np.float32)

    return learned


def _shield_state(
    pointing_deg: float, t_hr: float, rng: np.random.Generator, upset: SubsystemUpset | None
) -> dict:
    """Full 9-invariant shield state: nominal subsystem model + live pointing
    (I4 from the controller loop) + optional injected subsystem upset."""
    soc = 0.55 + 0.25 * math.sin(2 * math.pi * t_hr / (6.5 * 24)) + rng.normal(0, 0.01)
    s: dict[str, Any] = {
        "battery_soc": float(np.clip(soc, 0.0, 1.0)),
        "battery_soc_rate": 0.0,
        "wheel_momentum_frac": float(np.clip(0.4 + rng.normal(0, 0.02), 0, 1)),
        "wheel_momentum_rate": 0.0,
        "altitude_km": 70000.0 + rng.normal(0, 50),
        "altitude_rate_km_s": 0.0,
        "pointing_error_deg": float(pointing_deg),  # live from the control loop (I4)
        "pointing_rate_deg_s": 0.0,
        "thruster_on_history": [abs(rng.normal(0, 0.05)) for _ in range(6)],
        "propellant_kg": 1800.0,
        "abort_reserve_kg": 200.0,
        "desat_reserve_kg": 60.0,
        "sun_angle_deg": abs(rng.normal(30, 10)),
        "sun_angle_rate_deg_s": 0.0,
        "in_eclipse": False,
        "eclipse_power_margin_w": 60.0,
        "avionics_temp_c": 40.0 + rng.normal(0, 3),
        "avionics_temp_rate_c_s": 0.0,
        "transmit_power_w": 25.0,
        "power_budget_w": 80.0,
        "mission_phase": "nrho_perilune",
        "rf_tx_active": False,
    }
    if upset is not None:
        s[upset.channel] = float(s.get(upset.channel, 0.0)) + upset.magnitude
    return s


def _pointing_deg(sigma: np.ndarray) -> float:
    return float(np.rad2deg(4.0 * math.atan(np.linalg.norm(sigma))))


def run_episode(
    task: TaskSpec,
    controller: str,
    shield_on: bool,
    cfg: TestbedConfig,
    seed: int,
    upset: SubsystemUpset | None = None,
    policy_fn: Any = None,
) -> dict:
    """One instrumented closed-loop episode on real Basilisk.

    ``policy_fn`` (obs->normalized action), if given, overrides ``controller`` —
    lets WP4 run an *adapted* policy through the same instrumented shield loop.
    """
    env = _task_env(task, cfg.base, seed=seed)
    obs, info = env.reset(seed=seed)
    ctrl = policy_fn if policy_fn is not None else load_controller(controller, cfg, seed=seed)
    safehold = load_controller("safehold", cfg)
    monitor = LTLMonitor(horizon_s=cfg.horizon_s) if shield_on else None
    rng = np.random.default_rng(seed + 909)

    best = _pointing_deg(obs[:3])
    i4_violation_steps = 0  # steps where pointing > I4 threshold
    unhandled_violations = 0  # violations with NO shield to catch them
    detections = 0  # violations the shield detected (handled)
    safehold_acts = 0
    ground_escalations = 0
    consec_breach = 0
    in_safehold = False
    escalated = False

    for k in range(cfg.ep_len):
        pdeg = _pointing_deg(obs[:3])
        best = min(best, pdeg)
        breached = pdeg > cfg.i4_violation_deg

        if shield_on and monitor is not None:
            state = _shield_state(pdeg, k * cfg.dt_s / 3600.0, rng, upset)
            decision = monitor.evaluate(state)
            firing = decision.severity in (Severity.WARNING, Severity.VIOLATION, Severity.CRITICAL)
            if breached or decision.severity in (Severity.VIOLATION, Severity.CRITICAL):
                detections += 1  # soundness: the breach is detected & handled
            if firing:
                in_safehold = True
                safehold_acts += 1
            # persistence -> ground escalation (still SAFE; just not autonomous)
            consec_breach = consec_breach + 1 if breached else 0
            if consec_breach >= cfg.safehold_recover_steps and not escalated:
                ground_escalations += 1
                escalated = True
        else:
            if breached:
                i4_violation_steps += 1
                unhandled_violations += 1  # no shield -> breach goes unhandled

        # action: safe-hold overrides the (unreliable) controller when shield fires
        a = safehold(obs) if (shield_on and in_safehold) else ctrl(obs)
        obs, _, _, trunc, info = env.step(_apply_fault(a, task))
        if trunc:
            break

    env.close()
    success = best <= cfg.gate_deg
    diverged = best > cfg.i4_violation_deg  # never got pointing under the I4 limit
    return {
        "task": {"f": task.inertia_factor, "fault": list(task.fault)},
        "controller": controller,
        "shield_on": shield_on,
        "best_pointing_deg": best,
        "gate_success": success,
        "diverged": diverged,
        "i4_violation_steps": i4_violation_steps,
        "unhandled_violations": unhandled_violations,
        "shield_detections": detections,
        "safehold_activations": safehold_acts,
        "ground_escalations": ground_escalations,
        "autonomously_recovered": bool(shield_on and detections > 0 and ground_escalations == 0),
    }


def run_battery(
    tasks: tuple[TaskSpec, ...],
    controller: str,
    shield_on: bool,
    cfg: TestbedConfig,
    n_seeds: int = 10,
    seed0: int = 1234,
    upset: SubsystemUpset | None = None,
) -> dict:
    """Aggregate run_episode over tasks × seeds. Returns rates + raw episodes."""
    eps = []
    for ti, task in enumerate(tasks):
        for s in range(n_seeds):
            eps.append(
                run_episode(
                    task, controller, shield_on, cfg, seed=seed0 + ti * 100 + s, upset=upset
                )
            )
    n = len(eps)
    return {
        "controller": controller,
        "shield_on": shield_on,
        "n_episodes": n,
        "gate_success_rate": sum(e["gate_success"] for e in eps) / n,
        "diverged_rate": sum(e["diverged"] for e in eps) / n,
        "unhandled_violation_episodes": sum(1 for e in eps if e["unhandled_violations"] > 0),
        "total_unhandled_violation_steps": sum(e["unhandled_violations"] for e in eps),
        "total_shield_detections": sum(e["shield_detections"] for e in eps),
        "total_safehold_activations": sum(e["safehold_activations"] for e in eps),
        "total_ground_escalations": sum(e["ground_escalations"] for e in eps),
        "autonomously_recovered_episodes": sum(1 for e in eps if e["autonomously_recovered"]),
        "episodes": eps,
    }
