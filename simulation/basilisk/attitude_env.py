"""High-fidelity Basilisk attitude-control environment (curriculum Phase 1).

Replaces the synthetic linearized rigid-body sim in
``controller/maml/task_sampler.py`` with real Basilisk 6-DOF rigid-body
attitude dynamics under direct 3-axis external-torque actuation, at the
controller-tier control rate (default 0.5 s sub-step, matching the
synthetic sim's ``dt_sub_target``).

Why Basilisk-direct (not bsk_rl): bsk_rl operates at the tasking level
(image / charge / desat decisions via a 60-s setpoint FSW). The
controller tier is a 10 Hz inner-loop GNC controller that emits torque /
thrust directly, so we drive Basilisk's ``spacecraft`` +
``extForceTorque`` modules ourselves rather than through bsk_rl's
tasking abstraction. (bsk_rl remains the right tool for commander /
scheduler tasking validation.)

Gymnasium API so the env plugs into standard RL trainers and our own
controller. Observation: [sigma_BN(3), omega_BN_B(3)] (MRP attitude +
body rate). Action: torque command in [-1,1]^3, scaled by
``max_torque_nm``. Reward shapes toward the proposal's Phase-1 target
(<=0.2 deg within 200 s); ``success`` is exposed for the phase-aware MC.

Domain randomization (inertia, initial attitude/rate, max torque) is
applied per-reset and is compatible with ``simulation.domain_randomization``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as _e:  # pragma: no cover
    raise RuntimeError("gymnasium required: pip install gymnasium") from _e


def mrp_angle_deg(sigma: np.ndarray) -> float:
    """Rotation angle (deg) represented by an MRP set, from identity.

    phi = 4 * atan(|sigma|).
    """
    return math.degrees(4.0 * math.atan(float(np.linalg.norm(sigma))))


@dataclass
class AttitudeEnvConfig:
    """Phase-1 attitude-slew env configuration."""

    dt_s: float = 0.5
    episode_length: int = 400  # 400 * 0.5 s = 200 s (proposal Phase 1)
    max_torque_nm: float = 0.2  # per-axis actuator authority
    mass_kg: float = 100.0
    inertia_diag: tuple[float, float, float] = (10.0, 8.0, 6.0)
    # Initial-condition ranges (domain randomization)
    init_sigma_max: float = 0.3  # |sigma| up to 0.3 (~ 66 deg)
    init_omega_max_rad_s: float = 0.05
    # Reward weights
    rate_penalty: float = 0.01
    torque_penalty: float = 0.001
    # Success criterion (proposal Phase 1: <=0.2 deg). Settling within episode.
    success_pointing_deg: float = 0.2
    seed: int = 0
    # --- Reaction-wheel-model mode (OPT-IN; default off keeps the external-torque
    # path byte-identical, see Paper-C C-4) -------------------------------------
    # When True, three body-axis-aligned reaction wheels (Basilisk
    # reactionWheelStateEffector) produce body torque via real wheel dynamics
    # (speed/momentum, optional friction, motor-torque + momentum saturation)
    # instead of a direct external torque. The action still encodes a desired body
    # torque (action * max_torque_nm); the env converts it to per-wheel motor
    # commands (motor_i = -tau_body_i, since a wheel's reaction torque on the body
    # is opposite its spin-up torque), so the SAME control laws and the SAME
    # upstream actuator fault (applied = action*g + b) apply unchanged — the fault
    # now degrades/reverses/biases the WHEEL torque command.
    use_reaction_wheels: bool = False
    rw_max_momentum_nms: float = 50.0  # per-wheel momentum ceiling H_max (Nms)
    rw_max_torque_nm: float = 0.2  # per-wheel motor-torque saturation (matches authority)
    rw_use_friction: bool = False  # Coulomb/viscous wheel friction (HR16 defaults)
    rw_initial_speed_rpm: float = 0.0  # initial wheel speed (RPM); 0 = at rest


class BasiliskAttitudeEnv(gym.Env):
    """Gymnasium env wrapping a direct Basilisk attitude sim."""

    metadata: dict[str, list] = {"render_modes": []}

    def __init__(self, config: AttitudeEnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or AttitudeEnvConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._step_idx = 0
        self._sigma = np.zeros(3)
        self._omega = np.zeros(3)
        self._best_pointing_deg = 180.0
        # Basilisk handles (SWIG objects), built per-reset
        self._sc: Any = None
        self._ext: Any = None
        self._scSim: Any = None
        self._rec: Any = None
        self._tmsg: Any = None
        self._cmd_payload: Any = None
        self._macros: Any = None
        self._messaging: Any = None
        # Reaction-wheel-model handles (only populated when use_reaction_wheels)
        self._rw_eff: Any = None
        self._rw_speed_rec: Any = None
        self._wheel_momentum_frac: float = 0.0
        self._rw_js: float = 0.0

    # -- Basilisk lifecycle -------------------------------------------------

    def _build_sim(self) -> None:
        from Basilisk.architecture import messaging
        from Basilisk.simulation import spacecraft
        from Basilisk.utilities import SimulationBaseClass, macros

        cfg = self.config
        scSim = SimulationBaseClass.SimBaseClass()  # noqa: N806 (Basilisk convention)
        proc = scSim.CreateNewProcess("attproc")
        dt_ns = macros.sec2nano(cfg.dt_s)
        proc.addTask(scSim.CreateNewTask("atttask", dt_ns))

        sc = spacecraft.Spacecraft()
        sc.ModelTag = "ample_sat"
        sc.hub.mHub = cfg.mass_kg
        ix, iy, iz = cfg.inertia_diag
        sc.hub.IHubPntBc_B = [[ix, 0.0, 0.0], [0.0, iy, 0.0], [0.0, 0.0, iz]]
        sc.hub.sigma_BNInit = [[self._sigma[0]], [self._sigma[1]], [self._sigma[2]]]
        sc.hub.omega_BN_BInit = [[self._omega[0]], [self._omega[1]], [self._omega[2]]]
        scSim.AddModelToTask("atttask", sc)

        if cfg.use_reaction_wheels:
            self._build_reaction_wheels(scSim, sc, messaging, macros)
        else:
            self._build_external_torque(scSim, sc, messaging)

        rec = sc.scStateOutMsg.recorder()
        scSim.AddModelToTask("atttask", rec)

        self._scSim = scSim
        self._sc = sc
        self._rec = rec
        self._macros = macros
        self._messaging = messaging

    def _build_external_torque(self, scSim: Any, sc: Any, messaging: Any) -> None:  # noqa: N803
        """Direct external-torque actuator (the audited, default path).

        InitializeSimulation BEFORE wiring the command message, exactly as the
        original implementation, so this branch is byte-identical to the committed
        external-torque env.
        """
        from Basilisk.simulation import extForceTorque

        ext = extForceTorque.ExtForceTorque()
        ext.ModelTag = "ext_torque"
        sc.addDynamicEffector(ext)
        scSim.AddModelToTask("atttask", ext)

        scSim.InitializeSimulation()

        cmd = messaging.CmdTorqueBodyMsgPayload()
        cmd.torqueRequestBody = [0.0, 0.0, 0.0]
        tmsg = messaging.CmdTorqueBodyMsg().write(cmd)
        ext.cmdTorqueInMsg.subscribeTo(tmsg)

        self._ext = ext
        self._tmsg = tmsg
        self._cmd_payload = cmd

    def _build_reaction_wheels(self, scSim: Any, sc: Any, messaging: Any, macros: Any) -> None:  # noqa: N803
        """Three body-axis-aligned reaction wheels (Basilisk RW state effector).

        Each wheel's spin axis is a body axis (x, y, z), so an array motor-torque
        command maps one-to-one onto the per-axis body reaction torque (to first
        order, minus the small wheel-inertia coupling). The wheels carry real
        momentum, saturate (motor torque and momentum), and optionally have
        friction — the fidelity this mode adds over the external-torque idealisation.
        """
        from Basilisk.simulation import reactionWheelStateEffector
        from Basilisk.utilities import simIncludeRW

        cfg = self.config
        rwf = simIncludeRW.rwFactory()
        for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
            rw_cfg = rwf.create(
                "Honeywell_HR16",
                axis,
                Omega=float(cfg.rw_initial_speed_rpm),
                maxMomentum=float(cfg.rw_max_momentum_nms),
                useRWfriction=bool(cfg.rw_use_friction),
                useMaxTorque=True,
                u_max=float(cfg.rw_max_torque_nm),
            )
            # Wheel spin inertia Js (the factory derives it from maxMomentum/Omega_max,
            # identical for all three wheels); used to convert wheel speed -> momentum
            # fraction |Js*Omega|/H_max for the I2 wheel-momentum invariant.
            self._rw_js = float(rw_cfg.Js)
        rw_eff = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwf.addToSpacecraft("ample_rw", rw_eff, sc)
        scSim.AddModelToTask("atttask", rw_eff)

        scSim.InitializeSimulation()

        # ArrayMotorTorqueMsg carries the per-wheel motor torque command.
        cmd = messaging.ArrayMotorTorqueMsgPayload()
        cmd.motorTorque = [0.0, 0.0, 0.0]
        tmsg = messaging.ArrayMotorTorqueMsg().write(cmd)
        rw_eff.rwMotorCmdInMsg.subscribeTo(tmsg)

        speed_rec = rw_eff.rwSpeedOutMsg.recorder()
        scSim.AddModelToTask("atttask", speed_rec)

        self._rw_eff = rw_eff
        self._tmsg = tmsg
        self._cmd_payload = cmd
        self._rw_speed_rec = speed_rec
        self._wheel_momentum_frac = 0.0

    # -- Gym API ------------------------------------------------------------

    def reset(  # noqa: ANN201
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.config
        # Randomized initial attitude (random axis, |sigma| up to max) + rate
        v = self._rng.normal(size=3)
        v /= np.linalg.norm(v) + 1e-12
        self._sigma = v * self._rng.uniform(0.0, cfg.init_sigma_max)
        self._omega = self._rng.uniform(-cfg.init_omega_max_rad_s, cfg.init_omega_max_rad_s, size=3)
        self._step_idx = 0
        self._best_pointing_deg = mrp_angle_deg(self._sigma)
        self._build_sim()
        return self._obs(), {"pointing_deg": mrp_angle_deg(self._sigma)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.config
        torque = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * cfg.max_torque_nm
        if cfg.use_reaction_wheels:
            # Desired body torque -> per-wheel motor command. A wheel's reaction
            # torque on the body is opposite the spin-up torque, so motor = -tau_body
            # for body-aligned wheels (the env clips action to authority above; the
            # RW effector additionally enforces its own u_max motor saturation).
            self._cmd_payload.motorTorque = [
                float(-torque[0]),
                float(-torque[1]),
                float(-torque[2]),
            ]
        else:
            # Push commanded torque into the standalone message (audited path)
            self._cmd_payload.torqueRequestBody = [
                float(torque[0]),
                float(torque[1]),
                float(torque[2]),
            ]
        self._tmsg.write(self._cmd_payload, self._scSim.TotalSim.CurrentNanos)
        # Advance one control step
        self._scSim.ConfigureStopTime(self._macros.sec2nano((self._step_idx + 1) * cfg.dt_s))
        self._scSim.ExecuteSimulation()
        sig = np.array(self._rec.sigma_BN)[-1]
        om = np.array(self._rec.omega_BN_B)[-1]
        self._sigma = sig
        self._omega = om
        if cfg.use_reaction_wheels:
            # |H|/H_max per wheel (max of 3): the real I2 wheel-momentum invariant
            # signal, now from actual wheel state instead of external injection.
            speeds = np.array(self._rw_speed_rec.wheelSpeeds)[-1][:3]
            h_max = float(cfg.rw_max_momentum_nms)
            # Per-wheel momentum H_i = Js * Omega_i; the I2 fraction is the worst
            # wheel's |H_i| / H_max (Js captured from the factory at build time).
            self._wheel_momentum_frac = float(
                np.max(np.abs(speeds)) * self._rw_js / h_max if h_max > 0 else 0.0
            )

        pointing_deg = mrp_angle_deg(sig)
        self._best_pointing_deg = min(self._best_pointing_deg, pointing_deg)
        rate_norm = float(np.linalg.norm(om))
        torque_norm = float(np.linalg.norm(torque))
        reward = (
            -((pointing_deg / 180.0) ** 2)
            - cfg.rate_penalty * rate_norm
            - cfg.torque_penalty * torque_norm
        )
        if pointing_deg < cfg.success_pointing_deg:
            reward += 1.0
        elif pointing_deg < 5.0 * cfg.success_pointing_deg:
            reward += 0.5
        elif pointing_deg < 25.0 * cfg.success_pointing_deg:
            reward += 0.2

        self._step_idx += 1
        truncated = self._step_idx >= cfg.episode_length
        terminated = False
        info = {
            "pointing_deg": pointing_deg,
            "best_pointing_deg": self._best_pointing_deg,
            "rate_norm": rate_norm,
            "success": self._best_pointing_deg < cfg.success_pointing_deg,
        }
        if cfg.use_reaction_wheels:
            # Real I2 wheel-momentum-invariant signal (only in RW mode; the
            # external-torque info dict stays byte-identical to the audited path).
            info["wheel_momentum_frac"] = self._wheel_momentum_frac
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._sigma, self._omega]).astype(np.float32)

    def close(self) -> None:
        self._scSim = None
        self._sc = None
        self._ext = None
        self._rec = None
        self._rw_eff = None
        self._rw_speed_rec = None


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
def _smoke(n_steps: int = 50) -> dict:
    """Run a zero-torque + bang-bang episode to confirm dynamics + reward."""
    env = BasiliskAttitudeEnv(AttitudeEnvConfig(episode_length=n_steps, seed=1))
    obs, info = env.reset(seed=1)
    start_deg = info["pointing_deg"]
    total_r = 0.0
    for _ in range(n_steps):
        # Simple proportional-ish controller: torque opposes sigma to null it
        sigma = obs[:3]
        action = -np.sign(sigma)  # bang-bang toward identity
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        if trunc:
            break
    env.close()
    return {
        "start_pointing_deg": round(start_deg, 3),
        "final_pointing_deg": round(info["pointing_deg"], 3),
        "best_pointing_deg": round(info["best_pointing_deg"], 3),
        "total_reward": round(total_r, 3),
        "success": info["success"],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(_smoke(), indent=2))
