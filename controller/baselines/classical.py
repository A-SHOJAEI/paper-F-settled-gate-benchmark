"""Classical (non-learned) fault-tolerant attitude controllers — WS1 baselines.

Honest comparators for the learned RMA student. Each builds a ``policy(obs) -> a``
closure (normalised command in [-1, 1]^3) compatible with ``program.rollout``, so every
controller is scored on the identical settled-gate harness.

- ``pid_unaware_policy`` — fault-UNAWARE PID (PD + integral, anti-windup). Assumes a
  nominal +1 actuator; a naive floor that also shows integral action alone cannot help
  when the actuator sign/gain is wrong.
- ``adaptive_fl_policy`` — fixed nominal-slope feedback-linearising PD with online
  actuator-SIGN detection. A controlled diagnostic showed gain-MAGNITUDE error is benign
  (it does not move the regulate-to-zero equilibrium); the one online unknown that
  matters is the actuator SIGN, which it resolves from the sign of the command<->
  acceleration correlation. The CLASSICAL counterpart to the learned student on the
  failure mode that matters (sign reversal) — no privilege, no training. Handles SIGN and
  GAIN; fails GAIN_BIAS (no bias rejection) and TOTAL_LOSS (dead axis) — the same two
  classes the student also fails, so the head-to-head stays fair.

The desired closed-loop angular acceleration matches the privileged teacher's,
``alpha_des = (-kp*sigma - kd*omega)/I_base`` per axis, so with the correct sign the
controller reproduces the teacher — the residual gap is the online sign-resolution
transient, the same kind of online-adaptation challenge the learned student faces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ACT = 3
# Mirror controller.maml.meta_imitation_basilisk.BASE_INERTIA (the env hub inertia).
BASE_INERTIA = (10.0, 8.0, 6.0)

PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class ClassicalConfig:
    """Gains/knobs for the classical baselines. kp/kd match MetaConfig (kp0/kd0); the
    rest are engineering estimates tuned on TRAIN faults only."""

    kp: float = 0.2
    kd: float = 1.5
    ki: float = 0.0  # integral gain (0 = PD); only aids the bias case, off by default
    max_torque_nm: float = 0.2
    dt_s: float = 0.5  # control step (matches AttitudeEnvConfig.dt_s / the eval)
    i_clip: float = 0.5  # anti-windup clamp on the integral of sigma
    s_floor_frac: float = 0.2  # min |effectiveness estimate| (avoids divide blow-up)
    # exc_thresh / sign_eps tuned on TRAIN GAIN+SIGN faults (argmax settled-science).
    exc_thresh: float = 0.05  # only count a sign-vote sample when |command| exceeds this
    # (near equilibrium the command->accel map is uninformative about the actuator sign)
    sign_eps: float = 0.005  # min |correlation vote| before latching the detected sign
    # ICL adaptive baseline (adapted from arXiv:2504.12124): gradient + integral-
    # concurrent-learning effectiveness estimation. Gains are engineering estimates.
    icl_gamma: float = 0.3  # gradient/learning rate on the effectiveness estimate
    icl_k1: float = 5.0  # weight on the integral-concurrent-learning (history) term
    icl_buf: int = 20  # number of buffered informative samples for the ICL term
    # Nussbaum-gain baseline (unknown control direction; Nussbaum 1983, Hu et al.
    # IJRNC 2018). Values are set by program.tune_nussbaum on TRAIN SIGN+GAIN only
    # (evidence/program/nussbaum_tune.json).
    nuss_gamma: float = 5.0  # adaptation rate on the Nussbaum argument k
    nuss_lambda: float = 0.13  # composite-error weight: s = omega + lambda*sigma
    nuss_c: float = 0.5  # nominal-law gain in s-coordinates: u_bar = -c*s/s_nom
    nuss_k0: float = 1.0  # initial k (N(k0) sets the initial effective gain)
    nuss_kmax: float = 25.0  # |k| clip (bounds the N-sweep in discrete time)
    nuss_deadzone: float = 0.0  # freeze adaptation when |s| below this (anti-drift)


def pid_unaware_policy(cfg: ClassicalConfig) -> PolicyFn:
    """Fault-unaware PID assuming a nominal +1 actuator (applied = command)."""
    integ = np.zeros(ACT)

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal integ
        sigma, omega = obs[:3], obs[3:]
        integ = np.clip(integ + sigma, -cfg.i_clip, cfg.i_clip)
        u = (-cfg.kp * sigma - cfg.kd * omega - cfg.ki * integ) / cfg.max_torque_nm
        return np.clip(u, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    return policy


def adaptive_fl_policy(cfg: ClassicalConfig) -> PolicyFn:
    """Fixed-magnitude feedback-linearising PD + online actuator-SIGN detection.

    A controlled diagnostic showed the gain MAGNITUDE need not be estimated: a fixed
    nominal-slope feedback-linearising PD settles nominal AND wrong-magnitude gain faults
    to ~0 deg (a loop-gain error does not move the regulate-to-zero equilibrium). The one
    online unknown that matters is the actuator SIGN (a reversed axis is unstable under
    the +1 assumption). So this baseline fixes the slope magnitude at nominal and detects
    each axis's sign from the command<->acceleration correlation ``sign(sum a*alpha)``,
    flipping within a few excited steps. It is the classical counterpart to the learned
    student on the failure mode that matters (sign reversal). It handles SIGN and GAIN;
    it does not reject an additive bias (GAIN_BIAS) or actuate a dead axis (TOTAL_LOSS) —
    the same two classes the student also fails, so the head-to-head stays fair.
    """
    base = np.asarray(BASE_INERTIA, dtype=float)
    s_nom = cfg.max_torque_nm / base  # nominal |command->accel| slope per axis
    dt = cfg.dt_s

    integ = np.zeros(ACT)
    sign_vote = np.zeros(ACT)  # running corr(command, accel); its sign = the actuator sign
    latched = np.zeros(ACT)  # per-axis sign once decided (0 = undecided -> assume +1)
    prev_a: list[np.ndarray] = []
    prev_omega: list[np.ndarray] = []

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal integ
        sigma, omega = obs[:3].copy(), obs[3:].copy()

        # Accumulate the sign vote from the previous command and the resulting accel.
        if prev_a:
            accel = (omega - prev_omega[0]) / dt
            for i in range(ACT):
                if abs(prev_a[0][i]) > cfg.exc_thresh:  # only informative (excited) samples
                    sign_vote[i] += prev_a[0][i] * accel[i]

        integ = np.clip(integ + sigma, -cfg.i_clip, cfg.i_clip)
        alpha_des = (-cfg.kp * sigma - cfg.kd * omega - cfg.ki * integ) / base
        # Latch each axis's sign at the first confident vote (the open-loop correlation);
        # once controlling correctly the closed-loop a*accel sign weakens, so without a
        # latch the detector chatters and never stabilises.
        for i in range(ACT):
            if latched[i] == 0.0 and abs(sign_vote[i]) > cfg.sign_eps:
                latched[i] = float(np.sign(sign_vote[i]))
        det_sign = np.where(latched != 0.0, latched, 1.0)
        a: np.ndarray = np.clip(alpha_des / (det_sign * s_nom), -1.0, 1.0).astype(np.float32)

        prev_a.clear()
        prev_a.append(a.copy())
        prev_omega.clear()
        prev_omega.append(omega)
        return a  # type: ignore[no-any-return]

    return policy


def nussbaum_policy(cfg: ClassicalConfig) -> PolicyFn:
    """Per-axis Nussbaum-gain adaptive PD — the CLASSICAL answer to an unknown
    control DIRECTION (Nussbaum 1983; spacecraft attitude with actuator faults:
    Hu et al., IJRNC 2018). The nominal feedback-linearising PD command ``u_bar``
    is multiplied by the Nussbaum function ``N(k) = k^2 cos(k)`` and ``k``
    integrates ``gamma * s * u_bar`` (``s = omega + lambda*sigma`` the composite
    error), so the loop SEARCHES the effective-gain axis — sweeping magnitude AND
    sign — until the error settles; no sign detector, no learned estimate.

    Discrete-time practicalities at dt=0.5 s with saturation: a dead-zone on
    ``|s|`` freezes adaptation near equilibrium (the classical analogue of the
    student's latch — without it k drifts on noise and N(k) walks through a
    zero-gain region), and ``k`` is clipped. Saturation makes large |N(k)|
    effectively bang-bang, which is benign here: the repo's controlled
    diagnostic showed gain-MAGNITUDE error does not move the regulate-to-zero
    equilibrium — the search only has to land a stabilising SIGN.
    """
    base = np.asarray(BASE_INERTIA, dtype=float)
    s_nom = cfg.max_torque_nm / base  # nominal |command->accel| slope per axis
    dt = cfg.dt_s
    k = np.full(ACT, cfg.nuss_k0, dtype=float)

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal k
        sigma, omega = obs[:3].copy(), obs[3:].copy()
        # Nominal law in composite-error coordinates, u_bar = -c*s (the canonical
        # construction): then k_dot = gamma*s*u_bar = -gamma*c*s^2 <= 0 is MONOTONE,
        # so the sweep walks N(k)'s alternating-sign lobes without chattering and
        # freezes exactly when the error stops persisting. (Re-using a kp/kd law
        # with mismatched lambda breaks this monotonicity — measured on TRAIN.)
        s = omega + cfg.nuss_lambda * sigma
        u_bar = -cfg.nuss_c * s / s_nom
        active = (np.abs(s) > cfg.nuss_deadzone).astype(float)
        k = np.clip(k + cfg.nuss_gamma * s * u_bar * dt * active, -cfg.nuss_kmax, cfg.nuss_kmax)
        n_k = k * k * np.cos(k)
        return np.clip(n_k * u_bar, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    return policy


def icl_adaptive_policy(cfg: ClassicalConfig) -> PolicyFn:
    """ICL adaptive FTC, adapted from arXiv:2504.12124 (Lyapunov + integral concurrent
    learning). Estimates the per-axis actuator effectiveness online via a GRADIENT term
    plus an INTEGRAL-CONCURRENT-LEARNING term over buffered input/output samples, feeding
    a feedback-linearising control law u = J*accel_des(+gyro) / (phi_hat * torque).

    Adaptation to our testbed: the paper estimates a reaction-wheel HEALTH matrix
    Phi in [0,1] (degradation only) for a wheel array; we map it to 3-axis direct-torque
    control and let phi_hat be SIGNED so the mechanism can also attempt the sign-reversal
    faults in our taxonomy (an explicit EXTENSION — the original projects to [0,1]). The
    lumped estimate absorbs the unknown inertia scale, so it feedback-linearises both.
    """
    base = np.asarray(BASE_INERTIA, dtype=float)  # nominal J = diag(base)
    t_max = cfg.max_torque_nm
    dt = cfg.dt_s
    phi = np.ones(ACT)  # signed effectiveness estimate (init nominal +1)
    buf: list[tuple[np.ndarray, np.ndarray]] = []  # ICL history: (regressor y, measured m)
    prev_u: list[np.ndarray] = []
    prev_w: list[np.ndarray] = []

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal phi
        sigma, omega = obs[:3].copy(), obs[3:].copy()
        if prev_u:
            # measured control torque m = J*omega_dot + gyro = phi_true * u * t_max
            m = base * (omega - prev_w[0]) / dt + np.cross(prev_w[0], base * prev_w[0])
            y = prev_u[0] * t_max  # regressor: m_i = y_i * phi_i (diagonal)
            phi = phi + cfg.icl_gamma * y * (m - y * phi)  # gradient term
            if np.abs(prev_u[0]).max() > cfg.exc_thresh:  # buffer informative samples
                buf.append((y.copy(), m.copy()))
                if len(buf) > cfg.icl_buf:
                    buf.pop(0)
            if buf:  # integral-concurrent-learning term over the history
                icl = np.zeros(ACT)
                for yi, mi in buf:
                    icl = icl + yi * (mi - yi * phi)
                phi = phi + cfg.icl_gamma * cfg.icl_k1 * icl / len(buf)
        tau_d = -cfg.kp * sigma - cfg.kd * omega + np.cross(omega, base * omega)
        phi_eff = np.sign(phi) * np.maximum(np.abs(phi), cfg.s_floor_frac)
        phi_eff = np.where(phi_eff == 0.0, cfg.s_floor_frac, phi_eff)
        u = np.clip(tau_d / (phi_eff * t_max), -1.0, 1.0).astype(np.float32)
        prev_u.clear()
        prev_u.append(u.astype(float))
        prev_w.clear()
        prev_w.append(omega)
        return u  # type: ignore[no-any-return]

    return policy
