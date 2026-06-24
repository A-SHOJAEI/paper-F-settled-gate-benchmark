"""RMA fault-adaptive attitude controller: privileged teacher + history student.

The deployed controller is a fault-parameterised control law whose latent fault
input is supplied online by a *learned* recurrent adaptation module — the part
nominal BC (0%), pooled-BC and MAML (transfer ratio 1.0) all failed at.

Fault model: a per-axis actuator-effectiveness gain g (applied torque = command·g).
  - g ∈ {±1}³           : the testbed's actuator-sign faults (battery-comparable).
  - g ∈ ±[0.3,1.5] / axis: continuous reversal + degradation + over-effectiveness
                           (the harder, realistic generalisation).

  teacher (privileged) — π(obs, z), z = [g₀,g₁,g₂, f−1]: inertia-scaled PD with
      the command divided by the effectiveness, a = clip(u / g). With true z it
      reaches the 0.2° gate (the upper bound).
  student (learned)    — a GRU over [obs, prev-action, Δobs] → ẑ. The gain is
      identifiable from Δω / command, so Δobs is the key signal the obs-only
      GRU-BC lacked. z-regression on teacher rollouts + on-policy DAgger.
  deploy: π(obs, ẑ) with ẑ = student(history) — fault-adaptive, no privileged input.

Run:  python -m controller.rma.rma_attitude --phase all
Eval on the WP0 sign-fault battery AND a held-out continuous-fault set, vs PD
(fault-unaware) and nominal BC. Writes evidence/program/rma_controller.json.
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from controller.maml.meta_imitation_basilisk import MetaConfig, TaskSpec, _task_env

logger = logging.getLogger(__name__)

OBS, ACT, Z = 6, 3, 4
FEAT = OBS + ACT + OBS  # [obs, prev_action, Δobs] = 15
G_MIN, G_MAX = 0.3, 1.5  # continuous-fault effectiveness magnitude range
STUDENT_CKPT = "checkpoints/rma_student.pt"
STUDENT_V2_CKPT = "checkpoints/rma_student_v2.pt"  # taxonomy-TRAIN-conformant retrain
OUT = Path("evidence/program/rma_controller.json")
Task = tuple[float, np.ndarray]  # (inertia factor f, per-axis gain g[3])


def env_cfg() -> MetaConfig:
    # match the WP0 testbed env (400 steps, 0.2 N·m) so the policy plugs in directly
    return MetaConfig(ep_len=400, max_torque_nm=0.2)


def task_z(f: float, g: np.ndarray) -> np.ndarray:
    return np.array([g[0], g[1], g[2], f - 1.0], dtype=np.float32)


def sample_task(rng: np.random.Generator, mode: str) -> Task:
    f = float(rng.uniform(0.7, 2.2))
    if mode == "sign":
        g = np.array([rng.choice([-1.0, 1.0]) for _ in range(3)], dtype=np.float32)
    else:  # continuous: reversed + degraded + over-effective
        g = np.array(
            [rng.choice([-1.0, 1.0]) * rng.uniform(G_MIN, G_MAX) for _ in range(3)],
            dtype=np.float32,
        )
    return f, g


def sample_task_taxonomy(rng: np.random.Generator) -> Task:
    """Draw a training task from the DECLARED taxonomy TRAIN split (WS0).

    The v1 student was trained via ``sample_task``, whose support
    (f in U(0.7, 2.2), |g| in U(0.3, 1.5), independent per-axis signs) overlaps
    the held-out TEST regions of ``program.fault_taxonomy`` (audit residual
    M2/F-1): only |g| in (1.5, 2.0], f in (2.2, 2.3], and bias were truly
    extrapolative for that checkpoint. This sampler draws 50/50 SIGN/GAIN
    faults from ``Split.TRAIN`` instead, so a student trained with it has
    never seen ANY test cell — >=2-flip sign patterns, |g| in
    [0.3, 0.5) and (1.5, 2.0], and all TEST inertias are extrapolative by
    construction.
    """
    from program.fault_taxonomy import FaultClass, Split, sample_fault

    fclass = FaultClass.SIGN if rng.random() < 0.5 else FaultClass.GAIN
    fault = sample_fault(rng, fclass, Split.TRAIN)
    return fault.f, fault.g_arr()


def _sample_for_mode(rng: np.random.Generator, mode: str) -> Task:
    return sample_task_taxonomy(rng) if mode == "taxonomy" else sample_task(rng, mode)


def _g_eff(z: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Effectiveness used by the control law; clamp magnitude away from 0. A positive
    ``margin`` conservatively inflates the estimated effectiveness (the command is then
    divided by a larger number), lowering the effective loop gain — this suppresses the
    limit-cycle oscillation that an UNDER-estimated gain otherwise induces near the
    target. ``margin = 0`` is the audited no-op."""
    g = z[:3]
    # Treat a zero (or near-zero) entry as +1 direction so the divisor stays at >= G_MIN:
    # np.sign(0)=0 would make g_eff=0 -> a divide-by-zero in the control law on a fully
    # dead actuator axis (the TOTAL_LOSS class). The axis is uncontrollable either way
    # (applied = command*0 = 0); this only keeps the command finite instead of nan.
    sign = np.where(g < 0.0, -1.0, 1.0)
    eff = sign * np.clip(np.abs(g), G_MIN, None) * (1.0 + margin)
    return eff.astype(np.float32)  # type: ignore[no-any-return]


def analytic_teacher(
    obs: np.ndarray,
    z: np.ndarray,
    cfg: MetaConfig,
    *,
    kd_scale: float = 1.0,
    g_margin: float = 0.0,
    b_ff: Any = 0.0,
) -> np.ndarray:
    """Privileged law π(obs, z): inertia-scaled PD, command divided by the actuator
    effectiveness so the *applied* torque (command·g) is the desired PD torque.

    ``kd_scale`` and ``g_margin`` default to a no-op, so the audited privileged-teacher
    and training paths are unchanged. The DEPLOYED student policy may raise them to
    robustify steady-state holding against the gain-estimate error that drives the
    limit cycles diagnosed on held-out faults (see ``rma_policy``).

    ``b_ff`` (default 0 = audited no-op) is a constant-actuator-bias FEEDFORWARD: the
    fault model applies ``command·g + b`` (``fault_taxonomy``), so an integral-free PD
    law cannot null a constant ``b`` and leaves a steady-state offset (the GAIN_BIAS
    class scores 0% for every controller including the oracle). Subtracting ``b_ff`` in
    command space before dividing by the effectiveness gives ``a = (u − b_ff)/g_eff`` so
    the applied command is ``a·g + b = u`` when ``b_ff = b`` — exact cancellation (the
    privileged upper bound for GAIN_BIAS). Deployed online via a bias observer (Paper C)."""
    sigma, omega = obs[:3], obs[3:]
    f = 1.0 + float(z[3])
    kp, kd = cfg.kp0 * f, cfg.kd0 * f * kd_scale
    u = (-kp * sigma - kd * omega) / cfg.max_torque_nm
    a = (u - b_ff) / _g_eff(z, g_margin)
    return np.clip(a, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# student (adaptation module)
# --------------------------------------------------------------------------- #
class RMAStudent(nn.Module):
    """GRU over standardised [obs, prev-action, Δobs] → ẑ. Norm stats are buffers."""

    # Declare buffer types so mypy treats them as Tensors (register_buffer otherwise
    # yields Tensor | Module). Annotation-only; behaviour is unchanged.
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, h: int = 96, z_dim: int = Z) -> None:
        super().__init__()
        self.h = h
        self.z_dim = z_dim
        self.gru = nn.GRU(FEAT, h, batch_first=True)
        self.head = nn.Linear(h, z_dim)  # z_dim=4 (g,f-1) [audited]; 7 adds bias (Paper C C-1b)
        self.register_buffer("mean", torch.zeros(FEAT))
        self.register_buffer("std", torch.ones(FEAT))

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-4))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B,T,FEAT) -> (B,T,Z)
        y, _ = self.gru((seq - self.mean) / self.std)
        return self.head(y)  # type: ignore[no-any-return]

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.h)

    def step(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, hidden = self.gru(((x - self.mean) / self.std).view(1, 1, -1), hidden)
        return self.head(y).view(-1), hidden


def _feat(obs: np.ndarray, prev_a: np.ndarray, prev_obs: np.ndarray) -> np.ndarray:
    return np.concatenate([obs, prev_a, obs - prev_obs]).astype(np.float32)  # type: ignore[no-any-return]


def _make_env(f: float, seed: int, cfg: MetaConfig) -> Any:
    # inertia from f; the fault gain g is applied in the loop (not by the env)
    return _task_env(TaskSpec(f, (1, 1, 1)), cfg, seed=seed)


# --------------------------------------------------------------------------- #
# phase 2/3 — train the student (teacher warm-start + on-policy DAgger)
# --------------------------------------------------------------------------- #
def _rollout(
    cfg: MetaConfig, task: Task, seed: int, noise: float, student: RMAStudent | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One episode → (features[T,FEAT], z[Z], mask[T]). Actions from the teacher
    (true z) when ``student`` is None, else the deployed policy (teacher fed the
    GRU's ẑ) — the on-policy DAgger distribution, relabelled with the TRUE z."""
    f, g = task
    rng = np.random.default_rng(seed)
    T = cfg.ep_len
    z = task_z(f, g)
    env = _make_env(f, seed, cfg)
    obs, _ = env.reset(seed=seed)
    prev_a = np.zeros(ACT, dtype=np.float32)
    prev_obs = obs.copy()
    hidden = student.init_hidden() if student is not None else None
    S = []
    for _ in range(T):
        feat = _feat(obs, prev_a, prev_obs)
        S.append(feat)
        if student is None or hidden is None:
            a = analytic_teacher(obs, z, cfg)
        else:
            with torch.no_grad():
                zhat, hidden = student.step(torch.tensor(feat), hidden)
            a = analytic_teacher(obs, zhat.numpy(), cfg)
        a = np.clip(a + rng.normal(0, noise, ACT), -1, 1).astype(np.float32)
        prev_a, prev_obs = a, obs.copy()
        obs, _, _, trunc, _ = env.step((a * g).astype(np.float32))  # apply actuator fault
        if trunc:
            break
    env.close()
    arr = np.asarray(S, dtype=np.float32)
    m = np.zeros(T, dtype=np.float32)
    m[: arr.shape[0]] = 1.0
    if arr.shape[0] < T:
        arr = np.vstack([arr, np.zeros((T - arr.shape[0], FEAT), dtype=np.float32)])
    return arr, z, m


def collect(
    cfg: MetaConfig, n_tasks: int, seed0: int, noise: float, student: RMAStudent | None, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed0)
    seqs, zs, masks = [], [], []
    for i in range(n_tasks):
        arr, z, m = _rollout(cfg, _sample_for_mode(rng, mode), seed0 + i, noise, student)
        seqs.append(arr)
        zs.append(z)
        masks.append(m)
    T = lambda L: torch.tensor(np.asarray(L), dtype=torch.float32)  # noqa: E731
    return T(seqs), T(zs), T(masks)


def _fit(
    model: RMAStudent,
    seqs: torch.Tensor,
    zs: torch.Tensor,
    masks: torch.Tensor,
    epochs: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B, bs = seqs.shape[0], 64
    for ep in range(epochs):
        perm = torch.randperm(B, device=seqs.device)
        tot = 0.0
        for j in range(0, B, bs):
            idx = perm[j : j + bs]
            loss = ((model(seqs[idx]) - zs[idx].unsqueeze(1)) ** 2 * masks[idx].unsqueeze(-1)).sum()
            loss = loss / masks[idx].sum() / Z
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == epochs - 1:
            logger.info("    epoch %d  z-mse %.5f", ep, tot / max(1, math.ceil(B / bs)))


def train_student(
    n_tasks: int,
    epochs: int,
    lr: float,
    noise: float,
    dagger_iters: int,
    mode: str,
    seed: int,
    dev: str,
    ckpt: str = STUDENT_CKPT,
) -> RMAStudent:
    cfg = env_cfg()
    logger.info("student warm-start: teacher rollouts over %d %s-fault tasks", n_tasks, mode)
    seqs, zs, masks = collect(cfg, n_tasks, seed, noise, None, mode)
    model = RMAStudent()
    model.to(dev)
    flat = seqs.reshape(-1, FEAT)[masks.reshape(-1) > 0]
    model.set_norm(flat.mean(0).to(dev), flat.std(0).to(dev))
    seqs, zs, masks = seqs.to(dev), zs.to(dev), masks.to(dev)
    _fit(model, seqs, zs, masks, epochs, lr)

    for it in range(dagger_iters):
        model.cpu().eval()
        logger.info("DAgger iter %d: on-policy rollouts (deployed policy)…", it + 1)
        ns, nz, nm = collect(
            cfg, max(1, n_tasks // 2), seed + 1000 * (it + 1), noise * 0.5, model, mode
        )
        seqs = torch.cat([seqs, ns.to(dev)])
        zs = torch.cat([zs, nz.to(dev)])
        masks = torch.cat([masks, nm.to(dev)])
        model.to(dev).train()
        _fit(model, seqs, zs, masks, max(10, epochs // 2), lr)
        logger.info("  aggregated dataset: %d sequences", seqs.shape[0])

    model.cpu().eval()
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    # Provenance sidecar: which sampler/split produced this checkpoint. The
    # v1-vs-v2 distinction (wide generator vs declared TRAIN split) is a paper
    # claim, so it must be machine-readable next to the weights.
    prov = {
        "mode": mode,
        "sampler": (
            "sample_task_taxonomy: program.fault_taxonomy Split.TRAIN, 50/50 SIGN/GAIN "
            "(f in [0.8,2.0], |g| in [0.5,1.5], <=1 reversed axis, b=0)"
            if mode == "taxonomy"
            else f"sample_task(mode={mode}): f in U(0.7,2.2), |g| in U({G_MIN},{G_MAX}), "
            "independent per-axis signs"
        ),
        "n_tasks": n_tasks,
        "epochs": epochs,
        "lr": lr,
        "noise": noise,
        "dagger_iters": dagger_iters,
        "seed": seed,
    }
    Path(ckpt).with_suffix(".provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )
    logger.info("student saved -> %s (+ provenance sidecar)", ckpt)
    return model


# --------------------------------------------------------------------------- #
# deploy policies + eval
# --------------------------------------------------------------------------- #
def _pointing_deg(sigma: np.ndarray) -> float:
    return float(np.rad2deg(4.0 * math.atan(np.linalg.norm(sigma))))


def pd_policy(cfg: MetaConfig) -> Callable[[np.ndarray], np.ndarray]:
    def policy(obs: np.ndarray) -> np.ndarray:
        u = (-cfg.kp0 * obs[:3] - cfg.kd0 * obs[3:]) / cfg.max_torque_nm  # fault-UNAWARE
        return np.clip(u, -1.0, 1.0).astype(np.float32)

    return policy


def teacher_true_policy(
    f: float, g: np.ndarray, cfg: MetaConfig
) -> Callable[[np.ndarray], np.ndarray]:
    z = task_z(f, g)
    return lambda obs: analytic_teacher(obs, z, cfg)


def teacher_bias_true_policy(
    f: float, g: np.ndarray, b: np.ndarray, cfg: MetaConfig
) -> Callable[[np.ndarray], np.ndarray]:
    """Privileged teacher with the true constant-bias feedforward (Paper C, C-1 oracle).

    The capability upper bound for the GAIN_BIAS class: knows the true effectiveness ``g``
    AND the true additive bias ``b``, cancelling the latter exactly (``a = (u−b)/g_eff``).
    If this oracle cannot settle GAIN_BIAS, the augmented law is wrong (PLAN_C C-1)."""
    z = task_z(f, g)
    b = np.asarray(b, dtype=np.float32)
    return lambda obs: analytic_teacher(obs, z, cfg, b_ff=b)


def rma_policy(
    student: RMAStudent,
    cfg: MetaConfig,
    *,
    kd_scale: float = 1.0,
    g_margin: float = 0.0,
    latch_below_deg: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fresh stateful student policy (resets GRU hidden state per episode).

    Defaults are the audited no-op. ``latch_below_deg`` (0 = off) FREEZES the inferred
    fault ẑ while pointing stays below that angle, resuming live adaptation if it drifts
    back out: the gain is identifiable only while the loop is excited, so near the target
    the GRU's estimate wanders and that jitter drives the steady-state limit cycle
    diagnosed on held-out faults — latching holds the estimate that achieved approach,
    while re-adapting outside the basin avoids getting stuck on a bad freeze.
    ``kd_scale``/``g_margin`` see ``analytic_teacher`` (empirically unhelpful here; kept
    for completeness)."""
    hidden = student.init_hidden()
    prev_a = np.zeros(ACT, dtype=np.float32)
    last: list[np.ndarray] = []
    frozen_z: list[np.ndarray] = []  # holds the latched ẑ once engaged

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal hidden, prev_a
        po = last[0] if last else obs
        with torch.no_grad():
            zhat_t, hidden = student.step(torch.tensor(_feat(obs, prev_a, po)), hidden)
        zhat = zhat_t.numpy()
        if latch_below_deg > 0.0 and _pointing_deg(obs[:3]) < latch_below_deg:
            if not frozen_z:
                frozen_z.append(zhat)  # entered the basin: latch the converged estimate
            z = frozen_z[0]
        else:
            frozen_z.clear()  # outside the basin: resume live adaptation (no stuck latch)
            z = zhat
        a = analytic_teacher(obs, z, cfg, kd_scale=kd_scale, g_margin=g_margin)
        prev_a = a
        last.clear()
        last.append(obs.copy())
        return a

    return policy


def _wilson(succ: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    z = 1.96
    p = succ / n
    denom = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    center = (p + z * z / (2 * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def gate_eval(make_policy: Any, tasks: list[Task], cfg: MetaConfig, seed0: int) -> dict:
    """Run a controller (fresh per episode via ``make_policy(f,g)``) on each task;
    gate = best pointing ≤ 0.2°. Applies the per-axis actuator gain in the loop."""
    succ = div = n = 0
    bests = []
    for ti, (f, g) in enumerate(tasks):
        seed = seed0 + ti
        env = _make_env(f, seed, cfg)
        obs, _ = env.reset(seed=seed)
        policy = make_policy(f, g)
        best = _pointing_deg(obs[:3])
        for _ in range(cfg.ep_len):
            best = min(best, _pointing_deg(obs[:3]))
            a = policy(obs)
            obs, _, _, trunc, _ = env.step((np.asarray(a) * g).astype(np.float32))
            if trunc:
                break
        env.close()
        bests.append(best)
        succ += best <= 0.2
        div += best > 5.0
        n += 1
    return {
        "gate_success_rate": round(succ / n, 4),
        "wilson_95ci": _wilson(succ, n),
        "diverged_rate": round(div / n, 4),
        "median_best_pointing_deg": round(float(np.median(bests)), 3),
        "n": n,
    }


def _battery_sign_tasks() -> list[Task]:
    from program.battery import BATTERY

    return [
        (t.inertia_factor, np.asarray(t.fault, dtype=np.float32)) for t in BATTERY.faulted_tasks
    ]


def _continuous_tasks(n: int, seed: int) -> list[Task]:
    rng = np.random.default_rng(seed)
    return [sample_task(rng, "continuous") for _ in range(n)]


def evaluate(student: RMAStudent, n_seeds: int) -> dict:
    from program.battery import BATTERY

    cfg = env_cfg()
    bc_student = student  # alias for closure clarity

    def controllers(student_obj: RMAStudent) -> dict:
        from program.testbed import TestbedConfig, load_controller

        tc = TestbedConfig()
        return {
            "pd_fault_unaware": lambda f, g: pd_policy(cfg),
            "nominal_bc": lambda f, g: load_controller("learned", tc),
            "rma_teacher_privileged": lambda f, g: teacher_true_policy(f, g, cfg),
            "rma_student_inferred": lambda f, g: rma_policy(student_obj, cfg),
        }

    out: dict = {}
    # sign-fault battery (baseline-comparable: each task repeated over n_seeds)
    sign_tasks = [t for t in _battery_sign_tasks() for _ in range(n_seeds)]
    for name, mk in controllers(bc_student).items():
        out[name] = gate_eval(mk, sign_tasks, cfg, seed0=BATTERY.base_seed)
    # held-out continuous-fault set (the harder generalisation)
    cont_tasks = _continuous_tasks(len(sign_tasks), seed=99999)
    for name, mk in controllers(bc_student).items():
        out["continuous__" + name] = gate_eval(mk, cont_tasks, cfg, seed0=BATTERY.base_seed + 5000)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["student", "eval", "all"], default="all")
    ap.add_argument("--mode", choices=["sign", "continuous", "taxonomy"], default="continuous")
    ap.add_argument("--student-tasks", type=int, default=500)
    ap.add_argument("--student-epochs", type=int, default=60)
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint path; defaults to the v2 path for --mode taxonomy, v1 otherwise",
    )
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = a.ckpt or (STUDENT_V2_CKPT if a.mode == "taxonomy" else STUDENT_CKPT)
    if a.mode == "taxonomy" and ckpt == STUDENT_CKPT:
        # The audited v1 checkpoint is a committed paper artifact (the wide-
        # generator student); a taxonomy retrain must never clobber it.
        raise SystemExit("refusing to overwrite the v1 checkpoint with a taxonomy retrain")

    student = RMAStudent()
    if a.phase in ("student", "all"):
        student = train_student(
            a.student_tasks,
            a.student_epochs,
            1e-3,
            a.noise,
            a.dagger_iters,
            a.mode,
            a.seed,
            dev,
            ckpt=ckpt,
        )
    else:
        student.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        student.eval()

    if a.phase in ("eval", "all"):
        logger.info("evaluating on the sign-fault battery + continuous-fault set…")
        res = evaluate(student, a.n_seeds)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2))
        for k, v in res.items():
            logger.info(
                "  %-34s gate %.1f%% CI%s  div %.0f%%  med-best %.2f°",
                k,
                100 * v["gate_success_rate"],
                v["wilson_95ci"],
                100 * v["diverged_rate"],
                v["median_best_pointing_deg"],
            )
        logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
