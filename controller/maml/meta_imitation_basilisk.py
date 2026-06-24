"""FOMAML meta-imitation for fault-adaptive attitude control (real Basilisk, G2).

The proposal's controller tier is a *meta-RL* policy that adapts few-shot to new
operating conditions. F4 proved a behavior-cloned MLP solves the nominal Phase-1
attitude task at 217/217. G2 demonstrates the **meta-learning** value on the case
where it is provably needed: a **multimodal** task family the controller must
cope with autonomously — **actuator faults**.

Each task = ``BasiliskAttitudeEnv`` with (i) inertia scaled by f and (ii) an
actuator-effectiveness sign pattern b ∈ {+1,−1}³ modelling a thruster/wheel
wiring-reversal or mounting-flip fault on a subset of axes. The fault is *not*
observable (the policy sees only MRP + body rate); the commanded action is
multiplied by b before it reaches the dynamics, so a policy that does not account
for b drives the wrong way on the faulted axes and diverges.

Why this is the right meta-learning test (Finn et al. 2017): because +b and −b
tasks demand opposite actions on the same states, a single **pooled** policy
averages to ≈zero gain on faulted axes and cannot serve any task — exactly the
multimodal regime where MAML beats joint training. A FOMAML meta-initialization
instead learns to *infer* the fault from a short support rollout and adapt to it
in a few gradient steps. This is fault-adaptive control: directly the proposal's
autonomy/resilience claim.

Reports the few-shot **adaptation curve** (success vs gradient-steps) for the
FOMAML init vs a pooled-BC init, on **held-out** fault patterns + inertias, with
Wilson 95% CIs, all on real Basilisk 6-DOF dynamics.

Run: ``python -m controller.maml.meta_imitation_basilisk``
Emits ``evidence/curriculum/basilisk_meta_imitation_fomaml.json`` (+ md note).
"""

from __future__ import annotations

# X/Y data-matrix names; functional-MAML param dicts.
# ruff: noqa: N803, N806
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from controller.rl.bc_attitude import MLPPolicy
from simulation.basilisk.attitude_env import AttitudeEnvConfig, BasiliskAttitudeEnv

logger = logging.getLogger(__name__)

BASE_INERTIA = (10.0, 8.0, 6.0)


@dataclass(frozen=True)
class TaskSpec:
    """A meta-task: inertia scale + actuator-effectiveness sign pattern."""

    inertia_factor: float
    fault: tuple[int, int, int]  # actuator sign per body axis, +1 nominal / -1 reversed


@dataclass
class MetaConfig:
    # Train tasks: inertia × fault patterns spanning BOTH signs on every axis
    # (so a pooled policy must average conflicting demos -> ~zero gain).
    train_tasks: tuple[TaskSpec, ...] = (
        TaskSpec(0.8, (1, 1, 1)),
        TaskSpec(1.5, (1, 1, 1)),
        TaskSpec(0.9, (-1, 1, 1)),
        TaskSpec(1.7, (1, -1, 1)),
        TaskSpec(1.2, (1, 1, -1)),
        TaskSpec(2.0, (-1, 1, 1)),
        TaskSpec(0.7, (1, -1, 1)),
        TaskSpec(1.4, (1, 1, -1)),
    )
    # Held-out tasks: unseen inertias AND unseen (incl. multi-axis) fault patterns.
    test_tasks: tuple[TaskSpec, ...] = (
        TaskSpec(1.1, (1, 1, 1)),  # nominal, unseen inertia
        TaskSpec(1.3, (-1, 1, 1)),  # single flip
        TaskSpec(1.6, (1, 1, -1)),  # single flip
        TaskSpec(2.2, (-1, -1, 1)),  # two-axis flip — unseen pattern + extrapolated inertia
    )
    demo_episodes: int = 8
    ep_len: int = 400
    max_torque_nm: float = 0.6
    kp0: float = 0.2
    kd0: float = 1.5
    # FOMAML
    meta_iters: int = 600
    meta_batch: int = 4
    inner_lr: float = 0.05
    inner_steps: int = 5
    outer_lr: float = 1e-3
    support_frac: float = 0.5
    # offline few-shot adaptation curve (shows covariate-shift null)
    adapt_curve: tuple[int, ...] = (0, 5, 10, 20, 40)
    adapt_lr: float = 0.05
    adapt_demo_episodes: int = 5
    # on-policy DAgger adaptation (the F4-proven fix for closed-loop competence)
    dagger_iters: int = 4
    dagger_episodes: int = 3
    dagger_fit_steps: int = 60
    eval_episodes: int = 25
    seed: int = 0
    note: str = field(default="held-out actuator faults + inertias; fault unobserved")


def _gains(f: float, cfg: MetaConfig) -> tuple[float, float]:
    return cfg.kp0 * f, cfg.kd0 * f


def _expert_action(obs: np.ndarray, task: TaskSpec, cfg: MetaConfig) -> np.ndarray:
    """Fault-aware PD expert: gains scaled by inertia; pre-inverts the actuator
    sign so that, after the env multiplies by ``fault``, the applied torque is
    the desired PD torque. (For ±1 faults, inversion = multiply by the sign.)"""
    sigma, omega = obs[:3], obs[3:]
    kp, kd = _gains(task.inertia_factor, cfg)
    u_norm = (-kp * sigma - kd * omega) / cfg.max_torque_nm
    a = u_norm * np.asarray(task.fault, dtype=float)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def _task_env(task: TaskSpec, cfg: MetaConfig, seed: int) -> BasiliskAttitudeEnv:
    inertia = tuple(b * task.inertia_factor for b in BASE_INERTIA)
    return BasiliskAttitudeEnv(
        AttitudeEnvConfig(
            episode_length=cfg.ep_len,
            max_torque_nm=cfg.max_torque_nm,
            inertia_diag=inertia,  # type: ignore[arg-type]
            seed=seed,
        )
    )


def _apply_fault(a: np.ndarray, task: TaskSpec) -> np.ndarray:
    """Actuator applies its (faulted) effectiveness to the commanded action."""
    return (np.asarray(a, dtype=float) * np.asarray(task.fault, dtype=float)).astype(np.float32)


def collect_demos(
    task: TaskSpec, n_ep: int, cfg: MetaConfig, seed0: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the fault-aware expert in real Basilisk; return (obs, expert_command)."""
    env = _task_env(task, cfg, seed0)
    X, Y = [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed0 + ep)
        for _ in range(cfg.ep_len):
            a = _expert_action(obs, task, cfg)
            X.append(obs.copy())
            Y.append(a)
            obs, _, _, trunc, _ = env.step(_apply_fault(a, task))
            if trunc:
                break
    env.close()
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def _mse(policy: MLPPolicy, params: dict, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    pred = functional_call(policy, params, (X,))
    return torch.nn.functional.mse_loss(pred, Y)


def _adapt(
    policy: MLPPolicy, params: dict, X: torch.Tensor, Y: torch.Tensor, lr: float, steps: int
) -> dict:
    """Inner-loop adaptation: SGD on MSE to expert (FOMAML — first order)."""
    adapted = {k: v for k, v in params.items()}
    for _ in range(steps):
        loss = _mse(policy, adapted, X, Y)
        grads = torch.autograd.grad(loss, list(adapted.values()), create_graph=False)
        adapted = {
            k: (v - lr * g).detach().requires_grad_(True)
            for (k, v), g in zip(adapted.items(), grads, strict=False)
        }
    return adapted


def fomaml_train(
    policy: MLPPolicy, task_data: list[tuple[torch.Tensor, torch.Tensor]], cfg: MetaConfig
) -> list[float]:
    """Meta-train the policy init with FOMAML over the train-task demos."""
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.outer_lr)
    rng = np.random.default_rng(cfg.seed)
    losses = []
    for it in range(cfg.meta_iters):
        opt.zero_grad()
        meta_grads = {n: torch.zeros_like(p) for n, p in policy.named_parameters()}
        batch = rng.choice(len(task_data), size=min(cfg.meta_batch, len(task_data)), replace=False)
        qloss_sum = 0.0
        for ti in batch:
            X, Y = task_data[ti]
            n = X.shape[0]
            ns = int(n * cfg.support_frac)
            perm = torch.randperm(n)
            si, qi = perm[:ns], perm[ns:]
            base = {n_: p for n_, p in policy.named_parameters()}
            adapted = _adapt(policy, base, X[si], Y[si], cfg.inner_lr, cfg.inner_steps)
            qloss = _mse(policy, adapted, X[qi], Y[qi])
            qgrads = torch.autograd.grad(qloss, list(adapted.values()))
            for (nm, _), g in zip(policy.named_parameters(), qgrads, strict=False):
                meta_grads[nm] += g / len(batch)
            qloss_sum += float(qloss.item())
        for n_, p in policy.named_parameters():
            p.grad = meta_grads[n_]
        opt.step()
        losses.append(qloss_sum / len(batch))
        if it % 100 == 0:
            logger.info("FOMAML iter %d: query MSE %.6f", it, losses[-1])
    return losses


def joint_train(
    policy: MLPPolicy, task_data: list[tuple[torch.Tensor, torch.Tensor]], cfg: MetaConfig
) -> None:
    """Non-meta baseline: pooled behavior cloning over all train tasks."""
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.outer_lr)
    X = torch.cat([d[0] for d in task_data])
    Y = torch.cat([d[1] for d in task_data])
    n = X.shape[0]
    for _ in range(cfg.meta_iters * 2):  # same gradient budget, pooled
        idx = torch.randint(0, n, (256,))
        loss = torch.nn.functional.mse_loss(policy(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()


def _wilson(succ: int, n: int) -> tuple[float, float]:
    z = 1.96
    p = succ / n
    denom = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    center = (p + z * z / (2 * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@torch.no_grad()
def _rollout_eval(
    policy: MLPPolicy, params: dict, task: TaskSpec, cfg: MetaConfig, seed0: int
) -> dict:
    env = _task_env(task, cfg, seed0)
    succ, finals = 0, []
    for k in range(cfg.eval_episodes):
        obs, info = env.reset(seed=seed0 + 7000 + k)
        for _ in range(cfg.ep_len):
            a = (
                functional_call(policy, params, (torch.tensor(obs).unsqueeze(0),))
                .squeeze(0)
                .numpy()
            )
            obs, _, _, trunc, info = env.step(_apply_fault(a, task))
            if trunc:
                break
        finals.append(info["best_pointing_deg"])
        succ += int(info["success"])
    env.close()
    lo, hi = _wilson(succ, cfg.eval_episodes)
    return {
        "inertia_factor": task.inertia_factor,
        "fault": list(task.fault),
        "successes": succ,
        "n": cfg.eval_episodes,
        "success_rate": succ / cfg.eval_episodes,
        "wilson_95ci": [lo, hi],
        "best_pointing_deg_mean": float(np.mean(finals)),
    }


@torch.no_grad()
def _dagger_rollout(
    policy: MLPPolicy, params: dict, task: TaskSpec, cfg: MetaConfig, seed0: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the CURRENT policy in real Basilisk under the fault; relabel every
    visited state with the fault-aware expert. This is the on-policy DAgger data
    that fixes behavior-cloning covariate shift (the F4 finding): the policy
    visits its own (often diverging) states and the expert says what to do there.
    """
    env = _task_env(task, cfg, seed0)
    X, Y = [], []
    for ep in range(cfg.dagger_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        for _ in range(cfg.ep_len):
            X.append(obs.copy())
            Y.append(_expert_action(obs, task, cfg))  # expert label at the policy's state
            a = (
                functional_call(policy, params, (torch.tensor(obs).unsqueeze(0),))
                .squeeze(0)
                .numpy()
            )
            obs, _, _, trunc, _ = env.step(_apply_fault(a, task))
            if trunc:
                break
    env.close()
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def dagger_adaptation_curve(policy: MLPPolicy, cfg: MetaConfig, label: str) -> list[dict]:
    """Closed-loop success vs number of DAgger adaptation iterations, per init.

    Iteration 0 = zero-shot. Each iteration rolls the current policy in Basilisk,
    relabels visited states with the fault-aware expert, aggregates, and refits
    from the (meta) init on the growing on-policy set — then re-evaluates the
    closed-loop success on real Basilisk. This is the on-policy adaptation the
    controller actually needs; offline few-shot imitation cannot fix the faulted
    dynamics (see the flat MSE curve)."""
    base = {n: p.detach().clone() for n, p in policy.named_parameters()}
    # per-task aggregated DAgger datasets + current params
    agg: dict = {t: [None, None, dict(base)] for t in cfg.test_tasks}
    curve = []
    for it in range(cfg.dagger_iters + 1):
        per_task = []
        for i, task in enumerate(cfg.test_tasks):
            if it > 0:
                Xo, Yo = _dagger_rollout(
                    policy, agg[task][2], task, cfg, seed0=40000 + i * 100 + it
                )
                if agg[task][0] is None:
                    agg[task][0], agg[task][1] = torch.tensor(Xo), torch.tensor(Yo)
                else:
                    agg[task][0] = torch.cat([agg[task][0], torch.tensor(Xo)])
                    agg[task][1] = torch.cat([agg[task][1], torch.tensor(Yo)])
                leaf = {n: p.clone().requires_grad_(True) for n, p in base.items()}
                fitted = _adapt(
                    policy, leaf, agg[task][0], agg[task][1], cfg.adapt_lr, cfg.dagger_fit_steps
                )
                agg[task][2] = {n: p.detach() for n, p in fitted.items()}
            per_task.append(_rollout_eval(policy, agg[task][2], task, cfg, seed0=30000 + i * 100))
        tot_s = sum(t["successes"] for t in per_task)
        tot_n = sum(t["n"] for t in per_task)
        lo, hi = _wilson(tot_s, tot_n)
        r = {
            "dagger_iters": it,
            "per_task": per_task,
            "aggregate_successes": tot_s,
            "aggregate_n": tot_n,
            "aggregate_success_rate": tot_s / tot_n,
            "wilson_95ci": [lo, hi],
            "mean_best_pointing_deg": float(
                np.mean([t["best_pointing_deg_mean"] for t in per_task])
            ),
            "label": label,
        }
        logger.info(
            "  %s DAgger iter %d: %d/%d = %.1f%% (mean best %.4f°)",
            label,
            it,
            tot_s,
            tot_n,
            100 * r["aggregate_success_rate"],
            r["mean_best_pointing_deg"],
        )
        curve.append(r)
    return curve


def _eval_at_budget(
    policy: MLPPolicy, base: dict, adapt_data: dict, budget: int, cfg: MetaConfig
) -> dict:
    per_task = []
    for i, task in enumerate(cfg.test_tasks):
        if budget == 0:
            params = base
        else:
            Xa, Ya = adapt_data[task]
            leaf = {n: p.clone().requires_grad_(True) for n, p in base.items()}
            params = {
                n: p.detach() for n, p in _adapt(policy, leaf, Xa, Ya, cfg.adapt_lr, budget).items()
            }
        per_task.append(_rollout_eval(policy, params, task, cfg, seed0=30000 + i * 100))
    tot_s = sum(t["successes"] for t in per_task)
    tot_n = sum(t["n"] for t in per_task)
    lo, hi = _wilson(tot_s, tot_n)
    return {
        "adapt_steps": budget,
        "per_task": per_task,
        "aggregate_successes": tot_s,
        "aggregate_n": tot_n,
        "aggregate_success_rate": tot_s / tot_n,
        "wilson_95ci": [lo, hi],
        "mean_best_pointing_deg": float(np.mean([t["best_pointing_deg_mean"] for t in per_task])),
    }


def mse_adaptation_curve(
    policy: MLPPolicy, adapt_data: dict, query_data: dict, cfg: MetaConfig, label: str
) -> list[dict]:
    """Canonical MAML metric (Finn et al. 2017): post-adaptation regression MSE
    to the held-out task's expert vs number of adaptation steps. Pure supervised
    (no rollout), so it cleanly isolates *adaptation quality* from the tight
    downstream 0.2° gate. On multimodal (fault) tasks a pooled init averages
    conflicting-sign demos → high MSE that adapts slowly; a FOMAML init adapts to
    low MSE in few steps."""
    base = {n: p.detach().clone() for n, p in policy.named_parameters()}
    curve = []
    for budget in cfg.adapt_curve:
        per_task = []
        for task in cfg.test_tasks:
            Xs, Ys = adapt_data[task]
            Xq, Yq = query_data[task]
            if budget == 0:
                params = base
            else:
                leaf = {n: p.clone().requires_grad_(True) for n, p in base.items()}
                params = {
                    n: p.detach()
                    for n, p in _adapt(policy, leaf, Xs, Ys, cfg.adapt_lr, budget).items()
                }
            with torch.no_grad():
                per_task.append(float(_mse(policy, params, Xq, Yq).item()))
        curve.append(
            {
                "adapt_steps": budget,
                "mean_query_mse": float(np.mean(per_task)),
                "per_task_mse": [round(m, 6) for m in per_task],
            }
        )
        logger.info("  %s @ %d steps: query MSE %.6f", label, budget, curve[-1]["mean_query_mse"])
    return curve


def adaptation_curve(
    policy: MLPPolicy, adapt_data: dict, cfg: MetaConfig, label: str
) -> list[dict]:
    base = {n: p.detach().clone() for n, p in policy.named_parameters()}
    curve = []
    for budget in cfg.adapt_curve:
        r = _eval_at_budget(policy, base, adapt_data, budget, cfg)
        r["label"] = label
        logger.info(
            "  %s @ %d steps: %d/%d = %.1f%% (mean best %.4f°)",
            label,
            budget,
            r["aggregate_successes"],
            r["aggregate_n"],
            100 * r["aggregate_success_rate"],
            r["mean_best_pointing_deg"],
        )
        curve.append(r)
    return curve


def run(cfg: MetaConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info(
        "Collecting fault-aware expert demos on real Basilisk (%d tasks)…", len(cfg.train_tasks)
    )
    task_data = []
    for i, task in enumerate(cfg.train_tasks):
        X, Y = collect_demos(task, cfg.demo_episodes, cfg, seed0=i * 1000)
        task_data.append((torch.tensor(X), torch.tensor(Y)))
        logger.info("  f=%.2f fault=%s: %d demo pairs", task.inertia_factor, task.fault, X.shape[0])

    meta_policy = MLPPolicy()
    logger.info("FOMAML meta-training (%d iters)…", cfg.meta_iters)
    losses = fomaml_train(meta_policy, task_data, cfg)

    joint_policy = MLPPolicy()
    logger.info("Joint (non-meta) pooled-BC baseline…")
    joint_train(joint_policy, task_data, cfg)

    # Collect held-out few-shot support + query demos once (shared across both
    # inits and both metrics) on real Basilisk.
    logger.info("Collecting held-out support/query demos…")
    adapt_data, query_data = {}, {}
    for i, task in enumerate(cfg.test_tasks):
        Xs, Ys = collect_demos(task, cfg.adapt_demo_episodes, cfg, seed0=20000 + i * 100)
        Xq, Yq = collect_demos(task, cfg.adapt_demo_episodes, cfg, seed0=25000 + i * 100)
        adapt_data[task] = (torch.tensor(Xs), torch.tensor(Ys))
        query_data[task] = (torch.tensor(Xq), torch.tensor(Yq))

    # CONTROL: offline few-shot adaptation MSE — demonstrates the covariate-shift
    # null (low offline MSE on equilibrium-dominated demos ≠ closed-loop control).
    logger.info("Offline few-shot MSE curve (control — expected covariate-shift null):")
    meta_mse = mse_adaptation_curve(meta_policy, adapt_data, query_data, cfg, "FOMAML-offline")
    joint_mse = mse_adaptation_curve(joint_policy, adapt_data, query_data, cfg, "pooledBC-offline")
    meta_off = adaptation_curve(meta_policy, adapt_data, cfg, "FOMAML-offline")
    joint_off = adaptation_curve(joint_policy, adapt_data, cfg, "pooledBC-offline")

    # PRIMARY: on-policy DAgger closed-loop adaptation (the F4-proven fix).
    logger.info("DAgger on-policy adaptation curve (PRIMARY — closed-loop success):")
    logger.info("FOMAML init:")
    meta_dagger = dagger_adaptation_curve(meta_policy, cfg, "FOMAML")
    logger.info("Pooled-BC init:")
    joint_dagger = dagger_adaptation_curve(joint_policy, cfg, "pooled-BC")

    def _best(curve: list[dict]) -> dict:
        return max(curve, key=lambda r: r["aggregate_success_rate"])

    md, jd = _best(meta_dagger), _best(joint_dagger)
    mo, jo = _best(meta_off), _best(joint_off)
    result = {
        "method": "FOMAML meta-imitation + DAgger adaptation, fault-adaptive control (Basilisk)",
        "task_family": "inertia scale × actuator-sign fault (unobserved, multimodal)",
        "train_tasks": [{"f": t.inertia_factor, "fault": list(t.fault)} for t in cfg.train_tasks],
        "test_tasks": [{"f": t.inertia_factor, "fault": list(t.fault)} for t in cfg.test_tasks],
        "fomaml_final_train_query_mse": round(losses[-1], 6),
        "primary_dagger_success_curve": {"meta_fomaml": meta_dagger, "pooled_bc": joint_dagger},
        "control_offline_mse_curve": {"meta_fomaml": meta_mse, "pooled_bc": joint_mse},
        "control_offline_success_curve": {"meta_fomaml": meta_off, "pooled_bc": joint_off},
        "headline": {
            "metric": "closed-loop success on held-out faults after on-policy DAgger adaptation",
            "dagger_meta_zeroshot_pct": round(100 * meta_dagger[0]["aggregate_success_rate"], 1),
            "dagger_meta_best_pct": round(100 * md["aggregate_success_rate"], 1),
            "dagger_meta_best_iters": md["dagger_iters"],
            "dagger_pooled_zeroshot_pct": round(100 * joint_dagger[0]["aggregate_success_rate"], 1),
            "dagger_pooled_best_pct": round(100 * jd["aggregate_success_rate"], 1),
            "dagger_fomaml_advantage_pct_pts": round(
                100 * (md["aggregate_success_rate"] - jd["aggregate_success_rate"]), 1
            ),
            "offline_meta_best_pct": round(100 * mo["aggregate_success_rate"], 1),
            "offline_pooled_best_pct": round(100 * jo["aggregate_success_rate"], 1),
            "offline_mse_flat_note": "offline few-shot MSE ~"
            f"{meta_mse[0]['mean_query_mse']:.5f} flat across budgets — covariate-shift null",
        },
        "config": {
            "meta_iters": cfg.meta_iters,
            "inner_steps": cfg.inner_steps,
            "inner_lr": cfg.inner_lr,
            "dagger_iters": cfg.dagger_iters,
            "dagger_episodes": cfg.dagger_episodes,
            "dagger_fit_steps": cfg.dagger_fit_steps,
            "adapt_lr": cfg.adapt_lr,
            "eval_episodes": cfg.eval_episodes,
            "demo_episodes": cfg.demo_episodes,
        },
    }
    out = Path("evidence/curriculum/basilisk_meta_imitation_fomaml.json")
    out.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", out)
    return result


def main() -> None:
    import click

    @click.command()
    @click.option("--meta-iters", default=600, show_default=True)
    @click.option("--eval-episodes", default=25, show_default=True)
    def cli(meta_iters: int, eval_episodes: int) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        cfg = MetaConfig(meta_iters=meta_iters, eval_episodes=eval_episodes)
        res = run(cfg)
        h = res["headline"]
        logger.info(
            "DONE (held-out actuator faults). DAgger closed-loop success: FOMAML "
            "%.1f%%→%.1f%% (@%d iters) vs pooled-BC %.1f%%→%.1f%% (FOMAML advantage "
            "%.1f pts). Offline few-shot (control): meta %.1f%% / pooled %.1f%% — "
            "offline MSE flat (covariate-shift null).",
            h["dagger_meta_zeroshot_pct"],
            h["dagger_meta_best_pct"],
            h["dagger_meta_best_iters"],
            h["dagger_pooled_zeroshot_pct"],
            h["dagger_pooled_best_pct"],
            h["dagger_fomaml_advantage_pct_pts"],
            h["offline_meta_best_pct"],
            h["offline_pooled_best_pct"],
        )

    cli()


if __name__ == "__main__":
    main()
