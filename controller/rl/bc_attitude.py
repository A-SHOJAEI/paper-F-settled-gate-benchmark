"""Behavior-cloned (learned NN) attitude controller on the Basilisk env.

Validates the controller-tier *learning* claim: an MLP policy is trained
(supervised) to clone the MRP-PD imitation target on real Basilisk
6-DOF attitude dynamics, then evaluated via the same n=217 phase-aware
Monte Carlo as the PD baseline. If the learned policy matches the PD
217/217, it demonstrates a learned controller (not just a hand-tuned
PD) achieving the Phase-1 gate — the behavior-cloning warm-start that
``controller/maml/imitation.py`` feeds into FOMAML.

From-scratch RL stalls on this task (documented in FINDINGS.md / the
training-curve evidence); BC from the PD demonstrator is the proven
path, matching the proposal's imitation-then-meta-RL design.
"""

from __future__ import annotations

# X/Y/Xt/Yt are standard ML data-matrix names.
# ruff: noqa: N803, N806
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from simulation.basilisk.attitude_env import AttitudeEnvConfig, BasiliskAttitudeEnv

logger = logging.getLogger(__name__)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 6, act_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # bounded action in [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BCConfig:
    n_demo_episodes: int = 60
    demo_episode_len: int = 400  # match eval horizon
    kp: float = 0.2
    kd: float = 1.5
    max_torque_nm: float = 0.2
    epochs: int = 40
    lr: float = 1e-3
    dagger_iters: int = 4  # roll out learner, relabel w/ expert
    dagger_episodes: int = 30
    seed: int = 0


def _pd_action(obs: np.ndarray, cfg: BCConfig) -> np.ndarray:
    sigma, omega = obs[:3], obs[3:]
    return np.clip((-cfg.kp * sigma - cfg.kd * omega) / cfg.max_torque_nm, -1, 1).astype(np.float32)


def dagger_rollout(
    policy: MLPPolicy, cfg: BCConfig, n_episodes: int, seed0: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the LEARNER; relabel each visited state with the PD expert.

    This corrects behavior-cloning covariate shift: the learner visits its
    own state distribution, and the expert (PD) provides the correct action
    there, which is aggregated into the training set.
    """
    env = BasiliskAttitudeEnv(
        AttitudeEnvConfig(
            episode_length=cfg.demo_episode_len, max_torque_nm=cfg.max_torque_nm, seed=seed0
        )
    )
    obs_list, act_list = [], []
    policy.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed0 + ep)
            for _ in range(cfg.demo_episode_len):
                obs_list.append(obs.copy())
                act_list.append(_pd_action(obs, cfg))  # expert label at learner state
                a = policy(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                obs, r, term, trunc, info = env.step(a)
                if trunc:
                    break
    env.close()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def collect_demos(cfg: BCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the PD demonstrator; collect (obs, action) pairs."""
    env = BasiliskAttitudeEnv(
        AttitudeEnvConfig(
            episode_length=cfg.demo_episode_len, max_torque_nm=cfg.max_torque_nm, seed=cfg.seed
        )
    )
    obs_list, act_list = [], []
    for ep in range(cfg.n_demo_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        for _ in range(cfg.demo_episode_len):
            sigma, omega = obs[:3], obs[3:]
            u = np.clip((-cfg.kp * sigma - cfg.kd * omega) / cfg.max_torque_nm, -1, 1)
            obs_list.append(obs.copy())
            act_list.append(u.astype(np.float32))
            obs, r, term, trunc, info = env.step(u)
            if trunc:
                break
    env.close()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def _fit(policy: MLPPolicy, Xt: torch.Tensor, Yt: torch.Tensor, cfg: BCConfig) -> float:
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    n = len(Xt)
    last = 0.0
    policy.train()
    for _ep in range(cfg.epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            loss = nn.functional.mse_loss(policy(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        last = ep_loss / n
    return last


def train_bc(cfg: BCConfig) -> tuple[MLPPolicy, list[float]]:
    torch.manual_seed(cfg.seed)
    X, Y = collect_demos(cfg)
    logger.info("BC: collected %d expert demo pairs", len(X))
    policy = MLPPolicy()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    losses = [_fit(policy, Xt, Yt, cfg)]
    logger.info(
        "BC iter 0: mse=%.6f, success=%.3f",
        losses[-1],
        evaluate(policy, 30, cfg.demo_episode_len)["success_rate"],
    )
    # DAgger: aggregate learner-state / expert-action pairs to fix covariate shift
    for it in range(cfg.dagger_iters):
        Xd, Yd = dagger_rollout(policy, cfg, cfg.dagger_episodes, seed0=1000 + it * 100)
        Xt = torch.cat([Xt, torch.tensor(Xd)])
        Yt = torch.cat([Yt, torch.tensor(Yd)])
        losses.append(_fit(policy, Xt, Yt, cfg))
        sr = evaluate(policy, 30, cfg.demo_episode_len)["success_rate"]
        logger.info(
            "DAgger iter %d: dataset=%d, mse=%.6f, success=%.3f", it + 1, len(Xt), losses[-1], sr
        )
    return policy, losses


def evaluate(policy: MLPPolicy, n_trials: int = 217, ep_len: int = 400, seed: int = 42) -> dict:
    env = BasiliskAttitudeEnv(AttitudeEnvConfig(episode_length=ep_len, seed=seed))
    policy.eval()
    successes = 0
    finals = []
    with torch.no_grad():
        for k in range(n_trials):
            obs, info = env.reset(seed=seed + k)
            for _ in range(ep_len):
                a = policy(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                obs, r, term, trunc, info = env.step(a)
                if trunc:
                    break
            finals.append(info["best_pointing_deg"])
            if info["success"]:
                successes += 1
    env.close()
    z = 1.96
    p = successes / n_trials
    denom = 1 + z * z / n_trials
    half = z * math.sqrt(p * (1 - p) / n_trials + z * z / (4 * n_trials**2)) / denom
    center = (p + z * z / (2 * n_trials)) / denom
    return {
        "n_trials": n_trials,
        "successes": successes,
        "success_rate": p,
        "wilson_95ci": [max(0.0, center - half), min(1.0, center + half)],
        "best_pointing_deg_mean": float(np.mean(finals)),
        "policy": "behavior_cloned_MLP",
    }


def main() -> None:
    import click

    @click.command()
    @click.option("--n-trials", default=217, show_default=True)
    @click.option("--out", default="evidence/curriculum", show_default=True)
    def cli(n_trials: int, out: str) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        cfg = BCConfig()
        policy, losses = train_bc(cfg)
        res = evaluate(policy, n_trials=n_trials)
        res["bc_final_mse"] = losses[-1]
        logger.info(
            "Learned (BC) attitude policy: %d/%d = %.1f%% [%.3f, %.3f]; best pointing %.4f° "
            "(BC mse %.5f)",
            res["successes"],
            res["n_trials"],
            100 * res["success_rate"],
            res["wilson_95ci"][0],
            res["wilson_95ci"][1],
            res["best_pointing_deg_mean"],
            res["bc_final_mse"],
        )
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).isoformat().replace(":", "").split(".")[0]
        p = Path(out) / f"basilisk_phase1_learned_bc_{stamp}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))
        # also save the trained policy
        ckpt = Path("checkpoints") / "learned_attitude_bc.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), ckpt)
        logger.info("Wrote %s + %s", p, ckpt)

    cli()


if __name__ == "__main__":
    main()
