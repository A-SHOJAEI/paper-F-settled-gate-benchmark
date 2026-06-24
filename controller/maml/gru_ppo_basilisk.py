"""Recurrent GRU controller: sequence-BC + DAgger warm-start + PPO (real Basilisk).

WS1 Step-2 — the *fair* retest of the recurrent-RL controller. The audit found two
defects that meant the recurrent premise was never actually tested:

  1. **Obs-only features.** The GRU was fed only ``[sigma, omega]``, so it could not see
     how its command moved the rate and could not identify the unobserved actuator fault.
     This version adds ``delta_obs`` -> ``[obs, delta_obs]`` (12-d), which carries the
     fault response; the recurrence remembers the command. (NB: unlike the RMA student —
     which outputs the fault estimate z — a *direct-action* policy must NOT be fed
     ``prev_action``: it then minimises BC loss by echoing its own previous action, the
     "copycat" collapse, giving a degenerate constant output. So prev_action is excluded.)
  2. **Memoryless PPO update.** The old PPO step flattened every transition and fed each
     as a length-1 sequence (``model(O.unsqueeze(1))``), zero-resetting the hidden state
     each step — so PPO's policy gradient trained a *memoryless* view and the recurrence
     was never optimised by RL. This version runs the PPO update as **per-episode
     sequences with the hidden state propagated (BPTT)**, so the recurrence is actually
     trained.

Pipeline: sequence-BC warm-start (hidden carried per episode) -> on-policy DAgger
(expert-relabel, the proven covariate-shift fix) -> PPO+GAE polish to the gate. The
honest head-to-head eval is run separately through ``program.rollout`` (settled gate,
held-out taxonomy) via ``gru_ppo_policy`` so the GRU is scored on the identical harness
as every other controller.

Run: ``python -m controller.maml.gru_ppo_basilisk --bc-epochs 40 --ppo-iters 200``
Emits ``evidence/curriculum/gru_ppo_fair_retest.json`` and
``checkpoints/gru_ppo_controller_fair.pt`` (the pre-audit obs-only artifacts are left
in place).
"""

from __future__ import annotations

# ruff: noqa: N803, N806, E741
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from controller.maml.meta_imitation_basilisk import (
    MetaConfig,
    TaskSpec,
    _apply_fault,
    _expert_action,
    _task_env,
)

logger = logging.getLogger(__name__)

OBS, ACT = 6, 3
# [obs(6), delta_obs(6)] = 12. prev_action is EXCLUDED (it causes copycat collapse in a
# direct-action policy). delta_obs carries the fault response and the recurrence remembers
# the command; an AUXILIARY fault-prediction head (forward_aux) further forces the hidden
# state to encode the fault z, sharpening the (otherwise fragile) online sign inference.
FEAT_DIM = OBS + OBS  # 12


@dataclass
class PPOConfig:
    train_tasks: tuple[TaskSpec, ...] = (
        TaskSpec(0.9, (-1, 1, 1)),
        TaskSpec(1.3, (-1, 1, 1)),
        TaskSpec(1.6, (1, 1, -1)),
        TaskSpec(0.8, (1, -1, 1)),
        TaskSpec(1.2, (1, 1, -1)),
        TaskSpec(2.0, (-1, 1, 1)),
    )
    test_tasks: tuple[TaskSpec, ...] = (
        TaskSpec(1.3, (-1, 1, 1)),
        TaskSpec(1.6, (1, 1, -1)),
        TaskSpec(2.2, (-1, -1, 1)),  # 2-axis flip + extrapolated inertia
    )
    hidden: int = 256
    ep_len: int = 400
    max_torque_nm: float = 0.2
    gate_deg: float = 0.2
    bc_epochs: int = 40
    bc_demo_eps: int = 8
    bc_lr: float = 1e-3
    aux_w: float = 0.1  # weight on the auxiliary fault-prediction loss (sharpens sign inference)
    dagger_iters: int = 6  # on-policy expert-relabel passes (fixes BC covariate shift)
    dagger_fit_epochs: int = 15  # refit epochs on the aggregated set per DAgger iter
    ppo_iters: int = 200
    ppo_rollouts: int = 8  # episodes per PPO iteration
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    ppo_lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    reward_scale: float = 0.1
    eval_episodes: int = 25
    seed: int = 0
    base: MetaConfig = field(default_factory=lambda: MetaConfig(ep_len=400, max_torque_nm=0.2))


class GRUActorCritic(nn.Module):
    def __init__(self, obs_dim: int = FEAT_DIM, act_dim: int = ACT, hidden: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden, batch_first=True)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        self.value = nn.Linear(hidden, 1)
        # Auxiliary fault predictor z=[g0,g1,g2,f-1]: trained during imitation so the hidden
        # state must encode the fault (robust sign inference) and cannot copycat prev_action.
        self.aux_z = nn.Linear(hidden, 4)

    def forward(
        self, feat_seq: torch.Tensor, h0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, h = self.gru(feat_seq, h0)
        mean = torch.tanh(self.mean(out))
        value = self.value(out).squeeze(-1)
        return mean, value, h

    def forward_aux(
        self, feat_seq: torch.Tensor, h0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(mean_action, predicted_z, h) — for the imitation auxiliary loss."""
        out, h = self.gru(feat_seq, h0)
        return torch.tanh(self.mean(out)), self.aux_z(out), h


def _pointing_deg(sigma: np.ndarray) -> float:
    return float(np.rad2deg(4.0 * math.atan(np.linalg.norm(sigma))))


def _feat(obs: np.ndarray, _prev_a: np.ndarray, prev_obs: np.ndarray) -> np.ndarray:
    """[obs, delta_obs] (12-d). prev_action is excluded (copycat); delta_obs + recurrence
    + the auxiliary z-head (GRUActorCritic.forward_aux) carry the fault-inference signal."""
    return np.concatenate([obs, obs - prev_obs]).astype(np.float32)  # type: ignore[no-any-return]


def _task_z(task: TaskSpec) -> np.ndarray:
    """Auxiliary target: the true fault z = [g0, g1, g2, f-1]."""
    g = task.fault
    return np.array([g[0], g[1], g[2], task.inertia_factor - 1.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# imitation data (sequence-BC + DAgger)
# --------------------------------------------------------------------------- #
def _collect_seq_demos(task: TaskSpec, n_ep: int, cfg: PPOConfig, seed0: int) -> list[tuple]:
    """Per-episode (feat_seq, expert_action_seq) — fault-aware expert on Basilisk.

    Features are built along the expert's own trajectory (prev_action = expert action)."""
    env = _task_env(task, cfg.base, seed=seed0)
    z = _task_z(task)
    demos = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed0 + ep)
        prev_a = np.zeros(ACT, dtype=np.float32)
        prev_obs = obs.copy()
        F, A = [], []
        for _ in range(cfg.ep_len):
            a = _expert_action(obs, task, cfg.base)  # fault pre-inverted normalized torque
            F.append(_feat(obs, prev_a, prev_obs))
            A.append(a)
            prev_a, prev_obs = a, obs.copy()
            obs, _, _, tr, _ = env.step(_apply_fault(a, task))
            if tr:
                break
        demos.append((np.asarray(F, dtype=np.float32), np.asarray(A, dtype=np.float32), z))
    env.close()
    return demos


def bc_warmstart(model: GRUActorCritic, cfg: PPOConfig) -> None:
    """Sequence-BC: hidden carried across each episode so recurrence infers the fault."""
    demos: list[tuple] = []
    for i, t in enumerate(cfg.train_tasks):
        demos += _collect_seq_demos(t, cfg.bc_demo_eps, cfg, seed0=1000 + i * 100)
    opt = torch.optim.Adam(
        list(model.gru.parameters())
        + list(model.mean.parameters())
        + list(model.aux_z.parameters()),
        lr=cfg.bc_lr,
    )
    g = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(cfg.bc_epochs):
        perm = torch.randperm(len(demos), generator=g)
        tot = 0.0
        for idx in perm:
            F, A, Z = demos[idx]
            mean, zhat, _ = model.forward_aux(torch.tensor(F).unsqueeze(0))  # hidden 0 each ep
            Zt = torch.tensor(Z).view(1, 1, 4).expand(-1, mean.shape[1], -1)
            loss = nn.functional.mse_loss(mean, torch.tensor(A).unsqueeze(0))
            loss = loss + cfg.aux_w * nn.functional.mse_loss(zhat, Zt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if epoch % 10 == 0:
            logger.info("  BC epoch %d: loss %.5f", epoch, tot / len(demos))


@torch.no_grad()
def _collect_policy_traj(
    model: GRUActorCritic, task: TaskSpec, cfg: PPOConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the CURRENT GRU (mean action); return (visited_obs, visited_feats)."""
    env = _task_env(task, cfg.base, seed=seed)
    obs, _ = env.reset(seed=seed)
    h = None
    prev_a = np.zeros(ACT, dtype=np.float32)
    prev_obs = obs.copy()
    O, F = [], []
    for _ in range(cfg.ep_len):
        feat = _feat(obs, prev_a, prev_obs)
        mean, _, h = model(torch.tensor(feat, dtype=torch.float32).view(1, 1, -1), h)
        a = torch.clamp(mean.view(-1), -1.0, 1.0).numpy().astype(np.float32)
        O.append(obs.copy())
        F.append(feat)
        prev_a, prev_obs = a, obs.copy()
        obs, _, _, tr, _ = env.step(_apply_fault(a, task))
        if tr:
            break
    env.close()
    return np.asarray(O, dtype=np.float32), np.asarray(F, dtype=np.float32)


def dagger(model: GRUActorCritic, cfg: PPOConfig) -> list[dict]:
    """On-policy DAgger (recurrent): roll the GRU, relabel its OWN visited states with the
    fault-aware expert, aggregate, refit (sequence BC). History disambiguates the fault."""
    demos: list[tuple] = []
    for i, t in enumerate(cfg.train_tasks):
        demos += _collect_seq_demos(t, cfg.bc_demo_eps, cfg, seed0=1000 + i * 100)
    opt = torch.optim.Adam(
        list(model.gru.parameters())
        + list(model.mean.parameters())
        + list(model.aux_z.parameters()),
        lr=cfg.bc_lr,
    )
    g = torch.Generator().manual_seed(cfg.seed + 1)
    history = []
    for it in range(cfg.dagger_iters):
        for i, t in enumerate(cfg.train_tasks):  # relabel the policy's visited states
            O, F = _collect_policy_traj(model, t, cfg, seed=5000 + it * 100 + i)
            A = np.asarray([_expert_action(o, t, cfg.base) for o in O], dtype=np.float32)
            demos.append((F, A, _task_z(t)))
        for _ in range(cfg.dagger_fit_epochs):
            for idx in torch.randperm(len(demos), generator=g):
                F, A, Z = demos[idx]
                mean, zhat, _ = model.forward_aux(torch.tensor(F).unsqueeze(0))
                Zt = torch.tensor(Z).view(1, 1, 4).expand(-1, mean.shape[1], -1)
                loss = nn.functional.mse_loss(mean, torch.tensor(A).unsqueeze(0))
                loss = loss + cfg.aux_w * nn.functional.mse_loss(zhat, Zt)
                opt.zero_grad()
                loss.backward()
                opt.step()
        ev = evaluate(model, cfg)
        history.append({"dagger_iter": it, **ev})
        logger.info("  DAgger iter %d: gate %.1f%%", it, ev["aggregate_gate_pct"])
    return history


# --------------------------------------------------------------------------- #
# internal progress eval (transient-min gate; the OFFICIAL eval is via program.rollout)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: GRUActorCritic, cfg: PPOConfig, seed0: int = 7000) -> dict:
    per_task = []
    gate_pcts: list[float] = []
    for i, task in enumerate(cfg.test_tasks):
        succ, bests = 0, []
        for k in range(cfg.eval_episodes):
            env = _task_env(task, cfg.base, seed=seed0 + i * 100 + k)
            obs, _ = env.reset(seed=seed0 + i * 100 + k)
            h = None
            prev_a = np.zeros(ACT, dtype=np.float32)
            prev_obs = obs.copy()
            best = _pointing_deg(obs[:3])
            for _ in range(cfg.ep_len):
                feat = _feat(obs, prev_a, prev_obs)
                mean, _, h = model(torch.tensor(feat, dtype=torch.float32).view(1, 1, -1), h)
                a = torch.clamp(mean.view(-1), -1.0, 1.0).numpy().astype(np.float32)
                prev_a, prev_obs = a, obs.copy()
                obs, _, _, tr, _ = env.step(_apply_fault(a, task))
                best = min(best, _pointing_deg(obs[:3]))
                if tr:
                    break
            env.close()
            bests.append(best)
            succ += int(best <= cfg.gate_deg)
        gate_pct = round(100 * succ / cfg.eval_episodes, 1)
        gate_pcts.append(gate_pct)
        per_task.append(
            {
                "task": {"f": task.inertia_factor, "fault": list(task.fault)},
                "gate_pct": gate_pct,
                "mean_best_deg": round(float(np.mean(bests)), 3),
            }
        )
    agg = round(float(np.mean(gate_pcts)), 1) if gate_pcts else 0.0
    return {"aggregate_gate_pct": agg, "per_task": per_task}


# --------------------------------------------------------------------------- #
# PPO with per-episode BPTT (the recurrence-trained fix)
# --------------------------------------------------------------------------- #
def _rollout(model: GRUActorCritic, task: TaskSpec, cfg: PPOConfig, seed: int) -> dict:
    """One stochastic episode; stores the FEATURE sequence (for BPTT in the update)."""
    env = _task_env(task, cfg.base, seed=seed)
    obs, _ = env.reset(seed=seed)
    h = None
    std = model.log_std.clamp(-5, 2).exp()
    prev_a = np.zeros(ACT, dtype=np.float32)
    prev_obs = obs.copy()
    F, A, LP, V, R = [], [], [], [], []
    for _ in range(cfg.ep_len):
        feat = _feat(obs, prev_a, prev_obs)
        with torch.no_grad():
            mean, value, h = model(torch.tensor(feat).view(1, 1, -1), h)
        m = mean.view(-1)
        dist = torch.distributions.Normal(m, std)
        u = dist.sample()
        a = torch.clamp(u, -1.0, 1.0)
        logp = dist.log_prob(u).sum()
        F.append(feat)
        A.append(u.numpy())
        LP.append(float(logp))
        V.append(float(value.view(-1)))
        prev_a, prev_obs = a.numpy().astype(np.float32), obs.copy()
        obs, _, _, tr, _ = env.step(_apply_fault(a.numpy(), task))
        R.append(-_pointing_deg(obs[:3]) * cfg.reward_scale)
        if tr:
            break
    env.close()
    return {
        "F": np.asarray(F, dtype=np.float32),
        "A": np.asarray(A, dtype=np.float32),
        "LP": np.asarray(LP, dtype=np.float32),
        "V": np.asarray(V, dtype=np.float32),
        "R": np.asarray(R, dtype=np.float32),
    }


def _gae(R: np.ndarray, V: np.ndarray, cfg: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
    T = len(R)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nextv = V[t + 1] if t + 1 < T else 0.0
        delta = R[t] + cfg.gamma * nextv - V[t]
        last = delta + cfg.gamma * cfg.lam * last
        adv[t] = last
    ret = adv + V
    return adv, ret


def ppo_train(model: GRUActorCritic, cfg: PPOConfig) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ppo_lr)
    rng = np.random.default_rng(cfg.seed)
    history = []
    for it in range(cfg.ppo_iters):
        batch = [
            _rollout(
                model,
                cfg.train_tasks[int(rng.integers(len(cfg.train_tasks)))],
                cfg,
                seed=cfg.seed + it * 97 + j,
            )
            for j in range(cfg.ppo_rollouts)
        ]
        eps = []
        for b in batch:
            adv, ret = _gae(b["R"], b["V"], cfg)
            eps.append({"F": b["F"], "A": b["A"], "LP": b["LP"], "adv": adv, "ret": ret})
        cat = np.concatenate([e["adv"] for e in eps])
        a_mean, a_std = float(cat.mean()), float(cat.std() + 1e-6)

        for _ in range(cfg.ppo_epochs):
            opt.zero_grad()
            total_loss = torch.zeros(())
            nsteps = 0
            for e in eps:
                # Per-episode sequence: hidden propagated across T -> recurrence is trained.
                F = torch.tensor(e["F"]).unsqueeze(0)  # [1,T,15]
                A = torch.tensor(e["A"])
                LP_old = torch.tensor(e["LP"])
                adv_t = torch.tensor((e["adv"] - a_mean) / a_std)
                ret_t = torch.tensor(e["ret"])
                mean, value, _ = model(F)
                mean, value = mean.squeeze(0), value.squeeze(0)
                std = model.log_std.clamp(-5, 2).exp()
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(A).sum(-1)
                ratio = torch.exp(logp - LP_old)
                s1 = ratio * adv_t
                s2 = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv_t
                pol_loss = -torch.min(s1, s2).sum()
                val_loss = nn.functional.mse_loss(value, ret_t, reduction="sum")
                ent = dist.entropy().sum()
                total_loss = total_loss + pol_loss + 0.5 * val_loss - 0.01 * ent
                nsteps += len(A)
            (total_loss / max(nsteps, 1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        if it % 25 == 0 or it == cfg.ppo_iters - 1:
            ev = evaluate(model, cfg)
            history.append({"iter": it, **ev})
            logger.info("  PPO iter %d: gate %.1f%%", it, ev["aggregate_gate_pct"])
    return history


# --------------------------------------------------------------------------- #
# deployment policy for the honest settled-gate eval (program.rollout)
# --------------------------------------------------------------------------- #
def gru_ppo_policy(model: GRUActorCritic) -> Callable[[np.ndarray], np.ndarray]:
    """Fresh stateful policy(obs)->a (clipped mean action) for program.rollout. Builds the
    [obs, prev_action, delta_obs] feature and carries the GRU hidden state per episode."""
    h: list[torch.Tensor] = []
    prev_a = np.zeros(ACT, dtype=np.float32)
    prev_obs: list[np.ndarray] = []

    def policy(obs: np.ndarray) -> np.ndarray:
        nonlocal prev_a
        po = prev_obs[0] if prev_obs else obs
        feat = _feat(obs, prev_a, po)
        with torch.no_grad():
            mean, _, hn = model(torch.tensor(feat).view(1, 1, -1), h[0] if h else None)
        a = torch.clamp(mean.view(-1), -1.0, 1.0).numpy().astype(np.float32)
        h.clear()
        h.append(hn)
        prev_a = a
        prev_obs.clear()
        prev_obs.append(obs.copy())
        return a  # type: ignore[no-any-return]

    return policy


def load_gru_ppo(
    path: str | Path = "checkpoints/gru_ppo_controller_fair.pt", hidden: int = 256
) -> GRUActorCritic:
    model = GRUActorCritic(hidden=hidden)
    model.load_state_dict(torch.load(Path(path), map_location="cpu"))
    model.eval()
    return model


def run(cfg: PPOConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = GRUActorCritic(hidden=cfg.hidden)

    logger.info("Sequence-BC warm-start (GRU, hidden=%d, feat=%d)…", cfg.hidden, FEAT_DIM)
    bc_warmstart(model, cfg)
    bc_eval = evaluate(model, cfg)
    logger.info("After BC: gate %.1f%%", bc_eval["aggregate_gate_pct"])

    logger.info("DAgger (on-policy expert relabel, %d iters)…", cfg.dagger_iters)
    dagger_history = dagger(model, cfg)
    dagger_eval = evaluate(model, cfg)
    logger.info("After DAgger: gate %.1f%%", dagger_eval["aggregate_gate_pct"])

    logger.info("PPO polish, recurrent BPTT (%d iters)…", cfg.ppo_iters)
    history = ppo_train(model, cfg)
    final_eval = evaluate(model, cfg)

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/gru_ppo_controller_fair.pt")
    result = {
        "method": "GRU actor-critic, FAIR retest: [obs,prev_a,Δobs] features + recurrent "
        "(BPTT) PPO; sequence-BC + DAgger warm-start (real Basilisk, faults)",
        "fixes_vs_audit": [
            "features: obs-only -> [obs, delta_obs] (12-d); delta_obs makes the fault "
            "inferable via the recurrence. prev_action EXCLUDED: it induces imitation "
            "copycat-collapse in a direct-action policy (degenerate constant output).",
            "PPO update: length-1 memoryless -> per-episode sequence BPTT (recurrence trained)",
        ],
        "headline": {
            "bc_warmstart_gate_pct": bc_eval["aggregate_gate_pct"],
            "dagger_gate_pct": dagger_eval["aggregate_gate_pct"],
            "final_gate_pct": final_eval["aggregate_gate_pct"],
            "per_task_final": final_eval["per_task"],
            "note": "internal gate is the transient-min progress metric; the honest "
            "settled-gate head-to-head is run via program.rollout (gru_ppo_policy).",
        },
        "bc_eval": bc_eval,
        "dagger_eval": dagger_eval,
        "final_eval": final_eval,
        "dagger_history": dagger_history,
        "ppo_history": history,
        "config": {
            "hidden": cfg.hidden,
            "feat_dim": FEAT_DIM,
            "bc_epochs": cfg.bc_epochs,
            "ppo_iters": cfg.ppo_iters,
            "gate_deg": cfg.gate_deg,
            "eval_episodes": cfg.eval_episodes,
        },
    }
    out = Path("evidence/curriculum/gru_ppo_fair_retest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", out)
    return result


def main() -> None:
    import click

    @click.command()
    @click.option("--bc-epochs", default=40, show_default=True)
    @click.option("--ppo-iters", default=200, show_default=True)
    @click.option("--hidden", default=256, show_default=True)
    @click.option("--eval-episodes", default=25, show_default=True)
    def cli(bc_epochs: int, ppo_iters: int, hidden: int, eval_episodes: int) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        cfg = PPOConfig(
            bc_epochs=bc_epochs, ppo_iters=ppo_iters, hidden=hidden, eval_episodes=eval_episodes
        )
        res = run(cfg)
        h = res["headline"]
        logger.info(
            "DONE. GRU controller gate: BC %.1f%% → DAgger %.1f%% → PPO %.1f%%.",
            h["bc_warmstart_gate_pct"],
            h["dagger_gate_pct"],
            h["final_gate_pct"],
        )

    cli()


if __name__ == "__main__":
    main()
