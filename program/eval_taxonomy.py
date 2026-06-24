"""Honest settled-gate eval across the FULL structurally-held-out fault taxonomy.

Extends the GAIN-only first eval (``program.rollout.run_heldout_gain_eval``) to all four
fault classes — SIGN, GAIN, GAIN_BIAS, TOTAL_LOSS — for PD, the deployed RMA student
(baseline + estimate-latched), and the privileged teacher. The latch threshold is the
single value tuned on TRAIN GAIN faults (``program.rollout`` default), applied UNIFORMLY
to every held-out test class — no per-class re-tuning, hence no test-set leakage.

TOTAL_LOSS (one actuator axis fully dead, g=0) is out-of-distribution for every class:
the controller is EXPECTED to fail it (it is the shield-only regime), so a low score
there is the correct honest result, not a regression.

Run (requires Basilisk — HANDOFF.md §6):
    python -m program.eval_taxonomy --faults 50 --seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from program import rollout
from program.fault_taxonomy import Fault, FaultClass, heldout_test_faults

logger = logging.getLogger(__name__)

TAXONOMY_OUT = Path("evidence/program/taxonomy_settled.json")
CLASSES = [FaultClass.SIGN, FaultClass.GAIN, FaultClass.GAIN_BIAS, FaultClass.TOTAL_LOSS]


def run_taxonomy_eval(
    n_faults: int = 50,
    n_seeds: int = 10,
    *,
    student_ckpt: str | Path = "checkpoints/rma_student.pt",
    latch_below_deg: float = 3.0,
    out_path: Path = TAXONOMY_OUT,
    env_factory: rollout.EnvFactory = rollout.default_env_factory,
) -> dict:
    """Settled-gate eval over the four held-out fault classes x four controllers.

    Fails fast if Basilisk is absent (the harness would otherwise count every errored
    rollout as a failure and report a misleading 0% — Prime Directive, HANDOFF.md §1).
    """
    if env_factory is rollout.default_env_factory and not rollout.basilisk_available():
        raise ModuleNotFoundError(
            "Basilisk is required for the real rollout but is not installed (HANDOFF.md §6).",
            name="Basilisk",
        )

    from program import determinism, eval_harness

    determinism.set_global_determinism(7)
    from controller.baselines.classical import (
        ClassicalConfig,
        adaptive_fl_policy,
        icl_adaptive_policy,
        nussbaum_policy,
        pid_unaware_policy,
    )

    cfg = rollout.default_cfg()
    student = rollout.load_rma_student(student_ckpt)
    seeds = list(range(n_seeds))
    ccfg = ClassicalConfig(max_torque_nm=float(cfg.max_torque_nm))

    def pid_maker(_f: Fault) -> rollout.PolicyFn:
        return pid_unaware_policy(ccfg)

    def adaptive_maker(_f: Fault) -> rollout.PolicyFn:
        return adaptive_fl_policy(ccfg)

    def icl_maker(_f: Fault) -> rollout.PolicyFn:
        return icl_adaptive_policy(ccfg)

    def nussbaum_maker(_f: Fault) -> rollout.PolicyFn:
        return nussbaum_policy(ccfg)

    makers: dict[str, rollout.MakePolicy] = {
        "pd_fault_unaware": rollout.pd_maker(cfg),
        "pid_unaware": pid_maker,
        "classical_adaptive": adaptive_maker,
        "icl_adaptive_lit": icl_maker,
        "nussbaum_adaptive_lit": nussbaum_maker,
    }
    # Learned end-to-end RL controller (WS1 Step-2 fair retest), included if trained.
    gru_ckpt = Path("checkpoints/gru_ppo_controller_fair.pt")
    if gru_ckpt.exists():
        from controller.maml.gru_ppo_basilisk import gru_ppo_policy, load_gru_ppo

        gru_model = load_gru_ppo(gru_ckpt)

        def gru_maker(_f: Fault) -> rollout.PolicyFn:
            return gru_ppo_policy(gru_model)

        makers["gru_ppo_fair"] = gru_maker
    makers.update(
        {
            "rma_student_deployed": rollout.rma_student_maker(student, cfg),
            "rma_student_latched": rollout.rma_student_maker(
                student, cfg, latch_below_deg=latch_below_deg
            ),
        }
    )
    # Split-conformant retrains (audit residual M2/F-1): trained on the DECLARED
    # taxonomy TRAIN split only (sample_task_taxonomy; see the checkpoints'
    # .provenance.json sidecars), so every TEST cell is truly EXTRAPOLATIVE for
    # them — unlike the v1 student, whose wide training generator overlaps parts
    # of the TEST region. Included as separate columns; v1 rows are unchanged.
    for tag in ("v2", "v2big"):
        ck = Path(f"checkpoints/rma_student_{tag}.pt")
        if ck.exists():
            makers[f"rma_student_{tag}_latched"] = rollout.rma_student_maker(
                rollout.load_rma_student(ck), cfg, latch_below_deg=latch_below_deg
            )
    makers["teacher_privileged"] = rollout.teacher_maker(cfg)

    by_class: dict[str, dict] = {}
    for fclass in CLASSES:
        faults = heldout_test_faults(fclass, n=n_faults, seed=7_000)
        by_class[fclass.value] = {
            name: eval_harness.evaluate(
                rollout.make_rollout_fn(mk, cfg=cfg, env_factory=env_factory),
                faults,
                seeds,
                label=f"{fclass.value}::{name}",
            )
            for name, mk in makers.items()
        }
        logger.info("done class %s", fclass.value)

    payload = {
        "experiment": "taxonomy_settled",
        "split": "test (structurally held out by construction)",
        "classes": [c.value for c in CLASSES],
        "n_faults": n_faults,
        "n_seeds": n_seeds,
        "primary_gate": "settled_science (0.2 deg held over final dwell window)",
        "latch_below_deg": latch_below_deg,
        "latch_provenance": "tuned on TRAIN GAIN (seed 1000); applied uniformly to all classes",
        "classical_params": {
            "sign_eps": ccfg.sign_eps,
            "exc_thresh": ccfg.exc_thresh,
            "provenance": "tuned on TRAIN GAIN+SIGN faults (argmax settled-science)",
        },
        "nussbaum_params": {
            "nuss_gamma": ccfg.nuss_gamma,
            "nuss_lambda": ccfg.nuss_lambda,
            "nuss_c": ccfg.nuss_c,
            "nuss_deadzone": ccfg.nuss_deadzone,
            "provenance": "tuned on TRAIN SIGN+GAIN (program.tune_nussbaum -> "
            "evidence/program/nussbaum_tune.json)",
        },
        "student_provenance": {
            "rma_student (v1)": "wide training generator (f U(0.7,2.2), |g| U(0.3,1.5), all "
            "sign patterns) whose support OVERLAPS parts of the TEST region — held-out fault "
            "instances, not a held-out regime (audit M2/F-1)",
            "rma_student_v2*": "trained on the DECLARED taxonomy TRAIN split only "
            "(sample_task_taxonomy; ckpt .provenance.json) — every TEST cell extrapolative; "
            "v2big = 4x training tasks",
        },
        "note_total_loss": (
            "TOTAL_LOSS = one axis fully dead (g=0): OOD for every class, the shield-only "
            "regime; controller failure there is the expected honest result"
        ),
        "by_class": by_class,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_path)
    return payload


def _print_table(payload: dict) -> None:
    classes = payload["classes"]
    names = list(payload["by_class"][classes[0]].keys())
    print("\nsettled-science rate (held <=0.2 deg) — rows = controllers, cols = fault classes")
    print(f"{'controller':22s} " + " ".join(f"{c:>11s}" for c in classes))
    for name in names:
        cells = []
        for c in classes:
            rate = payload["by_class"][c][name]["settled_science"]["rate"]
            cells.append(f"{rate:>11.3f}")
        print(f"{name:22s} " + " ".join(cells))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Settled-gate eval across the full fault taxonomy.")
    p.add_argument("--faults", type=int, default=50)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--latch", type=float, default=3.0, help="latch_below_deg (train-tuned)")
    p.add_argument("--out", default=str(TAXONOMY_OUT))
    args = p.parse_args()
    payload = run_taxonomy_eval(
        n_faults=args.faults,
        n_seeds=args.seeds,
        latch_below_deg=args.latch,
        out_path=Path(args.out),
    )
    _print_table(payload)


if __name__ == "__main__":
    main()
