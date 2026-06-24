"""RMA-style fault-adaptive attitude control (the controller capability fix).

Rapid Motor Adaptation (Kumar et al., RSS 2021) applied to the actuator-sign
fault that defeats the nominal BC policy (0% gate) and pooled meta-imitation
(transfer ratio 1.0). Two phases:

  1. a privileged **teacher** π(obs, z) that observes the latent fault
     z = (actuator-sign pattern, inertia) and controls through it;
  2. a **history student** φ — a GRU over (obs, previous-action) pairs — that
     *infers* ẑ online (system identification), so the deployed policy
     π(obs, φ(history)) adapts to an unobserved fault with no privileged input.

The previous-action input is the ingredient the earlier obs-only GRU-BC lacked:
the sign of an actuator fault is identifiable only from how the state responds
to a *known command*.
"""
