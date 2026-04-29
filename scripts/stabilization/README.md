# Stabilization experiments

Two self-contained scripts. Neither touches `src/red_within_blue/`. Each
produces a standard `metrics.npz` + `summary.json` so downstream tools can
ingest them like any other experiment directory.

Design spec: `docs/08-stabilization-experiments.md`.

---

## EXP-A — `twin_critic_experiment.py`

**Question.** Would TD(0) + target-net + twin-Q have beaten the current MC
baseline for the CTDE blue critic?

Reference config: `configs/pair-cooperate-coop.yaml` (10×10 cooperative, N=2).

Four variants (all share `grad_clip: 0.5`, `ent_coef: 0.05`):

| code | target | # critics | description |
|------|--------|-----------|-------------|
| A0 | live (none) | 1 | **Baseline.** Monte-Carlo team returns — the current production loss (`src/red_within_blue/training/losses.py:222`). |
| A1 | Polyak (τ=0.005) | 1 | TD(0) against a slow target-net. |
| A2 | live min(V1, V2) | 2 | TD(0) against twin critics, no target. |
| A3 | Polyak min(V1_t, V2_t) | 2 | TD(0) + twin + target. (SAC's Q-target.) |

Run:

```bash
# smoke (150 eps × 2 seeds, ~3 s each)
python scripts/stabilization/twin_critic_experiment.py --variant A0 --smoke
python scripts/stabilization/twin_critic_experiment.py --variant A1 --smoke
python scripts/stabilization/twin_critic_experiment.py --variant A2 --smoke
python scripts/stabilization/twin_critic_experiment.py --variant A3 --smoke

# full (15000 eps × 5 seeds, ~20 min each)
python scripts/stabilization/twin_critic_experiment.py --variant A0
python scripts/stabilization/twin_critic_experiment.py --variant A1
python scripts/stabilization/twin_critic_experiment.py --variant A2
python scripts/stabilization/twin_critic_experiment.py --variant A3
```

Outputs → `experiments/stabilization/twin-critic-<V>/metrics.npz`.

Compare on three numbers from §6.6 of the meta-report:

- Final reward (mean of last 500 eps) per seed, cross-variant.
- `|loss|` p99 trajectory growth rate (bounded ⇒ stable).
- Late-dive count (seeds ending below their ep-5000 reward).

Decision rules live in the spec doc §EXP-A pass criteria.

---

## EXP-B — `offpolicy_red_experiment.py`

**Question.** At a matched red env-step budget, does off-policy Double-DQN
drive blue coverage down faster than on-policy REINFORCE?

Reference setup: `configs/compromise-16x16-5-3b2r.yaml`, frozen blue loaded
from `experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz`.

Two variants:

| code | red trainer | sample reuse |
|------|-------------|--------------|
| B0 | on-policy REINFORCE + entropy reg | 1× |
| B1 | Double-DQN + twin-Q + Polyak target + 50 k replay | ~50× |

Blue is frozen in both variants, so any ΔJ delta is attributable to the red
trainer. Metrics are logged at a cadence measured in **red env-steps**, not
episodes, so B0 and B1 are comparable on a single x-axis.

Run:

```bash
# smoke (2 seeds × 5 k steps, eval every 1 k, ~8 s / ~20 s)
python scripts/stabilization/offpolicy_red_experiment.py --variant B0 --smoke
python scripts/stabilization/offpolicy_red_experiment.py --variant B1 --smoke

# full (3 seeds × 500 k env-steps, eval every 25 k)
python scripts/stabilization/offpolicy_red_experiment.py --variant B0
python scripts/stabilization/offpolicy_red_experiment.py --variant B1
```

Outputs → `experiments/stabilization/offpolicy-red-<V>/metrics.npz`.

Compare on:

- Blue reward at matched env-steps (100 k, 500 k, 2 M). Lower = red winning.
- Red reward (smoke-test — red shouldn't collapse to STAY).
- Wall-clock to reach ΔJ ≥ 8 pp.

Decision rules live in the spec doc §EXP-B pass criteria.

---

## Compartmentalisation rules

Nothing here modifies `src/red_within_blue/`. Each script:

- Reuses only `GridCommEnv`, the existing `Actor` / `JointRedActor`
  networks, `make_multi_agent_reward`, and JIT-friendly rollout helpers.
- Defines its own loss functions, train-step, replay buffer, and driver in
  one file so the variants can't accidentally share weights or config
  state.
- Writes to its own `experiments/stabilization/<name>/` directory — nothing
  else on disk is touched.

This means the production training path in
`src/red_within_blue/training/trainer.py` is identical before and after
running these experiments; only the `experiments/stabilization/` folder
grows.
