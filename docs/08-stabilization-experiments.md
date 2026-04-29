# Stabilisation Experiments — Twin-Q + Target-Net + Off-Policy Red

**Goal.** Decide empirically whether two SAC-inspired stabilisers reduce variance and / or improve sample efficiency in our training pipeline — without bolting on a named algorithm. Each experiment is a controlled comparison whose headline statistic is **one specific number** we already report in the meta-report.

**Scope.** Two independent experiments, in order of cost:

- **EXP-A — Twin-Q + Target network for the central critic.** Addresses the critic-drift failure mode documented in `docs/superpowers/plans/bubbly-strolling-puddle.md`.
- **EXP-B — Off-policy replay for the joint-red trainer.** Addresses the per-generation rollout-discard waste in `scripts/coevo.py` / `scripts/coevo_r6.py`.

Both experiments reuse the existing `experiments/pair-cooperate-coop/` (EXP-A) and `experiments/compromise-16x16-5-*-coevo/` (EXP-B) baselines as controls. No new environments, no new configs beyond the two variants each experiment introduces.

**Not in scope.** SAC itself (continuous-action, replaces our entire training loop). Algorithmic ports we don't need yet: automatic entropy tuning, Orbax checkpointing, squashed-Gaussian policy, GAE(λ).

---

## EXP-A — Twin-Q + Target network (blue critic stability)

### Motivation

Current state: `actor_critic_loss_ctde` in `src/red_within_blue/training/losses.py` uses a single `Critic` V(s) trained with TD(0) bootstrapping. In the 2026-04-18 run (`experiments/pair-cooperate-coop/metrics.npz`) the `|loss|` p99 drifted 0.46 → 23,282 over 15 k episodes and reward collapsed +2.07 → +0.35. `grad_clip: 0.5` (merged since) keeps the lid on but is a bandage — the bootstrap target still depends on the live critic, so any transient overshoot self-amplifies.

Twin-Q and target networks are the two canonical fixes, attacking two different root causes:

| failure mode | canonical fix | cost |
|---|---|---|
| Self-referential bootstrap: `V(s) ← r + γ·V(s')` with the live critic. | **Target network** — bootstrap uses a Polyak-averaged slow copy (τ=0.005 → half-life ≈ 140 steps). | +1 critic pytree, +1 Polyak update per step. |
| Single-critic optimism: any early overshoot on one state attracts policy → critic back-chases. | **Twin-Q + min target** — two critics with different init seeds; bootstrap uses `jnp.minimum(V1, V2)`. | +1 critic pytree, +1 forward pass. |

### Variants (4 conditions — 3 experiment cells + 1 control)

All at `configs/pair-cooperate-coop.yaml`, 15 000 episodes, 5 seeds, `grad_clip: 0.5` **kept on** across all variants so we isolate the marginal effect of the stabiliser. `ent_coef: 0.05` held constant. TD(0) target held constant (the MC switch is Stage 3 of the existing critic-drift plan; twin-Q runs *before* that switch).

| code | target source | # critics | description |
|------|---------------|-----------|-------------|
| **A0** | live critic | 1 | **Control.** Status quo from the standing `bubbly-strolling-puddle` plan, Stage 1. |
| **A1** | Polyak target (τ=0.005) | 1 | Target-net only. |
| **A2** | live critic (min over two) | 2 | Twin-Q only. |
| **A3** | Polyak target (min over two) | 2 | Twin-Q **+** target net. (SAC's Q-target.) |

### Metrics (all read off `metrics.npz`)

- **M1 — Final reward** (mean of last 500 eps, mean ± σ over 5 seeds). **Primary.**
- **M2 — `|loss|` p99 trajectory.** Geometric-mean growth rate from ep 0→15 000. Report as a single `γ_|loss|` scalar per variant; values near 1.0 mean bounded.
- **M3 — Reward monotonicity.** Count how many of the 5 seeds end below their ep-5000 reward (a "late dive"). A0 shows 5/5 dives; target is 0/5.
- **M4 — Variance.** Per-seed σ of final reward. Twin-Q is *supposed* to cut variance.

### Pass criteria (per variant)

- M1 ≥ +1.8 mean, σ ≤ 0.30.
- M3 = 0/5 late dives.
- γ_|loss| ≤ 1.0005 per episode (no multi-order-of-magnitude drift).

### What each comparison tells us

| comparison | claim | evidence for / against |
|---|---|---|
| A1 vs A0 | "Target net alone stabilises" | If A1 passes and A0 fails: target-net is sufficient. |
| A2 vs A0 | "Twin-Q alone stabilises" | If A2 passes: the issue was optimism bias, not bootstrap self-reference. |
| A3 vs A1, A2 | "Both stabilisers compose" | If A3 ≥ max(A1, A2) on M1/M4: strictly better; adopt as new default. If A3 underperforms both: the two interact badly (double stabilisation is over-constrained). |
| A3 vs Stage-2 adv-norm (from standing plan) | "Which is the cheaper stabiliser?" | Decides where twin-Q+target lives in the critic-drift-plan ladder: between Stage 1 and Stage 2, or after. |

### Wall-clock estimate

15 000 eps × 5 seeds × 4 variants ≈ **4 × 20 min ≈ 80 min** on CPU. No retraining of red. Single command per variant.

### Rollback criterion

If A3 underperforms A0 on M1 by > 0.3 pp, revert. Keep grad_clip as the sole guardrail and move directly to Stage 3 (MC targets) in the standing plan.

---

## EXP-B — Off-policy replay for the joint red actor

### Motivation

Current state: `scripts/coevo.py` trains the joint red via on-policy REINFORCE. Each ES generation throws away its rollouts. At `r6` (`project_coevo_r6.md`) one generation is 8 × 2 eps × 200 steps ≈ 3 200 transitions *used once*. If we could reuse each transition ~50×, the same wall-clock buys a much denser ε-sweep or misbehavior-budget sweep.

A clean, discrete-action off-policy scheme for red: **DQN-style Q-learning** on the joint red action, with a preallocated replay buffer, twin-Q, and a Polyak target — i.e. the same machinery as EXP-A, applied to the adversary head. We're not porting SAC's continuous-action squashed Gaussian.

### Variants (2 conditions — 1 cell + 1 control)

| code | red trainer | sample reuse |
|------|-------------|--------------|
| **B0** | on-policy REINFORCE (existing `coevo_r6.py`) | 1× |
| **B1** | **Off-policy double-DQN** with replay buffer (50 k capacity, batch 256), twin-Q target, τ=0.005 | ~50× |

Blue stays **on-policy** in both variants — we're only swapping the red trainer. Blue training config is frozen so any coverage delta is attributable to the red change.

### Metrics

Both metrics measured at **matched env-step budget** (not matched generation count) so we're comparing sample efficiency fairly.

- **M5 — Blue coverage (mean %, 20 eval seeds)** after X total red env-steps, for X ∈ {100 k, 500 k, 2 M}. **Primary.** Lower is better — this is ΔJ; a more-effective red drives blue down.
- **M6 — Red coverage** (red-team own coverage). Smoke-test that red is actually learning something, not collapsing to STAY.
- **M7 — Wall-clock** to reach ΔJ ≥ 8 pp at N = 5 (roughly the published r6 number scaled to compromise-sweep terms).

### Pass criteria

- **B1 reaches B0's 500 k-step ΔJ in ≤ 200 k steps** (~2.5× sample efficiency). If not, off-policy isn't buying us enough for the complexity.
- **B1 final ΔJ ≥ B0 final ΔJ − 1.0 pp.** Asymptotic performance must not regress materially.
- **M6 ≥ 40%.** Red must still be doing *something*, not just obstructing — otherwise our "uncertainty-manipulation" channel collapses and the sabotage story changes.

### What each comparison tells us

| comparison | claim | evidence for / against |
|---|---|---|
| B1-500k vs B0-500k | "Off-policy is faster" | If B1 reaches deeper ΔJ at the same step budget: sample efficiency is real. |
| B1-2M vs B0-2M | "Off-policy doesn't cap lower" | If B1 plateaus above B0: no regret. If B1 plateaus below B0: off-policy bias from stale transitions is hurting. |
| B1-wall-clock vs B0-wall-clock | "Off-policy is cheaper in practice" | Dominates dense ε-sweep feasibility. |

### Wall-clock estimate

- **B0 re-run (baseline refresh):** 1 × existing `scripts/coevo.py` budget ≈ 45 min.
- **B1 prototype:** 2 M env-steps @ ~5 k steps/s on CPU ≈ 7 min compute; ~1 day implementation time for the new replay-DQN red trainer (`scripts/coevo_offpolicy.py`).

Total EXP-B wall-clock ≈ 1 day + 1 hour of runs.

### Rollback criterion

If B1 fails M5 at any horizon, the on-policy REINFORCE path is the right one for now — keep it, mark the off-policy attempt in the meta-report as "attempted, not adopted", and don't re-attempt until we want a 10× sweep density (e.g. a k × ρ × ε three-way grid for the paper).

---

## Decision tree — what to actually run first

```
START
  │
  ▼
Run EXP-A (≈ 80 min, no new code beyond ~20 lines in losses.py).
  │
  ├── A3 passes all criteria  →  adopt as default, keep grad_clip on, close the bubbly-strolling-puddle plan.
  │                              Then decide whether EXP-B is worth the implementation cost.
  │
  ├── A3 fails M1 or M3       →  go to Stage 3 (MC returns) in the standing plan, skip EXP-B.
  │                              The critic doesn't need more stabilisers — it needs a fundamentally
  │                              different target source.
  │
  └── A3 passes but A1 ≈ A3   →  adopt A1 (target-net only) — cheaper, same result.
                                 Twin-Q was not needed.
  │
  ▼
If EXP-A lands, decide on EXP-B:
  │
  ├── Planning ε × ρ × k three-way sweep for the paper?  →  Run EXP-B. The 10× step budget
  │                                                         only becomes tractable with replay.
  │
  └── Current sweeps (k×ρ, ε-only) are sufficient?       →  Skip EXP-B. On-policy REINFORCE is
                                                            fine; the implementation cost
                                                            doesn't pay for itself yet.
```

---

## Artifacts

After running, each experiment produces:

- **EXP-A** — `experiments/pair-cooperate-coop-{A0,A1,A2,A3}/metrics.npz`, `report.html`. A cross-variant figure `experiments/stabilisation/reward_vs_variant.png` (4 curves, shaded σ bands) goes into the meta-report as §6.6.
- **EXP-B** — `experiments/compromise-16x16-5-3b2r-offpolicy/metrics.npz` and a step-budget sweep `experiments/stabilisation/sample_efficiency.png` (two lines, blue coverage vs env-steps, matched x-axis). Goes into the meta-report as §6.7.

Both experiments update `experiments/README.md` with a "How to reproduce" subsection per feedback memory `feedback_experiments_readme.md`.

---

## What we compare — one-paragraph summary for each stakeholder

**For the "is training stable" question (EXP-A):** compare **A0 vs A3** on M1 and M3. One headline number: final reward mean ± σ.

**For the "is coevo fast enough" question (EXP-B):** compare **B0 vs B1 at the same env-step budget** on M5. One headline number: blue coverage at 500 k red env-steps.

**For the paper:** the four-way table from EXP-A (A0/A1/A2/A3) is the single most-informative ablation — it isolates target-nets vs twin-Q vs their combination. EXP-B's contribution is a sample-efficiency curve (B0 vs B1 on a single axis).

---

## Results (run 2026-04-20)

Both experiments produced clean negative results. See `experiments/stabilization/stabilization_report.html` for the full writeup + figures.

### EXP-A — MC baseline dominates

| code | critic target | final reward (mean ± σ, 5 seeds) | late dives | `\|loss\|` p99 final | verdict |
|---|---|---|---|---|---|
| **A0** | Monte-Carlo (production) | **+2.374 ± 0.026** | **1/5** | **11.7** | ✅ pass |
| A1 | TD(0) + target-net | +0.393 ± 0.024 | 4/5 | 2 894 | ❌ |
| A2 | TD(0) + twin-Q live | +0.382 ± 0.029 | 5/5 | 8 459 | ❌ |
| A3 | TD(0) + twin-Q + target-net | +0.360 ± 0.005 | 5/5 | 13 780 | ❌ |

The four curves are identical through ep 1 000 (all at +2.0 to +2.3). Between ep 1k and ep 10k, A1/A2/A3 each collapse to ~+0.4 as the TD(0) bootstrap self-amplifies; A0 holds +2.35 through 15 000 eps. `|loss|` p99 growth is *worse* as you stack stabilisers (A1: 2 894 → A2: 8 459 → A3: 13 780). Adding more bootstrap targets gives the critic more ways to self-amplify noise from the non-stationary `global_seen_mask`.

### EXP-B — On-policy wins

| code | red trainer | final blue reward (mean, 3 seeds) | wall-clock |
|---|---|---|---|
| **B0** | on-policy REINFORCE | **+1.08** | **133 s** |
| B1 | off-policy Double-DQN + replay + twin-Q + Polyak | +1.22 | 2 650 s |

Lower blue reward = stronger red. B1 leaves blue *higher* than B0 (weaker red) and takes 20× the wall-clock. No env-step horizon in [0, 500 k] has B1 materially below B0.

Two likely mechanisms: (a) additive factorised Q can't represent joint-red coordination that a shared softmax captures; (b) with blue frozen the environment is stationary, so replay's value proposition vanishes and the compute cost is pure overhead.

### Decisions landed

- **Close `bubbly-strolling-puddle` plan.** Monte-Carlo returns at `src/red_within_blue/training/losses.py:222` are empirically optimal — not just adequate. No further stage.
- **Keep `grad_clip: 0.5`.** Redundant given MC but cheap.
- **Do not port SAC.** Both halves of its critic toolkit (target-net, twin-Q) are counterproductive here.
- **Keep on-policy REINFORCE for joint red** (`scripts/coevo_r6.py` path). Off-policy replay is a net-negative tradeoff on this problem at this scale.
- **The "10× sweep density via replay" premise from EXP-B is dead.** Any future ε × ρ × k sweep must be budgeted on on-policy throughput, not replay.
