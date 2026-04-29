# Cooperative-Exploration — Operator's Manual

Guide for the cooperative-exploration experiments:
- **`configs/pair-cooperate-coop.yaml`** — 2 agents, 10×10 grid. The reference experiment; all tuning decisions trace back to this config. Documented in detail below.
- **`configs/quad-cooperate-coop.yaml`** — 4 agents, 14×14 grid. Scaled-up variant (same per-agent cell density). See the "Quad scale-up" section.

**Warm-start ladder for bigger grids.** Rather than training 32×32 from scratch, we build it on top of smaller grids via the grid-aware warm-start loader (actor + central critic):

- `configs/quad-cooperate-coop-8.yaml` — 4 agents, 8×8, train from scratch (~1 min).
- `configs/quad-cooperate-coop-16.yaml` — 4 agents, 16×16, warm-start from the 8×8 checkpoint (~2 min, gentle fine-tune).
- `configs/quad-cooperate-coop-32.yaml` — 4 agents, 32×32, warm-start from the 16×16 checkpoint (~5–15 min, gentle fine-tune).
- `configs/octa-cooperate-coop-32-r6.yaml` — 8 agents, 32×32, comm_radius=6, warm-start from the N=4 32×32 checkpoint via **per-agent block tiling** (N=4 → N=8). The first **N-mismatched** transfer in the ladder. Warm-start dominates the fine-tune at this regime — see the "8 agents on 32×32 (N-mismatched warm-start)" section.
- `configs/octa-cooperate-coop-32-r6-nocoop.yaml` — same N=8 setup with `cooperative_weight: 0.0` ablation. Tests whether the cluster-tethering bias is reward-pinned. Result: yes — coverage rose 41.9% → 55.9% on the raw warm-start, but training fragmented the team (`connected_frac` 1.000 → 0.269) because the only cohesion signal was removed.
- `configs/octa-cooperate-coop-32-r6-conn.yaml` — restoration variant with `cooperative_weight: 0.005` (1/3 of the original). Brings connectivity back to **1.000** without re-collapsing coverage to the cluster-bias floor. The cleanest "100% connectivity, near-zero coop" point on the tradeoff curve. See the "Cohesion ↔ coverage tradeoff at N=8" section.

**Survey-local prototype** (`configs/survey-local-16-N7.yaml`, 2026-04-19 — new). First experiment on the "split sensing vs per-cell survey" design. Decouples what the agent SEES (a 3×3 terrain window) from what it DOES (survey only the cell it is on), and drops the global H·W seen mask from the observation in favour of a 3×3 local-memory window. obs_dim shrinks from 256 + 9 + 5 = 270 at 16×16 to **23** — grid-invariant, constant regardless of map size. Diagnoses the "STAY is too good because the team already knows everything" failure mode from the `-norm` ladder: with survey_radius=0 and local_obs=true, coverage requires actual motion, and the policy has no global memory to short-circuit movement decisions. Fresh train (no warm-start — existing checkpoints have the global seen field baked into `Dense_0`). See the "Survey-local prototype" section for config details and running instructions.

**Survey-local warm-start ladder** (`configs/survey-local-{8-N1,8-N2,16-N4,32-N8}.yaml`, 2026-04-19 — new). Four-rung ladder that scales the survey-local design from N=1 on 8×8 up to N=8 on 32×32 by warm-starting each rung from the previous. Because `local_obs=true` makes obs_dim grid-invariant, the runner's fast-path just tiles the source Dense_0 kernel per target-agent block — no grid-row upsampling. Ladder converges in ~2.5 min total wall-clock; last rung lands at target competence from step 1. See the "Survey-local warm-start ladder" section.

**uid-normalised ladder** (`-norm` suffix, built 2026-04-18). Mirror of the chain above with `env.normalize_uid: true` so the actor sees `uid / num_agents ∈ (0, 1]` at every rung — fixes the per-seed geographic bias that appeared at the N=4 → N=8 transfer (ReLU extrapolation on raw uid ∈ {1..4} → {1..8}). Same network shape, LR schedule, reward weights, and episode counts as the raw-uid siblings; only the obs tail changes:
- `configs/quad-cooperate-coop-8-norm.yaml` — restart point (from scratch).
- `configs/quad-cooperate-coop-16-norm.yaml` — warm-start from `-8-norm`.
- `configs/quad-cooperate-coop-32-norm.yaml` — warm-start from `-16-norm`.
- `configs/octa-cooperate-coop-32-r6-norm.yaml` — N-mismatched warm-start from `-32-norm` (4-block tile). See the "uid-normalised ladder" section for per-rung final rewards.

Each rung does 4× more grid cells than the one below (or 2× more agents at constant grid). See the "32×32 scale-up", "8×8 → 16×16 warm-start smoke", and "8 agents on 32×32" sections for the chain's lessons (critic-must-transfer, gentle-fine-tune, run-the-pre-training-diagnostic-first, mismatched-N-uses-tile-and-can-skip-fine-tune).

All configs use **CTDE actor-critic** (shared decentralized policy + centralized value on the joint observation) with entropy regularization. Follow the steps in order from the repo root.

---

## What this experiment tests

Two agents must divide a 10×10 map while staying within scan-sharing range. The reward mix is tuned so that coverage from *different* cells dominates — overlapping cells contribute nothing beyond the cooperative bonus.

**Fixes vs. the previous 5000-ep run (which collapsed to −0.39), the first 15000-ep CTDE run (which collapsed to −0.34), the second 15000-ep CTDE run (which collapsed to +0.35 via runaway central critic), the third 15000-ep CTDE run with `grad_clip: 0.5` alone (which still collapsed to +0.49), and the fourth 15000-ep CTDE+MC run (stable at +2.57 stochastic but argmax degraded to +1.11 because the policy never consolidated toward a deterministic fixed point):**

- **Monte-Carlo critic target** — the central critic is now trained against full-episode MC returns (`Σ γ^k r_{t+k}`) instead of a TD(0) bootstrap (`r + γ V(s')`). TD(0) was self-referential with a 100-dim non-stationary `global_seen_mask` in the joint obs — the critic chased its own noisy predictions and diverged even with `grad_clip` damping the gradient norm. MC returns are a pure trajectory regression target: the critic's loss does not depend on its own predictions, so the feedback loop that caused every prior run to collapse cannot form. Episode length is 100, so MC variance is manageable.

- **Raw (un-normalised) advantages** — a 3000-ep A/B found that adding unit-variance advantage normalisation on top of MC *re-introduced* the collapse (reward +2.2 → +0.87). The critic is still noisy in early training; normalising the advantages to unit-magnitude amplifies that noise into unit-magnitude policy-gradient pushes. Keeping raw advantages preserves the "small when critic is uncertain" property, which lets the policy learn slowly while the critic converges.

- **Gradient clipping** — `grad_clip: 0.5` global-norm clip on both actor and critic Adam updates, retained as a defensive belt-and-suspenders layer on top of MC. The second CTDE run's failure mode was textbook value-function divergence: reward fell to +0.35 by ep 1500, then `|loss|` climbed from 0.08 → 196. Clipping at 0.5 alone slowed that divergence but did not prevent it (reward still fell to +0.49 by ep 15000); MC cures the root cause.

- **CTDE (centralized-training, decentralized-execution)** — one shared `Actor` runs per-agent at execution time; one **central `Critic`** takes the *joint* observation `[N × obs_dim]` and emits a team value `V(s)`. All agents update against the **same shared advantage**, so they stop fighting over a cooperative reward signal.
- **Stronger exploration pressure** — `ent_coef: 0.05` (up from 0.01) keeps π stochastic well past the point the previous run determinised, and `epsilon: 0.05` coin-flip exploration overrides the actor's sample ~5% of steps as a cheap backup when the entropy bonus alone isn't enough.

- **Annealed exploration schedule** — `epsilon` and `ent_coef` linearly decay 0.05 → 0.005 over the first **50 %** of training (`anneal_end_frac: 0.5`), then hold at the low floor for the remaining 7500 episodes. This gives the policy a long consolidation tail at near-zero entropy pressure so π can sharpen toward a reliable mode. Without annealing the previous MC run stayed stochastically good (+2.57) but deteriorated badly under argmax (+1.11). Annealing over the full 15000 eps was too slow and the policy never reached the low-entropy regime (argmax +0.84); finishing the anneal at 50 % struck the right balance (argmax +1.13 but — critically — see the eval-protocol note below).

- **Eval protocol: sample from π with ε = 0** — the correct deterministic measure of policy quality on this cooperative task is *not* argmax. An ε-sweep over the final checkpoint (200 episodes × 5 seeds × 8 configs in `/tmp/eval_epsilon_sweep.py`) shows argmax yields +1.13 with a 46 % failure rate, while pure π-sampling (no ε override) yields **+2.88 with 0.0 % catastrophic failures**. The policy has learned a genuinely stochastic cooperative strategy: symmetry-breaking at spawn requires a random tie-break between "go left / go right", and collapsing that to argmax forces both agents to make the same deterministic choice. The −0.5375 failure tail observed in *training* metrics is an artifact of the ε = 0.05 random-action override during rollouts — it vanishes as soon as ε = 0 at eval time.
- **Wider/deeper networks** — `256 × 3` for both actor and central critic (up from `128 × 2`). The central critic ingests `[N × obs_dim] = 212` features; the old trunk was thin.
- **Softer revisit penalty** — `revisit_weight: -0.001` (down from `-0.003`). With `obs_radius: 0` agents can't see where they've been, so the old magnitude was punishing ignorance.
- **Sharper terminal gradient** — `terminal_bonus_scale: 2.0` (up from 1.0). Full coverage → each agent gets +1.0 at episode end, dominating late-training noise.
- **`obs_radius: 0`** — each agent sees only its own cell. Forces coordination via messages, not local vision.
- **`comm_radius: 5.0`** — comfortable overlap on a 10×10 grid; no blunt disconnect gate needed.
- **`center_spawn: true`** — agents start clustered near the grid center (Gaussian, σ = comm_radius/2). Always connected at t=0.
- **`num_episodes: 15000`** — 3× the original run. Long enough to see whether entropy+ε-regularised CTDE stabilises at a plateau or still diverges.
- **Live progress bar** — the runner prints a tqdm progress bar with current episode, ETA, live loss and live reward (falls back to one status line per ~0.5 % progress when stdout isn't a TTY).
- **`jax.block_until_ready`** in the runner — elapsed time reflects real compute, not dispatch latency.

---

## Run it

```bash
# 1. Unit tests first — confirm loss signature + trainer wiring still clean
python -m pytest tests/test_pg.py -q

# 2. Train (15000 eps × 5 seeds, CTDE actor-critic, ent_coef=0.05, epsilon=0.05, nets 256x3)
python -m red_within_blue.training.runner --config configs/pair-cooperate-coop.yaml

# 3. Render HTML report + evaluation GIF
python -m red_within_blue.analysis.experiment_report \
    --config configs/pair-cooperate-coop.yaml \
    --experiment-dir experiments/pair-cooperate-coop
open experiments/pair-cooperate-coop/report.html
```

Expected wall-clock (Mac M-series, 5 seeds vmapped): **~15–25 min.** Watch for the timer now reporting realistic elapsed seconds — if it prints <5s again, `block_until_ready` regressed.

**What you'll see during training:**

```
============================================================
 Experiment : pair-cooperate-coop
 Config     : configs/pair-cooperate-coop.yaml
 Method     : actor_critic
 Env        : 10x10, N=2, steps=100
 Train      : 15000 eps x 5 seed(s), lr=0.00015, gamma=0.99
============================================================

pair-cooperate-coop (5 seeds):  34%|████████▎       | 5125/15000 [02:07<04:06, 40.11ep/s, loss=-0.0113, reward=+1.651]
```

The bar updates ~200 times over the run (host-callback overhead is <1 % of total time). `loss` and `reward` are means across seeds.

---

## Reading the output

The runner prints one summary block at the end:

```
  Final loss:   <x.xxxx> +/- <y.yyyy>
  Final reward: <x.xx>   +/- <y.yy>
```

**Pass criteria:**

Max theoretical episodic return at full coverage on 10×10 (both agents):
`2 × (self-discovery sum) + 2 × (terminal bonus per agent) ≈ 2 × 1.0 + 2 × 1.0 = +4.0` (before revisit/isolation debits).
Realistic plateau: **+1.8 to +2.5** team reward.

| Metric | Pass | Borderline | Fail |
|---|---|---|---|
| Final reward (mean across seeds) | ≥ +1.80 | +0.80 to +1.80 | ≤ +0.80 |
| Final reward seed stddev | ≤ 0.30 | 0.30 – 0.60 | ≥ 0.60 |
| Loss stddev | ≤ 1.0 | 1.0 – 3.0 | ≥ 3.0 (critic diverging) |
| Reward trajectory | monotone-ish rise to plateau | noisy plateau near max | descends late (collapse) |

Open the HTML report: the learning curve is the truth. A healthy run climbs to a plateau around ep 6000–10000 and stays there. A collapse looks like a rise followed by a dive — that's the entropy coefficient being too low or the lr too high for the tail of training.

### Policy-quality eval (post-training)

Training-time reward is measured with `ε = 0.05` rollouts, so the reported number is *π + noise floor*, not π itself. To measure the learned policy directly:

```bash
python /tmp/eval_epsilon_sweep.py       # 8 configs × 200 eps × 5 seeds, argmax vs sample-from-π
python /tmp/eval_deterministic.py       # per-seed argmax-only breakdown
```

**Headline numbers for the current checkpoint:**

| Eval mode | Mean reward | % episodes < +1.0 |
|---|---|---|
| argmax, ε = 0 | +1.13 | 46 % |
| argmax, ε = 0.005 | +2.65 | 11 % |
| **sample from π, ε = 0** | **+2.88** | **0.0 %** |
| sample from π, ε = 0.005 | +2.83 | 1 % |
| sample from π, ε = 0.05 (training-time) | +2.57 | 7 % |

**Report the `sample from π, ε = 0` row as the policy's true performance.** Argmax is the wrong metric on this task because the cooperative policy relies on spawn-time symmetry-breaking (both agents use the same shared π; only the random tie-break between equally-valued first moves decides who goes which way). Argmax forces both agents to pick the same cell.

The recurring `−0.5375` episodes in the training log are not a policy defect — they are the ε = 0.05 override driving one agent into a wall/away from the team during a critical early step. Under ε = 0 the failure mode vanishes.

---

## If it collapses again

Try in this order — change ONE knob at a time. **Check `|loss|` trajectory first**: if `|loss|` p99 explodes (>1000) while reward drops, that's critic divergence → tighten clip. If `|loss|` stays small but reward still falls, that's an exploration problem → raise entropy/epsilon.

1. **Tighten `grad_clip`** from `0.5` → `0.25`. Defensive damping if the critic still drifts; the MC target should already prevent runaway divergence but clip is cheap.
2. **Raise `ent_coef`** from `0.05` → `0.08`. Most non-critic collapses are exploration-starvation. With the MC critic in place, exploration is now the most likely failure mode.
3. **Raise `epsilon`** from `0.05` → `0.10`. Extra random-action noise floor; still on-policy-ish at this level.
4. **Lower `lr`** from `1.5e-4` → `5e-5`. Longer training needs a smaller step size.
5. **Raise `num_episodes`** to 25000 if the curve still climbs at ep 15000 — under-training is also a failure mode.
6. **Check per-seed reward trajectories** in the report. If 4/5 seeds converge and 1 collapses, increase `num_seeds` to 8 for reporting; don't over-tune for the worst seed.
7. **Do not re-add advantage normalisation.** A 3000-ep A/B already showed that `(adv - mean) / (std + ε)` on top of the MC critic re-creates the collapse because it amplifies critic noise. The existing raw-advantage path is intentional.
8. **Do not interpret a low argmax reward as training failure.** Check `sample from π, ε = 0` (see the eval-protocol section above) before concluding the run collapsed. For this cooperative task argmax and the true policy can differ by +1.75 reward.

Do **not** touch `comm_radius`, `obs_radius`, or `center_spawn` — those are the experiment's independent variables.

---

## Activation-function A/B (decided, keep ReLU)

A 3000-ep × 2-seed smoke (`/tmp/smoke_activations.py`) trained the full pair-cooperate-coop config from scratch for each of `{relu, gelu, tanh, silu}`:

| Activation | ep 0–500 r | ep 1000–1500 r | ep 2500–3000 r | final \|loss\| p99 |
|---|---|---|---|---|
| **relu** *(default)* | +2.201 | +2.638 | **+2.816** | 33.3 |
| gelu | +2.185 | +2.578 | +2.739 | **25.6** |
| silu | +2.189 | +2.576 | +2.691 | 31.5 |
| tanh | **+2.261** | +2.707 | +2.674 | 41.8 |

**Verdict: keep ReLU.** ReLU wins the final-reward ranking; the spread across activations is ~0.14 reward — inside seed-to-seed noise at n=2. None collapsed, all trained cleanly. GELU had the lowest critic-loss variance, so it's a reasonable fallback if a future run shows critic-divergence symptoms. Tanh started fastest but saturated earliest — consistent with its bounded range limiting late-training policy sharpening on this task.

The knob is preserved in config (`network.activation: relu`) so re-running the sweep is one-line. Do not change the default without a ≥ 5-seed smoke showing a ≥ 0.3-reward win.

---

## Reward knobs (read-only for this run)

All set in `configs/pair-cooperate-coop.yaml:reward`. Scale reference: self-discovery = +0.01 per new cell on a 10×10 grid.

| Knob | Value | What it does |
|---|---|---|
| `disconnect_penalty` | 0.0 | Off — center_spawn keeps the team connected at t=0. |
| `isolation_weight` | −0.002 | Per-step penalty when an agent's degree in the comm graph is 0. |
| `cooperative_weight` | 0.015 | Bonus to a connected neighbour when the other finds a new cell (1.5× self-discovery). |
| `revisit_weight` | −0.001 | Penalty for stepping on an already-explored cell (~10% of discovery magnitude). Softened because `obs_radius=0` makes revisits partly unavoidable. |
| `terminal_bonus_scale` | 2.0 | End-of-episode bonus = scale × coverage_fraction, split evenly across agents. Full coverage → each agent gets +1.0 (N=2). |
| `terminal_bonus_divide` | true | Split the terminal bonus across agents (true) or give it to all (false). |

### Training knobs (also read-only for this run)

Set in `configs/pair-cooperate-coop.yaml:train` and `:network`.

| Knob | Value | What it does |
|---|---|---|
| `method` | actor_critic | CTDE: shared Actor, central Critic on joint obs. Critic target is full-episode Monte-Carlo returns (not TD(0)); advantages are used raw (no unit-variance normalisation). |
| `lr` | 1.5e-4 | Adam learning rate (shared by actor and critic). |
| `gamma` | 0.99 | Return discount. Applied inside `compute_discounted_returns` when building the MC critic target. |
| `num_episodes` | 15000 | Episodes per seed. |
| `num_seeds` | 5 | Parallel vmapped training runs. Final metric is mean ± stddev across seeds. |
| `ent_coef` | 0.05 | Initial entropy bonus coefficient. Higher → stochastic policy holds longer. Annealed (see `ent_coef_final`). |
| `ent_coef_final` | 0.005 | Entropy coefficient at the end of the anneal window. Linearly interpolated from `ent_coef` over `anneal_end_frac × num_episodes`, then held. Set `< 0` to disable annealing. |
| `epsilon` | 0.05 | Initial per-step probability each agent's action is replaced with uniform random. Safety net on top of entropy. Annealed (see `epsilon_final`). |
| `epsilon_final` | 0.005 | Epsilon at the end of the anneal window. Linearly interpolated from `epsilon` over `anneal_end_frac × num_episodes`. Set `< 0` to disable annealing. |
| `anneal_end_frac` | 0.5 | Fraction of training over which `epsilon` and `ent_coef` complete their linear decay. `0.5` means the anneal finishes at ep 7500; the remaining 7500 eps train at the final (low) values so the policy can consolidate. `1.0` = anneal across all episodes (too slow for this task — policy never consolidates). |
| `grad_clip` | 0.5 | Global-norm gradient clip on both actor and critic updates. Secondary defence against critic divergence; the MC target is the primary fix. |
| `actor_hidden_dim / actor_num_layers` | 256 / 3 | Shared actor MLP. |
| `critic_hidden_dim / critic_num_layers` | 256 / 3 | Central critic MLP (input dim = `N × obs_dim` = 212). |
| `activation` | relu | Hidden-layer activation (applies to both actor and critic). Accepts `relu|gelu|tanh|silu`. See the "Activation-function A/B" section above for the empirical ranking. |

---

## Files produced

```
experiments/pair-cooperate-coop/
├── checkpoint.npz          # actor + critic params
├── metrics.npz             # per-episode loss, total_reward, coverage, per-agent splits
├── report.html             # learning-curve plots + interactive GIF player
└── episode.gif             # one evaluation rollout at 4 fps
```

The GIF is the fastest sanity check: if both agents wander toward the same cells, the cooperative reward isn't biting hard enough.

### Preserved A/B artifacts

Three adjacent directories preserve the anneal ablation so results are reproducible without retraining:

| Directory | Anneal | Training reward | argmax | sample π (ε=0) |
|---|---|---|---|---|
| `pair-cooperate-coop.baseline-noanneal/` | off (constant ε=0.05, ent=0.05) | +2.57 | +1.11 | ~+2.6 |
| `pair-cooperate-coop.anneal-full/` | `anneal_end_frac: 1.0` | +2.81 | +0.84 | — |
| `pair-cooperate-coop/` *(current)* | `anneal_end_frac: 0.5` | +2.89 | +1.13 | **+2.88** |

These are for regression checking only — point the report tool at any of them to inspect the corresponding learning curves.

---

## Bug-hunt starter tests

Run these *before* retraining if the training output looks wrong — they isolate the loss plumbing from the training dynamics:

```bash
python -m pytest tests/test_pg.py tests/test_rewards_training.py -q
python -m pytest tests/test_env.py -q
```

All three suites must be green before you trust a collapsed training run as "real."

---

## Quad scale-up (`configs/quad-cooperate-coop.yaml`)

Same algorithm, more agents and a bigger grid. Tests whether the CTDE AC + MC critic + annealed-ε recipe transfers from `N=2, 10×10` to `N=4, 14×14` — roughly a 4× increase in critic input dim and 2× increase in total cells.

**Scaling rules applied (relative to the pair config):**

| Knob | Pair | Quad | Why |
|---|---|---|---|
| `grid_width/height` | 10 | 14 | Keep per-agent density: pair had 50 cells/agent (100/2), quad has 49 cells/agent (196/4). |
| `num_agents` | 2 | 4 | The scale-up. |
| `max_steps` | 100 | 150 | 50% more steps for a 2× bigger grid. |
| `comm_radius` | 5.0 | 7.0 | Keeps `radius/width = 0.5` constant — same fractional coverage of the comm graph. |
| `terminal_bonus_scale` | 2.0 | 4.0 | With `divide=true`, per-agent full-coverage terminal = `scale / N`. Scaling up 2× keeps it at +1.0/agent. |
| `grad_clip` | 0.5 | **0.25** | Pair's `0.5` failed at quad scale: 3000-ep smoke had `\|loss\|` p99 8.6 → 218 and reward fell +7.6 → +4.9. Tighter clip stopped the runaway. The central critic now consumes `4 × obs_dim = 808` features (vs 212 for the pair config), so the same 256×3 trunk is easier to destabilise. |

Every other knob (lr, ent_coef, epsilon, annealing, activation, network width/depth, reward weights, gamma) is identical to the pair config.

### Run it

```bash
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop.yaml
python -m red_within_blue.analysis.experiment_report \
    --config configs/quad-cooperate-coop.yaml \
    --experiment-dir experiments/quad-cooperate-coop
open experiments/quad-cooperate-coop/report.html
```

Wall-clock (Mac M-series, 5 seeds vmapped, 15000 eps): **~13–15 min.**

### Current results (2026-04-18)

15000 eps × 5 seeds, `grad_clip: 0.25`. Evaluated with the same argmax-vs-sample sweep used for the pair config (`/tmp/eval_quad.py`, 200 eps × 5 seeds × 5 configs):

| Eval mode | Mean | p5 | median | p95 | % < +1.0 |
|---|---|---|---|---|---|
| argmax, ε = 0 | +3.21 | +1.05 | +3.00 | +5.97 | 4.4 % |
| **sample from π, ε = 0** | **+6.53** | +1.50 | **+7.38** | **+9.28** | **0.8 %** |
| sample from π, ε = 0.005 | +6.53 | +1.52 | +7.37 | +9.31 | 0.7 % |

**Per-seed (sample from π, ε = 0):**

| Seed | Mean | p5 | median | p95 |
|---|---|---|---|---|
| 0 | +7.31 | +5.42 | +7.43 | +8.80 |
| 1 | **+1.85** | +1.05 | +1.85 | +2.77 |
| 2 | +7.99 | +6.50 | +8.04 | +8.95 |
| 3 | +6.77 | +5.73 | +6.89 | +7.66 |
| 4 | +8.71 | +7.39 | +8.78 | +9.74 |

**Interpretation.** 4/5 seeds converged to a strong cooperative policy (+6.77 to +8.71, near the ~+10 realistic max). Seed 1 is the outlier — it got stuck at a degenerate plateau and never found the cooperative mode. The `+5.71 ± 2.88` training-time mean is dominated by this one seed plus the general training-time ε-noise. Pair config had a similar risk and it didn't trigger there; at quad scale the seed-variance is structurally higher because the critic's landscape is 4× larger.

**Same argmax lesson as the pair config:** sample-from-π is the correct eval mode (+6.53 vs argmax +3.21). The symmetry-breaking story is stronger here — 4 agents need to pick 4 different directions at spawn.

### If you want a fair headline number

Drop seed 1 from the mean (4/5 converged seeds): **mean +7.69, no failure outliers.** Or retrain with `num_seeds: 8` and report the aggregate — one structurally bad seed is roughly what you'd expect from any on-policy AC at this scale.

---

## 32×32 scale-up (ladder warm-start: 8 → 16 → 32)

Same algorithm, much bigger grid: 4 agents on 32×32 (**1024 cells** vs quad's 196). Warm-starts **both the actor and the central critic** from the **16×16 checkpoint** (which was itself warm-started from the 8×8 checkpoint) via the grid-aware loader. The cooperative policy *and* the learned V-function shape are carried across the grid-size jump by a spatial upsample of each network's first layer — the full chain is `8×8 (scratch) → 16×16 (warm + gentle fine-tune) → 32×32 (warm + gentle fine-tune)`.

**Why the 16×16 source instead of the 14×14 quad:** the 16×16 run was trained specifically as the intermediate rung of this ladder, ending at sample-π **+8.59** with all 5 seeds in [+8.03, +8.91] (tighter than the 14×14's +6.53 with one structurally bad seed). 16 → 32 is also a clean 2× linear / 4× cell scale-up — identical to the 8 → 16 smoke that validated the pipeline — so the upsample's aliasing behavior is predictable.

### Key lessons from the journey (read this before any future scale-up)

1. **Warm-start the critic too, or the run will collapse.** The first 32×32 attempt transferred only the actor and re-initialised the 4120-wide critic: reward **+12.9 → +0.55** over 25k eps while `|loss|` p99 grew **1.4k → 2.2M**. Classic central-critic runaway (sample-from-π = argmax, ε=0.05 random noise beat the learned policy). Archived at `experiments/quad-cooperate-coop-32-actor-only-baseline/`. The loader now supports `num_blocks=N` (see below) so the critic transfers alongside the actor.

2. **Off a converged warm-start, *gentle fine-tune*.** The second 32×32 attempt (14 → 32, actor + critic transferred) still collapsed because it re-used the source's `lr=1.5e-4, num_episodes=25000` — a recipe calibrated for from-scratch training. The 8 → 16 smoke pinned the diagnosis exactly: the raw upsampled actor scored **+7.72** on 16×16 with zero gradient steps, but 8000 eps of fine-tuning at lr=1.5e-4 drove it down to **+1.68** (agents walked to a corner and stopped). Dropping to `lr=3.0e-5, num_episodes=2000` lifted that same run to **+8.59**. For converged sources the fine-tune's job is *adaptation*, not *re-learning* — start with lr ~5× smaller and episodes ~4× shorter and escalate only if you plateau below the warm-start.

3. **Always run the pre-training diagnostic first.** `/tmp/eval_pretraining_warmstart.py <config.yaml>` loads + upsamples the source, evaluates each seed on the target grid with zero gradient steps, and records a GIF. It pins the diagnosis of any later collapse: if the raw warm-start is strong, the training recipe is the destroyer; if the raw warm-start is weak, the upsample itself isn't transferring. The 16 → 32 diagnostic showed sample-π **+13.98, coverage 50.7%** with zero gradient steps — a full 2× the 14×14 headline before any training. This is what told us the mechanism scales cleanly through 16 → 32, and that the previous "collapse" stories were all about training recipe rather than transfer.

### Why the upsample works

`obs_dim = scan(S) + grid_seen_mask(H·W) + tail(5)`. Only the grid-mask portion depends on grid shape — scan and tail are grid-invariant scalars, and every Dense layer past `Dense_0` operates on `[hidden, hidden]` (grid-invariant). So the only part of either network that needs transforming is `Dense_0`'s kernel along its input axis.

Transfer mechanics (in `runner.py:_upsample_first_layer_for_grid`):

**Actor (`num_blocks=1`):**
1. Split `Dense_0/kernel` along its input axis into `scan(1) | grid(256) | tail(5)` rows (numbers shown for a 16×16 source).
2. Copy scan and tail rows verbatim (they encode grid-invariant features: own-cell scan, map-fraction, normalised pos, uid, team).
3. Reshape grid rows `[256, hidden] → [16, 16, hidden]`, **nearest-neighbor** upsample to `[32, 32, hidden]` (binary seen_mask → nearest is the principled choice), flatten back to `[1024, hidden]`.
4. Concatenate → new `Dense_0/kernel: [1030, hidden]`. Bias `[hidden]` is grid-invariant, copies as-is.
5. Deeper layers (`Dense_1..N`) copy verbatim — all `[hidden, hidden]` or `[hidden, num_actions]`.

**Central critic (`num_blocks=N`):** the CTDE critic consumes `joint_obs = observations.reshape(T, -1)` — N concatenated copies of the per-agent obs (see `losses.py:actor_critic_loss_ctde:219`). So its `Dense_0/kernel` has input shape `[N × per_block_obs_dim, hidden]`. The loader splits the input axis into `N=4` per-agent blocks, runs steps 1–4 on each block independently, then concatenates the blocks back along the input axis. Quad-16's `[1048, 256]` kernel → quad-32's `[4120, 256]` kernel, seed-axis preserved.

**What must match for critic transfer:** `num_agents`, `critic_hidden_dim`, `critic_num_layers`. If any differ between source and target, the loader prints a clear message and falls back to critic re-init rather than raising. (This is why the target config keeps `critic_hidden_dim: 256`, not 512.)

The loader auto-detects grid and input-width mismatches and no-ops when both already match, so existing warm-start configs (e.g., `configs/sweeps/*.yaml` off `solo-explore`) are unaffected.

### Config

`configs/quad-cooperate-coop-32.yaml`:

| Knob | Quad-16 (16×16) | Quad-32 (32×32) | Why |
|---|---|---|---|
| `grid_width/height` | 16 | 32 | 2× linear / 4× cells. |
| `max_steps` | 150 | 400 | ~Cell-ratio scaled; expect partial coverage at 400 steps with 4 agents. |
| `comm_radius` | 8.0 | 16.0 | Keeps `radius/width = 0.5`. |
| `critic_hidden_dim` | 256 | **256** | Must match source so critic can warm-start. |
| `num_episodes` | 2000 | 3000 | Modest bump for the larger grid; cheapest escalation first. |
| `lr` | 3.0e-5 | **3.0e-5** | Validated gentle fine-tune. Source LR would overwrite the warm-start. |
| `warm_start` | `…coop-8/checkpoint.npz` | `…coop-16/checkpoint.npz` | The ladder: each rung warm-starts the next. |

Everything else (`ent_coef`, `epsilon` schedule, `anneal_end_frac`, `grad_clip: 0.25`, activation, actor shape, reward weights, gamma) is identical to quad-16. The grad_clip stays tight — critic input dim grew 4×, and even with the warm-start the critic's gradients on the new scale need throttling.

### Run it

```bash
# Optional — re-run the pre-training diagnostic first to verify the upsample
# produces a strong policy before committing to training wall-clock.
python /tmp/eval_pretraining_warmstart.py configs/quad-cooperate-coop-32.yaml
# -> writes experiments/quad-cooperate-coop-32/episode_pretraining.gif + eval_pretraining.txt

python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-32.yaml
# expect these lines before training starts:
#   Warm-starting from: experiments/quad-cooperate-coop-16/checkpoint.npz
#   Grid-aware warm-start: upsampling Dense_0 grid rows (16x16 -> 32x32, nearest-neighbor).
#   Grid-aware warm-start: upsampling Dense_0 (central-critic, 4-block) grid rows (16x16 -> 32x32, nearest-neighbor).
#   Warm-started critic: input 1048 -> 4120, hidden 256.
```

Report + GIF (single parameterised script):

```bash
python /tmp/render_experiment.py configs/quad-cooperate-coop-32.yaml
open experiments/quad-cooperate-coop-32/report.html
```

### Wall-clock (rough)

3k eps × 5 seeds × 400 steps on a 32×32 grid: **~5–15 min** on an M-series Mac. Dramatically shorter than the 45–90 min the old 25k-ep recipe predicted — the whole point of gentle fine-tune is that you *don't* need 25k eps off a converged warm-start.

### What "working" looks like

- **Warm-start takes effect, and holds** — pre-training diagnostic already shows strong performance; training should preserve or modestly improve on it. Unlike the actor-only baseline, reward should **not** monotonically collapse. Critic warm-start removes the main collapse mode; gentle fine-tune removes the second.
- **`|loss|` stays bounded** — p99 within roughly one order of magnitude across training (baseline run went 1.4k → 2.2M; a healthy run stays within ~10×).
- **Headline target** — sample-from-π mean ≥ **+10**, no more than 1/5 seeds stuck, and sample-from-π strictly better than argmax (entropy preserved). Realistic plateau on 32×32 with 4 agents and 400 steps is around **+10 to +14** (partial coverage, 40–60% cells).

### Knobs NOT to touch without a smoke first

- `grad_clip: 0.25` — held over from the 14×14 quad smoke; loosening it risks critic runaway at the larger critic input dim even with the warm-start.
- `actor_hidden_dim / actor_num_layers / activation` — warm-start requires these match the source checkpoint exactly.
- `critic_hidden_dim / critic_num_layers` — same rule applies to the critic. Changing either silently drops critic transfer and reverts to the failed actor-only baseline.
- `lr` / `num_episodes` — the gentle fine-tune recipe is tuned to the warm-start. Cranking lr back to 1.5e-4 or running 25k eps will overwrite the transferred policy (we tested both, both collapsed).

If it still collapses, the cheapest next escalation is to *further lower* `lr` (e.g., 3.0e-5 → 1.0e-5) rather than anything more invasive. Advantage normalization or a second-layer upsample would be the next rungs up the ladder.

---

## 8×8 → 16×16 warm-start smoke (`configs/quad-cooperate-coop-8.yaml` → `configs/quad-cooperate-coop-16.yaml`)

End-to-end validation of the warm-start pipeline (actor + central critic) before committing to the 45–90 min 32×32 production run. Train 4 agents from scratch on an 8×8 grid, then warm-start the same architecture onto a 16×16 grid. Grid ratio (4× cells) mirrors the 14→32 ratio (5.2×) closely enough to surface bugs at ~7 min wall-clock total.

### What it validated

- **The grid-aware loader transfers both actor and critic correctly.** Expected log lines during load:
  ```
  Grid-aware warm-start: upsampling Dense_0 grid rows (8x8 -> 16x16, nearest-neighbor).
  Grid-aware warm-start: upsampling Dense_0 (central-critic, 4-block) grid rows (8x8 -> 16x16, nearest-neighbor).
  Warm-started critic: input 280 -> 1048, hidden 256.
  ```

- **Pre-training diagnostic (`/tmp/eval_pretraining_warmstart.py`, zero gradient steps on 16×16):** sample-π mean **+7.72**, all 5 seeds in [+6.43, +8.38], final GIF coverage **82.7%**. The upsampled policy generalises to the bigger grid out of the box.

- **The LR lesson, measured directly.** First attempt used the same `lr=1.5e-4, num_episodes=8000` as the 14×14 source: reward collapsed +7.65 (ep 0–500) → +1.68 (final), final coverage 25.5%, agents walked to a corner and stopped — classic overwrite of the warm-started policy. Cheapest fix (`lr=3.0e-5, num_episodes=2000`) kept the warm-start intact *and* improved on it:

| Eval mode | Mean | p5 | median | p95 | % < +1.0 |
|---|---|---|---|---|---|
| argmax, ε = 0 | +3.89 | +0.54 | +3.55 | +8.00 | 12.6 % |
| **sample from π, ε = 0** | **+8.59** | +6.24 | **+8.74** | **+10.61** | **0.4 %** |

All 5 seeds converged in [+8.03, +8.91] — tighter than the 14×14 quad headline (one seed had been stuck at +1.85 there). Sample-from-π beats argmax by +4.7 reward, so entropy was preserved through the gentle fine-tune.

### Takeaway for future scale-ups

A successful warm-start is *already* a strong policy. The fine-tune's job is adaptation, not re-learning — so reuse the source LR only if the source wasn't near convergence. For a warm-start off a converged source, **start with lr ~5× smaller and ~4× fewer episodes** and escalate only if it plateaus. This is cheaper than the collapse-and-diagnose cycle.

Run it:

```bash
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-8.yaml
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-16.yaml
python /tmp/render_experiment.py configs/quad-cooperate-coop-16.yaml
open experiments/quad-cooperate-coop-16/report.html
```

---

## 8 agents on 32×32 with comm_radius=6 (`configs/octa-cooperate-coop-32-r6.yaml`)

First **N-mismatched** rung of the warm-start ladder: take the converged N=4 32×32 checkpoint and lift it to N=8 on the same grid, with a deliberately tight `comm_radius=6` (vs. the 0.5 × width = 16 used by the N=4 family). Tight radius forces a connected swarm rather than a loose flock — agents must be physically close to remain in communication, so the warm-started cooperation pattern has to *generalise* rather than *replicate*.

### Why the central critic needs to grow (and what we do about it)

The actor is decentralised (per-agent obs → per-agent logits) so its first layer is grid-shaped only — adding agents leaves the actor's input dim unchanged at `obs_dim = 1030`. The central critic is the opposite: its first layer takes the **N-concatenated joint obs**, so going N=4 → N=8 doubles its input width (`4 × 1030 = 4120` → `8 × 1030 = 8240`). Re-initialising 4.2M new parameters on top of a converged actor is the textbook "critic re-init collapse" mode (see "Cautionary tale" in the 32×32 scale-up section above).

**The fix: per-agent block tiling.** The central critic is *symmetric in agent identity* — agents are exchangeable, so the per-agent slot weights should be the same for every slot under random initialisation. The grid-aware loader exploits this directly: the source kernel `[5, 4120, 256]` is split into **4 per-agent blocks** of `[5, 1030, 256]`, each block is grid-upsampled (no-op here since the grid already matches), and the 4 upsampled blocks are then **tiled** `tile_factor = 8/4 = 2` times to fill 8 target blocks, yielding `[5, 8240, 256]`. The result is a critic whose every per-agent slot starts identical to the source's (any) per-agent slot — the minimum-assumption initialisation that respects the symmetry the critic is being trained to learn.

The loader gates this transparently: any time `env.num_agents != warm_start_source_num_agents`, set `warm_start_source_num_agents` in the YAML and the loader will validate divisibility (`8 % 4 == 0` ✓), tile the source blocks, and emit a clearly-labelled log line:

```
Grid-aware warm-start: upsampling Dense_0 (central-critic, 4-block source -> 8-block target via tile x2) grid rows (32x32 -> 32x32, nearest-neighbor).
Warm-started critic: input 4120 -> 8240, hidden 256, source_N=4 target_N=8.
```

If `num_agents` doesn't divide evenly into `warm_start_source_num_agents`, the loader skips the critic transfer and warns instead of guessing — agent counts that aren't integer multiples can't be tiled cleanly.

### Caveat: actor sees out-of-distribution `uid`

The actor was trained with `uid ∈ {1, 2, 3, 4}` in its observation tail and now sees `uid ∈ {1, …, 8}`. Observations are not renormalised, so `uid` is a raw float embedded in a ReLU MLP. ReLU networks extrapolate roughly linearly beyond their training range, so the policy doesn't shatter — but expect mild seed-to-seed variance from this distribution shift. The fine-tune's job (in principle) is to absorb that drift; in practice it didn't help here.

### What the warm-start delivers (zero gradient steps)

| Eval mode | Mean | p5 | median | p95 | % < +1.0 |
|---|---|---|---|---|---|
| argmax, ε = 0 | +8.05 | +1.19 | +6.46 | +19.31 | 3.8 % |
| argmax, ε = 0.005 | +8.79 | +1.89 | +7.53 | +20.16 | 0.8 % |
| **sample from π, ε = 0** | **+15.35** | **+8.43** | **+14.81** | **+24.43** | **0.0 %** |
| sample from π, ε = 0.05 | +15.41 | +7.95 | +14.81 | +23.88 | 0.0 % |

All 5 seeds in **[+13.18, +18.37]** under sample-from-π. Best-seed GIF (seed 0): **41.9% coverage with 100% connectivity** at comm_radius=6 — the agents form a connected swarm, not a flock. This is *higher* than the 32×32 N=4 warm-start headline (+13.98 mean) because doubling the team simply produces more discoveries on the same grid.

### What fine-tuning did (and didn't) achieve

Same ultra-gentle recipe as quad-32: `lr=1e-5, num_episodes=500, ent_coef=epsilon=0.005`. Result: **degraded** the warm-start.

| Metric | Warm-start (zero grad steps) | After 500 eps fine-tune |
|---|---|---|
| sample-π mean | **+15.35** | +11.71 |
| sample-π seed band | [+13.18, +18.37] (Δ=5.2) | [+5.38, +17.52] (Δ=12.1) |
| best-seed coverage | 41.9 % | 41.1 % |
| % < +1.0 sample | 0.0 % | 0.0 % |

The mean dropped 3.6 pts (worse than the 2.7-pt drop on quad-32) and seed 1 collapsed from a healthy +13.18 to +5.38 — the seed band more than doubled in width. Coverage on the best seed essentially unchanged. Same qualitative pattern as quad-32 but more severe, consistent with the tiled critic having larger initial TD errors that the gentle LR can't fully absorb in 500 eps.

**The canonical checkpoint at `experiments/octa-cooperate-coop-32-r6/checkpoint.npz` is the raw warm-start.** The 500-ep fine-tuned weights are preserved at `checkpoint_trained.npz` (with `eval_trained.txt`) for negative-result documentation. For N-mismatched transfers, the warm-start strictly dominates this fine-tune recipe.

### Lesson: for N-mismatched transfer, gauge with the diagnostic and consider skipping the fine-tune

The same gentle-fine-tune lesson from the same-N rungs holds, but the bar is *higher*: the tiled critic injects more initial value-prediction noise, so even an LR that worked for same-N can overshoot. Concrete heuristic, applied in the order written:

1. Run `/tmp/eval_pretraining_warmstart.py <config.yaml>` first — it costs ~5 s and tells you how strong the warm-start is on its own.
2. If the warm-start sample-π is already at the headline you'd hope to train *toward* (here +15.35 — strong), the fine-tune is on the wrong side of the bias-variance frontier; **skip training, save the warm-start as the canonical checkpoint** with `python /tmp/save_warm_start_as_checkpoint.py <config.yaml>` and render directly.
3. If the warm-start is decent but clearly leaving coverage on the table, *then* try the gentle fine-tune (lr=1e-5, ~500 eps) — and compare both checkpoints' eval sweeps before promoting.

For genuine improvement at this N-mismatched regime, the principled next moves are: **lr=3e-6** (3× more gentle still) or a brief **critic-only warmup** phase (freeze the actor for the first ~200 eps so the tiled critic learns the new N=8 reward distribution before policy gradients flow). Both are kept on the parking-list inside `configs/octa-cooperate-coop-32-r6.yaml`'s `train:` block.

### Run it

```bash
# Gauge the warm-start first (cheap, decides whether training is worth it):
python /tmp/eval_pretraining_warmstart.py configs/octa-cooperate-coop-32-r6.yaml

# Recommended path: skip training, save warm-start as canonical checkpoint:
python /tmp/save_warm_start_as_checkpoint.py configs/octa-cooperate-coop-32-r6.yaml
python /tmp/render_experiment.py configs/octa-cooperate-coop-32-r6.yaml
open experiments/octa-cooperate-coop-32-r6/report.html

# Or, to reproduce the (degraded) gentle-fine-tune for comparison:
python -m red_within_blue.training.runner --config configs/octa-cooperate-coop-32-r6.yaml
python /tmp/render_experiment.py configs/octa-cooperate-coop-32-r6.yaml
```

---

## Cohesion ↔ coverage tradeoff at N=8 (`-nocoop` and `-conn` variants)

The base `octa-cooperate-coop-32-r6` headline (`coop=0.015`, **41.9 % coverage / connected_frac 1.000**) had a visible cluster bias — agents huddled in the centre and never spread to the corners. We then ran two ablations along the `cooperative_weight` axis to separate "what's reward-pinned" from "what's weight-pinned" in the warm-started policy.

### The all-three results table

Same warm-start (the N=4 32×32 trained checkpoint, lifted by the 4-block-tile loader) and same gentle fine-tune recipe (`lr=1e-5, 500 eps, 5 seeds, ent_coef=epsilon=0.005`) — only `cooperative_weight` changes:

| variant | `coop` | warm-start sample-π | trained sample-π | best-seed coverage | best-seed `connected_frac` |
|---|---|---|---|---|---|
| `-r6`        | **0.015** | +15.35 | +11.71 | 41.1 % | **1.000** |
| `-r6-nocoop` | **0.000** | +0.37  | -0.45  | 48.7 % | 0.269 ⚠️ |
| `-r6-conn`   | **0.005** | +5.36  | +3.71  | 41.0 % | **1.000** |

(Reward magnitudes are not directly comparable across rows because the reward function changed; `connected_frac` and `coverage` are.)

### What we learned

1. **Cluster bias is reward-pinned, not weight-pinned.** The same warm-started weights, evaluated under `coop=0`, jumped to 55.9 % coverage on the raw warm-start (and 48.7 % after fine-tune, the best-seed pick). The huddle pattern is created in real time by the per-witness multiplication of the cooperative bonus (`rewards_training.py:171-176` — `neighbour_disc * cooperative_weight` scales O(N) at clustering), not baked into the policy network.

2. **Removing the cohesion signal entirely is over-correction.** Under `coop=0` the only positive reward for proximity is the terminal coverage bonus. The policy learned to spread to the boundary of comm range → brittle linear chains → eval-time fragmentation cascade (the **0.269** number — see point 3). Coverage went up but cohesion died.

3. **Eval/training guardrail mismatch was the cascade mechanism (now fixed).** During training the JIT `_connectivity_guardrail` (`rollout.py:412-486`) forces STAY when an action would disconnect — inductively the team stays connected because the initial state is connected (`center_spawn=true`). At eval/GIF time the eager `_connectivity_mask` (`rollout.py:48-94`) had a **`if not mask.any(): mask[:] = True`** fallback: when no action of agent `i` could preserve connectivity (the cut-vertex of a brittle chain), the mask flipped to "all 5 actions allowed" → the agent took a disconnecting action → every subsequent agent's mask was also empty → cascade. **Fix landed**: the fallback now allows only STAY (`mask[0] = True`), matching JIT semantics. Test added at `tests/test_rollout.py::TestConnectivityMask::test_fallback_when_no_action_connects_only_stay_allowed`.

4. **`coop=0.005` is the connectivity-restoration sweet spot we have.** 1/3 of the original cohesion strength is enough to keep `connected_frac` at 1.000 across all 5 seeds (warm-start AND trained). Best-seed coverage stays at 41 %, same cluster-bias floor as `coop=0.015` — meaning the `0.005 → 0.015` range produces essentially the same behaviour, and the cluster bias starts decaying only once `coop` drops to 0. This is the binding tradeoff in the current reward shape: **continuous cohesion-vs-coverage is a single dial, not two independent knobs**.

5. **Warm-start dominates fine-tune again.** Same pattern as the base `-r6` run: the gentle 500-ep fine-tune nudged mean reward down (+5.36 → +3.71) without improving coverage. The warm-start is preserved at `experiments/octa-cooperate-coop-32-r6-conn/checkpoint_warmstart.npz` and the trained checkpoint at the canonical `checkpoint.npz`. Same caveat as before — for N-mismatched transfer the warm-start is the safer pick at this LR.

### Where to go next (not yet attempted)

- **uid normalisation**: the residual per-seed *geographic* bias (left-stuck under `coop=0.015`, top-stuck under `coop=0`, right-stuck under `coop=0.005`, etc.) is a different mechanism — the actor was trained with `uid ∈ {1..4}` and now sees `{1..8}`, and ReLU networks extrapolate roughly linearly, adding a per-seed constant logit bias. Fixing this requires dividing `uid` by `num_agents` in `obs.py` and **restarting the warm-start ladder from `pair-8`** (the obs feature changes scale → the warm-start chain is invalidated). Right move for the next experiment, not bundled into this run.
- **Decoupling cohesion from coverage**: the cooperative-bonus shape (`+0.015 × witness_count`) couples them inseparably. A `(num_components - 1) * fragmentation_weight` penalty (broadcast to all agents) would give a connectivity-only signal that doesn't pull agents into a cluster — letting `cooperative_weight` drop while keeping the team connected via a different gradient. Considered for this run but deferred to keep scope tight; would be the right move if uid-normalisation alone doesn't get coverage past 50 %.

### Run it

```bash
# Connectivity-restoration variant (recommended N=8 entry point):
python /tmp/eval_pretraining_warmstart.py configs/octa-cooperate-coop-32-r6-conn.yaml
python -m red_within_blue.training.runner --config configs/octa-cooperate-coop-32-r6-conn.yaml
python /tmp/render_experiment.py configs/octa-cooperate-coop-32-r6-conn.yaml
open experiments/octa-cooperate-coop-32-r6-conn/report.html

# Reproduce the no-coop ablation (for the negative-result documentation):
python -m red_within_blue.training.runner --config configs/octa-cooperate-coop-32-r6-nocoop.yaml
python /tmp/render_experiment.py configs/octa-cooperate-coop-32-r6-nocoop.yaml
```

---

## uid-normalised ladder (`-norm` variants)

### Motivation

Even after the cohesion fix (`-conn`, `coop=0.005`), the rendered GIFs on `octa-cooperate-coop-32-r6*` still showed a per-seed geographic bias — each seed latched onto a different half of the grid (left, top, right, …) and ~10 % of the map was systematically unexplored. The mechanism: the actor was trained on `uid ∈ {1, 2, 3, 4}` but sees `{1, 2, …, 8}` after the N=4 → N=8 tile. ReLU networks extrapolate roughly linearly past their training range, so each unseen uid adds a per-seed constant to every action logit — a deterministic spatial bias.

**Fix (landed `env.py:263-264, 288-289`):** an optional flag `env.normalize_uid: bool` divides the uid feature by `num_agents` in both obs concat sites. With `normalize_uid=true` the uid feature lies in `(0, 1]` for every N, eliminating the OOD shift at transfer time. The flag defaults to **false** so all pre-existing checkpoints remain bit-exact reproducible; the four `-norm` configs opt in explicitly. Behaviour is covered by four tests in `tests/test_env.py` (three parametrised + one default-off).

**Cost:** changing the obs scale invalidates the raw-uid checkpoint chain, so the whole ladder is rebuilt from `quad-8-norm` up. Everything else — network shape, reward weights, LR schedule, episode counts — is identical to the raw-uid siblings, so the comparison is clean.

### Per-seed final-200ep reward

| rung | grid | N | from | final-200ep mean | std | best seed | notes |
|---|---|---|---|---|---|---|---|
| `quad-8-norm`    | 8×8   | 4 | scratch (6000 eps)     | +5.57  | 0.05 | +5.66  | Very tight — all 5 seeds inside ±0.05. |
| `quad-16-norm`   | 16×16 | 4 | `-8-norm` (2000 eps)   | +8.40  | 0.18 | +8.70  | Tight. Reference: raw-uid `-16` sample-π +8.59. |
| `quad-32-norm`   | 32×32 | 4 | `-16-norm` (warm-start canonical) | — | — | — | Fine-tune collapsed 2 of 5 seeds (see "What went wrong at 32×32"). Canonical checkpoint is the banked warm-start. |
| `octa-32-r6-norm`| 32×32 | 8 | `-32-norm` warm-start (500 eps, tile) | **+14.87** | 2.35 | **+17.97** | **N-mismatched rung — beats raw-uid sibling on every metric (raw `-r6` trained: +11.71 ± wide; warm-start: +15.35).** |

### What this bought us

1. **Cleaner same-N chain.** `quad-8-norm` converges to a cohort so tight (std 0.05) that every seed is effectively the same policy — no "structurally bad seed" of the raw chain. Same story at 16×16 (std 0.18).
2. **N-mismatched rung training is now well-behaved.** `octa-32-r6-norm` trained cohort is **+14.87 ± 2.35** with all 5 seeds in [+11.83, +17.97]. The raw-uid sibling's trained cohort was +11.71 with one seed at +5.38; the `-norm` version's worst seed (+11.83) beats the raw sibling's mean. Per-seed spread halved (4.76 → 2.35) after chaining off the clean warm-start (see below).
3. **Fine-tune is no longer strictly dominated by warm-start at the N-mismatched rung.** With uid normalisation the `lr=1e-5, 500-ep` recipe produces a trained mean (+14.87) just below the raw warm-start (+15.35) but with much tighter seeds — the gap between "warm-start alone is dominant" (raw-uid story) and "training helps" (the `-norm` story) is closed.

### What went wrong at 32×32 (same-N rung)

First run of `quad-32-norm` showed per-seed rewards `[7.76, 6.42, 13.24, 13.53, 9.08]` — a bimodal cohort with two seeds dragged down. Tracing seed 0: reward at eps 0–50 was +10.74 (the warm-start policy), then drifted steadily to +4.54 by eps 450–500. Seed 1 did the same. Three seeds held or improved.

Mechanism: the uid-normalised actor has ~4× smaller input magnitude on the uid column, so the gradients flowing back through that column are weaker. At the 32×32 scale (400-step episodes, ~4× reward per ep vs 16×16) the rest of the network adapts faster than the uid weights can keep up — some seeds lose agent differentiation and collapse. The ultra-gentle `lr=1e-5, 500 eps` that worked for raw-uid at this rung is insufficient margin for the `-norm` variant.

Fix: bank the pure warm-start (pre-fine-tune) as the canonical `checkpoint.npz`, preserve the trained version as `checkpoint_trained.npz`. Same move that landed for raw-uid `octa-32-r6` when fine-tune degraded that warm-start. **The fine-tune recipe at this rung of the `-norm` chain is currently a negative result** — kept in the config for reproducibility.

Knock-on effect: the first `octa-32-r6-norm` run was chained off the *degraded* `quad-32-norm` trained checkpoint (seed 0 at +4.54). After banking the warm-start, `octa-32-r6-norm` was retrained from the clean source. Results went from +12.61 ± 4.76 (seeds `[13.62, 4.66, 14.80, 19.05, 10.94]`) to +14.87 ± 2.35 (seeds `[17.97, 11.83, 17.25, 13.43, 13.88]`) — collapsed seed gone, cohort tight, and seed 0 (which the GIF renders) moved from +13.62 to +17.97 (the top seed). The v1 trained artifacts are preserved at `experiments/octa-cooperate-coop-32-r6-norm/*_trained_v1.*` for reproducibility.

### What's still open

- **Recover a clean same-N fine-tune at 32×32.** Try `lr=3e-6` (3× more gentle) or freeze everything except the uid column for the first ~100 eps, then unfreeze. Either would directly address the "uid-column can't keep up" diagnosis. Not bundled into this run.
- The `-norm` ladder was built with `coop=0.015` (original) rather than the `-conn` value `coop=0.005` — so "uid-normalisation + 100 %-connectivity restoration" has not been stacked yet. Worth a run if uid-norm alone still leaves a visible cluster bias in the GIF.
- No eval_trained.txt / pre-training diagnostic text artifacts were produced for the `-norm` rungs (the top-level runner saves `checkpoint.npz` + `metrics.npz` directly). Reports rendered via `python -m red_within_blue.analysis.experiment_report`.

### Run it

```bash
# Full ladder (sequential; each warm-starts from the previous):
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-8-norm.yaml
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-16-norm.yaml
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-32-norm.yaml
python -m red_within_blue.training.runner --config configs/octa-cooperate-coop-32-r6-norm.yaml

# Render HTML + GIF for any rung:
python -m red_within_blue.analysis.experiment_report \
    --config configs/octa-cooperate-coop-32-r6-norm.yaml \
    --experiment-dir experiments/octa-cooperate-coop-32-r6-norm
open experiments/octa-cooperate-coop-32-r6-norm/report.html
```

Total wall-clock: ~10 min for the full chain (165 s + 100 s + 85 s + 240 s on the M-series host).

---

## Survey-local prototype (`configs/survey-local-16-N7.yaml`)

### Motivation

Every run up to 2026-04-18 coupled three things into a single knob (`obs_radius`):

1. **View** — the terrain window the policy reads each step.
2. **Survey** — the footprint of cells committed to the shared `local_map` per step.
3. **Seen-memory shape** — whether the obs contains a full `H·W` binary known/unknown mask.

The action-distribution probe on `octa-32-r6[-norm]` showed `probs[STAY] ≈ 0%` while `committed[STAY]` was 70–83% (the connectivity guardrail was forcing STAY on most steps). Diagnosis: on a tight-comm N=8 regime the team rapidly "sees" its nearby cells and the policy cannot distinguish fresh frontier from already-visited terrain cheaply enough — the policy is indifferent past minute 1, then the guardrail takes over.

This prototype splits the three knobs into distinct config fields and flips the obs-memory from grid-sized to view-sized, so that:

* Agents must **step onto** a cell to mark it explored (`survey_radius=0`). Free survey of a 3×3 block per step is gone.
* The policy no longer has a global "what's explored" map to lean on (`local_obs=true`). It only knows the 3×3 window of its own local_map.
* Comms still work (Level A — raw patch broadcast over the adjacency graph), but the patch is now a single cell, so a comm message is literally "I am at (r, c), which is free".

### Config knobs (all tunable per-experiment)

| Knob | Meaning | Default | This config |
|---|---|---|---|
| `env.obs_radius` | Legacy single-radius fallback. Used only when the two below are left at `-1`. | `1` | `1` |
| `env.view_radius` | Half-size of the sensor window. Policy sees `(2r+1)²` terrain cells around the agent. | `-1` (inherit) | `1` → 3×3 view |
| `env.survey_radius` | Half-size of the per-step "survey" footprint committed to `local_map`. `0` = just the current cell. Also the size of each comm message. | `-1` (inherit) | `0` → 1-cell commits |
| `env.local_obs` | If `true`, replace the `H·W` seen mask in obs with a view-sized local window (OOB padded as known). obs_dim drops from `view_d² + H·W + 5` to `2·view_d² + 5`. | `false` | `true` → obs_dim=23 |

Constraint enforced at env init: `survey_radius ≤ view_radius` (sensor cannot commit what it did not observe).

### Messaging (Level A — unchanged mechanism, clarified semantics)

No learned message channel. When `adjacency[j, i] = True`, sender *j*'s survey patch gets scattered into receiver *i*'s `local_map` at *j*'s position. With `survey_radius=0`:

* **Each message carries exactly one cell** — sender's current position, marked `MAP_FREE`.
* Receivers learn teammate positions indirectly: if I see a new free cell appear at (r, c) and no teammate of mine is at (r, c), a connected teammate just moved there.
* Bandwidth is the thinnest it can be while still keeping any team-memory at all.

If coverage stalls because this is *too* thin, the next knob is `survey_radius=1` (3×3 per message, 9× the info), then reconsider Level B (learned broadcast vectors).

### Warm-start: not from `local_obs=false`, but along a `local_obs=true` ladder

`local_obs=false` checkpoints have a grid-sized `Dense_0` input block (`view_d² + H·W + 5` per agent). `local_obs=true` needs a much smaller input (`2·view_d² + 5` per agent). The runner's grid-aware upsampler refuses to cross-transfer between those two regimes because the block structure differs — not just the grid rows.

However, **within `local_obs=true`** the per-block size is grid-invariant (`2·view_d² + 5`), so N-scaling is a pure block-tiling problem. The runner takes a fast path when `per_block_old == per_block_new`: it tiles the source actor's `Dense_0` input kernel `tile_factor = N_new / N_old` times and transfers the central critic alongside. This enables the ladder below.

### Run it

```bash
python -m red_within_blue.training.runner --config configs/survey-local-16-N7.yaml
python -m red_within_blue.analysis.experiment_report \
    --config configs/survey-local-16-N7.yaml \
    --experiment-dir experiments/survey-local-16-N7
open experiments/survey-local-16-N7/report.html
```

Wall-clock: 4000 eps × 5 seeds on the M-series host should be ~3–5 min (obs_dim=23 is tiny — the network forward is much cheaper than prior rungs).

### What "working" looks like

* **Reward > 0** by episode ~1500 — with `survey_radius=0` and no warm-start this is a harder problem than the `-norm` ladder's 32×32 rungs.
* **Coverage climbs over training** — visible in `coverage_over_time` in the HTML report.
* **No guardrail-forced STAY cliff** — `committed[STAY]` should track `probs[STAY]` instead of diverging to 70%+. Probe it with the action-distribution tool after training.

If any of the above fails, the first thing to try is *not* a new config — it's the diagnostic. Load `checkpoint.npz`, probe the action distribution on a fresh reset, and check what the policy actually samples.

---

## Survey-local warm-start ladder (8×8 N=1 → 8×8 N=2 → 16×16 N=4 → 32×32 N=8)

Four-rung ladder that scales the `survey-local` design (`view_radius=1`, `survey_radius=0`, `local_obs=true`) from a single agent on a tiny grid up to an 8-agent team on a 32×32 grid, warm-starting each rung from the previous. Because `local_obs=true` keeps `obs_dim` grid-invariant, the only thing that changes at each step is the per-agent block count, which the runner's fast-path handles by tiling.

| Rung | Config | Grid | N | Eps | LR | Warm-start | tile_factor |
|---|---|---|---|---|---|---|---|
| 1 | `configs/survey-local-8-N1.yaml` | 8×8 | 1 | 4000 | 3e-4 | fresh | — |
| 2 | `configs/survey-local-8-N2.yaml` | 8×8 | 2 | 2000 | 1e-4 | rung 1 | 1 → 2 |
| 3 | `configs/survey-local-16-N4.yaml` | 16×16 | 4 | 2000 | 5e-5 | rung 2 | 2 → 4 |
| 4 | `configs/survey-local-32-N8.yaml` | 32×32 | 8 | 1000 | 1e-5 | rung 3 | 4 → 8 |

LR is cut as we go up — the warm-start feedback memory says a source-LR rerun overwrites the transferred policy. Each rung transfers both the actor and the central critic (block-tiled where needed).

### Run it

```bash
# Run each rung, then render its HTML report.
for rung in survey-local-8-N1 survey-local-8-N2 survey-local-16-N4 survey-local-32-N8; do
  python -m red_within_blue.training.runner --config configs/${rung}.yaml
  python -m red_within_blue.analysis.experiment_report \
      --config configs/${rung}.yaml \
      --experiment-dir experiments/${rung}
done
```

Total wall-clock: ~2.5 min for all four rungs × 5 seeds (10s + 13s + 31s + 93s on the M-series host).

### Observed results (2026-04-19)

Last-500-episode reward, per seed across 5 seeds:

| Rung | Per-seed last-500 reward | First-100 reward | Note |
|---|---|---|---|
| N=1 | final reward +2.66 ± 0.20 | +1.1–1.7 | Clean rise from cold start |
| N=2 | +4.72 – +4.83 | warm-starts already competent | Warm-start holds — N=1 policy tiles into 2 heads cleanly |
| N=4 | +7.26 – +8.19 | — | Same: warm-start transfers under 2→4 tiling on 16×16 |
| N=8 | +15.4 – +17.3 | +15.4 – +18.0 | Starts at target competence — no further learning needed in 1000 eps |

Reward scales roughly linearly with N (terminal bonus is divided per-agent and the cooperative reward adds per-teammate), so the absolute numbers are not directly comparable across rungs. The interesting quantities are:

* **First-100 ≈ Last-500 at rung 4** — transfer landed the policy in a basin the rung-3 optimum had already found. If the 4→8 tile had fractured the policy, first-100 would have been near 0.
* **All seeds converged** — no cross-seed blowups at any rung, which says the warm-start ladder is stable, not just median-case lucky.

### Soft disconnect-grace experiment (2026-04-19) — regressed, rolled back

Hypothesis: the hard connectivity guardrail (force-STAY when moving would disconnect the team) was pinning agents and hurting coverage at scale. Replace it with a **tunable grace window** — each agent may drift off the largest connected component for `disconnect_grace` steps before the episode fails. Expose per-agent disconnect timers in `info["disconnect_timer"]` for later adversarial-detection work.

Added knobs on `EnvConfig` (all zero by default → hard guardrail behaviour preserved):
- `disconnect_grace: int` — per-agent tolerance in steps. 0 = disabled.
- `disconnect_fail_penalty: float` — uniform reward on grace-trip termination.
- `disconnect_mode: "per_agent" | "team"` — trigger on any single agent tripping, or the whole team needing to reconnect first.

Enabled at 10 % of episode length on N≥2 rungs (grace = 8 / 20 / 40; `fail_penalty = -2.0`; `mode = per_agent`) and re-ran the full ladder. Hard-guardrail ladder preserved under `experiments/Backup/survey-local-hard-guardrail/`.

| Rung | Hard-guardrail coverage | Grace coverage | Grace disc-fail rate |
|---|---|---|---|
| N=1, 8×8  | 86.8 ± 8.6 % | 88.6 ± 8.2 % |  0 / 20 |
| N=2, 8×8  | 99.3 ± 1.2 % | 96.1 ± 4.7 % |  4 / 20 |
| N=4, 16×16 | 97.0 ± 2.8 % | 49.2 ± 5.7 % |  6 / 20 |
| N=8, 32×32 | 65.8 ± 4.5 % | 34.1 ± 2.2 % |  5 / 20 |

(Coverage = fraction of non-wall cells surveyed, averaged over 20 stochastic eval episodes. Disc-fail rate = grace-trip terminations out of those 20 episodes.)

**Result:** grace-at-10 % + `fail_penalty = -2.0` is strictly worse than the hard guardrail at every N≥2 rung. At N=4 coverage nearly halves (97 % → 49 %); at N=8 coverage drops from 66 % to 34 % with 25 % of eval episodes failing via disconnect-trip. The fail penalty appears to bias the policy toward an even tighter cluster than the hard guardrail produced, while also killing episodes early when the cluster still breaks anyway.

**Conclusion:** the mechanism is useful for **future adversarial-detection instrumentation** (per-agent `disconnect_timer` is a clean first-class env observable), but as a replacement for the hard guardrail at these knob settings it regresses the coverage it was meant to unlock. The ladder configs still carry the knobs, but the hard-guardrail ladder is the operating baseline.

Possible follow-ups if we revisit:
- Drop `fail_penalty` to 0 (observability without reward-shaping) and see if the bias disappears.
- Raise `grace` to 25-50 % of episode — 10 % may be too short for an N=8 agent to rejoin after breaking formation.
- Keep `mode=per_agent` for detection use cases but use `mode=team` during training so a single drifter doesn't end everyone's episode.
- Diagnose whether the coverage collapse was concentrated on warm-start rungs (N=4, N=8) vs the cold rung (N=2) — if yes, the transferred policy + new penalty shape is the real issue, not grace itself.

### Spread-reward experiment on 32×32 N=8 (2026-04-19) — 65.8% → 88.7% coverage

Diagnosis: on 32×32, base normalised exploration reward is `1/1024 ≈ 0.001` per new cell while `cooperative_weight=0.015` is ~15× larger, so "stand near a discovering teammate" pays more than "go scout." With the hard guardrail on, agents clump inside comm-range and leave most of the map unsurveyed.

Fix: a new `spread_weight` reward term that pays each agent ∝ mean L1 distance to its teammates. The hard guardrail caps the upper bound (can't drift past comm-range without force-STAY); the spread term drives the lower bound (don't cluster inside the range). Added in `make_multi_agent_reward(..., spread_weight=...)` with unit tests in `tests/test_rewards_training.py::TestSpreadWeight`.

Config (`configs/survey-local-32-N8-spread.yaml` — warm-starts from the hard-guardrail N=4 checkpoint via 4→8 tiling):

| Knob | Parent N=8 | Spread N=8 | Change |
|---|---|---|---|
| `cooperative_weight` | 0.015 | 0.0 | no cluster pull |
| `revisit_weight` | -0.001 | -0.005 | 5× stronger push off known cells |
| `spread_weight` | 0.0 | **0.003** | new dense signal |
| `terminal_bonus_scale` | 8.0 | 8.0 | unchanged |
| `enforce_connectivity` | true | true | hard guardrail preserved |

Result (20-episode stochastic rollout from the final checkpoint, 5 seeds training):

| Variant | Coverage | Mean pairwise L1 dist | Training time |
|---|---|---|---|
| Parent (coop=0.015, no spread) | **65.8 % ± 4.5 %** | 12.85 | 42 s |
| Spread (coop=0, spread=0.003) | **88.7 % ± 3.9 %** | 15.34 | 93 s |

Interpretation:
- Coverage jumped +23 pp on the same warm-start source, same episode count, same network. Pure reward engineering, no new parameters elsewhere.
- Mean pairwise distance rose from 12.85 to 15.34 — agents form a wider chain through the graph (the hard guardrail allows L1 > `2·comm_radius` so long as intermediate agents keep the graph connected).
- Training reward ballooned to +111 vs +16 for the parent because spread bonus is dense (`~0.003 × 5 × 8 × 400 ≈ 48` from spread alone, rest from exploration + terminal). Use **coverage**, not reward, to compare across variants.
- Run `python -m red_within_blue.training.runner --config configs/survey-local-32-N8-spread.yaml` then `python -m red_within_blue.analysis.experiment_report --config configs/... --experiment-dir experiments/survey-local-32-N8-spread` to reproduce.

### Backporting spread to 16×16 N=4 (chained ladder) — 2026-04-19

Ran the same recipe at the lower rung and chained the full ladder to see if N=4 gains compound into N=8.

| Variant | Coverage (20 ep × 5 seeds) | Mean pairwise L1 |
|---|---|---|
| N=4 hard-guardrail (parent, coop=0.015) | **97.0 % ± 2.8 %** | 7.79 |
| N=4 spread (coop=0, spread=0.003) | 96.2 % ± 2.5 % | 7.97 |
| N=8 hard-guardrail (parent) | 65.8 % ± 4.5 % | 12.85 |
| N=8 spread from **hard-guardrail** N=4 | **88.7 % ± 3.9 %** | 15.34 |
| N=8 spread from **spread** N=4 (full-chain) | 82.1 % ± 3.6 % | 14.98 |

Observations:
- N=4 is grid-saturated either way (~97 %). No knob change there helps or hurts measurably.
- **Full-spread ladder regresses N=8 by 6.6 pp** vs the mixed ladder (spread only at N=8). Chaining spread down the ladder is not a win.
- Hypothesis: `cooperative_weight=0.015` at lower rungs teaches the policy to *use the comm graph*. Stripping it everywhere removes that cohesion prior; when the policy hits N=8 and the guardrail suddenly starts force-STAYing, it has no fallback behaviour.
- Curriculum reading: keep the dense cohesion signal at easier rungs, swap to the spread push only when the target scale needs it.

**Recommendation.** Treat the mixed ladder as canonical:

```
N=1 → N=2 (coop=0.015) → N=4 (coop=0.015) → N=8 (coop=0, spread=0.003)
 ↑                              ↑                      ↑
 fresh                      hard guardrail            hard guardrail + spread
```

`configs/survey-local-32-N8-spread.yaml` already encodes this. The chained variant stays in the repo as documented negative result.

Next moves on the reward-engineering track:
- If we want > 90 % coverage on 32×32, layer frontier-potential shaping on top of spread (`prev_dist − new_dist` to nearest unseen non-wall cell).
- If we want a principled Bayesian framing, swap frontier-potential for EIG-view (count of unseen cells in the 3×3 private view after the step).
- Sweep `spread_weight ∈ {0.001, 0.003, 0.01}` at N=8 to find the coverage knee.


## Bayesian fog-of-war reward shaping

**Motivation.** Reframe the coverage task as *entropy reduction*: every cell has a binary state (wall / not wall), and under a deterministic sensor one visit fully disambiguates the cell. Prior entropy per cell is `H_cell = log 2`; post-visit entropy is `0`. Total world entropy `H₀ = N_playable · log 2`, and coverage fraction = `1 − H(t)/H₀`.

**Flat reward = EIG reward under binary observation.** Per-step reward `+1/N_playable` on first visit, `0` on revisit, is literally `ΔH/H₀` per step. The per-episode return is exactly the fraction of world entropy eliminated — no decay constant, no hidden hyperparameter. The flat reward path already encodes the Bayesian objective; the paper framing just makes this explicit.

**New knob: `fog_potential_weight`.** Potential-based shaping that pulls agents collectively toward the unknown:

```
Φ(agent) = − min_{c unknown, non-wall}  L1(agent_position, c)
reward_fog = fog_potential_weight · (Φ_new − Φ_prev)
          = fog_potential_weight · (prev_dist_to_fog − new_dist_to_fog)
```

Positive when the agent moves toward the nearest still-unknown cell, negative when moving away. Because each agent has its *own* nearest-unknown (different agents at different positions see different nearest frontiers), the team naturally spreads across the fog instead of piling onto one edge. No need for a separate spread term — the fog gradient implicitly distributes agents over the uncertainty field.

Potential-based (`Φ_new − Φ_prev`) means the shaping is *policy-invariant* under Ng-Harada-Russell: it changes learning dynamics but not the set of optimal policies. The agents can't "game" fog-chasing by oscillating — the shaping only pays on monotone progress toward unknown territory.

**Benchmark config.** `configs/survey-local-16fast-N5-fog.yaml` — N=5 on 16×16, `max_steps=100`, warm-start from `experiments/survey-local-16fast-N1/checkpoint.npz`. Recipe: `fog_potential_weight=0.01`, `revisit_weight=-0.003`, `terminal_bonus_scale=10`, no spread term. Compare coverage vs. `configs/survey-local-16fast-N5.yaml` (flat reward only, no fog pull).

**Unit tests.** `tests/test_rewards_training.py::TestFogPotential` covers: zero fog → zero reward, weight=0 is a noop, STAY action gives zero ΔΦ, fully-explored map vanishes the term, and moving toward fog is non-negative.

---

## Red contamination — adversarial fog via the comm graph

Red agents (the last `num_red_agents` indices) are adversarial fog generators with two parts to the mechanic:

**Ground-truth exploration counter** — `grid.apply_red_contamination`. After the standard `+1` update, every cell currently occupied by a red has its `explored` entry reset to `0`. The reward path reads `prev_explored` to judge "was this cell previously known?"; zeroing red cells re-opens them to blue discovery bonuses and to the fog-of-war potential pull. Wired into both `env.reset` and `env.step_env`; Python no-op when `num_red_agents == 0`.

**Per-agent belief, propagated through the comm graph** — `agents.update_local_maps_with_comm` now takes `team_ids`. Message content depends on `(sender_team, receiver_team)`:
- blue sender → any receiver: terrain truth (baseline behaviour).
- red sender → red receiver: terrain truth (reds share honest intel among themselves).
- red sender → *blue* receiver: `MAP_UNKNOWN` at the red's surveyed cells — the red fogs the blue's belief.

Critically, fog propagates *only one hop through adjacency*. A blue that is not directly adjacent to the red in the comm graph keeps its belief from truthful blue→blue passthrough. The asymmetry is exactly: "reds communicate normally among themselves; when they communicate to a blue, their message is destructive."

**What this encodes.** The reward/potential path sees the cell as unknown (global `explored == 0`). The directly-adjacent blues also have their subjective belief fogged. Blues outside the red's comm neighbourhood still "think" they know the cell — they only find out it's contaminated when (a) an adjacent blue gets fogged and relays that on subsequent steps, or (b) they re-enter the red's comm range themselves. This matches the user's phrasing: "red surveying something should propagate through msgs with neighboring agents and affect their confidence."

**Unit tests.** `tests/test_grid.py` (4 tests under `apply_red_contamination`) and `tests/test_env.py` (4 end-to-end tests) cover: `num_red_agents=0` is a noop, red spawn cell ends with `explored==0`, colocated blue+red cell stays `explored==0` over many steps, a blue directly adjacent to the red has its `local_map` at the red cell flipped to `MAP_UNKNOWN`, a blue disconnected from the red keeps its belief, and red→red messages stay truthful (reds don't fog themselves).

**Zero-sum reward overlay.** `make_multi_agent_reward(..., num_red_agents=K)` computes every reward term (exploration, disconnect penalty, isolation, revisit, terminal coverage bonus, spread, fog potential) for the first `N − K` blue slots as before, then *overwrites* the last `K` slots with `−sum(blue_rewards) / K`. Per-step team totals therefore sum to exactly zero — reds gain whenever blues lose ground (failed coverage, fragmentation, cluster collapse), not from a separate shaped objective. Wired through `trainer.py`: when `config.env.num_red_agents > 0` the unified `make_multi_agent_reward` path is used (the legacy `normalized_competitive_reward` shortcut has been retired). Tests in `tests/test_rewards_training.py::TestZeroSumRedReward` confirm: per-step total = 0 across all term combinations, all reds receive the same value, `num_red_agents=0` is bit-identical to the old blue-only path, the property holds under disconnect penalty and at the terminal coverage bonus, and the overlay JIT-compiles cleanly.

**Nash & duality diagnostics.** When `red_policy="joint"` (centralized red via `JointRedActor` co-trained against decentralized blues), the trainer emits two extra per-episode metrics designed to make equilibrium dynamics visible:

- `blue_policy_entropy`, `red_policy_entropy` — mean Shannon entropy (in nats) of each team's action distribution across the trajectory. Bounded above by `log(num_actions)`. *Reading the plot:* both entropies stable and positive ⇒ mixed-strategy regime consistent with a Nash equilibrium. Entropy collapse on one side ⇒ that team has settled on a near-deterministic best-response (the other side's policy hasn't kept up). Entropy oscillation on both sides ⇒ self-play cycling, no convergence.
- `duality_violation` = `|blue_total_reward + red_total_reward|` per episode. Always ~0 by construction (the zero-sum overlay is exact); included as a sanity check that the rollout's per-step rewards still cancel after gradient updates.

The report renderer (`src/red_within_blue/training/report.py`) plots these in a new "Nash & Duality" section: a mirror-plot of `blue_total_reward` vs `red_total_reward` (curves should be reflections about y = 0) alongside the per-team entropy curves. Tests in `tests/test_trainer.py::TestJointRedNashDiagnostics` confirm the metrics are emitted with the right shape, entropy is non-negative, the zero-sum property holds end-to-end through training, and the optional `red_pretrain_episodes` warm-up still emits diagnostics.

Both team losses now subtract `ent_coef × entropy` (joint-red trainer, `_blue_loss`/`_red_loss`). Until this fix the joint-red path observed entropy as a metric only; the loss had no entropy bonus and policies collapsed to deterministic STAY within a few hundred episodes regardless of `ent_coef`. Set `train.ent_coef > 0` (e.g. `0.05`) to keep both teams stochastic long enough to discover a mixed Nash. Smoke config `configs/survey-local-16fast-1red-4blue.yaml` (4 blue + 1 red, 16×16, 100-step, 1500 ep × 2 seed) with `ent_coef=0.05` and `fog_potential_weight=0.1` lifts eval coverage from ~5% (no entropy bonus, deterministic collapse) to ~35%, with blue entropy holding ~1.4 nats and red specializing toward ~0.7.

**Tri-view eval gif.** `render_dashboard_frame` (`src/red_within_blue/visualizer.py`) draws three side-by-side panels per frame: **Global** (ground truth + per-cell visit heatmap, what we always had), **Blue belief** (element-wise merge of every blue agent's `local_map`), and **Red belief** (same merge over reds). Cells render as light grey when known FREE, dark grey when known WALL/OBSTACLE, near-black when UNKNOWN. The asymmetric red→blue fog mechanic is visible directly: cells red has stood on while connected to a blue show up as black "fog islands" inside the otherwise-light Blue panel even though the Global panel shows them as visited floor. The metrics panel adds two new lines, `Blue known: X%` and `Red known: Y%`, computed against non-wall cells. No retraining needed to update old gifs — re-run `python -m red_within_blue.analysis.experiment_report --experiment-dir <dir> --config <cfg>`.

**Fog-footprint overlay (2026-04-19).** Plain-snapshot belief panels make the red effect *temporally* invisible: a cell that blue knew at t=10 and got fogged at t=20 looks identical to a cell blue never saw. `record_episode_gif` now accumulates a `blue_ever_known` mask across frames and passes it to `render_dashboard_frame`, which paints cells with `(blue_ever_known & blue_belief == UNKNOWN & non_wall)` a translucent adversarial red. The metrics panel adds `Blue ever-known: X%` and `Fogged-now cells: K`. Result: the red team's mission-denial signature shows up as bright pink scars in the Blue panel.

**STAY-source breakdown (2026-04-19).** `record_episode_gif` now distinguishes three per-agent action categories during rollout: `stay_intended` (policy chose STAY), `stay_forced` (connectivity guardrail forced STAY because the intended move would fragment the comm graph), and `move_taken`. `experiment_report.py` prints the per-agent breakdown when generating the report. Use this to diagnose "why are agents staying so much" — high `stay_forced` ⇒ comm-radius too tight or formation too spread; high `stay_intended` ⇒ undertrained policy / entropy collapse / weak exploration signal.

## Adversarial warm-start ladder (`adv-ladder-r1..r4`)

Goal: see red learn an *anti-policy* — coordinated mission denial against blue — by escalating grid + team sizes while warm-starting blue.

| Rung | Grid | Blue | Red | Episodes | Warm-start (blue actor) | Red |
|------|------|------|-----|----------|-------------------------|-----|
| 1 | 6×6 | 1 | 1 | 2000 | cold | cold |
| 2 | 8×8 | 2 | 1 | 2500 | from r1 (per-block obs grid-invariant under `local_obs=true`, fast-path tile only) | cold |
| 3 | 16×16 | 3 | 1 | 3000 | from r2 | cold |
| 4 | 16×16 | 4 | 3 | 4000 | from r3 (same grid, only N changes) | cold (joint-red shape changes when `n_red` changes) |
| 5 | 32×32 | 7 | 3 | 5000 | from r4 (16→32, fast-path tile) | cold |
| 6 | 32×32 | 6 | 4 | 5000 | from r5 (same grid, N retile) | cold (n_red 3→4 reshapes joint actor) |

All rungs use `red_policy=joint`, `ent_coef=0.05`, `grad_clip=0.5`, `fog_potential_weight=0.1`, `enforce_connectivity=true`. The joint-red REINFORCE trainer doesn't warm-start red params (and per-grid obs_dim differs across rungs anyway), so red is fresh on every rung. Run sequentially: rung N depends on rung N-1's `experiments/adv-ladder-rN-...//checkpoint.npz`.

**Episode budget after the smoke.** Re-ran the ladder with 4× episodes per rung (r1: 5k, r2: 7.5k, r3: 10k, r4: 15k, r5/r6: 20k each) and `max_steps=300` on the 32×32 rungs. Wall-clock: r5 ~18 min, r6 ~18 min, full ladder ~45 min.

**Rung 6 result — emergent red role differentiation.** With 20000 eps × 300 steps the joint-red team in `adv-ladder-r6-32x32-6b4r` now visibly *divides labor*. STAY breakdown over the 300-step eval episode (agents 0-5 = blue, 6-9 = red):

| Agent | Team | stay_intended | stay_forced | moved | Role |
|-------|------|---------------|-------------|-------|------|
| 9 | red | 0% | **95%** | 5%  | anchor at comm boundary |
| 8 | red | 0% | 36% | 64% | patrol |
| 7 | red | 0% | 17% | 83% | roamer |
| 6 | red | 0% | 11% | 89% | roamer |
| 1 | blue | 30% | **58%** | 12% | guardrail-pinned |
| 3 | blue | 36% | **59%** | 5%  | guardrail-pinned |
| 5 | blue | 39% | 45% | 16% | guardrail-pinned |
| 2 | blue | 41% | 41% | 19% | guardrail-pinned |
| 0 | blue | 45% | 37% | 18% | partly free |
| 4 | blue | 38% | 33% | 28% | partly free |

All 4 reds *intend* to move every step (intended_stay = 0%), but the team has split into 2 roamers (~85% movement) plus a near-stationary anchor pinned 95% of the time at the comm-graph boundary. Through that constraint, every single blue agent is also guardrail-forced 33-59% — blue can't disperse without fragmenting from the rest of the team or the surrounding reds.

Coverage tracks the constraint: blue ever-known halved from r5 19.0% → r6 11.0% when one blue was swapped for one red. The "swarm-vs-swarm" anti-policy is visible here — modest in absolute coverage, but qualitatively the right shape: red has learned it doesn't need a uniform behaviour, the team can specialise.

**Caveat.** Blue is still under-trained (11% ever-known means blue hasn't learned a good escape response yet). For a real "watch them adapt" study, scale `num_episodes` another 3-5× and consider lowering `ent_coef` from 0.05 to 0.01 in the late half of training so policies can sharpen.

### Coevolutionary r6 (`scripts/coevo_r6.py`)

A gradient-free cross-check of the REINFORCE rung-6 result: does coevolutionary search find a qualitatively different (better) policy than REINFORCE when the gradient signal is corrupted by opponent non-stationarity? Yes.

**Setup.** Two populations of POP=8 — blue Actors warm-started from r5 with σ=0.01 perturbation; JointRedActors cold-init. Round-robin 8×8 = 64 pairings × 2 episodes each per generation, truncation selection keeps TOPK=2 elites per team with Gaussian mutation σ=0.05. 20 generations. Writes standard checkpoints so `experiment_report.py` can ingest them.

**Run.** `python scripts/coevo_r6.py` — ~100 s wall-clock on an M-series Mac. Output lands in `experiments/adv-ladder-r6-coevo/`.

**Result.** Coevolution lifts blue ever-known coverage **11.0% → 25.5%** (2.3× the REINFORCE r6 result) at a fraction of the wall-clock. Role differentiation survives: 2 red anchors (agents 7, 8: 82-95% guardrail-forced), 2 red roamers (6, 9: 79-94% moved); blue discovers its own 2-anchor + 4-roamer split under ES. Blue fitness trajectory drifts best +0.19 → +0.28, mean +0.17 → +0.21 over 20 gens.

**Why it works.** REINFORCE's per-step policy gradient at r6 is noisy because the red team is simultaneously changing — credit assignment has to fight a moving target. ES only uses the mean team reward per pairing, so non-stationary opponents become bounded fitness noise that 16 per-individual rollouts average out. At this team size + grid, gradient-free wins.

**Follow-ups.** (a) PBT variant (elites continue REINFORCE between generations — blend of gradient + ES). (b) Fictitious-play alternation — freeze blue, train red with REINFORCE to convergence, then swap; lighter than ES but also sidesteps non-stationarity. (c) Seed ES from r6 REINFORCE weights and see whether it escapes the REINFORCE local minimum.

### ε-sweep: ΔJ vs detector-budget (`scripts/eps_sweep_r6.py`)

Eval-time Pareto experiment using the r6 coevo checkpoints. Blue was trained under ε=0 (hard guardrail); we now evaluate against varying ε to measure the paper's ΔJ(ε) curve (Stealth Attacks on Swarms, Mehr 2026, research question 2). ε is interpreted as "how many consecutive steps the attacker may hold the comm graph fragmented before the mission is considered compromised and aborts" — cleanly mapped onto `env.disconnect_grace`.

**Run.** `python scripts/eps_sweep_r6.py` — ~60 s on CPU (6 ε-settings × 20 eval seeds, JIT'd rollouts, no gif encoding).

**Result (r6 coevo policies).**

| ε (grace steps) | blue_ret | red_ret | ep_len | ΔJ vs ε=0 |
|---|---|---|---|---|
| 0    | +0.224 | −0.336 | 300   | 0          |
| 5    | +0.045 | −0.067 | 10.7  | **−0.179** (attacker-optimal) |
| 15   | +0.065 | −0.098 | 21.2  | −0.159     |
| 30   | +0.094 | −0.141 | 36.5  | −0.130     |
| 100  | +0.202 | −0.303 | 107.2 | −0.022     |
| ∞    | +0.313 | −0.470 | 300   | **+0.089** (blue beats baseline) |

Blue return is non-monotone in ε — there's a clear attack-effectiveness maximum at ε≈5 where red can trip the detector early and force the mission to abort before blue accumulates terminal coverage bonus. At ε=∞ the trained guardrail reflex in blue's policy still holds, AND blue is no longer pinned to the comm boundary — blue actually outperforms the ε=0 baseline by +0.089. The guardrail was partially helping red by constraining blue motion.

This is one of the paper's three headline figures (Pareto of ΔJ vs detection). Follow-ups on the same machinery: (i) k-sweep over red-team size to answer the paper's k\*(θ) question; (ii) channel ablation — disable red→blue MAP_UNKNOWN messages to isolate the observation/message-channel contribution to ΔJ.

### Compromise-sweep at fixed N=5 (`scripts/compromise_compare.py`)

Answers the paper's k\*(θ) question on a presentable scale: holding total team size constant at N=5 on 16×16, measure how blue-team coverage degrades as m of N agents are compromised. Coverage is `blue_ever_known` — fraction of non-wall cells at least one blue has seen or been told about at episode end.

**Why N=5.** The writeup constraint is *red-strictly-less-than-blue* (m < N−m) so the compromise never reaches parity. The smallest N that admits two non-trivial compromise points satisfying that constraint is N=5: m=1 (4b+1r) and m=2 (3b+2r). Both setups pair 3 or 4 decentralised blue actors against 2 or 1 centralised joint-red controller.

**Training pipeline.**
1. Clean N=5 baseline: `experiments/survey-local-16-N5-from-N4/checkpoint.npz` (warm-started from N=4, already trained).
2. Coevolutionary fine-tune against 1 red, warm-started from the clean baseline:
   `python scripts/coevo.py --config configs/compromise-16x16-5-4b1r.yaml --warm-blue experiments/survey-local-16-N5-from-N4/checkpoint.npz --output-dir experiments/compromise-16x16-5-4b1r-coevo --pop 8 --topk 2 --gens 20 --eps-per-pair 2`
3. Same for 2 red: swap in `configs/compromise-16x16-5-3b2r.yaml` and `experiments/compromise-16x16-5-3b2r-coevo`.

Each coevo run is ~30 s wall-clock on a Mac (20 gens × 64 pairings × 2 eps, JIT-compiled rollouts).

**Evaluation.** `python scripts/compromise_compare.py` — 20 eval seeds per setup, ~1 min total. Python-side env walk (no gif encoding), same `_merge_team_belief` routine the visualizer uses. N=5 clean was trained with `max_steps=250`; the eval overrides to 200 so B / C1 / C2 share a task horizon.

**Artifacts** (all under `experiments/compromise-compare/`):
- `report.png` — two-panel figure: per-step coverage curves (left, shaded p10–p90 over 20 seeds) + final-coverage box plot (right, with per-seed jitter).
- `compromise_compare.npz` — per-seed `finals_i`, `curve_i`, `ep_len_i` arrays (one per setup) for custom plotting.
- `run.log` — captured stdout of the sweep.

**Presentation report** (`../DL presentation/compromise_report.html` + 3 PNGs, regenerable via `python "../DL presentation/scripts/_render_compromise_report.py"` — the deck moved to a sibling folder outside the repo, with all render scripts now under `../DL presentation/scripts/`). Nature/NeurIPS-styled standalone HTML for showing the B / C1 / C2 contrast: bottom-line outcome table + significance verdicts (paired bootstrap on per-seed differences, n = 20, 5 000 resamples) + three figures (final-coverage scatter with bootstrap mean ± 95 % CI, coverage trajectory with SEM band, performance profile). Verdicts: B vs C1 different (very large effect, d = +2.36), B vs C2 different (very large effect, d = +1.58), C1 vs C2 indistinguishable (paired-bootstrap CI on Δ crosses 0; d ≈ +0.28). The first red does almost all the damage; the second adds little to the mean but inflates the variance, matching the Welch's-t headline numbers above.

**Per-setup reports.** Each coevo run also has its own dir with `report.html`, `episode.gif`, `checkpoint.npz`, `joint_red_checkpoint.npz`, `run.log`, `report.log`, `coevo_history.npz` and `metrics.npz` — produced by `python -m red_within_blue.analysis.experiment_report --config <cfg> --experiment-dir <dir>`.

**Result (20 seeds per setup).**

| setup                    | n | cov final %   | std | min   | max   | % seeds ≥90% |
|--------------------------|---|---------------|-----|-------|-------|--------------|
| S (N=1 solo blue)        | 1 | 46.1          | 6.8 | 34.7  | 57.1  | 0% (0/20)    |
| B (N=5 clean blue)       | 5 | **98.5**      | 2.3 | 89.3  | 100.0 | 95% (19/20)  |
| C1 (4 blue + 1 red, m=1) | 5 | 89.6          | 3.8 | 83.2  | 95.4  | 55% (11/20)  |
| C2 (3 blue + 2 red, m=2) | 5 | 87.1          | 7.8 | 65.8  | 94.9  | 45% (9/20)   |

**Pairwise Welch's t-tests:**

| contrast | ΔJ (pp) | t | p | Cohen d |
|---|---|---|---|---|
| S  vs B  | −52.5 | −32.5 | 6e-21 | −10.3 |
| C1 vs B  |  −9.0 |  −9.0 | 3e-10 |  −2.9 |
| C2 vs B  | −11.4 |  −6.3 | 3e-06 |  −2.0 |
| C2 vs C1 |  −2.5 |  −1.3 | 0.22  |  −0.4 |

**Narrative for the writeup.**
1. **Cooperation is necessary.** A single blue caps at 46% after 300 steps; N=5 reaches 98.5% in 200. ΔJ = +52 pp, Cohen d = 10. Single-agent exploration is not a substitute for a swarm on this grid.
2. **One compromised agent (m=1) already pushes the team below the 90% threshold.** ΔJ = −9.0 pp, p = 3e-10, d = −2.9. Only 55% (11/20) of seeds still hit 90% coverage — down from 95% of seeds in the clean baseline. The attacker's task at 4-vs-1 is to keep the guardrail pinning the remaining 4 blues against the comm boundary so they can't spread.
3. **A second compromised agent (m=2) drives the variance, not the mean.** ΔJ from C1 is only −2.5 pp and is *not* statistically significant at N=20 (p=0.22). But the std more than doubles (3.8 → 7.8) and worst-case coverage drops from 83% → 66%. At m=2 the outcome becomes regime-dependent: good red-blue coupling still gets ~95%, bad coupling collapses to 66%.
4. **The k\*(θ) knee is at m=1 for this setup.** Unlike the earlier N=4 run (where m=1 was small and m=2 was the break), at N=5 the very first compromise already clears the detection-relevant threshold. The incremental hurt from adding the 2nd red is marginal — if the attacker can get one agent in, they've already won most of the available damage.

Saves `experiments/compromise-compare/compromise_compare.npz` with per-seed finals, per-step curves, and episode lengths for plotting.

### Misbehavior-budget sweep (`scripts/misbehavior_budget_sweep.py`)

Reduces the proposal's (ε, T_int) stealth-budget pair to a single scalar per-step policy-negation probability `ρ ∈ [0, 1]`. At each step, each of the `k` compromised agents independently executes the trained joint-red policy with probability `ρ`, else the nominal action = blue actor applied to its own obs. The expected per-step TV distance between compromised and nominal policy equals `ρ`, so **M := k · ρ** is the total "deviation mass" per step — the proposal's budget collapsed to one axis.

Sweeps `k ∈ {1, 2}` × `ρ ∈ {0, 0.25, 0.5, 0.75, 1.0}` on the already-coevo-trained policy pairs in `experiments/compromise-16x16-5-{4b1r, 3b2r}-coevo/` — **no retraining**, just eval. 10 seeds per cell, max_steps = 200. ~45 s wall-clock for the full 100-episode grid.

Run: `python scripts/misbehavior_budget_sweep.py --n-seeds 10`

Saves `experiments/misbehavior-budget/budget_sweep.npz` with per-seed finals + metadata (k, ρ arrays), and `run.log`.

**Two findings worth flagging for the writeup.**
1. **ρ = 0 floor is non-zero.** At (k=2, ρ=0) the two red agents act nominally but still have `team_id = 1` — excluded from the blue-team message merge — so coverage caps at **81.9%** (ΔJ = +16.6 pp) before any adversarial policy is applied. A fraction of the damage comes "for free" with the compromise label.
2. **Non-monotonicity in ρ at k=2.** Damage at ρ=0 (+16.6 pp) exceeds damage at ρ=1 (+14.0 pp): the trained joint red occasionally plays positional moves that happen to help team coverage. Implication for the proposal: ΔJ(φ) is **not monotone in the stealth budget** because the nominal action is not information-neutral for a labelled adversary.

### Adversarial-validation deep-dive (`experiments/adversarial-validation/`)

n=60 follow-up to the misbehavior-budget sweep that fixes the OOD-nominal artefact (clean nominal = blue actor with `team_id` clamped to 0), ablates by adversary type, anchors k-scaling, and fits two competing closed-form ΔJ-decompositions. Each phase saves its own `phaseN_*.npz` and the run log is keyed off the output path.

Reproduce: see `experiments/adversarial-validation/PLAN.md` for the full command sequence (smoke → Step 0 regression → Phases 1–7).

Re-render the figures:

```bash
PYTHONPATH=scripts:src python3 experiments/adversarial-validation/render.py
PYTHONPATH=scripts:src python3 experiments/adversarial-validation/render_sensitivity.py
PYTHONPATH=scripts:src python3 experiments/adversarial-validation/fit_threat_model.py | tee experiments/adversarial-validation/fit_threat_model.txt
```

Outputs:

- `summary.png` — five-phase summary (adversary-type bars, ρ-curve at k=2, k-scaling).
- `sensitivity.png` — **sensitivity-analysis viz** (tornado of ΔJ-per-axis-unit, ρ-response curves with OLS slopes, held-out (k=2, ρ=1) falsification with z-scores). Use this figure when arguing the (k, ρ) sweep is a *sensitivity instrument*, not a fit.
- `fit_threat_model.txt` — raw fit output for M1 (additive `α·k + β·k·ρ`) vs M2 (concentration `α·k + β·max ρ`).
- `results.md` — written interpretation. **Read first.** Reframes the dataset as a detector-calibration instrument: ρ measures attacker visibility (defender signal), k measures post-detection cost. The action-stream channel is statistically silent at this scale; both M1 and M2 are over-predicting by ~4 pp at z>4.

### Meta-report for the writeup (`scripts/meta_report.py`)

Produces a single **self-contained** HTML synthesis that pairs each proposal claim (cooperation, ΔJ(1) ≥ θ_detect, sub-linear marginal damage, variance inflation at m=2) with the empirical quantity answering it. Reads the already-trained checkpoints in `experiments/survey-local-16-N5-from-N4/` and `experiments/compromise-16x16-5-{4b1r,3b2r}-coevo/`, runs one canonical eval episode per setup (seed 0, `max_steps=200`), and renders:

- `claims_evidence.png` — centerpiece: 5-panel evidence figure keyed to Claims 1–4 (S vs B bar, ΔJ(k) curve, marginal ΔJ bars, per-seed box+strip, %seeds ≥ 90% threshold).
- `trajectories.png` — per-setup (B/C1/C2): terrain + residual blue-unknown red tint + every agent's full 200-step polyline (hollow ring = start, filled disc = end). Makes the mechanism — where the red pins the sub-team and why a frontier doesn't close — visible in one image.
- `budget_heatmap.png` — (k, ρ) grid: absolute coverage on the left, ΔJ(k, ρ) on the right.
- `budget_curves.png` — coverage vs ρ (fixed k) on the left; collapsed onto the unified M = k · ρ axis on the right — shows k and ρ are NOT interchangeable at equal M.
- `budget_surface.png` — 3D isometric bars of ΔJ(k, ρ) + budget-Pareto scatter (ΔJ vs M with each point annotated by its ρ).
- `comparison_matrix.png` — 3 setups × 4 timesteps, ground truth with red tint over cells still hidden from the blue team (attacker's effective hidden region across time).
- `fog_footprint.png` — final-step blue-unknown masks, side-by-side (single seed, kept as an illustrative companion).
- `proximity_ecdf.png` + `proximity_bands.png` + `proximity_summary.json` — 20-seed aggregation of the residual-fog story in §6.6 of `meta_report_v2.html`. For each non-wall cell at `t=200`, Chebyshev distance to the nearest red (B: to nearest blue). ECDF and distance-band stacked bars pool unknown cells across 20 seeds per setup. Key finding: C1's median unknown cell sits **11 cells** from the red (near the opposite corner on a 16×16 map) — the lone red breaks the comm backbone rather than occluding its own neighbourhood. Generated by `scripts/meta_report_fog_proximity.py` (~90 s).
- `coverage_curves.png` — per-step coverage curves, all three setups on one axis.
- `compromise_compare.png` — copy of the 20-seed aggregate figure from `compromise-compare/`.
- `episode_{B,C1,C2}.gif` — canonical episode gifs copied from the per-experiment report dirs.
- `meta_report.html` — proposal math block (Unicode), code↔symbol mapping table, claim-vs-evidence sections, all figures + gifs inline. No external links — the directory opens standalone.
- `run.log` — captured stdout.

Per-agent belief panels (the old `viz_{B,C1,C2}.png`) were dropped: after the env's teammate-message merge all blue beliefs look near-identical, and compromised agents barely move, so the per-agent view was 3× redundancy without content.

Run: `python scripts/meta_report.py` (~30 s, single process, no training). Requires `experiments/compromise-compare/compromise_compare.npz` for the claim-section numbers and (optionally) `experiments/misbehavior-budget/budget_sweep.npz` for the §5 budget figures — if the budget sweep hasn't been run, the script skips those three figures and the §5 prose uses NaNs. All artifacts land under `experiments/meta-report/`.

#### `meta_report_v2.html` figures (`scripts/meta_report_v2_figs.py`)

Companion generator for the v2 rewrite (`meta_report_v2.html`). Reads the same two `.npz` caches (`compromise_compare.npz`, `budget_sweep.npz`) — no training, no new rollouts — and renders 11 analysis figures embedded inline in the v2 HTML:

- `claim1_invariant.png` — per-agent throughput bars + reach-probability curves for Claim 1 (cooperation is real).
- `time_to_coverage_multiseed.png` — median coverage curve per setup with p10–p90 band; median T(90%) vertical drops (89 / 159 / NEVER for B / C1 / C2).
- `forest_delta_j.png` — ΔJ(k) forest plot with bootstrap 95% CI whiskers and the θ_detect = 0.05 dashed line.
- `variance_bar.png` — σ bars + per-seed strip dots (tests the "variance inflation at m=2" claim).
- `kstar_staircase.png` — step function of k*(θ) with ΔJ(1), ΔJ(2) markers.
- `resilience_triangle.png` — 3-axis polar radar (Magnitude / Brittleness / Timeliness) summarising §8.2.
- `channels_stacked.png` — 4-segment bars attributing total ΔJ to {team-label exclusion, redundant labour, direct fogging, residual} using frozen §6.4 numerics.
- `budget_match_diff.png` — matched-M bar chart with bootstrap whiskers (tests whether k and ρ substitute at equal budget — they don't).
- `budget_raincloud_grid.png` — 2×5 small multiples: per-seed dot clouds for every (k, ρ) cell.
- `budget_pareto.png` — ΔJ vs M Pareto scatter with upper/lower envelopes.
- `model_decomposition.png` — calibrates α, β_C, γ for the ΔJ ≈ αk + β_C Σρ + γ (k choose 2) σ₀² model. **R² = −2.84 → model falsified**; the shape of Σρ matters, not just its sum. Ties into §5.2's open hetero-sweep question.

Run: `python scripts/meta_report_v2_figs.py` (~40 s). The companion proximity figures (`proximity_ecdf.png`, `proximity_bands.png`) come from `scripts/meta_report_fog_proximity.py` (20-seed rollouts, ~90 s).

#### `meta_report_v3.html` figures (`scripts/meta_report_v3_figs.py` + `scripts/misbehavior_hetero_sweep.py`)

v3 addresses v2's three remaining deferrals and **overturns** the v2 §8.1 decomposition model. Two new scripts:

- **`scripts/misbehavior_hetero_sweep.py`** — fixes `k = 2` on the `3b2r-coevo` checkpoint and sweeps (ρ_A, ρ_B) along the constraint line `ρ_A + ρ_B = Σρ` for Σρ ∈ {0.5, 1.0}, 6 asymmetry levels × 15 seeds × 200 steps. Writes `experiments/misbehavior-budget/hetero_sweep.npz` + `hetero_run.log`. ~85 s wall-clock (195 episodes). Tests whether the scalar Σρ is sufficient or whether the SHAPE of ρ matters.

  **Key finding**: at Σρ = 1.0 the ΔJ spread across shapes is **6.50 pp** (balanced (0.5, 0.5) → ΔJ = −5.20 pp; concentrated (0, 1.0) → ΔJ = +1.29 pp). That is larger than the k=2 effect we're trying to explain — so **Σρ alone is under-specified**. Concentrating the attack on one agent at ρ=1 is more damaging than spreading it at ρ=0.5 each, because the coevo-trained blue has learned to route around the 50%-STAY/MOVE flicker but not around a consistent single-agent adversary.

- **`scripts/meta_report_v3_figs.py`** — produces five figures for `meta_report_v3.html`:
  - `hetero_sweep.png` — ΔJ vs asymmetry at fixed Σρ, 2 curves × 6 points × 15 seeds with standard-error whiskers. The dotted horizontal per-Σρ mean is what the sum-only model predicts; curves visibly bend away from it.
  - `spacetime_tubes.png` — 3D isometric panel per setup (B/C1/C2): per-agent trajectories as 3D polylines through `(x, y, t)` with terrain walls + residual-fog cells on the `t = 0` plane. Makes "red hoards / blue explores" spatially legible.
  - `spacetime_entropy.png` — per-agent Shannon policy entropy \( H(\pi(\cdot|o_t)) \) as 3D tubes (x = agent, y = step, z = entropy nats). Shows the joint red collapsing to near-deterministic actions at k = 2 (red-mean H: 0.83 → 0.45 nats from C1 → C2) while blue stays roughly stochastic. Also writes `spacetime_entropy_summary.json`.
  - `spacetime_uncertainty.png` — red voxels at `(x, y, t)` for every cell blue once saw but red has re-fogged (Channel 3 — direct uncertainty manipulation). Peak-fogged: B = 0, C1 = 2, C2 = 3 cells. Area-under-fog: B = 0, C1 = 166, C2 = 215 cell-steps. Confirms the memory "fogging channel is thin". Also writes `spacetime_uncertainty_summary.json`.
  - `system_diagram.png` — Appendix A.1 block diagram of the env / blue actors / joint red / comm-merge / central critic / zero-sum reward flow (pure matplotlib, no graphviz dep).
  - Also writes `hetero_summary.json` with per-Σρ mean/spread/min/max ΔJ for citation in the v3 HTML prose.

Run: `python scripts/meta_report_v3_figs.py` (~15 s, requires the hetero sweep to have been run first). `meta_report_v3.html` embeds these figures inline; `meta_report_v2.html` is marked FINAL and links forward to v3.

#### `meta_report_threat.html` — condensed threat-model short form

A single-concern subset of v3 focused only on adversarial behaviour + threat modelling. Sections: (1) threat model — game, attacker's policy space, budget candidates (M = k·ρ vs Σρ vs c(𝛒)); (2) three attack channels; (3) damage scaling in k, Σρ, and shape (with the §3.3 hetero-sweep headline); (4) mechanism — sabotage is delay, not information theft; (5) defender specification — k*(θ), variance inflation, Attack-Resilience Triangle; (6) compact threat-model equation with rough coefficient magnitudes; (7) open questions scoped to threat modelling.

~470 lines HTML (v3 is ~1400). Reuses v3's figures verbatim — no new data, pure re-scoping. Hand-written, not script-generated, so update it manually if you rewrite a figure. Intended audience: reader who wants the adversary model without the cooperation claim, the engineering retrospective, or the full appendix.

#### `meta_report_experimental.html` — exploratory visualisation gallery (`scripts/meta_report_experimental.py`)

Twelve-figure gallery that re-projects the existing `compromise_compare.npz` and `hetero_sweep.npz` caches through unconventional chart styles (ridgeline KDE, Q-Q overlay, bump chart, streamgraph, parallel-coordinates, polar-wedge, 3D surface of ΔJ(k, ρ), seed-correlation matrix, etc.). No new rollouts — runs in ~5 s. Purpose: stress-test whether the headline findings (sabotage is delay; variance inflates under attack; scalar Σρ is under-specified) survive arbitrary re-slicing, and surface any pattern we missed.

**Headline new finding from the gallery**: seed-correlation matrix shows `r(C1, C2) = −0.06` — seeds that collapse hardest under k = 1 are *uncorrelated* with seeds that collapse under k = 2. Attack doesn't just inflate variance, it **reshuffles which seeds are fragile**. Cached at `experiments/meta-report/exp_summary.json`.

Run: `python scripts/meta_report_experimental.py`. Writes `exp_*.png` (twelve figures) + `exp_summary.json` to `experiments/meta-report/`. Gallery HTML at `experiments/meta-report/meta_report_experimental.html` is hand-written and embeds the figures inline with "what I hoped to see / what I actually saw" sniff-test captions. Not linked from v3 — it's a scratch pad, not a canonical writeup.

#### `meta_report_xai.html` — explainable-AI probe (`scripts/meta_report_xai.py`)

Vanilla input-saliency on the 3b2r-coevo checkpoint, designed to test the v3/threat-model "sabotage-is-delay, not info-theft" mechanism directly at the policy level. For every step of the canonical seed=0 episode in each setup (B/C1/C2), computes `∂logit_{a*}/∂obs` (a* = sampled action) and aggregates `|·|` by obs-block. For the C2 joint-red controller, splits the 46-dim input gradient into `own` and `cross` slices to measure how much each red's action depends on the other red's observation.

**Three findings (cached at `xai_summary.json`)**:

1. **Falsifies the info-theft hypothesis.** At C2, red's saliency on the seen-field block is 0.17 — same magnitude as its scan-frame attribution. If red's strategy were to track-and-fog blue's belief, seen + map_frac would dominate. They don't. Red's biggest blocks at C2 are `uid + team_id` (combined 0.35) — red is executing a **fixed identity-conditional policy** ("if I'm red_0 do X, if I'm red_1 do Y"), not an information-aware controller.

2. **Joint-red controller is genuinely joint, not two independent heads.** Cross-attention share = 0.43 (own 0.57 / other 0.43), biggest on `norm_pos` (1.5× own) — the two reds coordinate on **where each other is**, consistent with the spacing/parking division in the v3 spacetime-tubes figure. The "central controller" formalism does real work in the loss surface; ablating it should be measurable.

3. **Blue's mechanism is invariant to the attack.** Blue's per-block attribution shares are nearly identical at B/C1/C2 (norm_pos ≈ 0.25–0.28, seen ≈ 0.17–0.20). Coverage degrades because blue is denied 40 % of the labour budget, not because it computes differently — direct policy-level confirmation of the v3 §3.1 "load-bearing cooperation" claim.

Run: `python scripts/meta_report_xai.py` (~30 s, single seed, no training). Writes four figures (`xai_block_team_means.png`, `xai_red_self_vs_cross.png`, `xai_spatial_seen.png`, `xai_block_stack.png`) + `xai_summary.json` under `experiments/meta-report/`. Hand-written HTML at `meta_report_xai.html` is a separate report — not linked from v3.

#### XAI extensions — causal occlusion, identity-swap, linear probes (`scripts/meta_report_xai_causal.py`)

Vanilla saliency turned out to be misleading on the 3b2r-coevo checkpoint: it attributed 35 % of red's decision-mass to `uid + team_id` at C2, which the causal counterfactual then overturned. This script runs three orthogonal extensions on the same checkpoint, 5 seeds, all on canonical episode lengths.

1. **Counterfactual occlusion.** For each (setup × block × seed), zero an obs-block and re-roll the full episode; record Δ-coverage. Also recompute the policy distribution on the original (unmodified) trajectory with the block zeroed; record per-step KL between perturbed and original. Behavioural impact and policy sensitivity, separated. Headline: at C2, occluding `scan` on the joint-red gives KL = 3.20 (near-collapse) while `uid` and `team_id` give KL ≤ 0.03 — **red is purely scan-reactive, not identity-conditional**, the saliency-level result was a false positive.

2. **Identity-swap counterfactual.** Within the red team, swap the uid scalar values (the input-vector layout is unchanged) and re-roll. Compute swap_score = KL(swap || baseline-other) − KL(swap || baseline-self). Negative ⇒ swapping uid swaps behaviour (identity-conditional); non-negative ⇒ behaviour is determined by input-slot index, not uid. Headline: 0/5 seeds show a behaviour-swap; mean swap_score = +13.3 nats. **The joint-red coupling is positional / concat-order routing, not identity look-up.**

3. **Linear probes on the actor's last hidden layer.** Concepts: `will_stay_next`, `frontier_in_view`, `blue_in_view` (red only), and a label-shuffle baseline. 5-fold CV, sklearn `LogisticRegression`. Headline: red C2 encodes `blue_in_view` at 0.96 accuracy but its `will_stay_next` probe sits at 0.54 (≤ shuffled 0.57) — **red represents blue's location and does not act on it**. Open follow-up: probe the central critic to discriminate critic-driven encoding vs incidental co-occurrence.

Run: `python scripts/meta_report_xai_causal.py` (~3 min, 5 seeds, no training). Writes three figures (`xai_occlusion_coverage.png`, `xai_occlusion_kl.png`, `xai_identity_swap.png`, `xai_probes_accuracy.png`) + `xai_causal_summary.json` under `experiments/meta-report/`. Results merged into `meta_report_xai.html` §6–§8.

#### XAI extension — integrated gradients (`scripts/meta_report_xai_ig.py`)

Cleaner saliency. For each step, computes `IG_i = (o_i − baseline_i) · (1/M) Σ_α ∇logit_{a*}((1−α)·baseline + α·o)` with M = 32 path-steps and baseline = zero, 5 seeds. Removes two known vanilla-saliency failure modes: high-gradient near-zero inputs (the textbook source of false positives — exactly what bit us with `uid` and `team_id`) and per-step gradient artefacts.

Findings: red C2 `scan` share rises 17 % → 33 % (now agrees with the occlusion ranking); blue `team_id` share collapses to ~0 across all setups (vanilla had it rising B = 0.04 → C2 = 0.13 — that was the high-grad-low-value artefact). Direction matches the occlusion result; IG is the cheap-but-cleaner default if you can't afford the full counterfactual.

Run: `python scripts/meta_report_xai_ig.py` (~2 min). Writes `xai_ig_team_means.png`, `xai_ig_vs_saliency.png`, `xai_ig_summary.json`. Results merged into `meta_report_xai.html` §9.

#### Open-the-box — ten XAI methods × six checkpoints (`scripts/openthebox.py`)

Cross-experiment XAI sweep: applies ten attribution / probe methods to six trained checkpoints spanning team size (N=2/3/4/8), grid size (16×16/32×32), and red budget (k∈{0,1,2}). Methods: **M1** vanilla saliency, **M2** integrated gradients (32 path-steps), **M3** SmoothGrad (n=12, σ=0.10), **M4** Gradient×Input, **M5** block occlusion (KL on baseline trajectory + Δcov on full reroll, 5 seeds), **M6** per-cell spatial occlusion over the 9 scan/seen cells, **M7** identity-swap counterfactual (uid permutation within team), **M8** linear probes (`will_stay_next`, `frontier_in_view`, `blue_in_view`) with `LogisticRegression` + 5-fold CV, **M9** hidden-state PCA (2 components, fit globally across all setups), **M10** TCAV (head-weight cosine with each probe's normal direction).

Setup keys: **B** (`pair-cooperate-coop`, N=2 16×16, no red), **C1** (`compromise-16x16-5-3b2r-coevo`, k=1), **C2** (k=2), **D1** (3b1r-coevo), **D2** (2b2r-coevo), **L6** (`adv-ladder-r6-coevo`, N=4 32×32). Each setup's policies are loaded from existing `checkpoint.npz` / `joint_red_checkpoint.npz`; nothing is retrained.

Findings (all numbers from `openthebox_summary.json`):

- **Saliency-vs-occlusion disagreement generalises.** Across all 5 attacker setups, vanilla saliency over-weights `uid + team_id` while occlusion-KL on `scan` dominates (red KL on scan: B=0, C1=0.18, **C2=3.20**, D1=0.61, D2=0.40, L6=0.67). Spearman rank between M1–M4 and M5 stays ≈ 0.4 — the gradient family agrees with itself (r ≥ 0.85) but shares a systematic high-grad-low-value bias. **Use occlusion when stakes are high.**
- **Joint-red coupling is positional, not identity-based.** 0/15 trials produce a behaviour-swap; mean swap_score = +12.4 (positional). The "central controller" is concat-order routing — even at L6 with n_red=4.
- **Information availability ≠ information use.** Red `blue_in_view` linear-probe accuracy is 0.90–0.99 universally; M5 KL on red `seen` ranges 0.005–0.71. Encoded everywhere; acted on regime-dependently.
- **L6 (32×32) attenuation.** Blue's `seen` KL drops to 0.33 (vs 1.0–1.9 at 16×16) — single 3×3 windows are smaller fractions of mental state at scale. Hidden-state PCA splits L6 into two distant clusters (blue ≈ −20 PC1, red ≈ +5 PC1).
- **D1 anomaly.** Δcov(uid) = +7.3 ± 1.4 pp is the only 3σ-from-zero block effect across 6 setups × 6 blocks. Flagged as open follow-up.

Run: `python scripts/openthebox.py` (~12 min wall-clock end-to-end on M-series; loads 6 checkpoints, no training). Writes 8 figures (`openthebox_block_attribution.png`, `openthebox_method_correlation.png`, `openthebox_block_occlusion.png`, `openthebox_per_cell_occlusion.png`, `openthebox_identity_swap.png`, `openthebox_probes_grid.png`, `openthebox_pca_manifold.png`, `openthebox_tcav.png`, `openthebox_cross_summary.png`) + `openthebox_summary.json` and the hand-written report at `experiments/meta-report/openthebox.html` (10 sections, 8-row verdict table). The earlier `meta_report_xai*.html` reports cover only setup C2 — `openthebox.html` is the cross-experiment superset.

### Stabilisation experiments (`scripts/stabilization/`)

Two compartmentalised ablations that do NOT touch `src/red_within_blue/`. Each writes its own `metrics.npz` + `summary.json` under `experiments/stabilization/<name>/`. Full spec at `docs/08-stabilization-experiments.md`; per-script README at `scripts/stabilization/README.md`; rendered HTML report at `experiments/stabilization/stabilization_report.html`.

- **`twin_critic_experiment.py` (EXP-A)** — Four variants of the blue CTDE critic on `pair-cooperate-coop`: A0 (current MC baseline), A1 (TD(0) + target-net), A2 (TD(0) + twin-Q), A3 (TD(0) + twin-Q + target-net). Answers whether bootstrap + stabilisers would have beaten the Monte-Carlo switch we already made. Smoke runs in ~3 s per variant (`--smoke`); full run is 15 000 eps × 5 seeds × 4 variants ≈ 40 min wall-clock (≈ 10 min per variant in parallel).

- **`offpolicy_red_experiment.py` (EXP-B)** — Trains joint red against a frozen blue from `experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz` at matched env-step budgets. B0 = on-policy REINFORCE, B1 = Double-DQN with 50 k replay, twin-Q, Polyak target. Blue is frozen in both variants so any Δ(blue reward) at matched env-steps is attributable to the red trainer.

```bash
# smoke — ~3 s per variant
python scripts/stabilization/twin_critic_experiment.py --variant A0 --smoke
python scripts/stabilization/offpolicy_red_experiment.py --variant B0 --smoke

# full (all six variants in parallel on a 6+ core machine, ~45 min wall-clock)
for V in A0 A1 A2 A3; do
  python scripts/stabilization/twin_critic_experiment.py --variant $V &
done
for V in B0 B1; do
  python scripts/stabilization/offpolicy_red_experiment.py --variant $V &
done
wait

# render the comparison figures + HTML report
python scripts/stabilization/render_report.py
```

**2026-04-20 result (both experiments landed clean negative results).** EXP-A: A0 (MC baseline) won at +2.37 ± 0.03; A1/A2/A3 all collapsed to +0.36–+0.39 with |loss| p99 in the thousands — adding SAC-style stabilisers made the critic *worse*. EXP-B: B1 (off-policy DQN) produced a *weaker* red than B0 (on-policy REINFORCE) at every env-step horizon, and took 20× the wall-clock. Decisions: close the `bubbly-strolling-puddle` plan, keep MC returns + on-policy REINFORCE as the production path, don't port SAC. Full writeup in `experiments/stabilization/stabilization_report.html`.
