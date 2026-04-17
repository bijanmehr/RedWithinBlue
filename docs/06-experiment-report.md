# Experiment Design, Methods, and Findings

**Date:** 2026-04-15 (design approved) | Updated: 2026-04-17
**Status:** Stage 1 sweeps complete, Stages 2-3 pending
**Scope:** Plan A — proof of learning from 1 agent to 4-agent cooperative exploration. Plan B (10+ agents, 64x64, adversarial) deferred until all Plan A gates pass.

---

## 1. Experiment Design

### 1.1 Scope

A reproducible experiment pipeline proving that agents can learn cooperative grid exploration, scaling from a single agent on a small grid to four agents on a larger grid with obstacles. Pure cooperative — no adversarial agents, no game theory.

### 1.2 Success Criteria

| Stage | Gate |
|-------|------|
| Stage 1 (1 agent, 8x8) | Action distribution statistically different from uniform (chi-squared, p < 0.05). Coverage reaches 95%+ consistently. |
| Stage 2 (2 agents, 8x8) | Both agents learn non-uniform policies. Team coverage > single-agent coverage. |
| Stage 3 (4 agents, 16x16) | Team achieves 80%+ coverage. Communication graph connected >= 90% of timesteps. |

### 1.3 Dual-Track Approach

Two parallel algorithmic tracks run at every stage:

**Track 1 — Policy Gradient (bottom-up):** Raw PG -> subtract baseline -> add critic -> stability tricks (GAE, clipping, entropy, grad clip). Each layer is a checkpoint; move up only when hitting a concrete wall.

**Track 2 — Value-Based (bottom-up):** Tabular Q-learning -> DQN -> target network + replay buffer -> Double DQN, dueling. Same checkpoint discipline.

### 1.4 Three Stages

| Property | Stage 1 | Stage 2 | Stage 3 |
|----------|---------|---------|---------|
| Purpose | Prove reward is learnable | Prove multi-agent scaling | Prove communication matters |
| Grid size | 8x8 | 8x8 | 16x16 |
| Num agents | 1 | 2 | 4 |
| wall_density | 0.0 | 0.0 | 0.1 |
| max_steps | 100 | 100 | 256 |
| obs_radius | 5 | 5 | 5 |
| comm_radius | 4.0 | 4.0 | 6.0 |
| obs_dim | 255 | 255 | 255 |
| global_state_dim | N/A | 137 | 537 |
| Rollout fn | `collect_episode` | `collect_episode_multi` | `collect_episode_multi` |
| Init | Random | Warm-start from S1 | Warm-start from S2 |

**Key invariant:** obs_dim = 255 across all stages. This is what makes weight transfer possible.

### 1.5 Reward Design

Single exploration reward, everything else is hard enforcement:

```
reward_i = cells_discovered_by_agent_i / total_discoverable_cells
```

- Normalized to [0, 1] by construction
- No revisit penalty, no time penalty, no connectivity reward
- Multi-agent: each agent gets credit only for cells it personally discovers first

### 1.6 Guardrails (Hard Enforcement)

| Guardrail | Mechanism | Active |
|-----------|-----------|--------|
| Wall collision | Action mask via logit manipulation (`logits + where(mask, 0, -1e9)`) | Always |
| Boundary collision | Same action mask | Always |
| Connectivity preservation | Override to STAY if move would isolate agent > X steps (X = max(3, max_steps // 20)) | Stages 2-3 |

### 1.7 Core Metrics

**Logged every episode:**

| Metric | What It Tells You |
|--------|-------------------|
| Coverage (%) | Fraction of discoverable cells visited by the team |
| Action distribution | Per-action probability over eval episodes |
| Episode return | Total reward per episode |
| Steps to X% coverage | Efficiency (X = 50, 75, 90, 95) |
| Graph connectivity (%) | Fraction of timesteps with fully connected comm graph |
| Per-agent coverage contribution | Which agents discover which cells |

### 1.8 Statistical Rigor

- **10 random seeds** per experiment (environment + network initialization)
- **Chi-squared** on action distribution vs. uniform (proof of learning)
- **Welch's t-test** on final coverage (learned vs. random)
- **p < 0.05** with **Bonferroni correction** across stages
- **Explained variance** of value predictions for critic quality
- All seeds, hyperparams, and configs logged as JSON alongside checkpoints
- Every figure regenerable from saved logs

### 1.9 Baselines

| Baseline | Purpose |
|----------|---------|
| Uniform random | Lower bound — does learning beat chance? |
| Greedy heuristic (nearest unvisited) | Upper bound sanity check |

Both run with the same 10 seeds.

### 1.10 Figure Types Per Stage

Each stage produces a multi-panel summary figure:

1. Learning curve — coverage vs. episode, 95% CI bands
2. Action distribution — bar chart with significance stars
3. Visit heatmap — cell visit frequency
4. Graph health timeline — connected/disconnected per step (Stages 2-3)
5. Coverage contribution — per-agent breakdown (Stages 2-3)
6. Bias-variance panel — train vs. eval gap, explained variance
7. Transfer effectiveness — pre vs. post fine-tuning curves (Stages 2-3)

Visual style: Tufte + Nature blend — serif fonts, lettered panels, no gridlines, muted palette, significance annotations, colorblind-safe.

---

## 2. Architecture Decisions

### 2.1 Current Best Configuration

From hyperparameter sweep (Actor-Critic, 500 episodes, 3 seeds each):

| Parameter | Best Value | Coverage | Runner-up |
|-----------|-----------|---------|-----------|
| Method | Actor-Critic | 0.4705 | Baseline (0.4120) |
| Network | 128x2 (21K params) | 0.4705 | 64x2 (0.4545) |
| Learning rate | 3e-4 | 0.4705 | 5e-4 (0.4703) |
| Gamma | 0.90 | 0.4811 | 0.95 (0.4791) |
| PG vs DQN | PG (Actor-Critic) | 0.4213 | DQN best (0.3206) |

### 2.2 The Deeper Network Problem

Larger networks performed worse — but were tested with zero regularization:

| Architecture | Params | Coverage | Std (stability) |
|-------------|--------|---------|-----------------|
| 128x2 | 21K | **0.4705** | **0.0055** |
| 64x2 | 6.5K | 0.4545 | 0.0055 |
| 128x3 | 38K | 0.4537 | 0.0287 (5x worse) |
| 256x2 | 75K | 0.4377 | 0.0292 (5x worse) |
| 256x3 | 141K | 0.4222 | 0.0016 |

**Caveat:** These ran with zero gradient clipping, zero weight decay, no LayerNorm, raw single-episode REINFORCE, and vanilla Adam. The conclusion "bigger networks are worse" is premature. Deeper networks need gradient clipping, LayerNorm, and better optimization to train stably.

### 2.3 Entropy Sweep Findings

Entropy coefficient sweep on best config (Actor-Critic, 128x2, lr=3e-4, gamma=0.90, 500 episodes, 3 seeds):

| Entropy Coeff | Stable Mean | Stable Std | STAY % | Chi2 |
|:---:|:---:|:---:|:---:|:---:|
| **0.0** | **0.490** | **0.014** | 5.0% | 21,318 |
| 0.001 | 0.486 | 0.003 | 5.8% | 19,149 |
| 0.005 | 0.459 | 0.006 | 10.8% | 7,991 |
| 0.01 | 0.438 | 0.009 | 14.1% | 3,235 |
| 0.02 | 0.425 | 0.018 | 16.6% | 1,073 |
| 0.05 | 0.415 | 0.024 | 18.5% | 223 |
| 0.1 | 0.413 | 0.019 | 19.2% | 69 |

**Entropy hurts coverage monotonically.** Stable mean drops from 0.490 to 0.413 as entropy increases. The standard PPO default of ent_coef=0.01 yields 0.438 — a 10% relative drop.

**Why entropy is counterproductive in our environment:**

1. The exploration reward already prevents action collapse (STAY suppressed to 5% without entropy)
2. Small action space (5 actions) reduces collapse risk — entropy bonuses originate from large/continuous action spaces
3. Entropy fights the useful learning signal: the agent correctly learned "never stay still," and entropy pushes STAY back toward 20%

**Key implication:** Standard algorithm defaults don't transfer to our environment. We should draw techniques from PPO (clipped updates, GAE, mini-batches) without adopting its hyperparameter defaults. Each component must be validated.

---

## 3. Learning Detection

### 3.1 Detection Framework

To confidently claim "the agent learned," require ALL of:

1. **Coverage vs baseline PASS** — Mann-Whitney p < 0.05, Cohen's d > 0.2
2. **Learning trend PASS or WEAK PASS** — late episodes > early episodes
3. **Action distribution non-degenerate** — no single action > 0.6, STAY < 0.3

Supporting evidence (strengthens claim, not required): stable coverage std < 0.10, steps-to-50% faster than random, connectivity > 80% (multi-agent), explained variance > 0.3 (Actor-Critic).

### 3.2 Tier 1: Statistical Tests

#### Coverage vs Random Baseline (Mann-Whitney U + Cohen's d)

Run trained and random policies for 20 episodes each. One-sided Mann-Whitney U test (H1 = learned > baseline). Cohen's d for practical significance.

| Verdict | Condition |
|---------|-----------|
| PASS | p < 0.05 AND d > 0.5 |
| WEAK PASS | p < 0.05 AND d > 0.2 |
| MARGINAL | p < 0.05 AND d <= 0.2 |
| INCONCLUSIVE | p >= 0.05, learned > baseline |
| FAIL | learned <= baseline |

Implementation: `training/stats.py` — `coverage_vs_baseline()`

#### Learning Trend (Early vs Late Episodes)

Split training curve into first 50 and last 50 episodes. Mann-Whitney on late > early, Spearman rank correlation on smoothed curve.

| Verdict | Condition |
|---------|-----------|
| PASS | improvement > 0.05 AND p < 0.05 |
| WEAK PASS | improvement > 0.02 AND p < 0.05 |
| TREND | improvement > 0.05 AND rho > 0.3 (but p >= 0.05) |
| FAIL | improvement <= 0.01 |

Implementation: `training/stats.py` — `learning_trend()`

#### Action Distribution Chi-Squared

Bin all eval actions into 5 categories (STAY, UP, RIGHT, DOWN, LEFT). Chi-squared goodness-of-fit vs. uniform. **Caveat:** With many samples, even trivial bias produces p < 0.05, so chi-squared rejection alone is necessary but not sufficient.

**Actual results:**

| Experiment | STAY | UP | RIGHT | DOWN | LEFT | Chi2 | p-value |
|-----------|------|-----|-------|------|------|------|---------|
| Random baseline | ~0.20 | ~0.20 | ~0.20 | ~0.20 | ~0.20 | ~0 | >0.05 |
| PG AC ent=0.0 (best) | 0.050 | 0.228 | 0.254 | 0.230 | 0.238 | 21,318 | 0.0 |
| PG AC ent=0.01 | 0.141 | 0.215 | 0.219 | 0.211 | 0.214 | 3,235 | 0.0 |
| PG AC ent=0.1 | 0.192 | 0.204 | 0.202 | 0.201 | 0.201 | 69.4 | 3.1e-14 |
| Best Evo agent | **[PENDING]** | | | | | | |

### 3.3 Tier 2: Behavioral Tests

#### Stable Coverage Plateau

Average coverage over last 50 training episodes across seeds.

| Signal | Random | Weak Learning | Strong Learning |
|--------|--------|--------------|-----------------|
| Stable mean | 0.30-0.40 | 0.45-0.55 | 0.60-0.80+ |
| Stable std | 0.10-0.15 | 0.08-0.12 | 0.03-0.06 |

**Actual results:**

| Experiment | Stable Mean | Stable Std | Seeds |
|-----------|-------------|-----------|-------|
| PG AC ent=0.0 (best) | 0.490 | 0.014 | 3 |
| PG AC ent=0.001 | 0.486 | 0.003 | 3 |
| PG AC ent=0.01 | 0.438 | 0.009 | 3 |
| PG AC ent=0.1 | 0.413 | 0.019 | 3 |
| RL sweep best DQN | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Evo best agent | **[PENDING]** | **[PENDING]** | **[PENDING]** |

#### Coverage Efficiency (Steps to Threshold)

Track per-step coverage, record when 50% and 75% thresholds are first reached.

| Threshold | Random Baseline | Learned | Speedup |
|-----------|----------------|---------|---------|
| 50% coverage | ~50-80 steps | **[PENDING]** | **[PENDING]** |
| 75% coverage | often never | **[PENDING]** | **[PENDING]** |

Implementation: `training/metrics.py` — `compute_steps_to_coverage()`

#### Connectivity Maintenance (Multi-Agent)

Fraction of steps with fully connected comm graph. Compare guardrail override rate: learned agents should rarely trigger the guardrail (voluntary connectivity), random agents require frequent overrides.

| Experiment | Connectivity % | Guardrail Override % |
|-----------|---------------|---------------------|
| Stage 2 random + guardrail | **[PENDING]** | **[PENDING]** |
| Stage 2 learned + guardrail | **[PENDING]** | **[PENDING]** |
| Stage 3 random + guardrail | **[PENDING]** | **[PENDING]** |
| Stage 3 learned + guardrail | **[PENDING]** | **[PENDING]** |

#### Explained Variance (Actor-Critic)

`EV = 1 - Var(returns - V(s)) / Var(returns)`. EV > 0.5 means the critic learned a meaningful value mapping.

| EV Range | Interpretation |
|----------|---------------|
| < 0.1 | Critic hasn't learned |
| 0.1 - 0.3 | Weak value learning |
| 0.3 - 0.6 | Moderate |
| > 0.6 | Strong |

| Experiment | EV (early) | EV (late) |
|-----------|-----------|----------|
| Stage 1 L3 (AC) | **[PENDING]** | **[PENDING]** |
| RL sweep best AC | **[PENDING]** | **[PENDING]** |

### 3.4 Negative Controls

These scenarios should FAIL our tests. If they pass, our tests are broken.

| Control | Expected Outcome |
|---------|-----------------|
| Random policy (`randint(key, (), 0, 5)`) | FAIL all Tier 1 tests (p > 0.05, d ~ 0, flat trend) |
| Always-STAY policy | Chi-squared rejects (non-uniform) but coverage ~ 0, Mann-Whitney FAIL |
| Untrained network (random weights) | Approx-uniform actions, coverage ~ random baseline, Mann-Whitney p > 0.05 |

Always-STAY illustrates why chi-squared alone is insufficient: non-uniform does not mean useful.

---

## 4. Knowledge Transfer

### 4.1 Curriculum Warm-Start (Stage-to-Stage)

Trained actor parameters from Stage N initialize Stage N+1. Optimizer state is always reset.

| Property | Detail |
|----------|--------|
| Function | `transfer_actor_params()` in `training/transfer.py` |
| What transfers | Full actor param pytree (all Dense layer kernels + biases) |
| What resets | Optimizer state (fresh `optimizer.init(params)`) |
| Format | `.npz` checkpoint with flattened keys |
| Obs compatibility | All stages share obs_dim=255 (obs_radius and msg_dim constant) |

**Transfer path:**

Stage 1 (1 agent, 10x10) --> Stage 2 (2 agents, 10x10) --> Stage 3 (4 agents, 18x18, 10% walls)

At each arrow: load checkpoint, `transfer_actor_params`, fresh optimizer. Critic is re-initialized (global_state_dim changes between stages).

### 4.2 Parameter Sharing (Multi-Agent)

All agents in Stages 2 and 3 share a single actor network. The same `policy_fn` is applied to each agent's local observation independently. Gradients are computed over concatenated experience from all agents and applied as a single update. This provides implicit knowledge transfer: Agent 0 exploring the north generates gradients that also improve Agent 1's behavior in the south.

### 4.3 CKA Analysis (Representation Similarity)

Centered Kernel Alignment measures how similar hidden-layer representations are between two networks.

| Property | Detail |
|----------|--------|
| Function | `compute_cka()` in `training/transfer.py` |
| Input | Two [N, hidden_dim] activation matrices |
| Output | Scalar in [0, 1] (1 = identical representations) |
| Extraction | `extract_hidden_features()` — activations after Dense_1 ReLU |

| Comparison | Expected CKA |
|-----------|-------------|
| Warm-start S2 vs S1 source | High (>0.7) — S2 preserves S1 representations |
| Cold-start S2 vs S1 source | Low (<0.3) — independent learning finds different features |
| Early training vs late training | Increasing — network settling into stable representations |

| Comparison | CKA Score |
|-----------|----------|
| S1 final vs S2 warm (epoch 0) | **[PENDING]** |
| S1 final vs S2 warm (epoch 500) | **[PENDING]** |
| S1 final vs S2 cold (epoch 500) | **[PENDING]** |

### 4.4 Curriculum Skipping

Tests whether the intermediate Stage 2 step is necessary.

**Key question:** Does S2 teach multi-agent coordination that S1 alone can't provide? If S1->S3 performs nearly as well as S1->S2->S3, the curriculum step is unnecessary overhead.

### 4.5 Transfer Methods Summary

| Method | Transfer Type | Scope | Expected Impact |
|--------|-------------|-------|----------------|
| Curriculum warm-start | Weight initialization | Cross-stage | High — faster convergence, higher plateau |
| Parameter sharing | Implicit (multi-agent) | Within-episode | High — Nx sample efficiency |
| Evolutionary elite survival | Weight preservation | Cross-generation | Medium — preserves best solutions |
| Evolutionary mutation | Weight perturbation | Cross-generation | Medium — explores neighborhood of good solutions |
| Grid curriculum (evo) | Weight transfer to new env | Cross-grid-size | Open question — does local policy generalize? |

---

## 5. Results

### 5.1 Stage 1 Results

#### Coverage vs Random Baseline

| Experiment | Learned Coverage | Baseline Coverage | p-value | Cohen's d | Verdict |
|-----------|-----------------|-------------------|---------|-----------|---------|
| Stage 1 Layer 1 (REINFORCE) | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Stage 1 Layer 2 (+ baseline) | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Stage 1 Layer 3 (Actor-Critic) | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

#### Learning Trend

| Experiment | Early Mean | Late Mean | Improvement | Spearman rho | Verdict |
|-----------|-----------|----------|-------------|-------------|---------|
| Stage 1 L1 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Stage 1 L2 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Stage 1 L3 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

#### Method Comparison

| Method | Coverage | Time | Signal |
|--------|---------|------|--------|
| REINFORCE | **[PENDING]** | **[PENDING]** | Baseline PG |
| REINFORCE + baseline | **[PENDING]** | **[PENDING]** | Should be >= REINFORCE |
| Actor-Critic | **[PENDING]** | **[PENDING]** | Should be >= baseline |

#### PG vs DQN

| Method | Coverage | Action Distribution |
|--------|---------|---------------------|
| Best PG (AC) | **[PENDING]** | **[PENDING]** |
| Best DQN | **[PENDING]** | **[PENDING]** |
| Random | **[PENDING]** | [0.2, 0.2, 0.2, 0.2, 0.2] |

#### Evolutionary Results

Smoke test (4 agents, 3 generations, levels 0-1):

| Level | Gen 1 Mean | Gen 3 Mean | Gen 3 Best | Signal |
|-------|-----------|-----------|-----------|--------|
| 4x4 | 0.6875 | 0.7500 | 0.7500 | Selection improving mean |
| 6x6 | 0.5625 | 0.5938 | 0.8125 | Best agent improving fast |

Full evolutionary run (pop=32, elites=8, 20 gens/level, 64x2 network, mutation_std=0.02):

| Level | Gen 1 Best | Gen 1 Mean | Gen 20 Best | Gen 20 Mean | Mean Improvement |
|-------|-----------|-----------|------------|------------|-----------------|
| 4x4 (steps=20, walls=0.0) | 0.7500 | 0.6906 | 0.7500 | 0.7203 | +4.3% |
| 6x6 (steps=40, walls=0.0) | 0.7500 | 0.6262 | 0.8750 | 0.6746 | +7.7% |
| 8x8 (steps=70, walls=0.05) | 0.6200 | 0.5376 | 0.6944 | 0.5961 | +10.9% |
| 10x10 (steps=100, walls=0.1) | 0.5417 | 0.4647 | 0.6650 | 0.5393 | +16.1% |
| 12x12 (steps=150, walls=0.1) | 0.6036 | 0.4788 | 0.6429 | 0.5332 | +11.4% |

**Analysis:** Selection pressure improves population mean consistently (4-16% across levels). Best agents reach 0.66-0.87 coverage on trained grid sizes. Improvement is strongest on the 10x10 level where the task is hardest relative to step budget. Total time: ~173 minutes.

Generalization (best agent from final level evaluated on all grid sizes):

| Grid | Coverage (mean +/- std) | Trained On? |
|------|------------------------|-------------|
| 4x4 | 0.7500 +/- 0.0000 | Yes |
| 6x6 | 0.8688 +/- 0.0653 | Yes |
| 8x8 | 0.7547 +/- 0.1487 | Yes |
| 10x10 | 0.6279 +/- 0.1474 | Yes |
| 12x12 | 0.5640 +/- 0.0983 | Yes |
| 16x16 | **[PENDING]** | No (unseen) |
| 20x20 | **[PENDING]** | No (unseen) |

**Analysis:** Coverage degrades gracefully with grid size but variance increases significantly on larger grids (std 0.15 on 8x8/10x10 vs 0.00 on 4x4). The policy generalizes local exploration behavior but struggles with efficient full-grid coverage on larger maps — expected given the fixed obs_radius=1.

#### Evolutionary vs Standard PG

| Method | Coverage (10x10) | Training Time |
|--------|-----------------|---------------|
| Standard PG (best seed) | **[PENDING]** | **[PENDING]** |
| Evolutionary (best agent) | **[PENDING]** | **[PENDING]** |
| Random baseline | **[PENDING]** | - |

### 5.2 Stage 2 Results

#### Warm vs Cold Start (S1 -> S2)

| Metric | Warm Start | Cold Start |
|--------|-----------|------------|
| Coverage (mean +/- std) | **[PENDING]** | **[PENDING]** |
| Episodes to 50% coverage | **[PENDING]** | **[PENDING]** |
| Learning curve slope (first 100 ep) | **[PENDING]** | **[PENDING]** |

#### Coverage vs Random Baseline

| Experiment | Learned Coverage | Baseline Coverage | p-value | Cohen's d | Verdict |
|-----------|-----------------|-------------------|---------|-----------|---------|
| Stage 2 (warm start) | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

#### Learning Trend

| Experiment | Early Mean | Late Mean | Improvement | Spearman rho | Verdict |
|-----------|-----------|----------|-------------|-------------|---------|
| Evo 4x4 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Evo 10x10 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

#### Parameter Sharing

| Metric | Shared Policy | Independent Policies |
|--------|--------------|---------------------|
| Team coverage | **[PENDING]** | (not implemented) |
| Sample efficiency | Higher (Nx data per episode) | Lower |

### 5.3 Stage 3 Results

#### Warm vs Cold Start (S2 -> S3)

| Metric | Warm Start | Cold Start |
|--------|-----------|------------|
| Coverage (mean +/- std) | **[PENDING]** | **[PENDING]** |
| Episodes to 50% coverage | **[PENDING]** | **[PENDING]** |
| Learning curve slope (first 100 ep) | **[PENDING]** | **[PENDING]** |

#### Coverage vs Random Baseline

| Experiment | Learned Coverage | Baseline Coverage | p-value | Cohen's d | Verdict |
|-----------|-----------------|-------------------|---------|-----------|---------|
| Stage 3 (warm start) | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

#### Curriculum Skipping

| Metric | S1 -> S3 (skip) | S1 -> S2 -> S3 (full) |
|--------|-----------------|----------------------|
| Coverage (mean +/- std) | **[PENDING]** | **[PENDING]** |
| Connectivity % | **[PENDING]** | **[PENDING]** |

#### No Curriculum (Three-Way Comparison)

| Metric | S3 Cold | S1 -> S3 | S2 -> S3 |
|--------|---------|----------|----------|
| Coverage | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Connectivity % | **[PENDING]** | **[PENDING]** | **[PENDING]** |

### 5.4 Cross-Stage Comparisons

#### CKA Representation Similarity

| Comparison | CKA Score |
|-----------|----------|
| S1 final vs S2 warm (epoch 0) | **[PENDING]** |
| S1 final vs S2 warm (epoch 500) | **[PENDING]** |
| S1 final vs S2 cold (epoch 500) | **[PENDING]** |

---

## 6. Future Work

### Priority 1: PPO + Core Regularization (High impact, Medium effort)

Add clipped surrogate objective with GAE and mini-batch training to replace single-episode REINFORCE. Key components:

- **Clipped surrogate:** `min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)` with eps=0.2
- **GAE:** Exponentially-weighted multi-step advantages with lambda=0.95
- **Mini-batch training:** n_steps=2048, n_epochs=10, mini_batch_size=64

Regularization toolkit (currently absent — likely causing deeper network instability):

- LayerNorm after each Dense layer
- Dropout (0.1) during update epochs only
- Gradient clipping (global L2 norm <= 0.5)
- AdamW (weight_decay=1e-4)
- Running observation normalization (Welford's algorithm, clip to [-5, 5])
- Advantage normalization per mini-batch
- Orthogonal initialization (hidden: gain sqrt(2), policy output: gain 0.01, value output: gain 1.0)
- Linear LR annealing to zero
- Entropy coefficient = 0.0 (validated by our entropy sweep)

Re-test all architectures (64x2 through 256x3) with these additions. Key prediction: 128x3 with regularization should match or exceed 128x2 without.

### Priority 2: MAPPO — Centralized Critic (High impact for multi-agent, Medium effort)

Decentralized actors (local obs -> action logits) with a centralized critic (global state -> value). Parameter sharing across actors remains standard. Recommended critic input: agent obs + global stats (coverage %, connectivity %, team centroid, step number) for a compact ~50-dim input rather than naive observation concatenation.

### Priority 3: Enhanced DQN (Medium impact, Low-Medium effort)

Add Prioritized Experience Replay (sum tree, proportional to TD error), Double DQN (online net for action selection, target net for evaluation), and n-step returns. May close the gap with PG methods.

### Priority 4: Advanced Architecture (Uncertain impact, High effort)

- **GNN communication:** Message passing over dynamic comm graph for learned agent coordination
- SimbaV2-style hyperspherical normalization
- Transformer-based sequential PPO (TSPPO)

### Target Configuration: 10 Agents, 32x32

| Component | Design |
|-----------|--------|
| Actor | 128x3, shared params, LayerNorm + Dropout, orthogonal init (38K params) |
| Critic | Centralized, 256x2, global features (~50 dims) |
| Algorithm | PPO/MAPPO, n_steps=2000 (200 steps x 10 agents), 4 epochs, mini_batch=128 |
| Gamma | 0.90 |
| Entropy | 0.0 |
| LR | 3e-4 with linear annealing |
| Gradient clipping | Global L2 norm <= 0.5 |
| Weight decay | 1e-4 (AdamW) |

Training phases: (1) PPO + reg sweep on 1 agent 10x10, (2) MAPPO on 2->4 agents, (3) MAPPO on 10 agents 32x32, (4) architecture search at target scale.

---

## Appendix A: Plan B Scope (After Plan A Gates Pass)

**Phase 1: Stronger Communication Enforcement**
- Improve connectivity guardrail beyond binary connected/disconnected
- Penalize graph fragmentation proportionally (partial disconnections)
- Communication-aware reward shaping

**Phase 2: Adversarial Agents**
- Introduce "red" adversarial agents that obstruct exploration
- Mixed cooperative-competitive (blue team explores, red team blocks)
- Game-theoretic equilibrium analysis
- Self-play training for adversarial robustness

**Phase 3: Scaling**
- 10+ agents, grids 32x32 and 64x64
- Communication learning (TarMAC or similar)
- GNN-based policies for graph-native architecture
- Evolutionary/genetic hyperparameter search

---

## Appendix B: All Detection Strategies

| # | Strategy | Type | File | What It Catches |
|---|---------|------|------|----------------|
| 1 | Mann-Whitney + Cohen's d | Statistical | `stats.py` | Coverage above random |
| 2 | Early vs Late trend | Statistical | `stats.py` | Improvement during training |
| 3 | Chi-squared vs uniform | Statistical | `stats.py` | Non-random action selection |
| 4 | Spearman correlation | Statistical | `stats.py` | Monotonic learning curve |
| 5 | Stable coverage plateau | Behavioral | sweep scripts | Consistent final performance |
| 6 | Steps to threshold | Behavioral | `metrics.py` | Exploration efficiency |
| 7 | Connectivity fraction | Behavioral | `gif.py` | Multi-agent coordination |
| 8 | Explained variance | Behavioral | `metrics.py` | Value function quality |
| 9 | Learning curves | Visual | HTML reports | Training dynamics |
| 10 | Visitation heatmaps | Visual | `gif.py` | Spatial exploration patterns |
| 11 | Episode GIFs | Visual | `gif.py` | Qualitative behavior |
| 12 | Method comparison | Comparative | sweep scripts | Relative method strength |
| 13 | PG vs DQN | Comparative | `rl_sweep.py` | Cross-algorithm convergence |
| 14 | Evo vs standard PG | Comparative | `run_evolutionary.py` | Selection pressure value |

---

## Sources

- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [SimbaV2: Hyperspherical Normalization for Scalable Deep RL (ICML 2025)](https://arxiv.org/abs/2502.15280)
- [FP3O: Enabling PPO in Multi-Agent Cooperation with Parameter Sharing](https://arxiv.org/abs/2310.05053)
- [Relative Importance Sampling for Off-Policy Actor-Critic](https://www.nature.com/articles/s41598-025-96201-5)
- [Regularization Effects in Deep RL (2024)](https://arxiv.org/abs/2409.07606)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952)
- [PPO — Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [MAPPO Documentation — MARLlib](https://marllib.readthedocs.io/en/latest/algorithm/ppo_family.html)
- [TSPPO: Transformer-Based Sequential PPO for Multi-Agent Systems (2025)](https://link.springer.com/article/10.1007/s00530-025-02153-1)
- [Optimizing Deep RL through Vectorized Neuroevolution (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25006284)
- [RL Implementation Tips and Tricks — Edinburgh](https://agents.inf.ed.ac.uk/blog/reinforcement-learning-implementation-tricks/)
- [RL Tips and Tricks — Stable Baselines](https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html)
- [Survey on Graph-Based RL for Networked Coordination](https://www.mdpi.com/2673-4052/6/4/65)
- [E2CL: Evolutionary Curriculum Learning (2025)](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_193.pdf)
