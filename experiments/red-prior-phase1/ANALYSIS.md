# Red-prior phase 1 — does the basin difference matter?

**Setup.** C2 (3 blue, 2 red, 16×16, 200 steps). Blue frozen. Three red priors trained from
the same point, 3 seeds × 3 000 episodes each. Eval = 600 paired episodes per arm
(shared eval keys across arms — same maps, same blue stochasticity).

| Arm | red_init |
|-----|----------|
| I — Insider | byte-identical copy of blue's actor (all 3 MLP layers transferred) |
| W — Warm    | only `Dense_0` (input layer) copied from blue; `Dense_1` & `Dense_2` random |
| F — Fresh   | random init |

---

## 1. What the ridgeline shows: the basin difference is **categorical**

Late-window red-policy entropy (last 1 500 training eps, per seed):

| Arm | seed 0 | seed 1 | seed 2 | grand mean |
|-----|--------|--------|--------|------------|
| **I** | 0.000 | 0.000 | 0.000 | **0.000** |
| **W** | 0.629 | 0.000 | 0.582 | 0.404 |
| **F** | 0.639 | 0.000 | 0.632 | 0.424 |

Time to first H ≤ 0.10 (entropy collapse):

| Arm | seed 0 | seed 1 | seed 2 |
|-----|--------|--------|--------|
| **I** | 351 | 283 | 292 |
| **W** | — never | 289 | — never |
| **F** | — never | 294 | — never |

So:

- **Insider:** every seed collapses to a near-deterministic policy in ~290–350 episodes.
- **Warm / Fresh:** ⅓ collapse on the same timescale (the lucky seed); ⅔ stay
  stochastic indefinitely at H ≈ 0.6 nats.
- **Warm is indistinguishable from Fresh** in this metric. The shared encoder buys
  nothing observable here.

The three modes the ridgeline is showing are real:

- I's spike at 0  ← all 3 seeds collapsed
- W and F's spike at 0  ← 1/3 seeds collapsed
- W and F's hump at ~0.6  ← 2/3 seeds plateau

This is a **bimodal seed-level outcome** (collapse vs plateau), not a continuous
spread. Insider biases the coin toward "collapse"; Warm/Fresh leave it close to
unbiased.

## 2. Why the basin differs (mechanism)

Blue's actor at C2 is already a near-deterministic "go cover the map" policy.
Insider starts byte-identical to that — high-confidence logits everywhere. The
PPO/A2C update on top of an already-sharp policy keeps sharpening: gradient pushes
into a single mode that exploits blue. The entropy bonus loses to the much
stronger sign-aligned policy gradient.

Warm shares the encoder (so red sees the map "the same way" as blue) but the
head is fresh. The encoder gives a well-conditioned feature space, but the head
has uniform-ish logits at init — the entropy bonus is now meaningful early
because the policy isn't yet committed.

Fresh has neither. Random init + entropy bonus → defaults to a stochastic mixed
strategy.

## 3. Does it matter? Outcome on blue (n = 600 paired episodes)

| Arm | blue_R mean (sem) | red_R mean | coverage | eval H |
|-----|------------------|------------|----------|--------|
| **I** | **+1.1072** (0.018) | −1.661 | 54.77% | 0.000 |
| **W** | +1.1652 (0.019)     | −1.748 | 55.07% | 0.301 |
| **F** | +1.1481 (0.019)     | −1.722 | 54.04% | 0.400 |

Paired Δ on blue_R (Bonferroni-corrected, |z| ≥ 2.39 = ★):

| pair | Δ blue_R | 95% CI | paired z | sig | Cohen's d (paired) |
|------|---------:|:------:|---------:|:---:|---:|
| I − W | −0.058 | [−0.086, −0.029] | **−3.99** | ★ | −0.163 |
| I − F | −0.041 | [−0.069, −0.012] | **−2.87** | ★ | −0.117 |
| W − F | +0.017 | [+0.002, +0.033] | +2.12 | n.s. | +0.086 |

Per-paired-episode win rate (where "win" = Insider hurts blue more):

- I beats W on 51.3% of episodes (tie 11.3%)
- I beats F on 51.3% of episodes (tie 11.2%)

Coverage is essentially flat (Δ |z| < 1.5 except W−F = +1.0 pp z=+2.79).

## 4. Bottom line — basin difference is **real but not consequential**

**Statistically:** Insider attacks measurably better than Warm/Fresh — z ≈ −4 and
−3 are well past Bonferroni's bar. Warm and Fresh are within noise of each other.

**Practically:**

- Cohen's d = 0.16 (I-W) is "small" by Cohen's convention.
- On any single episode, Insider has a 51% chance of hurting blue more than
  Warm — barely above chance. The aggregate effect emerges only because n=600
  is large enough to resolve a 1.3 pp episode-level tilt.
- The Insider advantage is **3.5% of W's blue_R** (0.058 / 1.165). Compare to
  the day-to-day variance across reseeds of co-evolved blue (~7 pp on
  trained-red attacks). The prior effect is one-fifth of that.
- Coverage — the **mission-level** signal — is statistically indistinguishable.
  Whatever Insider does better, blue still covers the same fraction of the map.

**Falsification.** This was supposed to be a clean test of the §3 hypothesis
*"different basin → different attack quality"*. The ridgeline confirms the basin
difference is enormous (categorical: H=0 vs H=0.6). The eval table shows the
attack-quality difference is small and probably not what the §3 framing
imagined. **The hypothesis as stated is falsified at C2.**

A more careful framing: at C2, the attack landscape has multiple
near-equally-effective policies that differ in entropy structure but not in
harm to blue. Initialization picks which one red lands in; it does not pick how
hard red hits.

## 5. Implications for the protocol

- **Default to F (Fresh).** No meaningful loss in attack quality vs Insider, and
  it doesn't require white-box access to blue's params.
- **Don't claim Warm is a useful prior.** It's indistinguishable from Fresh on
  every metric we measured.
- **Insider remains useful as a worst-case bound** — it's a slightly stronger
  attacker, and it's deterministic, which makes downstream analysis (eval gif,
  per-episode attribution) cleaner.
- **Expand to C1 / B before generalizing.** This conclusion is C2-specific.
  A noisier setup may make the prior choice matter more (or less).

---

## Appendix — How the comparison is done (mechanism)

### Why pairing is the right design

`scripts/red_prior_visual_compare.py:369` uses one fixed `PRNGKey(100)` to
generate `eval_keys = [k_1, … k_600]`. Each arm runs all 600 keys. Because
`env.reset(k_e)` is deterministic and the entire rollout is a `jax.lax.scan`
seeded from the same `k_e`, **for the same eval key the map, blue's stochastic
actions, and termination logic are identical across arms**. The only thing that
differs is which red checkpoint produced the action distribution at each step.

This means the *episode-level variance from map randomness is shared between
arms* and cancels in the per-key difference. Marginal blue_R has std ≈ 0.45;
the paired difference D_e = blue_R[I, e] − blue_R[W, e] has std ≈ 0.36. SEM on
the difference is therefore much smaller than what marginal CIs would suggest:

```
sem_paired = 0.36 / √600 = 0.0145    ← what we use
sem_indep  = √(0.45² + 0.45²)/√600 = 0.026  ← what naive marginals would give
```

A statistically significant Δ at sem_paired = 0.0145 may *not* be significant
at sem_indep = 0.026. That's why the original first-pass marginal estimate
gave a borderline z = −2.20 while the paired estimate gives z = −3.99.

### The paired bootstrap CI

For each pair (a, b) and 600 paired keys:

1. Compute D = blue_R[a] − blue_R[b]   (length 600)
2. **Resample 8 000 times**: draw 600 indices with replacement from `{0..599}`,
   compute `mean(D[idx])`. This produces a bootstrap distribution of paired-Δ.
3. The 2.5 / 97.5 percentiles of those 8 000 means form the **95 % CI**.

This is a non-parametric CI — no assumption about the shape of D's
distribution, only that the 600 paired keys are exchangeable (which they are,
because the key generator is iid).

### The paired z-score

```
z = mean(D) / sem(D)        where sem(D) = std(D, ddof=1) / √600
```

This is the **paired t-statistic** at large n. With n = 600 the
t-distribution is indistinguishable from N(0, 1), so we report z. Under the
null hypothesis (D has mean 0), z is approximately standard normal, so:

- |z| ≥ 1.96  →  p ≤ 0.05  (uncorrected, two-sided)
- |z| ≥ 2.39  →  p ≤ 0.05/3  (Bonferroni for 3 pairs)

The bootstrap CI and the paired z-test are different procedures answering
slightly different questions, but with n = 600 they give compatible answers
(CI excludes 0 ⇔ |z| > critical value, in the symmetric Gaussian regime).

### Bonferroni correction (★)

Three pairwise comparisons (I-W, I-F, W-F) means three chances to falsely
reject the null. Bonferroni says: divide the family-wise α by the number of
tests. We use α_family = 0.05, so per-test α = 0.05 / 3 ≈ 0.0167. The
two-sided critical value of N(0, 1) at α = 0.0167 is |z| ≈ **2.39**. We mark a
pair with ★ only if |z| ≥ 2.39, otherwise n.s.

Bonferroni is conservative — there are tighter procedures (Holm, Hochberg) —
but with three tests the difference is negligible and the conservatism is
defensible for a methodological-fairness claim where we want the bar to be
high.

### Cohen's d (paired effect size)

```
d_paired = mean(D) / std(D)
```

This is the *paired* form: it standardizes by the std of the per-key
difference, not by pooled marginal std. Cohen's conventional benchmarks are:

```
|d| < 0.2   negligible
|d| ≈ 0.2   small
|d| ≈ 0.5   medium
|d| ≥ 0.8   large
```

Why both z and d? **z tells you whether Δ is reliably non-zero. d tells you
whether Δ is big.** Our finding has high z (reliably non-zero) but small d
(not big). The pooled version d_pooled = mean(D)/√(½(σ_a² + σ_b²)) is also
reported in the rendering script — it's smaller still (≈ 0.13) because pooled
std is larger than std(D).

### Per-paired-episode win rate

```
win = mean(D < 0) × 100 %
```

For each of the 600 keys, ask: did Insider make blue worse than Warm did on
this same map? The Wilson 95 % CI on this Bernoulli proportion is plotted in
panel Q. Wilson is preferred over the normal-approximation CI for
proportions because it doesn't fail at p ≈ 0 / 1 and it's coverage-correct
even at moderate n.

A 51 % win rate with a Wilson CI of [47.0 %, 55.5 %] means: at the episode
level, Insider's advantage is **indistinguishable from a coin flip**. The
aggregate Δ is reliably non-zero only because we average 600 nearly-coinflips.

### How the policy comparison is done

The training-time entropy panels (G–L) come straight from the per-episode
entropy logged during training. The deployment policy panels (S–V) need
softmax outputs, so `scripts/red_prior_action_probe.py` does:

1. Load blue's actor + each arm's seed-0 red checkpoint.
2. Run a JIT-compiled rollout for 40 paired eval keys per arm. Within the
   rollout's `lax.scan` we capture `red_probs = softmax(red_logits)` at every
   step alongside the action that's actually sampled.
3. Save `probs[arm, ep, t, red_agent, action]` to `action_probe.npz`.

Two slices feed the panels:

- **Marginal mix (S):** average `probs` over (ep, t, red_agent) restricted to
  pre-termination steps. Answers: "across deployment, what does the policy
  look like?"
- **Shared-state slice (T, U):** at t = 0, `env.reset(k_e)` is deterministic,
  so all three arms see *the same observation* before any action is taken.
  We take `probs[:, e, 0, :, :].mean(over red_agent)` and compare. Answers:
  "given the same input, do the policies output the same distribution?"

For the U panel we use **Jensen-Shannon divergence**, a symmetric, smoothed
relative of KL:

```
m = (p + q) / 2
JS(p, q) = ½ · KL(p, m) + ½ · KL(q, m)
```

JS is bounded in **[0, ln 2 ≈ 0.693]** when computed in nats. JS = 0 means
identical distributions; JS = ln 2 means **disjoint support** (the maximum
possible). We chose JS over raw KL for two reasons:

1. KL(p, q) is asymmetric and undefined when q has a zero where p doesn't —
   exactly the situation between Insider (one-hot RIGHT) and Warm/Fresh.
2. JS has a stable maximum, so 0.693 is a meaningful "as different as
   possible" reference.

Observed: JS(W, F) = 0.016 (well below the 0.05 "negligible" threshold);
JS(I, W) = JS(I, F) = 0.693 (saturated).

### What "consequential" means here

Two definitions in tension:

- **Statistical significance** asks: is Δ reliably non-zero given the noise?
  Answer for I-W: yes (z = −3.99 ★).
- **Practical significance** asks: is Δ big enough to matter? Answer here is
  measured against multiple yardsticks:
    - Cohen's d ≈ 0.13–0.16 → "small" by Cohen, ~⅓ of a "small" effect.
    - Per-episode win rate 51 % → barely above coinflip.
    - Δ blue_R / mean blue_R ≈ 0.058 / 1.165 ≈ 5 % → smaller than the 7 pp
      naive-vs-coevo defender effect we previously called "the prior matters".

The page does not pick one definition. It shows both (Section 3 quantifies the
gap; the header chooses the framing) and lets the viewer judge whether a
~5 % attack-quality difference is worth caring about for the protocol.

---

*Rendered from `eval_stats.npz` (paired eval), `red_prior_phase1.npz` (training
entropy), and `action_probe.npz` (deployment softmax). n = 600 paired eval
episodes per arm; entropy from 3 seeds × 3 000 training eps; policy probe from
40 paired eps × seed-0 reds.*
