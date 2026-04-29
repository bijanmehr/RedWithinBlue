# Adversarial-Model Validation — Results

**Setup.** k ∈ {1, 2} compromised agents on the 16×16 N=5 map, 200 steps,
co-evolved blue + JointRedActor checkpoints. ρ = per-step probability that a
compromised agent executes its trained adversarial action; otherwise it
executes the *nominal* (clean blue actor with `team_id` clamped to 0).
ΔJ = 98.5 (B baseline) − blue ever-known coverage at t=200, in pp.

All numbers below are at n=60 unless stated. Phases 1, 4 ran at n=60 from the
start; Phase 3 was re-run at n=60 after the n=30 result raised the alarm.

## Reframing — what (k, ρ) actually measures

Two channels through which a compromised agent could damage the mission:

1. **Action-stream channel** (the real attack surface). What the compromised
   agent does — where it moves, what it scans, whether it contributes
   misleading observations. This is the only channel a real attacker
   controls.

2. **Team-label exclusion channel** (a *post-detection* defender cost).
   Compromised agents have `team_id=1` and their scans are filtered out of
   blue's belief merge. This presumes blue *already knows* the agent is
   compromised — i.e., it is a model of the cost paid *after* successful
   detection, not an attack channel an adversary can exploit.

The current `(k, ρ)` sweep mixes both: the ρ=0 cells measure pure
post-detection cost, the ρ>0 cells add the action channel on top.

**Implication.** This dataset is best read as a **detector-calibration
instrument**, not a threat model:

- ρ = visibility of the compromised agent's action deviation = the signal
  a behavior-rate detector watches.
- Low ρ → stealthy (hard to detect, behaves like a blue agent that just
  doesn't share scans).
- High ρ → loud (easy to detect by action-distribution KL or entropy).
- The attacker's optimization isn't "max ΔJ at fixed M = k·ρ" — it's
  "minimize ρ subject to enough damage to matter" because higher ρ
  surrenders the agent to detection.

In that framing, the data say: **the action channel is statistically
silent at this scale**, so a behavior-rate defender doesn't have a useful
signal to watch — there is no signature to detect because there is no
detectable damage being done. This connects directly to the existing
finding that direct red→blue uncertainty manipulation peaks at 2–3 cells,
residual ≤1 pp (`project_uncertainty_manipulation_thin.md`).

---

## Headline — sensitivity analysis

We reframe the v2 / v3 closed-form ΔJ models as a **sensitivity analysis on
the budget axes (k, ρ)**, not as a fit to be accepted or rejected. The
question is: *how does mission damage respond to each axis at this scale?*

```
       axis              unit           ΔJ-sensitivity (pp per unit)
       ──────────────────────────────────────────────────────────────
       k   (# compromised)  +1 agent     +5.3   at ρ=0     [α-floor add]
                            +1 agent     +3.3   at ρ=1     [k-step at ρ=1]
       ρ   (per-step rate)  +1.0 unit    −0.3 ± 0.5  at k=1   [span 0]
                            +1.0 unit    −2.4 ± 1.3  at k=2   [defender-favorable]
```

**Mission damage is ~10× more sensitive to k than to ρ at this scale, and
the ρ-axis sign is the *wrong* direction the v3 model assumes** (it's flat
or defender-favorable, not attacker-favorable).

The closed-form models predict ρ to be a damage-amplifier; the data
disagree. Held-out cell:

```
held-out cell (k=2, ρ=1):  observed     = +12.21 ± 0.84 pp
                           v2 (Σρ form) = +15.83  →  z = −4.31
                           v3 (max form)= +16.19  →  z = −4.74
```

Both forms over-predict by ~4 pp. The k=2 ρ-curve is non-monotone with a
defender-favorable basin at ρ ≈ 0.5 (ΔJ = +10.73, **less** damage at
intermediate ρ than at the ρ=0 floor of +14.59).

**This is not (yet) a final falsification.** It is a statement that the v3
closed-form is not the right shape for this system at this scale. Two
follow-ups would either fix the model or refute it more sharply:

1. A *damage-maximizing* trained red (not zero-sum reward) — current red is
   Nash-protective against co-evo blue.
2. A larger-(map, N) regime where the action channel might dominate the
   label-exclusion channel.

Until then, we report ρ as a low-sensitivity axis at this scale and flag
the model as "shape under revision."

---

## Data tables (n=60 each)

### Clean ρ-curves (clamped-team-id nominal, trained_red adversary)

```
  k=1   ρ=0.00  ρ=0.10  ρ=0.25  ρ=0.50  ρ=0.75  ρ=1.00
   ΔJ   +9.25   +10.03  +9.13   +9.79   +9.57   +8.95
   SEM   0.43    0.48    0.60    0.61    0.51    0.49
   slope from OLS:  β₁ = −0.35 ± 0.50  pp/ρ   (CI spans zero)

  k=2   ρ=0.00  ρ=0.10  ρ=0.25  ρ=0.50  ρ=0.75  ρ=1.00
   ΔJ  +14.59  +14.38  +12.97  +10.73  +11.46  +12.21
   SEM   0.98    0.93    0.81    0.67    0.85    0.84
   slope:  non-monotone, dips at ρ=0.5  (Δ = −3.86 pp vs ρ=0 floor)
```

### Adversary-type ablation at (k=2, ρ=1), n=30

```
            stay              ΔJ = +14.79  ±  1.61   ← maximum damage
            nominal_raw       ΔJ = +14.78  ±  1.24
            uniform_random    ΔJ = +14.55  ±  1.11
            trained_red       ΔJ = +13.55  ±  1.25
            nominal_clamped   ΔJ = +12.70  ±  1.04   ← minimum damage
```

The most damaging "attacker" is the one that takes **no action**.
The trained red causes **less** damage than just sitting still. The
co-evolved blue routes around the trained red's predictable corner-camp
niche — so giving the trained red more action mass *helps* blue.

### k-scaling at ρ=1, clean (n=60)

```
   ΔJ(k=1, ρ=1) = +8.95 ± 0.49
   ΔJ(k=2, ρ=1) = +12.21 ± 0.84
   ratio        = 1.36     (linear would be 2.00)         sub-linear
```

### Phase 5 — Co-evolution defense effect (Nash-basin probe)

```
   co-evo blue, k=2, ρ=1  (n=60):  ΔJ = +12.21  ±  0.84
   naive  blue, k=2, ρ=1  (n=30):  ΔJ =  +5.15  ±  0.51
   defense effect                = −7.06  ±  0.98       (z = −7.18)
```

**The naive (non-coevo) blue defends ~7 pp BETTER than the co-evo blue
against the same trained red.** The co-evolved blue has overfit to its
training partner: against the same red it now sees more damage, not less.
This *kills* the "co-evolution is itself a defense" narrative — at least at
this scale and seed pool. Co-evolution here is a defender-side fragility,
not a defense.

### α-consistency check

```
  α from k=1 floor = +9.25 ± 0.43  pp/agent
  α from k=2 floor = +7.29 ± 0.49  pp/agent
  z = 2.99   (>2 → inconsistent — per-compromise floor is sub-linear in k)
```

---

## What this dataset does and does not support

**Open for revision (not yet ready as published claims):**

1. **v2 ΔJ-decomposition** (`α·k + β_C·Σρ + γ·(k C 2)·σ²`) — the Σρ slope is
   ≤ 0 here at z = −4.3 vs the held-out cell. Either the form is wrong, or
   the trained red used here is not representative of the worst case.

2. **v3 ΔJ-decomposition** (`α·k + β_C·max_i ρ + γ·(k C 2)·σ²`) — same
   problem at homogeneous ρ; z = −4.7. Concentration form does not rescue
   the prediction.

3. **"α·k is linear in k."** k=1 and k=2 anchors disagree at z = 2.99 — the
   per-compromise floor decays sub-linearly with k at this scale.

4. **"ρ=1 is the worst-case attacker."** Phase 2 shows `stay` ≥ `trained_red`
   by 1.2 pp at fixed (k=2, ρ=1). The worst-case attacker (in this dataset)
   is the *quietest* one.

5. **"Co-evolution is a defense for blue."** Phase 5 reverses the sign at
   z = 7.18: naive blue defends +7.06 pp better than co-evo blue. This may
   reflect overfitting of co-evo blue to its training partner rather than a
   universal property — needs a naive-blue ρ=0 floor to disentangle.

**Sensible conclusions to take forward (low confidence pending more studies):**

- The **action-stream channel is statistically silent** at this scale — the
  only attack surface a real adversary controls produces no measurable
  damage above the noise floor against a co-evolved blue. This is the load-
  bearing finding for a defender; it says a behavior-rate detector has no
  reliable signal to watch.
- The "α·k floor" is **not an attack channel**. Blue treats every observed
  agent as blue at runtime — `team_id=1` filtering only kicks in once the
  defender already knows who is compromised. The α·k numbers therefore
  measure the *cost paid after successful detection*, not the cost of
  attack. They are properly read as the floor of a TPR/cost curve.
- The (k, ρ) parameterization is **the wrong axis pair** for a threat model
  at this scale. ρ is a stealth/visibility budget for the attacker (and a
  detection signal for the defender), not an attack-strength multiplier.
  Mission damage in this regime is dominated by *how many agents the
  defender has already lost from its belief merge*, not by what those
  agents do at runtime.
- This is consistent with the existing direct-channel finding
  (`project_uncertainty_manipulation_thin.md`): red→blue uncertainty
  manipulation peaks at 2–3 cells with residual ≤1 pp. Two independent
  measurements now agree the action-stream channel is thin at 16×16 N=5.

---

## Working interpretation, given current data

This is **not** a working threat model — the data does not support
publishing one. It is what the dataset, read as a detector-calibration
instrument, actually says.

```
   post-detection cost (defender already knows agent is compromised):
       ΔJ_filter(k=1)  =  +9.25 ± 0.43
       ΔJ_filter(k=2)  = +14.59 ± 0.98     (sub-linear: α(1)≠α(2))

   action-stream channel (only surface a real attacker controls):
       β  ≈  0     statistically silent at this (map, N, co-evo blue)
       worst observed adversary at (k=2, ρ=1):  stay  (+14.79)
```

In words: at this scale (16×16, N=5, co-evolved blue), **the only
measurable mission damage is paid after the defender has already detected
the compromise**. The action-stream channel — the surface a real attacker
controls — produces no signal above the noise floor.

This is consistent with the parallel direct-channel result
(`project_uncertainty_manipulation_thin.md`, ≤1 pp residual). The
"misbehavior budget (k, ρ)" sweep is therefore best repurposed as a
**defender-side instrument**: ρ is the attacker's visibility-to-detector
knob, k is the post-detection cost ladder. Both are interesting for a
defender; neither describes an attacker's strategy in this regime.

---

## What this means for §8.1 of the meta-report

- The v3 *decomposition equation* should not be presented as a threat
  model; the data falsify both `Σρ` and `max ρ` forms at z > 4, and the
  channel it claimed to model (action-stream) is statistically silent.
- The "α·k" term must be **relabeled** in the meta-report from "attack
  channel" to "post-detection defender cost". It is the floor of a
  TPR/cost curve, not the worst-case mission damage from an attacker.
  Anywhere §8.1 says "α·k bounds attacker damage" is wrong — α·k bounds
  *defender cost given perfect detection*.
- The active-sabotage section should be reframed: the action-stream
  channel does not exhibit a budget-monotone response at this scale, so
  there is no closed-form `β_C` to report. Instead show the empirical
  shape (`ΔJ(k=2, ρ)` non-monotone with a defender-favorable basin at
  ρ ≈ 0.5) and acknowledge the channel is too thin to be modeled here.
- Connect to `project_uncertainty_manipulation_thin.md`: two independent
  measurements (action-stream sweep here, direct red→blue fogging there)
  converge on the same conclusion — the attack surfaces an adversary
  actually controls are thin against a co-evolved blue at 16×16 N=5.
- The right open question is now: **"under what conditions does the
  action-stream channel become non-thin?"** — larger map, larger N,
  damage-maximizing red, or naive (non-coevo) blue. The trained-red-helps-
  blue puzzle is a sub-question of that.

---

## Suggested follow-ups (each ≤ 30 min CPU)

1. **Phase 5 already in hand** (`phase5_naive_blue.npz`): if naive-blue ΔJ at
   (k=2, ρ=1) is significantly higher than co-evo blue's +12.21, the Nash
   basin *is* a defense. If not, blue's resilience is architectural.

2. **Re-train red as damage-maximizer** (not zero-sum reward) against the
   frozen co-evo blue. If the new red lifts ΔJ above the +14.79 stay floor,
   the current trained-red is a cherry-picked optimum that misses the worst
   case.

3. **Hetero-ρ revisit.** `experiments/adversarial-validation/misbehavior-budget/hetero_sweep.npz`
   has fixed-Σρ data at k=2. With the new clean-nominal patch, re-run that
   sweep at n=60 to see if the 6.5 pp asymmetry effect (project memory
   `hetero_sweep`) survives the OOD-nominal correction.

4. **Map-size sweep.** Re-run Phase 3 on a 32×32 grid. If the basin /
   sub-linear-α / negative-β all persist, the result is a property of the
   defender's redundancy, not of the 16×16 quirk.

---

## Files used

```
phase1_calibration.npz                 n=60, k=2, ρ∈{0,1}, trained_red, raw_obs
phase2_{trained_red,uniform_random,stay,nominal_raw,nominal_clamped}_rho1.npz
                                       n=30, k=2, ρ=1, all 5 attacker types
phase3_trained_clean_n60.npz           n=60, k=2, full ρ-curve, clean nominal
phase3_trained_clean_k1_n60.npz        n=60, k=1, full ρ-curve, clean nominal
phase4_k1_rho1_n60.npz                 n=60, k=1, ρ=1, k-scaling anchor
phase4_k2_rho1_n60.npz                 n=60, k=2, ρ=1, k-scaling anchor
phase5_naive_blue.npz                  n=30, k=2, ρ=1, naive-blue ckpt
fit_threat_model.txt                   raw fit output
```

## Figures

```
summary.png        five-phase summary panel    render.py
sensitivity.png    sensitivity-analysis viz    render_sensitivity.py
                   (tornado + ρ-response + held-out falsification)
```

Re-render commands:

```bash
PYTHONPATH=scripts:src python3 experiments/adversarial-validation/render.py
PYTHONPATH=scripts:src python3 experiments/adversarial-validation/render_sensitivity.py
```
