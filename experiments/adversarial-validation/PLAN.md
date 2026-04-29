# Adversarial Model Validation — Run Commands

All outputs land in `experiments/adversarial-validation/`. Run from repo root.

The patched sweep script lives at `scripts/misbehavior_budget_sweep.py` and
supports five new flags: `--adversary-type`, `--nominal-mode`, `--out-npz`,
`--k-filter`, `--blue-ckpt`. `--out-npz` also controls where the run-log lands
(next to the npz, named `<stem>.log`).

Total cost: ~1550 episodes, ~45 min on CPU.

---

## Pre-flight

```bash
ls -la experiments/misbehavior-budget/budget_sweep.npz
ls -la experiments/compromise-16x16-5-4b1r-coevo/checkpoint.npz
ls -la experiments/compromise-16x16-5-4b1r-coevo/joint_red_checkpoint.npz
ls -la experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz
ls -la experiments/compromise-16x16-5-3b2r-coevo/joint_red_checkpoint.npz
ls -la experiments/survey-local-16-N5-from-N4/checkpoint.npz
ls -la configs/compromise-16x16-5-4b1r.yaml
ls -la configs/compromise-16x16-5-3b2r.yaml
```

## Smoke test

```bash
python scripts/misbehavior_budget_sweep.py \
  --n-seeds 1 --rhos 1.0 --k-filter 2 \
  --out-npz experiments/adversarial-validation/_smoke.npz
```

## Step 0 — Regression check (defaults reproduce existing sweep)

```bash
python scripts/misbehavior_budget_sweep.py \
  --n-seeds 10 --rhos 0.0 0.25 0.5 0.75 1.0 \
  --adversary-type trained_red --nominal-mode raw_obs \
  --out-npz experiments/adversarial-validation/_regression.npz

python -c "
import numpy as np, numpy.testing as npt
old = np.load('experiments/misbehavior-budget/budget_sweep.npz')
new = np.load('experiments/adversarial-validation/_regression.npz')
npt.assert_allclose(old['mean'], new['mean'], atol=0.5)
print('regression OK — old and new sweep means agree within 0.5 pp')
"
```

## Phase 1 — Statistical calibration (k=2, n=60, ρ∈{0,1})

```bash
python scripts/misbehavior_budget_sweep.py \
  --n-seeds 60 --rhos 0.0 1.0 --k-filter 2 \
  --adversary-type trained_red --nominal-mode raw_obs \
  --out-npz experiments/adversarial-validation/phase1_calibration.npz

python -c "
import numpy as np
d = np.load('experiments/adversarial-validation/phase1_calibration.npz')
ks = d['k']; rhos = d['rho']; finals = d['finals']
B_mean = 98.5
print(f'{\"k\":>3} {\"rho\":>5} {\"n\":>3} {\"cov\":>6} {\"sigma\":>6} {\"SEM\":>5} {\"DJ\":>6} {\"95% CI\":>16}')
for ci in range(len(ks)):
    f = finals[ci]; mean = f.mean(); std = f.std(ddof=1); sem = std / np.sqrt(len(f))
    dj = B_mean - mean; lo, hi = dj - 1.96*sem, dj + 1.96*sem
    print(f'{int(ks[ci]):>3} {float(rhos[ci]):>5.2f} {len(f):>3} {mean:>6.2f} {std:>6.2f} {sem:>5.2f} {dj:>+6.2f} [{lo:+5.2f}, {hi:+5.2f}]')
"
```

## Phase 2 — Adversary-type ablation at k=2, ρ=1 (5 types × 30 seeds)

```bash
for ADV in trained_red uniform_random stay nominal_raw nominal_clamped; do
  python scripts/misbehavior_budget_sweep.py \
    --n-seeds 30 --rhos 1.0 --k-filter 2 \
    --adversary-type $ADV --nominal-mode clamp_team_id_zero \
    --out-npz experiments/adversarial-validation/phase2_${ADV}_rho1.npz
done

python -c "
import numpy as np, glob
B_mean = 98.5
rows = []
for path in sorted(glob.glob('experiments/adversarial-validation/phase2_*_rho1.npz')):
    name = path.split('phase2_')[1].split('_rho1')[0]
    f = np.load(path)['finals'].ravel()
    mean = f.mean(); std = f.std(ddof=1); sem = std/np.sqrt(len(f))
    rows.append((name, B_mean - mean, sem, len(f)))
rows.sort(key=lambda r: -r[1])
print(f'{\"adversary\":>20}  {\"DJ (pp)\":>9}  {\"95% CI\":>16}  n')
for n, dj, sem, ns in rows:
    print(f'{n:>20}  {dj:>+9.2f}  [{dj-1.96*sem:+5.2f}, {dj+1.96*sem:+5.2f}]  {ns}')
"
```

## Phase 3 — ρ-curves at k=2 and k=1 (3 sweeps × 6 ρ × 30 seeds × 2 k)

```bash
# k=2 — clean trained, clean random, raw trained
python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 0.0 0.1 0.25 0.5 0.75 1.0 --k-filter 2 \
  --adversary-type trained_red --nominal-mode clamp_team_id_zero \
  --out-npz experiments/adversarial-validation/phase3_trained_clean.npz

python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 0.0 0.1 0.25 0.5 0.75 1.0 --k-filter 2 \
  --adversary-type uniform_random --nominal-mode clamp_team_id_zero \
  --out-npz experiments/adversarial-validation/phase3_random_clean.npz

python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 0.0 0.1 0.25 0.5 0.75 1.0 --k-filter 2 \
  --adversary-type trained_red --nominal-mode raw_obs \
  --out-npz experiments/adversarial-validation/phase3_trained_raw.npz

# k=1 — same three configs (run as three explicit commands; the bash for-loop
# with `set --` does not work under zsh)
python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 0.0 0.1 0.25 0.5 0.75 1.0 --k-filter 1 \
  --adversary-type trained_red --nominal-mode clamp_team_id_zero \
  --out-npz experiments/adversarial-validation/phase3_trained_clean_k1.npz

python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 0.0 0.1 0.25 0.5 0.75 1.0 --k-filter 1 \
  --adversary-type uniform_random --nominal-mode clamp_team_id_zero \
  --out-npz experiments/adversarial-validation/phase3_random_clean_k1.npz

python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 0.0 0.1 0.25 0.5 0.75 1.0 --k-filter 1 \
  --adversary-type trained_red --nominal-mode raw_obs \
  --out-npz experiments/adversarial-validation/phase3_trained_raw_k1.npz

# Print all phase-3 curves
python -c "
import numpy as np, glob
B_mean = 98.5
for path in sorted(glob.glob('experiments/adversarial-validation/phase3_*.npz')):
    print(f'\n--- {path.split(\"/\")[-1]} ---')
    d = np.load(path)
    rhos = d['rho']; ks = d['k']; finals = d['finals']
    for ci in range(len(rhos)):
        f = finals[ci]; sem = f.std(ddof=1)/np.sqrt(len(f))
        dj = B_mean - f.mean()
        print(f'  k={int(ks[ci])} rho={float(rhos[ci]):.2f}  DJ = {dj:+5.2f} +/- {1.96*sem:.2f} pp')
"
```

## Phase 4 — k-scaling at ρ=1, clean nominal

```bash
python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 1.0 --k-filter 1 \
  --adversary-type trained_red --nominal-mode clamp_team_id_zero \
  --out-npz experiments/adversarial-validation/phase4_k1_rho1.npz

python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 1.0 --k-filter 2 \
  --adversary-type trained_red --nominal-mode clamp_team_id_zero \
  --out-npz experiments/adversarial-validation/phase4_k2_rho1.npz

python -c "
import numpy as np, glob
B_mean = 98.5
rows = [(int(p.split('_k')[1].split('_')[0]), p) for p in sorted(glob.glob('experiments/adversarial-validation/phase4_k*.npz'))]
print(f'{\"k\":>3}  {\"DJ (pp)\":>10}  {\"95% CI\":>16}')
djs = []
for k, p in rows:
    f = np.load(p)['finals'].ravel()
    sem = f.std(ddof=1)/np.sqrt(len(f))
    dj = B_mean - f.mean()
    djs.append((k, dj))
    print(f'{k:>3}  {dj:>+10.2f}  [{dj-1.96*sem:+5.2f}, {dj+1.96*sem:+5.2f}]')
if len(djs) >= 2:
    k1, dj1 = djs[0]; k2, dj2 = djs[1]
    ratio = dj2 / dj1 if dj1 else float('nan')
    print(f'\\nDJ(k={k2}) / DJ(k={k1}) = {ratio:.2f}   (linear would be {k2/k1:.2f})')
"
```

## Phase 5 — Naive-blue test (Nash basin probe)

```bash
python scripts/misbehavior_budget_sweep.py \
  --n-seeds 30 --rhos 1.0 --k-filter 2 \
  --adversary-type trained_red --nominal-mode clamp_team_id_zero \
  --blue-ckpt experiments/survey-local-16-N5-from-N4/checkpoint.npz \
  --out-npz experiments/adversarial-validation/phase5_naive_blue.npz

python -c "
import numpy as np
B_mean = 98.5
def stat(p):
    f = np.load(p)['finals'].ravel()
    sem = f.std(ddof=1)/np.sqrt(len(f))
    return B_mean - f.mean(), 1.96*sem, len(f)
dj_co, ci_co, n_co = stat('experiments/adversarial-validation/phase4_k2_rho1.npz')
dj_na, ci_na, n_na = stat('experiments/adversarial-validation/phase5_naive_blue.npz')
print(f'co-evolved blue : DJ = {dj_co:+.2f} +/- {ci_co:.2f}  (n={n_co})')
print(f'naive blue      : DJ = {dj_na:+.2f} +/- {ci_na:.2f}  (n={n_na})')
print(f'co-evo defense effect = {dj_na - dj_co:+.2f} pp')
"
```

## Phase 7 — Threat-model fit (no rollouts, ~5 s)

Fits competing models to Phases 1/3 outputs and decides which survives:

```
M1 (additive)        :  DJ = alpha * k  +  beta * k * rho
M2 (concentration)   :  DJ = alpha * k  +  beta * max_i rho_i
worst-case envelope  :  DJ <= alpha * k  +  beta_worst * max_i rho_i   (Phase 2)
```

Run:

```bash
python experiments/adversarial-validation/fit_threat_model.py \
  | tee experiments/adversarial-validation/fit_threat_model.txt
```

Outputs the fitted coefficients (with 95 % CIs), per-cell residuals, the
held-out cell verdict, and the worst-case envelope. Reads only `phase1_*`,
`phase3_trained_clean*`, and `phase2_*_rho1` — no rollouts.

## Phase 6 — Render summary panel

Outputs `experiments/adversarial-validation/summary.png`.

```bash
PYTHONPATH=scripts:src python3 experiments/adversarial-validation/render.py
```

## After all phases

Write `experiments/adversarial-validation/results.md` (one page: numbers +
conclusions matched to the decision tree below).

---

## Decision tree (what each result means for the meta-report)

```
Phase 1 -> calibration confirms n=30 is enough (or bump to n=60)
  Phase 2:
    trained_red dominates       -> rho=1 is worst-case; trust current heatmap
    uniform_random dominates    -> co-evo Nash protects blue; re-frame report
    nominal_raw matches trained -> OOD-nominal is real confound; use clamp downstream
  Phase 3:
    clean curve monotone in rho   -> original basin was OOD artifact
    clean curve still non-monotone -> real attacker-suboptimal sweet spot
    random > trained at all rho   -> co-evo Nash is the defense
  Phase 4:
    ratio ~ 2.0 -> linear scaling
    ratio < 2.0 -> sub-linear; cooperator surplus absorbs damage
    ratio > 2.0 -> super-linear; centralized red is qualitatively dangerous
  Phase 5:
    DJ(naive) ~ DJ(coevo)        -> co-evo doesn't matter
    DJ(naive) > DJ(coevo) by 5+  -> co-evo is itself a learnable defense
```

---

## Output paths cheatsheet

```
experiments/adversarial-validation/
  PLAN.md                         this file
  _smoke.npz                      smoke test
  _regression.npz                 step-0 acceptance
  phase1_calibration.npz          n=60 baseline noise
  phase2_<adv>_rho1.npz           5 files
  phase3_trained_clean.npz        k=2 main result
  phase3_random_clean.npz         k=2 berserk reference
  phase3_trained_raw.npz          k=2 raw-obs regression
  phase3_*_k1.npz                 k=1 versions of phase 3
  phase4_k1_rho1.npz              k-scaling
  phase4_k2_rho1.npz
  phase5_naive_blue.npz           Nash basin probe
  render.py                       phase-6 figure script
  summary.png                     phase-6 output
  results.md                      one-page summary written after phase 6
  <stem>.log                      run-log per --out-npz invocation
```
