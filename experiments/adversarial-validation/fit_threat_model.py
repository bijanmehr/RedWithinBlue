"""Phase 7 — Threat-model fit and falsification.

Fits two competing closed-form threat models to the homogeneous-rho clean
sweeps (Phase 3 k=1 and k=2, adversary=trained_red, nominal=clamp_team_id_zero):

    M1 (v2 additive)        :  DJ(k, rho) = alpha * k + beta_M1 * k * rho
    M2 (v3 concentration)   :  DJ(k, rho) = alpha * k + beta_M2 * c(rho)
                                            with c(rho) = max_i rho_i = rho
                                            at homogeneous rho.

At k=1 the two models are identical. They differ at k>=2: M1 says the active
sabotage scales with the *sum* of per-agent rates (= k*rho here), M2 with the
*max*. The held-out test cell is therefore (k=2, rho=1).

Procedure
---------
1. alpha_1, alpha_2  from clean rho=0 cells (k=1, k=2). Consistency check.
2. beta_M1, beta_M2  from k=1 slope and k=2 slope respectively.
3. Predict every (k, rho) cell from each model. Report residuals with 95% CI.
4. Held-out: predict (k=2, rho=1) using only k=1 anchors and slopes.
5. Worst-case envelope: replace beta with beta from Phase 2 worst adversary.

Inputs (all in this folder):
    phase3_trained_clean.npz       k=2, clean nominal, trained_red
    phase3_trained_clean_k1.npz    k=1, clean nominal, trained_red
    phase1_calibration.npz         high-precision n=60 anchor at (k=2, rho={0,1})
    phase2_*_rho1.npz              for worst-case envelope
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B_MEAN = 98.5  # baseline (5b/0r) coverage in pp


def load_finals(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefer the _n60 variant when present (Phase 7 re-run)."""
    n60 = HERE / name.replace(".npz", "_n60.npz")
    path = n60 if n60.exists() else (HERE / name)
    d = np.load(path)
    print(f"  loaded  {path.name}  (n={int(d['n_seeds'])})")
    return d["k"], d["rho"], d["finals"]


def cell_dj(finals_row: np.ndarray) -> tuple[float, float]:
    """Return (DJ_mean, SEM) for one (k, rho) cell."""
    n = len(finals_row)
    mean = float(finals_row.mean())
    sem = float(finals_row.std(ddof=1) / np.sqrt(n))
    return B_MEAN - mean, sem


def fit_alpha_beta(k_arr, rho_arr, finals):
    """Return (alpha, sem_alpha, beta, sem_beta) on this single-k sweep.

    Linear OLS:  DJ(rho) = a + b * rho   (no weights; SEMs reported)
    """
    djs = np.array([cell_dj(finals[i])[0] for i in range(len(rho_arr))])
    sems = np.array([cell_dj(finals[i])[1] for i in range(len(rho_arr))])
    rho = np.asarray(rho_arr, dtype=float)
    X = np.stack([np.ones_like(rho), rho], axis=1)
    coef, *_ = np.linalg.lstsq(X, djs, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    # Approximate parameter SEMs via OLS covariance with mean-residual variance.
    resid = djs - X @ coef
    dof = max(1, len(rho) - 2)
    sigma2 = float((resid ** 2).sum() / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return a, float(np.sqrt(cov[0, 0])), b, float(np.sqrt(cov[1, 1])), djs, sems


def main() -> None:
    print("=" * 72)
    print("Phase 7 — Threat-model fit (homogeneous rho, clean nominal, trained_red)")
    print("=" * 72)

    # --- Load clean curves ---
    k1_k, k1_r, k1_f = load_finals("phase3_trained_clean_k1.npz")
    k2_k, k2_r, k2_f = load_finals("phase3_trained_clean.npz")

    a1, sa1, b1, sb1, dj1, se1 = fit_alpha_beta(k1_k, k1_r, k1_f)
    a2, sa2, b2, sb2, dj2, se2 = fit_alpha_beta(k2_k, k2_r, k2_f)

    print("\nObserved DJ on the two clean sweeps (mean +/- SEM, pp):")
    print(f"  k=1  rho->  {' '.join(f'{r:>5.2f}' for r in k1_r)}")
    print(f"        DJ   {' '.join(f'{d:>+5.2f}' for d in dj1)}")
    print(f"        SEM  {' '.join(f'{s:>5.2f}' for s in se1)}")
    print(f"  k=2  rho->  {' '.join(f'{r:>5.2f}' for r in k2_r)}")
    print(f"        DJ   {' '.join(f'{d:>+5.2f}' for d in dj2)}")
    print(f"        SEM  {' '.join(f'{s:>5.2f}' for s in se2)}")

    # --- Fit alpha from rho=0 anchors ---
    dj_k1_r0, sem_k1_r0 = cell_dj(k1_f[0])      # rho=0, k=1
    dj_k2_r0, sem_k2_r0 = cell_dj(k2_f[0])      # rho=0, k=2
    alpha1 = dj_k1_r0 / 1.0
    alpha2 = dj_k2_r0 / 2.0
    print(f"\nalpha estimates (per-compromise floor, pp):")
    print(f"  alpha (from k=1 floor) = {alpha1:+.2f}  +/- {sem_k1_r0:.2f}")
    print(f"  alpha (from k=2 floor) = {alpha2:+.2f}  +/- {sem_k2_r0/2:.2f}")
    print(f"  consistency: |a1 - a2| / pooled_SEM = "
          f"{abs(alpha1-alpha2)/np.sqrt(sem_k1_r0**2 + (sem_k2_r0/2)**2):.2f} "
          f"(>2 = inconsistent)")
    alpha = 0.5 * (alpha1 + alpha2)
    print(f"  pooled alpha = {alpha:+.2f} pp")

    # --- Fit beta_M1 (additive) and beta_M2 (concentration) ---
    # M1 says DJ(k=1, rho) = alpha + beta_M1 * 1 * rho  -> slope at k=1 = beta_M1
    # M2 says DJ(k=1, rho) = alpha + beta_M2 * rho      -> slope at k=1 = beta_M2
    # At k=1 the two models coincide, so the k=1 slope alone can't separate them.
    beta_M1 = b1
    beta_M2 = b1
    print(f"\nbeta from k=1 slope = {b1:+.2f}  +/- {sb1:.2f}")
    print(f"  M1 (additive)        : beta_M1 = {beta_M1:+.2f}  (DJ ~ a*k + b*k*rho)")
    print(f"  M2 (concentration)   : beta_M2 = {beta_M2:+.2f}  (DJ ~ a*k + b*c(rho))")

    # --- Predict the full grid from each model ---
    def pred_M1(k, rho):  return alpha * k + beta_M1 * k * rho
    def pred_M2(k, rho):  return alpha * k + beta_M2 * rho   # c(rho)=rho at homog

    print("\nGrid residuals  (DJ_obs - DJ_pred,  pp;  > 2*SEM in brackets):")
    print(f"  {'k':>3} {'rho':>5} {'DJ_obs':>8} {'M1_pred':>8} {'M1_res':>8} "
          f"{'M2_pred':>8} {'M2_res':>8} {'2*SEM':>6}")
    for k_arr, rho_arr, finals in ((k1_k, k1_r, k1_f), (k2_k, k2_r, k2_f)):
        for i in range(len(rho_arr)):
            k = int(k_arr[i]); rho = float(rho_arr[i])
            dj, sem = cell_dj(finals[i])
            m1 = pred_M1(k, rho); m2 = pred_M2(k, rho)
            r1 = dj - m1; r2 = dj - m2
            tag1 = "*" if abs(r1) > 2*sem else " "
            tag2 = "*" if abs(r2) > 2*sem else " "
            print(f"  {k:>3} {rho:>5.2f} {dj:>+8.2f} {m1:>+8.2f} {r1:>+7.2f}{tag1} "
                  f"{m2:>+8.2f} {r2:>+7.2f}{tag2} {2*sem:>6.2f}")
    print("  (* marks cells where |residual| > 2 SEM)")

    # --- Held-out: predict (k=2, rho=1) from k=1-only fit ---
    held_obs, held_sem = cell_dj(k2_f[-1])
    held_M1 = pred_M1(2, 1.0)
    held_M2 = pred_M2(2, 1.0)
    print(f"\nHeld-out cell (k=2, rho=1):")
    print(f"  observed       = {held_obs:+.2f}  +/- {held_sem:.2f}")
    print(f"  M1 prediction  = {held_M1:+.2f}   residual = {held_obs - held_M1:+.2f}  "
          f"(z = {(held_obs - held_M1)/held_sem:+.2f})")
    print(f"  M2 prediction  = {held_M2:+.2f}   residual = {held_obs - held_M2:+.2f}  "
          f"(z = {(held_obs - held_M2)/held_sem:+.2f})")

    # Confirm against the n=60 phase1 anchor
    p1 = np.load(HERE / "phase1_calibration.npz")
    p1_finals = p1["finals"]; p1_rho = p1["rho"]
    idx_r1 = int(np.argmin(np.abs(p1_rho - 1.0)))
    p1_dj, p1_sem = cell_dj(p1_finals[idx_r1])
    print(f"  Phase 1 anchor (n=60) = {p1_dj:+.2f}  +/- {p1_sem:.2f}  "
          f"(uses this for falsification when |zM1 - zM2| ambiguous on n=30)")
    print(f"    M1 z (vs n=60): {(p1_dj - held_M1)/p1_sem:+.2f}")
    print(f"    M2 z (vs n=60): {(p1_dj - held_M2)/p1_sem:+.2f}")

    # --- Worst-case envelope from Phase 2 ---
    print("\nWorst-case envelope (Phase 2, k=2, rho=1):")
    rows = []
    for path in sorted(HERE.glob("phase2_*_rho1.npz")):
        name = path.stem.replace("phase2_", "").replace("_rho1", "")
        f = np.load(path)["finals"].ravel()
        dj = B_MEAN - f.mean()
        sem = f.std(ddof=1) / np.sqrt(len(f))
        rows.append((name, dj, sem))
    rows.sort(key=lambda r: -r[1])
    for n, dj, s in rows:
        print(f"  {n:>20}  DJ = {dj:>+6.2f}  +/- {s:.2f}")
    worst = rows[0]
    beta_worst = (worst[1] - 2 * alpha) / 1.0     # M2 form, c(rho=1)=1
    print(f"\n  worst-case beta (M2-form using {worst[0]}): {beta_worst:+.2f}")
    print(f"  envelope: DJ_max(k, rho) = {alpha:+.2f}*k + {beta_worst:+.2f}*max_i(rho_i)")

    # --- Final equation ---
    print("\n" + "=" * 72)
    print("Fitted threat-model equations:")
    print(f"  M1 (additive)      : DJ(k, rho)  =  {alpha:+.2f} * k  +  {beta_M1:+.2f} * k * rho")
    print(f"  M2 (concentration) : DJ(k, rho)  =  {alpha:+.2f} * k  +  {beta_M2:+.2f} * max_i rho_i")
    print(f"  Worst-case (M2)    : DJ(k, rho) <=  {alpha:+.2f} * k  +  {beta_worst:+.2f} * max_i rho_i")
    print("=" * 72)


if __name__ == "__main__":
    main()
