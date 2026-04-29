"""Sensitivity figure for the misbehavior-budget sweep.

12 bars on one panel.  Each (k, ρ) cell is plotted at its **absolute
final coverage** (% of grid).  Three vertical reference lines mark the
three setups we report elsewhere — B, C1, C2 — so the budget-sweep cells
sit on the same scale.

Run:  PYTHONPATH=scripts:src python3 experiments/adversarial-validation/render_sensitivity.py
Output: experiments/adversarial-validation/sensitivity.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "sensitivity.png"
DPI = 180

# Reference experiments (mean final coverage, % of grid)
REF = {"B": 98.5, "C1": 89.6, "C2": 87.1}
REF_COLOR = {"B": "#2c7a3f", "C1": "#cc7a00", "C2": "#7a0000"}
REF_LABEL = {"B": "B   5 blue · 0 red",
             "C1": "C1  4 blue · 1 red",
             "C2": "C2  3 blue · 2 red"}

COLOR_K1 = "#3b6dab"
COLOR_K2 = "#c0392b"


def _resolve(name: str) -> Path:
    n60 = HERE / name.replace(".npz", "_n60.npz")
    return n60 if n60.exists() else (HERE / name)


def cell_stats(arr: np.ndarray) -> tuple[float, float]:
    n = len(arr)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    return mean, sem


def main() -> None:
    k1 = np.load(_resolve("phase3_trained_clean_k1.npz"))
    k2 = np.load(_resolve("phase3_trained_clean.npz"))
    rho_k1 = np.asarray(k1["rho"], dtype=float)
    rho_k2 = np.asarray(k2["rho"], dtype=float)
    f_k1 = k1["finals"]
    f_k2 = k2["finals"]
    o1 = np.argsort(rho_k1); rho_k1, f_k1 = rho_k1[o1], f_k1[o1]
    o2 = np.argsort(rho_k2); rho_k2, f_k2 = rho_k2[o2], f_k2[o2]

    z = 1.96
    rows: list[tuple[str, float, float, str]] = []
    for i, r in enumerate(rho_k1):
        m, se = cell_stats(f_k1[i])
        rows.append((f"k = 1,  ρ = {r:.2f}", m, z * se, COLOR_K1))
    for i, r in enumerate(rho_k2):
        m, se = cell_stats(f_k2[i])
        rows.append((f"k = 2,  ρ = {r:.2f}", m, z * se, COLOR_K2))

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.0))

    labels   = [r[0] for r in rows]
    means    = np.array([r[1] for r in rows])
    errors   = np.array([r[2] for r in rows])
    colors   = [r[3] for r in rows]

    y = np.arange(len(rows))
    ax.barh(y, means, xerr=errors, color=colors, edgecolor="white",
            capsize=4, error_kw={"ecolor": "black", "linewidth": 1.2})

    # Reference lines for B, C1, C2 (absolute final coverage)
    ref_handles = []
    for name, val in REF.items():
        ax.axvline(val, color=REF_COLOR[name], lw=1.0, ls="--", alpha=0.9,
                   zorder=1)
        ref_handles.append(plt.Line2D(
            [0], [0], color=REF_COLOR[name], lw=1.4, ls="--",
            label=f"{REF_LABEL[name]}    {val:.1f} %"))

    # Light divider between the k=1 and k=2 groups
    ax.axhline(len(rho_k1) - 0.5, color="#999999", lw=0.7, ls="--", alpha=0.6)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)

    # Numeric labels at bar tips
    for yi, (m, err) in enumerate(zip(means, errors)):
        ax.text(m + err + 0.25, yi, f"{m:.1f} ± {err:.1f}",
                va="center", ha="left", fontsize=10, fontweight="bold")

    lo = float(min(means.min() - errors.max(), min(REF.values()))) - 1.0
    hi = float(max(means.max() + errors.max(), max(REF.values()))) + 4.0
    ax.set_xlim(lo, hi)
    ax.set_xlabel("final coverage  (% of grid,  mean ± 95% CI)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    ax.legend(handles=ref_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.07), ncol=3,
              frameon=False, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")

    print(f"wrote {OUT_PNG}")
    for lab, m, err, _ in rows:
        print(f"  {lab:<22}  coverage = {m:6.2f} ± {err:.2f} %")


if __name__ == "__main__":
    main()
