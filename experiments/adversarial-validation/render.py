"""Render the adversarial-validation summary panel from Phase 2/3/4 sweeps.

Run:  PYTHONPATH=scripts:src python3 experiments/adversarial-validation/render.py
Outputs: experiments/adversarial-validation/summary.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "summary.png"
B_MEAN = 98.5
DPI = 180


def _resolve(name: str) -> Path:
    n60 = HERE / name.replace(".npz", "_n60.npz")
    return n60 if n60.exists() else (HERE / name)


def load_finals(name: str) -> np.ndarray:
    return np.load(_resolve(name))["finals"].ravel()


def dj_stat(arr: np.ndarray) -> tuple[float, float]:
    sem = arr.std(ddof=1) / np.sqrt(len(arr))
    return B_MEAN - arr.mean(), 1.96 * sem


def load_curve(fname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(_resolve(fname))
    rhos, finals = d["rho"], d["finals"]
    xs, means, sems = [], [], []
    for i in range(len(rhos)):
        f = finals[i]
        xs.append(float(rhos[i]))
        means.append(B_MEAN - f.mean())
        sems.append(1.96 * f.std(ddof=1) / np.sqrt(len(f)))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(means)[order], np.array(sems)[order]


def main() -> None:
    adv_types = ["trained_red", "uniform_random", "stay", "nominal_raw", "nominal_clamped"]
    panel_a = [(t, dj_stat(load_finals(f"phase2_{t}_rho1.npz"))) for t in adv_types]
    panel_a.sort(key=lambda r: -r[1][0])

    curves = {
        "trained_red - clean": ("phase3_trained_clean.npz", "#d62728", "-"),
        "uniform_random - clean": ("phase3_random_clean.npz", "#9467bd", "-"),
        "trained_red - raw (current)": ("phase3_trained_raw.npz", "#888888", "--"),
    }

    ks_dj = []
    for k in (1, 2):
        f = load_finals(f"phase4_k{k}_rho1.npz")
        ks_dj.append((k, *dj_stat(f)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    names = [r[0] for r in panel_a]
    vals = [r[1][0] for r in panel_a]
    cis = [r[1][1] for r in panel_a]
    ax.barh(names, vals, xerr=cis, color="#d62728", edgecolor="white")
    ax.set_xlabel("DJ (pp)")
    ax.set_title("Adversary-type ablation at k=2, rho=1")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    for label, (fname, color, ls) in curves.items():
        xs, ys, sems = load_curve(fname)
        ax.plot(xs, ys, color=color, linewidth=2.2, linestyle=ls, label=label)
        ax.fill_between(xs, ys - sems, ys + sems, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("rho - fraction of agent's actions that are adversarial")
    ax.set_ylabel("DJ (pp vs B)")
    ax.set_title("rho-curve at k=2  (clean nominal vs raw-obs nominal)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    xs = np.array([k for k, _, _ in ks_dj])
    ys = np.array([dj for _, dj, _ in ks_dj])
    sems = np.array([s for _, _, s in ks_dj])
    ax.errorbar(xs, ys, yerr=sems, marker="o", linewidth=2.2,
                capsize=5, markersize=10, color="#d62728")
    if len(xs) >= 2:
        ratio = ys[1] / ys[0] if ys[0] else float("nan")
        ax.set_title(f"k-scaling at rho=1\nDJ(k=2)/DJ(k=1) = {ratio:.2f}  (linear = 2.00)")
    else:
        ax.set_title("k-scaling at rho=1")
    ax.set_xlabel("k - # compromised agents")
    ax.set_ylabel("DJ (pp)")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Adversarial-model validation - five-phase sweep summary",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
