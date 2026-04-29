"""Generate every v2 figure that can be built from existing `.npz` caches.

Replaces the placeholder boxes in `experiments/meta-report/meta_report_v2.html`.

Inputs
------
- experiments/compromise-compare/compromise_compare.npz
    labels (4,), finals_i (20,), curve_i (20, 200) for i in {0:S, 1:B, 2:C1, 3:C2}
    (finals_i for i=0 is 300-step N=1 solo; everything else is 200-step N=5)
- experiments/misbehavior-budget/budget_sweep.npz
    k (10,), rho (10,), finals (10, 10)=(config, seed), mean/std/n_seeds/max_steps

Outputs (in experiments/meta-report/)
-------
Single-source, no sweep needed:
  claim1_invariant.png        — throughput + T(θ) staircase (S vs B)
  time_to_coverage_multiseed.png — median + p10/p90 band per setup + T(90%)
  forest_delta_j.png          — horizontal forest with θ line
  variance_bar.png            — σ + min(cov) bar pair + per-seed dots
  kstar_staircase.png         — k*(θ) step plot
  resilience_triangle.png     — Magnitude / Brittleness / Timeliness radar
  channels_stacked.png        — per-setup {blue-only, BOTH, red-only, nobody} bar
  budget_match_diff.png       — matched-M bar chart + zero line
  budget_raincloud_grid.png   — 2 × 5 raincloud (k × ρ)
  budget_pareto.png           — ΔJ vs M with envelopes
  model_decomposition.png     — α·k + β·Σρ·C + γ·C(k,2)·σ² vs measured
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

OUT_DIR = Path("experiments/meta-report")
CC_NPZ = Path("experiments/compromise-compare/compromise_compare.npz")
BS_NPZ = Path("experiments/misbehavior-budget/budget_sweep.npz")

# Load caches
cc = np.load(CC_NPZ, allow_pickle=True)
SETUPS = ["S", "B", "C1", "C2"]
FINALS = {k: cc[f"finals_{i}"] for i, k in enumerate(SETUPS)}
CURVES = {k: cc[f"curve_{i}"] for i, k in enumerate(SETUPS)}  # (20, T)
PALETTE = {"S": "#8c564b", "B": "#1f77b4", "C1": "#ff7f0e", "C2": "#d62728"}
SHORT = {"S": "S — N=1 solo", "B": "B — 5 blue", "C1": "C1 — 4b+1r", "C2": "C2 — 3b+2r"}
N_AGENTS = {"S": 1, "B": 5, "C1": 5, "C2": 5}
N_BLUE = {"S": 1, "B": 5, "C1": 4, "C2": 3}

bs = np.load(BS_NPZ, allow_pickle=True)
BK = bs["k"]       # (10,)
BR = bs["rho"]     # (10,)
BF = bs["finals"]  # (10, 10) = (config, seed)
BS_SEEDS = int(bs["n_seeds"])

B_CLEAN_MEAN = float(FINALS["B"].mean())  # reference ΔJ baseline


def _bootstrap_ci(x: np.ndarray, n: int = 10_000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def _t_at_threshold(curve: np.ndarray, pct: float) -> int | None:
    """Earliest step index (1-based) at which coverage reaches `pct`. None if never."""
    above = np.where(curve >= pct)[0]
    return int(above[0] + 1) if len(above) else None


# ---------------------------------------------------------------- Claim 1
def claim1_invariant():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) throughput per-agent J/(N·T_budget)
    tb = {"S": 300, "B": 200, "C1": 200, "C2": 200}
    throughput = {k: FINALS[k] / (N_BLUE[k] * tb[k]) for k in SETUPS}
    xs = np.arange(len(SETUPS))
    vals = [throughput[k].mean() for k in SETUPS]
    ers = [throughput[k].std(ddof=1) for k in SETUPS]
    cols = [PALETTE[k] for k in SETUPS]
    axes[0].bar(xs, vals, yerr=ers, color=cols, edgecolor="black", linewidth=0.5, capsize=4)
    for i, (v, e) in enumerate(zip(vals, ers)):
        axes[0].text(i, v + e + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([SHORT[k] for k in SETUPS], fontsize=9)
    axes[0].set_ylabel("throughput  $\\bar{J} = J / (N_\\mathrm{blue} \\cdot T_\\mathrm{budget})$")
    axes[0].set_title("(a) Per-blue-agent throughput\n(size-invariant utilisation of the mission clock)")
    axes[0].grid(True, axis="y", alpha=0.3)

    # (b) T(θ) staircase across θ ∈ [0.3, 1.0] — % seeds reaching θ within budget
    thetas = np.linspace(0.3, 1.0, 36) * 100  # percent
    for k in SETUPS:
        frac_reach = []
        for th in thetas:
            reached = [_t_at_threshold(CURVES[k][s], th) for s in range(CURVES[k].shape[0])]
            frac_reach.append(np.mean([r is not None for r in reached]))
        axes[1].plot(thetas, frac_reach, label=SHORT[k],
                     color=PALETTE[k], linewidth=2,
                     linestyle="--" if k == "S" else "-")
    axes[1].axvspan(45, 100, color="#f0f0f0", alpha=0.4, zorder=0)
    axes[1].text(50, 0.06, "solo (N=1) cannot reach θ > 60 %\n— size-advantage region",
                 fontsize=8, color="#666", va="bottom")
    axes[1].set_xlabel("coverage threshold $\\theta$ (%)")
    axes[1].set_ylabel("fraction of seeds reaching $\\theta$")
    axes[1].set_title("(b) Reach-probability vs threshold\n(solo-vs-swarm survivorship)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_ylim(-0.02, 1.05)

    fig.suptitle("Claim 1 · swarm strictly dominates solo on any size-invariant metric",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "claim1_invariant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §6.1
def time_to_coverage_multiseed():
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.2))
    for k in ["B", "C1", "C2"]:  # skip solo (different T)
        c = CURVES[k]  # (seeds, T)
        xs = np.arange(1, c.shape[1] + 1)
        p10, p50, p90 = np.percentile(c, [10, 50, 90], axis=0)
        ax.fill_between(xs, p10, p90, color=PALETTE[k], alpha=0.20, linewidth=0)
        ax.plot(xs, p50, label=SHORT[k], color=PALETTE[k], linewidth=2.2)
        # T(90%)
        t90_seeds = [_t_at_threshold(c[s], 90) for s in range(c.shape[0])]
        valid = [t for t in t90_seeds if t is not None]
        if len(valid) >= c.shape[0] // 2:
            t90_med = int(np.median(valid))
            ax.axvline(t90_med, color=PALETTE[k], linestyle=":", linewidth=1.4, alpha=0.7)
            ax.text(t90_med + 2, 40 + 6 * {"B": 0, "C1": 1, "C2": 2}[k],
                    f"T(90%) = {t90_med}\n({len(valid)}/{c.shape[0]} seeds)",
                    color=PALETTE[k], fontsize=9, va="center")
        else:
            ax.text(c.shape[1] - 8, 92,
                    f"{SHORT[k]}: NEVER ({len(valid)}/{c.shape[0]} seeds)",
                    color=PALETTE[k], fontsize=9, ha="right", va="top")
    ax.axhline(90, color="#555", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.99, 91, "90 % threshold", ha="right", fontsize=9, color="#555")
    ax.set_xlabel("step $t$")
    ax.set_ylabel("blue ever-known coverage (%)")
    ax.set_title("Time-to-coverage — 20 seeds per setup  (median line, p10–p90 shaded)",
                 fontsize=11)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_to_coverage_multiseed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §7.1
def forest_delta_j(theta_pp: float = 10.0):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.0))
    rows = [("S", 0), ("C1", 1), ("C2", 2)]
    for k, y in rows:
        delta = FINALS[k] - FINALS["B"]  # (seeds,)
        mean = float(delta.mean())
        lo, hi = _bootstrap_ci(delta)
        pooled_std = np.sqrt((FINALS[k].var(ddof=1) + FINALS["B"].var(ddof=1)) / 2)
        cohen_d = (FINALS[k].mean() - FINALS["B"].mean()) / pooled_std
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                    fmt="D", color=PALETTE[k], ecolor=PALETTE[k],
                    markersize=8 + 1.5 * min(abs(cohen_d), 6),
                    capsize=4, linewidth=2)
        ax.text(mean, y + 0.22,
                f"{mean:+.1f} pp  [{lo:+.1f}, {hi:+.1f}]  d={cohen_d:+.2f}",
                ha="center", va="bottom", fontsize=9, color=PALETTE[k])
    ax.axvline(0, color="#555", linestyle="--", linewidth=1)
    ax.axvline(-theta_pp, color="#b44", linestyle=":", linewidth=1.4)
    ax.text(-theta_pp - 1, 2.6, f"detector θ = {theta_pp:.0f} pp",
            color="#b44", fontsize=9, ha="right")
    ax.set_yticks([y for _, y in rows])
    ax.set_yticklabels([SHORT[k] for k, _ in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta J$ vs clean B  (pp)   — more negative = worse for blue")
    ax.set_title("Forest plot — $\\Delta J$ with bootstrap 95 % CIs\n"
                 "(diamond marker size ∝ |Cohen d|)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(-58, 5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forest_delta_j.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §7.2
def variance_bar():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(SETUPS))
    sigmas = [FINALS[k].std(ddof=1) for k in SETUPS]
    mins = [FINALS[k].min() for k in SETUPS]
    cols = [PALETTE[k] for k in SETUPS]

    axes[0].bar(xs, sigmas, color=cols, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(sigmas):
        axes[0].text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(xs); axes[0].set_xticklabels([SHORT[k] for k in SETUPS], fontsize=9)
    axes[0].set_ylabel("σ(coverage)  (pp)")
    axes[0].set_title("(a) Per-setup σ — brittleness\n(B→C1→C2: 2.3 → 3.7 → 7.6)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(xs, mins, color=cols, edgecolor="black", linewidth=0.5)
    for k, i in zip(SETUPS, xs):
        axes[1].scatter(np.full(len(FINALS[k]), i) + np.linspace(-0.2, 0.2, len(FINALS[k])),
                        FINALS[k], s=14, color="black", alpha=0.5, zorder=3)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([SHORT[k] for k in SETUPS], fontsize=9)
    axes[1].set_ylabel("coverage (%)")
    axes[1].set_title("(b) Per-seed distribution with min-bar\n(black dots = individual seeds)")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_ylim(0, 102)

    fig.suptitle("Variance inflation at $k = 2$ —  the mean barely moved, the spread exploded",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variance_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §7.3
def kstar_staircase():
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.0))
    thetas = np.linspace(0.1, 20, 500)
    dJ1 = abs(FINALS["C1"].mean() - FINALS["B"].mean())
    dJ2 = abs(FINALS["C2"].mean() - FINALS["B"].mean())
    ks = []
    for th in thetas:
        if th <= dJ1:
            ks.append(1)
        elif th <= dJ2:
            ks.append(2)
        else:
            ks.append(np.nan)  # ∞ / unreached
    ks = np.array(ks, dtype=float)
    # Plot with step semantics
    ax.plot(thetas, ks, color="#333", linewidth=2.5, drawstyle="steps-post")
    ax.axhline(3, color="#a00", linestyle=":", linewidth=1.4)
    ax.text(0.2, 3.12, "k = ∞  (no tested attacker suffices)", color="#a00", fontsize=9)
    ax.axvline(dJ1, color=PALETTE["C1"], linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvline(dJ2, color=PALETTE["C2"], linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(dJ1, 1.15, f"θ = ΔJ(1) = {dJ1:.1f} pp", color=PALETTE["C1"],
            fontsize=9, rotation=90, ha="right", va="bottom")
    ax.text(dJ2, 2.15, f"θ = ΔJ(2) = {dJ2:.1f} pp", color=PALETTE["C2"],
            fontsize=9, rotation=90, ha="right", va="bottom")
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["1", "2", "∞"])
    ax.set_xlabel("detector threshold $\\theta$ (pp of coverage loss)")
    ax.set_ylabel("$k^*(\\theta)$ — smallest attacker that crosses θ")
    ax.set_title("$k^*(\\theta)$ staircase — defender's sensitivity curve",
                 fontsize=11)
    ax.set_xlim(0, 20); ax.set_ylim(0.5, 3.6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kstar_staircase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §8.2
def resilience_triangle():
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "polar"})
    axes_labels = ["Magnitude\n|ΔJ|/J_clean", "Brittleness\nσ_att / σ_clean", "Timeliness\n1 − T(90%)/T_budget"]
    T_budget = 200
    angles = np.linspace(0, 2 * np.pi, 4)[:-1]  # 3 axes
    metrics = {}
    for k in ["B", "C1", "C2"]:
        cov = FINALS[k]
        mag = abs(cov.mean() - B_CLEAN_MEAN) / B_CLEAN_MEAN
        brit = cov.std(ddof=1) / FINALS["B"].std(ddof=1)
        t90s = [_t_at_threshold(CURVES[k][s], 90) for s in range(CURVES[k].shape[0])]
        valid = [t for t in t90s if t is not None]
        if len(valid) >= CURVES[k].shape[0] // 2:
            t90 = float(np.median(valid))
            timeliness = max(0.0, 1.0 - t90 / T_budget)
        else:
            timeliness = 0.0
        metrics[k] = np.array([mag, brit / 4.0, 1 - timeliness])  # normalize brit by /4 for visual
        # We invert timeliness so "bigger = worse" on all 3 axes → polygon radius = attacker score
    for k in ["B", "C1", "C2"]:
        v = metrics[k]
        vals = np.concatenate([v, v[:1]])
        ang = np.concatenate([angles, angles[:1]])
        ax.plot(ang, vals, label=SHORT[k], color=PALETTE[k], linewidth=2.2)
        ax.fill(ang, vals, color=PALETTE[k], alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title("Attack-Resilience Triangle  (bigger polygon = attacker winning more)\n"
                 "[Brittleness axis /4 for visual; raw σ ratios: B=1.00, C1=1.63, C2=3.37]",
                 fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "resilience_triangle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §6.4
def channels_stacked():
    # Numeric values from §6.4 interim table (ground-truth from episode state,
    # frozen in the meta-report). If these ever change, regenerate from
    # scripts/meta_report.py's team-belief merge at t=200.
    CHANNEL_VALS = {  # {setup: [blue-only, BOTH, red-only, nobody]}
        "B":  [100.0, 0.0,  0.0, 0.0],
        "C1": [53.1, 41.3,  0.0, 5.6],
        "C2": [52.6, 36.2,  2.0, 9.2],
    }
    colors = ["#6fa8dc", "#cd87cd", "#ff9033", "#222222"]
    labels = ["blue-only (productive)", "BOTH (redundant labour)",
              "red-only (hoarded)", "nobody (mission miss)"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.8))
    setups = ["B", "C1", "C2"]
    y = np.arange(len(setups))
    for i, setup in enumerate(setups):
        left = 0.0
        for j, (val, col) in enumerate(zip(CHANNEL_VALS[setup], colors)):
            ax.barh(y[i], val, left=left, color=col, edgecolor="white", linewidth=1.0,
                    label=labels[j] if i == 0 else None)
            if val >= 4.0:
                ax.text(left + val / 2, y[i], f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, color="white" if j in (1, 3) else "black")
            left += val
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[s] for s in setups], fontsize=10)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("% of non-wall cells at t = 200")
    ax.set_title("Two separable attack channels — cell attribution at episode end  (canonical seed 0)",
                 fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "channels_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §5.1 diff
def budget_match_diff():
    # Matched M: M=0.25 (k=1 ρ=0.25 vs k=2 ρ=0.125 — we don't have 0.125 so skip),
    # M=0.5 (k=1 ρ=0.5 vs k=2 ρ=0.25), M=1.0 (k=1 ρ=1.0 vs k=2 ρ=0.5),
    # M=1.5 (k=1 ρ=1.5 — we don't have), M=0.75 (k=1 ρ=0.75 vs k=2 ρ=0.375 — skip).
    # We have only the (k, ρ) pairs in BK/BR; find matches we can compute.
    # k=1 indices: BK==1 → rows 0..4, ρ in {0,0.25,0.5,0.75,1}
    # k=2 indices: BK==2 → rows 5..9, ρ in {0,0.25,0.5,0.75,1}
    def _idx(k_val, rho_val):
        matches = np.where((BK == k_val) & np.isclose(BR, rho_val))[0]
        return int(matches[0]) if len(matches) else None

    # Pairs where k=1 ρ=X and k=2 ρ=X/2 both exist in the sweep:
    # M=0.5: k=1 ρ=0.5 and k=2 ρ=0.25 ✓
    # M=1.0: k=1 ρ=1.0 and k=2 ρ=0.5  ✓
    # M=0 (degenerate): skip — both 0.
    pairs = [
        (0.5, _idx(1, 0.5), _idx(2, 0.25)),
        (1.0, _idx(1, 1.0), _idx(2, 0.5)),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.4))
    xs = np.arange(len(pairs))
    mids = []
    lows = []
    highs = []
    annots = []
    for i, (m, i1, i2) in enumerate(pairs):
        f1 = BF[i1]  # (seeds,)  k=1 ρ=M
        f2 = BF[i2]  # k=2 ρ=M/2
        diff = f1 - f2
        mid = float(diff.mean())
        lo, hi = _bootstrap_ci(diff)
        mids.append(mid); lows.append(lo); highs.append(hi)
        annots.append(f"k=1 ρ={m:g}\nvs k=2 ρ={m/2:g}")
    bars = ax.bar(xs, mids, yerr=[np.array(mids) - np.array(lows),
                                  np.array(highs) - np.array(mids)],
                  color=["#4477aa", "#ee6677"], edgecolor="black", linewidth=0.5,
                  capsize=5)
    ax.axhline(0, color="#555", linestyle="--", linewidth=1)
    for i, (v, lo, hi) in enumerate(zip(mids, lows, highs)):
        ax.text(i, v + (1 if v >= 0 else -1) * 0.3, f"{v:+.2f} pp\n[{lo:+.2f}, {hi:+.2f}]",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(annots, fontsize=9)
    ax.set_ylabel("$J(k{=}1,\\rho{=}M) - J(k{=}2,\\rho{=}M/2)$  (pp)")
    ax.set_title("Matched-$M$ difference — scalar-$M$ would predict zero\n"
                 "(positive bar = splitting same budget across 2 agents hurts blue more)",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "budget_match_diff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §5.1 raincloud
def budget_raincloud_grid():
    rhos = sorted(set(BR.tolist()))
    ks = sorted(set(BK.tolist()))
    fig, axes = plt.subplots(len(ks), len(rhos),
                             figsize=(2.5 * len(rhos), 2.4 * len(ks)),
                             sharex=True, sharey=True, squeeze=False)
    for ri, k_val in enumerate(ks):
        for ci, rho_val in enumerate(rhos):
            m = np.where((BK == k_val) & np.isclose(BR, rho_val))[0]
            ax = axes[ri, ci]
            if len(m) == 0:
                ax.set_visible(False)
                continue
            vals = BF[int(m[0])]
            # raw strip
            x_jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(x_jitter, vals, s=16, color="#d62728", alpha=0.6, zorder=3)
            # half-violin (right side only)
            try:
                parts = ax.violinplot([vals], positions=[0.18], widths=0.25,
                                      showmeans=False, showmedians=False,
                                      showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#d62728"); pc.set_alpha(0.25)
                    paths = pc.get_paths()[0]
                    verts = paths.vertices
                    verts[:, 0] = np.clip(verts[:, 0], 0.18, None)
                    paths.vertices = verts
            except Exception:
                pass
            ax.axhline(vals.mean(), xmin=0.35, xmax=0.65,
                       color="black", linewidth=1.5, zorder=4)
            ax.text(0.5, vals.mean() + 1.2, f"μ={vals.mean():.1f}",
                    ha="center", va="bottom", fontsize=7.5)
            if ri == 0:
                ax.set_title(f"ρ = {rho_val:g}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"k = {k_val}\ncoverage (%)", fontsize=9)
            ax.set_xticks([])
            ax.set_xlim(-0.3, 0.55)
            ax.set_ylim(50, 105)
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-cell raincloud — seed-level distributions across $(k, \\rho)$  "
                 f"[{BS_SEEDS} seeds per cell]", fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "budget_raincloud_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §5.3 pareto
def budget_pareto():
    # Pareto: x = M = k·ρ (total deviation mass), y = ΔJ = B_clean - mean(finals)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.2))
    Ms = BK * BR
    dJ = B_CLEAN_MEAN - BF.mean(axis=1)
    markers = {1: "o", 2: "s"}
    for kv in sorted(set(BK.tolist())):
        mask = BK == kv
        sizes = 40 + 180 * BR[mask]
        ax.scatter(Ms[mask], dJ[mask], s=sizes, marker=markers[kv],
                   c=BR[mask], cmap="viridis", vmin=0, vmax=1.0,
                   edgecolor="black", linewidth=0.6,
                   label=f"k = {kv}  (marker size ∝ ρ)")
        for i in np.where(mask)[0]:
            ax.annotate(f"ρ={BR[i]:g}", (Ms[i], dJ[i]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    # Envelopes
    order = np.argsort(Ms)
    Ms_s = Ms[order]; dJ_s = dJ[order]
    # upper (max ΔJ at each M — attacker-optimal)
    unique_M = sorted(set(Ms_s.tolist()))
    upper = [max(dJ_s[Ms_s == m]) for m in unique_M]
    lower = [min(dJ_s[Ms_s == m]) for m in unique_M]
    ax.plot(unique_M, upper, color="#a00", linestyle="--", linewidth=1.2,
            label="upper envelope (attacker-optimal)", zorder=5)
    ax.plot(unique_M, lower, color="#060", linestyle="--", linewidth=1.2,
            label="lower envelope (best-case)", zorder=5)
    ax.axhline(0, color="#555", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("total deviation mass  $M = k \\cdot \\rho$")
    ax.set_ylabel("$\\Delta J$ = $J_\\mathrm{clean}$ − $J(k, \\rho)$  (pp)")
    ax.set_title("Budget Pareto — damage vs mass, labelled by (k, ρ)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "budget_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- §8.1 model
def model_decomposition():
    # Hypothesised ΔJ ≈ α·k + β_C·Σρ + γ·C(k,2)·σ₀²
    # Calibrate from: ΔJ(k=2, ρ=0) = 16.6 → 2α = 16.6 ⇒ α = 8.3
    #                  ΔJ(k=1, ρ=1) = ~8  → α + β_C·1 = 8 ⇒ β_C = -0.3 (near 0)
    # Actually let's compute α from BS_NPZ directly.
    alpha = 0.5 * (B_CLEAN_MEAN - BF[np.where((BK == 2) & (BR == 0))[0][0]].mean())
    # β_C: use k=1 slope in ρ
    idx_k1 = np.where(BK == 1)[0]
    rho_vals_k1 = BR[idx_k1]
    dJ_k1 = B_CLEAN_MEAN - BF[idx_k1].mean(axis=1)
    # Fit dJ_k1 ≈ alpha*1 + beta_C * rho → beta_C = slope of (dJ - alpha) vs rho
    residual = dJ_k1 - alpha
    slope = np.polyfit(rho_vals_k1, residual, deg=1)[0]
    beta_C = slope
    # γ: use ΔJ(k=2, ρ=1) = 2α + 2β_C + γ·(2C2)·σ₀² ; solve
    dJ_k2_r1 = B_CLEAN_MEAN - BF[np.where((BK == 2) & np.isclose(BR, 1.0))[0][0]].mean()
    # Variance inflation: use σ² of finals in C2 as σ₀²
    sigma0_sq = FINALS["C2"].var()
    # k=2 gives C(2,2)=1 pairs. So γ = (dJ_k2_r1 - 2α - 2β_C) / sigma0_sq
    gamma = (dJ_k2_r1 - 2 * alpha - 2 * beta_C) / max(sigma0_sq, 1.0)

    # Apply model to each (k, ρ) in budget_sweep
    n_pairs_k = BK * (BK - 1) / 2
    model_pred = alpha * BK + beta_C * (BK * BR) + gamma * n_pairs_k * sigma0_sq
    measured = B_CLEAN_MEAN - BF.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel (a): stacked bar of model components vs measured ΔJ
    xs = np.arange(len(BK))
    label_pairs = [f"k={k},ρ={r:g}" for k, r in zip(BK, BR)]
    comp_alpha = alpha * BK
    comp_beta = beta_C * (BK * BR)
    comp_gamma = gamma * n_pairs_k * sigma0_sq
    axes[0].bar(xs - 0.2, comp_alpha, width=0.4, color="#1f77b4",
                label=f"α·k  (α={alpha:.1f} pp)", edgecolor="black", linewidth=0.4)
    axes[0].bar(xs - 0.2, comp_beta, width=0.4, bottom=comp_alpha, color="#ff7f0e",
                label=f"β_C·Σρ  (β_C={beta_C:+.2f})", edgecolor="black", linewidth=0.4)
    axes[0].bar(xs - 0.2, comp_gamma, width=0.4,
                bottom=comp_alpha + comp_beta, color="#d62728",
                label=f"γ·(k,2)·σ₀²  (γ={gamma:+.3f})", edgecolor="black", linewidth=0.4)
    axes[0].bar(xs + 0.2, measured, width=0.4, color="#888", edgecolor="black",
                linewidth=0.4, label="measured ΔJ")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(label_pairs, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("ΔJ  (pp)")
    axes[0].set_title("(a) Model decomposition vs measured  (left = predicted, right = actual)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].axhline(0, color="#555", linestyle="--", linewidth=0.8)

    # Panel (b): residual scatter
    resid = measured - model_pred
    axes[1].scatter(model_pred, measured, s=60, c=BR, cmap="viridis",
                    edgecolor="black", linewidth=0.5)
    lim = max(measured.max(), model_pred.max()) + 2
    axes[1].plot([-2, lim], [-2, lim], color="#555", linestyle="--", linewidth=1)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = 1 - np.sum(resid ** 2) / np.sum((measured - measured.mean()) ** 2)
    for i, lbl in enumerate(label_pairs):
        axes[1].annotate(lbl, (model_pred[i], measured[i]),
                         xytext=(4, 4), textcoords="offset points", fontsize=6.5, alpha=0.8)
    axes[1].set_xlabel("predicted ΔJ (model)")
    axes[1].set_ylabel("measured ΔJ")
    axes[1].set_title(f"(b) predicted vs measured   RMSE = {rmse:.2f} pp   R² = {r2:.2f}")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("ΔJ decomposition: α·k + β_C·Σρ + γ·(k choose 2)·σ₀²   "
                 "— hypothesised model, calibrated on budget sweep",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generators = [
        ("claim1_invariant",          claim1_invariant),
        ("time_to_coverage_multiseed", time_to_coverage_multiseed),
        ("forest_delta_j",            forest_delta_j),
        ("variance_bar",              variance_bar),
        ("kstar_staircase",           kstar_staircase),
        ("resilience_triangle",       resilience_triangle),
        ("channels_stacked",          channels_stacked),
        ("budget_match_diff",         budget_match_diff),
        ("budget_raincloud_grid",     budget_raincloud_grid),
        ("budget_pareto",             budget_pareto),
        ("model_decomposition",       model_decomposition),
    ]
    for name, fn in generators:
        try:
            fn()
            print(f"  wrote {OUT_DIR / (name + '.png')}")
        except Exception as exc:
            print(f"  FAILED {name}: {exc}")
            raise


if __name__ == "__main__":
    main()
