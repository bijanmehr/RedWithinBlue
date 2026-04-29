"""Experimental visualization gallery for meta_report_experimental.html.

Goal: take the SAME data the v3 report summarises (compromise_compare.npz +
budget_sweep.npz + hetero_sweep.npz) and re-project it through ~10
unconventional visualization styles, looking for patterns the canonical
line-plots/bar-charts miss.

Each figure is one experimental style. None are load-bearing for any
scientific claim — they are exploratory. Where a figure surfaces something
unexpected, the HTML callout flags it as a hypothesis to verify.

Run:
    python scripts/meta_report_experimental.py

Outputs under experiments/meta-report/ (prefix `exp_`).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np


OUT = Path("experiments/meta-report")
OUT.mkdir(parents=True, exist_ok=True)

COMP = np.load("experiments/compromise-compare/compromise_compare.npz")
BUD = np.load("experiments/misbehavior-budget/budget_sweep.npz")
HET = np.load("experiments/misbehavior-budget/hetero_sweep.npz")

# setup shorthand
SETUPS = ["B", "C1", "C2"]
FINALS = {s: COMP[f"finals_{i + 1}"] for i, s in enumerate(SETUPS)}   # 20 seeds each
CURVES = {s: COMP[f"curve_{i + 1}"] for i, s in enumerate(SETUPS)}    # 20×200
SOLO_FINALS = COMP["finals_0"]     # 20 seeds at N=1
COLORS = {"B": "#2ca02c", "C1": "#e08d2a", "C2": "#d62728", "S": "#777"}

N_SEEDS, T_MAX = CURVES["B"].shape


# =====================================================================
# Fig 1 — 20-seed small multiples (facet-wrap)
# =====================================================================
def fig_small_multiples(out: Path) -> None:
    """Every seed its own tiny subplot; overlay B/C1/C2 inside each.
    Let the reader eyeball which seeds are attacker-hard vs easy.
    """
    fig, axes = plt.subplots(4, 5, figsize=(16, 9.5), sharex=True, sharey=True)
    t = np.arange(T_MAX)
    for seed in range(N_SEEDS):
        ax = axes[seed // 5, seed % 5]
        for s in SETUPS:
            ax.plot(t, CURVES[s][seed], color=COLORS[s], linewidth=1.4,
                    label=s if seed == 0 else None, alpha=0.9)
        ax.axhline(90, color="#888", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 102)
        ax.set_xlim(0, T_MAX)
        ax.grid(True, alpha=0.25)
        # per-seed final-gap annotation
        gap = CURVES["B"][seed, -1] - CURVES["C2"][seed, -1]
        col_gap = "#d62728" if gap > 15 else ("#e08d2a" if gap > 5 else "#2ca02c")
        ax.text(0.03, 0.06, f"seed {seed:02d}\nB−C2 = {gap:+.1f} pp",
                transform=ax.transAxes, fontsize=7, color=col_gap,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        if seed == 0:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    for i in range(4):
        axes[i, 0].set_ylabel("coverage (%)", fontsize=8)
    for j in range(5):
        axes[-1, j].set_xlabel("episode step t", fontsize=8)
    fig.suptitle("20-seed small multiples — every seed's B / C1 / C2 curve, "
                 "color-coded by final B–C2 gap", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 2 — Ridgeline plot of coverage distributions at 4 time slices
# =====================================================================
def fig_ridgeline(out: Path) -> None:
    slices = [50, 100, 150, 199]
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.0))

    row_h = 0.9
    y_off = 0.0
    y_ticks = []
    y_labels = []

    for ti, t in enumerate(slices):
        for s in SETUPS:
            vals = CURVES[s][:, t]
            kde_x = np.linspace(0, 100, 200)
            # crude Gaussian KDE
            sig = max(vals.std(), 1.5)
            dens = np.exp(-((kde_x[:, None] - vals[None, :]) / sig) ** 2).sum(axis=1)
            dens = dens / dens.max() * row_h
            ax.fill_between(kde_x, y_off, y_off + dens, color=COLORS[s],
                            alpha=0.55, edgecolor=COLORS[s], linewidth=1.2)
            # rug
            ax.scatter(vals, np.full(len(vals), y_off - 0.04),
                       color=COLORS[s], s=12, marker="|", alpha=0.8)
            y_ticks.append(y_off + row_h / 2)
            y_labels.append(f"t={t}  {s}")
            y_off += row_h + 0.1
        y_off += 0.25  # gap between time-groups

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("coverage (%)")
    ax.axvline(90, color="#888", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.text(90.5, y_off - 0.5, "  θ = 90%", fontsize=8, color="#444")
    ax.set_title("Ridgeline plot — 20-seed coverage distributions at t ∈ {50,100,150,199}\n"
                 "vertical gaps between time slices; C2 flattens out (fatter tail) at late t",
                 fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 3 — Streamgraph of coverage segmentation over time
# =====================================================================
def fig_streamgraph(out: Path) -> None:
    """Approximate map-ownership streamgraph using the coverage curves.
    Streams: solid-ground covered (mean over 20 seeds) and still-unknown
    (100 − cov). Stack per setup in a vertical 3-panel layout.
    """
    t = np.arange(T_MAX)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.0), sharex=True)
    for ax, s in zip(axes, SETUPS):
        cov = CURVES[s].mean(axis=0)
        p10 = np.percentile(CURVES[s], 10, axis=0)
        p90 = np.percentile(CURVES[s], 90, axis=0)

        ax.fill_between(t, 0, cov, color=COLORS[s], alpha=0.7,
                        label=f"{s} — covered (mean)")
        ax.fill_between(t, cov, 100, color="#d0d0d0", alpha=0.7,
                        label="still unknown")
        # p10/p90 as a darker band around the boundary
        ax.fill_between(t, p10, p90, color=COLORS[s], alpha=0.4,
                        label=f"{s} — p10–p90 band")
        ax.plot(t, cov, color="black", linewidth=1.2, alpha=0.8)
        ax.axhline(90, color="#222", linewidth=0.5, linestyle="--", alpha=0.7)
        ax.set_ylim(0, 100); ax.set_xlim(0, T_MAX)
        ax.set_ylabel(f"{s}", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("episode step t")
    fig.suptitle("Streamgraph — per-setup map-ownership over time (coverage / unknown)\n"
                 "black line = 20-seed mean; coloured band = p10–p90",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 4 — Bump chart: how does each seed's rank change B → C1 → C2?
# =====================================================================
def fig_bump_chart(out: Path) -> None:
    """Rank 20 seeds by final coverage in each setup. A "stable" attack
    preserves rank; a "chaotic" attack reshuffles.
    """
    def ranks(v):
        order = np.argsort(-v)                  # high cov → rank 0
        r = np.empty_like(order, dtype=int)
        r[order] = np.arange(len(order))
        return r

    rB = ranks(FINALS["B"])
    rC1 = ranks(FINALS["C1"])
    rC2 = ranks(FINALS["C2"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for seed in range(N_SEEDS):
        ys = [rB[seed], rC1[seed], rC2[seed]]
        # colour the bump line by how much the rank moved
        total_move = abs(rC2[seed] - rB[seed])
        col = plt.cm.RdYlGn_r(total_move / max(1, N_SEEDS - 1))
        ax.plot([0, 1, 2], ys, color=col, linewidth=2.0, alpha=0.85,
                marker="o", markersize=7, markeredgecolor="black",
                markeredgewidth=0.6)
        # label seed at the right
        ax.text(2.05, rC2[seed], f"s{seed:02d}", fontsize=7,
                color="#333", va="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["B (clean)", "C1 (k=1)", "C2 (k=2)"], fontsize=11)
    ax.set_yticks(range(N_SEEDS))
    ax.set_yticklabels([f"rank {r + 1}" for r in range(N_SEEDS)], fontsize=7)
    ax.invert_yaxis()   # rank 1 at top
    ax.grid(True, alpha=0.3)
    ax.set_title("Bump chart — does the attack preserve seed ranking?\n"
                 "Line colour = |rank(B) − rank(C2)|; darker-red = more shuffled\n"
                 "If the attack is a uniform ~10 pp shift, lines stay parallel; if "
                 "the attack is seed-dependent, lines cross.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 5 — Q-Q plot: B vs C1, B vs C2
# =====================================================================
def fig_qq(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    pairs = [("B", "C1"), ("B", "C2")]
    for ax, (a, b) in zip(axes, pairs):
        sa = np.sort(FINALS[a])
        sb = np.sort(FINALS[b])
        lo = min(sa.min(), sb.min()) - 2
        hi = max(sa.max(), sb.max()) + 2
        ax.plot([lo, hi], [lo, hi], color="#888", linewidth=0.8, linestyle="--",
                label="y = x (identical dist)")
        # shift-only prediction: y = x - ΔJ_mean
        dJ = FINALS[a].mean() - FINALS[b].mean()
        ax.plot([lo, hi], [lo - dJ, hi - dJ], color=COLORS[b], linewidth=0.8,
                linestyle=":", label=f"y = x − ΔJ ({dJ:+.1f} pp shift)")
        ax.scatter(sa, sb, color=COLORS[b], s=60, edgecolors="black",
                   linewidths=0.7, zorder=5)
        ax.set_xlabel(f"{a} quantile (coverage %)")
        ax.set_ylabel(f"{b} quantile (coverage %)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        # annotate the largest deviator
        devs = sb - (sa - dJ)
        i = int(np.argmax(np.abs(devs)))
        ax.annotate(f"Δ(shift-pred) = {devs[i]:+.1f} pp",
                    xy=(sa[i], sb[i]),
                    xytext=(15, -15), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#555", lw=0.6))
        ax.set_title(f"Q-Q: {a} vs {b}")

    fig.suptitle("Q-Q plots — is the attack a uniform shift or a distribution change?\n"
                 "Points on dotted line → shift-only; curvature → variance or shape change",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 6 — Polar "mission-clock" plot — T(θ)/T_budget per setup, several θ
# =====================================================================
def _time_to_threshold(curve: np.ndarray, theta: float) -> int | None:
    idx = np.argmax(curve >= theta)
    if curve[idx] < theta:
        return None
    return int(idx)


def fig_polar_clock(out: Path) -> None:
    """For each setup and each θ ∈ {50, 70, 80, 90, 95}, plot the median
    T(θ) as a polar wedge. Budget = T_MAX → full circle.
    """
    thetas = [50, 70, 80, 90, 95]
    angle_per_theta = 2 * np.pi / len(thetas)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8),
                             subplot_kw=dict(projection="polar"))
    for ax, s in zip(axes, SETUPS):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for ti, th in enumerate(thetas):
            vals = np.array([
                _time_to_threshold(CURVES[s][seed], th) or T_MAX
                for seed in range(N_SEEDS)
            ])
            reached_frac = float(np.mean(vals < T_MAX))
            med_t = float(np.median(vals))
            # radius = normalized-clock use (inner = reached fast, edge = never)
            r = med_t / T_MAX
            wedge_angle = angle_per_theta * 0.8
            theta_center = ti * angle_per_theta
            ax.bar(theta_center, r, width=wedge_angle, bottom=0.0,
                   color=COLORS[s], alpha=0.65 + 0.07 * ti, edgecolor="black")
            ax.text(theta_center, r + 0.06,
                    f"θ={th}\nmed T={med_t:.0f}\nreach={reached_frac:.0%}",
                    ha="center", va="center", fontsize=7,
                    color="#333")
        ax.set_ylim(0, 1.2)
        ax.set_yticklabels([])
        ax.set_xticks([ti * angle_per_theta for ti in range(len(thetas))])
        ax.set_xticklabels([f"θ={t}%" for t in thetas], fontsize=8)
        ax.set_title(f"{s}", fontsize=11, pad=12)
    fig.suptitle("Polar 'mission-clock' plot — wedge radius = median T(θ)/T_budget; "
                 "closer to centre = reaches θ fast; reaching edge = never",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 7 — ρ_A × ρ_B 2D density contour from hetero sweep
# =====================================================================
def fig_rho_density(out: Path) -> None:
    sigma = HET["sigma"]
    rho_a = HET["rho_a"]
    rho_b = HET["rho_b"]
    finals = HET["finals"]             # (12, 15)
    clean_mean = float(HET["clean_finals"].mean())
    deltaJ = clean_mean - finals.mean(axis=1)

    # Build a coarse grid by triangulating (symmetric ρ_A/ρ_B plane) —
    # sweep only lies on Σρ = 0.5 and Σρ = 1.0 lines, so render
    # those two line-traces + a scatter with filled markers.

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # Panel A: ρ_A vs ρ_B, coloured points + line traces
    ax = axes[0]
    cmap = plt.cm.coolwarm
    dJ_min, dJ_max = float(np.min(deltaJ)), float(np.max(deltaJ))
    norm = plt.Normalize(dJ_min, dJ_max)
    for sig in sorted(set(float(v) for v in sigma)):
        mask = np.isclose(sigma, sig)
        xa, xb = rho_a[mask], rho_b[mask]
        order = np.argsort(xb - xa)
        ax.plot(xa[order], xb[order], color="#999", linewidth=0.8, linestyle="--",
                zorder=1)
    sc = ax.scatter(rho_a, rho_b, c=deltaJ, cmap=cmap, norm=norm,
                    s=180, edgecolors="black", linewidths=0.8, zorder=3)
    # symmetrize by reflecting
    ax.scatter(rho_b, rho_a, c=deltaJ, cmap=cmap, norm=norm,
               s=90, edgecolors="black", linewidths=0.5, alpha=0.55, marker="s")
    ax.plot([0, 1], [0, 1], color="#333", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$\rho_A$"); ax.set_ylabel(r"$\rho_B$")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", "box")
    ax.set_title("ΔJ on the (ρ_A, ρ_B) plane\n"
                 "circles = measured points (ρ_A ≤ ρ_B);  squares = mirror ρ_A ↔ ρ_B",
                 fontsize=9.5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("ΔJ (pp)  — redder = worse for blue")

    # Panel B: ΔJ as a function of concentration c(ρ) = max(ρ_A, ρ_B)
    ax = axes[1]
    c_vals = np.maximum(rho_a, rho_b)
    for sig in sorted(set(float(v) for v in sigma)):
        mask = np.isclose(sigma, sig)
        order = np.argsort(c_vals[mask])
        ax.plot(c_vals[mask][order], deltaJ[mask][order],
                marker="o", markersize=9, linewidth=2.0,
                label=f"Σρ = {sig:.1f}")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xlabel(r"concentration  $c(\rho) = \max_i \rho_i$")
    ax.set_ylabel("ΔJ (pp)")
    ax.set_title("Collapse: ΔJ vs c(ρ) — the v3-proposed feature\n"
                 "if c(ρ) is the right aggregator, Σρ-curves collapse onto each other",
                 fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.suptitle("Hetero-sweep re-projection — 2D (ρ_A, ρ_B) plane + 1D c(ρ) collapse",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 8 — Parallel coordinates across seeds × setups
# =====================================================================
def fig_parallel_coords(out: Path) -> None:
    # Axes: final coverage at S, B, C1, C2
    vals = np.stack([SOLO_FINALS, FINALS["B"], FINALS["C1"], FINALS["C2"]], axis=1)
    labels_axes = ["S", "B", "C1", "C2"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    xs = np.arange(4)
    for seed in range(N_SEEDS):
        col = plt.cm.viridis(seed / (N_SEEDS - 1))
        ax.plot(xs, vals[seed], color=col, linewidth=1.2, alpha=0.7,
                marker="o", markersize=5)
    # summary band
    ax.fill_between(xs,
                    np.percentile(vals, 10, axis=0),
                    np.percentile(vals, 90, axis=0),
                    color="#333", alpha=0.10, label="p10–p90")
    ax.plot(xs, vals.mean(axis=0), color="#333", linewidth=2.5, label="mean",
            marker="s")
    ax.set_xticks(xs); ax.set_xticklabels(labels_axes, fontsize=11)
    ax.set_ylabel("final coverage (%)")
    ax.set_xlim(-0.3, 3.3); ax.set_ylim(25, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Parallel coordinates — 20 seeds across S / B / C1 / C2\n"
                 "each polyline = one seed; crossings = rank flip between setups",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 9 — Budget-sweep 3D surface + 2D isocontour
# =====================================================================
def fig_budget_surface(out: Path) -> None:
    k = BUD["k"]; rho = BUD["rho"]; finals = BUD["finals"]
    mean = BUD["mean"]

    # Infer the (k, rho) grid — the flattened arrays are one point per cell
    # in some ordering. Reshape by unique k and ρ.
    uniq_k = sorted(set(int(v) for v in k))
    uniq_rho = sorted(set(float(v) for v in rho))
    K, R = len(uniq_k), len(uniq_rho)
    ki = {v: i for i, v in enumerate(uniq_k)}
    ri = {v: i for i, v in enumerate(uniq_rho)}
    Z = np.full((K, R), np.nan, dtype=float)
    for i, (ki_, ri_) in enumerate(zip(k, rho)):
        Z[ki[int(ki_)], ri[float(ri_)]] = float(mean[i])

    fig = plt.figure(figsize=(13, 5.6))

    # 3D surface
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    RR, KK = np.meshgrid(uniq_rho, uniq_k)
    surf = ax3.plot_surface(RR, KK, Z, cmap="viridis", alpha=0.85,
                            edgecolor="k", linewidth=0.2)
    ax3.set_xlabel(r"$\rho$"); ax3.set_ylabel("k")
    ax3.set_zlabel("mean final coverage (%)")
    ax3.view_init(elev=25, azim=-55)
    ax3.set_title("3D surface — coverage(ρ, k)", fontsize=10)

    # Heatmap + contour
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(Z, origin="lower", aspect="auto",
                    extent=[min(uniq_rho), max(uniq_rho),
                            min(uniq_k) - 0.5, max(uniq_k) + 0.5],
                    cmap="viridis")
    cs = ax2.contour(RR, KK, Z, levels=[60, 70, 80, 85, 90, 95],
                     colors="white", linewidths=1.0)
    ax2.clabel(cs, inline=True, fontsize=7)
    ax2.set_xlabel(r"$\rho$"); ax2.set_ylabel("k")
    ax2.set_title("Heatmap + iso-coverage contours", fontsize=10)
    fig.colorbar(im, ax=ax2, label="coverage (%)")
    fig.suptitle("Budget sweep (k × ρ) — same data, 3D vs 2D isometric views",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 10 — Cumulative seed-outcomes "swim lanes"
# =====================================================================
def fig_seed_swim(out: Path) -> None:
    """Sort seeds by each setup's final coverage independently, stack
    horizontally — shows the seed-outcome distribution as a swimmable pool.
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    xs_per_setup = {}
    width = 0.25
    for i, s in enumerate(SETUPS):
        sorted_vals = np.sort(FINALS[s])[::-1]   # best-first
        xs = np.arange(N_SEEDS) + i * (N_SEEDS + 3)
        xs_per_setup[s] = xs
        ax.bar(xs, sorted_vals, width=0.8, color=COLORS[s], alpha=0.85,
               edgecolor="black", linewidth=0.4, label=s)
        # p50 line
        ax.axhline(np.median(sorted_vals), xmin=xs[0] / (len(xs) * 3.5),
                   xmax=xs[-1] / (len(xs) * 3.5), color=COLORS[s],
                   linewidth=0, alpha=0)
        ax.plot([xs[0] - 0.4, xs[-1] + 0.4],
                [np.median(sorted_vals)] * 2, color=COLORS[s],
                linewidth=2.4, linestyle=":", alpha=0.95)
        ax.text(xs.mean(), 103, f"{s}\nμ={sorted_vals.mean():.1f}  σ={sorted_vals.std():.1f}",
                ha="center", fontsize=9, fontweight="bold", color=COLORS[s])
    ax.axhline(90, color="#333", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.text(xs_per_setup["C2"][-1] + 1, 90, "θ=90%", fontsize=8, va="center")
    ax.set_xticks([])
    ax.set_ylabel("final coverage (%)")
    ax.set_ylim(50, 110)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Seed-ranked swim lanes — bars sorted within each setup (best→worst)\n"
                 "dotted horizontal = median per setup; the shape of the drop-off tells "
                 "you whether the damage is concentrated or uniform",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 11 — "Gain-loss" waterfall: what does each seed win/lose moving B→C1→C2
# =====================================================================
def fig_waterfall(out: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    order = np.argsort(-FINALS["B"])
    xs = np.arange(N_SEEDS)
    seeds_sorted = order

    # per-seed: the "clean base" (B final), and the delta at C1, C2
    base = FINALS["B"][seeds_sorted]
    d1 = FINALS["C1"][seeds_sorted] - base
    d2 = FINALS["C2"][seeds_sorted] - FINALS["C1"][seeds_sorted]

    ax.bar(xs, base, width=0.7, color=COLORS["B"], alpha=0.55,
           label="B final")
    ax.bar(xs, d1, width=0.7, bottom=base, color=COLORS["C1"], alpha=0.85,
           label="+ ΔC1")
    ax.bar(xs, d2, width=0.7, bottom=base + d1, color=COLORS["C2"], alpha=0.85,
           label="+ ΔC2")
    ax.axhline(90, color="#333", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{s:02d}" for s in seeds_sorted], fontsize=7, rotation=60)
    ax.set_ylabel("coverage (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Per-seed 'waterfall' — B baseline, + k=1 step, + k=2 step\n"
                 "seeds sorted by clean-B ranking; watch which seeds collapse vs stay",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Fig 12 — Correlation matrix between setups, per-seed final coverage
# =====================================================================
def fig_corr_matrix(out: Path) -> None:
    labels = ["S", "B", "C1", "C2"]
    M = np.stack([SOLO_FINALS, FINALS["B"], FINALS["C1"], FINALS["C2"]], axis=1)
    corr = np.corrcoef(M, rowvar=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(corr[i, j]) > 0.4 else "black",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black"
                                                if abs(corr[i, j]) <= 0.4 else "white")])
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title("Seed-level correlation across setups\n"
                 "does 'a hard-seed-for-B' predict 'a hard-seed-for-C2'?\n"
                 "r ≈ 1 → attack is uniform; r ≪ 1 → attack is seed-specific",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    figures = [
        ("exp_small_multiples.png", fig_small_multiples),
        ("exp_ridgeline.png", fig_ridgeline),
        ("exp_streamgraph.png", fig_streamgraph),
        ("exp_bump_chart.png", fig_bump_chart),
        ("exp_qq.png", fig_qq),
        ("exp_polar_clock.png", fig_polar_clock),
        ("exp_rho_density.png", fig_rho_density),
        ("exp_parallel_coords.png", fig_parallel_coords),
        ("exp_budget_surface.png", fig_budget_surface),
        ("exp_seed_swim.png", fig_seed_swim),
        ("exp_waterfall.png", fig_waterfall),
        ("exp_corr_matrix.png", fig_corr_matrix),
    ]
    for i, (name, fn) in enumerate(figures, start=1):
        print(f"[{i}/{len(figures)}] {name}")
        fn(OUT / name)

    # Dump a tiny stats block for the HTML to reference.
    stats = {
        "B_final_mean": float(FINALS["B"].mean()),
        "B_final_std": float(FINALS["B"].std()),
        "C1_final_mean": float(FINALS["C1"].mean()),
        "C1_final_std": float(FINALS["C1"].std()),
        "C2_final_mean": float(FINALS["C2"].mean()),
        "C2_final_std": float(FINALS["C2"].std()),
        "corr_B_C2": float(np.corrcoef(FINALS["B"], FINALS["C2"])[0, 1]),
        "corr_B_C1": float(np.corrcoef(FINALS["B"], FINALS["C1"])[0, 1]),
        "corr_C1_C2": float(np.corrcoef(FINALS["C1"], FINALS["C2"])[0, 1]),
    }
    (OUT / "exp_summary.json").write_text(json.dumps(stats, indent=2))
    print("wrote exp_summary.json")


if __name__ == "__main__":
    main()
