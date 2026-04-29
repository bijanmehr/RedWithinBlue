"""Replacement visualisation for meta-report §6.6 "Residual fog footprint".

v1/v2 showed three side-by-side grid maps with red-tinted unknown cells and
agent dots. That viz is compact but (a) the reader has to eyeball the red
attractor and (b) it does not aggregate across eval seeds.

This script replaces it with two aggregated figures computed from real
rollouts across 20 eval seeds per setup (B, C1, C2), t=200:

  Figure 1 · proximity_ecdf.png
    ECDF of distance (Chebyshev) from each *unknown non-wall* cell to the
    nearest red agent, pooled across seeds. For B (no red), distance to the
    nearest blue for reference. C2's curve should crush toward the left.

  Figure 2 · proximity_bands.png
    Horizontal stacked bars per setup, partitioning non-wall cells into
      {known, unknown within 0-2, unknown 3-5, unknown 6-9, unknown ≥ 10}
    of the nearest red (or blue for B). Shows both residual-fog magnitude
    AND its spatial co-location with red.

Output JSON `proximity_summary.json` captures the raw bin fractions + p50 /
p75 / p90 of the ECDF per setup so the meta-report can cite numbers.

Run:
  python scripts/meta_report_fog_proximity.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Re-use the loaders + rollout helper already in meta_report.py. This keeps
# the env / checkpoint / snapshot logic in one place.
from meta_report import (
    SETUPS,
    MAX_STEPS_EVAL,
    SNAPSHOT_STEPS,
    OUT_DIR,
    _load_blue,
    _load_red,
    _rollout_with_snapshots,
    Setup,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.types import CELL_WALL


N_SEEDS = 20
FIG_DIR = OUT_DIR
BANDS = [(0, 2), (3, 5), (6, 9), (10, 99)]
BAND_LABELS = ["0–2", "3–5", "6–9", "≥10"]
BAND_COLORS = ["#8b0000", "#d62728", "#ff9896", "#fcd5d2"]  # dark → light red
KNOWN_COLOR = "#c7dbec"

PALETTE = {"B": "#1f77b4", "C1": "#ff7f0e", "C2": "#d62728"}


def _chebyshev_distance_field(H: int, W: int, pts: np.ndarray) -> np.ndarray:
    """Per-cell distance to nearest point in `pts` (rows, cols). Chebyshev
    because GridCommEnv's movement primitive is 8-connected (king moves).
    Returns (H, W) float32; inf if pts is empty."""
    if len(pts) == 0:
        return np.full((H, W), np.inf, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.inf * np.ones((H, W), dtype=np.float32)
    for r, c in pts:
        dd = np.maximum(np.abs(yy - r), np.abs(xx - c))
        d = np.minimum(d, dd)
    return d


def _collect_one_setup(setup: Setup, n_seeds: int) -> Dict[str, np.ndarray]:
    """Returns dict with arrays pooled across seeds:
      unknown_dists  — distance to nearest red (or blue for B) for unknown cells
      known_dists    — same, for known cells
      n_nonwall      — total non-wall cells per seed
      n_unknown      — unknown count per seed (for magnitude)
      bandfrac       — (n_seeds, 5) fraction of non-wall in {known, u0-2, u3-5, u6-9, u≥10}
    """
    cfg = ExperimentConfig.from_yaml(setup.config)
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    red_actor, red_params = (None, None)
    if setup.red_ckpt is not None:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

    unknown_dists: List[np.ndarray] = []
    known_dists: List[np.ndarray] = []
    n_unknown: List[int] = []
    n_nonwall: List[int] = []
    bandfracs: List[np.ndarray] = []

    for s in range(n_seeds):
        snaps, _curve, _paths, _tids = _rollout_with_snapshots(
            cfg, blue_actor, blue_params, red_actor, red_params,
            seed=s, max_steps=MAX_STEPS_EVAL, snapshot_steps=SNAPSHOT_STEPS,
        )
        snap = snaps[-1]
        terrain = snap["terrain"]
        H, W = terrain.shape
        non_wall = terrain != CELL_WALL
        unknown = non_wall & (~snap["blue_ever"])
        known = non_wall & snap["blue_ever"]

        # Reference points for distance. Red if any red, else all blues.
        positions = snap["positions"]
        team_ids = snap["team_ids"]
        red_pts = positions[team_ids == 1]
        if len(red_pts) == 0:
            ref_pts = positions[team_ids == 0]
        else:
            ref_pts = red_pts

        dist_field = _chebyshev_distance_field(H, W, ref_pts)
        unknown_dists.append(dist_field[unknown])
        known_dists.append(dist_field[known])
        n_unknown.append(int(unknown.sum()))
        n_nonwall.append(int(non_wall.sum()))

        # Distance-band buckets
        total = max(1, non_wall.sum())
        known_frac = known.sum() / total
        band_fracs = [known_frac]
        for lo, hi in BANDS:
            in_band = unknown & (dist_field >= lo) & (dist_field <= hi)
            band_fracs.append(in_band.sum() / total)
        bandfracs.append(np.asarray(band_fracs, dtype=np.float32))

    return {
        "unknown_dists": np.concatenate(unknown_dists) if unknown_dists else np.array([]),
        "known_dists": np.concatenate(known_dists) if known_dists else np.array([]),
        "n_unknown": np.asarray(n_unknown),
        "n_nonwall": np.asarray(n_nonwall),
        "bandfrac": np.stack(bandfracs, axis=0),  # (n_seeds, 5)
    }


def _plot_ecdf(results: Dict[str, Dict[str, np.ndarray]], out_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    for setup in SETUPS:
        d = results[setup.key]["unknown_dists"]
        if len(d) == 0:
            continue
        dd = np.sort(d)
        y = np.arange(1, len(dd) + 1) / len(dd)
        label = f"{setup.short}  (n_unk={len(dd)})"
        ls = "--" if setup.key == "B" else "-"
        ax.plot(dd, y, label=label, color=PALETTE.get(setup.key, "k"),
                linewidth=2.2, linestyle=ls)
    ax.set_xlabel("distance (cells, Chebyshev) from unknown cell to nearest red agent"
                  "\n(for B: to nearest blue)", fontsize=9)
    ax.set_ylabel("cumulative fraction of unknown cells")
    ax.set_title("ECDF — where do the residual-fog cells sit relative to the nearest red?"
                 "\n20 eval seeds per setup, t = 200", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bands(results: Dict[str, Dict[str, np.ndarray]], out_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.2))
    y_positions = np.arange(len(SETUPS))
    for i, setup in enumerate(SETUPS):
        bf = results[setup.key]["bandfrac"].mean(axis=0)  # (5,)
        left = 0.0
        colors = [KNOWN_COLOR] + BAND_COLORS
        labels_once = (i == 0)
        labels = ["known (blue ever saw)"] + [f"unknown @ {b}" for b in BAND_LABELS]
        for j, (val, col, lab) in enumerate(zip(bf, colors, labels)):
            ax.barh(
                y_positions[i], val, left=left, color=col,
                edgecolor="white", linewidth=0.7,
                label=lab if labels_once else None,
            )
            if val >= 0.03:
                ax.text(left + val / 2, y_positions[i],
                        f"{val * 100:.1f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if j in (0, 1, 2) else "black")
            left += val
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s.short for s in SETUPS], fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of non-wall cells (mean across 20 seeds)")
    ax.set_title("Residual fog by distance-to-nearest-red band"
                 "\n(C2's unknown strip sits close to a red; B's residual ~0%)",
                 fontsize=10)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary(results: Dict[str, Dict[str, np.ndarray]]) -> dict:
    out = {}
    for setup in SETUPS:
        r = results[setup.key]
        d = r["unknown_dists"]
        qs = {}
        if len(d) > 0:
            qs = {
                "p50": float(np.percentile(d, 50)),
                "p75": float(np.percentile(d, 75)),
                "p90": float(np.percentile(d, 90)),
                "mean": float(np.mean(d)),
            }
        out[setup.key] = {
            "n_seeds": int(r["bandfrac"].shape[0]),
            "mean_unknown_pct": float(100.0 * r["n_unknown"].mean() / max(1, r["n_nonwall"].mean())),
            "unknown_dist": qs,
            "bandfrac_mean": {
                "known": float(r["bandfrac"][:, 0].mean()),
                **{f"unknown_{lab}": float(r["bandfrac"][:, 1 + j].mean())
                   for j, lab in enumerate(BAND_LABELS)},
            },
        }
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for setup in SETUPS:
        print(f"rolling out {setup.key} x {N_SEEDS} seeds …")
        results[setup.key] = _collect_one_setup(setup, N_SEEDS)
        s = results[setup.key]
        print(f"  mean unknown = {100.0 * s['n_unknown'].mean() / max(1, s['n_nonwall'].mean()):.2f}%"
              f"   p50 d(unk→red) = "
              f"{np.percentile(s['unknown_dists'], 50) if len(s['unknown_dists']) else float('nan'):.2f}")

    ecdf_png = FIG_DIR / "proximity_ecdf.png"
    band_png = FIG_DIR / "proximity_bands.png"
    _plot_ecdf(results, ecdf_png)
    _plot_bands(results, band_png)
    print(f"  wrote {ecdf_png}")
    print(f"  wrote {band_png}")

    summary = _summary(results)
    (FIG_DIR / "proximity_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {FIG_DIR / 'proximity_summary.json'}")


if __name__ == "__main__":
    main()
