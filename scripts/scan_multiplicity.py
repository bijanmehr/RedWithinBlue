"""How redundant is blue's coverage?

The channels_stacked chart treats each cell as a binary "any blue saw it",
collapsing scan multiplicity into a single yes/no. This script shows the
distribution that's hidden behind that yes/no.

For each setup B / C1 / C2 and each non-wall cell, count:
  - distinct_blue : how many blue agents have this cell != UNKNOWN at t=200
  - distinct_red  : how many red agents have this cell != UNKNOWN at t=200
  - total_visits  : total cumulative *visits* (cell was inside an obs window)
                    summed over agents and all timesteps in the episode

We aggregate per setup: histogram of distinct_blue (0..n_blue), and the
per-cell mean / p50 / p90 of total_visits, broken out by which channel
bucket the cell falls into (blue-only, BOTH, red-only, nobody).

Output:
  experiments/meta-report/scan_multiplicity.png
  experiments/meta-report/scan_multiplicity.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from meta_report import (
    SETUPS, MAX_STEPS_EVAL, SNAPSHOT_STEPS, OUT_DIR,
    _load_blue, _load_red, _rollout_with_snapshots, Setup,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN

N_SEEDS = 10  # smaller than the heatmap; multiplicity is high-precision per seed
OBS_RADIUS = 1  # all 3 setups use radius 1


def _per_cell_multiplicity(setup: Setup, n_seeds: int) -> Dict:
    cfg = ExperimentConfig.from_yaml(setup.config)
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    red_actor, red_params = (None, None)
    if setup.red_ckpt is not None:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red

    distinct_blue_hist = np.zeros(n_blue + 1, dtype=np.int64)
    distinct_red_hist = np.zeros(n_red + 1, dtype=np.int64)
    visits_by_bucket = {"blue-only": [], "BOTH": [], "red-only": [], "nobody": []}

    for s in range(n_seeds):
        snaps, _curve, paths, team_ids = _rollout_with_snapshots(
            cfg, blue_actor, blue_params, red_actor, red_params,
            seed=s, max_steps=MAX_STEPS_EVAL, snapshot_steps=SNAPSHOT_STEPS,
        )
        snap = snaps[-1]
        terrain = snap["terrain"]
        local_maps = snap["local_maps"]  # [N, H, W]
        H, W = terrain.shape
        non_wall = terrain != CELL_WALL

        # distinct-agent counts per cell at t=200
        blue_idx = np.where(team_ids == 0)[0]
        red_idx = np.where(team_ids == 1)[0]
        blue_known = (local_maps[blue_idx] != MAP_UNKNOWN)  # [B, H, W]
        red_known = (local_maps[red_idx] != MAP_UNKNOWN) if len(red_idx) else np.zeros((0, H, W), bool)

        n_blue_seeing = blue_known.sum(axis=0)   # [H, W]
        n_red_seeing = red_known.sum(axis=0) if red_known.shape[0] > 0 else np.zeros((H, W), int)

        # visits: walk through paths (T+1, N, 2). Each step, every agent's 3x3 window
        # scans 9 cells (intersected with grid bounds and non-wall). Add to a counter.
        visits = np.zeros((H, W), dtype=np.int32)
        T = paths.shape[0]
        for t in range(T):
            for a in range(paths.shape[1]):
                r, c = int(paths[t, a, 0]), int(paths[t, a, 1])
                rs = slice(max(0, r - OBS_RADIUS), min(H, r + OBS_RADIUS + 1))
                cs = slice(max(0, c - OBS_RADIUS), min(W, c + OBS_RADIUS + 1))
                visits[rs, cs] += 1

        # Histograms over non-wall cells only
        bcount = n_blue_seeing[non_wall]
        rcount = n_red_seeing[non_wall]
        for k in range(n_blue + 1):
            distinct_blue_hist[k] += int((bcount == k).sum())
        for k in range(n_red + 1):
            distinct_red_hist[k] += int((rcount == k).sum())

        # Bucket each non-wall cell, record visits[]
        any_blue = bcount > 0
        any_red = rcount > 0
        bucket_masks = {
            "blue-only": any_blue & ~any_red,
            "BOTH":      any_blue & any_red,
            "red-only":  ~any_blue & any_red,
            "nobody":    ~any_blue & ~any_red,
        }
        v = visits[non_wall]
        for name, m in bucket_masks.items():
            visits_by_bucket[name].extend(v[m].tolist())

    # Normalize histograms to fractions of non-wall cells
    n_blue_cells = distinct_blue_hist.sum()
    n_red_cells = distinct_red_hist.sum()

    return {
        "n_blue": n_blue,
        "n_red": n_red,
        "distinct_blue_hist": distinct_blue_hist,
        "distinct_red_hist": distinct_red_hist,
        "distinct_blue_frac": distinct_blue_hist / max(1, n_blue_cells),
        "distinct_red_frac": distinct_red_hist / max(1, n_red_cells),
        "visits_by_bucket": {k: np.asarray(v, dtype=np.int32)
                             for k, v in visits_by_bucket.items()},
    }


BUCKET_COLORS = {
    "blue-only": "#6fa8dc",
    "BOTH": "#cd87cd",
    "red-only": "#ff9033",
    "nobody": "#222222",
}


def _render(panels: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.0))

    # Row 1: distinct-blue histogram (how many blues saw each cell)
    for col, setup in enumerate(SETUPS):
        ax = axes[0, col]
        d = panels[setup.key]
        ks = np.arange(d["n_blue"] + 1)
        ax.bar(ks, d["distinct_blue_frac"] * 100, color="#6fa8dc",
               edgecolor="white", linewidth=0.6)
        ax.set_xticks(ks)
        ax.set_xlabel("# distinct blue agents that saw this cell")
        ax.set_ylabel("% of non-wall cells")
        ax.set_ylim(0, 100)
        ax.set_title(f"{setup.short}\n{d['n_blue']} blue agents")
        for k in ks:
            v = d["distinct_blue_frac"][k] * 100
            if v > 0.5:
                ax.text(k, v + 1.0, f"{v:.1f}%", ha="center", fontsize=8)

    # Row 2: visits-per-cell distribution by bucket (boxplot per bucket)
    for col, setup in enumerate(SETUPS):
        ax = axes[1, col]
        d = panels[setup.key]
        bucket_order = ["blue-only", "BOTH", "red-only", "nobody"]
        data = [d["visits_by_bucket"][b] for b in bucket_order]
        # filter empty
        positions = []
        plot_data = []
        plot_colors = []
        plot_labels = []
        for i, (b, arr) in enumerate(zip(bucket_order, data)):
            if len(arr) > 0:
                positions.append(i)
                plot_data.append(arr)
                plot_colors.append(BUCKET_COLORS[b])
                plot_labels.append(b)
        bp = ax.boxplot(
            plot_data, positions=positions, widths=0.6, patch_artist=True,
            showfliers=False,
        )
        for patch, c in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels, fontsize=9, rotation=0)
        ax.set_ylabel("visits per cell\n(sum across agents × timesteps)")
        ax.set_title(f"{setup.short} — scan multiplicity by bucket")
        # annotate medians
        for pos, arr in zip(positions, plot_data):
            ax.text(pos, np.median(arr) * 1.05, f"med={int(np.median(arr))}",
                    ha="center", fontsize=8)

    fig.suptitle(
        "Hidden redundancy behind the channels chart\n"
        "(top: how many distinct blues saw each non-wall cell · "
        "bottom: total scan-events per cell, broken out by bucket; whiskers exclude outliers)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "scan_multiplicity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    panels = {}
    for setup in SETUPS:
        print(f"rolling out {setup.key} x {N_SEEDS} seeds…")
        panels[setup.key] = _per_cell_multiplicity(setup, N_SEEDS)

    _render(panels)

    summary = {}
    for setup in SETUPS:
        d = panels[setup.key]
        summary[setup.key] = {
            "distinct_blue_frac_pct":
                {f"k={k}": float(d["distinct_blue_frac"][k] * 100)
                 for k in range(d["n_blue"] + 1)},
            "distinct_red_frac_pct":
                {f"k={k}": float(d["distinct_red_frac"][k] * 100)
                 for k in range(d["n_red"] + 1)},
            "visits_summary_by_bucket": {
                bucket: {
                    "n_cells": int(len(arr)),
                    "mean": float(arr.mean()) if len(arr) else 0.0,
                    "p50":  float(np.percentile(arr, 50)) if len(arr) else 0.0,
                    "p90":  float(np.percentile(arr, 90)) if len(arr) else 0.0,
                    "max":  int(arr.max()) if len(arr) else 0,
                }
                for bucket, arr in d["visits_by_bucket"].items()
            },
        }
    out_json = OUT_DIR / "scan_multiplicity.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
