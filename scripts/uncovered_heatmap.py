"""Direct replacement for proximity_bands.png + proximity_ecdf.png.

Both of those figures answered the same question — *"where do uncovered cells
sit relative to red?"* — through abstract distributions. This script answers
it on the actual map.

For each setup B / C1 / C2:
  1. Roll out 20 seeds to t=200.
  2. For each grid cell, record P(cell is uncovered at t=200) across seeds.
  3. Render the 16x16 grid as a heatmap (white = always covered,
     dark red = often uncovered).
  4. Overlay the time-averaged density of red and blue agent positions,
     so the reader can see the spatial relationship between agent activity
     and coverage gaps.

Output:
  experiments/meta-report/uncovered_heatmap.png   — 1×3 grid, one map per setup
  experiments/meta-report/uncovered_heatmap.json  — per-cell uncovered probs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
OUT_PNG = OUT_DIR / "uncovered_heatmap.png"
OUT_JSON = OUT_DIR / "uncovered_heatmap.json"

UNCOVERED_CMAP = LinearSegmentedColormap.from_list(
    "uncovered", ["#ffffff", "#fdd0d0", "#e85a5a", "#7a0000"], N=256
)


def _per_cell_uncovered_prob(setup: Setup, n_seeds: int) -> Dict[str, np.ndarray]:
    """Per cell, fraction of seeds in which the cell ended uncovered at t=200.

    Also collects time-averaged red and blue position densities so the heatmap
    can show *where each team operated* on top of the coverage failure map.
    """
    cfg = ExperimentConfig.from_yaml(setup.config)
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    red_actor, red_params = (None, None)
    if setup.red_ckpt is not None:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

    H = cfg.env.grid_height
    W = cfg.env.grid_width

    uncovered_count = np.zeros((H, W), dtype=np.float32)
    nonwall_count = np.zeros((H, W), dtype=np.float32)
    blue_density = np.zeros((H, W), dtype=np.float32)
    red_density = np.zeros((H, W), dtype=np.float32)

    for s in range(n_seeds):
        snaps, _curve, paths, team_ids = _rollout_with_snapshots(
            cfg, blue_actor, blue_params, red_actor, red_params,
            seed=s, max_steps=MAX_STEPS_EVAL, snapshot_steps=SNAPSHOT_STEPS,
        )
        snap = snaps[-1]
        terrain = snap["terrain"]
        non_wall = terrain != CELL_WALL
        # Uncovered = non-wall AND blue never observed it
        uncovered = non_wall & (~snap["blue_ever"])
        uncovered_count += uncovered.astype(np.float32)
        nonwall_count += non_wall.astype(np.float32)

        # Time-averaged agent density: paths is [T+1, N, 2] (row, col)
        for t in range(paths.shape[0]):
            for a in range(paths.shape[1]):
                r, c = int(paths[t, a, 0]), int(paths[t, a, 1])
                if 0 <= r < H and 0 <= c < W:
                    if team_ids[a] == 0:
                        blue_density[r, c] += 1.0
                    else:
                        red_density[r, c] += 1.0

    # Convert counts to probabilities; cells that were always walls become 0.
    eps = np.maximum(nonwall_count, 1e-9)
    p_uncovered = np.where(nonwall_count > 0, uncovered_count / eps, 0.0)

    # Normalize densities so they read as "share of agent-time spent here".
    if blue_density.sum() > 0:
        blue_density = blue_density / blue_density.sum()
    if red_density.sum() > 0:
        red_density = red_density / red_density.sum()

    return {
        "p_uncovered": p_uncovered,
        "blue_density": blue_density,
        "red_density": red_density,
        "terrain": terrain.astype(np.int32),
    }


def _render(panels: Dict[str, Dict[str, np.ndarray]]) -> None:
    n = len(SETUPS)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8))
    if n == 1:
        axes = [axes]

    for ax, setup in zip(axes, SETUPS):
        d = panels[setup.key]
        p = d["p_uncovered"]
        terrain = d["terrain"]
        H, W = p.shape

        # Walls in solid grey on top of the heatmap.
        wall = terrain == CELL_WALL
        display = np.where(wall, np.nan, p)

        im = ax.imshow(
            display, cmap=UNCOVERED_CMAP, vmin=0.0, vmax=1.0,
            origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5),
        )
        # Wall overlay
        wall_rgba = np.zeros((H, W, 4), dtype=np.float32)
        wall_rgba[wall] = (0.17, 0.17, 0.17, 1.0)
        ax.imshow(wall_rgba, origin="upper",
                  extent=(-0.5, W - 0.5, H - 0.5, -0.5))

        # Time-averaged red density as red contour rings.
        rd = d["red_density"]
        if rd.sum() > 0:
            rd_norm = rd / rd.max()
            ax.contour(
                np.arange(W), np.arange(H), rd_norm,
                levels=[0.05, 0.20, 0.50],
                colors="#d62728", linewidths=[0.6, 1.0, 1.6], alpha=0.9,
            )

        # Time-averaged blue density as blue contour rings (lighter).
        bd = d["blue_density"]
        if bd.sum() > 0:
            bd_norm = bd / bd.max()
            ax.contour(
                np.arange(W), np.arange(H), bd_norm,
                levels=[0.05, 0.20, 0.50],
                colors="#1f77b4", linewidths=[0.5, 0.8, 1.2], alpha=0.6,
            )

        # Per-cell uncovered count as the title statistic.
        nw = terrain != CELL_WALL
        mean_uncov = float(p[nw].mean()) * 100.0
        ax.set_title(
            f"{setup.short}\nmean uncovered: {mean_uncov:.1f}%",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("P(cell uncovered at t=200) across 20 seeds", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Where do coverage gaps live on the map?\n"
        "(red contours = where red spent most time; blue contours = where blue spent most time)",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels: Dict[str, Dict[str, np.ndarray]] = {}
    for setup in SETUPS:
        print(f"rolling out {setup.key} x {N_SEEDS} seeds…")
        panels[setup.key] = _per_cell_uncovered_prob(setup, N_SEEDS)

    _render(panels)
    print(f"wrote {OUT_PNG}")

    summary = {
        k: {
            "mean_uncovered_pct": float(panels[k]["p_uncovered"][panels[k]["terrain"] != CELL_WALL].mean() * 100.0),
            "max_uncovered_prob": float(panels[k]["p_uncovered"].max()),
            "p_uncovered_grid": panels[k]["p_uncovered"].round(4).tolist(),
        }
        for k in panels
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
