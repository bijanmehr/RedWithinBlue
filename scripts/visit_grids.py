"""Per-cell visit-count heatmaps split by team, with a dominance view.

For each setup B / C1 / C2:
  - Walk every (timestep, agent) and stamp the agent's 3x3 observation window
    onto a per-team visit grid.
  - Total over 10 seeds.

Output panels (3 rows × 3 columns):
  Row = setup (B, C1, C2)
  Col 1 = blue team total visits
  Col 2 = red team total visits   (blank for B)
  Col 3 = dominance: per-agent-normalised difference (blue − red)
          diverging colormap, blue → cell more often inside a blue window,
          red → more often inside a red window. Walls grey.

Output:
  experiments/meta-report/visit_grids.png
  experiments/meta-report/visit_grids.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from meta_report import (
    SETUPS, MAX_STEPS_EVAL, SNAPSHOT_STEPS, OUT_DIR,
    _load_blue, _load_red, _rollout_with_snapshots, Setup,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.types import CELL_WALL

N_SEEDS = 10
OBS_RADIUS = 1

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#ffffff", "#cce0f0", "#5a9bd4", "#1f4e79"], N=256,
)
RED_CMAP = LinearSegmentedColormap.from_list(
    "reds", ["#ffffff", "#fcd7d4", "#e85a5a", "#7a0000"], N=256,
)
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "diff", ["#7a0000", "#e85a5a", "#ffffff", "#5a9bd4", "#1f4e79"], N=256,
)


def _per_team_visits(setup: Setup, n_seeds: int):
    cfg = ExperimentConfig.from_yaml(setup.config)
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    red_actor, red_params = (None, None)
    if setup.red_ckpt is not None:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red

    H = cfg.env.grid_height
    W = cfg.env.grid_width

    blue_visits = np.zeros((H, W), dtype=np.int64)
    red_visits = np.zeros((H, W), dtype=np.int64)
    terrain_ref = None

    for s in range(n_seeds):
        snaps, _curve, paths, team_ids = _rollout_with_snapshots(
            cfg, blue_actor, blue_params, red_actor, red_params,
            seed=s, max_steps=MAX_STEPS_EVAL, snapshot_steps=SNAPSHOT_STEPS,
        )
        snap = snaps[-1]
        terrain_ref = snap["terrain"]
        T = paths.shape[0]
        for t in range(T):
            for a in range(paths.shape[1]):
                r, c = int(paths[t, a, 0]), int(paths[t, a, 1])
                rs = slice(max(0, r - OBS_RADIUS), min(H, r + OBS_RADIUS + 1))
                cs = slice(max(0, c - OBS_RADIUS), min(W, c + OBS_RADIUS + 1))
                if team_ids[a] == 0:
                    blue_visits[rs, cs] += 1
                else:
                    red_visits[rs, cs] += 1

    return {
        "blue_visits": blue_visits,
        "red_visits": red_visits,
        "terrain": terrain_ref,
        "n_blue": n_blue,
        "n_red": n_red,
    }


def _annotate_walls(ax, terrain):
    H, W = terrain.shape
    wall = terrain == CELL_WALL
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[wall] = (0.17, 0.17, 0.17, 1.0)
    ax.imshow(rgba, origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))


def _render(panels: Dict[str, dict]) -> None:
    n_setups = len(SETUPS)
    fig, axes = plt.subplots(n_setups, 3, figsize=(13.5, 4.2 * n_setups))
    if n_setups == 1:
        axes = axes[None, :]

    # Per-agent densities so cross-setup comparison is meaningful: B has 5 blue
    # agents, C2 has 3 — total counts are not comparable, per-agent counts are.
    for setup in SETUPS:
        d = panels[setup.key]
        d["blue_per_agent"] = d["blue_visits"].astype(np.float32) / max(1, d["n_blue"])
        d["red_per_agent"] = (
            d["red_visits"].astype(np.float32) / max(1, d["n_red"])
            if d["n_red"] > 0 else np.zeros_like(d["blue_visits"], dtype=np.float32)
        )

    baseline_blue = panels["B"]["blue_per_agent"]

    # Common color scales across rows so the visual change is direct.
    blue_max = max(panels[s.key]["blue_per_agent"].max() for s in SETUPS) or 1.0
    red_max = max(
        (panels[s.key]["red_per_agent"].max() for s in SETUPS if panels[s.key]["n_red"] > 0),
        default=1.0,
    )

    delta_max = 0.0
    for s in SETUPS:
        if s.key == "B":
            continue
        nw_s = panels[s.key]["terrain"] != CELL_WALL
        delta = panels[s.key]["blue_per_agent"] - baseline_blue
        delta_max = max(delta_max, float(np.abs(delta[nw_s]).max()))
    delta_max = max(delta_max, 1.0)

    for row, setup in enumerate(SETUPS):
        d = panels[setup.key]
        H, W = d["terrain"].shape
        wall = d["terrain"] == CELL_WALL
        nw = ~wall

        # ---------- col 0: blue visits per blue agent (common scale) ----------
        ax = axes[row, 0]
        bp = d["blue_per_agent"]
        bp_disp = np.where(wall, np.nan, bp)
        im0 = ax.imshow(bp_disp, cmap=BLUE_CMAP, vmin=0, vmax=blue_max,
                        origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        _annotate_walls(ax, d["terrain"])
        ax.set_title(
            f"{setup.short}\nBLUE visits / blue agent  (peak={bp.max():.0f})",
            fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            cbar0 = fig.colorbar(im0, ax=axes[:, 0].tolist(),
                                 fraction=0.025, pad=0.02, shrink=0.85)
            cbar0.set_label(
                "blue visits per blue agent  (10 seeds, common scale across rows)",
                fontsize=9,
            )

        # ---------- col 1: red visits per red agent (common scale) ----------
        ax = axes[row, 1]
        if d["n_red"] == 0:
            ax.imshow(np.full((H, W), np.nan), origin="upper",
                      extent=(-0.5, W - 0.5, H - 0.5, -0.5))
            _annotate_walls(ax, d["terrain"])
            ax.text(W / 2 - 0.5, H / 2 - 0.5, "no red agents\n(baseline)",
                    ha="center", va="center", fontsize=11, color="#888")
            ax.set_title(f"{setup.short}\nRED visits / red agent  (n=0)", fontsize=10)
        else:
            rp = d["red_per_agent"]
            rp_disp = np.where(wall, np.nan, rp)
            im1 = ax.imshow(rp_disp, cmap=RED_CMAP, vmin=0, vmax=red_max,
                            origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
            _annotate_walls(ax, d["terrain"])
            ax.set_title(
                f"{setup.short}\nRED visits / red agent  (peak={rp.max():.0f})",
                fontsize=10,
            )
        ax.set_xticks([]); ax.set_yticks([])

        # ---------- col 2: blue displacement vs B baseline + red contours ----------
        ax = axes[row, 2]
        if setup.key == "B":
            ax.imshow(np.zeros((H, W)), cmap=DIFF_CMAP,
                      vmin=-delta_max, vmax=delta_max,
                      origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
            _annotate_walls(ax, d["terrain"])
            ax.text(W / 2 - 0.5, H / 2 - 0.5,
                    "this is the B baseline\n(Δ = 0 vs itself)",
                    ha="center", va="center", fontsize=10, color="#444")
            ax.set_title(
                f"{setup.short}\nblue displacement vs B  —  reference",
                fontsize=10,
            )
        else:
            delta = d["blue_per_agent"] - baseline_blue
            delta_disp = np.where(wall, np.nan, delta)
            norm = TwoSlopeNorm(vmin=-delta_max, vcenter=0.0, vmax=delta_max)
            im2 = ax.imshow(delta_disp, cmap=DIFF_CMAP, norm=norm,
                            origin="upper",
                            extent=(-0.5, W - 0.5, H - 0.5, -0.5))
            # Red density contours sit ON TOP of the abandoned regions when
            # the adversary effect is real: red moves in, blue gives ground.
            rp = d["red_per_agent"]
            if rp.max() > 0:
                rp_norm = rp / rp.max()
                ax.contour(
                    np.arange(W), np.arange(H), rp_norm,
                    levels=[0.10, 0.30, 0.60],
                    colors="#5a0000", linewidths=[0.7, 1.2, 1.9], alpha=0.95,
                )
            _annotate_walls(ax, d["terrain"])
            d_nw = delta[nw]
            base_mean = float(baseline_blue[nw].mean())
            abandoned_pct = float((d_nw < -0.5 * base_mean).sum()) / max(1, int(nw.sum())) * 100
            crowded_pct = float((d_nw > 0.5 * base_mean).sum()) / max(1, int(nw.sum())) * 100
            # Mean Δ inside red hotspots (red density >= 1/3 of its peak): if
            # blue abandons specifically where red is, this is strongly negative
            # versus the all-cell mean (≈ 0).
            red_hot = (rp >= rp.max() / 3.0) & nw
            mean_delta_in_red = (
                float(delta[red_hot].mean()) if red_hot.any() else 0.0
            )
            ax.set_title(
                f"{setup.short}\nΔ blue per agent vs B  (red density contours)\n"
                f"abandoned {abandoned_pct:.0f}%   crowded {crowded_pct:.0f}%   "
                f"mean Δ in red hotspots = {mean_delta_in_red:+.0f}",
                fontsize=9,
            )
        ax.set_xticks([]); ax.set_yticks([])

    # Δ colorbar — placed once on the rightmost column.
    for row, setup in enumerate(SETUPS):
        if setup.key != "B" and len(axes[row, 2].images) > 0:
            cbar2 = fig.colorbar(
                axes[row, 2].images[0], ax=axes[:, 2].tolist(),
                fraction=0.025, pad=0.02, shrink=0.85,
            )
            cbar2.set_label(
                "Δ blue per agent vs B baseline\n"
                "blue tone = blue OVER-visits this cell vs B   "
                "red tone = blue ABANDONS this cell vs B",
                fontsize=9,
            )
            break

    # Per-agent red colorbar
    if any(panels[s.key]["n_red"] > 0 for s in SETUPS):
        for row, setup in enumerate(SETUPS):
            if panels[setup.key]["n_red"] > 0:
                cbar1 = fig.colorbar(
                    axes[row, 1].images[0], ax=axes[:, 1].tolist(),
                    fraction=0.025, pad=0.02, shrink=0.85,
                )
                cbar1.set_label(
                    "red visits per red agent  (10 seeds, common scale)",
                    fontsize=9,
                )
                break

    fig.suptitle(
        "Adversary effect on blue's footprint  —  per-cell visits, B / C1 / C2\n"
        "cols 0,1: per-agent visit density on a common scale across rows  "
        "·  col 2: blue displacement from the no-red baseline B, with red density contours overlaid",
        fontsize=11, y=0.995,
    )
    out = OUT_DIR / "visit_grids.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    panels = {}
    for setup in SETUPS:
        print(f"rolling out {setup.key} x {N_SEEDS} seeds…")
        panels[setup.key] = _per_team_visits(setup, N_SEEDS)

    _render(panels)

    summary = {}
    baseline_blue = panels["B"]["blue_visits"].astype(np.float32) / max(1, panels["B"]["n_blue"])
    for setup in SETUPS:
        d = panels[setup.key]
        nw = d["terrain"] != CELL_WALL
        bp = d["blue_visits"].astype(np.float32) / max(1, d["n_blue"])
        rp = (
            d["red_visits"].astype(np.float32) / max(1, d["n_red"])
            if d["n_red"] > 0 else np.zeros_like(bp)
        )
        delta = bp - baseline_blue
        d_nw = delta[nw]
        base_mean = float(baseline_blue[nw].mean())
        abandoned_pct = float((d_nw < -0.5 * base_mean).sum()) / max(1, int(nw.sum())) * 100
        crowded_pct = float((d_nw > 0.5 * base_mean).sum()) / max(1, int(nw.sum())) * 100
        red_hot = (rp >= rp.max() / 3.0) & nw if rp.max() > 0 else np.zeros_like(nw)
        mean_delta_in_red = (
            float(delta[red_hot].mean()) if red_hot.any() else 0.0
        )
        summary[setup.key] = {
            "n_blue": d["n_blue"],
            "n_red": d["n_red"],
            "blue_visits_total": int(d["blue_visits"][nw].sum()),
            "red_visits_total": int(d["red_visits"][nw].sum()),
            "blue_visits_max_cell": int(d["blue_visits"].max()),
            "red_visits_max_cell": int(d["red_visits"].max()),
            "blue_per_agent_mean_per_cell": float(bp[nw].mean()),
            "red_per_agent_mean_per_cell": float(rp[nw].mean()),
            "blue_per_agent_max_cell": float(bp.max()),
            "red_per_agent_max_cell": float(rp.max()),
            "delta_blue_vs_B_abandoned_pct": abandoned_pct,
            "delta_blue_vs_B_crowded_pct": crowded_pct,
            "delta_blue_vs_B_mean_in_red_hotspots": mean_delta_in_red,
            "delta_blue_vs_B_mean_abs": float(np.abs(d_nw).mean()),
        }
    out_json = OUT_DIR / "visit_grids.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
