"""Thirteen candidate visualizations of the adversary effect — pick the
one(s) that tell the story best.

Run me; open experiments/meta-report/adversary_candidates.html and pick.

Candidates (each PNG is a fundamentally different lens):
  1. mission_tape           — narrative film strip (1 seed × 3 setups × 5 frames)
  2. delay_map              — counterfactual: per-cell delay in first-seen-by-blue
  3. denied_cells           — binary failure map: cells covered in B but NOT in C*
  4. fate_transitions       — stacked bars: where did B's blue-only cells go?
  5. frontier_race          — coverage % vs time with shaded "stolen" gap
  6. agent_polygons         — per-cell argmax-agent territories
  7. spacetime_tubes        — 3D agent trajectories in (x, y, t)
  8. first_seen_landscape   — 3D bar towers: height = first-seen step
  9. visit_cityscape        — 3D bars: height = visits, colour = argmax agent
 10. coverage_slabs         — 3D voxel cloud: knowledge slabs at each snapshot
 11. delay_towers           — 3D bar towers: height = step delay vs B baseline
 12. bayesian_confidence    — signed evidence map: blue scans − red scans
 13. confidence_pillars     — 3D bars of #12: pillar height = |net evidence|
 14. confidence_evolution   — Bayesian map at t=40/80/120/160/200 (the gradient)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

from meta_report import (
    SETUPS, MAX_STEPS_EVAL, OUT_DIR,
    _load_blue, _load_red, _rollout_with_snapshots, Setup,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
from red_within_blue.visualizer import _merge_team_belief


N_SEEDS = 10
TAPE_SEED = 0
TAPE_STEPS = (40, 80, 120, 160, 200)

CAND_PNG = {
    1: ("candidate_1_mission_tape.png",
        "Mission tape — 1 canonical seed, 3 setups × 5 frames",
        "Watch left→right: blue's discovered area (light blue) and red's "
        "discovered area (light red) grow each frame, with named agent markers. "
        "You can SEE red carve out a corner and blue's frontier shrink."),
    2: ("candidate_2_delay_map.png",
        "Counterfactual delay map — when did blue first see each cell?",
        "Per cell, the median (over 10 seeds) frame at which blue first observed "
        "it (≤40 / ≤80 / ≤120 / ≤160 / ≤200 / never). Right column: ΔC*−B = how "
        "many extra frames red cost blue. Black = never reached under that setup."),
    3: ("candidate_3_denied_cells.png",
        "Cells lost to red — binary failure mask",
        "Red squares = cells reliably covered in B (≥80% of B's seeds saw them) "
        "but failed in C* (<20% of C*'s seeds saw them). Red density contours "
        "overlaid. The visual answer to 'what cells did red cost blue?'"),
    4: ("candidate_4_fate_transitions.png",
        "Cell-fate transitions — where did B's blue-only cells end up?",
        "Stacked bars: of cells blue-only in B at t=200, what fraction stayed "
        "blue-only / became BOTH (red also saw) / became RED-only / became "
        "NOBODY-saw under C1 and C2? The bar is the labor lost."),
    5: ("candidate_5_frontier_race.png",
        "Frontier race — coverage % vs time with stolen-area shading",
        "Three coverage curves over the 200-step episode (mean across 10 seeds, "
        "shaded ±1σ). The orange fill between B and C2 is the area-under-curve "
        "stolen by the adversary; integrated this is 'cell-frames denied'."),
    6: ("candidate_6_agent_polygons.png",
        "Per-agent territory polygons — who scanned this cell most often?",
        "Each cell coloured by which agent visited it most (10 seeds pooled). "
        "Blue palette = blue agents; red palette = red agents. Watch B's 5 "
        "tidy blue territories collapse to 3 in C2, with a large red block."),
    7: ("candidate_7_spacetime_tubes.png",
        "3D space-time tubes — agent trajectories in (x, y, t)",
        "Each agent is a coloured tube snaking up the time axis (t=0 at the "
        "floor, t=200 at the top). Blue tubes braid across the map; red tubes "
        "in C* climb almost straight up — they're camping. Square markers = "
        "final position."),
    8: ("candidate_8_first_seen_landscape.png",
        "3D first-seen landscape — height = step at which blue first saw the cell",
        "z = median first-seen step across 10 seeds. Cells blue never reached "
        "are pushed to a black 'denial spike' at the ceiling. The taller and "
        "blacker the landscape gets going B → C1 → C2, the more red is hurting "
        "blue's exploration tempo."),
    9: ("candidate_9_visit_cityscape.png",
        "3D visit-count cityscape — bar height = total visits, colour = argmax agent",
        "Each non-wall cell is a coloured bar; height is total visits across 10 "
        "seeds; colour is the agent that visited it most. You see both the "
        "territorial partition (top-down) and the labour intensity per cell "
        "(vertical) in one view."),
    10: ("candidate_10_coverage_slabs.png",
         "3D coverage slabs — knowledge growing through (x, y, t)",
         "Voxel cloud: at each snapshot timestep t, blue dots mark cells blue "
         "currently knows; red dots mark cells red currently knows (offset "
         "slightly so they don't collide). You watch the blue cloud thicken "
         "while the red cloud climbs in a single corner column."),
    11: ("candidate_11_delay_towers.png",
         "3D delay towers — C2 vs B baseline, side by side",
         "Two surfaces overlaid for each cell: B's first-seen-step (transparent "
         "blue floor) and C2's first-seen-step (solid surface). The vertical "
         "gap between them is the delay; black columns are cells C2 never "
         "reached at all. The 'haunted house' silhouette is red's footprint."),
    12: ("candidate_12_bayesian_confidence.png",
         "Bayesian confidence map — every blue scan adds evidence, every red scan removes it",
         "Signed per-cell evidence integrated over 200 steps × 10 seeds. Blue "
         "= cells where the team accumulated trustworthy intel (blue scans "
         "outnumber red scans). Red = cells red corrupted (more red scans than "
         "blue). White = no info or balanced. The red islands in C1/C2 are "
         "literal regions of poisoned belief."),
    13: ("candidate_13_confidence_pillars.png",
         "3D confidence pillars — same Bayesian map, signed bar towers",
         "Bar3d version of #12: each cell is a pillar whose height is the "
         "magnitude of net evidence and whose colour is the sign — blue "
         "pillars rise where blue won the cell, red pillars rise where red "
         "corrupted it. The skyline IS the team's posterior at t=200."),
    14: ("candidate_14_confidence_evolution.png",
         "Confidence evolution — Bayesian map over time (t=40/80/120/160/200)",
         "3 setups × 5 time slices showing the running net evidence per cell "
         "as the episode unfolds. Watch red islands GROW frame-by-frame in "
         "C1 and C2 while B stays uniformly blue. Each adversarial scan "
         "leaves a visible mark — the gradient is the effect."),
}
HTML = "adversary_candidates.html"


# -------------------- data collection --------------------

def _red_known_from_snap(snap: dict) -> np.ndarray:
    if (snap["team_ids"] == 1).sum() == 0:
        return np.zeros_like(snap["blue_ever"], dtype=bool)
    merged = _merge_team_belief(
        snap["local_maps"], snap["team_ids"], target_team=1,
    )
    non_wall = snap["terrain"] != CELL_WALL
    return (merged != MAP_UNKNOWN) & non_wall


def collect(setup: Setup, seeds: List[int]) -> List[dict]:
    cfg = ExperimentConfig.from_yaml(setup.config)
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    red_actor, red_params = (None, None)
    if setup.red_ckpt is not None:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)
    runs = []
    for s in seeds:
        snaps, curve, paths, team_ids = _rollout_with_snapshots(
            cfg, blue_actor, blue_params, red_actor, red_params,
            seed=s, max_steps=MAX_STEPS_EVAL, snapshot_steps=TAPE_STEPS,
        )
        for snap in snaps:
            snap["red_ever"] = _red_known_from_snap(snap)
        runs.append({
            "seed": s,
            "snaps": snaps,
            "curve": np.array(curve, dtype=np.float32),
            "paths": paths,
            "team_ids": team_ids,
            "n_blue": int((team_ids == 0).sum()),
            "n_red": int((team_ids == 1).sum()),
            "terrain": snaps[-1]["terrain"],
        })
    return runs


# -------------------- shared helpers --------------------

def _draw_walls(ax, terrain):
    H, W = terrain.shape
    wall = terrain == CELL_WALL
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[wall] = (0.17, 0.17, 0.17, 1.0)
    ax.imshow(rgba, origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))


def _imshow_grid(ax, img, **kw):
    H, W = img.shape[:2]
    return ax.imshow(img, origin="upper",
                     extent=(-0.5, W - 0.5, H - 0.5, -0.5), **kw)


def _clean_3d_axes(ax) -> None:
    """Remove the dark pane walls and harsh grid from a matplotlib 3d axes
    so the 3D content floats on a white background."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor((1, 1, 1, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis.line.set_color((0.6, 0.6, 0.6, 0.7))
    ax.set_facecolor("white")


# -------------------- candidate 1: mission tape --------------------

def candidate_1_mission_tape(runs_by_setup: Dict[str, List[dict]], out: Path) -> None:
    fig, axes = plt.subplots(len(SETUPS), len(TAPE_STEPS),
                             figsize=(3.0 * len(TAPE_STEPS), 3.0 * len(SETUPS)))
    if len(SETUPS) == 1:
        axes = axes[None, :]

    for row, setup in enumerate(SETUPS):
        run = runs_by_setup[setup.key][TAPE_SEED]
        terrain = run["terrain"]
        H, W = terrain.shape
        non_wall = terrain != CELL_WALL
        team_ids = run["team_ids"]
        paths = run["paths"]                       # [T+1, N, 2]

        for col, snap in enumerate(run["snaps"]):
            ax = axes[row, col]
            ax.set_facecolor("#f5f5f5")

            # base terrain
            base = np.zeros((H, W, 4), dtype=np.float32)
            base[non_wall] = (0.97, 0.97, 0.97, 1.0)
            _imshow_grid(ax, base)

            # blue_ever — light blue fill where blue has seen
            blue_known = snap["blue_ever"]
            blue_layer = np.zeros((H, W, 4), dtype=np.float32)
            blue_layer[blue_known] = (0.23, 0.55, 0.85, 0.45)
            _imshow_grid(ax, blue_layer)

            # red_ever — light red fill where red has seen
            red_known = snap["red_ever"]
            red_layer = np.zeros((H, W, 4), dtype=np.float32)
            red_layer[red_known] = (0.85, 0.20, 0.20, 0.45)
            _imshow_grid(ax, red_layer)

            # both = darker overlap
            both = blue_known & red_known
            both_layer = np.zeros((H, W, 4), dtype=np.float32)
            both_layer[both] = (0.45, 0.10, 0.55, 0.50)
            _imshow_grid(ax, both_layer)

            _draw_walls(ax, terrain)

            # agent markers at this snapshot's positions
            positions = snap["positions"]
            for a in range(positions.shape[0]):
                r, c = positions[a]
                if team_ids[a] == 0:
                    ax.scatter(c, r, s=110, marker="o",
                               edgecolors="white", linewidths=1.6,
                               facecolors="#1f4e79", zorder=5)
                    ax.annotate(f"B{a}", (c, r), color="white", fontsize=7,
                                ha="center", va="center", zorder=6, fontweight="bold")
                else:
                    ax.scatter(c, r, s=110, marker="s",
                               edgecolors="white", linewidths=1.6,
                               facecolors="#7a0000", zorder=5)
                    ax.annotate(f"R{a - run['n_blue']}", (c, r), color="white",
                                fontsize=7, ha="center", va="center", zorder=6,
                                fontweight="bold")

            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(
                f"t={snap['step']}   cov={snap['coverage_pct']:.0f}%",
                fontsize=10,
            )
            if col == 0:
                ax.set_ylabel(setup.short, fontsize=10, fontweight="bold")

    fig.suptitle(
        "Mission tape — seed 0, 3 setups × 5 frames\n"
        "blue fill = blue has discovered   ·   red fill = red has discovered   "
        "·   purple = both saw the cell   ·   markers = current agent positions",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 2: delay map --------------------

def _first_seen_bucket(run: dict) -> np.ndarray:
    """Per cell, lowest TAPE_STEPS index at which blue had seen it.
    -1 if never. Shape [H, W]; values in {0,1,2,3,4} or -1."""
    snaps = run["snaps"]
    H, W = run["terrain"].shape
    bucket = np.full((H, W), -1, dtype=np.int8)
    for i, snap in enumerate(snaps):
        # cells newly known at this snap that were not known earlier
        new = snap["blue_ever"] & (bucket < 0)
        bucket[new] = i
    return bucket


def candidate_2_delay_map(runs_by_setup, out: Path) -> None:
    bucket_labels = ["≤40", "≤80", "≤120", "≤160", "≤200", "never"]
    n_buckets = len(bucket_labels)
    cmap_seq = LinearSegmentedColormap.from_list(
        "delay_seq",
        ["#caefca", "#82c785", "#4ea050", "#2d6a30", "#0e3a12", "#000000"],
        N=n_buckets,
    )
    bounds = np.arange(n_buckets + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap_seq.N)

    # Median bucket per cell across seeds (cells with -1 set to n_buckets-1)
    def median_buckets(setup_key):
        per_seed = []
        for run in runs_by_setup[setup_key]:
            b = _first_seen_bucket(run).astype(np.int16)
            b[b < 0] = n_buckets - 1  # "never"
            per_seed.append(b)
        per_seed = np.stack(per_seed, axis=0)  # [S, H, W]
        return np.median(per_seed, axis=0).astype(np.int16)

    medians = {s.key: median_buckets(s.key) for s in SETUPS}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    terrain = runs_by_setup["B"][0]["terrain"]
    H, W = terrain.shape

    for col, setup in enumerate(SETUPS):
        ax = axes[0, col]
        m = medians[setup.key].astype(np.float32)
        m = np.where(terrain == CELL_WALL, np.nan, m)
        im = _imshow_grid(ax, m, cmap=cmap_seq, norm=norm)
        _draw_walls(ax, terrain)
        ax.set_title(f"{setup.short} — median first-seen bucket", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02,
                        ticks=np.arange(n_buckets), shrink=0.85)
    cbar.ax.set_yticklabels(bucket_labels, fontsize=8)
    cbar.set_label("median frame at which blue first observed the cell")

    # Bottom row: ΔC1−B, ΔC2−B, then a coverage-time barh
    diff_cmap = LinearSegmentedColormap.from_list(
        "delay_diff",
        ["#1f4e79", "#5a9bd4", "#ffffff", "#e85a5a", "#7a0000", "#000000"],
        N=11,
    )
    for col, setup in enumerate(SETUPS):
        ax = axes[1, col]
        if setup.key == "B":
            ax.axis("off")
            ax.text(0.5, 0.5, "B is the reference\n(Δ = 0)", ha="center",
                    va="center", color="#666", fontsize=11,
                    transform=ax.transAxes)
            continue
        diff = medians[setup.key] - medians["B"]  # in {-5..+5}
        # cells "denied" under setup but reachable under B → set to +6 (extra-bad)
        denied = (medians[setup.key] == n_buckets - 1) & (medians["B"] < n_buckets - 1)
        d_disp = diff.astype(np.float32)
        d_disp[denied] = 6.0
        d_disp = np.where(terrain == CELL_WALL, np.nan, d_disp)
        im = _imshow_grid(ax, d_disp, cmap=diff_cmap, vmin=-5, vmax=6)
        _draw_walls(ax, terrain)
        # red density contours
        red_dens = np.zeros((H, W), dtype=np.float32)
        for run in runs_by_setup[setup.key]:
            red_dens += run["snaps"][-1]["red_ever"].astype(np.float32)
        if red_dens.max() > 0:
            rd_norm = red_dens / red_dens.max()
            ax.contour(np.arange(W), np.arange(H), rd_norm,
                       levels=[0.20, 0.50, 0.80],
                       colors="#3a0000", linewidths=[0.7, 1.2, 1.8], alpha=0.95)
        n_denied = int(denied.sum())
        ax.set_title(
            f"Δ first-seen ({setup.key} − B)   denied cells = {n_denied}",
            fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])

    cbar2 = fig.colorbar(im, ax=axes[1, 1:].tolist(), fraction=0.025, pad=0.02, shrink=0.85)
    cbar2.set_label("Δ buckets (negative = earlier under C*; +6 = denied entirely)")

    fig.suptitle(
        "Counterfactual delay — extra frames red cost blue per cell\n"
        "(top: median first-seen bucket per setup · bottom: ΔC*−B with red density contours)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 3: denied cells --------------------

def candidate_3_denied_cells(runs_by_setup, out: Path) -> None:
    """Binary mask: cells covered in B but failed in C*."""
    terrain = runs_by_setup["B"][0]["terrain"]
    H, W = terrain.shape
    non_wall = terrain != CELL_WALL

    def p_seen(setup_key):
        acc = np.zeros((H, W), dtype=np.float32)
        for run in runs_by_setup[setup_key]:
            acc += run["snaps"][-1]["blue_ever"].astype(np.float32)
        return acc / max(1, len(runs_by_setup[setup_key]))

    p_B = p_seen("B")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
    for col, setup in enumerate(s for s in SETUPS if s.key != "B"):
        ax = axes[col]
        p_C = p_seen(setup.key)
        denied = (p_B >= 0.8) & (p_C <= 0.2) & non_wall
        rescued = (p_B <= 0.5) & (p_C >= 0.8) & non_wall  # rare case
        # base layer light grey
        rgba = np.full((H, W, 4), (0.96, 0.96, 0.96, 1.0), dtype=np.float32)
        rgba[denied] = (0.85, 0.10, 0.10, 1.0)
        rgba[rescued] = (0.10, 0.55, 0.20, 1.0)
        _imshow_grid(ax, rgba)
        _draw_walls(ax, terrain)
        # red density contours
        red_dens = np.zeros((H, W), dtype=np.float32)
        for run in runs_by_setup[setup.key]:
            red_dens += run["snaps"][-1]["red_ever"].astype(np.float32)
        if red_dens.max() > 0:
            rd_norm = red_dens / red_dens.max()
            ax.contour(np.arange(W), np.arange(H), rd_norm,
                       levels=[0.20, 0.50, 0.80],
                       colors="#3a0000", linewidths=[0.7, 1.2, 1.8], alpha=0.95)
        n_denied = int(denied.sum())
        n_rescued = int(rescued.sum())
        ax.set_title(
            f"{setup.short}\n{n_denied} cells denied   {n_rescued} cells gained\n"
            f"(red squares = covered in B but missed in {setup.key} on ≥80% of seeds)",
            fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Denied cells — binary failure map\n"
        "(red square = covered in B ≥80% of seeds AND missed in C* ≥80% of seeds   ·   "
        "green square = the reverse — rare 'red helped blue' cells)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 4: fate transitions --------------------

def candidate_4_fate_transitions(runs_by_setup, out: Path) -> None:
    """Where do B's blue-only cells end up under C1 / C2?"""
    terrain = runs_by_setup["B"][0]["terrain"]
    H, W = terrain.shape
    non_wall = terrain != CELL_WALL

    def cell_status(snap):
        b = snap["blue_ever"]
        r = snap["red_ever"]
        return b, r

    # In B (no red), every cell is blue-only or nobody-saw.
    # We average across seeds: a cell is "blue-known in B" if mean blue_ever >= 0.5
    p_B_blue = np.mean(
        [run["snaps"][-1]["blue_ever"].astype(np.float32)
         for run in runs_by_setup["B"]], axis=0,
    )
    B_blue_cells = (p_B_blue >= 0.5) & non_wall  # [H, W]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    width = 0.55
    fates = ["blue-only", "BOTH (blue+red saw)", "red-only", "NOBODY saw"]
    palette = {"blue-only": "#5a9bd4", "BOTH (blue+red saw)": "#9e69b6",
               "red-only": "#d67a3a", "NOBODY saw": "#222222"}

    for col, setup in enumerate(s for s in SETUPS if s.key != "B"):
        ax = axes[col]
        # Per-cell mean blue_ever, red_ever in setup
        p_b = np.mean([r["snaps"][-1]["blue_ever"].astype(np.float32)
                       for r in runs_by_setup[setup.key]], axis=0)
        p_r = np.mean([r["snaps"][-1]["red_ever"].astype(np.float32)
                       for r in runs_by_setup[setup.key]], axis=0)
        b = p_b >= 0.5
        rr = p_r >= 0.5
        n_total = int(B_blue_cells.sum())
        n_blue_only = int((B_blue_cells & b & ~rr).sum())
        n_both = int((B_blue_cells & b & rr).sum())
        n_red_only = int((B_blue_cells & ~b & rr).sum())
        n_nobody = int((B_blue_cells & ~b & ~rr).sum())
        counts = [n_blue_only, n_both, n_red_only, n_nobody]
        fracs = [c / max(1, n_total) * 100 for c in counts]
        bottom = 0
        for fate, frac, n in zip(fates, fracs, counts):
            ax.bar([setup.key], [frac], bottom=[bottom], width=width,
                   color=palette[fate], edgecolor="white", linewidth=0.8,
                   label=fate if col == 0 else None)
            if frac > 4:
                ax.text(0, bottom + frac / 2, f"{fate}\n{frac:.0f}%  ({n} cells)",
                        ha="center", va="center", fontsize=9,
                        color="white" if fate != "NOBODY saw" else "white")
            bottom += frac
        ax.set_ylim(0, 105)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylabel("% of B's blue-known cells")
        ax.set_title(
            f"{setup.short}\nfate of B's {n_total} blue-only cells",
            fontsize=10,
        )
        ax.set_xticks([]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Cell-fate transitions — what happened to the cells blue covered in B?\n"
        "(of the cells reliably blue-known in the no-red baseline, what fraction "
        "stayed blue-only / became contested / lost to red / lost entirely under C*?)",
        fontsize=11, y=1.0,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 5: frontier race --------------------

def candidate_5_frontier_race(runs_by_setup, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = {"B": "#1f4e79", "C1": "#cc7a00", "C2": "#7a0000"}

    means = {}
    for setup in SETUPS:
        curves = np.stack([r["curve"] for r in runs_by_setup[setup.key]])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        means[setup.key] = mean
        t = np.arange(1, mean.shape[0] + 1)
        ax.fill_between(t, mean - std, mean + std,
                        color=palette[setup.key], alpha=0.18, linewidth=0)
        ax.plot(t, mean, color=palette[setup.key], lw=2.2,
                label=f"{setup.short}  (final {mean[-1]:.1f}%)")

    # Stolen-area shading between B and C2
    t = np.arange(1, means["B"].shape[0] + 1)
    ax.fill_between(t, means["C2"], means["B"],
                    where=(means["B"] > means["C2"]),
                    color="#cc7a00", alpha=0.30, linewidth=0,
                    label="stolen area (B − C2)")

    # Annotate stolen integral
    stolen = float((means["B"] - means["C2"]).sum())
    ax.text(0.02, 0.97,
            f"area under (B − C2) = {stolen:.0f} cell-frames stolen by 2 reds",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(facecolor="#fff3e0", edgecolor="#cc7a00", boxstyle="round,pad=0.4"))

    ax.set_xlim(1, t[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("episode step")
    ax.set_ylabel("blue team coverage  (% of non-wall cells ever observed)")
    ax.set_title("Frontier race — coverage % vs time, mean ± 1σ across 10 seeds",
                 fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 6: agent polygons --------------------

def _per_agent_visit_counts(run: dict, obs_radius: int = 1) -> np.ndarray:
    """Per-agent visit count grid [N, H, W]. Stamp a 3×3 obs window each step."""
    H, W = run["terrain"].shape
    N = run["paths"].shape[1]
    counts = np.zeros((N, H, W), dtype=np.int32)
    paths = run["paths"]
    T = paths.shape[0]
    for t in range(T):
        for a in range(N):
            r, c = int(paths[t, a, 0]), int(paths[t, a, 1])
            rs = slice(max(0, r - obs_radius), min(H, r + obs_radius + 1))
            cs = slice(max(0, c - obs_radius), min(W, c + obs_radius + 1))
            counts[a, rs, cs] += 1
    return counts


def candidate_6_agent_polygons(runs_by_setup, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for col, setup in enumerate(SETUPS):
        ax = axes[col]
        # Sum per-agent visit counts across all seeds
        runs = runs_by_setup[setup.key]
        N = runs[0]["paths"].shape[1]
        H, W = runs[0]["terrain"].shape
        counts = np.zeros((N, H, W), dtype=np.int32)
        for run in runs:
            counts += _per_agent_visit_counts(run)
        team_ids = runs[0]["team_ids"]
        terrain = runs[0]["terrain"]
        non_wall = terrain != CELL_WALL

        # argmax agent per cell (cells with zero visits get -1)
        any_visit = counts.sum(axis=0) > 0
        argmax_a = np.where(any_visit, counts.argmax(axis=0), -1)

        # Build palette: blues for blue agents, reds for red agents
        n_blue = int((team_ids == 0).sum())
        n_red = N - n_blue
        blue_palette = plt.cm.Blues(np.linspace(0.45, 0.95, max(1, n_blue)))
        red_palette = plt.cm.Reds(np.linspace(0.55, 0.95, max(1, n_red)))
        agent_colors = np.zeros((N, 4), dtype=np.float32)
        for a in range(N):
            if team_ids[a] == 0:
                bi = a
                agent_colors[a] = blue_palette[bi]
            else:
                ri = a - n_blue
                agent_colors[a] = red_palette[ri]

        rgba = np.full((H, W, 4), (0.96, 0.96, 0.96, 1.0), dtype=np.float32)
        for a in range(N):
            mask = (argmax_a == a) & non_wall
            rgba[mask] = agent_colors[a]
        _imshow_grid(ax, rgba)
        _draw_walls(ax, terrain)
        ax.set_xticks([]); ax.set_yticks([])
        # Build a compact legend
        legend_handles = []
        for a in range(N):
            tag = f"B{a}" if team_ids[a] == 0 else f"R{a - n_blue}"
            legend_handles.append(
                plt.Line2D([0], [0], marker="s", color="none",
                           markerfacecolor=agent_colors[a], markeredgecolor="white",
                           markersize=10, label=tag)
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
                  ncol=1, frameon=True, framealpha=0.9, borderaxespad=0.4)
        ax.set_title(setup.short, fontsize=10)

    fig.suptitle(
        "Per-agent territories — who scanned each cell most often (10 seeds pooled)\n"
        "(each cell coloured by its argmax-visit agent; "
        "watch B's tidy 5-way split collapse into C2's 3 + a red block)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 7: 3D space-time tubes --------------------

def candidate_7_spacetime_tubes(runs_by_setup, out: Path) -> None:
    fig = plt.figure(figsize=(15.5, 5.6))
    for col, setup in enumerate(SETUPS):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        run = runs_by_setup[setup.key][TAPE_SEED]
        paths = run["paths"]                         # [T+1, N, 2]
        team_ids = run["team_ids"]
        terrain = run["terrain"]
        H, W = terrain.shape
        T = paths.shape[0]
        ts = np.arange(T)

        # Faint terrain "floor" at z=0: walls a soft grey, non-wall barely visible.
        wall = terrain == CELL_WALL
        Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))
        floor_z = np.zeros_like(Xg, dtype=np.float32)
        face = np.full((H, W, 4), (0.97, 0.97, 0.97, 0.20), dtype=np.float32)
        face[wall] = (0.70, 0.70, 0.70, 0.30)
        ax.plot_surface(Xg, Yg, floor_z, facecolors=face,
                        rstride=1, cstride=1, edgecolor="none",
                        antialiased=False, shade=False)

        n_blue = run["n_blue"]
        n_red = run["n_red"]
        for a in range(paths.shape[1]):
            xs = paths[:, a, 1].astype(np.float32)
            ys = paths[:, a, 0].astype(np.float32)
            if team_ids[a] == 0:
                base_c = plt.cm.Blues(0.45 + 0.45 * a / max(1, n_blue - 1))
                lw = 1.7
                marker_face = base_c
            else:
                base_c = plt.cm.Reds(0.55 + 0.35 * (a - n_blue) / max(1, n_red - 1))
                lw = 2.6
                marker_face = base_c
            # Tube approximation: 3 lightly offset polylines for thickness
            for dx, dy, alpha in [(0, 0, 0.95), (0.18, 0.0, 0.45), (0, 0.18, 0.45)]:
                ax.plot(xs + dx, ys + dy, ts, color=base_c, lw=lw,
                        alpha=alpha, solid_capstyle="round")
            # Start dot at floor
            ax.scatter([xs[0]], [ys[0]], [0], color=base_c, s=22,
                       edgecolors="white", linewidths=0.6, depthshade=True)
            # End cube at top
            ax.scatter([xs[-1]], [ys[-1]], [T - 1], color=marker_face, s=80,
                       marker="s", edgecolors="white", linewidths=1.0, depthshade=True)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_zlim(0, T)
        ax.set_xlabel("col (x)", fontsize=8)
        ax.set_ylabel("row (y)", fontsize=8)
        ax.set_zlabel("episode step", fontsize=8)
        ax.set_title(
            f"{setup.short}\n{n_blue} blue tubes  +  {n_red} red tubes  ·  seed 0",
            fontsize=10,
        )
        ax.view_init(elev=22, azim=-65)
        ax.tick_params(axis="both", labelsize=7)
        _clean_3d_axes(ax)
        ax.invert_yaxis()  # so row=0 sits at the back

    fig.suptitle(
        "Space-time tubes — agent trajectories in (x, y, t)\n"
        "blue tubes braid across the map; red tubes climb almost straight up "
        "(they camp).  Square = final position.",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- helpers for 3D first-seen --------------------

def _first_seen_step_grid(run: dict, max_step: int, obs_radius: int = 1) -> np.ndarray:
    """Earliest step at which any blue agent had this cell in its 3x3 obs window.
    Cells never observed get value max_step + 1."""
    H, W = run["terrain"].shape
    paths = run["paths"]
    team_ids = run["team_ids"]
    out = np.full((H, W), max_step + 1, dtype=np.int32)
    T = min(max_step + 1, paths.shape[0])
    for t in range(T):
        for a in range(paths.shape[1]):
            if team_ids[a] != 0:
                continue
            r, c = int(paths[t, a, 0]), int(paths[t, a, 1])
            rs = slice(max(0, r - obs_radius), min(H, r + obs_radius + 1))
            cs = slice(max(0, c - obs_radius), min(W, c + obs_radius + 1))
            sub = out[rs, cs]
            np.minimum(sub, t, out=sub)
    return out


# -------------------- candidate 8: first-seen landscape --------------------

def candidate_8_first_seen_landscape(runs_by_setup, out: Path) -> None:
    """Per-cell first-seen step rendered as bar towers (clearer than mesh)."""
    fig = plt.figure(figsize=(20.0, 7.5))
    max_step = MAX_STEPS_EVAL
    never_z = max_step + 30

    medians = {}
    for s in SETUPS:
        per_seed = []
        for run in runs_by_setup[s.key]:
            per_seed.append(_first_seen_step_grid(run, max_step))
        medians[s.key] = np.median(np.stack(per_seed), axis=0).astype(np.int32)

    for col, setup in enumerate(SETUPS):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        z = medians[setup.key]
        terrain = runs_by_setup[setup.key][0]["terrain"]
        H, W = z.shape
        wall = terrain == CELL_WALL
        denied = (z > max_step) & ~wall

        xs, ys, hs, cs = [], [], [], []
        for r in range(H):
            for c in range(W):
                if wall[r, c]:
                    continue
                if denied[r, c]:
                    xs.append(c); ys.append(r)
                    hs.append(float(never_z))
                    cs.append((0.05, 0.05, 0.05, 1.0))
                else:
                    xs.append(c); ys.append(r)
                    hs.append(float(z[r, c]))
                    norm = min(1.0, z[r, c] / max_step)
                    cs.append(plt.cm.viridis_r(norm))
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        hs = np.asarray(hs, dtype=np.float32)
        cs_arr = np.asarray(cs, dtype=np.float32)
        zs = np.zeros_like(hs)
        ax.bar3d(xs, ys, zs, 0.88, 0.88, hs,
                 color=cs_arr, shade=True, alpha=0.96,
                 edgecolor=(0, 0, 0, 0.25), linewidth=0.25)

        n_denied = int(denied.sum())
        ax.set_zlim(0, never_z)
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_xlabel("col (x)", fontsize=11, labelpad=8)
        ax.set_ylabel("row (y)", fontsize=11, labelpad=8)
        ax.set_zlabel("first-seen step  (median across 10 seeds)",
                      fontsize=11, labelpad=10)
        ax.set_title(
            f"{setup.short}\n{n_denied} denied (black towers reach the ceiling)",
            fontsize=12, pad=12,
        )
        ax.view_init(elev=32, azim=-58)
        ax.tick_params(axis="both", labelsize=9)
        _clean_3d_axes(ax)
        ax.invert_yaxis()

    fig.suptitle(
        "3D first-seen towers — bar height = step at which blue first observed each cell\n"
        "(yellow = found early, dark green = found late, black = blue never reached)",
        fontsize=13, y=1.04,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86,
                        bottom=0.04, wspace=0.05)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 9: visit-count cityscape --------------------

def candidate_9_visit_cityscape(runs_by_setup, out: Path) -> None:
    fig = plt.figure(figsize=(20.0, 7.5))
    for col, setup in enumerate(SETUPS):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        runs = runs_by_setup[setup.key]
        N = runs[0]["paths"].shape[1]
        H, W = runs[0]["terrain"].shape
        terrain = runs[0]["terrain"]
        team_ids = runs[0]["team_ids"]
        non_wall = terrain != CELL_WALL

        counts = np.zeros((N, H, W), dtype=np.int32)
        for run in runs:
            counts += _per_agent_visit_counts(run)

        n_blue = int((team_ids == 0).sum())
        n_red = N - n_blue
        blue_pal = plt.cm.Blues(np.linspace(0.50, 0.92, max(1, n_blue)))
        red_pal = plt.cm.Reds(np.linspace(0.55, 0.92, max(1, n_red)))
        agent_colors = np.zeros((N, 4), dtype=np.float32)
        for a in range(N):
            agent_colors[a] = (
                blue_pal[a] if team_ids[a] == 0 else red_pal[a - n_blue]
            )

        height = counts.sum(axis=0)
        argmax_a = counts.argmax(axis=0)

        xs, ys, hs, cs = [], [], [], []
        for r in range(H):
            for c in range(W):
                if not non_wall[r, c] or height[r, c] == 0:
                    continue
                xs.append(c)
                ys.append(r)
                hs.append(int(height[r, c]))
                cs.append(agent_colors[argmax_a[r, c]])
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        hs = np.array(hs, dtype=np.float32)
        cs_arr = np.array(cs, dtype=np.float32)
        zs = np.zeros_like(hs)
        ax.bar3d(xs, ys, zs, 0.88, 0.88, hs,
                 color=cs_arr, shade=True, alpha=0.96,
                 edgecolor=(0, 0, 0, 0.25), linewidth=0.25)

        # Per-agent colour legend (B0…Bn / R0…Rm)
        legend_handles = []
        for a in range(N):
            tag = f"B{a}" if team_ids[a] == 0 else f"R{a - n_blue}"
            legend_handles.append(
                plt.Line2D([0], [0], marker="s", linestyle="none",
                           markerfacecolor=agent_colors[a],
                           markeredgecolor="white", markersize=11, label=tag)
            )
        ax.legend(handles=legend_handles, loc="upper left",
                  bbox_to_anchor=(-0.02, 1.0), fontsize=9, ncol=1,
                  frameon=True, framealpha=0.9, borderaxespad=0.4)

        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_xlabel("col (x)", fontsize=11, labelpad=8)
        ax.set_ylabel("row (y)", fontsize=11, labelpad=8)
        ax.set_zlabel("total visits  (sum across agents × 10 seeds)",
                      fontsize=11, labelpad=10)
        ax.set_title(
            f"{setup.short}    {n_blue} blue + {n_red} red\n"
            f"tallest bar = {int(hs.max())} visits",
            fontsize=12, pad=12,
        )
        ax.view_init(elev=32, azim=-58)
        ax.tick_params(axis="both", labelsize=9)
        _clean_3d_axes(ax)
        ax.invert_yaxis()

    fig.suptitle(
        "Visit-count cityscape — bar height = total visits per cell, colour = the agent that visited it most\n"
        "(left B: 5 blue agents tile the map evenly · centre C1: red builds 1 tower · right C2: red builds 2 towers, blue's central skyline collapses)",
        fontsize=13, y=1.04,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86,
                        bottom=0.04, wspace=0.05)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 10: coverage slabs --------------------

def candidate_10_coverage_slabs(runs_by_setup, out: Path) -> None:
    fig = plt.figure(figsize=(20.0, 7.5))
    for col, setup in enumerate(SETUPS):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        run = runs_by_setup[setup.key][TAPE_SEED]
        terrain = run["terrain"]
        H, W = terrain.shape

        # Ghost terrain at z=0 — soft greys, no harsh wall border.
        wall = terrain == CELL_WALL
        Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))
        face = np.full((H, W, 4), (0.97, 0.97, 0.97, 0.20), dtype=np.float32)
        face[wall] = (0.70, 0.70, 0.70, 0.30)
        ax.plot_surface(Xg, Yg, np.zeros_like(Xg, dtype=np.float32),
                        facecolors=face, rstride=1, cstride=1,
                        edgecolor="none", antialiased=False, shade=False)

        for snap in run["snaps"]:
            z = float(snap["step"])
            blue_known = snap["blue_ever"]
            red_known = snap["red_ever"]
            br, bc = np.where(blue_known)
            ax.scatter(bc, br, np.full(br.size, z), color="#2a6fb8",
                       s=58, alpha=0.55, edgecolors="white",
                       linewidths=0.5, depthshade=False)
            if red_known.any():
                rr, rc = np.where(red_known)
                ax.scatter(rc, rr, np.full(rr.size, z + 1.5), color="#c53b3b",
                           s=70, alpha=0.92, edgecolors="white",
                           linewidths=0.6, depthshade=False)

        steps = [snap["step"] for snap in run["snaps"]]
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_zlim(0, max(steps) + 8)
        ax.set_xlabel("col (x)", fontsize=11, labelpad=8)
        ax.set_ylabel("row (y)", fontsize=11, labelpad=8)
        ax.set_zlabel("snapshot step", fontsize=11, labelpad=10)
        ax.set_title(
            f"{setup.short}\nslabs at t = {steps}  (seed 0)",
            fontsize=12, pad=12,
        )
        ax.view_init(elev=22, azim=-62)
        ax.tick_params(axis="both", labelsize=9)
        _clean_3d_axes(ax)
        ax.invert_yaxis()

    fig.suptitle(
        "Coverage slabs — voxel cloud of blue-known (blue) and red-known (red) over (x, y, t)\n"
        "(each horizontal slab is one snapshot; blue cloud spreads, red cloud climbs as a narrow column in the corners)",
        fontsize=13, y=1.04,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86,
                        bottom=0.04, wspace=0.05)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 11: delay towers vs B --------------------

def candidate_11_delay_towers(runs_by_setup, out: Path) -> None:
    """Per-cell delay vs B baseline as bar towers; denied cells are black spires."""
    fig = plt.figure(figsize=(15.0, 7.5))
    max_step = MAX_STEPS_EVAL
    never_z = max_step + 30

    def median_first_seen(setup_key):
        per_seed = []
        for run in runs_by_setup[setup_key]:
            per_seed.append(_first_seen_step_grid(run, max_step))
        return np.median(np.stack(per_seed), axis=0).astype(np.int32)

    z_B = median_first_seen("B")
    targets = [s for s in SETUPS if s.key != "B"]

    for col, setup in enumerate(targets):
        ax = fig.add_subplot(1, len(targets), col + 1, projection="3d")
        z_C = median_first_seen(setup.key)
        terrain = runs_by_setup[setup.key][0]["terrain"]
        H, W = z_C.shape
        wall = terrain == CELL_WALL
        denied = (z_C > max_step) & ~wall

        delay = np.where(wall | denied, 0, np.maximum(0, z_C - z_B)).astype(np.int32)
        live_delay = delay[~wall & ~denied]
        delay_max = max(1.0, float(live_delay.max()) if live_delay.size else 1.0)

        xs_d, ys_d, hs_d, cs_d = [], [], [], []
        xs_f, ys_f = [], []
        for r in range(H):
            for c in range(W):
                if wall[r, c]:
                    continue
                if denied[r, c]:
                    xs_d.append(c); ys_d.append(r)
                    hs_d.append(float(never_z))
                    cs_d.append((0.05, 0.05, 0.05, 1.0))
                elif delay[r, c] > 0:
                    xs_d.append(c); ys_d.append(r)
                    hs_d.append(float(delay[r, c]))
                    norm = min(1.0, delay[r, c] / delay_max)
                    cs_d.append(plt.cm.OrRd(0.30 + 0.65 * norm))
                else:
                    xs_f.append(c); ys_f.append(r)

        if xs_d:
            ax.bar3d(np.asarray(xs_d, dtype=np.float32),
                     np.asarray(ys_d, dtype=np.float32),
                     np.zeros(len(xs_d), dtype=np.float32),
                     0.88, 0.88, np.asarray(hs_d, dtype=np.float32),
                     color=np.asarray(cs_d, dtype=np.float32),
                     shade=True, alpha=0.96,
                     edgecolor=(0, 0, 0, 0.25), linewidth=0.25)

        # "no delay" cells as a thin blue floor
        if xs_f:
            ax.bar3d(np.asarray(xs_f, dtype=np.float32),
                     np.asarray(ys_f, dtype=np.float32),
                     np.zeros(len(xs_f), dtype=np.float32),
                     0.88, 0.88, np.full(len(xs_f), 1.5, dtype=np.float32),
                     color=(0.45, 0.62, 0.85, 0.85), shade=True, edgecolor="none")

        n_denied = int(denied.sum())
        delay_total = int(delay.sum())
        ax.set_zlim(0, never_z)
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_xlabel("col (x)", fontsize=11, labelpad=8)
        ax.set_ylabel("row (y)", fontsize=11, labelpad=8)
        ax.set_zlabel("delay  (extra steps vs B)", fontsize=11, labelpad=10)
        ax.set_title(
            f"{setup.short}  vs  B baseline\n"
            f"orange tower = delay magnitude  ·  total delay = {delay_total} cell-steps  ·  "
            f"{n_denied} denied (black spires)",
            fontsize=11, pad=12,
        )
        ax.view_init(elev=30, azim=-58)
        ax.tick_params(axis="both", labelsize=9)
        _clean_3d_axes(ax)
        ax.invert_yaxis()

    fig.suptitle(
        "3D delay towers — orange height = extra steps red made blue wait per cell\n"
        "(blue floor = no delay; orange tower = delay; black spire = cell C* never reached)",
        fontsize=13, y=1.04,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86,
                        bottom=0.04, wspace=0.05)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidates 12 / 13: Bayesian confidence --------------------

def _scan_event_counts(run: dict, obs_radius: int = 1):
    """Per-cell total scan-events for blue and red, separately.
    A scan event = the cell was inside *some* agent's 3×3 obs window at time t."""
    H, W = run["terrain"].shape
    paths = run["paths"]
    team_ids = run["team_ids"]
    blue = np.zeros((H, W), dtype=np.int32)
    red = np.zeros((H, W), dtype=np.int32)
    T = paths.shape[0]
    for t in range(T):
        for a in range(paths.shape[1]):
            r, c = int(paths[t, a, 0]), int(paths[t, a, 1])
            rs = slice(max(0, r - obs_radius), min(H, r + obs_radius + 1))
            cs = slice(max(0, c - obs_radius), min(W, c + obs_radius + 1))
            if team_ids[a] == 0:
                blue[rs, cs] += 1
            else:
                red[rs, cs] += 1
    return blue, red


def _bayes_setup_stats(runs_by_setup):
    stats = {}
    for setup in SETUPS:
        runs = runs_by_setup[setup.key]
        H, W = runs[0]["terrain"].shape
        blue_acc = np.zeros((H, W), dtype=np.int64)
        red_acc = np.zeros((H, W), dtype=np.int64)
        for run in runs:
            b, r = _scan_event_counts(run)
            blue_acc += b
            red_acc += r
        stats[setup.key] = {
            "blue": blue_acc, "red": red_acc,
            "net": (blue_acc - red_acc).astype(np.int64),
            "terrain": runs[0]["terrain"],
        }
    return stats


def candidate_12_bayesian_confidence(runs_by_setup, out: Path) -> None:
    """Per-cell signed evidence — blue scans = +1, red scans = −1, summed."""
    stats = _bayes_setup_stats(runs_by_setup)
    vmax = max(int(np.abs(d["net"]).max()) for d in stats.values())
    cmap = plt.cm.RdBu  # red ←→ white ←→ blue, centred at 0

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0))
    last_im = None
    for col, setup in enumerate(SETUPS):
        ax = axes[col]
        d = stats[setup.key]
        terrain = d["terrain"]
        net = d["net"].astype(np.float32)
        non_wall = terrain != CELL_WALL
        disp = np.where(non_wall, net, np.nan)
        last_im = _imshow_grid(ax, disp, cmap=cmap, vmin=-vmax, vmax=vmax)
        _draw_walls(ax, terrain)
        n_blue = int(d["blue"].sum())
        n_red = int(d["red"].sum())
        n_corrupt = int(((net < 0) & non_wall).sum())
        n_confid = int(((net > 0) & non_wall).sum())
        ax.set_title(
            f"{setup.short}\n"
            f"{n_confid} blue-confident cells   ·   {n_corrupt} red-corrupted cells\n"
            f"total scans  →  blue={n_blue:,}   red={n_red:,}",
            fontsize=11,
        )
        ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(last_im, ax=axes.tolist(),
                        orientation="horizontal", fraction=0.05,
                        pad=0.07, shrink=0.6)
    cbar.set_label(
        "net evidence per cell  =  (blue scan-events) − (red scan-events)   "
        "→  blue = trusted, white = no info / balanced, red = red corrupted it",
        fontsize=10,
    )

    fig.suptitle(
        "Bayesian-style confidence map — every blue scan adds evidence, every red scan removes it\n"
        "(signed per-cell evidence integrated over 200 steps × 10 seeds   ·   "
        "B has only blue evidence; in C1/C2 red carves out poisoned-belief regions)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def candidate_13_confidence_pillars(runs_by_setup, out: Path) -> None:
    """3D bar3d of the signed Bayesian confidence — blue pillars rise where
    blue dominates; red pillars where red corrupts."""
    stats = _bayes_setup_stats(runs_by_setup)
    vmax = max(int(np.abs(d["net"]).max()) for d in stats.values())

    fig = plt.figure(figsize=(20.0, 7.5))
    for col, setup in enumerate(SETUPS):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        d = stats[setup.key]
        terrain = d["terrain"]
        net = d["net"]
        H, W = terrain.shape
        wall = terrain == CELL_WALL

        xs, ys, hs, cs = [], [], [], []
        for r in range(H):
            for c in range(W):
                if wall[r, c]:
                    continue
                v = net[r, c]
                if v == 0:
                    continue
                xs.append(c); ys.append(r)
                hs.append(float(abs(v)))
                if v > 0:
                    cs.append(plt.cm.Blues(0.45 + 0.5 * min(1.0, v / vmax)))
                else:
                    cs.append(plt.cm.Reds(0.50 + 0.45 * min(1.0, -v / vmax)))
        if xs:
            ax.bar3d(np.asarray(xs, dtype=np.float32),
                     np.asarray(ys, dtype=np.float32),
                     np.zeros(len(xs), dtype=np.float32),
                     0.88, 0.88, np.asarray(hs, dtype=np.float32),
                     color=np.asarray(cs, dtype=np.float32),
                     shade=True, alpha=0.96,
                     edgecolor=(0, 0, 0, 0.25), linewidth=0.25)

        ax.set_zlim(0, vmax)
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_xlabel("col (x)", fontsize=11, labelpad=8)
        ax.set_ylabel("row (y)", fontsize=11, labelpad=8)
        ax.set_zlabel("|net evidence|", fontsize=11, labelpad=10)
        ax.set_title(
            f"{setup.short}\nblue pillar = trusted   ·   red pillar = corrupted",
            fontsize=12, pad=12,
        )
        ax.view_init(elev=32, azim=-58)
        ax.tick_params(axis="both", labelsize=9)
        _clean_3d_axes(ax)
        ax.invert_yaxis()

    fig.suptitle(
        "3D confidence pillars — height = magnitude of net evidence, colour = sign\n"
        "(blue pillars: blue scans dominate this cell · red pillars: red scans corrupt this cell · "
        "the skyline IS the team's posterior over the map at t=200)",
        fontsize=13, y=1.04,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86,
                        bottom=0.04, wspace=0.05)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- candidate 14: confidence evolution over time --------------------

CONF_TIMES = (40, 80, 120, 160, 200)


def _scan_event_counts_cumulative(run, obs_radius: int = 1):
    """Per-step cumulative scan counts for blue and red.
    Returns blue_cum[T, H, W], red_cum[T, H, W] where index t is the count
    at the END of step t (so index 0 = after step 0)."""
    H, W = run["terrain"].shape
    paths = run["paths"]
    team_ids = run["team_ids"]
    T = paths.shape[0]
    blue_inc = np.zeros((T, H, W), dtype=np.int32)
    red_inc = np.zeros((T, H, W), dtype=np.int32)
    for t in range(T):
        for a in range(paths.shape[1]):
            r, c = int(paths[t, a, 0]), int(paths[t, a, 1])
            rs = slice(max(0, r - obs_radius), min(H, r + obs_radius + 1))
            cs = slice(max(0, c - obs_radius), min(W, c + obs_radius + 1))
            if team_ids[a] == 0:
                blue_inc[t, rs, cs] += 1
            else:
                red_inc[t, rs, cs] += 1
    return np.cumsum(blue_inc, axis=0), np.cumsum(red_inc, axis=0)


def candidate_14_confidence_evolution(runs_by_setup, out: Path) -> None:
    """Time-strip of the Bayesian confidence map. Shows the cumulative net
    evidence per cell at each snapshot, so you can see red islands growing."""
    nets_by_setup: Dict[str, List[np.ndarray]] = {}
    for setup in SETUPS:
        runs = runs_by_setup[setup.key]
        H, W = runs[0]["terrain"].shape
        net_at = [np.zeros((H, W), dtype=np.int64) for _ in CONF_TIMES]
        for run in runs:
            blue_cum, red_cum = _scan_event_counts_cumulative(run)
            T = blue_cum.shape[0]
            for i, t in enumerate(CONF_TIMES):
                idx = min(t - 1, T - 1)
                net_at[i] += (blue_cum[idx] - red_cum[idx]).astype(np.int64)
        nets_by_setup[setup.key] = net_at

    vmax = max(int(np.abs(arr).max())
               for arrs in nets_by_setup.values() for arr in arrs)
    cmap = plt.cm.RdBu

    n_rows = len(SETUPS)
    n_cols = len(CONF_TIMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 2.9 * n_rows))
    last_im = None
    for r, setup in enumerate(SETUPS):
        terrain = runs_by_setup[setup.key][0]["terrain"]
        non_wall = terrain != CELL_WALL
        for c, t in enumerate(CONF_TIMES):
            ax = axes[r, c]
            net = nets_by_setup[setup.key][c].astype(np.float32)
            disp = np.where(non_wall, net, np.nan)
            last_im = _imshow_grid(ax, disp, cmap=cmap, vmin=-vmax, vmax=vmax)
            _draw_walls(ax, terrain)
            n_corrupt = int(((net < 0) & non_wall).sum())
            n_confid = int(((net > 0) & non_wall).sum())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {t}", fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(setup.short, fontsize=11, fontweight="bold")
            ax.text(
                0.02, 0.98,
                f"+{n_confid}  -{n_corrupt}",
                transform=ax.transAxes, fontsize=8,
                color="#222", ha="left", va="top",
                bbox=dict(facecolor=(1, 1, 1, 0.85),
                          edgecolor="none", boxstyle="round,pad=0.18"),
            )

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                        orientation="horizontal", fraction=0.04,
                        pad=0.06, shrink=0.5)
    cbar.set_label(
        "running net evidence  =  (blue scan-events up to t)  −  (red scan-events up to t)\n"
        "blue cell = trusted, red cell = corrupted by adversarial scans, white = no info",
        fontsize=10,
    )

    fig.suptitle(
        "Confidence evolution — Bayesian-style map over time (10 seeds pooled)\n"
        "(left→right: episode unfolds   ·   B stays uniformly blue; in C1/C2 red islands grow each frame)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------- HTML viewer --------------------

def build_html(out: Path) -> None:
    items = []
    for k in sorted(CAND_PNG):
        png, title, blurb = CAND_PNG[k]
        items.append(f"""
        <section class="card">
          <h2>{k}. {title}</h2>
          <p class="blurb">{blurb}</p>
          <img src="{png}" alt="{title}">
        </section>
        """)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Adversary effect — pick one</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif;
         max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222;
         background: #fafafa; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 17px; margin-bottom: 4px; }}
  .blurb {{ color: #555; margin: 0 0 10px 0; font-size: 14px; }}
  .card {{ background: white; padding: 16px 20px; margin: 18px 0;
          border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  img {{ max-width: 100%; height: auto; border-radius: 6px;
        border: 1px solid #eee; }}
  .lede {{ background: #fff3e0; padding: 10px 14px; border-radius: 8px;
           border: 1px solid #ffd6a3; }}
</style>
</head><body>
<h1>Adversary effect — fourteen candidate visualizations</h1>
<p class="lede">Each panel is a <em>different lens</em> on the question
"what does red do to blue?" Pick the one that tells the story best — or
mix-and-match — and tell me which you want kept.</p>
{''.join(items)}
</body></html>"""
    out.write_text(html)
    print(f"wrote {out}")


def main() -> None:
    seeds = list(range(N_SEEDS))
    runs_by_setup: Dict[str, List[dict]] = {}
    for setup in SETUPS:
        print(f"rolling out {setup.key} x {len(seeds)} seeds…")
        runs_by_setup[setup.key] = collect(setup, seeds)

    candidate_1_mission_tape(runs_by_setup, OUT_DIR / CAND_PNG[1][0])
    candidate_2_delay_map(runs_by_setup, OUT_DIR / CAND_PNG[2][0])
    candidate_3_denied_cells(runs_by_setup, OUT_DIR / CAND_PNG[3][0])
    candidate_4_fate_transitions(runs_by_setup, OUT_DIR / CAND_PNG[4][0])
    candidate_5_frontier_race(runs_by_setup, OUT_DIR / CAND_PNG[5][0])
    candidate_6_agent_polygons(runs_by_setup, OUT_DIR / CAND_PNG[6][0])
    candidate_7_spacetime_tubes(runs_by_setup, OUT_DIR / CAND_PNG[7][0])
    candidate_8_first_seen_landscape(runs_by_setup, OUT_DIR / CAND_PNG[8][0])
    candidate_9_visit_cityscape(runs_by_setup, OUT_DIR / CAND_PNG[9][0])
    candidate_10_coverage_slabs(runs_by_setup, OUT_DIR / CAND_PNG[10][0])
    candidate_11_delay_towers(runs_by_setup, OUT_DIR / CAND_PNG[11][0])
    candidate_12_bayesian_confidence(runs_by_setup, OUT_DIR / CAND_PNG[12][0])
    candidate_13_confidence_pillars(runs_by_setup, OUT_DIR / CAND_PNG[13][0])
    candidate_14_confidence_evolution(runs_by_setup, OUT_DIR / CAND_PNG[14][0])
    build_html(OUT_DIR / HTML)


if __name__ == "__main__":
    main()
