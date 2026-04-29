"""Isometric episode GIFs with per-team toggleable belief layers.

For each canonical setup (B, C1, C2) at N=5 on 16×16, run one episode from
the trained checkpoints and render an isometric 3D view as an animated GIF
for each "layer" the reader can toggle in the meta-report:

  iso_{key}_global.gif  — ground truth terrain + all agents (reference)
  iso_{key}_blue.gif    — team-blue merged belief; unknown cells raised
                          as red fog columns; red agents faded
  iso_{key}_red.gif     — team-red merged belief (skipped for B — no red)

Output dir: ``experiments/meta-report/``. The HTML in ``scripts/meta_report.py``
wires these into a radio-button UI so readers flip between views in place.

Run:  python scripts/isometric_episode.py [--subsample 5] [--fps 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, JointRedActor
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward,
    normalized_exploration_reward,
)
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
from red_within_blue.visualizer import _merge_team_belief


OUT_DIR = Path("experiments/meta-report")
CANON_SEED = 0
MAX_STEPS = 200


@dataclass
class Setup:
    key: str
    label: str
    config: str
    blue_ckpt: str
    red_ckpt: Optional[str]


SETUPS: List[Setup] = [
    Setup(
        key="B",
        label="B — 5 blue, 0 red",
        config="configs/survey-local-16-N5-from-N4.yaml",
        blue_ckpt="experiments/survey-local-16-N5-from-N4/checkpoint.npz",
        red_ckpt=None,
    ),
    Setup(
        key="C1",
        label="C1 — 4 blue, 1 red",
        config="configs/compromise-16x16-5-4b1r.yaml",
        blue_ckpt="experiments/compromise-16x16-5-4b1r-coevo/checkpoint.npz",
        red_ckpt="experiments/compromise-16x16-5-4b1r-coevo/joint_red_checkpoint.npz",
    ),
    Setup(
        key="C2",
        label="C2 — 3 blue, 2 red",
        config="configs/compromise-16x16-5-3b2r.yaml",
        blue_ckpt="experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz",
        red_ckpt="experiments/compromise-16x16-5-3b2r-coevo/joint_red_checkpoint.npz",
    ),
]


# --- checkpoint loaders (shared pattern with scripts/meta_report.py)

def _strip_seed_dim(flat, ref_flat):
    return {k: (v[0] if v.ndim == ref_flat[k].ndim + 1 else v) for k, v in flat.items()}


def _load_blue(cfg: ExperimentConfig, ckpt_path: str) -> Tuple[Actor, dict]:
    flat = load_checkpoint(ckpt_path)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
    )
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    return actor, unflatten_params(_strip_seed_dim(flat, flatten_params(ref)), ref)


def _load_red(cfg: ExperimentConfig, ckpt_path: str) -> Tuple[JointRedActor, dict]:
    flat = load_checkpoint(ckpt_path)
    n_red = cfg.env.num_red_agents
    actor = JointRedActor(
        num_red=n_red,
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.train.red_hidden_dim,
        num_layers=cfg.train.red_num_layers,
    )
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(n_red * cfg.obs_dim))
    return actor, unflatten_params(_strip_seed_dim(flat, flatten_params(ref)), ref)


# --- rollout with per-step state capture

def _rollout_capture(cfg: ExperimentConfig, blue_actor, blue_params,
                    red_actor, red_params, seed: int, max_steps: int):
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red
    reward_fn = normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)

    @jax.jit
    def _blue_act(bp, obs, key):
        return jax.random.categorical(key, blue_actor.apply(bp, obs))

    @jax.jit
    def _red_act(rp, obs_flat, key):
        logits = red_actor.apply(rp, obs_flat)
        keys = jax.random.split(key, n_red)
        return jax.vmap(jax.random.categorical)(keys, logits)

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env.reset(key)
    steps = []
    # Running union: cells blue has ever known at some previous step.
    # blue_ever[t] & ~blue_current[t] == fogged cells at step t (active red
    # manipulation — red wrote MAP_UNKNOWN into a blue receiver's local_map).
    blue_ever_union = None

    def _snapshot(st):
        nonlocal blue_ever_union
        local_maps_np = np.asarray(st.agent_state.local_map)
        team_ids_np = np.asarray(st.agent_state.team_ids)
        terrain_np = np.asarray(st.global_state.grid.terrain)
        non_wall = terrain_np != CELL_WALL
        blue_belief = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)
        red_belief = (_merge_team_belief(local_maps_np, team_ids_np, target_team=1)
                      if n_red > 0 else None)
        blue_cur = (blue_belief != MAP_UNKNOWN) & non_wall
        if blue_ever_union is None:
            blue_ever_union = blue_cur.copy()
        else:
            blue_ever_union |= blue_cur
        fogged_now = blue_ever_union & ~blue_cur
        return {
            "positions":  np.asarray(st.agent_state.positions).copy(),
            "team_ids":   team_ids_np.copy(),
            "terrain":    terrain_np.copy(),
            "blue_bel":   blue_belief,
            "red_bel":    red_belief,
            "blue_ever":  blue_ever_union.copy(),
            "fogged_now": fogged_now.copy(),
        }

    steps.append(_snapshot(state))

    for _ in range(1, max_steps + 1):
        key, *agent_keys = jax.random.split(key, n_total + 2)
        step_key = agent_keys[-1]
        action_dict = {}
        if red_actor is not None and n_red > 0:
            red_obs_flat = jnp.concatenate(
                [obs_dict[env.agents[n_blue + r]] for r in range(n_red)]
            )
            red_actions = _red_act(red_params, red_obs_flat, agent_keys[n_blue])
            for r in range(n_red):
                action_dict[env.agents[n_blue + r]] = int(red_actions[r])
        for i in range(n_blue):
            action_dict[env.agents[i]] = int(
                _blue_act(blue_params, obs_dict[env.agents[i]], agent_keys[i])
            )
        obs_dict, state, _rew, dones, _info = env.step_env(step_key, state, action_dict)
        steps.append(_snapshot(state))
        if bool(dones["__all__"]):
            while len(steps) < max_steps + 1:
                steps.append(steps[-1])
            break
    return steps


# --- isometric frame render

# Colour palette kept consistent with scripts/meta_report.py
_BLUE = "#1f77b4"
_RED = "#d62728"
_FADED = "#d0d0d0"

# Floor-cell RGBA used per view
_FLOOR_KNOWN_FREE = {  # cells the team knows are floor
    "global":   (0.93, 0.93, 0.93, 1.0),
    "blue":     (0.82, 0.88, 0.98, 1.0),
    "red":      (0.98, 0.87, 0.87, 1.0),
    "sabotage": (0.82, 0.88, 0.98, 1.0),  # blue-known floor identical to "blue" view
}
_FLOOR_KNOWN_WALL = (0.30, 0.30, 0.30, 1.0)  # seen-wall
_FLOOR_UNKNOWN = (0.22, 0.22, 0.22, 0.85)    # dark fog
_WALL_COLOUR = (0.10, 0.10, 0.10, 1.0)
# Sabotage view — 4 cell-types that jointly tell the sabotage story:
#   blue-only     : blue agents saw it (red never did)     → light blue floor
#   both-know     : blue AND red saw it — RED EFFORT WASTED (red duplicated
#                   work that blue was going to do anyway) → magenta floor
#   red-only      : red saw it but blue never did (hoarded) → bright orange column
#   unknown       : nobody has seen it                      → dark fog column
_FLOOR_BOTH_KNOW = (0.88, 0.55, 0.88, 1.0)   # magenta — red's redundant effort
_FLOOR_RED_HOARDED = (1.00, 0.55, 0.10, 1.0)  # orange — info red withheld
_HOARDED_HEIGHT = 0.38
# Uncertainty view: cells BLUE USED TO KNOW that red has actively fogged
# (writes MAP_UNKNOWN into blue receivers' local_maps each step in comm range).
# This is the proposal's "uncertainty manipulation" channel made visible —
# taller than orange hoarding columns so it dominates the eye.
_FLOOR_FOGGED_NOW = (0.95, 0.15, 0.15, 1.0)  # bright red
_FOGGED_HEIGHT = 0.55

# Column heights tuned so walls and unknown-fog don't visually hide the interior
# from the isometric camera at elev = 28°.
_UNKNOWN_HEIGHT_BLUE = 0.25
_UNKNOWN_HEIGHT_RED = 0.25
_UNKNOWN_HEIGHT_SABOTAGE = 0.22
_WALL_HEIGHT = 0.45
_AGENT_Z = 0.75
_AGENT_SIZE_FOCUS = 220
_AGENT_SIZE_FADED = 90


def _draw_iso(
    ax,
    snap,
    view: str,             # "global" | "blue" | "red"
    setup_label: str,
    step_idx: int,
    total_steps: int,
    coverage_pct: Optional[float],
    compact_title: bool = False,
):
    terrain = snap["terrain"]
    positions = snap["positions"]
    team_ids = snap["team_ids"]
    H, W = terrain.shape

    # Pick which belief drives the floor colouring / unknown fog.
    belief = None
    unk_height = 0.05
    if view == "blue":
        belief = snap["blue_bel"]
        unk_height = _UNKNOWN_HEIGHT_BLUE
    elif view == "red":
        belief = snap["red_bel"]
        unk_height = _UNKNOWN_HEIGHT_RED
    # "sabotage" view uses BOTH blue_bel and red_bel — handled inline below.

    blue_bel_sab = snap.get("blue_bel") if view == "sabotage" else None
    red_bel_sab = snap.get("red_bel") if view == "sabotage" else None

    # Build bar3d lists in one pass.
    xs, ys, zs = [], [], []
    dxs, dys, dzs = [], [], []
    cols = []

    for r in range(H):
        for c in range(W):
            is_wall = terrain[r, c] == CELL_WALL
            x0 = c - 0.5; y0 = (H - 1 - r) - 0.5  # flip y so the scene reads nicely in -60 azim
            if is_wall:
                xs.append(x0); ys.append(y0); zs.append(0.0)
                dxs.append(1.0); dys.append(1.0); dzs.append(_WALL_HEIGHT)
                cols.append(_WALL_COLOUR)
                continue

            if view == "global":
                xs.append(x0); ys.append(y0); zs.append(0.0)
                dxs.append(1.0); dys.append(1.0); dzs.append(0.05)
                cols.append(_FLOOR_KNOWN_FREE["global"])
            elif view == "uncertainty":
                # Fogged cells dominate: bright red raised column. Then the same
                # 4-color sabotage palette underneath.
                fogged = bool(snap["fogged_now"][r, c])
                if fogged:
                    xs.append(x0); ys.append(y0); zs.append(0.0)
                    dxs.append(1.0); dys.append(1.0); dzs.append(_FOGGED_HEIGHT)
                    cols.append(_FLOOR_FOGGED_NOW)
                else:
                    blue_knows = snap["blue_bel"][r, c] != MAP_UNKNOWN
                    red_knows = snap["red_bel"] is not None and snap["red_bel"][r, c] != MAP_UNKNOWN
                    xs.append(x0); ys.append(y0); zs.append(0.0)
                    dxs.append(1.0); dys.append(1.0)
                    if blue_knows and red_knows:
                        dzs.append(0.05); cols.append(_FLOOR_BOTH_KNOW)
                    elif blue_knows:
                        dzs.append(0.05); cols.append(_FLOOR_KNOWN_FREE["sabotage"])
                    elif red_knows:
                        dzs.append(_HOARDED_HEIGHT); cols.append(_FLOOR_RED_HOARDED)
                    else:
                        dzs.append(_UNKNOWN_HEIGHT_SABOTAGE); cols.append(_FLOOR_UNKNOWN)
            elif view == "sabotage":
                blue_knows = blue_bel_sab is not None and blue_bel_sab[r, c] != MAP_UNKNOWN
                red_knows = red_bel_sab is not None and red_bel_sab[r, c] != MAP_UNKNOWN
                xs.append(x0); ys.append(y0); zs.append(0.0)
                dxs.append(1.0); dys.append(1.0)
                if blue_knows and red_knows:
                    # BOTH saw it — red duplicated blue's work; red effort wasted.
                    dzs.append(0.05)
                    cols.append(_FLOOR_BOTH_KNOW)
                elif blue_knows:
                    # Blue alone — legitimate blue coverage.
                    dzs.append(0.05)
                    cols.append(_FLOOR_KNOWN_FREE["sabotage"])
                elif red_knows:
                    # RED alone saw it → raised orange hoarding column.
                    dzs.append(_HOARDED_HEIGHT)
                    cols.append(_FLOOR_RED_HOARDED)
                else:
                    # Nobody knows → dark fog.
                    dzs.append(_UNKNOWN_HEIGHT_SABOTAGE)
                    cols.append(_FLOOR_UNKNOWN)
            else:
                if belief is None:
                    # Red view on B-setup would hit this; we skip rendering it instead.
                    xs.append(x0); ys.append(y0); zs.append(0.0)
                    dxs.append(1.0); dys.append(1.0); dzs.append(0.05)
                    cols.append(_FLOOR_KNOWN_FREE["global"])
                elif belief[r, c] == MAP_UNKNOWN:
                    # Unknown → raised dark fog column
                    xs.append(x0); ys.append(y0); zs.append(0.0)
                    dxs.append(1.0); dys.append(1.0); dzs.append(unk_height)
                    cols.append(_FLOOR_UNKNOWN)
                else:
                    xs.append(x0); ys.append(y0); zs.append(0.0)
                    dxs.append(1.0); dys.append(1.0); dzs.append(0.05)
                    cols.append(_FLOOR_KNOWN_FREE[view])

    # Agent pins: each agent is a skinny coloured pillar (no scatter cap — the
    # circles matplotlib's 3D painter couldn't z-sort reliably around walls).
    pin_xs, pin_ys, pin_zs = [], [], []
    pin_dxs, pin_dys, pin_dzs = [], [], []
    pin_cols = []

    for i in range(positions.shape[0]):
        tid = int(team_ids[i])
        y_iso = (H - 1 - positions[i, 0])
        x_iso = positions[i, 1]
        if view == "blue":
            focus = (tid == 0); col = _BLUE if focus else _FADED
        elif view == "red":
            focus = (tid == 1); col = _RED if focus else _FADED
        elif view in ("sabotage", "uncertainty"):
            focus = True; col = _BLUE if tid == 0 else _RED
        else:
            focus = True; col = _BLUE if tid == 0 else _RED
        # Pillar (width 0.4 so it stays inside the cell).
        pin_xs.append(x_iso - 0.20); pin_ys.append(y_iso - 0.20)
        pin_zs.append(0.0)
        pin_dxs.append(0.40); pin_dys.append(0.40); pin_dzs.append(_AGENT_Z)
        pin_cols.append(col)

    # Single bar3d call bundling terrain + agents so matplotlib z-sorts them together.
    all_x = xs + pin_xs
    all_y = ys + pin_ys
    all_z = zs + pin_zs
    all_dx = dxs + pin_dxs
    all_dy = dys + pin_dys
    all_dz = dzs + pin_dzs
    all_cols = cols + pin_cols
    ax.bar3d(all_x, all_y, all_z, all_dx, all_dy, all_dz,
             color=all_cols, edgecolor=(0, 0, 0, 0.15),
             linewidth=0.25, shade=True)

    # Axes / camera
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_zlim(0, 1.1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((W, H, 5))
    ax.view_init(elev=32, azim=-55)
    # Transparent axis planes for cleanness
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0); pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)

    view_caption = {
        "global":   "Ground truth",
        "blue":     "Blue-team belief  (unknown = fog columns)",
        "red":      "Red-team belief   (unknown = fog columns)",
        "sabotage": ("Sabotage view  ·  blue = blue-only  ·  MAGENTA = both (red wasted)  ·  "
                     "ORANGE = red-only (hoarded)  ·  dark = never seen"),
        "uncertainty": ("Uncertainty manipulation  ·  RED COLUMNS = cells red just fogged "
                        "(blue used to know)  ·  blue floor = blue still holds  ·  "
                        "magenta/orange/dark as in sabotage view"),
    }[view]
    cov_txt = f"   cov = {coverage_pct:.1f}%" if coverage_pct is not None else ""
    if compact_title:
        # 3-up compare mode: per-axis title must be short or it collides with
        # the neighbours and the suptitle. View-caption is published in the
        # HTML legend below the GIF instead.
        ax.set_title(f"{setup_label}   ·   t = {step_idx}/{total_steps}{cov_txt}",
                     fontsize=11, pad=4)
    else:
        ax.set_title(f"{setup_label}   ·   {view_caption}\n"
                     f"t = {step_idx} / {total_steps}{cov_txt}",
                     fontsize=10)


def _fig_to_rgba(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def _coverage_curve(steps) -> List[float]:
    """Blue-team ever-known coverage (%) per snapshot — monotone non-decreasing."""
    curve = []
    for s in steps:
        terrain = s["terrain"]
        non_wall = terrain != CELL_WALL
        curve.append(100.0 * s["blue_ever"].sum() / max(1, non_wall.sum()))
    return curve


def _current_coverage_curve(steps) -> List[float]:
    """Blue-team currently-known coverage (%) per snapshot.

    Unlike the ever-known curve this is NOT monotone — red's fogging writes
    ``MAP_UNKNOWN`` back into blue receivers' local_maps each step, so the
    curve can DROP when blue loses cells it previously held. The ever vs
    currently gap is the volume of active uncertainty manipulation.
    """
    curve = []
    for s in steps:
        terrain = s["terrain"]
        non_wall = terrain != CELL_WALL
        cur = (s["blue_bel"] != MAP_UNKNOWN) & non_wall
        curve.append(100.0 * cur.sum() / max(1, non_wall.sum()))
    return curve


def _fogged_count_curve(steps) -> List[int]:
    """Number of cells actively fogged at step t (ever-known ∧ ¬currently-known)."""
    return [int(s["fogged_now"].sum()) for s in steps]


def _time_to_threshold(curve: List[float], theta: float) -> Optional[int]:
    """Return first step-index t where curve[t] >= theta; None if never reached."""
    for t, v in enumerate(curve):
        if v >= theta:
            return t
    return None


def _merged_coverage_curve(steps) -> List[float]:
    """Counterfactual coverage if red's info were merged into blue's.

    Union of (blue ever-known) and (red ever-known) — i.e. the ceiling blue
    could reach if red weren't hoarding its observations. For setups with no
    red agents this reduces to the plain blue coverage curve.
    """
    ever = None
    curve = []
    for s in steps:
        terrain = s["terrain"]
        non_wall = terrain != CELL_WALL
        blue_known = (s["blue_bel"] != MAP_UNKNOWN) & non_wall
        if s.get("red_bel") is not None:
            red_known = (s["red_bel"] != MAP_UNKNOWN) & non_wall
            known_now = blue_known | red_known
        else:
            known_now = blue_known
        ever = known_now if ever is None else (ever | known_now)
        curve.append(100.0 * ever.sum() / max(1, non_wall.sum()))
    return curve


def _render_gif_for_view(
    steps, subsample: int, fps: int, view: str,
    setup_label: str, coverage_curve: List[float], out_path: Path,
):
    frame_idxs = list(range(0, len(steps), subsample))
    if frame_idxs[-1] != len(steps) - 1:
        frame_idxs.append(len(steps) - 1)
    total_steps = len(steps) - 1

    frames: List[Image.Image] = []
    for i, fi in enumerate(frame_idxs):
        fig = plt.figure(figsize=(8.8, 7.4))
        ax = fig.add_subplot(111, projection="3d")
        _draw_iso(ax, steps[fi], view=view,
                  setup_label=setup_label,
                  step_idx=fi, total_steps=total_steps,
                  coverage_pct=coverage_curve[fi] if fi < len(coverage_curve) else None)
        fig.tight_layout()
        rgb = _fig_to_rgba(fig)
        plt.close(fig)
        frames.append(Image.fromarray(rgb))

    dur = max(1, 1000 // fps)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=dur, loop=0)


def _render_compare_gif(
    all_data: List[Tuple[Setup, list, List[float], List[float]]],
    subsample: int, fps: int, view: str, out_path: Path,
    current_curves: Optional[Dict[str, List[float]]] = None,
):
    """3-up synchronized iso + live coverage-vs-time curve.

    The compare GIF is the "money shot" for the degradation story: three
    panels side by side, plus a coverage curve showing the three lines
    diverging over time with a marker that advances with the animation.

    For the sabotage view we additionally plot the counterfactual "if red's
    info were merged" ceiling as a dashed line and shade the gap between
    actual blue coverage and that ceiling — the shaded area IS the sabotage.
    """
    n_total = max(len(d[1]) for d in all_data)  # len(steps) (== max_steps + 1)
    total_steps = n_total - 1
    frame_idxs = list(range(0, n_total, subsample))
    if frame_idxs[-1] != n_total - 1:
        frame_idxs.append(n_total - 1)

    curve_colors = ["#1a8a1a", "#ff8a1a", "#d62728"]  # B green, C1 amber, C2 red
    show_sabotage_gap = (view == "sabotage")
    show_fogging_gap = (view == "uncertainty") and current_curves is not None

    frames: List[Image.Image] = []
    for fi in frame_idxs:
        fig = plt.figure(figsize=(22.0, 14.6))
        gs = fig.add_gridspec(
            2, 3, height_ratios=[3.2, 1.8], hspace=0.14, wspace=0.01,
            left=0.03, right=0.99, top=0.945, bottom=0.055,
        )

        for i, (setup, steps, curve, merged) in enumerate(all_data):
            fi_local = min(fi, len(steps) - 1)
            ax = fig.add_subplot(gs[0, i], projection="3d")
            _draw_iso(
                ax, steps[fi_local], view=view,
                setup_label=setup.label,
                step_idx=fi_local, total_steps=len(steps) - 1,
                coverage_pct=curve[fi_local],
                compact_title=True,
            )

        ax_cov = fig.add_subplot(gs[1, :])
        for (setup, steps, curve, merged), col in zip(all_data, curve_colors):
            t = np.arange(len(curve))
            ax_cov.plot(t, curve, color=col, linewidth=2.2,
                        label=f"{setup.label} — blue ever-known", alpha=0.95)
            # Counterfactual merged ceiling only interesting for setups with red.
            has_sab_gap = show_sabotage_gap and not np.allclose(curve, merged, atol=0.05)
            if has_sab_gap:
                ax_cov.plot(t, merged, color=col, linewidth=1.3, linestyle="--",
                            alpha=0.75, label=f"{setup.label} — if merged")
                ax_cov.fill_between(t, curve, merged, color=col, alpha=0.15)
            # Uncertainty-manipulation view: plot "currently known" as dashed
            # same colour. The gap between ever (solid) and currently (dashed)
            # is the cell count red has actively fogged OUT of blue's belief.
            if show_fogging_gap:
                cur = current_curves.get(setup.key)
                if cur is not None:
                    ax_cov.plot(t, cur, color=col, linewidth=1.4, linestyle="--",
                                alpha=0.85, label=f"{setup.label} — currently known")
                    ax_cov.fill_between(t, cur, curve, color=col, alpha=0.22)
            fi_local = min(fi, len(curve) - 1)
            ax_cov.scatter([fi_local], [curve[fi_local]], color=col, s=75,
                           zorder=10, edgecolor="black", linewidth=1.0)
        ax_cov.axvline(fi, color="#888", linewidth=0.6, linestyle="--", alpha=0.7)
        # Mission-objective threshold lines + per-curve crossing markers.
        # 90% is the "mission objective" bar: B hits it, C2 never does.
        for theta, style, label in ((90, "-", "objective = 90%"),
                                    (80, "--", None),
                                    (70, ":", None)):
            ax_cov.axhline(theta, color="#555", linewidth=0.9, linestyle=style, alpha=0.55)
            if label is not None:
                ax_cov.text(total_steps * 0.01, theta + 1.2, label,
                            fontsize=8, color="#333", alpha=0.85)
        # For each curve, mark T(90) crossing (if any) with a vertical tick.
        for (setup, steps_, curve, _merged), col in zip(all_data, curve_colors):
            t90 = _time_to_threshold(curve, 90.0)
            if t90 is not None:
                ax_cov.plot([t90, t90], [0, 90], color=col, linewidth=1.0,
                            linestyle=(0, (2, 2)), alpha=0.75)
                ax_cov.scatter([t90], [90], color=col, s=40, marker="v",
                               edgecolor="black", linewidth=0.7, zorder=11)
        ax_cov.set_xlim(0, total_steps)
        ax_cov.set_ylim(0, 102)
        ax_cov.set_xlabel("episode step  t", fontsize=11)
        if show_sabotage_gap:
            cov_ylab = "coverage (%) — solid: blue actual · dashed: if red merged"
        elif show_fogging_gap:
            cov_ylab = "coverage (%) — solid: ever-known · dashed: currently known (red fogs gap)"
        else:
            cov_ylab = "blue-team coverage  (%)"
        ax_cov.set_ylabel(cov_ylab, fontsize=10.5)
        ax_cov.tick_params(axis="both", labelsize=10)
        ax_cov.grid(True, alpha=0.3)
        ax_cov.legend(loc="lower right", fontsize=9.5, framealpha=0.9, ncol=2)

        cov_B = all_data[0][2][min(fi, len(all_data[0][2]) - 1)]
        cov_C1 = all_data[1][2][min(fi, len(all_data[1][2]) - 1)]
        cov_C2 = all_data[2][2][min(fi, len(all_data[2][2]) - 1)]
        merged_C1 = all_data[1][3][min(fi, len(all_data[1][3]) - 1)]
        merged_C2 = all_data[2][3][min(fi, len(all_data[2][3]) - 1)]
        sab_C1 = merged_C1 - cov_C1
        sab_C2 = merged_C2 - cov_C2
        if show_sabotage_gap:
            title = (
                f"RED SABOTAGE of blue mission — t = {fi} / {total_steps}    "
                f"·    B = {cov_B:.1f}%    "
                f"·    C1: blue = {cov_C1:.1f}% vs merged = {merged_C1:.1f}%  "
                f"(red hoards {sab_C1:+.1f} pp)    "
                f"·    C2: blue = {cov_C2:.1f}% vs merged = {merged_C2:.1f}%  "
                f"(red hoards {sab_C2:+.1f} pp)"
            )
        elif show_fogging_gap:
            def _fog_frac(key: str) -> float:
                cur = (current_curves.get(key) or [0.0])[min(fi, len(current_curves.get(key) or [0]) - 1)]
                ever = all_data[["B", "C1", "C2"].index(key)][2][min(fi, total_steps)]
                return ever - cur
            fog_B = _fog_frac("B");  fog_C1 = _fog_frac("C1");  fog_C2 = _fog_frac("C2")
            title = (
                f"UNCERTAINTY MANIPULATION — red fogs blue's belief — t = {fi} / {total_steps}    "
                f"·    B: fogged {fog_B:.1f} pp    "
                f"·    C1: fogged {fog_C1:.1f} pp    "
                f"·    C2: fogged {fog_C2:.1f} pp"
            )
        else:
            title = (
                f"Mission degradation — blue coverage at  t = {fi} / {total_steps}    "
                f"·    B = {cov_B:.1f}%    "
                f"·    C1 = {cov_C1:.1f}%  (Δ = {cov_B - cov_C1:+.1f} pp)    "
                f"·    C2 = {cov_C2:.1f}%  (Δ = {cov_B - cov_C2:+.1f} pp)"
            )
        fig.suptitle(title, fontsize=12.0, y=0.965, weight="bold")

        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        frames.append(Image.fromarray(rgb))

    dur = max(1, 1000 // fps)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=dur, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=5,
                    help="Keep every Nth timestep as a GIF frame (default 5)")
    ap.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "isometric.log", "w")

    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, x):
            for st in self.s: st.write(x); st.flush()
        def flush(self):
            for st in self.s: st.flush()
    sys.stdout = _Tee(sys.__stdout__, log)

    print(f"Isometric-view renderer: subsample={args.subsample} fps={args.fps} "
          f"max_steps={args.max_steps}")
    t0 = time.time()

    all_data: List[Tuple[Setup, list, List[float], List[float]]] = []
    current_curves: Dict[str, List[float]] = {}

    for setup in SETUPS:
        print(f"\n=== {setup.key}: {setup.label} ===")
        cfg = ExperimentConfig.from_yaml(setup.config)
        blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
        red_actor = red_params = None
        if setup.red_ckpt is not None:
            red_actor, red_params = _load_red(cfg, setup.red_ckpt)

        t_r = time.time()
        steps = _rollout_capture(cfg, blue_actor, blue_params,
                                 red_actor, red_params,
                                 seed=CANON_SEED, max_steps=args.max_steps)
        curve = _coverage_curve(steps)
        merged = _merged_coverage_curve(steps)
        cur_curve = _current_coverage_curve(steps)
        fogged_counts = _fogged_count_curve(steps)
        current_curves[setup.key] = cur_curve
        hoard = merged[-1] - curve[-1]
        # End-of-episode 4-way cell breakdown to confirm the sabotage story.
        last = steps[-1]
        non_wall = last["terrain"] != CELL_WALL
        blue_k = (last["blue_bel"] != MAP_UNKNOWN) & non_wall
        if last.get("red_bel") is not None:
            red_k = (last["red_bel"] != MAP_UNKNOWN) & non_wall
            both = blue_k & red_k
            blue_only = blue_k & ~red_k
            red_only = red_k & ~blue_k
            nobody = ~(blue_k | red_k) & non_wall
            denom = max(1, non_wall.sum())
            print(
                f"  rollout {len(steps)-1} steps, final blue={curve[-1]:.1f}%  "
                f"merged={merged[-1]:.1f}%  ({time.time() - t_r:.1f}s)\n"
                f"    cell breakdown at t=200:  "
                f"blue-only {100.0*blue_only.sum()/denom:.1f}%   "
                f"BOTH-KNOW {100.0*both.sum()/denom:.1f}% (red effort duplicating blue)   "
                f"red-only {100.0*red_only.sum()/denom:.1f}% (red hoards)   "
                f"nobody {100.0*nobody.sum()/denom:.1f}% (sabotage: never seen)"
            )
        else:
            print(f"  rollout {len(steps)-1} steps, final blue={curve[-1]:.1f}%  "
                  f"(baseline — no red)  ({time.time() - t_r:.1f}s)")
        fog_peak = max(fogged_counts)
        fog_peak_t = fogged_counts.index(fog_peak)
        fog_final = fogged_counts[-1]
        fog_gap_final = curve[-1] - cur_curve[-1]
        print(
            f"    fogging: peak {fog_peak} cells fogged at t={fog_peak_t}  "
            f"·  final {fog_final} cells fogged  "
            f"·  ever-vs-currently gap at t=200 = {fog_gap_final:.1f} pp "
            f"(ever {curve[-1]:.1f}% vs currently {cur_curve[-1]:.1f}%)"
        )
        all_data.append((setup, steps, curve, merged))

        views = ["global", "blue"]
        if cfg.env.num_red_agents > 0:
            views += ["red", "sabotage", "uncertainty"]
        for view in views:
            t_v = time.time()
            out_path = out_dir / f"iso_{setup.key}_{view}.gif"
            _render_gif_for_view(
                steps, args.subsample, args.fps, view,
                setup_label=setup.label, coverage_curve=curve, out_path=out_path,
            )
            print(f"  wrote {out_path}  ({time.time() - t_v:.1f}s)")

    print("\n=== Time-to-threshold (canonical seed 0, blue-team coverage) ===")
    thresholds = [70.0, 80.0, 90.0, 95.0]
    t_thresh_dump: Dict[str, Dict[str, Optional[object]]] = {}
    for setup, steps, curve, _merged in all_data:
        t_thresh_dump[setup.key] = {}
        parts = []
        for theta in thresholds:
            t_cross = _time_to_threshold(curve, theta)
            t_thresh_dump[setup.key][f"T_{int(theta)}"] = t_cross
            parts.append(f"T({int(theta)}%) = "
                         + (f"{t_cross}" if t_cross is not None
                            else "NEVER CROSSED"))
        final = curve[-1]
        t_thresh_dump[setup.key]["final_coverage"] = float(final)
        t_thresh_dump[setup.key]["n_steps"] = len(curve) - 1
        cur = current_curves.get(setup.key) or [0.0]
        t_thresh_dump[setup.key]["final_currently_known"] = float(cur[-1])
        t_thresh_dump[setup.key]["final_fogged_gap_pp"] = float(curve[-1] - cur[-1])
        print(f"  {setup.key}: final={final:.1f}%   "
              + "   ".join(parts))
    (out_dir / "iso_time_to_threshold.json").write_text(
        json.dumps(t_thresh_dump, indent=2)
    )
    print(f"  wrote {out_dir / 'iso_time_to_threshold.json'}")

    print("\n=== 3-up synchronized comparison (mission degradation + sabotage + uncertainty) ===")
    for view in ("sabotage", "uncertainty", "blue", "global"):
        t_c = time.time()
        out_path = out_dir / f"iso_compare_{view}.gif"
        _render_compare_gif(
            all_data, args.subsample, args.fps, view, out_path,
            current_curves=current_curves if view == "uncertainty" else None,
        )
        print(f"  wrote {out_path}  ({time.time() - t_c:.1f}s)")

    print(f"\nTotal render time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
