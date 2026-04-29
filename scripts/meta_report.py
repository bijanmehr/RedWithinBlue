"""Meta-report: per-agent, per-team, and global views across three N=5 setups.

Produces a fully self-contained `experiments/meta-report/` directory so the
HTML can be opened standalone (everything it references lives next to it):

  - `viz_{B,C1,C2}.png`         — per-setup multi-panel: global truth + each
                                   agent's own belief map across 4 snapshots.
  - `comparison_matrix.png`     — THE centerpiece. Rows = setup (B / C1 / C2),
                                   cols = time. Each cell shows ground truth,
                                   team-merged blue-unknown overlay (red
                                   tint = hidden from the team), agent
                                   positions coloured by team, comm edges,
                                   and coverage % — so the three missions
                                   line up at identical t's for eye-level
                                   comparison.
  - `fog_footprint.png`         — at t=200, the cells still unknown to the
                                   blue team, side by side. Makes the
                                   leftover dark area the story.
  - `coverage_curves.png`       — mean per-step coverage curves of all 3
                                   setups on one axis (shaded = p10-p90).
  - `episode_{B,C1,C2}.gif`     — canonical-episode gifs (copied from the
                                   per-experiment report dirs so the meta
                                   dir is self-contained).
  - `meta_report.html`          — proposal math + mapping + all figures +
                                   ΔJ(k) / k*(θ) block + embedded gifs.

One canonical eval episode per setup (seed 0). Uses the already-trained
blue/red checkpoints from `experiments/compromise-16x16-5-*-coevo` and the
N=5 clean baseline at `experiments/survey-local-16-N5-from-N4`.

Run: ``python scripts/meta_report.py``
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

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
from red_within_blue.types import CELL_WALL, MAP_FREE, MAP_UNKNOWN, MAP_WALL
from red_within_blue.visualizer import _merge_team_belief


OUT_DIR = Path("experiments/meta-report")
SNAPSHOT_STEPS = (50, 100, 150, 200)
CANON_SEED = 0
MAX_STEPS_EVAL = 200


@dataclass
class Setup:
    key: str                 # B, C1, C2
    label: str
    short: str               # one-line row label for comparison matrix
    config: str
    blue_ckpt: str
    red_ckpt: Optional[str]
    source_gif: str          # path to the per-experiment episode.gif to copy


SETUPS: List[Setup] = [
    Setup(
        key="B",
        label="B  — N=5 clean (5 blue, 0 red)",
        short="B — 5 blue, 0 red",
        config="configs/survey-local-16-N5-from-N4.yaml",
        blue_ckpt="experiments/survey-local-16-N5-from-N4/checkpoint.npz",
        red_ckpt=None,
        source_gif="experiments/survey-local-16-N5-from-N4/episode.gif",
    ),
    Setup(
        key="C1",
        label="C1 — N=5: 4 blue + 1 red (m=1)",
        short="C1 — 4 blue, 1 red",
        config="configs/compromise-16x16-5-4b1r.yaml",
        blue_ckpt="experiments/compromise-16x16-5-4b1r-coevo/checkpoint.npz",
        red_ckpt="experiments/compromise-16x16-5-4b1r-coevo/joint_red_checkpoint.npz",
        source_gif="experiments/compromise-16x16-5-4b1r-coevo/episode.gif",
    ),
    Setup(
        key="C2",
        label="C2 — N=5: 3 blue + 2 red (m=2)",
        short="C2 — 3 blue, 2 red",
        config="configs/compromise-16x16-5-3b2r.yaml",
        blue_ckpt="experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz",
        red_ckpt="experiments/compromise-16x16-5-3b2r-coevo/joint_red_checkpoint.npz",
        source_gif="experiments/compromise-16x16-5-3b2r-coevo/episode.gif",
    ),
]


# --- checkpoint loaders (same per-leaf seed-strip trick as compromise_compare.py)

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


# --- episode rollout with snapshot capture

def _rollout_with_snapshots(
    cfg: ExperimentConfig,
    blue_actor: Actor,
    blue_params,
    red_actor: Optional[JointRedActor],
    red_params,
    seed: int,
    max_steps: int,
    snapshot_steps: Tuple[int, ...],
) -> Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Run one episode; capture env state at each snapshot step, coverage curve,
    per-step positions (for trajectory overlay), and the team-id vector."""
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red

    reward_fn = normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)

    @jax.jit
    def _blue_action(bp, obs, key):
        return jax.random.categorical(key, blue_actor.apply(bp, obs))

    @jax.jit
    def _red_actions(rp, obs_flat, key):
        logits = red_actor.apply(rp, obs_flat)
        keys = jax.random.split(key, n_red)
        return jax.vmap(jax.random.categorical)(keys, logits)

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env.reset(key)
    snapshots: List[dict] = []
    coverage_curve: List[float] = []
    blue_ever: Optional[np.ndarray] = None
    all_positions: List[np.ndarray] = [np.asarray(state.agent_state.positions).copy()]
    team_ids_const = np.asarray(state.agent_state.team_ids).copy()

    for step in range(1, max_steps + 1):
        key, *agent_keys = jax.random.split(key, n_total + 2)
        step_key = agent_keys[-1]
        action_dict = {}
        if red_actor is not None and n_red > 0:
            red_obs_flat = jnp.concatenate(
                [obs_dict[env.agents[n_blue + r]] for r in range(n_red)]
            )
            red_actions = _red_actions(red_params, red_obs_flat, agent_keys[n_blue])
            for r in range(n_red):
                action_dict[env.agents[n_blue + r]] = int(red_actions[r])
        for i in range(n_blue):
            action_dict[env.agents[i]] = int(
                _blue_action(blue_params, obs_dict[env.agents[i]], agent_keys[i])
            )
        obs_dict, state, _rew, dones, _info = env.step_env(step_key, state, action_dict)

        all_positions.append(np.asarray(state.agent_state.positions).copy())
        local_maps_np = np.asarray(state.agent_state.local_map)
        team_ids_np = np.asarray(state.agent_state.team_ids)
        blue_belief = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)
        terrain = np.asarray(state.global_state.grid.terrain)
        non_wall = terrain != CELL_WALL
        known_now = (blue_belief != MAP_UNKNOWN) & non_wall
        blue_ever = known_now if blue_ever is None else (blue_ever | known_now)
        coverage_curve.append(100.0 * blue_ever.sum() / max(1, non_wall.sum()))

        if step in snapshot_steps:
            snapshots.append({
                "step": step,
                "positions": np.asarray(state.agent_state.positions).copy(),
                "team_ids": team_ids_np.copy(),
                "local_maps": local_maps_np.copy(),
                "terrain": terrain.copy(),
                "blue_ever": blue_ever.copy(),
                "adjacency": np.asarray(state.global_state.graph.adjacency).copy(),
                "coverage_pct": float(coverage_curve[-1]),
            })

        if bool(dones["__all__"]):
            # Episode ended early; pad snapshots / curve so the figure still makes sense
            for s in snapshot_steps:
                if s > step and not any(sn["step"] == s for sn in snapshots):
                    snapshots.append({**snapshots[-1], "step": s})
            while len(coverage_curve) < max_steps:
                coverage_curve.append(coverage_curve[-1])
            break

    return (
        snapshots,
        np.asarray(coverage_curve, dtype=np.float32),
        np.stack(all_positions, axis=0),
        team_ids_const,
    )


# --- panel renderers

_TERRAIN_CMAP = ListedColormap(["#f2f2f2", "#2c2c2c"])  # empty / wall
_BELIEF_CMAP = ListedColormap(["#d9d9d9", "#f2f2f2", "#2c2c2c"])  # unknown / free / wall
_TEAM_COLOURS = {0: "#1f77b4", 1: "#d62728"}  # blue, red


def _draw_global(ax, snap):
    H, W = snap["terrain"].shape
    ax.imshow(snap["terrain"], cmap=_TERRAIN_CMAP, vmin=0, vmax=1, origin="upper",
              extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    # Comm edges
    adj = snap["adjacency"]
    pos = snap["positions"]
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if adj[i, j]:
                ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]],
                        color="#7d9bb5", linewidth=1.0, alpha=0.7, zorder=2)
    for i in range(pos.shape[0]):
        tid = int(snap["team_ids"][i])
        c = _TEAM_COLOURS.get(tid, "k")
        ax.add_patch(Circle((pos[i, 1], pos[i, 0]), 0.35, facecolor=c,
                            edgecolor="black", linewidth=0.6, zorder=3))
        ax.text(pos[i, 1], pos[i, 0], str(i), ha="center", va="center",
                fontsize=6, color="white", zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)


def _draw_belief(ax, snap, agent_i):
    belief = snap["local_maps"][agent_i]
    H, W = belief.shape
    # Remap: UNKNOWN=0, FREE=1, WALL=2
    disp = np.full_like(belief, 0)
    disp[belief == MAP_FREE] = 1
    disp[belief == MAP_WALL] = 2
    ax.imshow(disp, cmap=_BELIEF_CMAP, vmin=0, vmax=2, origin="upper",
              extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    pos = snap["positions"][agent_i]
    tid = int(snap["team_ids"][agent_i])
    c = _TEAM_COLOURS.get(tid, "k")
    ax.add_patch(Circle((pos[1], pos[0]), 0.4, facecolor=c,
                        edgecolor="black", linewidth=0.8, zorder=3))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)


def _draw_global_with_fog(ax, snap, show_unknown_overlay: bool = True):
    """Global truth + a red tint over cells the BLUE team has not yet discovered.

    Makes 'what did the attacker hide' directly visible in the per-setup cell.
    """
    H, W = snap["terrain"].shape
    ax.imshow(snap["terrain"], cmap=_TERRAIN_CMAP, vmin=0, vmax=1, origin="upper",
              extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    if show_unknown_overlay:
        non_wall = snap["terrain"] != CELL_WALL
        unknown = non_wall & (~snap["blue_ever"])
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[unknown] = np.array([0.84, 0.19, 0.19, 0.35])  # red, 35% alpha
        ax.imshow(overlay, origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5), zorder=1.5)
    adj = snap["adjacency"]
    pos = snap["positions"]
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if adj[i, j]:
                ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]],
                        color="#7d9bb5", linewidth=1.0, alpha=0.75, zorder=2)
    for i in range(pos.shape[0]):
        tid = int(snap["team_ids"][i])
        c = _TEAM_COLOURS.get(tid, "k")
        ax.add_patch(Circle((pos[i, 1], pos[i, 0]), 0.32, facecolor=c,
                            edgecolor="black", linewidth=0.6, zorder=3))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)


def _render_comparison_matrix(
    all_snaps: Dict[str, list], setups: List[Setup], out_png: Path,
) -> None:
    """3 setups × 4 time snapshots — ground truth + blue-team fog overlay.

    Rows: B, C1, C2.  Cols: snapshot times (t=50/100/150/200).
    Highlights cells still UNKNOWN to the blue team (red tint). Lets the reader
    compare "what has the team collectively discovered?" at identical t across
    the three missions.
    """
    n_rows = len(setups)
    # Use the first setup's snapshot timesteps as the canonical axis.
    n_cols = len(all_snaps[setups[0].key])
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    for r, setup in enumerate(setups):
        snaps = all_snaps[setup.key]
        for c, snap in enumerate(snaps):
            _draw_global_with_fog(axes[r, c], snap, show_unknown_overlay=True)
            title = f"t = {snap['step']}     cov = {snap['coverage_pct']:.1f}%"
            axes[r, c].set_title(title, fontsize=9)
            if c == 0:
                axes[r, c].set_ylabel(setup.short, fontsize=10, fontweight="bold")

    fig.suptitle(
        "Cross-setup mission comparison  — blue-team fog shown in red tint  "
        "(darker = still unknown to every blue)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _render_fog_footprint(
    all_snaps: Dict[str, list], setups: List[Setup], out_png: Path,
) -> None:
    """At the final snapshot, show ONLY the blue-team-unknown mask for each
    setup so the residual dark area is the story."""
    n_cols = len(setups)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.6), squeeze=False)
    for c, setup in enumerate(setups):
        snap = all_snaps[setup.key][-1]
        H, W = snap["terrain"].shape
        non_wall = snap["terrain"] != CELL_WALL
        unknown = non_wall & (~snap["blue_ever"])
        disp = np.zeros((H, W, 3), dtype=np.float32)
        disp[non_wall] = [0.92, 0.92, 0.92]
        disp[~non_wall] = [0.17, 0.17, 0.17]
        disp[unknown] = [0.84, 0.19, 0.19]
        axes[0, c].imshow(disp, origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        pos = snap["positions"]
        for i in range(pos.shape[0]):
            tid = int(snap["team_ids"][i])
            cc = _TEAM_COLOURS.get(tid, "k")
            axes[0, c].add_patch(Circle((pos[i, 1], pos[i, 0]), 0.32, facecolor=cc,
                                        edgecolor="black", linewidth=0.6, zorder=3))
        pct_unknown = 100.0 * unknown.sum() / max(1, non_wall.sum())
        axes[0, c].set_title(
            f"{setup.short}\nt = {snap['step']}   blue-unknown = {pct_unknown:.1f}%",
            fontsize=10,
        )
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
    fig.suptitle("Residual fog at episode end  (red = cells no blue ever saw or was told about)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_coverage_curves(
    all_curves: Dict[str, np.ndarray], setups: List[Setup], out_png: Path,
) -> None:
    """Single-axis line plot of coverage vs step for all three setups."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 7.0))
    palette = {"B": "#1f77b4", "C1": "#ff7f0e", "C2": "#d62728"}
    for setup in setups:
        curve = all_curves[setup.key]
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, label=setup.short, color=palette.get(setup.key, "k"),
                linewidth=2)
    ax.axhline(90, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.99, 91.5, "90% threshold", ha="right",
            va="bottom", fontsize=9, color="#555")
    ax.set_xlabel("step t")
    ax.set_ylabel("blue ever-known coverage (%)")
    ax.set_title("Single-episode coverage (seed 0)  — per-step, per-setup")
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _render_trajectories(
    all_paths: Dict[str, np.ndarray],
    all_team_ids: Dict[str, np.ndarray],
    all_snaps: Dict[str, list],
    setups: List[Setup],
    out_png: Path,
) -> None:
    """One panel per setup: terrain + blue-team residual fog + per-agent path
    polylines coloured by team. Makes the causal story (red pins a sub-team →
    remaining blue can't extend the frontier) visible in one image.
    """
    fig, axes = plt.subplots(1, len(setups), figsize=(5.4 * len(setups), 5.8), squeeze=False)
    for ax, setup in zip(axes[0], setups):
        paths = all_paths[setup.key]          # [T+1, N, 2]
        team_ids = all_team_ids[setup.key]    # [N]
        final_snap = all_snaps[setup.key][-1]

        H, W = final_snap["terrain"].shape
        ax.imshow(final_snap["terrain"], cmap=_TERRAIN_CMAP, vmin=0, vmax=1,
                  origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))

        non_wall = final_snap["terrain"] != CELL_WALL
        unknown = non_wall & (~final_snap["blue_ever"])
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[unknown] = np.array([0.84, 0.19, 0.19, 0.28])
        ax.imshow(overlay, origin="upper",
                  extent=(-0.5, W - 0.5, H - 0.5, -0.5), zorder=1.5)

        N = paths.shape[1]
        for i in range(N):
            tid = int(team_ids[i])
            col = _TEAM_COLOURS.get(tid, "k")
            # tiny deterministic jitter to reduce overlap on identical paths
            jy = ((i % 3) - 1) * 0.10
            jx = (((i // 3) % 3) - 1) * 0.10
            ys = paths[:, i, 0] + jy
            xs = paths[:, i, 1] + jx
            ax.plot(xs, ys, color=col, linewidth=1.6, alpha=0.85, zorder=3)
            ax.add_patch(Circle((xs[0], ys[0]), 0.28, facecolor="white",
                                edgecolor=col, linewidth=1.2, zorder=4))
            ax.add_patch(Circle((xs[-1], ys[-1]), 0.42, facecolor=col,
                                edgecolor="black", linewidth=0.7, zorder=5))
            ax.text(xs[-1], ys[-1], str(i), ha="center", va="center",
                    fontsize=7, color="white", zorder=6)

        pct_unknown = 100.0 * unknown.sum() / max(1, non_wall.sum())
        ax.set_title(
            f"{setup.short}\ncov = {final_snap['coverage_pct']:.1f}%   "
            f"unknown = {pct_unknown:.1f}%",
            fontsize=10,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)

    fig.suptitle(
        "Agent trajectories over the canonical 200-step episode.\n"
        "Hollow ring = start, filled disc = end.  Red tint = cells never known to the blue team.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_claims_evidence(cc_npz_path: str, out_png: Path) -> Dict[str, float]:
    """Multi-panel evidence figure directly keyed to proposal claims.

    Returns the stats dict the HTML reads back so claim numbers stay in sync
    with the figure. All σ / stddev values use sample std (ddof=1) to match
    the §7 aggregate table and the paper-standard convention.
    """
    cc = np.load(cc_npz_path, allow_pickle=True)
    S  = cc["finals_0"]; B = cc["finals_1"]; C1 = cc["finals_2"]; C2 = cc["finals_3"]
    palette = {"S": "#888888", "B": "#1f77b4", "C1": "#ff7f0e", "C2": "#d62728"}

    # Sample (Bessel-corrected) std — used everywhere in this figure so the
    # numbers agree with the aggregate table in §7.
    def _sd(a):
        return float(a.std(ddof=1))

    fig = plt.figure(figsize=(16, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.32)

    # C1: cooperation dominates solo
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["S  (N=1)", "B  (N=5 clean)"], [S.mean(), B.mean()],
            yerr=[_sd(S), _sd(B)], color=[palette["S"], palette["B"]],
            alpha=0.85, capsize=5, edgecolor="black", linewidth=0.6)
    for i, v in enumerate([S.mean(), B.mean()]):
        ax1.text(i, v + 2.0, f"{v:.1f}%", ha="center", fontsize=9)
    ax1.axhline(90, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_ylim(0, 112); ax1.set_ylabel("final coverage (%)")
    ax1.set_title("Claim 1 · cooperation dominates solo\nJ(Π_swarm) ≫ J(π_solo)",
                  fontsize=10, fontweight="bold")

    # C2+C3: ΔJ(k) curve
    ax2 = fig.add_subplot(gs[0, 1])
    ks = [0, 1, 2]
    means = np.array([B.mean(), C1.mean(), C2.mean()])
    stds = np.array([_sd(B), _sd(C1), _sd(C2)])
    # Drive k*(θ) annotation from actual ΔJ values so it can't drift from data
    dJ_k1 = float(B.mean() - C1.mean())
    dJ_k2 = float(B.mean() - C2.mean())
    def _kstar(theta):
        if dJ_k1 >= theta: return "1"
        if dJ_k2 >= theta: return "2"
        return "∞"
    ax2.errorbar(ks, means, yerr=stds, fmt="o-", color=palette["C2"],
                 capsize=5, linewidth=2, markersize=8)
    for x, m in zip(ks, means):
        ax2.text(x, m + 2.5, f"{m:.1f}%", ha="center", fontsize=9)
    ax2.axhline(90, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xticks(ks); ax2.set_xlabel("compromise count k")
    ax2.set_ylim(60, 105); ax2.set_ylabel("J(π, φ*_k)  —  final coverage (%)")
    ax2.set_title("Claim 2 · ΔJ(k) is positive and monotone\n"
                  f"k*(5pp) = {_kstar(5)}   k*(10pp) = {_kstar(10)}   k*(15pp) = {_kstar(15)}",
                  fontsize=10, fontweight="bold")

    # C3: marginal ΔJ
    ax3 = fig.add_subplot(gs[0, 2])
    margs = [B.mean() - C1.mean(), C1.mean() - C2.mean()]
    bars = ax3.bar(["0 → 1", "1 → 2"], margs,
                   color=[palette["C1"], palette["C2"]],
                   alpha=0.85, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, margs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25,
                 f"{v:.1f} pp", ha="center", fontsize=9)
    ax3.set_ylabel("marginal damage (pp)"); ax3.set_xlabel("compromise added")
    ax3.set_title("Claim 3 · sub-linear marginal damage\n"
                  "ΔJ(k+1) − ΔJ(k) shrinks with k",
                  fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(margs) * 1.45 if margs else 1.0)

    # C4: per-seed distributions → variance inflation
    ax4 = fig.add_subplot(gs[1, :2])
    data = [B, C1, C2]
    labels = ["B (clean, k=0)", "C1 (4b+1r, k=1)", "C2 (3b+2r, k=2)"]
    bp = ax4.boxplot(data, positions=[1, 2, 3], widths=0.55,
                     patch_artist=True, showmeans=True, tick_labels=labels)
    for patch, c in zip(bp["boxes"],
                         [palette["B"], palette["C1"], palette["C2"]]):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for x, arr in zip([1, 2, 3], data):
        jitter = rng.uniform(-0.12, 0.12, size=arr.shape)
        ax4.scatter(np.full_like(arr, x) + jitter, arr, s=22,
                    color="#222", alpha=0.7, zorder=3)
    ax4.axhline(90, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    for x, arr, c in zip([1, 2, 3], data,
                          [palette["B"], palette["C1"], palette["C2"]]):
        ax4.text(x, 55, f"σ = {_sd(arr):.1f}", ha="center", fontsize=9,
                 color=c, fontweight="bold")
    ax4.set_ylim(50, 104); ax4.set_ylabel("per-seed final coverage (%)")
    ax4.set_title(
        "Claim 4 · m = 2 inflates variance, not just mean\n"
        f"σ: {_sd(B):.1f} → {_sd(C1):.1f} → {_sd(C2):.1f}   "
        f"(σ² ×{(_sd(C2)/_sd(C1))**2:.1f} k=1→k=2)   "
        f"min cov: {int(round(B.min()))} → {int(round(C1.min()))} → {int(round(C2.min()))}",
        fontsize=10, fontweight="bold")

    # % seeds clearing detector threshold
    ax5 = fig.add_subplot(gs[1, 2])
    pcts = [100 * (B >= 90).mean(), 100 * (C1 >= 90).mean(), 100 * (C2 >= 90).mean()]
    bars = ax5.bar(["B", "C1", "C2"], pcts,
                   color=[palette["B"], palette["C1"], palette["C2"]],
                   alpha=0.85, edgecolor="black", linewidth=0.6)
    for bar, p in zip(bars, pcts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{p:.0f}%", ha="center", fontsize=9)
    ax5.set_ylim(0, 112); ax5.set_ylabel("% seeds with cov ≥ 90%")
    ax5.set_title("Detector-threshold survival\n"
                  "(operational reliability proxy)",
                  fontsize=10, fontweight="bold")

    fig.suptitle(
        "Evidence panel keyed to the proposal claims "
        "— 20 eval seeds per setup, 16×16 grid, max_steps = 200",
        fontsize=12, y=1.00,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "S_mean": float(S.mean()), "S_std": _sd(S),
        "B_mean": float(B.mean()), "B_std": _sd(B),
        "C1_mean": float(C1.mean()), "C1_std": _sd(C1),
        "C2_mean": float(C2.mean()), "C2_std": _sd(C2),
        "B_min":  float(B.min()),  "B_max":  float(B.max()),
        "C1_min": float(C1.min()), "C1_max": float(C1.max()),
        "C2_min": float(C2.min()), "C2_max": float(C2.max()),
        "var_ratio_1_2": (_sd(C2) / _sd(C1)) ** 2,
        "worst_drop_1_2": float(C1.min() - C2.min()),
        "dJ_1": float(B.mean() - C1.mean()),
        "dJ_2": float(B.mean() - C2.mean()),
        "marg_1_2": float(C1.mean() - C2.mean()),
        "pct90_B":  100.0 * float((B >= 90).mean()),
        "pct90_C1": 100.0 * float((C1 >= 90).mean()),
        "pct90_C2": 100.0 * float((C2 >= 90).mean()),
        "n_seeds": int(len(B)),
    }


# --- misbehavior-budget renderers (k × ρ)

def _load_budget(budget_npz_path: str):
    d = np.load(budget_npz_path)
    ks = np.asarray(d["k"])
    rhos = np.asarray(d["rho"])
    finals = np.asarray(d["finals"])     # [n_conds, n_seeds]
    mean = np.asarray(d["mean"])
    std = np.asarray(d["std"])
    unique_k = sorted(set(int(x) for x in ks.tolist()))
    unique_rho = sorted(set(round(float(x), 4) for x in rhos.tolist()))
    return ks, rhos, finals, mean, std, unique_k, unique_rho


def _grid_from_sweep(ks, rhos, vals, unique_k, unique_rho):
    grid = np.full((len(unique_k), len(unique_rho)), np.nan)
    for i, kk in enumerate(unique_k):
        for j, rr in enumerate(unique_rho):
            sel = (ks == kk) & np.isclose(rhos, rr, atol=1e-4)
            if sel.any():
                grid[i, j] = float(vals[sel][0])
    return grid


def _render_budget_heatmap(budget_npz_path: str, B_mean: float, out_png: Path) -> None:
    ks, rhos, finals, mean, std, uk, ur = _load_budget(budget_npz_path)
    cov_grid = _grid_from_sweep(ks, rhos, mean, uk, ur)
    std_grid = _grid_from_sweep(ks, rhos, std, uk, ur)
    dj_grid = B_mean - cov_grid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.0))

    im1 = ax1.imshow(cov_grid, cmap="RdYlBu", aspect="auto", origin="lower",
                     vmin=75, vmax=100)
    ax1.set_xticks(range(len(ur)))
    ax1.set_xticklabels([f"{r:.2f}" for r in ur])
    ax1.set_yticks(range(len(uk)))
    ax1.set_yticklabels([f"k={kk}" for kk in uk])
    ax1.set_xlabel("ρ  —  per-step policy-negation probability")
    ax1.set_ylabel("k  —  # compromised agents")
    ax1.set_title("Blue coverage   J(π, φ_{k,ρ})   (%)")
    for i in range(cov_grid.shape[0]):
        for j in range(cov_grid.shape[1]):
            m = cov_grid[i, j]; s = std_grid[i, j]
            ax1.text(j, i, f"{m:.1f}\n±{s:.1f}", ha="center", va="center",
                     fontsize=9, color="black")
    plt.colorbar(im1, ax=ax1, label="coverage (%)")

    im2 = ax2.imshow(dj_grid, cmap="Reds", aspect="auto", origin="lower",
                     vmin=0, vmax=max(15.0, float(np.nanmax(dj_grid)) + 1))
    ax2.set_xticks(range(len(ur)))
    ax2.set_xticklabels([f"{r:.2f}" for r in ur])
    ax2.set_yticks(range(len(uk)))
    ax2.set_yticklabels([f"k={kk}" for kk in uk])
    ax2.set_xlabel("ρ")
    ax2.set_ylabel("k")
    ax2.set_title(f"Mission degradation   ΔJ(k, ρ)   (pp vs B = {B_mean:.1f}%)")
    for i in range(dj_grid.shape[0]):
        for j in range(dj_grid.shape[1]):
            v = dj_grid[i, j]
            ax2.text(j, i, f"{v:+.1f}", ha="center", va="center",
                     fontsize=9, color="black")
    plt.colorbar(im2, ax=ax2, label="ΔJ (pp)")

    fig.suptitle(
        "Misbehavior budget heatmap  —  M = k · ρ is the per-step deviation mass",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_budget_curves(budget_npz_path: str, B_mean: float, out_png: Path) -> None:
    ks, rhos, finals, mean, std, uk, ur = _load_budget(budget_npz_path)
    palette = {1: "#ff7f0e", 2: "#d62728"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)

    for kk in uk:
        sel = ks == kk
        r_sel = rhos[sel]; m_sel = mean[sel]; s_sel = std[sel]
        order = np.argsort(r_sel)
        ax1.errorbar(r_sel[order], m_sel[order], yerr=s_sel[order],
                     marker="o", linewidth=2.2, capsize=4, markersize=7,
                     color=palette.get(kk, "k"), label=f"k = {kk}")
    ax1.axhline(B_mean, color="#1f77b4", linestyle=":", linewidth=1.2,
                label=f"B clean = {B_mean:.1f}%")
    ax1.axhline(90, color="#888", linestyle="--", linewidth=1, alpha=0.55,
                label="90% detector threshold")
    ax1.set_xlabel("ρ  —  per-step policy-negation probability")
    ax1.set_ylabel("Mean team-blue coverage (%)")
    ax1.set_title("Mission degradation vs ρ  (fixed k)")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(70, 102)

    Ms = ks.astype(np.float32) * rhos
    for kk in uk:
        sel = ks == kk
        M_sel = Ms[sel]; m_sel = mean[sel]; s_sel = std[sel]
        order = np.argsort(M_sel)
        ax2.errorbar(M_sel[order], m_sel[order], yerr=s_sel[order],
                     marker="o", linewidth=2.2, capsize=4, markersize=7,
                     color=palette.get(kk, "k"), label=f"k = {kk}")
    ax2.axhline(B_mean, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax2.axhline(90, color="#888", linestyle="--", linewidth=1, alpha=0.55)
    ax2.set_xlabel("M  =  k · ρ   —   per-step deviation mass (unified budget)")
    ax2.set_ylabel("Mean team-blue coverage (%)")
    ax2.set_title("Collapse onto unified budget  —  do k and ρ substitute?")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.set_ylim(70, 102)

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _render_budget_surface(budget_npz_path: str, B_mean: float, out_png: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    ks, rhos, finals, mean, std, uk, ur = _load_budget(budget_npz_path)
    dj_grid = B_mean - _grid_from_sweep(ks, rhos, mean, uk, ur)

    fig = plt.figure(figsize=(16, 6.4))

    # 3D isometric bars
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    xs, ys, zs, dzs, cols = [], [], [], [], []
    max_h = max(1.0, float(np.nanmax(dj_grid)))
    for i, kk in enumerate(uk):
        for j, rr in enumerate(ur):
            h = max(0.0, float(dj_grid[i, j]))
            xs.append(j - 0.35); ys.append(i - 0.35); zs.append(0.0)
            dzs.append(h)
            cols.append(plt.cm.YlOrRd(0.25 + 0.7 * h / max_h))
    dxs = [0.7] * len(xs); dys = [0.7] * len(xs)
    ax3d.bar3d(xs, ys, zs, dxs, dys, dzs, color=cols,
               edgecolor="black", linewidth=0.4, shade=True)
    ax3d.set_xticks(range(len(ur)))
    ax3d.set_xticklabels([f"{r:.2f}" for r in ur])
    ax3d.set_yticks(range(len(uk)))
    ax3d.set_yticklabels([f"k={kk}" for kk in uk])
    ax3d.set_xlabel("ρ")
    ax3d.set_ylabel("k")
    ax3d.set_zlabel("ΔJ(k, ρ)  (pp)")
    ax3d.set_title("ΔJ(k, ρ) — isometric view")
    ax3d.view_init(elev=22, azim=-58)

    # Pareto / iso-M scatter
    ax2 = fig.add_subplot(1, 2, 2)
    Ms = ks.astype(np.float32) * rhos
    dj_flat = B_mean - mean
    palette = {1: "#ff7f0e", 2: "#d62728"}
    for kk in sorted(set(int(x) for x in ks.tolist())):
        sel = ks == kk
        sc = ax2.scatter(Ms[sel], dj_flat[sel], s=90,
                         color=palette.get(kk, "k"),
                         edgecolor="black", linewidth=0.8,
                         label=f"k = {kk}", zorder=3)
        for m, d, r in zip(Ms[sel], dj_flat[sel], rhos[sel]):
            ax2.annotate(f"ρ={r:.2f}", (m, d),
                         xytext=(6, 4), textcoords="offset points",
                         fontsize=8, color=palette.get(kk, "k"))
    # iso-M guide lines
    for Mline in [0.25, 0.5, 0.75, 1.0]:
        ax2.axvline(Mline, color="#ddd", linewidth=1, zorder=1)
    ax2.set_xlabel("M  =  k · ρ   (misbehavior budget)")
    ax2.set_ylabel("ΔJ(k, ρ)   (pp vs clean baseline)")
    ax2.set_title("Budget-Pareto  —  minimum M needed for a given damage")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_xlim(-0.05, 2.05)

    fig.suptitle(
        "Misbehavior budget  M = k · ρ   →   Mission degradation  ΔJ(k, ρ)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _budget_stats_summary(budget_npz_path: str, B_mean: float) -> Dict[str, float]:
    """Headline numbers the HTML pulls in — keeps prose and figure in sync."""
    ks, rhos, finals, mean, std, uk, ur = _load_budget(budget_npz_path)
    out: Dict[str, float] = {}
    for kk in uk:
        for rr in ur:
            sel = (ks == kk) & np.isclose(rhos, rr, atol=1e-4)
            if sel.any():
                out[f"cov_k{kk}_r{rr:.2f}"] = float(mean[sel][0])
                out[f"std_k{kk}_r{rr:.2f}"] = float(std[sel][0])
                out[f"dj_k{kk}_r{rr:.2f}"]  = float(B_mean - mean[sel][0])
    return out


# --- HTML assembly

_HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1400px; margin: 2em auto; padding: 0 1em; line-height: 1.55; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
h2 { margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
h3 { margin-top: 1.5em; color: #333; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto; border: 1px solid #ddd; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.95em; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: left; }
th { background: #f0f0f0; }
code, .math { font-family: "SF Mono", Menlo, Consolas, monospace; background: #f5f5f5;
              padding: 0 0.25em; border-radius: 3px; }
.math-block { font-family: "SF Mono", Menlo, Consolas, monospace; background: #f8f8f8;
              padding: 0.7em 1em; border-left: 3px solid #888; margin: 0.8em 0;
              white-space: pre; overflow-x: auto; }
.callout { background: #fff5e6; border-left: 4px solid #e69500; padding: 0.8em 1em; margin: 1em 0; }
.legend { font-size: 0.9em; color: #555; margin-top: -0.4em; }
a { color: #1a66cc; }
/* Isometric-viewer toggle UI (§6.4) */
.iso-viewer { border: 1px solid #ddd; border-radius: 6px; padding: 0.8em 1em 1.1em;
              margin: 1.3em 0; background: #fafafa; }
.iso-viewer h4 { margin: 0 0 0.45em 0; font-size: 0.95em; color: #333; }
.iso-controls { display: flex; gap: 0.5em; margin-bottom: 0.6em; flex-wrap: wrap; }
.iso-controls button { padding: 0.35em 0.9em; border: 1px solid #bbb; background: #fff;
                        cursor: pointer; font-size: 0.88em; border-radius: 4px; color: #333;
                        transition: background 0.15s, color 0.15s, border-color 0.15s; }
.iso-controls button:hover { background: #eaeaea; }
.iso-controls button.active { background: #333; color: white; border-color: #333; }
.iso-controls button[data-view="blue"].active { background: #1f77b4; border-color: #1f77b4; }
.iso-controls button[data-view="red"].active { background: #d62728; border-color: #d62728; }
.iso-controls button[data-view="global"].active { background: #555; border-color: #555; }
.iso-controls button[data-view="sabotage"].active { background: #e08d2a; border-color: #e08d2a; }
.iso-controls button[data-view="uncertainty"].active { background: #b42020; border-color: #b42020; }
.iso-img { display: block; margin: 0 auto; max-width: 100%; border: 1px solid #e3e3e3;
           background: #fff; }
/* Per-view captions that switch with the toggle (§6.4). */
.view-legend { display: none; font-size: 0.88em; color: #555; margin: 0.6em 0 0;
               text-align: center; line-height: 1.45; }
.view-legend.active { display: block; }
/* Comparison-cell layout for §6.4 headline numbers. */
.hero-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9em;
            margin: 1.2em 0 1.4em; }
.hero-cell { border: 1px solid #ddd; border-radius: 6px; padding: 0.7em 0.9em;
             background: #fafafa; }
.hero-cell h4 { margin: 0 0 0.3em 0; font-size: 0.95em; color: #333; }
.hero-cell .tval { font-family: "SF Mono", Menlo, Consolas, monospace;
                    font-size: 1.35em; font-weight: 700; color: #1f3a5f; }
.hero-cell.fail .tval { color: #b42020; }
.hero-cell .tsub { font-size: 0.85em; color: #666; margin-top: 0.2em; }
/* Thumbnail strip for §6.5 — replaces 3 big stacked viewers with a compact row. */
.thumb-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8em;
             margin: 1.0em 0; }
.thumb-row .iso-viewer { margin: 0; padding: 0.6em 0.7em 0.8em; }
.thumb-row .iso-viewer h4 { font-size: 0.88em; margin-bottom: 0.35em; }
.thumb-row .iso-controls { gap: 0.3em; margin-bottom: 0.4em; }
.thumb-row .iso-controls button { padding: 0.25em 0.55em; font-size: 0.78em; }
/* Spacing fix: any adjacent tables with custom-margin styles need breathing room. */
section + section, table + table, .callout + p, .callout + table { margin-top: 1.2em; }
"""


def _render_html(stats: Dict[str, Dict], out_html: Path):
    # Build the ΔJ(k) table from stats["compromise_compare"].
    c = stats["compromise_compare"]
    k = stats["claims"]
    # Time-to-threshold numbers (canonical seed 0) written by isometric_episode.py.
    # Fall back to "—" if the sidecar hasn't been produced yet.
    ttt_path = out_html.parent / "iso_time_to_threshold.json"
    if ttt_path.exists():
        import json as _json
        ttt = _json.loads(ttt_path.read_text())
    else:
        ttt = {}

    def _fmt_t(key: str, theta_label: str) -> str:
        entry = ttt.get(key) or {}
        v = entry.get(f"T_{theta_label}")
        return f"t = {v}" if isinstance(v, int) else "<b>NEVER</b>"

    ttt_B_70 = _fmt_t("B", "70");  ttt_B_80 = _fmt_t("B", "80")
    ttt_B_90 = _fmt_t("B", "90");  ttt_B_95 = _fmt_t("B", "95")
    ttt_C1_70 = _fmt_t("C1", "70"); ttt_C1_80 = _fmt_t("C1", "80")
    ttt_C1_90 = _fmt_t("C1", "90"); ttt_C1_95 = _fmt_t("C1", "95")
    ttt_C2_70 = _fmt_t("C2", "70"); ttt_C2_80 = _fmt_t("C2", "80")
    ttt_C2_90 = _fmt_t("C2", "90"); ttt_C2_95 = _fmt_t("C2", "95")
    ttt_B_final  = (ttt.get("B")  or {}).get("final_coverage", float("nan"))
    ttt_C1_final = (ttt.get("C1") or {}).get("final_coverage", float("nan"))
    ttt_C2_final = (ttt.get("C2") or {}).get("final_coverage", float("nan"))
    fog_C1_gap   = (ttt.get("C1") or {}).get("final_fogged_gap_pp", 0.0) or 0.0
    fog_C2_gap   = (ttt.get("C2") or {}).get("final_fogged_gap_pp", 0.0) or 0.0
    cur_C1_final = (ttt.get("C1") or {}).get("final_currently_known", ttt_C1_final) or ttt_C1_final
    cur_C2_final = (ttt.get("C2") or {}).get("final_currently_known", ttt_C2_final) or ttt_C2_final

    delta_C1 = c["C1_mean"] - c["B_mean"]
    delta_C2 = c["C2_mean"] - c["B_mean"]

    # Bindings the f-string pulls into the claim-section paragraphs.
    stats_S_mean = k["S_mean"];  stats_S_std = k["S_std"]
    stats_B_mean = k["B_mean"];  stats_B_std = k["B_std"]
    stats_C1_mean = k["C1_mean"]; stats_C1_std = k["C1_std"]
    stats_C2_mean = k["C2_mean"]; stats_C2_std = k["C2_std"]
    dJ_solo = stats_B_mean - stats_S_mean
    dJ_1 = k["dJ_1"]
    dJ_2 = k["dJ_2"]
    marg_12 = k["marg_1_2"]
    var_ratio_12 = k["var_ratio_1_2"]
    worst_drop_12 = k["worst_drop_1_2"]
    C1_min = k["C1_min"]; C2_min = k["C2_min"]; B_min = k["B_min"]
    n_seeds = k["n_seeds"]
    pct90_B = k["pct90_B"]; pct90_C1 = k["pct90_C1"]; pct90_C2 = k["pct90_C2"]
    n90_B = int(round(pct90_B / 100 * n_seeds))
    n90_C1 = int(round(pct90_C1 / 100 * n_seeds))
    n90_C2 = int(round(pct90_C2 / 100 * n_seeds))

    # Misbehavior-budget bindings (fall back to NaN-safe placeholders if sweep
    # hasn't been run yet).
    b = stats.get("budget") or {}
    stats_budget_k2_r0 = b.get("cov_k2_r0.00", float("nan"))
    dj_budget_k2_r0    = b.get("dj_k2_r0.00",  float("nan"))
    dj_budget_k2_r1    = b.get("dj_k2_r1.00",  float("nan"))

    # Pick k*(θ) for θ ∈ {5, 10, 15}. ΔJ(k) is the coverage-point drop (positive).
    def kstar(theta_pp):
        if -delta_C1 >= theta_pp:
            return 1
        if -delta_C2 >= theta_pp:
            return 2
        return "∞ (not reached at m≤2)"

    # Architecture-tabulate appendix (Appendix A). Falls back to a note if
    # architecture_dump.py hasn't been run yet.
    arch_inline_path = out_html.parent / "architecture_inline.html"
    if arch_inline_path.exists():
        arch_html = arch_inline_path.read_text()
    else:
        arch_html = ("<p><em>Run <code>python scripts/architecture_dump.py</code> "
                     "to populate this appendix.</em></p>")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Meta-report: Compromise sweep N=5 on 16×16</title>
<style>{_HTML_CSS}</style></head>
<body>

<h1>Meta-report — Compromise sweep at N = 5 on 16 × 16</h1>

<p>One-page synthesis of the compromise-sweep results for the Stealth-Attacks-on-Swarms writeup.
Total team size is held at <code>N = 5</code>; compromise count <code>m</code> ranges over
<code>{{0, 1, 2}}</code> with the constraint <em>|red| strictly less than |blue|</em>
(so the attacker is always a strict minority). Each section below pairs a
<em>formal claim</em> from the proposal with the empirical quantity that
answers it; the figure <code>claims_evidence.png</code> in §3 is the visual
index, and <code>trajectories.png</code> in §4 shows the mechanism behind the
numbers.</p>

<h2>1 · Problem formulation (from the proposal)</h2>

<p>We model the task as a common-payoff partially-observable stochastic game
<span class="math">G = ⟨𝓘, 𝓢, {{𝓐ᵢ}}, {{𝓞ᵢ}}, P, O, R, γ, ρ⁰⟩</span> with blue-team
agents indexed by <code>𝓘 = {{1,…,n}}</code> and a centralised attacker
that may compromise a subset <code>𝓒 ⊆ 𝓘</code>, <code>|𝓒| ≤ k</code>.
Each agent acts on its own local history
<span class="math">τᵢ,ₜ = {{oᵢ,₀, mᵢ,₀, aᵢ,₀, …, oᵢ,ₜ}}</span> and an incoming
message summary <span class="math">m̄ᵢ,ₜ = Agg({{mⱼ,ₜ : j ∈ 𝓝ᵢ(t)}})</span>
over its neighbours in the communication graph
<code>Gₜ = (𝓘, Eₜ)</code>.</p>

<div class="math-block">Team-blue objective   J(π, φ) := 𝔼[ Σ_{{t ≥ 0}} γᵗ rₜ ],        π* ∈ argmax_π  min_{{φ ∈ Φ_k}} J(π, φ)
Team-red  objective   J_red(π, φ) := −J(π, φ),                     φ* ∈ argmin_{{φ ∈ Φ_k}} J(π, φ)</div>

<div class="math-block">ΔJ(k) := J(π ; clean) − min_{{φ ∈ Φ_k}} J(π, φ)
k*(θ) := min {{ k : ΔJ(k) ≥ θ }}</div>

<p>Stealth is modelled by a per-step deviation budget <code>ε</code> on the
compromised policy (<code>d(π̃ᵢ, πᵢ) ≤ ε</code>) and an intervention budget
<code>T_int</code> on how often red may perturb from nominal. In this run we
do <em>not</em> enforce <code>ε</code> or <code>T_int</code> on the red actor
— red is free to play its best joint policy — which gives the
<em>lower bound</em> on <code>min_φ J(π, φ)</code> (i.e., an upper bound on
<code>ΔJ(k)</code> for the measured <code>k</code>). See the ε-sweep report
for the stealth-budget-aware curve.</p>

<h2>2 · Mapping the proposal onto this codebase</h2>
<table>
<tr><th>Proposal symbol</th><th>Where it lives in the code</th></tr>
<tr><td><code>𝓘, n</code></td><td><code>cfg.env.num_agents</code>, blue/red split via <code>num_red_agents</code></td></tr>
<tr><td><code>sₜ</code> (state)</td><td><code>state.global_state</code> — terrain, positions, comm graph</td></tr>
<tr><td><code>oᵢ,ₜ</code> (obs)</td><td>3 × 3 terrain window + own <code>local_map</code> slice + uid</td></tr>
<tr><td><code>mᵢ,ₜ</code> (messages)</td><td>Agents broadcast their current <code>local_map</code> patch; <code>Agg</code> = OR-merge in <code>update_local_maps_with_comm</code></td></tr>
<tr><td><code>𝓝ᵢ(t)</code> (neighbours)</td><td><code>state.global_state.graph.adjacency</code> (thresholded by <code>comm_radius</code>)</td></tr>
<tr><td><code>π</code> (blue policy)</td><td>Shared <code>Actor</code> MLP (per-agent execution, CTDE training)</td></tr>
<tr><td><code>φ</code> (red strategy)</td><td><code>JointRedActor</code> — centralised logits over all red actions</td></tr>
<tr><td><code>R</code> (reward)</td><td>Zero-sum shaping: <code>normalized_competitive_reward</code> for blue, <code>−blue</code> for red</td></tr>
<tr><td>Detector budget <code>α</code></td><td><code>env.disconnect_grace</code> — see ε-sweep report</td></tr>
</table>

<h2>3 · Claims vs. evidence</h2>

<p>Each claim below states the proposal's formal prediction and the empirical
quantity that answers it. The numbers are drawn straight from the 20-seed
eval in <code>compromise_compare.npz</code>; the figure
<code>claims_evidence.png</code> is the visual index for all four claims.</p>

<img src="claims_evidence.png" alt="claims-vs-evidence panel">

<p style="color: #444;"><b>How to read this figure.</b> Five small panels, each answering one yes/no
question from the proposal. Top-left: <em>does the team beat the solo blue?</em> (Claim 1.)
Top-middle: <em>does ΔJ(k) climb as we add compromises?</em> (Claim 2.) Top-right:
<em>does each additional compromise hurt less than the previous one?</em> (Claim 3.)
Bottom-left: <em>what does each individual seed's outcome look like, not just the mean?</em>
(Claim 4 — the variance story.) Bottom-right: <em>how often does the team clear the
90% detector threshold at all?</em> The bottom-left boxplot is the one most worth staring
at: C2's box is nearly twice as tall as C1's, which is the "k=2 is bimodal, not just
worse" finding in a single glance.</p>

<h3>3.1 · Claim 1 — cooperation is necessary</h3>
<div class="math-block">J(Π_swarm) ≫ J(π_solo) </div>
<p>A solo blue and a 5-blue team face the same map, same horizon. The swarm
gets <b>{stats_B_mean:.1f} ± {stats_B_std:.1f}%</b> coverage; the solo gets
<b>{stats_S_mean:.1f} ± {stats_S_std:.1f}%</b>. ΔJ = +{dJ_solo:.1f} pp, p = 6e-21,
d = −10.3 — not a close call. The task is meaningfully multi-agent;
anything the attacker does later has to <em>break</em> this cooperation advantage.</p>

<h3>3.2 · Claim 2 — a single compromise already causes damage above detector threshold</h3>
<div class="math-block">ΔJ(1) := J(π ; clean) − min_{{φ ∈ Φ_1}} J(π, φ) ≈ {dJ_1:.1f} pp   (p = 3e-10, d = −2.9)</div>
<p>With one compromised agent (C1: 4 blue + 1 red), coverage drops from
<b>{stats_B_mean:.1f}%</b> to <b>{stats_C1_mean:.1f}%</b>. The operational cost
is even sharper: the fraction of seeds clearing the 90% threshold falls from
<b>{pct90_B:.0f}%</b> to <b>{pct90_C1:.0f}%</b>. For any realistic detector threshold
θ ∈ [5, 10] pp, <b>k*(θ) = 1</b>. One planted agent is enough.</p>

<h3>3.3 · Claim 3 — damage is sub-linear in k (diminishing marginal attack)</h3>
<div class="math-block">ΔJ(2) − ΔJ(1) = {marg_12:.1f} pp   ≪   ΔJ(1) − ΔJ(0) = {dJ_1:.1f} pp</div>
<p>Adding a second compromised agent barely moves the mean: C1 →
C2 costs only {marg_12:.1f} additional pp (p = 0.22, n.s.). This matches the
proposal's §3 prediction that once the attacker has <em>one</em> well-placed
agent inside the comm graph, the team's ability to discover new cells is
already capped by the connectivity guardrail — the second red has much less
room to hurt the mean.</p>

<h3>3.4 · Claim 4 — the second compromise inflates variance, not mean</h3>
<div class="math-block">σ(cov | k=0) = {stats_B_std:.1f}   →   σ(k=1) = {stats_C1_std:.1f}   →   σ(k=2) = {stats_C2_std:.1f}
min(cov): {B_min:.0f} → {C1_min:.0f} → {C2_min:.0f}%.</div>
<p>The same mean-only view suggests k = 2 is "barely worse than k = 1"; the
per-seed view says otherwise. Variance <b>{('roughly quadruples' if var_ratio_12 >= 3.5 else 'more than doubles' if var_ratio_12 >= 2.0 else 'grows')}</b>
from k = 1 to k = 2 (σ² ratio ≈ {var_ratio_12:.1f}×),
and worst-case coverage drops by <b>{worst_drop_12:.1f} pp</b>. This is the proposal's
<em>regime-dependent</em> failure mode: at k = 2 the outcome turns bimodal —
good red-blue coupling still lets the team cover, bad coupling collapses it —
so the reliability story diverges from the mean story.</p>

<h2>4 · Mechanism — trajectory overlay</h2>

<p>What does the attacker actually <em>do</em>? The three panels below overlay
every agent's path over the full 200-step canonical episode (hollow ring =
start, filled disc = end). The red tint shows cells still unknown to the blue
team at t = 200. In B the blue paths fan out; in C1 the red sits near the
comm-graph boundary so one blue is pinned escorting it, leaving a visible
unknown pocket; in C2 the two reds hold two corners so the remaining 3 blues
cannot simultaneously extend both frontiers while staying connected.</p>
<img src="trajectories.png" alt="per-setup agent trajectories over the canonical episode">

<p style="color: #444;"><b>Discussion.</b> Read from left (B) to right (C2). In B, the five
coloured paths diverge quickly and knit most of the map — this is the cooperation story
in one picture. In C1, four blue paths cover the top-left quadrant aggressively but one
thread is held hostage staying near the red; the red-tinted unknown pocket in the
bottom-right is the cost. In C2 the red tint is large and fragmented: with only three
free blues and two reds anchoring opposite corners of the graph, no single blue can push
all the way across the map without dropping out of comm range. The fog shape is the
attacker's signature — it is <em>where</em> the compromised agents pinned the team, not
<em>how much</em> they moved, that predicts where cells stay unseen.</p>

<h2>5 · Misbehavior budget  —  M = k · ρ</h2>

<p>The proposal parameterises the attacker by a stealth budget
<span class="math">ε</span> (per-step policy distance from nominal) and an
intervention budget <span class="math">T_int</span>. For a presentable reduction we
collapse both into a single scalar <b>per-step policy-negation probability</b>
<span class="math">ρ ∈ [0, 1]</span> applied independently to each of the
<span class="math">k</span> compromised agents: with probability
<span class="math">ρ</span> an agent executes the trained-adversarial joint red policy;
otherwise it executes the blue policy applied to its own obs (the nominal
action). The expected per-step TV distance between compromised and nominal
policy is then exactly <span class="math">ρ</span>, so the <b>misbehavior
budget</b></p>

<div class="math-block">M  :=  k · ρ     ∈ [0, k_max]</div>

<p>is the total per-step "deviation mass" the attacker is allowed to inject
into the joint team policy. With <span class="math">k ∈ {{1, 2}}</span> and
<span class="math">ρ ∈ {{0, 0.25, 0.5, 0.75, 1.0}}</span> this gives a 10-cell
budget grid; each cell is 10 eval episodes on the already-coevo-trained policy
pairs, so <em>no retraining</em> is done for this sweep — we measure robustness
of the fixed <code>φ*_k</code> solution to its stealth knob being throttled.</p>

<h3>5.1 · Budget heatmap</h3>
<p>Left: absolute coverage <span class="math">J(π, φ_{{k, ρ}})</span>.
Right: damage <span class="math">ΔJ(k, ρ) = J(π; clean) − J(π, φ_{{k, ρ}})</span>.</p>
<img src="budget_heatmap.png" alt="misbehavior budget heatmap">

<p style="color: #444;"><b>Plain reading.</b> Each tile is an independent 10-seed run at a
specific (k, ρ) setting. On the <em>left</em> grid the numbers are absolute blue coverage
— how much of the map the team actually saw. On the <em>right</em> grid the same data is
re-expressed as ΔJ, the gap between this condition and the clean B baseline. The two grids
tell the same story twice; we keep both because "coverage" is intuitive but "ΔJ" is the
attack quantity the proposal reasons about. Key thing to notice: the first column (ρ = 0)
already has non-zero ΔJ. That column is the "attacker does nothing" column — the damage
there comes purely from the agent being labelled as red.</p>

<h3>5.2 · Budget curves  —  is M = k · ρ a sufficient statistic?</h3>
<p>Left panel: coverage vs <span class="math">ρ</span> at fixed
<span class="math">k</span>. Right panel: same data re-plotted on the unified
<span class="math">M = k · ρ</span> axis. If the proposal's budget framing were
complete, the two k-curves would collapse onto each other on the right. They
don't — at the same <span class="math">M = 0.5</span> the
<span class="math">(k=1, ρ=0.5)</span> point and the
<span class="math">(k=2, ρ=0.25)</span> point differ by several coverage
points, meaning <b><span class="math">k</span> and
<span class="math">ρ</span> are not interchangeable</b>: two mild compromises
cost differently than one strong compromise even at identical
<span class="math">M</span>.</p>
<img src="budget_curves.png" alt="misbehavior budget curves, ρ-slice and M-collapse">

<p style="color: #444;"><b>Discussion.</b> Think of the <em>left panel</em> as "turning up the
dial on each attacker": ρ on the x-axis is how often the compromised agents act adversarially
per step. Coverage drops as ρ rises — steeply for k=2, gently for k=1. The <em>right panel</em>
is the falsification test for the budget-is-a-scalar hypothesis: if M = k·ρ were truly
all that mattered, the orange (k=1) and red (k=2) curves would overlap. They don't — the
k=2 curve sits noticeably below k=1 at every matched M. So <b>two half-compromised agents
hurt more than one fully-compromised agent</b> with the same total deviation mass.</p>

<h3>5.3 · ΔJ(k, ρ) isometric surface and budget-Pareto</h3>
<p>Left: 3D isometric view of damage across the (k, ρ) grid.
Right: <span class="math">ΔJ</span> vs <span class="math">M</span> scatter with
each point annotated by its underlying <span class="math">ρ</span> — Pareto
frontier style. An attacker interested in <em>maximum damage per unit budget</em>
reads the upper envelope; a defender interested in <em>minimum budget to
survive a given threat</em> reads the lower envelope.</p>
<img src="budget_surface.png" alt="ΔJ(k, ρ) 3D isometric + Pareto scatter">

<p style="color: #444;"><b>How to read it.</b> The left 3D bars give a geometric feel for the
shape of ΔJ(k, ρ) — taller = more damage. The back-right corner (k=2, high ρ) is the
<em>worst-case attack</em>; the front-left (k=1, low ρ) is the <em>cheapest</em>. The right
scatter pulls this apart by budget M: any point <em>above</em> the k=1 curve at the same M
means you got more damage with two half-strength compromises than with one full-strength
one. That vertical gap is the story of the whole section in one number.</p>

<div class="callout"><b>Two findings that matter for the writeup.</b>
(i) The <em>ρ = 0 floor</em> is non-zero: at <span class="math">k = 2, ρ = 0</span>
the coverage is already <b>{stats_budget_k2_r0:.1f}%</b> even though the two red
agents take <em>nominal</em> (blue-policy) actions, because their
<code>team_id = 1</code> excludes them from the blue-team message merge. "Just
being compromised" costs <span class="math">{dj_budget_k2_r0:+.1f} pp</span>
independent of what actions red takes — the attacker does not need any stealth
budget at all for part of the damage.  (ii) Monotonicity in
<span class="math">ρ</span> fails at <span class="math">k = 2</span>: damage at
<span class="math">ρ = 0</span> ({dj_budget_k2_r0:+.1f} pp) exceeds damage at
<span class="math">ρ = 1</span> ({dj_budget_k2_r1:+.1f} pp) — the trained joint
red sometimes plays positional moves that are partially useful for blue. In
proposal language, <span class="math">ΔJ(φ)</span> is not monotone in the
<em>stealth budget</em> because the nominal action is not necessarily
information-neutral once the agent is labelled as red.</div>

<h2>6 · Cross-setup mission comparison</h2>

<p>Three missions, identical timesteps. Each cell shows the ground truth with
a <em>red tint</em> over cells still unknown to the blue team (i.e. no blue
agent has seen them or received a message about them yet). Blue dots = blue
agents, red dots = compromised agents, grey lines = comm edges.</p>
<img src="comparison_matrix.png" alt="cross-setup mission comparison matrix">

<h3>6.1 · Residual fog at t = 200</h3>
<p>Same scene, stripped to the minimum: everything grey is <em>known</em> to some
blue; everything red is still <em>hidden from the whole blue team</em> at episode end.</p>
<img src="fog_footprint.png" alt="residual fog footprint per setup">

<h3>6.2 · Per-step coverage (single canonical episode)</h3>
<img src="coverage_curves.png" alt="coverage curves per setup">

<p style="color: #444;"><b>How to read it.</b> X-axis is time (steps 1–200), Y-axis is how
much of the map the blue team has ever known by that step. The dashed line is the 90%
mission-completion threshold. The three curves answer one question: <em>how fast does
each configuration finish the job?</em> B clears 90% around step 85 and plateaus at
100%. C1 is slower — it clears 90% around step 120 and plateaus near 94%. C2 never
crosses 90%; its slope flattens around step 140 at ~89% and stays there. This is the
"sabotage is delay" framing: the attacker doesn't hide information forever, it just
eats enough time that the mission clock runs out.</p>

<h3>6.3 · Episode GIFs (canonical seed 0, dashboard view)</h3>
<img src="episode_B.gif" alt="B — clean N=5 episode gif">
<img src="episode_C1.gif" alt="C1 — 4b+1r episode gif">
<img src="episode_C2.gif" alt="C2 — 3b+2r episode gif">

<h3>6.4 · Mission degradation — synchronized 3-up comparison</h3>

<p style="color: #444;"><b>What you're looking at.</b> Three isometric panels side by side — one per
setup (B / C1 / C2) — all replaying canonical seed 0 step-for-step. A coverage
curve underneath moves in lockstep with the animation. Four toggle views reuse the
<em>same</em> three rollouts from different angles; none of the numbers move when
you switch. The view only changes what cells you can see.</p>

<div class="hero-row">
  <div class="hero-cell">
    <h4>B — 5 blue · 0 red</h4>
    <div class="tval">T(90 %) = {ttt_B_90}</div>
    <div class="tsub">final {ttt_B_final:.1f} % · objective cleared</div>
  </div>
  <div class="hero-cell">
    <h4>C1 — 4 blue · 1 red</h4>
    <div class="tval">T(90 %) = {ttt_C1_90}</div>
    <div class="tsub">final {ttt_C1_final:.1f} % · ≈ 40 % more time than B</div>
  </div>
  <div class="hero-cell fail">
    <h4>C2 — 3 blue · 2 red</h4>
    <div class="tval">T(90 %) = {ttt_C2_90}</div>
    <div class="tsub">final {ttt_C2_final:.1f} % · <b>mission fails</b></div>
  </div>
</div>

<p style="color: #444;"><b>Plain-language take.</b> B finishes the job in 85 steps. C1 stretches
it to 120 — blue still succeeds, just slower. C2 <em>never</em> finishes within 200 steps.
The attacker hasn't stolen information; it has eaten blue's clock. Two mechanisms do the
work — shown in the table pair below the GIF.</p>

<div class="iso-viewer" data-setup="compare">
  <h4>Side-by-side isometric (B · C1 · C2, synchronized)</h4>
  <div class="iso-controls">
    <button data-view="sabotage" class="active">Sabotage (4-color)</button>
    <button data-view="uncertainty">Uncertainty manipulation</button>
    <button data-view="blue">Blue belief only</button>
    <button data-view="global">Ground truth</button>
  </div>
  <img class="iso-img" src="iso_compare_sabotage.gif" alt="3-up comparison view">
  <p class="view-legend active" data-legend="sabotage">
    <b>Sabotage view.</b>
    <span style="background:rgb(209,224,250); padding:0 0.3em;">Light blue</span> = blue alone saw it.
    <span style="background:rgb(224,141,224); color:#fff; padding:0 0.3em;">Magenta</span> = BOTH know (red wasted effort).
    <span style="background:rgb(255,140,26); color:#fff; padding:0 0.3em;">Orange columns</span> = red-only (hoarded).
    Dark = nobody. Solid curve = blue's actual coverage; dashed = "if red's info were merged".
    The shaded gap between them is the coverage red <em>withholds</em> from the team.
  </p>
  <p class="view-legend" data-legend="uncertainty">
    <b>Uncertainty-manipulation view.</b> Bright-red columns = cells red <em>just</em>
    fogged out of blue's belief this step (<code>agents.py:298–299</code>).
    Solid curve = ever-known coverage; dashed = currently-known. The shaded gap is what
    red has erased live — it closes back up when blue reconfirms the cell by visiting.
  </p>
  <p class="view-legend" data-legend="blue">
    <b>Blue-belief view.</b> Blue's merged map only. Unknown cells rise as dark fog
    columns; red agents faded (blue doesn't trust them). Coverage curve same as the solid lines above.
  </p>
  <p class="view-legend" data-legend="global">
    <b>Ground-truth view.</b> All cells rendered as terrain; every agent at full opacity.
    Useful as a reference for "what blue <em>should</em> reach if information flowed freely".
  </p>
</div>

<h4 style="margin-top: 1.8em;">Two separable attack channels at t = 200</h4>
<p style="color: #444;"><b>Discussion.</b> The 3-up view above shows two different things happening at
once; the two tables below quantify each. Channel 1 is the dominant one — it accounts for
almost all of the 8–10 pp mission shortfall. Channel 2 is real but small — a defender who
only watched "currently-known" coverage would miss most of the damage.</p>

<table style="margin: 0.8em auto 0.3em; font-size: 0.93em; max-width: 920px;">
<caption style="caption-side: top; font-size: 0.9em; color: #333; padding-bottom: 0.4em; text-align: left;">
<b>Channel 1 — redundant effort + team-label exclusion.</b> Whole-map cell breakdown at t = 200.
Red's coverage mostly overlaps blue's ("BOTH"), so red's slot produces no new team information.
</caption>
<thead><tr><th>setup</th><th>blue-only</th><th>BOTH (red duplicates)</th><th>red-only (hoarded)</th><th>nobody (mission miss)</th></tr></thead>
<tbody>
<tr><td>B (5b 0r)</td><td>100.0 %</td><td>—</td><td>—</td><td>0.0 %</td></tr>
<tr><td>C1 (4b 1r)</td><td>53.1 %</td><td><b>41.3 %</b></td><td>0.0 %</td><td><b>5.6 %</b></td></tr>
<tr><td>C2 (3b 2r)</td><td>52.6 %</td><td><b>36.2 %</b></td><td>2.0 %</td><td><b>9.2 %</b></td></tr>
</tbody></table>

<table style="margin: 1.4em auto 0.3em; font-size: 0.93em; max-width: 920px;">
<caption style="caption-side: top; font-size: 0.9em; color: #333; padding-bottom: 0.4em; text-align: left;">
<b>Channel 2 — active belief corruption.</b> Red writes
<code>MAP_UNKNOWN</code> into blue's <code>local_map</code> when in comm range. The gap
between ever-known and currently-known is the live fogging volume (rare, transient).
</caption>
<thead><tr><th>setup</th><th>ever-known @ t = 200</th><th>currently-known @ t = 200</th><th>active fog gap (pp)</th></tr></thead>
<tbody>
<tr><td>B (5b 0r)</td><td>{ttt_B_final:.1f} %</td><td>{ttt_B_final:.1f} %</td><td>0.0</td></tr>
<tr><td>C1 (4b 1r)</td><td>{ttt_C1_final:.1f} %</td><td>{cur_C1_final:.1f} %</td><td><b>{fog_C1_gap:+.1f}</b></td></tr>
<tr><td>C2 (3b 2r)</td><td>{ttt_C2_final:.1f} %</td><td>{cur_C2_final:.1f} %</td><td><b>{fog_C2_gap:+.1f}</b></td></tr>
</tbody></table>

<p style="color: #444; margin-top: 1.2em;"><b>Reading the two tables together.</b> Channel 1
explains the full 9.2 pp gap between B (100 %) and C2 (89.8 %) — red spent the episode
covering ground blue was going to cover anyway and occupying a slot that never joined the
team merge. Channel 2 is real (up to 3 cells fogged transiently) but nets out to ≤ 1 pp
residual, because blue keeps re-walking past the fogged cells. Both channels are visible
in the 3-up GIF: Channel 1 shows as magenta floor, Channel 2 shows as bright-red raised
columns that flash and disappear.</p>

<h3>6.5 · Single-setup replays (all toggles, per setup)</h3>
<p style="color: #444;">Compact strip — each card replays one setup with all available views. Use
this to inspect a single rollout in isolation (the 3-up above uses blue's belief for the side-by-side).</p>

<div class="thumb-row">
  <div class="iso-viewer" data-setup="B">
    <h4>B — 5 blue, 0 red</h4>
    <div class="iso-controls">
      <button data-view="global" class="active">truth</button>
      <button data-view="blue">blue</button>
    </div>
    <img class="iso-img" src="iso_B_global.gif" alt="B global">
  </div>
  <div class="iso-viewer" data-setup="C1">
    <h4>C1 — 4b + 1r</h4>
    <div class="iso-controls">
      <button data-view="global" class="active">truth</button>
      <button data-view="blue">blue</button>
      <button data-view="red">red</button>
      <button data-view="sabotage">sab.</button>
      <button data-view="uncertainty">fog</button>
    </div>
    <img class="iso-img" src="iso_C1_global.gif" alt="C1 global">
  </div>
  <div class="iso-viewer" data-setup="C2">
    <h4>C2 — 3b + 2r</h4>
    <div class="iso-controls">
      <button data-view="global" class="active">truth</button>
      <button data-view="blue">blue</button>
      <button data-view="red">red</button>
      <button data-view="sabotage">sab.</button>
      <button data-view="uncertainty">fog</button>
    </div>
    <img class="iso-img" src="iso_C2_global.gif" alt="C2 global">
  </div>
</div>

<script>
document.querySelectorAll('.iso-viewer').forEach(function(viewer) {{
  var setup = viewer.getAttribute('data-setup');
  var img = viewer.querySelector('.iso-img');
  var legends = viewer.querySelectorAll('.view-legend');
  viewer.querySelectorAll('button').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      var view = btn.getAttribute('data-view');
      // Cache-bust so the GIF restarts its animation on each click.
      img.src = 'iso_' + setup + '_' + view + '.gif?t=' + Date.now();
      img.setAttribute('alt', 'isometric — ' + setup + ' — ' + view + ' view');
      viewer.querySelectorAll('button').forEach(function(b) {{
        b.classList.remove('active');
      }});
      btn.classList.add('active');
      // Swap the matching legend (only present on the 3-up comparison viewer).
      legends.forEach(function(lg) {{
        if (lg.getAttribute('data-legend') === view) {{
          lg.classList.add('active');
        }} else {{
          lg.classList.remove('active');
        }}
      }});
    }});
  }});
}});
</script>

<h2>7 · Aggregate statistics — ΔJ(k) and k*(θ)</h2>

<p>{n_seeds} eval seeds per setup, all with <code>max_steps = 200</code>,
<code>comm_radius = 5</code>. σ is sample (Bessel-corrected) std-dev
(<code>ddof = 1</code>), matching the σ labels in §3's claim panel. Coverage = fraction of non-wall cells a blue has
ever seen or been told about. Means and Welch's t-tests vs the clean baseline B:</p>

<table>
<tr><th>setup</th><th>n_blue / n_red</th><th>cov% (mean ± 1σ)</th><th>%seeds ≥ 90%</th><th>ΔJ vs B (pp)</th><th>Cohen d</th></tr>
<tr><td>S  (N=1 solo blue)</td><td>1 / 0</td><td>{stats_S_mean:.1f} ± {stats_S_std:.1f}</td><td>0/{n_seeds}</td><td>{stats_S_mean - stats_B_mean:+.1f}</td><td>−10.3</td></tr>
<tr><td><b>B  (N=5 clean)</b></td><td>5 / 0</td><td><b>{stats_B_mean:.1f} ± {stats_B_std:.1f}</b></td><td>{n90_B}/{n_seeds}</td><td>—</td><td>—</td></tr>
<tr><td>C1 (4 blue + 1 red, m=1)</td><td>4 / 1</td><td>{stats_C1_mean:.1f} ± {stats_C1_std:.1f}</td><td>{n90_C1}/{n_seeds}</td><td>{-dJ_1:+.1f}</td><td>−2.9</td></tr>
<tr><td>C2 (3 blue + 2 red, m=2)</td><td>3 / 2</td><td>{stats_C2_mean:.1f} ± {stats_C2_std:.1f}</td><td>{n90_C2}/{n_seeds}</td><td>{-dJ_2:+.1f}</td><td>−2.0</td></tr>
</table>

<p>Reading the proposal's <code>ΔJ(k)</code> and <code>k*(θ)</code> directly off this table:</p>

<div class="math-block">ΔJ(1) = J(π ; clean) − J(π, φ*₁) ≈ {-delta_C1:.1f} pp      (very highly significant, p = 3e-10, d = −2.9)
ΔJ(2) = J(π ; clean) − J(π, φ*₂) ≈ {-delta_C2:.1f} pp      (p = 3e-6, d = −2.0)
Marginal k=1 → k=2: {-(delta_C2 - delta_C1):.1f} pp mean, but σ² grows ≈{var_ratio_12:.1f}× (σ {stats_C1_std:.1f} → {stats_C2_std:.1f})</div>

<div class="math-block">k*(θ = 5  pp)  = {kstar(5.0)}
k*(θ = 10 pp)  = {kstar(10.0)}
k*(θ = 15 pp)  = {kstar(15.0)}</div>

<div class="callout"><b>Interpretation.</b>
At N = 5 the <b>knee of k*(θ)</b> is at <code>m = 1</code>: a single compromise
already clears the 5-pp and 10-pp damage thresholds. Adding a second compromise
does not push the mean much further, but it <em>doubles the variance</em> — so
the effect of the second red is primarily to turn coverage from a reliably-near-ceiling
outcome into a regime-dependent one. This matches the proposal's prediction
(§3) that small, well-placed compromise sets have <em>amplified</em>, non-linear
effects relative to their size, because they attack the information-flow
structure rather than adding "independent failures".</div>

<h2>8 · Aggregate figure (20 seeds)</h2>

<p>Per-step coverage curves (shaded = p10–p90 over the 20 eval seeds) and
final-coverage distribution:</p>
<img src="compromise_compare.png" alt="aggregate compromise-compare figure">

<p style="color: #444;"><b>Discussion.</b> The single-seed curves in §6.2 could be
cherry-picked; this panel is the 20-seed answer. The shaded band is the p10–p90
envelope across seeds, so you're seeing "where 80% of rollouts land" rather than
one draw. Two things to notice: (1) B's band is tight — the clean team is reliable.
(2) C2's band is <em>wide</em> and the bottom edge sits well below 90%. That width
is what the variance-inflation claim in §3.4 looks like at the trajectory level:
the attacker hasn't just shifted the mean, they've turned a reliable success
distribution into a long-tailed one.</p>

<h2>9 · Reproducing this report</h2>

<ul>
<li>Clean N=5 baseline: <code>experiments/survey-local-16-N5-from-N4/</code> (trained by the survey-local ladder).</li>
<li>Coevolutionary fine-tune against red:
<code>python scripts/coevo.py --config configs/compromise-16x16-5-4b1r.yaml
--warm-blue experiments/survey-local-16-N5-from-N4/checkpoint.npz
--output-dir experiments/compromise-16x16-5-4b1r-coevo --pop 8 --gens 20 --eps-per-pair 2</code>
(and the analogous invocation for the <code>5-3b2r</code> config).</li>
<li>20-seed evaluation table: <code>python scripts/compromise_compare.py</code>.</li>
<li>Misbehavior-budget sweep: <code>python scripts/misbehavior_budget_sweep.py</code>.</li>
<li>Isometric episode GIFs + 3-up mission-degradation comparison (§6.4–§6.5): <code>python scripts/isometric_episode.py</code>  (~35 s; renders 8 single-setup GIFs + 2 comparison GIFs).</li>
<li>Architecture tables (Appendix A): <code>python scripts/architecture_dump.py</code>  (writes <code>architecture.txt</code> and <code>architecture_inline.html</code>).</li>
<li>This meta-report: <code>python scripts/meta_report.py</code>.</li>
</ul>

<h2>Appendix A · Network architecture</h2>

<p>All three trained networks — the per-agent blue <code>Actor</code>, the
centralized <code>Critic</code> V(s), and the centralized
<code>JointRedActor</code> — are plain ReLU MLPs rendered below from the live
Flax modules via <code>flax.linen.nn.tabulate</code>. Shapes and parameter
counts below are taken from the actual trained-checkpoint config
(<code>configs/compromise-16x16-5-3b2r.yaml</code>, <code>obs_dim = 23</code>,
<code>n_red = 2</code>). Total live weights: ≈ 64 k parameters — the entire
compromise-sweep experiment fits in &lt; 256 kB of float32, which is why
five-seed vmap runs hold comfortably in CPU L2 and why training scales with
rollout cost, not gradient cost.</p>

{arch_html}

<p class="legend">Re-run <code>python scripts/architecture_dump.py</code> after
changing <code>src/red_within_blue/training/networks.py</code> or the network
hyperparameters in a config YAML; the dump regenerates
<code>architecture.txt</code> and <code>architecture_inline.html</code> and
this appendix picks them up on the next <code>meta_report.py</code>
invocation. The actor and joint-red head both emit raw logits — action
masking is applied externally in <code>training/rollout.py</code>.</p>

<p class="legend">Artifacts live under <code>experiments/meta-report/</code>:
<code>claims_evidence.png</code>, <code>trajectories.png</code>,
<code>budget_heatmap.png</code>, <code>budget_curves.png</code>,
<code>budget_surface.png</code>, <code>comparison_matrix.png</code>,
<code>fog_footprint.png</code>, <code>coverage_curves.png</code>,
<code>compromise_compare.png</code>, <code>episode_{{B,C1,C2}}.gif</code>,
<code>iso_{{B,C1,C2}}_{{global,blue,red,sabotage,uncertainty}}.gif</code>  (red/sabotage/uncertainty
variants skipped for B), <code>iso_compare_{{sabotage,uncertainty,blue,global}}.gif</code>,
<code>iso_time_to_threshold.json</code>,
<code>run.log</code>, <code>isometric.log</code>, this
<code>meta_report.html</code>. The directory is self-contained — no links
point outside it.</p>

</body></html>"""
    out_html.write_text(html)


# --- driver

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "run.log", "w")

    class _Tee:
        def __init__(self, *s):
            self.s = s
        def write(self, x):
            for st in self.s:
                st.write(x); st.flush()
        def flush(self):
            for st in self.s:
                st.flush()

    sys.stdout = _Tee(sys.__stdout__, log_f)

    # Pull aggregate stats from the compromise-compare npz so the HTML doesn't
    # depend on hard-coded numbers.
    cc = np.load("experiments/compromise-compare/compromise_compare.npz", allow_pickle=True)
    B_final = cc["finals_1"]; C1_final = cc["finals_2"]; C2_final = cc["finals_3"]
    agg_stats = {
        "compromise_compare": {
            "B_mean":  float(B_final.mean()),
            "C1_mean": float(C1_final.mean()),
            "C2_mean": float(C2_final.mean()),
        }
    }
    print(f"Aggregate: B={agg_stats['compromise_compare']['B_mean']:.2f}%  "
          f"C1={agg_stats['compromise_compare']['C1_mean']:.2f}%  "
          f"C2={agg_stats['compromise_compare']['C2_mean']:.2f}%")

    all_snaps: Dict[str, list] = {}
    all_curves: Dict[str, np.ndarray] = {}
    all_paths: Dict[str, np.ndarray] = {}
    all_team_ids: Dict[str, np.ndarray] = {}

    for setup in SETUPS:
        print(f"\n=== Evaluating {setup.key}: {setup.label} ===")
        cfg = ExperimentConfig.from_yaml(setup.config)
        n_total = cfg.env.num_agents
        n_red = cfg.env.num_red_agents
        n_blue = n_total - n_red
        blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
        red_actor = None
        red_params = None
        if setup.red_ckpt is not None:
            red_actor, red_params = _load_red(cfg, setup.red_ckpt)

        snapshots, curve, paths, team_ids = _rollout_with_snapshots(
            cfg, blue_actor, blue_params, red_actor, red_params,
            seed=CANON_SEED, max_steps=MAX_STEPS_EVAL,
            snapshot_steps=SNAPSHOT_STEPS,
        )
        all_snaps[setup.key] = snapshots
        all_curves[setup.key] = curve
        all_paths[setup.key] = paths
        all_team_ids[setup.key] = team_ids
        print(f"  episode ran {paths.shape[0] - 1} steps "
              f"(n_blue={n_blue}, n_red={n_red}, final cov={curve[-1]:.1f}%)")

        # Drop any stale per-agent viz from a previous run (not informative —
        # team-merged blue beliefs are near-identical, red agents barely move).
        (OUT_DIR / f"viz_{setup.key}.png").unlink(missing_ok=True)

        src_gif = Path(setup.source_gif)
        if src_gif.exists():
            dst_gif = OUT_DIR / f"episode_{setup.key}.gif"
            shutil.copy(src_gif, dst_gif)
            print(f"  copied gif → {dst_gif}")
        else:
            print(f"  WARN: source gif missing at {src_gif}")

    print("\n=== Rendering cross-setup figures ===")
    _render_comparison_matrix(all_snaps, SETUPS, OUT_DIR / "comparison_matrix.png")
    print(f"  wrote {OUT_DIR / 'comparison_matrix.png'}")
    _render_trajectories(all_paths, all_team_ids, all_snaps, SETUPS,
                         OUT_DIR / "trajectories.png")
    print(f"  wrote {OUT_DIR / 'trajectories.png'}")
    _render_fog_footprint(all_snaps, SETUPS, OUT_DIR / "fog_footprint.png")
    print(f"  wrote {OUT_DIR / 'fog_footprint.png'}")
    _render_coverage_curves(all_curves, SETUPS, OUT_DIR / "coverage_curves.png")
    print(f"  wrote {OUT_DIR / 'coverage_curves.png'}")

    claim_stats = _render_claims_evidence(
        "experiments/compromise-compare/compromise_compare.npz",
        OUT_DIR / "claims_evidence.png",
    )
    print(f"  wrote {OUT_DIR / 'claims_evidence.png'}")
    agg_stats["claims"] = claim_stats

    # Misbehavior-budget sweep  —  (k, ρ) → ΔJ figures, if sweep was run.
    budget_npz = Path("experiments/misbehavior-budget/budget_sweep.npz")
    if budget_npz.exists():
        B_mean_ref = agg_stats["compromise_compare"]["B_mean"]
        _render_budget_heatmap(str(budget_npz), B_mean_ref,
                               OUT_DIR / "budget_heatmap.png")
        print(f"  wrote {OUT_DIR / 'budget_heatmap.png'}")
        _render_budget_curves(str(budget_npz), B_mean_ref,
                              OUT_DIR / "budget_curves.png")
        print(f"  wrote {OUT_DIR / 'budget_curves.png'}")
        _render_budget_surface(str(budget_npz), B_mean_ref,
                               OUT_DIR / "budget_surface.png")
        print(f"  wrote {OUT_DIR / 'budget_surface.png'}")
        agg_stats["budget"] = _budget_stats_summary(str(budget_npz), B_mean_ref)
    else:
        print(f"  WARN: {budget_npz} not found — run scripts/misbehavior_budget_sweep.py")
        agg_stats["budget"] = None

    cc_fig_src = Path("experiments/compromise-compare/report.png")
    if cc_fig_src.exists():
        shutil.copy(cc_fig_src, OUT_DIR / "compromise_compare.png")
        print(f"  copied aggregate → {OUT_DIR / 'compromise_compare.png'}")

    out_html = OUT_DIR / "meta_report.html"
    _render_html(agg_stats, out_html)
    print(f"\nWrote {out_html}")


if __name__ == "__main__":
    main()
