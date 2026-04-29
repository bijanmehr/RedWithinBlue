"""Figures for meta_report_v3.html.

Produces three figures that v2 deferred:

  1. `hetero_sweep.png`      — ΔJ at fixed Σρ vs asymmetry; falsifies or
                                 confirms the Σρ-is-sufficient hypothesis from
                                 v2 §8.1 (R² = -2.84 model). Reads
                                 `experiments/misbehavior-budget/hetero_sweep.npz`.
  2. `spacetime_tubes.png`    — 3D isometric per setup (B/C1/C2): per-agent
                                 trajectories as 3D tubes (x, y, t), fog
                                 recession shown as transparent shading.
                                 Rolls seed-0 episodes via `_rollout_with_snapshots`.
  3. `system_diagram.png`     — Appendix A.1 architecture block diagram: env,
                                 blue actors (x N-k), red joint actor, comm
                                 merge, central critic, reward flow. Pure
                                 matplotlib, no graphviz dep.

Run:
    python scripts/meta_report_v3_figs.py

Outputs all three under experiments/meta-report/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  - registers 3D proj

import jax
import jax.numpy as jnp

from meta_report import (
    SETUPS,
    MAX_STEPS_EVAL,
    OUT_DIR,
    _load_blue,
    _load_red,
    _rollout_with_snapshots,
)
from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward, normalized_exploration_reward,
)
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
from red_within_blue.visualizer import _merge_team_belief


FIG_DIR = OUT_DIR
HETERO_NPZ = Path("experiments/misbehavior-budget/hetero_sweep.npz")


# =====================================================================
# Figure 1: hetero-sweep ΔJ vs asymmetry at fixed Σρ
# =====================================================================
def fig_hetero_sweep(out_png: Path) -> dict:
    data = np.load(HETERO_NPZ)
    sigma = data["sigma"]
    rho_a = data["rho_a"]
    rho_b = data["rho_b"]
    finals = data["finals"]              # (n_cond, n_seeds)
    clean = data["clean_finals"]         # (n_seeds,) - Σρ=0 baseline

    clean_mean = float(clean.mean())
    delta_j = clean_mean - finals.mean(axis=1)     # ΔJ in coverage-pp
    delta_j_se = finals.std(axis=1) / np.sqrt(finals.shape[1])

    asym = np.abs(rho_b - rho_a)         # |ρ_B − ρ_A| in [0, Σρ]

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.2))
    palette = {0.5: "#1f77b4", 1.0: "#d62728"}
    markers = {0.5: "o", 1.0: "s"}

    for sig in sorted(set(float(s) for s in sigma)):
        mask = np.isclose(sigma, sig)
        x = asym[mask]
        y = delta_j[mask]
        ye = delta_j_se[mask]
        order = np.argsort(x)
        ax.errorbar(
            x[order], y[order], yerr=ye[order],
            marker=markers[sig], color=palette[sig],
            linewidth=2.0, markersize=7, capsize=3,
            label=f"Σρ = {sig:.1f}",
        )
        # horizontal "sum-only-model" prediction: constant ΔJ independent of asymmetry.
        mean_flat = float(np.mean(y))
        ax.axhline(mean_flat, color=palette[sig], alpha=0.35, linestyle=":",
                   linewidth=1.5)
        ax.annotate(
            f"mean ΔJ  (Σρ={sig:.1f})  = {mean_flat:+.2f} pp",
            xy=(x.max() * 0.98, mean_flat),
            xytext=(2, 5), textcoords="offset points",
            fontsize=8, color=palette[sig],
        )

    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"asymmetry  $|\rho_B - \rho_A|$  (keeping $\rho_A + \rho_B = \Sigma\rho$ fixed)")
    ax.set_ylabel(r"$\Delta J$  =  clean coverage − compromised coverage (pp)")
    ax.set_title("Heterogeneous-ρ sweep at fixed Σρ — does the shape matter?\n"
                 "k = 2 on 3b2r-coevo, 15 seeds per point", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # annotate endpoint contrast for Σρ=1.0
    mask_sig = np.isclose(sigma, 1.0)
    x_sig = asym[mask_sig]; y_sig = delta_j[mask_sig]
    order = np.argsort(x_sig)
    xs = x_sig[order]; ys = y_sig[order]
    spread = ys.max() - ys.min()
    ax.annotate(
        f"Σρ = 1.0 spread across shapes:\n  min={ys.min():+.2f} pp  max={ys.max():+.2f} pp  (Δ={spread:.2f})",
        xy=(xs[len(xs) // 2], ys[len(ys) // 2]),
        xytext=(0.55 * xs.max(), ys.max() + 1.2),
        fontsize=9, color="#444",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e6", ec="#c09050"),
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary JSON for the HTML section
    summary = {}
    for sig in sorted(set(float(s) for s in sigma)):
        mask = np.isclose(sigma, sig)
        xs = asym[mask]; ys = delta_j[mask]
        order = np.argsort(xs)
        summary[f"sigma_{sig:.1f}"] = {
            "asymmetry": [float(v) for v in xs[order]],
            "deltaJ_pp": [float(v) for v in ys[order]],
            "mean_deltaJ": float(ys.mean()),
            "spread_deltaJ": float(ys.max() - ys.min()),
            "min_deltaJ": float(ys.min()),
            "max_deltaJ": float(ys.max()),
        }
    summary["clean_coverage_mean"] = clean_mean
    summary["n_seeds"] = int(data["n_seeds"])
    return summary


# =====================================================================
# Figure 2: spacetime tubes
# =====================================================================
_TEAM_COLOURS = {0: "#1f77b4", 1: "#d62728"}


def _rollout_paths(setup):
    cfg = ExperimentConfig.from_yaml(setup.config)
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    red_actor, red_params = (None, None)
    if setup.red_ckpt is not None:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)
    snaps, curve, paths, team_ids = _rollout_with_snapshots(
        cfg, blue_actor, blue_params, red_actor, red_params,
        seed=0, max_steps=MAX_STEPS_EVAL,
        snapshot_steps=(MAX_STEPS_EVAL,),
    )
    # paths: (T+1, N, 2)   team_ids: (N,)   snaps[-1] has final terrain + blue_ever
    return paths, team_ids, snaps[-1], curve


def fig_spacetime_tubes(out_png: Path) -> None:
    fig = plt.figure(figsize=(14.0, 5.3))
    for i, setup in enumerate(SETUPS):
        paths, team_ids, final_snap, curve = _rollout_paths(setup)
        T_plus_1 = paths.shape[0]
        ts = np.arange(T_plus_1)
        H, W = final_snap["terrain"].shape

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.set_box_aspect((W, H, T_plus_1 / 8.0))

        # terrain as a scatter at z=0 — walls dark, empty light
        terrain = final_snap["terrain"]
        wall_rc = np.argwhere(terrain == CELL_WALL)
        if len(wall_rc):
            ax.scatter(wall_rc[:, 1], H - 1 - wall_rc[:, 0], np.zeros(len(wall_rc)),
                       s=12, c="#2c2c2c", marker="s", alpha=0.35,
                       edgecolors="none", depthshade=False)

        # residual-fog cells at z=0 (red tint, transparent)
        non_wall = terrain != CELL_WALL
        unknown_final = non_wall & (~final_snap["blue_ever"])
        unk_rc = np.argwhere(unknown_final)
        if len(unk_rc):
            ax.scatter(unk_rc[:, 1], H - 1 - unk_rc[:, 0], np.zeros(len(unk_rc)),
                       s=22, c="#d62728", marker="s", alpha=0.22,
                       edgecolors="none", depthshade=False)

        # agent trajectories
        N = paths.shape[1]
        for a in range(N):
            col = paths[:, a, 1]
            row_flip = H - 1 - paths[:, a, 0]
            tid = int(team_ids[a])
            c = _TEAM_COLOURS[tid]
            ax.plot(col, row_flip, ts, color=c, linewidth=2.0,
                    alpha=0.85, zorder=3)
            # start + end markers
            ax.scatter([col[0]], [row_flip[0]], [ts[0]], c=c, s=45,
                       edgecolors="black", linewidths=0.8, marker="o",
                       depthshade=False)
            ax.scatter([col[-1]], [row_flip[-1]], [ts[-1]], c=c, s=55,
                       edgecolors="black", linewidths=0.8, marker="^",
                       depthshade=False)

        ax.set_title(f"{setup.short}\n"
                     f"final coverage = {curve[-1]:.1f}% · "
                     f"residual fog = {unknown_final.sum()} cells",
                     fontsize=9.5)
        ax.set_xlabel("x", fontsize=8, labelpad=2)
        ax.set_ylabel("y", fontsize=8, labelpad=2)
        ax.set_zlabel("t (step)", fontsize=8, labelpad=2)
        ax.set_xlim(0, W - 1); ax.set_ylim(0, H - 1); ax.set_zlim(0, T_plus_1)
        ax.view_init(elev=22, azim=-58)
        ax.tick_params(labelsize=7, pad=1)

    fig.suptitle("Spacetime tubes — per-agent trajectories as 3D tubes through "
                 "(x, y, t); terrain slice + residual-fog (red) at t = 0",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Shared: rollout capture with per-step entropy + fog masks
# =====================================================================
def _rollout_capture_entropy_uncertainty(setup, seed: int = 0, max_steps: int = None):
    """Re-run one seed-0 episode and capture, at every step:
        - per-agent Shannon entropy of the policy action distribution,
        - per-agent positions,
        - blue "currently known" mask (cells in blue's merged belief RIGHT NOW),
        - blue "ever known" mask (cumulative OR).

    Returns a dict of numpy arrays:
        paths           (T+1, N, 2)
        team_ids        (N,)
        entropy         (T+1, N)    — row 0 zeroed (pre-action)
        ever_known      (T+1, H, W) — row 0 zeroed
        current_known   (T+1, H, W) — row 0 zeroed
        terrain         (H, W)
        coverage_curve  (T+1,)      — row 0 zeroed
    """
    cfg = ExperimentConfig.from_yaml(setup.config)
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    red_actor = red_params = None
    if setup.red_ckpt is not None:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

    if max_steps is None:
        max_steps = MAX_STEPS_EVAL

    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red

    reward_fn = normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)

    @jax.jit
    def _blue_step(bp, obs, key):
        logits = blue_actor.apply(bp, obs)              # (A,)
        log_p = jax.nn.log_softmax(logits)
        p = jnp.exp(log_p)
        H = -jnp.sum(p * log_p)
        a = jax.random.categorical(key, logits)
        return a, H

    @jax.jit
    def _red_step(rp, obs_flat, key):
        logits = red_actor.apply(rp, obs_flat)          # (n_red, A)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        p = jnp.exp(log_p)
        H = -jnp.sum(p * log_p, axis=-1)                # (n_red,)
        keys = jax.random.split(key, n_red)
        a = jax.vmap(jax.random.categorical)(keys, logits)
        return a, H

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env.reset(key)
    terrain = np.asarray(state.global_state.grid.terrain)
    H_g, W_g = terrain.shape
    non_wall = terrain != CELL_WALL
    team_ids = np.asarray(state.agent_state.team_ids).copy()

    paths = [np.asarray(state.agent_state.positions).copy()]
    entropy_t = [np.zeros(n_total, dtype=np.float32)]
    ever_stack = [np.zeros((H_g, W_g), dtype=bool)]
    current_stack = [np.zeros((H_g, W_g), dtype=bool)]
    cov_curve = [0.0]

    blue_ever = np.zeros((H_g, W_g), dtype=bool)

    for step in range(1, max_steps + 1):
        key, *agent_keys = jax.random.split(key, n_total + 2)
        step_key = agent_keys[-1]
        action_dict = {}
        Hvec = np.zeros(n_total, dtype=np.float32)

        if red_actor is not None and n_red > 0:
            red_obs_flat = jnp.concatenate(
                [obs_dict[env.agents[n_blue + r]] for r in range(n_red)]
            )
            red_actions, Hred = _red_step(red_params, red_obs_flat, agent_keys[n_blue])
            for r in range(n_red):
                action_dict[env.agents[n_blue + r]] = int(red_actions[r])
                Hvec[n_blue + r] = float(Hred[r])

        for i in range(n_blue):
            a_i, Hi = _blue_step(blue_params, obs_dict[env.agents[i]], agent_keys[i])
            action_dict[env.agents[i]] = int(a_i)
            Hvec[i] = float(Hi)

        obs_dict, state, _r, dones, _info = env.step_env(step_key, state, action_dict)

        paths.append(np.asarray(state.agent_state.positions).copy())
        local_maps_np = np.asarray(state.agent_state.local_map)
        team_ids_np = np.asarray(state.agent_state.team_ids)
        blue_belief = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)

        known_now = (blue_belief != MAP_UNKNOWN) & non_wall
        blue_ever = blue_ever | known_now

        entropy_t.append(Hvec)
        current_stack.append(known_now.copy())
        ever_stack.append(blue_ever.copy())
        cov_curve.append(100.0 * blue_ever.sum() / max(1, non_wall.sum()))

        if bool(dones["__all__"]):
            break

    return {
        "paths": np.stack(paths, axis=0),
        "team_ids": team_ids,
        "entropy": np.stack(entropy_t, axis=0),
        "ever_known": np.stack(ever_stack, axis=0),
        "current_known": np.stack(current_stack, axis=0),
        "terrain": terrain,
        "coverage_curve": np.array(cov_curve, dtype=np.float32),
    }


# =====================================================================
# Figure 2b: spacetime entropy tubes
# =====================================================================
def fig_spacetime_entropy(out_png: Path) -> dict:
    """Per-agent policy entropy H_a(t) as 3D tubes, one panel per setup.

    Axes:   x = agent index (team-ordered, blues then reds)
            y = t (step)
            z = entropy  (nats)
    """
    fig = plt.figure(figsize=(14.0, 5.3))
    summary = {}

    for i, setup in enumerate(SETUPS):
        d = _rollout_capture_entropy_uncertainty(setup, seed=0)
        ent = d["entropy"]                     # (T+1, N)
        team_ids = d["team_ids"]
        T_plus_1, N = ent.shape

        # Order agents: team 0 (blue) first, then team 1 (red).
        order = np.argsort(team_ids, kind="stable")
        ent_ord = ent[:, order]
        team_ord = team_ids[order]

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.set_box_aspect((max(3, N), T_plus_1 / 12.0, 2.0))

        # draw per-agent entropy polyline through time
        x_base = np.arange(N)
        for a in range(N):
            c = _TEAM_COLOURS[int(team_ord[a])]
            ax.plot(
                np.full(T_plus_1, x_base[a]),
                np.arange(T_plus_1),
                ent_ord[:, a],
                color=c, linewidth=2.0, alpha=0.9,
            )
            # mean-entropy marker at the final step
            ax.scatter([x_base[a]], [T_plus_1 - 1], [ent_ord[-1, a]],
                       c=c, s=50, edgecolors="black", linewidths=0.7,
                       marker="^", depthshade=False)

        # team-mean ribbons at z=0 for context
        for tid in [0, 1]:
            mask = team_ord == tid
            if mask.any():
                team_mean = ent_ord[:, mask].mean(axis=1)
                ax.plot(np.full(T_plus_1, -0.6), np.arange(T_plus_1),
                        team_mean, color=_TEAM_COLOURS[int(tid)],
                        linewidth=1.4, alpha=0.45, linestyle="--")

        ax.set_title(
            f"{setup.short}\n"
            f"blue-mean H(final) = {ent_ord[-1, team_ord == 0].mean():.2f} nats"
            + ("" if not (team_ord == 1).any() else
               f" · red-mean H(final) = {ent_ord[-1, team_ord == 1].mean():.2f}"),
            fontsize=9.5,
        )
        ax.set_xlabel("agent  (blue … red)", fontsize=8, labelpad=2)
        ax.set_ylabel("t (step)", fontsize=8, labelpad=2)
        ax.set_zlabel("H(π)  (nats)", fontsize=8, labelpad=2)
        # natural entropy range: 0 .. ln(5) ≈ 1.609 for 5-action softmax
        ax.set_xlim(-1.0, N); ax.set_ylim(0, T_plus_1); ax.set_zlim(0, 1.7)
        ax.view_init(elev=22, azim=-58)
        ax.tick_params(labelsize=7, pad=1)

        summary[setup.key] = {
            "blue_final_H_mean": float(ent_ord[-1, team_ord == 0].mean()),
            "blue_final_H_std": float(ent_ord[-1, team_ord == 0].std()),
            "red_final_H_mean": (float(ent_ord[-1, team_ord == 1].mean())
                                 if (team_ord == 1).any() else None),
            "blue_traj_H_mean": float(ent_ord[:, team_ord == 0].mean()),
            "red_traj_H_mean": (float(ent_ord[:, team_ord == 1].mean())
                                if (team_ord == 1).any() else None),
        }

    fig.suptitle(
        "Spacetime entropy tubes — per-agent policy entropy H(π(·|oₜ)) through "
        "time; dashed ribbons at x = −1 are team means",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


# =====================================================================
# Figure 2c: spacetime uncertainty-manipulation voxels
# =====================================================================
def fig_spacetime_uncertainty(out_png: Path) -> dict:
    """Cells blue ONCE saw but red has now re-fogged out of blue's belief,
    plotted as voxels in (x, y, t).

    A cell is "fogged" at step t when   ever_known[t, c] = True   AND
                                         current_known[t, c] = False.

    For each setup draws one 3D panel; also draws a thin z=0 residual-fog
    pad (cells never covered by blue at all) in lighter red for reference.
    """
    fig = plt.figure(figsize=(14.0, 5.3))
    summary = {}

    for i, setup in enumerate(SETUPS):
        d = _rollout_capture_entropy_uncertainty(setup, seed=0)
        ever = d["ever_known"]
        curr = d["current_known"]
        terrain = d["terrain"]
        paths = d["paths"]
        team_ids = d["team_ids"]
        T_plus_1 = ever.shape[0]
        H_g, W_g = terrain.shape

        fog_mask = ever & (~curr)            # (T+1, H, W)
        fog_per_t = fog_mask.reshape(T_plus_1, -1).sum(axis=1)

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.set_box_aspect((W_g, H_g, T_plus_1 / 8.0))

        # residual-fog plane at z=0: cells never ever-known at final step
        non_wall = terrain != CELL_WALL
        residual = non_wall & (~ever[-1])
        rc = np.argwhere(residual)
        if len(rc):
            ax.scatter(rc[:, 1], H_g - 1 - rc[:, 0], np.zeros(len(rc)),
                       s=20, c="#d62728", marker="s", alpha=0.18,
                       edgecolors="none", depthshade=False)

        # walls as faint dark tiles
        wall_rc = np.argwhere(terrain == CELL_WALL)
        if len(wall_rc):
            ax.scatter(wall_rc[:, 1], H_g - 1 - wall_rc[:, 0],
                       np.zeros(len(wall_rc)),
                       s=10, c="#2c2c2c", marker="s", alpha=0.28,
                       edgecolors="none", depthshade=False)

        # uncertainty-manipulation events: for each step, plot fogged cells
        # as small red dots at that z.  Sub-sample by time to keep the scene
        # legible; keep every 2nd step.
        subsample = 2
        step_idx = np.arange(0, T_plus_1, subsample)
        for t in step_idx:
            fc = np.argwhere(fog_mask[t])
            if len(fc):
                ax.scatter(fc[:, 1], H_g - 1 - fc[:, 0],
                           np.full(len(fc), t),
                           s=10, c="#d62728", marker="s",
                           alpha=min(0.85, 0.35 + 0.02 * len(fc)),
                           edgecolors="none", depthshade=False)

        # overlay red-agent trajectories so the reader can see the fogging is
        # spatially near the red token(s)
        if np.any(team_ids == 1):
            ts = np.arange(T_plus_1)
            for a in np.where(team_ids == 1)[0]:
                col = paths[:, a, 1]
                row_flip = H_g - 1 - paths[:, a, 0]
                ax.plot(col, row_flip, ts, color="#7b1010", linewidth=2.2,
                        alpha=0.9)

        max_fog = int(fog_per_t.max()) if len(fog_per_t) else 0
        auc_fog = float(fog_per_t.sum())
        ax.set_title(
            f"{setup.short}\n"
            f"peak fogged = {max_fog} cells · "
            f"area-under-fog = {auc_fog:.0f} cell-steps",
            fontsize=9.5,
        )
        ax.set_xlabel("x", fontsize=8, labelpad=2)
        ax.set_ylabel("y", fontsize=8, labelpad=2)
        ax.set_zlabel("t (step)", fontsize=8, labelpad=2)
        ax.set_xlim(0, W_g - 1); ax.set_ylim(0, H_g - 1); ax.set_zlim(0, T_plus_1)
        ax.view_init(elev=22, azim=-58)
        ax.tick_params(labelsize=7, pad=1)

        summary[setup.key] = {
            "peak_fogged_cells": max_fog,
            "area_under_fog_cellsteps": auc_fog,
            "final_fogged_cells": int(fog_mask[-1].sum()),
        }

    fig.suptitle(
        "Spacetime uncertainty manipulation — red voxels = cells blue once "
        "saw but red has re-fogged; thick dark-red curve = red trajectory",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


# =====================================================================
# Figure 3: system diagram
# =====================================================================
def _box(ax, x, y, w, h, text, fc="#eef5ff", ec="#1f77b4", fontsize=9, zorder=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor=fc, edgecolor=ec, linewidth=1.4,
                                zorder=zorder))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1)


def _arrow(ax, x0, y0, x1, y1, label="", color="#555", style="-|>", lw=1.4, pad=0.0,
           label_offset=(0, 0.015), zorder=2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                  color=color, linewidth=lw, mutation_scale=14,
                                  zorder=zorder))
    if label:
        ax.text((x0 + x1) / 2 + label_offset[0],
                (y0 + y1) / 2 + label_offset[1],
                label, fontsize=7.5, ha="center", color=color, zorder=zorder + 1)


def fig_system_diagram(out_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Central env block
    _box(ax, 0.50, 0.80, 0.30, 0.14,
         "GridCommEnv\n(state: terrain, positions,\nlocal_map, comm graph)",
         fc="#fff6e6", ec="#e69500", fontsize=9.5)

    # Blue actors (N-k shared instances)
    _box(ax, 0.15, 0.55, 0.20, 0.12,
         "Blue Actor  ×(N−k)\n(shared MLP\nπ(a | oᵢ))",
         fc="#eef5ff", ec="#1f77b4")

    # Comm merge
    _box(ax, 0.50, 0.55, 0.18, 0.10,
         "Comm merge (OR)\nlocal_map broadcast",
         fc="#f0f0f0", ec="#777")

    # Red joint actor
    _box(ax, 0.85, 0.55, 0.20, 0.12,
         "Joint Red Actor\n(centralised MLP\nφ(a₁,…,aₖ | o₁…oₖ))",
         fc="#fbe9ea", ec="#d62728")

    # Central critic
    _box(ax, 0.50, 0.30, 0.28, 0.12,
         "Central Critic  V(sₜ)\n(joint obs → ℝ,\ntraining only)",
         fc="#eafaea", ec="#2ca02c")

    # Reward shaping
    _box(ax, 0.15, 0.15, 0.22, 0.10,
         "Blue reward R\n(normalized_competitive)",
         fc="#eef5ff", ec="#1f77b4")
    _box(ax, 0.85, 0.15, 0.22, 0.10,
         "Red reward\n(= −R, zero-sum)",
         fc="#fbe9ea", ec="#d62728")

    # Environment → obs paths (down)
    _arrow(ax, 0.38, 0.79, 0.22, 0.62, label="oᵢ, local_map")
    _arrow(ax, 0.62, 0.79, 0.78, 0.62, label="oⱼ, local_map")
    _arrow(ax, 0.50, 0.73, 0.50, 0.60, label="comm graph")

    # Actors → actions back (up)
    _arrow(ax, 0.22, 0.62, 0.38, 0.81, label="aᵢ", color="#1f77b4")
    _arrow(ax, 0.78, 0.62, 0.62, 0.81, label="aⱼ (joint)", color="#d62728")

    # Env → comm merge (down)
    _arrow(ax, 0.35, 0.55, 0.41, 0.55, style="-|>")
    _arrow(ax, 0.65, 0.55, 0.59, 0.55, style="-|>")

    # Env → critic (training signal)
    _arrow(ax, 0.50, 0.73, 0.50, 0.37, label="sₜ (joint obs)  [CTDE]",
           color="#2ca02c", style="-|>", label_offset=(0.14, 0))
    # Critic → blue actor (policy gradient)
    _arrow(ax, 0.36, 0.30, 0.22, 0.49, label="∇ᴬ  advantage", color="#2ca02c",
           style="-|>", label_offset=(-0.04, -0.02))

    # Env → rewards → actors
    _arrow(ax, 0.38, 0.77, 0.22, 0.21, label="R", color="#1f77b4")
    _arrow(ax, 0.62, 0.77, 0.78, 0.21, label="−R", color="#d62728")

    # Legend / caption
    ax.text(0.50, 0.04,
            "Two-team, one-env, CTDE training loop — blue actors share an MLP "
            "and critic with oracle-obs; red is a centralised joint actor over "
            "k compromised agents; reward is exact zero-sum.",
            ha="center", fontsize=8.5, color="#444", style="italic")

    ax.set_title("System architecture — blue CTDE actor/critic + centralised "
                 "joint red, zero-sum coupling",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = {}

    spacetime_png = FIG_DIR / "spacetime_tubes.png"
    print("[1/5] spacetime_tubes")
    fig_spacetime_tubes(spacetime_png)
    print(f"  wrote {spacetime_png}")

    entropy_png = FIG_DIR / "spacetime_entropy.png"
    print("[2/5] spacetime_entropy")
    out["entropy"] = fig_spacetime_entropy(entropy_png)
    (FIG_DIR / "spacetime_entropy_summary.json").write_text(
        json.dumps(out["entropy"], indent=2))
    print(f"  wrote {entropy_png}")

    uncertainty_png = FIG_DIR / "spacetime_uncertainty.png"
    print("[3/5] spacetime_uncertainty")
    out["uncertainty"] = fig_spacetime_uncertainty(uncertainty_png)
    (FIG_DIR / "spacetime_uncertainty_summary.json").write_text(
        json.dumps(out["uncertainty"], indent=2))
    print(f"  wrote {uncertainty_png}")

    diagram_png = FIG_DIR / "system_diagram.png"
    print("[4/5] system_diagram")
    fig_system_diagram(diagram_png)
    print(f"  wrote {diagram_png}")

    if HETERO_NPZ.exists():
        hetero_png = FIG_DIR / "hetero_sweep.png"
        print("[5/5] hetero_sweep")
        out["hetero"] = fig_hetero_sweep(hetero_png)
        (FIG_DIR / "hetero_summary.json").write_text(json.dumps(out["hetero"], indent=2))
        print(f"  wrote {hetero_png}")
        print(f"  wrote {FIG_DIR / 'hetero_summary.json'}")
    else:
        print(f"[5/5] skipped hetero_sweep — {HETERO_NPZ} not found")


if __name__ == "__main__":
    main()
