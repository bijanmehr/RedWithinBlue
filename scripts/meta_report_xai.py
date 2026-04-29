"""Saliency-based XAI on 3b2r-coevo.

Rolls canonical seed-0 episodes for the three setups (B, C1, C2). At every
step computes the gradient of the *sampled* action's logit w.r.t. the agent's
observation, then aggregates by obs-block. Tests the sabotage-as-delay
hypothesis: if red is parking/hoarding, its saliency should be dominated by
own-position and minimal on the seen-field; blue, actively exploring, should
look the other way around.

Obs schema for `local_obs=true, view_radius=1`  →  obs_dim = 23:

    [ 0:9 )  scan       — 3×3 terrain sensor frame this step
    [ 9:18)  seen       — 3×3 known / unknown window from local_map
    [18    ] map_frac   — global fraction-of-grid known (scalar)
    [19:21) norm_pos    — own (row, col) / (H, W)
    [21    ] uid        — normalised agent uid
    [22    ] team_id    — own team label  (always self → blue=0 or red=1)

For the joint red controller (C2: 2 reds), the input is a concat of length
46 = 2 × 23. We split saliency into ``own`` and ``cross`` contributions per
red, which directly measures how much each red's chosen action depends on
the *other* red's observation — a quantitative read on whether the central
controller does meaningful coordination or just reduces to two independent
heads.

Outputs (under ``experiments/meta-report/``):

    xai_block_stack_{B,C1,C2}.png  — per-agent stacked area over time
    xai_block_team_means.png       — team-mean bars across setups
    xai_red_self_vs_cross.png      — C2 joint-red own vs cross attention
    xai_spatial_seen.png           — 3×3 seen-field saliency, B/C1/C2 × blue/red
    xai_summary.json               — headline numbers for the report

Run:
    python scripts/meta_report_xai.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from meta_report import (
    SETUPS,
    MAX_STEPS_EVAL,
    OUT_DIR,
    _load_blue,
    _load_red,
)
from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward,
    normalized_exploration_reward,
)


# Obs-block schema (must match env._build_obs_array for view_radius=1 + local_obs=true)
BLOCKS = [
    ("scan",     0,  9),    # 3×3 terrain window
    ("seen",     9,  18),   # 3×3 known/unknown window
    ("map_frac", 18, 19),   # global fraction known
    ("norm_pos", 19, 21),   # own (row, col) / (H, W)
    ("uid",      21, 22),   # uid (normalised)
    ("team_id",  22, 23),   # team label
]
BLOCK_COLOURS = {
    "scan":     "#5b8def",   # blue
    "seen":     "#23a47e",   # green
    "map_frac": "#f2a73b",   # amber
    "norm_pos": "#d6594d",   # red
    "uid":      "#9b6dd7",   # purple
    "team_id":  "#7d7d7d",   # grey
}


def _block_attribution(grad_obs: np.ndarray) -> np.ndarray:
    """Reduce a length-23 gradient to length-len(BLOCKS) by mean-|.| per block."""
    out = np.zeros(len(BLOCKS), dtype=np.float64)
    for i, (_, lo, hi) in enumerate(BLOCKS):
        out[i] = float(np.mean(np.abs(grad_obs[lo:hi])))
    return out


def _rollout_with_saliency(setup, seed: int = 0):
    """Roll one episode and capture per-agent block-attribution at every step.

    Returns
    -------
    info : dict
      paths            (N, T+1, 2) int  positions per step
      team_ids         (N,) int
      block_attr       (N, T, n_blocks) float  per-step |∂logit/∂obs| per block
      red_self_attr    (n_red, T, n_blocks) or None  C2 only — own-slice attention
      red_cross_attr   (n_red, T, n_blocks) or None  C2 only — other-red-slice attention
      seen_spatial     (N, 9) float  time-mean |∂logit/∂obs[seen]| reshaped flat
    """
    cfg = ExperimentConfig.from_yaml(setup.config)
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red

    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    if n_red > 0:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)
    else:
        red_actor, red_params = None, None

    reward_fn = normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)
    obs_dim = env.obs_dim
    assert obs_dim == 23, f"Expected obs_dim=23 for local_obs=true,view_radius=1; got {obs_dim}"

    @jax.jit
    def _blue_logit_grad(bp, obs, action):
        """∂logit_a/∂obs for the chosen action. Returns (logits, grad)."""
        def f(o):
            return blue_actor.apply(bp, o)[action]
        return blue_actor.apply(bp, obs), jax.grad(f)(obs)

    @jax.jit
    def _red_logit_grads(rp, obs_flat, actions):
        """For each red r, ∂logits[r, actions[r]]/∂obs_flat (length n_red*obs_dim)."""
        def f_r(r):
            def g(o):
                return red_actor.apply(rp, o)[r, actions[r]]
            return jax.grad(g)(obs_flat)
        # Vectorise over r — but actions has dynamic value; just python-loop over n_red
        grads = jnp.stack([
            jax.grad(lambda o, rr=r: red_actor.apply(rp, o)[rr, actions[rr]])(obs_flat)
            for r in range(n_red)
        ])
        full_logits = red_actor.apply(rp, obs_flat)
        return full_logits, grads

    @jax.jit
    def _blue_sample(bp, obs, key):
        return jax.random.categorical(key, blue_actor.apply(bp, obs))

    @jax.jit
    def _red_sample(rp, obs_flat, key):
        logits = red_actor.apply(rp, obs_flat)
        keys = jax.random.split(key, n_red)
        return jax.vmap(jax.random.categorical)(keys, logits)

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env.reset(key)
    team_ids = np.asarray(state.agent_state.team_ids).copy()
    paths = [np.asarray(state.agent_state.positions).copy()]
    block_attr = []                                     # list of (N, n_blocks)
    red_self_attr = [] if n_red > 0 else None
    red_cross_attr = [] if n_red > 0 else None
    seen_grad_acc = np.zeros((n_total, 9), dtype=np.float64)
    seen_grad_count = np.zeros(n_total, dtype=np.int64)

    max_steps = cfg.env.max_steps
    for step in range(1, max_steps + 1):
        key, *subkeys = jax.random.split(key, n_total + 2)
        step_key = subkeys[-1]

        # 1. Blue actions + per-blue saliency
        blue_actions_np = np.zeros(n_blue, dtype=np.int32)
        per_step_blocks = np.zeros((n_total, len(BLOCKS)), dtype=np.float64)
        per_step_seen = np.zeros((n_total, 9), dtype=np.float64)
        for i in range(n_blue):
            a_i = int(_blue_sample(blue_params, obs_dict[env.agents[i]], subkeys[i]))
            blue_actions_np[i] = a_i
            _, g = _blue_logit_grad(blue_params, obs_dict[env.agents[i]], a_i)
            g_np = np.asarray(g)
            per_step_blocks[i] = _block_attribution(g_np)
            per_step_seen[i] = np.abs(g_np[9:18])

        # 2. Red actions + per-red saliency
        red_actions_np = np.zeros(n_red, dtype=np.int32)
        if n_red > 0:
            red_obs_flat = jnp.concatenate(
                [obs_dict[env.agents[n_blue + r]] for r in range(n_red)]
            )
            red_actions = _red_sample(red_params, red_obs_flat, subkeys[n_blue])
            red_actions_np = np.asarray(red_actions).astype(np.int32)
            _, red_grads = _red_logit_grads(red_params, red_obs_flat, red_actions_np)
            red_grads_np = np.asarray(red_grads)        # (n_red, n_red*obs_dim)
            self_step = np.zeros((n_red, len(BLOCKS)), dtype=np.float64)
            cross_step = np.zeros((n_red, len(BLOCKS)), dtype=np.float64)
            for r in range(n_red):
                own_slice = red_grads_np[r, r * obs_dim:(r + 1) * obs_dim]
                per_step_blocks[n_blue + r] = _block_attribution(own_slice)
                per_step_seen[n_blue + r] = np.abs(own_slice[9:18])
                self_step[r] = _block_attribution(own_slice)
                if n_red >= 2:
                    cross_idx = np.concatenate([
                        np.arange(o * obs_dim, (o + 1) * obs_dim)
                        for o in range(n_red) if o != r
                    ])
                    cross_slice = red_grads_np[r, cross_idx]
                    cross_per_other = cross_slice.reshape(n_red - 1, obs_dim).mean(axis=0)
                    cross_step[r] = _block_attribution(cross_per_other)
            red_self_attr.append(self_step)
            red_cross_attr.append(cross_step)

        block_attr.append(per_step_blocks)
        seen_grad_acc += per_step_seen
        seen_grad_count += 1

        # 3. step env
        action_dict = {env.agents[i]: int(blue_actions_np[i]) for i in range(n_blue)}
        for r in range(n_red):
            action_dict[env.agents[n_blue + r]] = int(red_actions_np[r])
        obs_dict, state, _r, dones, _info = env.step_env(step_key, state, action_dict)
        paths.append(np.asarray(state.agent_state.positions).copy())
        if bool(dones["__all__"]):
            break

    seen_spatial = (seen_grad_acc / np.maximum(1, seen_grad_count)[:, None])

    return {
        "paths": np.stack(paths, axis=0),
        "team_ids": team_ids,
        "block_attr": np.stack(block_attr, axis=0).transpose(1, 0, 2),  # (N, T, B)
        "red_self_attr": (np.stack(red_self_attr, axis=0).transpose(1, 0, 2)
                         if red_self_attr else None),
        "red_cross_attr": (np.stack(red_cross_attr, axis=0).transpose(1, 0, 2)
                          if red_cross_attr else None),
        "seen_spatial": seen_spatial,
        "n_blue": n_blue,
        "n_red": n_red,
    }


# ===================================================================
# Figure 1: per-agent stacked area of block-attribution over time
# ===================================================================
def fig_block_stack(rollouts: dict, out_png: Path) -> None:
    """One row per setup, one panel per agent. Stacked area = block share over t."""
    n_rows = len(rollouts)
    max_agents = max(r["block_attr"].shape[0] for r in rollouts.values())
    fig, axes = plt.subplots(
        n_rows, max_agents,
        figsize=(2.8 * max_agents, 2.6 * n_rows),
        squeeze=False,
        sharex=True,
    )
    setup_keys = list(rollouts.keys())

    for r, key in enumerate(setup_keys):
        roll = rollouts[key]
        attr = roll["block_attr"]                       # (N, T, B)
        team_ids = roll["team_ids"]
        N, T, B = attr.shape
        # Normalise per-step so blocks sum to 1
        norm = attr / np.maximum(1e-12, attr.sum(axis=-1, keepdims=True))
        steps = np.arange(T)
        for i in range(max_agents):
            ax = axes[r, i]
            if i >= N:
                ax.axis("off")
                continue
            stack_data = norm[i].T                      # (B, T)
            colours = [BLOCK_COLOURS[name] for name, *_ in BLOCKS]
            labels = [name for name, *_ in BLOCKS]
            ax.stackplot(steps, stack_data, labels=labels, colors=colours, alpha=0.9)
            ax.set_xlim(0, T - 1)
            ax.set_ylim(0, 1)
            tid = int(team_ids[i])
            tag = "blue" if tid == 0 else "red"
            ax.set_title(f"{key} — agent {i} ({tag})", fontsize=9)
            if i == 0:
                ax.set_ylabel("share of |∂logit/∂obs|", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("step", fontsize=8)
            ax.tick_params(labelsize=7)
        # Single legend on the right of each row
        axes[r, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                          fontsize=7, frameon=False)
    fig.suptitle("Per-agent saliency by obs-block (canonical seed=0 episode)\n"
                 "Stack = share of total absolute-gradient mass per block",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Figure 2: team-mean attribution bars across setups (the headline test)
# ===================================================================
def fig_team_means(rollouts: dict, out_png: Path) -> dict:
    """Side-by-side grouped bars: B/C1/C2 × {blue-mean, red-mean} × n_blocks.

    The hypothesis test: at C2, red's bars should peak on norm_pos / scan
    (parking, react-to-current-frame) and have minimal seen mass; blue's
    bars should peak on seen.
    """
    setups = list(rollouts.keys())
    block_names = [name for name, *_ in BLOCKS]
    summary = {"setups": setups, "blocks": block_names, "blue_mean": {}, "red_mean": {}}

    fig, axes = plt.subplots(1, len(setups), figsize=(5.2 * len(setups), 4.6),
                             sharey=True)
    if len(setups) == 1:
        axes = [axes]
    bar_w = 0.38
    x = np.arange(len(BLOCKS))

    for ax, setup_key in zip(axes, setups):
        roll = rollouts[setup_key]
        attr = roll["block_attr"]                       # (N, T, B)
        team_ids = roll["team_ids"]
        n_blue = roll["n_blue"]
        n_red = roll["n_red"]

        # mean across (agent in team, time) of |grad|/sum -> normalised share
        norm = attr / np.maximum(1e-12, attr.sum(axis=-1, keepdims=True))
        blue_mask = team_ids == 0
        red_mask = team_ids == 1

        blue_share = norm[blue_mask].mean(axis=(0, 1)) if blue_mask.any() else np.zeros(len(BLOCKS))
        red_share = norm[red_mask].mean(axis=(0, 1)) if red_mask.any() else np.zeros(len(BLOCKS))

        ax.bar(x - bar_w / 2, blue_share, bar_w,
               color=[BLOCK_COLOURS[b] for b in block_names],
               edgecolor="#1f4e8c", linewidth=1.4, label="blue mean")
        ax.bar(x + bar_w / 2, red_share, bar_w,
               color=[BLOCK_COLOURS[b] for b in block_names],
               edgecolor="#a6231f", linewidth=1.4, hatch="//", label="red mean")
        ax.set_xticks(x)
        ax.set_xticklabels(block_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{setup_key}  ({n_blue} blue, {n_red} red)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("normalised attribution\n(share of |∂logit/∂obs|)", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        summary["blue_mean"][setup_key] = {b: float(v) for b, v in zip(block_names, blue_share)}
        summary["red_mean"][setup_key] = {b: float(v) for b, v in zip(block_names, red_share)}

    fig.suptitle("Team-mean saliency by obs-block — sabotage-as-delay vs info-theft\n"
                 "Solid edge = blue,  hatched edge = red", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


# ===================================================================
# Figure 3: red joint-policy own vs cross attention (C2 only)
# ===================================================================
def fig_red_self_vs_cross(rollouts: dict, out_png: Path) -> dict:
    if "C2" not in rollouts:
        return {}
    roll = rollouts["C2"]
    if roll["red_self_attr"] is None:
        return {}
    self_attr = roll["red_self_attr"]                   # (n_red, T, n_blocks)
    cross_attr = roll["red_cross_attr"]
    n_red, T, B = self_attr.shape
    block_names = [name for name, *_ in BLOCKS]

    # mean magnitude (not normalised) — comparable across own/cross
    self_mean = self_attr.mean(axis=(0, 1))             # (n_blocks,)
    cross_mean = cross_attr.mean(axis=(0, 1))
    own_total = float(self_mean.sum())
    cross_total = float(cross_mean.sum())
    cross_share = cross_total / max(1e-12, own_total + cross_total)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(B)
    bar_w = 0.38
    ax1.bar(x - bar_w / 2, self_mean, bar_w, color="#a6231f",
            edgecolor="black", linewidth=0.6, label="own slice")
    ax1.bar(x + bar_w / 2, cross_mean, bar_w, color="#dc8a86",
            edgecolor="black", linewidth=0.6, hatch="//", label="other-red slice")
    ax1.set_xticks(x); ax1.set_xticklabels(block_names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("mean |∂logit/∂obs| per dim", fontsize=9)
    ax1.set_title("C2 joint-red — own vs cross attention by block", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

    # Right panel: cross-share over time
    self_t = self_attr.sum(axis=-1)                     # (n_red, T)
    cross_t = cross_attr.sum(axis=-1)
    cross_frac = cross_t / np.maximum(1e-12, self_t + cross_t)
    steps = np.arange(T)
    palette = ["#a6231f", "#dc8a86"]
    for r in range(n_red):
        ax2.plot(steps, cross_frac[r], color=palette[r % len(palette)],
                 linewidth=1.7, alpha=0.95, label=f"red {r}")
    ax2.axhline(0.5, color="k", linewidth=0.6, alpha=0.4, linestyle="--")
    ax2.set_xlabel("step", fontsize=9)
    ax2.set_ylabel("cross-share  =  |grad on other| / (|own| + |other|)", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(0, T - 1)
    ax2.set_title(f"Cross-attention share over time  (mean = {cross_frac.mean():.2f})",
                  fontsize=10)
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    fig.suptitle("Joint-red controller — does it actually couple the two reds?\n"
                 "If cross ≪ own across blocks AND time, the joint policy is effectively two independent heads.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "own_mean_per_block": {b: float(v) for b, v in zip(block_names, self_mean)},
        "cross_mean_per_block": {b: float(v) for b, v in zip(block_names, cross_mean)},
        "cross_share_total": cross_share,
        "cross_share_mean_over_time": float(cross_frac.mean()),
    }


# ===================================================================
# Figure 4: spatial saliency on the 3×3 seen-field window
# ===================================================================
def fig_spatial_seen(rollouts: dict, out_png: Path) -> None:
    setups = list(rollouts.keys())
    n_rows = len(setups)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.2, 2.9 * n_rows), squeeze=False)
    setup_idx = {k: i for i, k in enumerate(setups)}

    for setup_key in setups:
        r = setup_idx[setup_key]
        roll = rollouts[setup_key]
        attr = roll["seen_spatial"]                     # (N, 9)
        team_ids = roll["team_ids"]
        for col, (mask_team, label, cmap) in enumerate([
            (team_ids == 0, "blue mean", "Blues"),
            (team_ids == 1, "red mean", "Reds"),
        ]):
            ax = axes[r, col]
            if not mask_team.any():
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="grey")
                ax.set_title(f"{setup_key} {label}", fontsize=9)
                ax.axis("off"); continue
            grid = attr[mask_team].mean(axis=0).reshape(3, 3)
            im = ax.imshow(grid, cmap=cmap, origin="upper",
                          vmin=0, vmax=max(1e-6, grid.max()))
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{grid[i, j]:.2f}",
                           ha="center", va="center",
                           color="white" if grid[i, j] > grid.max() * 0.5 else "black",
                           fontsize=8)
            ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=2.0))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{setup_key} {label}", fontsize=9)

    fig.suptitle("Spatial seen-field saliency  (3×3 window centered on agent)\n"
                 "Black square = own cell.  Higher = policy weights that cell more.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Roll all three setups
    rollouts = {}
    for setup in SETUPS:
        print(f"[xai] rolling {setup.key}  ({setup.short})")
        rollouts[setup.key] = _rollout_with_saliency(setup, seed=0)
        roll = rollouts[setup.key]
        T = roll["block_attr"].shape[1]
        print(f"      captured T={T} steps  N={roll['block_attr'].shape[0]}")

    # Figures
    fig_block_stack(rollouts, OUT_DIR / "xai_block_stack.png")
    block_summary = fig_team_means(rollouts, OUT_DIR / "xai_block_team_means.png")
    cross_summary = fig_red_self_vs_cross(rollouts, OUT_DIR / "xai_red_self_vs_cross.png")
    fig_spatial_seen(rollouts, OUT_DIR / "xai_spatial_seen.png")

    summary = {
        "blocks": [name for name, *_ in BLOCKS],
        "block_team_means": block_summary,
        "joint_red_cross": cross_summary,
        "method": "vanilla input-saliency, |∂logit_a/∂obs| with a = sampled action",
        "rollout": "canonical seed=0, max_steps from each config",
    }
    out_json = OUT_DIR / "xai_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[xai] wrote {out_json}")


if __name__ == "__main__":
    main()
