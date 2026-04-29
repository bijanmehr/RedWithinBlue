"""Integrated-gradients version of the saliency probe + 5-seed extension.

Vanilla saliency (``meta_report_xai.py``) is well-known to be noisy on
ReLU MLPs. Integrated gradients (Sundararajan et al. 2017) accumulate the
gradient along the straight-line path from a baseline input to the actual
input, which gives much cleaner per-step traces and a clean theoretical
interpretation: the result satisfies *completeness* (sum over input dims
recovers the model output difference from the baseline).

For each step:
    IG_i(o) = (o_i - baseline_i) · ∫_{α=0}^{1} ∂ logit_{a*}((1-α)·baseline + α·o) / ∂o_i  dα
       ≈ (o_i - baseline_i) · (1/M) Σ_{m=1..M} ∂ logit_{a*}(α_m·o + (1-α_m)·baseline) / ∂o_i

Baseline is chosen as the zero vector (matches occlusion baseline). M=32 steps.

5 seeds × 3 setups; mean shares averaged across (seeds × agents × time).

Outputs (under experiments/meta-report/):
  xai_ig_team_means.png          — same shape as xai_block_team_means.png
                                    but IG instead of saliency, 5 seeds
  xai_ig_vs_saliency.png         — direct comparison of the two methods
  xai_ig_summary.json
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

from meta_report import SETUPS, OUT_DIR, _load_blue, _load_red
from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward, normalized_exploration_reward,
)


SEEDS = (0, 1, 2, 3, 4)
IG_STEPS = 32
BLOCKS = [
    ("scan",     0,  9),
    ("seen",     9,  18),
    ("map_frac", 18, 19),
    ("norm_pos", 19, 21),
    ("uid",      21, 22),
    ("team_id",  22, 23),
]
BLOCK_NAMES = [b[0] for b in BLOCKS]
BLOCK_COLOURS = {
    "scan": "#5b8def", "seen": "#23a47e", "map_frac": "#f2a73b",
    "norm_pos": "#d6594d", "uid": "#9b6dd7", "team_id": "#7d7d7d",
}


def _block_attribution(grad_obs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(BLOCKS), dtype=np.float64)
    for i, (_, lo, hi) in enumerate(BLOCKS):
        out[i] = float(np.mean(np.abs(grad_obs[lo:hi])))
    return out


def _rollout_ig(setup, seed: int):
    """Roll one episode and capture IG attribution by block at every step."""
    cfg = ExperimentConfig.from_yaml(setup.config)
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red
    obs_dim = 23

    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    if n_red > 0:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

    reward_fn = normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)

    alphas = jnp.linspace(0.0, 1.0, IG_STEPS + 1)[1:]   # avoid baseline-only

    @jax.jit
    def _blue_ig(bp, obs, action):
        baseline = jnp.zeros_like(obs)
        def grad_at(alpha):
            interp = baseline + alpha * (obs - baseline)
            return jax.grad(lambda o: blue_actor.apply(bp, o)[action])(interp)
        # Average gradient along the path
        avg_grad = jax.vmap(grad_at)(alphas).mean(axis=0)
        return (obs - baseline) * avg_grad

    @jax.jit
    def _red_ig(rp, obs_flat, actions):
        baseline = jnp.zeros_like(obs_flat)
        # For each red r, IG of logit[r, action[r]] wrt obs_flat
        def per_red(r):
            def grad_at(alpha):
                interp = baseline + alpha * (obs_flat - baseline)
                return jax.grad(lambda o: red_actor.apply(rp, o)[r, actions[r]])(interp)
            avg_grad = jax.vmap(grad_at)(alphas).mean(axis=0)
            return (obs_flat - baseline) * avg_grad
        return jnp.stack([per_red(r) for r in range(n_red)])

    @jax.jit
    def _blue_sample(bp, obs, key):
        return jax.random.categorical(key, blue_actor.apply(bp, obs))

    @jax.jit
    def _red_sample(rp, obs_flat, key):
        keys = jax.random.split(key, n_red)
        return jax.vmap(jax.random.categorical)(keys, red_actor.apply(rp, obs_flat))

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env.reset(key)
    team_ids = np.asarray(state.agent_state.team_ids).copy()
    block_attr = []                                      # per-step (N, n_blocks)
    seen_grad_acc = np.zeros((n_total, 9), dtype=np.float64)
    seen_grad_count = np.zeros(n_total, dtype=np.int64)

    max_steps = cfg.env.max_steps
    for step in range(1, max_steps + 1):
        key, *subkeys = jax.random.split(key, n_total + 2)
        step_key = subkeys[-1]

        per_step_blocks = np.zeros((n_total, len(BLOCKS)), dtype=np.float64)
        per_step_seen = np.zeros((n_total, 9), dtype=np.float64)
        blue_actions_np = np.zeros(n_blue, dtype=np.int32)
        for i in range(n_blue):
            a_i = int(_blue_sample(blue_params, obs_dict[env.agents[i]], subkeys[i]))
            blue_actions_np[i] = a_i
            ig = np.asarray(_blue_ig(blue_params, obs_dict[env.agents[i]], a_i))
            per_step_blocks[i] = _block_attribution(ig)
            per_step_seen[i] = np.abs(ig[9:18])

        red_actions_np = np.zeros(n_red, dtype=np.int32)
        if n_red > 0:
            red_obs_flat = jnp.concatenate(
                [obs_dict[env.agents[n_blue + r]] for r in range(n_red)]
            )
            red_actions = _red_sample(red_params, red_obs_flat, subkeys[n_blue])
            red_actions_np = np.asarray(red_actions).astype(np.int32)
            ig_red = np.asarray(_red_ig(red_params, red_obs_flat, red_actions_np))   # (n_red, n_red*obs_dim)
            for r in range(n_red):
                own_slice = ig_red[r, r * obs_dim:(r + 1) * obs_dim]
                per_step_blocks[n_blue + r] = _block_attribution(own_slice)
                per_step_seen[n_blue + r] = np.abs(own_slice[9:18])

        block_attr.append(per_step_blocks)
        seen_grad_acc += per_step_seen
        seen_grad_count += 1

        action_dict = {env.agents[i]: int(blue_actions_np[i]) for i in range(n_blue)}
        for r in range(n_red):
            action_dict[env.agents[n_blue + r]] = int(red_actions_np[r])
        obs_dict, state, _r, dones, _info = env.step_env(step_key, state, action_dict)
        if bool(dones["__all__"]):
            break

    return {
        "team_ids": team_ids,
        "block_attr": np.stack(block_attr, axis=0).transpose(1, 0, 2),  # (N, T, B)
        "n_blue": n_blue, "n_red": n_red,
        "seen_spatial": seen_grad_acc / np.maximum(1, seen_grad_count)[:, None],
    }


def fig_ig_team_means(rollouts_by_setup, out_png: Path) -> dict:
    """Same shape as the saliency team-means figure but uses IG and averages
    over 5 seeds."""
    setups = list(rollouts_by_setup.keys())
    block_names = BLOCK_NAMES
    fig, axes = plt.subplots(1, len(setups), figsize=(5.2 * len(setups), 4.6),
                             sharey=True)
    if len(setups) == 1: axes = [axes]
    summary = {"blue_mean": {}, "red_mean": {}}

    for ax, sk in zip(axes, setups):
        seeds_data = rollouts_by_setup[sk]                  # list of dicts (1 per seed)
        n_blue = seeds_data[0]["n_blue"]; n_red = seeds_data[0]["n_red"]
        team_ids = seeds_data[0]["team_ids"]
        # Stack across seeds → (n_seeds, N, T, B), but Ts may differ; use share per-(seed,agent,t)
        blue_shares = []; red_shares = []
        for d in seeds_data:
            attr = d["block_attr"]                          # (N, T, B)
            norm = attr / np.maximum(1e-12, attr.sum(axis=-1, keepdims=True))
            if (team_ids == 0).any():
                blue_shares.append(norm[team_ids == 0].mean(axis=(0, 1)))
            if (team_ids == 1).any():
                red_shares.append(norm[team_ids == 1].mean(axis=(0, 1)))

        bs = np.stack(blue_shares).mean(axis=0) if blue_shares else np.zeros(len(BLOCKS))
        bs_se = (np.stack(blue_shares).std(axis=0) / np.sqrt(max(1, len(blue_shares)))
                 if blue_shares else np.zeros(len(BLOCKS)))
        rs = np.stack(red_shares).mean(axis=0) if red_shares else np.zeros(len(BLOCKS))
        rs_se = (np.stack(red_shares).std(axis=0) / np.sqrt(max(1, len(red_shares)))
                 if red_shares else np.zeros(len(BLOCKS)))

        x = np.arange(len(BLOCKS)); bar_w = 0.38
        ax.bar(x - bar_w / 2, bs, bar_w, yerr=bs_se,
               color=[BLOCK_COLOURS[b] for b in block_names],
               edgecolor="#1f4e8c", linewidth=1.4, capsize=3, label="blue")
        ax.bar(x + bar_w / 2, rs, bar_w, yerr=rs_se,
               color=[BLOCK_COLOURS[b] for b in block_names],
               edgecolor="#a6231f", linewidth=1.4, hatch="//", capsize=3, label="red")
        ax.set_xticks(x); ax.set_xticklabels(block_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{sk}  ({n_blue} blue, {n_red} red)", fontsize=10)
        if ax is axes[0]: ax.set_ylabel("IG attribution share\n(5-seed mean)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        summary["blue_mean"][sk] = {b: float(v) for b, v in zip(block_names, bs)}
        summary["red_mean"][sk] = {b: float(v) for b, v in zip(block_names, rs)}

    fig.suptitle("Integrated-gradients team-mean attribution by obs-block (5 seeds)\n"
                 "Cleaner per-step gradient than vanilla saliency",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def fig_ig_vs_saliency(ig_summary: dict, sal_path: Path, out_png: Path) -> None:
    """Side-by-side comparison: vanilla saliency from xai_summary.json vs IG."""
    sal_summary = json.loads(sal_path.read_text())
    sal_blue = sal_summary["block_team_means"]["blue_mean"]
    sal_red = sal_summary["block_team_means"]["red_mean"]
    setups = list(sal_blue.keys())

    fig, axes = plt.subplots(2, len(setups), figsize=(4.8 * len(setups), 7.2),
                             sharey="row")
    if len(setups) == 1: axes = axes[:, None]
    block_names = BLOCK_NAMES
    x = np.arange(len(BLOCKS))

    for col, sk in enumerate(setups):
        # Top row: blue
        b_sal = np.array([sal_blue[sk][b] for b in block_names])
        b_ig = np.array([ig_summary["blue_mean"][sk][b] for b in block_names])
        ax = axes[0, col]
        bw = 0.4
        ax.bar(x - bw / 2, b_sal, bw, color="#888", edgecolor="black",
               label="vanilla saliency (1 seed)")
        ax.bar(x + bw / 2, b_ig, bw, color="#1f77b4", edgecolor="black",
               label="integrated grads (5 seeds)")
        ax.set_xticks(x); ax.set_xticklabels(block_names, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{sk}  blue", fontsize=10)
        if col == 0: ax.set_ylabel("attribution share", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=7)

        # Bottom row: red
        ax = axes[1, col]
        r_sal = np.array([sal_red[sk][b] for b in block_names]) if sk in sal_red else np.zeros(len(BLOCKS))
        r_ig = np.array([ig_summary["red_mean"][sk][b] for b in block_names]) if sk in ig_summary["red_mean"] else np.zeros(len(BLOCKS))
        ax.bar(x - bw / 2, r_sal, bw, color="#888", edgecolor="black",
               label="vanilla saliency (1 seed)")
        ax.bar(x + bw / 2, r_ig, bw, color="#d62728", edgecolor="black",
               label="integrated grads (5 seeds)")
        ax.set_xticks(x); ax.set_xticklabels(block_names, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{sk}  red", fontsize=10)
        if col == 0: ax.set_ylabel("attribution share", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=7)

    fig.suptitle("Vanilla saliency  vs.  Integrated gradients (5 seeds)\n"
                 "If IG agrees, the saliency story is robust. Where they disagree, IG wins.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rollouts = {s.key: [] for s in SETUPS}
    for setup in SETUPS:
        for seed in SEEDS:
            print(f"[ig] {setup.key} seed={seed}")
            rollouts[setup.key].append(_rollout_ig(setup, seed=seed))

    summary = fig_ig_team_means(rollouts, OUT_DIR / "xai_ig_team_means.png")
    fig_ig_vs_saliency(summary, OUT_DIR / "xai_summary.json",
                      OUT_DIR / "xai_ig_vs_saliency.png")

    out_json = OUT_DIR / "xai_ig_summary.json"
    out_json.write_text(json.dumps({"ig_team_means": summary, "seeds": list(SEEDS),
                                     "ig_steps": IG_STEPS}, indent=2))
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
