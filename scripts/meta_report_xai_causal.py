"""Causal XAI on 3b2r-coevo: occlusion, identity-swap, and linear probes.

Three analyses, all on the canonical 3b2r-coevo checkpoint:

  Part A — counterfactual occlusion. For every (setup × obs-block × seed),
           roll a full episode with that block zeroed in the policy input
           and measure Δ-coverage from the un-occluded baseline. Bonus: on
           the baseline trajectory itself, compute per-step KL(π_clean ||
           π_occluded) — a per-step sensitivity measure that does not
           require a divergent rollout.

  Part B — identity-swap counterfactual. Swap ``uid`` values across
           same-team agents at every step before the policy sees the obs.
           Causally tests "red is identity-conditional": if so, the swap
           should produce an action-distribution mirror image at the policy
           level. Quantified by KL(π_swap_redR || π_baseline_red(1-R)) vs
           KL(π_swap_redR || π_baseline_redR) — if red R under swap acts
           like baseline (1-R), the cross-comparison should be smaller.

  Part C — linear probes on hidden activations. Capture each layer's hidden
           state on the baseline rollout. Train logistic regression per
           (layer × team × concept), compare to label-shuffled baseline.
           Concepts:
              will_stay_next     — next action == STAY (action 4)
              frontier_in_view   — own 3×3 seen window has any unknown
              blue_in_view       — any blue agent within 3×3 (red only)

5 seeds × 3 setups. ~90 s wall-clock.

Outputs (under experiments/meta-report/):
  xai_occlusion_coverage.png
  xai_occlusion_kl.png
  xai_identity_swap.png
  xai_probes_accuracy.png
  xai_causal_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from meta_report import SETUPS, OUT_DIR, _load_blue, _load_red
from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward,
    normalized_exploration_reward,
)
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
from red_within_blue.visualizer import _merge_team_belief


SEEDS = (0, 1, 2, 3, 4)
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


# ===================================================================
# Generic rollout with obs_transform hook
# ===================================================================
def _rollout(setup, seed: int,
             obs_transform: Optional[Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]] = None,
             capture_logits: bool = False,
             clean_obs_for_logit_compare: bool = False):
    """Run one episode. ``obs_transform(obs_per_agent, team_ids, n_blue, n_red)``
    returns transformed obs (same shape) before each step's policy forward.

    Returns dict with paths, team_ids, coverage_curve, final_coverage,
    all_obs (T+1, N, obs_dim), all_logits (T, N, num_actions) if capture_logits,
    all_clean_logits (T, N, num_actions) if clean_obs_for_logit_compare,
    all_actions (T, N), all_hidden (T, N, hidden_dim) for layer 1 (blue MLP only).

    For the joint red, ``all_logits`` and ``all_hidden`` reflect the policy's
    output for each red agent's own slice; the joint controller's hidden state
    is captured once per step at index n_blue (re-broadcast across reds).
    """
    cfg = ExperimentConfig.from_yaml(setup.config)
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red
    obs_dim = 23  # validated below

    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    if n_red > 0:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

    reward_fn = normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)
    assert env.obs_dim == obs_dim, env.obs_dim

    @jax.jit
    def _blue_logits(bp, obs):
        return blue_actor.apply(bp, obs)

    @jax.jit
    def _red_logits(rp, obs_flat):
        return red_actor.apply(rp, obs_flat)  # [n_red, num_actions]

    # Hidden capture — Flax's capture_intermediates=True returns the output of
    # every submodule call. We pick the activation right before the final Dense.
    @jax.jit
    def _blue_hidden(bp, obs):
        out, state = blue_actor.apply(bp, obs, capture_intermediates=True)
        # The final hidden state is the output of the second-to-last Dense.
        # Submodule names are Dense_0, Dense_1, ..., Dense_{N-1}; the last
        # one is the output head.
        intermediates = state["intermediates"]
        last_hidden_key = f"Dense_{cfg.network.actor_num_layers - 1}"
        return intermediates[last_hidden_key]["__call__"][0]

    @jax.jit
    def _red_hidden(rp, obs_flat):
        out, state = red_actor.apply(rp, obs_flat, capture_intermediates=True)
        intermediates = state["intermediates"]
        last_hidden_key = f"Dense_{cfg.train.red_num_layers - 1}"
        return intermediates[last_hidden_key]["__call__"][0]

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env.reset(key)
    team_ids = np.asarray(state.agent_state.team_ids).copy()

    paths = [np.asarray(state.agent_state.positions).copy()]
    coverage_curve = []
    blue_ever = None

    all_obs = []                            # list of (N, obs_dim) — clean
    all_logits = []                         # list of (N, num_actions) — applied
    all_clean_logits = []                   # list of (N, num_actions) — on clean obs
    all_actions = []                        # list of (N,)
    all_hidden = []                         # list of (N, hidden_dim)
    next_action_buffer = None               # for "will_stay_next" probe target

    max_steps = cfg.env.max_steps
    for step in range(1, max_steps + 1):
        key, *subkeys = jax.random.split(key, n_total + 2)
        step_key = subkeys[-1]

        clean_obs_array = np.stack([np.asarray(obs_dict[env.agents[i]]) for i in range(n_total)], axis=0)
        if obs_transform is not None:
            mod_obs_array = obs_transform(clean_obs_array, team_ids, n_blue, n_red)
        else:
            mod_obs_array = clean_obs_array

        # 1. Compute logits + actions
        actions_np = np.zeros(n_total, dtype=np.int32)
        logits_np = np.zeros((n_total, cfg.env.num_actions), dtype=np.float64)
        clean_logits_np = np.zeros_like(logits_np) if clean_obs_for_logit_compare else None
        hidden_np = None  # set below if we know shape

        for i in range(n_blue):
            l = np.asarray(_blue_logits(blue_params, jnp.asarray(mod_obs_array[i])))
            logits_np[i] = l
            actions_np[i] = int(jax.random.categorical(subkeys[i], jnp.asarray(l)))
            if clean_obs_for_logit_compare:
                clean_logits_np[i] = np.asarray(_blue_logits(blue_params, jnp.asarray(clean_obs_array[i])))

        if n_red > 0:
            mod_red_flat = jnp.concatenate([jnp.asarray(mod_obs_array[n_blue + r]) for r in range(n_red)])
            red_l = np.asarray(_red_logits(red_params, mod_red_flat))      # (n_red, num_actions)
            logits_np[n_blue:n_blue + n_red] = red_l
            red_keys = jax.random.split(subkeys[n_blue], n_red)
            for r in range(n_red):
                actions_np[n_blue + r] = int(jax.random.categorical(red_keys[r], jnp.asarray(red_l[r])))
            if clean_obs_for_logit_compare:
                clean_red_flat = jnp.concatenate([jnp.asarray(clean_obs_array[n_blue + r]) for r in range(n_red)])
                cr = np.asarray(_red_logits(red_params, clean_red_flat))
                clean_logits_np[n_blue:n_blue + n_red] = cr

        # Hidden capture (always on clean obs for probe analysis)
        blue_h = []
        for i in range(n_blue):
            blue_h.append(np.asarray(_blue_hidden(blue_params, jnp.asarray(clean_obs_array[i]))))
        if n_red > 0:
            clean_red_flat = jnp.concatenate([jnp.asarray(clean_obs_array[n_blue + r]) for r in range(n_red)])
            joint_h = np.asarray(_red_hidden(red_params, clean_red_flat))
        else:
            joint_h = None
        # Construct per-agent hidden array — blues each have own; reds share the joint hidden.
        hidden_dim_blue = blue_h[0].shape[0]
        hidden_dim_red = joint_h.shape[0] if joint_h is not None else hidden_dim_blue
        # Pad to common size for storage convenience (truncate to min)
        hidden_common = max(hidden_dim_blue, hidden_dim_red)
        hidden_np = np.zeros((n_total, hidden_common), dtype=np.float32)
        for i in range(n_blue):
            hidden_np[i, :hidden_dim_blue] = blue_h[i]
        for r in range(n_red):
            hidden_np[n_blue + r, :hidden_dim_red] = joint_h

        all_obs.append(clean_obs_array.copy())
        all_logits.append(logits_np)
        if clean_obs_for_logit_compare:
            all_clean_logits.append(clean_logits_np)
        all_actions.append(actions_np)
        all_hidden.append(hidden_np)

        # 2. Step env
        action_dict = {env.agents[i]: int(actions_np[i]) for i in range(n_total)}
        obs_dict, state, _r, dones, _info = env.step_env(step_key, state, action_dict)
        paths.append(np.asarray(state.agent_state.positions).copy())

        # 3. Coverage tracking
        local_maps_np = np.asarray(state.agent_state.local_map)
        team_ids_np = np.asarray(state.agent_state.team_ids)
        blue_belief = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)
        terrain = np.asarray(state.global_state.grid.terrain)
        non_wall = terrain != CELL_WALL
        known_now = (blue_belief != MAP_UNKNOWN) & non_wall
        blue_ever = known_now if blue_ever is None else (blue_ever | known_now)
        coverage_curve.append(100.0 * blue_ever.sum() / max(1, non_wall.sum()))

        if bool(dones["__all__"]):
            break

    return {
        "paths": np.stack(paths, axis=0),
        "team_ids": team_ids,
        "coverage_curve": np.asarray(coverage_curve, dtype=np.float32),
        "final_coverage": float(coverage_curve[-1]) if coverage_curve else 0.0,
        "all_obs": np.stack(all_obs, axis=0),                  # (T, N, obs_dim)
        "all_logits": np.stack(all_logits, axis=0),            # (T, N, A)
        "all_clean_logits": (np.stack(all_clean_logits, axis=0) if all_clean_logits
                             else None),
        "all_actions": np.stack(all_actions, axis=0),          # (T, N)
        "all_hidden": np.stack(all_hidden, axis=0),            # (T, N, H)
        "n_blue": n_blue, "n_red": n_red,
    }


# ===================================================================
# Obs transforms
# ===================================================================
def _make_occlude_block(block_name: str):
    lo = next(b[1] for b in BLOCKS if b[0] == block_name)
    hi = next(b[2] for b in BLOCKS if b[0] == block_name)
    def transform(obs_array, team_ids, n_blue, n_red):
        out = obs_array.copy()
        out[:, lo:hi] = 0.0
        return out
    return transform


def _swap_uids_within_team(obs_array, team_ids, n_blue, n_red):
    """Permute uid values within each team. For n_red=2 this is a swap."""
    out = obs_array.copy()
    uid_idx = 21
    blue_uids = out[:n_blue, uid_idx].copy()
    red_uids = out[n_blue:n_blue + n_red, uid_idx].copy()
    if n_blue >= 2:
        # Reverse permutation — for n_blue=2 it's a swap, for n_blue>2 a cycle
        out[:n_blue, uid_idx] = blue_uids[::-1]
    if n_red >= 2:
        out[n_blue:n_blue + n_red, uid_idx] = red_uids[::-1]
    return out


# ===================================================================
# KL between logit distributions
# ===================================================================
def _kl_per_step(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    """KL(p_a || p_b) per (T, N) row. logits_a/b shape (T, N, A)."""
    log_pa = logits_a - jax.scipy.special.logsumexp(logits_a, axis=-1, keepdims=True)
    log_pb = logits_b - jax.scipy.special.logsumexp(logits_b, axis=-1, keepdims=True)
    pa = np.exp(log_pa)
    return np.sum(pa * (log_pa - log_pb), axis=-1)


# ===================================================================
# Part A — Occlusion
# ===================================================================
def run_occlusion(setups) -> Dict:
    """For each (setup × block × seed), full counterfactual rollout. Captures
    final coverage and per-step KL on the *baseline* trajectory."""
    results = {s.key: {b: {"final_cov": [], "baseline_cov": [], "kl": []} for b in BLOCK_NAMES}
               for s in setups}

    for setup in setups:
        print(f"[occlusion] {setup.key}")
        for seed in SEEDS:
            base = _rollout(setup, seed=seed, obs_transform=None,
                           capture_logits=True, clean_obs_for_logit_compare=False)
            base_cov = base["final_coverage"]

            for block in BLOCK_NAMES:
                # Per-step KL on baseline trajectory: re-evaluate logits on
                # occluded obs at every captured step, compare to clean logits.
                lo = next(b[1] for b in BLOCKS if b[0] == block)
                hi = next(b[2] for b in BLOCKS if b[0] == block)
                clean_obs = base["all_obs"]                # (T, N, obs_dim)
                occluded_obs = clean_obs.copy()
                occluded_obs[:, :, lo:hi] = 0.0

                # Re-run forward pass on occluded obs (no env step) — cheap
                cfg = ExperimentConfig.from_yaml(setup.config)
                blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
                if cfg.env.num_red_agents > 0:
                    red_actor, red_params = _load_red(cfg, setup.red_ckpt)
                T = clean_obs.shape[0]
                n_blue = base["n_blue"]; n_red = base["n_red"]
                occ_logits = np.zeros_like(base["all_logits"])

                @jax.jit
                def _bl(bp, obs): return blue_actor.apply(bp, obs)
                @jax.jit
                def _rl(rp, obs_flat): return red_actor.apply(rp, obs_flat) if n_red > 0 else jnp.zeros((1, cfg.env.num_actions))

                for t in range(T):
                    for i in range(n_blue):
                        occ_logits[t, i] = np.asarray(_bl(blue_params, jnp.asarray(occluded_obs[t, i])))
                    if n_red > 0:
                        rf = jnp.concatenate([jnp.asarray(occluded_obs[t, n_blue + r]) for r in range(n_red)])
                        rl = np.asarray(_rl(red_params, rf))
                        occ_logits[t, n_blue:n_blue + n_red] = rl

                kl = _kl_per_step(base["all_logits"], occ_logits)   # (T, N)
                results[setup.key][block]["kl"].append(kl)

                # Full counterfactual rollout
                tr = _make_occlude_block(block)
                roll = _rollout(setup, seed=seed, obs_transform=tr)
                results[setup.key][block]["final_cov"].append(roll["final_coverage"])
                results[setup.key][block]["baseline_cov"].append(base_cov)

    return results


def fig_occlusion_coverage(results: Dict, out_png: Path) -> Dict:
    setups = list(results.keys())
    fig, axes = plt.subplots(1, len(setups), figsize=(5.0 * len(setups), 4.4),
                            sharey=True)
    if len(setups) == 1: axes = [axes]
    summary = {}
    for ax, sk in zip(axes, setups):
        means = []; stds = []
        baseline_means = []
        for b in BLOCK_NAMES:
            cov = np.asarray(results[sk][b]["final_cov"])
            base = np.asarray(results[sk][b]["baseline_cov"])
            d = base - cov
            means.append(d.mean()); stds.append(d.std() / np.sqrt(len(d)))
            baseline_means.append(base.mean())
        ax.bar(np.arange(len(BLOCK_NAMES)), means, yerr=stds,
               color=[BLOCK_COLOURS[b] for b in BLOCK_NAMES],
               edgecolor="black", linewidth=1.2, capsize=4)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xticks(np.arange(len(BLOCK_NAMES)))
        ax.set_xticklabels(BLOCK_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{sk}  (baseline cov ≈ {np.mean(baseline_means):.1f}%)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Δ-coverage (pp)  =  baseline − occluded", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        summary[sk] = {b: {"mean_dcov": float(m), "se_dcov": float(s)}
                      for b, m, s in zip(BLOCK_NAMES, means, stds)}
    fig.suptitle("Counterfactual occlusion — Δ-coverage when a block is zeroed in the policy input\n"
                 "5 seeds; positive = removing the block hurts coverage",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def fig_occlusion_kl(results: Dict, out_png: Path) -> Dict:
    setups = list(results.keys())
    fig, axes = plt.subplots(1, len(setups), figsize=(5.0 * len(setups), 4.4),
                            sharey=True)
    if len(setups) == 1: axes = [axes]
    summary = {}
    for ax, sk in zip(axes, setups):
        # mean KL over (seeds, T, agents in team)
        blue_means = []; red_means = []
        for b in BLOCK_NAMES:
            kls = results[sk][b]["kl"]                 # list of (T, N) arrays
            # team_ids shape — get from first rollout (all should match)
            blue_kls = []; red_kls = []
            for kl_arr in kls:
                T, N = kl_arr.shape
                # Need to know n_blue/n_red. Stored in base rollout — we don't have it
                # here. Reconstruct from the setup config.
                cfg = ExperimentConfig.from_yaml(next(s for s in SETUPS if s.key == sk).config)
                n_red = cfg.env.num_red_agents
                n_blue = cfg.env.num_agents - n_red
                blue_kls.append(kl_arr[:, :n_blue].mean())
                if n_red > 0:
                    red_kls.append(kl_arr[:, n_blue:].mean())
            blue_means.append(np.mean(blue_kls))
            red_means.append(np.mean(red_kls) if red_kls else 0.0)
        x = np.arange(len(BLOCK_NAMES)); bar_w = 0.38
        ax.bar(x - bar_w / 2, blue_means, bar_w,
               color=[BLOCK_COLOURS[b] for b in BLOCK_NAMES],
               edgecolor="#1f4e8c", linewidth=1.4, label="blue")
        ax.bar(x + bar_w / 2, red_means, bar_w,
               color=[BLOCK_COLOURS[b] for b in BLOCK_NAMES],
               edgecolor="#a6231f", linewidth=1.4, hatch="//", label="red")
        ax.set_xticks(x); ax.set_xticklabels(BLOCK_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{sk}", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("mean KL(π_clean || π_occluded)\nper (step, agent)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        summary[sk] = {"blue_mean_kl": dict(zip(BLOCK_NAMES, [float(x) for x in blue_means])),
                      "red_mean_kl": dict(zip(BLOCK_NAMES, [float(x) for x in red_means]))}
    fig.suptitle("Counterfactual KL on baseline trajectory — fine-grained sensitivity per block\n"
                 "Same trajectory, just re-evaluate the policy with one block zeroed.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


# ===================================================================
# Part B — Identity-swap
# ===================================================================
def run_identity_swap(setups) -> Dict:
    """For each setup, baseline + uid-swapped rollout × 5 seeds.

    Quantify: at every step on the swapped trajectory, ask whether red R
    under swap acts more like baseline red R (no swap) or baseline red (1-R).
    Done by KL on action distributions on *matched positions*.

    Simpler proxy used here: compare the action distribution of red R under
    swap with baseline red R, marginalised over time. If identity-conditional,
    swap_red0_actions ≈ baseline_red1_actions.
    """
    results = {}
    for setup in setups:
        if setup.key == "B":
            continue  # No reds to swap among
        print(f"[id-swap] {setup.key}")
        seed_rows = []
        for seed in SEEDS:
            base = _rollout(setup, seed=seed, obs_transform=None)
            swap = _rollout(setup, seed=seed, obs_transform=_swap_uids_within_team)
            cfg = ExperimentConfig.from_yaml(setup.config)
            n_red = cfg.env.num_red_agents; n_blue = cfg.env.num_agents - n_red
            row = {
                "seed": seed,
                "base_cov": base["final_coverage"],
                "swap_cov": swap["final_coverage"],
                "n_red": n_red,
            }
            # Action histograms per red, baseline vs swap
            for r in range(n_red):
                base_acts = base["all_actions"][:, n_blue + r]
                swap_acts = swap["all_actions"][:, n_blue + r]
                hb = np.bincount(base_acts, minlength=cfg.env.num_actions) / len(base_acts)
                hs = np.bincount(swap_acts, minlength=cfg.env.num_actions) / len(swap_acts)
                row[f"red{r}_base_dist"] = hb.tolist()
                row[f"red{r}_swap_dist"] = hs.tolist()
            # If n_red == 2, compute "did behaviours swap?"
            if n_red == 2:
                hb0 = np.asarray(row["red0_base_dist"])
                hb1 = np.asarray(row["red1_base_dist"])
                hs0 = np.asarray(row["red0_swap_dist"])
                hs1 = np.asarray(row["red1_swap_dist"])
                eps = 1e-9
                kl_swap_vs_baseSelf = (
                    np.sum(hs0 * np.log((hs0 + eps) / (hb0 + eps))) +
                    np.sum(hs1 * np.log((hs1 + eps) / (hb1 + eps)))
                ) / 2
                kl_swap_vs_baseOther = (
                    np.sum(hs0 * np.log((hs0 + eps) / (hb1 + eps))) +
                    np.sum(hs1 * np.log((hs1 + eps) / (hb0 + eps)))
                ) / 2
                row["kl_self"] = float(kl_swap_vs_baseSelf)
                row["kl_other"] = float(kl_swap_vs_baseOther)
                # Identity-swap score: log-ratio. If <0, swap behaves more like
                # baseline (1-R) than baseline (R) — a behaviour-swap.
                row["swap_score"] = float(kl_swap_vs_baseOther - kl_swap_vs_baseSelf)
            seed_rows.append(row)
        results[setup.key] = seed_rows
    return results


def fig_identity_swap(results: Dict, out_png: Path) -> Dict:
    """Two-panel: (left) coverage clean vs swap per setup,
    (right) C2-only swap-score and per-red action distributions."""
    fig = plt.figure(figsize=(11.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.6])
    ax_cov = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])

    setups = list(results.keys())
    summary = {}

    # Coverage panel
    bw = 0.38
    x = np.arange(len(setups))
    base_covs = [np.mean([r["base_cov"] for r in results[s]]) for s in setups]
    swap_covs = [np.mean([r["swap_cov"] for r in results[s]]) for s in setups]
    base_se = [np.std([r["base_cov"] for r in results[s]]) / np.sqrt(len(results[s])) for s in setups]
    swap_se = [np.std([r["swap_cov"] for r in results[s]]) / np.sqrt(len(results[s])) for s in setups]
    ax_cov.bar(x - bw / 2, base_covs, bw, yerr=base_se, color="#888",
               edgecolor="black", capsize=3, label="baseline")
    ax_cov.bar(x + bw / 2, swap_covs, bw, yerr=swap_se, color="#d62728",
               edgecolor="black", capsize=3, label="uid-swapped")
    ax_cov.set_xticks(x); ax_cov.set_xticklabels(setups)
    ax_cov.set_ylabel("final coverage (%)", fontsize=9)
    ax_cov.set_title("Coverage under uid-swap\n(swap within each team)", fontsize=10)
    ax_cov.legend(fontsize=8); ax_cov.grid(True, alpha=0.3, axis="y")

    # Swap-score panel (C2 only)
    if "C2" in results:
        scores = [r["swap_score"] for r in results["C2"]]
        ax_score.bar(np.arange(len(scores)), scores, color="#d62728",
                     edgecolor="black", linewidth=0.6)
        ax_score.axhline(0, color="k", linewidth=0.7)
        ax_score.set_xlabel("seed", fontsize=9)
        ax_score.set_ylabel("KL(swap || base-other) − KL(swap || base-self)", fontsize=8)
        ax_score.set_title("C2 identity-swap score per seed\n(positive = identity holds; negative = behaviour swapped)",
                          fontsize=9)
        ax_score.grid(True, alpha=0.3, axis="y")
        ax_score.set_xticks(np.arange(len(scores)))
        ax_score.set_xticklabels([f"s{r['seed']}" for r in results["C2"]], fontsize=8)
        summary["C2_mean_swap_score"] = float(np.mean(scores))
        summary["C2_seeds_with_negative_score"] = int(np.sum(np.asarray(scores) < 0))

    # Action-distribution panel (C2 seed=0)
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    if "C2" in results:
        row = next(r for r in results["C2"] if r["seed"] == 0)
        bw2 = 0.18
        xa = np.arange(5)
        ax_dist.bar(xa - 1.5 * bw2, row["red0_base_dist"], bw2,
                    color="#a6231f", label="baseline red 0")
        ax_dist.bar(xa - 0.5 * bw2, row["red1_base_dist"], bw2,
                    color="#dc8a86", label="baseline red 1")
        ax_dist.bar(xa + 0.5 * bw2, row["red0_swap_dist"], bw2,
                    color="#a6231f", hatch="//", edgecolor="black", linewidth=0.4,
                    label="swap red 0")
        ax_dist.bar(xa + 1.5 * bw2, row["red1_swap_dist"], bw2,
                    color="#dc8a86", hatch="//", edgecolor="black", linewidth=0.4,
                    label="swap red 1")
        ax_dist.set_xticks(xa); ax_dist.set_xticklabels(action_names, fontsize=8)
        ax_dist.set_ylabel("share of timesteps", fontsize=9)
        ax_dist.set_title("C2 seed=0 — red action distributions, base vs swap\n"
                         "If identity-conditional, swap-red-0 should resemble base-red-1",
                         fontsize=9)
        ax_dist.legend(fontsize=7, loc="upper left")
        ax_dist.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Identity-swap counterfactual — does the policy actually depend on uid?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


# ===================================================================
# Part C — Linear probes on hidden activations
# ===================================================================
def _build_concept_targets(rollouts):
    """For each (setup, agent, t), build ground-truth concept labels.

    Concepts (per agent):
      will_stay_next   — action_t == STAY (action 4). Defined for t < T-1.
      frontier_in_view — own seen-field has any 0 (unknown). Always defined.
      blue_in_view     — any blue agent within the agent's 3×3 window. Defined
                         only for red agents.
    """
    out = {}
    for setup_key, by_seed in rollouts.items():
        per_setup = {"will_stay_next": [], "frontier_in_view": [], "blue_in_view": [],
                    "team_ids": [], "hidden": []}
        for seed_roll in by_seed:
            obs = seed_roll["all_obs"]                     # (T, N, 23)
            actions = seed_roll["all_actions"]             # (T, N)
            paths = seed_roll["paths"]                     # (T+1, N, 2)
            team_ids = seed_roll["team_ids"]
            hidden = seed_roll["all_hidden"]               # (T, N, H)

            T, N, _ = obs.shape
            # will_stay_next: action_t == STAY. Pair (hidden_t, target_t).
            stay_target = (actions == 4).astype(np.int32)  # (T, N)
            # frontier_in_view: any of the 9 seen-window dims == 0
            seen_window = obs[:, :, 9:18]                  # (T, N, 9)
            frontier_target = (seen_window == 0).any(axis=-1).astype(np.int32)
            # blue_in_view: any blue agent within Chebyshev distance 1 of the agent
            blue_target = np.zeros((T, N), dtype=np.int32)
            for t in range(T):
                pos_t = paths[t]
                for i in range(N):
                    if team_ids[i] != 1:
                        continue
                    near = False
                    for j in range(N):
                        if team_ids[j] == 0:
                            d = np.max(np.abs(pos_t[i] - pos_t[j]))
                            if d <= 1:
                                near = True; break
                    blue_target[t, i] = int(near)
            per_setup["will_stay_next"].append(stay_target)
            per_setup["frontier_in_view"].append(frontier_target)
            per_setup["blue_in_view"].append(blue_target)
            per_setup["team_ids"].append(team_ids)
            per_setup["hidden"].append(hidden)
        out[setup_key] = per_setup
    return out


def _probe_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """5-fold stratified CV mean accuracy of logistic regression."""
    if len(np.unique(y)) < 2:
        return float("nan")
    if X.shape[0] < 50:
        return float("nan")
    # Subsample if huge
    if X.shape[0] > 5000:
        idx = np.random.RandomState(0).choice(X.shape[0], 5000, replace=False)
        X, y = X[idx], y[idx]
    accs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=0).split(X, y):
        clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def run_probes(rollouts) -> Dict:
    targets = _build_concept_targets(rollouts)
    summary = {}
    for setup_key, packed in targets.items():
        # Stack across seeds; keep team-id index
        H = np.concatenate(packed["hidden"], axis=0)        # (sumT, N, H)
        sumT, N, hdim = H.shape
        team_ids = packed["team_ids"][0]                    # all seeds share
        stay = np.concatenate(packed["will_stay_next"], axis=0)  # (sumT, N)
        front = np.concatenate(packed["frontier_in_view"], axis=0)
        blue_iv = np.concatenate(packed["blue_in_view"], axis=0)
        out_setup = {}
        for team_label, team_val in [("blue", 0), ("red", 1)]:
            mask_agents = team_ids == team_val
            if not mask_agents.any():
                continue
            X = H[:, mask_agents, :].reshape(-1, hdim)       # (sumT*nteam, hdim)
            y_stay = stay[:, mask_agents].reshape(-1)
            y_front = front[:, mask_agents].reshape(-1)
            y_blue_iv = blue_iv[:, mask_agents].reshape(-1)

            # Real
            acc_stay = _probe_accuracy(X, y_stay)
            acc_front = _probe_accuracy(X, y_front)
            # Random-label baseline (negative control)
            rng = np.random.RandomState(7)
            y_rand = rng.permutation(y_stay)
            acc_rand = _probe_accuracy(X, y_rand)
            entry = {
                "will_stay_next": acc_stay,
                "frontier_in_view": acc_front,
                "shuffled_stay": acc_rand,
                "n_examples": int(X.shape[0]),
            }
            if team_label == "red":
                entry["blue_in_view"] = _probe_accuracy(X, y_blue_iv)
            out_setup[team_label] = entry
        summary[setup_key] = out_setup
    return summary


def fig_probes(probe_summary: Dict, out_png: Path) -> None:
    setups = list(probe_summary.keys())
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.6))
    rows = []
    labels = []
    for sk in setups:
        for team in ["blue", "red"]:
            if team not in probe_summary[sk]:
                continue
            entry = probe_summary[sk][team]
            row = [
                entry.get("will_stay_next", np.nan),
                entry.get("frontier_in_view", np.nan),
                entry.get("blue_in_view", np.nan),
                entry.get("shuffled_stay", np.nan),
            ]
            rows.append(row)
            labels.append(f"{sk}/{team}")
    arr = np.asarray(rows)
    cols = ["will_stay_next", "frontier_in_view", "blue_in_view", "shuffled_stay\n(neg control)"]
    im = ax.imshow(arr, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            colour = "white" if (not np.isnan(v) and v < 0.7) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=colour)
    fig.colorbar(im, ax=ax, label="5-fold CV accuracy")
    ax.set_title("Linear probes on the last hidden layer\n"
                 "Higher than the shuffled-label control = the network encodes that concept",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Driver
# ===================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-roll baselines once for probe analysis (5 seeds × 3 setups)
    print("[baseline] capturing 5-seed rollouts for probes...")
    rollouts = {s.key: [] for s in SETUPS}
    for setup in SETUPS:
        for seed in SEEDS:
            print(f"  [baseline] {setup.key} seed={seed}")
            rollouts[setup.key].append(_rollout(setup, seed=seed))

    # --- Part A
    print("[A] running occlusion sweep...")
    occlusion = run_occlusion(SETUPS)
    cov_summary = fig_occlusion_coverage(occlusion, OUT_DIR / "xai_occlusion_coverage.png")
    kl_summary = fig_occlusion_kl(occlusion, OUT_DIR / "xai_occlusion_kl.png")

    # --- Part B
    print("[B] running identity swap...")
    id_swap = run_identity_swap(SETUPS)
    swap_summary = fig_identity_swap(id_swap, OUT_DIR / "xai_identity_swap.png")

    # --- Part C
    print("[C] running linear probes...")
    probe_summary = run_probes(rollouts)
    fig_probes(probe_summary, OUT_DIR / "xai_probes_accuracy.png")

    summary = {
        "occlusion_coverage": cov_summary,
        "occlusion_kl": kl_summary,
        "identity_swap": swap_summary,
        "probes": probe_summary,
        "seeds": list(SEEDS),
    }
    out_json = OUT_DIR / "xai_causal_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
