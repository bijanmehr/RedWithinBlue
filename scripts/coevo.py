"""Generalized coevolutionary search, parameterized by config + warm-start.

Usage:
  python scripts/coevo.py \
      --config configs/compromise-16x16-3b1r.yaml \
      --warm-blue experiments/survey-local-16-N4/checkpoint.npz \
      --output-dir experiments/compromise-16x16-3b1r-coevo \
      --gens 20 --pop 8 --eps-per-pair 2

Two populations (blue Actors, JointRedActors). Blue optionally warm-started
from a previous checkpoint; red cold-init. Round-robin POP×POP pairings,
EPS_PER_PAIR episodes each, truncation selection with Gaussian mutation.

Saves the top blue + top red as standard checkpoints so experiment_report
can ingest them, plus `coevo_history.npz` with fitness curves.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


class _Tee:
    """Mirror writes to stdout AND a log file so runs leave a record in out_dir."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self._streams:
            st.flush()

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    save_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, JointRedActor
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import collect_episode_multi_scan_joint


def _build_env(cfg: ExperimentConfig) -> Tuple[GridCommEnv, int]:
    env_cfg = cfg.to_env_config()
    n_red = cfg.env.num_red_agents
    reward_fn = make_multi_agent_reward(
        disconnect_penalty=cfg.reward.disconnect_penalty,
        isolation_weight=cfg.reward.isolation_weight,
        cooperative_weight=cfg.reward.cooperative_weight,
        revisit_weight=cfg.reward.revisit_weight,
        terminal_bonus_scale=cfg.reward.terminal_bonus_scale,
        terminal_bonus_divide=cfg.reward.terminal_bonus_divide,
        spread_weight=cfg.reward.spread_weight,
        fog_potential_weight=cfg.reward.fog_potential_weight,
        num_red_agents=n_red,
    )
    return GridCommEnv(env_cfg, reward_fn=reward_fn), n_red


def _load_warm_blue(cfg: ExperimentConfig, blue_actor: Actor, ckpt_path: str):
    flat = load_checkpoint(ckpt_path)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    ref = blue_actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {k: (v[0] if v.ndim == ref_flat[k].ndim + 1 else v) for k, v in flat.items()}
    return unflatten_params(stripped, ref)


def _tree_perturb(params, key, sigma: float):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    noisy = [
        leaf + sigma * jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        for leaf, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, noisy)


def _stack_pop(params_list):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params_list)


def _index_pop(stacked, i: int):
    return jax.tree_util.tree_map(lambda x: x[i], stacked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--warm-blue", default=None,
                    help="Optional blue actor checkpoint to warm-start from.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--eps-per-pair", type=int, default=2)
    ap.add_argument("--mut-sigma", type=float, default=0.05)
    ap.add_argument("--init-perturb-sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_f = open(out_dir / "run.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    env, n_red = _build_env(cfg)
    n_blue = cfg.env.num_agents - n_red
    obs_dim = cfg.obs_dim
    max_steps = cfg.env.max_steps
    enforce_conn = cfg.enforce_connectivity

    blue_actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
        activation=cfg.network.activation,
    )
    red_actor = JointRedActor(
        num_red=max(1, n_red),
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.train.red_hidden_dim,
        num_layers=cfg.train.red_num_layers,
    )

    print(f"Coevo on {cfg.experiment_name}: POP={args.pop} TOPK={args.topk} "
          f"GENS={args.gens} EPS_PER_PAIR={args.eps_per_pair}  "
          f"n_blue={n_blue} n_red={n_red}")

    # --- init populations ---
    rng = jax.random.PRNGKey(args.seed)
    rng, *blue_keys = jax.random.split(rng, args.pop + 1)
    rng, *red_keys = jax.random.split(rng, args.pop + 1)

    if args.warm_blue:
        print(f"Loading warm-start blue from {args.warm_blue}")
        blue_warm = _load_warm_blue(cfg, blue_actor, args.warm_blue)
        blue_pop_list = [_tree_perturb(blue_warm, k, args.init_perturb_sigma) for k in blue_keys]
    else:
        print("Cold-init blue population.")
        blue_pop_list = [
            blue_actor.init(k, jnp.zeros(obs_dim)) for k in blue_keys
        ]

    red_pop_list = [
        red_actor.init(k, jnp.zeros(max(1, n_red) * obs_dim)) for k in red_keys
    ]
    blue_pop = _stack_pop(blue_pop_list)
    red_pop = _stack_pop(red_pop_list)

    def _single_pair_reward(b_params, r_params, key):
        traj = collect_episode_multi_scan_joint(
            env=env,
            blue_actor=blue_actor,
            blue_params=b_params,
            joint_red_actor=red_actor,
            joint_red_params=r_params,
            key=key,
            max_steps=max_steps,
            num_red_agents=max(1, n_red),  # rollout fn requires n_red>=1
            enforce_connectivity=enforce_conn,
        )
        total_per_agent = jnp.sum(traj.rewards, axis=0)
        blue_total = jnp.sum(total_per_agent[:n_blue]) / max(1, n_blue)
        if n_red > 0:
            red_total = jnp.sum(total_per_agent[n_blue:]) / n_red
        else:
            red_total = jnp.float32(0.0)
        return blue_total, red_total

    eval_pair = jax.jit(_single_pair_reward)

    history = {"gen": [], "best_blue": [], "best_red": [], "mean_blue": [], "mean_red": []}

    for gen in range(args.gens):
        gen_t0 = time.time()
        blue_fit_acc = jnp.zeros(args.pop)
        red_fit_acc = jnp.zeros(args.pop)
        for i in range(args.pop):
            for j in range(args.pop):
                bp = _index_pop(blue_pop, i)
                rp = _index_pop(red_pop, j)
                pair_blue = 0.0
                pair_red = 0.0
                for e in range(args.eps_per_pair):
                    rng, k = jax.random.split(rng)
                    b_r, r_r = eval_pair(bp, rp, k)
                    pair_blue += float(b_r) / args.eps_per_pair
                    pair_red += float(r_r) / args.eps_per_pair
                blue_fit_acc = blue_fit_acc.at[i].add(pair_blue)
                red_fit_acc = red_fit_acc.at[j].add(pair_red)

        blue_fit = blue_fit_acc / args.pop
        red_fit = red_fit_acc / args.pop

        blue_order = jnp.argsort(-blue_fit)
        red_order = jnp.argsort(-red_fit)
        blue_elite_idx = blue_order[:args.topk]
        red_elite_idx = red_order[:args.topk]

        new_blue_list = []
        new_red_list = []
        for i in range(args.pop):
            if i < args.topk:
                new_blue_list.append(_index_pop(blue_pop, int(blue_elite_idx[i])))
                new_red_list.append(_index_pop(red_pop, int(red_elite_idx[i])))
            else:
                rng, kb, kr, ks_b, ks_r = jax.random.split(rng, 5)
                parent_b_idx = int(blue_elite_idx[jax.random.randint(ks_b, (), 0, args.topk)])
                parent_r_idx = int(red_elite_idx[jax.random.randint(ks_r, (), 0, args.topk)])
                new_blue_list.append(
                    _tree_perturb(_index_pop(blue_pop, parent_b_idx), kb, args.mut_sigma)
                )
                new_red_list.append(
                    _tree_perturb(_index_pop(red_pop, parent_r_idx), kr, args.mut_sigma)
                )
        blue_pop = _stack_pop(new_blue_list)
        red_pop = _stack_pop(new_red_list)

        elapsed = time.time() - gen_t0
        history["gen"].append(gen)
        history["best_blue"].append(float(blue_fit.max()))
        history["best_red"].append(float(red_fit.max()))
        history["mean_blue"].append(float(blue_fit.mean()))
        history["mean_red"].append(float(red_fit.mean()))
        print(f"gen {gen:02d}  t={elapsed:.1f}s  "
              f"best_blue={float(blue_fit.max()):+.3f} mean_blue={float(blue_fit.mean()):+.3f}  "
              f"best_red={float(red_fit.max()):+.3f} mean_red={float(red_fit.mean()):+.3f}")

    best_blue = _index_pop(blue_pop, 0)
    best_red = _index_pop(red_pop, 0)

    save_checkpoint(flatten_params(best_blue), str(out_dir / "checkpoint.npz"))
    if n_red > 0:
        save_checkpoint(flatten_params(best_red), str(out_dir / "joint_red_checkpoint.npz"))

    np.savez(
        out_dir / "coevo_history.npz",
        gen=np.array(history["gen"]),
        best_blue=np.array(history["best_blue"]),
        best_red=np.array(history["best_red"]),
        mean_blue=np.array(history["mean_blue"]),
        mean_red=np.array(history["mean_red"]),
    )

    import shutil
    shutil.copy(args.config, out_dir / "config.yaml")
    np.savez(
        out_dir / "metrics.npz",
        rewards=np.zeros((1, args.gens)),
        losses=np.zeros((1, args.gens)),
    )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
