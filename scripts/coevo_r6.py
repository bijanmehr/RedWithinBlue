"""Coevolutionary search on the rung-6 32x32 6-blue-vs-4-red setup.

Why this exists. The REINFORCE trainer at rung 6 finds an emergent role split
(2 roamers + 1 patrol + 1 anchor) but blue's gradient signal is corrupted by
opponent non-stationarity, so blue caps at ~11% ever-known coverage. This is
a quick experimental check: does a gradient-free coevolutionary search find a
qualitatively different (better) policy than REINFORCE for either team?

Design.
- Two populations of size POP=8: one of per-agent shared blue Actors, one of
  4-headed JointRedActors. Blue pop initialised from rung-5 warm-start
  + small Gaussian perturbation; red pop initialised cold (random).
- Each generation, every blue is paired with every red (round-robin 8x8 = 64
  pairings, EPS_PER_PAIR episodes each). Per-pair fitness for blue is the mean
  blue team reward; for red, the negation. Each individual's fitness is its
  mean across the 8 pairings it participated in.
- Selection: top TOPK=2 by team fitness survive untouched (elitism). The
  remaining POP-TOPK slots are filled with parent_i + Gaussian(0, MUT_SIGMA),
  parents drawn uniformly from the elite set.
- After GENS=20 generations, save the top blue + top red as a checkpoint and
  drop a numpy log of the fitness curves so report.html can ingest them.

Run: ``python scripts/coevo_r6.py``
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, JointRedActor
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    save_checkpoint,
    unflatten_params,
)
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import collect_episode_multi_scan_joint
from red_within_blue.env import GridCommEnv

CONFIG_PATH = "configs/adv-ladder-r6-32x32-6b4r.yaml"
WARM_BLUE_CKPT = "experiments/adv-ladder-r5-32x32-7b3r/checkpoint.npz"
OUTPUT_DIR = Path("experiments/adv-ladder-r6-coevo")

POP = 8
TOPK = 2
GENS = 20
EPS_PER_PAIR = 2
MUT_SIGMA = 0.05
INIT_PERTURB_SIGMA = 0.01
SEED = 0


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


def _load_warm_blue(cfg: ExperimentConfig, blue_actor: Actor):
    # r5 was trained with num_seeds=2, so every leaf has a leading seed axis.
    # Strip seed 0 against the reference shape so we land on single-seed params.
    flat = load_checkpoint(WARM_BLUE_CKPT)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    ref = blue_actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {}
    for k, v in flat.items():
        ref_nd = ref_flat[k].ndim
        stripped[k] = v[0] if v.ndim == ref_nd + 1 else v
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
    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        num_red=n_red,
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.train.red_hidden_dim,
        num_layers=cfg.train.red_num_layers,
    )

    print(f"Coevolution on {cfg.experiment_name}: POP={POP} TOPK={TOPK} "
          f"GENS={GENS} EPS_PER_PAIR={EPS_PER_PAIR}")
    print(f"Loading warm-start blue from {WARM_BLUE_CKPT}")
    blue_warm = _load_warm_blue(cfg, blue_actor)

    # --- init populations ---
    rng = jax.random.PRNGKey(SEED)
    rng, *blue_keys = jax.random.split(rng, POP + 1)
    rng, *red_keys = jax.random.split(rng, POP + 1)

    blue_pop_list = [_tree_perturb(blue_warm, k, INIT_PERTURB_SIGMA) for k in blue_keys]
    red_pop_list = [
        red_actor.init(k, jnp.zeros(n_red * obs_dim)) for k in red_keys
    ]
    blue_pop = _stack_pop(blue_pop_list)
    red_pop = _stack_pop(red_pop_list)

    # --- JIT a single-pair episode rollout returning per-team total rewards ---
    def _single_pair_reward(b_params, r_params, key):
        traj = collect_episode_multi_scan_joint(
            env=env,
            blue_actor=blue_actor,
            blue_params=b_params,
            joint_red_actor=red_actor,
            joint_red_params=r_params,
            key=key,
            max_steps=max_steps,
            num_red_agents=n_red,
            enforce_connectivity=enforce_conn,
        )
        # rewards: [T, N], already masked. Sum across time, then split by team.
        total_per_agent = jnp.sum(traj.rewards, axis=0)         # [N]
        blue_total = jnp.sum(total_per_agent[:n_blue]) / n_blue
        red_total = jnp.sum(total_per_agent[n_blue:]) / n_red
        return blue_total, red_total

    eval_pair = jax.jit(_single_pair_reward)

    history = {"gen": [], "best_blue": [], "best_red": [], "mean_blue": [], "mean_red": []}

    for gen in range(GENS):
        gen_t0 = time.time()
        blue_fit_acc = jnp.zeros(POP)
        red_fit_acc = jnp.zeros(POP)
        # Round-robin pairings
        for i in range(POP):
            for j in range(POP):
                bp = _index_pop(blue_pop, i)
                rp = _index_pop(red_pop, j)
                pair_blue = 0.0
                pair_red = 0.0
                for e in range(EPS_PER_PAIR):
                    rng, k = jax.random.split(rng)
                    b_r, r_r = eval_pair(bp, rp, k)
                    pair_blue += float(b_r) / EPS_PER_PAIR
                    pair_red += float(r_r) / EPS_PER_PAIR
                blue_fit_acc = blue_fit_acc.at[i].add(pair_blue)
                red_fit_acc = red_fit_acc.at[j].add(pair_red)

        blue_fit = blue_fit_acc / POP   # mean across opponents
        red_fit = red_fit_acc / POP

        # selection (top-K by team fitness)
        blue_order = jnp.argsort(-blue_fit)
        red_order = jnp.argsort(-red_fit)
        blue_elite_idx = blue_order[:TOPK]
        red_elite_idx = red_order[:TOPK]

        # build new populations: keep elites verbatim + spawn perturbed children
        new_blue_list = []
        new_red_list = []
        for i in range(POP):
            if i < TOPK:
                new_blue_list.append(_index_pop(blue_pop, int(blue_elite_idx[i])))
                new_red_list.append(_index_pop(red_pop, int(red_elite_idx[i])))
            else:
                rng, kb, kr, ks_b, ks_r = jax.random.split(rng, 5)
                parent_b_idx = int(blue_elite_idx[jax.random.randint(ks_b, (), 0, TOPK)])
                parent_r_idx = int(red_elite_idx[jax.random.randint(ks_r, (), 0, TOPK)])
                new_blue_list.append(
                    _tree_perturb(_index_pop(blue_pop, parent_b_idx), kb, MUT_SIGMA)
                )
                new_red_list.append(
                    _tree_perturb(_index_pop(red_pop, parent_r_idx), kr, MUT_SIGMA)
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

    # --- save best individuals as a checkpoint usable by experiment_report ---
    best_blue = _index_pop(blue_pop, 0)   # elite slot 0
    best_red = _index_pop(red_pop, 0)

    blue_flat = flatten_params(best_blue)
    save_checkpoint(blue_flat, str(OUTPUT_DIR / "checkpoint.npz"))
    red_flat = flatten_params(best_red)
    save_checkpoint(red_flat, str(OUTPUT_DIR / "joint_red_checkpoint.npz"))

    np.savez(
        OUTPUT_DIR / "coevo_history.npz",
        gen=np.array(history["gen"]),
        best_blue=np.array(history["best_blue"]),
        best_red=np.array(history["best_red"]),
        mean_blue=np.array(history["mean_blue"]),
        mean_red=np.array(history["mean_red"]),
    )

    # write a tiny config copy so experiment_report can find the env spec
    import shutil
    shutil.copy(CONFIG_PATH, OUTPUT_DIR / "config.yaml")

    # write a placeholder metrics.npz so experiment_report doesn't crash on it
    np.savez(
        OUTPUT_DIR / "metrics.npz",
        rewards=np.zeros((1, GENS)),
        losses=np.zeros((1, GENS)),
    )

    print(f"\nSaved best blue + best red to {OUTPUT_DIR}")
    print(f"Run: python -m red_within_blue.analysis.experiment_report "
          f"--config {CONFIG_PATH} --experiment-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
