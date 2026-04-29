"""Heterogeneous-ρ sweep at fixed Σρ.

Addresses v2 §5.2 / §8.1 open question: does Σρ alone drive ΔJ, or does the
SHAPE of the per-agent ρ vector matter? If ΔJ is flat across a line of
constant Σρ, the scalar model holds and the sum is sufficient. If ΔJ bends,
asymmetry matters and Σρ is under-specified.

Scope: k = 2 (use the 3b2r coevo checkpoint). Hold Σρ at two levels {0.5, 1.0}.
At each level sweep the asymmetry ratio, i.e. (ρ_A, ρ_B) along the constraint
line ρ_A + ρ_B = Σρ with ρ_A ≤ ρ_B:

  Σρ = 0.5:  (0.25, 0.25) (0.20, 0.30) (0.15, 0.35) (0.10, 0.40) (0.05, 0.45) (0.00, 0.50)
  Σρ = 1.0:  (0.50, 0.50) (0.40, 0.60) (0.30, 0.70) (0.20, 0.80) (0.10, 0.90) (0.00, 1.00)

N seeds per condition. Save finals to hetero_sweep.npz.

Usage:
    python scripts/misbehavior_hetero_sweep.py                   # defaults: 15 seeds
    python scripts/misbehavior_hetero_sweep.py --n-seeds 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, JointRedActor
from red_within_blue.training.rewards_training import normalized_competitive_reward
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
from red_within_blue.visualizer import _merge_team_belief


CONFIG = "configs/compromise-16x16-5-3b2r.yaml"
BLUE_CKPT = "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz"
RED_CKPT = "experiments/compromise-16x16-5-3b2r-coevo/joint_red_checkpoint.npz"

SIGMA_LEVELS = [0.5, 1.0]
ASYMMETRY_STEPS = 6  # how many (ρ_A, ρ_B) pairs per Σρ


class _Tee:
    def __init__(self, *streams):
        self._s = streams

    def write(self, x):
        for st in self._s:
            st.write(x); st.flush()

    def flush(self):
        for st in self._s:
            st.flush()


def _strip_seed_dim(flat, ref_flat):
    return {k: (v[0] if v.ndim == ref_flat[k].ndim + 1 else v) for k, v in flat.items()}


def _load_blue(cfg: ExperimentConfig, ckpt_path: str):
    flat = load_checkpoint(ckpt_path)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
    )
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    return actor, unflatten_params(_strip_seed_dim(flat, flatten_params(ref)), ref)


def _load_red(cfg: ExperimentConfig, ckpt_path: str):
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


def _rollout_hetero(
    cfg, blue_actor, blue_params, red_actor, red_params,
    rho_vec: np.ndarray, seed: int, max_steps: int,
) -> float:
    """One episode where compromised agent r flips to red with per-step prob
    rho_vec[r]. Returns final blue-team coverage (%)."""
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red
    assert rho_vec.shape == (n_red,), f"rho_vec must be length {n_red}"
    rho_jax = jnp.asarray(rho_vec, dtype=jnp.float32)
    env = GridCommEnv(cfg.to_env_config(), reward_fn=normalized_competitive_reward)

    @jax.jit
    def _blue_actions_batched(bp, obs_batch, keys):
        logits = jax.vmap(lambda o: blue_actor.apply(bp, o))(obs_batch)
        return jax.vmap(jax.random.categorical)(keys, logits)

    @jax.jit
    def _red_joint(rp, obs_flat, key):
        logits = red_actor.apply(rp, obs_flat)
        keys = jax.random.split(key, n_red)
        return jax.vmap(jax.random.categorical)(keys, logits)

    key = jax.random.PRNGKey(seed)
    obs_dict, state = env.reset(key)
    blue_ever = None
    cov = 0.0

    for step in range(max_steps):
        key, k_all, k_red, k_coin, k_step = jax.random.split(key, 5)

        obs_stack = jnp.stack([obs_dict[env.agents[i]] for i in range(n_total)])
        nom_keys = jax.random.split(k_all, n_total)
        nominal_all = _blue_actions_batched(blue_params, obs_stack, nom_keys)

        red_obs_flat = obs_stack[n_blue:].reshape(-1)
        adv_red = _red_joint(red_params, red_obs_flat, k_red)

        # per-agent Bernoulli with rho_vec
        coins = jax.random.bernoulli(k_coin, rho_jax, shape=(n_red,))

        action_dict = {}
        for i in range(n_blue):
            action_dict[env.agents[i]] = int(nominal_all[i])
        for r in range(n_red):
            if bool(coins[r]):
                action_dict[env.agents[n_blue + r]] = int(adv_red[r])
            else:
                action_dict[env.agents[n_blue + r]] = int(nominal_all[n_blue + r])

        obs_dict, state, _rew, dones, _info = env.step_env(k_step, state, action_dict)

        local_maps_np = np.asarray(state.agent_state.local_map)
        team_ids_np = np.asarray(state.agent_state.team_ids)
        blue_belief = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)
        terrain = np.asarray(state.global_state.grid.terrain)
        non_wall = terrain != CELL_WALL
        known_now = (blue_belief != MAP_UNKNOWN) & non_wall
        blue_ever = known_now if blue_ever is None else (blue_ever | known_now)
        cov = 100.0 * blue_ever.sum() / max(1, non_wall.sum())

        if bool(dones["__all__"]):
            break

    return float(cov)


def _rho_pairs_for_sigma(sigma: float, n_steps: int) -> list:
    """Return n_steps (ρ_A, ρ_B) pairs with ρ_A + ρ_B = σ, ρ_A ≤ ρ_B, spread
    evenly from the balanced point (σ/2, σ/2) to the maximally asymmetric
    (0, σ). n_steps includes both endpoints."""
    # t = 0 is balanced; t = 1 is all-on-one-agent.
    ts = np.linspace(0.0, 1.0, n_steps)
    pairs = []
    for t in ts:
        rho_a = (sigma / 2) * (1.0 - t)
        rho_b = sigma - rho_a
        pairs.append((round(float(rho_a), 4), round(float(rho_b), 4)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--output-dir", default="experiments/misbehavior-budget")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "hetero_run.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    cfg = ExperimentConfig.from_yaml(CONFIG)
    assert cfg.env.num_red_agents == 2, "hetero sweep expects k=2 (3b2r)"

    print(f"Hetero-ρ sweep:  Σρ ∈ {SIGMA_LEVELS}  steps={ASYMMETRY_STEPS}  "
          f"seeds={args.n_seeds}  max_steps={args.max_steps}")
    print(f"  config={CONFIG}  blue={BLUE_CKPT}  red={RED_CKPT}")

    blue_actor, blue_params = _load_blue(cfg, BLUE_CKPT)
    red_actor, red_params = _load_red(cfg, RED_CKPT)

    conds = []
    for sigma in SIGMA_LEVELS:
        for pair in _rho_pairs_for_sigma(sigma, ASYMMETRY_STEPS):
            conds.append((sigma, pair[0], pair[1]))

    n_conds = len(conds)
    sigmas_arr = np.zeros(n_conds, dtype=np.float32)
    rhoA_arr = np.zeros(n_conds, dtype=np.float32)
    rhoB_arr = np.zeros(n_conds, dtype=np.float32)
    finals = np.zeros((n_conds, args.n_seeds), dtype=np.float32)

    t0 = time.time()
    for ci, (sigma, rA, rB) in enumerate(conds):
        sigmas_arr[ci] = sigma
        rhoA_arr[ci] = rA
        rhoB_arr[ci] = rB
        t1 = time.time()
        rho_vec = np.asarray([rA, rB], dtype=np.float32)
        for s in range(args.n_seeds):
            finals[ci, s] = _rollout_hetero(
                cfg, blue_actor, blue_params, red_actor, red_params,
                rho_vec=rho_vec, seed=s, max_steps=args.max_steps,
            )
        m = float(finals[ci].mean())
        sd = float(finals[ci].std())
        print(f"  Σρ={sigma:.2f}  (ρA,ρB)=({rA:.3f},{rB:.3f})  "
              f"|asym|={abs(rB - rA):.3f}  cov={m:.2f} ± {sd:.2f}%   "
              f"t={time.time() - t1:.1f}s")

    # also produce the Σρ=0 baseline (clean) using one condition
    clean_finals = np.zeros(args.n_seeds, dtype=np.float32)
    for s in range(args.n_seeds):
        clean_finals[s] = _rollout_hetero(
            cfg, blue_actor, blue_params, red_actor, red_params,
            rho_vec=np.asarray([0.0, 0.0], dtype=np.float32),
            seed=s, max_steps=args.max_steps,
        )
    print(f"  [clean]  Σρ=0  cov={clean_finals.mean():.2f} ± {clean_finals.std():.2f}%")

    print(f"\nTotal sweep time {time.time() - t0:.1f}s  "
          f"({n_conds * args.n_seeds + args.n_seeds} episodes)")

    out_npz = out_dir / "hetero_sweep.npz"
    np.savez(
        out_npz,
        sigma=sigmas_arr,
        rho_a=rhoA_arr,
        rho_b=rhoB_arr,
        finals=finals,
        clean_finals=clean_finals,
        n_seeds=np.int32(args.n_seeds),
        max_steps=np.int32(args.max_steps),
    )
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
