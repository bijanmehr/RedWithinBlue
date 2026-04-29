"""Misbehavior-budget sweep  —  blue mission degradation vs (k, ρ).

Terminology (ties to the proposal's stealth budget ε and intervention budget T_int):

    k  = number of compromised agents               (same k as in ΔJ(k), k*(θ))
    ρ  = per-step probability a compromised agent
         executes the TRAINED red policy instead of
         the NOMINAL policy (= blue actor applied to
         that agent's own obs). ρ is exactly the
         expected total-variation distance between
         compromised and nominal per step.
    M := k · ρ  — total per-step "deviation mass"
         the attacker is allowed to inject into the
         joint team policy. In the proposal's
         notation M is the product of compromise
         count × stealth-budget ε × action rate.

Setup: two trained coevo policies (k=1 → `4b1r`, k=2 → `3b2r`) at N = 5 on
16×16, max_steps = 200. For each (k, ρ) condition we run `--n-seeds`
episodes, measure `blue_ever_known` coverage at t = 200, and save per-seed
finals so downstream plotting can recompute means / stds / quantiles without
a re-run.

Usage:
    python scripts/misbehavior_budget_sweep.py             # defaults: 10 seeds, 5 ρ values
    python scripts/misbehavior_budget_sweep.py --n-seeds 20 --rhos 0 0.1 0.3 0.5 0.7 0.9 1.0

Saves:
    experiments/misbehavior-budget/budget_sweep.npz   (k, rho, finals, mean, std, ...)
    experiments/misbehavior-budget/run.log
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

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


def _rollout_blend(
    cfg: ExperimentConfig,
    blue_actor: Actor,
    blue_params,
    red_actor: JointRedActor,
    red_params,
    rho: float,
    seed: int,
    max_steps: int,
    adversary_type: str = "trained_red",
    nominal_mode: str = "raw_obs",
) -> float:
    """Run one episode with blended red; return final blue-team coverage (%).

    adversary_type — picks the action used when the per-step coin lands HEADS
        (executed with prob ρ on each red agent). Choices:
            trained_red       : learned JointRedActor (default; original behavior)
            uniform_random    : iid uniform over actions
            stay              : action 0 = STAY
            nominal_raw       : same as nominal action (ρ becomes a no-op)
            nominal_clamped   : same as nominal action with team_id clamped to 0

    nominal_mode — controls the action used when the coin lands TAILS:
            raw_obs            : current behavior — `blue_actor(red_obs)`,
                                 OOD because team_id=1 was unseen at training.
            clamp_team_id_zero : override the team_id feature to 0.0 before
                                 applying the blue actor — clean nominal.

    The team_id feature is the LAST element of the obs vector (see env.py:377,
    `_build_obs_array` concatenates [...., uid, team]).
    """
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red
    num_actions = cfg.env.num_actions
    team_id_idx = cfg.obs_dim - 1
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
        # Keep the original 5-way split so the default trained_red / raw_obs
        # path is bit-identical to the pre-patch script. New adversary types
        # reuse k_red (unused outside the trained branch).
        key, k_all, k_red, k_coin, k_step = jax.random.split(key, 5)

        obs_stack = jnp.stack([obs_dict[env.agents[i]] for i in range(n_total)])

        if nominal_mode == "clamp_team_id_zero":
            obs_for_nominal = obs_stack.at[n_blue:, team_id_idx].set(0.0)
        else:
            obs_for_nominal = obs_stack
        nom_keys = jax.random.split(k_all, n_total)
        nominal_all = _blue_actions_batched(blue_params, obs_for_nominal, nom_keys)

        if adversary_type == "trained_red":
            red_obs_flat = obs_stack[n_blue:].reshape(-1)
            adv_red = _red_joint(red_params, red_obs_flat, k_red)
        elif adversary_type == "uniform_random":
            adv_red = jax.random.randint(k_red, (n_red,), 0, num_actions)
        elif adversary_type == "stay":
            adv_red = jnp.zeros((n_red,), dtype=jnp.int32)
        elif adversary_type == "nominal_raw":
            nom_raw_keys = jax.random.split(k_red, n_total)
            adv_red = _blue_actions_batched(blue_params, obs_stack, nom_raw_keys)[n_blue:]
        elif adversary_type == "nominal_clamped":
            obs_clamp = obs_stack.at[n_blue:, team_id_idx].set(0.0)
            nom_cl_keys = jax.random.split(k_red, n_total)
            adv_red = _blue_actions_batched(blue_params, obs_clamp, nom_cl_keys)[n_blue:]
        else:
            raise ValueError(f"unknown adversary_type {adversary_type!r}")

        coins = jax.random.bernoulli(k_coin, rho, shape=(n_red,))

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


@dataclass
class Setup:
    k: int
    config: str
    blue_ckpt: str
    red_ckpt: str


SETUPS: List[Setup] = [
    Setup(
        k=1,
        config="configs/compromise-16x16-5-4b1r.yaml",
        blue_ckpt="experiments/compromise-16x16-5-4b1r-coevo/checkpoint.npz",
        red_ckpt="experiments/compromise-16x16-5-4b1r-coevo/joint_red_checkpoint.npz",
    ),
    Setup(
        k=2,
        config="configs/compromise-16x16-5-3b2r.yaml",
        blue_ckpt="experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz",
        red_ckpt="experiments/compromise-16x16-5-3b2r-coevo/joint_red_checkpoint.npz",
    ),
]

DEFAULT_RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--rhos", type=float, nargs="*", default=DEFAULT_RHOS)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--output-dir", default="experiments/misbehavior-budget")
    ap.add_argument(
        "--adversary-type",
        choices=["trained_red", "uniform_random", "stay", "nominal_raw", "nominal_clamped"],
        default="trained_red",
        help="Action policy used when the per-step coin lands heads (prob ρ).",
    )
    ap.add_argument(
        "--nominal-mode",
        choices=["raw_obs", "clamp_team_id_zero"],
        default="raw_obs",
        help="raw_obs = blue actor on red's real obs (OOD because team_id=1 unseen). "
             "clamp_team_id_zero = override team_id to 0.0 first.",
    )
    ap.add_argument(
        "--out-npz",
        default=None,
        help="Override save path. Defaults to <output-dir>/budget_sweep.npz.",
    )
    ap.add_argument(
        "--k-filter",
        type=int,
        default=None,
        help="If set (1 or 2), only run the matching Setup row.",
    )
    ap.add_argument(
        "--blue-ckpt",
        default=None,
        help="Override the per-Setup blue checkpoint path (used in Phase 5).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_npz is not None:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        log_dir = out_npz.parent
        log_name = f"{out_npz.stem}.log"
    else:
        out_npz = out_dir / "budget_sweep.npz"
        log_dir = out_dir
        log_name = "run.log"
    log_f = open(log_dir / log_name, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    setups = SETUPS if args.k_filter is None else [s for s in SETUPS if s.k == args.k_filter]
    if not setups:
        raise SystemExit(f"--k-filter {args.k_filter} matched no Setup rows; have k∈{[s.k for s in SETUPS]}")

    print(f"Misbehavior-budget sweep: k∈{[s.k for s in setups]}  "
          f"ρ∈{args.rhos}  seeds={args.n_seeds}  max_steps={args.max_steps}  "
          f"adversary_type={args.adversary_type}  nominal_mode={args.nominal_mode}")
    if args.blue_ckpt is not None:
        print(f"  (blue checkpoint override: {args.blue_ckpt})")

    n_conds = len(setups) * len(args.rhos)
    ks_arr = np.zeros(n_conds, dtype=np.int32)
    rhos_arr = np.zeros(n_conds, dtype=np.float32)
    finals = np.zeros((n_conds, args.n_seeds), dtype=np.float32)
    means = np.zeros(n_conds, dtype=np.float32)
    stds = np.zeros(n_conds, dtype=np.float32)

    ci = 0
    t0 = time.time()
    for setup in setups:
        print(f"\n=== k={setup.k}: {setup.config} ===")
        cfg = ExperimentConfig.from_yaml(setup.config)
        blue_ckpt_path = args.blue_ckpt if args.blue_ckpt else setup.blue_ckpt
        blue_actor, blue_params = _load_blue(cfg, blue_ckpt_path)
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)

        for rho in args.rhos:
            ks_arr[ci] = setup.k
            rhos_arr[ci] = rho
            t1 = time.time()
            for s in range(args.n_seeds):
                finals[ci, s] = _rollout_blend(
                    cfg, blue_actor, blue_params, red_actor, red_params,
                    rho=float(rho), seed=s, max_steps=args.max_steps,
                    adversary_type=args.adversary_type,
                    nominal_mode=args.nominal_mode,
                )
            means[ci] = finals[ci].mean()
            stds[ci] = finals[ci].std()
            print(
                f"  k={setup.k}  ρ={rho:.2f}  M={setup.k * rho:.2f}  "
                f"cov={means[ci]:.2f} ± {stds[ci]:.2f}%   "
                f"(min={finals[ci].min():.1f}, max={finals[ci].max():.1f})   "
                f"t={time.time() - t1:.1f}s"
            )
            ci += 1

    print(f"\nTotal sweep time {time.time() - t0:.1f}s  ({n_conds * args.n_seeds} episodes)")
    np.savez(
        out_npz,
        k=ks_arr,
        rho=rhos_arr,
        finals=finals,
        mean=means,
        std=stds,
        n_seeds=np.int32(args.n_seeds),
        max_steps=np.int32(args.max_steps),
        adversary_type=np.array(args.adversary_type),
        nominal_mode=np.array(args.nominal_mode),
        blue_ckpt=np.array(args.blue_ckpt or ""),
    )
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
