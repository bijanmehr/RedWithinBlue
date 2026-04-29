"""Capture per-step red action distributions per arm (I, W, F) for policy comparison.

Reuses blue checkpoint + red seed-0 checkpoints from
`experiments/red-prior-phase1/`. Runs N_PROBE paired eval episodes per arm
(shared env reset keys across arms), records the full softmax-action
distribution at every (episode, step, red_agent), and writes
`experiments/red-prior-phase1/action_probe.npz`.

Output keys:
  arms                    str array, ('I','W','F')
  probs                   float32 [arms, n_eps, T, n_red, n_actions]
                          softmax probabilities the red policy assigned at each step
  acts                    int32   [arms, n_eps, T, n_red]
                          the action actually sampled (for visitation overlap)
  positions               int32   [arms, n_eps, T, n_red, 2]
                          red (row, col) positions BEFORE the step (for state-visitation)
  active_mask             bool    [arms, n_eps, T]
                          step-was-pre-termination flag (cum_done not yet set)
  blue_R, red_R, coverage [arms, n_eps]  per-episode aggregates (sanity)

Run:  PYTHONPATH=src python scripts/red_prior_action_probe.py
Wall: ~30-90 s on M-series (40 eps × 3 arms, jitted scan).
"""
from __future__ import annotations

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
from red_within_blue.training.networks import Actor
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import _connectivity_guardrail
from red_within_blue.types import CELL_WALL

CONFIG_PATH = "experiments/compromise-16x16-5-3b2r-coevo/config.yaml"
BLUE_CKPT = "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz"
PHASE1_DIR = Path("experiments/red-prior-phase1")
ARMS = ("I", "W", "F")
N_PROBE = 40                # paired episodes per arm (shared env reset keys)
SEED_OFFSET = 17            # different from EVAL_SEED_OFFSET to avoid overlap
PROBE_RED_SEED = 0          # use seed-0 red checkpoint per arm


def _build_env(cfg: ExperimentConfig):
    env_cfg = cfg.to_env_config()
    n_red = cfg.env.num_red_agents
    n_blue = cfg.env.num_agents - n_red
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
    return GridCommEnv(env_cfg, reward_fn=reward_fn), n_blue, n_red


def _load_actor_params(ckpt_path: str, actor: Actor, obs_dim: int):
    flat = load_checkpoint(ckpt_path)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {}
    for k, v in flat.items():
        if k not in ref_flat:
            continue
        ref_nd = ref_flat[k].ndim
        stripped[k] = v[0] if v.ndim == ref_nd + 1 else v
    return unflatten_params(stripped, ref)


def make_probe_rollout(env, blue_actor, red_actor, n_blue, n_red,
                       max_steps, num_actions, enforce_conn):
    num_agents = env.config.num_agents

    def rollout(blue_params, red_params, key):
        key, reset_key = jax.random.split(key)
        _o, state = env.reset(reset_key)

        def _scan_body(carry, _):
            state, rng, cum_done = carry
            rng, bk, rk, sk = jax.random.split(rng, 4)
            obs_all = env.obs_array(state)
            blue_obs = obs_all[:n_blue]
            blue_logits = jax.vmap(blue_actor.apply, in_axes=(None, 0))(blue_params, blue_obs)
            blue_acts = jax.vmap(jax.random.categorical)(jax.random.split(bk, n_blue), blue_logits)
            red_obs = obs_all[n_blue:]
            red_logits = jax.vmap(red_actor.apply, in_axes=(None, 0))(red_params, red_obs)
            red_probs = jax.nn.softmax(red_logits, axis=-1)
            red_acts = jax.vmap(jax.random.categorical)(jax.random.split(rk, n_red), red_logits)
            actions = jnp.concatenate([blue_acts, red_acts], axis=0)
            safe_acts = jax.lax.cond(
                jnp.bool_(enforce_conn) & (num_agents >= 2),
                lambda a: _connectivity_guardrail(
                    state.agent_state.positions, state.agent_state.comm_ranges,
                    a, state.global_state.grid.terrain),
                lambda a: a, actions,
            )
            _o, new_state, rewards, done, _i = env.step_array(sk, state, safe_acts)
            active = ~cum_done                          # bool, scalar (per-episode)
            new_cum_done = cum_done | done
            explored = state.global_state.grid.explored
            terrain = state.global_state.grid.terrain
            non_wall = (terrain != CELL_WALL)
            cov = jnp.sum((explored > 0) & non_wall) / jnp.maximum(jnp.sum(non_wall), 1)
            red_pos = state.agent_state.positions[n_blue:]   # pre-step red positions
            return (new_state, rng, new_cum_done), (
                red_probs, red_acts, red_pos, active, rewards, cov,
            )

        _, (probs_seq, acts_seq, pos_seq, active_seq, rew_seq, cov_seq) = jax.lax.scan(
            _scan_body, (state, key, jnp.bool_(False)), jnp.arange(max_steps),
        )
        # Mask post-termination rewards for sane aggregates.
        active_f = active_seq.astype(jnp.float32)[:, None]
        masked_rew = rew_seq * active_f
        blue_total = jnp.sum(masked_rew[:, :n_blue]) / n_blue
        red_total = jnp.sum(masked_rew[:, n_blue])
        final_cov = cov_seq[-1]
        return probs_seq, acts_seq, pos_seq, active_seq, blue_total, red_total, final_cov

    return jax.jit(rollout)


def main() -> None:
    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    env, n_blue, n_red = _build_env(cfg)
    obs_dim = cfg.obs_dim
    max_steps = cfg.env.max_steps
    num_actions = cfg.env.num_actions

    blue_actor = Actor(num_actions=num_actions,
                       hidden_dim=cfg.network.actor_hidden_dim,
                       num_layers=cfg.network.actor_num_layers,
                       activation=cfg.network.activation)
    red_actor = Actor(num_actions=num_actions,
                      hidden_dim=cfg.network.actor_hidden_dim,
                      num_layers=cfg.network.actor_num_layers,
                      activation=cfg.network.activation)
    blue_params = _load_actor_params(BLUE_CKPT, blue_actor, obs_dim)
    print(f"setup: n_blue={n_blue} n_red={n_red} obs_dim={obs_dim} "
          f"max_steps={max_steps} num_actions={num_actions}")

    red_params = {
        a: _load_actor_params(str(PHASE1_DIR / f"red_{a}_seed{PROBE_RED_SEED}.npz"),
                              red_actor, obs_dim)
        for a in ARMS
    }

    rollout = make_probe_rollout(env, blue_actor, red_actor, n_blue, n_red,
                                 max_steps, num_actions, cfg.enforce_connectivity)

    eval_keys = jax.random.split(jax.random.PRNGKey(SEED_OFFSET), N_PROBE)

    probs_arr = np.zeros((len(ARMS), N_PROBE, max_steps, n_red, num_actions),
                         dtype=np.float32)
    acts_arr = np.zeros((len(ARMS), N_PROBE, max_steps, n_red), dtype=np.int32)
    pos_arr = np.zeros((len(ARMS), N_PROBE, max_steps, n_red, 2), dtype=np.int32)
    active_arr = np.zeros((len(ARMS), N_PROBE, max_steps), dtype=bool)
    blueR = np.zeros((len(ARMS), N_PROBE), dtype=np.float32)
    redR = np.zeros((len(ARMS), N_PROBE), dtype=np.float32)
    cov = np.zeros((len(ARMS), N_PROBE), dtype=np.float32)

    t0 = time.time()
    for ai, a in enumerate(ARMS):
        for ei in range(N_PROBE):
            p, ac, ps, act, br, rr, cv = rollout(blue_params, red_params[a], eval_keys[ei])
            probs_arr[ai, ei] = np.asarray(p)
            acts_arr[ai, ei] = np.asarray(ac)
            pos_arr[ai, ei] = np.asarray(ps)
            active_arr[ai, ei] = np.asarray(act)
            blueR[ai, ei] = float(br)
            redR[ai, ei] = float(rr)
            cov[ai, ei] = float(cv)
        print(f"  {a}: blue_R = {blueR[ai].mean():+.3f}  red_R = {redR[ai].mean():+.3f}"
              f"  cov = {cov[ai].mean()*100:.1f}%")
    print(f"  total {time.time() - t0:.1f}s")

    out = PHASE1_DIR / "action_probe.npz"
    np.savez(
        out,
        arms=np.asarray(ARMS),
        probs=probs_arr,
        acts=acts_arr,
        positions=pos_arr,
        active_mask=active_arr,
        blue_R=blueR,
        red_R=redR,
        coverage=cov,
        n_blue=np.int32(n_blue),
        n_red=np.int32(n_red),
        max_steps=np.int32(max_steps),
        num_actions=np.int32(num_actions),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
