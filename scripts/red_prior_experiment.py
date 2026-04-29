"""Same-class per-agent red, three prior arms (I/W/F) — Phase 1 of wild-ideas §3.

Why this exists. `experiments/meta-report/red_prior_report.md` shows every red
checkpoint we have is the F-prior × γ-alien (concat-MLP) × β-alien (concat obs)
corner of the design matrix from `docs/10-wild-ideas.md` §3. This script fills
the *same-class* column (architecturally identical to blue, decentralised
per-agent) for all three prior arms in one run:

  I (Insider)    — red params := byte-identical copy of blue C2 checkpoint
                   (all 3 actor layers transferred)
  W (Warm-start) — red params := blue's Dense_0 (input MLP layer) + random
                   Dense_1 & Dense_2 (only the bottom layer is transferred)
  F (Fresh)      — red params := random init for all 3 layers

Setup. C2 (compromise-16x16-5-3b2r), n_blue=3, n_red=2, blue frozen at the
coevo checkpoint, red trained REINFORCE per-agent with sign-flipped reward at
``rewards_training.py:264-267`` (env-level, no script change needed).

Output. ``experiments/red-prior-phase1/red_prior_phase1.npz`` with arrays
keyed by arm × seed × episode: red entropy, blue total reward, red total
reward, prior-posterior KL trajectory.

Run: ``python scripts/red_prior_experiment.py``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    save_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.losses import compute_discounted_returns
from red_within_blue.training.networks import Actor
from red_within_blue.training.rewards_training import make_multi_agent_reward

CONFIG_PATH = "experiments/compromise-16x16-5-3b2r-coevo/config.yaml"
BLUE_CKPT = "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz"
OUTPUT_DIR = Path("experiments/red-prior-phase1")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=CONFIG_PATH,
                   help="ExperimentConfig YAML for the (frozen) blue setup")
    p.add_argument("--blue-ckpt", default=BLUE_CKPT,
                   help="path to the trained blue actor checkpoint")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="where to write red_prior_phase1.npz + per-arm checkpoints")
    return p.parse_args()

NUM_SEEDS = 3
NUM_EPISODES = 3000
LR = 1e-4
GAMMA = 0.99
ENT_COEF = 0.05
GRAD_CLIP = 0.5
ARMS = ("I", "W", "F")


# --------------------------------------------------------------------------- #
# Env + actor construction
# --------------------------------------------------------------------------- #


def _build_env(cfg: ExperimentConfig) -> Tuple[GridCommEnv, int, int]:
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


def _load_blue(cfg: ExperimentConfig, blue_actor: Actor):
    flat = load_checkpoint(BLUE_CKPT)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    ref = blue_actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {}
    for k, v in flat.items():
        ref_nd = ref_flat[k].ndim
        stripped[k] = v[0] if v.ndim == ref_nd + 1 else v
    return unflatten_params(stripped, ref)


def _make_red_init(arm: str, blue_params, red_actor: Actor, obs_dim: int):
    """Construct a function ``key -> red_params`` for the given prior arm."""
    if arm == "I":
        # Insider: byte-identical copy of blue (drops the random key).
        return lambda _key: jax.tree_util.tree_map(jnp.array, blue_params)

    if arm == "W":
        # Warm-start: blue's Dense_0 (input MLP layer) tiled in, Dense_1/2 random.
        def init_w(key):
            fresh = red_actor.init(key, jnp.zeros(obs_dim))
            new = jax.tree_util.tree_map(jnp.array, fresh)
            new["params"]["Dense_0"] = jax.tree_util.tree_map(
                jnp.array, blue_params["params"]["Dense_0"],
            )
            return new
        return init_w

    # Fresh: random init.
    return lambda key: red_actor.init(key, jnp.zeros(obs_dim))


# --------------------------------------------------------------------------- #
# Per-step rollout (blue + per-agent red, both same-class Actor)
# --------------------------------------------------------------------------- #


def _make_rollout(env, blue_actor, red_actor, n_blue, n_red, max_steps, enforce_conn):
    from red_within_blue.training.rollout import _connectivity_guardrail
    num_agents = env.config.num_agents

    def rollout(blue_params, red_params, key):
        key, reset_key = jax.random.split(key)
        _obs_dict, state = env.reset(reset_key)

        def _scan_body(carry, _):
            state, rng, cum_done = carry
            rng, bk, rk, sk = jax.random.split(rng, 4)
            obs_all = env.obs_array(state)                                  # [N, D]

            # Blue: per-agent shared Actor.
            blue_obs = obs_all[:n_blue]
            blue_logits = jax.vmap(blue_actor.apply, in_axes=(None, 0))(
                blue_params, blue_obs,
            )
            bks = jax.random.split(bk, n_blue)
            blue_acts = jax.vmap(jax.random.categorical)(bks, blue_logits)
            blue_lp_full = jax.nn.log_softmax(blue_logits)
            blue_lp = jax.vmap(lambda lp, a: lp[a])(blue_lp_full, blue_acts)

            # Red: per-agent shared Actor (same class as blue).
            red_obs = obs_all[n_blue:]                                      # [n_red, D]
            red_logits = jax.vmap(red_actor.apply, in_axes=(None, 0))(
                red_params, red_obs,
            )                                                               # [n_red, A]
            rks = jax.random.split(rk, n_red)
            red_acts = jax.vmap(jax.random.categorical)(rks, red_logits)
            red_lp_full = jax.nn.log_softmax(red_logits)
            red_lp = jax.vmap(lambda lp, a: lp[a])(red_lp_full, red_acts)

            actions = jnp.concatenate([blue_acts, red_acts], axis=0)
            lp_full = jnp.concatenate([blue_lp_full, red_lp_full], axis=0)

            safe_acts = jax.lax.cond(
                jnp.bool_(enforce_conn) & (num_agents >= 2),
                lambda a: _connectivity_guardrail(
                    state.agent_state.positions,
                    state.agent_state.comm_ranges,
                    a,
                    state.global_state.grid.terrain,
                ),
                lambda a: a,
                actions,
            )
            safe_lp = jax.vmap(lambda lp, a: lp[a])(lp_full, safe_acts)

            _o, new_state, rewards, done, _info = env.step_array(sk, state, safe_acts)
            mask = 1.0 - cum_done.astype(jnp.float32)
            masked_rewards = rewards * mask
            new_cum_done = cum_done | done

            # Per-agent entropy for the red slice.
            red_probs = jax.nn.softmax(red_logits, axis=-1)
            red_entropy = -jnp.sum(red_probs * red_lp_full, axis=-1)        # [n_red]

            step = (safe_acts, masked_rewards, done, safe_lp, mask,
                    red_logits, red_entropy)
            return (new_state, rng, new_cum_done), step

        _, (acts, rews, dones, lps, masks, red_logits_seq, red_ent_seq) = jax.lax.scan(
            _scan_body, (state, key, jnp.bool_(False)), jnp.arange(max_steps),
        )
        return acts, rews, dones, lps, masks, red_logits_seq, red_ent_seq

    return rollout


# --------------------------------------------------------------------------- #
# REINFORCE step on red only (blue frozen)
# --------------------------------------------------------------------------- #


def _make_train_step(env, blue_actor, red_actor, blue_params, n_blue, n_red,
                     max_steps, enforce_conn):
    rollout = _make_rollout(env, blue_actor, red_actor, n_blue, n_red,
                            max_steps, enforce_conn)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adam(LR),
    )

    def red_loss(red_params, key):
        acts, rews, _dones, _lps, masks, red_logits_seq, _ent = rollout(
            blue_params, red_params, key,
        )
        # rewards layout: [T, N]. Red reward is the sign-flipped per-blue mean
        # broadcast across red agents (handled inside reward_fn). Take any red
        # column (they're identical).
        red_rew = rews[:, n_blue]                                           # [T]
        # Done flag: boolean per step; use the broadcast OR across red.
        # Actually env emits `done` as scalar broadcast — pull from any agent.
        done_seq = jnp.any(rews != rews, axis=-1)  # placeholder False-array
        # `dones` from the scan body is per-agent — episode-end flag is a
        # function of the env step. It's boolean and identical across agents
        # for this env, so we just take agent 0.
        # rebuild done_seq from the masks: cumulative, but we only need the
        # per-step done for compute_discounted_returns.
        # Simpler: use mask diffs — first 0 in mask after a 1 marks the end.
        # Cleanest: take done from the scan output; but it's broadcast already.
        # For this env, masking rewards by `mask` already zero-pads after end,
        # so passing `done = (mask diff)` is equivalent.
        # Construct done_seq from mask:
        next_mask = jnp.concatenate([masks[1:], jnp.zeros((1,))], axis=0)
        done_seq = (masks > 0.5) & (next_mask < 0.5)

        returns = compute_discounted_returns(red_rew, done_seq, GAMMA)      # [T]

        # Per-agent log-probs and entropies for red, summed over the n_red axis
        # (equivalent to per-agent REINFORCE with a shared policy + averaging).
        red_acts = acts[:, n_blue:]                                         # [T, n_red]
        red_lp_full = jax.nn.log_softmax(red_logits_seq, axis=-1)           # [T, n_red, A]
        T = red_acts.shape[0]
        red_lp = jnp.take_along_axis(
            red_lp_full, red_acts[:, :, None], axis=-1,
        ).squeeze(-1)                                                       # [T, n_red]
        red_lp_team = jnp.sum(red_lp, axis=-1)                              # [T] joint log-prob
        red_probs = jax.nn.softmax(red_logits_seq, axis=-1)
        red_ent = -jnp.sum(red_probs * red_lp_full, axis=-1)                # [T, n_red]
        red_ent_mean = jnp.mean(red_ent)

        masks_eff = masks                                                    # [T]
        pg = -jnp.sum(red_lp_team * returns * masks_eff) / jnp.sum(masks_eff)
        loss = pg - ENT_COEF * red_ent_mean

        # Telemetry for the caller.
        blue_total = jnp.sum(rews[:, :n_blue]) / jnp.maximum(n_blue, 1)
        red_total = jnp.sum(red_rew)
        return loss, (red_ent_mean, blue_total, red_total)

    grad_fn = jax.value_and_grad(red_loss, has_aux=True)

    def train_step(red_params, opt_state, key):
        (loss, (ent, b_tot, r_tot)), grads = grad_fn(red_params, key)
        updates, opt_state = optimizer.update(grads, opt_state, red_params)
        red_params = optax.apply_updates(red_params, updates)
        return red_params, opt_state, loss, ent, b_tot, r_tot

    return jax.jit(train_step), optimizer


# --------------------------------------------------------------------------- #
# Single arm × single seed driver
# --------------------------------------------------------------------------- #


def run_arm_seed(arm: str, seed: int, cfg: ExperimentConfig, blue_params,
                 env, blue_actor, red_actor, n_blue, n_red, max_steps,
                 enforce_conn) -> dict:
    obs_dim = cfg.obs_dim
    init_fn = _make_red_init(arm, blue_params, red_actor, obs_dim)
    key = jax.random.PRNGKey(seed)
    init_key, train_key = jax.random.split(key)
    red_params = init_fn(init_key)

    train_step, optimizer = _make_train_step(
        env, blue_actor, red_actor, blue_params,
        n_blue, n_red, max_steps, enforce_conn,
    )
    opt_state = optimizer.init(red_params)

    # Snapshot prior parameters for the prior-posterior KL trajectory.
    prior_params = jax.tree_util.tree_map(jnp.array, red_params)

    ent_log = np.zeros(NUM_EPISODES, dtype=np.float32)
    blue_log = np.zeros(NUM_EPISODES, dtype=np.float32)
    red_log = np.zeros(NUM_EPISODES, dtype=np.float32)
    kl_log = np.zeros(NUM_EPISODES // 50 + 1, dtype=np.float32)
    kl_eps = []

    # KL probe: feed the env's first reset obs to both prior and current params,
    # compare softmax distributions per red agent. Cheap and fixed-state.
    probe_key = jax.random.PRNGKey(seed + 1000)
    _o, probe_state = env.reset(probe_key)
    probe_obs = env.obs_array(probe_state)[n_blue:]                          # [n_red, D]

    def kl_value(params_a, params_b):
        la = jax.vmap(red_actor.apply, in_axes=(None, 0))(params_a, probe_obs)
        lb = jax.vmap(red_actor.apply, in_axes=(None, 0))(params_b, probe_obs)
        pa = jax.nn.softmax(la, axis=-1)
        log_pa = jax.nn.log_softmax(la, axis=-1)
        log_pb = jax.nn.log_softmax(lb, axis=-1)
        return jnp.mean(jnp.sum(pa * (log_pa - log_pb), axis=-1))
    kl_value = jax.jit(kl_value)

    print(f"  arm={arm} seed={seed}: training {NUM_EPISODES} eps...")
    t0 = time.time()
    for ep in range(NUM_EPISODES):
        train_key, k = jax.random.split(train_key)
        red_params, opt_state, _loss, ent, b_tot, r_tot = train_step(
            red_params, opt_state, k,
        )
        ent_log[ep] = float(ent)
        blue_log[ep] = float(b_tot)
        red_log[ep] = float(r_tot)
        if ep % 50 == 0:
            kl = float(kl_value(red_params, prior_params))
            kl_log[ep // 50] = kl
            kl_eps.append(ep)
    dt = time.time() - t0
    print(f"  arm={arm} seed={seed}: done in {dt:.1f}s | "
          f"H[red] {ent_log[0]:.3f}->{ent_log[-1]:.3f} | "
          f"blue_R last100 {np.mean(blue_log[-100:]):+.2f}")

    # Save final red params for this (arm, seed) for downstream rendering / eval.
    ckpt_path = OUTPUT_DIR / f"red_{arm}_seed{seed}.npz"
    save_checkpoint(red_params, str(ckpt_path))

    return {
        "arm": arm,
        "seed": seed,
        "red_entropy": ent_log,
        "blue_reward": blue_log,
        "red_reward": red_log,
        "prior_kl": kl_log[: len(kl_eps)],
        "kl_eps": np.asarray(kl_eps, dtype=np.int32),
        "wall_clock": dt,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    args = _parse_args()
    config_path = args.config
    blue_ckpt = args.blue_ckpt
    output_dir = Path(args.output_dir)

    # Patch module-level paths so existing helpers ( _load_blue, run_arm_seed )
    # observe the chosen setup without further plumbing.
    global CONFIG_PATH, BLUE_CKPT, OUTPUT_DIR
    CONFIG_PATH = config_path
    BLUE_CKPT = blue_ckpt
    OUTPUT_DIR = output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    env, n_blue, n_red = _build_env(cfg)
    obs_dim = cfg.obs_dim
    max_steps = cfg.env.max_steps
    enforce_conn = cfg.enforce_connectivity

    blue_actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
        activation=cfg.network.activation,
    )
    # Same-class red: identical Actor architecture.
    red_actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
        activation=cfg.network.activation,
    )

    blue_params = _load_blue(cfg, blue_actor)
    print(f"Loaded blue checkpoint from {BLUE_CKPT}")
    print(f"Setup: n_blue={n_blue} n_red={n_red} obs_dim={obs_dim} "
          f"max_steps={max_steps} arms={ARMS} seeds={NUM_SEEDS} "
          f"eps={NUM_EPISODES}")

    results = []
    for arm in ARMS:
        for seed in range(NUM_SEEDS):
            r = run_arm_seed(
                arm, seed, cfg, blue_params,
                env, blue_actor, red_actor, n_blue, n_red, max_steps,
                enforce_conn,
            )
            results.append(r)

    # Pack into per-arm × per-seed × per-ep arrays.
    A = len(ARMS)
    S = NUM_SEEDS
    E = NUM_EPISODES
    K = len(results[0]["prior_kl"])

    red_entropy = np.zeros((A, S, E), dtype=np.float32)
    blue_reward = np.zeros((A, S, E), dtype=np.float32)
    red_reward = np.zeros((A, S, E), dtype=np.float32)
    prior_kl = np.zeros((A, S, K), dtype=np.float32)
    kl_eps = np.asarray(results[0]["kl_eps"], dtype=np.int32)

    for r in results:
        ai = ARMS.index(r["arm"])
        si = r["seed"]
        red_entropy[ai, si] = r["red_entropy"]
        blue_reward[ai, si] = r["blue_reward"]
        red_reward[ai, si] = r["red_reward"]
        prior_kl[ai, si] = r["prior_kl"]

    out_path = OUTPUT_DIR / "red_prior_phase1.npz"
    np.savez(
        out_path,
        arms=np.asarray(ARMS),
        red_entropy=red_entropy,
        blue_reward=blue_reward,
        red_reward=red_reward,
        prior_kl=prior_kl,
        kl_eps=kl_eps,
        n_blue=np.int32(n_blue),
        n_red=np.int32(n_red),
        num_episodes=np.int32(E),
        num_seeds=np.int32(S),
        config_path=CONFIG_PATH,
        blue_ckpt=BLUE_CKPT,
    )
    print(f"\nSaved {out_path}")
    # Quick text summary.
    print("\nFinal-100ep means (mean ± std across seeds):")
    print(f"  arm | H[red]_final     | blue_R_final     | red_R_final")
    for ai, arm in enumerate(ARMS):
        h = red_entropy[ai, :, -100:].mean(axis=-1)
        b = blue_reward[ai, :, -100:].mean(axis=-1)
        r = red_reward[ai, :, -100:].mean(axis=-1)
        print(f"  {arm}   | {h.mean():.3f} ± {h.std():.3f}   "
              f"| {b.mean():+.3f} ± {b.std():.3f}   "
              f"| {r.mean():+.3f} ± {r.std():.3f}")


if __name__ == "__main__":
    main()
