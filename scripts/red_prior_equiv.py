"""Paired-eval + TOST equivalence test for the I/W/F red-prior arms.

Runs 3 seeds × 200 paired episodes = 600 paired-eval blue_R per arm against the
frozen blue specified by `--config / --blue-ckpt`, then computes per-pair
paired-bootstrap CI and a TOST equivalence verdict at margin Δ (default 0.05).

Why TOST and not just "no significant difference"?
    A non-significant Welch's t with n=600 paired eps is *underpowered evidence*
    for equality. TOST flips the question — instead of asking "is Δ ≠ 0", it
    asks "is |Δ| < margin?". If the entire 95% CI of the paired difference
    sits inside [−margin, +margin], we can declare the arms practically
    equivalent at that margin.

Inputs
------
    --config       experiment YAML for the setup
    --blue-ckpt    frozen blue checkpoint
    --red-dir      directory with red_{I,W,F}_seed{0,1,2}.npz from
                       scripts/red_prior_experiment.py
    --margin       equivalence margin in reward units (default 0.05)
    --output-dir   where to save the npz (default = red-dir)

Output
------
    <output-dir>/equiv_stats.npz   — paired blue_R [3, 600] + per-pair Δ stats

Run examples
------------
    # C2
    python scripts/red_prior_equiv.py \
      --config experiments/compromise-16x16-5-3b2r-coevo/config.yaml \
      --blue-ckpt experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz \
      --red-dir experiments/red-prior-phase1

    # C1
    python scripts/red_prior_equiv.py \
      --config experiments/compromise-16x16-5-4b1r-coevo/config.yaml \
      --blue-ckpt experiments/compromise-16x16-5-4b1r-coevo/checkpoint.npz \
      --red-dir experiments/red-prior-phase1-C1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import load_checkpoint, unflatten_params
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.types import CELL_WALL


ARMS = ("I", "W", "F")
PAIRS = (("I", "W"), ("I", "F"), ("W", "F"))
NUM_SEEDS = 3
EVAL_EPISODES = 200             # per (arm, seed); 600 total per arm
EVAL_SEED_OFFSET = 100          # paired keys are deterministic in this offset


# --------------------------------------------------------------------------- #
def _build_env(cfg: ExperimentConfig):
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
    return GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn), n_blue, n_red


def _load_actor_params(ckpt_path: str, actor: Actor, obs_dim: int):
    flat = load_checkpoint(ckpt_path)
    if isinstance(flat, dict) and "actor" in flat:
        flat = flat["actor"]
    sample = jnp.zeros((obs_dim,))
    init_params = actor.init(jax.random.PRNGKey(0), sample)
    return unflatten_params(flat, init_params)


def _make_rollout(env, blue_actor, red_actor, n_blue, n_red, max_steps,
                  enforce_conn):
    from red_within_blue.training.rollout import _connectivity_guardrail
    num_agents = env.config.num_agents

    def rollout(blue_params, red_params, key):
        key, reset_key = jax.random.split(key)
        _o, state = env.reset(reset_key)

        def body(carry, _):
            state, rng, cum_done = carry
            rng, bk, rk, sk = jax.random.split(rng, 4)
            obs_all = env.obs_array(state)
            blue_obs = obs_all[:n_blue]
            red_obs = obs_all[n_blue:]
            bl = jax.vmap(blue_actor.apply, in_axes=(None, 0))(blue_params, blue_obs)
            rl = jax.vmap(red_actor.apply,  in_axes=(None, 0))(red_params,  red_obs)
            ba = jax.vmap(jax.random.categorical)(jax.random.split(bk, n_blue), bl)
            ra = jax.vmap(jax.random.categorical)(jax.random.split(rk, n_red),  rl)
            actions = jnp.concatenate([ba, ra], axis=0)
            safe = jax.lax.cond(
                jnp.bool_(enforce_conn) & (num_agents >= 2),
                lambda a: _connectivity_guardrail(
                    state.agent_state.positions, state.agent_state.comm_ranges,
                    a, state.global_state.grid.terrain),
                lambda a: a, actions,
            )
            _o, new_state, rewards, done, _i = env.step_array(sk, state, safe)
            mask = 1.0 - cum_done.astype(jnp.float32)
            masked = rewards * mask
            new_cum_done = cum_done | done
            explored = state.global_state.grid.explored
            terrain = state.global_state.grid.terrain
            non_wall = (terrain != CELL_WALL)
            cov = jnp.sum((explored > 0) & non_wall) / jnp.maximum(jnp.sum(non_wall), 1)
            return (new_state, rng, new_cum_done), (masked, cov)

        _, (rew_seq, cov_seq) = jax.lax.scan(
            body, (state, key, jnp.bool_(False)), jnp.arange(max_steps),
        )
        blue_total = jnp.sum(rew_seq[:, :n_blue]) / n_blue
        return blue_total, cov_seq[-1]

    return jax.jit(rollout)


def _paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, (n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def _tost_verdict(lo: float, hi: float, margin: float) -> str:
    """TOST: equivalent if entire CI ⊂ [−margin, +margin]."""
    if lo >= -margin and hi <= +margin:
        return "equivalent"
    if hi <  -margin or lo >  +margin:
        return "different"
    return "inconclusive"


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--blue-ckpt",  required=True)
    ap.add_argument("--red-dir",    required=True)
    ap.add_argument("--margin",     type=float, default=0.05)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir or args.red_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.from_yaml(args.config)
    env, n_blue, n_red = _build_env(cfg)
    obs_dim = cfg.obs_dim

    actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
        activation=cfg.network.activation,
    )
    blue_params = _load_actor_params(args.blue_ckpt, actor, obs_dim)
    print(f"Setup: n_blue={n_blue} n_red={n_red} obs_dim={obs_dim} "
          f"max_steps={cfg.env.max_steps} margin=±{args.margin}")

    rollout = _make_rollout(env, actor, actor, n_blue, n_red,
                            cfg.env.max_steps, cfg.enforce_connectivity)

    red_dir = Path(args.red_dir)
    red_params = {
        (a, s): _load_actor_params(str(red_dir / f"red_{a}_seed{s}.npz"),
                                   actor, obs_dim)
        for a in ARMS for s in range(NUM_SEEDS)
    }

    total = NUM_SEEDS * EVAL_EPISODES
    eval_keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED_OFFSET),
                                 EVAL_EPISODES * NUM_SEEDS
                                 ).reshape(NUM_SEEDS, EVAL_EPISODES, 2)

    blue_R = {a: np.zeros(total) for a in ARMS}
    cov    = {a: np.zeros(total) for a in ARMS}

    t0 = time.time()
    for a in ARMS:
        for s in range(NUM_SEEDS):
            rp = red_params[(a, s)]
            for i in range(EVAL_EPISODES):
                b, c = rollout(blue_params, rp, eval_keys[s, i])
                idx = s * EVAL_EPISODES + i
                blue_R[a][idx] = float(b)
                cov[a][idx]    = float(c)
        print(f"  {a}: blue_R = {blue_R[a].mean():+.3f} ± {blue_R[a].std():.3f} "
              f"| cov_final = {cov[a].mean():.3f}")
    print(f"  total {time.time() - t0:.1f}s")

    # ----- Paired bootstrap + TOST verdicts ---------------------------------
    pair_stats: dict[str, dict] = {}
    for x, y in PAIRS:
        diff = blue_R[x] - blue_R[y]
        m, lo, hi = _paired_bootstrap_ci(diff)
        verdict = _tost_verdict(lo, hi, args.margin)
        pair_stats[f"{x}_{y}"] = {
            "mean": m, "lo": lo, "hi": hi, "verdict": verdict,
        }
        print(f"  Δ {x}-{y}: {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  → {verdict}")

    out_npz = out_dir / "equiv_stats.npz"
    np.savez(
        out_npz,
        arms=np.asarray(ARMS),
        pairs=np.asarray([f"{x}_{y}" for x, y in PAIRS]),
        margin=np.float64(args.margin),
        blue_R=np.stack([blue_R[a] for a in ARMS]),
        coverage=np.stack([cov[a]    for a in ARMS]),
        diff_mean=np.asarray([pair_stats[f"{x}_{y}"]["mean"]    for x, y in PAIRS]),
        diff_lo  =np.asarray([pair_stats[f"{x}_{y}"]["lo"]      for x, y in PAIRS]),
        diff_hi  =np.asarray([pair_stats[f"{x}_{y}"]["hi"]      for x, y in PAIRS]),
        verdicts =np.asarray([pair_stats[f"{x}_{y}"]["verdict"] for x, y in PAIRS]),
        n_blue=np.int32(n_blue), n_red=np.int32(n_red),
    )
    print(f"\nwrote {out_npz}")


if __name__ == "__main__":
    main()
