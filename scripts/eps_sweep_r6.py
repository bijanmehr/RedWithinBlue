"""Guardrail-relaxation sweep on the r6 coevolutionary policies.

This is the paper's ΔJ vs ε Pareto experiment, built out of machinery we
already have. Blue + joint-red were trained against a *hard* connectivity
guardrail (ε = 0 in the Stealth-Attacks-on-Swarms formulation). At eval
time we relax that constraint — either by turning the guardrail off
entirely or by giving the disconnect timer a finite grace budget — and
measure how much blue's return degrades.

ε-axis interpretation (paper: "how long may the attacker stealthily
deviate before the swarm detects it and the mission aborts"):

  eps=0          : enforce=True,  grace=0    — training baseline
  eps=5..300     : enforce=False, grace=G    — attacker may deviate for ≤G
                                                consecutive steps; on the
                                                G+1'th the episode terminates
                                                (== "mission aborted")
  eps=inf        : enforce=False, grace=∞    — attacker completely unbounded

Metrics per setting, averaged over ``N_EVAL_SEEDS`` independent rollouts:
- blue mean return  J(π, φ)  — paper's primary quantity
- red  mean return  J_red    — paper's -J
- early-termination % (fraction of eps where the grace budget was spent)
- mean terminal step

Run: ``python scripts/eps_sweep_r6.py``.  ~30 s after JIT warmup.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import collect_episode_multi_scan_joint

CONFIG_PATH = "configs/adv-ladder-r6-32x32-6b4r.yaml"
EXPERIMENT_DIR = Path("experiments/adv-ladder-r6-coevo")
OUTPUT_PATH = EXPERIMENT_DIR / "eps_sweep.npz"

N_EVAL_SEEDS = 20


def _load_blue(cfg: ExperimentConfig, ckpt_path: Path) -> Tuple[Actor, dict]:
    flat = load_checkpoint(str(ckpt_path))
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
    )
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {k: (v[0] if v.ndim == ref_flat[k].ndim + 1 else v) for k, v in flat.items()}
    return actor, unflatten_params(stripped, ref)


def _load_red(cfg: ExperimentConfig, ckpt_path: Path) -> Tuple[JointRedActor, dict]:
    flat = load_checkpoint(str(ckpt_path))
    n_red = cfg.env.num_red_agents
    obs_dim = cfg.obs_dim
    actor = JointRedActor(
        num_red=n_red,
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.train.red_hidden_dim,
        num_layers=cfg.train.red_num_layers,
    )
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(n_red * obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {k: (v[0] if v.ndim == ref_flat[k].ndim + 1 else v) for k, v in flat.items()}
    return actor, unflatten_params(stripped, ref)


def _build_env(cfg: ExperimentConfig, *, grace: int) -> GridCommEnv:
    env_cfg = cfg.to_env_config()
    env_cfg = replace(env_cfg, disconnect_grace=int(grace))
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
    return GridCommEnv(env_cfg, reward_fn=reward_fn)


def _settings() -> List[Dict[str, Any]]:
    # (label, enforce_connectivity, disconnect_grace).
    # grace = 0 with enforce=True is the training-time baseline.
    return [
        {"label": "eps=0 (hard guardrail)", "enforce": True,  "grace": 0},
        {"label": "eps=5  steps  (soft)",    "enforce": False, "grace": 5},
        {"label": "eps=15 steps  (soft)",    "enforce": False, "grace": 15},
        {"label": "eps=30 steps  (soft)",    "enforce": False, "grace": 30},
        {"label": "eps=100 steps (soft)",    "enforce": False, "grace": 100},
        {"label": "eps=inf (no enforcement)", "enforce": False, "grace": 10_000},
    ]


def main():
    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    n_red = cfg.env.num_red_agents
    n_blue = cfg.env.num_agents - n_red
    max_steps = cfg.env.max_steps
    obs_dim = cfg.obs_dim

    blue_actor, blue_params = _load_blue(cfg, EXPERIMENT_DIR / "checkpoint.npz")
    red_actor, red_params = _load_red(cfg, EXPERIMENT_DIR / "joint_red_checkpoint.npz")

    print(
        f"eps-sweep | {cfg.experiment_name} | n_blue={n_blue} n_red={n_red} "
        f"max_steps={max_steps} N_EVAL_SEEDS={N_EVAL_SEEDS}\n"
    )
    header = (
        f"{'setting':<30}  {'blue_ret':>8}  {'blue_std':>8}  "
        f"{'red_ret':>8}  {'ep_len':>7}  {'early_term%':>11}"
    )
    print(header)
    print("-" * len(header))

    results: Dict[str, Dict[str, Any]] = {}

    for s in _settings():
        env = _build_env(cfg, grace=int(s["grace"]))

        # One JIT per setting — enforce_connectivity is a python bool closed
        # over, and env geometry is fixed.
        def _single_episode(key, enforce=bool(s["enforce"])):
            traj = collect_episode_multi_scan_joint(
                env=env,
                blue_actor=blue_actor,
                blue_params=blue_params,
                joint_red_actor=red_actor,
                joint_red_params=red_params,
                key=key,
                max_steps=max_steps,
                num_red_agents=n_red,
                enforce_connectivity=enforce,
            )
            # rewards[T, N] already masked by cumulative-done.
            team_total = jnp.sum(traj.rewards, axis=0)          # [N]
            blue_total = jnp.sum(team_total[:n_blue]) / n_blue  # scalar
            red_total = jnp.sum(team_total[n_blue:]) / n_red     # scalar
            # episode length = # of steps executed before the global done flag
            # flips. traj.dones is scalar-per-step, shape [T].
            step_done = traj.dones.astype(jnp.int32)
            ever_done = jnp.cumsum(step_done) > 0
            ep_len = jnp.sum((~ever_done).astype(jnp.int32)) + jnp.int32(1)
            ep_len = jnp.minimum(ep_len, jnp.int32(max_steps))
            early = ep_len < jnp.int32(max_steps)
            return blue_total, red_total, ep_len, early

        eval_jit = jax.jit(_single_episode)

        blue_rets: List[float] = []
        red_rets: List[float] = []
        ep_lens: List[int] = []
        earlies: List[int] = []

        for seed in range(N_EVAL_SEEDS):
            k = jax.random.PRNGKey(seed)
            bret, rret, L, early = eval_jit(k)
            blue_rets.append(float(bret))
            red_rets.append(float(rret))
            ep_lens.append(int(L))
            earlies.append(int(early))

        bmean = float(np.mean(blue_rets))
        bstd = float(np.std(blue_rets))
        rmean = float(np.mean(red_rets))
        lmean = float(np.mean(ep_lens))
        epct = 100.0 * float(np.mean(earlies))

        print(
            f"{s['label']:<30}  {bmean:>+8.3f}  {bstd:>8.3f}  "
            f"{rmean:>+8.3f}  {lmean:>7.1f}  {epct:>10.1f}%"
        )

        results[s["label"]] = {
            "blue_ret": blue_rets,
            "red_ret": red_rets,
            "ep_len": ep_lens,
            "early_term": earlies,
            "grace": int(s["grace"]),
            "enforce": bool(s["enforce"]),
        }

    np.savez(
        OUTPUT_PATH,
        labels=np.array(list(results.keys())),
        blue_ret=np.array([results[k]["blue_ret"] for k in results]),
        red_ret=np.array([results[k]["red_ret"] for k in results]),
        ep_len=np.array([results[k]["ep_len"] for k in results]),
        early_term=np.array([results[k]["early_term"] for k in results]),
        graces=np.array([results[k]["grace"] for k in results]),
        enforces=np.array([results[k]["enforce"] for k in results]),
    )
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
