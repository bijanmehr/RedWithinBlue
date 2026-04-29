"""Render a single experiment's metrics + eval GIF into an HTML report.

Given a finished training run (one directory containing ``checkpoint.npz`` and
``metrics.npz``) plus the YAML config used to produce it, this module:

  1. Rebuilds the blue actor and loads its parameters from the checkpoint.
  2. Runs one evaluation episode, recording every step into an animated GIF.
  3. Converts ``metrics.npz`` into per-episode records.
  4. Renders a self-contained ``report.html`` embedding the learning curves,
     coverage stats, action distribution, visitation heatmap, connectivity
     timeline, and the interactive GIF player.

The training runner does not yet emit this report automatically — it is
invoked as a post-processing step so the user can regenerate it at any time.

Usage::

    python -m red_within_blue.analysis.experiment_report \\
        --config configs/solo-explore.yaml \\
        --experiment-dir experiments/solo-explore
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

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
from red_within_blue.training.gif import record_episode_gif
from red_within_blue.training.networks import Actor
from red_within_blue.training.report import generate_report
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward,
    normalized_exploration_reward,
)


# ---------------------------------------------------------------------------
# Actor reconstruction from a checkpoint
# ---------------------------------------------------------------------------


def _load_blue_actor_params(config: ExperimentConfig, ckpt_path: Path):
    """Rebuild the blue actor and load its parameters from a checkpoint."""
    flat = load_checkpoint(str(ckpt_path))
    actor_flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}

    dummy_obs = jnp.zeros(config.obs_dim)
    actor = Actor(
        num_actions=config.env.num_actions,
        hidden_dim=config.network.actor_hidden_dim,
        num_layers=config.network.actor_num_layers,
    )
    ref_params = actor.init(jax.random.PRNGKey(0), dummy_obs)

    # Multi-seed training stacks params with a leading seed axis; single-seed
    # coevo/ES checkpoints don't. Decide per-leaf whether to strip a leading
    # dim by comparing ndim against the ref leaf (see _load_joint_red_actor_params).
    ref_flat = flatten_params(ref_params)
    stripped = {}
    for k, v in actor_flat.items():
        ref_nd = ref_flat[k].ndim
        stripped[k] = v[0] if v.ndim == ref_nd + 1 else v
    params = unflatten_params(stripped, ref_params)
    return actor, params


def _build_policy_fn(actor, params):
    """Return ``(key, obs) -> int`` that samples from the actor's logits."""
    def policy_fn(key, obs):
        logits = actor.apply(params, obs)
        return int(jax.random.categorical(key, logits))
    return policy_fn


def _load_joint_red_actor_params(config: ExperimentConfig, ckpt_path: Path):
    """Rebuild the centralized joint-red actor and load its parameters."""
    from red_within_blue.training.networks import JointRedActor
    flat = load_checkpoint(str(ckpt_path))
    n_red = config.env.num_red_agents
    obs_dim = config.obs_dim
    actor = JointRedActor(
        num_red=n_red,
        num_actions=config.env.num_actions,
        hidden_dim=config.train.red_hidden_dim,
        num_layers=config.train.red_num_layers,
    )
    ref_params = actor.init(jax.random.PRNGKey(0), jnp.zeros(n_red * obs_dim))
    # Strip a leading seed axis per-leaf rather than globally: a coevo/single-
    # seed checkpoint has ndim matching the ref exactly (no seed axis), while a
    # multi-seed trainer checkpoint has ndim+1 (leading num_seeds). Compare
    # every leaf against its own ref shape instead of one representative key.
    ref_flat = flatten_params(ref_params)
    stripped = {}
    for k, v in flat.items():
        ref_nd = ref_flat[k].ndim
        stripped[k] = v[0] if v.ndim == ref_nd + 1 else v
    params = unflatten_params(stripped, ref_params)
    return actor, params


# ---------------------------------------------------------------------------
# Metrics handling
# ---------------------------------------------------------------------------


def _metrics_to_records(metrics: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Convert ``metrics.npz`` arrays into one dict per episode.

    Supports both single-seed ``[episodes]`` and multi-seed ``[seeds, episodes]``
    arrays. For multi-seed, values are averaged across seeds.
    """
    records: List[Dict[str, Any]] = []
    items = {k: np.asarray(v) for k, v in metrics.items()}
    if not items:
        return records

    # Determine episode axis
    sample = next(iter(items.values()))
    if sample.ndim == 1:
        n_eps = sample.shape[0]
        def value(arr, i):
            return float(arr[i])
    elif sample.ndim == 2:
        n_eps = sample.shape[1]
        def value(arr, i):
            return float(arr[:, i].mean())
    else:
        # Higher-rank arrays (e.g. per_agent_reward [S, E, N]): skip
        return records

    scalar_keys = [k for k, v in items.items() if v.ndim == sample.ndim]

    for i in range(n_eps):
        record = {"step": i + 1}
        for k in scalar_keys:
            v = items[k]
            if v.ndim not in (1, 2):
                continue
            try:
                record[k] = value(v, i)
            except Exception:
                continue
        records.append(record)

    return records


def _final_coverages(metrics: Dict[str, np.ndarray], window: int = 100) -> List[float]:
    """Per-seed final coverage averaged over the last ``window`` episodes."""
    key = "blue_total_reward" if "blue_total_reward" in metrics else "total_reward"
    if key not in metrics:
        return []
    arr = np.asarray(metrics[key])
    if arr.ndim == 1:
        arr = arr[None, :]
    w = min(window, arr.shape[1])
    return [float(x) for x in arr[:, -w:].mean(axis=1)]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def build_report(
    config_path: Path,
    experiment_dir: Path,
    fps: int = 4,
    eval_seed: int = 42,
) -> Path:
    """Build ``report.html`` + ``episode.gif`` for one experiment directory."""
    config = ExperimentConfig.from_yaml(str(config_path))

    ckpt_path = experiment_dir / "checkpoint.npz"
    metrics_path = experiment_dir / "metrics.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics:    {metrics_path}")

    # Build the eval environment. Reward choice mirrors the training path:
    # zero-sum when there are red agents, otherwise cooperative.
    env_cfg = config.to_env_config()
    if config.env.num_red_agents > 0:
        reward_fn = normalized_competitive_reward
    else:
        reward_fn = normalized_exploration_reward
    env = GridCommEnv(env_cfg, reward_fn=reward_fn)

    actor, params = _load_blue_actor_params(config, ckpt_path)
    policy_fn = _build_policy_fn(actor, params)

    joint_red_actor = None
    joint_red_params = None
    n_red = config.env.num_red_agents
    if n_red > 0 and getattr(config.train, "red_policy", None) == "joint":
        joint_red_ckpt = experiment_dir / "joint_red_checkpoint.npz"
        if joint_red_ckpt.exists():
            joint_red_actor, joint_red_params = _load_joint_red_actor_params(
                config, joint_red_ckpt,
            )

    # --- Record the episode ---
    gif_path = experiment_dir / "episode.gif"
    gif_info = record_episode_gif(
        env=env,
        policy_fn=policy_fn,
        key=jax.random.PRNGKey(eval_seed),
        output_path=str(gif_path),
        fps=fps,
        enforce_connectivity=config.enforce_connectivity,
        joint_red_actor=joint_red_actor,
        joint_red_params=joint_red_params,
        n_red=n_red if joint_red_actor is not None else 0,
    )

    # --- STAY-source breakdown (was the agent forced to STAY by the
    # connectivity guardrail, or did its policy actually choose STAY?) ---
    steps_total = int(gif_info.get("steps_total", 0)) or 1
    stay_int = gif_info.get("stay_intended", [])
    stay_frc = gif_info.get("stay_forced", [])
    move_t = gif_info.get("move_taken", [])
    print(f"[gif] STAY breakdown over {steps_total} steps:")
    for i, agent in enumerate(env.agents):
        if i >= len(stay_int):
            break
        ints = int(stay_int[i]); frc = int(stay_frc[i]); mv = int(move_t[i])
        tot = max(1, ints + frc + mv)
        print(
            f"  {agent}: stay_intended={ints} ({100*ints/tot:.0f}%)  "
            f"stay_forced={frc} ({100*frc/tot:.0f}%)  "
            f"moved={mv} ({100*mv/tot:.0f}%)"
        )
    print(f"[gif] blue ever-known: {gif_info.get('blue_ever_known_pct', 0.0):.1f}%")

    # --- Action distribution over the recorded episode ---
    # Replay the same policy on a fresh episode to gather action counts.
    action_dist = _sample_action_distribution(
        env, policy_fn, jax.random.PRNGKey(eval_seed + 1),
    )

    # --- Metrics → records ---
    metrics_raw = dict(np.load(metrics_path, allow_pickle=False))
    records = _metrics_to_records(metrics_raw)
    coverages = _final_coverages(metrics_raw)

    # --- Hyperparameter summary (shown in the HTML) ---
    hyperparams = {
        "experiment": config.experiment_name,
        "grid": f"{config.env.grid_width}x{config.env.grid_height}",
        "num_agents": config.env.num_agents,
        "num_red_agents": config.env.num_red_agents,
        "wall_density": config.env.wall_density,
        "max_steps": config.env.max_steps,
        "comm_radius": config.env.comm_radius,
        "obs_radius": config.env.obs_radius,
        "view_radius": (
            config.env.view_radius
            if config.env.view_radius >= 0
            else config.env.obs_radius
        ),
        "survey_radius": (
            config.env.survey_radius
            if config.env.survey_radius >= 0
            else config.env.obs_radius
        ),
        "local_obs": config.env.local_obs,
        "disconnect_grace": config.env.disconnect_grace,
        "disconnect_fail_penalty": config.env.disconnect_fail_penalty,
        "disconnect_mode": config.env.disconnect_mode,
        "actor_hidden_dim": config.network.actor_hidden_dim,
        "actor_num_layers": config.network.actor_num_layers,
        "method": config.train.method,
        "lr": config.train.lr,
        "gamma": config.train.gamma,
        "num_episodes": config.train.num_episodes,
        "num_seeds": config.train.num_seeds,
        "warm_start": config.warm_start or "none",
        "connectivity_guardrail": config.enforce_connectivity,
        "cooperative_weight": config.reward.cooperative_weight,
        "revisit_weight": config.reward.revisit_weight,
        "isolation_weight": config.reward.isolation_weight,
        "spread_weight": config.reward.spread_weight,
        "terminal_bonus_scale": config.reward.terminal_bonus_scale,
    }

    title = f"Experiment: {config.experiment_name}"
    out_path = experiment_dir / "report.html"
    generate_report(
        title=title,
        output_path=str(out_path),
        hyperparams=hyperparams,
        coverages=coverages or None,
        action_dist=action_dist,
        metrics=records,
        gif_path=str(gif_path),
        visit_heatmap=gif_info["visit_heatmap"],
        connectivity=gif_info["connectivity"],
        coverage_over_time=gif_info["coverage_over_time"],
        max_steps=config.env.max_steps,
    )
    return out_path


def _sample_action_distribution(env, policy_fn, key) -> List[float]:
    """Run one episode and return the action-frequency distribution."""
    from red_within_blue.wrappers import TrajectoryWrapper
    wrapper = TrajectoryWrapper(env)
    key, reset_key = jax.random.split(key)
    obs, state = wrapper.reset(reset_key)
    counts = np.zeros(env.config.num_actions, dtype=np.int64)
    done = False
    while not done:
        num_splits = 1 + len(env.agents) + 1
        keys = jax.random.split(key, num_splits)
        key = keys[0]
        agent_keys = keys[1:1 + len(env.agents)]
        step_key = keys[1 + len(env.agents)]
        action_dict = {}
        for i, agent in enumerate(env.agents):
            action = int(policy_fn(agent_keys[i], obs[agent]))
            counts[action] += 1
            action_dict[agent] = jnp.int32(action)
        obs, state, rewards, dones, info = wrapper.step(step_key, state, action_dict)
        done = bool(dones["__all__"])
    total = max(1, int(counts.sum()))
    return [float(c / total) for c in counts]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="red_within_blue.analysis.experiment_report",
        description="Render HTML report + eval GIF for one experiment.",
    )
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the YAML config that produced the run.")
    parser.add_argument("--experiment-dir", required=True, type=str,
                        help="Directory containing checkpoint.npz and metrics.npz.")
    parser.add_argument("--fps", type=int, default=4,
                        help="Frames per second in the episode GIF.")
    parser.add_argument("--eval-seed", type=int, default=42)
    args = parser.parse_args(argv)

    out = build_report(
        config_path=Path(args.config),
        experiment_dir=Path(args.experiment_dir),
        fps=args.fps,
        eval_seed=args.eval_seed,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
