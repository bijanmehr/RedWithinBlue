"""CLI entry point for running RedWithinBlue training experiments.

Usage::

    python -m red_within_blue.training.runner --config configs/stage1.yaml
    python -m red_within_blue.training.runner --config configs/stage2.yaml --num-seeds 3
    python -m red_within_blue.training.runner --config configs/stage1.yaml --output-dir /tmp/runs
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import jax
import numpy as np

from red_within_blue.training.config import ExperimentConfig, TrainParams
from red_within_blue.training.checkpoint import (
    save_checkpoint, load_checkpoint, unflatten_params,
)
from red_within_blue.training.networks import Actor, Critic
from red_within_blue.training.trainer import make_train, make_train_multi_seed


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="red_within_blue.training.runner",
        description="Run a RedWithinBlue training experiment from a YAML config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory from the config.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Override the number of seeds from the config.",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Path to a checkpoint.npz to warm-start from (overrides config).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : list of str or None
        If ``None``, reads from ``sys.argv``.
    """
    parser = build_parser()
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading + CLI override application
# ---------------------------------------------------------------------------


def load_config_with_overrides(args: argparse.Namespace) -> ExperimentConfig:
    """Load YAML config and apply any CLI overrides."""
    config = ExperimentConfig.from_yaml(args.config)

    replacements: dict = {}

    if args.output_dir is not None:
        replacements["output_dir"] = args.output_dir

    if args.num_seeds is not None:
        replacements["train"] = dataclasses.replace(
            config.train, num_seeds=args.num_seeds,
        )

    if args.warm_start is not None:
        replacements["warm_start"] = args.warm_start

    if replacements:
        config = dataclasses.replace(config, **replacements)

    return config


def load_warm_start_params(config: ExperimentConfig):
    """Load actor (and optionally critic) params from a warm-start checkpoint.

    Returns ``(actor_params, critic_params)`` where critic_params may be None.
    """
    import jax.numpy as jnp

    ckpt_path = config.warm_start
    flat = load_checkpoint(ckpt_path)

    # Split actor vs critic keys
    actor_flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    critic_flat = {k.removeprefix("critic/"): v for k, v in flat.items() if k.startswith("critic/")}

    # Build reference params from the config's network shape
    dummy_obs = jnp.zeros(config.obs_dim)
    key = jax.random.PRNGKey(0)

    actor = Actor(
        num_actions=config.env.num_actions,
        hidden_dim=config.network.actor_hidden_dim,
        num_layers=config.network.actor_num_layers,
    )
    ref_actor = actor.init(key, dummy_obs)
    actor_params = unflatten_params(actor_flat, ref_actor)

    critic_params = None
    if critic_flat and config.train.method == "actor_critic":
        critic = Critic(
            hidden_dim=config.network.critic_hidden_dim,
            num_layers=config.network.critic_num_layers,
        )
        ref_critic = critic.init(key, dummy_obs)
        critic_params = unflatten_params(critic_flat, ref_critic)

    return actor_params, critic_params


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def run_training(config: ExperimentConfig) -> tuple:
    """Build trainer, run training, return (actor_params, critic_params, metrics)."""
    num_seeds = config.train.num_seeds

    # Load warm-start params if specified
    init_actor, init_critic = None, None
    if config.warm_start is not None:
        print(f"Warm-starting from: {config.warm_start}")
        init_actor, init_critic = load_warm_start_params(config)

    if num_seeds > 1:
        train_fn = make_train_multi_seed(config, init_actor, init_critic)
    else:
        train_fn = make_train(config, init_actor, init_critic)

    key = jax.random.PRNGKey(0)
    actor_params, critic_params, metrics = train_fn(key)

    return actor_params, critic_params, metrics


def save_results(
    config: ExperimentConfig,
    actor_params,
    critic_params,
    metrics: dict,
) -> Path:
    """Save checkpoints and metrics to the output directory.

    Returns the output directory path.
    """
    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint (actor + critic if present)
    ckpt_path = out_dir / "checkpoint.npz"
    save_checkpoint(actor_params, str(ckpt_path), critic_params=critic_params)

    # Save metrics as .npz
    metrics_path = out_dir / "metrics.npz"
    metrics_np = {k: np.asarray(v) for k, v in metrics.items()}
    np.savez(str(metrics_path), **metrics_np)

    return out_dir


def print_summary(config: ExperimentConfig, metrics: dict, out_dir: Path, elapsed: float) -> None:
    """Print a human-readable training summary to stdout."""
    loss = np.asarray(metrics["loss"])
    reward = np.asarray(metrics["total_reward"])

    num_seeds = config.train.num_seeds

    print()
    print("=" * 60)
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Method:     {config.train.method}")
    print(f"  Agents:     {config.env.num_agents}")
    print(f"  Grid:       {config.env.grid_width}x{config.env.grid_height}")
    print(f"  Episodes:   {config.train.num_episodes}")
    print(f"  Seeds:      {num_seeds}")
    print(f"  Time:       {elapsed:.1f}s")
    print("-" * 60)

    if num_seeds > 1:
        # metrics shape: [num_seeds, num_episodes]
        final_loss = loss[:, -1]
        final_reward = reward[:, -1]
        print(f"  Final loss:   {np.mean(final_loss):.4f} +/- {np.std(final_loss):.4f}")
        print(f"  Final reward: {np.mean(final_reward):.2f} +/- {np.std(final_reward):.2f}")
    else:
        # metrics shape: [num_episodes]
        print(f"  Final loss:   {loss[-1]:.4f}")
        print(f"  Final reward: {reward[-1]:.2f}")

    print("-" * 60)
    print(f"  Output: {out_dir}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    config = load_config_with_overrides(args)

    print(f"Loading config from: {args.config}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Training {config.train.num_episodes} episodes "
          f"x {config.train.num_seeds} seed(s) ...")

    t0 = time.time()
    actor_params, critic_params, metrics = run_training(config)
    elapsed = time.time() - t0

    out_dir = save_results(config, actor_params, critic_params, metrics)
    print_summary(config, metrics, out_dir, elapsed)


if __name__ == "__main__":
    main()
