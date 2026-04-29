"""CLI entry point for running RedWithinBlue training experiments.

Usage::

    python -m red_within_blue.training.runner --config configs/stage1.yaml
    python -m red_within_blue.training.runner --config configs/stage2.yaml --num-seeds 3
    python -m red_within_blue.training.runner --config configs/stage1.yaml --output-dir /tmp/runs
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Optional

import jax
import numpy as np

from red_within_blue.training.config import ExperimentConfig, TrainParams
from red_within_blue.training.checkpoint import (
    flatten_params, save_checkpoint, load_checkpoint, unflatten_params,
)
from red_within_blue.training.networks import Actor, Critic
from red_within_blue.training.trainer import make_train, make_train_multi_seed
from red_within_blue.training import progress as progress_bar


# Observation layout tail: map_fraction(1) + pos_xy(2) + uid(1) + team(1). Must
# match ExperimentConfig.obs_dim in config.py.
_OBS_TAIL_DIM = 5


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


def _upsample_first_layer_for_grid(
    flat: dict[str, np.ndarray],
    config: ExperimentConfig,
    num_blocks: int = 1,
    source_num_blocks: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Spatially upsample a checkpoint's ``Dense_0/kernel`` when the grid size
    changed between training and the target config.

    The per-agent observation layout is
    ``[scan(S), grid_seen_mask(H*W), tail(5)]``; only the grid rows depend on
    grid shape. Deeper layers are grid-invariant, so ``Dense_0`` is the only
    layer that needs resizing.

    ``num_blocks`` controls how the kernel's input axis is interpreted in the
    target config:

    * ``num_blocks=1`` (default) — actor and single-agent critic: the kernel
      input axis is one copy of the per-agent obs layout.
    * ``num_blocks=N`` (``N == env.num_agents``) — CTDE central critic: the
      kernel input axis is ``N`` concatenated copies (losses.py reshapes
      ``observations[T, N, obs_dim] -> [T, N*obs_dim]`` before the critic
      call). Each block is upsampled independently with the same rule and
      the blocks are concatenated back in order, preserving the
      block-per-agent structure the trainer expects.

    ``source_num_blocks`` is the number of blocks the *source* kernel was
    trained with. If ``None`` (default), it equals ``num_blocks`` — the same-N
    case. When set to a smaller integer that divides ``num_blocks`` (e.g.
    source N=4, target N=8), each source per-agent block is upsampled
    independently and the result is then *tiled* to fill the target's blocks.
    This is the principled extension because the central critic is symmetric
    in agent identity (agents are exchangeable). Requires
    ``num_blocks % source_num_blocks == 0``.

    Scan and tail rows copy verbatim. Any leading seed axis is preserved
    transparently. Non-``Dense_0`` arrays (biases, deeper kernels) are
    untouched. No-op if the kernel's input dim already matches
    ``num_blocks * config.obs_dim``.
    """
    import jax.numpy as jnp
    import jax.image as jimage

    kernel_key = "params/Dense_0/kernel"
    if kernel_key not in flat:
        return flat

    if source_num_blocks is None:
        source_num_blocks = num_blocks
    if num_blocks % source_num_blocks != 0:
        raise ValueError(
            f"Target num_blocks ({num_blocks}) must be an integer multiple of "
            f"source_num_blocks ({source_num_blocks}). Cannot tile per-agent "
            f"blocks."
        )
    tile_factor = num_blocks // source_num_blocks

    kernel = flat[kernel_key]
    old_input_dim = int(kernel.shape[-2])
    per_block_new = config.obs_dim
    new_input_dim = per_block_new * num_blocks
    if old_input_dim == new_input_dim:
        return flat

    if old_input_dim % source_num_blocks != 0:
        raise ValueError(
            f"Warm-start kernel input ({old_input_dim}) is not divisible by "
            f"source_num_blocks={source_num_blocks}; cannot split into "
            f"per-agent blocks."
        )
    per_block_old = old_input_dim // source_num_blocks

    view_r = config.env.view_radius if config.env.view_radius >= 0 else config.env.obs_radius
    scan_dim = (2 * view_r + 1) ** 2

    # Fast path: per-block obs_dim already matches (same view_radius + same
    # seen-field layout). Only N tiling is needed, no grid-row upsampling.
    # This covers:
    #   - local_obs=True transfers at any grid size (seen field is view-sized
    #     and therefore grid-invariant).
    #   - local_obs=False transfers where the grid size didn't change.
    if per_block_old == per_block_new:
        import jax.numpy as jnp
        k = jnp.asarray(kernel)
        source_blocks = [
            k[..., b * per_block_old:(b + 1) * per_block_old, :]
            for b in range(source_num_blocks)
        ]
        new_blocks = source_blocks * tile_factor
        new_kernel = jnp.concatenate(new_blocks, axis=-2)
        label = (
            f"Dense_0 ({source_num_blocks}-block source -> "
            f"{num_blocks}-block target via tile x{tile_factor}, no grid upsample)"
        )
        print(
            f"Grid-aware warm-start: tiling {label} — per-block obs_dim "
            f"unchanged at {per_block_new}."
        )
        out = dict(flat)
        out[kernel_key] = np.asarray(new_kernel)
        return out

    # Slow path: per-block obs_dim changed because the grid size changed under
    # a grid-sized seen field (local_obs=False). Upsample grid rows per block.
    if config.env.local_obs:
        raise ValueError(
            "local_obs=True checkpoint has per_block_old != per_block_new "
            f"({per_block_old} != {per_block_new}). This can only happen if "
            "view_radius or the obs-layout tail changed, which is not a "
            "supported warm-start transition."
        )
    old_grid_dim = per_block_old - scan_dim - _OBS_TAIL_DIM
    new_grid_dim = per_block_new - scan_dim - _OBS_TAIL_DIM
    if old_grid_dim <= 0 or new_grid_dim <= 0:
        raise ValueError(
            f"Warm-start per-block obs_dim delta not explained by grid size "
            f"change (old_per_block={per_block_old}, new_per_block={per_block_new}, "
            f"scan={scan_dim}, tail={_OBS_TAIL_DIM}). Did view_radius or the "
            f"obs layout change?"
        )

    old_side = int(math.isqrt(old_grid_dim))
    if old_side * old_side != old_grid_dim:
        raise ValueError(
            f"Checkpoint grid rows ({old_grid_dim}) is not a perfect square; "
            f"cannot infer old grid shape. Non-square source grids are not "
            f"supported."
        )
    new_H, new_W = config.env.grid_height, config.env.grid_width
    if new_H * new_W != new_grid_dim:
        raise ValueError(
            f"Config accounting failed: {new_H}*{new_W} != {new_grid_dim}. "
            f"Check view_radius matches between the checkpoint and this config."
        )

    if num_blocks == 1:
        label = "Dense_0"
    elif tile_factor == 1:
        label = f"Dense_0 (central-critic, {num_blocks}-block)"
    else:
        label = (
            f"Dense_0 (central-critic, {source_num_blocks}-block source -> "
            f"{num_blocks}-block target via tile x{tile_factor})"
        )
    print(
        f"Grid-aware warm-start: upsampling {label} grid rows "
        f"({old_side}x{old_side} -> {new_H}x{new_W}, nearest-neighbor)."
    )

    k = jnp.asarray(kernel)
    hidden = int(k.shape[-1])
    leading = k.shape[:-2]

    upsampled_source_blocks = []
    for b in range(source_num_blocks):
        start = b * per_block_old
        block = k[..., start:start + per_block_old, :]
        scan_rows = block[..., :scan_dim, :]
        grid_rows = block[..., scan_dim:scan_dim + old_grid_dim, :]
        tail_rows = block[..., scan_dim + old_grid_dim:, :]

        grid_2d = grid_rows.reshape(*leading, old_side, old_side, hidden)
        new_2d = jimage.resize(grid_2d, (*leading, new_H, new_W, hidden), method="nearest")
        grid_new = new_2d.reshape(*leading, new_grid_dim, hidden)

        upsampled_source_blocks.append(
            jnp.concatenate([scan_rows, grid_new, tail_rows], axis=-2)
        )

    # Tile blocks: target_blocks = source_blocks * tile_factor (round-robin tile,
    # not block-repeat — preserves the original ordering pattern across copies).
    new_blocks = upsampled_source_blocks * tile_factor
    new_kernel = jnp.concatenate(new_blocks, axis=-2)

    out = dict(flat)
    out[kernel_key] = np.asarray(new_kernel)
    return out


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

    # If the checkpoint was trained on a different grid size, spatially
    # upsample each network's Dense_0/kernel before unflattening so the actor
    # (and critic, when transferred) match the new obs_dim.
    actor_flat = _upsample_first_layer_for_grid(actor_flat, config)

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
        # Transfer the critic too. For a multi-agent CTDE critic the kernel
        # input is N concatenated copies of the per-agent obs. The loader
        # supports two cases:
        #   (a) source N == target N (most common): each agent's block is
        #       spatially upsampled independently and concatenated back.
        #   (b) target N is an integer multiple of source N (e.g., 4 -> 8):
        #       each source block is upsampled, then the full set of upsampled
        #       blocks is tiled tile_factor times to reach the target width.
        #       Principled because the central critic is symmetric in agent
        #       identity; agents are exchangeable. Set
        #       warm_start_source_num_agents in the YAML when this applies.
        # Critic transfer also requires hidden_dim and num_layers to match.
        num_blocks = config.env.num_agents
        source_num_blocks = (
            config.warm_start_source_num_agents or num_blocks
        )
        kernel_key = "params/Dense_0/kernel"
        source_hidden = (
            int(critic_flat[kernel_key].shape[-1])
            if kernel_key in critic_flat
            else None
        )
        target_hidden = config.network.critic_hidden_dim
        source_input = (
            int(critic_flat[kernel_key].shape[-2])
            if kernel_key in critic_flat
            else None
        )
        input_divisible = (
            source_input is not None and source_input % source_num_blocks == 0
        )
        n_tiles_ok = num_blocks % source_num_blocks == 0

        if source_hidden is not None and source_hidden != target_hidden:
            print(
                f"Skipping critic warm-start: source critic hidden_dim "
                f"({source_hidden}) != target ({target_hidden}). Set "
                f"critic_hidden_dim={source_hidden} in the target config to "
                f"enable critic transfer."
            )
        elif not n_tiles_ok:
            print(
                f"Skipping critic warm-start: target num_agents "
                f"({num_blocks}) is not an integer multiple of "
                f"warm_start_source_num_agents ({source_num_blocks}); per-agent "
                f"block tiling would be ambiguous."
            )
        elif not input_divisible:
            print(
                f"Skipping critic warm-start: source critic input "
                f"({source_input}) is not divisible by source_num_agents="
                f"{source_num_blocks}. Set warm_start_source_num_agents in "
                f"the target YAML if it differs from env.num_agents."
            )
        else:
            critic_flat = _upsample_first_layer_for_grid(
                critic_flat, config,
                num_blocks=num_blocks,
                source_num_blocks=source_num_blocks,
            )
            critic = Critic(
                hidden_dim=config.network.critic_hidden_dim,
                num_layers=config.network.critic_num_layers,
            )
            dummy_joint_obs = jnp.zeros(config.obs_dim * num_blocks)
            ref_critic = critic.init(key, dummy_joint_obs)
            critic_params = unflatten_params(critic_flat, ref_critic)
            print(
                f"Warm-started critic: input {source_input} -> "
                f"{config.obs_dim * num_blocks}, hidden {target_hidden}, "
                f"source_N={source_num_blocks} target_N={num_blocks}."
            )

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
        train_fn = make_train_multi_seed(
            config, init_actor, init_critic, report_progress=True,
        )
    else:
        train_fn = make_train(
            config, init_actor, init_critic, report_progress=True,
        )

    key = jax.random.PRNGKey(0)
    desc = f"{config.experiment_name} ({num_seeds} seed{'s' if num_seeds > 1 else ''})"
    progress_bar.start(total=config.train.num_episodes, desc=desc)
    try:
        actor_params, critic_params, metrics = train_fn(key)
        jax.block_until_ready(metrics["loss"])
    finally:
        progress_bar.finish()

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

    # Save checkpoint. For the POSG joint-red path the trainer carries the
    # central red policy in the slot normally reserved for critic params;
    # persist it to its own file instead of mixing into the actor archive.
    ckpt_path = out_dir / "checkpoint.npz"
    if config.train.red_policy == "joint" and critic_params is not None:
        save_checkpoint(actor_params, str(ckpt_path), critic_params=None)
        red_ckpt_path = out_dir / "joint_red_checkpoint.npz"
        flat = flatten_params(critic_params)
        np.savez(str(red_ckpt_path), **flat)
    else:
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


def _print_banner(config: ExperimentConfig, config_path: str) -> None:
    """Pretty-print the experiment header before training kicks off."""
    lines = [
        f"Experiment : {config.experiment_name}",
        f"Config     : {config_path}",
        f"Method     : {config.train.method}",
        f"Env        : {config.env.grid_width}x{config.env.grid_height}, "
        f"N={config.env.num_agents}, steps={config.env.max_steps}",
        f"Train      : {config.train.num_episodes} eps x "
        f"{config.train.num_seeds} seed(s), lr={config.train.lr}, "
        f"gamma={config.train.gamma}",
    ]
    if config.warm_start is not None:
        lines.append(f"Warm-start : {config.warm_start}")
    width = max(len(s) for s in lines) + 2
    bar = "=" * width
    print()
    print(bar)
    for s in lines:
        print(" " + s)
    print(bar)
    print()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    config = load_config_with_overrides(args)

    _print_banner(config, args.config)

    t0 = time.time()
    actor_params, critic_params, metrics = run_training(config)
    # Safety net: block one more time in case run_training's block slipped
    # past an async dispatch. Cheap — arrays are already materialised.
    jax.block_until_ready(metrics["loss"])
    elapsed = time.time() - t0

    out_dir = save_results(config, actor_params, critic_params, metrics)
    print_summary(config, metrics, out_dir, elapsed)


if __name__ == "__main__":
    main()
