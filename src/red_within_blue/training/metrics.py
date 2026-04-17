"""Metrics computation for training monitoring and evaluation."""

import jax
import jax.numpy as jnp

from red_within_blue.types import CELL_WALL


def compute_coverage(terrain: jnp.ndarray, explored: jnp.ndarray) -> jnp.ndarray:
    """Compute fraction of non-wall cells that have been explored.

    Args:
        terrain:  [H, W] int32 grid of cell types; walls have value CELL_WALL.
        explored: [H, W] int32 grid; non-zero means the cell was visited.

    Returns:
        Scalar float — explored_non_wall / total_non_wall.
        Returns 0.0 when there are no non-wall cells.
    """
    non_wall = terrain != CELL_WALL          # [H, W] bool
    total_non_wall = jnp.sum(non_wall)

    explored_non_wall = jnp.sum((explored != 0) & non_wall)

    coverage = jnp.where(
        total_non_wall > 0,
        explored_non_wall / total_non_wall,
        0.0,
    )
    return coverage.astype(jnp.float32)


def compute_action_distribution(
    actions: jnp.ndarray,
    num_actions: int = 5,
) -> jnp.ndarray:
    """Compute a probability distribution over actions.

    Args:
        actions:     [T] int array of action indices.
        num_actions: total number of distinct actions.

    Returns:
        [num_actions] float32 array summing to 1.0.
        Returns a uniform distribution when T == 0.
    """
    counts = jnp.zeros(num_actions, dtype=jnp.float32)
    counts = counts.at[actions].add(1.0)

    total = jnp.sum(counts)
    dist = jnp.where(
        total > 0,
        counts / total,
        jnp.ones(num_actions, dtype=jnp.float32) / num_actions,
    )
    return dist.astype(jnp.float32)


def compute_explained_variance(
    returns: jnp.ndarray,
    predictions: jnp.ndarray,
) -> jnp.ndarray:
    """Compute explained variance of value predictions.

    EV = 1 - Var(returns - predictions) / Var(returns)

    Args:
        returns:     [T] float array of empirical returns.
        predictions: [T] float array of value estimates.

    Returns:
        Scalar float.  Returns 0.0 when Var(returns) == 0.
    """
    residuals = returns - predictions
    var_residuals = jnp.var(residuals)
    var_returns = jnp.var(returns)

    ev = jnp.where(
        var_returns > 0.0,
        1.0 - var_residuals / var_returns,
        0.0,
    )
    return ev.astype(jnp.float32)


def compute_steps_to_coverage(
    per_step_coverage: jnp.ndarray,
    threshold: float,
) -> jnp.ndarray:
    """Return the first timestep at which coverage reaches threshold.

    Args:
        per_step_coverage: [T] float array of coverage values.
        threshold:         target coverage level (scalar float).

    Returns:
        Scalar int32 — first index where per_step_coverage >= threshold,
        or T if the threshold is never reached.
    """
    T = per_step_coverage.shape[0]
    reached = per_step_coverage >= threshold          # [T] bool

    # argmax returns the first True index; if none are True it returns 0.
    first_idx = jnp.argmax(reached)

    # If no step reached the threshold, return T.
    steps = jnp.where(reached[first_idx], first_idx, T)
    return steps.astype(jnp.int32)


def compute_connectivity_fraction(connected_timeline: jnp.ndarray) -> jnp.ndarray:
    """Compute the fraction of timesteps where the swarm was connected.

    Args:
        connected_timeline: [T] bool array.

    Returns:
        Scalar float.  Returns 0.0 for empty input.
    """
    T = connected_timeline.shape[0]
    fraction = jnp.where(
        T > 0,
        jnp.sum(connected_timeline) / T,
        0.0,
    )
    return fraction.astype(jnp.float32)
