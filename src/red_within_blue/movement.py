"""Pure-JAX movement resolution for multi-agent grid environments.

All operations use jnp (no Python loops) so the function is fully JIT-compatible.
"""

from typing import Optional, Tuple

import jax.numpy as jnp

from red_within_blue.types import ACTION_DELTAS_ARRAY, CELL_EMPTY


def resolve_actions(
    positions: jnp.ndarray,
    actions: jnp.ndarray,
    terrain: jnp.ndarray,
    grid_shape: Tuple[int, int],
    passable_types: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Resolve agent movement actions on a grid, handling terrain and collisions.

    Parameters
    ----------
    positions : jnp.ndarray, shape [N, 2], dtype int32
        Current (row, col) positions of each agent.
    actions : jnp.ndarray, shape [N], dtype int32
        Action index per agent (0-4, mapping to ``Action`` enum).
    terrain : jnp.ndarray, shape [H, W], dtype int32
        Static terrain map.  Each cell holds a cell-type constant.
    grid_shape : tuple (H, W)
        Height and width of the grid.
    passable_types : jnp.ndarray, shape [num_types], dtype bool, optional
        Boolean mask over cell type values.  ``passable_types[c]`` is ``True``
        iff agents may walk on cells of type ``c``.  By default only
        ``CELL_EMPTY`` (0) is passable.

    Returns
    -------
    new_positions : jnp.ndarray, shape [N, 2], dtype int32
        Updated positions after resolution.
    collision_mask : jnp.ndarray, shape [N], dtype bool
        ``True`` for every agent that was involved in an agent-agent collision
        (two or more agents targeting the same cell).
    """
    H, W = grid_shape
    N = positions.shape[0]

    # Default passable_types: only CELL_EMPTY is passable.
    if passable_types is None:
        # Create a boolean array of length max(cell type constants) + 1.
        # Only index 0 (CELL_EMPTY) is True.
        passable_types = jnp.array([True, False, False], dtype=jnp.bool_)

    # ------------------------------------------------------------------
    # Step 1: Look up deltas from ACTION_DELTAS_ARRAY
    # ------------------------------------------------------------------
    deltas = ACTION_DELTAS_ARRAY[actions]  # [N, 2]

    # ------------------------------------------------------------------
    # Step 2: Compute intended positions
    # ------------------------------------------------------------------
    intended = positions + deltas  # [N, 2]

    # ------------------------------------------------------------------
    # Step 3: Clamp to grid bounds
    # ------------------------------------------------------------------
    intended_row = jnp.clip(intended[:, 0], 0, H - 1)
    intended_col = jnp.clip(intended[:, 1], 0, W - 1)
    intended_clamped = jnp.stack([intended_row, intended_col], axis=-1)  # [N, 2]

    # ------------------------------------------------------------------
    # Step 4: Terrain passability check
    # ------------------------------------------------------------------
    cell_types = terrain[intended_clamped[:, 0], intended_clamped[:, 1]]  # [N]
    # Look up whether each cell type is passable.  Clip to handle any
    # cell-type values that exceed the length of passable_types.
    safe_cell_types = jnp.clip(cell_types, 0, passable_types.shape[0] - 1)
    terrain_ok = passable_types[safe_cell_types]  # [N] bool

    # If terrain is impassable, revert to current position.
    after_terrain = jnp.where(
        terrain_ok[:, None], intended_clamped, positions
    )  # [N, 2]

    # ------------------------------------------------------------------
    # Step 5: Agent-agent collision detection
    # ------------------------------------------------------------------
    # Encode each target cell as a single integer for easy comparison.
    target_flat = after_terrain[:, 0] * W + after_terrain[:, 1]  # [N]

    # Count how many agents target each cell.
    # Build a flat count array of size H*W, scatter-add ones.
    counts = jnp.zeros(H * W, dtype=jnp.int32)
    counts = counts.at[target_flat].add(jnp.ones(N, dtype=jnp.int32))

    # An agent collides if the cell it targets is targeted by 2+ agents.
    target_counts = counts[target_flat]  # [N]
    collision_mask = target_counts > 1  # [N] bool

    # Collided agents revert to their current position.
    new_positions = jnp.where(
        collision_mask[:, None], positions, after_terrain
    )  # [N, 2]

    return new_positions, collision_mask
