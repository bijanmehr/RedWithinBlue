"""Pure-JAX grid utilities for the RedWithinBlue multi-agent RL environment.

All functions are JIT-compatible: no Python control flow, static shapes.
"""

import jax
import jax.numpy as jnp
from red_within_blue.types import (
    GridState,
    CELL_EMPTY,
    CELL_WALL,
    CELL_OBSTACLE,
    CELL_OCCUPIED,
)


def create_grid(
    width: int,
    height: int,
    wall_density: float,
    key: jax.Array,
) -> GridState:
    """Create a random grid with walls on the boundary and interior.

    Args:
        width:  number of columns (W).
        height: number of rows (H).
        wall_density: probability that an *interior* cell becomes a wall.
        key: JAX PRNG key.

    Returns:
        A ``GridState`` with terrain, occupancy, and explored arrays.
    """
    # Start with all-empty terrain.
    terrain = jnp.full((height, width), CELL_EMPTY, dtype=jnp.int32)

    # ----- boundary walls -----
    # Top and bottom rows.
    terrain = terrain.at[0, :].set(CELL_WALL)
    terrain = terrain.at[height - 1, :].set(CELL_WALL)
    # Left and right columns.
    terrain = terrain.at[:, 0].set(CELL_WALL)
    terrain = terrain.at[:, width - 1].set(CELL_WALL)

    # ----- interior random walls -----
    # Generate a Bernoulli mask over the full grid, then zero-out the boundary.
    interior_mask = jax.random.bernoulli(key, p=wall_density, shape=(height, width))

    # Mask out boundaries so they are not double-counted.
    boundary_mask = jnp.ones((height, width), dtype=jnp.bool_)
    boundary_mask = boundary_mask.at[0, :].set(False)
    boundary_mask = boundary_mask.at[height - 1, :].set(False)
    boundary_mask = boundary_mask.at[:, 0].set(False)
    boundary_mask = boundary_mask.at[:, width - 1].set(False)

    interior_walls = interior_mask & boundary_mask  # True → wall
    terrain = jnp.where(interior_walls, CELL_WALL, terrain)

    # Occupancy and explored start at zero.
    occupancy = jnp.zeros((height, width), dtype=jnp.int32)
    explored = jnp.zeros((height, width), dtype=jnp.int32)

    return GridState(terrain=terrain, occupancy=occupancy, explored=explored)


def get_local_scan(
    terrain: jax.Array,
    occupancy: jax.Array,
    position: jax.Array,
    obs_radius: int,
) -> jax.Array:
    """Return a local observation patch for a single agent.

    The returned array has shape ``(obs_d, obs_d)`` where
    ``obs_d = 2 * obs_radius + 1``.

    Out-of-bounds cells are filled with ``CELL_WALL``.
    If an occupancy value is > 0, the cell shows ``CELL_OCCUPIED`` instead
    of the underlying terrain.

    This function is designed to be vmapped over agents.

    Args:
        terrain:   [H, W] int32 static terrain.
        occupancy: [H, W] int32 dynamic occupancy map.
        position:  [2] int32  (row, col) of the agent.
        obs_radius: integer observation radius (static).

    Returns:
        [obs_d, obs_d] int32 local observation.
    """
    obs_d = 2 * obs_radius + 1
    h, w = terrain.shape

    # Combined view: occupancy overrides terrain when > 0.
    combined = jnp.where(occupancy > 0, CELL_OCCUPIED, terrain)

    # Pad the combined grid with CELL_WALL on every side by obs_radius.
    padded = jnp.pad(
        combined,
        pad_width=obs_radius,
        mode="constant",
        constant_values=CELL_WALL,
    )

    # After padding, the original cell (r, c) lives at (r + obs_radius, c + obs_radius).
    # The top-left corner of the patch is (r + obs_radius - obs_radius, c + obs_radius - obs_radius) = (r, c).
    row = position[0]
    col = position[1]

    # dynamic_slice requires index and size; size must be a static int.
    patch = jax.lax.dynamic_slice(padded, (row, col), (obs_d, obs_d))
    return patch


def update_occupancy(
    positions: jax.Array,
    uids: jax.Array,
    grid_shape: tuple[int, int],
) -> jax.Array:
    """Build a fresh occupancy map from agent positions and UIDs.

    Args:
        positions: [N, 2] int32 — (row, col) per agent.
        uids:      [N] int32  — unique identifier per agent (> 0).
        grid_shape: (H, W) static shape of the grid.

    Returns:
        [H, W] int32 array with uid at each agent's position, 0 elsewhere.
    """
    occ = jnp.zeros(grid_shape, dtype=jnp.int32)
    rows = positions[:, 0]
    cols = positions[:, 1]
    occ = occ.at[rows, cols].set(uids)
    return occ


def update_exploration(
    explored: jax.Array,
    positions: jax.Array,
) -> jax.Array:
    """Increment exploration counts at each agent's current position.

    Args:
        explored:  [H, W] int32 cumulative visit-count map.
        positions: [N, 2] int32 — (row, col) per agent.

    Returns:
        Updated [H, W] int32 explored array.
    """
    rows = positions[:, 0]
    cols = positions[:, 1]
    explored = explored.at[rows, cols].add(1)
    return explored


def apply_red_contamination(
    explored: jax.Array,
    positions: jax.Array,
    num_red_agents: int,
) -> jax.Array:
    """Reset the ground-truth visit counter at red-occupied cells.

    Models reds as adversaries that fog cells they pass through: after the
    standard ``+1`` exploration update, each cell currently occupied by a
    red is set back to ``explored == 0``. The reward functions read
    ``prev_explored`` to judge "was this cell previously known?" — zeroing
    red cells re-opens them to blue discovery bonuses.

    Per-agent ``local_map`` fog is handled separately by the team-asymmetric
    message propagation in :func:`agents.update_local_maps_with_comm`, which
    overwrites a *blue* receiver's local_map cells with ``MAP_UNKNOWN`` when
    the sender is red — so only comm-connected blues get their belief erased.

    No-op when ``num_red_agents == 0`` (the scatter is skipped entirely,
    leaving the input bit-identical to the standard pipeline).

    Args:
        explored:        [H, W] int32 visit-count map (post +1 update).
        positions:       [N, 2] int32 post-move agent positions.
        num_red_agents:  Python int — reds occupy the last ``num_red_agents``
                         indices of ``positions``.

    Returns:
        ``explored`` with red cells zeroed.
    """
    if num_red_agents == 0:
        return explored
    red_positions = positions[-num_red_agents:]   # [n_red, 2]
    red_rows = red_positions[:, 0]
    red_cols = red_positions[:, 1]
    return explored.at[red_rows, red_cols].set(jnp.int32(0))
