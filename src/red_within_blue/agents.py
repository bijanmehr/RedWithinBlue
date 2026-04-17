"""Agent initialization, local-map update, and message preparation.

All functions are pure-JAX and JIT-compatible.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp

from red_within_blue.types import (
    AgentState,
    EnvConfig,
    CELL_EMPTY,
    CELL_WALL,
    CELL_OBSTACLE,
    MAP_UNKNOWN,
    MAP_FREE,
    MAP_WALL,
    MAP_OBSTACLE,
)

# ── helpers ──────────────────────────────────────────────────────────

def _cell_to_map(cell_val: chex.Array) -> chex.Array:
    """Convert a terrain cell value to the corresponding local-map value.

    CELL_EMPTY (0)    -> MAP_FREE (1)
    CELL_WALL  (1)    -> MAP_WALL (2)
    CELL_OBSTACLE (2) -> MAP_OBSTACLE (3)
    anything else     -> MAP_FREE (1)
    """
    return jnp.where(
        cell_val == CELL_WALL,
        MAP_WALL,
        jnp.where(
            cell_val == CELL_OBSTACLE,
            MAP_OBSTACLE,
            MAP_FREE,  # CELL_EMPTY and any other value
        ),
    )


# ── 1. init_agents ──────────────────────────────────────────────────

def init_agents(
    config: EnvConfig,
    terrain: chex.Array,
    key: chex.PRNGKey,
) -> AgentState:
    """Spawn *num_agents* at random empty cells and return the initial AgentState.

    Parameters
    ----------
    config : EnvConfig
    terrain : Array [H, W] int32 — the static terrain grid
    key : PRNGKey
    """
    H, W = terrain.shape
    N = config.num_agents
    obs_d = 2 * config.obs_radius + 1
    scan_dim = obs_d * obs_d
    total_msg_dim = scan_dim + config.msg_dim

    # --- random valid positions -------------------------------------------
    flat = terrain.reshape(-1)                          # [H*W]
    mask = (flat == CELL_EMPTY).astype(jnp.float32)     # 1 where empty
    # Normalise to probability distribution (jax.random.choice needs `p`)
    probs = mask / mask.sum()

    chosen_flat = jax.random.choice(
        key, a=H * W, shape=(N,), replace=False, p=probs,
    )
    rows = chosen_flat // W
    cols = chosen_flat % W
    positions = jnp.stack([rows, cols], axis=-1).astype(jnp.int32)  # [N, 2]

    # --- other fields -----------------------------------------------------
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    local_scan = jnp.zeros((N, obs_d, obs_d), dtype=jnp.int32)
    messages_out = jnp.zeros((N, total_msg_dim), dtype=jnp.float32)
    messages_in = jnp.zeros((N, total_msg_dim), dtype=jnp.float32)

    uids = jnp.arange(1, N + 1, dtype=jnp.int32)
    team_ids = jnp.zeros(N, dtype=jnp.int32)
    comm_ranges = jnp.full(N, config.comm_radius, dtype=jnp.float32)

    return AgentState(
        positions=positions,
        comm_ranges=comm_ranges,
        team_ids=team_ids,
        uids=uids,
        messages_out=messages_out,
        messages_in=messages_in,
        local_map=local_map,
        local_scan=local_scan,
    )


# ── 2. update_local_maps ────────────────────────────────────────────

def update_local_maps(
    local_map: chex.Array,
    local_scan: chex.Array,
    positions: chex.Array,
    obs_radius: int,
) -> chex.Array:
    """Write each agent's *local_scan* into its *local_map* at the correct position.

    Parameters
    ----------
    local_map  : Array [N, H, W] int32
    local_scan : Array [N, obs_d, obs_d] int32  (terrain cell values)
    positions  : Array [N, 2] int32  (row, col)
    obs_radius : int

    Returns
    -------
    Array [N, H, W] — updated local maps
    """
    N, H, W = local_map.shape
    obs_d = 2 * obs_radius + 1

    def _update_one(lmap, scan, pos):
        """Update a single agent's local map."""
        r, c = pos[0], pos[1]
        # Top-left corner of the observation window in map coordinates
        top = r - obs_radius
        left = c - obs_radius

        # For each cell (i, j) in the obs_d x obs_d scan:
        #   map_row = top + i,  map_col = left + j
        # We need to handle boundary clipping.

        # Create index grids for the scan patch
        scan_rows = jnp.arange(obs_d)  # 0 .. obs_d-1
        scan_cols = jnp.arange(obs_d)
        si, sj = jnp.meshgrid(scan_rows, scan_cols, indexing="ij")  # [obs_d, obs_d]

        # Corresponding map coordinates
        mi = top + si  # [obs_d, obs_d]
        mj = left + sj

        # Validity mask: inside [0, H) x [0, W)
        valid = (mi >= 0) & (mi < H) & (mj >= 0) & (mj < W)

        # Convert cell values to map values
        map_vals = _cell_to_map(scan)  # [obs_d, obs_d]

        # Clamp indices for safe indexing (the invalid ones will be masked out)
        mi_safe = jnp.clip(mi, 0, H - 1)
        mj_safe = jnp.clip(mj, 0, W - 1)

        # Flatten for scatter
        flat_idx = mi_safe.reshape(-1) * W + mj_safe.reshape(-1)  # [obs_d**2]
        flat_vals = map_vals.reshape(-1)
        flat_valid = valid.reshape(-1)

        # Current map flattened
        lmap_flat = lmap.reshape(-1)  # [H*W]

        # Only write where valid
        new_vals = jnp.where(flat_valid, flat_vals, lmap_flat[flat_idx])
        lmap_flat = lmap_flat.at[flat_idx].set(new_vals)

        return lmap_flat.reshape(H, W)

    return jax.vmap(_update_one)(local_map, local_scan, positions)


# ── 3. prepare_messages ─────────────────────────────────────────────

def prepare_messages(
    local_scan: chex.Array,
    msg_dim: int,
    learned_vectors: Optional[chex.Array] = None,
) -> chex.Array:
    """Build outgoing message vectors for all agents.

    Parameters
    ----------
    local_scan : Array [N, obs_d, obs_d] int32
    msg_dim : int — size of the learned message component
    learned_vectors : Array [N, msg_dim] or None

    Returns
    -------
    Array [N, scan_dim + msg_dim] float32
    """
    N = local_scan.shape[0]
    flat_scan = local_scan.reshape(N, -1).astype(jnp.float32)  # [N, scan_dim]

    if learned_vectors is None:
        learned_part = jnp.zeros((N, msg_dim), dtype=jnp.float32)
    else:
        learned_part = learned_vectors.astype(jnp.float32)

    return jnp.concatenate([flat_scan, learned_part], axis=-1)
