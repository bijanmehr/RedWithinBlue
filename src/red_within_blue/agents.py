"""Agent initialization and local-map update helpers (pure-JAX, JIT-safe).

Design note
-----------
Agents do **not** exchange learned messages or scan embeddings.  When two
agents are within `comm_radius`, the receiver merges the sender's raw
*survey patch* (the ``(2·survey_radius+1)²`` block of terrain the sender just
committed) into its own ``local_map``.  All inter-agent information sharing
therefore happens at the knowledge-base level, not in the observation vector.
See :func:`update_local_maps_with_comm`.

Vocabulary
----------
* **view** — what the sensor returns. ``local_scan`` has shape
  ``(2·view_radius+1, 2·view_radius+1)`` and is what the policy's observation
  uses. Wider view = more situational awareness, no commitment.
* **survey** — the per-cell mission the agent executes each step. The central
  ``(2·survey_radius+1, 2·survey_radius+1)`` sub-patch of the view is written
  into the agent's ``local_map`` and is also what is broadcast over comms.
  ``survey_radius=0`` is the simplest form: "I commit only the cell I am on".

When ``survey_radius == view_radius`` the two coincide and behavior matches the
legacy single-radius code path.
"""

from __future__ import annotations

from functools import partial

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
    """Spawn *num_agents* at random empty cells and return the initial AgentState."""
    from red_within_blue.types import resolve_view_radius
    H, W = terrain.shape
    N = config.num_agents
    view_r = resolve_view_radius(config)
    obs_d = 2 * view_r + 1

    flat = terrain.reshape(-1)
    mask = (flat == CELL_EMPTY).astype(jnp.float32)

    if getattr(config, "center_spawn", False):
        # Gaussian-weighted sampling concentrated around the grid centre.
        # sigma set to roughly half the comm_radius so spawned agents are
        # almost always within communication range of each other at t=0.
        rr, cc = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        center_r = (H - 1) / 2.0
        center_c = (W - 1) / 2.0
        dist_sq = (
            (rr.astype(jnp.float32) - center_r) ** 2
            + (cc.astype(jnp.float32) - center_c) ** 2
        ).reshape(-1)
        sigma = jnp.maximum(config.comm_radius * 0.5, 1.0).astype(jnp.float32)
        weights = jnp.exp(-dist_sq / (2.0 * sigma * sigma))
        probs = mask * weights
        probs = probs / jnp.maximum(probs.sum(), 1e-8)
    else:
        probs = mask / mask.sum()

    chosen_flat = jax.random.choice(
        key, a=H * W, shape=(N,), replace=False, p=probs,
    )
    rows = chosen_flat // W
    cols = chosen_flat % W
    positions = jnp.stack([rows, cols], axis=-1).astype(jnp.int32)

    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    local_scan = jnp.zeros((N, obs_d, obs_d), dtype=jnp.int32)

    uids = jnp.arange(1, N + 1, dtype=jnp.int32)
    n_red = config.num_red_agents
    team_ids = (jnp.arange(N, dtype=jnp.int32) >= (N - n_red)).astype(jnp.int32)
    comm_ranges = jnp.full(N, config.comm_radius, dtype=jnp.float32)

    return AgentState(
        positions=positions,
        comm_ranges=comm_ranges,
        team_ids=team_ids,
        uids=uids,
        local_map=local_map,
        local_scan=local_scan,
        disconnect_timer=jnp.zeros((config.num_agents,), dtype=jnp.int32),
    )


# ── 2. survey-to-map scatter primitive ──────────────────────────────

def _crop_survey_patch(
    local_scan: chex.Array,
    view_radius: int,
    survey_radius: int,
) -> chex.Array:
    """Return the central ``(2·survey+1)²`` block of a view-sized scan.

    ``local_scan`` has shape ``(2·view_radius+1, 2·view_radius+1)``. The result
    has shape ``(2·survey_radius+1, 2·survey_radius+1)``. When the two radii
    are equal this is a no-op.
    """
    if survey_radius > view_radius:
        raise ValueError(
            f"survey_radius ({survey_radius}) must be <= view_radius "
            f"({view_radius}); the sensor cannot commit more cells than it "
            f"observed."
        )
    if survey_radius == view_radius:
        return local_scan
    survey_d = 2 * survey_radius + 1
    start = view_radius - survey_radius  # inclusive start along each axis
    return jax.lax.dynamic_slice(local_scan, (start, start), (survey_d, survey_d))


def _survey_write_indices(
    survey_patch: chex.Array,
    pos: chex.Array,
    survey_radius: int,
    H: int,
    W: int,
):
    """Compute (flat_indices, flat_values, flat_valid) for one sender's survey patch.

    ``survey_patch`` already has shape ``(2·survey_radius+1, 2·survey_radius+1)``
    — callers crop it from the view-sized ``local_scan`` via
    :func:`_crop_survey_patch` before invoking this function.
    """
    survey_d = 2 * survey_radius + 1
    r, c = pos[0], pos[1]
    top, left = r - survey_radius, c - survey_radius
    si, sj = jnp.meshgrid(jnp.arange(survey_d), jnp.arange(survey_d), indexing="ij")
    mi = top + si
    mj = left + sj
    valid = (mi >= 0) & (mi < H) & (mj >= 0) & (mj < W)
    mi_safe = jnp.clip(mi, 0, H - 1)
    mj_safe = jnp.clip(mj, 0, W - 1)
    flat_idx = (mi_safe * W + mj_safe).reshape(-1)
    flat_vals = _cell_to_map(survey_patch).reshape(-1)
    flat_valid = valid.reshape(-1)
    return flat_idx, flat_vals, flat_valid


def update_local_maps(
    local_map: chex.Array,
    local_scan: chex.Array,
    positions: chex.Array,
    view_radius: int,
    survey_radius: int | None = None,
) -> chex.Array:
    """Write each agent's own survey patch into its *local_map*.

    ``local_scan`` is the view-sized sensor frame. The central
    ``(2·survey_radius+1)²`` block is cropped and written; the rest of the
    view is *not* committed to the map (it is only used by the policy).

    No neighbor sharing — see :func:`update_local_maps_with_comm` for that.

    Parameters
    ----------
    local_map : [N, H, W] int32
    local_scan : [N, 2·view_radius+1, 2·view_radius+1] int32
    positions : [N, 2] int32
    view_radius : int
    survey_radius : int, optional
        Defaults to ``view_radius`` (legacy single-radius behaviour).
    """
    if survey_radius is None:
        survey_radius = view_radius
    N, H, W = local_map.shape

    def _update_one(lmap, scan, pos):
        patch = _crop_survey_patch(scan, view_radius, survey_radius)
        flat_idx, flat_vals, flat_valid = _survey_write_indices(patch, pos, survey_radius, H, W)
        lmap_flat = lmap.reshape(-1)
        new_vals = jnp.where(flat_valid, flat_vals, lmap_flat[flat_idx])
        lmap_flat = lmap_flat.at[flat_idx].set(new_vals)
        return lmap_flat.reshape(H, W)

    return jax.vmap(_update_one)(local_map, local_scan, positions)


# ── 3. update_local_maps_with_comm ──────────────────────────────────

def update_local_maps_with_comm(
    local_map: chex.Array,
    local_scan: chex.Array,
    positions: chex.Array,
    adjacency: chex.Array,
    view_radius: int,
    survey_radius: int | None = None,
    team_ids: chex.Array | None = None,
) -> chex.Array:
    """Merge every agent's own survey AND its neighbors' surveys into its local_map.

    Receiver *i* accepts the survey patch of sender *j* (pasted at the
    sender's position) whenever ``adjacency[j, i]`` is True or when ``j == i``.
    Survey patches are terrain-truthful so overlapping writes are consistent
    by construction.

    This is the complete Level-A messaging mechanism: the "message" is the
    survey footprint itself, conveyed by scattering the sender's surveyed
    cell values into the receiver's local_map. There are no learned message
    embeddings; the observation vector never contains per-agent messages.

    When ``team_ids`` is provided, the message *content* is team-asymmetric:
    a **red sender** (``team_id == 1``) writing into a **blue receiver**
    (``team_id == 0``) overwrites the receiver's cells with ``MAP_UNKNOWN``
    instead of the terrain truth — the red fogs the blue's belief. Red →
    red and blue → * stay truthful, so reds still share truth among
    themselves. When ``team_ids`` is None, the function behaves exactly as
    the all-blue baseline (no fogging).

    Parameters
    ----------
    local_map : [N, H, W] int32
    local_scan : [N, 2·view_radius+1, 2·view_radius+1] int32
        View-sized sensor frames. The central ``(2·survey_radius+1)²`` block
        of each is what actually gets scattered into the map.
    positions : [N, 2] int32
    adjacency : [N, N] bool — adjacency[j, i] True means j can send to i
    view_radius : int
    survey_radius : int, optional
        Defaults to ``view_radius``. When smaller, each message carries
        fewer cells — ``survey_radius=0`` means agents broadcast only the
        single cell they currently occupy.
    team_ids : [N] int32, optional
        0 = blue, 1 = red. When ``None``, treated as all-blue (no fogging).
    """
    if survey_radius is None:
        survey_radius = view_radius
    N, H, W = local_map.shape

    if team_ids is None:
        team_ids = jnp.zeros((N,), dtype=jnp.int32)

    # Crop each view-sized scan down to its survey patch (static shapes),
    # then compute one scatter buffer per sender.
    all_idx, all_vals, all_valid = jax.vmap(
        lambda s, p: _survey_write_indices(
            _crop_survey_patch(s, view_radius, survey_radius),
            p, survey_radius, H, W,
        )
    )(local_scan, positions)  # shapes: [N, (2·survey+1)**2]

    # contributes[j, i] = True iff sender j writes into receiver i.
    contributes = adjacency | jnp.eye(N, dtype=jnp.bool_)  # [N_sender, N_receiver]

    def apply_sender(carry_maps, j_idx):
        idx_j = all_idx[j_idx]
        vals_j = all_vals[j_idx]
        valid_j = all_valid[j_idx]
        does_contrib = contributes[j_idx]            # [N_receiver] bool
        sender_is_red = team_ids[j_idx] == 1          # scalar bool

        def apply_to_receiver(lmap_i, does_contrib_i, recv_team_i):
            flat_lmap = lmap_i.reshape(-1)
            # Asymmetric message: red → blue gets fogged, everything else
            # is truthful.
            fog = sender_is_red & (recv_team_i == 0)
            msg_vals = jnp.where(fog, jnp.int32(MAP_UNKNOWN), vals_j)
            write_mask = valid_j & does_contrib_i
            new_vals = jnp.where(write_mask, msg_vals, flat_lmap[idx_j])
            flat_lmap = flat_lmap.at[idx_j].set(new_vals)
            return flat_lmap.reshape(H, W)

        new_maps = jax.vmap(apply_to_receiver)(carry_maps, does_contrib, team_ids)
        return new_maps, None

    new_local_map, _ = jax.lax.scan(apply_sender, local_map, jnp.arange(N))
    return new_local_map
