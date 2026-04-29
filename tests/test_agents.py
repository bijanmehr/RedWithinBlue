"""Tests for red_within_blue.agents — init, local-map update, comm-merged map."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.types import (
    EnvConfig,
    CELL_EMPTY,
    CELL_WALL,
    CELL_OBSTACLE,
    MAP_UNKNOWN,
    MAP_FREE,
    MAP_WALL,
    MAP_OBSTACLE,
)
from red_within_blue.agents import (
    init_agents,
    update_local_maps,
    update_local_maps_with_comm,
)

# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def small_config():
    """8x8 grid, 2 agents, obs_radius=2."""
    return EnvConfig(
        grid_width=8,
        grid_height=8,
        num_agents=2,
        obs_radius=2,
        comm_radius=5.0,
        max_steps=64,
    )


@pytest.fixture
def terrain_8x8():
    """8x8 terrain: all empty except a wall ring on row 0 and column 0."""
    t = jnp.zeros((8, 8), dtype=jnp.int32)
    t = t.at[0, :].set(CELL_WALL)
    t = t.at[:, 0].set(CELL_WALL)
    t = t.at[3, 3].set(CELL_OBSTACLE)
    return t


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ── init_agents ─────────────────────────────────────────────────────

def test_init_agents_valid_positions(small_config, terrain_8x8, key):
    state = init_agents(small_config, terrain_8x8, key)

    N = small_config.num_agents
    H, W = small_config.grid_height, small_config.grid_width

    assert state.positions.shape == (N, 2)
    for i in range(N):
        r, c = int(state.positions[i, 0]), int(state.positions[i, 1])
        assert 0 <= r < H
        assert 0 <= c < W
        assert int(terrain_8x8[r, c]) == CELL_EMPTY


def test_init_agents_seeded(small_config, terrain_8x8):
    k = jax.random.PRNGKey(123)
    s1 = init_agents(small_config, terrain_8x8, k)
    s2 = init_agents(small_config, terrain_8x8, k)

    assert jnp.array_equal(s1.positions, s2.positions)
    assert jnp.array_equal(s1.uids, s2.uids)
    assert jnp.array_equal(s1.local_map, s2.local_map)


# ── update_local_maps (own-scan only) ──────────────────────────────

def test_update_local_map(small_config):
    H, W = small_config.grid_height, small_config.grid_width
    N = small_config.num_agents
    obs_r = small_config.obs_radius
    obs_d = 2 * obs_r + 1

    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[4, 4], [6, 6]], dtype=jnp.int32)

    scan0 = jnp.full((obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    scan0 = scan0.at[obs_r, obs_r].set(CELL_WALL)

    scan1 = jnp.full((obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    scan1 = scan1.at[0, 0].set(CELL_OBSTACLE)

    local_scan = jnp.stack([scan0, scan1], axis=0)

    updated = update_local_maps(local_map, local_scan, positions, obs_r)

    assert int(updated[0, 4, 4]) == MAP_WALL
    assert int(updated[0, 2, 2]) == MAP_FREE
    assert int(updated[0, 0, 0]) == MAP_UNKNOWN

    assert int(updated[1, 4, 4]) == MAP_OBSTACLE
    assert int(updated[1, 6, 6]) == MAP_FREE


def test_local_map_persistence(small_config):
    H, W = small_config.grid_height, small_config.grid_width
    obs_r = small_config.obs_radius
    obs_d = 2 * obs_r + 1

    N = 1
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)

    pos1 = jnp.array([[3, 3]], dtype=jnp.int32)
    scan1 = jnp.full((N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    local_map = update_local_maps(local_map, scan1, pos1, obs_r)
    assert int(local_map[0, 3, 3]) == MAP_FREE

    pos2 = jnp.array([[6, 6]], dtype=jnp.int32)
    scan2 = jnp.full((N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    local_map = update_local_maps(local_map, scan2, pos2, obs_r)

    assert int(local_map[0, 6, 6]) == MAP_FREE
    assert int(local_map[0, 3, 3]) == MAP_FREE
    assert int(local_map[0, 0, 0]) == MAP_UNKNOWN


# ── update_local_maps_with_comm (agent sharing) ───────────────────

def test_comm_merge_isolated_matches_self_only():
    """With no edges in the adjacency graph, merging must match self-only update."""
    H = W = 6
    N = 2
    obs_r = 1
    obs_d = 2 * obs_r + 1
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[1, 1], [4, 4]], dtype=jnp.int32)
    scans = jnp.full((N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    adj_none = jnp.zeros((N, N), dtype=jnp.bool_)

    merged = update_local_maps_with_comm(local_map, scans, positions, adj_none, obs_r)
    self_only = update_local_maps(local_map, scans, positions, obs_r)
    assert jnp.array_equal(merged, self_only)


def test_comm_merge_neighbor_extends_map():
    """Receiver accepts neighbour's scan and gains knowledge outside its own patch."""
    H = W = 6
    N = 2
    obs_r = 1
    obs_d = 2 * obs_r + 1
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)

    # Agent 0 at (1, 1), Agent 1 at (4, 4) — non-overlapping patches.
    positions = jnp.array([[1, 1], [4, 4]], dtype=jnp.int32)
    scans = jnp.full((N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)

    # Fully-connected adjacency: both can send to each other.
    adj_full = ~jnp.eye(N, dtype=jnp.bool_)

    merged = update_local_maps_with_comm(local_map, scans, positions, adj_full, obs_r)

    # Agent 0 must know cells around agent 1 (4,4) thanks to the shared scan.
    assert int(merged[0, 4, 4]) == MAP_FREE
    assert int(merged[0, 3, 4]) == MAP_FREE
    # And still know its own surroundings.
    assert int(merged[0, 1, 1]) == MAP_FREE
    # Cell far from both patches stays unknown.
    assert int(merged[0, 0, 5]) == MAP_UNKNOWN

    # Symmetric check for agent 1.
    assert int(merged[1, 1, 1]) == MAP_FREE
    assert int(merged[1, 4, 4]) == MAP_FREE


def test_comm_merge_directed_one_way():
    """adj[0, 1]=True only: 1 receives from 0, but 0 stays self-only."""
    H = W = 6
    N = 2
    obs_r = 1
    obs_d = 2 * obs_r + 1
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[1, 1], [4, 4]], dtype=jnp.int32)
    scans = jnp.full((N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)

    adj = jnp.zeros((N, N), dtype=jnp.bool_)
    adj = adj.at[0, 1].set(True)  # 0 -> 1 only

    merged = update_local_maps_with_comm(local_map, scans, positions, adj, obs_r)

    # Agent 1 learned agent 0's neighbourhood.
    assert int(merged[1, 1, 1]) == MAP_FREE
    # Agent 0 did NOT learn agent 1's neighbourhood.
    assert int(merged[0, 4, 4]) == MAP_UNKNOWN


def test_survey_radius_zero_writes_only_current_cell():
    """With survey_radius=0, only the cell the agent stands on is committed."""
    H = W = 6
    N = 1
    view_r = 2
    view_d = 2 * view_r + 1

    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[3, 3]], dtype=jnp.int32)
    scan = jnp.full((N, view_d, view_d), CELL_EMPTY, dtype=jnp.int32)
    adj_none = jnp.zeros((N, N), dtype=jnp.bool_)

    merged = update_local_maps_with_comm(
        local_map, scan, positions, adj_none,
        view_radius=view_r, survey_radius=0,
    )

    # Only (3, 3) should be MAP_FREE. Neighbouring cells must stay unknown.
    assert int(merged[0, 3, 3]) == MAP_FREE
    for (r, c) in [(2, 3), (4, 3), (3, 2), (3, 4), (2, 2), (4, 4)]:
        assert int(merged[0, r, c]) == MAP_UNKNOWN, f"cell ({r},{c}) leaked"


def test_survey_radius_equals_view_matches_legacy():
    """survey_radius=view_radius must reproduce the single-radius behavior."""
    H = W = 6
    N = 2
    r = 1
    view_d = 2 * r + 1
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[2, 2], [4, 4]], dtype=jnp.int32)
    scans = jnp.full((N, view_d, view_d), CELL_EMPTY, dtype=jnp.int32)
    adj = ~jnp.eye(N, dtype=jnp.bool_)

    legacy = update_local_maps_with_comm(local_map, scans, positions, adj, r)
    split = update_local_maps_with_comm(
        local_map, scans, positions, adj, view_radius=r, survey_radius=r,
    )
    assert jnp.array_equal(legacy, split)


def test_survey_radius_zero_over_comm_channel():
    """Across comm, each sender broadcasts exactly its own cell when survey_radius=0."""
    H = W = 6
    N = 2
    view_r = 1
    view_d = 2 * view_r + 1

    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[1, 1], [4, 4]], dtype=jnp.int32)
    scans = jnp.full((N, view_d, view_d), CELL_EMPTY, dtype=jnp.int32)
    adj_full = ~jnp.eye(N, dtype=jnp.bool_)

    merged = update_local_maps_with_comm(
        local_map, scans, positions, adj_full,
        view_radius=view_r, survey_radius=0,
    )

    # Each agent knows both cells (its own + the other's), but NOT the 3×3
    # neighbourhoods of those cells — only the two points.
    assert int(merged[0, 1, 1]) == MAP_FREE
    assert int(merged[0, 4, 4]) == MAP_FREE
    assert int(merged[0, 0, 0]) == MAP_UNKNOWN
    assert int(merged[0, 3, 4]) == MAP_UNKNOWN  # neighbour's neighbour stays unknown
    assert int(merged[1, 1, 1]) == MAP_FREE
    assert int(merged[1, 4, 4]) == MAP_FREE


def test_survey_radius_larger_than_view_rejected():
    """survey_radius > view_radius must raise — sensor can't commit what it didn't see."""
    H = W = 6
    N = 1
    view_r = 1
    view_d = 2 * view_r + 1
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[3, 3]], dtype=jnp.int32)
    scans = jnp.full((N, view_d, view_d), CELL_EMPTY, dtype=jnp.int32)
    adj = jnp.zeros((N, N), dtype=jnp.bool_)

    with pytest.raises(ValueError, match="must be <="):
        update_local_maps_with_comm(
            local_map, scans, positions, adj, view_radius=1, survey_radius=2,
        )


def test_comm_merge_jit_and_vmap():
    """Function must be jittable and vmappable over batch dim."""
    B = 3
    N = 2
    H = W = 5
    obs_r = 1
    obs_d = 2 * obs_r + 1
    keys = jax.random.split(jax.random.PRNGKey(0), B)

    local_map = jnp.full((B, N, H, W), MAP_UNKNOWN, dtype=jnp.int32)
    positions = jnp.array([[[1, 1], [3, 3]]] * B, dtype=jnp.int32)
    scans = jnp.full((B, N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    adj = jnp.broadcast_to(~jnp.eye(N, dtype=jnp.bool_), (B, N, N))

    fn = jax.jit(lambda lm, sc, p, a: update_local_maps_with_comm(lm, sc, p, a, obs_r))
    out = jax.vmap(fn)(local_map, scans, positions, adj)
    assert out.shape == (B, N, H, W)
