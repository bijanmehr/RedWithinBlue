"""Tests for red_within_blue.agents — agent init, local-map update, messages."""

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
    prepare_messages,
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
        msg_dim=4,
        comm_radius=5.0,
        max_steps=64,
    )


@pytest.fixture
def terrain_8x8():
    """8x8 terrain: all empty except a wall ring on row 0 and column 0."""
    t = jnp.zeros((8, 8), dtype=jnp.int32)
    t = t.at[0, :].set(CELL_WALL)
    t = t.at[:, 0].set(CELL_WALL)
    # One obstacle at (3, 3)
    t = t.at[3, 3].set(CELL_OBSTACLE)
    return t


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ── 27. test_init_agents_valid_positions ────────────────────────────

def test_init_agents_valid_positions(small_config, terrain_8x8, key):
    """All spawn positions must be on empty cells and within grid bounds."""
    state = init_agents(small_config, terrain_8x8, key)

    N = small_config.num_agents
    H, W = small_config.grid_height, small_config.grid_width

    assert state.positions.shape == (N, 2)

    for i in range(N):
        r, c = int(state.positions[i, 0]), int(state.positions[i, 1])
        assert 0 <= r < H, f"Agent {i} row {r} out of bounds"
        assert 0 <= c < W, f"Agent {i} col {c} out of bounds"
        assert int(terrain_8x8[r, c]) == CELL_EMPTY, (
            f"Agent {i} spawned on non-empty cell ({r},{c})={terrain_8x8[r, c]}"
        )


# ── 28. test_init_agents_seeded ─────────────────────────────────────

def test_init_agents_seeded(small_config, terrain_8x8):
    """Same PRNG key must produce identical agent states."""
    k = jax.random.PRNGKey(123)
    s1 = init_agents(small_config, terrain_8x8, k)
    s2 = init_agents(small_config, terrain_8x8, k)

    assert jnp.array_equal(s1.positions, s2.positions)
    assert jnp.array_equal(s1.uids, s2.uids)
    assert jnp.array_equal(s1.local_map, s2.local_map)
    assert jnp.array_equal(s1.messages_out, s2.messages_out)


# ── 29. test_update_local_map ───────────────────────────────────────

def test_update_local_map(small_config):
    """After update, scanned cells must transition from UNKNOWN to correct values."""
    H, W = small_config.grid_height, small_config.grid_width
    N = small_config.num_agents
    obs_r = small_config.obs_radius
    obs_d = 2 * obs_r + 1

    # Start with fully unknown local maps
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)

    # Agent 0 at (4, 4), Agent 1 at (6, 6) — well inside the 8x8 grid
    positions = jnp.array([[4, 4], [6, 6]], dtype=jnp.int32)

    # Build a scan patch for agent 0: all empty except center is wall
    scan0 = jnp.full((obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    scan0 = scan0.at[obs_r, obs_r].set(CELL_WALL)

    # Scan for agent 1: all empty except top-left is obstacle
    scan1 = jnp.full((obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    scan1 = scan1.at[0, 0].set(CELL_OBSTACLE)

    local_scan = jnp.stack([scan0, scan1], axis=0)  # [2, obs_d, obs_d]

    updated = update_local_maps(local_map, local_scan, positions, obs_r)

    # Agent 0: center of its patch is (4,4) in the map → should be MAP_WALL
    assert int(updated[0, 4, 4]) == MAP_WALL
    # The cell at (4-obs_r, 4-obs_r) = (2,2) should be MAP_FREE
    assert int(updated[0, 2, 2]) == MAP_FREE
    # A cell outside the patch should still be MAP_UNKNOWN
    assert int(updated[0, 0, 0]) == MAP_UNKNOWN

    # Agent 1: top-left of its scan is at map pos (6-obs_r, 6-obs_r) = (4,4)
    assert int(updated[1, 4, 4]) == MAP_OBSTACLE
    # Center (6,6) should be MAP_FREE
    assert int(updated[1, 6, 6]) == MAP_FREE


# ── 30. test_local_map_persistence ──────────────────────────────────

def test_local_map_persistence(small_config):
    """Previously scanned cells remain in the local map after a second update at a new position."""
    H, W = small_config.grid_height, small_config.grid_width
    obs_r = small_config.obs_radius
    obs_d = 2 * obs_r + 1

    # Single agent for simplicity (override config)
    N = 1
    local_map = jnp.full((N, H, W), MAP_UNKNOWN, dtype=jnp.int32)

    # Step 1: agent at (3, 3), scan all-empty
    pos1 = jnp.array([[3, 3]], dtype=jnp.int32)
    scan1 = jnp.full((N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    local_map = update_local_maps(local_map, scan1, pos1, obs_r)

    # The region around (3,3) should now be MAP_FREE
    assert int(local_map[0, 3, 3]) == MAP_FREE

    # Step 2: agent moves to (6, 6), scan all-empty
    pos2 = jnp.array([[6, 6]], dtype=jnp.int32)
    scan2 = jnp.full((N, obs_d, obs_d), CELL_EMPTY, dtype=jnp.int32)
    local_map = update_local_maps(local_map, scan2, pos2, obs_r)

    # New position scanned
    assert int(local_map[0, 6, 6]) == MAP_FREE
    # Old position still known (persistence!)
    assert int(local_map[0, 3, 3]) == MAP_FREE
    # A cell never in either patch should still be unknown
    assert int(local_map[0, 0, 0]) == MAP_UNKNOWN


# ── 31. test_prepare_messages_scan_only ─────────────────────────────

def test_prepare_messages_scan_only(small_config):
    """Without learned vectors, the message is [flat_scan | zeros]."""
    obs_r = small_config.obs_radius
    obs_d = 2 * obs_r + 1
    N = small_config.num_agents
    msg_dim = small_config.msg_dim
    scan_dim = obs_d * obs_d

    local_scan = jnp.ones((N, obs_d, obs_d), dtype=jnp.int32) * 2  # arbitrary value

    msgs = prepare_messages(local_scan, msg_dim, learned_vectors=None)

    assert msgs.shape == (N, scan_dim + msg_dim)
    # Scan part should equal the flattened scan (as float)
    expected_scan = local_scan.reshape(N, -1).astype(jnp.float32)
    assert jnp.allclose(msgs[:, :scan_dim], expected_scan)
    # Learned part should be all zeros
    assert jnp.allclose(msgs[:, scan_dim:], 0.0)


# ── 32. test_prepare_messages_with_learned ──────────────────────────

def test_prepare_messages_with_learned(small_config):
    """Learned vectors are appended correctly after the flattened scan."""
    obs_r = small_config.obs_radius
    obs_d = 2 * obs_r + 1
    N = small_config.num_agents
    msg_dim = small_config.msg_dim
    scan_dim = obs_d * obs_d

    local_scan = jnp.ones((N, obs_d, obs_d), dtype=jnp.int32)
    learned = jnp.arange(N * msg_dim, dtype=jnp.float32).reshape(N, msg_dim)

    msgs = prepare_messages(local_scan, msg_dim, learned_vectors=learned)

    assert msgs.shape == (N, scan_dim + msg_dim)
    # Scan portion
    expected_scan = local_scan.reshape(N, -1).astype(jnp.float32)
    assert jnp.allclose(msgs[:, :scan_dim], expected_scan)
    # Learned portion
    assert jnp.allclose(msgs[:, scan_dim:], learned)
