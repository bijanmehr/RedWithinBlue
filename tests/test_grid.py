"""Tests for red_within_blue.grid — pure-JAX grid utilities."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.grid import (
    create_grid,
    get_local_scan,
    update_occupancy,
    update_exploration,
    apply_red_contamination,
)
from red_within_blue.types import (
    CELL_EMPTY,
    CELL_WALL,
    CELL_OBSTACLE,
    CELL_OCCUPIED,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W, H = 8, 8  # small grid used across most tests
KEY = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# create_grid tests
# ---------------------------------------------------------------------------


def test_create_grid_boundaries():
    """Boundary cells (row 0, row H-1, col 0, col W-1) are always walls."""
    gs = create_grid(W, H, wall_density=0.0, key=KEY)
    terrain = gs.terrain

    # Top row, bottom row
    assert jnp.all(terrain[0, :] == CELL_WALL).item()
    assert jnp.all(terrain[H - 1, :] == CELL_WALL).item()
    # Left col, right col
    assert jnp.all(terrain[:, 0] == CELL_WALL).item()
    assert jnp.all(terrain[:, W - 1] == CELL_WALL).item()


def test_create_grid_density():
    """Interior wall count roughly matches wall_density."""
    density = 0.3
    gs = create_grid(W, H, wall_density=density, key=KEY)
    terrain = gs.terrain

    # Interior = rows 1..H-2, cols 1..W-2
    interior = terrain[1 : H - 1, 1 : W - 1]
    num_interior = interior.size
    num_walls = int(jnp.sum(interior == CELL_WALL).item())

    # Allow generous tolerance for small grids.
    expected = density * num_interior
    assert abs(num_walls - expected) < 0.5 * num_interior, (
        f"Expected ~{expected:.1f} interior walls, got {num_walls}"
    )


def test_create_grid_seeded():
    """Same PRNG key produces identical grids."""
    k = jax.random.PRNGKey(99)
    gs1 = create_grid(W, H, wall_density=0.2, key=k)
    gs2 = create_grid(W, H, wall_density=0.2, key=k)

    assert jnp.array_equal(gs1.terrain, gs2.terrain)
    assert jnp.array_equal(gs1.occupancy, gs2.occupancy)
    assert jnp.array_equal(gs1.explored, gs2.explored)


# ---------------------------------------------------------------------------
# get_local_scan tests
# ---------------------------------------------------------------------------


def test_local_scan_center():
    """Agent at center of an open grid gets the expected patch."""
    # Fully open grid (density 0 → only boundary walls).
    gs = create_grid(W, H, wall_density=0.0, key=KEY)
    terrain = gs.terrain
    occupancy = gs.occupancy

    obs_radius = 2
    obs_d = 2 * obs_radius + 1  # 5x5

    # Place agent at (4, 4) — well inside the 8x8 grid.
    pos = jnp.array([4, 4], dtype=jnp.int32)
    scan = get_local_scan(terrain, occupancy, pos, obs_radius)

    assert scan.shape == (obs_d, obs_d)

    # The center cell should match terrain at (4, 4) which is CELL_EMPTY.
    assert scan[obs_radius, obs_radius].item() == CELL_EMPTY

    # All cells in a 5x5 window centred on (4,4) are interior (rows 2-6, cols 2-6),
    # so everything should be CELL_EMPTY.
    assert jnp.all(scan == CELL_EMPTY).item()


def test_local_scan_edge():
    """Agent near boundary gets CELL_WALL for out-of-bounds cells."""
    gs = create_grid(W, H, wall_density=0.0, key=KEY)
    terrain = gs.terrain
    occupancy = gs.occupancy

    obs_radius = 2
    obs_d = 2 * obs_radius + 1  # 5x5

    # Place agent at top-left interior corner (1, 1).
    pos = jnp.array([1, 1], dtype=jnp.int32)
    scan = get_local_scan(terrain, occupancy, pos, obs_radius)

    assert scan.shape == (obs_d, obs_d)

    # Top-left corner of the scan window corresponds to grid (-1, -1) → OOB → CELL_WALL.
    assert scan[0, 0].item() == CELL_WALL

    # The centre of the scan is the agent's own cell at (1,1) which is CELL_EMPTY
    # (interior cell in a 0-density grid).
    assert scan[obs_radius, obs_radius].item() == CELL_EMPTY


def test_local_scan_sees_obstacles():
    """Scan correctly reflects terrain types including obstacles."""
    gs = create_grid(W, H, wall_density=0.0, key=KEY)
    terrain = gs.terrain

    # Manually inject an obstacle at (3, 3).
    terrain = terrain.at[3, 3].set(CELL_OBSTACLE)

    occupancy = gs.occupancy
    obs_radius = 2

    # Agent at (3, 4) — the obstacle at (3,3) is 1 cell to the left.
    pos = jnp.array([3, 4], dtype=jnp.int32)
    scan = get_local_scan(terrain, occupancy, pos, obs_radius)

    # In the 5x5 scan centred on (3,4), the cell at relative (-0, -1) = (obs_radius, obs_radius-1)
    # should be CELL_OBSTACLE.
    assert scan[obs_radius, obs_radius - 1].item() == CELL_OBSTACLE


# ---------------------------------------------------------------------------
# update_occupancy tests
# ---------------------------------------------------------------------------


def test_update_occupancy():
    """Occupancy map shows agent UIDs at their positions."""
    positions = jnp.array([[2, 3], [5, 6]], dtype=jnp.int32)
    uids = jnp.array([10, 20], dtype=jnp.int32)

    occ = update_occupancy(positions, uids, (H, W))

    assert occ[2, 3].item() == 10
    assert occ[5, 6].item() == 20
    # Everything else should be zero.
    total = jnp.sum(occ).item()
    assert total == 30  # 10 + 20


# ---------------------------------------------------------------------------
# update_exploration tests
# ---------------------------------------------------------------------------


def test_update_exploration():
    """Visit counts increment correctly for multiple agents and repeated visits."""
    explored = jnp.zeros((H, W), dtype=jnp.int32)

    # Two agents, one at (2,3) and one at (4,5).
    positions = jnp.array([[2, 3], [4, 5]], dtype=jnp.int32)
    explored = update_exploration(explored, positions)

    assert explored[2, 3].item() == 1
    assert explored[4, 5].item() == 1

    # Second step: both agents stay put.
    explored = update_exploration(explored, positions)
    assert explored[2, 3].item() == 2
    assert explored[4, 5].item() == 2

    # Third step: one agent moves, the other stays.
    positions2 = jnp.array([[2, 3], [1, 1]], dtype=jnp.int32)
    explored = update_exploration(explored, positions2)
    assert explored[2, 3].item() == 3
    assert explored[4, 5].item() == 2  # no longer visited


# ---------------------------------------------------------------------------
# apply_red_contamination tests
# ---------------------------------------------------------------------------


def _seed_explored(positions):
    explored = jnp.zeros((H, W), dtype=jnp.int32)
    return update_exploration(explored, positions)


def test_red_contamination_noop_when_zero_red():
    """num_red_agents=0 leaves explored bit-identical."""
    positions = jnp.array([[2, 3], [4, 5]], dtype=jnp.int32)
    explored = _seed_explored(positions)
    new_explored = apply_red_contamination(explored, positions, num_red_agents=0)
    assert jnp.array_equal(new_explored, explored).item()


def test_red_contamination_zeros_red_cell():
    """Last `num_red_agents` agents zero `explored` at their cells."""
    # 1 blue + 1 red, distinct cells.
    positions = jnp.array([[2, 3], [4, 5]], dtype=jnp.int32)
    explored = _seed_explored(positions)
    new_explored = apply_red_contamination(explored, positions, num_red_agents=1)
    # Blue cell unchanged, red cell zeroed.
    assert new_explored[2, 3].item() == 1
    assert new_explored[4, 5].item() == 0


def test_red_contamination_colocated_blue_red_stays_zero():
    """1 blue + 1 red on the same cell → explored stays 0 across many steps."""
    positions = jnp.array([[3, 3], [3, 3]], dtype=jnp.int32)
    explored = jnp.zeros((H, W), dtype=jnp.int32)
    for _ in range(5):
        explored = update_exploration(explored, positions)
        explored = apply_red_contamination(explored, positions, num_red_agents=1)
        # Cell visited by red same step → always reset to 0.
        assert explored[3, 3].item() == 0


def test_red_contamination_jit_compatible():
    """Helper compiles under jit when num_red_agents is supplied as a static arg."""
    positions = jnp.array([[2, 3], [4, 5]], dtype=jnp.int32)
    explored = _seed_explored(positions)

    @jax.jit
    def _runner(expl, pos):
        return apply_red_contamination(expl, pos, num_red_agents=1)

    new_explored = _runner(explored, positions)
    assert new_explored[4, 5].item() == 0
    assert new_explored[2, 3].item() == 1
