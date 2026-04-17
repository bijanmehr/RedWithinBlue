"""Tests for red_within_blue.movement.resolve_actions."""

import jax.numpy as jnp
import pytest

from red_within_blue.movement import resolve_actions
from red_within_blue.types import (
    Action,
    CELL_EMPTY,
    CELL_OBSTACLE,
    CELL_WALL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_SHAPE = (8, 8)


def _empty_terrain(shape=GRID_SHAPE):
    """Return an all-empty 8x8 terrain."""
    return jnp.zeros(shape, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_movement():
    """Each action produces the correct position delta on an open grid."""
    terrain = _empty_terrain()

    # Place a single agent at (4, 4) — room to move in every direction.
    positions = jnp.array([[4, 4]], dtype=jnp.int32)

    expected = {
        Action.STAY:  [4, 4],
        Action.UP:    [3, 4],
        Action.DOWN:  [5, 4],
        Action.LEFT:  [4, 3],
        Action.RIGHT: [4, 5],
    }

    for action, expected_pos in expected.items():
        actions = jnp.array([action], dtype=jnp.int32)
        new_pos, coll = resolve_actions(positions, actions, terrain, GRID_SHAPE)
        assert jnp.array_equal(new_pos[0], jnp.array(expected_pos, dtype=jnp.int32)), (
            f"Action {Action(action).name}: expected {expected_pos}, got {new_pos[0].tolist()}"
        )
        assert not coll[0], f"No collision expected for single agent moving {Action(action).name}"


def test_wall_collision():
    """An agent trying to move into a wall cell stays at its current position."""
    terrain = _empty_terrain()
    # Place a wall at (3, 4) — directly above the agent at (4, 4).
    terrain = terrain.at[3, 4].set(CELL_WALL)

    positions = jnp.array([[4, 4]], dtype=jnp.int32)
    actions = jnp.array([Action.UP], dtype=jnp.int32)

    new_pos, coll = resolve_actions(positions, actions, terrain, GRID_SHAPE)
    assert jnp.array_equal(new_pos[0], jnp.array([4, 4], dtype=jnp.int32)), (
        f"Agent should stay at (4,4) when moving into a wall, got {new_pos[0].tolist()}"
    )
    # Wall rejection is not an agent-agent collision.
    assert not coll[0]


def test_boundary_clamp():
    """Moving off the grid edge is rejected (clamped, then same-cell = stay)."""
    terrain = _empty_terrain()

    # Agent at top-left corner trying to move UP and LEFT.
    positions = jnp.array([[0, 0], [0, 0]], dtype=jnp.int32)
    actions = jnp.array([Action.UP, Action.LEFT], dtype=jnp.int32)

    new_pos, coll = resolve_actions(positions, actions, terrain, GRID_SHAPE)
    # Both agents clamp to (0,0) — same cell — so they collide and stay.
    assert jnp.array_equal(new_pos[0], jnp.array([0, 0], dtype=jnp.int32))
    assert jnp.array_equal(new_pos[1], jnp.array([0, 0], dtype=jnp.int32))

    # Single agent at edge: should stay at boundary.
    positions_single = jnp.array([[0, 3]], dtype=jnp.int32)
    actions_single = jnp.array([Action.UP], dtype=jnp.int32)
    new_pos_s, coll_s = resolve_actions(
        positions_single, actions_single, terrain, GRID_SHAPE
    )
    assert jnp.array_equal(new_pos_s[0], jnp.array([0, 3], dtype=jnp.int32))
    assert not coll_s[0], "Single agent at boundary should not be flagged as collision"


def test_agent_collision():
    """Two agents targeting the same cell both stay at their original positions."""
    terrain = _empty_terrain()

    # Agent 0 at (3, 4) moves DOWN → targets (4, 4).
    # Agent 1 at (4, 3) moves RIGHT → targets (4, 4).
    positions = jnp.array([[3, 4], [4, 3]], dtype=jnp.int32)
    actions = jnp.array([Action.DOWN, Action.RIGHT], dtype=jnp.int32)

    new_pos, coll = resolve_actions(positions, actions, terrain, GRID_SHAPE)
    # Both should revert to their original positions.
    assert jnp.array_equal(new_pos[0], jnp.array([3, 4], dtype=jnp.int32))
    assert jnp.array_equal(new_pos[1], jnp.array([4, 3], dtype=jnp.int32))
    assert coll[0] and coll[1], "Both agents should be marked as collided"


def test_obstacle_passability():
    """passable_types configuration is respected — obstacles can be made passable."""
    terrain = _empty_terrain()
    terrain = terrain.at[3, 4].set(CELL_OBSTACLE)  # obstacle at (3, 4)

    positions = jnp.array([[4, 4]], dtype=jnp.int32)
    actions = jnp.array([Action.UP], dtype=jnp.int32)

    # Default: obstacles are NOT passable.
    new_pos, _ = resolve_actions(positions, actions, terrain, GRID_SHAPE)
    assert jnp.array_equal(new_pos[0], jnp.array([4, 4], dtype=jnp.int32)), (
        "Obstacle should block by default"
    )

    # Custom passable_types: CELL_EMPTY=True, CELL_WALL=False, CELL_OBSTACLE=True
    passable = jnp.array([True, False, True], dtype=jnp.bool_)
    new_pos2, _ = resolve_actions(
        positions, actions, terrain, GRID_SHAPE, passable_types=passable
    )
    assert jnp.array_equal(new_pos2[0], jnp.array([3, 4], dtype=jnp.int32)), (
        "Obstacle should be passable with custom passable_types"
    )


def test_collision_mask():
    """collision_mask correctly flags only the agents involved in a collision."""
    terrain = _empty_terrain()

    # Agent 0 at (2, 2) moves DOWN  → (3, 2)  — no conflict.
    # Agent 1 at (5, 4) moves LEFT  → (5, 3)  — conflicts with agent 2.
    # Agent 2 at (5, 2) moves RIGHT → (5, 3)  — conflicts with agent 1.
    # Agent 3 at (6, 6) STAY        → (6, 6)  — no conflict.
    positions = jnp.array(
        [[2, 2], [5, 4], [5, 2], [6, 6]], dtype=jnp.int32
    )
    actions = jnp.array(
        [Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY], dtype=jnp.int32
    )

    new_pos, coll = resolve_actions(positions, actions, terrain, GRID_SHAPE)

    # Agent 0 should have moved successfully.
    assert jnp.array_equal(new_pos[0], jnp.array([3, 2], dtype=jnp.int32))
    assert not coll[0], "Agent 0 should not be flagged"

    # Agents 1 and 2 collided — they should stay put.
    assert jnp.array_equal(new_pos[1], jnp.array([5, 4], dtype=jnp.int32))
    assert jnp.array_equal(new_pos[2], jnp.array([5, 2], dtype=jnp.int32))
    assert coll[1], "Agent 1 should be flagged as collided"
    assert coll[2], "Agent 2 should be flagged as collided"

    # Agent 3 stayed, no collision.
    assert jnp.array_equal(new_pos[3], jnp.array([6, 6], dtype=jnp.int32))
    assert not coll[3], "Agent 3 should not be flagged"
