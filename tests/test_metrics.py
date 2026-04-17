"""Tests for red_within_blue.training.metrics."""

import jax.numpy as jnp
import pytest

from red_within_blue.training.metrics import (
    compute_action_distribution,
    compute_connectivity_fraction,
    compute_coverage,
    compute_explained_variance,
    compute_steps_to_coverage,
)
from red_within_blue.types import CELL_WALL, CELL_EMPTY


# ---------------------------------------------------------------------------
# 1. compute_coverage
# ---------------------------------------------------------------------------

class TestComputeCoverage:
    """4x4 grid whose border is CELL_WALL (value=1), interior is CELL_EMPTY."""

    def _make_grid(self):
        """Return a 4x4 terrain with boundary walls."""
        # 1 1 1 1
        # 1 0 0 1
        # 1 0 0 1
        # 1 1 1 1
        terrain = jnp.ones((4, 4), dtype=jnp.int32) * CELL_WALL
        terrain = terrain.at[1:3, 1:3].set(CELL_EMPTY)
        return terrain  # 4 non-wall cells

    def test_half_explored(self):
        terrain = self._make_grid()
        # Mark 2 of 4 inner cells as explored (non-zero = visited)
        explored = jnp.zeros((4, 4), dtype=jnp.int32)
        explored = explored.at[1, 1].set(1)
        explored = explored.at[1, 2].set(1)

        result = compute_coverage(terrain, explored)
        assert float(result) == pytest.approx(0.5, abs=1e-6)

    def test_fully_explored(self):
        terrain = self._make_grid()
        explored = jnp.zeros((4, 4), dtype=jnp.int32)
        explored = explored.at[1:3, 1:3].set(1)

        result = compute_coverage(terrain, explored)
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_none_explored(self):
        terrain = self._make_grid()
        explored = jnp.zeros((4, 4), dtype=jnp.int32)

        result = compute_coverage(terrain, explored)
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_wall_visits_ignored(self):
        """Marking wall cells as explored must not affect the result."""
        terrain = self._make_grid()
        explored = jnp.zeros((4, 4), dtype=jnp.int32)
        # Mark all border (wall) cells visited, but no inner cells
        explored = explored.at[0, :].set(1)
        explored = explored.at[3, :].set(1)
        explored = explored.at[:, 0].set(1)
        explored = explored.at[:, 3].set(1)

        result = compute_coverage(terrain, explored)
        assert float(result) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. compute_action_distribution
# ---------------------------------------------------------------------------

class TestComputeActionDistribution:
    def test_sums_to_one(self):
        actions = jnp.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=jnp.int32)
        dist = compute_action_distribution(actions, num_actions=5)
        assert float(jnp.sum(dist)) == pytest.approx(1.0, abs=1e-6)

    def test_correct_per_action_count(self):
        # actions: 0 appears 2x, 1 appears 2x, 2 appears 2x, 3 appears 1x, 4 appears 1x
        # total = 8
        actions = jnp.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=jnp.int32)
        dist = compute_action_distribution(actions, num_actions=5)
        assert float(dist[0]) == pytest.approx(2 / 8, abs=1e-6)
        assert float(dist[1]) == pytest.approx(2 / 8, abs=1e-6)
        assert float(dist[2]) == pytest.approx(2 / 8, abs=1e-6)
        assert float(dist[3]) == pytest.approx(1 / 8, abs=1e-6)
        assert float(dist[4]) == pytest.approx(1 / 8, abs=1e-6)

    def test_single_action(self):
        actions = jnp.array([3], dtype=jnp.int32)
        dist = compute_action_distribution(actions, num_actions=5)
        assert float(dist[3]) == pytest.approx(1.0, abs=1e-6)
        assert float(jnp.sum(dist)) == pytest.approx(1.0, abs=1e-6)

    def test_output_shape(self):
        actions = jnp.array([0, 1, 2], dtype=jnp.int32)
        dist = compute_action_distribution(actions, num_actions=5)
        assert dist.shape == (5,)


# ---------------------------------------------------------------------------
# 3. compute_explained_variance
# ---------------------------------------------------------------------------

class TestComputeExplainedVariance:
    def test_perfect_predictions_ev_one(self):
        returns = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = returns  # perfect
        ev = compute_explained_variance(returns, predictions)
        assert float(ev) == pytest.approx(1.0, abs=1e-6)

    def test_random_predictions_ev_below_half(self):
        import jax
        key = jax.random.PRNGKey(42)
        returns = jax.random.normal(key, shape=(100,))
        key2 = jax.random.PRNGKey(99)
        predictions = jax.random.normal(key2, shape=(100,))  # uncorrelated
        ev = compute_explained_variance(returns, predictions)
        assert float(ev) < 0.5

    def test_zero_variance_returns_guard(self):
        """Constant returns should not cause division by zero."""
        returns = jnp.ones(10)
        predictions = jnp.zeros(10)
        ev = compute_explained_variance(returns, predictions)
        assert jnp.isfinite(ev)
        assert float(ev) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. compute_steps_to_coverage
# ---------------------------------------------------------------------------

class TestComputeStepsToCoverage:
    def test_threshold_reached_at_index_2(self):
        per_step_coverage = jnp.array([0.1, 0.3, 0.5, 0.75, 0.9])
        steps = compute_steps_to_coverage(per_step_coverage, threshold=0.5)
        assert int(steps) == 2

    def test_threshold_never_reached_returns_T(self):
        per_step_coverage = jnp.array([0.1, 0.2, 0.3])
        steps = compute_steps_to_coverage(per_step_coverage, threshold=0.9)
        assert int(steps) == 3  # T = len(per_step_coverage)

    def test_threshold_reached_at_first_step(self):
        per_step_coverage = jnp.array([1.0, 0.5, 0.2])
        steps = compute_steps_to_coverage(per_step_coverage, threshold=0.8)
        assert int(steps) == 0

    def test_exact_threshold_value(self):
        per_step_coverage = jnp.array([0.0, 0.5, 1.0])
        steps = compute_steps_to_coverage(per_step_coverage, threshold=0.5)
        assert int(steps) == 1


# ---------------------------------------------------------------------------
# 5. compute_connectivity_fraction
# ---------------------------------------------------------------------------

class TestComputeConnectivityFraction:
    def test_eight_of_ten_true(self):
        timeline = jnp.array(
            [True, True, True, True, True, True, True, True, False, False]
        )
        frac = compute_connectivity_fraction(timeline)
        assert float(frac) == pytest.approx(0.8, abs=1e-6)

    def test_all_true(self):
        timeline = jnp.ones(5, dtype=jnp.bool_)
        frac = compute_connectivity_fraction(timeline)
        assert float(frac) == pytest.approx(1.0, abs=1e-6)

    def test_all_false(self):
        timeline = jnp.zeros(5, dtype=jnp.bool_)
        frac = compute_connectivity_fraction(timeline)
        assert float(frac) == pytest.approx(0.0, abs=1e-6)

    def test_single_true(self):
        timeline = jnp.array([True])
        frac = compute_connectivity_fraction(timeline)
        assert float(frac) == pytest.approx(1.0, abs=1e-6)
