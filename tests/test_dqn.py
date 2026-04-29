"""Tests for the DQN training module."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.training.dqn import (
    tabular_q_update,
    dqn_loss,
    compute_dqn_targets,
    epsilon_greedy,
)
from red_within_blue.training.networks import QNetwork


# ---------------------------------------------------------------------------
# tabular_q_update
# ---------------------------------------------------------------------------

class TestTabularQUpdate:
    def test_basic_update(self):
        """Q(0,2) starts at 0; after update with r=1.0, alpha=0.1, gamma=0.99
        and all next-state Q-values zero -> Q(0,2) = 0.1."""
        num_states, num_actions = 5, 4
        q_table = jnp.zeros((num_states, num_actions))

        updated = tabular_q_update(
            q_table,
            state=0,
            action=2,
            reward=1.0,
            next_state=1,
            alpha=0.1,
            gamma=0.99,
            done=False,
        )

        assert updated.shape == (num_states, num_actions)
        # next_state Q-values are all 0, so TD target = 1.0 + 0.99 * 0 = 1.0
        # update: 0 + 0.1 * (1.0 - 0) = 0.1
        assert jnp.isclose(updated[0, 2], 0.1, atol=1e-6)
        # All other entries must remain 0
        assert jnp.allclose(
            updated.at[0, 2].set(0.0), jnp.zeros((num_states, num_actions))
        )

    def test_terminal_no_bootstrap(self):
        """When done=True the next-state value must not be bootstrapped."""
        q_table = jnp.ones((3, 3))  # all Q-values = 1

        updated = tabular_q_update(
            q_table,
            state=0,
            action=0,
            reward=2.0,
            next_state=1,
            alpha=0.5,
            gamma=0.99,
            done=True,
        )

        # TD target = 2.0 + 0.99 * max_next_q * (1 - 1) = 2.0
        # TD error = 2.0 - 1.0 = 1.0
        # new value = 1.0 + 0.5 * 1.0 = 1.5
        assert jnp.isclose(updated[0, 0], 1.5, atol=1e-6)

        # Verify: if done were False, the value would differ (bootstrap matters)
        updated_no_done = tabular_q_update(
            q_table,
            state=0,
            action=0,
            reward=2.0,
            next_state=1,
            alpha=0.5,
            gamma=0.99,
            done=False,
        )
        assert not jnp.isclose(updated_no_done[0, 0], updated[0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# dqn_loss
# ---------------------------------------------------------------------------

class TestDqnLoss:
    def test_returns_finite_scalar(self):
        """dqn_loss should return a finite scalar for a valid batch."""
        batch_size = 8
        obs_dim = 376
        num_actions = 5

        model = QNetwork(num_actions=num_actions, hidden_dim=64)
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.ones((obs_dim,))
        params = model.init(key, dummy_obs)

        observations = jax.random.normal(key, (batch_size, obs_dim))
        actions = jax.random.randint(key, (batch_size,), 0, num_actions)
        targets = jax.random.normal(key, (batch_size,))

        loss = dqn_loss(model, params, observations, actions, targets)

        assert loss.shape == ()
        assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# epsilon_greedy
# ---------------------------------------------------------------------------

class TestEpsilonGreedy:
    def test_greedy_selects_argmax(self):
        """With epsilon=0, the greedy action (argmax) must always be selected."""
        key = jax.random.PRNGKey(42)
        q_values = jnp.array([0.1, 0.5, 0.9, 0.3])
        action = epsilon_greedy(key, q_values, epsilon=0.0)
        assert int(action) == 2  # index of 0.9

    def test_random_returns_valid_action(self):
        """With epsilon=1, action must still be a valid action index."""
        num_actions = 5
        q_values = jnp.zeros(num_actions)
        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            action = epsilon_greedy(key, q_values, epsilon=1.0)
            assert 0 <= int(action) < num_actions
