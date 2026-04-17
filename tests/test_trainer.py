"""Tests for the PureJaxRL trainer in red_within_blue.training.trainer."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.training.config import (
    ExperimentConfig,
    EnvParams,
    NetworkParams,
    TrainParams,
    RewardParams,
)
from red_within_blue.training.trainer import make_train, make_train_multi_seed


# ---------------------------------------------------------------------------
# Shared small config builder
# ---------------------------------------------------------------------------

def _small_config(
    method: str = "actor_critic",
    num_agents: int = 1,
    num_episodes: int = 30,
    num_seeds: int = 2,
) -> ExperimentConfig:
    """Build a tiny config for fast tests."""
    return ExperimentConfig(
        env=EnvParams(
            grid_width=4,
            grid_height=4,
            num_agents=num_agents,
            wall_density=0.0,
            max_steps=20,
            comm_radius=3.0,
        ),
        network=NetworkParams(
            actor_hidden_dim=32,
            actor_num_layers=1,
            critic_hidden_dim=32,
            critic_num_layers=1,
        ),
        train=TrainParams(
            method=method,
            lr=3e-3,
            gamma=0.9,
            vf_coef=0.5,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
        ),
        reward=RewardParams(disconnect_penalty=-0.5),
        enforce_connectivity=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMakeTrainReinforce:
    """Test make_train with method='reinforce'."""

    def test_returns_callable(self):
        cfg = _small_config(method="reinforce", num_episodes=5)
        train_fn = make_train(cfg)
        assert callable(train_fn)

    def test_output_shapes(self):
        cfg = _small_config(method="reinforce", num_episodes=5)
        train_fn = make_train(cfg)
        key = jax.random.PRNGKey(0)
        actor_params, critic_params, metrics = train_fn(key)

        # reinforce has no critic
        assert critic_params is None

        # actor_params should be a nested dict (Flax params)
        assert "params" in actor_params

        # metrics should have per-episode arrays
        assert "loss" in metrics
        assert "total_reward" in metrics
        assert metrics["loss"].shape == (5,)
        assert metrics["total_reward"].shape == (5,)


class TestMakeTrainActorCritic:
    """Test make_train with method='actor_critic'."""

    def test_returns_both_params(self):
        cfg = _small_config(method="actor_critic", num_episodes=5)
        train_fn = make_train(cfg)
        key = jax.random.PRNGKey(1)
        actor_params, critic_params, metrics = train_fn(key)

        # actor_critic should return both
        assert actor_params is not None
        assert critic_params is not None
        assert "params" in actor_params
        assert "params" in critic_params

        assert metrics["loss"].shape == (5,)
        assert metrics["total_reward"].shape == (5,)


class TestLossDecreases:
    """Train for 50 episodes and check loss trend."""

    @pytest.mark.slow
    def test_loss_trend(self):
        cfg = _small_config(method="actor_critic", num_episodes=50)
        train_fn = make_train(cfg)
        key = jax.random.PRNGKey(42)
        _, _, metrics = train_fn(key)

        loss = metrics["loss"]
        first_10 = jnp.mean(loss[:10])
        last_10 = jnp.mean(loss[-10:])

        # The loss should generally decrease. We allow some slack: the last
        # 10 episodes should have lower mean loss than the first 10.
        # If this fails intermittently, the test is still valuable as a
        # smoke test that training runs end-to-end.
        assert last_10 < first_10, (
            f"Loss did not decrease: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )


class TestMultiSeed:
    """Test make_train_multi_seed."""

    def test_output_leading_dim(self):
        cfg = _small_config(method="reinforce", num_episodes=5, num_seeds=2)
        train_multi = make_train_multi_seed(cfg)
        key = jax.random.PRNGKey(10)
        actor_params, critic_params, metrics = train_multi(key)

        # metrics should have leading [num_seeds] dimension
        assert metrics["loss"].shape == (2, 5)
        assert metrics["total_reward"].shape == (2, 5)

        # critic_params should be None for reinforce (vmapped None stays None)
        assert critic_params is None

    def test_actor_critic_multi_seed(self):
        cfg = _small_config(method="actor_critic", num_episodes=5, num_seeds=2)
        train_multi = make_train_multi_seed(cfg)
        key = jax.random.PRNGKey(11)
        actor_params, critic_params, metrics = train_multi(key)

        assert metrics["loss"].shape == (2, 5)
        assert critic_params is not None


class TestDeterministic:
    """Same key must produce identical results."""

    def test_same_key_same_result(self):
        cfg = _small_config(method="actor_critic", num_episodes=5)
        train_fn = make_train(cfg)

        key = jax.random.PRNGKey(99)
        _, _, metrics_a = train_fn(key)
        _, _, metrics_b = train_fn(key)

        assert jnp.allclose(metrics_a["loss"], metrics_b["loss"]), (
            "Different results for the same key"
        )
        assert jnp.allclose(metrics_a["total_reward"], metrics_b["total_reward"]), (
            "Different results for the same key"
        )


class TestMultiAgent:
    """Config with num_agents=2 should run without error."""

    def test_reinforce_multi_agent(self):
        cfg = _small_config(method="reinforce", num_agents=2, num_episodes=5)
        train_fn = make_train(cfg)
        key = jax.random.PRNGKey(20)
        actor_params, critic_params, metrics = train_fn(key)

        assert critic_params is None
        assert metrics["loss"].shape == (5,)

    def test_actor_critic_multi_agent(self):
        cfg = _small_config(method="actor_critic", num_agents=2, num_episodes=5)
        train_fn = make_train(cfg)
        key = jax.random.PRNGKey(21)
        actor_params, critic_params, metrics = train_fn(key)

        assert critic_params is not None
        assert metrics["loss"].shape == (5,)

    def test_baseline_multi_agent(self):
        cfg = _small_config(method="baseline", num_agents=2, num_episodes=5)
        train_fn = make_train(cfg)
        key = jax.random.PRNGKey(22)
        actor_params, critic_params, metrics = train_fn(key)

        assert critic_params is None
        assert metrics["loss"].shape == (5,)
