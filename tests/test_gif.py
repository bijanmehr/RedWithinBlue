"""Tests for the GIF recording module."""

import os
import tempfile

import jax
import jax.numpy as jnp
import pytest
from PIL import Image

from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig, EnvParams
from red_within_blue.training.gif import record_episode_gif
from red_within_blue.training.rewards_training import normalized_exploration_reward


def _make_env(num_agents=1, max_steps=10):
    """Build a small test env using the new config system."""
    cfg = ExperimentConfig(
        env=EnvParams(grid_width=10, grid_height=10, num_agents=num_agents,
                      max_steps=max_steps, comm_radius=3.0),
    )
    return GridCommEnv(cfg.to_env_config(), reward_fn=normalized_exploration_reward)


def _random_policy(key, obs):
    return jax.random.randint(key, shape=(), minval=0, maxval=5)


class TestRecordEpisodeGif:
    """Tests for record_episode_gif."""

    def test_creates_gif_file_single_agent(self):
        env = _make_env(num_agents=1, max_steps=10)
        key = jax.random.PRNGKey(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gif")
            result = record_episode_gif(env, _random_policy, key, path, fps=4)

            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            assert result["n_frames"] > 0
            assert result["visit_heatmap"] is not None
            assert isinstance(result["connectivity"], list)
            assert isinstance(result["coverage_over_time"], list)

    def test_creates_gif_file_multi_agent(self):
        env = _make_env(num_agents=2, max_steps=10)
        key = jax.random.PRNGKey(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multi.gif")
            result = record_episode_gif(env, _random_policy, key, path, fps=2)

            assert os.path.exists(path)
            assert result["n_frames"] > 0

    def test_gif_has_multiple_frames(self):
        env = _make_env(num_agents=1, max_steps=10)
        key = jax.random.PRNGKey(7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frames.gif")
            result = record_episode_gif(env, _random_policy, key, path)

            # At least the reset frame + 1 step recorded.
            assert result["n_frames"] >= 2

            # The GIF file should contain multiple frames.
            img = Image.open(path)
            assert getattr(img, "n_frames", 1) >= 2

    def test_frame_count_bounded_by_max_steps(self):
        max_steps = 5
        env = _make_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(99)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bounded.gif")
            result = record_episode_gif(env, _random_policy, key, path)

            # reset frame + up to max_steps step frames
            assert result["n_frames"] <= max_steps + 1
