"""Tests for TrajectoryWrapper."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from red_within_blue.env import GridCommEnv
from red_within_blue.types import EnvConfig
from red_within_blue.wrappers import TrajectoryWrapper


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _small_config(**overrides):
    defaults = dict(
        grid_width=8, grid_height=8, num_agents=2, max_steps=5,
        obs_radius=2, comm_radius=5.0, wall_density=0.0,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_wrapped_env(**overrides):
    cfg = _small_config(**overrides)
    env = GridCommEnv(cfg)
    wrapper = TrajectoryWrapper(env)
    return wrapper, env, cfg


def _random_actions(env, key):
    """Return a dict of random valid actions for every agent."""
    keys = jax.random.split(key, env.num_agents)
    return {a: jax.random.randint(keys[i], (), 0, 5) for i, a in enumerate(env.agents)}


# ------------------------------------------------------------------
# 47. test_trajectory_wrapper_records
# ------------------------------------------------------------------

def test_trajectory_wrapper_records():
    """Wrapper captures steps: buffer length == 1 (reset) + N (steps)."""
    wrapper, env, cfg = _make_wrapped_env()
    key = jax.random.PRNGKey(47)

    obs, state = wrapper.reset(key)
    n_steps = 3

    for i in range(n_steps):
        k_act, k_step = jax.random.split(jax.random.PRNGKey(100 + i))
        actions = _random_actions(env, k_act)
        obs, state, rewards, dones, info = wrapper.step(k_step, state, actions)

    traj = wrapper.get_trajectory()
    # 1 reset snapshot + n_steps step snapshots
    assert len(traj) == 1 + n_steps

    # Reset snapshot should have obs keys but no actions/rewards/dones
    reset_snap = traj[0]
    assert any(k.startswith("obs/") for k in reset_snap)
    assert not any(k.startswith("actions/") for k in reset_snap)

    # Step snapshots should have obs, actions, rewards, and dones
    for step_snap in traj[1:]:
        assert any(k.startswith("obs/") for k in step_snap)
        assert any(k.startswith("actions/") for k in step_snap)
        assert any(k.startswith("rewards/") for k in step_snap)
        assert any(k.startswith("dones/") for k in step_snap)


# ------------------------------------------------------------------
# 48. test_trajectory_wrapper_transparent
# ------------------------------------------------------------------

def test_trajectory_wrapper_transparent():
    """Wrapper does not alter env behaviour: obs and rewards match unwrapped."""
    wrapper, env, cfg = _make_wrapped_env()
    key = jax.random.PRNGKey(48)

    # Wrapped path
    w_obs, w_state = wrapper.reset(key)
    actions = {a: jnp.int32(1) for a in env.agents}  # all UP
    k_step = jax.random.PRNGKey(480)
    w_obs2, w_state2, w_rew, w_dones, _ = wrapper.step(k_step, w_state, actions)

    # Unwrapped path (use step_env for parity)
    u_obs, u_state = env.reset(key)
    u_obs2, u_state2, u_rew, u_dones, _ = env.step_env(k_step, u_state, actions)

    # Observations must match
    for a in env.agents:
        np.testing.assert_array_equal(np.asarray(w_obs[a]), np.asarray(u_obs[a]))
        np.testing.assert_array_equal(np.asarray(w_obs2[a]), np.asarray(u_obs2[a]))

    # Rewards must match
    for a in env.agents:
        assert float(w_rew[a]) == float(u_rew[a])

    # Dones must match
    for a in list(env.agents) + ["__all__"]:
        assert bool(w_dones[a]) == bool(u_dones[a])


# ------------------------------------------------------------------
# 49. test_trajectory_save_load
# ------------------------------------------------------------------

def test_trajectory_save_load(tmp_path):
    """Save trajectory to temp file, reload, and verify keys exist."""
    cfg = _small_config()
    env = GridCommEnv(cfg)
    wrapper = TrajectoryWrapper(env, save_dir=str(tmp_path))

    key = jax.random.PRNGKey(49)
    obs, state = wrapper.reset(key)

    n_steps = 2
    for i in range(n_steps):
        k_act, k_step = jax.random.split(jax.random.PRNGKey(200 + i))
        actions = _random_actions(env, k_act)
        obs, state, rewards, dones, info = wrapper.step(k_step, state, actions)

    saved_path = wrapper.save_trajectory("test_traj")
    assert os.path.isfile(saved_path)
    assert saved_path.endswith(".npz")

    loaded = TrajectoryWrapper.load_trajectory(saved_path)
    assert isinstance(loaded, dict)

    # We should have entries for step_0 (reset obs) and step_1..step_n (step obs/actions/rewards/dones)
    obs_keys = [k for k in loaded if k.startswith("step_0/obs/")]
    assert len(obs_keys) == cfg.num_agents

    # Step snapshots should have saved actions and rewards
    for step_idx in range(1, 1 + n_steps):
        action_keys = [k for k in loaded if k.startswith(f"step_{step_idx}/actions/")]
        reward_keys = [k for k in loaded if k.startswith(f"step_{step_idx}/rewards/")]
        assert len(action_keys) == cfg.num_agents
        assert len(reward_keys) == cfg.num_agents


# ------------------------------------------------------------------
# 50. test_trajectory_reset_clears
# ------------------------------------------------------------------

def test_trajectory_reset_clears():
    """Buffer resets when env.reset() is called through the wrapper."""
    wrapper, env, cfg = _make_wrapped_env()
    key = jax.random.PRNGKey(50)

    # First episode
    obs, state = wrapper.reset(key)
    for i in range(3):
        k_act, k_step = jax.random.split(jax.random.PRNGKey(300 + i))
        actions = _random_actions(env, k_act)
        obs, state, rewards, dones, info = wrapper.step(k_step, state, actions)

    assert len(wrapper.get_trajectory()) == 4  # 1 reset + 3 steps

    # Second episode -- buffer should be cleared
    obs, state = wrapper.reset(jax.random.PRNGKey(51))
    assert len(wrapper.get_trajectory()) == 1  # only the new reset snapshot

    # Take one step in the new episode
    k_act, k_step = jax.random.split(jax.random.PRNGKey(400))
    actions = _random_actions(env, k_act)
    obs, state, rewards, dones, info = wrapper.step(k_step, state, actions)
    assert len(wrapper.get_trajectory()) == 2  # 1 reset + 1 step
