"""Integration tests for GridCommEnv."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.env import GridCommEnv
from red_within_blue.types import EnvConfig, MAP_UNKNOWN


# Small config for fast tests
def _small_config(**overrides):
    defaults = dict(grid_width=8, grid_height=8, num_agents=3, max_steps=10,
                    obs_radius=2, comm_radius=5.0, msg_dim=4, wall_density=0.0)
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_env(**overrides):
    cfg = _small_config(**overrides)
    return GridCommEnv(cfg), cfg


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_reset_shapes():
    env, cfg = _make_env()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Obs dict has one entry per agent
    assert len(obs) == cfg.num_agents
    for a in env.agents:
        assert obs[a].shape == (env.obs_dim,)

    # Agent state shapes
    assert state.agent_state.positions.shape == (cfg.num_agents, 2)
    assert state.agent_state.local_map.shape == (cfg.num_agents, cfg.grid_height, cfg.grid_width)
    assert state.agent_state.messages_in.shape == (cfg.num_agents, env.total_msg_dim)

    # Global state
    assert state.global_state.grid.terrain.shape == (cfg.grid_height, cfg.grid_width)
    assert state.global_state.graph.adjacency.shape == (cfg.num_agents, cfg.num_agents)


def test_step_shapes():
    env, cfg = _make_env()
    key = jax.random.PRNGKey(1)
    obs, state = env.reset(key)

    # Random actions
    actions = {a: jnp.int32(1) for a in env.agents}  # all move UP
    key2 = jax.random.PRNGKey(2)
    obs2, state2, rewards, dones, info = env.step_env(key2, state, actions)

    assert len(obs2) == cfg.num_agents
    assert len(rewards) == cfg.num_agents
    for a in env.agents:
        assert obs2[a].shape == (env.obs_dim,)
        assert rewards[a].shape == ()
    assert "__all__" in dones
    assert "adjacency" in info
    assert "collisions" in info


def test_ctde_boundary():
    """get_obs must contain no global info; get_global_state must contain full truth."""
    env, cfg = _make_env()
    key = jax.random.PRNGKey(3)
    obs, state = env.reset(key)

    # Obs should NOT contain all positions (only own position normalized)
    for a in env.agents:
        obs_vec = obs[a]
        # The obs should be finite-dimensional and not contain the full terrain
        assert obs_vec.shape[0] == env.obs_dim
        assert obs_vec.shape[0] < cfg.grid_height * cfg.grid_width  # obs is smaller than full grid

    # Global state should contain terrain + all positions
    gs = env.get_global_state(state)
    expected_size = (cfg.num_agents * 2 + cfg.grid_height * cfg.grid_width * 2 +
                     cfg.num_agents ** 2 + 1)
    assert gs.shape == (expected_size,)


def test_jit_compiles():
    env, cfg = _make_env()
    key = jax.random.PRNGKey(4)

    # JIT reset
    jit_reset = jax.jit(env.reset)
    obs, state = jit_reset(key)

    # JIT step
    actions = {a: jnp.int32(0) for a in env.agents}
    jit_step = jax.jit(env.step_env)
    obs2, state2, rewards, dones, info = jit_step(jax.random.PRNGKey(5), state, actions)

    # Should not crash
    assert obs2[env.agents[0]].shape == (env.obs_dim,)


def test_vmap_batched():
    env, cfg = _make_env()
    batch_size = 4

    keys = jax.random.split(jax.random.PRNGKey(6), batch_size)

    # vmap reset
    v_reset = jax.vmap(env.reset)
    obs_batch, state_batch = v_reset(keys)

    # Check batched shapes
    assert state_batch.agent_state.positions.shape == (batch_size, cfg.num_agents, 2)

    # vmap step
    actions_batch = {a: jnp.zeros(batch_size, dtype=jnp.int32) for a in env.agents}
    step_keys = jax.random.split(jax.random.PRNGKey(7), batch_size)
    v_step = jax.vmap(env.step_env)
    obs2, state2, rewards, dones, info = v_step(step_keys, state_batch, actions_batch)

    assert obs2[env.agents[0]].shape == (batch_size, env.obs_dim)


def test_determinism():
    env, cfg = _make_env()
    key = jax.random.PRNGKey(8)

    obs1, state1 = env.reset(key)
    obs2, state2 = env.reset(key)

    for a in env.agents:
        assert jnp.array_equal(obs1[a], obs2[a])
    assert jnp.array_equal(state1.agent_state.positions, state2.agent_state.positions)

    # Same actions → same next state
    actions = {a: jnp.int32(2) for a in env.agents}
    k = jax.random.PRNGKey(9)
    _, s1, r1, _, _ = env.step_env(k, state1, actions)
    _, s2, r2, _, _ = env.step_env(k, state2, actions)
    assert jnp.array_equal(s1.agent_state.positions, s2.agent_state.positions)


def test_termination():
    env, cfg = _make_env(max_steps=3)
    key = jax.random.PRNGKey(10)
    obs, state = env.reset(key)
    actions = {a: jnp.int32(0) for a in env.agents}

    for i in range(3):
        k = jax.random.PRNGKey(100 + i)
        obs, state, rewards, dones, info = env.step_env(k, state, actions)

    assert bool(dones["__all__"])


def test_reward_fn_none():
    env, cfg = _make_env()
    key = jax.random.PRNGKey(11)
    obs, state = env.reset(key)
    actions = {a: jnp.int32(1) for a in env.agents}
    _, _, rewards, _, _ = env.step_env(jax.random.PRNGKey(12), state, actions)

    for a in env.agents:
        assert float(rewards[a]) == 0.0


def test_reward_fn_custom():
    def my_reward(new_state, prev_state, info):
        agents = [f"agent_{i}" for i in range(new_state.agent_state.positions.shape[0])]
        return {a: jnp.float32(1.0) for a in agents}

    cfg = _small_config()
    env = GridCommEnv(cfg, reward_fn=my_reward)
    key = jax.random.PRNGKey(13)
    obs, state = env.reset(key)
    actions = {a: jnp.int32(1) for a in env.agents}
    _, _, rewards, _, _ = env.step_env(jax.random.PRNGKey(14), state, actions)

    for a in env.agents:
        assert float(rewards[a]) == 1.0


def test_reward_fn_compose():
    """Test that multiple reward functions can be composed."""
    def reward_a(new_state, prev_state, info):
        agents = [f"agent_{i}" for i in range(new_state.agent_state.positions.shape[0])]
        return {a: jnp.float32(2.0) for a in agents}

    def reward_b(new_state, prev_state, info):
        agents = [f"agent_{i}" for i in range(new_state.agent_state.positions.shape[0])]
        return {a: jnp.float32(3.0) for a in agents}

    def composed(new_state, prev_state, info):
        r_a = reward_a(new_state, prev_state, info)
        r_b = reward_b(new_state, prev_state, info)
        agents = list(r_a.keys())
        return {a: r_a[a] + r_b[a] for a in agents}

    cfg = _small_config()
    env = GridCommEnv(cfg, reward_fn=composed)
    key = jax.random.PRNGKey(15)
    obs, state = env.reset(key)
    actions = {a: jnp.int32(1) for a in env.agents}
    _, _, rewards, _, _ = env.step_env(jax.random.PRNGKey(16), state, actions)

    for a in env.agents:
        assert float(rewards[a]) == 5.0


# ------------------------------------------------------------------
# obs_array / step_array tests
# ------------------------------------------------------------------

@pytest.mark.parametrize("num_agents", [1, 2, 4])
def test_obs_array_shape(num_agents):
    env, cfg = _make_env(num_agents=num_agents)
    key = jax.random.PRNGKey(20)
    _, state = env.reset(key)

    obs = env.obs_array(state)
    assert obs.shape == (num_agents, env.obs_dim)


@pytest.mark.parametrize("num_agents", [1, 2, 4])
def test_obs_array_matches_get_obs(num_agents):
    env, cfg = _make_env(num_agents=num_agents)
    key = jax.random.PRNGKey(21)
    _, state = env.reset(key)

    obs_arr = env.obs_array(state)
    obs_dict = env.get_obs(state)
    expected = jnp.stack([obs_dict[a] for a in env.agents])

    assert jnp.allclose(obs_arr, expected), "obs_array must match stacked get_obs"


def test_step_array_shapes():
    env, cfg = _make_env()
    key = jax.random.PRNGKey(22)
    _, state = env.reset(key)

    action_array = jnp.array([1, 2, 0], dtype=jnp.int32)  # 3 agents
    key2 = jax.random.PRNGKey(23)
    obs, new_state, rewards, done, info = env.step_array(key2, state, action_array)

    assert obs.shape == (cfg.num_agents, env.obs_dim)
    assert rewards.shape == (cfg.num_agents,)
    assert done.shape == ()
    assert new_state.agent_state.positions.shape == (cfg.num_agents, 2)


def test_step_array_matches_step_env():
    env, cfg = _make_env()
    key = jax.random.PRNGKey(24)
    _, state = env.reset(key)

    actions_int = jnp.array([1, 2, 0], dtype=jnp.int32)
    actions_dict = {env.agents[i]: actions_int[i] for i in range(cfg.num_agents)}

    key2 = jax.random.PRNGKey(25)
    obs_arr, state_arr, rewards_arr, done_arr, _ = env.step_array(key2, state, actions_int)
    obs_dict, state_dict, rewards_dict, dones_dict, _ = env.step_env(key2, state, actions_dict)

    # States must be identical
    assert jnp.array_equal(state_arr.agent_state.positions, state_dict.agent_state.positions)
    assert jnp.array_equal(state_arr.global_state.step, state_dict.global_state.step)
    assert jnp.array_equal(state_arr.agent_state.local_scan, state_dict.agent_state.local_scan)

    # Rewards must match
    expected_rewards = jnp.stack([rewards_dict[a] for a in env.agents])
    assert jnp.allclose(rewards_arr, expected_rewards)

    # Done must match
    assert jnp.array_equal(done_arr, dones_dict["__all__"])

    # Obs must match
    expected_obs = jnp.stack([obs_dict[a] for a in env.agents])
    assert jnp.allclose(obs_arr, expected_obs)
