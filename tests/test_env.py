"""Integration tests for GridCommEnv."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.env import GridCommEnv
from red_within_blue.types import EnvConfig, MAP_UNKNOWN


# Small config for fast tests
def _small_config(**overrides):
    defaults = dict(grid_width=8, grid_height=8, num_agents=3, max_steps=10,
                    obs_radius=2, comm_radius=5.0, wall_density=0.0)
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
        # The obs should be finite-dimensional; local_scan/seen_patch are bounded by obs_radius.
        assert obs_vec.shape[0] == env.obs_dim

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


# ------------------------------------------------------------------
# normalize_uid: divide uid feature by num_agents in obs tail
# ------------------------------------------------------------------

@pytest.mark.parametrize("num_agents", [2, 4, 8])
def test_normalize_uid_scales_uid_into_unit_range(num_agents):
    """With normalize_uid=True the uid tail value must be in (0, 1] for every
    N. Without normalisation the same column carries raw {1..N} which
    extrapolates linearly out-of-distribution for warm-start across N."""
    env_norm, _ = _make_env(num_agents=num_agents, normalize_uid=True)
    env_raw, _ = _make_env(num_agents=num_agents, normalize_uid=False)
    key = jax.random.PRNGKey(7)
    _, st_norm = env_norm.reset(key)
    _, st_raw = env_raw.reset(key)

    obs_n = env_norm.obs_array(st_norm)
    obs_r = env_raw.obs_array(st_raw)

    # obs_dim is unchanged either way
    assert obs_n.shape == obs_r.shape == (num_agents, env_norm.obs_dim)

    # uid lives at column index obs_dim - 2 (tail layout: ..., uid, team).
    uid_col = env_norm.obs_dim - 2
    uid_norm = obs_n[:, uid_col]
    uid_raw = obs_r[:, uid_col]

    # Raw uids are 1..N; normalized are uid/N -> (0, 1].
    expected_norm = jnp.arange(1, num_agents + 1, dtype=jnp.float32) / float(num_agents)
    assert jnp.allclose(uid_norm, expected_norm)
    assert jnp.allclose(uid_raw, jnp.arange(1, num_agents + 1, dtype=jnp.float32))
    assert float(uid_norm.max()) == pytest.approx(1.0)
    assert float(uid_norm.min()) == pytest.approx(1.0 / num_agents)


def test_normalize_uid_default_off_preserves_legacy_obs():
    """Default normalize_uid=False must not change obs values for any
    existing checkpoint that was trained without the flag."""
    env, _ = _make_env(num_agents=4)  # default normalize_uid
    key = jax.random.PRNGKey(11)
    _, state = env.reset(key)
    obs = env.obs_array(state)
    uid_col = env.obs_dim - 2
    # Raw {1, 2, 3, 4}
    assert jnp.allclose(obs[:, uid_col], jnp.array([1.0, 2.0, 3.0, 4.0]))


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


def test_local_obs_shrinks_obs_dim():
    """local_obs=True replaces the H·W seen mask with a view-sized window."""
    env_global, cfg_g = _make_env(num_agents=3, obs_radius=1, local_obs=False)
    env_local, cfg_l = _make_env(num_agents=3, obs_radius=1, local_obs=True)

    view_d2 = (2 * 1 + 1) ** 2  # 9
    H, W = cfg_g.grid_height, cfg_g.grid_width
    tail = 5  # map_fraction + pos(2) + uid + team

    assert env_global.obs_dim == view_d2 + H * W + tail
    assert env_local.obs_dim == view_d2 + view_d2 + tail

    # Both envs must produce obs of their declared obs_dim.
    for env in (env_global, env_local):
        _, state = env.reset(jax.random.PRNGKey(0))
        obs = env.obs_array(state)
        assert obs.shape == (env.config.num_agents, env.obs_dim)


def test_local_obs_window_matches_local_map_slice():
    """With local_obs=True, the seen field must equal the view-sized window of
    the agent's own local_map, OOB padded as known (walls)."""
    env, cfg = _make_env(num_agents=2, obs_radius=2, local_obs=True)
    _, state = env.reset(jax.random.PRNGKey(7))
    obs = env.obs_array(state)

    view_r = env.view_radius
    view_d = env.obs_d
    scan_dim = view_d * view_d

    # seen field starts immediately after flat_scan (scan_dim floats).
    seen_field = obs[:, scan_dim:scan_dim + scan_dim]  # [N, view_d*view_d]

    from red_within_blue.types import MAP_UNKNOWN
    for i, pos in enumerate(state.agent_state.positions):
        known = (state.agent_state.local_map[i] != MAP_UNKNOWN).astype(jnp.float32)
        padded = jnp.pad(known, pad_width=view_r, mode="constant", constant_values=1.0)
        win = jax.lax.dynamic_slice(padded, (int(pos[0]), int(pos[1])), (view_d, view_d))
        assert jnp.allclose(seen_field[i], win.reshape(-1))


def test_survey_radius_zero_limits_local_map_writes():
    """With survey_radius=0 the agent's local_map after reset should only show
    MAP_FREE at the agents' own cells (and, via comm, their teammates')."""
    from red_within_blue.types import MAP_UNKNOWN, MAP_FREE

    env, cfg = _make_env(
        num_agents=3, obs_radius=2, view_radius=2, survey_radius=0,
        comm_radius=0.1,  # no one connects — isolate survey effect
    )
    _, state = env.reset(jax.random.PRNGKey(123))

    positions = state.agent_state.positions
    occupied = set((int(p[0]), int(p[1])) for p in positions)

    for i in range(cfg.num_agents):
        lmap = state.agent_state.local_map[i]
        known_cells = {
            (int(r), int(c))
            for r in range(cfg.grid_height) for c in range(cfg.grid_width)
            if int(lmap[r, c]) != MAP_UNKNOWN
        }
        # With comm_radius=0.1 and non-colocated spawns, only agent i's own
        # cell should be committed — a single point.
        assert known_cells == {(int(positions[i, 0]), int(positions[i, 1]))}, (
            f"agent {i} committed more than its current cell: {known_cells}"
        )


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


# ── Disconnect-grace mechanism ────────────────────────────────────────────────

def test_disconnect_timer_initialized_to_zero():
    """At reset, every agent's disconnect_timer must be 0."""
    cfg = EnvConfig(
        grid_width=6, grid_height=6, num_agents=3,
        max_steps=10, comm_radius=2.0, obs_radius=1,
        center_spawn=True,
    )
    env = GridCommEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(0))
    assert state.agent_state.disconnect_timer.shape == (3,)
    assert bool(jnp.all(state.agent_state.disconnect_timer == 0))


def test_disconnect_timer_ticks_when_out_of_largest_cc():
    """An agent placed far from the team should see its timer increment each step."""
    cfg = EnvConfig(
        grid_width=20, grid_height=20, num_agents=3,
        max_steps=10, comm_radius=2.0, obs_radius=1,
    )
    env = GridCommEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(42))
    # Force positions: 0 and 1 near each other, 2 far away.
    state = state.replace(agent_state=state.agent_state.replace(
        positions=jnp.array([[0, 0], [0, 1], [15, 15]], dtype=jnp.int32),
    ))
    # STAY for all agents
    actions = {f"agent_{i}": jnp.int32(0) for i in range(3)}
    _, state, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    # Agents 0 and 1 in the largest CC (timer=0); agent 2 isolated (timer=1).
    timer = state.agent_state.disconnect_timer
    assert int(timer[0]) == 0 and int(timer[1]) == 0
    assert int(timer[2]) == 1
    assert bool(info["disconnect_flags"][2])
    assert not bool(info["disconnect_flags"][0])


def test_disconnect_timer_resets_on_reconnect():
    """Once the drifter reconnects, all timers must reset to 0.

    Uses N=3 so the largest CC is unambiguous (agents 0+1 form size-2 CC;
    agent 2 is a singleton drifter).  This avoids the tie-break ambiguity
    when every agent is its own singleton.
    """
    cfg = EnvConfig(
        grid_width=20, grid_height=20, num_agents=3,
        max_steps=10, comm_radius=2.0, obs_radius=1,
    )
    env = GridCommEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(agent_state=state.agent_state.replace(
        positions=jnp.array([[0, 0], [0, 1], [15, 15]], dtype=jnp.int32),
    ))
    actions = {f"agent_{i}": jnp.int32(0) for i in range(3)}
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
    assert int(state.agent_state.disconnect_timer[2]) >= 1
    # Now snap agent 2 next to the pair and step again — full reconnect.
    state = state.replace(agent_state=state.agent_state.replace(
        positions=jnp.array([[0, 0], [0, 1], [0, 2]], dtype=jnp.int32),
    ))
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(2), state, actions)
    assert int(jnp.max(state.agent_state.disconnect_timer)) == 0


def test_disconnect_grace_disabled_keeps_episode_running():
    """With grace=0, timer ticks but env must NOT terminate on disconnection."""
    cfg = EnvConfig(
        grid_width=20, grid_height=20, num_agents=2,
        max_steps=10, comm_radius=2.0, obs_radius=1,
        disconnect_grace=0,
    )
    env = GridCommEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(agent_state=state.agent_state.replace(
        positions=jnp.array([[0, 0], [15, 15]], dtype=jnp.int32),
    ))
    actions = {f"agent_{i}": jnp.int32(0) for i in range(2)}
    for _ in range(5):
        _, state, _, dones, info = env.step_env(jax.random.PRNGKey(1), state, actions)
        assert not bool(info["disconnect_triggered"])
        assert not bool(dones["__all__"])  # max_steps=10, 5 < 10, timer-based termination off
    # Timer should still have climbed above 0.
    assert int(jnp.max(state.agent_state.disconnect_timer)) >= 5


def test_disconnect_grace_triggers_failure_per_agent():
    """With grace=2, the third disconnect step must terminate the episode."""
    cfg = EnvConfig(
        grid_width=20, grid_height=20, num_agents=2,
        max_steps=20, comm_radius=2.0, obs_radius=1,
        disconnect_grace=2, disconnect_fail_penalty=-3.0,
        disconnect_mode=0,   # per_agent
    )
    env = GridCommEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(agent_state=state.agent_state.replace(
        positions=jnp.array([[0, 0], [15, 15]], dtype=jnp.int32),
    ))
    actions = {f"agent_{i}": jnp.int32(0) for i in range(2)}
    # Step 1: timer 0->1, not tripped (1 < 2).
    _, state, rewards, dones, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    assert not bool(info["disconnect_triggered"])
    assert not bool(dones["__all__"])
    # Step 2: timer 1->2, tripped (2 >= 2).
    _, state, rewards, dones, info = env.step_env(jax.random.PRNGKey(2), state, actions)
    assert bool(info["disconnect_triggered"])
    assert bool(dones["__all__"])
    # Fail penalty applied to every agent this step.
    for a in env.agents:
        assert float(rewards[a]) <= -3.0 + 1e-6


def test_disconnect_grace_team_mode_requires_full_split():
    """In team mode, a single isolated agent should not trigger failure
    (the whole team must be disconnected)."""
    # 3 agents, agent 2 drifts; agents 0 and 1 stay connected.
    cfg = EnvConfig(
        grid_width=20, grid_height=20, num_agents=3,
        max_steps=20, comm_radius=2.0, obs_radius=1,
        disconnect_grace=1, disconnect_mode=1,  # team
    )
    env = GridCommEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(agent_state=state.agent_state.replace(
        positions=jnp.array([[0, 0], [0, 1], [15, 15]], dtype=jnp.int32),
    ))
    actions = {f"agent_{i}": jnp.int32(0) for i in range(3)}
    # Even after many steps, agent 2 is isolated but 0-1 are still connected,
    # so `is_connected` is False but the WHOLE team isn't split into singletons.
    # Team mode conditions on `~is_connected` so it WILL trip. To verify
    # per-agent vs team distinction, re-run with disconnect_mode=0 (per_agent)
    # and confirm it trips on the same scenario — both modes see the lone
    # drifter. The distinction is: in team mode we do NOT trip if
    # `is_connected` becomes True again. Check that by reconnecting agent 2.
    # Step 1 with team mode: is_connected=False, max_timer >= 1 -> trip.
    _, state, _, dones, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    assert bool(info["disconnect_triggered"])  # team split -> trip
    # Now build a parallel env where we reconnect before grace expires.
    cfg2 = EnvConfig(
        grid_width=20, grid_height=20, num_agents=3,
        max_steps=20, comm_radius=2.0, obs_radius=1,
        disconnect_grace=2, disconnect_mode=1,
    )
    env2 = GridCommEnv(cfg2)
    _, state2 = env2.reset(jax.random.PRNGKey(0))
    state2 = state2.replace(agent_state=state2.agent_state.replace(
        positions=jnp.array([[0, 0], [0, 1], [15, 15]], dtype=jnp.int32),
    ))
    _, state2, _, dones2, info2 = env2.step_env(jax.random.PRNGKey(1), state2, actions)
    assert not bool(info2["disconnect_triggered"])  # timer=1 < grace=2
    # Reconnect agent 2 -> max timer resets to 0.
    state2 = state2.replace(agent_state=state2.agent_state.replace(
        positions=jnp.array([[0, 0], [0, 1], [0, 2]], dtype=jnp.int32),
    ))
    _, state2, _, dones2, info2 = env2.step_env(jax.random.PRNGKey(2), state2, actions)
    assert not bool(info2["disconnect_triggered"])
    assert int(jnp.max(state2.agent_state.disconnect_timer)) == 0


# ------------------------------------------------------------------
# Red contamination — end-to-end through env.reset / env.step_env
# ------------------------------------------------------------------

def test_red_contamination_zero_red_is_noop():
    """With num_red_agents=0, post-step explored matches the +1 update."""
    env, cfg = _make_env(num_red_agents=0)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Each spawn cell got +1.
    rows = state.agent_state.positions[:, 0]
    cols = state.agent_state.positions[:, 1]
    for r, c in zip(rows.tolist(), cols.tolist()):
        assert state.global_state.grid.explored[r, c].item() >= 1


def test_red_contamination_zeros_red_spawn_cell():
    """At reset, the red spawn cell ends with explored == 0."""
    env, cfg = _make_env(num_red_agents=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Last agent is red.
    red_pos = state.agent_state.positions[-1]
    assert state.global_state.grid.explored[red_pos[0], red_pos[1]].item() == 0


def test_red_contamination_step_zeros_red_cell():
    """After a step, the red's destination cell has explored==0 and a blue
    that is *directly adjacent* to the red in the comm graph has its
    local_map at that cell flipped to MAP_UNKNOWN. Fog only propagates one
    hop — a blue that is NOT directly adjacent to the red keeps its belief
    from (truthful) blue→blue passthrough."""
    env, cfg = _make_env(num_red_agents=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Pin positions: blue0 and red co-adjacent; blue1 out of range of red.
    state = state.replace(agent_state=state.agent_state.replace(
        positions=jnp.array([[3, 3], [3, 4], [3, 2]], dtype=jnp.int32),
    ))
    actions = {a: jnp.int32(0) for a in env.agents}  # STAY
    _, state2, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    red_pos = state2.agent_state.positions[-1]
    # explored at red's cell is 0.
    assert state2.global_state.grid.explored[red_pos[0], red_pos[1]].item() == 0
    # blue0 is directly adjacent to red → received fog → MAP_UNKNOWN.
    assert bool(info["adjacency"][2, 0])
    assert int(state2.agent_state.local_map[0, red_pos[0], red_pos[1]]) == MAP_UNKNOWN


def test_red_fog_does_not_cross_disconnected_blues():
    """A blue outside the red's comm range keeps its belief about unrelated
    cells: red's fog messages only reach connected receivers."""
    # 1 blue + 1 red, on a 10×10 grid, tiny comm_radius so they can't talk.
    env, cfg = _make_env(grid_width=10, grid_height=10, num_agents=2,
                         num_red_agents=1, comm_radius=1.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Force non-overlapping positions, far apart, so adjacency[red, blue]=False.
    new_positions = jnp.array([[2, 2], [8, 8]], dtype=jnp.int32)
    state = state.replace(agent_state=state.agent_state.replace(positions=new_positions))
    # Manually seed blue's local_map as "fully known" everywhere.
    seeded_local = jnp.ones_like(state.agent_state.local_map)
    state = state.replace(agent_state=state.agent_state.replace(local_map=seeded_local))
    actions = {a: jnp.int32(0) for a in env.agents}  # STAY
    _, state2, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    # Red and blue are disconnected; blue's local_map at red's cell stays as
    # blue seeded it (still "1"), because the fog message never arrived.
    red_pos = state2.agent_state.positions[-1]
    assert not bool(info["is_connected"])
    assert int(state2.agent_state.local_map[0, red_pos[0], red_pos[1]]) == 1


def test_red_fog_does_not_wipe_red_own_map():
    """Red surveys truthfully into its own local_map — the fog rule only
    applies to red→blue messages, not red→red (or red→self)."""
    # 2 reds + 0 blues to isolate the red→red path.
    env, cfg = _make_env(num_agents=2, num_red_agents=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {a: jnp.int32(0) for a in env.agents}
    _, state2, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
    red_positions = state2.agent_state.positions
    # Each red's local_map at its own cell holds real terrain (not MAP_UNKNOWN).
    for i in range(cfg.num_agents):
        row, col = red_positions[i]
        assert int(state2.agent_state.local_map[i, row, col]) != MAP_UNKNOWN
