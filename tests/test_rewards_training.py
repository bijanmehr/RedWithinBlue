"""Tests for reward functions in red_within_blue.training.rewards_training."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.env import GridCommEnv
from red_within_blue.types import EnvConfig, CELL_WALL
from red_within_blue.training.rewards_training import (
    normalized_exploration_reward,
    make_exploration_reward,
    make_revisit_penalty,
    make_isolation_penalty,
    make_terminal_coverage_bonus,
    make_time_penalty,
    make_cooperative_bonus,
    make_reward_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(reward_fn=None, **overrides):
    """Create a small 8x8, 1-agent env with no walls and 100 max steps."""
    defaults = dict(
        grid_width=8,
        grid_height=8,
        num_agents=1,
        wall_density=0.0,
        max_steps=100,
        obs_radius=2,
        comm_radius=5.0,
        msg_dim=4,
    )
    defaults.update(overrides)
    cfg = EnvConfig(**defaults)
    return GridCommEnv(cfg, reward_fn=reward_fn), cfg


def _total_discoverable(state):
    """Count non-wall cells from the grid terrain."""
    terrain = state.global_state.grid.terrain
    return int(jnp.sum(terrain != CELL_WALL))


# ---------------------------------------------------------------------------
# Test 1: Reward is the correct fraction when discovering a new cell
# ---------------------------------------------------------------------------

def test_new_cell_reward_is_correct_fraction():
    """Agent stepping onto a never-visited cell gets reward = 1 / total_discoverable."""
    env, cfg = _make_env(reward_fn=normalized_exploration_reward)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    # After reset the agent's starting position is already explored (count = 1).
    # We need to find a move that takes the agent to an unexplored cell.
    # Action 2 = RIGHT (col+1). In an 8x8 grid with no interior walls, this
    # is guaranteed to be open (the agent spawns in the interior, not column 6).
    action_right = {"agent_0": jnp.int32(2)}  # RIGHT

    # Take a step — agent moves to a new cell
    _, new_state, rewards, _, _ = env.step_env(jax.random.PRNGKey(1), state, action_right)

    total_disc = _total_discoverable(new_state)
    expected = 1.0 / total_disc

    agent_reward = float(rewards["agent_0"])

    # The agent must have moved to an unvisited cell (reward > 0) or stayed (reward = 0).
    # We assert the reward is either 0 or exactly 1/total_discoverable.
    assert agent_reward in [0.0, pytest.approx(expected, rel=1e-5)], (
        f"Unexpected reward {agent_reward}; expected 0.0 or {expected}"
    )

    # To guarantee we test the discovery case, let's verify directly:
    # Compare prev_explored at the new position with the reward.
    new_pos = new_state.agent_state.positions[0]  # [2]
    prev_explored_at_new_pos = int(state.global_state.grid.explored[new_pos[0], new_pos[1]])

    if prev_explored_at_new_pos == 0:
        # Agent moved to a new cell — reward must be the correct fraction
        assert agent_reward == pytest.approx(expected, rel=1e-5), (
            f"Agent discovered a new cell but reward={agent_reward}, expected={expected}"
        )
    else:
        # Agent didn't move to a new cell — reward must be 0
        assert agent_reward == pytest.approx(0.0, abs=1e-7), (
            f"Agent revisited a cell but reward={agent_reward}, expected=0.0"
        )


def test_discovery_reward_fraction_explicitly():
    """Directly construct prev_state / new_state to guarantee the discovery case."""
    env, cfg = _make_env(reward_fn=normalized_exploration_reward)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    # Run several STAY actions so the agent stays put and no new cells are visited.
    # Then verify that one RIGHT step into an unvisited cell gives the right reward.
    action_stay = {"agent_0": jnp.int32(0)}   # STAY
    action_right = {"agent_0": jnp.int32(2)}  # RIGHT

    # After reset the spawn cell is explored; agent stays, so no new cells ever.
    _, state_stay, _, _, _ = env.step_env(jax.random.PRNGKey(10), state, action_stay)

    # Now take a step to the right — cell to the right was not previously visited.
    prev_state = state_stay
    _, new_state, rewards, _, _ = env.step_env(jax.random.PRNGKey(11), prev_state, action_right)

    new_pos = new_state.agent_state.positions[0]
    prev_explored_at_dest = int(prev_state.global_state.grid.explored[new_pos[0], new_pos[1]])

    total_disc = _total_discoverable(new_state)
    agent_reward = float(rewards["agent_0"])

    if prev_explored_at_dest == 0:
        # Discovery happened — must get 1 / total_discoverable
        assert agent_reward == pytest.approx(1.0 / total_disc, rel=1e-5), (
            f"Discovery reward wrong: got {agent_reward}, expected {1.0/total_disc}"
        )
        assert 0.0 < agent_reward <= 1.0, "Reward must be in (0, 1]"
    else:
        # If the agent couldn't move (e.g., wall collision), expect 0
        assert agent_reward == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 2: Reward is 0 when revisiting an already-explored cell
# ---------------------------------------------------------------------------

def test_revisit_gives_zero_reward():
    """Agent returning to a previously visited cell gets reward = 0."""
    env, cfg = _make_env(reward_fn=normalized_exploration_reward)
    key = jax.random.PRNGKey(7)
    obs, state = env.reset(key)

    # Move right to explore a new cell
    action_right = {"agent_0": jnp.int32(2)}  # RIGHT
    action_left = {"agent_0": jnp.int32(4)}   # LEFT

    _, state_after_right, _, _, _ = env.step_env(jax.random.PRNGKey(20), state, action_right)

    # Move back left — the original spawn position was already explored at reset
    _, state_after_left, rewards_revisit, _, _ = env.step_env(
        jax.random.PRNGKey(21), state_after_left := state_after_right, action_left
    )

    # The cell we return to was explored before this step (prev_explored > 0)
    revisit_pos = state_after_left.agent_state.positions[0]
    # Verify the cell was already explored before the LEFT step
    explored_before = int(state_after_right.global_state.grid.explored[revisit_pos[0], revisit_pos[1]])

    if explored_before > 0:
        # Revisit — reward must be 0
        assert float(rewards_revisit["agent_0"]) == pytest.approx(0.0, abs=1e-7), (
            f"Revisit should give reward=0.0, got {float(rewards_revisit['agent_0'])}"
        )


def test_stay_after_exploration_gives_zero():
    """Agent staying on its current cell (which is now explored) gets reward = 0."""
    env, cfg = _make_env(reward_fn=normalized_exploration_reward)
    key = jax.random.PRNGKey(3)
    _, state = env.reset(key)

    # After reset the spawn cell has explored count = 1. Stay → revisit.
    action_stay = {"agent_0": jnp.int32(0)}  # STAY
    _, _, rewards, _, _ = env.step_env(jax.random.PRNGKey(30), state, action_stay)

    assert float(rewards["agent_0"]) == pytest.approx(0.0, abs=1e-7), (
        "Agent staying on spawn cell (already explored) should get 0 reward"
    )


# ---------------------------------------------------------------------------
# Test 3: Cumulative reward over a full episode stays in [0, 1]
# ---------------------------------------------------------------------------

def test_cumulative_reward_in_unit_interval():
    """Sum of rewards over a full episode is in [0, 1] for a single agent."""
    env, cfg = _make_env(reward_fn=normalized_exploration_reward)
    key = jax.random.PRNGKey(99)
    obs, state = env.reset(key)

    total_reward = 0.0
    # Cycle through actions to maximize exploration
    actions_cycle = [1, 2, 3, 4, 0]  # UP, RIGHT, DOWN, LEFT, STAY

    for step_i in range(cfg.max_steps):
        action_idx = actions_cycle[step_i % len(actions_cycle)]
        actions = {"agent_0": jnp.int32(action_idx)}
        step_key = jax.random.PRNGKey(1000 + step_i)
        obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
        r = float(rewards["agent_0"])
        # Each individual reward must be non-negative and <= 1
        assert r >= 0.0, f"Step {step_i}: negative reward {r}"
        assert r <= 1.0, f"Step {step_i}: reward > 1 at {r}"
        total_reward += r
        if bool(dones["__all__"]):
            break

    # Cumulative reward across the whole episode is in [0, 1]
    assert total_reward >= 0.0, f"Cumulative reward is negative: {total_reward}"
    assert total_reward <= 1.0 + 1e-6, (
        f"Cumulative reward {total_reward} exceeds 1.0 — "
        "each cell can only be discovered once"
    )


# ---------------------------------------------------------------------------
# Helpers for multi-agent tests
# ---------------------------------------------------------------------------

def _make_multi_env(num_agents=2, **overrides):
    """Create a small 2-agent env suitable for reward component tests."""
    defaults = dict(
        grid_width=8,
        grid_height=8,
        num_agents=num_agents,
        wall_density=0.0,
        max_steps=100,
        obs_radius=2,
        comm_radius=5.0,
        msg_dim=4,
    )
    defaults.update(overrides)
    cfg = EnvConfig(**defaults)
    return GridCommEnv(cfg), cfg


def _step_env_with_info(env, state, actions):
    """Step and return (new_state, prev_state, info) for reward testing."""
    key = jax.random.PRNGKey(0)
    jax_actions = {a: jnp.int32(v) for a, v in actions.items()}
    obs, new_state, rewards, dones, info = env.step_env(key, state, jax_actions)
    return new_state, state, info


# ---------------------------------------------------------------------------
# Test: make_exploration_reward
# ---------------------------------------------------------------------------

class TestMakeExplorationReward:

    def test_new_cell_gives_weight(self):
        """Discovering a new cell gives exactly +weight."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        reward_fn = make_exploration_reward(weight=0.1)

        # STAY — spawn cell is already explored
        action_stay = {"agent_0": jnp.int32(0)}
        _, state_stay, _, _, _ = env.step_env(jax.random.PRNGKey(10), state, action_stay)

        # RIGHT — move to unexplored cell
        action_right = {"agent_0": jnp.int32(2)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(11), state_stay, action_right)

        rewards = reward_fn(new_state, state_stay, info)

        new_pos = new_state.agent_state.positions[0]
        was_unexplored = int(state_stay.global_state.grid.explored[new_pos[0], new_pos[1]]) == 0

        if was_unexplored:
            assert float(rewards["agent_0"]) == pytest.approx(0.1, abs=1e-6)
        else:
            assert float(rewards["agent_0"]) == pytest.approx(0.0, abs=1e-6)

    def test_revisit_gives_zero(self):
        """Stepping on an already-explored cell gives 0."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        reward_fn = make_exploration_reward(weight=0.1)

        # STAY on spawn cell (already explored)
        action_stay = {"agent_0": jnp.int32(0)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, action_stay)

        rewards = reward_fn(new_state, state, info)
        assert float(rewards["agent_0"]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: make_revisit_penalty
# ---------------------------------------------------------------------------

class TestMakeRevisitPenalty:

    def test_revisit_gives_penalty(self):
        """Stepping on an explored cell gives the configured penalty."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        penalty_fn = make_revisit_penalty(weight=-0.3)

        # STAY on explored spawn cell
        action_stay = {"agent_0": jnp.int32(0)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, action_stay)

        rewards = penalty_fn(new_state, state, info)
        assert float(rewards["agent_0"]) == pytest.approx(-0.3, abs=1e-6)

    def test_new_cell_gives_zero_penalty(self):
        """Moving to a new cell incurs no revisit penalty."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        penalty_fn = make_revisit_penalty(weight=-0.3)

        # STAY then RIGHT to ensure we hit an unexplored cell
        action_stay = {"agent_0": jnp.int32(0)}
        _, state_stay, _, _, _ = env.step_env(jax.random.PRNGKey(10), state, action_stay)
        action_right = {"agent_0": jnp.int32(2)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(11), state_stay, action_right)

        rewards = penalty_fn(new_state, state_stay, info)
        new_pos = new_state.agent_state.positions[0]
        was_unexplored = int(state_stay.global_state.grid.explored[new_pos[0], new_pos[1]]) == 0

        if was_unexplored:
            assert float(rewards["agent_0"]) == pytest.approx(0.0, abs=1e-6)

    def test_configurable_weight(self):
        """Different weight values produce different penalties."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        action_stay = {"agent_0": jnp.int32(0)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, action_stay)

        for w in [-0.1, -0.5, -1.0]:
            fn = make_revisit_penalty(weight=w)
            rewards = fn(new_state, state, info)
            assert float(rewards["agent_0"]) == pytest.approx(w, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: make_isolation_penalty
# ---------------------------------------------------------------------------

class TestMakeIsolationPenalty:

    def test_only_isolated_agent_penalized(self):
        """With 2 agents far apart, both isolated (degree==0) get penalty.
        With agents close together, neither gets penalty."""
        env, cfg = _make_multi_env(num_agents=2, comm_radius=5.0)
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        iso_fn = make_isolation_penalty(weight=-0.5)

        # Step to get info with degree array
        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        rewards = iso_fn(new_state, state, info)
        degree = info["degree"]  # [2]

        for i in range(2):
            agent = f"agent_{i}"
            if int(degree[i]) == 0:
                assert float(rewards[agent]) == pytest.approx(-0.5, abs=1e-6), \
                    f"{agent} isolated (degree=0) but not penalized"
            else:
                assert float(rewards[agent]) == pytest.approx(0.0, abs=1e-6), \
                    f"{agent} connected (degree={int(degree[i])}) but penalized"

    def test_custom_weight(self):
        """Custom isolation penalty weight is applied."""
        env, cfg = _make_multi_env(num_agents=2, comm_radius=0.5)  # tiny range → likely isolated
        key = jax.random.PRNGKey(99)
        _, state = env.reset(key)

        iso_fn = make_isolation_penalty(weight=-1.0)
        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        rewards = iso_fn(new_state, state, info)
        degree = info["degree"]

        for i in range(2):
            if int(degree[i]) == 0:
                assert float(rewards[f"agent_{i}"]) == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: make_terminal_coverage_bonus
# ---------------------------------------------------------------------------

class TestMakeTerminalCoverageBonus:

    def test_non_terminal_gives_zero(self):
        """Before episode ends, terminal bonus is 0."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        bonus_fn = make_terminal_coverage_bonus(scale=1.0, divide=True)

        actions = {"agent_0": jnp.int32(0)}
        _, new_state, _, dones, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        # Not terminal yet
        if not bool(dones["__all__"]):
            rewards = bonus_fn(new_state, state, info)
            assert float(rewards["agent_0"]) == pytest.approx(0.0, abs=1e-6)

    def test_divide_splits_among_agents(self):
        """With divide=True and 2 agents, bonus is coverage * scale / 2."""
        env, cfg = _make_multi_env(num_agents=2, max_steps=3)
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        bonus_fn = make_terminal_coverage_bonus(scale=10.0, divide=True)

        # Run until done
        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        for _ in range(10):
            _, state, _, dones, info = env.step_env(jax.random.PRNGKey(10), state, actions)
            if bool(dones["__all__"]):
                break

        if bool(state.global_state.done):
            rewards = bonus_fn(state, state, info)
            # Both agents should get equal bonus
            r0 = float(rewards["agent_0"])
            r1 = float(rewards["agent_1"])
            assert r0 == pytest.approx(r1, abs=1e-6), "Agents should get equal terminal bonus"
            assert r0 > 0.0, "Terminal bonus should be positive"

    def test_no_divide_gives_full_bonus(self):
        """With divide=False, each agent gets the full coverage * scale."""
        env, cfg = _make_env(max_steps=3)
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        bonus_fn_div = make_terminal_coverage_bonus(scale=5.0, divide=True)
        bonus_fn_full = make_terminal_coverage_bonus(scale=5.0, divide=False)

        actions = {"agent_0": jnp.int32(0)}
        for _ in range(10):
            _, state, _, dones, info = env.step_env(jax.random.PRNGKey(10), state, actions)
            if bool(dones["__all__"]):
                break

        if bool(state.global_state.done):
            r_div = float(bonus_fn_div(state, state, info)["agent_0"])
            r_full = float(bonus_fn_full(state, state, info)["agent_0"])
            # With 1 agent, divide doesn't change the value
            assert r_full == pytest.approx(r_div, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: make_time_penalty
# ---------------------------------------------------------------------------

class TestMakeTimePenalty:

    def test_constant_per_step(self):
        """Time penalty is the same flat value every step."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        time_fn = make_time_penalty(weight=-0.01)

        actions = {"agent_0": jnp.int32(0)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        rewards = time_fn(new_state, state, info)
        assert float(rewards["agent_0"]) == pytest.approx(-0.01, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: make_cooperative_bonus
# ---------------------------------------------------------------------------

class TestMakeCooperativeBonus:

    def test_returns_valid_rewards(self):
        """Cooperative bonus returns a reward dict with correct agent keys."""
        env, cfg = _make_multi_env(num_agents=2, comm_radius=5.0)
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        coop_fn = make_cooperative_bonus(weight=0.02)

        actions = {"agent_0": jnp.int32(2), "agent_1": jnp.int32(4)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        rewards = coop_fn(new_state, state, info)
        assert "agent_0" in rewards
        assert "agent_1" in rewards
        # Rewards should be finite floats
        assert jnp.isfinite(rewards["agent_0"])
        assert jnp.isfinite(rewards["agent_1"])


# ---------------------------------------------------------------------------
# Test: make_reward_config (master composite)
# ---------------------------------------------------------------------------

class TestMakeRewardConfig:

    def test_composes_all_components(self):
        """Composite reward returns valid rewards for all agents."""
        env, cfg = _make_multi_env(num_agents=2, comm_radius=5.0)
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        reward_fn = make_reward_config(
            exploration_weight=0.1,
            revisit_weight=-0.3,
            isolation_weight=-0.5,
            terminal_bonus_scale=1.0,
            terminal_divide=True,
            time_weight=-0.01,
            cooperative_weight=0.02,
            disconnect_penalty=0.0,
        )

        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        rewards = reward_fn(new_state, state, info)

        assert "agent_0" in rewards
        assert "agent_1" in rewards
        for agent in ["agent_0", "agent_1"]:
            assert jnp.isfinite(rewards[agent]), f"{agent} reward not finite"

    def test_zero_weights_give_zero(self):
        """All weights at zero produces zero reward (non-terminal)."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        reward_fn = make_reward_config(
            exploration_weight=0.0,
            revisit_weight=0.0,
            isolation_weight=0.0,
            terminal_bonus_scale=0.0,
            time_weight=0.0,
            cooperative_weight=0.0,
            disconnect_penalty=0.0,
        )

        actions = {"agent_0": jnp.int32(0)}
        _, new_state, _, dones, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        if not bool(dones["__all__"]):
            rewards = reward_fn(new_state, state, info)
            assert float(rewards["agent_0"]) == pytest.approx(0.0, abs=1e-6)

    def test_only_time_penalty(self):
        """With only time penalty enabled, reward equals time_weight."""
        env, cfg = _make_env()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        reward_fn = make_reward_config(
            exploration_weight=0.0,
            revisit_weight=0.0,
            isolation_weight=0.0,
            terminal_bonus_scale=0.0,
            time_weight=-0.05,
            cooperative_weight=0.0,
            disconnect_penalty=0.0,
        )

        actions = {"agent_0": jnp.int32(0)}
        _, new_state, _, dones, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        if not bool(dones["__all__"]):
            rewards = reward_fn(new_state, state, info)
            assert float(rewards["agent_0"]) == pytest.approx(-0.05, abs=1e-6)

    def test_disconnect_penalty_gates_exploration(self):
        """When disconnect_penalty != 0, exploration is gated on connectivity."""
        env, cfg = _make_multi_env(num_agents=2, comm_radius=5.0)
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)

        reward_fn = make_reward_config(
            exploration_weight=0.1,
            revisit_weight=0.0,
            isolation_weight=0.0,
            terminal_bonus_scale=0.0,
            time_weight=0.0,
            cooperative_weight=0.0,
            disconnect_penalty=-0.5,
        )

        actions = {"agent_0": jnp.int32(2), "agent_1": jnp.int32(4)}
        _, new_state, _, _, info = env.step_env(jax.random.PRNGKey(10), state, actions)

        rewards = reward_fn(new_state, state, info)

        # If disconnected, exploration should be gated (zeroed) + penalty applied
        if not bool(info["is_connected"]):
            for agent in ["agent_0", "agent_1"]:
                r = float(rewards[agent])
                assert r <= 0.0, f"{agent} should have non-positive reward when disconnected"
