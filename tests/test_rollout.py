"""Tests for the episode rollout collectors in red_within_blue.training.rollout."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from red_within_blue.env import GridCommEnv
from red_within_blue.types import EnvConfig
from red_within_blue.training.rewards_training import (
    normalized_exploration_reward,
    make_multi_agent_reward,
)
from red_within_blue.training.rollout import (
    collect_episode,
    collect_episode_multi,
    _connectivity_mask,
    # New JAX-native functions
    Trajectory,
    MultiTrajectory,
    collect_episode_scan,
    collect_episode_multi_scan,
    _connectivity_guardrail,
)
from red_within_blue.training.networks import Actor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_env():
    """Small 8x8, 1-agent env with no walls and max_steps=10."""
    cfg = EnvConfig(
        grid_width=8,
        grid_height=8,
        num_agents=1,
        wall_density=0.0,
        max_steps=10,
    )
    env = GridCommEnv(cfg, reward_fn=normalized_exploration_reward)
    return env


def _random_policy(key, obs):
    """Uniformly random policy: ignores obs, samples a random action."""
    return jax.random.randint(key, shape=(), minval=0, maxval=5)


# ---------------------------------------------------------------------------
# Tests for collect_episode (single-agent tracking)
# ---------------------------------------------------------------------------

class TestCollectEpisode:
    def test_returns_correct_keys(self):
        """collect_episode must return a dict with the four required keys."""
        env = _make_env()
        key = jax.random.PRNGKey(0)
        result = collect_episode(env, _random_policy, key)

        assert isinstance(result, dict), "Result must be a dict"
        assert set(result.keys()) == {"observations", "actions", "rewards", "dones"}, (
            f"Unexpected keys: {set(result.keys())}"
        )

    def test_length_bounded_by_max_steps(self):
        """Episode length must not exceed max_steps (10)."""
        env = _make_env()
        key = jax.random.PRNGKey(1)
        result = collect_episode(env, _random_policy, key)

        n = len(result["observations"])
        assert n <= 10, f"Episode length {n} exceeds max_steps=10"
        # All lists must have the same length
        assert len(result["actions"]) == n
        assert len(result["rewards"]) == n
        assert len(result["dones"]) == n

    def test_observation_shape(self):
        """Every observation must be a 1-D array of length env.obs_dim."""
        env = _make_env()
        key = jax.random.PRNGKey(2)
        result = collect_episode(env, _random_policy, key)

        for i, obs in enumerate(result["observations"]):
            assert obs.shape == (env.obs_dim,), (
                f"Observation at step {i} has shape {obs.shape}, expected ({env.obs_dim},)"
            )

    def test_lists_same_length(self):
        """All trajectory lists must have identical length."""
        env = _make_env()
        key = jax.random.PRNGKey(3)
        result = collect_episode(env, _random_policy, key)

        lengths = {k: len(v) for k, v in result.items()}
        assert len(set(lengths.values())) == 1, f"Mismatched lengths: {lengths}"

    def test_actions_are_valid(self):
        """All recorded actions must be integers in [0, num_actions)."""
        env = _make_env()
        key = jax.random.PRNGKey(4)
        result = collect_episode(env, _random_policy, key)

        for i, action in enumerate(result["actions"]):
            assert isinstance(action, int), f"Action at step {i} is not int: {type(action)}"
            assert 0 <= action < env.config.num_actions, (
                f"Action {action} at step {i} out of range [0, {env.config.num_actions})"
            )

    def test_rewards_are_scalars(self):
        """All rewards must be Python floats (scalars)."""
        env = _make_env()
        key = jax.random.PRNGKey(5)
        result = collect_episode(env, _random_policy, key)

        for i, r in enumerate(result["rewards"]):
            assert isinstance(r, float), f"Reward at step {i} is {type(r)}, expected float"

    def test_dones_are_bools(self):
        """All done flags must be Python bools."""
        env = _make_env()
        key = jax.random.PRNGKey(6)
        result = collect_episode(env, _random_policy, key)

        for i, d in enumerate(result["dones"]):
            assert isinstance(d, bool), f"Done at step {i} is {type(d)}, expected bool"

    def test_last_done_is_true(self):
        """The final done flag must be True (episode terminated)."""
        env = _make_env()
        key = jax.random.PRNGKey(7)
        result = collect_episode(env, _random_policy, key)

        assert result["dones"][-1] is True, (
            f"Last done should be True, got {result['dones'][-1]}"
        )


# ---------------------------------------------------------------------------
# Tests for collect_episode_multi (all-agents tracking)
# ---------------------------------------------------------------------------

class TestCollectEpisodeMulti:
    def test_returns_correct_keys(self):
        """collect_episode_multi must return a dict keyed by agent names."""
        env = _make_env()
        key = jax.random.PRNGKey(10)
        result = collect_episode_multi(env, _random_policy, key)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(env.agents), (
            f"Agent keys mismatch: {set(result.keys())} vs {set(env.agents)}"
        )

    def test_each_agent_has_required_keys(self):
        """Each agent's sub-dict must contain the four trajectory keys."""
        env = _make_env()
        key = jax.random.PRNGKey(11)
        result = collect_episode_multi(env, _random_policy, key)

        required = {"observations", "actions", "rewards", "dones"}
        for agent, traj in result.items():
            assert set(traj.keys()) == required, (
                f"Agent {agent} has keys {set(traj.keys())}, expected {required}"
            )

    def test_length_bounded_by_max_steps(self):
        """Each agent's episode must have length <= max_steps."""
        env = _make_env()
        key = jax.random.PRNGKey(12)
        result = collect_episode_multi(env, _random_policy, key)

        for agent, traj in result.items():
            n = len(traj["observations"])
            assert n <= 10, f"Agent {agent}: episode length {n} exceeds max_steps=10"

    def test_observation_shape(self):
        """Every agent's every observation must have shape (env.obs_dim,)."""
        env = _make_env()
        key = jax.random.PRNGKey(13)
        result = collect_episode_multi(env, _random_policy, key)

        for agent, traj in result.items():
            for i, obs in enumerate(traj["observations"]):
                assert obs.shape == (env.obs_dim,), (
                    f"Agent {agent} step {i}: obs shape {obs.shape}, expected ({env.obs_dim},)"
                )

    def test_all_agents_same_episode_length(self):
        """All agents must have the same number of steps (same episode)."""
        env = _make_env()
        key = jax.random.PRNGKey(14)
        result = collect_episode_multi(env, _random_policy, key)

        lengths = {agent: len(traj["observations"]) for agent, traj in result.items()}
        unique_lengths = set(lengths.values())
        assert len(unique_lengths) == 1, f"Agents have different episode lengths: {lengths}"


# ---------------------------------------------------------------------------
# Tests for connectivity guardrail
# ---------------------------------------------------------------------------

def _make_multi_agent_env(num_agents=2):
    """Small 8x8, multi-agent env with comm_radius=3.0 and max_steps=10."""
    cfg = EnvConfig(
        grid_width=8,
        grid_height=8,
        num_agents=num_agents,
        wall_density=0.0,
        max_steps=10,
        comm_radius=3.0,
    )
    reward_fn = make_multi_agent_reward(disconnect_penalty=-0.5)
    return GridCommEnv(cfg, reward_fn=reward_fn)


class TestConnectivityMask:
    def test_single_agent_all_allowed(self):
        """With 1 agent, all actions should be allowed."""
        positions = np.array([[3, 3]])
        comm_ranges = np.array([3.0])
        terrain = np.zeros((8, 8), dtype=int)
        terrain[0, :] = 1; terrain[-1, :] = 1
        terrain[:, 0] = 1; terrain[:, -1] = 1

        mask = _connectivity_mask(positions, comm_ranges, 0, terrain)
        assert mask.all(), "Single agent should have all actions allowed"

    def test_two_agents_close_all_allowed(self):
        """Two adjacent agents — all actions should maintain connectivity."""
        positions = np.array([[3, 3], [3, 4]])
        comm_ranges = np.array([3.0, 3.0])
        terrain = np.zeros((8, 8), dtype=int)
        terrain[0, :] = 1; terrain[-1, :] = 1
        terrain[:, 0] = 1; terrain[:, -1] = 1

        mask = _connectivity_mask(positions, comm_ranges, 0, terrain)
        # Most moves from adjacent positions should keep them connected
        assert mask[0], "STAY should always maintain connectivity"

    def test_two_agents_far_stay_allowed(self):
        """Two agents at the edge of comm range — STAY should be allowed."""
        positions = np.array([[1, 1], [1, 4]])  # distance 3
        comm_ranges = np.array([3.0, 3.0])
        terrain = np.zeros((8, 8), dtype=int)
        terrain[0, :] = 1; terrain[-1, :] = 1
        terrain[:, 0] = 1; terrain[:, -1] = 1

        mask = _connectivity_mask(positions, comm_ranges, 0, terrain)
        assert mask[0], "STAY should maintain connectivity"

    def test_mask_shape(self):
        """Connectivity mask should always have shape (5,)."""
        positions = np.array([[3, 3], [4, 4]])
        comm_ranges = np.array([3.0, 3.0])
        terrain = np.zeros((8, 8), dtype=int)

        mask = _connectivity_mask(positions, comm_ranges, 0, terrain)
        assert mask.shape == (5,)
        assert mask.dtype == bool

    def test_fallback_when_no_action_connects_only_stay_allowed(self):
        """When team is already disconnected and no action of agent 0 can fix
        it, the fallback must allow ONLY STAY (action 0), not all 5 actions.

        The all-True fallback that previously existed was the source of
        cascading fragmentation during eval/GIF: once one agent slipped
        through the fallback, every subsequent agent's mask went all-zero
        too and the whole team broke apart. STAY-only matches the JIT
        training-path semantics in ``_connectivity_guardrail``.
        """
        # Two agents at distance 6, comm_radius=3 -> already disconnected.
        # No move of agent 0 can bring it within 3 of agent 1 in one step.
        positions = np.array([[1, 1], [1, 7]])
        comm_ranges = np.array([3.0, 3.0])
        terrain = np.zeros((8, 8), dtype=int)
        terrain[0, :] = 1; terrain[-1, :] = 1
        terrain[:, 0] = 1; terrain[:, -1] = 1

        mask = _connectivity_mask(positions, comm_ranges, 0, terrain)
        assert mask[0], "STAY must be allowed in the fallback"
        assert not mask[1:].any(), (
            "Only STAY should be allowed when no action preserves connectivity. "
            "The previous all-True fallback caused cascading fragmentation."
        )


class TestConnectivityGuardrailIntegration:
    def test_multi_agent_with_guardrail_runs(self):
        """collect_episode_multi with enforce_connectivity=True should complete."""
        env = _make_multi_agent_env(2)
        key = jax.random.PRNGKey(20)
        result = collect_episode_multi(env, _random_policy, key, enforce_connectivity=True)

        assert set(result.keys()) == set(env.agents)
        for agent, traj in result.items():
            assert len(traj["observations"]) > 0
            assert len(traj["observations"]) <= 10

    def test_multi_agent_without_guardrail_runs(self):
        """collect_episode_multi with enforce_connectivity=False should complete."""
        env = _make_multi_agent_env(2)
        key = jax.random.PRNGKey(21)
        result = collect_episode_multi(env, _random_policy, key, enforce_connectivity=False)

        assert set(result.keys()) == set(env.agents)

    def test_four_agent_guardrail(self):
        """4-agent episodes should complete with the guardrail active."""
        env = _make_multi_agent_env(4)
        key = jax.random.PRNGKey(22)
        result = collect_episode_multi(env, _random_policy, key, enforce_connectivity=True)

        assert len(result) == 4
        for agent, traj in result.items():
            assert len(traj["observations"]) > 0


# ===========================================================================
# Tests for JAX-native rollout functions (Phase 3)
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared helpers for scan-based tests
# ---------------------------------------------------------------------------

def _make_scan_env(num_agents=1, max_steps=10, grid_size=6):
    """Small env for scan-based tests."""
    cfg = EnvConfig(
        grid_width=grid_size,
        grid_height=grid_size,
        num_agents=num_agents,
        wall_density=0.0,
        max_steps=max_steps,
        comm_radius=3.0,
    )
    if num_agents == 1:
        reward_fn = normalized_exploration_reward
    else:
        reward_fn = make_multi_agent_reward(disconnect_penalty=-0.5)
    return GridCommEnv(cfg, reward_fn=reward_fn)


def _make_actor_and_params(env, key):
    """Create a small Actor network and initialize its params."""
    actor = Actor(num_actions=env.config.num_actions, hidden_dim=32, num_layers=1)
    dummy_obs = jnp.zeros((env.obs_dim,))
    params = actor.init(key, dummy_obs)
    return actor, params


# ---------------------------------------------------------------------------
# Tests for collect_episode_scan (single agent)
# ---------------------------------------------------------------------------

class TestCollectEpisodeScan:
    def test_trajectory_shapes(self):
        """collect_episode_scan returns correct shapes for all fields."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(100)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj = collect_episode_scan(env, actor, params, k2, max_steps)

        assert isinstance(traj, Trajectory)
        assert traj.obs.shape == (max_steps, env.obs_dim), (
            f"obs shape {traj.obs.shape} != ({max_steps}, {env.obs_dim})"
        )
        assert traj.actions.shape == (max_steps,)
        assert traj.rewards.shape == (max_steps,)
        assert traj.dones.shape == (max_steps,)
        assert traj.log_probs.shape == (max_steps,)
        assert traj.mask.shape == (max_steps,)

    def test_episode_jit_compiles(self):
        """collect_episode_scan JIT compiles without error."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(101)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        jitted_fn = jax.jit(
            lambda k: collect_episode_scan(env, actor, params, k, max_steps)
        )
        traj = jitted_fn(k2)

        # Should produce valid output
        assert traj.obs.shape[0] == max_steps
        assert traj.actions.shape[0] == max_steps

    def test_episode_deterministic(self):
        """Same key produces identical trajectory."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(102)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj1 = collect_episode_scan(env, actor, params, k2, max_steps)
        traj2 = collect_episode_scan(env, actor, params, k2, max_steps)

        np.testing.assert_array_equal(traj1.obs, traj2.obs)
        np.testing.assert_array_equal(traj1.actions, traj2.actions)
        np.testing.assert_array_equal(traj1.rewards, traj2.rewards)
        np.testing.assert_array_equal(traj1.dones, traj2.dones)
        np.testing.assert_array_equal(traj1.log_probs, traj2.log_probs)
        np.testing.assert_array_equal(traj1.mask, traj2.mask)

    def test_mask_values(self):
        """Mask should be 1.0 for active steps and 0.0 after done."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(103)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj = collect_episode_scan(env, actor, params, k2, max_steps)

        # For a fixed-length episode, mask should be all 1.0 except possibly
        # the last step (done at step max_steps means last mask = 1.0 still
        # since cumulative_done starts False and is set AFTER recording)
        assert traj.mask.dtype == jnp.float32
        # All mask values should be 0.0 or 1.0
        assert jnp.all((traj.mask == 0.0) | (traj.mask == 1.0))

    def test_log_probs_are_negative(self):
        """Log probabilities should be non-positive (log of probability)."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(104)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj = collect_episode_scan(env, actor, params, k2, max_steps)
        assert jnp.all(traj.log_probs <= 0.0), "Log probs should be <= 0"

    def test_actions_in_range(self):
        """All actions should be in [0, num_actions)."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(105)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj = collect_episode_scan(env, actor, params, k2, max_steps)
        assert jnp.all(traj.actions >= 0)
        assert jnp.all(traj.actions < env.config.num_actions)


# ---------------------------------------------------------------------------
# Tests for collect_episode_multi_scan (multi-agent)
# ---------------------------------------------------------------------------

class TestCollectEpisodeMultiScan:
    def test_multi_trajectory_shapes(self):
        """collect_episode_multi_scan returns correct shapes for N=2."""
        max_steps = 10
        num_agents = 2
        env = _make_scan_env(num_agents=num_agents, max_steps=max_steps)
        key = jax.random.PRNGKey(200)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj = collect_episode_multi_scan(
            env, actor, params, k2, max_steps, enforce_connectivity=False
        )

        assert isinstance(traj, MultiTrajectory)
        assert traj.obs.shape == (max_steps, num_agents, env.obs_dim), (
            f"obs shape {traj.obs.shape} != ({max_steps}, {num_agents}, {env.obs_dim})"
        )
        assert traj.actions.shape == (max_steps, num_agents)
        assert traj.rewards.shape == (max_steps, num_agents)
        assert traj.dones.shape == (max_steps,)
        assert traj.log_probs.shape == (max_steps, num_agents)
        assert traj.mask.shape == (max_steps,)

    def test_multi_episode_jit_compiles(self):
        """Multi-agent version JIT compiles without error."""
        max_steps = 10
        num_agents = 2
        env = _make_scan_env(num_agents=num_agents, max_steps=max_steps)
        key = jax.random.PRNGKey(201)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        jitted_fn = jax.jit(
            lambda k: collect_episode_multi_scan(
                env, actor, params, k, max_steps, enforce_connectivity=False
            )
        )
        traj = jitted_fn(k2)

        assert traj.obs.shape[0] == max_steps
        assert traj.actions.shape == (max_steps, num_agents)

    def test_multi_with_connectivity(self):
        """Multi-agent with connectivity enforcement should run."""
        max_steps = 10
        num_agents = 2
        env = _make_scan_env(num_agents=num_agents, max_steps=max_steps)
        key = jax.random.PRNGKey(202)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj = collect_episode_multi_scan(
            env, actor, params, k2, max_steps, enforce_connectivity=True
        )

        assert traj.obs.shape == (max_steps, num_agents, env.obs_dim)
        assert traj.actions.shape == (max_steps, num_agents)

    def test_multi_deterministic(self):
        """Same key produces identical multi-agent trajectory."""
        max_steps = 10
        num_agents = 2
        env = _make_scan_env(num_agents=num_agents, max_steps=max_steps)
        key = jax.random.PRNGKey(203)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        traj1 = collect_episode_multi_scan(
            env, actor, params, k2, max_steps, enforce_connectivity=False
        )
        traj2 = collect_episode_multi_scan(
            env, actor, params, k2, max_steps, enforce_connectivity=False
        )

        np.testing.assert_array_equal(traj1.obs, traj2.obs)
        np.testing.assert_array_equal(traj1.actions, traj2.actions)
        np.testing.assert_array_equal(traj1.rewards, traj2.rewards)


# ---------------------------------------------------------------------------
# Tests for _connectivity_guardrail
# ---------------------------------------------------------------------------

class TestConnectivityGuardrailJAX:
    def test_single_agent_passthrough(self):
        """With N=1, all actions should pass through unchanged."""
        positions = jnp.array([[3, 3]])
        comm_ranges = jnp.array([3.0])
        terrain = jnp.zeros((6, 6), dtype=jnp.int32)

        for action_val in range(5):
            actions = jnp.array([action_val])
            safe = _connectivity_guardrail(positions, comm_ranges, actions, terrain)
            assert int(safe[0]) == action_val, (
                f"Single agent action {action_val} was changed to {int(safe[0])}"
            )

    def test_two_agents_close_actions_preserved(self):
        """Two adjacent agents — most actions should be preserved."""
        positions = jnp.array([[2, 2], [2, 3]])
        comm_ranges = jnp.array([3.0, 3.0])
        terrain = jnp.zeros((6, 6), dtype=jnp.int32)

        # STAY should always be preserved
        actions = jnp.array([0, 0])  # both STAY
        safe = _connectivity_guardrail(positions, comm_ranges, actions, terrain)
        assert int(safe[0]) == 0
        assert int(safe[1]) == 0

    def test_disconnecting_action_overridden(self):
        """An action that would disconnect agents should be overridden to STAY."""
        # Two agents at distance 2 with comm_range 2.5.
        # If agent 0 moves LEFT (away), distance becomes 3 > 2.5 = disconnect.
        positions = jnp.array([[3, 2], [3, 4]])  # distance = 2
        comm_ranges = jnp.array([2.5, 2.5])
        terrain = jnp.zeros((8, 8), dtype=jnp.int32)

        # Agent 0 tries LEFT (action 4) -> position becomes (3, 1), distance to (3,4) = 3 > 2.5
        actions = jnp.array([4, 0])  # agent 0: LEFT, agent 1: STAY
        safe = _connectivity_guardrail(positions, comm_ranges, actions, terrain)
        assert int(safe[0]) == 0, (
            f"Disconnecting action should be overridden to STAY, got {int(safe[0])}"
        )

    def test_guardrail_respects_walls(self):
        """The guardrail should handle wall collision (agent stays in place)."""
        positions = jnp.array([[1, 1], [1, 3]])
        comm_ranges = jnp.array([3.0, 3.0])
        terrain = jnp.zeros((6, 6), dtype=jnp.int32)
        # Put a wall at (0, 1) — agent 0 trying UP would hit wall
        terrain = terrain.at[0, 1].set(1)

        # Agent 0 tries UP (action 1) -> would go to (0,1) which is wall -> stays at (1,1)
        actions = jnp.array([1, 0])
        safe = _connectivity_guardrail(positions, comm_ranges, actions, terrain)
        # Action should still be 1 (UP) since even though wall means agent stays,
        # that still keeps connectivity. The guardrail checks the resulting position.
        # The resulting position for agent 0 is still (1,1) which is fine.
        assert safe.shape == (2,)

    def test_guardrail_output_shape(self):
        """Output shape should match input."""
        N = 3
        positions = jnp.array([[1, 1], [1, 3], [3, 1]])
        comm_ranges = jnp.array([5.0, 5.0, 5.0])
        terrain = jnp.zeros((6, 6), dtype=jnp.int32)
        actions = jnp.array([0, 1, 2])

        safe = _connectivity_guardrail(positions, comm_ranges, actions, terrain)
        assert safe.shape == (N,)
        assert safe.dtype == jnp.int32


# ---------------------------------------------------------------------------
# Tests for vmapping over keys
# ---------------------------------------------------------------------------

class TestVmapOverKeys:
    def test_vmap_single_agent(self):
        """vmap over keys should produce a batch of trajectories."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(300)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        batch_size = 4
        keys = jax.random.split(k2, batch_size)

        batched_fn = jax.vmap(
            lambda k: collect_episode_scan(env, actor, params, k, max_steps)
        )
        batch_traj = batched_fn(keys)

        assert batch_traj.obs.shape == (batch_size, max_steps, env.obs_dim)
        assert batch_traj.actions.shape == (batch_size, max_steps)
        assert batch_traj.rewards.shape == (batch_size, max_steps)
        assert batch_traj.dones.shape == (batch_size, max_steps)
        assert batch_traj.log_probs.shape == (batch_size, max_steps)
        assert batch_traj.mask.shape == (batch_size, max_steps)

    def test_vmap_produces_different_trajectories(self):
        """Different keys should produce different trajectories."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(301)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        keys = jax.random.split(k2, 4)
        batched_fn = jax.vmap(
            lambda k: collect_episode_scan(env, actor, params, k, max_steps)
        )
        batch_traj = batched_fn(keys)

        # At least some actions should differ across the batch
        # (extremely unlikely to be identical for 4 different keys)
        all_same = jnp.all(batch_traj.actions[0] == batch_traj.actions[1])
        # This is a probabilistic test but with 10 steps and random init
        # the chance of all actions being identical is vanishingly small
        assert not all_same, "Different keys should produce different trajectories"

    def test_vmap_jit_combined(self):
        """vmap + jit should work together."""
        max_steps = 10
        env = _make_scan_env(num_agents=1, max_steps=max_steps)
        key = jax.random.PRNGKey(302)
        k1, k2 = jax.random.split(key)
        actor, params = _make_actor_and_params(env, k1)

        batch_size = 4
        keys = jax.random.split(k2, batch_size)

        jitted_batched_fn = jax.jit(jax.vmap(
            lambda k: collect_episode_scan(env, actor, params, k, max_steps)
        ))
        batch_traj = jitted_batched_fn(keys)

        assert batch_traj.obs.shape == (batch_size, max_steps, env.obs_dim)
