"""Episode rollout collectors for RedWithinBlue.

Provides two categories of rollout functions:

**Legacy (Python-loop based):**

- ``collect_episode``: runs one episode tracking a single named agent.
- ``collect_episode_multi``: runs one episode tracking all agents with
  parameter-shared policy.
- ``collect_episode_multi_with_early_termination``: same with early stop.

These operate in Python (not JAX-traced) so they can handle dynamic-length
episodes and Python-side data collection.

**JAX-native (lax.scan based):**

- ``collect_episode_scan``: single-agent, fully JIT-compilable.
- ``collect_episode_multi_scan``: multi-agent with parameter-shared policy,
  fully JIT-compilable and vmappable.
- ``_connectivity_guardrail``: pure-JAX connectivity enforcement.

The JAX-native functions use ``Trajectory`` and ``MultiTrajectory`` dataclasses
as return types (proper JAX pytrees via ``flax.struct``).
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from red_within_blue.env import GridCommEnv
from red_within_blue.comm_graph import build_adjacency, compute_components
from red_within_blue.types import ACTION_DELTAS_ARRAY


# ---------------------------------------------------------------------------
# Connectivity guardrail
# ---------------------------------------------------------------------------


def _connectivity_mask(positions: np.ndarray, comm_ranges: np.ndarray,
                       agent_idx: int, terrain: np.ndarray) -> np.ndarray:
    """Compute which actions keep the communication graph connected.

    Parameters
    ----------
    positions : (N, 2) int array of current agent positions.
    comm_ranges : (N,) float array of per-agent comm ranges.
    agent_idx : index of the agent choosing an action.
    terrain : (H, W) int grid (0 = passable, 1 = wall).

    Returns
    -------
    (5,) bool array — True for actions that maintain connectivity.
    If no action maintains connectivity, only STAY (action 0) is allowed.
    This matches the JIT training-path semantics in
    ``_connectivity_guardrail`` (force-STAY on disconnect) and prevents
    cascading fragmentation from a brittle-chain configuration: previously
    the all-True fallback released the agent to take any action including
    disconnecting ones, which then made every subsequent agent's mask
    empty too.
    """
    N = positions.shape[0]
    if N < 2:
        return np.ones(5, dtype=bool)

    deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
    H, W = terrain.shape
    mask = np.zeros(5, dtype=bool)

    for a in range(5):
        new_pos = positions.copy()
        intended = new_pos[agent_idx] + deltas[a]
        # Clamp to grid and check walls
        r, c = int(intended[0]), int(intended[1])
        r = max(0, min(H - 1, r))
        c = max(0, min(W - 1, c))
        if terrain[r, c] != 0:  # wall — agent stays in place
            r, c = int(positions[agent_idx, 0]), int(positions[agent_idx, 1])
        new_pos[agent_idx] = [r, c]

        # Check connectivity with JAX functions
        pos_jax = jnp.array(new_pos)
        cr_jax = jnp.array(comm_ranges)
        adj = build_adjacency(pos_jax, cr_jax)
        _, connected = compute_components(adj)
        mask[a] = bool(connected)

    # If nothing maintains connectivity, force STAY (matches JIT training path).
    if not mask.any():
        mask[0] = True

    return mask


def collect_episode(
    env: GridCommEnv,
    policy_fn: Callable,
    key: jax.Array,
    agent_name: str = "agent_0",
) -> Dict[str, List]:
    """Run one episode, tracking a single agent's experience.

    Parameters
    ----------
    env : GridCommEnv
        Environment instance with ``reward_fn`` set.
    policy_fn : Callable
        ``(key, obs_array) -> action_int`` — maps a JAX PRNGKey and a 1-D
        observation array to an integer action.  Applied only to
        ``agent_name``; all other agents act randomly.
    key : jax.Array
        JAX PRNGKey for the episode (split internally for reset / steps).
    agent_name : str
        Name of the agent to track (must be in ``env.agents``).

    Returns
    -------
    dict with keys
        - ``"observations"`` : list of 1-D JAX arrays, one per step (shape ``(obs_dim,)``).
        - ``"actions"``      : list of int actions chosen by ``agent_name``.
        - ``"rewards"``      : list of scalar floats received by ``agent_name``.
        - ``"dones"``        : list of bool done flags for ``agent_name``.
    """
    key, reset_key = jax.random.split(key)
    obs_dict, state = env.reset(reset_key)

    trajectory: Dict[str, List] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    done = False
    while not done:
        # Split keys: one for the policy, one for each random agent, one for step.
        num_splits = 1 + len(env.agents) + 1  # policy + N random agents + env step
        keys = jax.random.split(key, num_splits)
        key = keys[0]           # carry-forward key
        policy_key = keys[1]
        random_keys = keys[2: 2 + len(env.agents)]
        step_key = keys[2 + len(env.agents)]

        # Build action dict
        action_dict = {}
        for i, agent in enumerate(env.agents):
            if agent == agent_name:
                action_dict[agent] = int(policy_fn(policy_key, obs_dict[agent]))
            else:
                action_dict[agent] = int(
                    jax.random.randint(random_keys[i], shape=(), minval=0, maxval=env.config.num_actions)
                )

        # Record current observation for the tracked agent
        trajectory["observations"].append(obs_dict[agent_name])
        trajectory["actions"].append(action_dict[agent_name])

        # Step environment (pass actions as jnp arrays)
        jax_actions = {a: jnp.int32(v) for a, v in action_dict.items()}
        obs_dict, state, rewards, dones, _info = env.step_env(step_key, state, jax_actions)

        trajectory["rewards"].append(float(rewards[agent_name]))
        trajectory["dones"].append(bool(dones[agent_name]))

        done = bool(dones["__all__"])

    return trajectory


def collect_episode_multi(
    env: GridCommEnv,
    policy_fn: Callable,
    key: jax.Array,
    enforce_connectivity: bool = True,
) -> Dict[str, Dict[str, List]]:
    """Run one episode tracking all agents with a shared policy.

    The same ``policy_fn`` is applied to every agent (parameter sharing).

    Parameters
    ----------
    env : GridCommEnv
        Environment instance with ``reward_fn`` set.
    policy_fn : Callable
        ``(key, obs_array) -> action_int`` — applied independently to each
        agent's observation at every step.
    key : jax.Array
        JAX PRNGKey for the episode.
    enforce_connectivity : bool
        If True (default), actions that would disconnect the communication
        graph are replaced with STAY (action 0).  Agents are processed
        sequentially so earlier agents' committed moves are visible to
        later agents' connectivity checks.

    Returns
    -------
    dict
        Mapping ``agent_name -> {"observations", "actions", "rewards", "dones"}``.
        Each value list has the same length (number of steps taken).
    """
    key, reset_key = jax.random.split(key)
    obs_dict, state = env.reset(reset_key)

    trajectories: Dict[str, Dict[str, List]] = {
        agent: {"observations": [], "actions": [], "rewards": [], "dones": []}
        for agent in env.agents
    }

    done = False
    while not done:
        # Split: one carry-forward + one per agent (policy) + one env step
        num_splits = 1 + len(env.agents) + 1
        keys = jax.random.split(key, num_splits)
        key = keys[0]
        agent_keys = keys[1: 1 + len(env.agents)]
        step_key = keys[1 + len(env.agents)]

        # Extract positions and comm_ranges for connectivity guardrail.
        # We work on a mutable copy so sequential agent processing sees
        # already-committed moves from earlier agents in the same step.
        if enforce_connectivity and len(env.agents) >= 2:
            positions = np.array(state.agent_state.positions)   # (N, 2)
            comm_ranges = np.array(state.agent_state.comm_ranges)  # (N,)
            terrain = np.array(state.global_state.grid.terrain)  # (H, W)

        # Record observations and choose actions for all agents
        action_dict = {}
        for i, agent in enumerate(env.agents):
            trajectories[agent]["observations"].append(obs_dict[agent])
            action = int(policy_fn(agent_keys[i], obs_dict[agent]))

            # Connectivity guardrail: replace action with STAY if it would
            # disconnect the communication graph.
            if enforce_connectivity and len(env.agents) >= 2:
                mask = _connectivity_mask(positions, comm_ranges, i, terrain)
                if not mask[action]:
                    action = 0  # STAY
                # Commit this agent's move so the next agent sees it
                deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
                H, W = terrain.shape
                intended = positions[i] + deltas[action]
                r = max(0, min(H - 1, int(intended[0])))
                c = max(0, min(W - 1, int(intended[1])))
                if terrain[r, c] != 0:
                    r, c = int(positions[i, 0]), int(positions[i, 1])
                positions[i] = [r, c]

            action_dict[agent] = action
            trajectories[agent]["actions"].append(action)

        # Step environment
        jax_actions = {a: jnp.int32(v) for a, v in action_dict.items()}
        obs_dict, state, rewards, dones, _info = env.step_env(step_key, state, jax_actions)

        # Record rewards and dones
        for agent in env.agents:
            trajectories[agent]["rewards"].append(float(rewards[agent]))
            trajectories[agent]["dones"].append(bool(dones[agent]))

        done = bool(dones["__all__"])

    return trajectories


def collect_episode_multi_with_early_termination(
    env: GridCommEnv,
    policy_fn: Callable,
    key: jax.Array,
    enforce_connectivity: bool = True,
    isolation_threshold: int = 10,
    termination_penalty: float = -1.0,
) -> Tuple[Dict[str, Dict[str, List]], bool]:
    """Like ``collect_episode_multi`` but with isolation-based early termination.

    If ALL agents are simultaneously isolated (degree==0) for
    ``isolation_threshold`` consecutive steps, the episode terminates early
    and ``termination_penalty`` is injected into each agent's final reward.

    Parameters
    ----------
    isolation_threshold : int
        Number of consecutive all-isolated steps to trigger termination.
        0 disables early termination (behaves like ``collect_episode_multi``).
    termination_penalty : float
        Penalty added to each agent's last reward on early termination.

    Returns
    -------
    (trajectories, early_terminated)
        trajectories : same format as ``collect_episode_multi``
        early_terminated : True if episode was cut short
    """
    key, reset_key = jax.random.split(key)
    obs_dict, state = env.reset(reset_key)

    trajectories: Dict[str, Dict[str, List]] = {
        agent: {"observations": [], "actions": [], "rewards": [], "dones": []}
        for agent in env.agents
    }

    done = False
    early_terminated = False
    consecutive_all_isolated = 0

    while not done:
        num_splits = 1 + len(env.agents) + 1
        keys = jax.random.split(key, num_splits)
        key = keys[0]
        agent_keys = keys[1: 1 + len(env.agents)]
        step_key = keys[1 + len(env.agents)]

        if enforce_connectivity and len(env.agents) >= 2:
            positions = np.array(state.agent_state.positions)
            comm_ranges = np.array(state.agent_state.comm_ranges)
            terrain = np.array(state.global_state.grid.terrain)

        action_dict = {}
        for i, agent in enumerate(env.agents):
            trajectories[agent]["observations"].append(obs_dict[agent])
            action = int(policy_fn(agent_keys[i], obs_dict[agent]))

            if enforce_connectivity and len(env.agents) >= 2:
                mask = _connectivity_mask(positions, comm_ranges, i, terrain)
                if not mask[action]:
                    action = 0
                deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
                H, W = terrain.shape
                intended = positions[i] + deltas[action]
                r = max(0, min(H - 1, int(intended[0])))
                c = max(0, min(W - 1, int(intended[1])))
                if terrain[r, c] != 0:
                    r, c = int(positions[i, 0]), int(positions[i, 1])
                positions[i] = [r, c]

            action_dict[agent] = action
            trajectories[agent]["actions"].append(action)

        jax_actions = {a: jnp.int32(v) for a, v in action_dict.items()}
        obs_dict, state, rewards, dones, info = env.step_env(step_key, state, jax_actions)

        for agent in env.agents:
            trajectories[agent]["rewards"].append(float(rewards[agent]))
            trajectories[agent]["dones"].append(bool(dones[agent]))

        done = bool(dones["__all__"])

        # --- Isolation-based early termination ---
        if isolation_threshold > 0 and not done and len(env.agents) >= 2:
            degree = np.array(info["degree"])
            all_isolated = bool(np.all(degree == 0))
            if all_isolated:
                consecutive_all_isolated += 1
            else:
                consecutive_all_isolated = 0

            if consecutive_all_isolated >= isolation_threshold:
                # Inject termination penalty into each agent's last reward
                for agent in env.agents:
                    trajectories[agent]["rewards"][-1] += termination_penalty
                    trajectories[agent]["dones"][-1] = True
                done = True
                early_terminated = True

    return trajectories, early_terminated


# ===========================================================================
# JAX-native rollout functions (Phase 3 — jax.lax.scan based)
# ===========================================================================

# ---------------------------------------------------------------------------
# Trajectory data types
# ---------------------------------------------------------------------------


@struct.dataclass
class Trajectory:
    """Single-agent trajectory from one episode.

    All arrays have leading dimension ``T`` (max_steps).
    """
    obs: chex.Array        # [T, obs_dim]
    actions: chex.Array    # [T]
    rewards: chex.Array    # [T]
    dones: chex.Array      # [T] bool
    log_probs: chex.Array  # [T]
    mask: chex.Array       # [T] float32 (1.0 before done, 0.0 after)


@struct.dataclass
class MultiTrajectory:
    """Multi-agent trajectory from one episode.

    All arrays have leading dimension ``T`` (max_steps).
    ``N`` is the number of agents.
    """
    obs: chex.Array        # [T, N, obs_dim]
    actions: chex.Array    # [T, N]
    rewards: chex.Array    # [T, N]
    dones: chex.Array      # [T] bool
    log_probs: chex.Array  # [T, N]
    mask: chex.Array       # [T] float32


# ---------------------------------------------------------------------------
# Connectivity guardrail (pure JAX, lax.scan over agents)
# ---------------------------------------------------------------------------


def _connectivity_guardrail(
    positions: chex.Array,    # [N, 2]
    comm_ranges: chex.Array,  # [N]
    actions: chex.Array,      # [N] int
    terrain: chex.Array,      # [H, W] int
) -> chex.Array:
    """Override actions that would disconnect the communication graph.

    Processes agents sequentially via ``jax.lax.scan`` — agent *i*'s
    committed move is visible to agent *i+1*'s connectivity check.

    When ``N == 1``, returns the actions unchanged (single agent is always
    connected).

    Parameters
    ----------
    positions : [N, 2] int array of current agent positions.
    comm_ranges : [N] float array of per-agent comm ranges.
    actions : [N] int array of proposed actions.
    terrain : [H, W] int grid (0 = passable, nonzero = wall).

    Returns
    -------
    [N] int array of safe actions (disconnecting actions replaced with STAY=0).
    """
    N = positions.shape[0]
    H, W = terrain.shape

    def _compute_new_pos(pos, action):
        """Compute position after action, handling grid bounds and walls."""
        intended = pos + ACTION_DELTAS_ARRAY[action]
        r = jnp.clip(intended[0], 0, H - 1)
        c = jnp.clip(intended[1], 0, W - 1)
        is_wall = terrain[r, c] != 0
        final_r = jnp.where(is_wall, pos[0], r)
        final_c = jnp.where(is_wall, pos[1], c)
        return jnp.array([final_r, final_c])

    def _scan_body(carry, agent_idx):
        """Process one agent: check if its action disconnects the graph."""
        current_positions, current_actions = carry

        proposed_action = actions[agent_idx]
        new_pos = _compute_new_pos(current_positions[agent_idx], proposed_action)

        # Tentatively place this agent at the new position
        tentative_positions = current_positions.at[agent_idx].set(new_pos)

        # Check connectivity
        adj = build_adjacency(tentative_positions, comm_ranges)
        _, is_connected = compute_components(adj)

        # If disconnected, override to STAY and revert position
        stay_pos = _compute_new_pos(current_positions[agent_idx], jnp.int32(0))
        safe_action = jnp.where(is_connected, proposed_action, jnp.int32(0))
        safe_pos = jnp.where(is_connected, new_pos, stay_pos)

        # Commit this agent's position
        committed_positions = current_positions.at[agent_idx].set(safe_pos)
        committed_actions = current_actions.at[agent_idx].set(safe_action)

        return (committed_positions, committed_actions), None

    # For N=1, skip guardrail entirely
    init_actions = actions
    (_, safe_actions), _ = jax.lax.scan(
        _scan_body,
        (positions, init_actions),
        jnp.arange(N),
    )

    # If N == 1, just return original actions (single agent always connected)
    safe_actions = jnp.where(N < 2, actions, safe_actions)

    return safe_actions


# ---------------------------------------------------------------------------
# Single-agent episode collection (jax.lax.scan)
# ---------------------------------------------------------------------------


def collect_episode_scan(
    env: GridCommEnv,
    actor: nn.Module,
    actor_params,
    key: jax.Array,
    max_steps: int,
) -> Trajectory:
    """Collect one episode of single-agent experience using ``jax.lax.scan``.

    The tracked agent (index 0) uses the learned policy; all other agents
    take random actions.  Fully JIT-compilable and vmappable over ``key``.

    Parameters
    ----------
    env : GridCommEnv
        Environment instance with ``reward_fn`` set.
    actor : nn.Module
        Flax actor network mapping obs -> logits.
    actor_params : pytree
        Parameters for the actor network.
    key : jax.Array
        PRNG key for the episode.
    max_steps : int
        Number of environment steps to collect.

    Returns
    -------
    Trajectory
        Single-agent trajectory with arrays of shape ``[max_steps, ...]``.
    """
    key, reset_key = jax.random.split(key)
    obs_dict, state = env.reset(reset_key)

    num_agents = env.config.num_agents
    num_actions = env.config.num_actions

    def _scan_body(carry, _step_idx):
        state, rng, cumulative_done = carry

        # Split keys: policy, per-agent random, env step
        rng, policy_key, step_key = jax.random.split(rng, 3)
        rng, random_key = jax.random.split(rng)

        # Get observations from state
        obs_all = env.obs_array(state)  # [N, obs_dim]
        obs_agent = obs_all[0]          # [obs_dim]

        # Forward pass through actor
        logits = actor.apply(actor_params, obs_agent)  # [num_actions]

        # Sample action
        action = jax.random.categorical(policy_key, logits)

        # Compute log probability
        log_prob = jax.nn.log_softmax(logits)[action]

        # Build action array for all agents
        # Agent 0 uses policy, others use random actions
        random_actions = jax.random.randint(
            random_key, shape=(num_agents,), minval=0, maxval=num_actions
        )
        action_array = random_actions.at[0].set(action)

        # Step environment
        obs_new, new_state, rewards, done, info = env.step_array(
            step_key, state, action_array
        )

        # Mask: 1.0 if episode was not done before this step, 0.0 after
        mask = (1.0 - cumulative_done.astype(jnp.float32))

        # Reward for the tracked agent, masked
        reward = rewards[0] * mask

        # Update cumulative done
        new_cumulative_done = cumulative_done | done

        # Per-step output
        step_data = (obs_agent, action, reward, done, log_prob, mask)

        return (new_state, rng, new_cumulative_done), step_data

    # Initialize carry
    init_carry = (state, key, jnp.bool_(False))

    # Run scan
    _final_carry, (obs_seq, act_seq, rew_seq, done_seq, lp_seq, mask_seq) = (
        jax.lax.scan(_scan_body, init_carry, jnp.arange(max_steps))
    )

    return Trajectory(
        obs=obs_seq,          # [T, obs_dim]
        actions=act_seq,      # [T]
        rewards=rew_seq,      # [T]
        dones=done_seq,       # [T]
        log_probs=lp_seq,     # [T]
        mask=mask_seq,        # [T]
    )


# ---------------------------------------------------------------------------
# Multi-agent episode collection (jax.lax.scan)
# ---------------------------------------------------------------------------


def collect_episode_multi_scan(
    env: GridCommEnv,
    actor: nn.Module,
    actor_params,
    key: jax.Array,
    max_steps: int,
    enforce_connectivity: bool = False,
    red_policy: str = "shared",
    num_red_agents: int = 0,
    epsilon: float = 0.0,
) -> MultiTrajectory:
    """Collect one episode of multi-agent experience using ``jax.lax.scan``.

    All agents share the same policy (parameter sharing).  Fully
    JIT-compilable and vmappable over ``key``.

    Parameters
    ----------
    env : GridCommEnv
        Environment instance with ``reward_fn`` set.
    actor : nn.Module
        Flax actor network mapping obs -> logits.
    actor_params : pytree
        Parameters for the actor network.
    key : jax.Array
        PRNG key for the episode.
    max_steps : int
        Number of environment steps to collect.
    enforce_connectivity : bool
        If True, apply the connectivity guardrail after sampling actions.
    red_policy : str
        ``"shared"`` — red agents use the same actor as blue (default).
        ``"random"`` — the last ``num_red_agents`` entries' sampled actions
        are replaced with uniform-random actions.
    num_red_agents : int
        Count of red agents occupying the last indices.  Only used when
        ``red_policy == "random"``.

    Returns
    -------
    MultiTrajectory
        Multi-agent trajectory with arrays of shape ``[max_steps, N, ...]``.
    """
    key, reset_key = jax.random.split(key)
    obs_dict, state = env.reset(reset_key)

    num_agents = env.config.num_agents
    num_actions = env.config.num_actions

    # Trace-time mask: True for red agent indices (last num_red_agents entries).
    red_idx_mask = jnp.arange(num_agents) >= (num_agents - num_red_agents)

    def _scan_body(carry, _step_idx):
        state, rng, cumulative_done = carry

        # Split keys
        rng, policy_key, step_key, red_key, eps_key, eps_act_key = jax.random.split(rng, 6)

        # Get observations from state
        obs_all = env.obs_array(state)  # [N, obs_dim]

        # Vectorized forward pass over all agents (parameter sharing)
        all_logits = jax.vmap(actor.apply, in_axes=(None, 0))(
            actor_params, obs_all
        )  # [N, num_actions]

        # Sample actions for all agents
        agent_keys = jax.random.split(policy_key, num_agents)
        actions = jax.vmap(jax.random.categorical)(
            agent_keys, all_logits
        )  # [N]

        # Epsilon-greedy override: per-agent coin flip, replace with uniform.
        # JIT-safe: uses jnp.where so the branch is fused with eps=0 -> all-False mask.
        coin = jax.random.uniform(eps_key, shape=(num_agents,))
        eps_mask = coin < epsilon                                       # [N] bool
        rand_actions = jax.random.randint(
            eps_act_key, shape=(num_agents,), minval=0, maxval=num_actions,
        )
        actions = jnp.where(eps_mask, rand_actions, actions)

        # Random-red override: replace red agents' actions with uniform samples.
        if red_policy == "random" and num_red_agents > 0:
            rand_red_actions = jax.random.randint(
                red_key, shape=(num_agents,), minval=0, maxval=num_actions,
            )
            actions = jnp.where(red_idx_mask, rand_red_actions, actions)

        # Compute log probabilities
        all_log_probs_full = jax.nn.log_softmax(all_logits)  # [N, num_actions]
        log_probs = jax.vmap(lambda lp, a: lp[a])(
            all_log_probs_full, actions
        )  # [N]

        # Optionally enforce connectivity
        safe_actions = jax.lax.cond(
            jnp.bool_(enforce_connectivity) & (num_agents >= 2),
            lambda acts: _connectivity_guardrail(
                state.agent_state.positions,
                state.agent_state.comm_ranges,
                acts,
                state.global_state.grid.terrain,
            ),
            lambda acts: acts,
            actions,
        )

        # Recompute log_probs for any overridden actions
        safe_log_probs = jax.vmap(lambda lp, a: lp[a])(
            all_log_probs_full, safe_actions
        )  # [N]

        # Step environment
        obs_new, new_state, rewards, done, info = env.step_array(
            step_key, state, safe_actions
        )

        # Mask: 1.0 if episode was not done before this step, 0.0 after
        mask = (1.0 - cumulative_done.astype(jnp.float32))

        # Masked rewards
        masked_rewards = rewards * mask  # [N]

        # Update cumulative done
        new_cumulative_done = cumulative_done | done

        # Per-step output
        step_data = (obs_all, safe_actions, masked_rewards, done, safe_log_probs, mask)

        return (new_state, rng, new_cumulative_done), step_data

    # Initialize carry
    init_carry = (state, key, jnp.bool_(False))

    # Run scan
    _final_carry, (obs_seq, act_seq, rew_seq, done_seq, lp_seq, mask_seq) = (
        jax.lax.scan(_scan_body, init_carry, jnp.arange(max_steps))
    )

    return MultiTrajectory(
        obs=obs_seq,          # [T, N, obs_dim]
        actions=act_seq,      # [T, N]
        rewards=rew_seq,      # [T, N]
        dones=done_seq,       # [T]
        log_probs=lp_seq,     # [T, N]
        mask=mask_seq,        # [T]
    )


# ---------------------------------------------------------------------------
# Dec-POMDP blue + POMDP red (joint-policy) rollout
# ---------------------------------------------------------------------------


def collect_episode_multi_scan_joint(
    env: GridCommEnv,
    blue_actor: nn.Module,
    blue_params,
    joint_red_actor: nn.Module,
    joint_red_params,
    key: jax.Array,
    max_steps: int,
    num_red_agents: int,
    enforce_connectivity: bool = False,
) -> MultiTrajectory:
    """POSG rollout: Dec-POMDP blue + centralized POMDP red.

    Blue agents (indices ``[0:n_blue]``) act independently through a
    parameter-shared per-agent ``Actor``. Red agents (indices
    ``[n_blue:N]``) are driven by a single centralized ``JointRedActor``
    that consumes the concatenation of all red observations and outputs
    factorized action logits of shape ``[n_red, num_actions]``.

    The returned ``MultiTrajectory.log_probs`` keeps the ``[T, N]`` layout:
    the blue slice holds per-agent log-probs (decentralized); the red slice
    holds the per-head log-probs from the joint policy (their sum along the
    red axis is the joint log-prob of the red action vector).

    Parameters
    ----------
    env : GridCommEnv
    blue_actor, blue_params : per-agent policy for blue.
    joint_red_actor, joint_red_params : centralized joint red policy.
    key : PRNG key.
    max_steps : trajectory length.
    num_red_agents : n_red; red agents occupy the last indices.
    """
    key, reset_key = jax.random.split(key)
    _obs_dict, state = env.reset(reset_key)

    num_agents = env.config.num_agents
    num_actions = env.config.num_actions
    n_red = num_red_agents
    n_blue = num_agents - n_red

    def _scan_body(carry, _step_idx):
        state, rng, cumulative_done = carry

        rng, blue_key, red_key, step_key = jax.random.split(rng, 4)

        obs_all = env.obs_array(state)  # [N, obs_dim]

        # -- Blue: per-agent decentralized actor --
        blue_obs = obs_all[:n_blue]                                     # [n_blue, obs_dim]
        blue_logits = jax.vmap(blue_actor.apply, in_axes=(None, 0))(
            blue_params, blue_obs,
        )                                                               # [n_blue, num_actions]
        blue_keys = jax.random.split(blue_key, n_blue)
        blue_actions = jax.vmap(jax.random.categorical)(blue_keys, blue_logits)  # [n_blue]
        blue_log_probs_full = jax.nn.log_softmax(blue_logits)           # [n_blue, num_actions]
        blue_log_probs = jax.vmap(lambda lp, a: lp[a])(
            blue_log_probs_full, blue_actions,
        )                                                               # [n_blue]

        # -- Red: centralized joint actor --
        red_obs_joint = obs_all[n_blue:].reshape(-1)                    # [n_red * obs_dim]
        red_logits = joint_red_actor.apply(joint_red_params, red_obs_joint)  # [n_red, num_actions]
        red_keys = jax.random.split(red_key, n_red)
        red_actions = jax.vmap(jax.random.categorical)(red_keys, red_logits)  # [n_red]
        red_log_probs_full = jax.nn.log_softmax(red_logits)             # [n_red, num_actions]
        red_log_probs = jax.vmap(lambda lp, a: lp[a])(
            red_log_probs_full, red_actions,
        )                                                               # [n_red]

        actions = jnp.concatenate([blue_actions, red_actions], axis=0)  # [N]
        log_probs_full = jnp.concatenate(
            [blue_log_probs_full, red_log_probs_full], axis=0,
        )                                                               # [N, num_actions]

        # Optional hard guardrail: force-STAY any agent whose move would
        # fragment the comm graph. Required for the asymmetric fog-of-war
        # mechanic — red→blue belief contamination flows along comm edges,
        # so a fragmented graph silently kills the channel.
        safe_actions = jax.lax.cond(
            jnp.bool_(enforce_connectivity) & (num_agents >= 2),
            lambda acts: _connectivity_guardrail(
                state.agent_state.positions,
                state.agent_state.comm_ranges,
                acts,
                state.global_state.grid.terrain,
            ),
            lambda acts: acts,
            actions,
        )
        # Recompute log-probs against possibly-overridden actions so the
        # policy gradient targets what the env actually executed.
        safe_log_probs = jax.vmap(lambda lp, a: lp[a])(log_probs_full, safe_actions)

        _obs_new, new_state, rewards, done, _info = env.step_array(
            step_key, state, safe_actions,
        )

        mask = 1.0 - cumulative_done.astype(jnp.float32)
        masked_rewards = rewards * mask                                 # [N]
        new_cumulative_done = cumulative_done | done

        step_data = (obs_all, safe_actions, masked_rewards, done, safe_log_probs, mask)
        return (new_state, rng, new_cumulative_done), step_data

    init_carry = (state, key, jnp.bool_(False))
    _final_carry, (obs_seq, act_seq, rew_seq, done_seq, lp_seq, mask_seq) = (
        jax.lax.scan(_scan_body, init_carry, jnp.arange(max_steps))
    )

    return MultiTrajectory(
        obs=obs_seq,
        actions=act_seq,
        rewards=rew_seq,
        dones=done_seq,
        log_probs=lp_seq,
        mask=mask_seq,
    )
