"""Reward functions for the RedWithinBlue multi-agent RL environment.

All functions are JAX-compatible (no Python control flow on traced values)
and follow the signature::

    (new_state: EnvState, prev_state: EnvState, info: Dict) -> Dict[str, float]

Use :func:`compose_rewards` to build weighted combinations.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import jax.numpy as jnp

from red_within_blue.types import CELL_WALL, EnvState

# Type alias (mirrors env.py)
RewardFn = Callable[[EnvState, EnvState, Dict], Dict[str, jnp.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_names(state: EnvState) -> int:
    """Return the number of agents from the state."""
    return state.agent_state.positions.shape[0]


def _per_agent_dict(values: jnp.ndarray, num_agents: int) -> Dict[str, jnp.ndarray]:
    """Pack a 1-D array of per-agent scalars into the canonical dict."""
    return {f"agent_{i}": values[i] for i in range(num_agents)}


# ---------------------------------------------------------------------------
# 1. Exploration reward
# ---------------------------------------------------------------------------

def exploration_reward(
    new_state: EnvState,
    prev_state: EnvState,
    info: Dict,
) -> Dict[str, jnp.ndarray]:
    """+1.0 per agent for each NEW cell explored this step.

    A cell is "newly explored" for an agent if the agent is standing on it
    and the cell's explored count went from 0 (in prev_state) to >= 1
    (in new_state).
    """
    num_agents = _agent_names(new_state)
    positions = new_state.agent_state.positions  # [N, 2]
    prev_explored = prev_state.global_state.grid.explored  # [H, W]

    # For each agent, check if the cell they are on was previously unvisited.
    rows = positions[:, 0]
    cols = positions[:, 1]
    prev_counts = prev_explored[rows, cols]  # [N]

    # +1.0 if the cell was unvisited (count == 0) before this step
    rewards = jnp.where(prev_counts == 0, 1.0, 0.0).astype(jnp.float32)
    return _per_agent_dict(rewards, num_agents)


# ---------------------------------------------------------------------------
# 2. Revisit penalty
# ---------------------------------------------------------------------------

def revisit_penalty(
    new_state: EnvState,
    prev_state: EnvState,
    info: Dict,
) -> Dict[str, jnp.ndarray]:
    """-0.1 per agent if the cell they moved to was already explored."""
    num_agents = _agent_names(new_state)
    positions = new_state.agent_state.positions  # [N, 2]
    prev_explored = prev_state.global_state.grid.explored  # [H, W]

    rows = positions[:, 0]
    cols = positions[:, 1]
    prev_counts = prev_explored[rows, cols]  # [N]

    # -0.1 if the cell had already been visited (count > 0)
    rewards = jnp.where(prev_counts > 0, -0.1, 0.0).astype(jnp.float32)
    return _per_agent_dict(rewards, num_agents)


# ---------------------------------------------------------------------------
# 3. Connectivity reward
# ---------------------------------------------------------------------------

def connectivity_reward(
    new_state: EnvState,
    prev_state: EnvState,
    info: Dict,
) -> Dict[str, jnp.ndarray]:
    """-1.0 per agent when the communication graph is NOT connected."""
    num_agents = _agent_names(new_state)
    is_connected = info["is_connected"]

    # -1.0 if not connected, 0.0 otherwise
    penalty = jnp.where(is_connected, 0.0, -1.0).astype(jnp.float32)
    rewards = jnp.broadcast_to(penalty, (num_agents,))
    return _per_agent_dict(rewards, num_agents)


# ---------------------------------------------------------------------------
# 4. Time penalty
# ---------------------------------------------------------------------------

def time_penalty(
    new_state: EnvState,
    prev_state: EnvState,
    info: Dict,
) -> Dict[str, jnp.ndarray]:
    """-0.01 per agent per step (constant)."""
    num_agents = _agent_names(new_state)
    rewards = jnp.full((num_agents,), -0.01, dtype=jnp.float32)
    return _per_agent_dict(rewards, num_agents)


# ---------------------------------------------------------------------------
# 5. Terminal coverage bonus
# ---------------------------------------------------------------------------

def terminal_coverage_bonus(
    new_state: EnvState,
    prev_state: EnvState,
    info: Dict,
) -> Dict[str, jnp.ndarray]:
    """Bonus proportional to fraction of non-wall cells explored, given at the last step."""
    num_agents = _agent_names(new_state)
    grid = new_state.global_state.grid
    done = new_state.global_state.done

    # Count non-wall cells
    non_wall_mask = (grid.terrain != CELL_WALL)  # [H, W] bool
    total_non_wall = non_wall_mask.sum().astype(jnp.float32)

    # Count explored non-wall cells (explored > 0 AND not a wall)
    explored_mask = (grid.explored > 0) & non_wall_mask
    total_explored = explored_mask.sum().astype(jnp.float32)

    # Coverage fraction (guard against division by zero)
    coverage = total_explored / jnp.maximum(total_non_wall, 1.0)

    # Only give bonus on the terminal step
    bonus = jnp.where(done, coverage, 0.0).astype(jnp.float32)
    rewards = jnp.broadcast_to(bonus, (num_agents,))
    return _per_agent_dict(rewards, num_agents)


# ---------------------------------------------------------------------------
# 6. Competitive reward
# ---------------------------------------------------------------------------

def competitive_reward(
    new_state: EnvState,
    prev_state: EnvState,
    info: Dict,
) -> Dict[str, jnp.ndarray]:
    """Blue (team_id==0) gets +exploration, Red (team_id==1) gets -exploration."""
    num_agents = _agent_names(new_state)
    positions = new_state.agent_state.positions  # [N, 2]
    team_ids = new_state.agent_state.team_ids  # [N]
    prev_explored = prev_state.global_state.grid.explored  # [H, W]

    rows = positions[:, 0]
    cols = positions[:, 1]
    prev_counts = prev_explored[rows, cols]  # [N]

    # Base exploration signal: +1.0 if new cell, 0.0 otherwise
    base = jnp.where(prev_counts == 0, 1.0, 0.0).astype(jnp.float32)

    # Blue team (team_id==0): positive; Red team (team_id==1): negative
    sign = jnp.where(team_ids == 0, 1.0, -1.0).astype(jnp.float32)
    rewards = (base * sign).astype(jnp.float32)
    return _per_agent_dict(rewards, num_agents)


# ---------------------------------------------------------------------------
# 7. Compose rewards
# ---------------------------------------------------------------------------

def compose_rewards(
    *reward_fns: RewardFn,
    weights: Optional[Sequence[float]] = None,
) -> RewardFn:
    """Return a new RewardFn that computes a weighted sum of the given fns.

    Parameters
    ----------
    *reward_fns : RewardFn
        One or more reward functions with the standard signature.
    weights : sequence of float, optional
        Per-function weights.  Defaults to all 1.0.

    Returns
    -------
    RewardFn
        A composite reward function.
    """
    if weights is None:
        weights_list: List[float] = [1.0] * len(reward_fns)
    else:
        weights_list = list(weights)

    if len(weights_list) != len(reward_fns):
        raise ValueError(
            f"Number of weights ({len(weights_list)}) must match "
            f"number of reward functions ({len(reward_fns)})."
        )

    def _composed(
        new_state: EnvState,
        prev_state: EnvState,
        info: Dict,
    ) -> Dict[str, jnp.ndarray]:
        # Compute all individual reward dicts
        all_rewards = [fn(new_state, prev_state, info) for fn in reward_fns]

        # Get agent names from the first result
        agent_names = list(all_rewards[0].keys())

        result: Dict[str, jnp.ndarray] = {}
        for name in agent_names:
            total = jnp.float32(0.0)
            for w, rdict in zip(weights_list, all_rewards):
                total = total + jnp.float32(w) * rdict[name]
            result[name] = total
        return result

    return _composed
