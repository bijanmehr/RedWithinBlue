"""Training reward functions for RedWithinBlue.

These rewards are designed for RL training scenarios and follow the same
signature as the base reward functions in ``red_within_blue.rewards``::

    (new_state: EnvState, prev_state: EnvState, info: Dict) -> Dict[str, jnp.ndarray]

There are two main reward functions here:

- ``normalized_exploration_reward`` — pure exploration, for single-agent stages.
- ``multi_agent_reward`` — exploration + harsh connectivity enforcement,
  for multi-agent stages (2+).  Disconnection zeroes out the exploration
  bonus and adds a flat penalty, so agents learn that scattering is never
  worth it.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from red_within_blue.types import CELL_WALL, EnvState


def normalized_exploration_reward(
    new_state: EnvState,
    prev_state: EnvState,
    info: Dict,
) -> Dict[str, jnp.ndarray]:
    """Reward each agent proportionally when they discover a new cell.

    The reward for agent *i* at each step is::

        reward_i = cells_discovered_by_agent_i / total_discoverable_cells

    A cell is "discovered by agent i" if:
      - Agent i is standing on it after this step, AND
      - The cell's explored count in ``prev_state`` was 0 (never visited before).

    ``total_discoverable_cells`` is the number of non-wall cells in the grid.

    The reward is in [0, 1] by construction: each step an agent can discover
    at most the one cell they stand on, giving a maximum of 1 / total_discoverable.

    Parameters
    ----------
    new_state : EnvState
        State after the step (contains new agent positions).
    prev_state : EnvState
        State before the step (contains pre-step explored counts).
    info : Dict
        Step info dict (not used here, included for interface consistency).

    Returns
    -------
    Dict[str, jnp.ndarray]
        Mapping from "agent_0", "agent_1", ... to scalar float32 rewards.
    """
    num_agents = new_state.agent_state.positions.shape[0]
    positions = new_state.agent_state.positions       # [N, 2] int32
    prev_explored = prev_state.global_state.grid.explored  # [H, W] int32
    terrain = new_state.global_state.grid.terrain          # [H, W] int32

    # Total number of cells agents can visit (non-wall cells)
    total_discoverable = jnp.sum(terrain != CELL_WALL).astype(jnp.float32)
    # Guard against degenerate empty grids
    total_discoverable = jnp.maximum(total_discoverable, 1.0)

    # Per-agent: was the cell they now occupy previously unexplored?
    rows = positions[:, 0]  # [N]
    cols = positions[:, 1]  # [N]
    prev_counts = prev_explored[rows, cols]  # [N]

    # 1.0 if this cell was unexplored before this step, else 0.0
    discovered = jnp.where(prev_counts == 0, 1.0, 0.0).astype(jnp.float32)  # [N]

    # Normalize by total discoverable cells
    rewards = discovered / total_discoverable  # [N]

    return {f"agent_{i}": rewards[i] for i in range(num_agents)}


def make_multi_agent_reward(disconnect_penalty: float = -0.5):
    """Create a multi-agent reward function with configurable disconnect penalty.

    Parameters
    ----------
    disconnect_penalty : float
        Flat penalty per agent per step when the graph is fragmented.
        More negative = harsher. Default -0.5.

    Returns
    -------
    A reward function with the standard (new_state, prev_state, info) signature.
    """
    def _reward_fn(
        new_state: EnvState,
        prev_state: EnvState,
        info: Dict,
    ) -> Dict[str, jnp.ndarray]:
        num_agents = new_state.agent_state.positions.shape[0]
        positions = new_state.agent_state.positions       # [N, 2]
        prev_explored = prev_state.global_state.grid.explored  # [H, W]
        terrain = new_state.global_state.grid.terrain          # [H, W]
        is_connected = info["is_connected"]                    # scalar bool

        total_discoverable = jnp.sum(terrain != CELL_WALL).astype(jnp.float32)
        total_discoverable = jnp.maximum(total_discoverable, 1.0)

        rows = positions[:, 0]
        cols = positions[:, 1]
        prev_counts = prev_explored[rows, cols]

        discovered = jnp.where(prev_counts == 0, 1.0, 0.0).astype(jnp.float32)
        exploration = discovered / total_discoverable  # [N]

        # Gate exploration on connectivity: zero reward if disconnected
        exploration = jnp.where(is_connected, exploration, 0.0)

        # Flat penalty when disconnected
        penalty = jnp.where(is_connected, 0.0, disconnect_penalty).astype(jnp.float32)
        penalty = jnp.broadcast_to(penalty, (num_agents,))

        rewards = exploration + penalty

        return {f"agent_{i}": rewards[i] for i in range(num_agents)}

    return _reward_fn


# Backwards-compatible default instance
multi_agent_reward = make_multi_agent_reward(-0.5)


# ---------------------------------------------------------------------------
# Configurable reward components (for reward engineering sweeps)
# ---------------------------------------------------------------------------


def make_exploration_reward(weight: float = 0.1):
    """Flat ``+weight`` per new cell discovered (unnormalized).

    Unlike ``normalized_exploration_reward``, this is NOT divided by
    total_discoverable — the signal magnitude is the same regardless
    of grid size.
    """
    def _fn(new_state, prev_state, info):
        num_agents = new_state.agent_state.positions.shape[0]
        positions = new_state.agent_state.positions
        prev_explored = prev_state.global_state.grid.explored

        rows = positions[:, 0]
        cols = positions[:, 1]
        prev_counts = prev_explored[rows, cols]

        discovered = jnp.where(prev_counts == 0, weight, 0.0).astype(jnp.float32)
        return {f"agent_{i}": discovered[i] for i in range(num_agents)}

    return _fn


def make_revisit_penalty(weight: float = -0.3):
    """``weight`` (negative) when agent steps on an already-explored cell."""
    def _fn(new_state, prev_state, info):
        num_agents = new_state.agent_state.positions.shape[0]
        positions = new_state.agent_state.positions
        prev_explored = prev_state.global_state.grid.explored

        rows = positions[:, 0]
        cols = positions[:, 1]
        prev_counts = prev_explored[rows, cols]

        penalties = jnp.where(prev_counts > 0, weight, 0.0).astype(jnp.float32)
        return {f"agent_{i}": penalties[i] for i in range(num_agents)}

    return _fn


def make_isolation_penalty(weight: float = -0.5):
    """Per-agent penalty: only the isolated agent (degree==0) is penalized.

    Unlike ``connectivity_reward`` which penalizes ALL agents when the graph
    fragments, this targets individual agents that have no outgoing edges.
    Uses ``info["degree"]`` from ``env.step_env()``.
    """
    def _fn(new_state, prev_state, info):
        num_agents = new_state.agent_state.positions.shape[0]
        degree = info["degree"]  # [N] int32
        isolated = (degree == 0)  # [N] bool

        penalties = jnp.where(isolated, weight, 0.0).astype(jnp.float32)
        return {f"agent_{i}": penalties[i] for i in range(num_agents)}

    return _fn


def make_terminal_coverage_bonus(scale: float = 1.0, divide: bool = True):
    """Terminal bonus proportional to coverage fraction.

    When ``divide=True``, each agent gets ``coverage * scale / num_agents``
    instead of the full ``coverage * scale``.  This splits credit so that
    the total team bonus equals ``coverage * scale``.
    """
    def _fn(new_state, prev_state, info):
        num_agents = new_state.agent_state.positions.shape[0]
        grid = new_state.global_state.grid
        done = new_state.global_state.done

        non_wall = (grid.terrain != CELL_WALL)
        total = non_wall.sum().astype(jnp.float32)
        explored = ((grid.explored > 0) & non_wall).sum().astype(jnp.float32)
        coverage = explored / jnp.maximum(total, 1.0)

        per_agent = coverage * scale
        if divide:
            per_agent = per_agent / jnp.maximum(num_agents, 1)

        bonus = jnp.where(done, per_agent, 0.0).astype(jnp.float32)
        rewards = jnp.broadcast_to(bonus, (num_agents,))
        return {f"agent_{i}": rewards[i] for i in range(num_agents)}

    return _fn


def make_time_penalty(weight: float = -0.01):
    """Flat per-step penalty with configurable magnitude."""
    def _fn(new_state, prev_state, info):
        num_agents = new_state.agent_state.positions.shape[0]
        rewards = jnp.full((num_agents,), weight, dtype=jnp.float32)
        return {f"agent_{i}": rewards[i] for i in range(num_agents)}

    return _fn


def make_cooperative_bonus(weight: float = 0.02):
    """Bonus to connected neighbours when an agent discovers a new cell.

    If agent A discovers a new cell and agent B is connected to A
    (via ``info["adjacency"]``), B receives ``+weight``.
    """
    def _fn(new_state, prev_state, info):
        num_agents = new_state.agent_state.positions.shape[0]
        positions = new_state.agent_state.positions
        prev_explored = prev_state.global_state.grid.explored
        adjacency = info["adjacency"]  # [N, N] bool

        rows = positions[:, 0]
        cols = positions[:, 1]
        prev_counts = prev_explored[rows, cols]
        discovered = (prev_counts == 0).astype(jnp.float32)  # [N]

        # For each agent j, sum discoveries of agents i that are connected to j
        # adjacency[i, j] = True means i can send to j
        neighbour_discoveries = jnp.dot(adjacency.T.astype(jnp.float32), discovered)  # [N]
        rewards = (neighbour_discoveries * weight).astype(jnp.float32)
        return {f"agent_{i}": rewards[i] for i in range(num_agents)}

    return _fn


def make_reward_config(
    exploration_weight: float = 0.1,
    revisit_weight: float = -0.3,
    isolation_weight: float = -0.5,
    terminal_bonus_scale: float = 1.0,
    terminal_divide: bool = True,
    time_weight: float = -0.01,
    cooperative_weight: float = 0.0,
    disconnect_penalty: float = 0.0,
):
    """Master factory: compose all reward components into one RewardFn.

    Parameters
    ----------
    exploration_weight : float
        Per new-cell bonus (unnormalized).  0 to disable.
    revisit_weight : float
        Penalty for stepping on already-explored cell.  0 to disable.
    isolation_weight : float
        Per-agent penalty when degree==0.  0 to disable.
    terminal_bonus_scale : float
        Coverage fraction multiplier at episode end.  0 to disable.
    terminal_divide : bool
        If True, terminal bonus is divided among agents.
    time_weight : float
        Flat per-step penalty.  0 to disable.
    cooperative_weight : float
        Bonus to connected neighbours on discovery.  0 to disable.
    disconnect_penalty : float
        Legacy full-graph disconnect penalty (applied to ALL agents).  0 to disable.
        When nonzero, exploration is also gated on connectivity.
    """
    def _fn(new_state, prev_state, info):
        num_agents = new_state.agent_state.positions.shape[0]
        positions = new_state.agent_state.positions
        prev_explored = prev_state.global_state.grid.explored
        terrain = new_state.global_state.grid.terrain

        rows = positions[:, 0]
        cols = positions[:, 1]
        prev_counts = prev_explored[rows, cols]

        total = jnp.zeros(num_agents, dtype=jnp.float32)

        # --- Exploration ---
        if exploration_weight != 0:
            discovered = jnp.where(prev_counts == 0, exploration_weight, 0.0).astype(jnp.float32)
            if disconnect_penalty != 0:
                # Gate on connectivity (legacy behaviour)
                discovered = jnp.where(info["is_connected"], discovered, 0.0)
            total = total + discovered

        # --- Revisit penalty ---
        if revisit_weight != 0:
            revisit = jnp.where(prev_counts > 0, revisit_weight, 0.0).astype(jnp.float32)
            total = total + revisit

        # --- Per-agent isolation penalty ---
        if isolation_weight != 0:
            degree = info["degree"]
            iso = jnp.where(degree == 0, isolation_weight, 0.0).astype(jnp.float32)
            total = total + iso

        # --- Time penalty ---
        if time_weight != 0:
            total = total + time_weight

        # --- Legacy disconnect penalty (all agents) ---
        if disconnect_penalty != 0:
            penalty = jnp.where(info["is_connected"], 0.0, disconnect_penalty).astype(jnp.float32)
            total = total + penalty

        # --- Cooperative bonus ---
        if cooperative_weight != 0:
            adjacency = info["adjacency"]
            disc_mask = (prev_counts == 0).astype(jnp.float32)
            neighbour_disc = jnp.dot(adjacency.T.astype(jnp.float32), disc_mask)
            total = total + neighbour_disc * cooperative_weight

        # --- Terminal coverage bonus ---
        if terminal_bonus_scale != 0:
            done = new_state.global_state.done
            non_wall = (terrain != CELL_WALL)
            total_cells = non_wall.sum().astype(jnp.float32)
            explored_cells = ((new_state.global_state.grid.explored > 0) & non_wall).sum().astype(jnp.float32)
            coverage = explored_cells / jnp.maximum(total_cells, 1.0)
            per_agent = coverage * terminal_bonus_scale
            if terminal_divide:
                per_agent = per_agent / jnp.maximum(num_agents, 1)
            bonus = jnp.where(done, per_agent, 0.0).astype(jnp.float32)
            total = total + bonus

        return {f"agent_{i}": total[i] for i in range(num_agents)}

    return _fn
