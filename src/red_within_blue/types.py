"""Shared data types, enums, and constants for the RedWithinBlue environment."""

from enum import IntEnum
from typing import Tuple

import chex
import jax.numpy as jnp
from flax import struct

# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class Action(IntEnum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


# Row, col deltas for each action.
ACTION_DELTAS: dict[int, Tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}

# JAX-friendly version: array indexed by action int → (drow, dcol)
ACTION_DELTAS_ARRAY = jnp.array([
    [0, 0],   # STAY
    [-1, 0],  # UP
    [0, 1],   # RIGHT
    [1, 0],   # DOWN
    [0, -1],  # LEFT
], dtype=jnp.int32)

# ---------------------------------------------------------------------------
# Cell type constants
# ---------------------------------------------------------------------------

CELL_EMPTY: int = 0
CELL_WALL: int = 1
CELL_OBSTACLE: int = 2
CELL_OCCUPIED: int = 3  # dynamic — written into occupancy map, not terrain

# Agent local map values (what the agent *believes* about a cell)
MAP_UNKNOWN: int = 0
MAP_FREE: int = 1
MAP_WALL: int = 2
MAP_OBSTACLE: int = 3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@struct.dataclass
class EnvConfig:
    grid_width: int = 32
    grid_height: int = 32
    max_steps: int = 256

    num_agents: int = 4
    num_actions: int = 5  # len(Action); set to 4 to exclude STAY
    comm_radius: float = 5.0
    obs_radius: int = 5
    msg_dim: int = 8  # learned message vector size (optional, zeros by default)

    wall_density: float = 0.0
    # Note: passable_types handled as a jnp array at runtime, not in the dataclass
    # (Flax struct dataclasses do not support tuple/list fields well)

    node_feature_dim: int = 5  # pos_x, pos_y, degree, team_id, uid (for GraphTracker)

# ---------------------------------------------------------------------------
# Agent state (Level 1 — feeds policy observations)
# ---------------------------------------------------------------------------

@struct.dataclass
class AgentState:
    positions: chex.Array      # [N, 2] int32
    comm_ranges: chex.Array    # [N] float32
    team_ids: chex.Array       # [N] int32
    uids: chex.Array           # [N] int32
    messages_out: chex.Array   # [N, total_msg_dim]  (scan_dim + msg_dim)
    messages_in: chex.Array    # [N, total_msg_dim]
    local_map: chex.Array      # [N, H, W] int32
    local_scan: chex.Array     # [N, obs_d, obs_d] int32

# ---------------------------------------------------------------------------
# Grid state (part of global)
# ---------------------------------------------------------------------------

@struct.dataclass
class GridState:
    terrain: chex.Array    # [H, W] int32 — static cell types
    occupancy: chex.Array  # [H, W] int32 — dynamic: 0=empty, >0=agent UID
    explored: chex.Array   # [H, W] int32 — global visit counts

# ---------------------------------------------------------------------------
# Graph tracker (part of global — full timeline)
# ---------------------------------------------------------------------------

@struct.dataclass
class GraphTracker:
    # Current step snapshot
    adjacency: chex.Array       # [N, N] bool
    degree: chex.Array          # [N] int32
    num_components: chex.Array  # scalar int32
    is_connected: chex.Array    # scalar bool

    # Full timeline (preallocated [max_steps, ...])
    adjacency_timeline: chex.Array      # [T, N, N] bool
    num_components_timeline: chex.Array  # [T] int32
    is_connected_timeline: chex.Array    # [T] bool
    degree_timeline: chex.Array          # [T, N] int32
    isolated_timeline: chex.Array        # [T, N] bool

    # GNN-ready node features
    node_features: chex.Array  # [T, N, F] float32

    # Write cursor
    current_step: chex.Array   # scalar int32

# ---------------------------------------------------------------------------
# Global state (Level 2 — critic / logging only)
# ---------------------------------------------------------------------------

@struct.dataclass
class GlobalState:
    grid: GridState
    graph: GraphTracker
    all_positions: chex.Array  # [N, 2] — reference copy for critic
    step: chex.Array           # scalar int32
    done: chex.Array           # scalar bool
    key: chex.PRNGKey

# ---------------------------------------------------------------------------
# Top-level env state
# ---------------------------------------------------------------------------

@struct.dataclass
class EnvState:
    agent_state: AgentState
    global_state: GlobalState
