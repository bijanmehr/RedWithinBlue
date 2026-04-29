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

def resolve_view_radius(cfg: "EnvConfig") -> int:
    """Return the effective view radius (legacy obs_radius when unset)."""
    vr = int(cfg.view_radius)
    return int(cfg.obs_radius) if vr < 0 else vr


def resolve_survey_radius(cfg: "EnvConfig") -> int:
    """Return the effective survey radius (legacy obs_radius when unset)."""
    sr = int(cfg.survey_radius)
    return int(cfg.obs_radius) if sr < 0 else sr


@struct.dataclass
class EnvConfig:
    grid_width: int = 32
    grid_height: int = 32
    max_steps: int = 256

    num_agents: int = 4
    num_red_agents: int = 0  # number of adversarial agents (last N indices); 0 = pure cooperative
    num_actions: int = 5  # len(Action); set to 4 to exclude STAY
    comm_radius: float = 5.0

    # ------------------------------------------------------------------
    # Sensing vs per-cell mission (split of the legacy `obs_radius`).
    #
    # `obs_radius`  — legacy single-radius knob kept as a default source for
    #                 both view_radius and survey_radius below. Runtime code
    #                 reads the resolved values, not this field directly.
    # `view_radius` — half-size of the sensor window. The agent's local_scan
    #                 has shape (2·view_radius+1)² and shows terrain codes in
    #                 a square window centered on the agent. This controls
    #                 what the policy SEES.
    # `survey_radius`— half-size of the per-step "survey" footprint (the
    #                 per-cell mission). When an agent is at cell (r, c),
    #                 the (2·survey_radius+1)² square around (r, c) is
    #                 written into its own local_map and propagated to
    #                 connected neighbours. Controls what the agent DOES.
    #                 `survey_radius=0` means "only the cell I'm on".
    # `local_obs`   — if True, swap the global H·W `flat_seen` memory field
    #                 in the observation for a (2·view_radius+1)² window of
    #                 the agent's own local_map (binary known/unknown).
    #                 Shrinks obs_dim from view_d² + H·W + 5 to 2·view_d² + 5.
    #
    # Sentinel -1 for view_radius / survey_radius means "inherit from
    # obs_radius" (preserving legacy behaviour for existing configs).
    obs_radius: int = 5
    view_radius: int = -1
    survey_radius: int = -1
    local_obs: bool = False

    red_blocks_blue: bool = False  # Phase 2: make red-occupied cells impassable to blue

    center_spawn: bool = False  # spawn agents in a Gaussian cluster around grid center

    # If True, divide the per-agent uid feature by num_agents in the obs tail
    # so its range is always (0, 1] regardless of N. Fixes the OOD shift that
    # makes warm-starting an N=4-trained actor onto N=8 inject a per-seed
    # geographic logit bias (ReLU networks extrapolate raw {1..8} past
    # trained {1..4} as a linear scalar). Default False preserves the legacy
    # raw-uid behavior — checkpoints trained without this flag must keep it
    # off at eval/transfer time, and vice versa.
    normalize_uid: bool = False

    # ------------------------------------------------------------------
    # Disconnect-grace mechanism (soft connectivity constraint).
    #
    # The default hard guardrail (force-STAY whenever a move would disconnect
    # the comm graph) traps the policy — agents cluster and never fan out.
    # Grace replaces the hard constraint with a per-agent tolerance window:
    # an agent may leave the graph for up to `disconnect_grace` consecutive
    # steps; when that counter expires the episode terminates with a fixed
    # penalty ("mission failed").
    #
    # `disconnect_grace`      — 0 keeps the legacy hard guardrail in place.
    #                            >0 disables the guardrail and activates the
    #                            soft grace window.
    # `disconnect_fail_penalty` — per-agent reward added when grace expires.
    #                            Typical value: ~1× terminal_bonus per agent.
    # `disconnect_mode`       — 0 = "per_agent" (any i's timer hitting grace
    #                            trips failure), 1 = "team" (only trip when
    #                            the whole graph is disconnected >grace).
    #
    # Timers are **always computed** (even when grace=0) and exposed in the
    # step info dict so adversarial-detection code can read per-agent
    # disconnection history regardless of whether the failure trigger is on.
    disconnect_grace: int = 0
    disconnect_fail_penalty: float = 0.0
    disconnect_mode: int = 0  # 0 = per_agent, 1 = team

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
    local_map: chex.Array      # [N, H, W] int32  (fuses own + neighbor scans via adjacency)
    local_scan: chex.Array     # [N, obs_d, obs_d] int32
    # Per-agent consecutive steps outside the largest connected component.
    # Ticks every env step. Resets to 0 when the agent is in the largest CC.
    # Always populated, regardless of `EnvConfig.disconnect_grace`.
    disconnect_timer: chex.Array   # [N] int32

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
