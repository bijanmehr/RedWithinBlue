# Grid Communication RL Environment — Design Spec

## Context

RedWithinBlue needs a general-purpose, JAX-native multi-agent RL environment built on JaxMARL. The environment models agents on a discrete grid with distance-based communication, local partial observability, and per-agent knowledge bases. It must support the full spectrum from cooperative Dec-POMDP to competitive POSG scenarios via pluggable reward functions, and serve as the foundation for a wide range of MARL experiments (PPO, MAPPO, QMIX, CommNet, self-play, curriculum learning, PBT, GNN-based methods, social dilemma studies, etc.).

The first concrete task will be cooperative grid exploration, but the architecture must not be coupled to that objective.

## Architecture: Modular Composition

Separate pure-JAX modules composed by a thin JaxMARL-compatible orchestrator.

```
src/red_within_blue/
├── __init__.py          # Public API exports
├── types.py             # Shared dataclasses, enums, constants
├── grid.py              # Grid world: terrain, occupancy, spatial queries, local views
├── agents.py            # Agent state init, local map updates, message preparation
├── comm_graph.py        # Adjacency, message routing, graph metrics, GraphTracker
├── movement.py          # Action resolution, collision handling
├── env.py               # JaxMARL MultiAgentEnv orchestrator (thin)
├── rewards.py           # Default reward functions (pluggable, optional)
├── wrappers.py          # TrajectoryWrapper (recording) + other JaxMARL-style wrappers
├── replay.py            # ReplayPlayer for post-experiment trajectory playback
├── logger.py            # ExperimentLogger — file-based experiment & hyperparameter logging
└── visualizer.py        # Rendering (frame output) + live matplotlib dashboard
```

Each module is a collection of pure JAX functions. No mutable class state inside modules. Only `env.py` contains a class (to satisfy JaxMARL's `MultiAgentEnv` interface).

### Data Flow Per Step

```
actions Dict[str, Array]
    |
    v
movement.resolve_actions()  -->  new positions + collision info
    |
    v
grid.update_occupancy()     -->  occupancy map refreshed
grid.update_exploration()   -->  global visit counts incremented
    |
    v
grid.get_local_scan()       -->  per-agent local grid patches
agents.update_local_maps()  -->  agent knowledge bases updated
agents.prepare_messages()   -->  outgoing messages = scan encoding (+ optional learned vector)
    |
    v
comm_graph.build_adjacency()   -->  new adjacency matrix from positions + radii
comm_graph.route_messages()    -->  aggregate neighbor messages -> messages_in
comm_graph.compute_metrics()   -->  degree, components, connectivity
comm_graph.update_tracker()    -->  append to full GraphTracker timeline
    |
    v
env.get_obs()               -->  agent-local observations
reward_fn(state, prev, info) -->  per-agent rewards (or zeros if None)
env.check_termination()      -->  step >= max_steps
    |
    v
return (obs, new_state, rewards, dones, info)
```

## Core Data Types (`types.py`)

### Action Enum

```python
class Action(IntEnum):
    """Discrete agent actions. Defined as IntEnum for named access + int compatibility."""
    STAY  = 0    # optional — can be excluded via config
    UP    = 1
    RIGHT = 2
    DOWN  = 3
    LEFT  = 4
```

Actions map to grid deltas: `UP=(-1,0)`, `RIGHT=(0,+1)`, `DOWN=(+1,0)`, `LEFT=(0,-1)`, `STAY=(0,0)`. The enum lives in `types.py` and is the single source of truth for action semantics across all modules.

### Configuration

```python
@struct.dataclass
class EnvConfig:
    # Grid
    grid_width: int = 32
    grid_height: int = 32
    max_steps: int = 256

    # Agents
    num_agents: int = 4
    num_actions: int = 5           # len(Action) — set to 4 to exclude STAY
    comm_radius: float = 5.0       # default per-agent comms range
    obs_radius: int = 5            # local view half-width (scan range = obs range)
    msg_dim: int = 8               # learned message vector size (optional, default zeros)
    # scan_dim is derived at init: (obs_radius * 2 + 1) ** 2

    # Grid content
    wall_density: float = 0.0      # fraction of interior cells as walls
    passable_types: tuple = (CELL_EMPTY,)  # which terrain types agents can enter

    # Graph tracking
    # No history_length — GraphTracker preallocates full [max_steps, ...] timeline
```

### Three-Level State Model (CTDE)

**Level 1 — Agent-Local State** (feeds policy observations):

```python
@struct.dataclass
class AgentState:
    positions: chex.Array          # [N, 2] int32 — own grid position
    comm_ranges: chex.Array        # [N] float32 — per-agent comms radius
    team_ids: chex.Array           # [N] int32 — team assignment
    uids: chex.Array               # [N] int32 — unique identifiers
    messages_out: chex.Array       # [N, scan_dim + msg_dim] — outgoing broadcast
    messages_in: chex.Array        # [N, scan_dim + msg_dim] — aggregated received
    local_map: chex.Array          # [N, H, W] int32 — per-agent accumulated knowledge
                                   #   0=unknown, 1=free, 2=wall, 3=obstacle, etc.
    local_scan: chex.Array         # [N, obs_d, obs_d] int32 — current step's raw scan
```

**Level 2 — Global State** (critic / logging / metrics only):

```python
@struct.dataclass
class GridState:
    terrain: chex.Array            # [H, W] int32 — static cell types (walls, obstacles)
    occupancy: chex.Array          # [H, W] int32 — dynamic: 0=empty, >0=agent UID
    explored: chex.Array           # [H, W] int32 — global visit counts (ground truth)

@struct.dataclass
class GraphTracker:
    """Full graph timeline — preallocated for max_steps.
    Designed to be directly consumable by GNN / graph embedding methods."""

    # Current step snapshot
    adjacency: chex.Array          # [N, N] bool — current communication links
    degree: chex.Array             # [N] int32 — neighbor count per agent
    num_components: chex.Array     # scalar int32
    is_connected: chex.Array       # scalar bool

    # Full timeline (preallocated [max_steps, ...])
    adjacency_timeline: chex.Array # [T, N, N] bool — full graph at every step
    num_components_timeline: chex.Array  # [T] int32 — component count per step
    is_connected_timeline: chex.Array    # [T] bool — was graph in one piece?
    degree_timeline: chex.Array    # [T, N] int32 — each agent's degree over time
    isolated_timeline: chex.Array  # [T, N] bool — was agent isolated at step t?

    # GNN-ready node features
    node_features: chex.Array      # [T, N, F] float32 — per-node features per step
                                   #   F = [pos_x, pos_y, degree, team_id, ...]

    # Write cursor
    current_step: chex.Array       # scalar int32

@struct.dataclass
class GlobalState:
    grid: GridState
    graph: GraphTracker
    all_positions: chex.Array      # [N, 2] — reference copy of AgentState.positions for critic convenience
    step: int
    done: chex.Array
    key: chex.PRNGKey
```

**Top-Level Composition:**

```python
@struct.dataclass
class EnvState:
    agent_state: AgentState        # feeds get_obs() for decentralized policies
    global_state: GlobalState      # feeds info dict for centralized critic / logging
```

### Cell Types

```python
CELL_EMPTY    = 0
CELL_WALL     = 1
CELL_OBSTACLE = 2   # passability configurable via EnvConfig.passable_types
CELL_OCCUPIED = 3   # dynamic — agent present this step
# Extensible: CELL_RESOURCE, CELL_HAZARD, CELL_TARGET, etc.
```

## Module Specifications

### `grid.py` — Grid World

Pure functions operating on `GridState`:

- `create_grid(width, height, wall_density, key) -> GridState` — initialize terrain with boundary walls + random interior walls/obstacles.
- `get_local_scan(terrain, occupancy, position, obs_radius) -> Array[obs_d, obs_d]` — extract local patch centered on position. Out-of-bounds = wall. Single agent; vmap for batch.
- `update_occupancy(terrain, positions, uids) -> Array[H, W]` — recompute occupancy from agent positions.
- `update_exploration(explored, positions) -> Array[H, W]` — increment visit counts at agent positions.

### `agents.py` — Agent State Management

Pure functions operating on `AgentState`:

- `init_agents(config, key) -> AgentState` — spawn agents at random valid positions, initialize empty local maps and message buffers.
- `update_local_maps(local_map, local_scan, positions, obs_radius) -> Array[N, H, W]` — integrate current scan into each agent's accumulated knowledge map.
- `prepare_messages(local_scan, learned_vector) -> Array[N, scan_dim + msg_dim]` — compose outgoing message from scan encoding + learned vector.

### `comm_graph.py` — Communication Graph, Messaging & GraphTracker

Pure functions operating on positions, comms ranges, and `GraphTracker`:

**Graph construction:**
- `build_adjacency(positions, comm_ranges) -> Array[N, N] bool` — pairwise Euclidean distance vs per-agent radius. Asymmetric-capable: `adj[i,j] = dist(i,j) <= comm_ranges[i]` (i can send to j). Symmetric when all radii are equal.

**Message routing:**
- `route_messages(adjacency, messages_out) -> Array[N, scan_dim + msg_dim]` — aggregate neighbor messages. Default: mean aggregation. `messages_in[i] = mean(messages_out[j] for j where adj[j,i])`. Zero vector if isolated.

**Metrics (modular):**
- `compute_degree(adjacency) -> Array[N]` — per-node neighbor count.
- `compute_components(adjacency) -> (num_components, is_connected)` — spectral method via Laplacian eigenvalues. `is_connected = (num_components == 1)`.
- `compute_isolated(degree) -> Array[N] bool` — mask of degree-zero agents.
- Future: `compute_algebraic_connectivity`, `compute_diameter`, `compute_centrality` — opt-in via modular registration.

**GraphTracker update:**
- `init_tracker(max_steps, num_agents, node_feature_dim) -> GraphTracker` — preallocate full timeline arrays.
- `update_tracker(tracker, adjacency, degree, num_components, is_connected, node_features) -> GraphTracker` — write current step's data to the timeline at `current_step` index, increment cursor.
- `get_fragmentation_count(tracker) -> int` — count steps where `is_connected_timeline == False`.
- `get_agent_isolation_duration(tracker, agent_idx) -> int` — count steps agent was isolated.

The full timeline format (`adjacency_timeline[t]` + `node_features[t]`) is directly consumable by temporal GNN methods (T-GCN, EvolveGCN) and graph embedding approaches without any transformation.

### `movement.py` — Action Resolution

Pure function:

- `resolve_actions(positions, actions, terrain, grid_shape, passable_types) -> (new_positions, collision_mask)` — convert discrete actions to intended positions, clamp to bounds, reject moves into impassable cells, resolve agent-agent conflicts (both stay on collision). Returns updated positions and bool mask of collided agents.

### `env.py` — JaxMARL Orchestrator

```python
class GridCommEnv(MultiAgentEnv):
    def __init__(self, config: EnvConfig, reward_fn=None):
        super().__init__(num_agents=config.num_agents)
        self.config = config
        self.reward_fn = reward_fn   # (state, prev_state, info) -> Dict[str, float]
        self.agents = [f"agent_{i}" for i in range(config.num_agents)]
        # Set observation_spaces and action_spaces per agent

    def reset(self, key) -> (Dict[str, Array], EnvState): ...
    def step_env(self, key, state, actions) -> (obs, state, rewards, dones, info): ...
    def get_obs(self, state) -> Dict[str, Array]: ...
    def get_global_state(self, state) -> Array: ...
```

**Observation per agent** (from `get_obs`):
1. Flattened `local_scan[i]` — current surroundings (obs_d x obs_d)
2. `local_map[i]` summary — fraction of cells in agent's map that are non-unknown (scalar float, 0.0 to 1.0)
3. `messages_in[i]` — received neighbor data (scan_dim + msg_dim)
4. Own position (normalized to [0, 1])
5. Own UID, team_id

**Info dict** (from `step_env`):
- `info["global_state"]` — flat array for centralized critic
- `info["graph_state"]` — adjacency, degree, components for GNN methods
- `info["collisions"]` — collision mask for analysis
- `info["step"]` — current timestep

**Reward interface:**
```python
# Signature: (new_state, old_state, info) -> Dict[str, float]
# Returns per-agent rewards keyed by agent name
# None reward_fn -> all zeros (env is pure simulator)
```

**Message flow:**
1. **Default (scan only):** Each agent automatically broadcasts its flattened `local_scan` to neighbors. No policy involvement — this happens every step. `messages_out[i] = [flatten(local_scan[i]), zeros(msg_dim)]`.
2. **With learned comms (opt-in):** The policy can additionally provide a learned vector via `actions["agent_i_msg"] = Array[msg_dim]`. This gets appended to the scan: `messages_out[i] = [flatten(local_scan[i]), learned_vector]`. If no `_msg` key is present, the learned portion stays zeros.
3. **Receiving:** `messages_in[i] = mean(messages_out[j] for neighbors j)`. Isolated agents receive zeros.

This keeps the basic action space simple (discrete movement only) while allowing learned communication when the training code supports it.

**Termination:** Episode ends when `step >= max_steps`. The `dones` dict has per-agent entries + `"__all__"`.

### `rewards.py` — Pluggable Reward Functions

The env computes **no rewards by default** — it is a pure world simulator. All reward logic lives in external functions with signature:

```python
RewardFn = Callable[[EnvState, EnvState, Dict], Dict[str, float]]
#                     new_state  prev_state  info   per-agent rewards
```

Shipped example reward functions (composable building blocks):

- `exploration_reward(new, prev, info)` — +1 per newly explored tile per agent
- `revisit_penalty(new, prev, info)` — -0.1 per revisited tile
- `connectivity_reward(new, prev, info)` — penalty when `is_connected == False`
- `time_penalty(new, prev, info)` — small negative reward per step
- `terminal_coverage_bonus(new, prev, info)` — large bonus at final step based on % explored
- `competitive_reward(new, prev, info)` — blue vs red team scoring

**Composition helper:**
```python
def compose_rewards(*reward_fns, weights=None) -> RewardFn:
    """Combine multiple reward functions with optional weights.
    Example: compose_rewards(exploration_reward, connectivity_reward, weights=[1.0, -0.5])
    """
```

Users can freely add, remove, reweight, or replace any component. The compose helper makes it easy to experiment with different reward combinations without touching env code.

### `visualizer.py` — Rendering & Dashboard

Two-layer visualization:

**Layer 1: Frame renderer** (lightweight, for logging/recording):
- `render_frame(state, config) -> np.ndarray` — returns an RGB image (H, W, 3) of the current grid state. Shows terrain, agent positions, comm links, explored cells. Pure function, no side effects.
- Usable for: episode recording, wandb logging, notebook display.

**Layer 2: Live dashboard** (interactive, for debugging):
- `EnvDashboard` class wrapping matplotlib.
- Left panel: grid view with agents (colored by team), comm links (dashed lines), obstacles, explored cells.
- Right panel: metrics — coverage %, graph connectivity status, per-agent degree, step count.
- Comm graph overlay: draw edges between connected agents, highlight isolated agents.
- Fragmentation indicator: visual alert when `is_connected == False`.
- `dashboard.update(state)` called each step to refresh.
- Optional pause/resume for step-by-step inspection.

The dashboard depends on matplotlib (already a project dependency). The frame renderer uses only numpy for the image array.

### `wrappers.py` — Trajectory Recording & Env Wrappers

JaxMARL-style wrappers that sit on top of any `MultiAgentEnv`:

**TrajectoryWrapper:**
```python
class TrajectoryWrapper(MultiAgentEnv):
    """Records full trajectory (states, actions, rewards, obs) during env execution.
    
    Usage:
        env = GridCommEnv(config)
        env = TrajectoryWrapper(env, save_dir="experiments/run_001/trajectories")
        # ... run episode ...
        env.save_trajectory("episode_42")   # writes to save_dir/episode_42.npz
    """
```

- Intercepts `step()` and `reset()`, stores each transition in an in-memory buffer.
- `save_trajectory(name)` — serializes the buffer to a compressed `.npz` file (numpy format, efficient for large arrays).
- `get_trajectory()` — returns the raw buffer as a dict of stacked arrays without saving.
- Buffer contents per step: `state` (full EnvState), `actions`, `rewards`, `dones`, `obs`.
- Memory-efficient: stores only what's needed for replay. For very long episodes, can flush to disk incrementally.

Composable with other wrappers — wrap first, then add other wrappers on top.

### `replay.py` — Post-Experiment Trajectory Playback

```python
class ReplayPlayer:
    """Load and visually replay saved trajectories.
    
    Usage:
        player = ReplayPlayer("experiments/run_001/trajectories/episode_42.npz", config)
        player.play()             # auto-play at real-time speed
        player.play(speed=2.0)    # 2x speed
        player.step_forward()     # advance one step
        player.step_back()        # go back one step
        player.jump_to(step=100)  # jump to specific timestep
    """
```

- Loads a saved trajectory file and feeds states to the visualizer (`render_frame` or `EnvDashboard`).
- Playback controls: play/pause, step forward/back, jump to step N, adjustable speed.
- Displays current step metrics alongside the grid visualization.
- Can export replay as a sequence of frames (for GIF/video generation).
- Works with any trajectory saved by `TrajectoryWrapper` — decoupled from the env.

### `logger.py` — Experiment Logging & Hyperparameter Tracking

File-based experiment management with structured directories:

```python
class ExperimentLogger:
    """Log experiments, hyperparameters, and metrics to structured directories.
    
    Usage:
        logger = ExperimentLogger(base_dir="experiments", experiment_name="ppo_baseline")
        logger.log_config(env_config)            # save EnvConfig as JSON
        logger.log_hyperparams({"lr": 3e-4, ...}) # save training hyperparams
        logger.log_metrics(step=100, {"coverage": 0.75, "reward": 12.3})  # append metrics
        logger.save_checkpoint(params, step=100)  # save model params
        logger.close()                            # finalize
    """
```

**Directory structure per experiment:**
```
experiments/
└── 20260414_143052_ppo_baseline/       # timestamp + name
    ├── config.json                      # EnvConfig (full env parameters)
    ├── hyperparams.json                 # training hyperparameters
    ├── metrics.jsonl                    # per-step/episode metrics (append-only, one JSON per line)
    ├── trajectories/                    # saved trajectory files (.npz)
    └── checkpoints/                     # model parameter checkpoints
```

- **Experiment ID** — auto-generated from timestamp + user-provided name. Sortable, unique.
- **config.json** — full `EnvConfig` serialized. Captures all env parameters for reproducibility.
- **hyperparams.json** — training hyperparameters (lr, gamma, batch_size, etc.). Separate from env config because they come from the training code, not the env.
- **metrics.jsonl** — append-only JSON lines. Each line: `{"step": N, "episode": M, "coverage": 0.75, ...}`. Efficient for streaming writes during training. Loadable with `pandas.read_json(lines=True)`.
- **Checkpoints** — model parameters saved as `.npz` or pickle. Keyed by step number.
- `list_experiments(base_dir)` — scan experiments directory, return sorted list with metadata.
- `load_experiment(experiment_id)` — load config, hyperparams, metrics for a past experiment.
- No external service dependency. Works offline everywhere (M4 Mac, Colab, GPU cluster). Optional wandb integration can be layered on top by reading from the same file structure.

## Formal Model

**Networked Dec-POMDP / POSG:**

`(I, S, {A_i}, T, {R_i}, {Omega_i}, {O_i}, {M_i}, G(t), gamma, H)`

| Symbol | Meaning | Implementation |
|--------|---------|----------------|
| I | Agent set | `config.num_agents`, `agent.uids`, `agent.team_ids` |
| S | Global state space | `EnvState` |
| A_i | Per-agent action space | `Discrete(num_actions)` |
| T | Transition function | `step_env(key, state, actions)` |
| R_i | Per-agent reward | Pluggable `reward_fn -> Dict[str, float]` |
| Omega_i | Observation space | `Box(obs_dim)` per agent |
| O_i | Observation function | `get_obs()` — agent-local only |
| M_i | Message space | `R^(scan_dim + msg_dim)` |
| G(t) | Dynamic comm graph | `build_adjacency()` each step |
| H | Finite horizon | `config.max_steps` |

The per-agent reward function supports the full POSG spectrum:
- Cooperative (shared reward) = Dec-POMDP
- Competitive (zero-sum) = competitive POSG
- Mixed-motive = social dilemma / general POSG

## Parallelism & Hardware

All operations are pure JAX with static shapes. Three parallelism levels:

1. **`jax.jit`** — compile step_env to XLA. Runs on CPU (M4 Mac), GPU (NVIDIA/Colab), TPU.
2. **`jax.vmap`** — batch B parallel environments. Shapes become `[B, N, 2]`, `[B, N, N]`, etc.
3. **`jax.pmap`** — shard batches across devices for multi-GPU/TPU.

No Python control flow in hot path. No dynamic shapes. Functional immutable state with `.replace()`.

## Verification Plan

Unit tests go in `tests/` directory, one test file per module. All tests use pytest + JAX.

### `tests/test_grid.py`
1. `test_create_grid_boundaries` — verify boundary cells are always walls regardless of wall_density.
2. `test_create_grid_density` — verify interior wall count roughly matches wall_density fraction.
3. `test_create_grid_seeded` — same key produces identical grids (determinism).
4. `test_local_scan_center` — agent at center of open grid gets expected patch.
5. `test_local_scan_edge` — agent near boundary gets walls for out-of-bounds cells.
6. `test_local_scan_sees_obstacles` — local scan correctly reflects terrain types.
7. `test_update_occupancy` — occupancy map correctly shows agent UIDs at positions.
8. `test_update_exploration` — visit counts increment correctly for visited cells.

### `tests/test_movement.py`
9. `test_basic_movement` — each action produces correct position delta.
10. `test_wall_collision` — move into wall keeps agent at current position.
11. `test_boundary_clamp` — move off grid edge is rejected.
12. `test_agent_collision` — two agents targeting same cell both stay.
13. `test_obstacle_passability` — passable_types config respected.
14. `test_collision_mask` — collision_mask correctly flags collided agents.

### `tests/test_comm_graph.py`
15. `test_adjacency_within_range` — agents within comm_radius are connected.
16. `test_adjacency_out_of_range` — agents beyond comm_radius are not connected.
17. `test_adjacency_no_self_loop` — diagonal of adjacency is False.
18. `test_adjacency_asymmetric` — different comm_ranges produce asymmetric adjacency.
19. `test_compute_components_connected` — all agents in range gives num_components=1.
20. `test_compute_components_fragmented` — two clusters gives num_components=2.
21. `test_compute_isolated` — agent far from all others flagged as isolated.
22. `test_route_messages_mean` — verify mean aggregation with known message vectors.
23. `test_route_messages_isolated` — isolated agent receives zero vector.
24. `test_tracker_timeline_write` — verify GraphTracker records at correct step index.
25. `test_tracker_fragmentation_count` — verify fragmentation counter over known timeline.
26. `test_tracker_isolation_duration` — verify per-agent isolation duration computation.

### `tests/test_agents.py`
27. `test_init_agents_valid_positions` — all spawn positions are on empty cells, within bounds.
28. `test_init_agents_seeded` — same key produces identical agent state.
29. `test_update_local_map` — local map integrates scan correctly, unknown cells become known.
30. `test_local_map_persistence` — previously scanned cells remain in local map.
31. `test_prepare_messages_scan_only` — without learned vector, message is scan + zeros.
32. `test_prepare_messages_with_learned` — learned vector appended correctly.

### `tests/test_env.py` (integration)
33. `test_reset_shapes` — all returned arrays have correct shapes per config.
34. `test_step_shapes` — obs, rewards, dones, info have correct shapes and keys.
35. `test_ctde_boundary` — get_obs() contains no global information; info["global_state"] contains full truth.
36. `test_jit_compiles` — `jax.jit(env.step)` compiles and runs without error.
37. `test_vmap_batched` — `jax.vmap(env.step)` over batch of 32 environments produces correct shapes.
38. `test_determinism` — same key + actions produces identical trajectory.
39. `test_termination` — episode terminates at max_steps, dones["__all__"] is True.
40. `test_reward_fn_none` — no reward function returns all zeros.
41. `test_reward_fn_custom` — custom reward function is called and returns per-agent rewards.
42. `test_reward_fn_compose` — compose_rewards combines multiple reward functions correctly.

### `tests/test_visualizer.py`
43. `test_render_frame_shape` — render_frame returns RGB array with correct dimensions.
44. `test_render_frame_deterministic` — same state produces identical image.
45. `test_dashboard_init` — EnvDashboard initializes without error.
46. `test_dashboard_update` — dashboard.update(state) runs without error for multiple steps.

### `tests/test_wrappers.py`
47. `test_trajectory_wrapper_records` — wrapper captures states/actions/rewards for each step.
48. `test_trajectory_wrapper_transparent` — wrapper doesn't alter env behavior (obs, rewards, dones identical to unwrapped).
49. `test_trajectory_save_load` — save trajectory to .npz, reload, verify data matches.
50. `test_trajectory_reset_clears` — buffer resets on env.reset().

### `tests/test_replay.py`
51. `test_replay_load` — ReplayPlayer loads a saved trajectory without error.
52. `test_replay_step_forward` — step_forward advances to correct state.
53. `test_replay_step_back` — step_back returns to previous state.
54. `test_replay_jump_to` — jump_to(N) lands on correct step.

### `tests/test_logger.py`
55. `test_logger_creates_directory` — ExperimentLogger creates structured directory.
56. `test_log_config` — config.json written with correct EnvConfig fields.
57. `test_log_hyperparams` — hyperparams.json written with correct fields.
58. `test_log_metrics_append` — multiple log_metrics calls append to metrics.jsonl.
59. `test_list_experiments` — list_experiments finds and sorts saved experiments.
60. `test_load_experiment` — load_experiment retrieves config, hyperparams, metrics correctly.

### Notes
- The formal model section (Networked Dec-POMDP / POSG) is documentation only — no code generated from it.
- All tests must run without GPU (CPU-only JAX) to work in CI.
- Tests should be fast — small grids (8x8), few agents (2-4), few steps (5-10).
