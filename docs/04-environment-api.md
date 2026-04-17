# RedWithinBlue API Reference

## Module Overview

### `red_within_blue.env` — Environment

**`GridCommEnv(config, reward_fn=None)`**

The main environment class. Extends `jaxmarl.environments.multi_agent_env.MultiAgentEnv`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `(key) -> (obs_dict, EnvState)` | Initialize a new episode. JIT-compiled. |
| `step_env` | `(key, state, actions) -> (obs, state, rewards, dones, info)` | Advance one step. JIT-compiled. |
| `get_obs` | `(state) -> Dict[str, Array]` | Agent-local observations only (decentralized). |
| `get_global_state` | `(state) -> Array` | Flat vector of full world state (centralized critic). |
| `get_avail_actions` | `(state) -> Dict[str, Array]` | Legal action masks (all ones by default). |

**Attributes:** `agents` (list of agent name strings), `obs_dim`, `config`, `num_agents`.

**`actions` dict format:** `{"agent_0": int, "agent_1": int, ...}`. Optionally include `"agent_0_msg": Array` for learned message vectors.

**`info` dict contents:** `global_state`, `adjacency`, `degree`, `num_components`, `is_connected`, `collisions`, `step`.

---

### `red_within_blue.types` — Data Types

**`EnvConfig`** — Flax struct dataclass. All fields have defaults. See README for full field list.

**`EnvState`** — Top-level state container.
- `.agent_state` (`AgentState`) — per-agent data
- `.global_state` (`GlobalState`) — world-level data

**`AgentState`** fields: `positions`, `comm_ranges`, `team_ids`, `uids`, `messages_out`, `messages_in`, `local_map`, `local_scan`.

**`GlobalState`** fields: `grid` (`GridState`), `graph` (`GraphTracker`), `all_positions`, `step`, `done`, `key`.

**`GridState`** fields: `terrain`, `occupancy`, `explored`.

**`GraphTracker`** fields: `adjacency`, `degree`, `num_components`, `is_connected`, `adjacency_timeline`, `num_components_timeline`, `is_connected_timeline`, `degree_timeline`, `isolated_timeline`, `node_features`, `current_step`.

**`Action`** — IntEnum: `STAY=0`, `UP=1`, `RIGHT=2`, `DOWN=3`, `LEFT=4`.

**Constants:** `CELL_EMPTY=0`, `CELL_WALL=1`, `CELL_OBSTACLE=2`, `CELL_OCCUPIED=3`, `MAP_UNKNOWN=0`, `MAP_FREE=1`, `MAP_WALL=2`, `MAP_OBSTACLE=3`.

---

### `red_within_blue.grid` — Grid Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_grid` | `(width, height, wall_density, key) -> GridState` | Create terrain with boundary walls and random interior walls. |
| `get_local_scan` | `(terrain, occupancy, pos, obs_radius) -> Array[d,d]` | Extract local observation patch (padded with CELL_WALL at edges). |
| `update_occupancy` | `(positions, uids, grid_shape) -> Array[H,W]` | Scatter agent UIDs onto occupancy grid. |
| `update_exploration` | `(explored, positions) -> Array[H,W]` | Increment visit counts at agent positions. |

---

### `red_within_blue.movement` — Movement Resolution

| Function | Signature | Description |
|----------|-----------|-------------|
| `resolve_actions` | `(positions, actions, terrain, grid_shape, passable_types=None) -> (new_positions, collision_mask)` | Resolve movement actions with wall blocking and agent-agent collision detection. |

---

### `red_within_blue.agents` — Agent Management

| Function | Signature | Description |
|----------|-----------|-------------|
| `init_agents` | `(config, terrain, key) -> AgentState` | Random valid placement, zero message buffers, sequential UIDs. |
| `update_local_maps` | `(local_map, local_scan, positions, obs_radius) -> Array[N,H,W]` | Write scan data into each agent's local knowledge map. |
| `prepare_messages` | `(local_scan, msg_dim, learned_vectors=None) -> Array[N, total_msg_dim]` | Flatten scan + concatenate optional learned vectors. |

---

### `red_within_blue.comm_graph` — Communication Graph

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_adjacency` | `(positions, comm_ranges) -> Array[N,N]` | Pairwise Euclidean distance, asymmetric-capable. |
| `route_messages` | `(adjacency, messages_out) -> Array[N, msg_dim]` | Mean-pool aggregation: `adj.T @ msgs / degree`. |
| `compute_degree` | `(adjacency) -> Array[N]` | Per-agent neighbor count. |
| `compute_components` | `(adjacency) -> (num_components, is_connected)` | Laplacian eigenvalue method. |
| `compute_isolated` | `(degree) -> Array[N]` | Boolean mask of agents with zero neighbors. |
| `init_tracker` | `(max_steps, num_agents, node_feature_dim) -> GraphTracker` | Allocate empty timeline arrays. |
| `update_tracker` | `(tracker, adjacency, degree, ...) -> GraphTracker` | Record current step into timeline, advance cursor. |
| `get_fragmentation_count` | `(tracker) -> int` | Number of timesteps the graph was disconnected. |
| `get_agent_isolation_duration` | `(tracker, agent_idx) -> int` | Number of timesteps a specific agent was isolated. |

---

### `red_within_blue.rewards` — Reward Functions

All reward functions have signature: `(new_state, prev_state, info) -> Dict[str, Array]`

| Function | Signal |
|----------|--------|
| `exploration_reward` | +1.0 per new cell explored |
| `revisit_penalty` | -0.1 for revisiting explored cells |
| `connectivity_reward` | -1.0 when graph is fragmented |
| `time_penalty` | -0.01 per step |
| `terminal_coverage_bonus` | Coverage fraction at terminal step |
| `competitive_reward` | +1.0 blue / -1.0 red for exploration |

**`compose_rewards(*fns, weights=None) -> RewardFn`** — Weighted sum combinator.

---

### `red_within_blue.wrappers` — Trajectory Recording

**`TrajectoryWrapper(env, save_dir=None)`**

| Method | Description |
|--------|-------------|
| `reset(key)` | Reset env, clear buffer, record initial obs. |
| `step(key, state, actions)` | Step env, record transition. |
| `get_trajectory()` | Return raw buffer (list of snapshot dicts). |
| `save_trajectory(name)` | Save to `.npz` file, return path. |
| `load_trajectory(path)` | Static method. Load `.npz` to flat dict. |

Not JIT-compatible — use for evaluation/logging, not training.

---

### `red_within_blue.replay` — Trajectory Playback

**`ReplayPlayer(trajectory_path, config=None)`**

| Method / Property | Description |
|-------------------|-------------|
| `num_steps` | Number of recorded steps. |
| `current_step` | Current playback position. |
| `step_forward()` | Advance one step, return step data dict. |
| `step_back()` | Go back one step, return step data dict. |
| `jump_to(step)` | Jump to arbitrary step (clamped to valid range). |
| `play(speed=1.0, render_fn=None)` | Auto-play trajectory. |
| `export_frames(render_fn, output_dir)` | Export each step as PNG frame. |

---

### `red_within_blue.logger` — Experiment Logging

**`ExperimentLogger(base_dir="experiments", experiment_name="run")`**

| Method | Description |
|--------|-------------|
| `log_config(config)` | Serialize env config to `config.json`. |
| `log_hyperparams(params)` | Write training hyperparams to `hyperparams.json`. |
| `log_metrics(step, metrics)` | Append to `metrics.jsonl` (one JSON line per call). |
| `save_checkpoint(params, step)` | Save model params as `.npz`. |
| `close()` | Finalize logger. |

**`list_experiments(base_dir) -> List[dict]`** — Scan and list all experiments with metadata.

**`load_experiment(base_dir, experiment_id) -> dict`** — Load config, hyperparams, and metrics for a past experiment.

---

### `red_within_blue.visualizer` — Visualization

**`render_frame(state, config) -> np.ndarray`** — Pure function returning RGB uint8 image.

**`EnvDashboard(config)`**

| Method | Description |
|--------|-------------|
| `update(state)` | Redraw grid and metrics panels. |
| `close()` | Close the matplotlib figure. |
