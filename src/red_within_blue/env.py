"""JaxMARL-compatible multi-agent grid environment with communication graphs.

This module provides ``GridCommEnv``, a thin orchestrator that composes the
pure-JAX modules (grid, movement, agents, comm_graph) and exposes the standard
JaxMARL ``MultiAgentEnv`` interface.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

from red_within_blue.types import (
    Action,
    EnvConfig,
    EnvState,
    AgentState,
    GlobalState,
    GridState,
    GraphTracker,
    CELL_EMPTY,
    MAP_UNKNOWN,
    resolve_view_radius,
    resolve_survey_radius,
)
from red_within_blue import grid as grid_mod
from red_within_blue import movement as movement_mod
from red_within_blue import agents as agents_mod
from red_within_blue import comm_graph as comm_graph_mod

# Type alias for pluggable reward functions.
RewardFn = Callable[[EnvState, EnvState, Dict], Dict[str, chex.Array]]


class GridCommEnv(MultiAgentEnv):
    """General-purpose grid world with distance-based communication.

    Parameters
    ----------
    config : EnvConfig
        Environment configuration.
    reward_fn : callable, optional
        ``(new_state, prev_state, info) -> Dict[str, Array]``.
        If *None*, the env returns zero rewards every step.
    """

    def __init__(self, config: EnvConfig, reward_fn: Optional[RewardFn] = None):
        super().__init__(num_agents=config.num_agents)
        self.config = config
        self.reward_fn = reward_fn

        # Derived constants
        self.view_radius = resolve_view_radius(config)
        self.survey_radius = resolve_survey_radius(config)
        if self.survey_radius > self.view_radius:
            raise ValueError(
                f"survey_radius ({self.survey_radius}) must be <= view_radius "
                f"({self.view_radius}); agents cannot commit cells they do "
                f"not observe."
            )
        self.local_obs = bool(config.local_obs)
        self.obs_d = 2 * self.view_radius + 1
        self.scan_dim = self.obs_d ** 2

        # Agent names (JaxMARL convention)
        self.agents = [f"agent_{i}" for i in range(config.num_agents)]

        # Observation dim: local_scan (flat) + seen (local window or global H·W)
        # + map_fraction + pos(2) + uid + team_id.
        # Inter-agent information sharing happens entirely at the local_map level
        # (see agents.update_local_maps_with_comm) — there are no learned messages
        # and no message bytes in the observation vector.
        self.grid_dim = config.grid_height * config.grid_width
        seen_dim = self.scan_dim if self.local_obs else self.grid_dim
        self.seen_dim = seen_dim
        self.obs_dim = self.scan_dim + seen_dim + 1 + 2 + 1 + 1

        # Spaces
        for agent in self.agents:
            self.observation_spaces[agent] = Box(
                low=-1.0, high=float(max(config.grid_height, config.grid_width)),
                shape=(self.obs_dim,),
            )
            self.action_spaces[agent] = Discrete(config.num_actions)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        cfg = self.config
        k1, k2 = jax.random.split(key)

        # 1. Grid
        grid_state = grid_mod.create_grid(cfg.grid_width, cfg.grid_height, cfg.wall_density, k1)

        # 2. Agents
        agent_state = agents_mod.init_agents(cfg, grid_state.terrain, k2)

        # 3. Update occupancy with initial positions
        occupancy = grid_mod.update_occupancy(agent_state.positions, agent_state.uids,
                                               (cfg.grid_height, cfg.grid_width))
        grid_state = grid_state.replace(occupancy=occupancy)

        # 4. Initial exploration counts
        explored = grid_mod.update_exploration(grid_state.explored, agent_state.positions)
        grid_state = grid_state.replace(explored=explored)

        # 5. Initial local scans (sensor frames, view_radius-sized).
        local_scan = jax.vmap(
            lambda pos: grid_mod.get_local_scan(grid_state.terrain, grid_state.occupancy, pos, self.view_radius)
        )(agent_state.positions)
        agent_state = agent_state.replace(local_scan=local_scan)

        # 6. Build communication graph (needed for map merging)
        adjacency = comm_graph_mod.build_adjacency(agent_state.positions, agent_state.comm_ranges)
        degree = comm_graph_mod.compute_degree(adjacency)
        num_components, is_connected = comm_graph_mod.compute_components(adjacency)
        isolated = comm_graph_mod.compute_isolated(degree)

        # 7. Update local maps — each agent writes its own survey patch AND
        #    receives the survey patches of its comm-graph neighbours. With
        #    team_ids passed through, red → blue messages are replaced with
        #    MAP_UNKNOWN so the red *fogs* any comm-connected blue's belief.
        local_map = agents_mod.update_local_maps_with_comm(
            agent_state.local_map, local_scan, agent_state.positions,
            adjacency, self.view_radius, self.survey_radius,
            team_ids=agent_state.team_ids,
        )
        # 7b. Ground-truth explored counter: reds zero their spawn cell
        #     (no-op when num_red_agents == 0).
        explored = grid_mod.apply_red_contamination(
            grid_state.explored, agent_state.positions, cfg.num_red_agents,
        )
        grid_state = grid_state.replace(explored=explored)
        agent_state = agent_state.replace(local_map=local_map)

        # 8. Initialize GraphTracker and record step 0
        tracker = comm_graph_mod.init_tracker(cfg.max_steps, cfg.num_agents, cfg.node_feature_dim)
        node_features = self._build_node_features(agent_state, degree)
        tracker = comm_graph_mod.update_tracker(
            tracker, adjacency, degree, num_components, is_connected, isolated, node_features,
        )

        # 11. Compose global state
        global_state = GlobalState(
            grid=grid_state,
            graph=tracker,
            all_positions=agent_state.positions,
            step=jnp.int32(0),
            done=jnp.bool_(False),
            key=key,
        )

        state = EnvState(agent_state=agent_state, global_state=global_state)
        obs = self.get_obs(state)
        return obs, state

    # ------------------------------------------------------------------
    # step_env
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, chex.Array], Dict[str, chex.Array], Dict]:
        cfg = self.config
        prev_state = state
        agent_state = state.agent_state
        global_state = state.global_state
        grid_state = global_state.grid

        # Convert actions dict to array [N]
        action_array = jnp.stack([actions[a] for a in self.agents])

        # 1. Movement resolution
        new_positions, collision_mask = movement_mod.resolve_actions(
            agent_state.positions, action_array, grid_state.terrain,
            (cfg.grid_height, cfg.grid_width),
        )
        agent_state = agent_state.replace(positions=new_positions)

        # 2. Update grid occupancy and exploration
        occupancy = grid_mod.update_occupancy(new_positions, agent_state.uids,
                                               (cfg.grid_height, cfg.grid_width))
        explored = grid_mod.update_exploration(grid_state.explored, new_positions)
        grid_state = grid_state.replace(occupancy=occupancy, explored=explored)

        # 3. Local scans (sensor frames, view_radius-sized).
        local_scan = jax.vmap(
            lambda pos: grid_mod.get_local_scan(grid_state.terrain, grid_state.occupancy, pos, self.view_radius)
        )(new_positions)
        agent_state = agent_state.replace(local_scan=local_scan)

        # 4. Communication graph (built from post-move positions)
        adjacency = comm_graph_mod.build_adjacency(new_positions, agent_state.comm_ranges)
        degree = comm_graph_mod.compute_degree(adjacency)
        num_components, is_connected = comm_graph_mod.compute_components(adjacency)
        isolated = comm_graph_mod.compute_isolated(degree)

        # 4b. Per-agent disconnection state (always computed, used by the
        #     grace mechanism below *and* surfaced in info for diagnostics).
        in_largest_cc = comm_graph_mod.compute_largest_cc_mask(adjacency)   # [N] bool
        disconnect_flags = ~in_largest_cc                                    # [N] bool
        # Timer ticks when disconnected, resets when reconnected.
        new_disconnect_timer = jnp.where(
            disconnect_flags,
            agent_state.disconnect_timer + jnp.int32(1),
            jnp.int32(0),
        )
        main_component_size = jnp.sum(in_largest_cc).astype(jnp.int32)

        # 5. Merge own + neighbor survey patches into each agent's local_map.
        #    Red senders' messages to *blue* receivers carry MAP_UNKNOWN
        #    instead of terrain truth — the red fogs the blue's belief.
        #    Red→red and blue→* stay truthful.
        local_map = agents_mod.update_local_maps_with_comm(
            agent_state.local_map, local_scan, new_positions,
            adjacency, self.view_radius, self.survey_radius,
            team_ids=agent_state.team_ids,
        )
        # 5b. Ground-truth explored counter: reds zero their destination
        #     (no-op when num_red_agents == 0).
        explored = grid_mod.apply_red_contamination(
            grid_state.explored, new_positions, cfg.num_red_agents,
        )
        grid_state = grid_state.replace(explored=explored)
        agent_state = agent_state.replace(
            local_map=local_map, disconnect_timer=new_disconnect_timer,
        )

        # 6. Update GraphTracker
        node_features = self._build_node_features(agent_state, degree)
        tracker = comm_graph_mod.update_tracker(
            global_state.graph, adjacency, degree, num_components, is_connected, isolated, node_features,
        )

        # 7. Disconnect-grace failure trigger.
        #    - disconnect_grace == 0: legacy behaviour, no trigger.
        #    - mode 0 (per_agent): any agent's timer hitting grace trips failure.
        #    - mode 1 (team): trip only when the WHOLE graph is disconnected
        #      and has been so for >= grace steps.
        grace = jnp.int32(cfg.disconnect_grace)
        grace_enabled = grace > 0
        per_agent_trip = jnp.any(new_disconnect_timer >= grace)
        # For team mode we need a team-level counter. Approximate it cheaply
        # as: "max timer across agents, but only while the team is split".
        # When the graph is connected, all timers reset to 0, so the max is 0.
        team_trip = (~is_connected) & (jnp.max(new_disconnect_timer) >= grace)
        disconnect_triggered = grace_enabled & jnp.where(
            jnp.int32(cfg.disconnect_mode) == 0, per_agent_trip, team_trip,
        )

        # 9. Update step counter and check termination
        new_step = global_state.step + 1
        done = (new_step >= cfg.max_steps) | disconnect_triggered

        # 10. Compose new state
        new_global = global_state.replace(
            grid=grid_state,
            graph=tracker,
            all_positions=new_positions,
            step=new_step,
            done=done,
            key=key,
        )
        new_state = EnvState(agent_state=agent_state, global_state=new_global)

        # 11. Build info dict
        info = {
            "global_state": self.get_global_state(new_state),
            "adjacency": adjacency,
            "degree": degree,
            "num_components": num_components,
            "is_connected": is_connected,
            "collisions": collision_mask,
            "step": new_step,
            "disconnect_timer": new_disconnect_timer,
            "disconnect_flags": disconnect_flags,
            "disconnect_triggered": disconnect_triggered,
            "main_component_size": main_component_size,
        }

        # 12. Compute rewards
        if self.reward_fn is not None:
            rewards = self.reward_fn(new_state, prev_state, info)
        else:
            rewards = {a: jnp.float32(0.0) for a in self.agents}

        # 12b. Apply disconnect-fail penalty on the trigger step. Applied to
        #      all agents uniformly (the team failed together). This does not
        #      wipe rewards earned before the trigger — coverage still counts.
        fail_penalty = jnp.where(
            disconnect_triggered,
            jnp.float32(cfg.disconnect_fail_penalty),
            jnp.float32(0.0),
        )
        rewards = {a: rewards[a] + fail_penalty for a in self.agents}

        # 13. Dones
        dones = {a: done for a in self.agents}
        dones["__all__"] = done

        # 14. Observations
        obs = self.get_obs(new_state)

        return obs, new_state, rewards, dones, info

    # ------------------------------------------------------------------
    # get_obs — agent-local observations only (CTDE: decentralized)
    # ------------------------------------------------------------------

    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        obs_array = self._build_obs_array(state)
        return {self.agents[i]: obs_array[i] for i in range(self.config.num_agents)}

    # ------------------------------------------------------------------
    # obs_array — flat [N, obs_dim] observations for use in jax.lax.scan
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def obs_array(self, state: EnvState) -> chex.Array:
        """Return observations as a single [N, obs_dim] array (no dict wrapping)."""
        return self._build_obs_array(state)

    # Shared observation builder used by both get_obs and obs_array.
    def _build_obs_array(self, state: EnvState) -> chex.Array:
        cfg = self.config
        agent_state = state.agent_state
        N = cfg.num_agents
        H, W = cfg.grid_height, cfg.grid_width

        # 1. Flattened local scan [N, scan_dim] — raw sensor frame this step.
        flat_scan = agent_state.local_scan.reshape(N, -1).astype(jnp.float32)

        # 2. Seen field. Two modes:
        #    - local_obs=False (default): full H·W known/unknown mask of the
        #      agent's own local_map. Large, grid-size-dependent.
        #    - local_obs=True: a (2·view_radius+1)² window of the same mask,
        #      centered on the agent's position. Constant-size; agents only
        #      "remember" what they can currently see. Out-of-bounds cells
        #      are reported as known (walls). Forces the policy to move
        #      deliberately because it has no global memory to lean on.
        known = (agent_state.local_map != MAP_UNKNOWN).astype(jnp.float32)
        if self.local_obs:
            view_r = self.view_radius
            obs_d = self.obs_d

            def _window(lmap_known: chex.Array, pos: chex.Array) -> chex.Array:
                padded = jnp.pad(lmap_known, pad_width=view_r, mode="constant", constant_values=1.0)
                return jax.lax.dynamic_slice(padded, (pos[0], pos[1]), (obs_d, obs_d))

            seen_windows = jax.vmap(_window)(known, agent_state.positions)  # [N, d, d]
            seen_field = seen_windows.reshape(N, -1)
        else:
            seen_field = known.reshape(N, -1)

        # 3. Fraction of the grid currently known [N, 1] — always over the
        #    full map, regardless of local_obs (cheap, diagnostic).
        map_fraction = (known.sum(axis=(1, 2)) / (H * W))[:, None]

        # 4. Normalized position, UID, team_id.
        norm_pos = agent_state.positions.astype(jnp.float32) / jnp.array([H, W], dtype=jnp.float32)
        uid = agent_state.uids[:, None].astype(jnp.float32)
        if cfg.normalize_uid:
            uid = uid / jnp.float32(cfg.num_agents)
        team = agent_state.team_ids[:, None].astype(jnp.float32)

        return jnp.concatenate(
            [flat_scan, seen_field, map_fraction, norm_pos, uid, team], axis=-1,
        )

    # ------------------------------------------------------------------
    # step_array — array-based step for use in jax.lax.scan
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def step_array(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action_array: chex.Array,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict]:
        """Array-based step: takes action_array [N] int, returns arrays not dicts.

        Returns
        -------
        obs : [N, obs_dim]
        new_state : EnvState
        rewards : [N]
        done : scalar bool
        info : dict
        """
        actions_dict = {self.agents[i]: action_array[i] for i in range(len(self.agents))}
        obs_dict, new_state, rewards_dict, dones_dict, info = self.step_env(key, state, actions_dict)
        obs = self.obs_array(new_state)
        rewards = jnp.stack([rewards_dict[a] for a in self.agents])
        done = dones_dict["__all__"]
        return obs, new_state, rewards, done, info

    # ------------------------------------------------------------------
    # get_global_state — for centralized critic (CTDE: centralized)
    # ------------------------------------------------------------------

    def get_global_state(self, state: EnvState) -> chex.Array:
        cfg = self.config
        gs = state.global_state

        # Flatten: all_positions + terrain + explored + adjacency + step
        flat_pos = gs.all_positions.reshape(-1).astype(jnp.float32)
        flat_terrain = gs.grid.terrain.reshape(-1).astype(jnp.float32)
        flat_explored = gs.grid.explored.reshape(-1).astype(jnp.float32)
        flat_adj = gs.graph.adjacency.reshape(-1).astype(jnp.float32)
        step_arr = jnp.array([gs.step], dtype=jnp.float32)

        return jnp.concatenate([flat_pos, flat_terrain, flat_explored, flat_adj, step_arr])

    # ------------------------------------------------------------------
    # get_avail_actions
    # ------------------------------------------------------------------

    def get_avail_actions(self, state: EnvState) -> Dict[str, chex.Array]:
        # All actions always available (collision resolution handles invalid moves)
        avail = jnp.ones(self.config.num_actions, dtype=jnp.float32)
        return {a: avail for a in self.agents}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_node_features(self, agent_state: AgentState, degree: chex.Array) -> chex.Array:
        """Build per-agent node feature vectors for the GraphTracker.

        Features: [pos_x, pos_y, degree, team_id, uid]
        Returns [N, node_feature_dim] float32
        """
        pos = agent_state.positions.astype(jnp.float32)  # [N, 2]
        deg = degree[:, None].astype(jnp.float32)         # [N, 1]
        team = agent_state.team_ids[:, None].astype(jnp.float32)  # [N, 1]
        uid = agent_state.uids[:, None].astype(jnp.float32)       # [N, 1]
        return jnp.concatenate([pos, deg, team, uid], axis=-1)    # [N, 5]

    @property
    def agent_classes(self) -> dict:
        return {"agent": self.agents}
