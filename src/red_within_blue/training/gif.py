"""Animated GIF recording for RedWithinBlue episodes.

Runs an evaluation episode, renders every timestep with the existing
``render_frame`` visualiser, and stitches the frames into an animated GIF
via Pillow.  Also extracts per-frame metrics (visit heatmap, connectivity)
for use in experiment reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from red_within_blue.env import GridCommEnv
from red_within_blue.training.rollout import _connectivity_mask
from red_within_blue.visualizer import render_dashboard_frame
from red_within_blue.wrappers import TrajectoryWrapper


def record_episode_gif(
    env: GridCommEnv,
    policy_fn: Callable,
    key: jax.Array,
    output_path: str,
    fps: int = 4,
    enforce_connectivity: bool = True,
    joint_red_actor=None,
    joint_red_params=None,
    n_red: int = 0,
) -> Dict[str, Any]:
    """Run one episode and save an animated GIF of the agent(s) moving.

    The *policy_fn* is applied to **every** agent at each step (parameter
    sharing).  For single-agent stages this is equivalent to controlling
    the sole agent; for multi-agent stages each agent acts independently
    under the same policy.

    Parameters
    ----------
    env : GridCommEnv
        Environment instance (with reward_fn already set).
    policy_fn : Callable
        ``(key, obs_array) -> action_int`` — maps a JAX PRNGKey and a
        1-D observation to an integer action.
    key : jax.Array
        JAX PRNGKey used for the episode.
    output_path : str
        Destination file path (should end in ``.gif``).
    fps : int
        Frames per second in the output GIF (default 4).

    Returns
    -------
    dict
        ``n_frames``          : int — number of frames written.
        ``visit_heatmap``     : np.ndarray [H, W] — final visit counts.
        ``connectivity``      : list[bool] — per-step graph connectivity.
        ``coverage_over_time``: list[float] — per-step coverage fraction.
    """
    wrapper = TrajectoryWrapper(env)

    # Grace supersedes the hard guardrail: if the env will terminate on
    # disconnect, we do not force-STAY during rollout.
    if int(getattr(env.config, "disconnect_grace", 0)) > 0:
        enforce_connectivity = False

    key, reset_key = jax.random.split(key)
    obs, state = wrapper.reset(reset_key)

    n_total = len(env.agents)
    n_blue_total = n_total - n_red
    stay_intended = np.zeros(n_total, dtype=np.int64)   # policy chose STAY
    stay_forced = np.zeros(n_total, dtype=np.int64)     # guardrail forced STAY
    move_taken = np.zeros(n_total, dtype=np.int64)      # action != STAY
    steps_total = 0

    done = False
    while not done:
        num_splits = 1 + len(env.agents) + 1
        keys = jax.random.split(key, num_splits)
        key = keys[0]
        agent_keys = keys[1 : 1 + len(env.agents)]
        step_key = keys[1 + len(env.agents)]

        # Extract positions for connectivity guardrail
        if enforce_connectivity and len(env.agents) >= 2:
            positions = np.array(state.agent_state.positions)
            comm_ranges = np.array(state.agent_state.comm_ranges)
            terrain = np.array(state.global_state.grid.terrain)

        n_blue = n_blue_total
        steps_total += 1

        red_actions_precomputed = None
        if joint_red_actor is not None and n_red > 0:
            red_obs_flat = jnp.concatenate(
                [obs[env.agents[n_blue + r]] for r in range(n_red)]
            )
            red_logits = joint_red_actor.apply(joint_red_params, red_obs_flat)
            red_action_keys = jax.random.split(agent_keys[n_blue], n_red)
            red_actions_precomputed = [
                int(jax.random.categorical(red_action_keys[r], red_logits[r]))
                for r in range(n_red)
            ]

        action_dict = {}
        for i, agent in enumerate(env.agents):
            if red_actions_precomputed is not None and i >= n_blue:
                intended_action = red_actions_precomputed[i - n_blue]
            else:
                intended_action = int(policy_fn(agent_keys[i], obs[agent]))

            action = intended_action
            forced_by_guardrail = False
            if enforce_connectivity and len(env.agents) >= 2:
                mask = _connectivity_mask(positions, comm_ranges, i, terrain)
                if not mask[action]:
                    action = 0  # STAY
                    forced_by_guardrail = (intended_action != 0)
                # Commit move for sequential processing
                deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
                H, W = terrain.shape
                intended = positions[i] + deltas[action]
                r = max(0, min(H - 1, int(intended[0])))
                c = max(0, min(W - 1, int(intended[1])))
                if terrain[r, c] != 0:
                    r, c = int(positions[i, 0]), int(positions[i, 1])
                positions[i] = [r, c]

            if forced_by_guardrail:
                stay_forced[i] += 1
            elif action == 0:
                stay_intended[i] += 1
            else:
                move_taken[i] += 1

            action_dict[agent] = jnp.int32(action)

        obs, state, rewards, dones, info = wrapper.step(
            step_key, state, action_dict
        )
        done = bool(dones["__all__"])

    # Extract EnvState from every recorded snapshot.
    trajectory = wrapper.get_trajectory()
    frames: list[Image.Image] = []
    connectivity: list[bool] = []
    coverage_over_time: list[float] = []
    visit_heatmap = None

    from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
    from red_within_blue.visualizer import _merge_team_belief

    blue_ever_known: Optional[np.ndarray] = None

    for snapshot in trajectory:
        if "state" not in snapshot:
            continue
        st = snapshot["state"]

        local_maps_np = np.asarray(st.agent_state.local_map)
        team_ids_np = np.asarray(st.agent_state.team_ids)
        blue_belief_now = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)
        known_now = (blue_belief_now != MAP_UNKNOWN)
        if blue_ever_known is None:
            blue_ever_known = known_now.copy()
        else:
            blue_ever_known = blue_ever_known | known_now

        rgb = render_dashboard_frame(st, env.config, blue_ever_known=blue_ever_known)
        frames.append(Image.fromarray(rgb))

        # Track connectivity
        connectivity.append(bool(st.global_state.graph.is_connected))

        # Track coverage
        terrain = np.asarray(st.global_state.grid.terrain)
        explored = np.asarray(st.global_state.grid.explored)
        non_wall = terrain != CELL_WALL
        total = int(non_wall.sum())
        visited = int(((explored > 0) & non_wall).sum())
        coverage_over_time.append(visited / max(total, 1))

        # Keep last heatmap
        visit_heatmap = explored.copy()

    if not frames:
        return {"n_frames": 0, "visit_heatmap": None, "connectivity": [],
                "coverage_over_time": []}

    # Ensure output directory exists.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frame_duration_ms = max(1, 1000 // fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )

    return {
        "n_frames": len(frames),
        "visit_heatmap": visit_heatmap,
        "connectivity": connectivity,
        "coverage_over_time": coverage_over_time,
        "stay_intended": stay_intended.tolist(),
        "stay_forced": stay_forced.tolist(),
        "move_taken": move_taken.tolist(),
        "steps_total": int(steps_total),
        "blue_ever_known_pct": (
            float(blue_ever_known.sum()) / float(blue_ever_known.size) * 100.0
            if blue_ever_known is not None else 0.0
        ),
    }
