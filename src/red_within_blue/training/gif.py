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

    key, reset_key = jax.random.split(key)
    obs, state = wrapper.reset(reset_key)

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

        action_dict = {}
        for i, agent in enumerate(env.agents):
            action = int(policy_fn(agent_keys[i], obs[agent]))

            if enforce_connectivity and len(env.agents) >= 2:
                mask = _connectivity_mask(positions, comm_ranges, i, terrain)
                if not mask[action]:
                    action = 0  # STAY
                # Commit move for sequential processing
                deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
                H, W = terrain.shape
                intended = positions[i] + deltas[action]
                r = max(0, min(H - 1, int(intended[0])))
                c = max(0, min(W - 1, int(intended[1])))
                if terrain[r, c] != 0:
                    r, c = int(positions[i, 0]), int(positions[i, 1])
                positions[i] = [r, c]

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

    from red_within_blue.types import CELL_WALL

    for snapshot in trajectory:
        if "state" not in snapshot:
            continue
        st = snapshot["state"]
        rgb = render_dashboard_frame(st, env.config)
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
    }
