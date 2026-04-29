"""Visualisation utilities for the RedWithinBlue environment.

Provides:
- ``render_frame``: pure function that returns an RGB numpy array of the current state.
- ``EnvDashboard``: live matplotlib dashboard with grid + metrics panels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from red_within_blue.types import EnvConfig, EnvState

from red_within_blue.types import (
    CELL_WALL, CELL_OBSTACLE,
    MAP_UNKNOWN, MAP_FREE, MAP_WALL, MAP_OBSTACLE,
)

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

_COLOUR_EMPTY = np.array([1.0, 1.0, 1.0])        # white (unvisited)
_COLOUR_WALL = np.array([0.12, 0.12, 0.12])       # near-black walls
_COLOUR_OBSTACLE = np.array([0.5, 0.5, 0.5])      # mid-gray obstacles

_TEAM_COLOURS = {
    0: "#3366CC",  # blue
    1: "#CC3333",  # red
}
_DEFAULT_AGENT_COLOUR = "#888888"

_CELL_PX = 30  # target pixels per cell
_LINK_COLOUR = "#AAAAAA"
_LINK_LINEWIDTH = 0.8
_GRID_LINE_COLOUR = "#CCCCCC"
_GRID_LINE_WIDTH = 0.3


# ---------------------------------------------------------------------------
# Helper: build cell-colour image
# ---------------------------------------------------------------------------

def _cell_colour_map(terrain: np.ndarray, explored: np.ndarray,
                     max_steps: int = 256) -> np.ndarray:
    """Return an [H, W, 3] float64 greyscale colour array for the grid.

    Unvisited cells are white. Visited cells darken proportionally to their
    visit count, scaled by *max_steps* so the darkest possible cell is near-black.
    Walls are rendered as near-black regardless of visits.
    """
    H, W = terrain.shape
    img = np.ones((H, W, 3), dtype=np.float64)  # default white (unvisited)

    # Walls and obstacles — fixed colours
    img[terrain == CELL_WALL] = _COLOUR_WALL
    img[terrain == CELL_OBSTACLE] = _COLOUR_OBSTACLE

    # Explored cells — greyscale intensity proportional to visit count.
    # intensity: 1.0 (white, 0 visits) -> 0.25 (dark grey, max visits)
    passable_mask = (terrain == 0)
    visited_mask = (explored > 0) & passable_mask
    counts = explored[visited_mask].astype(np.float64)
    # Normalise by max_steps (total budget); clamp to [0, 1]
    frac = np.clip(counts / max(max_steps, 1), 0.0, 1.0)
    # Map: 0 visits -> 0.92 (light grey, just barely tinted), max -> 0.25 (dark)
    grey = 0.92 - 0.67 * frac
    img[visited_mask] = grey[:, None]  # broadcast to RGB

    return img


_BELIEF_COLOURS = {
    MAP_UNKNOWN: np.array([0.10, 0.10, 0.10]),   # near-black "fog"
    MAP_FREE:    np.array([0.92, 0.92, 0.92]),   # light grey known floor
    MAP_WALL:    np.array([0.30, 0.30, 0.30]),   # dark grey known wall
    MAP_OBSTACLE:np.array([0.55, 0.55, 0.55]),   # mid grey known obstacle
}


def _merge_team_belief(local_maps: np.ndarray, team_ids: np.ndarray,
                       target_team: int) -> np.ndarray:
    """Element-wise team belief merge: a cell is known if any team member knows it.

    Red→blue fogging writes ``MAP_UNKNOWN`` into blue receivers' local_maps —
    if every blue agent who held that cell got fogged, the merged belief is
    UNKNOWN. Conflicts between known values cannot occur because every truthful
    write carries the same terrain code.
    """
    members = np.where(team_ids == target_team)[0]
    if members.size == 0:
        return np.full(local_maps.shape[1:], MAP_UNKNOWN, dtype=np.int32)
    team_maps = local_maps[members]                            # [K, H, W]
    known_mask = team_maps != MAP_UNKNOWN                      # [K, H, W]
    any_known = known_mask.any(axis=0)                         # [H, W]
    # For each cell, take the first known value (any team member's truth).
    first_known_idx = np.argmax(known_mask, axis=0)            # [H, W]
    H, W = any_known.shape
    rr, cc = np.indices((H, W))
    picked = team_maps[first_known_idx, rr, cc]                # [H, W]
    return np.where(any_known, picked, MAP_UNKNOWN).astype(np.int32)


def _belief_colour_map(belief: np.ndarray) -> np.ndarray:
    """Render a [H, W] team-belief map as an [H, W, 3] RGB float image."""
    H, W = belief.shape
    img = np.tile(_BELIEF_COLOURS[MAP_UNKNOWN], (H, W, 1))
    for code, colour in _BELIEF_COLOURS.items():
        img[belief == code] = colour
    return img


# ---------------------------------------------------------------------------
# render_frame  (pure function)
# ---------------------------------------------------------------------------

def render_frame(state: "EnvState", config: "EnvConfig") -> np.ndarray:
    """Render the current environment state to an RGB uint8 numpy array.

    Parameters
    ----------
    state : EnvState
        The current environment state.
    config : EnvConfig
        The environment configuration.

    Returns
    -------
    np.ndarray
        RGB image with shape ``(H_px, W_px, 3)`` and dtype ``uint8``.
    """
    H = int(config.grid_height)
    W = int(config.grid_width)

    terrain = np.asarray(state.global_state.grid.terrain)
    explored = np.asarray(state.global_state.grid.explored)
    positions = np.asarray(state.agent_state.positions)  # [N, 2] (row, col)
    team_ids = np.asarray(state.agent_state.team_ids)    # [N]
    adjacency = np.asarray(state.global_state.graph.adjacency)  # [N, N]

    # --- figure setup ---
    dpi = 100
    fig_w = W * _CELL_PX / dpi
    fig_h = H * _CELL_PX / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # y-axis top-down
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # --- draw grid cells (greyscale: darker = more visited) ---
    max_steps = int(config.max_steps)
    cell_img = _cell_colour_map(terrain, explored, max_steps)
    ax.imshow(cell_img, origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))

    # --- draw grid lines ---
    for x in range(W + 1):
        ax.axvline(x - 0.5, color=_GRID_LINE_COLOUR, linewidth=_GRID_LINE_WIDTH, zorder=1)
    for y in range(H + 1):
        ax.axhline(y - 0.5, color=_GRID_LINE_COLOUR, linewidth=_GRID_LINE_WIDTH, zorder=1)

    # --- draw communication links ---
    N = positions.shape[0]
    segments = []
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j]:
                # positions are (row, col) -> plot as (col, row) for (x, y)
                p1 = (float(positions[i, 1]), float(positions[i, 0]))
                p2 = (float(positions[j, 1]), float(positions[j, 0]))
                segments.append([p1, p2])
    if segments:
        lc = LineCollection(segments, colors=_LINK_COLOUR, linewidths=_LINK_LINEWIDTH, zorder=2)
        ax.add_collection(lc)

    # --- draw agents ---
    for i in range(N):
        row, col = float(positions[i, 0]), float(positions[i, 1])
        tid = int(team_ids[i])
        colour = _TEAM_COLOURS.get(tid, _DEFAULT_AGENT_COLOUR)
        circle = Circle((col, row), 0.35, facecolor=colour, edgecolor="black",
                         linewidth=1.0, zorder=3)
        ax.add_patch(circle)

    # --- render to numpy array ---
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf)
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)

    return rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# render_dashboard_frame  (pure function — grid + metrics panel)
# ---------------------------------------------------------------------------

def render_dashboard_frame(
    state: "EnvState",
    config: "EnvConfig",
    blue_ever_known: "np.ndarray | None" = None,
) -> np.ndarray:
    """Render the current state as a dashboard: grid (left) + metrics (right).

    Same layout as ``EnvDashboard.update`` but as a pure function that returns
    an RGB uint8 numpy array — suitable for GIF recording.

    Parameters
    ----------
    state : EnvState
        The current environment state.
    config : EnvConfig
        The environment configuration.
    blue_ever_known : np.ndarray, optional
        Boolean [H, W] mask of cells the blue team has *ever* known across the
        episode so far. When provided, cells in this mask that are *currently*
        UNKNOWN in the blue merged belief get a translucent red overlay — this
        is the visible "fog footprint" of red's adversarial messages.
    """
    H = int(config.grid_height)
    W = int(config.grid_width)

    terrain = np.asarray(state.global_state.grid.terrain)
    explored = np.asarray(state.global_state.grid.explored)
    positions = np.asarray(state.agent_state.positions)  # [N, 2]
    team_ids = np.asarray(state.agent_state.team_ids)    # [N]
    adjacency = np.asarray(state.global_state.graph.adjacency)  # [N, N]
    degree = np.asarray(state.global_state.graph.degree)
    local_maps = np.asarray(state.agent_state.local_map)  # [N, H, W]
    step = int(state.global_state.step)
    is_connected = bool(state.global_state.graph.is_connected)
    num_components = int(state.global_state.graph.num_components)
    N = positions.shape[0]

    blue_belief = _merge_team_belief(local_maps, team_ids, target_team=0)
    red_belief = _merge_team_belief(local_maps, team_ids, target_team=1)

    # --- figure setup: global | blue belief | red belief | colorbar | metrics ---
    dpi = 100
    max_steps = int(config.max_steps)

    fig = plt.figure(figsize=(20, 6), dpi=dpi)
    gs = fig.add_gridspec(1, 5, width_ratios=[2, 2, 2, 0.08, 1.4], wspace=0.08)
    ax_global = fig.add_subplot(gs[0])
    ax_blue = fig.add_subplot(gs[1])
    ax_red = fig.add_subplot(gs[2])
    ax_cbar = fig.add_subplot(gs[3])
    ax_metrics = fig.add_subplot(gs[4])

    fig.suptitle("RedWithinBlue", fontsize=12, fontweight="bold")

    global_img = _cell_colour_map(terrain, explored, max_steps)
    blue_img = _belief_colour_map(blue_belief)
    red_img = _belief_colour_map(red_belief)

    # --- Fog footprint overlay on blue panel ---
    # Cells the blue team once knew but currently sees as UNKNOWN are exactly
    # the cells red has fogged via msgs (red->blue carries MAP_UNKNOWN). Tint
    # them with translucent red so the adversarial effect is visible.
    fogged_blue = None
    if blue_ever_known is not None:
        non_wall = (terrain != CELL_WALL)
        fogged_blue = (
            blue_ever_known.astype(bool)
            & (blue_belief == MAP_UNKNOWN)
            & non_wall
        )
        if fogged_blue.any():
            tint = np.array([0.80, 0.20, 0.20])  # adversarial red
            alpha = 0.55
            blue_img[fogged_blue] = (
                (1.0 - alpha) * blue_img[fogged_blue] + alpha * tint
            )

    blue_title = "Blue belief (merged blue local_maps)"
    if fogged_blue is not None and fogged_blue.any():
        blue_title += f"  |  fogged: {int(fogged_blue.sum())}"

    panels = [
        (ax_global, global_img, "Global (ground truth + visit heatmap)"),
        (ax_blue,   blue_img,   blue_title),
        (ax_red,    red_img,    "Red belief (red local_map)"),
    ]
    for ax, img, title in panels:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        ax.imshow(img, origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))

        for x in range(W + 1):
            ax.axvline(x - 0.5, color=_GRID_LINE_COLOUR,
                       linewidth=_GRID_LINE_WIDTH, zorder=1)
        for y in range(H + 1):
            ax.axhline(y - 0.5, color=_GRID_LINE_COLOUR,
                       linewidth=_GRID_LINE_WIDTH, zorder=1)

        segments = []
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency[i, j]:
                    p1 = (float(positions[i, 1]), float(positions[i, 0]))
                    p2 = (float(positions[j, 1]), float(positions[j, 0]))
                    segments.append([p1, p2])
        if segments:
            lc = LineCollection(segments, colors=_LINK_COLOUR,
                                linewidths=_LINK_LINEWIDTH, zorder=2)
            ax.add_collection(lc)

        for i in range(N):
            row, col = float(positions[i, 0]), float(positions[i, 1])
            tid = int(team_ids[i])
            colour = _TEAM_COLOURS.get(tid, _DEFAULT_AGENT_COLOUR)
            circle = Circle((col, row), 0.35, facecolor=colour,
                            edgecolor="black", linewidth=1.0, zorder=3)
            ax.add_patch(circle)

    # --- Visit-count colorbar ---
    # Build a vertical gradient matching _cell_colour_map mapping:
    #   0 visits  -> 0.92 (light)  ...  max_steps visits -> 0.25 (dark)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    grey_vals = 0.92 - 0.67 * gradient  # same formula as _cell_colour_map
    cbar_img = np.stack([grey_vals, grey_vals, grey_vals], axis=2)

    ax_cbar.imshow(cbar_img, aspect="auto", extent=[0, 1, max_steps, 0], origin="upper")
    ax_cbar.set_xlim(0, 1)
    ax_cbar.set_ylim(max_steps, 0)
    ax_cbar.set_xticks([])
    ax_cbar.yaxis.tick_right()
    ax_cbar.yaxis.set_label_position("right")
    ax_cbar.set_ylabel("Visits", fontsize=8, rotation=270, labelpad=12)
    # Show a few tick marks
    tick_count = 5
    ticks = np.linspace(0, max_steps, tick_count, dtype=int)
    ax_cbar.set_yticks(ticks)
    ax_cbar.tick_params(labelsize=7)
    ax_cbar.set_title("Scale", fontsize=8, pad=4)

    # --- Metrics panel ---
    ax_metrics.axis("off")
    ax_metrics.set_title("Metrics")

    non_wall_mask = terrain != CELL_WALL
    total_non_wall = int(non_wall_mask.sum())
    explored_non_wall = int(((explored > 0) & non_wall_mask).sum())
    coverage_pct = (explored_non_wall / total_non_wall * 100) if total_non_wall > 0 else 0.0

    blue_known = int(((blue_belief != MAP_UNKNOWN) & non_wall_mask).sum())
    red_known = int(((red_belief != MAP_UNKNOWN) & non_wall_mask).sum())
    blue_known_pct = blue_known / total_non_wall * 100 if total_non_wall > 0 else 0.0
    red_known_pct = red_known / total_non_wall * 100 if total_non_wall > 0 else 0.0

    lines = []
    lines.append(f"Step: {step} / {max_steps}")
    lines.append(f"Coverage: {coverage_pct:.1f}%")
    lines.append(f"Blue known: {blue_known_pct:.1f}%")
    lines.append(f"Red  known: {red_known_pct:.1f}%")
    if blue_ever_known is not None:
        ever_pct = (
            int((blue_ever_known.astype(bool) & non_wall_mask).sum())
            / total_non_wall * 100
        ) if total_non_wall > 0 else 0.0
        fog_n = int(fogged_blue.sum()) if fogged_blue is not None else 0
        lines.append(f"Blue ever-known: {ever_pct:.1f}%")
        lines.append(f"Fogged-now cells: {fog_n}")
    if is_connected:
        lines.append("Graph: Connected")
    else:
        lines.append(f"Graph: Fragmented ({num_components} components)")
    lines.append("")
    lines.append("Per-agent degree:")
    for i in range(N):
        tid = int(team_ids[i])
        team_label = "blue" if tid == 0 else ("red" if tid == 1 else f"team{tid}")
        lines.append(f"  agent {i} ({team_label}): {int(degree[i])}")

    text_block = "\n".join(lines)
    ax_metrics.text(
        0.05, 0.95, text_block,
        transform=ax_metrics.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)

    # --- render to numpy array ---
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf)
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)

    return rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# EnvDashboard  (live dashboard)
# ---------------------------------------------------------------------------

class EnvDashboard:
    """Live matplotlib dashboard with grid (left) and metrics (right) panels.

    Parameters
    ----------
    config : EnvConfig
        Environment configuration (used for layout sizing and labels).
    """

    def __init__(self, config: "EnvConfig"):
        self.config = config
        self._H = int(config.grid_height)
        self._W = int(config.grid_width)

        plt.ion()
        self.fig, (self.ax_grid, self.ax_metrics) = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]},
        )
        self.fig.suptitle("RedWithinBlue Environment Dashboard", fontsize=12)

        # Initial setup for metrics axis
        self.ax_metrics.axis("off")

        self._grid_image = None  # will hold the AxesImage for blitting

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, state: "EnvState") -> None:
        """Redraw both panels with the given state."""
        config = self.config
        H, W = self._H, self._W

        terrain = np.asarray(state.global_state.grid.terrain)
        explored = np.asarray(state.global_state.grid.explored)
        positions = np.asarray(state.agent_state.positions)
        team_ids = np.asarray(state.agent_state.team_ids)
        adjacency = np.asarray(state.global_state.graph.adjacency)
        degree = np.asarray(state.global_state.graph.degree)
        step = int(state.global_state.step)
        is_connected = bool(state.global_state.graph.is_connected)
        num_components = int(state.global_state.graph.num_components)
        N = positions.shape[0]

        # --- Grid panel ---
        ax = self.ax_grid
        ax.clear()
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title("Grid")
        ax.axis("off")

        max_steps = int(config.max_steps)
        cell_img = _cell_colour_map(terrain, explored, max_steps)
        ax.imshow(cell_img, origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))

        # Grid lines
        for x in range(W + 1):
            ax.axvline(x - 0.5, color=_GRID_LINE_COLOUR, linewidth=_GRID_LINE_WIDTH, zorder=1)
        for y in range(H + 1):
            ax.axhline(y - 0.5, color=_GRID_LINE_COLOUR, linewidth=_GRID_LINE_WIDTH, zorder=1)

        # Communication links
        segments = []
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency[i, j]:
                    p1 = (float(positions[i, 1]), float(positions[i, 0]))
                    p2 = (float(positions[j, 1]), float(positions[j, 0]))
                    segments.append([p1, p2])
        if segments:
            lc = LineCollection(segments, colors=_LINK_COLOUR, linewidths=_LINK_LINEWIDTH, zorder=2)
            ax.add_collection(lc)

        # Agents
        for i in range(N):
            row, col = float(positions[i, 0]), float(positions[i, 1])
            tid = int(team_ids[i])
            colour = _TEAM_COLOURS.get(tid, _DEFAULT_AGENT_COLOUR)
            circle = Circle((col, row), 0.35, facecolor=colour, edgecolor="black",
                             linewidth=1.0, zorder=3)
            ax.add_patch(circle)

        # --- Metrics panel ---
        ax_m = self.ax_metrics
        ax_m.clear()
        ax_m.axis("off")
        ax_m.set_title("Metrics")

        # Coverage: explored non-wall cells / total non-wall cells
        non_wall_mask = terrain != CELL_WALL
        total_non_wall = int(non_wall_mask.sum())
        explored_non_wall = int(((explored > 0) & non_wall_mask).sum())
        coverage_pct = (explored_non_wall / total_non_wall * 100) if total_non_wall > 0 else 0.0

        # Build text lines
        lines = []
        lines.append(f"Step: {step} / {int(config.max_steps)}")
        lines.append(f"Coverage: {coverage_pct:.1f}%")
        if is_connected:
            lines.append("Graph: Connected")
        else:
            lines.append(f"Graph: Fragmented ({num_components} components)")
        lines.append("")
        lines.append("Per-agent degree:")
        for i in range(N):
            tid = int(team_ids[i])
            team_label = "blue" if tid == 0 else ("red" if tid == 1 else f"team{tid}")
            lines.append(f"  agent {i} ({team_label}): {int(degree[i])}")

        text_block = "\n".join(lines)
        ax_m.text(
            0.05, 0.95, text_block,
            transform=ax_m.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
        )

        # Refresh
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        try:
            plt.pause(0.01)
        except Exception:
            pass  # non-interactive backend

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the dashboard figure."""
        plt.close(self.fig)
