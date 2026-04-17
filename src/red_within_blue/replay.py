"""Post-experiment trajectory playback."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class ReplayPlayer:
    """Load and visually replay saved trajectories.

    Usage::

        player = ReplayPlayer("experiments/run_001/trajectories/episode_42.npz", config)
        player.play()             # auto-play
        player.play(speed=2.0)    # 2x speed
        player.step_forward()     # advance one step
        player.step_back()        # go back one step
        player.jump_to(step=100)  # jump to specific timestep

    Parameters
    ----------
    trajectory_path : str or Path
        Path to a ``.npz`` file saved by :class:`TrajectoryWrapper`.
    config : EnvConfig, optional
        Environment config (used for rendering).
    """

    def __init__(self, trajectory_path, config=None):
        self.path = Path(trajectory_path)
        self.config = config
        self._data = np.load(str(self.path), allow_pickle=True)
        self._current_step = 0

        # Discover how many steps are recorded
        self._num_steps = self._count_steps()

    def _count_steps(self) -> int:
        """Count the number of recorded steps based on key prefixes."""
        step_indices = set()
        for key in self._data.keys():
            if key.startswith("step_"):
                parts = key.split("/")
                idx_str = parts[0].replace("step_", "")
                try:
                    step_indices.add(int(idx_str))
                except ValueError:
                    pass
        return len(step_indices) if step_indices else 0

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def current_step(self) -> int:
        return self._current_step

    def step_forward(self) -> dict:
        """Advance one step and return the step data."""
        if self._current_step < self._num_steps - 1:
            self._current_step += 1
        return self.get_step_data(self._current_step)

    def step_back(self) -> dict:
        """Go back one step and return the step data."""
        if self._current_step > 0:
            self._current_step -= 1
        return self.get_step_data(self._current_step)

    def jump_to(self, step: int) -> dict:
        """Jump to a specific step."""
        self._current_step = max(0, min(step, self._num_steps - 1))
        return self.get_step_data(self._current_step)

    def get_step_data(self, step: int) -> dict:
        """Extract all data for a given step."""
        prefix = f"step_{step}/"
        result = {}
        for key in self._data.keys():
            if key.startswith(prefix):
                field = key[len(prefix):]
                result[field] = self._data[key]
        return result

    def get_all_keys(self) -> List[str]:
        """Return all keys in the trajectory file."""
        return list(self._data.keys())

    def play(self, speed: float = 1.0, render_fn=None) -> None:
        """Auto-play the trajectory.

        Parameters
        ----------
        speed : float
            Playback speed multiplier (1.0 = real time, 2.0 = double speed).
        render_fn : callable, optional
            Function that takes step_data dict and renders it.
            If None, just prints step info.
        """
        import time

        delay = max(0.01, 0.1 / speed)

        for step in range(self._num_steps):
            self._current_step = step
            data = self.get_step_data(step)

            if render_fn is not None:
                render_fn(data)
            else:
                print(f"Step {step}/{self._num_steps - 1}", end="\r")

            time.sleep(delay)

        print()  # newline after progress

    def export_frames(self, render_fn, output_dir: str = "frames") -> List[str]:
        """Export each step as an image frame.

        Parameters
        ----------
        render_fn : callable
            ``(step_data) -> np.ndarray`` returning an RGB image.
        output_dir : str
            Directory to save frames to.

        Returns
        -------
        list of str
            Paths to saved frame images.
        """
        from pathlib import Path as P
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = P(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths = []
        for step in range(self._num_steps):
            data = self.get_step_data(step)
            frame = render_fn(data)

            frame_path = out / f"frame_{step:04d}.png"
            if isinstance(frame, np.ndarray):
                plt.imsave(str(frame_path), frame)
            paths.append(str(frame_path))

        return paths
