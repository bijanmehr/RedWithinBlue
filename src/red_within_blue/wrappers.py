"""Environment wrappers for evaluation, logging, and trajectory recording.

TrajectoryWrapper wraps a MultiAgentEnv (specifically GridCommEnv) and records
full trajectories during execution.  It collects Python-side lists of numpy
arrays and is **not** JIT-compatible -- use it for evaluation / logging, not
training.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp


class TrajectoryWrapper:
    """Records observations, actions, rewards, and dones at every step.

    Parameters
    ----------
    env : MultiAgentEnv
        The underlying environment (e.g. ``GridCommEnv``).
    save_dir : str or None
        Default directory for ``save_trajectory``.  If *None*, files are saved
        to the current working directory.
    """

    def __init__(self, env, save_dir: Optional[str] = None):
        self.env = env
        self.save_dir = save_dir
        self._buffer: List[dict] = []

        # Delegate common attributes so the wrapper looks like an env.
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, key):
        """Reset the wrapped env and start a fresh trajectory buffer."""
        self._buffer = []
        obs, state = self.env.reset(key)
        self._record_step(obs=obs, state=state, actions=None, rewards=None, dones=None)
        return obs, state

    def step(self, key, state, actions):
        """Take one env step (via ``step_env``) and record the transition.

        We call ``step_env`` rather than ``step`` to avoid the JIT-compiled
        auto-reset wrapper in the base ``MultiAgentEnv``.
        """
        obs, new_state, rewards, dones, info = self.env.step_env(key, state, actions)
        self._record_step(obs=obs, state=new_state, actions=actions, rewards=rewards, dones=dones)
        return obs, new_state, rewards, dones, info

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _record_step(self, **kwargs):
        """Snapshot the supplied keyword arguments into the buffer.

        Dicts (obs, actions, rewards, dones) are flattened into
        ``"key/subkey"`` numpy arrays.  Other values (like EnvState) are kept
        raw so that ``get_trajectory`` can still inspect them.
        """
        snapshot: dict = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, dict):
                for dk, dv in v.items():
                    snapshot[f"{k}/{dk}"] = np.asarray(dv)
            else:
                # Complex objects like EnvState -- store raw for get_trajectory
                snapshot[k] = v
        self._buffer.append(snapshot)

    # ------------------------------------------------------------------
    # Trajectory access / persistence
    # ------------------------------------------------------------------

    def get_trajectory(self) -> List[dict]:
        """Return the raw buffer (list of snapshot dicts)."""
        return list(self._buffer)

    def save_trajectory(self, name: str) -> str:
        """Save the recorded trajectory to a compressed ``.npz`` file.

        Only numpy-serialisable entries (obs, actions, rewards, dones) are
        persisted.  Complex objects (EnvState) are silently skipped.

        Parameters
        ----------
        name : str
            Base name for the file (without extension).

        Returns
        -------
        str
            Absolute path to the saved file.
        """
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"{name}.npz")
        else:
            path = f"{name}.npz"

        save_dict: Dict[str, np.ndarray] = {}
        for step_idx, snapshot in enumerate(self._buffer):
            for k, v in snapshot.items():
                if isinstance(v, np.ndarray):
                    save_dict[f"step_{step_idx}/{k}"] = v
                elif isinstance(v, (int, float)):
                    save_dict[f"step_{step_idx}/{k}"] = np.asarray(v)
                elif isinstance(v, jnp.ndarray):
                    save_dict[f"step_{step_idx}/{k}"] = np.asarray(v)
                # Skip non-serialisable objects (EnvState, etc.)

        np.savez_compressed(path, **save_dict)
        return path

    @staticmethod
    def load_trajectory(path: str) -> dict:
        """Load a trajectory previously saved with ``save_trajectory``.

        Returns a flat dict mapping ``"step_<i>/<key>"`` to numpy arrays.
        """
        data = np.load(path, allow_pickle=True)
        return dict(data)
