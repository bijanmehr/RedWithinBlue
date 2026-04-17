"""Typed experiment configuration using frozen dataclasses.

Replaces the dict-based ``configs.py``.  Provides:

* ``ExperimentConfig`` — top-level frozen dataclass with nested env/network/
  train/reward groups.
* ``from_yaml`` class method — load a *new-format* YAML file.
* ``from_legacy_config`` — load the old multi-stage YAML format into a
  single ``ExperimentConfig``.
* ``get_stage_configs`` — convenience that returns three
  ``ExperimentConfig`` instances (stages 1-3) from a legacy YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from red_within_blue.types import EnvConfig


# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvParams:
    grid_width: int = 10
    grid_height: int = 10
    num_agents: int = 1
    wall_density: float = 0.0
    max_steps: int = 100
    comm_radius: float = 3.0
    obs_radius: int = 1
    msg_dim: int = 8
    num_actions: int = 5
    node_feature_dim: int = 5


@dataclass(frozen=True)
class NetworkParams:
    actor_hidden_dim: int = 128
    actor_num_layers: int = 2
    critic_hidden_dim: int = 128
    critic_num_layers: int = 2


@dataclass(frozen=True)
class TrainParams:
    method: str = "actor_critic"  # "reinforce" | "baseline" | "actor_critic"
    lr: float = 3e-4
    gamma: float = 0.90
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    num_episodes: int = 2000
    num_seeds: int = 5
    eval_interval: int = 50
    eval_episodes: int = 20
    checkpoint_interval: int = 100


@dataclass(frozen=True)
class RewardParams:
    disconnect_penalty: float = -0.5


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvParams = field(default_factory=EnvParams)
    network: NetworkParams = field(default_factory=NetworkParams)
    train: TrainParams = field(default_factory=TrainParams)
    reward: RewardParams = field(default_factory=RewardParams)
    experiment_name: str = "default"
    output_dir: str = "experiments"
    enforce_connectivity: bool = True
    warm_start: Optional[str] = None  # path to checkpoint.npz for transfer

    # ------------------------------------------------------------------
    # Computed property
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Observation dimension (must match env.py formula)."""
        scan_dim = (2 * self.env.obs_radius + 1) ** 2
        total_msg = scan_dim + self.env.msg_dim
        return scan_dim + 1 + total_msg + 2 + 1 + 1

    # ------------------------------------------------------------------
    # YAML loader (new format)
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load a new-format YAML file and merge with defaults.

        The YAML may contain top-level keys ``env``, ``network``, ``train``,
        ``reward``, ``experiment_name``, ``output_dir``, and
        ``enforce_connectivity``.  Missing keys fall back to defaults.
        """
        import yaml

        with open(path) as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        env_kwargs = raw.get("env", {})
        network_kwargs = raw.get("network", {})
        train_kwargs = raw.get("train", {})
        reward_kwargs = raw.get("reward", {})

        top_kwargs: Dict[str, Any] = {}
        for key in ("experiment_name", "output_dir", "enforce_connectivity", "warm_start"):
            if key in raw:
                top_kwargs[key] = raw[key]

        return cls(
            env=EnvParams(**env_kwargs),
            network=NetworkParams(**network_kwargs),
            train=TrainParams(**train_kwargs),
            reward=RewardParams(**reward_kwargs),
            **top_kwargs,
        )

    # ------------------------------------------------------------------
    # Bridge to Flax EnvConfig
    # ------------------------------------------------------------------

    def to_env_config(self) -> EnvConfig:
        """Return a ``red_within_blue.types.EnvConfig`` (Flax struct)."""
        return EnvConfig(
            grid_width=self.env.grid_width,
            grid_height=self.env.grid_height,
            max_steps=self.env.max_steps,
            num_agents=self.env.num_agents,
            num_actions=self.env.num_actions,
            comm_radius=self.env.comm_radius,
            obs_radius=self.env.obs_radius,
            msg_dim=self.env.msg_dim,
            wall_density=self.env.wall_density,
            node_feature_dim=self.env.node_feature_dim,
        )


# ---------------------------------------------------------------------------
# Legacy format helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


_LEGACY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "network": dict(
        actor_hidden_dim=128,
        actor_num_layers=2,
        critic_hidden_dim=128,
        critic_num_layers=2,
    ),
    "comm": dict(
        obs_radius=1,
        msg_dim=8,
        node_feature_dim=5,
    ),
    "rewards": dict(
        disconnect_penalty=-0.5,
    ),
    "training": dict(
        gamma=0.99,
        num_seeds=5,
        eval_interval=50,
        eval_episodes=20,
        checkpoint_interval=100,
        num_episodes=2000,
    ),
    "stage1": dict(
        grid_width=10, grid_height=10, num_agents=1,
        wall_density=0.0, max_steps=100, comm_radius=3.0, lr=3e-4,
    ),
    "stage2": dict(
        grid_width=10, grid_height=10, num_agents=2,
        wall_density=0.0, max_steps=100, comm_radius=3.0, lr=1.5e-4,
    ),
    "stage3": dict(
        grid_width=18, grid_height=18, num_agents=4,
        wall_density=0.1, max_steps=256, comm_radius=5.0, lr=1.5e-4,
    ),
}


def _build_experiment_from_legacy(
    cfg: Dict[str, Any],
    stage_key: str,
    experiment_name: str | None = None,
) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from the merged legacy dict."""
    comm = cfg["comm"]
    training = cfg["training"]
    stage = cfg[stage_key]
    network = cfg["network"]
    rewards = cfg["rewards"]

    env = EnvParams(
        grid_width=stage["grid_width"],
        grid_height=stage["grid_height"],
        num_agents=stage["num_agents"],
        wall_density=stage["wall_density"],
        max_steps=stage["max_steps"],
        comm_radius=stage["comm_radius"],
        obs_radius=comm["obs_radius"],
        msg_dim=comm["msg_dim"],
        num_actions=5,
        node_feature_dim=comm["node_feature_dim"],
    )

    net = NetworkParams(
        actor_hidden_dim=network["actor_hidden_dim"],
        actor_num_layers=network["actor_num_layers"],
        critic_hidden_dim=network["critic_hidden_dim"],
        critic_num_layers=network["critic_num_layers"],
    )

    train = TrainParams(
        lr=stage["lr"],
        gamma=training["gamma"],
        num_seeds=training["num_seeds"],
        eval_interval=training["eval_interval"],
        eval_episodes=training["eval_episodes"],
        checkpoint_interval=training["checkpoint_interval"],
        num_episodes=training["num_episodes"],
    )

    reward = RewardParams(
        disconnect_penalty=rewards["disconnect_penalty"],
    )

    return ExperimentConfig(
        env=env,
        network=net,
        train=train,
        reward=reward,
        experiment_name=experiment_name or stage_key,
    )


def from_legacy_config(path: str) -> ExperimentConfig:
    """Load a legacy multi-stage YAML and return the stage-1 config.

    Parameters
    ----------
    path : str
        Path to a YAML file in the old format (keys: network, comm, rewards,
        training, stage1, stage2, stage3).

    Returns
    -------
    ExperimentConfig
        The stage-1 experiment configuration.
    """
    import yaml

    with open(path) as f:
        user: Dict[str, Any] = yaml.safe_load(f) or {}

    merged = _deep_merge(
        {k: dict(v) for k, v in _LEGACY_DEFAULTS.items()},
        user,
    )
    return _build_experiment_from_legacy(merged, "stage1")


def get_stage_configs(
    path: str | None = None,
) -> Tuple[ExperimentConfig, ExperimentConfig, ExperimentConfig]:
    """Return ``(stage1, stage2, stage3)`` experiment configs.

    Parameters
    ----------
    path : str or None
        Path to a legacy YAML config.  ``None`` uses built-in defaults.

    Returns
    -------
    tuple of three ``ExperimentConfig`` instances.
    """
    import yaml

    if path is not None:
        with open(path) as f:
            user: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        user = {}

    merged = _deep_merge(
        {k: dict(v) for k, v in _LEGACY_DEFAULTS.items()},
        user,
    )

    return (
        _build_experiment_from_legacy(merged, "stage1", "stage1"),
        _build_experiment_from_legacy(merged, "stage2", "stage2"),
        _build_experiment_from_legacy(merged, "stage3", "stage3"),
    )
