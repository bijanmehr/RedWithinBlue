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
    num_red_agents: int = 0  # adversarial agents (last-N indices)
    wall_density: float = 0.0
    max_steps: int = 100
    comm_radius: float = 3.0
    # Sensing / survey split. See EnvConfig in types.py for full semantics.
    # obs_radius is the legacy single knob; view_radius / survey_radius inherit
    # from it when left at -1. local_obs swaps the global H·W seen mask for a
    # view-sized window of the agent's own local_map.
    obs_radius: int = 1
    view_radius: int = -1
    survey_radius: int = -1
    local_obs: bool = False
    num_actions: int = 5
    node_feature_dim: int = 5
    red_blocks_blue: bool = False  # Phase 2: red cells are walls for blue
    center_spawn: bool = False     # spawn agents in a Gaussian cluster around grid center
    normalize_uid: bool = False    # divide uid feature by num_agents in obs tail (range -> (0,1])
    # Disconnect-grace mechanism (soft connectivity constraint). See EnvConfig
    # in types.py for full semantics. `disconnect_grace=0` keeps the legacy
    # hard guardrail; any positive value disables the guardrail and activates
    # the per-agent grace window (episode ends on expiry with fail penalty).
    disconnect_grace: int = 0
    disconnect_fail_penalty: float = 0.0
    disconnect_mode: str = "per_agent"  # "per_agent" | "team"


@dataclass(frozen=True)
class NetworkParams:
    actor_hidden_dim: int = 128
    actor_num_layers: int = 2
    critic_hidden_dim: int = 128
    critic_num_layers: int = 2
    # Hidden-layer activation for actor and critic MLPs. One of
    # "relu", "gelu", "tanh", "silu". Applied after every Dense except the
    # output head.
    activation: str = "relu"


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
    red_policy: str = "shared"  # "shared" | "random" | "joint"
    red_hidden_dim: int = 256  # joint-red trunk width (used when red_policy == "joint")
    red_num_layers: int = 2    # joint-red trunk depth
    red_pretrain_episodes: int = 0  # episodes training red vs. frozen blue before joint phase
    # epsilon-greedy exploration override. Each step, every agent's action is
    # replaced with a uniform-random action with probability ``epsilon``.
    # Gradient stays on-policy w.r.t. pi (log_prob from the actor); the small
    # distribution shift is absorbed as off-policy noise. Fine for eps ≲ 0.1.
    epsilon: float = 0.0
    # Optional linear anneal targets. If a *_final value is < 0 (sentinel), the
    # corresponding quantity is held constant at its initial value. Otherwise
    # the value decays linearly from initial to final across num_episodes so
    # the policy consolidates toward argmax by end of training (fix for the
    # stochastic-vs-argmax reward gap observed on pair-cooperate-coop).
    epsilon_final: float = -1.0
    ent_coef_final: float = -1.0
    # Fraction of training over which linear anneal completes. After this
    # fraction, values are held at their *_final targets. Default 1.0 = anneal
    # over all num_episodes. Smaller values (e.g. 0.5) give the policy more
    # training at the final (low) epsilon/ent_coef so it can commit to argmax.
    anneal_end_frac: float = 1.0
    # Global-norm gradient clip applied before each Adam step. 0.0 disables
    # clipping. Non-zero values prevent runaway critic/actor updates — the
    # standard remedy for actor-critic training where the value loss can
    # occasionally produce exploding gradients that destabilise the policy.
    grad_clip: float = 0.0


@dataclass(frozen=True)
class RewardParams:
    disconnect_penalty: float = -0.5
    isolation_weight: float = 0.0          # per-agent penalty when degree==0
    cooperative_weight: float = 0.0        # bonus when a connected neighbour discovers a new cell
    revisit_weight: float = 0.0            # penalty for stepping on an already-explored cell
    terminal_bonus_scale: float = 0.0      # coverage-fraction bonus at episode end (0 = disabled)
    terminal_bonus_divide: bool = True     # if True, split terminal bonus evenly across agents
    spread_weight: float = 0.0             # per-agent bonus ∝ mean L1 distance to teammates (drives spatial spread)
    # Fog-of-war potential shaping. Each step, per-agent reward gets
    #   +w * (prev_dist_to_nearest_unknown - new_dist_to_nearest_unknown)
    # where the distance is L1 from the agent's position to the nearest
    # never-visited non-wall cell. Pulls the team *toward* the unknown as
    # a group: every agent drifts to its own nearest frontier, which
    # naturally distributes the team across the uncertainty field.
    fog_potential_weight: float = 0.0


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
    # If the warm-start source was trained with a different num_agents than
    # this config, set this to the source's num_agents. The CTDE critic loader
    # will then split the source kernel into source_num_agents per-agent blocks
    # and tile them up to the target num_agents (requires target N to be an
    # integer multiple of source N). When None, defaults to env.num_agents.
    warm_start_source_num_agents: Optional[int] = None

    # ------------------------------------------------------------------
    # Computed property
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Observation dimension (must match env.py formula).

        Layout (local_obs=False, default):
            local_scan (view_d²) + global_seen_mask (H·W) + map_fraction + pos(2) + uid + team
        Layout (local_obs=True):
            local_scan (view_d²) + local_seen (view_d²)    + map_fraction + pos(2) + uid + team

        All neighbour information is fused into the local_map (and therefore
        the seen field) by :func:`update_local_maps_with_comm`.
        """
        view_r = self.env.view_radius if self.env.view_radius >= 0 else self.env.obs_radius
        scan_dim = (2 * view_r + 1) ** 2
        if self.env.local_obs:
            seen_dim = scan_dim
        else:
            seen_dim = self.env.grid_height * self.env.grid_width
        return scan_dim + seen_dim + 1 + 2 + 1 + 1

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
        for key in ("experiment_name", "output_dir", "enforce_connectivity", "warm_start", "warm_start_source_num_agents"):
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
            num_red_agents=self.env.num_red_agents,
            num_actions=self.env.num_actions,
            comm_radius=self.env.comm_radius,
            obs_radius=self.env.obs_radius,
            view_radius=self.env.view_radius,
            survey_radius=self.env.survey_radius,
            local_obs=self.env.local_obs,
            wall_density=self.env.wall_density,
            node_feature_dim=self.env.node_feature_dim,
            red_blocks_blue=self.env.red_blocks_blue,
            center_spawn=self.env.center_spawn,
            normalize_uid=self.env.normalize_uid,
            disconnect_grace=int(self.env.disconnect_grace),
            disconnect_fail_penalty=float(self.env.disconnect_fail_penalty),
            disconnect_mode=1 if str(self.env.disconnect_mode).lower() == "team" else 0,
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
