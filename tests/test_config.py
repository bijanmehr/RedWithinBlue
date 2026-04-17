"""Tests for the typed config system (config.py)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from red_within_blue.types import EnvConfig
from red_within_blue.training.config import (
    EnvParams,
    ExperimentConfig,
    NetworkParams,
    RewardParams,
    TrainParams,
    from_legacy_config,
    get_stage_configs,
)


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# 1. Default ExperimentConfig has expected values
# ------------------------------------------------------------------


class TestDefaults:
    """Default-constructed ExperimentConfig matches the spec."""

    def test_env_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.env.grid_width == 10
        assert cfg.env.grid_height == 10
        assert cfg.env.num_agents == 1
        assert cfg.env.wall_density == 0.0
        assert cfg.env.max_steps == 100
        assert cfg.env.comm_radius == 3.0
        assert cfg.env.obs_radius == 1
        assert cfg.env.msg_dim == 8
        assert cfg.env.num_actions == 5
        assert cfg.env.node_feature_dim == 5

    def test_network_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.network.actor_hidden_dim == 128
        assert cfg.network.actor_num_layers == 2
        assert cfg.network.critic_hidden_dim == 128
        assert cfg.network.critic_num_layers == 2

    def test_train_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.train.method == "actor_critic"
        assert cfg.train.lr == 3e-4
        assert cfg.train.gamma == 0.90
        assert cfg.train.ent_coef == 0.0
        assert cfg.train.vf_coef == 0.5
        assert cfg.train.num_episodes == 2000
        assert cfg.train.num_seeds == 5
        assert cfg.train.eval_interval == 50
        assert cfg.train.eval_episodes == 20
        assert cfg.train.checkpoint_interval == 100

    def test_reward_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.reward.disconnect_penalty == -0.5

    def test_top_level_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.experiment_name == "default"
        assert cfg.output_dir == "experiments"
        assert cfg.enforce_connectivity is True

    def test_frozen(self):
        cfg = ExperimentConfig()
        with pytest.raises(AttributeError):
            cfg.experiment_name = "changed"  # type: ignore[misc]


# ------------------------------------------------------------------
# 2. obs_dim property
# ------------------------------------------------------------------


class TestObsDim:
    """obs_dim property computes the expected value."""

    def test_default_obs_dim(self):
        cfg = ExperimentConfig()
        # obs_radius=1 -> scan_dim = 3*3 = 9
        # total_msg = 9 + 8 = 17
        # obs_dim = 9 + 1 + 17 + 2 + 1 + 1 = 31
        assert cfg.obs_dim == 31

    def test_custom_obs_dim(self):
        """obs_dim with non-default obs_radius and msg_dim."""
        cfg = ExperimentConfig(
            env=EnvParams(obs_radius=2, msg_dim=16),
        )
        # obs_radius=2 -> scan_dim = 5*5 = 25
        # total_msg = 25 + 16 = 41
        # obs_dim = 25 + 1 + 41 + 2 + 1 + 1 = 71
        assert cfg.obs_dim == 71


# ------------------------------------------------------------------
# 3. to_env_config
# ------------------------------------------------------------------


class TestToEnvConfig:
    """to_env_config produces a valid Flax EnvConfig."""

    def test_returns_env_config(self):
        cfg = ExperimentConfig()
        env_cfg = cfg.to_env_config()
        assert isinstance(env_cfg, EnvConfig)

    def test_fields_match(self):
        cfg = ExperimentConfig(
            env=EnvParams(
                grid_width=18,
                grid_height=18,
                num_agents=4,
                wall_density=0.1,
                max_steps=256,
                comm_radius=5.0,
                obs_radius=1,
                msg_dim=8,
                num_actions=5,
                node_feature_dim=5,
            ),
        )
        env_cfg = cfg.to_env_config()
        assert env_cfg.grid_width == 18
        assert env_cfg.grid_height == 18
        assert env_cfg.num_agents == 4
        assert env_cfg.wall_density == 0.1
        assert env_cfg.max_steps == 256
        assert env_cfg.comm_radius == 5.0
        assert env_cfg.obs_radius == 1
        assert env_cfg.msg_dim == 8
        assert env_cfg.num_actions == 5
        assert env_cfg.node_feature_dim == 5

    def test_default_fields_match(self):
        cfg = ExperimentConfig()
        env_cfg = cfg.to_env_config()
        assert env_cfg.grid_width == cfg.env.grid_width
        assert env_cfg.grid_height == cfg.env.grid_height
        assert env_cfg.num_agents == cfg.env.num_agents
        assert env_cfg.max_steps == cfg.env.max_steps
        assert env_cfg.comm_radius == cfg.env.comm_radius
        assert env_cfg.obs_radius == cfg.env.obs_radius
        assert env_cfg.msg_dim == cfg.env.msg_dim
        assert env_cfg.wall_density == cfg.env.wall_density
        assert env_cfg.node_feature_dim == cfg.env.node_feature_dim


# ------------------------------------------------------------------
# 4. from_yaml (new format)
# ------------------------------------------------------------------


class TestFromYaml:
    """from_yaml loads new-format YAML and overrides defaults."""

    def _write_yaml(self, content: str) -> str:
        """Write content to a temp YAML file, return path."""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, content.encode())
        os.close(fd)
        return path

    def test_empty_yaml_gives_defaults(self):
        path = self._write_yaml("")
        try:
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg == ExperimentConfig()
        finally:
            os.unlink(path)

    def test_partial_override(self):
        yaml_str = """\
experiment_name: my_run
env:
  grid_width: 20
  num_agents: 4
train:
  lr: 1.0e-3
  method: reinforce
"""
        path = self._write_yaml(yaml_str)
        try:
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg.experiment_name == "my_run"
            assert cfg.env.grid_width == 20
            assert cfg.env.num_agents == 4
            # un-overridden defaults preserved
            assert cfg.env.grid_height == 10
            assert cfg.env.obs_radius == 1
            assert cfg.train.lr == 1e-3
            assert cfg.train.method == "reinforce"
            # un-overridden train defaults preserved
            assert cfg.train.gamma == 0.90
        finally:
            os.unlink(path)

    def test_full_override(self):
        yaml_str = """\
experiment_name: full
output_dir: /tmp/out
enforce_connectivity: false
env:
  grid_width: 32
  grid_height: 32
  num_agents: 8
  wall_density: 0.2
  max_steps: 512
  comm_radius: 7.0
  obs_radius: 2
  msg_dim: 16
  num_actions: 4
  node_feature_dim: 10
network:
  actor_hidden_dim: 256
  actor_num_layers: 3
  critic_hidden_dim: 256
  critic_num_layers: 3
train:
  method: baseline
  lr: 1.0e-4
  gamma: 0.95
  ent_coef: 0.01
  vf_coef: 0.25
  num_episodes: 5000
  num_seeds: 10
  eval_interval: 100
  eval_episodes: 50
  checkpoint_interval: 200
reward:
  disconnect_penalty: -1.0
"""
        path = self._write_yaml(yaml_str)
        try:
            cfg = ExperimentConfig.from_yaml(path)
            assert cfg.experiment_name == "full"
            assert cfg.output_dir == "/tmp/out"
            assert cfg.enforce_connectivity is False
            assert cfg.env.grid_width == 32
            assert cfg.env.obs_radius == 2
            assert cfg.network.actor_hidden_dim == 256
            assert cfg.train.method == "baseline"
            assert cfg.train.ent_coef == 0.01
            assert cfg.reward.disconnect_penalty == -1.0
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# 5. from_legacy_config
# ------------------------------------------------------------------


class TestFromLegacyConfig:
    """from_legacy_config loads the old multi-stage YAML format."""

    def test_loads_legacy_defaults(self):
        """from_legacy_config with built-in defaults produces valid config."""
        import yaml
        import tempfile, os
        # Write a minimal legacy YAML (empty = pure defaults)
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, b"{}")
        os.close(fd)
        try:
            cfg = from_legacy_config(path)
            assert cfg.env.grid_width == 10
            assert cfg.env.num_agents == 1
            assert cfg.obs_dim == 31
        finally:
            os.unlink(path)


# ------------------------------------------------------------------
# 6. get_stage_configs
# ------------------------------------------------------------------


class TestGetStageConfigs:
    """get_stage_configs returns three ExperimentConfigs."""

    def test_defaults_three_stages(self):
        s1, s2, s3 = get_stage_configs()
        assert s1.env.num_agents == 1
        assert s2.env.num_agents == 2
        assert s3.env.num_agents == 4

    def test_stage3_grid(self):
        _, _, s3 = get_stage_configs()
        assert s3.env.grid_width == 18
        assert s3.env.grid_height == 18
        assert s3.env.wall_density == 0.1
        assert s3.env.max_steps == 256
        assert s3.env.comm_radius == 5.0

    def test_obs_dim_consistent(self):
        """All stages share the same obs_dim (needed for weight transfer)."""
        s1, s2, s3 = get_stage_configs()
        assert s1.obs_dim == s2.obs_dim == s3.obs_dim == 31

    def test_experiment_names(self):
        s1, s2, s3 = get_stage_configs()
        assert s1.experiment_name == "stage1"
        assert s2.experiment_name == "stage2"
        assert s3.experiment_name == "stage3"

    def test_lr_values(self):
        s1, s2, s3 = get_stage_configs()
        assert s1.train.lr == 3e-4
        assert s2.train.lr == 1.5e-4
        assert s3.train.lr == 1.5e-4

    def test_to_env_config_per_stage(self):
        s1, s2, s3 = get_stage_configs()
        for cfg in (s1, s2, s3):
            env_cfg = cfg.to_env_config()
            assert isinstance(env_cfg, EnvConfig)
            assert env_cfg.num_agents == cfg.env.num_agents
