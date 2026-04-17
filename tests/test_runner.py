"""Tests for the runner CLI (red_within_blue.training.runner)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from red_within_blue.training.runner import (
    parse_args,
    load_config_with_overrides,
    run_training,
    save_results,
    main,
)
from red_within_blue.training.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_yaml(content: str) -> str:
    """Write content to a temp YAML file, return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.write(fd, content.encode())
    os.close(fd)
    return path


_TINY_YAML = """\
experiment_name: tiny_test
env:
  grid_width: 4
  grid_height: 4
  num_agents: 1
  wall_density: 0.0
  max_steps: 10
  comm_radius: 3.0
network:
  actor_hidden_dim: 16
  actor_num_layers: 1
  critic_hidden_dim: 16
  critic_num_layers: 1
train:
  method: actor_critic
  lr: 3.0e-3
  gamma: 0.9
  num_episodes: 2
  num_seeds: 1
enforce_connectivity: false
"""


# ---------------------------------------------------------------------------
# Test: CLI arg parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """parse_args correctly interprets CLI flags."""

    def test_config_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_config_only(self):
        args = parse_args(["--config", "foo.yaml"])
        assert args.config == "foo.yaml"
        assert args.output_dir is None
        assert args.num_seeds is None

    def test_all_flags(self):
        args = parse_args([
            "--config", "bar.yaml",
            "--output-dir", "/tmp/out",
            "--num-seeds", "7",
        ])
        assert args.config == "bar.yaml"
        assert args.output_dir == "/tmp/out"
        assert args.num_seeds == 7


# ---------------------------------------------------------------------------
# Test: config loading + overrides
# ---------------------------------------------------------------------------


class TestLoadConfigWithOverrides:
    """load_config_with_overrides applies CLI overrides to YAML config."""

    def test_no_overrides(self):
        path = _write_yaml(_TINY_YAML)
        try:
            args = parse_args(["--config", path])
            config = load_config_with_overrides(args)
            assert config.experiment_name == "tiny_test"
            assert config.train.num_seeds == 1
            assert config.output_dir == "experiments"
        finally:
            os.unlink(path)

    def test_output_dir_override(self):
        path = _write_yaml(_TINY_YAML)
        try:
            args = parse_args(["--config", path, "--output-dir", "/tmp/custom"])
            config = load_config_with_overrides(args)
            assert config.output_dir == "/tmp/custom"
            # Other fields unchanged
            assert config.experiment_name == "tiny_test"
        finally:
            os.unlink(path)

    def test_num_seeds_override(self):
        path = _write_yaml(_TINY_YAML)
        try:
            args = parse_args(["--config", path, "--num-seeds", "3"])
            config = load_config_with_overrides(args)
            assert config.train.num_seeds == 3
            # Other train fields unchanged
            assert config.train.lr == 3e-3
            assert config.train.num_episodes == 2
        finally:
            os.unlink(path)

    def test_both_overrides(self):
        path = _write_yaml(_TINY_YAML)
        try:
            args = parse_args([
                "--config", path,
                "--output-dir", "/tmp/both",
                "--num-seeds", "10",
            ])
            config = load_config_with_overrides(args)
            assert config.output_dir == "/tmp/both"
            assert config.train.num_seeds == 10
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: training + save (integration, tiny config)
# ---------------------------------------------------------------------------


class TestRunTrainingAndSave:
    """End-to-end test: train with a tiny config, save results, verify files."""

    def test_single_seed_run(self):
        path = _write_yaml(_TINY_YAML)
        try:
            args = parse_args(["--config", path])
            config = load_config_with_overrides(args)

            actor_params, critic_params, metrics = run_training(config)

            assert actor_params is not None
            assert critic_params is not None  # actor_critic method
            assert "loss" in metrics
            assert "total_reward" in metrics
            # Single seed, 2 episodes -> shape [2]
            assert np.asarray(metrics["loss"]).shape == (2,)
        finally:
            os.unlink(path)

    def test_save_results_creates_files(self):
        path = _write_yaml(_TINY_YAML)
        try:
            args = parse_args(["--config", path])
            config = load_config_with_overrides(args)

            with tempfile.TemporaryDirectory() as tmpdir:
                config = ExperimentConfig.from_yaml(path)
                import dataclasses
                config = dataclasses.replace(config, output_dir=tmpdir)

                actor_params, critic_params, metrics = run_training(config)
                out_dir = save_results(config, actor_params, critic_params, metrics)

                assert out_dir.exists()
                assert (out_dir / "checkpoint.npz").exists()
                assert (out_dir / "metrics.npz").exists()

                # Verify metrics file content
                loaded = np.load(str(out_dir / "metrics.npz"))
                assert "loss" in loaded
                assert "total_reward" in loaded
                assert loaded["loss"].shape == (2,)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: main() end-to-end
# ---------------------------------------------------------------------------


class TestMain:
    """main() runs without error on a tiny config."""

    def test_main_tiny(self, capsys):
        path = _write_yaml(_TINY_YAML)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                main(["--config", path, "--output-dir", tmpdir])
                captured = capsys.readouterr()
                assert "tiny_test" in captured.out
                assert "Final loss" in captured.out
                assert "Final reward" in captured.out

                # Verify output files were created
                out_dir = Path(tmpdir) / "tiny_test"
                assert (out_dir / "checkpoint.npz").exists()
                assert (out_dir / "metrics.npz").exists()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: example YAML configs load correctly
# ---------------------------------------------------------------------------


class TestExampleConfigs:
    """The bundled YAML configs in configs/ load and have expected values."""

    @pytest.mark.parametrize("name,agents,grid,lr", [
        ("solo-explore", 1, 10, 3e-4),
        ("pair-cooperate", 2, 10, 1.5e-4),
        ("team-coordinate", 4, 18, 1.5e-4),
    ])
    def test_config_loads(self, name, agents, grid, lr):
        yaml_path = _PROJECT_ROOT / "configs" / f"{name}.yaml"
        if not yaml_path.exists():
            pytest.skip(f"{yaml_path} not found")

        config = ExperimentConfig.from_yaml(str(yaml_path))
        assert config.experiment_name == name
        assert config.env.num_agents == agents
        assert config.env.grid_width == grid
        assert config.env.grid_height == grid
        assert config.train.lr == lr
        assert config.train.method == "actor_critic"
        assert config.train.num_episodes == 2000
