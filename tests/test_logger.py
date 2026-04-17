"""Tests for ExperimentLogger."""

import json

import numpy as np
import pytest

from red_within_blue.logger import ExperimentLogger, list_experiments, load_experiment


def test_logger_creates_directory(tmp_path):
    logger = ExperimentLogger(base_dir=str(tmp_path), experiment_name="test_run")
    assert logger.experiment_dir.exists()
    assert (logger.experiment_dir / "trajectories").exists()
    assert (logger.experiment_dir / "checkpoints").exists()


def test_log_config(tmp_path):
    logger = ExperimentLogger(base_dir=str(tmp_path), experiment_name="cfg_test")
    logger.log_config({"grid_width": 32, "num_agents": 4, "comm_radius": 5.0})

    config_path = logger.experiment_dir / "config.json"
    assert config_path.exists()

    with open(config_path) as f:
        data = json.load(f)
    assert data["grid_width"] == 32
    assert data["num_agents"] == 4
    assert data["comm_radius"] == 5.0


def test_log_hyperparams(tmp_path):
    logger = ExperimentLogger(base_dir=str(tmp_path), experiment_name="hp_test")
    logger.log_hyperparams({"lr": 3e-4, "gamma": 0.99, "batch_size": 256})

    hp_path = logger.experiment_dir / "hyperparams.json"
    assert hp_path.exists()

    with open(hp_path) as f:
        data = json.load(f)
    assert data["lr"] == 3e-4
    assert data["gamma"] == 0.99


def test_log_metrics_append(tmp_path):
    logger = ExperimentLogger(base_dir=str(tmp_path), experiment_name="met_test")
    logger.log_metrics(step=0, metrics={"coverage": 0.1, "reward": 1.0})
    logger.log_metrics(step=10, metrics={"coverage": 0.5, "reward": 5.0})
    logger.log_metrics(step=20, metrics={"coverage": 0.9, "reward": 8.0})

    # Read JSONL
    records = []
    with open(logger.metrics_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    assert len(records) == 3
    assert records[0]["step"] == 0
    assert records[1]["coverage"] == 0.5
    assert records[2]["reward"] == 8.0


def test_list_experiments(tmp_path):
    # Create two experiments
    l1 = ExperimentLogger(base_dir=str(tmp_path), experiment_name="exp_a")
    l1.log_config({"width": 8})
    l1.log_metrics(step=0, metrics={"r": 1.0})
    l1.close()

    l2 = ExperimentLogger(base_dir=str(tmp_path), experiment_name="exp_b")
    l2.log_config({"width": 16})
    l2.close()

    exps = list_experiments(str(tmp_path))
    assert len(exps) == 2
    # Sorted by name (timestamps ensure order)
    assert "exp_a" in exps[0]["id"]
    assert "exp_b" in exps[1]["id"]
    assert "config" in exps[0]
    assert exps[0]["num_metrics_records"] == 1


def test_load_experiment(tmp_path):
    logger = ExperimentLogger(base_dir=str(tmp_path), experiment_name="load_test")
    logger.log_config({"grid_width": 32})
    logger.log_hyperparams({"lr": 1e-3})
    logger.log_metrics(step=0, metrics={"val": 0.0})
    logger.log_metrics(step=5, metrics={"val": 0.5})
    exp_id = logger.experiment_id
    logger.close()

    loaded = load_experiment(str(tmp_path), exp_id)
    assert loaded["config"]["grid_width"] == 32
    assert loaded["hyperparams"]["lr"] == 1e-3
    assert len(loaded["metrics"]) == 2
    assert loaded["metrics"][1]["val"] == 0.5
