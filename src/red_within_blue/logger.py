"""File-based experiment logging and hyperparameter tracking."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ExperimentLogger:
    """Log experiments, hyperparameters, and metrics to structured directories.

    Usage::

        logger = ExperimentLogger(base_dir="experiments", experiment_name="ppo_baseline")
        logger.log_config(env_config)
        logger.log_hyperparams({"lr": 3e-4, "gamma": 0.99})
        logger.log_metrics(step=100, metrics={"coverage": 0.75, "reward": 12.3})
        logger.save_checkpoint(params, step=100)
        logger.close()
    """

    def __init__(self, base_dir: str = "experiments", experiment_name: str = "run"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{timestamp}_{experiment_name}"
        self.experiment_dir = Path(base_dir) / self.experiment_id

        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "trajectories").mkdir(exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)

        self._metrics_file: Optional[Any] = None

    @property
    def metrics_path(self) -> Path:
        return self.experiment_dir / "metrics.jsonl"

    def log_config(self, config) -> None:
        """Serialize environment config to config.json.

        Accepts Flax struct dataclasses, dicts, or any object with a
        ``__dict__`` attribute.
        """
        if hasattr(config, "__dict__"):
            data = {k: _to_serializable(v) for k, v in config.__dict__.items()}
        elif isinstance(config, dict):
            data = {k: _to_serializable(v) for k, v in config.items()}
        else:
            data = {"config": str(config)}

        path = self.experiment_dir / "config.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Write training hyperparameters to hyperparams.json."""
        data = {k: _to_serializable(v) for k, v in params.items()}
        path = self.experiment_dir / "hyperparams.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Append one metrics record to metrics.jsonl (one JSON line)."""
        record = {"step": step}
        record.update({k: _to_serializable(v) for k, v in metrics.items()})
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_checkpoint(self, params, step: int) -> str:
        """Save model parameters as a .npz file keyed by step number."""
        path = self.experiment_dir / "checkpoints" / f"step_{step}.npz"
        if isinstance(params, dict):
            np.savez(str(path), **{k: np.asarray(v) for k, v in params.items()})
        else:
            np.savez(str(path), params=np.asarray(params))
        return str(path)

    def close(self) -> None:
        """Finalize the logger (no-op for file-based, but keeps the interface clean)."""
        pass


def list_experiments(base_dir: str = "experiments") -> List[Dict[str, Any]]:
    """Scan experiments directory and return sorted list with metadata."""
    base = Path(base_dir)
    if not base.exists():
        return []

    results = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        info: Dict[str, Any] = {"id": entry.name, "path": str(entry)}

        config_path = entry / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                info["config"] = json.load(f)

        hp_path = entry / "hyperparams.json"
        if hp_path.exists():
            with open(hp_path) as f:
                info["hyperparams"] = json.load(f)

        metrics_path = entry / "metrics.jsonl"
        if metrics_path.exists():
            info["num_metrics_records"] = sum(1 for _ in open(metrics_path))

        results.append(info)

    return results


def load_experiment(base_dir: str, experiment_id: str) -> Dict[str, Any]:
    """Load config, hyperparams, and metrics for a past experiment."""
    exp_dir = Path(base_dir) / experiment_id
    result: Dict[str, Any] = {"id": experiment_id, "path": str(exp_dir)}

    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            result["config"] = json.load(f)

    hp_path = exp_dir / "hyperparams.json"
    if hp_path.exists():
        with open(hp_path) as f:
            result["hyperparams"] = json.load(f)

    metrics_path = exp_dir / "metrics.jsonl"
    if metrics_path.exists():
        records = []
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        result["metrics"] = records

    return result


def _to_serializable(v: Any) -> Any:
    """Convert a value to a JSON-serializable form."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)
