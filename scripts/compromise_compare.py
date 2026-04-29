"""Compromise-sweep comparison: clean-blue baseline vs m/N compromised.

Presentation experiment for the writeup. Total team size N = 5 on 16x16
(minimum N that keeps |red| strictly less than |blue| for both compromise
rates we sweep — 3 decentralised blue + 2 centralised red). Setups:

  (S)  single blue (N=1, no red)         — "why cooperation matters"
  (B)  N=5 clean blue, no red            — "swarm reaches >90% coverage"
  (C1) N=5: 4 blue + 1 red (m=1)         — light compromise
  (C2) N=5: 3 blue + 2 red (m=2)         — max compromise with red<blue

Metric: blue-team-merged `ever-known` coverage — fraction of non-wall cells
that at least one blue has seen or been told about at episode end.

This script avoids the expensive gif-encoding path. It walks the env in
Python per step but does NOT render frames, so it's ~10× faster than
`record_episode_gif` while still producing the same `blue_ever_known`
metric (same `_merge_team_belief` routine).

Run: ``python scripts/compromise_compare.py``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class _Tee:
    """Mirror writes to stdout AND a log file so runs leave a record in out_dir."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self._streams:
            st.flush()

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, JointRedActor
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward,
    normalized_exploration_reward,
)
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
from red_within_blue.visualizer import _merge_team_belief

N_EVAL_SEEDS = 20

SETUPS = [
    {
        "label": "S  (N=1 solo blue)",
        "config": "configs/survey-local-16-N1.yaml",
        "blue_ckpt": "experiments/survey-local-16-N1/checkpoint.npz",
        "red_ckpt": None,
        "max_steps_override": None,
    },
    {
        "label": "B  (N=5 clean blue)",
        "config": "configs/survey-local-16-N5-from-N4.yaml",
        "blue_ckpt": "experiments/survey-local-16-N5-from-N4/checkpoint.npz",
        "red_ckpt": None,
        # Clean N=5 was trained with max_steps=250, but compromise setups use
        # 200. Harmonise the time budget so B/C1/C2 share a task horizon.
        "max_steps_override": 200,
    },
    {
        "label": "C1 (N=5: 4 blue + 1 red)",
        "config": "configs/compromise-16x16-5-4b1r.yaml",
        "blue_ckpt": "experiments/compromise-16x16-5-4b1r-coevo/checkpoint.npz",
        "red_ckpt": "experiments/compromise-16x16-5-4b1r-coevo/joint_red_checkpoint.npz",
        "max_steps_override": None,
    },
    {
        "label": "C2 (N=5: 3 blue + 2 red)",
        "config": "configs/compromise-16x16-5-3b2r.yaml",
        "blue_ckpt": "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz",
        "red_ckpt": "experiments/compromise-16x16-5-3b2r-coevo/joint_red_checkpoint.npz",
        "max_steps_override": None,
    },
]


def _strip_seed_dim(flat: Dict[str, np.ndarray], ref_flat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        k: (v[0] if v.ndim == ref_flat[k].ndim + 1 else v)
        for k, v in flat.items()
    }


def _load_blue(cfg: ExperimentConfig, ckpt_path: str) -> Tuple[Actor, dict]:
    flat = load_checkpoint(ckpt_path)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
    )
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    ref_flat = flatten_params(ref)
    return actor, unflatten_params(_strip_seed_dim(flat, ref_flat), ref)


def _load_red(cfg: ExperimentConfig, ckpt_path: str) -> Tuple[JointRedActor, dict]:
    flat = load_checkpoint(ckpt_path)
    n_red = cfg.env.num_red_agents
    actor = JointRedActor(
        num_red=n_red,
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.train.red_hidden_dim,
        num_layers=cfg.train.red_num_layers,
    )
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(n_red * cfg.obs_dim))
    ref_flat = flatten_params(ref)
    return actor, unflatten_params(_strip_seed_dim(flat, ref_flat), ref)


def _eval_setup(
    cfg: ExperimentConfig,
    blue_actor: Actor,
    blue_params,
    joint_red_actor: Optional[JointRedActor],
    joint_red_params,
    n_seeds: int,
    max_steps_override: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Run n_seeds episodes; return per-episode final coverage + curve + len."""
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red
    max_steps = int(max_steps_override) if max_steps_override is not None else cfg.env.max_steps

    reward_fn = (
        normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    )
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)

    @jax.jit
    def _blue_action(b_params, obs, key):
        logits = blue_actor.apply(b_params, obs)
        return jax.random.categorical(key, logits)

    @jax.jit
    def _red_actions(r_params, obs_flat, key):
        logits = joint_red_actor.apply(r_params, obs_flat)  # [n_red, A]
        keys = jax.random.split(key, n_red)
        return jax.vmap(jax.random.categorical)(keys, logits)

    finals: List[float] = []
    curves: List[np.ndarray] = []
    lengths: List[int] = []

    for seed in range(n_seeds):
        key = jax.random.PRNGKey(seed)
        obs_dict, state = env.reset(key)

        blue_ever = None
        curve: List[float] = []
        done = False
        step = 0
        while not done and step < max_steps:
            step += 1
            key, *agent_keys = jax.random.split(key, n_total + 2)
            step_key = agent_keys[-1]

            action_dict = {}
            if joint_red_actor is not None and n_red > 0:
                red_obs_flat = jnp.concatenate(
                    [obs_dict[env.agents[n_blue + r]] for r in range(n_red)]
                )
                red_key = agent_keys[n_blue]
                red_actions = _red_actions(joint_red_params, red_obs_flat, red_key)
                for r in range(n_red):
                    action_dict[env.agents[n_blue + r]] = int(red_actions[r])

            for i in range(n_blue):
                a = int(_blue_action(blue_params, obs_dict[env.agents[i]], agent_keys[i]))
                action_dict[env.agents[i]] = a

            obs_dict, state, _rewards, dones_dict, _info = env.step_env(
                step_key, state, action_dict,
            )
            done = bool(dones_dict["__all__"])

            local_maps_np = np.asarray(state.agent_state.local_map)
            team_ids_np = np.asarray(state.agent_state.team_ids)
            blue_belief = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)
            known_now = (blue_belief != MAP_UNKNOWN)
            terrain = np.asarray(state.global_state.grid.terrain)
            non_wall = terrain != CELL_WALL
            if blue_ever is None:
                blue_ever = known_now & non_wall
            else:
                blue_ever = blue_ever | (known_now & non_wall)
            cov_pct = 100.0 * blue_ever.sum() / max(1, non_wall.sum())
            curve.append(float(cov_pct))

        finals.append(curve[-1] if curve else 0.0)
        curves.append(np.asarray(curve, dtype=np.float32))
        lengths.append(step)

    # Pad curves to max_steps for clean stacking.
    padded = np.full((n_seeds, max_steps), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c
        # Carry the final value forward for plotting continuity.
        if len(c) < max_steps:
            padded[i, len(c):] = c[-1]

    return {
        "final": np.asarray(finals, dtype=np.float32),
        "curve": padded,
        "ep_len": np.asarray(lengths, dtype=np.int32),
    }


def main():
    out = Path("experiments/compromise-compare")
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "run.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print(f"Compromise-sweep comparison on 16x16, N=5, N_EVAL_SEEDS={N_EVAL_SEEDS}\n")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    header = (
        f"{'setup':<28}  {'n':<3}  {'cov_final%':>11}  "
        f"{'cov_std':>8}  {'cov_min':>8}  {'cov_max':>8}  {'ep_len':>7}"
    )
    print(header)
    print("-" * len(header))

    for s in SETUPS:
        cfg = ExperimentConfig.from_yaml(s["config"])
        blue_actor, blue_params = _load_blue(cfg, s["blue_ckpt"])

        joint_red_actor = None
        joint_red_params = None
        if s["red_ckpt"] is not None:
            joint_red_actor, joint_red_params = _load_red(cfg, s["red_ckpt"])

        metrics = _eval_setup(
            cfg, blue_actor, blue_params, joint_red_actor, joint_red_params,
            n_seeds=N_EVAL_SEEDS,
            max_steps_override=s.get("max_steps_override"),
        )
        results[s["label"]] = metrics

        fin = metrics["final"]
        lens = metrics["ep_len"]
        print(
            f"{s['label']:<28}  {cfg.env.num_agents:<3}  "
            f"{fin.mean():>10.1f}%  {fin.std():>8.2f}  "
            f"{fin.min():>8.1f}  {fin.max():>8.1f}  {lens.mean():>7.1f}"
        )

    # Save for plotting. Curves differ in length (S uses max_steps=300 on N=1
    # config; B/C1/C2 use 200), so save per-setup instead of stacking.
    save_kwargs = {"labels": np.array([s["label"] for s in SETUPS])}
    for i, s in enumerate(SETUPS):
        save_kwargs[f"finals_{i}"] = results[s["label"]]["final"]
        save_kwargs[f"curve_{i}"] = results[s["label"]]["curve"]
        save_kwargs[f"ep_len_{i}"] = results[s["label"]]["ep_len"]
    np.savez(out / "compromise_compare.npz", **save_kwargs)
    print(f"\nWrote: {out / 'compromise_compare.npz'}")

    _write_summary_figure(out, SETUPS, results)
    print(f"Wrote: {out / 'report.png'}")


def _write_summary_figure(out: Path, setups, results) -> None:
    """Two-panel PNG: coverage curves (left) + final-coverage box plot (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s["label"] for s in setups]
    colors = ["#888888", "#1f77b4", "#ff7f0e", "#d62728"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

    for lab, col in zip(labels, colors):
        c = results[lab]["curve"]  # [n_seeds, max_steps]
        mean = np.nanmean(c, axis=0)
        lo = np.nanpercentile(c, 10, axis=0)
        hi = np.nanpercentile(c, 90, axis=0)
        xs = np.arange(1, len(mean) + 1)
        ax_l.plot(xs, mean, label=lab, color=col, linewidth=2)
        ax_l.fill_between(xs, lo, hi, color=col, alpha=0.15)

    ax_l.axhline(90, color="#999", linestyle="--", linewidth=1, alpha=0.6)
    ax_l.text(ax_l.get_xlim()[1] * 0.99, 91.5, "90% threshold", ha="right",
              va="bottom", fontsize=8, color="#555")
    ax_l.set_xlabel("step")
    ax_l.set_ylabel("blue ever-known coverage (%)")
    ax_l.set_title("Per-step mean coverage (shaded = p10–p90 over 20 seeds)")
    ax_l.set_ylim(0, 102)
    ax_l.legend(loc="lower right", fontsize=8)
    ax_l.grid(True, alpha=0.3)

    finals = [results[lab]["final"] for lab in labels]
    bp = ax_r.boxplot(finals, tick_labels=[lab.split(" ")[0] for lab in labels], patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    for i, f in enumerate(finals, start=1):
        ax_r.scatter(np.full_like(f, i) + np.random.uniform(-0.07, 0.07, size=len(f)),
                     f, color="k", s=10, alpha=0.6, zorder=3)
    ax_r.axhline(90, color="#999", linestyle="--", linewidth=1, alpha=0.6)
    ax_r.set_ylabel("final coverage (%)")
    ax_r.set_title("Final coverage distribution (20 seeds each)")
    ax_r.set_ylim(0, 102)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle("Compromise sweep at N=5 on 16×16 (red strictly less than blue)")
    fig.tight_layout()
    fig.savefig(out / "report.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
