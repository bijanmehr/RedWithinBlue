"""Open the Box — exhaustive XAI sweep across six checkpoints.

Six setups:

  B   — N=5 clean (5 blue, 0 red), 16x16
  C1  — N=5 m=1 (4 blue, 1 red), 16x16
  C2  — N=5 m=2 (3 blue, 2 red), 16x16
  D1  — N=4 m=1 (3 blue, 1 red), 16x16    (compromise-16x16-3b1r-coevo)
  D2  — N=4 m=2 (2 blue, 2 red), 16x16    (compromise-16x16-2b2r-coevo)
  L6  — N=10 (6 blue, 4 red), 32x32        (adv-ladder-r6-coevo)

Ten methods:

  M1  vanilla input-saliency           — |∂ logit_a*/∂ obs|
  M2  integrated gradients (32 steps)  — (o - 0) · ∫ ∇logit
  M3  SmoothGrad (n=12, σ=0.10)        — noise-averaged gradient
  M4  Gradient × Input                 — ∇logit · obs (cheap signed IG)
  M5  block occlusion (KL + Δcov)      — zero a block, re-roll + per-step KL
  M6  per-cell spatial occlusion       — zero one of 9 scan / seen cells
  M7  identity-swap counterfactual     — permute uid within team, re-roll
  M8  linear probes on hidden state    — will_stay_next, frontier, blue_in_view
  M9  hidden-state PCA                 — global 2D projection, colour by behaviour
  M10 TCAV concept-direction sensitivity — ∂logit / ∂(probe normal direction)

Outputs (all under experiments/meta-report/):

  openthebox_cross_summary.png         — ONE figure across all 6 setups, all methods
  openthebox_block_attribution.png     — M1+M2+M3+M4 stacked, blue/red, per setup
  openthebox_method_correlation.png    — M1 vs M2 vs M3 vs M4 vs M5 (rank corr per setup)
  openthebox_block_occlusion.png       — M5 Δcov + KL across all 6 setups
  openthebox_per_cell_occlusion.png    — M6: 3x3 KL heatmaps for scan + seen, all setups
  openthebox_identity_swap.png         — M7: swap-score per setup, action dist mirrors
  openthebox_probes_grid.png           — M8: probe accuracy heatmap, all setups
  openthebox_pca_manifold.png          — M9: global PCA, six panels coloured by action
  openthebox_tcav.png                  — M10: per-concept logit sensitivity
  openthebox_summary.json              — every headline number cached

Run:
    python scripts/openthebox.py
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import jax
import jax.numpy as jnp

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from meta_report import OUT_DIR, _load_blue, _load_red
from meta_report_xai_causal import (
    BLOCKS,
    BLOCK_NAMES,
    BLOCK_COLOURS,
    _kl_per_step,
    _make_occlude_block,
    _rollout,
    _swap_uids_within_team,
)
from red_within_blue.training.config import ExperimentConfig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The max_iter*")


# =============================================================================
# Setup definitions
# =============================================================================
@dataclass
class BoxSetup:
    key: str
    label: str
    config: str
    blue_ckpt: str
    red_ckpt: Optional[str]


OPENBOX_SETUPS: List[BoxSetup] = [
    BoxSetup("B",  "N=5 clean (5b/0r) 16x16",
             "configs/survey-local-16-N5-from-N4.yaml",
             "experiments/survey-local-16-N5-from-N4/checkpoint.npz",
             None),
    BoxSetup("C1", "N=5 m=1 (4b/1r) 16x16",
             "configs/compromise-16x16-5-4b1r.yaml",
             "experiments/compromise-16x16-5-4b1r-coevo/checkpoint.npz",
             "experiments/compromise-16x16-5-4b1r-coevo/joint_red_checkpoint.npz"),
    BoxSetup("C2", "N=5 m=2 (3b/2r) 16x16",
             "configs/compromise-16x16-5-3b2r.yaml",
             "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz",
             "experiments/compromise-16x16-5-3b2r-coevo/joint_red_checkpoint.npz"),
    BoxSetup("D1", "N=4 m=1 (3b/1r) 16x16",
             "configs/compromise-16x16-3b1r.yaml",
             "experiments/compromise-16x16-3b1r-coevo/checkpoint.npz",
             "experiments/compromise-16x16-3b1r-coevo/joint_red_checkpoint.npz"),
    BoxSetup("D2", "N=4 m=2 (2b/2r) 16x16",
             "configs/compromise-16x16-2b2r.yaml",
             "experiments/compromise-16x16-2b2r-coevo/checkpoint.npz",
             "experiments/compromise-16x16-2b2r-coevo/joint_red_checkpoint.npz"),
    BoxSetup("L6", "N=10 (6b/4r) 32x32",
             "configs/adv-ladder-r6-32x32-6b4r.yaml",
             "experiments/adv-ladder-r6-coevo/checkpoint.npz",
             "experiments/adv-ladder-r6-coevo/joint_red_checkpoint.npz"),
]

SEEDS = (0, 1, 2, 3, 4)
NUM_ACTIONS = 5            # 5-action grid
ACTION_NAMES = ["U", "D", "L", "R", "S"]


# =============================================================================
# Forward-pass helpers (compiled once per setup)
# =============================================================================
def _build_apply_fns(setup: BoxSetup):
    cfg = ExperimentConfig.from_yaml(setup.config)
    n_red = cfg.env.num_red_agents
    n_blue = cfg.env.num_agents - n_red
    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    if n_red > 0:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)
    else:
        red_actor, red_params = None, None

    @jax.jit
    def blue_logits(obs):
        return blue_actor.apply(blue_params, obs)
    @jax.jit
    def blue_logit_for(obs, action):
        return blue_actor.apply(blue_params, obs)[action]
    @jax.jit
    def blue_grad(obs, action):
        return jax.grad(lambda o: blue_actor.apply(blue_params, o)[action])(obs)

    if n_red > 0:
        @jax.jit
        def red_logits(obs_flat):
            return red_actor.apply(red_params, obs_flat)
        @jax.jit
        def red_logit_for(obs_flat, r, action):
            return red_actor.apply(red_params, obs_flat)[r, action]
        @jax.jit
        def red_grad(obs_flat, r, action):
            return jax.grad(lambda o: red_actor.apply(red_params, o)[r, action])(obs_flat)
    else:
        red_logits = red_logit_for = red_grad = None

    return {
        "cfg": cfg, "n_blue": n_blue, "n_red": n_red,
        "blue_actor": blue_actor, "blue_params": blue_params,
        "red_actor": red_actor, "red_params": red_params,
        "blue_logits": blue_logits, "blue_grad": blue_grad,
        "red_logits": red_logits, "red_grad": red_grad,
    }


# =============================================================================
# M1–M4: gradient-family attribution methods on baseline trajectory
# =============================================================================
def collect_attributions(setup: BoxSetup, base: dict, fns: dict,
                         ig_steps: int = 32, smooth_n: int = 12, smooth_sigma: float = 0.10
                         ) -> Dict[str, np.ndarray]:
    """For each (T, N) on the baseline trajectory, compute four attribution
    arrays of shape (T, N, obs_dim). Returns dict keyed by method name.
    Joint-red obs is concat across reds; we split back to per-red 23-vec
    by taking that red's own slice."""
    T, N, obs_dim = base["all_obs"].shape
    n_blue = fns["n_blue"]; n_red = fns["n_red"]
    actions = base["all_actions"]  # (T, N)
    obs = base["all_obs"]          # (T, N, obs_dim)

    sal = np.zeros((T, N, obs_dim))
    ig  = np.zeros((T, N, obs_dim))
    sg  = np.zeros((T, N, obs_dim))
    gxi = np.zeros((T, N, obs_dim))

    alphas = jnp.linspace(1.0 / ig_steps, 1.0, ig_steps)
    rng = np.random.default_rng(0)

    for t in range(T):
        # Blues — independent input
        for i in range(n_blue):
            o = jnp.asarray(obs[t, i]); a = int(actions[t, i])
            g = np.asarray(fns["blue_grad"](o, a))
            sal[t, i] = np.abs(g)
            gxi[t, i] = g * np.asarray(o)
            # IG
            base_input = jnp.zeros_like(o)
            ig_grads = []
            for alpha in alphas:
                interp = base_input + alpha * (o - base_input)
                ig_grads.append(np.asarray(fns["blue_grad"](interp, a)))
            ig[t, i] = (np.asarray(o) - 0.0) * np.mean(np.stack(ig_grads), axis=0)
            # SmoothGrad
            sg_grads = []
            for _ in range(smooth_n):
                noise = rng.normal(0.0, smooth_sigma, size=obs_dim).astype(np.float32)
                sg_grads.append(np.asarray(fns["blue_grad"](o + jnp.asarray(noise), a)))
            sg[t, i] = np.mean(np.abs(np.stack(sg_grads)), axis=0)

        # Reds — joint input, gradient on the concat then slice
        if n_red > 0:
            obs_red_concat = jnp.concatenate(
                [jnp.asarray(obs[t, n_blue + r]) for r in range(n_red)]
            )
            for r in range(n_red):
                a = int(actions[t, n_blue + r])
                g = np.asarray(fns["red_grad"](obs_red_concat, r, a))
                # split back into per-red 23-vec slices, take own slice
                own_slice = slice(r * obs_dim, (r + 1) * obs_dim)
                own_obs = np.asarray(obs[t, n_blue + r])
                sal[t, n_blue + r] = np.abs(g[own_slice])
                gxi[t, n_blue + r] = g[own_slice] * own_obs
                # IG on joint input — keep only own slice in attribution
                base_input = jnp.zeros_like(obs_red_concat)
                ig_grads = []
                for alpha in alphas:
                    interp = base_input + alpha * (obs_red_concat - base_input)
                    ig_grads.append(np.asarray(fns["red_grad"](interp, r, a)))
                ig_joint = np.asarray(obs_red_concat) * np.mean(np.stack(ig_grads), axis=0)
                ig[t, n_blue + r] = ig_joint[own_slice]
                # SmoothGrad on joint
                sg_grads = []
                for _ in range(smooth_n):
                    noise = rng.normal(0.0, smooth_sigma,
                                       size=obs_red_concat.shape[0]).astype(np.float32)
                    sg_grads.append(
                        np.asarray(fns["red_grad"](obs_red_concat + jnp.asarray(noise), r, a))
                    )
                sg[t, n_blue + r] = np.mean(np.abs(np.stack(sg_grads)), axis=0)[own_slice]

    return {"saliency": sal, "ig": ig, "smoothgrad": sg, "gradxinput": gxi}


def _block_share(attr: np.ndarray) -> np.ndarray:
    """attr shape (T, N, obs_dim). Returns (T, N, len(BLOCKS)) of per-block share.
    Uses |attr| so signed methods (ig, gxi) become magnitudes for share comparison."""
    T, N, _ = attr.shape
    out = np.zeros((T, N, len(BLOCKS)))
    for i, (_, lo, hi) in enumerate(BLOCKS):
        out[:, :, i] = np.abs(attr[:, :, lo:hi]).mean(axis=-1)
    out = out / np.maximum(1e-12, out.sum(axis=-1, keepdims=True))
    return out


def team_block_means(attr: np.ndarray, team_ids: np.ndarray, n_blue: int, n_red: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (blue_means, red_means) shape (len(BLOCKS),) — share averaged
    over team agents and time."""
    share = _block_share(attr)  # (T, N, B)
    blue = share[:, team_ids == 0].mean(axis=(0, 1)) if (team_ids == 0).any() else np.zeros(len(BLOCKS))
    red  = share[:, team_ids == 1].mean(axis=(0, 1)) if (team_ids == 1).any() else np.zeros(len(BLOCKS))
    return blue, red


# =============================================================================
# M5 — block occlusion (KL on baseline trajectory + Δcov)
# =============================================================================
def run_block_occlusion(setup: BoxSetup, base: dict, fns: dict) -> Dict:
    """Returns per-block dict with kl arrays (T, N) and Δcov rerolls."""
    T, N, obs_dim = base["all_obs"].shape
    n_blue = fns["n_blue"]; n_red = fns["n_red"]
    out = {}
    for block_name, lo, hi in BLOCKS:
        # KL on baseline trajectory
        occluded_obs = base["all_obs"].copy()
        occluded_obs[:, :, lo:hi] = 0.0
        occ_logits = np.zeros_like(base["all_logits"])
        for t in range(T):
            for i in range(n_blue):
                occ_logits[t, i] = np.asarray(fns["blue_logits"](jnp.asarray(occluded_obs[t, i])))
            if n_red > 0:
                rf = jnp.concatenate([jnp.asarray(occluded_obs[t, n_blue + r]) for r in range(n_red)])
                rl = np.asarray(fns["red_logits"](rf))
                occ_logits[t, n_blue:n_blue + n_red] = rl
        kl = _kl_per_step(base["all_logits"], occ_logits)  # (T, N)
        out[block_name] = {"kl": kl, "blue_kl": kl[:, :n_blue].mean(),
                           "red_kl": kl[:, n_blue:].mean() if n_red > 0 else 0.0}
    return out


def aggregate_dcov(setup: BoxSetup, fns: dict) -> Dict[str, Tuple[float, float]]:
    """Δcov per block, averaged across SEEDS — full reroll. Slowest part."""
    out = {}
    base_covs = []
    for seed in SEEDS:
        b = _rollout(setup, seed=seed, obs_transform=None)
        base_covs.append(b["final_coverage"])
    for block_name, lo, hi in BLOCKS:
        deltas = []
        for seed, base_cov in zip(SEEDS, base_covs):
            tr = _make_occlude_block(block_name)
            r = _rollout(setup, seed=seed, obs_transform=tr)
            deltas.append(base_cov - r["final_coverage"])
        out[block_name] = (float(np.mean(deltas)),
                           float(np.std(deltas) / np.sqrt(len(deltas))))
    return out


# =============================================================================
# M6 — per-cell spatial occlusion (KL when zeroing one of 9 scan or seen cells)
# =============================================================================
def run_per_cell_occlusion(setup: BoxSetup, base: dict, fns: dict) -> Dict:
    """For scan (cells 0..9) and seen (cells 9..18), zero one cell at a time
    and measure mean KL on the baseline trajectory. Returns:
        {"scan": {team: (3,3) heatmap}, "seen": {...}}
    """
    T, N, obs_dim = base["all_obs"].shape
    n_blue = fns["n_blue"]; n_red = fns["n_red"]
    base_logits = base["all_logits"]
    out = {"scan": {"blue": np.zeros(9), "red": np.zeros(9)},
           "seen": {"blue": np.zeros(9), "red": np.zeros(9)}}
    for block_name, lo, hi in [("scan", 0, 9), ("seen", 9, 18)]:
        for cell in range(9):
            occluded_obs = base["all_obs"].copy()
            occluded_obs[:, :, lo + cell] = 0.0
            occ_logits = np.zeros_like(base_logits)
            for t in range(T):
                for i in range(n_blue):
                    occ_logits[t, i] = np.asarray(fns["blue_logits"](jnp.asarray(occluded_obs[t, i])))
                if n_red > 0:
                    rf = jnp.concatenate([jnp.asarray(occluded_obs[t, n_blue + r]) for r in range(n_red)])
                    occ_logits[t, n_blue:n_blue + n_red] = np.asarray(fns["red_logits"](rf))
            kl = _kl_per_step(base_logits, occ_logits)
            out[block_name]["blue"][cell] = kl[:, :n_blue].mean()
            if n_red > 0:
                out[block_name]["red"][cell] = kl[:, n_blue:].mean()
    return out


# =============================================================================
# M7 — identity swap (single seed-set, retain action distributions)
# =============================================================================
def run_identity_swap(setup: BoxSetup) -> Optional[Dict]:
    if setup.key == "B":
        return None
    cfg = ExperimentConfig.from_yaml(setup.config)
    n_red = cfg.env.num_red_agents; n_blue = cfg.env.num_agents - n_red
    if n_red < 2:
        return None  # Single-red has no within-team swap
    rows = []
    for seed in SEEDS:
        base = _rollout(setup, seed=seed, obs_transform=None)
        swap = _rollout(setup, seed=seed, obs_transform=_swap_uids_within_team)
        # action histograms per red
        eps = 1e-9
        # n_red could be 2 or 4 (L6). Reverse permutation; identity-swap holds at any size.
        base_dists = []
        swap_dists = []
        for r in range(n_red):
            ba = base["all_actions"][:, n_blue + r]
            sa = swap["all_actions"][:, n_blue + r]
            base_dists.append(np.bincount(ba, minlength=NUM_ACTIONS) / max(1, len(ba)))
            swap_dists.append(np.bincount(sa, minlength=NUM_ACTIONS) / max(1, len(sa)))
        # swap_score: average over reds of KL(swap_r || base_other(r)) - KL(swap_r || base_self(r))
        # where other(r) is reverse permutation index
        other_idx = list(range(n_red))[::-1]
        kl_self_avg = 0.0; kl_other_avg = 0.0
        for r in range(n_red):
            hs = swap_dists[r]; hb = base_dists[r]; ho = base_dists[other_idx[r]]
            kl_self_avg += np.sum(hs * np.log((hs + eps) / (hb + eps)))
            kl_other_avg += np.sum(hs * np.log((hs + eps) / (ho + eps)))
        kl_self_avg /= n_red; kl_other_avg /= n_red
        rows.append({
            "seed": seed,
            "base_cov": base["final_coverage"],
            "swap_cov": swap["final_coverage"],
            "kl_self": float(kl_self_avg),
            "kl_other": float(kl_other_avg),
            "swap_score": float(kl_other_avg - kl_self_avg),
            "base_dists": [d.tolist() for d in base_dists],
            "swap_dists": [d.tolist() for d in swap_dists],
        })
    return {"n_red": n_red, "rows": rows}


# =============================================================================
# M8 — linear probes on hidden activations
# =============================================================================
def _safe_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if len(y) < n_splits * 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=400, C=1.0)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def _safe_probe_with_direction(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Returns (mean_accuracy, mean_normal_direction). Direction is unit-normalized
    average of cv-fitted coef vectors."""
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return float("nan"), None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []; coefs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=400, C=1.0)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
        coefs.append(clf.coef_[0])
    direction = np.mean(np.stack(coefs), axis=0)
    n = np.linalg.norm(direction) + 1e-12
    return float(np.mean(accs)), direction / n


def collect_probe_data(setup: BoxSetup, base: dict, fns: dict) -> Dict:
    """For each team × concept, fit a probe on the actor's last hidden layer."""
    T, N, _ = base["all_obs"].shape
    n_blue = fns["n_blue"]; n_red = fns["n_red"]
    obs = base["all_obs"]               # (T, N, 23)
    actions = base["all_actions"]       # (T, N)
    hidden = base["all_hidden"]         # (T, N, H)
    team_ids = base["team_ids"]

    seen_block = obs[..., 9:18]          # (T, N, 9)
    frontier_in_view = (seen_block == 1).any(axis=-1).astype(int) | (seen_block == 0).any(axis=-1).astype(int)
    # In env the seen cell takes value 0 = unknown, 1 = empty, 2 = wall.
    # Frontier = adjacent-to-unknown ⇒ any seen cell == 0 (own cell included if unknown).
    frontier_in_view = (seen_block == 0).any(axis=-1).astype(int)
    will_stay_next = np.zeros_like(actions)
    will_stay_next[:-1] = (actions[1:] == 4).astype(int)  # 4 = STAY
    will_stay_next = will_stay_next[:-1]                  # truncate last step

    # blue_in_view (red-only): is any blue's position within the red's 3x3 sensor frame?
    # We approximate with a position check using base["paths"] aligned to action steps
    paths = base["paths"]                    # (T+1, N, 2)
    H_grid = paths[..., 0].max() + 1; W_grid = paths[..., 1].max() + 1
    blue_in_view = np.zeros_like(actions)
    if n_red > 0:
        for t in range(T):
            blue_pos = paths[t, :n_blue]     # (n_blue, 2)
            red_pos = paths[t, n_blue:n_blue + n_red]
            for r in range(n_red):
                rp = red_pos[r]
                # is any blue within Chebyshev 1 of red?
                deltas = np.abs(blue_pos - rp).max(axis=-1)
                blue_in_view[t, n_blue + r] = int((deltas <= 1).any())

    # Fit probes
    out = {}
    for team_name, mask in [("blue", team_ids == 0), ("red", team_ids == 1)]:
        if not mask.any():
            continue
        team_h = hidden[:, mask].reshape(-1, hidden.shape[-1])
        # Discard last-step entries that don't have next-action info
        T_eff = T - 1
        team_h_eff = hidden[:T_eff, mask].reshape(-1, hidden.shape[-1])
        team_stay = will_stay_next[:T_eff, mask].reshape(-1)
        team_frontier = frontier_in_view[:T_eff, mask].reshape(-1)
        team_h_full = hidden[:, mask].reshape(-1, hidden.shape[-1])

        # Probes
        out[team_name] = {}
        acc, dir_stay = _safe_probe_with_direction(team_h_eff, team_stay)
        out[team_name]["will_stay_next"] = acc
        out[team_name]["dir_will_stay_next"] = dir_stay
        # shuffled baseline
        rng = np.random.default_rng(0)
        out[team_name]["shuffled_stay"] = _safe_probe(team_h_eff, rng.permutation(team_stay))

        acc, dir_front = _safe_probe_with_direction(team_h_eff, team_frontier)
        out[team_name]["frontier_in_view"] = acc
        out[team_name]["dir_frontier"] = dir_front

        if team_name == "red" and n_red > 0:
            team_blue_iv = blue_in_view[:T_eff, mask].reshape(-1)
            acc, dir_biv = _safe_probe_with_direction(team_h_eff, team_blue_iv)
            out[team_name]["blue_in_view"] = acc
            out[team_name]["dir_blue_in_view"] = dir_biv
        out[team_name]["n_examples"] = int(team_h_eff.shape[0])
    return out


# =============================================================================
# M10 — TCAV-style logit sensitivity along probe directions
# =============================================================================
def run_tcav(setup: BoxSetup, base: dict, fns: dict, probe_results: Dict) -> Dict:
    """For each (team, concept) probe direction, average the absolute change in
    logit_a* under a small step along the probe direction. Approximated by
    computing the gradient of logit_a* w.r.t. the *hidden activation*, then
    projecting onto the probe direction.

    Implementation note: we don't have a closed-form ∂logit/∂h because the
    actor's last linear layer is the head — instead we use the head weights
    directly. For an MLP whose last layer is logits = W @ h + b (no nonlinearity
    on the head), ∂logit_a*/∂h = W[a*]. Then sensitivity = |W[a*] · direction|.
    """
    T, N, _ = base["all_obs"].shape
    n_blue = fns["n_blue"]; n_red = fns["n_red"]
    actions = base["all_actions"]
    team_ids = base["team_ids"]

    # Extract head weights from blue actor
    blue_params = fns["blue_params"]
    cfg = fns["cfg"]
    blue_layers = cfg.network.actor_num_layers
    blue_head_key = f"Dense_{blue_layers}"
    if "params" in blue_params and blue_head_key in blue_params["params"]:
        blue_head_W = np.asarray(blue_params["params"][blue_head_key]["kernel"])  # (H, A)
    else:
        blue_head_W = None

    if n_red > 0:
        red_params = fns["red_params"]
        red_layers = cfg.train.red_num_layers
        red_head_key = f"Dense_{red_layers}"
        if "params" in red_params and red_head_key in red_params["params"]:
            red_head_W = np.asarray(red_params["params"][red_head_key]["kernel"])
            # red head is shape (H, n_red * num_actions) — re-shape
            if red_head_W.shape[-1] == n_red * NUM_ACTIONS:
                red_head_W = red_head_W.reshape(red_head_W.shape[0], n_red, NUM_ACTIONS)
            else:
                red_head_W = red_head_W.reshape(red_head_W.shape[0], 1, NUM_ACTIONS)
        else:
            red_head_W = None
    else:
        red_head_W = None

    out = {}
    for team_name, mask in [("blue", team_ids == 0), ("red", team_ids == 1)]:
        if not mask.any():
            continue
        team_acts = actions[:, mask]  # (T, n_team)
        head_W = blue_head_W if team_name == "blue" else red_head_W
        if head_W is None:
            continue
        out[team_name] = {}
        for concept_key in ["will_stay_next", "frontier_in_view", "blue_in_view"]:
            dir_key = f"dir_{concept_key}"
            if probe_results.get(team_name, {}).get(dir_key) is None:
                continue
            direction = probe_results[team_name][dir_key]
            # Per-step sensitivity: |W[a*] · direction|
            sens = []
            for t in range(team_acts.shape[0]):
                for a in team_acts[t]:
                    if team_name == "red" and head_W.ndim == 3:
                        # Use mean across reds since action mapping is per-slot
                        sens.append(float(np.abs((head_W[:, :, int(a)].mean(axis=1) * direction).sum())))
                    else:
                        sens.append(float(np.abs((head_W[:, int(a)] * direction).sum())))
            out[team_name][concept_key] = float(np.mean(sens))
    return out


# =============================================================================
# Driver — run all methods on all setups
# =============================================================================
def collect_all() -> Dict:
    """One pass: rollout each setup × seed; collect all method outputs."""
    bigD: Dict[str, Dict] = {}
    pca_pool_h: List[np.ndarray] = []      # (n_total_hidden_rows, H)
    pca_pool_meta: List[Tuple[str, str, int]] = []   # (setup_key, team, action)

    for setup in OPENBOX_SETUPS:
        print(f"\n[setup] {setup.key}: {setup.label}")
        fns = _build_apply_fns(setup)
        sd = bigD[setup.key] = {}

        # Single canonical (seed=0) rollout for cheap-per-step methods
        base_seed0 = _rollout(setup, seed=0, obs_transform=None)

        # M1–M4 attribution (only seed=0 for cost; team-mean shares)
        print("  M1–M4 attribution …")
        attr = collect_attributions(setup, base_seed0, fns)
        team_means = {}
        for name, A in attr.items():
            blue_m, red_m = team_block_means(A, base_seed0["team_ids"], fns["n_blue"], fns["n_red"])
            team_means[name] = {"blue": blue_m.tolist(), "red": red_m.tolist()}
        sd["attribution"] = team_means

        # M5 — block occlusion KL on baseline trajectory (seed=0 only)
        print("  M5 block-occlusion KL …")
        kl_per_block = run_block_occlusion(setup, base_seed0, fns)
        sd["block_occlusion_kl"] = {b: {"blue": float(v["blue_kl"]),
                                         "red": float(v["red_kl"])}
                                     for b, v in kl_per_block.items()}

        # M5 — Δcov over 5 seeds (slowest)
        print("  M5 Δcov rerolls …")
        sd["block_occlusion_dcov"] = {b: {"mean": m, "se": s}
                                       for b, (m, s) in aggregate_dcov(setup, fns).items()}

        # M6 — per-cell spatial occlusion (seed=0 only, scan + seen)
        print("  M6 per-cell …")
        sd["per_cell"] = run_per_cell_occlusion(setup, base_seed0, fns)
        # Convert numpy arrays to lists for JSON
        sd["per_cell_serializable"] = {
            blk: {team: arr.tolist() for team, arr in d.items()}
            for blk, d in sd["per_cell"].items()
        }

        # M7 — identity-swap (5 seeds, only setups with n_red >= 2)
        print("  M7 identity-swap …")
        sd["identity_swap"] = run_identity_swap(setup)

        # M8 — probes
        print("  M8 probes …")
        probe_res = collect_probe_data(setup, base_seed0, fns)
        sd["probes"] = {team: {k: (None if v is None else
                                   (float(v) if not isinstance(v, np.ndarray) else None))
                                for k, v in d.items() if not k.startswith("dir_")}
                         for team, d in probe_res.items()}

        # M10 — TCAV
        print("  M10 TCAV …")
        sd["tcav"] = run_tcav(setup, base_seed0, fns, probe_res)

        # M9 — pool hidden states for global PCA
        team_ids = base_seed0["team_ids"]; H = base_seed0["all_hidden"]
        actions = base_seed0["all_actions"]
        for t in range(H.shape[0]):
            for i in range(H.shape[1]):
                pca_pool_h.append(H[t, i])
                team = "blue" if team_ids[i] == 0 else "red"
                pca_pool_meta.append((setup.key, team, int(actions[t, i])))

    # Fit global PCA across all setups
    print("\n[pca] fitting global PCA on pooled hidden states …")
    H_pool = np.stack(pca_pool_h, axis=0)
    pca = PCA(n_components=2, random_state=0).fit(H_pool)
    H2 = pca.transform(H_pool)

    pca_data = {sk: {"blue": {"x": [], "y": [], "a": []},
                     "red": {"x": [], "y": [], "a": []}}
                for sk in [s.key for s in OPENBOX_SETUPS]}
    for (setup_key, team, action), (x, y) in zip(pca_pool_meta, H2):
        pca_data[setup_key][team]["x"].append(float(x))
        pca_data[setup_key][team]["y"].append(float(y))
        pca_data[setup_key][team]["a"].append(int(action))

    bigD["_pca"] = pca_data
    bigD["_pca_explained"] = pca.explained_variance_ratio_.tolist()
    return bigD


# =============================================================================
# Plots
# =============================================================================
def fig_block_attribution(D: Dict, out_png: Path) -> None:
    """For each setup: 4 grouped bars (M1, M2, M3, M4) × 6 blocks, blue & red."""
    setups = [s for s in OPENBOX_SETUPS]
    method_keys = ["saliency", "ig", "smoothgrad", "gradxinput"]
    method_labels = ["Saliency", "IG (32)", "SmoothGrad", "Grad×In"]
    fig, axes = plt.subplots(2, len(setups), figsize=(3.6 * len(setups), 7.5),
                             sharey="row")
    for col, setup in enumerate(setups):
        for row, team in enumerate(["blue", "red"]):
            ax = axes[row, col]
            for mi, mkey in enumerate(method_keys):
                shares = D[setup.key]["attribution"][mkey][team]
                ax.bar(np.arange(len(BLOCKS)) + mi * 0.20 - 0.30, shares, 0.20,
                       color=BLOCK_COLOURS[BLOCKS[0][0]] if False else None,
                       label=method_labels[mi] if (col == 0 and row == 0) else None,
                       edgecolor=["#1f4e8c", "#a6231f"][row], linewidth=0.5)
            ax.set_xticks(np.arange(len(BLOCKS)))
            ax.set_xticklabels(BLOCK_NAMES, rotation=30, ha="right", fontsize=7)
            if row == 0:
                ax.set_title(f"{setup.key}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{team}\nblock share", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("M1–M4 — gradient-family attribution by obs-block, four methods × six setups\n"
                 "Top row: blue policy. Bottom row: joint-red controller (B has no red).",
                 fontsize=11)
    fig.legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout(rect=[0, 0, 0.97, 0.93])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_method_correlation(D: Dict, out_png: Path) -> None:
    """For each setup, Spearman-corr matrix between block-share rankings of
    M1..M4 + occlusion-KL (treated as a 5th method). Diagnoses agreement."""
    from scipy.stats import spearmanr
    setups = [s for s in OPENBOX_SETUPS]
    method_keys = ["saliency", "ig", "smoothgrad", "gradxinput", "occlusion_kl"]
    method_labels = ["Saliency", "IG", "Smooth", "Grad×In", "Occl-KL"]
    fig, axes = plt.subplots(2, len(setups), figsize=(2.4 * len(setups), 5.4),
                             sharey="row")
    for col, setup in enumerate(setups):
        for row, team in enumerate(["blue", "red"]):
            ax = axes[row, col]
            sd = D[setup.key]
            if team == "red" and setup.key == "B":
                ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.set_axis_off()
                continue
            mat = []
            for mk in method_keys:
                if mk == "occlusion_kl":
                    vec = [sd["block_occlusion_kl"][b][team] for b in BLOCK_NAMES]
                else:
                    vec = sd["attribution"][mk][team]
                mat.append(np.asarray(vec))
            mat = np.stack(mat)
            R = np.zeros((len(method_keys), len(method_keys)))
            for i in range(len(method_keys)):
                for j in range(len(method_keys)):
                    if mat[i].std() < 1e-9 or mat[j].std() < 1e-9:
                        R[i, j] = float("nan")
                    else:
                        R[i, j] = spearmanr(mat[i], mat[j]).correlation
            im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(len(method_keys)):
                for j in range(len(method_keys)):
                    if not np.isnan(R[i, j]):
                        ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center",
                                fontsize=6, color="black")
            ax.set_xticks(range(len(method_keys))); ax.set_yticks(range(len(method_keys)))
            ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=7)
            ax.set_yticklabels(method_labels, fontsize=7)
            if row == 0: ax.set_title(setup.key, fontsize=10)
            if col == 0: ax.set_ylabel(team, fontsize=9)
    fig.suptitle("Spearman rank-correlation between attribution methods (per setup × team)\n"
                 "Block ranks: M1–M4 vs M5 (occlusion KL). Disagreement = which method is wrong.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_block_occlusion(D: Dict, out_png: Path) -> None:
    """Two rows: top = Δcov (M5 reroll), bottom = mean KL (M5 baseline). Six setups."""
    setups = [s for s in OPENBOX_SETUPS]
    fig, axes = plt.subplots(2, len(setups), figsize=(3.4 * len(setups), 6.0),
                             sharey="row")
    x = np.arange(len(BLOCKS))
    for col, setup in enumerate(setups):
        sd = D[setup.key]
        # Δcov bars (single bar per block — team-agnostic since coverage is global)
        ax = axes[0, col]
        means = [sd["block_occlusion_dcov"][b]["mean"] for b in BLOCK_NAMES]
        ses = [sd["block_occlusion_dcov"][b]["se"] for b in BLOCK_NAMES]
        ax.bar(x, means, yerr=ses,
               color=[BLOCK_COLOURS[b] for b in BLOCK_NAMES],
               edgecolor="black", linewidth=1.2, capsize=3)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(BLOCK_NAMES, rotation=30, ha="right", fontsize=7)
        ax.set_title(setup.key, fontsize=10)
        if col == 0: ax.set_ylabel("Δcov (pp, 5 seeds)", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        # KL — grouped bars blue/red
        ax = axes[1, col]
        bar_w = 0.38
        bk = [sd["block_occlusion_kl"][b]["blue"] for b in BLOCK_NAMES]
        rk = [sd["block_occlusion_kl"][b]["red"] for b in BLOCK_NAMES]
        ax.bar(x - bar_w/2, bk, bar_w, color=[BLOCK_COLOURS[b] for b in BLOCK_NAMES],
               edgecolor="#1f4e8c", linewidth=1.2, label="blue")
        ax.bar(x + bar_w/2, rk, bar_w, color=[BLOCK_COLOURS[b] for b in BLOCK_NAMES],
               edgecolor="#a6231f", linewidth=1.2, hatch="//", label="red")
        ax.set_xticks(x); ax.set_xticklabels(BLOCK_NAMES, rotation=30, ha="right", fontsize=7)
        ax.set_yscale("symlog", linthresh=0.05)
        if col == 0: ax.set_ylabel("mean KL (log scale)", fontsize=9); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("M5 — block occlusion. Top: Δcov from full reroll (5 seeds). "
                 "Bottom: mean KL on baseline trajectory (log-y).",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_per_cell(D: Dict, out_png: Path) -> None:
    """3x3 KL heatmaps for scan and seen blocks, all setups, blue and red rows."""
    setups = [s for s in OPENBOX_SETUPS]
    cols = len(setups)
    fig, axes = plt.subplots(4, cols, figsize=(2.5 * cols, 9),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})
    for col, setup in enumerate(setups):
        sd = D[setup.key]
        per_cell = sd["per_cell"]
        for row, (block_name, team) in enumerate([("scan", "blue"),
                                                    ("scan", "red"),
                                                    ("seen", "blue"),
                                                    ("seen", "red")]):
            ax = axes[row, col]
            arr = np.asarray(per_cell[block_name][team]).reshape(3, 3)
            if (team == "red" and setup.key == "B") or arr.max() == 0:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.set_axis_off()
                continue
            im = ax.imshow(arr, cmap="hot", vmin=0)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if arr[i, j] > arr.max()/2 else "black")
            # Mark own cell
            ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, edgecolor="lime", lw=2))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{block_name}\n({team})", fontsize=9)
            if row == 0:
                ax.set_title(setup.key, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.suptitle("M6 — per-cell spatial occlusion: mean KL when zeroing one of the 9 cells\n"
                 "of scan / seen. Lime square = agent's own cell. Hot = important.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_identity_swap_box(D: Dict, out_png: Path) -> None:
    """Per-setup swap-score (negative = behaviour swapped) + action-distribution mirror."""
    setups_with_swap = [s for s in OPENBOX_SETUPS if D[s.key].get("identity_swap") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 2]})

    # Left — swap-score bar per setup
    ax = axes[0]
    keys = [s.key for s in setups_with_swap]
    means = []; ses = []
    for s in setups_with_swap:
        rows = D[s.key]["identity_swap"]["rows"]
        scores = [r["swap_score"] for r in rows]
        means.append(np.mean(scores)); ses.append(np.std(scores) / np.sqrt(len(scores)))
    ax.bar(np.arange(len(keys)), means, yerr=ses, color="#5b8def", edgecolor="black", capsize=4)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(keys)
    ax.set_ylabel("mean swap-score (nats)\n(neg ⇒ identity-swap → behaviour-swap)", fontsize=9)
    ax.set_title("M7 — identity-swap counterfactual", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Right — action-distribution mirror panel for representative setup C2 (or first available)
    ax = axes[1]
    rep = next((s for s in setups_with_swap if s.key == "C2"), setups_with_swap[0])
    rows = D[rep.key]["identity_swap"]["rows"]
    n_red = D[rep.key]["identity_swap"]["n_red"]
    seed0 = rows[0]
    bar_w = 0.18
    x = np.arange(NUM_ACTIONS)
    colors = ["#5b8def", "#23a47e", "#d6594d", "#9b6dd7", "#7d7d7d"]
    for r in range(min(n_red, 4)):
        ax.bar(x + (2 * r - n_red) * bar_w, seed0["base_dists"][r], bar_w,
               label=f"base red{r}", edgecolor="black", linewidth=0.5,
               color=colors[r % len(colors)])
        ax.bar(x + (2 * r + 1 - n_red) * bar_w, seed0["swap_dists"][r], bar_w,
               edgecolor="black", linewidth=0.5, hatch="//",
               color=colors[r % len(colors)], alpha=0.7,
               label=f"swap red{r}" if r == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(ACTION_NAMES)
    ax.set_ylabel("action probability", fontsize=9)
    ax.set_title(f"action distributions on {rep.key} (seed 0): base vs uid-swapped",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("M7 — does swapping uid swap behaviour? Negative score = yes; positive = positional routing.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_probes_grid(D: Dict, out_png: Path) -> None:
    """Heatmap of probe accuracy (rows: setup × team, cols: concepts)."""
    rows = []; row_labels = []
    cols = ["will_stay_next", "frontier_in_view", "blue_in_view", "shuffled_stay"]
    for s in OPENBOX_SETUPS:
        for team in ["blue", "red"]:
            sd = D[s.key]["probes"].get(team)
            if sd is None: continue
            row = []
            for c in cols:
                v = sd.get(c)
                row.append(v if v is not None else float("nan"))
            rows.append(row)
            row_labels.append(f"{s.key}/{team}")
    M = np.asarray(rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    cmap = LinearSegmentedColormap.from_list("greenred",
                                              ["#fff8e6", "#ffd066", "#23a47e", "#1f3a5f"])
    im = ax.imshow(M, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="black" if (np.isnan(v) or v < 0.7) else "white")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, label="5-fold CV accuracy")
    ax.set_title("M8 — Linear probes on actor's last hidden layer\n"
                 "(— = degenerate label distribution: only one class observed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pca_manifold(D: Dict, out_png: Path) -> None:
    """Six panels — for each setup, scatter (PC1, PC2) of all hidden states,
    blue=circles, red=triangles, colour by sampled action."""
    setups = [s for s in OPENBOX_SETUPS]
    cols = 3; rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.0 * rows))
    action_colors = ["#5b8def", "#23a47e", "#d6594d", "#9b6dd7", "#7d7d7d"]
    explained = D["_pca_explained"]
    for i, setup in enumerate(setups):
        ax = axes[i // cols, i % cols]
        pd = D["_pca"][setup.key]
        for team, marker, edge in [("blue", "o", "#1f4e8c"),
                                    ("red", "^", "#a6231f")]:
            xs = np.asarray(pd[team]["x"]); ys = np.asarray(pd[team]["y"])
            acts = np.asarray(pd[team]["a"])
            if len(xs) == 0:
                continue
            for a in range(NUM_ACTIONS):
                mask = acts == a
                if not mask.any(): continue
                ax.scatter(xs[mask], ys[mask], s=6, alpha=0.35,
                           marker=marker, color=action_colors[a],
                           edgecolors="none",
                           label=f"{team}/{ACTION_NAMES[a]}" if i == 0 else None)
        ax.set_title(f"{setup.key}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f"PC1 ({100*explained[0]:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({100*explained[1]:.1f}%)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=2,
               bbox_to_anchor=(0.99, 0.97))
    fig.suptitle("M9 — global hidden-state PCA. All setups projected through the same 2-component fit;\n"
                 "colour = sampled action, marker = team. Clusters = behaviourally distinct activation regimes.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.94, 0.93])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_tcav(D: Dict, out_png: Path) -> None:
    """Bar chart per setup × concept × team — logit sensitivity along probe direction."""
    concepts = ["will_stay_next", "frontier_in_view", "blue_in_view"]
    setups = [s for s in OPENBOX_SETUPS]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bar_w = 0.13
    x = np.arange(len(setups))
    method_offset = 0
    for ci, concept in enumerate(concepts):
        for ti, team in enumerate(["blue", "red"]):
            vals = []
            for setup in setups:
                v = D[setup.key].get("tcav", {}).get(team, {}).get(concept)
                vals.append(v if v is not None else 0.0)
            offset = (ci * 2 + ti - 2.5) * bar_w
            ax.bar(x + offset, vals, bar_w,
                   label=f"{concept} ({team})",
                   color=BLOCK_COLOURS[BLOCK_NAMES[ci % 6]],
                   edgecolor=["#1f4e8c", "#a6231f"][ti], linewidth=0.8,
                   hatch="" if ti == 0 else "//")
    ax.set_xticks(x); ax.set_xticklabels([s.key for s in setups])
    ax.set_ylabel("|head_W[a*] · probe_direction|  (mean over steps)", fontsize=9)
    ax.set_title("M10 — TCAV concept-direction sensitivity\n"
                 "How much does the action-logit move along the probe-direction in hidden space?",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_cross_summary(D: Dict, out_png: Path) -> None:
    """Headline cross-experiment figure. Six panels, one per setup; in each panel:
    - bar: red KL on `scan` vs `seen` vs `uid+team_id`
    - bar: blue KL on `seen` vs `norm_pos` vs `uid+team_id`
    Plus an overlay text with each setup's red entropy / coverage drop / swap score."""
    setups = [s for s in OPENBOX_SETUPS]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, setup in enumerate(setups):
        ax = axes[idx // 3, idx % 3]
        sd = D[setup.key]
        # Compute summary scalars
        red_scan_kl = sd["block_occlusion_kl"]["scan"]["red"]
        red_seen_kl = sd["block_occlusion_kl"]["seen"]["red"]
        red_id_kl = sd["block_occlusion_kl"]["uid"]["red"] + sd["block_occlusion_kl"]["team_id"]["red"]
        blue_scan_kl = sd["block_occlusion_kl"]["scan"]["blue"]
        blue_seen_kl = sd["block_occlusion_kl"]["seen"]["blue"]
        blue_pos_kl = sd["block_occlusion_kl"]["norm_pos"]["blue"]
        labels = ["B/scan", "B/seen", "B/pos", "R/scan", "R/seen", "R/uid+tid"]
        vals = [blue_scan_kl, blue_seen_kl, blue_pos_kl, red_scan_kl, red_seen_kl, red_id_kl]
        colors = ["#5b8def"]*3 + ["#a6231f"]*3
        bars = ax.bar(np.arange(len(labels)), vals, color=colors,
                      edgecolor="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylabel("mean KL when block zeroed", fontsize=8)
        # Annotate values
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, max(v, 0.06) * 1.05,
                    f"{v:.2f}" if v >= 0.01 else f"{v:.1e}",
                    ha="center", va="bottom", fontsize=7)
        # Overlay swap_score and Δcov(uid)
        swap_info = ""
        if sd.get("identity_swap"):
            mean_score = np.mean([r["swap_score"] for r in sd["identity_swap"]["rows"]])
            swap_info = f"swap_score={mean_score:+.1f}"
        dcov_uid = sd["block_occlusion_dcov"]["uid"]["mean"]
        ax.set_title(f"{setup.key}  —  {setup.label}\nΔcov(uid)={dcov_uid:+.1f} pp,  {swap_info}",
                     fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Cross-experiment headline — block-occlusion KL sensitivity, six setups\n"
                 "Red bars: joint-red controller. Blue bars: blue policy. Identity blocks = uid+team_id.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# JSON serialization
# =============================================================================
def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Open the Box — running 10 XAI methods on 6 setups ===")
    D = collect_all()

    print("\n[plots]")
    fig_block_attribution(D,    OUT_DIR / "openthebox_block_attribution.png")
    fig_method_correlation(D,   OUT_DIR / "openthebox_method_correlation.png")
    fig_block_occlusion(D,      OUT_DIR / "openthebox_block_occlusion.png")
    fig_per_cell(D,             OUT_DIR / "openthebox_per_cell_occlusion.png")
    fig_identity_swap_box(D,    OUT_DIR / "openthebox_identity_swap.png")
    fig_probes_grid(D,          OUT_DIR / "openthebox_probes_grid.png")
    fig_pca_manifold(D,         OUT_DIR / "openthebox_pca_manifold.png")
    fig_tcav(D,                 OUT_DIR / "openthebox_tcav.png")
    fig_cross_summary(D,        OUT_DIR / "openthebox_cross_summary.png")

    # Strip per_cell numpy arrays from the JSON copy
    D_json = {sk: {k: v for k, v in sd.items() if k != "per_cell"}
              for sk, sd in D.items() if not sk.startswith("_")}
    D_json["_pca_explained"] = D["_pca_explained"]
    D_json["setups"] = [{"key": s.key, "label": s.label} for s in OPENBOX_SETUPS]
    (OUT_DIR / "openthebox_summary.json").write_text(
        json.dumps(_to_serializable(D_json), indent=2)
    )
    print(f"[done] wrote {OUT_DIR}/openthebox_*.png + openthebox_summary.json")


if __name__ == "__main__":
    main()
