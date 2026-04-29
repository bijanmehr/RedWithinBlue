"""Tier-1 + Tier-2 advanced comparison metrics for the I/W/F red-prior arms.

Computes (cheap → expensive):

  * Tier 1 #1 — `rliable`-style stats from the existing 600 paired eval eps:
        IQM (interquartile mean) per arm
        Performance profile = empirical CDF of paired blue_R per arm
        Probability-of-improvement P(A < B | shared eval key) per pair
        Stratified bootstrap CI on IQM
  * Tier 2 #5 — Vargha-Delaney A₁₂ alongside Cohen's d on paired Δ blue_R
  * Tier 2 #4 — 3×3 Jensen-Shannon distance matrix on flattened
        16×16 state-visitation distributions (positions arr from action_probe)
  * Tier 2 #3 — Cross-play 3 × 2 matchup matrix:
        rows = {I, W, F} reds (seed 0)
        cols = {trained blue (existing checkpoint),
                random-init blue (free reference opponent)}
  * Tier 1 #2 — Linear Mode Connectivity barrier matrix (3×3) and
        CKA(Dense_1) / CKA(Dense_2) on hidden activations.
        LMC: interpolate weights θ(α)=(1−α)θ_a + α θ_b on a fixed grid of α,
             evaluate blue_R; barrier = max(endpoint_mean − blue_R(α)) over α.
        CKA: Kornblith et al. 2019 (linear) on a fixed 5 000-obs probe buffer.

Reads:
    experiments/red-prior-phase1/eval_stats.npz   (paired eval blue_R/red_R/coverage/red_ent)
    experiments/red-prior-phase1/action_probe.npz (positions, probs, acts)
    experiments/red-prior-phase1/red_{I,W,F}_seed0.npz
    experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz

Writes:
    experiments/red-prior-phase1/advanced_analysis.npz

Run:    PYTHONPATH=src python scripts/red_prior_advanced_analysis.py
Wall:   ~30–90 s (LMC dominates; ~33 evals × ~0.6 s).
"""
from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import _connectivity_guardrail
from red_within_blue.types import CELL_WALL

CONFIG_PATH = "experiments/compromise-16x16-5-3b2r-coevo/config.yaml"
BLUE_CKPT = "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz"
PHASE1_DIR = Path("experiments/red-prior-phase1")
ARMS = ("I", "W", "F")
PAIRS = (("I", "W"), ("I", "F"), ("W", "F"))

# LMC sweep
N_LMC_EPS = 60
ALPHA_GRID = np.linspace(0.0, 1.0, 11)
LMC_KEY = 91

# CKA probe buffer
N_PROBE_OBS = 5_000
CKA_KEY = 73

# Cross-play
N_XPLAY_EPS = 80
RANDOM_BLUE_KEY = 9_001


# --------------------------- env / network helpers ----------------------------

def _build_env(cfg: ExperimentConfig):
    env_cfg = cfg.to_env_config()
    n_red = cfg.env.num_red_agents
    n_blue = cfg.env.num_agents - n_red
    reward_fn = make_multi_agent_reward(
        disconnect_penalty=cfg.reward.disconnect_penalty,
        isolation_weight=cfg.reward.isolation_weight,
        cooperative_weight=cfg.reward.cooperative_weight,
        revisit_weight=cfg.reward.revisit_weight,
        terminal_bonus_scale=cfg.reward.terminal_bonus_scale,
        terminal_bonus_divide=cfg.reward.terminal_bonus_divide,
        spread_weight=cfg.reward.spread_weight,
        fog_potential_weight=cfg.reward.fog_potential_weight,
        num_red_agents=n_red,
    )
    return GridCommEnv(env_cfg, reward_fn=reward_fn), n_blue, n_red


def _load_actor_params(ckpt_path: str, actor: Actor, obs_dim: int):
    flat = load_checkpoint(ckpt_path)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    ref = actor.init(jax.random.PRNGKey(0), jnp.zeros(obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {}
    for k, v in flat.items():
        if k not in ref_flat:
            continue
        ref_nd = ref_flat[k].ndim
        stripped[k] = v[0] if v.ndim == ref_nd + 1 else v
    return unflatten_params(stripped, ref)


def _build_eval_rollout(env, blue_actor, red_actor, n_blue, n_red,
                        max_steps, num_actions, enforce_conn):
    """JIT-able rollout that returns (blue_total, red_total, final_cov)."""
    num_agents = env.config.num_agents

    def rollout(blue_params, red_params, key):
        key, reset_key = jax.random.split(key)
        _o, state = env.reset(reset_key)

        def _scan_body(carry, _):
            state, rng, cum_done = carry
            rng, bk, rk, sk = jax.random.split(rng, 4)
            obs_all = env.obs_array(state)
            blue_obs = obs_all[:n_blue]
            blue_logits = jax.vmap(blue_actor.apply, in_axes=(None, 0))(
                blue_params, blue_obs)
            blue_acts = jax.vmap(jax.random.categorical)(
                jax.random.split(bk, n_blue), blue_logits)
            red_obs = obs_all[n_blue:]
            red_logits = jax.vmap(red_actor.apply, in_axes=(None, 0))(
                red_params, red_obs)
            red_acts = jax.vmap(jax.random.categorical)(
                jax.random.split(rk, n_red), red_logits)
            actions = jnp.concatenate([blue_acts, red_acts], axis=0)
            safe_acts = jax.lax.cond(
                jnp.bool_(enforce_conn) & (num_agents >= 2),
                lambda a: _connectivity_guardrail(
                    state.agent_state.positions, state.agent_state.comm_ranges,
                    a, state.global_state.grid.terrain),
                lambda a: a, actions,
            )
            _o, new_state, rewards, done, _i = env.step_array(sk, state, safe_acts)
            active = ~cum_done
            new_cum_done = cum_done | done
            explored = state.global_state.grid.explored
            terrain = state.global_state.grid.terrain
            non_wall = (terrain != CELL_WALL)
            cov = jnp.sum((explored > 0) & non_wall) / jnp.maximum(jnp.sum(non_wall), 1)
            return (new_state, rng, new_cum_done), (active, rewards, cov)

        _, (active_seq, rew_seq, cov_seq) = jax.lax.scan(
            _scan_body, (state, key, jnp.bool_(False)), jnp.arange(max_steps),
        )
        active_f = active_seq.astype(jnp.float32)[:, None]
        masked_rew = rew_seq * active_f
        blue_total = jnp.sum(masked_rew[:, :n_blue]) / n_blue
        red_total = jnp.sum(masked_rew[:, n_blue])
        final_cov = cov_seq[-1]
        return blue_total, red_total, final_cov

    return jax.jit(rollout)


def _build_obs_collector(env, blue_actor, red_actor, n_blue, n_red,
                         max_steps, enforce_conn):
    """JIT'd rollout that also returns the flat red obs buffer for the episode."""
    num_agents = env.config.num_agents

    def rollout(blue_params, red_params, key):
        key, reset_key = jax.random.split(key)
        _o, state = env.reset(reset_key)

        def _scan_body(carry, _):
            state, rng, cum_done = carry
            rng, bk, rk, sk = jax.random.split(rng, 4)
            obs_all = env.obs_array(state)
            blue_obs = obs_all[:n_blue]
            red_obs = obs_all[n_blue:]
            blue_logits = jax.vmap(blue_actor.apply, in_axes=(None, 0))(
                blue_params, blue_obs)
            blue_acts = jax.vmap(jax.random.categorical)(
                jax.random.split(bk, n_blue), blue_logits)
            red_logits = jax.vmap(red_actor.apply, in_axes=(None, 0))(
                red_params, red_obs)
            red_acts = jax.vmap(jax.random.categorical)(
                jax.random.split(rk, n_red), red_logits)
            actions = jnp.concatenate([blue_acts, red_acts], axis=0)
            safe_acts = jax.lax.cond(
                jnp.bool_(enforce_conn) & (num_agents >= 2),
                lambda a: _connectivity_guardrail(
                    state.agent_state.positions, state.agent_state.comm_ranges,
                    a, state.global_state.grid.terrain),
                lambda a: a, actions,
            )
            _o, new_state, _r, done, _i = env.step_array(sk, state, safe_acts)
            return (new_state, rng, cum_done | done), red_obs

        _, red_obs_seq = jax.lax.scan(
            _scan_body, (state, key, jnp.bool_(False)), jnp.arange(max_steps),
        )
        return red_obs_seq    # [T, n_red, obs_dim]

    return jax.jit(rollout)


# --------------------------- statistical helpers -----------------------------

def iqm(values: np.ndarray) -> float:
    """Interquartile mean — Agarwal et al. 2021."""
    lo, hi = np.percentile(values, [25, 75])
    band = values[(values >= lo) & (values <= hi)]
    return float(band.mean()) if band.size else float(values.mean())


def iqm_bootstrap_ci(values: np.ndarray, n_boot: int = 5_000,
                     seed: int = 17) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.size
    boots = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[k] = iqm(values[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """A₁₂ ∈ [0, 1]: Pr(X > Y) + 0.5 · Pr(X = Y).
    A₁₂ = 0.5 ⇒ no effect.  Magnitude scale: 0.56 small, 0.64 medium, 0.71 large."""
    n_x, n_y = x.size, y.size
    # rank-based formulation, robust to ties
    ranks = np.argsort(np.argsort(np.concatenate([x, y])))
    r1 = ranks[:n_x].astype(np.float64) + 1.0
    A = (r1.sum() / n_x - (n_x + 1) / 2.0) / n_y
    return float(A)


def prob_improvement_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Probability A < B on shared eval keys.  Returns P(x < y) + 0.5·P(x = y)."""
    return float(((x < y).astype(np.float64) + 0.5 * (x == y).astype(np.float64)).mean())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (float((p * (np.log(p) - np.log(m))).sum())
                  + float((q * (np.log(q) - np.log(m))).sum()))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA — Kornblith et al. 2019 'Similarity of NN Representations'.
    X, Y are [N, D_x] / [N, D_y]; rows are paired observations.
    Returns scalar in [0, 1]; 1 = identical up to invertible linear map."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XY = (X.T @ Y)
    num = float(np.sum(XY * XY))
    den = float(np.sqrt(np.sum((X.T @ X) ** 2) * np.sum((Y.T @ Y) ** 2)))
    return num / max(den, 1e-12)


# --------------------------- CKA activation extractor ------------------------

def _actor_hidden_activations(params: dict, obs: np.ndarray, activation="relu"
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Manually run the 3-layer Actor MLP on a probe buffer; return (h1, h2)
    after the activation, before the next linear transform.

    Bypasses Flax to avoid a recompile per param set."""
    W0 = np.asarray(params["params"]["Dense_0"]["kernel"])
    b0 = np.asarray(params["params"]["Dense_0"]["bias"])
    W1 = np.asarray(params["params"]["Dense_1"]["kernel"])
    b1 = np.asarray(params["params"]["Dense_1"]["bias"])
    if activation == "relu":
        act = lambda x: np.maximum(x, 0.0)
    elif activation == "tanh":
        act = np.tanh
    else:
        raise ValueError(activation)
    h1 = act(obs @ W0 + b0)
    h2 = act(h1 @ W1 + b1)
    return h1, h2


# --------------------------- main pipeline -----------------------------------

def main() -> None:
    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    env, n_blue, n_red = _build_env(cfg)
    obs_dim = cfg.obs_dim
    max_steps = cfg.env.max_steps
    num_actions = cfg.env.num_actions

    blue_actor = Actor(num_actions=num_actions,
                       hidden_dim=cfg.network.actor_hidden_dim,
                       num_layers=cfg.network.actor_num_layers,
                       activation=cfg.network.activation)
    red_actor = Actor(num_actions=num_actions,
                      hidden_dim=cfg.network.actor_hidden_dim,
                      num_layers=cfg.network.actor_num_layers,
                      activation=cfg.network.activation)
    blue_params = _load_actor_params(BLUE_CKPT, blue_actor, obs_dim)

    # Per-arm seed-0 reds
    red_params = {
        a: _load_actor_params(str(PHASE1_DIR / f"red_{a}_seed0.npz"),
                              red_actor, obs_dim)
        for a in ARMS
    }

    # ---- Tier 1 #1 / Tier 2 #5: post-process eval_stats.npz ---------------
    print("[1/5] post-processing existing eval_stats.npz ...")
    ev = np.load(PHASE1_DIR / "eval_stats.npz", allow_pickle=True)
    arms_ev = [str(a) for a in ev["arms"]]
    blue_R = np.asarray(ev["blue_R"], dtype=np.float64)     # [3, 600]
    coverage = np.asarray(ev["coverage"], dtype=np.float64)
    rng = np.random.default_rng(2026)

    n_arms = len(arms_ev); n_eps_eval = blue_R.shape[1]
    iqm_vals = np.array([iqm(blue_R[i]) for i in range(n_arms)])
    iqm_ci = np.array([iqm_bootstrap_ci(blue_R[i]) for i in range(n_arms)])

    # Performance profile: P(blue_R[arm] >= τ) on a fine τ grid
    taus = np.linspace(blue_R.min() - 0.05, blue_R.max() + 0.05, 256)
    perf_profile = np.zeros((n_arms, taus.size))
    for i in range(n_arms):
        for j, tau in enumerate(taus):
            perf_profile[i, j] = (blue_R[i] >= tau).mean()

    # Probability of improvement (paired) and A12 + Cohen's d on Δ
    pair_stats = {}
    for a, b in PAIRS:
        x = blue_R[arms_ev.index(a)]
        y = blue_R[arms_ev.index(b)]
        d = x - y
        cohen = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
        a12 = vargha_delaney_a12(x, y)
        # Probability that BLUE does better against b than against a (i.e. b is a weaker attacker).
        # Equivalent: P(x < y).  Upper paragraph in HTML uses both phrasings.
        p_b_beats_a = prob_improvement_paired(x, y)
        # bootstrap CI for P_b_beats_a
        n_boot = 5_000
        boots = np.empty(n_boot)
        for k in range(n_boot):
            idx = rng.integers(0, x.size, x.size)
            boots[k] = prob_improvement_paired(x[idx], y[idx])
        pi_lo, pi_hi = np.percentile(boots, [2.5, 97.5])
        pair_stats[f"{a}-{b}"] = dict(
            mean_diff=float(d.mean()), cohen_d=cohen, a12=a12,
            p_improvement=p_b_beats_a,
            p_improvement_lo=float(pi_lo), p_improvement_hi=float(pi_hi),
        )

    # ---- Tier 2 #4: state-visitation JS matrix from action_probe.npz ------
    print("[2/5] state-visitation JS matrix ...")
    pr = np.load(PHASE1_DIR / "action_probe.npz", allow_pickle=True)
    pos = np.asarray(pr["positions"])      # [3, eps, T, n_red, 2]
    active = np.asarray(pr["active_mask"]) # [3, eps, T]
    H = W = 16
    visit = np.zeros((n_arms, H, W), dtype=np.float64)
    for ai in range(n_arms):
        rows = pos[ai, ..., 0]
        cols = pos[ai, ..., 1]
        mask = np.broadcast_to(active[ai][..., None],
                               rows.shape).astype(bool)
        rr = np.clip(rows[mask], 0, H - 1)
        cc = np.clip(cols[mask], 0, W - 1)
        np.add.at(visit[ai], (rr, cc), 1.0)
        visit[ai] /= max(visit[ai].sum(), 1.0)
    flat = visit.reshape(n_arms, -1)
    visit_js = np.zeros((n_arms, n_arms))
    for i in range(n_arms):
        for j in range(n_arms):
            visit_js[i, j] = js_divergence(flat[i], flat[j])

    # ---- Tier 2 #3: cross-play (trained vs random blue) -------------------
    print("[3/5] cross-play (trained / random blue) ...")
    eval_roll = _build_eval_rollout(
        env, blue_actor, red_actor, n_blue, n_red,
        max_steps, num_actions, cfg.enforce_connectivity,
    )
    keys_xplay = jax.random.split(jax.random.PRNGKey(123), N_XPLAY_EPS)
    random_blue = blue_actor.init(jax.random.PRNGKey(RANDOM_BLUE_KEY),
                                  jnp.zeros(obs_dim))
    blue_pool = {"trained": blue_params, "random": random_blue}
    xplay = {bn: {a: np.zeros(N_XPLAY_EPS, dtype=np.float64)
                   for a in ARMS}
             for bn in blue_pool}
    t0 = time.time()
    for bn, bp in blue_pool.items():
        for a in ARMS:
            for ei in range(N_XPLAY_EPS):
                br, _rr, _cv = eval_roll(bp, red_params[a], keys_xplay[ei])
                xplay[bn][a][ei] = float(br)
        print(f"   blue={bn}: I={xplay[bn]['I'].mean():+.3f}  "
              f"W={xplay[bn]['W'].mean():+.3f}  F={xplay[bn]['F'].mean():+.3f}")
    print(f"   xplay done in {time.time()-t0:.1f}s")

    # Pack: 3 reds × 2 blues × N_XPLAY_EPS  paired across reds (shared keys)
    xplay_arr = np.zeros((n_arms, 2, N_XPLAY_EPS), dtype=np.float64)
    blue_names = ["trained", "random"]
    for ai, a in enumerate(ARMS):
        for bi, bn in enumerate(blue_names):
            xplay_arr[ai, bi] = xplay[bn][a]

    # ---- Tier 1 #2(a): CKA between hidden activations ---------------------
    print("[4/5] CKA(Dense_1) and CKA(Dense_2) ...")
    obs_collector = _build_obs_collector(
        env, blue_actor, red_actor, n_blue, n_red,
        max_steps, cfg.enforce_connectivity,
    )
    probe_obs_chunks: list[np.ndarray] = []
    n_collected = 0
    cka_key = jax.random.PRNGKey(CKA_KEY)
    while n_collected < N_PROBE_OBS:
        cka_key, sub = jax.random.split(cka_key)
        red_obs_seq = obs_collector(blue_params, red_params["I"], sub)
        red_obs_flat = np.asarray(red_obs_seq).reshape(-1, obs_dim)
        probe_obs_chunks.append(red_obs_flat)
        n_collected += red_obs_flat.shape[0]
    probe_obs = np.concatenate(probe_obs_chunks, axis=0)[:N_PROBE_OBS]
    print(f"   probe buffer: {probe_obs.shape}")

    h1_per_arm: dict[str, np.ndarray] = {}
    h2_per_arm: dict[str, np.ndarray] = {}
    for a in ARMS:
        h1, h2 = _actor_hidden_activations(red_params[a], probe_obs)
        h1_per_arm[a] = h1
        h2_per_arm[a] = h2

    cka_d1 = np.zeros((n_arms, n_arms))
    cka_d2 = np.zeros((n_arms, n_arms))
    for i, a in enumerate(ARMS):
        for j, b in enumerate(ARMS):
            cka_d1[i, j] = linear_cka(h1_per_arm[a], h1_per_arm[b])
            cka_d2[i, j] = linear_cka(h2_per_arm[a], h2_per_arm[b])

    # ---- Tier 1 #2(b): linear-mode-connectivity barriers ------------------
    print("[5/5] LMC barrier matrix (3 pairs × 11 alphas) ...")
    keys_lmc = jax.random.split(jax.random.PRNGKey(LMC_KEY), N_LMC_EPS)
    lmc_curves = {}      # pair -> [11] mean blue_R per alpha
    t0 = time.time()
    for a, b in PAIRS:
        pa, pb = red_params[a], red_params[b]
        curve = np.zeros(ALPHA_GRID.size)
        for ai_alpha, alpha in enumerate(ALPHA_GRID):
            interp = jax.tree_util.tree_map(
                lambda x, y, _a=alpha: (1.0 - _a) * x + _a * y, pa, pb)
            vals = np.zeros(N_LMC_EPS)
            for ei in range(N_LMC_EPS):
                br, _rr, _cv = eval_roll(blue_params, interp, keys_lmc[ei])
                vals[ei] = float(br)
            curve[ai_alpha] = vals.mean()
        lmc_curves[f"{a}-{b}"] = curve
        endpoints = 0.5 * (curve[0] + curve[-1])
        # Barrier here = how far blue_R DIPS below the endpoint mean along the
        # interpolation path (positive = there's a performance trough = different
        # basins).  For RL "loss" is unsigned, so we work in blue_R directly.
        barrier = float(endpoints - curve.min())
        print(f"   {a}-{b}: ep_mean = {endpoints:+.3f}  min(α) = "
              f"{curve.min():+.3f}  barrier = {barrier:+.3f}")
    print(f"   LMC done in {time.time()-t0:.1f}s")

    # ---- Save -------------------------------------------------------------
    out_path = PHASE1_DIR / "advanced_analysis.npz"
    np.savez(
        out_path,
        # rliable
        arms=np.asarray(ARMS),
        blue_R=blue_R,
        coverage=coverage,
        iqm=iqm_vals,
        iqm_ci=iqm_ci,
        perf_profile_taus=taus,
        perf_profile=perf_profile,
        # paired effect sizes
        pair_keys=np.asarray([f"{a}-{b}" for a, b in PAIRS]),
        cohen_d=np.array([pair_stats[f"{a}-{b}"]["cohen_d"] for a, b in PAIRS]),
        a12=np.array([pair_stats[f"{a}-{b}"]["a12"] for a, b in PAIRS]),
        p_improvement=np.array([pair_stats[f"{a}-{b}"]["p_improvement"]
                                for a, b in PAIRS]),
        p_improvement_lo=np.array(
            [pair_stats[f"{a}-{b}"]["p_improvement_lo"] for a, b in PAIRS]),
        p_improvement_hi=np.array(
            [pair_stats[f"{a}-{b}"]["p_improvement_hi"] for a, b in PAIRS]),
        # state-visitation JS
        visit_grid=visit,            # [3, 16, 16]
        visit_js=visit_js,           # [3, 3]
        # cross-play
        xplay_blue_R=xplay_arr,      # [3, 2, N_XPLAY_EPS]
        xplay_blues=np.asarray(blue_names),
        xplay_reds=np.asarray(ARMS),
        # CKA
        cka_dense_1=cka_d1,
        cka_dense_2=cka_d2,
        # LMC
        lmc_alphas=ALPHA_GRID,
        lmc_pairs=np.asarray([f"{a}-{b}" for a, b in PAIRS]),
        lmc_curves=np.stack([lmc_curves[f"{a}-{b}"] for a, b in PAIRS]),
        n_lmc_eps=np.int32(N_LMC_EPS),
        n_xplay_eps=np.int32(N_XPLAY_EPS),
        n_probe_obs=np.int32(N_PROBE_OBS),
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
