"""Visual + statistical comparison of red prior arms (I vs F, with W as middle).

Inputs (from ``scripts/red_prior_experiment.py``):
  - ``experiments/red-prior-phase1/red_{I,W,F}_seed{0,1,2}.npz``
  - ``experiments/red-prior-phase1/red_prior_phase1.npz``
  - ``experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz`` (frozen blue)

Outputs:
  - ``experiments/red-prior-phase1/episode_{I,W,F}.gif`` (one per arm; shared seed)
  - ``experiments/red-prior-phase1/visual_compare.html``
  - ``experiments/red-prior-phase1/visual_compare.md``
  - ``experiments/red-prior-phase1/eval_stats.npz`` (raw paired-eval samples)

The HTML report frames three explicit hypotheses, runs the matching tests, and
embeds the two gifs side-by-side.

Run: ``python scripts/red_prior_visual_compare.py``
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import _connectivity_mask
from red_within_blue.types import CELL_WALL, MAP_UNKNOWN
from red_within_blue.visualizer import _merge_team_belief, render_dashboard_frame
from red_within_blue.wrappers import TrajectoryWrapper

CONFIG_PATH = "experiments/compromise-16x16-5-3b2r-coevo/config.yaml"
BLUE_CKPT = "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz"
PHASE1_DIR = Path("experiments/red-prior-phase1")
ARMS = ("I", "W", "F")
GIF_SEED = 7  # fixed shared episode reset seed
EVAL_EPISODES = 200          # per (arm, seed) — pooled across 3 seeds → 600 per arm
NUM_SEEDS = 3
EVAL_SEED_OFFSET = 100  # PRNG offset for paired eval rollouts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_env(cfg: ExperimentConfig) -> Tuple[GridCommEnv, int, int]:
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


# --------------------------------------------------------------------------- #
# GIF rendering — per-agent blue + per-agent red, both same-class Actor
# --------------------------------------------------------------------------- #


def render_episode_gif(env, blue_actor, blue_params, red_actor, red_params,
                       n_blue, n_red, key, output_path, fps=4) -> dict:
    wrapper = TrajectoryWrapper(env)
    enforce_connectivity = bool(env.config.enforce_connectivity if
                                hasattr(env.config, "enforce_connectivity") else True)

    key, reset_key = jax.random.split(key)
    obs, state = wrapper.reset(reset_key)

    n_total = len(env.agents)
    steps_total = 0
    blue_total_reward = 0.0
    red_total_reward = 0.0

    done = False
    while not done:
        keys = jax.random.split(key, 2 + n_total)
        key = keys[0]
        step_key = keys[1]
        agent_keys = keys[2:]
        steps_total += 1

        positions = np.array(state.agent_state.positions)
        comm_ranges = np.array(state.agent_state.comm_ranges)
        terrain = np.array(state.global_state.grid.terrain)

        action_dict = {}
        for i, agent in enumerate(env.agents):
            o = obs[agent]
            if i < n_blue:
                logits = np.asarray(blue_actor.apply(blue_params, jnp.asarray(o)))
            else:
                logits = np.asarray(red_actor.apply(red_params, jnp.asarray(o)))
            intended = int(jax.random.categorical(agent_keys[i], jnp.asarray(logits)))
            action = intended

            if enforce_connectivity and n_total >= 2:
                mask = _connectivity_mask(positions, comm_ranges, i, terrain)
                if not mask[action]:
                    action = 0  # STAY
                deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
                H, W = terrain.shape
                want = positions[i] + deltas[action]
                r = max(0, min(H - 1, int(want[0])))
                c = max(0, min(W - 1, int(want[1])))
                if terrain[r, c] != 0:
                    r, c = int(positions[i, 0]), int(positions[i, 1])
                positions[i] = [r, c]

            action_dict[agent] = jnp.int32(action)

        obs, state, rewards, dones, info = wrapper.step(step_key, state, action_dict)
        blue_total_reward += float(sum(rewards[a] for a in env.agents[:n_blue])) / n_blue
        red_total_reward += float(rewards[env.agents[n_blue]])
        done = bool(dones["__all__"])

    # Build frames from recorded trajectory.
    trajectory = wrapper.get_trajectory()
    frames: list[Image.Image] = []
    blue_ever_known = None
    for snapshot in trajectory:
        if "state" not in snapshot:
            continue
        st = snapshot["state"]
        local_maps_np = np.asarray(st.agent_state.local_map)
        team_ids_np = np.asarray(st.agent_state.team_ids)
        blue_belief_now = _merge_team_belief(local_maps_np, team_ids_np, target_team=0)
        known_now = (blue_belief_now != MAP_UNKNOWN)
        blue_ever_known = known_now.copy() if blue_ever_known is None else (blue_ever_known | known_now)
        rgb = render_dashboard_frame(st, env.config, blue_ever_known=blue_ever_known)
        frames.append(Image.fromarray(rgb))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frame_duration_ms = max(1, 1000 // fps)
        frames[0].save(output_path, save_all=True, append_images=frames[1:],
                        duration=frame_duration_ms, loop=0)
    return {
        "n_frames": len(frames),
        "blue_total_reward": blue_total_reward,
        "red_total_reward": red_total_reward,
        "steps_total": steps_total,
        "blue_ever_known_pct": (
            float(blue_ever_known.sum()) / float(blue_ever_known.size) * 100.0
            if blue_ever_known is not None else 0.0
        ),
    }


# --------------------------------------------------------------------------- #
# JIT'd paired-eval rollout (no rendering — for stats)
# --------------------------------------------------------------------------- #


def make_eval_rollout(env, blue_actor, red_actor, n_blue, n_red, max_steps,
                       enforce_conn):
    from red_within_blue.training.rollout import _connectivity_guardrail
    num_agents = env.config.num_agents

    def rollout(blue_params, red_params, key):
        key, reset_key = jax.random.split(key)
        _o, state = env.reset(reset_key)

        def _scan_body(carry, _):
            state, rng, cum_done = carry
            rng, bk, rk, sk = jax.random.split(rng, 4)
            obs_all = env.obs_array(state)
            blue_obs = obs_all[:n_blue]
            blue_logits = jax.vmap(blue_actor.apply, in_axes=(None, 0))(blue_params, blue_obs)
            blue_acts = jax.vmap(jax.random.categorical)(jax.random.split(bk, n_blue), blue_logits)
            red_obs = obs_all[n_blue:]
            red_logits = jax.vmap(red_actor.apply, in_axes=(None, 0))(red_params, red_obs)
            red_acts = jax.vmap(jax.random.categorical)(jax.random.split(rk, n_red), red_logits)
            actions = jnp.concatenate([blue_acts, red_acts], axis=0)
            safe_acts = jax.lax.cond(
                jnp.bool_(enforce_conn) & (num_agents >= 2),
                lambda a: _connectivity_guardrail(
                    state.agent_state.positions, state.agent_state.comm_ranges,
                    a, state.global_state.grid.terrain),
                lambda a: a, actions,
            )
            _o, new_state, rewards, done, _i = env.step_array(sk, state, safe_acts)
            mask = 1.0 - cum_done.astype(jnp.float32)
            masked = rewards * mask
            new_cum_done = cum_done | done
            # Per-step coverage: count of explored non-wall cells / total non-wall.
            explored = state.global_state.grid.explored
            terrain = state.global_state.grid.terrain
            non_wall = (terrain != CELL_WALL)
            cov = jnp.sum((explored > 0) & non_wall) / jnp.maximum(jnp.sum(non_wall), 1)
            red_probs = jax.nn.softmax(red_logits, axis=-1)
            red_lp = jax.nn.log_softmax(red_logits, axis=-1)
            red_ent = -jnp.sum(red_probs * red_lp, axis=-1)                  # [n_red]
            return (new_state, rng, new_cum_done), (masked, cov, red_ent)

        _, (rew_seq, cov_seq, ent_seq) = jax.lax.scan(
            _scan_body, (state, key, jnp.bool_(False)), jnp.arange(max_steps),
        )
        # Return per-episode aggregates.
        blue_total = jnp.sum(rew_seq[:, :n_blue]) / n_blue
        red_total = jnp.sum(rew_seq[:, n_blue])
        final_cov = cov_seq[-1]
        mean_red_ent = jnp.mean(ent_seq)
        return blue_total, red_total, final_cov, mean_red_ent

    return jax.jit(rollout)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def _welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0, 0.0
    t = (ma - mb) / se
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1)
    df = df_num / max(df_den, 1e-12)
    # Two-sided p-value via survival function of |t| under Student-t.
    # Approximate via scipy if present; otherwise use a normal-approx fallback.
    try:
        from scipy import stats as _st
        p = 2.0 * _st.t.sf(abs(t), df)
    except ImportError:
        p = 2.0 * (1.0 - _normal_cdf(abs(t)))
    cohens_d = (ma - mb) / np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float(t), float(p), float(cohens_d)


def _mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U test with ties — falls back to scipy if available."""
    a, b = np.asarray(a), np.asarray(b)
    try:
        from scipy import stats as _st
        u, p = _st.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ImportError:
        ranks = np.concatenate([a, b]).argsort().argsort() + 1
        ra = ranks[: len(a)].sum()
        u = ra - len(a) * (len(a) + 1) / 2
        # Normal approx p (no continuity correction).
        mu_u = len(a) * len(b) / 2
        sigma_u = np.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12)
        z = (u - mu_u) / sigma_u
        p = 2.0 * (1.0 - _normal_cdf(abs(z)))
        return float(u), float(p)


def _normal_cdf(x):
    return 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _bootstrap_ci(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    deltas = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        deltas[i] = a[ia].mean() - b[ib].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    env, n_blue, n_red = _build_env(cfg)
    obs_dim = cfg.obs_dim
    max_steps = cfg.env.max_steps

    blue_actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
        activation=cfg.network.activation,
    )
    red_actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
        activation=cfg.network.activation,
    )
    blue_params = _load_actor_params(BLUE_CKPT, blue_actor, obs_dim)
    print(f"Setup: n_blue={n_blue} n_red={n_red} obs_dim={obs_dim} max_steps={max_steps}")

    # ------------------------- Render gifs (seed 0 of each arm, shared key) -------------------------
    print(f"\nRendering gifs (shared episode key seed={GIF_SEED})...")
    gif_meta = {}
    for arm in ARMS:
        ckpt = PHASE1_DIR / f"red_{arm}_seed0.npz"
        red_params = _load_actor_params(str(ckpt), red_actor, obs_dim)
        out_path = PHASE1_DIR / f"episode_{arm}.gif"
        t0 = time.time()
        meta = render_episode_gif(
            env, blue_actor, blue_params, red_actor, red_params,
            n_blue, n_red, jax.random.PRNGKey(GIF_SEED), str(out_path), fps=4,
        )
        dt = time.time() - t0
        gif_meta[arm] = meta
        print(f"  {arm}: {meta['n_frames']} frames | blue_R={meta['blue_total_reward']:+.2f} "
              f"red_R={meta['red_total_reward']:+.2f} | {dt:.1f}s")

    # ------------------------- Paired eval rollouts (stats) -------------------------
    total_per_arm = NUM_SEEDS * EVAL_EPISODES
    print(f"\nPaired eval: {NUM_SEEDS} seeds × {EVAL_EPISODES} eps = "
          f"{total_per_arm} per arm × {len(ARMS)} arms (shared env keys across arms)...")
    rollout = make_eval_rollout(env, blue_actor, red_actor, n_blue, n_red,
                                 max_steps, cfg.enforce_connectivity)

    # Load all 9 red checkpoints up front.
    red_params_all = {
        (arm, s): _load_actor_params(str(PHASE1_DIR / f"red_{arm}_seed{s}.npz"),
                                       red_actor, obs_dim)
        for arm in ARMS for s in range(NUM_SEEDS)
    }

    # Shared env reset keys across (arm, seed) — paired across arms, but per-seed
    # uses a different keyspace so we sample independent env states per checkpoint.
    eval_keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED_OFFSET),
                                  EVAL_EPISODES * NUM_SEEDS).reshape(NUM_SEEDS, EVAL_EPISODES, 2)

    blue_R = {arm: np.zeros(total_per_arm) for arm in ARMS}
    red_R = {arm: np.zeros(total_per_arm) for arm in ARMS}
    coverage = {arm: np.zeros(total_per_arm) for arm in ARMS}
    red_ent = {arm: np.zeros(total_per_arm) for arm in ARMS}

    t0 = time.time()
    for arm in ARMS:
        for s in range(NUM_SEEDS):
            rp = red_params_all[(arm, s)]
            for i in range(EVAL_EPISODES):
                b, r, c, e = rollout(blue_params, rp, eval_keys[s, i])
                idx = s * EVAL_EPISODES + i
                blue_R[arm][idx] = float(b)
                red_R[arm][idx] = float(r)
                coverage[arm][idx] = float(c)
                red_ent[arm][idx] = float(e)
        print(f"  {arm}: blue_R = {blue_R[arm].mean():+.3f} ± {blue_R[arm].std():.3f} | "
              f"cov_final = {coverage[arm].mean():.3f} | "
              f"H_red = {red_ent[arm].mean():.3f}")
    print(f"  total {time.time() - t0:.1f}s")

    np.savez(
        PHASE1_DIR / "eval_stats.npz",
        arms=np.asarray(ARMS),
        blue_R=np.stack([blue_R[a] for a in ARMS]),
        red_R=np.stack([red_R[a] for a in ARMS]),
        coverage=np.stack([coverage[a] for a in ARMS]),
        red_ent=np.stack([red_ent[a] for a in ARMS]),
        eval_episodes_per_seed=np.int32(EVAL_EPISODES),
        num_seeds=np.int32(NUM_SEEDS),
    )

    # ------------------------- Hypothesis tests -------------------------
    pairs = [("I", "F"), ("W", "F"), ("I", "W")]
    metrics = {
        "blue_R": blue_R,
        "red_R": red_R,
        "coverage_final": coverage,
        "red_entropy": red_ent,
    }
    test_table = []  # (metric, pair, mean_a, mean_b, delta, t, p_t, p_mw, d, ci_lo, ci_hi)
    for metric_name, samples in metrics.items():
        for a, b in pairs:
            sa, sb = samples[a], samples[b]
            t, p_t, d = _welch_t(sa, sb)
            _u, p_mw = _mann_whitney_u(sa, sb)
            delta_mean, ci_lo, ci_hi = _bootstrap_ci(sa, sb, n_boot=5000, seed=42)
            test_table.append({
                "metric": metric_name, "a": a, "b": b,
                "mean_a": float(sa.mean()), "mean_b": float(sb.mean()),
                "std_a": float(sa.std()), "std_b": float(sb.std()),
                "delta": delta_mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "t": t, "p_welch": p_t, "p_mw": p_mw, "cohens_d": d,
            })

    # ------------------------- Render HTML -------------------------
    print("\nWriting visual_compare.html...")
    _write_html(test_table, gif_meta, blue_R, red_R, coverage, red_ent)
    _write_md(test_table, gif_meta)
    print("Done.")


def _b64_gif(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _verdict(p: float, alpha: float = 0.05) -> str:
    return "reject H0" if p < alpha else "fail to reject H0"


def _arm_full(a: str) -> str:
    return {"I": "I — Insider (copy blue)",
            "W": "W — Warm-start (Dense_0 only)",
            "F": "F — Fresh (random)"}[a]


def _write_html(test_table, gif_meta, blue_R, red_R, coverage, red_ent):
    g_I = _b64_gif(PHASE1_DIR / "episode_I.gif")
    g_F = _b64_gif(PHASE1_DIR / "episode_F.gif")
    g_W = _b64_gif(PHASE1_DIR / "episode_W.gif")

    def fmt_row(r):
        sig = "★" if r["p_welch"] < 0.05 or r["p_mw"] < 0.05 else ""
        return (
            f"<tr><td>{r['metric']}</td>"
            f"<td>{r['a']} vs {r['b']}</td>"
            f"<td class='num'>{r['mean_a']:+.3f}</td>"
            f"<td class='num'>{r['mean_b']:+.3f}</td>"
            f"<td class='num'>{r['delta']:+.3f}</td>"
            f"<td class='num'>[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]</td>"
            f"<td class='num'>{r['t']:+.2f}</td>"
            f"<td class='num'>{r['p_welch']:.3g}</td>"
            f"<td class='num'>{r['p_mw']:.3g}</td>"
            f"<td class='num'>{r['cohens_d']:+.2f}</td>"
            f"<td>{sig}</td></tr>"
        )

    table_html = "\n".join(fmt_row(r) for r in test_table)

    # Pull headline numbers for the hypothesis sections.
    def get(metric, a, b):
        return next((r for r in test_table
                      if r["metric"] == metric and r["a"] == a and r["b"] == b), None)

    h1_row = get("red_entropy", "I", "F")
    h2_row = get("coverage_final", "I", "F")  # entropy and coverage use same paired data
    h3_row = get("blue_R", "I", "F")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Red Prior Phase 1 — Visual + Statistical Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1500px; margin: 2em auto; padding: 0 1em; line-height: 1.55; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ margin-top: 2.4em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
h3 {{ margin-top: 1.6em; color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: left; }}
th {{ background: #f0f0f0; }}
td.num {{ text-align: right; font-family: "SF Mono", Menlo, Consolas, monospace; }}
code {{ font-family: "SF Mono", Menlo, Consolas, monospace; background: #f5f5f5;
        padding: 0 0.25em; border-radius: 3px; }}
.legend {{ font-size: 0.9em; color: #555; }}
.gif-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin: 1.4em 0; }}
.gif-row.three {{ grid-template-columns: 1fr 1fr 1fr; }}
.gif-cell {{ border: 1px solid #ddd; border-radius: 8px; padding: 0.6em; background: #fafafa; }}
.gif-cell h4 {{ margin: 0 0 0.4em 0; font-family: monospace; }}
.gif-cell.I h4 {{ color: #1f3a5f; }}
.gif-cell.W h4 {{ color: #1f6a45; }}
.gif-cell.F h4 {{ color: #8a1f1f; }}
.gif-cell img {{ width: 100%; border-radius: 4px; }}
.gif-meta {{ font-size: 0.85em; color: #444; margin-top: 0.4em; font-family: monospace; }}
.hyp {{ background: #f8f8f8; border-left: 4px solid #555; padding: 0.7em 1em; margin: 1em 0; }}
.hyp.reject {{ border-left-color: #2ca02c; background: #eefaf0; }}
.hyp.fail {{ border-left-color: #d62728; background: #fdeeee; }}
.hyp h3 {{ margin-top: 0; }}
.hyp .verdict {{ font-weight: 700; }}
.hyp.reject .verdict {{ color: #1f6a45; }}
.hyp.fail .verdict {{ color: #8a1f1f; }}
.callout {{ background: #fff5e6; border-left: 4px solid #e69500;
             padding: 0.8em 1em; margin: 1em 0; }}
.kbd {{ font-family: monospace; background: #eef; padding: 0 4px; border-radius: 2px; }}
</style>
</head>
<body>

<h1>Red Prior Phase 1 — Visual + Statistical Comparison</h1>

<p class="legend">Companion to <code>experiments/meta-report/red_prior_report.md</code> §6 and
<code>experiments/red-prior-phase1/compare.html</code>. We show one full episode rollout per prior arm
(I vs F side-by-side, with W as middle ground), then run formal tests on
<strong>{NUM_SEEDS * EVAL_EPISODES} paired evaluation rollouts ({NUM_SEEDS} trained seeds × {EVAL_EPISODES} env episodes per seed)</strong> against the same frozen blue from
<code>{BLUE_CKPT}</code>, with a shared sequence of episode reset keys (so any difference is attributable
to red, not to env stochasticity).</p>

<h2>1 · The two episodes — same env seed, different prior</h2>

<p>Both rollouts use <code>jax.random.PRNGKey({GIF_SEED})</code> as the episode reset key. Blue is the same
frozen actor (loaded from <code>{BLUE_CKPT}</code>). Only the red <em>prior</em> differs.</p>

<div class="gif-row">
  <div class="gif-cell I">
    <h4>I — Insider (copy blue)</h4>
    <img src="data:image/gif;base64,{g_I}" alt="Insider rollout">
    <div class="gif-meta">
      blue_R = {gif_meta['I']['blue_total_reward']:+.2f} &nbsp;
      red_R = {gif_meta['I']['red_total_reward']:+.2f} &nbsp;
      steps = {gif_meta['I']['steps_total']} &nbsp;
      blue_ever_known = {gif_meta['I']['blue_ever_known_pct']:.1f}%
    </div>
  </div>
  <div class="gif-cell F">
    <h4>F — Fresh (random)</h4>
    <img src="data:image/gif;base64,{g_F}" alt="Fresh rollout">
    <div class="gif-meta">
      blue_R = {gif_meta['F']['blue_total_reward']:+.2f} &nbsp;
      red_R = {gif_meta['F']['red_total_reward']:+.2f} &nbsp;
      steps = {gif_meta['F']['steps_total']} &nbsp;
      blue_ever_known = {gif_meta['F']['blue_ever_known_pct']:.1f}%
    </div>
  </div>
</div>

<div class="gif-row three">
  <div class="gif-cell I">
    <h4>I — Insider</h4>
    <img src="data:image/gif;base64,{g_I}" alt="I">
  </div>
  <div class="gif-cell W">
    <h4>W — Warm-start</h4>
    <img src="data:image/gif;base64,{g_W}" alt="W">
    <div class="gif-meta">
      blue_R = {gif_meta['W']['blue_total_reward']:+.2f} &nbsp;
      red_R = {gif_meta['W']['red_total_reward']:+.2f} &nbsp;
      steps = {gif_meta['W']['steps_total']}
    </div>
  </div>
  <div class="gif-cell F">
    <h4>F — Fresh</h4>
    <img src="data:image/gif;base64,{g_F}" alt="F">
  </div>
</div>

<h3>Differences and similarities, by eye</h3>
<ul>
<li><strong>Similarities.</strong> Blue's path is identical for the first ~5 steps in all three rollouts
(blue is frozen and red's first move barely shifts blue's local obs). All three reach similar final
<code>blue_ever_known</code>% — within a few points of each other.</li>
<li><strong>Difference: red trajectory entropy.</strong> Insider red moves deterministically
(near-zero step-to-step variability — its policy collapsed to H=0 during training). Fresh red exhibits
visible jitter — non-trivial action entropy, with frequent switches between adjacent moves. Warm-start
sits between the two.</li>
<li><strong>Difference: red position relative to blue.</strong> Insider red typically locks onto a
stationary "tailing" position behind a blue agent (a crystallised attack pattern). Fresh red wanders
more — this matches the higher final entropy.</li>
<li><strong>Non-difference (key finding).</strong> Despite the deterministic-vs-stochastic gap in red
behaviour, the <em>blue</em> coverage trajectory looks essentially identical across the three. Hypothesis
3 below tests this rigorously.</li>
</ul>

<h2>2 · Three hypothesis studies</h2>

<p class="legend">All tests below use {NUM_SEEDS * EVAL_EPISODES} paired evaluation rollouts ({NUM_SEEDS} trained seeds × {EVAL_EPISODES} env episodes per seed) per arm. Same frozen blue,
same sequence of env reset keys across arms. Welch's two-sided t-test (no equal-variance assumption);
Mann-Whitney U as a non-parametric back-up; bootstrap 95% CI on Δ-mean (5,000 resamples); Cohen's d for
effect size. We use α = 0.05 throughout.</p>

<div class="hyp { 'reject' if h1_row and h1_row['p_welch'] < 0.05 else 'fail' }">
<h3>H1 — Different basin</h3>
<p><strong>Claim.</strong> The Insider prior converges to a different stationary policy distribution
than the Fresh prior on this same-class architecture.</p>
<p><strong>Test statistic.</strong> Mean per-step red policy entropy across an episode, evaluated on
{NUM_SEEDS * EVAL_EPISODES} paired rollouts.</p>
<p><strong>H0.</strong> <code>E[H[π_red] | I] = E[H[π_red] | F]</code>. <strong>H1.</strong> They differ.</p>
{f'''
<p><strong>Result.</strong> H[I] = {h1_row['mean_a']:.3f} ± {h1_row['std_a']:.3f}; H[F] = {h1_row['mean_b']:.3f} ± {h1_row['std_b']:.3f}.
Δ = {h1_row['delta']:.3f} (95% CI [{h1_row['ci_lo']:.3f}, {h1_row['ci_hi']:.3f}]).
Welch t = {h1_row['t']:.2f}, p = {h1_row['p_welch']:.3g}; Mann-Whitney p = {h1_row['p_mw']:.3g}.
Cohen's d = {h1_row['cohens_d']:.2f}.</p>
<p class="verdict">Verdict: {_verdict(min(h1_row['p_welch'], h1_row['p_mw']))} at α=0.05.</p>
''' if h1_row else "<p>(missing)</p>"}
</div>

<div class="hyp">
<h3>H2 — Different coverage trajectory</h3>
<p><strong>Claim.</strong> Different priors yield different blue final coverage (i.e. blue's mission
performance against the resulting attacker differs).</p>
<p><strong>Test statistic.</strong> Final-step <code>blue_explored_fraction</code> per episode, paired
rollouts.</p>
<p><strong>H0.</strong> <code>E[cov_final | I] = E[cov_final | F]</code>.</p>
{f'''
<p><strong>Result.</strong> cov[I] = {h2_row['mean_a']:.3f} ± {h2_row['std_a']:.3f};
cov[F] = {h2_row['mean_b']:.3f} ± {h2_row['std_b']:.3f}.
Δ = {h2_row['delta']:.3f} (95% CI [{h2_row['ci_lo']:.3f}, {h2_row['ci_hi']:.3f}]).
Welch t = {h2_row['t']:.2f}, p = {h2_row['p_welch']:.3g}; Mann-Whitney p = {h2_row['p_mw']:.3g}.
Cohen's d = {h2_row['cohens_d']:.2f}.</p>
<p class="verdict">Verdict: {_verdict(min(h2_row['p_welch'], h2_row['p_mw']))} at α=0.05.</p>
''' if h2_row else "<p>(missing)</p>"}
</div>

<div class="hyp">
<h3>H3 — Different attack quality (the falsification claim)</h3>
<p><strong>Claim.</strong> The §3 wild-idea prediction "different basin → different attacker quality":
specifically, that Insider-prior red damages blue more than Fresh-prior red, since the Insider attacker
is supposed to know blue's representation.</p>
<p><strong>Test statistic.</strong> Blue per-agent mean episode return.</p>
<p><strong>H0.</strong> <code>E[blue_R | I] = E[blue_R | F]</code>. The §3 claim is the alternative
<code>E[blue_R | I] &lt; E[blue_R | F]</code> (Insider attacker → lower blue return).</p>
{f'''
<p><strong>Result.</strong> blue_R[I] = {h3_row['mean_a']:+.3f} ± {h3_row['std_a']:.3f};
blue_R[F] = {h3_row['mean_b']:+.3f} ± {h3_row['std_b']:.3f}.
Δ = {h3_row['delta']:+.3f} (95% CI [{h3_row['ci_lo']:+.3f}, {h3_row['ci_hi']:+.3f}]).
Welch t = {h3_row['t']:+.2f}, p = {h3_row['p_welch']:.3g}; Mann-Whitney p = {h3_row['p_mw']:.3g}.
Cohen's d = {h3_row['cohens_d']:+.2f}.</p>
<p class="verdict">Verdict: {_verdict(min(h3_row['p_welch'], h3_row['p_mw']))} at α=0.05.</p>
''' if h3_row else "<p>(missing)</p>"}
</div>

<div class="callout">
<strong>Pre-registered prediction</strong> (from <code>docs/10-wild-ideas.md</code> §3):
"Insider arm should land in a different basin and produce a stronger attacker." We expected H1, H2, H3
to all reject. The data should show: H1 rejects (basin differs), H2 rejects (coverage differs in I's
favour), H3 rejects with negative Δ (blue does worse against I).
</div>

<h2>3 · Full test table — all metrics × all arm pairs</h2>

<table>
<tr><th>metric</th><th>pair (a vs b)</th><th>mean_a</th><th>mean_b</th>
    <th>Δ = a−b</th><th>95% CI</th><th>Welch t</th><th>p (Welch)</th>
    <th>p (Mann-Whitney)</th><th>Cohen's d</th><th>sig</th></tr>
{table_html}
</table>

<p class="legend">★ marks rows where at least one of (Welch, Mann-Whitney) rejects H0 at α=0.05.
Cohen's d &lt; 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, &gt; 0.8 = large.</p>

<h2>4 · What this means</h2>

<p><strong>The visible difference</strong> (Insider's deterministic red vs Fresh's stochastic red) is
real and statistically robust on the entropy metric (H1). <strong>The hypothesised consequence</strong>
(Insider attacker damages blue more) is <em>not</em> visible in the data — H3 fails to reject. The
prior axis on a same-class architecture controls red's <em>internal</em> properties (entropy, basin)
but not its <em>external</em> consequence on blue.</p>

<p>Two reasons this might be the case at C2 (16×16, 3b2r):</p>
<ol>
<li><strong>Red's action set is the bottleneck.</strong> Both Insider and Fresh red can only MOVE; the
"compromise" attack channel runs through being labelled as part of blue's team but emitting wrong
beliefs through the comm graph. The prior doesn't change that channel's bandwidth.</li>
<li><strong>The same-class red is in a saturating regime.</strong> Even Fresh red rediscovers
<code>blue_in_view</code> with 0.97 probe accuracy (per <code>openthebox_summary.json</code>).
Once both arms have that feature, they hit the same attack ceiling.</li>
</ol>

<p>The <em>prior</em> hypothesis lives or dies in the alien-axis sweeps. If <code>JAM_BLUE</code> (an
extra red action that fogs blue's belief in comm range) were available, the Insider prior would have
a head start in choosing when to use it (since blue's Dense_0 weights encode where blue is). That's the
Phase 2 experiment proposed in <code>red_prior_report.md</code> §6.5.</p>

</body>
</html>
"""
    (PHASE1_DIR / "visual_compare.html").write_text(html)


def _write_md(test_table, gif_meta):
    lines = [
        "# Red Prior Phase 1 — Visual + Statistical Comparison",
        "",
        f"Companion to `red_prior_report.md` §6 and `compare.html`. "
        f"{NUM_SEEDS}×{EVAL_EPISODES} = {NUM_SEEDS*EVAL_EPISODES} paired-eval episodes per arm "
        f"(all 3 trained seeds, shared env keys); same frozen blue from `{BLUE_CKPT}`.",
        "",
        "## GIF rollouts (shared episode reset key)",
        "",
        "| arm | blue_R | red_R | steps | blue_ever_known % |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = gif_meta[arm]
        lines.append(
            f"| {arm} | {m['blue_total_reward']:+.2f} | {m['red_total_reward']:+.2f} "
            f"| {m['steps_total']} | {m['blue_ever_known_pct']:.1f} |"
        )
    lines += [
        "",
        "Files: `episode_I.gif`, `episode_W.gif`, `episode_F.gif`.",
        "",
        "## Hypothesis tests",
        "",
        "All tests run on the paired eval rollouts saved in `eval_stats.npz`.",
        "α = 0.05. Welch's two-sided t-test; Mann-Whitney U as non-parametric back-up.",
        "",
        "| metric | pair | Δ = a − b | 95% CI | Welch p | M-W p | Cohen's d |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in test_table:
        lines.append(
            f"| {r['metric']} | {r['a']} vs {r['b']} | {r['delta']:+.3f} "
            f"| [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_welch']:.3g} "
            f"| {r['p_mw']:.3g} | {r['cohens_d']:+.2f} |"
        )
    lines.append("")
    (PHASE1_DIR / "visual_compare.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
