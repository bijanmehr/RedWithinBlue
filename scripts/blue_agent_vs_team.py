"""Counterfactual saliency split for blue at C2: agent vs team.

For each blue agent at every step we maintain two local_maps in parallel:
  • merged  — the env's actual local_map (own surveys ∪ teammates' via comm)
  • solo    — same agent's surveys but with the comm graph severed

At every step we compute |∂π/∂o| on the merged obs (what the agent acts on).
Then per dim of the seen / map_frac blocks, we split the saliency by the
fraction of the input that came from the agent vs from teammates:

  agent_frac_d  = solo_d  / max(merged_d, ε)
  team_frac_d   = 1 - agent_frac_d        (clipped to [0, 1])
  agent_attr_d  = |grad_d| · agent_frac_d
  team_attr_d   = |grad_d| · team_frac_d

For scan / position / agent id (channels the comm graph never touches),
agent_attr = |grad|, team_attr = 0.

Output: experiments/meta-report/xai_blue_agent_vs_team.json with the same
shape as joint_red_cross.{own,cross}_mean_per_block, plus headline scalars.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from meta_report import SETUPS, _load_blue
from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.rewards_training import normalized_competitive_reward
from red_within_blue import agents as agents_mod
from red_within_blue.types import MAP_UNKNOWN

OUT = ROOT / "experiments" / "meta-report" / "xai_blue_agent_vs_team.json"

BLOCKS = [
    ("scan",     0,  9),
    ("seen",     9,  18),
    ("map_frac", 18, 19),
    ("norm_pos", 19, 21),
    ("uid",      21, 22),
]
COMM_BLOCKS = {"seen", "map_frac"}
SEED = 0


def _solo_seen_window(local_map_solo: np.ndarray, pos: np.ndarray, view_r: int) -> np.ndarray:
    """3×3 known-mask window around pos from a *solo* local_map (no comm)."""
    H, W = local_map_solo.shape
    d = 2 * view_r + 1
    padded = np.pad((local_map_solo != MAP_UNKNOWN).astype(np.float32),
                    pad_width=view_r, mode="constant", constant_values=1.0)
    return padded[pos[0]:pos[0] + d, pos[1]:pos[1] + d].reshape(-1)


def main() -> None:
    setup = next(s for s in SETUPS if s.key == "C2")
    cfg = ExperimentConfig.from_yaml(setup.config)
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red

    env = GridCommEnv(cfg.to_env_config(), reward_fn=normalized_competitive_reward)
    obs_dim = env.obs_dim
    assert obs_dim == 23
    H, W = cfg.env.grid_height, cfg.env.grid_width
    view_r = env.view_radius
    survey_r = env.survey_radius

    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)

    # We also need a (deterministic) policy to drive red, otherwise blue's
    # rollout drifts vs the canonical seed-0 episode. For simplicity we drive
    # red with the blue policy: this is fine since we only consume blue
    # gradients, and the env is the same env they use for the headline figure.
    @jax.jit
    def _blue_logit_grad(bp, obs, action):
        def f(o):
            return blue_actor.apply(bp, o)[action]
        return jax.grad(f)(obs)

    @jax.jit
    def _blue_sample(bp, obs, key):
        return jax.random.categorical(key, blue_actor.apply(bp, obs))

    key = jax.random.PRNGKey(SEED)
    obs_dict, state = env.reset(key)
    team_ids = np.asarray(state.agent_state.team_ids)
    blue_idx = np.where(team_ids == 0)[0]

    # Solo local_maps — initialised to UNKNOWN, then immediately updated with
    # the post-reset local_scan (so the first-step obs is consistent with the
    # env's first-step merged obs in the no-teammate baseline).
    local_map_solo = np.full((n_total, H, W), MAP_UNKNOWN, dtype=np.int32)
    local_map_solo = np.asarray(agents_mod.update_local_maps(
        jnp.asarray(local_map_solo),
        state.agent_state.local_scan,
        state.agent_state.positions,
        view_r,
        survey_r,
    ))

    agent_attr_acc = np.zeros((n_blue, len(BLOCKS)), dtype=np.float64)
    team_attr_acc  = np.zeros((n_blue, len(BLOCKS)), dtype=np.float64)
    n_steps = 0

    max_steps = cfg.env.max_steps
    for step in range(1, max_steps + 1):
        key, *subkeys = jax.random.split(key, n_total + 2)

        # ---- compute saliency for every blue on its merged obs ----
        action_dict = {}
        for bi, gi in enumerate(blue_idx):
            obs_i = obs_dict[env.agents[gi]]
            a_i = int(_blue_sample(blue_params, obs_i, subkeys[gi]))
            action_dict[env.agents[gi]] = a_i
            grad = np.asarray(_blue_logit_grad(blue_params, obs_i, a_i))   # (23,)
            obs_np = np.asarray(obs_i)

            # Merged seen/map_frac (already in obs); solo counterparts:
            pos_np = np.asarray(state.agent_state.positions[gi])
            solo_seen   = _solo_seen_window(local_map_solo[gi], pos_np, view_r)
            solo_known  = (local_map_solo[gi] != MAP_UNKNOWN).astype(np.float64)
            solo_frac   = float(solo_known.sum() / (H * W))
            merged_frac = float(obs_np[18])
            merged_seen = obs_np[9:18].astype(np.float64)

            for bi_blk, (name, lo, hi) in enumerate(BLOCKS):
                g_blk = np.abs(grad[lo:hi]).astype(np.float64)
                if name not in COMM_BLOCKS:
                    agent_attr_acc[bi, bi_blk] += float(g_blk.mean())
                    continue
                if name == "seen":
                    agent_frac = np.where(merged_seen > 0,
                                          np.clip(solo_seen / np.maximum(merged_seen, 1e-12), 0.0, 1.0),
                                          0.0)
                else:  # map_frac
                    a = (solo_frac / merged_frac) if merged_frac > 1e-12 else 0.0
                    agent_frac = np.array([np.clip(a, 0.0, 1.0)])
                team_frac = 1.0 - agent_frac
                agent_attr_acc[bi, bi_blk] += float((g_blk * agent_frac).mean())
                team_attr_acc[bi,  bi_blk] += float((g_blk * team_frac).mean())

        # ---- drive red with the blue policy (same setup as headline rollout) ----
        for r in range(n_red):
            gi = n_blue + r
            obs_i = obs_dict[env.agents[gi]]
            a_i = int(_blue_sample(blue_params, obs_i, subkeys[gi]))
            action_dict[env.agents[gi]] = a_i

        # ---- step env ----
        obs_dict, state, _r, dones, _info = env.step_env(key, state, action_dict)
        # Update solo local_maps with the new scans (no comm).
        local_map_solo = np.asarray(agents_mod.update_local_maps(
            jnp.asarray(local_map_solo),
            state.agent_state.local_scan,
            state.agent_state.positions,
            view_r,
            survey_r,
        ))
        n_steps += 1
        if bool(dones["__all__"]):
            break

    agent_per_block = (agent_attr_acc / n_steps).mean(axis=0)
    team_per_block  = (team_attr_acc  / n_steps).mean(axis=0)
    team_share = float(team_per_block.sum() / max(1e-12, agent_per_block.sum() + team_per_block.sum()))

    out = {
        "blocks": [b for b, *_ in BLOCKS],
        "agent_mean_per_block": {b: float(v) for b, (b_, v) in zip([n for n, *_ in BLOCKS], zip([n for n, *_ in BLOCKS], agent_per_block))},
        "team_mean_per_block":  {b: float(v) for b, (b_, v) in zip([n for n, *_ in BLOCKS], zip([n for n, *_ in BLOCKS], team_per_block))},
        "team_share": team_share,
        "n_steps": n_steps,
        "n_blue": int(n_blue),
        "setup": "C2",
        "method": "counterfactual no-comm split (agent_frac = solo / merged per dim)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"agent: {dict(zip([n for n,*_ in BLOCKS], np.round(agent_per_block, 4)))}")
    print(f"team:  {dict(zip([n for n,*_ in BLOCKS], np.round(team_per_block, 4)))}")
    print(f"team-share = {team_share:.3f}")


if __name__ == "__main__":
    main()
