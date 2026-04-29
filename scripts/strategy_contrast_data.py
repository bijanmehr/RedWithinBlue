"""Behavioural rollout at C2 for strategy-contrast figure.

Captures per-step:
  • all-agent positions
  • per-blue cells-known (solo, merged) — to quantify the comm-graph dividend
  • per-team intra-team mean pairwise distance (computed at render time)

Drives blue with the trained blue policy and red with the trained joint-red
policy (matching the canonical seed-0 episode used elsewhere).

Output: experiments/meta-report/strategy_contrast.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from meta_report import SETUPS, _load_blue, _load_red
from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.rewards_training import (
    normalized_competitive_reward,
    normalized_exploration_reward,
)
from red_within_blue import agents as agents_mod
from red_within_blue.types import MAP_UNKNOWN

SEED = 0


def run_setup(setup_key: str) -> Path:
    out = ROOT / "experiments" / "meta-report" / f"strategy_contrast_{setup_key}.npz"
    setup = next(s for s in SETUPS if s.key == setup_key)
    cfg = ExperimentConfig.from_yaml(setup.config)
    n_total = cfg.env.num_agents
    n_red = cfg.env.num_red_agents
    n_blue = n_total - n_red
    H, W = cfg.env.grid_height, cfg.env.grid_width

    reward_fn = normalized_competitive_reward if n_red > 0 else normalized_exploration_reward
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)
    view_r = env.view_radius
    survey_r = env.survey_radius

    blue_actor, blue_params = _load_blue(cfg, setup.blue_ckpt)
    if n_red > 0:
        red_actor, red_params = _load_red(cfg, setup.red_ckpt)
    else:
        red_actor, red_params = None, None

    @jax.jit
    def _blue_sample(bp, obs, key):
        return jax.random.categorical(key, blue_actor.apply(bp, obs))

    if n_red > 0:
        @jax.jit
        def _red_sample(rp, obs_flat, key):
            logits = red_actor.apply(rp, obs_flat)
            keys = jax.random.split(key, n_red)
            return jax.vmap(jax.random.categorical)(keys, logits)

    key = jax.random.PRNGKey(SEED)
    obs_dict, state = env.reset(key)
    team_ids = np.asarray(state.agent_state.team_ids)
    blue_idx = np.where(team_ids == 0)[0]

    local_map_solo = np.full((n_total, H, W), MAP_UNKNOWN, dtype=np.int32)
    local_map_solo = np.asarray(agents_mod.update_local_maps(
        jnp.asarray(local_map_solo),
        state.agent_state.local_scan,
        state.agent_state.positions,
        view_r, survey_r,
    ))

    positions_log = [np.asarray(state.agent_state.positions).copy()]
    solo_known   = []   # [T, n_blue]
    merged_known = []   # [T, n_blue]

    max_steps = cfg.env.max_steps
    for step in range(1, max_steps + 1):
        key, *subkeys = jax.random.split(key, n_total + 2)

        action_dict = {}
        # Blue
        for bi, gi in enumerate(blue_idx):
            obs_i = obs_dict[env.agents[gi]]
            a_i = int(_blue_sample(blue_params, obs_i, subkeys[gi]))
            action_dict[env.agents[gi]] = a_i
        # Red (joint policy) — only if there is any red.
        if n_red > 0:
            red_obs_flat = jnp.concatenate(
                [obs_dict[env.agents[n_blue + r]] for r in range(n_red)]
            )
            red_actions = _red_sample(red_params, red_obs_flat, subkeys[n_blue])
            red_actions_np = np.asarray(red_actions).astype(np.int32)
            for r in range(n_red):
                action_dict[env.agents[n_blue + r]] = int(red_actions_np[r])

        # Knowledge counts BEFORE stepping (matches obs the agent acted on).
        merged_local_maps = np.asarray(state.agent_state.local_map)  # [N, H, W]
        merged_known.append(np.array([
            int((merged_local_maps[gi] != MAP_UNKNOWN).sum()) for gi in blue_idx
        ], dtype=np.int32))
        solo_known.append(np.array([
            int((local_map_solo[gi] != MAP_UNKNOWN).sum()) for gi in blue_idx
        ], dtype=np.int32))

        # Step env
        obs_dict, state, _r, dones, _info = env.step_env(key, state, action_dict)
        positions_log.append(np.asarray(state.agent_state.positions).copy())
        local_map_solo = np.asarray(agents_mod.update_local_maps(
            jnp.asarray(local_map_solo),
            state.agent_state.local_scan,
            state.agent_state.positions,
            view_r, survey_r,
        ))
        if bool(dones["__all__"]):
            break

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        positions=np.stack(positions_log, axis=0),       # (T+1, N, 2)
        team_ids=team_ids,
        solo_known=np.stack(solo_known, axis=0),         # (T, n_blue)
        merged_known=np.stack(merged_known, axis=0),     # (T, n_blue)
        grid_h=np.int32(H),
        grid_w=np.int32(W),
        n_blue=np.int32(n_blue),
        n_red=np.int32(n_red),
    )
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    import sys as _sys
    keys = _sys.argv[1:] or ["B", "C1", "C2"]
    for k in keys:
        run_setup(k)
