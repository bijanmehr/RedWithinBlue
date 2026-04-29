"""Twin-Q + target-net ablation for the central (blue) critic.

Compartmentalised so nothing in ``src/red_within_blue/`` changes. We reuse
only the env + networks + config + the pure ``compute_discounted_returns``
helper; every loss function used by the four variants lives in this file.

The existing CTDE critic already uses Monte-Carlo returns (not TD(0)), so
the question this experiment answers is narrower than the original plan:

    "If we had kept TD(0) bootstrapping and fixed the critic drift with
     target-net / twin-Q / both, would that have beaten the MC switch?"

Four variants, all at the same ``pair-cooperate-coop`` config:

    A0 — current baseline: single critic, Monte-Carlo returns (no bootstrap).
    A1 — single critic, TD(0) bootstrap against a Polyak-averaged target.
    A2 — twin critics, TD(0) against live min(V1, V2).
    A3 — twin critics, TD(0) against Polyak-averaged min(V1_tgt, V2_tgt).
         (This is SAC's Q-target applied to the CTDE V critic.)

Run:
    # quick smoke (150 eps, 2 seeds, ~90 s) — sanity check both paths JIT.
    python scripts/stabilization/twin_critic_experiment.py --variant A0 --smoke
    python scripts/stabilization/twin_critic_experiment.py --variant A3 --smoke

    # full (15000 eps, 5 seeds, ~20 min per variant).
    python scripts/stabilization/twin_critic_experiment.py --variant A0
    python scripts/stabilization/twin_critic_experiment.py --variant A1
    python scripts/stabilization/twin_critic_experiment.py --variant A2
    python scripts/stabilization/twin_critic_experiment.py --variant A3

Outputs (per variant):
    experiments/stabilization/twin-critic-<V>/metrics.npz
    experiments/stabilization/twin-critic-<V>/summary.json

Metric schema matches the standard trainer (``loss``, ``total_reward``,
``blue_total_reward``, ``red_total_reward``, ``per_agent_reward``) so
downstream tools see it as a normal training run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from red_within_blue.env import GridCommEnv
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.losses import compute_discounted_returns
from red_within_blue.training.networks import Actor, Critic
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import (
    MultiTrajectory,
    collect_episode_multi_scan,
)


VARIANTS = ("A0", "A1", "A2", "A3")
REF_CONFIG = "configs/pair-cooperate-coop.yaml"
TAU = 0.005   # Polyak soft-update coefficient (SAC default).


# ---------------------------------------------------------------------------
# Training-state pytree
# ---------------------------------------------------------------------------


class TrainState(NamedTuple):
    """Everything carried across train-steps for ONE seed.

    For single-critic variants (A0, A1), ``critic2_params`` / ``target2_params``
    are ignored (initialised but never read).  For no-target variants
    (A0, A2), ``target1_params`` / ``target2_params`` are ignored.
    """
    actor_params: any
    actor_opt_state: any
    critic1_params: any
    critic1_opt_state: any
    critic2_params: any
    critic2_opt_state: any
    target1_params: any
    target2_params: any


# ---------------------------------------------------------------------------
# Loss functions — one per variant, each fully local to this file.
# ---------------------------------------------------------------------------


def _policy_logprobs(actor, actor_params, observations, actions):
    """Compute per-step log pi(a_t^i | o_t^i) and entropy.

    ``observations`` is ``[T, N, obs_dim]``, ``actions`` is ``[T, N]``.
    """
    batch_actor = jax.vmap(
        jax.vmap(actor.apply, in_axes=(None, 0)),
        in_axes=(None, 1), out_axes=1,
    )
    logits = batch_actor(actor_params, observations)                 # [T, N, A]
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)
    log_probs = jnp.take_along_axis(
        log_probs_all, actions[..., None], axis=-1,
    ).squeeze(-1)                                                    # [T, N]
    probs = jnp.exp(log_probs_all)
    entropy = -jnp.mean(jnp.sum(probs * log_probs_all, axis=-1))
    return log_probs, entropy


def _central_values(critic, critic_params, observations):
    """Central critic on joint obs: [T, N, obs_dim] -> [T]."""
    T = observations.shape[0]
    joint = observations.reshape(T, -1)
    return jax.vmap(lambda o: critic.apply(critic_params, o))(joint)


def _joint_next_obs(observations):
    """Build the next-step joint-obs array [T, N*obs_dim].

    Shift observations by one step and pad with zeros for the terminal step
    (the last step's ``next_value`` is zeroed out by the done mask anyway).
    """
    T, N, D = observations.shape
    joint = observations.reshape(T, N * D)
    return jnp.concatenate([joint[1:], jnp.zeros((1, N * D))], axis=0)  # [T, N*D]


def _loss_A0_mc(actor, critic, actor_params, critic1_params, traj, gamma):
    """A0 — single critic, Monte-Carlo returns (current baseline)."""
    obs = traj.obs                                                   # [T, N, D]
    actions = traj.actions                                           # [T, N]
    rewards = traj.rewards                                           # [T, N]
    dones = traj.dones                                               # [T]

    log_probs, entropy = _policy_logprobs(actor, actor_params, obs, actions)
    values = _central_values(critic, critic1_params, obs)            # [T]

    team_rew = jnp.sum(rewards, axis=-1)
    mc = jax.lax.stop_gradient(compute_discounted_returns(team_rew, dones, gamma))
    advantages = mc - values

    policy_loss = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages)[:, None])
    value_loss = jnp.mean((values - mc) ** 2)
    return policy_loss, value_loss, entropy


def _loss_A1_td_target(actor, critic, actor_params, critic1_params,
                        target1_params, traj, gamma):
    """A1 — single critic, TD(0) against Polyak-averaged target."""
    obs = traj.obs
    actions = traj.actions
    rewards = traj.rewards
    dones = traj.dones

    log_probs, entropy = _policy_logprobs(actor, actor_params, obs, actions)
    values = _central_values(critic, critic1_params, obs)            # [T]
    next_joint = _joint_next_obs(obs)
    next_values_tgt = jax.vmap(
        lambda o: critic.apply(target1_params, o)
    )(next_joint)                                                    # [T]

    team_rew = jnp.sum(rewards, axis=-1)
    d = dones.astype(jnp.float32)
    td_target = team_rew + gamma * next_values_tgt * (1.0 - d)       # [T]
    td_target = jax.lax.stop_gradient(td_target)
    advantages = td_target - values

    policy_loss = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages)[:, None])
    value_loss = jnp.mean((values - td_target) ** 2)
    return policy_loss, value_loss, entropy


def _loss_A2_twin_live(actor, critic, actor_params, critic1_params,
                        critic2_params, traj, gamma):
    """A2 — twin critics, TD(0) against live min(V1, V2)."""
    obs = traj.obs
    actions = traj.actions
    rewards = traj.rewards
    dones = traj.dones

    log_probs, entropy = _policy_logprobs(actor, actor_params, obs, actions)
    v1 = _central_values(critic, critic1_params, obs)                # [T]
    v2 = _central_values(critic, critic2_params, obs)                # [T]

    next_joint = _joint_next_obs(obs)
    next_v1 = jax.vmap(lambda o: critic.apply(critic1_params, o))(next_joint)
    next_v2 = jax.vmap(lambda o: critic.apply(critic2_params, o))(next_joint)
    next_min = jnp.minimum(next_v1, next_v2)                         # [T]

    team_rew = jnp.sum(rewards, axis=-1)
    d = dones.astype(jnp.float32)
    td_target = jax.lax.stop_gradient(team_rew + gamma * next_min * (1.0 - d))

    advantages = td_target - v1                                      # policy uses v1
    policy_loss = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages)[:, None])
    value1_loss = jnp.mean((v1 - td_target) ** 2)
    value2_loss = jnp.mean((v2 - td_target) ** 2)
    return policy_loss, value1_loss, value2_loss, entropy


def _loss_A3_twin_target(actor, critic, actor_params, critic1_params,
                          critic2_params, target1_params, target2_params,
                          traj, gamma):
    """A3 — twin critics, TD(0) against Polyak-averaged min(V1_tgt, V2_tgt)."""
    obs = traj.obs
    actions = traj.actions
    rewards = traj.rewards
    dones = traj.dones

    log_probs, entropy = _policy_logprobs(actor, actor_params, obs, actions)
    v1 = _central_values(critic, critic1_params, obs)
    v2 = _central_values(critic, critic2_params, obs)

    next_joint = _joint_next_obs(obs)
    next_v1 = jax.vmap(lambda o: critic.apply(target1_params, o))(next_joint)
    next_v2 = jax.vmap(lambda o: critic.apply(target2_params, o))(next_joint)
    next_min = jnp.minimum(next_v1, next_v2)

    team_rew = jnp.sum(rewards, axis=-1)
    d = dones.astype(jnp.float32)
    td_target = jax.lax.stop_gradient(team_rew + gamma * next_min * (1.0 - d))

    advantages = td_target - v1
    policy_loss = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages)[:, None])
    value1_loss = jnp.mean((v1 - td_target) ** 2)
    value2_loss = jnp.mean((v2 - td_target) ** 2)
    return policy_loss, value1_loss, value2_loss, entropy


# ---------------------------------------------------------------------------
# Per-variant build + train-step factory
# ---------------------------------------------------------------------------


def _polyak(tgt, src, tau):
    """Polyak soft-update: tgt <- tau*src + (1-tau)*tgt, per-leaf."""
    return jax.tree_util.tree_map(
        lambda t, s: tau * s + (1.0 - tau) * t, tgt, src
    )


def _build_train_step(variant, actor, critic, env, optimizer_actor,
                      optimizer_critic, cfg):
    """Return a JITted per-episode train_step(state, key) -> (state, metrics).

    Each variant gets its own train_step so the graph is specialised and JIT
    doesn't branch on Python-side variant flags.
    """
    max_steps = cfg.env.max_steps
    gamma = cfg.train.gamma
    ent_coef = cfg.train.ent_coef
    vf_coef = 0.5

    def rollout(actor_params, key):
        return collect_episode_multi_scan(
            env, actor, actor_params, key, max_steps,
            enforce_connectivity=cfg.enforce_connectivity,
            red_policy="shared", num_red_agents=0,
            epsilon=cfg.train.epsilon,
        )

    if variant == "A0":
        def loss_fn(actor_params, critic1_params, traj):
            pl, vl1, ent = _loss_A0_mc(
                actor, critic, actor_params, critic1_params, traj, gamma,
            )
            return pl + vf_coef * vl1 - ent_coef * ent, (pl, vl1, 0.0, ent)

        def train_step(state, key):
            traj = rollout(state.actor_params, key)
            (total, (pl, vl1, vl2, ent)), grads = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True,
            )(state.actor_params, state.critic1_params, traj)
            actor_upd, new_actor_opt = optimizer_actor.update(
                grads[0], state.actor_opt_state, state.actor_params,
            )
            critic_upd, new_c1_opt = optimizer_critic.update(
                grads[1], state.critic1_opt_state, state.critic1_params,
            )
            new_actor = optax.apply_updates(state.actor_params, actor_upd)
            new_c1 = optax.apply_updates(state.critic1_params, critic_upd)
            total_reward = jnp.sum(traj.rewards)
            per_agent = jnp.sum(traj.rewards, axis=0)                # [N]
            new_state = state._replace(
                actor_params=new_actor,
                actor_opt_state=new_actor_opt,
                critic1_params=new_c1,
                critic1_opt_state=new_c1_opt,
            )
            return new_state, (total, pl, vl1, vl2, ent, total_reward, per_agent)

    elif variant == "A1":
        def loss_fn(actor_params, critic1_params, target1_params, traj):
            pl, vl1, ent = _loss_A1_td_target(
                actor, critic, actor_params, critic1_params,
                target1_params, traj, gamma,
            )
            return pl + vf_coef * vl1 - ent_coef * ent, (pl, vl1, 0.0, ent)

        def train_step(state, key):
            traj = rollout(state.actor_params, key)
            (total, (pl, vl1, vl2, ent)), grads = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True,
            )(state.actor_params, state.critic1_params,
              state.target1_params, traj)
            actor_upd, new_actor_opt = optimizer_actor.update(
                grads[0], state.actor_opt_state, state.actor_params,
            )
            critic_upd, new_c1_opt = optimizer_critic.update(
                grads[1], state.critic1_opt_state, state.critic1_params,
            )
            new_actor = optax.apply_updates(state.actor_params, actor_upd)
            new_c1 = optax.apply_updates(state.critic1_params, critic_upd)
            new_t1 = _polyak(state.target1_params, new_c1, TAU)
            total_reward = jnp.sum(traj.rewards)
            per_agent = jnp.sum(traj.rewards, axis=0)
            new_state = state._replace(
                actor_params=new_actor,
                actor_opt_state=new_actor_opt,
                critic1_params=new_c1,
                critic1_opt_state=new_c1_opt,
                target1_params=new_t1,
            )
            return new_state, (total, pl, vl1, vl2, ent, total_reward, per_agent)

    elif variant == "A2":
        def loss_fn(actor_params, critic1_params, critic2_params, traj):
            pl, vl1, vl2, ent = _loss_A2_twin_live(
                actor, critic, actor_params, critic1_params,
                critic2_params, traj, gamma,
            )
            return pl + vf_coef * (vl1 + vl2) - ent_coef * ent, (pl, vl1, vl2, ent)

        def train_step(state, key):
            traj = rollout(state.actor_params, key)
            (total, (pl, vl1, vl2, ent)), grads = jax.value_and_grad(
                loss_fn, argnums=(0, 1, 2), has_aux=True,
            )(state.actor_params, state.critic1_params,
              state.critic2_params, traj)
            actor_upd, new_actor_opt = optimizer_actor.update(
                grads[0], state.actor_opt_state, state.actor_params,
            )
            c1_upd, new_c1_opt = optimizer_critic.update(
                grads[1], state.critic1_opt_state, state.critic1_params,
            )
            c2_upd, new_c2_opt = optimizer_critic.update(
                grads[2], state.critic2_opt_state, state.critic2_params,
            )
            new_actor = optax.apply_updates(state.actor_params, actor_upd)
            new_c1 = optax.apply_updates(state.critic1_params, c1_upd)
            new_c2 = optax.apply_updates(state.critic2_params, c2_upd)
            total_reward = jnp.sum(traj.rewards)
            per_agent = jnp.sum(traj.rewards, axis=0)
            new_state = state._replace(
                actor_params=new_actor,
                actor_opt_state=new_actor_opt,
                critic1_params=new_c1,
                critic1_opt_state=new_c1_opt,
                critic2_params=new_c2,
                critic2_opt_state=new_c2_opt,
            )
            return new_state, (total, pl, vl1, vl2, ent, total_reward, per_agent)

    elif variant == "A3":
        def loss_fn(actor_params, critic1_params, critic2_params,
                    target1_params, target2_params, traj):
            pl, vl1, vl2, ent = _loss_A3_twin_target(
                actor, critic, actor_params, critic1_params, critic2_params,
                target1_params, target2_params, traj, gamma,
            )
            return pl + vf_coef * (vl1 + vl2) - ent_coef * ent, (pl, vl1, vl2, ent)

        def train_step(state, key):
            traj = rollout(state.actor_params, key)
            (total, (pl, vl1, vl2, ent)), grads = jax.value_and_grad(
                loss_fn, argnums=(0, 1, 2), has_aux=True,
            )(state.actor_params, state.critic1_params, state.critic2_params,
              state.target1_params, state.target2_params, traj)
            actor_upd, new_actor_opt = optimizer_actor.update(
                grads[0], state.actor_opt_state, state.actor_params,
            )
            c1_upd, new_c1_opt = optimizer_critic.update(
                grads[1], state.critic1_opt_state, state.critic1_params,
            )
            c2_upd, new_c2_opt = optimizer_critic.update(
                grads[2], state.critic2_opt_state, state.critic2_params,
            )
            new_actor = optax.apply_updates(state.actor_params, actor_upd)
            new_c1 = optax.apply_updates(state.critic1_params, c1_upd)
            new_c2 = optax.apply_updates(state.critic2_params, c2_upd)
            new_t1 = _polyak(state.target1_params, new_c1, TAU)
            new_t2 = _polyak(state.target2_params, new_c2, TAU)
            total_reward = jnp.sum(traj.rewards)
            per_agent = jnp.sum(traj.rewards, axis=0)
            new_state = state._replace(
                actor_params=new_actor,
                actor_opt_state=new_actor_opt,
                critic1_params=new_c1,
                critic1_opt_state=new_c1_opt,
                critic2_params=new_c2,
                critic2_opt_state=new_c2_opt,
                target1_params=new_t1,
                target2_params=new_t2,
            )
            return new_state, (total, pl, vl1, vl2, ent, total_reward, per_agent)

    else:
        raise ValueError(f"unknown variant {variant!r}")

    return jax.jit(train_step)


# ---------------------------------------------------------------------------
# Initial state factory (vmappable)
# ---------------------------------------------------------------------------


def _init_state(actor, critic, optimizer_actor, optimizer_critic,
                obs_dim, n_agents, key):
    """Build one seed's TrainState.  Shape-stable across variants."""
    k_a, k_c1, k_c2 = jax.random.split(key, 3)
    dummy_obs = jnp.zeros(obs_dim)
    dummy_joint = jnp.zeros(n_agents * obs_dim)
    actor_params = actor.init(k_a, dummy_obs)
    c1_params = critic.init(k_c1, dummy_joint)
    c2_params = critic.init(k_c2, dummy_joint)
    return TrainState(
        actor_params=actor_params,
        actor_opt_state=optimizer_actor.init(actor_params),
        critic1_params=c1_params,
        critic1_opt_state=optimizer_critic.init(c1_params),
        critic2_params=c2_params,
        critic2_opt_state=optimizer_critic.init(c2_params),
        target1_params=c1_params,
        target2_params=c2_params,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run(variant, num_episodes, num_seeds, output_dir, cfg_path=REF_CONFIG):
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    # Force consistent sizes across variants (deterministic reference run).
    n = cfg.env.num_agents
    obs_dim = cfg.obs_dim

    reward_fn = make_multi_agent_reward(
        disconnect_penalty=cfg.reward.disconnect_penalty,
        isolation_weight=cfg.reward.isolation_weight,
        cooperative_weight=cfg.reward.cooperative_weight,
        revisit_weight=cfg.reward.revisit_weight,
        terminal_bonus_scale=cfg.reward.terminal_bonus_scale,
        terminal_bonus_divide=cfg.reward.terminal_bonus_divide,
        spread_weight=cfg.reward.spread_weight,
        fog_potential_weight=cfg.reward.fog_potential_weight,
        num_red_agents=cfg.env.num_red_agents,
    )
    env = GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)

    actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
    )
    critic = Critic(
        hidden_dim=cfg.network.critic_hidden_dim,
        num_layers=cfg.network.critic_num_layers,
    )

    lr = cfg.train.lr
    clip = cfg.train.grad_clip
    make_opt = lambda: optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    optimizer_actor = make_opt()
    optimizer_critic = make_opt()

    train_step = _build_train_step(
        variant, actor, critic, env, optimizer_actor, optimizer_critic, cfg,
    )

    # Build per-seed initial states, vmap over seed.
    init_keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(num_seeds))
    init_fn = lambda k: _init_state(
        actor, critic, optimizer_actor, optimizer_critic,
        obs_dim, n, k,
    )
    states = jax.vmap(init_fn)(init_keys)

    # vmap the JITted train_step over the 5 seeds.
    vstep = jax.vmap(train_step, in_axes=(0, 0))

    # Pre-split PRNG keys for every (seed, episode) pair to keep the inner
    # loop allocation-free.
    root_key = jax.random.PRNGKey(0)
    ep_keys = jax.random.split(root_key, num_episodes * num_seeds).reshape(
        num_episodes, num_seeds, -1,
    )

    loss_hist = np.zeros((num_episodes, num_seeds), dtype=np.float32)
    pl_hist = np.zeros_like(loss_hist)
    vl1_hist = np.zeros_like(loss_hist)
    vl2_hist = np.zeros_like(loss_hist)
    ent_hist = np.zeros_like(loss_hist)
    reward_hist = np.zeros_like(loss_hist)
    per_agent_hist = np.zeros((num_episodes, num_seeds, n), dtype=np.float32)

    print(f"variant={variant}  episodes={num_episodes}  seeds={num_seeds}  "
          f"config={cfg_path}")
    t0 = time.time()
    log_every = max(1, num_episodes // 15)
    for ep in range(num_episodes):
        states, (total, pl, vl1, vl2, ent, tot_r, per_a) = vstep(
            states, ep_keys[ep],
        )
        loss_hist[ep] = np.asarray(total)
        pl_hist[ep] = np.asarray(pl)
        vl1_hist[ep] = np.asarray(vl1)
        vl2_hist[ep] = np.asarray(vl2)
        ent_hist[ep] = np.asarray(ent)
        reward_hist[ep] = np.asarray(tot_r)
        per_agent_hist[ep] = np.asarray(per_a)
        if ep % log_every == 0 or ep == num_episodes - 1:
            elapsed = time.time() - t0
            r = reward_hist[max(0, ep - 50): ep + 1].mean()
            print(f"  ep {ep:5d}/{num_episodes}  "
                  f"reward(last 50) = {r:+.3f}  "
                  f"|loss| p50 = {np.median(np.abs(loss_hist[ep])):.3f}  "
                  f"elapsed = {elapsed:5.1f}s")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "metrics.npz",
        loss=loss_hist,
        policy_loss=pl_hist,
        value1_loss=vl1_hist,
        value2_loss=vl2_hist,
        entropy=ent_hist,
        total_reward=reward_hist,
        blue_total_reward=reward_hist,   # mirrors standard schema (no red here)
        red_total_reward=np.zeros_like(reward_hist),
        per_agent_reward=per_agent_hist,
    )

    # Headline summary for A/B diffs later.
    last = reward_hist[-min(500, num_episodes):].mean(axis=0)   # per-seed
    mid = reward_hist[num_episodes // 2: num_episodes // 2 + 500].mean(axis=0)
    late_dive = int(np.sum(last < mid))
    summary = {
        "variant": variant,
        "config": cfg_path,
        "num_episodes": num_episodes,
        "num_seeds": num_seeds,
        "final_reward_mean": float(last.mean()),
        "final_reward_std": float(last.std(ddof=1)) if num_seeds > 1 else 0.0,
        "final_reward_per_seed": [float(x) for x in last],
        "late_dive_count": late_dive,
        "loss_abs_p99_final": float(
            np.percentile(np.abs(loss_hist[-500:]), 99)
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ndone: final_reward = {last.mean():+.3f} ± "
          f"{last.std(ddof=1) if num_seeds > 1 else 0.0:.3f}    "
          f"late_dives = {late_dive}/{num_seeds}    "
          f"wall = {time.time() - t0:.1f}s")
    return summary


def _parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--episodes", type=int, default=15000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--config", type=str, default=REF_CONFIG)
    ap.add_argument("--smoke", action="store_true",
                    help="Short run for JIT sanity: 150 eps, 2 seeds.")
    return ap.parse_args()


def main():
    args = _parse()
    if args.smoke:
        args.episodes = 150
        args.seeds = 2
    if args.output_dir is None:
        args.output_dir = f"experiments/stabilization/twin-critic-{args.variant}"
        if args.smoke:
            args.output_dir += "-smoke"
    run(args.variant, args.episodes, args.seeds, args.output_dir, args.config)


if __name__ == "__main__":
    main()
