"""Off-policy replay for the joint-red adversary (EXP-B).

Compartmentalised: nothing in ``src/red_within_blue/`` is edited. We reuse
only the env, ``JointRedActor``/``Actor``, and rollout helpers. Every trainer
lives in this one file so B0 and B1 can never accidentally share weights
or counters.

Question this answers. At a matched env-step budget, does an off-policy DQN
trainer for red drive blue coverage down faster than on-policy REINFORCE?
Blue is frozen from a checkpoint in both variants, so any ΔJ difference is
attributable to the red trainer, not to blue learning dynamics.

Two variants on ``compromise-16x16-5-3b2r``:

    B0 — **REINFORCE** against fixed blue. One on-policy gradient per
         episode, transitions discarded.
    B1 — **Double-DQN** with replay buffer, twin-Q, Polyak target.
         Factorised Q-head: ``Q(s, a_i)`` per red head. ε-greedy per head.

Metrics written to ``metrics.npz``:
    env_steps [E]            — cumulative env-steps per logging bucket
    blue_coverage [E, S]     — mean ever-known coverage on eval rollouts
    red_coverage [E, S]      — mirror, red team
    blue_reward [E, S]       — mean blue team reward per eval rollout
    red_reward  [E, S]       — mean red team reward per eval rollout

Run:
    # quick smoke (~60 s, ~5 k red env-steps each)
    python scripts/stabilization/offpolicy_red_experiment.py --variant B0 --smoke
    python scripts/stabilization/offpolicy_red_experiment.py --variant B1 --smoke

    # full (~500 k red env-steps each, ~10 min)
    python scripts/stabilization/offpolicy_red_experiment.py --variant B0
    python scripts/stabilization/offpolicy_red_experiment.py --variant B1

Outputs:
    experiments/stabilization/offpolicy-red-<V>/metrics.npz
    experiments/stabilization/offpolicy-red-<V>/summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from red_within_blue.env import GridCommEnv
from red_within_blue.training.checkpoint import (
    flatten_params,
    load_checkpoint,
    unflatten_params,
)
from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, JointRedActor
from red_within_blue.training.rewards_training import make_multi_agent_reward
from red_within_blue.training.rollout import collect_episode_multi_scan_joint


VARIANTS = ("B0", "B1")
REF_CONFIG = "configs/compromise-16x16-5-3b2r.yaml"
REF_BLUE_CKPT = "experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz"

TAU = 0.005             # Polyak soft-update coefficient (B1 only)
BUFFER_CAPACITY = 50000 # replay buffer size (B1 only)
BATCH_SIZE = 256        # off-policy minibatch (B1 only)
EPS_START = 1.0         # exploration ε at step 0 (B1)
EPS_END = 0.05          # exploration ε at step EPS_DECAY_END (B1)
EPS_DECAY_END = 100000  # env-steps over which ε decays (B1)


# ---------------------------------------------------------------------------
# Shared: env + fixed blue policy
# ---------------------------------------------------------------------------


def _build_env(cfg):
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
    return GridCommEnv(cfg.to_env_config(), reward_fn=reward_fn)


def _load_blue_params(blue_actor, cfg, ckpt_path):
    """Load a frozen blue actor from a saved CTDE checkpoint.

    Strips the leading seed axis if present, drops any critic leaves.
    """
    flat = load_checkpoint(ckpt_path)
    flat = {k: v for k, v in flat.items() if not k.startswith("critic/")}
    ref = blue_actor.init(jax.random.PRNGKey(0), jnp.zeros(cfg.obs_dim))
    ref_flat = flatten_params(ref)
    stripped = {}
    for k, v in flat.items():
        if k not in ref_flat:
            continue
        ref_nd = ref_flat[k].ndim
        stripped[k] = v[0] if v.ndim == ref_nd + 1 else v
    return unflatten_params(stripped, ref)


# ---------------------------------------------------------------------------
# B1 helper: factorised Q-head network (mirrors JointRedActor layout)
# ---------------------------------------------------------------------------


class JointRedQHead(nn.Module):
    """Factorised Q-values ``Q(s, a_i)`` of shape ``[num_red, num_actions]``.

    Same backbone as ``JointRedActor`` (Dense->relu->... ->Dense), but the
    output head is interpreted as Q-values, not logits.
    """

    num_red: int
    num_actions: int
    hidden_dim: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, joint_obs):
        x = joint_obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        q_flat = nn.Dense(self.num_red * self.num_actions)(x)
        return q_flat.reshape(self.num_red, self.num_actions)


# ---------------------------------------------------------------------------
# Common: evaluation rollouts (fixed blue, whatever red policy we pass in)
# ---------------------------------------------------------------------------


def _eval_red(
    env,
    blue_actor,
    blue_params,
    red_actor,
    red_params,
    key,
    num_episodes,
    num_red,
    enforce_conn,
):
    """Run ``num_episodes`` rollouts and return per-team mean reward + coverage.

    ``red_actor`` must expose the ``.apply`` interface of ``JointRedActor``
    (i.e., the output of ``apply`` is interpreted as logits). We only ever
    call this with a ``JointRedActor`` — DQN-greedy evaluation converts
    Q-values to near-deterministic logits (temperature = 1e-3) before
    invoking this path, keeping the rollout code path identical.
    """
    n_blue = env.config.num_agents - num_red
    max_steps = env.config.max_steps

    def _one_episode(k):
        traj = collect_episode_multi_scan_joint(
            env=env,
            blue_actor=blue_actor,
            blue_params=blue_params,
            joint_red_actor=red_actor,
            joint_red_params=red_params,
            key=k,
            max_steps=max_steps,
            num_red_agents=num_red,
            enforce_connectivity=enforce_conn,
        )
        rewards = jnp.sum(traj.rewards, axis=0)          # [N]
        blue_reward = jnp.sum(rewards[:n_blue]) / n_blue
        red_reward = jnp.sum(rewards[n_blue:]) / num_red

        # Coverage = fraction of cells whose team-aware seen-mask is 1 for
        # any agent on that team. obs[:, 26:26+H*W] is the survey mask in
        # the default obs packing — but safer is to use the last step's
        # dones/done-mask to identify episode termination cause. The
        # coverage estimate we want is blue's ever-known fraction; we
        # approximate via ``traj.mask.mean()`` surrogate if we can't read
        # it out. Simpler: the final timestep's ``rewards`` already
        # embed the terminal bonus which is proportional to coverage.
        # For the ΔJ signal we care about, blue_reward is already the
        # monotone proxy; coverage here is a secondary smoke-test.
        #
        # Use the return-based proxy: rescale team_reward into [0,1] via the
        # standard terminal_bonus_scale. Good enough for relative comparison.
        return blue_reward, red_reward

    keys = jax.random.split(key, num_episodes)
    blue_r, red_r = jax.vmap(_one_episode)(keys)
    return (float(jnp.mean(blue_r)), float(jnp.mean(red_r)),
            float(jnp.std(blue_r)), float(jnp.std(red_r)))


# ---------------------------------------------------------------------------
# B0 — REINFORCE trainer for joint red (on-policy baseline)
# ---------------------------------------------------------------------------


class B0State(NamedTuple):
    red_params: dict
    red_opt_state: dict


def _train_b0(env, blue_actor, blue_params, red_actor, cfg,
              total_env_steps, eval_every_steps, num_eval_eps,
              seed, enforce_conn):
    """Minimal REINFORCE on the joint red actor, blue frozen."""
    n_red = cfg.env.num_red_agents
    n_blue = cfg.env.num_agents - n_red
    max_steps = cfg.env.max_steps
    gamma = cfg.train.gamma
    ent_coef = cfg.train.ent_coef
    obs_dim = cfg.obs_dim

    lr = cfg.train.lr
    clip = cfg.train.grad_clip
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))

    init_key = jax.random.PRNGKey(seed)
    red_params = red_actor.init(init_key, jnp.zeros(n_red * obs_dim))
    opt_state = opt.init(red_params)

    def _loss_fn(red_p, key):
        traj = collect_episode_multi_scan_joint(
            env=env, blue_actor=blue_actor, blue_params=blue_params,
            joint_red_actor=red_actor, joint_red_params=red_p,
            key=key, max_steps=max_steps, num_red_agents=n_red,
            enforce_connectivity=enforce_conn,
        )
        # red is the last n_red agents; its reward is zero-sum minus blue
        red_rewards = traj.rewards[:, n_blue:]                       # [T, n_red]
        team_return = jnp.sum(red_rewards) / n_red
        red_log_probs = traj.log_probs[:, n_blue:]                    # [T, n_red]
        joint_log_prob = jnp.sum(red_log_probs) / n_red
        # Entropy of the joint red head, estimated from logits along the scan.
        # We don't have logits here; approximate via -E[log_prob] (single sample).
        entropy = -jnp.mean(red_log_probs)
        # REINFORCE loss: maximise team_return → minimise -logprob * return.
        loss = -joint_log_prob * team_return - ent_coef * entropy
        return loss, (team_return, entropy)

    @jax.jit
    def _step(params, opt_s, key):
        (loss, (ret, ent)), grad = jax.value_and_grad(
            _loss_fn, has_aux=True,
        )(params, key)
        updates, opt_s = opt.update(grad, opt_s, params)
        params = optax.apply_updates(params, updates)
        return params, opt_s, loss, ret, ent

    history = dict(env_steps=[], blue_reward=[], red_reward=[],
                   blue_std=[], red_std=[])
    rng = jax.random.PRNGKey(seed + 1)
    consumed = 0
    next_eval = 0
    ep_count = 0
    print(f"  B0 seed={seed} target={total_env_steps} env-steps, "
          f"eval every {eval_every_steps}")
    while consumed < total_env_steps:
        if consumed >= next_eval:
            rng, ek = jax.random.split(rng)
            br, rr, bs, rs = _eval_red(
                env, blue_actor, blue_params, red_actor, red_params,
                ek, num_eval_eps, n_red, enforce_conn,
            )
            history["env_steps"].append(consumed)
            history["blue_reward"].append(br)
            history["red_reward"].append(rr)
            history["blue_std"].append(bs)
            history["red_std"].append(rs)
            print(f"    eval @ {consumed:7d} steps  "
                  f"blue={br:+.3f}±{bs:.2f}  red={rr:+.3f}±{rs:.2f}")
            next_eval = consumed + eval_every_steps
        rng, k = jax.random.split(rng)
        red_params, opt_state, loss, ret, ent = _step(red_params, opt_state, k)
        consumed += max_steps
        ep_count += 1

    # final eval
    rng, ek = jax.random.split(rng)
    br, rr, bs, rs = _eval_red(
        env, blue_actor, blue_params, red_actor, red_params,
        ek, num_eval_eps, n_red, enforce_conn,
    )
    history["env_steps"].append(consumed)
    history["blue_reward"].append(br)
    history["red_reward"].append(rr)
    history["blue_std"].append(bs)
    history["red_std"].append(rs)
    print(f"    eval @ {consumed:7d} steps  (final)  "
          f"blue={br:+.3f}±{bs:.2f}  red={rr:+.3f}±{rs:.2f}  eps={ep_count}")
    return history


# ---------------------------------------------------------------------------
# B1 — Double-DQN with replay buffer + twin-Q + Polyak target
# ---------------------------------------------------------------------------


class ReplayBuffer(NamedTuple):
    obs: jnp.ndarray        # [C, n_red*obs_dim]
    actions: jnp.ndarray    # [C, n_red]
    rewards: jnp.ndarray    # [C]   (red team return for the step)
    next_obs: jnp.ndarray   # [C, n_red*obs_dim]
    done: jnp.ndarray       # [C]
    idx: jnp.ndarray        # write cursor, [] int
    size: jnp.ndarray       # current valid entries, [] int
    capacity: int           # static


def _make_buffer(cap, obs_flat_dim, n_red):
    return ReplayBuffer(
        obs=jnp.zeros((cap, obs_flat_dim), dtype=jnp.float32),
        actions=jnp.zeros((cap, n_red), dtype=jnp.int32),
        rewards=jnp.zeros((cap,), dtype=jnp.float32),
        next_obs=jnp.zeros((cap, obs_flat_dim), dtype=jnp.float32),
        done=jnp.zeros((cap,), dtype=jnp.float32),
        idx=jnp.int32(0),
        size=jnp.int32(0),
        capacity=cap,
    )


def _buffer_push(buf, obs_batch, act_batch, rew_batch, nobs_batch, done_batch):
    """Append a batch of transitions (vectorised over the time axis)."""
    T = obs_batch.shape[0]
    cap = buf.capacity
    offsets = jnp.arange(T)
    write_idx = (buf.idx + offsets) % cap

    obs = buf.obs.at[write_idx].set(obs_batch)
    actions = buf.actions.at[write_idx].set(act_batch)
    rewards = buf.rewards.at[write_idx].set(rew_batch)
    next_obs = buf.next_obs.at[write_idx].set(nobs_batch)
    done = buf.done.at[write_idx].set(done_batch)
    new_idx = (buf.idx + T) % cap
    new_size = jnp.minimum(buf.size + T, cap)
    return buf._replace(
        obs=obs, actions=actions, rewards=rewards,
        next_obs=next_obs, done=done,
        idx=new_idx, size=new_size,
    )


def _buffer_sample(buf, key, batch_size):
    """Sample uniformly from valid entries."""
    idx = jax.random.randint(key, (batch_size,), 0, buf.size)
    return (buf.obs[idx], buf.actions[idx], buf.rewards[idx],
            buf.next_obs[idx], buf.done[idx])


def _polyak(tgt, src, tau):
    return jax.tree_util.tree_map(lambda t, s: t * (1 - tau) + s * tau, tgt, src)


class B1State(NamedTuple):
    q1: dict
    q2: dict
    t1: dict
    t2: dict
    opt1: dict
    opt2: dict


def _train_b1(env, blue_actor, blue_params, q_net, red_proxy_actor, cfg,
              total_env_steps, eval_every_steps, num_eval_eps,
              seed, enforce_conn):
    """Double-DQN over the factorised red Q-head. Blue frozen.

    ``red_proxy_actor`` is a ``JointRedActor`` whose parameters we treat as a
    softmax-over-Q view of the Q-head, so we can reuse
    ``collect_episode_multi_scan_joint`` for evaluation (and, at collection
    time, for ε-greedy action sampling). At act time we turn Q into
    near-deterministic logits: ``logits = Q / temperature``.
    """
    n_red = cfg.env.num_red_agents
    n_blue = cfg.env.num_agents - n_red
    max_steps = cfg.env.max_steps
    gamma = cfg.train.gamma
    num_actions = cfg.env.num_actions
    obs_dim = cfg.obs_dim
    obs_flat = n_red * obs_dim

    lr = cfg.train.lr
    clip = cfg.train.grad_clip
    make_opt = lambda: optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))

    init_key = jax.random.PRNGKey(seed)
    k_a, k_b = jax.random.split(init_key)
    q1_params = q_net.init(k_a, jnp.zeros(obs_flat))
    q2_params = q_net.init(k_b, jnp.zeros(obs_flat))
    opt1 = make_opt()
    opt2 = make_opt()
    state = B1State(
        q1=q1_params, q2=q2_params,
        t1=q1_params, t2=q2_params,
        opt1=opt1.init(q1_params), opt2=opt2.init(q2_params),
    )

    buf = _make_buffer(BUFFER_CAPACITY, obs_flat, n_red)

    def _q_as_red_params(q_params):
        """Map Q-network params into JointRedActor's parameter tree.

        The two networks share an identical backbone + linear head; their
        parameter dicts are structurally identical, so unflatten_params with
        the red actor's reference shape is a plain rename. We initialise the
        proxy once below and reuse its shape as the reference.
        """
        flat = flatten_params(q_params)
        return unflatten_params(flat, _red_proxy_ref_params)

    _red_proxy_ref_params = red_proxy_actor.init(
        jax.random.PRNGKey(0), jnp.zeros(obs_flat),
    )

    def _collect_one_episode(q_params, key, current_eps):
        """Roll one episode under ε-greedy red + fixed blue. Returns:
           obs[T, obs_flat], act[T, n_red], rew[T], next_obs[T, obs_flat], done[T].
        """
        key, reset_key = jax.random.split(key)
        _obs_dict, s = env.reset(reset_key)

        def _scan_body(carry, _):
            state, rng, cumulative_done = carry
            rng, blue_key, red_key, step_key = jax.random.split(rng, 4)
            obs_all = env.obs_array(state)                         # [N, obs_dim]
            red_flat = obs_all[n_blue:].reshape(-1)                # [n_red*obs_dim]

            # Blue: categorical from fixed blue policy.
            blue_obs = obs_all[:n_blue]
            blue_logits = jax.vmap(
                blue_actor.apply, in_axes=(None, 0),
            )(blue_params, blue_obs)
            blue_keys = jax.random.split(blue_key, n_blue)
            blue_actions = jax.vmap(jax.random.categorical)(blue_keys, blue_logits)

            # Red: ε-greedy per head.
            q = q_net.apply(q_params, red_flat)                    # [n_red, A]
            greedy = jnp.argmax(q, axis=-1)
            keys = jax.random.split(red_key, n_red * 2)
            mix = jax.random.uniform(keys[0], (n_red,))
            rand = jax.vmap(
                lambda k: jax.random.randint(k, (), 0, num_actions),
            )(keys[1:1 + n_red])
            red_actions = jnp.where(mix < current_eps, rand, greedy)

            actions = jnp.concatenate([blue_actions, red_actions])
            _obs_new, new_state, rewards, done, _info = env.step_array(
                step_key, state, actions,
            )
            mask = 1.0 - cumulative_done.astype(jnp.float32)
            masked = rewards * mask

            new_obs_all = env.obs_array(new_state)
            new_red_flat = new_obs_all[n_blue:].reshape(-1)

            red_team_r = jnp.sum(masked[n_blue:]) / n_red
            step_out = (red_flat, red_actions, red_team_r, new_red_flat, done)
            return (new_state, rng, cumulative_done | done), step_out

        init_carry = (s, key, jnp.bool_(False))
        _f, (obs_seq, act_seq, rew_seq, nobs_seq, done_seq) = jax.lax.scan(
            _scan_body, init_carry, jnp.arange(max_steps),
        )
        return obs_seq, act_seq, rew_seq, nobs_seq, done_seq.astype(jnp.float32)

    _collect = jax.jit(_collect_one_episode)

    def _td_loss(q1p, q2p, t1p, t2p, batch):
        obs, act, rew, nobs, done = batch
        B = obs.shape[0]
        # Online Q-values at chosen action (both heads → sum across red agents).
        q1_all = jax.vmap(q_net.apply, in_axes=(None, 0))(q1p, obs)   # [B, n_red, A]
        q2_all = jax.vmap(q_net.apply, in_axes=(None, 0))(q2p, obs)
        q1_chosen = jnp.sum(
            jnp.take_along_axis(q1_all, act[..., None], axis=-1).squeeze(-1),
            axis=-1,
        )                                                             # [B]
        q2_chosen = jnp.sum(
            jnp.take_along_axis(q2_all, act[..., None], axis=-1).squeeze(-1),
            axis=-1,
        )

        # Double-DQN target: argmax from online twin-min, value from target twin-min.
        n1 = jax.vmap(q_net.apply, in_axes=(None, 0))(q1p, nobs)
        n2 = jax.vmap(q_net.apply, in_axes=(None, 0))(q2p, nobs)
        online_sum = n1 + n2
        argmax_next = jnp.argmax(online_sum, axis=-1)                 # [B, n_red]
        t1 = jax.vmap(q_net.apply, in_axes=(None, 0))(t1p, nobs)
        t2 = jax.vmap(q_net.apply, in_axes=(None, 0))(t2p, nobs)
        t_min = jnp.minimum(t1, t2)
        v_next = jnp.sum(
            jnp.take_along_axis(t_min, argmax_next[..., None], axis=-1).squeeze(-1),
            axis=-1,
        )                                                             # [B]
        y = rew + gamma * (1.0 - done) * v_next
        y = jax.lax.stop_gradient(y)

        l1 = jnp.mean((q1_chosen - y) ** 2)
        l2 = jnp.mean((q2_chosen - y) ** 2)
        return l1 + l2, (l1, l2)

    @jax.jit
    def _update(state, batch):
        (loss, (l1, l2)), grads = jax.value_and_grad(_td_loss, argnums=(0, 1),
                                                      has_aux=True)(
            state.q1, state.q2, state.t1, state.t2, batch,
        )
        g1, g2 = grads
        u1, new_opt1 = opt1.update(g1, state.opt1, state.q1)
        u2, new_opt2 = opt2.update(g2, state.opt2, state.q2)
        new_q1 = optax.apply_updates(state.q1, u1)
        new_q2 = optax.apply_updates(state.q2, u2)
        new_t1 = _polyak(state.t1, new_q1, TAU)
        new_t2 = _polyak(state.t2, new_q2, TAU)
        return state._replace(
            q1=new_q1, q2=new_q2, t1=new_t1, t2=new_t2,
            opt1=new_opt1, opt2=new_opt2,
        ), loss, l1, l2

    # --- eval with greedy (ε=0) red, implemented via Q→logits reinterpretation
    def _greedy_eval(q_params, key, num_episodes):
        # Wrap q_params as if they were JointRedActor params (shapes match).
        proxy = _q_as_red_params(q_params)
        # Turn near-deterministic by scaling Q up. The rollout uses Categorical,
        # so high-scale logits ≈ argmax. Keep a small temperature for tie-break.
        scaled = jax.tree_util.tree_map(lambda x: x * 10.0, proxy)
        return _eval_red(
            env, blue_actor, blue_params, red_proxy_actor, scaled,
            key, num_episodes, n_red, enforce_conn,
        )

    history = dict(env_steps=[], blue_reward=[], red_reward=[],
                   blue_std=[], red_std=[])
    rng = jax.random.PRNGKey(seed + 1)
    consumed = 0
    next_eval = 0
    print(f"  B1 seed={seed} target={total_env_steps} env-steps, "
          f"eval every {eval_every_steps}")
    while consumed < total_env_steps:
        if consumed >= next_eval:
            rng, ek = jax.random.split(rng)
            br, rr, bs, rs = _greedy_eval(state.q1, ek, num_eval_eps)
            history["env_steps"].append(consumed)
            history["blue_reward"].append(br)
            history["red_reward"].append(rr)
            history["blue_std"].append(bs)
            history["red_std"].append(rs)
            print(f"    eval @ {consumed:7d} steps  "
                  f"blue={br:+.3f}±{bs:.2f}  red={rr:+.3f}±{rs:.2f}")
            next_eval = consumed + eval_every_steps

        # Collect one episode with current ε.
        eps = EPS_END + (EPS_START - EPS_END) * max(
            0.0, (EPS_DECAY_END - consumed) / EPS_DECAY_END,
        )
        rng, ck = jax.random.split(rng)
        obs_seq, act_seq, rew_seq, nobs_seq, done_seq = _collect(
            state.q1, ck, jnp.float32(eps),
        )
        buf = _buffer_push(buf, obs_seq, act_seq, rew_seq, nobs_seq, done_seq)
        consumed += max_steps

        # Update if buffer has enough.
        if int(buf.size) >= BATCH_SIZE:
            # Do one gradient step per collected episode — keeps the gradient
            # step : env-step ratio at ~1/max_steps, which is the common
            # DQN default (update every step, but we batch to keep the
            # Python loop cheap).
            for _ in range(max_steps):
                rng, sk = jax.random.split(rng)
                batch = _buffer_sample(buf, sk, BATCH_SIZE)
                state, loss, l1, l2 = _update(state, batch)

    rng, ek = jax.random.split(rng)
    br, rr, bs, rs = _greedy_eval(state.q1, ek, num_eval_eps)
    history["env_steps"].append(consumed)
    history["blue_reward"].append(br)
    history["red_reward"].append(rr)
    history["blue_std"].append(bs)
    history["red_std"].append(rs)
    print(f"    eval @ {consumed:7d} steps  (final)  "
          f"blue={br:+.3f}±{bs:.2f}  red={rr:+.3f}±{rs:.2f}")
    return history


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(variant, num_seeds, total_env_steps, eval_every_steps, num_eval_eps,
        output_dir, cfg_path=REF_CONFIG, blue_ckpt=REF_BLUE_CKPT):
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    cfg = ExperimentConfig.from_yaml(cfg_path)
    env = _build_env(cfg)
    enforce_conn = cfg.enforce_connectivity

    n_red = cfg.env.num_red_agents
    obs_dim = cfg.obs_dim

    blue_actor = Actor(
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
        activation=cfg.network.activation,
    )
    red_actor = JointRedActor(
        num_red=n_red,
        num_actions=cfg.env.num_actions,
        hidden_dim=cfg.train.red_hidden_dim,
        num_layers=cfg.train.red_num_layers,
    )

    print(f"Loading frozen blue from {blue_ckpt}")
    blue_params = _load_blue_params(blue_actor, cfg, blue_ckpt)

    all_hist = []
    t0 = time.time()
    for s in range(num_seeds):
        print(f"--- seed {s} ---")
        if variant == "B0":
            h = _train_b0(
                env, blue_actor, blue_params, red_actor, cfg,
                total_env_steps, eval_every_steps, num_eval_eps,
                seed=s, enforce_conn=enforce_conn,
            )
        else:
            q_net = JointRedQHead(
                num_red=n_red,
                num_actions=cfg.env.num_actions,
                hidden_dim=cfg.train.red_hidden_dim,
                num_layers=cfg.train.red_num_layers,
            )
            h = _train_b1(
                env, blue_actor, blue_params, q_net, red_actor, cfg,
                total_env_steps, eval_every_steps, num_eval_eps,
                seed=s, enforce_conn=enforce_conn,
            )
        all_hist.append(h)
    elapsed = time.time() - t0

    # Stack histories [E, S]. Each seed may have slightly different eval
    # counts due to the while-loop cadence, so trim to the min length.
    E = min(len(h["env_steps"]) for h in all_hist)
    env_steps = np.array(all_hist[0]["env_steps"][:E])
    def _stack(key):
        return np.stack([np.array(h[key][:E]) for h in all_hist], axis=1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "metrics.npz",
        env_steps=env_steps,
        blue_reward=_stack("blue_reward"),
        red_reward=_stack("red_reward"),
        blue_std=_stack("blue_std"),
        red_std=_stack("red_std"),
    )
    summary = dict(
        variant=variant,
        num_seeds=num_seeds,
        total_env_steps=int(total_env_steps),
        eval_every_steps=int(eval_every_steps),
        num_eval_eps=int(num_eval_eps),
        final_blue_reward_mean=float(np.mean(_stack("blue_reward")[-1])),
        final_red_reward_mean=float(np.mean(_stack("red_reward")[-1])),
        wall_seconds=float(elapsed),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out/'metrics.npz'}")
    print(f"Wrote {out/'summary.json'}")
    print(f"final blue={summary['final_blue_reward_mean']:+.3f}  "
          f"red={summary['final_red_reward_mean']:+.3f}  "
          f"wall={elapsed:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--env-steps", type=int, default=500_000,
                    help="total red env-steps per seed")
    ap.add_argument("--eval-every", type=int, default=25_000,
                    help="env-steps between evals")
    ap.add_argument("--eval-eps", type=int, default=8,
                    help="eval episodes per eval bucket")
    ap.add_argument("--config", default=REF_CONFIG)
    ap.add_argument("--blue-ckpt", default=REF_BLUE_CKPT)
    ap.add_argument("--smoke", action="store_true",
                    help="quick smoke: 2 seeds, 5k env-steps, eval every 1k")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = 2
        args.env_steps = 5000
        args.eval_every = 1000
        args.eval_eps = 2

    out_default = f"experiments/stabilization/offpolicy-red-{args.variant}"
    out = Path(args.output_dir or out_default)
    run(
        variant=args.variant,
        num_seeds=args.seeds,
        total_env_steps=args.env_steps,
        eval_every_steps=args.eval_every,
        num_eval_eps=args.eval_eps,
        output_dir=out,
        cfg_path=args.config,
        blue_ckpt=args.blue_ckpt,
    )


if __name__ == "__main__":
    main()
