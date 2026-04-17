"""Unified PureJaxRL trainer for RedWithinBlue.

The entire training loop is a single JIT-compiled function built by
``make_train(config)``.  This function:

1. Initialises actor/critic parameters and optimizer state.
2. Runs ``jax.lax.scan`` over ``num_episodes`` training steps.
3. Each step: collect episode -> compute loss -> gradient update -> accumulate metrics.
4. Returns final parameters and per-episode metrics.

``make_train_multi_seed(config)`` vmaps the training function over
independent seeds for parallel evaluation.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, Critic
from red_within_blue.training.losses import (
    compute_discounted_returns,
    pg_loss,
    pg_loss_with_baseline,
    actor_critic_loss,
)
from red_within_blue.training.rollout import (
    collect_episode_scan,
    collect_episode_multi_scan,
)
from red_within_blue.training.rewards_training import (
    make_multi_agent_reward,
    normalized_exploration_reward,
)
from red_within_blue.env import GridCommEnv


# ---------------------------------------------------------------------------
# Factory: make_train
# ---------------------------------------------------------------------------


def make_train(config: ExperimentConfig, init_actor_params=None, init_critic_params=None):
    """Build a JIT-compiled training function from *config*.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    init_actor_params : optional
        Pre-trained actor params for warm-starting. If ``None``, params
        are initialised randomly.
    init_critic_params : optional
        Pre-trained critic params for warm-starting (actor-critic only).

    Returns
    -------
    train_fn : ``(key: jax.Array) -> (actor_params, critic_params | None, metrics)``
        A JIT-compiled function that runs the full training loop.
        ``metrics`` is a dict with arrays of shape ``[num_episodes]``.
    """
    # ---- unpack config (all resolved at trace time) ----
    method = config.train.method
    num_episodes = config.train.num_episodes
    lr = config.train.lr
    gamma = config.train.gamma
    vf_coef = config.train.vf_coef
    max_steps = config.env.max_steps
    num_agents = config.env.num_agents
    obs_dim = config.obs_dim
    num_actions = config.env.num_actions
    enforce_connectivity = config.enforce_connectivity

    # ---- build env ----
    env_config = config.to_env_config()
    if num_agents == 1:
        reward_fn = normalized_exploration_reward
    else:
        reward_fn = make_multi_agent_reward(
            disconnect_penalty=config.reward.disconnect_penalty,
        )
    env = GridCommEnv(env_config, reward_fn=reward_fn)

    # ---- build networks ----
    actor = Actor(
        num_actions=num_actions,
        hidden_dim=config.network.actor_hidden_dim,
        num_layers=config.network.actor_num_layers,
    )
    use_critic = method == "actor_critic"
    if use_critic:
        critic = Critic(
            hidden_dim=config.network.critic_hidden_dim,
            num_layers=config.network.critic_num_layers,
        )

    # ---- define the training function ----
    @jax.jit
    def train_fn(key: jax.Array):
        # -- initialise parameters (warm-start or random) --
        key, actor_key, critic_key = jax.random.split(key, 3)
        dummy_obs = jnp.zeros(obs_dim)

        if init_actor_params is not None:
            actor_params = init_actor_params
        else:
            actor_params = actor.init(actor_key, dummy_obs)

        if use_critic:
            if init_critic_params is not None:
                critic_params = init_critic_params
            else:
                critic_params = critic.init(critic_key, dummy_obs)
        else:
            critic_params = None

        # -- initialise optimisers --
        actor_opt = optax.adam(lr)
        actor_opt_state = actor_opt.init(actor_params)

        if use_critic:
            critic_opt = optax.adam(lr)
            critic_opt_state = critic_opt.init(critic_params)

        # ---- per-episode training step (for lax.scan) ----

        if method == "reinforce" or method == "baseline":
            # ---- REINFORCE / BASELINE ----
            loss_fn = pg_loss if method == "reinforce" else pg_loss_with_baseline

            if num_agents == 1:
                # -- single-agent --
                def _train_step(carry, _):
                    actor_params, actor_opt_state, rng = carry
                    rng, ep_key = jax.random.split(rng)

                    # Collect episode
                    traj = collect_episode_scan(
                        env, actor, actor_params, ep_key, max_steps,
                    )

                    # Compute returns and loss
                    def _loss(ap):
                        logits = jax.vmap(actor.apply, in_axes=(None, 0))(ap, traj.obs)
                        returns = compute_discounted_returns(traj.rewards, traj.dones, gamma)
                        return loss_fn(logits, traj.actions, returns)

                    loss_val, grads = jax.value_and_grad(_loss)(actor_params)
                    updates, new_opt_state = actor_opt.update(grads, actor_opt_state)
                    new_params = optax.apply_updates(actor_params, updates)

                    total_reward = jnp.sum(traj.rewards * traj.mask)
                    metrics = {"loss": loss_val, "total_reward": total_reward}

                    return (new_params, new_opt_state, rng), metrics

            else:
                # -- multi-agent --
                def _train_step(carry, _):
                    actor_params, actor_opt_state, rng = carry
                    rng, ep_key = jax.random.split(rng)

                    traj = collect_episode_multi_scan(
                        env, actor, actor_params, ep_key, max_steps,
                        enforce_connectivity=enforce_connectivity,
                    )

                    def _loss(ap):
                        # returns per agent: vmap over agent dim (axis 1 of [T, N])
                        per_agent_returns = jax.vmap(
                            lambda r, d: compute_discounted_returns(r, d, gamma),
                            in_axes=(1, None),
                        )(traj.rewards, traj.dones)  # [N, T]

                        # logits per agent
                        per_agent_logits = jax.vmap(
                            lambda obs: jax.vmap(actor.apply, in_axes=(None, 0))(ap, obs),
                            in_axes=1,
                        )(traj.obs)  # [N, T, num_actions]

                        # loss per agent and mean
                        per_agent_loss = jax.vmap(loss_fn)(
                            per_agent_logits, traj.actions.T, per_agent_returns,
                        )
                        return jnp.mean(per_agent_loss)

                    loss_val, grads = jax.value_and_grad(_loss)(actor_params)
                    updates, new_opt_state = actor_opt.update(grads, actor_opt_state)
                    new_params = optax.apply_updates(actor_params, updates)

                    total_reward = jnp.sum(traj.rewards * traj.mask[:, None])
                    metrics = {"loss": loss_val, "total_reward": total_reward}

                    return (new_params, new_opt_state, rng), metrics

            # Scan over episodes
            def _scan_body(carry, step_idx):
                ap, ao, rng = carry
                (new_ap, new_ao, new_rng), metrics = _train_step(
                    (ap, ao, rng), step_idx,
                )
                return (new_ap, new_ao, new_rng), metrics

            init_carry = (actor_params, actor_opt_state, key)
            (final_actor_params, _, _), all_metrics = jax.lax.scan(
                _scan_body, init_carry, jnp.arange(num_episodes),
            )

            return final_actor_params, None, all_metrics

        elif method == "actor_critic":
            # ---- ACTOR-CRITIC ----

            if num_agents == 1:
                # -- single-agent --
                def _train_step(carry, _):
                    actor_params, critic_params, actor_opt_state, critic_opt_state, rng = carry
                    rng, ep_key = jax.random.split(rng)

                    traj = collect_episode_scan(
                        env, actor, actor_params, ep_key, max_steps,
                    )

                    # Joint loss w.r.t. both actor and critic
                    def _joint_loss(ap, cp):
                        policy_loss, value_loss = actor_critic_loss(
                            actor, critic, ap, cp,
                            traj.obs, traj.actions, traj.rewards, traj.dones, gamma,
                        )
                        total = policy_loss + vf_coef * value_loss
                        return total, (policy_loss, value_loss)

                    (loss_val, (_pl, _vl)), (a_grads, c_grads) = jax.value_and_grad(
                        _joint_loss, argnums=(0, 1), has_aux=True,
                    )(actor_params, critic_params)

                    a_updates, new_ao = actor_opt.update(a_grads, actor_opt_state)
                    new_ap = optax.apply_updates(actor_params, a_updates)

                    c_updates, new_co = critic_opt.update(c_grads, critic_opt_state)
                    new_cp = optax.apply_updates(critic_params, c_updates)

                    total_reward = jnp.sum(traj.rewards * traj.mask)
                    metrics = {"loss": loss_val, "total_reward": total_reward}

                    return (new_ap, new_cp, new_ao, new_co, rng), metrics

            else:
                # -- multi-agent --
                def _train_step(carry, _):
                    actor_params, critic_params, actor_opt_state, critic_opt_state, rng = carry
                    rng, ep_key = jax.random.split(rng)

                    traj = collect_episode_multi_scan(
                        env, actor, actor_params, ep_key, max_steps,
                        enforce_connectivity=enforce_connectivity,
                    )

                    def _joint_loss(ap, cp):
                        # vmap actor_critic_loss over agent dimension
                        def _per_agent_loss(obs_i, actions_i, rewards_i):
                            return actor_critic_loss(
                                actor, critic, ap, cp,
                                obs_i, actions_i, rewards_i, traj.dones, gamma,
                            )

                        per_agent_pl, per_agent_vl = jax.vmap(
                            _per_agent_loss, in_axes=(1, 1, 1),
                        )(traj.obs, traj.actions, traj.rewards)  # each [N]

                        policy_loss = jnp.mean(per_agent_pl)
                        value_loss = jnp.mean(per_agent_vl)
                        total = policy_loss + vf_coef * value_loss
                        return total, (policy_loss, value_loss)

                    (loss_val, (_pl, _vl)), (a_grads, c_grads) = jax.value_and_grad(
                        _joint_loss, argnums=(0, 1), has_aux=True,
                    )(actor_params, critic_params)

                    a_updates, new_ao = actor_opt.update(a_grads, actor_opt_state)
                    new_ap = optax.apply_updates(actor_params, a_updates)

                    c_updates, new_co = critic_opt.update(c_grads, critic_opt_state)
                    new_cp = optax.apply_updates(critic_params, c_updates)

                    total_reward = jnp.sum(traj.rewards * traj.mask[:, None])
                    metrics = {"loss": loss_val, "total_reward": total_reward}

                    return (new_ap, new_cp, new_ao, new_co, rng), metrics

            def _scan_body(carry, step_idx):
                ap, cp, ao, co, rng = carry
                (new_ap, new_cp, new_ao, new_co, new_rng), metrics = _train_step(
                    (ap, cp, ao, co, rng), step_idx,
                )
                return (new_ap, new_cp, new_ao, new_co, new_rng), metrics

            init_carry = (
                actor_params, critic_params,
                actor_opt_state, critic_opt_state,
                key,
            )
            (final_actor_params, final_critic_params, _, _, _), all_metrics = (
                jax.lax.scan(_scan_body, init_carry, jnp.arange(num_episodes))
            )

            return final_actor_params, final_critic_params, all_metrics

        else:
            raise ValueError(f"Unknown training method: {method!r}")

    return train_fn


# ---------------------------------------------------------------------------
# Multi-seed wrapper
# ---------------------------------------------------------------------------


def make_train_multi_seed(config: ExperimentConfig, init_actor_params=None, init_critic_params=None):
    """Build a training function that vmaps over independent seeds.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    init_actor_params : optional
        Pre-trained actor params for warm-starting.
    init_critic_params : optional
        Pre-trained critic params for warm-starting.

    Returns
    -------
    train_multi : ``(key: jax.Array) -> (actor_params, critic_params | None, metrics)``
        All outputs gain a leading ``[num_seeds]`` dimension.
    """
    train_fn = make_train(config, init_actor_params, init_critic_params)

    @jax.jit
    def train_multi(key: jax.Array):
        keys = jax.random.split(key, config.train.num_seeds)
        return jax.vmap(train_fn)(keys)

    return train_multi
