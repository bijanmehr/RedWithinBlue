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
from red_within_blue.training.networks import Actor, Critic, JointRedActor
from red_within_blue.training.losses import (
    compute_discounted_returns,
    pg_loss,
    pg_loss_with_baseline,
    actor_critic_loss,
    actor_critic_loss_ctde,
)
from red_within_blue.training.rollout import (
    collect_episode_scan,
    collect_episode_multi_scan,
    collect_episode_multi_scan_joint,
)
from red_within_blue.training.rewards_training import (
    make_multi_agent_reward,
    normalized_exploration_reward,
)
from red_within_blue.training import progress as progress_bar
from red_within_blue.env import GridCommEnv


def _maybe_report_progress(step_idx, metrics, num_episodes: int, enabled: bool):
    """Emit a gated host callback (~200 updates per run) to the progress bar.

    Always safe to call: when ``enabled`` is False this is a no-op and the
    XLA trace stays free of host hooks. When True it fires roughly every
    ``num_episodes // 200`` steps plus on the final step.
    """
    if not enabled or num_episodes <= 0:
        return
    interval = max(1, num_episodes // 200)
    should_log = jnp.logical_or(
        step_idx % interval == 0, step_idx == num_episodes - 1,
    )

    def _emit(_):
        jax.debug.callback(
            progress_bar.update,
            step_idx, metrics["loss"], metrics["total_reward"],
            ordered=True,
        )
        return 0

    def _skip(_):
        return 0

    jax.lax.cond(should_log, _emit, _skip, 0)


# ---------------------------------------------------------------------------
# Factory: make_train
# ---------------------------------------------------------------------------


def make_train(
    config: ExperimentConfig,
    init_actor_params=None,
    init_critic_params=None,
    *,
    report_progress: bool = False,
):
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
    report_progress : bool, keyword-only
        If True, emit ``jax.debug.callback`` progress pings into
        ``training.progress`` so the runner's tqdm bar can update live.
        The runner sets this; tests leave it off so the trace stays free
        of host hooks.

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
    ent_coef = config.train.ent_coef
    max_steps = config.env.max_steps
    num_agents = config.env.num_agents
    obs_dim = config.obs_dim
    num_actions = config.env.num_actions
    # Disconnect-grace supersedes the hard guardrail — when grace > 0, the env
    # handles the soft failure trigger and the policy is free to disconnect.
    enforce_connectivity = bool(config.enforce_connectivity) and int(config.env.disconnect_grace) == 0
    epsilon = float(config.train.epsilon)
    epsilon_final = float(config.train.epsilon_final)
    ent_coef_final = float(config.train.ent_coef_final)
    anneal_end_frac = float(config.train.anneal_end_frac)
    grad_clip = float(config.train.grad_clip)

    # Linear schedule: if *_final >= 0, anneal from initial to final across
    # the first ``anneal_end_frac * num_episodes`` episodes, then hold at
    # final for the remainder. Otherwise hold constant at init for all.
    # Evaluated at trace time to a JAX-friendly scalar function of step_idx.
    _anneal_end_step = max(int(anneal_end_frac * (num_episodes - 1)), 1)

    def _sched(step_idx, init_val, final_val):
        frac = jnp.minimum(
            step_idx.astype(jnp.float32) / jnp.float32(_anneal_end_step),
            jnp.float32(1.0),
        )
        annealed = init_val + (final_val - init_val) * frac
        return jnp.where(jnp.float32(final_val) >= 0.0, annealed, jnp.float32(init_val))

    def _build_opt(lr_: float) -> optax.GradientTransformation:
        """Adam, optionally wrapped with global-norm gradient clipping."""
        if grad_clip > 0.0:
            return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_))
        return optax.adam(lr_)

    # Team masks (resolved at trace time)
    _n_red = config.env.num_red_agents
    _n_blue = num_agents - _n_red
    _blue_mask_np = jnp.arange(num_agents) < _n_blue  # [N] bool
    _red_policy = config.train.red_policy  # "shared" | "random"

    # Per-agent learning weight: 1.0 for agents whose trajectories drive
    # policy gradients, 0.0 otherwise.  Random-red agents don't learn.
    if _red_policy == "random" and _n_red > 0:
        _learn_mask_np = _blue_mask_np.astype(jnp.float32)
    else:
        _learn_mask_np = jnp.ones(num_agents, dtype=jnp.float32)
    _learn_mask_sum = jnp.maximum(jnp.sum(_learn_mask_np), 1.0)

    # ---- build env ----
    env_config = config.to_env_config()
    num_red_agents = config.env.num_red_agents
    # Unified path for solo, cooperative multi-agent, and adversarial cases:
    # for N=1 the graph is trivially connected and all cohesion terms evaluate
    # to 0; for ``num_red_agents > 0`` the zero-sum overlay inside
    # ``make_multi_agent_reward`` overwrites the last ``num_red_agents`` slots
    # with ``-sum(blue_rewards) / num_red_agents`` so team totals sum to zero.
    reward_fn = make_multi_agent_reward(
        disconnect_penalty=config.reward.disconnect_penalty,
        isolation_weight=config.reward.isolation_weight,
        cooperative_weight=config.reward.cooperative_weight,
        revisit_weight=config.reward.revisit_weight,
        terminal_bonus_scale=config.reward.terminal_bonus_scale,
        terminal_bonus_divide=config.reward.terminal_bonus_divide,
        spread_weight=config.reward.spread_weight,
        fog_potential_weight=config.reward.fog_potential_weight,
        num_red_agents=num_red_agents,
    )
    env = GridCommEnv(env_config, reward_fn=reward_fn)

    # ---- build networks ----
    actor = Actor(
        num_actions=num_actions,
        hidden_dim=config.network.actor_hidden_dim,
        num_layers=config.network.actor_num_layers,
        activation=config.network.activation,
    )
    use_critic = method == "actor_critic"
    if use_critic:
        critic = Critic(
            hidden_dim=config.network.critic_hidden_dim,
            num_layers=config.network.critic_num_layers,
            activation=config.network.activation,
        )

    # ---- POSG path: Dec-POMDP blue + POMDP joint-policy red ----
    if _red_policy == "joint" and _n_red > 0:
        if method != "reinforce":
            raise ValueError(
                f"red_policy='joint' requires train.method='reinforce', got {method!r}"
            )
        return _make_train_joint_red(
            config, env, actor, init_actor_params,
            obs_dim=obs_dim, num_agents=num_agents,
            n_red=_n_red, n_blue=_n_blue,
            num_actions=num_actions, num_episodes=num_episodes,
            red_pretrain_episodes=config.train.red_pretrain_episodes,
            max_steps=max_steps, lr=lr, gamma=gamma,
            blue_mask=_blue_mask_np,
            grad_clip=grad_clip,
            ent_coef=ent_coef,
            enforce_connectivity=enforce_connectivity,
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
            # CTDE: for multi-agent AC the central critic sees the joint
            # (concatenated) observation, so its first Dense layer is wider.
            if num_agents > 1:
                critic_dummy = jnp.zeros(num_agents * obs_dim)
            else:
                critic_dummy = dummy_obs
            if init_critic_params is not None:
                critic_params = init_critic_params
            else:
                critic_params = critic.init(critic_key, critic_dummy)
        else:
            critic_params = None

        # -- initialise optimisers --
        actor_opt = _build_opt(lr)
        actor_opt_state = actor_opt.init(actor_params)

        if use_critic:
            critic_opt = _build_opt(lr)
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
                        red_policy=_red_policy,
                        num_red_agents=_n_red,
                        epsilon=epsilon,
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

                        # loss per agent, weighted by learn mask
                        per_agent_loss = jax.vmap(loss_fn)(
                            per_agent_logits, traj.actions.T, per_agent_returns,
                        )
                        return jnp.sum(per_agent_loss * _learn_mask_np) / _learn_mask_sum

                    loss_val, grads = jax.value_and_grad(_loss)(actor_params)
                    updates, new_opt_state = actor_opt.update(grads, actor_opt_state)
                    new_params = optax.apply_updates(actor_params, updates)

                    rewards_m = traj.rewards * traj.mask[:, None]
                    per_agent_reward = jnp.sum(rewards_m, axis=0)
                    blue_total_reward = jnp.sum(per_agent_reward * _blue_mask_np)
                    red_total_reward = jnp.sum(per_agent_reward * (~_blue_mask_np))
                    metrics = {
                        "loss": loss_val,
                        "total_reward": blue_total_reward + red_total_reward,
                        "blue_total_reward": blue_total_reward,
                        "red_total_reward": red_total_reward,
                        "per_agent_reward": per_agent_reward,
                    }

                    return (new_params, new_opt_state, rng), metrics

            # Scan over episodes
            def _scan_body(carry, step_idx):
                ap, ao, rng = carry
                (new_ap, new_ao, new_rng), metrics = _train_step(
                    (ap, ao, rng), step_idx,
                )
                _maybe_report_progress(step_idx, metrics, num_episodes, report_progress)
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
                        policy_loss, value_loss, entropy = actor_critic_loss(
                            actor, critic, ap, cp,
                            traj.obs, traj.actions, traj.rewards, traj.dones, gamma,
                        )
                        total = policy_loss + vf_coef * value_loss - ent_coef * entropy
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
                # -- multi-agent (CTDE: shared actor + central critic) --
                def _train_step(carry, step_idx):
                    actor_params, critic_params, actor_opt_state, critic_opt_state, rng = carry
                    rng, ep_key = jax.random.split(rng)

                    eps_t = _sched(step_idx, epsilon, epsilon_final)
                    ent_t = _sched(step_idx, ent_coef, ent_coef_final)

                    traj = collect_episode_multi_scan(
                        env, actor, actor_params, ep_key, max_steps,
                        enforce_connectivity=enforce_connectivity,
                        red_policy=_red_policy,
                        num_red_agents=_n_red,
                        epsilon=eps_t,
                    )

                    def _joint_loss(ap, cp):
                        policy_loss, value_loss, entropy = actor_critic_loss_ctde(
                            actor, critic, ap, cp,
                            traj.obs, traj.actions, traj.rewards, traj.dones, gamma,
                        )
                        total = policy_loss + vf_coef * value_loss - ent_t * entropy
                        return total, (policy_loss, value_loss)

                    (loss_val, (_pl, _vl)), (a_grads, c_grads) = jax.value_and_grad(
                        _joint_loss, argnums=(0, 1), has_aux=True,
                    )(actor_params, critic_params)

                    a_updates, new_ao = actor_opt.update(a_grads, actor_opt_state)
                    new_ap = optax.apply_updates(actor_params, a_updates)

                    c_updates, new_co = critic_opt.update(c_grads, critic_opt_state)
                    new_cp = optax.apply_updates(critic_params, c_updates)

                    rewards_m = traj.rewards * traj.mask[:, None]
                    per_agent_reward = jnp.sum(rewards_m, axis=0)
                    blue_total_reward = jnp.sum(per_agent_reward * _blue_mask_np)
                    red_total_reward = jnp.sum(per_agent_reward * (~_blue_mask_np))
                    metrics = {
                        "loss": loss_val,
                        "total_reward": blue_total_reward + red_total_reward,
                        "blue_total_reward": blue_total_reward,
                        "red_total_reward": red_total_reward,
                        "per_agent_reward": per_agent_reward,
                    }

                    return (new_ap, new_cp, new_ao, new_co, rng), metrics

            def _scan_body(carry, step_idx):
                ap, cp, ao, co, rng = carry
                (new_ap, new_cp, new_ao, new_co, new_rng), metrics = _train_step(
                    (ap, cp, ao, co, rng), step_idx,
                )
                _maybe_report_progress(step_idx, metrics, num_episodes, report_progress)
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
# POSG path: Dec-POMDP blue + POMDP joint-policy red
# ---------------------------------------------------------------------------


def _make_train_joint_red(
    config: ExperimentConfig,
    env: GridCommEnv,
    blue_actor: Actor,
    init_blue_params,
    *,
    obs_dim: int,
    num_agents: int,
    n_red: int,
    n_blue: int,
    num_actions: int,
    num_episodes: int,
    red_pretrain_episodes: int,
    max_steps: int,
    lr: float,
    gamma: float,
    blue_mask: jnp.ndarray,
    grad_clip: float = 0.0,
    ent_coef: float = 0.0,
    enforce_connectivity: bool = False,
):
    """Build the POSG train_fn: per-agent blue actor + centralized joint red.

    Two independent REINFORCE losses, two Adam optimizers, no shared gradient.
    An optional warm-up ``red_pretrain_episodes`` phase freezes blue so red
    can reach a best-response against the warm-started blue before joint
    Nash training begins.

    The return signature matches ``make_train``'s other paths:
    ``(blue_params, red_params, metrics)`` — red params are carried in the
    slot normally used for critic params; ``runner.save_results`` persists
    them to a separate ``joint_red_checkpoint.npz`` file.
    """
    joint_red_actor = JointRedActor(
        num_red=n_red,
        num_actions=num_actions,
        hidden_dim=config.train.red_hidden_dim,
        num_layers=config.train.red_num_layers,
    )
    not_blue_mask = ~blue_mask  # [N] bool

    @jax.jit
    def train_fn(key: jax.Array):
        key, blue_key, red_key = jax.random.split(key, 3)
        dummy_blue_obs = jnp.zeros(obs_dim)
        dummy_red_obs = jnp.zeros(n_red * obs_dim)

        if init_blue_params is not None:
            blue_params = init_blue_params
        else:
            blue_params = blue_actor.init(blue_key, dummy_blue_obs)
        red_params = joint_red_actor.init(red_key, dummy_red_obs)

        def _opt(lr_: float) -> optax.GradientTransformation:
            if grad_clip > 0.0:
                return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_))
            return optax.adam(lr_)
        blue_opt = _opt(lr)
        red_opt = _opt(lr)
        blue_opt_state = blue_opt.init(blue_params)
        red_opt_state = red_opt.init(red_params)

        def _entropy_from_logits(logits):
            """Mean Shannon entropy across all leading axes."""
            log_p = jax.nn.log_softmax(logits, axis=-1)
            p = jnp.exp(log_p)
            return -jnp.mean(jnp.sum(p * log_p, axis=-1))

        def _blue_loss(bp, traj):
            blue_obs = traj.obs[:, :n_blue, :]            # [T, n_blue, obs_dim]
            blue_actions = traj.actions[:, :n_blue]       # [T, n_blue]
            blue_rewards = traj.rewards[:, :n_blue]       # [T, n_blue]

            per_agent_returns = jax.vmap(
                lambda r, d: compute_discounted_returns(r, d, gamma),
                in_axes=(1, None),
            )(blue_rewards, traj.dones)                   # [n_blue, T]

            per_agent_logits = jax.vmap(
                lambda obs: jax.vmap(blue_actor.apply, in_axes=(None, 0))(bp, obs),
                in_axes=1,
            )(blue_obs)                                   # [n_blue, T, num_actions]

            per_agent_loss = jax.vmap(pg_loss)(
                per_agent_logits, blue_actions.T, per_agent_returns,
            )
            entropy = _entropy_from_logits(per_agent_logits)
            total = jnp.mean(per_agent_loss) - ent_coef * entropy
            return total, entropy

        def _red_loss(rp, traj):
            red_obs = traj.obs[:, n_blue:, :]             # [T, n_red, obs_dim]
            red_obs_joint = red_obs.reshape(max_steps, n_red * obs_dim)
            red_actions = traj.actions[:, n_blue:]        # [T, n_red]
            red_rewards = traj.rewards[:, n_blue:]        # [T, n_red]

            team_rewards = jnp.sum(red_rewards, axis=-1)  # [T]
            team_returns = compute_discounted_returns(team_rewards, traj.dones, gamma)

            red_logits = jax.vmap(joint_red_actor.apply, in_axes=(None, 0))(
                rp, red_obs_joint,
            )                                             # [T, n_red, num_actions]
            red_log_probs_full = jax.nn.log_softmax(red_logits, axis=-1)
            joint_log_probs = jax.vmap(
                lambda lp, a: jnp.sum(jax.vmap(lambda l, ai: l[ai])(lp, a))
            )(red_log_probs_full, red_actions)            # [T]

            entropy = _entropy_from_logits(red_logits)
            total = -jnp.mean(joint_log_probs * team_returns) - ent_coef * entropy
            return total, entropy

        def _make_step(train_blue: bool):
            def _step(carry, _):
                bp, rp, bo, ro, rng = carry
                rng, ep_key = jax.random.split(rng)

                traj = collect_episode_multi_scan_joint(
                    env, blue_actor, bp, joint_red_actor, rp,
                    ep_key, max_steps, num_red_agents=n_red,
                    enforce_connectivity=enforce_connectivity,
                )

                (blue_loss_val, blue_entropy), blue_grads = jax.value_and_grad(
                    _blue_loss, has_aux=True,
                )(bp, traj)
                (red_loss_val, red_entropy), red_grads = jax.value_and_grad(
                    _red_loss, has_aux=True,
                )(rp, traj)

                if train_blue:
                    b_updates, new_bo = blue_opt.update(blue_grads, bo)
                    new_bp = optax.apply_updates(bp, b_updates)
                else:
                    new_bp = bp
                    new_bo = bo

                r_updates, new_ro = red_opt.update(red_grads, ro)
                new_rp = optax.apply_updates(rp, r_updates)

                rewards_m = traj.rewards * traj.mask[:, None]
                per_agent_reward = jnp.sum(rewards_m, axis=0)
                blue_total_reward = jnp.sum(per_agent_reward * blue_mask)
                red_total_reward = jnp.sum(per_agent_reward * not_blue_mask)
                metrics = {
                    "loss": blue_loss_val + red_loss_val,
                    "blue_loss": blue_loss_val,
                    "red_loss": red_loss_val,
                    "total_reward": blue_total_reward + red_total_reward,
                    "blue_total_reward": blue_total_reward,
                    "red_total_reward": red_total_reward,
                    "per_agent_reward": per_agent_reward,
                    "blue_policy_entropy": blue_entropy,
                    "red_policy_entropy": red_entropy,
                    "duality_violation": jnp.abs(
                        blue_total_reward + red_total_reward
                    ),
                }
                return (new_bp, new_rp, new_bo, new_ro, rng), metrics

            return _step

        init_carry = (blue_params, red_params, blue_opt_state, red_opt_state, key)

        if red_pretrain_episodes > 0:
            init_carry, pretrain_metrics = jax.lax.scan(
                _make_step(train_blue=False),
                init_carry,
                jnp.arange(red_pretrain_episodes),
            )

        (final_bp, final_rp, _, _, _), joint_metrics = jax.lax.scan(
            _make_step(train_blue=True),
            init_carry,
            jnp.arange(num_episodes),
        )

        if red_pretrain_episodes > 0:
            all_metrics = jax.tree.map(
                lambda p, j: jnp.concatenate([p, j], axis=0),
                pretrain_metrics, joint_metrics,
            )
        else:
            all_metrics = joint_metrics

        return final_bp, final_rp, all_metrics

    return train_fn


# ---------------------------------------------------------------------------
# Multi-seed wrapper
# ---------------------------------------------------------------------------


def _prepare_warm_start(params, num_seeds: int):
    """Broadcast/slice warm-start params to have leading dim equal to num_seeds."""
    if params is None:
        return None
    leaves = jax.tree_util.tree_leaves(params)
    ckpt_seed_dim = leaves[0].shape[0] if leaves[0].ndim >= 2 else None

    if ckpt_seed_dim == num_seeds:
        return jax.tree.map(jnp.asarray, params)
    if ckpt_seed_dim is None:
        return jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (num_seeds,) + x.shape),
            params,
        )
    # Checkpoint has a different seed count: take seed 0 and broadcast
    sliced = jax.tree.map(lambda x: jnp.asarray(x)[0], params)
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_seeds,) + x.shape),
        sliced,
    )


def make_train_multi_seed(
    config: ExperimentConfig,
    init_actor_params=None,
    init_critic_params=None,
    *,
    report_progress: bool = False,
):
    """Build a training function that vmaps over independent seeds.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    init_actor_params : optional
        Pre-trained actor params for warm-starting.
    init_critic_params : optional
        Pre-trained critic params for warm-starting (actor-critic only).
    report_progress : bool, keyword-only
        Forwarded to :func:`make_train`; enables the tqdm progress hook.

    Returns
    -------
    train_multi : ``(key: jax.Array) -> (actor_params, critic_params | None, metrics)``
        All outputs gain a leading ``[num_seeds]`` dimension.
    """
    num_seeds = config.train.num_seeds

    if init_actor_params is None:
        train_fn = make_train(config, report_progress=report_progress)

        @jax.jit
        def train_multi(key: jax.Array):
            keys = jax.random.split(key, num_seeds)
            return jax.vmap(train_fn)(keys)

        return train_multi

    # Warm-start: shape params so each seed gets its own slice via vmap in_axes=0.
    actor_rep = _prepare_warm_start(init_actor_params, num_seeds)
    critic_rep = _prepare_warm_start(init_critic_params, num_seeds)

    if critic_rep is not None:
        def _train_for_seed(key, act_p, crt_p):
            return make_train(config, act_p, crt_p, report_progress=report_progress)(key)

        @jax.jit
        def train_multi(key: jax.Array):
            keys = jax.random.split(key, num_seeds)
            return jax.vmap(_train_for_seed, in_axes=(0, 0, 0))(keys, actor_rep, critic_rep)

        return train_multi

    def _train_for_seed(key, act_p):
        return make_train(config, act_p, None, report_progress=report_progress)(key)

    @jax.jit
    def train_multi(key: jax.Array):
        keys = jax.random.split(key, num_seeds)
        return jax.vmap(_train_for_seed, in_axes=(0, 0))(keys, actor_rep)

    return train_multi
