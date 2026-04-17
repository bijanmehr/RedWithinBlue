"""Policy gradient loss functions for RedWithinBlue.

Three progressive layers:
  Layer 1 — plain REINFORCE (pg_loss)
  Layer 2 — REINFORCE with mean-return baseline (pg_loss_with_baseline)
  Layer 3 — actor-critic with TD(0) advantages (actor_critic_loss)

All functions are pure JAX; no side effects, no in-place mutation.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn


# ---------------------------------------------------------------------------
# 1. Discounted returns
# ---------------------------------------------------------------------------

def compute_discounted_returns(
    rewards: Array,  # [T] float
    dones: Array,    # [T] bool
    gamma: float,
) -> Array:
    """Compute discounted returns G_t = r_t + gamma * G_{t+1} * (1 - done_t).

    Iterates backwards from T-1 to 0.  When done_t is True the episode has
    ended at step t, so the future return from t+1 is zeroed out.

    Args:
        rewards: shape [T], per-step rewards.
        dones:   shape [T], episode-termination flags (True = episode ended).
        gamma:   discount factor in [0, 1].

    Returns:
        returns: shape [T], discounted returns.
    """
    T = rewards.shape[0]

    def _step(carry, t):
        g_next = carry
        r_t = rewards[t]
        d_t = dones[t].astype(jnp.float32)
        g_t = r_t + gamma * g_next * (1.0 - d_t)
        return g_t, g_t

    # Scan from T-1 down to 0.
    _, returns_reversed = jax.lax.scan(
        _step,
        init=jnp.float32(0.0),
        xs=jnp.arange(T - 1, -1, -1),
    )
    # scan produces results in the order the xs were visited (T-1, T-2, ..., 0)
    # so we reverse to recover chronological order.
    return returns_reversed[::-1]


# ---------------------------------------------------------------------------
# 2. Plain REINFORCE loss
# ---------------------------------------------------------------------------

def pg_loss(
    logits: Array,   # [T, 5]
    actions: Array,  # [T] int
    returns: Array,  # [T] float
) -> Array:
    """REINFORCE policy gradient loss (no baseline).

    loss = -mean( log pi(a_t | s_t) * G_t )

    Args:
        logits:  shape [T, num_actions], unnormalised action scores.
        actions: shape [T], indices of actions taken.
        returns: shape [T], discounted returns.

    Returns:
        Scalar loss value.
    """
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)  # [T, num_actions]
    T = actions.shape[0]
    log_probs = log_probs_all[jnp.arange(T), actions]    # [T]
    return -jnp.mean(log_probs * returns)


# ---------------------------------------------------------------------------
# 3. REINFORCE with mean-return baseline
# ---------------------------------------------------------------------------

def pg_loss_with_baseline(
    logits: Array,   # [T, 5]
    actions: Array,  # [T] int
    returns: Array,  # [T] float
) -> Array:
    """REINFORCE policy gradient loss with a simple mean-return baseline.

    loss = -mean( log pi(a_t | s_t) * (G_t - mean(G)) )

    The baseline reduces variance without introducing bias.

    Args:
        logits:  shape [T, num_actions], unnormalised action scores.
        actions: shape [T], indices of actions taken.
        returns: shape [T], discounted returns.

    Returns:
        Scalar loss value.
    """
    baseline = jnp.mean(returns)
    advantages = returns - baseline
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)  # [T, num_actions]
    T = actions.shape[0]
    log_probs = log_probs_all[jnp.arange(T), actions]    # [T]
    return -jnp.mean(log_probs * advantages)


# ---------------------------------------------------------------------------
# 4. Actor-critic loss with TD(0) advantages
# ---------------------------------------------------------------------------

def actor_critic_loss(
    actor: nn.Module,
    critic: nn.Module,
    actor_params,
    critic_params,
    observations: Array,  # [T, obs_dim]
    actions: Array,       # [T] int
    rewards: Array,       # [T] float
    dones: Array,         # [T] bool
    gamma: float,
) -> Tuple[Array, Array]:
    """Actor-critic loss using one-step TD(0) advantages.

    advantage_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

    policy_loss = -mean( log pi(a_t | s_t) * stop_gradient(advantage_t) )
    value_loss  = mean( (V(s_t) - stop_gradient(td_target_t))^2 )

    V(s_{T+1}) is approximated as 0 (no bootstrap past end of trajectory).

    Args:
        actor:          Flax Actor module.
        critic:         Flax Critic module.
        actor_params:   Parameters for actor.
        critic_params:  Parameters for critic.
        observations:   shape [T, obs_dim].
        actions:        shape [T] int.
        rewards:        shape [T] float.
        dones:          shape [T] bool.
        gamma:          Discount factor.

    Returns:
        (policy_loss, value_loss): both scalars.
    """
    T = observations.shape[0]

    # --- Forward passes (batched via vmap) ---
    batch_actor  = jax.vmap(lambda o: actor.apply(actor_params, o))
    batch_critic = jax.vmap(lambda o: critic.apply(critic_params, o))

    logits = batch_actor(observations)   # [T, num_actions]
    values = batch_critic(observations)  # [T]

    # --- Bootstrap targets ---
    # Shift values by one; pad with 0 for V(s_{T+1}).
    next_values = jnp.concatenate([values[1:], jnp.zeros(1)], axis=0)  # [T]

    d = dones.astype(jnp.float32)
    td_targets  = rewards + gamma * next_values * (1.0 - d)             # [T]
    advantages  = td_targets - values                                    # [T]

    # --- Policy loss ---
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)                 # [T, A]
    log_probs     = log_probs_all[jnp.arange(T), actions]               # [T]
    policy_loss   = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages))

    # --- Value loss ---
    value_loss = jnp.mean((values - jax.lax.stop_gradient(td_targets)) ** 2)

    return policy_loss, value_loss
