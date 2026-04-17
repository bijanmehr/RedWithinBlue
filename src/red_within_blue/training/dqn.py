"""DQN training utilities for RedWithinBlue.

Provides pure-JAX functions for tabular Q-learning and deep Q-network (DQN)
training steps. All functions are designed to be jit-compilable.
"""

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn


def tabular_q_update(
    q_table: Array,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    alpha: float,
    gamma: float,
    done: bool,
) -> Array:
    """Perform a single tabular Q-learning update.

    Args:
        q_table: Q-value table of shape [S, A].
        state: Current state index.
        action: Action taken.
        reward: Reward received.
        next_state: Resulting next state index.
        alpha: Learning rate.
        gamma: Discount factor.
        done: Whether the episode ended at this transition.

    Returns:
        Updated q_table with the same shape [S, A].
    """
    max_next_q = jnp.max(q_table[next_state])
    td_target = reward + gamma * max_next_q * (1.0 - done)
    td_error = td_target - q_table[state, action]
    new_value = q_table[state, action] + alpha * td_error
    return q_table.at[state, action].set(new_value)


def dqn_loss(
    model: nn.Module,
    params: dict,
    observations: Array,
    actions: Array,
    targets: Array,
) -> Array:
    """Compute the DQN MSE loss for a batch of transitions.

    Args:
        model: Flax QNetwork module.
        params: Model parameters.
        observations: Batch of observations, shape [B, 255].
        actions: Batch of integer actions taken, shape [B].
        targets: Batch of TD targets, shape [B].

    Returns:
        Scalar MSE loss.
    """
    # Batched forward pass: [B, A]
    q_values_batch = jax.vmap(model.apply, in_axes=(None, 0))(params, observations)
    # Select Q-values for the actions taken: [B]
    q_taken = q_values_batch[jnp.arange(q_values_batch.shape[0]), actions]
    loss = jnp.mean((q_taken - targets) ** 2)
    return loss


def compute_dqn_targets(
    model: nn.Module,
    target_params: dict,
    rewards: Array,
    next_observations: Array,
    dones: Array,
    gamma: float,
) -> Array:
    """Compute DQN TD targets using the target network.

    Args:
        model: Flax QNetwork module.
        target_params: Parameters of the frozen target network.
        rewards: Batch of rewards, shape [B].
        next_observations: Batch of next observations, shape [B, 255].
        dones: Batch of done flags (float or bool), shape [B].
        gamma: Discount factor.

    Returns:
        TD targets of shape [B].
    """
    # Batched forward pass through target network: [B, A]
    next_q_values = jax.vmap(model.apply, in_axes=(None, 0))(
        target_params, next_observations
    )
    max_next_q = jnp.max(next_q_values, axis=-1)  # [B]
    targets = rewards + gamma * max_next_q * (1.0 - dones)
    return targets


def epsilon_greedy(
    key: Array,
    q_values: Array,
    epsilon: float,
) -> Array:
    """Select an action using epsilon-greedy exploration.

    Args:
        key: JAX PRNG key.
        q_values: Q-values for each action, shape [A].
        epsilon: Exploration probability in [0, 1].

    Returns:
        Selected action index as a scalar integer array.
    """
    num_actions = q_values.shape[0]
    key_explore, key_action = jax.random.split(key)
    greedy_action = jnp.argmax(q_values)
    random_action = jax.random.randint(key_action, shape=(), minval=0, maxval=num_actions)
    explore = jax.random.uniform(key_explore) < epsilon
    action = jnp.where(explore, random_action, greedy_action)
    return action
