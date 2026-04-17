"""Neural network modules for RedWithinBlue RL agents.

All networks are pure Flax nn.Module MLPs. Action masking is intentionally
external — networks output raw logits or Q-values and the caller applies masks.

Network dimensions are configurable via the experiment config YAML.
"""

import flax.linen as nn
import jax.numpy as jnp
from jax import Array


class Actor(nn.Module):
    """Policy network: obs -> action logits.

    Maps a flat observation vector to unnormalised logits over
    ``num_actions`` discrete actions.
    """

    num_actions: int = 5
    hidden_dim: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        logits = nn.Dense(self.num_actions)(x)
        return logits


class Critic(nn.Module):
    """Value network: obs -> scalar V(s).

    Maps a flat observation vector to a single scalar state-value estimate.
    """

    hidden_dim: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


class QNetwork(nn.Module):
    """Action-value network: obs -> Q-values for each action.

    Maps a flat observation vector to per-action Q-values.
    """

    num_actions: int = 5
    hidden_dim: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        q_values = nn.Dense(self.num_actions)(x)
        return q_values
