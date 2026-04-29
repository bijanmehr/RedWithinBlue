"""Neural network modules for RedWithinBlue RL agents.

All networks are pure Flax nn.Module MLPs. Action masking is intentionally
external — networks output raw logits or Q-values and the caller applies masks.

Network dimensions are configurable via the experiment config YAML.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from jax import Array


_ACTIVATIONS: dict[str, Callable[[Array], Array]] = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "tanh": jnp.tanh,
    "silu": nn.silu,
}


def _resolve_activation(name: str) -> Callable[[Array], Array]:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {name!r}. Expected one of {sorted(_ACTIVATIONS)}."
        )
    return _ACTIVATIONS[name]


class Actor(nn.Module):
    """Policy network: obs -> action logits.

    Maps a flat observation vector to unnormalised logits over
    ``num_actions`` discrete actions.
    """

    num_actions: int = 5
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        act = _resolve_activation(self.activation)
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = act(x)
        logits = nn.Dense(self.num_actions)(x)
        return logits


class Critic(nn.Module):
    """Value network: obs -> scalar V(s).

    Maps a flat observation vector to a single scalar state-value estimate.
    """

    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        act = _resolve_activation(self.activation)
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = act(x)
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


class JointRedActor(nn.Module):
    """Centralized joint policy for the red team (POMDP).

    Consumes the concatenated observations of all ``num_red`` red agents and
    outputs factorized action logits of shape ``[num_red, num_actions]``. One
    network, one gradient — the red team shares a single central controller.

    Inputs are flat arrays of length ``num_red * obs_dim``; outputs are
    reshaped from a single Dense head of size ``num_red * num_actions``.
    """

    num_red: int
    num_actions: int = 5
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"

    @nn.compact
    def __call__(self, joint_obs: Array) -> Array:
        act = _resolve_activation(self.activation)
        x = joint_obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = act(x)
        flat_logits = nn.Dense(self.num_red * self.num_actions)(x)
        return flat_logits.reshape(self.num_red, self.num_actions)


class QNetwork(nn.Module):
    """Action-value network: obs -> Q-values for each action.

    Maps a flat observation vector to per-action Q-values.
    """

    num_actions: int = 5
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        act = _resolve_activation(self.activation)
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = act(x)
        q_values = nn.Dense(self.num_actions)(x)
        return q_values
