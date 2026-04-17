"""Weight transfer utilities and representation alignment metrics.

Provides helpers for warm-starting agents across curriculum stages and for
measuring how similar learned representations are via Centered Kernel
Alignment (CKA).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import Array


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

def transfer_actor_params(source_params):
    """Deep-copy a JAX param pytree for warm-starting the next stage.

    Args:
        source_params: A Flax/JAX parameter pytree (nested dict of arrays).

    Returns:
        An identical pytree with every leaf array copied — safe to mutate
        independently of the source.
    """
    return jax.tree.map(lambda x: x.copy(), source_params)


def init_fresh_critic(critic: nn.Module, input_dim: int, key: Array):
    """Initialise a new Critic with a potentially different input dimension.

    Args:
        critic:    A Flax nn.Module (e.g. ``Critic(hidden_dim=128)``).
        input_dim: Dimensionality of the observation vector fed to the critic.
        key:       JAX PRNGKey used for parameter initialisation.

    Returns:
        Fresh parameter pytree for ``critic`` on inputs of shape
        ``(input_dim,)``.
    """
    return critic.init(key, jnp.zeros(input_dim))


# ---------------------------------------------------------------------------
# Centered Kernel Alignment
# ---------------------------------------------------------------------------

def compute_cka(X: Array, Y: Array) -> Array:
    """Linear CKA between two representation matrices.

    Measures how similar two sets of learned representations are, regardless
    of invertible linear transformations.  A value of 1.0 means the
    representations are identical up to such a transformation; 0.0 means they
    are completely dissimilar.

    Args:
        X: ``[n_samples, n_features_x]`` array of activations.
        Y: ``[n_samples, n_features_y]`` array of activations.

    Returns:
        Scalar in ``[0, 1]``.
    """
    # Centre columns (mean-subtract each feature)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram-style products via X^T Y
    XtY = X.T @ Y          # [n_features_x, n_features_y]
    XtX = X.T @ X          # [n_features_x, n_features_x]
    YtY = Y.T @ Y          # [n_features_y, n_features_y]

    # Frobenius norms squared
    hsic_xy = jnp.sum(XtY ** 2)
    hsic_xx = jnp.sum(XtX ** 2)
    hsic_yy = jnp.sum(YtY ** 2)

    cka = hsic_xy / jnp.sqrt(hsic_xx * hsic_yy + 1e-10)
    return cka


# ---------------------------------------------------------------------------
# Hidden feature extraction
# ---------------------------------------------------------------------------

def extract_hidden_features(model: nn.Module, params, observations: Array) -> Array:
    """Extract activations after the second hidden layer of a 2-layer MLP.

    For networks following the pattern
        Dense_0 -> relu -> Dense_1 -> relu -> Dense_2 (output)
    this function returns the relu output of Dense_1, i.e. the representation
    just before the final projection.

    Note: This relies on Flax's ``@nn.compact`` auto-naming convention where
    sequential ``nn.Dense`` calls are named ``Dense_0``, ``Dense_1``, etc.
    If the Actor architecture changes (e.g. adding LayerNorm), the numbering
    shifts and this function must be updated to match.

    Args:
        model:        The Flax nn.Module (e.g. ``Actor``).
        params:       Full parameter pytree from ``model.init(...)``.
        observations: ``[N, obs_dim]`` array of observations.

    Returns:
        ``[N, hidden_dim]`` array of hidden activations.
    """
    # Extract per-layer params (coupled to Actor's @nn.compact layer order)
    dense0_params = params["params"]["Dense_0"]
    dense1_params = params["params"]["Dense_1"]
    # Infer each layer's output dim from its own kernel shape
    hidden_dim_0 = dense0_params["kernel"].shape[-1]
    hidden_dim_1 = dense1_params["kernel"].shape[-1]

    def _forward_one(obs):
        h = nn.Dense(hidden_dim_0).apply({"params": dense0_params}, obs)
        h = nn.relu(h)
        h = nn.Dense(hidden_dim_1).apply({"params": dense1_params}, h)
        h = nn.relu(h)
        return h

    return jax.vmap(_forward_one)(observations)
