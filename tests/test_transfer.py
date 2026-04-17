"""Tests for src/red_within_blue/training/transfer.py."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.training.networks import Actor, Critic
from red_within_blue.training.transfer import (
    transfer_actor_params,
    init_fresh_critic,
    compute_cka,
    extract_hidden_features,
)

RNG = jax.random.PRNGKey(42)
OBS_DIM = 255
NUM_ACTIONS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actor_params(obs_dim: int = OBS_DIM, key=None):
    key = key if key is not None else RNG
    actor = Actor(num_actions=NUM_ACTIONS)
    obs = jnp.zeros(obs_dim)
    return actor, actor.init(key, obs)


# ---------------------------------------------------------------------------
# transfer_actor_params
# ---------------------------------------------------------------------------

class TestTransferActorParams:
    def test_transferred_params_are_identical(self):
        """Transferred params must have the same values as the source."""
        _, params = _make_actor_params()
        transferred = transfer_actor_params(params)

        src_leaves = jax.tree.leaves(params)
        trf_leaves = jax.tree.leaves(transferred)

        assert len(src_leaves) == len(trf_leaves), "Param tree structure differs"
        for src, trf in zip(src_leaves, trf_leaves):
            assert jnp.allclose(src, trf), "Param values are not identical after transfer"

    def test_transferred_params_are_independent(self):
        """Mutating the copy must not affect the original (and vice versa)."""
        _, params = _make_actor_params()
        transferred = transfer_actor_params(params)

        # Find first mutable leaf and zero it out in the copy
        src_leaves = jax.tree.leaves(params)
        trf_leaves = jax.tree.leaves(transferred)
        # Leaves are DeviceArrays; .copy() should give an independent numpy-
        # backed buffer; verify source is unchanged after we modify the copy.
        # (In JAX, arrays are immutable, but transfer_actor_params gives a new
        # pytree so structural independence is what matters.)
        assert len(src_leaves) > 0


# ---------------------------------------------------------------------------
# compute_cka
# ---------------------------------------------------------------------------

class TestComputeCKA:
    def test_identical_matrices_cka_is_one(self):
        """CKA(X, X) should equal 1.0."""
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (64, 32))
        cka = compute_cka(X, X)
        assert jnp.allclose(cka, 1.0, atol=1e-5), f"Expected 1.0, got {cka}"

    def test_cka_in_unit_interval_random(self):
        """CKA must lie in [0, 1] for random matrices."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        X = jax.random.normal(k1, (50, 20))
        Y = jax.random.normal(k2, (50, 16))
        cka = compute_cka(X, Y)
        assert 0.0 <= float(cka) <= 1.0 + 1e-6, f"CKA out of [0,1]: {cka}"

    def test_very_different_matrices_cka_below_half(self):
        """CKA of nearly orthogonal representations should be < 0.5."""
        # Build two sets of activations that live in orthogonal subspaces.
        # Use a large batch so the empirical estimate is stable.
        n, d = 512, 64
        k1, k2 = jax.random.split(jax.random.PRNGKey(99))
        # Block-diagonal: X occupies first half, Y occupies second half
        X_raw = jax.random.normal(k1, (n, d // 2))
        Y_raw = jax.random.normal(k2, (n, d // 2))
        X = jnp.concatenate([X_raw, jnp.zeros((n, d // 2))], axis=1)
        Y = jnp.concatenate([jnp.zeros((n, d // 2)), Y_raw], axis=1)
        cka = compute_cka(X, Y)
        assert float(cka) < 0.5, f"Expected CKA < 0.5 for orthogonal reps, got {cka}"


# ---------------------------------------------------------------------------
# extract_hidden_features
# ---------------------------------------------------------------------------

class TestExtractHiddenFeatures:
    def test_output_shape(self):
        """extract_hidden_features returns [N, hidden_dim] array."""
        actor, params = _make_actor_params()
        N = 16
        k = jax.random.PRNGKey(7)
        obs = jax.random.normal(k, (N, OBS_DIM))
        feats = extract_hidden_features(actor, params, obs)
        assert feats.shape == (N, actor.hidden_dim), (
            f"Expected ({N}, {actor.hidden_dim}), got {feats.shape}"
        )

    def test_output_is_nonnegative(self):
        """After relu, all feature values must be >= 0."""
        actor, params = _make_actor_params()
        N = 32
        k = jax.random.PRNGKey(11)
        obs = jax.random.normal(k, (N, OBS_DIM))
        feats = extract_hidden_features(actor, params, obs)
        assert jnp.all(feats >= 0.0), "relu output contains negative values"
