"""Tests for policy gradient training utilities in training/pg.py."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.training.losses import (
    compute_discounted_returns,
    pg_loss,
    pg_loss_with_baseline,
    actor_critic_loss,
)
from red_within_blue.training.networks import Actor, Critic

RNG = jax.random.PRNGKey(42)
OBS_DIM = 376
NUM_ACTIONS = 5
T = 10  # trajectory length used in most tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory(t: int = T, obs_dim: int = OBS_DIM, seed: int = 0):
    """Generate a random trajectory of length t."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    observations = jax.random.normal(k1, (t, obs_dim))
    actions      = jax.random.randint(k2, (t,), 0, NUM_ACTIONS)
    rewards      = jax.random.normal(k3, (t,))
    dones        = jnp.zeros(t, dtype=bool)
    returns      = jax.random.normal(k4, (t,))
    logits       = jax.random.normal(jax.random.PRNGKey(seed + 1), (t, NUM_ACTIONS))
    return observations, actions, rewards, dones, returns, logits


# ---------------------------------------------------------------------------
# 1. compute_discounted_returns — correctness
# ---------------------------------------------------------------------------

class TestComputeDiscountedReturns:

    def test_simple_three_step(self):
        """[1, 1, 1] rewards with gamma=0.99 -> G_0 ≈ 2.9701."""
        rewards = jnp.array([1.0, 1.0, 1.0])
        dones   = jnp.array([False, False, False])
        gamma   = 0.99

        returns = compute_discounted_returns(rewards, dones, gamma)

        # G_2 = 1
        # G_1 = 1 + 0.99 * 1       = 1.99
        # G_0 = 1 + 0.99 * 1.99    = 2.9701
        assert returns.shape == (3,), f"Expected shape (3,), got {returns.shape}"
        assert jnp.isclose(returns[0], 2.9701, atol=1e-4), (
            f"G_0 should be ~2.9701, got {returns[0]}"
        )
        assert jnp.isclose(returns[1], 1.99, atol=1e-4), (
            f"G_1 should be ~1.99, got {returns[1]}"
        )
        assert jnp.isclose(returns[2], 1.0, atol=1e-4), (
            f"G_2 should be 1.0, got {returns[2]}"
        )

    def test_done_boundary_resets_future(self):
        """Returns must not propagate past a done=True step."""
        # Episode ends at t=1; t=2 starts a new episode.
        rewards = jnp.array([1.0, 1.0, 5.0])
        dones   = jnp.array([False, True, False])
        gamma   = 0.99

        returns = compute_discounted_returns(rewards, dones, gamma)

        # G_2 = 5  (no future)
        # G_1 = 1 + 0.99 * 5 * (1 - 1) = 1  (done resets bootstrap)
        # G_0 = 1 + 0.99 * 1 * (1 - 0) = 1.99
        assert jnp.isclose(returns[2], 5.0, atol=1e-4), (
            f"G_2 should be 5.0, got {returns[2]}"
        )
        assert jnp.isclose(returns[1], 1.0, atol=1e-4), (
            f"G_1 should be 1.0 (done blocks future), got {returns[1]}"
        )
        assert jnp.isclose(returns[0], 1.99, atol=1e-4), (
            f"G_0 should be 1.99, got {returns[0]}"
        )

    def test_output_shape(self):
        rewards = jnp.ones(T)
        dones   = jnp.zeros(T, dtype=bool)
        returns = compute_discounted_returns(rewards, dones, 0.99)
        assert returns.shape == (T,)


# ---------------------------------------------------------------------------
# 2. pg_loss — finite scalar
# ---------------------------------------------------------------------------

class TestPGLoss:

    def test_returns_finite_scalar(self):
        _, actions, _, _, returns, logits = _make_trajectory()
        loss = pg_loss(logits, actions, returns)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"

    def test_gradient_flows(self):
        """Loss gradient w.r.t. logits should be non-zero."""
        _, actions, _, _, returns, logits = _make_trajectory()

        def _loss(l):
            return pg_loss(l, actions, returns)

        grad = jax.grad(_loss)(logits)
        assert jnp.any(grad != 0), "Gradient is all zeros — something is wrong"


# ---------------------------------------------------------------------------
# 3. pg_loss_with_baseline — finite scalar
# ---------------------------------------------------------------------------

class TestPGLossWithBaseline:

    def test_returns_finite_scalar(self):
        _, actions, _, _, returns, logits = _make_trajectory()
        loss = pg_loss_with_baseline(logits, actions, returns)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"

    def test_constant_returns_gives_zero_loss(self):
        """When all returns are equal the baseline cancels them -> loss == 0."""
        _, actions, _, _, _, logits = _make_trajectory()
        returns = jnp.ones(T)  # constant -> advantage = 0 everywhere
        loss = pg_loss_with_baseline(logits, actions, returns)
        assert jnp.isclose(loss, 0.0, atol=1e-6), (
            f"Expected loss ≈ 0 for constant returns, got {loss}"
        )


# ---------------------------------------------------------------------------
# 4. actor_critic_loss — (policy_loss, value_loss) both finite scalars
# ---------------------------------------------------------------------------

class TestActorCriticLoss:

    @pytest.fixture(scope="class")
    def model_and_params(self):
        actor  = Actor(num_actions=NUM_ACTIONS, hidden_dim=64)
        critic = Critic(hidden_dim=64)

        obs_dim = OBS_DIM
        dummy_obs = jnp.zeros(obs_dim)

        k1, k2 = jax.random.split(RNG, 2)
        actor_params  = actor.init(k1, dummy_obs)
        critic_params = critic.init(k2, dummy_obs)

        return actor, critic, actor_params, critic_params

    def test_returns_two_finite_scalars(self, model_and_params):
        actor, critic, actor_params, critic_params = model_and_params
        observations, actions, rewards, dones, _, _ = _make_trajectory()

        policy_loss, value_loss, entropy = actor_critic_loss(
            actor, critic, actor_params, critic_params,
            observations, actions, rewards, dones, gamma=0.99,
        )

        assert policy_loss.shape == (), f"policy_loss should be scalar, got {policy_loss.shape}"
        assert value_loss.shape  == (), f"value_loss should be scalar, got {value_loss.shape}"
        assert entropy.shape == (), f"entropy should be scalar, got {entropy.shape}"
        assert jnp.isfinite(policy_loss), f"policy_loss is not finite: {policy_loss}"
        assert jnp.isfinite(value_loss),  f"value_loss is not finite: {value_loss}"
        assert entropy >= 0.0, f"entropy should be non-negative, got {entropy}"

    def test_value_loss_nonnegative(self, model_and_params):
        """MSE value loss must be >= 0."""
        actor, critic, actor_params, critic_params = model_and_params
        observations, actions, rewards, dones, _, _ = _make_trajectory()

        _, value_loss, _ = actor_critic_loss(
            actor, critic, actor_params, critic_params,
            observations, actions, rewards, dones, gamma=0.99,
        )
        assert value_loss >= 0.0, f"value_loss should be non-negative, got {value_loss}"

    def test_gradient_flows_actor(self, model_and_params):
        """Policy gradient should flow back through actor parameters."""
        actor, critic, actor_params, critic_params = model_and_params
        observations, actions, rewards, dones, _, _ = _make_trajectory()

        def _policy_loss(ap):
            pl, _, _ = actor_critic_loss(
                actor, critic, ap, critic_params,
                observations, actions, rewards, dones, gamma=0.99,
            )
            return pl

        grads = jax.grad(_policy_loss)(actor_params)
        leaves = jax.tree_util.tree_leaves(grads)
        any_nonzero = any(jnp.any(g != 0) for g in leaves)
        assert any_nonzero, "Actor gradient is all zeros"

    def test_gradient_flows_critic(self, model_and_params):
        """Value gradient should flow back through critic parameters."""
        actor, critic, actor_params, critic_params = model_and_params
        observations, actions, rewards, dones, _, _ = _make_trajectory()

        def _value_loss(cp):
            _, vl, _ = actor_critic_loss(
                actor, critic, actor_params, cp,
                observations, actions, rewards, dones, gamma=0.99,
            )
            return vl

        grads = jax.grad(_value_loss)(critic_params)
        leaves = jax.tree_util.tree_leaves(grads)
        any_nonzero = any(jnp.any(g != 0) for g in leaves)
        assert any_nonzero, "Critic gradient is all zeros"
