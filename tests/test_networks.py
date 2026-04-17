"""Tests for the neural network modules in training/networks.py."""

import jax
import jax.numpy as jnp
import pytest

from red_within_blue.training.networks import Actor, Critic, QNetwork

OBS_DIM = 255
NUM_ACTIONS = 5
BATCH_SIZE = 32
RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(n: int = 2):
    keys = jax.random.split(RNG, n)
    return keys[0], keys[1]


# ---------------------------------------------------------------------------
# Actor tests
# ---------------------------------------------------------------------------

class TestActor:
    def test_output_shape_unbatched(self):
        """Actor maps (255,) -> (5,)."""
        init_key, obs_key = _split()
        obs = jax.random.normal(obs_key, (OBS_DIM,))
        actor = Actor(num_actions=NUM_ACTIONS)
        params = actor.init(init_key, obs)
        logits = actor.apply(params, obs)
        assert logits.shape == (NUM_ACTIONS,), f"Expected ({NUM_ACTIONS},), got {logits.shape}"

    def test_output_shape_batched_vmap(self):
        """vmap over Actor maps (32, 255) -> (32, 5)."""
        init_key, obs_key = _split()
        obs_batch = jax.random.normal(obs_key, (BATCH_SIZE, OBS_DIM))
        actor = Actor(num_actions=NUM_ACTIONS)
        # Initialise with a single observation
        params = actor.init(init_key, obs_batch[0])
        # vmap over the batch dimension
        batched_apply = jax.vmap(lambda o: actor.apply(params, o))
        logits_batch = batched_apply(obs_batch)
        assert logits_batch.shape == (BATCH_SIZE, NUM_ACTIONS), (
            f"Expected ({BATCH_SIZE}, {NUM_ACTIONS}), got {logits_batch.shape}"
        )


# ---------------------------------------------------------------------------
# Critic tests
# ---------------------------------------------------------------------------

class TestCritic:
    def test_output_shape_unbatched(self):
        """Critic maps (255,) -> scalar ()."""
        init_key, obs_key = _split()
        obs = jax.random.normal(obs_key, (OBS_DIM,))
        critic = Critic()
        params = critic.init(init_key, obs)
        value = critic.apply(params, obs)
        assert value.shape == (), f"Expected scalar (), got {value.shape}"


# ---------------------------------------------------------------------------
# QNetwork tests
# ---------------------------------------------------------------------------

class TestQNetwork:
    def test_output_shape_unbatched(self):
        """QNetwork maps (255,) -> (5,)."""
        init_key, obs_key = _split()
        obs = jax.random.normal(obs_key, (OBS_DIM,))
        qnet = QNetwork(num_actions=NUM_ACTIONS)
        params = qnet.init(init_key, obs)
        q_values = qnet.apply(params, obs)
        assert q_values.shape == (NUM_ACTIONS,), f"Expected ({NUM_ACTIONS},), got {q_values.shape}"


# ---------------------------------------------------------------------------
# External action masking test
# ---------------------------------------------------------------------------

class TestActionMasking:
    def test_masked_softmax_zeroes_invalid_actions(self):
        """External masking sets invalid action probabilities to ~0."""
        init_key, obs_key = _split()
        obs = jax.random.normal(obs_key, (OBS_DIM,))
        actor = Actor(num_actions=NUM_ACTIONS)
        params = actor.init(init_key, obs)
        logits = actor.apply(params, obs)

        # Allow only first 3 actions; mask out actions 3 and 4
        mask = jnp.array([True, True, True, False, False])
        masked_logits = logits + jnp.where(mask, 0.0, -1e9)
        probs = jax.nn.softmax(masked_logits)

        # Masked-out actions should be negligibly close to 0
        assert probs[3] < 1e-6, f"Action 3 prob should be ~0, got {probs[3]}"
        assert probs[4] < 1e-6, f"Action 4 prob should be ~0, got {probs[4]}"
        # Valid actions should sum to ~1
        valid_prob_sum = float(probs[:3].sum())
        assert abs(valid_prob_sum - 1.0) < 1e-5, (
            f"Valid action probs should sum to 1, got {valid_prob_sum}"
        )
