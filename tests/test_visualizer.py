"""Tests for the visualiser module (render_frame and EnvDashboard)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import jax
import pytest

from red_within_blue.env import GridCommEnv
from red_within_blue.types import EnvConfig
from red_within_blue.visualizer import render_frame, EnvDashboard


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _small_config(**overrides):
    defaults = dict(
        grid_width=8, grid_height=8, num_agents=2, max_steps=10,
        obs_radius=2, comm_radius=5.0, wall_density=0.0,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_state():
    """Create a small env, reset it, and return (state, config)."""
    cfg = _small_config()
    env = GridCommEnv(cfg)
    key = jax.random.PRNGKey(42)
    _obs, state = env.reset(key)
    return state, cfg


# ------------------------------------------------------------------
# 43. render_frame returns correct shape and dtype
# ------------------------------------------------------------------

def test_render_frame_shape():
    state, cfg = _make_state()
    img = render_frame(state, cfg)

    assert isinstance(img, np.ndarray), "render_frame must return a numpy array"
    assert img.ndim == 3, f"Expected 3-D array, got {img.ndim}-D"
    assert img.shape[2] == 3, f"Last dimension must be 3 (RGB), got {img.shape[2]}"
    assert img.dtype == np.uint8, f"Expected uint8 dtype, got {img.dtype}"
    # Sanity: image should be non-trivially sized
    assert img.shape[0] > 0 and img.shape[1] > 0


# ------------------------------------------------------------------
# 44. render_frame is deterministic
# ------------------------------------------------------------------

def test_render_frame_deterministic():
    state, cfg = _make_state()
    img1 = render_frame(state, cfg)
    img2 = render_frame(state, cfg)

    np.testing.assert_array_equal(img1, img2, err_msg="Same state must produce identical images")


# ------------------------------------------------------------------
# 45. EnvDashboard initialises without error
# ------------------------------------------------------------------

def test_dashboard_init():
    cfg = _small_config()
    dashboard = EnvDashboard(cfg)
    try:
        assert dashboard.fig is not None
        assert dashboard.ax_grid is not None
        assert dashboard.ax_metrics is not None
    finally:
        dashboard.close()


# ------------------------------------------------------------------
# 46. dashboard.update(state) runs without error
# ------------------------------------------------------------------

def test_dashboard_update():
    state, cfg = _make_state()
    dashboard = EnvDashboard(cfg)
    try:
        # Should not raise
        dashboard.update(state)
    finally:
        dashboard.close()
