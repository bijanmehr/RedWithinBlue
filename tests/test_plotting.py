"""Tests for the plotting module (apply_style, COLORS, plot_stage_summary)."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from red_within_blue.training.plotting import apply_style, COLORS, plot_stage_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(T: int = 20, H: int = 8, W: int = 8) -> dict:
    """Construct minimal valid data dict for plot_stage_summary."""
    rng = np.random.default_rng(0)
    episodes = np.arange(T, dtype=float)
    return {
        'coverage_mean':        rng.uniform(0.1, 0.9, T),
        'coverage_std':         rng.uniform(0.01, 0.05, T),
        'coverage_random_mean': rng.uniform(0.05, 0.5, T),
        'coverage_random_std':  rng.uniform(0.01, 0.05, T),
        'action_dist_learned':  np.array([0.1, 0.25, 0.25, 0.25, 0.15]),
        'action_dist_random':   np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        'visit_heatmap':        rng.integers(0, 50, (H, W)),
        'episodes':             episodes,
    }


# ---------------------------------------------------------------------------
# Test 1 — apply_style sets font.family='serif' and disables spines / grid
# ---------------------------------------------------------------------------

def test_apply_style_sets_serif_and_disables_spines_and_grid():
    apply_style()

    assert matplotlib.rcParams['font.family'] == ['serif'], (
        "font.family should be ['serif']"
    )
    assert matplotlib.rcParams['axes.spines.top'] is False, (
        "axes.spines.top should be False"
    )
    assert matplotlib.rcParams['axes.spines.right'] is False, (
        "axes.spines.right should be False"
    )
    assert matplotlib.rcParams['axes.grid'] is False, (
        "axes.grid should be False"
    )


# ---------------------------------------------------------------------------
# Test 2 — COLORS has required keys
# ---------------------------------------------------------------------------

def test_colors_has_required_keys():
    required = {'blue', 'red', 'gray', 'green'}
    missing = required - set(COLORS.keys())
    assert not missing, f"COLORS is missing keys: {missing}"

    # Values should look like hex colour strings
    for key in required:
        val = COLORS[key]
        assert isinstance(val, str), f"COLORS['{key}'] should be a string"
        assert val.startswith('#'), f"COLORS['{key}'] should start with '#'"


# ---------------------------------------------------------------------------
# Test 3 — plot_stage_summary creates .pdf and .png files
# ---------------------------------------------------------------------------

def test_plot_stage_summary_creates_pdf_and_png():
    data = _make_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, 'stage_summary')
        plot_stage_summary(data, base_path, stage_name='Test Stage')

        pdf_path = base_path + '.pdf'
        png_path = base_path + '.png'

        assert os.path.isfile(pdf_path), f".pdf file not found at {pdf_path}"
        assert os.path.isfile(png_path), f".png file not found at {png_path}"

        # Files should be non-empty
        assert os.path.getsize(pdf_path) > 0, ".pdf file is empty"
        assert os.path.getsize(png_path) > 0, ".png file is empty"
