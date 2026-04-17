"""Tests for red_within_blue.training.stats"""

import numpy as np
import pytest

from red_within_blue.training.stats import (
    bonferroni_correct,
    chi_squared_vs_uniform,
    coverage_vs_baseline,
    evaluate_learning,
    learning_trend,
    welch_t_test,
)


# ---------------------------------------------------------------------------
# chi_squared_vs_uniform (legacy, kept for backwards compat)
# ---------------------------------------------------------------------------

def test_chi_squared_biased_actions_significant():
    """Heavily biased distribution should be detected as non-uniform (p < 0.05)."""
    actions = np.concatenate(
        [
            np.full(100, 0),
            np.full(100, 1),
            np.full(500, 2),
            np.full(100, 3),
            np.full(100, 4),
        ]
    )
    _, p = chi_squared_vs_uniform(actions, num_actions=5)
    assert p < 0.05, f"Expected p < 0.05 for biased actions, got p={p}"


def test_chi_squared_uniform_actions_not_significant():
    """Uniform distribution should NOT be detected as biased (p > 0.05)."""
    actions = np.concatenate([np.full(200, i) for i in range(5)])
    _, p = chi_squared_vs_uniform(actions, num_actions=5)
    assert p > 0.05, f"Expected p > 0.05 for uniform actions, got p={p}"


# ---------------------------------------------------------------------------
# welch_t_test
# ---------------------------------------------------------------------------

def test_welch_t_test_clearly_different_means():
    """Samples from very different distributions should yield p < 0.001."""
    rng = np.random.default_rng(42)
    sample_a = rng.normal(loc=0.9, scale=0.05, size=200)
    sample_b = rng.normal(loc=0.3, scale=0.10, size=200)
    _, p = welch_t_test(sample_a, sample_b)
    assert p < 0.001, f"Expected p < 0.001 for distinct populations, got p={p}"


# ---------------------------------------------------------------------------
# bonferroni_correct
# ---------------------------------------------------------------------------

def test_bonferroni_correct_values():
    """[0.01, 0.04, 0.06] corrected by n=3 should equal [0.03, 0.12, 0.18]."""
    raw = [0.01, 0.04, 0.06]
    corrected = bonferroni_correct(raw)
    expected = [0.03, 0.12, 0.18]
    assert len(corrected) == len(expected)
    for got, want in zip(corrected, expected):
        assert abs(got - want) < 1e-9, f"Expected {want}, got {got}"


def test_bonferroni_correct_capped_at_one():
    """Values that exceed 1.0 after correction must be capped at 1.0."""
    raw = [0.5, 0.8]
    corrected = bonferroni_correct(raw)
    for p in corrected:
        assert p <= 1.0, f"Corrected p-value {p} exceeds 1.0"


# ---------------------------------------------------------------------------
# coverage_vs_baseline
# ---------------------------------------------------------------------------

class TestCoverageVsBaseline:
    def test_clearly_better_policy_passes(self):
        """A policy with much higher coverage should PASS."""
        rng = np.random.default_rng(0)
        learned = rng.normal(loc=0.80, scale=0.05, size=20)
        baseline = rng.normal(loc=0.35, scale=0.10, size=20)
        result = coverage_vs_baseline(learned, baseline)
        assert result["p_value"] < 0.05
        assert result["cohens_d"] > 0.5
        assert "PASS" in result["verdict"]

    def test_identical_distributions_fail(self):
        """Same distribution should FAIL — no evidence of improvement."""
        rng = np.random.default_rng(1)
        a = rng.normal(loc=0.40, scale=0.10, size=20)
        b = rng.normal(loc=0.40, scale=0.10, size=20)
        result = coverage_vs_baseline(a, b)
        assert "PASS" not in result["verdict"]

    def test_slightly_better_gets_weak_or_marginal(self):
        """Small improvement should not get full PASS."""
        rng = np.random.default_rng(2)
        learned = rng.normal(loc=0.42, scale=0.10, size=30)
        baseline = rng.normal(loc=0.40, scale=0.10, size=30)
        result = coverage_vs_baseline(learned, baseline)
        # Should NOT be a strong PASS — effect size is tiny
        assert result["cohens_d"] < 0.5

    def test_worse_policy_fails(self):
        """Policy worse than baseline should definitely FAIL."""
        rng = np.random.default_rng(3)
        learned = rng.normal(loc=0.20, scale=0.05, size=20)
        baseline = rng.normal(loc=0.40, scale=0.10, size=20)
        result = coverage_vs_baseline(learned, baseline)
        assert "FAIL" in result["verdict"] or "INCONCLUSIVE" in result["verdict"]

    def test_returns_all_keys(self):
        result = coverage_vs_baseline(np.array([0.5, 0.6]), np.array([0.3, 0.4]))
        for key in ("learned_mean", "baseline_mean", "p_value", "cohens_d", "verdict"):
            assert key in result

    def test_effect_size_interpretation(self):
        """Cohen's d: 0.2=small, 0.5=medium, 0.8=large."""
        rng = np.random.default_rng(4)
        # Large effect
        learned = rng.normal(loc=0.80, scale=0.10, size=50)
        baseline = rng.normal(loc=0.30, scale=0.10, size=50)
        result = coverage_vs_baseline(learned, baseline)
        assert result["cohens_d"] > 2.0, "0.50 vs 0.30 with std=0.10 should be huge effect"


# ---------------------------------------------------------------------------
# learning_trend
# ---------------------------------------------------------------------------

class TestLearningTrend:
    def test_improving_curve_passes(self):
        """Steadily improving coverage should PASS."""
        rng = np.random.default_rng(0)
        # Linear improvement from 0.2 to 0.7 with noise
        t = np.linspace(0.2, 0.7, 200)
        curve = t + rng.normal(0, 0.03, size=200)
        result = learning_trend(curve)
        assert result["improvement"] > 0.3
        assert result["mann_whitney_p"] < 0.05
        assert "PASS" in result["verdict"]

    def test_flat_curve_fails(self):
        """Constant coverage should FAIL."""
        rng = np.random.default_rng(1)
        curve = 0.4 + rng.normal(0, 0.03, size=200)
        result = learning_trend(curve)
        assert abs(result["improvement"]) < 0.05
        assert "FAIL" in result["verdict"] or "INCONCLUSIVE" in result["verdict"]

    def test_declining_curve_fails(self):
        """Declining coverage should FAIL."""
        rng = np.random.default_rng(2)
        t = np.linspace(0.6, 0.2, 200)
        curve = t + rng.normal(0, 0.03, size=200)
        result = learning_trend(curve)
        assert "PASS" not in result["verdict"]

    def test_very_short_curve_skips(self):
        """< 6 episodes should return SKIP."""
        result = learning_trend(np.array([0.3, 0.4, 0.5]))
        assert "SKIP" in result["verdict"]

    def test_spearman_positive_for_upward_trend(self):
        """Upward trend should have positive Spearman rho."""
        curve = np.linspace(0.1, 0.8, 100)
        result = learning_trend(curve)
        assert result["spearman_rho"] > 0.8


# ---------------------------------------------------------------------------
# evaluate_learning (combined)
# ---------------------------------------------------------------------------

class TestEvaluateLearning:
    def test_good_policy_overall_passes(self):
        """Policy that clearly beats baseline and shows improvement -> PASS."""
        rng = np.random.default_rng(0)
        learned_cov = rng.normal(loc=0.75, scale=0.05, size=10)
        baseline_cov = rng.normal(loc=0.35, scale=0.10, size=20)
        curve = np.linspace(0.3, 0.8, 200) + rng.normal(0, 0.03, size=200)

        result = evaluate_learning(learned_cov, baseline_cov, curve, "TestPolicy")
        assert result["overall_pass"] is True
        assert "PASS" in result["summary"]

    def test_random_policy_fails(self):
        """Random vs random should FAIL overall (large N to avoid fluke)."""
        rng = np.random.default_rng(1)
        learned_cov = rng.normal(loc=0.35, scale=0.10, size=50)
        baseline_cov = rng.normal(loc=0.35, scale=0.10, size=50)

        result = evaluate_learning(learned_cov, baseline_cov, label="RandomVsRandom")
        assert result["overall_pass"] is False

    def test_summary_is_readable(self):
        """Summary should contain the key sections."""
        rng = np.random.default_rng(2)
        result = evaluate_learning(
            rng.normal(0.6, 0.05, 5),
            rng.normal(0.3, 0.10, 10),
            np.linspace(0.2, 0.6, 100),
        )
        assert "Coverage vs Baseline" in result["summary"]
        assert "Learning Trend" in result["summary"]
        assert "OVERALL" in result["summary"]
