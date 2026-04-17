"""
Statistical tests for post-hoc analysis of training runs.

The key question is NOT "is the action distribution non-uniform" (chi-squared
will say yes for any large sample), but:

1. Does the learned policy explore better than random? (coverage_vs_baseline)
2. Did coverage improve during training?            (learning_trend)
3. Is the improvement practically meaningful?        (effect_size)

All functions operate on numpy arrays and use scipy — not JAX.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Low-level tests (building blocks)
# ---------------------------------------------------------------------------


def chi_squared_vs_uniform(
    actions: np.ndarray, num_actions: int = 5
) -> Tuple[float, float]:
    """Test whether an action distribution differs from uniform.

    WARNING: with large N (>1000 actions) this test will almost always
    reject the null.  Use ``evaluate_learning`` for a meaningful verdict.

    Returns (chi2_statistic, p_value).
    """
    observed = np.bincount(actions.astype(int), minlength=num_actions)[:num_actions]
    expected_per_bin = len(actions) / num_actions
    expected = np.full(num_actions, expected_per_bin, dtype=float)
    result = stats.chisquare(f_obs=observed, f_exp=expected)
    return float(result.statistic), float(result.pvalue)


def welch_t_test(
    sample_a: np.ndarray, sample_b: np.ndarray
) -> Tuple[float, float]:
    """Welch's two-sample t-test (unequal variances).

    Returns (t_statistic, p_value).
    """
    result = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def bonferroni_correct(p_values: List[float]) -> List[float]:
    """Apply Bonferroni correction to a list of p-values."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


# ---------------------------------------------------------------------------
# Coverage vs baseline
# ---------------------------------------------------------------------------


def coverage_vs_baseline(
    learned_coverages: np.ndarray,
    baseline_coverages: np.ndarray,
) -> Dict[str, float]:
    """Test whether learned policy coverage exceeds a baseline.

    Uses a one-sided Mann-Whitney U test (non-parametric, no normality
    assumption) and reports Cohen's d effect size.

    Parameters
    ----------
    learned_coverages : (K,) array of per-seed final coverage values.
    baseline_coverages : (M,) array of per-seed baseline coverage values.

    Returns
    -------
    dict with keys:
        learned_mean, learned_std, baseline_mean, baseline_std,
        mann_whitney_U, p_value (one-sided: learned > baseline),
        cohens_d, verdict (str).
    """
    learned = np.asarray(learned_coverages, dtype=np.float64)
    baseline = np.asarray(baseline_coverages, dtype=np.float64)

    l_mean = float(np.mean(learned))
    l_std = float(np.std(learned, ddof=1)) if len(learned) > 1 else 0.0
    b_mean = float(np.mean(baseline))
    b_std = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0

    # Mann-Whitney U (one-sided: learned > baseline)
    if len(learned) >= 2 and len(baseline) >= 2:
        u_stat, p_two = stats.mannwhitneyu(learned, baseline, alternative="greater")
        p_value = float(p_two)
        u_value = float(u_stat)
    else:
        u_value = 0.0
        p_value = 1.0

    # Cohen's d (pooled std)
    pooled_std = np.sqrt(
        ((len(learned) - 1) * l_std**2 + (len(baseline) - 1) * b_std**2)
        / max(len(learned) + len(baseline) - 2, 1)
    )
    cohens_d = (l_mean - b_mean) / pooled_std if pooled_std > 1e-12 else 0.0

    # Verdict
    if p_value < 0.05 and cohens_d > 0.5:
        verdict = "PASS: policy significantly outperforms baseline (p<0.05, d>0.5)"
    elif p_value < 0.05 and cohens_d > 0.2:
        verdict = "WEAK PASS: statistically significant but small effect (p<0.05, d<0.5)"
    elif p_value < 0.05:
        verdict = "MARGINAL: significant but negligible effect size (d<0.2)"
    elif l_mean > b_mean:
        verdict = "INCONCLUSIVE: higher mean but not statistically significant"
    else:
        verdict = "FAIL: no evidence policy outperforms baseline"

    return {
        "learned_mean": l_mean,
        "learned_std": l_std,
        "baseline_mean": b_mean,
        "baseline_std": b_std,
        "mann_whitney_U": u_value,
        "p_value": p_value,
        "cohens_d": float(cohens_d),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Learning trend
# ---------------------------------------------------------------------------


def learning_trend(
    coverage_curve: np.ndarray,
    window: int = 50,
) -> Dict[str, float]:
    """Test whether coverage improved during training.

    Compares the first ``window`` episodes against the last ``window``
    episodes using a one-sided Mann-Whitney U test and reports the
    absolute improvement.

    Also computes Spearman rank correlation on the smoothed curve to
    detect monotonic trends.

    Parameters
    ----------
    coverage_curve : (T,) array of per-episode coverage values.
    window : int
        Number of episodes to compare at the start vs end.

    Returns
    -------
    dict with keys:
        early_mean, late_mean, improvement,
        mann_whitney_p (one-sided: late > early),
        spearman_rho, spearman_p,
        verdict (str).
    """
    curve = np.asarray(coverage_curve, dtype=np.float64)
    T = len(curve)
    w = min(window, T // 3)  # at least 3 windows must fit

    if T < 6:
        return {
            "early_mean": float(np.mean(curve)),
            "late_mean": float(np.mean(curve)),
            "improvement": 0.0,
            "mann_whitney_p": 1.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "verdict": "SKIP: too few episodes to test",
        }

    early = curve[:w]
    late = curve[-w:]

    e_mean = float(np.mean(early))
    l_mean = float(np.mean(late))
    improvement = l_mean - e_mean

    # Mann-Whitney: late > early
    _, mw_p = stats.mannwhitneyu(late, early, alternative="greater")

    # Spearman on smoothed curve (rolling mean to reduce noise)
    smooth_w = max(1, T // 20)
    kernel = np.ones(smooth_w) / smooth_w
    smoothed = np.convolve(curve, kernel, mode="valid")
    rho, sp_p = stats.spearmanr(np.arange(len(smoothed)), smoothed)

    # Verdict
    if mw_p < 0.05 and improvement > 0.05:
        verdict = "PASS: coverage improved significantly during training"
    elif mw_p < 0.05 and improvement > 0.02:
        verdict = "WEAK PASS: small but significant improvement"
    elif improvement > 0.05 and rho > 0.3:
        verdict = "TREND: upward trend visible but not yet significant (need more episodes)"
    elif improvement <= 0.01:
        verdict = "FAIL: no meaningful improvement during training"
    else:
        verdict = "INCONCLUSIVE: some improvement but not statistically significant"

    return {
        "early_mean": e_mean,
        "late_mean": l_mean,
        "improvement": improvement,
        "mann_whitney_p": float(mw_p),
        "spearman_rho": float(rho),
        "spearman_p": float(sp_p),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate_learning(
    learned_coverages: np.ndarray,
    baseline_coverages: np.ndarray,
    coverage_curve: Optional[np.ndarray] = None,
    label: str = "Policy",
) -> Dict[str, any]:
    """Run all learning evaluation tests and return a combined verdict.

    This is the function that scripts should call instead of
    ``chi_squared_vs_uniform``.

    Parameters
    ----------
    learned_coverages : (K,) per-seed final coverage from the learned policy.
    baseline_coverages : (M,) per-seed coverage from a random/greedy baseline.
    coverage_curve : (T,) optional per-episode coverage during training.
    label : str for display.

    Returns
    -------
    dict with:
        baseline_test  — full coverage_vs_baseline result
        trend_test     — full learning_trend result (if curve provided)
        overall_pass   — bool, True only if baseline_test passes
        summary        — human-readable multi-line string
    """
    bt = coverage_vs_baseline(learned_coverages, baseline_coverages)

    tt = None
    if coverage_curve is not None and len(coverage_curve) >= 6:
        tt = learning_trend(coverage_curve)

    # Overall: must beat baseline with meaningful effect size
    overall = bt["p_value"] < 0.05 and bt["cohens_d"] > 0.2

    # Build summary
    lines = [
        f"  === LEARNING EVALUATION: {label} ===",
        f"",
        f"  1. Coverage vs Baseline",
        f"     Learned : {bt['learned_mean']:.4f} +/- {bt['learned_std']:.4f}",
        f"     Baseline: {bt['baseline_mean']:.4f} +/- {bt['baseline_std']:.4f}",
        f"     Mann-Whitney p = {bt['p_value']:.4f}  |  Cohen's d = {bt['cohens_d']:.2f}",
        f"     {bt['verdict']}",
    ]

    if tt is not None:
        lines += [
            f"",
            f"  2. Learning Trend (first vs last {min(50, len(coverage_curve) // 3)} episodes)",
            f"     Early: {tt['early_mean']:.4f}  ->  Late: {tt['late_mean']:.4f}"
            f"  (improvement: {tt['improvement']:+.4f})",
            f"     Mann-Whitney p = {tt['mann_whitney_p']:.4f}"
            f"  |  Spearman rho = {tt['spearman_rho']:.3f} (p={tt['spearman_p']:.4f})",
            f"     {tt['verdict']}",
        ]

    verdict_symbol = "PASS" if overall else "FAIL"
    lines += [
        f"",
        f"  OVERALL: {verdict_symbol}",
    ]

    return {
        "baseline_test": bt,
        "trend_test": tt,
        "overall_pass": overall,
        "summary": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Baseline coverage collector (for use in training scripts)
# ---------------------------------------------------------------------------


def collect_random_baseline_coverages(
    env,
    num_episodes: int = 20,
    seed: int = 9999,
) -> np.ndarray:
    """Run random policy episodes and return per-episode coverage.

    Light-weight — imports and runs inline so training scripts can call
    this without heavy setup.
    """
    import jax
    from red_within_blue.training.rollout import collect_episode, collect_episode_multi

    coverages = []
    key = jax.random.PRNGKey(seed)

    for _ in range(num_episodes):
        key, ep_key = jax.random.split(key)

        def random_policy(k, obs):
            return int(jax.random.randint(k, shape=(), minval=0, maxval=5))

        if len(env.agents) == 1:
            traj = collect_episode(env, random_policy, ep_key)
            cov = sum(float(r) for r in traj["rewards"])
        else:
            trajs = collect_episode_multi(env, random_policy, ep_key,
                                          enforce_connectivity=False)
            cov = sum(
                sum(float(r) for r in t["rewards"])
                for t in trajs.values()
            )
            cov = min(cov, 1.0)
        coverages.append(cov)

    return np.array(coverages, dtype=np.float64)
