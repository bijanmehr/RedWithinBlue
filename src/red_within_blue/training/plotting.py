"""Tufte + Nature + Academic visualization style for RedWithinBlue training."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import numpy as np
from pathlib import Path
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

COLORS: Dict[str, str] = {
    'blue':   '#4878A8',
    'red':    '#C05746',
    'gray':   '#888888',
    'green':  '#5A9E6F',
    'purple': '#8B6DAF',
    'orange': '#D49A3E',
}

ACTIONS = ['STAY', 'UP', 'RIGHT', 'DOWN', 'LEFT']

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply Tufte/Nature/Academic matplotlib rcParams globally."""
    matplotlib.rcParams.update({
        # Fonts
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif', 'Georgia'],
        'font.size':          9,
        'axes.titlesize':     10,
        'axes.labelsize':     9,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    7.5,

        # Figure and axes background
        'figure.facecolor':   '#FAFAF5',
        'axes.facecolor':     '#FAFAF5',

        # Spines
        'axes.spines.top':    False,
        'axes.spines.right':  False,

        # Grid
        'axes.grid':          False,

        # Edge styling
        'axes.edgecolor':     '#333333',
        'axes.linewidth':     0.6,

        # Save defaults
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
    })


# ---------------------------------------------------------------------------
# Multi-panel summary figure
# ---------------------------------------------------------------------------

def plot_stage_summary(
    data: Dict[str, Any],
    output_path: str,
    stage_name: str = "Stage",
) -> None:
    """Create and save a three-panel training summary figure.

    Required keys in *data*:
        coverage_mean        : array [T]  — learned agent mean coverage per episode
        coverage_std         : array [T]  — standard deviation
        coverage_random_mean : array [T]  — random baseline mean coverage
        coverage_random_std  : array [T]  — random baseline std
        action_dist_learned  : array [5]  — action probabilities, learned policy
        action_dist_random   : array [5]  — action probabilities, random policy
        visit_heatmap        : array [H,W] — visitation counts
        episodes             : array [T]  — episode indices (x-axis)

    Parameters
    ----------
    data        : dict with the keys listed above.
    output_path : base path (without extension); .pdf and .png are appended.
    stage_name  : human-readable label used in the figure title.
    """
    apply_style()

    coverage_mean        = np.asarray(data['coverage_mean'])
    coverage_std         = np.asarray(data['coverage_std'])
    coverage_random_mean = np.asarray(data['coverage_random_mean'])
    coverage_random_std  = np.asarray(data['coverage_random_std'])
    action_dist_learned  = np.asarray(data['action_dist_learned'])
    action_dist_random   = np.asarray(data['action_dist_random'])
    visit_heatmap        = np.asarray(data['visit_heatmap'])
    episodes             = np.asarray(data['episodes'])

    # ---- layout -----------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 6.4))
    fig.suptitle(stage_name, fontsize=11, fontweight='bold', y=0.98)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.15, 1],
        hspace=0.45,
    )
    inner_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        wspace=0.38,
    )

    ax_a = fig.add_subplot(outer[0])          # panel a — full width
    ax_b = fig.add_subplot(inner_bottom[0])   # panel b — action dist
    ax_c = fig.add_subplot(inner_bottom[1])   # panel c — heatmap

    # ---- panel a: learning curve with 95 % CI bands ----------------------
    ci_mult = 1.96  # 95 % confidence interval

    ax_a.fill_between(
        episodes,
        coverage_random_mean - ci_mult * coverage_random_std,
        coverage_random_mean + ci_mult * coverage_random_std,
        alpha=0.18, color=COLORS['gray'], linewidth=0,
    )
    ax_a.plot(
        episodes, coverage_random_mean,
        color=COLORS['gray'], linewidth=1.0,
        label='Random baseline', linestyle='--',
    )

    ax_a.fill_between(
        episodes,
        coverage_mean - ci_mult * coverage_std,
        coverage_mean + ci_mult * coverage_std,
        alpha=0.22, color=COLORS['blue'], linewidth=0,
    )
    ax_a.plot(
        episodes, coverage_mean,
        color=COLORS['blue'], linewidth=1.4,
        label='Learned policy',
    )

    ax_a.set_xlabel('Episode')
    ax_a.set_ylabel('Coverage')
    ax_a.set_title('Learning curve', fontsize=10)
    ax_a.legend(frameon=False, loc='lower right')
    ax_a.set_xlim(episodes[0], episodes[-1])
    ax_a.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f')
    )

    # Nature-style panel label
    ax_a.text(
        -0.06, 1.08, 'a',
        transform=ax_a.transAxes,
        fontsize=11, fontweight='bold',
        va='top', ha='left',
    )

    # ---- panel b: action distribution bar chart ---------------------------
    x = np.arange(len(ACTIONS))
    bar_w = 0.35

    ax_b.bar(
        x - bar_w / 2, action_dist_random,
        width=bar_w, color=COLORS['gray'], alpha=0.75,
        label='Random',
    )
    ax_b.bar(
        x + bar_w / 2, action_dist_learned,
        width=bar_w, color=COLORS['blue'], alpha=0.85,
        label='Learned',
    )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ACTIONS, rotation=35, ha='right', fontsize=7)
    ax_b.set_ylabel('Probability')
    ax_b.set_title('Action distribution', fontsize=10)
    ax_b.legend(frameon=False, fontsize=7)

    ax_b.text(
        -0.15, 1.08, 'b',
        transform=ax_b.transAxes,
        fontsize=11, fontweight='bold',
        va='top', ha='left',
    )

    # ---- panel c: visit heatmap -------------------------------------------
    im = ax_c.imshow(
        visit_heatmap,
        cmap='Greys',
        origin='upper',
        aspect='equal',
        interpolation='nearest',
    )
    plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    ax_c.set_title('Visitation heatmap', fontsize=10)
    ax_c.set_xlabel('Grid x')
    ax_c.set_ylabel('Grid y')
    # Remove ticks for cleaner look
    ax_c.tick_params(left=False, bottom=False,
                     labelleft=False, labelbottom=False)

    ax_c.text(
        -0.15, 1.08, 'c',
        transform=ax_c.transAxes,
        fontsize=11, fontweight='bold',
        va='top', ha='left',
    )

    # ---- save -------------------------------------------------------------
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(base) + '.pdf')
    fig.savefig(str(base) + '.png')
    plt.close(fig)
