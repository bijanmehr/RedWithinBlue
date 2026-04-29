"""Self-contained HTML experiment reports for RedWithinBlue.

Generates a single HTML file per experiment with:
  - Experiment metadata with parameter descriptions
  - Coverage statistics table + per-seed bar chart
  - Action distribution bar chart
  - Chi-squared test results
  - Training curves (coverage, loss) for training experiments
  - Interactive animated GIF player with playback speed control

All images are base64-encoded inline so the report is a single portable file.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from red_within_blue.training.plotting import apply_style, COLORS, ACTIONS


# ---------------------------------------------------------------------------
# Parameter descriptions
# ---------------------------------------------------------------------------

_PARAM_DESCRIPTIONS: Dict[str, str] = {
    "policy": "The decision-making strategy used (e.g. random, greedy, or learned).",
    "stage": "Curriculum stage (1 = single-agent, 2 = 2 agents, 3 = 4 agents with walls).",
    "num_seeds": "Number of independent runs with different random seeds for statistical robustness.",
    "num_episodes": "Training episodes per seed. Each episode is one complete grid exploration.",
    "num_agents": "Number of agents exploring the grid simultaneously.",
    "max_steps": "Maximum timesteps per episode before termination.",
    "grid_width": "Horizontal size of the grid world.",
    "grid_height": "Vertical size of the grid world.",
    "grid": "Grid dimensions (width x height). Includes 1-cell wall border, so playable area is (w-2)x(h-2).",
    "wall_density": "Fraction of cells that are impassable walls (0.0 = open, 0.1 = 10% walls).",
    "lr": "Learning rate for the Adam optimizer.",
    "gamma": "Discount factor for future rewards (higher = more foresight).",
    "track": "Training algorithm track (pg = policy gradient, dqn = deep Q-network).",
    "layer": "PG complexity layer (1 = REINFORCE, 2 = with baseline, 3 = actor-critic).",
    "warm_start": "Checkpoint path for weight transfer from a previous stage, or 'none'.",
    "eval_interval": "How often (in episodes) to log evaluation metrics.",
    "checkpoint_interval": "How often (in episodes) to save model checkpoints.",
    "comm_radius": "Communication range (Euclidean distance). Agents within range form graph edges.",
    "obs_radius": "Legacy single-radius knob; default source for view_radius and survey_radius when they are left at -1.",
    "view_radius": "Sensor half-size. The policy sees a (2r+1)x(2r+1) window of terrain around the agent.",
    "survey_radius": "Per-cell mission footprint. Only (2r+1)^2 cells around the agent get committed to the local_map each step; 0 = just the current cell.",
    "local_obs": "If True, the obs' seen field is a view-sized window instead of the full grid mask. Shrinks obs_dim dramatically.",
    "obs_dim": "Total observation vector dimension (derived from view_radius, local_obs, and grid size).",
    "disconnect_penalty": "Per-agent per-step penalty when communication graph is disconnected.",
    "actor_hidden_dim": "Hidden layer size in the actor (policy) network.",
    "actor_num_layers": "Number of hidden layers in the actor network.",
    "critic_hidden_dim": "Hidden layer size in the critic (value) network.",
    "critic_num_layers": "Number of hidden layers in the critic network.",
    "connectivity_guardrail": "Whether actions that disconnect the comm graph are blocked.",
}


# ---------------------------------------------------------------------------
# Chart generators (return base64-encoded PNG strings)
# ---------------------------------------------------------------------------


def _fig_to_base64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _coverage_per_seed_chart(coverages: List[float]) -> str:
    """Bar chart of coverage per seed."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 2.5))
    x = np.arange(len(coverages))
    ax.bar(x, coverages, color=COLORS["blue"], alpha=0.85, width=0.7)
    mean_val = float(np.mean(coverages))
    ax.axhline(mean_val, color=COLORS["red"], linewidth=1.2, linestyle="--",
               label=f"mean = {mean_val:.3f}")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage per seed")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def _action_distribution_chart(action_dist: List[float]) -> str:
    """Bar chart of action probabilities."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4, 2.5))
    x = np.arange(len(action_dist))
    labels = ACTIONS[: len(action_dist)]
    ax.bar(x, action_dist, color=COLORS["blue"], alpha=0.85, width=0.6)
    ax.axhline(0.2, color=COLORS["gray"], linewidth=0.8, linestyle="--",
               label="uniform (0.20)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Action distribution")
    ax.set_ylim(0, max(action_dist) * 1.3 if max(action_dist) > 0 else 0.5)
    ax.legend(frameon=False, fontsize=7)
    return _fig_to_base64(fig)


def _visit_heatmap_chart(heatmap: np.ndarray) -> str:
    """Greyscale heatmap of visit counts with colorbar."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(heatmap, cmap="Greys", origin="upper", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Visit count")
    ax.set_title("Visitation heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return _fig_to_base64(fig)


def _connectivity_timeline_chart(connectivity: list, max_steps: int) -> str:
    """Timeline showing connected vs fragmented at each step."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 2))
    steps = np.arange(len(connectivity))
    connected = np.array(connectivity, dtype=np.float64)

    # Fill regions
    ax.fill_between(steps, 0, 1, where=connected > 0.5,
                     color=COLORS["green"], alpha=0.35, label="Connected")
    ax.fill_between(steps, 0, 1, where=connected < 0.5,
                     color=COLORS["red"], alpha=0.25, label="Fragmented")

    pct = float(np.mean(connected)) * 100
    ax.set_title(f"Graph connectivity ({pct:.0f}% connected)")
    ax.set_xlabel("Step")
    ax.set_yticks([])
    ax.set_xlim(0, len(connectivity) - 1)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    return _fig_to_base64(fig)


def _coverage_over_time_chart(coverage_over_time: list) -> str:
    """Coverage fraction over the course of one episode."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 2.5))
    steps = np.arange(len(coverage_over_time))
    ax.plot(steps, coverage_over_time, color=COLORS["blue"], linewidth=1.2)
    ax.fill_between(steps, 0, coverage_over_time, color=COLORS["blue"], alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage over episode")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, len(coverage_over_time) - 1)
    return _fig_to_base64(fig)


def _learning_curve_chart(
    metrics: List[Dict[str, Any]],
    y_key: str = "coverage",
    y_label: str = "Coverage",
) -> Optional[str]:
    """Line chart from metrics records. Returns None if no data."""
    steps = [m["step"] for m in metrics if y_key in m]
    values = [m[y_key] for m in metrics if y_key in m]
    if not steps:
        return None
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(steps, values, color=COLORS["blue"], linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} over training")
    return _fig_to_base64(fig)


def _loss_curve_chart(metrics: List[Dict[str, Any]]) -> Optional[str]:
    """Loss curve from metrics records."""
    for key in ("loss", "total_loss"):
        steps = [m["step"] for m in metrics if key in m]
        values = [m[key] for m in metrics if key in m]
        if steps:
            apply_style()
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.plot(steps, values, color=COLORS["red"], linewidth=1.0)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Loss over training")
            return _fig_to_base64(fig)
    return None


def _team_reward_duality_chart(metrics: List[Dict[str, Any]]) -> Optional[str]:
    """Mirror plot: blue + red total rewards on the same axes per episode.

    Under zero-sum, the two curves are reflections about y = 0; the plot
    makes the duality visible at a glance and the gap to zero shows the
    cumulative scale of the conflict.
    """
    steps = [m["step"] for m in metrics if "blue_total_reward" in m and "red_total_reward" in m]
    if not steps:
        return None
    blue = [m["blue_total_reward"] for m in metrics if "blue_total_reward" in m]
    red = [m["red_total_reward"] for m in metrics if "red_total_reward" in m]
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(steps, blue, color=COLORS["blue"], linewidth=1.0, label="Blue team")
    ax.plot(steps, red, color=COLORS["red"], linewidth=1.0, label="Red team")
    ax.axhline(0.0, color="#888", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total team reward")
    ax.set_title("Zero-sum reward duality")
    ax.legend(frameon=False, fontsize=7, loc="best")
    return _fig_to_base64(fig)


def _policy_entropy_chart(metrics: List[Dict[str, Any]]) -> Optional[str]:
    """Per-team policy entropy over episodes — a Nash convergence diagnostic.

    At a mixed-strategy Nash, both entropies are positive and stable. Entropy
    collapse on either side suggests one team is dominating; oscillation
    suggests cycling without convergence.
    """
    steps = [
        m["step"] for m in metrics
        if "blue_policy_entropy" in m and "red_policy_entropy" in m
    ]
    if not steps:
        return None
    blue_ent = [m["blue_policy_entropy"] for m in metrics if "blue_policy_entropy" in m]
    red_ent = [m["red_policy_entropy"] for m in metrics if "red_policy_entropy" in m]
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(steps, blue_ent, color=COLORS["blue"], linewidth=1.0, label="Blue")
    ax.plot(steps, red_ent, color=COLORS["red"], linewidth=1.0, label="Red")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_title("Policy entropy over training")
    ax.legend(frameon=False, fontsize=7, loc="best")
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# GIF embedding
# ---------------------------------------------------------------------------


def _gif_to_base64(gif_path: str) -> Optional[str]:
    """Read a GIF file and return its base64 encoding."""
    p = Path(gif_path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --bg: #FAFAF5;
    --bg-card: #FFFFFF;
    --text: #222;
    --text-muted: #666;
    --border: #e0ddd8;
    --border-strong: #333;
    --accent: #4878A8;
    --accent-light: #e8f0f8;
    --pass: #2a7a3a;
    --fail: #b33;
  }}

  * {{ box-sizing: border-box; }}

  body {{
    font-family: 'Georgia', 'Times New Roman', serif;
    max-width: 920px;
    margin: 0 auto;
    padding: 32px 24px;
    background: var(--bg);
    color: var(--text);
    line-height: 1.65;
  }}

  /* --- Header --- */
  .report-header {{
    border-bottom: 2px solid var(--border-strong);
    padding-bottom: 12px;
    margin-bottom: 28px;
  }}
  .report-header h1 {{
    font-size: 1.6em;
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
  }}
  .report-header .subtitle {{
    font-size: 0.85em;
    color: var(--text-muted);
  }}

  /* --- Sections --- */
  h2 {{
    font-size: 1.1em;
    color: var(--text);
    margin-top: 36px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    letter-spacing: 0.02em;
  }}

  .section-desc {{
    font-size: 0.85em;
    color: var(--text-muted);
    margin: -8px 0 14px 0;
    font-style: italic;
  }}

  /* --- Tables --- */
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 0.88em;
    background: var(--bg-card);
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  th, td {{
    padding: 8px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: #f3f2ee;
    font-weight: 600;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #555;
  }}
  tr:last-child td {{ border-bottom: none; }}
  td.desc {{
    font-size: 0.82em;
    color: var(--text-muted);
    font-style: italic;
    max-width: 340px;
  }}

  .stat-pass {{ color: var(--pass); font-weight: 600; }}
  .stat-fail {{ color: var(--fail); font-weight: 600; }}

  /* --- Charts --- */
  .chart-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    margin: 14px 0;
    justify-content: center;
  }}
  .chart-row img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }}

  /* --- GIF Player --- */
  .gif-player {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    margin: 16px 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  }}
  .gif-player .gif-display {{
    text-align: center;
    margin-bottom: 12px;
  }}
  .gif-player img {{
    max-width: 100%;
    border-radius: 4px;
  }}
  .gif-controls {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
  }}
  .gif-controls button {{
    font-family: inherit;
    font-size: 0.85em;
    padding: 6px 14px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg);
    color: var(--text);
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }}
  .gif-controls button:hover {{
    background: var(--accent-light);
    border-color: var(--accent);
  }}
  .gif-controls button.active {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }}
  .gif-controls .speed-label {{
    font-size: 0.82em;
    color: var(--text-muted);
    margin-left: 6px;
  }}
  .gif-controls input[type="range"] {{
    width: 120px;
    accent-color: var(--accent);
  }}
  .frame-info {{
    text-align: center;
    font-size: 0.8em;
    color: var(--text-muted);
    margin-top: 6px;
  }}

  /* --- Footer --- */
  .footer {{
    margin-top: 48px;
    font-size: 0.78em;
    color: #aaa;
    border-top: 1px solid var(--border);
    padding-top: 10px;
    text-align: center;
  }}
</style>
</head>
<body>

<div class="report-header">
  <h1>{title}</h1>
  <div class="subtitle">{timestamp}</div>
</div>

{body}

<div class="footer">Generated by RedWithinBlue experiment runner</div>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# GIF player JavaScript (frame-by-frame via canvas)
# ---------------------------------------------------------------------------

_GIF_PLAYER_JS = """\
<script>
(function() {
  // SuperGif-lite: parse GIF frames from base64 data
  // We use a simpler approach: swap CSS animation-duration for speed control,
  // and use canvas + img for pause/play.

  const container = document.getElementById('gif-player');
  if (!container) return;

  const img = container.querySelector('img');
  const src = img.getAttribute('data-src');

  // State
  let playing = true;
  let speed = 1.0;

  // Store original src
  const gifSrc = src;

  // For pause: capture current frame by drawing to canvas
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  function pause() {
    if (!playing) return;
    playing = false;
    // Capture current frame
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    img.src = canvas.toDataURL();
    updateButtons();
  }

  function play() {
    if (playing) return;
    playing = true;
    // Reload GIF to restart animation
    img.src = '';
    img.src = gifSrc;
    updateButtons();
  }

  function togglePlay() {
    if (playing) pause(); else play();
  }

  function updateButtons() {
    const btn = document.getElementById('btn-play');
    if (btn) {
      btn.textContent = playing ? '\\u23F8 Pause' : '\\u25B6 Play';
      btn.classList.toggle('active', playing);
    }
  }

  function setSpeed(s) {
    speed = s;
    // Update speed buttons
    document.querySelectorAll('.speed-btn').forEach(b => {
      b.classList.toggle('active', parseFloat(b.dataset.speed) === s);
    });
    // Speed shown by active button highlight
    // Restart GIF to apply (browser limitation: can't change GIF speed directly)
    if (playing) {
      img.src = '';
      img.src = gifSrc;
    }
  }

  // Expose to HTML onclick handlers
  window.gifTogglePlay = togglePlay;
  window.gifSetSpeed = setSpeed;

  // Initialize
  img.src = gifSrc;
  updateButtons();
})();
</script>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    title: str,
    output_path: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    coverages: Optional[List[float]] = None,
    action_dist: Optional[List[float]] = None,
    chi2: Optional[float] = None,
    chi2_p: Optional[float] = None,
    metrics: Optional[List[Dict[str, Any]]] = None,
    gif_path: Optional[str] = None,
    visit_heatmap: Optional[np.ndarray] = None,
    connectivity: Optional[List[bool]] = None,
    coverage_over_time: Optional[List[float]] = None,
    max_steps: Optional[int] = None,
    extra_sections: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Generate a self-contained HTML experiment report.

    Parameters
    ----------
    title : str
        Report title (e.g. "Stage 1 | Baseline: random").
    output_path : str
        Where to write the HTML file.
    hyperparams : dict, optional
        Key-value pairs shown in the config table.
    coverages : list[float], optional
        Per-seed coverage values.
    action_dist : list[float], optional
        Action probability distribution (5 values).
    chi2, chi2_p : float, optional
        Chi-squared statistic and p-value.
    metrics : list[dict], optional
        Training metrics records (for learning/loss curves).
    gif_path : str, optional
        Path to an animated GIF to embed.
    visit_heatmap : np.ndarray, optional
        [H, W] visit-count array from the eval episode.
    connectivity : list[bool], optional
        Per-step graph connectivity from the eval episode.
    coverage_over_time : list[float], optional
        Per-step coverage fraction from the eval episode.
    max_steps : int, optional
        Episode length limit (used for connectivity timeline x-axis).
    extra_sections : list[dict], optional
        Additional sections, each with "title" and "html" keys.

    Returns
    -------
    str
        Absolute path to the written report.
    """
    parts: List[str] = []

    # --- Configuration table with descriptions ---
    if hyperparams:
        rows = []
        for k, v in hyperparams.items():
            desc = _PARAM_DESCRIPTIONS.get(k, "")
            desc_cell = f'<td class="desc">{desc}</td>' if desc else '<td class="desc"></td>'
            rows.append(f"<tr><td><strong>{k}</strong></td><td>{v}</td>{desc_cell}</tr>")
        rows_html = "".join(rows)
        parts.append(
            f'<h2>Configuration</h2>\n'
            f'<p class="section-desc">Experiment hyperparameters and environment settings.</p>\n'
            f'<table><tr><th>Parameter</th><th>Value</th><th>Description</th></tr>{rows_html}</table>'
        )

    # --- Coverage statistics ---
    if coverages:
        arr = np.array(coverages, dtype=np.float32)
        parts.append(
            f'<h2>Coverage</h2>\n'
            f'<p class="section-desc">Fraction of reachable grid cells visited by the end of each episode.</p>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>Seeds</td><td>{len(coverages)}</td></tr>\n'
            f'<tr><td>Mean</td><td>{float(np.mean(arr)):.4f}</td></tr>\n'
            f'<tr><td>Std</td><td>{float(np.std(arr)):.4f}</td></tr>\n'
            f'<tr><td>Min</td><td>{float(np.min(arr)):.4f}</td></tr>\n'
            f'<tr><td>Max</td><td>{float(np.max(arr)):.4f}</td></tr>\n'
            f'</table>'
        )
        chart_b64 = _coverage_per_seed_chart(coverages)
        parts.append(f'<div class="chart-row"><img src="data:image/png;base64,{chart_b64}"></div>')

    # --- Action distribution ---
    if action_dist:
        chart_b64 = _action_distribution_chart(action_dist)
        parts.append(
            f'<h2>Action Distribution</h2>\n'
            f'<p class="section-desc">'
            f'How often each action was chosen across all seeds and agents. '
            f'A uniform distribution (0.20 each) indicates no learned preference.'
            f'</p>\n'
            f'<div class="chart-row"><img src="data:image/png;base64,{chart_b64}"></div>'
        )

    # --- Chi-squared test ---
    if chi2 is not None and chi2_p is not None:
        if chi2_p < 0.05:
            verdict_cls = "stat-pass"
            verdict_txt = "PASS: policy differs significantly from uniform (p &lt; 0.05)"
        else:
            verdict_cls = "stat-fail"
            verdict_txt = "FAIL: distribution is consistent with uniform (p &ge; 0.05)"
        parts.append(
            f'<h2>Statistical Test</h2>\n'
            f'<p class="section-desc">'
            f'Chi-squared goodness-of-fit test against a uniform action distribution. '
            f'A low p-value means the policy has learned to prefer certain actions over others.'
            f'</p>\n'
            f'<table>\n'
            f'<tr><th>Statistic</th><th>Value</th></tr>\n'
            f'<tr><td>Chi-squared</td><td>{chi2:.4f}</td></tr>\n'
            f'<tr><td>p-value</td><td>{chi2_p:.6f}</td></tr>\n'
            f'<tr><td>Verdict</td><td class="{verdict_cls}">{verdict_txt}</td></tr>\n'
            f'</table>'
        )

    # --- Learning curves (training experiments) ---
    if metrics:
        cov_chart = _learning_curve_chart(metrics, "coverage", "Coverage")
        if cov_chart is None:
            cov_chart = _learning_curve_chart(metrics, "team_coverage", "Team Coverage")
        loss_chart = _loss_curve_chart(metrics)

        if cov_chart or loss_chart:
            parts.append(
                '<h2>Training Curves</h2>\n'
                '<p class="section-desc">'
                'How coverage and loss evolved during training across episodes.'
                '</p>'
            )
            row_items = []
            if cov_chart:
                row_items.append(f'<img src="data:image/png;base64,{cov_chart}">')
            if loss_chart:
                row_items.append(f'<img src="data:image/png;base64,{loss_chart}">')
            parts.append(f'<div class="chart-row">{"".join(row_items)}</div>')

        # --- Nash & duality diagnostics (joint-red trainer) ---
        duality_chart = _team_reward_duality_chart(metrics)
        entropy_chart = _policy_entropy_chart(metrics)
        if duality_chart or entropy_chart:
            parts.append(
                '<h2>Nash &amp; Duality</h2>\n'
                '<p class="section-desc">'
                'Two diagnostics for the adversarial training loop. The reward '
                'mirror plot shows the zero-sum coupling between blue and red '
                'team totals; under exact zero-sum the two curves are reflections '
                'about y = 0. Per-team policy entropy tracks whether either side '
                'has converged to a near-deterministic best-response (entropy '
                'collapse) or both remain mixed (typical of a Nash equilibrium).'
                '</p>'
            )
            row_items = []
            if duality_chart:
                row_items.append(f'<img src="data:image/png;base64,{duality_chart}">')
            if entropy_chart:
                row_items.append(f'<img src="data:image/png;base64,{entropy_chart}">')
            parts.append(f'<div class="chart-row">{"".join(row_items)}</div>')

    # --- Visitation heatmap ---
    if visit_heatmap is not None:
        heatmap_b64 = _visit_heatmap_chart(visit_heatmap)
        parts.append(
            f'<h2>Visitation Heatmap</h2>\n'
            f'<p class="section-desc">'
            f'Spatial distribution of agent visits during the evaluation episode. '
            f'Darker cells were visited more often.'
            f'</p>\n'
            f'<div class="chart-row"><img src="data:image/png;base64,{heatmap_b64}"></div>'
        )

    # --- Connectivity timeline ---
    if connectivity:
        ms = max_steps or len(connectivity)
        conn_b64 = _connectivity_timeline_chart(connectivity, ms)
        parts.append(
            f'<h2>Connectivity Timeline</h2>\n'
            f'<p class="section-desc">'
            f'Whether the agent communication graph was fully connected at each step. '
            f'Green = connected, red = fragmented.'
            f'</p>\n'
            f'<div class="chart-row"><img src="data:image/png;base64,{conn_b64}"></div>'
        )

    # --- Coverage over time ---
    if coverage_over_time:
        cov_time_b64 = _coverage_over_time_chart(coverage_over_time)
        parts.append(
            f'<h2>Coverage Over Episode</h2>\n'
            f'<p class="section-desc">'
            f'How the fraction of explored cells grew during the evaluation episode.'
            f'</p>\n'
            f'<div class="chart-row"><img src="data:image/png;base64,{cov_time_b64}"></div>'
        )

    # --- Animated GIF with player controls ---
    if gif_path:
        gif_b64 = _gif_to_base64(gif_path)
        if gif_b64:
            parts.append(
                f'<h2>Agent Behavior</h2>\n'
                f'<p class="section-desc">'
                f'Animated replay of one episode. Darker cells = visited more often. '
                f'Blue circles = agents. Gray lines = communication links. '
                f'The metrics panel shows real-time step count, coverage, and graph connectivity.'
                f'</p>\n'
                f'<div class="gif-player" id="gif-player">\n'
                f'  <div class="gif-display">\n'
                f'    <img data-src="data:image/gif;base64,{gif_b64}" '
                f'src="data:image/gif;base64,{gif_b64}" alt="Episode animation">\n'
                f'  </div>\n'
                f'  <div class="gif-controls">\n'
                f'    <button id="btn-play" class="active" onclick="gifTogglePlay()">'
                f'\u23F8 Pause</button>\n'
                f'    <span class="speed-label">Speed:</span>\n'
                f'    <button class="speed-btn" data-speed="0.5" '
                f'onclick="gifSetSpeed(0.5)">0.5x</button>\n'
                f'    <button class="speed-btn active" data-speed="1" '
                f'onclick="gifSetSpeed(1)">1x</button>\n'
                f'    <button class="speed-btn" data-speed="2" '
                f'onclick="gifSetSpeed(2)">2x</button>\n'
                f'    <button class="speed-btn" data-speed="4" '
                f'onclick="gifSetSpeed(4)">4x</button>\n'
                f''
                f'  </div>\n'
                f'</div>\n'
                f'{_GIF_PLAYER_JS}'
            )

    # --- Extra sections ---
    if extra_sections:
        for section in extra_sections:
            parts.append(f"<h2>{section['title']}</h2>\n{section['html']}")

    body = "\n".join(parts)
    html = _HTML_TEMPLATE.format(
        title=title,
        body=body,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return str(out.resolve())
