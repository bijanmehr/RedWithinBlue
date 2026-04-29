"""Render comparison HTML + markdown for the Phase 1 red-prior experiment.

Consumes ``experiments/red-prior-phase1/red_prior_phase1.npz`` produced by
``scripts/red_prior_experiment.py`` and writes a comparison report next to it.

Plots are inline SVG (no matplotlib dependency). The HTML is styled to match
``experiments/meta-report/red_prior_report.html``.

Run: ``python scripts/red_prior_compare.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT = Path("experiments/red-prior-phase1/red_prior_phase1.npz")
OUT_HTML = Path("experiments/red-prior-phase1/compare.html")
OUT_MD = Path("experiments/red-prior-phase1/compare.md")
ARM_COLORS = {"I": "#1f3a5f", "W": "#1f6a45", "F": "#8a1f1f"}
ARM_LABEL = {"I": "I — Insider (copy blue)",
             "W": "W — Warm-start (Dense_0 only)",
             "F": "F — Fresh (random)"}


# --------------------------------------------------------------------------- #
# Plot helpers (inline SVG)
# --------------------------------------------------------------------------- #


def _smooth(x: np.ndarray, k: int = 50) -> np.ndarray:
    """Centred running mean — same length, edges padded with the original."""
    if len(x) < k:
        return x
    pad = k // 2
    out = np.copy(x)
    cs = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    win = (cs[k:] - cs[:-k]) / k
    out[pad:pad + len(win)] = win
    return out


def _line_svg(curves: dict[str, tuple[np.ndarray, np.ndarray]], *,
              width: int = 720, height: int = 220, ylabel: str = "",
              ylim: tuple[float, float] | None = None,
              hline: float | None = None) -> str:
    """Return an SVG string with a multi-line plot.

    ``curves[name] = (mean, std)`` arrays of equal length; mean is plotted
    as a solid line, std as a translucent band.
    """
    pad_l, pad_r, pad_t, pad_b = 50, 16, 12, 32
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    # Combine to find y range.
    all_means = np.concatenate([m for m, _ in curves.values()])
    if ylim is None:
        y_lo = float(min(all_means.min(), 0.0))
        y_hi = float(all_means.max())
        if y_hi - y_lo < 1e-6:
            y_hi = y_lo + 1.0
        pad_y = 0.05 * (y_hi - y_lo)
        y_lo -= pad_y; y_hi += pad_y
    else:
        y_lo, y_hi = ylim

    n = max(len(m) for m, _ in curves.values())
    x_lo, x_hi = 0, n - 1

    def x_to_px(xi: float) -> float:
        return pad_l + (xi - x_lo) / (x_hi - x_lo) * inner_w

    def y_to_px(yi: float) -> float:
        return pad_t + (1.0 - (yi - y_lo) / (y_hi - y_lo)) * inner_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family: monospace; font-size: 11px; max-width: 100%;">']

    # Background.
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # Frame.
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{inner_w}" height="{inner_h}" '
                 f'fill="#fafafa" stroke="#ccc"/>')

    # Y gridlines (5 ticks).
    for i in range(5):
        ty = y_lo + (y_hi - y_lo) * i / 4
        py = y_to_px(ty)
        parts.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{pad_l + inner_w}" '
                     f'y2="{py:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{py + 4:.1f}" text-anchor="end" '
                     f'fill="#666">{ty:.2f}</text>')

    # H-line (e.g. H_max).
    if hline is not None and y_lo < hline < y_hi:
        py = y_to_px(hline)
        parts.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{pad_l + inner_w}" '
                     f'y2="{py:.1f}" stroke="#bb8800" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{pad_l + inner_w - 4}" y="{py - 4:.1f}" '
                     f'text-anchor="end" fill="#bb8800">{hline:.3f}</text>')

    # X labels (start, mid, end).
    for frac in (0.0, 0.5, 1.0):
        xv = int(x_lo + (x_hi - x_lo) * frac)
        px = x_to_px(xv)
        parts.append(f'<text x="{px:.1f}" y="{height - 10}" text-anchor="middle" '
                     f'fill="#666">{xv}</text>')

    # Y label.
    if ylabel:
        parts.append(f'<text x="6" y="{pad_t + 12}" fill="#444">{ylabel}</text>')

    # Curves.
    for name, (mean, std) in curves.items():
        color = ARM_COLORS.get(name, "#444")
        # Std band as a polygon.
        if std is not None and np.any(std > 0):
            top = mean + std
            bot = mean - std
            poly_pts = []
            for i, v in enumerate(top):
                poly_pts.append(f'{x_to_px(i):.1f},{y_to_px(float(v)):.1f}')
            for i in range(len(bot) - 1, -1, -1):
                poly_pts.append(f'{x_to_px(i):.1f},{y_to_px(float(bot[i])):.1f}')
            parts.append(f'<polygon points="{" ".join(poly_pts)}" '
                         f'fill="{color}" fill-opacity="0.15" stroke="none"/>')
        # Mean line.
        path_pts = ' '.join(
            ('M' if i == 0 else 'L') + f'{x_to_px(i):.1f},{y_to_px(float(v)):.1f}'
            for i, v in enumerate(mean)
        )
        parts.append(f'<path d="{path_pts}" fill="none" stroke="{color}" '
                     f'stroke-width="1.5"/>')

    parts.append('</svg>')
    return ''.join(parts)


def _legend_html(arms_present: list[str]) -> str:
    chips = []
    for a in arms_present:
        c = ARM_COLORS.get(a, "#444")
        chips.append(f'<span style="display:inline-block;width:10px;height:10px;'
                     f'background:{c};margin-right:4px;border-radius:2px;"></span>'
                     f'{ARM_LABEL[a]}')
    return ' &nbsp; '.join(chips)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    data = np.load(INPUT, allow_pickle=False)
    arms = list(data["arms"].astype(str))
    H = data["red_entropy"]                # [A, S, E]
    Bj = data["blue_reward"]               # [A, S, E]
    Rj = data["red_reward"]                # [A, S, E]
    KL = data["prior_kl"]                  # [A, S, K]
    kl_eps = data["kl_eps"]
    n_blue = int(data["n_blue"])
    n_red = int(data["n_red"])
    E = int(data["num_episodes"])
    S = int(data["num_seeds"])
    config_path = str(data["config_path"])
    blue_ckpt = str(data["blue_ckpt"])

    H_MAX = float(np.log(5))

    # Smooth across episodes for plotting.
    K_SMOOTH = 50
    H_smooth = np.stack([[_smooth(H[a, s], K_SMOOTH) for s in range(S)]
                          for a in range(len(arms))])
    B_smooth = np.stack([[_smooth(Bj[a, s], K_SMOOTH) for s in range(S)]
                          for a in range(len(arms))])

    # Mean ± std across seeds, per-arm.
    H_mean = H_smooth.mean(axis=1); H_std = H_smooth.std(axis=1)
    B_mean = B_smooth.mean(axis=1); B_std = B_smooth.std(axis=1)

    # Final-window summary numbers (last 100 eps, raw).
    rows = []
    for ai, a in enumerate(arms):
        h0 = H[ai, :, :50].mean()
        hf = H[ai, :, -100:].mean(axis=-1)
        bf = Bj[ai, :, -100:].mean(axis=-1)
        rf = Rj[ai, :, -100:].mean(axis=-1)
        # First episode where mean entropy across seeds drops below 0.05.
        h_seed_mean = H[ai].mean(axis=0)
        below = np.where(h_seed_mean < 0.05)[0]
        t_collapse = int(below[0]) if len(below) else None
        # Final KL to prior (use last KL probe point per seed).
        klf = KL[ai, :, -1]
        rows.append({
            "arm": a, "h0": float(h0),
            "hf_mean": float(hf.mean()), "hf_std": float(hf.std()),
            "bf_mean": float(bf.mean()), "bf_std": float(bf.std()),
            "rf_mean": float(rf.mean()), "rf_std": float(rf.std()),
            "t_collapse": t_collapse,
            "klf_mean": float(klf.mean()), "klf_std": float(klf.std()),
        })

    # ---------------- HTML ----------------

    h_curves = {a: (H_mean[ai], H_std[ai]) for ai, a in enumerate(arms)}
    b_curves = {a: (B_mean[ai], B_std[ai]) for ai, a in enumerate(arms)}

    h_svg = _line_svg(h_curves, ylabel="H[π_red]  (nats; H_max=ln 5≈1.609)",
                      hline=H_MAX, ylim=(0.0, max(H_MAX, 1.7)))
    b_svg = _line_svg(b_curves, ylabel="blue mean per-agent return")
    legend = _legend_html(arms)

    summary_table = "\n".join(
        f"<tr><td><strong>{r['arm']}</strong> {ARM_LABEL[r['arm']].split('—',1)[1].strip()}</td>"
        f"<td class='num'>{r['h0']:.3f}</td>"
        f"<td class='num'>{r['hf_mean']:.3f} ± {r['hf_std']:.3f}</td>"
        f"<td class='num'>{r['t_collapse'] if r['t_collapse'] is not None else 'never'}</td>"
        f"<td class='num'>{r['bf_mean']:+.2f} ± {r['bf_std']:.2f}</td>"
        f"<td class='num'>{r['rf_mean']:+.2f} ± {r['rf_std']:.2f}</td>"
        f"<td class='num'>{r['klf_mean']:.3f} ± {r['klf_std']:.3f}</td></tr>"
        for r in rows
    )

    # Pick a "headline" — does I land at a different stationary point than F?
    h_I = next((r for r in rows if r["arm"] == "I"), None)
    h_F = next((r for r in rows if r["arm"] == "F"), None)
    h_W = next((r for r in rows if r["arm"] == "W"), None)
    headline_lines = []
    if h_I and h_F:
        delta_b = h_I["bf_mean"] - h_F["bf_mean"]
        verdict = ("F-prior collapses to a different basin than I"
                   if abs(h_I["hf_mean"] - h_F["hf_mean"]) > 0.1
                   else "I and F converge to similar entropy")
        headline_lines.append(f"<li><strong>{verdict}.</strong> "
                              f"Δblue_R(I − F) = {delta_b:+.2f} per-agent at "
                              f"the final 100 eps.</li>")
    if h_W and h_F:
        delta_h = h_W["hf_mean"] - h_F["hf_mean"]
        headline_lines.append(f"<li>Warm-start vs Fresh entropy gap: "
                              f"ΔH_final = {delta_h:+.3f} nats.</li>")
    headline_html = "<ul>" + "".join(headline_lines) + "</ul>" if headline_lines else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Red Prior Phase 1 — Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1400px; margin: 2em auto; padding: 0 1em; line-height: 1.55; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ margin-top: 2.4em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.95em; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: left; }}
th {{ background: #f0f0f0; }}
td.num {{ text-align: right; font-family: "SF Mono", Menlo, Consolas, monospace; }}
code {{ font-family: "SF Mono", Menlo, Consolas, monospace; background: #f5f5f5;
        padding: 0 0.25em; border-radius: 3px; }}
.callout {{ background: #fff5e6; border-left: 4px solid #e69500; padding: 0.8em 1em;
            margin: 1em 0; }}
.callout.green {{ background: #eefaf0; border-left-color: #2ca02c; }}
.callout.red {{ background: #fdeeee; border-left-color: #d62728; }}
.legend {{ font-size: 0.9em; color: #555; }}
.plot-wrap {{ margin: 1em 0; }}
.plot-legend {{ font-size: 0.9em; margin: 0.4em 0 0.8em; }}
</style>
</head>
<body>

<h1>Red Prior Phase 1 — same-class red, three priors compared</h1>

<p class="legend">Companion to <code>experiments/meta-report/red_prior_report.html</code> (which framed
existing red runs as the F × γ-alien corner of the design matrix). This report fills the <em>same-class</em>
column of that matrix for all three prior arms — I, W, F — using a per-agent <code>RedActor</code>
architecturally identical to blue. Blue is frozen at <code>{blue_ckpt}</code>; red is trained with
sign-flipped reward at <code>rewards_training.py:264-267</code>. {S} seeds × {E} episodes per arm.</p>

<h2>Headline numbers</h2>

{headline_html}

<table>
<tr><th>arm</th><th>H[π_red] @ ep 0–50</th><th>H[π_red] last 100ep</th>
    <th>first ep with H̄&lt;0.05</th>
    <th>blue R last 100ep</th><th>red R last 100ep</th><th>KL(π_t ‖ π_0) final</th></tr>
{summary_table}
</table>

<p class="legend">All numbers are mean ± std over {S} seeds. Red R is per-agent (sign-flipped blue mean).
KL(π_t ‖ π_0) is the policy distribution under the current params vs the prior, evaluated on a
fixed-seed reset state.</p>

<h2>Red policy entropy across episodes</h2>

<div class="plot-wrap">
<div class="plot-legend">{legend}</div>
{h_svg}
</div>

<p>The §3 prediction in <code>docs/10-wild-ideas.md</code> is
<span class="math">t_stable(I) ≪ t_stable(W) &lt; t_stable(F)</span>.
The Insider arm starts at low entropy (it inherits blue's near-deterministic move policy) and either stays
low (the red task is consistent with blue's moves) or climbs as the red gradient pushes against the blue
basin. The Fresh arm starts near H_max=ln 5 ≈ 1.609 and concentrates if the task gives a non-degenerate
gradient. The Warm-start arm starts near H_max but inherits blue's Dense_0 weights (the bottom MLP layer that maps obs → hidden), so its inputs are pre-shaped while its action head is random.</p>

<h2>Blue mean per-agent return across episodes</h2>

<div class="plot-wrap">
<div class="plot-legend">{legend}</div>
{b_svg}
</div>

<p>This is the <em>defender's</em> side of the same training curves: lower is better attack. Differences
between arms in this curve at matched episode are the prior's contribution to attack quality, controlled
for everything downstream of red's first action.</p>

<h2>Reading the comparison</h2>

<div class="callout">
<strong>Interpretation guide.</strong> Three things to watch for:
<ol>
<li><strong>Different terminal basin?</strong> If <code>H[π_red]</code> at ep {E - 1} differs by more than
≈0.10 nats between arms, the priors converge to different stationary distributions on this same-class
architecture. That's a <em>basin attractor</em> story, not a concentration-rate story.</li>
<li><strong>Different concentration rate?</strong> Compare the "first ep with H̄&lt;0.05" column. The §3
prediction is I &lt; W &lt; F (fastest first). If they're all the same, the rate is task-bound, not
prior-bound.</li>
<li><strong>Different attack quality?</strong> Compare blue's last-100-ep mean across arms. A larger
<code>|Δblue_R|</code> at matched eps indicates priors give meaningfully different attacker policies.</li>
</ol>
</div>

<p class="legend">Source: <code>{INPUT}</code>; config <code>{config_path}</code>; n_blue={n_blue}, n_red={n_red}.</p>

</body>
</html>
"""

    OUT_HTML.write_text(html)
    print(f"Wrote {OUT_HTML}")

    # ---------------- Markdown ----------------

    md_lines = [
        "# Red Prior Phase 1 — same-class red, three priors compared",
        "",
        f"Companion to `experiments/meta-report/red_prior_report.md`. Fills the **same-class** column of",
        f"the I/W/F × α/β/γ design matrix for all three prior arms.",
        "",
        f"- Blue frozen at `{blue_ckpt}`",
        f"- Red: per-agent `RedActor`, identical to blue's `Actor`",
        f"- Reward: sign-flipped per-blue mean (`rewards_training.py:264-267`)",
        f"- {S} seeds × {E} episodes per arm; n_blue={n_blue}, n_red={n_red}",
        "",
        "## Final-window summary",
        "",
        "| arm | H@0-50 | H last 100ep | t_collapse | blue R last 100ep | red R last 100ep | KL(π_t‖π_0) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['arm']} | {r['h0']:.3f} | {r['hf_mean']:.3f} ± {r['hf_std']:.3f} "
            f"| {r['t_collapse'] if r['t_collapse'] is not None else 'never'} "
            f"| {r['bf_mean']:+.2f} ± {r['bf_std']:.2f} "
            f"| {r['rf_mean']:+.2f} ± {r['rf_std']:.2f} "
            f"| {r['klf_mean']:.3f} ± {r['klf_std']:.3f} |"
        )
    md_lines += [
        "",
        "## Findings",
        "",
    ]
    if h_I and h_F:
        delta_b = h_I["bf_mean"] - h_F["bf_mean"]
        delta_h = h_I["hf_mean"] - h_F["hf_mean"]
        md_lines.append(
            f"- **I vs F final entropy**: ΔH = {delta_h:+.3f} nats. "
            f"**I vs F blue return**: Δ = {delta_b:+.2f} per-agent."
        )
    if h_W and h_F:
        delta_h_wf = h_W["hf_mean"] - h_F["hf_mean"]
        delta_b_wf = h_W["bf_mean"] - h_F["bf_mean"]
        md_lines.append(
            f"- **W vs F final entropy**: ΔH = {delta_h_wf:+.3f} nats. "
            f"**W vs F blue return**: Δ = {delta_b_wf:+.2f} per-agent."
        )
    md_lines += [
        "",
        f"See `compare.html` for plots; raw arrays in `red_prior_phase1.npz`.",
        "",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
