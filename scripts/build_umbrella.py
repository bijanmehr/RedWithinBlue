"""Build experiments/umbrella.html — a data-only review of every artifact in
experiments/.

This is intentionally NOT interpretive. It assembles plots, GIFs, raw JSON
summaries, and pointers to checkpoints/metrics — and presents them with
provenance only. No bottom-line callouts, no hypothesis tables, no claims.

The companion stylesheet reuses the sticky-TOC + click-to-fullscreen pattern
from earlier reports, but every panel here is just `<figure>` + caption.

Run:
    python scripts/build_umbrella.py
"""

from __future__ import annotations

import json
import pathlib
from html import escape

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
META = EXP / "meta-report"
PHASE1 = EXP / "red-prior-phase1"
OUT = EXP / "umbrella.html"

CSS = """
:root {
    --fg: #1c1c1e;
    --fg-soft: #555;
    --bg: #fafafa;
    --bg-card: #ffffff;
    --rule: #e0e0e0;
    --accent: #1f3a5f;
    --accent-soft: #eaeef5;
    --tag: #555;
    --tag-bg: #ececec;
}
* { box-sizing: border-box; }
body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--fg); background: var(--bg); line-height: 1.55;
}
#layout { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
#toc {
    position: sticky; top: 0; align-self: start;
    height: 100vh; overflow-y: auto;
    background: var(--bg-card); border-right: 1px solid var(--rule);
    padding: 2em 1.5em;
}
#toc h2 { margin: 0 0 0.3em 0; font-size: 1.05em; color: var(--accent); }
#toc .subtitle { font-size: 0.85em; color: var(--fg-soft); margin-bottom: 1.2em; }
#toc ol { padding-left: 1.4em; margin: 0; }
#toc li { margin: 0.5em 0; font-size: 0.94em; }
#toc a { color: var(--accent); text-decoration: none; }
#toc a:hover { text-decoration: underline; }
#toc .footer { margin-top: 2em; padding-top: 1em; border-top: 1px solid var(--rule);
    font-size: 0.85em; color: var(--fg-soft); }
#toc .footer a { color: var(--accent); display: block; margin: 0.3em 0; }

#main { padding: 3em 4em; max-width: 1200px; }
.page-header {
    border-bottom: 2px solid var(--accent); padding-bottom: 1.2em; margin-bottom: 1.6em;
}
.page-header .eyebrow {
    text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.78em;
    color: var(--fg-soft); margin: 0 0 0.4em 0;
}
.page-header h1 { margin: 0 0 0.4em 0; color: var(--accent); font-size: 2.0em; }
.page-header .lead { color: var(--fg-soft); font-size: 1.05em; margin: 0; }

h2 {
    margin: 2.6em 0 0.6em 0; padding-bottom: 0.3em;
    border-bottom: 1px solid var(--rule); color: var(--accent);
    font-size: 1.4em;
}
h2:first-of-type { margin-top: 0.5em; }
h3 { margin: 1.6em 0 0.4em 0; color: var(--fg); font-size: 1.05em; }

p { margin: 0.5em 0; }

.grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 1.2em; margin: 1em 0;
}
figure {
    margin: 0; background: var(--bg-card); border: 1px solid var(--rule);
    border-radius: 6px; padding: 0.6em; display: flex; flex-direction: column;
}
figure img, figure video {
    max-width: 100%; border-radius: 4px; cursor: zoom-in;
    background: #fff; align-self: center;
}
figure img:fullscreen, figure img:-webkit-full-screen {
    max-width: 100vw; max-height: 100vh; object-fit: contain;
    background: #000; cursor: zoom-out; border: 0; border-radius: 0;
}
figcaption {
    font-size: 0.82em; color: var(--fg-soft); margin-top: 0.5em;
    font-family: "SF Mono", Menlo, Consolas, monospace;
}
figcaption .src { color: var(--accent); }

table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em;
    background: var(--bg-card); }
th, td { border: 1px solid var(--rule); padding: 0.45em 0.7em; text-align: left;
    vertical-align: top; }
th { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }

code, .path { background: #f0f0f0; padding: 0 0.3em; border-radius: 3px;
    font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.9em; }

.tag { display: inline-block; padding: 0.1em 0.55em; margin: 0.1em 0.2em 0.1em 0;
    background: var(--tag-bg); color: var(--tag); border-radius: 10px;
    font-size: 0.78em; }

ul.bare { list-style: none; padding-left: 0; }
ul.bare li { padding: 0.2em 0; font-size: 0.92em; }

.note {
    background: var(--accent-soft); border-left: 4px solid var(--accent);
    padding: 0.6em 1em; border-radius: 4px; margin: 1em 0;
    font-size: 0.92em; color: var(--fg);
}
"""

JS = """
document.addEventListener('click', function(ev) {
    var t = ev.target;
    if (t && (t.tagName === 'IMG')) {
        if (t.requestFullscreen) t.requestFullscreen();
        else if (t.webkitRequestFullscreen) t.webkitRequestFullscreen();
    }
});
"""


def fig(rel: str, caption: str = "", source: str = "") -> str:
    """Render an image/gif as a figure. `rel` is path relative to OUT's parent."""
    cap = escape(caption) if caption else escape(rel)
    src_html = (
        f' <span class="src">[{escape(source)}]</span>' if source else ""
    )
    return (
        f'<figure><img src="{escape(rel)}" alt="{escape(rel)}" loading="lazy">'
        f'<figcaption>{cap}{src_html}</figcaption></figure>'
    )


def grid(*figs: str) -> str:
    return '<div class="grid">' + "\n".join(figs) + "</div>"


def kvtable(rows: list[tuple[str, str]], headers: tuple[str, str] = ("key", "value")) -> str:
    h = f"<thead><tr><th>{escape(headers[0])}</th><th>{escape(headers[1])}</th></tr></thead>"
    body = "\n".join(
        f"<tr><td><code>{escape(k)}</code></td><td class='num'>{escape(str(v))}</td></tr>"
        for k, v in rows
    )
    return f"<table>{h}<tbody>{body}</tbody></table>"


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def flatten_for_table(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.extend(flatten_for_table(v, key))
        elif isinstance(v, list):
            if len(v) <= 6 and all(isinstance(x, (int, float)) for x in v):
                out.append((key, "[" + ", ".join(fmt(x) for x in v) + "]"))
            else:
                out.append((key, f"<list len={len(v)}>"))
        else:
            out.append((key, fmt(v)))
    return out


def json_table(path: pathlib.Path, max_rows: int = 80) -> str:
    if not path.exists():
        return f"<p><em>missing: {escape(str(path))}</em></p>"
    data = json.loads(path.read_text())
    rows = (
        flatten_for_table(data) if isinstance(data, dict)
        else [(f"[{i}]", fmt(v)) for i, v in enumerate(data)]
    )
    if len(rows) > max_rows:
        rows = rows[:max_rows] + [("...", f"<truncated, {len(rows) - max_rows} more>")]
    return kvtable(rows, headers=(f"key — {path.name}", "value"))


def section(sid: str, title: str, body: str) -> str:
    return f'<section id="{sid}"><h2>{escape(title)}</h2>{body}</section>'


def list_experiment_dirs() -> list[tuple[str, dict]]:
    rows = []
    for d in sorted(EXP.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith(".") or d.name == "Backup":
            continue
        info = {
            "checkpoint": (d / "checkpoint.npz").exists(),
            "metrics": (d / "metrics.npz").exists(),
            "report": (d / "report.html").exists(),
            "gifs": len(list(d.glob("*.gif"))),
            "npz": len(list(d.glob("*.npz"))),
            "png": len(list(d.glob("*.png"))),
        }
        rows.append((d.name, info))
    return rows


# ---------- section builders ----------


def s_inventory() -> str:
    rows = list_experiment_dirs()
    head = (
        "<thead><tr><th>directory</th>"
        "<th>ckpt</th><th>metrics</th><th>report</th>"
        "<th>gifs</th><th>npz</th><th>png</th></tr></thead>"
    )

    def cell(b: bool) -> str:
        return "&#10004;" if b else "&middot;"

    body = "\n".join(
        f"<tr><td><code>experiments/{escape(name)}</code></td>"
        f"<td class='num'>{cell(info['checkpoint'])}</td>"
        f"<td class='num'>{cell(info['metrics'])}</td>"
        f"<td class='num'>{cell(info['report'])}</td>"
        f"<td class='num'>{info['gifs'] or '&middot;'}</td>"
        f"<td class='num'>{info['npz'] or '&middot;'}</td>"
        f"<td class='num'>{info['png'] or '&middot;'}</td></tr>"
        for name, info in rows
    )
    intro = (
        "<p>Every experiment directory under <code>experiments/</code>. "
        "Columns mark which canonical artifacts each one contains. "
        "<code>&middot;</code> = absent. The <code>meta-report/</code> directory "
        "is a shared visualisation pool — not an experiment of its own.</p>"
    )
    return intro + f"<table>{head}<tbody>{body}</tbody></table>"


def s_cooperation() -> str:
    figs = []
    intro = (
        "<p>Plots and GIFs from cooperation-only experiments "
        "(no red, varying N and grid size). Source images live in their "
        "respective experiment directories — paths shown in each caption.</p>"
    )

    pair_dir = EXP / "pair-cooperate-coop"
    if (pair_dir / "episode.gif").exists():
        figs.append(fig(
            f"pair-cooperate-coop/episode.gif",
            "pair-cooperate-coop / N=2 / 10×10",
            "experiments/pair-cooperate-coop/episode.gif",
        ))

    quad32 = EXP / "quad-cooperate-coop-32"
    if (quad32 / "episode.gif").exists():
        figs.append(fig(
            "quad-cooperate-coop-32/episode.gif",
            "quad-cooperate-coop-32 / N=4 / 32×32 (warm-start ladder)",
            "experiments/quad-cooperate-coop-32/episode.gif",
        ))

    octa = EXP / "octa-cooperate-coop-32-r6-conn"
    if (octa / "episode.gif").exists():
        figs.append(fig(
            "octa-cooperate-coop-32-r6-conn/episode.gif",
            "octa N=8 / 32×32 / connectivity-restore variant",
            "experiments/octa-cooperate-coop-32-r6-conn/episode.gif",
        ))

    survey_dir = EXP
    for name in [
        "survey-local-32-N8/episode.gif",
        "survey-local-32-N8-spread/episode.gif",
        "survey-local-32-N8-spread-chained/episode.gif",
        "survey-local-16fast-N5-fog/episode.gif",
    ]:
        p = EXP / name
        if p.exists():
            figs.append(fig(name, name.replace("/", " — "), f"experiments/{name}"))

    body = intro + grid(*figs) if figs else intro
    return body


def s_compromise() -> str:
    intro = (
        "<p>Compromise / k*(θ) sweeps. Plots come from "
        "<code>scripts/budget_visualizations.py</code>, "
        "<code>scripts/compromise_visualizations.py</code>, "
        "<code>scripts/exp_visualizations.py</code> "
        "(generators in <code>scripts/</code>; outputs cached in "
        "<code>experiments/meta-report/</code>).</p>"
    )

    plots = [
        ("meta-report/kstar_staircase.png", "k*(θ) staircase", "scripts/compromise_visualizations.py"),
        ("meta-report/compromise_compare.png", "compromise compare", "scripts/compromise_visualizations.py"),
        ("meta-report/comparison_matrix.png", "comparison matrix", "scripts/compromise_visualizations.py"),
        ("meta-report/budget_pareto.png", "budget Pareto", "scripts/budget_visualizations.py"),
        ("meta-report/budget_curves.png", "budget curves", "scripts/budget_visualizations.py"),
        ("meta-report/budget_heatmap.png", "budget heatmap (k × ρ)", "scripts/budget_visualizations.py"),
        ("meta-report/budget_match_diff.png", "matched-budget difference", "scripts/budget_visualizations.py"),
        ("meta-report/budget_raincloud_grid.png", "budget raincloud grid", "scripts/budget_visualizations.py"),
        ("meta-report/budget_surface.png", "budget surface", "scripts/budget_visualizations.py"),
        ("meta-report/forest_delta_j.png", "forest plot ΔJ", "scripts/exp_visualizations.py"),
        ("meta-report/coverage_curves.png", "coverage curves", "scripts/exp_visualizations.py"),
        ("meta-report/claim1_invariant.png", "claim-1 invariant view", "scripts/exp_visualizations.py"),
        ("meta-report/claims_evidence.png", "claims-evidence stacked", "scripts/exp_visualizations.py"),
    ]
    figs = [fig(p, c, s) for p, c, s in plots if (EXP / p).exists()]

    table = json_table(META / "exp_summary.json")
    return intro + table + grid(*figs)


def s_hetero() -> str:
    intro = (
        "<p>Heterogeneous-ρ misbehavior-budget sweep. Source: "
        "<code>scripts/hetero_visualizations.py</code> + "
        "<code>experiments/misbehavior-budget/</code>.</p>"
    )
    figs = []
    for p, c in [
        ("meta-report/hetero_sweep.png", "hetero-ρ sweep"),
        ("meta-report/exp_ridgeline.png", "exp ridgeline"),
        ("meta-report/exp_seed_swim.png", "seed swim"),
        ("meta-report/exp_qq.png", "QQ plot"),
        ("meta-report/exp_corr_matrix.png", "exp corr matrix"),
        ("meta-report/exp_parallel_coords.png", "parallel coords"),
        ("meta-report/exp_polar_clock.png", "polar clock"),
        ("meta-report/exp_streamgraph.png", "streamgraph"),
        ("meta-report/exp_waterfall.png", "waterfall"),
        ("meta-report/exp_small_multiples.png", "small multiples"),
        ("meta-report/exp_bump_chart.png", "bump chart"),
        ("meta-report/exp_rho_density.png", "ρ density"),
        ("meta-report/exp_budget_surface.png", "exp budget surface"),
    ]:
        if (EXP / p).exists():
            figs.append(fig(p, c, p.replace("meta-report/", "experiments/meta-report/")))
    table = json_table(META / "hetero_summary.json")
    return intro + table + grid(*figs)


def s_red_prior() -> str:
    intro = (
        "<p>Red-prior Phase 1 — same-class red, three priors I/W/F, "
        "frozen blue at C2 (16×16, 3b2r), 3 seeds × 3000 episodes. "
        "Source data: <code>experiments/red-prior-phase1/</code>.</p>"
    )
    figs = [
        fig("red-prior-phase1/episode_I.gif", "I — Insider (full blue copy)",
            "experiments/red-prior-phase1/episode_I.gif"),
        fig("red-prior-phase1/episode_W.gif", "W — Warm-start (Dense_0 only)",
            "experiments/red-prior-phase1/episode_W.gif"),
        fig("red-prior-phase1/episode_F.gif", "F — Fresh (random init)",
            "experiments/red-prior-phase1/episode_F.gif"),
    ]
    npz_path = PHASE1 / "eval_stats.npz"
    table_html = ""
    if npz_path.exists():
        try:
            import numpy as np
            d = np.load(npz_path, allow_pickle=False)
            rows = []
            for k in sorted(d.files):
                a = d[k]
                if a.ndim == 0:
                    rows.append((k, fmt(a.item())))
                elif a.size <= 8:
                    rows.append((k, "[" + ", ".join(fmt(x) for x in a.flatten()) + "]"))
                else:
                    rows.append((k, f"<array shape={a.shape} dtype={a.dtype} mean={float(np.mean(a)):.4g} std={float(np.std(a)):.4g}>"))
            table_html = kvtable(rows, headers=("key — eval_stats.npz", "value"))
        except Exception as e:
            table_html = f"<p><em>could not load eval_stats.npz: {escape(str(e))}</em></p>"

    return intro + table_html + grid(*figs)


def s_coevo() -> str:
    intro = (
        "<p>Coevolution-related artifacts — adversarial ladder rungs, "
        "ES r6 coevo, eps sweeps. GIF rollouts come straight from each "
        "rung's <code>episode.gif</code>.</p>"
    )
    figs = []
    for sub in [
        "adv-ladder-r1-6x6-1b1r",
        "adv-ladder-r2-8x8-2b1r",
        "adv-ladder-r3-16x16-3b1r",
        "adv-ladder-r4-16x16-4b3r",
        "adv-ladder-r5-32x32-7b3r",
        "adv-ladder-r6-32x32-6b4r",
        "adv-ladder-r6-coevo",
    ]:
        gif = EXP / sub / "episode.gif"
        if gif.exists():
            figs.append(fig(f"{sub}/episode.gif", sub, f"experiments/{sub}/episode.gif"))
    return intro + grid(*figs)


def s_compromise_setups() -> str:
    intro = (
        "<p>Per-setup rollouts at the canonical compromise configs. "
        "B = baseline (cooperation only), C1 = k=1, C2 = k=2.</p>"
    )
    figs = []
    for tag in ["B", "C1", "C2"]:
        path = f"meta-report/episode_{tag}.gif"
        if (EXP / path).exists():
            figs.append(fig(path, f"setup {tag}", f"experiments/{path}"))
    iso = []
    for tag in ["B", "C1", "C2"]:
        for view in ["blue", "global", "red", "sabotage", "uncertainty"]:
            p = f"meta-report/iso_{tag}_{view}.gif"
            if (EXP / p).exists():
                iso.append(fig(p, f"isometric {tag} / {view}", f"experiments/{p}"))
    iso_compare = []
    for view in ["blue", "global", "sabotage", "uncertainty"]:
        p = f"meta-report/iso_compare_{view}.gif"
        if (EXP / p).exists():
            iso_compare.append(fig(p, f"iso compare / {view}", f"experiments/{p}"))

    iso_table = json_table(META / "iso_time_to_threshold.json")

    return (
        intro
        + "<h3>baseline / C1 / C2 episode rollouts</h3>"
        + grid(*figs)
        + "<h3>isometric per-setup views</h3>"
        + grid(*iso)
        + "<h3>cross-setup isometric comparison</h3>"
        + grid(*iso_compare)
        + "<h3>iso_time_to_threshold.json</h3>"
        + iso_table
        + "<h3>where do coverage gaps live on the map?</h3>"
        + grid(*[fig(p, c, f"experiments/{p}") for p, c in [
            ("meta-report/uncovered_heatmap.png",
             "P(cell uncovered at t=200) per setup, with red & blue position contours overlaid"),
        ] if (EXP / p).exists()])
        + "<h3>uncovered_heatmap.json</h3>"
        + json_table(META / "uncovered_heatmap.json", max_rows=12)
        + "<h3>per-cell visit counts (3 views: blue / red / dominance)</h3>"
        + grid(*[fig(p, c, f"experiments/{p}") for p, c in [
            ("meta-report/visit_grids.png",
             "left = blue total visits, middle = red total visits, "
             "right = per-agent dominance (blue−red)"),
        ] if (EXP / p).exists()])
        + "<h3>visit_grids.json</h3>"
        + json_table(META / "visit_grids.json", max_rows=12)
        + "<h3>other map / time views</h3>"
        + grid(
            *[fig(p, c, f"experiments/{p}") for p, c in [
                ("meta-report/channels_stacked.png", "channels stacked"),
                ("meta-report/fog_footprint.png", "fog footprint (legacy 1-shot map)"),
                ("meta-report/resilience_triangle.png", "resilience triangle"),
                ("meta-report/time_to_coverage_multiseed.png", "time-to-coverage multi-seed"),
                ("meta-report/spacetime_entropy.png", "spacetime entropy"),
                ("meta-report/spacetime_uncertainty.png", "spacetime uncertainty"),
                ("meta-report/spacetime_tubes.png", "spacetime tubes"),
                ("meta-report/trajectories.png", "trajectories"),
                ("meta-report/variance_bar.png", "variance bar"),
            ] if (EXP / p).exists()]
        )
        + "<h3>spacetime_entropy_summary.json</h3>"
        + json_table(META / "spacetime_entropy_summary.json")
        + "<h3>spacetime_uncertainty_summary.json</h3>"
        + json_table(META / "spacetime_uncertainty_summary.json")
    )


def s_xai() -> str:
    intro = (
        "<p>openthebox / mechanism / XAI artifacts. Plots from "
        "<code>scripts/openthebox.py</code> and <code>scripts/xai_*.py</code>; "
        "JSON tables show the raw saliency / probe / occlusion / IG numbers.</p>"
    )
    plots = [
        ("openthebox_probes_grid.png", "linear probes — accuracy grid"),
        ("openthebox_block_attribution.png", "block attribution"),
        ("openthebox_block_occlusion.png", "block occlusion"),
        ("openthebox_per_cell_occlusion.png", "per-cell occlusion"),
        ("openthebox_identity_swap.png", "identity swap KL"),
        ("openthebox_pca_manifold.png", "PCA manifold"),
        ("openthebox_method_correlation.png", "method correlation"),
        ("openthebox_cross_summary.png", "cross-setup summary"),
        ("openthebox_tcav.png", "TCAV"),
        ("xai_block_stack.png", "block stack"),
        ("xai_block_team_means.png", "block team means"),
        ("xai_identity_swap.png", "xai identity swap"),
        ("xai_ig_team_means.png", "IG team means"),
        ("xai_ig_vs_saliency.png", "IG vs saliency"),
        ("xai_occlusion_coverage.png", "occlusion coverage"),
        ("xai_occlusion_kl.png", "occlusion KL"),
        ("xai_probes_accuracy.png", "probes accuracy"),
        ("xai_red_self_vs_cross.png", "red self vs cross"),
        ("xai_spatial_seen.png", "spatial seen"),
        ("model_decomposition.png", "model decomposition"),
        ("system_diagram.png", "system diagram"),
    ]
    figs = []
    for name, cap in plots:
        rel = f"meta-report/{name}"
        if (EXP / rel).exists():
            figs.append(fig(rel, cap, f"experiments/{rel}"))

    return (
        intro
        + grid(*figs)
        + "<h3>openthebox_summary.json</h3>"
        + json_table(META / "openthebox_summary.json")
        + "<h3>xai_summary.json</h3>"
        + json_table(META / "xai_summary.json")
        + "<h3>xai_causal_summary.json</h3>"
        + json_table(META / "xai_causal_summary.json")
        + "<h3>xai_ig_summary.json</h3>"
        + json_table(META / "xai_ig_summary.json")
    )


def s_preserved() -> str:
    """Pre-existing reports we kept intact (not authored in this thread)."""
    items = [
        ("meta-report/openthebox.html", "openthebox.html — generated by scripts/openthebox.py"),
        ("meta-report/engineering_retrospective.html", "engineering_retrospective.html"),
        ("meta-report/meta_report_v3.html", "meta_report_v3.html (legacy)"),
        ("meta-report/architecture.txt", "architecture.txt"),
        ("README.md", "experiments/README.md (operator's manual)"),
    ]
    rows = []
    for rel, label in items:
        path = EXP / rel
        if path.exists():
            rows.append(
                f'<li><a href="{escape(rel)}">{escape(label)}</a> '
                f'<span class="tag">untouched</span></li>'
            )
    return (
        "<p>Documents that already existed and have not been edited or "
        "interpreted in this review.</p>"
        f"<ul class='bare'>{''.join(rows)}</ul>"
    )


# ---------- assembly ----------


SECTIONS = [
    ("inventory", "Inventory — every experiment directory", s_inventory),
    ("cooperation", "Cooperation rollouts", s_cooperation),
    ("compromise-setups", "B / C1 / C2 setups — episode + isometric + spacetime", s_compromise_setups),
    ("compromise", "Compromise & budget sweeps", s_compromise),
    ("hetero", "Hetero-ρ misbehavior budget", s_hetero),
    ("xai", "openthebox / mechanism / XAI", s_xai),
    ("coevo", "Adversarial ladder & coevolution", s_coevo),
    ("red-prior", "Red prior Phase 1 (I / W / F)", s_red_prior),
    ("preserved", "Pre-existing documents (untouched)", s_preserved),
]


def build() -> None:
    nav = "\n".join(
        f'    <li><a href="#{sid}">{escape(title)}</a></li>'
        for sid, title, _ in SECTIONS
    )
    body = "\n".join(section(sid, title, fn()) for sid, title, fn in SECTIONS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Umbrella Review &mdash; RedWithinBlue</title>
<style>{CSS}</style>
</head>
<body>
<div id="layout">
<nav id="toc">
  <h2>Umbrella Review</h2>
  <p class="subtitle">Evidence atlas. No conclusions.</p>
  <ol>
{nav}
  </ol>
  <div class="footer">
    <p><b>Pre-existing</b></p>
    <a href="meta-report/openthebox.html">openthebox.html</a>
    <a href="meta-report/engineering_retrospective.html">engineering_retrospective.html</a>
    <a href="meta-report/meta_report_v3.html">meta_report_v3.html</a>
    <a href="README.md">README.md</a>
  </div>
</nav>
<main id="main">
<header class="page-header">
  <p class="eyebrow">RedWithinBlue</p>
  <h1>Umbrella Review</h1>
  <p class="lead">All artifacts. Plots, GIFs, raw JSON. No interpretation, no claims, no bottom-line callouts. Click any image to fullscreen.</p>
</header>
{body}
</main>
</div>
<script>{JS}</script>
</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html):,} chars, {len(SECTIONS)} sections)")


if __name__ == "__main__":
    build()
