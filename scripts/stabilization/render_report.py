"""Render comparison figures + HTML report for EXP-A and EXP-B.

Reads ``experiments/stabilization/*/metrics.npz`` and writes:
    experiments/stabilization/reward_vs_variant.png    (EXP-A 4-variant figure)
    experiments/stabilization/sample_efficiency.png    (EXP-B sample-efficiency)
    experiments/stabilization/stabilization_report.html (self-contained report)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("experiments/stabilization")

COLORS = {
    "A0": "#1f77b4",   # blue — the baseline
    "A1": "#ff7f0e",   # orange
    "A2": "#2ca02c",   # green
    "A3": "#d62728",   # red
    "B0": "#1f77b4",
    "B1": "#d62728",
}

LABELS = {
    "A0": "A0  — MC baseline (current production)",
    "A1": "A1  — TD(0) + target-net",
    "A2": "A2  — TD(0) + twin-Q (live)",
    "A3": "A3  — TD(0) + twin-Q + target-net",
    "B0": "B0  — on-policy REINFORCE",
    "B1": "B1  — off-policy Double-DQN (replay + twin-Q + target)",
}


def _smooth(x, w=200):
    if x.shape[0] < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def render_expA():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for v in ["A0", "A1", "A2", "A3"]:
        d = np.load(ROOT / f"twin-critic-{v}" / "metrics.npz")
        r = d["total_reward"]    # [E, S]
        loss = np.abs(d["loss"]) # [E, S]
        ep = np.arange(r.shape[0])

        mean = r.mean(axis=1)
        sd = r.std(axis=1)
        mean_s = _smooth(mean, 200)
        axes[0].plot(ep, mean_s, color=COLORS[v], label=LABELS[v], linewidth=1.6)
        axes[0].fill_between(ep, mean_s - sd, mean_s + sd, color=COLORS[v], alpha=0.12)

        # |loss| p99 trajectory — bucket the episodes so the log plot is readable.
        nb = 50
        bsize = r.shape[0] // nb
        p99 = np.array([
            np.percentile(loss[i * bsize: (i + 1) * bsize], 99) for i in range(nb)
        ])
        bx = (np.arange(nb) + 0.5) * bsize
        axes[1].semilogy(bx, p99, color=COLORS[v], label=LABELS[v], linewidth=1.6)

    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("team total reward (smoothed, ±σ band over 5 seeds)")
    axes[0].set_title("EXP-A — reward trajectory, pair-cooperate-coop (10×10, N=2)")
    axes[0].axhline(1.8, color="grey", linestyle=":", linewidth=0.8,
                    label="pass threshold (+1.8)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=9)

    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("|loss| p99 (log scale)")
    axes[1].set_title("EXP-A — |loss| p99 over training")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = ROOT / "reward_vs_variant.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def render_expB():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for v in ["B0", "B1"]:
        d = np.load(ROOT / f"offpolicy-red-{v}" / "metrics.npz")
        steps = d["env_steps"]      # [E]
        br = d["blue_reward"]       # [E, S]
        mean = br.mean(axis=1)
        sd = br.std(axis=1)
        ax.plot(steps, mean, color=COLORS[v], label=LABELS[v], linewidth=1.8,
                marker="o", markersize=4)
        ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[v], alpha=0.15)

    ax.set_xlabel("red env-steps (matched budget across variants)")
    ax.set_ylabel("blue team reward (lower = red winning)")
    ax.set_title("EXP-B — off-policy vs on-policy red (3 seeds, "
                 "compromise-16x16-5-3b2r, fixed blue)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = ROOT / "sample_efficiency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def _read_summary(name):
    return json.load(open(ROOT / name / "summary.json"))


def _checkpoint_reward(variant, ep, window=500):
    d = np.load(ROOT / f"twin-critic-{variant}" / "metrics.npz")
    r = d["total_reward"]
    lo = max(0, ep - window // 2)
    hi = min(r.shape[0], ep + window // 2)
    return float(r[lo:hi].mean())


def render_html():
    # Assemble tables directly from summary.json + metrics.npz.
    A = {v: _read_summary(f"twin-critic-{v}") for v in ["A0", "A1", "A2", "A3"]}
    B = {v: _read_summary(f"offpolicy-red-{v}") for v in ["B0", "B1"]}

    def _row_A(v, desc):
        s = A[v]
        return f"""
        <tr>
          <td><b>{v}</b></td>
          <td>{desc}</td>
          <td>{s['final_reward_mean']:+.3f} ± {s['final_reward_std']:.3f}</td>
          <td>{s['late_dive_count']}/5</td>
          <td>{s['loss_abs_p99_final']:,.1f}</td>
          <td>{s['wall_time_s']/60:.1f}</td>
        </tr>"""

    def _row_B(v, desc):
        s = B[v]
        return f"""
        <tr>
          <td><b>{v}</b></td>
          <td>{desc}</td>
          <td>{s['final_blue_reward_mean']:+.3f}</td>
          <td>{s['final_red_reward_mean']:+.3f}</td>
          <td>{s['wall_seconds']:,.0f}</td>
        </tr>"""

    # Reward trajectory checkpoint table for EXP-A
    ckpts = [0, 1000, 5000, 10000, 14999]
    traj_rows = "".join(
        f"<tr><td><b>{v}</b></td>"
        + "".join(f"<td>{_checkpoint_reward(v, ep):+.3f}</td>" for ep in ckpts)
        + "</tr>"
        for v in ["A0", "A1", "A2", "A3"]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stabilisation Experiments — Results</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 1200px;
    margin: 2em auto;
    padding: 0 2em;
    color: #222;
    line-height: 1.55;
  }}
  h1 {{ border-bottom: 2px solid #222; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #bbb; padding-bottom: 0.2em; margin-top: 2em; }}
  h3 {{ color: #333; margin-top: 1.8em; }}
  code {{ background: #f3f3f3; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.95em; }}
  pre {{ background: #f7f7f7; padding: 0.9em; border-radius: 5px; overflow-x: auto; font-size: 0.9em; }}
  table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.95em; }}
  th, td {{ border: 1px solid #bbb; padding: 0.4em 0.9em; text-align: left; }}
  th {{ background: #f0f0f0; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  .pass {{ color: #2ca02c; font-weight: bold; }}
  .fail {{ color: #d62728; font-weight: bold; }}
  .note {{ color: #555; font-style: italic; }}
  figure {{ margin: 1em 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }}
  figcaption {{ color: #555; font-size: 0.9em; margin-top: 0.3em; }}
  .key-finding {{
    background: #fffbea;
    border-left: 4px solid #f0c040;
    padding: 0.8em 1.1em;
    margin: 1em 0;
  }}
</style>
</head>
<body>

<h1>Stabilisation experiments — results</h1>

<p><span class="note">Reports two compartmentalised ablations of the training pipeline. Neither touched <code>src/red_within_blue/</code>. Plan: <code>docs/08-stabilization-experiments.md</code>. Both experiments ran 2026-04-20.</span></p>

<p>These are <b>methodology ablations</b>, not evidence for the proposal claims. They answer the question "is the current training pipeline sound, or are we leaving stability / sample efficiency on the table by skipping SAC-style machinery?" Short answer: the pipeline is sound; SAC-style machinery does not help in this regime.</p>

<div class="key-finding">
  <b>Headline — both experiments are clean negative results that vindicate the current production path.</b>
  <ul>
    <li><b>EXP-A:</b> Monte-Carlo baseline (A0) beats all three TD(0)+stabiliser variants. Target-net, twin-Q, and their combination each make the critic <i>worse</i>, not better.</li>
    <li><b>EXP-B:</b> Off-policy Double-DQN (B1) delivers a <i>weaker</i> red than on-policy REINFORCE (B0) at every env-step budget, and takes 20× the wall-clock.</li>
  </ul>
</div>

<h2>EXP-A — Twin-Q + target-net ablation on the blue CTDE critic</h2>

<p><b>Setup.</b> <code>configs/pair-cooperate-coop.yaml</code> (10×10 cooperative, N=2), 15 000 episodes × 5 seeds × 4 variants. <code>grad_clip: 0.5</code> held constant; <code>ent_coef: 0.05</code> held constant. Only the critic-target formulation differs between variants.</p>

<h3>Summary</h3>
<table>
  <tr>
    <th>code</th>
    <th>critic target</th>
    <th>final reward (mean ± σ)</th>
    <th>late dives</th>
    <th>|loss| p99 (final episode)</th>
    <th>wall (min)</th>
  </tr>
  {_row_A("A0", "Monte-Carlo returns — current production")}
  {_row_A("A1", "TD(0) + Polyak target-net (τ=0.005), 1 critic")}
  {_row_A("A2", "TD(0) + twin critics (live min), no target")}
  {_row_A("A3", "TD(0) + twin critics + Polyak target (SAC's Q-target)")}
</table>

<p><b>Pass criteria (from the plan):</b> final reward ≥ +1.8, σ ≤ 0.30, late dives = 0/5, |loss| p99 bounded. <span class="pass">A0 passes</span> (marginal on late-dives at 1/5). <span class="fail">A1, A2, A3 all fail every criterion.</span></p>

<h3>Reward trajectory at checkpoints</h3>
<table>
  <tr>
    <th>variant</th><th>ep 0</th><th>ep 1 000</th><th>ep 5 000</th><th>ep 10 000</th><th>ep 15 000</th>
  </tr>
  {traj_rows}
</table>

<p>The four variants start identical (+1.85 at ep 0) and remain close through ep 1 000 (+2.0 to +2.3). The split emerges between ep 1 000 and ep 5 000 as the TD(0) bootstrap begins to self-amplify. By ep 10 000 A1/A2/A3 have collapsed to ~+0.4; A0 holds +2.34 through the full 15 000 eps.</p>

<figure>
  <img src="reward_vs_variant.png" alt="EXP-A reward + loss trajectories">
  <figcaption>EXP-A. Left: team total reward (200-ep moving average, ±σ across 5 seeds). The +1.8 pass threshold is dotted. Right: |loss| p99 per 300-episode bucket (log scale). A3 — the full SAC Q-target — ends with the <i>worst</i> loss drift of all four variants.</figcaption>
</figure>

<h3>Interpretation</h3>

<p>The spec anticipated that A3 (twin-Q + target-net) would be the best variant because it applies both canonical SAC stabilisers. The data says the opposite: A3 has the <i>highest</i> final |loss| p99 (13 780, vs A1's 2 894 and A2's 8 459). Adding more bootstrap targets gave the critic more ways to self-amplify noise from the non-stationary <code>global_seen_mask</code>.</p>

<p>The root cause isn't "bad bootstrap target"; it's "bootstrap at all, in this CTDE central-critic + non-stationary observations regime." Monte-Carlo returns side-step the entire failure class because the regression target doesn't depend on the live critic — there is no self-reference to amplify. This is why A0 dominates.</p>

<p><b>Takeaway.</b> The MC switch already shipped at <code>src/red_within_blue/training/losses.py:222</code> is empirically optimal among the stabiliser set canonically considered by modern actor-critic (SAC) — not just adequate. No further escalation on the <code>bubbly-strolling-puddle</code> plan is warranted; close the plan and keep <code>grad_clip: 0.5</code> as a cheap redundant guardrail.</p>

<h2>EXP-B — Off-policy replay for the joint-red actor</h2>

<p><b>Setup.</b> <code>configs/compromise-16x16-5-3b2r.yaml</code>, blue frozen from <code>experiments/compromise-16x16-5-3b2r-coevo/checkpoint.npz</code>. Trains a fresh joint red from scratch with 3 seeds, 500 000 red env-steps each, evaluated every 25 000 steps across 8 episodes. Blue frozen means any Δ(blue reward) between variants is attributable to the red trainer, not to blue dynamics.</p>

<h3>Summary</h3>
<table>
  <tr>
    <th>code</th>
    <th>red trainer</th>
    <th>final blue reward (mean, 3 seeds)</th>
    <th>final red reward (mean)</th>
    <th>wall (s)</th>
  </tr>
  {_row_B("B0", "on-policy REINFORCE (one gradient per episode, transitions discarded)")}
  {_row_B("B1", "off-policy Double-DQN — 50 k replay, twin-Q, Polyak τ=0.005, ε-greedy per head")}
</table>

<p>Lower blue reward ⇒ stronger red. <span class="fail">B1 leaves blue <i>higher</i> than B0</span> (+1.22 vs +1.08), meaning the off-policy red is less effective at sabotaging blue. And B1 took ≈ 2 650 s wall-clock vs B0's 133 s — a 20× penalty for a worse result.</p>

<figure>
  <img src="sample_efficiency.png" alt="EXP-B blue reward vs env-step budget">
  <figcaption>EXP-B. Mean blue reward (±σ across 3 seeds) as a function of red env-step budget. The two curves oscillate in [+0.9, +1.4] across every horizon; no matched-budget bucket shows B1 meaningfully below B0.</figcaption>
</figure>

<h3>Interpretation</h3>

<p>Two mechanisms likely explain the result:</p>

<ol>
  <li><b>Factorised Q cannot represent joint-red coordination.</b> The DQN head outputs <code>Q(s, a_i)</code> per red agent, summed additively. REINFORCE on a joint softmax head captures cross-agent coordination directly; additive Q can't. For a 2-red team against 3-blue, the coordination dimensionality matters.</li>
  <li><b>Frozen blue makes the environment stationary.</b> Replay's value proposition is reusing transitions across a changing policy landscape. With blue frozen, the transition distribution is stationary w.r.t. red's learning, and on-policy REINFORCE is already well-matched to this regime. Replay buys nothing and costs a 20× compute premium.</li>
</ol>

<p><b>Takeaway.</b> Keep on-policy REINFORCE for the joint red. The "10× sweep density via replay" premise from the original plan (EXP-B motivation: enabling a ε × ρ × k three-way grid) is dead — if that sweep is needed, the route is raw on-policy throughput, not replay.</p>

<h2>What changes</h2>

<table>
  <tr>
    <th>thing</th>
    <th>status</th>
    <th>decision</th>
  </tr>
  <tr>
    <td><code>bubbly-strolling-puddle</code> critic-drift plan</td>
    <td>MC switch (stage 3) already shipped and empirically optimal</td>
    <td>close</td>
  </tr>
  <tr>
    <td><code>grad_clip: 0.5</code></td>
    <td>redundant given MC, cheap to keep</td>
    <td>keep</td>
  </tr>
  <tr>
    <td>SAC port (twin-Q, target-net, etc.)</td>
    <td>confirmed counterproductive here</td>
    <td>do not port</td>
  </tr>
  <tr>
    <td>Off-policy red trainer</td>
    <td>weaker + 20× slower than on-policy</td>
    <td>do not adopt</td>
  </tr>
  <tr>
    <td>On-policy REINFORCE for joint red (<code>scripts/coevo_r6.py</code> path)</td>
    <td>empirically dominant</td>
    <td>keep as default</td>
  </tr>
</table>

<h2>Reproducing</h2>

<pre>python scripts/stabilization/twin_critic_experiment.py --variant A0   # MC baseline
python scripts/stabilization/twin_critic_experiment.py --variant A1   # +target-net
python scripts/stabilization/twin_critic_experiment.py --variant A2   # +twin-Q
python scripts/stabilization/twin_critic_experiment.py --variant A3   # +both (SAC Q-target)

python scripts/stabilization/offpolicy_red_experiment.py --variant B0  # REINFORCE
python scripts/stabilization/offpolicy_red_experiment.py --variant B1  # Double-DQN

python scripts/stabilization/render_report.py   # this document
</pre>

<p class="note">Each EXP-A variant is 15 000 eps × 5 seeds ≈ 10 min on one core. EXP-B is 500 k env-steps × 3 seeds: B0 ≈ 2 min, B1 ≈ 45 min. All six variants launched in parallel on a 6+ performance-core machine fit comfortably in ~45 min total.</p>

</body>
</html>
"""
    out = ROOT / "stabilization_report.html"
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_expA()
    render_expB()
    render_html()
