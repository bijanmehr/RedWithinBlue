"""Analyze adversarial Phase 1 experiments and produce the threat model.

Reads metrics from ``experiments/adv-*/metrics.npz`` and emits:

1. ``experiments/_analysis/threat_degradation.png`` — blue coverage bars per
   fleet size, one group per mechanism (BASELINE-COOP / BASELINE-RAND /
   SHARED-ZSUM / DUAL-NASH).
2. ``experiments/_analysis/agent_denial.png`` — per-blue-agent final coverage
   contribution (box plot across seeds), one subplot per (N, mechanism).
3. ``experiments/_analysis/nash_diagnostic.png`` — blue_total_reward +
   red_total_reward across training for trained-red runs (zero-sum check).
4. ``docs/08-threat-model.md`` — markdown degradation table.

Usage::

    python -m red_within_blue.analysis.threat_model
    python -m red_within_blue.analysis.threat_model --experiments-dir other_dir
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FLEET_SIZES = (5, 7, 10)
N_RED_BY_N = {5: 2, 7: 3, 10: 4}
N_BLUE_BY_N = {n: n - N_RED_BY_N[n] for n in FLEET_SIZES}

MECHANISM_ORDER = ("baseline", "random", "shared", "dual")
MECHANISM_LABELS = {
    "baseline": "BASELINE-COOP (0 red)",
    "random": "BASELINE-RAND (random red)",
    "shared": "SHARED-ZSUM (shared-actor red)",
    "dual": "DUAL-NASH (centralized red)",
}


@dataclass
class RunMetrics:
    name: str
    mechanism: str
    num_agents: int
    num_red: int
    blue_total_reward: np.ndarray
    red_total_reward: np.ndarray
    per_agent_reward: np.ndarray | None


_NAME_PAT = re.compile(
    r"adv-(?P<mech>baseline|random|shared|dual)-(?P<n>\d+)(?:blue|-(?P<red>\d+)red)$"
)


def parse_name(name: str) -> tuple[str, int, int] | None:
    """Parse ``adv-baseline-5blue`` or ``adv-shared-7-3red`` → (mech, N, n_red)."""
    m = _NAME_PAT.match(name)
    if not m:
        return None
    mech = m.group("mech")
    n = int(m.group("n"))
    red = int(m.group("red")) if m.group("red") else 0
    return mech, n, red


def load_run(run_dir: Path) -> RunMetrics | None:
    """Load one experiment's metrics.npz and parse its name."""
    parsed = parse_name(run_dir.name)
    if parsed is None:
        return None
    mech, n, n_red = parsed

    metrics_path = run_dir / "metrics.npz"
    if not metrics_path.exists():
        return None

    data = np.load(metrics_path)
    blue = np.asarray(data["blue_total_reward"])
    red = np.asarray(data["red_total_reward"])
    per_agent = np.asarray(data["per_agent_reward"]) if "per_agent_reward" in data.files else None

    if blue.ndim == 1:
        blue = blue[None, :]
        red = red[None, :]
        if per_agent is not None:
            per_agent = per_agent[None, ...]

    return RunMetrics(
        name=run_dir.name,
        mechanism=mech,
        num_agents=n,
        num_red=n_red,
        blue_total_reward=blue,
        red_total_reward=red,
        per_agent_reward=per_agent,
    )


def load_all_runs(experiments_dir: Path) -> list[RunMetrics]:
    runs: list[RunMetrics] = []
    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("adv-"):
            continue
        r = load_run(run_dir)
        if r is not None:
            runs.append(r)
    return runs


def final_blue_coverage(run: RunMetrics, window: int = 100) -> tuple[float, float]:
    """Mean/std of blue coverage averaged over the last ``window`` episodes."""
    last = run.blue_total_reward[:, -window:]
    per_seed = last.mean(axis=1)
    return float(per_seed.mean()), float(per_seed.std())


def plot_degradation(runs: list[RunMetrics], out_path: Path) -> None:
    """Blue coverage per fleet size, bar per mechanism."""
    fig, axes = plt.subplots(1, len(FLEET_SIZES), figsize=(4 * len(FLEET_SIZES), 4),
                             sharey=True)
    if len(FLEET_SIZES) == 1:
        axes = [axes]

    for ax, n in zip(axes, FLEET_SIZES):
        by_mech: dict[str, tuple[float, float]] = {}
        for mech in MECHANISM_ORDER:
            match = [r for r in runs if r.mechanism == mech and r.num_agents == n]
            if match:
                by_mech[mech] = final_blue_coverage(match[0])

        labels = [MECHANISM_LABELS[m] for m in MECHANISM_ORDER if m in by_mech]
        means = [by_mech[m][0] for m in MECHANISM_ORDER if m in by_mech]
        stds = [by_mech[m][1] for m in MECHANISM_ORDER if m in by_mech]
        colors = ["#3b7dd8", "#9aa0a6", "#f4b400", "#d93025"][:len(means)]

        xs = np.arange(len(means))
        ax.bar(xs, means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"N={n} (n_red={N_RED_BY_N[n]})")
        ax.set_ylabel("Blue coverage (fraction)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Threat model: blue mission coverage under adversarial red")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_agent_denial(runs: list[RunMetrics], out_path: Path, window: int = 100) -> None:
    """Per-blue-agent final coverage contribution across seeds (box plot)."""
    relevant = [
        r for r in runs
        if r.mechanism in ("random", "shared", "dual") and r.per_agent_reward is not None
    ]
    if not relevant:
        return

    fig, axes = plt.subplots(1, len(relevant), figsize=(3 * len(relevant), 4), sharey=True)
    if len(relevant) == 1:
        axes = [axes]

    for ax, run in zip(axes, relevant):
        n_blue = run.num_agents - run.num_red
        last = run.per_agent_reward[:, -window:, :n_blue]
        per_seed_final = last.mean(axis=1)
        ax.boxplot(per_seed_final, positions=np.arange(n_blue))
        ax.set_xticks(np.arange(n_blue))
        ax.set_xticklabels([f"B{i}" for i in range(n_blue)])
        ax.set_title(run.name, fontsize=9)
        ax.set_ylabel("Coverage contribution")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-blue-agent mission denial")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_nash_diagnostic(runs: list[RunMetrics], out_path: Path) -> None:
    """For trained-red runs, plot blue+red reward across training (should → 0)."""
    trained = [r for r in runs if r.mechanism in ("shared", "dual")]
    if not trained:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for run in trained:
        total = run.blue_total_reward + run.red_total_reward
        mean = total.mean(axis=0)
        ax.plot(mean, label=f"{run.name} (N={run.num_agents}, n_red={run.num_red})")

    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("blue_total_reward + red_total_reward")
    ax.set_title("Nash / zero-sum diagnostic (trained red)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_markdown_table(runs: list[RunMetrics], out_path: Path, window: int = 100) -> None:
    """Write the threat-model degradation table."""
    table: dict[tuple[str, int], tuple[float, float]] = {}
    for run in runs:
        table[(run.mechanism, run.num_agents)] = final_blue_coverage(run, window)

    lines: list[str] = []
    lines.append("# Threat Model: Adversarial Mission Denial (32×32)\n")
    lines.append(
        "Blue coverage (fraction of discoverable cells) averaged over the last "
        f"{window} episodes, mean ± std across seeds.\n"
    )
    lines.append("")
    lines.append("| Fleet (N / n_blue / n_red) | BASELINE-COOP | BASELINE-RAND | SHARED-ZSUM | DUAL-NASH | DUAL degradation |")
    lines.append("|---|---|---|---|---|---|")

    for n in FLEET_SIZES:
        n_red = N_RED_BY_N[n]
        n_blue = N_BLUE_BY_N[n]

        def _fmt(key: tuple[str, int]) -> str:
            if key not in table:
                return "—"
            m, s = table[key]
            return f"{m:.3f} ± {s:.3f}"

        ceiling = table.get(("baseline", n), (None, None))[0]
        dual = table.get(("dual", n), (None, None))[0]
        if ceiling is not None and dual is not None and ceiling > 0:
            deg = 1.0 - (dual / ceiling)
            deg_str = f"{100 * deg:.1f}% lost"
        else:
            deg_str = "—"

        lines.append(
            f"| N={n} / n_blue={n_blue} / n_red={n_red} "
            f"| {_fmt(('baseline', n))} "
            f"| {_fmt(('random', n))} "
            f"| {_fmt(('shared', n))} "
            f"| {_fmt(('dual', n))} "
            f"| {deg_str} |"
        )

    lines.append("")
    lines.append("*Degradation is computed as ``1 - DUAL-NASH / BASELINE-COOP`` — the headline threat-model number.*")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("- `experiments/_analysis/threat_degradation.png`")
    lines.append("- `experiments/_analysis/agent_denial.png`")
    lines.append("- `experiments/_analysis/nash_diagnostic.png`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="red_within_blue.analysis.threat_model",
        description="Analyze adversarial Phase 1 sweeps.",
    )
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    parser.add_argument("--docs-dir", type=str, default="docs")
    parser.add_argument("--window", type=int, default=100,
                        help="Episodes (from end) to average when computing final coverage.")
    args = parser.parse_args(argv)

    exp_dir = Path(args.experiments_dir)
    runs = load_all_runs(exp_dir)
    if not runs:
        print(f"No adv-* runs found in {exp_dir}.")
        return

    print(f"Loaded {len(runs)} adversarial runs:")
    for r in runs:
        print(f"  {r.name}  blue={r.blue_total_reward.shape}  red={r.red_total_reward.shape}")

    analysis_dir = exp_dir / "_analysis"
    plot_degradation(runs, analysis_dir / "threat_degradation.png")
    plot_agent_denial(runs, analysis_dir / "agent_denial.png", window=args.window)
    plot_nash_diagnostic(runs, analysis_dir / "nash_diagnostic.png")

    md_path = Path(args.docs_dir) / "08-threat-model.md"
    write_markdown_table(runs, md_path, window=args.window)

    print()
    print(f"Wrote plots to: {analysis_dir}")
    print(f"Wrote table to: {md_path}")


if __name__ == "__main__":
    main()
