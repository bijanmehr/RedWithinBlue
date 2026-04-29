"""Delete redundant per-experiment artifacts.

Each adversarial-ladder / coevo experiment dir tends to accumulate:
    checkpoint.npz, checkpoint_trained.npz       (the "trained" cousin)
    episode.gif,  episode_pretraining.gif        (the pre-warm-start eyeballer)
    eval.txt,     eval_pretraining.txt           (pre-warm-start summary)
                  eval_trained.txt               (post-warm-start summary)

The "_pretraining" and "_trained" cousins were debugging artifacts during the
warm-start ladder migration. The lessons they captured live in the engineering
retrospective. Keeping them costs ~50 MB per dir.

Default: dry-run. Pass --execute to actually delete.

Usage:
    python scripts/clean_experiment_dirs.py             # dry-run
    python scripts/clean_experiment_dirs.py --execute   # actually delete
"""

from __future__ import annotations

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"

REDUNDANT_NAMES = (
    "checkpoint_trained.npz",
    "episode_pretraining.gif",
    "eval_pretraining.txt",
    "eval_trained.txt",
)


def find_redundant() -> list[pathlib.Path]:
    targets: list[pathlib.Path] = []
    for sub in sorted(EXPERIMENTS.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_") or sub.name in {
            "Backup", "meta-report"
        }:
            continue
        for name in REDUNDANT_NAMES:
            f = sub / name
            if f.exists():
                targets.append(f)
    return targets


def fmt_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    targets = find_redundant()
    if not targets:
        print("nothing to clean.")
        return

    total = sum(t.stat().st_size for t in targets)
    print(f"{'WILL DELETE' if args.execute else 'DRY-RUN — found'} "
          f"{len(targets)} files, {fmt_size(total)} total:")
    for t in targets:
        print(f"  {fmt_size(t.stat().st_size):>10}  {t.relative_to(ROOT)}")

    if args.execute:
        for t in targets:
            t.unlink()
        print(f"deleted {len(targets)} files, freed {fmt_size(total)}")
    else:
        print("(re-run with --execute to actually delete)")


if __name__ == "__main__":
    main()
