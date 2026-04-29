"""Live training progress bar.

Module-level singleton. The runner calls ``start`` before training and
``finish`` after. The trainer emits ``update`` via ``jax.debug.callback``
from inside its compiled scan; this module is the host-side sink.
"""
from __future__ import annotations

import shutil
import sys
import time
from typing import Optional

import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
    _HAVE_TQDM = True
except ImportError:  # pragma: no cover
    _tqdm = None
    _HAVE_TQDM = False


class _ProgressBar:
    """Thin wrapper around tqdm with a plain-text fallback."""

    def __init__(self) -> None:
        self._bar: Optional["_tqdm"] = None
        self._total: int = 0
        self._last_ep: int = -1
        self._fallback: bool = False
        self._t0: float = 0.0

    def start(self, total: int, desc: str) -> None:
        self._total = int(total)
        self._last_ep = -1
        self._t0 = time.time()
        if _HAVE_TQDM and sys.stderr.isatty():
            self._bar = _tqdm(
                total=self._total,
                desc=desc,
                unit="ep",
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            self._fallback = False
        else:
            self._bar = None
            self._fallback = True
            print(f"{desc}: 0 / {self._total} eps ...", flush=True)

    def update(self, ep, loss, reward) -> None:
        if self._total <= 0:
            return
        ep_val = int(np.asarray(ep).reshape(-1)[0])
        loss_val = float(np.mean(np.asarray(loss)))
        reward_val = float(np.mean(np.asarray(reward)))
        if ep_val <= self._last_ep:
            return
        delta = ep_val - self._last_ep
        self._last_ep = ep_val

        if self._bar is not None:
            self._bar.update(delta)
            self._bar.set_postfix(
                loss=f"{loss_val:+.4f}", reward=f"{reward_val:+.3f}",
            )
        elif self._fallback:
            pct = 100.0 * (ep_val + 1) / self._total
            elapsed = time.time() - self._t0
            eta = elapsed * (self._total - ep_val - 1) / max(ep_val + 1, 1)
            cols = max(40, shutil.get_terminal_size((80, 20)).columns)
            line = (
                f"[{pct:5.1f}%] ep {ep_val + 1:>6}/{self._total} "
                f"| loss {loss_val:+.4f} | reward {reward_val:+.3f} "
                f"| {elapsed:6.1f}s elapsed / {eta:6.1f}s eta"
            )
            print(line[:cols], flush=True)

    def finish(self) -> None:
        if self._bar is not None:
            if self._last_ep < self._total - 1:
                self._bar.update(self._total - 1 - self._last_ep)
            self._bar.close()
        elif self._fallback:
            elapsed = time.time() - self._t0
            print(f"done: {self._total} eps in {elapsed:.1f}s", flush=True)
        self._bar = None
        self._total = 0
        self._last_ep = -1
        self._fallback = False


_BAR = _ProgressBar()


def start(total: int, desc: str = "Training") -> None:
    _BAR.start(total, desc)


def update(ep, loss, reward) -> None:
    _BAR.update(ep, loss, reward)


def finish() -> None:
    _BAR.finish()
