"""Checkpoint utilities for saving and loading Flax parameter pytrees.

Provides a single implementation of flatten/unflatten and .npz I/O,
replacing the duplicated ``_flatten_params`` helpers scattered across scripts.
"""

from __future__ import annotations

from typing import Optional

import jax
import numpy as np


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------


def flatten_params(params) -> dict:
    """Convert a nested Flax param pytree to a flat ``{str: np.ndarray}`` dict.

    Each key is the ``/``-separated path through the pytree, e.g.
    ``"params/Dense_0/kernel"``.

    Args:
        params: A Flax parameter pytree (typically the output of ``model.init``).

    Returns:
        A flat dictionary mapping path strings to numpy arrays.
    """
    flat = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        key_str = "/".join(str(p.key if hasattr(p, "key") else p) for p in path)
        flat[key_str] = np.asarray(leaf)
    return flat


def unflatten_params(flat: dict, reference_params):
    """Reconstruct a param pytree from a flat dict and a reference structure.

    The *reference_params* pytree defines the tree structure (and dtypes/shapes)
    that the flat dict should be mapped back into.  Keys in *flat* must match
    those produced by :func:`flatten_params` on the same reference.

    Args:
        flat: A flat ``{str: np.ndarray}`` dict (e.g. from :func:`flatten_params`).
        reference_params: A pytree with the target structure.

    Returns:
        A pytree with the same structure as *reference_params*, filled with
        arrays from *flat*.
    """
    treedef = jax.tree_util.tree_structure(reference_params)
    ref_paths_and_leaves = list(jax.tree_util.tree_leaves_with_path(reference_params))

    leaves = []
    for path, _ref_leaf in ref_paths_and_leaves:
        key_str = "/".join(str(p.key if hasattr(p, "key") else p) for p in path)
        if key_str not in flat:
            raise KeyError(
                f"Key {key_str!r} expected by reference_params but missing from flat dict."
            )
        leaves.append(flat[key_str])

    return treedef.unflatten(leaves)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_checkpoint(params, path: str, critic_params=None) -> None:
    """Flatten params and save as a ``.npz`` file.

    If *critic_params* is provided, its flattened keys are prefixed with
    ``"critic/"`` so they can coexist in the same archive without colliding
    with the actor keys.

    Args:
        params: Actor (or main) parameter pytree.
        path: Destination file path (should end in ``.npz``).
        critic_params: Optional critic parameter pytree.
    """
    flat = flatten_params(params)
    if critic_params is not None:
        critic_flat = {
            f"critic/{k}": v for k, v in flatten_params(critic_params).items()
        }
        flat.update(critic_flat)
    np.savez(path, **flat)


def load_checkpoint(path: str) -> dict:
    """Load a ``.npz`` checkpoint and return a flat dict of numpy arrays.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        A flat ``{str: np.ndarray}`` dictionary.
    """
    data = np.load(path)
    return {key: data[key] for key in data.files}
