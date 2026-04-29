"""Render Actor / Critic / JointRedActor architectures via ``flax.linen.tabulate``.

Emits two outputs:

  experiments/meta-report/architecture.txt
      Plain-text tabulated summary embedded into ``meta_report.html`` §10.

  experiments/meta-report/architecture_inline.html
      Same content wrapped in ``<pre>`` so ``meta_report.py`` can read it
      verbatim without re-importing the networks.

Self-documenting: if network shapes change, re-running this script picks up
the new sizes with zero manual update to the meta-report or the architecture
guide.

Run:
    python scripts/architecture_dump.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn

from red_within_blue.training.config import ExperimentConfig
from red_within_blue.training.networks import Actor, Critic, JointRedActor


OUT_DIR = Path("experiments/meta-report")
REF_CONFIG = "configs/compromise-16x16-5-3b2r.yaml"


def _tabulate(model: nn.Module, *inputs) -> str:
    """Return a plain-text Rich-style tabulated network summary."""
    return nn.tabulate(
        model,
        jax.random.PRNGKey(0),
        compute_flops=True,
        console_kwargs={"force_terminal": False, "no_color": True, "width": 110},
    )(*inputs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.from_yaml(REF_CONFIG)
    obs_dim = cfg.obs_dim
    n_red = cfg.env.num_red_agents
    num_actions = cfg.env.num_actions

    actor = Actor(
        num_actions=num_actions,
        hidden_dim=cfg.network.actor_hidden_dim,
        num_layers=cfg.network.actor_num_layers,
    )
    critic = Critic(
        hidden_dim=cfg.network.critic_hidden_dim,
        num_layers=cfg.network.critic_num_layers,
    )
    jred = JointRedActor(
        num_red=n_red,
        num_actions=num_actions,
        hidden_dim=cfg.train.red_hidden_dim,
        num_layers=cfg.train.red_num_layers,
    )

    header = (
        f"Reference config : {REF_CONFIG}\n"
        f"obs_dim          : {obs_dim}\n"
        f"num_actions (|A|): {num_actions}\n"
        f"num_red          : {n_red}  (joint-obs dim = {n_red * obs_dim})\n"
        f"\n"
        "All three networks are plain MLPs with Flax `nn.Dense` + ReLU. The actor\n"
        "and joint-red head emit unnormalised logits; action masking is applied\n"
        "externally. The critic emits a scalar V(s). Weights are tiny (<100 kB\n"
        "each) — a full 5-seed `vmap` trains on CPU at ~200 eps/s.\n"
    )

    dummy_obs = jnp.ones(obs_dim)
    dummy_joint = jnp.ones(n_red * obs_dim)

    parts = [
        header,
        "=" * 90,
        "Actor — blue & shared per-agent policy",
        "=" * 90,
        _tabulate(actor, dummy_obs),
        "=" * 90,
        "Critic — centralized V(s) for CTDE training",
        "=" * 90,
        _tabulate(critic, dummy_obs),
        "=" * 90,
        f"JointRedActor — centralized {n_red}-agent red policy (joint obs → joint action logits)",
        "=" * 90,
        _tabulate(jred, dummy_joint),
    ]
    text = "\n".join(parts).rstrip() + "\n"

    txt_path = OUT_DIR / "architecture.txt"
    txt_path.write_text(text)
    print(f"wrote {txt_path}  ({len(text)} chars)")

    # HTML fragment so meta_report.py can embed without re-importing flax.
    inline = (
        "<pre class=\"arch-dump\" style=\"background:#0f1420; color:#d6e6ff; "
        "padding:1em; border-radius:6px; font-size:0.78em; line-height:1.2; "
        "overflow-x:auto;\">\n"
        + text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre>\n"
    )
    html_path = OUT_DIR / "architecture_inline.html"
    html_path.write_text(inline)
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
