<div align="center">

# RedWithinBlue

**JAX-native multi-agent reinforcement learning for cooperative grid exploration**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/built%20with-JAX-orange.svg)](https://github.com/google/jax)
[![Tests](https://img.shields.io/badge/tests-244%20passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

Agents explore grid worlds, communicate over distance-based graphs, and learn cooperative strategies — all inside a single JIT-compiled training loop.

> **Note** &mdash; This project is under active development. Docs and sweep results are being filled in as experiments run.

</div>

---

## Installation

```bash
git clone https://github.com/bijanmehr/RedWithinBlue.git
cd RedWithinBlue
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

> **GPU users:** Install [JAX with CUDA](https://github.com/google/jax#installation) before `pip install`.

## Usage

### Train

```bash
python -m red_within_blue.training.runner --config configs/solo-explore.yaml
```

Results go to `experiments/solo-explore/` — trained weights (`checkpoint.npz`) and per-episode metrics (`metrics.npz`).

### Available Configs

| Config | Agents | Grid | Description |
|:-------|:------:|:----:|:------------|
| `solo-explore.yaml` | 1 | 10x10 | Single agent learns exploration from scratch |
| `pair-cooperate.yaml` | 2 | 10x10 | Cooperative pair with connectivity enforcement |
| `team-coordinate.yaml` | 4 | 18x18 | Team coordination with walls and larger comm radius |

Override settings from the command line:

```bash
python -m red_within_blue.training.runner --config configs/pair-cooperate.yaml --num-seeds 3
python -m red_within_blue.training.runner --config configs/solo-explore.yaml --output-dir /tmp/runs
```

### Warm-Start (Policy Transfer)

Multi-agent and larger-grid experiments warm-start from a trained single-agent checkpoint. The observation dimension is constant across grid sizes and agent counts (determined only by `obs_radius` and `msg_dim`), so actor/critic weights transfer directly without architecture changes.

```bash
# Step 1: train the base policy
python -m red_within_blue.training.runner --config configs/solo-explore.yaml

# Step 2: warm-start multi-agent from the trained checkpoint
python -m red_within_blue.training.runner --config configs/sweeps/multi-2agent-10x10.yaml
```

Configs that require warm-start include a `warm_start:` field pointing to the source checkpoint. The runner loads the pretrained weights via `unflatten_params()` and passes them to the trainer as initial parameters instead of random initialization.

### Hyperparameter Sweeps

22 sweep experiments in `configs/sweeps/`, organized into two stages:

**Stage 1 — Single-agent ablations** (train from scratch, no dependencies):

| Sweep | Configs | What it isolates |
|:------|:--------|:-----------------|
| **Policy gradient method** | `configs/hyperparameter_search/method-*.yaml` | REINFORCE vs. mean-return baseline vs. actor-critic (TD advantage with learned value function) |
| **Discount factor (gamma)** | `configs/hyperparameter_search/gamma-*.yaml` | Temporal credit assignment horizon — how far ahead the agent looks when discounting future rewards |
| **Learning rate** | `configs/hyperparameter_search/lr-*.yaml` | Optimizer step size for Adam; too high causes policy oscillation, too low slows convergence |
| **Network capacity** | `configs/hyperparameter_search/net-*.yaml` | Hidden dimension and depth of actor/critic MLPs — tradeoff between expressiveness and training stability |

**Stage 2 — Multi-agent & generalization** (warm-start from `solo-explore` checkpoint):

| Sweep | Configs | What it isolates |
|:------|:--------|:-----------------|
| **Agent scaling** | `multi-2agent-10x10`, `multi-4agent-10x10`, `multi-4agent-18x18` | Shared-parameter policy scaling — all agents run the same network via `jax.vmap`, testing whether a solo-trained policy generalizes to cooperative settings |
| **Connectivity guardrail** | `conn-enforced`, `conn-free` | Whether the sequential connectivity check (spectral graph analysis via eigenvalue decomposition of the Laplacian) helps or hinders exploration when agents share a communication graph |
| **Grid generalization** | `grid-14x14`, `grid-18x18` | Zero-shot policy transfer to unseen grid sizes — tests whether learned exploration behavior is spatially invariant |

All sweep configs use 1000 episodes, 5 seeds, and vary one dimension at a time against the best known baseline (actor-critic, 128x2 MLP, lr=3e-4, gamma=0.90).

```bash
# Hyperparameter search (lr / gamma / net / method) — all configs in the folder
./configs/hyperparameter_search/run_search.sh

# Run a specific subset
./configs/hyperparameter_search/run_search.sh configs/hyperparameter_search/lr-*.yaml

# Single experiment stages (adversarial, multi-agent, grid) — see experiments/README.md
python -m red_within_blue.training.runner --config configs/sweeps/multi-4agent-18x18.yaml \
    --warm-start experiments/solo-explore/checkpoint.npz
```

### Use the Environment Directly

```python
import jax
from red_within_blue import GridCommEnv, EnvConfig

config = EnvConfig(grid_width=10, grid_height=10, num_agents=2, max_steps=100)
env = GridCommEnv(config)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

actions = {a: jax.random.randint(jax.random.PRNGKey(1), (), 0, 5) for a in env.agents}
obs, state, rewards, dones, info = env.step_env(jax.random.PRNGKey(2), state, actions)
```

## Testing

```bash
pytest tests/ -q
```

244 tests covering environment, training pipeline, config system, rollout, networks, metrics, statistics, and visualization.

## Project Structure

```
RedWithinBlue/
├── configs/              Experiment YAML configs
│   └── sweeps/           22 hyperparameter sweep configs
├── src/red_within_blue/
│   ├── env.py            GridCommEnv (JaxMARL-compatible)
│   ├── grid.py           Grid operations & scanning
│   ├── comm_graph.py     Communication graph & connectivity
│   └── training/
│       ├── trainer.py    JIT-compiled training loop (PureJaxRL pattern)
│       ├── rollout.py    Episode collection via jax.lax.scan
│       ├── runner.py     CLI entry point
│       └── config.py     Typed config (frozen dataclasses)
├── tests/                244 tests
└── docs/                 Design docs & experiment reports
```

## Documentation

> **TODO** &mdash; Some documents contain placeholder sections marked `[PENDING]` that will be filled as experiments complete. Results sections in the experiment report are being updated as sweeps run on GPU.

| Doc | Status | What's inside |
|:----|:------:|:--------------|
| [Architecture Guide](docs/07-architecture-guide.md) | Complete | Full codebase walkthrough. PureJaxRL training pattern, `jax.lax.scan` rollout, parameter sharing across agents, connectivity guardrail implementation, how to add new training methods. Start here if you want to understand or extend the code. |
| [Experiment Report](docs/06-experiment-report.md) | In progress | Three-stage experiment plan (solo -> pair -> team), success criteria, reward design, hyperparameter sweep methodology. Contains results from evolutionary training runs. Sweep results being added. |
| [Environment API](docs/04-environment-api.md) | Complete | Full API reference for `GridCommEnv`. Dict-based interface (JaxMARL standard) and array-based interface (for `lax.scan` rollouts). Observation space, action space, state structure. |
| [Environment Design](docs/05-environment-design-spec.md) | Complete | Design specification for the grid environment. Grid representation, agent observations, communication graph construction, reward functions, terrain generation. |
| [RL Overview](docs/01-rl-overview.md) | Complete | Reinforcement learning fundamentals. MDP formulation, policy gradient theorem, value functions, temporal difference learning. Reference material for the methods used in training. |
| [Training Guide](docs/02-rl-training-guide.md) | Complete | Detailed walkthrough of training methods. REINFORCE, baselines for variance reduction, actor-critic with TD advantage, entropy regularization, experience replay. Covers the theory behind each layer of the training pipeline. |
| [RL Taxonomy](docs/03-rl-taxonomy.md) | Complete | Algorithm taxonomy mapping. On-policy vs. off-policy, model-free vs. model-based, value-based vs. policy-based. Includes PPO, MAPPO, GAE, and other algorithms on the roadmap. |

## Roadmap

- [ ] Entropy regularization and gradient clipping for larger networks
- [ ] Experience replay buffer integration for off-policy methods (DQN track)
- [ ] Stronger communication enforcement (continuous penalty vs. binary guardrail)
- [ ] Adversarial agents — red team obstructs, blue team explores
- [ ] PPO/MAPPO with Generalized Advantage Estimation (GAE) and LayerNorm
- [ ] Scaling to 10+ agents on 32x32+ grids

## License

MIT
