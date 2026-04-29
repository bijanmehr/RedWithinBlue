<div align="center">

# RedWithinBlue

**JAX-native multi-agent reinforcement learning for cooperative grid exploration**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/built%20with-JAX-orange.svg)](https://github.com/google/jax)
[![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen.svg)](#testing)
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
python -m red_within_blue.training.runner --config configs/pair-cooperate-coop.yaml
```

Results go to `experiments/<config-name>/` — trained weights (`checkpoint.npz`), per-episode metrics (`metrics.npz`), and an HTML report. The experiments folder tracks the data files (metrics, json summaries, logs) but ignores the large model checkpoints and the rendered reports.

### Available Configs

The `configs/` folder holds 43 base configs grouped into a few families. Pick one and run.

| Family | Configs | What it is |
|:-------|:--------|:-----------|
| **Cooperative warm-start ladder** | `pair-cooperate-coop`, `quad-cooperate-coop-{8,16,32}{,-norm}`, `octa-cooperate-coop-32-r6{,-conn,-nocoop,-norm}` | The base N=2 → N=8, 10×10 → 32×32 ladder. Each rung warm-starts from the one below; `-norm` siblings carry uid-normalisation for N-mismatched transfer. |
| **Survey-local prototype** | `survey-local-*` | Local-obs (3×3 window), grid-invariant observations. Decouples sensing from per-cell survey. |
| **Adversarial warm-start ladder** | `adv-ladder-r{1..6}-*` | Six-rung red/blue scale-up (6×6 → 32×32). Warm blue, fresh red each rung. |
| **Compromise sweep** | `compromise-16x16-*` | k-of-N adversarial sweep at 16×16. S=46%, B=98.5%, C1=89.6%, C2=87.1%; knee at m=1. |

Detailed per-experiment instructions live in [`experiments/README.md`](experiments/README.md).

Override settings from the command line:

```bash
python -m red_within_blue.training.runner --config configs/pair-cooperate-coop.yaml --num-seeds 3
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-16.yaml --output-dir /tmp/runs
```

### Warm-Start (Policy Transfer)

Multi-agent and larger-grid experiments warm-start from the rung below in the ladder. The observation dimension is constant across grid sizes and agent counts (determined only by `obs_radius` and `msg_dim`), so actor and **central critic** weights transfer directly. *Always warm-start both* — a re-initialised central critic is the main collapse mode for CTDE scale-ups (see `docs/09-engineering-retrospective.md` §3.1).

```bash
# Step 1: train the rung below
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-8.yaml

# Step 2: warm-start the next rung (config sets warm_start: path automatically)
python -m red_within_blue.training.runner --config configs/quad-cooperate-coop-16.yaml
```

For N-mismatched transfer (e.g. N=4 → N=8), the runner tiles `Dense_0` per-agent block. Use the `-norm` siblings for N-mismatched transfer to avoid the uid-extrapolation failure mode documented in `project_uid_normalization_tradeoff`.

### Hyperparameter Sweeps

**Single-agent ablations** (`configs/hyperparameter_search/`) — vary one knob at a time against the actor-critic baseline:

| Sweep | Configs | What it isolates |
|:------|:--------|:-----------------|
| **Method** | `method-*.yaml` | REINFORCE vs. mean-return baseline vs. actor-critic (TD advantage with learned value) |
| **Discount factor** | `gamma-*.yaml` | Temporal credit assignment horizon |
| **Learning rate** | `lr-*.yaml` | Adam step size; oscillation vs. convergence speed |
| **Network capacity** | `net-*.yaml` | Actor/critic MLP width × depth |

**Adversarial sweeps** (`configs/sweeps/`) — red-team variants used by the adversarial ladder:

| Sweep | Configs | What it is |
|:------|:--------|:-----------|
| **Baseline** | `adv-baseline-{5,7,10}blue.yaml` | Coverage upper bound — no red, varying blue count |
| **Random red** | `adv-random-{5-2red, 7-3red, 10-4red}.yaml` | Red picks uniform-random actions |
| **Shared red** | `adv-shared-*.yaml` | Red shares the blue policy |
| **Dual red** | `adv-dual-*.yaml` | Independently-trained red opponent |

```bash
# Single-agent ablations (lr / gamma / net / method) — all configs in the folder
./configs/hyperparameter_search/run_search.sh

# Run a specific subset
./configs/hyperparameter_search/run_search.sh configs/hyperparameter_search/lr-*.yaml
```

For experiment-specific running and result interpretation, see [`experiments/README.md`](experiments/README.md).

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

105 test cases across 22 files covering environment, training pipeline, config system, rollout, networks, metrics, statistics, and visualization.

## Project Structure

```
RedWithinBlue/
├── configs/                       43 experiment YAML configs
│   ├── sweeps/                    Adversarial-variant sweep configs
│   └── hyperparameter_search/     Single-agent ablation sweeps
├── src/red_within_blue/
│   ├── env.py                     GridCommEnv (JaxMARL-compatible)
│   ├── grid.py                    Grid operations & scanning
│   ├── comm_graph.py              Communication graph & connectivity
│   ├── agents.py                  Per-agent policy & state
│   ├── visualizer.py              Episode rendering (gif / png)
│   ├── analysis/
│   │   ├── experiment_report.py   HTML report + eval-gif generation
│   │   └── threat_model.py        Compromise / misbehavior analysis
│   └── training/
│       ├── trainer.py             JIT-compiled CTDE training loop
│       ├── rollout.py             Episode collection via jax.lax.scan
│       ├── runner.py              CLI entry point
│       ├── losses.py              Actor-critic and DQN losses
│       ├── networks.py            Actor / central critic / joint-red MLPs
│       ├── rewards_training.py    Reward shaping (coverage + cohesion + fog)
│       ├── transfer.py            Warm-start + per-agent block tiling
│       └── config.py              Typed config (frozen dataclasses)
├── tests/                         22 files, 105 test cases
├── experiments/                   Per-experiment data + operator's manual
└── docs/                          Theory, design, retrospective, roadmap
```

The companion presentation deck (figures, render scripts) lives in the sibling
folder `../DL presentation/` — outside this repo so the slides aren't versioned with the code.

## Documentation

> **TODO** &mdash; Some documents contain placeholder sections marked `[PENDING]` that will be filled as experiments complete. Results sections in the experiment report are being updated as sweeps run on GPU.

| Doc | Status | What's inside |
|:----|:------:|:--------------|
| [Architecture Guide](docs/07-architecture-guide.md) | Complete | Full codebase walkthrough. PureJaxRL training pattern, `jax.lax.scan` rollout, parameter sharing across agents, connectivity guardrail implementation, how to add new training methods. Start here if you want to understand or extend the code. |
| [Experiment Report](docs/06-experiment-report.md) | In progress | Three-stage experiment plan (solo -> pair -> team), success criteria, reward design, hyperparameter sweep methodology. Contains results from evolutionary training runs. Sweep results being added. |
| [Stabilization Experiments](docs/08-stabilization-experiments.md) | Complete | The TD(0) → MC critic story. Why we landed on Monte-Carlo returns over TD(0) / twin-Q / target-net. EXP-A and EXP-B negative-result writeups (`don't do this`). |
| [Engineering Retrospective](docs/09-engineering-retrospective.md) | Living | Decisions taken and abandoned, named failures with root causes, open questions, future directions. Companion to the meta-report. |
| [Wild Ideas](docs/10-wild-ideas.md) | Living | Research-direction scratchpad — hex grid, general-sum red, terrain + energy, AoE-style swarm-vs-swarm. Cheapest-first scoping, not commitments. |
| [Environment API](docs/04-environment-api.md) | Complete | Full API reference for `GridCommEnv`. Dict-based interface (JaxMARL standard) and array-based interface (for `lax.scan` rollouts). Observation space, action space, state structure. |
| [Environment Design](docs/05-environment-design-spec.md) | Complete | Design specification for the grid environment. Grid representation, agent observations, communication graph construction, reward functions, terrain generation. |
| [RL Overview](docs/01-rl-overview.md) | Complete | Reinforcement learning fundamentals. MDP formulation, policy gradient theorem, value functions, temporal difference learning. Reference material for the methods used in training. |
| [Training Guide](docs/02-rl-training-guide.md) | Complete | Detailed walkthrough of training methods. REINFORCE, baselines for variance reduction, actor-critic with TD advantage, entropy regularization, experience replay. Covers the theory behind each layer of the training pipeline. |
| [RL Taxonomy](docs/03-rl-taxonomy.md) | Complete | Algorithm taxonomy mapping. On-policy vs. off-policy, model-free vs. model-based, value-based vs. policy-based. Includes PPO, MAPPO, GAE, and other algorithms on the roadmap. |

## Roadmap

**Done**
- [x] CTDE actor-critic with shared per-agent actor and central critic
- [x] Entropy regularisation and gradient clipping
- [x] Monte-Carlo critic target — chosen over TD(0) / twin-Q / Polyak target after the stabilisation study
- [x] Adversarial agents — joint-red CTDE actor with REINFORCE + entropy
- [x] Six-rung adversarial warm-start ladder (6×6 → 32×32)
- [x] Compromise / k-of-N misbehavior-budget sweep with paired-bootstrap CIs
- [x] Coevolutionary ES for red (`adv-ladder-r6-coevo`)
- [x] Scaling to N=8 on 32×32 grids

**Open** *(see `docs/09-engineering-retrospective.md` §6–7 and `docs/10-wild-ideas.md`)*
- [ ] **General-sum red** — drop the `per_red = -blue_sum/n_red` reduction; PSRO drop-in for `coevo.py`
- [ ] **Detection mechanism + flexible guardrails** — defender-side detector, learnable soft constraint floors
- [ ] **Resilience metrics** — magnitude · brittleness · timeliness triangle as §8 of the meta-report
- [ ] **Hierarchical RL** — macro-actions for routing / scanning / fall-back
- [ ] **Hex grid + cell physics + new actions**
- [ ] **Energy economy + heterogeneous terrain**
- [ ] **JaxMARL / Multi-Agent Craftax integration** for swarm-scale work

## License

MIT
