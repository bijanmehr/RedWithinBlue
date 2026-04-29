# Codebase Architecture: JAX-Native Training Pipeline

RedWithinBlue is a JAX-native multi-agent RL environment for cooperative grid exploration. The training pipeline follows the PureJaxRL pattern: the entire training loop (rollout + loss + gradient update) is a single JIT-compiled function.

---

## 1. Project Structure

```
src/red_within_blue/
├── env.py              — GridCommEnv (JaxMARL MultiAgentEnv)
├── grid.py             — Grid operations (terrain, occupancy, scanning)
├── movement.py         — Action resolution with collision handling
├── agents.py           — Agent state initialization and updates
├── comm_graph.py       — Adjacency, components, message routing
├── types.py            — Flax struct dataclasses (EnvState, AgentState, EnvConfig, etc.)
├── rewards.py          — Environment-level reward functions
├── wrappers.py         — Environment wrappers
├── logger.py           — Logging utilities
├── replay.py           — Replay buffer
├── visualizer.py       — Matplotlib dashboard
└── training/
    ├── config.py       — Typed ExperimentConfig (frozen dataclasses)
    ├── rollout.py      — JAX-native episode collection (lax.scan)
    ├── losses.py       — Pure JAX loss functions (PG, baseline, actor-critic)
    ├── trainer.py      — make_train() / make_train_multi_seed()
    ├── runner.py       — CLI entry point
    ├── networks.py     — Actor, Critic, QNetwork (Flax nn.Module)
    ├── checkpoint.py   — Save/load parameter pytrees as .npz
    ├── metrics.py      — Coverage, action distribution, explained variance
    ├── stats.py        — Chi-squared, Welch's t, Bonferroni
    ├── transfer.py     — Weight transfer, CKA analysis
    ├── rewards_training.py — Training reward functions (normalized_exploration_reward, etc.)
    ├── report.py       — HTML experiment reports
    ├── plotting.py     — Tufte-style visualization
    ├── gif.py          — Episode GIF recording
    └── dqn.py          — Q-learning utilities
configs/
    ├── solo-explore.yaml      — 1 agent, 10x10
    ├── pair-cooperate.yaml    — 2 agents, 10x10
    └── team-coordinate.yaml   — 4 agents, 18x18
tests/
    ├── test_env.py, test_config.py, test_rollout.py, test_trainer.py, test_runner.py
    ├── test_grid.py, test_movement.py, test_agents.py, test_comm_graph.py
    ├── test_networks.py, test_losses (test_pg.py), test_metrics.py, test_stats.py
    ├── test_dqn.py, test_transfer.py, test_rewards_training.py
    ├── test_plotting.py, test_gif.py, test_visualizer.py, test_wrappers.py
    └── test_configs.py, test_logger.py, test_replay.py
```

---

## 2. How Training Works (PureJaxRL Pattern)

The training pipeline is built in `training/trainer.py`. The flow:

1. **Load config.** `ExperimentConfig.from_yaml("configs/stage1.yaml")` reads YAML into frozen dataclasses.

2. **Build trainer.** `make_train(config)` constructs a JIT-compiled `train_fn` closure. All config values are resolved at Python trace time (plain `if/elif` on `config.train.method`), so method dispatch has zero runtime cost.

3. **Inside `train_fn(key)`:**
   - Initialize `Actor` (and `Critic` for actor-critic) params via `model.init(key, dummy_obs)`.
   - Initialize `optax.adam` optimizer states.
   - `jax.lax.scan` over `num_episodes` steps. Each step:
     - Collect one episode via `collect_episode_scan` (single-agent) or `collect_episode_multi_scan` (multi-agent).
     - Compute loss via `jax.value_and_grad`.
     - Apply gradient update with `optax.apply_updates`.
   - Return `(actor_params, critic_params_or_None, metrics_dict)`.

4. **Multi-seed.** `make_train_multi_seed(config)` wraps `train_fn` with `jax.vmap` over `config.train.num_seeds` independent PRNG keys. All seeds run in parallel on a single `jax.jit` call.

5. **Orchestration.** `runner.py` ties it together: parse CLI args, load config with overrides, call `run_training`, save checkpoints + metrics via `save_results`, print summary.

### Key Design Decisions

- **Fixed-length episodes.** `lax.scan` over `max_steps` with a `mask` field (1.0 before done, 0.0 after) for correct reward accumulation. No dynamic-length Python loops inside JIT.
- **Parameter sharing.** One `Actor` network serves all agents. The forward pass is vmapped over the agent dimension: `jax.vmap(actor.apply, in_axes=(None, 0))(params, obs_all)`.
- **Method dispatch at trace time.** `trainer.py` uses `if method == "reinforce"` / `elif method == "actor_critic"` in Python. This selects the loss function and carry structure before JIT compilation, not at runtime.
- **Connectivity guardrail.** `_connectivity_guardrail` in `rollout.py` uses `jax.lax.scan` over agents sequentially. Each agent's committed move is visible to the next agent's connectivity check via `build_adjacency` and `compute_components` from `comm_graph.py`.

---

## 3. Running Experiments

```bash
python -m red_within_blue.training.runner --config configs/solo-explore.yaml
python -m red_within_blue.training.runner --config configs/pair-cooperate.yaml --num-seeds 3
python -m red_within_blue.training.runner --config configs/team-coordinate.yaml --output-dir /tmp/runs
```

CLI flags (`--output-dir`, `--num-seeds`) override YAML values via `dataclasses.replace` in `load_config_with_overrides`.

### Output Structure

```
experiments/{experiment_name}/
├── checkpoint.npz    — Actor + critic parameters (flattened pytree)
└── metrics.npz       — Per-episode loss and total_reward arrays
```

Checkpoints use `save_checkpoint` from `training/checkpoint.py`. Critic keys are prefixed with `"critic/"` so they coexist in one `.npz` file. To reload:

```python
from red_within_blue.training.checkpoint import load_checkpoint, unflatten_params
flat = load_checkpoint("experiments/stage1/checkpoint.npz")
actor_params = unflatten_params(flat, reference_params)
```

### Warm-starting a run from a prior checkpoint

`runner.py:load_warm_start_params` re-hydrates actor + critic params from a source checkpoint and — crucially — handles grid-size changes by spatially upsampling the first-layer kernel. This is what makes the `8×8 → 16×16 → 32×32` ladder work: one set of weights, reshaped to fit progressively larger observation spaces.

**Shape of the problem.** Agents observe `obs = scan(S) | grid_seen_mask(H·W) | tail(5)`. Only the `grid_seen_mask` block depends on grid shape — `scan` and `tail` are grid-invariant scalars. Every Dense layer past `Dense_0` operates on `[hidden, hidden]`, also grid-invariant. So the only kernel that needs transforming is `Dense_0`'s input axis.

**Actor (`num_blocks=1`).** `_upsample_first_layer_for_grid` splits `Dense_0/kernel` along its input axis into `scan | grid | tail` rows, reshapes the grid slice `[H·W, hidden] → [H, W, hidden]`, does a **nearest-neighbor** spatial upsample to `[H', W', hidden]`, flattens back, and re-concatenates. Bias and deeper layers copy verbatim. Seed axis `[S, ...]` is preserved.

**Central critic (`num_blocks=N`).** The CTDE critic ingests `joint_obs = observations.reshape(T, -1)` — N concatenated copies of the per-agent obs. Its `Dense_0/kernel` therefore has input shape `[N × per_block_obs_dim, hidden]`. The loader splits that axis into N per-agent blocks, upsamples each block's grid rows exactly like the actor, then concatenates the blocks back. Critic transfer requires `critic_hidden_dim` and `critic_num_layers` to match between source and target. If they don't, the loader logs a warning and falls back to re-init rather than raising.

**Mismatched agent counts (`source_num_blocks ≠ num_blocks`).** Adding agents at fixed grid is supported when the target N is an integer multiple of the source N (e.g. `4 → 8`). Set `warm_start_source_num_agents` in the target YAML; the loader splits the source kernel into `source_num_blocks` per-agent blocks, upsamples each one, and **tiles** the upsampled set `tile_factor = num_blocks // source_num_blocks` times along the input axis. This is principled because the central critic is symmetric in agent identity — the per-agent slot weights *should* be identical for every slot under random initialisation, so duplicating the source's slot weights to fill new slots is the minimum-assumption initialisation. Non-integer ratios are rejected (you can't tile a 4-block kernel into a 6-slot critic without ambiguity); the loader skips the critic transfer in that case.

**Lessons baked in (see `experiments/README.md` for the receipts):**

1. **Actor-only warm-start is a trap for CTDE runs.** Re-initialising the critic produces increasingly wrong regression targets on the new joint-obs scale and drags the policy into a low-entropy corner — the reward trajectory looks strong for ~500 eps, then collapses monotonically as the critic diverges. Always transfer both.

2. **Off a converged warm-start, fine-tune gently.** The source's from-scratch LR is the wrong LR for an already-good starting point. Start with `lr ~5× smaller` and `num_episodes ~4× shorter` than the source used; the fine-tune's job is adaptation, not re-learning. Going hot overwrites the transferred policy.

3. **For N-mismatched transfer, the bar is even higher.** The tiled critic injects more initial value-prediction noise than a same-N transfer (every per-agent slot starts identical, which is the right prior but not the truth on any specific trajectory). Even the gentle recipe that works for same-N rungs can overshoot. Gauge the warm-start with the diagnostic first; if it's already strong, *skip the fine-tune* and save the warm-start as the canonical checkpoint.

4. **Always run the pre-training diagnostic first.** `/tmp/eval_pretraining_warmstart.py <config.yaml>` evaluates the raw upsampled weights on the target grid with zero gradient steps. It pins any later collapse to *mechanism* vs *training recipe* in under 2 minutes — and at the same time tells you whether training is even worth attempting.

See `experiments/README.md` for the worked examples and concrete numbers from the `8 → 16 → 32` ladder (same-N) and the `quad-32 → octa-32-r6` rung (N-mismatched).

---

## 4. Configuration System

`ExperimentConfig` is a frozen Python dataclass (not a Flax struct) with nested groups:

| Dataclass       | Key fields                                                              |
|-----------------|-------------------------------------------------------------------------|
| `EnvParams`     | `grid_width`, `grid_height`, `num_agents`, `wall_density`, `max_steps`, `comm_radius`, `obs_radius`, `msg_dim` |
| `NetworkParams` | `actor_hidden_dim`, `actor_num_layers`, `critic_hidden_dim`, `critic_num_layers` |
| `TrainParams`   | `method`, `lr`, `gamma`, `vf_coef`, `num_episodes`, `num_seeds`         |
| `RewardParams`  | `disconnect_penalty`                                                    |

YAML format mirrors the dataclass hierarchy:

```yaml
experiment_name: solo-explore
env:
  grid_width: 10
  num_agents: 1
  max_steps: 100
train:
  method: actor_critic
  lr: 3.0e-4
  num_episodes: 2000
  num_seeds: 5
enforce_connectivity: false
```

Missing keys fall back to defaults defined on each dataclass.

### Bridging to the Environment

`config.to_env_config()` converts `ExperimentConfig` into a `red_within_blue.types.EnvConfig` (Flax struct), which `GridCommEnv` consumes. This separation keeps the config layer in pure Python while the env operates on JAX-compatible structs.

### Computed Properties

`config.obs_dim` computes the observation dimension from env params. This must match the formula in `env.py`.

### Legacy Support

`get_stage_configs(path)` loads old multi-stage YAML files and returns a tuple of three `ExperimentConfig` instances (stages 1-3). `from_legacy_config(path)` returns stage 1 only.

---

## 5. Adding a New Training Method

1. **Add loss function to `losses.py`.** Pure JAX, no side effects. Follow the signature pattern of `pg_loss(logits, actions, returns) -> scalar` or the actor-critic pattern returning `(policy_loss, value_loss)`.

2. **Add the method name to `TrainParams.method`** docstring/comments (currently: `"reinforce"`, `"baseline"`, `"actor_critic"`).

3. **Add elif branch in `trainer.py`'s `make_train()`.** You need two sub-branches: single-agent (`num_agents == 1`) and multi-agent. Define `_train_step` with the appropriate carry structure and loss computation. Wire it into the `_scan_body` / `jax.lax.scan` loop.

4. **Write tests in `tests/test_trainer.py`.** Verify that `make_train(config)` JIT-compiles, runs without error, and that loss decreases over episodes.

5. **Create a YAML config** in `configs/` with `train.method` set to the new method name.

---

## 6. Key Abstractions

### Trajectory / MultiTrajectory

Flax `struct.dataclass` types defined in `rollout.py`. Proper JAX pytrees, compatible with `jax.lax.scan`, `jax.vmap`, and `jax.jit`.

- **`Trajectory`** (single-agent): `obs [T, obs_dim]`, `actions [T]`, `rewards [T]`, `dones [T]`, `log_probs [T]`, `mask [T]`
- **`MultiTrajectory`** (multi-agent): `obs [T, N, obs_dim]`, `actions [T, N]`, `rewards [T, N]`, `dones [T]`, `log_probs [T, N]`, `mask [T]`

The `mask` field is `1.0` before the episode terminates and `0.0` after, enabling fixed-length `lax.scan` with correct reward masking.

### GridCommEnv

Inherits from `jaxmarl.environments.multi_agent_env.MultiAgentEnv`. Composes `grid.py`, `movement.py`, `agents.py`, and `comm_graph.py`. Exposes two interfaces:

- **Dict-based** (standard JaxMARL): `reset(key)`, `step_env(key, state, actions_dict)`, `get_obs(state)` -- actions and observations keyed by agent name strings.
- **Array-based** (for `lax.scan`): `obs_array(state) -> [N, obs_dim]`, `step_array(key, state, actions) -> (obs, state, rewards, done, info)` -- flat arrays, no dicts.

### ExperimentConfig

Frozen Python dataclass. Not a Flax struct, not a JAX array. All fields are resolved at trace time. Use `dataclasses.replace(config, ...)` to derive modified configs.

### Networks

Three Flax `nn.Module` MLPs in `training/networks.py`:

- **`Actor`**: `obs -> logits [num_actions]` (policy)
- **`Critic`**: `obs -> scalar V(s)` (value function)
- **`QNetwork`**: `obs -> Q-values [num_actions]` (for DQN)

Action masking is external -- networks output raw logits, callers apply masks.

---

## 7. Testing

Run the full suite:

```bash
pytest tests/ -q
```

Test categories:

| Category   | Files                                         | What they verify                                    |
|------------|-----------------------------------------------|-----------------------------------------------------|
| Env        | `test_env.py`, `test_grid.py`, `test_movement.py`, `test_agents.py` | `obs_array`, `step_array`, collision, scanning       |
| Comm       | `test_comm_graph.py`                          | Adjacency, components, connectivity                  |
| Config     | `test_config.py`, `test_configs.py`           | YAML loading, defaults, legacy bridge, `to_env_config` |
| Rollout    | `test_rollout.py`                             | JIT compilation, vmap, determinism, guardrail         |
| Training   | `test_trainer.py`, `test_pg.py`               | All methods, multi-seed, loss trend                   |
| Runner     | `test_runner.py`                              | CLI parsing, config overrides, end-to-end             |
| Networks   | `test_networks.py`, `test_dqn.py`             | Forward pass shapes, init, Q-learning                 |
| Analysis   | `test_metrics.py`, `test_stats.py`, `test_transfer.py` | Coverage, statistical tests, CKA                |
| Viz        | `test_plotting.py`, `test_gif.py`, `test_visualizer.py` | Rendering, GIF output                          |

---

## 8. Live Architecture Dump (auto-generated)

Re-run `python scripts/architecture_dump.py` after editing
`src/red_within_blue/training/networks.py` or the `network.*` / `train.red_*`
fields in a config YAML. The script re-renders the tables below by calling
`flax.linen.nn.tabulate(...)` on the live modules — shapes, FLOPs, and
parameter counts never drift from the code.

The numbers below are taken from `configs/compromise-16x16-5-3b2r.yaml`
(`obs_dim = 23`, `num_actions = 5`, `num_red = 2`) — the setup that drives
the compromise-sweep meta-report.

Full rendered output: [`experiments/meta-report/architecture.txt`](../experiments/meta-report/architecture.txt).
A live HTML copy is embedded in **Appendix A** of
[`experiments/meta-report/meta_report.html`](../experiments/meta-report/meta_report.html).

**Parameter budget (C2 setup):**

| network          | params | bytes (fp32) | FLOPs / forward |
|------------------|-------:|-------------:|----------------:|
| `Actor`          | 20,229 | 80.9 kB      | 40,453          |
| `Critic`         | 19,713 | 78.9 kB      | 39,425          |
| `JointRedActor`  | 23,818 | 95.3 kB      | 47,626          |
| **Total live**   | **63,760** | **≈ 255 kB** | **127,504**   |

All three networks are 2×128 ReLU MLPs with a single final `Dense`. The
joint-red actor scales its input (`n_red · obs_dim`) and output
(`n_red · num_actions`) with team size — everything else is identical. Live
weights fit in CPU L2 cache; training throughput is rollout-bound, not
gradient-bound.
