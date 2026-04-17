# RL Training Guide: Policy Gradient Methods for RedWithinBlue

This document explains how to train agents in the RedWithinBlue grid exploration environment (built on JaxMARL [Rutherford et al., 2024]) using policy gradient reinforcement learning. It covers algorithm mechanics, input/output specifications, a three-phase training curriculum (single-agent PPO → multi-agent IPPO → collaborative MAPPO), model saving and transfer, regularization, experience replay, and alternatives to multi-component reward engineering (constrained RL, intrinsic motivation, multi-objective RL). The intended reader has the environment running and wants to build the training pipeline.

For a visual summary and navigation map across all docs, see [01-rl-overview.md](01-rl-overview.md). For a broader view of RL methods beyond policy gradient — value decomposition, GNN policies, communication learning, hierarchical RL, model-based approaches, and how training paradigms (CTDE, DTDE, Networked) map to different method families — see [03-rl-taxonomy.md](03-rl-taxonomy.md).

---

## 1. Environment Interface for Training

### Per-Agent Observation (Decentralized)

Each agent receives a flat vector of dimension **255** (with default `EnvConfig`):

| Component | Dim | Description |
|-----------|-----|-------------|
| `local_scan` (flattened) | 121 | `(2*obs_radius+1)^2` = 11x11 patch of terrain+occupancy around the agent |
| `map_fraction` | 1 | Fraction of the agent's local knowledge map that is non-unknown |
| `messages_in` | 129 | Aggregated messages from neighbors: `scan_dim(121) + msg_dim(8)` |
| `norm_pos` | 2 | Agent position normalized by grid size: `[row/H, col/W]` |
| `uid` | 1 | Agent unique identifier |
| `team_id` | 1 | Team assignment |
| **Total** | **255** | |

**Critical property:** `obs_dim` depends only on `obs_radius` and `msg_dim`, NOT grid size or agent count. This means the actor network architecture is identical across curriculum phases and weights transfer directly.

### Global State (Centralized Critic)

The centralized critic input from `env.get_global_state()` is a flat vector:

```
global_state_dim = N*2 + H*W + H*W + N*N + 1
                   ^^^   ^^^   ^^^   ^^^   ^
                   pos   terrain explored adj step
```

This dimension **changes** with grid size and agent count:

| Phase | Grid | Agents | Global Dim |
|-------|------|--------|-----------|
| Phase 1 | 8x8 | 1 | 132 |
| Phase 1 | 16x16 | 1 | 515 |
| Phase 2/3 | 32x32 | 4 | 2,073 |

The critic network cannot be transferred between phases — it must be re-initialized when grid size or agent count changes.

### Action Space

`Discrete(5)`: STAY=0, UP=1, RIGHT=2, DOWN=3, LEFT=4.

Set `num_actions=4` in `EnvConfig` to exclude STAY.

### Step Function

```python
obs, state, rewards, dones, info = env.step_env(key, state, actions)
```

| Return | Type | Contents |
|--------|------|----------|
| `obs` | `Dict[str, Array]` | Per-agent observations, shape `(255,)` each |
| `state` | `EnvState` | Full state (agent_state + global_state) |
| `rewards` | `Dict[str, Array]` | Per-agent float32 scalars |
| `dones` | `Dict[str, Array]` | Per-agent bools + `"__all__"` |
| `info` | `dict` | `global_state`, `adjacency`, `degree`, `num_components`, `is_connected`, `collisions`, `step` |

The centralized critic reads `info["global_state"]` (or call `env.get_global_state(state)`).

### Reward Functions

The environment returns zero rewards by default. Rewards are external, composable functions (following potential-based reward shaping principles [Ng et al., 1999]):

| Function | Signal | Typical Weight |
|----------|--------|---------------|
| `exploration_reward` | +1.0 per newly explored cell | 1.0 |
| `revisit_penalty` | -0.1 per revisited cell | 0.5 |
| `connectivity_reward` | -1.0 when comm graph is fragmented | 2.0 |
| `time_penalty` | -0.01 per step | 0.1 |
| `terminal_coverage_bonus` | Coverage fraction on final step | 5.0 |
| `competitive_reward` | +1.0 blue team / -1.0 red team for exploration | 1.0 |

Compose them:

```python
from red_within_blue import compose_rewards, exploration_reward, revisit_penalty, time_penalty, terminal_coverage_bonus

reward_fn = compose_rewards(
    exploration_reward, revisit_penalty, time_penalty, terminal_coverage_bonus,
    weights=[1.0, 0.5, 0.1, 5.0],
)
env = GridCommEnv(config, reward_fn=reward_fn)
```

---

## 2. Algorithm Deep Dives

### 2.1 REINFORCE (Baseline)

**How it works:** Based on the policy gradient theorem [Sutton et al., 1999], collect a full episode, compute discounted returns `G_t = sum(gamma^k * r_{t+k})`, update policy by ascending the gradient of `E[log pi(a_t|o_t) * (G_t - baseline)]`. The baseline is a moving average of returns to reduce variance.

**I/O:**

| Component | Input | Output |
|-----------|-------|--------|
| Actor | `obs [255]` | `logits [5]` → softmax → `pi(a\|o)` |
| Critic | — | — (no critic, uses return baseline) |

**Network:** `MLP(255 → 128 → 64 → 5)` with ReLU activations.

**Loss:**

```
L = -mean(log pi(a_t | o_t) * (G_t - b))
```

where `b` is a running mean of episode returns.

**Checkpoint contents:** `{actor_params, step, config}`.

**Verdict:** Unbiased but high-variance gradient. Very slow convergence. Useful only as a sanity check to verify the reward signal is learnable before investing in PPO. If REINFORCE learns nothing, the reward composition needs tuning before trying more complex algorithms.

---

### 2.2 PPO (Proximal Policy Optimization) — Phase 1

**How it works:** PPO [Schulman et al., 2017] collects a T-step rollout, computes advantages using Generalized Advantage Estimation (GAE) [Schulman et al., 2016], then performs K epochs of minibatch SGD on the collected data. The clipped surrogate objective prevents destructively large policy updates:

```
L_clip = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)

where r_t = pi_new(a_t|o_t) / pi_old(a_t|o_t)  [likelihood ratio]
```

**GAE (Generalized Advantage Estimation):**

```
delta_t = r_t + gamma * V(o_{t+1}) - V(o_t)         [TD residual]
A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}  [advantage]
```

GAE [Schulman et al., 2016] provides a bias-variance tradeoff controlled by `lambda`: at `lambda=1` it equals Monte Carlo returns (high variance, no bias), at `lambda=0` it equals one-step TD (low variance, high bias). `lambda=0.95` is the standard middle ground.

**I/O:**

| Component | Input | Output |
|-----------|-------|--------|
| Actor | `obs [255]` | `logits [5]` → softmax → `pi(a\|o)` |
| Critic | `obs [255]` | `V(o)` scalar |

For single-agent Phase 1, the critic can use the local observation (equivalent to global state when N=1). Alternatively, use `global_state` from `info` dict for a clean CTDE pattern even in Phase 1.

**Network architecture:**

```
Shared trunk: Linear(255 → 256) → tanh → Linear(256 → 128) → tanh
Actor head:   Linear(128 → 5)
Critic head:  Linear(128 → 1)
```

Using `tanh` rather than `ReLU` is standard in PPO — it bounds activations, which helps stability with the clipped objective.

**Combined loss:**

```
L = -L_clip + c1 * L_value - c2 * H(pi)

where:
  L_clip   = clipped surrogate (maximize → negate for loss)
  L_value  = (V(o_t) - R_t)^2     [value prediction error]
  H(pi)    = -sum(pi * log(pi))    [entropy bonus, maximize → negate]
  c1 = 0.5   [value loss coefficient]
  c2 = 0.01  [entropy coefficient]
```

**Training loop structure with RedWithinBlue:**

```python
import jax
import jax.numpy as jnp

# Vectorize environments for parallel rollouts
num_envs = 32
v_reset = jax.vmap(env.reset)
v_step  = jax.vmap(env.step_env)

# Reset all envs
keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
obs_batch, state_batch = v_reset(keys)
# obs_batch["agent_0"].shape == (32, 255)

# Rollout collection (use jax.lax.scan for efficiency)
def rollout_step(carry, _):
    state, key = carry
    key, k_act, k_step = jax.random.split(key, 3)
    
    # Policy forward pass (batched)
    obs = ...  # extract from state
    logits = actor_apply(actor_params, obs)       # (32, 5)
    actions = jax.random.categorical(k_act, logits)  # (32,)
    values = critic_apply(critic_params, obs)      # (32, 1)
    
    # Step all envs
    actions_dict = {"agent_0": actions}
    step_keys = jax.random.split(k_step, num_envs)
    obs_new, state_new, rewards, dones, info = v_step(step_keys, state, actions_dict)
    
    # Store transition
    transition = (obs, actions, rewards["agent_0"], dones["agent_0"], values, logits)
    return (state_new, key), transition

# Collect T-step rollout
(final_state, _), rollout = jax.lax.scan(
    rollout_step, (state_batch, key), None, length=rollout_length
)

# Compute GAE advantages, then K epochs of minibatch PPO updates
```

**Checkpoint contents:** `{actor_params, critic_params, optimizer_state, step, config, hyperparams}`.

**Strengths for RedWithinBlue:** Stable training, handles the partial observability and sparse reward structure well with proper entropy bonus. The vectorized rollout via `jax.vmap` enables training at 32+ parallel environments on a single GPU.

---

### 2.3 IPPO (Independent PPO) — Phase 2

**How it works:** Each agent runs its own PPO independently. Other agents are treated as part of the environment dynamics. With **parameter sharing** [Terry et al., 2020] (homogeneous agents), a single actor-critic network is applied to each agent's observations — this is literally single-agent PPO applied N times per step.

**I/O (with parameter sharing):**

| Component | Input | Output |
|-----------|-------|--------|
| Actor (shared) | `obs_i [255]` per agent | `logits_i [5]` per agent |
| Critic (shared) | `obs_i [255]` per agent | `V(o_i)` per agent |

The actor and critic are the same architecture as single-agent PPO. The only change is that at each step, the network processes N observations instead of 1:

```python
# Phase 2: apply the same actor to all agents
all_obs = jnp.stack([obs[f"agent_{i}"] for i in range(num_agents)])  # [N, 255]
all_logits = jax.vmap(actor_apply, in_axes=(None, 0))(shared_params, all_obs)  # [N, 5]
```

**Loss:** Same as PPO, averaged across all agents.

**Checkpoint contents:** Same as PPO (one set of shared params).

**When to use:** As the stepping stone between Phase 1 (single agent) and Phase 3 (MAPPO). IPPO with parameter sharing is often surprisingly competitive — it should be the first thing to try in multi-agent before investing in MAPPO's centralized critic.

**Weakness:** Each agent's critic only sees local observations. It cannot reason about what other agents are doing or the global coverage state. This limits coordination.

---

### 2.4 MAPPO (Multi-Agent PPO) — Phase 3

**How it works:** MAPPO [Yu et al., 2022] implements Centralized Training with Decentralized Execution (CTDE). Each agent has a **decentralized actor** that takes only its local observation. Training uses a **centralized critic** that takes the full global state. This lets the critic reason about all agents' positions, the full exploration map, and the communication graph — information that individual agents cannot see.

During execution (deployment), only the decentralized actor is needed. The critic is discarded.

**I/O:**

| Component | Input | Output | Shared? |
|-----------|-------|--------|---------|
| Actor | `obs_i [255]` per agent | `logits_i [5]` per agent | Yes (parameter sharing) |
| Critic | `global_state [2073]` | `V(s)` scalar | Yes (single critic) |

**Network architecture:**

```
Actor (same as PPO — transferable from Phase 1/2):
  Linear(255 → 256) → tanh → Linear(256 → 128) → tanh → Linear(128 → 5)

Critic (new — larger input):
  Linear(2073 → 512) → tanh → Linear(512 → 256) → tanh → Linear(256 → 1)
```

The actor architecture is identical to Phase 1/2 PPO. Weights transfer directly. The critic must be re-initialized because the input dimension changes.

**Advantage computation uses centralized V(s):**

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)    [TD residual with global state]
A_t = GAE(delta_t, gamma, lambda)
```

All agents share the same advantage estimates (since rewards and the global state are common in cooperative settings). Individual agent rewards can differ if using per-agent reward functions.

**Loss:** Same PPO clipped objective, but the value loss uses global state:

```
L_actor = -mean over agents [L_clip(r_t, A_t)]
L_critic = mean [(V(s_t) - R_t)^2]
L = L_actor + c1 * L_critic - c2 * H(pi)
```

**Checkpoint contents:** `{actor_params, critic_params, actor_optimizer_state, critic_optimizer_state, step, config, hyperparams}`.

**Strengths for RedWithinBlue:** The centralized critic sees the full exploration map, all agent positions, and the adjacency matrix. It can learn that fragmented groups are bad (via connectivity_reward) and that overlapping coverage is wasteful. This information flows back to the actors through the advantage estimates, guiding them toward coordinated exploration without any agent seeing the global picture directly.

---

### 2.5 Off-Policy Alternatives (Brief)

**SAC (Soft Actor-Critic)** [Haarnoja et al., 2018]: Designed for continuous action spaces. A discrete variant exists but PPO is more natural for `Discrete(5)`. SAC uses a replay buffer, which introduces staleness issues in MARL (see Section 7).

**QMIX** [Rashid et al., 2018]: Value decomposition method for cooperative MARL. Learns individual Q-functions mixed via a monotonic hypernetwork. Works with discrete actions. The `GraphTracker` data could feed the mixing network. However, QMIX requires substantially more implementation effort and PPO-based methods have been shown to match or exceed QMIX on cooperative tasks [Yu et al., 2022].

**Verdict:** PPO → IPPO → MAPPO is the recommended path. It's well-understood, stable, and maps cleanly onto RedWithinBlue's CTDE architecture. Off-policy methods add complexity without clear benefit for cooperative exploration.

---

## 3. The Training Curriculum

### Phase 1: Single-Agent Grid Scanning

**Goal:** Train one agent to explore ALL non-wall cells in minimum time on a small grid.

**Configuration:**

```python
config = EnvConfig(
    grid_width=8, grid_height=8,
    num_agents=1, max_steps=128,
    obs_radius=5, comm_radius=5.0, msg_dim=8,
    wall_density=0.0,
)

reward_fn = compose_rewards(
    exploration_reward, revisit_penalty, time_penalty, terminal_coverage_bonus,
    weights=[1.0, 0.5, 0.1, 5.0],
)
env = GridCommEnv(config, reward_fn=reward_fn)
```

**Why these rewards:**
- `exploration_reward` (w=1.0): primary signal — reward discovering new cells
- `revisit_penalty` (w=0.5): discourages circling back to already-seen areas
- `time_penalty` (w=0.1): mild pressure to be efficient
- `terminal_coverage_bonus` (w=5.0): strong reward at episode end for total coverage achieved

**Algorithm:** PPO (single-agent, Section 2.2).

**Success criteria:**
- 100% coverage of non-wall cells (36 cells on 8x8 with boundary walls) consistently
- Episode length < 50 steps (near-optimal for 36 cells)

**Scaling test:** Once 8x8 is solved, train on 16x16 (196 non-wall cells) using the same actor network. The obs_dim stays 255 — the actor architecture doesn't change. Only the critic needs adjustment if using global state (132 → 515). This validates that the learned scanning behavior generalizes to larger grids.

### Phase 2: Scale and Clone (Warm Start)

**Goal:** Take the trained Phase 1 policy, clone it to N agents on a larger grid, and verify they can explore without destructive interference.

**Configuration:**

```python
config = EnvConfig(
    grid_width=32, grid_height=32,
    num_agents=4, max_steps=256,
    obs_radius=5, comm_radius=5.0, msg_dim=8,
    wall_density=0.0,
)

# Same rewards as Phase 1 — no connectivity pressure yet
reward_fn = compose_rewards(
    exploration_reward, revisit_penalty, time_penalty, terminal_coverage_bonus,
    weights=[1.0, 0.5, 0.1, 5.0],
)
```

**Warm-starting procedure:**

```python
import numpy as np

# 1. Load Phase 1 actor params
phase1_ckpt = np.load("experiments/<phase1_id>/checkpoints/step_N.npz")
phase1_actor_params = reconstruct_params(phase1_ckpt)  # rebuild Flax pytree

# 2. Actor: direct parameter copy (obs_dim = 255, unchanged)
phase2_actor_params = phase1_actor_params

# 3. Critic: re-initialize (global_state_dim changed from 132 to 2073)
phase2_critic_params = critic_network.init(key, jnp.zeros(2073))

# 4. Optimizer: fresh state (or just for critic; keep actor optimizer momentum)
phase2_optimizer = optax.adam(learning_rate=1e-4)
phase2_opt_state = phase2_optimizer.init((phase2_actor_params, phase2_critic_params))
```

**Why this works:** The obs_dim is 255 regardless of grid size or agent count. The actor learned to read a local scan, track its knowledge map fraction, process incoming messages, and move toward unexplored areas. All of these behaviors are local and transfer directly. The only thing that changes is the scale of the world the agent operates in.

**Algorithm:** IPPO with parameter sharing (Section 2.3).

**Learning rate:** 1e-4 (lower than Phase 1's 3e-4). The warm-started actor already has a good exploration policy — we want gradual adaptation, not relearning from scratch.

**Success criteria:**
- Multi-agent coverage ≥ 80% on 32x32 within 256 steps
- Coverage per-agent-second improves over Phase 1 (4 agents should be more efficient than 1)
- Agents don't deadlock or oscillate in same region

### Phase 3: Collaborative MAPPO

**Goal:** Train agents to maintain communication graph connectivity while maximizing coverage. Agents should stay within comm range of at least one neighbor, explore efficiently as a coordinated team, and avoid fragmenting into isolated groups.

**Configuration:**

```python
config = EnvConfig(
    grid_width=32, grid_height=32,
    num_agents=4, max_steps=256,
    obs_radius=5, comm_radius=5.0, msg_dim=8,
    wall_density=0.1,  # obstacles make the task harder
)

reward_fn = compose_rewards(
    exploration_reward, revisit_penalty, connectivity_reward,
    time_penalty, terminal_coverage_bonus,
    weights=[1.0, 0.5, 2.0, 0.1, 5.0],
)
```

**Key addition: `connectivity_reward` (weight=2.0)**

This is the signal that teaches collaboration. Every agent receives -1.0 per step whenever the communication graph is fragmented (disconnected). With weight 2.0, this is the dominant penalty — agents learn that breaking the graph is more costly than leaving a cell unexplored. The tension between exploration_reward (spread out to cover more) and connectivity_reward (stay together to maintain the graph) is exactly the cooperative challenge.

**Algorithm:** MAPPO (Section 2.4).

**Transfer from Phase 2:**

```python
# Actor: from Phase 2 (already warm-started from Phase 1)
phase3_actor_params = phase2_actor_params

# Critic: re-initialize for MAPPO centralized critic
# Input is global_state [2073] instead of per-agent obs [255]
mappo_critic_params = mappo_critic_network.init(key, jnp.zeros(2073))
```

**Hyperparameter adjustments for fine-tuning:**
- Learning rate: 1e-4 (same as Phase 2, conservative)
- Entropy coefficient: 0.005 (lower than Phase 1's 0.01 — agents already have a good exploration policy, don't want too much randomness disrupting coordination)
- Clip epsilon: 0.1 (tighter than 0.2 — prevent sudden policy shifts that could destroy learned exploration behavior)

**Monitoring via GraphTracker:**

```python
from red_within_blue.comm_graph import get_fragmentation_count, get_agent_isolation_duration

# After each episode
tracker = final_state.global_state.graph
frag_steps = get_fragmentation_count(tracker)
print(f"Graph fragmented for {frag_steps}/{config.max_steps} steps")

for i in range(config.num_agents):
    iso = get_agent_isolation_duration(tracker, i)
    print(f"Agent {i} isolated for {iso} steps")
```

**Success criteria:**
- Coverage ≥ 90% on 32x32 with wall_density=0.1
- Communication graph connected ≥ 95% of timesteps
- No agent isolated for more than 3 consecutive steps
- Agents exhibit emergent formation behavior (spreading out while staying connected)

### Curriculum Summary

```
Phase 1 (PPO)          Phase 2 (IPPO)         Phase 3 (MAPPO)
  8x8, 1 agent    →     32x32, 4 agents   →    32x32, 4 agents
  Learn scanning   →     Scale + clone      →    Learn collaboration
  
  Actor params ────────→ Actor params ──────────→ Actor params
  (direct copy)          (direct copy)            (direct copy)
  
  Critic params          Critic params            Critic params
  (discard)              (re-init, 2073 dim)      (re-init, MAPPO)
```

---

## 4. Model Lifecycle

### What to Save

A complete checkpoint for resumable training contains:

| Component | Purpose | Required for... |
|-----------|---------|-----------------|
| `actor_params` | Policy network weights | Resume, fine-tune, inference |
| `critic_params` | Value network weights | Resume only |
| `optimizer_state` | Adam momentum/variance | Resume only |
| `step` | Training step counter | Resume, logging |
| `config` | `EnvConfig` as dict | Reproducibility |
| `hyperparams` | lr, gamma, etc. | Reproducibility |

### Saving with ExperimentLogger

The existing `ExperimentLogger` saves numpy `.npz` files. Flax params are JAX pytrees that need flattening:

```python
import jax
import numpy as np
from red_within_blue import ExperimentLogger

logger = ExperimentLogger(base_dir="experiments", experiment_name="phase1_ppo")
logger.log_config(config)
logger.log_hyperparams({"lr": 3e-4, "gamma": 0.99, "clip_eps": 0.2, ...})

# Flatten Flax params to saveable dict
def flatten_params(params, prefix=""):
    flat = {}
    leaves_with_path = jax.tree_util.tree_leaves_with_path(params)
    for path, leaf in zip(*jax.tree_util.tree_flatten_with_path(params)):
        key = prefix + "/".join(str(k) for k in path)
        flat[key] = np.asarray(leaf)
    return flat

# Save checkpoint
save_dict = {}
save_dict.update(flatten_params(actor_params, prefix="actor/"))
save_dict.update(flatten_params(critic_params, prefix="critic/"))
save_dict["step"] = np.array(training_step)
logger.save_checkpoint(save_dict, step=training_step)

# Save metrics every N steps
logger.log_metrics(step=training_step, metrics={
    "mean_reward": float(mean_reward),
    "coverage": float(coverage),
    "entropy": float(policy_entropy),
    "value_loss": float(value_loss),
    "policy_loss": float(policy_loss),
})
```

### Loading for Different Purposes

**Resume training** (same config, same phase):

```python
ckpt = np.load("experiments/<id>/checkpoints/step_5000.npz")

# Reconstruct actor params
actor_params = unflatten_params(ckpt, prefix="actor/")
critic_params = unflatten_params(ckpt, prefix="critic/")
step = int(ckpt["step"])

# Re-create optimizer and restore state
optimizer = optax.adam(lr)
opt_state = optimizer.init((actor_params, critic_params))
# Note: optimizer state is lost — training resumes with fresh momentum
# For full resume, save opt_state too (large but exact)
```

**Fine-tune** (new phase, curriculum transition):

```python
# Load only actor params
ckpt = np.load("experiments/<phase1>/checkpoints/step_final.npz")
actor_params = unflatten_params(ckpt, prefix="actor/")

# Re-initialize critic for new input dimension
new_critic_params = new_critic.init(key, jnp.zeros(new_global_dim))

# Fresh optimizer with lower learning rate
optimizer = optax.adam(1e-4)  # lower than original 3e-4
opt_state = optimizer.init((actor_params, new_critic_params))
```

**Inference / evaluation:**

```python
from red_within_blue import TrajectoryWrapper

ckpt = np.load("experiments/<id>/checkpoints/step_final.npz")
actor_params = unflatten_params(ckpt, prefix="actor/")

wrapper = TrajectoryWrapper(env, save_dir="eval_trajectories")
obs, state = wrapper.reset(key)

for step in range(config.max_steps):
    all_obs = jnp.stack([obs[a] for a in env.agents])
    logits = jax.vmap(actor_apply, in_axes=(None, 0))(actor_params, all_obs)
    actions_arr = jnp.argmax(logits, axis=-1)  # greedy (or sample for stochastic)
    actions = {env.agents[i]: actions_arr[i] for i in range(env.num_agents)}
    obs, state, rewards, dones, info = wrapper.step(step_key, state, actions)

wrapper.save_trajectory("eval_episode")
```

### Future: Orbax Checkpointing

For production use, consider upgrading to [Orbax](https://orbax.readthedocs.io/) (`pip install orbax-checkpoint`), which handles Flax pytree serialization natively, supports async checkpointing, and is the Flax team's recommended approach. The current `.npz` approach works well for prototyping.

---

## 5. Training Guardrails: Hard Enforcement Beyond Rewards

Rewards are suggestions — agents can learn to ignore them if the gradient signal is weak or noisy. Guardrails are enforcement: hard limits that the agent physically cannot violate. This section covers four layers of protection, ordered from simplest to most sophisticated.

### Layer 1: Action Masking — Prevent Unsafe Actions at the Source

**What:** Make `get_avail_actions(state)` return a context-dependent mask [Huang & Ontañón, 2022]. Before the policy samples an action, set the logits of unsafe actions to `-inf`. The agent literally cannot select a masked action.

**Currently:** `get_avail_actions` in `env.py:313` is a stub that returns all-ones. Standard JaxMARL PPO/MAPPO doesn't call it. We need to implement it and wire it into the training loop.

**Implementation:**

```python
# In env.py — replace the stub
def get_avail_actions(self, state: EnvState) -> Dict[str, chex.Array]:
    cfg = self.config
    positions = state.agent_state.positions  # [N, 2]
    terrain = state.global_state.grid.terrain  # [H, W]
    H, W = cfg.grid_height, cfg.grid_width
    
    # For each agent, for each action: check if target cell is passable
    # positions[:, None, :] is [N, 1, 2], ACTION_DELTAS_ARRAY[None, :, :] is [1, 5, 2]
    intended = positions[:, None, :] + ACTION_DELTAS_ARRAY[None, :, :]  # [N, 5, 2]
    intended_r = jnp.clip(intended[..., 0], 0, H - 1)
    intended_c = jnp.clip(intended[..., 1], 0, W - 1)
    
    cell_types = terrain[intended_r, intended_c]  # [N, 5]
    passable = (cell_types == CELL_EMPTY).astype(jnp.float32)  # [N, 5]
    
    # STAY is always safe
    passable = passable.at[:, 0].set(1.0)
    
    return {self.agents[i]: passable[i] for i in range(cfg.num_agents)}
```

**In the training loop — apply the mask:**

```python
# Policy forward pass
logits = actor_network.apply(params, obs)           # [N, 5]
avail = env.get_avail_actions(state)                 # Dict[str, [5]]
mask = jnp.stack([avail[a] for a in env.agents])     # [N, 5]

# Mask: -1e9 for unavailable actions, 0 for available
masked_logits = logits + jnp.where(mask, 0.0, -1e9)  # [N, 5]
action = jax.random.categorical(key, masked_logits)   # [N]
log_prob = jax.nn.log_softmax(masked_logits)           # for PPO ratio
```

**What this prevents:** Moving into walls and obstacles.

**JIT-compatible:** Yes — pure array operations, fully vmappable.

**Guarantee:** HARD — masked actions have zero probability.

### Layer 2: Connectivity Shield — Prevent Graph Fragmentation

**What:** A safety shield [Alshiekh et al., 2018] that, before executing actions, checks if the resulting state would isolate any agent (degree drops to 0). If so, override that agent's action with STAY.

This is inserted inside `step_env`, before `resolve_actions`:

```python
# Inside step_env, before line 174 (movement resolution)

# Predict next positions from proposed actions
deltas = ACTION_DELTAS_ARRAY[action_array]  # [N, 2]
intended = agent_state.positions + deltas    # [N, 2]
intended = jnp.clip(intended, 0, jnp.array([cfg.grid_height-1, cfg.grid_width-1]))

# Build hypothetical adjacency from intended positions
hyp_adjacency = comm_graph_mod.build_adjacency(intended, agent_state.comm_ranges)
hyp_degree = comm_graph_mod.compute_degree(hyp_adjacency)

# Any agent with degree 0 would be isolated — revert to STAY
would_isolate = (hyp_degree == 0)  # [N] bool
safe_actions = jnp.where(would_isolate, 0, action_array)  # 0 = STAY

# Use safe_actions instead of action_array for movement
new_positions, collision_mask = movement_mod.resolve_actions(
    agent_state.positions, safe_actions, grid_state.terrain,
    (cfg.grid_height, cfg.grid_width),
)
```

**What this prevents:** Individual agent isolation — no agent ever has zero neighbors.

**Cost:** One extra `build_adjacency` call per step (pairwise distances, O(N^2)). Negligible for N < 100.

**JIT-compatible:** Yes — `build_adjacency` is already JIT-compatible.

**Guarantee:** HARD — agents physically cannot become isolated.

**Limitation:** Preventing individual isolation doesn't guarantee full graph connectivity. The graph could fragment into two connected subgroups where every agent has degree >= 1 but the overall graph has 2 components. For full connectivity enforcement, see Layer 3.

### Layer 3: Full Connectivity Enforcement

**What:** Check if the proposed joint action would fragment the graph into multiple components. If so, find and override the minimal set of actions to maintain connectivity.

This is harder than Layer 2 because it requires reasoning about the joint effect of all actions simultaneously:

```python
# Check full connectivity after proposed moves
hyp_adjacency = comm_graph_mod.build_adjacency(intended_positions, comm_ranges)
num_components, is_connected = comm_graph_mod.compute_components(hyp_adjacency)

# If graph would fragment, selectively override actions
# Strategy: agents that would move away from the graph center → STAY
if not is_connected:
    # Compute current graph center (mean position of all agents)
    center = jnp.mean(agent_state.positions, axis=0)
    
    # For each agent: does their proposed move increase distance from center?
    current_dist = jnp.linalg.norm(positions - center, axis=1)
    intended_dist = jnp.linalg.norm(intended - center, axis=1)
    moving_away = intended_dist > current_dist
    
    # Override agents moving away to STAY
    safe_actions = jnp.where(moving_away, 0, action_array)
```

**JIT concern:** `compute_components` uses eigenvalue decomposition (O(N^3)), which is JIT-compatible but expensive. For N=4 this is negligible. For N > 20, consider the simpler degree-based check (Layer 2).

**Guarantee:** HARD — graph never fragments.

**Trade-off:** Overly conservative — may force agents to stay still when they could safely explore by moving in a different direction. The "stay if moving away" heuristic is simple but not optimal.

### Layer 4: Training Monitor — Detect and Recover from Instability

Layers 1-3 guard the environment. Layer 4 guards the training process itself.

**What to monitor:**

```python
class TrainingMonitor:
    """Detect training instability and trigger recovery."""
    
    def __init__(self, window=100):
        self.reward_history = []
        self.entropy_history = []
        self.violation_history = []
        self.best_checkpoint_path = None
        self.best_score = -float('inf')
    
    def check(self, metrics):
        self.reward_history.append(metrics['mean_reward'])
        self.entropy_history.append(metrics['policy_entropy'])
        self.violation_history.append(metrics['constraint_violations'])
        
        alerts = []
        
        # 1. Reward collapse: >30% drop over 100 episodes
        if len(self.reward_history) > 200:
            recent = np.mean(self.reward_history[-100:])
            baseline = np.mean(self.reward_history[-200:-100])
            if baseline > 0 and recent < 0.7 * baseline:
                alerts.append('REWARD_COLLAPSE')
        
        # 2. Entropy collapse: policy becoming deterministic too early
        if len(self.entropy_history) > 50:
            max_entropy = np.log(5)  # log(num_actions)
            if self.entropy_history[-1] < 0.1 * max_entropy:
                alerts.append('ENTROPY_COLLAPSE')
        
        # 3. Constraint violation spike
        if len(self.violation_history) > 100:
            recent_rate = np.mean(self.violation_history[-10:])
            baseline_rate = np.mean(self.violation_history[-100:-10])
            if recent_rate > 2.0 * max(baseline_rate, 0.01):
                alerts.append('VIOLATION_SPIKE')
        
        return alerts
    
    def save_if_best(self, score, checkpoint_fn):
        if score > self.best_score:
            self.best_score = score
            self.best_checkpoint_path = checkpoint_fn()
    
    def rollback(self):
        """Load last known good checkpoint."""
        if self.best_checkpoint_path:
            return load_checkpoint(self.best_checkpoint_path)
        return None
```

**Recovery strategy:**

| Alert | Response |
|-------|----------|
| `REWARD_COLLAPSE` | Rollback to best checkpoint, reduce learning rate by 50% |
| `ENTROPY_COLLAPSE` | Increase entropy coefficient by 2x |
| `VIOLATION_SPIKE` | Increase Lagrange multiplier or tighten action mask |

**Guarantee:** SOFT — detection and recovery, not prevention.

### Layer 5: Curriculum Gates — Don't Advance Until Safe

Hard gates that prevent phase transitions until safety criteria are met:

```python
# Phase transition criteria
PHASE_GATES = {
    1: {  # Phase 1 → Phase 2
        'min_coverage': 0.95,           # 95% grid coverage
        'min_episodes': 500,            # at least 500 episodes
        'eval_window': 100,             # evaluated over last 100 episodes
    },
    2: {  # Phase 2 → Phase 3
        'min_coverage': 0.80,           # 80% coverage with 4 agents
        'max_fragmentation_rate': 0.10, # graph fragmented < 10% of steps
        'min_episodes': 500,
    },
    3: {  # Phase 3 success criteria
        'min_coverage': 0.90,
        'max_fragmentation_rate': 0.05, # < 5%
        'max_isolation_steps': 3,       # no agent isolated > 3 consecutive steps
    },
}

def check_gate(phase, episode_metrics, window=100):
    gate = PHASE_GATES[phase]
    recent = episode_metrics[-window:]
    
    coverage_ok = np.mean([m['coverage'] for m in recent]) >= gate['min_coverage']
    episodes_ok = len(episode_metrics) >= gate['min_episodes']
    
    if 'max_fragmentation_rate' in gate:
        frag_ok = np.mean([m['frag_rate'] for m in recent]) <= gate['max_fragmentation_rate']
    else:
        frag_ok = True
    
    return coverage_ok and episodes_ok and frag_ok
```

### Putting It All Together

The layers stack — each one catches what the layer below doesn't:

```
Layer 1: Action Mask      → Agent can't select wall-moves
Layer 2: Connectivity Shield → Agent can't isolate itself  
Layer 3: Full Connectivity   → Graph can't fragment
Layer 4: Training Monitor    → Training can't silently diverge
Layer 5: Curriculum Gate     → Can't advance to harder phase without proving safety
```

**Recommended starting stack:** Layers 1 + 2 + 4 + 5. Skip Layer 3 initially (the Lagrangian constraint from Section 7 handles connectivity more efficiently than hard overrides). Add Layer 3 only if Lagrangian enforcement alone isn't sufficient.

| Layer | Prevents | Guarantee | JIT-Safe | Implementation |
|-------|----------|-----------|----------|---------------|
| Action Mask | Wall collisions | HARD | Yes | ~30 LOC in `env.py` |
| Connectivity Shield | Agent isolation | HARD | Yes | ~20 LOC in `env.py` |
| Full Connectivity | Graph fragmentation | HARD | Yes (O(N^3)) | ~40 LOC in `env.py` |
| Training Monitor | Training instability | SOFT | N/A (outer loop) | ~60 LOC in training script |
| Curriculum Gate | Premature phase advance | HARD | N/A (outer loop) | ~30 LOC in curriculum script |

---

## 6. Regularization

### Entropy Regularization — ESSENTIAL

**What:** Add an entropy bonus to the policy loss: `-c2 * H(pi(.|o))` where `H(pi) = -sum(pi(a|o) * log(pi(a|o)))`.

**Why critical for RedWithinBlue:** The exploration reward becomes sparse as the episode progresses — early on, every step discovers new cells, but late in the episode most reachable cells are already explored. Without entropy regularization, the policy collapses to deterministic behavior (e.g., always move RIGHT) before it learns systematic scanning patterns. The entropy bonus keeps the action distribution spread out, maintaining exploration of the action space.

**Recommended values:**
- Phase 1: `c2 = 0.01` — strong exploration pressure while learning from scratch
- Phase 2: `c2 = 0.01` — same, agents are adapting to new scale
- Phase 3: `c2 = 0.005` — slightly lower because the warm-started policy already has good exploration behavior; too much entropy disrupts the coordination signal from connectivity_reward

**Decay schedule:** Optionally anneal entropy coefficient linearly from 0.01 to 0.001 over training. This lets the policy become more decisive as it improves.

### PPO Clipping — IMPORTANT

**What:** The clipped surrogate objective `clip(r_t, 1-eps, 1+eps)` bounds the policy ratio, preventing any single update from changing the policy too much.

**Why important:** During curriculum phase transitions (Phase 2 → Phase 3), the addition of connectivity_reward creates a sudden shift in the reward landscape. Without clipping, the first few gradient updates could destroy the learned exploration behavior as the policy tries to optimize the new connectivity signal. Tighter clipping preserves the exploration skill while gradually incorporating collaboration.

**Recommended values:**
- Phase 1-2: `eps = 0.2` (standard)
- Phase 3: `eps = 0.1` (tighter — protect the warm-started exploration policy during fine-tuning)

### Gradient Clipping — IMPORTANT

**What:** Clip the global gradient norm before applying the optimizer update: `grad = grad * min(1, max_norm / ||grad||)`.

**Why important:** JAX's pure functional approach means gradients can occasionally spike, especially early in training with random policies or when the spectral computation in GraphTracker produces extreme values. Gradient clipping prevents these spikes from corrupting the optimizer state.

**Recommended:** `max_grad_norm = 0.5` throughout all phases.

```python
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=lr),
)
```

### Policy Distillation — SITUATIONAL

**What:** When transferring between phases, add a KL divergence term between the old (teacher) policy and the new (student) policy: `L_distill = KL(pi_teacher || pi_student)`.

**When useful:** If the observation dimension changes between phases (e.g., different `obs_radius` or `msg_dim`). In that case, you can't directly copy actor params — distillation transfers the behavioral knowledge through a soft constraint.

**For the default curriculum:** NOT NEEDED. The obs_dim is 255 in all phases, so direct parameter copying works. Keep distillation as a tool for future experiments where you might change `obs_radius` or `msg_dim` between phases.

### L2 Weight Regularization — MARGINAL

**What:** Add `lambda * ||theta||^2` to the loss.

**Verdict:** The networks are small (2-3 layers, 128-256 units) and not prone to overfitting. L2 regularization provides marginal benefit. If used, `lambda = 1e-4`.

### Summary

| Technique | Importance | Phase 1 | Phase 2 | Phase 3 |
|-----------|-----------|---------|---------|---------|
| Entropy bonus | Essential | 0.01 | 0.01 | 0.005 |
| PPO clip eps | Important | 0.2 | 0.2 | 0.1 |
| Gradient clip | Important | 0.5 | 0.5 | 0.5 |
| Policy distillation | Situational | — | — | — |
| L2 weight decay | Marginal | — | — | — |

---

## 7. Experience Replay

### On-Policy Methods Don't Use Replay Buffers

PPO and MAPPO are on-policy algorithms. They collect a rollout of T steps using the current policy, compute advantages, perform K epochs of minibatch updates on that rollout, then **discard the data** and collect a fresh rollout. This is fundamental — the policy gradient estimate requires data from the current policy, not a historical one.

GAE (Generalized Advantage Estimation) is the mechanism that makes on-policy methods sample-efficient enough to work without replay. By bootstrapping from the critic's value estimates, GAE produces low-variance advantage estimates from a single rollout without needing to average over many episodes.

### Why Replay is Problematic in MARL

In multi-agent settings, experience replay introduces a **non-stationarity problem**: transitions stored when other agents had policy `v1` become stale when those agents have evolved to policy `v2`. The stored transition `(s, a, r, s')` assumed a particular joint behavior that no longer holds. This makes the Q-values or returns computed from replayed data systematically biased.

Off-policy MARL methods (like QMIX with replay) mitigate this via importance sampling weights, but this adds complexity and variance. For cooperative exploration, the added complexity is not justified.

### TrajectoryWrapper is NOT a Replay Buffer

The `TrajectoryWrapper` in `wrappers.py` records full episodes for post-hoc analysis. It is:
- NOT JIT-compatible (Python-side lists)
- NOT designed for training (no sampling interface)
- Used for evaluation, debugging, and visualization

### What Trajectory Data IS Useful For

Even though replay isn't used for training, recorded trajectories serve important purposes:

1. **Coverage analysis:** Replay episodes to see how efficiently the policy scans the grid. Identify blind spots or repetitive patterns.
2. **Fragmentation debugging:** Use the `GraphTracker` data within trajectories to pinpoint exactly when and why the communication graph fragments.
3. **Demonstration data:** Recorded trajectories from a trained policy can serve as demonstrations for future imitation learning or hindsight experience replay [Andrychowicz et al., 2017] experiments.
4. **Visualization and presentation:** Use `ReplayPlayer` with `render_frame` to create videos of agent behavior.

### What to Focus on Instead

Instead of replay, invest in:
- **Rollout length:** Longer rollouts (128-256 steps) give GAE more data to work with per update.
- **Number of parallel environments:** Use `jax.vmap` to run 32+ environments simultaneously, increasing data throughput.
- **PPO epochs:** Reuse each rollout for 4 epochs of minibatch updates (the clipped objective makes this safe).

```python
# Maximize data throughput without replay
num_envs = 32           # parallel environments via vmap
rollout_length = 256    # steps per rollout
ppo_epochs = 4          # reuse each rollout 4 times
minibatches = 4         # split rollout into 4 minibatches per epoch

# Effective batch size per update: 32 * 256 = 8,192 transitions
# Used for: 4 epochs * 4 minibatches = 16 gradient updates per rollout
```

---

## 8. Beyond Reward Engineering: Better Approaches

The composed reward approach (`compose_rewards` with 5-6 weighted terms) works, but it has a fundamental problem: **balancing those weights is itself a hyperparameter search**. Changing one weight shifts the relative importance of all others. The reward landscape is sensitive to these ratios, and there's no principled way to pick them — you just tune until behavior looks right.

This section presents three approaches that largely eliminate this problem, ordered by recommendation strength.

### 7.1 Constrained RL with Lagrangian Multipliers — RECOMMENDED

**The core idea:** Constrained policy optimization [Achiam et al., 2017] reframes reward balancing: instead of encoding connectivity and time efficiency as *rewards* with hand-tuned weights, encode them as *constraints* with interpretable thresholds. The algorithm automatically learns how strongly to enforce each constraint. First-order methods like FOCOPS [Zhang et al., 2020] make this practical for deep RL.

**Formulation:**

```
maximize    E[exploration_reward]              ← single clear objective
subject to  E[connectivity] >= 0.95            ← graph connected 95% of steps
            E[episode_length] <= time_budget    ← coverage achieved in time
```

This is solved via the Lagrangian:

```
L(theta, lambda) = E[r_exploration] - lambda_1 * (0.95 - E[connectivity])
                                     - lambda_2 * (E[episode_length] - budget)
```

The Lagrange multipliers `lambda_1, lambda_2` are **learned alongside the policy**. If the connectivity constraint is violated, `lambda_1` increases automatically, making the agent care more about staying connected. If the constraint is satisfied with slack, `lambda_1` decreases. No manual tuning.

**Why this is better for RedWithinBlue:**

| Reward Engineering | Constrained RL |
|---|---|
| "Set connectivity_weight to 2.0" (why 2.0?) | "Graph must be connected 95% of the time" (interpretable) |
| Changing exploration_weight shifts connectivity balance | Constraints are independent — changing one doesn't affect others |
| 5-6 weights to tune | 2-3 thresholds to set (with clear physical meaning) |
| Weights have no units | Thresholds have natural units (%, steps) |

**Implementation sketch:**

```python
import jax.numpy as jnp
import optax

# Constraint definitions (interpretable thresholds, not arbitrary weights)
CONNECTIVITY_THRESHOLD = 0.95   # graph connected 95% of steps
TIME_BUDGET = 200               # achieve 90% coverage within 200 steps

# Learnable Lagrange multipliers (initialized small, auto-tuned)
lambda_connectivity = jnp.float32(0.1)
lambda_time = jnp.float32(0.1)
dual_lr = 1e-3  # learning rate for multipliers

# PPO loss becomes:
def constrained_ppo_loss(actor_params, critic_params, batch,
                         lambda_conn, lambda_time):
    # Standard PPO components
    policy_loss = ppo_clip_loss(actor_params, batch)
    value_loss = value_fn_loss(critic_params, batch)
    entropy = entropy_bonus(actor_params, batch)
    
    # Constraint violations (from batch statistics)
    connectivity_violation = CONNECTIVITY_THRESHOLD - batch["mean_connectivity"]
    time_violation = batch["mean_episode_length"] - TIME_BUDGET
    
    # Lagrangian penalty (multipliers are detached — updated separately)
    lagrangian = (jax.lax.stop_gradient(lambda_conn) * connectivity_violation
                + jax.lax.stop_gradient(lambda_time) * time_violation)
    
    return policy_loss + 0.5 * value_loss - 0.01 * entropy + lagrangian

# Dual update (after each PPO epoch):
def update_multipliers(lambda_conn, lambda_time, batch_stats):
    conn_violation = CONNECTIVITY_THRESHOLD - batch_stats["mean_connectivity"]
    time_violation = batch_stats["mean_episode_length"] - TIME_BUDGET
    
    # Gradient ascent on multipliers (maximize Lagrangian w.r.t. lambda)
    lambda_conn = jnp.maximum(0.0, lambda_conn + dual_lr * conn_violation)
    lambda_time = jnp.maximum(0.0, lambda_time + dual_lr * time_violation)
    return lambda_conn, lambda_time
```

**What you keep vs. what you drop:**

| Before (reward weights) | After (constraints) |
|---|---|
| `exploration_reward` (w=1.0) | Keep as primary objective (no weight needed — it's THE objective) |
| `revisit_penalty` (w=0.5) | Drop — replace with intrinsic motivation (see 7.2) |
| `connectivity_reward` (w=2.0) | Drop → constraint: `connectivity >= 0.95` |
| `time_penalty` (w=0.1) | Drop → constraint: `episode_length <= budget` |
| `terminal_coverage_bonus` (w=5.0) | Drop — absorbed into primary exploration objective |
| `competitive_reward` (w=1.0) | Keep if needed (competitive scenarios only) |

**Result: 5 arbitrary weights → 2 interpretable thresholds.**

Recent work [Lu et al., 2024] confirms this approach is stable in multi-agent settings with shared constraints and decentralized actors — exactly our MAPPO setup.

### 7.2 Intrinsic Motivation (RND) — Replace Exploration Rewards

**The problem with explicit exploration_reward + revisit_penalty:** You're manually defining what "novel" means (cell visit count == 0) and what "revisiting" costs (-0.1 per step). These are brittle — the agent optimizes the metric, not the underlying behavior. An agent might learn to touch cells briefly without actually scanning them, or avoid cells it visited 3 steps ago even if re-scanning would help.

**Random Network Distillation (RND)** [Burda et al., 2019] provides a principled alternative (building on the earlier ICM framework [Pathak et al., 2017]):

1. Initialize a random, frozen target network `f_target(obs) -> embedding`
2. Train a predictor network `f_predictor(obs) -> embedding` to match the target
3. The prediction error `||f_target(obs) - f_predictor(obs)||^2` is the intrinsic reward
4. Novel states have high error (predictor hasn't seen them). Familiar states have low error.

```python
import flax.linen as nn

class RNDTarget(nn.Module):
    """Frozen random network — never trained."""
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        return x

class RNDPredictor(nn.Module):
    """Trained to match target — prediction error = novelty."""
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        return x

# Intrinsic reward = prediction error
def rnd_reward(target_params, predictor_params, obs):
    target_embed = target_network.apply(target_params, obs)
    pred_embed = predictor_network.apply(predictor_params, obs)
    return jnp.mean((target_embed - pred_embed) ** 2, axis=-1)

# Total reward = task_reward + rnd_coefficient * rnd_reward
# Single hyperparameter: rnd_coefficient (typically 0.01 - 0.1)
```

**Why RND is better than explicit exploration/revisit rewards:**
- Automatically discovers what's "novel" without hard-coding visit counts
- Naturally decays as the agent covers the grid (predictor learns → error drops)
- Single hyperparameter (`rnd_coefficient`) instead of two weights
- Works with partial observability — novelty is based on the observation, not global state
- Battle-tested since 2019 [Burda et al., 2019], minimal implementation overhead (~50 LOC)

**For grid exploration specifically:** RND captures not just "have I been to this cell" but "have I seen this configuration of terrain + occupancy + messages." An agent returning to a cell with different neighbors nearby gets different intrinsic reward than one returning to the same cell alone. This naturally encourages diverse exploration patterns.

### 7.3 Multi-Objective RL (MORL) — Eliminate Weights Entirely

**The idea:** Instead of scalarizing objectives into one weighted sum, treat each as a separate objective and find the **Pareto front** — the set of policies where you can't improve one objective without worsening another [Liu et al., 2025].

```
Objective 1: maximize exploration coverage
Objective 2: maximize graph connectivity
Objective 3: minimize time to coverage
```

Train a set of policies, each representing a different trade-off. At deployment time, pick the one that matches your priorities.

**Why this eliminates weight tuning:** You don't choose weights. You get a menu of trained policies:
- Policy A: 95% coverage, 100% connected, 200 steps (conservative)
- Policy B: 99% coverage, 85% connected, 250 steps (aggressive exploration)
- Policy C: 90% coverage, 98% connected, 150 steps (time-efficient)

The user selects the policy that fits the deployment scenario.

**Practical approach — Pareto Conditioned Networks:**

Train a single network conditioned on a preference vector `w`:

```python
class ParetoActor(nn.Module):
    """Policy conditioned on preference vector."""
    @nn.compact
    def __call__(self, obs, preference):
        # preference: [w_exploration, w_connectivity, w_time] summing to 1
        x = jnp.concatenate([obs, preference], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        return nn.Dense(5)(x)  # action logits

# Train with varied preferences per rollout
# At deployment: pick preference that matches your priorities
```

**When MORL makes sense:** When you genuinely don't know the right trade-off in advance, or when different deployment scenarios need different trade-offs. It's more work than constrained RL but provides maximum flexibility.

**When it's overkill:** If you know connectivity is a hard requirement (not negotiable), use constrained RL instead. MORL is for when you're exploring the trade-off space.

### 7.4 Comparison and Recommendation

| Approach | Weights to Tune | Interpretability | Implementation Effort | Best For |
|---|---|---|---|---|
| **Current** (compose_rewards) | 5-6 arbitrary weights | Low | Already done | Quick prototyping |
| **Constrained RL** | 2-3 thresholds | High | ~2-3 weeks | Production training |
| **RND** (intrinsic motivation) | 1 coefficient | Medium | ~1 week | Replacing exploration/revisit signals |
| **MORL** (Pareto) | 0 (choose at runtime) | High | ~3-4 weeks | Unknown trade-offs |
| **PBT** (auto-tuning) | 0 (searched automatically) | Low | ~2 weeks | "Just find good weights for me" |

**Recommended path:**

1. **Start** with `compose_rewards` (current approach) to validate the environment and training loop work end-to-end.

2. **Then** replace `exploration_reward` + `revisit_penalty` with RND. This cuts 2 weights to 1 coefficient and gives cleaner exploration signal.

3. **Then** replace `connectivity_reward` + `time_penalty` with Lagrangian constraints. This cuts 2 more weights to 2 interpretable thresholds.

4. **End state:** Primary objective (exploration) + 1 curiosity coefficient + 2 constraint thresholds. Down from 5-6 arbitrary weights to 3 interpretable parameters.

```
Before:  compose_rewards(exploration, revisit, connectivity, time, terminal,
                         weights=[1.0, 0.5, 2.0, 0.1, 5.0])  ← 5 fragile numbers

After:   reward = exploration_from_rnd(rnd_coeff=0.05)  ← 1 number
         constraints = {connectivity >= 0.95, steps <= 200}  ← 2 thresholds
```

### 7.5 Population-Based Training (Automatic Weight Search)

If you want to keep `compose_rewards` but don't want to hand-tune weights, **Population-Based Training (PBT)** [Jaderberg et al., 2017] automates the search:

1. Spawn 10-20 agents with random reward weight vectors
2. Train all in parallel (JAX makes this efficient via `jax.vmap` over weight configs)
3. Periodically evaluate each agent's coverage + connectivity metrics
4. Copy weights from top performers, mutate to explore nearby weight vectors
5. After N generations, the best weight vector emerges

```python
# PBT pseudocode
population_size = 16
weight_vectors = jax.random.uniform(key, (population_size, 5))  # random weights

for generation in range(num_generations):
    # Train each member for K steps (all in parallel via vmap)
    fitness = evaluate_all(weight_vectors)  # coverage * connectivity
    
    # Exploit: bottom 25% copies top 25%
    # Explore: mutate copied weights by ±20%
    weight_vectors = exploit_and_explore(weight_vectors, fitness)

best_weights = weight_vectors[jnp.argmax(fitness)]
```

This is the pragmatic "just find good weights" approach. Less principled than constrained RL but requires minimal architectural changes — you keep `compose_rewards` as-is.

---

## 9. Hyperparameter Reference (Baseline Approach)

| Hyperparameter | Phase 1 (PPO) | Phase 2 (IPPO) | Phase 3 (MAPPO) |
|----------------|---------------|----------------|-----------------|
| **Environment** | | | |
| Grid size | 8x8 (then 16x16) | 32x32 | 32x32 |
| Num agents | 1 | 4 | 4 |
| Max steps/episode | 128 | 256 | 256 |
| Wall density | 0.0 | 0.0 | 0.1 |
| Obs radius | 5 | 5 | 5 |
| Comm radius | 5.0 | 5.0 | 5.0 |
| **Algorithm** | | | |
| Method | PPO | IPPO (shared) | MAPPO |
| Actor obs dim | 255 | 255 | 255 |
| Critic input dim | 255 (or 132) | 255 (or 2073) | 2073 |
| Actor network | MLP(255,256,128,5) | Same | Same |
| Critic network | MLP(255,256,128,1) | Same | MLP(2073,512,256,1) |
| Param sharing | N/A (1 agent) | Yes | Yes (actors) |
| **Optimization** | | | |
| Learning rate | 3e-4 | 1e-4 | 1e-4 |
| Gamma (discount) | 0.99 | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 | 0.95 |
| Clip epsilon | 0.2 | 0.2 | 0.1 |
| Entropy coeff | 0.01 | 0.01 | 0.005 |
| Value coeff | 0.5 | 0.5 | 0.5 |
| Max grad norm | 0.5 | 0.5 | 0.5 |
| **Rollout** | | | |
| Num envs (vmap) | 32 | 32 | 32 |
| Rollout length | 128 | 128 | 256 |
| PPO epochs | 4 | 4 | 4 |
| Minibatches | 4 | 4 | 4 |
| **Training budget** | | | |
| Total timesteps | ~1M | ~2M | ~5M |
| **Rewards** | | | |
| exploration_reward | 1.0 | 1.0 | 1.0 |
| revisit_penalty | 0.5 | 0.5 | 0.5 |
| connectivity_reward | — | — | 2.0 |
| time_penalty | 0.1 | 0.1 | 0.1 |
| terminal_coverage_bonus | 5.0 | 5.0 | 5.0 |

---

## 10. Implementation Roadmap

The environment is complete (12 modules, 60 tests). The training infrastructure is built bottom-up from raw RL mechanics. Each layer adds one concept. Move to the next layer only when the current one hits a concrete wall.

### Layer 0: Policy Network + Random Baseline

**File:** `src/red_within_blue/networks.py`

Start with the simplest possible policy — an MLP that maps observations to action logits:

```python
class PolicyNet(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(128)(obs)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        return nn.Dense(5)(x)  # action logits
```

Run it with random weights. Record coverage, episode length, connectivity. This is your floor — everything must beat random.

### Layer 1: Raw Policy Gradient

**File:** `src/red_within_blue/train.py`

The simplest thing that learns. Collect a full episode, compute discounted returns, push the policy toward actions that led to good outcomes:

```python
loss = -mean(log_prob(action_taken) * discounted_return)
params = params - lr * grad(loss)
```

No critic. No baseline. No clipping. No minibatches. Just: run episode → compute returns → gradient step → repeat. This will be noisy and slow, but if the reward signal is learnable at all, you'll see it here.

**What you learn:** Does the reward composition produce a gradient signal the network can follow? If raw policy gradient learns nothing on 8x8, adding PPO on top won't fix it — the reward design needs work first.

### Layer 2: Subtract a Baseline (Variance Reduction)

Same loop, but subtract the average return to reduce variance:

```python
baseline = running_mean(episode_returns)
loss = -mean(log_prob(action_taken) * (return - baseline))
```

**What you learn:** How much of Layer 1's noise was variance vs. fundamental difficulty. If this converges significantly faster, variance was the bottleneck.

### Layer 3: Add a Critic (Actor-Critic)

Add a value network V(o) that predicts expected return from a state. Use the TD error `r + gamma*V(next) - V(current)` as the advantage instead of full episode returns:

```python
# Value network (separate or shared trunk)
class ValueNet(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(128)(obs)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        return nn.Dense(1)(x)  # scalar value

advantage = reward + gamma * V(next_obs) - V(obs)
actor_loss = -mean(log_prob(action) * stop_gradient(advantage))
critic_loss = mean((V(obs) - target_return) ** 2)
```

**What you learn:** Whether bootstrapping from a learned value function helps — it should, because 256-step episodes have high-variance returns. This is where training gets meaningfully faster.

### Layer 4: Stability Tricks (Clipping, GAE, Entropy)

Now you're hitting the walls that motivated PPO. Add them one at a time:

1. **GAE** — replace raw TD advantage with the lambda-weighted sum. Smoother advantages, less variance.
2. **PPO clipping** — reuse data for multiple gradient steps safely. Better sample efficiency.
3. **Entropy bonus** — prevent premature policy collapse to a single action.
4. **Gradient clipping** — prevent occasional gradient spikes from corrupting training.

Each of these exists because someone hit a specific problem. Add each one when YOU hit that problem, and you'll understand exactly what it does.

### Layer 5: Parallel Rollouts

**File:** `scripts/train.py`

Use `jax.vmap` to run 32 environments in parallel. Use `jax.lax.scan` for efficient rollout collection. This doesn't change the algorithm — it just gives you 32x more data per wall-clock second.

### Layer 6: Multi-Agent (Parameter Sharing)

Apply the same single-agent policy to all N agents independently. Each agent's obs goes through the same network. This is IPPO — but you built it by just running your single-agent code N times per step. No new algorithm, just a for-loop (or vmap over agents).

### Layer 7: Centralized Critic

Add a second value network that takes the full global state (2073-dim) instead of the agent-local obs (255-dim). Use it for advantage estimation during training; discard it at execution. This is the CTDE pattern — but it's just "give the critic more information."

### Layer 8: Everything Else

From here, additions are driven by observed failures:

- **Connectivity reward not working?** → Try Lagrangian constraints (Section 8.1)
- **Exploration stalls?** → Try RND intrinsic motivation (Section 8.2)
- **Need better coordination?** → Try GNN policy over comm graph (see [03-rl-taxonomy.md](03-rl-taxonomy.md))
- **Want automatic curriculum?** → Try PAIRED/ACCEL environment design
- **Want to scale to N=16+?** → Try transformer/attention policy (UPDeT)

### Evaluation (parallel to all layers)

**File:** `scripts/eval.py`

- Load checkpoint → run episodes with `TrajectoryWrapper`
- Compute coverage, connectivity, fragmentation metrics
- Render visualization frames / videos
- Compare every layer against the random baseline and the previous layer

---

## 11. References

Peer-reviewed publications are listed with venue and year. Preprints that lack peer review are marked with *(preprint)*.

### Foundational RL

- **[Sutton et al., 1999]** R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *Advances in Neural Information Processing Systems (NeurIPS)*, 1999.

- **[Schulman et al., 2016]** J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *International Conference on Learning Representations (ICLR)*, 2016.

- **[Schulman et al., 2017]** J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*, 2017. *(preprint — not formally peer-reviewed, but the de facto standard on-policy algorithm widely adopted across the field)*

### Multi-Agent RL

- **[Yu et al., 2022]** C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Baez, and Y. Wu. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

- **[Rashid et al., 2018]** T. Rashid, M. Samvelyan, C. S. de Witt, G. Farquhar, J. Foerster, and S. Whiteson. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." *International Conference on Machine Learning (ICML)*, 2018.

- **[Terry et al., 2020]** J. Terry, N. Grammel, S. Son, and B. Black. "Parameter Sharing is Surprisingly Useful for Multi-Agent Deep Reinforcement Learning." *arXiv:2005.13625*, 2020. *(preprint)*

- **[Rutherford et al., 2024]** A. Rutherford, B. Ellis, M. Gallici, J. Cook, A. Lupu, G. Ingvarsson, T. Willi, A. Khan, C. de Witt, A. Souly, S. Bandyopadhyay, M. Golber, E. Jiang, Y. Tewari, F. Christianos, M. Papoudakis, G. Georgiev, G. Palmer, R. Lange, N. Maymin, G. Maymin, A. J. Hughes, D. Radetic, S. Mohanty, E. Gorsane, R. Salakhutdinov, N. Bishop, and J. Foerster. "JaxMARL: Multi-Agent RL Environments and Algorithms in JAX." *Autonomous Agents and Multi-Agent Systems (AAMAS)*, 2024.

### Constrained and Safe RL

- **[Achiam et al., 2017]** J. Achiam, D. Held, A. Tamar, and P. Abbeel. "Constrained Policy Optimization." *International Conference on Machine Learning (ICML)*, 2017.

- **[Zhang et al., 2020]** Y. Zhang, Q. Vuong, and K. Ross. "First Order Constrained Optimization in Policy Space." *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

- **[Lu et al., 2024]** S. Lu, K. Zhang, R. Chen, Z. Yang, and T. Basar. "Decentralized Policy Gradient for Nash Equilibria in Multi-Player General-Sum Constrained Markov Games." *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

- **[Alshiekh et al., 2018]** M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, and U. Topcu. "Safe Reinforcement Learning via Shielding." *AAAI Conference on Artificial Intelligence (AAAI)*, 2018.

### Intrinsic Motivation

- **[Burda et al., 2019]** Y. Burda, H. Edwards, A. Storkey, and O. Klimov. "Exploration by Random Network Distillation." *International Conference on Learning Representations (ICLR)*, 2019.

- **[Pathak et al., 2017]** D. Pathak, P. Agrawal, A. A. Efros, and T. Darke. "Curiosity-driven Exploration by Self-Supervised Prediction." *International Conference on Machine Learning (ICML)*, 2017.

### Reward Shaping and Multi-Objective RL

- **[Ng et al., 1999]** A. Y. Ng, D. Harada, and S. Russell. "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." *International Conference on Machine Learning (ICML)*, 1999.

- **[Liu et al., 2025]** R. Liu, J. Shu, W. Zhan, J. T. Kwok, and B. Liu. "Constrained Multi-Objective Reinforcement Learning with Lexicographic Ordering." *International Conference on Learning Representations (ICLR)*, 2025.

### Off-Policy and Replay

- **[Haarnoja et al., 2018]** T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *International Conference on Machine Learning (ICML)*, 2018.

- **[Andrychowicz et al., 2017]** M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, P. Abbeel, and W. Zaremba. "Hindsight Experience Replay." *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

### Action Masking

- **[Huang & Ontañón, 2022]** S. Huang and S. Ontañón. "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms." *Florida Artificial Intelligence Research Society (FLAIRS)*, 2022.

### Hyperparameter Search

- **[Jaderberg et al., 2017]** M. Jaderberg, V. Dalibard, S. Osindero, W. M. Czarnecki, J. Donahue, A. Razavi, O. Vinyals, T. Green, I. Dunning, K. Simonyan, C. Fernando, and K. Kavukcuoglu. "Population Based Training of Neural Networks." *arXiv:1711.09846*, 2017. *(preprint)*
