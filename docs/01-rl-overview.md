# RedWithinBlue RL Overview

A distilled map of everything in [02-rl-training-guide.md](02-rl-training-guide.md) and [03-rl-taxonomy.md](03-rl-taxonomy.md). Start here, follow links for depth.

---

## The Problem in One Paragraph

N agents on a grid. Each sees an 11x11 patch around itself. They communicate over a distance-based graph — if you're within range, you're connected. The team must explore every cell while keeping the communication graph intact. Partial observability + cooperation + communication constraints + discrete actions (5 moves). Formally: a cooperative Dec-POMDP with networked communication.

---

## Decision Tree: Choosing Your Approach

```
START
 │
 ├─ "I want to get something learning"
 │   └─→ Raw policy gradient (Layer 1) on 8x8, 1 agent
 │        Add complexity one piece at a time as you hit walls
 │        See: 02-rl-training-guide.md §10 (Layers 0-7)
 │
 ├─ "Basic training works but I'm hitting specific problems"
 │   ├─ High variance? ──────→ Add baseline (Layer 2), then critic (Layer 3)
 │   ├─ Training unstable? ──→ Add clipping + GAE (Layer 4)
 │   ├─ Too slow? ───────────→ Add parallel envs (Layer 5)
 │   ├─ Need multi-agent? ──→ Apply same policy to N agents (Layer 6)
 │   └─ Need coordination? ─→ Centralized critic (Layer 7)
 │
 ├─ "Layers 0-7 work, but I need more"
 │   ├─ Reward tuning hell? ──→ Constrained RL (Lagrangian)
 │   ├─ Agents ignore graph? ─→ GNN policy / TarMAC
 │   ├─ Poor coordination? ───→ QMIX (value decomposition)
 │   └─ Manual curriculum? ───→ PAIRED / ACCEL
 │        See: 03-rl-taxonomy.md §2, 02-rl-training-guide.md §8
 │
 └─ "I want to scale to 16-32 agents"
     ├─ Fixed architecture? ──→ UPDeT transformer (train N=4, deploy N=16)
     └─ Graph-native? ────────→ GNN policy (permutation invariant, any N)
          See: 03-rl-taxonomy.md §2.4, §2.7
```

---

## The RL Landscape (Filtered for Our Problem)

```
RL Methods
│
├── Model-Free
│   │
│   ├── On-Policy (use data once, discard)
│   │   ├── PPO ★ ─────── Single agent baseline        [Phase 1]
│   │   ├── IPPO ★ ────── Independent PPO, shared params [Phase 2]
│   │   └── MAPPO ★ ───── Centralized critic (CTDE)      [Phase 3]
│   │
│   ├── Off-Policy (replay buffer, reuse data)
│   │   ├── QMIX ★ ────── Value decomposition, discrete actions
│   │   ├── QPLEX ────── More expressive than QMIX
│   │   └── SAC ────────── Continuous actions (NOT a fit)
│   │
│   ├── Communication Learning
│   │   ├── CommNet ───── Mean-pool (we already do this)
│   │   ├── TarMAC ★ ──── Attention-based, learns WHO + WHAT
│   │   └── IC3Net ────── Learns WHEN to communicate
│   │
│   └── Graph Neural Networks
│       ├── DGN ★ ──────── Graph convolution Q-learning
│       └── MAGIC ──────── Graph attention + teaming
│
├── Model-Based
│   ├── DreamerV3 ─────── World model, imagined rollouts (JAX-native)
│   └── MuZero ─────────── Learned model + MCTS planning
│
├── Hierarchical
│   ├── Options ─────────── "Go to region X" + low-level movement
│   └── HAVEN ★ ──────── Hierarchy + QMIX (both levels)
│
├── Evolutionary
│   ├── MAP-Elites ────── Diverse strategy repertoire
│   └── CMA-ES ──────────── Hyperparameter search
│
└── Meta / Curriculum
    ├── PAIRED ★ ────────── Adversarial environment design
    ├── ACCEL ──────────── Evolutionary curriculum
    └── PBT ───────────── Auto-tune reward weights

★ = recommended for this problem
```

---

## Training Paradigms

How information flows during training vs. execution:

```
                        EXECUTION
                   Centralized    Decentralized
                 ┌──────────────┬──────────────┐
  TRAINING       │              │              │
  Centralized    │    CTCE      │    CTDE ★    │
                 │  Joint DQN   │  MAPPO       │
                 │  (N≤8 only)  │  QMIX        │
                 │              │  MADDPG       │
                 ├──────────────┼──────────────┤
  Decentralized  │              │    DTDE      │
                 │   (unused)   │  IPPO        │
                 │              │  IQL         │
                 └──────────────┴──────────────┘

              Networked (orthogonal — layered on top):
              CommNet, TarMAC, IC3Net, DGN, GNN policies

★ = our primary paradigm
```

**Why CTDE:** Our env separates agent-local obs (255-dim) from global state (2073-dim). The centralized critic sees the full explored map + all positions + adjacency matrix during training. At deployment, only the 255-dim decentralized actor runs. This separation is already built into the environment API.

---

## The Three-Phase Curriculum

```
PHASE 1                    PHASE 2                    PHASE 3
Single-Agent PPO           IPPO (warm start)          MAPPO (collaboration)
────────────────           ──────────────────         ──────────────────────
8x8 grid, 1 agent         32x32 grid, 4 agents       32x32 grid, 4 agents
Learn to scan              Scale + clone              Learn to stay connected
wall_density=0.0           wall_density=0.0           wall_density=0.1

Reward:                    Reward:                    Reward:
  explore(1.0)               explore(1.0)               explore(1.0)
  revisit(-0.5)              revisit(-0.5)              revisit(-0.5)
  time(-0.1)                 time(-0.1)                 connectivity(2.0) ← NEW
  coverage(5.0)              coverage(5.0)              time(-0.1)
                                                        coverage(5.0)

Actor:  255 → 256 → 128 → 5     (SAME across all phases)
Critic: 255 → 256 → 128 → 1     255 → ... → 1          2073 → 512 → 256 → 1
        └──── transfers ──────→  └──── transfers ──────→  (re-initialized)

Gate:   95% coverage             80% coverage            90% coverage
        500+ episodes            <10% fragmentation       <5% fragmentation
                                 500+ episodes            <3 steps isolated
```

**Key insight:** The actor network (obs_dim=255) is grid-size-independent. Weights transfer directly between phases. Only the critic changes dimension when grid size or agent count changes.

---

## Five Layers of Safety

```
Layer 5: Curriculum Gate ─── Can't advance phases without proof ──── HARD
Layer 4: Training Monitor ── Detect reward/entropy collapse ──────── SOFT
Layer 3: Full Connectivity ─ Graph can't fragment ────────────────── HARD
Layer 2: Connectivity Shield  Agent can't isolate itself ──────────── HARD
Layer 1: Action Mask ──────── Agent can't walk into walls ─────────── HARD

Recommended starting stack: Layers 1 + 2 + 4 + 5
Add Layer 3 only if Lagrangian constraint isn't enough
```

---

## Escaping Reward Engineering Hell

The compose_rewards approach has 5-6 arbitrary weights. Three paths out:

```
CURRENT STATE                     TARGET STATE
─────────────                     ────────────
compose_rewards(                  reward = exploration (RND)
  exploration    w=1.0               ↑ single coefficient (0.05)
  revisit        w=0.5
  connectivity   w=2.0  ──→      constraints:
  time_penalty   w=0.1              connectivity ≥ 95%  ← interpretable
  coverage_bonus w=5.0              episode_length ≤ 200 ← interpretable
)
5 fragile weights                 1 coefficient + 2 thresholds
```

| Path | What Changes | Effort | Recommended? |
|------|-------------|--------|-------------|
| **Constrained RL** (Lagrangian) | Weights → thresholds, auto-tuned multipliers | 2-3 weeks | Yes, primary |
| **RND** (intrinsic motivation) | Replace exploration+revisit with curiosity | 1 week | Yes, complementary |
| **PBT** (population search) | Keep weights, auto-search values | 2 weeks | If you want quick fix |
| **MORL** (Pareto front) | Eliminate weights, get menu of policies | 3-4 weeks | If trade-offs unknown |

See: [02-rl-training-guide.md §8](02-rl-training-guide.md) for implementation details.

---

## Implementation Roadmap

Build from raw mechanics. Each layer adds ONE concept. Move up when you hit a wall.

```
Layer 0: Policy net + random baseline
         "What does random look like?"
         │
Layer 1: Raw policy gradient ──── loss = -log_prob * return
         "Can the reward signal be learned at all?"
         │
Layer 2: Subtract baseline ────── loss = -log_prob * (return - avg)
         "Was variance the problem?"
         │
Layer 3: Add a critic ─────────── advantage = r + γV(next) - V(current)
         "Does bootstrapping help?"
         │
Layer 4: Stability tricks ─────── GAE, clipping, entropy, grad clip
         "What specific instability am I hitting?"
         │
Layer 5: Parallel rollouts ────── jax.vmap over 32 envs
         "Same algorithm, 32x more data"
         │
Layer 6: Multi-agent ──────────── Same policy applied to N agents
         "Just a vmap over agents"
         │
Layer 7: Centralized critic ───── V(global_state) instead of V(obs)
         "Give the critic more information"
         │
Layer 8: Everything else ─────── Driven by observed failures:
         ├── Reward tuning hell? → Lagrangian constraints
         ├── Exploration stalls? → RND intrinsic motivation
         ├── Poor coordination?  → GNN policy / TarMAC
         ├── Manual curriculum?  → PAIRED / ACCEL
         └── Scale to N=16+?    → UPDeT transformer
```

---

## Method Comparison at a Glance

| Method | Type | Sample Eff. | Coordination | Scalability | JAX Ready? | Our Priority |
|--------|------|------------|--------------|-------------|------------|-------------|
| PPO/MAPPO | On-policy PG | Medium | Good (CTDE) | Good | Build | **Phase 1-3** |
| QMIX | Off-policy VD | High | Good (mixing) | Good | JaxMARL | **Phase 4** |
| GNN (DGN) | Graph policy | Medium | Best (native) | Best | jraph | **Phase 5** |
| TarMAC | Comm learning | Medium | Good (learned) | Good | Build | **Phase 6** |
| PAIRED | Curriculum | N/A (meta) | N/A | Good | Build | **Phase 7** |
| UPDeT | Transformer | Medium | Good (attn) | Best | Build | Phase 8 |
| HAVEN | Hierarchical | Medium | Good (2-level) | Medium | Build | Phase 8 |
| DreamerV3 | Model-based | Best | Unknown (MARL) | Unknown | JAX-native | Phase 8+ |

---

## Quick Reference: What Goes Where

| Question | Answer | Doc |
|----------|--------|-----|
| What's the obs dimension? | 255 (grid-size independent) | [training-guide §1](02-rl-training-guide.md) |
| What's the global state dim? | N\*2 + H\*W + H\*W + N\*N + 1 | [training-guide §1](02-rl-training-guide.md) |
| How does PPO work here? | Clipped surrogate + GAE + vmap rollouts | [training-guide §2.2](02-rl-training-guide.md) |
| How do I transfer weights between phases? | Copy actor (255→5), re-init critic | [training-guide §3-4](02-rl-training-guide.md) |
| What safety layers exist? | Action mask, connectivity shield, monitor, gates | [training-guide §5](02-rl-training-guide.md) |
| What regularization matters? | Entropy (essential), PPO clip + grad clip (important) | [training-guide §6](02-rl-training-guide.md) |
| Should I use experience replay? | No — on-policy methods, use longer rollouts instead | [training-guide §7](02-rl-training-guide.md) |
| How to escape reward weight hell? | Constrained RL + RND | [training-guide §8](02-rl-training-guide.md) |
| What hyperparams for each phase? | Full table with all values | [training-guide §9](02-rl-training-guide.md) |
| What should I build next? | networks.py → ppo.py → mappo.py → scripts | [training-guide §10](02-rl-training-guide.md) |
| What methods exist beyond PG? | Value decomp, GNN, comm learning, hierarchical, etc. | [taxonomy §2](03-rl-taxonomy.md) |
| CTDE vs DTDE vs Networked? | CTDE primary, Networked target | [taxonomy §1](03-rl-taxonomy.md) |
| What's NOT worth trying? | Dec-POMDP solvers, mean-field, MCTS, reward-free | [taxonomy §4](03-rl-taxonomy.md) |
| What's the long-term roadmap? | Phase 4 QMIX → Phase 5 GNN → ... → Phase 8 Dreamer | [taxonomy §5](03-rl-taxonomy.md) |

---

## Document Map

```
docs/
├── 01-rl-overview.md ◄── YOU ARE HERE (summary + visual navigation)
│
├── 02-rl-training-guide.md ── Deep dive: policy gradient training pipeline
│   ├── §1  Environment interface (obs, state, actions, rewards)
│   ├── §2  Algorithms (REINFORCE, PPO, IPPO, MAPPO)
│   ├── §3  Three-phase curriculum with code examples
│   ├── §4  Model lifecycle (save/load/transfer)
│   ├── §5  Training guardrails (5 safety layers)
│   ├── §6  Regularization analysis
│   ├── §7  Experience replay (verdict: don't)
│   ├── §8  Beyond reward engineering (Lagrangian, RND, MORL, PBT)
│   ├── §9  Hyperparameter reference table
│   ├── §10 Implementation roadmap (8 steps)
│   └── §11 References (17 papers, peer-reviewed)
│
├── 03-rl-taxonomy.md ── Broad survey: all RL methods for this problem
│   ├── §1  Training paradigms (CTDE, DTDE, CTCE, Networked)
│   ├── §2  Method families (9 categories, each with verdict)
│   ├── §3  Method-to-paradigm mapping table
│   ├── §4  What NOT to pursue (5 dead ends)
│   ├── §5  Implementation progression (Phase 4-8)
│   └── §6  References (30+ papers)
│
├── 04-environment-api.md ── Environment API reference (all modules + signatures)
├── 05-environment-design-spec.md ── Original env design spec
├── 06-experiment-plan-a.md ── Experiment Plan A: proof of learning
│
└── viz-samples/ ── Visualization style samples + final style PDF
```
