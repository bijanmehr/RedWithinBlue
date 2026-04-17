# RL Taxonomy for Multi-Agent Grid Exploration

A focused map of reinforcement learning methods relevant to the RedWithinBlue problem: N agents on a grid, partial observability, discrete actions, distance-based communication graph, cooperative exploration with connectivity maintenance. This is NOT a general RL survey — every method is evaluated against this specific problem.

For a visual summary of all three docs, see [01-rl-overview.md](01-rl-overview.md). For algorithm implementation details and the training curriculum, see [02-rl-training-guide.md](02-rl-training-guide.md).

---

## The Landscape at a Glance

The general RL taxonomy (adapted from OpenAI Spinning Up [Achiam, 2018]) splits into model-free vs. model-based, then further by what is learned:

```
RL Algorithms
├── Model-Free
│   ├── Policy Optimization (on-policy)
│   │   └── REINFORCE, A2C, TRPO, PPO
│   ├── Value-Based (off-policy)
│   │   └── DQN, C51, Rainbow
│   └── Hybrid (actor-critic, off-policy)
│       └── DDPG, TD3, SAC
│
└── Model-Based
    ├── Learn model, plan with it
    │   └── Dreamer, MuZero, World Models
    └── Learn model, augment model-free
        └── MBVE, I2A
```

That taxonomy is single-agent. Multi-agent RL adds two dimensions: **training paradigm** (who sees what during training vs. execution) and **coordination mechanism** (how agents share information). The taxonomy below is organized around these dimensions because they determine what's practical for our problem.

---

## 1. Training Paradigms

The training paradigm is the single most important architectural decision. It determines what information flows where.

### CTDE — Centralized Training, Decentralized Execution

During training, a centralized component (critic, mixing network) accesses global state — all agent positions, full exploration map, adjacency matrix. During execution, each agent acts on local observations only. The centralized component is discarded.

**Methods:** MAPPO [Yu et al., 2022], MADDPG [Lowe et al., 2017], COMA [Foerster et al., 2018], QMIX [Rashid et al., 2018], QPLEX [Wang et al., 2021]

**Fit for RedWithinBlue:** Excellent. The environment is already structured for this — `get_obs()` returns agent-local views, `get_global_state()` returns the full world state. The centralized critic can see the full `explored` grid, all positions, and the adjacency matrix, while the decentralized actor sees only the 255-dim local observation. This is the recommended primary paradigm.

### DTDE — Decentralized Training, Decentralized Execution

No centralized information at any point. Each agent trains independently from its own observations. Other agents are treated as part of the environment dynamics.

**Methods:** IQL [Tan, 1993], IPPO [de Witt et al., 2020]

**Fit for RedWithinBlue:** Good as a baseline. IPPO with parameter sharing is surprisingly competitive [Yu et al., 2022] and already the Phase 2 algorithm in our curriculum. The key limitation: no mechanism for learning coordination beyond what emerges from shared rewards. Agents cannot learn that fragmentation is bad unless they individually observe it.

### CTCE — Centralized Training, Centralized Execution

A single centralized controller takes all observations and outputs all actions jointly. No communication constraint — the controller sees everything, always.

**Methods:** Joint-action DQN, centralized PPO over joint action space

**Fit for RedWithinBlue:** Only for small-scale upper-bound baselines. Joint action space = 5^N. At N=4 that's 625 (tractable). At N=8 it's 390,625 (borderline). At N=16+ it's intractable. Also contradicts the problem's communication-constrained premise.

### Networked / Communication-Based

Agents share information through a communication graph during both training and execution. Not fully centralized (only neighbors communicate), not fully independent (information propagates through the graph).

**Methods:** CommNet [Sukhbaatar et al., 2016], TarMAC [Das et al., 2019], IC3Net [Singh et al., 2019], DGN [Jiang et al., 2020]

**Fit for RedWithinBlue:** Natural fit. The distance-based adjacency graph IS the communication topology. These methods learn what to communicate over it, rather than just routing raw scans. This is the most promising direction for advancing beyond MAPPO.

### Paradigm Comparison for This Problem

| Paradigm | Scalability | Coordination | Deployment Realism | Our Use |
|----------|-------------|--------------|-------------------|---------|
| CTCE | Poor (N>8 intractable) | Best | Worst (needs oracle) | Upper-bound baseline only |
| CTDE | Good (tested to N~100) | Good | Good (actors only at runtime) | **Primary paradigm** |
| DTDE | Best (fully independent) | Limited | Best (truly decentralized) | Lower-bound baseline |
| Networked | Good (graph-structured) | Good | Good (local comm only) | **Target paradigm** |

---

## 2. Method Families

### 2.1 Policy Gradient (On-Policy)

Collect rollouts with current policy, compute advantages, update policy via gradient ascent on the expected return. Data is used once then discarded.

| Method | How It Works | MARL Variant | Venue |
|--------|-------------|-------------|-------|
| REINFORCE | Monte Carlo returns, no critic | Independent REINFORCE | [Sutton et al., 1999] |
| PPO | Clipped surrogate + GAE advantages | IPPO, MAPPO | [Schulman et al., 2017] *(preprint)* |
| TRPO | Trust region constraint on KL divergence | MATRPO | [Schulman et al., 2015] |
| A2C | Advantage actor-critic, synchronous | IA2C | [Mnih et al., 2016] |

**Verdict for RedWithinBlue:** This is the recommended starting point. PPO → IPPO → MAPPO is the curriculum path described in [02-rl-training-guide.md](02-rl-training-guide.md). Well-understood, stable, and maps cleanly onto CTDE. The main weakness is sample efficiency — on-policy methods discard data after each update.

### 2.2 Value Decomposition (Off-Policy)

Factorize a joint Q-function Q_tot(s, a_1,...,a_N) into per-agent utilities Q_i(o_i, a_i) that can be maximized independently at execution time. The Individual-Global-Max (IGM) condition ensures the joint argmax equals the combination of per-agent argmaxes.

| Method | Decomposition | Expressiveness | Venue |
|--------|--------------|----------------|-------|
| VDN | Q_tot = sum(Q_i) | Low (additive only) | [Sunehag et al., 2018] AAMAS |
| QMIX | Monotonic mixing via hypernetwork | Medium (monotonic) | [Rashid et al., 2018] ICML |
| Weighted QMIX | Importance-weighted monotonic mixing | Medium+ | [Rashid et al., 2020] NeurIPS |
| QPLEX | Duplex dueling advantage decomposition | High | [Wang et al., 2021] ICML |
| QTRAN | Soft regularization of IGM constraint | Highest (theory) | [Son et al., 2019] ICML |

**Verdict for RedWithinBlue:** Strong alternative to policy gradient. Discrete action space (5 actions) is a perfect match — value-based methods excel here. QMIX is already implemented in JaxMARL. However, the monotonicity assumption in QMIX can be a problem: maintaining connectivity sometimes requires "sacrifice" actions (one agent stays so others can explore), which is a non-monotonic interaction. Weighted QMIX or QPLEX handle this better.

**Key concern:** QMIX's exploration is weak by default. The exploration-heavy nature of our problem would likely require MAVEN-style extensions [Mahajan et al., 2019, NeurIPS] or epsilon-decay schedules.

**Recommendation:** Run QMIX as a parallel baseline alongside the PPO curriculum. It's available in JaxMARL out of the box. If QMIX outperforms MAPPO (it might — discrete actions favor value methods), switch the primary path.

### 2.3 Communication Learning

Instead of hand-designing what agents communicate (our current mean-pooled raw scans), learn the communication protocol end-to-end. Agents learn WHAT to say, WHEN to speak, and WHO to address.

| Method | Mechanism | Key Feature | Venue |
|--------|-----------|-------------|-------|
| CommNet | Mean-pool hidden states | Simplest; broadcast to all | [Sukhbaatar et al., 2016] NeurIPS |
| DIAL | Continuous train / discrete execute | First gradient-based comm | [Foerster et al., 2016] NeurIPS |
| IC3Net | Gated communication | Learns WHEN to communicate | [Singh et al., 2019] ICLR |
| TarMAC | Attention over neighbors | Learns WHO to address + WHAT to say | [Das et al., 2019] ICML |
| NDQ | Info-theoretic regularization | Minimizes communication bandwidth | [Wang et al., 2020] ICML |

**Verdict for RedWithinBlue:** TarMAC is the strongest fit. Our adjacency matrix naturally constrains the attention mask — agents only attend to comm-graph neighbors. The attention weights reveal which messages are most valuable, providing interpretability ("agent 2 is listening to agent 0's scan data"). Our `msg_dim=8` learned vector slot in `messages_out` is already designed for this.

**Practical note:** Our current mean-pooled message routing (`route_messages` in `comm_graph.py`) is essentially a simplified CommNet. Upgrading to TarMAC-style attention is an incremental change to the message aggregation step.

### 2.4 Graph Neural Network Policies

Use the communication graph structure directly as the computational backbone of the policy. Each agent is a node; edges are communication links. GNN message passing IS inter-agent communication.

| Method | Architecture | Key Feature | Venue |
|--------|-------------|-------------|-------|
| DGN | Multi-head attention over graph | Q-learning with graph convolution | [Jiang et al., 2020] ICLR |
| MAGIC | Graph Attention Network (GAT) | Dynamic topology learning | [Niu et al., 2021] AAMAS |

**Verdict for RedWithinBlue:** Arguably the most natural fit for our architecture. The `GraphTracker` already stores exactly what GNNs need: adjacency matrices `[T, N, N]`, node features `[T, N, F]` (position, degree, team, uid), and dynamic topology. GNN policies are permutation-invariant (work regardless of agent ordering) and transferable across different N — train on 4 agents, deploy on 8.

**JAX tooling:** DeepMind's `jraph` library provides JAX-native GNN primitives. The adjacency matrix from `build_adjacency()` converts directly to jraph's edge format. Fixed-size padded representation handles JIT shape requirements.

**Recommendation:** This is the top architecture recommendation after establishing MLP baselines. Build a GNN actor-critic where node features = agent observations, edge structure = adjacency from `build_adjacency()`, GNN output = per-agent action logits.

### 2.5 Model-Based RL

Learn a predictive model of environment dynamics, then train policies on imagined rollouts or plan through the model.

| Method | Approach | Key Feature | Venue |
|--------|---------|-------------|-------|
| DreamerV3 | Latent world model + imagine | Single hyperparameter set | [Hafner et al., 2025] Nature |
| MuZero | Learned dynamics + MCTS | No reconstruction loss | [Schrittwieser et al., 2020] Nature |
| World Models | VAE + RNN + controller | Compact latent dynamics | [Ha & Schmidhuber, 2018] NeurIPS |

**Verdict for RedWithinBlue:** High potential but high implementation cost. Grid exploration is inherently a planning problem — a world model that predicts coverage after N steps could enable much more efficient exploration. The grid dynamics are deterministic and simple, so a world model should be easy to learn. However, multi-agent world models are an open research problem. No off-the-shelf multi-agent Dreamer exists.

**Practical path:** Use single-agent DreamerV3 (JAX-native) where each agent maintains its own world model and shares predictions through the communication graph. This is conceptually similar to MAMBA [Egorov & Shpilman, 2022] but using your existing message routing.

**Recommendation:** Phase 4+ exploration. Establish model-free baselines first.

### 2.6 Hierarchical RL

Decompose the problem into levels: a high-level policy selects subgoals ("explore northwest quadrant"), a low-level policy executes primitive actions to achieve them.

| Method | Structure | Key Feature | Venue |
|--------|----------|-------------|-------|
| Options Framework | Option = (init, policy, termination) | Temporal abstraction | [Sutton et al., 1999] JAIR |
| Feudal Networks | Manager + Worker | Latent goal spaces | [Vezhnevets et al., 2017] ICML |
| HAVEN | Hierarchy + value decomposition | Combines HRL with QMIX | [Xu et al., 2023] AAAI |

**Verdict for RedWithinBlue:** High value. Our problem has a natural hierarchical structure:
- **High level:** Which region to explore next (Voronoi partition of unexplored cells)
- **Low level:** Navigate there while staying connected

A high-level decision every 16-32 steps reduces the effective horizon from 256 to 8-16 decisions. HAVEN's integration with value decomposition means you can combine hierarchy with QMIX.

**Complication:** The connectivity constraint makes high-level planning hard — a goal of "go to cell (25,3)" might be infeasible without disconnecting the graph. The high-level policy needs to be connectivity-aware.

### 2.7 Transformer / Attention Policies

Use self-attention to handle variable and relational structure. Agents attend to entities (other agents, objects, messages) with learned attention weights.

| Method | Architecture | Key Feature | Venue |
|--------|-------------|-------------|-------|
| UPDeT | Observation-entity transformer | Transfer across different N | [Hu et al., 2021] ICLR (Spotlight) |
| MAT | Multi-Agent Transformer | Sequential agent-by-agent generation | [Wen et al., 2022] NeurIPS |

**Verdict for RedWithinBlue:** Strong upgrade path from MLP policies. UPDeT treats observations as entities processed by a transformer, decoupling the policy from fixed input/output dimensions. This enables transfer across different agent counts — train at N=4, deploy at N=16. Reported 100x faster than training from scratch on transfer tasks [Hu et al., 2021].

**Recommendation:** Implement after MLP baselines work. UPDeT + GraphTracker data is a natural combination.

### 2.8 Evolutionary / Population Methods

Evaluate a population of candidate policies by running them in the environment, then evolve toward higher fitness. Gradient-free.

| Method | Mechanism | Key Feature | Venue |
|--------|-----------|-------------|-------|
| OpenAI ES | Gaussian perturbation + fitness | Embarrassingly parallel | [Salimans et al., 2017] *(preprint)* |
| CMA-ES | Covariance matrix adaptation | Best for <10K params | [Hansen, 2003] Evol. Comput. |
| MAP-Elites | Quality-diversity archive | Produces diverse repertoire | [Mouret & Clune, 2015] Evol. Comput. |

**Verdict for RedWithinBlue:** Best as a complement, not the primary method. Two practical uses:
1. **Hyperparameter search:** OpenAI ES (via `evosax` JAX library) to optimize reward weights or curriculum parameters
2. **Behavioral diversity:** MAP-Elites to discover diverse exploration strategies indexed by (coverage_rate, connectivity_rate), then distill the best into an RL policy

Gradient-free methods struggle with the 256-step horizon and cannot exploit the graph structure the way GNN or value decomposition can.

### 2.9 Curriculum Learning / Environment Design

Automatically generate training scenarios of increasing difficulty rather than training on a fixed task.

| Method | Mechanism | Key Feature | Venue |
|--------|-----------|-------------|-------|
| PAIRED | Adversary designs envs, maximize regret | Prevents impossible tasks | [Dennis et al., 2020] NeurIPS |
| MAESTRO | Multi-agent PAIRED | Extends to cooperative MARL | [Samvelyan et al., 2023] ICLR |
| ACCEL | Evolutionary env parameters | Autocurriculum without adversary | [Parker-Holder et al., 2022] ICML |

**Verdict for RedWithinBlue:** Highly relevant. Our environment is parameterized (grid_width, grid_height, wall_density, num_agents, comm_radius) — a natural fit for environment design methods. PAIRED/ACCEL could vary wall_density and obstacle placement to create increasingly challenging maps, potentially replacing or augmenting our manual Phase 1→2→3 curriculum.

**Recommendation:** High priority for Phase 3+. Can automate the curriculum progression over grid complexity (open field → sparse walls → mazes) and communication constraints (large radius → small radius).

---

## 3. Method-to-Paradigm Map

Every method operates within a training paradigm. This table shows which paradigm each method uses:

| Method Family | Paradigm | Training Sees | Execution Sees |
|---------------|----------|---------------|----------------|
| MAPPO | CTDE | Global state for V(s) | Per-agent obs for pi(a\|o) |
| QMIX / VDN / QPLEX | CTDE | Global state for mixing | Per-agent Q_i(o_i, a_i) |
| IPPO / IQL | DTDE | Per-agent obs only | Per-agent obs only |
| CommNet / TarMAC / IC3Net | Networked | Neighbor messages | Neighbor messages |
| DGN / GNN policies | Networked + CTDE | Graph structure + global | Graph structure + local |
| DreamerV3 / MuZero | Any (per-agent models) | Depends on setup | Per-agent model |
| HAVEN (hierarchical) | CTDE | Global state at both levels | Per-agent at both levels |
| UPDeT (transformer) | CTDE | Entity features + global | Entity features + local |
| PAIRED / ACCEL | Meta (wraps any) | Depends on inner algo | Depends on inner algo |
| MAP-Elites / ES | Meta (wraps any) | Fitness evaluation only | Learned policy |

---

## 4. What's NOT Worth Pursuing (and Why)

**Dec-POMDP Solvers:** The formal model underlying our problem (Dec-POMDP) is NEXP-complete [Bernstein et al., 2002]. Classical solvers (MAA*, JESP) max out at ~3 agents with tiny state spaces. Our 32x32 grid with 4+ agents is many orders of magnitude beyond their capability. Use the theory for formal reasoning, not for computation.

**Mean-Field MARL** [Yang et al., 2018, ICML]: Approximates all other agents as a single aggregate statistic. Useful at N>100 where individual interactions are statistically interchangeable. At N=4-32 with structured spatial positions and a dynamic communication graph, the mean-field approximation discards exactly the information that matters — which specific neighbor is where. Not recommended for our scale.

**MCTS / Planning Methods:** Monte Carlo Tree Search is inherently sequential tree-search, hard to vectorize/JIT in JAX. For 4 agents, joint branching factor is 5^4=625 per step — feasible but expensive. For N>8, intractable. Dec-MCTS [Skrynnik et al., 2024, AAAI] exists for multi-agent pathfinding but requires a C++-style implementation.

**Reward-Free Exploration** [Jin et al., 2020, ICML]: Two-phase approach where Phase 1 explores without any reward, then Phase 2 plans with a learned model. Conceptually appealing, but for discrete grids our existing `exploration_reward` already provides a natural curiosity-like signal. APT and BYOL-Explore are designed for high-dimensional continuous observations (images), not structured grid data.

**Offline RL / Decision Transformers:** Useful after you have online-trained demonstrations, not as the starting method for a novel task. MADT [Meng et al., 2023] is interesting for transfer learning later — train on small grids offline, fine-tune on larger grids.

---

## 5. Recommended Implementation Progression

Based on the analysis above, ordered by priority and building on each previous step:

```
Phase 1-3: Policy Gradient Path (current plan)
  PPO → IPPO → MAPPO (MLP policies, CTDE)
  See 02-rl-training-guide.md for details

Phase 4: Value Decomposition Baseline
  QMIX (from JaxMARL, minimal new code)
  Compare against MAPPO — value methods may win on discrete actions
  If QMIX wins → try Weighted QMIX or QPLEX

Phase 5: GNN Policy Architecture
  Replace MLP actor with GNN actor (using jraph)
  Node features = agent obs, edges = adjacency from build_adjacency()
  GNN output = per-agent logits
  Combine with MAPPO (GNN actor, centralized critic) or QMIX (GNN Q_i)

Phase 6: Learned Communication
  TarMAC-style attention over comm-graph neighbors
  Replace mean-pool aggregation in route_messages with learned attention
  Agents learn what to communicate, not just forward raw scans

Phase 7: Automatic Curriculum
  PAIRED or ACCEL for environment design
  Vary wall_density, obstacle placement, comm_radius
  Potentially replaces manual Phase 1→2→3 curriculum

Phase 8: Architecture Upgrades
  UPDeT transformer policy for N-transfer (train N=4, deploy N=16)
  Hierarchical options (high-level region assignment, low-level navigation)
  World model per agent (DreamerV3-style) for exploration planning
```

### Priority Matrix

| Method | Impact | Effort | Existing JAX Support | Priority |
|--------|--------|--------|---------------------|----------|
| QMIX baseline | High | Low | JaxMARL has it | **Do next** |
| GNN policy (jraph) | High | Medium | jraph library | **High** |
| TarMAC communication | High | Medium | Build on existing comm_graph | **High** |
| PAIRED/ACCEL curriculum | Medium-High | Medium | No off-the-shelf | **Medium** |
| UPDeT transformer | Medium | Medium | Flax implementation | Medium |
| Hierarchical options | Medium | High | No off-the-shelf | Medium |
| DreamerV3 per-agent | Medium | High | JAX-native (single-agent) | Low (Phase 8+) |
| MAP-Elites diversity | Low-Medium | Low | evosax library | Low |

---

## 6. References

### Training Paradigms
- **[Yu et al., 2022]** C. Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." *NeurIPS*, 2022.
- **[Bernstein et al., 2002]** D. S. Bernstein, R. Givan, N. Immerman, and S. Zilberstein. "The Complexity of Decentralized Control of Markov Decision Processes." *Mathematics of Operations Research*, 27(4), 2002.

### Policy Gradient
- **[Sutton et al., 1999]** R. S. Sutton et al. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *NeurIPS*, 1999.
- **[Schulman et al., 2016]** J. Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR*, 2016.
- **[Schulman et al., 2017]** J. Schulman et al. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*, 2017. *(preprint)*
- **[Schulman et al., 2015]** J. Schulman et al. "Trust Region Policy Optimization." *ICML*, 2015.

### Value Decomposition
- **[Rashid et al., 2018]** T. Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." *ICML*, 2018.
- **[Rashid et al., 2020]** T. Rashid et al. "Weighted QMIX: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." *NeurIPS*, 2020.
- **[Sunehag et al., 2018]** P. Sunehag et al. "Value-Decomposition Networks for Cooperative Multi-Agent Learning Based on Team Reward." *AAMAS*, 2018.
- **[Wang et al., 2021]** J. Wang et al. "QPLEX: Duplex Dueling Multi-Agent Q-Learning." *ICML*, 2021.
- **[Son et al., 2019]** K. Son et al. "QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning." *ICML*, 2019.
- **[Mahajan et al., 2019]** A. Mahajan et al. "MAVEN: Multi-Agent Variational Exploration." *NeurIPS*, 2019.

### Communication Learning
- **[Sukhbaatar et al., 2016]** S. Sukhbaatar, A. Szlam, and R. Fergus. "Learning Multiagent Communication with Backpropagation." *NeurIPS*, 2016.
- **[Foerster et al., 2016]** J. Foerster et al. "Learning to Communicate with Deep Multi-Agent Reinforcement Learning." *NeurIPS*, 2016.
- **[Singh et al., 2019]** A. Singh, T. Jain, and S. Sukhbaatar. "Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks." *ICLR*, 2019.
- **[Das et al., 2019]** A. Das et al. "TarMAC: Targeted Multi-Agent Communication." *ICML*, 2019.
- **[Wang et al., 2020]** T. Wang et al. "Learning Nearly Decomposable Value Functions Via Communication Minimization." *ICML*, 2020.

### GNN Policies
- **[Jiang et al., 2020]** J. Jiang et al. "Graph Convolutional Reinforcement Learning." *ICLR*, 2020.
- **[Niu et al., 2021]** Y. Niu et al. "Multi-Agent Graph-Attention Communication and Teaming." *AAMAS*, 2021.

### Model-Based
- **[Hafner et al., 2025]** D. Hafner et al. "Mastering Diverse Domains through World Models." *Nature*, 2025.
- **[Schrittwieser et al., 2020]** J. Schrittwieser et al. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." *Nature*, 2020.

### Hierarchical RL
- **[Sutton et al., 1999b]** R. S. Sutton, D. Precup, and S. Singh. "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning." *Artificial Intelligence*, 112(1-2), 1999.
- **[Vezhnevets et al., 2017]** A. S. Vezhnevets et al. "FeUdal Networks for Hierarchical Reinforcement Learning." *ICML*, 2017.
- **[Xu et al., 2023]** Z. Xu et al. "Hierarchical Cooperative Multi-Agent Reinforcement Learning with Dual Coordination Mechanism." *AAAI*, 2023.

### Transformer / Attention Policies
- **[Hu et al., 2021]** S. Hu et al. "UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers." *ICLR*, 2021.
- **[Wen et al., 2022]** M. Wen et al. "Multi-Agent Reinforcement Learning is a Sequence Modeling Problem." *NeurIPS*, 2022.

### Evolutionary Methods
- **[Hansen, 2003]** N. Hansen and A. Ostermeier. "Completely Derandomized Self-Adaptation in Evolution Strategies." *Evolutionary Computation*, 9(2), 2001.
- **[Mouret & Clune, 2015]** J.-B. Mouret and J. Clune. "Illuminating Search Spaces by Mapping Elites." *arXiv:1504.04909*, 2015.

### Curriculum / Environment Design
- **[Dennis et al., 2020]** M. Dennis et al. "Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design." *NeurIPS*, 2020.
- **[Samvelyan et al., 2023]** M. Samvelyan et al. "MAESTRO: Open-Ended Environment Design for Multi-Agent Reinforcement Learning." *ICLR*, 2023.
- **[Parker-Holder et al., 2022]** J. Parker-Holder et al. "Evolving Curricula with Regret-Based Environment Design." *ICML*, 2022.

### Mean-Field (for reference)
- **[Yang et al., 2018]** Y. Yang et al. "Mean Field Multi-Agent Reinforcement Learning." *ICML*, 2018.

### Reward-Free Exploration (for reference)
- **[Jin et al., 2020]** C. Jin et al. "Reward-Free Exploration for Reinforcement Learning." *ICML*, 2020.

### JaxMARL
- **[Rutherford et al., 2024]** A. Rutherford et al. "JaxMARL: Multi-Agent RL Environments and Algorithms in JAX." *AAMAS*, 2024.
