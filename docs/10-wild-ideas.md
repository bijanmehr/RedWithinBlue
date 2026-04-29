# Wild Ideas

A living scratchpad of speculative directions for RedWithinBlue. **Nothing here is a plan.** Each entry is a research direction we've kicked around — what it would change, what we'd actually learn, the cheapest first experiment, and the known risks. When an idea graduates to a real plan, it gets a `docs/NN-<name>.md` of its own.

Maintained as a flat list. New ideas go at the bottom. When something is killed or shipped, mark it and leave the entry — the graveyard is informative.

---

## 1. Hex grid instead of square grid

**Status:** unstarted, smoke-experiment scoped.

**The idea.** Replace the square 4-connected grid with a hex 6-connected grid (axial coords). Action space grows from 5 (STAY + 4 cardinals) to 7 (STAY + 6 hex directions). Local scan k-ring of radius `r` has `1 + 3r(r+1)` cells (r=1 → 7, r=2 → 19) instead of `(2r+1)²`.

**Why it might matter.**
- **Isotropy.** Hex gives uniform-distance neighbours; square has the L1/L2 mismatch (axial neighbours dist 1, diagonals dist √2 but illegal moves). Coverage and "spread" metrics are anisotropic on square by construction.
- **Cleaner story for swarm coverage.** Hex centres tile the plane uniformly — closer to "drones over an area" than a square grid is.
- **Different entropy regime.** 7-action policies under entropy regularisation behave differently from 5-action; `H_max = ln 7 ≈ 1.95` vs `ln 5 ≈ 1.61`. Joint-red entropy collapse signature (`project_red_entropy_collapse.md`) needs re-derivation — and the answer might be different.

**What changes mechanically.**
- `types.py:32` `ACTION_DELTAS_ARRAY` → 7 axial deltas: `{(+1,0), (−1,0), (0,+1), (0,−1), (+1,−1), (−1,+1)}` plus STAY.
- `grid.py:66` `get_local_scan` — square `dynamic_slice` becomes hex k-ring extraction (gather via index table or rhombus slice + mask).
- `movement.py:67` — bounds-clipping changes shape (parallelogram in axial-with-offset).
- `agents.py:115` — `comm_radius` either stays Euclidean over Cartesian-projected hex centres, or switches to integer hex distance.
- `visualizer.py` — full rewrite of cell rendering (matplotlib `PolyCollection` of hexes).
- **Obs dim shifts.** `view_radius=1, local_obs=true`: 9+9+1+2+1+1 = 23 → 7+7+1+2+1+1 = 19. Every existing checkpoint becomes shape-incompatible. All XAI block-attribution results need redo.

**Cheapest first experiment.** Add a `geometry: square|hex` flag to `EnvConfig` (default square), implement hex as opt-in. Run **one** smoke: hex-`pair-cooperate-coop` end-to-end with the current MC critic. ~25 min. Compare final coverage / `|loss|` trajectory / k=1 joint-red entropy collapse to square baseline. If qualitatively the same → geometric reskin; if not → real finding.

**Risks.** Sunk cost — switching geometries before the critic-drift Stage 1/2/3 ladder lands on square loses the comparison point. Tooling cost (visualizer rewrite). JAX-friendliness slightly worse — k-ring extraction is uglier than `dynamic_slice`.

---

## 2. General-sum red (drop the policy-negation)

**Status:** literature surveyed, minimal-change identified, unstarted.

**The idea.** Today red's reward is literally `per_red = -blue_sum / n_red` (`rewards_training.py:265`). That's pure two-team zero-sum; red has no intrinsic objective. Replace with a general-sum formulation where red has its own utility (own-coverage, stealth, action cost) plus a damped zero-sum coupling.

**Why it might matter.**
- **Cost-aware adversaries.** The unconstrained-budget pathology — red converges to "the strategy that hurts blue most regardless of cost" — disappears once red has its own cost term. The empirical signatures we've already documented (`project_uncertainty_manipulation_thin.md`, `project_sabotage_as_delay.md`, `project_red_entropy_collapse.md`) read as symptoms of this pathology.
- **The equilibrium concept stops being trivial.** In zero-sum, max-min = min-max. In general-sum you can ask: what does blue lose at the **Stackelberg equilibrium** (blue commits, red best-responds — the deployed-in-production framing for security games) versus a Nash, versus a quantal-response equilibrium against a bounded-rationality red?
- **`ΔJ(red)` becomes informative again.** Right now `ΔJ(red) ≡ −ΔJ(blue)` by definition. In general-sum you can decompose blue's loss into damage-attributable-to-red versus damage-from-red-pursuing-its-own-goal.

**The minimal change.** One line in `rewards_training.py:265`:
```python
per_red = α * red_intrinsic + β * (-blue_sum / n_red) - λ * action_cost
```
With `red_intrinsic = own_coverage` implemented as **potential-based shaping** (Φ(s′) − Φ(s)), Devlin & Kudenko (2011) prove the Nash equilibria of the stochastic game are *unchanged*. So previous `k*(θ)` and `ΔJ(ε)` results still mean what they used to.

**Algorithm choice.** **M-FOS** (Lu et al. 2022, arXiv:2205.01447) is the right opponent-shaping method for our stack — it's model-free and doesn't need differentiable opponent params, so it slots into `coevo.py` without a rewrite. **LOLA / SOS / COLA are out** (require differentiable opponent updates). **PSRO** (Lanctot et al. 2017) is a strict drop-in upgrade for `coevo.py` and gives principled exploitability measurement.

**Cheapest first experiment.** α small, `red_intrinsic = own_coverage`, λ = 0, on the C2 setup (`compromise-16x16-5-3b2r-coevo`). Predict: the k=2 entropy collapse softens — entropy stays ≥ 0.7 nats because red has multi-modal reward. Either result is publishable.

**Risks.** Reward hacking by red (α too high → red abandons attacking, just self-rewards). Cyclic / chaotic coevolution under non-zero-sum (Ficici & Pollack 2007). Equilibrium drift if `red_intrinsic` is *not* potential-based.

**Papers to cite.** Leibo et al. 2017 (sequential social dilemmas), Lu et al. 2022 (M-FOS), Lanctot et al. 2017 (PSRO), Devlin & Kudenko 2011 (PBRS in stochastic games), Sinha/Tambe 2018 (Stackelberg security games), Černý et al. 2021 (QRE in security games), Gleave et al. 2020 (adversarial policies), Mohammadi et al. 2024 (cost-constrained MARL attacks).

---

## 3. Red's prior — Bayesian warm-start, defector, alien

**Status:** unstarted, two-phase experiment scoped.

**The idea.** Treat red's initialisation as a **prior over policy space** `P(π_red)`. Training data → posterior. Two orthogonal axes:

- **Prior axis** (how informative is red's init, holding policy class fixed).
- **Alien axis** (how different is red's policy class from blue's — different actions, obs, architecture).

### Prior axis — three arms (same policy class)

- **I — Insider (defector).** Take a converged blue agent, relabel `team_id = 1`, flip reward sign. `P(π_red(t=0)) = δ(π_blue*)`. Red literally *is* blue at t=0 — full insider knowledge of map representations, comm-graph use, scan parsing. The "blue who turned red" experiment.
- **W — Architectural warm-start.** Blue's `Dense_0` (the bottom MLP layer that maps obs → hidden) tiled into red's input layer; remaining layers (`Dense_1`, `Dense_2`) re-init'd. The actor is a 3-layer MLP, not encoder-decoder — Warm just transfers the input projection. Prior is informative about *how to read the world* but not about *what to do*. (This is the original A2 arm.)
- **F — Fresh.** Standard random init. Uninformative prior.

For joint-red there's an architectural mismatch: blue actor is `obs[D] → logits[5]` per agent (decentralised); joint-red is `obs[n_red·D] → logits[n_red·5]` (concat). For the **I** arm we can avoid the mismatch entirely by going decentralised on red — each red agent uses a copy of blue's actor with sign-flipped reward.

### Bayesian quantities to measure

Per arm × seeds:
1. **Posterior concentration rate** — env-steps until per-step policy entropy stabilises. Sharper prior → faster concentration. Predicted ordering: `t_stable(I) ≪ t_stable(W) < t_stable(F)`.
2. **Prior–posterior KL** — `KL(π_red(t=0) ‖ π_red(t=∞))`. **This is the headline number.** Large KL on the Insider arm = the optimal adversary lives *far* from the cooperative policy in π-space (genuinely different policy classes). Small KL = adversarial and cooperative are near-neighbours, the difference is just a few logit signs.
3. **Effective-sample-size gain** — at matched final adversarial impact `ΔJ_blue`, what's `N_fresh / N_warm`? Direct answer to "how much wall-clock does the prior save in `coevo.py`?".
4. **Cosine similarity of red `Dense_0` to blue Dense_0**, every 1k eps. Tracks how fast red "unlearns" the blue representation under each prior.

### Alien axis — three increasingly-alien red variants

Different policy class, not just different prior.

- **α — Same env, different action.** Add a 6th red-only action `JAM_BLUE` that fogs blue agents in comm range. ~30 LoC: extend `ACTION_DELTAS_ARRAY`-equivalent for red, handle JAM in `env.step`. Blue training untouched. The Insider arm becomes interestingly ill-defined: blue's 5-way head can't directly init red's 6-way head — the choice of how to fill the JAM logit (zero / small-positive / small-negative) *is* a prior on whether JAM is useful.
- **β — Different observation.** Red gets blue's positions directly (omniscient adversary). Or red has a *smaller* view (weakened insider). ~50 LoC in `agents.py`.
- **γ — Different architecture.** Attention pooling over agent obs instead of MLP, recurrent state instead of feedforward. Bigger refactor.

### What each outcome means (across both axes)

- **Insider faster, same plateau as Fresh** → blue's representation transfers, cooperative-and-adversarial policies live near each other, the warm-start is purely a wall-clock win.
- **Insider lands at a *worse* plateau than Fresh** → blue's policy is in a basin of attraction the optimiser can't escape; the prior is *misspecified* for adversarial play. Higher initial ent_coef on red likely fixes it.
- **Insider lands at a *different* plateau** (different ΔJ_blue mechanism, different action histogram) → cooperative and adversarial policies live in different regions of π-space; the prior just biases which region you find first.
- **No difference across arms** → red's learning is bottlenecked by something else (reward sparsity, joint-action coordination, the entropy collapse from `project_red_entropy_collapse.md`). Next experiment is in reward shape (§2), not init.

### Cheapest first runs

**Phase 1 (prior axis).** Insider vs Warm-start vs Fresh, on C2 (`compromise-16x16-5-3b2r-coevo`). 3 seeds each, blue frozen at C2 checkpoint, red trained for 3000 eps, no coevo outer loop. Plot the four Bayesian quantities. **~30 min wall-clock.**

**Phase 2 (alien α).** Add `JAM_BLUE`, retrain red from each of the three priors. Compare `ΔJ_blue` at matched env-steps. **~1 hr.**

**Phase 3 (alien β / γ).** Only if Phase 1+2 produces something interesting. Otherwise alien-axis is a red herring and we go elsewhere.

### Risks

- **Output-head re-init dominates the warm-start.** If `Dense_last` carries most of the policy mass at C2 init, the input-side warm-start is washed out and W collapses onto F.
- **Entropy collapse interaction.** Warm-started red starts deterministic (inherited from blue's converged policy). May need higher initial `ent_coef` on the red optimiser to allow the unlearning step.
- **The alien-α JAM action is a confound.** If red wins more under α, is it because of the new capability or the warm-start? Need a fresh-init α arm too — sample budget grows from 3 arms to 6.

---

## 4. Terrain types + energy economy

**Status:** new, literature scanned, smoke-experiment scoped.

**The idea (verbatim from user).** "Make the world with different terrain so we add an energy level to each agent and then they have to spend them wisely — every operation in each terrain costs differently."

**What this looks like concretely.**
- Each cell has a type: `{grass, forest, water, mountain, road, mud, ...}`. Currently we already have `{empty, wall, obstacle}` — this generalises that.
- Each agent has an **energy pool** `E ∈ [0, E_max]`, depleted per action. Cost depends on the destination terrain: `e.g. road=0.5, grass=1.0, forest=2.0, mountain=4.0, water=∞ (impassable)`.
- Episode terminates for an agent when `E ≤ 0` (or it becomes immobile / loses ability to act).
- **Optional layers**: terrain affects scan radius (forest = reduced view), terrain affects comm_radius (mountains block messages), regenerative cells (food/charging stations).

**Why it might matter.**
- **Coverage strategy becomes routing.** Today the optimal cooperative policy is "fan out and minimise overlap". With heterogeneous costs the optimal becomes "find the cheap-energy spine, branch off it, spend energy where it pays". This is qualitatively a different policy class — we'd predict the cohesion ↔ coverage dial (`project_cohesion_coverage_dial.md`) gets a new third axis: **energy-budget axis**.
- **Asymmetry productively breaks zero-sum.** If blue and red have different terrain costs (e.g. red can pass through obstacles cheaply but pays double for grass — like an adversary moving through ducts), the zero-sum reduction `per_red = -blue_sum/n_red` is no longer the right symmetry. This is a natural bridge to the general-sum reformulation in §2.
- **Realism.** Drone-swarm coverage with battery constraint, search-and-rescue with limited fuel, security patrols — all of these have a real energy budget. Today our env doesn't.

**Mechanical changes.**
- `types.py`: extend `CELL_*` constants to ~6-8 terrain types; add `energy: float` field to `AgentState`.
- `grid.py`: `create_grid` generates a terrain mosaic (Perlin noise? simple Voronoi? hand-painted seed maps?). Add a `terrain_cost_table: jnp.ndarray[num_types]`.
- `movement.py`: after collision resolution, deduct `terrain_cost_table[destination_terrain]` from `agent.energy`. Force STAY when `energy <= 0`.
- `env.py`: episode termination condition extended (`done = step_limit OR all_agents_immobile`).
- Observation: add `energy / E_max` (1 dim) to the obs tail. Possibly add a local terrain-cost view alongside `local_scan`.
- Reward: add a small `-energy_spent` shaping term? Or let the energy constraint speak through the terminal? **TBD by experiment.**

**Cheapest first experiment.** Binary terrain only: `cheap=cost 1, expensive=cost 2`. Random 50/50 split per episode. `E_max = 100`, `max_steps = 100` so the budget is just-tight. Single seed smoke first to see whether the policy actually routes through cheap cells (visualise: heatmap of agent visits coloured by terrain). If yes → expand to 4-type terrain. If no → suspect the terminal-bonus is washing out the energy pressure.

**Risks (literature-flagged).**
- **Dawdle trap / STAY collapse.** Documented in NetHack (Küttler 2020) and Crafter (Hafner 2021): agents learn STAY = optimal energy policy when the per-step pressure is too weak. Fix: keep the existing per-step penalty *and* the fog-of-war potential bonus (`project_fog_of_war_reward`). Watch for the `feedback_ent_coef_in_loss` bug pattern — entropy term must be subtracted into the loss, not returned as `has_aux`.
- **Premature termination starves the critic.** Constrained-Policy-Optimization (Achiam 2017) motivates avoiding hard termination: terminating at `energy=0` shrinks the effective horizon and the team-conditioned critic gets even less signal. Use a *soft* penalty before going to a hard energy-out termination.
- **Asymmetric-terrain Leibo collapse.** Sequential-Social-Dilemmas (Leibo 2017): with low resource pressure, both teams sit in their cheap zones and *no contest emerges*. Need to tune `(max_steps × mean_cost) / E_max` so neither team can self-sustain in their cheap zone alone.
- **Curriculum.** Probably start with `E_max = max_steps × max_cost` (budget-trivially-satisfiable) and anneal down. Otherwise random init never sees a successful episode.

**Literature backup.**
- **Achiam et al. 2017 — Constrained Policy Optimization** ([arXiv:1705.10528](https://arxiv.org/abs/1705.10528)). Canonical CMDP framing; you don't need CPO itself, but the constraint-vs-shaping distinction should drive the reward design.
- **Hafner 2021 — Crafter** ([arXiv:2109.06780](https://arxiv.org/abs/2109.06780)) and **Matthews et al. 2024 — Craftax** ([arXiv:2402.16801](https://arxiv.org/abs/2402.16801)). Crafter is the canonical "energy/hunger/thirst on a procedural map"; Craftax is the JAX-native 250×-faster reimpl. Steal the **achievement-style scoring** — measure terrain-aware policy emergence by *what cells the agent enters*, not just coverage %.
- **Küttler et al. 2020 — NetHack Learning Environment** ([arXiv:2006.13760](https://arxiv.org/abs/2006.13760)). Source of the "dawdle trap" finding.
- **Energy-Aware MARL for Mission-Oriented Drone Networks 2024** ([arXiv:2410.22578](https://arxiv.org/abs/2410.22578)). Reward shape `progress + λ · min(battery in swarm)` — drop-in for our cooperative-energy term.
- **Bamford et al. 2020 — Griddly** ([arXiv:2011.06363](https://arxiv.org/abs/2011.06363)). Not JAX, but the **GDY YAML schema** is the right template if we want terrain types to be data-driven (cells × types × per-team costs) instead of hard-coded.
- **JaxMARL `overcooked_v2`** ([repo](https://github.com/FLAIROx/JaxMARL)). Already has tile-typed JAX-jitted observation infrastructure — reusable for terrain types without writing the indexing from scratch.

**Open questions.** Should energy be observable to the *team* (centralised) or only to the agent (decentralised)? Should agents be able to *transfer* energy via comm-graph edges? Does terrain affect the **survey radius**, the **scan radius**, or both? Should `red_blocks_blue` extend to per-terrain-type passability (red can pass through "ducts" that blue can't)?

---

## 5. AoE-style swarm-vs-swarm

**Status:** new, literature scanned, the punchline is "don't build an RTS".

**The idea (verbatim from user).** "Something like Age of Empires games — swarm A vs Swarm B, team A vs team B."

**What this looks like concretely.** Bigger populations (N ≫ 8 — think 50-200 agents per team), multiple unit types (scouts, fighters, builders, gatherers), resource economy on the map (food / wood / stone / gold cells that deplete), build-orders or tech-tree, base-building (immobile structures that produce units / contribute to vision / get destroyed), **two large teams** instead of small-N asymmetric red/blue.

**Why it might matter.**
- **The strategy class changes qualitatively.** Today's policies are short-horizon tactical (10-100 step coverage). AoE-style is multi-scale: tactical micro (per-unit movement), operational meso (group composition, where to build), strategic macro (econ vs military, tech timing). The literature on this gap — **AlphaStar**, **Lux AI**, **MicroRTS** — is rich and mostly non-JAX.
- **Population scale forces architectural rethink.** Joint-red doesn't scale beyond ~10 agents (concat input dim explodes). At N=100 you need either **mean-field MARL** (Yang et al. 2018) or **attention-over-agents** (MAVEN, COMA). Our current CTDE central critic also breaks at this scale.
- **Genuinely general-sum from day one.** Two large teams with their own economies is naturally general-sum — Pareto improvements exist (mutual non-aggression while building econ). Nash ≠ Stackelberg ≠ QRE here. This is the right benchmark for the §2 reformulation.

**The honest challenge.** AlphaStar reportedly used ≈384 TPUs for 14 days for the v1 result. Lux AI competition entries top out at *much* lower compute but still require careful engineering. **This is not a 25-minute smoke experiment.** If we go this direction it's a months-long commitment with a real chance of producing something that's just a worse SMAC.

**Two scoped routes.**

(a) **Light AoE on our existing engine.** Stay on the JAX grid env. Add: 2 unit types (worker / fighter), 1 resource (food), 1 building (base — gives vision, produces units), simple combat (collision = damage). Two teams of N=20 each on a 32×32. **No tech tree, no build orders.** This is "MARL with economy" not "RTS clone". Wall-clock: ~weeks to engineer, then training similar to today.

(b) **Adopt an existing benchmark.** Port our analysis stack onto **MicroRTS** (Ontañón) or extend **JaxMARL**'s `Hanabi`/`SMAX` to AoE-shaped tasks. Cheaper engineering (we don't rebuild the env), but our hard-won XAI / coevo / probe infrastructure has to be re-targeted.

**Cheapest first experiment.** Before any engineering: load **MicroRTS** or **JaxMARL** in a notebook, train MAPPO baseline for 1 day, measure final win-rate against a scripted opponent. If the baseline takes >10 hours to break 30%, that's the wall-clock floor — extrapolate from there.

**Risks (literature-flagged).**
- **Compute wall.** AlphaStar trained for ~14 days on hundreds of TPUs. Even Lux AI competition entries run on much smaller budgets but require careful engineering. The relevant lesson from AlphaStar is *not* "do AlphaStar"; it's that **PFSP (prioritized fictitious self-play, Vinyals 2019)** is the population mechanism that worked, and PFSP is implementable on top of our existing red/blue setup with no engine port.
- **Credit-assignment cliff at N>20.** Documented in QMIX → SMAC v2, in the open-multi-agent-systems survey ([arXiv:2510.27659](https://arxiv.org/abs/2510.27659)), and *implicit* in our own k=2 entropy collapse (`project_red_entropy_collapse.md`). Plan ablations at N ∈ {4, 8, 16, 32} so we *see* the cliff rather than be surprised.
- **Sample inefficiency.** SMAC battles are 100s of steps; AoE games are 10s of thousands. Replay buffer / curriculum / hierarchical RL becomes mandatory.
- **Non-stationarity from co-evolution.** Predator-prey RL ([arXiv:2002.03267](https://arxiv.org/abs/2002.03267), MDPI Entropy 2021) explicitly observes Lotka-Volterra-style cycles between teams that *never converge*. Already relevant to our `project_coevo_r6.md` result — at swarm scale this gets worse.
- **The "just a worse SMAC" failure mode.** If we end up reproducing what SMAC already does, we've spent months for no marginal contribution.

**Strong recommendation from the lit scan: don't build an RTS, and don't even write a new gridworld.** Port the swarm dynamics we care about onto an existing JAX-fast benchmark. **Multi-Agent Craftax** ([arXiv:2511.04904](https://arxiv.org/abs/2511.04904)) is purpose-built for this — JAX-native, 250M steps/hour on one GPU, terrain biomes + heterogeneous units + trading. It's the single benchmark that overlaps *both* ideas 4 and 5 simultaneously.

**Literature backup.**
- **Vinyals et al. 2019 — AlphaStar.** Discussion preprint at [arXiv:2012.13169](https://arxiv.org/abs/2012.13169). Take-away: PFSP > League > raw self-play.
- **Samvelyan et al. 2019 — SMAC** ([arXiv:1902.04043](https://arxiv.org/abs/1902.04043)). The benchmark to beat (or to deliberately *not* beat — see "just a worse SMAC" risk above). Scenarios `3s5z_vs_3s6z` and `MMM2` are where the credit-assignment cliff shows up cleanly.
- **Yang et al. 2018 — Mean Field MARL** ([arXiv:1802.05438](https://arxiv.org/abs/1802.05438)). The principled answer for N>50; treats neighbour interactions as agent ↔ neighbour-mean. Connects to our `project_uid_normalization_tradeoff.md` — that note hinted at the dimensionality problem; mean-field is the formal version.
- **Zheng et al. 2018 — MAgent** ([arXiv:1712.00600](https://arxiv.org/abs/1712.00600)). 1M agents on one GPU; the most direct "swarm A vs swarm B on a grid" precedent. Not JAX, but pip-installable.
- **Suarez et al. 2021 — Neural MMO Platform** ([arXiv:2110.07594](https://arxiv.org/abs/2110.07594)). 1–1024 agents, persistent world, terrain + resources + combat. Closest published thing to "AoE-style RL benchmark"; population-size *itself* drives skill emergence — a result worth replicating at smaller scale.
- **Huang/Ontañón et al. 2021 — Gym-µRTS** ([arXiv:2105.13807](https://arxiv.org/abs/2105.13807)). Full RTS economy + terrain + multi-unit, single-machine RL (60 GPU-hours to SOTA). The realistic AoE-port target if we go that route.
- **Al Omari et al. 2025 — Multi-Agent Craftax** ([arXiv:2511.04904](https://arxiv.org/abs/2511.04904)). **Read this first if §5 is going to happen.**
- **JaxMARL** ([arXiv:2311.10090](https://arxiv.org/abs/2311.10090), [github](https://github.com/FLAIROx/JaxMARL)). The relevant envs: `jaxmarl/environments/smax/` (JAX SMAC — closest to §5), `jaxmarl/environments/coin_game/` (two-player red/blue grid resource collection — closest to current setup; sanity-check baseline), `jaxmarl/environments/jaxnav/` (multi-robot continuous navigation). Baselines `IPPO/MAPPO/IQL/VDN/QMIX/TransfQMIX/SHAQ/PQN-VDN` — all CleanRL-style single-file, Hydra-configurable. Use **VDN/QMIX as the credit-assignment-aware baseline** for any N>8 scale-up.

**Open questions.** Discrete-time (turn-based, Lux AI) or real-time (SMAC)? Do we keep the **fog-of-war / belief-asymmetry primitives** that drive most of our current XAI? What's the *one* scientific question that justifies this scope (not just "MARL at scale")? Candidate: **"Does the cohesion ↔ coverage Pareto dial (`project_cohesion_coverage_dial.md`) survive at swarm scale, or does mean-field smearing dissolve it?"** That's a falsifiable scientific claim that connects directly to our existing work.

---

## 6. From the engineering retrospective — open questions and future directions

**Status:** audit/import. Not new ideas; pointers to lines in `docs/09-engineering-retrospective.md` so the wild-ideas scratchpad doesn't lose track of them. When one of these graduates to a real experiment it gets its own §N entry with cheapest-first scoping.

### 6.1 · Open questions (retro §6)

Each line: **question — what an answer would change — pointer.**

- **Q1 — Heterogeneous per-agent deviation rate.** Is mission damage a function of `Σᵢ ρᵢ` or of the full vector `(ρ₁, …, ρ_k)`? Already partially answered in `project_hetero_sweep.md` (ΔJ varies 6.5 pp across asymmetry at fixed Σρ=1.0); the retrospective version is the formal `(ρ₁, ρ₂)` sweep at matched Σ. — Pointer: `docs/09-engineering-retrospective.md` §6 Q1.
- **Q2 — ΔJ decomposition model.** Fit `ΔJ(k, ρ) ≈ α·k + β·k·ρ·C + γ·(k choose 2)·σ²` to the existing sweep. R² high → predictive; R² low → residuals tell you what term is missing. Tie to F1 in §6.2 — this *is* the Triangle. — Pointer: `docs/09-engineering-retrospective.md` §6 Q2.
- **Q3 — Mission-agnostic resilience triangle.** Magnitude / brittleness / timeliness has only been computed for coverage. Re-derive on search-and-rescue, escort, patrol. **Blocked by F6** (mission primitive library) and naturally aligns with §5's Multi-Agent Craftax port. — Pointer: `docs/09-engineering-retrospective.md` §6 Q3.
- **Q4 — Training-time vs eval-time attacks.** Today red shows up at evaluation only. Coevolutionary training with red present from ep 0 likely produces qualitatively different blue routing protocols. Architectural overlap: the I/W/F prior axis from §3 is *naturally* a training-time attack model. — Pointer: `docs/09-engineering-retrospective.md` §6 Q4; F3 in §6.2.
- **Q5 — Comm-topology attacks.** Today red attacks the *team-label merge*. A topology attack (red drops/rewrites specific edges in the comm graph) is a different threat surface with its own `k*(θ)` and the env already supports it. — Pointer: `docs/09-engineering-retrospective.md` §6 Q5.
- **Q6 — Realistic detector ROC.** Today blue knows who red is. Sweep detector (TPR, FPR) → `ΔJ(k, ρ, detector)` surface. The "cleanest publishable plot" candidate. — Pointer: `docs/09-engineering-retrospective.md` §6 Q6; F4 in §6.2.
- **Q7 — Comm-radius as an attack surface.** Coordinated red walking *away* from blue breaks the team graph without any `MAP_UNKNOWN` writes. Known-occasionally-happens, never measured. — Pointer: `docs/09-engineering-retrospective.md` §6 Q7.
- **Q8 — Larger N, larger grid regime.** All compromise-sweep work is N=5 on 16×16. Off-ladder warm-start (§3.1 of retro / `project_offladder_warmstart.md`) caps coverage at ~60% from N=4 → N≥6. With the current MC critic, walk the ladder; with the swarm-scale plan in §5, this question becomes "does the Triangle survive at N=50+?". — Pointer: `docs/09-engineering-retrospective.md` §6 Q8.

### 6.2 · Future directions (retro §7)

- **F1 — ΔJ decomposition + Attack-Resilience Triangle as §8 of `meta-report.html`.** Closes the narrative arc from "evidence → aggregate" to "model → metric → generalization". Direct dependency: Q1 + Q2. — Pointer: `docs/09-engineering-retrospective.md` §7 F1.
- **F2 — Integrate with JaxMARL.** Free access to SMAX, Overcooked, MPE for testing the Triangle's generalization claim. Already cited in §5 of this doc as the right port target. — Pointer: `docs/09-engineering-retrospective.md` §7 F2.
- **F3 — Training-time attacks (coevo from ep 0).** Architectural change: red policy updated *inside* the blue training loop. Composes with §3 Insider arm. — Pointer: `docs/09-engineering-retrospective.md` §7 F3.
- **F4 — Realistic detector.** Blue gets noisy team_id signal. Sweep ROC vs. ΔJ. Produces `k*(θ, detector)` surface — joint attack/defense optimisation space. — Pointer: `docs/09-engineering-retrospective.md` §7 F4.
- **F5 — Publish codebase as a MARL library.** Dependency cleanup, config schema freeze, public API. Composes with F2. — Pointer: `docs/09-engineering-retrospective.md` §7 F5.
- **F6 — Mission primitive library.** Coverage is one mission. Search, rendezvous, escort, patrol, delivery share structure (partial obs, time budget, team goal). Required for Q3. — Pointer: `docs/09-engineering-retrospective.md` §7 F6.
- **F7 — Formal verification of invariants.** Extract team-label exclusion + comm-merge protocol as formal invariants ("red's `local_map` writes never reach blue's merged belief"). Prove resilience properties, not just measure them. The most-academic of the futures; lowest priority unless someone wants it. — Pointer: `docs/09-engineering-retrospective.md` §7 F7.

---

## 7. Architecture and curriculum directions from the taxonomy / RL guide

**Status:** audit/import. Pointers from `docs/01-rl-overview.md`, `docs/02-rl-training-guide.md` §15, `docs/03-rl-taxonomy.md` §2, and `docs/06-experiment-report.md` §6. Not new ideas — locking them into the scratchpad so they don't disappear into long-form docs.

### 7.1 · Automatic curriculum for env design (UED family)

`docs/03-rl-taxonomy.md` §2 surveys three Unsupervised-Environment-Design methods. Highly relevant — our env is parameterised on `grid_width / grid_height / wall_density / num_agents / comm_radius`, exactly the regime UED targets.

- **PAIRED — Dennis et al. 2020 (NeurIPS)** ([arXiv:2012.02096](https://arxiv.org/abs/2012.02096)). Adversary designs envs to maximise regret between protagonist and antagonist. Prevents impossible tasks by construction.
- **MAESTRO — Samvelyan et al. 2023 (ICLR)**. Multi-agent PAIRED; the only one purpose-built for cooperative MARL. **The most relevant of the three for us.**
- **ACCEL — Parker-Holder et al. 2022 (ICML)** ([arXiv:2203.01302](https://arxiv.org/abs/2203.01302)). Evolutionary env parameters, no adversary required. Cheaper to train than PAIRED/MAESTRO.

**Why it matters here.** Today we hand-tune (`grid_K`, `wall_density`, `comm_radius`) curricula via the warm-start ladder (`project_adv_ladder.md`, retro §3.1). UED would automate the schedule and could surface env regions where the cohesion ↔ coverage dial (`project_cohesion_coverage_dial.md`) breaks. Predict: ACCEL > MAESTRO > PAIRED for our scale (single-machine, ~50k-step envs).

**Cheapest first experiment.** ACCEL on the existing C2 setup, varying `wall_density ∈ [0, 0.3]` and `comm_radius ∈ {3, 5, 7}`. ~2 days engineering (writing the env-mutator), then training time similar to today.

**Risk.** UED on a *3-axis* env-param space is well-studied; on a 5-axis space (our setup) it's not. Population-based regret ranking can degenerate when env-param dimensions interact non-linearly.

### 7.2 · Architectures called out in the guide but not implemented

`docs/06-experiment-report.md` §6 (Future Work) and `docs/01-rl-overview.md` cite specific architectures that would change the policy class without changing the env:

- **GNN policy (jraph).** Graph-native policy over the comm-graph. Already supported by the env; jraph is a JAX-native library (no DLPack bridge). Highest-priority architectural extension per `docs/03-rl-taxonomy.md` §6. — Composes with **§5** (swarm-vs-swarm; mean-field MARL is what GNN policies look like at N≫50).
- **TarMAC (learned communication).** Agents learn what to broadcast over the comm-graph. Builds on existing `comm_graph.py`. Higher engineering cost than GNN policy but the same risk profile. — Composes with **§3** alien-β (red gets a *different* learned-comm channel from blue → "alien language").
- **UPDeT (transformer over agent observations).** Attention pooling instead of MLP. The natural answer for `N≥16` where joint-red's concat-MLP breaks (see retro §6 Q8). — Composes with **§3** alien-γ.
- **TSPPO (transformer-based sequential PPO).** Same parameter footprint as UPDeT but treats agents as a sequence. Cited in `docs/06-experiment-report.md` §6 Priority 4.
- **SimbaV2 hyperspherical normalization.** A regulariser, not an architecture; cited as a high-impact-low-effort add-on for any of the above. — Composes with the §2.4 obs-normalisation discussion in the retrospective.

**Why this is in §7 not §3.** §3 is about red's *prior over policy space*; this is about *changing the policy space itself*. Orthogonal; can be combined.

**Cheapest first experiment.** GNN policy for blue on existing C2 setup, frozen red, MC critic. ~3 days engineering (jraph + Flax integration), training comparable to current.

### 7.3 · Reward-shaping and stabilisation knobs flagged in the training guide

`docs/06-experiment-report.md` §6 Priority 1 lists the standard PPO regularisation toolkit not currently present:

- LayerNorm after each Dense.
- Dropout (0.1) during update epochs only.
- AdamW with `weight_decay = 1e-4`.
- Running observation normalization (Welford, clip to [-5, 5]).
- Orthogonal initialization (hidden gain `√2`, policy head gain `0.01`, value head gain `1.0`).
- Linear LR annealing to zero.

**The retrospective §2.3 picks the *target* (MC) but not these *training-time regularisers*.** Most are 1-3 LoC each; the bundle is the canonical "PPO actually works in practice" setup. We adopted MC + grad_clip; the rest is on the table.

**Why it's in §7 not as its own §.** It's not a research idea — it's engineering hygiene to do *before* any of the wild ideas above are scoped at scale. The §3 (prior), §4 (energy), §5 (swarm-scale) experiments will all run cleaner with these in place.

---

## Cross-cutting observations

- **Ideas 4 and 5 share an axis: heterogeneity.** Terrain types, unit types, resource types — each is "the env / agents are no longer interchangeable". This is also what general-sum (§2) buys us in reward space. The natural sequence: §2 (heterogeneous reward) → §4 (heterogeneous env) → §5 (heterogeneous agents).
- **Ideas 1 and 4 are both about geometry/topology.** Hex grid is symmetric heterogeneity (every cell same kind, different connectivity); terrain is asymmetric heterogeneity (cells differ but connectivity is uniform). Could compose.
- **Idea 3 (red's prior) is orthogonal** — it's an init trick that applies to any of the above. The Insider arm is also the *cleanest test* of "does cooperative-policy-space neighbour adversarial-policy-space?" — answer carries through to all the other ideas.
- **Strong cross-cutting recommendation from the lit scan.** **Don't write new envs.** For §4 (terrain+energy) the JaxMARL `overcooked_v2` tile-typed infrastructure is reusable. For §5 (swarm-vs-swarm) **Multi-Agent Craftax** ([arXiv:2511.04904](https://arxiv.org/abs/2511.04904)) is the single benchmark that overlaps both directions and is JAX-native. Building infrastructure is where research projects die.
- **Read-once-before-anything list.** AlphaStar discussion ([arXiv:2012.13169](https://arxiv.org/abs/2012.13169)) for PFSP. Mean-Field MARL ([arXiv:1802.05438](https://arxiv.org/abs/1802.05438)) for N>50 plumbing. Multi-Agent Craftax ([arXiv:2511.04904](https://arxiv.org/abs/2511.04904)) for the most relevant existing benchmark. Constrained Policy Optimization ([arXiv:1705.10528](https://arxiv.org/abs/1705.10528)) for the energy-budget formalism.

## Graveyard

*(Empty for now. When ideas die, archive them here with one-line "what we tried, why it didn't work".)*
