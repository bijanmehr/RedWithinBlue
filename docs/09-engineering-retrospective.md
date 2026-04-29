# Engineering Retrospective — RedWithinBlue

> Companion to `experiments/meta-report/meta_report.html`. The meta-report
> answers "what did we find?". This document answers "how did we get
> there, what went wrong, and what should the next person do differently?"
>
> Results-oriented readers can stay on the meta-report. This document is
> for engineers picking up the codebase or extending it.

**Status:** living document. Update when a decision is made, a strategy
is abandoned, or an experiment produces a transferable lesson.

---

## 1 · Preface — why this document exists

The meta-report is a results deliverable. It does not capture the
dozens of decisions, abandoned branches, and negative results that
shaped what the codebase actually is. Without this retrospective,
that knowledge lives in commit messages and in the author's head —
both lossy. The goal here is to write down, while it is still
remembered:

- **Architectural decisions** and the alternatives we ruled out.
- **Training strategies** that worked, and *why* they worked.
- **Named failures** — with their symptoms, root causes, and resolution.
- **Open questions** the current experiments cannot answer.
- **Future directions** that would be worth a fresh worktree.

The bar for inclusion is: *would a future engineer regret it if this
weren't written down?*

---

## 2 · Architectural decisions

### 2.1 · JAX-native, not PyTorch or multiprocessing-based

**Decision.** End-to-end JAX: `jax.jit`, `jax.vmap`, `jax.lax.scan`,
Flax for modules, Optax for optimizers. No Python-level multiprocessing
for seed parallelism.

**Alternatives considered.** (a) PyTorch + DDP + multiprocessing seeds.
(b) Pure NumPy env + PyTorch agents. (c) JAX env + PyTorch agents via
DLPack bridge.

**Why JAX-native won.** `vmap` over seeds gives us N-seed training
essentially for free — the env, the agents, the rollout, and the loss
all compose through one `jit`. On CPU we get ~200 eps/s for a 5-seed
vmap on the coverage task; on a single GPU the same code is bound by
env compute, not gradients, because the networks are tiny (§Appendix A
of meta-report: ≈64 k total parameters). Cross-language DLPack bridges
would have made profiling and debugging significantly harder.

**What this costs.** A steep JAX learning curve for anyone without prior
functional-programming exposure. `vmap` bugs manifest as shape errors
several function calls away from the cause, and `ConcretizationTypeError`
under `jit` is the canonical trap (any Python-level `int(tracer)` or
`if tracer > 0` breaks). The rollout code in `training/rollout.py` is
denser than a PyTorch equivalent would be.

---

### 2.2 · CTDE with a shared per-agent actor and centralized critic

**Decision.** Blue uses decentralized actors (shared weights across all
N agents) plus a centralized critic V(s) that sees the joint
observation. Red uses a single centralized `JointRedActor` that emits
factorized per-agent logits — `[n_red, |A|]` — from the concatenated
joint red obs.

**Alternatives considered.**
- Independent actor per blue agent (no weight sharing): abandoned
  because it does not generalize across N and the per-agent data
  volume is too thin at N = 5.
- Fully centralized blue policy emitting a joint action: rejected
  because deployment-time communication becomes mandatory and the
  policy cannot run with any agent offline.
- Fully decentralized red with independent heads: rejected because red
  has n_red = 2, which is small enough for a joint policy to be
  tractable *and* lets red exploit coordination that factorized-red
  cannot express.

**Why this split works.** Blue benefits most from decentralization
(robust to agent loss, scales with N) and shared weights (sample
efficiency). Red benefits most from coordination (n_red is small, the
attack is a small structured perturbation). The asymmetry between blue
and red is architectural, not just a reward change.

**Memory trail.** `project_rwb.md` (clean slate, JAX-native swarm MARL);
`feedback_jax_native.md` (use vmap/pmap, avoid multiprocessing).

---

### 2.3 · Monte-Carlo returns for the central critic — not TD(0), not GAE

**Decision.** The CTDE actor-critic loss in
`src/red_within_blue/training/losses.py` uses **Monte-Carlo returns**
as the critic target. Advantages are `G_t - V(s_t)` with stop-gradient
on returns.

**What we tried before settling here.**
1. **TD(0) with a live central critic** (no target net). Failed on the
   pair-cooperate-coop run: `|loss|` median 0.08 → 196, p99 0.46 →
   23 282 across 15 000 episodes. All 5 seeds collapsed reward +2.07 →
   +0.35 by ep ~1500 and stayed flat. Textbook self-amplifying critic
   feedback on a 100-dim non-stationary observation. Documented in
   `docs/08-stabilization-experiments.md`.
2. **TD(0) + grad_clip = 0.5** (stage 1 of the empirical escalation
   plan). 1500-ep smoke held reward steady at +2.0 but `|loss|` p99
   still drifted 0.13 → 25 inside the smoke window — the clip bandaged
   the catastrophic feedback but didn't fix the underlying instability.
3. **TD(0) + grad_clip + advantage normalization** (stage 2). Did not
   test at full scale — superseded by stage 3.
4. **Monte-Carlo returns + grad_clip** (stage 3, current). Stable at
   full scale. The critic target no longer depends on its own
   predictions, so the self-amplifying feedback loop is structurally
   impossible.

**Alternative not taken: GAE(λ=0.95).** The earlier PettingZoo
version (since removed; see Appendix A) used this. We chose MC over GAE for two reasons: (a) episode
length is 100 steps — MC variance is manageable; (b) the terminal
coverage bonus is ~80× per-step reward, and GAE(0.95) discounts the
terminal to ≈0.006 at t=0, effectively zeroing it out for early
timesteps. MC gives the terminal bonus full credit at every t, which
is what a coverage task needs.

**Negative result: twin critic + target net (EXP-A).** The stabilization
experiments tested whether SAC-style twin-Q with Polyak target would
have beaten MC if we had used them from the start. All three variants
(A1 target-only, A2 twin-only, A3 both) underperformed the MC baseline
A0 on final reward at 15 000 × 5 seeds. MC won cleanly. Recorded in
`experiments/stabilization/stabilization_report.html`.

**Memory trail.** `feedback_empirical_escalation.md` (stage cheapest
test first); the full staged plan at
`/Users/bijanmehr/.claude/plans/bubbly-strolling-puddle.md`.

---

### 2.4 · Observation design — flat 23-dim vector, `log1p`-normalized

**Decision.** Per-agent observation is a 23-dim flat vector containing
local fog-of-war patch, positional encoding, agent-uid embedding,
neighbor count, and a coverage progress signal. All magnitude/count
features are wrapped in `log1p(x) / 4.0` before concatenation.

**Why `log1p`.** Raw visit counts and neighbor counts span 0–~80 in the
same episode; unnormalized features that big dominate the L1 norm of
the input and make the early training unstable (gradients are chased
by the high-magnitude features, the fog patch gets ignored). `log1p`
compresses to O(1) without zero-clipping and is monotone, so the
policy still sees "more is more". Division by 4 empirically gets the
compressed features to the same scale as the fog-binary features. This
was found by iteration, not principle.

**Memory trail.** `feedback_log1p_normalization.md` — when the user
says "add log1p on X" the transformation is this one.

**What this doesn't generalize.** Longer episodes or larger grids will
eventually saturate `log1p(x)/4` and this normalization will need
revisiting. Not a universal recipe — a local solution to the 100-step,
16×16 regime.

---

### 2.5 · Reward design — coverage + cohesion + fog potential

**Current reward** (in `make_multi_agent_reward`):
1. **Team coverage delta** — per-step new-cells-seen reward.
2. **Terminal coverage bonus** — big one-shot at episode end.
3. **Cooperative weight** — small team-reward component on top of
   per-agent reward, tunes cohesion vs. coverage.
4. **Fog-of-war potential** — potential-based shaping that pulls agents
   toward nearest unknown cell (policy-invariant; verified to preserve
   optimality).
5. **Disconnect grace** — soft guardrail on comm-graph disconnection.

**What we tried and kept.**
- Fog-of-war potential: +6.6 pp mean, +21.9 pp min coverage on N=5
  16×16 (memory: `project_fog_of_war_reward.md`).
- Spread-weight bonus (mean pairwise distance): with coop=0 and hard
  guardrail, lifted 32×32 N=8 coverage 65.8% → 88.7%
  (`project_spread_reward.md`). Compare variants on coverage, not
  reward — spread adds a reward confound.

**What we tried and abandoned.**
- Dense idleness penalties for patrolling: in notes but never shipped —
  the coverage-reward pull already produces patrol-like behavior after
  full saturation.
- Count-based exploration bonuses: discussed, not needed once fog
  potential was in place.
- Hard disconnect guardrail at fail_penalty=-2.0: *tried* disconnect
  grace as a soft replacement; it regressed coverage at the 10%
  fail-rate regime. Keep the hard guardrail as the baseline; keep
  disconnect grace as a per-agent observable but not as the team-level
  rule (`project_disconnect_grace.md`).

**Trade-off that is not solved, only dialed.** Cohesion vs. coverage is
a continuous dial at N=8: `cooperative_weight ∈ [0, 0.015]` is one
tradeoff; cluster floor pins at ~41% for any non-zero coop
(`project_cohesion_coverage_dial.md`). We do not have a single reward
that gives both maximum coverage and enforced cohesion — the user
chooses a point on the dial per experiment.

---

## 3 · Training strategies that worked

### 3.1 · Warm-start ladder for scaling N and grid size

**Recipe.** When scaling from N → N' or grid_K → grid_K', always warm
from the nearest-dominated checkpoint rather than initialize fresh.
For the main adversarial ladder we used four sequential rungs
(6→8→16→16): warm blue each rung, fresh red each rung.

**Critical rule.** **Warm-start BOTH the actor and the critic.** A
re-initialized central critic is the main collapse mode for CTDE
scale-ups — the actor arrives competent, the critic arrives random,
and early advantages are noise that the actor follows off the good
policy (`feedback_warm_start_both.md`).

**Gentle fine-tune rule.** Off a converged source, cut learning rate
5× and total episodes 4×. The source LR is tuned for a randomly-init
policy; using it on a converged policy overwrites what you transferred
(`feedback_gentle_fine_tune.md`).

**When the ladder breaks.** Off-ladder jumps (e.g. N=4 → N=7 skipping
N=5, N=6) block critic tiling and cap coverage at ~60% versus N=4
baseline's 97%. Needs an intermediate rung
(`project_offladder_warmstart.md`).

### 3.2 · Coevolutionary evolution strategies for red

**Recipe.** Once blue is trained, freeze blue and train red via
evolution strategies (ES) with a small population (pop=8, 20
generations, 2 eps per pair at r6). At r6 this lifted blue coverage
from 11.0% → 25.5% red-achievable-damage in under 2 min wall-clock
(`project_coevo_r6.py`, `project_coevo_r6.md`).

**Why ES beat gradient-based red.** Red's reward is sparse,
gradient-free objectives (mission degradation of blue) are cleaner
with ES than with REINFORCE on this problem scale. See also §4.7 for
the off-policy DQN negative result on red.

### 3.3 · Minimal-change-first debugging

Before stacking speculative new code, try the smallest config tweak
that could plausibly fix the issue. Only escalate to new code if the
minimal change fails at full scale
(`feedback_minimal_change.md`). The stabilization experiments are the
canonical example: grad_clip first (stage 1), adv-norm second
(stage 2), MC target third (stage 3). Stage 3 was needed; stages 1–2
were not sufficient on their own.

---

## 4 · Named failures and their lessons

Most valuable section. Each entry: **symptom → root cause → fix → lesson.**

### 4.1 · Critic drift on pair-cooperate-coop (TD(0) collapse)

- **Symptom.** All 5 seeds of `configs/pair-cooperate-coop.yaml` went
  from reward +2.07 to +0.35 by ep ~1500 and stayed flat for 13 500
  more episodes. Critic loss p99 grew from 0.46 to 23 282.
- **Root cause.** TD(0) bootstrapping on a 100-dim non-stationary
  `global_seen_mask` observation. The critic's own predictions became
  its own targets; any bias self-amplified; grad_clip alone could not
  break the feedback.
- **Fix.** Switch critic target from TD(0) to Monte-Carlo returns
  (stage 3 of the empirical escalation plan). See §2.3.
- **Lesson.** On central critics with high-dim non-stationary
  observations, TD(0) is structurally fragile. Default to MC or GAE;
  reach for TD(0) only when the observation is low-dim and nearly
  Markov in the agent features.

### 4.2 · Joint-red entropy bug — `ent_coef` not entering the loss

- **Symptom.** Red policies collapsed to `STAY` across the board; red
  reward went negative and stayed there.
- **Root cause.** The REINFORCE loss for joint-red was returning
  entropy via `has_aux=True` (a logging path) instead of subtracting
  `ent_coef * entropy` from the scalar loss. The intended regularizer
  was not in the gradient.
- **Fix.** Move entropy into the loss scalar:
  `loss = -E[log π · A] - ent_coef * H(π)`.
- **Lesson** (`feedback_ent_coef_in_loss.md`). `has_aux` is logging,
  not regularization. A policy collapsing to STAY is diagnostic for
  entropy not being in the gradient — grep the loss, not the logs.

### 4.3 · Eval GIF loading the wrong actors

- **Symptom.** Evaluation GIFs of trained red policies looked
  identical to random red — red agents didn't coordinate, didn't
  hoard, didn't block.
- **Root cause.** `experiment_report.py` loaded only the blue
  checkpoint and applied the blue policy to red agents. Red was driven
  by a blue-trained actor the whole time.
- **Fix.** For `red_policy=joint`, always load
  `joint_red_checkpoint.npz` separately; never fall back to the blue
  actor (`feedback_eval_loads_all_actors.md`).
- **Lesson.** Eval-time code paths are where the lies live — training
  code sees the real networks because the loss depends on them; eval
  code can silently substitute and nothing fails. Add an assertion
  that checkpoints load cleanly, don't assert on match.

### 4.4 · Off-ladder N warm-start

- **Symptom.** Warm-starting N=5 and N=7 directly from an N=4 16×16
  checkpoint capped coverage at ~60% versus N=4 baseline's 97%.
- **Root cause.** The central critic's learned value function tiles
  the grid at the source N. Jumping multiple rungs breaks the tiling;
  the warm-started critic disagrees with the warm-started actor and
  the actor follows noise.
- **Fix.** Introduce intermediate rung (e.g. N=4 → N=5 → N=6 → N=7)
  (`project_offladder_warmstart.md`).
- **Lesson.** Warm-start transfer is continuous, not arbitrary. The
  jump size that worked for one (N, grid) combination is not the jump
  size for another. Walk the ladder.

### 4.5 · `uid` normalization tradeoff

- **Symptom.** After enabling `env.normalize_uid`, same-N 32×32
  fine-tune destabilized — the uid-column gradient became ~4× weaker
  and the agent-identity signal got drowned by positional features.
- **Root cause.** `normalize_uid` scales uid to [0, 1/N], which helps
  N-mismatched transfer (fresh agents land in a range the critic has
  seen) but squashes the uid gradient in same-N training where full-
  range uids are expected.
- **Fix.** Condition-gate `normalize_uid=True` only on transfers across
  N; leave it off for same-N fine-tunes
  (`project_uid_normalization_tradeoff.md`).
- **Lesson.** Features that help transfer can hurt fine-tune, and vice
  versa. Do not flip this kind of flag globally — make it a config
  switch and document when to use it.

### 4.6 · Disconnect-grace mechanism regressed coverage

- **Symptom.** Shipped a soft-guardrail disconnect-grace mechanism to
  replace the hard `fail_penalty=-2.0`; coverage regressed in the 10%
  fail-rate regime.
- **Root cause.** The soft guardrail lets blue teams occasionally
  "cheat" the cohesion constraint in high-reward rollouts, shifting
  the learned policy toward borderline-disconnected configurations.
  Once the hard penalty is gone, the gradient has no reason to enforce
  the constraint consistently.
- **Fix.** Keep the hard guardrail as the team-level rule. Keep
  disconnect-grace as a **per-agent observable** (useful as a
  diagnostic in eval gifs) but not as the training signal
  (`project_disconnect_grace.md`).
- **Lesson.** "Softer" is not automatically "better". Hard constraints
  are sometimes load-bearing precisely because they are hard.

### 4.7 · Off-policy Double-DQN red (EXP-B) — worse than REINFORCE

- **Symptom.** At a matched red env-step budget, off-policy
  Double-DQN with twin-Q, Polyak target, and 50 k replay produced
  *weaker* red than on-policy REINFORCE with entropy regularization —
  at 20× the wall-clock.
- **Root cause hypothesis.** At n_red = 2 the joint policy space is
  small enough that REINFORCE's sample inefficiency is not actually
  bottlenecking performance. Off-policy replay helps when samples are
  expensive relative to gradient steps; here, env-steps are cheap
  (JAX-jit env), so the replay machinery pays overhead it can't
  amortize. Additionally, Double-DQN's min(V1, V2) target is
  conservative, which hurts exploration in the small-action regime.
- **Fix.** Keep REINFORCE + entropy for joint-red; do not adopt
  off-policy machinery for the current scale. Recorded in
  `experiments/stabilization/stabilization_report.html`.
- **Lesson.** "More sophisticated" ≠ "better". Off-policy DRL papers
  target regimes where samples are bottleneck; our bottleneck is
  learning signal, not sample count. Match the algorithm to where the
  actual cost is.

### 4.8 · Sabotage framing flipped mid-analysis

- **Symptom.** The proposal framed red's attack as *information
  corruption* — red writes `MAP_UNKNOWN` into blue's `local_map`.
  After measuring, we found this channel ("Channel 2") accounts for
  ≤1 pp of damage.
- **Root cause.** Red's real damage is **delay**, not theft. Most of
  the 9.2 pp shortfall at C2 is team-label exclusion (the red slot
  never joins the blue message merge) plus redundant coverage (red
  walks over cells blue would have covered anyway). Blue's active
  belief corruption is a minor channel.
- **Fix.** Reframe the §6 narrative from "information theft" to
  "delay" — T(90%) = 85 for B, 120 for C1, never for C2
  (`project_sabotage_as_delay.md`,
  `project_uncertainty_manipulation_thin.md`).
- **Lesson.** The attack model you start with is a hypothesis, not a
  finding. Measure the contribution of each channel before writing the
  discussion; otherwise the discussion describes an attack the system
  doesn't actually have.

---

## 5 · Meta-lessons (extract: what generalizes across failures)

1. **Name the algorithm last.** Start from raw REINFORCE / actor-critic
   and a plain MSE critic; only add named-algorithm machinery (SAC, PPO,
   DQN, twin-Q, target nets) when a plain-vanilla baseline fails at
   full scale. Several of the failures above (4.1, 4.7, the whole
   stabilization sweep) came from reaching for machinery before the
   baseline was tested (`feedback_bottom_up.md`).

2. **Warm-start both, or warm-start neither.** CTDE scale-ups die on
   the critic side, not the actor side. See §2.3, §3.1, §4.4.

3. **Negative results are findings.** EXP-A (twin critic) and EXP-B
   (off-policy DQN) both failed cleanly. They are reported as findings
   because "don't do this" is useful — and the measured
   counterfactuals rule out the plausible alternatives so the next
   reader doesn't re-run them.

4. **Clean-slate over patching.** When multiple patches to a component
   haven't worked, stop patching. The stabilization negative results
   were cleaner than six months of patches to a TD(0) critic would
   have been (`feedback_clean_slate.md`).

5. **Update `experiments/README.md` in the same turn as any change
   that affects how an experiment is run or interpreted.** Otherwise
   the operator manual lies (`feedback_experiments_readme.md`).

6. **Coverage is the observable, cohesion is the confounder.** Do not
   compare variants on `reward` when cohesion weights differ across
   variants — compare on `coverage %`. Reward is a scalarization; the
   scalarization differs across runs (`project_spread_reward.md`).

7. **The eval path is where the lies live.** Training sees real
   networks because the loss depends on them. Eval can silently
   substitute. Add explicit checkpoint-loading assertions in every
   eval entry point.

---

## 6 · Open questions (experiments not yet run)

**Q1. Heterogeneous per-agent deviation rate.** The misbehavior budget
sweep varies ρ but assumes all k compromised agents share the same ρ.
The math framing in the meta-report (`M = k·ρ`) is a scalar collapse
of a vector quantity. Is mission damage a function of the sum
`Σᵢ ρᵢ`, or of the full vector? Needed experiment: sweep (ρ₁, ρ₂) at
matched Σρᵢ with k=2. Tests whether one loud traitor is worse than
two quiet ones.

**Q2. ΔJ decomposition model.** Fit
`ΔJ(k, ρ) ≈ α·k + β·k·ρ·C + γ·(k choose 2)·σ²` to the current sweep
data. If R² is high, we have a predictive model; if low, the
residuals tell us what term is missing. Not run because the hetero-ρ
data (Q1) would stabilize the fit.

**Q3. Mission-agnostic resilience metric.** The Attack-Resilience
Triangle proposal (magnitude / brittleness / timeliness) is
dimensionless by construction but has only been computed for the
coverage mission. Does the same triangle make sense for a
search-and-rescue task? An escort task? A patrol task? Would need
those tasks instantiated in the env first.

**Q4. Training-time vs eval-time attacks.** Current experiments
assume red appears at evaluation. What if red is present throughout
training (blue co-adapts to compromise)? Intuition: blue would
develop routing protocols that route around red — qualitatively
different resilience. Not tested.

**Q5. Communication-topology attacks.** Current red attacks the
message merge at team-label level. A topology attack (red drops/
rewrites specific edges in the comm graph) is a different threat
surface with its own `k*(θ)`. Env supports it; no training runs yet.

**Q6. Partial observability of red (detector realism).** Current
experiments assume blue knows who red is (team_id). A realistic
detector would have false positives and false negatives. Sweep
detector ROC vs. ΔJ — at what detector accuracy does blue's coverage
recover? This is the proposal's `k*(θ)` story extended to imperfect
detection.

**Q7. Comm-radius as an attack surface.** `comm_radius = 5` is
fixed. A coordinated red could walk *away* from blue to break the
team graph without any `MAP_UNKNOWN` writes. Known to happen
occasionally; not measured systematically.

**Q8. Larger N, larger grid regime.** All compromise-sweep work is
at N=5 on 16×16. Scaling to N=16 on 32×32 is the stated target
(per the adversarial warm-start ladder), and early off-ladder
results (§4.4) suggest critic tiling is the bottleneck. Not
attempted with the current MC critic; revisit after §Q2.

---

## 7 · Future directions

**F1. ΔJ decomposition + Attack-Resilience Triangle as §8 of the
meta-report.** Closes the narrative arc from "claims → evidence →
aggregate → *model → metric → generalization*". See critique #19 in
the conversation record.

**F2. Integrate with jaxmarl.** The env currently speaks a custom
interface. jaxmarl integration would give us free access to SMAX,
Overcooked, MPE, etc. as mission variants to test the Triangle's
generalization claim (F1 depends on this for Q3 answers).

**F3. Training-time attacks.** Implement a coevolutionary training
mode where red is in the population from ep 0. Blue co-adapts;
compare to eval-time-only compromise. Architectural change: red
policy needs to be updated inside the blue training loop.

**F4. Realistic detector.** Blue gets a noisy team_id signal. Sweep
detector ROC (TPR, FPR) vs. ΔJ recovery. Produces the
`k*(θ, detector)` surface — joint attack/defense optimization space.

**F5. Publish the codebase as a MARL library.** The JAX-native
env + CTDE trainer + coevolutionary red + compromise-sweep tooling
is reusable. Dependency cleanup, config schema freeze, and a public
API are needed before external users.

**F6. Mission primitive library.** Coverage is one mission. Search,
rendezvous, escort, patrol, delivery share structure (partial obs,
time budget, team goal). A small library of primitives would let
the Triangle claim (F1) be tested across missions and would make
the codebase useful beyond the proposal.

**F7. Formal verification of invariants.** The team-label exclusion
mechanism and the comm-merge protocol are both currently implicit
in the env code. Extracting them as formal invariants (e.g. "red's
local_map writes never reach blue's merged belief") would let us
prove resilience properties, not just measure them.

---

## Appendix A · Abandoned branches and why

| Branch / experiment | Abandoned because | Date |
|---|---|---|
| PettingZoo + PPO (`temp/maexp_pettingzoo_env_py.py`, since removed from the repo) | Too much boilerplate, no JAX parallelism, PettingZoo API adds overhead that JAX env avoids | pre-v1.0 |
| TD(0) critic path | Critic drift; unable to stabilize without restructuring the target (see §4.1, §2.3) | 2026-04 |
| Twin-Q + Polyak target for central critic (EXP-A variants) | Underperformed MC baseline at full scale | 2026-04-20 |
| Off-policy Double-DQN for joint-red (EXP-B) | Weaker than REINFORCE at 20× wall-clock (see §4.7) | 2026-04-20 |
| Soft disconnect-grace as team-level rule | Coverage regression at fail_penalty=-2.0 (see §4.6) | before 2026-04 |
| Hard `fail_penalty=-2.0` removal in favor of potential-only shaping | Needed as guardrail for training stability | before 2026-04 |

---

## Appendix B · Key configs and their purpose

- `configs/pair-cooperate-coop.yaml` — 10×10, N=2 cooperative, used as
  the stabilization experiments reference.
- `configs/compromise-16x16-5-3b2r.yaml` — 16×16, N=5 with 3 blue +
  2 red, used as the compromise-sweep reference.
- `configs/compromise-16x16-5-4b1r.yaml` — same but 4 blue + 1 red
  (C1 in the meta-report).
- `configs/*-survey-local-*.yaml` — the N-scaling ladder rungs.

Update this appendix when adding or retiring a reference config.

---

## Appendix C · How to extend this document

- **New failure?** Add a §4.x entry with symptom / root cause / fix /
  lesson. If the failure generalizes, update §5.
- **New decision?** Add a §2.x entry with alternatives-considered.
- **Experiment run and learned something?** Update the relevant memory
  file under `~/.claude/projects/.../memory/` *and* update this doc.
  Memory is for cross-conversation recall; this doc is for future
  engineers.
- **Question answered?** Move from §6 to §4 (if it produced a
  failure-worth-knowing) or §3 (if it produced a working strategy).
  Leave a note in §6 with the answer and a pointer.
