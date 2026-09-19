# Differentiation

Phase 0 deliverable. The claim, stated once, then defended paper by paper against the **seven**
closest works — the audit's five plus the two verification added. Written to survive the strongest
adversarial reading available, since the audit's two-independent-readers criterion cannot be met
with one executor (blocker B4).

---

## 1. The claim

> Long-horizon degradation in tool-using LLM agents has been described as a curve, named as a
> residual, labelled by taxonomy, localized per trace, and separated once — coarsely — into
> transcript and environment components. It has not been **decomposed by intervention into
> mechanism-specific shares**. We branch logged trajectories at individual error steps in a
> stateful tool environment and re-execute them under three `do()`-interventions — history scrub,
> state refresh, oracle fix — to estimate what fraction of the horizon gap is attributable to
> self-conditioning, state staleness and irrecoverability, as functions of normalized trajectory
> position and error type, with uncertainty intervals, under a controlled horizon sweep in which
> subtask difficulty is identical by construction between the short and long conditions.

The novelty is the conjunction: **matched horizon sweep + stateful tool trajectories +
within-trajectory counterfactual replay + population-level mechanism shares**. Every individual
component is borrowed and cited. Nothing here rests on being first to any one of them.

---

## 2. Paper by paper

**HORIZON (arXiv:2604.11978).** Sweeps horizon, types errors into seven FMEA-grounded categories,
validates a judge (κ=0.61 human, 0.84 judge), and shows failure *composition* shifts as horizon
grows. Attribution is per-trace and by label. There is no step localization, no independence null,
no counterfactual, and no causal estimand. **We adopt their taxonomy, their H\* definition, and
their finding as our starting point, and we ask the question their design cannot answer: not which
labels become more frequent, but which mechanisms cause the excess.** The sharpest instance is their
own category definition — catastrophic forgetting is defined as the constraint being *"still present
in the context but not attended to during later reasoning"*, with Memory Limitation kept separate
for genuine overflow. That is an attention claim wearing a memory label, and it is testable by
scrubbing the erroneous history while leaving the constraint in place. HORIZON cannot run that test;
we can.

**Khanal et al. (arXiv:2603.29231).** Establish that reliability decays super-linearly against a
geometric baseline across 10 models and 23,392 episodes, and that decay is domain-stratified. No
error typing, no localization, no mechanism. Their duration measure is a human-time proxy they
themselves flag as imperfect, and their meltdown-onset metric is uncalibrated. **They quantify the
gap from a null; we explain it.** We also inherit a warning from them that we act on: high variance
bifurcates by capability tier and is a capability signature rather than an instability signal, so we
do not read variance as unreliability anywhere in our analysis.

**Sinha et al. (ICLR 2026, arXiv:2509.09677).** The mechanism paper. Demonstrates self-conditioning
by injecting a model's own prior errors into context, and shows thinking mitigates it. The setting
is pure in-context execution: **no tools, no environment state, plans supplied**. They quantify
per-step accuracy and H\_0.5; they do **not** quantify a share of degradation. **We test whether
their mechanism survives when an environment can absorb, propagate or reverse the error, and we
estimate how much of the gap it accounts for.** That is not a restatement, because in their setting
the environment channel does not exist — the only thing an error can contaminate is the transcript.

**ChainSWE (arXiv:2607.02606).** The closest experimental design in existence, and the one that most
constrains our claim. Its Oracle / Seq / Seq+Mem modes separate environment state from transcript:
Oracle→Seq isolates repository state, Seq→Seq+Mem isolates the conversation. It reports 58.9% →
36.5% → 36.9%. **We concede the separation idea entirely — it is theirs.** Four differences carry
our contribution. (i) *Granularity*: they reset the transcript at bug boundaries, roughly twice in a
length-3 chain; we branch at individual error steps, so within-stage self-conditioning — the only
kind Sinha's result predicts — is invisible to their design and visible to ours. (ii) *Estimand*:
they report accuracy differences between whole arms with **no confidence intervals anywhere**; we
estimate shares of a defined quantity with bootstrap CIs and a stated identification assumption.
(iii) *Resolution*: they have no positional hazard, no propagation by error type, and no
irrecoverability arm; π(e,u) and R(e,u) have no counterpart in their work. (iv) *Scope*: Python
coding, seven frontier closed models, chains capped at length 3. **Most importantly, their null is
our motivation.** A 0.4-point transcript effect contradicts Sinha, and the contradiction is
resolvable only at a granularity neither paper used.

**Lu et al. (arXiv:2509.25370).** Root-causes failures into a five-module taxonomy with step-level
human labels, and identifies error propagation as the primary bottleneck. **Propagation is asserted
from labels, not measured by intervention**, and there is no horizon sweep. Our π(e,u) is the
measured version of their claim: the difference in terminal-failure probability between continuing
from the erroneous action and continuing from a corrected one, stratified by category and position.

**CAR (arXiv:2606.08275).** Supplies the machinery we use: an intervention algebra
(`do_resample`/`do_action`/`do_observation`/`do_context`/`do_policy`), a contrastive estimator, the
point-of-commitment rule that prevents false attribution when resampling re-rolls downstream
stochasticity, and a Monte-Carlo Shapley estimator with CIs. **We adopt it and claim none of it.**
Its validation is synthetic SCMs with planted ground truth — no real agent benchmark, not even
Who&When — and it states plainly that **"real tools with side effects are out of scope
(demonstrations use mocked, reproducible tools)."** That sentence is the boundary of the existing
work and the location of ours: side-effecting tools are precisely what makes staleness and
irrecoverability distinguishable from self-conditioning at all.

**Peng et al. (arXiv:2607.27283).** Names our central quantity — the **horizon residual**
Γ_H = log(P_expected/P_observed) with P_expected = Π q_i, which is our Δ(s) with the sign reversed —
and argues that no benchmark may claim a long-horizon failure without computing it. **We cite it on
first use and do not claim the quantity.** It is a position paper with no experiments, and its
stated limitation is our contribution: *"Γ_H does not identify context degradation, planning failure,
state contamination, or recovery; it identifies a contrast under one protocol."* Its §8 agenda calls
for exactly our interventions. We also adopt its methodological warning as a required analysis:
two admissible checkpoint protocols can yield opposite-signed residuals, so our stage protocol is
pre-registered and its sensitivity reported.

---

## 3. Method precedents we adopt rather than claim

**Shao et al. (arXiv:2608.20563)** contribute two things we take. *Exposure*: instrument checkpoints
and report success conditional on the agent having reached the point where the capability is
exercisable — in their CSR task 9 of 10 failures occur before exposure, so an uncorrected analysis
would have attributed to memory what was a discovery failure. Without this conditioning our θ_state
and θ_self are both inflated by runs where the agent never observed the state at issue. *Placebo
matching*: their rescue/placebo pairs are matched on byte and line count, with McNemar tests and
discordant-pair CIs. The audit's simulated reviewer demanded a placebo control as a novelty-bearing
addition; it is prior art, and we cite it as such. We also carry their warning: their intervention
effect **reverses sign** between Gemini 2.5 Flash and Gemini 3.7 Flash, so mechanism shares are not
assumed to transfer across model generations.

**REFLECT (arXiv:2606.09071)** validates error attributions by prefix-preserving replay with a
repair patch, using verified outcome flips as evidence. This is the criterion the audit proposed for
causal validation of judge labels. It is **demoted from contribution to method** and cited.

**TraceToChain (arXiv:2604.24579)** is the nearest reliability model to our survival layer: it fits
absorbing DTMCs to traces and reconciles pass@k and pass^k as projections of one success-time
distribution. Verified as **not** overlapping: it estimates first-passage/absorption distributions
rather than a hazard, uses only a binary success/fatal absorber pair rather than cause-specific
competing risks, indexes by raw step rather than normalized position, includes no covariates or
random effects, and does not model recovery. Its own stated next step — applying the method to real
SWE-bench and τ-bench trajectories — is still open.

**CVT-RL (arXiv:2606.05263)** supplies an intervention-validity gate for rejecting out-of-distribution
counterfactuals, which we adopt. **CausalFlow (arXiv:2605.25338)** and the hierarchical attribution
work (arXiv:2602.23701) occupy per-trace attribution; the latter already uses "irrecoverability" as a
screening heuristic, so we define ours formally as an estimand rather than reusing the term loosely.

---

## 4. What we concede, stated plainly

Listing these in the paper is cheaper than having a reviewer list them for us.

1. The horizon residual quantity is Peng et al.'s, named and motivated before us.
2. Separating transcript from environment state is ChainSWE's, demonstrated before us.
3. Placebo-matched intervention control is Shao et al.'s.
4. Exposure conditioning is Shao et al.'s.
5. Outcome-flip validation of error labels is REFLECT's.
6. The intervention algebra, contrastive estimator, point-of-commitment rule and Shapley credit are
   CAR's.
7. Self-conditioning as a named mechanism is Sinha et al.'s; the error taxonomy and H\* are HORIZON's.

What remains after all seven concessions is the conjunction in §1, and specifically: **no prior work
estimates population-level mechanism shares of the horizon gap, position-resolved and
error-type-stratified, from within-trajectory counterfactual replay in a stateful tool environment.**
Peng et al. state in print that this has not been done and should be.

---

## 5. Anticipated attacks

**"This is ChainSWE at step granularity."** Partly, and we say so. The additions are an estimand with
CIs instead of arm-level accuracy deltas, a third mechanism (irrecoverability) with no counterpart in
their design, position-resolved hazard and propagation, and open-weight scaling. The decisive point:
their transcript effect is 0.4 points, Sinha's is large, and only a step-granularity design can tell
us which regime real agents are in.

**"This is Sinha with tools."** The environment is not a setting detail. In Sinha's design an error
can only contaminate the transcript; with side-effecting tools it can also contaminate the world, and
the two channels are separable only by intervention. If θ_self turns out large, we have extended
Sinha into a regime where ChainSWE predicts it should vanish; if small, we have bounded his result.
Either is informative.

**"You are just measuring Peng's residual."** The residual is the setup, not the result. Their paper
says so.

**"This is repackaged survival analysis."** The hazard model is the descriptive layer and is not
claimed as a contribution. The contribution is the shares. TraceToChain shows what a genuine
reliability-modelling paper in this space looks like, and ours is not that.

**"Your judge labels are subjective."** Two of three label tiers need no judge. The third is
human-validated and additionally validated causally by outcome flip. If judge–human κ is inadequate,
the taxonomy-dependent analysis is restricted rather than the standard lowered — the interventions do
not depend on judge labels.

**"Your surprising result was predicted."** It was predicted in both directions, by Sinha and by
ChainSWE, which is why it is worth measuring.

**"Open-weight only, 4B–30B."** Stated as a scope limit, not hidden. Shao et al.'s sign reversal
across model generations is cited as direct evidence that extrapolation beyond the tested range is
unwarranted.
