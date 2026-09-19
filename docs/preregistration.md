# Preregistration (draft v0.1)

**Project:** Interventional decomposition of the horizon gap in long-horizon tool-using LLM agents.
**Date drafted:** 6 September 2026. **Status:** draft. To be frozen and posted before any Phase 3
run, per audit §35. Nothing below may be revised after data collection begins except through a
dated, numbered amendment recorded in §12.

Governing documents: `docs/prompt.txt` (contract), `docs/failure_geometry_research_audit.md` (specification),
`docs/phase0_verification.md` (novelty gate), `docs/differentiation.md`, `docs/feasibility.md`.

---

## 1. Research questions

**Primary.** When a tool-using agent's long-horizon success falls below what its matched
short-horizon competence predicts, how much of the shortfall is caused by self-conditioning on
earlier errors, by stale environment-state beliefs, and by irrecoverable typed errors?

**Secondary A.** Does the hazard of the first unrecovered critical error increase with normalized
position u = t/H\* after controlling for task difficulty, and does that increase flatten under the
interventions?

**Secondary B.** Is recoverability a property of error type × position or of the model — do models
differ in *which* errors they make, or in *whether* the same error becomes terminal?

---

## 2. Design

Matched short-versus-long construction in a stateful tool environment, with counterfactual replay.

| Dimension | Levels |
|---|---|
| Environment | AppWorld (primary). τ²-bench is the designated fallback if Phase 1 replay fidelity fails. |
| Horizon level s | 1, 2, 3 (MVE); s = 4 only if budget permits. s = 1 are atomic subtasks. |
| Model | Qwen3-4B, 8B, 14B, 30B-A3B (within-family scaling axis); Mistral-Small-3.2-24B (cross-family) |
| Reasoning | off (main condition); on/off ablation on the largest feasible Qwen3 |
| Seeds | 3 |
| Intervention | none · history-scrub · state-refresh · oracle-fix |
| Order | half of scenarios run with subtask order reversed |

**Difficulty control.** Subtask difficulty is identical *by construction* between the short and long
conditions: the same atomic subtasks that yield p̂_i are the ones chained to form level s. What
changes between conditions is only shared history and shared environment state. This is the design's
answer to "task difficulty explains the degradation".

**Harness.** One ReAct loop with native tool calling, one system prompt, one tool schema, fixed
truncation and token limits, step limit 3·H\*, zero retries on malformed calls (a malformed call is
a logged event, not a retry). No planner, no memory module, no verifier in the baseline — those are
interventions or ablations, never part of the main condition.

---

## 3. Estimands and identification

Agent loop: environment state S_t, history H_t, action A_t = π(H_t), S_{t+1} = E(S_t, A_t).

| Quantity | Definition | Intervention | Held fixed | Identifies, under assumption |
|---|---|---|---|---|
| Δ(s) | log P_obs(s) − Σ_{i≤s} log p̂_i | none (observational) | subtask identity and difficulty | excess degradation beyond independent errors, **assuming** the atomic estimate p̂_i is the correct independence null for the composed setting |
| θ_self | (Δ − Δ_after,self)/Δ | `do(H_t := H_t^clean)` | environment state factual | effect of erroneous history, **assuming** scrubbing alters behaviour only through the history channel |
| θ_state | (Δ − Δ_after,state)/Δ | oracle state summary injected | history factual | effect of stale state belief, **assuming** the summary alters behaviour only through the belief channel — tested by placebo |
| θ_irrecov | (Δ − Δ_after,fix)/Δ | `do(A_t := A_t^fix)` | everything before t | downstream propagation of the specific error |
| π(e,u) | P(terminal \| do(a_t := a^err)) − P(terminal \| do(a_t := a^fix)) | oracle fix | prefix τ_{<t} | propagation probability by category and position |
| R(e,u) | P(recovered \| error e at u) | none | — | recovery, assigned by state checks not judge |
| h(u) | P(first unrecovered critical error at u \| none before) | none | — | positional hazard, competing risks recovered vs terminal |

**Identification threats explicitly registered:**

1. **The oracle summary may alter policy behaviour beyond the belief channel** (it adds tokens,
   authority, and a fresh framing). Addressed by a **placebo summary** matched on token count and
   surface form but carrying no task-relevant state, following Shao et al. If the placebo moves the
   outcome, θ_state is not identified and is reported as an upper bound only.
2. **Exposure.** All mechanism estimates are conditioned on the agent having reached the point where
   the relevant state was observable. Runs failing before exposure are reported separately and
   excluded from θ_state and θ_self. Without this, discovery failures are misattributed to memory.
3. **Point of commitment.** Resampling a step re-rolls all downstream stochasticity, which produces
   spurious attribution at early steps. We adopt CAR's rule: report the latest step whose contrastive
   effect CI still excludes zero.
4. **Order dependence.** Sequential decomposition is order-dependent. We report a sequential-order
   sensitivity table and, where all 2³ intervention combinations are run, Shapley shares.
5. **Protocol sensitivity.** Peng et al. show two admissible checkpoint protocols can yield
   opposite-signed residuals. The stage/checkpoint protocol is fixed in §4 before data collection and
   its sensitivity is a required reported analysis.
6. **Replay fidelity.** All causal claims are void if replay is not reproducible. See kill condition C.

---

## 4. Checkpoint and stage protocol — fixed in advance

- A **stage** is one atomic subtask as defined by the AppWorld scenario decomposition used to
  generate p̂_i. Stage boundaries are identical in the short and long conditions.
- A **checkpoint** is a programmatic state assertion evaluated at a stage boundary. Checkpoints are
  written once, before Phase 3, and are not edited after seeing outcomes.
- **Critical error** = a state-check failure at a stage boundary, or a judge-flagged planning or
  false-assumption error. The survival event is the **first unrecovered** critical error.
- **Recovered** = a subsequent checkpoint passes. **Persisted** = it does not, but the run continues.
  **Terminal** = the run cannot complete. Assigned by state checks, never by judge.
- **H\*** = minimum effective actions by an optimal policy (HORIZON's definition), taken from the
  reference solution where available. If reference solutions are unavailable, H\* is constructed and
  the construction is reported; an H\* sensitivity analysis becomes mandatory.

---

## 5. Hypotheses

Pre-registered, directional, and falsifiable. Tags per prompt §20.

| ID | Statement | Tag |
|---|---|---|
| H1 | Δ(s) < 0 with CI excluding 0 for ≥1 model at s ≥ 2 | HYP |
| H2 | h(u) increases with u after scenario random effects | HYP |
| H3 | θ_self > 0 with CI excluding 0 | HYP |
| H4 | θ_state > 0 with CI excluding 0 | HYP |
| H5 | π(e,u) ≈ 1 for state-changing tool errors, ≈ 0 for read-only errors, CIs disjoint | HYP |
| H6 | θ_self declines with model scale and under reasoning-on | HYP |
| H7 | Events labelled *catastrophic forgetting* are substantially removed by history scrubbing while the constraint remains in context | HYP |
| H8 | The Sinha/ChainSWE conflict resolves by granularity or position: θ_self is non-zero at step granularity and increases with u, even where stage-level transcript effects are ~0 | HYP |

**H3 is directionally contested in print and may fail.** Sinha et al. support a positive transcript
effect; ChainSWE reports 36.5% vs 36.9% between fresh and preserved transcripts, i.e. approximately
zero. A well-estimated θ_self ≈ 0 with a tight interval, alongside a large θ_state, is a valid and
publishable resolution — not a failed experiment. This is recorded here so it cannot be
retrospectively reframed.

**H7 is the headline non-obvious result if it holds.** It is a reinterpretation of a category whose
own definition (HORIZON, verified verbatim) concedes the constraint is still in context.

---

## 6. Primary tests and multiplicity

Exactly five confirmatory tests. Everything else is exploratory and will be labelled as such.

1. Δ(3) CI excludes 0 — paired bootstrap over scenarios, 2,000 resamples
2. Hazard slope in u positive — cloglog GLMM, natural spline (3 df), scenario random intercept
3. θ_self CI excludes 0
4. θ_state CI excludes 0
5. θ_irrecov CI excludes 0

**Correction:** Holm across these five. **Effect reporting:** risk differences with 95% CIs
throughout; no p-value is reported without its effect size and interval.

**Exploratory, declared now:** change-point analysis of h(u); H_τ under intervention; Shapley shares;
SWE-bench observational replication; cross-model share comparisons; anything responsive to H8.

---

## 7. Analysis plan

- **Survival:** discrete-time competing-risk hazard, complementary log-log link, natural spline in u
  (3 df), fixed effects for model / level s / intervention, random intercept for scenario.
  Cause-specific hazards for recovered vs terminal; cumulative incidence functions. Discrete time is
  chosen so Cox and its tie-handling are unnecessary.
- **Horizon gap:** paired bootstrap over scenarios, 2,000 resamples.
- **Replay effects:** mixed-effects logistic on continuation outcomes, trajectory random effect.
- **Shares:** θ_k = (Δ_before − Δ_after,k)/Δ with bootstrap CIs, sequential-order sensitivity table,
  Shapley where the full factorial is available.
- **Not used:** permutation tests, Cox regression, latent-state or regime-switching models. None is
  justified by the data shape; adding them post hoc would be p-hacking by menu expansion.

**Power.** With 40 scenarios × 3 seeds per level, a hazard ratio of 1.5 per 0.25u is detectable at
approximately 80% power. Smaller effects require the ideal set and will be reported as
underpowered rather than as null.

---

## 8. Exclusions and stopping rules — fixed in advance

Declared now so they cannot be tuned later.

- Runs with infrastructure errors (API failure, container crash, OOM) are excluded and counted in a
  **completion-rate report**. Target ≥95% completion; below that the batch is rerun, not filtered.
- Runs failing before **exposure** are excluded from θ_state and θ_self and reported separately.
- No scenario, seed, model or intervention branch may be dropped after inspecting outcomes.
- The judge is applied only to judgement-required events. If judge–human κ < 0.4 for a category, that
  category is dropped from taxonomy-dependent analysis and the drop is reported. The threshold is not
  moved to retain categories.
- Replay budget: k = 4 continuations, first 2 candidate error events per failed trajectory, s ≥ 2
  only.

---

## 9. Kill conditions

| ID | Condition | Action |
|---|---|---|
| A | Δ(s) not distinguishable from 0 | Independent errors explain the degradation. Report as a negative result with power analysis. Do not manufacture a mechanism story. |
| B | All intervention effects small, CIs including 0 | Mechanisms unidentified. Pivot to positional hazard modelling as the primary contribution. |
| C | Replay not reproducible (>2% nondeterminism) | **Stop all causal analysis.** Fix infrastructure or switch to τ²-bench. Causal claims on nondeterministic replay are not permitted. |
| D | Judge–human κ < 0.4 on judgement-required categories | Restrict to rule and state tiers. Interventions do not depend on judge labels, so the core survives. |
| E | Novelty substantially duplicated | Reformulate or abandon. **Assessed at Phase 0: not triggered**, but the gap is narrower than the audit asserted — see `docs/differentiation.md` §4. |

**Additional kill condition, added by verification (not in the audit):**

| F | Phase 3a floor effect: no set of ≥30 atomic subtasks reaches median p̂_i ≥ 0.5 with implied P_ind(3) ≥ 0.05 | Δ(s) becomes a log-ratio of near-zero probabilities and every downstream estimator loses power for instrumental reasons. Escalate the scope decision — hosted API, τ²-bench, or report infeasible. **Do not run the composed sweep and hope.** |

---

## 10. What would falsify the programme

Recorded explicitly so that a null cannot be reframed as a finding:

- Δ(3) ≈ 0 → the horizon gap does not exist in this setting; the mechanisms have nothing to explain.
- All three θ ≈ 0 with tight intervals → the gap exists but is not attributable to any of the three
  proposed mechanisms; the residual is the finding, and the paper becomes a hazard-modelling paper.
- Placebo summary moves the outcome as much as the oracle summary → θ_state is not identified.
- Replay fidelity below threshold → nothing causal may be claimed at all.

---

## 11. Reproducibility commitments

Release: trajectory schema and full JSONL logs; harness pinned to a vLLM version; model revisions
and hashes; tokenizer versions; chat templates; seed lists; AppWorld version and task-chaining
scripts; replay implementation; judge prompts; human annotations; statistical analysis code; figure
generation code from logs; container spec; this preregistration; and a completion-rate report. Any
result that cannot be regenerated from released logs is treated as a reproducibility failure.

---

## 12. Amendments

Numbered, dated, with reason and the phase at which they were made. Amendments after data collection
begins must state whether they were made with knowledge of outcomes.

| # | Date | Change | Reason | Outcome-aware? |
|---|---|---|---|---|
| — | — | *(none yet)* | — | — |

**Pre-collection deviations from the audit, recorded before any data exists:**

- D1 exposure conditioning added (Shao et al.)
- D2 placebo control cited as prior art, not novelty
- D3 checkpoint-protocol sensitivity analysis added (Peng et al.)
- D4 horizon gap cited as Peng et al.'s horizon residual, sign convention stated
- D5 replay-validated labels demoted from contribution to method (REFLECT)
- D6 H8 added — pre-registered directional hypothesis on the Sinha/ChainSWE conflict
- D7 CAR point-of-commitment rule and CVT-RL intervention-validity gate adopted
- D8 model set amended to Qwen3-4B/8B/14B/30B-A3B + Mistral-24B (hardware; user-approved 6 Sep 2026)
- D9 Phase 3a atomic-feasibility pre-gate and kill condition F added (floor effect)
