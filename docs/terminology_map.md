# Terminology map

Phase 0 deliverable. Fixes the vocabulary before any writing, per prompt §21 (no terminology drift)
and audit §36. Every term is either **ours to define**, **borrowed and cited**, or **banned**.

---

## 1. Banned terms

| Term | Why banned |
|---|---|
| **failure geometry** | Metaphor, not an object. No established meaning in agent evaluation, sequential decision making, or reliability engineering. The nearest established use — failure regions in *input space* for DRL testing — invites exactly the wrong reading for a claim about *position within a trajectory*. Audit §8. Retained only in the filename of the audit itself. |
| **reliability collapse** | Not a measured quantity. Permissible as informal description of a region where h(u) rises and Δ(s) is strongly negative; never as a defined term or a headline. Audit §9. |
| **Agent Stability Index / ASI** | Any scalar collapses onto pass^k, H_50, RDC slope, GDS, or a survival functional — all of which already exist. A new composite adds an arbitrary weighting and hides rare-but-catastrophic vs frequent-but-recoverable. Audit §10. |
| **early warning / failure prediction** | Off-thesis and crowded. Permitted only as a single appendix validation that intervention-identified pivotal steps coincide with a prefix predictor's risk jump. Audit §15. |
| "nobody has studied X" | False for every X in this area. Prompt §3. |

---

## 2. Core quantities — borrowed, cite on first use

| Our term | Definition | Prior owner | Note |
|---|---|---|---|
| **horizon gap** Δ(s) | Δ(s) = log P_obs(success \| s) − Σ_{i≤s} log p̂_i. Δ<0 with CI excluding 0 = excess horizon degradation. | **Peng et al., arXiv:2607.27283** call this the **horizon residual** Γ_H = log(P_expected/P_observed), P_expected = Π q_i. | **Δ = −Γ_H.** Opposite sign convention: our excess degradation is Δ<0, theirs is Γ_H>0. Must be stated explicitly on first use or it reads as reinvention. We keep "horizon gap" per prompt §21; we do **not** claim the quantity. |
| **independence null** P_ind(s) | P_ind(s) = Π_{i≤s} p̂_i, with p̂_i estimated on the same model and harness with the subtask run in isolation. | Peng et al. (product baseline); Khanal et al. (geometric p^T baseline). | Ours is the matched-subtask version, not a single per-step p. |
| **intrinsic horizon** H* | Minimum number of effective actions required by an optimal policy to complete the task. | **HORIZON, arXiv:2604.11978**, verbatim. | Borrowed as-is. Cite. |
| **normalized position** u | u = t / H*. | Position normalization by H* is ours; H* is HORIZON's. | The denominator is borrowed, the use as a regression covariate is ours. |
| **reliability horizon** H_τ | H_τ = max{t : P(success \| horizon t) ≥ τ}. | METR (arXiv:2503.14499) in time; Sinha et al. in steps; HORIZON's "breaking region" as a band. | **Report as a descriptive covariate only. Never claim as a contribution.** Audit §14. |
| **trajectory-induced degradation** | Accumulated execution — transcript, tool outputs, environment state, earlier errors — makes later work harder. | **Peng et al.**, verbatim. | Useful umbrella for our three mechanisms. Cite. |
| **context rot** | The narrower case where the growing *visible text* is the source of decline. | Peng et al.; Hong et al. (Chroma). | Not one of our mechanisms; a special case of self-conditioning's channel. |
| **exposure** | The point at which the capability under test first becomes exercisable; success reported conditional on reaching it. | **Shao et al., arXiv:2608.20563**, verbatim. | **Adopted as a required conditioning variable** (change D1). Cite. |
| **point of commitment** | The latest step whose contrastive effect CI still excludes zero. | **CAR, arXiv:2606.08275**. | Adopted as estimator hygiene. Cite. |

---

## 3. Core quantities — ours to define

| Term | Definition | Guard |
|---|---|---|
| **positional hazard** h(u) | P(first unrecovered critical error at u \| none before u). Discrete-time, competing risks (recovered vs terminal), cloglog link, natural spline in u (3 df), scenario random intercept. | Nearest prior work is TraceToChain (arXiv:2604.24579), which fits absorbing DTMCs and reports first-passage distributions. Ours differs by position normalization, covariates and competing risks — **verify its full text before claiming novelty**. |
| **survival** S(u) | Survival to first critical error; competing-risk cumulative incidence for recovered vs terminal. | Standard method, new object. Not a contribution on its own. |
| **recovery function** R(e,u) | P(recovered \| error of type e at position u), assigned by **state checks**, not by judge. | Distinguish "recoverable error" from "recovering agent" — that separation is secondary RQ-B. |
| **propagation probability** π(e,u) | π = P(terminal \| τ_{<t}, do(a_t := a_t^err)) − P(terminal \| τ_{<t}, do(a_t := a_t^fix)). | CAR's contrastive estimator used as a **population** estimator stratified by (category, u). The estimator is borrowed; the stratified population use is ours. |
| **mechanism shares** θ_self, θ_state, θ_irrecov, θ_resid | θ_k = (Δ_before − Δ_after,k) / Δ, with bootstrap CIs, sequential-order sensitivity, and Shapley or full factorial where budget permits. | **This is the paper's headline quantity.** No prior work computes shares of the horizon gap — Peng et al. explicitly state Γ_H "does not identify" mechanism. |
| **interventional decomposition** | The contribution: estimating θ by counterfactual replay under a controlled horizon sweep. | Not "causal decomposition" (too broad), not "attribution" (taken by CAR/CausalFlow/REFLECT for per-trace localization). |

---

## 4. The three mechanisms

Named by the intervention that identifies them, never by the label that describes them.

| Mechanism | Intervention | Held fixed | Identifies |
|---|---|---|---|
| **self-conditioning** | history scrub, `do(H_t := H_t^clean)` | environment state factual | effect of the agent's own erroneous history on its future actions |
| **state staleness** | state refresh, oracle-derived state summary injected | history factual | effect of a stale or incorrect belief about environment state |
| **irrecoverability** | oracle fix, `do(A_t := A_t^fix)` | everything before t | downstream propagation of the specific error |

Term provenance: **self-conditioning** is Sinha et al.'s (arXiv:2509.09677) — borrowed, cited, and
extended from their no-tool setting to stateful tool use. **Irrecoverability** already appears as a
screening heuristic ("irrecoverability tie-break", arXiv:2602.23701) — we must define ours formally
as an estimand rather than reuse it as a rule. **State staleness** is not established terminology;
ChainSWE's repository-state accumulation and Peng et al.'s "state contamination" are the nearest
neighbours.

---

## 5. Composition vocabulary — a distinction the audit blurs

HORIZON defines **compositional depth** s = max_p |decision_nodes(p)|, i.e. nested sub-goals and
conditional branches, and separately defines two construction operators:

- **Depth Extension** — adding intermediate steps;
- **Breadth Extension** — combining k independent baseline tasks.

Our s chains subtasks 1..s with shared state. That is **Breadth Extension with a shared state
dependency**, *not* HORIZON's compositional depth. Using "compositional depth s" for our construction
would misattribute the definition and invite a correctness objection.

**Ruling:** call ours the **horizon level s**, define it as the number of chained atomic subtasks
sharing environment state, and state its relation to HORIZON's operators explicitly.

---

## 6. Error labels

Label = **(HORIZON category, detectability class, position u, recoverability outcome)**.

- **HORIZON categories** (borrowed, all seven): environment error, instruction error, catastrophic
  forgetting, false assumption, planning error, history error accumulation, memory limitation.
- **Detectability** (ours): rule-detectable · state-checkable · judgement-required.
- **Recoverability** (ours): recovered · persisted · terminal — assigned by state checks, never by judge.

**No ninth taxonomy.** Audit §12.

Note the two HORIZON categories our H7 turns on, verified verbatim:
- *Catastrophic forgetting* — "the constraint is still present in the context but not attended to
  during later reasoning"; "the constraint remains in the interaction history but is overlooked
  during later planning".
- *Memory limitation* — information lost because "the interaction history exceeds the context
  window".

HORIZON therefore already distinguishes forgetting-despite-retention from genuine overflow. H7 tests
whether the first category is better explained as self-conditioning. Because HORIZON's own definition
concedes the constraint is present, this is a well-posed reinterpretation, not a strawman.

---

## 7. Claim tags

Carried through all research notes per prompt §20: `[LIT]` `[INF]` `[HYP]` `[REC]` `[SPEC]`.
Translate to normal prose in the manuscript but preserve the epistemic distinction. Never promote
`[HYP]`→`[LIT]`, `[INF]`→fact, `[SPEC]`→conclusion, or `[REC]`→evidence.
