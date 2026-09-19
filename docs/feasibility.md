# Feasibility: compute envelope and the floor-effect problem

**Date:** 6 September 2026. **Status:** blocks Phase 3 planning, not Phase 0/1/2.
**Available compute (confirmed with user):** Kaggle free tier. Local machine has **no GPU**
(`torch 2.13.0+cpu`, no `nvidia-smi`), so all model execution goes to Kaggle.

Two independent problems. The second is the serious one.

---

## 1. Compute envelope

Kaggle free tier: 2× Tesla T4 (16 GB each, 32 GB total), ~30 GPU-hours/week, ~9 h maximum session.
Turing architecture — fp16 tensor cores, **no bf16, no FlashAttention-2, no Marlin kernels** (all
need Ampere+), and **no NVLink**, so tensor parallelism across the two cards is PCIe-bound.

Weights at 4-bit AWQ, against a 32 GB ceiling:

| Model | ~4-bit weights | Fits? | Verdict |
|---|---|---|---|
| Qwen3-4B | ~2.8 GB | 1×T4 | comfortable |
| Qwen3-8B | ~5.5 GB | 1×T4 | comfortable |
| Qwen3-14B | ~8.5 GB | 1×T4 | comfortable |
| Qwen3-30B-A3B (MoE) | ~16 GB | TP=2 | feasible; only 3B active → decode stays fast |
| **Qwen3-32B (dense)** | ~19–20 GB | TP=2, ~10 GB left for KV | **feasible but PCIe-bound; 4–8× slower than 8B** |
| Qwen3-235B-A22B | ~118 GB | — | **impossible** |
| Llama-3.3-70B | ~40 GB | — | **impossible** |
| Mistral-Small-3.2-24B | ~13 GB | TP=2 | feasible |

**MVE cost estimate, Qwen3-8B (audit §22: 540 composed + 270 atomic + ~1,500 replay continuations).**
Assuming vLLM continuous batching (~16 concurrent trajectories) and automatic prefix caching, which
matters enormously in an agent loop because each step otherwise re-prefills the whole history:

- ~40,900 total steps; ~6.2 M output tokens; ~41 M prefill tokens (incremental, with APC)
- decode at ~200 tok/s aggregate → ~8.6 h; prefill at ~1–2 k tok/s → ~6–11 h
- **≈ 15–25 GPU-hours pure inference; 30–50 h wall clock with AppWorld env overhead**

That fits in roughly **1.5–2 weeks** of Kaggle quota. Qwen3-32B at TP=2 over PCIe lands at
**120–400 hours — 4 to 13 weeks**, which is not a sensible use of the budget.

**Consequence:** the audit's model plan (§19: Qwen3-8B/32B/235B primary, Llama-3.3-70B and
Mistral-24B cross-family) is **not executable on this hardware**. Three of the five named models are
impossible or impractical.

---

## 2. The floor-effect problem — the one that actually threatens the design

Verified: **Qwen3-8B scores 5.4% Task Goal Completion on AppWorld at fp16 (12 K truncated context)
and 3.0% at 4-bit AWQ (32 K context)**, rising to 8.9% only with added scaffolding
(arXiv:2604.11465, evaluated under a 24 GB single-GPU budget — nearly our exact setting). Reference
points on the same split: GPT-4o 48.8%, Claude 3.5 Sonnet 33.2%. The original paper reports GPT-4
at 48.7% TGC / 21.0% SGC.

The audit (§18) asserts AppWorld has "non-trivial [success] for 32B+" open-weight models and lists
floor effects as a disqualifying property for the primary environment. For 8B that assertion is now
**contradicted by measurement**; for 32B it remains **unverified**.

**Why this is fatal to the estimators, not merely inconvenient.** With native AppWorld tasks at
~5% success:

- Δ(s) = log P_obs(s) − log P_ind(s) is a **log-ratio of two near-zero probabilities**. At s≥2 both
  terms approach zero and the estimator is undefined or wildly unstable. No paired bootstrap rescues
  this.
- Positional hazard h(u) degenerates: essentially every trajectory takes its first unrecovered
  critical error almost immediately, so there is no positional variation left to model.
- Replay branches collapse: factual and counterfactual continuations both fail, so
  π̂ = P(fail | err) − P(fail | fix) ≈ 0 − 0. The intervention has no power by construction, and a
  null θ would be an artefact of the floor rather than evidence about mechanism.

This would trigger audit Kill condition A (Δ indistinguishable from zero) for a purely instrumental
reason. That is the worst possible failure: a null that means nothing.

### 2.1 Why it is nonetheless probably survivable

We do **not** run AppWorld's native tasks. Per audit §25 the horizon sweep is *constructed*: atomic
subtasks are run in isolation to estimate p̂_i, then chained into composed tasks at s = 1..3. The 5.4%
figure is for AppWorld's full-length native tasks, which are exactly the long, hard objects our
design decomposes.

The design is therefore feasible **iff** we can select atomic subtasks on which Qwen3-8B achieves a
usable p̂_i. If p̂_i ≈ 0.7, then P_ind(3) ≈ 0.34 and a horizon gap is comfortably estimable; if
p̂_i ≈ 0.2, then P_ind(3) ≈ 0.008 and it is not.

**This converts a benchmark risk into a cheap, decidable measurement.** Atomic runs are short and
are the least expensive thing in the whole programme.

### 2.2 Proposed pre-gate — Phase 3a (new)

Insert before any composed run:

> **Phase 3a — atomic feasibility.** Estimate p̂_i for candidate atomic subtasks on Qwen3-8B,
> 3 seeds. **Accept** if a set of ≥30 subtasks exists with median p̂_i ≥ 0.5 and lower-quartile
> p̂_i ≥ 0.35, such that the implied P_ind(3) ≥ 0.05. **Fail** → the primary environment cannot
> support the estimand at this model scale; escalate the scope decision rather than proceeding.

Cost: ~2–4 GPU-hours. It is the highest-information-per-GPU-hour measurement in the project and it
gates the single largest expenditure.

---

## 3. Proposed model-set amendment — decision required

Under prompt §5 this touches **the model set** and nothing else. It does not alter the research
question, the estimand, the causal interpretation, the benchmark, the interventions, the statistical
plan, the contribution claim, or the scope. It does not invalidate any completed experiment (there
are none). It creates no new overlap with prior work. It is **not** a SCOPE CHANGE under §5, but it
is a deviation from the audit that must be recorded and approved rather than made silently.

**Audit specifies:** Qwen3-8B / 32B / 235B-A22B primary; Llama-3.3-70B + Mistral-Small-3.2-24B
cross-family; Qwen3-32B reasoning on/off.

**Proposed:** Qwen3-4B → 8B → 14B → 30B-A3B as a **four-point within-family scaling axis**, with
Mistral-Small-3.2-24B retained for cross-family and reasoning on/off run on the largest feasible
Qwen3.

Rationale:
1. Every model listed is executable inside the Kaggle envelope; four scale points is a **better**
   axis for "does θ_self shrink with scale" than the audit's three, and scaling is the axis the
   audit cares about (§19).
2. Qwen3-30B-A3B is the audit's **own named fallback** ("the 30B-A3B MoE if budget is tight"), and
   its 3B active parameters make it unusually cheap for its capability — the best available proxy
   for the 32B rung.
3. Llama-3.3-70B and Qwen3-235B are dropped as physically impossible, not as a preference.
4. **Caveat that must be stated in the paper:** the axis spans 4B–30B, so any claim about how
   mechanism shares scale is scoped to small and mid-size open-weight models. The audit's hope of
   observing a share trend that continues to frontier scale is not testable on this hardware.

**Risk this does not remove:** if §2.2's pre-gate fails at 8B, larger models may be needed precisely
where the hardware runs out. In that case the honest options are (a) a hosted API for the larger
models (audit §23 budgets ~$1–3k API-equivalent for exactly this), (b) τ²-bench as the audit's
designated fallback environment, or (c) reporting the programme as infeasible at this budget. That
decision should be made **after** the pre-gate produces a number, not now.

---

## 4. What this does not change

Phases 0, 1 and 2 are unaffected and remain the correct next actions:

- **Phase 0** (novelty, preregistration) is compute-free and is not finished.
- **Phase 1** (deterministic harness, replay-from-step, checkpoint hooks, rule labeller) is cheap
  and its acceptance criterion — 100% replay fidelity on 20 trajectories — can be tested with the
  smallest model available. AppWorld determinism is **unverified in the primary paper** and is the
  single highest-risk technical assumption in the programme; it can kill the plan for reasons that
  have nothing to do with model scale.
- **Phase 2** (synthetic toolified execution, planted irrecoverability) is deliberately trivial
  environmentally and can run on Qwen3-4B/8B. It validates the estimators against planted ground
  truth before any AppWorld budget is committed.

Per audit §30 the ordering is novelty → infrastructure → estimator validation → horizon gap →
interventions. Nothing here justifies reordering it.
