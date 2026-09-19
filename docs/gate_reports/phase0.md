# Gate Report — Phase 0 (Novelty Resolution)

**Date:** 6 September 2026. **Format:** prompt §32.

---

## Status

**CONDITIONAL PASS.**

Conditional on two things, both recorded rather than waived: the acceptance criterion as written
cannot be met (§3), and the surviving novelty is materially narrower than the audit asserts (§2).

---

## Evidence

**Verification performed.** Every CRITICAL and HIGH threat in audit §3 was checked against its
primary source, not against recall — necessary because the audit postdates this executor's knowledge
cutoff by three months and cites arXiv 2602–2608. Full-text inspection was performed for the seven
papers capable of changing the contribution.

**All audit citations are real.** No fabricated references. Corrections required: Sinha et al. is
*published* at ICLR 2026, not a submission; HORIZON's COLM 2026 status is a submission per its own
ethics statement, not confirmed acceptance; Khanal's models are "open-source" not "open-weight" in
the source's own wording.

**Five relevant papers were missing from the audit's threat table**, two of them touching the central
construction:

| Paper | What it does that the audit did not account for |
|---|---|
| Peng et al., 2607.27283 (29 Jul 2026) | Defines and **names** the audit's central quantity: horizon residual Γ_H = log(P_expected/P_observed), P_expected = Π q_i. Our Δ(s) = −Γ_H. Position paper, no experiments. |
| ChainSWE, 2607.02606 (v2, 1 Sep 2026) | **Already separates transcript from environment state** via Oracle/Seq/Seq+Mem. Reports 58.9% → 36.5% → 36.9%. |
| Shao et al., 2608.20563 (20 Aug 2026) | Exposure conditioning; placebo-matched intervention with population CIs. |
| REFLECT, 2606.09071 (8 Jun 2026) | Outcome-flip validation of error labels — our stated supporting contribution. |
| TraceToChain, 2604.24579 (27 Apr 2026) | Nearest reliability model to our survival layer. |

**The gap survives, and was checked rather than assumed.** No single paper estimates population-level
mechanism shares of the horizon gap, position-resolved and error-type-stratified, from
within-trajectory counterfactual replay in a stateful tool environment. Verified individually:

- **Peng et al.** run no experiments and state that Γ_H *"does not identify context degradation,
  planning failure, state contamination, or recovery"*; their §8 agenda calls for exactly our
  interventions.
- **ChainSWE** compares whole arms at bug granularity with **no confidence intervals anywhere**, no
  causal estimand, no hazard model, no propagation by error type, no irrecoverability arm.
- **CAR** supplies the intervention algebra but validates only on synthetic SCMs and states that
  *"real tools with side effects are out of scope."*
- **Sinha et al.** establish self-conditioning with no tools, no environment state, and **no share
  quantified**.
- **TraceToChain** estimates first-passage distributions, not hazards; binary absorbers, not
  cause-specific competing risks; raw step index, not normalized position; no covariates, no random
  effects, no recovery model.
- **REFLECT** localizes per trace rather than estimating population effects.
- **HORIZON** attributes per trace by label, with no counterfactual and no independence null.

**Audit Kill condition E (novelty substantially duplicated) is NOT triggered.**

**Infrastructure verification (bearing on Phase 1).** AppWorld exposes `world.save_state()` /
`world.load_state(state_id)` — the replay-from-step primitive the design requires — and reloads tasks
in <0.5 s from stored database diffs. Reference solutions (`.compiled_solution_module`, `.api_calls`,
`.evaluation_code`) are available on **train and dev splits only**. Evaluation is final-state unit
tests, so stage checkpoints must be constructed by decomposing `evaluation_code`. Determinism is
seeded but **not documented as guaranteed** and must be measured.

**Feasibility verification.** Qwen3-8B scores **5.4% TGC** on AppWorld at fp16 and **3.0%** at 4-bit,
measured under a 24 GB single-GPU budget. The audit's claim of non-trivial open-weight success is
contradicted at 8B and unverified at 32B.

---

## Deviations

| # | Deviation | Reason |
|---|---|---|
| 1 | Closest-paper set expanded from five to seven | Verification found two papers touching the central construction |
| 2 | D1–D7 binding design changes adopted | Forced by verified prior work; each is a method change, none alters the RQ, estimand, benchmark, intervention set, statistics, or contribution |
| 3 | Model set amended to Qwen3-4B/8B/14B/30B-A3B + Mistral-24B | Qwen3-235B and Llama-3.3-70B are physically impossible on 2×T4; Qwen3-32B costs 4–13 weeks of quota. User-approved 6 Sep 2026. Touches the model set only. |
| 4 | Phase 3a atomic-feasibility pre-gate inserted; kill condition F added | Floor effect would make Δ(s) a log-ratio of near-zero probabilities |
| 5 | H8 added as a pre-registered directional hypothesis | A live contradiction between Sinha and ChainSWE, absent from the audit |

None of these is a SCOPE CHANGE under prompt §5. The research question, the three mechanisms, the
estimands, the causal framework, the primary benchmark, the contribution stack (less its demoted
fourth item) and the target version are unchanged.

---

## Scientific implication

**The project remains worth running, with a better motivation than the audit supplied and a thinner
novelty margin than it claimed.**

Verification surfaced a genuine, unresolved empirical disagreement in print. Sinha et al. show that
models become measurably more error-prone when their own errors are in context — with no tools and
no environment. ChainSWE shows that removing the erroneous transcript while holding repository state
fixed moves per-bug accuracy by **0.4 points** — with real tools, real state, and frontier models.
Both cannot be the general case.

This changes the paper's centre of gravity for the better. The question is no longer "has anyone
decomposed the gap" — a question whose answer keeps shrinking as the field publishes — but "does
self-conditioning survive in stateful tool-using agents, and at what granularity and position." That
is falsifiable, contested by two strong papers pointing opposite ways, and answerable precisely by
the design already specified.

It also means **H3 (θ_self > 0) may well fail**, and this is now recorded in the preregistration
before any data exists so it cannot be reframed later. Per prompt §17 the project must remain valid
if self-conditioning is weak; it does. A tightly estimated θ_self ≈ 0 beside a large θ_state would
resolve a printed disagreement and confirm ChainSWE's null at a granularity that paper could not
reach. That is a publishable result, not a failed experiment.

Two risks are now quantified rather than assumed. The floor effect is real and is the most likely
cause of an uninformative null; it is addressed by a cheap pre-gate that costs 2–4 GPU-hours and
gates the largest expenditure in the programme. And a preliminary signal has been recorded *before
running anything*: AppWorld's own GPT-4 numbers (48.7% TGC vs 21.0% SGC) are, on a crude independence
calculation, **better** than independent composition would predict — a directional hint against H1
that must be checked honestly at Phase 3 rather than discovered and rationalised later.

---

## Next permitted phase

**Phase 0 is not yet closed.** Remaining: full-text verification of Lu et al. (2509.25370),
CausalFlow (2605.25338) and Rabanser et al. (2602.16666) — all LOW severity, none capable of changing
the contribution; and confirmation of HORIZON's venue before it is cited as COLM 2026.

**Acceptance criterion cannot be met as written.** The audit requires two independent readers to
agree the differentiation is precise for each closest paper. There is one executor. Mitigation
applied rather than criterion waived: the differentiation is written against seven papers instead of
five, every borrowed component is conceded explicitly in `docs/differentiation.md` §4, and the
anticipated attacks are answered in §5. **A human reader should review `docs/differentiation.md`
before Phase 3 spend is committed.** This is the honest residual risk of the gate.

**Permitted next:** Phase 1 (infrastructure). Its acceptance criterion — 100% replay fidelity on 20
trajectories — is now testable, since `save_state`/`load_state` exist and reset is fast. Determinism
remains the single highest-risk technical assumption and can end the programme under kill condition C
regardless of anything above.

**Not permitted:** any composed-horizon run before Phase 3a returns a p̂_i distribution; any causal
claim before Phase 1 replay fidelity passes.
