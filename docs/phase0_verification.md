# Phase 0 — Novelty Resolution: Verification Log and Gate Assessment

**Date:** 6 September 2026.
**Executor knowledge cutoff:** May 2026. The audit is dated 24 August 2026 and cites work from
arXiv 2602–2608. Everything after the cutoff was verified against the primary source; nothing
here rests on recall.
**Method:** (a) direct verification of every CRITICAL/HIGH threat in audit §3; (b) adversarial
search designed to *find* a paper that already owns the interventional decomposition; (c) primary
full-text inspection for any paper that could change the contribution, per prompt §25 ("never
declare novelty from search snippets alone").

**Headline:** the audit's threat table is **materially incomplete**. Five relevant papers are
missing, one of which names the audit's central quantity and one of which already separates
history from environment state. The gap survives, but it is **narrower than the audit claims** and
the contribution must be restated. Gate verdict in §6 below: **CONDITIONAL PASS**.

---

## 1. Verification of the audit's own citations

| Audit claim | Verified? | Correction required |
|---|---|---|
| HORIZON, arXiv:2604.11978, Wang et al., COLM 2026 | **Real.** Submitted 13 Apr 2026, cs.AI. Xinyu Jessica Wang, Haoyue Bai, Yiyou Sun, Haorui Wang, Shuibai Zhang, Wenjie Hu, Mya Schroder, Bilge Mutlu, Dawn Song, Robert D. Nowak (UW–Madison / UC Berkeley / Georgia Tech). 3,100+ trajectories, 4 domains, GPT-5 variants + Claude-4. κ=0.61 inter-annotator, κ=0.84 human–judge. Leaderboard at xwang2775.github.io/horizon-leaderboard. | **"COLM 2026" venue is unverified.** arXiv listing shows cs.AI submission only. Do not assert the venue until checked. Author list in audit ("Wang, Bai, …, Song, Nowak") is correct but abbreviated. |
| Khanal et al., arXiv:2603.29231 | **Real.** Aaditya Khanal, Yangyang Tao, Junxiu Zhou. 23pp, 4 figs, cs.AI. 10 models, 23,392 episodes, 396 tasks, 4 duration buckets, 3 domains. RDC / VAF / GDS / MOP. | Audit says "10 open-weight"; source says "10 open-source". Findings the audit omits: decay is **domain-stratified** (SE GDS 0.90→0.44; document processing 0.74→0.71), and **VAF bifurcates by capability tier — high VAF is a capability signature, not an instability signal**. The second point matters: it warns against reading variance as unreliability. |
| Sinha et al., arXiv:2509.09677 | **Real.** Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, Jonas Geiping. Submitted 11 Sep 2025, v3 13 Mar 2026. | Audit says "ICLR 2026 submission"; it is **published at ICLR 2026**. Confirmed: pure in-context execution, **no tools, no environment**, plans given. Confirmed: "thinking mitigates self-conditioning". **Confirmed absent: any share or fraction of degradation attributed to self-conditioning.** Our θ_self is therefore not a restatement. |
| Causal Agent Replay, arXiv:2606.08275 | **Real.** Jaineet Shah (CMU), 6 Jun 2026, sole author. | Intervention algebra confirmed and richer than the audit records: `do_resample`, `do_action`, `do_observation`, `do_context`, `do_policy`. Point-of-commitment rule = "latest step whose effect CI still excludes zero". **Validation is synthetic SCMs only — no real agent benchmark, not even Who&When.** Explicit limitation: *"Real tools with side effects are out of scope (demonstrations use mocked, reproducible tools)."* This is the single most useful sentence in the audit's reference list for us. |
| Lu et al., arXiv:2509.25370 | Real (title and ID confirmed in search index). | Full-text verification still outstanding — see §5. |
| CausalFlow, arXiv:2605.25338 | Real (ID confirmed). | Full-text verification outstanding. |
| Rabanser et al., arXiv:2602.16666 | Not yet re-verified. | Outstanding. Low consequence: we cite, we do not claim. |
| METR, τ-bench, Lost-in-the-Middle, MAST, Who&When | Pre-cutoff, known. | No action. |
| "Failure geometry has no established meaning" (audit §8) | Consistent with search. | Recommendation to drop the term stands and is adopted. |

---

## 2. Papers the audit missed — found by adversarial search

These are the reason the gate verdict is conditional rather than clean.

### 2.1 Peng et al., "Benchmarking the Residual" — arXiv:2607.27283, 29 July 2026 — **CRITICAL (terminology and credit)**

Chao Peng, Zhiheng Lyu, Peijie Dong, Hande Dong, Qiang Lin. **Position paper, no experiments.**

It defines, and names, the audit's central quantity:

> Γ_H = log(P_expected / P_observed), with P_expected = Π_i q_i, stage-wise Γ_H = Σ_i log(q_i/r_i)

This is the audit's Δ(s) with the **opposite sign convention** (Δ = −Γ_H; excess degradation is
Δ<0 ⟺ Γ_H>0). They call it the **horizon residual**. They also name **trajectory-induced
degradation** = "accumulated execution (the transcript, tool outputs, environment state, or earlier
errors) makes later work harder", with context rot as the narrower text-only case.

Their §8 research agenda is, almost verbatim, our Phase 4:

> "pair each recurring residual pattern with single-factor interventions, history reset or
> compression, state repair, verification, rollback, explicit planning, so that residual patterns
> become testable causal hypotheses"

And their stated limitation is our stated contribution:

> "Γ_H does not identify context degradation, planning failure, state contamination, or recovery;
> it identifies a contrast under one protocol"

**Consequences, all binding:**
1. We **cannot** present the matched-baseline residual as our construction. It is Peng et al.'s
   named quantity. Cite on first use.
2. Terminology: keep the audit's **horizon gap** (prompt §21 forbids drift) but state the synonymy
   and the sign convention explicitly in the paper, or reviewers will read it as reinvention.
3. This is **net positive for positioning**: a published position paper argues the field must
   measure this and says the mechanism experiments have not been done. Our Phase 3 becomes a
   prerequisite the field has asked for; our Phase 4 becomes the answer to a standing call.
4. They flag a genuine methodological threat we must pre-register against: *"two admissible
   checkpoint protocols can in principle yield opposite-signed residuals."* Protocol sensitivity is
   now a required analysis, not an optional one.

### 2.2 ChainSWE — arXiv:2607.02606v2, 1 Sept 2026 — **HIGH (design overlap)**

Qirui Jin, Lingching Tung, Kenan Li, et al. Coding agents on multi-bug maintenance chains.
Three modes that map onto our first two interventions at coarse granularity:

| ChainSWE mode | Repo (env) state | Transcript | Our analogue |
|---|---|---|---|
| **Oracle** | gold patches applied | fresh | state-refresh ∧ history-scrub |
| **Seq** | agent's own prior patches | fresh | factual state, scrubbed history |
| **Seq+Mem** | agent's own prior patches | preserved, incl. dead ends | fully factual |

So **Oracle→Seq isolates environment state** and **Seq→Seq+Mem isolates the transcript**. That is
the self-conditioning / staleness separation, already performed.

Results: per-bug accuracy **58.9% Oracle → 36.5% Seq → 36.9% Seq+Mem** (7 frontier models,
100 chains, 304 bugs, 54 repos, SWE-Edit scaffold). Position analysis: *"Under oracle resets, later
bugs are not harder"*; under accumulation, difficulty compounds with depth (GPT-5.5: 62.5% → 40.5%
→ 28.2% by position).

**What they do not do:** no within-trajectory replay (position breakdowns are separate full
re-runs, not branches); **no confidence intervals anywhere**; no causal estimand; no hazard or
survival model; no propagation estimate by error type; no irrecoverability arm; Python/SWE only;
frontier closed models only; chains capped at length 3.

**The finding that matters most to us is their null.** Seq 36.5% vs Seq+Mem 36.9% — preserving the
erroneous transcript adds essentially nothing once the repository already carries the agent's
mistakes. In our vocabulary that is **θ_state large, θ_self ≈ 0**, in a real tool-using setting.

### 2.3 Shao et al., security agents — arXiv:2608.20563, 20 Aug 2026 — **HIGH (method precedent + confound)**

Wei Shao, Chongzhou Fang, Zuxiong Tan, Zequan Liang, Setareh Rafatirad, Avesta Sasan, Houman
Homayoun. Two contributions that bind on us:

1. **Exposure.** They instrument tasks with checkpoints and define *exposure* as reaching the point
   where the capability under test becomes exercisable, then report success **conditioned on
   exposure**. Their result: in the CSR task, **9 of 10 failures occur before exposure** — the agent
   never acquired the state it was later supposed to reuse. Their argument: *"a run that never
   discovers a piece of state provides no evidence of a memory failure."*
2. **Placebo-matched intervention.** Baseline / Rescue / **Placebo**, where the placebo is matched
   on byte and line count. Population effects with McNemar tests: C3 reach 95.4% vs 65.5%,
   Δ=29.9pp, p=2.6×10⁻⁶, discordant-pair CI [75.0%, 98.0%].

**Consequences, both binding:**
- **Exposure conditioning is now mandatory in our design.** Without it, a failure where the agent
  never observed the relevant state gets scored as staleness or self-conditioning. This is a real
  identification threat to θ_state and to the "forgetting is not forgetting" hypothesis, and it is
  cheap to fix: our AppWorld checkpoints already give us the instrumentation.
- The **placebo control is not our idea.** Audit §32's simulated R1 demanded it as a novelty-bearing
  addition; it is prior art. Adopt and cite.
- Warning on generality: their intervention effect **reverses sign** between Gemini 2.5 Flash and
  Gemini 3.7 Flash (+29.9pp → −19.8pp). Mechanism shares may not transfer across model generations.
  Our claims must be scoped accordingly.

### 2.4 REFLECT — arXiv:2606.09071, 8 June 2026 — **MEDIUM (supporting contribution overlap)**

Xiaofeng Lin et al. Intervention-supported error attribution for silent failures. Prefix-preserving
replay from a rollback point with a diagnosis-specific repair patch; **uses verified outcome flips
as attribution evidence**; per-trace localization, not population estimation. WTQ/GAIA/BBM/SWE-bench,
GPT-5.2 and Claude Opus 4.6, 435 traces.

**Consequence:** the audit's fourth contribution ("replay-validated failure labels", prompt §6
Supporting) is **substantially anticipated**. Our "a judge label is causally valid iff oracle-fix
replay flips the outcome" is REFLECT's verification criterion. Demote from contribution to method,
and cite.

### 2.5 Others, lower threat but must be cited

- **Markov Chain Reliability / TraceToChain — arXiv:2604.24579**, Tran-Truong & Le, 27 Apr 2026.
  Fits absorbing DTMCs to traces; reports **first-passage distributions**; reconciles pass@k and
  pass^k as "projections of one success-time distribution". Nearest existing work to our survival
  layer. No hazard-with-covariates, no competing risks, no position normalization, no intervention.
  **MEDIUM threat to the secondary contribution**, not the primary.
- **Causal Memory Intervention (CMI) — arXiv:2605.17641.** Swaps memories under controlled
  intervention to estimate causal utility per memory type (+0.307 useful, −0.009 irrelevant,
  −0.033 harmful). Establishes intervention-on-what-the-agent-conditions-on in long-horizon agents,
  but as a *selection method*, not a decomposition of degradation. MEDIUM.
- **CVT-RL — arXiv:2606.05263.** Policy-conditioned counterfactual credit; estimands for deletion,
  semantic substitution, evidence substitution, tool-output perturbation, with intervention-validity
  gating for OOD counterfactuals. Relevant to estimator design; the validity gate is worth adopting.
- Hierarchical failure attribution (arXiv:2602.23701) carries a **"deviation-aware reversibility
  filter"** and an **"irrecoverability tie-break"** — our third mechanism as a heuristic screening
  rule in multi-agent attribution. LOW as prior art, but the term is taken; define ours formally.

---

## 3. The conflict that should drive the paper

Verification turned up a genuine, unresolved empirical disagreement that the audit does not record:

- **Sinha et al. (ICLR 2026):** models become measurably more error-prone when their own prior
  errors are in context. No tools, no environment state, plans supplied.
- **ChainSWE (Sept 2026):** in real coding agents, removing the erroneous transcript while holding
  repository state fixed changes per-bug accuracy by **0.4 points** (36.9 → 36.5). Self-conditioning
  looks nearly absent once a real environment carries the damage.

Both cannot be the general case. The reconciling hypotheses are testable and cheap to state:

- **H-A:** self-conditioning is real but *dominated* whenever the environment can absorb the error —
  it shows up only for errors that leave no state trace (read-only, reasoning, planning errors).
- **H-B:** ChainSWE's granularity hides it — the transcript is reset at *bug* boundaries, which for
  a length-3 chain is a handful of resets; within-bug self-conditioning is untouched by their design
  and would only be visible at step granularity.
- **H-C:** it is position-dependent — θ_self rises with u, and chain-position-1 dominates their average.

This is a better novelty story than the audit's "nobody has decomposed the gap", because it is
falsifiable, non-obvious, already contested in print, and answerable precisely by our design
(step-granularity replay, position-resolved, error-type-stratified, in a stateful tool environment).
It also directly serves prompt §19's "your surprising result was predicted by prior literature"
attack: the prediction exists, it is contradictory, and resolving it is the contribution.

**This does not become the research question.** The primary RQ and estimands are unchanged
(prompt §1, §6). It sharpens the motivation and supplies a pre-registered directional hypothesis
that can fail.

---

## 4. Binding design changes forced by verification

Each is a change to *method*, not to the research question, estimand, benchmark, model set or
contribution. Under prompt §5 none is a SCOPE CHANGE; all are recorded here as research-control
events.

| # | Change | Forced by | Effect on the design |
|---|---|---|---|
| D1 | Condition all mechanism estimates on **exposure**; report success-before / success-after exposure separately | Shao et al. §2.3 | Removes a confound that would inflate θ_state and θ_self. Uses existing AppWorld checkpoints. |
| D2 | **Placebo-summary control** cited as prior art, not offered as novelty | Shao et al. §2.3 | Same experiment, honest credit. |
| D3 | Pre-register **checkpoint/stage protocol** and run a protocol-sensitivity analysis on Δ(s) | Peng et al. §2.1 | Guards the "opposite-signed residuals" failure. New required robustness row. |
| D4 | Cite **horizon residual** on first use of the horizon gap; state sign convention | Peng et al. §2.1 | Terminology only. |
| D5 | Demote replay-validated labels from contribution to method | REFLECT §2.4 | Contribution stack loses its fourth item. |
| D6 | Add a pre-registered directional hypothesis on the Sinha/ChainSWE conflict, with H-A/H-B/H-C | §3 | Gives the paper a result that can be wrong. |
| D7 | Adopt CAR's **point-of-commitment** rule and CVT-RL's **intervention-validity gate** | CAR, CVT-RL | Estimator hygiene; both already cited by the audit's method plan. |

---

## 5. Verification still outstanding

Not blocking the gate, but required before the differentiation paragraph is final:

1. HORIZON full text — the exact wording of its **catastrophic-forgetting** definition. The audit's
   headline hypothesis (§16: "the constraint is still in context, so it is a self-conditioning claim,
   not a memory claim") rests on that wording and is currently second-hand.
2. HORIZON's venue (COLM 2026 claim unverified).
3. AppWorld — determinism, reset cost, checkpoint granularity, and whether reference solutions give
   a usable H*. **This is the Phase 1 gate and the highest-risk unverified item.**
4. Lu et al. (2509.25370) and CausalFlow (2605.25338) full text.
5. Rabanser et al. (2602.16666).
6. Whether any paper does **competing-risk hazard on agent trajectories** — 2604.24579 is the
   nearest and needs full-text reading before we claim the survival layer is new.

---

## 6. Gate verdict

**Status: CONDITIONAL PASS.**

**Evidence.** No single paper performs within-trajectory, step-indexed counterfactual replay in a
stateful tool-using environment to estimate population-level mechanism shares of the horizon gap,
position-resolved and error-type-stratified, with uncertainty intervals. Verified individually:
Peng et al. define the quantity but run no experiments and explicitly call for these interventions;
ChainSWE separates history from state but only between whole arms, at bug granularity, without CIs
or a causal estimand; Shao et al. run matched interventions with placebo but on 3 Gemini models,
without a horizon sweep or any decomposition; CAR supplies the algebra but validates only on
synthetic SCMs and excludes real tools with side effects; Sinha et al. establish the mechanism
without tools, environment, or shares; REFLECT localizes per trace rather than estimating a
population effect.

**Why conditional, not clean.** The audit asserted a wider gap than exists. Five relevant papers
were missing from its threat table, two of them (Peng, ChainSWE) touching the central construction
directly. The surviving contribution is a **synthesis-plus-population-estimation** claim, which is
defensible but thinner than "nobody has done this". Audit §18 Kill condition E is **not** triggered
— no superficially renamed replication is on offer — but the differentiation paragraph must now be
written against seven papers, not five, and must concede §4's D1–D5 openly.

**Deviations from protocol.** The audit's Phase 0 task list said "read the five closest papers".
The closest-paper set is now **seven**: HORIZON, Khanal, Sinha, Lu, CAR + **Peng et al.** and
**ChainSWE**. Shao et al. is an eighth for method, not for positioning.

**Scientific implication.** The project remains worth running, and its motivation is stronger and
more concrete than the audit's: there is a live contradiction between Sinha and ChainSWE about
whether self-conditioning survives in stateful tool-using agents, and our design is the instrument
that settles it. Correspondingly, the *expected* result should be treated as genuinely open —
ChainSWE's null is real evidence that θ_self may be small, and prompt §17 requires the project to
remain valid in that case. It does: a well-estimated θ_self ≈ 0 with a tight CI, alongside a large
θ_state, is a publishable resolution of a printed disagreement.

**Next permitted phase.** Phase 0 is not yet closed — items in §5 remain, and the acceptance
criterion (two independent readers agree the differentiation is precise) is unmet. Permitted next
actions: finish §5 verification, write the differentiation paragraph and preregistration.
**Phase 1 is not authorised**, and independently of novelty it is blocked on compute — see
`docs/research_state.json` and `tasks/todo.md`.
