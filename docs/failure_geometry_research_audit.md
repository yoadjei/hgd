# Research Audit: "Failure Geometry of Long-Horizon AI Agents"

**Audit date:** 24 August 2026. **Scope:** novelty, methodology, experimental design, positioning.
**Tagging convention used throughout:** `[LIT]` known from literature (cited) · `[INF]` inference from literature · `[HYP]` hypothesis to be tested · `[REC]` recommended design · `[SPEC]` speculation.

---

## 1. Executive Verdict

**Verdict: do not pursue the idea as written. Narrow it hard and re-centre it on an interventional decomposition of horizon-induced degradation. The descriptive core (taxonomy + "reliability vs. horizon" curve + a scalar Agent Stability Index on AgentBench/WebArena/SWE-bench Lite with Llama/Qwen/Mistral) is already published, in some cases twice, in 2026.**

Specifically:

- The "where do long-horizon agents break, and how does failure composition shift with horizon" question — with a 7-category taxonomy, controlled horizon extension, WebArena + AgentBench environments, a validated LLM-judge pipeline, and a "breaking point as transition region" concept — is **HORIZON (Wang et al., arXiv:2604.11978, COLM 2026)**. `[LIT]`
- The "reliability decays super-linearly relative to a geometric p^T baseline; positive inter-step error correlation; meltdown onset; open-weight Llama/Qwen/Mistral; ReAct vs. memory scaffold" study is **Khanal et al., arXiv:2603.29231 (Mar 2026)** — 10 open-weight models, 23k episodes, four duration buckets. `[LIT]`
- The controlled demonstration that per-step accuracy falls with step index *because models condition on their own prior errors* (not merely context length) is **Sinha et al., arXiv:2509.09677 (ICLR 2026 submission)**. `[LIT]`
- A twelve-metric "reliability profile" (consistency, robustness, predictability, safety) replacing single success scores is **Rabanser et al., arXiv:2602.16666 (ICML 2026)**. `[LIT]`
- Error-propagation-centred single-agent taxonomies exist (**AgentErrorTaxonomy / AgentErrorBench, Lu et al., arXiv:2509.25370**); counterfactual step attribution tooling exists (**Causal Agent Replay, arXiv:2606.08275; CausalFlow, arXiv:2605.25338**); early-warning/online auditing exists (**AgentForesight, arXiv:2605.08715; real-time detection, arXiv:2608.02464; PrefixGuard; Trajectory Guard**). `[LIT]`

What none of these do — and what is both feasible under an evaluation-only budget and defensible at ICLR — is **causally decompose the horizon gap into mechanisms by intervention**. Every paper above either (a) describes the curve, (b) labels the failure, or (c) attributes a single failed trace post hoc. None estimates, at population scale and under a controlled horizon sweep, how much of the gap between observed long-horizon success and the independent-error prediction is due to (i) self-conditioning on prior errors, (ii) context/state staleness, or (iii) irrecoverability of specific error types at specific positions — using `do()`-style replays that hold everything else fixed.

That is the paper. The rest of this audit builds it.

---

## 2. Most Important Discovery

Two things, one negative and one positive.

**Negative.** The 2026 literature has closed the descriptive door faster than the prompt assumed. HORIZON and Khanal et al. together occupy the exact benchmark/model/metric cell the proposal targets. Reviewers at ICLR 2027 will know HORIZON (COLM) and Rabanser et al. (ICML). A paper whose headline is "we show failure composition shifts with horizon and propose an index" will be desk-rejected on novelty. `[INF]`

**Positive.** The same literature has a hole that is visible only once you line the papers up:

| Paper | Sweeps horizon? | Types errors? | Localizes step? | Intervenes (do-op)? | Population-scale causal estimates? |
|---|---|---|---|---|---|
| HORIZON | Yes | Yes (7) | Coarse (trace-level) | No | No |
| Khanal et al. | Yes (duration buckets) | No | No (entropy heuristic) | No | No |
| Sinha et al. | Yes (synthetic) | No | Yes (step accuracy) | Yes (injected error history) | Yes, but synthetic, no tools |
| Lu et al. (AgentErrorTaxonomy) | No | Yes (5 modules) | Yes (human) | No | No |
| CAR / CausalFlow | No | No | Yes | Yes | No (single-trace tooling, synthetic SCM validation) |
| Rabanser et al. | No | No | No | Perturbations, not step-level | Profile, not mechanism |
| AgentForesight / online monitors | No | Partial | Yes (decisive error) | No | Prediction, not causation |

Sinha et al. proved self-conditioning exists in a no-tool, plan-given, synthetic setting by injecting synthetic error histories. `[LIT]` Nobody has asked whether it is the dominant mechanism in *real tool-using agent trajectories*, relative to state staleness and irrecoverable typed errors, and nobody has measured its share of the horizon gap. That is a mechanism question answerable with replays, not labels. `[INF]`

---

## 3. Novelty Threat Assessment

Threat levels: CRITICAL = substantially the same contribution exists · HIGH = major conceptual overlap · MEDIUM = important overlap, defensible gap · LOW = adjacent · NONE = background.

| Paper | Year | Venue | Research question | What it measures | Failure taxonomy | Long-horizon analysis | Metric | Benchmark | Overlap with proposal | Genuine gap remaining | Threat |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| HORIZON: The Long-Horizon Task Mirage (Wang, Bai, …, Song, Nowak) arXiv:2604.11978 | 2026 | COLM 2026 | Where do agents break as horizon grows, and why | Success vs. compositional depth s; failure-type distribution per domain | 7 categories (environment, instruction, false assumption, planning, catastrophic forgetting, history error accumulation, memory limitation), FMEA-grounded | Yes: nested task sets, depth/breadth extension, "breaking region" | Success rate per s; intrinsic horizon H*; compositional depth | WebArena, AgentBench (OS/DB), MAC-SQL, Isaac Sim; GPT-5 variants, Claude-4 | Taxonomy, horizon sweep, benchmarks, "collapse" framing, LLM-judge validation (κ=0.61 human, 0.84 judge) | No hazard/survival modelling; no step localization; no counterfactual; no independence null; frontier closed models only; failure attribution is per-trace not per-step | **CRITICAL** for the proposal as written; MEDIUM for the redesigned paper |
| Beyond pass@1: Reliability Science Framework (Khanal, Tao, Zhou) arXiv:2603.29231 | 2026 | arXiv (unreviewed) | How does pass^k degrade with task duration | RDC, VAF, GDS (partial credit), MOP (entropy meltdown) | None (no error typing) | Yes: 4 duration buckets, super-linear vs. geometric baseline, memory scaffold ablation | RDC slope, VAF, GDS, MOP | Own 396-task suite (SE/WR/DP); 10 open-weight incl. Llama 3.x, Qwen3, Mistral | Reliability-decay curve, "collapse" detection, geometric null, exact model families proposed | Duration = human-time proxy (they admit it fails); MOP uncalibrated (Table 15 empty); no failure localization; no causal decomposition; weak statistics | **CRITICAL** for ASI/decay-curve contribution; LOW for mechanism paper (it is a foil) |
| The Illusion of Diminishing Returns (Sinha, Arun, Goel, Staab, Geiping) arXiv:2509.09677 | 2025 | ICLR 2026 (OpenReview) | Does per-step accuracy fall with steps, and why | Step/turn accuracy, horizon length H_s, self-conditioning | None | Yes: synthetic execution tasks up to thousands of steps | Per-step accuracy vs. step; H_0.5 | Synthetic dictionary/arithmetic execution, no tools | Directly answers "independent vs. state-dependent" (Model 1 vs. 3) in synthetic setting | Not tool-using agents; plan given; no environment state; no recovery; no error typing | **HIGH** for RQ2 as framed; the redesigned paper should extend it to tool trajectories |
| Towards a Science of AI Agent Reliability (Rabanser, Kapoor, …, Narayanan) arXiv:2602.16666 | 2026 | ICML 2026 | Define and measure reliability beyond accuracy | 12 metrics: consistency, robustness, predictability, safety | None | No (task-level) | Profile vector | Two benchmarks, 12–15 frontier models | "Reliability profile not scalar" argument; capability vs. reliability | Horizon absent; trajectory dynamics absent | **HIGH** for the "vector not scalar" framing (cite, don't claim) |
| Where LLM Agents Fail and How They Can Learn From Failures (Lu et al.) arXiv:2509.25370 | 2025 | arXiv/ICLR sub. | Root-cause failures; propagation | AgentErrorTaxonomy (5 modules), AgentErrorBench, AgentDebug | Yes: memory, reflection, planning, action, system | Qualitative cascade heatmaps | Root-cause step accuracy | ALFWorld, WebShop, GAIA | Error propagation as "primary bottleneck"; step-level root cause labels | Propagation is asserted from labels, not measured by intervention; no horizon sweep | **HIGH** for taxonomy; MEDIUM for propagation |
| Causal Agent Replay (Shah) arXiv:2606.08275 | 2026 | arXiv | Which step caused a failure | do-operations on steps, outcome-distribution shift, Shapley | None | No | Contrastive causal effect, Shapley credit | Synthetic SCMs + demos | Counterfactual replay methodology | Tooling + synthetic validation only; no population estimates by error type/position | **MEDIUM** (adopt as method, cite) |
| CausalFlow arXiv:2605.25338 | 2026 | arXiv | Attribute and repair failed traces | Causal Responsibility Score | None | No | CRS | Agent traces | Counterfactual step replacement | Same as CAR | **MEDIUM** |
| AgentForesight arXiv:2605.08715; Real-time detection arXiv:2608.02464; PrefixGuard; Trajectory Guard | 2026 | arXiv | Predict failure online from prefix | AUROC, earliness | Decisive-error step | No horizon sweep | Prefix risk | AFTraj-2K, ATBench | Early-warning component | Prediction, not mechanism; multi-agent focus | **HIGH** for any early-warning contribution → drop it |
| Measuring AI Ability to Complete Long Tasks (METR, Kwa et al.) arXiv:2503.14499 | 2025 | arXiv | Task-length capability horizon | 50%-success time horizon | None | Yes (human-time) | H_50 | METR suite | "Reliability horizon" concept | Threshold horizon exists; per-step hazard does not | **HIGH** for "Reliability Horizon"; do not claim as new |
| τ-bench / τ²-bench (Yao et al.) arXiv:2406.12045 | 2024 | arXiv/ICLR | Consistency across runs | pass^k | None | No | pass^k | Retail/airline | Capability vs. reliability distinction | Horizon absent | **MEDIUM** background |
| LLMs Get Lost in Multi-Turn Conversation (Laban et al.) arXiv:2505.06120 | 2025 | arXiv | Multi-turn degradation | Aptitude vs. unreliability decomposition | None | Sharded vs. full instructions | Aptitude/unreliability | Sharded tasks | Capability/reliability decomposition; premature-commitment mechanism | No tools, no environment | **MEDIUM** |
| Why Do Multi-Agent LLM Systems Fail (Cemri et al.) arXiv:2503.13657 | 2025 | arXiv/NeurIPS | MAS failure taxonomy | MAST 14 modes | Yes | No | — | 7 MAS frameworks | Taxonomy method (grounded theory, κ) | Multi-agent | **LOW** |
| Which Agent Causes Task Failures and When (Who&When) arXiv:2505.00212 | 2025 | ICML 2025 | Attribution | Step-level accuracy (~14% SOTA per CAR) | Decisive step | No | Attribution accuracy | Who&When | Failure localization | Multi-agent; correlational | **LOW–MEDIUM** |
| Beyond Resolution Rates (coding agents) arXiv:2604.02547 | 2026 | arXiv | Behavioural drivers of SWE-agent success | Paired success/failure trajectories; length confounded by difficulty | Behavioural | No | — | SWE-bench | Warns that trajectory length is difficulty-confounded | No intervention | **MEDIUM** (cite as confound evidence) |
| Beyond the Leaderboard synthesis arXiv:2607.05775 | 2026 | arXiv | Survey of tool-use/planning/reasoning failures | — | Synthesis | Reviews | — | — | Related-work map | — | **NONE** (use as map) |
| LEAD: no-recovery bottleneck (Pushkin, Abbe) arXiv:2603.06870 | 2026 | COLM sub. | Decomposition vs. recovery in execution | Accuracy vs. n; non-uniform per-step error | None | Yes (puzzles) | — | Algorithmic puzzles | "Irrecoverability of a few hard steps" | No tools; method paper | **MEDIUM** (mechanism support) |
| Context Rot (Hong et al., Chroma) 2025; Lost in the Middle | 2025 | tech report / TACL | Long-context degradation | Accuracy vs. input length | None | Length sweep | — | NIAH variants | Context-length mechanism | Not agentic | **LOW** |
| MIRAGE-Bench (agent hallucination) arXiv:2507.21017 | 2025 | arXiv | Where agents hallucinate | — | Hallucination types | No | — | Multiple | "Hallucination accumulation" | No horizon sweep | **LOW** |
| Holistic Agent Leaderboard (Kapoor et al.) arXiv:2510.11977 | 2025 | ICLR 2026 | Infrastructure; logged traces | — | Behaviours found by judge | No | — | Many | Trace data source | — | **NONE** (data source) |
| TRAIL (Deshpande et al.) 2025 | 2025 | arXiv | Trace issue localization | Localization accuracy | Yes | No | — | GAIA, SWE-bench traces | Step-level labels | No horizon | **LOW** |
| LongDS-Bench arXiv:2605.30434; Long-Horizon-Terminal-Bench arXiv:2607.08964; OdysseyBench | 2026 | arXiv | Long-horizon benchmarks | Accuracy by progress | Coarse | Yes (within-benchmark) | Dense reward | Own | Confirms degradation is generic | No mechanism | **LOW** |

**Five closest papers, in one sentence each (acceptance criterion for Phase 0):**
1. HORIZON describes composition shift with horizon; we estimate the causal share of each mechanism by replay.
2. Khanal et al. show the gap from a geometric null; we explain the gap.
3. Sinha et al. show self-conditioning in synthetic no-tool execution; we test whether it survives, and what share it holds, in tool-using trajectories with environment state.
4. Lu et al. assert propagation from labels; we measure propagation by intervention, per error type and position.
5. CAR/CausalFlow attribute single traces; we use replay as a population estimator under a controlled horizon sweep.

---

## 4. Current State of the Field

**What is already measured `[LIT]`:**
- Terminal success on rich, logged trajectories: WebArena, SWE-bench (Verified/Pro), AppWorld, τ²-bench, OSWorld, TheAgentCompany, Terminal-Bench, HAL all expose full traces; HAL has released billions of tokens of logs (arXiv:2510.11977).
- Repeated-run consistency: pass^k (τ-bench), consistency/robustness/predictability profiles (Rabanser et al.).
- Horizon-conditioned success: METR time horizon; HORIZON compositional depth; Khanal duration buckets; LongDS-Bench progress-binned accuracy.
- Failure taxonomies with agreement statistics: HORIZON (κ 0.61/0.84), MAST, AgentErrorTaxonomy, TRAIL, MIRAGE.
- Step-level attribution: Who&When, FALAT, AgentForesight, CAR, CausalFlow.
- Per-step accuracy decay and self-conditioning: Sinha et al.
- Partial credit and dense reward: GDS (Khanal), Long-Horizon-Terminal-Bench.

**What is not measured `[INF]`:**
- The *hazard* of first unrecovered error as a function of normalized position (t/H*), with competing risks (recovered vs. terminal). Khanal's entropy MOP is the only "onset" quantity and it is uncalibrated.
- The *decomposition* of the horizon gap into independent-error, self-conditioning, state-staleness and irrecoverability components — every existing paper attributes by label or asserts by curve shape.
- Propagation probabilities P(terminal failure | error type, position) estimated by intervention rather than judged.
- Whether "memory" failures in traces are really attention/self-conditioning failures (HORIZON's own definition of catastrophic forgetting says the constraint is *still in context*, which is a self-conditioning/attention claim, not a memory claim — nobody tests it).
- Whether recovery is a distinct capability or an artefact of error type (recoverable errors vs. recoverable agents).

**Premise check.** The premise "benchmarks mainly report success/failure" is false as a description of 2026 (HAL, HORIZON, Rabanser). It is true that *most analysis of those traces is still correlational*. The paper's motivation must say that.

---

## 5. Defensible Research Gap

**Gap statement `[REC]`:** Long-horizon degradation in tool-using LLM agents has been described (curves), labelled (taxonomies), and forecast (online monitors), but not *explained by intervention*. We do not know what fraction of the gap between observed long-horizon success and the success predicted from matched short-horizon competence is caused by (a) the agent conditioning on its own earlier errors, (b) stale beliefs about environment state, versus (c) irrecoverable typed errors at specific positions — because no study has held the task, policy and sampling fixed and replayed trajectories under `do()`-interventions at scale across a controlled horizon sweep.

Why this is defensible: it is a causal question with an identification strategy (replay under fixed policy and seed), it is falsifiable (interventions may not move the gap), it directly extends the two strongest recent papers (HORIZON, Sinha), and it needs no training.

---

## 6. Recommended Research Question

**Primary RQ.** When a tool-using agent's long-horizon success falls below what its matched short-horizon competence predicts, how much of that shortfall is caused by conditioning on its own earlier errors, by stale environment-state beliefs, and by irrecoverable typed errors — as measured by counterfactual replay under a controlled horizon sweep?

**Secondary RQ-A.** Does the hazard of the first unrecovered error increase with normalized position (t/H*) after controlling for subtask difficulty, and is the increase attributable to the mechanisms above (i.e., does hazard flatten under history-scrubbing or state-refresh interventions)?

**Secondary RQ-B.** Is recoverability a property of the error type × position or of the model — i.e., do models differ mainly in *which* errors they make or in *whether* the same error becomes terminal?

Ranking of the prompt's RQ1–RQ8: RQ2 (mechanism) > RQ5 (recoverable vs. irreversible) > RQ3 (where instability starts) > RQ6 (model-specific profiles) > RQ1 (trivial, already shown) > RQ8 (METR/Sinha) > RQ4 and RQ7 (prediction; crowded, drop).

---

## 7. Recommended Contribution Type (ranked)

1. **Reliability model + empirical decomposition** (interventional mechanism attribution) — the paper.
2. **Diagnostic framework** — the replay protocol and estimators, released as a tool; second contribution.
3. **Empirical law** — only if the decomposition is stable across models/domains (e.g., "self-conditioning share grows with t/H*; irrecoverability is concentrated in state-changing tool errors"); do not promise it.
4. **Taxonomy** — minimal, reused from HORIZON/Lu with a recoverability axis added; supporting only.
5. **Metric** — hazard and propagation estimates as outputs, not as a named index; no ASI.
6. **Benchmark** — no; reuse existing environments.
7. **Early-warning system** — no; crowded and off-thesis.
8. **Theory** — a short section deriving what each intervention identifies under a stated SCM; not a theory paper.

---

## 8. What "Failure Geometry" Should Mean

**Search result `[LIT]`.** "Failure geometry" has no established meaning in agent evaluation, sequential decision making, or reliability engineering. Adjacent uses: "geometric characteristics of failures" for input-space failure regions in DRL testing (arXiv:2606.31372); "critic-geometric" evidence for cascading failures in MARL (arXiv:2602.08104); "geometric decay" of p^T is standard probability. None conflicts fatally, but the DRL-testing usage (failure regions in *input space*) invites confusion with a proposal about *time/position within a trajectory*.

**Recommendation `[REC]`.** Drop it. It is a metaphor, not an object. The scientific objects in the redesigned paper are (i) a position-indexed hazard function with competing outcomes and (ii) a causal decomposition of the horizon gap. Accurate names: **"horizon-gap decomposition"** for the contribution, **"positional hazard profile"** for the curve. If a family name is needed for the per-model tuple (hazard profile, recovery function, propagation kernel), use **"reliability profile"** (already in Rabanser's vocabulary — cite, extend).

---

## 9. What "Reliability Collapse" Should Mean

Compare the candidate definitions on testability with modest data:

| Definition | Testable? | Verdict |
|---|---|---|
| Superlinear degradation of P(success) in T | Yes, but conflates difficulty growth with horizon; Khanal already did it | Reject as headline |
| Sharp increase in hazard | Yes (discrete hazard GLM with spline/knot) | Keep as *diagnostic* |
| Change point in survival curve | Yes but low power at n≈100 tasks per level | Optional |
| Departure from independent-error prediction | Yes (log-residual vs. Σ log p_i) | Keep as *definition of the gap* |
| Sudden increase in correlated failures | Needs within-trajectory error labels; identifiable via replay | Keep as *mechanism* |
| Regime transition | Needs latent-state model; over-engineered | Reject |

**Recommended definition `[REC]`.** Do not use "collapse" as a term of art. Define two quantities:

- **Horizon gap** at level s: Δ(s) = log P_obs(success | s) − Σ_{i≤s} log p̂_i, where p̂_i is the success rate of subtask i evaluated in isolation with the same model/harness (the independence null). Δ < 0 with CI excluding 0 is "excess horizon degradation."
- **Positional hazard** h(u) = P(first unrecovered critical error at normalized position u | none before), u = t/H*. "Rising hazard" is a statistically significant positive slope of the cloglog-linear predictor in u, after task random effects.

"Collapse" may be used descriptively for regions where h(u) rises and Δ(s) is strongly negative, never as a measured quantity.

---

## 10. ASI Audit

**Should ASI exist? No. `[REC]`**

Redundancy check: any scalar summary of trajectory reliability collapses onto one of: pass^k (consistency), METR H_50 / HORIZON breaking region (horizon), RDC slope (Khanal), GDS (partial credit), or a survival-curve functional (area under survival = expected fraction of horizon survived — which is just mean normalized time-to-first-unrecovered-error). Each of those exists. `[LIT]/[INF]` A new composite adds a weighting choice reviewers will call arbitrary, and — the prompt's own example — it hides "rare-but-catastrophic" vs. "frequent-but-recoverable."

**Better representation.** Per model × domain, report a **reliability profile tuple**:
- S(u): survival to first *critical* error (competing-risk cumulative incidence for recovered vs. terminal);
- h(u): positional hazard;
- R(e, u): recovery probability by error type and position;
- π(e, u): propagation probability P(terminal | error e at u) estimated by replay;
- Δ(s): horizon gap and its decomposition shares (θ_self, θ_state, θ_irrecov, θ_resid).

If a scalar is demanded for a table, use **expected surviving horizon** E[U_fail] = ∫S(u)du (units: fraction of H*). It preserves the location of degradation and sacrifices type and recovery information — say so explicitly.

---

## 11. Alternative Metrics

| Metric | What it captures | What it hides | Status | Use |
|---|---|---|---|---|
| Survival S(t) / Kaplan–Meier | Where trajectories leave the on-track set | Type; recoverability unless competing risks | Standard | Yes, discrete-time |
| Hazard h(t) | Instantaneous risk; shape (rising/flat/U) | Absolute level small at each t → needs pooling | Standard; new in agent evaluation as position-normalized with covariates | Yes, primary diagnostic |
| Horizon residual Δ(s) | Excess degradation beyond independence | Mechanism | Khanal did crude version; HORIZON nested sets ≈ implicit | Yes, as gap definition |
| Recovery-adjusted reliability | Distinguishes fragile vs. resilient | Needs error labels | Partially in Lu et al. | Yes, as R(e,u) |
| Propagation score π(e,u) | Causal downstream effect of an error | Cost: replays | CAR/CausalFlow tooling | Yes, core |
| pass^k | Run-to-run consistency | Trajectory dynamics | τ-bench | Report for context |
| GDS / dense reward | Partial progress | Dynamics | Khanal; LH-Terminal-Bench | Optional |
| Entropy meltdown (MOP) | Looping | Uncalibrated | Khanal | Do not adopt |
| Composite index (ASI) | — | Everything | — | No |

Better alternative not on the prompt's list: **decomposition shares** θ = (θ_self, θ_state, θ_irrecov) — the fraction of Δ(s) removed by each intervention. This is the paper's headline quantity.

---

## 12. Failure Taxonomy

**Do not build a new taxonomy.** HORIZON (7), AgentErrorTaxonomy (5 modules), TRAIL and MAST already exist with agreement statistics; a ninth taxonomy is a reviewer attack. `[REC]` Instead:

1. **Adopt** HORIZON's categories as the *what* (they are FMEA-grounded and already validated against humans).
2. **Add two orthogonal axes** that existing taxonomies lack and that the interventions need:
   - **Detectability class:** rule-detectable (malformed tool call, invalid tool, env error string, exact repeated call, step-limit) vs. state-checkable (env assertion, subtask checkpoint failed) vs. judgement-required (wrong plan, false assumption).
   - **Recoverability outcome:** recovered (subsequent checkpoint passes), persisted, terminal — assigned by *state checks*, not by judge.
3. **Collapse** planning/tool-selection/tool-execution/state/memory/reasoning/recovery/verification/termination (the prompt's nine) into the HORIZON seven plus the two axes. The prompt's "memory failures" become a *hypothesis* (H8) to be tested by the state-refresh intervention, not a label.

Resulting label = (HORIZON category, detectability class, position u, recoverability outcome). Only the first field ever needs an LLM judge, and only for judgement-required events.

---

## 13. Error Propagation Framework

Represent a trajectory as s₀ → (a₁,o₁) → s₁ → … → s_T, with a fixed stochastic policy π_θ and deterministic environment E (required; see benchmarks).

**Estimand.** For an error event e at step t in trajectory τ: π(e,t) = P(terminal failure | τ_{<t}, do(a_t := a_t^err)) − P(terminal failure | τ_{<t}, do(a_t := a_t^fix)), where a_t^fix is a corrected action. This is CAR's contrastive estimator `[LIT]`; we use it as a population estimator, stratified by (category, u).

**Estimation `[REC]`.**
1. Take the observed failed trajectory; identify candidate error steps by rules/state checks (and judge where needed).
2. Replay the environment to s_{t−1} by re-executing the logged action prefix (deterministic env, fixed seed).
3. Branch A (factual): continue from s_{t−1} with the erroneous a_t, then sample k continuations from π_θ.
4. Branch B (counterfactual): substitute a_t^fix (oracle-corrected action: correct tool/args from the reference solution or a minimal repair), sample k continuations.
5. π̂ = mean(fail_A) − mean(fail_B); CI by bootstrap over continuations; aggregate across trajectories with a mixed-effects logistic model (task random effect).
6. **Propagation depth:** first step in A where a state check fails that passes in B; **time-to-recovery:** steps until A's checkpoints re-align with B's.

CAR's "point-of-commitment" issue (resampling re-rolls downstream stochasticity) is handled by sampling k continuations in both branches from the same seed set and reporting the distributional effect. `[LIT]/[REC]`

Cost control: k = 4 continuations, only for the first two candidate errors per failed trajectory, only in the deterministic environment. Everything else uses labels.

---

## 14. Reliability Horizon

**Should "Reliability Horizon" H_τ = max{t : P(success | horizon t) ≥ τ} be a contribution? No. `[REC]`** It is METR's time horizon at threshold τ (arXiv:2503.14499) restated in steps; Sinha et al. define H_s the same way in steps; HORIZON's "breaking region" is the same object with a band instead of a point. `[LIT]` Report H_0.5 as a *descriptive* covariate for each model (it normalizes u), cite all three, and do not claim it.

What is new and worth one figure: **H_τ under intervention** — how far the horizon extends when the history is scrubbed or state refreshed. That is a consequence of the decomposition, not a separate contribution.

---

## 15. Failure Prediction / Early Warning

**Exclude. `[REC]`** AgentForesight (RL-trained online auditor), PrefixGuard, Trajectory Guard, and the ESN monitor paper (arXiv:2608.02464) already occupy prefix-based failure prediction with earliness metrics and benchmarks (AFTraj-2K, ATBench). `[LIT]` A prediction section would be a weaker version of those and would dilute the causal thesis. The only permissible use: as a *validation* that intervention-identified pivotal steps coincide with where a simple prefix predictor's risk jumps (one appendix figure).

---

## 16. ICLR Positioning

To be an ML paper and not a benchmark report, the paper must have: a stated causal model, estimands, identification assumptions, estimators with CIs, and a falsifiable claim about mechanism shares.

**Structure `[REC]`:**
1. *Problem:* horizon gap Δ(s) exists (one figure reproducing HORIZON/Khanal-style curves on our setup — a check, not a contribution).
2. *Model:* an SCM of the agent loop; three named mechanisms; what each `do()` identifies (history-scrub identifies self-conditioning; state-refresh identifies staleness; oracle-fix identifies irrecoverability). One proposition with proof sketch.
3. *Estimators:* discrete-time competing-risk hazard with task random effects; contrastive replay estimator; decomposition shares by sequential intervention with order-robustness check (Shapley over the three interventions if budget allows; CAR already supplies the estimator).
4. *Results:* shares θ by model, domain, and u; hazard curves under each intervention; recoverability by error type × position.
5. *Implication:* which intervention class (history hygiene vs. state re-observation vs. verification) is worth building — an actionable, testable claim that HORIZON could only speculate about.

The "significant ML paper" line is crossed when the reader learns something they could not have learned from labels: e.g., that most of what HORIZON calls "catastrophic forgetting" disappears when the erroneous *history* is scrubbed but the constraint is left untouched — meaning it was never forgetting. `[HYP]`

---

## 17. Alternative Positionings (ranked)

1. **Interventional decomposition of the horizon gap** (recommended; ICLR/NeurIPS main track).
2. **Positional hazard modelling of agent trajectories with competing risks** — statistically cleaner, less mechanistic; TMLR or an evaluation-track venue. Risk: "repackaged survival analysis."
3. **Recoverability science:** is recovery an agent property or an error property? Focused, smaller; COLM.
4. **Testing the self-conditioning hypothesis in tool-using agents** — a direct extension of Sinha et al.; strong single-claim paper; ICLR workshop → main if effect sizes are large.
5. **Replay-based validation of failure taxonomies** — use counterfactuals to test whether judge labels are causally meaningful (labels whose "fix" doesn't change outcome are noise). Methodological; NAACL/EMNLP evaluation track.
6. **Reliability profile for open-weight agents** — descriptive; already done (Khanal); only as a workshop reproduction.

---

## 18. Benchmark Recommendation

Requirements for the redesigned paper: deterministic, resettable, cheap prefix replay; programmatic state checks; compositional tasks with known intrinsic horizon H*; tool use; open-weight models can achieve non-trivial success (otherwise floor effects kill hazard estimation).

| Benchmark | Trajectories exposed | Deterministic replay | State checks | H* known | Open-weight success | Verdict |
|---|---|---|---|---|---|---|
| AgentBench | Yes | OS/DB: yes; others mixed | Partial | No | Low on many envs | Drop: heterogeneous, stale (2023), tasks short; HORIZON already used it |
| WebArena | Yes | Docker reset is slow (minutes); Playwright actions replayable | Programmatic evaluators, but intermediate checks weak | No | Low (<20% for open ≤70B) | Drop as primary; floor effects and replay cost |
| SWE-bench Lite | Yes | Docker, deterministic | Tests only at end; no intermediate checkpoints | No | Moderate for 32B+ | Replace with **SWE-bench Verified subset** if a coding domain is kept; add intermediate checks (file-state assertions, test-subset runs) |
| **AppWorld** | Yes | Yes, fully deterministic, fast reset | Unit-test-style state checks; scenario dependencies; programmatic | Reference solutions give H* | Non-trivial for 32B+ | **Primary** |
| τ²-bench | Yes | Deterministic DB, but user simulator is an LLM | Final DB state | Approx. | Moderate | Secondary alternative; user sim adds noise |
| HORIZON task families | Yes | Depends on domain | Yes at level boundaries | Yes (by construction) | Unknown | Use its OS/DB families if released; otherwise replicate the nesting idea in AppWorld |

**Minimum set `[REC]`:** one primary environment (AppWorld) with a HORIZON-style nested horizon sweep constructed by chaining scenario subtasks, plus one ecological-validity environment (SWE-bench Verified, 100-instance stratified subset) where only the observational hazard analysis and label-based recovery outcomes are run (no replays). Cross-benchmark comparison of failure *types* is not attempted; only the decomposition shares are compared, with the caveat that they are domain-conditional. `[REC]`

---

## 19. Model Recommendation

Principle: control family/format first; breadth last.

**Primary (within-family scaling, same tool-call format, same chat template):** Qwen3-8B, Qwen3-32B, Qwen3-235B-A22B (or the 30B-A3B MoE if budget is tight), all instruction variants, native function calling, thinking mode **off** as the main condition. `[REC]` Why: isolates capability from harness; three sizes give a scaling axis for hazard shape and decomposition shares.

**Cross-family robustness (two models):** Llama-3.3-70B-Instruct and Mistral-Small-3.2-24B. Both have native tool calling; both appear in Khanal's study, which lets the descriptive curves be cross-checked. `[LIT]`

**Ablation:** Qwen3-32B thinking on vs. off — Sinha et al. report thinking mitigates self-conditioning `[LIT]`; this is the cheapest test of whether θ_self shrinks with reasoning.

Fix: temperature 0.6 (need stochasticity for pass^k and replays), top-p fixed, max tokens fixed, identical system prompt, identical tool schemas, 3 seeds. Serve all models through one inference stack (vLLM) to eliminate provider-routing noise Khanal reports. `[LIT]/[REC]`

What must not be claimed: that family differences are "reliability" differences. Report family as a random effect; only the within-family scaling trend and the intervention effects are interpreted causally.

---

## 20. Agent Harness Design

One harness, one loop (ReAct with native tool calls), no planner, no memory module, no verifier in the main condition — those are the interventions, not the baseline. `[REC]`

**Standardize:** system prompt; tool schema (JSON); observation truncation length; step limit = 3·H*; termination rule; retry policy on malformed calls (0 retries; malformed call is an event, not a retry); context management (none — truncation only at hard limit, logged as an event).

**Log per step (schema, JSONL):** `run_id, task_id, seed, model, step, u=step/H*, prompt_hash, raw_output, parsed_action, tool_name, tool_args, tool_result, tool_result_hash, env_state_hash, checkpoint_results[], event_labels[] (rule), judge_labels[] (if any), tokens_in/out, latency, intervention_branch (factual|history_scrub|state_refresh|oracle_fix|none), parent_run_id, branch_step`.

**Intervention hooks:** (i) history-scrub: replace the message history before step t with the *counterfactual clean* history (branch B's prefix) while keeping env state factual; (ii) state-refresh: inject an oracle state summary at step t (env-derived, not model-derived) without touching history; (iii) oracle-fix: substitute a_t. Each must be a pure function of logged data so replays are reproducible.

---

## 21. Experimental Blueprint

| Dimension | Levels |
|---|---|
| Environment | AppWorld (primary); SWE-bench Verified subset (observational only) |
| Horizon level s | 1, 2, 3, 4 (nested; s=1 are atomic subtasks) |
| Model | Qwen3-8B, 32B, 235B; Llama-3.3-70B; Mistral-Small-24B |
| Reasoning | off (all); on (Qwen3-32B only) |
| Seeds | 3 |
| Intervention | none; history-scrub; state-refresh; oracle-fix |
| Analysis | Δ(s); h(u) with competing risks; R(e,u); π(e,u); θ shares |
| Annotation | rules + state checks (all); judge (judgement-required events only, human-validated) |

Cells: primary factual runs = 5 models × 4 levels × N tasks × 3 seeds. With N = 40 base scenarios → 40 tasks per level → 5×4×40×3 = 2,400 trajectories (plus 400 atomic-subtask runs for p̂_i). Replays: on failed trajectories at s≥2 only, first 2 error events, 3 interventions × k=4 → bounded to ~6,000 short continuations. SWE-bench: 100 instances × 3 models × 3 seeds = 900 trajectories, no replays.

---

## 22. Minimum Viable Experiment Set

1. Qwen3-8B and Qwen3-32B; AppWorld; s ∈ {1,2,3}; 30 base scenarios; 3 seeds → 540 trajectories + 270 atomic runs.
2. Estimate Δ(s) and h(u) with competing risks.
3. Replays on failed s=3 trajectories only: history-scrub and oracle-fix (drop state-refresh), k=3 → ≤1,500 continuations.
4. Human-validate judge labels on 150 events.
Supports the central claim if θ_self + θ_irrecov explain a CI-bounded majority of Δ(3) and hazard flattens under history-scrub. Cost `[INF]`: ~10–20 GPU-days on 2×A100/H100 for 32B (vLLM), negligible for 8B.

---

## 23. Ideal Experiment Set

All of §21 plus: Qwen3-235B and cross-family models; state-refresh intervention; reasoning on/off; Shapley over interventions (CAR estimator) to make shares order-robust; SWE-bench Verified observational replication of hazard shape; a synthetic control (§24). Cost `[INF]`: ~60–100 GPU-days or ~$1–3k API-equivalent for the 235B via a hosted endpoint.

---

## 24. Synthetic / Controlled Experiments

Purpose: isolate mechanisms where the environment is trivial so that tool/domain confounds vanish.

- **Toolified execution task** (extension of Sinha et al.): the model must apply a sequence of state updates via a `store(key,value)`/`read(key)` tool over an external key-value store rather than in context. Sweep steps 10–200. Interventions: inject synthetic erroneous history (self-conditioning test) vs. inject stale stored state (staleness test). Prediction `[HYP]`: self-conditioning persists when errors are in *history* even if the *store* is correct; staleness matters only when the store is wrong.
- **Irrecoverability planting:** insert one irreversible state-changing tool (`delete`) at controlled positions; measure π(e,u) exactly.

This gives ground truth for the estimators before applying them to AppWorld.

---

## 25. Matched Short-vs-Long Experiments

- Atomic tasks: each AppWorld scenario subtask i run alone with the same harness; p̂_i from 3 seeds × 40 → p̂_i with Wilson CIs.
- Composed tasks: level s chains subtasks 1..s in fixed order with a shared state (HORIZON breadth extension) `[LIT]`.
- Null prediction: P_ind(s) = Π p̂_i. Gap Δ(s) = log P_obs − log P_ind with CI by paired bootstrap over scenarios.
- Difficulty control: subtask difficulty is *identical by construction* between short and long conditions; what changes is only shared history and state. Any residual is horizon-induced (history+state), which is exactly what the interventions then split. `[REC]`
- Order ablation: reverse subtask order for half the scenarios to detect position × subtask confounds.

---

## 26. Counterfactual Experiments

What can be replayed: AppWorld (fully), SWE-bench (docker; prefix re-execution deterministic but slow — observational only in this plan), τ²-bench (DB deterministic, user sim stochastic — seedable). WebArena: replayable in principle, environment reset cost high — excluded.

Interventions and what each identifies (stated SCM: history H_t, env state S_t, action A_t = π(H_t), S_{t+1} = E(S_t, A_t)):
- `do(H_t := H_t^clean)`, S_t factual → effect of erroneous history on future actions = **self-conditioning**.
- `do(S_t-belief := oracle summary)`, H_t factual → effect of stale/incorrect state belief = **staleness**.
- `do(A_t := A_t^fix)` → downstream effect of the specific error = **propagation/irrecoverability**.
Approximate alternative where replay is impossible: prefix-matched observational comparison (trajectories identical up to t, diverging at t), reported as correlational and clearly labelled.

---

## 27. Statistical Evaluation

Only what the data shape needs `[REC]`:
- **Discrete-time survival:** event = first *critical* error (state-check failure or judge-flagged planning/assumption error); complementary log-log GLMM with fixed effects for u (natural spline, 3 df), model, level s, intervention; random intercept for scenario. Discrete time makes Cox unnecessary and avoids tie problems.
- **Competing risks:** cause-specific hazards for recovered vs. terminal outcomes; cumulative incidence functions.
- **Change point:** likelihood-ratio test of a piecewise-linear u-effect vs. linear, only if the spline suggests it; report as exploratory.
- **Horizon gap:** paired bootstrap over scenarios (2,000 resamples).
- **Replay effects:** mixed-effects logistic on continuation outcomes with trajectory random effect; effect sizes as risk differences with 95% CI.
- **Decomposition shares:** θ_k = (Δ_before − Δ_after,k)/Δ; report with bootstrap CIs and a sequential-order sensitivity table; Shapley over the three interventions if all 2³ combinations are run.
- **Multiplicity:** Holm across the pre-registered primary tests (three θ shares, hazard slope); everything else labelled exploratory.
- **Power `[INF]`:** with 40 scenarios × 3 seeds per level, a hazard-ratio of 1.5 per 0.25u is detectable at ~80% power; smaller effects need the ideal set.
No permutation tests, no mixed-effects for their own sake.

---

## 28. Annotation Protocol

Three tiers, escalating cost:
1. **Rule-based (100% coverage):** malformed JSON, unknown tool, schema-invalid args, tool error string, exact repeat of (tool,args) within 5 steps, step-limit termination, text-only turn. Deterministic; reported with counts.
2. **State-checkable (100% coverage):** AppWorld checkpoint assertions after each subtask; SWE-bench: test-subset run after each patch. Gives recoverability outcome without a judge.
3. **Judge (judgement-required events only):** HORIZON categories planning / false assumption / instruction / forgetting. Judge = one open-weight model (Qwen3-235B) with the trajectory window ±5 steps and the checkpoint results; prompt released. Validation: two humans label 200 events (κ target ≥ 0.6, matching HORIZON's inter-annotator level `[LIT]`); judge–human κ reported; per-category confusion matrix; **causal validation:** for 50 judge-labelled "decisive" errors, run oracle-fix replay — a label is "causally valid" if π̂ > 0 with CI excluding 0; report the validity rate. Judge bias check: compare label rates on succeeded vs. failed trajectories with outcome hidden.

---

## 29. Ablations

**Required:** reasoning on/off (Qwen3-32B); subtask order reversal; k continuations (2 vs 4) sensitivity; judge model swap (Qwen3-235B vs Llama-70B) on the validation set; step-limit 2·H* vs 3·H*; temperature 0.6 vs 1.0 on one model.
**Optional:** intervention order (Shapley); context-truncation policy; tool-result truncation length; scaffold with a verifier step (to see whether θ_irrecov shrinks).

---

## 30. Expected Results

**Strongly supportive `[HYP]`:** Δ(s) significantly negative at s≥2; hazard rising in u for base condition; history-scrub flattens hazard and removes ≥40% of Δ; oracle-fix on state-changing tool errors has π near 1 while on read-only errors near 0; θ_self falls with model size and with reasoning on; shares stable across seeds and roughly similar in the SWE-bench observational proxies.
**Kill conditions:** Δ(s) not distinguishable from 0 (then the paper becomes "independent errors explain agents; long tasks are just more steps" — publishable as a negative result at a workshop, not ICLR); interventions each remove <10% of Δ with CIs overlapping 0 (mechanisms unidentified — pivot to positioning #2, hazard modelling); judge–human κ < 0.4 for judgement-required categories (restrict to rule/state tiers; still viable since interventions don't depend on judge labels).

---

## 31. Reviewer Attack Surface

| Attack | Mitigation |
|---|---|
| "Just a failure taxonomy" | No new taxonomy; reuse HORIZON; contributions are estimands and shares |
| "Descriptive, not scientific" | Interventions with stated identification; falsifiable shares |
| "Longer tasks obviously fail more" | We measure the *gap from independence* and explain it; Δ≈0 would refute us |
| "Confusing difficulty with horizon" | Subtask-matched composition; difficulty identical by construction |
| "ASI arbitrary" | No ASI |
| "Labels subjective / LLM judges unreliable" | Two label tiers need no judge; judge validated by humans *and* by causal replay |
| "Benchmarks incomparable" | One primary env; cross-env comparison only of shares, hedged |
| "Harness dominates" | One harness; harness variants are interventions; family as random effect |
| "Different tool interfaces" | Native function calling for all; within-family primary analysis |
| "Cannot infer memory degradation" | We don't; we test it by intervention and rename by result |
| "No causality from observation" | Replay is interventional; observational parts labelled |
| "Covered by HORIZON/Khanal/Sinha" | Table in §2; each named and differentiated in the intro |
| "Repackaged survival analysis" | Survival is the descriptive layer; shares are the contribution |
| "Benchmark stale" | AppWorld + SWE-bench Verified; no AgentBench |
| "No intervention or mechanism" | Three interventions, three mechanisms |
| "Lacks theory" | Short identification proposition; not a theory paper |
| "Doesn't generalize across architectures" | Scaffold ablation (verifier) reported; claim scoped to ReAct-class agents |
| "Open-weight only; frontier may differ" | Scope stated; Sinha shows thinking models reduce self-conditioning — we test that axis |

---

## 32. Five Simulated ICLR Reviews (of the redesigned paper)

**R1 — ML theory/statistics.** *Summary:* an SCM for agent loops, three do-interventions identifying self-conditioning, staleness and irrecoverability shares of the horizon gap; discrete-time competing-risk hazards. *Strengths:* clear estimands; interventions rather than labels; sensible discrete-time modelling. *Weaknesses:* identification assumes the oracle state summary doesn't itself alter policy behaviour beyond the belief channel (a "placebo summary" control is needed); shares are order-dependent unless Shapley is run; spline df arbitrary. *Novelty:* moderate-high — the decomposition is new; the estimators are borrowed. *Score:* 6. *Confidence:* 4. *Required:* placebo-summary control; Shapley or full factorial of interventions; sensitivity to spline df.

**R2 — agent systems.** *Summary:* as above. *Strengths:* directly actionable — tells builders whether history hygiene or state re-observation matters. *Weaknesses:* ReAct-only; no context-compression policy; AppWorld is single-user API-world, not browser/OS; wants a coding domain with replays. *Score:* 6. *Confidence:* 4. *Required:* one scaffold variant (verifier or summarizer) as an intervention; discussion of SWE-bench replays as future work is not enough — do 20.

**R3 — benchmark/evaluation.** *Summary:* as above. *Strengths:* reuses HORIZON's taxonomy instead of a new one; validates judge by replay — a nice methodological idea. *Weaknesses:* 40 scenarios is thin; H* from reference solutions may be non-unique; wants pass^k reported. *Score:* 7. *Confidence:* 4. *Required:* scenario count ≥60 or CIs shown everywhere; H* sensitivity; pass^k table.

**R4 — skeptical senior.** *Summary:* "Sinha et al. with tools." *Strengths:* admits the shares result is not in Sinha. *Weaknesses:* if θ_self dominates, the paper is a confirmation; if θ_irrecov dominates, it's Lu et al. quantified; wants a result that could not be predicted from either — e.g., the "forgetting is not forgetting" finding, or a scaling reversal. *Score:* 5. *Confidence:* 5. *Required:* pre-registered predictions and at least one surprising cross-mechanism result; frontier-model spot check on 100 trajectories via API.

**R5 — broad ML.** *Summary:* as above. *Strengths:* readable, well-scoped, reproducible. *Weaknesses:* open-weight only; some figures dense. *Score:* 7. *Confidence:* 3.

**AC discussion.** Consensus that the interventional decomposition is a genuine step beyond HORIZON/Khanal/Sinha; R4's concern decides the outcome: the paper is accepted if it contains a pre-registered, non-obvious mechanism result (the re-interpretation of "catastrophic forgetting" as self-conditioning, or a size/reasoning interaction on θ) and the placebo control; otherwise borderline. Recommendation: accept (poster) conditional on placebo control and factorial interventions; without them, reject with encouragement.

---

## 33. Figure Plan

1. **Horizon gap:** P_obs(s) vs. P_ind(s) with CIs, per model — the problem.
2. **Positional hazard** h(u) with competing risks (recovered vs. terminal stacked), per model, base condition.
3. **Hazard under intervention:** same as 2 with three intervention overlays — the core result.
4. **Decomposition shares** θ by model (stacked bars) with CIs, and vs. model size.
5. **Propagation kernel** π(e,u): heatmap error category × position, from replays.
6. **Recovery function** R(e,u) by model — is recovery agent- or error-driven.
7. **Short-vs-long residual** by subtask position and order ablation.
8. **Judge validity:** fraction of judge-labelled decisive errors with π̂>0, by category.
9. (Appendix) Synthetic control results; H_0.5 under intervention.
Explicitly *not* included: an "early-warning" figure, a "failure geometry" radar chart.

---

## 34. Table Plan

T1 Closest-work comparison (the §2 matrix). T2 Environments, H*, checkpoints. T3 Models, formats, settings. T4 Event counts by tier and category. T5 Δ(s) with CIs. T6 Hazard GLMM coefficients. T7 θ shares with sequential and Shapley variants. T8 π(e,u) summary. T9 Judge validation (κ, causal validity). T10 Ablations. T11 Compute/cost.

---

## 35. Reproducibility Plan

Release: trajectory schema (§20) and all JSONL logs; harness code pinned (vLLM version, model revisions/hashes, chat templates); AppWorld version and task-chaining scripts; seed lists; replay tool (branch from any logged step); judge prompts and human labels; analysis notebooks (R/Python) producing every figure from the JSONL; docker image; pre-registration document (hypotheses, primary tests, kill conditions) posted to OSF before the main runs. Version-pin everything including tokenizers. Report completion rate (fraction of planned episodes that ran without infrastructure error), following Khanal's validity point `[LIT]`.

---

## 36. Research Deliverable Phases

**Phase 0 — Novelty resolution.** *Objective:* fix the claim. *RQ:* what exactly is not in HORIZON/Khanal/Sinha/Lu/CAR/Rabanser? *Inputs:* this audit. *Tasks:* read the six papers fully; write the differentiation paragraph; pre-register. *Deliverables:* literature matrix; differentiation paragraph; pre-registration draft; terminology map (no "failure geometry", no ASI). *Acceptance:* two independent readers agree the differentiation is precise for each of the five closest papers. *Failure:* differentiation rests on benchmark/model choice only → abandon. *Gate:* proceed only if acceptance met.

**Phase 1 — Infrastructure.** *Objective:* deterministic harness + replay. *Inputs:* AppWorld, vLLM, Qwen3-8B. *Tasks:* harness; logging schema; replay-from-step; checkpoint hooks; rule labeller. *Deliverables:* 100 logged trajectories; replay reproduces a logged trajectory bit-exactly under fixed seed. *Acceptance:* 100% replay fidelity on 20 trajectories. *Failure:* nondeterminism >2% → fix env or switch to τ²-bench. *Dependency:* Phase 0.

**Phase 2 — Synthetic validation of estimators.** *Objective:* recover planted mechanisms. *Deliverables:* toolified execution task; planted-irrecoverability results; estimator bias/variance tables. *Acceptance:* shares recover planted ground truth within CI. *Failure:* estimators biased → revise SCM/estimators before real data. *Dependency:* Phase 1.

**Phase 3 — Matched short/long runs (MVE).** *Deliverables:* atomic p̂_i; composed runs s=1..3 for Qwen3-8B/32B; Δ(s); h(u). *Acceptance:* Δ(3) CI excludes 0 for at least one model. *Failure:* Δ≈0 → negative-result pivot (§30). *Gate:* decides whether replays are worth running.

**Phase 4 — Interventions.** *Deliverables:* replays; θ shares; π(e,u); R(e,u). *Acceptance:* at least one share CI excludes 0. *Failure:* none identified → positioning #2. *Dependency:* Phase 3.

**Phase 5 — Annotation & validation.** *Deliverables:* judge labels; 200 human labels; κ; causal validity rate. *Acceptance:* κ ≥ 0.5 or judgement-required categories dropped with justification.

**Phase 6 — Scale & ablate.** *Deliverables:* remaining models; reasoning on/off; SWE-bench observational; Shapley; placebo control. *Acceptance:* main shares stable in sign across seeds and models.

**Phase 7 — Write-up.** *Deliverables:* paper; artefacts; pre-registration reconciliation. *Acceptance:* every figure regenerates from released logs.

---

## 37. Agent-Ready Task Specification (atomic tasks for Phases 1–4)

**T1.1** *Objective:* implement ReAct harness with native tool calling for Qwen3 via vLLM. *Why:* single controlled harness. *Input:* AppWorld API spec; tool schemas. *Action:* write `harness.py` exposing `run(task_id, model, seed, step_limit) -> JSONL`. *Output:* JSONL per §20. *Acceptance:* 10 tasks run end-to-end; every step logs all schema fields non-null. *Failure:* any missing field. *Dependency:* none. *Escalation:* vLLM tool-call parsing errors >5% → switch parser.

**T1.2** *Objective:* replay-from-step. *Input:* JSONL run. *Action:* implement `replay(run_id, t)` that resets AppWorld, re-executes actions 1..t−1, returns env handle. *Output:* env state hash equal to logged `env_state_hash[t−1]`. *Acceptance:* 20/20 hash matches. *Failure:* mismatch → log diff, escalate.

**T1.3** *Objective:* rule labeller. *Input:* JSONL. *Action:* implement the seven rule events (§28 tier 1). *Output:* `event_labels[]` populated. *Acceptance:* unit tests on 30 hand-made cases pass. *Dependency:* T1.1.

**T1.4** *Objective:* checkpoint hooks. *Action:* after each subtask boundary, run AppWorld checks; write `checkpoint_results`. *Acceptance:* checks agree with final evaluator on 50 runs.

**T2.1** *Objective:* toolified execution synthetic task. *Action:* KV-store tools; 200-step sequences; inject (a) erroneous history, (b) stale store. *Output:* per-step accuracy curves under each injection. *Acceptance:* injected-history effect replicates Sinha's direction on Qwen3-8B. *Failure:* no effect → check prompt fidelity, escalate.

**T2.2** *Objective:* planted irrecoverability. *Action:* insert `delete` at u∈{0.2,0.5,0.8}; run oracle-fix replays. *Acceptance:* π̂ ≈ 1 for delete, ≈ 0 for read errors, CIs disjoint.

**T3.1** *Objective:* atomic runs. *Action:* 40 scenarios × subtasks × 3 seeds × {8B,32B}. *Output:* p̂_i table with Wilson CIs. *Acceptance:* every subtask has ≥3 runs per model.

**T3.2** *Objective:* composed runs. *Action:* chain subtasks 1..s, s=1..3, 3 seeds, both models; half scenarios reversed. *Output:* trajectories; Δ(s). *Acceptance:* ≥95% episode completion (no infra errors).

**T3.3** *Objective:* hazard fit. *Action:* cloglog GLMM per §27. *Output:* coefficients, curves. *Acceptance:* model converges; diagnostics reported.

**T4.1** *Objective:* select replay points. *Action:* for each failed s≥2 trajectory, first two events with tier-1/2 labels. *Output:* replay manifest. *Acceptance:* manifest ≤ budget.

**T4.2** *Objective:* run three interventions, k=4. *Action:* per manifest, branches per §26. *Output:* continuation outcomes. *Acceptance:* all branches logged with `parent_run_id`.

**T4.3** *Objective:* estimate θ, π, R. *Action:* mixed-effects logistic; bootstrap. *Output:* T7, T8, F3–F6. *Acceptance:* CIs reported for every estimate. *Escalation:* any share CI spans [−0.2, 0.2] → flag for Phase 6 scaling before interpretation.

---

## 38. Three Versions of the Paper

**Version A — Minimal.** *RQ:* does self-conditioning explain the horizon gap in a tool-using agent? *Contribution:* history-scrub intervention + horizon gap on AppWorld. *Metrics:* Δ(s), h(u), θ_self. *Benchmarks:* AppWorld. *Models:* Qwen3-8B, 32B. *Experiments:* §22. *Expected:* θ_self substantial and size-dependent. *Risk:* "Sinha with tools" (R4); workshop-level unless effect is striking.

**Version B — Strong ICLR (recommended).** *RQ:* §6 primary + RQ-A. *Contribution:* three-mechanism interventional decomposition; positional competing-risk hazard; replay-validated labels. *Metrics:* Δ, h, θ (3), π(e,u), R(e,u). *Benchmarks:* AppWorld + SWE-bench Verified (observational). *Models:* Qwen3 ×3, Llama-70B, Mistral-24B; reasoning ablation. *Experiments:* §21 + placebo control + factorial interventions. *Expected:* mechanism shares differ by error type and shift with scale/reasoning; "forgetting" reinterpretation. *Risk:* replay budget; judge κ.

**Version C — Ambitious.** *Adds:* replays on SWE-bench Verified (20–50 instances); a frontier-model API spot check; scaffold interventions (verifier, summarizer) as a fourth mechanism (recovery capacity); an "empirical law" section if shares are stable; released replay toolkit generalizing CAR to population estimates. *Risk:* scope creep; SWE-bench replay cost; law may not exist.

---

## 39. Potential Titles

1. Why Long-Horizon Agents Fail: An Interventional Decomposition of the Horizon Gap
2. Self-Conditioning, Staleness, or Irrecoverability? Attributing Long-Horizon Agent Failure by Counterfactual Replay
3. Beyond the Curve: Causal Shares of Horizon-Induced Degradation in Tool-Using LLM Agents
4. Positional Hazard and Recoverability in LLM Agent Trajectories
5. Forgetting That Isn't: Replay Evidence on the Mechanisms of Long-Horizon Agent Breakdown
6. The Horizon Gap: Separating Task Difficulty from Trajectory-Induced Failure in Agents
7. Which Errors Become Terminal? Propagation Kernels for LLM Agent Trajectories
8. Do Agents Fail Independently? Testing the Geometric Null with Matched Short-and-Long Tasks
9. Mechanism Attribution for Long-Horizon Reliability of Open-Weight Agents
10. From Failure Labels to Failure Causes: Interventional Reliability Analysis of LLM Agents

---

## 40. Final Recommendation

**If I were leading this research project, I would** stop the current plan this week; drop AgentBench, WebArena, SWE-bench Lite, the nine-category taxonomy, the Agent Stability Index, the early-warning component, and the phrase "failure geometry"; spend Phase 0 writing the one paragraph that distinguishes the work from HORIZON, Khanal et al., Sinha et al., Lu et al., and Causal Agent Replay; build the deterministic AppWorld harness with bit-exact replay; validate the three interventions on a synthetic toolified task; run the matched short/long sweep on Qwen3-8B and 32B; and only if Δ(3) is significantly negative, spend the replay budget to estimate the self-conditioning, staleness and irrecoverability shares. The paper I would submit is Version B, pre-registered, with the headline being a mechanism share result and — if the data cooperate — the finding that much of what current taxonomies call catastrophic forgetting is the agent conditioning on its own mistakes while the constraint sits, unread, in context.

---

## References (primary sources cited above)

- Wang, Bai, Sun, … Song, Nowak. The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break (HORIZON). arXiv:2604.11978, COLM 2026.
- Khanal, Tao, Zhou. Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents. arXiv:2603.29231, 2026.
- Sinha, Arun, Goel, Staab, Geiping. The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs. arXiv:2509.09677, 2025 (OpenReview 3lm8lWYxiq).
- Rabanser, Kapoor, Kirgis, Liu, Utpala, Narayanan. Towards a Science of AI Agent Reliability. arXiv:2602.16666, ICML 2026.
- Lu et al. Where LLM Agents Fail and How They Can Learn From Failures. arXiv:2509.25370, 2025.
- Shah. Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures. arXiv:2606.08275, 2026.
- CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures. arXiv:2605.25338, 2026.
- AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems. arXiv:2605.08715, 2026.
- Real-Time Detection and Repair of LLM Agent Failures. arXiv:2608.02464, 2026.
- Pushkin, Abbe. LEAD: Breaking the No-Recovery Bottleneck in Long-Horizon Reasoning. arXiv:2603.06870, 2026.
- Kwa et al. (METR). Measuring AI Ability to Complete Long Tasks. arXiv:2503.14499, 2025.
- Yao et al. τ-bench. arXiv:2406.12045, 2024.
- Laban et al. LLMs Get Lost in Multi-Turn Conversation. arXiv:2505.06120, 2025.
- Cemri et al. Why Do Multi-Agent LLM Systems Fail? arXiv:2503.13657, 2025.
- Zhang et al. Which Agent Causes Task Failures and When? (Who&When). arXiv:2505.00212, 2025.
- Beyond Resolution Rates: Behavioral Drivers of Coding Agent Success and Failure. arXiv:2604.02547, 2026.
- Beyond the Leaderboard: A Synthesis of Tool-Use, Planning, and Reasoning Failures in LLM Agents. arXiv:2607.05775, 2026.
- Kapoor et al. Holistic Agent Leaderboard. arXiv:2510.11977, ICLR 2026.
- Trivedi et al. AppWorld. arXiv:2407.18901, 2024.
- Xu et al. LongDS-Bench. arXiv:2605.30434, 2026. Li et al. Long-Horizon-Terminal-Bench. arXiv:2607.08964, 2026.
- Zhang et al. MIRAGE-Bench. arXiv:2507.21017, 2025. Liu et al. Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents. arXiv:2608.01913, 2026.
- Failure-Based Testing for Deep RL Agents. arXiv:2606.31372 (use of "geometric" failure patterns). Interpretable Failure Analysis in MARL. arXiv:2602.08104.
