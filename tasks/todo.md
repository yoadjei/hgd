# Task plan

Objective: determine whether the interventional decomposition of the horizon gap in long-horizon
tool-using LLM agents is valid, identifiable, novel and useful — per `docs/prompt.txt` (contract)
and `docs/failure_geometry_research_audit.md` (specification).

Progression is fixed by audit §36 and prompt §30 and must not be reordered:
**novelty → infrastructure → estimator validation → horizon gap → interventions → annotation →
scaling → write-up.**

---

## Phase 0 — Novelty resolution  *(in progress)*

- [x] Verify every CRITICAL/HIGH threat in audit §3 against primary sources
- [x] Adversarial search for a paper that already owns the interventional decomposition
- [x] Record verification log → `docs/phase0_verification.md`
- [x] Literature matrix with the prompt §25 fields → `docs/lit_matrix.csv`
- [x] Terminology map → `docs/terminology_map.md`
- [x] Compute and floor-effect feasibility → `docs/feasibility.md`
- [x] Machine-readable research state → `docs/research_state.json`
- [ ] **Differentiation paragraph, written against seven closest papers** (was five)
- [ ] **Preregistration draft** — hypotheses H1–H8, five primary tests, Holm correction, kill
      conditions, protocol-sensitivity plan (D3), exposure conditioning (D1)
- [ ] Close outstanding full-text verification: Lu 2509.25370, CausalFlow 2605.25338,
      Rabanser 2602.16666, TraceToChain 2604.24579 *(blocker B5)*
- [ ] Confirm HORIZON's venue status before citing it as COLM 2026

**Acceptance:** two independent readers agree the differentiation is precise for each closest paper.
**Structurally unmet — there is one executor** *(blocker B4)*. Mitigation: write the differentiation
so that it survives the strongest available adversarial reading, and record the seven concessions in
`docs/phase0_verification.md` §4 openly rather than defending them.

**Gate verdict so far: CONDITIONAL PASS.** Kill condition E is not triggered — no single paper
duplicates the contribution — but the gap is narrower than the audit asserts.

---

## Phase 1 — Infrastructure  *(in progress)*

Built test-first against a narrow `Environment` protocol so replay and labelling are verifiable
without a GPU or an AppWorld install. **41 tests passing, `filterwarnings = ["error"]`.**

- [x] Verify AppWorld determinism/reset/checkpoints/reference solutions *(blocker B1 downgraded —
      `save_state`/`load_state` exist, reset <0.5s, reference solutions on train/dev only,
      stage checkpoints must be built from `evaluation_code`, determinism still unmeasured)*
- [x] Logging schema per audit §20 → `src/hgd/schema.py`, 5 tests. Invalid records cannot be
      constructed, so a missing field fails at collection time rather than at analysis time.
- [x] Rule labeller — the seven tier-1 events → `src/hgd/rule_labeller.py`, 26 tests
      *(T1.3 acceptance was 30 hand-made cases; the suite has 31 total)*
- [x] `Environment` protocol + deterministic `DictEnvironment` → `src/hgd/env.py`.
      Doubles as the audit §24 toolified KV store, with an irreversible `delete` for planting
      known propagation ground truth in Phase 2.
- [x] `replay_prefix` + `verify_fidelity` → `src/hgd/replay.py`, 10 tests. Gate is exact 100%,
      not a tolerance; a deliberately nondeterministic environment is proven to fail it.
- [x] Action parsing, both formats → `src/hgd/parsing.py`, 16 tests. Distinguishes *no action
      attempted* from *action attempted and unparseable*; audit §20 sets retries to zero, so the
      second is a tier-1 event and collapsing them would erase it.
- [x] `harness.py` — ReAct loop, zero retries, step limit 3·H\*, u = step/H\*, 12 tests.
      Includes the integration test that a harness trajectory passes the replay gate.
- [x] `model.py` — `Model` protocol + `ScriptedModel`. Not a test double: replay branches are a
      scripted policy by definition, and Phase 2's planted conditions are scripted by construction.
- [x] Checkpoint hooks and recoverability → `src/hgd/checkpoints.py`, 16 tests. Includes
      `first_unrecovered_error`, the survival event the hazard model is built on.
- [x] H\* derivation and normalized position → `src/hgd/horizon.py`, 10 tests. H\* is a *named
      strategy* (`api_calls` on train/dev, `solution_lines` everywhere) so R3's demanded H\*
      sensitivity analysis is a change of argument, not a rewrite.
- [x] AppWorld adapter implementing `Environment` → `src/hgd/appworld_env.py`, 13 tests.
      Written against the API verified from AppWorld's docs, tested against a fake implementing
      that surface. **The state digest is injected, not guessed** — AppWorld exposes no state
      hash, and the choice between a behavioural digest (unit-test outcome vector) and an
      exhaustive one (database bytes) is a real methodological decision.
- [x] Phase 1 gate script → `notebooks/phase1_gate.py`. Probes the real API, compares both digest
      strategies, runs the fidelity gate. **Needs no GPU**: replay re-executes logged actions, so
      determinism is settled before any inference quota is spent, using released gold solutions
      as the action source.
### Defects found by the first real run against AppWorld (6 Sep 2026)

The probe did its job. Every one of these would have corrupted results silently.

| # | Defect | Consequence if unfixed | Status |
|---|---|---|---|
| R1 | `evaluation_digest` read key `fails`; the real key is `failures` | `.get()` returned `None`, so the **entire failure vector was dropped from the hash**. Two states differing only in which tests failed would hash identically. | fixed + regression test |
| R2 | Gate used `ground_truth.api_calls` as actions | They are HTTP record dicts, which `execute()` evaluates as no-op literals. Nothing ran; both replay arms matched trivially; **the gate would have printed 20/20 PASS while measuring nothing**. | fixed — uses `compiled_solution_code` |
| R3 | No no-op guard in the gate | A pair of no-ops matches perfectly, so the gate returned its most reassuring answer precisely when broken. | fixed — state must move vs a fresh world, else `no_effect` |
| R4 | `intrinsic_horizon` assumed mapping access | `ground_truth` is a `GroundTruth` object; H\* would have raised on every real task. | fixed — object or mapping |
| R5 | Harness terminated on exact action string | AppWorld signals completion via `apis.supervisor.complete_task()` inside a code block, and that call must *execute*. Every episode would have run to the 3H\* limit, inflating cost and stamping a step-limit event on finished trajectories. | fixed — sentinel path + `env.task_completed()` |
| R6 | `evaluate()` prints a full report per call | Called once per step for `state_hash`/`snapshot`; one episode buries its own log, and the I/O is a real slowdown. | fixed — suppressed in adapter and gate |

### Defects found by the second gate run and the H\* census (20 Sep 2026)

| # | Defect | Consequence if unfixed | Status |
|---|---|---|---|
| R7 | `compiled_solution_code` is a `def solution(apis, requester)` wrapper, not a script | `execute()` defines the function and never calls it, so the world never moves. The no-op guard caught it; without the guard the gate would have reported 20/20 a second time. | **open** — `notebooks/solution_source_probe.py` enumerates the real invocation |
| R8 | `num_solution_code_lines` is exactly 3 on all 90 train tasks | A constant cannot measure solution length. `HStarStrategy.SOLUTION_LINES` would set H\* = 3 for every task on held-out splits, where it is the *only* available strategy, making u meaningless exactly where `api_calls` is withheld. | fixed — prefers `num_compiled_solution_code_lines` (18–86, median 37), 2 regression tests |

### Built since

- [x] `logging.py` — append-only JSONL, flushed per trajectory, grouped by `run_id`, 9 tests.
      A session killed at the 9-hour limit keeps every episode it finished.
- [x] `outcomes.py` — `task_success` / `pass_fraction` / `checkpoint_vector`, 14 tests.
      Bridges AppWorld evaluation to the survival layer. **The graded view exists but is
      not adopted** — swapping the estimand is a §5 scope change, decided by the Phase 3a
      pilot, not by an import.
- [x] `vllm_client.py` — `Model` over the OpenAI-compatible API, injected transport, 16 tests.
      HTTP rather than in-process so the server batches concurrent episodes; that batching is
      what makes the budget feasible.
- [ ] **RUN the gate** ← the actual acceptance test. Two attempts so far, both
      void rather than failed. Run 1 executed `api_calls` (no-op literals); run 2
      executed `compiled_solution_code` (a function definition, never called).
      Blocked on R7. **Determinism remains unmeasured.**
- [ ] Rerun the gate on model-generated trajectories (gold solutions never hit error paths)
- [ ] vLLM client implementing `Model` (not needed for the gate; needed for Phase 2 onward)
- **Acceptance:** 100% replay fidelity on 20 real AppWorld trajectories.
- **Failure:** nondeterminism >2% → fix env or switch to τ²-bench (audit's designated fallback).
- **Status: NOT PASSED.** Everything above is infrastructure that passes its own tests. The gate
  has not been run against AppWorld. That distinction is the whole of kill condition C and must
  not be blurred into "Phase 1 complete" in any later claim.

---

## Phase 2 — Synthetic validation of estimators  *(not started)*

Cheap, environment-trivial, runs on the smallest model. Validates the estimators against planted
ground truth **before** any AppWorld budget is committed.

- [ ] Toolified execution task — `store(k,v)`/`read(k)` over an external KV store, steps 10–200
- [ ] Inject erroneous history (self-conditioning) vs stale stored state (staleness)
- [ ] Plant an irreversible `delete` at u ∈ {0.2, 0.5, 0.8}
- **Acceptance:** shares recover planted ground truth within CI; π̂ ≈ 1 for delete, ≈ 0 for read
  errors, CIs disjoint.

---

## Phase 3a — Atomic feasibility  *(new; not in the audit)*

Inserted because **blocker B2**: Qwen3-8B scores 5.4% TGC on native AppWorld tasks, which would make
Δ(s) a log-ratio of two near-zero probabilities and render the whole estimand meaningless. See
`docs/feasibility.md` §2.

- [ ] Estimate p̂_i for candidate atomic subtasks, 3 seeds, smallest viable model
- **Accept:** ≥30 subtasks with median p̂_i ≥ 0.5, lower-quartile ≥ 0.35, implied P_ind(3) ≥ 0.05
- **Fail:** escalate the scope decision — hosted API for larger models, τ²-bench, or report
  infeasible at this budget. Do not proceed and hope.
- Cost: ~2–4 GPU-hours. Highest information per GPU-hour in the programme.

### H\* census, all 90 train tasks (20 Sep 2026) → `hstar_census.json`

| measure | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| `api_calls` | 5 | 18 | 36 | 70 | 244 |
| `num_compiled_solution_code_lines` | 18 | 28 | 37 | 46 | 86 |
| `num_solution_code_lines` | 3 | 3 | 3 | 3 | 3 — degenerate, see R8 |

**The audit's 3·H\* step limit is affordable and is retained.** At the median,
2·H\* is 72 steps and 3·H\* is 108; the 810-episode MVE costs 17.5M and 26.2M
output tokens respectively, which is 9.7 h and 14.6 h on 2×T4 at 500 tok/s, or
0.3 and 0.5 weeks of Kaggle quota. The earlier worry that H\* = 71 on the first
task threatened the budget was wrong — that task sits at p75, not the median.

Caveat on the estimate: it counts output tokens only. Prefill grows with the
transcript over 108 steps, and T4 (compute capability 7.5) cannot run
FlashAttention-2, so vLLM falls back to xformers. Treat 14.6 h as a floor and
budget two to three times it. Still inside one week of quota.

**Pilot set:** the 30 cheapest tasks have H\* from 5 to 22, median 11, starting
`e85d92a_1/_2/_3`, `cf6abd2_1/_2/_3`, `60d0b5b_1/_2/_3`. Note the `_1/_2/_3`
suffixes: AppWorld ships three instances per scenario, which is what the matched
short/long design needs for scenario pairing.

---

## Phase 3 — Matched short/long  *(blocked by 3a)*

- [ ] Atomic runs → p̂_i with Wilson CIs
- [ ] Composed runs s = 1..3, shared state, half the scenarios order-reversed
- [ ] Δ(s) by paired bootstrap over scenarios; protocol-sensitivity analysis (D3)
- [ ] h(u) cloglog GLMM with competing risks
- **Gate:** Δ(3) CI excludes zero for ≥1 model, else negative-result pivot (audit §30).

---

## Phases 4–7  *(not started)*

4 interventions → θ, π(e,u), R(e,u) · 5 annotation and judge validation · 6 scale and ablations ·
7 write-up. Expand when Phase 3 passes; do not plan in detail before then.

---

## Open decisions

1. **Model set** *(blocker B3)* — the audit's Qwen3-8B/32B/235B + Llama-3.3-70B + Mistral-24B is not
   executable on Kaggle 2×T4. Proposed: Qwen3-4B/8B/14B/30B-A3B + Mistral-Small-24B. Method change,
   not a scope change under prompt §5. **Awaiting user decision.**
2. **Fallback if 3a fails** — decide *after* the pre-gate produces a number, not before.
3. **Step limit** — ~~resolved~~ the census says 3·H\* is affordable; the audit's value stands.
4. **Platform** — Colab or Kaggle. The install log showed `google-colab` and a
   `/usr/local/lib/python3.12/dist-packages` path, which are Colab markers. It matters
   because Colab free gives one T4, not two, so `--tensor-parallel-size 2` would fail at
   server startup and the census throughput figure would halve. **Awaiting confirmation.**

---

## Lessons

Recorded in `tasks/lessons.md`.
