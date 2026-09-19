# Gate Report — Phase 2 (Synthetic validation of estimators)

**Date:** 6 September 2026. **Format:** prompt §32. **Cost:** CPU, seconds. No GPU, no model.
**Artefacts:** `experiments/phase2_validation.py`, `results/phase2_validation.json`.

---

## Status

**CONDITIONAL PASS.**

T2.2 passes outright. The share-recovery result passes but is **partly tautological**, and that is
recorded here rather than presented as evidence.

---

## Evidence

Planted policy: `base = 0.18`, `self_conditioning = 0.45`, `staleness = 0.30`; 400 scenarios per
cell; horizon levels s ∈ {1,2,3}; matched atomic rates.

**Horizon gap exists in the synthetic world.** Δ(3) = **−0.0793**. Negative, as constructed.

**Share recovery.**

| mechanism | planted (true) | recovered | abs. error |
|---|---:|---:|---:|
| self-conditioning | 0.573 | 0.573 | 0.000 |
| state staleness | 0.309 | 0.309 | 0.000 |
| null control (no mechanism planted) | 0.000 | −0.000 | 0.000 |

**T2.2 — propagation, irreversible delete vs harmless read** (audit acceptance: π̂ ≈ 1 for delete,
≈ 0 for read, CIs disjoint):

| u | delete | read |
|---|---|---|
| 0.2 | **+1.000** [+0.987, +1.000] | +0.000 [−0.010, +0.010] |
| 0.5 | **+1.000** [+0.987, +1.000] | +0.000 [−0.010, +0.010] |
| 0.8 | **+1.000** [+0.987, +1.000] | +0.000 [−0.010, +0.010] |

CIs disjoint at every position. **T2.2 PASSES.**

**Identification probe.** With a policy also sensitive to transcript *length*
(`length_sensitivity = 0.004`), which violates history-scrub's identification assumption.
True θ_self = **+0.532**:

| scrub mode | θ_self recovered | bias |
|---|---:|---:|
| `remove` (delete the turns) | +0.603 | **+0.071** |
| `length_matched` (D10) | +0.532 | **+0.000** |

**Placebo control for the staleness arm.**

| arm | θ_state |
|---|---:|
| oracle summary | **+0.309** |
| placebo summary (byte- and line-matched, no state) | **−0.000** |

The placebo does not relieve staleness, so in this setting the state-refresh effect is not an
artefact of appending an authoritative-looking message.

---

## Deviations

Two defects were found and fixed during the phase. Both were found by the phase, which is what it is
for.

1. **The planted error action was not an error.** `store(k, WRONG)` succeeded in the environment and
   stored the string, so no error observation ever entered the transcript, `history_scrub` had
   nothing to scrub, and θ_self was unidentifiable — while every unit test passed, because the unit
   tests fed handcrafted transcripts rather than exercising the loop. Fixed by making the planted
   action arity-invalid so the environment rejects it, and covered by end-to-end tests
   (`tests/test_synthetic_loop.py`).
2. **The planted irrecoverability was recoverable.** The first run returned π = 0.000 at every
   position: the policy cycles keys and simply re-stored the deleted one. `delete` was a defect in
   the *plant*, not the estimator, but an environment with no genuinely irreversible operation
   cannot validate the irrecoverability estimator at all. Fixed by adding opt-in tombstones to
   `DictEnvironment` (`tests/test_irreversibility.py`), after which π = 1.000.

---

## Scientific implication

**The estimator arithmetic is correct, and one identification assumption is now quantified.**

The exact-zero share recovery is **not** strong evidence and must not be reported as though it were.
It is close to tautological by construction: scrubbing drives the error count to zero, which makes
the planted policy's error probability *identically* the `self_conditioning = 0` counterfactual, and
with a shared RNG stream the two arms are bit-identical. That result confirms `mechanism_share`
computes what it claims and that `scrub_history` removes what the policy actually reads. It says
nothing about whether the intervention would recover the mechanism in a system where the assumption
fails.

**The identification probe is the informative result.** When the policy is also length-sensitive —
which is the realistic case, since context-length degradation is documented (Peng et al.'s "context
rot", Hong et al.) — history-scrub **overestimates** θ_self by 0.071 on a true value of 0.532, a
13% relative overstatement. The mechanism is straightforward: scrubbing removes turns, which
shortens the transcript, which improves a length-sensitive policy through a channel that has nothing
to do with erroneous history.

This is a real threat to the paper's primary estimand, it was invisible until a probe was built that
could fail, and it has a clean fix.

**D10 — length-matched scrub — implemented and effective.** Instead of deleting the erroneous
turns, their content is replaced with neutral filler matched on byte and line count, exactly as
`placebo_summary` matches the oracle summary. Turn count, position structure and transcript length
are all preserved, so the erroneous-history channel is the only one altered. The measured bias falls
from **+0.071 to +0.000**. This is the direct answer to audit §31's "your causal effect is an
artefact of the intervention", and it costs nothing.

**But D10's exact zero is subject to the same caveat, and introduces a new assumption.** In this
synthetic policy the length term is `length_sensitivity * len(messages)`, and length-matched
scrubbing preserves `len(messages)` exactly, so the term is unchanged by construction. What the
probe demonstrates is that the mechanism *works as designed* where removal demonstrably did not —
not that it neutralises every length-like channel in a real model.

Two reasons it will not transfer cleanly:

1. A language model's sensitivity is to **tokens**, not to message count, and filler text of equal
   byte length is not of equal token length under a BPE tokenizer. Matching should be done on
   tokens for the real runs.
2. **Filler is not behaviourally neutral to a language model.** A transcript containing several
   turns of "This line contains no task-relevant information." is out of distribution and may
   change behaviour on its own. D10 trades a known bias for an untested assumption, which is an
   improvement only if the new assumption is checked — so both scrub modes must be run on the real
   data and reported, with the difference between them as the sensitivity analysis. Reporting only
   the mode that gives the tidier number would be exactly the selective reporting prompt §16 forbids.

**Placebo control passes**, closing the second Phase 2 carry-forward: the byte-matched placebo moves
the gap by −0.000 against the oracle summary's +0.309, so `state_refresh` is identifying a belief
channel rather than a message-appending artefact. Same caveat about transfer applies.

**Scope limit.** Δ(3) = −0.0793 is a property of a policy we wrote. It is evidence about the
estimators, and about nothing else. It is not evidence that a horizon gap exists in AppWorld, and
must never be cited as though it were.

---

## Next permitted phase

**Phase 3a (atomic feasibility) is permitted** once Phase 1's gate has actually run against
AppWorld. Phase 2 validated the estimators against the synthetic environment only; it does not
discharge kill condition C, which is about AppWorld's determinism and remains **unmeasured**.

**Blocked:** Phase 3 composed runs, pending Phase 3a's p̂_i distribution and kill condition F.

**Carry forward:**
1. ~~Implement D10 and re-run the identification probe.~~ **Done** — bias +0.071 → +0.000.
2. ~~Add a placebo probe for the staleness arm.~~ **Done** — oracle +0.309 vs placebo −0.000.
3. **New:** match the scrub filler on **tokens**, not bytes, before the real runs. Byte matching is
   the wrong invariant for a BPE tokenizer.
4. **New:** run *both* scrub modes on real data and report the difference as a sensitivity analysis.
   D10 replaces a measured bias with an untested assumption (that filler is behaviourally neutral to
   a language model, which is not obviously true), so reporting only one mode would be selective.
