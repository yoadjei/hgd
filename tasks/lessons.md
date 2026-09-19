# Lessons

Corrections that changed how this codebase is built. Each entry is a rule that
would have prevented the defect, applied to all later work.

## 1. Verify third-party APIs against the installed package, not the docs

**What went wrong.** The AppWorld adapter was written from documentation. The
first run against the real package showed the evaluation payload key is
`failures`, not `fails`; `ground_truth` is an object, not a mapping; and
`api_calls` are HTTP record dicts, not executable actions. Each defect was
silent: the digest dropped the failure vector, H\* raised on every task, and the
gate would have reported 20/20 while executing nothing.

**Rule.** Before any adapter is trusted, run a probe that asserts every assumed
attribute, key and type against the installed version and records what it saw.
Pin the verified shape in a test fake with the date and package version.

## 2. A validation that cannot fail validates nothing

**What went wrong.** The first Phase 2 run recovered both planted shares with
zero error. That was tautological: scrubbing drove the error count to zero,
which made the intervened arm bit-identical to the counterfactual arm. The
planted `store(k, WRONG)` also succeeded silently, so the self-conditioning
term never fired and every unit test still passed.

**Rule.** For every estimator validation, plant a condition that violates its
identification assumption and check the bias is measurable. For every planted
mechanism, add an end-to-end test that the mechanism is actually visible in a
real harness run, not only in unit arithmetic.

## 3. Reassuring outputs need a no-op guard

**What went wrong.** The determinism gate compared two replays of the same
actions. Two no-ops match perfectly, so a broken action source produced the
gate's most reassuring answer.

**Rule.** Any check that compares two outcomes for equality must also confirm
the outcomes moved away from the untouched baseline.

## 4. Consistency tests must parse, not grep

**What went wrong.** The test that the gate script imports nothing from the
package matched the substring `from hgd` inside a comment.

**Rule.** When a test constrains source structure, parse the AST.

## 5. Probe before the GPU, and hand over runnable cells

**What went wrong.** A mangled line reached the user in a paste-able cell, and
GPU quota would have been spent on a gate that needed no GPU.

**Rule.** Anything handed over to run elsewhere is syntax-checked and
lint-checked first. Determinism and census work run on CPU before any
inference quota is committed.
