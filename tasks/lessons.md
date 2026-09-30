# Lessons

Corrections that changed how this codebase is built. Each entry is a rule that
would have prevented the defect, applied to all later work.

## 0. Do not ship defects. Reproduce first, then fix

This one supersedes the rest, because it is the pattern the rest are instances of.

**What went wrong.** Eight consecutive failures against the real package shipped
with a fully green test suite, and four of the last five were defects introduced
while fixing the previous one. Every one was found by a run on the user's machine
rather than by a test here. The suite stayed green because **every test double
encoded the same wrong assumption as the code it tested**, so no test was capable
of failing:

| Real behaviour | What the double did instead | Bug it could not catch |
|---|---|---|
| pass/fail entries are dicts after completion | returned strings | three: `sorted()` raising in the adapter, `checkpoint_vector` raising on an unhashable key, and a checkpoint predicate silently returning False |
| reporter holds the stream from before any redirect | called `print()` | every real report leaking through two redirect layers |
| `execute()` installs a guard: `faulthandler.enable()`, read-only `open()` | no guard at all | a StringIO sink with no descriptor breaking every action |
| a fresh install resolves extras atomically | never installed anything | `vllm`'s source build taking `appworld` down with it |

**Rule.** A fix for an externally-observed failure starts with a local
reproduction that fails. Build doubles from observed behaviour of the real system,
never from the assumption the code makes. Before fixing, ask what the double would
have to do for this bug to show, make it do that, and watch it fail first. When a
fix rests on a theory, say so and design the test to falsify the theory. A green
suite across a real failure is a defect in the suite, so fix the double too.

**Applied.** The `faulthandler` and read-only `open()` failure was fixed against a
fake that installs both, and the fix was confirmed to turn a failing reproduction
green before it shipped. Two earlier silencing fixes shipped on theories and both
were wrong.

## 0b. Read the dependency's source before theorising about its behaviour

**What went wrong, nearly.** Two anomalies came back from a real run: `close()`
raising inside appworld's time freezer, and pass fraction falling 1.0 to 0.5. The
obvious move was to write a defensive `close()` and a guard against calling
`complete_task()` twice. Both would have been wrong. Downloading the
`appworld==0.1.3.post1` wheel and reading forty lines of it showed a single cause:
`load_state()` calls `AppWorld.close_all()`, which stops the task's time freezer
and restarts nothing, so the world silently continues on wall-clock time and the
next `close()` double-stops the same freezegun instance. The score drop was a
downstream symptom of the unfrozen clock, not a second defect. A guard against
double completion would have been a fix for a symptom, hiding a determinism break.

**Rule.** When a dependency misbehaves, read its source before designing the fix.
`pip download --no-deps` plus unzip is cheaper than one wrong fix, and far cheaper
than a fix that hides the real cause. Reproduce the mechanism locally against the
dependency's real internals — here, real freezegun plus appworld's wrapper copied
verbatim — so the reproduction cannot encode the same assumption the fix does. Two
symptoms with one cause is the common case, and patching them separately leaves the
cause in place.

## 0c. A confirmed root cause is not a licence to explain the next symptom with it

**What went wrong.** Two anomalies came back together: `close()` raising, and pass
fraction dropping 1.0 to 0.5. Reading the source found a real defect that explained
the first, and I used it to explain the second as well — the unfrozen clock must
have made `complete_task()` write wall-clock timestamps that a validator rejected.
It was coherent, it fitted every observation, and it was wrong. Measured on a fresh
world with the clock verified frozen and no `load_state` anywhere, a second
`complete_task()` still drops pass fraction to 0.5. Completion is simply not
idempotent.

An adversarial audit agreed with me, which made it worse. It reported the
alternative as "not supported by the source" while also recording that appworld's
supervisor app ships pre-compiled and could not be read. I took the conclusion and
discounted the caveat that invalidated it.

**Rule.** One confirmed cause explains one symptom. A second symptom needs its own
measurement, especially when the shared explanation is elegant — elegance is what
makes it persuasive, not what makes it true. When a source-reading exercise reports
that the deciding code could not be read, that is an unanswered question, not a
refutation. Keep the check that measures it and let it run.

**Applied.** The probe kept measuring the double completion on a fresh world
instead of asserting the tidy story, which is the only reason this was caught
before the pilot ran on it.

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

**It happened again, in the guard's own probe.** The adapter probe checked that
`task_completed()` flips after `complete_task()` and reported `True -> True` as a
PASS. It asserted only the final value, never the baseline, and the gold solution
run earlier in the same world had already completed the task. So the check proved
nothing and said the assumption held. The rule above is not only for gates and
estimators — it binds hardest on the checks written to verify something, because
those are the ones whose PASS gets believed. Each scenario now gets a fresh world
and the check asserts the baseline first.

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

## 6. A diagnostic must survive the failure it is diagnosing

**What went wrong.** The adapter probe ran twelve checks against the real package
and then died in teardown, inside appworld's time freezer, on `close()`. It wrote
its record *after* the teardown, so every finding was lost and the only evidence
was the console scrollback. A second defect was hidden by the same structure: one
long-lived world meant the failing `close()` could not be attributed to the call
sequence that caused it.

**Rule.** A probe writes its record in a `finally`, before anything that can
raise. Teardown that can fail is a check, not teardown. When a probe exists to
find which operation breaks something, give each candidate sequence its own
fixture, or the answer is one bit wide.
