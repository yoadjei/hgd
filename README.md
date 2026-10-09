# hgd

Interventional decomposition of the horizon gap in long-horizon tool-using LLM
agents. Research code for an ICLR 2027 submission.

The horizon gap is Δ(s) = log P_obs − Σ log p̂ᵢ: how much worse a composed task
goes than independent per-step errors predict. The question is what the gap is
made of. Three interventions, each a pure function of logged data so that a
released log reproduces the branch, attribute it to three mechanisms:

| Intervention | Holds fixed | Identifies |
|---|---|---|
| `history_scrub` | environment state factual, transcript cleaned | self-conditioning |
| `state_refresh` | transcript factual, oracle state summary added | state staleness |
| `oracle_fix` | prefix factual, one action replaced | irrecoverability |

## Install

The extras are deliberately not bundled. pip resolves an extra atomically, so
pairing `appworld` with `vllm` means vLLM's source build failing takes `appworld`
down with it, and the CPU-only gate cannot run at all. That happened.

```bash
git clone https://github.com/yoadjei/hgd.git
cd hgd
pip install -e ".[dev]"            # tests and analysis. numpy only, no compiler
pip install -e ".[bench,dev]"      # adds appworld, for the gate and the census
pip install -e ".[serve]"          # adds vllm. GPU inference only, builds from source
```

Install `serve` only when you are about to serve a model. Nothing else needs it.

## Running things

```bash
pytest -q                                   # 320 tests, no appworld needed
python experiments/phase2_validation.py     # estimator validation, ~1 min, cpu
python notebooks/adapter_probe.py           # verify the adapter against real appworld
python notebooks/phase1_gate.py --tasks 3   # determinism gate, smoke test
python notebooks/phase1_gate.py             # determinism gate, 20 tasks
python notebooks/hstar_census.py            # H* distribution and compute budget
```

Run the probe before the gate. It is the only thing that drives
`AppWorldEnvironment` against the real package, and it reports on the assumptions
the pilot runner rests on — `execute()`'s return type, whether `task_completed()`
flips, whether `save_state`/`load_state` restore exactly, and what the evaluation
payload actually contains. Each check reports rather than raises, so one run
answers all of them.

The gate and the census need the benchmark and its data:

```bash
appworld install && appworld download data
```

Neither needs a GPU. Replay re-executes logged actions with no model in the loop,
so determinism is a property of the environment and can be settled before any
inference quota is spent.

## On Kaggle

Three cells, with a kernel restart after the first. `appworld` pins pydantic 1.x
and downgrades it, so anything importing it before the restart sees the old version
still loaded. `%cd` does not survive the restart, hence the repeat. None of this
needs a GPU; leave the accelerator off.

```python
# cell 1: install. run once, then restart the kernel
!rm -rf /kaggle/working/hgd
!git clone https://github.com/yoadjei/hgd.git /kaggle/working/hgd
%cd /kaggle/working/hgd
!pip install -q -e ".[bench,dev]"
!appworld install
!appworld download data
```

Restart the kernel. Do not re-run cell 1.

```python
# cell 2: verify, then run
%cd /kaggle/working/hgd
!python -m pytest -q                          # the suite, on kaggle's interpreter
!python notebooks/adapter_probe.py            # adapter against the real package
!python notebooks/phase1_gate.py --tasks 3    # smoke test
!python notebooks/phase1_gate.py              # the gate, kill condition C
```

`dev` is pytest plus freezegun pinned to appworld's own range, so the suite runs
against the same freezegun the adapter meets in production.

Everything writes into `results/`, alongside the Phase 2 evidence:
`results/phase1_gate.json` after every task, so an interrupted run still leaves
something, and `results/adapter_probe.json`. Read the probe's failures before
trusting anything built on the adapter.

```python
# cell 3: get the evidence off the machine before the session ends
!cat results/adapter_probe.json
!cat results/phase1_gate.json
```

These files are the evidence for their gates, and a hosted runtime's working
directory does not survive the session. Copy them out and commit them from a
machine with push access. Pushing from the notebook itself needs a GitHub token
stored as a Kaggle Secret; a plain `git push` there has no credentials and fails.

For the GPU phase later, Kaggle gives two T4s. They are compute capability 7.5,
so bfloat16 and FlashAttention-2 are both unavailable and `--dtype float16` is
required, not optional.

## Layout

```
src/hgd/          the package: schema, harness, replay, estimators, interventions
tests/            287 tests, warnings are errors
experiments/      phase 2 estimator validation against planted ground truth
notebooks/        runners for the gate and the H* census
docs/             the contract, the audit, preregistration, phase 0 novelty work
tasks/            plan of record and the lessons that changed how this is built
results/          gate evidence, regenerated deterministically
```

## Where the project stands

Phase 0 closed as a conditional pass: no single paper owns the contribution, but
the gap is narrower than the audit claimed and the concessions are recorded in
`docs/phase0_verification.md` rather than argued away.

Phase 2 passed. Both planted mechanism shares recover exactly, the null control
returns zero, and propagation separates cleanly with disjoint intervals. Two
controls make it able to fail: a length-sensitive policy that violates
`history_scrub`'s identification assumption, and a byte-matched placebo summary.

Phase 1 passed on 2026-09-30: 20/20 matched, `no_effect=0`, `skipped=0`,
`errored=0` against appworld 0.1.3.post1. Kill condition C is satisfied and the
run is not vacuous — every task moved away from its baseline, which is what the
no-op guard checks. Three earlier attempts were void rather than failed, each
caught by that guard rather than by a number looking wrong.

The caveat worth carrying into the paper: fidelity is established under
`evaluation_digest`, which is behavioural. Two states agreeing on every unit test
are equivalent *for our estimands*, but the digest cannot see collateral state the
tests ignore. `database_digest` is the strict alternative and has not been run.

The adapter probe found two anomalies on real hardware, and both trace to one
upstream defect in appworld 0.1.3.post1. `load_state()` calls
`AppWorld.close_all()`, which stops the task's time freezer, and restarts nothing.
So the world silently continues on wall-clock time, and the following `close()`
stops the same freezegun instance twice. Reproduced locally against real freezegun;
the mechanism is documented in `src/hgd/appworld_env.py`.

The silent half is the dangerous one: unfrozen time makes timestamps
irreproducible, which is precisely what kill condition C certifies against, and it
would never show up in a digest that ignores timestamps. It also explains the
second anomaly — pass fraction falling 1.0 to 0.5 — without needing a second
defect, because the `complete_task()` call that appeared to cause it ran after the
clock had already been unfrozen.

`load_state` is therefore refused rather than delegated, and branching goes through
`hgd.replay.replay_prefix`, which is the path the gate validated. `close()` releases
the world even when appworld's teardown raises, and records the failure on
`teardown_errors` rather than swallowing it. `reset()` closes before constructing,
which is load-bearing: `initialize()` also calls `close_all()`, so two live worlds
leave the first unable to close.

**Completion is not idempotent**, and this one is independent of the clock. Measured
on 2026-09-30 on a fresh world with the freeze verified: running the gold solution
gives `task_success` true and pass fraction 1.0, and one further
`apis.supervisor.complete_task()` drops them to false and 0.5. One of the task's two
unit tests flips. A redundant completion therefore destroys P_obs, so the episode
loop stops at the first completion the environment reports, and `run_episode`
refuses to start from a task that already reports completion — the way to reach that
state is to branch from a replayed prefix that already finished. Any runner that
consumes `oracle_fix`'s `forced_action` inherits the same constraint.

`tasks/todo.md` has the defect table; it is worth reading before trusting any result
from this repository.

## A note on the guards

Several pieces of this code exist to stop it reporting a confident number about
the wrong thing, and each was added after it had already happened:

- the gate treats a task whose state never moved as void, not as a match, because
  two no-ops agree perfectly;
- `horizon_gap` refuses near-zero probabilities rather than returning `-inf`,
  because the floor effect would otherwise become a headline finding;
- `mechanism_share` refuses a zero gap, where a share is undefined, not small;
- the fidelity threshold is exact, never a tolerance.
