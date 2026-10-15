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
!PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q   # the suite, on kaggle's interpreter
!python notebooks/adapter_probe.py                     # adapter against the real package
!python notebooks/phase1_gate.py --tasks 3             # smoke test
!python notebooks/phase1_gate.py                       # the gate, kill condition C
```

The variable is not optional on Kaggle. pytest loads every installed plugin at
startup, and the image ships `langsmith`, whose plugin needs pydantic 2. After
appworld downgrades pydantic to 1.x, that plugin raises on import and pytest dies
before collecting anything. The suite uses no third-party plugins, so turning
autoload off loses nothing.

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

### The GPU smoke test

Before any pilot quota is spent, check that a model server actually runs on this
hardware. Three facts constrain how, each read from the packages themselves:

- **vLLM and appworld cannot share an environment.** vLLM needs pydantic 2.12 or
  newer and appworld pins pydantic below 2. The server goes in its own
  environment; the harness talks to it over HTTP.
- **The vLLM version depends on the NVIDIA driver.** vLLM 0.20 and later pin a
  torch built for CUDA 13, which needs driver 580 or newer. vLLM 0.19.1 is the
  newest on CUDA 12.8, which runs on older drivers.
- **A T4 is compute capability 7.5.** vLLM still supports it in float16, with
  attention falling back from FlashAttention to Triton and 4-bit AWQ falling back
  from Marlin to slower kernels. So the throughput has to be measured, not assumed.

Set the accelerator to **GPU T4 x2** with internet on. No appworld install is needed
for this, and no kernel restart.

```python
# cell 1: what is this machine, and a clean start
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
!rm -rf /kaggle/working/hgd /tmp/vllm-env /tmp/vllm.log
!git clone -q https://github.com/yoadjei/hgd.git /kaggle/working/hgd
```

Cell 1 removes anything an earlier attempt left in this session, including a
half-built environment, but keeps the downloaded weights.

Read the driver version, then install vLLM into its own environment, outside
`/kaggle/working`:

```python
# cell 2: driver 580 or newer
!pip install uv && uv venv /tmp/vllm-env --python 3.12
!uv pip install --python /tmp/vllm-env/bin/python "vllm==0.31.0"
!/tmp/vllm-env/bin/vllm --version

# cell 2, instead, if the driver is older than 580
!pip install uv && uv venv /tmp/vllm-env --python 3.12
!uv pip install --python /tmp/vllm-env/bin/python "vllm==0.19.1"
!/tmp/vllm-env/bin/vllm --version
```

No `-q` on purpose: the install takes minutes, and a quiet install that failed
looks the same as one still running. The last line has to print a version before
cell 3 is worth running.

```python
# cell 3: start the server on one T4, in the background, then measure it
!pkill -f "[v]llm serve"; timeout 60 sh -c 'while pgrep -f "[v]llm serve" > /dev/null; do sleep 1; done'
!nohup sh -c '/tmp/vllm-env/bin/vllm serve Qwen/Qwen3-8B-AWQ --dtype float16 --max-model-len 16384 --port 8000; echo "server exited with status $?"' > /tmp/vllm.log 2>&1 &
!python /kaggle/working/hgd/notebooks/serve_smoke.py --model Qwen/Qwen3-8B-AWQ --log /tmp/vllm.log
```

The server runs in the background, so its output, including the weight download,
goes to `/tmp/vllm.log` rather than the cell. The smoke test shows the latest line
of that log every 30 seconds while it waits. The `echo` after the server is what
lets it stop at once if the server exits instead of waiting out the full 20
minutes; the end of the log then says why. A traceback alone does not stop it,
because libraries log exceptions they recover from.

The first line stops any server left from an earlier run, which would otherwise
hold the port and the GPU memory, and waits up to a minute for it to exit. The
brackets in `[v]llm` match `vllm` but not the literal text `[v]llm`, so the line
never matches its own command.

Once healthy, the smoke test checks that a request with thinking off comes back
without a `<think>` block, and measures aggregate decode throughput with 16
requests in flight. It writes `results/serve_smoke.json` whatever happens.

```python
# cell 4: the evidence, before the session ends
!cat /kaggle/working/hgd/results/serve_smoke.json
!tail -n 40 /tmp/vllm.log
```

The weights download to `/root/.cache/huggingface`, which does not survive the
session: every new session downloads them again.

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
caught by that guard rather than by a number looking wrong. It reproduced on
2026-10-15 from a fresh session, 20/20 again with the adapter probe at 25/25; the
console transcript is `results/kaggle_2026-10-15.log`.

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
`apis.supervisor.complete_task()` drops them to false and 0.5. The test that flips is
`assert answers match.` (measured 2026-10-15). That fits the second call, made with no
`answer` argument, replacing the answer the gold solution submitted, so the last
completion's answer is the one scored. This reading is inferred from the test's name
and not yet measured; a second call that carries the same answer would settle it.
A redundant completion therefore destroys P_obs, so the episode
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
