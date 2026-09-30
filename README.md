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
pytest -q                                   # 287 tests, no appworld needed
python experiments/phase2_validation.py     # estimator validation, ~1 min, cpu
python notebooks/phase1_gate.py --tasks 3   # determinism gate, smoke test
python notebooks/phase1_gate.py             # determinism gate, 20 tasks
python notebooks/hstar_census.py            # H* distribution and compute budget
```

The gate and the census need the benchmark and its data:

```bash
appworld install && appworld download data
```

Neither needs a GPU. Replay re-executes logged actions with no model in the loop,
so determinism is a property of the environment and can be settled before any
inference quota is spent.

## On Kaggle

Two cells, with a kernel restart between them. `appworld` pins pydantic 1.x and
downgrades it, so anything importing it before the restart sees the old version
still loaded. `%cd` does not survive the restart, hence the repeat.

```python
# cell 1, accelerator off
!git clone https://github.com/yoadjei/hgd.git /kaggle/working/hgd
%cd /kaggle/working/hgd
!pip install -q -e ".[bench,dev]"   # dev is only pytest, so you can verify the clone
!appworld install
!appworld download data
```

Restart the kernel. Do not re-run cell 1.

```python
# cell 2
%cd /kaggle/working/hgd
!python notebooks/phase1_gate.py --tasks 3
```

Drop `--tasks 3` for the full twenty-task gate. The result is written to
`phase1_gate.json` after every task, so an interrupted run still leaves evidence.

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

Phase 1 is not passed. The determinism gate, kill condition C, has run clean on
three tasks but not yet on twenty. Three earlier attempts were void rather than
failed, each caught by the no-op guard rather than by a number looking wrong.
`tasks/todo.md` has the defect table; it is worth reading before trusting any
result from this repository.

## A note on the guards

Several pieces of this code exist to stop it reporting a confident number about
the wrong thing, and each was added after it had already happened:

- the gate treats a task whose state never moved as void, not as a match, because
  two no-ops agree perfectly;
- `horizon_gap` refuses near-zero probabilities rather than returning `-inf`,
  because the floor effect would otherwise become a headline finding;
- `mechanism_share` refuses a zero gap, where a share is undefined, not small;
- the fidelity threshold is exact, never a tolerance.
