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

```bash
git clone https://github.com/yoadjei/hgd.git
cd hgd
pip install -e ".[dev]"        # analysis and replay, no GPU needed
pip install -e ".[run]"        # adds appworld and vllm
```

The core install is numpy only, so everything except the benchmark runs on a
laptop. `appworld` pins pydantic 1.x and will downgrade it, which breaks
unrelated packages in a shared environment; use a fresh one, and restart the
kernel after installing in a notebook.

## Running things

```bash
pytest -q                                   # 287 tests
python experiments/phase2_validation.py     # estimator validation, ~1 min, cpu
python notebooks/phase1_gate.py --tasks 3   # determinism gate, smoke test
python notebooks/phase1_gate.py             # determinism gate, 20 tasks
python notebooks/hstar_census.py            # H* distribution and compute budget
```

The gate and the census need `appworld` installed with its data downloaded:

```bash
appworld install && appworld download data
```

Neither needs a GPU. Replay re-executes logged actions with no model in the
loop, so determinism is a property of the environment and can be settled before
any inference quota is spent.

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
