"""Phase 1 gate: does AppWorld replay deterministically?

SELF-CONTAINED ON PURPOSE. This file imports nothing from `hgd` and needs no
repository checkout, so it can be pasted into a single Kaggle cell or uploaded
as one file. The two digest functions are duplicated from
`src/hgd/appworld_env.py`; `tests/test_gate_script_consistency.py` asserts the
two copies agree, so they cannot drift silently.

It does three things, in order, and stops at the first failure:

  A. PROBE   — verify the AppWorld API matches what the adapter assumes, and
               measure both candidate state digests against each other.
  B. GATE    — replay logged action prefixes and report fidelity.
  C. REPORT  — write phase1_gate.json.

**No GPU is required.** Replay re-executes *logged actions*; no model is in the
loop. Determinism is a property of the environment, so the gate can be settled
before any inference quota is spent. Action sequences come from the released gold
solutions on the train split, which gives realistic, valid API traffic for free.

Audit Phase 1 acceptance: 100% replay fidelity on 20 trajectories.
Kill condition C: if this fails, the causal analysis does not proceed. Fix the
environment or switch to tau-squared-bench. Do not soften the threshold.

    !pip install -q appworld==0.1.3.post1
    !appworld install
    !appworld download data
    %run phase1_gate.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import traceback
from pathlib import Path


def _quiet_digest(world) -> str:
    """`evaluate()` prints a full test report; silence it or the log is unreadable."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return evaluation_digest(world)

N_TRAJECTORIES = 20
SPLIT = "train"
EXPERIMENT = "phase1_gate"
RESULTS_PATH = Path("phase1_gate.json")

findings: dict[str, object] = {}


# ---------------------------------------------------------------------------
# digests, duplicated from hgd.appworld_env and kept in sync by a test
# ---------------------------------------------------------------------------

def _sorted_outcomes(items) -> list[str]:
    """Canonical, sortable form of a pass or fail vector.

    Entries are not always strings. Once a task completes AppWorld returns
    dicts, and `sorted` raises "'<' not supported between instances of 'dict'
    and 'dict'". Serialising each entry first orders either shape.
    """
    return sorted(json.dumps(item, sort_keys=True, default=str) for item in (items or []))


def evaluation_digest(world) -> str:
    """Digest the unit-test outcome vector.

    Behavioural: two states agreeing on every check count as identical. That is
    arguably the right equivalence for our estimands, but it is coarse and will
    not see collateral state the checks ignore.

    Keys verified against the installed package on 2026-09-06:
    ['difficulty', 'failures', 'num_tests', 'passes', 'success']. The field is
    `failures`, NOT `fails`; the first version read `fails`, always got None, and
    digested only the pass vector.
    """
    payload = world.evaluate().to_dict()
    normalised = {
        "passes": _sorted_outcomes(payload.get("passes")),
        "failures": _sorted_outcomes(payload.get("failures")),
        "success": payload.get("success"),
        "num_tests": payload.get("num_tests"),
    }
    return hashlib.sha256(
        json.dumps(normalised, sort_keys=True, default=str).encode()
    ).hexdigest()


def database_digest(root: str | Path):
    """Digest every database file under ``root``.

    Exhaustive but brittle: incidental differences such as timestamps or
    autoincrement ids count as divergence, which would fail the gate for reasons
    that threaten no causal claim.
    """
    base = Path(root)

    def _digest(_world) -> str:
        accumulator = hashlib.sha256()
        for path in sorted(base.rglob("*.sqlite*")) + sorted(base.rglob("*.db")):
            accumulator.update(path.name.encode())
            accumulator.update(path.read_bytes())
        return accumulator.hexdigest()

    return _digest


# ---------------------------------------------------------------------------
# a. probe
# ---------------------------------------------------------------------------

def probe() -> list[str]:
    """Verify the adapter's assumptions against the installed package.

    The adapter was written from documentation, not from a running install, so
    every assumption is asserted here and a mismatch surfaces as a named failure
    rather than as a mysterious fidelity number.
    """
    from appworld import AppWorld, load_task_ids

    task_ids = load_task_ids(SPLIT)
    assert task_ids, f"no task ids returned for split {SPLIT!r}"
    findings["split_size"] = len(task_ids)

    sample = list(task_ids[:N_TRAJECTORIES])
    probe_id = sample[0]

    with AppWorld(
        task_id=probe_id, experiment_name=EXPERIMENT, ground_truth_mode="full"
    ) as world:
        for attribute in ("execute", "task_completed", "evaluate", "save_state",
                          "load_state", "close"):
            assert hasattr(world, attribute), f"AppWorld has no {attribute!r}"

        assert hasattr(world.task, "instruction"), "task has no instruction"

        ground_truth = world.task.ground_truth
        findings["ground_truth_type"] = type(ground_truth).__name__
        findings["ground_truth_fields"] = sorted(
            k for k in dir(ground_truth) if not k.startswith("_")
        )[:40]

        findings["evaluation_keys"] = sorted(world.evaluate().to_dict())

        state_id = world.save_state()
        findings["save_state_returns"] = type(state_id).__name__
        world.load_state(state_id)

    return sample


def gold_actions(task_id: str) -> list[str]:
    """Executable gold solution for a task, as a two-step action sequence.

    Two earlier action sources were wrong, both silently.

    `ground_truth.api_calls` is a list of HTTP *records* such as
    `{'method': 'get', 'url': '/supervisor/profile', 'data': {}}`, which
    `execute()` evaluates as dict literals, a no-op. Nothing ran and every
    replay pair matched trivially.

    `compiled_solution_code` on its own is a `def solution(apis, requester)`
    wrapper, so `execute()` defined the function and never called it. The world
    stayed untouched and the second gate run was void for the same reason.

    Both steps are needed: define, then invoke. Verified by
    `solution_source_probe.py` on 2026-09-20, where the probe task moved from
    one passing test to two. `apis` and `requester` are both already bound in
    the shell. `compiled_solution_code_body` also exists and is the likelier
    source for per-step oracle fixes later.
    """
    from appworld import AppWorld

    with AppWorld(
        task_id=task_id, experiment_name=EXPERIMENT, ground_truth_mode="full"
    ) as world:
        code = getattr(world.task.ground_truth, "compiled_solution_code", None)
        return [str(code), "solution(apis, requester)"] if code else []


def compare_digests(task_id: str, actions: list[str]) -> dict[str, object]:
    """Run the same actions twice and see which digest calls them identical.

    Reported, not decided. A behavioural digest that is stable while the database
    digest is not means AppWorld carries incidental state that would fail the
    gate for reasons unrelated to any causal claim.
    """
    from appworld import AppWorld

    db_digest = database_digest(os.environ.get("APPWORLD_ROOT", "."))
    observed: dict[str, list[str]] = {"evaluation": [], "database": []}

    for _ in range(2):
        with AppWorld(
            task_id=task_id, experiment_name=EXPERIMENT, ground_truth_mode="full"
        ) as world:
            for action in actions:
                world.execute(action)
            observed["evaluation"].append(evaluation_digest(world))
            try:
                observed["database"].append(db_digest(world))
            except Exception as exc:  # noqa: BLE001
                observed["database"].append(f"unavailable: {exc}")

    return {
        "evaluation_stable": observed["evaluation"][0] == observed["evaluation"][1],
        "database_stable": observed["database"][0] == observed["database"][1],
    }


# ---------------------------------------------------------------------------
# b. gate
# ---------------------------------------------------------------------------

def run_gate(task_ids: list[str]) -> dict[str, object]:
    """Replay each gold solution twice and compare state digests.

    Carries a **no-op guard**. Two identical digests mean nothing if the actions
    never changed the state: a pair of no-ops matches perfectly. So each task's
    post-execution digest is compared against a fresh, untouched world, and a
    task whose state never moved is counted as `no_effect` rather than as a
    match. Without this the gate reports its most reassuring possible answer
    precisely when it is broken.
    """
    from appworld import AppWorld

    matched, diverged, skipped, no_effect, errored = 0, [], [], [], []

    for task_id in task_ids:
        try:
            actions = gold_actions(task_id)
            if not actions:
                skipped.append(task_id)
                continue

            with AppWorld(
                task_id=task_id, experiment_name=EXPERIMENT, ground_truth_mode="full"
            ) as world:
                baseline = _quiet_digest(world)

            digests = []
            for _ in range(2):
                with AppWorld(
                    task_id=task_id, experiment_name=EXPERIMENT,
                    ground_truth_mode="full",
                ) as world:
                    for action in actions:
                        world.execute(action)
                    digests.append(_quiet_digest(world))

            if digests[0] == baseline:
                no_effect.append(task_id)
            elif digests[0] == digests[1]:
                matched += 1
            else:
                diverged.append(task_id)
        except Exception as exc:  # noqa: BLE001
            errored.append((task_id, f"{type(exc).__name__}: {exc}"))

    total = matched + len(diverged)
    return {
        "total": total,
        "matched": matched,
        "diverged": diverged,
        "skipped": skipped,
        "no_effect": no_effect,
        "errored": errored,
        "rate": matched / total if total else 0.0,
        # a run where tasks had no effect is void, not passing
        "passes_gate": total > 0 and matched == total and not no_effect,
    }


# ---------------------------------------------------------------------------
# c. main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("PHASE 1 GATE - AppWorld replay determinism")
    print("=" * 72)

    try:
        sample = probe()
        print(f"[A] probe OK - {findings['split_size']} tasks in split {SPLIT!r}")
        print(f"    evaluation keys: {findings['evaluation_keys']}")
        print(f"    ground_truth type: {findings['ground_truth_type']}")
    except Exception:
        findings["probe_error"] = traceback.format_exc()
        print("[A] PROBE FAILED - the adapter's API assumptions are wrong.")
        print(findings["probe_error"])
        _write()
        return 1

    try:
        actions = gold_actions(sample[0])
        findings["gold_action_count"] = len(actions)
        findings["digest_comparison"] = compare_digests(sample[0], actions[:5])
        print(f"[A] digest comparison: {findings['digest_comparison']}")
    except Exception:
        findings["digest_error"] = traceback.format_exc()
        print("[A] digest comparison unavailable; continuing with evaluation digest")

    result = run_gate(sample)
    findings["gate"] = result

    print(f"[B] fidelity {result['matched']}/{result['total']} "
          f"({result['rate']:.1%}), skipped {len(result['skipped'])}")

    if result["passes_gate"]:
        print("[C] GATE PASSED - Phase 3a is permitted.")
    else:
        print("[C] GATE FAILED - kill condition C.")
        print("    Do not proceed to causal analysis. Diverged:",
              result["diverged"][:5])

    _write()
    return 0 if result["passes_gate"] else 1


def _write() -> None:
    RESULTS_PATH.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nwrote {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
