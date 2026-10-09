"""Verify the AppWorld adapter against the real package. No GPU.

Every defect that reached a real run so far came from a test double that encoded
the same assumption as the code, so the suite could not fail. This probe removes
the assumptions by asking the installed package directly, and it drives the
adapter itself — the gate exercises raw AppWorld and a digest function, so
``AppWorldEnvironment``, the checkpoint layer and the outcome views were all
unobserved until this ran.

**One world per scenario, and closing is itself under test.** The first real run
raised inside AppWorld's freezegun time freezer on ``close()``, which the gate
never sees because the gate never saves or loads state. Attributing that to a
particular sequence of calls needs a fresh world per sequence, so each scenario
gets one and reports its own close.

Every check reports rather than raises, and the record is written even when a
scenario dies, so one run answers everything instead of stopping at the first
surprise. The previous version wrote its findings after teardown and lost all
twelve of them when teardown raised.

    !pip install -q -e ".[bench]"
    !appworld install && appworld download data
    # restart the kernel here: appworld downgrades pydantic
    !python notebooks/adapter_probe.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hgd.appworld_env import (  # noqa: E402
    SOLUTION_INVOCATION,
    AppWorldEnvironment,
    evaluation_digest,
    silenced,
)
from hgd.checkpoints import Checkpoint, evaluate_checkpoints  # noqa: E402
from hgd.gate import unique_experiment_names  # noqa: E402
from hgd.outcomes import checkpoint_vector, pass_fraction, task_success  # noqa: E402

# alongside the phase 2 evidence rather than in the caller's directory, so a probe
# result is archived with everything else instead of living on an ephemeral disk
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COMPLETE_TASK = "apis.supervisor.complete_task()"

results: list[dict[str, Any]] = []


def check(name: str, fn: Callable[[], Any]) -> Any:
    """Run one assumption check and record what it saw.

    A check that raises is recorded and the probe continues: the point is to
    collect every answer in one run, not to stop at the first surprise.
    """
    try:
        observed = fn()
        results.append({"check": name, "kind": "check", "ok": True,
                        "observed": observed})
        print(f"  PASS  {name}\n        {observed}", flush=True)
        return observed
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        results.append({"check": name, "kind": "check", "ok": False,
                        "observed": detail, "traceback": traceback.format_exc()})
        print(f"  FAIL  {name}\n        {detail}", flush=True)
        return None


def observe(name: str, fn: Callable[[], Any]) -> Any:
    """Record something about the package that nothing here depends on.

    Unlike ``check``, the value is never a failure, because the pilot runner is
    correct whatever it turns out to be. Reporting such a thing as a failure keeps
    the probe permanently red over something that needs no fix, and a probe that
    is always red stops being read. An observation that cannot be *made* still
    fails: if the measurement raises, nothing was observed.
    """
    try:
        observed = fn()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        results.append({"check": name, "kind": "observation", "ok": False,
                        "observed": detail, "traceback": traceback.format_exc()})
        print(f"  FAIL  {name}\n        could not observe: {detail}", flush=True)
        return None
    results.append({"check": name, "kind": "observation", "ok": True,
                    "observed": observed})
    print(f"  NOTE  {name}\n        {observed}", flush=True)
    return observed


def run_gold_solution(env: AppWorldEnvironment) -> None:
    """Define the gold solution and then call it.

    ``compiled_solution_code`` is a ``def solution(apis, requester)`` wrapper, so
    executing it only binds a name. The second call is what moves the world.
    """
    with silenced():
        env.execute(env.gold_solution_code())
        env.execute(SOLUTION_INVOCATION)


def freezer_depth() -> int | None:
    """Active freezegun freezes, or None if that cannot be read."""
    try:
        from freezegun import api

        return len(api.freeze_factories)
    except Exception:
        return None


def outcome_views(env: AppWorldEnvironment) -> dict[str, Any]:
    state = env.snapshot()
    return {"task_success": task_success(state),
            "pass_fraction": round(pass_fraction(state), 4),
            "n_checkpoints": len(checkpoint_vector(state))}


def snapshot_is_normalised(env: AppWorldEnvironment) -> dict[str, Any]:
    """Checkpoint predicates read this, so its shape matters more than its content."""
    state = env.snapshot()
    for key in ("passes", "failures"):
        bad = [e for e in (state.get(key) or []) if not isinstance(e, str)]
        assert not bad, f"{key} still holds non-strings: {bad[:2]}"
    return {"keys": sorted(state),
            "value_types": {k: type(v).__name__ for k, v in state.items()}}


# --- scenarios --------------------------------------------------------------
# each runs on its own world, and the name prefixes every check, so a close()
# failure is attributable to the sequence that caused it


def scenario_open_and_close(env: AppWorldEnvironment) -> None:
    """The floor. If this cannot close, nothing else about close is diagnostic."""


def scenario_execute(env: AppWorldEnvironment) -> None:
    def returns_a_string() -> str:
        out = env.execute("print('probe')")
        assert isinstance(out, str), f"got {type(out).__name__}, not str"
        return f"str, {len(out)} chars, repr {out[:60]!r}"

    check("execute: returns a str", returns_a_string)


def scenario_gold_solution(env: AppWorldEnvironment) -> None:
    # measured, not assumed. an earlier version asserted depth == 1 and failed on
    # real hardware at depth 2, because a live world holds its own freeze plus the
    # one its Requester starts (environment.py builds the requester through
    # ApiCollection.load, and requester.py starts a freeze when given a datetime).
    # the invariant is that the depth does not drift, not that it equals a number
    # i guessed.
    depth_at_start = freezer_depth()

    def digest_moves() -> str:
        before = env.state_hash()
        run_gold_solution(env)
        after = env.state_hash()
        assert before != after, "gold solution did not move the digest"
        return f"{before[:12]} -> {after[:12]}"

    check("gold: state_hash() moves when the gold solution runs", digest_moves)

    # observation, not an assertion: the released solution calls complete_task()
    # itself. an earlier version checked the completion flip *after* running the
    # gold solution, so the baseline was already True and the check passed while
    # proving nothing.
    check("gold: does the solution complete the task by itself",
          env.task_completed)

    check("gold: snapshot() normalises passes and failures to names",
          lambda: snapshot_is_normalised(env))
    check("gold: the three outcome views all compute", lambda: outcome_views(env))
    check("gold: evaluate_checkpoints() runs against a live world",
          lambda: [r.to_dict() for r in evaluate_checkpoints(
              env, (Checkpoint("any_pass", stage=0,
                               predicate=lambda s: bool(s.get("passes"))),))])

    # the check that would have caught the original bug. appworld's evaluator
    # starts a time freezer and stops it with no try/finally, so one failed
    # evaluation leaks a freeze permanently and every later task runs on a
    # corrupted clock. this scenario has evaluated several times by now.
    def freezer_stack_has_not_drifted() -> str:
        depth = freezer_depth()
        if depth is None or depth_at_start is None:
            raise AssertionError("cannot read freezegun's freeze stack")
        assert depth == depth_at_start, (
            f"{depth} freezes active, {depth_at_start} at the start of this "
            "scenario; a leaked freeze means timestamps stop being reproducible"
        )
        return f"{depth} freezes, unchanged across evaluation"

    check("gold: evaluating does not leak a time freeze",
          freezer_stack_has_not_drifted)


def scenario_completion_flip(env: AppWorldEnvironment) -> None:
    """The assumption harness.py breaks its loop on.

    On a fresh world, so the baseline is genuinely incomplete. If completion never
    reported, every episode would run to the full 3*H* step limit.
    """
    def flips() -> str:
        before = env.task_completed()
        assert not before, (
            "already complete before complete_task() ran, so no flip can be "
            "observed and this check would prove nothing"
        )
        with silenced():
            env.execute(COMPLETE_TASK)
        after = env.task_completed()
        assert after, f"task_completed() stayed {after!r} after complete_task()"
        return f"{before!r} -> {after!r}"

    check("flip: task_completed() goes False -> True on complete_task()", flips)


def scenario_double_complete(env: AppWorldEnvironment) -> None:
    """What does completing an already-complete task do to the evaluation?

    Measured against appworld 0.1.3.post1 on 2026-09-30, on a fresh world with
    the clock verified frozen and no load_state: one of the task's two unit tests
    flips, and pass fraction falls from 1.0 to 0.5. Completion is not idempotent.
    An earlier version of this docstring blamed the unfrozen clock instead; that
    was wrong, and this measurement is what showed it.

    The harness copes by stopping at the first completion and refusing to start
    from a completed task, which keeps the pilot correct whichever way this comes
    out. So it is an observation, recorded with the tests that flipped, not a
    check: if a later appworld makes completion idempotent, this says so.
    """
    def failing_tests() -> list[str]:
        return sorted(env.snapshot().get("failures") or [])

    def second_completion() -> dict[str, Any]:
        run_gold_solution(env)
        before, before_failures = outcome_views(env), failing_tests()
        with silenced():
            env.execute(COMPLETE_TASK)
        after, after_failures = outcome_views(env), failing_tests()
        return {"changed": before != after, "before": before, "after": after,
                "newly_failing": sorted(set(after_failures) - set(before_failures))}

    observe("double: what a second complete_task() does to the evaluation",
            second_completion)


def scenario_state_markers(env: AppWorldEnvironment) -> None:
    """Saving a checkpoint is fine; restoring from one is refused.

    This is the sequence the first real run died on. appworld's ``load_state()``
    calls ``AppWorld.close_all()``, which stops the task's time freezer, and
    restarts nothing, so the world silently continues on wall-clock time and the
    next ``close()`` double-stops the same freezegun instance. Reproduced locally
    against real freezegun, so the probe does not poison a world to watch it again.
    """
    check("markers: save_state() returns a marker",
          lambda: f"{env.save_state()!r}")

    def refuses_restore() -> str:
        try:
            env.load_state("0")
        except RuntimeError as exc:
            assert "replay" in str(exc), "the refusal must name the supported route"
            return "refused, and it names replay as the alternative"
        raise AssertionError(
            "load_state() was not refused; it unfreezes the clock silently"
        )

    check("markers: load_state() is refused rather than corrupting the clock",
          refuses_restore)


def scenario_frozen_clock(env: AppWorldEnvironment) -> None:
    """Is the shell clock actually frozen to the task's datetime?

    Never verified, and determinism rests on it: appworld freezes time per task so
    that two replays of the same actions write the same timestamps. If the shell
    sees wall-clock time, the Phase 1 gate's 20/20 is an accident of a digest that
    happens not to hash timestamps, and any stricter digest would fail.
    """
    def clock_is_frozen_to_the_task() -> str:
        expected = getattr(env.world.task, "datetime", None)
        if expected is None:
            raise AssertionError("task exposes no datetime to compare against")
        with silenced():
            printed = env.execute(
                "import datetime as _dt; print(_dt.datetime.now().isoformat())"
            )
        seen = (printed or "").strip().splitlines()[-1].strip()
        assert seen.startswith(str(expected.year)), (
            f"shell clock reads {seen!r}, task datetime is {expected!r}; "
            "time is not frozen, so timestamps are not reproducible"
        )
        return f"shell {seen}, task {expected}"

    check("clock: the shell clock is frozen to the task datetime",
          clock_is_frozen_to_the_task)


SCENARIOS: tuple[tuple[str, Callable[[AppWorldEnvironment], None]], ...] = (
    ("open", scenario_open_and_close),
    ("execute", scenario_execute),
    ("gold", scenario_gold_solution),
    ("flip", scenario_completion_flip),
    ("double", scenario_double_complete),
    ("markers", scenario_state_markers),
    ("clock", scenario_frozen_clock),
)


def write_report(task_id: str, split: str, result_path: Path) -> None:
    failed = [row for row in results if not row["ok"]]
    checks = [row for row in results if row["kind"] == "check"]
    notes = [row for row in results if row["kind"] == "observation"]
    payload = {"task_id": task_id, "split": split, "n_checks": len(checks),
               "n_observations": len(notes), "n_failed": len(failed),
               "checks": results}
    with contextlib.suppress(OSError):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2, default=str))

    held = sum(1 for row in checks if row["ok"])
    print(f"\n{held}/{len(checks)} assumptions hold")
    if notes:
        print(f"{len(notes)} observation(s) recorded, not counted: nothing here "
              "depends on them")
    if failed:
        print("\nassumptions that do NOT hold, fix these before the pilot runner:")
        for row in failed:
            print(f"  - {row['check']}\n      {row['observed']}")
        print("\n--- first traceback ---")
        print(failed[0]["traceback"])
    else:
        print("every assumption the pilot runner rests on is now observed.")

    print(f"\nwrote {result_path.resolve()}")


def main(split: str = "train", out_dir: Path | None = None) -> int:
    from appworld import AppWorld, load_task_ids

    result_path = (out_dir or RESULTS_DIR) / "adapter_probe.json"
    task_id = list(load_task_ids(split))[0]
    name_for = unique_experiment_names("adapter_probe")
    print(f"probing the adapter on {task_id}, one world per scenario\n", flush=True)

    def make_env() -> AppWorldEnvironment:
        def world_factory(tid: str, **kwargs: Any) -> Any:
            with silenced():
                return AppWorld(task_id=tid, **kwargs)

        return AppWorldEnvironment(
            world_factory=world_factory,
            state_digest=evaluation_digest,
            experiment_name=name_for(),
        )

    try:
        for name, body in SCENARIOS:
            print(f"--- {name} ---", flush=True)
            env = make_env()
            try:
                loaded = check(f"{name}: reset() loads the task",
                               lambda e=env: (e.reset(task_id, seed=0), task_id)[1])
                if loaded:
                    body(env)
            finally:
                # closing is under test, not teardown. the harness closes on every
                # reset, so a close that raises ends a multi-task run at its second
                # task, and the gate never sees it because it never saves state.
                def close_cleanly(active: AppWorldEnvironment = env) -> str:
                    with silenced():
                        active.close()
                    return "closed without raising"

                check(f"{name}: close() after this sequence", close_cleanly)
    finally:
        # a probe that dies without writing its findings is worth nothing. the
        # previous version wrote after teardown and lost all twelve findings when
        # teardown raised inside appworld's time freezer.
        write_report(task_id, split, result_path)

    return 1 if any(not row["ok"] for row in results) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the result (default: results/)")
    args = parser.parse_args()
    raise SystemExit(main(args.split, args.out))
