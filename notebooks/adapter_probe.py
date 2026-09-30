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

RESULT_PATH = Path("adapter_probe.json")
COMPLETE_TASK = "apis.supervisor.complete_task()"

results: list[dict[str, Any]] = []


def check(name: str, fn: Callable[[], Any]) -> Any:
    """Run one assumption check and record what it saw.

    A check that raises is recorded and the probe continues: the point is to
    collect every answer in one run, not to stop at the first surprise.
    """
    try:
        observed = fn()
        results.append({"check": name, "ok": True, "observed": observed})
        print(f"  PASS  {name}\n        {observed}", flush=True)
        return observed
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        results.append({"check": name, "ok": False, "observed": detail,
                        "traceback": traceback.format_exc()})
        print(f"  FAIL  {name}\n        {detail}", flush=True)
        return None


def run_gold_solution(env: AppWorldEnvironment) -> None:
    """Define the gold solution and then call it.

    ``compiled_solution_code`` is a ``def solution(apis, requester)`` wrapper, so
    executing it only binds a name. The second call is what moves the world.
    """
    with silenced():
        env.execute(env.gold_solution_code())
        env.execute(SOLUTION_INVOCATION)


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
    """Does completing an already-complete task change the evaluation?

    The first real run measured pass_fraction 1.0 after the gold solution and 0.5
    later, with a second complete_task() in between. If that is the cause, a model
    that calls complete_task() twice scores itself down and P_obs stops being a
    property of the trajectory.
    """
    def unchanged_by_a_second_call() -> dict[str, Any]:
        run_gold_solution(env)
        before = outcome_views(env)
        with silenced():
            env.execute(COMPLETE_TASK)
        after = outcome_views(env)
        assert before == after, f"evaluation changed: {before} -> {after}"
        return {"before": before, "after": after}

    check("double: a second complete_task() leaves the evaluation alone",
          unchanged_by_a_second_call)


def scenario_save_and_load(env: AppWorldEnvironment) -> None:
    """The replay layer assumes a restore is exact. Branching at step k needs it.

    This is also the sequence the first real run died on: close() raised inside
    appworld's time freezer afterwards, so this scenario's close check is the one
    to read.
    """
    def restores_exactly() -> str:
        marker = env.save_state()
        saved = env.state_hash()
        with silenced():
            env.execute(COMPLETE_TASK)
        moved = env.state_hash()
        env.load_state(marker)
        restored = env.state_hash()
        assert restored == saved, (
            f"restore is not exact: saved {saved[:12]}, moved {moved[:12]}, "
            f"restored {restored[:12]}"
        )
        return (f"marker {marker!r} ({type(marker).__name__}), "
                f"moved to {moved[:12]}, restored exactly")

    check("saveload: save_state()/load_state() restore exactly", restores_exactly)


SCENARIOS: tuple[tuple[str, Callable[[AppWorldEnvironment], None]], ...] = (
    ("open", scenario_open_and_close),
    ("execute", scenario_execute),
    ("gold", scenario_gold_solution),
    ("flip", scenario_completion_flip),
    ("double", scenario_double_complete),
    ("saveload", scenario_save_and_load),
)


def write_report(task_id: str, split: str) -> None:
    failed = [row for row in results if not row["ok"]]
    payload = {"task_id": task_id, "split": split, "n_checks": len(results),
               "n_failed": len(failed), "checks": results}
    with contextlib.suppress(OSError):
        RESULT_PATH.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n{len(results) - len(failed)}/{len(results)} assumptions hold")
    if failed:
        print("\nassumptions that do NOT hold, fix these before the pilot runner:")
        for row in failed:
            print(f"  - {row['check']}\n      {row['observed']}")
        print("\n--- first traceback ---")
        print(failed[0]["traceback"])
    else:
        print("every assumption the pilot runner rests on is now observed.")

    print(f"\nwrote {RESULT_PATH.resolve()}")


def main(split: str = "train") -> int:
    from appworld import AppWorld, load_task_ids

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
        write_report(task_id, split)

    return 1 if any(not row["ok"] for row in results) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    raise SystemExit(main(args.split))
