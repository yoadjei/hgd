"""Verify the AppWorld adapter against the real package. No GPU.

Every defect that reached a real run so far came from a test double that encoded
the same assumption as the code, so the suite could not fail. This probe removes
the remaining assumptions by asking the installed package directly, and it runs
the adapter itself — ``AppWorldEnvironment`` has never touched real AppWorld,
because the gate drives the raw package instead.

Four assumptions are still guesses, and the pilot runner is built on all four:

1. ``execute()`` returns a string.
2. ``task_completed()`` flips after ``apis.supervisor.complete_task()``.
3. ``save_state()`` and ``load_state()`` really restore state.
4. ``snapshot()`` survives whatever shapes the payload actually contains.

Every check reports rather than raises, so one run answers all of them instead of
stopping at the first surprise.

    !pip install -q -e ".[bench]"
    !appworld install && appworld download data
    # restart the kernel here: appworld downgrades pydantic
    !python notebooks/adapter_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hgd.appworld_env import (  # noqa: E402
    AppWorldEnvironment,
    evaluation_digest,
    silenced,
)
from hgd.checkpoints import Checkpoint, evaluate_checkpoints  # noqa: E402
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


def main(split: str = "train") -> int:
    from appworld import AppWorld, load_task_ids

    task_id = list(load_task_ids(split))[0]
    print(f"probing the adapter on {task_id}\n", flush=True)

    def world_factory(tid: str, **kwargs: Any) -> Any:
        with silenced():
            return AppWorld(task_id=tid, **kwargs)

    env = AppWorldEnvironment(
        world_factory=world_factory,
        state_digest=evaluation_digest,
        experiment_name="adapter_probe",
    )

    try:
        check("reset() loads a task", lambda: (env.reset(task_id, seed=0), task_id)[1])
        check("intrinsic_horizon() is a positive int", lambda: env.intrinsic_horizon())
        check("gold_solution_code() returns source",
              lambda: f"{len(env.gold_solution_code())} chars")

        # 1. execute() return type. the adapter types it str and truncates the
        # result with a slice, which would raise on anything else.
        def execute_returns_a_string() -> str:
            out = env.execute("print('probe')")
            assert isinstance(out, str), f"got {type(out).__name__}, not str"
            return f"str, {len(out)} chars, repr {out[:60]!r}"

        check("execute() returns a str", execute_returns_a_string)

        # 2. the digest has to move when state moves, or the fidelity gate and
        # every replay check are comparing constants.
        def digest_responds_to_the_gold_solution() -> str:
            before = env.state_hash()
            with silenced():
                env.execute(env.gold_solution_code())
                env.execute("solution(apis, requester)")
            after = env.state_hash()
            assert before != after, "gold solution did not move the digest"
            return f"{before[:12]} -> {after[:12]}"

        check("state_hash() moves when the gold solution runs",
              digest_responds_to_the_gold_solution)

        # 3. snapshot() feeds checkpoint predicates and the oracle summary, so its
        # shape matters more than its content.
        def snapshot_is_usable() -> dict[str, Any]:
            state = env.snapshot()
            shapes = {k: type(v).__name__ for k, v in state.items()}
            for key in ("passes", "failures"):
                entries = state.get(key) or []
                bad = [e for e in entries if not isinstance(e, str)]
                assert not bad, f"{key} still holds non-strings: {bad[:2]}"
            return {"keys": sorted(state), "value_types": shapes}

        snapshot = check("snapshot() normalises passes and failures to names",
                         snapshot_is_usable)

        def outcome_views_agree() -> dict[str, Any]:
            state = env.snapshot()
            return {"task_success": task_success(state),
                    "pass_fraction": round(pass_fraction(state), 4),
                    "n_checkpoints": len(checkpoint_vector(state))}

        check("the three outcome views all compute", outcome_views_agree)

        def checkpoints_evaluate() -> Any:
            probe = Checkpoint("any_pass", stage=0,
                               predicate=lambda s: bool(s.get("passes")))
            return [r.to_dict() for r in evaluate_checkpoints(env, (probe,))]

        check("evaluate_checkpoints() runs against a live world",
              checkpoints_evaluate)

        # 4. save_state/load_state. the replay layer assumes a restore is exact;
        # if it is not, branching an episode at step k is unsound.
        def save_and_load_restore_state() -> str:
            marker = env.save_state()
            saved = env.state_hash()
            with silenced():
                env.execute("apis.supervisor.complete_task()")
            moved = env.state_hash()
            env.load_state(marker)
            restored = env.state_hash()
            assert restored == saved, (
                f"restore is not exact: saved {saved[:12]}, "
                f"moved {moved[:12]}, restored {restored[:12]}"
            )
            return f"marker {marker!r}, exact restore confirmed"

        check("save_state()/load_state() restore exactly",
              save_and_load_restore_state)

        # 5. the harness breaks the loop on this, so a completion that never
        # reports would run every episode to the full 3*H* step limit.
        def completion_is_reported() -> str:
            before = env.task_completed()
            with silenced():
                env.execute(COMPLETE_TASK)
            after = env.task_completed()
            assert after, f"task_completed() stayed {after!r} after complete_task()"
            return f"{before!r} -> {after!r}"

        check("task_completed() flips after complete_task()",
              completion_is_reported)

        # entries change shape once a task completes, which is the defect class
        # that reached a real run three times. re-check every view after the fact.
        check("snapshot() is still name-normalised after completion",
              snapshot_is_usable)
        check("the outcome views still compute after completion",
              outcome_views_agree)
    finally:
        with silenced():
            env.close()

    failed = [r for r in results if not r["ok"]]
    payload = {"task_id": task_id, "split": split, "n_checks": len(results),
               "n_failed": len(failed), "checks": results,
               "first_snapshot": snapshot}
    RESULT_PATH.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n{len(results) - len(failed)}/{len(results)} assumptions hold")
    if failed:
        print("\nassumptions that do NOT hold, fix these before the pilot runner:")
        for row in failed:
            print(f"  - {row['check']}: {row['observed']}")
        print("\n--- first traceback ---")
        print(failed[0]["traceback"])
    else:
        print("every assumption the pilot runner rests on is now observed, not guessed.")

    print(f"\nwrote {RESULT_PATH.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    raise SystemExit(main(args.split))
