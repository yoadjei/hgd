"""Phase 1 gate runner, kill condition C.

Thin wrapper over ``hgd.gate``, which holds the logic and the tests. No GPU:
replay re-executes the released gold solutions with no model in the loop, so
determinism is settled before any inference quota is spent.

    !pip install -q appworld==0.1.3.post1
    !appworld install
    !appworld download data
    !python notebooks/phase1_gate.py --tasks 3      # smoke test
    !python notebooks/phase1_gate.py                # the real gate
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hgd.appworld_env import evaluation_digest, silenced  # noqa: E402
from hgd.gate import run_gate, unique_experiment_names  # noqa: E402

DEFAULT_TASKS = 20
SPLIT = "train"
RESULT_PATH = Path("phase1_gate.json")


def main(n_tasks: int = DEFAULT_TASKS, split: str = SPLIT) -> int:
    from appworld import AppWorld, load_task_ids

    task_ids = list(load_task_ids(split))[:n_tasks]
    name_for = unique_experiment_names("phase1_gate")

    @contextlib.contextmanager
    def open_world(task_id: str):
        # silenced covers construction and teardown, not only evaluate(): the
        # package reports at every one of them, and unsuppressed a single run
        # buries its own result under hundreds of lines of test report.
        with silenced():
            world = AppWorld(
                task_id=task_id,
                experiment_name=name_for(),
                ground_truth_mode="full",
            )
        try:
            with silenced():
                yield world
        finally:
            with silenced():
                world.close()

    def report_line(index: int, task_id: str, verdict: str, detail: str) -> None:
        print(f"[{index:3}/{len(task_ids)}] {task_id:<16}{verdict:<10}{detail}",
              flush=True)

    print(f"gate over {len(task_ids)} tasks from split {split!r}\n", flush=True)
    report = run_gate(task_ids, open_world, evaluation_digest, on_result=report_line)

    payload = report.to_dict(expected=len(task_ids))
    RESULT_PATH.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\nfidelity {payload['n_matched']}/{payload['total']}"
          f"   no_effect={len(payload['no_effect'])}"
          f"   skipped={len(payload['skipped'])}"
          f"   errored={len(payload['errored'])}")
    if report.first_traceback:
        print("\n--- first traceback ---")
        print(report.first_traceback)

    if report.is_void:
        print("VOID - those tasks never moved the state, so the action source is wrong.")
    elif payload["passes_gate"]:
        print("GATE PASSED - Phase 3a permitted.")
    else:
        print("GATE NOT PASSED - do not proceed to causal analysis.")

    print(f"\nwrote {RESULT_PATH.resolve()}")
    return 0 if payload["passes_gate"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS,
                        help="how many tasks to gate (use 3 for a smoke test)")
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args()
    raise SystemExit(main(args.tasks, args.split))
