"""AppWorld replay determinism gate — audit Phase 1, kill condition C.

Runs the released gold solution for a task twice, in two fresh worlds, and asks
whether the resulting state digests agree. No model is in the loop, so any
divergence is a property of the environment rather than of sampling, which is
what makes the number interpretable. It needs no GPU.

Two guards carry most of the weight, and both have already fired on real data.

A task whose actions never moved the state is ``NO_EFFECT``, not a match. Two
no-ops agree perfectly, so without this the gate returns its most reassuring
answer precisely when the action source is broken. It caught two void runs.

Every world gets an experiment name of its own. Sharing one lets the second run
inherit the first run's output directory, so a match could come from a shared
artefact rather than from the environment.

Kill condition C: below total fidelity the causal analysis does not proceed.
Fix the environment or switch to tau-squared-bench; do not soften the threshold.
"""

from __future__ import annotations

import itertools
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Sequence

from hgd.appworld_env import SOLUTION_INVOCATION, gold_solution_code

MATCHED = "matched"
DIVERGED = "diverged"
NO_EFFECT = "no_effect"
SKIPPED = "skipped"
ERRORED = "errored"

VERDICTS: tuple[str, ...] = (MATCHED, DIVERGED, NO_EFFECT, SKIPPED, ERRORED)


def solution_actions(ground_truth: Any) -> list[str]:
    """The two steps that actually run a gold solution: define, then invoke.

    Executing the code alone only binds a function. That was the second wrong
    action source, and it left every world untouched.
    """
    return [gold_solution_code(ground_truth), SOLUTION_INVOCATION]


def gate_task(
    task_id: str,
    open_world: Callable[[str], ContextManager[Any]],
    digest: Callable[[Any], str],
) -> str:
    """Replay one task twice and classify the outcome.

    Builds two worlds, not four: the baseline and the first run share one, since
    the baseline is taken before any action executes.
    """
    with open_world(task_id) as world:
        try:
            actions = solution_actions(world.task.ground_truth)
        except ValueError:
            return SKIPPED
        baseline = digest(world)
        for action in actions:
            world.execute(action)
        first = digest(world)

    with open_world(task_id) as world:
        for action in actions:
            world.execute(action)
        second = digest(world)

    if first == baseline:
        return NO_EFFECT
    return MATCHED if first == second else DIVERGED


@dataclass
class GateReport:
    """Outcome of the gate over a set of tasks."""

    by_verdict: dict[str, list[str]] = field(
        default_factory=lambda: {verdict: [] for verdict in VERDICTS}
    )
    errors: list[tuple[str, str]] = field(default_factory=list)
    first_traceback: str | None = None

    def record(self, task_id: str, verdict: str) -> None:
        self.by_verdict[verdict].append(task_id)

    def record_error(self, task_id: str, exc: BaseException) -> str:
        detail = f"{type(exc).__name__}: {exc}"[:200]
        if self.first_traceback is None:
            self.first_traceback = traceback.format_exc()[-2000:]
        self.errors.append((task_id, detail))
        self.record(task_id, ERRORED)
        return detail

    @property
    def total(self) -> int:
        """Tasks that produced a usable comparison."""
        return len(self.by_verdict[MATCHED]) + len(self.by_verdict[DIVERGED])

    @property
    def matched(self) -> int:
        return len(self.by_verdict[MATCHED])

    @property
    def rate(self) -> float:
        return self.matched / self.total if self.total else 0.0

    @property
    def is_void(self) -> bool:
        """True when tasks never moved the state, which is not a failure to fix
        in the environment but a broken action source to fix here."""
        return bool(self.by_verdict[NO_EFFECT])

    def passes_gate(self, expected: int) -> bool:
        """Total fidelity over every task asked for. A tolerance defeats the point."""
        return (
            self.total == expected
            and not self.by_verdict[DIVERGED]
            and not self.is_void
            and not self.by_verdict[ERRORED]
        )

    def to_dict(self, expected: int) -> dict[str, Any]:
        return {
            **self.by_verdict,
            "total": self.total,
            "n_matched": self.matched,
            "rate": self.rate,
            "is_void": self.is_void,
            "passes_gate": self.passes_gate(expected),
            "errors": [list(pair) for pair in self.errors],
            "first_traceback": self.first_traceback,
        }


def run_gate(
    task_ids: Sequence[str],
    open_world: Callable[[str], ContextManager[Any]],
    digest: Callable[[Any], str],
    on_result: Callable[[int, str, str, str], None] | None = None,
) -> GateReport:
    """Gate every task, surviving individual failures.

    ``on_result`` is called after each task with its index, id, verdict and any
    error detail. Progress is reported as it happens because a summary printed
    only at the end is lost whenever a notebook truncates its output, which is
    how one real run left no evidence at all.
    """
    report = GateReport()

    for index, task_id in enumerate(task_ids, start=1):
        detail = ""
        try:
            verdict = gate_task(task_id, open_world, digest)
            report.record(task_id, verdict)
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the run
            verdict = ERRORED
            detail = report.record_error(task_id, exc)
        if on_result is not None:
            on_result(index, task_id, verdict, detail)

    return report


def unique_experiment_names(prefix: str) -> Callable[[], str]:
    """Names a world can be built under without colliding with an earlier one."""
    counter = itertools.count()
    return lambda: f"{prefix}_{next(counter)}"
