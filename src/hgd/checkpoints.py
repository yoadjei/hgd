"""Checkpoints and recoverability — audit task T1.4, sections 12 and 28.

Tier-2 labels: programmatic state assertions evaluated at stage boundaries. This
tier carries more weight than its simplicity suggests. Audit section 12 assigns
the recoverability outcome by state check and *never* by judge, which is what
keeps the competing-risk analysis alive if judge agreement fails (kill condition
D). The survival event of the whole hazard model — the first *unrecovered*
critical error — is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

RECOVERED = "recovered"
PERSISTED = "persisted"
TERMINAL = "terminal"

RECOVERABILITY_OUTCOMES: tuple[str, ...] = (RECOVERED, PERSISTED, TERMINAL)


@dataclass(frozen=True)
class Checkpoint:
    """A programmatic assertion about environment state at a stage boundary."""

    checkpoint_id: str
    stage: int
    predicate: Callable[[Any], bool]


@dataclass(frozen=True)
class CheckpointResult:
    checkpoint_id: str
    stage: int
    passed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "passed": self.passed,
            "error": self.error,
        }


def evaluate_checkpoints(
    env: Any,
    checkpoints: Sequence[Checkpoint],
) -> tuple[CheckpointResult, ...]:
    """Evaluate every checkpoint against current environment state.

    A predicate that raises is recorded as failed with its error text rather
    than propagated: a broken assertion must not destroy a long run, and the
    error is needed to tell a genuine state failure from a broken check.
    """
    # snapshot before the guard would cost an evaluate() per step on appworld,
    # which runs the task's unit tests, for runs that configured no checkpoints
    if not checkpoints:
        return ()

    state = env.snapshot()
    results: list[CheckpointResult] = []

    for checkpoint in checkpoints:
        try:
            passed = bool(checkpoint.predicate(state))
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            passed, error = False, f"{type(exc).__name__}: {exc}"

        results.append(
            CheckpointResult(
                checkpoint_id=checkpoint.checkpoint_id,
                stage=checkpoint.stage,
                passed=passed,
                error=error,
            )
        )

    return tuple(results)


def _first_failure(series: Sequence[bool]) -> int | None:
    for index, passed in enumerate(series):
        if not passed:
            return index
    return None


def recoverability(series: Sequence[bool], *, run_succeeded: bool) -> str | None:
    """Classify one checkpoint's history into the audit's three outcomes.

    Returns ``None`` when the checkpoint never failed — there is nothing to
    recover from, and such a checkpoint contributes no event.
    """
    first_fail = _first_failure(series)
    if first_fail is None:
        return None

    if any(series[first_fail + 1 :]):
        return RECOVERED

    return PERSISTED if run_succeeded else TERMINAL


def first_unrecovered_error(
    series_by_checkpoint: Mapping[str, Sequence[bool]],
    *,
    run_succeeded: bool,
) -> int | None:
    """Step index of the survival event, or ``None`` if the trajectory is censored.

    The event is the first failure that is never subsequently repaired. A failure
    that recovers is not an event — that distinction is the whole reason the
    hazard model uses competing risks rather than a single failure time.
    """
    unrecovered: list[int] = []

    for series in series_by_checkpoint.values():
        outcome = recoverability(series, run_succeeded=run_succeeded)
        if outcome in (PERSISTED, TERMINAL):
            first_fail = _first_failure(series)
            if first_fail is not None:
                unrecovered.append(first_fail)

    return min(unrecovered) if unrecovered else None
