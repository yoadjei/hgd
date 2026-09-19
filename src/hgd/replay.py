"""Replay from a logged step — audit task T1.2 and the Phase 1 gate.

Replay re-executes *logged actions* against a fresh environment. No model is
called, so any divergence between the replayed state and the logged state is a
property of the environment, not of sampling. That separation is what makes the
fidelity number interpretable.

Kill condition C: if fidelity is below 100% on the validation set, the causal
analysis does not proceed. ``verify_fidelity`` is therefore written to fail
loudly on a single divergence rather than report an encouraging average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from hgd.env import Environment
from hgd.schema import StepRecord


@dataclass(frozen=True)
class FidelityReport:
    """Outcome of the Phase 1 gate over a set of trajectories."""

    total: int
    matched: int
    diverged_trajectories: tuple[int, ...] = ()

    @property
    def rate(self) -> float:
        return self.matched / self.total if self.total else 0.0

    @property
    def passes_gate(self) -> bool:
        """Audit Phase 1 requires 100%. A tolerance here would defeat the point."""
        return self.total > 0 and self.matched == self.total


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of replaying one prefix."""

    matches: bool
    state_hash: str
    steps_executed: int
    diverged_at: int | None = None
    expected_hash: str | None = None


def replay_prefix(
    records: Sequence[StepRecord],
    boundary: int,
    env: Environment,
) -> ReplayResult:
    """Re-execute steps ``0 .. boundary-1`` and check state against the log.

    Returns as soon as the state diverges: continuing from a state we already
    know is wrong would only produce misleading downstream comparisons.
    """
    if boundary > len(records):
        raise ValueError(
            f"cannot replay {boundary} steps: trajectory has {len(records)}, "
            "requested boundary is beyond its length"
        )
    if boundary < 0:
        raise ValueError("boundary must be non-negative")

    anchor = records[0] if records else None
    env.reset(anchor.task_id if anchor else "", anchor.seed if anchor else 0)

    for index in range(boundary):
        record = records[index]
        env.execute(record.parsed_action)
        observed = env.state_hash()
        if observed != record.env_state_hash:
            return ReplayResult(
                matches=False,
                state_hash=observed,
                steps_executed=index + 1,
                diverged_at=index,
                expected_hash=record.env_state_hash,
            )

    return ReplayResult(
        matches=True,
        state_hash=env.state_hash(),
        steps_executed=boundary,
    )


def verify_fidelity(
    trajectories: Sequence[Sequence[StepRecord]],
    env_factory: Callable[[], Environment],
) -> FidelityReport:
    """Run the Phase 1 gate over a validation set of logged trajectories."""
    diverged: list[int] = []
    for index, records in enumerate(trajectories):
        result = replay_prefix(records, len(records), env_factory())
        if not result.matches:
            diverged.append(index)

    return FidelityReport(
        total=len(trajectories),
        matched=len(trajectories) - len(diverged),
        diverged_trajectories=tuple(diverged),
    )
