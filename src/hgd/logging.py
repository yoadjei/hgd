"""Trajectory persistence — audit section 35.

Append-only JSONL, flushed on every write. The audit requires that every result
be reconstructible from released logs; that is only true if the logs survive the
session that produced them. A Kaggle session killed at the nine-hour limit must
leave behind everything it had already finished, so buffering across trajectories
is not an option.

Grouping is by ``run_id`` rather than by file, so factual runs and their
intervention branches can share one file while remaining separable — which is
what the decomposition needs, since a branch is only interpretable next to its
parent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

from hgd.schema import SchemaError, StepRecord


class RunWriter:
    """Append trajectories to a JSONL file, flushing each one."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> RunWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, records: Sequence[StepRecord]) -> None:
        if self._handle is None:
            raise RuntimeError("RunWriter must be used as a context manager")

        for record in records:
            self._handle.write(record.to_json() + "\n")

        # flush per trajectory, so an interrupted run keeps every finished episode
        self._handle.flush()


def read_trajectories(path: str | Path) -> Iterator[list[StepRecord]]:
    """Read a JSONL log back, grouped into trajectories by ``run_id``.

    Order within a trajectory follows the file, which is the order the harness
    produced. Groups are yielded in first-appearance order.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no trajectory log at {source}")

    grouped: dict[str, list[StepRecord]] = {}

    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = StepRecord.from_json(line)
            except (json.JSONDecodeError, SchemaError) as exc:
                raise ValueError(
                    f"{source}: line {number} is not a valid step record: {exc}"
                ) from exc
            grouped.setdefault(record.run_id, []).append(record)

    yield from grouped.values()
