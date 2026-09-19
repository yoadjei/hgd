"""Trajectory logging schema.

The field set is pinned against audit section 20 (equivalently prompt.txt
section 14). Nothing may be logged that is not in ``SCHEMA_FIELDS``, and nothing
in ``SCHEMA_FIELDS`` may be silently omitted: audit task T1.1 fails a batch if
any step is missing a field, because a field that is absent during collection
cannot be recovered afterwards and every downstream estimate is traceable to
these records.

One deliberate deviation from the audit's field list: ``latency`` is named
``latency_s`` so that the unit is carried by the name. No other field is
renamed.
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_FIELDS: tuple[str, ...] = (
    "run_id",
    "task_id",
    "seed",
    "model",
    "step",
    "u",
    "prompt_hash",
    "raw_output",
    "parsed_action",
    "tool_name",
    "tool_args",
    "tool_result",
    "tool_result_hash",
    "env_state_hash",
    "checkpoint_results",
    "event_labels",
    "judge_labels",
    "tokens_in",
    "tokens_out",
    "latency_s",
    "intervention_branch",
    "parent_run_id",
    "branch_step",
)

# a text-only turn has no tool call, and a factual root run has no parent
NULLABLE_FIELDS: frozenset[str] = frozenset(
    {
        "parsed_action",
        "tool_name",
        "tool_args",
        "tool_result",
        "parent_run_id",
        "branch_step",
    }
)

# closed set: a typo here would silently pool an intervention arm with the
# factual arm
INTERVENTION_BRANCHES: frozenset[str] = frozenset(
    {"factual", "history_scrub", "state_refresh", "oracle_fix", "none"}
)


class SchemaError(ValueError):
    """Raised when a step record does not satisfy the audit section 20 schema."""


class StepRecord:
    """One logged step of one trajectory.

    Validation happens in the constructor so that an invalid record cannot exist,
    rather than being caught at write time when the run is already lost.
    """

    __slots__ = SCHEMA_FIELDS

    def __init__(self, **fields: Any) -> None:
        missing = [name for name in SCHEMA_FIELDS if name not in fields]
        if missing:
            raise SchemaError(f"missing required field(s): {', '.join(missing)}")

        unknown = sorted(set(fields) - set(SCHEMA_FIELDS))
        if unknown:
            raise SchemaError(f"unknown field(s): {', '.join(unknown)}")

        for name in SCHEMA_FIELDS:
            value = fields[name]
            if value is None and name not in NULLABLE_FIELDS:
                raise SchemaError(f"field {name!r} must not be null")
            setattr(self, name, value)

        if self.intervention_branch not in INTERVENTION_BRANCHES:
            raise SchemaError(
                f"intervention_branch {self.intervention_branch!r} is not one of "
                f"{sorted(INTERVENTION_BRANCHES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in SCHEMA_FIELDS}

    def to_json(self) -> str:
        """One JSONL line. Keys sorted so that logs diff cleanly across runs."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> StepRecord:
        return cls(**json.loads(line))

    def __repr__(self) -> str:
        return (
            f"StepRecord(run_id={self.run_id!r}, step={self.step}, "
            f"branch={self.intervention_branch!r})"
        )
