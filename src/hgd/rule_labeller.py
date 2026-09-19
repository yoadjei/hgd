"""Tier-1 rule labeller — audit section 28, task T1.3.

Seven deterministic events, computed from the log alone with no model in the
loop. This tier matters disproportionately: audit section 11 and prompt section
11 both require that the causal estimates survive a failed judge, which is only
true if the events the interventions branch on are rule-detectable. Everything
here is therefore a pure function of ``StepRecord`` fields, with no heuristics
that could drift between runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from hgd.schema import StepRecord

MALFORMED_ACTION = "malformed_action"
UNKNOWN_TOOL = "unknown_tool"
SCHEMA_INVALID_ARGS = "schema_invalid_args"
TOOL_ERROR = "tool_error"
EXACT_REPEAT = "exact_repeat"
STEP_LIMIT = "step_limit"
TEXT_ONLY = "text_only"

RULE_EVENTS: tuple[str, ...] = (
    MALFORMED_ACTION,
    UNKNOWN_TOOL,
    SCHEMA_INVALID_ARGS,
    TOOL_ERROR,
    EXACT_REPEAT,
    STEP_LIMIT,
    TEXT_ONLY,
)

# anchored at line start: that is what separates a real traceback from a
# playlist called "Trial and Error". shared with the synthetic policy so its
# notion of "an error in my history" is exactly the tier-1 rule.
EXCEPTION_LINE = re.compile(
    r"^[A-Za-z0-9_.]*(?:Error|Exception)\s*:", re.MULTILINE
)
_TRACEBACK_PREFIX = "Traceback (most recent call last)"


@dataclass(frozen=True)
class RuleConfig:
    """Everything the labeller needs to know about the environment.

    Kept explicit rather than imported from the environment so that labelling a
    released log never requires the environment to be installed.
    """

    known_tools: frozenset[str]
    tool_required_args: Mapping[str, frozenset[str]]
    step_limit: int
    repeat_window: int = 5
    action_markers: tuple[str, ...] = ("```python", "```", "<tool_call>")


def _canonical_call(record: StepRecord) -> str | None:
    """Identity of a tool call, invariant to argument ordering."""
    if record.tool_name is None:
        return None
    args = record.tool_args if record.tool_args is not None else {}
    return json.dumps([record.tool_name, args], sort_keys=True, default=str)


def _looks_like_tool_error(result: Any) -> bool:
    if not isinstance(result, str):
        return False
    stripped = result.strip()
    if stripped.startswith(_TRACEBACK_PREFIX):
        return True
    if EXCEPTION_LINE.search(stripped):
        return True
    try:
        payload = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and "error" in payload


def _attempted_an_action(record: StepRecord, config: RuleConfig) -> bool:
    raw = record.raw_output or ""
    return any(marker in raw for marker in config.action_markers)


def _label_step(
    record: StepRecord,
    history: Sequence[StepRecord],
    config: RuleConfig,
) -> list[str]:
    """Rule events for one step, given the steps that preceded it."""
    labels: list[str] = []

    if record.tool_name is None:
        # never tried and tried-but-unparseable are different events; the audit
        # grants no retry for a malformed call
        if _attempted_an_action(record, config):
            labels.append(MALFORMED_ACTION)
        else:
            labels.append(TEXT_ONLY)
    else:
        if record.tool_name not in config.known_tools:
            labels.append(UNKNOWN_TOOL)
        else:
            required = config.tool_required_args.get(record.tool_name, frozenset())
            supplied = set(record.tool_args or {})
            if required - supplied:
                labels.append(SCHEMA_INVALID_ARGS)

        call = _canonical_call(record)
        window = history[-config.repeat_window :] if config.repeat_window else []
        if any(_canonical_call(previous) == call for previous in window):
            labels.append(EXACT_REPEAT)

    if _looks_like_tool_error(record.tool_result):
        labels.append(TOOL_ERROR)

    return labels


def label_trajectory(
    records: Sequence[StepRecord],
    config: RuleConfig,
) -> list[list[str]]:
    """Rule events for every step of a trajectory, one row per step."""
    labels = [
        _label_step(record, records[:index], config)
        for index, record in enumerate(records)
    ]

    if records and len(records) >= config.step_limit:
        labels[-1].append(STEP_LIMIT)

    return labels
