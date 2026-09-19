"""Parsing a model turn into an action.

Two formats coexist by necessity. AppWorld is natively code-as-action; audit
section 19 mandates native function calling so tool interfaces are identical
across model families. Which one produced a given action is recorded on the
result and logged, because silently mixing them would invalidate every
cross-model comparison the paper makes.

The three outcomes are deliberately distinct:

* an action was parsed;
* no action was attempted (``text_only``);
* an action was attempted and could not be parsed (``malformed``).

Audit section 20 sets retries to zero, so the third case is an event to be
counted, not a condition to be retried. Collapsing it into the second would
erase a tier-1 rule event.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

# non-greedy so the first fenced block wins: one action per step, because
# position indexing depends on steps being atomic
_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
_FENCE_MARKER = "```"


class ActionFormat(Enum):
    CODE = "code"
    NATIVE = "native"


@dataclass(frozen=True)
class ParsedAction:
    """The outcome of interpreting one model turn."""

    action_format: ActionFormat
    action: str | None = None
    tool_name: str | None = None
    tool_args: Mapping[str, Any] | None = None
    malformed: bool = False
    text_only: bool = False


def _parse_code(raw_output: str) -> dict[str, Any]:
    match = _FENCE.search(raw_output)
    if match is not None:
        code = match.group(1).strip()
        if not code:
            return {"malformed": True}
        return {"action": code}

    if _FENCE_MARKER in raw_output:
        # a fence was opened and never closed: the model tried to act
        return {"malformed": True}

    return {"text_only": True}


def _parse_native(tool_calls: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    if not tool_calls:
        return {"text_only": True}

    call = tool_calls[0]
    name = call.get("name")
    arguments = call.get("arguments")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return {"malformed": True}

    if not isinstance(arguments, Mapping) or not name:
        return {"malformed": True}

    return {
        "action": json.dumps({"name": name, "arguments": arguments}, sort_keys=True),
        "tool_name": name,
        "tool_args": dict(arguments),
    }


def parse_action(
    raw_output: str,
    action_format: ActionFormat,
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
) -> ParsedAction:
    """Interpret one model turn under the configured action format."""
    raw_output = raw_output or ""

    if action_format is ActionFormat.NATIVE:
        parsed = _parse_native(tool_calls)
    else:
        parsed = _parse_code(raw_output)

    return ParsedAction(action_format=action_format, **parsed)
