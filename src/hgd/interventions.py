"""The three interventions — prompt section 7, audit sections 20 and 26.

Each is a pure function of logged data, as audit section 20 requires: an
intervention that consulted anything outside the log could not be reproduced
from released artefacts, and every causal estimate resting on it would be
unverifiable.

What each holds fixed *is* its identification claim:

* ``history_scrub`` — ``do(H_t := H_t^clean)`` with environment state left
  factual. Identifies self-conditioning. The environment is deliberately **not**
  repaired: repairing it too would confound self-conditioning with staleness,
  and the whole decomposition would collapse into a single arm.
* ``state_refresh`` — an oracle state summary derived from the environment, with
  the transcript left factual. Identifies staleness, *under the assumption* that
  the summary changes behaviour only through the belief channel. That assumption
  is not free, which is why ``placebo_summary`` exists.
* ``oracle_fix`` — ``do(A_t := A_t^fix)`` with the prefix factual. Identifies the
  downstream propagation of one specific error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Collection, Mapping, Sequence

FACTUAL = "factual"
HISTORY_SCRUB = "history_scrub"
STATE_REFRESH = "state_refresh"
ORACLE_FIX = "oracle_fix"
PLACEBO_SUMMARY = "placebo_summary"

INTERVENTION_KINDS: tuple[str, ...] = (
    FACTUAL,
    HISTORY_SCRUB,
    STATE_REFRESH,
    ORACLE_FIX,
    PLACEBO_SUMMARY,
)

# public because the planted policy detects a refresh by looking for it
SUMMARY_HEADER = "Current environment state:"
_FILLER = "This line contains no task-relevant information. "


class ScrubMode(Enum):
    """How erroneous turns are removed from the transcript.

    ``REMOVE`` deletes them. It is the operational analogue of ChainSWE's Seq
    mode — a fresh transcript over an environment that still carries the damage —
    but it also shortens the context, which is a second causal channel.

    ``LENGTH_MATCHED`` replaces their content with neutral filler matched on byte
    and line count, so turn count, position structure and length are all
    preserved and the erroneous-history channel is the only one altered. Design
    change D10, added after the Phase 2 identification probe measured a +0.071
    bias in theta_self attributable to the length channel.
    """

    REMOVE = "remove"
    LENGTH_MATCHED = "length_matched"


@dataclass(frozen=True)
class Intervention:
    kind: str
    step: int
    error_steps: Collection[int] = field(default_factory=frozenset)
    corrected_action: str | None = None
    scrub_mode: ScrubMode = ScrubMode.REMOVE


@dataclass(frozen=True)
class InterventionResult:
    messages: list[dict[str, Any]]
    forced_action: str | None = None


def _assistant_turn_spans(
    messages: Sequence[Mapping[str, Any]]
) -> list[tuple[int, int]]:
    """Index span of each agent turn: the assistant message plus its observations.

    Derived from message roles rather than assuming strict alternation or the
    presence of a system prompt, so that scrubbing stays correct for text-only
    turns, which produce no observation at all.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None

    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            if start is not None:
                spans.append((start, index))
            start = index

    if start is not None:
        spans.append((start, len(messages)))

    return spans


def scrub_history(
    messages: Sequence[Mapping[str, Any]],
    error_steps: Collection[int],
    mode: ScrubMode = ScrubMode.REMOVE,
) -> list[dict[str, Any]]:
    """Take the named agent turns out of the transcript.

    This is the operational meaning of "the agent's own errors are no longer in
    context". See ``ScrubMode`` for why there are two ways of doing it and what
    each one leaves confounded.
    """
    if not error_steps:
        return [dict(message) for message in messages]

    spans = _assistant_turn_spans(messages)
    targeted: set[int] = set()
    for step in error_steps:
        if 0 <= step < len(spans):
            start, end = spans[step]
            targeted.update(range(start, end))

    if mode is ScrubMode.REMOVE:
        return [
            dict(message)
            for index, message in enumerate(messages)
            if index not in targeted
        ]

    scrubbed: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        if index in targeted:
            copied["content"] = placebo_summary(str(message.get("content", "")))
        scrubbed.append(copied)
    return scrubbed


def oracle_summary(state: Mapping[str, Any]) -> str:
    """Environment-derived state summary.

    Derived from the environment, never from the model: a model-written summary
    would carry the model's own errors and would re-introduce exactly the
    channel this intervention is meant to hold fixed.
    """
    lines = [SUMMARY_HEADER]
    lines.extend(f"{key}: {state[key]}" for key in sorted(state, key=str))
    return "\n".join(lines)


def _fill(n_bytes: int) -> str:
    if n_bytes <= 0:
        return ""
    repeats = (n_bytes // len(_FILLER)) + 1
    return (_FILLER * repeats)[:n_bytes]


def placebo_summary(real_summary: str) -> str:
    """A summary-shaped message carrying no state, matched byte for byte.

    Shao et al. (arXiv:2608.20563) match rescue and placebo on byte and line
    count; audit section 32's R1 makes this control a condition of acceptance.
    Without it, a state-refresh effect cannot be separated from the effect of
    simply appending an authoritative-looking message.
    """
    return "\n".join(
        _fill(len(line.encode())) for line in real_summary.split("\n")
    )


def apply_intervention(
    intervention: Intervention,
    *,
    messages: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> InterventionResult:
    """Produce the branch's starting messages and any forced action."""
    kind = intervention.kind
    copied = [dict(message) for message in messages]

    if kind == FACTUAL:
        return InterventionResult(messages=copied)

    if kind == HISTORY_SCRUB:
        return InterventionResult(
            messages=scrub_history(
                messages, intervention.error_steps, intervention.scrub_mode
            )
        )

    if kind == STATE_REFRESH:
        copied.append({"role": "user", "content": oracle_summary(state)})
        return InterventionResult(messages=copied)

    if kind == PLACEBO_SUMMARY:
        copied.append(
            {"role": "user", "content": placebo_summary(oracle_summary(state))}
        )
        return InterventionResult(messages=copied)

    if kind == ORACLE_FIX:
        if intervention.corrected_action is None:
            raise ValueError("oracle_fix requires a corrected_action")
        return InterventionResult(
            messages=copied, forced_action=intervention.corrected_action
        )

    raise ValueError(
        f"unknown intervention kind {kind!r}; expected one of {INTERVENTION_KINDS}"
    )
