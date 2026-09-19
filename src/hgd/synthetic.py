"""Planted-mechanism policy for Phase 2 — audit section 24.

Phase 2's acceptance condition is that the estimators "recover planted effects
within their uncertainty intervals". That requires ground truth, which a language
model cannot supply: its self-conditioning strength is exactly the unknown we are
trying to estimate. So the synthetic policy makes those strengths *parameters*.

The policy errs with probability

    p = base + self_conditioning * 1[errors in history]
             + staleness        * 1[no fresh state summary]

capped at one. The interventions then have known targets:

* ``history_scrub`` removes the erroneous turns, so the self-conditioning term
  vanishes and the recovered ``theta_self`` should track ``self_conditioning``;
* ``state_refresh`` appends an oracle summary, so the staleness term vanishes;
* ``oracle_fix`` replaces a destructive action, so propagation is measurable
  against a known irreversibility.

It is a real ``Model``, driven through the same harness, logged through the same
schema and branched by the same interventions as a language model would be. A
validation that bypassed the harness would say nothing about the pipeline Phase 3
actually runs.

Detection is deliberately coupled to the interventions' own output format: if
``scrub_history`` stopped removing what the policy reads, or ``oracle_summary``
changed shape, these tests fail. That coupling is the point — it is an end-to-end
check that the intervention does what its name claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from hgd.interventions import SUMMARY_HEADER
from hgd.model import ModelResponse, approx_tokens
from hgd.rule_labeller import EXCEPTION_LINE

GOOD_ACTION = "store(k{index}, {index})"
# wrong arity on purpose: the environment must reject it so an error lands in
# the transcript. store(k, WRONG) would succeed and leave history_scrub nothing
# to remove, and theta_self would be unidentified while every unit test passed.
BAD_ACTION = "store(WRONG_{index})"
DESTRUCTIVE_ACTION = "delete(k{index})"


def error_turns_in_history(messages: Sequence[Mapping[str, Any]]) -> int:
    """How many observations in the transcript report an error.

    This is what "the agent's own errors are in its context" means operationally,
    and it is what ``history_scrub`` is supposed to drive to zero.
    """
    return sum(
        1
        for message in messages
        if message.get("role") != "assistant"
        and EXCEPTION_LINE.search(str(message.get("content", "")))
    )


def has_fresh_state_summary(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Whether an oracle state summary is present in the transcript."""
    return any(
        SUMMARY_HEADER in str(message.get("content", "")) for message in messages
    )


class PlantedPolicy:
    """A stochastic policy with known mechanism sensitivities."""

    def __init__(
        self,
        *,
        base_error_rate: float,
        self_conditioning: float,
        staleness: float,
        seed: int,
        length_sensitivity: float = 0.0,
        n_keys: int = 8,
        destructive_step: int | None = None,
        probe_action: str = DESTRUCTIVE_ACTION,
    ) -> None:
        self.base_error_rate = base_error_rate
        self.self_conditioning = self_conditioning
        self.staleness = staleness
        # planted confound: scrubbing shortens the transcript, so a
        # length-sensitive policy moves through a second channel and the
        # theta_self bias becomes measurable instead of assumed away
        self.length_sensitivity = length_sensitivity
        self.n_keys = n_keys
        self.destructive_step = destructive_step
        # planted at destructive_step; a harmless read is the T2.2 control arm
        self.probe_action = probe_action
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._seed = seed

    @property
    def name(self) -> str:
        return (
            f"planted(base={self.base_error_rate},self={self.self_conditioning},"
            f"state={self.staleness},seed={self._seed})"
        )

    def error_probability(
        self,
        messages: Sequence[Mapping[str, Any]],
        state_is_fresh: bool | None = None,
    ) -> float:
        """Planted error probability for this context."""
        if state_is_fresh is None:
            state_is_fresh = has_fresh_state_summary(messages)

        probability = self.base_error_rate
        if error_turns_in_history(messages) > 0:
            probability += self.self_conditioning
        if not state_is_fresh:
            probability += self.staleness
        probability += self.length_sensitivity * len(messages)

        return min(1.0, max(0.0, probability))

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ModelResponse:
        index = self._step % self.n_keys
        self._step += 1

        if self.destructive_step is not None and self._step - 1 == self.destructive_step:
            body = self.probe_action.format(index=index)
        elif self._rng.random() < self.error_probability(messages):
            body = BAD_ACTION.format(index=index)
        else:
            body = GOOD_ACTION.format(index=index)

        text = f"```python\n{body}\n```"
        return ModelResponse(
            text=text,
            tokens_in=sum(approx_tokens(str(m.get("content", ""))) for m in messages),
            tokens_out=approx_tokens(text),
        )
