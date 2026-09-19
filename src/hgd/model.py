"""Model interface.

The harness depends on this protocol rather than on vLLM, so the loop, the
logging and the replay machinery are all testable without a GPU. ``ScriptedModel``
is not a test double: replay branches re-execute a logged action prefix with no
sampling, which is exactly a scripted policy, and Phase 2's planted-error
conditions are scripted by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ModelResponse:
    text: str
    tokens_in: int
    tokens_out: int
    tool_calls: tuple[Mapping[str, Any], ...] = ()


class Model(Protocol):
    """Anything that can take a message history and return one turn."""

    @property
    def name(self) -> str:
        """Model identifier, logged on every step."""

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ModelResponse: ...


def approx_tokens(text: str) -> int:
    """Cheap token proxy for scripted and synthetic models.

    Real counts come from the inference stack in Phase 3; these are logged, not
    load-bearing.
    """
    return max(1, len(text) // 4) if text else 0


class ScriptedModel:
    """Emits a fixed sequence of turns, then falls silent.

    Falling silent rather than raising lets a caller exercise the step limit
    without having to script every step up to it.
    """

    def __init__(self, turns: Sequence[str | ModelResponse], name: str = "scripted") -> None:
        self._turns = list(turns)
        self._index = 0
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ModelResponse:
        tokens_in = sum(approx_tokens(str(m.get("content", ""))) for m in messages)

        if self._index >= len(self._turns):
            return ModelResponse(text="", tokens_in=tokens_in, tokens_out=0)

        turn = self._turns[self._index]
        self._index += 1

        if isinstance(turn, ModelResponse):
            return turn
        return ModelResponse(
            text=turn, tokens_in=tokens_in, tokens_out=approx_tokens(turn)
        )
