"""vLLM client — a ``Model`` over the OpenAI-compatible HTTP API.

Why HTTP rather than vLLM's in-process ``LLM`` class: the server batches
concurrent requests automatically, which is what makes the compute budget
feasible. Episodes are run in a thread pool, each stepping sequentially, and the
server interleaves them. The in-process API would force us to restructure the
harness around lockstep batching for the same effect.

**On determinism.** A fixed ``seed`` makes a single request reproducible, but
vLLM's batched execution is not bitwise reproducible across different batch
compositions, so re-issuing the same request in a different batch may not return
the same text. This does not threaten replay fidelity: replay re-executes logged
*actions*, never model calls, so the environment is driven identically regardless
of sampling. It does mean branch continuations genuinely resample — which is why
the audit's estimator draws k continuations per branch and compares
distributions rather than single outcomes.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from hgd.model import ModelResponse


class VLLMError(RuntimeError):
    """Raised when the server returns something we cannot use."""


def _requests_transport(url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    import requests

    response = requests.post(url, json=payload, timeout=600)
    return response.json()


class VLLMClient:
    """Chat-completions client implementing the ``Model`` protocol."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        seed: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        transport: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._transport = transport or _requests_transport

    @property
    def name(self) -> str:
        return f"{self.model}@seed{self.seed}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        if tools:
            payload["tools"] = list(tools)

        body = self._transport(self.endpoint, payload)

        choices = body.get("choices")
        if choices is None:
            detail = (body.get("error") or {})
            message = detail.get("message") if isinstance(detail, Mapping) else detail
            raise VLLMError(f"server returned no choices: {message or body}")
        if not choices:
            raise VLLMError("server returned no choices (empty list)")

        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}

        return ModelResponse(
            # content is null on a pure tool-call turn; the schema forbids a null
            # raw_output
            text=message.get("content") or "",
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            tool_calls=_normalise_tool_calls(message.get("tool_calls")),
        )


def _normalise_tool_calls(
    raw: Sequence[Mapping[str, Any]] | None
) -> tuple[Mapping[str, Any], ...]:
    """Flatten OpenAI's nested tool-call shape to ``{name, arguments}``.

    ``hgd.parsing.parse_action`` expects the flat form and already handles
    arguments arriving as either a JSON string or a mapping.
    """
    if not raw:
        return ()

    flattened: list[Mapping[str, Any]] = []
    for call in raw:
        function = call.get("function") or call
        flattened.append(
            {
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return tuple(flattened)
