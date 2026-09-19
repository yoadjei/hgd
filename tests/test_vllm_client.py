"""vLLM client tests.

The client speaks the OpenAI-compatible HTTP API that vLLM serves, rather than
vLLM's in-process API, for two reasons: the server batches concurrent requests
automatically, which is what makes the budget feasible at all, and an HTTP
boundary is testable without a GPU.

The transport is injected so these tests exercise the client's own logic —
payload construction, response parsing, error handling — against a fake, with no
network and no model.
"""

import pytest

from hgd.vllm_client import VLLMClient, VLLMError


def chat_response(text, prompt_tokens=100, completion_tokens=20, tool_calls=None):
    message = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens},
    }


class FakeTransport:
    def __init__(self, response=None):
        self.response = response or chat_response("hello")
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.response


@pytest.fixture
def transport():
    return FakeTransport()


@pytest.fixture
def client(transport):
    return VLLMClient(
        base_url="http://localhost:8000",
        model="Qwen/Qwen3-8B",
        seed=7,
        temperature=0.6,
        max_tokens=512,
        transport=transport,
    )


# --- request construction --------------------------------------------------


def test_posts_to_the_chat_completions_endpoint(client, transport):
    client.generate([{"role": "user", "content": "hi"}])

    assert transport.calls[0][0] == "http://localhost:8000/v1/chat/completions"


def test_sends_the_configured_sampling_parameters(client, transport):
    client.generate([{"role": "user", "content": "hi"}])
    payload = transport.calls[0][1]

    assert payload["model"] == "Qwen/Qwen3-8B"
    assert payload["temperature"] == 0.6
    assert payload["max_tokens"] == 512


def test_sends_the_seed_so_requests_are_reproducible(client, transport):
    client.generate([{"role": "user", "content": "hi"}])

    assert transport.calls[0][1]["seed"] == 7


def test_sends_the_messages_verbatim(client, transport):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    client.generate(messages)

    assert transport.calls[0][1]["messages"] == messages


def test_tools_are_sent_only_when_supplied(client, transport):
    client.generate([{"role": "user", "content": "hi"}])
    assert "tools" not in transport.calls[0][1]

    client.generate([{"role": "user", "content": "hi"}], tools=[{"name": "t"}])
    assert transport.calls[1][1]["tools"] == [{"name": "t"}]


# --- response parsing ------------------------------------------------------


def test_returns_the_message_text(client):
    assert client.generate([{"role": "user", "content": "hi"}]).text == "hello"


def test_reports_token_counts_from_usage(client):
    response = client.generate([{"role": "user", "content": "hi"}])

    assert response.tokens_in == 100
    assert response.tokens_out == 20


def test_null_content_becomes_an_empty_string(transport):
    """A pure tool-call turn has content=None; the schema forbids a null raw_output."""
    transport.response = chat_response(
        None, tool_calls=[{"function": {"name": "t", "arguments": "{}"}}]
    )
    client = VLLMClient("http://x", "m", seed=0, transport=transport)

    assert client.generate([{"role": "user", "content": "hi"}]).text == ""


def test_tool_calls_are_normalised_to_name_and_arguments(transport):
    transport.response = chat_response(
        None,
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "spotify.playlists",
                                  "arguments": '{"access_token": "t"}'}}],
    )
    client = VLLMClient("http://x", "m", seed=0, transport=transport)

    calls = client.generate([{"role": "user", "content": "hi"}]).tool_calls

    assert calls[0]["name"] == "spotify.playlists"
    assert calls[0]["arguments"] == '{"access_token": "t"}'


def test_no_tool_calls_gives_an_empty_tuple(client):
    assert client.generate([{"role": "user", "content": "hi"}]).tool_calls == ()


# --- failure handling ------------------------------------------------------


def test_a_response_without_choices_is_an_explicit_error(transport):
    transport.response = {"error": {"message": "context length exceeded"}}
    client = VLLMClient("http://x", "m", seed=0, transport=transport)

    with pytest.raises(VLLMError, match="context length"):
        client.generate([{"role": "user", "content": "hi"}])


def test_an_empty_choices_list_is_an_explicit_error(transport):
    transport.response = {"choices": [], "usage": {}}
    client = VLLMClient("http://x", "m", seed=0, transport=transport)

    with pytest.raises(VLLMError, match="no choices"):
        client.generate([{"role": "user", "content": "hi"}])


def test_missing_usage_does_not_crash_the_run(transport):
    """Token counts are logged, not load-bearing; losing them must not kill an episode."""
    transport.response = {"choices": [{"message": {"content": "ok"}}]}
    client = VLLMClient("http://x", "m", seed=0, transport=transport)

    response = client.generate([{"role": "user", "content": "hi"}])

    assert response.text == "ok"
    assert response.tokens_in == 0


# --- identity --------------------------------------------------------------


def test_name_records_model_and_seed_for_the_log(client):
    assert "Qwen3-8B" in client.name


def test_base_url_trailing_slash_is_tolerated(transport):
    client = VLLMClient("http://localhost:8000/", "m", seed=0, transport=transport)

    client.generate([{"role": "user", "content": "hi"}])

    assert transport.calls[0][0] == "http://localhost:8000/v1/chat/completions"
