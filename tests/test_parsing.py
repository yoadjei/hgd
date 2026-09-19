"""Action parsing tests.

Two action formats have to coexist. AppWorld is natively code-as-action (the
agent writes Python against `apis.*` in an IPython shell), while audit section
19 mandates native function calling so that tool interfaces are identical across
model families. The parser therefore reports which format produced an action,
and the format is logged, because a silent switch between them would make
cross-model comparisons meaningless.

Audit section 20 fixes the retry policy at zero: a malformed call is an event,
never a retry. So the parser must distinguish "no action attempted" from
"action attempted and unparseable" — those become different rule events.
"""

import pytest

from hgd.parsing import ActionFormat, parse_action


# --- code-as-action --------------------------------------------------------


def test_extracts_python_from_a_fenced_block():
    raw = "I will list playlists.\n```python\napis.spotify.playlists(access_token=t)\n```"

    parsed = parse_action(raw, ActionFormat.CODE)

    assert parsed.action == "apis.spotify.playlists(access_token=t)"
    assert not parsed.malformed


def test_accepts_a_fence_without_a_language_tag():
    parsed = parse_action("```\napis.spotify.playlists()\n```", ActionFormat.CODE)

    assert parsed.action == "apis.spotify.playlists()"


def test_takes_the_first_fence_when_several_are_present():
    raw = "```python\nfirst()\n```\nand then\n```python\nsecond()\n```"

    assert parse_action(raw, ActionFormat.CODE).action == "first()"


def test_multiline_code_is_preserved():
    raw = "```python\nx = apis.a()\napis.b(x)\n```"

    assert parse_action(raw, ActionFormat.CODE).action == "x = apis.a()\napis.b(x)"


def test_prose_with_no_fence_is_a_text_only_turn():
    parsed = parse_action("Let me think about this first.", ActionFormat.CODE)

    assert parsed.action is None
    assert not parsed.malformed
    assert parsed.text_only


def test_unclosed_fence_is_malformed_not_text_only():
    parsed = parse_action("```python\napis.spotify.playlists(", ActionFormat.CODE)

    assert parsed.action is None
    assert parsed.malformed
    assert not parsed.text_only


def test_empty_fence_is_malformed():
    parsed = parse_action("```python\n\n```", ActionFormat.CODE)

    assert parsed.action is None
    assert parsed.malformed


# --- native function calling ----------------------------------------------


def test_native_tool_call_is_parsed_from_structured_output():
    parsed = parse_action(
        "",
        ActionFormat.NATIVE,
        tool_calls=[{"name": "spotify.playlists", "arguments": {"access_token": "t"}}],
    )

    assert parsed.tool_name == "spotify.playlists"
    assert parsed.tool_args == {"access_token": "t"}
    assert not parsed.malformed


def test_native_call_with_unparseable_arguments_is_malformed():
    parsed = parse_action(
        "",
        ActionFormat.NATIVE,
        tool_calls=[{"name": "spotify.playlists", "arguments": "{not json"}],
    )

    assert parsed.malformed
    assert parsed.tool_name is None


def test_native_arguments_supplied_as_a_json_string_are_decoded():
    """vLLM returns arguments as a JSON string for several chat templates."""
    parsed = parse_action(
        "",
        ActionFormat.NATIVE,
        tool_calls=[{"name": "venmo.send_money", "arguments": '{"amount": 5}'}],
    )

    assert parsed.tool_args == {"amount": 5}


def test_native_response_with_no_tool_call_is_text_only():
    parsed = parse_action("I am done.", ActionFormat.NATIVE, tool_calls=[])

    assert parsed.text_only
    assert not parsed.malformed


def test_only_the_first_native_tool_call_is_taken():
    """One action per step; parallel calls would break position indexing."""
    parsed = parse_action(
        "",
        ActionFormat.NATIVE,
        tool_calls=[
            {"name": "a.one", "arguments": {}},
            {"name": "a.two", "arguments": {}},
        ],
    )

    assert parsed.tool_name == "a.one"


# --- contract --------------------------------------------------------------


def test_parsing_is_deterministic():
    raw = "```python\napis.spotify.playlists()\n```"

    assert parse_action(raw, ActionFormat.CODE) == parse_action(raw, ActionFormat.CODE)


def test_parsed_action_records_the_format_that_produced_it():
    parsed = parse_action("```python\nx()\n```", ActionFormat.CODE)

    assert parsed.action_format is ActionFormat.CODE


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_empty_output_is_text_only_not_malformed(raw):
    parsed = parse_action(raw, ActionFormat.CODE)

    assert parsed.text_only
    assert not parsed.malformed
