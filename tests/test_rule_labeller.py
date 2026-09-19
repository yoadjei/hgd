"""Rule labeller tests — audit section 28 tier 1, task T1.3.

These seven events are the deterministic backbone of the whole analysis. Audit
section 11 and prompt section 11 both insist the causal estimates must not
depend on a judge; that is only true if these rules are exhaustive and exact,
so every rule is pinned here including its boundaries.
"""

import pytest

from hgd.rule_labeller import (
    EXACT_REPEAT,
    MALFORMED_ACTION,
    SCHEMA_INVALID_ARGS,
    STEP_LIMIT,
    TEXT_ONLY,
    TOOL_ERROR,
    UNKNOWN_TOOL,
    RuleConfig,
    label_trajectory,
)


@pytest.fixture
def config():
    return RuleConfig(
        known_tools=frozenset({"spotify.playlists", "venmo.send_money"}),
        tool_required_args={
            "spotify.playlists": frozenset({"access_token"}),
            "venmo.send_money": frozenset({"access_token", "amount", "recipient"}),
        },
        step_limit=10,
        repeat_window=5,
    )


# --- text-only vs malformed ------------------------------------------------


def test_turn_with_no_action_and_no_code_fence_is_text_only(make_step, config):
    steps = [make_step(raw_output="I should think about this first.", parsed_action=None,
                       tool_name=None, tool_args=None, tool_result=None)]

    assert TEXT_ONLY in label_trajectory(steps, config)[0]


def test_turn_with_a_code_fence_that_did_not_parse_is_malformed_not_text_only(make_step, config):
    """An attempted call that failed to parse is an error event, never a retry."""
    steps = [make_step(raw_output="```python\napis.spotify.playlists(", parsed_action=None,
                       tool_name=None, tool_args=None, tool_result=None)]

    labels = label_trajectory(steps, config)[0]

    assert MALFORMED_ACTION in labels
    assert TEXT_ONLY not in labels


def test_successful_call_is_neither_text_only_nor_malformed(make_step, config):
    steps = [make_step()]

    labels = label_trajectory(steps, config)[0]

    assert MALFORMED_ACTION not in labels
    assert TEXT_ONLY not in labels


# --- unknown tool ----------------------------------------------------------


def test_call_to_a_tool_outside_the_schema_is_unknown_tool(make_step, config):
    steps = [make_step(tool_name="spotify.delete_universe", tool_args={"access_token": "t"})]

    assert UNKNOWN_TOOL in label_trajectory(steps, config)[0]


def test_known_tool_is_not_flagged(make_step, config):
    assert UNKNOWN_TOOL not in label_trajectory([make_step()], config)[0]


# --- schema-invalid arguments ---------------------------------------------


def test_missing_required_argument_is_schema_invalid(make_step, config):
    steps = [make_step(tool_name="venmo.send_money",
                       tool_args={"access_token": "t", "amount": 5})]

    assert SCHEMA_INVALID_ARGS in label_trajectory(steps, config)[0]


def test_all_required_arguments_present_is_valid(make_step, config):
    steps = [make_step(tool_name="venmo.send_money",
                       tool_args={"access_token": "t", "amount": 5, "recipient": "amy"})]

    assert SCHEMA_INVALID_ARGS not in label_trajectory(steps, config)[0]


def test_unknown_tool_is_not_also_reported_as_schema_invalid(make_step, config):
    """We cannot know the schema of a tool that does not exist; one event, not two."""
    steps = [make_step(tool_name="spotify.nonexistent", tool_args={})]

    labels = label_trajectory(steps, config)[0]

    assert UNKNOWN_TOOL in labels
    assert SCHEMA_INVALID_ARGS not in labels


# --- explicit tool error ---------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        "Traceback (most recent call last):\n  File ...",
        "Exception: invalid access token",
        '{"error": "unauthorized", "status": 401}',
    ],
)
def test_error_shaped_tool_result_is_tool_error(make_step, config, result):
    assert TOOL_ERROR in label_trajectory([make_step(tool_result=result)], config)[0]


def test_ordinary_tool_result_is_not_an_error(make_step, config):
    steps = [make_step(tool_result="[{'id': 1, 'name': 'Chill'}]")]

    assert TOOL_ERROR not in label_trajectory(steps, config)[0]


def test_result_merely_containing_the_word_error_is_not_an_error(make_step, config):
    """A song called 'Error' must not create a spurious event."""
    steps = [make_step(tool_result="[{'id': 7, 'name': 'Trial and Error'}]")]

    assert TOOL_ERROR not in label_trajectory(steps, config)[0]


# --- exact repeated call ---------------------------------------------------


def test_identical_call_within_the_window_is_an_exact_repeat(make_step, config):
    call = {"tool_name": "spotify.playlists", "tool_args": {"access_token": "t"}}
    steps = [make_step(step=0, **call), make_step(step=1, **call)]

    labels = label_trajectory(steps, config)

    assert EXACT_REPEAT not in labels[0]
    assert EXACT_REPEAT in labels[1]


def test_identical_call_outside_the_window_is_not_a_repeat(make_step, config):
    call = {"tool_name": "spotify.playlists", "tool_args": {"access_token": "t"}}
    other = {"tool_name": "venmo.send_money",
             "tool_args": {"access_token": "t", "amount": 1, "recipient": "b"}}
    steps = [make_step(step=0, **call)] + [
        make_step(step=i, **other) for i in range(1, 7)
    ] + [make_step(step=7, **call)]

    assert EXACT_REPEAT not in label_trajectory(steps, config)[7]


def test_same_tool_with_different_arguments_is_not_a_repeat(make_step, config):
    steps = [
        make_step(step=0, tool_name="spotify.playlists", tool_args={"access_token": "t"}),
        make_step(step=1, tool_name="spotify.playlists", tool_args={"access_token": "u"}),
    ]

    assert EXACT_REPEAT not in label_trajectory(steps, config)[1]


def test_argument_key_order_does_not_affect_repeat_detection(make_step, config):
    """Dict ordering is an artefact of generation, not a behavioural difference."""
    steps = [
        make_step(step=0, tool_name="venmo.send_money",
                  tool_args={"access_token": "t", "amount": 5, "recipient": "amy"}),
        make_step(step=1, tool_name="venmo.send_money",
                  tool_args={"recipient": "amy", "amount": 5, "access_token": "t"}),
    ]

    assert EXACT_REPEAT in label_trajectory(steps, config)[1]


def test_text_only_turns_do_not_count_as_repeats(make_step, config):
    steps = [
        make_step(step=0, raw_output="thinking", parsed_action=None, tool_name=None,
                  tool_args=None, tool_result=None),
        make_step(step=1, raw_output="thinking", parsed_action=None, tool_name=None,
                  tool_args=None, tool_result=None),
    ]

    assert EXACT_REPEAT not in label_trajectory(steps, config)[1]


# --- step-limit termination ------------------------------------------------


def test_trajectory_reaching_the_step_limit_flags_its_final_step(make_step, config):
    steps = [make_step(step=i) for i in range(config.step_limit)]

    labels = label_trajectory(steps, config)

    assert STEP_LIMIT in labels[-1]
    assert all(STEP_LIMIT not in row for row in labels[:-1])


def test_trajectory_ending_early_has_no_step_limit_event(make_step, config):
    steps = [make_step(step=i) for i in range(3)]

    assert all(STEP_LIMIT not in row for row in label_trajectory(steps, config))


# --- general contract ------------------------------------------------------


def test_labels_are_returned_one_row_per_step(make_step, config):
    steps = [make_step(step=i) for i in range(4)]

    assert len(label_trajectory(steps, config)) == 4


def test_a_step_can_carry_several_events_at_once(make_step, config):
    """Events are not mutually exclusive; a repeat can also error."""
    call = {"tool_name": "spotify.playlists", "tool_args": {"access_token": "t"}}
    steps = [
        make_step(step=0, **call),
        make_step(step=1, tool_result="Exception: rate limited", **call),
    ]

    labels = label_trajectory(steps, config)[1]

    assert EXACT_REPEAT in labels and TOOL_ERROR in labels


def test_clean_trajectory_produces_no_events(make_step, config):
    steps = [
        make_step(step=0, tool_name="spotify.playlists", tool_args={"access_token": "t"}),
        make_step(step=1, tool_name="venmo.send_money",
                  tool_args={"access_token": "t", "amount": 5, "recipient": "amy"}),
    ]

    assert label_trajectory(steps, config) == [[], []]


def test_empty_trajectory_is_handled(config):
    assert label_trajectory([], config) == []


def test_labelling_is_deterministic(make_step, config):
    steps = [make_step(step=i) for i in range(5)]

    assert label_trajectory(steps, config) == label_trajectory(steps, config)


def test_labelling_does_not_mutate_the_input_records(make_step, config):
    steps = [make_step(step=0, tool_name="bogus.tool", tool_args={})]
    before = steps[0].to_json()

    label_trajectory(steps, config)

    assert steps[0].to_json() == before
