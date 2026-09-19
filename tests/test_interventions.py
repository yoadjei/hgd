"""Intervention tests — prompt section 7, audit section 20 and 26.

Three interventions, three mechanisms. Audit section 20 requires each to be "a
pure function of logged data so replays are reproducible": an intervention that
consulted anything outside the log could not be reproduced from released
artefacts, and every causal estimate in the paper would become unverifiable.

What each holds fixed is the whole of its identification claim:

  history_scrub  — transcript altered, environment state factual  -> self-conditioning
  state_refresh  — oracle state summary added, transcript factual -> state staleness
  oracle_fix     — one action replaced, prefix factual            -> irrecoverability

The placebo is not decoration. Audit section 32's R1 makes it a condition of
acceptance, and Shao et al. (arXiv:2608.20563) match theirs on byte and line
count, so ours does too and is tested for it.
"""

import pytest

from hgd.env import DictEnvironment
from hgd.interventions import (
    HISTORY_SCRUB,
    ORACLE_FIX,
    PLACEBO_SUMMARY,
    STATE_REFRESH,
    Intervention,
    apply_intervention,
    oracle_summary,
    placebo_summary,
    scrub_history,
)


@pytest.fixture
def messages():
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "assistant", "content": "```python\nstore(a, 1)\n```"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "```python\nstore(b, WRONG)\n```"},
        {"role": "user", "content": "Exception: bad value"},
        {"role": "assistant", "content": "```python\nread(b)\n```"},
        {"role": "user", "content": "KeyError: missing key"},
    ]


# --- history scrub ---------------------------------------------------------


def test_scrub_removes_the_erroneous_turn_and_its_observation(messages):
    scrubbed = scrub_history(messages, error_steps={1})

    contents = [m["content"] for m in scrubbed]
    assert "```python\nstore(b, WRONG)\n```" not in contents
    assert "Exception: bad value" not in contents


def test_scrub_keeps_every_other_turn_in_order(messages):
    scrubbed = scrub_history(messages, error_steps={1})

    assert [m["content"] for m in scrubbed] == [
        "You are an agent.",
        "```python\nstore(a, 1)\n```",
        "ok",
        "```python\nread(b)\n```",
        "KeyError: missing key",
    ]


def test_scrub_preserves_the_system_prompt(messages):
    scrubbed = scrub_history(messages, error_steps={0, 1, 2})

    assert scrubbed[0]["role"] == "system"


def test_scrub_of_nothing_is_the_identity(messages):
    assert scrub_history(messages, error_steps=set()) == messages


def test_scrub_does_not_mutate_the_input(messages):
    before = [dict(m) for m in messages]

    scrub_history(messages, error_steps={1})

    assert messages == before


def test_scrub_leaves_environment_state_untouched():
    """The point of the contrast: the transcript is cleaned, the world is not.

    This is what separates self-conditioning from staleness. If scrubbing also
    repaired the environment the two mechanisms would be confounded.
    """
    env = DictEnvironment()
    env.reset("t", 0)
    env.execute("store(a, 1)")
    before = env.state_hash()

    scrub_history([{"role": "user", "content": "x"}], error_steps={0})

    assert env.state_hash() == before


# --- state refresh ---------------------------------------------------------


def test_oracle_summary_is_derived_from_environment_state():
    env = DictEnvironment()
    env.reset("t", 0)
    env.execute("store(a, 1)")

    summary = oracle_summary(env.snapshot())

    assert "a" in summary and "1" in summary


def test_oracle_summary_of_identical_states_is_identical():
    assert oracle_summary({"a": "1"}) == oracle_summary({"a": "1"})


def test_placebo_matches_the_oracle_summary_on_bytes_and_lines():
    """Shao et al. match on byte and line count; anything less is not a control."""
    real = oracle_summary({"a": "1", "b": "2", "c": "3"})

    placebo = placebo_summary(real)

    assert len(placebo.encode()) == len(real.encode())
    assert placebo.count("\n") == real.count("\n")


def test_placebo_carries_no_task_relevant_state():
    real = oracle_summary({"secret_key": "value42"})

    placebo = placebo_summary(real)

    assert "secret_key" not in placebo
    assert "value42" not in placebo


def test_placebo_is_deterministic():
    real = oracle_summary({"a": "1"})

    assert placebo_summary(real) == placebo_summary(real)


# --- applying interventions ------------------------------------------------


def test_history_scrub_alters_messages_and_forces_no_action(messages):
    result = apply_intervention(
        Intervention(kind=HISTORY_SCRUB, step=3, error_steps={1}),
        messages=messages,
        state={},
    )

    assert result.forced_action is None
    assert len(result.messages) < len(messages)


def test_state_refresh_appends_a_summary_and_leaves_history_intact(messages):
    result = apply_intervention(
        Intervention(kind=STATE_REFRESH, step=3),
        messages=messages,
        state={"a": "1"},
    )

    assert len(result.messages) == len(messages) + 1
    assert [m["content"] for m in messages] == [
        m["content"] for m in result.messages[:-1]
    ]


def test_placebo_refresh_appends_a_matched_message(messages):
    real = apply_intervention(
        Intervention(kind=STATE_REFRESH, step=3), messages=messages, state={"a": "1"}
    )
    sham = apply_intervention(
        Intervention(kind=PLACEBO_SUMMARY, step=3), messages=messages, state={"a": "1"}
    )

    assert len(sham.messages) == len(real.messages)
    assert len(sham.messages[-1]["content"].encode()) == len(
        real.messages[-1]["content"].encode()
    )
    assert sham.messages[-1]["content"] != real.messages[-1]["content"]


def test_oracle_fix_forces_the_corrected_action_without_touching_history(messages):
    result = apply_intervention(
        Intervention(kind=ORACLE_FIX, step=3, corrected_action="store(b, 2)"),
        messages=messages,
        state={},
    )

    assert result.forced_action == "store(b, 2)"
    assert result.messages == messages


def test_oracle_fix_without_a_corrected_action_is_rejected(messages):
    with pytest.raises(ValueError, match="corrected_action"):
        apply_intervention(
            Intervention(kind=ORACLE_FIX, step=3), messages=messages, state={}
        )


def test_unknown_intervention_kind_is_rejected(messages):
    with pytest.raises(ValueError, match="kind"):
        apply_intervention(
            Intervention(kind="wishful_thinking", step=1), messages=messages, state={}
        )


def test_interventions_are_pure_functions_of_logged_data(messages):
    """Same inputs, same output — otherwise released logs cannot reproduce the branch."""
    intervention = Intervention(kind=HISTORY_SCRUB, step=3, error_steps={1})

    first = apply_intervention(intervention, messages=messages, state={"a": "1"})
    second = apply_intervention(intervention, messages=messages, state={"a": "1"})

    assert first.messages == second.messages
