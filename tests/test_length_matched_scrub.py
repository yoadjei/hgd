"""Length-matched history scrub — design change D10, from the Phase 2 probe.

The Phase 2 identification probe measured history-scrub overestimating theta_self
by +0.071 on a true +0.532. The mechanism: removing turns shortens the
transcript, and a length-sensitive policy improves for reasons unrelated to
erroneous history. Real models are length-sensitive — that is Peng et al.'s
"context rot" — so this bias is expected to be present, not hypothetical.

The fix mirrors `placebo_summary`: instead of deleting the erroneous turns,
replace their content with neutral filler matched on byte and line count. Turn
count, position structure and transcript length are all preserved, so the only
channel left is the one the intervention names.

Removal mode is kept, because it is the operational analogue of ChainSWE's Seq
mode (fresh transcript over an environment that still carries the damage) and we
want to be able to report both.
"""

import pytest

from hgd.interventions import (
    HISTORY_SCRUB,
    Intervention,
    ScrubMode,
    apply_intervention,
    scrub_history,
)
from hgd.synthetic import error_turns_in_history


@pytest.fixture
def messages():
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "assistant", "content": "```python\nstore(a, 1)\n```"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "```python\nstore(WRONG_1)\n```"},
        {"role": "user", "content": "TypeError: store expects (key, value)"},
    ]


def test_removal_mode_is_still_the_default(messages):
    scrubbed = scrub_history(messages, error_steps={1})

    assert len(scrubbed) == 3


def test_length_matched_mode_keeps_the_turn_count(messages):
    scrubbed = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert len(scrubbed) == len(messages)


def test_length_matched_mode_preserves_total_byte_length(messages):
    scrubbed = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    before = sum(len(m["content"].encode()) for m in messages)
    after = sum(len(m["content"].encode()) for m in scrubbed)

    assert after == before


def test_length_matched_mode_preserves_line_counts_per_message(messages):
    scrubbed = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert [m["content"].count("\n") for m in scrubbed] == [
        m["content"].count("\n") for m in messages
    ]


def test_length_matched_mode_preserves_roles(messages):
    scrubbed = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert [m["role"] for m in scrubbed] == [m["role"] for m in messages]


def test_length_matched_mode_still_removes_the_error_signal(messages):
    """The whole point: the error is gone even though the bytes remain."""
    scrubbed = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert error_turns_in_history(messages) > 0
    assert error_turns_in_history(scrubbed) == 0


def test_length_matched_mode_leaves_untouched_turns_verbatim(messages):
    scrubbed = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert scrubbed[0]["content"] == messages[0]["content"]
    assert scrubbed[1]["content"] == messages[1]["content"]
    assert scrubbed[2]["content"] == messages[2]["content"]


def test_length_matched_mode_carries_no_task_content(messages):
    scrubbed = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert "WRONG_1" not in scrubbed[3]["content"]
    assert "TypeError" not in scrubbed[4]["content"]


def test_length_matched_scrub_of_nothing_is_the_identity(messages):
    scrubbed = scrub_history(messages, error_steps=set(), mode=ScrubMode.LENGTH_MATCHED)

    assert scrubbed == messages


def test_length_matched_mode_is_deterministic(messages):
    first = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)
    second = scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert first == second


def test_length_matched_mode_does_not_mutate_the_input(messages):
    before = [dict(m) for m in messages]

    scrub_history(messages, error_steps={1}, mode=ScrubMode.LENGTH_MATCHED)

    assert messages == before


def test_apply_intervention_honours_the_scrub_mode(messages):
    """The mode must survive the intervention wrapper, not only the raw helper."""
    result = apply_intervention(
        Intervention(
            kind=HISTORY_SCRUB, step=2, error_steps={1},
            scrub_mode=ScrubMode.LENGTH_MATCHED,
        ),
        messages=messages,
        state={},
    )

    assert len(result.messages) == len(messages)
    assert error_turns_in_history(result.messages) == 0


def test_apply_intervention_defaults_to_removal(messages):
    result = apply_intervention(
        Intervention(kind=HISTORY_SCRUB, step=2, error_steps={1}),
        messages=messages,
        state={},
    )

    assert len(result.messages) < len(messages)
