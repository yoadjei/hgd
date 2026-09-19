"""Planted-mechanism policy — audit section 24 and Phase 2.

Phase 2 asks whether the estimators recover mechanisms whose true strength is
known. That needs a policy whose self-conditioning and staleness sensitivities
are parameters rather than emergent properties, so that a recovered share can be
compared against a number we set.

The policy is a real ``Model``: it plugs into the same harness, produces the same
logs, and is branched by the same interventions as a language model would be.
That is the point — a validation that bypassed the harness would validate
nothing about the pipeline that Phase 3 actually runs.
"""

import pytest

from hgd.interventions import STATE_REFRESH, Intervention, apply_intervention, scrub_history
from hgd.parsing import ActionFormat, parse_action
from hgd.synthetic import PlantedPolicy, error_turns_in_history, has_fresh_state_summary


ERROR_HISTORY = [
    {"role": "assistant", "content": "```python\nstore(a, 1)\n```"},
    {"role": "user", "content": "Exception: bad value"},
]

CLEAN_HISTORY = [
    {"role": "assistant", "content": "```python\nstore(a, 1)\n```"},
    {"role": "user", "content": "ok"},
]


# --- history and state detection ------------------------------------------


def test_error_observations_in_history_are_counted():
    assert error_turns_in_history(ERROR_HISTORY) == 1


def test_clean_history_has_no_error_turns():
    assert error_turns_in_history(CLEAN_HISTORY) == 0


def test_scrubbed_history_has_no_error_turns():
    """The history-scrub intervention must actually remove what the policy reads."""
    scrubbed = scrub_history(ERROR_HISTORY, error_steps={0})

    assert error_turns_in_history(scrubbed) == 0


def test_state_summary_is_detected_after_a_refresh():
    result = apply_intervention(
        Intervention(kind=STATE_REFRESH, step=1),
        messages=CLEAN_HISTORY,
        state={"a": "1"},
    )

    assert has_fresh_state_summary(result.messages)


def test_no_state_summary_in_ordinary_history():
    assert not has_fresh_state_summary(CLEAN_HISTORY)


# --- planted mechanism strengths ------------------------------------------


def test_error_rate_is_the_base_rate_on_clean_history_with_fresh_state():
    policy = PlantedPolicy(
        base_error_rate=0.1, self_conditioning=0.4, staleness=0.3, seed=0
    )

    assert policy.error_probability(CLEAN_HISTORY, state_is_fresh=True) == pytest.approx(0.1)


def test_erroneous_history_adds_the_self_conditioning_term():
    policy = PlantedPolicy(
        base_error_rate=0.1, self_conditioning=0.4, staleness=0.0, seed=0
    )

    assert policy.error_probability(ERROR_HISTORY, state_is_fresh=True) == pytest.approx(0.5)


def test_stale_state_adds_the_staleness_term():
    policy = PlantedPolicy(
        base_error_rate=0.1, self_conditioning=0.0, staleness=0.3, seed=0
    )

    assert policy.error_probability(CLEAN_HISTORY, state_is_fresh=False) == pytest.approx(0.4)


def test_mechanisms_are_additive_and_capped_at_one():
    policy = PlantedPolicy(
        base_error_rate=0.5, self_conditioning=0.4, staleness=0.4, seed=0
    )

    assert policy.error_probability(ERROR_HISTORY, state_is_fresh=False) == pytest.approx(1.0)


def test_a_policy_with_no_planted_mechanisms_is_flat():
    """The null against which a recovered share must be indistinguishable from zero."""
    policy = PlantedPolicy(
        base_error_rate=0.2, self_conditioning=0.0, staleness=0.0, seed=0
    )

    assert policy.error_probability(ERROR_HISTORY, state_is_fresh=False) == pytest.approx(0.2)


# --- behaviour as a Model -------------------------------------------------


def test_policy_emits_a_parseable_action():
    policy = PlantedPolicy(base_error_rate=0.0, self_conditioning=0.0, staleness=0.0, seed=0)

    response = policy.generate(CLEAN_HISTORY)

    assert parse_action(response.text, ActionFormat.CODE).action is not None


def test_policy_is_deterministic_given_a_seed():
    def run():
        policy = PlantedPolicy(
            base_error_rate=0.5, self_conditioning=0.0, staleness=0.0, seed=11
        )
        return [policy.generate(CLEAN_HISTORY).text for _ in range(10)]

    assert run() == run()


def test_different_seeds_give_different_realisations():
    def run(seed):
        policy = PlantedPolicy(
            base_error_rate=0.5, self_conditioning=0.0, staleness=0.0, seed=seed
        )
        return [policy.generate(CLEAN_HISTORY).text for _ in range(20)]

    assert run(1) != run(2)


def test_higher_error_probability_produces_more_errors_in_expectation():
    def error_fraction(rate):
        policy = PlantedPolicy(
            base_error_rate=rate, self_conditioning=0.0, staleness=0.0, seed=3
        )
        texts = [policy.generate(CLEAN_HISTORY).text for _ in range(400)]
        return sum("WRONG" in t for t in texts) / len(texts)

    assert error_fraction(0.8) > error_fraction(0.2)


def test_policy_reports_a_name_for_the_log():
    policy = PlantedPolicy(base_error_rate=0.1, self_conditioning=0.4, staleness=0.3, seed=0)

    assert "planted" in policy.name
