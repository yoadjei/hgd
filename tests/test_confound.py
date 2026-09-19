"""Length confound — making the Phase 2 validation able to fail.

The first Phase 2 run recovered both planted shares with zero error. That was
tautological: scrubbing drives the error count to zero, which makes the policy's
error probability *identically* the `self_conditioning = 0` counterfactual, and
with a shared RNG stream the two arms are bit-identical. A test that cannot fail
validates arithmetic, not identification.

The identification assumption behind history_scrub is that scrubbing changes
behaviour *only* through the erroneous-history channel. A policy that is also
sensitive to history *length* violates it, because scrubbing shortens the
transcript. Planting that violation lets us measure the resulting bias instead
of assuming it away — which is the substance of the reviewer objection that the
effect is an artefact of the intervention.
"""

import pytest

from hgd.synthetic import PlantedPolicy

SHORT = [{"role": "user", "content": "x"}]
LONG = [{"role": "user", "content": "x"}] * 40


def test_length_insensitive_policy_ignores_transcript_length():
    policy = PlantedPolicy(
        base_error_rate=0.2, self_conditioning=0.0, staleness=0.0,
        length_sensitivity=0.0, seed=0,
    )

    assert policy.error_probability(SHORT, state_is_fresh=True) == pytest.approx(
        policy.error_probability(LONG, state_is_fresh=True)
    )


def test_length_sensitive_policy_errs_more_on_a_longer_transcript():
    policy = PlantedPolicy(
        base_error_rate=0.1, self_conditioning=0.0, staleness=0.0,
        length_sensitivity=0.01, seed=0,
    )

    assert policy.error_probability(LONG, state_is_fresh=True) > policy.error_probability(
        SHORT, state_is_fresh=True
    )


def test_length_term_is_proportional_to_turn_count():
    policy = PlantedPolicy(
        base_error_rate=0.0, self_conditioning=0.0, staleness=0.0,
        length_sensitivity=0.01, seed=0,
    )

    assert policy.error_probability(LONG, state_is_fresh=True) == pytest.approx(0.40)


def test_length_sensitivity_defaults_to_zero_so_existing_plants_are_unchanged():
    policy = PlantedPolicy(
        base_error_rate=0.2, self_conditioning=0.0, staleness=0.0, seed=0
    )

    assert policy.error_probability(LONG, state_is_fresh=True) == pytest.approx(0.2)


def test_probability_remains_capped_at_one_with_the_length_term():
    policy = PlantedPolicy(
        base_error_rate=0.5, self_conditioning=0.0, staleness=0.0,
        length_sensitivity=0.5, seed=0,
    )

    assert policy.error_probability(LONG, state_is_fresh=True) == pytest.approx(1.0)
