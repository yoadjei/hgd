"""Estimator tests — audit sections 25 and 27, prompt sections 8 and 16.

The horizon gap is Peng et al.'s horizon residual with the sign reversed
(Delta = -Gamma_H); we keep the audit's convention and cite theirs. Delta < 0
means excess degradation beyond what independent errors predict.

The floor-effect guard is not defensive programming. Verified Qwen3-8B AppWorld
performance is 5.4% TGC, which would make Delta a log-ratio of two near-zero
probabilities — an estimator that silently returned -inf there would convert an
instrumental artefact into a headline finding. Kill condition F exists for this,
and it has to be enforced in code, not in a document.
"""

import math

import pytest

from hgd.estimators import (
    EstimationError,
    horizon_gap,
    independence_prediction,
    mechanism_share,
    paired_bootstrap_ci,
    risk_difference,
    wilson_interval,
)


# --- independence null -----------------------------------------------------


def test_independence_prediction_is_the_product_of_atomic_rates():
    assert independence_prediction([0.5, 0.5, 0.5]) == pytest.approx(0.125)


def test_independence_prediction_of_a_single_stage_is_that_rate():
    assert independence_prediction([0.7]) == pytest.approx(0.7)


def test_independence_prediction_rejects_an_empty_stage_list():
    with pytest.raises(EstimationError, match="at least one"):
        independence_prediction([])


def test_independence_prediction_rejects_rates_outside_the_unit_interval():
    with pytest.raises(EstimationError, match="probability"):
        independence_prediction([0.5, 1.2])


# --- horizon gap -----------------------------------------------------------


def test_gap_is_zero_when_observation_matches_the_independence_prediction():
    assert horizon_gap(0.125, [0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_gap_is_negative_under_excess_degradation():
    """Observed worse than independent errors predict — the phenomenon we explain."""
    assert horizon_gap(0.05, [0.5, 0.5, 0.5]) < 0


def test_gap_is_positive_when_composition_helps():
    """Not a bug. AppWorld's own TGC/SGC numbers hint this direction is possible."""
    assert horizon_gap(0.30, [0.5, 0.5, 0.5]) > 0


def test_gap_is_the_log_ratio():
    assert horizon_gap(0.0625, [0.5, 0.5, 0.5]) == pytest.approx(math.log(0.5))


def test_zero_observed_success_is_refused_not_returned_as_negative_infinity():
    """Kill condition F: the floor effect must fail loudly, not silently."""
    with pytest.raises(EstimationError, match="floor"):
        horizon_gap(0.0, [0.5, 0.5, 0.5])


def test_zero_independence_prediction_is_refused():
    with pytest.raises(EstimationError, match="floor"):
        horizon_gap(0.1, [0.5, 0.0, 0.5])


def test_gap_warns_below_the_preregistered_estimability_floor():
    """Preregistration fixes P_ind(3) >= 0.05 as the Phase 3a accept condition."""
    with pytest.raises(EstimationError, match="floor"):
        horizon_gap(0.001, [0.1, 0.1, 0.1], min_probability=0.01)


# --- Wilson intervals ------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(successes=7, n=10)

    assert low < 0.7 < high


def test_wilson_interval_stays_inside_the_unit_interval_at_the_boundary():
    """The reason Wilson is specified rather than the normal approximation."""
    low, high = wilson_interval(successes=0, n=10)

    assert low >= 0.0
    assert high > 0.0


def test_wilson_interval_narrows_as_the_sample_grows():
    narrow = wilson_interval(successes=700, n=1000)
    wide = wilson_interval(successes=7, n=10)

    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_rejects_an_empty_sample():
    with pytest.raises(EstimationError, match="n"):
        wilson_interval(successes=0, n=0)


# --- risk difference (propagation) ----------------------------------------


def test_risk_difference_is_the_difference_in_failure_rates():
    estimate = risk_difference(
        treated_failures=9, treated_n=10, control_failures=1, control_n=10
    )

    assert estimate.point == pytest.approx(0.8)


def test_risk_difference_of_identical_arms_is_zero():
    estimate = risk_difference(
        treated_failures=5, treated_n=10, control_failures=5, control_n=10
    )

    assert estimate.point == pytest.approx(0.0)
    assert estimate.low < 0 < estimate.high


def test_propagation_near_one_for_a_destructive_action():
    """Phase 2 plants exactly this: pi near 1 for delete, near 0 for read."""
    estimate = risk_difference(
        treated_failures=20, treated_n=20, control_failures=0, control_n=20
    )

    assert estimate.point == pytest.approx(1.0)
    assert estimate.low > 0.5
    assert estimate.excludes_zero


def test_an_interval_straddling_zero_does_not_exclude_it():
    estimate = risk_difference(
        treated_failures=5, treated_n=10, control_failures=5, control_n=10
    )

    assert not estimate.excludes_zero


def test_wilson_interval_honours_a_non_default_confidence():
    narrow = wilson_interval(successes=7, n=10, confidence=0.80)
    wide = wilson_interval(successes=7, n=10, confidence=0.99)

    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --- mechanism shares ------------------------------------------------------


def test_share_is_the_fraction_of_the_gap_removed_by_the_intervention():
    # gap closes from -1.0 to -0.4, so the intervention removed 60% of it
    assert mechanism_share(delta_before=-1.0, delta_after=-0.4) == pytest.approx(0.6)


def test_share_is_zero_when_the_intervention_changes_nothing():
    assert mechanism_share(delta_before=-1.0, delta_after=-1.0) == pytest.approx(0.0)


def test_share_is_one_when_the_intervention_closes_the_gap_entirely():
    assert mechanism_share(delta_before=-1.0, delta_after=0.0) == pytest.approx(1.0)


def test_share_can_exceed_one_and_is_not_clipped():
    """Overshoot is evidence about the design, not a number to be tidied away."""
    assert mechanism_share(delta_before=-1.0, delta_after=0.5) > 1.0


def test_share_is_undefined_when_there_is_no_gap_to_explain():
    """Dividing by a zero gap would manufacture a share out of nothing."""
    with pytest.raises(EstimationError, match="no gap"):
        mechanism_share(delta_before=0.0, delta_after=-0.2)


# --- paired bootstrap ------------------------------------------------------


def test_bootstrap_interval_contains_the_sample_mean():
    values = [-0.5, -0.4, -0.6, -0.55, -0.45, -0.5, -0.52, -0.48]

    low, high = paired_bootstrap_ci(values, resamples=500, seed=0)

    assert low < sum(values) / len(values) < high


def test_bootstrap_is_deterministic_given_a_seed():
    values = [-0.5, -0.4, -0.6, -0.55]

    first = paired_bootstrap_ci(values, resamples=200, seed=7)
    second = paired_bootstrap_ci(values, resamples=200, seed=7)

    assert first == second


def test_bootstrap_interval_excludes_zero_for_a_clearly_negative_sample():
    values = [-0.5] * 40

    low, high = paired_bootstrap_ci(values, resamples=500, seed=0)

    assert high < 0


def test_bootstrap_interval_includes_zero_for_a_sample_centred_on_zero():
    values = [-0.5, 0.5] * 20

    low, high = paired_bootstrap_ci(values, resamples=500, seed=0)

    assert low < 0 < high


def test_bootstrap_rejects_an_empty_sample():
    with pytest.raises(EstimationError, match="empty"):
        paired_bootstrap_ci([], resamples=100, seed=0)
