"""Intrinsic horizon tests.

H* is HORIZON's definition, verified verbatim: "the minimum number of effective
actions required by an optimal policy to complete the task". Everything
positional depends on it — u = t/H*, the step limit 3H*, and the x-axis of every
hazard curve.

Audit section 32's simulated reviewer R3 objects that "H* from reference
solutions may be non-unique" and demands a sensitivity analysis. So H* is not a
single number here but a named strategy, and the strategy is recorded, so that
the sensitivity analysis is a change of argument rather than a rewrite.
"""

import pytest

from hgd.horizon import HStarStrategy, intrinsic_horizon, normalized_position


def test_api_call_strategy_counts_gold_api_calls():
    ground_truth = {"api_calls": ["spotify.login", "spotify.playlists", "spotify.like"]}

    assert intrinsic_horizon(ground_truth, HStarStrategy.API_CALLS) == 3


def test_solution_line_strategy_uses_solution_code_lines():
    """Available for every split, including test, where api_calls is withheld."""
    ground_truth = {"metadata": {"num_solution_code_lines": 12}}

    assert intrinsic_horizon(ground_truth, HStarStrategy.SOLUTION_LINES) == 12


def test_strategies_can_disagree_which_is_the_point_of_the_sensitivity_analysis():
    ground_truth = {
        "api_calls": ["a", "b", "c"],
        "metadata": {"num_solution_code_lines": 12},
    }

    by_calls = intrinsic_horizon(ground_truth, HStarStrategy.API_CALLS)
    by_lines = intrinsic_horizon(ground_truth, HStarStrategy.SOLUTION_LINES)

    assert by_calls != by_lines


def test_missing_field_for_the_requested_strategy_is_an_explicit_error():
    """Silently defaulting H* would corrupt u for a whole split."""
    with pytest.raises(ValueError, match="api_calls"):
        intrinsic_horizon({"metadata": {}}, HStarStrategy.API_CALLS)


def test_zero_horizon_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        intrinsic_horizon({"api_calls": []}, HStarStrategy.API_CALLS)


def test_duplicate_api_calls_are_counted_not_deduplicated():
    """An optimal policy that calls the same API twice takes two actions."""
    ground_truth = {"api_calls": ["spotify.login", "spotify.like", "spotify.like"]}

    assert intrinsic_horizon(ground_truth, HStarStrategy.API_CALLS) == 3


# --- normalized position ---------------------------------------------------


def test_normalized_position_is_step_over_h_star():
    assert normalized_position(3, 12) == 0.25


def test_position_at_the_intrinsic_horizon_is_one():
    assert normalized_position(12, 12) == 1.0


def test_position_beyond_the_intrinsic_horizon_exceeds_one():
    """A trajectory may run to 3H*; u > 1 is meaningful and must not be clipped."""
    assert normalized_position(24, 12) == 2.0


def test_normalized_position_rejects_non_positive_horizon():
    with pytest.raises(ValueError, match="positive"):
        normalized_position(1, 0)
