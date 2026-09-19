"""H* must read a GroundTruth object, not only a mapping.

The Phase 1 probe against the installed package returned
``ground_truth_type: GroundTruth`` — an object exposing ``api_calls``,
``num_solution_code_lines``, ``compiled_solution_code`` and friends as
attributes. `intrinsic_horizon` was written from documentation and assumed
mapping access, so on real data it would raise "ground truth has no 'api_calls'"
for every task and H* would be unavailable for the whole split.

Mapping access is kept because the synthetic environments and the tests use
plain dicts.
"""

import pytest

from hgd.horizon import HStarStrategy, ground_truth_field, intrinsic_horizon


class GroundTruthObject:
    """Shape verified against appworld 0.1.3.post1 on 2026-09-06."""

    def __init__(self, api_calls=None, num_solution_code_lines=None):
        if api_calls is not None:
            self.api_calls = api_calls
        if num_solution_code_lines is not None:
            self.num_solution_code_lines = num_solution_code_lines


def test_api_calls_read_from_an_object_attribute():
    ground_truth = GroundTruthObject(api_calls=[{"method": "get"}] * 71)

    assert intrinsic_horizon(ground_truth, HStarStrategy.API_CALLS) == 71


def test_solution_lines_read_from_an_object_attribute():
    """On the real object this is a top-level attribute, not nested in metadata."""
    ground_truth = GroundTruthObject(num_solution_code_lines=18)

    assert intrinsic_horizon(ground_truth, HStarStrategy.SOLUTION_LINES) == 18


def test_mapping_access_still_works():
    assert intrinsic_horizon({"api_calls": ["a", "b"]}, HStarStrategy.API_CALLS) == 2


def test_nested_metadata_mapping_still_works():
    ground_truth = {"metadata": {"num_solution_code_lines": 12}}

    assert intrinsic_horizon(ground_truth, HStarStrategy.SOLUTION_LINES) == 12


def test_object_missing_the_requested_field_still_raises_clearly():
    with pytest.raises(ValueError, match="api_calls"):
        intrinsic_horizon(GroundTruthObject(num_solution_code_lines=5),
                          HStarStrategy.API_CALLS)


def test_object_with_an_empty_api_call_list_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        intrinsic_horizon(GroundTruthObject(api_calls=[]), HStarStrategy.API_CALLS)


# --- the shared accessor ---------------------------------------------------


def test_ground_truth_field_reads_attributes_and_keys_alike():
    assert ground_truth_field(GroundTruthObject(api_calls=["a"]), "api_calls") == ["a"]
    assert ground_truth_field({"api_calls": ["a"]}, "api_calls") == ["a"]


def test_ground_truth_field_is_none_when_absent_from_either_shape():
    assert ground_truth_field(GroundTruthObject(), "api_calls") is None
    assert ground_truth_field({}, "api_calls") is None
