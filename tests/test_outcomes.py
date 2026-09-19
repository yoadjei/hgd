"""Outcome extraction from an AppWorld evaluation payload.

This is the bridge between the environment's evaluation and our checkpoint,
survival and horizon-gap machinery. It deliberately exposes three views of the
same payload, because they answer different questions and the choice between
them is a research decision, not an implementation detail:

* ``task_success`` — binary Task Goal Completion. The audit's P_obs.
* ``pass_fraction`` — graded, tests passed over tests total.
* ``checkpoint_vector`` — per-test outcomes, which feed `recoverability` and
  `first_unrecovered_error`; the survival layer needs this regardless.

Why the graded view exists at all: blocker B2. Small open-weight models score in
the low single digits of TGC on AppWorld, and Delta(s) = log P_obs - sum log p_hat
is undefined when both terms approach zero. Having the graded outcome available
does **not** mean the estimand changes — that would be a scope change under
prompt section 5, and it is to be decided by the Phase 3a pilot measurement, not
assumed here.
"""

import pytest

from hgd.checkpoints import first_unrecovered_error
from hgd.outcomes import checkpoint_vector, pass_fraction, task_success

PASSING = {"success": True, "passes": ["a", "b", "c"], "failures": [],
           "num_tests": 3, "difficulty": 1}
PARTIAL = {"success": False, "passes": ["a"], "failures": ["b", "c"],
           "num_tests": 3, "difficulty": 1}
FAILING = {"success": False, "passes": [], "failures": ["a", "b", "c"],
           "num_tests": 3, "difficulty": 1}


# --- binary outcome --------------------------------------------------------


def test_task_success_reads_the_success_flag():
    assert task_success(PASSING) is True
    assert task_success(PARTIAL) is False


def test_task_success_falls_back_to_an_empty_failure_list():
    """Some payloads may omit `success`; an empty failure vector still means passed."""
    assert task_success({"passes": ["a"], "failures": [], "num_tests": 1}) is True


def test_task_success_of_an_empty_payload_is_false():
    """No evidence of success is not success."""
    assert task_success({}) is False


# --- graded outcome --------------------------------------------------------


def test_pass_fraction_is_passes_over_total():
    assert pass_fraction(PARTIAL) == pytest.approx(1 / 3)


def test_pass_fraction_of_a_full_pass_is_one():
    assert pass_fraction(PASSING) == pytest.approx(1.0)


def test_pass_fraction_of_a_full_failure_is_zero():
    assert pass_fraction(FAILING) == pytest.approx(0.0)


def test_pass_fraction_is_monotone_in_passes():
    assert pass_fraction(FAILING) < pass_fraction(PARTIAL) < pass_fraction(PASSING)


def test_pass_fraction_derives_the_total_when_num_tests_is_absent():
    payload = {"passes": ["a"], "failures": ["b"]}

    assert pass_fraction(payload) == pytest.approx(0.5)


def test_pass_fraction_of_a_task_with_no_tests_is_undefined():
    """Zero tests is a broken task, not a zero score; it must not silently be 0.0."""
    with pytest.raises(ValueError, match="no tests"):
        pass_fraction({"passes": [], "failures": [], "num_tests": 0})


# --- per-test vector -------------------------------------------------------


def test_checkpoint_vector_maps_each_test_to_its_outcome():
    vector = checkpoint_vector(PARTIAL)

    assert vector == {"a": True, "b": False, "c": False}


def test_checkpoint_vector_is_stable_across_calls():
    assert checkpoint_vector(PARTIAL) == checkpoint_vector(PARTIAL)


def test_checkpoint_vector_of_a_full_pass_is_all_true():
    assert all(checkpoint_vector(PASSING).values())


def test_checkpoint_vector_feeds_the_survival_event():
    """The reason this exists: per-test series drive first_unrecovered_error."""
    per_step = [checkpoint_vector(FAILING), checkpoint_vector(PARTIAL),
                checkpoint_vector(PARTIAL)]
    series = {name: [step[name] for step in per_step] for name in per_step[0]}

    # a fails at step 0 then recovers; b and c never recover
    assert first_unrecovered_error(series, run_succeeded=False) == 0


def test_checkpoint_vector_ignores_non_test_keys():
    vector = checkpoint_vector(PASSING)

    assert "difficulty" not in vector
    assert "num_tests" not in vector
