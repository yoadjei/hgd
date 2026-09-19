"""Checkpoint and recoverability tests — audit task T1.4, sections 12 and 28.

Two jobs. Checkpoints are the tier-2 label source: programmatic state assertions
that give a recoverability outcome without a judge, which is what lets the causal
analysis survive a failed judge (kill condition D).

The recoverability outcome is not cosmetic. It defines the competing risks in
the hazard model (recovered vs terminal) and it defines the survival event
itself — the *first unrecovered* critical error. Getting "recovered" wrong would
silently move mass between the two competing risks.
"""

import pytest

from hgd.checkpoints import (
    PERSISTED,
    RECOVERED,
    TERMINAL,
    Checkpoint,
    evaluate_checkpoints,
    first_unrecovered_error,
    recoverability,
)
from hgd.env import DictEnvironment


@pytest.fixture
def checkpoints():
    return (
        Checkpoint("a_is_1", stage=0, predicate=lambda s: s.get("a") == "1"),
        Checkpoint("b_is_2", stage=1, predicate=lambda s: s.get("b") == "2"),
    )


# --- evaluation ------------------------------------------------------------


def test_checkpoint_passes_when_state_satisfies_the_predicate(checkpoints):
    env = DictEnvironment()
    env.reset("t", 0)
    env.execute("store(a, 1)")

    results = evaluate_checkpoints(env, checkpoints)

    assert results[0].passed
    assert not results[1].passed


def test_results_carry_identity_and_stage(checkpoints):
    env = DictEnvironment()
    env.reset("t", 0)

    results = evaluate_checkpoints(env, checkpoints)

    assert [r.checkpoint_id for r in results] == ["a_is_1", "b_is_2"]
    assert [r.stage for r in results] == [0, 1]


def test_a_predicate_that_raises_is_recorded_as_failed_not_propagated(checkpoints):
    """A broken assertion must not take down a 12-hour run."""
    def explode(state):
        raise KeyError("boom")

    env = DictEnvironment()
    env.reset("t", 0)

    results = evaluate_checkpoints(env, (Checkpoint("bad", stage=0, predicate=explode),))

    assert not results[0].passed
    assert results[0].error is not None


def test_evaluation_is_serialisable_for_the_log(checkpoints):
    env = DictEnvironment()
    env.reset("t", 0)

    results = evaluate_checkpoints(env, checkpoints)

    assert [r.to_dict() for r in results][0]["checkpoint_id"] == "a_is_1"


# --- recoverability outcome ------------------------------------------------


def test_failure_followed_by_a_pass_is_recovered():
    assert recoverability([True, False, True], run_succeeded=True) == RECOVERED


def test_failure_that_never_passes_in_a_successful_run_is_persisted():
    """The checkpoint stayed broken but the task still completed."""
    assert recoverability([True, False, False], run_succeeded=True) == PERSISTED


def test_failure_that_never_passes_in_a_failed_run_is_terminal():
    """The pass before the failure does not count as recovery from it."""
    assert recoverability([True, False, False], run_succeeded=False) == TERMINAL


def test_a_checkpoint_that_never_failed_has_no_outcome():
    assert recoverability([True, True, True], run_succeeded=True) is None


def test_late_recovery_still_counts():
    assert recoverability([False, False, False, True], run_succeeded=True) == RECOVERED


def test_empty_series_has_no_outcome():
    assert recoverability([], run_succeeded=True) is None


# --- the survival event ----------------------------------------------------


def test_first_unrecovered_error_returns_the_step_of_the_unrecovered_failure():
    # c0 fails at step 1 and recovers; c1 fails at step 2 and does not
    series = {"c0": [True, False, True, True], "c1": [True, True, False, False]}

    assert first_unrecovered_error(series, run_succeeded=False) == 2


def test_a_recovered_failure_is_not_the_survival_event():
    series = {"c0": [True, False, True, True]}

    assert first_unrecovered_error(series, run_succeeded=True) is None


def test_earliest_unrecovered_failure_wins_when_several_exist():
    series = {"c0": [True, True, False, False], "c1": [True, False, False, False]}

    assert first_unrecovered_error(series, run_succeeded=False) == 1


def test_a_clean_run_is_censored():
    """No event: the trajectory survived, and the hazard model must treat it as censored."""
    series = {"c0": [True, True], "c1": [True, True]}

    assert first_unrecovered_error(series, run_succeeded=True) is None


def test_persisted_failure_in_a_successful_run_still_counts_as_unrecovered():
    """The event is the first *unrecovered* error, not the first fatal one."""
    series = {"c0": [True, False, False]}

    assert first_unrecovered_error(series, run_succeeded=True) == 1
