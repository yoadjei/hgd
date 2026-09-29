"""Gate tests — audit Phase 1, kill condition C.

The gate decides whether any causal claim is permitted, so what matters most is
that it can fail. Three of these fix the ways it has already reported the wrong
thing on real data: a void run dressed as a pass, a crash taking the whole run
down, and a fidelity count that was not a number.
"""

import contextlib

import pytest

from hgd.gate import (
    DIVERGED,
    ERRORED,
    MATCHED,
    NO_EFFECT,
    SKIPPED,
    GateReport,
    gate_task,
    run_gate,
    solution_actions,
    unique_experiment_names,
)


class FakeWorld:
    """Moves state only when the solution is invoked, not merely defined."""

    def __init__(self, *, moves=True, drift=None, code="def solution(): ..."):
        self.task = type("T", (), {"ground_truth": {"compiled_solution_code": code}})()
        self._ran, self._moves, self._drift = False, moves, drift
        self.executed: list[str] = []

    def execute(self, action):
        self.executed.append(action)
        if action.startswith("solution("):
            self._ran = True
        return "ok"

    def evaluate(self):
        passed = self._ran and self._moves
        return type("E", (), {"to_dict": lambda _s: {
            "passes": ["a", "b"] if passed else ["a"],
            "failures": [] if passed else ["b"],
            "success": passed,
            "num_tests": 2,
            # a drifting world answers differently on each construction
            "drift": next(self._drift) if self._drift else None,
        }})()


def opener(**kwargs):
    @contextlib.contextmanager
    def _open(task_id):
        yield FakeWorld(**kwargs)
    return _open


def digest(world):
    return repr(sorted(world.evaluate().to_dict().items(), key=str))


# --- the action source -----------------------------------------------------


def test_solution_actions_define_then_invoke():
    """Executing the code alone only binds a function; the call is what runs it."""
    actions = solution_actions({"compiled_solution_code": "def solution(): ..."})

    assert actions[0] == "def solution(): ..."
    assert actions[1] == "solution(apis, requester)"


def test_solution_actions_reject_a_task_with_no_released_solution():
    with pytest.raises(ValueError, match="solution"):
        solution_actions({})


def test_both_actions_reach_the_world():
    world_seen = []

    @contextlib.contextmanager
    def _open(task_id):
        world = FakeWorld()
        world_seen.append(world)
        yield world

    gate_task("t1", _open, digest)

    assert world_seen[0].executed == ["def solution(): ...", "solution(apis, requester)"]


# --- verdicts --------------------------------------------------------------


def test_deterministic_environment_matches():
    assert gate_task("t1", opener(), digest) == MATCHED


def test_environment_that_answers_differently_each_time_diverges():
    """A leaky environment must fail the gate rather than pass it quietly."""
    import itertools

    assert gate_task("t1", opener(drift=itertools.count()), digest) == DIVERGED


def test_actions_that_never_move_the_state_are_void_not_matched():
    """Two no-ops agree perfectly. This guard has fired twice on real data."""
    assert gate_task("t1", opener(moves=False), digest) == NO_EFFECT


def test_task_without_a_released_solution_is_skipped():
    assert gate_task("t1", opener(code=""), digest) == SKIPPED


def test_only_two_worlds_are_built_per_task():
    """Four builds per task was the bulk of the runtime on the real benchmark."""
    built = []

    @contextlib.contextmanager
    def _open(task_id):
        built.append(task_id)
        yield FakeWorld()

    gate_task("t1", _open, digest)

    assert len(built) == 2


# --- surviving failure -----------------------------------------------------


def test_one_failing_task_does_not_end_the_run():
    @contextlib.contextmanager
    def _open(task_id):
        if task_id == "bad":
            raise RuntimeError("world would not open")
        yield FakeWorld()

    report = run_gate(["good", "bad", "also_good"], _open, digest)

    assert report.matched == 2
    assert report.by_verdict[ERRORED] == ["bad"]


def test_the_first_failure_keeps_a_traceback():
    """A summary printed only at the end is lost when a notebook truncates."""
    @contextlib.contextmanager
    def _open(task_id):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    report = run_gate(["t1"], _open, digest)

    assert "RuntimeError" in report.first_traceback
    assert report.errors[0][0] == "t1"


def test_progress_is_reported_as_each_task_finishes():
    seen = []

    run_gate(["t1", "t2"], opener(), digest,
             on_result=lambda i, tid, verdict, detail: seen.append((i, tid, verdict)))

    assert seen == [(1, "t1", MATCHED), (2, "t2", MATCHED)]


# --- the verdict itself ----------------------------------------------------


def test_gate_passes_only_on_total_fidelity():
    """Audit Phase 1 says 100%; 19 of 20 is a failure, not a good rate."""
    report = GateReport()
    for index in range(19):
        report.record(f"t{index}", MATCHED)
    report.record("t19", DIVERGED)

    assert not report.passes_gate(expected=20)


def test_a_void_run_does_not_pass_however_consistent():
    report = GateReport()
    for index in range(20):
        report.record(f"t{index}", NO_EFFECT)

    assert report.is_void
    assert not report.passes_gate(expected=20)


def test_a_short_run_does_not_pass():
    """Three matches out of twenty asked for is not total fidelity."""
    report = GateReport()
    for index in range(3):
        report.record(f"t{index}", MATCHED)

    assert not report.passes_gate(expected=20)


def test_a_clean_full_run_passes():
    report = GateReport()
    for index in range(20):
        report.record(f"t{index}", MATCHED)

    assert report.passes_gate(expected=20)
    assert report.rate == 1.0


def test_matched_count_is_a_number_not_a_list():
    """Regression: spreading the id buckets overwrote the count with the ids."""
    report = GateReport()
    report.record("t1", MATCHED)

    payload = report.to_dict(expected=1)

    assert payload["n_matched"] == 1
    assert payload["matched"] == ["t1"]


# --- world naming ----------------------------------------------------------


def test_experiment_names_never_repeat():
    """A shared name lets run two inherit run one's output directory."""
    names = unique_experiment_names("gate")

    assert len({names() for _ in range(50)}) == 50
