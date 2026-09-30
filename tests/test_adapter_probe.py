"""The adapter probe has to work before it is handed to a Kaggle session.

Lesson 5: anything that runs elsewhere is checked here first. The probe's job is
to answer questions about the real package, so this cannot verify its answers —
only that it runs, exercises every scenario, reports honestly, and survives the
failures the real package has already shown it.

Three of those are now regressions rather than hypotheses. The first real run
reported ``task_completed() flips: True -> True`` as a PASS, because the gold
solution had already completed the task and the check never looked at its
baseline. It then died in teardown inside appworld's time freezer and lost all
twelve findings, because it wrote its record after the close.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from freezegun import freeze_time

PROBE_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "adapter_probe.py"

GOLD = "def solution(apis, requester):\n    apis.spotify.like_song(song_id=1)\n"
FREEZER_ERROR = "'_freeze_time' object has no attribute 'fake_names'"


class ProbeGroundTruth:
    """Ground truth as the real package exposes it: an object, not a mapping."""

    def __init__(self):
        self.compiled_solution_code = GOLD
        self.api_calls = [{"method": "get", "url": "/x", "data": {}}] * 4


TASK_DATETIME = datetime.datetime(2023, 5, 12, 9, 0, 0)


class ProbeTask:
    def __init__(self):
        self.instruction = "Like the song."
        self.ground_truth = ProbeGroundTruth()
        # appworld freezes the shell clock to this, and determinism rests on it
        self.datetime = TASK_DATETIME


class ProbeEvaluation:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class ProbeWorld:
    """A world that satisfies every assumption the probe checks.

    Entries come back as dicts once the task completes and strings before, which
    is what the real package does and what every earlier fake got wrong. The gold
    solution completes the task by itself, which is also real, and is why the
    completion check needs a world of its own.
    """

    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.task = ProbeTask()
        self.closed = False
        self._completed = False
        self._liked = False
        self._states: dict[str, tuple] = {}
        # a real freeze, because appworld starts one per world and the probe now
        # checks the freeze stack. a fake that skipped this could not exercise that
        # check, which is the same hole that let three earlier defects ship.
        self._freezer = freeze_time(TASK_DATETIME)
        self._freezer.start()

    def execute(self, code: str) -> str:
        if "like_song" in code or "solution(apis" in code:
            self._liked = True
            self._completed = True
        if "complete_task" in code:
            self._completed = True
        if "_dt.datetime.now()" in code:
            # the shell clock, frozen to the task datetime as appworld freezes it
            return TASK_DATETIME.isoformat() + "\n"
        return f"output of {code[:20]}"

    def task_completed(self) -> bool:
        return self._completed

    def evaluate(self):
        names = ["login_ok", "song_liked"] if self._liked else ["login_ok"]
        fails = [] if self._liked else ["song_liked"]
        as_dicts = [{"name": x, "score": 1} for x in names]
        return ProbeEvaluation({
            "success": self._liked,
            "passes": as_dicts if self._completed else list(names),
            "failures": ([{"name": x, "score": 0} for x in fails]
                         if self._completed else list(fails)),
            "num_tests": 2,
            "difficulty": 1,
        })

    def save_state(self) -> str:
        marker = str(len(self._states))
        self._states[marker] = (self._liked, self._completed)
        return marker

    def load_state(self, marker: str) -> None:
        self._liked, self._completed = self._states[marker]

    def close(self) -> None:
        self.closed = True
        self._freezer.stop()


def freeze_depth() -> int:
    from freezegun import api

    return len(api.freeze_factories)


@pytest.fixture(autouse=True)
def no_leaked_freezes():
    """A test that leaks a freeze leaves datetime patched for every later test.

    This is the hygiene guard for the fakes above: without it a broken-close double
    could poison the whole session and the failure would surface somewhere else
    entirely.
    """
    before = freeze_depth()
    yield
    assert freeze_depth() == before, "this test leaked a freezegun freeze"


def load_probe(monkeypatch, world_class):
    """Import the probe with a stubbed ``appworld`` package."""
    stub = types.ModuleType("appworld")
    stub.AppWorld = lambda task_id, **kwargs: world_class(task_id, **kwargs)
    stub.load_task_ids = lambda split: ["probe_task_1", "probe_task_2"]
    monkeypatch.setitem(sys.modules, "appworld", stub)

    spec = importlib.util.spec_from_file_location("adapter_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.results.clear()
    return module


def report(tmp_path):
    return json.loads((tmp_path / "adapter_probe.json").read_text())


def failed_checks(payload):
    return [row["check"] for row in payload["checks"] if not row["ok"]]


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_probe_reports_every_assumption_holding(in_tmp, monkeypatch):
    probe = load_probe(monkeypatch, ProbeWorld)

    exit_code = probe.main("train")
    payload = report(in_tmp)

    assert exit_code == 0
    assert payload["n_failed"] == 0
    assert payload["task_id"] == "probe_task_1"
    names = [row["check"] for row in payload["checks"]]
    for scenario, _ in probe.SCENARIOS:
        assert f"{scenario}: close() after this sequence" in names


def test_every_scenario_gets_its_own_world(in_tmp, monkeypatch):
    """One world per scenario is what makes a close() failure attributable."""
    built = []

    class Counted(ProbeWorld):
        def __init__(self, task_id, **kwargs):
            super().__init__(task_id, **kwargs)
            built.append(self)

    probe = load_probe(monkeypatch, Counted)

    probe.main("train")

    assert len(built) == len(probe.SCENARIOS)
    assert all(world.closed for world in built)


def test_a_close_that_raises_is_attributed_to_its_sequence(in_tmp, monkeypatch):
    """The real failure: close() raised inside appworld's time freezer after
    save_state/load_state. It has to be pinned to that sequence, and every
    scenario around it has to keep reporting."""

    class BreaksCloseAfterSave(ProbeWorld):
        _broken = False

        def save_state(self):
            self._broken = True
            return super().save_state()

        def close(self):
            # release the freeze first: the real package raises from inside its own
            # teardown, but a double that leaks one corrupts every later test
            super().close()
            if self._broken:
                raise AttributeError(FREEZER_ERROR)

    probe = load_probe(monkeypatch, BreaksCloseAfterSave)

    exit_code = probe.main("train")
    payload = report(in_tmp)

    assert exit_code == 1
    assert failed_checks(payload) == ["markers: close() after this sequence"]
    broken = next(row for row in payload["checks"]
                  if row["check"] == "markers: close() after this sequence")
    assert "fake_names" in broken["observed"]


def test_the_probe_catches_a_load_state_that_stops_refusing(in_tmp, monkeypatch):
    """The fix that must not silently regress. appworld's load_state() unfreezes the
    world clock, so if the adapter ever delegates to it again the probe has to say
    so rather than let a non-deterministic environment through."""
    probe = load_probe(monkeypatch, ProbeWorld)
    monkeypatch.setattr(probe.AppWorldEnvironment, "load_state",
                        lambda self, state_id: self.world.load_state(state_id))

    probe.main("train")

    assert "markers: load_state() is refused rather than corrupting the clock" in \
        failed_checks(report(in_tmp))


def test_the_probe_catches_a_shell_clock_that_is_not_frozen(in_tmp, monkeypatch):
    """Determinism rests on the clock being frozen to the task datetime. A shell
    reading wall-clock time makes timestamps irreproducible, which no digest that
    ignores them would ever reveal."""

    class WallClockShell(ProbeWorld):
        def execute(self, code):
            if "_dt.datetime.now()" in code:
                return "2026-09-30T20:22:24\n"
            return super().execute(code)

    probe = load_probe(monkeypatch, WallClockShell)

    probe.main("train")

    assert "clock: the shell clock is frozen to the task datetime" in \
        failed_checks(report(in_tmp))


def test_the_record_is_written_even_when_every_close_raises(in_tmp, monkeypatch):
    """The regression that lost twelve findings: the report came after the close."""

    class AlwaysBreaksClose(ProbeWorld):
        def close(self):
            super().close()
            raise AttributeError(FREEZER_ERROR)

    probe = load_probe(monkeypatch, AlwaysBreaksClose)

    probe.main("train")
    payload = report(in_tmp)

    assert payload["n_checks"] > len(probe.SCENARIOS)
    assert len(failed_checks(payload)) == len(probe.SCENARIOS)


def test_an_already_complete_task_fails_the_flip_check(in_tmp, monkeypatch):
    """The vacuous PASS: the first real run reported True -> True and proved
    nothing. A world already complete before complete_task() runs must fail this
    check. Lesson 3 — a comparison has to confirm the outcome moved.
    """

    class BornComplete(ProbeWorld):
        def __init__(self, task_id, **kwargs):
            super().__init__(task_id, **kwargs)
            self._completed = True

    probe = load_probe(monkeypatch, BornComplete)

    probe.main("train")
    payload = report(in_tmp)

    flip = next(row for row in payload["checks"]
                if row["check"].startswith("flip: task_completed()"))
    assert flip["ok"] is False
    assert "prove nothing" in flip["observed"]


def test_a_second_complete_task_that_changes_the_score_is_reported(in_tmp, monkeypatch):
    """The real run went from pass_fraction 1.0 to 0.5 with a second
    complete_task() in between. If that is the cause, P_obs stops being a property
    of the trajectory and the probe has to say so."""

    class DegradesOnSecondComplete(ProbeWorld):
        def execute(self, code):
            if "complete_task" in code and self._completed:
                self._liked = False
            return super().execute(code)

    probe = load_probe(monkeypatch, DegradesOnSecondComplete)

    probe.main("train")

    assert "double: a second complete_task() leaves the evaluation alone" in \
        failed_checks(report(in_tmp))


def test_probe_catches_a_digest_that_does_not_move(in_tmp, monkeypatch):
    """A digest blind to the gold solution is the no-op failure that made the
    first gate report total fidelity while executing nothing."""

    class FrozenState(ProbeWorld):
        def execute(self, code):
            return f"output of {code[:20]}"

    probe = load_probe(monkeypatch, FrozenState)

    probe.main("train")

    assert "gold: state_hash() moves when the gold solution runs" in \
        failed_checks(report(in_tmp))


def test_probe_flags_a_snapshot_that_leaks_raw_entries(in_tmp, monkeypatch):
    """The defect this whole exercise came from: a predicate written
    `name in state["failures"]` is silently False on raw dict entries."""
    probe = load_probe(monkeypatch, ProbeWorld)
    monkeypatch.setattr(probe.AppWorldEnvironment, "snapshot",
                        lambda self: dict(self.world.evaluate().to_dict()))

    probe.main("train")

    assert "gold: snapshot() normalises passes and failures to names" in \
        failed_checks(report(in_tmp))


def test_the_depth_check_accepts_whatever_baseline_the_world_establishes(in_tmp, monkeypatch):
    """The false positive it shipped with: it asserted depth == 1 and failed on real
    hardware at depth 2, because a live world holds its own freeze plus the one its
    Requester starts. The invariant is no drift, not a number I guessed."""

    class TwoFreezes(ProbeWorld):
        def __init__(self, task_id, **kwargs):
            super().__init__(task_id, **kwargs)
            self._extra = freeze_time(TASK_DATETIME)
            self._extra.start()

        def close(self):
            self._extra.stop()
            super().close()

    probe = load_probe(monkeypatch, TwoFreezes)

    probe.main("train")

    assert "gold: evaluating does not leak a time freeze" not in \
        failed_checks(report(in_tmp))


def test_the_depth_check_still_catches_a_freeze_leaked_while_evaluating(in_tmp, monkeypatch):
    """It has to keep working, or removing the magic number removed the guard."""

    class LeaksOnEvaluate(ProbeWorld):
        _leaked: list = []

        def evaluate(self):
            if not self._leaked:
                extra = freeze_time(TASK_DATETIME)
                extra.start()
                self._leaked.append(extra)
            return super().evaluate()

        def close(self):
            while self._leaked:
                self._leaked.pop().stop()
            super().close()

    probe = load_probe(monkeypatch, LeaksOnEvaluate)

    probe.main("train")

    assert "gold: evaluating does not leak a time freeze" in \
        failed_checks(report(in_tmp))
