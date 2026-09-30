"""The adapter probe has to work before it is handed to a Kaggle session.

Lesson 5: anything that runs elsewhere is checked here first. The probe's job is
to answer questions about the real package, so this cannot verify its answers —
only that the probe runs, exercises every check, and reports honestly. A fake that
satisfies every assumption must make it report all-pass, and a fake that breaks
one must make it report that one and keep going.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROBE_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "adapter_probe.py"

GOLD = "def solution(apis, requester):\n    apis.spotify.like_song(song_id=1)\n"


class ProbeGroundTruth:
    """Ground truth as the real package exposes it: an object, not a mapping."""

    def __init__(self):
        self.compiled_solution_code = GOLD
        self.api_calls = [{"method": "get", "url": "/x", "data": {}}] * 4


class ProbeTask:
    def __init__(self):
        self.instruction = "Like the song."
        self.ground_truth = ProbeGroundTruth()


class ProbeEvaluation:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class ProbeWorld:
    """A world that satisfies every assumption the probe checks.

    Entries are returned as dicts once the task completes and strings before,
    which is what the real package does and what every earlier fake got wrong.
    """

    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.task = ProbeTask()
        self.closed = False
        self._completed = False
        self._liked = False
        self._states: dict[str, tuple] = {}

    def execute(self, code: str) -> str:
        if "like_song" in code:
            self._liked = True
        if "complete_task" in code:
            self._completed = True
        return f"output of {code[:20]}"

    def task_completed(self) -> bool:
        return self._completed

    def evaluate(self):
        names = ["login_ok", "song_liked"] if self._liked else ["login_ok"]
        fails = [] if self._liked else ["song_liked"]
        shape = (lambda n: [{"name": x, "score": 1} for x in n]) if self._completed \
            else list
        return ProbeEvaluation({
            "success": self._liked,
            "passes": shape(names),
            "failures": shape(fails),
            "num_tests": 2,
            "difficulty": 1,
        })

    def save_state(self) -> str:
        marker = f"s{len(self._states)}"
        self._states[marker] = (self._liked, self._completed)
        return marker

    def load_state(self, marker: str) -> None:
        self._liked, self._completed = self._states[marker]

    def close(self) -> None:
        self.closed = True


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


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_probe_reports_every_assumption_holding(in_tmp, monkeypatch):
    probe = load_probe(monkeypatch, ProbeWorld)

    exit_code = probe.main("train")

    payload = json.loads((in_tmp / "adapter_probe.json").read_text())
    assert exit_code == 0
    assert payload["n_failed"] == 0
    assert payload["n_checks"] >= 11
    assert payload["task_id"] == "probe_task_1"


def test_probe_records_a_broken_assumption_and_keeps_going(in_tmp, monkeypatch):
    """The failure mode that matters: one bad answer must not hide the rest."""

    class NeverCompletes(ProbeWorld):
        def task_completed(self) -> bool:
            return False

    probe = load_probe(monkeypatch, NeverCompletes)

    exit_code = probe.main("train")

    payload = json.loads((in_tmp / "adapter_probe.json").read_text())
    failed = [row["check"] for row in payload["checks"] if not row["ok"]]
    assert exit_code == 1
    assert failed == ["task_completed() flips after complete_task()"]
    # the checks after the broken one still ran
    assert payload["checks"][-1]["check"].startswith("the outcome views still")


def test_probe_catches_a_digest_that_does_not_move(in_tmp, monkeypatch):
    """A digest blind to the gold solution is the no-op failure that made the
    first gate report total fidelity while executing nothing."""

    class FrozenState(ProbeWorld):
        def execute(self, code: str) -> str:
            return f"output of {code[:20]}"

    probe = load_probe(monkeypatch, FrozenState)

    probe.main("train")

    payload = json.loads((in_tmp / "adapter_probe.json").read_text())
    failed = [row["check"] for row in payload["checks"] if not row["ok"]]
    assert "state_hash() moves when the gold solution runs" in failed


def test_probe_flags_a_snapshot_that_leaks_raw_entries(in_tmp, monkeypatch):
    """The defect this whole exercise came from: a predicate written
    `name in state["failures"]` is silently False on raw dict entries."""

    class RawEntries(ProbeWorld):
        def evaluate(self):
            return ProbeEvaluation({
                "success": False,
                "passes": [{"name": "login_ok", "score": 1}],
                "failures": [{"name": "song_liked", "score": 0}],
                "num_tests": 2,
                "difficulty": 1,
            })

    probe = load_probe(monkeypatch, RawEntries)
    monkeypatch.setattr(probe.AppWorldEnvironment, "snapshot",
                        lambda self: dict(self.world.evaluate().to_dict()))

    probe.main("train")

    payload = json.loads((in_tmp / "adapter_probe.json").read_text())
    failed = [row["check"] for row in payload["checks"] if not row["ok"]]
    assert "snapshot() normalises passes and failures to names" in failed
