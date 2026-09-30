"""Adapter hardening, driven by what the real package actually did.

Two defects the Phase 1 probe exposed:

1. `evaluate()` prints a full formatted test report on every call. The adapter
   calls it for `state_hash` and `snapshot`, which happens at least once per
   step, so a single episode would bury its own log under thousands of lines of
   report — and on Kaggle that is also a real slowdown.

2. The oracle-fix intervention needs the *correct* action for a step, which
   comes from the released gold solution. `ground_truth.compiled_solution_code`
   holds it, and `api_calls` does not — the latter is a list of HTTP record
   dicts that `execute()` treats as no-op literals, which is exactly how the
   first gate run measured nothing while reporting success.
"""

import os
import sys

import pytest

from hgd.appworld_env import AppWorldEnvironment, evaluation_digest


class NoisyEvaluation:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        # the real package prints its report during evaluation
        print("=" * 60)
        print("Overall Stats: Num Passed Tests : 1  Num Failed Tests : 1")
        print("=" * 60)
        return dict(self._payload)


class NoisyWorld:
    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.task = type("T", (), {
            "instruction": "do the thing",
            "ground_truth": type("GT", (), {
                "api_calls": [{"method": "get", "url": "/x", "data": {}}] * 71,
                "compiled_solution_code": "apis.spotify.login()\napis.spotify.like()",
                "num_solution_code_lines": 18,
            })(),
        })()
        self.closed = False
        self._done = False

    def execute(self, code):
        if "complete_task" in code:
            self._done = True
        return "ok"

    def task_completed(self):
        return self._done

    def evaluate(self):
        return NoisyEvaluation({"passes": ["a"], "failures": ["b"],
                                "success": False, "num_tests": 2})

    def save_state(self):
        return "s0"

    def load_state(self, sid):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def env():
    return AppWorldEnvironment(
        world_factory=lambda task_id, **kw: NoisyWorld(task_id, **kw),
        state_digest=evaluation_digest,
        experiment_name="test",
    )


# --- output suppression ----------------------------------------------------


def test_state_hash_does_not_print_the_evaluation_report(env, capsys):
    env.reset("train/t1", seed=0)
    capsys.readouterr()

    env.state_hash()

    assert capsys.readouterr().out == ""


def test_snapshot_does_not_print_the_evaluation_report(env, capsys):
    env.reset("train/t1", seed=0)
    capsys.readouterr()

    env.snapshot()

    assert capsys.readouterr().out == ""


def test_suppression_does_not_change_the_digest(env):
    """Silencing the report must not silence the data."""
    env.reset("train/t1", seed=0)

    assert env.state_hash() == env.state_hash()
    assert env.snapshot()["failures"] == ["b"]


class FileDescriptorEvaluation:
    """Writes past ``sys.stdout``, as the real package's console does.

    AppWorld reports through a console holding the stream it was constructed
    with, so redirecting ``sys.stdout`` afterwards does not reach it. Writing
    straight to the descriptor reproduces that without depending on rich.
    """

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        os.write(1, b"=" * 60 + b"\nOverall Stats: Num Passed Tests : 1\n")
        return dict(self._payload)


def test_silenced_leaves_stderr_with_a_real_descriptor():
    """Regression: redirecting to io.StringIO broke AppWorld's safety guard.

    `execute()` calls `safety_guard.disable()`, which calls
    `faulthandler.enable()`, and that needs a genuine descriptor. A StringIO has
    none, so every action raised `io.UnsupportedOperation: fileno` and the gate
    errored on every task while reporting a misleading PermissionError instead.
    """
    import faulthandler

    from hgd.appworld_env import silenced

    was_enabled = faulthandler.is_enabled()
    try:
        with silenced():
            sys.stderr.fileno()
            faulthandler.enable()
    finally:
        if not was_enabled:
            faulthandler.disable()


def test_silenced_opens_no_file_while_active(monkeypatch):
    """Regression: a TemporaryFile inside the silencer hit AppWorld's guard.

    While an action executes, AppWorld replaces `open` with a read-only version
    that raises on any write. The silencer must not need to open anything, or it
    masks the real failure with a PermissionError of its own.
    """
    import builtins
    import io as io_module

    from hgd.appworld_env import silenced

    def forbidden(*args, **kwargs):
        raise PermissionError("Writing to OS file system is disabled.")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(io_module, "open", forbidden)

    with silenced():
        pass


def test_report_written_past_sys_stdout_is_still_suppressed(capfd):
    """Regression: two nested redirect_stdout layers did not stop the real report."""
    class World(NoisyWorld):
        def evaluate(self):
            return FileDescriptorEvaluation(
                {"passes": ["a"], "failures": ["b"], "success": False, "num_tests": 2}
            )

    env = AppWorldEnvironment(
        world_factory=lambda task_id, **kw: World(task_id, **kw),
        state_digest=evaluation_digest,
        experiment_name="test",
    )
    env.reset("train/t1", seed=0)
    capfd.readouterr()

    digest = env.state_hash()

    assert capfd.readouterr().out == ""
    assert digest


# --- gold solution ---------------------------------------------------------


def test_gold_solution_code_comes_from_compiled_solution_code(env):
    env.reset("train/t1", seed=0)

    assert env.gold_solution_code() == "apis.spotify.login()\napis.spotify.like()"


def test_gold_solution_is_not_taken_from_api_calls(env):
    """api_calls are HTTP record dicts; execute() treats them as no-op literals."""
    env.reset("train/t1", seed=0)

    assert "method" not in env.gold_solution_code()


def test_missing_gold_solution_is_an_explicit_error():
    class Bare:
        def __init__(self, task_id, **kw):
            self.task = type("T", (), {"ground_truth": type("GT", (), {})()})()

        def close(self):
            pass

    env = AppWorldEnvironment(
        world_factory=lambda task_id, **kw: Bare(task_id, **kw),
        state_digest=lambda w: "x",
        experiment_name="test",
    )
    env.reset("train/t1", seed=0)

    with pytest.raises(ValueError, match="solution"):
        env.gold_solution_code()


# --- horizon ---------------------------------------------------------------


def test_intrinsic_horizon_reads_the_object_ground_truth(env):
    env.reset("train/t1", seed=0)

    assert env.intrinsic_horizon() == 71
