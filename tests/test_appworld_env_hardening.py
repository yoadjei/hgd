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

import datetime
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


# --- appworld's time freezer -----------------------------------------------
# appworld 0.1.3.post1 keeps three independent freezer registries and stops them
# without a shared ordering discipline. reproduced against real freezegun:
#
#   freeze_time.stop() (common/utils.py) guards on `self._freezer is not None`
#   but never nulls it. AppWorld.close_all() (environment.py:789) calls
#   time_freezer.stop() directly rather than unset_local_date_and_time(), so the
#   freezer stays non-None. load_state() calls close_all() and restarts nothing,
#   and initialize() calls close_all() too.
#
# two consequences, and the silent one is worse:
#   - after load_state() the world's time is no longer frozen, so every timestamp
#     written afterwards is wall-clock instead of the task's datetime;
#   - the next close() stops the same freezegun instance twice, popping its global
#     stack again, which raises IndexError or, when a nested freezer is left to
#     empty the stack, AttributeError: no attribute 'fake_names'.


class FreezerBreakingWorld:
    """A world whose close() fails the way the real package's does.

    Verified on kaggle: AttributeError out of appworld's own teardown, with the
    world already unusable by then.
    """

    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.task = None
        self.closed = False

    def close(self):
        self.closed = True
        raise AttributeError("'_freeze_time' object has no attribute 'fake_names'")


def test_load_state_is_refused_rather_than_silently_unfreezing_time():
    """appworld's load_state() leaves the world running on wall-clock time.

    Nothing warns. Timestamps written after it differ from the task's frozen
    datetime, which breaks determinism — kill condition C — and the failure would
    show up as an inexplicable unit-test failure much later.
    """
    env = AppWorldEnvironment(
        world_factory=lambda task_id, **kw: NoisyWorld(task_id, **kw),
        state_digest=evaluation_digest,
        experiment_name="test_exp",
    )
    env.reset("train/t1", seed=0)

    with pytest.raises(RuntimeError, match="unfreez"):
        env.load_state("0")


def test_the_refusal_names_the_supported_alternative():
    """A refusal that does not say what to do instead gets worked around."""
    env = AppWorldEnvironment(
        world_factory=lambda task_id, **kw: NoisyWorld(task_id, **kw),
        state_digest=evaluation_digest,
        experiment_name="test_exp",
    )
    env.reset("train/t1", seed=0)

    with pytest.raises(RuntimeError, match="replay"):
        env.load_state("0")


def test_a_failing_close_still_releases_the_world():
    """Teardown raised on the real package. If the reference survives, the adapter
    is wedged: every later reset() retries the same broken close and no further
    task can run."""
    env = AppWorldEnvironment(
        world_factory=lambda task_id, **kw: FreezerBreakingWorld(task_id, **kw),
        state_digest=evaluation_digest,
        experiment_name="test_exp",
    )
    env.reset("train/t1", seed=0)

    with pytest.raises(AttributeError):
        env.close()

    # the world is gone even though close raised, so the adapter is reusable
    with pytest.raises(RuntimeError, match="reset"):
        env.execute("print(1)")


def test_a_failing_close_is_recorded_not_swallowed():
    """Swallowing it would hide freezegun corruption that poisons the next world's
    clock. The runner needs to see it to decide whether to keep going."""
    env = AppWorldEnvironment(
        world_factory=lambda task_id, **kw: FreezerBreakingWorld(task_id, **kw),
        state_digest=evaluation_digest,
        experiment_name="test_exp",
    )
    env.reset("train/t1", seed=0)

    with pytest.raises(AttributeError):
        env.close()

    assert len(env.teardown_errors) == 1
    assert "fake_names" in env.teardown_errors[0]


def test_reset_survives_a_previous_world_that_cannot_close():
    """The previous episode is already logged, so a broken teardown must not stop
    the next task. It is still recorded."""
    worlds = []

    def factory(task_id, **kw):
        world = FreezerBreakingWorld(task_id, **kw) if not worlds else NoisyWorld(task_id, **kw)
        worlds.append(world)
        return world

    env = AppWorldEnvironment(
        world_factory=factory,
        state_digest=evaluation_digest,
        experiment_name="test_exp",
    )
    env.reset("train/t1", seed=0)

    env.reset("train/t2", seed=0)

    assert env.world.task_id == "train/t2"
    assert len(env.teardown_errors) == 1


def test_reset_closes_the_old_world_before_building_the_new_one():
    """Order matters and is not cosmetic. AppWorld.initialize() calls close_all(),
    which stops every registered freezer without nulling it, so a second live world
    poisons the first world's close. Two worlds must never overlap."""
    events = []

    class Recording:
        def __init__(self, task_id, **kwargs):
            self.task_id = task_id
            self.kwargs = kwargs
            events.append(f"build {task_id}")

        def close(self):
            events.append(f"close {self.task_id}")

    env = AppWorldEnvironment(
        world_factory=lambda task_id, **kw: Recording(task_id, **kw),
        state_digest=evaluation_digest,
        experiment_name="test_exp",
    )
    env.reset("train/t1", seed=0)

    env.reset("train/t2", seed=0)

    assert events == ["build train/t1", "close train/t1", "build train/t2"]


# --- the evaluator leaks a freeze on any exception --------------------------
# appworld/evaluator.py starts a time freezer (line ~470) and stops it (~519) with
# no try/finally, and raises explicitly in between on a db_version mismatch. an
# exception anywhere in evaluation therefore leaks a freeze that is never popped.
#
# the leak is silent and permanent. the world's own close() then pops down to the
# leaked frame instead of emptying the stack, so freezegun never restores
# datetime, and the next task's freezer becomes a nested one that will raise when
# it eventually empties the stack. one failed evaluate() corrupts every later task
# in the process, which for the pilot is tens of thousands of calls.


class LeakyEvaluationWorld:
    """Evaluates by starting a freeze and then failing, exactly as appworld does."""

    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.task = None
        self.closed = False
        self.leaked: list = []

    def evaluate(self):
        from freezegun import api

        freezer = api._freeze_time(
            datetime.datetime(2023, 5, 12, 9, 0, 0), tz_offset=0, ignore=[],
            tick=False, as_arg=False, as_kwarg="", auto_tick_seconds=0,
            real_asyncio=False,
        )
        freezer.start()
        self.leaked.append(freezer)
        raise RuntimeError("task was generated with a different db_version")

    def close(self):
        self.closed = True


def _freeze_depth() -> int:
    from freezegun import api

    return len(api.freeze_factories)


def test_a_leaked_time_freeze_during_evaluation_is_refused(monkeypatch):
    """The guard that keeps a corrupted clock from being written to thousands of
    rows. Detection, not repair: freezegun's stack is not ours to rewrite, and a
    run that stops is worth far more than one that silently records wall-clock
    timestamps as if they were reproducible."""
    world = LeakyEvaluationWorld("train/t1")
    depth_before = _freeze_depth()

    try:
        with pytest.raises(RuntimeError, match="freezegun"):
            evaluation_digest(world)
    finally:
        for freezer in world.leaked:
            freezer.stop()

    assert _freeze_depth() == depth_before


def test_a_clean_evaluation_leaves_the_freezer_stack_alone(env):
    """The guard must not fire on the normal path, or it is worse than no guard."""
    env.reset("train/t1", seed=0)
    depth_before = _freeze_depth()

    env.state_hash()
    env.snapshot()

    assert _freeze_depth() == depth_before
