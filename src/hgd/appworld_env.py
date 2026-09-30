"""AppWorld adapter implementing the ``Environment`` protocol.

AppWorld is code-as-action: the agent writes Python against ``apis.*`` in a
persistent IPython shell, and `world.execute(code)` returns printed output or a
traceback. Variables persist across calls, so the shell namespace is part of the
agent's own memory rather than of environment state — replaying the logged action
prefix reconstructs it.

**The one thing AppWorld does not provide is a state digest**, and the Phase 1
fidelity gate is meaningless without one. Rather than guess, the digest is
injected. Two candidates are supplied here and `notebooks/phase1_gate.py` decides
between them empirically against the real package:

* ``evaluation_digest`` — hashes the unit-test outcome vector. Behavioural rather
  than exhaustive: two states agreeing on every check are equivalent *for our
  estimands*, which is arguably the right equivalence, but it is coarse and it
  will not see collateral state the checks ignore.
* ``database_digest`` — hashes the on-disk databases. Exhaustive, but will report
  divergence for incidental differences such as timestamps, which would fail the
  gate for reasons that do not threaten any causal claim.

Choosing between them is a real methodological decision, not a detail: too coarse
and replay fidelity is overstated, too fine and kill condition C fires spuriously.
The probe reports both.

**On appworld's time freezer.** appworld 0.1.3.post1 keeps three independent
freezer registries — per world (``AppWorld.id_to_time_freezer``), per requester
(``Requester.time_freezers_or_ids``) and one inside ``evaluator.py`` — and stops
them with no shared ordering discipline. Two facts combine badly, both reproduced
against real freezegun:

* ``freeze_time.stop()`` in ``common/utils.py`` guards on
  ``self._freezer is not None`` but never nulls it, and
  ``AppWorld.close_all()`` calls ``time_freezer.stop()`` directly instead of
  ``unset_local_date_and_time()``, so the freezer stays non-None and can be
  stopped twice.
* freezegun's ``_freeze_time.start()`` returns early when a freeze is already
  active, so a *nested* freezer never gets ``fake_names``, while ``stop()`` reads
  ``fake_names`` only when it empties the global stack. Whichever freezer empties
  the stack must therefore be the one that was outermost.

``load_state()`` calls ``close_all()`` and restarts nothing, which leaves the
world running on wall-clock time with no warning, and makes the following
``close()`` a double stop. ``initialize()`` calls ``close_all()`` too, so two live
worlds poison the first one's teardown. Hence two rules here: ``load_state`` is
refused outright, and no two worlds are ever alive at once.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Protocol

from hgd.horizon import HStarStrategy, ground_truth_field, intrinsic_horizon
from hgd.outcomes import outcome_name


# opened once at import, on purpose, and never closed. AppWorld replaces ``open``
# with a read-only version while it executes an action, so anything that opens a
# file inside that window raises PermissionError; an earlier version used
# tempfile.TemporaryFile() here and the guard turned a silencer into a crash.
# ``dup`` and ``dup2`` are syscalls rather than opens, so reusing one descriptor
# is safe where creating a file is not.
try:
    _NULL_STREAM: Any = open(os.devnull, "w", encoding="utf-8")
    atexit.register(_NULL_STREAM.close)
except OSError:  # pragma: no cover - a host with no null device
    _NULL_STREAM = None


@contextlib.contextmanager
def silenced():
    """Send writes to descriptors 1 and 2 to the null device.

    `evaluate()` prints a full formatted test report every time it is called, and
    the adapter calls it for both `state_hash` and `snapshot`, at least once per
    step. Unsuppressed, one episode buries its own log under thousands of lines.

    Both layers matter and both are descriptor-backed. Duplicating 1 and 2 catches
    console and C-level writes; the stream redirect catches ordinary prints in a
    kernel whose streams are not descriptor-backed. Literal 1 and 2 rather than
    ``sys.stdout.fileno()``, because under pytest capture and in notebook kernels
    the stream object has no usable descriptor and asking raises, which silently
    skipped the redirect altogether.

    The sink is a real file, never an ``io.StringIO``. AppWorld calls
    ``faulthandler.enable()`` when it disables its safety guard, which needs a
    genuine descriptor; redirecting to a StringIO made every ``execute()`` raise
    ``io.UnsupportedOperation: fileno`` and errored the gate on every task.
    """
    if _NULL_STREAM is None:
        yield
        return

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()

    saved: list[tuple[int, int]] = []
    try:
        for descriptor in (1, 2):
            with contextlib.suppress(OSError, ValueError):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(_NULL_STREAM.fileno(), descriptor)
        with contextlib.redirect_stdout(_NULL_STREAM), \
             contextlib.redirect_stderr(_NULL_STREAM):
            yield
    finally:
        for descriptor, backup in saved:
            with contextlib.suppress(OSError, ValueError):
                os.dup2(backup, descriptor)
                os.close(backup)


class _World(Protocol):
    """The documented AppWorld surface this adapter depends on."""

    task: Any

    def execute(self, code: str) -> str: ...
    def task_completed(self) -> bool: ...
    def evaluate(self) -> Any: ...
    def save_state(self) -> str: ...
    def load_state(self, state_id: str) -> None: ...
    def close(self) -> None: ...


# the gold solution is a `def solution(apis, requester)` wrapper, so executing
# it only defines a function. this call is what actually runs it, and both names
# are already bound in the shell.
SOLUTION_INVOCATION = "solution(apis, requester)"


def gold_solution_code(ground_truth: Any) -> str:
    """The released gold solution for a task, as executable Python.

    Source of corrected actions for the oracle-fix intervention, and of
    realistic action traffic for the determinism gate.

    Explicitly **not** ``api_calls``: that field holds HTTP record dicts such as
    ``{'method': 'get', 'url': '/supervisor/profile', 'data': {}}``, which
    ``execute()`` evaluates as dict literals, a silent no-op. The first gate run
    used it, measured nothing, and would have reported total fidelity.
    """
    for name in ("compiled_solution_code", "solution_code"):
        code = ground_truth_field(ground_truth, name)
        if code:
            return str(code)
    raise ValueError(
        "ground truth exposes no compiled_solution_code or solution_code; "
        "gold solution is released for train/dev only"
    )


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _sorted_outcomes(items: Any) -> list[str]:
    """Canonical, sortable form of a pass or fail vector.

    Entries are not always strings. Once a task completes, AppWorld returns
    dicts, and ``sorted`` raises ``TypeError: '<' not supported between
    instances of 'dict' and 'dict'``. Serialising each entry first gives a
    total order that holds for either shape, and keeps ordering from
    masquerading as a state difference.
    """
    return sorted(json.dumps(item, sort_keys=True, default=str) for item in (items or []))


def _freezer_depth() -> int | None:
    """How many freezegun freezes are active, or None if that cannot be read.

    Reaching into a third-party global is not something to do lightly, but the
    alternative is worse. See ``_evaluate``.
    """
    try:
        from freezegun import api

        return len(api.freeze_factories)
    except Exception:  # pragma: no cover - freezegun absent or internals moved
        return None


def _evaluate(world: _World) -> dict[str, Any]:
    """Evaluate quietly, and refuse to continue if the time freezer leaked.

    appworld's evaluator starts a time freezer and stops it with no ``try``/
    ``finally`` (``evaluator.py`` around lines 470 and 519) and raises explicitly
    in between on a ``db_version`` mismatch. Any exception during evaluation
    therefore leaks a freeze that is never popped, and the leak is silent and
    permanent: the world's own ``close()`` then pops down to the leaked frame
    instead of emptying the stack, so freezegun never restores ``datetime``, and
    the next task's freezer becomes a nested one that raises when it finally does
    empty the stack. One failed evaluation corrupts every later task in the
    process.

    This detects rather than repairs. freezegun's stack is not ours to rewrite,
    and the pilot calls this tens of thousands of times: a run that stops is worth
    much more than one that keeps writing wall-clock timestamps as though they
    were reproducible.
    """
    before = _freezer_depth()
    try:
        with silenced():
            return dict(world.evaluate().to_dict())
    finally:
        after = _freezer_depth()
        if before is not None and after is not None and after != before:
            raise RuntimeError(
                f"appworld's evaluate() left the freezegun freeze stack at {after}, "
                f"not {before}. its evaluator starts a time freezer and stops it "
                "with no try/finally, so an exception inside evaluation leaks the "
                "freeze permanently: time stops being restored and every later task "
                "in this process runs on a corrupted clock. stopping here rather "
                "than recording timestamps that cannot be reproduced."
            )


def evaluation_digest(world: _World) -> str:
    """Digest the unit-test outcome vector.

    Keys verified against appworld 0.1.3.post1 on 2026-09-06:
    ``['difficulty', 'failures', 'num_tests', 'passes', 'success']``. The field
    is ``failures``, not ``fails``; the first version read ``fails``, always got
    None, and silently digested only the pass vector.

    Lists are sorted so that their ordering cannot masquerade as a state
    difference. ``difficulty`` is excluded because it is a static property of the
    task, not of the state.
    """
    payload = _evaluate(world)
    normalised = {
        "passes": _sorted_outcomes(payload.get("passes")),
        "failures": _sorted_outcomes(payload.get("failures")),
        "success": payload.get("success"),
        "num_tests": payload.get("num_tests"),
    }
    return _sha256(json.dumps(normalised, sort_keys=True, default=str))


def database_digest(root: str | Path) -> Callable[[_World], str]:
    """Digest every database file under ``root``.

    Exhaustive but brittle: any incidental byte difference counts as divergence.
    Supplied so the probe can quantify how much stricter it is than the
    behavioural digest before we commit to one.

    It hashes timestamps along with everything else, so it is only sound while the
    task clock stays frozen. Any event that unfreezes time — appworld's
    ``load_state()`` being the known one, hence the refusal above — makes it report
    divergence for two runs that are behaviourally identical, and kill condition C
    would fire on an artefact. Choosing this digest means the frozen-clock check in
    ``notebooks/adapter_probe.py`` stops being a nicety.
    """
    base = Path(root)

    def _digest(_world: _World) -> str:
        accumulator = hashlib.sha256()
        for path in sorted(base.rglob("*.sqlite*")) + sorted(base.rglob("*.db")):
            accumulator.update(path.name.encode())
            accumulator.update(path.read_bytes())
        return accumulator.hexdigest()

    return _digest


class AppWorldEnvironment:
    """``Environment`` over AppWorld, with an injected state digest."""

    def __init__(
        self,
        world_factory: Callable[..., _World],
        state_digest: Callable[[_World], str],
        experiment_name: str,
        *,
        ground_truth_mode: str = "full",
        h_star_strategy: HStarStrategy = HStarStrategy.API_CALLS,
    ) -> None:
        self._factory = world_factory
        self._digest = state_digest
        self._experiment_name = experiment_name
        self._ground_truth_mode = ground_truth_mode
        self._h_star_strategy = h_star_strategy
        self._world: _World | None = None
        # appworld's teardown can raise, see the freezer note above. a runner needs
        # to see that rather than have it swallowed, so it is recorded here.
        self.teardown_errors: list[str] = []

    @property
    def world(self) -> _World:
        if self._world is None:
            raise RuntimeError("no active world: call reset() before use")
        return self._world

    def reset(self, task_id: str, seed: int) -> None:
        """Load a task.

        ``seed`` is logged but not passed to AppWorld: the environment is
        deterministic given the task, and the seed governs model sampling. Keeping
        it in the signature preserves one ``Environment`` protocol across the real
        and synthetic environments.

        The previous world is closed *before* the next is built, and that ordering
        is load-bearing: ``AppWorld.initialize()`` calls ``close_all()``, which
        stops every registered time freezer without nulling it, so two live worlds
        leave the first one unable to close. A teardown failure is recorded and
        then tolerated, because the previous episode is already logged and the next
        task still has to run.
        """
        with contextlib.suppress(Exception):
            self.close()
        self._world = self._factory(
            task_id,
            experiment_name=self._experiment_name,
            ground_truth_mode=self._ground_truth_mode,
        )

    def execute(self, action: str) -> str:
        return self.world.execute(action)

    def task_completed(self) -> bool:
        return self.world.task_completed()

    def state_hash(self) -> str:
        return self._digest(self.world)

    def snapshot(self) -> dict[str, Any]:
        """Evaluation state, for checkpoint predicates and the oracle summary.

        The pass and fail vectors are normalised to names. Raw entries are strings
        until a task completes and dicts afterwards, and a predicate written
        ``name in state["failures"]`` silently returns False on the dict form.
        Predicates are the tier-2 label source, so that False would move mass
        between competing risks with nothing looking wrong.
        """
        payload = _evaluate(self.world)
        for key in ("passes", "failures"):
            if key in payload:
                payload[key] = [outcome_name(entry) for entry in (payload[key] or [])]
        return payload

    def gold_solution_code(self) -> str:
        """The released gold solution for the active task."""
        return gold_solution_code(self.world.task.ground_truth)

    def save_state(self) -> str:
        """Write a checkpoint directory.

        Kept, unlike ``load_state``: writing the checkpoint touches no freezer and
        the directory is useful for post-hoc inspection. Restoring from it is the
        part that corrupts the clock.
        """
        return self.world.save_state()

    def load_state(self, state_id: str) -> None:
        """Refused. See the freezer note in the module docstring.

        Not a delegation. appworld's ``load_state()`` calls ``AppWorld.close_all()``
        and restarts no freezer, so the world silently continues on wall-clock time
        and the next ``close()`` stops the same freezegun instance twice. The first
        consequence is the dangerous one: it is invisible, and it makes the
        environment non-deterministic, which is exactly what kill condition C
        certifies against.
        """
        raise RuntimeError(
            "appworld's load_state() unfreezes the world clock and does not restart "
            "it: it calls AppWorld.close_all(), which stops the task's time freezer, "
            "and restores no freezer afterwards. every timestamp written after it is "
            "wall-clock rather than the task's frozen datetime, which breaks "
            "determinism silently, and the next close() then double-stops the same "
            "freezegun instance. branch by replay instead: hgd.replay.replay_prefix "
            "re-executes a logged action prefix, and that is the path the phase 1 "
            "gate validated at total fidelity."
        )

    def intrinsic_horizon(self) -> int:
        return intrinsic_horizon(self.world.task.ground_truth, self._h_star_strategy)

    def close(self) -> None:
        """Release the world, whether or not appworld's teardown succeeds.

        The reference is dropped first, on purpose. appworld's ``close()`` can raise
        out of its own time freezer, and a surviving reference would wedge the
        adapter: every later ``reset()`` would retry the same broken close and no
        further task could run.

        The error is recorded and then re-raised rather than swallowed. Swallowing
        it would hide freezegun corruption that silently poisons the next world's
        clock, and a wrong clock is worse than a loud failure.
        """
        world, self._world = self._world, None
        if world is None:
            return
        try:
            world.close()
        except Exception as exc:
            self.teardown_errors.append(f"{type(exc).__name__}: {exc}")
            raise
