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
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol

from hgd.horizon import HStarStrategy, ground_truth_field, intrinsic_horizon


@contextlib.contextmanager
def _silenced():
    """Suppress AppWorld's evaluation report.

    `evaluate()` prints a full formatted test report every time it is called, and
    the adapter calls it for both `state_hash` and `snapshot` — at least once per
    step. Unsuppressed, one episode buries its own log under thousands of lines,
    and on a hosted notebook the I/O is a measurable slowdown.

    Redirected at the file descriptor rather than at ``sys.stdout``. AppWorld
    reports through a console that captures the stream when it is constructed,
    so a later ``redirect_stdout`` never sees the writes: the Phase 1 probe on
    2026-09-20 had reports flood through two nested layers of it. Falls back to
    the stream-level redirect where descriptors cannot be duplicated, which is
    the case in some notebook kernels.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()

    saved: list[tuple[int, int]] = []
    with tempfile.TemporaryFile() as sink:
        try:
            # literal 1 and 2, not sys.stdout.fileno(): under pytest capture and
            # in notebook kernels the stream object has no usable descriptor and
            # asking it raises, which silently skipped the whole redirect.
            for descriptor in (1, 2):
                with contextlib.suppress(OSError, ValueError):
                    saved.append((descriptor, os.dup(descriptor)))
                    os.dup2(sink.fileno(), descriptor)
            # both layers: the descriptors catch console and C-level writes, the
            # stream redirect catches ordinary prints in kernels whose streams
            # are not descriptor-backed.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


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
    with _silenced():
        payload = world.evaluate().to_dict()
    normalised = {
        "passes": sorted(payload.get("passes") or []),
        "failures": sorted(payload.get("failures") or []),
        "success": payload.get("success"),
        "num_tests": payload.get("num_tests"),
    }
    return _sha256(json.dumps(normalised, sort_keys=True, default=str))


def database_digest(root: str | Path) -> Callable[[_World], str]:
    """Digest every database file under ``root``.

    Exhaustive but brittle: any incidental byte difference counts as divergence.
    Supplied so the probe can quantify how much stricter it is than the
    behavioural digest before we commit to one.
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
        """
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
        """Evaluation state, for checkpoint predicates and the oracle summary."""
        with _silenced():
            return self.world.evaluate().to_dict()

    def gold_solution_code(self) -> str:
        """The released gold solution, as executable Python.

        Source of corrected actions for the oracle-fix intervention, and of
        realistic action traffic for the determinism gate.

        Explicitly **not** ``ground_truth.api_calls``: that field is a list of
        HTTP record dicts such as ``{'method': 'get', 'url': '/supervisor/profile',
        'data': {}}``, which ``execute()`` evaluates as dict literals — a silent
        no-op. The first Phase 1 gate run used it and measured nothing while
        reporting success.
        """
        ground_truth = self.world.task.ground_truth
        for name in ("compiled_solution_code", "solution_code"):
            code = ground_truth_field(ground_truth, name)
            if code:
                return str(code)
        raise ValueError(
            "ground truth exposes no compiled_solution_code or solution_code; "
            "gold solution is released for train/dev only"
        )

    def save_state(self) -> str:
        return self.world.save_state()

    def load_state(self, state_id: str) -> None:
        self.world.load_state(state_id)

    def intrinsic_horizon(self) -> int:
        return intrinsic_horizon(self.world.task.ground_truth, self._h_star_strategy)

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None
