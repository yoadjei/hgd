"""Environment protocol and a deterministic in-memory implementation.

The harness and replay talk to this narrow protocol rather than to AppWorld
directly, for three reasons: replay logic stays testable without a GPU or a
benchmark install; the audit's designated fallback (tau-squared-bench) can be
swapped in without touching the estimators; and the Phase 2 synthetic task of
audit section 24 is itself an ``Environment``, so the estimators are validated
against planted ground truth through exactly the interface they will later use.

``DictEnvironment`` is the toolified key-value store of audit section 24: an
external store the model must read and write through tools rather than holding
state in context. That is what separates state staleness from self-conditioning
in the synthetic validation.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Protocol, runtime_checkable

_CALL = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$", re.DOTALL)


@runtime_checkable
class Environment(Protocol):
    """The minimum an environment must offer for logging, replay and branching."""

    def reset(self, task_id: str, seed: int) -> None:
        """Return to the task's initial state. Must be deterministic given seed."""

    def execute(self, action: str) -> str:
        """Apply one action and return the observation the agent will see."""

    def state_hash(self) -> str:
        """Stable digest of task-relevant state. Equal hashes mean equal states."""

    def snapshot(self) -> object:
        """Oracle view of true state.

        Two consumers: checkpoint predicates (tier-2 labels) and the
        state-refresh intervention, which derives its summary from the
        environment rather than from the model so that the belief channel is the
        only thing the intervention touches.
        """


class DictEnvironment:
    """A deterministic key-value world with store / read / delete.

    ``delete`` is recoverable by default. With ``irreversible_delete`` a deleted
    key is tombstoned and can never be stored again, which is the planted
    destructive action of audit section 24: it gives the propagation estimator a
    known ground truth (pi near 1 for a delete, near 0 for a read).

    State is fully determined by the action sequence, so ``task_id`` and ``seed``
    are accepted for protocol compatibility and otherwise ignored.
    """

    def __init__(
        self, *, nondeterministic: bool = False, irreversible_delete: bool = False
    ) -> None:
        self._store: dict[str, str] = {}
        self._tombstones: set[str] = set()
        # opt-in: without tombstones a deleted key can simply be stored again,
        # which is how the first phase 2 run measured pi = 0 for a planted delete
        self._irreversible_delete = irreversible_delete
        # hidden entropy, so a test can prove the fidelity gate is able to fail
        self._nondeterministic = nondeterministic
        self._entropy: str = ""

    def reset(self, task_id: str, seed: int) -> None:
        self._store = {}
        self._tombstones = set()
        self._entropy = ""

    def execute(self, action: str) -> str:
        match = _CALL.match(action or "")
        if match is None:
            return "Exception: could not parse action"

        op, raw_args = match.group(1), match.group(2)
        args = [part.strip() for part in raw_args.split(",")] if raw_args.strip() else []

        if self._nondeterministic:
            self._entropy = uuid.uuid4().hex

        if op == "store":
            if len(args) != 2:
                return "TypeError: store expects (key, value)"
            if args[0] in self._tombstones:
                return f"PermissionError: {args[0]} was deleted and cannot be restored"
            self._store[args[0]] = args[1]
            return "ok"
        if op == "read":
            if len(args) != 1:
                return "TypeError: read expects (key)"
            return self._store.get(args[0], "KeyError: missing key")
        if op == "delete":
            if len(args) != 1:
                return "TypeError: delete expects (key)"
            self._store.pop(args[0], None)
            if self._irreversible_delete:
                self._tombstones.add(args[0])
            return "ok"
        return f"AttributeError: unknown operation {op!r}"

    def state_hash(self) -> str:
        payload = json.dumps(
            {
                "store": self._store,
                # tombstones are state: replay must tell a deleted world from a
                # never-written one
                "tombstones": sorted(self._tombstones),
                "entropy": self._entropy,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def snapshot(self) -> dict[str, str]:
        """Oracle view of true state, for the state-refresh intervention."""
        return dict(self._store)
