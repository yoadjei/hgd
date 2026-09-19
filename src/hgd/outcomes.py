"""Outcome extraction from an AppWorld evaluation payload.

Payload keys verified against appworld 0.1.3.post1 on 2026-09-06:
``['difficulty', 'failures', 'num_tests', 'passes', 'success']``.

Three views of the same payload, because they answer different questions:

* ``task_success`` — binary Task Goal Completion; the audit's P_obs.
* ``pass_fraction`` — graded; tests passed over tests total.
* ``checkpoint_vector`` — per-test outcomes, which drive `recoverability` and
  `first_unrecovered_error`.

**On the graded view.** Blocker B2 is that small open-weight models score in the
low single digits of TGC on AppWorld, and Delta(s) = log P_obs - sum log p_hat is
undefined when both terms approach zero — `horizon_gap` refuses it rather than
returning -inf. A graded outcome would keep the estimator away from the floor.

Providing it here is **not** a decision to use it. Swapping a log-probability gap
for a gap in expected pass fraction changes the estimand, which is a scope change
under prompt section 5 and requires the Phase 3a pilot measurement first. The code
is ready for either; the research question is not settled by an import.
"""

from __future__ import annotations

from typing import Any, Mapping


def _passes(payload: Mapping[str, Any]) -> list[str]:
    return list(payload.get("passes") or [])


def _failures(payload: Mapping[str, Any]) -> list[str]:
    return list(payload.get("failures") or [])


def task_success(payload: Mapping[str, Any]) -> bool:
    """Binary Task Goal Completion.

    Falls back to "no failures and at least one pass" when the ``success`` flag
    is absent. An empty payload is false: absence of evidence of success is not
    success, and defaulting the other way would silently inflate P_obs.
    """
    flag = payload.get("success")
    if flag is not None:
        return bool(flag)

    passes, failures = _passes(payload), _failures(payload)
    return bool(passes) and not failures


def pass_fraction(payload: Mapping[str, Any]) -> float:
    """Fraction of unit tests passed.

    Raises on a task with no tests rather than returning zero: a task carrying no
    assertions is broken instrumentation, and scoring it 0.0 would quietly drag
    down every aggregate it entered.
    """
    passes, failures = _passes(payload), _failures(payload)
    total = payload.get("num_tests")
    if not total:
        total = len(passes) + len(failures)
    if not total:
        raise ValueError("evaluation payload reports no tests; cannot score it")

    return len(passes) / total


def checkpoint_vector(payload: Mapping[str, Any]) -> dict[str, bool]:
    """Per-test outcomes, as ``{test description: passed}``.

    This is what connects AppWorld's evaluation to the survival layer: collected
    across steps it gives one boolean series per test, which `recoverability`
    classifies and `first_unrecovered_error` reduces to the survival event.
    """
    vector = {name: True for name in _passes(payload)}
    vector.update({name: False for name in _failures(payload)})
    return vector
