"""Intrinsic horizon H* and normalized trajectory position u.

H* is HORIZON's definition (arXiv:2604.11978), verified verbatim: "the minimum
number of effective actions required by an optimal policy to complete the task".
We borrow the definition and cite it; only its use as a regression covariate is
ours.

H* is deliberately a *strategy* rather than a number. Audit section 32's R3
raises that reference solutions do not determine H* uniquely, and the audit's
ablation list requires an H* sensitivity analysis. Making the choice explicit
and logged turns that analysis into a change of argument.

The two strategies differ in availability as well as in value: ``api_calls``
comes from `ground_truth` and exists only on train/dev, while the solution-line
count is exposed as task metadata on every split.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class HStarStrategy(Enum):
    """How the intrinsic horizon is read off a task's ground truth."""

    API_CALLS = "api_calls"
    SOLUTION_LINES = "solution_lines"


def ground_truth_field(source: Any, name: str) -> Any:
    """Read ``name`` from an object attribute or a mapping key.

    AppWorld's ``ground_truth`` is a ``GroundTruth`` object, verified against
    appworld 0.1.3.post1 on 2026-09-06; the synthetic environments and tests use
    plain dicts. Both are supported so that H* has one implementation rather than
    a real one and a test one.
    """
    value = getattr(source, name, None)
    if value is None and isinstance(source, Mapping):
        value = source.get(name)
    return value


def intrinsic_horizon(
    ground_truth: Any,
    strategy: HStarStrategy = HStarStrategy.API_CALLS,
) -> int:
    """Minimum effective actions for a task, under the chosen strategy.

    Raises rather than defaulting: a silently substituted H* would corrupt u for
    an entire split, and u is the axis every hazard estimate is indexed by.
    """
    if strategy is HStarStrategy.API_CALLS:
        calls = ground_truth_field(ground_truth, "api_calls")
        if calls is None:
            raise ValueError(
                "ground truth has no 'api_calls'; it is released for train/dev only. "
                "Use HStarStrategy.SOLUTION_LINES on held-out splits."
            )
        # repeats count: calling the same api twice is two actions
        value = len(calls)
    else:
        # the compiled count is preferred because the uncompiled one is
        # degenerate. the census over all 90 train tasks on 2026-09-20 read
        # num_solution_code_lines as exactly 3 for every task, against api call
        # counts spanning 5 to 244, so it carries no information about task
        # size; num_compiled_solution_code_lines spans 18 to 86. taking the
        # uncompiled field would set H* to 3 across a whole split and make u
        # meaningless precisely where api_calls is withheld.
        for name in ("num_compiled_solution_code_lines", "num_solution_code_lines"):
            value = ground_truth_field(ground_truth, name)
            if value is None:
                metadata = ground_truth_field(ground_truth, "metadata") or {}
                value = ground_truth_field(metadata, name)
            if value is not None:
                break
        if value is None:
            raise ValueError(
                "ground truth has no 'num_compiled_solution_code_lines' or "
                "'num_solution_code_lines'"
            )

    if value <= 0:
        raise ValueError(f"intrinsic horizon must be positive, got {value}")

    return int(value)


def normalized_position(step: int, h_star: int) -> float:
    """u = t / H*.

    Values above 1 are meaningful and are never clipped: the step limit is 3H*,
    so the interesting region of the hazard curve extends past the intrinsic
    horizon, and clipping would pile mass at the boundary.
    """
    if h_star <= 0:
        raise ValueError(f"h_star must be positive, got {h_star}")
    return step / h_star
