"""Estimators — audit sections 25 and 27, prompt sections 8 and 16.

Four quantities and their uncertainty:

* ``horizon_gap`` — Delta(s) = log P_obs - sum log p_hat_i. This is Peng et al.'s
  **horizon residual** with the sign reversed (Delta = -Gamma_H); we keep the
  audit's convention and cite theirs on first use. Delta < 0 is excess
  degradation.
* ``risk_difference`` — the propagation estimator pi(e,u), as a difference of
  failure rates between the erroneous and oracle-fixed branches.
* ``mechanism_share`` — theta_k = (Delta_before - Delta_after,k) / Delta.
* ``paired_bootstrap_ci`` — the audit's specified interval for Delta.

Two deliberate refusals. ``horizon_gap`` refuses near-zero probabilities rather
than returning -inf: verified Qwen3-8B AppWorld performance is 5.4% TGC, so this
is a live risk, and an estimator that quietly returned -inf would turn an
instrumental floor effect into a headline result. ``mechanism_share`` refuses a
zero gap: with nothing to explain, a share is not small, it is undefined.

Intervals use Newcombe's method rather than a normal approximation, because the
planted Phase 2 conditions sit at the boundary (pi near 1 for a destructive
action, near 0 for a read) where the normal approximation is worst.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

# below this a log-ratio reports the floor, not the mechanism. phase 3a passes
# the stricter preregistered bound explicitly.
DEFAULT_MIN_PROBABILITY = 1e-6


class EstimationError(ValueError):
    """Raised when an estimate would be undefined or uninterpretable."""


@dataclass(frozen=True)
class Estimate:
    point: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def independence_prediction(p_hats: Sequence[float]) -> float:
    """P_ind(s) = product of matched atomic success rates."""
    if not p_hats:
        raise EstimationError("need at least one atomic success rate")
    for rate in p_hats:
        if not 0.0 <= rate <= 1.0:
            raise EstimationError(f"{rate} is not a probability")
    return float(np.prod(p_hats))


def horizon_gap(
    p_obs: float,
    p_hats: Sequence[float],
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> float:
    """Delta(s) = log P_obs - sum_i log p_hat_i."""
    if not 0.0 <= p_obs <= 1.0:
        raise EstimationError(f"{p_obs} is not a probability")

    p_ind = independence_prediction(p_hats)

    if p_obs <= min_probability or p_ind <= min_probability:
        raise EstimationError(
            f"floor effect: P_obs={p_obs:g}, P_ind={p_ind:g}, both must exceed "
            f"{min_probability:g}. A log-ratio of near-zero probabilities reports "
            "the floor, not the mechanism (kill condition F)."
        )

    return math.log(p_obs) - math.log(p_ind)


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval.

    Specified by the audit for p_hat_i because atomic rates sit near 0 and 1,
    where the normal approximation leaves the unit interval.
    """
    if n <= 0:
        raise EstimationError("n must be positive")
    if not 0 <= successes <= n:
        raise EstimationError(f"successes={successes} outside [0, {n}]")

    z = _z_for(confidence)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))

    return max(0.0, centre - margin), min(1.0, centre + margin)


def _z_for(confidence: float) -> float:
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def risk_difference(
    *,
    treated_failures: int,
    treated_n: int,
    control_failures: int,
    control_n: int,
    confidence: float = 0.95,
) -> Estimate:
    """Difference in failure rates, with a Newcombe interval.

    This is pi(e,u): the effect of leaving the erroneous action in place rather
    than replacing it with a corrected one.
    """
    if treated_n <= 0 or control_n <= 0:
        raise EstimationError("both arms need at least one observation")

    p1 = treated_failures / treated_n
    p2 = control_failures / control_n
    l1, u1 = wilson_interval(treated_failures, treated_n, confidence)
    l2, u2 = wilson_interval(control_failures, control_n, confidence)

    point = p1 - p2
    low = point - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = point + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

    return Estimate(point=point, low=max(-1.0, low), high=min(1.0, high))


def mechanism_share(*, delta_before: float, delta_after: float) -> float:
    """theta_k = (Delta_before - Delta_after,k) / Delta_before.

    Never clipped to [0, 1]. A share above one means the intervention more than
    closed the gap, and a negative share means it widened it; both are evidence
    about the design and belong in the results, not in a clamp.
    """
    if delta_before == 0.0:
        raise EstimationError(
            "no gap to decompose: Delta_before is zero, so a share is undefined "
            "rather than small"
        )
    return (delta_before - delta_after) / delta_before


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap over scenarios — the audit's interval for Delta.

    Pairing is by scenario: each element is one scenario's statistic, so
    resampling scenarios preserves the matched short/long structure that makes
    subtask difficulty identical by construction.
    """
    if len(values) == 0:
        raise EstimationError("cannot bootstrap an empty sample")

    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[draws].mean(axis=1)

    tail = (1 - confidence) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))
