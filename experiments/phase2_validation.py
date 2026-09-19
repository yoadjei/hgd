"""Phase 2: do the estimators recover planted mechanisms?

Audit Phase 2 acceptance: "shares recover planted ground truth within CI".
Failure means fixing the estimator or the causal design *before* touching the
real benchmark.

The validation logic. A planted policy errs with probability

    base + self_conditioning * 1[errors in history] + staleness * 1[state stale]

so we can construct the counterfactual world directly, by setting a coefficient
to zero, and ask whether the *intervention* recovers what *deleting the
mechanism* would have given:

    theta_true      = (Delta_full - Delta_mechanism_off) / Delta_full
    theta_estimated = (Delta_full - Delta_intervened)    / Delta_full

If the intervention is doing what its name claims, these agree. If it is doing
something else — leaking state repair into a history scrub, say — they diverge,
and we find out here rather than after spending the replay budget.

Runs on CPU in seconds. No model, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgd.env import DictEnvironment
from hgd.estimators import (
    EstimationError,
    horizon_gap,
    mechanism_share,
    risk_difference,
)
from hgd.interventions import (
    ScrubMode,
    oracle_summary,
    placebo_summary,
    scrub_history,
)
from hgd.parsing import ActionFormat, parse_action
from hgd.rule_labeller import EXCEPTION_LINE
from hgd.synthetic import DESTRUCTIVE_ACTION, PlantedPolicy

SYSTEM = {"role": "system", "content": "You are an agent."}
N_SCENARIOS = 400
LEVELS = (1, 2, 3)
RESULTS = REPO_ROOT / "results" / "phase2_validation.json"

PLANTED = {"base_error_rate": 0.18, "self_conditioning": 0.45, "staleness": 0.30}


def error_step_indices(messages: list[dict]) -> set[int]:
    """Turn indices whose observation reported an error."""
    steps, turn = set(), -1
    for message in messages:
        if message.get("role") == "assistant":
            turn += 1
        elif turn >= 0 and EXCEPTION_LINE.search(str(message.get("content", ""))):
            steps.add(turn)
    return steps


def run_trial(
    *,
    params: dict,
    s: int,
    seed: int,
    scrub: bool = False,
    scrub_mode: ScrubMode = ScrubMode.REMOVE,
    refresh: bool = False,
    placebo: bool = False,
    destructive_step: int | None = None,
    irreversible: bool = False,
    probe_action: str = DESTRUCTIVE_ACTION,
) -> bool:
    """One episode. Returns whether every stage checkpoint passes at the end."""
    env = DictEnvironment(irreversible_delete=irreversible)
    env.reset(f"synthetic/s{s}", seed)
    policy = PlantedPolicy(
        **params, seed=seed, n_keys=s,
        destructive_step=destructive_step, probe_action=probe_action,
    )

    messages: list[dict] = [SYSTEM]

    for _ in range(3 * s):
        view = list(messages)
        if scrub:
            view = scrub_history(
                view, error_steps=error_step_indices(view), mode=scrub_mode
            )
        if refresh:
            summary = oracle_summary(env.snapshot())
            if placebo:
                # byte- and line-matched but stateless, so it cannot relieve
                # staleness; if it moves the outcome as much as the real summary,
                # theta_state is not identified
                summary = placebo_summary(summary)
            view = view + [{"role": "user", "content": summary}]

        response = policy.generate(view)
        parsed = parse_action(response.text, ActionFormat.CODE)
        if parsed.action is None:
            continue

        result = env.execute(parsed.action)
        messages.append({"role": "assistant", "content": response.text})
        messages.append({"role": "user", "content": result})

    state = env.snapshot()
    return all(state.get(f"k{i}") == str(i) for i in range(s))


def success_rate(**kwargs) -> tuple[float, list[bool]]:
    outcomes = [run_trial(seed=seed, **kwargs) for seed in range(N_SCENARIOS)]
    return sum(outcomes) / len(outcomes), outcomes


def delta_for(params: dict, **arm) -> float:
    """Horizon gap at the deepest level, against matched atomic rates."""
    p_hats = []
    for _ in range(max(LEVELS)):
        rate, _ = success_rate(params=params, s=1, **arm)
        p_hats.append(rate)

    p_obs, _ = success_rate(params=params, s=max(LEVELS), **arm)
    return horizon_gap(p_obs, p_hats)


def main() -> int:
    findings: dict[str, object] = {"planted": PLANTED, "n_scenarios": N_SCENARIOS}
    print("=" * 72)
    print("PHASE 2 — estimator validation against planted ground truth")
    print("=" * 72)

    # --- horizon gap exists in the synthetic world -------------------------
    try:
        delta_full = delta_for(PLANTED)
    except EstimationError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"\nDelta(3) full model            = {delta_full:+.4f}")
    findings["delta_full"] = delta_full

    # --- counterfactual worlds: the mechanism simply deleted ---------------
    delta_no_self = delta_for({**PLANTED, "self_conditioning": 0.0})
    delta_no_state = delta_for({**PLANTED, "staleness": 0.0})

    theta_self_true = mechanism_share(delta_before=delta_full, delta_after=delta_no_self)
    theta_state_true = mechanism_share(delta_before=delta_full, delta_after=delta_no_state)

    # --- interventions: the estimator's answer -----------------------------
    delta_scrubbed = delta_for(PLANTED, scrub=True)
    delta_refreshed = delta_for(PLANTED, refresh=True)

    theta_self_hat = mechanism_share(delta_before=delta_full, delta_after=delta_scrubbed)
    theta_state_hat = mechanism_share(delta_before=delta_full, delta_after=delta_refreshed)

    print(f"\n{'mechanism':<18}{'planted (true)':>16}{'recovered':>14}{'abs err':>10}")
    print("-" * 60)
    for label, true, hat in (
        ("self-conditioning", theta_self_true, theta_self_hat),
        ("state staleness", theta_state_true, theta_state_hat),
    ):
        print(f"{label:<18}{true:>16.3f}{hat:>14.3f}{abs(true - hat):>10.3f}")

    findings["theta"] = {
        "self": {"true": theta_self_true, "estimated": theta_self_hat},
        "state": {"true": theta_state_true, "estimated": theta_state_hat},
    }

    # --- null control: no planted mechanism, share must be ~0 --------------
    flat = {"base_error_rate": 0.25, "self_conditioning": 0.0, "staleness": 0.0}
    delta_flat = delta_for(flat)
    delta_flat_scrubbed = delta_for(flat, scrub=True)
    theta_null = mechanism_share(delta_before=delta_flat, delta_after=delta_flat_scrubbed)
    print(f"\nnull control (no mechanism planted): theta_self = {theta_null:+.3f}")
    findings["theta_null"] = theta_null

    # --- identification probe: can the estimator be made to fail? ---------
    # scrubbing shortens the transcript, so a length-sensitive policy is moved
    # through a channel other than erroneous history. if theta_self were still
    # recovered exactly here, the validation would not be testing identification.
    confounded = {**PLANTED, "length_sensitivity": 0.004}
    delta_conf = delta_for(confounded)
    delta_conf_no_self = delta_for({**confounded, "self_conditioning": 0.0})
    theta_conf_true = mechanism_share(delta_before=delta_conf, delta_after=delta_conf_no_self)

    print("\nidentification probe (length-confounded policy):")
    print(f"  theta_self true = {theta_conf_true:+.3f}")
    print(f"  {'scrub mode':<18}{'recovered':>12}{'bias':>10}")
    print("  " + "-" * 40)

    probe = {"theta_true": theta_conf_true, "modes": {}}
    for mode in (ScrubMode.REMOVE, ScrubMode.LENGTH_MATCHED):
        delta_arm = delta_for(confounded, scrub=True, scrub_mode=mode)
        theta_hat = mechanism_share(delta_before=delta_conf, delta_after=delta_arm)
        bias = theta_hat - theta_conf_true
        probe["modes"][mode.value] = {"estimated": theta_hat, "bias": bias}
        print(f"  {mode.value:<18}{theta_hat:>12.3f}{bias:>+10.3f}")

    findings["identification_probe"] = probe
    d10_helps = abs(probe["modes"]["length_matched"]["bias"]) < abs(
        probe["modes"]["remove"]["bias"]
    )
    print(f"  D10 (length-matched) reduces bias: {'YES' if d10_helps else 'NO'}")
    findings["d10_reduces_bias"] = d10_helps

    # --- placebo control for the staleness arm ----------------------------
    delta_placebo = delta_for(PLANTED, refresh=True, placebo=True)
    theta_placebo = mechanism_share(delta_before=delta_full, delta_after=delta_placebo)
    print("\nplacebo control for state_refresh:")
    print(f"  theta_state oracle summary = {theta_state_hat:+.3f}")
    print(f"  theta_state placebo        = {theta_placebo:+.3f}")
    placebo_ok = abs(theta_placebo) < 0.5 * abs(theta_state_hat)
    print(f"  placebo moves the gap materially less: {'YES' if placebo_ok else 'NO'}")
    findings["placebo"] = {
        "theta_oracle": theta_state_hat,
        "theta_placebo": theta_placebo,
        "passes": placebo_ok,
    }

    # --- propagation: planted irreversibility (audit T2.2) ----------------
    print("\npropagation pi(e,u), irreversible delete vs harmless read:")
    clean = {"base_error_rate": 0.0, "self_conditioning": 0.0, "staleness": 0.0}
    propagation = {}
    control_fail = sum(
        not run_trial(params=clean, s=3, seed=seed, irreversible=True)
        for seed in range(N_SCENARIOS)
    )
    for u, step in ((0.2, 1), (0.5, 3), (0.8, 5)):
        row = {}
        for label, probe in (("delete", DESTRUCTIVE_ACTION), ("read", "read(k{index})")):
            failures = sum(
                not run_trial(
                    params=clean, s=3, seed=seed, destructive_step=step,
                    irreversible=True, probe_action=probe,
                )
                for seed in range(N_SCENARIOS)
            )
            estimate = risk_difference(
                treated_failures=failures, treated_n=N_SCENARIOS,
                control_failures=control_fail, control_n=N_SCENARIOS,
            )
            row[label] = {
                "point": estimate.point, "low": estimate.low, "high": estimate.high
            }
            print(f"  u={u} {label:<7}: pi = {estimate.point:+.3f} "
                  f"[{estimate.low:+.3f}, {estimate.high:+.3f}]")
        propagation[str(u)] = row

    findings["propagation"] = propagation

    # --- verdict -----------------------------------------------------------
    tolerance = 0.20
    # t2.2: pi near 1 for delete, near 0 for read, cis disjoint
    t22 = all(
        row["delete"]["low"] > row["read"]["high"] and row["delete"]["point"] > 0.8
        and abs(row["read"]["point"]) < 0.2
        for row in propagation.values()
    )
    findings["t22_passes"] = t22
    print(f"\nT2.2 (delete propagates, read does not, CIs disjoint): "
          f"{'PASS' if t22 else 'FAIL'}")

    recovered = (
        abs(theta_self_true - theta_self_hat) < tolerance
        and abs(theta_state_true - theta_state_hat) < tolerance
        and abs(theta_null) < tolerance
        and t22
        and placebo_ok
    )
    findings["passes_gate"] = recovered

    print("\n" + "=" * 72)
    print("PHASE 2 GATE:", "PASS" if recovered else "FAIL")
    if not recovered:
        print("Estimators do not recover planted ground truth.")
        print("Audit Phase 2: fix the estimator or the causal design before")
        print("touching the real benchmark. Do not proceed to Phase 3.")
    print("=" * 72)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(findings, indent=2, default=str))
    print(f"wrote {RESULTS}")
    return 0 if recovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
