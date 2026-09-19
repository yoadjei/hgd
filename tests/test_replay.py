"""Replay tests — audit task T1.2, and the Phase 1 gate.

The gate is "replay reproduces a logged trajectory bit-exactly under fixed seed,
20/20 hash matches". Kill condition C makes this load-bearing for the entire
project: no causal claim survives a replay system that cannot reproduce its own
logs, so the fidelity check has to be an assertion, not an inspection.
"""

import pytest

from hgd.env import DictEnvironment
from hgd.replay import FidelityReport, replay_prefix, verify_fidelity


@pytest.fixture
def env():
    return DictEnvironment()


def _record_trajectory(env, task_id, seed, actions, make_step):
    """Run actions against a fresh env, returning logged records."""
    env.reset(task_id, seed)
    records = []
    for index, action in enumerate(actions):
        result = env.execute(action)
        records.append(
            make_step(
                step=index,
                task_id=task_id,
                seed=seed,
                parsed_action=action,
                tool_name=action.split("(")[0],
                tool_args={"raw": action},
                tool_result=result,
                env_state_hash=env.state_hash(),
            )
        )
    return records


ACTIONS = ["store(a, 1)", "store(b, 2)", "read(a)", "store(a, 3)", "delete(b)"]


def test_replay_reproduces_the_logged_state_hash_at_every_prefix(env, make_step):
    records = _record_trajectory(env, "task-1", 0, ACTIONS, make_step)

    for boundary in range(1, len(records) + 1):
        result = replay_prefix(records, boundary, DictEnvironment())

        assert result.matches, f"prefix of length {boundary} diverged"
        assert result.state_hash == records[boundary - 1].env_state_hash


def test_replay_of_an_empty_prefix_returns_the_initial_state(env, make_step):
    records = _record_trajectory(env, "task-1", 0, ACTIONS, make_step)

    result = replay_prefix(records, 0, DictEnvironment())

    assert result.matches
    assert result.diverged_at is None


def test_replay_is_deterministic_across_repeated_calls(env, make_step):
    records = _record_trajectory(env, "task-1", 0, ACTIONS, make_step)

    first = replay_prefix(records, len(records), DictEnvironment())
    second = replay_prefix(records, len(records), DictEnvironment())

    assert first.state_hash == second.state_hash


def test_replay_reports_the_first_diverging_step(env, make_step):
    records = _record_trajectory(env, "task-1", 0, ACTIONS, make_step)
    tampered = list(records)
    tampered[2] = make_step(
        step=2,
        parsed_action=records[2].parsed_action,
        tool_name=records[2].tool_name,
        tool_args=records[2].tool_args,
        tool_result=records[2].tool_result,
        env_state_hash="deadbeef" * 8,
    )

    result = replay_prefix(tampered, len(tampered), DictEnvironment())

    assert not result.matches
    assert result.diverged_at == 2


def test_replay_does_not_call_a_model(env, make_step):
    """Replay re-executes logged actions only; a model in the loop would break determinism."""
    records = _record_trajectory(env, "task-1", 0, ACTIONS, make_step)

    result = replay_prefix(records, len(records), DictEnvironment())

    assert result.steps_executed == len(records)


def test_replaying_beyond_the_trajectory_length_is_rejected(env, make_step):
    records = _record_trajectory(env, "task-1", 0, ACTIONS, make_step)

    with pytest.raises(ValueError, match="beyond"):
        replay_prefix(records, len(records) + 1, DictEnvironment())


def test_state_diverges_when_the_environment_is_nondeterministic(make_step):
    """A leaky environment must fail the gate rather than pass it quietly."""
    env = DictEnvironment(nondeterministic=True)
    records = _record_trajectory(env, "task-1", 0, ACTIONS, make_step)

    result = replay_prefix(records, len(records), DictEnvironment(nondeterministic=True))

    assert not result.matches


# --- the gate itself -------------------------------------------------------


def test_fidelity_over_many_trajectories_reports_a_pass_rate(env, make_step):
    trajectories = [
        _record_trajectory(DictEnvironment(), f"task-{i}", 0, ACTIONS, make_step)
        for i in range(20)
    ]

    report = verify_fidelity(trajectories, DictEnvironment)

    assert report.total == 20
    assert report.matched == 20
    assert report.rate == 1.0
    assert report.passes_gate


def test_gate_fails_when_any_trajectory_diverges(env, make_step):
    trajectories = [
        _record_trajectory(DictEnvironment(), f"task-{i}", 0, ACTIONS, make_step)
        for i in range(20)
    ]
    trajectories[7][-1] = make_step(
        step=len(ACTIONS) - 1,
        parsed_action=ACTIONS[-1],
        tool_name="delete",
        tool_args={"raw": ACTIONS[-1]},
        tool_result="ok",
        env_state_hash="0" * 64,
    )

    report = verify_fidelity(trajectories, DictEnvironment)

    assert report.matched == 19
    assert not report.passes_gate
    assert 7 in report.diverged_trajectories


def test_gate_threshold_is_total_fidelity_not_a_tolerance(env, make_step):
    """Audit Phase 1 says 100% on the validation set; 19/20 is a failure."""
    report = FidelityReport(total=20, matched=19, diverged_trajectories=(3,))

    assert not report.passes_gate
