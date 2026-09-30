"""Harness tests — audit task T1.1 and section 20.

One ReAct loop, no planner, no memory module, no verifier: those are
interventions, never part of the baseline. The harness's job is to produce logs
that satisfy the schema and that replay reproduces exactly. The final test here
is the one that matters — a trajectory this harness produced must pass the
Phase 1 fidelity gate, or the logging is not capturing what replay needs.
"""

import json

import pytest

from hgd.checkpoints import Checkpoint
from hgd.env import DictEnvironment
from hgd.harness import HarnessConfig, run_episode
from hgd.model import ScriptedModel
from hgd.parsing import ActionFormat
from hgd.replay import replay_prefix
from hgd.schema import NULLABLE_FIELDS, SCHEMA_FIELDS


def code(body: str) -> str:
    return f"```python\n{body}\n```"


@pytest.fixture
def config():
    return HarnessConfig(
        action_format=ActionFormat.CODE,
        step_limit_multiplier=3,
        terminal_actions=("done()",),
        system_prompt="You are an agent.",
    )


@pytest.fixture
def three_step_model():
    return ScriptedModel(
        [code("store(a, 1)"), code("store(b, 2)"), code("done()")]
    )


def test_episode_runs_until_the_terminal_action(three_step_model, config):
    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    assert len(records) == 3
    assert records[-1].parsed_action == "done()"


def test_episode_stops_at_three_times_h_star(config):
    model = ScriptedModel([code("store(a, 1)")] * 100)

    records = run_episode(
        task_id="t1", model=model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    assert len(records) == 12


def test_every_record_satisfies_the_schema(three_step_model, config):
    """T1.1 acceptance: every step logs all schema fields non-null."""
    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    for record in records:
        payload = record.to_dict()
        assert set(payload) == set(SCHEMA_FIELDS)
        for name, value in payload.items():
            if name not in NULLABLE_FIELDS:
                assert value is not None, f"{name} was null at step {record.step}"


def test_each_record_carries_its_normalized_position(three_step_model, config):
    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    assert [r.u for r in records] == [0.0, 0.25, 0.5]


def test_baseline_runs_are_logged_as_the_factual_branch(three_step_model, config):
    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    assert all(r.intervention_branch == "factual" for r in records)
    assert all(r.parent_run_id is None for r in records)


def test_malformed_action_is_logged_and_not_retried(config):
    """Audit section 20 fixes retries at zero: a malformed call is an event."""
    model = ScriptedModel(["```python\nstore(a, 1", code("done()")])
    env = DictEnvironment()

    records = run_episode(
        task_id="t1", model=model, env=env, seed=0, h_star=4, config=config,
    )

    assert records[0].parsed_action is None
    assert records[0].tool_name is None
    assert len(records) == 2
    assert env.snapshot() == {}


def test_text_only_turn_does_not_touch_the_environment(config):
    model = ScriptedModel(["Let me think.", code("done()")])
    env = DictEnvironment()

    records = run_episode(
        task_id="t1", model=model, env=env, seed=0, h_star=4, config=config,
    )

    assert records[0].parsed_action is None
    assert env.snapshot() == {}


def test_environment_state_hash_is_recorded_after_each_action(three_step_model, config):
    env = DictEnvironment()

    records = run_episode(
        task_id="t1", model=three_step_model, env=env, seed=0, h_star=4, config=config,
    )

    assert records[-1].env_state_hash == env.state_hash()
    assert records[0].env_state_hash != records[1].env_state_hash


# --- checkpoints ------------------------------------------------------------


def test_configured_checkpoints_are_evaluated_at_every_step(three_step_model):
    """The survival event needs one boolean series per checkpoint per step.

    A checkpoint can only be read against live environment state, so unlike the
    event and judge labels it cannot be backfilled from the log afterwards. If
    the harness does not record it during the episode, it is gone.
    """
    seen_a = Checkpoint("a_stored", stage=0, predicate=lambda state: "a" in state)
    config = HarnessConfig(
        action_format=ActionFormat.CODE,
        terminal_actions=("done()",),
        checkpoints=(seen_a,),
    )

    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    series = [
        next(r["passed"] for r in record.checkpoint_results
             if r["checkpoint_id"] == "a_stored")
        for record in records
    ]

    # store(a, 1) lands on the first step and nothing removes it
    assert series == [True, True, True]


def test_a_record_carrying_checkpoints_still_serialises(three_step_model):
    """The gap that let a TypeError through: no test both ran checkpoints and
    wrote the log, so storing the dataclass looked fine until the first write."""
    config = HarnessConfig(
        action_format=ActionFormat.CODE,
        terminal_actions=("done()",),
        checkpoints=(Checkpoint("a", stage=0, predicate=lambda state: "a" in state),),
    )

    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    restored = json.loads(records[0].to_json())

    assert restored["checkpoint_results"] == [
        {"checkpoint_id": "a", "stage": 0, "passed": True, "error": None}
    ]


def test_no_checkpoints_configured_means_no_evaluation():
    """Every checkpoint pass costs an extra evaluate() call, which on appworld
    runs the task's unit tests. Runs that do not ask for checkpoints pay nothing."""
    calls = []

    class CountingEnvironment(DictEnvironment):
        def snapshot(self):
            calls.append(1)
            return super().snapshot()

    run_episode(
        task_id="t1", model=ScriptedModel([code("done()")]),
        env=CountingEnvironment(), seed=0, h_star=4,
        config=HarnessConfig(action_format=ActionFormat.CODE,
                             terminal_actions=("done()",)),
    )

    assert calls == []


def test_a_broken_predicate_is_recorded_as_an_error_not_lost(three_step_model):
    """A predicate that raises is logged with its error text. Without that, a
    broken check is indistinguishable from a genuine state failure, and tier-2
    labels are the competing-risk fallback if judge agreement fails."""
    def explode(_state):
        raise RuntimeError("predicate is wrong")

    config = HarnessConfig(
        action_format=ActionFormat.CODE,
        terminal_actions=("done()",),
        checkpoints=(Checkpoint("broken", stage=0, predicate=explode),),
    )

    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    result = records[0].checkpoint_results[0]
    assert result["passed"] is False
    assert "predicate is wrong" in result["error"]


def test_token_counts_and_latency_are_logged(three_step_model, config):
    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    assert all(r.tokens_out > 0 for r in records)
    assert all(r.latency_s >= 0.0 for r in records)


def test_run_id_is_shared_across_the_episode_and_steps_are_sequential(
    three_step_model, config
):
    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    assert len({r.run_id for r in records}) == 1
    assert [r.step for r in records] == [0, 1, 2]


def test_two_runs_with_the_same_seed_and_scripted_model_agree(config):
    def go():
        return run_episode(
            task_id="t1",
            model=ScriptedModel([code("store(a, 1)"), code("done()")]),
            env=DictEnvironment(), seed=0, h_star=4, config=config,
        )

    first, second = go(), go()

    assert [r.env_state_hash for r in first] == [r.env_state_hash for r in second]


def test_a_harness_trajectory_passes_the_replay_fidelity_gate(three_step_model, config):
    """The point of the whole phase: what the harness logs, replay reproduces."""
    records = run_episode(
        task_id="t1", model=three_step_model, env=DictEnvironment(),
        seed=0, h_star=4, config=config,
    )

    result = replay_prefix(records, len(records), DictEnvironment())

    assert result.matches
    assert result.diverged_at is None
