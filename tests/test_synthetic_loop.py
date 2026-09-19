"""End-to-end tests for the planted policy inside the real harness.

`test_synthetic.py` feeds the policy handcrafted transcripts, which proves the
mechanism arithmetic but says nothing about whether a real run ever reaches the
conditions that trigger it. These tests close that gap: the planted error action
must actually be rejected by the environment, or the self-conditioning term can
never fire and Phase 2 would validate an estimator against a mechanism that was
silently absent.
"""

import pytest

from hgd.env import DictEnvironment
from hgd.harness import HarnessConfig, run_episode
from hgd.interventions import scrub_history
from hgd.parsing import ActionFormat
from hgd.synthetic import PlantedPolicy, error_turns_in_history


@pytest.fixture
def config():
    return HarnessConfig(
        action_format=ActionFormat.CODE,
        step_limit_multiplier=3,
        system_prompt="You are an agent.",
    )


def _transcript(records):
    """Rebuild the message history the policy would have seen."""
    messages = []
    for record in records:
        messages.append({"role": "assistant", "content": record.raw_output})
        if record.tool_result is not None:
            messages.append({"role": "user", "content": record.tool_result})
    return messages


def test_planted_error_action_is_actually_rejected_by_the_environment():
    """The bad action must produce an error observation, not a silent success."""
    env = DictEnvironment()
    env.reset("t", 0)
    always_wrong = PlantedPolicy(
        base_error_rate=1.0, self_conditioning=0.0, staleness=0.0, seed=0
    )

    action = always_wrong.generate([]).text
    body = action.split("```python\n")[1].split("\n```")[0]
    result = env.execute(body)

    assert "Error" in result or "Exception" in result


def test_a_high_error_run_leaves_error_observations_in_the_transcript(config):
    """Without this, history_scrub has nothing to scrub and theta_self is unidentified."""
    policy = PlantedPolicy(
        base_error_rate=1.0, self_conditioning=0.0, staleness=0.0, seed=0
    )

    records = run_episode(
        task_id="t", model=policy, env=DictEnvironment(), seed=0, h_star=5, config=config
    )

    assert error_turns_in_history(_transcript(records)) > 0


def test_a_clean_run_leaves_no_error_observations(config):
    policy = PlantedPolicy(
        base_error_rate=0.0, self_conditioning=0.0, staleness=0.0, seed=0
    )

    records = run_episode(
        task_id="t", model=policy, env=DictEnvironment(), seed=0, h_star=5, config=config
    )

    assert error_turns_in_history(_transcript(records)) == 0


def test_scrubbing_a_real_transcript_removes_its_error_observations(config):
    """The intervention must work on transcripts the harness actually produces."""
    policy = PlantedPolicy(
        base_error_rate=1.0, self_conditioning=0.0, staleness=0.0, seed=0
    )
    records = run_episode(
        task_id="t", model=policy, env=DictEnvironment(), seed=0, h_star=5, config=config
    )
    transcript = _transcript(records)

    scrubbed = scrub_history(transcript, error_steps=set(range(len(records))))

    assert error_turns_in_history(transcript) > 0
    assert error_turns_in_history(scrubbed) == 0


def test_self_conditioning_raises_the_error_rate_over_a_real_run(config):
    """The planted mechanism must be visible end to end, not only in unit arithmetic."""
    def error_count(self_conditioning):
        policy = PlantedPolicy(
            base_error_rate=0.15,
            self_conditioning=self_conditioning,
            staleness=0.0,
            seed=5,
        )
        records = run_episode(
            task_id="t", model=policy, env=DictEnvironment(), seed=0, h_star=20,
            config=config,
        )
        return error_turns_in_history(_transcript(records))

    assert error_count(0.7) > error_count(0.0)
