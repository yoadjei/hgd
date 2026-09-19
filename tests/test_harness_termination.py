"""Termination against a real environment.

The synthetic environment uses a sentinel action (`done()`) that must NOT be
executed — `DictEnvironment` would answer "unknown operation" and the tier-1
labeller would record a spurious tool error on the final step of every clean run.

AppWorld is the opposite: the agent signals completion by calling
`apis.supervisor.complete_task()`, which *must* execute because it is how the
environment learns the episode is over. Exact string matching also fails there,
because the action is a code block that contains the call rather than equals it.

So termination has two paths, and both are pinned here. Getting this wrong does
not crash: it silently runs every episode to the 3*H* step limit, inflating cost
and putting a step-limit event on trajectories that had actually finished.
"""

import pytest

from hgd.env import DictEnvironment
from hgd.harness import HarnessConfig, run_episode
from hgd.model import ScriptedModel
from hgd.parsing import ActionFormat
from hgd.rule_labeller import TOOL_ERROR, RuleConfig, label_trajectory


def code(body: str) -> str:
    return f"```python\n{body}\n```"


class CompletionEnvironment(DictEnvironment):
    """A DictEnvironment that also reports task completion, as AppWorld does."""

    def __init__(self):
        super().__init__()
        self._done = False

    def execute(self, action: str) -> str:
        if "complete_task" in action:
            self._done = True
            return "Task marked complete."
        return super().execute(action)

    def task_completed(self) -> bool:
        return self._done


@pytest.fixture
def config():
    return HarnessConfig(
        action_format=ActionFormat.CODE,
        step_limit_multiplier=3,
        terminal_actions=("done()",),
        system_prompt="You are an agent.",
    )


# --- sentinel path (synthetic) --------------------------------------------


def test_sentinel_terminal_action_is_not_executed(config):
    """`done()` is a harness sentinel; executing it would be an unknown operation."""
    env = DictEnvironment()
    model = ScriptedModel([code("store(a, 1)"), code("done()")])

    records = run_episode(
        task_id="t", model=model, env=env, seed=0, h_star=5, config=config
    )

    assert len(records) == 2
    assert records[-1].tool_result is None


def test_sentinel_termination_leaves_no_tool_error(config):
    env = DictEnvironment()
    model = ScriptedModel([code("store(a, 1)"), code("done()")])
    records = run_episode(
        task_id="t", model=model, env=env, seed=0, h_star=5, config=config
    )

    labels = label_trajectory(
        records,
        RuleConfig(known_tools=frozenset({"store", "done"}),
                   tool_required_args={}, step_limit=15),
    )

    assert all(TOOL_ERROR not in row for row in labels)


# --- environment-signalled path (AppWorld) --------------------------------


def test_environment_reported_completion_ends_the_episode(config):
    env = CompletionEnvironment()
    model = ScriptedModel(
        [code("store(a, 1)"), code("apis.supervisor.complete_task()")] +
        [code("store(b, 2)")] * 20
    )

    records = run_episode(
        task_id="t", model=model, env=env, seed=0, h_star=5, config=config
    )

    assert len(records) == 2


def test_the_completion_action_is_executed_not_skipped(config):
    """AppWorld learns the episode is over only if the call actually runs."""
    env = CompletionEnvironment()
    model = ScriptedModel([code("apis.supervisor.complete_task()")])

    records = run_episode(
        task_id="t", model=model, env=env, seed=0, h_star=5, config=config
    )

    assert records[-1].tool_result == "Task marked complete."
    assert env.task_completed()


def test_an_environment_without_completion_reporting_still_works(config):
    """DictEnvironment has no task_completed; the harness must not require it."""
    env = DictEnvironment()
    model = ScriptedModel([code("store(a, 1)")] * 3)

    records = run_episode(
        task_id="t", model=model, env=env, seed=0, h_star=1, config=config
    )

    assert len(records) == 3


def test_episode_without_completion_runs_to_the_step_limit(config):
    env = CompletionEnvironment()
    model = ScriptedModel([code("store(a, 1)")] * 100)

    records = run_episode(
        task_id="t", model=model, env=env, seed=0, h_star=4, config=config
    )

    assert len(records) == 12
