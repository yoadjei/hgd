import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def complete_step_kwargs():
    """Every audit section 20 field, populated. Tests mutate a copy to break it."""
    return {
        "run_id": "run-0001",
        "task_id": "appworld/train/82e2fac_1",
        "seed": 0,
        "model": "Qwen3-8B",
        "step": 3,
        "u": 0.25,
        "prompt_hash": "a" * 64,
        "raw_output": "I will list the playlists.\n```python\napis.spotify.playlists()\n```",
        "parsed_action": "apis.spotify.playlists()",
        "tool_name": "spotify.playlists",
        "tool_args": {"access_token": "tok"},
        "tool_result": "[{'id': 1, 'name': 'Chill'}]",
        "tool_result_hash": "b" * 64,
        "env_state_hash": "c" * 64,
        "checkpoint_results": [],
        "event_labels": [],
        "judge_labels": [],
        "tokens_in": 1024,
        "tokens_out": 48,
        "latency_s": 1.37,
        "intervention_branch": "factual",
        "parent_run_id": None,
        "branch_step": None,
    }


@pytest.fixture
def complete_step(complete_step_kwargs):
    from hgd.schema import StepRecord

    return StepRecord(**complete_step_kwargs)


@pytest.fixture
def make_step(complete_step_kwargs):
    """Factory for step records: valid by default, overridable field by field."""
    from hgd.schema import StepRecord

    def _make(**overrides):
        fields = dict(complete_step_kwargs)
        fields.update(overrides)
        return StepRecord(**fields)

    return _make
