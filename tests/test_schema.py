"""Schema tests.

The acceptance criterion for audit task T1.1 is "every step logs all schema
fields non-null". These tests make that criterion executable: the field set is
pinned against audit section 20, and a record missing any of it is rejected
before it can reach a log file.
"""

import json

import pytest

from hgd.schema import SCHEMA_FIELDS, StepRecord, SchemaError


def test_step_record_serializes_exactly_the_audit_field_set(complete_step):
    """A serialized step carries every audit section 20 field and nothing else."""
    payload = json.loads(complete_step.to_json())

    assert set(payload) == set(SCHEMA_FIELDS)


def test_record_missing_a_required_field_is_rejected(complete_step_kwargs):
    """T1.1 fails the batch if any required field is absent."""
    del complete_step_kwargs["env_state_hash"]

    with pytest.raises(SchemaError, match="env_state_hash"):
        StepRecord(**complete_step_kwargs)


def test_null_in_a_non_nullable_field_is_rejected(complete_step_kwargs):
    """Presence is not enough: the criterion says non-null."""
    complete_step_kwargs["prompt_hash"] = None

    with pytest.raises(SchemaError, match="prompt_hash"):
        StepRecord(**complete_step_kwargs)


def test_nullable_fields_accept_none(complete_step_kwargs):
    """A factual root run has no parent and no branch step; that is not an error."""
    complete_step_kwargs["parent_run_id"] = None
    complete_step_kwargs["branch_step"] = None

    record = StepRecord(**complete_step_kwargs)

    assert record.parent_run_id is None


def test_unknown_intervention_branch_is_rejected(complete_step_kwargs):
    """Branch identity drives the whole decomposition; typos must not survive."""
    complete_step_kwargs["intervention_branch"] = "history-scrub"

    with pytest.raises(SchemaError, match="intervention_branch"):
        StepRecord(**complete_step_kwargs)
