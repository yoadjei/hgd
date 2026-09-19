"""Trajectory persistence — audit section 35.

"Any result that cannot be reconstructed from recorded data should be treated as
a reproducibility failure." Logs are written append-only, one JSON object per
line, so a session that dies at hour nine of a Kaggle run keeps everything it
had already produced. That matters more than it sounds: a crash that loses a
whole batch costs a week of quota.
"""

import json

import pytest

from hgd.logging import RunWriter, read_trajectories
from hgd.schema import SCHEMA_FIELDS


def test_records_are_written_one_json_object_per_line(tmp_path, make_step):
    path = tmp_path / "runs.jsonl"

    with RunWriter(path) as writer:
        writer.write([make_step(step=0), make_step(step=1)])

    assert len(path.read_text().strip().splitlines()) == 2


def test_written_records_round_trip(tmp_path, make_step):
    path = tmp_path / "runs.jsonl"
    original = [make_step(step=0), make_step(step=1, tool_name="venmo.send_money")]

    with RunWriter(path) as writer:
        writer.write(original)

    recovered = list(read_trajectories(path))

    assert len(recovered) == 1
    assert [r.to_dict() for r in recovered[0]] == [r.to_dict() for r in original]


def test_multiple_trajectories_are_grouped_by_run_id(tmp_path, make_step):
    path = tmp_path / "runs.jsonl"

    with RunWriter(path) as writer:
        writer.write([make_step(step=0, run_id="a"), make_step(step=1, run_id="a")])
        writer.write([make_step(step=0, run_id="b")])

    recovered = list(read_trajectories(path))

    assert [len(t) for t in recovered] == [2, 1]


def test_writing_appends_rather_than_truncating(tmp_path, make_step):
    path = tmp_path / "runs.jsonl"

    with RunWriter(path) as writer:
        writer.write([make_step(step=0, run_id="a")])
    with RunWriter(path) as writer:
        writer.write([make_step(step=0, run_id="b")])

    assert len(list(read_trajectories(path))) == 2


def test_records_are_flushed_so_a_crash_keeps_what_finished(tmp_path, make_step):
    """Nine hours into a session, unflushed buffers are lost quota."""
    path = tmp_path / "runs.jsonl"
    writer = RunWriter(path)
    writer.__enter__()

    writer.write([make_step(step=0)])

    assert path.read_text().strip() != ""
    writer.__exit__(None, None, None)


def test_parent_directory_is_created(tmp_path, make_step):
    path = tmp_path / "nested" / "deep" / "runs.jsonl"

    with RunWriter(path) as writer:
        writer.write([make_step(step=0)])

    assert path.exists()


def test_reading_a_missing_file_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_trajectories(tmp_path / "absent.jsonl"))


def test_a_corrupt_line_names_its_line_number(tmp_path, make_step):
    """A truncated final line from a killed session must be diagnosable."""
    path = tmp_path / "runs.jsonl"
    with RunWriter(path) as writer:
        writer.write([make_step(step=0)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "truncated"\n')

    with pytest.raises(ValueError, match="line 2"):
        list(read_trajectories(path))


def test_written_lines_are_valid_json_with_the_full_schema(tmp_path, make_step):
    path = tmp_path / "runs.jsonl"
    with RunWriter(path) as writer:
        writer.write([make_step(step=0)])

    payload = json.loads(path.read_text().splitlines()[0])

    assert set(payload) == set(SCHEMA_FIELDS)
