"""AppWorld adapter tests.

These test the *adapter's* translation logic — lifecycle, delegation, terminal
detection, digest strategy — against a fake that implements the AppWorld API as
documented (`execute`, `task_completed`, `evaluate`, `save_state`/`load_state`,
`task.ground_truth`, `close`). They do not test AppWorld itself, and they cannot:
AppWorld is not installable under this interpreter.

The residual risk is that the documented API differs from the real one. That is
discharged by `notebooks/phase1_gate.py`, which probes the real package on Kaggle
before the gate runs, rather than by asserting harder against a fake here.

The digest is injected because AppWorld exposes no state hash, and the fidelity
gate is meaningless without one. Which strategy is correct is an empirical
question the probe answers.
"""

import pytest

from hgd.appworld_env import AppWorldEnvironment, database_digest, evaluation_digest


class FakeTask:
    def __init__(self):
        self.instruction = "Like the song."
        self.ground_truth = {"api_calls": ["spotify.login", "spotify.like"]}


class FakeEvaluation:
    """Mirrors the real payload verified on 2026-09-06: keys are
    ['difficulty', 'failures', 'num_tests', 'passes', 'success'].
    Note `failures`, not `fails` — the original adapter read the wrong key."""

    def __init__(self, passes, failures):
        self._payload = {
            "success": not failures,
            "passes": passes,
            "failures": failures,
            "num_tests": len(passes) + len(failures),
            "difficulty": 1,
        }

    def to_dict(self):
        return dict(self._payload)


def as_dict_entries(names):
    """The shape the real package returns once a task has completed."""
    return [{"name": name, "score": 1} for name in names]


class FakeWorld:
    """Implements the documented AppWorld surface, and nothing more.

    ``outcome_shape`` decides whether the pass and fail vectors hold strings or
    dicts. Both occur: the real package returns strings until a task completes and
    dicts afterwards. Every adapter test runs under both, because a fake that only
    ever returned strings is precisely why a ``sorted()`` crash on dict entries
    shipped into the adapter and went unnoticed until a real run hit it.
    """

    def __init__(self, task_id, outcome_shape="strings", **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.task = FakeTask()
        self.executed: list[str] = []
        self.closed = False
        self.completed = False
        self._shape = outcome_shape
        self._passes = ["login_ok"]
        self._fails = ["song_liked"]
        self._states: dict[str, tuple] = {}

    def _shaped(self, names):
        return as_dict_entries(names) if self._shape == "dicts" else list(names)

    def execute(self, code):
        self.executed.append(code)
        if "complete_task" in code:
            self.completed = True
        if "like_song" in code:
            self._passes, self._fails = ["login_ok", "song_liked"], []
        return f"output of {code}"

    def task_completed(self):
        return self.completed

    def evaluate(self):
        return FakeEvaluation(self._shaped(self._passes), self._shaped(self._fails))

    def save_state(self):
        state_id = f"s{len(self._states)}"
        self._states[state_id] = (list(self._passes), list(self._fails))
        return state_id

    def load_state(self, state_id):
        self._passes, self._fails = (list(x) for x in self._states[state_id])

    def close(self):
        self.closed = True


@pytest.fixture(params=["strings", "dicts"], ids=["string_outcomes", "dict_outcomes"])
def env(request):
    """The adapter, exercised against both outcome shapes the real package returns."""
    shape = request.param
    return AppWorldEnvironment(
        world_factory=lambda task_id, **kw: FakeWorld(task_id, outcome_shape=shape, **kw),
        state_digest=evaluation_digest,
        experiment_name="test_exp",
    )


# --- lifecycle -------------------------------------------------------------


def test_reset_constructs_a_world_for_the_task(env):
    env.reset("train/task_1", seed=0)

    assert env.world.task_id == "train/task_1"


def test_reset_closes_the_previous_world(env):
    env.reset("train/task_1", seed=0)
    first = env.world

    env.reset("train/task_2", seed=0)

    assert first.closed


def test_reset_requests_full_ground_truth_so_h_star_is_available(env):
    env.reset("train/task_1", seed=0)

    assert env.world.kwargs["ground_truth_mode"] == "full"


def test_close_releases_the_world(env):
    env.reset("train/task_1", seed=0)
    world = env.world

    env.close()

    assert world.closed


# --- delegation ------------------------------------------------------------


def test_execute_delegates_and_returns_the_shell_output(env):
    env.reset("train/task_1", seed=0)

    result = env.execute("apis.spotify.login()")

    assert result == "output of apis.spotify.login()"
    assert env.world.executed == ["apis.spotify.login()"]


def test_execute_before_reset_is_an_explicit_error(env):
    with pytest.raises(RuntimeError, match="reset"):
        env.execute("apis.spotify.login()")


def test_task_completed_reflects_the_supervisor_call(env):
    env.reset("train/task_1", seed=0)
    assert not env.task_completed()

    env.execute("apis.supervisor.complete_task()")

    assert env.task_completed()


# --- state digest ----------------------------------------------------------


def test_state_hash_changes_when_task_relevant_state_changes(env):
    env.reset("train/task_1", seed=0)
    before = env.state_hash()

    env.execute("apis.spotify.like_song(song_id=1)")

    assert env.state_hash() != before


def test_state_hash_is_stable_for_an_unchanged_state(env):
    env.reset("train/task_1", seed=0)

    assert env.state_hash() == env.state_hash()


def test_identical_action_sequences_produce_identical_digests(env):
    def digest_after(actions):
        e = AppWorldEnvironment(
            world_factory=lambda task_id, **kw: FakeWorld(task_id, **kw),
            state_digest=evaluation_digest,
            experiment_name="test_exp",
        )
        e.reset("train/task_1", seed=0)
        for action in actions:
            e.execute(action)
        return e.state_hash()

    actions = ["apis.spotify.login()", "apis.spotify.like_song(song_id=1)"]

    assert digest_after(actions) == digest_after(actions)


def test_snapshot_exposes_evaluation_state_for_checkpoint_predicates(env):
    env.reset("train/task_1", seed=0)

    snapshot = env.snapshot()

    # names, whichever shape the package returned. a predicate written this way is
    # the point: on raw dict entries it silently returned False.
    assert "song_liked" in snapshot["failures"]
    assert "login_ok" in snapshot["passes"]


def test_digest_reads_the_failure_vector_not_a_misspelled_key(env):
    """Regression: the first version read `fails`, always got None, and digested
    only the pass vector — so two states differing solely in which tests failed
    would have hashed identically."""
    env.reset("train/task_1", seed=0)
    world = env.world

    world._passes, world._fails = ["a"], ["b"]
    one = evaluation_digest(world)
    world._passes, world._fails = ["a"], ["c"]
    other = evaluation_digest(world)

    assert one != other


def test_digest_handles_dict_outcome_entries(env):
    """Regression: the gate died with "'<' not supported between dicts".

    The pass and fail vectors are not always lists of strings. Once a task
    completes AppWorld returns dicts, and sorting them raises, so every real
    run would have crashed at the third digest of the first task.
    """
    env.reset("train/task_1", seed=0)
    world = env.world

    world._passes = [{"name": "answers match", "score": 1}]
    world._fails = [{"name": "no model changes", "score": 0}]

    assert evaluation_digest(world) == evaluation_digest(world)


def test_dict_outcome_order_is_still_ignored(env):
    """Two states differing only in entry order remain identical."""
    env.reset("train/task_1", seed=0)
    world = env.world
    one, other = {"name": "a"}, {"name": "b"}

    world._passes, world._fails = [one, other], []
    forward = evaluation_digest(world)
    world._passes = [other, one]

    assert evaluation_digest(world) == forward


def test_dict_outcomes_differing_in_content_still_separate(env):
    env.reset("train/task_1", seed=0)
    world = env.world

    world._passes, world._fails = [{"name": "a"}], []
    one = evaluation_digest(world)
    world._passes = [{"name": "b"}]

    assert evaluation_digest(world) != one


def test_database_digest_tracks_the_bytes_on_disk(tmp_path):
    """The exhaustive alternative: any change to a database file is a divergence."""
    database = tmp_path / "apps" / "spotify.db"
    database.parent.mkdir()
    database.write_bytes(b"v1")
    digest = database_digest(tmp_path)

    before = digest(None)
    database.write_bytes(b"v2")

    assert digest(None) != before
    assert digest(None) == digest(None)


# --- state save and restore ------------------------------------------------


def test_saved_state_can_be_restored(env):
    env.reset("train/task_1", seed=0)
    marker = env.save_state()
    env.execute("apis.spotify.like_song(song_id=1)")
    changed = env.state_hash()

    env.load_state(marker)

    assert env.state_hash() != changed


# --- horizon ---------------------------------------------------------------


def test_intrinsic_horizon_is_read_from_the_task_ground_truth(env):
    env.reset("train/task_1", seed=0)

    assert env.intrinsic_horizon() == 2
