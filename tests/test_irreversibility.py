"""Irreversible deletion — audit task T2.2.

T2.2's acceptance is "pi_hat approx 1 for delete, approx 0 for read errors, CIs
disjoint". The first Phase 2 run returned pi = 0.000 at every position: the
planted `delete` was fully recoverable, because the policy simply stored the key
again a few steps later.

That is a defect in the *plant*, not in the estimator — but it matters, because
an environment with no genuinely irreversible operation cannot validate the
irrecoverability estimator at all, and theta_irrecov would be unidentifiable for
reasons that have nothing to do with agents.

Real irreversibility means the state cannot be restored by any later action.
A tombstone models that: a deleted key stays deleted.
"""

from hgd.env import DictEnvironment


def test_ordinary_delete_is_recoverable_by_storing_again():
    """The default remains recoverable; irreversibility is opt-in."""
    env = DictEnvironment()
    env.reset("t", 0)
    env.execute("store(a, 1)")
    env.execute("delete(a)")

    env.execute("store(a, 1)")

    assert env.snapshot() == {"a": "1"}


def test_irreversible_delete_cannot_be_undone_by_a_later_store():
    env = DictEnvironment(irreversible_delete=True)
    env.reset("t", 0)
    env.execute("store(a, 1)")
    env.execute("delete(a)")

    env.execute("store(a, 1)")

    assert "a" not in env.snapshot()


def test_irreversible_delete_reports_the_refusal():
    env = DictEnvironment(irreversible_delete=True)
    env.reset("t", 0)
    env.execute("store(a, 1)")
    env.execute("delete(a)")

    result = env.execute("store(a, 1)")

    assert "Error" in result or "Exception" in result


def test_tombstones_do_not_survive_a_reset():
    env = DictEnvironment(irreversible_delete=True)
    env.reset("t", 0)
    env.execute("store(a, 1)")
    env.execute("delete(a)")

    env.reset("t", 0)
    env.execute("store(a, 1)")

    assert env.snapshot() == {"a": "1"}


def test_tombstones_are_part_of_the_state_digest():
    """Otherwise replay could not tell a deleted world from a never-written one."""
    deleted = DictEnvironment(irreversible_delete=True)
    deleted.reset("t", 0)
    deleted.execute("store(a, 1)")
    deleted.execute("delete(a)")

    never = DictEnvironment(irreversible_delete=True)
    never.reset("t", 0)

    assert deleted.state_hash() != never.state_hash()


def test_deleting_an_untouched_key_still_tombstones_it():
    env = DictEnvironment(irreversible_delete=True)
    env.reset("t", 0)

    env.execute("delete(a)")
    env.execute("store(a, 1)")

    assert "a" not in env.snapshot()


def test_other_keys_are_unaffected_by_a_tombstone():
    env = DictEnvironment(irreversible_delete=True)
    env.reset("t", 0)
    env.execute("delete(a)")

    env.execute("store(b, 2)")

    assert env.snapshot() == {"b": "2"}
