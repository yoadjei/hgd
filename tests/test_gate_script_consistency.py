"""The gate script duplicates the digests; this asserts the copies agree.

`notebooks/phase1_gate.py` is deliberately self-contained so it can be pasted
into a Kaggle cell without a repository checkout. The cost of that is two copies
of the digest functions. A silent divergence between them would mean the gate
measured something other than what the pipeline uses — the worst kind of failure,
because it would produce a confident number about the wrong thing.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from hgd.appworld_env import evaluation_digest as package_digest

GATE_SCRIPT = Path(__file__).resolve().parent.parent / "notebooks" / "phase1_gate.py"


@pytest.fixture(scope="module")
def gate_module():
    spec = importlib.util.spec_from_file_location("phase1_gate", GATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEvaluation:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class FakeWorld:
    def __init__(self, payload):
        self._payload = payload

    def evaluate(self):
        return FakeEvaluation(self._payload)


PAYLOADS = [
    {"success": False, "passes": ["a"], "failures": ["b"]},
    {"success": True, "passes": ["a", "b"], "failures": []},
    {"success": False, "passes": [], "failures": ["x", "y", "z"]},
    {"success": False},
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_script_digest_matches_the_package_digest(gate_module, payload):
    world = FakeWorld(payload)

    assert gate_module.evaluation_digest(world) == package_digest(world)


def test_both_digests_ignore_ordering_of_the_outcome_lists(gate_module):
    one = FakeWorld({"passes": ["a", "b"], "failures": ["x"]})
    other = FakeWorld({"passes": ["b", "a"], "failures": ["x"]})

    assert gate_module.evaluation_digest(one) == gate_module.evaluation_digest(other)
    assert package_digest(one) == package_digest(other)


def test_both_digests_separate_different_states(gate_module):
    passing = FakeWorld({"passes": ["a", "b"], "failures": []})
    failing = FakeWorld({"passes": ["a"], "failures": ["b"]})

    assert gate_module.evaluation_digest(passing) != gate_module.evaluation_digest(failing)


def test_the_script_imports_nothing_from_the_package():
    """If it grows an `hgd` import it stops being pasteable, and this catches it.

    Parsed rather than grepped: the file legitimately *mentions* hgd in prose,
    saying where the duplicated code came from. Only real import statements count.
    """
    tree = ast.parse(GATE_SCRIPT.read_text(encoding="utf-8"))
    imported: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not [name for name in imported if name.split(".")[0] == "hgd"]
