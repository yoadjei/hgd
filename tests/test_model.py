"""Model interface tests.

`ScriptedModel` is exercised end to end by the harness tests; this file pins the
two behaviours those tests rely on without stating: the scripted model falls
silent rather than raising, and the shared token proxy is sane.
"""

from hgd.model import ScriptedModel, approx_tokens


def test_scripted_model_falls_silent_after_its_last_turn():
    model = ScriptedModel(["one"])

    model.generate([])
    response = model.generate([])

    assert response.text == ""
    assert response.tokens_out == 0


def test_approx_tokens_is_zero_for_empty_and_at_least_one_otherwise():
    assert approx_tokens("") == 0
    assert approx_tokens("ab") == 1
    assert approx_tokens("a" * 40) == 10
