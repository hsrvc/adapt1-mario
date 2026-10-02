"""findings #34 (2026-09-24): an empty domain abstains with 0.0 for every macro; the fallback
must not read that tie as "noop is best" (it stood at x=40 for 60 decisions) but keep moving."""

from typesafe_mario.adapt1_policy import Adapt1Policy
from typesafe_mario.macros import Macro


def _policy():
    return Adapt1Policy(client=None, domain_id="d", learn=False)


MACROS = tuple(m for m in Macro if m is not Macro.JUMP)  # noop first, as the domain lists them


def test_all_zero_diagnostics_fall_through_to_forward():
    diag = {
        m.value: {"selection_expected_reward": 0.0, "calibrated_contextual_reward": 0.0}
        for m in MACROS
    }
    scores = [{"policy": m.value, "score": 0.0} for m in MACROS]
    assert _policy()._best_effort(MACROS, scores, diag) == "right_run"


def test_an_informative_vector_still_wins():
    diag = {m.value: {"selection_expected_reward": 0.0} for m in MACROS}
    diag["right_run_jump_full"]["selection_expected_reward"] = 0.01
    assert _policy()._best_effort(MACROS, [], diag) == "right_run_jump_full"
    scores = [{"policy": "left", "score": 0.3}, {"policy": "noop", "score": 0.1}]
    assert _policy()._best_effort(MACROS, scores, None) == "left"
