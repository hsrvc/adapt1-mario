"""`refine` frontier exploration (frontier.py): offline, no network, 0 Records.

The properties the live run relies on: a trial game replays the frozen game's moves exactly up to ``j``, executes
the forced edit at ``j``, and sends exactly one feedback (to the corrector, never the base); the frontier is the
last decisions that made progress; the trial list holds each distinct edit once.
"""

from __future__ import annotations

from typesafe_mario.adapt1_policy import build_domain_config
from typesafe_mario.frontier import (
    CachedBase,
    EpochCache,
    FrontierConfig,
    Trial,
    first_divergence,
    frontier_decisions,
    play,
    trials_at,
)
from typesafe_mario.macros import APEX_NAMES
from typesafe_mario.offline_client import OfflineAdapt1Client
from typesafe_mario.online_duo import RecordsBudget
from typesafe_mario.refinement import (
    BASE_APEX_FEATURES,
    BASE_MACROS,
    BASE_TAKEOFF_FEATURES,
    GuardedClient,
    Pair,
    refiner_domain_config,
)

BASE = Pair("t-base", "a-base")
REF = Pair("t-ref", "a-ref")


def _clients():
    inner = OfflineAdapt1Client()
    inner.create_domain(
        build_domain_config(
            BASE.takeoff,
            feature_names=BASE_TAKEOFF_FEATURES,
            macros=[m.value for m in BASE_MACROS],
            model_type="extra_trees",
        )
    )
    inner.create_domain(
        build_domain_config(
            BASE.apex,
            feature_names=BASE_APEX_FEATURES,
            macros=list(APEX_NAMES),
            model_type="extra_trees",
        )
    )
    for dom, kind in ((REF.takeoff, "takeoff"), (REF.apex, "apex")):
        inner.create_domain(
            refiner_domain_config(dom, kind, credible_dominance=True, reward_max=0.0076)
        )
    guarded = GuardedClient(inner, {BASE.takeoff, BASE.apex})
    live = CachedBase(guarded, (BASE.takeoff, BASE.apex))
    return guarded, {"live": live, "cached": EpochCache(live, (REF.takeoff, REF.apex))}


def test_a_trial_replays_the_frozen_moves_and_sends_one_feedback_to_the_corrector():
    guarded, clients = _clients()
    cfg = FrontierConfig(base=BASE, refiner=REF, max_decisions=40)
    game = play(clients, cfg)
    j = frontier_decisions(game)[0]
    step = game["steps"][j]
    edit = next(t.edit for t in trials_at(step, j) if t.edit not in ("keep_base",))
    res = play(
        clients,
        cfg,
        prefix=game["steps"][:j],
        trial=Trial(j, "takeoff", edit),
        budget=RecordsBudget(5),
    )
    assert [s["executed"] for s in res["steps"][:j]] == [s["executed"] for s in game["steps"][:j]]
    live = [s for s in res["steps"] if not s.get("replayed")]
    assert live[0]["i"] == j and live[0]["trial"] and live[0]["edit"] == edit
    assert live[0]["context"] == step["context"]
    assert res["records"] == 1 and res["feedback"][0]["domain"] == REF.takeoff
    # the feedback must REACH the client (2026-10-04: records were counted while nothing was posted)
    assert guarded.feedback_calls.get(REF.takeoff, 0) == 1
    assert 0.0 <= res["credit"] <= 0.0076 + 1e-9
    assert sum(guarded.feedback_calls.get(d, 0) for d in (BASE.takeoff, BASE.apex)) == 0


def test_a_frozen_game_sends_nothing():
    guarded, clients = _clients()
    play(clients, FrontierConfig(base=BASE, refiner=REF, max_decisions=40), budget=RecordsBudget(5))
    assert sum(guarded.feedback_calls.values()) == 0


def test_the_frontier_is_the_last_decisions_that_made_progress():
    steps = [{"x": x, "x_after": xa} for x, xa in ((0, 10), (10, 20), (20, 30), (30, 30), (30, 30))]
    assert frontier_decisions({"steps": steps, "end": "dead"}, 3) == [2, 1, 0]
    assert frontier_decisions({"steps": steps, "end": "stall"}, 2) == [2, 1]


def test_trials_hold_each_distinct_edit_once():
    step = {"base": "right_run", "executed": "right_run", "apex_executed": None, "apex_base": None}
    labels = [t.edit for t in trials_at(step, 4)]
    assert (
        "keep_base" in labels
        and "override_right_run" not in labels
        and len(labels) == len(set(labels)) == 9
    )
    arc = {
        "base": "right_jump_full",
        "executed": "right_jump_full",
        "apex_executed": "apex_keep",
        "apex_base": "apex_keep",
    }
    apex = [t.edit for t in trials_at(arc, 4) if t.kind == "apex"]
    assert apex == ["keep_base", "override_apex_brake", "override_apex_pull_back"]


def test_a_plan_is_forced_without_feedback_and_credited_only_when_asked():
    guarded, clients = _clients()
    cfg = FrontierConfig(base=BASE, refiner=REF, max_decisions=40)
    game = play(clients, cfg)
    j = frontier_decisions(game)[0]
    plan = [Trial(j, "takeoff", "override_left"), Trial(j + 1, "takeoff", "override_left")]
    last = Trial(j + 2, "takeoff", "override_jump")
    res = play(
        clients, cfg, prefix=game["steps"][:j], plan=plan, trial=last, budget=RecordsBudget(9)
    )
    live = [s for s in res["steps"] if not s.get("replayed")]
    assert [s["executed"] for s in live[:3]] == ["left", "left", "jump"]
    assert res["records"] == 1 and res["credit"] == res["credits"][0]
    assert guarded.feedback_calls.get(REF.takeoff, 0) == 1
    res2 = play(
        clients,
        cfg,
        prefix=game["steps"][:j],
        plan=plan,
        trial=last,
        credit_plan=True,
        budget=RecordsBudget(9),
    )
    assert res2["records"] == 3
    assert guarded.feedback_calls.get(REF.takeoff, 0) == 4


def test_keep_dominance_overrides_only_on_overlap_and_only_above_keep():
    from typesafe_mario.refinement import keep_dominance_pick

    iv = {"keep_base": [0.0, 0.17], "override_a": [0.31, 0.60], "override_b": [0.27, 0.50]}
    assert keep_dominance_pick(iv, "abstained", "credible_intervals_overlap") == "override_a"
    # an OOD (or any other) abstention keeps the base
    assert keep_dominance_pick(iv, "abstained", "out_of_distribution") is None
    # a selected answer is the server's own: the rule does not apply
    assert keep_dominance_pick(iv, "selected", None) is None
    # the best lower bound must clear keep_base's upper bound
    low = {"keep_base": [0.0, 0.35], "override_a": [0.31, 0.60]}
    assert keep_dominance_pick(low, "abstained", "credible_intervals_overlap") is None


def test_a_deployed_game_is_compared_with_the_tested_one():
    """2026-11-16: the tested edit (decision 6, base after it) reached 1591; the corrector fired again at decision 7
    and the deployed game reached 735. The first departure names the move to teach: the tested one."""

    def st(i, x, executed, edit="keep_base", apex=None, apex_edit=None):
        return {
            "i": i,
            "x": x,
            "executed": executed,
            "edit": edit,
            "apex_executed": apex,
            "apex_edit": apex_edit,
        }

    tested = [
        st(5, 231, "right_run_jump_full"),
        st(6, 292, "right", "override_right"),
        st(7, 306, "right_run_jump_full", apex="apex_keep", apex_edit="keep_base"),
    ]
    leak = [*tested[:2], st(7, 306, "right", "override_right")]
    assert first_divergence(leak, tested) == Trial(7, "takeoff", "keep_base")
    assert first_divergence(tested, tested) is None
    in_air = [
        *tested[:2],
        st(
            7,
            306,
            "right_run_jump_full",
            apex="apex_pull_back",
            apex_edit="override_apex_pull_back",
        ),
    ]
    assert first_divergence(in_air, tested) == Trial(7, "apex", "keep_base")
    lost_fix = [tested[0], st(6, 292, "right_run_jump_full")]
    assert first_divergence(lost_fix, tested) == Trial(6, "takeoff", "override_right")
    assert (
        first_divergence([st(5, 230, "right")], tested) is None
    )  # another state: replay drift, not a leak


def test_a_bounce_off_a_wall_is_not_progress():
    from typesafe_mario.frontier import frontier_decisions

    steps = [
        {"x": 0, "x_after": 50},
        {"x": 50, "x_after": 100},
        {"x": 100, "x_after": 93},
        {"x": 93, "x_after": 100},
        {"x": 100, "x_after": 94},
        {"x": 94, "x_after": 100},
    ]
    assert frontier_decisions({"steps": steps}, 2) == [1, 0]
