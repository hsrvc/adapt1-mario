"""`refine` (2026-10-03): base + corrector over the frozen zero-start 1-1 policy.

The two non-negotiables (Rei's handoff §25, design check ``mario/refine-design-check-2026-10-03.md``):

* ``keep_base`` replays the base EXACTLY — as a mapping, and as a whole emulator trajectory: the base alone, the
  base with a corrector that always answers ``keep_base``, one that always abstains, and ``force_keep`` must play
  the same moves to the same pixels;
* the base domains receive ZERO feedback — refused by the client guard before any network call, and absent from
  the contract client's log after a learning episode.
"""

from __future__ import annotations

import hashlib
import json
import uuid

import pytest

from typesafe_mario.adapt1_policy import (
    APEX_V3_FEATURE_NAMES,
    FULL_V3_FEATURE_NAMES,
    Adapt1Policy,
    build_domain_config,
)
from typesafe_mario.macros import APEX_NAMES, ApexChoice, Macro
from typesafe_mario.offline_client import OfflineAdapt1Client
from typesafe_mario.refinement import (
    BASE_APEX,
    BASE_MACROS,
    KEEP_BASE,
    ApexEdit,
    BaseFeedbackError,
    GuardedClient,
    RefinedPolicy,
    TakeoffEdit,
    apex_context,
    apex_refiner_features,
    apply_apex_correction,
    apply_takeoff_correction,
    fingerprint_changed,
    frozen_base,
    interaction_features,
    refiner_domain_config,
    refiner_policy,
    takeoff_context,
    takeoff_edits,
    takeoff_refiner_features,
)

# --- the edit mapping -----------------------------------------------------------------------


@pytest.mark.parametrize("base", list(Macro))
def test_keep_base_is_the_identity_on_every_takeoff_macro(base):
    assert apply_takeoff_correction(base, KEEP_BASE) is base


@pytest.mark.parametrize("base", list(ApexChoice))
def test_keep_base_is_the_identity_on_every_apex_choice(base):
    assert apply_apex_correction(base, KEEP_BASE) is base


def test_overrides_execute_their_named_move_whatever_the_base():
    for base in BASE_MACROS:
        for edit in TakeoffEdit:
            if edit is TakeoffEdit.KEEP_BASE:
                continue
            assert apply_takeoff_correction(base, edit.value).value == edit.value.removeprefix(
                "override_"
            )
    assert apply_takeoff_correction(Macro.RIGHT_RUN, "override_jump") is Macro.JUMP
    for base in BASE_APEX:
        assert apply_apex_correction(base, ApexEdit.OVERRIDE_APEX_BRAKE.value) is ApexChoice.BRAKE
    with pytest.raises(ValueError):
        apply_takeoff_correction(Macro.RIGHT, "right")


def test_the_edit_space_is_keep_plus_every_macro_and_jump_only_when_allowed():
    assert {e.value for e in takeoff_edits()} == {KEEP_BASE} | {
        f"override_{m.value}" for m in Macro
    }
    assert "override_jump" not in {e.value for e in takeoff_edits(allow_jump=False)}
    assert {e.value for e in ApexEdit} == {KEEP_BASE} | {f"override_{a}" for a in APEX_NAMES}


# --- the corrector's context ----------------------------------------------------------------


def test_interactions_are_zero_and_flagged_absent_for_the_sentinel():
    f = {
        "speed_x": 2.5,
        "gap_dist": 999.0,
        "obstacle_dist": 999.0,
        "enemy_dist": 999.0,
        "obstacle_height": 0.0,
        "drop_depth": 3.0,
    }
    ix = interaction_features(f)
    assert ix["gap_present"] == ix["obstacle_present"] == ix["enemy_present"] == 0.0
    assert all(ix[k] == 0.0 for k in ix if k.startswith("ix_"))


def test_interactions_multiply_present_geometry():
    f = {
        "speed_x": 2.0,
        "gap_dist": 3.0,
        "obstacle_dist": 4.0,
        "enemy_dist": 50.0,
        "obstacle_height": 2.0,
        "drop_depth": 1.5,
    }
    ix = interaction_features(f)
    assert ix == {
        "gap_present": 1.0,
        "obstacle_present": 1.0,
        "enemy_present": 1.0,
        "ix_speed_gap": 6.0,
        "ix_speed_obstacle": 8.0,
        "ix_speed_enemy": 100.0,
        "ix_speed_height": 4.0,
        "ix_gap_drop": 4.5,
    }


def test_contexts_carry_exactly_the_declared_feature_names_and_no_position():
    base_t = {n: 0.0 for n in FULL_V3_FEATURE_NAMES}
    ctx = takeoff_context(base_t, "right_run")
    assert tuple(ctx) == takeoff_refiner_features()
    assert (
        ctx["base_is_right_run"] == 1.0
        and sum(v for k, v in ctx.items() if k.startswith("base_is_")) == 1.0
    )
    base_a = {n: 0.0 for n in APEX_V3_FEATURE_NAMES}
    actx = apex_context(base_a, "apex_brake", "right_run_jump_full")
    assert tuple(actx) == apex_refiner_features()
    assert actx["base_apex_is_apex_brake"] == 1.0 and actx["arc_from_right_run_jump_full"] == 1.0
    for names in (takeoff_refiner_features(), apex_refiner_features()):
        assert len(names) == len(set(names))
        assert not {"x", "level_x", "level_screen", "x_pos"} & set(names)


def test_refiner_configs_follow_the_zero2_contract_and_pass_the_offline_client():
    client = OfflineAdapt1Client()
    for kind in ("takeoff", "apex"):
        cfg = refiner_domain_config(f"r-{kind}", kind)
        learning = cfg["learning"]
        assert learning["context"]["max_samples"] == 8192
        assert learning["training"]["model_type"] == "extra_trees"
        assert learning["policy"]["exploration_mode"] == "ucb"
        assert learning["reward"]["components"][0]["max"] == 1.0
        client.create_domain(cfg)
    assert {h["policy"] for h in refiner_domain_config("x", "takeoff")["hypotheses"]} == {
        e.value for e in TakeoffEdit
    }


# --- the guard ------------------------------------------------------------------------------


def test_the_guard_refuses_every_write_to_a_base_domain_before_sending():
    inner = OfflineAdapt1Client()
    guard = GuardedClient(inner, {"base-t"})
    with pytest.raises(BaseFeedbackError):
        guard.feedback("base-t", {})
    with pytest.raises(BaseFeedbackError):
        guard.call("DELETE", "/domains/base-t")
    with pytest.raises(BaseFeedbackError):
        guard.create_domain({"domain_id": "base-t"})
    with pytest.raises(BaseFeedbackError):
        guard.events("base-t", {})
    assert guard.feedback_calls == {}


def test_a_frozen_base_is_required():
    client = OfflineAdapt1Client()
    learning_base = Adapt1Policy(client, "b", feature_names=FULL_V3_FEATURE_NAMES)
    with pytest.raises(ValueError):
        RefinedPolicy("takeoff", learning_base)


def test_fingerprint_change_is_detected():
    a = {"b": {"model_type": "extra_trees", "live_sample_count": 5450, "model_current_version": 7}}
    assert fingerprint_changed(a, a) == []
    assert fingerprint_changed(a, {"b": {**a["b"], "model_current_version": 8}}) == ["b"]


# --- a scripted server: deterministic picks, so trajectories can be compared ----------------


class ScriptedClient(OfflineAdapt1Client):
    """The contract client, but the BASE domains select a macro deterministically from the context (a hash, so the
    picks vary with the state and exercise jumps + apex looks), and the corrector domains answer a fixed edit —
    ``None`` = abstain."""

    def __init__(self, base_ids: tuple[str, str], refiner_answer: dict[str, str | None]) -> None:
        super().__init__()
        self.base_ids = base_ids
        self.refiner_answer = refiner_answer

    def query(self, domain_id, body):
        status, resp = super().query(domain_id, body)
        values = (body.get("context") or {}).get("values")
        if values is None:
            return status, resp
        if domain_id in self.base_ids:
            vocab = sorted(self.domains[domain_id]["policies"])
            h = int(hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(), 16)
            pick = vocab[h % len(vocab)]
            # keep the run moving right most of the time so it reaches jumps and apex looks
            if domain_id == self.base_ids[0] and h % 3:
                pick = ("right_run", "right_run_jump_mid", "right_run_jump_full")[
                    h % 3 - 1 if h % 3 else 0
                ]
        else:
            pick = self.refiner_answer.get(domain_id)
        if pick is not None:
            resp = {
                **resp,
                "selection": {
                    "status": "selected",
                    "policy": pick,
                    "decision_id": resp["decision_id"],
                },
            }
        return status, resp


BASE = ("base-t", "base-a")
REF = ("ref-t", "ref-a")


def _scripted(refiner_answer):
    client = ScriptedClient(BASE, refiner_answer)
    client.create_domain(
        build_domain_config(
            BASE[0], feature_names=FULL_V3_FEATURE_NAMES, macros=[m.value for m in BASE_MACROS]
        )
    )
    client.create_domain(
        build_domain_config(BASE[1], feature_names=APEX_V3_FEATURE_NAMES, macros=list(APEX_NAMES))
    )
    client.create_domain(refiner_domain_config(REF[0], "takeoff"))
    client.create_domain(refiner_domain_config(REF[1], "apex"))
    return client


pytest.importorskip("gym_super_mario_bros")
pytest.importorskip("PIL")

from typesafe_mario.online_duo import (
    CreditConfig,
    DelayedCredit,
    RecordsBudget,
    play_episode,
)
from typesafe_mario.record import record_episode
from typesafe_mario.runner import create_mario_env

DECISIONS = 30


def _frozen_trajectory(tmp_path, mode, refiner_answer=None):
    client = GuardedClient(_scripted(refiner_answer or {}), BASE)
    bt = frozen_base(client, BASE[0], FULL_V3_FEATURE_NAMES, run_id="t")
    ba = frozen_base(client, BASE[1], APEX_V3_FEATURE_NAMES, run_id="t")
    if mode == "raw":  # the historical base-only path: the bare policies, no wrapper
        takeoff, apex = bt, ba
    else:
        rt = rf = None
        if mode == "refiner":
            rt = refiner_policy(client, REF[0], "takeoff", learn=False, run_id="t")
            rf = refiner_policy(client, REF[1], "apex", learn=False, run_id="t")
        takeoff = RefinedPolicy("takeoff", bt, rt, force_keep=(mode == "force_keep"))
        apex = RefinedPolicy("apex", ba, rf, force_keep=(mode == "force_keep"), takeoff=takeoff)
    stats = record_episode(
        "SuperMarioBros-1-1-v0",
        takeoff,
        tmp_path / f"{mode}-{uuid.uuid4().hex}.gif",
        max_decisions=DECISIONS,
        seed=777,
        stall_timeout=15,
        cadence="grounded",
        apex_policy=apex,
        actions=BASE_MACROS,
    )
    assert client.feedback_calls == {}
    return stats, takeoff, apex


def _moves(stats):
    return [
        (t["kind"], t["x"], t.get("macro") or t.get("choice"), t.get("frames"), t.get("x_after"))
        for t in stats["trace"]
    ]


def test_keep_base_replays_the_base_trajectory_exactly(tmp_path):
    raw, _, _ = _frozen_trajectory(tmp_path, "raw")
    assert raw["decisions"] >= 10 and any(t["kind"] == "apex" for t in raw["trace"]), (
        "too short to mean much"
    )
    for mode, answer in (
        ("base_only", None),
        ("force_keep", None),
        ("refiner", {REF[0]: KEEP_BASE, REF[1]: KEEP_BASE}),
        ("refiner", {REF[0]: None, REF[1]: None}),  # abstains every time → keep_base fallback
    ):
        got, takeoff, apex = _frozen_trajectory(tmp_path, mode, answer)
        assert _moves(got) == _moves(raw), f"{mode} {answer} diverged from the base"
        assert got["max_x"] == raw["max_x"]
        assert takeoff.stats.overrides == 0 and apex.stats.overrides == 0
        if mode == "refiner" and answer[REF[0]] is None:
            assert takeoff.stats.abstained == takeoff.stats.decisions > 0


def test_an_override_changes_the_executed_move_and_is_counted(tmp_path):
    got, takeoff, _apex = _frozen_trajectory(
        tmp_path, "refiner", {REF[0]: "override_jump", REF[1]: KEEP_BASE}
    )
    assert all(t["macro"] == "jump" for t in got["trace"] if t["kind"] == "takeoff")
    assert not any(t["kind"] == "apex" for t in got["trace"])  # a vertical jump has no apex look
    assert takeoff.stats.jump_overrides == takeoff.stats.overrides == takeoff.stats.decisions
    assert all(r["base"] != "jump" and r["executed"] == "jump" for r in takeoff.log)


def test_a_learning_episode_sends_feedback_only_to_the_correctors(tmp_path):
    inner = _scripted({REF[0]: None, REF[1]: None})
    client = GuardedClient(inner, BASE)
    bt = frozen_base(client, BASE[0], FULL_V3_FEATURE_NAMES, run_id="t")
    ba = frozen_base(client, BASE[1], APEX_V3_FEATURE_NAMES, run_id="t")
    rt = refiner_policy(client, REF[0], "takeoff", learn=True, epsilon=0.5, rng_seed=1, run_id="t")
    ra = refiner_policy(client, REF[1], "apex", learn=True, epsilon=0.5, rng_seed=2, run_id="t")
    takeoff = RefinedPolicy("takeoff", bt, rt)
    apex = RefinedPolicy("apex", ba, ra, takeoff=takeoff)
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    try:
        credit = DelayedCredit(CreditConfig(), RecordsBudget(500), "ep-refine")
        stats = play_episode(
            env,
            takeoff=takeoff,
            apex=apex,
            macros=BASE_MACROS,
            credit=credit,
            seed=777,
            max_decisions=DECISIONS,
            stall_timeout=10,
        )
    finally:
        env.close()
    assert inner.feedback_log[BASE[0]] == [] and inner.feedback_log[BASE[1]] == []
    assert set(client.feedback_calls) <= set(REF)
    t_log, a_log = inner.feedback_log[REF[0]], inner.feedback_log[REF[1]]
    assert len(t_log) == stats.decisions and len(a_log) == stats.apex_looks
    assert set(t_log[0]["context"]["values"]) == set(takeoff_refiner_features())
    assert set(t_log[0]["values"]["next_state"]) == set(takeoff_refiner_features())
    assert {f["policy"] for f in t_log} <= {e.value for e in TakeoffEdit}
    # the credited arm is the one that was executed: every logged edit matches a feedback, in order
    assert [f["policy"] for f in t_log] == [r["refinement"] for r in takeoff.log]
    if a_log:
        assert set(a_log[0]["context"]["values"]) == set(apex_refiner_features())
        assert [f["policy"] for f in a_log] == [r["refinement"] for r in apex.log]
    # the base was asked once per decision, the correctors too; nothing else touched the base
    assert client.query_calls[BASE[0]] == stats.decisions


def test_refine_run_caps_records_checkpoints_and_never_feeds_the_base(tmp_path):
    import json
    from pathlib import Path

    from typesafe_mario.refinement import Pair, RefineConfig, refine_run

    inner = _scripted({REF[0]: None, REF[1]: None})
    client = GuardedClient(inner, BASE)
    cfg = RefineConfig(
        base=Pair(*BASE),
        refiner=Pair(*REF),
        max_records=40,
        checkpoints=(16, 32, 64),
        eval_seeds=(777,),
        epsilon=0.3,
        max_decisions=25,
        stall_timeout=10,
        out_dir=tmp_path,
    )
    summary = refine_run(client, cfg, log=lambda _m: None)
    assert summary["records_spent"] == 40 and summary["base_feedback_calls"] == 0
    assert inner.feedback_log[BASE[0]] == [] and inner.feedback_log[BASE[1]] == []
    assert sum(len(inner.feedback_log[d]) for d in REF) == 40
    tags = [
        json.loads(line)["tag"]
        for line in Path(summary["run_log"]).read_text().splitlines()
        if '"checkpoint"' in line
    ]
    assert tags[0] == "r0000" and len(tags) == len(set(tags)) and len(tags) >= 2
    with pytest.raises(
        BaseFeedbackError
    ):  # a client that does not guard the base is refused outright
        refine_run(inner, cfg, log=lambda _m: None)


def test_credible_dominance_gate_is_set_only_when_asked():
    plain = refiner_domain_config("r", "takeoff")["learning"]["policy"]
    gated = refiner_domain_config("r", "takeoff", credible_dominance=True, dominance_margin=0.01)
    assert "require_credible_dominance" not in plain
    assert gated["learning"]["policy"]["require_credible_dominance"] is True
    assert gated["learning"]["policy"]["dominance_margin"] == 0.01


def test_anchored_exploration_credits_the_executed_edit_and_queries_in_exploit(tmp_path):
    inner = _scripted(
        {REF[0]: None, REF[1]: None}
    )  # the corrector abstains: its own choice = keep_base
    client = GuardedClient(inner, BASE)
    sent_bodies = []
    real_query = inner.query

    def spy(domain_id, body):
        if domain_id in REF and body.get("context"):
            sent_bodies.append(body)
        return real_query(domain_id, body)

    inner.query = spy
    bt = frozen_base(client, BASE[0], FULL_V3_FEATURE_NAMES, run_id="t")
    ba = frozen_base(client, BASE[1], APEX_V3_FEATURE_NAMES, run_id="t")
    rt = refiner_policy(client, REF[0], "takeoff", learn=True, run_id="t", server_exploration=False)
    ra = refiner_policy(client, REF[1], "apex", learn=True, run_id="t", server_exploration=False)
    takeoff = RefinedPolicy("takeoff", bt, rt, anchor_epsilon=0.5, rng_seed=4)
    apex = RefinedPolicy("apex", ba, ra, takeoff=takeoff, anchor_epsilon=0.5, rng_seed=5)
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    try:
        credit = DelayedCredit(CreditConfig(), RecordsBudget(500), "ep-anchored")
        stats = play_episode(
            env,
            takeoff=takeoff,
            apex=apex,
            macros=BASE_MACROS,
            credit=credit,
            seed=777,
            max_decisions=DECISIONS,
            stall_timeout=10,
        )
    finally:
        env.close()
    assert sent_bodies and all(b["allow_exploration"] is False for b in sent_bodies)
    assert all(b.get("selection_mode") == "exploit" for b in sent_bodies)
    t_log = inner.feedback_log[REF[0]]
    assert [f["policy"] for f in t_log] == [r["refinement"] for r in takeoff.log]
    statuses = {r["refiner_status"] for r in takeoff.log}
    assert statuses <= {"explored", "abstained"} and "explored" in statuses
    # not explored → the abstaining corrector keeps the base
    assert all(
        r["refinement"] == KEEP_BASE for r in takeoff.log if r["refiner_status"] == "abstained"
    )
    assert inner.feedback_log[BASE[0]] == [] and stats.decisions == len(t_log)


def test_reward_max_is_declared_on_the_corrector():
    cfg = refiner_domain_config("r", "apex", credible_dominance=True, reward_max=0.0076)
    comp = cfg["learning"]["reward"]["components"][0]
    assert comp["max"] == 0.0076 and comp["min"] == 0.0
    assert (
        cfg["learning"]["training"]["model_type"] == "extra_trees"
    )  # the pin that keeps #30's flip away


def test_directed_epsilon_follows_the_correctors_own_keep_score():
    client = OfflineAdapt1Client()
    base = frozen_base(client, "b", FULL_V3_FEATURE_NAMES, run_id="t")
    pol = RefinedPolicy("takeoff", base, None, anchor_epsilon=0.25, directed=(0.6, 0.05, 0.75))
    assert pol.epsilon_for({KEEP_BASE: 0.17}) == pytest.approx(0.43)  # the 2-1 wall
    assert pol.epsilon_for({KEEP_BASE: 0.65}) == 0.05  # where the base is measured good
    assert pol.epsilon_for({KEEP_BASE: -1.0}) == 0.75
    assert pol.epsilon_for({}) == 0.25  # no score: the plain anchored ε
    plain = RefinedPolicy("takeoff", base, None, anchor_epsilon=0.25)
    assert plain.epsilon_for({KEEP_BASE: 0.17}) == 0.25


def test_frozen_prefix_sends_no_feedback_and_replays_the_frozen_system(tmp_path):
    inner = _scripted({REF[0]: None, REF[1]: None})
    client = GuardedClient(inner, BASE)
    bt = frozen_base(client, BASE[0], FULL_V3_FEATURE_NAMES, run_id="t")
    ba = frozen_base(client, BASE[1], APEX_V3_FEATURE_NAMES, run_id="t")
    rt = refiner_policy(client, REF[0], "takeoff", learn=True, run_id="t", server_exploration=False)
    ra = refiner_policy(client, REF[1], "apex", learn=True, run_id="t", server_exploration=False)
    takeoff = RefinedPolicy("takeoff", bt, rt, anchor_epsilon=1.0, rng_seed=4, learn_after=6)
    apex = RefinedPolicy("apex", ba, ra, takeoff=takeoff, anchor_epsilon=1.0, rng_seed=5)
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    try:
        credit = DelayedCredit(CreditConfig(), RecordsBudget(500), "ep-prefix")
        stats = play_episode(
            env,
            takeoff=takeoff,
            apex=apex,
            macros=BASE_MACROS,
            credit=credit,
            seed=777,
            max_decisions=DECISIONS,
            stall_timeout=10,
        )
    finally:
        env.close()
    raw, _, _ = _frozen_trajectory(tmp_path, "raw")
    first6 = [r for r in raw["trace"] if r["kind"] == "takeoff"][:6]
    assert [(r["x_debug"], r["base"]) for r in takeoff.log[:6]] == [
        (r["x"], r["macro"]) for r in first6
    ]
    assert all(r["prefix"] and r["refinement"] == KEEP_BASE for r in takeoff.log[:6])
    assert all(not r["prefix"] for r in takeoff.log[6:])
    t_log = inner.feedback_log[REF[0]]
    assert len(t_log) == stats.decisions - 6  # the prefix cost nothing
    assert [f["policy"] for f in t_log] == [r["refinement"] for r in takeoff.log[6:]]
    prefix_apex = sum(1 for r in apex.log if r["prefix"])
    assert len(inner.feedback_log[REF[1]]) == len(apex.log) - prefix_apex
