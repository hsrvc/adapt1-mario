"""Offline contract tests for the Adapt-1 Mario policy.

No network: a fake client records the request bodies and returns canned responses, so
these assert the *request/response shape* against the documented Adapt-1 contract
without touching the live API or burning quota. Live smokes live in scripts/, separately.
"""

from __future__ import annotations

import json

from typesafe_mario.actions import Action
from typesafe_mario.adapt1_policy import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SETS,
    PERCEPTION_FEATURE_NAMES,
    Adapt1Policy,
    build_domain_config,
    context_features,
    feature_paths,
    snapshot_features,
)
from typesafe_mario.state import MarioStateParser


def make_snapshot(**info_overrides):
    info = {
        "x_pos": 100,
        "y_pos": 79,
        "y_pixel": 79,
        "world": 1,
        "stage": 1,
        "area": 1,
        "status": "small",
        "player_state": 8,
        "life": 2,
        "coins": 0,
        "score": 200,
        "time": 390,
        "progress": 100,
        "progress_max": 100,
    }
    info.update(info_overrides)
    return MarioStateParser().parse(info, None, previous_action="right_run", previous_reward=8.0)


class FakeClient:
    def __init__(self, query_resp, feedback_resp=None):
        self.query_resp = query_resp
        self.feedback_resp = feedback_resp or {}
        self.queries = []
        self.feedbacks = []

    def query(self, domain_id, body):
        self.queries.append((domain_id, body))
        return 200, self.query_resp

    def feedback(self, domain_id, body):
        self.feedbacks.append((domain_id, body))
        return 200, self.feedback_resp


def test_features_match_names_and_are_leakage_safe():
    feats = snapshot_features(make_snapshot().to_state())
    # The flattener emits every feature (the apex extras included, zero when grounded); the
    # "full" arm projects it back to exactly FEATURE_NAMES.
    assert tuple(feats.keys()) == ALL_FEATURE_NAMES
    assert tuple(context_features(make_snapshot().to_state(), FEATURE_NAMES)) == FEATURE_NAMES
    # None of the outcome / absolute-position fields may leak into the decision context.
    forbidden = {"reward", "score", "x", "y", "progress", "best_progress", "time_left"}
    assert forbidden.isdisjoint(feats.keys())
    # Every numeric-or-string feature is JSON serialisable and non-null.
    assert all(v is not None for v in feats.values())


def test_domain_config_is_a_valid_bandit():
    cfg = build_domain_config("mario-1-1")
    hyps = cfg["hypotheses"]
    assert len(hyps) == 7
    assert {h["policy"] for h in hyps} == {a.value for a in Action}
    assert {h["relation"] for h in hyps} == {"controls"}
    learning = cfg["learning"]
    assert learning["enabled"] is True
    assert learning["reward"]["components"][0]["field"] == "values.reward"
    assert learning["context"]["feature_paths"] == feature_paths()
    assert learning["context"]["max_samples"] == 4096
    # Outcomes must be declared or /feedback 422s ("outcome must be recognized").
    assert cfg["query_templates"]["feedback_outcomes"]
    assert "advanced" in cfg["query_templates"]["feedback_outcomes"]


def test_choose_selects_returned_policy_and_sends_query():
    query_resp = {
        "decision_id": "dec_1",
        "selection": {
            "status": "selected",
            "policy": "right_run_jump",
            "policy_scores": [
                {"policy": "right_run_jump", "score": 0.8},
                {"policy": "right", "score": 0.2},
            ],
        },
    }
    client = FakeClient(query_resp)
    policy = Adapt1Policy(client, "mario-1-1")
    decision = policy.choose(make_snapshot(), tuple(Action))

    assert decision.action == Action.RIGHT_RUN_JUMP
    _domain, body = client.queries[0]
    assert body["session_id"] == "ignored"  # required by the request schema
    assert body["question"]
    assert body["allow_exploration"] is True
    assert "top_k" in body
    assert tuple(body["context"]["values"].keys()) == FEATURE_NAMES
    # Probabilities normalise and the chosen macro is the most likely.
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
    assert max(decision.probabilities, key=decision.probabilities.get) == "right_run_jump"


def test_observe_attributes_feedback_with_decision_id():
    query_resp = {"decision_id": "dec_1", "selection": {"status": "selected", "policy": "right"}}
    feedback_resp = {
        "credit_assignment": {"contextual_learning_applied": True, "context_source": "decision"},
        "policy_scores": [{"policy": "right", "observations": 2}],
    }
    client = FakeClient(query_resp, feedback_resp)
    policy = Adapt1Policy(client, "mario-1-1")
    policy.choose(make_snapshot(), tuple(Action))
    record = policy.observe(
        next_state=make_snapshot(x_pos=140).to_state(),
        reward=12.0,
        terminal=False,
        episode_id="ep-x",
        step=0,
    )

    _domain, body = client.feedbacks[0]
    assert body["session_id"] == "ignored"  # required by DomainFeedbackRequest
    assert body["decision_id"] == "dec_1"
    assert body["relation"] == "controls"
    assert body["policy"] == "right"
    assert body["feedback_kind"] == "execution"
    assert body["outcome"] == "advanced"  # reward > 0, not terminal
    assert body["values"]["reward"] == 12.0
    assert tuple(body["context"]["values"].keys()) == FEATURE_NAMES
    assert body["metadata"]["episode_id"] == "ep-x"
    assert body["metadata"]["step"] == 0
    # The audit record surfaces the learning signal, not just the HTTP status.
    assert record["contextual_learning_applied"] is True
    assert (
        "sample_count" in record
    )  # the monotonic learner counter (None here; fake query has none)
    assert record["reward"] == 12.0


def test_abstain_falls_back_to_a_legal_action_and_still_learns():
    query_resp = {
        "decision_id": "dec_2",
        "selection": {"status": "abstained", "reason": "no_evidence"},
    }
    client = FakeClient(query_resp, {"credit_assignment": {"contextual_learning_applied": True}})
    policy = Adapt1Policy(client, "mario-1-1", rng_seed=0)
    decision = policy.choose(make_snapshot(), tuple(Action))
    assert decision.action in set(Action)

    record = policy.observe(
        next_state=make_snapshot().to_state(),
        reward=-5.0,
        terminal=True,
        episode_id="ep-y",
        step=3,
    )
    # Abstention is recorded, but feedback still fires on the fallback action so the
    # cold domain accumulates evidence.
    assert record["abstained"] is True
    assert record["selection_status"] == "abstained"
    assert client.feedbacks[0][1]["policy"] == decision.action.value


def test_perception_arm_strips_the_parser_verdicts():
    assert set(PERCEPTION_FEATURE_NAMES) < set(FEATURE_NAMES)
    verdicts = {"must_jump", "takeoff_deadline", "contact_frames", "obstacle_ahead", "gap_ahead"}
    # The perception arm must NOT contain the pre-computed tactical verdicts.
    assert verdicts.isdisjoint(PERCEPTION_FEATURE_NAMES)
    # But it must keep raw perception it can learn from.
    assert {"enemy_dist", "gap_dist", "grounded", "speed_x"} <= set(PERCEPTION_FEATURE_NAMES)
    assert FEATURE_SETS["full"] == FEATURE_NAMES
    assert FEATURE_SETS["perception"] == PERCEPTION_FEATURE_NAMES


def test_sequential_domain_config_declares_the_paths():
    cfg = build_domain_config("mario-seq", feature_names=PERCEPTION_FEATURE_NAMES, sequential=True)
    seq = cfg["learning"]["sequential"]
    assert seq["enabled"] is True
    assert seq["episode_path"] == "metadata.episode_id"
    assert seq["step_path"] == "metadata.step"
    assert seq["next_context_path"] == "values.next_state"
    assert seq["reward_path"] == "values.reward"
    assert seq["terminal_path"] == "metadata.terminal"
    # Perception arm → 12 feature paths; max_samples stays 4096 (512 evicts transitions).
    assert len(cfg["learning"]["context"]["feature_paths"]) == len(PERCEPTION_FEATURE_NAMES)
    assert cfg["learning"]["context"]["max_samples"] == 4096
    # A bandit config has no sequential block.
    assert "sequential" not in build_domain_config("m")["learning"]
    # n_step defaults to 5; credit_assignment is off by default and absent from the config.
    assert seq["n_step"] == 5
    assert "credit_assignment" not in cfg["learning"]


def test_credit_assignment_and_n_step_knobs():
    cfg = build_domain_config(
        "mario-seq", sequential=True, n_step=8, credit_assignment=True, discount=0.95
    )
    assert cfg["learning"]["sequential"]["n_step"] == 8
    ca = cfg["learning"]["credit_assignment"]
    # mode must match the /openapi.json regex ^(none|eligibility_trace|counterfactual_trace)$
    assert ca["mode"] == "eligibility_trace"
    assert ca["discount"] == 0.95
    # credit_assignment is opt-in even when sequential is on.
    assert "credit_assignment" not in build_domain_config("m", sequential=True)["learning"]


def test_sequential_feedback_carries_next_state_in_the_arm_features():
    query_resp = {"decision_id": "dec_s", "selection": {"status": "selected", "policy": "right"}}
    client = FakeClient(query_resp, {"credit_assignment": {"contextual_learning_applied": True}})
    policy = Adapt1Policy(
        client, "mario-seq", feature_names=PERCEPTION_FEATURE_NAMES, sequential=True
    )
    policy.choose(make_snapshot(), tuple(Action))
    policy.observe(
        next_state=make_snapshot(x_pos=150).to_state(),
        reward=9.0,
        terminal=False,
        episode_id="ep-s",
        step=0,
    )
    _domain, body = client.feedbacks[0]
    # next_state is a JSON object projected to the SAME (perception) feature set as context.
    assert tuple(body["values"]["next_state"].keys()) == PERCEPTION_FEATURE_NAMES
    assert tuple(body["context"]["values"].keys()) == PERCEPTION_FEATURE_NAMES
    assert body["metadata"]["terminal"] is False  # JSON boolean, not "false"


def test_bandit_feedback_omits_next_state():
    query_resp = {"decision_id": "dec_b", "selection": {"status": "selected", "policy": "right"}}
    client = FakeClient(query_resp, {})
    policy = Adapt1Policy(client, "mario-1-1")  # sequential=False by default
    policy.choose(make_snapshot(), tuple(Action))
    policy.observe(
        next_state=make_snapshot().to_state(), reward=1.0, terminal=False, episode_id="ep", step=0
    )
    assert "next_state" not in client.feedbacks[0][1]["values"]


def test_trace_sidecar_is_written(tmp_path):
    trace = tmp_path / "trace.jsonl"
    client = FakeClient(
        {"decision_id": "dec_3", "selection": {"status": "selected", "policy": "right_run"}},
        {"credit_assignment": {"contextual_learning_applied": True}},
    )
    policy = Adapt1Policy(client, "mario-1-1", trace_path=str(trace))
    policy.choose(make_snapshot(), tuple(Action))
    policy.observe(
        next_state=make_snapshot().to_state(), reward=8.0, terminal=False, episode_id="ep-z", step=0
    )
    policy.close()

    lines = trace.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["episode_id"] == "ep-z"
    assert row["decision_id"] == "dec_3"
    assert row["policy"] == "right_run"
    assert row["reward"] == 8.0


def test_abstain_fallback_prefers_state_dependent_diagnostics_over_context_free_means():
    """findings #22: `policy_scores[].score` is the context-free mean reward per macro (the same
    vector in every state). Falling back to its argmax makes eval a CONSTANT policy. Prefer the
    per-state `policy_diagnostics` numbers when the server abstains on a tie."""
    scores = [{"policy": "right_jump", "score": 0.37}, {"policy": "right_run", "score": 0.09}]
    # 1. selection_expected_reward alive for some macro → its argmax wins.
    diag = {
        "right_run": {"selection_expected_reward": 0.6, "calibrated_contextual_reward": 0.1},
        "right_jump": {"selection_expected_reward": 0.2, "calibrated_contextual_reward": 0.9},
    }
    resp = {
        "decision_id": "d",
        "selection": {"status": "abstained", "reason": "tie"},
        "policy_scores": scores,
        "policy_diagnostics": diag,
    }
    policy = Adapt1Policy(FakeClient(resp), "m")
    assert policy.choose(make_snapshot(), tuple(Action)).action is Action.RIGHT_RUN
    # 2. all selection_expected_reward == 0 (the measured tie) → calibrated contextual reward.
    for d in diag.values():
        d["selection_expected_reward"] = 0
    assert policy.choose(make_snapshot(), tuple(Action)).action is Action.RIGHT_JUMP
    # 3. no diagnostics at all → the old context-free argmax, never random.
    resp.pop("policy_diagnostics")
    assert policy.choose(make_snapshot(), tuple(Action)).action is Action.RIGHT_JUMP
    resp["policy_scores"] = []
    assert policy.choose(make_snapshot(), tuple(Action)).action is Action.RIGHT_RUN


def test_epsilon_explores_only_when_exploration_is_allowed():
    """findings #22: client-side ε-greedy for the ONLINE phase; frozen eval (allow_exploration=False)
    must never explore, and the executed (explored) macro is what feedback credits."""
    resp = {
        "decision_id": "d",
        "selection": {"status": "selected", "selected_policy": "right_run"},
        "policy_scores": [{"policy": "right_run", "score": 0.5}],
    }
    frozen = Adapt1Policy(FakeClient(resp), "m", allow_exploration=False, epsilon=1.0, rng_seed=1)
    assert all(
        frozen.choose(make_snapshot(), tuple(Action)).action is Action.RIGHT_RUN for _ in range(5)
    )
    assert frozen.status_counts == {"selected": 5}

    client = FakeClient(resp, {"credit_assignment": {"contextual_learning_applied": True}})
    online = Adapt1Policy(client, "m", allow_exploration=True, epsilon=1.0, rng_seed=1)
    picks = {online.choose(make_snapshot(), tuple(Action)).action for _ in range(20)}
    assert len(picks) > 1 and online.status_counts.get("explored") == 20
    online.choose(make_snapshot(), tuple(Action))
    record = online.observe(
        next_state=make_snapshot().to_state(), reward=0.01, terminal=False, episode_id="e", step=0
    )
    assert record["explored"] is True
    assert client.feedbacks[-1][1]["policy"] == record["policy"]


def test_tcp_config_and_per_query_metadata():
    """TCP (temporal-context-projection.md): numeric input paths only; every query carries
    (episode_id, step) inside context.metadata, monotone and never reused; feedback echoes it."""
    cfg = build_domain_config("t", sequential=True, temporal_context=True, maximum_lag=16)
    tc = cfg["learning"]["temporal_context"]
    assert tc["enabled"] and tc["maximum_lag"] == 16
    assert tc["episode_path"] == "metadata.episode_id" and tc["step_path"] == "metadata.step"
    assert not any(p.endswith(("jump_phase", "reliability", "outcome")) for p in tc["input_paths"])
    assert "values.drop_dist" in tc["input_paths"]

    resp = {"decision_id": "d", "selection": {"status": "selected", "selected_policy": "right_run"}}
    client = FakeClient(resp, {"credit_assignment": {"contextual_learning_applied": True}})
    policy = Adapt1Policy(client, "t", sequential=True, temporal_context=True)
    policy.start_episode("ep-1")
    policy.choose(make_snapshot(), tuple(Action))
    policy.choose(make_snapshot(), tuple(Action))  # frozen-eval style: no observe in between
    metas = [q[1]["context"]["metadata"] for q in client.queries]
    assert metas == [{"episode_id": "ep-1", "step": 0}, {"episode_id": "ep-1", "step": 1}]
    policy.observe(
        next_state=make_snapshot().to_state(),
        reward=0.01,
        terminal=False,
        episode_id="ep-1",
        step=1,
    )
    assert client.feedbacks[0][1]["context"]["metadata"] == {"episode_id": "ep-1", "step": 1}
    plain = Adapt1Policy(FakeClient(resp), "t")
    plain.choose(make_snapshot(), tuple(Action))
    assert "metadata" not in plain._client.queries[0][1]["context"]


def test_telemetry_reports_the_server_view_and_frozen_mode_sends_no_feedback():
    """Dashboard telemetry (findings #22): deployed model, selection status/reason, per-macro learned
    values, running counts; learn=False queries but never sends feedback."""
    resp = {
        "decision_id": "d",
        "selection": {"status": "abstained", "reason": "tie", "tie_size": 7},
        "policy_scores": [{"policy": "right_run", "score": 0.1}],
        "policy_diagnostics": {
            "right_run": {"sequential_expected_reward": 0.021, "selection_expected_reward": 0.02},
            "right_jump": {"sequential_expected_reward": 0.054, "selection_expected_reward": 0.05},
        },
        "learning_state": {
            "sample_counts": {"feedback_policy": 1633},
            "subsystems": {
                "feedback_policy": {
                    "model": {
                        "current_model_type": "extra_trees",
                        "current_validation_skill": 0.74,
                        "report": {"sample_count": 1631},
                    }
                }
            },
        },
    }
    client = FakeClient(resp, {"credit_assignment": {"contextual_learning_applied": True}})
    frozen = Adapt1Policy(client, "mario-floor-gp", learn=False)
    decision = frozen.choose(make_snapshot(), tuple(Action))
    tel = decision.telemetry
    assert tel["domain"] == "mario-floor-gp" and tel["model_type"] == "extra_trees"
    assert (
        tel["validation_skill"] == 0.74 and tel["status"] == "abstained" and tel["reason"] == "tie"
    )
    assert tel["values"] == {"right_run": 0.021, "right_jump": 0.054} and tel["learning"] is False
    assert (
        decision.action is Action.RIGHT_JUMP
    )  # state-aware fallback: argmax of the learned values
    record = frozen.observe(
        next_state=make_snapshot().to_state(), reward=0.01, terminal=False, episode_id="e", step=0
    )
    assert client.feedbacks == []  # frozen: nothing sent
    assert frozen.last_feedback == {
        "reward": 0.01,
        "outcome": "advanced",
        "terminal": False,
        "sent": False,
    }
    assert record["decision_id"] == "d"
    live = Adapt1Policy(FakeClient(resp, {}), "m")
    live.choose(make_snapshot(), tuple(Action))
    live.observe(
        next_state=make_snapshot().to_state(), reward=0.01, terminal=False, episode_id="e", step=0
    )
    assert live.last_feedback["sent"] is True and len(live._client.feedbacks) == 1


def test_replay_policy_feeds_a_logged_run_back_with_its_telemetry(tmp_path):
    """Replay mode (dashboard without queries): the logged macros come back in order, with the
    logged Adapt-1 telemetry when present; cadence is inferred from the action names."""
    from typesafe_mario.macros import Macro
    from typesafe_mario.policy import ReplayPolicy

    rows = [
        {
            "action": "right_run",
            "confidence": 0.2,
            "probabilities": {"right_run": 1.0},
            "selection_status": "selected",
            "telemetry": {"model_type": "extra_trees", "values": {"right_run": 0.01}},
        },
        {
            "action": "right_run_jump_full",
            "confidence": 0.2,
            "probabilities": {},
            "selection_status": "explored",
            "telemetry": None,
        },
    ]
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    policy = ReplayPolicy(log)
    assert ReplayPolicy.infer_cadence(policy.rows) == "grounded"
    first = policy.choose(make_snapshot(), tuple(Macro))
    assert first.action is Macro.RIGHT_RUN and first.latency_ms == 0.0
    assert first.telemetry["model_type"] == "extra_trees" and first.telemetry["replay"] is True
    second = policy.choose(make_snapshot(), tuple(Macro))
    assert second.action is Macro.RIGHT_RUN_JUMP_FULL and second.telemetry is None
    assert second.selection_status == "explored"
    assert policy.remaining == 0
    assert policy.choose(make_snapshot(), tuple(Macro)).selection_status == "replay-ended"
    assert ReplayPolicy.infer_cadence([{"action": "right_run_jump"}]) == "frame"
