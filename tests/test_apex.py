"""Offline tests for the apex decision (design-notes §14a) — a fake emulator, no API."""

from __future__ import annotations

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.adapt1_policy import (
    ALL_FEATURE_NAMES,
    APEX_EXTRA_NAMES,
    FEATURE_NAMES,
    FEATURE_SETS,
    QUESTION,
    _parse_query,
    build_domain_config,
    snapshot_features,
)
from typesafe_mario.macros import (
    APEX_BRAKE_FRAMES,
    APEX_NAMES,
    ApexChoice,
    Macro,
    apex_choices,
    execute,
)
from typesafe_mario.offline_client import OfflineAdapt1Client, OfflineContractError
from typesafe_mario.policy import DiversifiedHeuristicPolicy
from typesafe_mario.state import MarioStateParser

LEFT, NOOP, RUN = (
    ACTION_TO_INDEX[Action.LEFT],
    ACTION_TO_INDEX[Action.NOOP],
    ACTION_TO_INDEX[Action.RIGHT_RUN],
)
JUMP_INDICES = {ACTION_TO_INDEX[a] for a in (Action.RIGHT_JUMP, Action.RIGHT_RUN_JUMP, Action.JUMP)}


class ArcEnv:
    """Scripted NES with a jump arc: a grounded jump press makes Mario rise `rise` frames
    (y_pos +5/frame) then fall the same, RAM float-state 1 while airborne, x +3/frame."""

    def __init__(self, rise: int = 12) -> None:
        self.ram = bytearray(0x0800)
        self.ram[0x57] = 48  # 3 px/frame
        self.steps: list[int] = []
        self.rise = rise
        self.air_t = -1
        self.x = 100
        self.y = 79

    def step(self, index: int):
        self.steps.append(index)
        if index in JUMP_INDICES and self.air_t < 0:
            self.air_t = 0
        if self.air_t >= 0:
            self.air_t += 1
            self.y = 79 + 5 * (
                self.air_t if self.air_t <= self.rise else 2 * self.rise - self.air_t
            )
            if self.air_t >= 2 * self.rise:
                self.air_t = -1
                self.y = 79
        self.ram[0x1D] = 1 if self.air_t >= 0 else 0
        self.x += 3
        return None, 1.0, False, False, {"x_pos": self.x, "y_pos": self.y, "y_pixel": 200 - self.y}


def _run(choice: ApexChoice | None, macro: Macro = Macro.RIGHT_RUN_JUMP_FULL):
    env = ArcEnv()
    events = []

    def hook(event):
        events.append(event)
        return choice

    ex = execute(env, macro, grounded=True, ram=env.ram, apex=hook if choice is not None else None)
    return env, events, ex


def test_hook_fires_once_at_the_top_and_keep_is_byte_identical():
    plain_env, _, plain = _run(None)
    env, events, ex = _run(ApexChoice.KEEP)
    assert len(events) == 1
    event = events[0]
    assert event.macro is Macro.RIGHT_RUN_JUMP_FULL
    assert event.frame == env.rise + 2  # release edge + rise frames + the first non-rising frame
    assert event.vx == 3.0 and event.y == 79 + 5 * (env.rise - 1)  # first frame past the peak
    assert ex.apex is event and ex.apex_choice is ApexChoice.KEEP
    assert ex.frames_to_apex == event.frame and ex.reward_to_apex == float(event.frame)
    assert env.steps == plain_env.steps  # keep changes nothing
    assert plain.apex is None and plain.apex_choice is None


def test_pull_back_holds_left_from_the_apex_to_the_landing():
    env, events, _ex = _run(ApexChoice.PULL_BACK)
    after = env.steps[events[0].frame :]
    assert after and all(i == LEFT for i in after)
    assert env.ram[0x1D] == 0  # ran to the landing


def test_brake_holds_left_then_releases():
    env, events, _ex = _run(ApexChoice.BRAKE)
    after = env.steps[events[0].frame :]
    assert after[:APEX_BRAKE_FRAMES] == [LEFT] * APEX_BRAKE_FRAMES
    assert after[APEX_BRAKE_FRAMES:] and all(i == NOOP for i in after[APEX_BRAKE_FRAMES:])


def test_non_jump_macros_never_reach_an_apex():
    env, events, ex = _run(ApexChoice.PULL_BACK, macro=Macro.RIGHT_RUN)
    assert events == [] and ex.apex is None
    assert all(i == RUN for i in env.steps)


def test_apex_names_and_choice_set():
    assert apex_choices() == tuple(ApexChoice)
    assert APEX_NAMES == ("apex_keep", "apex_brake", "apex_pull_back")


# --- features --------------------------------------------------------------------------


def _info(x: int = 100, y: int = 80) -> dict:
    return {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": x,
        "y_pos": y,
        "y_pixel": 80,
        "progress": x,
        "time": 390,
        "life": 2,
        "status": "small",
    }


def _ram(*, pit_cols=(), floor_rows=(5, 6)) -> bytearray:
    ram = bytearray(0x0800)
    for column in range(16):
        if column in pit_cols:
            continue
        for row in floor_rows:
            ram[0x0500 + row * 16 + column] = 0x54
    return ram


def test_flattener_emits_the_superset_and_the_apex_arm_projects_raw_terrain():
    state = MarioStateParser().parse(_info(), _ram()).to_state()
    feats = snapshot_features(state)
    assert (
        tuple(feats) == ALL_FEATURE_NAMES
        and ALL_FEATURE_NAMES[:25] == FEATURE_NAMES + APEX_EXTRA_NAMES
    )
    assert (feats["apex_vx"], feats["apex_rise"], feats["apex_height"]) == (0.0, 0.0, 0.0)
    state["apex"] = {"vx": 3.0, "rise": 27, "height": 9}
    feats = snapshot_features(state)
    assert (feats["apex_vx"], feats["apex_rise"], feats["apex_height"]) == (3.0, 27.0, 9.0)
    state["apex"] = {"vx": 3.0, "rise": 27, "height": None}  # over a pit: sentinel, not null
    assert snapshot_features(state)["apex_height"] == 999.0
    apex_arm = FEATURE_SETS["apex"]
    assert set(APEX_EXTRA_NAMES) <= set(apex_arm)
    assert "must_jump" not in apex_arm  # perception-style: no takeoff-preview distances


# --- the teacher's apex rule ------------------------------------------------------------


def _apex_choice(ram: bytearray, extra: dict) -> ApexChoice:
    snapshot = MarioStateParser().parse(_info(), ram)
    return (
        DiversifiedHeuristicPolicy(epsilon=0.0).choose(snapshot, apex_choices(), extra=extra).action
    )


def test_teacher_pulls_back_when_the_projected_landing_column_is_bottomless():
    # Mario at column 6; the pit is at columns 13-14 = 7-8 tiles ahead (the pipe-4 geometry).
    # Full jump from a pipe top: vx 3 px/f, rose 27 frames, 9 tiles up -> lands ~7 tiles on.
    pit = _ram(pit_cols=(13, 14))
    assert _apex_choice(pit, {"vx": 3.0, "rise": 27, "height": 9}) is ApexChoice.PULL_BACK


def test_teacher_keeps_when_the_arc_lands_on_floor_or_is_already_over_the_pit():
    pit = _ram(pit_cols=(13, 14))
    assert _apex_choice(_ram(), {"vx": 3.0, "rise": 27, "height": 9}) is ApexChoice.KEEP  # no pit
    assert (
        _apex_choice(pit, {"vx": 3.0, "rise": 16, "height": 0}) is ApexChoice.KEEP
    )  # lands short of it
    assert (
        _apex_choice(pit, {"vx": 3.0, "rise": 27, "height": None}) is ApexChoice.KEEP
    )  # over a pit
    assert (
        _apex_choice(pit, {"vx": -1.0, "rise": 27, "height": 9}) is ApexChoice.KEEP
    )  # moving left
    assert _apex_choice(pit, {}) is ApexChoice.KEEP


def test_teacher_epsilon_explores_apex_choices():
    snapshot = MarioStateParser().parse(_info(), _ram())
    teacher = DiversifiedHeuristicPolicy(epsilon=1.0)
    seen = {teacher.choose(snapshot, apex_choices(), extra={}).action for _ in range(40)}
    assert seen == set(ApexChoice)
    assert teacher.choose(snapshot, apex_choices(), extra={}).selection_status == "explored"


# --- the apex domain's offline contract ---------------------------------------------------


def test_apex_domain_accepts_apex_choices_and_rejects_macros():
    feats = FEATURE_SETS["apex"]
    client = OfflineAdapt1Client()
    client.create_domain(
        build_domain_config("apex", feature_names=feats, macros=APEX_NAMES, sequential=False)
    )
    ctx = {f: 0.0 for f in feats}
    ctx.update({"jump_phase": "apex", "reliability": "low_airborne"})
    context = {"values": ctx}

    def feed(policy: str) -> None:
        _s, qr = client.query(
            "apex", {"session_id": "x", "question": QUESTION, "context": context, "top_k": 1}
        )
        client.feedback(
            "apex",
            {
                "session_id": "x",
                "decision_id": _parse_query(qr)["decision_id"],
                "relation": "controls",
                "policy": policy,
                "feedback_kind": "execution",
                "outcome": "advanced",
                "context": context,
                "values": {"reward": 0.02, "next_state": dict(ctx)},
                "metadata": {"episode_id": "demo-ep0000", "step": 4, "terminal": False},
            },
        )

    feed("apex_pull_back")
    assert client.samples["apex"] == 1
    with pytest.raises(OfflineContractError):
        feed("right_run_jump_full")


# --- client-side returns (shared by kstep_returns.py and apex_coverage.py) -----------------


def test_kstep_return_is_normalised_and_truncates_at_the_episode_end():
    from typesafe_mario.returns import kstep_return, kstep_returns

    r = [0.01, 0.01, 0.01, 0.0]
    assert kstep_return(r, 0, 2, 1.0) == 0.01  # mean of the next two
    assert kstep_return(r, 3, 2, 1.0) == 0.0  # a death: nothing after it
    assert kstep_return(r, 2, 2, 1.0) == 0.005  # half the window is past the end
    assert kstep_return([5.0], 0, 3, 0.9) == 1.0  # 5/2.71 clipped to [0,1]
    rows = [{"episode_id": "e", "step": i, "reward": v} for i, v in enumerate(r)]
    out = kstep_returns(rows, 2, 1.0)
    assert [o["immediate_reward"] for o in out] == r and out[0]["reward"] == 0.01


# --- level memory (design-notes §14b): position as a named, projected exception ----------


def test_position_features_exist_only_in_the_position_arm():
    from typesafe_mario.adapt1_policy import POSITION_FEATURE_NAMES, context_features

    state = MarioStateParser().parse(_info(x=1300), _ram()).to_state()
    feats = snapshot_features(state)
    assert feats["level_screen"] == 5.0 and abs(feats["level_x"] - 1300 / 3161) < 1e-3
    assert set(POSITION_FEATURE_NAMES).isdisjoint(context_features(state, FEATURE_SETS["full"]))
    assert set(POSITION_FEATURE_NAMES).isdisjoint(context_features(state, FEATURE_SETS["apex"]))
    with_pos = context_features(state, FEATURE_SETS["full+screen"])
    assert tuple(with_pos) == FEATURE_NAMES + ("level_screen",) and with_pos["level_screen"] == 5.0


# --- the online path: replay feeds the logged apex look back; the CLI knows the flags -------


def test_replay_policy_returns_the_logged_apex_choice(tmp_path):
    import json

    from typesafe_mario.policy import ReplayPolicy

    rows = [
        {
            "action": "right_run_jump_full",
            "confidence": 1.0,
            "probabilities": {},
            "selection_status": None,
            "apex": {"choice": "apex_pull_back", "selection_status": "selected", "confidence": 0.7},
        },
        {
            "action": "right_run",
            "confidence": 1.0,
            "probabilities": {},
            "selection_status": None,
            "apex": None,
        },
    ]
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    replay = ReplayPolicy(log)
    assert replay.has_apex
    snap = MarioStateParser().parse(_info(), _ram())
    assert replay.choose(snap, tuple(Macro)).action is Macro.RIGHT_RUN_JUMP_FULL
    assert replay.choose(snap, apex_choices(), extra={}).action is ApexChoice.PULL_BACK
    assert replay.choose(snap, tuple(Macro)).action is Macro.RIGHT_RUN
    assert replay.choose(snap, apex_choices(), extra={}).action is ApexChoice.KEEP  # no look logged


def test_play_cli_knows_the_apex_flags():
    from typesafe_mario.cli import build_parser

    args = build_parser().parse_args(
        ["play", "--policy", "heuristic", "--cadence", "grounded", "--apex-teacher"]
    )
    assert args.apex_teacher and args.apex_domain is None
    args = build_parser().parse_args(
        [
            "play",
            "--policy",
            "adapt1",
            "--domain-id",
            "d",
            "--cadence",
            "grounded",
            "--apex-domain",
            "a",
            "--frozen",
        ]
    )
    assert args.apex_domain == "a" and args.frozen


# --- fixed frame-horizon return (2026-09-23): the action cannot choose its own denominator -----
def test_frame_horizon_return_pays_progress_per_frame_over_a_fixed_window():
    from typesafe_mario.returns import frame_horizon_return

    # a run step (8 f, 24 px) then more run steps: 3 px/frame throughout
    r = [24 / 3161] * 8
    f = [8] * 8
    assert frame_horizon_return(r, f, 0, 32) == round(24 / 3161, 6)  # 4 rows fill 32 frames exactly
    # a 51-frame full jump earning 152 px is NOT worth more than a run step at the same speed
    jump_r = [152 / 3161, 24 / 3161, 24 / 3161]
    jump_f = [51, 8, 8]
    assert (
        abs(frame_horizon_return(jump_r, jump_f, 0, 64) - frame_horizon_return(r, f, 0, 64)) < 2e-4
    )


def test_frame_horizon_return_prorates_the_crossing_row_and_forfeits_frames_after_a_death():
    from typesafe_mario.returns import frame_horizon_return

    # a 51-frame jump crossing a 32-frame horizon is prorated
    assert frame_horizon_return([152 / 3161], [51], 0, 32) == round(
        152 / 3161 * 32 / 51 * 8 / 32, 6
    )
    # dying after 17 frames with 50 px earned: the other 47 frames of a 64-frame horizon earn nothing
    dead = frame_horizon_return([50 / 3161], [17], 0, 64)
    alive = frame_horizon_return(
        [50 / 3161, 24 / 3161, 24 / 3161, 24 / 3161, 24 / 3161, 24 / 3161, 24 / 3161],
        [17] + [8] * 6,
        0,
        64,
    )
    assert dead < alive
    assert dead == round(50 / 3161 * 8 / 64, 6)


def test_frame_horizon_return_credits_a_clear_at_full_speed_and_a_stall_at_nothing():
    from typesafe_mario.returns import FULL_SPEED_RATE, frame_horizon_return, frame_horizon_returns

    r, f = [24 / 3161], [8]
    assert frame_horizon_return(r, f, 0, 64, cleared=True) > frame_horizon_return(r, f, 0, 64)
    assert frame_horizon_return([], [], 0, 64, cleared=True) == round(FULL_SPEED_RATE, 6)
    rows = [
        {"episode_id": "e", "step": 1, "reward": 24 / 3161, "frames": 8, "outcome": "advanced"},
        {"episode_id": "e", "step": 0, "reward": 24 / 3161, "frames": 8, "outcome": "advanced"},
        {"episode_id": "e", "step": 2, "reward": 0.0, "frames": 8, "outcome": "cleared"},
    ]
    out = {(o["episode_id"], o["step"]): o["reward_rate"] for o in frame_horizon_returns(rows, 32)}
    assert out[("e", 2)] == round(FULL_SPEED_RATE * 24 / 32, 6)  # 24 frames of credited full speed
    assert out[("e", 0)] > 0 and all(
        o["reward"] == 24 / 3161 for o in frame_horizon_returns(rows, 32)[:1]
    )
