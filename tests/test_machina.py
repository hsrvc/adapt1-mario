"""Offline tests for the Machina adapter (mario/machina-arm.md) — no API, stub only."""

from __future__ import annotations

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.machina import (
    ACTION_DIMENSIONS,
    CONFIG_A,
    DECODER_VERSION,
    GOAL,
    STATE_DIMENSIONS,
    OfflineMachina,
    decode_command,
    encode_command,
    state_row,
)
from typesafe_mario.state import MarioStateParser


def test_decoder_v4_duty_cycles_with_a_dedicated_left_channel():
    assert DECODER_VERSION == 4 and ACTION_DIMENSIONS == 4
    assert decode_command([1, -1, 1, 1]) == [Action.RIGHT_RUN_JUMP] * 8
    assert (
        decode_command([1, -1, 1, 0]) == [Action.RIGHT_RUN_JUMP] * 4 + [Action.RIGHT_RUN] * 4
    )  # jump 0 -> 4 frames of A
    assert decode_command([1, -1, 1, 0.25]).count(Action.RIGHT_RUN_JUMP) == 5  # +0.25 = +1 frame
    assert (
        decode_command([0, -1, -1, -1]) == [Action.RIGHT] * 4 + [Action.NOOP] * 4
    )  # right 0 -> half the frames
    assert decode_command([-1, 1, 1, 1]) == [Action.LEFT] * 8  # left, never with B/A
    assert (
        decode_command([0, 1, -1, -1]) == [Action.RIGHT] * 4 + [Action.LEFT] * 4
    )  # right precedence, then left
    assert decode_command([1, 1, -1, -1]) == [Action.RIGHT] * 8  # full right hides left
    assert decode_command([-1, -1, -1, -1]) == [Action.NOOP] * 8
    assert decode_command([-1, -1, -1, 1]) == [Action.JUMP] * 8


def test_encoding_reports_what_ran_and_round_trips():
    assert encode_command([Action.RIGHT_RUN_JUMP] * 4 + [Action.RIGHT_RUN] * 4) == [
        1.0,
        -1.0,
        1.0,
        0.0,
    ]
    assert encode_command([Action.RIGHT] * 4 + [Action.LEFT] * 4) == [0.0, 1.0, -1.0, -1.0]
    assert encode_command([Action.NOOP] * 8) == [-1.0, -1.0, -1.0, -1.0]
    for row in (
        [1, -1, 1, 1],
        [1, -1, 1, 0],
        [0, 1, -1, -1],
        [-1, 0.5, 0.25, 0.25],
        [0.3, 0.3, 0.3, 0.3],
        [-1, 1, 1, 1],
    ):
        assert decode_command(encode_command(decode_command(row))) == decode_command(
            row
        )  # idempotent


def test_state_row_shape_and_goal_coordinate():
    info = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 1580,
        "y_pos": 79,
        "y_pixel": 79,
        "progress": 1580,
        "time": 390,
        "life": 2,
        "status": "small",
    }
    row = state_row(MarioStateParser().parse(info, None))
    assert len(row) == STATE_DIMENSIONS == CONFIG_A["state_dimensions"]
    assert abs(row[0] - 1580 / 3161) < 1e-6 and CONFIG_A["goal_indices"] == [0] and GOAL == [1.0]


def test_offline_stub_enforces_the_observation_contract():
    m = OfflineMachina()
    assert m.trajectory("state")["status"] == "not_configured"
    m.trajectory("configure", {"config": CONFIG_A})
    p = m.trajectory(
        "propose", {"state": [0.0] * STATE_DIMENSIONS, "goal": GOAL, "mode": "explore"}
    )
    assert len(p["actions"]) == CONFIG_A["horizon"] and len(p["actions"][0]) == ACTION_DIMENSIONS
    with pytest.raises(AssertionError):
        m.trajectory(
            "propose", {"state": [0.0] * STATE_DIMENSIONS, "goal": GOAL, "mode": "explore"}
        )
    ok = {
        "decision_id": p["decision_id"],
        "actions": [[1.0, -1.0, 1.0, 0.0]] * 3,
        "states": [[0.0] * STATE_DIMENSIONS] * 4,
        "step_outcomes": [0.001] * 3,
        "outcome": 0.01,
    }
    with pytest.raises(AssertionError):
        m.trajectory("observe", {**ok, "states": [[0.0] * STATE_DIMENSIONS] * 3})
    with pytest.raises(AssertionError):
        m.trajectory("observe", {**ok, "step_outcomes": [0.0]})
    assert m.trajectory("observe", ok)["status"] == "learned"
