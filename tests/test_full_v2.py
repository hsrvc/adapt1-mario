"""full_v2 / apex_v2 (2026-09-24, findings #33): the feature layer in per-frame units.

The parser's ``dx``/``dy``/``relative_velocity_x`` are per-parse deltas; every decision loop
parses once per decision, so the v1 arms' ``speed_x`` is the previous macro's displacement
and the enemy projection is computed from a closing speed 8–50× too large (``enemy_dist`` 0,
deadline negative, ``must_jump`` never for an enemy). v2 divides by the parse interval that
``cadence.CadenceParser`` stamps on the snapshot. ``reliability`` (a relabelling of
``grounded``) is dropped.
"""

from __future__ import annotations

import copy

import pytest

from typesafe_mario.adapt1_policy import (
    APEX_EXTRA_NAMES,
    APEX_V2_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SETS,
    FULL_V2_FEATURE_NAMES,
    PERCEPTION_FEATURE_NAMES,
    context_features,
    is_v2_arm,
    snapshot_features_v2,
)

# One grounded decision of 8 frames at full run speed: Mario moved 24 px, the goomba 127 px
# ahead closed 29 px over the interval (Mario's 24 + its own 5 → 3.625 px/frame). These are
# the exact per-decision numbers the audit measured on 1-1 seed 777 (findings #33).
_GROUNDED_8F = {
    "parse_frames": 8,
    "player": {
        "x": 213,
        "y": 79,
        "grounded": True,
        "jump_phase": "grounded",
        "horizontal_speed_px_per_frame": 24,
        "vertical_speed_px_per_frame": 0,
    },
    "trajectory": {"airborne_frames": 0, "crossing_known_gap": False},
    "hazard": {
        "enemy_ahead": True,
        "nearest_enemy_distance_pixels": 127,
        "relative_velocity_x": -29,
        # what the v1 parser derived from -29 "px/frame": contact 4, deadline -4, projected 0
        "projected_distance_after_reaction_pixels": 0,
        "estimated_contact_frames": 4,
        "takeoff_deadline_frames": -4,
        "jump_must_start_this_decision": False,
    },
    "terrain": {
        "obstacle_ahead": False,
        "obstacle_distance_tiles": None,
        "gap_ahead": False,
        "gap_distance_tiles": None,
        "clear_forward_tiles": 8,
        "observation_reliability": "high",
        "drop_distance_tiles": None,
        "drop_depth_tiles": 0,
        "floor_below_tiles": 0,
        "last_grounded_preview": {},
    },
    "reaction_timing": {"action_horizon_frames": 8, "last_inference_delay_frames": 0},
    "recent_control": {"outcome": "advanced"},
    "episode": {"stalled_frames": 0},
}


def test_v2_arms_are_the_v1_arms_without_reliability_plus_last_frames():
    assert FULL_V2_FEATURE_NAMES == tuple(n for n in FEATURE_NAMES if n != "reliability") + (
        "last_frames",
    )
    assert APEX_V2_FEATURE_NAMES == (
        tuple(n for n in PERCEPTION_FEATURE_NAMES if n != "reliability") + APEX_EXTRA_NAMES
    )
    assert "last_frames" not in APEX_V2_FEATURE_NAMES  # the apex arm has it as apex_rise
    assert len(FULL_V2_FEATURE_NAMES) == 22 and len(APEX_V2_FEATURE_NAMES) == 17
    assert FEATURE_SETS["full_v2"] is FULL_V2_FEATURE_NAMES
    assert FEATURE_SETS["apex_v2"] is APEX_V2_FEATURE_NAMES
    # the marker: every v1 arm carries reliability, no v2-family arm (v2, v3) does
    for name, arm in FEATURE_SETS.items():
        assert is_v2_arm(arm) == (name.endswith(("_v2", "_v3"))), name


def test_speeds_are_per_frame_and_the_enemy_projection_is_recomputed():
    f = context_features(_GROUNDED_8F, FULL_V2_FEATURE_NAMES)
    assert tuple(f) == FULL_V2_FEATURE_NAMES
    assert "reliability" not in f
    assert f["speed_x"] == pytest.approx(3.0)  # 24 px / 8 frames, not 24
    assert f["speed_y"] == 0.0
    assert f["last_frames"] == 8.0  # the previous macro's duration, the factor v1 multiplied in
    # goomba 127 px away closing at 3.625 px/frame: contact in 35 frames, not 4
    assert f["contact_frames"] == pytest.approx(127 / 3.625, abs=1e-3)
    # deadline = contact - delay(0) - 8 frames clearance = +27, not -4
    assert f["takeoff_deadline"] == pytest.approx(127 / 3.625 - 8, abs=1e-3)
    # projected where it WILL be after the 8-frame horizon: 127 - 29 = 98, not 0
    assert f["enemy_dist"] == pytest.approx(98.0)
    assert f["must_jump"] == 0.0  # the deadline is far outside the 8-frame horizon


def test_must_jump_fires_when_the_per_frame_deadline_is_inside_the_horizon():
    state = copy.deepcopy(_GROUNDED_8F)
    state["hazard"]["nearest_enemy_distance_pixels"] = 40  # 40 / 3.625 = 11.03 frames to contact
    f = context_features(state, FULL_V2_FEATURE_NAMES)
    assert f["takeoff_deadline"] == pytest.approx(40 / 3.625 - 8, abs=1e-3)  # 3.03: jump now
    assert f["must_jump"] == 1.0
    assert f["enemy_dist"] == pytest.approx(11.0)  # 40 - 29


def test_v1_full_arm_is_unchanged_on_the_same_state():
    """The frozen v1 arm keeps reporting what the #28/#32 domains were trained on."""
    v1 = context_features(_GROUNDED_8F, FEATURE_NAMES)
    assert v1["speed_x"] == 24.0 and v1["enemy_dist"] == 0.0
    assert v1["contact_frames"] == 4.0 and v1["takeoff_deadline"] == -4.0
    assert v1["reliability"] == "high"


def test_terrain_mirror_uses_the_per_frame_speed():
    state = copy.deepcopy(_GROUNDED_8F)
    state["hazard"] = {
        "enemy_ahead": False,
        "nearest_enemy_distance_pixels": None,
        "relative_velocity_x": None,
        "projected_distance_after_reaction_pixels": None,
        "estimated_contact_frames": None,
        "takeoff_deadline_frames": None,
        "jump_must_start_this_decision": False,
    }
    state["terrain"]["obstacle_ahead"] = True
    state["terrain"]["obstacle_distance_tiles"] = 3  # a pipe 48 px ahead
    v2 = context_features(state, FULL_V2_FEATURE_NAMES)
    v1 = context_features(state, FEATURE_NAMES)
    assert v1["contact_frames"] == pytest.approx(48 / 24)  # 2 "frames": the v1 number
    assert v2["contact_frames"] == pytest.approx(48 / 3.0)  # 16 frames at 3 px/frame
    assert v2["takeoff_deadline"] == pytest.approx(16 - 8)
    assert v2["must_jump"] == 1.0 and v1["must_jump"] == 0.0
    assert v2["enemy_dist"] == 999.0  # no enemy: the sentinel, as in v1


def test_apex_v2_keeps_raw_enemy_distance_and_divides_the_rise():
    state = copy.deepcopy(_GROUNDED_8F)
    state["parse_frames"] = 20  # the rise: takeoff parse → apex parse
    state["player"].update(
        {
            "grounded": False,
            "jump_phase": "apex",
            "horizontal_speed_px_per_frame": 60,
            "vertical_speed_px_per_frame": 64,
        }
    )
    state["trajectory"]["airborne_frames"] = 1
    state["apex"] = {"vx": 3.0, "rise": 20, "height": 4}
    f = context_features(state, APEX_V2_FEATURE_NAMES, projected=False)
    assert tuple(f) == APEX_V2_FEATURE_NAMES
    assert f["speed_x"] == pytest.approx(3.0) and f["speed_y"] == pytest.approx(3.2)
    assert f["enemy_dist"] == 127.0  # raw current distance, as the v1 apex arm
    assert f["apex_rise"] == 20.0 and f["apex_height"] == 4.0


def test_first_parse_of_an_episode_has_no_interval():
    state = copy.deepcopy(_GROUNDED_8F)
    state["parse_frames"] = None
    state["player"]["horizontal_speed_px_per_frame"] = 0
    state["hazard"]["relative_velocity_x"] = 0
    f = context_features(state, FULL_V2_FEATURE_NAMES)
    assert f["speed_x"] == 0.0 and f["contact_frames"] == 999.0 and f["must_jump"] == 0.0
    assert f["enemy_dist"] == 127.0  # no velocity yet → projected = current
    assert f["last_frames"] == 0.0  # episode start


def test_a_plain_parser_snapshot_is_refused_by_the_v2_arms():
    state = copy.deepcopy(_GROUNDED_8F)
    del state["parse_frames"]
    with pytest.raises(ValueError, match="CadenceParser"):
        snapshot_features_v2(state)
    with pytest.raises(ValueError, match="CadenceParser"):
        context_features(state, FULL_V2_FEATURE_NAMES)
    # the v1 arms do not care
    assert context_features(state, FEATURE_NAMES)["speed_x"] == 24.0


# ---------------------------------------------------------------------------------------------
# Emulator-pinned (skipped without gym_super_mario_bros): the numbers of findings #33.

pytest.importorskip("gym_super_mario_bros")

from typesafe_mario.cadence import CadenceParser, CadenceSnapshot
from typesafe_mario.macros import Macro, execute
from typesafe_mario.runner import _unwrap_ram, create_mario_env


def _run_right(env_id: str, macros: int, macro: Macro = Macro.RIGHT_RUN):
    env = create_mario_env(env_id, render_mode="rgb_array")
    parser = CadenceParser(decision_horizon_frames=8)
    rows = []
    try:
        _f, info = env.reset(seed=777)
        ram = _unwrap_ram(env)
        snap = parser.parse(info, ram, previous_action=None)
        rows.append((snap, context_features(snap.to_state(), FULL_V2_FEATURE_NAMES), None))
        for _ in range(macros):
            ex = execute(env, macro, grounded=snap.grounded, ram=ram)
            snap = parser.parse(ex.info, ram, previous_action=macro.value, frames=ex.frames)
            rows.append((snap, context_features(snap.to_state(), FULL_V2_FEATURE_NAMES), ex))
            if ex.terminated or snap.dead:
                break
    finally:
        env.close()
    return rows


def test_cadence_parser_requires_the_interval_after_the_first_parse():
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    try:
        _f, info = env.reset(seed=777)
        ram = _unwrap_ram(env)
        parser = CadenceParser(decision_horizon_frames=8)
        first = parser.parse(info, ram, previous_action=None)
        assert isinstance(first, CadenceSnapshot) and first.parse_frames is None
        assert first.to_state()["parse_frames"] is None
        ex = execute(env, Macro.RIGHT_RUN, grounded=True, ram=ram)
        with pytest.raises(ValueError, match="frames="):
            parser.parse(ex.info, ram, previous_action="right_run")
        snap = parser.parse(ex.info, ram, previous_action="right_run", frames=ex.frames)
        assert snap.parse_frames == ex.frames == 8
        zero = parser.parse(
            ex.info, ram, previous_action="right_run", frames=0
        )  # same frame parsed twice
        assert zero.parse_frames == 0
        f0 = context_features(zero.to_state(), FULL_V2_FEATURE_NAMES)
        assert f0["speed_x"] == 0.0 and f0["last_frames"] == 0.0 and f0["contact_frames"] == 999.0
        with pytest.raises(ValueError, match=">= 0"):
            parser.parse(ex.info, ram, previous_action="right_run", frames=-1)
        parser.reset()  # a new episode: the first parse needs no interval again
        _f, info = env.reset(seed=777)
        assert parser.parse(info, _unwrap_ram(env), previous_action=None).parse_frames is None
    finally:
        env.close()


def test_1_1_run_speed_reads_3_px_per_frame_and_the_first_goomba_is_projected():
    """1-1 seed 777, RIGHT_RUN ×13: v1 read speed_x 24 and enemy_dist 0 at every decision with
    the goomba ahead (127 → 11 px); v2 reads 3.0 px/frame and a closing goomba."""
    rows = _run_right("SuperMarioBros-1-1-v0", 13)
    speeds = [f["speed_x"] for _s, f, ex in rows[5:12]]  # at full speed, before the death step
    assert all(abs(v - 3.0) < 0.01 for v in speeds), speeds
    assert rows[0][1]["last_frames"] == 0.0
    assert all(f["last_frames"] == float(ex.frames) == 8.0 for _s, f, ex in rows[1:12])
    # v1's speed_x = v2's speed_x × last_frames (the product it carried as one number)
    v1 = [context_features(s.to_state(), FEATURE_NAMES)["speed_x"] for s, _f, _e in rows[1:12]]
    assert all(
        abs(a - f["speed_x"] * f["last_frames"]) < 0.05 for a, (_s, f, _e) in zip(v1, rows[1:12])
    )
    with_goomba = [(s, f) for s, f, _ in rows if f["enemy_ahead"] == 1.0 and s.grounded]
    assert len(with_goomba) >= 4
    for s, f in with_goomba:
        raw = min(e.dx_pixels for e in s.enemies if e.dx_pixels >= 0)
        assert 0 < f["enemy_dist"] < raw or raw <= 29, (raw, f["enemy_dist"])  # closing, not 0
        assert f["takeoff_deadline"] != -4.0
    # the exact audit point: 127 px away, deadline ≈ +27 frames and no must_jump yet
    far = next((s, f) for s, f in with_goomba if 120 <= min(e.dx_pixels for e in s.enemies) <= 134)
    assert far[1]["contact_frames"] == pytest.approx(35, abs=1.5)
    assert far[1]["takeoff_deadline"] == pytest.approx(27, abs=1.5)
    # and the decision where a jump would have saved Mario: must_jump fires on the way in
    assert any(f["must_jump"] == 1.0 for _s, f in with_goomba), [
        f["takeoff_deadline"] for _s, f in with_goomba
    ]


def test_1_2_first_goomba_at_151_px_is_a_closing_enemy_not_contact_now():
    """1-2 seed 777, RIGHT_RUN: Mario drops in with a goomba pair ahead (findings #33 — v1 read
    enemy_dist 0 / deadline -1 at the 151-px sighting). v2: at 151 px the goomba is 57 frames
    from contact (deadline +49), and must_jump fires once the deadline enters the 8-frame
    horizon (36 px, deadline ≈ 1.9) — the decision where a jump would have saved Mario."""
    rows = _run_right("SuperMarioBros-1-2-v0", 6)
    grounded = [
        (s, f) for s, f, ex in rows if ex is not None and s.grounded and f["enemy_ahead"] == 1.0
    ]
    by_raw = {min(e.dx_pixels for e in s.enemies if e.dx_pixels >= 0): f for s, f in grounded}
    assert 151 in by_raw, sorted(by_raw)
    at_151 = by_raw[151]
    assert at_151["enemy_dist"] == pytest.approx(130.0, abs=2)  # projected 8 frames ahead, not 0
    assert at_151["contact_frames"] == pytest.approx(57.5, abs=1.5)
    assert at_151["takeoff_deadline"] == pytest.approx(49.5, abs=1.5)
    assert at_151["must_jump"] == 0.0
    fired = [(raw, f["takeoff_deadline"]) for raw, f in by_raw.items() if f["must_jump"] == 1.0]
    assert fired == [(36, pytest.approx(1.93, abs=0.1))], fired
    # v1 on the same snapshots: enemy_dist 0 and a negative deadline at every sighting
    v1 = [context_features(s.to_state(), FEATURE_NAMES) for s, _f in grounded]
    assert all(f["enemy_dist"] == 0.0 and f["takeoff_deadline"] < 0 for f in v1)
