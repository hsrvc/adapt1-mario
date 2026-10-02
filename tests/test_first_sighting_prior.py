"""v3 arms: the first-sighting closing prior (findings #38) — an enemy whose relative velocity is unknown (0, the
parse where it first appears) is assumed to close at Mario's speed + a goomba's; v2 arms unchanged."""

import copy

import pytest

from typesafe_mario.adapt1_policy import (
    FULL_V2_FEATURE_NAMES,
    FULL_V3_FEATURE_NAMES,
    context_features,
)

_DEATH_STATE = {  # the gate run's x=1950 (findings #38): goomba 50 px ahead, first sighted in this parse
    "parse_frames": 53,
    "player": {
        "x": 1950,
        "y": 79,
        "grounded": True,
        "jump_phase": "grounded",
        "horizontal_speed_px_per_frame": 151,
        "vertical_speed_px_per_frame": 0,
    },
    "trajectory": {"airborne_frames": 0, "crossing_known_gap": False},
    "hazard": {
        "enemy_ahead": True,
        "nearest_enemy_distance_pixels": 50,
        "relative_velocity_x": 0,
        "projected_distance_after_reaction_pixels": 50,
        "estimated_contact_frames": None,
        "takeoff_deadline_frames": None,
        "jump_must_start_this_decision": False,
    },
    "terrain": {
        "obstacle_ahead": False,
        "obstacle_distance_tiles": None,
        "obstacle_height_tiles": 0,
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


def test_v2_still_reads_no_contact_on_first_sighting():
    f = context_features(_DEATH_STATE, FULL_V2_FEATURE_NAMES)
    assert f["contact_frames"] == 999.0 and f["takeoff_deadline"] == 999.0 and f["must_jump"] == 0.0
    assert f["enemy_dist"] == 50.0


def test_v3_assumes_the_enemy_closes_on_first_sighting():
    f = context_features(_DEATH_STATE, FULL_V3_FEATURE_NAMES)
    speed = 151 / 53  # 2.849 px/frame
    assert f["speed_x"] == pytest.approx(speed, abs=1e-3)
    assert f["contact_frames"] == pytest.approx(50 / (speed + 0.6), abs=0.01)  # 14.5 frames
    assert f["takeoff_deadline"] == pytest.approx(
        50 / (speed + 0.6) - 8, abs=0.01
    )  # 6.5: inside the horizon
    assert f["must_jump"] == 1.0
    assert f["enemy_dist"] == pytest.approx(
        50 - (speed + 0.6) * 8, abs=0.05
    )  # projected 8 frames on


def test_prior_fires_on_slot_reuse_garbage_too():
    """x=1799 of the v2 gate run: goomba 12 px ahead in a slot another enemy held on the previous parse,
    velocity +142 per 38 frames — v2 projected it 30 px further away and saw no contact."""
    s = copy.deepcopy(_DEATH_STATE)
    s["parse_frames"] = 38
    s["player"]["horizontal_speed_px_per_frame"] = 114
    s["hazard"].update(
        nearest_enemy_distance_pixels=12,
        relative_velocity_x=142,
        projected_distance_after_reaction_pixels=42,
    )
    v2 = context_features(s, FULL_V2_FEATURE_NAMES)
    v3 = context_features(s, FULL_V3_FEATURE_NAMES)
    assert (
        v2["enemy_dist"] == pytest.approx(41.9, abs=0.1) and v2["contact_frames"] == 999.0
    )  # inflated, contact-less v2 reading
    speed = 114 / 38
    assert v3["contact_frames"] == pytest.approx(
        12 / (speed + 0.6), abs=0.01
    )  # 3.3 frames: too late, and it says so
    assert v3["takeoff_deadline"] < 0 and v3["must_jump"] == 0.0 and v3["enemy_dist"] == 0.0


def test_prior_only_fires_when_the_velocity_is_unknown():
    s = copy.deepcopy(_DEATH_STATE)
    s["hazard"]["relative_velocity_x"] = -29  # measured closing over 53 frames
    v3 = context_features(s, FULL_V3_FEATURE_NAMES)
    v2 = context_features(s, FULL_V2_FEATURE_NAMES)
    assert v3["contact_frames"] == v2["contact_frames"] and v3["must_jump"] == v2["must_jump"]
