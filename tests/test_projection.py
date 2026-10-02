"""Move A (2026-09-18): the terrain projection ported into the Adapt-1 feature layer.

Locks the fix for the measured bug (probe_pipe.py): the shared parser computes the
contact/takeoff/jump-now projection for ENEMIES only, so `must_jump`/`contact_frames` stayed
silent at a pipe (a static terrain obstacle) even though a forward jump is required. These
tests assert the terrain analogue fires — with lead time — and is preview-aware while airborne.
"""

from __future__ import annotations

from typesafe_mario.adapt1_policy import (
    FEATURE_SETS,
    _terrain_distances,
    _terrain_hazard,
    context_features,
)

# A full state: airborne, a closing enemy (60px now → 32px projected) and an obstacle only the
# last-grounded preview still sees. Used to assert the full arm gets the projection/preview
# enrichment while the perception arm stays on raw current-frame facts (no silent confound).
_FULL_STATE = {
    "player": {
        "grounded": False,
        "jump_phase": "rising",
        "horizontal_speed_px_per_frame": 8.0,
        "vertical_speed_px_per_frame": -5.0,
    },
    "trajectory": {"airborne_frames": 3, "crossing_known_gap": False},
    "hazard": {
        "enemy_ahead": True,
        "nearest_enemy_distance_pixels": 60,
        "projected_distance_after_reaction_pixels": 32,
        "estimated_contact_frames": 7,
        "takeoff_deadline_frames": None,
        "jump_must_start_this_decision": False,
    },
    "terrain": {
        "obstacle_ahead": False,
        "obstacle_distance_tiles": None,
        "gap_ahead": False,
        "gap_distance_tiles": None,
        "clear_forward_tiles": 2,
        "observation_reliability": "low_airborne",
        "last_grounded_preview": {"obstacle_distance_tiles": 2, "gap_distance_tiles": None},
    },
    "reaction_timing": {
        "action_horizon_frames": 8,
        "last_inference_delay_frames": 2,
        "total_reaction_horizon_frames": 10,
    },
    "recent_control": {"outcome": "jump_in_progress"},
    "episode": {"stalled_frames": 1},
}


def test_full_arm_gets_projected_and_preview_distances():
    f = context_features(_FULL_STATE, FEATURE_SETS["full"])
    assert f["enemy_dist"] == 32.0  # projected after reaction, not the raw 60
    assert f["obstacle_dist"] == 2.0  # from last_grounded_preview, not blank


def test_perception_arm_stays_raw_current_frame():
    # The purity guard: the "did it learn the timing itself?" arm must not inherit the
    # parser's projection/preview enrichment, or it stops being a clean test.
    p = context_features(_FULL_STATE, FEATURE_SETS["perception"])
    assert p["enemy_dist"] == 60.0  # raw current distance
    assert p["obstacle_dist"] == 999.0  # blank while airborne (no preview fallback)


# decision_horizon_frames=8, last_inference_delay_frames=2 → a 3-tile (48px) obstacle at
# 8px/frame contacts in 6 frames; takeoff = 6 - 2 - 8 = -4 (already inside/late). At 4px/frame
# it contacts in 12 frames; takeoff = 12 - 2 - 8 = 2 (fire this decision).
RT = {
    "action_horizon_frames": 8,
    "last_inference_delay_frames": 2,
    "total_reaction_horizon_frames": 10,
}


def _state(*, grounded, speed_x, obstacle_tiles=None, gap_tiles=None, preview=None):
    return {
        "player": {"grounded": grounded, "horizontal_speed_px_per_frame": speed_x},
        "terrain": {
            "obstacle_distance_tiles": obstacle_tiles,
            "gap_distance_tiles": gap_tiles,
            "last_grounded_preview": preview or {},
        },
        "reaction_timing": RT,
    }


def test_terrain_hazard_fires_for_a_closing_obstacle():
    contact, takeoff = _terrain_hazard(_state(grounded=True, speed_x=4.0, obstacle_tiles=3))
    assert contact == (3 * 16.0) / 4.0  # 12 frames to contact
    assert takeoff == 12.0 - 2.0 - 8.0  # 2 frames of lead — jump now


def test_terrain_hazard_is_silent_when_not_closing():
    # speed_x <= 0 (bouncing off / stalled) → no terrain takeoff pressure
    assert _terrain_hazard(_state(grounded=True, speed_x=-5.0, obstacle_tiles=1)) == (None, None)
    # nothing ahead at all
    assert _terrain_hazard(_state(grounded=True, speed_x=8.0)) == (None, None)


def test_terrain_hazard_takes_the_nearer_of_obstacle_and_gap():
    contact, _ = _terrain_hazard(_state(grounded=True, speed_x=8.0, obstacle_tiles=5, gap_tiles=2))
    assert contact == (2 * 16.0) / 8.0  # the gap (2 tiles) is nearer than the pipe (5)


def test_distances_use_current_geometry_when_grounded():
    assert _terrain_distances(
        _state(grounded=True, speed_x=8.0, obstacle_tiles=3, gap_tiles=7)
    ) == (3, 7)


def test_distances_fall_back_to_preview_when_airborne():
    # current frame sees nothing (airborne, low reliability) but the last grounded preview does —
    # this is the Tier-2 fix for obstacle_dist collapsing to the sentinel mid-jump.
    st = _state(
        grounded=False,
        speed_x=8.0,
        obstacle_tiles=None,
        preview={"obstacle_distance_tiles": 2, "gap_distance_tiles": None},
    )
    assert _terrain_distances(st) == (2, None)
