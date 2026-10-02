"""full_v3 / apex_v3 (2026-09-25, findings #36 correction): v2 + ``obstacle_height`` (tiles), the parser's
``obstacle_height_tiles`` that the rule teacher used for its jump size and no arm carried — so a 3-tile pipe and
2-1's 9-tile tower were indistinguishable except by distance."""

from __future__ import annotations

import copy
import json

import pytest

from typesafe_mario.adapt1_policy import (
    APEX_V2_FEATURE_NAMES,
    APEX_V3_FEATURE_NAMES,
    FEATURE_SETS,
    FULL_V2_FEATURE_NAMES,
    FULL_V3_FEATURE_NAMES,
    context_features,
    is_v2_arm,
)

_STATE = {
    "parse_frames": 8,
    "player": {
        "x": 560,
        "y": 79,
        "grounded": True,
        "jump_phase": "grounded",
        "horizontal_speed_px_per_frame": 24,
        "vertical_speed_px_per_frame": 0,
    },
    "trajectory": {"airborne_frames": 0, "crossing_known_gap": False},
    "hazard": {
        "enemy_ahead": False,
        "nearest_enemy_distance_pixels": None,
        "relative_velocity_x": None,
        "projected_distance_after_reaction_pixels": None,
        "estimated_contact_frames": None,
        "takeoff_deadline_frames": None,
        "jump_must_start_this_decision": False,
    },
    "terrain": {
        "obstacle_ahead": True,
        "obstacle_distance_tiles": 3,
        "obstacle_height_tiles": 3,
        "gap_ahead": False,
        "gap_distance_tiles": None,
        "clear_forward_tiles": 3,
        "observation_reliability": "high",
        "drop_distance_tiles": None,
        "drop_depth_tiles": 0,
        "floor_below_tiles": 0,
        "last_grounded_preview": {
            "obstacle_distance_tiles": 2,
            "obstacle_height_tiles": 4,
            "gap_distance_tiles": None,
            "gap_width_tiles_visible": 0,
        },
    },
    "reaction_timing": {"action_horizon_frames": 8, "last_inference_delay_frames": 0},
    "recent_control": {"outcome": "advanced"},
    "episode": {"stalled_frames": 0},
}


def test_v3_arms_are_v2_plus_obstacle_height():
    assert FULL_V3_FEATURE_NAMES == FULL_V2_FEATURE_NAMES + ("obstacle_height",)
    assert APEX_V3_FEATURE_NAMES == APEX_V2_FEATURE_NAMES + ("obstacle_height",)
    assert (
        FEATURE_SETS["full_v3"] is FULL_V3_FEATURE_NAMES
        and FEATURE_SETS["apex_v3"] is APEX_V3_FEATURE_NAMES
    )
    assert is_v2_arm(FULL_V3_FEATURE_NAMES) and is_v2_arm(APEX_V3_FEATURE_NAMES)
    assert "obstacle_height" not in FULL_V2_FEATURE_NAMES  # v2 unchanged


def test_grounded_reads_the_current_obstacle_height():
    f = context_features(_STATE, FULL_V3_FEATURE_NAMES)
    assert tuple(f) == FULL_V3_FEATURE_NAMES
    assert f["obstacle_height"] == 3.0 and f["obstacle_dist"] == 3.0
    # v2 on the same state: identical except the new key
    v2 = context_features(_STATE, FULL_V2_FEATURE_NAMES)
    assert {k: v for k, v in f.items() if k != "obstacle_height"} == v2


def test_airborne_full_arm_uses_the_preview_and_apex_arm_the_current_frame():
    s = copy.deepcopy(_STATE)
    s["player"]["grounded"] = False
    s["player"]["jump_phase"] = "rising"
    s["terrain"]["obstacle_distance_tiles"] = None
    s["terrain"]["obstacle_height_tiles"] = 0  # mid-air: nothing at foot level
    f = context_features(s, FULL_V3_FEATURE_NAMES)
    assert (
        f["obstacle_height"] == 4.0 and f["obstacle_dist"] == 2.0
    )  # both from the last grounded preview
    s["apex"] = {"vx": 3.0, "rise": 20, "height": 3}
    a = context_features(s, APEX_V3_FEATURE_NAMES, projected=False)
    assert a["obstacle_height"] == 0.0  # raw current frame, like the apex arm's other distances


def test_no_obstacle_reads_zero():
    s = copy.deepcopy(_STATE)
    s["terrain"].update(obstacle_ahead=False, obstacle_distance_tiles=None, obstacle_height_tiles=0)
    assert context_features(s, FULL_V3_FEATURE_NAMES)["obstacle_height"] == 0.0


pytest.importorskip("gym_super_mario_bros")

from typesafe_mario.cadence import CadenceParser
from typesafe_mario.macros import ApexChoice, Macro, execute
from typesafe_mario.runner import _unwrap_ram, create_mario_env


def test_1_1_obstacle_heights_along_the_lookahead_mainline():
    """Replays the #28-era 8-macro mainline (mario/runs bundle) and reads the feature at every grounded
    decision with an obstacle within 4 tiles: pipe 2 reads 3 tiles, the staircase top 4, the final
    staircase 6, the 1-tile steps 1 — increasing with the real geometry, never 0 with an obstacle ahead."""
    import gzip
    from pathlib import Path

    # runs/ sits beside tests/ in the public tree and two levels up (mario/runs) in this repo
    here = Path(__file__).resolve()
    roots = [
        p
        for p in (here.parents[1], here.parents[2])
        if (p / "runs" / "2026-09-22-apex-clear-28").is_dir()
    ]
    if not roots:
        pytest.skip("the 2026-09-22 apex-clear bundle is not shipped in this tree")
    src = next(
        p / "runs" / "2026-09-22-apex-clear-28" / "w11-coverage-v4.jsonl.gz"
        for p in (here.parents[1], here.parents[2])
        if (p / "runs" / "2026-09-22-apex-clear-28").is_dir()
    )
    with gzip.open(src, "rt") as fh:
        rows = [json.loads(l) for l in fh]
    main = sorted(
        [r for r in rows if r["episode_id"] == "main-1-1-s777-best"], key=lambda r: r["step"]
    )
    seq, i = [], 0
    while i < len(main):
        r, ap = main[i], None
        if i + 1 < len(main) and main[i + 1]["kind"] == "apex":
            ap = ApexChoice(main[i + 1]["policy"])
            i += 1
        seq.append((Macro(r["policy"]), ap))
        i += 1
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    parser = CadenceParser(decision_horizon_frames=8)
    seen = {}
    try:
        _f, info = env.reset(seed=777)
        ram = _unwrap_ram(env)
        snap = parser.parse(info, ram, previous_action=None)
        for m, ap in seq:
            f = context_features(snap.to_state(), FULL_V3_FEATURE_NAMES)
            if snap.grounded and f["obstacle_dist"] <= 4:
                seen[snap.x] = (f["obstacle_dist"], f["obstacle_height"])
            hook = (lambda _e, _a=ap: _a) if ap is not None else None
            ex = execute(env, m, grounded=snap.grounded, ram=ram, apex=hook)
            snap = parser.parse(ex.info, ram, previous_action=m.value, frames=ex.frames)
            if ex.terminated or snap.dead or snap.clear:
                break
    finally:
        env.close()
    assert snap.clear, seen
    assert seen[565] == (3.0, 3.0)  # pipe 2, three tiles, from three tiles away
    assert seen[2222] == (2.0, 4.0)  # the 4-tile staircase top
    assert seen[3129] == (3.0, 6.0)  # the final staircase
    assert all(h >= 1.0 for _d, h in seen.values())  # an obstacle ahead never reads 0 tall
