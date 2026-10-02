"""Offline unit tests for the tactical layer (findings #18, Option 2). Pure; no env, no API.

The coarse encoders and event trigger are the contract the (future) Adapt-1/Jev decision layer
sees, so they are pinned here. Skill *execution* is validated by `run_tactical_episode` in the
integration smoke, not unit-tested (it needs the emulator)."""

from __future__ import annotations

from typesafe_mario.tactical import (
    ENEMY_STOMP_MAX_PX,
    ENEMY_STOMP_MIN_PX,
    HeuristicTacticalPolicy,
    TacticalSkill,
    _bucket_enemy,
    _bucket_obstacle,
    encode_tactical_state,
    is_tactical_event,
)


class FakeSnapshot:
    """Minimal stand-in exposing only what the encoders read."""

    def __init__(self, *, grounded=True, obstacle_tiles=None, gap=False, enemy_px=None):
        self.grounded = grounded
        self._nav = {"obstacle_distance_tiles": obstacle_tiles, "gap_ahead": gap}
        self._threat = {"nearest_enemy_distance_pixels": enemy_px}

    def navigation_features(self):
        return self._nav

    def threat_features(self):
        return self._threat


def test_bucket_obstacle():
    assert _bucket_obstacle(None) == "none"
    assert _bucket_obstacle(2) == "near"
    assert _bucket_obstacle(3) == "near"
    assert _bucket_obstacle(4) == "far"
    assert _bucket_obstacle(9) == "none"


def test_bucket_enemy_stomp_window():
    assert _bucket_enemy(None) == "none"
    assert _bucket_enemy(-10) == "none"
    assert _bucket_enemy(ENEMY_STOMP_MIN_PX - 1) == "too_close"
    assert _bucket_enemy(ENEMY_STOMP_MIN_PX) == "stomp_window"
    assert _bucket_enemy(ENEMY_STOMP_MAX_PX) == "stomp_window"
    assert _bucket_enemy(ENEMY_STOMP_MAX_PX + 1) == "approaching"
    assert _bucket_enemy(200) == "none"


def test_encode_tactical_state():
    st = encode_tactical_state(FakeSnapshot(grounded=True, obstacle_tiles=2, enemy_px=48))
    assert st == {"mario": "grounded", "obstacle": "near", "gap": "none", "enemy": "stomp_window"}
    air = encode_tactical_state(FakeSnapshot(grounded=False))
    assert air["mario"] == "airborne"


def test_is_tactical_event_only_when_grounded_and_actionable():
    # airborne is never a decision point
    assert is_tactical_event(FakeSnapshot(grounded=False, obstacle_tiles=1)) is False
    # grounded + near obstacle / gap / stomp-window enemy each trigger
    assert is_tactical_event(FakeSnapshot(grounded=True, obstacle_tiles=1)) is True
    assert is_tactical_event(FakeSnapshot(grounded=True, gap=True)) is True
    assert is_tactical_event(FakeSnapshot(grounded=True, enemy_px=48)) is True
    # grounded but nothing in reach ⇒ no decision due (ADVANCE keeps running)
    assert is_tactical_event(FakeSnapshot(grounded=True)) is False
    # a far obstacle is not yet a trigger
    assert is_tactical_event(FakeSnapshot(grounded=True, obstacle_tiles=5)) is False


def test_heuristic_policy_maps_state_to_skill():
    pol = HeuristicTacticalPolicy()
    assert (
        pol.choose(None, {"mario": "airborne", "obstacle": "near", "gap": "none", "enemy": "none"})
        is TacticalSkill.ADVANCE
    )
    assert (
        pol.choose(None, {"mario": "grounded", "obstacle": "near", "gap": "none", "enemy": "none"})
        is TacticalSkill.JUMP_CLEAR
    )
    assert (
        pol.choose(None, {"mario": "grounded", "obstacle": "none", "gap": "ahead", "enemy": "none"})
        is TacticalSkill.JUMP_CLEAR
    )
    assert (
        pol.choose(
            None, {"mario": "grounded", "obstacle": "none", "gap": "none", "enemy": "stomp_window"}
        )
        is TacticalSkill.JUMP_CLEAR
    )
    assert (
        pol.choose(
            None, {"mario": "grounded", "obstacle": "none", "gap": "none", "enemy": "approaching"}
        )
        is TacticalSkill.ADVANCE
    )
