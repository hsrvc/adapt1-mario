"""Offline tests for the diversified heuristic teacher (no emulator, no network)."""

from __future__ import annotations

from dataclasses import replace

from typesafe_mario.actions import Action
from typesafe_mario.policy import DiversifiedHeuristicPolicy
from typesafe_mario.state import EnemyObservation, MarioStateParser


def _clear_snapshot():
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
        "time": 390,
        "progress": 100,
    }
    return MarioStateParser().parse(info, None, previous_action="right_run")


def _enemy_snapshot(dx: int):
    base = _clear_snapshot()
    enemy = EnemyObservation(
        slot=0, kind_id=6, kind="goomba", dx_pixels=dx, dy_pixels=0, relative_velocity_x=-2
    )
    return replace(base, enemies=(enemy,), grounded=True)


def test_deterministic_rule_runs_right_on_a_clear_path():
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0)
    decision = teacher.choose(_clear_snapshot(), tuple(Action))
    assert decision.action == Action.RIGHT_RUN


def test_deterministic_rule_jumps_for_an_enemy_in_the_stomp_window():
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0)
    decision = teacher.choose(_enemy_snapshot(dx=40), tuple(Action))
    assert decision.action == Action.RIGHT_RUN_JUMP  # 24..72px stomp window -> jump


def test_far_enemy_does_not_force_a_jump():
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0)
    decision = teacher.choose(_enemy_snapshot(dx=200), tuple(Action))
    assert decision.action == Action.RIGHT_RUN


def test_very_close_enemy_does_not_force_a_late_jump():
    # Below the stomp window (24px): a fresh takeoff here lands Mario *into* the enemy.
    # Measured 2026-09-18 — the old `< 56` rule died deterministically at x~694 for exactly
    # this reason; the window now leaves the too-close case to run.
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0)
    decision = teacher.choose(_enemy_snapshot(dx=12), tuple(Action))
    assert decision.action == Action.RIGHT_RUN


def test_epsilon_injects_diverse_but_legal_actions():
    teacher = DiversifiedHeuristicPolicy(epsilon=1.0, rng_seed=0)  # always explore
    snap = _clear_snapshot()
    chosen = {teacher.choose(snap, tuple(Action)).action for _ in range(40)}
    assert chosen <= set(Action)
    assert len(chosen) > 1  # exploration actually diversifies


def test_probabilities_are_normalised():
    teacher = DiversifiedHeuristicPolicy(epsilon=0.2, rng_seed=1)
    decision = teacher.choose(_clear_snapshot(), tuple(Action))
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
    assert decision.probabilities[decision.action.value] == 1.0
