"""Offline unit tests for the composite reward (design-notes §5). Pure, no env, no API."""

from __future__ import annotations

from typesafe_mario.reward import DEFAULT, OBSTACLE_AWARE, RewardConfig, shaped_reward


def test_normal_progress_is_small_positive():
    # A full-run decision (~24px) reads as a modest positive, well under the terminal spikes.
    r = shaped_reward(dx=24.0, terminal=False, cleared=False, dead=False)
    assert 0.2 < r < 0.4


def test_standing_still_is_slightly_negative():
    # No progress → only the time penalty bites, so dawdling costs something.
    r = shaped_reward(dx=0.0, terminal=False, cleared=False, dead=False)
    assert r < 0.0
    assert r == -DEFAULT.time_penalty


def test_backward_is_negative():
    assert shaped_reward(dx=-24.0, terminal=False, cleared=False, dead=False) < 0.0


def test_death_saturates_negative():
    # A death dominates whatever small progress happened on the fatal decision.
    r = shaped_reward(dx=6.0, terminal=True, cleared=False, dead=True)
    assert r < -0.8


def test_flag_saturates_positive():
    r = shaped_reward(dx=6.0, terminal=True, cleared=True, dead=False)
    assert r == DEFAULT.clip  # clipped to +1


def test_flag_beats_death_beats_progress():
    flag = shaped_reward(dx=10.0, terminal=True, cleared=True, dead=False)
    step = shaped_reward(dx=24.0, terminal=False, cleared=False, dead=False)
    death = shaped_reward(dx=10.0, terminal=True, cleared=False, dead=True)
    assert death < step < flag


def test_output_is_always_clipped():
    cfg = RewardConfig(clip=1.0)
    for dx in (-10_000.0, -100.0, 0.0, 100.0, 10_000.0):
        for cleared, dead in ((False, False), (True, False), (False, True)):
            r = shaped_reward(dx=dx, terminal=True, cleared=cleared, dead=dead, config=cfg)
            assert -1.0 <= r <= 1.0


def test_non_terminal_dead_flag_is_ignored():
    # `dead` without `terminal` must not apply the death penalty (guards a mid-episode flicker).
    r_alive = shaped_reward(dx=24.0, terminal=False, cleared=False, dead=True)
    r_normal = shaped_reward(dx=24.0, terminal=False, cleared=False, dead=False)
    assert r_alive == r_normal


def test_config_validation():
    import pytest

    with pytest.raises(ValueError):
        RewardConfig(progress_scale=0.0)
    with pytest.raises(ValueError):
        RewardConfig(clip=0.0)
    with pytest.raises(ValueError):
        RewardConfig(obstacle_jump_bonus=-0.1)
    with pytest.raises(ValueError):
        RewardConfig(obstacle_stall_penalty=-0.1)
    with pytest.raises(ValueError):
        RewardConfig(stall_dx=0.0)


# --- obstacle-aware terms (findings #17) -------------------------------------------------


def test_default_config_ignores_obstacle_flags():
    # Defaults leave the obstacle terms inert → the pre-#17 baseline is byte-identical even
    # when the trigger inputs are supplied. This is what keeps the shaped arm a clean A/B twin.
    base = shaped_reward(dx=20.0, terminal=False, cleared=False, dead=False)
    with_flags = shaped_reward(
        dx=20.0,
        terminal=False,
        cleared=False,
        dead=False,
        obstacle_ahead=True,
        grounded=True,
        left_ground=True,
    )
    assert with_flags == base


def test_obstacle_jump_earns_the_bonus():
    r = shaped_reward(
        dx=20.0,
        terminal=False,
        cleared=False,
        dead=False,
        obstacle_ahead=True,
        grounded=True,
        left_ground=True,
        config=OBSTACLE_AWARE,
    )
    baseline = shaped_reward(dx=20.0, terminal=False, cleared=False, dead=False)
    assert r == baseline + OBSTACLE_AWARE.obstacle_jump_bonus


def test_obstacle_stall_earns_the_penalty():
    # Grounded, obstacle flagged, ~0 Δx, did not leave the ground → jammed at the pipe.
    r = shaped_reward(
        dx=3.0,
        terminal=False,
        cleared=False,
        dead=False,
        obstacle_ahead=True,
        grounded=True,
        left_ground=False,
        config=OBSTACLE_AWARE,
    )
    baseline = shaped_reward(dx=3.0, terminal=False, cleared=False, dead=False)
    assert r == baseline - OBSTACLE_AWARE.obstacle_stall_penalty


def test_airborne_crossing_is_not_a_stall():
    # Already airborne over the pipe with small Δx must NOT be penalised as a stall — the
    # terms are grounded-only, so a legitimate mid-jump crossing isn't punished.
    r = shaped_reward(
        dx=3.0,
        terminal=False,
        cleared=False,
        dead=False,
        obstacle_ahead=True,
        grounded=False,
        left_ground=False,
        config=OBSTACLE_AWARE,
    )
    assert r == shaped_reward(dx=3.0, terminal=False, cleared=False, dead=False)


def test_no_obstacle_means_no_obstacle_terms():
    # Open ground: leaving the ground earns no bonus (nothing to jump over).
    r = shaped_reward(
        dx=20.0,
        terminal=False,
        cleared=False,
        dead=False,
        obstacle_ahead=False,
        grounded=True,
        left_ground=True,
        config=OBSTACLE_AWARE,
    )
    assert r == shaped_reward(dx=20.0, terminal=False, cleared=False, dead=False)


def test_terminal_events_override_obstacle_terms():
    # A flag/death on the same decision must dominate; obstacle terms never touch a terminal.
    flag = shaped_reward(
        dx=6.0,
        terminal=True,
        cleared=True,
        dead=False,
        obstacle_ahead=True,
        grounded=True,
        left_ground=True,
        config=OBSTACLE_AWARE,
    )
    death = shaped_reward(
        dx=6.0,
        terminal=True,
        cleared=False,
        dead=True,
        obstacle_ahead=True,
        grounded=True,
        left_ground=False,
        config=OBSTACLE_AWARE,
    )
    assert flag == OBSTACLE_AWARE.clip  # still +1
    assert death < -0.8


def _jump_minus_run(config, *, dx_jump, dx_run):
    """Reward(commit the jump) − Reward(keep running), both from one grounded pipe state."""
    jump = shaped_reward(
        dx=dx_jump,
        terminal=False,
        cleared=False,
        dead=False,
        obstacle_ahead=True,
        grounded=True,
        left_ground=True,
        config=config,
    )
    run = shaped_reward(
        dx=dx_run,
        terminal=False,
        cleared=False,
        dead=False,
        obstacle_ahead=True,
        grounded=True,
        left_ground=False,
        config=config,
    )
    return jump - run


def test_pipe_separation_gate_decision_frame():
    """The finding-#17 go/no-go, part 1 — the *decision* frame.

    At the takeoff frame run-jump and run advance Mario by ~the same Δx (that ~equality is the
    whole reason the dense reward underfits the decision), so the run has NOT jammed yet and its
    Δx is above ``stall_dx``. The only term that separates the two here is the immediate jump
    bonus. Baseline leaves them ~tied; obstacle-aware restores a substantial, learnable gap. If
    this fails the lever is dead and the warm-start ingest is not worth spending.
    """
    dx_jump, dx_run = 22.0, 18.0  # ~equal immediate progress; run not yet jammed (18 > stall_dx)
    shaped_sep = _jump_minus_run(DEFAULT, dx_jump=dx_jump, dx_run=dx_run)
    obstacle_sep = _jump_minus_run(OBSTACLE_AWARE, dx_jump=dx_jump, dx_run=dx_run)
    assert shaped_sep < 0.10, f"baseline already separates ({shaped_sep:.3f}) — model wrong"
    assert obstacle_sep >= OBSTACLE_AWARE.obstacle_jump_bonus  # the bonus is the immediate signal
    assert obstacle_sep > 5 * shaped_sep, (
        f"immediate signal barely restored: {obstacle_sep:.3f} vs baseline {shaped_sep:.3f}"
    )


def test_pipe_separation_gate_jammed_frame():
    """Part 2 — a few frames later, once the run has jammed against the pipe (Δx→0 < stall_dx),
    the stall penalty compounds the jump bonus into a near-full separation. Together the two
    terms cover both the decision frame and the jam that follows it.
    """
    dx_jump, dx_run = 22.0, 2.0  # run is now stuck against the pipe
    obstacle_sep = _jump_minus_run(OBSTACLE_AWARE, dx_jump=dx_jump, dx_run=dx_run)
    assert obstacle_sep > 0.80, f"jammed-frame separation too weak ({obstacle_sep:.3f})"
