"""Offline tests for the grounded-cadence macros (findings #21) — a fake emulator, no API."""

from __future__ import annotations

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.macros import (
    MACRO_PLANS,
    MAX_MACRO_FRAMES,
    Macro,
    action_set,
    execute,
    is_jump,
)
from typesafe_mario.policy import DiversifiedHeuristicPolicy
from typesafe_mario.reward import DEFAULT, GROUNDED, RewardConfig, shaped_reward
from typesafe_mario.state import MarioStateParser

JUMP_INDICES = {ACTION_TO_INDEX[a] for a in (Action.RIGHT_JUMP, Action.RIGHT_RUN_JUMP, Action.JUMP)}
RUN = ACTION_TO_INDEX[Action.RIGHT_RUN]
RUN_JUMP = ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP]


class FakeEnv:
    """Scripted NES: a jump press while grounded makes RAM float-state 1 for `air_frames`
    frames; `walk_off_at` makes a non-jump walk off a ledge; `terminate_at` ends the episode."""

    def __init__(
        self,
        *,
        air_frames: int = 40,
        walk_off_at: int | None = None,
        terminate_at: int | None = None,
        stuck_airborne: bool = False,
    ) -> None:
        self.ram = bytearray(0x0800)
        self.steps: list[int] = []
        self.air = 0
        self.air_frames = air_frames
        self.walk_off_at = walk_off_at
        self.terminate_at = terminate_at
        self.stuck = stuck_airborne

    def step(self, index: int):
        self.steps.append(index)
        n = len(self.steps)
        if index in JUMP_INDICES and self.ram[0x1D] == 0 and self.air == 0:
            self.air = self.air_frames
        if self.walk_off_at is not None and n == self.walk_off_at:
            self.air = self.air_frames
        if self.air > 0:
            self.air -= 1
        self.ram[0x1D] = 1 if (self.air > 0 or self.stuck) else 0
        done = self.terminate_at is not None and n >= self.terminate_at
        return None, 1.0, done, False, {"x_pos": n}


def test_action_sets_per_cadence():
    assert action_set("frame") == tuple(Action)
    assert action_set("grounded") == tuple(Macro)
    assert all(MACRO_PLANS[m].hold_frames in (0, 8, 16, 24, 32) for m in Macro)
    assert is_jump(Macro.RIGHT_RUN_JUMP_FULL) and not is_jump(Macro.RIGHT_RUN)
    assert is_jump(Action.JUMP) and not is_jump(Action.RIGHT)


def test_frame_cadence_reproduces_the_old_loop_release_edge():
    env = FakeEnv()
    ex = execute(env, Action.RIGHT_RUN_JUMP, grounded=True, frames_per_decision=8, ram=env.ram)
    assert ex.frames == 8
    assert env.steps == [RUN] + [RUN_JUMP] * 7  # frame-0 button-up edge, then hold
    assert ex.total_reward == 8.0


def test_frame_cadence_airborne_jump_holds_all_frames():
    env = FakeEnv()
    execute(env, Action.RIGHT_RUN_JUMP, grounded=False, frames_per_decision=8, ram=env.ram)
    assert env.steps == [RUN_JUMP] * 8


def test_macro_holds_a_for_its_size_then_carries_until_landed():
    env = FakeEnv(air_frames=40)
    ex = execute(env, Macro.RIGHT_RUN_JUMP_MID, grounded=True, ram=env.ram)
    assert env.steps[0] == RUN  # release edge
    assert env.steps[1:17] == [RUN_JUMP] * 16  # A held exactly 16 frames
    assert set(env.steps[17:]) == {RUN}  # then the carry combo, no A
    assert env.ram[0x1D] == 0  # ended on the ground
    assert 17 < ex.frames <= 1 + 40 + 1  # ran to the landing, no longer
    assert ex.terminated is False


def test_jump_sizes_are_distinct():
    holds = [
        sum(1 for s in _run(m).steps if s in JUMP_INDICES)
        for m in (Macro.RIGHT_RUN_JUMP_SHORT, Macro.RIGHT_RUN_JUMP_MID, Macro.RIGHT_RUN_JUMP_FULL)
    ]
    assert holds == [8, 16, 32]


def _run(macro: Macro) -> FakeEnv:
    env = FakeEnv(air_frames=60)
    execute(env, macro, grounded=True, ram=env.ram)
    return env


def test_non_jump_macro_runs_to_the_landing_when_it_walks_off_a_ledge():
    env = FakeEnv(air_frames=30, walk_off_at=3)
    ex = execute(env, Macro.RIGHT_RUN, grounded=True, frames_per_decision=8, ram=env.ram)
    assert ex.frames > 8 and env.ram[0x1D] == 0
    assert set(env.steps) == {RUN}


def test_non_jump_macro_on_flat_ground_is_one_decision_long():
    env = FakeEnv()
    ex = execute(env, Macro.RIGHT_RUN, grounded=True, frames_per_decision=8, ram=env.ram)
    assert ex.frames == 8


def test_macro_stops_on_terminal():
    env = FakeEnv(air_frames=40, terminate_at=5)
    ex = execute(env, Macro.RIGHT_RUN_JUMP_FULL, grounded=True, ram=env.ram)
    assert ex.frames == 5 and ex.terminated is True


def test_macro_is_capped_when_stuck_airborne():
    env = FakeEnv(stuck_airborne=True)
    ex = execute(env, Macro.RIGHT_RUN_JUMP_FULL, grounded=True, ram=env.ram)
    assert ex.frames == MAX_MACRO_FRAMES


def test_reward_frames_scale_the_time_penalty_only():
    base = shaped_reward(dx=40, terminal=False, cleared=False, dead=False)
    long = shaped_reward(dx=40, terminal=False, cleared=False, dead=False, frames=64)
    assert base - long == (DEFAULT.time_penalty * 7)


def test_grounded_reward_keeps_death_negative_after_a_long_jump():
    # Frame-cadence config: a death after 190 px would net +1 (2.3 − 1, clipped). Bug.
    assert shaped_reward(dx=190, terminal=True, cleared=False, dead=True) == 1.0
    assert shaped_reward(dx=190, terminal=True, cleared=False, dead=True, config=GROUNDED) == -1.0
    assert shaped_reward(dx=190, terminal=True, cleared=True, dead=False, config=GROUNDED) == 1.0
    assert (
        0.7
        < shaped_reward(
            dx=190, terminal=False, cleared=False, dead=False, frames=64, config=GROUNDED
        )
        < 0.8
    )


def test_terminal_overrides_default_off_is_byte_identical():
    assert RewardConfig().terminal_overrides is False
    for dx in (-10, 0, 40, 190):
        for dead in (False, True):
            assert shaped_reward(dx=dx, terminal=True, cleared=False, dead=dead) == shaped_reward(
                dx=dx, terminal=True, cleared=False, dead=dead, config=RewardConfig()
            )


# --- teacher macro rule -----------------------------------------------------------------


def _info(x: int = 100) -> dict:
    return {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": x,
        "y_pos": 80,
        "y_pixel": 80,
        "progress": x,
        "time": 390,
        "life": 2,
        "status": "small",
    }


def _ram(*, pit_cols=(), pipe_cols=(), pipe_top_row=5, floor_rows=(5, 6)) -> bytearray:
    ram = bytearray(0x0800)
    for column in range(16):
        if column in pit_cols:
            continue
        for row in floor_rows:
            ram[0x0500 + row * 16 + column] = 0x54
    for column in pipe_cols:
        for row in range(pipe_top_row, 13):
            ram[0x0500 + row * 16 + column] = 0x14
    return ram


def _teacher_choice(ram: bytearray) -> Macro:
    snapshot = MarioStateParser().parse(_info(), ram)
    return DiversifiedHeuristicPolicy(epsilon=0.0).choose(snapshot, tuple(Macro)).action


def test_teacher_walks_off_a_drop():
    # Mario (column 6) on a pipe top at row 5; the floor is at rows 9–10: a 4-tile drop.
    ram = _ram(pipe_cols=(6, 7), floor_rows=(9, 10))
    assert _teacher_choice(ram) is Macro.RIGHT_RUN


def test_teacher_sizes_the_jump_to_the_pit():
    assert _teacher_choice(_ram(pit_cols=(9, 10))) is Macro.RIGHT_RUN_JUMP_MID  # 2 wide, 3 ahead
    assert _teacher_choice(_ram(pit_cols=(9, 10, 11))) is Macro.RIGHT_RUN_JUMP_FULL  # 3 wide


def test_teacher_sizes_the_jump_to_the_pipe():
    low = _ram(pipe_cols=(8,), pipe_top_row=3)  # 2 tiles above the floor at row 5
    tall = _ram(pipe_cols=(8,), pipe_top_row=1)  # 4 tiles
    assert _teacher_choice(low) is Macro.RIGHT_RUN_JUMP_MID
    assert _teacher_choice(tall) is Macro.RIGHT_RUN_JUMP_FULL


def test_progress_fraction_reward_sums_to_at_most_one_per_episode():
    """findings #22: the engine's sequential target is an n-step SUM clipped to [0,1], so the
    per-episode sum of rewards must stay <= 1. A full clear pays exactly 1.0; a death pays 0
    extra (the forfeited future progress is the penalty); no time drag; backward steps <= 0."""
    from typesafe_mario.reward import LEVEL_LENGTH_PX, PROGRESS_FRACTION

    steps = [23] * 40 + [152] * 10  # 920 + 1520 px of progress
    total = sum(
        shaped_reward(
            dx=d, terminal=False, cleared=False, dead=False, frames=51, config=PROGRESS_FRACTION
        )
        for d in steps
    )
    assert abs(total - sum(steps) / LEVEL_LENGTH_PX) < 1e-9
    assert (
        shaped_reward(
            dx=LEVEL_LENGTH_PX, terminal=True, cleared=True, dead=False, config=PROGRESS_FRACTION
        )
        == 1.0
    )
    assert (
        shaped_reward(dx=20, terminal=True, cleared=False, dead=True, config=PROGRESS_FRACTION)
        == 20 / LEVEL_LENGTH_PX
    )  # no −1 spike
    assert (
        shaped_reward(dx=-30, terminal=False, cleared=False, dead=False, config=PROGRESS_FRACTION)
        < 0
    )  # clipped to 0 by the component min


def test_progress_rate_reward_pays_per_eight_frames():
    """findings #22 (curve): per-decision progress favoured long macros (a 51-frame full jump paid 6.6x
    an 8-frame run step). PROGRESS_RATE pays Δx per 8 frames so both pay the same per decision."""
    from typesafe_mario.reward import LEVEL_LENGTH_PX, PROGRESS_FRACTION, PROGRESS_RATE

    run = shaped_reward(
        dx=23, terminal=False, cleared=False, dead=False, frames=8, config=PROGRESS_RATE
    )
    jump = shaped_reward(
        dx=152, terminal=False, cleared=False, dead=False, frames=51, config=PROGRESS_RATE
    )
    assert abs(run - 23 / LEVEL_LENGTH_PX) < 1e-12
    assert abs(jump - (152 * 8 / 51) / LEVEL_LENGTH_PX) < 1e-12
    assert 0.7 < jump / run < 1.3  # same order per decision, no length bias
    frac_jump = shaped_reward(
        dx=152, terminal=False, cleared=False, dead=False, frames=51, config=PROGRESS_FRACTION
    )
    assert frac_jump / run > 6  # the bias the rate preset removes


def test_domain_config_carries_the_measured_selection_knobs():
    from typesafe_mario.adapt1_policy import build_domain_config

    pol = build_domain_config("d", sequential=True)["learning"]["policy"]
    assert pol["model_max_weight"] == 0.9 and pol["min_context_observations"] == 16
    comp = build_domain_config("d")["learning"]["reward"]["components"][0]
    assert comp["min"] == 0.0 and comp["max"] == 1.0


def test_vertical_jump_is_a_jump_without_an_apex_look():
    """2026-09-23: `jump` (A only, 24 frames) is the primitive 2-1's springboard needs; it is a jump
    for the release-edge logic but has no forward arc, so the apex look never fires on it."""
    from typesafe_mario.macros import APEX_MACROS, JUMP_MACROS, Macro, has_apex, is_jump

    assert Macro.JUMP in JUMP_MACROS and is_jump(Macro.JUMP)
    assert Macro.JUMP not in APEX_MACROS and not has_apex(Macro.JUMP)
    assert has_apex(Macro.RIGHT_RUN_JUMP_FULL)
    assert (
        MACRO_PLANS[Macro.JUMP].press is Action.JUMP
        and MACRO_PLANS[Macro.JUMP].carry is Action.NOOP
    )
