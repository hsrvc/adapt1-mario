"""Geometry-pinned regression for findings #21 — runs the real emulator, no API.

Replays the deterministic heuristic teacher through World 1-1 and asserts what the parser
reports at the two places that mattered:

* on top of pipe 4 (x≈918, 4 tiles up) the terrain is a **drop**, not a gap — the old parser's
  phantom "gap 2 ahead, 7 wide" is what sent the teacher (and Adapt-1) max-jumping into the
  first pit;
* from the floor after the drop, the **real** first pit (tiles 69–70, x=1104–1135) is seen as a
  2-wide pit a few tiles ahead.

Then the same unchanged teacher must clear that pit (max_x well past 1153). Skipped when the
emulator isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gym_super_mario_bros")

from typesafe_mario.actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action
from typesafe_mario.policy import DiversifiedHeuristicPolicy
from typesafe_mario.runner import _unwrap_ram, create_mario_env
from typesafe_mario.state import MarioStateParser


def _replay(max_decisions: int = 120) -> tuple[list, int]:
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0, rng_seed=0)
    actions = tuple(Action)
    grounded: list = []
    try:
        _frame, info = env.reset(seed=777)
        snapshot = parser.parse(info, _unwrap_ram(env), previous_action=None)
        for _ in range(max_decisions):
            if snapshot.dead or snapshot.clear:
                break
            action = teacher.choose(snapshot, actions).action
            if snapshot.grounded:
                grounded.append((snapshot, action))
            index = ACTION_TO_INDEX[action]
            for f in range(8):
                if f == 0 and action in JUMP_ACTIONS and snapshot.grounded:
                    step = ACTION_TO_INDEX[JUMP_RELEASE_ACTION[action]]
                else:
                    step = index
                _frame, reward, terminated, truncated, info = env.step(step)
                if terminated or truncated:
                    break
            snapshot = parser.parse(
                info, _unwrap_ram(env), previous_action=action.value, previous_reward=reward
            )
            if terminated or truncated:
                break
        return grounded, int(info.get("x_pos_max", snapshot.x))
    finally:
        env.close()


@pytest.fixture(scope="module")
def replay():
    return _replay()


def test_pipe_4_top_is_a_drop_not_a_gap(replay):
    grounded, _ = replay
    on_pipe_4 = [(s, a) for s, a in grounded if 900 <= s.x <= 945 and s.y > 130]
    assert on_pipe_4, "teacher never stood on pipe 4"
    snapshot, action = on_pipe_4[0]
    terrain = snapshot.to_state()["terrain"]
    assert terrain["gap_ahead"] is False
    assert terrain["gap_distance_tiles"] is None  # the real pit is beyond the 8-tile screen
    assert terrain["drop_distance_tiles"] == 2
    assert terrain["drop_depth_tiles"] == 4
    assert terrain["obstacle_distance_tiles"] is None  # the hidden 1-UP block is stripped
    assert action is Action.RIGHT_RUN  # walks off instead of max-jumping


def test_first_pit_is_two_tiles_wide_at_floor_level(replay):
    grounded, _ = replay
    approach = [s for s, _ in grounded if 1000 <= s.x < 1104 and s.y == 79]
    assert approach, "teacher never approached the first pit on the floor"
    widths = {s.to_state()["terrain"]["gap_width_tiles_visible"] for s in approach}
    assert widths <= {1, 2}  # 1 only when the pit is cut by the 8-tile window edge
    nearest = approach[-1].to_state()["terrain"]
    assert nearest["gap_distance_tiles"] <= 4  # the teacher takes off at ≤4 tiles at speed
    assert nearest["gap_width_tiles_visible"] == 2


def test_unchanged_teacher_now_clears_the_first_pit(replay):
    _, max_x = replay
    assert max_x >= 1500  # was 1153 on the phantom-gap parser; measured 1525 on 2026-09-21


# --- grounded cadence (macros.py) on the real emulator -------------------------------------


def _run_macros(macros, seed: int = 777):
    from typesafe_mario.macros import execute

    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    try:
        _frame, info = env.reset(seed=seed)
        ram = _unwrap_ram(env)
        snapshot = parser.parse(info, ram, previous_action=None)
        out = []
        for macro in macros:
            x0 = snapshot.x
            ex = execute(env, macro, grounded=snapshot.grounded, ram=ram)
            snapshot = parser.parse(ex.info, ram, previous_action=macro.value)
            out.append((snapshot.x - x0, ex.frames, snapshot.grounded))
        return out
    finally:
        env.close()


def test_jump_sizes_fly_further_in_order_and_land_grounded():
    from typesafe_mario.macros import Macro

    flights = {}
    for macro in (Macro.RIGHT_RUN_JUMP_SHORT, Macro.RIGHT_RUN_JUMP_MID, Macro.RIGHT_RUN_JUMP_FULL):
        runup = [Macro.RIGHT_RUN] * 3
        dx, frames, grounded = _run_macros(runup + [macro])[-1]
        assert grounded, f"{macro} did not end on the ground"
        flights[macro] = (dx, frames)
    short, mid, full = (
        flights[m]
        for m in (Macro.RIGHT_RUN_JUMP_SHORT, Macro.RIGHT_RUN_JUMP_MID, Macro.RIGHT_RUN_JUMP_FULL)
    )
    assert short[0] < mid[0] < full[0]  # measured 89 < 119 < 152 px on 2026-09-21
    assert short[1] < mid[1] < full[1]  # 30 < 40 < 51 frames


def test_grounded_teacher_reaches_the_second_half_of_1_1():
    from typesafe_mario.macros import Macro, execute

    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0, rng_seed=0)
    try:
        _frame, info = env.reset(seed=777)
        ram = _unwrap_ram(env)
        snapshot = parser.parse(info, ram, previous_action=None)
        decisions = 0
        while not (snapshot.dead or snapshot.clear) and decisions < 150:
            macro = teacher.choose(snapshot, tuple(Macro)).action
            ex = execute(env, macro, grounded=snapshot.grounded, ram=ram)
            snapshot = parser.parse(ex.info, ram, previous_action=macro.value)
            decisions += 1
            if ex.terminated or ex.truncated:
                break
        max_x = int(ex.info.get("x_pos_max", snapshot.x))
    finally:
        env.close()
    assert (
        max_x >= 3000
    )  # arm 2 teacher (mid hop): clears 1-1, x=3161 (was 2028 with the short hop)
    assert decisions <= 90  # a full clear is ~74 grounded decisions (vs ~123 at the frame cadence)
