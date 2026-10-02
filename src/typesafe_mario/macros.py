"""Grounded-cadence macros — committed jump sizes, decided only on the ground.

Findings #21 (2026-09-21) measured two things about the 8-frame cadence: only 27 of 81
decisions in a 1-1 run are grounded (the airborne two thirds carry almost no real choice but
cost a Record each and dilute credit), and the delay from a takeoff decision to the death it
causes is median 7 / p90 13 decisions — beyond the default ``n_step``. This module moves the
decision to the altitude between #18 (frame-timed presses, unlearnable) and #19 (hand-coded
skills, nothing to learn):

* a **Macro** is a whole committed move: a direction plus, for jumps, how long A is held
  (8 / 16 / 32 frames — the NES variable-height jump). The learner still decides *when* to
  jump and *how big*, from geometry; code only executes the button hold, which is exactly what
  an 8-frame ``Action`` already was.
* a macro runs until Mario is **back on the ground** (RAM ``0x001D`` float state == 0), so the
  next decision is always grounded. Walking off a drop therefore also runs to the landing.

**Apex decision (design-notes §14a, 2026-09-21).** ``execute(..., apex=hook)`` adds ONE look during
a jump macro: on the first airborne frame where Mario stops rising it calls ``hook(ApexEvent)``,
and the hook answers with an ``ApexChoice`` — keep the direction (byte-identical to the committed
macro), brake (left 8 frames, then release) or pull back (left to landing). Measured with
``scripts/probe_apex.py``: from the pipe-4 top the full jump lands in the pit at 1104–1135 with
keep, and on the ground before it with pull back (x=1045 from takeoff 921, 1057 from 931); the
pipe-3 apex sees no pit and pull back there merely lands short. "Let go" was dropped: with no
direction held NES Mario keeps his horizontal speed, so release lands exactly where keep lands.

``execute`` is the single execution path for BOTH cadences: given a plain ``Action`` it
reproduces the old 8-frame loop byte-for-byte (frame-0 release edge on a grounded jump, then
hold), so the frame cadence, the demos and the frozen-eval stay identical to #20; given a
``Macro`` it runs the committed move. The runner / recorder / demo loops all call it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action

# NES RAM: player float state. 0 = on the ground, non-zero = airborne (1 jumping, 2 walked
# off a ledge, 3 sliding down the flagpole). Measured 2026-09-21 (probe): 0 grounded, 1 mid-jump.
FLOAT_STATE_ADDR = 0x001D

# Hard cap on one macro, in frames. A full held run-jump lands in ~64–72 frames; falling into
# a pit dies in ~40. Anything longer means Mario is stuck airborne (flagpole slide) — return.
MAX_MACRO_FRAMES = 96


class Macro(StrEnum):
    NOOP = "noop"
    RIGHT = "right"
    RIGHT_RUN = "right_run"
    LEFT = "left"
    RIGHT_RUN_JUMP_SHORT = "right_run_jump_short"  # A held 8 frames: a hop
    RIGHT_RUN_JUMP_MID = "right_run_jump_mid"  # A held 16 frames
    RIGHT_RUN_JUMP_FULL = "right_run_jump_full"  # A held 32 frames: the max jump
    RIGHT_JUMP_FULL = "right_jump_full"  # walk-speed max jump: height without the distance
    JUMP = "jump"  # A only, 24 frames: vertical, keeps momentum (design-notes §16 item 2; 2-1's springboard)


@dataclass(frozen=True)
class MacroPlan:
    """How a macro is executed: the button combo while A is held, how long, and the combo
    held afterwards until Mario lands."""

    press: Action  # buttons for the hold phase (a jump action for jumps)
    hold_frames: int  # frames A is held; 0 for non-jump macros
    carry: Action  # buttons after release, held until grounded


MACRO_PLANS: dict[Macro, MacroPlan] = {
    Macro.NOOP: MacroPlan(Action.NOOP, 0, Action.NOOP),
    Macro.RIGHT: MacroPlan(Action.RIGHT, 0, Action.RIGHT),
    Macro.RIGHT_RUN: MacroPlan(Action.RIGHT_RUN, 0, Action.RIGHT_RUN),
    Macro.LEFT: MacroPlan(Action.LEFT, 0, Action.LEFT),
    Macro.RIGHT_RUN_JUMP_SHORT: MacroPlan(Action.RIGHT_RUN_JUMP, 8, Action.RIGHT_RUN),
    Macro.RIGHT_RUN_JUMP_MID: MacroPlan(Action.RIGHT_RUN_JUMP, 16, Action.RIGHT_RUN),
    Macro.RIGHT_RUN_JUMP_FULL: MacroPlan(Action.RIGHT_RUN_JUMP, 32, Action.RIGHT_RUN),
    Macro.RIGHT_JUMP_FULL: MacroPlan(Action.RIGHT_JUMP, 32, Action.RIGHT),
    Macro.JUMP: MacroPlan(Action.JUMP, 24, Action.NOOP),
}

JUMP_MACROS = frozenset(m for m, p in MACRO_PLANS.items() if p.hold_frames > 0)
# The apex look (design-notes §14a) exists for the forward arc — where will this jump land? A
# vertical jump has no arc to steer, so it gets no in-air decision (2026-09-23, `jump`).
APEX_MACROS = frozenset(JUMP_MACROS - {Macro.JUMP})


class ApexChoice(StrEnum):
    """What to do with the rest of the arc, decided once at the top of a jump."""

    KEEP = "apex_keep"  # hold the direction to landing — today's committed macro, unchanged
    BRAKE = "apex_brake"  # left for APEX_BRAKE_FRAMES, then release: land a few tiles short
    PULL_BACK = "apex_pull_back"  # hold left to landing: land near the takeoff column


APEX_BRAKE_FRAMES = 8
APEX_NAMES: tuple[str, ...] = tuple(c.value for c in ApexChoice)

# NES RAM: Player_X_Speed, signed, in 1/16 px per frame (0x30 = 48 = 3 px/frame at full run).
X_SPEED_ADDR = 0x0057


@dataclass(frozen=True)
class ApexEvent:
    """The top of a jump, as the apex hook sees it."""

    macro: Macro
    frame: int  # frames since the macro started (= the rise time)
    x: int
    y: int  # info["y_pos"]: height above the level bottom, px
    vx: float  # horizontal speed, px/frame (signed)
    reward: float  # raw gym reward accumulated from the macro start to the apex
    info: dict[str, Any]
    ram: Any


ApexHook = Callable[[ApexEvent], ApexChoice]

# The names Adapt-1 hypotheses use for each cadence.
ACTION_NAMES: tuple[str, ...] = tuple(a.value for a in Action)
MACRO_NAMES: tuple[str, ...] = tuple(m.value for m in Macro)

CADENCES = ("frame", "grounded")


def action_set(cadence: str) -> tuple[Action, ...] | tuple[Macro, ...]:
    """The choices a policy picks from under a cadence."""
    if cadence == "frame":
        return tuple(Action)
    if cadence == "grounded":
        return tuple(Macro)
    raise ValueError(f"unknown cadence {cadence!r}; expected one of {CADENCES}")


def apex_choices() -> tuple[ApexChoice, ...]:
    """The choices an apex policy picks from (the second, in-air decision of a jump)."""
    return tuple(ApexChoice)


def is_jump(choice: Action | Macro) -> bool:
    return choice in JUMP_ACTIONS if isinstance(choice, Action) else choice in JUMP_MACROS


def has_apex(choice: Action | Macro) -> bool:
    """Whether the apex look fires during this choice (forward jump macros only)."""
    return isinstance(choice, Macro) and choice in APEX_MACROS


@dataclass(frozen=True)
class Execution:
    total_reward: float  # summed raw gym reward over the executed frames
    frames: int
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    # Apex decision (only when ``execute`` was given an ``apex`` hook and the macro reached one):
    apex: ApexEvent | None = None
    apex_choice: ApexChoice | None = None
    reward_to_apex: float = 0.0  # raw gym reward accumulated before the apex frame
    frames_to_apex: int = 0


def _float_state(ram: Any) -> int:
    try:
        return int(ram[FLOAT_STATE_ADDR])
    except (TypeError, IndexError, KeyError):
        return 0


def _x_speed(ram: Any) -> float:
    try:
        raw = int(ram[X_SPEED_ADDR])
    except (TypeError, IndexError, KeyError):
        return 0.0
    return (raw - 256 if raw > 127 else raw) / 16.0


def execute(
    env: Any,
    choice: Action | Macro,
    *,
    grounded: bool,
    frames_per_decision: int = 8,
    ram: Any = None,
    on_frame: Callable[[Any], None] | None = None,
    max_frames: int = MAX_MACRO_FRAMES,
    apex: ApexHook | None = None,
) -> Execution:
    """Run one decision on the emulator and return what happened.

    ``Action`` (frame cadence): exactly the old loop — ``frames_per_decision`` frames, with the
    frame-0 button-up edge on a grounded jump so a fresh press fires (findings #20 fix #2).

    ``Macro`` (grounded cadence): frame-0 release edge if it is a grounded jump, A held for the
    plan's ``hold_frames``, then the ``carry`` combo until RAM says Mario is on the ground again
    (at least ``frames_per_decision`` frames in total), capped at ``max_frames``. Stops early on
    a terminal step. ``ram`` is required for landing detection; without it the macro runs its
    press phase plus one carry step of ``frames_per_decision`` frames.
    """
    total = 0.0
    frames = 0
    terminated = truncated = False
    info: dict[str, Any] = {}

    def step(action: Action) -> bool:
        nonlocal total, frames, terminated, truncated, info
        frame, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
        if on_frame is not None:
            on_frame(frame)
        total += float(reward)
        frames += 1
        return bool(terminated or truncated)

    if isinstance(choice, Action):
        for f in range(frames_per_decision):
            if f == 0 and choice in JUMP_ACTIONS and grounded:
                action = JUMP_RELEASE_ACTION[choice]
            else:
                action = choice
            if step(action):
                break
        return Execution(total, frames, terminated, truncated, info)

    plan = MACRO_PLANS[choice]
    done = False
    apex_event: ApexEvent | None = None
    apex_choice: ApexChoice | None = None
    reward_to_apex = 0.0
    frames_to_apex = 0
    frames_since_apex = 0
    prev_y: int | None = None

    def after_apex(default: Action) -> Action:
        """The input for this frame: the plan's, unless an apex choice overrides the arc."""
        nonlocal frames_since_apex
        if apex_choice is None or apex_choice is ApexChoice.KEEP:
            return default
        f, frames_since_apex = frames_since_apex, frames_since_apex + 1
        if apex_choice is ApexChoice.PULL_BACK:
            return Action.LEFT
        return Action.LEFT if f < APEX_BRAKE_FRAMES else Action.NOOP

    def check_apex() -> None:
        """After a frame: the first non-rising airborne frame of a jump macro is the apex."""
        nonlocal apex_event, apex_choice, reward_to_apex, frames_to_apex, prev_y
        y = info.get("y_pos") if isinstance(info, dict) else None
        if y is None:
            return
        if (
            apex is not None
            and apex_event is None
            and choice in APEX_MACROS
            and frames > 2
            and prev_y is not None
            and int(y) <= prev_y
            and _float_state(ram) != 0
        ):
            apex_event = ApexEvent(
                macro=choice,
                frame=frames,
                x=int(info.get("x_pos", 0)),
                y=int(y),
                vx=_x_speed(ram),
                reward=total,
                info=dict(info),
                ram=ram,
            )
            apex_choice = apex(apex_event)
            reward_to_apex, frames_to_apex = total, frames
        prev_y = int(y)

    # Press phase: release edge on frame 0 of a grounded jump, then hold A for hold_frames.
    if plan.hold_frames > 0:
        if grounded:
            done = step(JUMP_RELEASE_ACTION[plan.press])
            check_apex()
        for _ in range(plan.hold_frames):
            if done:
                break
            done = step(after_apex(plan.press))
            check_apex()
    else:
        for _ in range(frames_per_decision):
            if done:
                break
            done = step(plan.press)
    # Carry phase: hold the direction until Mario is back on the ground.
    while not done and frames < max_frames:
        airborne = _float_state(ram) != 0 if ram is not None else False
        if frames >= frames_per_decision and not airborne:
            break
        done = step(after_apex(plan.carry))
        check_apex()
        if ram is None and frames >= plan.hold_frames + frames_per_decision:
            break
    return Execution(
        total,
        frames,
        terminated,
        truncated,
        info,
        apex=apex_event,
        apex_choice=apex_choice,
        reward_to_apex=reward_to_apex,
        frames_to_apex=frames_to_apex,
    )
