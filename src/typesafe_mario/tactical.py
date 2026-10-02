"""Tactical layer — Option 2 reformulation (findings #18).

The frozen-clone workstream (#15–#18) had Adapt-1 deciding *motor timing* (which fixed
8-frame press), which is exactly the real-time control the blog shows it is NOT built for.
This module moves the timing into **code** and leaves Adapt-1 (later) only a **discrete
tactical choice**, matching the symbolic/discrete tasks it actually wins on (Alchemy,
ALFWorld):

- **Closed-loop skill macros** (`TacticalSkill`) own their own frame-timing. `ADVANCE` runs
  right until a tactical event; `JUMP_CLEAR` initiates a running jump, holds it while rising,
  and rides it back to the ground — a self-terminating skill, not a fixed press.
- **Event-driven decisions.** A decision happens only at a *tactical event* (obstacle / gap /
  enemy in range) or when a skill finishes — ~20–40 decisions/episode, not ~500.
- **Coarse symbolic state** (`encode_tactical_state`) — obstacle/enemy/gap/mario as a handful
  of categories, the low-dimensional representation Adapt-1's envelope wants.

Everything here is offline and free (local emulator, no API). `run_tactical_episode` lets us
validate that this layer *plays* (via `HeuristicTacticalPolicy`) before wiring Adapt-1 or Jev
on top. If a tactical heuristic clears the x=723 pipe through this loop, the reformulation is
sound and the decision layer is swappable — the same swap point the whole workstream uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .actions import ACTION_TO_INDEX, Action
from .runner import _unwrap_ram, create_mario_env
from .state import MarioStateParser

TILE_PIXELS = 16

# Trigger thresholds (tuned to the proven #16 heuristic: jump an enemy only in a stomp window
# so the arc lands on it, not into it). These live in code — Adapt-1 never sees raw timing.
OBSTACLE_TRIGGER_TILES = 3  # a terrain obstacle this close ⇒ a jump decision is due
GAP_TRIGGER = True  # any flagged gap ahead ⇒ a jump decision is due
ENEMY_STOMP_MIN_PX = 24  # #16: jumping earlier than this lands past/short of the enemy
ENEMY_STOMP_MAX_PX = 72

_RIGHT_RUN = ACTION_TO_INDEX[Action.RIGHT_RUN]
_RIGHT_RUN_JUMP = ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP]

ADVANCE_MAX_FRAMES = 24  # a skill never runs forever; the runner also caps decisions
JUMP_CLEAR_MAX_FRAMES = 32


class TacticalSkill(StrEnum):
    """The discrete choices Adapt-1 (or a heuristic) selects among — NOT motor presses."""

    ADVANCE = "advance"  # run right until the next tactical event
    JUMP_CLEAR = "jump_clear"  # running-jump an obstacle / gap / enemy, held and landed by code


# ---- perception → coarse symbolic state -------------------------------------------------


def _obstacle_tiles(snapshot: Any) -> float | None:
    nav = snapshot.navigation_features()
    d = nav.get("obstacle_distance_tiles")
    return float(d) if isinstance(d, (int, float)) else None


def _gap_ahead(snapshot: Any) -> bool:
    return bool(snapshot.navigation_features().get("gap_ahead"))


def _enemy_px(snapshot: Any) -> float | None:
    d = snapshot.threat_features().get("nearest_enemy_distance_pixels")
    return float(d) if isinstance(d, (int, float)) else None


def _bucket_obstacle(tiles: float | None) -> str:
    if tiles is None:
        return "none"
    if tiles <= OBSTACLE_TRIGGER_TILES:
        return "near"
    if tiles <= 5:
        return "far"
    return "none"


def _bucket_enemy(px: float | None) -> str:
    if px is None or px < 0:
        return "none"
    if px <= ENEMY_STOMP_MAX_PX:
        return "stomp_window" if px >= ENEMY_STOMP_MIN_PX else "too_close"
    if px <= 128:
        return "approaching"
    return "none"


def encode_tactical_state(snapshot: Any) -> dict[str, str]:
    """The coarse, symbolic decision state — a handful of categories, no raw timing."""
    return {
        "mario": "grounded" if snapshot.grounded else "airborne",
        "obstacle": _bucket_obstacle(_obstacle_tiles(snapshot)),
        "gap": "ahead" if _gap_ahead(snapshot) else "none",
        "enemy": _bucket_enemy(_enemy_px(snapshot)),
    }


def is_tactical_event(snapshot: Any) -> bool:
    """A decision is *due* when, grounded, something needs handling within reach."""
    if not snapshot.grounded:
        return False
    st = encode_tactical_state(snapshot)
    return st["obstacle"] == "near" or st["gap"] == "ahead" or st["enemy"] == "stomp_window"


# ---- closed-loop skill execution --------------------------------------------------------


@dataclass
class SkillResult:
    next_snapshot: Any
    frames: int
    dx: int
    terminal: bool
    clear: bool
    dead: bool


def _jump_clear_press(frame: int, live: Any, left_ground: bool) -> int:
    """Motor policy for JUMP_CLEAR, in code. Grounded jumps need a released→pressed edge to
    fire (NES: jump triggers on button-down, not hold — the same reason the 8-frame loop
    releases on frame 0), so frame 0 releases, then we hold jump while rising for max height
    and let go at the apex to keep forward momentum without re-triggering."""
    if not left_ground:
        return _RIGHT_RUN if frame == 0 else _RIGHT_RUN_JUMP  # release edge, then launch
    return _RIGHT_RUN_JUMP if live.jump_phase == "rising" else _RIGHT_RUN


def execute_skill(
    env: Any, parser: Any, snapshot: Any, skill: TacticalSkill, *, prev_reward: float = 0.0
) -> SkillResult:
    """Run one skill to its own termination, stepping the emulator frame by frame."""
    start_x = snapshot.x
    live = snapshot
    left_ground = not snapshot.grounded
    dead = clear = terminal = False
    prev_action = None

    cap = ADVANCE_MAX_FRAMES if skill is TacticalSkill.ADVANCE else JUMP_CLEAR_MAX_FRAMES
    frames = 0
    for f in range(cap):
        if skill is TacticalSkill.ADVANCE:
            press = _RIGHT_RUN
        else:
            press = _jump_clear_press(f, live, left_ground)
        action_name = next(a for a, i in ACTION_TO_INDEX.items() if i == press)
        _frame, reward, terminated, truncated, info = env.step(press)
        prev_reward += float(reward)
        prev_action = action_name.value
        live = parser.parse(
            info, _unwrap_ram(env), previous_action=prev_action, previous_reward=float(reward)
        )
        frames += 1
        if live.airborne:
            left_ground = True
        dead, clear = live.dead, live.clear
        if terminated or truncated or dead or clear:
            terminal = True
            break
        # Termination conditions per skill:
        if skill is TacticalSkill.JUMP_CLEAR:
            if left_ground and live.grounded:  # landed — the jump is complete
                break
        else:  # ADVANCE: stop as soon as a new decision is due (min 1 frame of progress)
            if is_tactical_event(live):
                break

    return SkillResult(
        next_snapshot=live,
        frames=frames,
        dx=live.x - start_x,
        terminal=terminal,
        clear=clear,
        dead=dead,
    )


# ---- a heuristic tactical policy (validates the layer plays) -----------------------------


class TacticalPolicy(Protocol):
    def choose(self, snapshot: Any, state: dict[str, str]) -> TacticalSkill: ...


class HeuristicTacticalPolicy:
    """The proven #16 rule, lifted to the *skill* level: jump when grounded and an obstacle,
    gap, or stomp-window enemy is in reach; otherwise advance. This is the reference the
    Adapt-1/Jev decision layer must match or beat on the SAME loop."""

    def choose(self, snapshot: Any, state: dict[str, str]) -> TacticalSkill:
        if state["mario"] != "grounded":
            return TacticalSkill.ADVANCE
        if (
            state["obstacle"] == "near"
            or state["gap"] == "ahead"
            or state["enemy"] == "stomp_window"
        ):
            return TacticalSkill.JUMP_CLEAR
        return TacticalSkill.ADVANCE


def run_tactical_episode(
    env_id: str,
    policy: TacticalPolicy,
    *,
    seed: int = 0,
    max_decisions: int = 80,
    stall_timeout: int = 12,
) -> dict[str, Any]:
    """Play one episode through the tactical loop. Offline, free. Returns per-episode stats
    and the decision trace (each tactical state → skill → dx), so we can see both that it
    plays and how *few* decisions it takes."""
    env = create_mario_env(env_id, render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    trace: list[dict[str, Any]] = []
    try:
        _frame, info = env.reset(seed=seed)
        snapshot = parser.parse(info, _unwrap_ram(env), previous_action=None)
        best_x = snapshot.x
        stalled = 0
        for _ in range(max_decisions):
            if snapshot.dead or snapshot.clear:
                break
            state = encode_tactical_state(snapshot)
            skill = policy.choose(snapshot, state)
            result = execute_skill(env, parser, snapshot, skill)
            trace.append(
                {
                    "state": state,
                    "skill": skill.value,
                    "dx": result.dx,
                    "frames": result.frames,
                    "x": result.next_snapshot.x,
                }
            )
            snapshot = result.next_snapshot
            if snapshot.x > best_x:
                best_x, stalled = snapshot.x, 0
            else:
                stalled += 1
            if result.terminal or stalled >= stall_timeout:
                break
    finally:
        env.close()

    return {
        "seed": seed,
        "max_x": best_x,
        "reached_flag": bool(snapshot.clear),
        "died": bool(snapshot.dead),
        "decisions": len(trace),
        "cleared_pipe": best_x > 723,
        "trace": trace,
    }
