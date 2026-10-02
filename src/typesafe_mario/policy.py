from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .actions import ACTION_DESCRIPTIONS, Action
from .macros import ApexChoice, Macro
from .state import MarioSnapshot


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    probabilities: Mapping[str, float]
    latency_ms: float
    # Jev-only telemetry (Noul / Score). None for the Adapt-1 brain.
    jump_needed_probability: float | None = None
    danger_score: float | None = None
    # Adapt-1-only telemetry. None for the Jev / heuristic brains. `selection_status`
    # being non-None is what marks a decision as Adapt-1-sourced for the dashboard.
    selection_status: str | None = None
    expected_reward: float | None = None
    sample_count: int | None = None
    # Adapt-1 dashboard telemetry (findings #22): deployed model, selection reason, per-macro
    # learned values, explored flag, running selected/abstained/explored counts, episode/step.
    telemetry: Mapping[str, Any] | None = None


class Policy(Protocol):
    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision: ...


class TypeSafePolicy:
    def __init__(self) -> None:
        try:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
        except ImportError as exc:
            raise RuntimeError(
                "typesafe-sdk is not installed. Install the project before using Jev."
            ) from exc
        self._Choice = Choice
        self._Noul = Noul
        self._Score = Score
        self._client = TypeSafeClient()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _answer(response: Any, question_id: str, typed_collection: str) -> Any:
        collection = getattr(response, typed_collection, None)
        if collection is not None and question_id in collection:
            return collection[question_id]
        answers = getattr(response, "answers", None)
        if answers is not None and question_id in answers:
            return answers[question_id]
        raise KeyError(f"TypeSafe response omitted {question_id!r}")

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        criteria = {action.value: ACTION_DESCRIPTIONS[action] for action in actions}
        questions = {
            "next_action": self._Choice(
                instructions={
                    "question": "Which controller macro should Mario commit to next?",
                    "goal": "Advance toward the stage flag while avoiding death.",
                    "timing": "The selected action is held for at least 8 emulator frames.",
                    "geometry": (
                        "Use `terrain.observation_reliability`. While airborne, prefer "
                        "`terrain.last_grounded_preview` over low-reliability current geometry. "
                        "A trusted obstacle or gap within three tiles requires a forward jump."
                    ),
                    "trajectory": (
                        "Use `trajectory`. If `crossing_known_gap` is true, preserve forward "
                        "speed and keep a forward jump held while rising. Do not switch to noop "
                        "or left over a gap."
                    ),
                    "stall": (
                        "If `episode.stalled_frames` is increasing and "
                        "`recent_control.outcome` is blocked, the current non-jump action failed."
                    ),
                    "enemy_timing": (
                        "Use `hazard` projections, not distance alone. Code has already accounted "
                        "for inference delay, action cadence, and the frames needed to clear an "
                        "enemy. If `jump_must_start_this_decision` is true, choose a forward jump "
                        "now; another right-run decision will miss the takeoff deadline. If "
                        "`contact_within_reaction_horizon` is true, also jump immediately. If "
                        "`will_land_before_contact` is true, the current jump will not clear the "
                        "enemy and another takeoff will be needed after landing. Use "
                        "`upcoming_enemies` and their spacing to avoid landing on a second or "
                        "third enemy hidden behind the nearest one."
                    ),
                    "delay": (
                        "`reaction_timing` describes how far the world moves before this choice "
                        "takes effect. Judge urgency from projected rather than current distance."
                    ),
                },
                criteria=criteria,
            ),
            "jump_needed": self._Noul(
                instructions=(
                    "Do trusted `terrain`, projected `hazard`, `trajectory`, and "
                    "`player.jump_phase` indicate that a forward jump should begin or remain "
                    "held now? `hazard.jump_must_start_this_decision=true` is unambiguously yes. "
                    "Also count a trusted obstacle/gap within three tiles, immediate projected "
                    "contact, or a rising jump over a known gap as yes."
                )
            ),
            "danger": self._Score(
                instructions="How dangerous is Mario's immediate situation?",
                criteria=[
                    "Safe open movement",
                    "Potential obstacle or enemy soon",
                    "Immediate collision, fall, or enemy threat",
                ],
            ),
        }
        started = time.perf_counter()
        response = self._client.system_one(state=snapshot.to_state(), questions=questions)
        latency_ms = (time.perf_counter() - started) * 1000

        action_answer = self._answer(response, "next_action", "choices")
        jump_answer = self._answer(response, "jump_needed", "nouls")
        danger_answer = self._answer(response, "danger", "scores")
        action = Action(str(action_answer.choice))
        probabilities = {
            str(key): float(value) for key, value in dict(action_answer.probabilities).items()
        }
        return Decision(
            action=action,
            confidence=float(action_answer.confidence),
            probabilities=probabilities,
            latency_ms=latency_ms,
            jump_needed_probability=float(jump_answer.noul),
            danger_score=float(danger_answer.score),
        )


class HeuristicPolicy:
    """Offline smoke-test policy; not intended as the Mario benchmark baseline."""

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        allowed = set(actions)
        action = Action.RIGHT_RUN if Action.RIGHT_RUN in allowed else actions[0]
        if snapshot.stalled_steps >= 2 and Action.RIGHT_RUN_JUMP in allowed:
            action = Action.RIGHT_RUN_JUMP
        return Decision(
            action=action,
            confidence=1.0,
            probabilities={candidate.value: float(candidate == action) for candidate in actions},
            latency_ms=0.0,
        )


class DiversifiedHeuristicPolicy:
    """A competent-but-noisy Mario heuristic — the "teacher" for warm-start demos.

    Rule: run right, and start a running jump when the parser flags an imminent
    obstacle, gap, or enemy. With probability ``epsilon`` it takes a **random** macro
    instead — deliberately injecting mistakes and recoveries so the recorded demonstration
    data is *diverse*, not a narrow expert distribution (the offline-RL trap where a
    learner only ever sees flawless play and flails the moment it deviates).

    Local and free — no API. Not a `Policy` that learns; it only generates data.
    """

    def __init__(self, *, epsilon: float = 0.15, rng_seed: int = 0) -> None:
        self._epsilon = max(0.0, min(1.0, epsilon))
        self._rng = random.Random(rng_seed)

    def choose(
        self,
        snapshot: MarioSnapshot,
        actions: Sequence[Any],
        extra: Mapping[str, Any] | None = None,
    ) -> Decision:
        allowed = list(actions)
        if self._rng.random() < self._epsilon:
            action = self._rng.choice(allowed)
            explored = True
        elif allowed and isinstance(allowed[0], ApexChoice):
            action = self._apex_rule(snapshot, extra or {}, allowed)
            explored = False
        elif allowed and isinstance(allowed[0], Macro):
            action = self._macro_rule(snapshot, allowed)
            explored = False
        else:
            action = self._rule(snapshot, allowed)
            explored = False
        return Decision(
            action=action,
            confidence=float(self._epsilon if explored else 1.0 - self._epsilon),
            probabilities={candidate.value: float(candidate == action) for candidate in actions},
            latency_ms=0.0,
            selection_status="explored" if explored else None,  # off-policy rows = coverage data
        )

    # Fall time from the apex, in frames: rise + APEX_FALL_PER_TILE * tiles above the floor.
    # Measured on the 1-1 pipe tops (probe_apex.py, 2026-09-21): full jump rise 27 frames, fall
    # to the ground 9.3 tiles below in 36; mid 21 -> 31; short 16 -> 26; a same-level landing
    # falls about as long as it rose.
    APEX_FALL_PER_TILE = 1.1

    def _apex_rule(
        self, snapshot: MarioSnapshot, apex: Mapping[str, Any], allowed: Sequence[ApexChoice]
    ) -> ApexChoice:
        """The in-air look (design-notes §14a): pull back when the arc is heading into a pit
        that was invisible at takeoff. Projected landing column = vx * fall_frames / 16 tiles
        ahead; if the floor profile there is bottomless, hold left to landing (measured: lands
        on the ground before the pit from both pipe-4 takeoff points). Over a pit already (no
        floor below) or drifting left, the only way out is forward: keep."""
        keep = ApexChoice.KEEP if ApexChoice.KEEP in allowed else allowed[0]
        nav = snapshot.navigation_features()
        profile = nav.get("floor_profile_tiles") or []
        height = apex.get("height")
        vx = float(apex.get("vx") or 0.0)
        rise = float(apex.get("rise") or 0.0)
        if not profile or height is None or vx <= 0.0:
            return keep
        fall_frames = rise + self.APEX_FALL_PER_TILE * float(height)
        column = min(round(vx * fall_frames / 16.0), len(profile) - 1)
        if profile[column] is None and ApexChoice.PULL_BACK in allowed:
            return ApexChoice.PULL_BACK
        return keep

    def _macro_rule(self, snapshot: MarioSnapshot, allowed: Sequence[Macro]) -> Macro:
        """Grounded-cadence teacher (findings #21): same triggers as ``_rule``, but the jump
        *size* is chosen from geometry — a hop for a stomp, a mid jump for a small pit or a
        low pipe, the full jump for a wide pit or a tall pipe. Drops are walked off (the
        floor-profile parser no longer reports them as gaps). Deliberately keeps ``_rule``'s
        weak enemy timing so online learning has something left to improve."""
        state = snapshot.to_state()
        hazard, terrain = state["hazard"], state["terrain"]
        run = Macro.RIGHT_RUN
        if not state["player"]["grounded"]:
            return run  # a macro ends grounded; this only happens after a max_frames cutoff
        pit = terrain.get("gap_distance_tiles")
        pit_width = int(terrain.get("gap_width_tiles_visible") or 0)
        wall = terrain.get("obstacle_distance_tiles")
        wall_height = int(terrain.get("obstacle_height_tiles") or 0)
        e = hazard["nearest_enemy_distance_pixels"]
        e = 999.0 if e is None else float(e)
        if pit is not None and pit <= 4:
            return Macro.RIGHT_RUN_JUMP_FULL if pit_width >= 3 else Macro.RIGHT_RUN_JUMP_MID
        if wall is not None and wall <= 3:
            return Macro.RIGHT_RUN_JUMP_FULL if wall_height >= 3 else Macro.RIGHT_RUN_JUMP_MID
        if hazard["enemy_ahead"] and 24.0 <= e <= 72.0:
            # Grounded arm 2 (2026-09-21): the offline sweep found that hopping enemies with the
            # 16-frame jump instead of the 8-frame hop takes this rule from x=2028 to the flag
            # deterministically (3161) and 3/12 flags at eps 0.10. Same window, bigger hop.
            return Macro.RIGHT_RUN_JUMP_MID
        return run

    def _rule(self, snapshot: MarioSnapshot, allowed: Sequence[Action]) -> Action:
        state = snapshot.to_state()
        player = state["player"]
        hazard = state["hazard"]
        terrain = state["terrain"]
        trajectory = state["trajectory"]
        enemy_distance = hazard["nearest_enemy_distance_pixels"]
        jump = Action.RIGHT_RUN_JUMP if Action.RIGHT_RUN_JUMP in allowed else allowed[0]
        run = Action.RIGHT_RUN if Action.RIGHT_RUN in allowed else allowed[0]

        # A single 8-frame macro is only a short hop. To clear a tall pipe / wide gap the
        # jump must be *held across decisions* to reach full height — so while airborne and
        # still rising (or mid-gap), keep holding the running jump.
        airborne = not player["grounded"]
        if airborne and (
            player["jump_phase"] in ("rising", "apex") or trajectory["crossing_known_gap"]
        ):
            return jump

        # Grounded: start a running jump for an imminent obstacle, gap, or a gap we are
        # committed to crossing. At full run speed the 3-tile "ahead" flags fire too late for
        # a wide gap, so also jump on the raw gap distance a couple tiles earlier.
        gap_distance = terrain.get("gap_distance_tiles")
        terrain_need = bool(
            hazard["jump_must_start_this_decision"]
            or terrain.get("obstacle_ahead")
            or terrain.get("gap_ahead")
            or trajectory["crossing_known_gap"]
            or (gap_distance is not None and gap_distance <= 4)  # jump a tile earlier at speed
        )
        # Enemy timing: jump only inside a tight *stomp window* (24–72px), not for any enemy
        # within 56px. Measured (2026-09-18): the old `< 56` rule took off too early and
        # landed Mario *into* the first advancing enemy — a deterministic death at x≈694 on
        # every seed. Restricting the jump to the window so the arc lands *on* the enemy moved
        # the deterministic death forward to x≈1153 (+66% reach), the next obstacle
        # (measured 2026-09-18).
        e = enemy_distance if enemy_distance is not None else 999.0
        enemy_need = hazard["enemy_ahead"] and 24.0 <= e <= 72.0
        return jump if (terrain_need or enemy_need) else run


class ReplayPolicy:
    """Replay a recorded run log through the harness with NO queries (findings #22 dashboard).

    The emulator is deterministic, so feeding the logged macro sequence back from the same
    reset reproduces the run frame for frame. Each ``choose`` returns the logged decision —
    action, confidence, probabilities, selection status and, when the log has it, the Adapt-1
    telemetry the server returned at the time — so the dashboard panel redraws as it was, with
    zero latency. Older logs without telemetry replay the moves only.
    """

    def __init__(self, log_path: Any) -> None:
        import json
        from pathlib import Path

        rows = [
            json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()
        ]
        if not rows:
            raise ValueError(f"empty run log: {log_path}")
        self.rows = rows
        self._i = 0

    @property
    def remaining(self) -> int:
        return len(self.rows) - self._i

    def rewind(self) -> None:
        """Start the replay over (dashboard Restart)."""
        self._i = 0

    @staticmethod
    def infer_cadence(rows: Sequence[Mapping[str, Any]]) -> str:
        """grounded if any logged action is a Macro-only name, else frame."""
        action_names = {a.value for a in Action}
        return "grounded" if any(r["action"] not in action_names for r in rows) else "frame"

    @property
    def has_apex(self) -> bool:
        """Whether the log carries in-air looks (runs played with an apex policy)."""
        return any(r.get("apex") for r in self.rows)

    def choose(
        self,
        snapshot: MarioSnapshot,
        actions: Sequence[Any],
        extra: Mapping[str, Any] | None = None,
    ) -> Decision:
        if actions and isinstance(actions[0], ApexChoice):
            # The in-air look of the decision just replayed: the logged choice, so the replay
            # takes the same arc (keep when the log predates the apex look).
            logged = (self.rows[self._i - 1].get("apex") or {}) if self._i > 0 else {}
            name = logged.get("choice", ApexChoice.KEEP.value)
            by_name = {a.value: a for a in actions}
            telemetry = logged.get("telemetry")
            if telemetry is not None:
                telemetry = {**telemetry, "learning": False, "replay": True}
            return Decision(
                action=by_name.get(name, actions[0]),
                confidence=float(logged.get("confidence") or 0.0),
                probabilities={},
                latency_ms=0.0,
                selection_status=logged.get("selection_status") or "replay",
                telemetry=telemetry,
            )
        if self._i >= len(self.rows):
            fallback = actions[0]
            return Decision(
                action=fallback,
                confidence=0.0,
                probabilities={},
                latency_ms=0.0,
                selection_status="replay-ended",
            )
        row = self.rows[self._i]
        self._i += 1
        by_name = {a.value: a for a in actions}
        action = by_name[row["action"]]
        telemetry = row.get("telemetry")
        if telemetry is not None:
            telemetry = {**telemetry, "learning": False, "replay": True}
        return Decision(
            action=action,
            confidence=float(row.get("confidence") or 0.0),
            probabilities=dict(row.get("probabilities") or {}),
            latency_ms=0.0,
            selection_status=row.get("selection_status")
            or ("replay" if telemetry is None else None),
            telemetry=telemetry,
        )
