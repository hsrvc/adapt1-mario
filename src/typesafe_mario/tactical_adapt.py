"""Adapt-1 as the tactical decision layer (findings #18, Option 2 → the online experiment).

Wires Adapt-1 into the *tactical* loop from `tactical.py`: at each event it picks a **skill**
(ADVANCE / JUMP_CLEAR) from the coarse symbolic state, executes it (timing owned by code),
and attributes the skill's outcome back as feedback. This is the mode + altitude the whole
workstream was circling:

- **Right altitude** — a discrete skill choice over ~48 coarse states, not motor timing. The
  reward is now *immediately discriminative* (at an obstacle, JUMP_CLEAR clears → +Δx while
  ADVANCE jams → ~0), which is exactly the weak-immediate-signal problem the frame-level task
  had and this level does not.
- **Right mode** — online continual learning with exploration + feedback, the mode Adapt-1's
  demonstrated wins use (Alchemy: useful behaviour from live decisions). `train_tactical_curve`
  interleaves online learning with frozen-eval checkpoints to produce the learning *curve*.

Reuses the *verified* bandit query/feedback body shapes from `adapt1_policy` (the 7-macro
bandit was confirmed live 2026-09-17). Only the contract changes: 2 skills, a categorical
tactical context, `values.reward` from the skill's outcome.
"""

from __future__ import annotations

import random
from typing import Any

from .adapt1_policy import QUESTION, _parse_feedback, _parse_query
from .runner import _unwrap_ram, create_mario_env
from .state import MarioStateParser
from .tactical import (
    SkillResult,
    TacticalSkill,
    encode_tactical_state,
    execute_skill,
)

TACTICAL_RELATION = "act"
TACTICAL_SKILLS = (TacticalSkill.ADVANCE, TacticalSkill.JUMP_CLEAR)
TACTICAL_FEATURES = ("mario", "obstacle", "gap", "enemy")
TACTICAL_OUTCOMES = ("advanced", "stalled", "died", "cleared")

_SKILL_DX_SCALE = 96.0  # px a skill covers that maps to reward 1.0 (skills move ~72–96px)


def build_tactical_domain_config(domain_id: str) -> dict[str, Any]:
    """A contextual bandit over the two skills, conditioned on the coarse tactical state."""
    return {
        "domain_id": domain_id,
        "schema": {
            "event_types": ["decision"],
            "relations": [TACTICAL_RELATION],
            "signals": ["reward", *TACTICAL_FEATURES],
        },
        "hypotheses": [
            {
                "name": s.value,
                "relation": TACTICAL_RELATION,
                "policy": s.value,
                "predicts": ["advanced"],
            }
            for s in TACTICAL_SKILLS
        ],
        "learning": {
            "enabled": True,
            "context": {
                "feature_paths": [f"values.{f}" for f in TACTICAL_FEATURES],
                "max_samples": 4096,
            },
            "reward": {
                "aggregation": "weighted_mean",
                "components": [{"field": "values.reward", "goal": "maximize", "weight": 1.0}],
            },
            "policy": {"exploration_mode": "ucb", "min_context_observations": 2},
        },
        "query_templates": {"feedback_outcomes": list(TACTICAL_OUTCOMES)},
    }


def tactical_reward(result: SkillResult) -> float:
    """Immediate, discriminative reward for a skill's outcome: normalised Δx, with terminal
    clear/death spikes. A skill that clears an obstacle earns its Δx; one that jams earns ~0."""
    r = max(-1.0, min(1.0, result.dx / _SKILL_DX_SCALE))
    if result.clear:
        r += 1.0
    elif result.dead:
        r -= 1.0
    return max(-1.0, min(1.0, r))


def tactical_outcome(result: SkillResult) -> str:
    if result.clear:
        return "cleared"
    if result.dead:
        return "died"
    return "advanced" if result.dx > 8 else "stalled"


class Adapt1TacticalPolicy:
    """Queries an Adapt-1 tactical bandit for a skill; attributes the outcome as feedback.

    Implements ``choose(snapshot, state) -> TacticalSkill`` so it drops straight into
    ``tactical.run_tactical_episode`` for **frozen eval** (set ``allow_exploration=False`` and
    never call :meth:`reward`). For **online learning**, use :meth:`select` + :meth:`reward`.
    """

    def __init__(
        self,
        client: Any,
        domain_id: str,
        *,
        allow_exploration: bool = True,
        rng_seed: int = 0,
        run_id: str = "tactical",
    ) -> None:
        self._client = client
        self._domain = domain_id
        self._allow_exploration = allow_exploration
        self._rng = random.Random(rng_seed)
        self._run_id = run_id

    def select(self, state: dict[str, str]) -> tuple[TacticalSkill, str | None]:
        """Query the domain for a skill. Returns (skill, decision_id). On abstention, commit to
        the learner's own argmax of the scores (its best guess), NOT a random skill — random
        fallback corrupted eval (findings #20; the same bug that faked the x=723 wall)."""
        body = {
            "session_id": "ignored",
            "question": QUESTION,
            "context": {"values": state},
            "top_k": len(TACTICAL_SKILLS),
            "allow_exploration": self._allow_exploration,
        }
        _s, resp = self._client.query(self._domain, body)
        parsed = _parse_query(resp)
        try:
            skill = TacticalSkill(str(parsed["policy"]))
        except ValueError:
            skill = self._best_effort(parsed["scores"])
        return skill, parsed["decision_id"]

    def _best_effort(self, scores: Any) -> TacticalSkill:
        """Argmax of the learner's scores over the skills; default ADVANCE (keep moving) if
        there are no scores. Never a coin flip — so eval measures the LEARNED policy."""
        valid = {s.value for s in TACTICAL_SKILLS}
        best, best_score = None, float("-inf")
        for s in scores or []:
            if isinstance(s, dict) and s.get("policy") in valid:
                sc = s.get("score")
                if isinstance(sc, (int, float)) and sc > best_score:
                    best, best_score = str(s["policy"]), float(sc)
        return TacticalSkill(best) if best is not None else TacticalSkill.ADVANCE

    def choose(self, snapshot: Any, state: dict[str, str]) -> TacticalSkill:
        """`TacticalPolicy` protocol — for frozen eval (query only, no feedback)."""
        return self.select(state)[0]

    def reward(
        self,
        *,
        decision_id: str | None,
        skill: TacticalSkill,
        state: dict[str, str],
        result: SkillResult,
        episode_id: str,
        step: int,
    ) -> dict[str, Any]:
        """Attribute the skill's outcome. No-op if the query abstained (no decision_id)."""
        if decision_id is None:
            return {"applied": None, "observations": None}
        body = {
            "session_id": "ignored",
            "decision_id": decision_id,
            "relation": TACTICAL_RELATION,
            "policy": skill.value,
            "feedback_kind": "execution",
            "outcome": tactical_outcome(result),
            "context": {"values": state},
            "values": {"reward": tactical_reward(result)},
            "metadata": {
                "run_id": self._run_id,
                "episode_id": episode_id,
                "step": step,
                "terminal": bool(result.terminal),
            },
        }
        _s, resp = self._client.feedback(self._domain, body)
        return _parse_feedback(resp)


def run_online_tactical_episode(
    client: Any,
    domain_id: str,
    policy: Adapt1TacticalPolicy,
    *,
    env_id: str = "SuperMarioBros-1-1-v0",
    seed: int = 0,
    episode_id: str = "online-ep",
    max_decisions: int = 80,
    stall_timeout: int = 12,
) -> dict[str, Any]:
    """One online episode: query → execute skill → reward, learning as it goes. Returns stats
    incl. `records_spent` (= feedback calls = tactical decisions)."""
    env = create_mario_env(env_id, render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    decisions = 0
    records = 0
    try:
        _frame, info = env.reset(seed=seed)
        snapshot = parser.parse(info, _unwrap_ram(env), previous_action=None)
        best_x = snapshot.x
        stalled = 0
        for step in range(max_decisions):
            if snapshot.dead or snapshot.clear:
                break
            state = encode_tactical_state(snapshot)
            skill, decision_id = policy.select(state)
            result = execute_skill(env, parser, snapshot, skill)
            policy.reward(
                decision_id=decision_id,
                skill=skill,
                state=state,
                result=result,
                episode_id=episode_id,
                step=step,
            )
            decisions += 1
            records += 1 if decision_id is not None else 0
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
        "decisions": decisions,
        "records_spent": records,
        "cleared_pipe": best_x > 723,
    }
