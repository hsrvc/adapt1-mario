"""Record demonstration transitions to a dataset — the offline, FREE half of warm-start.

`record_demonstrations` plays episodes with any `Policy` (e.g. `DiversifiedHeuristicPolicy`)
entirely locally — no API, no quota — and writes one transition per decision as JSONL:

    {"episode_id", "step", "context": {features}, "policy": macro,
     "reward": float, "next_state": {features}, "terminal": bool, "outcome": label}

This is the *same* information the sequential learner consumes, projected to a chosen
feature arm (full / perception). It is **route-agnostic** on purpose: a separate ingestion
step (which spends Records) decides how to feed it to Adapt-1 (/feedback vs /events vs
/batch). Recording here costs nothing, so we can generate and inspect the dataset — and its
*quality* (max-x, flag rate) — before spending any quota.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .adapt1_policy import (
    APEX_EXTRA_NAMES,
    FEATURE_NAMES,
    POSITION_FEATURE_NAMES,
    _outcome_label,
    context_features,
)
from .cadence import CadenceParser
from .macros import action_set, apex_choices, execute
from .policy import Policy
from .reward import RewardConfig, shaped_reward
from .runner import _unwrap_ram, create_mario_env
from .state import MarioStateParser  # noqa: F401  (re-exported for callers)


def build_rows(
    *,
    snapshot: Any,
    decision: Any,
    execution: Any,
    apex_state: dict[str, Any] | None,
    next_snapshot: Any,
    terminal: bool,
    names: Sequence[str],
    reward_config: RewardConfig | None,
    episode_id: str,
    first_step: int,
    tag_kind: bool,
) -> list[dict[str, Any]]:
    """The dataset rows of one decision: one, or takeoff -> apex -> landing when the apex look
    fired (``apex_state`` = ``{"snapshot", "extra", "decision"}`` captured by the hook). The
    jump's progress is split between the two rows; steps are ``first_step``, ``first_step+1``.
    Shared by ``record_demonstrations`` and ``scripts/apex_coverage.py``."""

    def features_of(snap: Any, extra: dict[str, Any] | None) -> dict[str, Any]:
        state = snap.to_state()
        if extra is None:
            return context_features(state, names)
        state["apex"] = dict(extra)
        return context_features(state, names, projected=False)  # raw terrain: the in-air look

    apex_fired = execution.apex is not None and apex_state is not None
    if apex_fired:
        a = apex_state
        segments = [
            (
                snapshot,
                None,
                decision,
                a["snapshot"],
                a["extra"],
                execution.reward_to_apex,
                execution.frames_to_apex,
                False,
                "takeoff",
            ),
            (
                a["snapshot"],
                a["extra"],
                a["decision"],
                next_snapshot,
                None,
                execution.total_reward - execution.reward_to_apex,
                execution.frames - execution.frames_to_apex,
                terminal,
                "apex",
            ),
        ]
    else:
        segments = [
            (
                snapshot,
                None,
                decision,
                next_snapshot,
                None,
                execution.total_reward,
                execution.frames,
                terminal,
                "takeoff",
            )
        ]
    rows = []
    for i, (frm, frm_extra, dec, to, to_extra, raw, seg_frames, seg_terminal, kind) in enumerate(
        segments
    ):
        to_state_full = to.to_state()
        if reward_config is None:
            learn_reward = round(raw, 4)
        else:
            nav = frm.navigation_features()
            learn_reward = round(
                shaped_reward(
                    dx=to.x - frm.x,
                    terminal=seg_terminal,
                    cleared=seg_terminal and to.clear,
                    dead=seg_terminal and to.dead,
                    obstacle_ahead=bool(nav.get("obstacle_ahead")),
                    grounded=frm.grounded,
                    left_ground=frm.grounded and not to.grounded,
                    frames=seg_frames,
                    config=reward_config,
                ),
                4,
            )
        record = {
            "episode_id": episode_id,
            "step": first_step + i,
            "context": features_of(frm, frm_extra),
            "policy": dec.action.value,
            "reward": learn_reward,
            "next_state": features_of(to, to_extra),
            "terminal": seg_terminal,
            # outcome label keys off the *raw* reward sign (advanced/setback) — it is a
            # categorical audit label, independent of the numeric shaping.
            "outcome": _outcome_label(to_state_full, raw, seg_terminal),
            # True when the teacher's epsilon fired: an off-policy transition, the kind that
            # gives rarely-used macros real observations (coverage rows).
            "explored": dec.selection_status == "explored",
            # Emulator frames this row's decision took (2026-09-23, findings #29): a per-decision
            # return favours long macros; per-frame returns need this.
            "frames": int(seg_frames),
        }
        if tag_kind:
            record["kind"] = kind
        rows.append(record)
    return rows


def apex_hook(parser: Any, env: Any, policy: Any, decision: Any, sink: dict[str, Any]) -> Any:
    """The in-air look as an ``execute`` hook: parse the apex as a decision point, attach the
    arc facts, ask ``policy`` for an ``ApexChoice``; leaves what it saw in ``sink``."""

    def on_apex(event: Any) -> Any:
        ram_now = _unwrap_ram(env)
        apex_snapshot = parser.parse(
            event.info,
            ram_now,
            previous_action=decision.action.value,
            previous_reward=event.reward,
            frames=event.frame,  # the rise: frames since the takeoff parse (findings #33)
        )
        extra = {
            "vx": event.vx,
            "rise": event.frame,
            "height": parser.floor_height_tiles(
                ram_now, event.x, int(event.info.get("y_pixel", 0))
            ),
        }
        apex_decision = policy.choose(apex_snapshot, apex_choices(), extra=extra)
        sink.update(snapshot=apex_snapshot, extra=extra, decision=apex_decision)
        return apex_decision.action

    return on_apex


def record_demonstrations(
    env_id: str,
    policy: Policy,
    out_path: Any,
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    episodes: int = 50,
    frames_per_decision: int = 8,
    max_decisions: int = 200,
    stall_timeout: int | None = 20,
    seed: int = 0,
    reward_config: RewardConfig | None = None,
    cadence: str = "frame",
    apex: bool = False,
    actions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Play `episodes` and write every transition to `out_path` (JSONL). Returns stats.

    ``apex`` (grounded cadence, design-notes §14a): every jump macro becomes TWO rows — the
    takeoff row (grounded context -> apex state, ``kind: "takeoff"``) and the apex row (apex
    context with the ``apex_*`` extras -> landing, ``kind: "apex"``, policy = the
    ``ApexChoice`` the teacher took). Rows carry the feature superset (``feature_names`` +
    ``APEX_EXTRA_NAMES``); the ingest projects each domain's arm out of it. Steps stay
    contiguous within the episode so client-side k-step returns flow through the apex row.

    Each episode ends on death / flag / stall-timeout / max-decisions. Parses each snapshot
    exactly once (reused as this transition's next_state and the next step's state).

    ``reward_config`` selects the reward the transition records: ``None`` keeps the raw gym
    reward (the finding-#15 baseline), a ``RewardConfig`` writes the shaped composite
    (design-notes §5) computed from Δx and terminal flag/death. The recorded ``reward`` is
    what a warm-start ingestion feeds to ``values.reward``, so this is where reward-shaping
    enters the offline dataset.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".manifest.jsonl")
    # ``actions``: the choice set the teacher (and its ε draws) sees. The 8 macros of #28 reproduce
    # the 2026-09-22 demos exactly; the default set gained `jump` on 2026-09-23 and the ε sequence
    # diverges from the first draw (measured 2026-09-25: 1 of 30 episodes identical).
    actions = tuple(actions) if actions is not None else action_set(cadence)
    use_apex = bool(apex) and cadence == "grounded"
    # Rows carry the recorded superset: the arm's names, the apex extras when the apex look is on,
    # and the position features always (§14b: one dataset feeds the with/without-position pair).
    names = tuple(
        dict.fromkeys(
            (*feature_names, *(APEX_EXTRA_NAMES if use_apex else ()), *POSITION_FEATURE_NAMES)
        )
    )

    transitions = 0
    apex_rows = 0
    apex_choice_counts: dict[str, int] = {}
    flag_episodes = 0
    death_episodes = 0
    max_x_values: list[int] = []
    manifest = manifest_path.open("w", encoding="utf-8")

    with out.open("w", encoding="utf-8") as sink:
        for ep in range(episodes):
            env = create_mario_env(env_id, render_mode="rgb_array")
            parser = CadenceParser(decision_horizon_frames=frames_per_decision)
            episode_id = f"demo-ep{ep:04d}"
            reached_flag = died = False
            best_x = 0
            stalled = 0
            ep_transitions = 0
            row_step = 0
            try:
                _frame, info = env.reset(seed=seed + ep)
                snapshot = parser.parse(info, _unwrap_ram(env), previous_action=None)
                best_x = snapshot.x
                for step in range(max_decisions):
                    if snapshot.dead or snapshot.clear:
                        break
                    if stall_timeout is not None and stalled >= stall_timeout:
                        break

                    decision = policy.choose(snapshot, actions)
                    apex_state: dict[str, Any] = {}
                    execution = execute(
                        env,
                        decision.action,
                        grounded=snapshot.grounded,
                        frames_per_decision=frames_per_decision,
                        ram=_unwrap_ram(env),
                        apex=apex_hook(parser, env, policy, decision, apex_state)
                        if use_apex
                        else None,
                    )
                    total_reward = execution.total_reward
                    terminated, _truncated = execution.terminated, execution.truncated
                    info = execution.info
                    apex_fired = execution.apex is not None and bool(apex_state)

                    next_snapshot = parser.parse(
                        info,
                        _unwrap_ram(env),
                        previous_action=(
                            apex_state["decision"].action.value
                            if apex_fired
                            else decision.action.value
                        ),
                        previous_reward=(
                            total_reward - execution.reward_to_apex if apex_fired else total_reward
                        ),
                        frames=(
                            execution.frames - execution.frames_to_apex
                            if apex_fired
                            else execution.frames
                        ),
                    )
                    terminal = bool(terminated or next_snapshot.dead or next_snapshot.clear)

                    rows = build_rows(
                        snapshot=snapshot,
                        decision=decision,
                        execution=execution,
                        apex_state=apex_state if apex_fired else None,
                        next_snapshot=next_snapshot,
                        terminal=terminal,
                        names=names,
                        reward_config=reward_config,
                        episode_id=episode_id,
                        first_step=row_step,
                        tag_kind=use_apex,
                    )
                    for record in rows:
                        sink.write(json.dumps(record, separators=(",", ":")) + "\n")
                        transitions += 1
                        ep_transitions += 1
                        row_step += 1
                        if record.get("kind") == "apex":
                            apex_rows += 1
                            name = record["policy"]
                            apex_choice_counts[name] = apex_choice_counts.get(name, 0) + 1

                    if next_snapshot.x > best_x:
                        best_x = next_snapshot.x
                        stalled = 0
                    else:
                        stalled += 1
                    reached_flag = reached_flag or next_snapshot.clear
                    died = died or next_snapshot.dead
                    snapshot = next_snapshot
                    if terminal:
                        break
            finally:
                env.close()
            flag_episodes += int(reached_flag)
            death_episodes += int(died)
            max_x_values.append(best_x)
            manifest.write(
                json.dumps(
                    {
                        "episode_id": episode_id,
                        "transitions": ep_transitions,
                        "max_x": best_x,
                        "reached_flag": reached_flag,
                        "died": died,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            manifest.flush()

    manifest.close()
    return {
        "episodes": episodes,
        "transitions": transitions,
        "flag_episodes": flag_episodes,
        "death_episodes": death_episodes,
        "flag_rate": round(flag_episodes / episodes, 3) if episodes else 0.0,
        "max_x_mean": round(sum(max_x_values) / len(max_x_values), 1) if max_x_values else 0,
        "max_x_best": max(max_x_values) if max_x_values else 0,
        "feature_names": list(names),
        "apex": use_apex,
        "apex_rows": apex_rows,
        "apex_choices": apex_choice_counts,
        "out": str(out),
        "manifest": str(manifest_path),
    }
