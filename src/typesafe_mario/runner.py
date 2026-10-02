from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action
from .cadence import CadenceParser
from .dashboard import DashboardCommand, LiveDashboard
from .macros import action_set, execute
from .policy import Decision, Policy
from .reward import RewardConfig, shaped_reward
from .state import MarioStateParser


def _unwrap_ram(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        ram = getattr(current, "ram", None)
        if ram is not None:
            return ram
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    return None


def create_mario_env(env_id: str, render_mode: str = "human") -> Any:
    try:
        import gym_super_mario_bros  # noqa: F401
        import gymnasium as gym
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise RuntimeError(
            'Mario dependencies are missing. Install with: pip install -e ".[mario]"'
        ) from exc

    env = gym.make(env_id, render_mode=render_mode)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def _record_decision(
    log: Any,
    *,
    decision_index: int,
    snapshot: Any,
    decision: Decision,
    reward: float,
    terminated: bool,
    truncated: bool,
    apex: dict[str, Any] | None = None,
) -> None:
    record = {
        "decision": decision_index,
        "state": snapshot.to_state(),
        "debug_state": snapshot.to_debug_state(),
        "state_text": snapshot.to_text(),
        "action": decision.action.value,
        "confidence": decision.confidence,
        "probabilities": dict(decision.probabilities),
        "jump_needed_probability": decision.jump_needed_probability,
        "danger_score": decision.danger_score,
        "latency_ms": decision.latency_ms,
        "selection_status": decision.selection_status,
        "telemetry": dict(decision.telemetry) if decision.telemetry else None,
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        # The in-air look (design-notes §14a): what the apex policy chose and saw, when a jump
        # reached its apex. ReplayPolicy feeds it back so a replay takes the same arc.
        "apex": apex,
    }
    log.write(json.dumps(record, separators=(",", ":")) + "\n")
    log.flush()


def _run_realtime_dashboard(
    *,
    env: Any,
    dashboard: LiveDashboard,
    policy: Policy,
    parser: MarioStateParser,
    frame: Any,
    info: dict[str, Any],
    log: Any,
    frames_per_decision: int,
    max_decisions: int,
    screenshot_path: Path | None,
) -> None:
    actions = tuple(Action)
    active_decision: Decision | None = None
    pending: Future[Decision] | None = None
    pending_snapshot: Any = None
    pending_index = 0
    pending_request_frame = 0
    next_index = 0
    frame_index = 0
    last_request_frame = -frames_per_decision
    episode_reward = 0.0
    reward_since_decision = 0.0
    previous_action: Action | None = None
    previous_reward = 0.0
    previous_latency_ms = 0.0
    previous_response_delay_frames = 0
    screenshot_saved = False
    terminated = truncated = False

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="typesafe-jev") as executor:
        while True:
            decision_updated = False
            snapshot = parser.parse(
                info,
                _unwrap_ram(env),
                previous_action=previous_action.value if previous_action else None,
                previous_reward=previous_reward,
                previous_latency_ms=previous_latency_ms,
                previous_response_delay_frames=previous_response_delay_frames,
            )

            if pending is not None and pending.done():
                active_decision = pending.result()
                _record_decision(
                    log,
                    decision_index=pending_index,
                    snapshot=pending_snapshot,
                    decision=active_decision,
                    reward=reward_since_decision,
                    terminated=terminated,
                    truncated=truncated,
                )
                print(
                    f"#{pending_index:04d} x={pending_snapshot.x:04d} "
                    f"action={active_decision.action.value:<15} "
                    f"confidence={active_decision.confidence:.2f} "
                    f"latency={active_decision.latency_ms:.0f}ms"
                )
                pending = None
                reward_since_decision = 0.0
                decision_updated = True
                previous_latency_ms = active_decision.latency_ms
                previous_response_delay_frames = frame_index - pending_request_frame

            run_ended = bool(
                snapshot.dead
                or snapshot.clear
                or terminated
                or truncated
                or (next_index >= max_decisions and pending is None)
            )

            if (
                not run_ended
                and pending is None
                and next_index < max_decisions
                and frame_index - last_request_frame >= frames_per_decision
            ):
                pending_snapshot = snapshot
                pending_index = next_index
                pending = executor.submit(policy.choose, snapshot, actions)
                pending_request_frame = frame_index
                next_index += 1
                last_request_frame = frame_index

            if not run_ended:
                action = active_decision.action if active_decision else Action.NOOP
                if decision_updated and action in JUMP_ACTIONS and snapshot.grounded:
                    # A new jump macro needs a button-up edge before A is pressed again.
                    action = JUMP_RELEASE_ACTION[action]
                frame, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
                previous_action = action
                previous_reward = float(reward)
                reward_since_decision += float(reward)
                episode_reward += float(reward)
                frame_index += 1

            command = dashboard.draw(
                frame,
                snapshot,
                active_decision,
                decision_index=max(0, next_index - 1),
                episode_reward=episode_reward,
                waiting=pending is not None,
                run_ended=run_ended,
            )
            if screenshot_path is not None and active_decision is not None and not screenshot_saved:
                dashboard.save(screenshot_path)
                screenshot_saved = True
            if command == DashboardCommand.QUIT:
                break
            if command == DashboardCommand.RESTART:
                if pending is not None:
                    pending.cancel()
                frame, info = env.reset()
                parser.reset()
                active_decision = None
                pending = None
                pending_snapshot = None
                pending_index = 0
                pending_request_frame = 0
                next_index = 0
                frame_index = 0
                last_request_frame = -frames_per_decision
                episode_reward = 0.0
                reward_since_decision = 0.0
                previous_action = None
                previous_reward = 0.0
                previous_latency_ms = 0.0
                previous_response_delay_frames = 0
                terminated = truncated = False
                print("--- Restarted ---")


def run_episode(
    *,
    env_id: str,
    policy: Policy,
    frames_per_decision: int,
    max_decisions: int,
    seed: int,
    artifacts_dir: Path,
    display: str = "dashboard",
    screenshot_path: Path | None = None,
    episode_id: str | None = None,
    stall_timeout: int | None = None,
    reward_config: RewardConfig | None = None,
    cadence: str = "frame",
    apex_policy: Policy | None = None,
) -> Path:
    """Play one episode. ``cadence`` = ``"frame"`` (one 8-frame ``Action`` per decision, the
    #20 baseline) or ``"grounded"`` (one committed ``Macro`` per decision, decided only on the
    ground — findings #21, ``macros.py``). Both go through ``macros.execute``.

    ``apex_policy`` (grounded cadence, design-notes §14a): a second policy asked once at the top
    of every jump for an ``ApexChoice``. If it has ``observe`` it gets its own feedback — the
    landing outcome, charged to the apex choice — while the takeoff decision is charged only
    with the progress to the apex (the same split ``record_demos.py --apex`` writes)."""
    if frames_per_decision < 1:
        raise ValueError("frames_per_decision must be at least 1")
    from .adapt1_policy import ALL_FEATURE_NAMES  # local: demos imports this module
    from .demos import apex_hook, build_rows

    choices = action_set(cadence)
    use_apex = apex_policy is not None and cadence == "grounded"

    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    render_mode = "human" if display == "game" else "rgb_array"
    env = create_mario_env(env_id, render_mode=render_mode)
    dashboard = LiveDashboard() if display == "dashboard" else None
    realtime = display == "dashboard" and cadence == "frame"
    # The decision loop parses once per decision: a CadenceParser stamps the interval so the
    # v2 arms can convert per-parse deltas to per-frame units (findings #33). The realtime
    # viewer parses every frame and keeps the plain parser.
    parser = (MarioStateParser if realtime else CadenceParser)(
        decision_horizon_frames=frames_per_decision
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    episode_id = episode_id or f"ep-{timestamp}"
    log_path = artifacts_dir / f"run-{timestamp}.jsonl"

    frame, info = env.reset(seed=seed)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            if dashboard and cadence == "frame":
                # The overlapped (query-while-playing) loop only knows the 7 frame actions and
                # never sends feedback. Grounded macros use the sequential loop below with the
                # dashboard redrawn on every emulator frame (findings #22 telemetry panel).
                _run_realtime_dashboard(
                    env=env,
                    dashboard=dashboard,
                    policy=policy,
                    parser=parser,
                    frame=frame,
                    info=info,
                    log=log,
                    frames_per_decision=frames_per_decision,
                    max_decisions=max_decisions,
                    screenshot_path=screenshot_path,
                )
                return log_path

            actions = choices
            restart = True
            while restart:
                restart = False
                # Optional feedback hook (e.g. Adapt1Policy). A learning brain needs the
                # *next* state and the outcome, so we parse each snapshot exactly once and
                # reuse it both as this transition's next_state and the next iteration's
                # state — parsing twice would double-advance the parser's internal counters.
                observe = getattr(policy, "observe", None)
                apex_observe = getattr(apex_policy, "observe", None) if use_apex else None
                start_episode = getattr(policy, "start_episode", None)
                if callable(start_episode):
                    start_episode(episode_id)  # TCP: one ordered history per episode
                apex_start = getattr(apex_policy, "start_episode", None) if use_apex else None
                if callable(apex_start):
                    apex_start(episode_id)
                snapshot = parser.parse(
                    info,
                    _unwrap_ram(env),
                    previous_action=None,
                    previous_reward=0.0,
                    previous_latency_ms=0.0,
                    previous_response_delay_frames=0,
                )
                best_x = snapshot.x
                stalled_decisions = 0
                screenshot_saved = False
                episode_return = 0.0  # cumulative learn reward, shown on the dashboard
                last_decision = None  # kept on screen while the next query is in flight (no blink)
                for decision_index in range(max_decisions):
                    if snapshot.dead or snapshot.clear:
                        break

                    if stall_timeout is not None and stalled_decisions >= stall_timeout:
                        # Stuck-but-alive (e.g. headbutting a pipe): end the episode so training
                        # spends its budget on fresh experience, not the decision cap. This is a
                        # truncation (Mario didn't die) — the last transition's terminal stays False.
                        print(f"--- stalled {stalled_decisions} decisions, ending episode ---")
                        break

                    # Show the state while the query is in flight (the game pauses; the header
                    # says "Querying"). Quit closes the window and ends the episode.
                    if dashboard and (
                        dashboard.draw(
                            frame,
                            snapshot,
                            last_decision,
                            decision_index=decision_index,
                            episode_reward=episode_return,
                            waiting=True,
                        )
                        == DashboardCommand.QUIT
                    ):
                        break
                    decision = policy.choose(snapshot, actions)
                    last_decision = decision
                    apex_state: dict[str, Any] = {}

                    def _on_frame(f: Any, _d=decision, _s=snapshot, _i=decision_index) -> None:
                        nonlocal frame, screenshot_saved
                        frame = f
                        if dashboard:
                            dashboard.draw(
                                f,
                                _s,
                                _d,
                                decision_index=_i,
                                episode_reward=episode_return,  # noqa: B023 — per-frame, same decision
                            )
                            if screenshot_path is not None and not screenshot_saved and _i >= 1:
                                dashboard.save(
                                    screenshot_path
                                )  # first frame with a populated panel
                                screenshot_saved = True

                    # One execution path for both cadences (macros.execute): the frame cadence keeps
                    # the frame-0 button-up edge on a grounded jump (a fresh press fires; held A
                    # across airborne decisions keeps rising) — identical to demos.py / record.py.
                    execution = execute(
                        env,
                        decision.action,
                        grounded=snapshot.grounded,
                        frames_per_decision=frames_per_decision,
                        ram=_unwrap_ram(env),
                        on_frame=_on_frame,
                        apex=apex_hook(parser, env, apex_policy, decision, apex_state)
                        if use_apex
                        else None,
                    )
                    total_reward = execution.total_reward
                    terminated, truncated, info = (
                        execution.terminated,
                        execution.truncated,
                        execution.info,
                    )
                    apex_fired = execution.apex is not None and bool(apex_state)
                    apex_decision = apex_state["decision"] if apex_fired else None

                    next_snapshot = parser.parse(
                        info,
                        _unwrap_ram(env),
                        previous_action=(
                            apex_decision.action.value if apex_fired else decision.action.value
                        ),
                        previous_reward=(
                            total_reward - execution.reward_to_apex if apex_fired else total_reward
                        ),
                        previous_latency_ms=decision.latency_ms,
                        previous_response_delay_frames=0,
                        frames=(
                            execution.frames - execution.frames_to_apex
                            if apex_fired
                            else execution.frames
                        ),
                    )
                    apex_log = None
                    if apex_fired:
                        ev, ex = execution.apex, apex_state["extra"]
                        apex_log = {
                            "choice": apex_decision.action.value,
                            "selection_status": apex_decision.selection_status,
                            "confidence": apex_decision.confidence,
                            "telemetry": dict(apex_decision.telemetry)
                            if apex_decision.telemetry
                            else None,
                            "x": ev.x,
                            "y": ev.y,
                            "vx": ex["vx"],
                            "rise": ex["rise"],
                            "height": ex["height"],
                            "reward_to_apex": execution.reward_to_apex,
                        }
                        # Show the in-air look on the takeoff decision's panel entry.
                        tel = dict(decision.telemetry or {})
                        tel["apex"] = {
                            "choice": apex_decision.action.value,
                            "status": apex_decision.selection_status,
                            "values": (apex_decision.telemetry or {}).get("values") or {},
                            "x": ev.x,
                        }
                        last_decision = replace(decision, telemetry=tel)

                    _record_decision(
                        log,
                        decision_index=decision_index,
                        snapshot=snapshot,
                        decision=decision,
                        reward=total_reward,
                        terminated=terminated,
                        truncated=truncated,
                        apex=apex_log,
                    )
                    terminal = bool(terminated or next_snapshot.dead or next_snapshot.clear)
                    if apex_fired and (observe is not None or apex_observe is not None):
                        # Two feedbacks for one jump (design-notes §14a): the takeoff decision
                        # gets the progress to the apex (never terminal), the apex choice gets the
                        # landing outcome — the same split the offline rows carry.
                        rows = build_rows(
                            snapshot=snapshot,
                            decision=decision,
                            execution=execution,
                            apex_state=apex_state,
                            next_snapshot=next_snapshot,
                            terminal=terminal,
                            names=ALL_FEATURE_NAMES,
                            reward_config=reward_config,
                            episode_id=episode_id,
                            first_step=decision_index,
                            tag_kind=True,
                        )
                        takeoff_reward, apex_reward = (
                            float(rows[0]["reward"]),
                            float(rows[1]["reward"]),
                        )
                        episode_return += takeoff_reward + apex_reward
                        if observe is not None:
                            apex_next = apex_state["snapshot"].to_state()
                            apex_next["apex"] = dict(apex_state["extra"])
                            observe(
                                next_state=apex_next,
                                reward=takeoff_reward,
                                terminal=False,
                                truncated=False,
                                episode_id=episode_id,
                                step=decision_index,
                            )
                        if apex_observe is not None:
                            apex_observe(
                                next_state=next_snapshot.to_state(),
                                reward=apex_reward,
                                terminal=terminal,
                                truncated=bool(truncated),
                                episode_id=episode_id,
                                step=decision_index,
                            )
                        apex_state_line = (
                            f"      apex x={apex_log['x']} {apex_log['choice']:<14} "
                            f"status={apex_log['selection_status']} "
                            f"r_takeoff={takeoff_reward:.4f} r_apex={apex_reward:.4f}"
                        )
                        print(apex_state_line)
                    elif observe is not None:
                        if reward_config is None:
                            learn_reward = total_reward
                        else:
                            nav = snapshot.navigation_features()
                            learn_reward = shaped_reward(
                                dx=next_snapshot.x - snapshot.x,
                                terminal=terminal,
                                cleared=next_snapshot.clear,
                                dead=next_snapshot.dead,
                                obstacle_ahead=bool(nav.get("obstacle_ahead")),
                                grounded=snapshot.grounded,
                                left_ground=snapshot.grounded and not next_snapshot.grounded,
                                frames=execution.frames,
                                config=reward_config,
                            )
                        episode_return += float(learn_reward)
                        observe(
                            next_state=next_snapshot.to_state(),
                            reward=learn_reward,
                            terminal=terminal,
                            truncated=bool(truncated),
                            episode_id=episode_id,
                            step=decision_index,
                        )
                    print(
                        f"#{decision_index:04d} x={snapshot.x:04d} "
                        f"action={decision.action.value:<15} "
                        f"confidence={decision.confidence:.2f} latency={decision.latency_ms:.0f}ms"
                    )

                    if next_snapshot.x > best_x:
                        best_x = next_snapshot.x
                        stalled_decisions = 0
                    else:
                        stalled_decisions += 1

                    snapshot = next_snapshot
                    if terminated or truncated:
                        break

                if dashboard:
                    # Hold the final frame: Restart / R replays the episode from the start
                    # (a ReplayPolicy rewinds; a live policy starts a new episode), Esc / Q closes.
                    if snapshot.clear:
                        reason = f"reached the flag at x={snapshot.x}"
                    elif snapshot.dead or terminated:
                        reason = f"died at x={snapshot.x} (max {best_x})"
                    elif getattr(policy, "remaining", None) == 0:
                        reason = f"replay log ended at x={snapshot.x} (max {best_x})"
                    elif stall_timeout is not None and stalled_decisions >= stall_timeout:
                        reason = f"stalled at x={snapshot.x} for {stalled_decisions} decisions"
                    else:
                        reason = f"decision cap reached at x={snapshot.x}"
                    while True:
                        cmd = dashboard.draw(
                            frame,
                            snapshot,
                            decision,
                            decision_index=decision_index + 1,
                            episode_reward=episode_return,
                            run_ended=True,
                            ended_reason=reason,
                        )
                        if cmd == DashboardCommand.QUIT:
                            break
                        if cmd == DashboardCommand.RESTART:
                            frame, info = env.reset(seed=seed)
                            parser.reset()
                            rewind = getattr(policy, "rewind", None)
                            if callable(rewind):
                                rewind()
                            restart = True
                            break
    finally:
        if dashboard:
            dashboard.close()
        env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    return log_path
