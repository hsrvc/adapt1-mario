"""Record a Mario episode to an animated GIF — for the "learning → competent" reel.

`record_episode` runs ONE episode headless (rgb_array), driving it with any `Policy`,
collecting every emulator frame and writing a GIF. It calls only `policy.choose` — never
`observe` — so a frozen-eval recording is **read-only**: it queries the domain but sends
no feedback, and therefore never adds transitions or contaminates the training buffer.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cadence import CadenceParser
from .macros import action_set, apex_choices, execute
from .policy import Policy
from .runner import _unwrap_ram, create_mario_env


def write_gif(frames: list[np.ndarray], out_path: Any, *, fps: int = 30, decimate: int = 2) -> int:
    """Encode RGB frames to a looping GIF. Returns the number of GIF frames written.

    Frames are palette-quantised (`P` mode) — the reliable idiom for animated GIFs — and
    every `decimate`-th frame is kept to bound size. Callers pass *copies* of frames; the
    emulator reuses its buffer.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without pillow
        raise RuntimeError(
            'Recording needs pillow. Install with: pip install -e ".[mario]"'
        ) from exc

    if not frames:
        raise ValueError("no frames to write")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, decimate)
    imgs = [
        Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames[::step]
    ]
    imgs[0].save(
        out,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=0,
        disposal=2,
    )
    return len(imgs)


def record_episode(
    env_id: str,
    policy: Policy,
    out_path: Any,
    *,
    frames_per_decision: int = 8,
    max_decisions: int = 1200,
    seed: int = 0,
    decimate: int = 2,
    fps: int = 30,
    stall_timeout: int | None = None,
    cadence: str = "frame",
    apex_policy: Any = None,
    actions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Play one episode with `policy`, record it to `out_path` (GIF), return stats.

    ``apex_policy`` (grounded cadence, design-notes §14a): a second policy asked once at the
    top of every jump, with the apex snapshot and the ``apex_*`` extras, for an ``ApexChoice``.

    Does not call `policy.observe` (frozen eval) and does not close the policy — the
    caller owns the policy's lifecycle so it can be reused across checkpoints.
    ``cadence`` selects 8-frame ``Action``s or grounded committed ``Macro``s (macros.py).
    """
    env = create_mario_env(env_id, render_mode="rgb_array")
    parser = CadenceParser(decision_horizon_frames=frames_per_decision)
    # ``actions``: restrict the choices (e.g. the 8 macros of #28, without `jump`) — the policy's
    # fallback on abstention picks among these, so they must match the domain's hypotheses.
    actions = tuple(actions) if actions is not None else action_set(cadence)
    frames: list[np.ndarray] = []
    decisions = 0
    reward_total = 0.0
    apex_counts: dict[str, int] = {}
    info: dict[str, Any] = {}
    # Per-decision trace (2026-09-25): what the policy saw and chose, so a frozen eval's death
    # or stall can be replayed from the record instead of re-derived from the GIF.
    trace: list[dict[str, Any]] = []
    try:
        _frame, info = env.reset(seed=seed)
        start_episode = getattr(policy, "start_episode", None)
        if callable(start_episode):
            # TCP: a fresh, never-reused episode id per eval run ((episode_id, step) is a dedup key)
            from datetime import UTC, datetime

            start_episode(f"eval-{seed}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}")
        snapshot = parser.parse(info, _unwrap_ram(env), previous_action=None)
        best_x = snapshot.x
        stalled = 0
        while decisions < max_decisions and not (snapshot.dead or snapshot.clear):
            if stall_timeout is not None and stalled >= stall_timeout:
                break
            decision = policy.choose(snapshot, actions)
            last_action = decision.action.value

            def on_apex(event: Any, _decision: Any = decision) -> Any:
                nonlocal last_action
                ram_now = _unwrap_ram(env)
                apex_snapshot = parser.parse(
                    event.info,
                    ram_now,
                    previous_action=_decision.action.value,
                    previous_reward=event.reward,
                    frames=event.frame,
                )
                extra = {
                    "vx": event.vx,
                    "rise": event.frame,
                    "height": parser.floor_height_tiles(
                        ram_now, event.x, int(event.info.get("y_pixel", 0))
                    ),
                }
                choice = apex_policy.choose(apex_snapshot, apex_choices(), extra=extra).action
                apex_counts[choice.value] = apex_counts.get(choice.value, 0) + 1
                last_action = choice.value
                trace.append(
                    {
                        "decision": decisions,  # noqa: B023 — called synchronously within this decision
                        "kind": "apex",
                        "x": int(event.x),
                        "choice": choice.value,
                        "frame": int(event.frame),
                        "vx": float(event.vx),
                    }
                )
                return choice

            execution = execute(
                env,
                decision.action,
                grounded=snapshot.grounded,
                frames_per_decision=frames_per_decision,
                ram=_unwrap_ram(env),
                on_frame=lambda f: frames.append(np.array(f, dtype=np.uint8)),  # copy buffer
                apex=on_apex if (apex_policy is not None and cadence == "grounded") else None,
            )
            terminated, truncated, info = execution.terminated, execution.truncated, execution.info
            total_reward = execution.total_reward
            reward_total += total_reward
            decisions += 1
            trace.append(
                {
                    "decision": decisions,
                    "kind": "takeoff",
                    "x": int(snapshot.x),
                    "grounded": bool(snapshot.grounded),
                    "macro": decision.action.value,
                    "status": decision.selection_status,
                    "frames": int(execution.frames),
                    "x_after": int(info.get("x_pos", 0)),
                }
            )
            snapshot = parser.parse(
                info,
                _unwrap_ram(env),
                previous_action=last_action,
                previous_reward=(
                    total_reward - execution.reward_to_apex
                    if execution.apex is not None
                    else total_reward
                ),
                frames=(
                    execution.frames - execution.frames_to_apex
                    if execution.apex is not None
                    else execution.frames
                ),
            )
            if snapshot.x > best_x:
                best_x = snapshot.x
                stalled = 0
            else:
                stalled += 1
            if terminated or truncated:
                break
    finally:
        env.close()

    gif_frames = write_gif(frames, out_path, fps=fps, decimate=decimate)
    return {
        "decisions": decisions,
        "frames": len(frames),
        "gif_frames": gif_frames,
        "max_x": int(info.get("x_pos_max", info.get("x_pos", 0))),
        "reached_flag": bool(info.get("flag_get", False)),
        "reward_total": round(reward_total, 1),
        "apex_choices": apex_counts,
        "trace": trace,
        "gif": str(out_path),
    }
