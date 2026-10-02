"""Client-side k-step returns (findings #22) — the delayed credit computed on the client.

    R_t = ( sum_{i<k} gamma^i * r_{t+i} ) / ( sum_{i<k} gamma^i )      within the episode

A decision that ends the episode early (a death) is worth only the progress it earned before
dying; a decision followed by progress is worth that progress. Normalised by the geometric sum
so R_t stays in [0,1] when r is (PROGRESS_FRACTION rewards are). Feed the result to a
DIRECT-feedback domain (no ``sequential`` block). This is doing the sequential learner's job on
the client — say so in any write-up.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence


def kstep_return(rewards: Sequence[float], t: int, k: int, gamma: float) -> float:
    """The normalised k-step return of step ``t`` over an episode's reward sequence."""
    norm = sum(gamma**i for i in range(k))
    ret = sum(gamma**i * float(rewards[t + i]) for i in range(k) if t + i < len(rewards))
    return round(max(0.0, min(1.0, ret / norm)), 6)


def kstep_returns(rows: list[dict], k: int, gamma: float) -> list[dict]:
    """Rewrite each row's ``reward`` as its k-step return (``immediate_reward`` keeps the old)."""
    by_ep: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)
    out = []
    for ep in by_ep.values():
        ep.sort(key=lambda r: r["step"])
        rewards = [float(r["reward"]) for r in ep]
        for t, r in enumerate(ep):
            new = dict(r)
            new["immediate_reward"] = r["reward"]
            new["reward"] = kstep_return(rewards, t, k, gamma)
            out.append(new)
    return out


# --- fixed frame-horizon return (2026-09-23, findings #29 follow-up) -----------------------------
#
# A k-DECISION return pays a long macro more than a short one at the same state, because a jump
# lasts 3-8x a run step and so earns 3-8x the progress inside its own row (#29: the frozen policy
# never runs). The first per-frame fix (progress / frames lived, per 8 frames) has the opposite
# hole: a branch that DIES early is scored over the few frames it lived, so at pipe 4 the fatal
# full jump (0.0066) out-scores the surviving brake (0.0062). Both errors come from letting the
# action choose its own denominator.
#
# The fixed-horizon return closes both: every decision is worth the progress made in the next
# ``horizon`` emulator frames, whatever it does with them. A death (or a stall) forfeits the
# frames it did not live; a clear is credited as if full speed continued (``clear_rate``), so
# the last decisions before a flag are not scored like a death. Units: level fraction per 8
# frames, capped at 1 (full run speed, 3 px/frame, reads ~0.0076 on 1-1's 3161 px).
FULL_SPEED_RATE = 3.0 * 8.0 / 3161.0  # per-8-frame progress fraction at full run speed


def frame_horizon_return(
    rewards: Sequence[float],
    frames: Sequence[int],
    t: int,
    horizon: int,
    *,
    cleared: bool = False,
    clear_rate: float = FULL_SPEED_RATE,
) -> float:
    """Progress (level fraction) per 8 frames over the next ``horizon`` frames from decision ``t``.

    ``rewards[i]`` is the progress fraction earned by decision ``i`` over ``frames[i]`` frames. The
    row that crosses the horizon is prorated. When the sequence ends before the horizon (the
    episode ended, or the record did), the remaining frames earn nothing unless ``cleared`` — the
    flag — in which case they earn ``clear_rate`` (as if the level went on at full speed)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    budget = int(horizon)
    total = 0.0
    for i in range(t, len(rewards)):
        f = max(1, int(frames[i]))
        if f <= budget:
            total += float(rewards[i])
            budget -= f
        else:
            total += float(rewards[i]) * budget / f
            budget = 0
        if budget <= 0:
            break
    if budget > 0 and cleared:
        total += clear_rate * budget / 8.0
    return round(max(0.0, min(1.0, total * 8.0 / horizon)), 6)


def frame_horizon_returns(
    rows: list[dict], horizon: int, *, field: str = "reward_rate"
) -> list[dict]:
    """Write each row's fixed-horizon return under ``field`` (``reward`` and ``immediate_reward``
    untouched). Rows need ``frames``; an episode whose last row's outcome is ``cleared`` is credited
    at full speed past its end."""
    by_ep: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)
    out = []
    for ep in by_ep.values():
        ep.sort(key=lambda r: r["step"])
        base = [float(r.get("immediate_reward", r["reward"])) for r in ep]
        frames = [int(r["frames"]) for r in ep]
        cleared = ep[-1].get("outcome") == "cleared"
        for t, r in enumerate(ep):
            new = dict(r)
            new[field] = frame_horizon_return(base, frames, t, horizon, cleared=cleared)
            out.append(new)
    return out
