#!/usr/bin/env python
"""Replay a logged frozen game of the base + correctors on the emulator. Offline, no key, 0 credits.

The run log of `refine_frontier.py` holds every frozen game with the move executed at each decision (and the
mid-air choice, when a jump had one). This script plays those moves from the level start and checks, decision by
decision, that Mario stands where the log says. It prints how the game ended and can record a GIF.

    python scripts/refine_replay.py --log runs/refine-2-1/refine12/leg2-frontier.jsonl.gz --tag final --out flag.gif

Exit status 1 if the replay leaves the logged game or ends differently.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario.cadence import CadenceParser
from typesafe_mario.frontier import ReplayMismatch, _replay_hook
from typesafe_mario.macros import Macro, execute
from typesafe_mario.record import write_gif
from typesafe_mario.runner import _unwrap_ram, create_mario_env


def load_game(log: Path, tag: str, env_id: str | None) -> dict[str, Any]:
    """The last game of kind `frozen` with this tag (and level, if given)."""
    opener = gzip.open if log.suffix == ".gz" else open
    found = None
    with opener(log, "rt") as fh:
        for line in fh:
            e = json.loads(line)
            if e.get("kind") != "frozen" or e.get("tag") != tag:
                continue
            if env_id and e.get("env") != env_id:
                continue
            found = e
    if found is None:
        raise SystemExit(f"no frozen game tagged {tag!r} in {log}")
    return found


def replay(game: dict[str, Any], *, seed: int, frames: list[Any] | None) -> dict[str, Any]:
    """Play the logged moves. Raises ReplayMismatch at the first decision that differs from the log."""
    env = create_mario_env(game["env"], render_mode="rgb_array")
    parser = CadenceParser(decision_horizon_frames=8)
    on_frame = (
        (lambda f: frames.append(np.array(f, dtype=np.uint8))) if frames is not None else None
    )
    try:
        _f, info = env.reset(seed=seed)
        snap = parser.parse(info, _unwrap_ram(env), previous_action=None)
        best = snap.x
        for st in game["steps"]:
            i = st["i"]
            if int(snap.x) != int(st["x"]):
                raise ReplayMismatch(f"decision {i}: x {snap.x} != logged {st['x']}")
            macro = Macro(st["executed"])
            sink: dict[str, Any] = {}
            ex = execute(
                env,
                macro,
                grounded=snap.grounded,
                ram=_unwrap_ram(env),
                on_frame=on_frame,
                apex=_replay_hook(parser, env, macro, st.get("apex_executed"), sink),
            )
            fired = ex.apex is not None and bool(sink)
            if bool(st.get("apex_executed")) != fired:
                raise ReplayMismatch(
                    f"decision {i}: mid-air look {fired}, logged {st.get('apex_executed')}"
                )
            snap = parser.parse(
                ex.info,
                _unwrap_ram(env),
                previous_action=(st["apex_executed"] if fired else macro.value),
                previous_reward=(ex.total_reward - ex.reward_to_apex if fired else ex.total_reward),
                frames=(ex.frames - ex.frames_to_apex if fired else ex.frames),
            )
            if int(snap.x) != int(st["x_after"]):
                raise ReplayMismatch(f"decision {i}: x after {snap.x} != logged {st['x_after']}")
            best = max(best, snap.x)
        return {"decisions": len(game["steps"]), "max_x": int(best), "flag": bool(snap.clear)}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--log", type=Path, required=True, help="a refine_frontier.py run log (.jsonl or .jsonl.gz)"
    )
    ap.add_argument(
        "--tag", default="final", help="the frozen game's tag in the log (default: final)"
    )
    ap.add_argument("--env", help="pick the game on this level when a tag has one per level")
    ap.add_argument(
        "--seed", type=int, default=777, help="the emulator seed of the logged game (default 777)"
    )
    ap.add_argument("--out", type=Path, help="write a GIF of the replay here")
    ap.add_argument("--decimate", type=int, default=3)
    args = ap.parse_args()

    game = load_game(args.log, args.tag, args.env)
    frames: list[Any] | None = [] if args.out else None
    try:
        got = replay(game, seed=args.seed, frames=frames)
    except ReplayMismatch as exc:
        print(f"MISMATCH {exc}")
        return 1
    steps = game["steps"]
    ground = sum(1 for s in steps if s.get("edit") not in (None, "keep_base"))
    air = sum(1 for s in steps if s.get("apex_edit") not in (None, "keep_base"))
    print(
        f"{game['env']} tag {args.tag}: {got['decisions']} decisions replayed, every position matches the log; "
        f"max x {got['max_x']}, flag {got['flag']}; corrections {ground} on the ground + {air} in the air"
    )
    if args.out and frames:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_gif(frames, args.out, decimate=args.decimate)
        print(f"wrote {args.out}")
    if got["flag"] != bool(game.get("flag")) or got["max_x"] != int(
        game.get("max_x", got["max_x"])
    ):
        print(f"MISMATCH end: logged flag {game.get('flag')} max x {game.get('max_x')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
