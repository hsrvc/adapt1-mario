#!/usr/bin/env python
"""Replay a journaled Machina proposal on the emulator and record a GIF (+ last-frame PNG). Offline, free.

    PYTHONPATH=src python scripts/machina_replay.py --journal artifacts/machina/<run>.jsonl --index 5 \
        --out ../clips/machina/a6-first-flag.gif [--env SuperMarioBros-2-1-v0] [--use-executed]
--use-executed replays the rows that actually ran (needed for a seeded attempt, whose proposal was ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from typesafe_mario.machina import execute_sequence, state_row  # noqa: E402
from typesafe_mario.record import write_gif  # noqa: E402
from typesafe_mario.runner import _unwrap_ram, create_mario_env  # noqa: E402
from typesafe_mario.state import MarioStateParser  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0, help="0-based attempt index within the journal")
    ap.add_argument("--use-executed", action="store_true")
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decimate", type=int, default=3)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.journal.read_text().splitlines() if l.strip()]
    props = [r["record"] for r in rows if r["kind"] == "proposal"]
    execs = [r["record"] for r in rows if r["kind"] == "execution"]
    actions = execs[args.index]["actions"] if args.use_executed else props[args.index]["actions"]
    env = create_mario_env(args.env, render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    _f, info = env.reset(seed=777)
    snap = parser.parse(info, _unwrap_ram(env), previous_action=None)
    frames: list[np.ndarray] = []
    trace = execute_sequence(env, parser, snap, actions, on_frame=lambda f: frames.append(np.array(f, dtype=np.uint8)))
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = write_gif(frames, args.out, fps=30, decimate=args.decimate)
    from PIL import Image
    Image.fromarray(frames[-1]).save(args.out.with_suffix(".last.png"))
    print(f"{args.out}: {len(frames)} frames -> {n} gif frames; executed {trace['executed']}/{len(actions)} "
          f"max_x={trace['max_x']} {'FLAG' if trace['flag'] else 'dead' if trace['dead'] else 'stall/horizon'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
