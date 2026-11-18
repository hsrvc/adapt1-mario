#!/usr/bin/env python
"""Scan the published frozen games for wall jumps. Offline, no key.

A wall jump in Super Mario Bros.: Mario hits a wall in mid-air with his feet at a block seam, the game counts him
as standing for a few frames, and a jump pressed then is accepted. Our loops ask for the next move as soon as the
game reports "standing", so they can get one by accident.

How the scan finds one. Every emulator frame of a replay is watched (the emulator's own step is wrapped, so each
replay runs through its normal code). A candidate is a frame where Mario goes from the air to "standing" and his
sideways speed drops from 16 or more to 2 or less on that same frame (he hit a wall as he "landed"). Each candidate
is then tested on the spot: the emulator state is saved, 5 frames run with no buttons, and the state is restored.
On a real floor Mario is still standing after those frames. If he is falling, there was no floor: a wall catch.
The speed rule is a filter (a wall jump needs speed above 16 toward the wall), not an exhaustive test.

    python scripts/wall_jump_scan.py            # from the repo's agent folder (private layout: ../runs)
    python scripts/wall_jump_scan.py --runs runs  # in the public tree
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nes_py.nes_env import NESEnv

STATE, FACING, XSPEED, YSPEED, XPAGE, XPOS, YSCREEN = 0x1D, 0x33, 0x57, 0x9F, 0x6D, 0x86, 0xCE
FRAMES: list[dict[str, Any]] = []
_original_step = NESEnv.step
_probing = False


def _signed(v: int) -> int:
    return v - 256 if v > 127 else v


def _read(env: NESEnv) -> dict[str, Any]:
    ram = env.ram
    return {
        "state": int(ram[STATE]),
        "xspeed": _signed(int(ram[XSPEED])),
        "yspeed": _signed(int(ram[YSPEED])),
        "x": int(ram[XPAGE]) * 256 + int(ram[XPOS]),
        "y_screen": int(ram[YSCREEN]),
    }


def _floor_test(env: NESEnv) -> list[int]:
    """Save, run 5 frames with no buttons, restore. Returns Mario's state on each of the 5 frames."""
    global _probing
    keep = {k: v for k, v in env.__dict__.items() if isinstance(v, (int, float, bool)) or v is None}
    env._backup()
    _probing = True
    states = []
    try:
        for _ in range(5):
            if env.done:
                break
            _original_step(env, 0)
            states.append(int(env.ram[STATE]))
    finally:
        _probing = False
        env._restore()
        env.__dict__.update(keep)
    return states


def _watched_step(self: NESEnv, action: int):
    out = _original_step(self, action)
    if _probing:
        return out
    now = _read(self)
    now["n"] = len(FRAMES)
    if FRAMES:
        prev = FRAMES[-1]
        hit = (
            prev["state"] != 0
            and now["state"] == 0
            and abs(prev["xspeed"]) >= 16
            and abs(now["xspeed"]) <= 2
        )
        if hit:
            now["candidate"] = True
            now["speed_before"] = prev["xspeed"]
            now["no_button_states"] = _floor_test(self)
    FRAMES.append(now)
    return out


def _report(label: str) -> dict[str, Any]:
    frames = list(FRAMES)
    FRAMES.clear()
    found = []
    for f in frames:
        if not f.get("candidate"):
            continue
        after = f["no_button_states"]
        wall = any(s != 0 for s in after)
        nxt = frames[f["n"] + 1 : f["n"] + 4]
        jumped = any(g["state"] == 1 and g["yspeed"] < 0 for g in nxt)
        found.append(
            {
                "frame": f["n"],
                "x": f["x"],
                "y_screen": f["y_screen"],
                "speed_before": f["speed_before"],
                "states_with_no_buttons": after,
                "floor_under_him": not wall,
                "jump_followed": jumped,
                "wall_jump": wall and jumped,
            }
        )
    jumps = [c for c in found if c["wall_jump"]]
    end_x = max((f["x"] for f in frames), default=0)
    print(
        f"{label}: {len(frames)} frames, furthest x {end_x}, wall-stop landings {len(found)}, "
        f"wall jumps {len(jumps)}"
        + "".join(f"\n    wall jump at x {c['x']}, frame {c['frame']}" for c in jumps)
    )
    return {"game": label, "frames": len(frames), "furthest_x": end_x, "wall_stop_landings": found}


def refine_game(log: Path, tag: str) -> None:
    from refine_replay import load_game, replay

    replay(load_game(log, tag, None), seed=777, frames=None)


def duo_evidence(path: Path, env_id: str = "SuperMarioBros-1-1-v0") -> None:
    """A frozen takeoff + apex game recorded by dashboard_live.py: replay its decisions."""
    import dashboard_live as dl

    from typesafe_mario import record as rec
    from typesafe_mario.macros import Macro

    e = json.loads(path.read_text())
    keep = rec.write_gif
    rec.write_gif = lambda frames, out, **kw: 0
    try:
        rec.record_episode(
            env_id,
            dl.Recorded(e["decisions"], "takeoff"),
            Path(tempfile.mkdtemp()) / "x.gif",
            max_decisions=400,
            seed=e["seed"],
            stall_timeout=60,
            cadence="grounded",
            apex_policy=dl.Recorded(e["apex"], "apex"),
            actions=tuple(Macro(m) for m in dl.MACROS.split(",")),
        )
    finally:
        rec.write_gif = keep


def machina_journal(path: Path, env_id: str, index: int = 0) -> None:
    from typesafe_mario.machina import execute_sequence
    from typesafe_mario.runner import _unwrap_ram, create_mario_env
    from typesafe_mario.state import MarioStateParser

    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    # a frozen journal holds the proposal Machina returned (a `propose` call); an acquire journal holds executions
    actions = next(
        (
            r["record"]["resp"]["actions"]
            for r in rows
            if r["kind"] == "call" and r["record"].get("op") == "propose"
        ),
        None,
    )
    if actions is None:
        actions = [r["record"] for r in rows if r["kind"] == "execution"][index]["actions"]
    env = create_mario_env(env_id, render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    _f, info = env.reset(seed=777)
    snap = parser.parse(info, _unwrap_ram(env), previous_action=None)
    try:
        execute_sequence(env, parser, snap, actions)
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--runs", type=Path, default=None, help="the runs folder (default: found from the layout)"
    )
    ap.add_argument("--out", type=Path, default=None, help="write the full result as JSON")
    args = ap.parse_args()
    here = Path(__file__).resolve().parents[1]
    public = args.runs or here / "runs"
    private = here.parent / "runs"
    if args.runs is None and not (public / "refine-2-1").exists():
        public = None
    names = (
        {
            "refine": public / "refine-2-1" / "refine12" / "leg2-frontier.jsonl.gz",
            "zero": public / "dashboard" / "zero-start-2026-10-02.json",
            "warm": public / "dashboard" / "warm-start-2026-10-02.json",
            "machina": public / "machina" / "logs",
            "machina21": public / "machina-2-1" / "logs",
        }
        if public
        else {
            "refine": private / "2026-11-17-base2-refine12" / "refine12" / "leg2-frontier.jsonl.gz",
            "zero": private
            / "2026-10-02-dashboard-live"
            / "mario-zero2-takeoff-20261002T145652Z.json",
            "warm": private
            / "2026-10-02-dashboard-live"
            / "mario-gate4-takeoff-20261002T150432Z.json",
            "machina": private / "2026-10-02-machina-z7r-replication" / "logs",
            "machina21": private / "2026-10-03-mach2-seeded-clear" / "logs",
        }
    )
    m11 = next(names["machina"].glob("*frozen-SuperMarioBros-1-1*"))
    m21 = next(names["machina21"].glob("*frozen-SuperMarioBros-2-1*"))

    NESEnv.step = _watched_step
    results = []
    try:
        for label, play in (
            (
                "correctors, frozen 2-1 (the flag game)",
                lambda: refine_game(names["refine"], "final"),
            ),
            ("correctors, frozen 1-1", lambda: refine_game(names["refine"], "r01-1-1")),
            ("zero start, frozen 1-1", lambda: duo_evidence(names["zero"])),
            ("warm start, frozen 1-1", lambda: duo_evidence(names["warm"])),
            ("Machina, frozen 1-1", lambda: machina_journal(m11, "SuperMarioBros-1-1-v0")),
            ("Machina on 2-1, frozen 2-1", lambda: machina_journal(m21, "SuperMarioBros-2-1-v0")),
        ):
            FRAMES.clear()
            play()
            results.append(_report(label))
    finally:
        NESEnv.step = _original_step
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
