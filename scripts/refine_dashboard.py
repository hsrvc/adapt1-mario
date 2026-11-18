#!/usr/bin/env python
"""Render a logged frozen game of the base + correctors as an MP4 with the correctors' scores. Offline, no key.

The game comes from a `refine_frontier.py` run log (see `refine_replay.py`). The moves are replayed on the
emulator and checked against the log. Beside the game, the panel shows for each decision: the move that played,
whether it is the player's own move or a correction, Adapt-1's score for keeping and for the best replacements,
and how the move was picked (Adapt-1's own choice, no choice so the player's move, or no choice and the client
rule). The same for the mid-air choice at the top of a jump.

    python scripts/refine_dashboard.py --log runs/refine-2-1/refine12/leg2-frontier.jsonl.gz --tag final \\
        --out correctors-2-1.mp4

Needs ffmpeg on the PATH.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_replay import load_game, replay

from typesafe_mario.dashboard import LiveDashboard

LABEL = {
    "noop": "wait",
    "right": "walk right",
    "right_run": "run right",
    "left": "walk left",
    "jump": "plain jump",
    "right_run_jump_short": "running jump, short",
    "right_run_jump_mid": "running jump, medium",
    "right_run_jump_full": "running jump, full",
    "right_jump_full": "walking jump, full",
    "apex_keep": "keep going",
    "apex_brake": "brake",
    "apex_pull_back": "pull back",
}
KEEP = "keep_base"
LEVEL = {"SuperMarioBros-1-1-v0": "World 1-1", "SuperMarioBros-2-1-v0": "World 2-1"}


def how(status: str | None, edit: str) -> str:
    """One line on how the move was picked, from the step's status in the log."""
    if status == "selected":
        return "Adapt-1 chose to keep it" if edit == KEEP else "Adapt-1 chose the replacement"
    if status == "abstained":
        return "Adapt-1 gave no choice: the player's move plays"
    if status == "keep_dominance":
        return "Adapt-1 gave no choice: the client rule plays the replacement"
    return status or ""


def option_label(option: str, base: str) -> str:
    if option == KEEP:
        return f"keep the player's move ({LABEL.get(base, base)})"
    move = option.removeprefix("override_")
    return f"replace with {LABEL.get(move, move)}"


def scores(
    d: LiveDashboard, x: int, y: int, w: int, values: dict, base: str, edit: str, rows: int
) -> int:
    """Bars for "keep" and the best replacements; the option that played is highlighted. Returns the next y."""
    t = d.theme
    # a replacement that is the player's own move plays the same move as "keep": leave it out
    same = f"override_{base}"
    others = sorted((k for k in values if k not in (KEEP, same)), key=lambda k: -float(values[k]))
    shown = [KEEP, *others[: rows - 1]]
    if edit not in shown:
        shown[-1] = edit
    for name in shown:
        played = name == edit
        d._bar(
            label=option_label(name, base),
            value=float(values.get(name, 0.0)),
            x=x,
            y=y,
            width=w,
            selected=played,
            color=t.warning if played and name != KEEP else None,
        )
        y += 29
    return y


def draw(
    d: LiveDashboard, frame: Any, game: dict, n: int, look: bool, done: int, total: int
) -> bytes:
    t, pg, c = d.theme, d.pg, d.config
    st = game["steps"][n]
    d.screen.fill(t.canvas)
    margin, header_h = 24, 58
    panel_x = c.width - c.panel_width
    d._text("Adapt-1 plays Mario", d.font_title, t.text, margin, 18)
    status = "Frozen game · learning off"
    sw = d.font_small.size(status)[0]
    pg.draw.circle(d.screen, t.accent, (c.width - sw - 40, 31), 4)
    d._text(status, d.font_small, t.muted, c.width - sw - 28, 21)
    d._draw_game(
        frame, margin, header_h + 14, panel_x - margin * 2, c.height - header_h - 14 - margin
    )

    x, y, w = panel_x + 8, header_h + 22, c.panel_width - 40
    d._text(LEVEL.get(game["env"], game["env"]), d.font_label, t.muted, x, y)
    right = f"Move {n + 1} of {len(game['steps'])}"
    d._text(right, d.font_small, t.muted, x + w - d.font_small.size(right)[0], y + 1)
    y += 28
    d._text(LABEL.get(st["executed"], st["executed"]), d.font_action, t.text, x, y)
    y += 46
    corrected = st["edit"] != KEEP
    d._text(
        f"correction: the player chose {LABEL.get(st['base'], st['base'])}"
        if corrected
        else "the player's own move",
        d.font_small,
        t.warning if corrected else t.muted,
        x,
        y,
    )
    y += 28
    d._rule(x, y, w)
    y += 14

    d._text("Corrector on the ground: Adapt-1's scores", d.font_label, t.text, x, y)
    y += 24
    y = scores(d, x, y, w, st.get("values") or {}, st["base"], st["edit"], rows=5)
    d._text(
        how(st["status"], st["edit"]), d.font_small, t.warning if corrected else t.muted, x, y + 2
    )
    y += 28
    d._rule(x, y, w)
    y += 14

    d._text("Corrector at the top of the jump", d.font_label, t.text, x, y)
    y += 24
    if st.get("apex_executed") and look:
        air_corrected = st["apex_edit"] != KEEP
        y = scores(
            d, x, y, w, st.get("apex_values") or {}, st["apex_base"], st["apex_edit"], rows=4
        )
        d._text(
            how(st["apex_status"], st["apex_edit"]),
            d.font_small,
            t.warning if air_corrected else t.muted,
            x,
            y + 2,
        )
    else:
        d._text(
            "waiting for the top of the jump"
            if st.get("apex_executed")
            else "no mid-air choice on this move",
            d.font_small,
            t.muted,
            x,
            y,
        )
    y = c.height - margin - 44
    d._rule(x, y - 12, w)
    d._text(f"Position  x {st['x']}", d.font_mono, t.muted, x, y)
    d._text(f"Corrections so far: {done} of {total}", d.font_mono, t.muted, x, y + 22)
    pg.display.flip()
    return pg.image.tobytes(d.screen, "RGB")


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
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to hold the last frame")
    ap.add_argument(
        "--pause", type=float, default=1.5, help="seconds to hold the picture at each correction"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    game = load_game(args.log, args.tag, args.env)
    frames: list[Any] = []
    marks: list[tuple[int, int | None]] = []
    got = replay(game, seed=args.seed, frames=frames, marks=marks)
    steps = game["steps"]

    def count(st: dict) -> int:
        return int(st["edit"] != KEEP) + int(st.get("apex_edit") not in (None, KEEP))

    total = sum(count(st) for st in steps)
    dash = LiveDashboard(brain="Adapt-1")
    out, f, done = [], 0, 0
    pause = int(args.pause * args.fps)
    for n, (st, (length, apex_frame)) in enumerate(zip(steps, marks, strict=True)):
        ground = st["edit"] != KEEP
        air = st.get("apex_edit") not in (None, KEEP)
        done += int(ground)
        for k in range(length):
            look = apex_frame is not None and k >= apex_frame
            at_look = look and k == apex_frame
            done += int(at_look and air)
            shot = draw(dash, frames[f], game, n, look, done, total)
            out.append(shot)
            # hold the picture at the moment a correction shows, so it can be read
            if (k == 0 and ground) or (at_look and air):
                out += [shot] * pause
            f += 1
    out += out[-1:] * int(args.hold * args.fps)
    w, h = dash.screen.get_size()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(args.fps),
         "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest", "-vf", "format=yuv420p",
         "-c:v", "libx264", "-profile:v", "high", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "64k", str(args.out)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for fr in out:
        enc.stdin.write(fr)
    enc.stdin.close()
    enc.wait()
    print(
        f"wrote {args.out}: {len(out) / args.fps:.1f} s, {len(steps)} decisions, {total} corrections, "
        f"max x {got['max_x']}, flag {got['flag']}"
    )
    return 0 if enc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
