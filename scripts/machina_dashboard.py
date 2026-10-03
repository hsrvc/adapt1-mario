#!/usr/bin/env python
"""Render a Machina frozen run (1-1, or 2-1 with --env) as a dashboard MP4: the command playing, its four channels, the whole sequence.

Offline, 0 Records, 0 Queries: the frozen journal recorded the exact proposal Machina returned (244 commands of
4 channels, `retained_episode_policy`). The emulator is deterministic, so replaying that proposal from
`reset(seed=777)` reproduces the run frame for frame; each command is 8 frames, so every frame maps to one command.

    python scripts/machina_dashboard.py --journal runs/machina/logs/<frozen 1-1 journal>.jsonl.gz --out machina.mp4
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario.dashboard import LiveDashboard  # noqa: E402
from typesafe_mario.machina import (  # noqa: E402
    FRAMES_PER_COMMAND,
    _frames,
    execute_sequence,
)
from typesafe_mario.runner import _unwrap_ram, create_mario_env  # noqa: E402
from typesafe_mario.state import MarioStateParser  # noqa: E402

CHANNELS = ("right", "left", "run", "jump")


def load(journal: Path) -> tuple[list[list[float]], dict]:
    rows, meta = None, {}
    opener = gzip.open if journal.suffix == ".gz" else open
    with opener(journal, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "call" and r["record"].get("op") == "propose" and rows is None:
                rows = r["record"]["resp"]["actions"]
                meta = {
                    k: r["record"]["resp"].get(k)
                    for k in ("source", "version", "reference_record_id")
                }
    if rows is None:
        raise SystemExit(f"no propose call in {journal}")
    return rows, meta


def held(row: list[float]) -> list[int]:
    """Frames (0..8) each channel is held in this command, as the decoder plays it."""
    return [_frames(v) for v in row]


def buttons(h: list[int]) -> str:
    right, left, run, jump = h
    parts = []
    if left and not right:
        parts.append("left")
    if right:
        parts.append("right")
    if run:
        parts.append("run")
    if jump:
        parts.append("jump")
    return " + ".join(parts) or "no buttons"


def draw(d: LiveDashboard, frame, rows, i: int, x_now: int, args) -> bytes:
    t, pg, c = d.theme, d.pg, d.config
    d.screen.fill(t.canvas)
    margin, header_h = 24, 58
    panel_x = c.width - c.panel_width
    d._text("Adapt-1 Machina plays Mario", d.font_title, t.text, margin, 18)
    status = "Frozen replay of its best attempt · learning off"
    sw = d.font_small.size(status)[0]
    pg.draw.circle(d.screen, t.accent, (c.width - sw - 40, 31), 4)
    d._text(status, d.font_small, t.muted, c.width - sw - 28, 21)
    d._draw_game(
        frame, margin, header_h + 14, panel_x - margin * 2, c.height - header_h - 14 - margin
    )

    x, y, w = panel_x + 8, header_h + 22, c.panel_width - 40
    h = held(rows[i])
    d._text(args.world, d.font_label, t.muted, x, y)
    right = f"Command {i + 1} of {len(rows)}"
    d._text(right, d.font_small, t.muted, x + w - d.font_small.size(right)[0], y + 1)
    y += 28
    d._text(buttons(h), d.font_action, t.text, x, y)
    y += 46
    d._text(
        "Each command lasts 8 frames; a channel is held for part of it", d.font_small, t.muted, x, y
    )
    y += 30
    d._rule(x, y, w)
    y += 16
    d._text("This command", d.font_label, t.text, x, y)
    y += 26
    for name, n in zip(CHANNELS, h):
        d._text(name, d.font_small, t.text if n else t.muted, x, y)
        note = f"{n} of 8 frames"
        d._text(note, d.font_small, t.muted, x + w - d.font_small.size(note)[0], y)
        pg.draw.rect(d.screen, t.raised, (x, y + 19, w, 5), border_radius=2)
        if n:
            pg.draw.rect(
                d.screen, t.accent, (x, y + 19, int(w * n / FRAMES_PER_COMMAND), 5), border_radius=2
            )
        y += 30
    y += 6
    d._rule(x, y, w)
    y += 16

    d._text("The whole sequence Machina found", d.font_label, t.text, x, y)
    y += 24
    # one row per channel, one column per command; brightness = frames held; playhead on the current command
    row_h, gap = 14, 4
    col_w = w / len(rows)
    for ci, name in enumerate(CHANNELS):
        yy = y + ci * (row_h + gap)
        d._text(name, d.font_small, t.muted, x, yy - 2)
        x0 = x + 44
        cw = (w - 44) / len(rows)
        for k, r in enumerate(rows):
            n = held(r)[ci]
            if n:
                a = n / FRAMES_PER_COMMAND
                col = tuple(int(t.surface[j] + (t.accent[j] - t.surface[j]) * a) for j in range(3))
                pg.draw.rect(d.screen, col, (int(x0 + k * cw), yy, max(1, int(cw + 0.5)), row_h))
    x0, cw = x + 44, (w - 44) / len(rows)
    px = int(x0 + (i + 0.5) * cw)
    pg.draw.line(d.screen, t.warning, (px, y - 4), (px, y + 4 * (row_h + gap)), 2)
    y += 4 * (row_h + gap) + 14
    d._rule(x, y, w)
    y += 16
    d._text(
        f"First flag at attempt {args.first_flag} of {args.attempts},", d.font_mono, t.muted, x, y
    )
    y += 22
    d._text(
        args.origin or f"{args.minutes} wall-clock minutes from untrained",
        d.font_mono,
        t.muted,
        x,
        y,
    )
    y += 22
    d._text(f"Position  x {x_now}", d.font_mono, t.muted, x, y)
    pg.display.flip()
    return pg.image.tobytes(d.screen, "RGB")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--journal", type=Path, required=True, help="the frozen 1-1 journal (jsonl or jsonl.gz)"
    )
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--first-flag", type=int, default=403)
    ap.add_argument("--attempts", type=int, default=467)
    ap.add_argument("--minutes", type=int, default=11)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--hold", type=float, default=2.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--env",
        default="SuperMarioBros-1-1-v0",
        help="the level the journal's run played (mach2: 2-1)",
    )
    ap.add_argument(
        "--level-length",
        type=float,
        default=None,
        help="px of the flag, the run's goal scale (default: the level's, 1-1 3161 / 2-1 3193)",
    )
    ap.add_argument(
        "--origin",
        default=None,
        help="the second stats line (default '<minutes> wall-clock minutes from untrained'); mach2: how it started",
    )
    args = ap.parse_args()
    from typesafe_mario import machina as _machina

    _machina.set_level_length(args.level_length or _machina.LEVEL_LENGTHS.get(args.env, 3161.0))
    args.world = "World " + args.env.removeprefix("SuperMarioBros-").removesuffix("-v0")

    rows, meta = load(args.journal)
    env = create_mario_env(args.env, render_mode="rgb_array")
    parser = MarioStateParser(decision_horizon_frames=8)
    _f, info = env.reset(seed=args.seed)
    snap = parser.parse(info, _unwrap_ram(env), previous_action=None)
    frames: list = []
    trace = execute_sequence(env, parser, snap, rows, on_frame=lambda f: frames.append(f.copy()))
    env.close()
    print(
        f"proposal: {len(rows)} commands, source={meta.get('source')}; replay: max_x={trace['max_x']} "
        f"flag={trace['flag']} executed={trace['executed']} frames={len(frames)}",
        flush=True,
    )

    dash = LiveDashboard(brain="Adapt-1")
    out = []
    for k, fr in enumerate(frames):
        i = min(k // FRAMES_PER_COMMAND, trace["executed"] - 1)
        x_now = int(round(trace["states"][i][0] * _machina.LEVEL_LENGTH_PX))
        out.append(draw(dash, fr, rows[: trace["executed"]], i, x_now, args))
    w, h = dash.screen.get_size()
    out += out[-1:] * int(args.hold * args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-shortest",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(args.out),
        ],
        stdin=subprocess.PIPE,
    )
    for fr in out:
        enc.stdin.write(fr)
    enc.stdin.close()
    enc.wait()
    print(f"{len(out)} frames -> {args.out}")
    return 0 if enc.returncode == 0 and trace["flag"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
