#!/usr/bin/env python
"""Play one frozen evaluation live and render it as an MP4 with Adapt-1's score for every move.

LIVE: Queries only (one per decision and one per apex look, about 75 for 1-1), 0 Records: exploration off,
no feedback. Uses frozen_eval.py's setup (v3 features on both learners, the 8 declared macros) and
record.py's episode loop unchanged; wrappers keep each decision Adapt-1 returned (with its per-move
expected values from the server response), then every frame is drawn with the decision active at that frame.

    REI_KEY=... python scripts/dashboard_live.py --domain-id mario-zero2-takeoff --apex-domain mario-zero2-apex \\
        --label "zero start" --out zero-start-live.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario import record as rec  # noqa: E402
from typesafe_mario.adapt1_client import Adapt1Client  # noqa: E402
from typesafe_mario.adapt1_policy import FEATURE_SETS, Adapt1Policy  # noqa: E402
from typesafe_mario.dashboard import LiveDashboard  # noqa: E402
from typesafe_mario.macros import Macro  # noqa: E402

MACROS = "noop,right,right_run,left,right_run_jump_short,right_run_jump_mid,right_run_jump_full,right_jump_full"
LABEL = {
    "noop": "wait",
    "right": "walk right",
    "right_run": "run right",
    "left": "walk left",
    "right_run_jump_short": "running jump, short",
    "right_run_jump_mid": "running jump, medium",
    "right_run_jump_full": "running jump, full",
    "right_jump_full": "standing jump, full",
    "apex_keep": "keep going",
    "apex_brake": "brake",
    "apex_pull_back": "pull back",
}


DESCRIBE = {
    "noop": "Stand still for 8 frames",
    "right": "Walk right for 8 frames",
    "right_run": "Run right for 8 frames",
    "left": "Walk left for 8 frames",
    "right_run_jump_short": "Run and hold jump 8 frames, steer until landing",
    "right_run_jump_mid": "Run and hold jump 16 frames, steer until landing",
    "right_run_jump_full": "Run and hold jump 32 frames, steer until landing",
    "right_jump_full": "Walk and hold jump 32 frames, steer until landing",
}


class Recorded:
    """Stand-in policy for --from-evidence: returns the decisions a live run recorded, in order (no API)."""

    def __init__(self, rows, kind):
        from typesafe_mario.macros import ApexChoice
        from typesafe_mario.policy import Decision

        self.rows, self.i, self.kind, self.Decision, self.ApexChoice = (
            rows,
            0,
            kind,
            Decision,
            ApexChoice,
        )
        self.status_counts = {}

    def choose(self, snapshot, actions, extra=None):
        r = self.rows[self.i]
        self.i += 1
        name = r["action"] if self.kind == "takeoff" else r["choice"]
        by_name = {a.value: a for a in actions}
        self.status_counts[r["status"]] = self.status_counts.get(r["status"], 0) + 1
        return self.Decision(
            action=by_name[name],
            confidence=0.0,
            probabilities={},
            latency_ms=0.0,
            selection_status=r["status"],
            telemetry=r["telemetry"],
        )


class Recorder:
    """Wrap a policy: pass every call through and keep what it returned."""

    def __init__(self, policy, log):
        self.policy, self.log = policy, log

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def choose(self, snapshot, actions, extra=None):
        d = (
            self.policy.choose(snapshot, actions, extra=extra)
            if extra
            else self.policy.choose(snapshot, actions)
        )
        self.log.append({"snapshot": snapshot, "decision": d})
        return d


def ahead(snapshot) -> str:
    nav = snapshot.navigation_features()
    parts = []
    enemy = getattr(snapshot, "enemies", None)
    if enemy:
        parts.append(", ".join(e.kind.replace("_", " ") for e in enemy[:2]))
    for label, key in (
        ("pit", "gap_distance_tiles"),
        ("drop", "drop_distance_tiles"),
        ("wall", "obstacle_distance_tiles"),
    ):
        v = nav.get(key)
        if v is not None:
            parts.append(f"{label} in {v} tile{'s' if v != 1 else ''}")
    return " · ".join(parts) or "clear ground"


def draw(
    d: LiveDashboard, frame, n: int, total: int, step: dict, apex: dict | None, chosen_so_far: int
):
    t, pg = d.theme, d.pg
    c = d.config
    d.screen.fill(t.canvas)
    margin, header_h = 24, 58
    panel_x = c.width - c.panel_width
    d._text("Adapt-1 plays Mario", d.font_title, t.text, margin, 18)
    status = "Frozen evaluation · learning off"
    sw = d.font_small.size(status)[0]
    pg.draw.circle(d.screen, t.accent, (c.width - sw - 40, 31), 4)
    d._text(status, d.font_small, t.muted, c.width - sw - 28, 21)
    d._draw_game(
        frame, margin, header_h + 14, panel_x - margin * 2, c.height - header_h - 14 - margin
    )

    x, y, w = panel_x + 8, header_h + 22, c.panel_width - 40
    dec = step["decision"]
    d._text("World 1-1", d.font_label, t.muted, x, y)
    right = f"Move {n} of {total}"
    d._text(right, d.font_small, t.muted, x + w - d.font_small.size(right)[0], y + 1)
    y += 28
    d._text(LABEL.get(dec.action.value, dec.action.value), d.font_action, t.text, x, y)
    y += 46
    d._text(DESCRIBE.get(dec.action.value, ""), d.font_small, t.muted, x, y)
    y += 30
    d._rule(x, y, w)
    y += 16

    values = (dec.telemetry or {}).get("values") or {}
    d._text("Adapt-1's score for each move", d.font_label, t.text, x, y)
    y += 20
    d._text("expected progress, best move = 100%", d.font_small, t.muted, x, y)
    y += 26
    top = max(values.values()) if values else 0.0
    for name in MACROS.split(","):
        v = values.get(name, 0.0)
        share = v / top if top > 0 else 0.0
        d._bar(label=LABEL[name], value=share, x=x, y=y, width=w, selected=name == dec.action.value)
        y += 30
    y += 6
    d._rule(x, y, w)
    y += 16

    d._text("At the top of the jump", d.font_label, t.text, x, y)
    y += 26
    if apex:
        av = (apex["decision"].telemetry or {}).get("values") or {}
        atop = max(av.values()) if av else 0.0
        for name in ("apex_keep", "apex_brake", "apex_pull_back"):
            v = av.get(name, 0.0)
            share = v / atop if atop > 0 else 0.0
            d._bar(
                label=LABEL[name],
                value=share,
                x=x,
                y=y,
                width=w,
                selected=name == apex["decision"].action.value,
            )
            y += 30
    else:
        d._text(
            "no mid-air choice on this move"
            if not dec.action.value.startswith(("right_run_jump", "right_jump"))
            else "waiting for the top of the jump",
            d.font_small,
            t.muted,
            x,
            y,
        )
        y += 84
    y += 6
    d._rule(x, y, w)
    y += 16
    snap = (apex or step)["snapshot"]
    d._text(f"Ahead   {ahead(step['snapshot'])}", d.font_mono, t.muted, x, y)
    y += 22
    d._text(f"Position  x {snap.x}", d.font_mono, t.muted, x, y)
    y += 22
    d._text(f"Chosen by Adapt-1: {chosen_so_far} of {n} moves", d.font_mono, t.muted, x, y)
    pg.display.flip()
    return pg.image.tobytes(d.screen, "RGB")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--domain-id", default=None, help="takeoff domain (live runs)")
    ap.add_argument("--apex-domain", default=None, help="apex domain (live runs)")
    ap.add_argument("--features", default="full_v3")
    ap.add_argument("--apex-features", default="apex_v3")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--stall-timeout", type=int, default=60)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--hold", type=float, default=2.5)
    ap.add_argument("--evidence-dir", type=Path, default=Path("artifacts/dashboard-live"))
    ap.add_argument(
        "--from-evidence",
        type=Path,
        default=None,
        help="re-render a recorded live run (its evidence JSON) offline: no API calls",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    steps, looks = [], []
    if not args.from_evidence and not (args.domain_id and args.apex_domain):
        ap.error("a live run needs --domain-id and --apex-domain (or use --from-evidence)")
    if args.from_evidence:
        e = json.loads(args.from_evidence.read_text())
        args.seed = e["seed"]
        policy = Recorder(Recorded(e["decisions"], "takeoff"), steps)
        apex_policy = Recorder(Recorded(e["apex"], "apex"), looks)
    else:
        client = Adapt1Client()
        policy = Recorder(
            Adapt1Policy(
                client,
                args.domain_id,
                allow_exploration=False,
                feature_names=FEATURE_SETS[args.features],
                sequential=False,
                run_id=f"{args.domain_id}-dashboard",
            ),
            steps,
        )
        apex_policy = Recorder(
            Adapt1Policy(
                client,
                args.apex_domain,
                allow_exploration=False,
                feature_names=FEATURE_SETS[args.apex_features],
                sequential=False,
                run_id=f"{args.apex_domain}-dashboard",
            ),
            looks,
        )
    captured: list = []
    original_write_gif = rec.write_gif
    rec.write_gif = lambda frames, out, **kw: captured.extend(frames) or 0
    try:
        stats = rec.record_episode(
            args.env,
            policy,
            Path(tempfile.mkdtemp()) / "x.gif",
            max_decisions=args.max_decisions,
            seed=args.seed,
            stall_timeout=args.stall_timeout,
            cadence="grounded",
            apex_policy=apex_policy,
            actions=tuple(Macro(m) for m in MACROS.split(",")),
        )
    finally:
        rec.write_gif = original_write_gif
    print(
        f"played: max_x={stats['max_x']} flag={stats['reached_flag']} decisions={stats['decisions']} "
        f"frames={len(captured)} status={dict(policy.status_counts)}",
        flush=True,
    )

    # frame -> (takeoff index, apex look or None): takeoffs run back to back; an apex look happens
    # `frame` frames into its takeoff (the trace's apex row with decision d belongs to takeoff d + 1)
    trace = stats["trace"]
    takeoffs = [r for r in trace if r["kind"] == "takeoff"]
    apex_rows = [r for r in trace if r["kind"] == "apex"]
    apex_at = {}
    for i, r in enumerate(apex_rows):
        apex_at[r["decision"]] = (r, looks[i])
    dash = LiveDashboard(brain="Adapt-1")
    out_frames, f = [], 0
    chosen = 0
    for i, t in enumerate(takeoffs):
        step = steps[i]
        chosen += step["decision"].selection_status == "selected"
        a = apex_at.get(t["decision"] - 1)
        for k in range(t["frames"]):
            if f >= len(captured):
                break
            look = a[1] if a and k >= a[0]["frame"] else None
            out_frames.append(draw(dash, captured[f], i + 1, len(takeoffs), step, look, chosen))
            f += 1
    w, h = dash.screen.get_size()
    out_frames += out_frames[-1:] * int(args.hold * args.fps)
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
    for fr in out_frames:
        enc.stdin.write(fr)
    enc.stdin.close()
    enc.wait()

    if args.from_evidence:
        print(f"{len(out_frames)} frames -> {args.out} (re-rendered from {args.from_evidence})")
        return 0 if enc.returncode == 0 else 1
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    ev = args.evidence_dir / f"{args.domain_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    ev.write_text(
        json.dumps(
            {
                "domain_id": args.domain_id,
                "apex_domain": args.apex_domain,
                "seed": args.seed,
                "stats": {k: v for k, v in stats.items() if k != "trace"},
                "trace": trace,
                "decisions": [
                    {
                        "action": s["decision"].action.value,
                        "status": s["decision"].selection_status,
                        "telemetry": s["decision"].telemetry,
                    }
                    for s in steps
                ],
                "apex": [
                    {
                        "choice": s["decision"].action.value,
                        "status": s["decision"].selection_status,
                        "telemetry": s["decision"].telemetry,
                    }
                    for s in looks
                ],
            },
            indent=1,
            default=str,
        )
    )
    print(f"{len(out_frames)} frames -> {args.out}; evidence {ev}")
    return 0 if enc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
