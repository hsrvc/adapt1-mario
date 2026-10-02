"""Plot the zero-start learning curves from the stage bundles — a static PNG + SVG (offline, 0 Records).

    python scripts/plot_zero_curve.py                      # every run listed in RUNS (missing stages are skipped)
    python scripts/plot_zero_curve.py -o ../runs/zero-learning-curve

Reads each stage bundle's `curve.json` (written by scripts/zero_curve.py from the run log) and concatenates a run's
stages the way the curve page does (episode numbers and Records become cumulative; a later stage's ep000 re-measures
the previous final and is dropped). Runs are drawn on the same axes, one hue per run: dots joined by a thin line are
online episodes (a ring on every clear), diamonds are frozen checkpoints (hollow when the checkpoint was played with no
model installed). No plotting library: SVG by hand, PNG with Pillow (already a dependency). The subtitle's numbers come
from the same data as the marks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FLAG_X = 3161
Y_MAX = 3300
W, H = 1400, 720
M = {"t": 96, "r": 40, "b": 70, "l": 90}
PAPER, INK, INK2, INK3, RULE = "#fcfcfb", "#1c1b18", "#55544e", "#8a8880", "#e5e3dd"
# one hue per run (the curve page's validated pair): the live run blue, the superseded run orange
RUNS = [
    {
        "key": "zero2",
        "label": "zero2 (budget 16,384)",
        "color": "#2a78d6",
        "stages": ["2026-10-01-zero2-stage1", "2026-10-01-zero2-stage2"]
        + [f"2026-10-02-zero2-stage{i}" for i in range(3, 5)]
        + ["2026-10-02-zero2-stage5-frozen-clear"],
    },
    {
        "key": "zero",
        "label": "zero (budget 4,096, evicting from stage 4)",
        "color": "#eb6834",
        "stages": [f"2026-10-01-zero-pilot-stage{i}" for i in range(1, 5)],
    },
]


def load_run(runs_dir: Path, run: dict) -> dict | None:
    online, frozen, marks = [], [], []
    ep_off = rec_off = 0
    for i, d in enumerate(run["stages"]):
        p = runs_dir / d / "curve.json"
        if not p.exists():  # the public tree names the stages runs/zero-start/stage-N
            p = runs_dir / "zero-start" / f"stage-{i + 1}" / "curve.json"
        if not p.exists() or (run["key"] != "zero2" and "zero-start" in str(p)):
            continue
        c = json.loads(p.read_text())
        marks.append({"label": f"stage {i + 1}", "records_from": rec_off})
        last = rec_off
        for r in c["online"]:
            online.append(
                {**r, "episode": r["episode"] + ep_off, "records": r["records"] + rec_off}
            )
            last = r["records"] + rec_off
        for f in c["frozen"]:
            if f["episode"] == 0 and frozen and f["reach"] == frozen[-1]["reach"]:
                continue  # a later stage's opening read that merely repeats the previous final
            frozen.append(
                {**f, "episode": f["episode"] + ep_off, "records": f["records"] + rec_off}
            )
        ep_off = online[-1]["episode"]
        rec_off = last
    return {**run, "online": online, "frozen": frozen, "marks": marks} if online else None


def layout(runs):
    iw, ih = W - M["l"] - M["r"], H - M["t"] - M["b"]
    x1 = max(max(r["records"] for r in run["online"]) for run in runs) or 1
    x = lambda v: M["l"] + v / x1 * iw  # noqa: E731
    y = lambda v: M["t"] + ih - v / Y_MAX * ih  # noqa: E731
    step = 1000 if x1 > 3000 else 500 if x1 > 1200 else 250
    return iw, ih, x, y, list(range(0, int(x1) + 1, step)), list(range(0, 3001, 500))


def subtitle(run) -> str:
    clears = sum(r["reach"] >= FLAG_X for r in run["online"])
    fz = " → ".join(
        f"{f['reach']}{'*' if f.get('takeoff_installed') is False else ''}" for f in run["frozen"]
    )
    return (
        f"{run['label']}: {len(run['online'])} episodes, {run['online'][-1]['records']:,} Records, {clears} online clears; "
        f"frozen {fz}"
    )


def svg(runs) -> str:
    iw, ih, x, y, xt, yt = layout(runs)
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
        f'<text x="{M["l"]}" y="30" font-size="22" font-weight="bold" fill="{INK}">Adapt-1 learns Mario 1-1 from nothing — online reach per episode against Records spent</text>',
    ]
    for i, run in enumerate(runs):
        o.append(f'<circle cx="{M["l"] + 6}" cy="{52 + 18 * i}" r="5" fill="{run["color"]}"/>')
        o.append(
            f'<text x="{M["l"] + 18}" y="{57 + 18 * i}" font-size="13" fill="{INK2}">{subtitle(run)}</text>'
        )
    o.append(
        f'<text x="{M["l"] + iw}" y="{57 + 18 * (len(runs) - 1)}" font-size="12" text-anchor="end" fill="{INK3}">* = checkpoint played with no model installed</text>'
    )
    for v in yt:
        o.append(
            f'<line x1="{M["l"]}" x2="{M["l"] + iw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{RULE}"/>'
        )
        o.append(
            f'<text x="{M["l"] - 10}" y="{y(v) + 4:.1f}" font-size="12" text-anchor="end" fill="{INK3}">{v:,}</text>'
        )
    for v in xt:
        o.append(
            f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{M["t"] + ih}" y2="{M["t"] + ih + 5}" stroke="{RULE}"/>'
        )
        o.append(
            f'<text x="{x(v):.1f}" y="{M["t"] + ih + 20}" font-size="12" text-anchor="middle" fill="{INK3}">{v:,}</text>'
        )
    o.append(
        f'<line x1="{M["l"]}" x2="{M["l"] + iw}" y1="{M["t"] + ih}" y2="{M["t"] + ih}" stroke="{RULE}"/>'
    )
    o.append(
        f'<text x="{M["l"] + iw}" y="{M["t"] + ih + 42}" font-size="12" text-anchor="end" fill="{INK3}">Records spent (cumulative, per run)</text>'
    )
    o.append(
        f'<text x="{M["l"] - 10}" y="{M["t"] - 12}" font-size="12" text-anchor="end" fill="{INK3}">reach, px</text>'
    )
    o.append(
        f'<line x1="{M["l"]}" x2="{M["l"] + iw}" y1="{y(FLAG_X):.1f}" y2="{y(FLAG_X):.1f}" stroke="{INK3}" stroke-dasharray="4 3"/>'
    )
    o.append(
        f'<text x="{M["l"] + 6}" y="{y(FLAG_X) - 6:.1f}" font-size="12" fill="{INK2}">flag · 3161</text>'
    )
    for run in reversed(runs):  # the live run (first) draws on top
        c = run["color"]
        pts = " ".join(f"{x(r['records']):.1f},{y(r['reach']):.1f}" for r in run["online"])
        o.append(
            f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.5" stroke-linejoin="round" opacity="0.75"/>'
        )
        for r in run["online"]:
            cx, cy = x(r["records"]), y(r["reach"])
            if r["reach"] >= FLAG_X:
                o.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{PAPER}" stroke="{c}" stroke-width="1.5"/>'
                )
            o.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{c}" stroke="{PAPER}" stroke-width="1.2"/>'
            )
    for run in reversed(runs):
        c = run["color"]
        for f in run["frozen"]:
            cx, cy = x(f["records"]), y(f["reach"])
            hollow = f.get("takeoff_installed") is False
            o.append(
                f'<rect x="{cx - 7:.1f}" y="{cy - 7:.1f}" width="14" height="14" fill="{PAPER if hollow else c}" stroke="{c if hollow else PAPER}" stroke-width="2" transform="rotate(45 {cx:.1f} {cy:.1f})"/>'
            )
            right = cx > M["l"] + iw - 70
            o.append(
                f'<text x="{cx:.1f}" y="{cy + 24:.1f}" font-size="12" font-weight="bold" text-anchor="middle" fill="{INK}">{f["reach"]:,}</text>'
                if right
                else f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}" font-size="12" font-weight="bold" text-anchor="start" fill="{INK}">{f["reach"]:,}</text>'
            )
    lx, ly = M["l"] + iw - 420, M["t"] + ih - 24
    o.append(
        f'<circle cx="{lx}" cy="{ly}" r="4" fill="{INK3}"/><text x="{lx + 10}" y="{ly + 4}" font-size="12" fill="{INK2}">online episode (learning on)</text>'
    )
    o.append(
        f'<rect x="{lx + 193}" y="{ly - 6}" width="12" height="12" fill="{INK3}" transform="rotate(45 {lx + 199} {ly})"/><text x="{lx + 212}" y="{ly + 4}" font-size="12" fill="{INK2}">frozen checkpoint (learning off)</text>'
    )
    o.append("</svg>")
    return "\n".join(o)


def png(runs, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    iw, ih, x, y, xt, yt = layout(runs)
    S = 2
    im = Image.new("RGB", (W * S, H * S), PAPER)
    dr = ImageDraw.Draw(im)

    def font(size, bold=False):
        for name in (
            ("Helvetica-Bold.ttf" if bold else "Helvetica.ttf"),
            ("Arial Bold.ttf" if bold else "Arial.ttf"),
        ):
            for folder in (
                "/System/Library/Fonts/Supplemental",
                "/System/Library/Fonts",
                "/Library/Fonts",
            ):
                p = Path(folder) / name
                if p.exists():
                    return ImageFont.truetype(str(p), size * S)
        try:
            return ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", size * S, index=1 if bold else 0
            )
        except OSError:
            return ImageFont.load_default()

    f12, f13, f22, f12b = font(12), font(13), font(22, True), font(12, True)
    s = lambda v: v * S  # noqa: E731

    def text(tx, ty, msg, fnt, fill, anchor="la"):
        dr.text((s(tx), s(ty)), msg, font=fnt, fill=fill, anchor=anchor)

    def dashed(p0, p1, dash=(4, 3), fill=INK3):
        (x0, y0), (x1_, y1_) = p0, p1
        length = ((x1_ - x0) ** 2 + (y1_ - y0) ** 2) ** 0.5
        t = 0.0
        while t < length:
            a, b = t / length, min((t + dash[0]) / length, 1.0)
            dr.line(
                [
                    (s(x0 + (x1_ - x0) * a), s(y0 + (y1_ - y0) * a)),
                    (s(x0 + (x1_ - x0) * b), s(y0 + (y1_ - y0) * b)),
                ],
                fill=fill,
                width=S,
            )
            t += sum(dash)

    text(
        M["l"],
        16,
        "Adapt-1 learns Mario 1-1 from nothing — online reach per episode against Records spent",
        f22,
        INK,
    )
    for i, run in enumerate(runs):
        cx, cy = s(M["l"] + 6), s(52 + 18 * i)
        dr.ellipse([cx - 5 * S, cy - 5 * S, cx + 5 * S, cy + 5 * S], fill=run["color"])
        text(M["l"] + 18, 52 + 18 * i, subtitle(run), f13, INK2, "lm")
    text(
        M["l"] + iw,
        52 + 18 * (len(runs) - 1),
        "* = checkpoint played with no model installed",
        f12,
        INK3,
        "rm",
    )
    for v in yt:
        dr.line([(s(M["l"]), s(y(v))), (s(M["l"] + iw), s(y(v)))], fill=RULE, width=S)
        text(M["l"] - 10, y(v), f"{v:,}", f12, INK3, "rm")
    for v in xt:
        dr.line([(s(x(v)), s(M["t"] + ih)), (s(x(v)), s(M["t"] + ih + 5))], fill=RULE, width=S)
        text(x(v), M["t"] + ih + 12, f"{v:,}", f12, INK3, "mt")
    dr.line([(s(M["l"]), s(M["t"] + ih)), (s(M["l"] + iw), s(M["t"] + ih))], fill=RULE, width=S)
    text(M["l"] + iw, M["t"] + ih + 32, "Records spent (cumulative, per run)", f12, INK3, "rt")
    text(M["l"] - 10, M["t"] - 12, "reach, px", f12, INK3, "rm")
    dashed((M["l"], y(FLAG_X)), (M["l"] + iw, y(FLAG_X)))
    text(M["l"] + 6, y(FLAG_X) - 16, "flag · 3161", f12, INK2)
    for run in reversed(runs):
        c = run["color"]
        dr.line(
            [(s(x(r["records"])), s(y(r["reach"]))) for r in run["online"]],
            fill=c,
            width=int(1.5 * S),
            joint="curve",
        )
        for r in run["online"]:
            cx, cy = s(x(r["records"])), s(y(r["reach"]))
            if r["reach"] >= FLAG_X:
                dr.ellipse(
                    [cx - 8 * S, cy - 8 * S, cx + 8 * S, cy + 8 * S],
                    fill=PAPER,
                    outline=c,
                    width=int(1.5 * S),
                )
            dr.ellipse(
                [cx - 3.5 * S, cy - 3.5 * S, cx + 3.5 * S, cy + 3.5 * S],
                fill=c,
                outline=PAPER,
                width=S,
            )
    for run in reversed(runs):
        c = run["color"]
        for f in run["frozen"]:
            cx, cy = s(x(f["records"])), s(y(f["reach"]))
            hollow = f.get("takeoff_installed") is False
            r = 9 * S
            dr.polygon(
                [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                fill=PAPER if hollow else c,
                outline=c if hollow else PAPER,
                width=2 * S,
            )
            if x(f["records"]) > M["l"] + iw - 70:
                text(x(f["records"]), y(f["reach"]) + 14, f"{f['reach']:,}", f12b, INK, "mt")
            else:
                text(x(f["records"]) + 12, y(f["reach"]), f"{f['reach']:,}", f12b, INK, "lm")
    lx, ly = M["l"] + iw - 420, M["t"] + ih - 24
    dr.ellipse([s(lx) - 4 * S, s(ly) - 4 * S, s(lx) + 4 * S, s(ly) + 4 * S], fill=INK3)
    text(lx + 10, ly, "online episode (learning on)", f12, INK2, "lm")
    cx, cy, r = s(lx + 199), s(ly), 6 * S
    dr.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=INK3)
    text(lx + 212, ly, "frozen checkpoint (learning off)", f12, INK2, "lm")
    im.resize((W, H), Image.LANCZOS).save(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output path without extension (default: mario/runs/zero-learning-curve)",
    )
    args = ap.parse_args()
    # runs/ sits beside scripts/ in the public tree and two levels up (mario/runs) in this repo
    here = Path(__file__).resolve()
    runs_dir = next(p / "runs" for p in (here.parents[1], here.parents[2]) if (p / "runs").is_dir())
    runs = [r for r in (load_run(runs_dir, r) for r in RUNS) if r]
    if not runs:
        raise SystemExit("no bundle with a curve.json found")
    out = args.out or runs_dir / "zero-learning-curve"
    out.with_suffix(".svg").write_text(svg(runs))
    png(runs, out.with_suffix(".png"))
    print(f"wrote {out}.svg and {out}.png: " + " | ".join(subtitle(r) for r in runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
