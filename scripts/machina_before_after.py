"""Offline, 0 Records: attempt 1 (Machina's own 1-1 sequence on 2-1) beside attempt 278 (the first 2-1 flag), re-executed
from the acquisition journal, side by side at 2x speed (`mach2`, 2026-10-03). Run from mario/agent:

    python scripts/machina_before_after.py <seeded-b acquisition journal .jsonl.gz> before-after.mp4
"""
import gzip, json, subprocess, sys
sys.path.insert(0, "src")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typesafe_mario.machina import execute_sequence, set_level_length
from typesafe_mario.runner import _unwrap_ram, create_mario_env
from typesafe_mario.state import MarioStateParser

journal, out = sys.argv[1], sys.argv[2]
set_level_length(3193.0)
ex = [json.loads(l)["record"] for l in gzip.open(journal, "rt") if '"kind": "execution"' in l or '"kind":"execution"' in l]
def frames_of(rec):
    env = create_mario_env("SuperMarioBros-2-1-v0", render_mode="rgb_array"); p = MarioStateParser(decision_horizon_frames=8)
    _f, info = env.reset(seed=777); snap = p.parse(info, _unwrap_ram(env), previous_action=None)
    fr = []; t = execute_sequence(env, p, snap, rec["actions"], on_frame=lambda f: fr.append(f.copy())); env.close()
    return fr, t
a, ta = frames_of(ex[0]); b, tb = frames_of(ex[277])
print("attempt 1:", ta["max_x"], ta["flag"], len(a), "| attempt 278:", tb["max_x"], tb["flag"], len(b))
SCALE, GAP, TOP = 3, 24, 92
w, h = 256 * SCALE, 240 * SCALE
W, H = 2 * w + GAP + 48, h + TOP + 40
try:
    F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34); f2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
except Exception:
    F = f2 = ImageFont.load_default()
n = max(len(a), len(b)); hold = 120
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "60", "-i", "-",
                         "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
for k in list(range(0, n, 2)) + [n - 1] * hold:   # 2x speed, then hold the last frame 2 s
    im = Image.new("RGB", (W, H), (16, 18, 22)); d = ImageDraw.Draw(im)
    for col, frs, t, title, end in ((0, a, ta, "Attempt 1: its own 1-1 sequence", f"dies at x {ta['max_x']}"),
                                    (1, b, tb, "Attempt 278: revised into 2-1", "flag")):
        x0 = 24 + col * (w + GAP)
        d.text((x0, 20), title, font=F, fill=(235, 235, 235))
        im.paste(Image.fromarray(frs[min(k, len(frs) - 1)]).resize((w, h), Image.NEAREST), (x0, TOP))
        if k >= len(frs) - 1:
            d.text((x0, TOP + h + 6), end, font=f2, fill=(240, 180, 90) if not t["flag"] else (120, 220, 140))
    d.text((W - 430, H - 32), "Adapt-1 Machina · World 2-1 · 2x speed", font=f2, fill=(140, 140, 150))
    proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close(); proc.wait()
print("wrote", out)
