#!/usr/bin/env python
"""Rewrite a demonstration dataset's per-step reward as a client-side k-step return — offline, FREE.

findings #22: on every Mario domain the server's own delayed credit never reached the decision that
mattered (the max jump off pipe 4 kept a higher value than running off it after 216 observations,
with n_step 5 and 20). This computes that credit on the client from the recorded trajectory:

    R_t = ( sum_{i<k} gamma^i * r_{t+i} ) / ( sum_{i<k} gamma^i )      within the episode

so a decision that ends the episode early (a death) is worth only the progress it earned before
dying, and a decision followed by progress is worth that progress. Normalised by the geometric sum
so R_t stays in [0,1] when r is (PROGRESS_FRACTION rewards are). Feed the result to a DIRECT-feedback
domain (no `sequential` block): the learner then only has to regress a per-state immediate value.

This is doing the sequential learner's job on the client. Say so in any write-up.

    python scripts/kstep_returns.py --dataset in.jsonl --out out.jsonl --k 6 --gamma 0.9 --report
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


from typesafe_mario.returns import frame_horizon_returns, kstep_returns  # noqa: E402  (shared with apex_coverage.py)


def _pipe_top_like(ctx: dict) -> bool:
    # standing on a platform with a 4-tile drop 2 tiles ahead and no pit visible: the pipe-4 top
    return (ctx.get("grounded") == 1.0 and ctx.get("drop_dist") == 2.0
            and ctx.get("drop_depth") == 4.0 and ctx.get("gap_dist", 999.0) >= 999.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--report", action="store_true", help="print per-macro mean returns overall and at pipe-4-top-like states")
    ap.add_argument("--frames-horizon", type=int, default=0,
                    help="also write `reward_rate` = the fixed frame-horizon return over this many frames "
                         "(rows need `frames`; 2026-09-23, findings #29 follow-up). 0 = off.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
    out = kstep_returns(rows, args.k, args.gamma)
    if args.frames_horizon > 0:
        out = frame_horizon_returns(out, args.frames_horizon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(out)} rows to {args.out} (k={args.k}, gamma={args.gamma})")
    if args.report:
        by = collections.defaultdict(list)
        top = collections.defaultdict(list)
        for r in out:
            by[r["policy"]].append(r["reward"])
            if _pipe_top_like(r["context"]):
                top[r["policy"]].append(r["reward"])
        print("per-macro mean k-step return (n):")
        for m, v in sorted(by.items()):
            print(f"   {m:<22} {statistics.mean(v):.4f}  (n={len(v)})")
        print("at pipe-4-top-like states:")
        for m, v in sorted(top.items(), key=lambda kv: -statistics.mean(kv[1])):
            print(f"   {m:<22} {statistics.mean(v):.4f}  (n={len(v)})")
        print("episode return sums max:", round(max(sum(r['reward'] for r in out if r['episode_id'] == e)
                                                   for e in {r['episode_id'] for r in out}), 3))
    from typesafe_mario.provenance import write_provenance
    write_provenance(args.out, kind="returns", inputs={"dataset": args.dataset}, extra={k: v for k, v in vars(args).items() if k not in ('dataset', 'out')})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
