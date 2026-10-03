#!/usr/bin/env python
"""Frozen use of an acquired Machina sequence. Queries only, 0 Records.

`propose` with mode "frozen" from the reset observation, execute it for real, repeat N times; no observe
is sent, so nothing is learned. --env SuperMarioBros-2-1-v0 is the transfer contrast (the sequence was
acquired on 1-1). Prints max_x / flag per repeat; journals proposals and traces.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "src")

from typesafe_mario.adapt1_client import Adapt1Client  # noqa: E402
from typesafe_mario.machina import GOAL, Journal, MachinaClient, execute_sequence, state_row  # noqa: E402
from typesafe_mario.runner import _unwrap_ram, create_mario_env  # noqa: E402
from typesafe_mario.state import MarioStateParser  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain-id", required=True)
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--journal-dir", type=Path, default=Path("artifacts/machina"))
    ap.add_argument(
        "--level-length",
        type=float,
        default=None,
        help="px where the flag is; must match the domain's acquisition run (default 3161; mach2 2-1 domains: 3193)",
    )
    args = ap.parse_args()
    if args.level_length is not None:
        from typesafe_mario.machina import set_level_length

        set_level_length(args.level_length)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    journal = Journal(args.journal_dir / f"{args.domain_id}-frozen-{args.env}-{ts}.jsonl")
    m = MachinaClient(Adapt1Client(), args.domain_id, journal)
    st = m.trajectory("state", {})
    print(
        "state:",
        {k: st.get(k) for k in ("status", "retained_trajectories", "pending_decisions", "version")},
    )
    results = []
    for i in range(args.repeats):
        env = create_mario_env(args.env, render_mode="rgb_array")
        parser = MarioStateParser(decision_horizon_frames=8)
        _f, info = env.reset(seed=777)
        snap = parser.parse(info, _unwrap_ram(env), previous_action=None)
        prop = m.trajectory(
            "propose",
            {
                "state": state_row(snap),
                "goal": GOAL,
                "mode": "frozen",
                "request_id": f"{args.domain_id}-frozen-{args.env}-{ts}-{i}",
            },
        )
        trace = execute_sequence(env, parser, snap, prop["actions"])
        env.close()
        journal.write(
            "frozen_execution",
            {"proposal_meta": {k: v for k, v in prop.items() if k != "actions"}, "trace": trace},
        )
        results.append(trace)
        print(
            f"repeat {i}: proposal rows={len(prop['actions'])} source={prop.get('source')} status={prop.get('status')} "
            f"-> executed {trace['executed']} max_x={trace['max_x']} {'FLAG' if trace['flag'] else 'dead' if trace['dead'] else 'stall/horizon'}",
            flush=True,
        )
    print(
        f"\nfrozen {args.domain_id} on {args.env}: max_x {[t['max_x'] for t in results]} flags {sum(t['flag'] for t in results)}/{len(results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
