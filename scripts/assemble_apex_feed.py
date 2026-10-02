#!/usr/bin/env python
"""Assemble the warm-start apex feed (377 rows) from its two pools. Offline, 0 Records.

The feed is every apex row of the curated demonstrations, in file order, followed by the first N rows per apex
choice of the coverage pool, ordered branch by branch (choices alphabetical within a branch). With the pools built
by the commands in runs/warm-start/README.md and N = 90 this writes the fed file byte for byte
(sha256 18b116aaece4c8b48096404a58f36ec728acd633f052acfafa856ae4da5957dd).

    python scripts/assemble_apex_feed.py --demos artifacts/demos/curated-k6h128.jsonl \\
        --coverage artifacts/demos/apex-coverage-k6h128.jsonl --out artifacts/demos/warm-start-apex.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def assemble(demos: list[dict], coverage: list[dict], per_choice: int) -> list[dict]:
    rows = [r for r in demos if r["kind"] == "apex"]
    taken: Counter[str] = Counter()
    picked = []
    for r in coverage:
        if r["kind"] == "apex" and taken[r["policy"]] < per_choice:
            picked.append(r)
            taken[r["policy"]] += 1

    def branch(r: dict) -> str:
        return r["episode_id"].rsplit("-apex_", 1)[0]

    first: dict[str, int] = {}
    for i, r in enumerate(picked):
        first.setdefault(branch(r), i)
    return rows + sorted(picked, key=lambda r: (first[branch(r)], r["policy"]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--demos", type=Path, required=True, help="curated demonstrations with returns")
    ap.add_argument("--coverage", type=Path, required=True, help="apex coverage pool with returns")
    ap.add_argument("--per-choice", type=int, default=90)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    load = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]  # noqa: E731
    rows = assemble(load(args.demos), load(args.coverage), args.per_choice)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    print(
        f"wrote {len(rows)} rows to {args.out}  sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
