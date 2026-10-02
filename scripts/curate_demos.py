#!/usr/bin/env python
"""Curate a demonstration dataset down to a transition cap — offline, FREE.

Ranks whole episodes by quality (reached_flag, then how far they got) and greedily keeps
the best ones until --cap transitions, so the curated set is both high-quality AND fits the
Records budget you'll spend ingesting it (1 transition = 1 Record). Episodes are kept
contiguous and intact — sequential credit assignment needs whole episodes, so we never
split one.

Usage:
    python scripts/curate_demos.py --dataset artifacts/demos/heuristic-full.jsonl \
        --cap 1000 --out artifacts/demos/curated-1000.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: <dataset>.manifest.jsonl")
    ap.add_argument("--cap", type=int, default=1000, help="max transitions to keep (= Records)")
    ap.add_argument("--min-episodes", type=int, default=8,
                    help="the sequential_q_mlp needs >=8 episodes to install (minimum_episodes "
                         "default). Curation ASSERTS this floor and exits non-zero if unmet, so "
                         "an ingest can never be spent on a set that won't install a model.")
    ap.add_argument("--max-episode-transitions", type=int, default=0,
                    help="skip episodes longer than this in selection (0 = no limit). A few very "
                         "long episodes (e.g. full clears) otherwise hog the cap and starve the "
                         "episode count; capping length keeps many moderate pipe-clearers instead.")
    ap.add_argument("--max-flag-episodes", type=int, default=0,
                    help="keep at most this many flag-reaching episodes (0 = unlimited). Flag runs "
                         "rank first, so a small cap otherwise becomes a highlight reel with no "
                         "deaths/recoveries (design-notes §8) and no headroom for online learning "
                         "to show a curve (findings #21 plan: terrain by imitation, enemy timing "
                         "online).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest_path = args.manifest or args.dataset.with_suffix(".manifest.jsonl")
    episodes = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]

    # Optionally exclude oversized episodes so they can't monopolise the transition cap and
    # push the kept episode count below the install floor.
    selectable = episodes
    if args.max_episode_transitions > 0:
        selectable = [e for e in episodes if e["transitions"] <= args.max_episode_transitions]

    # Rank: flag-reaching episodes first, then by how far they got. Best demonstrations win.
    selectable = sorted(
        selectable, key=lambda e: (e.get("reached_flag", False), e.get("max_x", 0)), reverse=True
    )

    kept_ids: set[str] = set()
    kept_total = 0
    kept_flags = 0
    for e in selectable:
        if e.get("reached_flag") and args.max_flag_episodes and kept_flags >= args.max_flag_episodes:
            continue
        if kept_total + e["transitions"] <= args.cap:
            kept_ids.add(e["episode_id"])
            kept_total += e["transitions"]
            kept_flags += int(bool(e.get("reached_flag")))
    # If nothing fit (every episode bigger than the cap), keep the single best, truncated later.
    if not kept_ids and selectable:
        kept_ids.add(selectable[0]["episode_id"])

    # Filter the dataset to kept episodes, preserving file (episode/step) order.
    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for line in args.dataset.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["episode_id"] in kept_ids and written < args.cap:
                sink.write(line + "\n")
                written += 1

    kept_meta = [e for e in episodes if e["episode_id"] in kept_ids]
    pipe_clearing = sum(1 for e in kept_meta if e.get("max_x", 0) > 723)
    print(json.dumps({
        "cap": args.cap,
        "episodes_available": len(episodes),
        "episodes_selectable": len(selectable),
        "episodes_kept": len(kept_ids),
        "transitions_written": written,
        "flag_episodes_kept": sum(1 for e in kept_meta if e.get("reached_flag")),
        "pipe_clearing_episodes_kept": pipe_clearing,  # max_x > 723 → contains the pipe-jump
        "max_x_kept_best": max((e.get("max_x", 0) for e in kept_meta), default=0),
        "max_x_kept_worst": min((e.get("max_x", 0) for e in kept_meta), default=0),
        "out": str(args.out),
    }, indent=2))

    if len(kept_ids) < args.min_episodes:
        print(f"\nERROR: kept {len(kept_ids)} episodes < --min-episodes {args.min_episodes}: the "
              f"sequential_q_mlp will NOT install. Raise --cap or --max-episode-transitions.")
        return 1
    from typesafe_mario.provenance import write_provenance
    write_provenance(args.out, kind="curation", inputs={"dataset": args.dataset}, extra={k: v for k, v in vars(args).items() if k not in ('dataset', 'out')})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
