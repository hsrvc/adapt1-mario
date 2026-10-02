#!/usr/bin/env python
"""Record a demonstration dataset with the diversified heuristic teacher — offline, FREE.

No API, no quota: the teacher plays locally at emulator speed and every transition is
written to a JSONL dataset for a later (quota-spending) warm-start ingestion. Prints the
dataset's quality stats (flag rate, mean/best max-x) so you can judge it before ingesting.

Usage:
    python scripts/record_demos.py --features full --episodes 200 --epsilon 0.15 \
        --out artifacts/demos/heuristic-full.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typesafe_mario.adapt1_policy import FEATURE_SETS
from typesafe_mario.demos import record_demonstrations
from typesafe_mario.policy import DiversifiedHeuristicPolicy, TypeSafePolicy
from typesafe_mario.macros import Macro, CADENCES
from typesafe_mario.reward import DEFAULT as SHAPED_REWARD
from typesafe_mario.reward import GROUNDED, OBSTACLE_AWARE, PROGRESS_FRACTION, PROGRESS_RATE

REWARD_CONFIGS = {
    "shaped": SHAPED_REWARD,
    "obstacle-aware": OBSTACLE_AWARE,
    "grounded": GROUNDED,  # macro-scale progress + terminal overrides (findings #21)
    "progress": PROGRESS_FRACTION,  # sums to <= 1 per episode: the engine's [0,1] contract (#22)
    "progress-rate": PROGRESS_RATE,  # same, per 8 frames: no bias toward long macros (#22, curve)
    "raw": None,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", choices=tuple(FEATURE_SETS), default="full")
    ap.add_argument("--macros", default=None,
                    help="comma-separated grounded macros the teacher may choose (ε draws included); "
                         "the 8 of #28 = every Macro but `jump`. Default: every Macro.")
    ap.add_argument("--episodes", type=int, default=200)
    # eps=0.10 measured (2026-09-18) as the reach/flag sweet spot: mean max_x ~1122 with
    # occasional full clears; higher just adds noise and lowers reach.
    ap.add_argument("--epsilon", type=float, default=0.10, help="exploration/noise fraction")
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--frames-per-decision", type=int, default=8)
    ap.add_argument("--max-decisions", type=int, default=250)
    ap.add_argument("--stall-timeout", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reward",
        choices=tuple(REWARD_CONFIGS),
        default="shaped",
        help="shaped = §5 normalised composite (Δx + terminal flag/death); "
        "obstacle-aware = shaped + #17 immediate jump-bonus/stall-penalty at a flagged "
        "obstacle (the x=723 lever); raw = gym reward",
    )
    # jev = TypeSafe's pretrained model (the upstream teacher). It is competent and
    # level-general (unlike the 1-1-tuned heuristic) but needs the `typesafe-sdk` package
    # (pip install -e ".[jev]") and a TYPESAFE_API_KEY, and each decision is a hosted call
    # (slow, possibly metered). Use it to distill a strong, multi-level teacher into the
    # warm-start set. See design-notes §11.
    ap.add_argument("--teacher", choices=("heuristic", "jev"), default="heuristic")
    ap.add_argument("--cadence", choices=CADENCES, default="frame",
                    help="frame = one 8-frame Action per decision (the #20 baseline); grounded = "
                         "one committed Macro per decision, decided only on the ground (#21). "
                         "Use --reward grounded with grounded cadence.")
    ap.add_argument("--apex", action="store_true",
                    help="grounded cadence only (design-notes §14a): ask the teacher once at the top "
                         "of every jump (keep / brake / pull back) and write the jump as two rows "
                         "(kind takeoff / apex) carrying the feature superset (+ apex_vx/rise/height).")
    ap.add_argument("--out", type=Path, default=Path("artifacts/demos/heuristic.jsonl"))
    args = ap.parse_args()

    if args.teacher == "jev":
        teacher = TypeSafePolicy()  # raises a clear error if typesafe-sdk / key is missing
    else:
        teacher = DiversifiedHeuristicPolicy(epsilon=args.epsilon, rng_seed=args.seed)
    from typesafe_mario.provenance import write_provenance  # noqa: E402

    stats = record_demonstrations(
        args.env,
        teacher,
        args.out,
        feature_names=FEATURE_SETS[args.features],
        episodes=args.episodes,
        frames_per_decision=args.frames_per_decision,
        max_decisions=args.max_decisions,
        stall_timeout=args.stall_timeout,
        seed=args.seed,
        reward_config=REWARD_CONFIGS[args.reward],
        cadence=args.cadence,
        apex=args.apex,
        actions=(tuple(Macro(m.strip()) for m in args.macros.split(",") if m.strip()) if args.macros else None),
    )
    stats["reward"] = args.reward
    stats["cadence"] = args.cadence
    print(json.dumps(stats, indent=2))
    write_provenance(args.out, kind="demos", extra={"stats": {k: v for k, v in stats.items() if k != 'manifest'}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
