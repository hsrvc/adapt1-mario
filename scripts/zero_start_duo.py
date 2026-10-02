#!/usr/bin/env python
"""The zero-start run: the two-domain policy learning ONLINE from nothing.

Two direct-feedback domains — takeoff (8 macros) and apex (three
in-air choices) — no teacher rows, exploration on from the first query, client-side delayed credit
(`online_duo.py`: the frame-horizon return of 2026-09-23 by default). Every decision is one
feedback = one Record; a hard `--max-records` cap is checked before each one. A frozen eval
(Queries only, GIF per seed) runs at episode 0, every `--eval-every`, and at the end; the run
stops on the cap, on `--stop-flat` episodes without a frozen-reach gain, or after `--episodes`.

LIVE and quota-consuming. `--dry-run` runs the same loop against the offline contract client
(0 Records, no network) — run it first. Say which account the key belongs to in the write-up.

    export REI_KEY=...   # your key, from the environment only
    python scripts/zero_start_duo.py --takeoff-domain mario-zero-takeoff --apex-domain mario-zero-apex \\
        --recreate --episodes 100 --max-records 2000 --eval-every 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario.adapt1_client import Adapt1Client, Adapt1Error  # noqa: E402
from typesafe_mario.adapt1_policy import FEATURE_SETS, build_domain_config  # noqa: E402
from typesafe_mario.macros import APEX_NAMES  # noqa: E402
from typesafe_mario.offline_client import OfflineAdapt1Client  # noqa: E402
from typesafe_mario.online_duo import MACROS_28, CreditConfig, RunConfig, wait_ready, zero_start_run  # noqa: E402
from typesafe_mario.provenance import check_clean, git_state, write_provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--takeoff-domain", required=True)
    ap.add_argument("--apex-domain", required=True)
    ap.add_argument(
        "--features",
        choices=("full_v2", "full_v3", "full"),
        default="full_v2",
        help="takeoff feature set (full_v3 for the published run)",
    )
    ap.add_argument("--apex-features", choices=("apex_v2", "apex_v3", "apex"), default="apex_v2")
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="episode e plays seed+e")
    ap.add_argument(
        "--max-records", type=int, default=2000, help="hard cap on feedbacks sent (1 = 1 Record)"
    )
    ap.add_argument("--credit", choices=("frame_horizon", "kstep"), default="frame_horizon")
    ap.add_argument(
        "--horizon", type=int, default=128, help="frame_horizon: frames of future progress credited"
    )
    ap.add_argument("--k", type=int, default=6, help="kstep: decisions")
    ap.add_argument("--gamma", type=float, default=0.9, help="kstep: discount")
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="client-side ε-greedy on top of the server's exploration",
    )
    ap.add_argument(
        "--model-type",
        choices=("auto", "extra_trees", "mlp_v2"),
        default="extra_trees",
        help="pin the model family on both domains (learning.training.model_type; Rei's 2026-09-30 fix). "
        "extra_trees = the class every clear used; auto = let the server choose (it can pick a class that outputs a constant policy)",
    )
    ap.add_argument(
        "--exploration-mode",
        choices=("ucb", "auto"),
        default="ucb",
        help="domain-level exploration (create time): ucb = every domain so far; auto = Rei's default",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=4096,
        help="learning.context.max_samples on new domains = the learner's retained-row budget; at the cap replay "
        "eviction drops rows and the frozen policy regresses (we saw 2757 → 710 past 4,096). Size it above "
        "the Records you intend to spend on the domain",
    )
    ap.add_argument("--max-decisions", type=int, default=200)
    ap.add_argument("--stall-timeout", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument(
        "--eval-seeds", default="777", help="comma-separated seeds for every frozen eval"
    )
    ap.add_argument(
        "--stop-flat",
        type=int,
        default=100,
        help="stop after this many episodes without a frozen-reach gain (0 = never)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/zero-start"))
    ap.add_argument("--run-id", default="zero-start")
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="delete + create both domains first (a fresh, empty learner)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="OFFLINE: the same loop on OfflineAdapt1Client — 0 Records",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="spend Records from a dirty working tree (the diff is recorded in the provenance file)",
    )
    args = ap.parse_args()
    git_at_start = git_state()
    if not args.dry_run:
        check_clean(allow_dirty=args.allow_dirty, what="a zero-start run")

    takeoff_features = FEATURE_SETS[args.features]
    apex_features = FEATURE_SETS[args.apex_features]
    if args.dry_run:
        client: object = OfflineAdapt1Client()
        args.recreate = True
        print("DRY RUN — offline contract client: nothing is sent, no Records are spent")
    else:
        client = Adapt1Client()

    if args.recreate:
        for dom in (args.takeoff_domain, args.apex_domain):
            try:
                client.call("DELETE", f"/domains/{dom}")  # type: ignore[attr-defined]
            except Adapt1Error:
                pass
        client.create_domain(
            build_domain_config(  # type: ignore[attr-defined]
                args.takeoff_domain,
                feature_names=takeoff_features,
                macros=[m.value for m in MACROS_28],
                sequential=False,
                reward_max=1.0,
                exploration_mode=args.exploration_mode,
                max_samples=args.max_samples,
                model_type=None if args.model_type == "auto" else args.model_type,
            )
        )
        client.create_domain(
            build_domain_config(  # type: ignore[attr-defined]
                args.apex_domain,
                feature_names=apex_features,
                macros=list(APEX_NAMES),
                sequential=False,
                reward_max=1.0,
                exploration_mode=args.exploration_mode,
                max_samples=args.max_samples,
                model_type=None if args.model_type == "auto" else args.model_type,
            )
        )
        print(
            f"created bandit domains {args.takeoff_domain} ({args.features}, {len(MACROS_28)} macros) and "
            f"{args.apex_domain} ({args.apex_features}, {len(APEX_NAMES)} choices), "
            f"exploration_mode={args.exploration_mode}, reward max 1.0, model_type={args.model_type}"
        )
        if not args.dry_run:
            # A fresh domain's engine answers 503 (engine_session_loading) for about a minute after
            # creation — longer than the client's retry schedule (findings #34). Wait for a 200.
            wait_ready(client, (args.takeoff_domain, args.apex_domain))

    if not args.dry_run:
        # Pre-flight (findings #46): a learner past learning.context.max_samples evicts rows silently. Refuse to spend
        # a stage that would carry either domain past its budget. Live counts come from a status query (Queries only);
        # the budget is read from the domain itself, so a continuation stage on an old pair is checked against ITS cap.
        from typesafe_mario.online_duo import status_query

        for dom in (args.takeoff_domain, args.apex_domain):
            _s, g = client.call("GET", f"/domains/{dom}")  # type: ignore[attr-defined]
            budget = int(((g.get("learning") or {}).get("context") or {}).get("max_samples") or 0)
            held = status_query(client, dom).get("live_sample_count") or 0
            if budget and held + args.max_records > budget:
                raise SystemExit(
                    f"{dom}: holds {held} rows with context.max_samples {budget}; a {args.max_records}-Record stage could "
                    f"push it past the budget and the learner would evict rows. Raise max_samples on a new "
                    f"domain (--max-samples) or lower --max-records."
                )
            print(
                f"  {dom}: {held} rows held, budget {budget} — a {args.max_records}-Record stage fits"
            )
    cfg = RunConfig(
        takeoff_domain=args.takeoff_domain,
        apex_domain=args.apex_domain,
        env_id=args.env,
        takeoff_features=tuple(takeoff_features),
        apex_features=tuple(apex_features),
        macros=MACROS_28,
        episodes=args.episodes,
        seed=args.seed,
        max_records=args.max_records,
        credit=CreditConfig(kind=args.credit, horizon=args.horizon, k=args.k, gamma=args.gamma),
        epsilon=args.epsilon,
        max_decisions=args.max_decisions,
        stall_timeout=args.stall_timeout,
        eval_every=args.eval_every,
        eval_seeds=tuple(int(s) for s in args.eval_seeds.split(",") if s.strip()),
        stop_flat=args.stop_flat or None,
        out_dir=args.out_dir,
        run_id=args.run_id,
        max_samples=args.max_samples,
    )
    print(
        f"credit={cfg.credit} macros={[m.value for m in cfg.macros]} cap={cfg.max_records} Records "
        f"eval every {cfg.eval_every} on seeds {cfg.eval_seeds}"
    )
    # The sidecar is written when the run log is named (so a crash still leaves one) and rewritten
    # with the summary at the end; git is captured at start AND at the end because commits land
    # during multi-hour runs (findings #43's sidecar was written post hoc — this replaces that).
    prov_extra: dict = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "macros": [m.value for m in MACROS_28],
        "git_at_start": git_at_start,
    }
    run_log_path: list[Path] = []

    def _on_run_log(path: Path) -> None:
        run_log_path.append(path)
        write_provenance(path, kind="zero-run-log", extra={**prov_extra, "status": "running"})

    try:
        summary = zero_start_run(client, cfg, on_run_log=_on_run_log)
    except BaseException as exc:
        if run_log_path:
            write_provenance(
                run_log_path[0],
                kind="zero-run-log",
                extra={**prov_extra, "status": f"aborted: {type(exc).__name__}: {exc}"},
            )
        raise
    write_provenance(
        Path(summary["run_log"]),
        kind="zero-run-log",
        extra={
            **prov_extra,
            "status": "finished",
            "summary": {k: v for k, v in summary.items() if k not in ("run_log", "evals")},
        },
    )
    print(
        f"\ndone: episodes={summary['episodes']} records={summary['records_spent']} "
        f"best_reach={summary['best_reach']} (ep{summary['best_reach_episode']}) stop={summary['stop_reason']}"
    )
    print(f"run log: {summary['run_log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
