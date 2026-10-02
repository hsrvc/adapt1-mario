"""Frozen-eval a trained Mario domain: exploration off, no feedback, queries only.

This is the honest play measurement (design-notes §12, findings #15/#16): `validation_skill`
is Q-prediction accuracy, NOT play skill, so the only way to know whether warm-start moved
actual behaviour is to freeze the policy and run whole episodes on fixed seeds.

FREE of Records — `record_episode` calls only `policy.choose` (→ `/domains/{id}/query`), never
`observe`/feedback, so it spends Queries quota only and adds no training transitions.

Reports per-seed max_x + flag, then the aggregate (flag_rate, mean/median/max reach) — the
numbers that compare against #15's x≈723 wall and the teacher's deterministic reach.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from typesafe_mario.adapt1_client import Adapt1Client
from typesafe_mario.adapt1_policy import FEATURE_SETS, Adapt1Policy
from typesafe_mario.macros import CADENCES
from typesafe_mario.record import record_episode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain-id", required=True)
    ap.add_argument("--features", choices=tuple(FEATURE_SETS), default="full")
    ap.add_argument(
        "--apex-features",
        choices=("apex", "apex_v2", "apex_v3"),
        default="apex",
        help="the apex domain's arm: apex_v2 = per-frame units (findings #33)",
    )
    ap.add_argument(
        "--apex-domain",
        default=None,
        help="grounded cadence (design-notes §14a): a second domain (features `apex`, "
        "hypotheses apex_keep/brake/pull_back) queried once at the top of every jump",
    )
    ap.add_argument(
        "--macros",
        default=None,
        help="comma-separated grounded macros the domain declares (e.g. the 8 of #28 without `jump`); "
        "the fallback on abstention picks among these. Default: every Macro.",
    )
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument(
        "--seeds",
        default="777,1,2,3,4",
        help="comma-separated eval seeds (each is one frozen episode)",
    )
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument(
        "--stall-timeout",
        type=int,
        default=60,
        help="break an episode after N decisions with no forward progress",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/frozen-eval"))
    ap.add_argument(
        "--tcp",
        action="store_true",
        help="the domain has learning.temporal_context: send (episode_id, step) on every query",
    )
    ap.add_argument(
        "--cadence",
        choices=CADENCES,
        default="frame",
        help="must match the cadence the domain was trained with (its hypotheses "
        "are Action names for frame, Macro names for grounded)",
    )
    ap.add_argument(
        "--wait-installed",
        type=float,
        default=0,
        metavar="SECONDS",
        help="first wait up to SECONDS while a learner is retraining with no model installed "
        "(right after feeding rows the server answers from a fallback and does not say so)",
    )
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    actions = None
    if args.macros:
        from typesafe_mario.macros import Macro

        actions = tuple(Macro(m.strip()) for m in args.macros.split(",") if m.strip())
    client = Adapt1Client()
    if args.wait_installed:
        from typesafe_mario.online_duo import wait_installed

        domains = [args.domain_id] + ([args.apex_domain] if args.apex_domain else [])
        for dom, st in wait_installed(client, domains, timeout_s=args.wait_installed).items():
            print(
                f"  {dom}: installed={st.get('installed')} model_status={st.get('model_status')}",
                flush=True,
            )
    feature_names = FEATURE_SETS[args.features]

    print(
        f"frozen-eval {args.domain_id} ({args.features}) over {len(seeds)} seeds: {seeds}",
        flush=True,
    )
    results = []
    for seed in seeds:
        policy = Adapt1Policy(
            client,
            args.domain_id,
            allow_exploration=False,  # exploit the installed policy; no epsilon
            feature_names=feature_names,
            sequential=False,  # eval never sends feedback
            run_id=f"{args.domain_id}-frozeneval",
            temporal_context=args.tcp,
        )
        apex_policy = None
        if args.apex_domain:
            apex_policy = Adapt1Policy(
                client,
                args.apex_domain,
                allow_exploration=False,
                feature_names=FEATURE_SETS[args.apex_features],
                sequential=False,
                run_id=f"{args.apex_domain}-frozeneval",
            )
        level = args.env.replace("SuperMarioBros-", "").replace("-v0", "")
        gif = args.out_dir / f"{level}-seed{seed}.gif"
        stats = record_episode(
            args.env,
            policy,
            gif,
            max_decisions=args.max_decisions,
            seed=seed,
            stall_timeout=args.stall_timeout,
            cadence=args.cadence,
            apex_policy=apex_policy,
            actions=actions,
        )
        counts = dict(policy.status_counts)
        policy.close()
        if apex_policy is not None:
            stats["apex_server"] = dict(apex_policy.status_counts)
            apex_policy.close()
        stats["selection"] = counts
        results.append((seed, stats))
        trace_path = args.out_dir / f"{level}-seed{seed}.trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for rec in stats.get("trace", []):
                fh.write(json.dumps(rec) + "\n")
        print(
            f"  seed {seed:>4}: max_x={stats['max_x']:>4} flag={stats['reached_flag']} "
            f"decisions={stats['decisions']:>3} reward={stats['reward_total']} "
            f"server={counts}"
            + (
                f" apex={stats['apex_choices']} apex_server={stats.get('apex_server')}"
                if args.apex_domain
                else ""
            )
            + f" -> {gif}",
            flush=True,
        )

    reaches = [s["max_x"] for _, s in results]
    flags = sum(1 for _, s in results if s["reached_flag"])
    total_sel: dict[str, int] = {}
    for _, s in results:
        for k, v in s["selection"].items():
            total_sel[k] = total_sel.get(k, 0) + v
    abstained = total_sel.get("abstained", 0)
    decided = sum(total_sel.values()) or 1
    print("\n=== frozen-eval summary ===", flush=True)
    print(
        f"  server selected / abstained: {decided - abstained} / {abstained}  "
        f"({100 * abstained / decided:.0f}% abstained — if ~100%, this measured "
        f"the client's fallback, not the learner)",
        flush=True,
    )
    print(f"  seeds:      {len(results)}", flush=True)
    print(f"  flag_rate:  {flags}/{len(results)} = {flags / len(results):.2f}", flush=True)
    print(f"  mean max_x: {statistics.mean(reaches):.0f}", flush=True)
    print(f"  median:     {statistics.median(reaches):.0f}", flush=True)
    print(f"  min / max:  {min(reaches)} / {max(reaches)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
