#!/usr/bin/env python
"""Warm-start ingest: feed a curated demonstration dataset into a sequential Adapt-1 domain.

⚠️ LIVE + SPENDS RECORDS. 1 ingested transition = 1 Record (deletes do NOT refund). A HARD
cap (--max-records, default 1000) is enforced: the script counts every feedback and STOPS
the instant it would exceed the cap. It also refuses to start if the dataset is larger than
the cap unless you pass --truncate.

Per transition: query(context) -> decision_id (Queries quota — cheap), then feedback
attributing the DEMONSTRATED macro with reward + next_state + terminal (1 Record). Feeds in
file order so episodes stay contiguous and the sequential learner reconstructs them.

Usage (only when you mean to spend Records):
    REI_KEY=... python scripts/ingest_demos.py --dataset artifacts/demos/curated-1000.jsonl \
        --domain-id mario-warm-full --features full --max-records 1000 --recreate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typesafe_mario.adapt1_client import Adapt1Client, Adapt1Error
from typesafe_mario.adapt1_policy import (
    FEATURE_SETS,
    QUESTION,
    Adapt1Policy,
    _parse_query,
    build_domain_config,
    learner_status,
)
from typesafe_mario.macros import ACTION_NAMES, APEX_NAMES, CADENCES, MACRO_NAMES
from typesafe_mario.offline_client import OfflineAdapt1Client
from typesafe_mario.record import record_episode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--domain-id", required=True)
    ap.add_argument(
        "--features",
        choices=tuple(FEATURE_SETS),
        default="full",
        help="the arm; `apex` = the in-air decision's domain (hypotheses = the three "
        "ApexChoice names, features = raw terrain + apex_vx/rise/height)",
    )
    ap.add_argument(
        "--takeoff-under",
        choices=("all", "teacher"),
        default="teacher",
        help="coverage rows (apex_coverage.py) exist per apex choice; feed the takeoff "
        "domain only the branches played under the teacher's apex rule (default) "
        "so its returns are not pooled over all three arcs. Rows without the tag pass.",
    )
    ap.add_argument(
        "--kind",
        choices=("all", "takeoff", "apex"),
        default="all",
        help="rows to feed from an --apex dataset (rows without a kind count as takeoff). "
        "The takeoff domain takes `takeoff`, the apex domain `apex`.",
    )
    ap.add_argument("--relation", default="controls")
    ap.add_argument(
        "--reward-field",
        default="reward",
        help="row field fed as values.reward: `reward` (k-step per-decision return) or "
        "`reward_rate` (fixed frame-horizon return, 2026-09-23 — no long-macro bias)",
    )
    ap.add_argument(
        "--reward-max",
        type=float,
        default=1.0,
        help="declared reward component max (learning.reward.components[].max). Declare it at "
        "the data's scale: ~0.008 for a per-frame rate. Only takes effect with --recreate.",
    )
    ap.add_argument(
        "--model-type",
        choices=("auto", "extra_trees", "mlp_v2", "sequential_q_mlp"),
        default=None,
        help="pin the feedback-policy model family (learning.training.model_type; live since Rei's "
        "2026-09-30 fix). Default: server auto. Only takes effect with --recreate.",
    )
    ap.add_argument(
        "--max-records", type=int, default=1000, help="hard Records cap — never exceeded"
    )
    ap.add_argument("--recreate", action="store_true", help="delete+create the domain first")
    # #17 delayed-credit levers, applied at domain-create (only take effect with --recreate).
    ap.add_argument(
        "--n-step",
        type=int,
        default=5,
        help="sequential n-step return horizon (5 default; 8 reaches further back "
        "to the pipe-jump decision)",
    )
    ap.add_argument(
        "--credit-assignment",
        action="store_true",
        help="enable learning.credit_assignment mode=trace (delayed-outcome "
        "subsystem, off by default; spec-only — verify by the sample counter)",
    )
    ap.add_argument(
        "--skip",
        type=int,
        default=0,
        help="resume: skip the first N rows (already fed). 2026-09-22: a feedback got 404 "
        "'decision not found' mid-ingest; with --skip the run continues without --recreate.",
    )
    ap.add_argument(
        "--truncate",
        action="store_true",
        help="allow a dataset larger than the cap (ingest only the first cap rows)",
    )
    ap.add_argument(
        "--macros",
        default=None,
        help="comma-separated hypotheses to declare (grounded cadence); rows whose policy is not "
        "listed are dropped. Default: every Macro. `--macros` without `jump` = the 8 of #28 — "
        "an undeclared-but-fed macro 422s, a declared-but-unfed one wins on the prior (#22 item 4).",
    )
    ap.add_argument(
        "--cadence",
        choices=CADENCES,
        default="frame",
        help="the cadence the dataset was recorded with; sets the domain's hypotheses "
        "(Action names for frame, Macro names for grounded — findings #21)",
    )
    ap.add_argument(
        "--tcp",
        action="store_true",
        help="enable learning.temporal_context (bounded episode history of the numeric "
        "features; findings #22: separates the two aliased 4-tall pipe tops). Every "
        "query and feedback then carries context.metadata {episode_id, step}.",
    )
    ap.add_argument(
        "--bandit",
        action="store_true",
        help="create a DIRECT-feedback domain (no learning.sequential block). findings #22: "
        "use with rewards that already carry the delayed credit (kstep_returns.py); "
        "avoids the sequential_q_mlp path that output constants on every Mario domain.",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="spend Records from a dirty working tree (the diff is recorded in the provenance file); "
        "by default a live ingest refuses — commit first so the run is reproducible from git",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="OFFLINE: run the whole ingest against OfflineAdapt1Client, which enforces "
        "the measured request contract (policy names, next_state object, boolean "
        "terminal, feature keys) and spends nothing. Run this before every live "
        "ingest — a 422 on the server costs the Records already spent.",
    )
    # Progression recording — FREE (frozen-eval = queries only, no feedback, no Records).
    ap.add_argument(
        "--record-every",
        type=int,
        default=0,
        help="record a frozen-eval GIF every N fed transitions (0=off). Free.",
    )
    ap.add_argument("--record-dir", type=Path, default=Path("artifacts/warm-reel"))
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--eval-seed", type=int, default=777)
    args = ap.parse_args()

    def spent_label(n: int) -> str:
        if args.dry_run:
            return f"dry run: 0 Records spent; a live run would spend {n}"
        return f"Records spent this run: {n}"

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if args.kind != "all":
        rows = [r for r in rows if r.get("kind", "takeoff") == args.kind]
        print(f"{args.kind} rows: {len(rows)}")
    if args.takeoff_under == "teacher":
        before = len(rows)
        rows = [
            r
            for r in rows
            if r.get("kind", "takeoff") != "takeoff"
            or r.get("branch_apex") is None
            or r.get("branch_apex") == r.get("teacher_apex")
        ]
        if len(rows) != before:
            print(
                f"takeoff rows under the teacher's apex rule: {len(rows)} (dropped {before - len(rows)})"
            )
    if len(rows) > args.max_records and not args.truncate:
        raise SystemExit(
            f"dataset has {len(rows)} transitions > cap {args.max_records}. "
            f"Curate it down first, or pass --truncate to feed only the first {args.max_records}."
        )
    rows = rows[: args.max_records]  # belt-and-suspenders hard cap
    if args.skip:
        rows = rows[args.skip :]
        print(f"skipping the first {args.skip} rows (resume)")

    from typesafe_mario.provenance import check_clean, write_provenance

    if not args.dry_run:
        check_clean(allow_dirty=args.allow_dirty, what=f"live ingest into {args.domain_id}")
    if args.dry_run:
        client = OfflineAdapt1Client()
        args.recreate = True  # the stub has no domains; create is free
        args.record_every = 0  # no policy to evaluate offline
        print("DRY RUN — offline contract check only, nothing is sent and no Records are spent")
    else:
        client = Adapt1Client()
    feature_names = FEATURE_SETS[args.features]
    if args.features in ("apex", "apex_v2", "apex_v3"):
        policy_names: tuple[str, ...] = APEX_NAMES
    else:
        policy_names = MACRO_NAMES if args.cadence == "grounded" else ACTION_NAMES
        if args.macros:
            policy_names = tuple(m.strip() for m in args.macros.split(",") if m.strip())
            unknown = [m for m in policy_names if m not in MACRO_NAMES]
            if unknown:
                raise SystemExit(f"--macros: not Macro names: {unknown}")
            before = len(rows)
            rows = [r for r in rows if r["policy"] in policy_names]
            print(
                f"macros {policy_names}: kept {len(rows)} rows (dropped {before - len(rows)} with other macros)"
            )

    def project(values: dict) -> dict:
        """Rows carry the recorded feature superset; send exactly the arm's keys (a missing one
        is a parser-version mismatch and should fail here, not on the server)."""
        return {name: values[name] for name in feature_names}

    def record_checkpoint(tag: str) -> None:
        """Frozen-eval GIF of the current policy — FREE (queries only, no feedback)."""
        if args.record_every <= 0:
            return
        eval_policy = Adapt1Policy(
            client,
            args.domain_id,
            allow_exploration=False,  # exploit current policy; no learning
            feature_names=feature_names,
            sequential=False,  # eval never sends feedback
            run_id=f"{args.domain_id}-eval",
        )
        gif = args.record_dir / f"{tag}.gif"
        stats = record_episode(
            args.env,
            eval_policy,
            gif,
            max_decisions=400,
            seed=args.eval_seed,
            stall_timeout=60,
            cadence=args.cadence,
        )
        eval_policy.close()
        print(
            f"  [eval {tag}] max_x={stats['max_x']} flag={stats['reached_flag']} "
            f"decisions={stats['decisions']} -> {gif}"
        )

    if args.recreate:
        try:
            client.call("DELETE", f"/domains/{args.domain_id}")
        except Adapt1Error:
            pass
        client.create_domain(
            build_domain_config(
                args.domain_id,
                feature_names=feature_names,
                macros=policy_names,
                sequential=not args.bandit,
                n_step=args.n_step,
                credit_assignment=args.credit_assignment,
                temporal_context=args.tcp,
                reward_max=args.reward_max,
                model_type=None if args.model_type == "auto" else args.model_type,
            )
        )
        print(
            f"created {'bandit' if args.bandit else 'sequential'} domain {args.domain_id} ({args.features}, {args.cadence}) "
            f"n_step={args.n_step} credit_assignment={args.credit_assignment} reward_max={args.reward_max} model_type={args.model_type or 'auto'}"
        )
        if not args.dry_run:
            from typesafe_mario.online_duo import wait_ready

            wait_ready(client, (args.domain_id,))  # findings #34: a fresh domain 503s for ~1 min
    missing = [i for i, r in enumerate(rows) if args.reward_field not in r]
    if missing:
        raise SystemExit(
            f"{len(missing)} rows lack the reward field {args.reward_field!r} (first at row {missing[0]})"
        )
    over = sum(1 for r in rows if float(r[args.reward_field]) > args.reward_max)
    print(
        f"reward field {args.reward_field!r}: max in data {max(float(r[args.reward_field]) for r in rows):.5f}, "
        f"declared max {args.reward_max} ({over} rows above it would clip)"
    )

    print(f"ingesting {len(rows)} transitions (hard cap {args.max_records} Records)...")
    record_checkpoint("fed0000")  # cold baseline — first frame of the reel
    fed = 0
    for row in rows:
        if fed >= args.max_records:  # never spend beyond the cap
            print(f"hit Records cap ({args.max_records}) — stopping")
            break
        context = {"values": project(row["context"])}
        if args.tcp:
            # TCP builds the episode history from the query stream: (episode_id, step) in the
            # same context envelope as the observation, monotone within the episode.
            context["metadata"] = {"episode_id": row["episode_id"], "step": row["step"]}
        # A query's decision_id can vanish before the feedback lands ('404 decision not found',
        # seen 2026-09-22 mid-ingest, plausibly an engine checkpoint reload). Re-query once and
        # resend; the first query cost a Query, not a Record.
        for pair_attempt in range(3):
            _s, qr = client.query(
                args.domain_id,
                {"session_id": "ignored", "question": QUESTION, "context": context, "top_k": 1},
            )
            decision_id = _parse_query(qr)["decision_id"]
            try:
                client.feedback(
                    args.domain_id,
                    {
                        "session_id": "ignored",
                        "decision_id": decision_id,
                        "relation": args.relation,
                        "policy": row["policy"],
                        "feedback_kind": "execution",
                        "outcome": row["outcome"],
                        "context": context,
                        "values": {
                            "reward": row[args.reward_field],
                            "next_state": project(row["next_state"]),
                        },
                        "metadata": {
                            "episode_id": row["episode_id"],
                            "step": row["step"],
                            "terminal": row["terminal"],
                        },
                    },
                )
                break
            except Adapt1Error as e:
                if e.status == 404 and pair_attempt < 2:
                    print(
                        f"  row {fed}: 404 decision not found — re-querying ({pair_attempt + 1})",
                        flush=True,
                    )
                    continue
                raise
        fed += 1
        if fed % 100 == 0:
            print(f"  fed {fed}/{len(rows)} ({spent_label(fed)})")
        if args.record_every > 0 and fed % args.record_every == 0:
            record_checkpoint(f"fed{fed:04d}")

    if args.record_every > 0 and fed % args.record_every != 0:
        record_checkpoint(f"fed{fed:04d}-final")

    _s, qr = client.query(
        args.domain_id, {"session_id": "ignored", "question": "status", "top_k": 1}
    )
    st = learner_status(qr)
    print(f"\ningested {fed} transitions ({spent_label(fed)})")
    print(
        f"learner: installed={st['installed']} model_type={st['model_type']} "
        f"episodes={st['episode_count']} validation_skill={st['validation_skill']} "
        f"samples={st['sample_count']}"
    )
    print("(note: validation_skill = Q-prediction accuracy on held-out data, NOT play skill)")
    if not args.dry_run:
        # What was fed, to which domain, under which config — next to a hash of the exact file.
        import datetime as _dt

        run_dir = Path("artifacts/ingests")
        run_dir.mkdir(parents=True, exist_ok=True)
        rec = (
            run_dir
            / f"{args.domain_id}-{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        rec.write_text(
            json.dumps(
                {
                    "domain_id": args.domain_id,
                    "fed": fed,
                    "skip": args.skip,
                    "kind": args.kind,
                    "features": args.features,
                    "macros": list(policy_names),
                    "reward_field": args.reward_field,
                    "reward_max": args.reward_max,
                    "recreate": args.recreate,
                    "learner_after": st,
                    "domain_config": build_domain_config(
                        args.domain_id,
                        feature_names=feature_names,
                        macros=policy_names,
                        sequential=not args.bandit,
                        n_step=args.n_step,
                        credit_assignment=args.credit_assignment,
                        temporal_context=args.tcp,
                        reward_max=args.reward_max,
                        model_type=None if args.model_type in (None, "auto") else args.model_type,
                    ),
                },
                indent=1,
                default=str,
            )
        )
        write_provenance(rec, kind="ingest", inputs={"dataset": args.dataset})
        print(f"provenance: {rec} (+ .provenance.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
