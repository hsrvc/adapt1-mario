#!/usr/bin/env python
"""Machina acquisition on 1-1: propose, play, observe, repeat. LIVE unless --dry-run.

Each attempt: propose (1 Query) -> execute the whole sequence on the emulator -> observe the
executed prefix with per-command outcomes (1 Record). Journals everything BEFORE the observe.
Resumable: --resume skips create/configure when the domain's trajectory state is `ready`.

    PYTHONPATH=src python scripts/machina_acquire.py --domain-id mario-machina-a --attempts 256 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "src")

from typesafe_mario.machina import (  # noqa: E402
    CONFIG_A,
    GOAL,
    Journal,
    MachinaClient,
    OfflineMachina,
    seed_rows_from_journal,
    set_level_length,
    execute_policy,
    execute_sequence,
    state_row,
)
from typesafe_mario.policy import DiversifiedHeuristicPolicy  # noqa: E402
from typesafe_mario.provenance import check_clean, write_provenance  # noqa: E402
from typesafe_mario.runner import _unwrap_ram, create_mario_env  # noqa: E402
from typesafe_mario.state import MarioStateParser  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain-id", required=True)
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument(
        "--attempts", type=int, default=256, help="hard cap on attempts this run (= Records)"
    )
    ap.add_argument(
        "--start", type=int, default=1, help="first attempt number (for resumed runs / request ids)"
    )
    ap.add_argument(
        "--stop-after-flag", type=int, default=64, help="attempts to continue after the first flag"
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=CONFIG_A["horizon"],
        help="max commands per attempt (8 frames each); fixed at configure time. 1-1 needs ~170 at the "
        "seeded run's pace. Schema max 256.",
    )
    ap.add_argument(
        "--capacity",
        type=int,
        default=CONFIG_A["capacity"],
        help="retention capacity. 2026-09-22: capacity 128 x horizon 256 x (8 state + 4 action) dims -> "
        "422 'trajectory retention budget exceeded'; 3 action dims passed. retained_trajectories "
        "never read above 8 in any run, so 96 costs nothing.",
    )
    ap.add_argument(
        "--profile",
        default="",
        help="comma-separated acquisition options to enable, e.g. progress_guided_mutation",
    )
    ap.add_argument("--journal-dir", type=Path, default=Path("artifacts/machina"))
    ap.add_argument(
        "--seed-teacher",
        type=int,
        default=0,
        help="SEEDED variant: for the first N attempts, execute the frame-cadence heuristic teacher "
        "(eps 0) instead of the proposal and report what ran. Label the run 'refined from a "
        "demonstration'. The teacher reaches ~2473 on 1-1; the learner must finish the rest.",
    )
    ap.add_argument(
        "--seed-sequence",
        type=Path,
        default=None,
        help="SEEDED from Machina's own result (`mach2`): for the first --seed-count attempts, execute the action rows of "
        "the first proposal in this journal (e.g. the z7r frozen 1-1 run) instead of the new proposal, and report what ran",
    )
    ap.add_argument("--seed-count", type=int, default=1)
    ap.add_argument(
        "--level-length",
        type=float,
        default=None,
        help="px where the flag is (goal coordinate and outcome scale). Default 3161 (1-1, every earlier run); 2-1 = 3193",
    )
    ap.add_argument("--dry-run", action="store_true", help="offline stub, spends nothing")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="spend Records from a dirty working tree (the diff is recorded in the provenance file)",
    )
    args = ap.parse_args()
    if not args.dry_run:
        check_clean(allow_dirty=args.allow_dirty, what="a Machina acquisition run")
    if args.level_length is not None:
        set_level_length(args.level_length)
    seed_rows = seed_rows_from_journal(args.seed_sequence) if args.seed_sequence else None

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    journal = Journal(args.journal_dir / f"{args.domain_id}-{ts}.jsonl")
    attempts_path = args.journal_dir / f"{args.domain_id}-attempts.jsonl"
    prov_extra = {"domain_id": args.domain_id, "dry_run": args.dry_run, "env": args.env}
    journal.write("run_args", {k: str(v) for k, v in vars(args).items()})
    write_provenance(
        journal.path, kind="machina-journal", extra={**prov_extra, "status": "running"}
    )
    if args.dry_run:
        m = OfflineMachina(seed=args.start)
        print("DRY RUN — offline stub, nothing sent")
    else:
        from typesafe_mario.adapt1_client import Adapt1Client

        m = MachinaClient(Adapt1Client(), args.domain_id, journal)

    config = dict(CONFIG_A)
    config["horizon"] = args.horizon
    config["capacity"] = args.capacity
    for opt in (o.strip() for o in args.profile.split(",") if o.strip()):
        name, _, val = opt.partition("=")
        config[name] = (val.lower() != "false") if val else True
    # Idempotent setup: create if missing, configure if not configured, continue if ready.
    from typesafe_mario.adapt1_client import Adapt1Error

    if not args.dry_run:
        # POST /domains hung twice for >3 min on fresh Machina domains (2026-09-22) although the
        # domain was created server-side. GET first (fast), create only when 404, with a short
        # timeout client so a hung create retries instead of blocking the run.
        from typesafe_mario.adapt1_client import Adapt1Client as _C

        try:
            _C().get_domain(args.domain_id)
            print("domain exists:", args.domain_id, flush=True)
        except Adapt1Error as e:
            if e.status != 404:
                raise
            try:
                _C(timeout=10.0).create_domain(
                    {
                        "domain_id": args.domain_id,
                        "description": "Mario 1-1 Machina variant A (whole-attempt sequence)",
                    }
                )
                print("created domain", args.domain_id, flush=True)
            except Adapt1Error as e2:
                print(f"create -> {e2.status}; checking again", flush=True)
                _C().get_domain(args.domain_id)
    else:
        m.create_domain("dry")
    st = m.trajectory("state", {})
    print(
        "state:",
        {k: st.get(k) for k in ("status", "retained_trajectories", "pending_decisions", "version")},
        flush=True,
    )
    if st.get("status") == "not_configured":
        resolved = m.trajectory("configure", {"config": config})
        journal.write("resolved_config", resolved)
        print("configured:", json.dumps(resolved.get("config", resolved))[:300], flush=True)
    elif st.get("pending_decisions"):
        raise SystemExit("pending decision on the domain — observe or cancel it first")
    else:
        print("resuming a configured domain", flush=True)

    best = 0
    first_flag = None
    flags = 0
    spent = 0
    for n in range(args.start, args.start + args.attempts):
        env = create_mario_env(args.env, render_mode="rgb_array")
        parser = MarioStateParser(decision_horizon_frames=8)
        _f, info = env.reset(seed=777)
        snap = parser.parse(info, _unwrap_ram(env), previous_action=None)
        body = {
            "state": state_row(snap),
            "goal": GOAL,
            "mode": "explore",
            "request_id": f"{args.domain_id}-attempt-{n:06d}",
            "seed": n,
        }
        journal.write("propose_intent", body)
        prop = m.trajectory("propose", body)
        journal.write("proposal", prop)
        rows = prop["actions"]
        if n - args.start < args.seed_teacher:
            trace = execute_policy(
                env,
                parser,
                snap,
                DiversifiedHeuristicPolicy(epsilon=0.0, rng_seed=n),
                max_commands=len(rows),
            )
        elif seed_rows is not None and n - args.start < args.seed_count:
            trace = execute_sequence(env, parser, snap, seed_rows)
            trace["seeded"] = True
        else:
            trace = execute_sequence(env, parser, snap, rows)
        env.close()
        journal.write("execution", trace)  # persisted before feedback
        obs_body = {
            "decision_id": prop["decision_id"],
            "states": trace["states"],
            "actions": trace["actions"],
            "step_outcomes": trace["step_outcomes"],
            "outcome": float(trace["outcome"]),
            "observation_validity": [True] * len(trace["states"]),
        }
        obs = m.trajectory("observe", obs_body)
        spent += 1
        rec = {
            "attempt": n,
            "seeded": bool(trace.get("seeded")),
            "max_x": trace["max_x"],
            "executed": trace["executed"],
            "proposed": trace["proposed"],
            "flag": trace["flag"],
            "dead": trace["dead"],
            "outcome": round(trace["outcome"], 4),
            "source": prop.get("source"),
            "parent": prop.get("reference_record_id"),
            "observe_status": obs.get("status"),
            "improved_parent": obs.get("improved_parent"),
            "retained": obs.get("retained_trajectories"),
            "version": obs.get("version"),
            "cost": obs.get("cost"),
        }
        with attempts_path.open("a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        best = max(best, trace["max_x"])
        if trace["flag"]:
            flags += 1
            first_flag = first_flag or n
        print(
            f"#{n:04d} max_x={trace['max_x']:4d} best={best:4d} exec={trace['executed']:3d}/{trace['proposed']} "
            f"{'FLAG' if trace['flag'] else 'dead' if trace['dead'] else 'stall'} src={('SEEDED-sequence' if seed_rows is not None else 'SEEDED-teacher') if trace.get('seeded') else prop.get('source')} "
            f"obs={obs.get('status')} improved={obs.get('improved_parent')} retained={obs.get('retained_trajectories')}",
            flush=True,
        )
        if first_flag is not None and n >= first_flag + args.stop_after_flag:
            print(f"stopping: {args.stop_after_flag} attempts after the first flag")
            break
    print(
        f"\nattempts this run: {spent} (Records spent: {0 if args.dry_run else spent}); best max_x {best}; "
        f"flags {flags}; first flag at attempt {first_flag}"
    )
    print(f"attempt log: {attempts_path}")
    summary = {"spent": spent, "best_max_x": best, "flags": flags, "first_flag": first_flag}
    write_provenance(
        journal.path, kind="machina-journal", extra={**prov_extra, "status": "finished", **summary}
    )
    write_provenance(
        attempts_path,
        kind="machina-attempts",
        extra={**prov_extra, **summary},
        inputs={"journal": journal.path},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
