#!/usr/bin/env python
"""Feed saved decision rows into fresh domains, in the original order.

Input: a rows file (one JSON row per fed event) and, in `--config-dir`, the two `POST /domains` bodies
`config-<which>-takeoff.json` and `config-<which>-apex.json`. `runs/base-rebuild/` holds the rows and the configs
of the base player.

Each row becomes one feedback with the row's own context, move (`policy`), reward, outcome and terminal flag, and
the original `episode_id` / `step` in metadata. Two routes:

  decision   query the domain at the row's context (exploit, a Query) for a `decision_id`, then the feedback with it.
             2 credits per row.
  explicit   the feedback alone, with the context and no `decision_id` (docs: sequential-learning, "Replay
             historical transitions with explicit context"; expect `context_source: "request"`). 1 credit per row.

Every response must say `contextual_learning_applied: true` with the expected `context_source`, else the run stops.
The live counter (`learning_state.sample_counts.feedback_policy`) is read before and after and must move by exactly
the rows sent. A 5xx on a feedback is ambiguous (the write may have landed): the feeder reads the live counter
before it resends, so a row is never sent twice. The two domains feed in parallel, one writer each, each in row order.

Credits: a meter counts 1 per call; `--max-credits` stops the run before the call that would pass it.
Resume: `--resume LOG` skips the rows that LOG records as applied (same rows file, same domains).

    python scripts/rebuild_domains.py --account mine --rows runs/base-rebuild/rows-base.jsonl \\
        --config-dir runs/base-rebuild --which base --create --route explicit --max-credits 9500
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario.adapt1_client import Adapt1Client, Adapt1Error  # noqa: E402
from typesafe_mario.adapt1_policy import QUESTION, _parse_query  # noqa: E402
from typesafe_mario.offline_client import OfflineAdapt1Client  # noqa: E402
from typesafe_mario.online_duo import status_query, wait_ready  # noqa: E402
from typesafe_mario.provenance import check_clean, git_state, write_provenance  # noqa: E402

RELATION = "controls"


class CapReached(Exception):
    pass


class Meter:
    """Thread-safe credit count: 1 per call. Transport retries inside the client are not counted."""

    def __init__(self, cap: int | None) -> None:
        self.cap, self.credits, self.by_kind = cap, 0, {}
        self._lock = threading.Lock()

    def take(self, kind: str) -> None:
        with self._lock:
            if self.cap is not None and self.credits + 1 > self.cap:
                raise CapReached(f"credit cap {self.cap} reached")
            self.credits += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


def live_count(client: Any, meter: Meter, domain: str) -> int:
    meter.take("status")
    st = status_query(client, domain)
    return int(st.get("live_sample_count") or 0)


def feedback_body(row: dict, decision_id: str | None, run_id: str) -> dict:
    next_state = row.get("next_state") or row["context"]  # corrector rows carry no next_state: send the context
    body = {
        "session_id": "ignored",
        "relation": RELATION,
        "policy": row["policy"],
        "feedback_kind": "execution",
        "outcome": row["outcome"],
        "context": {"values": row["context"]},
        "values": {"reward": float(row["reward"]), "next_state": next_state},
        "metadata": {
            "run_id": run_id,
            "episode_id": row.get("episode_id") or f"{run_id}-{row['n']:05d}",
            "step": int(row.get("step", 0)),
            "terminal": bool(row["terminal"]),
        },
    }
    if decision_id is not None:
        body["decision_id"] = decision_id
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--account", required=True, help="a label for the account, written to the log")
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--config-dir", type=Path, required=True)
    ap.add_argument("--which", required=True, choices=("base", "v7"),
                    help="the config pair to read: config-<which>-takeoff.json and -apex.json (v7 = a corrector pair)")
    ap.add_argument("--only-95pct-state", action="store_true",
                    help="corrector rows (--which v7) only: keep the rows flagged in_95pct_state")
    ap.add_argument("--route", required=True, choices=("decision", "explicit"))
    ap.add_argument("--create", action="store_true", help="create both domains (refuses if one exists)")
    ap.add_argument("--max-credits", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N rows per domain only (smoke)")
    ap.add_argument("--resume", type=Path, help="a previous feed log on the same rows and domains")
    ap.add_argument("--run-id", default="rebuild-zero2r")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/rebuild-zero2/feeds"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git_at_start = git_state()
    if not args.dry_run:
        check_clean(allow_dirty=args.allow_dirty, what="a rebuild feed")

    configs = {k: json.loads((args.config_dir / f"config-{args.which}-{k}.json").read_text())
               for k in ("takeoff", "apex")}
    domains = {k: c["domain_id"] for k, c in configs.items()}
    rows = [json.loads(x) for x in args.rows.read_text().splitlines() if x.strip()]
    for n, r in enumerate(rows):
        r["n"] = n
    if args.only_95pct_state:
        rows = [r for r in rows if r.get("in_95pct_state")]
    streams = {k: [r for r in rows if r["kind"] == k] for k in domains}
    if args.limit:
        streams = {k: v[: args.limit] for k, v in streams.items()}
    done: set[int] = set()
    if args.resume:
        for x in args.resume.read_text().splitlines():
            e = json.loads(x)
            if e.get("kind") == "row" and e.get("applied") is True:
                done.add(int(e["n"]))
        streams = {k: [r for r in v if r["n"] not in done] for k, v in streams.items()}
    total = sum(len(v) for v in streams.values())
    calls_per_row = 2 if args.route == "decision" else 1
    print(f"{total} rows to feed {({k: len(v) for k, v in streams.items()})} via {args.route} "
          f"(≈ {total * calls_per_row} credits) into {domains}; skipped {len(done)} done")

    client: Any = OfflineAdapt1Client() if args.dry_run else Adapt1Client()
    one_shot: Any = client if args.dry_run else Adapt1Client(max_attempts=1)  # feedback: no blind resend
    meter = Meter(args.max_credits)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / ("dry-run" if args.dry_run else "") / f"feed-{args.which}-{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(entry: dict) -> None:
        with lock, out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    if not args.dry_run:
        write_provenance(out, kind="rebuild-feed", inputs={"rows": str(args.rows)},
                         extra={"args": {k: str(v) for k, v in vars(args).items()}, "domains": domains,
                                "configs": configs, "git_at_start": git_at_start})
    version = None
    if not args.dry_run:
        meter.take("version")
        version = client.version()
    log({"kind": "start", "stamp": stamp, "server_version": version, "domains": domains, "route": args.route,
         "rows": {k: len(v) for k, v in streams.items()}, "account": args.account, "which": args.which})

    if args.create or args.dry_run:
        for k, cfg in configs.items():
            if not args.dry_run:
                try:
                    meter.take("get_domain")
                    client.get_domain(cfg["domain_id"])
                    raise SystemExit(f"{cfg['domain_id']} exists: refusing to overwrite")
                except Adapt1Error as exc:
                    if exc.status != 404:
                        raise
            meter.take("create_domain")
            client.create_domain(cfg)
        if not args.dry_run:
            wait_ready(client, tuple(domains.values()))
        print(f"created {domains}")

    expected_source = "decision" if args.route == "decision" else "request"
    before = {k: (live_count(client, meter, d) if not args.dry_run else 0) for k, d in domains.items()}
    sent = {k: 0 for k in domains}
    log({"kind": "before", "live": before})
    stop = threading.Event()
    errors: list[str] = []

    def feed(kind: str) -> None:
        dom = domains[kind]
        t0 = time.time()
        for i, r in enumerate(streams[kind]):
            if stop.is_set():
                return
            did = None
            if args.route == "decision":
                meter.take("query")
                _s, q = client.query(dom, {"session_id": "ignored", "question": QUESTION,
                                           "context": {"values": r["context"]}, "top_k": 1,
                                           "allow_exploration": False, "selection_mode": "exploit"})
                did = _parse_query(q)["decision_id"]
            body = feedback_body(r, did, args.run_id)
            resp = None
            for attempt in range(6):
                try:
                    meter.take("feedback")
                    _s, resp = one_shot.feedback(dom, body)
                    break
                except Adapt1Error as exc:
                    if exc.status == 404 and did is not None:  # decision vanished: re-issue
                        meter.take("query")
                        _s, q = client.query(dom, {"session_id": "ignored", "question": QUESTION,
                                                   "context": {"values": r["context"]}, "top_k": 1,
                                                   "allow_exploration": False, "selection_mode": "exploit"})
                        did = _parse_query(q)["decision_id"]
                        body = feedback_body(r, did, args.run_id)
                        continue
                    if exc.status not in (0, 502, 503, 504):
                        raise
                    # ambiguous: did it land? the live counter says
                    time.sleep(min(8.0, 0.5 * 2**attempt))
                    now = live_count(client, meter, dom)
                    if now >= before[kind] + sent[kind] + 1:
                        resp = {"credit_assignment": {"contextual_learning_applied": True,
                                                      "context_source": expected_source},
                                "_landed_after_5xx": True}
                        break
            if resp is None:
                errors.append(f"{kind} row {r['n']}: feedback failed after retries")
                stop.set()
                return
            ca = (resp or {}).get("credit_assignment") or {}
            ok = ca.get("contextual_learning_applied") is True and ca.get("context_source") == expected_source
            sent[kind] += 1
            obs = {p.get("policy"): p.get("observations") for p in (resp or {}).get("policy_scores") or []
                   if isinstance(p, dict)}
            log({"kind": "row", "n": r["n"], "domain": dom, "applied": ok,
                 "context_source": ca.get("context_source"), "scope": ca.get("scope"),
                 "observations": obs or None, "landed_after_5xx": bool(resp.get("_landed_after_5xx"))})
            if not ok:
                errors.append(f"{kind} row {r['n']}: not applied as expected: {ca}")
                stop.set()
                return
            if (i + 1) % 200 == 0:
                rate = (i + 1) / max(1e-9, time.time() - t0)
                print(f"  {kind}: {i + 1}/{len(streams[kind])} ({rate:.2f} rows/s) credits={meter.credits}", flush=True)

    def guarded(kind: str) -> None:
        try:
            feed(kind)
        except CapReached as exc:
            errors.append(f"{kind}: {exc}")
            stop.set()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{kind}: {type(exc).__name__}: {exc}")
            stop.set()

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(guarded, list(domains)))

    after = {}
    for k, d in domains.items():
        try:
            after[k] = live_count(client, meter, d) if not args.dry_run else client.samples[d]
        except CapReached:
            after[k] = None
    if not args.dry_run:
        try:
            meter.take("version")
            version_end = client.version()
        except CapReached:
            version_end = None
    else:
        version_end = None
    summary = {"kind": "summary", "sent": sent, "live_before": before, "live_after": after,
               "consumed_ok": {k: (after[k] is not None and after[k] - before[k] == sent[k]) for k in domains},
               "errors": errors, "credits": meter.credits, "credits_by_kind": meter.by_kind,
               "server_version_end": version_end, "log": str(out)}
    log(summary)
    print(json.dumps(summary, indent=1, default=str))
    return 0 if not errors and all(summary["consumed_ok"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
