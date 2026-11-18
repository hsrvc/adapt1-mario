#!/usr/bin/env python
"""Correctors on a frozen base: learn the 2-1 corrections where the frozen system fails.

Module: ``typesafe_mario.frontier``. What our runs did with it: ``runs/refine-2-1/``.

Each round:
  1. play the frozen system on 2-1 (base cached, correctors exploit); stop on the flag;
  2. sweep: at each of the last ``--depth`` decisions before it fails, one trial per distinct edit (takeoff, and apex
     when that decision had an apex look); 1 trial = 1 Record (the trial decision's 512-frame credit);
  3. the best edit that gets past the failure point and beats keep_base: repeat it in batches of ``--batch`` until
     the corrector, probed at that exact context, selects a passing edit (gate passed), then a frozen game must get
     past the failure point; cap ``--repeat-cap`` repeats;
  4. retention: the frozen system on 1-1.
Stops on: the flag, no passing edit in a sweep, the repeat cap, the Records cap, or a 1-1 retention loss.

      python scripts/refine_frontier.py --account mine --max-records 120 --rounds 1 --dry-run
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario.adapt1_client import Adapt1Client, Adapt1Error
from typesafe_mario.adapt1_policy import QUESTION, _parse_query, build_domain_config
from typesafe_mario.frontier import (
    CachedBase,
    EpochCache,
    FrontierConfig,
    LockedBudget,
    Trial,
    TrialStats,
    corrector_context,
    first_divergence,
    frontier_decisions,
    play,
    trials_at,
)
from typesafe_mario.macros import APEX_NAMES
from typesafe_mario.offline_client import OfflineAdapt1Client
from typesafe_mario.online_duo import status_query, wait_installed
from typesafe_mario.credit_meter import CreditMeter
from typesafe_mario.provenance import check_clean, git_state, write_provenance
from typesafe_mario.refinement import (
    BASE_APEX_FEATURES,
    BASE_MACROS,
    BASE_TAKEOFF_FEATURES,
    GuardedClient,
    Pair,
    base_fingerprint,
    fingerprint_changed,
    keep_dominance_pick,
    refiner_domain_config,
)

ENV_1_1, ENV_2_1 = "SuperMarioBros-1-1-v0", "SuperMarioBros-2-1-v0"
PASS_MARGIN = 48  # px past the frozen game's best x: the trial got past the failure point


def _game_line(g: dict) -> str:
    return f"{g['env'][15:18]} end={g['end']} max_x={g['max_x']} decisions={g['decisions']}"


class CreditCappedBudget(LockedBudget):
    """``LockedBudget`` that also reads 0 remaining once the credit meter reaches ``max_credits`` (0 = off)."""

    def __init__(self, max_records: int, meter: CreditMeter, max_credits: int) -> None:
        super().__init__(max_records)
        self.meter, self.max_credits = meter, int(max_credits)
        self.halt: str | None = None  # a guard that cannot be met ends the run: no more rows of any kind

    def _over(self) -> bool:
        return self.halt is not None or (self.max_credits > 0 and self.meter.credits >= self.max_credits)

    @property
    def remaining(self) -> int:
        return 0 if self._over() else super().remaining

    def take(self) -> bool:
        return False if self._over() else super().take()


class CappedRewards:
    """Sends min(reward, cap) to the corrector domains (2026-11-17). The server's interval is count-based on the
    reward divided by the declared maximum, so a low maximum makes a small local gain a large share of the scale.
    A reward above the declared maximum is never sent."""

    def __init__(self, client, domain_ids, cap: float, override_cost: float = 0.0) -> None:
        self._client, self._ids, self.cap = client, frozenset(domain_ids), float(cap)
        self.cost = float(override_cost)
        self.capped = 0

    def sent(self, reward: float, policy: str | None) -> float:
        """The reward the server receives: capped, and an override's reward is reduced by ``override_cost`` so that
        where keep and an override both reach the cap, keep stays ahead (2026-11-17: with a cap, an override at 1.0
        read above keep on 30 of 31 in-air decisions of 1-1; only the interval held it)."""
        r = min(float(reward), self.cap)
        return r if policy == "keep_base" else r * (1.0 - self.cost)

    def __getattr__(self, name):
        return getattr(self._client, name)

    def feedback(self, domain_id, body):
        v = (body or {}).get("values") or {}
        if domain_id in self._ids and isinstance(v.get("reward"), (int, float)):
            r = self.sent(v["reward"], body.get("policy"))
            if r != v["reward"]:
                self.capped += v["reward"] > self.cap
                body = {**body, "values": {**v, "reward": r}}
        return self._client.feedback(domain_id, body)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--account", required=True, help="a label for the account, written to the log")
    ap.add_argument("--base-takeoff-domain", default="mario-zero2-takeoff")
    ap.add_argument("--base-apex-domain", default="mario-zero2-apex")
    ap.add_argument("--refine-takeoff-domain", default="mario-zero2-takeoff-refine-2-1-v4")
    ap.add_argument("--refine-apex-domain", default="mario-zero2-apex-refine-2-1-v4")
    ap.add_argument(
        "--max-records", type=int, default=120, help="hard cap on feedbacks (1 feedback = 1 Record)"
    )
    ap.add_argument("--rounds", type=int, default=1, help="blockers to attempt")
    ap.add_argument(
        "--reward-max", type=float, default=0.0076,
        help="declared reward maximum of the correctors; rewards above it are sent as the maximum (same on every leg)",
    )
    ap.add_argument(
        "--override-cost", type=float, default=0.0,
        help="fraction taken off every override's reward (keep_base rows are not reduced); same on every leg",
    )
    ap.add_argument("--depth", type=int, default=3, help="decisions before the failure to sweep")
    ap.add_argument("--batch", type=int, default=4, help="repeats of the leading edit per batch")
    ap.add_argument(
        "--repeat-cap",
        type=int,
        default=96,
        help="Records per blocker for the gate's evidence before giving up",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--horizon", type=int, default=512, help="credit window in frames"
    )
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument(
        "--no-retention",
        action="store_true",
        help="skip the 1-1 frozen game after each accepted fix",
    )
    ap.add_argument("--keep-going-on-retention-loss", action="store_true")
    ap.add_argument(
        "--max-credits",
        type=int,
        default=0,
        help="hard cap on ALL calls (1 credit each); the run stops taking Records once the meter reaches it; "
        "0 = off",
    )
    ap.add_argument(
        "--archive-records",
        type=int,
        default=0,
        help="multi-move blockers (the springboard): when no single edit passes, search sequences of takeoff edits "
        "from the frontier decisions (breadth first, one new state cell per kept node, each expansion = 1 Record "
        "credited at its last edit); 0 = off",
    )
    ap.add_argument("--archive-depth", type=int, default=6)
    ap.add_argument(
        "--esc-starts",
        type=int,
        default=0,
        help="escalation: start sequences from the last N decisions that made progress (0 = the sweep's)",
    )
    ap.add_argument(
        "--esc-beam", type=int, default=0, help="escalation: max nodes kept per depth (0 = all)"
    )
    ap.add_argument(
        "--create-refiners",
        action="store_true",
        help="create the corrector pair (refuses if it exists): gate on, the reward maximum of --reward-max, trees pinned, "
        "--model-max-weight / --min-context-observations",
    )
    ap.add_argument("--model-max-weight", type=float, default=0.2)
    ap.add_argument(
        "--abstain-on-ood",
        action="store_true",
        help="--create-refiners: learning.policy.abstain_on_ood",
    )
    ap.add_argument(
        "--keep-fixes-from",
        nargs="*",
        default=[],
        help="earlier frontier logs: their accepted corrections are kept (probed every gate batch, re-taught when "
        "the corrector stops selecting them)",
    )
    ap.add_argument(
        "--path", default=None, help="round 1: comma-separated moves to adopt (e.g. left,left,jump)"
    )
    ap.add_argument("--path-start", type=int, default=0, help="decision index where --path starts")
    ap.add_argument(
        "--path-from",
        default=None,
        help="frontier log whose first frozen 2-1 game gives the prefix",
    )
    ap.add_argument(
        "--keep-anchor",
        type=int,
        default=0,
        help="each round, credit keep_base (takeoff, and apex when it had a look) at every K-th decision of the frozen "
        "game before the frontier where the frozen system keeps the base: honest evidence that the base is fine there "
        "(the frontier alone tries keep_base only where it fails); 0 = off",
    )
    ap.add_argument(
        "--anchor-from",
        default=None,
        help="a frontier log: before round 1, anchor keep_base along its first frozen 2-1 game (with --keep-anchor)",
    )
    ap.add_argument(
        "--anchor-1-1",
        type=int,
        default=0,
        help="credit keep_base at every K-th decision of the frozen 1-1 game where the frozen system keeps the base "
        "(before round 1, and again whenever override rows outnumber keep rows): the corrector must see that the base "
        "is right on the level it already clears; 0 = off. Required with --keep-dominance",
    )
    ap.add_argument(
        "--max-override-ratio",
        type=float,
        default=1.0,
        help="override rows sent by this run may not exceed this many times its keep_base rows; past it the run "
        "anchors keep_base again before the next batch (2026-11-16: 194 override rows against 38 keep rows taught "
        "'override beats keep' at every decision and broke 1-1)",
    )
    ap.add_argument(
        "--leak-patience",
        type=int,
        default=6,
        help="when the deployed frozen game departs from the tested leader trial, teach the tested move at the first "
        "departure and replay, at most this many times per acceptance check",
    )
    ap.add_argument(
        "--keep-dominance",
        action="store_true",
        help="trust region = override only when proven better than keep_base: when the server abstains ONLY for "
        "overlapping intervals, play the override whose lower bound beats keep_base's upper bound (frozen play, "
        "learning continuations and probes alike)",
    )
    ap.add_argument(
        "--leader", default=None, help="round 1: start with this passing candidate (j:kind:edit)"
    )
    ap.add_argument(
        "--resume-sweep",
        default=None,
        help="an earlier frontier log: reuse its last sweep when this run's first failure point is the same",
    )
    ap.add_argument(
        "--fight-patience",
        type=int,
        default=3,
        help="switch to the next passing candidate after this many batches with a kept fix lost",
    )
    ap.add_argument("--lead-repeats", type=int, default=3, help="leader trials per gate batch")
    ap.add_argument(
        "--block-repeats", type=int, default=2, help="trials per overlapping arm per gate batch"
    )
    ap.add_argument("--min-context-observations", type=int, default=4)
    ap.add_argument(
        "--pass-margin",
        type=int,
        default=PASS_MARGIN,
        help="px past the failure x that counts as passing",
    )
    ap.add_argument("--path-repeat-cap", type=int, default=12, help="repeats of a found sequence")
    ap.add_argument("--out-dir", default="artifacts/refine-2-1/frontier")
    ap.add_argument("--run-id", default="refine-frontier")
    ap.add_argument(
        "--dry-run", action="store_true", help="offline contract client: 0 Records, 0 Queries"
    )
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.keep_dominance and (args.keep_anchor < 1 or args.anchor_1_1 < 1):
        # measured twice: keep-dominance is safe only on a corrector that holds keep_base
        # evidence where the base is right. The same flags without anchor rows broke 1-1.
        raise SystemExit("--keep-dominance needs --keep-anchor >= 1 and --anchor-1-1 >= 1")
    git_at_start = git_state()
    if not args.dry_run:
        check_clean(allow_dirty=args.allow_dirty, what="refine frontier")
    base = Pair(args.base_takeoff_domain, args.base_apex_domain)
    refiner = Pair(args.refine_takeoff_domain, args.refine_apex_domain)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / ("dry-run" if args.dry_run else "") / f"run-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "frontier.jsonl"

    if args.dry_run:
        inner = OfflineAdapt1Client()
        inner.create_domain(
            build_domain_config(
                base.takeoff,
                feature_names=BASE_TAKEOFF_FEATURES,
                macros=[m.value for m in BASE_MACROS],
                max_samples=16384,
                model_type="extra_trees",
            )
        )
        inner.create_domain(
            build_domain_config(
                base.apex,
                feature_names=BASE_APEX_FEATURES,
                macros=list(APEX_NAMES),
                max_samples=16384,
                model_type="extra_trees",
            )
        )
        for dom, kind in ((refiner.takeoff, "takeoff"), (refiner.apex, "apex")):
            inner.create_domain(
                refiner_domain_config(dom, kind, credible_dominance=True, reward_max=args.reward_max)
            )
        print("DRY RUN — offline contract client: 0 Records, 0 Queries")
    else:
        inner = Adapt1Client()
    # 2026-11-16: one credit pool: queries cost as much as feedbacks, so cap the CALLS, not only the Records
    meter = CreditMeter(inner)
    inner = meter
    inner = CappedRewards(inner, (refiner.takeoff, refiner.apex), args.reward_max, args.override_cost)
    guarded = GuardedClient(inner, {base.takeoff, base.apex})
    cache_path = (
        Path(args.out_dir) / f"base-cache-{base.takeoff}.jsonl" if not args.dry_run else None
    )
    live = CachedBase(guarded, (base.takeoff, base.apex), path=cache_path)
    cached = EpochCache(live, (refiner.takeoff, refiner.apex))
    clients = {"live": live, "cached": cached}
    cfg = FrontierConfig(
        base=base,
        refiner=refiner,
        env_id=ENV_2_1,
        seed=args.seed,
        horizon=args.horizon,
        run_id=args.run_id,
        keep_dominance=args.keep_dominance,
    )
    budget = CreditCappedBudget(args.max_records, meter, args.max_credits)
    stats = TrialStats()
    sent = {refiner.takeoff: 0, refiner.apex: 0}
    rows = {"keep": 0, "override": 0}  # feedback rows of this run by kind (the balance guard)
    rows_dom: dict = {}  # the same per corrector domain: each corrector needs its own balance

    def add_row(domain: str | None, policy: str | None) -> None:
        kind = "keep" if policy == "keep_base" else "override"
        rows[kind] += 1
        rows_dom.setdefault(domain or "?", {"keep": 0, "override": 0})[kind] += 1

    def unbalanced() -> bool:
        return any(v["override"] > args.max_override_ratio * v["keep"] for v in [rows, *rows_dom.values()])
    t0 = time.time()

    def say(msg: str) -> None:
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    def write(entry: dict) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    if args.create_refiners and not args.dry_run:
        from typesafe_mario.online_duo import wait_ready

        for dom, kind in ((refiner.takeoff, "takeoff"), (refiner.apex, "apex")):
            try:
                inner.call("GET", f"/domains/{dom}")
                raise SystemExit(f"{dom} exists: refusing to overwrite a corrector")
            except Adapt1Error as exc:
                if exc.status != 404:
                    raise
            dcfg = refiner_domain_config(
                dom,
                kind,
                credible_dominance=True,
                reward_max=args.reward_max,
                max_samples=16384,
                model_max_weight=args.model_max_weight,
                min_context_observations=args.min_context_observations,
            )
            if args.abstain_on_ood:
                dcfg["learning"]["policy"]["abstain_on_ood"] = True
            guarded.create_domain(dcfg)
        wait_ready(guarded, (refiner.takeoff, refiner.apex))
        say(f"created {refiner.takeoff} / {refiner.apex}")
    if not args.dry_run:
        for dom in (
            refiner.takeoff,
            refiner.apex,
        ):  # the corrector must hold every Record it will get
            _s, g = inner.call("GET", f"/domains/{dom}")
            cap = int(((g.get("learning") or {}).get("context") or {}).get("max_samples") or 0)
            held = status_query(inner, dom).get("live_sample_count") or 0
            if cap and held + args.max_records > cap:
                raise SystemExit(
                    f"{dom}: {held} rows + {args.max_records} Records > max_samples {cap}"
                )
            say(f"{dom}: {held} rows held, max_samples {cap}")

            def declared_max(o):  # the reward component: a dict with "goal" and "max"
                if isinstance(o, dict):
                    if "goal" in o and "max" in o:
                        return o["max"]
                    o = list(o.values())
                if isinstance(o, list):
                    for v in o:
                        m = declared_max(v)
                        if m is not None:
                            return m
                return None

            dmax = declared_max(g)
            if dmax is None:
                raise SystemExit(f"{dom}: no declared reward maximum found in the domain config")
            if abs(float(dmax) - args.reward_max) > 1e-12:
                raise SystemExit(f"{dom}: declared reward max {dmax} != --reward-max {args.reward_max}")
    fp0 = base_fingerprint(guarded, (base.takeoff, base.apex))
    if cache_path is not None:
        # The base cache holds answers of ONE base model. A server refit between legs would mix two bases.
        fp_path = cache_path.with_suffix(".fingerprint.json")
        if fp_path.exists():
            moved = fingerprint_changed(json.loads(fp_path.read_text()), fp0)
            if moved:
                raise SystemExit(f"the base changed since {cache_path.name} was written: {moved}")
        elif cache_path.exists() and cache_path.stat().st_size > 0:
            raise SystemExit(f"{cache_path.name} has no fingerprint file: move it away or record the base it holds")
        else:
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            fp_path.write_text(json.dumps(fp0, default=str))
    prov = {
        "account": args.account,
        "config": vars(args),
        "git_at_start": git_at_start,
        "invariants": {
            "demonstrations_used": 0,
            "absolute_x_in_learning_context": False,
            "base_feedback": "refused by GuardedClient",
            "credit": f"frame_horizon {args.horizon} frames of the trial decision only",
            "exploration": "one trial per game at the frozen system's failure point; every tried "
            "edit is sent with its measured credit",
        },
    }
    write({"kind": "start", "base_fingerprint": fp0, **prov})
    write_provenance(log_path, kind="refine-frontier-log", extra={**prov, "status": "running"})

    counter = [0]
    current = {"game": None}

    def prefix_ctx(t: Trial) -> dict:
        return current["game"]["steps"][t.j]["context"]

    class NotConsumed(RuntimeError):
        pass

    def check_applied(res: dict) -> None:
        """Every credited feedback must come back applied (2026-10-04: a silent no-send was counted as a Record)."""
        if args.dry_run:
            return
        bad = [r for r in res["feedback"] if r.get("contextual_learning_applied") is not True]
        if bad:
            raise NotConsumed(f"{res.get('trial')}: {len(bad)} feedback(s) not applied: {bad[0]}")

    count_lock = threading.Lock()

    def count_rows(res: dict) -> None:
        """Every feedback row of the run, by domain and by kind (the balance guard reads ``rows``)."""
        with count_lock:
            for rec in res["feedback"]:
                sent[rec["domain"]] = sent.get(rec["domain"], 0) + 1
                add_row(rec.get("domain"), rec.get("policy"))
                if isinstance(rec.get("reward"), (int, float)):
                    rec["reward_sent"] = inner.sent(rec["reward"], rec.get("policy"))  # what the server received

    def next_id() -> int:
        with count_lock:
            counter[0] += 1
            return counter[0]

    def run_trial(
        t: Trial, prefix: list, tag: str, expect: dict | None = None, env_id: str | None = None
    ) -> dict | None:
        if budget.remaining == 0:  # the cap is reached: a trial would pay its queries and then drop its row
            return None
        n = next_id()
        res = play(
            clients,
            cfg,
            prefix=prefix,
            trial=t,
            budget=budget,
            episode_id=f"{args.run_id}-{stamp}-{tag}-{n:04d}",
            env_id=env_id,
        )
        count_rows(res)
        stats.add(res)
        live0 = next((s for s in res["steps"] if not s.get("replayed")), None)
        res["context_match"] = bool(live0) and live0["context"] == (
            expect["context"] if expect is not None else prefix_ctx(t)
        )
        if not res["context_match"]:
            say(
                f"!! {t.label()}: the trial's context at j differs from the frozen game's (replay drift)"
            )
        write(
            {
                "kind": "trial",
                "tag": tag,
                **{k: v for k, v in res.items() if k != "steps"},
                "steps_live": [s for s in res["steps"] if not s.get("replayed")],
            }
        )
        check_applied(res)
        return res

    def pmap_warm(fn, items, key):
        """Identical trials in one batch all miss the corrector cache when they start together. Play one of each
        distinct trial first, then the duplicates (their continuation queries are then cache hits)."""
        seen, first, rest = set(), [], []
        for it in items:
            k = key(it)
            (rest if k in seen else first).append(it)
            seen.add(k)
        return pmap(fn, first) + pmap(fn, rest)

    def pmap(fn, items):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return list(ex.map(fn, items))

    def frozen(env_id: str, tag: str) -> dict:
        wait_installed(guarded, (refiner.takeoff, refiner.apex), log=say)
        cached.bump()
        g = play(clients, cfg, env_id=env_id)
        write({"kind": "frozen", "tag": tag, **g})
        say(
            f"frozen {tag}: {_game_line(g)} overrides="
            f"{[(s['i'], s['x'], s['base'], s['executed']) for s in g['steps'] if s['executed'] != s['base']]}"
            f" apex_overrides={sum(1 for s in g['steps'] if s['apex_executed'] and s['apex_executed'] != s['apex_base'])}"
        )
        return g

    def probe(step: dict, kind: str) -> dict:
        ctx = corrector_context(step, kind)
        dom = refiner.takeoff if kind == "takeoff" else refiner.apex
        _s, r = guarded.query(
            dom,
            {
                "session_id": "ignored",
                "question": QUESTION,
                "context": {"values": ctx},
                "top_k": 3,
                "allow_exploration": False,
                "selection_mode": "exploit",
            },
        )
        p = _parse_query(r)
        diag = p["diagnostics"] or {}
        status, policy, via = p["status"], p["policy"], None
        if args.keep_dominance:
            pick = keep_dominance_pick(
                {k: (v or {}).get("credible_interval") for k, v in diag.items()},
                status,
                p["reason"],
            )
            if pick is not None:
                status, policy, via = "selected", pick, "keep_dominance"
        return {
            "status": status,
            "reason": p["reason"],
            "policy": policy,
            "effective": policy if status == "selected" and policy else "keep_base",
            "via": via,
            "values": {k: (v or {}).get("selection_expected_reward") for k, v in diag.items()},
            "diagnostics": diag,
        }

    game_1_1: dict = {"game": None}

    def retention(rnd: int) -> bool:
        if args.no_retention:
            return True
        a = frozen(ENV_1_1, f"r{rnd:02d}-1-1")
        ref = game_1_1["game"]
        if not a["flag"] and ref is not None and ref.get("flag"):
            # the last clearing 1-1 game is the tested game: teach its move where the new game first departs
            a = repair_leaks(a, ref["steps"], int(ref["max_x"]), rnd, ENV_1_1)
        if a["flag"]:
            game_1_1["game"] = a  # the anchors follow the game the frozen system plays now
            keep_margin(a, rnd)
            return True
        say(f"!! RETENTION: 1-1 frozen {a['max_x']} (no flag) after round {rnd}")
        return args.keep_going_on_retention_loss

    def keep_margin(g: dict, rnd: int) -> None:
        """How close keep-dominance is to firing on 1-1 (diagnostic, 1 query per takeoff decision): at each decision
        the gap = keep_base's upper bound minus the best override's lower bound. An override fires when gap < 0."""
        if not args.keep_dominance or args.dry_run:
            return
        for kind in ("takeoff", "apex"):
            gaps, mean_above = [], 0
            for st in g["steps"]:
                if kind == "apex" and not st.get("apex_executed"):
                    continue
                d = probe(st, kind).get("diagnostics") or {}
                ivs = {k: (v or {}).get("credible_interval") for k, v in d.items()}
                keep = ivs.get("keep_base")
                lows = [float(iv[0]) for k, iv in ivs.items() if k != "keep_base" and iv]
                if keep and lows:
                    gaps.append((round(float(keep[1]) - max(lows), 4), st["i"], st["x"]))
                ev = {k: (v or {}).get("selection_expected_reward") for k, v in d.items()}
                if ev.get("keep_base") is not None and any(
                    v is not None and v > ev["keep_base"] for k, v in ev.items() if k != "keep_base"
                ):
                    mean_above += 1  # an override's mean is above keep's here: only the interval overlap holds it
            gaps.sort()
            write({"kind": "keep_margin", "round": rnd, "env": ENV_1_1, "corrector": kind, "gaps": gaps,
                   "override_mean_above_keep": mean_above})
            if gaps:
                say(
                    f"   1-1 keep margin ({kind}) after round {rnd}: smallest {gaps[:3]} (an override fires below 0); "
                    f"an override's mean is above keep's at {mean_above} of {len(gaps)} decisions"
                )

    def archive_search(game: dict, cands: list, best_x: int, rnd: int) -> list | None:
        """Escalation when no single edit passes: breadth first over sequences of takeoff edits from the last
        ``--esc-starts`` decisions that made progress. Every expansion is one trial game, credited at its newest edit
        (the earlier edits were credited when their own nodes were expanded), so the corrector experiences every try.
        A node survives if its newest edit reached a state cell (x // 8, y // 16) not seen before; at most
        ``--esc-beam`` nodes per depth (half furthest right, half random). A passing sequence is then adopted with
        ``learn_path``, crediting its whole game. Same rule at every failure point; nothing level-specific."""
        import random as _random

        from typesafe_mario.refinement import takeoff_edits

        edits = [e.value for e in takeoff_edits()]
        cap = min(args.archive_records, budget.remaining)
        spent0 = budget.spent
        starts = frontier_decisions(game, args.esc_starts) if args.esc_starts else cands
        level = [(j, []) for j in starts]
        seen: set = set()
        rng = _random.Random(args.seed + rnd)
        for depth in range(1, args.archive_depth + 1):
            jobs = [
                (j, plan, Trial(j + len(plan), "takeoff", e)) for j, plan in level for e in edits
            ]
            room = cap - (budget.spent - spent0)
            if room <= 0 or not jobs:
                break
            jobs = jobs[:room]
            say(f"archive depth {depth}: {len(level)} nodes, {len(jobs)} trials (room {room})")
            rebalance(game, rnd)

            def one(job, g=game, depth=depth):
                j, plan, t = job
                counter[0] += 1
                res = play(
                    clients,
                    cfg,
                    prefix=g["steps"][:j],
                    plan=plan,
                    trial=t,
                    budget=budget,
                    episode_id=f"{args.run_id}-{stamp}-r{rnd:02d}-archive-{counter[0]:04d}",
                )
                count_rows(res)
                write(
                    {
                        "kind": "archive",
                        "applied_check": check_applied(res),
                        "round": rnd,
                        "depth": depth,
                        "start": j,
                        "plan": [x.label() for x in plan],
                        **{k: v for k, v in res.items() if k != "steps"},
                        "steps_live": [s for s in res["steps"] if not s.get("replayed")],
                    }
                )
                return job, res

            nxt = []
            found = None
            for (j, plan, t), res in pmap(one, jobs):
                if res["flag"] or res["max_x"] > best_x + args.pass_margin:
                    if found is None or (res["credit"] or 0) > (found[1]["credit"] or 0):
                        found = ([*plan, t], res)
                    continue
                st = next((s for s in res["steps"] if s.get("i") == t.j), None)
                if st is None or res["end"] == "dead" and st["x_after"] == st["x"]:
                    continue
                cell = (
                    st["x_after"] // 8,
                    st["y_after"] // 16,
                )  # one cell per state, whatever the start
                if cell not in seen:
                    seen.add(cell)
                    nxt.append((j, [*plan, t], int(st["x_after"])))
            if found is not None:
                path, res = found
                say(
                    f"archive: FOUND {[x.label() for x in path]} reach={res['max_x']} credit={res['credit']}"
                )
                write(
                    {
                        "kind": "archive_found",
                        "round": rnd,
                        "path": [x.label() for x in path],
                        "reach": res["max_x"],
                        "records": budget.spent - spent0,
                    }
                )
                return path
            if args.esc_beam and len(nxt) > args.esc_beam:
                nxt.sort(key=lambda n: -n[2])
                half = args.esc_beam // 2
                nxt = nxt[:half] + rng.sample(nxt[half:], args.esc_beam - half)
            level = [(j, plan) for j, plan, _x in nxt]
        say(
            f"archive: nothing passes x={best_x} within depth {args.archive_depth} / {budget.spent - spent0} Records"
        )
        return None

    def learn_path(game: dict, path: list, best_x: int, rnd: int) -> bool:
        """Adopt a found sequence. Each batch: the sequence ``--lead-repeats`` times, crediting every edit of it (each
        by its own 512-frame window of the game as played), plus, at every step the corrector does not yet select,
        ``--block-repeats`` trials of each arm whose interval still overlaps the sequence's edit there (the earlier
        steps forced, only that arm credited). Then probe every step; when all select the sequence, a frozen game
        must pass."""
        j0 = path[0].j
        spent0 = budget.spent
        blocking: dict[int, list[str]] = {}
        live_steps: dict[int, dict] = {}
        rounds_done = 0

        def run(job, g=game):
            kind, k, arm = job
            counter[0] += 1
            eid = f"{args.run_id}-{stamp}-r{rnd:02d}-path-{counter[0]:04d}"
            if kind == "path":
                return job, play(
                    clients,
                    cfg,
                    prefix=g["steps"][:j0],
                    plan=path[:-1],
                    trial=path[-1],
                    credit_plan=True,
                    budget=budget,
                    episode_id=eid,
                )
            return job, play(
                clients,
                cfg,
                prefix=g["steps"][:j0],
                plan=path[:k],
                trial=Trial(path[k].j, path[k].kind, arm),
                budget=budget,
                episode_id=eid,
            )

        while budget.spent - spent0 < args.path_repeat_cap * len(path) and budget.remaining >= len(
            path
        ):
            rebalance(game, rnd)
            jobs = [("path", -1, None)] * args.lead_repeats
            for k, arms in blocking.items():
                jobs += [("arm", k, a) for a in arms for _ in range(args.block_repeats)]
            for (kind, k, arm), res in pmap(run, jobs):
                count_rows(res)
                write(
                    {
                        "kind": "path" if kind == "path" else "path_arm",
                        "applied_check": check_applied(res),
                        "round": rnd,
                        "path": [x.label() for x in path],
                        "step": k,
                        "arm": arm,
                        **{kk: v for kk, v in res.items() if kk != "steps"},
                        "steps_live": [st for st in res["steps"] if not st.get("replayed")],
                    }
                )
                if kind == "path":
                    live_steps = {st["i"]: st for st in res["steps"] if not st.get("replayed")}
            rounds_done += 1
            wait_installed(guarded, (refiner.takeoff, refiner.apex), log=say)
            cached.bump()
            picks, blocking = [], {}
            for k, t in enumerate(path):
                pr = probe(live_steps[t.j], t.kind)
                diag = pr.get("diagnostics") or {}

                def ci(arm: str, d=diag) -> tuple[float, float]:
                    c = (d.get(arm) or {}).get("credible_interval") or [0.0, 1.0]
                    return float(c[0]), float(c[1])

                picks.append((t.edit, pr["status"], pr["policy"], ci(t.edit)))
                if pr["effective"] != t.edit:
                    lo = ci(t.edit)[0]
                    blocking[k] = [a for a in diag if a != t.edit and ci(a)[1] >= lo]
            write(
                {
                    "kind": "path_probe",
                    "round": rnd,
                    "batches": rounds_done,
                    "picks": picks,
                    "blocking": {str(k): v for k, v in blocking.items()},
                    "records": budget.spent - spent0,
                }
            )
            ok = len(path) - len(blocking)
            say(
                f"   path batch {rounds_done}: {ok}/{len(path)} selected; blocking "
                + "; ".join(f"step {k} ({path[k].edit}): {len(v)}" for k, v in blocking.items())
                + f" [{budget.spent - spent0} Records]"
            )
            if not blocking:
                g2 = frozen(ENV_2_1, f"r{rnd:02d}-path{rounds_done}")
                if g2["flag"] or g2["max_x"] > best_x + args.pass_margin:
                    current["game"] = g2
                    return True
                say(f"   every step selects the sequence but the frozen game reached {g2['max_x']}")
                if g2["max_x"] < best_x - args.pass_margin:
                    current["game"] = g2
                    return True
        say(f"round {rnd}: the sequence was not adopted within {budget.spent - spent0} Records")
        return False

    def load_fixes(paths: list) -> list[dict]:
        out = []
        for pth in paths:
            last_game, sweep_game = None, {}
            for line in Path(pth).read_text().splitlines():
                e = json.loads(line)
                if e["kind"] == "frozen" and e.get("env", "").endswith("2-1-v0"):
                    last_game = e
                elif e["kind"] == "sweep":
                    sweep_game[e["round"]] = last_game
                elif e["kind"] == "round" and e.get("accepted"):
                    g = sweep_game[e["round"]]
                    j, kind, edit = e["leader"].split(":")
                    out.append(
                        {
                            "trial": Trial(int(j), kind, edit),
                            "prefix": g["steps"][: int(j)],
                            "step": g["steps"][int(j)],
                            "from": f"{pth}#r{e['round']}",
                        }
                    )
        dedup: dict[str, dict] = {}
        for fx in out:  # the same fix accepted again in a later run: keep the latest recorded game
            dedup[fx["trial"].label()] = fx
        return list(dedup.values())

    fixes = load_fixes(args.keep_fixes_from)
    for pth in args.keep_fixes_from:  # a resumed leg continues the balance of the earlier legs
        for line in Path(pth).read_text().splitlines():
            e = json.loads(line)
            for rec in e.get("feedback") or []:
                if isinstance(rec, dict) and rec.get("policy"):
                    add_row(rec.get("domain"), rec["policy"])
    if args.keep_fixes_from:
        say(f"rows of the earlier legs: {rows}")
    for fx in fixes:
        say(f"keeping fix {fx['trial'].label()} at x={fx['step']['x']} (from {fx['from']})")

    def probe_fix(fx: dict) -> tuple[bool, list, dict]:
        """Is a kept fix still selected? Returns (selected, blocking arms, probe)."""
        t = fx["trial"]
        pr = probe(fx["step"], t.kind)
        d = pr.get("diagnostics") or {}

        def ci(arm: str) -> tuple[float, float]:
            c = (d.get(arm) or {}).get("credible_interval") or [0.0, 1.0]
            return float(c[0]), float(c[1])

        ok = pr["effective"] == t.edit
        lo = ci(t.edit)[0]
        return ok, ([] if ok else [a for a in d if a != t.edit and ci(a)[1] >= lo]), pr

    def fix_jobs(fx: dict, blocking: list) -> list:
        t = fx["trial"]
        arms = [a for a in blocking if a == "keep_base" or not args.keep_dominance]
        return [(fx, t)] * args.lead_repeats + [
            (fx, Trial(t.j, t.kind, a)) for a in arms for _ in range(args.block_repeats)
        ]

    def resume_sweep(game: dict, trials: list) -> list | None:
        """The sweep results of an earlier run at the SAME failure point (same x, same decisions, same edits), so a
        relaunch does not re-measure them. The game is deterministic: the trials would return the same credits."""
        want = {t.label() for t in trials}
        rows = [
            json.loads(line)
            for line in Path(args.resume_sweep).read_text().splitlines()
            if line.strip()
        ]
        sweeps = [e for e in rows if e["kind"] == "sweep"]
        if not sweeps or sweeps[-1]["failure_x"] != game["max_x"]:
            return None
        tag = f"r{sweeps[-1]['round']:02d}-sweep"
        got = {e["trial"]: e for e in rows if e["kind"] == "trial" and e.get("tag") == tag}
        if set(got) != want:
            return None
        return [
            {
                "trial": k,
                "credit": v["credit"],
                "max_x": v["max_x"],
                "end": v["end"],
                "flag": v["flag"],
            }
            for k, v in got.items()
        ]

    def anchor_keep(
        game: dict, upto: int, rnd: int, env_id: str | None = None, every: int | None = None
    ) -> None:
        """keep_base trials where the frozen game keeps the base and survives (decisions before the frontier)."""
        # A keep row is honest evidence only when its credit window (--horizon frames) ends before the game does: a
        # window that runs into the death credits keep_base low at a decision that did nothing wrong (review
        # 2026-11-16: the anchors of an earlier run fell 0.28 -> 0.00 over the last decisions before the failure).
        left, tail = 0, {}
        for st in reversed(game["steps"]):
            left += int(st.get("frames") or 0)
            tail[st["i"]] = left
        jobs = []
        for st in game["steps"][: upto : max(1, every or args.keep_anchor)]:
            if not game.get("flag") and tail.get(st["i"], 0) <= args.horizon:
                continue
            if st.get("edit") == "keep_base":
                jobs.append(Trial(st["i"], "takeoff", "keep_base"))
            if st.get("apex_executed") and st.get("apex_edit") == "keep_base":
                jobs.append(Trial(st["i"], "apex", "keep_base"))
        jobs = jobs[: budget.remaining]
        where = "of the 1-1 game" if env_id == ENV_1_1 else "before the frontier"
        say(f"round {rnd}: anchoring keep_base at {len(jobs)} decisions {where}")
        pmap(
            lambda t, g=game, r=rnd: run_trial(
                t,
                g["steps"][: t.j],
                f"r{r:02d}-anchor{'-1-1' if env_id == ENV_1_1 else ''}",
                g["steps"][t.j],
                env_id,
            ),
            jobs,
        )

    def anchor_1_1(rnd: int) -> None:
        g = game_1_1["game"]
        if args.anchor_1_1 > 0 and g is not None:
            anchor_keep(g, len(g["steps"]), rnd, ENV_1_1, args.anchor_1_1)

    def rebalance(game: dict, rnd: int) -> None:
        """Keep rows must keep pace with override rows (a failed run of 2026-11-16): anchor again when the
        override rows of this run exceed --max-override-ratio times its keep rows."""
        while (
            (args.keep_anchor > 0 or args.anchor_1_1 > 0)
            and unbalanced()
            and budget.remaining > 0
        ):
            before = rows["keep"]
            say(f"round {rnd}: balance {rows} {rows_dom} — anchoring keep_base before more override rows")
            if args.keep_anchor > 0:
                anchor_keep(game, min(frontier_decisions(game, args.depth), default=0), rnd)
            anchor_1_1(rnd)
            if rows["keep"] == before:
                say("!! no keep_base decision left to anchor: the balance guard cannot be met — the run stops")
                budget.halt = "balance_unmet"
                write({"kind": "halt", "why": budget.halt, "round": rnd, "rows": dict(rows)})
                break
            wait_installed(guarded, (refiner.takeoff, refiner.apex), log=say)
            cached.bump()

    leak_count: dict = {}

    def repair_leaks(
        g2: dict, expected: list | None, reach: int, rnd: int, env_id: str = ENV_2_1
    ) -> dict:
        """The deployed frozen game must be the game that was tested. Where it first plays another move than the
        tested game, teach the tested move there and play again. A decision that leaks more than --leak-patience
        times in one round is left alone (the corrector cannot separate it from the state of the fix)."""
        if not expected:
            say("   (no recorded steps of the tested game: the deployed game is not compared with it)")
        for n in range(args.leak_patience):
            if not expected or g2["flag"] or g2["max_x"] >= reach - args.pass_margin:
                break
            t = first_divergence(g2["steps"], expected)
            if t is None or budget.remaining == 0:
                break
            key = (rnd, env_id, t.j, t.kind)
            leak_count[key] = leak_count.get(key, 0) + 1
            if leak_count[key] > args.leak_patience:
                say(f"   LEAK at decision {t.j} came back {leak_count[key]} times this round: not repaired again")
                write({"kind": "leak_unresolved", "round": rnd, "env": env_id, "trial": t.label()})
                break
            say(
                f"   LEAK: the deployed game departs from the tested one at decision {t.j} "
                f"(x={g2['steps'][t.j]['x']}): reach {g2['max_x']} vs tested {reach}; teaching {t.label()}"
            )
            write({"kind": "leak", "round": rnd, "env": env_id, "attempt": n + 1, "trial": t.label(),
                   "deployed_max_x": g2["max_x"], "tested_max_x": reach})
            pmap_warm(
                lambda tt, g=g2, r=rnd: run_trial(
                    tt, g["steps"][: tt.j], f"r{r:02d}-leak", g["steps"][tt.j], env_id
                ),
                [t] * min(args.lead_repeats, budget.remaining),
                lambda tt: tt.label(),
            )
            lvl = "-1-1" if env_id == ENV_1_1 else ""
            g2 = frozen(env_id, f"r{rnd:02d}{lvl}-leakfix{leak_count[key]}-d{t.j}")
        return g2

    if (
        args.anchor_from
    ):  # bootstrap: keep_base along a recorded frozen game (exact replays of its own moves)
        g0 = next(
            e
            for e in (
                json.loads(x) for x in Path(args.anchor_from).read_text().splitlines() if x.strip()
            )
            if e["kind"] == "frozen" and e.get("env", "").endswith("2-1-v0")
        )
        anchor_keep(g0, max(0, len(g0["steps"]) - args.depth), 0)
        wait_installed(guarded, (refiner.takeoff, refiner.apex), log=say)

    outcome = "rounds_done"
    game = frozen(ENV_2_1, "r00")
    # A resumed leg starts from the game the earlier legs accepted. Rows sent after that game (a failed round's sweep,
    # keep anchors) can make a kept fix drop out (2026-11-17: the mid-air pull back at decision 28 after a failed round).
    # Compare with the furthest frozen 2-1 game of the earlier logs and repair the first departure.
    ref_game = None
    for pth in args.keep_fixes_from:
        for line in Path(pth).read_text().splitlines():
            if '"kind": "frozen"' not in line:
                continue
            e = json.loads(line)
            if e.get("env") == ENV_2_1 and (ref_game is None or e["max_x"] >= ref_game["max_x"]):
                ref_game = e
    if ref_game is not None and not game["flag"] and game["max_x"] < ref_game["max_x"] - args.pass_margin:
        say(f"the frozen game ({game['max_x']}) is behind the earlier legs' accepted game ({ref_game['max_x']})")
        game = repair_leaks(game, ref_game["steps"], int(ref_game["max_x"]), 0)
    current["game"] = game
    if args.anchor_1_1 > 0:
        g11 = frozen(ENV_1_1, "r00-1-1")
        if not g11["flag"] and not args.dry_run:
            raise SystemExit(f"the frozen system does not clear 1-1 before any training ({g11['max_x']})")
        game_1_1["game"] = g11
        anchor_1_1(0)
    for rnd in range(1, args.rounds + 1):
        if game["flag"]:
            outcome = "flag"
            break
        if budget.remaining == 0:
            outcome = "records_cap"
            break
        best_x = game["max_x"]
        if (
            rnd == 1 and args.path
        ):  # adopt a given move sequence (found by an offline search), disclosed
            src = game
            if args.path_from:
                src = next(
                    e
                    for e in (
                        json.loads(x)
                        for x in Path(args.path_from).read_text().splitlines()
                        if x.strip()
                    )
                    if e["kind"] == "frozen" and e.get("env", "").endswith("2-1-v0")
                )
            j0 = args.path_start
            arcs = {
                "right_run_jump_short",
                "right_run_jump_mid",
                "right_run_jump_full",
                "right_jump_full",
            }
            path = []
            for k, m in enumerate(args.path.split(",")):
                path.append(Trial(j0 + k, "takeoff", f"override_{m}"))
                if (
                    m in arcs
                ):  # the search played every arc with the in-air choice "keep": force and credit it too
                    path.append(Trial(j0 + k, "apex", "override_apex_keep"))
            say(
                f"round 1: adopting a given route of {len(path)} moves from decision {j0} of "
                f"{'the recorded game' if args.path_from else 'this frozen game'}: {args.path}"
            )
            write(
                {
                    "kind": "given_path",
                    "path": [t.label() for t in path],
                    "from": args.path_from or "r00",
                }
            )
            if not learn_path(src, path, src["max_x"], rnd):
                outcome = "path_cap"
                break
            game = current["game"]
            if not retention(rnd):
                outcome = "retention_loss"
                break
            continue
        cands = frontier_decisions(game, args.depth)
        trials = [t for j in cands for t in trials_at(game["steps"][j], j)]
        say(
            f"round {rnd}: failure at x={best_x} ({game['end']}), sweeping decisions {cands}: {len(trials)} trials "
            f"(Records left {budget.remaining})"
        )
        if args.keep_anchor > 0:
            anchor_keep(game, min(cands), rnd)
        rebalance(game, rnd)
        trials = trials[: budget.remaining]
        reused = resume_sweep(game, trials) if rnd == 1 and args.resume_sweep else None
        if reused is not None:
            results = reused
            say(
                f"round {rnd}: reusing {len(results)} sweep results from {args.resume_sweep} (same failure point)"
            )
        else:
            results = [
                r
                for r in pmap(
                    lambda t, g=game, r=rnd: run_trial(t, g["steps"][: t.j], f"r{r:02d}-sweep"),
                    trials,
                )
                if r is not None
            ]
        keep = {
            (r["trial"].split(":")[0], r["trial"].split(":")[1]): r["credit"]
            for r in results
            if r["trial"].endswith(":keep_base")
        }
        table = sorted(
            ((r["credit"] or 0.0, r["max_x"], r["end"], r["trial"]) for r in results), reverse=True
        )
        for c, x, e, lab in table[:12]:
            say(f"   {lab:44s} credit={c:.5f} reach={x} end={e}")
        def extras(r: dict) -> list:
            """Overrides the corrector made by itself after the tested edit: the trial is then not a single edit, and
            a refit can drop them (seen once: 228 of 244 repeats of round 2's leader reached 1735, not 1828)."""
            return [
                f"{st['i']}:{kd}:{st.get(key)}"
                for st in r.get("steps") or []
                if not st.get("replayed")
                for kd, key in (("takeoff", "edit"), ("apex", "apex_edit"))
                if st.get(key) not in (None, "keep_base")
                and not (st.get("trial") and kd == r["trial"].split(":")[1])  # the tested edit itself
            ]

        def cc(x, edit: str = "keep_base") -> float:  # the reward the corrector receives
            return inner.sent(float(x or 0.0), edit)

        def gain(r: dict) -> float:
            j_, kind_, e_ = r["trial"].split(":")
            return cc(r["credit"], e_) - cc(keep.get((j_, kind_)))

        passing = []
        for r in results:
            j, kind, edit = r["trial"].split(":")
            k = keep.get((j, kind))
            undo = edit == "keep_base" and game["steps"][int(j)].get(
                "edit" if kind == "takeoff" else "apex_edit"
            ) not in (None, "keep_base")
            if (
                (edit != "keep_base" or undo)
                and r["credit"] is not None
                and (r["flag"] or r["max_x"] > best_x + args.pass_margin)
                and (undo or k is None or cc(r["credit"], edit) > cc(k))
            ):
                passing.append(r)
        write(
            {
                "kind": "sweep",
                "round": rnd,
                "failure_x": best_x,
                "decisions": cands,
                "keep_credit": {f"{a}:{b}": v for (a, b), v in keep.items()},
                "passing": [r["trial"] for r in passing],
                "extras": {r["trial"]: extras(r) for r in results if extras(r)},
                "gain_over_keep": {r["trial"]: round(gain(r), 6) for r in passing},
            }
        )
        clean = [r for r in passing if not extras(r)]
        if passing and not clean:
            say(f"round {rnd}: every passing trial holds an untested override: {[r['trial'] for r in passing]}")
        elif len(clean) < len(passing):
            say(f"round {rnd}: left out (untested extra override): {[r['trial'] for r in passing if extras(r)]}")
        passing = clean or passing
        if not passing and args.archive_records > 0:
            path = archive_search(game, cands, best_x, rnd)
            if path is None:
                outcome = "unsolved"
                break
            if not learn_path(game, path, best_x, rnd):
                outcome = "path_cap"
                break
            game = current["game"]
            if not retention(rnd):
                outcome = "retention_loss"
                break
            continue
        if not passing:
            say(
                f"round {rnd}: NO single edit at {cands} gets past x={best_x} — stop (multi-move blocker?)"
            )
            outcome = "unsolved"
            break
        lead = max(passing, key=lambda r: (gain(r), r["max_x"]))  # credit above keep's at the same decision
        if rnd == 1 and args.leader:  # a relaunch resumes the leader the last run was teaching
            lead = next((r for r in passing if r["trial"] == args.leader), lead)
        j, kind, edit = lead["trial"].split(":")
        leader = Trial(int(j), kind, edit)
        tested = {r["trial"]: r for r in results}  # the leader trial's own game: what a deployed game must match
        pass_edits = {
            r["trial"].split(":")[2] for r in passing if r["trial"].split(":")[:2] == [j, kind]
        }
        say(
            f"round {rnd}: leader {leader.label()} credit={lead['credit']:.5f} reach={lead['max_x']}; "
            f"passing edits here {sorted(pass_edits)}"
        )
        # Other passing candidates, best credit first, for when the leader fights a kept fix (seen in an earlier run: the
        # in-air brake at x 2459 knocked out the in-air pull_back kept at x 1431, a similar state).
        ranked = sorted(passing, key=lambda r: (-gain(r), -r["max_x"]))
        tried_keys = {(leader.j, leader.kind)}
        fights = 0
        reps, accepted = 0, False
        drift = 0  # repeats of the leader that did not play the tested game (another reach)
        batch = [leader] * args.batch
        upkeep: list = []  # (fix, trial) jobs for kept fixes the corrector stopped selecting
        while reps < args.repeat_cap and budget.remaining > 0:
            rebalance(game, rnd)
            jobs = [(None, t) for t in batch] + upkeep
            jobs = jobs[: budget.remaining]
            outs = pmap_warm(
                lambda job, g=game, r=rnd: (
                    job,
                    run_trial(
                        job[1],
                        g["steps"][: job[1].j] if job[0] is None else job[0]["prefix"],
                        f"r{r:02d}-repeat" if job[0] is None else f"r{r:02d}-keep",
                        None if job[0] is None else job[0]["step"],
                    ),
                ),
                jobs,
                lambda job: (job[0]["trial"].label() if job[0] else None, job[1].label()),
            )
            reps += len(jobs)
            want = int((tested.get(leader.label()) or {}).get("max_x") or 0)
            for job, res in outs:
                if job[0] is None and res and job[1].label() == leader.label() and want:
                    if abs(int(res["max_x"]) - want) > args.pass_margin:
                        old = tested.get(leader.label()) or {}
                        if (res["flag"] or res["max_x"] > best_x + args.pass_margin) and cc(
                            res["credit"], leader.edit
                        ) >= cc(old.get("credit"), leader.edit):
                            # the corrector changed and the same edit now plays another game that still passes:
                            # that game is what a deployed game must match from here on
                            write({"kind": "retest", "round": rnd, "trial": leader.label(), "was": want,
                                   "now": int(res["max_x"])})
                            tested[leader.label()] = res
                            want = int(res["max_x"])
                        else:
                            drift += 1
            if drift >= 4:
                say(
                    f"   the leader's repeats no longer pass ({drift} repeats did not get past x={best_x}): "
                    "the round stops, the sweep is stale"
                )
                write({"kind": "leader_drift", "round": rnd, "trial": leader.label(), "tested_reach": want,
                       "drifted": drift, "repeats": reps})
                break
            wait_installed(guarded, (refiner.takeoff, refiner.apex), log=say)
            cached.bump()
            pr = probe(game["steps"][leader.j], leader.kind)
            pr["live_sample_count"] = status_query(
                guarded, refiner.takeoff if leader.kind == "takeoff" else refiner.apex
            ).get("live_sample_count")
            write({"kind": "probe", "round": rnd, "repeats": reps, "trial": leader.label(), **pr})
            say(
                f"   probe after {reps} repeats: {pr['status']} {pr['policy']} ({pr['reason']}) "
                f"samples={pr['live_sample_count']} sent={sent}"
            )
            # The gate needs the leader's lower bound above EVERY competitor's upper bound. Next batch: the passing
            # edit with the best lower bound, twice, plus one more trial of each arm whose interval still overlaps
            # it (the experiments the gate is waiting for). Deterministic game: a repeat returns the same credit.
            diag = pr.get("diagnostics") or {}

            def ci(arm: str, d=diag) -> tuple[float, float]:
                c = (d.get(arm) or {}).get("credible_interval") or [0.0, 1.0]
                return float(c[0]), float(c[1])

            if diag:
                # the leader stays the best MEASURED edit: the server's interval pools other states (2026-10-04,
                # round 3: it ranked pull_back above brake, which reaches 340 px further at this exact state)
                best_pass = leader.edit
                lo = ci(best_pass)[0]
                if (
                    best_pass == "keep_base"
                ):  # undoing an override: no override may beat keep's upper bound
                    blocking = [
                        a for a in diag if a != "keep_base" and ci(a)[0] > ci("keep_base")[1]
                    ]
                else:
                    blocking = [a for a in diag if a != best_pass and ci(a)[1] >= lo]
                # Under keep-dominance the leader only has to beat keep_base. Repeating a RIVAL override sends rows
                # that say "this other override is good", and the server pools them over states: on 2026-11-16
                # 16 such `override_jump` rows at decision 11 made `jump` fire at decision 3 and broke 1-1.
                repeat_arms = [a for a in blocking if a == "keep_base" or not args.keep_dominance]
                batch = [leader] * args.lead_repeats + [
                    Trial(leader.j, leader.kind, a)
                    for a in repeat_arms
                    for _ in range(args.block_repeats)
                ]
                say(
                    f"   leader {best_pass} ci={ci(best_pass)}; blocking {len(blocking)}: "
                    + ", ".join(f"{a}{ci(a)}" for a in blocking[:6])
                )
            else:
                batch = [leader] * args.batch
            upkeep, lost = [], []
            for fx in fixes:
                ok, blk, fpr = probe_fix(fx)
                write(
                    {
                        "kind": "fix_probe",
                        "round": rnd,
                        "fix": fx["trial"].label(),
                        "selected": ok,
                        "blocking": blk,
                        "status": fpr["status"],
                        "policy": fpr["policy"],
                    }
                )
                if not ok:
                    lost.append(fx["trial"].label())
                    upkeep += fix_jobs(fx, blk)
            if lost:
                say(
                    f"   kept fixes not selected now: {lost} — re-teaching ({len(upkeep)} trials next batch)"
                )
            fights = fights + 1 if lost else 0
            alt = next(
                (
                    r
                    for r in ranked
                    if tuple(r["trial"].split(":")[:2]) not in {(str(a), b) for a, b in tried_keys}
                ),
                None,
            )
            if fights >= args.fight_patience and alt is not None:
                j2, kind2, edit2 = alt["trial"].split(":")
                leader = Trial(int(j2), kind2, edit2)
                tried_keys.add((leader.j, leader.kind))
                pass_edits = {
                    r["trial"].split(":")[2]
                    for r in passing
                    if r["trial"].split(":")[:2] == [j2, kind2]
                }
                batch, fights = [leader] * args.lead_repeats, 0
                say(
                    f"   the leader fights a kept fix {args.fight_patience} batches in a row — switching to "
                    f"{leader.label()} (credit {alt['credit']:.5f}, reach {alt['max_x']})"
                )
                write(
                    {"kind": "leader_switch", "round": rnd, "leader": leader.label(), "lost": lost}
                )
                continue
            if pr["effective"] in pass_edits and not lost:
                g2 = frozen(ENV_2_1, f"r{rnd:02d}-after{reps}")
                lt = tested.get(leader.label()) or {}
                g2 = repair_leaks(g2, lt.get("steps"), int(lt.get("max_x") or 0), rnd)
                if g2["flag"] or g2["max_x"] > best_x + args.pass_margin:
                    if leader.label() not in {fx["trial"].label() for fx in fixes}:
                        fixes.append(
                            {
                                "trial": leader,
                                "prefix": game["steps"][: leader.j],
                                "step": game["steps"][leader.j],
                                "from": f"this run#r{rnd}",
                            }
                        )
                    accepted, game = True, g2
                    current["game"] = game
                    break
                say(
                    f"   the probe selects {pr['policy']} but the frozen game did not get past x={best_x}"
                )
                game_check = g2
                if (
                    game_check["max_x"] < best_x - args.pass_margin
                ):  # the frozen game changed before the blocker
                    accepted, game = True, g2  # new failure point earlier: continue from it
                    current["game"] = game
                    say(
                        "   the frozen game now fails EARLIER — continuing from the new failure point"
                    )
                    break
        write(
            {
                "kind": "round",
                "round": rnd,
                "leader": leader.label(),
                "repeats": reps,
                "accepted": accepted,
                "records_spent": budget.spent, "credits": meter.credits, "credits_by_kind": dict(meter.by_kind),
            }
        )
        if not accepted:
            say(f"round {rnd}: the gate did not pass within {reps} repeats — stop")
            outcome = "leader_drift" if drift >= 4 else "gate_cap" if reps >= args.repeat_cap else "records_cap"
            retention(rnd)  # the rows of a failed round are in the corrector too: 1-1 is checked (and repaired)
            break
        if not retention(rnd):
            outcome = "retention_loss"
            break
    else:
        if game["flag"]:
            outcome = "flag"
    # The end state must be a played one: rows sent after the last accepted game (a failed round) can undo a fix.
    # Play the frozen game on the final models, repair it against the last accepted game, and check 1-1 again.
    if not args.dry_run or args.rounds > 0:
        accepted_game = current["game"]
        final = frozen(ENV_2_1, "final")
        first_max, first_flag = final["max_x"], final["flag"]
        if not final["flag"] and final["max_x"] < accepted_game["max_x"] - args.pass_margin:
            final = repair_leaks(final, accepted_game["steps"], int(accepted_game["max_x"]), 99)
        kept_1_1 = retention(99)  # 1-1 on the final models, always
        write({"kind": "final", "max_x": final["max_x"], "end": final["end"], "flag": final["flag"],
               "accepted_max_x": accepted_game["max_x"], "before_repair": {"max_x": first_max, "flag": first_flag},
               "retention_1_1": bool(kept_1_1)})
        game = final
    fp1 = base_fingerprint(guarded, (base.takeoff, base.apex))
    consumed = {
        d: status_query(guarded, d).get("live_sample_count")
        for d in (refiner.takeoff, refiner.apex)
    }
    summary = {
        "kind": "summary",
        "outcome": budget.halt or outcome,
        "records_spent": budget.spent, "credits": meter.credits, "credits_by_kind": dict(meter.by_kind),
        "sent_by_domain": sent,
        "rows_by_kind": dict(rows),
        "rows_by_domain": rows_dom,
        "reward_max": args.reward_max, "override_cost": args.override_cost, "rewards_capped": inner.capped,
        "corrector_live_sample_count_after": consumed,
        "final_frozen": _game_line(game),
        "base_changed": fingerprint_changed(fp0, fp1),
        "base_feedback_calls": sum(
            guarded.feedback_calls.get(d, 0) for d in (base.takeoff, base.apex)
        ),
        "base_cache": {"hits": live.hits, "misses": live.misses},
        "corrector_cache": {"hits": cached.hits, "misses": cached.misses},
        "wall_s": round(time.time() - t0),
    }
    write(summary)
    write_provenance(
        log_path,
        kind="refine-frontier-log",
        extra={**prov, "status": "finished", "summary": summary},
    )
    say(f"DONE {json.dumps(summary)}")
    say(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
