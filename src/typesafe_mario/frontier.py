"""Frontier exploration for correctors on a frozen base: try every edit where the frozen system fails.

The reasoning in short: 2-1 is deterministic, so the frozen
system (frozen base + frozen correctors) reaches the same state every game and fails at the same place. Earlier
`refine` runs spread their Records over the whole level and over ten edits, and the decisive state got about one
try per 17 games. Here each learning game is ONE trial:

1. replay the frozen system's own moves up to decision ``j`` (emulator only: no Queries, no Records);
2. at ``j``, query the corrector (a real decision id) and execute a given edit instead of its answer;
3. play the frozen system after ``j`` until the 512-frame credit of decision ``j`` is known;
4. send ONE feedback: decision ``j``, the executed edit, its credit. Nothing else is sent.

Which ``j``: the last decisions before the frozen system dies or stalls. That is a property of the frozen system's
own play, not a position: no x in any context, no trigger, no label. Every tried edit is sent with its measured
credit, failures included, so the learner gets exploration data, never a "correct action". The frozen gate
(``require_credible_dominance``) decides whether an override is executed frozen. With ``keep_dominance`` the client
also plays an override when the server abstains on overlapping intervals and the override's interval lies above
``keep_base``'s (``refinement.keep_dominance_pick``).

Two caches keep the Queries affordable (a query took ~5 s in October 2026):

- ``CachedBase``: the base pair is frozen and gets no feedback, so its answer to an exact context is cached for the
  whole run. This also keeps a server transient out of learning (a run started on an empty cache fetches every base answer live).
- ``EpochCache``: the correctors' exploit answers are cached until ``bump()``, which the driver calls after each
  batch of feedback once the new model is installed. Within an epoch, the play after ``j`` is a fixed policy.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cadence import CadenceParser
from .demos import apex_hook, build_rows
from .macros import ApexChoice, Macro, execute
from .online_duo import CreditConfig, DelayedCredit, RecordsBudget
from .refinement import (
    ARC_MACROS,
    BASE_APEX_FEATURES,
    BASE_MACROS,
    BASE_TAKEOFF_FEATURES,
    KEEP_BASE,
    OVERRIDE_PREFIX,
    Pair,
    RefinedPolicy,
    apex_context,
    apex_edits,
    frozen_base,
    refiner_policy,
    takeoff_context,
    takeoff_edits,
)
from .reward import PROGRESS_FRACTION
from .runner import _unwrap_ram, create_mario_env


def _key(domain_id: str, body: Mapping[str, Any]) -> tuple[str, str, Any, Any]:
    values = (body.get("context") or {}).get("values")
    return (
        domain_id,
        json.dumps(values, sort_keys=True, default=str),
        body.get("selection_mode"),
        body.get("allow_exploration"),
    )


class _Passthrough:
    def __init__(self, client: Any) -> None:
        self._client = client

    def feedback(self, domain_id: str, body: Any) -> Any:
        return self._client.feedback(domain_id, body)

    def events(self, domain_id: str, body: Any) -> Any:
        return self._client.events(domain_id, body)

    def create_domain(self, config: Mapping[str, Any]) -> Any:
        return self._client.create_domain(config)

    def get_domain(self, domain_id: str) -> Any:
        return self._client.get_domain(domain_id)

    def call(self, method: str, path: str, body: Any = None) -> Any:
        return self._client.call(method, path, body)

    def __getattr__(self, name: str) -> Any:  # feedback_calls / query_calls of a GuardedClient
        return getattr(self._client, name)


class CachedBase(_Passthrough):
    """Caches query answers of the frozen base domains by exact context, for the whole run. A status query (no
    context) is never cached. ``path``: a JSONL file the cache is loaded from and appended to."""

    def __init__(self, client: Any, base_ids: Sequence[str], path: Path | None = None) -> None:
        super().__init__(client)
        self.base_ids = frozenset(base_ids)
        self._d: dict[tuple, Any] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        self._path = path
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._d[tuple(e["k"][:2]) + (e["k"][2], e["k"][3])] = (e["s"], e["r"])

    def query(self, domain_id: str, body: Any) -> Any:
        if domain_id not in self.base_ids or not (body.get("context") or {}).get("values"):
            return self._client.query(domain_id, body)
        k = _key(domain_id, body)
        with self._lock:
            if k in self._d:
                self.hits += 1
                return self._d[k]
        out = self._client.query(domain_id, body)
        with self._lock:
            self.misses += 1
            self._d[k] = out
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps({"k": list(k), "s": out[0], "r": out[1]}, default=str) + "\n"
                    )
        return out


class EpochCache(_Passthrough):
    """Caches the correctors' exploit answers until ``bump()``. Only used for decisions that send no feedback."""

    def __init__(self, client: Any, domain_ids: Sequence[str]) -> None:
        super().__init__(client)
        self.domain_ids = frozenset(domain_ids)
        self._d: dict[tuple, Any] = {}
        self._lock = threading.Lock()
        self.epoch = 0
        self.hits = self.misses = 0

    def bump(self) -> None:
        with self._lock:
            self._d.clear()
            self.epoch += 1

    def query(self, domain_id: str, body: Any) -> Any:
        if domain_id not in self.domain_ids or not (body.get("context") or {}).get("values"):
            return self._client.query(domain_id, body)
        k = _key(domain_id, body)
        with self._lock:
            if k in self._d:
                self.hits += 1
                return self._d[k]
        out = self._client.query(domain_id, body)
        with self._lock:
            self.misses += 1
            self._d[k] = out
        return out


class LockedBudget(RecordsBudget):
    """``RecordsBudget`` shared by parallel trials."""

    def __init__(self, max_records: int) -> None:
        super().__init__(max_records)
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            return super().take()


@dataclass(frozen=True)
class Trial:
    """Decision ``j`` of the frozen game executes ``edit`` on the ``kind`` corrector (takeoff or apex); that
    decision's credit is the one feedback of the game."""

    j: int
    kind: str
    edit: str

    def label(self) -> str:
        return f"{self.j}:{self.kind}:{self.edit}"


@dataclass
class FrontierConfig:
    base: Pair
    refiner: Pair
    env_id: str = "SuperMarioBros-2-1-v0"
    seed: int = 777
    horizon: int = 512
    max_decisions: int = 400
    stall: int = 12
    run_id: str = "refine-frontier"
    keep_dominance: bool = False


class ReplayMismatch(RuntimeError):
    """The emulator replay of the frozen moves did not land where the recorded game did."""


class _Sender:
    """What ``DelayedCredit`` calls when a credited row's window closes: the LIVE corrector policy that issued the
    decision. The ``RefinedPolicy`` cannot be used for this: its ``refiner`` is switched back to the frozen copy
    (``learn=False``, which silently sends nothing) on the next decision, before the window closes. Measured
    2026-10-04: 39 of 42 trial feedbacks of the first calibration were never posted."""

    def __init__(self, refined: RefinedPolicy, live_refiner: Any) -> None:
        self._refined, self._live = refined, live_refiner

    @property
    def domain_id(self) -> str:
        return self._live.domain_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._refined.feature_names

    def next_context(self, next_state: Mapping[str, Any]) -> dict[str, Any]:
        return self._refined.next_context(next_state)

    def send_feedback(self, pending: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self._live.send_feedback(pending, **kwargs)


def _replay_hook(
    parser: Any, env: Any, macro: Macro, choice: str | None, sink: dict[str, Any]
) -> Any:
    """The apex look of a replayed decision: parse exactly as ``demos.apex_hook`` does (the parser keeps state),
    then return the recorded choice instead of asking a domain."""

    def on_apex(event: Any) -> Any:
        ram_now = _unwrap_ram(env)
        parser.parse(
            event.info,
            ram_now,
            previous_action=macro.value,
            previous_reward=event.reward,
            frames=event.frame,
        )
        parser.floor_height_tiles(ram_now, event.x, int(event.info.get("y_pixel", 0)))
        sink["fired"] = True
        if choice is None:
            raise ReplayMismatch("the replay reached an apex the recorded game did not")
        return ApexChoice(choice)

    return on_apex


def play(
    clients: Mapping[str, Any],
    cfg: FrontierConfig,
    *,
    prefix: Sequence[Mapping[str, Any]] = (),
    trial: Trial | None = None,
    budget: RecordsBudget | None = None,
    episode_id: str = "frozen",
    env_id: str | None = None,
    stop_when_credited: bool = True,
    plan: Sequence[Trial] = (),
    credit_plan: bool = False,
) -> dict[str, Any]:
    """One game of the frozen system, optionally replaying ``prefix`` (the recorded steps of an earlier game of the
    same frozen system) and running ``trial`` at decision ``trial.j == len(prefix)``.

    ``clients``: ``live`` (guarded, base cached: the trial's decision) and ``cached`` (``live`` + ``EpochCache`` on
    the correctors: every other decision). Returns the steps (each with the base's context, so a later game can
    replay them and a probe can rebuild the corrector context), how the game ended and the trial's credit.

    ``plan`` (the springboard stage): edits forced at decisions after the prefix, before the trial. With
    ``credit_plan=False`` they send nothing (the trial is the only feedback); with ``credit_plan=True`` each planned
    decision is credited too, by its own 512-frame window of the game as played (the online loop's rule)."""
    forced = [*plan, *([trial] if trial is not None else [])]
    if forced and min(t.j for t in forced) < len(prefix):
        raise ValueError("planned and trial decisions must follow the replayed prefix")
    if trial is not None and not plan and trial.j != len(prefix):
        raise ValueError("the trial decision must follow the replayed prefix")
    credited = {t for t in forced if t is trial or credit_plan}
    by_i: dict[int, list[Trial]] = {}
    for t in forced:
        by_i.setdefault(t.j, []).append(t)
    expected = len(credited)
    env_id = env_id or cfg.env_id
    live, cached = clients["live"], clients["cached"]
    bt = frozen_base(live, cfg.base.takeoff, BASE_TAKEOFF_FEATURES, run_id=cfg.run_id)
    ba = frozen_base(live, cfg.base.apex, BASE_APEX_FEATURES, run_id=cfg.run_id)
    rt = refiner_policy(cached, cfg.refiner.takeoff, "takeoff", learn=False, run_id=cfg.run_id)
    ra = refiner_policy(cached, cfg.refiner.apex, "apex", learn=False, run_id=cfg.run_id)
    rt_live = refiner_policy(
        live,
        cfg.refiner.takeoff,
        "takeoff",
        learn=True,
        run_id=cfg.run_id,
        server_exploration=False,
    )
    ra_live = refiner_policy(
        live, cfg.refiner.apex, "apex", learn=True, run_id=cfg.run_id, server_exploration=False
    )
    takeoff = RefinedPolicy("takeoff", bt, rt, keep_dominance=cfg.keep_dominance)
    apex = RefinedPolicy("apex", ba, ra, takeoff=takeoff, keep_dominance=cfg.keep_dominance)
    names = tuple(dict.fromkeys((*BASE_TAKEOFF_FEATURES, *BASE_APEX_FEATURES)))
    budget = budget if budget is not None else RecordsBudget(0)
    credit = DelayedCredit(CreditConfig(horizon=cfg.horizon), budget, episode_id)
    env = create_mario_env(env_id)
    parser = CadenceParser(decision_horizon_frames=8)
    steps: list[dict[str, Any]] = []
    end, trial_note = "max", None
    try:
        _f, info = env.reset(seed=cfg.seed)
        snap = parser.parse(info, _unwrap_ram(env), previous_action=None)
        best, still = snap.x, 0
        for i in range(cfg.max_decisions):
            if snap.dead or snap.clear:
                break
            if i < len(prefix):  # --- replay: emulator only
                st = prefix[i]
                if int(snap.x) != int(st["x"]):
                    raise ReplayMismatch(f"decision {i}: x {snap.x} != recorded {st['x']}")
                macro = Macro(st["executed"])
                sink: dict[str, Any] = {}
                ex = execute(
                    env,
                    macro,
                    grounded=snap.grounded,
                    ram=_unwrap_ram(env),
                    apex=_replay_hook(parser, env, macro, st.get("apex_executed"), sink),
                )
                fired = ex.apex is not None and bool(sink)
                if bool(st.get("apex_executed")) != fired:
                    raise ReplayMismatch(
                        f"decision {i}: apex fired={fired}, recorded {st.get('apex_executed')}"
                    )
                nxt = parser.parse(
                    ex.info,
                    _unwrap_ram(env),
                    previous_action=(st["apex_executed"] if fired else macro.value),
                    previous_reward=(
                        ex.total_reward - ex.reward_to_apex if fired else ex.total_reward
                    ),
                    frames=(ex.frames - ex.frames_to_apex if fired else ex.frames),
                )
                steps.append({**st, "replayed": True})
            else:  # --- live: the frozen system, or the trial at j
                here = by_i.get(i, [])
                tk = next((t for t in here if t.kind == "takeoff"), None)
                ta = next((t for t in here if t.kind == "apex"), None)
                is_t = bool(here) and trial is not None and trial in here
                keep_t, want_a = tk in credited, ta in credited
                takeoff.refiner = rt_live if keep_t else rt
                apex.refiner = ra_live if want_a else ra
                if tk is not None:
                    takeoff.force_edit = tk.edit
                if ta is not None:
                    apex.force_edit = ta.edit
                decision = takeoff.choose(snap, BASE_MACROS)
                tlog = takeoff.log[-1]
                pending_t = takeoff.take_pending()
                apex_state: dict[str, Any] = {}
                ex = execute(
                    env,
                    decision.action,
                    grounded=snap.grounded,
                    ram=_unwrap_ram(env),
                    apex=apex_hook(parser, env, apex, decision, apex_state),
                )
                fired = ex.apex is not None and bool(apex_state)
                pending_a = apex.take_pending() if fired else None
                alog = apex.log[-1] if fired else None
                if ta is not None and not fired:
                    apex.force_edit = None
                    trial_note = (
                        f"no_apex at {i}"  # the takeoff produced no apex look: nothing to credit
                    )
                    expected -= 1 if want_a else 0
                nxt = parser.parse(
                    ex.info,
                    _unwrap_ram(env),
                    previous_action=(
                        apex_state["decision"].action.value if fired else decision.action.value
                    ),
                    previous_reward=(
                        ex.total_reward - ex.reward_to_apex if fired else ex.total_reward
                    ),
                    frames=(ex.frames - ex.frames_to_apex if fired else ex.frames),
                )
                terminal = bool(ex.terminated or nxt.dead or nxt.clear)
                rows = build_rows(
                    snapshot=snap,
                    decision=decision,
                    execution=ex,
                    apex_state=apex_state if fired else None,
                    next_snapshot=nxt,
                    terminal=terminal,
                    names=names,
                    reward_config=PROGRESS_FRACTION,
                    episode_id=episode_id,
                    first_step=len(credit.entries),
                    tag_kind=True,
                )
                for r in rows:
                    r["x"], r["x_after"] = int(snap.x), int(nxt.x)
                keep_a = want_a and fired
                credit.add(
                    rows[0],
                    pending_t if keep_t else None,
                    _Sender(takeoff, rt_live) if keep_t else None,
                )
                if fired:
                    credit.add(
                        rows[1],
                        pending_a if keep_a else None,
                        _Sender(apex, ra_live) if keep_a else None,
                    )
                credit.flush()
                steps.append(
                    {
                        "i": i,
                        "x": int(snap.x),
                        "x_after": int(nxt.x),
                        "y": int(snap.y),
                        "y_after": int(nxt.y),
                        "base": tlog["base"],
                        "edit": tlog["refinement"],
                        "executed": tlog["executed"],
                        "status": tlog["refiner_status"],
                        "values": tlog["refiner_values"],
                        "context": tlog["context"],
                        "apex_base": alog["base"] if alog else None,
                        "apex_edit": alog["refinement"] if alog else None,
                        "apex_executed": alog["executed"] if alog else None,
                        "apex_status": alog["refiner_status"] if alog else None,
                        "apex_values": alog["refiner_values"] if alog else None,
                        "apex_context": alog["context"] if alog else None,
                        "frames": int(ex.frames),
                        "trial": is_t,
                        "forced": [t.label() for t in here],
                    }
                )
            snap = nxt
            if snap.x > best:
                best, still = snap.x, 0
            else:
                still += 1
            if snap.clear:
                end = "flag"
                break
            if snap.dead or ex.terminated:
                end = "dead"
                break
            if still >= cfg.stall:
                end = "stall"
                break
            if expected and stop_when_credited and len(credit.sent_records) >= expected:
                end = "credited"
                break
        if expected and len(credit.sent_records) < expected:
            credit.flush(final=True)
    finally:
        env.close()
        takeoff.close()
        apex.close()
    sent = credit.sent_records
    return {
        "env": env_id,
        "trial": trial.label() if trial else None,
        "end": end,
        "flag": end == "flag",
        "max_x": int(best),
        "decisions": len(steps),
        "steps": steps,
        "credit": _trial_credit(sent, steps, trial),
        "credits": [r["reward"] for r in sent],
        "records": len(sent),
        "feedback": sent,
        "note": trial_note,
    }


def _trial_credit(
    sent: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], trial: Trial | None
) -> float | None:
    """The credit sent for ``trial``'s decision (matched by kind and the decision's x), or None."""
    if trial is None:
        return None
    x = next((s["x"] for s in steps if s.get("i") == trial.j), None)
    return next(
        (r["reward"] for r in sent if r.get("kind") == trial.kind and r.get("x") == x), None
    )


# --- choosing the trials -------------------------------------------------------------------------


def frontier_decisions(game: Mapping[str, Any], depth: int = 3) -> list[int]:
    """The decisions to try, latest first: the last ``depth`` up to the last decision that made progress (the
    rule for a stall). The same rule for a death: decisions taken while Mario no longer moves
    (pinned at a wall, or in the death animation before the parser flags it) cannot change the outcome."""
    steps = game["steps"]
    if not steps:
        return []
    # Progress = a new furthest x. A decision that only bounces off a wall (x 3019 -> 3026 -> 3019 ...) moves but
    # gains nothing (2026-11-17: the sweep at 2-1's tower took the last three bounces, decisions 51-53, and never
    # tried the decisions on the way in).
    best, moved = int(steps[0]["x"]), []
    for k, st in enumerate(steps):
        if int(st["x_after"]) > best:
            best = int(st["x_after"])
            moved.append(k)
    d = moved[-1] if moved else len(steps) - 1
    return [j for j in range(d, d - depth, -1) if j >= 0]


def first_divergence(
    deployed: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> Trial | None:
    """The first decision where a deployed frozen game plays another move than the game that was TESTED (the leader
    trial: the old frozen game up to ``j``, the edit at ``j``, the frozen system after it). Returns the trial that
    teaches the tested move at that decision (``keep_base`` where the test kept the base), or None when the deployed
    game follows the tested one for as long as both were recorded.

    Why (a failed run of 2026-11-16): a tested edit reached 1591, the corrector also fired one decision later, and
    the deployed game reached 735. The runner did not compare the two games."""
    for a, b in zip(deployed, expected, strict=False):
        if int(a["x"]) != int(b["x"]):
            return None  # another state without another move: replay drift, not a leak
        if a["executed"] != b["executed"]:
            return Trial(int(b["i"]), "takeoff", b.get("edit") or KEEP_BASE)
        if a.get("apex_executed") != b.get("apex_executed"):
            if not b.get("apex_edit"):
                return None
            return Trial(int(b["i"]), "apex", b["apex_edit"])
    return None


def trials_at(step: Mapping[str, Any], j: int, *, allow_jump: bool = True) -> list[Trial]:
    """Every distinct edit at decision ``j``: ``keep_base`` and each override that executes something other than the
    base's move (an ``override_<base>`` arm runs the same game as ``keep_base``); the same on the apex look when the
    executed move was an arc jump with a look."""
    out = [
        Trial(j, "takeoff", e.value)
        for e in takeoff_edits(allow_jump=allow_jump)
        if e.value == KEEP_BASE or e.value.removeprefix(OVERRIDE_PREFIX) != step["base"]
    ]
    if step.get("apex_executed") and Macro(step["executed"]) in ARC_MACROS:
        out += [
            Trial(j, "apex", e.value)
            for e in apex_edits()
            if e.value == KEEP_BASE or e.value.removeprefix(OVERRIDE_PREFIX) != step["apex_base"]
        ]
    return out


def corrector_context(step: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """The exact corrector context of a recorded decision (for a probe query)."""
    if kind == "takeoff":
        return takeoff_context(step["context"], step["base"])
    return apex_context(step["apex_context"], step["apex_base"], step["executed"])


@dataclass
class TrialStats:
    """Per (j, kind, edit): every credit measured, and how far the game got."""

    credits: dict[str, list[float]] = field(default_factory=dict)
    reach: dict[str, int] = field(default_factory=dict)
    ends: dict[str, str] = field(default_factory=dict)

    def add(self, res: Mapping[str, Any]) -> None:
        if res["credit"] is None:
            return
        k = res["trial"]
        self.credits.setdefault(k, []).append(float(res["credit"]))
        self.reach[k] = max(self.reach.get(k, 0), int(res["max_x"]))
        self.ends[k] = res["end"]
