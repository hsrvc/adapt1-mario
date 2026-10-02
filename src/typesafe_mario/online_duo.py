"""Online learning for the duo-domain policy (takeoff + apex) with client-side delayed credit.

Rei's zero-start request (HANDOFF-2026-09-23 item 8): the #28 method — two direct-feedback
domains, the 8 grounded macros, closed-loop, no position, no teacher rows — learning online from
its own play, exploration on from the first query. Every decision is one feedback (one Record);
every jump adds one apex feedback.

The credit is computed on the client, as it was for every warm start (findings #22): a decision
is worth the progress made over the following emulator frames (``returns.frame_horizon_return``,
the fix that held on 2026-09-23) or, optionally, the normalised k-step return of #28. Online that
means **holding each decision's feedback** until enough of the future has been played:
``DelayedCredit`` keeps the rows of the current episode in step order and sends a row's feedback
once the frames after it cover the horizon, or the episode ends. The return it sends equals what
``returns.frame_horizon_returns`` / ``returns.kstep_returns`` would compute offline on the same
rows (pinned in ``tests/test_online_duo.py``). This is doing the sequential learner's job on the
client — say so in any write-up.

Cost discipline: a hard Records cap (``RecordsBudget``) is checked before every feedback; the
frozen evals spend Queries only. Read ``model_type`` at every checkpoint (#30 §7: an MLP under a
small declared reward range is a constant policy).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapt1_policy import (
    APEX_V2_FEATURE_NAMES,
    FULL_V2_FEATURE_NAMES,
    QUESTION,
    Adapt1Policy,
    learner_status,
)
from .cadence import CadenceParser
from .demos import apex_hook, build_rows
from .macros import Macro, execute
from .returns import frame_horizon_return, kstep_return
from .reward import PROGRESS_FRACTION, RewardConfig
from .runner import _unwrap_ram, create_mario_env

# The 8 grounded macros of #28 — every macro but `jump`, which was added for 2-1's springboard.
MACROS_28: tuple[Macro, ...] = tuple(m for m in Macro if m is not Macro.JUMP)


@dataclass(frozen=True)
class CreditConfig:
    """How a decision's reward is computed from the rows that follow it."""

    kind: str = "frame_horizon"  # "frame_horizon" (returns.frame_horizon_return) or "kstep"
    horizon: int = 128  # frames, frame_horizon
    k: int = 6  # decisions, kstep
    gamma: float = 0.9  # kstep

    def __post_init__(self) -> None:
        if self.kind not in ("frame_horizon", "kstep"):
            raise ValueError(f"unknown credit kind {self.kind!r}")


class RecordsBudget:
    """A hard cap on feedbacks sent (1 feedback = 1 Record), shared by both domains."""

    def __init__(self, max_records: int) -> None:
        self.max_records = int(max_records)
        self.spent = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_records - self.spent)

    def take(self) -> bool:
        if self.spent >= self.max_records:
            return False
        self.spent += 1
        return True


@dataclass
class _Entry:
    row: dict[str, Any]
    pending: dict[str, Any] | None
    policy: Adapt1Policy | None
    sent: bool = False
    dropped: bool = False
    reward_sent: float | None = None


@dataclass
class DelayedCredit:
    """The rows of one episode, in step order, each holding its decision's pending feedback until
    the return over the following rows is known.

    ``add`` appends a row (from ``demos.build_rows``: ``reward`` = the immediate progress
    fraction, ``frames``, ``outcome``, ``next_state``) with the decision's pending record and the
    policy that owns the domain. ``flush`` sends every row whose return is final — for the
    frame-horizon return, once the frames from the row onward cover the horizon; for the k-step
    return, once ``k`` rows exist from it — and ``flush(final=True)`` sends the rest at the
    episode's end (a death forfeits the frames it did not live; a clear is credited at full
    speed past the end, exactly as offline)."""

    config: CreditConfig
    budget: RecordsBudget
    episode_id: str
    entries: list[_Entry] = field(default_factory=list)
    sent_records: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self, row: Mapping[str, Any], pending: Mapping[str, Any] | None, policy: Adapt1Policy | None
    ) -> None:
        self.entries.append(
            _Entry(row=dict(row), pending=dict(pending) if pending else None, policy=policy)
        )

    # -- the return of row t over the rows so far (final iff ready(t) or the episode ended) -----
    def _rewards(self) -> list[float]:
        return [float(e.row["reward"]) for e in self.entries]

    def _frames(self) -> list[int]:
        return [int(e.row["frames"]) for e in self.entries]

    def ready(self, t: int) -> bool:
        if self.config.kind == "frame_horizon":
            return sum(self._frames()[t:]) >= self.config.horizon
        return len(self.entries) - t >= self.config.k

    def value(self, t: int, *, final: bool) -> float:
        if self.config.kind == "frame_horizon":
            cleared = final and self.entries[-1].row.get("outcome") == "cleared"
            return frame_horizon_return(
                self._rewards(), self._frames(), t, self.config.horizon, cleared=cleared
            )
        return kstep_return(self._rewards(), t, self.config.k, self.config.gamma)

    def flush(self, *, final: bool = False) -> list[dict[str, Any]]:
        sent_now: list[dict[str, Any]] = []
        for t, entry in enumerate(self.entries):
            if entry.sent or entry.dropped:
                continue
            if not (final or self.ready(t)):
                break  # rows are in step order: nothing later is ready either
            reward = self.value(t, final=final)
            entry.reward_sent = reward
            if entry.pending is None or entry.policy is None:
                entry.dropped = True  # no decision id (client-side exploration with none issued)
                continue
            if not self.budget.take():
                entry.dropped = True
                continue
            names = entry.policy.feature_names
            next_features = {n: entry.row["next_state"][n] for n in names}
            record = entry.policy.send_feedback(
                entry.pending,
                next_state={},
                reward=reward,
                terminal=bool(entry.row["terminal"]),
                episode_id=self.episode_id,
                step=int(entry.row["step"]),
                next_features=next_features,
                outcome=str(entry.row["outcome"]),
            )
            entry.sent = True
            record = dict(record)
            record.update(
                kind=entry.row.get("kind"),
                immediate_reward=float(entry.row["reward"]),
                frames=int(entry.row["frames"]),
                domain=entry.policy.domain_id,
                x=entry.row.get("x"),
                x_after=entry.row.get("x_after"),
            )
            self.sent_records.append(record)
            sent_now.append(record)
        return sent_now

    @property
    def dropped(self) -> int:
        return sum(1 for e in self.entries if e.dropped)


# ---------------------------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    episode_id: str
    seed: int
    max_x: int
    decisions: int
    apex_looks: int
    cleared: bool
    dead: bool
    rows: int
    sent: int
    dropped: int
    records_spent: int
    takeoff_status: dict[str, int]
    apex_status: dict[str, int]
    takeoff_learner: dict[str, Any]
    apex_learner: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def play_episode(
    env: Any,
    *,
    takeoff: Adapt1Policy,
    apex: Adapt1Policy,
    macros: Sequence[Macro],
    credit: DelayedCredit,
    seed: int,
    max_decisions: int = 200,
    stall_timeout: int | None = 20,
    reward_config: RewardConfig = PROGRESS_FRACTION,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> EpisodeStats:
    """One learning episode: the takeoff domain chooses a macro at every grounded decision, the
    apex domain an ``ApexChoice`` at the top of every jump; each decision's feedback goes to
    ``credit`` and is sent when its return is known."""
    names = tuple(dict.fromkeys((*takeoff.feature_names, *apex.feature_names)))
    parser = CadenceParser(decision_horizon_frames=8)
    _frame, info = env.reset(seed=seed)
    takeoff.start_episode(credit.episode_id)
    apex.start_episode(credit.episode_id)
    snapshot = parser.parse(info, _unwrap_ram(env), previous_action=None)
    best_x = snapshot.x
    stalled = 0
    decisions = apex_looks = 0
    row_step = 0
    takeoff_learner: dict[str, Any] = {}
    apex_learner: dict[str, Any] = {}
    for _ in range(max_decisions):
        if snapshot.dead or snapshot.clear:
            break
        if stall_timeout is not None and stalled >= stall_timeout:
            break
        decision = takeoff.choose(snapshot, tuple(macros))
        pending_t = takeoff.take_pending()
        takeoff_learner = {
            k: (decision.telemetry or {}).get(k) for k in ("model_type", "sample_count")
        }
        apex_state: dict[str, Any] = {}
        execution = execute(
            env,
            decision.action,
            grounded=snapshot.grounded,
            ram=_unwrap_ram(env),
            apex=apex_hook(parser, env, apex, decision, apex_state),
        )
        fired = execution.apex is not None and bool(apex_state)
        pending_a = apex.take_pending() if fired else None
        if fired:
            apex_looks += 1
            tel = apex_state["decision"].telemetry or {}
            apex_learner = {k: tel.get(k) for k in ("model_type", "sample_count")}
        next_snapshot = parser.parse(
            execution.info,
            _unwrap_ram(env),
            previous_action=(
                apex_state["decision"].action.value if fired else decision.action.value
            ),
            previous_reward=(
                execution.total_reward - execution.reward_to_apex
                if fired
                else execution.total_reward
            ),
            frames=(execution.frames - execution.frames_to_apex if fired else execution.frames),
        )
        terminal = bool(execution.terminated or next_snapshot.dead or next_snapshot.clear)
        rows = build_rows(
            snapshot=snapshot,
            decision=decision,
            execution=execution,
            apex_state=apex_state if fired else None,
            next_snapshot=next_snapshot,
            terminal=terminal,
            names=names,
            reward_config=reward_config,
            episode_id=credit.episode_id,
            first_step=row_step,
            tag_kind=True,
        )
        for r in rows:  # where the decision was taken and where the macro landed (#49: slice the death zone offline)
            r["x"], r["x_after"] = int(snapshot.x), int(next_snapshot.x)
        credit.add(rows[0], pending_t, takeoff)
        if fired:
            credit.add(rows[1], pending_a, apex)
        row_step += len(rows)
        decisions += 1
        sent = credit.flush()
        if on_decision is not None:
            on_decision(
                {
                    "step": decisions,
                    "x": next_snapshot.x,
                    "macro": decision.action.value,
                    "status": decision.selection_status,
                    "apex": apex_state["decision"].action.value if fired else None,
                    "sent": len(sent),
                    "spent": credit.budget.spent,
                }
            )
        if next_snapshot.x > best_x:
            best_x, stalled = next_snapshot.x, 0
        else:
            stalled += 1
        snapshot = next_snapshot
        if terminal:
            break
    sent = credit.flush(final=True)
    return EpisodeStats(
        episode_id=credit.episode_id,
        seed=seed,
        max_x=int(best_x),
        decisions=decisions,
        apex_looks=apex_looks,
        cleared=bool(snapshot.clear),
        dead=bool(snapshot.dead),
        rows=len(credit.entries),
        sent=sum(1 for e in credit.entries if e.sent),
        dropped=credit.dropped,
        records_spent=credit.budget.spent,
        takeoff_status=dict(takeoff.status_counts),
        apex_status=dict(apex.status_counts),
        takeoff_learner=takeoff_learner,
        apex_learner=apex_learner,
    )


# ---------------------------------------------------------------------------------------------


def wait_ready(
    client: Any,
    domains: Sequence[str],
    *,
    timeout_s: float = 180.0,
    log: Callable[[str], None] = print,
) -> None:
    """Poll a bare status query on each domain until it answers 200 (Queries only). A fresh
    domain's engine answers 503 for about a minute after creation — longer than the client's
    retry schedule (findings #34)."""
    import time

    from .adapt1_client import Adapt1Error

    for dom in domains:
        started = time.monotonic()
        while True:
            try:
                client.query(dom, {"session_id": "ignored", "question": QUESTION, "top_k": 1})
                log(f"  {dom}: engine ready after {time.monotonic() - started:.0f}s")
                break
            except Adapt1Error as exc:
                if exc.status not in (502, 503, 504) or time.monotonic() - started > timeout_s:
                    raise
                time.sleep(10)


def wait_installed(
    client: Any,
    domains: Sequence[str],
    *,
    timeout_s: float = 600.0,
    poll_s: float = 15.0,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Before a frozen checkpoint: wait while a domain's learner reports a retrain in progress
    (``model_status: running`` with nothing installed). Findings #44: stage 2's ep-25 checkpoint was
    played in that window and measured the nearest-rows fallback, not the policy. A learner that has
    never trained (``idle``, fresh domain) or is ``trained`` is not waited for. Returns the last status
    per domain; logs a warning when the timeout passes with the retrain still running."""
    import time

    out: dict[str, dict[str, Any]] = {}
    for dom in domains:
        started = time.monotonic()
        while True:
            st = status_query(client, dom)
            out[dom] = st
            if st.get("installed") or st.get("model_status") != "running":
                break
            if time.monotonic() - started > timeout_s:
                log(
                    f"  {dom}: retrain still running after {timeout_s:.0f}s — evaluating anyway (checkpoint will read installed=false)"
                )
                break
            log(
                f"  {dom}: retrain running, no model installed — waiting {poll_s:.0f}s before the frozen eval"
            )
            time.sleep(poll_s)
    return out


def status_query(client: Any, domain_id: str) -> dict[str, Any]:
    """The learner's deployed model and sample counts (a Query, no Record).

    ``sample_count`` (from ``learner_status``) prefers ``report.sample_count`` — the training-set
    size at the last retrain; ``live_sample_count`` is ``learning_state.sample_counts.feedback_policy``,
    the counter that advances on every consumed feedback (findings #34: 29 live vs 24 report after
    29 feedbacks). Compare the live one with feedbacks sent — that is the #31 rollback check."""
    _s, qr = client.query(domain_id, {"session_id": "ignored", "question": QUESTION, "top_k": 1})
    st = learner_status(qr)
    ls = qr.get("learning_state", {}) if isinstance(qr, Mapping) else {}
    live = (
        (ls.get("sample_counts") or {}).get("feedback_policy") if isinstance(ls, Mapping) else None
    )
    st["live_sample_count"] = live
    return st


def frozen_eval(
    client: Any,
    *,
    takeoff_domain: str,
    apex_domain: str,
    takeoff_features: Sequence[str],
    apex_features: Sequence[str],
    macros: Sequence[Macro],
    env_id: str,
    seeds: Sequence[int],
    out_dir: Path,
    tag: str,
    max_decisions: int = 400,
    stall_timeout: int = 60,
) -> list[dict[str, Any]]:
    """Frozen evaluation of both domains (Queries only): ``allow_exploration`` false and
    ``selection_mode`` exploit, no feedback, one GIF per seed."""
    from .record import record_episode

    results = []
    for seed in seeds:
        t_pol = Adapt1Policy(
            client,
            takeoff_domain,
            allow_exploration=False,
            feature_names=takeoff_features,
            sequential=False,
            learn=False,
            run_id=f"{takeoff_domain}-frozen",
        )
        a_pol = Adapt1Policy(
            client,
            apex_domain,
            allow_exploration=False,
            feature_names=apex_features,
            sequential=False,
            learn=False,
            run_id=f"{apex_domain}-frozen",
        )
        gif = out_dir / f"{tag}-seed{seed}.gif"
        stats = record_episode(
            env_id,
            t_pol,
            gif,
            max_decisions=max_decisions,
            seed=seed,
            stall_timeout=stall_timeout,
            cadence="grounded",
            apex_policy=a_pol,
            actions=tuple(macros),
        )
        stats.update(
            seed=seed,
            tag=tag,
            gif=str(gif),
            takeoff_server=dict(t_pol.status_counts),
            apex_server=dict(a_pol.status_counts),
        )
        t_pol.close()
        a_pol.close()
        results.append(stats)
    return results


@dataclass
class RunConfig:
    takeoff_domain: str
    apex_domain: str
    env_id: str = "SuperMarioBros-1-1-v0"
    takeoff_features: tuple[str, ...] = FULL_V2_FEATURE_NAMES
    apex_features: tuple[str, ...] = APEX_V2_FEATURE_NAMES
    macros: tuple[Macro, ...] = MACROS_28
    episodes: int = 100
    seed: int = 0
    max_records: int = 2000
    credit: CreditConfig = field(default_factory=CreditConfig)
    epsilon: float = 0.0
    max_decisions: int = 200
    stall_timeout: int | None = 20
    eval_every: int = 25
    eval_seeds: tuple[int, ...] = (777,)
    stop_flat: int | None = (
        100  # stop when the best frozen reach has not improved for this many episodes
    )
    out_dir: Path = Path("artifacts/zero-start")
    run_id: str = "zero-start"
    max_samples: int = 4096  # learning.context.max_samples of the domains — the learner's retained-row budget (#46)


def zero_start_run(
    client: Any,
    cfg: RunConfig,
    *,
    log: Callable[[str], None] = print,
    on_run_log: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """The pilot: ``episodes`` learning episodes with a frozen eval every ``eval_every`` (and at
    the end), a run log (JSONL) of every episode and eval, the Records cap, and the stop rule.
    Returns the summary. ``on_run_log`` is called with the run-log path as soon as it is chosen
    (the CLI writes the provenance sidecar there, so a crash mid-run still leaves one)."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_log = cfg.out_dir / f"run-{stamp}.jsonl"
    if on_run_log is not None:
        on_run_log(run_log)
    budget = RecordsBudget(cfg.max_records)
    env = create_mario_env(cfg.env_id, render_mode="rgb_array")
    evals: list[dict[str, Any]] = []
    best_reach, best_reach_episode = -1, 0
    episodes_done = 0
    stop_reason = "episodes"
    sent_by_domain: dict[str, int] = {cfg.takeoff_domain: 0, cfg.apex_domain: 0}
    alerts: list[str] = []

    live_at_start: dict[str, int] = {}

    def evaluate(tag: str, episode: int) -> None:
        nonlocal best_reach, best_reach_episode
        before = wait_installed(client, (cfg.takeoff_domain, cfg.apex_domain), log=log)
        if (
            not live_at_start
        ):  # the ep000 checkpoint: what the domains already held (continuation stages, #44)
            for dom, st in before.items():
                if isinstance(st.get("live_sample_count"), int):
                    live_at_start[dom] = st["live_sample_count"]
        res = frozen_eval(
            client,
            takeoff_domain=cfg.takeoff_domain,
            apex_domain=cfg.apex_domain,
            takeoff_features=cfg.takeoff_features,
            apex_features=cfg.apex_features,
            macros=cfg.macros,
            env_id=cfg.env_id,
            seeds=cfg.eval_seeds,
            out_dir=cfg.out_dir / "frozen",
            tag=tag,
        )
        t_st, a_st = status_query(client, cfg.takeoff_domain), status_query(client, cfg.apex_domain)
        reach = max(int(r["max_x"]) for r in res)
        if reach > best_reach:
            best_reach, best_reach_episode = reach, episode
        # The two open questions to Rei (#30 §7, #31), watched rather than assumed answered:
        # consumption — every feedback sent must show in the LIVE counter (a rollback reads as
        # consumed < sent); model class — an mlp_v2 is a constant policy under a small reward range.
        checks = []
        for dom, st in ((cfg.takeoff_domain, t_st), (cfg.apex_domain, a_st)):
            # cumulative: a continuation stage starts with rows already consumed (live_at_start)
            live, sent = (
                st.get("live_sample_count"),
                sent_by_domain[dom] + live_at_start.get(dom, 0),
            )
            if isinstance(live, int) and live < sent:
                if live == cfg.max_samples:
                    checks.append(
                        f"SAMPLE CAP {dom}: learner holds {live} = context.max_samples while {sent} feedbacks were sent — "
                        f"replay eviction is dropping rows (#46); raise max_samples before continuing"
                    )
                else:
                    checks.append(
                        f"ROLLBACK? {dom}: live sample_count {live} < {sent} feedbacks sent (#31)"
                    )
            if str(st.get("model_type") or "").startswith("mlp"):
                checks.append(
                    f"MODEL CLASS {dom}: {st.get('model_type')} — a constant policy (#30 §7)"
                )
        alerts.extend(checks)
        entry = {
            "kind": "eval",
            "tag": tag,
            "episode": episode,
            "records_spent": budget.spent,
            "results": res,
            "reach": reach,
            "best_reach": best_reach,
            "takeoff_learner": t_st,
            "apex_learner": a_st,
            "learner_before_eval": before,  # what the checkpoint was played against (#44)
            "sent_by_domain": dict(sent_by_domain),
            "alerts": checks,
        }
        _b = before.get(cfg.takeoff_domain, {})
        if _b.get("model_status") == "running" and not _b.get("installed"):
            checks.append(
                f"NO MODEL at {tag}: takeoff learner retrain still running — checkpoint is void (#44)"
            )
            alerts.append(checks[-1])
        evals.append(entry)
        with run_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        log(
            f"[eval {tag}] reach={reach} (best {best_reach} @ep{best_reach_episode}) "
            f"flag={[r['reached_flag'] for r in res]} records={budget.spent} "
            f"takeoff model={t_st.get('model_type')} consumed={t_st.get('live_sample_count')}/{sent_by_domain[cfg.takeoff_domain]} "
            f"apex model={a_st.get('model_type')} consumed={a_st.get('live_sample_count')}/{sent_by_domain[cfg.apex_domain]} "
            f"server={[r['takeoff_server'] for r in res]}"
        )
        for line in checks:
            log(f"  !! {line}")

    try:
        evaluate("ep000", 0)
        for ep in range(cfg.episodes):
            seed = cfg.seed + ep
            episode_id = f"{cfg.run_id}-ep{ep:04d}"
            takeoff = Adapt1Policy(
                client,
                cfg.takeoff_domain,
                allow_exploration=True,
                rng_seed=seed,
                feature_names=cfg.takeoff_features,
                sequential=False,
                epsilon=cfg.epsilon,
                run_id=cfg.run_id,
                trace_path=cfg.out_dir / "trace-takeoff.jsonl" if ep == 0 else None,
            )
            apex = Adapt1Policy(
                client,
                cfg.apex_domain,
                allow_exploration=True,
                rng_seed=seed,
                feature_names=cfg.apex_features,
                sequential=False,
                epsilon=cfg.epsilon,
                run_id=cfg.run_id,
            )
            credit = DelayedCredit(cfg.credit, budget, episode_id)
            stats = play_episode(
                env,
                takeoff=takeoff,
                apex=apex,
                macros=cfg.macros,
                credit=credit,
                seed=seed,
                max_decisions=cfg.max_decisions,
                stall_timeout=cfg.stall_timeout,
            )
            takeoff.close()
            apex.close()
            for rec in credit.sent_records:
                sent_by_domain[rec["domain"]] = sent_by_domain.get(rec["domain"], 0) + 1
            episodes_done = ep + 1
            entry = {
                "kind": "episode",
                "episode": episodes_done,
                **stats.as_dict(),
                "feedback": credit.sent_records,
            }
            with run_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
            end = "FLAG" if stats.cleared else ("dead" if stats.dead else "stall")
            log(
                f"ep{episodes_done:04d} seed={seed} x={stats.max_x} {end} dec={stats.decisions} looks={stats.apex_looks} "
                f"sent={stats.sent}/{stats.rows} spent={budget.spent}/{cfg.max_records} "
                f"takeoff={stats.takeoff_status} apex={stats.apex_status} "
                f"model={stats.takeoff_learner.get('model_type')}/{stats.apex_learner.get('model_type')}"
            )
            if budget.remaining == 0:
                stop_reason = "records_cap"
                log(f"Records cap {cfg.max_records} reached — stopping")
                break
            if episodes_done % cfg.eval_every == 0 and episodes_done < cfg.episodes:
                evaluate(f"ep{episodes_done:03d}", episodes_done)
                if (
                    cfg.stop_flat is not None
                    and episodes_done - best_reach_episode >= cfg.stop_flat
                ):
                    stop_reason = "flat"
                    log(
                        f"no frozen-reach gain for {episodes_done - best_reach_episode} episodes — stopping"
                    )
                    break
        evaluate(f"ep{episodes_done:03d}-final", episodes_done)
    finally:
        env.close()
    summary = {
        "episodes": episodes_done,
        "records_spent": budget.spent,
        "best_reach": best_reach,
        "best_reach_episode": best_reach_episode,
        "stop_reason": stop_reason,
        "run_log": str(run_log),
        "sent_by_domain": dict(sent_by_domain),
        "alerts": alerts,
        "evals": [{k: e[k] for k in ("tag", "episode", "reach", "records_spent")} for e in evals],
    }
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "summary", **summary}, default=str) + "\n")
    return summary
