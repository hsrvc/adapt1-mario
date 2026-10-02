"""online_duo (2026-09-24): the online duo-domain loop with client-side delayed credit.

The return a decision's feedback carries must equal what the offline scripts compute on the same
rows (returns.frame_horizon_returns / kstep_returns), the feedback must land in the right domain
under the contract offline_client.py enforces, and the Records cap must hold.
"""

from __future__ import annotations

import pytest

from typesafe_mario.adapt1_policy import (
    APEX_V2_FEATURE_NAMES,
    FULL_V2_FEATURE_NAMES,
    build_domain_config,
)
from typesafe_mario.macros import APEX_NAMES
from typesafe_mario.offline_client import OfflineAdapt1Client
from typesafe_mario.online_duo import (
    MACROS_28,
    CreditConfig,
    DelayedCredit,
    RecordsBudget,
)
from typesafe_mario.returns import frame_horizon_returns, kstep_returns


class _Sink:
    """Stands in for Adapt1Policy: records what send_feedback was asked to send."""

    def __init__(self, name: str, names=("a", "b")):
        self.domain_id = name
        self.feature_names = tuple(names)
        self.sent: list[dict] = []

    def send_feedback(
        self,
        pending,
        *,
        next_state,
        reward,
        terminal,
        episode_id,
        step,
        next_features=None,
        outcome=None,
        **kw,
    ):
        rec = {
            "decision_id": pending["decision_id"],
            "reward": reward,
            "terminal": terminal,
            "step": step,
            "next_features": next_features,
            "outcome": outcome,
        }
        self.sent.append(rec)
        return rec


def _rows(frames_seq, rewards_seq, *, last_outcome="died"):
    rows = []
    for i, (f, r) in enumerate(zip(frames_seq, rewards_seq)):
        rows.append(
            {
                "episode_id": "ep",
                "step": i,
                "reward": r,
                "frames": f,
                "terminal": i == len(frames_seq) - 1,
                "outcome": last_outcome if i == len(frames_seq) - 1 else "advanced",
                "next_state": {"a": 1.0, "b": 2.0, "c": 3.0},
                "kind": "takeoff",
            }
        )
    return rows


def _pending(i):
    return {"decision_id": f"d{i}", "policy": "right_run", "features": {"a": 0.0, "b": 0.0}}


@pytest.mark.parametrize("last_outcome", ["died", "cleared"])
def test_frame_horizon_credit_online_equals_offline(last_outcome):
    frames = [8, 8, 51, 8, 27, 8, 8, 8, 8, 42, 8, 8, 8]
    rewards = [
        0.0076,
        0.006,
        0.04,
        0.0076,
        0.02,
        0.0076,
        0.0,
        0.0076,
        0.0076,
        0.03,
        0.0076,
        0.0076,
        0.002,
    ]
    rows = _rows(frames, rewards, last_outcome=last_outcome)
    offline = frame_horizon_returns(rows, 128)
    sink = _Sink("takeoff")
    credit = DelayedCredit(
        CreditConfig(kind="frame_horizon", horizon=128), RecordsBudget(100), "ep"
    )
    sent_before_final = 0
    for t, row in enumerate(rows):
        credit.add(row, _pending(t), sink)
        sent_before_final += len(credit.flush())
    assert 0 < sent_before_final < len(rows)  # some rows wait for the horizon, none waits forever
    credit.flush(final=True)
    assert [r["decision_id"] for r in sink.sent] == [
        f"d{t}" for t in range(len(rows))
    ]  # step order
    assert [r["reward"] for r in sink.sent] == [o["reward_rate"] for o in offline]
    assert sink.sent[-1]["terminal"] is True and sink.sent[0]["terminal"] is False
    assert sink.sent[0]["next_features"] == {"a": 1.0, "b": 2.0}  # projected to the domain's arm
    assert sink.sent[-1]["outcome"] == last_outcome


def test_frame_horizon_rows_are_held_until_the_horizon_is_covered():
    rows = _rows([8] * 20, [0.007] * 20)
    sink = _Sink("takeoff")
    credit = DelayedCredit(CreditConfig(horizon=128), RecordsBudget(100), "ep")
    sent_at = {}
    for t, row in enumerate(rows):
        credit.add(row, _pending(t), sink)
        for rec in credit.flush():
            sent_at[rec["decision_id"]] = t
    # row 0 is sent once rows 0..15 (16 × 8 = 128 frames) exist, i.e. when row 15 is added
    assert sent_at["d0"] == 15 and sent_at["d4"] == 19
    assert "d5" not in sent_at  # rows 5..19 wait for the episode's end
    credit.flush(final=True)
    assert len(sink.sent) == 20


def test_kstep_credit_online_equals_offline():
    rows = _rows([8] * 11, [0.01, 0.0, 0.02, 0.01, 0.0, 0.03, 0.01, 0.01, 0.0, 0.02, 0.0])
    offline = kstep_returns(rows, 6, 0.9)
    sink = _Sink("takeoff")
    credit = DelayedCredit(CreditConfig(kind="kstep", k=6, gamma=0.9), RecordsBudget(100), "ep")
    for t, row in enumerate(rows):
        credit.add(row, _pending(t), sink)
        credit.flush()
    assert len(sink.sent) == 6  # rows 0..5 have 6 rows from them; 6..10 wait
    credit.flush(final=True)
    assert [r["reward"] for r in sink.sent] == [o["reward"] for o in offline]


def test_records_cap_drops_instead_of_sending():
    rows = _rows([8] * 5, [0.01] * 5)
    sink = _Sink("takeoff")
    budget = RecordsBudget(3)
    credit = DelayedCredit(CreditConfig(), budget, "ep")
    for t, row in enumerate(rows):
        credit.add(row, _pending(t), sink)
    credit.flush(final=True)
    assert (
        len(sink.sent) == 3 and credit.dropped == 2 and budget.spent == 3 and budget.remaining == 0
    )


def test_rows_of_both_domains_interleave_in_step_order():
    """A jump = a takeoff row then an apex row; each goes to its own domain, both get the return
    over the same following rows."""
    takeoff, apex = _Sink("takeoff"), _Sink("apex", names=("a", "c"))
    credit = DelayedCredit(CreditConfig(horizon=128), RecordsBudget(100), "ep")
    frames = [8, 20, 31, 8, 8]  # run, takeoff→apex, apex→landing, run, run
    rows = _rows(frames, [0.007, 0.02, 0.03, 0.007, 0.007])
    owners = [takeoff, takeoff, apex, takeoff, takeoff]
    rows[1]["kind"], rows[2]["kind"] = "takeoff", "apex"
    for t, (row, owner) in enumerate(zip(rows, owners)):
        credit.add(row, _pending(t), owner)
    credit.flush(final=True)
    offline = frame_horizon_returns(rows, 128)
    assert [r["decision_id"] for r in takeoff.sent] == ["d0", "d1", "d3", "d4"]
    assert [r["decision_id"] for r in apex.sent] == ["d2"]
    assert apex.sent[0]["reward"] == offline[2]["reward_rate"]
    assert apex.sent[0]["next_features"] == {"a": 1.0, "c": 3.0}


# ---------------------------------------------------------------------------------------------
# The whole loop against the offline contract client (the emulator, 0 Records, no network).

pytest.importorskip("gym_super_mario_bros")

from typesafe_mario.adapt1_policy import Adapt1Policy
from typesafe_mario.online_duo import RunConfig, play_episode, zero_start_run
from typesafe_mario.runner import create_mario_env


def _offline_domains():
    client = OfflineAdapt1Client()
    client.create_domain(
        build_domain_config(
            "t",
            feature_names=FULL_V2_FEATURE_NAMES,
            macros=[m.value for m in MACROS_28],
            sequential=False,
        )
    )
    client.create_domain(
        build_domain_config(
            "a", feature_names=APEX_V2_FEATURE_NAMES, macros=list(APEX_NAMES), sequential=False
        )
    )
    return client


def test_one_episode_sends_every_row_to_its_domain_under_the_contract():
    client = _offline_domains()
    env = create_mario_env("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    try:
        takeoff = Adapt1Policy(
            client,
            "t",
            feature_names=FULL_V2_FEATURE_NAMES,
            sequential=False,
            epsilon=0.5,
            rng_seed=3,
        )
        apex = Adapt1Policy(client, "a", feature_names=APEX_V2_FEATURE_NAMES, sequential=False)
        credit = DelayedCredit(CreditConfig(), RecordsBudget(500), "ep-test")
        stats = play_episode(
            env,
            takeoff=takeoff,
            apex=apex,
            macros=MACROS_28,
            credit=credit,
            seed=777,
            max_decisions=40,
            stall_timeout=10,
        )
    finally:
        env.close()
    assert stats.decisions >= 5 and stats.rows == stats.decisions + stats.apex_looks
    assert stats.sent == stats.rows and stats.dropped == 0 and stats.records_spent == stats.rows
    t_log, a_log = client.feedback_log["t"], client.feedback_log["a"]
    assert len(t_log) == stats.decisions and len(a_log) == stats.apex_looks
    if stats.apex_looks:
        assert set(a_log[0]["values"]["next_state"]) == set(APEX_V2_FEATURE_NAMES)
        assert a_log[0]["policy"] in APEX_NAMES
    assert set(t_log[0]["values"]["next_state"]) == set(FULL_V2_FEATURE_NAMES)
    assert set(t_log[0]["context"]["values"]) == set(FULL_V2_FEATURE_NAMES)
    assert all(0.0 <= f["values"]["reward"] <= 1.0 for f in t_log + a_log)
    assert all(f["policy"] in {m.value for m in MACROS_28} for f in t_log)
    assert [f["metadata"]["step"] for f in t_log] == sorted(f["metadata"]["step"] for f in t_log)
    assert (
        t_log[-1]["metadata"]["terminal"] is True
        or stats.decisions == 40
        or not (stats.dead or stats.cleared)
    )
    ids = [f["decision_id"] for f in t_log + a_log]
    assert len(ids) == len(set(ids))


def test_zero_start_run_respects_the_cap_and_logs(tmp_path):
    client = _offline_domains()
    cfg = RunConfig(
        takeoff_domain="t",
        apex_domain="a",
        episodes=3,
        max_records=25,
        eval_every=2,
        eval_seeds=(777,),
        stop_flat=None,
        out_dir=tmp_path,
        run_id="test",
        max_decisions=15,
        stall_timeout=5,
    )
    lines: list[str] = []
    summary = zero_start_run(client, cfg, log=lines.append)
    assert summary["records_spent"] <= 25
    assert summary["stop_reason"] in ("records_cap", "episodes")
    assert client.samples["t"] + client.samples["a"] == summary["records_spent"]
    assert summary["sent_by_domain"] == {"t": client.samples["t"], "a": client.samples["a"]}
    assert summary["alerts"] == []  # the stub consumes everything and installs no model
    assert any(line.startswith("[eval ep000]") for line in lines)
    assert (tmp_path / "frozen").exists()
    import json

    with open(summary["run_log"]) as fh:
        entries = [json.loads(line) for line in fh]
    assert entries[0]["kind"] == "eval" and entries[-1]["kind"] == "summary"
    assert any(e["kind"] == "episode" for e in entries)


def test_zero_start_run_reports_the_run_log_before_it_starts(tmp_path):
    """The CLI writes the provenance sidecar as soon as the run log is named, so a crash mid-run
    still leaves one; the hook receives the same path the summary reports."""
    client = _offline_domains()
    cfg = RunConfig(
        takeoff_domain="t",
        apex_domain="a",
        episodes=1,
        max_records=5,
        eval_every=0,
        eval_seeds=(777,),
        stop_flat=None,
        out_dir=tmp_path,
        run_id="test",
        max_decisions=5,
        stall_timeout=5,
    )
    seen: list = []
    summary = zero_start_run(client, cfg, log=lambda _s: None, on_run_log=seen.append)
    assert seen and str(seen[0]) == summary["run_log"]


def test_wait_installed_waits_out_a_running_retrain_and_not_an_idle_learner(monkeypatch):
    """#44: a checkpoint played while `model_status: running` with nothing installed measures the
    nearest-rows fallback. The wait polls until installed; an idle (never trained) learner is not waited for."""
    from typesafe_mario import online_duo

    calls = {"n": 0}
    seq = [
        {"installed": False, "model_status": "running"},
        {"installed": False, "model_status": "running"},
        {"installed": True, "model_status": "trained", "model_type": "extra_trees"},
    ]

    def fake_status(client, dom):
        calls["n"] += 1
        return (
            seq[min(calls["n"] - 1, len(seq) - 1)]
            if dom == "t"
            else {"installed": False, "model_status": "idle"}
        )

    monkeypatch.setattr(online_duo, "status_query", fake_status)
    monkeypatch.setattr(online_duo.time, "sleep", lambda s: None) if hasattr(
        online_duo, "time"
    ) else None
    import time as _t

    monkeypatch.setattr(_t, "sleep", lambda s: None)
    lines: list[str] = []
    out = online_duo.wait_installed(object(), ("t", "a"), poll_s=0, log=lines.append)
    assert out["t"]["installed"] is True and out["a"]["model_status"] == "idle"
    assert calls["n"] == 4  # three polls on t, one on a
    assert any("retrain running" in line for line in lines)


def test_checkpoint_flags_the_sample_cap_and_counts_rows_cumulatively(tmp_path, monkeypatch):
    """#46: a learner whose live sample_count sits at context.max_samples while more rows were sent is evicting,
    not rolling back; a continuation stage counts the rows the domain already held at its ep000 checkpoint."""
    from typesafe_mario import online_duo

    client = _offline_domains()
    cfg = RunConfig(
        takeoff_domain="t",
        apex_domain="a",
        episodes=2,
        max_records=30,
        eval_every=1,
        eval_seeds=(777,),
        stop_flat=None,
        out_dir=tmp_path,
        run_id="test",
        max_decisions=8,
        stall_timeout=5,
        max_samples=10,
    )
    calls = {"n": 0}
    real = online_duo.status_query

    def capped(c, dom):
        st = real(c, dom)
        calls["n"] += 1
        if dom == "t":
            st["live_sample_count"] = 10  # the ep000 read and every later read: stuck at the cap
            st["installed"], st["model_status"] = True, "trained"
        return st

    monkeypatch.setattr(online_duo, "status_query", capped)
    summary = zero_start_run(client, cfg, log=lambda _s: None)
    assert any(a.startswith("SAMPLE CAP t") for a in summary["alerts"]), summary["alerts"]
    assert not any(a.startswith("ROLLBACK") for a in summary["alerts"])
