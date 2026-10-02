"""The offline ingest contract (offline_client.py) — what a live ingest must satisfy, checked
for free. Uses the same domain config and feedback body shape as scripts/ingest_demos.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typesafe_mario.adapt1_policy import FEATURE_SETS, QUESTION, _parse_query, build_domain_config
from typesafe_mario.macros import MACRO_NAMES
from typesafe_mario.offline_client import OfflineAdapt1Client, OfflineContractError

FEATS = FEATURE_SETS["full"]


def _row(policy: str = "right_run_jump_mid", **over) -> dict:
    ctx = {f: 0.0 for f in FEATS}
    ctx.update({"jump_phase": "grounded", "reliability": "high", "outcome": "advanced"})
    row = {
        "episode_id": "demo-ep0000",
        "step": 3,
        "context": ctx,
        "policy": policy,
        "reward": 0.4,
        "next_state": dict(ctx),
        "terminal": False,
        "outcome": "advanced",
    }
    row.update(over)
    return row


def _feed(client: OfflineAdapt1Client, domain_id: str, row: dict) -> None:
    context = {"values": row["context"]}
    _s, qr = client.query(
        domain_id, {"session_id": "ignored", "question": QUESTION, "context": context, "top_k": 1}
    )
    client.feedback(
        domain_id,
        {
            "session_id": "ignored",
            "decision_id": _parse_query(qr)["decision_id"],
            "relation": "controls",
            "policy": row["policy"],
            "feedback_kind": "execution",
            "outcome": row["outcome"],
            "context": context,
            "values": {"reward": row["reward"], "next_state": row["next_state"]},
            "metadata": {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "terminal": row["terminal"],
            },
        },
    )


@pytest.fixture
def grounded_domain():
    client = OfflineAdapt1Client()
    client.create_domain(
        build_domain_config("d", feature_names=FEATS, macros=MACRO_NAMES, sequential=True)
    )
    return client


def test_valid_grounded_transition_is_accepted(grounded_domain):
    _feed(grounded_domain, "d", _row())
    assert grounded_domain.samples["d"] == 1


def test_frame_policy_on_a_grounded_domain_is_a_cadence_mismatch(grounded_domain):
    with pytest.raises(OfflineContractError, match="cadence mismatch"):
        _feed(grounded_domain, "d", _row(policy="right_run_jump"))  # an Action name, not a Macro


def test_string_terminal_is_rejected(grounded_domain):
    with pytest.raises(OfflineContractError, match="terminal"):
        _feed(grounded_domain, "d", _row(terminal="false"))


def test_feature_drift_is_rejected(grounded_domain):
    stale = _row()
    del stale["context"]["floor_below"]  # a dataset recorded before the floor-profile parser
    with pytest.raises(OfflineContractError, match="floor_below"):
        _feed(grounded_domain, "d", stale)


def test_next_state_must_be_an_object(grounded_domain):
    with pytest.raises(OfflineContractError, match="next_state"):
        _feed(grounded_domain, "d", _row(next_state=json.dumps(_row()["next_state"])))


def test_bandit_config_is_accepted_for_a_direct_feedback_warm_start():
    """findings #22: rewards that already carry the delayed credit (kstep_returns.py) go to a
    direct-feedback domain with no sequential block; the stub must admit it."""
    client = OfflineAdapt1Client()
    client.create_domain(build_domain_config("b", feature_names=FEATS, macros=MACRO_NAMES))
    _feed(client, "b", _row())
    assert client.samples["b"] == 1


def test_curated_grounded_dataset_passes_the_contract():
    path = Path("artifacts/demos/grounded-warmstart.jsonl")
    if not path.exists():
        pytest.skip("no curated grounded dataset recorded locally")
    client = OfflineAdapt1Client()
    client.create_domain(
        build_domain_config("w", feature_names=FEATS, macros=MACRO_NAMES, sequential=True)
    )
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for row in rows:
        _feed(client, "w", row)
    assert client.samples["w"] == len(rows)
    episodes = {r["episode_id"] for r in rows}
    assert len(episodes) >= 8  # minimum_episodes for the sequential model to install
    assert any(r["terminal"] and r["outcome"] == "cleared" for r in rows)  # a flag run is in
    assert any(r["terminal"] and r["outcome"] == "died" for r in rows)  # and a death (diversity)
