"""The credit counter: 1 per call, `value/observe` 1 per row, deletes free, everything else passed through."""

from __future__ import annotations

from typing import Any

from typesafe_mario.credit_meter import CreditMeter


class _Client:
    marker = "passed through"

    def query(self, domain_id: str, body: Any) -> tuple[int, Any]:
        return 200, {"domain": domain_id}

    def feedback(self, domain_id: str, body: Any) -> tuple[int, Any]:
        return 200, {}

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        return 200, {}


def test_counts_one_credit_per_call_and_per_observed_row() -> None:
    meter = CreditMeter(_Client())
    assert meter.query("d", {}) == (200, {"domain": "d"})
    meter.feedback("d", {})
    meter.feedback("d", {})
    meter.call("GET", "/domains/d/learning/status")
    meter.call("POST", "/domains/d/value/observe", {"observations": [{}, {}, {}]})
    meter.call("DELETE", "/domains/d")
    assert meter.credits == 7
    assert meter.by_kind == {
        "query": 1,
        "feedback": 2,
        "status": 1,
        "value_observe_rows": 3,
        "delete": 0,
    }
    assert meter.marker == "passed through"
