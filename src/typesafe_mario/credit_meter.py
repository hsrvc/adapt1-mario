"""A client wrapper that counts the credits a run spends (the Adapt-1 API has no balance route)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CreditMeter:
    """Counts credits as the account dashboard does: 1 per call, `value/observe` 1 per row. Transport retries inside
    the client are not counted (whether a 503 costs a credit is unmeasured). Wraps a client transparently."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.credits = 0
        self.by_kind: dict[str, int] = {}

    def _add(self, kind: str, n: int = 1) -> None:
        self.credits += n
        self.by_kind[kind] = self.by_kind.get(kind, 0) + n

    def query(self, domain_id: str, body: Any) -> tuple[int, Any]:
        self._add("query")
        return self._client.query(domain_id, body)

    def feedback(self, domain_id: str, body: Any) -> tuple[int, Any]:
        self._add("feedback")
        return self._client.feedback(domain_id, body)

    def create_domain(self, config: Any) -> tuple[int, Any]:
        self._add("create_domain")
        return self._client.create_domain(config)

    def version(self) -> Any:
        self._add("version")
        return self._client.version()

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        if method == "DELETE":
            kind, n = (
                "delete",
                0,
            )  # deletes did not show on the meter in September 2026
        elif path.endswith("/value/observe") and isinstance(body, Mapping):
            kind, n = "value_observe_rows", len(body.get("observations") or [])
        else:
            kind, n = path.rsplit("/", 1)[-1] or "call", 1
        self._add(kind, n)
        return self._client.call(method, path, body)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
