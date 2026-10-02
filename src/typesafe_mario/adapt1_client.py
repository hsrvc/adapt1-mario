"""Minimal Adapt-1 (Rei Labs NeuroAdapt) HTTP client for the Mario agent.

Design constraints:

- The API key lives in ``REI_KEY`` only. It is read lazily, never logged, never written
  to a file.
- A browser ``User-Agent`` is mandatory or Cloudflare returns ``403 / error 1010``.
- ``502/503/504`` are transient with possibly-ambiguous writes: honour ``Retry-After``
  first, then exponential backoff with jitter ``0.5 -> 1 -> 2 -> 4 -> 8`` s.
  ``413`` (payload) and ``409`` (state conflict) are not retryable.

The policy code depends only on the small duck-typed surface ``query(...)`` /
``feedback(...)`` / ``create_domain(...)``, so an offline fake can be substituted in
tests without any network.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://rei-neuroadapt-api.reilabs.org/api/v1"
BROWSER_UA = "Mozilla/5.0"
RETRYABLE_STATUS = {502, 503, 504}
BACKOFF_SCHEDULE = (0.5, 1.0, 2.0, 4.0, 8.0)


class Adapt1Error(RuntimeError):
    """Raised when a request fails after exhausting retries, or on a non-retryable error."""

    def __init__(self, status: int, body: Any, message: str | None = None) -> None:
        self.status = status
        self.body = body
        super().__init__(message or f"Adapt-1 request failed ({status}): {body!r}")


class Adapt1Client:
    """Thin JSON-in/JSON-out client with the specified retry behaviour."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_attempts: int = 6,
    ) -> None:
        self._base_url = (base_url or os.environ.get("REI_API_BASE") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        # Read lazily so importing the module never requires the key.
        self._explicit_key = api_key
        self._timeout = timeout
        self._max_attempts = max_attempts

    def _key(self) -> str:
        key = self._explicit_key or os.environ.get("REI_KEY")
        if not key:
            raise Adapt1Error(0, None, "REI_KEY is not set in the environment.")
        return key

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        """Issue one request, retrying transient 5xx per the backoff schedule.

        Returns ``(status, parsed_json)``. Raises :class:`Adapt1Error` on a
        non-retryable failure or after the retry budget is spent.
        """
        url = f"{self._base_url}{path}"
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._key()}",
            "User-Agent": BROWSER_UA,
            "Accept": "application/json",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"

        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            request = urllib.request.Request(url, data=payload, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return response.status, _read_json(response.read())
            except urllib.error.HTTPError as exc:
                status = exc.code
                raw = exc.read()
                parsed = _read_json(raw)
                if status not in RETRYABLE_STATUS:
                    # 413/409/4xx are not retryable — surface immediately.
                    raise Adapt1Error(status, parsed) from exc
                last_exc = exc
                retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
                _sleep_before_retry(attempt, retry_after)
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.HTTPException,
                ConnectionError,
            ) as exc:
                # Network-level hiccup — including the server closing the socket without a
                # response (`http.client.RemoteDisconnected`, seen mid-eval 2026-09-25; it is a
                # ConnectionResetError, not a URLError, so it used to escape the retry loop and
                # kill a frozen eval). Treat like a transient error.
                last_exc = exc
                _sleep_before_retry(attempt, None)

        raise Adapt1Error(
            503, None, f"Exhausted {self._max_attempts} attempts on {method} {path}"
        ) from last_exc

    # --- domain helpers -------------------------------------------------------

    def create_domain(self, config: dict[str, Any]) -> tuple[int, Any]:
        return self.call("POST", "/domains", config)

    def get_domain(self, domain_id: str) -> tuple[int, Any]:
        return self.call("GET", f"/domains/{domain_id}")

    def query(self, domain_id: str, body: dict[str, Any]) -> tuple[int, Any]:
        return self.call("POST", f"/domains/{domain_id}/query", body)

    def feedback(self, domain_id: str, body: dict[str, Any]) -> tuple[int, Any]:
        return self.call("POST", f"/domains/{domain_id}/feedback", body)

    def events(self, domain_id: str, body: dict[str, Any]) -> tuple[int, Any]:
        return self.call("POST", f"/domains/{domain_id}/events", body)


def _read_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _sleep_before_retry(attempt: int, retry_after: float | None) -> None:
    if retry_after is not None:
        time.sleep(retry_after)
        return
    base = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
    time.sleep(base + random.uniform(0.0, base * 0.25))
