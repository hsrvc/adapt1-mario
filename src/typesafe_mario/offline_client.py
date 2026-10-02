"""An offline stand-in for ``Adapt1Client`` that validates request shapes — no API, no Records.

Every live ingest spends 1 Record per transition and deletes never refund, so a
malformed body discovered on the server is money gone. This stub enforces, locally, the
contract the live API has been measured to require (building-on-adapt1.md §3c, findings #15):

* the domain config carries ``learning.sequential`` with the declared paths and the
  hypotheses' ``policy`` names are the executed-macro vocabulary;
* every feedback binds to a ``decision_id`` this stub issued, names a declared ``policy`` and a
  declared ``outcome``, carries ``values.reward`` (finite number), ``values.next_state`` (a
  JSON **object**, not a string), boolean ``metadata.terminal`` and a numeric ``metadata.step``;
* context keys match the domain's ``feature_paths`` exactly, on the query and the feedback;
* ``session_id`` is present (the spec lists it as required on several bodies).

It returns responses shaped like the live ones closely enough for ``_parse_query`` /
``learner_status`` (a ``decision_id``, an abstained selection, a growing
``learning_state.sample_counts.feedback_policy``), so the ingest and training scripts run
end-to-end against it. It *does not* learn anything.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping
from typing import Any


class OfflineContractError(ValueError):
    """A request the live API would reject (or silently ignore) — fix it before spending."""


class OfflineAdapt1Client:
    def __init__(self) -> None:
        self.domains: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, str] = {}  # decision_id -> domain_id
        self.samples: dict[str, int] = {}
        self.feedback_log: dict[str, list[dict[str, Any]]] = {}

    # --- the subset of Adapt1Client the scripts use ------------------------------------

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        if method == "DELETE" and path.startswith("/domains/"):
            self.domains.pop(path.split("/")[2], None)
            return 200, {}
        raise OfflineContractError(f"offline client does not implement {method} {path}")

    def create_domain(self, config: Mapping[str, Any]) -> tuple[int, Any]:
        domain_id = config["domain_id"]
        learning = config.get("learning") or {}
        seq = learning.get("sequential")
        if seq and seq.get("enabled"):
            for key in (
                "episode_path",
                "step_path",
                "next_context_path",
                "reward_path",
                "terminal_path",
            ):
                if key not in seq:
                    raise OfflineContractError(f"learning.sequential.{key} missing")
        else:
            seq = None  # direct-feedback (bandit) domain: rewards must already carry any delayed credit
        paths = (learning.get("context") or {}).get("feature_paths") or []
        if not paths:
            raise OfflineContractError("learning.context.feature_paths is empty")
        if (learning.get("context") or {}).get("max_samples", 512) < 2048:
            raise OfflineContractError("learning.context.max_samples must be >= 2048 (512 evicts)")
        policies = {h["policy"] for h in config.get("hypotheses", [])}
        if not policies:
            raise OfflineContractError("no hypotheses")
        outcomes = set((config.get("query_templates") or {}).get("feedback_outcomes") or [])
        tcp = learning.get("temporal_context") or {}
        if tcp.get("enabled"):
            # temporal-context-projection.md: input paths must be numeric scalars; every query
            # must carry (episode_id, step) in the context envelope.
            string_features = {"jump_phase", "reliability", "outcome"}
            bad = [
                p
                for p in tcp.get("input_paths", [])
                if p.removeprefix("values.") in string_features
            ]
            if bad:
                raise OfflineContractError(
                    f"temporal_context.input_paths must be numeric scalars: {bad}"
                )
            if not tcp.get("input_paths"):
                raise OfflineContractError("temporal_context.input_paths is empty")
        self.domains[domain_id] = {
            "config": json.loads(json.dumps(config)),
            "features": [p.removeprefix("values.") for p in paths],
            "policies": policies,
            "outcomes": outcomes,
            "seq": seq,
            "tcp": bool(tcp.get("enabled")),
            "seen_steps": set(),
        }
        self.samples[domain_id] = 0
        self.feedback_log[domain_id] = []
        return 201, {"domain_id": domain_id, "learning": learning}

    def query(self, domain_id: str, body: Mapping[str, Any]) -> tuple[int, Any]:
        dom = self._domain(domain_id)
        self._require_session(body)
        context = body.get("context")
        if context is not None:
            self._check_context(dom, context, "query.context")
        if (
            dom["tcp"] and context is not None
        ):  # decision queries only; a bare status query has no context
            meta = context.get("metadata") if isinstance(context, Mapping) else None
            if (
                not isinstance(meta, Mapping)
                or not isinstance(meta.get("episode_id"), str)
                or not isinstance(meta.get("step"), (int, float))
                or isinstance(meta.get("step"), bool)
            ):
                raise OfflineContractError(
                    "TCP domain: query.context.metadata must carry a string episode_id and a numeric step"
                )
            key = (meta["episode_id"], float(meta["step"]))
            if key in dom["seen_steps"]:
                raise OfflineContractError(
                    f"TCP dedup: (episode_id, step) {key} reused — the server drops it"
                )
            dom["seen_steps"].add(key)
        decision_id = str(uuid.uuid4())
        self.decisions[decision_id] = domain_id
        return 200, {
            "decision_id": decision_id,
            "selection": {"status": "abstained", "reason": "offline_stub"},
            "policy_scores": [],
            "learning_state": {
                "sample_counts": {"feedback_policy": self.samples[domain_id]},
                "subsystems": {"feedback_policy": {"model": {"installed": False, "report": {}}}},
            },
        }

    def feedback(self, domain_id: str, body: Mapping[str, Any]) -> tuple[int, Any]:
        dom = self._domain(domain_id)
        self._require_session(body)
        decision_id = body.get("decision_id")
        if self.decisions.get(decision_id) != domain_id:
            raise OfflineContractError(
                "feedback.decision_id was not issued by a query on this domain"
            )
        if body.get("policy") not in dom["policies"]:
            raise OfflineContractError(
                f"feedback.policy {body.get('policy')!r} is not a declared hypothesis "
                f"(cadence mismatch? declared: {sorted(dom['policies'])})"
            )
        if dom["outcomes"] and body.get("outcome") not in dom["outcomes"]:
            raise OfflineContractError(f"feedback.outcome {body.get('outcome')!r} not declared")
        self._check_context(dom, body.get("context"), "feedback.context")
        values = body.get("values") or {}
        reward = values.get("reward")
        if (
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(reward)
        ):
            raise OfflineContractError("values.reward must be a finite number")
        next_state = values.get("next_state")
        if not isinstance(next_state, Mapping):
            raise OfflineContractError(
                "values.next_state must be a JSON object (not a string/None)"
            )
        if set(next_state) != set(dom["features"]):
            raise OfflineContractError("values.next_state keys must equal the domain feature_paths")
        meta = body.get("metadata") or {}
        if not isinstance(meta.get("terminal"), bool):
            raise OfflineContractError("metadata.terminal must be a JSON boolean")
        if not isinstance(meta.get("step"), (int, float)) or isinstance(meta.get("step"), bool):
            raise OfflineContractError("metadata.step must be a number")
        if not isinstance(meta.get("episode_id"), str) or not meta["episode_id"]:
            raise OfflineContractError("metadata.episode_id must be a non-empty string")
        self.samples[domain_id] += 1
        self.feedback_log[domain_id].append(json.loads(json.dumps(body)))
        return 200, {
            "credit_assignment": {
                "contextual_learning_applied": True,
                "context_source": "decision",
            },
            "policy_scores": [],
        }

    # --- helpers ----------------------------------------------------------------------

    def _domain(self, domain_id: str) -> dict[str, Any]:
        if domain_id not in self.domains:
            raise OfflineContractError(f"domain {domain_id!r} does not exist (create it first)")
        return self.domains[domain_id]

    @staticmethod
    def _require_session(body: Mapping[str, Any]) -> None:
        if "session_id" not in body:
            raise OfflineContractError("session_id missing (spec lists it as required)")

    @staticmethod
    def _check_context(dom: Mapping[str, Any], context: Any, where: str) -> None:
        values = (context or {}).get("values") if isinstance(context, Mapping) else None
        if not isinstance(values, Mapping):
            raise OfflineContractError(f"{where}.values must be an object")
        expected = set(dom["features"])
        if set(values) != expected:
            missing, extra = expected - set(values), set(values) - expected
            raise OfflineContractError(
                f"{where}.values keys differ from feature_paths (missing {sorted(missing)}, "
                f"extra {sorted(extra)}) — feature arm / parser version mismatch"
            )
        for key, value in values.items():
            if value is None or isinstance(value, (list, dict)):
                raise OfflineContractError(f"{where}.values.{key} must be a scalar, got {value!r}")
