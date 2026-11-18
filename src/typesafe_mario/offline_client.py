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
        self.values: dict[str, dict[str, Any]] = {}  # value learner per domain
        self.value_skill = (
            0.5  # the validation skill a fake fit reports (tests lower it to exercise the gate)
        )

    # --- the subset of Adapt1Client the scripts use ------------------------------------

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        parts = path.strip("/").split("/")
        if method == "DELETE" and path.startswith("/domains/"):
            self.domains.pop(parts[1], None)
            self.values.pop(parts[1], None)
            return 200, {}
        if method == "GET" and len(parts) == 2 and parts[0] == "domains":
            if parts[1] not in self.domains:
                from .adapt1_client import Adapt1Error

                raise Adapt1Error(404, {"detail": f"domain not found: {parts[1]}"})
            return 200, {"domain_id": parts[1], **(self.domains[parts[1]].get("config") or {})}
        if method == "POST" and len(parts) >= 4 and parts[0] == "domains" and parts[2] == "value":
            return self._value(parts[1], parts[3], body)
        if method == "POST" and parts[2:] == ["planning", "adversarial"]:
            return self._plan(parts[1], body)
        raise OfflineContractError(f"offline client does not implement {method} {path}")

    # --- value learner + native planning (measured shapes, findings #75) ----------------------

    def _value(self, domain_id: str, op: str, body: Any) -> tuple[int, Any]:
        self._domain(domain_id)
        self._require_session(body or {})
        v = self.values.setdefault(
            domain_id, {"config": None, "rows": {}, "model": None, "training": None}
        )
        if op == "state":
            if v["config"] is None:
                return 200, {"domain_id": domain_id, "status": "not_configured", "sample_count": 0}
            return 200, {
                "domain_id": domain_id,
                "status": "configured",
                "config": v["config"],
                "sample_count": len(v["rows"]),
                "model": v["model"],
                "training": v["training"],
                "frozen": False,
            }
        if op == "configure":
            cfg = (body or {}).get("config") or {}
            if not (isinstance(cfg.get("relation"), str) and isinstance(cfg.get("policy"), str)):
                raise OfflineContractError(
                    "value/configure needs config.relation and config.policy"
                )
            v["config"] = dict(cfg)
            return 200, {"domain_id": domain_id, "status": "configured", "config": v["config"]}
        if v["config"] is None:
            raise OfflineContractError(f"value/{op} before value/configure")
        if op == "observe":
            rows = (body or {}).get("observations") or []
            if not 1 <= len(rows) <= 50:
                raise OfflineContractError("value/observe takes 1-50 observations")
            for r in rows:
                for key in ("record_id", "episode_id", "step"):
                    if key not in r:
                        raise OfflineContractError(f"value observation missing {key}")
                if sum(k in r for k in ("target", "outcome", "continuation")) != 1:
                    raise OfflineContractError(
                        "value observation needs exactly one of target/outcome/continuation"
                    )
                pf = r.get("policy_features") or {}
                if not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in pf.values()
                ):
                    raise OfflineContractError(
                        "value policy_features must be numeric (encode categories as one-hots)"
                    )
                v["rows"][r["record_id"]] = r
            return 200, {"accepted": len(rows), "duplicates": 0}
        if op == "train":
            if not (body or {}).get("request_id"):
                raise OfflineContractError("value/train needs request_id")
            version = len(v["rows"])
            v["model"] = {
                "model_type": "mlp_v2",
                "model_version": version,
                "model_sha256": f"{version:064x}",
                "validation_skill": self.value_skill,
                "version": version,
            }
            v["training"] = {
                "status": "trained",
                "request_id": body["request_id"],
                "installed": True,
            }
            return 202, {"domain_id": domain_id, "training": v["training"]}
        if op == "predict":
            return 200, {
                "predictions": [
                    self._fake_value(i.get("policy_features") or {})
                    for i in (body or {}).get("inputs") or []
                ]
            }
        raise OfflineContractError(f"offline client does not implement value/{op}")

    @staticmethod
    def _fake_value(features: Mapping[str, Any]) -> float:
        # a stand-in, not a learner: more open road ahead and more speed look better
        return (
            float(features.get("clear_forward", 0.0)) * 0.01
            + float(features.get("speed_x", 0.0)) * 0.001
        )

    def _plan(self, domain_id: str, body: Any) -> tuple[int, Any]:
        self._domain(domain_id)
        self._require_session(body or {})
        for key in ("relation", "policy", "root"):
            if key not in (body or {}):
                raise OfflineContractError(f"planning request missing {key}")
        v = self.values.get(domain_id) or {}
        if body.get("evaluator_source") == "value_model" and not v.get("model"):
            raise OfflineContractError(
                "planning with value_model before a value model is installed (live: 409)"
            )
        cfg = v.get("config") or {}
        if (body["relation"], body["policy"]) != (cfg.get("relation"), cfg.get("policy")):
            raise OfflineContractError("planning relation/policy must match the value config")
        nodes = {n["id"]: n for n in body.get("nodes") or []}
        root = nodes.get(body["root"]["id"])
        if root is None or not root.get("children"):
            raise OfflineContractError(
                "the root must link to the graph through a node with the same id and children"
            )
        best_action, best_value = None, -math.inf
        for edge in root["children"]:
            node = nodes.get(edge["target"])
            if node is None:
                raise OfflineContractError(f"edge target {edge['target']} is not a node")
            val = (
                node["terminal_value"]
                if node.get("terminal_value") is not None
                else self._fake_value(node.get("policy_features") or {})
            )
            if val > best_value:
                best_action, best_value = edge["action"], val
        return 200, {
            "plan": {
                "action": best_action,
                "value": best_value,
                "completed_depth": 1,
                "evaluations": len(root["children"]),
            },
            "evaluator": {
                "source": "value_model",
                "model_version": (v.get("model") or {}).get("model_version"),
                "learning_updates": False,
            },
        }

    def version(self) -> Any:
        return {"commit_sha": "offline", "schema_version": "offline", "version": "offline"}

    def create_domain(self, config: Mapping[str, Any]) -> tuple[int, Any]:
        domain_id = config["domain_id"]
        learning = config.get("learning") or {}
        if not learning and not config.get("hypotheses"):
            # a plain domain (e.g. one that hosts only a value learner): the live API creates it with defaults
            self.domains[domain_id] = {
                "config": json.loads(json.dumps(config)),
                "features": [],
                "policies": set(),
                "outcomes": set(),
                "seq": None,
                "tcp": False,
                "seen_steps": set(),
            }
            self.samples[domain_id] = 0
            self.feedback_log[domain_id] = []
            return 201, {"domain_id": domain_id}
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
        # two routes: a decision id from a query on this domain, or no id and the context in the request
        # (the docs' "replay historical transitions with explicit context")
        explicit = decision_id is None and body.get("context") is not None
        if not explicit and self.decisions.get(decision_id) != domain_id:
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
                "context_source": "request" if explicit else "decision",
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
