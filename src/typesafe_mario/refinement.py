"""World 2-1 by REFINING a frozen 1-1 policy, not relearning it (a proposal by Rei Labs, 2026-10-02).

At every decision:

1. the frozen base domain (``mario-zero2-takeoff`` / ``-apex``) picks its move — exploit, no exploration, and it
   NEVER receives feedback;
2. a corrector domain sees the base's own context + the base's move (one-hot) + a few interaction terms, and
   picks ``keep_base`` or ``override_<move>``;
3. the executed move is the base's (``keep_base``) or the override; only the corrector gets the delayed credit.

What is deliberately NOT here (checked against the API first): no normalization and no squared terms (the pinned
``extra_trees`` learner is invariant to both), and nothing from Machina's refinement config (``context_basis``,
``response_guidance`` are trajectory-domain options; the domain ``learning`` block has none of them). The
interaction terms are client-side features, and every write-up must say so.

``RefinedPolicy`` duck-types ``Adapt1Policy`` (``choose`` / ``take_pending`` / ``send_feedback`` /
``feature_names`` / ``status_counts`` / ``domain_id`` / ``start_episode``), so the existing loops run it unchanged:
``online_duo.play_episode`` (learning) and ``record.record_episode`` (frozen eval). With ``refiner=None`` it is
the base alone; with ``force_keep=True`` it never queries a corrector and executes the base move (the identity
check: same trajectory as the base alone).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .adapt1_client import Adapt1Error
from .adapt1_policy import (
    APEX_V3_FEATURE_NAMES,
    FULL_V3_FEATURE_NAMES,
    NONE_SENTINEL,
    Adapt1Policy,
    build_domain_config,
)
from .macros import ApexChoice, Macro
from .policy import Decision

KEEP_BASE = "keep_base"
OVERRIDE_PREFIX = "override_"

# The base pair's vocabulary : the 8 grounded macros of the warm start, and the 3 apex choices.
BASE_MACROS: tuple[Macro, ...] = tuple(m for m in Macro if m is not Macro.JUMP)
BASE_APEX: tuple[ApexChoice, ...] = tuple(ApexChoice)
# The macros an apex look can follow (macros.APEX_MACROS, in enum order): every jump but the vertical `jump`.
ARC_MACROS: tuple[Macro, ...] = tuple(
    m for m in BASE_MACROS if m.value.startswith(("right_run_jump", "right_jump"))
)


class TakeoffEdit(StrEnum):
    """The takeoff corrector's choices: keep the base's macro, or replace it. ``OVERRIDE_JUMP`` is the one
    structural addition for 2-1 (the vertical jump the springboard needs) — an option the
    learner may pick, never a trigger."""

    KEEP_BASE = KEEP_BASE
    OVERRIDE_NOOP = "override_noop"
    OVERRIDE_RIGHT = "override_right"
    OVERRIDE_RIGHT_RUN = "override_right_run"
    OVERRIDE_LEFT = "override_left"
    OVERRIDE_RIGHT_RUN_JUMP_SHORT = "override_right_run_jump_short"
    OVERRIDE_RIGHT_RUN_JUMP_MID = "override_right_run_jump_mid"
    OVERRIDE_RIGHT_RUN_JUMP_FULL = "override_right_run_jump_full"
    OVERRIDE_RIGHT_JUMP_FULL = "override_right_jump_full"
    OVERRIDE_JUMP = "override_jump"


class ApexEdit(StrEnum):
    KEEP_BASE = KEEP_BASE
    OVERRIDE_APEX_KEEP = "override_apex_keep"
    OVERRIDE_APEX_BRAKE = "override_apex_brake"
    OVERRIDE_APEX_PULL_BACK = "override_apex_pull_back"


def takeoff_edits(*, allow_jump: bool = True) -> tuple[TakeoffEdit, ...]:
    return tuple(e for e in TakeoffEdit if allow_jump or e is not TakeoffEdit.OVERRIDE_JUMP)


def apex_edits() -> tuple[ApexEdit, ...]:
    return tuple(ApexEdit)


def apply_takeoff_correction(base: Macro, edit: str) -> Macro:
    """The executed macro. ``keep_base`` is the identity — exactly the base's macro."""
    if edit == KEEP_BASE:
        return base
    if not edit.startswith(OVERRIDE_PREFIX):
        raise ValueError(f"not a takeoff edit: {edit!r}")
    return Macro(edit.removeprefix(OVERRIDE_PREFIX))


def apply_apex_correction(base: ApexChoice, edit: str) -> ApexChoice:
    if edit == KEEP_BASE:
        return base
    if not edit.startswith(OVERRIDE_PREFIX):
        raise ValueError(f"not an apex edit: {edit!r}")
    return ApexChoice(edit.removeprefix(OVERRIDE_PREFIX))


# --- the corrector's context ------------------------------------------------------------------

# Interaction terms (the terms Rei Labs proposed, minus the squares). Distances that read the sentinel 999 ("nothing ahead") make a
# term 0, and the ``*_present`` flag beside it says whether the thing exists (999 × speed
# would read "no gap at run speed" as the most extreme gap). Same names on both correctors.
PRESENT_FEATURES: tuple[str, ...] = ("gap_present", "obstacle_present", "enemy_present")
INTERACTION_FEATURES: tuple[str, ...] = (
    "ix_speed_gap",  # speed_x × gap_dist (tiles)
    "ix_speed_obstacle",  # speed_x × obstacle_dist (tiles)
    "ix_speed_enemy",  # speed_x × enemy_dist (px)
    "ix_speed_height",  # speed_x × obstacle_height (tiles)
    "ix_gap_drop",  # gap_dist × drop_depth
)


def _present(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) < NONE_SENTINEL


def interaction_features(features: Mapping[str, Any]) -> dict[str, float]:
    """The present flags and interaction terms, from the base's own context (no position, no level id)."""
    speed = float(features.get("speed_x") or 0.0)
    gap, obstacle, enemy = (
        features.get("gap_dist"),
        features.get("obstacle_dist"),
        features.get("enemy_dist"),
    )
    height = float(features.get("obstacle_height") or 0.0)
    drop = float(features.get("drop_depth") or 0.0)
    has_gap, has_obstacle, has_enemy = _present(gap), _present(obstacle), _present(enemy)
    return {
        "gap_present": 1.0 if has_gap else 0.0,
        "obstacle_present": 1.0 if has_obstacle else 0.0,
        "enemy_present": 1.0 if has_enemy else 0.0,
        "ix_speed_gap": round(speed * float(gap), 4) if has_gap else 0.0,
        "ix_speed_obstacle": round(speed * float(obstacle), 4) if has_obstacle else 0.0,
        "ix_speed_enemy": round(speed * float(enemy), 4) if has_enemy else 0.0,
        "ix_speed_height": round(speed * height, 4) if has_obstacle else 0.0,
        "ix_gap_drop": round(float(gap) * drop, 4) if has_gap else 0.0,
    }


def one_hot(prefix: str, value: str | None, vocabulary: Iterable[str]) -> dict[str, float]:
    """``{prefix+v: 1.0 if v == value else 0.0}`` — categorical, never an arbitrary scalar order.
    ``value=None`` (unknown: a next state the base was never asked about) gives all zeros."""
    return {f"{prefix}{v}": 1.0 if v == value else 0.0 for v in vocabulary}


BASE_MACRO_PREFIX = "base_is_"
BASE_APEX_PREFIX = "base_apex_is_"
ARC_PREFIX = "arc_from_"


def takeoff_refiner_features(
    base_features: Sequence[str] = FULL_V3_FEATURE_NAMES,
) -> tuple[str, ...]:
    return (
        *base_features,
        *one_hot(BASE_MACRO_PREFIX, None, (m.value for m in BASE_MACROS)),
        *PRESENT_FEATURES,
        *INTERACTION_FEATURES,
    )


def apex_refiner_features(base_features: Sequence[str] = APEX_V3_FEATURE_NAMES) -> tuple[str, ...]:
    return (
        *base_features,
        *one_hot(BASE_APEX_PREFIX, None, (c.value for c in BASE_APEX)),
        *one_hot(ARC_PREFIX, None, (m.value for m in ARC_MACROS)),
        *PRESENT_FEATURES,
        *INTERACTION_FEATURES,
    )


def takeoff_context(base_ctx: Mapping[str, Any], base_macro: str | None) -> dict[str, Any]:
    return {
        **base_ctx,
        **one_hot(BASE_MACRO_PREFIX, base_macro, (m.value for m in BASE_MACROS)),
        **interaction_features(base_ctx),
    }


def apex_context(
    base_ctx: Mapping[str, Any], base_apex: str | None, arc_macro: str | None
) -> dict[str, Any]:
    return {
        **base_ctx,
        **one_hot(BASE_APEX_PREFIX, base_apex, (c.value for c in BASE_APEX)),
        **one_hot(ARC_PREFIX, arc_macro, (m.value for m in ARC_MACROS)),
        **interaction_features(base_ctx),
    }


def refiner_domain_config(
    domain_id: str,
    kind: str,
    *,
    allow_jump: bool = True,
    max_samples: int = 8192,
    model_type: str | None = "extra_trees",
    exploration_mode: str = "ucb",
    credible_dominance: bool = False,
    dominance_margin: float = 0.0,
    reward_max: float = 1.0,
    model_max_weight: float = 0.9,
    min_context_observations: int = 16,
) -> dict[str, Any]:
    """The corrector domain: the zero2 learning contract unchanged (reward ``values.reward`` 0..1, trees pinned,
    ``ucb``, ``min_context_observations`` 16 / ``model_max_weight`` 0.9), with the edit vocabulary as hypotheses
    and the corrector context as feature paths. ``max_samples`` 8,192 sits above the 4,096-Record ceiling."""
    if kind == "takeoff":
        names, edits = takeoff_refiner_features(), takeoff_edits(allow_jump=allow_jump)
    elif kind == "apex":
        names, edits = apex_refiner_features(), apex_edits()
    else:
        raise ValueError(f"kind must be takeoff or apex, not {kind!r}")
    config = build_domain_config(
        domain_id,
        macros=[e.value for e in edits],
        feature_names=names,
        sequential=False,
        reward_max=reward_max,
        exploration_mode=exploration_mode,
        max_samples=max_samples,
        model_type=model_type,
        model_max_weight=model_max_weight,
        min_context_observations=min_context_observations,
    )
    if credible_dominance:
        # The trust region as a documented abstention gate (sequential-learning.md "abstention" table; verified to
        # abstain in 2026-08 tests): the winner's credible interval must dominate every competitor, else the server
        # abstains and the client executes keep_base. An override needs evidence; keep_base never does.
        config["learning"]["policy"]["require_credible_dominance"] = True
        config["learning"]["policy"]["dominance_margin"] = float(dominance_margin)
    return config


# --- the guard: the base never learns ---------------------------------------------------------


OVERLAP_REASONS = frozenset({"credible_intervals_overlap"})


def keep_dominance_pick(
    intervals: Mapping[str, Sequence[float]], status: str | None, reason: str | None
) -> str | None:
    """The `refine` trust region as "override only when proven better than KEEPING the base" (2026-10-04).

    The server's credible-dominance gate abstains unless its winner beats EVERY competitor. Two good overrides at one
    state (both far above keep_base, close to each other) then abstain forever. When the server abstained ONLY for
    that reason (``credible_intervals_overlap``), return the override with the highest lower bound if that bound is
    above keep_base's upper bound; else None (keep the base). Any other abstention (OOD, no support) keeps the base.
    Uses only the server's own intervals."""
    if status != "abstained" or reason not in OVERLAP_REASONS:
        return None
    keep = intervals.get(KEEP_BASE)
    if not keep or len(keep) < 2:
        return None
    overrides = [(a, iv) for a, iv in intervals.items() if a != KEEP_BASE and iv and len(iv) >= 2]
    if not overrides:
        return None
    best, iv = max(overrides, key=lambda kv: float(kv[1][0]))
    return best if float(iv[0]) > float(keep[1]) else None


class BaseFeedbackError(RuntimeError):
    """A feedback addressed to a frozen base domain — the experiment would be invalid."""


class GuardedClient:
    """Wraps the API client and refuses, before any network call, a feedback / event / delete / recreate aimed at
    a protected (base) domain. Counts the calls per domain so a run can record ``base_feedback_calls = 0``."""

    def __init__(self, client: Any, protected: Iterable[str]) -> None:
        self._client = client
        self.protected = frozenset(protected)
        self.feedback_calls: dict[str, int] = {}
        self.query_calls: dict[str, int] = {}

    def _refuse(self, domain_id: str, what: str) -> None:
        if domain_id in self.protected:
            raise BaseFeedbackError(f"refused: {what} on frozen base domain {domain_id!r}")

    def query(self, domain_id: str, body: Any) -> Any:
        self.query_calls[domain_id] = self.query_calls.get(domain_id, 0) + 1
        return self._client.query(domain_id, body)

    def feedback(self, domain_id: str, body: Any) -> Any:
        self._refuse(domain_id, "feedback")
        self.feedback_calls[domain_id] = self.feedback_calls.get(domain_id, 0) + 1
        return self._client.feedback(domain_id, body)

    def events(self, domain_id: str, body: Any) -> Any:
        self._refuse(domain_id, "events")
        return self._client.events(domain_id, body)

    def create_domain(self, config: Mapping[str, Any]) -> Any:
        self._refuse(str(config.get("domain_id")), "create_domain")
        return self._client.create_domain(config)

    def get_domain(self, domain_id: str) -> Any:
        return self._client.get_domain(domain_id)

    def call(self, method: str, path: str, body: Any = None) -> Any:
        parts = path.strip("/").split("/")
        if method.upper() != "GET" and len(parts) >= 2 and parts[0] == "domains":
            self._refuse(parts[1], f"{method} {path}")
        return self._client.call(method, path, body)


# --- the refined policy -----------------------------------------------------------------------


@dataclass
class RefineStats:
    decisions: int = 0
    overrides: int = 0  # executed != base
    override_same_as_base: int = 0  # an override_<b> arm on base b: executes the base
    keep_base: int = 0
    abstained: int = 0  # corrector abstained → keep_base
    jump_overrides: int = 0
    by_edit: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["override_rate"] = round(self.overrides / self.decisions, 4) if self.decisions else 0.0
        return d


class RefinedPolicy:
    """Base + corrector for one decision kind (``takeoff`` or ``apex``). See the module docstring.

    The base must be an ``Adapt1Policy`` built with ``learn=False, allow_exploration=False`` (``frozen_base``); it is
    queried with ``base_actions`` (the base domain's own hypotheses), whatever choice set the loop passes. The
    corrector's context is the exact dict the base was queried with (``base._pending["features"]``), so the two
    domains read one observation.
    """

    def __init__(
        self,
        kind: str,
        base: Adapt1Policy,
        refiner: Adapt1Policy | None = None,
        *,
        allow_jump: bool = True,
        force_keep: bool = False,
        takeoff: RefinedPolicy | None = None,
        anchor_epsilon: float = 0.0,
        rng_seed: int = 0,
        directed: tuple[float, float, float] | None = None,
        learn_after: int = 0,
        keep_dominance: bool = False,
    ) -> None:
        if kind not in ("takeoff", "apex"):
            raise ValueError(kind)
        if base._learn or base._allow_exploration:
            raise ValueError("the base must be frozen: learn=False, allow_exploration=False")
        self.kind = kind
        self.base = base
        self.refiner = None if force_keep else refiner
        self.force_keep = force_keep
        self._takeoff = takeoff  # the apex corrector reads which takeoff macro produced the arc
        self.base_actions: tuple[Any, ...] = BASE_MACROS if kind == "takeoff" else BASE_APEX
        self.edits: tuple[Any, ...] = (
            takeoff_edits(allow_jump=allow_jump) if kind == "takeoff" else apex_edits()
        )
        self.stats = RefineStats()
        self.log: list[dict[str, Any]] = []
        self.last_base: str | None = None
        self.last_executed: str | None = None
        self._pending: dict[str, Any] | None = None
        # Anchored exploration (2026-10-03, after a first 64-row test): the corrector is queried in exploit mode and,
        # with probability anchor_epsilon, its edit is replaced by a uniformly random edit; the feedback credits the
        # executed edit. Online play = the current frozen system + ε tries, so episodes reach where the base fails.
        import random

        self.anchor_epsilon = float(anchor_epsilon)
        self._rng = random.Random(rng_seed)
        # Directed exploration (2026-10-03): ε = clip(k − v_keep, lo, hi), where v_keep is the
        # corrector's OWN ranking score for keep_base in this context (normalized to the declared reward range). The
        # learner explores where it has measured the base to be weak; no position, no level id, no hand-set trigger.
        self.directed = directed
        # Frozen prefix (2026-10-03, stage 3): the first ``learn_after`` takeoff decisions of an episode play the CURRENT
        # frozen system (exploit, no exploration) and send no feedback (0 Records); learning starts after them. The
        # retained behaviour carries the run to its own states, so the frozen game's exact states get explored. The apex
        # corrector follows the takeoff wrapper's count.
        self.learn_after = int(learn_after)
        # Frontier trial (2026-10-04, ``frontier.py``): the next ``choose`` executes this edit instead of the
        # corrector's, and the pending feedback credits it (status ``trial``). Consumed by that one call.
        self.force_edit: str | None = None
        self.keep_dominance = bool(keep_dominance)

    def in_prefix(self) -> bool:
        if (
            self._takeoff is not None
        ):  # apex: the takeoff decision this arc belongs to (already counted)
            return self._takeoff.stats.decisions - 1 < self._takeoff.learn_after
        return self.stats.decisions < self.learn_after

    def epsilon_for(self, selection_values: Mapping[str, float]) -> float:
        if self.directed is None:
            return self.anchor_epsilon
        v_keep = selection_values.get(KEEP_BASE)
        if v_keep is None:
            return self.anchor_epsilon
        k, lo, hi = self.directed
        return min(hi, max(lo, k - float(v_keep)))

    # -- the Adapt1Policy surface the loops use --------------------------------------------------
    @property
    def domain_id(self) -> str:
        return self.refiner.domain_id if self.refiner is not None else self.base.domain_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The STATE features (what ``demos.build_rows`` can compute for a row); the corrector's wider context is
        built by ``next_context`` / at query time."""
        return self.base.feature_names

    @property
    def status_counts(self) -> dict[str, int]:
        return (
            dict(self.refiner.status_counts)
            if self.refiner is not None
            else dict(self.base.status_counts)
        )

    def start_episode(self, episode_id: str) -> None:
        self.base.start_episode(episode_id)
        if self.refiner is not None:
            self.refiner.start_episode(episode_id)

    def close(self) -> None:
        self.base.close()
        if self.refiner is not None:
            self.refiner.close()

    def take_pending(self) -> dict[str, Any] | None:
        pending, self._pending = self._pending, None
        return pending

    def send_feedback(self, pending: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self.refiner is None:
            raise BaseFeedbackError(
                "no corrector: a base-only / force_keep policy never sends feedback"
            )
        return self.refiner.send_feedback(pending, **kwargs)

    def next_context(self, next_state: Mapping[str, Any]) -> dict[str, Any]:
        """``values.next_state`` for a corrector feedback: the next state's features, the base's choice there
        unknown (one-hot all 0 — the base was not asked), interaction terms from the features. The corrector
        domain is a bandit; this field only satisfies the feedback contract (offline_client.py)."""
        base_ctx = {n: next_state[n] for n in self.base.feature_names}
        if self.kind == "takeoff":
            return takeoff_context(base_ctx, None)
        return apex_context(base_ctx, None, None)

    # -- the decision ----------------------------------------------------------------------------
    def choose(
        self, snapshot: Any, actions: Sequence[Any], extra: Mapping[str, Any] | None = None
    ) -> Decision:
        base_dec = self.base.choose(snapshot, self.base_actions, extra=extra)
        base_pending = self.base.take_pending() or {}
        base_ctx = base_pending.get("features") or {}
        base_value = base_dec.action.value
        arc = self._takeoff.last_executed if self._takeoff is not None else None
        eps, prefix = 0.0, False
        if self.refiner is None:
            edit, dec, status = KEEP_BASE, None, ("forced_keep" if self.force_keep else "base_only")
        else:
            ctx = (
                takeoff_context(base_ctx, base_value)
                if self.kind == "takeoff"
                else apex_context(base_ctx, base_value, arc)
            )
            dec = self.refiner.choose(snapshot, self.edits, features=ctx)
            self._pending = self.refiner.take_pending()
            edit = dec.action.value
            status = dec.selection_status or "selected"
            if self.keep_dominance:
                tel = dec.telemetry or {}
                pick = keep_dominance_pick(tel.get("intervals") or {}, status, tel.get("reason"))
                if pick is not None:
                    edit, status = pick, "keep_dominance"
                    if self._pending is not None:
                        self._pending["policy"] = edit
            prefix = self.in_prefix()
            eps = (
                0.0
                if prefix
                else self.epsilon_for((dec.telemetry or {}).get("selection_values") or {})
            )
            if prefix:
                self._pending = None  # no feedback inside the frozen prefix: no Record
            if eps > 0 and self._rng.random() < eps:
                edit = self._rng.choice(self.edits).value
                status = "explored"
                if self._pending is not None:
                    self._pending["policy"] = edit
                    self._pending["explored"] = True
                    self._pending["selection_status"] = "explored"
            if self.force_edit is not None:
                if self.force_edit not in {e.value for e in self.edits}:
                    raise ValueError(f"not a {self.kind} edit: {self.force_edit!r}")
                edit, self.force_edit = self.force_edit, None
                status = "trial"
                if self._pending is not None:
                    self._pending["policy"] = edit
                    self._pending["explored"] = True
                    self._pending["selection_status"] = "trial"
        executed = (
            apply_takeoff_correction(Macro(base_value), edit)
            if self.kind == "takeoff"
            else apply_apex_correction(ApexChoice(base_value), edit)
        )
        s = self.stats
        s.decisions += 1
        s.by_edit[edit] = s.by_edit.get(edit, 0) + 1
        if status == "abstained":
            s.abstained += 1
        if edit == KEEP_BASE:
            s.keep_base += 1
        elif executed.value == base_value:
            s.override_same_as_base += 1
        else:
            s.overrides += 1
            if executed is Macro.JUMP:
                s.jump_overrides += 1
        self.last_base, self.last_executed = base_value, executed.value
        self.log.append(
            {
                "kind": self.kind,
                "x_debug": int(getattr(snapshot, "x", 0)),  # diagnostics only — never in a context
                "base": base_value,
                "base_status": base_dec.selection_status,
                "refinement": edit,
                "executed": executed.value,
                "refiner_status": status,
                "refiner_values": dict((dec.telemetry or {}).get("selection_values") or {})
                if dec is not None
                else {},
                "epsilon": eps,
                "prefix": prefix,
                **({"arc_from": arc} if self.kind == "apex" else {}),
                "context": dict(base_ctx),  # what the base saw (the local feature vector)
                # what the base's SERVER said (2026-10-03 baseline: identical inputs, same model version, different
                # picks within one run — the per-move values make such a transient visible in the trace)
                "base_selected_by": base_pending.get("selected_by"),
                "base_values": dict((base_dec.telemetry or {}).get("values") or {}),
                "base_model": {
                    k: (base_dec.telemetry or {}).get(k)
                    for k in ("model_type", "sample_count", "validation_skill", "reason")
                },
            }
        )
        telemetry = dict((dec.telemetry if dec is not None else base_dec.telemetry) or {})
        telemetry.update(base_macro=base_value, refinement=edit, executed=executed.value)
        return Decision(
            action=executed,  # type: ignore[arg-type]
            confidence=dec.confidence if dec is not None else base_dec.confidence,
            probabilities={executed.value: 1.0},
            latency_ms=base_dec.latency_ms + (dec.latency_ms if dec is not None else 0.0),
            selection_status=status,
            expected_reward=dec.expected_reward if dec is not None else base_dec.expected_reward,
            sample_count=dec.sample_count if dec is not None else base_dec.sample_count,
            telemetry=telemetry,
        )


def frozen_base(
    client: Any, domain_id: str, feature_names: Sequence[str], *, run_id: str
) -> Adapt1Policy:
    """The base domain as a frozen policy: exploit, no exploration, no feedback — the stage-5 confirm's shape."""
    return Adapt1Policy(
        client,
        domain_id,
        allow_exploration=False,
        feature_names=feature_names,
        sequential=False,
        learn=False,
        run_id=run_id,
    )


def refiner_policy(
    client: Any,
    domain_id: str,
    kind: str,
    *,
    learn: bool,
    rng_seed: int = 0,
    epsilon: float = 0.0,
    run_id: str,
    server_exploration: bool = True,
) -> Adapt1Policy:
    """A corrector as an ``Adapt1Policy``: falls back to ``keep_base`` on abstention. ``learn=False`` = frozen
    (exploit, no exploration, no feedback). ``server_exploration=False`` with ``learn=True`` = exploit queries that
    still collect feedback (anchored exploration happens client-side, in ``RefinedPolicy``)."""
    names = takeoff_refiner_features() if kind == "takeoff" else apex_refiner_features()
    return Adapt1Policy(
        client,
        domain_id,
        allow_exploration=learn and server_exploration,
        feature_names=names,
        sequential=False,
        learn=learn,
        epsilon=epsilon if learn else 0.0,
        rng_seed=rng_seed,
        run_id=run_id,
        abstain_fallback=KEEP_BASE,
    )


def base_fingerprint(client: Any, domain_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """What the frozen base IS right now: deployed model, version, training time, skill and
    the live sample counter — one status query per domain (Queries only). A change between checkpoints means the
    base moved (a server refit can move frozen reach by hundreds of px) and only the 1-1 anchor can tell
    whether it still clears."""
    from .adapt1_policy import QUESTION, learner_status

    out: dict[str, dict[str, Any]] = {}
    for dom in domain_ids:
        try:
            _s, q = client.query(dom, {"session_id": "ignored", "question": QUESTION, "top_k": 1})
        except Adapt1Error as exc:
            out[dom] = {"error": exc.status}
            continue
        st = learner_status(q)
        ls = (q.get("learning_state") or {}) if isinstance(q, Mapping) else {}
        st["live_sample_count"] = (ls.get("sample_counts") or {}).get("feedback_policy")
        model = ((ls.get("subsystems") or {}).get("feedback_policy") or {}).get("model") or {}
        for k in ("current_version", "version", "trained_at", "current_model_type"):
            if k in model:
                st[f"model_{k}"] = model[k]
        out[dom] = st
    return out


FINGERPRINT_KEYS = (
    "model_type",
    "live_sample_count",
    "model_current_version",
    "model_version",
    "model_trained_at",
)


def fingerprint_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """The base domains whose fingerprint differs (any of ``FINGERPRINT_KEYS``) between two reads."""
    changed = []
    for dom, b in before.items():
        a = after.get(dom) or {}
        if any(b.get(k) != a.get(k) for k in FINGERPRINT_KEYS):
            changed.append(dom)
    return changed


# --- frozen evaluation and the learning run ---------------------------------------------------

BASE_TAKEOFF_FEATURES = FULL_V3_FEATURE_NAMES
BASE_APEX_FEATURES = APEX_V3_FEATURE_NAMES


@dataclass(frozen=True)
class Pair:
    takeoff: str
    apex: str


def frozen_run(
    client: Any,
    *,
    base: Pair,
    refiner: Pair | None,
    env_id: str,
    seed: int,
    out_dir: Any,
    tag: str,
    force_keep: bool = False,
    allow_jump: bool = True,
    max_decisions: int = 400,
    stall_timeout: int = 60,
    keep_dominance: bool = False,
) -> dict[str, Any]:
    """One frozen episode (Queries only, no feedback): the base alone (``refiner=None``), the base with the
    correctors frozen, or ``force_keep``. Writes ``<tag>-seed<seed>.gif`` and ``.trace.jsonl`` — one line per
    decision with base / refinement / executed and the base's context, plus the recorder's x / frames trace."""
    import json
    from pathlib import Path

    from .record import record_episode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"refine-frozen-{tag}"
    bt = frozen_base(client, base.takeoff, BASE_TAKEOFF_FEATURES, run_id=run_id)
    ba = frozen_base(client, base.apex, BASE_APEX_FEATURES, run_id=run_id)
    rt = ra = None
    if refiner is not None and not force_keep:
        rt = refiner_policy(client, refiner.takeoff, "takeoff", learn=False, run_id=run_id)
        ra = refiner_policy(client, refiner.apex, "apex", learn=False, run_id=run_id)
    takeoff = RefinedPolicy(
        "takeoff",
        bt,
        rt,
        allow_jump=allow_jump,
        force_keep=force_keep,
        keep_dominance=keep_dominance,
    )
    apex = RefinedPolicy(
        "apex", ba, ra, force_keep=force_keep, takeoff=takeoff, keep_dominance=keep_dominance
    )
    gif = out_dir / f"{tag}-seed{seed}.gif"
    stats = record_episode(
        env_id,
        takeoff,
        gif,
        max_decisions=max_decisions,
        seed=seed,
        stall_timeout=stall_timeout,
        cadence="grounded",
        apex_policy=apex,
        actions=BASE_MACROS,
    )
    takeoff.close()
    apex.close()
    # interleave the corrector logs with the recorder's trace (both are in decision order per kind)
    logs = {"takeoff": iter(takeoff.log), "apex": iter(apex.log)}
    trace = [{**t, **(next(logs[t["kind"]], {}) or {})} for t in stats.pop("trace")]
    with (out_dir / f"{tag}-seed{seed}.trace.jsonl").open("w", encoding="utf-8") as fh:
        for row in trace:
            fh.write(json.dumps(row, default=str) + "\n")
    stats.update(
        tag=tag,
        env=env_id,
        seed=seed,
        mode="force_keep" if force_keep else ("base_only" if refiner is None else "base+refiner"),
        takeoff_refine=takeoff.stats.as_dict(),
        apex_refine=apex.stats.as_dict(),
        base_server={"takeoff": dict(bt.status_counts), "apex": dict(ba.status_counts)},
        trace_path=str(out_dir / f"{tag}-seed{seed}.trace.jsonl"),
    )
    return stats


def summarize_failure(trace: Sequence[Mapping[str, Any]], *, last: int = 10) -> dict[str, Any]:
    """The mechanical half of the failure report: where the run ended and what was decided just before. The classification
    (wrong macro / apex / missing primitive / features / credit / harness) is a human reading of this."""
    takeoffs = [t for t in trace if t.get("kind") == "takeoff"]
    apexes = [t for t in trace if t.get("kind") == "apex"]
    best, stall_at, run = 0, None, 0
    for t in takeoffs:
        xa = int(t.get("x_after") or 0)
        if xa > best:
            best, run = xa, 0
        else:
            run += 1
            if run >= 3 and stall_at is None:
                stall_at = best
    keys = ("x", "macro", "x_after", "frames", "base", "refinement", "executed")
    return {
        "max_x_after": best,
        "first_stall_region": stall_at,
        "last_takeoffs": [{k: t.get(k) for k in keys} for t in takeoffs[-last:]],
        "last_apex": [
            {k: t.get(k) for k in ("x", "choice", "base", "refinement")} for t in apexes[-last:]
        ],
        "context_at_end": takeoffs[-1].get("context") if takeoffs else None,
    }


@dataclass
class RefineConfig:
    base: Pair
    refiner: Pair
    env_id: str = "SuperMarioBros-2-1-v0"
    retention_env: str = "SuperMarioBros-1-1-v0"
    max_records: int = 64
    checkpoints: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)
    episodes: int = 1000  # an upper bound; the Records cap ends the run
    seed: int = 0
    eval_seeds: tuple[int, ...] = (777,)
    allow_jump: bool = True
    epsilon: float = 0.0
    exploration: str = "ucb"  # "ucb" (server, Rei's plan as written) or "anchored" (exploit + client ε, after the smoke)
    anchor_epsilon: float = 0.25
    directed: tuple[float, float, float] | None = (
        None  # exploration "directed": (k, lo, hi), ε = clip(k − v_keep)
    )
    records_offset: int = (
        0  # Records the corrector pair already holds: checkpoints are named cumulatively
    )
    initial_checkpoint: bool = True
    prefix_max: int = 0  # >0: each learning episode starts with k ~ U{0..prefix_max} frozen, feedback-free decisions
    anchor_every_checkpoint: bool = (
        True  # the base-only 1-1 anchor at every checkpoint, or only at the first / final
    )
    horizon: int = 128
    max_decisions: int = 200
    stall_timeout: int = 20
    out_dir: Any = "artifacts/refine-2-1"
    run_id: str = "refine-2-1"


def refine_run(
    client: Any, cfg: RefineConfig, *, log: Any = print, on_run_log: Any = None
) -> dict[str, Any]:
    """The learning run: episodes on ``env_id`` with the base frozen and the correctors learning (one Record per
    decision, delayed 128-frame credit), a frozen checkpoint at 0 and whenever the Records spent cross the next
    value of ``checkpoints``, and at the end. A checkpoint = wait for installed correctors, fingerprint the base,
    the base-only 1-1 anchor, base+correctors on 1-1 and on 2-1, fingerprint again. ``client`` must be a
    ``GuardedClient`` protecting the base pair."""
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from .online_duo import (
        CreditConfig,
        DelayedCredit,
        RecordsBudget,
        play_episode,
        status_query,
        wait_installed,
    )
    from .runner import create_mario_env

    if (
        not isinstance(client, GuardedClient)
        or {cfg.base.takeoff, cfg.base.apex} - client.protected
    ):
        raise BaseFeedbackError("refine_run needs a GuardedClient that protects both base domains")
    if {cfg.refiner.takeoff, cfg.refiner.apex} & {cfg.base.takeoff, cfg.base.apex}:
        raise BaseFeedbackError("a corrector domain id equals a base domain id")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_log = out / f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    if on_run_log is not None:
        on_run_log(run_log)
    budget = RecordsBudget(cfg.max_records)
    sent: dict[str, int] = {cfg.refiner.takeoff: 0, cfg.refiner.apex: 0}
    pending_checkpoints = sorted(c for c in cfg.checkpoints if c <= cfg.max_records)
    fp0 = base_fingerprint(client, (cfg.base.takeoff, cfg.base.apex))
    alerts: list[str] = []

    def write(entry: dict[str, Any]) -> None:
        with run_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def checkpoint(tag: str, *, anchor: bool = True) -> dict[str, Any]:
        refiners = wait_installed(client, (cfg.refiner.takeoff, cfg.refiner.apex), log=log)
        before = base_fingerprint(client, (cfg.base.takeoff, cfg.base.apex))
        cdir = out / "frozen" / tag
        results = []

        def play(env: str, seed: int, tag: str, refiner: Pair | None) -> None:
            results.append(
                frozen_run(
                    client,
                    base=cfg.base,
                    refiner=refiner,
                    env_id=env,
                    seed=seed,
                    out_dir=cdir,
                    tag=tag,
                    allow_jump=cfg.allow_jump,
                )
            )

        # Seeds are repeats (a note from Rei Labs); B is played on every eval seed so a server transient  shows as a
        # disagreement between repeats. A (1-1, 47 decisions, the slow game) once per checkpoint; the base-only
        # anchor at the first and final checkpoints, or at every one (``anchor_every_checkpoint``).
        if anchor or cfg.anchor_every_checkpoint:
            play(cfg.retention_env, cfg.eval_seeds[0], "A-base", None)
        play(cfg.retention_env, cfg.eval_seeds[0], "A-refined", cfg.refiner)
        for seed in cfg.eval_seeds:
            play(cfg.env_id, seed, "B-refined", cfg.refiner)
        after = base_fingerprint(client, (cfg.base.takeoff, cfg.base.apex))
        checks = []
        moved = fingerprint_changed(fp0, after)
        if moved:
            checks.append(
                f"BASE CHANGED since the run start: {moved} — read the A-base anchor before trusting A/B"
            )
        for dom in (cfg.refiner.takeoff, cfg.refiner.apex):
            live = status_query(client, dom).get("live_sample_count")
            if isinstance(live, int) and live < sent[dom]:
                checks.append(f"CONSUMED < SENT on {dom}: {live} < {sent[dom]}")
        alerts.extend(checks)
        entry = {
            "kind": "checkpoint",
            "tag": tag,
            "records_spent": budget.spent,
            "records_total": cfg.records_offset + budget.spent,
            "sent_by_domain": dict(sent),
            "base_before": before,
            "base_after": after,
            "refiners": refiners,
            "results": [{k: v for k, v in r.items() if k != "gif"} for r in results],
            "alerts": checks,
        }
        write(entry)
        cell = {
            r["tag"] + ("" if len(cfg.eval_seeds) == 1 else f"@{r['seed']}"): (
                r["max_x"],
                r["reached_flag"],
            )
            for r in results
        }
        b = [r for r in results if r["tag"] == "B-refined"]
        log(
            f"[checkpoint {tag}] records={budget.spent} {cell} "
            f"B overrides takeoff={b[0]['takeoff_refine']['overrides'] if b else '-'} "
            f"apex={b[0]['apex_refine']['overrides'] if b else '-'} jump={b[0]['takeoff_refine']['jump_overrides'] if b else '-'}"
        )
        for c in checks:
            log(f"  !! {c}")
        return entry

    write(
        {
            "kind": "start",
            "base_fingerprint": fp0,
            "config": {k: str(v) for k, v in cfg.__dict__.items()},
        }
    )
    off = cfg.records_offset
    last_cp = checkpoint(f"r{off:04d}") if cfg.initial_checkpoint else {"records_spent": -1}
    env = create_mario_env(cfg.env_id, render_mode="rgb_array")
    episodes = 0
    try:
        for ep in range(cfg.episodes):
            seed = cfg.seed + ep
            episode_id = f"{cfg.run_id}-ep{ep:04d}"
            bt = frozen_base(client, cfg.base.takeoff, BASE_TAKEOFF_FEATURES, run_id=cfg.run_id)
            ba = frozen_base(client, cfg.base.apex, BASE_APEX_FEATURES, run_id=cfg.run_id)
            rt = refiner_policy(
                client,
                cfg.refiner.takeoff,
                "takeoff",
                learn=True,
                rng_seed=seed,
                epsilon=cfg.epsilon,
                run_id=cfg.run_id,
                server_exploration=cfg.exploration == "ucb",
            )
            ra = refiner_policy(
                client,
                cfg.refiner.apex,
                "apex",
                learn=True,
                rng_seed=seed + 100_000,
                epsilon=cfg.epsilon,
                run_id=cfg.run_id,
                server_exploration=cfg.exploration == "ucb",
            )
            eps = cfg.anchor_epsilon if cfg.exploration in ("anchored", "directed") else 0.0
            directed = cfg.directed if cfg.exploration == "directed" else None
            import random as _random

            learn_after = (
                _random.Random(seed).randint(0, cfg.prefix_max) if cfg.prefix_max > 0 else 0
            )
            takeoff = RefinedPolicy(
                "takeoff",
                bt,
                rt,
                allow_jump=cfg.allow_jump,
                anchor_epsilon=eps,
                rng_seed=seed,
                directed=directed,
                learn_after=learn_after,
            )
            apex = RefinedPolicy(
                "apex",
                ba,
                ra,
                takeoff=takeoff,
                anchor_epsilon=eps,
                rng_seed=seed + 100_000,
                directed=directed,
            )
            credit = DelayedCredit(CreditConfig(horizon=cfg.horizon), budget, episode_id)
            stats = play_episode(
                env,
                takeoff=takeoff,
                apex=apex,
                macros=BASE_MACROS,
                credit=credit,
                seed=seed,
                max_decisions=cfg.max_decisions,
                stall_timeout=cfg.stall_timeout,
            )
            takeoff.close()
            apex.close()
            for rec in credit.sent_records:
                sent[rec["domain"]] = sent.get(rec["domain"], 0) + 1
            episodes = ep + 1
            write(
                {
                    "kind": "episode",
                    "episode": episodes,
                    **stats.as_dict(),
                    "takeoff_refine": takeoff.stats.as_dict(),
                    "apex_refine": apex.stats.as_dict(),
                    "learn_after": learn_after,
                    "decisions_log": takeoff.log + apex.log,
                    "feedback": credit.sent_records,
                }
            )
            end = "FLAG" if stats.cleared else ("dead" if stats.dead else "stall")
            log(
                f"ep{episodes:04d} seed={seed} x={stats.max_x} {end} dec={stats.decisions} looks={stats.apex_looks} "
                f"spent={budget.spent}/{cfg.max_records} overrides={takeoff.stats.overrides}/{takeoff.stats.decisions}"
                f"+{apex.stats.overrides}/{apex.stats.decisions} jump={takeoff.stats.jump_overrides}"
            )
            crossed = []
            while pending_checkpoints and budget.spent >= pending_checkpoints[0]:
                crossed.append(pending_checkpoints.pop(0))
            if crossed:  # several thresholds in one episode: evaluate once, named after the highest
                last_cp = checkpoint(f"r{off + crossed[-1]:04d}", anchor=budget.remaining == 0)
            if budget.remaining == 0:
                break
    finally:
        env.close()
    final = (
        last_cp
        if last_cp["records_spent"] == budget.spent
        else checkpoint(f"r{off + budget.spent:04d}-final")
    )
    summary = {
        "kind": "summary",
        "episodes": episodes,
        "records_spent": budget.spent,
        "sent_by_domain": dict(sent),
        "base_feedback_calls": sum(
            client.feedback_calls.get(d, 0) for d in (cfg.base.takeoff, cfg.base.apex)
        ),
        "alerts": alerts,
        "final": final["tag"],
        "run_log": str(run_log),
    }
    write(summary)
    return summary
