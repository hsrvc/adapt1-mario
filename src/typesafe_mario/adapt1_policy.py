"""An Adapt-1 Domain used as the Mario decision layer, in place of Jev.

This mirrors ``TypeSafePolicy`` but swaps the brain: each decision is a
``POST /domains/{id}/query`` returning a ``decision_id`` and a selected controller
macro; after the emulator advances and the outcome is known, ``observe`` attributes it
with ``POST /domains/{id}/feedback`` carrying that same ``decision_id``. Over episodes
the domain's ``policy_scores`` grow and the selection sharpens.

Two designs share this file:

- **Bandit / direct feedback** (verified live in the KB, finding #14) — each decision is
  an independent choice among the 7 macros conditioned on context; reward is the
  per-decision gym reward. This is what ``Adapt1Policy`` implements today.
- **Sequential Discovery** (spec-only, ``building-on-adapt1.md`` §3c) — state-dependent
  action values with delayed credit. Reuses the same feature extraction; enabled via a
  different ``learning`` block and an episode-aware feedback path. Left for the next
  increment once the bandit loop is verified end-to-end.

Leakage discipline (Structure Discovery's six safeguards, §3b): the context features are
*local, generalisable* game geometry and motion — never the reward, the absolute score,
or absolute progress/position, which would let the policy memorise "where in 1-1 am I"
instead of "what should I do in this situation".
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from .actions import Action
from .adapt1_client import Adapt1Error
from .policy import Decision
from .state import MarioSnapshot

QUESTION = "Which controller macro should Mario commit to next?"
DEFAULT_RELATION = "controls"

# Feedback outcome labels. Must be declared in the domain's
# query_templates.feedback_outcomes or the server 422s ("outcome must be a recognized
# feedback outcome"). Kept in sync with _outcome_label below.
OUTCOMES: tuple[str, ...] = ("advanced", "setback", "no_progress", "died", "cleared")

# The 7 macros, in the fixed order used everywhere else in the harness.
MARIO_MACROS: tuple[str, ...] = tuple(a.value for a in Action)

# Ordered, leakage-aware feature names. The domain's `feature_paths` and every
# query/feedback context must agree on exactly this set.
FEATURE_NAMES: tuple[str, ...] = (
    "speed_x",
    "speed_y",
    "grounded",
    "jump_phase",
    "airborne_frames",
    "crossing_gap",
    "enemy_ahead",
    "enemy_dist",
    "contact_frames",
    "takeoff_deadline",
    "must_jump",
    "obstacle_ahead",
    "obstacle_dist",
    "gap_ahead",
    "gap_dist",
    "clear_forward",
    "reliability",
    "outcome",
    "stalled",
    # Floor profile (findings #21, 2026-09-21): `gap_*` now means a real pit; these three
    # add the walk-off-able drop and what is under Mario. Geometry, not verdicts.
    "drop_dist",
    "drop_depth",
    "floor_below",
)

# Perception-only feature set: the raw world facts, WITHOUT the parser's pre-computed
# tactical verdicts (must_jump / takeoff_deadline / contact_frames / the thresholded
# ahead-flags / crossing_gap / the last-action verdict). Forces the learner to discover
# the timing the parser otherwise hands it. See mario/README.md "fairness ladder".
PERCEPTION_FEATURE_NAMES: tuple[str, ...] = (
    "speed_x",
    "speed_y",
    "grounded",
    "jump_phase",
    "airborne_frames",
    "enemy_ahead",
    "enemy_dist",
    "obstacle_dist",
    "gap_dist",
    "clear_forward",
    "reliability",
    "stalled",
    "drop_dist",
    "drop_depth",
    "floor_below",
)

# Apex decision (design-notes §14a): three raw facts about the arc, known only at the top of a
# jump — horizontal speed, how long the rise took, and how far the floor is below. Grounded rows
# carry zeros. Read from RAM/geometry, never from a simulator or the level position.
APEX_EXTRA_NAMES: tuple[str, ...] = ("apex_vx", "apex_rise", "apex_height")
# The apex arm is perception-style on purpose: the full arm's airborne distances come from the
# *last grounded preview* (the takeoff window), which is exactly what cannot see the pipe-4 pit.
# Raw current-frame terrain at the apex is what a player sees mid-flight (probe_apex.py: the
# pit reads gap_dist 7 at the pipe-4 apex and 999 at the pipe-3 apex).
APEX_FEATURE_NAMES: tuple[str, ...] = PERCEPTION_FEATURE_NAMES + APEX_EXTRA_NAMES
# Level memory (design-notes §14b): position as a feature, the one deliberate exception to the
# leakage rule below. Every method that has cleared 1-1 from pixels memorised 1-1; our "no
# position" rule was a generalisation requirement we imposed on ourselves. The paired experiment
# trains the same rows with and without it and evaluates on 1-1 AND 1-3, so the tradeoff is a
# measured pair of numbers instead of a hidden rule. Forms, mildest to strongest (§14b):
#   level_screen = x // 256  ("which screen" — single-level, like PPO / NEAT)   <- the paired arm
#   level_x      = x / 3161  (absolute position, normalised to 1-1's length)   <- form 3, unused
# Form 1 (count of tall pipes passed this episode) needs parser state and is what TCP would
# reconstruct; not built.
POSITION_FEATURE_NAMES: tuple[str, ...] = ("level_screen", "level_x")
LEVEL_LENGTH_PX = 3161.0
# Every feature the flattener emits, in order (recorders write this superset; arms project it).
ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + APEX_EXTRA_NAMES + POSITION_FEATURE_NAMES

# v2 arms (2026-09-24, findings #33): the same features in PER-FRAME units. The parser's
# ``dx``/``dy`` and enemy ``relative_velocity_x`` are per-parse deltas, and every decision loop
# parses once per decision (8–96 frames), so in the v1 arms ``speed_x`` is the previous macro's
# displacement (24 after a run step, 40–54 after a jump) and the enemy projection — ``enemy_dist``
# (projected), ``contact_frames``, ``takeoff_deadline``, ``must_jump`` — is computed from a closing
# speed 8–50× too large: ``enemy_dist`` reads 0 for every approaching enemy, the deadline is
# negative, ``must_jump`` never fires for an enemy. v2 divides the deltas by the parse interval
# (``state["parse_frames"]``, stamped by ``cadence.CadenceParser``) and recomputes the projection
# and the terrain mirror from the per-frame values. ``reliability`` is dropped: it is
# ``"high" if grounded else "low_airborne"`` — a relabelling of ``grounded`` (already a feature),
# constant within each domain (takeoff rows are all grounded, apex rows all airborne).
# One feature added, ``last_frames``: the parse interval itself — the frames the previous macro
# took (0 at the episode start). The v1 ``speed_x`` was speed × that duration in one number, so
# it doubled as a label for "what did I just do" (24 = a run step, 54 = a full jump); v2 hands
# the learner the two factors separately, which is strictly more than v1 carried. History, not
# position, so it stays inside the fairness rule. The apex arm has it already as ``apex_rise``.
# Everything else is unchanged; the v1 arms stay frozen for the domains trained on them
# (#28, #32): feature semantics freeze with a domain's first row.
# Marker: a v2 arm is exactly an arm WITHOUT ``reliability`` (``is_v2_arm``).
FULL_V2_FEATURE_NAMES: tuple[str, ...] = tuple(n for n in FEATURE_NAMES if n != "reliability") + (
    "last_frames",
)
APEX_V2_FEATURE_NAMES: tuple[str, ...] = (
    tuple(n for n in PERCEPTION_FEATURE_NAMES if n != "reliability") + APEX_EXTRA_NAMES
)

# v3 arms (2026-09-25, findings #36 correction): v2 + ``obstacle_height`` — the nearest obstacle's height in
# tiles, which the parser has always computed (``obstacle_height_tiles``, the rule teacher's jump-size input)
# and no arm ever carried. Without it a 3-tile pipe and 2-1's 9-tile springboard tower look the same to the
# learner except by distance, so "walk when standing at a wall", learned on the tower, transferred to 1-1's
# last pipe (the 2851 stall). Grounded: the current terrain; airborne (full arm): the last grounded preview,
# as the distance features do; the apex arm reads the current frame. 0 = no obstacle ahead.
FULL_V3_FEATURE_NAMES: tuple[str, ...] = FULL_V2_FEATURE_NAMES + ("obstacle_height",)
APEX_V3_FEATURE_NAMES: tuple[str, ...] = APEX_V2_FEATURE_NAMES + ("obstacle_height",)

# Named feature sets = the experiment "arms".
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full": FEATURE_NAMES,  # Jev-comparable: identical perception to Jev's parser
    "perception": PERCEPTION_FEATURE_NAMES,  # strips the verdicts; the "did it learn?" arm
    "apex": APEX_FEATURE_NAMES,  # the in-air decision's domain (§14a): raw terrain + arc facts
    "full+screen": FEATURE_NAMES + ("level_screen",),  # §14b paired arm: full + which screen
    "full_v2": FULL_V2_FEATURE_NAMES,  # full in per-frame units, no reliability (findings #33)
    "apex_v2": APEX_V2_FEATURE_NAMES,  # apex in per-frame units, no reliability (findings #33)
    "full_v3": FULL_V3_FEATURE_NAMES,  # v2 + obstacle_height (findings #36 correction)
    "apex_v3": APEX_V3_FEATURE_NAMES,  # v2 + obstacle_height
}


def is_v2_arm(feature_names: Sequence[str]) -> bool:
    """v2-family arms (v2, v3: per-frame units, ``CadenceParser`` snapshots required) are exactly
    the arms without ``reliability`` — every v1 arm carries it."""
    return "reliability" not in feature_names


# Sentinel for "no enemy / no obstacle / no gap within the observed horizon". A large
# finite number keeps the feature numeric (Adapt-1 feature paths want scalars/strings,
# not nulls) while reading as "far away / not applicable".
NONE_SENTINEL = 999.0

# NES tile edge in pixels — to convert the terrain block's tile distances into the pixel
# units the hazard/takeoff projection works in.
TILE_PIXELS = 16.0
# Frames a running jump needs to clear an obstacle; matches threat_features()'s
# jump_clearance_frames so the terrain takeoff deadline is computed identically to enemies.
JUMP_CLEARANCE_FRAMES = 8.0


def _terrain_distances(state: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """(obstacle_tiles, gap_tiles) toward the nearest terrain hazard, **preview-aware**.

    Uses current geometry when grounded (high reliability), else the last grounded preview
    (Jev: "while airborne, prefer terrain.last_grounded_preview"). Tier-2 fix: before this,
    ``obstacle_dist``/``gap_dist`` read the current frame only, so they collapsed to the
    "nothing" sentinel while airborne (probe_pipe.py: ``obstacle_dist=999`` mid-jump at the
    pipe) — blinding the learner to an obstacle it was actively clearing.
    """
    player = state["player"]
    terr = state["terrain"]
    if player["grounded"]:
        return terr.get("obstacle_distance_tiles"), terr.get("gap_distance_tiles")
    preview = terr.get("last_grounded_preview") or {}
    return preview.get("obstacle_distance_tiles"), preview.get("gap_distance_tiles")


def _terrain_hazard(
    state: Mapping[str, Any], speed_x: float | None = None
) -> tuple[float | None, float | None]:
    """(contact_frames, takeoff_deadline_frames) for the nearest terrain obstacle/gap — the
    terrain analogue of the enemy hazard projection in ``threat_features``.

    The shared parser computes ``estimated_contact_frames`` / ``takeoff_deadline_frames`` /
    ``jump_must_start_this_decision`` only for **enemies**; a pipe is a static terrain obstacle
    with no enemy, so those — and therefore ``must_jump`` / ``contact_frames`` — never fired
    for it (measured 2026-09-18, ``probe_pipe.py``: ``must_jump=0`` at the pipe though a
    forward jump is required). We mirror the projection here, in the Adapt-1 feature layer, so
    ``state.py`` stays byte-identical with upstream (README: byte-identical parser = clean Jev
    comparison). A static obstacle's closing speed is Mario's own forward speed; returns
    (None, None) when not closing on any trusted terrain hazard. ``speed_x`` overrides the
    parser's per-parse value (the v2 arms pass the per-frame speed, findings #33).
    """
    if speed_x is None:
        speed_x = float(state["player"]["horizontal_speed_px_per_frame"])
    if speed_x <= 0:  # not closing on a static obstacle → no terrain contact/takeoff pressure
        return None, None
    obstacle_tiles, gap_tiles = _terrain_distances(state)
    tiles = min((t for t in (obstacle_tiles, gap_tiles) if t is not None), default=None)
    if tiles is None:
        return None, None
    delay = float(state["reaction_timing"]["last_inference_delay_frames"])
    contact_frames = (float(tiles) * TILE_PIXELS) / speed_x
    return contact_frames, contact_frames - delay - JUMP_CLEARANCE_FRAMES


def select_features(features: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Project the full feature dict down to a named arm (order preserved)."""
    return {name: features[name] for name in names}


def _arm_uses_projection(feature_names: Sequence[str]) -> bool:
    """Whether an arm gets the parser's projection/preview enrichment on its distances.

    The full (Jev-comparable) arm carries the tactical verdicts (``must_jump`` etc.), so it
    also gets projected/preview-aware distances — the perception fed to Jev. The perception
    arm has no verdicts and stays on raw current-frame facts, so it remains a clean test of
    whether Adapt-1 discovers the timing itself.
    """
    return "must_jump" in feature_names


def context_features(
    state: Mapping[str, Any], feature_names: Sequence[str], *, projected: bool | None = None
) -> dict[str, Any]:
    """Build the Adapt-1 context for a given arm: flatten (projection enrichment only for
    verdict-carrying arms, unless ``projected`` is forced), then project to the arm's names."""
    if projected is None:
        projected = _arm_uses_projection(feature_names)
    if is_v2_arm(feature_names):
        # v3 arms (they carry obstacle_height) also get the first-sighting closing prior.
        prior = "obstacle_height" in feature_names
        return select_features(
            snapshot_features_v2(state, projected=projected, first_sighting_prior=prior),
            feature_names,
        )
    return select_features(snapshot_features(state, projected=projected), feature_names)


def _num(value: Any, sentinel: float = NONE_SENTINEL) -> float:
    return sentinel if value is None else float(value)


def snapshot_features(state: Mapping[str, Any], *, projected: bool = True) -> dict[str, Any]:
    """Flatten a ``MarioSnapshot.to_state()`` dict into the Adapt-1 context features.

    Deliberately excludes reward, score, absolute x / progress / best_progress and the
    clock — including any of those would leak the outcome or turn position into an
    identifier the policy could overfit to.

    ``projected`` gates the parser's projection/preview enrichment on the three distance
    features shared by both arms (``enemy_dist``, ``obstacle_dist``, ``gap_dist``). The full
    (Jev-comparable) arm passes ``projected=True`` — reaction-projected enemy distance and
    airborne-preview terrain distances, matching what Jev is fed. The perception arm passes
    ``projected=False`` to keep those on **raw current-frame** facts, so it stays a clean
    "did it discover the timing itself?" test and the enrichment isn't a silent confound.
    (The verdict features — ``must_jump``/``takeoff_deadline``/``contact_frames`` — exist only
    in the full arm, so they are always computed here and simply not selected by perception.)
    """
    player = state["player"]
    traj = state["trajectory"]
    haz = state["hazard"]
    terr = state["terrain"]
    recent = state["recent_control"]
    episode = state["episode"]
    # Move A (2026-09-18): port Jev's projection-aware perception into the flat feature layer.
    # The shared parser computes contact/takeoff/jump-now projections for ENEMIES only; a pipe
    # is a static terrain obstacle with no enemy, so `must_jump`/`contact_frames`/`takeoff` all
    # went silent at the pipe (probe_pipe.py). Mirror the enemy math for terrain and combine.
    horizon = float(state["reaction_timing"]["action_horizon_frames"])
    enemy_takeoff = haz["takeoff_deadline_frames"]
    terrain_contact, terrain_takeoff = _terrain_hazard(state)
    enemy_must_jump = bool(haz["jump_must_start_this_decision"])
    terrain_must_jump = bool(
        player["grounded"] and terrain_takeoff is not None and 0.0 <= terrain_takeoff <= horizon
    )
    _deadlines = [d for d in (enemy_takeoff, terrain_takeoff) if d is not None]
    _contacts = [c for c in (haz["estimated_contact_frames"], terrain_contact) if c is not None]
    # The three distance features: projected/preview-aware for the full arm, raw for perception.
    if projected:
        enemy_dist = haz["projected_distance_after_reaction_pixels"]  # where it WILL be
        obstacle_tiles, gap_tiles = _terrain_distances(state)  # airborne-preview aware
    else:
        enemy_dist = haz["nearest_enemy_distance_pixels"]  # raw current
        obstacle_tiles = terr.get("obstacle_distance_tiles")  # raw current frame
        gap_tiles = terr.get("gap_distance_tiles")
    features = {
        "speed_x": float(player["horizontal_speed_px_per_frame"]),
        "speed_y": float(player["vertical_speed_px_per_frame"]),
        "grounded": 1.0 if player["grounded"] else 0.0,
        "jump_phase": str(player["jump_phase"]),
        "airborne_frames": float(traj["airborne_frames"]),
        "crossing_gap": 1.0 if traj["crossing_known_gap"] else 0.0,
        "enemy_ahead": 1.0 if haz["enemy_ahead"] else 0.0,
        "enemy_dist": _num(enemy_dist),
        "contact_frames": _num(min(_contacts) if _contacts else None),
        "takeoff_deadline": _num(min(_deadlines) if _deadlines else None),
        "must_jump": 1.0 if (enemy_must_jump or terrain_must_jump) else 0.0,
        "obstacle_ahead": 1.0 if terr.get("obstacle_ahead") else 0.0,
        "obstacle_dist": _num(obstacle_tiles),
        "gap_ahead": 1.0 if terr.get("gap_ahead") else 0.0,
        "gap_dist": _num(gap_tiles),
        "clear_forward": float(terr.get("clear_forward_tiles") or 0),
        "reliability": str(terr.get("observation_reliability", "unknown")),
        "outcome": str(recent["outcome"]),
        "stalled": float(episode["stalled_frames"]),
        "drop_dist": _num(terr.get("drop_distance_tiles")),
        "drop_depth": float(terr.get("drop_depth_tiles") or 0),
        "floor_below": _num(terr.get("floor_below_tiles")),  # sentinel = airborne over a pit
    }
    # Guard against silent drift between this function and FEATURE_NAMES.
    assert tuple(features.keys()) == FEATURE_NAMES, "feature set drifted from FEATURE_NAMES"
    # Apex extras: present only when the caller attached ``state["apex"]`` (the in-air look).
    apex = state.get("apex") or {}
    features["apex_vx"] = float(apex.get("vx", 0.0))
    features["apex_rise"] = float(apex.get("rise", 0.0))
    features["apex_height"] = _num(apex.get("height", 0.0))  # sentinel = nothing solid below
    # Position (§14b): only the "+screen"/"+x" arms select these; every other arm drops them.
    x = float(player.get("x", 0.0))
    features["level_screen"] = float(int(x // 256))
    features["level_x"] = round(x / LEVEL_LENGTH_PX, 4)
    assert tuple(features.keys()) == ALL_FEATURE_NAMES, "feature set drifted from ALL_FEATURE_NAMES"
    return features


# The parser's constants, mirrored (state.py threat_features): frames a jump needs to clear.
ENEMY_JUMP_CLEARANCE_FRAMES = 8.0


# A goomba walks toward Mario at ≈ 0.6 px/frame (audit 2026-09-24: closing 3.62 at Mario's 3.0).
GOOMBA_SPEED_PX_PER_FRAME = 0.6


def snapshot_features_v2(
    state: Mapping[str, Any], *, projected: bool = True, first_sighting_prior: bool = False
) -> dict[str, Any]:
    """The v2 flattener (findings #33): ``snapshot_features`` with the per-parse quantities
    converted to per-frame units and ``reliability`` dropped.

    Requires ``state["parse_frames"]`` — the emulator frames since the previous parse, stamped
    by ``cadence.CadenceParser`` (``None`` on an episode's first parse, where every delta is 0
    anyway). A snapshot from the plain parser raises: a v2 domain must never receive v1 numbers.

    Per frame: ``speed_x``/``speed_y`` (displacement ÷ interval) and the enemy's relative
    velocity, from which the projection is recomputed exactly as the parser does it —
    closing speed, contact frames, ``projected = dist + v·(horizon + delay)``, the takeoff
    deadline (contact − delay − 8 frames of clearance) and ``must_jump`` (grounded and the
    deadline inside the decision horizon) — and the terrain mirror (``_terrain_hazard``) with
    the per-frame speed. ``projected`` keeps the v1 meaning: the full arm's ``enemy_dist`` is the
    projected distance, the apex arm's the raw current distance.

    Adds ``last_frames`` = the interval (0 on the first parse): the previous macro's duration,
    which the v1 ``speed_x`` carried multiplied into the speed.

    ``first_sighting_prior`` (v3 arms, findings #38): the parser's relative velocity is 0 on the parse
    where an enemy first appears (no previous distance), so the projection reads "no contact" for an
    enemy that may be 50 px away — the gate run died twice exactly there. With the prior, an enemy
    ahead whose velocity is unknown (0) is assumed to close at Mario's speed plus a goomba's, so
    contact / deadline / ``must_jump`` / the projected distance are finite at first sight.

    Left as they are (documented, not corrected): ``airborne_frames`` and ``stalled`` count
    parses, not frames (constant within a domain: takeoff rows 0, apex rows 1); ``apex_rise``
    already carries the true rise in frames.
    """
    if "parse_frames" not in state:
        raise ValueError(
            "v2 arms need a CadenceParser snapshot (state['parse_frames'] missing): the plain "
            "MarioStateParser's speeds are per-parse displacements (findings #33)"
        )
    frames = state["parse_frames"]
    frames_f = float(frames) if frames else None  # None: first parse of the episode
    player = state["player"]
    haz = state["hazard"]
    timing = state["reaction_timing"]
    horizon = float(timing["action_horizon_frames"])
    delay = float(timing["last_inference_delay_frames"])

    def per_frame(value: Any) -> float:
        return float(value) / frames_f if frames_f else 0.0

    speed_x = per_frame(player["horizontal_speed_px_per_frame"])
    speed_y = per_frame(player["vertical_speed_px_per_frame"])

    # Enemy projection in per-frame units (mirrors state.py threat_features).
    enemy_contact = enemy_deadline = None
    enemy_projected = None
    enemy_must_jump = False
    if haz["enemy_ahead"] and haz.get("nearest_enemy_distance_pixels") is not None:
        dist = float(haz["nearest_enemy_distance_pixels"])
        rel_v = per_frame(haz["relative_velocity_x"] or 0)
        if first_sighting_prior and rel_v >= 0.0:
            # Unknown or implausible velocity: 0 on the parse where an enemy first appears, and a
            # POSITIVE ("receding") value when the parser's per-slot tracking hands the slot to a
            # different enemy between two parses — common at grounded cadence (38–53-frame macros):
            # at the v2 gate's death states the nearest goomba read +142 / +184 px per parse while
            # standing 12 / 22 px away (findings #40). An enemy ahead that outruns Mario is a kicked
            # shell or slot garbage; assume it closes at Mario's speed plus a goomba's.
            rel_v = -(max(speed_x, 0.0) + GOOMBA_SPEED_PX_PER_FRAME)
        closing = max(0.0, -rel_v)
        enemy_contact = dist / closing if closing > 0 else None
        enemy_projected = max(0.0, dist + rel_v * (horizon + delay))
        if enemy_contact is not None:
            enemy_deadline = enemy_contact - delay - ENEMY_JUMP_CLEARANCE_FRAMES
            enemy_must_jump = bool(player["grounded"] and 0.0 <= enemy_deadline <= horizon)

    terrain_contact, terrain_deadline = _terrain_hazard(state, speed_x=speed_x)
    terrain_must_jump = bool(
        player["grounded"] and terrain_deadline is not None and 0.0 <= terrain_deadline <= horizon
    )
    contacts = [c for c in (enemy_contact, terrain_contact) if c is not None]
    deadlines = [d for d in (enemy_deadline, terrain_deadline) if d is not None]

    features = snapshot_features(state, projected=projected)
    features["speed_x"] = round(speed_x, 4)
    features["speed_y"] = round(speed_y, 4)
    if projected:
        features["enemy_dist"] = _num(enemy_projected)
    features["contact_frames"] = _num(round(min(contacts), 3) if contacts else None)
    features["takeoff_deadline"] = _num(round(min(deadlines), 3) if deadlines else None)
    features["must_jump"] = 1.0 if (enemy_must_jump or terrain_must_jump) else 0.0
    del features["reliability"]
    features["last_frames"] = float(frames or 0)
    features["obstacle_height"] = _obstacle_height(state, projected=projected)
    return features


def _obstacle_height(state: Mapping[str, Any], *, projected: bool) -> float:
    """Height in tiles of the nearest obstacle ahead (0 = none). Grounded, or an unprojected arm
    (apex): the current terrain block. Airborne on a projected arm: the last grounded preview, the
    same source ``_terrain_distances`` uses for ``obstacle_dist``."""
    terr = state["terrain"]
    if state["player"]["grounded"] or not projected:
        return float(terr.get("obstacle_height_tiles") or 0)
    preview = terr.get("last_grounded_preview") or {}
    return float(preview.get("obstacle_height_tiles") or 0)


def feature_paths(feature_names: Sequence[str] = FEATURE_NAMES) -> list[str]:
    return [f"values.{name}" for name in feature_names]


def build_domain_config(
    domain_id: str,
    *,
    relation: str = DEFAULT_RELATION,
    macros: Sequence[str] = MARIO_MACROS,
    feature_names: Sequence[str] = FEATURE_NAMES,
    max_samples: int = 4096,
    exploration_mode: str = "ucb",
    # findings #22: the final ranking blends the trained model (up to `model_max_weight`) with a
    # contextual Beta posterior whose prior (~0.20) out-scores every honestly-measured macro when
    # the reward is a small progress fraction. Measured on mario-floor-gp: at the server default
    # 0.45 / our old 3, it selected the two least-seen macros (`right`, `left`) everywhere; at
    # 0.9 / 16 it followed the model (cp0 149 -> 723). Held fixed from here on.
    min_context_observations: int = 16,
    model_max_weight: float = 0.9,
    sequential: bool = False,
    discount: float = 0.95,
    n_step: int = 5,
    credit_assignment: bool = False,
    temporal_context: bool = False,
    maximum_lag: int = 16,
    reward_max: float = 1.0,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Domain config for the Mario decision layer.

    - ``model_type`` (2026-09-30, Rei's model-validation fix): pins the feedback-policy model family via
      ``learning.training.model_type`` — ``extra_trees`` / ``mlp_v2`` / ``sequential_q_mlp``; ``None`` = the
      server's ``auto`` (candidate validation, findings #30 §7 / #37: could land on the policy-blind MLP). The spec
      says a pinned family "must still pass validation; no automatic fallback". Verified in the spec snapshot
      ``docs-snapshots/2026-09-30/openapi.json`` (``DomainPolicyTrainingRequest``).
    - Bandit (``sequential=False``): 7 macros as competing hypotheses sharing one
      ``relation``; reward from ``values.reward`` (declared so it can't be silently
      dropped). ✅ verified.
    - Sequential (``sequential=True``): adds ``learning.sequential`` so the episode-aware
      ``feedback_policy`` (``sequential_q_mlp``) learns state-dependent action values with
      delayed credit. The declared paths must match where ``observe`` writes them in the
      feedback body. 📄 spec-only until verified live. ``max_samples`` stays 4096 (the
      512 default evicts transitions).

    ``feature_names`` selects the experiment arm (full vs perception-only).
    """
    config = {
        "domain_id": domain_id,
        "schema": {
            "event_types": ["frame"],
            "relations": [relation],
            "signals": ["reward", *feature_names],
        },
        "hypotheses": [
            {"name": macro, "relation": relation, "policy": macro, "predicts": ["advanced"]}
            for macro in macros
        ],
        "learning": {
            "enabled": True,
            "context": {"feature_paths": feature_paths(feature_names), "max_samples": max_samples},
            **({"training": {"enabled": True, "model_type": model_type}} if model_type else {}),
            "reward": {
                "aggregation": "weighted_mean",
                # `min`/`max` = "expected numeric range used to normalize utility to [0,1]"
                # (sequential-learning.md §6). Declared explicitly (findings #22): the server
                # default is 0..1, so a [-1,1] reward silently floored every death at 0.
                # `reward_max` (2026-09-23): declare the range at the scale of the data — a
                # per-frame progress rate caps at ~0.0076, and at max 1.0 every measured value
                # sits far below the contextual prior (~0.2), the #22/#28 prior trap. This is
                # the documented lever we had never used.
                "components": [
                    {
                        "field": "values.reward",
                        "goal": "maximize",
                        "weight": 1.0,
                        "min": 0.0,
                        "max": float(reward_max),
                    }
                ],
            },
            "policy": {
                "exploration_mode": exploration_mode,
                "min_context_observations": min_context_observations,
                "model_max_weight": model_max_weight,
            },
        },
        "query_templates": {"feedback_outcomes": list(OUTCOMES)},
    }
    if sequential:
        config["learning"]["sequential"] = {
            "enabled": True,
            "episode_path": "metadata.episode_id",
            "step_path": "metadata.step",
            "next_context_path": "values.next_state",
            "reward_path": "values.reward",
            "terminal_path": "metadata.terminal",
            "discount": discount,
            "n_step": n_step,
        }
    if temporal_context:
        # TCP (temporal-context-projection.md): bounded recent history of NUMERIC scalar inputs,
        # keyed by (episode_id, step) which every query must carry under context.metadata.
        # findings #22: the two 4-tall pipe tops are aliased in the current observation (the pit
        # is beyond the 8-tile screen); only episode history separates them — TCP's stated use.
        numeric = [n for n in feature_names if n not in ("jump_phase", "reliability", "outcome")]
        config["learning"]["temporal_context"] = {
            "enabled": True,
            "episode_path": "metadata.episode_id",
            "step_path": "metadata.step",
            "input_paths": feature_paths(numeric),
            "maximum_lag": maximum_lag,
        }
    if credit_assignment:
        # `learning.credit_assignment` is a *typed*, spec-declared subsystem
        # (`DomainCreditAssignmentRequest`) for delayed outcomes, defaulting `mode: "none"`.
        # It was never set in #15/#16/#17, so the only delayed credit in play was the
        # sequential learner's own n-step returns (capped at `n_step`). `eligibility_trace`
        # turns on the dedicated TD-style propagation path (the `mode` regex accepts exactly
        # none|eligibility_trace|counterfactual_trace — verified from /openapi.json after the
        # docs' loose "trace" 422'd on 2026-09-18; counterfactual_trace is the CUP variant).
        # 📄 spec-only — verify by the counter, not the 200 ("accepted+echoed" ≠ "consumed").
        # `delay_path` left at its default (`values.delay_steps`); we send no explicit per-step
        # delay, so trace propagates over the episode structure (next_context/terminal) instead.
        config["learning"]["credit_assignment"] = {
            "mode": "eligibility_trace",
            "discount": discount,
        }
    return config


def _outcome_label(next_state: Mapping[str, Any], reward: float, terminal: bool) -> str:
    """A required categorical outcome label, derived from the transition."""
    episode = next_state.get("episode", {}) if isinstance(next_state, Mapping) else {}
    if episode.get("stage_clear"):
        return "cleared"
    if terminal and episode.get("dead"):
        return "died"
    if reward > 0:
        return "advanced"
    if reward < 0:
        return "setback"
    return "no_progress"


def _dig(obj: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, Mapping):
            return None
        obj = obj.get(key)
    return obj


def _parse_query(resp: Any) -> dict[str, Any]:
    """Extract decision_id, selected policy, status and any scores, defensively.

    The exact response field names for a domain query are not in the public docs; this
    tolerates several spellings and is tightened once confirmed against the live smoke.
    """
    if not isinstance(resp, Mapping):
        return {"decision_id": None, "policy": None, "status": None, "reason": None, "scores": None}
    selection = resp.get("selection") or resp.get("decision") or {}
    if not isinstance(selection, Mapping):
        selection = {}
    decision_id = (
        resp.get("decision_id")
        or selection.get("decision_id")
        or _dig(resp, "decision", "decision_id")
    )
    status = selection.get("status") or resp.get("status")
    reason = selection.get("reason") or selection.get("abstention_reason")
    policy = None
    for key in ("policy", "selected_policy", "chosen_policy", "action", "choice"):
        candidate = selection.get(key) or resp.get(key)
        if candidate:
            policy = candidate
            break
    if status == "abstained":
        policy = None
    scores = resp.get("policy_scores") or selection.get("policy_scores")
    diagnostics = resp.get("policy_diagnostics")
    return {
        "decision_id": decision_id,
        "policy": policy,
        "status": status,
        "reason": reason,
        "scores": scores,
        "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else None,
    }


def _learning_signal(resp: Any, policy: str) -> dict[str, Any]:
    """The real learning signals live in the *query* response, not the feedback one:
    the monotonic counter ``learning_state.sample_counts.feedback_policy`` and the
    per-macro beta-Bernoulli posterior under ``policy_diagnostics.<macro>``.
    ``policy_scores`` is often ``[]`` until evidence exists — do not rely on it.
    """
    return {
        "sample_count": _dig(resp, "learning_state", "sample_counts", "feedback_policy"),
        "expected_reward": _dig(resp, "policy_diagnostics", policy, "selection_expected_reward"),
        "posterior_alpha": _dig(resp, "policy_diagnostics", policy, "selection_posterior_alpha"),
    }


def learner_status(query_response: Any) -> dict[str, Any]:
    """Read the feedback-policy model status from a domain query response.

    ⚠️ The install/model info lives at
    ``learning_state.subsystems.feedback_policy.model`` — NOT top-level ``feedback_policy``
    (reading the wrong path reports ``model_type: None`` even when a model is installed).
    """
    ls = query_response.get("learning_state", {}) if isinstance(query_response, Mapping) else {}
    model = _dig(ls, "subsystems", "feedback_policy", "model") or {}
    report = model.get("report", {}) if isinstance(model, Mapping) else {}
    if not isinstance(model, Mapping):
        model = {}
    # ⚠️ Measured 2026-09-21 (mario-floor-g): after a retrain the block describes the *candidate*
    # (`report`, `installed: false`, `reason: candidate_objective_mismatch`) while the DEPLOYED
    # model is under `current_model_type` / `current_validation_skill` / `current_version` with
    # `status: retained`. Reading `report.model_type` alone reports "extra_trees, not installed"
    # when a sequential_q_mlp is live. Prefer the `current_*` fields when present.
    current_type = model.get("current_model_type")
    deployed = current_type is not None or bool(model.get("installed"))
    return {
        "installed": deployed,
        "model_type": current_type or report.get("model_type"),
        "candidate_model_type": report.get("model_type"),
        "model_status": model.get("status"),
        "reason": model.get("reason"),
        "episode_count": report.get("episode_count"),
        "validation_skill": model.get("current_validation_skill", report.get("validation_skill")),
        "sample_count": report.get("sample_count") or _dig(ls, "sample_counts", "feedback_policy"),
    }


def _parse_feedback(resp: Any) -> dict[str, Any]:
    if not isinstance(resp, Mapping):
        return {"applied": None, "context_source": None, "observations": None, "delta": None}
    credit = resp.get("credit_assignment") or {}
    observations = None
    scores = resp.get("policy_scores")
    if isinstance(scores, list):
        observations = sum(int(s.get("observations", 0)) for s in scores if isinstance(s, Mapping))
    return {
        "applied": credit.get("contextual_learning_applied")
        if isinstance(credit, Mapping)
        else None,
        "context_source": credit.get("context_source") if isinstance(credit, Mapping) else None,
        "observations": observations,
        "delta": resp.get("sample_count_delta")
        or _dig(resp, "credit_assignment", "sample_count_delta"),
    }


def _probabilities(scores: Any, actions: Sequence[Action], chosen: Action) -> dict[str, float]:
    if isinstance(scores, list) and scores:
        raw: dict[str, float] = {}
        for entry in scores:
            if isinstance(entry, Mapping):
                name = entry.get("policy") or entry.get("name")
                value = entry.get("score", entry.get("mean"))
                if name is not None and value is not None:
                    raw[str(name)] = max(0.0, float(value))
        total = sum(raw.values())
        if total > 0:
            return {a.value: raw.get(a.value, 0.0) / total for a in actions}
    return {a.value: float(a == chosen) for a in actions}


class Adapt1Policy:
    """Drop-in ``Policy`` whose ``choose`` queries an Adapt-1 bandit domain.

    The runner calls :meth:`observe` after each macro executes; that is where feedback is
    attributed and a full audit line is written to the trace sidecar.
    """

    def __init__(
        self,
        client: Any,
        domain_id: str,
        *,
        relation: str = DEFAULT_RELATION,
        allow_exploration: bool = True,
        top_k: int = 3,
        rng_seed: int = 0,
        run_id: str = "mario",
        trace_path: Any = None,
        feature_names: Sequence[str] = FEATURE_NAMES,
        sequential: bool = False,
        explore_after_x: int | None = None,
        epsilon: float = 0.0,
        temporal_context: bool = False,
        learn: bool = True,
    ) -> None:
        self._client = client
        self._domain = domain_id
        self._relation = relation
        self._allow_exploration = allow_exploration
        # Phased/curriculum exploration: below this x, exploit the already-learned policy to
        # reach the hard region; at/after it, explore so learning budget lands on the hard
        # obstacle (findings #20 next-lever). None → exploration follows allow_exploration.
        self._explore_after_x = explore_after_x
        # findings #22 (extended curve, flat at 1127 for 290 Records): the server's UCB explores per
        # macro *globally*, so a macro with 400+ observations (`right_run`) is never the "uncertain"
        # arm at the one rare state where it matters (top of pipe 4). Plain client-side ε-greedy —
        # uniform over the macros, level-agnostic, only when exploration is allowed — gives every
        # state a few off-policy tries. The executed macro is what the feedback credits, so the
        # learner sees the consequence. Off (0.0) in every frozen eval.
        self._epsilon = max(0.0, min(1.0, epsilon))
        # TCP: every query must carry (episode_id, step) under context.metadata, monotone within
        # the episode and never reused (the pair is a dedup key). The runner/recorder call
        # start_episode(); choose() advances the step itself so frozen eval (no observe) works.
        self._tcp = temporal_context
        self._episode_id: str = "ep-unset"
        self._step = 0
        # learn=False: a frozen demo/eval through run_episode — queries the domain (Queries only)
        # but never sends feedback, so it spends no Records and adds no training rows.
        self._learn = learn
        self.last_feedback: dict[str, Any] | None = None  # what observe() last sent (dashboard)
        self._top_k = top_k
        self._rng = random.Random(rng_seed)
        self._run_id = run_id
        self._feature_names = tuple(feature_names)
        self._sequential = sequential
        self._pending: dict[str, Any] | None = None
        # findings #22: count how often the SERVER selected vs abstained. An eval where every
        # decision is "abstained" measured the fallback, not the learner — report this always.
        self.status_counts: dict[str, int] = {}
        # Long-lived: one handle for the policy's lifetime, closed in close().
        self._trace = (
            open(trace_path, "w", encoding="utf-8") if trace_path else None  # noqa: SIM115
        )

    def close(self) -> None:
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def start_episode(self, episode_id: str) -> None:
        """Begin a new ordered history (TCP). Call once per episode, before the first choose()."""
        self._episode_id = str(episode_id)
        self._step = 0

    def _context_metadata(self) -> dict[str, Any]:
        return {"episode_id": self._episode_id, "step": self._step}

    @property
    def domain_id(self) -> str:
        return self._domain

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def take_pending(self) -> dict[str, Any] | None:
        """Detach the last decision's pending record (decision_id, context, policy) so a caller
        can send its feedback later with ``send_feedback`` — the delayed-credit loop
        (``online_duo.py``) holds it until the return over the next frames is known. After this,
        ``observe`` has nothing to send."""
        pending, self._pending = self._pending, None
        return pending

    def choose(
        self,
        snapshot: MarioSnapshot,
        actions: Sequence[Action],
        extra: Mapping[str, Any] | None = None,
    ) -> Decision:
        state = snapshot.to_state()
        if extra is not None:
            # The apex look (design-notes §14a): raw arc facts the flattener emits as apex_*.
            # An apex domain reads raw current-frame terrain, never the takeoff preview.
            state["apex"] = dict(extra)
        features = context_features(
            state, self._feature_names, projected=False if extra is not None else None
        )
        explore = self._allow_exploration
        if self._explore_after_x is not None:
            # Exploit until the hard region, then explore — spends the exploration/feedback
            # budget on the obstacle we actually need to learn, not on the already-solved run-up.
            explore = self._allow_exploration and snapshot.x >= self._explore_after_x
        context: dict[str, Any] = {"values": features}
        if self._tcp:
            context["metadata"] = self._context_metadata()
        body = {
            "session_id": "ignored",  # value is ignored (proxy substitutes), but presence is required
            "question": QUESTION,
            "context": context,
            "top_k": self._top_k,
            "allow_exploration": explore,
        }
        if not explore:
            # sequential-learning.md §16 (frozen evaluation): allow_exploration false AND
            # selection_mode "exploit". Measured 2026-09-22 that we had only sent the first.
            body["selection_mode"] = "exploit"
        started = perf_counter()
        _status, resp = self._client.query(self._domain, body)
        latency_ms = (perf_counter() - started) * 1000
        parsed = _parse_query(resp)

        abstained = parsed["policy"] is None
        # On abstention, commit to the learner's OWN best guess (argmax of the scores), not a
        # random action. Random fallback silently corrupts eval — it measures coin-flips, not
        # the learned policy. Abstention means "no *confident* winner", not "no information".
        selected = parsed["policy"] or self._best_effort(
            actions, parsed["scores"], parsed["diagnostics"]
        )
        # The choices may be 8-frame Actions or grounded Macros (macros.py); resolve the
        # server's policy name against whatever set the runner passed in.
        by_name = {a.value: a for a in actions}
        action = by_name.get(str(selected))
        if action is None:
            # Server returned a policy name we do not recognise; use the best-guess instead.
            action = by_name[self._best_effort(actions, parsed["scores"], parsed["diagnostics"])]
            abstained = True
        explored = False
        if explore and self._epsilon > 0 and self._rng.random() < self._epsilon:
            action = self._rng.choice(list(actions))
            explored = True
        probabilities = _probabilities(parsed["scores"], actions, action)
        signal = _learning_signal(resp, action.value)
        status_key = (
            "explored"
            if explored
            else ("abstained" if abstained else str(parsed["status"] or "selected"))
        )
        self.status_counts[status_key] = self.status_counts.get(status_key, 0) + 1

        meta_used = self._context_metadata() if self._tcp else None
        if self._tcp:
            self._step += 1  # never reuse an (episode_id, step) pair; frozen eval has no observe()
        self._pending = {
            "decision_id": parsed["decision_id"],
            "policy": action.value,
            "features": features,
            "context_metadata": meta_used,
            "abstained": abstained,
            "explored": explored,
            "selection_status": parsed["status"],
            "reason": parsed["reason"],
            "latency_ms": latency_ms,
            "signal": signal,
        }
        # Dashboard telemetry: what the server actually said, not what we did with it.
        learner = learner_status(resp)
        selection = resp.get("selection") if isinstance(resp, Mapping) else None
        selection = selection if isinstance(selection, Mapping) else {}
        # Which exploration mode the server actually applied on this decision (`selection.mode`:
        # exploit / ucb / thompson — under `auto` it varies per query) and what selected it
        # (`core_unique_argmax`, `core_exploration`, …). Measured 2026-09-24 (findings #35).
        self._pending["mode"] = selection.get("mode")
        self._pending["selected_by"] = selection.get("selected_by")
        values: dict[str, float] = {}
        for name, diag in (parsed["diagnostics"] or {}).items():
            if isinstance(diag, Mapping):
                v = diag.get("sequential_expected_reward")
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    v = diag.get("selection_expected_reward")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values[str(name)] = float(v)
        telemetry = {
            "domain": self._domain,
            "model_type": learner.get("model_type"),
            "validation_skill": learner.get("validation_skill"),
            "sample_count": learner.get("sample_count"),
            "status": status_key,
            "reason": selection.get("reason"),
            "tie_size": selection.get("tie_size"),
            "server_policy": parsed["policy"],
            "explored": explored,
            "values": values,
            "counts": dict(self.status_counts),
            "episode_id": self._episode_id if self._tcp else None,
            "step": (self._step - 1) if self._tcp else None,
            "learning": self._learn,
            "last_feedback": self.last_feedback,
        }
        return Decision(
            action=action,
            confidence=max(probabilities.values()),
            probabilities=probabilities,
            latency_ms=latency_ms,
            selection_status="abstained" if abstained else (parsed["status"] or "selected"),
            expected_reward=signal["expected_reward"],
            sample_count=signal["sample_count"],
            telemetry=telemetry,
        )

    def observe(
        self,
        *,
        next_state: Mapping[str, Any],
        reward: float,
        terminal: bool,
        truncated: bool = False,
        episode_id: str,
        step: int,
    ) -> dict[str, Any] | None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return None
        return self.send_feedback(
            pending,
            next_state=next_state,
            reward=reward,
            terminal=terminal,
            truncated=truncated,
            episode_id=episode_id,
            step=step,
        )

    def send_feedback(
        self,
        pending: Mapping[str, Any],
        *,
        next_state: Mapping[str, Any],
        reward: float,
        terminal: bool,
        truncated: bool = False,
        episode_id: str,
        step: int,
        next_features: Mapping[str, Any] | None = None,
        outcome: str | None = None,
    ) -> dict[str, Any]:
        """Post one feedback for a decision ``pending`` (from ``take_pending`` / ``observe``).

        ``next_state`` is the parser state dict of the transition's end (for the outcome label and,
        unless ``next_features`` is given, the ``values.next_state`` projection). A ``404 decision
        not found`` — the decision id can vanish before a delayed feedback lands (seen mid-ingest
        2026-09-22) — is repaired by re-querying the same context (a Query, not a Record) and
        resending under the fresh id. ``outcome`` overrides the label derived from ``next_state``
        (the delayed loop already has the row's label).
        """
        feedback = {
            "applied": None,
            "context_source": None,
            "observations": None,
            "delta": None,
        }
        if outcome is None:
            outcome = _outcome_label(next_state, reward, terminal)
        self.last_feedback = {
            "reward": float(reward),
            "outcome": outcome,
            "terminal": bool(terminal),
            "sent": bool(self._learn and pending["decision_id"] is not None),
        }
        decision_id = pending["decision_id"]
        if decision_id is not None and self._learn:
            values: dict[str, Any] = {"reward": float(reward)}
            if self._sequential or next_features is not None:
                # Sequential learning needs the NEXT state as a JSON object at the
                # declared next_context_path (same feature projection as the context),
                # and terminal as a JSON boolean (already in metadata). The delayed-credit
                # loop passes the projection it already built (ingest sends it for bandit
                # domains too — the measured contract, offline_client.py).
                values["next_state"] = (
                    dict(next_features)
                    if next_features is not None
                    else context_features(next_state, self._feature_names)
                )
            body = {
                "session_id": "ignored",  # required by DomainFeedbackRequest (value ignored)
                "decision_id": decision_id,
                "relation": self._relation,
                "policy": pending["policy"],
                "feedback_kind": "execution",
                # `outcome` is a required label; `values.reward` is the numeric signal
                # the declared reward component actually learns from.
                "outcome": outcome,
                "context": (
                    {"values": pending["features"], "metadata": pending["context_metadata"]}
                    if pending.get("context_metadata")
                    else {"values": pending["features"]}
                ),
                "values": values,
                "metadata": {
                    "run_id": self._run_id,
                    "episode_id": episode_id,
                    "step": step,
                    "terminal": bool(terminal),
                },
            }
            resp = None
            for attempt in range(2):
                try:
                    _status, resp = self._client.feedback(self._domain, body)
                    break
                except Adapt1Error as exc:
                    if exc.status != 404 or attempt == 1:
                        raise
                    # decision id gone: re-issue one under the same context and resend
                    _s, qr = self._client.query(
                        self._domain,
                        {
                            "session_id": "ignored",
                            "question": QUESTION,
                            "context": body["context"],
                            "top_k": 1,
                            "allow_exploration": False,
                            "selection_mode": "exploit",
                        },
                    )
                    decision_id = _parse_query(qr)["decision_id"]
                    body["decision_id"] = decision_id
            feedback = _parse_feedback(resp)

        record = {
            "episode_id": episode_id,
            "step": step,
            "decision_id": decision_id,
            "policy": pending["policy"],
            "abstained": pending["abstained"],
            "explored": pending.get("explored", False),
            "selection_status": pending["selection_status"],
            "reason": pending["reason"],
            "mode": pending.get("mode"),
            "selected_by": pending.get("selected_by"),
            "latency_ms": round(pending["latency_ms"], 2),
            "reward": float(reward),
            "outcome": outcome,
            "terminal": bool(terminal),
            "truncated": bool(truncated),
            "contextual_learning_applied": feedback["applied"],
            "context_source": feedback["context_source"],
            # The monotonic learner counter + posterior for the executed macro, read from
            # the query response at decision time (the signal that cannot lie).
            "sample_count": pending["signal"]["sample_count"],
            "expected_reward": pending["signal"]["expected_reward"],
            "posterior_alpha": pending["signal"]["posterior_alpha"],
        }
        self._write_trace(record)
        return record

    def _explore(self, actions: Sequence[Action]) -> str:
        return self._rng.choice([a.value for a in actions])

    def _best_effort(self, actions: Sequence[Action], scores: Any, diagnostics: Any = None) -> str:
        """Fallback when the server abstains: the learner's *own* best guess, never a coin flip.

        ⚠️ Measured 2026-09-21 (findings #22): ``policy_scores[].score`` is the CONTEXT-FREE
        mean reward per macro (``policy_support.reward_sum / observations``) — the same vector
        in every state. Using it as the fallback turns eval into a *constant* policy (always
        the macro with the highest mean Δx), which is what #20's "1153" and mario-floor-g's
        cp0 "288" actually were. So prefer the state-dependent numbers in
        ``policy_diagnostics[<macro>]``: ``selection_expected_reward`` (the ranking score the
        server itself abstained on; use it when any is > 0), then
        ``calibrated_contextual_reward`` (nearest-feedback estimate, varies with the state),
        and only then the context-free mean. No scores at all → keep moving forward.
        """
        valid = {a.value for a in actions}

        def argmax(pairs: list[tuple[str, float]]) -> str | None:
            # An all-equal vector carries no information (an EMPTY domain returns 0.0 for every
            # macro — measured live 2026-09-24, findings #34: the frozen eval of a fresh domain
            # picked `noop` 60 times and stood at x=40). Fall through to "keep moving forward".
            values = [v for name, v in pairs if name in valid]
            if not values or all(v == values[0] for v in values):
                return None
            best, best_v = None, float("-inf")
            for name, v in pairs:
                if name in valid and v > best_v:
                    best, best_v = name, v
            return best

        if isinstance(diagnostics, Mapping):
            for key, floor in (
                ("selection_expected_reward", 0.0),
                ("calibrated_contextual_reward", None),
            ):
                pairs = []
                for name, d in diagnostics.items():
                    v = d.get(key) if isinstance(d, Mapping) else None
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        pairs.append((str(name), float(v)))
                if pairs and (floor is None or any(v > floor for _, v in pairs)):
                    best = argmax(pairs)
                    if best is not None:
                        return best
        pairs = [
            (str(s["policy"]), float(s["score"]))
            for s in scores or []
            if isinstance(s, Mapping) and isinstance(s.get("score"), (int, float))
        ]
        best = argmax(pairs)
        if best is not None:
            return best
        for pref in (Action.RIGHT_RUN, Action.RIGHT):
            if pref in actions:
                return pref.value
        return actions[0].value

    def _write_trace(self, record: Mapping[str, Any]) -> None:
        if self._trace is None:
            return
        self._trace.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._trace.flush()
