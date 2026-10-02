"""Parse-cadence-aware parser: the boundary fix for findings #33 (2026-09-24).

``MarioStateParser`` (upstream, byte-identical) computes ``dx``/``dy`` and every enemy's
``relative_velocity_x`` as the difference between two consecutive *parses* and divides
distances by them as if they were per frame. That is correct when it is parsed every frame
(the realtime viewer) and wrong by the parse interval in every headless decision loop, which
parses once per decision (8–96 frames). ``CadenceParser`` records the interval on the snapshot
(``parse_frames``) so the ``full_v2`` / ``apex_v2`` feature layer can divide it back out.

The parser itself is not changed: this subclass only stamps the interval the caller measured
(``Execution.frames``, ``frames_to_apex``, ``ApexEvent.frame`` — all exact; the NES frame
counter at RAM 0x09 is NOT usable for this, it drifts ±1 per macro — measured 2026-09-24,
11 of 55 macros on 1-1 / 1-2).

Contract: ``frames`` is required on every parse after the first of an episode (``reset``
starts a new one). A loop that forgets it fails here, not by silently feeding a v2 domain
per-decision numbers again.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .state import MarioSnapshot, MarioStateParser


@dataclass(frozen=True)
class CadenceSnapshot(MarioSnapshot):
    """A ``MarioSnapshot`` plus the emulator frames elapsed since the previous parse
    (``None`` on the first parse of an episode). ``to_state()`` exposes it as
    ``state["parse_frames"]`` for the v2 feature layer."""

    parse_frames: int | None = None

    def to_state(self) -> dict[str, Any]:
        state = super().to_state()
        state["parse_frames"] = self.parse_frames
        return state


class CadenceParser(MarioStateParser):
    """``MarioStateParser`` whose snapshots carry the parse interval. See the module doc."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cadence_parses = 0

    # ``reset()`` is inherited: it re-runs ``__init__`` (on this subclass), so the parse
    # count restarts with the episode.

    def parse(
        self, info: Any, ram: Any = None, *, frames: int | None = None, **kwargs: Any
    ) -> CadenceSnapshot:  # type: ignore[override]
        if self._cadence_parses > 0 and frames is None:
            raise ValueError(
                "CadenceParser.parse needs frames=<emulator frames since the previous parse> "
                "after the first parse of an episode (findings #33: without it the v2 feature "
                "layer cannot convert per-parse deltas to per-frame units)"
            )
        if frames is not None and frames < 0:
            raise ValueError(f"frames must be >= 0, got {frames}")
        # frames == 0 is legitimate: the apex parse fell on a macro's last frame, so the
        # post-macro parse sees the same emulator frame again (every delta is 0; the v2
        # flattener treats a 0 interval like a first parse — speeds 0, no enemy velocity).
        snapshot = super().parse(info, ram, **kwargs)
        self._cadence_parses += 1
        values = {f.name: getattr(snapshot, f.name) for f in fields(snapshot)}
        return CadenceSnapshot(**values, parse_frames=frames)
