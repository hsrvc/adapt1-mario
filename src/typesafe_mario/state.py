from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

# Tiles to strip from the RAM tile buffer because a player cannot see them until they are
# hit. Measured 2026-09-21: the 1-1 hidden 1-UP block (column 64) is metatile 0x60 and it was
# the ONLY feature separating "top of pipe 3" from "top of pipe 4" (findings #21). 0x5F is the
# hidden coin block per the SMB disassembly metatile table (not exercised in 1-1).
HIDDEN_METATILES: frozenset[int] = frozenset({0x5F, 0x60})

# Forward window in tiles. Measured: while running the camera holds Mario at screen-x ≈ 128
# of 256, so the rendered screen shows ~8 tiles ahead. Reading further would see past the
# screen edge (RAM has 32 columns buffered) — that is the "no more than the screen" line.
FORWARD_TILES = 8
BACKWARD_TILES = 2
# Rows above / below Mario in the local grid. 13 rows = the whole tile buffer height, so a
# 4-tile drop off a pipe top still shows the floor (the old 4-down window did not).
ROWS_ABOVE = 4
ROWS_BELOW = 8
TILE_BUFFER_ROWS = 13
SCREEN_WIDTH_PX = 256

ENEMY_NAMES: dict[int, str] = {
    # 0x00 is the GREEN KOOPA TROOPA in SMB's enemy table, not "none". The upstream parser
    # mapped it to "none" and skipped kind_id == 0, so every green koopa was invisible — the
    # 1-1 koopa at x≈1700 killed both the grounded-cadence teacher (1682) and Jev (1670).
    # Presence is the slot's active byte (0x000F+slot), never the kind id. Measured 2026-09-21.
    0x00: "koopa_green",
    0x06: "goomba",
    0x05: "hammer_bro",
    0x07: "bloober",
    0x0E: "cheep_cheep",
    0x12: "piranha_plant",
    0x2D: "bowser",
    0x31: "flagpole",
}


@dataclass(frozen=True)
class EnemyObservation:
    slot: int
    kind_id: int
    kind: str
    dx_pixels: int
    dy_pixels: int
    relative_velocity_x: int = 0

    def to_state(self) -> dict[str, Any]:
        horizontal = "ahead" if self.dx_pixels >= 0 else "behind"
        return {
            "slot": self.slot,
            "kind": self.kind,
            "kind_id": self.kind_id,
            "relative_x_pixels": self.dx_pixels,
            "relative_y_pixels": self.dy_pixels,
            "relative_velocity_x": self.relative_velocity_x,
            "horizontal_relation": horizontal,
        }


@dataclass(frozen=True)
class MarioSnapshot:
    goal: str
    world: int
    stage: int
    area: int
    x: int
    y: int
    dx: int
    dy: int
    direction: str
    vertical_motion: str
    airborne: bool
    status: str
    player_state: int
    lives: int
    coins: int
    score: int
    time_left: int
    progress: int
    best_progress: int
    stalled_steps: int
    enemies: tuple[EnemyObservation, ...] = field(default_factory=tuple)
    local_grid: tuple[str, ...] = field(default_factory=tuple)
    previous_action: str | None = None
    previous_reward: float = 0.0
    dead: bool = False
    clear: bool = False
    grounded: bool = True
    jump_phase: str = "grounded"
    action_frames: int = 0
    action_progress: int = 0
    airborne_frames: int = 0
    jump_distance_pixels: int = 0
    crossing_gap: bool = False
    gap_width_at_commit: int = 0
    decision_horizon_frames: int = 8
    last_response_delay_frames: int = 0
    last_grounded_gap_distance_tiles: int | None = None
    last_grounded_gap_width_tiles: int = 0
    last_grounded_obstacle_distance_tiles: int | None = None
    last_grounded_obstacle_height_tiles: int = 0
    # Floor profile (2026-09-21, findings #21): for dx = 0..FORWARD_TILES, the floor offset in
    # tiles relative to the ground Mario stands on (0 = level, +k = k tiles LOWER = a drop you
    # can walk off, -k = higher = a step/wall), or None = no floor anywhere below within the
    # tile buffer = a real, bottomless pit. Empty when RAM is unavailable.
    floor_profile: tuple[int | None, ...] = field(default_factory=tuple)

    def navigation_features(self) -> dict[str, Any]:
        """Terrain geometry ahead, from the RAM tile buffer.

        ⚠️ Semantics changed 2026-09-21 (findings #21). The old version called any column
        ahead with no solid tile *in Mario's own ground row* a "gap", so standing on top of a
        4-tall pipe read as "gap 2 tiles ahead, 7 wide" — a phantom. The teacher jumped on it
        at full speed and that max jump landed exactly in the real 2-tile pit 12 tiles on
        (x=1104–1135), the x=1153 death. Now:

        - ``gap_*`` means a **real pit**: a column with no floor at any row below, down to the
          bottom of the tile buffer (``floor_profile`` is None there).
        - ``drop_*`` is new: the floor continues but lower (walk-off-able).
        - ``obstacle_*`` is unchanged: solid tiles rising above Mario's ground row.
        - ``floor_below_tiles`` is the offset under Mario (None = airborne over a pit).

        The forward window stays 8 tiles: measured, the rendered screen shows ~8 tiles ahead
        of Mario while running (screen-x byte 0x03AD ≈ 128 of 256), so 8 is what a player
        sees. Hidden blocks are stripped from the buffer read (see ``HIDDEN_METATILES``).
        """
        rows = self.local_grid
        if not rows:
            return {
                "geometry_available": False,
                "summary": "Local collision geometry is unavailable.",
            }

        mario_row = mario_col = -1
        for row_index, row in enumerate(rows):
            if "M" in row:
                mario_row = row_index
                mario_col = row.index("M")
                break
        if mario_row < 0:
            return {
                "geometry_available": False,
                "summary": "Mario was not located in the collision grid.",
            }

        ground_row = next(
            (row for row in range(mario_row + 1, len(rows)) if rows[row][mario_col] == "#"),
            None,
        )
        profile = self.floor_profile
        obstacle_distance: int | None = None
        obstacle_height = 0
        pit_distance: int | None = None
        pit_width = 0
        drop_distance: int | None = None
        drop_depth = 0
        clear_forward = 0

        for column in range(mario_col + 1, len(rows[0])):
            distance = column - mario_col
            if ground_row is not None:
                height = 0
                row = ground_row - 1
                while row >= 0 and rows[row][column] == "#":
                    height += 1
                    row -= 1
                if height and obstacle_distance is None:
                    obstacle_distance = distance
                    obstacle_height = height
            floor = profile[distance] if distance < len(profile) else 0
            if floor is None:
                if pit_distance is None:
                    pit_distance = distance
                if pit_distance is not None and distance - pit_distance == pit_width:
                    pit_width += 1  # contiguous bottomless columns only
            elif floor > 0 and drop_distance is None and pit_distance is None:
                drop_distance = distance
                drop_depth = floor
            if obstacle_distance is None and pit_distance is None:
                clear_forward = distance

        obstacle_ahead = obstacle_distance is not None and obstacle_distance <= 3
        gap_ahead = pit_distance is not None and pit_distance <= 3
        if obstacle_ahead:
            summary = (
                f"Blocking obstacle {obstacle_distance} tile(s) ahead, "
                f"{obstacle_height} tile(s) high. A forward jump is required."
            )
        elif gap_ahead:
            summary = f"Pit begins {pit_distance} tile(s) ahead. A forward jump is required."
        elif drop_distance is not None and drop_distance <= 3:
            summary = (
                f"Floor drops {drop_depth} tile(s) lower {drop_distance} tile(s) ahead; "
                "it can be walked off."
            )
        else:
            summary = f"Forward path is clear for at least {clear_forward} tile(s)."
        return {
            "geometry_available": True,
            "obstacle_ahead": obstacle_ahead,
            "obstacle_distance_tiles": obstacle_distance,
            "obstacle_height_tiles": obstacle_height,
            "gap_ahead": gap_ahead,
            "gap_distance_tiles": pit_distance,
            "gap_width_tiles_visible": pit_width,
            "drop_distance_tiles": drop_distance,
            "drop_depth_tiles": drop_depth,
            "floor_below_tiles": profile[0] if profile else 0,
            "floor_profile_tiles": list(profile),
            "clear_forward_tiles": clear_forward,
            "summary": summary,
        }

    def threat_features(self) -> dict[str, Any]:
        enemies_ahead = sorted(
            (enemy for enemy in self.enemies if enemy.dx_pixels >= 0),
            key=lambda enemy: enemy.dx_pixels,
        )
        if not enemies_ahead:
            return {
                "enemy_ahead": False,
                "upcoming_enemies": [],
                "spacing_to_second_enemy_pixels": None,
                "nearest_enemy_kind": None,
                "nearest_enemy_distance_pixels": None,
                "relative_velocity_x": None,
                "closing_speed_pixels_per_frame": 0,
                "estimated_contact_frames": None,
                "jump_clearance_frames_required": 8,
                "takeoff_deadline_frames": None,
                "jump_must_start_this_decision": False,
                "takeoff_window_already_missed": False,
                "projected_distance_after_reaction_pixels": None,
                "contact_within_reaction_horizon": False,
                "estimated_landing_frames": None,
                "will_land_before_contact": None,
            }
        nearest = enemies_ahead[0]
        closing_speed = max(0, -nearest.relative_velocity_x)
        contact_frames = round(nearest.dx_pixels / closing_speed) if closing_speed > 0 else None
        reaction_horizon = self.decision_horizon_frames + self.last_response_delay_frames
        projected_distance = max(
            0,
            nearest.dx_pixels + nearest.relative_velocity_x * reaction_horizon,
        )
        landing_frames = None if self.grounded else max(0, 42 - self.airborne_frames)
        jump_clearance_frames = 8
        takeoff_deadline = (
            contact_frames - self.last_response_delay_frames - jump_clearance_frames
            if contact_frames is not None
            else None
        )
        jump_must_start_now = bool(
            self.grounded
            and takeoff_deadline is not None
            and 0 <= takeoff_deadline <= self.decision_horizon_frames
        )
        upcoming_enemies = [
            {
                "kind": enemy.kind,
                "distance_pixels": enemy.dx_pixels,
                "vertical_offset_pixels": enemy.dy_pixels,
                "relative_velocity_x": enemy.relative_velocity_x,
                "projected_distance_after_reaction_pixels": max(
                    0,
                    enemy.dx_pixels + enemy.relative_velocity_x * reaction_horizon,
                ),
            }
            for enemy in enemies_ahead[:3]
        ]
        return {
            "enemy_ahead": True,
            "upcoming_enemies": upcoming_enemies,
            "spacing_to_second_enemy_pixels": (
                enemies_ahead[1].dx_pixels - nearest.dx_pixels if len(enemies_ahead) > 1 else None
            ),
            "nearest_enemy_kind": nearest.kind,
            "nearest_enemy_distance_pixels": nearest.dx_pixels,
            "relative_velocity_x": nearest.relative_velocity_x,
            "closing_speed_pixels_per_frame": closing_speed,
            "estimated_contact_frames": contact_frames,
            "jump_clearance_frames_required": jump_clearance_frames,
            "takeoff_deadline_frames": takeoff_deadline,
            "jump_must_start_this_decision": jump_must_start_now,
            "takeoff_window_already_missed": (
                takeoff_deadline is not None and takeoff_deadline < 0
            ),
            "projected_distance_after_reaction_pixels": projected_distance,
            "contact_within_reaction_horizon": (
                contact_frames is not None and contact_frames <= reaction_horizon
            ),
            "estimated_landing_frames": landing_frames,
            "will_land_before_contact": (
                landing_frames < contact_frames
                if landing_frames is not None and contact_frames is not None
                else None
            ),
        }

    def to_state(self) -> dict[str, Any]:
        terrain = self.navigation_features()
        terrain.pop("summary", None)
        terrain["observation_reliability"] = "high" if self.grounded else "low_airborne"
        terrain["last_grounded_preview"] = {
            "gap_distance_tiles": self.last_grounded_gap_distance_tiles,
            "gap_width_tiles_visible": self.last_grounded_gap_width_tiles,
            "obstacle_distance_tiles": self.last_grounded_obstacle_distance_tiles,
            "obstacle_height_tiles": self.last_grounded_obstacle_height_tiles,
        }
        return {
            "objective": self.goal,
            "level": {"world": self.world, "stage": self.stage, "area": self.area},
            "player": {
                "x": self.x,
                "y": self.y,
                "horizontal_speed_px_per_frame": self.dx,
                "vertical_speed_px_per_frame": self.dy,
                "grounded": self.grounded,
                "jump_phase": self.jump_phase,
                "powerup_status": self.status,
            },
            "trajectory": {
                "airborne_frames": self.airborne_frames,
                "horizontal_distance_since_takeoff_pixels": self.jump_distance_pixels,
                "crossing_known_gap": self.crossing_gap,
                "gap_width_at_commit_tiles": self.gap_width_at_commit,
            },
            "hazard": self.threat_features(),
            "terrain": terrain,
            "reaction_timing": {
                "action_horizon_frames": self.decision_horizon_frames,
                "last_inference_delay_frames": self.last_response_delay_frames,
                "total_reaction_horizon_frames": (
                    self.decision_horizon_frames + self.last_response_delay_frames
                ),
            },
            "recent_control": {
                "action": self.previous_action,
                "frames_observed": self.action_frames,
                "progress_gained_pixels": self.action_progress,
                "outcome": self.control_outcome(),
            },
            "episode": {
                "lives": self.lives,
                "time_left": self.time_left,
                "progress": self.progress,
                "best_progress": self.best_progress,
                "stalled_frames": self.stalled_steps,
                "dead": self.dead,
                "stage_clear": self.clear,
            },
        }

    def control_outcome(self) -> str:
        if self.dead:
            return "death"
        if self.clear:
            return "stage_clear"
        if self.previous_action is None or self.action_frames < 2:
            return "not_enough_evidence"
        if not self.grounded:
            return "jump_in_progress"
        if self.action_progress > 0:
            return "advanced"
        if self.stalled_steps >= 4:
            return "blocked"
        return "no_progress_yet"

    def to_debug_state(self) -> dict[str, Any]:
        return {
            "model_state": self.to_state(),
            "raw": {
                "player_state": self.player_state,
                "coins": self.coins,
                "score": self.score,
                "previous_reward": self.previous_reward,
                "horizontal_motion": self.direction,
                "vertical_motion": self.vertical_motion,
            },
            "visible_enemies": [enemy.to_state() for enemy in self.enemies],
            "local_grid": {
                "legend": {".": "empty", "#": "solid", "E": "enemy", "M": "mario"},
                "orientation": "Mario is at M; columns run left-to-right and rows top-to-bottom.",
                "rows": list(self.local_grid),
            },
        }

    def to_text(self) -> str:
        enemy_text = (
            ", ".join(
                f"{enemy.kind} {abs(enemy.dx_pixels)}px "
                f"{'ahead' if enemy.dx_pixels >= 0 else 'behind'}"
                for enemy in self.enemies
            )
            or "none visible"
        )
        grid = "\n".join(self.local_grid) if self.local_grid else "(unavailable)"
        threat = self.threat_features()
        threat_text = (
            "none visible"
            if not threat["enemy_ahead"]
            else (
                f"{threat['nearest_enemy_kind']} {threat['nearest_enemy_distance_pixels']}px "
                f"ahead; contact estimate={threat['estimated_contact_frames']} frames"
            )
        )
        return (
            f"Goal: {self.goal}\n"
            f"Mario: x={self.x} y={self.y}, {self.direction}, "
            f"vertical={self.vertical_motion}, airborne={self.airborne}, status={self.status}\n"
            f"Progress: {self.progress} (best {self.best_progress}), "
            f"time={self.time_left}, lives={self.lives}, stalled={self.stalled_steps}\n"
            f"Navigation: {self.navigation_features()['summary']}\n"
            f"Threats: {threat_text}\n"
            f"Nearby enemies: {enemy_text}\n"
            "Local grid (# solid, . empty, E enemy, M Mario):\n"
            f"{grid}"
        )


class MarioStateParser:
    """Convert Gym/RAM observations into compact structured game state.

    The RAM grid is deliberately coarse. It communicates local collision geometry,
    not artwork. Addresses follow the original SMB NES memory layout used by the
    Gym wrapper and historical Mario AI tooling.
    """

    def __init__(
        self,
        goal: str = "Reach the flag in World 1-1 without dying.",
        decision_horizon_frames: int = 8,
    ) -> None:
        self.goal = goal
        self.decision_horizon_frames = decision_horizon_frames
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._best_x = 0
        self._stalled_steps = 0
        self._last_enemy_dx: dict[int, int] = {}
        self._tracked_action: str | None = None
        self._action_start_x = 0
        self._action_frames = 0
        self._last_jump_phase = "grounded"
        self._airborne_frames = 0
        self._takeoff_x = 0
        self._crossing_gap = False
        self._gap_width_at_commit = 0
        self._last_grounded_gap_distance: int | None = None
        self._last_grounded_gap_width = 0
        self._last_grounded_obstacle_distance: int | None = None
        self._last_grounded_obstacle_height = 0

    def reset(self) -> None:
        """Clear episode history while preserving the configured objective."""
        self.__init__(
            goal=self.goal,
            decision_horizon_frames=self.decision_horizon_frames,
        )

    @staticmethod
    def _integer(info: Mapping[str, Any], key: str, default: int = 0) -> int:
        value = info.get(key, default)
        return int(value) if value is not None else default

    @staticmethod
    def _ram_byte(ram: Sequence[int] | None, address: int, default: int = 0) -> int:
        if ram is None or address < 0 or address >= len(ram):
            return default
        return int(ram[address])

    def parse(
        self,
        info: Mapping[str, Any],
        ram: Sequence[int] | None = None,
        *,
        previous_action: str | None = None,
        previous_reward: float = 0.0,
        previous_latency_ms: float = 0.0,
        previous_response_delay_frames: int | None = None,
    ) -> MarioSnapshot:
        x = self._integer(info, "x_pos", self._integer(info, "progress"))
        y = self._integer(info, "y_pos", self._integer(info, "y_pixel"))
        screen_y = self._integer(
            info,
            "y_pixel",
            self._ram_byte(ram, 0x03B8, max(0, 255 - y)),
        )
        dx = 0 if self._last_x is None else x - self._last_x
        dy = 0 if self._last_y is None else y - self._last_y

        if dx > 1:
            direction = "moving_right"
        elif dx < -1:
            direction = "moving_left"
        else:
            direction = "nearly_stationary"

        if dy > 1:
            vertical_motion = "rising"
        elif dy < -1:
            vertical_motion = "falling"
        else:
            vertical_motion = "level_or_grounded"

        self._best_x = max(self._best_x, self._integer(info, "progress_max", x), x)
        if self._last_x is None:
            self._stalled_steps = 0
        elif x <= self._last_x:
            self._stalled_steps += 1
        else:
            self._stalled_steps = 0

        enemies = self._extract_enemies(info, ram, x, screen_y)
        grid = self._extract_local_grid(ram, x, screen_y, enemies)
        floor_profile = self._extract_floor_profile(ram, x, screen_y)
        support_below = self._has_support_below(grid)
        # Grounded = the game's own float state (RAM 0x001D == 0), when RAM is available. The
        # old dy-heuristic breaks at the grounded cadence (macros.py): dy is then the y change
        # over a whole macro, so landing ON a pipe top (dy = 64) read as "airborne". Without
        # RAM (unit fixtures) fall back to the heuristic.
        if ram is not None:
            grounded = self._ram_byte(ram, 0x001D, 0) == 0 and support_below
        else:
            grounded = abs(dy) <= 1 and support_below
        if grounded:
            jump_phase = "grounded"
        elif dy > 1:
            jump_phase = "rising"
        elif dy < -1:
            jump_phase = "falling"
        elif self._last_jump_phase == "rising":
            jump_phase = "apex"
        else:
            jump_phase = "airborne"

        if grounded:
            self._airborne_frames = 0
            self._takeoff_x = x
            self._crossing_gap = False
            self._gap_width_at_commit = 0
        else:
            if self._airborne_frames == 0:
                self._takeoff_x = self._last_x if self._last_x is not None else x
                if (
                    self._last_grounded_gap_distance is not None
                    and self._last_grounded_gap_distance <= 3
                ):
                    self._crossing_gap = True
                    self._gap_width_at_commit = self._last_grounded_gap_width
            self._airborne_frames += 1

        if previous_action != self._tracked_action:
            self._tracked_action = previous_action
            self._action_start_x = x
            self._action_frames = 1 if previous_action else 0
        elif previous_action is not None:
            self._action_frames += 1
        action_progress = x - self._action_start_x

        snapshot = MarioSnapshot(
            goal=self.goal,
            world=self._integer(info, "world", 1),
            stage=self._integer(info, "stage", 1),
            area=self._integer(info, "area", 1),
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            direction=direction,
            vertical_motion=vertical_motion,
            airborne=not grounded,
            status=str(info.get("status", "small")),
            player_state=self._integer(info, "player_state", 8),
            lives=self._integer(info, "life", self._integer(info, "lives", 0)),
            coins=self._integer(info, "coins"),
            score=self._integer(info, "score"),
            time_left=self._integer(info, "time"),
            progress=self._integer(info, "progress", x),
            best_progress=self._best_x,
            stalled_steps=self._stalled_steps,
            enemies=tuple(enemies),
            local_grid=tuple(grid),
            previous_action=previous_action,
            previous_reward=float(previous_reward),
            dead=bool(info.get("death", info.get("is_dead", False))),
            clear=bool(info.get("clear", info.get("flag_get", False))),
            grounded=grounded,
            jump_phase=jump_phase,
            action_frames=self._action_frames,
            action_progress=action_progress,
            airborne_frames=self._airborne_frames,
            jump_distance_pixels=max(0, x - self._takeoff_x),
            crossing_gap=self._crossing_gap,
            gap_width_at_commit=self._gap_width_at_commit,
            decision_horizon_frames=self.decision_horizon_frames,
            last_response_delay_frames=(
                max(0, previous_response_delay_frames)
                if previous_response_delay_frames is not None
                else max(0, round(previous_latency_ms / (1000 / 60)))
            ),
            last_grounded_gap_distance_tiles=self._last_grounded_gap_distance,
            last_grounded_gap_width_tiles=self._last_grounded_gap_width,
            last_grounded_obstacle_distance_tiles=self._last_grounded_obstacle_distance,
            last_grounded_obstacle_height_tiles=self._last_grounded_obstacle_height,
            floor_profile=tuple(floor_profile),
        )
        navigation = snapshot.navigation_features()
        if grounded:
            self._last_grounded_gap_distance = navigation.get("gap_distance_tiles")
            self._last_grounded_gap_width = int(navigation.get("gap_width_tiles_visible", 0))
            self._last_grounded_obstacle_distance = navigation.get("obstacle_distance_tiles")
            self._last_grounded_obstacle_height = int(navigation.get("obstacle_height_tiles", 0))
        elif (
            navigation.get("gap_distance_tiles") is not None
            and navigation["gap_distance_tiles"] <= 3
        ):
            self._crossing_gap = True
            self._gap_width_at_commit = max(
                self._gap_width_at_commit,
                int(navigation.get("gap_width_tiles_visible", 0)),
            )
            snapshot = replace(
                snapshot,
                crossing_gap=True,
                gap_width_at_commit=self._gap_width_at_commit,
            )
        self._last_x = x
        self._last_y = y
        self._last_jump_phase = jump_phase
        return snapshot

    @classmethod
    def _tile_solid(cls, ram: Sequence[int], sample_x: int, sample_y: int) -> bool:
        """Whether the SMB tile buffer (0x0500, two 13×16 pages) holds a *visible* solid tile
        at level pixel x / screen pixel y. Hidden blocks read as empty (a player can't see
        them, and they were the only thing separating pipe 3's top from pipe 4's top)."""
        sub_y = (sample_y - 32) // 16
        if not 0 <= sub_y < TILE_BUFFER_ROWS:
            return False
        page = (sample_x // 256) % 2
        sub_x = (sample_x % 256) // 16
        value = cls._ram_byte(ram, 0x0500 + page * TILE_BUFFER_ROWS * 16 + sub_y * 16 + sub_x)
        return value != 0 and value not in HIDDEN_METATILES

    def _extract_floor_profile(
        self, ram: Sequence[int] | None, mario_x: int, mario_screen_y: int
    ) -> list[int | None]:
        """Floor offset per column for dx = 0..FORWARD_TILES (findings #21).

        For each column, scan **downward from the row below Mario to the bottom of the tile
        buffer** for the first solid tile. Offsets are in tiles relative to the floor under
        Mario's own column (0 = level, +k = lower, -k = higher); ``None`` = nothing solid all
        the way down = a real pit. Scanning the whole buffer, not a 4-row window, is what
        makes a 4-tile drop off a pipe top read as a drop instead of a bottomless gap.
        """
        if ram is None:
            return []
        first_row = mario_screen_y + 16  # the row below Mario's own tile
        last_row = 32 + TILE_BUFFER_ROWS * 16  # bottom of the buffer, screen px (exclusive)

        def floor_row(dx_tiles: int) -> int | None:
            sample_x = mario_x + dx_tiles * 16
            sample_y = first_row
            while sample_y < last_row:
                if self._tile_solid(ram, sample_x, sample_y):
                    return (sample_y - first_row) // 16
                sample_y += 16
            return None

        own = floor_row(0)
        reference = 0 if own is None else own
        profile: list[int | None] = []
        for dx in range(FORWARD_TILES + 1):
            row = floor_row(dx)
            profile.append(None if row is None else row - reference)
        return profile

    def floor_height_tiles(
        self, ram: Sequence[int] | None, mario_x: int, mario_screen_y: int
    ) -> int | None:
        """Tiles from the row below Mario down to the first solid tile in his own column
        (0 = standing on it), or None when nothing solid is below (over a pit). The apex
        decision (design-notes §14a) uses it as the height of the arc still to fall."""
        if ram is None:
            return None
        first_row = mario_screen_y + 16
        last_row = 32 + TILE_BUFFER_ROWS * 16
        sample_y = first_row
        while sample_y < last_row:
            if self._tile_solid(ram, mario_x, sample_y):
                return (sample_y - first_row) // 16
            sample_y += 16
        return None

    @staticmethod
    def _has_support_below(grid: Sequence[str]) -> bool:
        for row_index, row in enumerate(grid):
            if "M" not in row:
                continue
            column = row.index("M")
            return any(
                grid[next_row][column] == "#"
                for next_row in range(row_index + 1, min(len(grid), row_index + 3))
            )
        return False

    def _extract_enemies(
        self,
        info: Mapping[str, Any],
        ram: Sequence[int] | None,
        mario_x: int,
        mario_screen_y: int,
    ) -> list[EnemyObservation]:
        enemies: list[EnemyObservation] = []
        fallback_types = tuple(int(value) for value in info.get("enemy_types", ()))
        for slot in range(5):
            active = self._ram_byte(ram, 0x000F + slot, 0)
            kind_id = self._ram_byte(
                ram,
                0x0016 + slot,
                fallback_types[slot] if slot < len(fallback_types) else 0,
            )
            if ram is not None and active == 0:
                continue
            if ram is None and kind_id == 0:
                continue  # no RAM → only the info fallback list, where 0 means empty
            if ram is None:
                enemy_x = mario_x
                enemy_y = mario_screen_y
            else:
                enemy_x = self._ram_byte(ram, 0x006E + slot) * 256 + self._ram_byte(
                    ram, 0x0087 + slot
                )
                enemy_y = self._ram_byte(ram, 0x00CF + slot)
            dx = enemy_x - mario_x
            dy = enemy_y - mario_screen_y
            previous_dx = self._last_enemy_dx.get(slot)
            relative_velocity_x = 0 if previous_dx is None else dx - previous_dx
            self._last_enemy_dx[slot] = dx
            # Visible-screen horizon (2026-09-21): the old fixed -192..320 px reached ~4 tiles
            # past the rendered right edge. Cap to what is on screen, from Mario's on-screen
            # x (RAM 0x03AD): behind to the left edge, ahead to the right edge (+1 sprite).
            if ram is not None:
                screen_x = self._ram_byte(ram, 0x03AD, 128)
                behind_limit = -(screen_x + 16)
                ahead_limit = SCREEN_WIDTH_PX - screen_x
            else:
                behind_limit, ahead_limit = -192, 320
            if behind_limit <= dx <= ahead_limit:
                enemies.append(
                    EnemyObservation(
                        slot=slot,
                        kind_id=kind_id,
                        kind=ENEMY_NAMES.get(kind_id, f"enemy_0x{kind_id:02x}"),
                        dx_pixels=dx,
                        dy_pixels=dy,
                        relative_velocity_x=relative_velocity_x,
                    )
                )
        return enemies

    def _extract_local_grid(
        self,
        ram: Sequence[int] | None,
        mario_x: int,
        mario_screen_y: int,
        enemies: Sequence[EnemyObservation],
    ) -> list[str]:
        if ram is None:
            return []

        x_offsets = range(-BACKWARD_TILES, FORWARD_TILES + 1)
        y_offsets = range(-ROWS_ABOVE, ROWS_BELOW + 1)
        width, height = len(x_offsets), len(y_offsets)
        cells = [["." for _ in range(width)] for _ in range(height)]

        for row, dy_tiles in enumerate(y_offsets):
            for col, dx_tiles in enumerate(x_offsets):
                sample_x = mario_x + dx_tiles * 16
                sample_y = mario_screen_y + dy_tiles * 16
                if self._tile_solid(ram, sample_x, sample_y):
                    cells[row][col] = "#"

        mario_col = BACKWARD_TILES
        mario_row = ROWS_ABOVE
        cells[mario_row][mario_col] = "M"
        for enemy in enemies:
            col = mario_col + round(enemy.dx_pixels / 16)
            row = mario_row + round(enemy.dy_pixels / 16)
            if 0 <= row < height and 0 <= col < width and cells[row][col] != "M":
                cells[row][col] = "E"
        return ["".join(row) for row in cells]
