from __future__ import annotations

import unittest

from typesafe_mario.state import MarioStateParser


class MarioStateParserTests(unittest.TestCase):
    def base_info(self, **overrides: object) -> dict[str, object]:
        info: dict[str, object] = {
            "world": 1,
            "stage": 1,
            "area": 1,
            "x_pos": 100,
            "y_pos": 80,
            "y_pixel": 80,
            "progress": 100,
            "time": 390,
            "life": 2,
            "status": "small",
        }
        info.update(overrides)
        return info

    def test_first_snapshot_is_not_stalled(self) -> None:
        snapshot = MarioStateParser().parse(self.base_info())

        self.assertEqual(snapshot.stalled_steps, 0)
        self.assertEqual(snapshot.direction, "nearly_stationary")

    def test_motion_and_stall_are_derived_across_frames(self) -> None:
        parser = MarioStateParser()
        parser.parse(self.base_info())

        moving = parser.parse(self.base_info(x_pos=108, progress=108, y_pos=75))
        stalled = parser.parse(self.base_info(x_pos=108, progress=108, y_pos=75))

        self.assertEqual(moving.direction, "moving_right")
        self.assertEqual(moving.vertical_motion, "falling")
        self.assertEqual(moving.best_progress, 108)
        self.assertEqual(stalled.stalled_steps, 1)

    def test_ram_enemy_and_grid_become_structured_state(self) -> None:
        ram = bytearray(0x0800)
        ram[0x000F] = 1
        ram[0x0016] = 0x06
        ram[0x006E] = 0
        ram[0x0087] = 142
        ram[0x00CF] = 80
        ram[0x0010] = 1
        ram[0x0017] = 0x06
        ram[0x006F] = 0
        ram[0x0088] = 180
        ram[0x00D0] = 80

        snapshot = MarioStateParser().parse(self.base_info(), ram)
        state = snapshot.to_state()
        debug = snapshot.to_debug_state()

        self.assertEqual(state["hazard"]["nearest_enemy_kind"], "goomba")
        self.assertEqual(state["hazard"]["nearest_enemy_distance_pixels"], 42)
        self.assertEqual(len(state["hazard"]["upcoming_enemies"]), 2)
        self.assertEqual(state["hazard"]["spacing_to_second_enemy_pixels"], 38)
        self.assertEqual(len(debug["local_grid"]["rows"]), 13)  # whole tile-buffer height
        self.assertIn("M", "".join(debug["local_grid"]["rows"]))
        self.assertIn("goomba 42px ahead", snapshot.to_text())

        moving = MarioStateParser()
        moving.parse(self.base_info(), ram)
        next_state = moving.parse(
            self.base_info(x_pos=108, progress=108),
            ram,
            previous_latency_ms=100,
        ).to_state()
        self.assertEqual(next_state["hazard"]["relative_velocity_x"], -8)
        self.assertEqual(next_state["hazard"]["estimated_contact_frames"], 4)
        self.assertTrue(next_state["hazard"]["contact_within_reaction_horizon"])
        self.assertTrue(next_state["hazard"]["takeoff_window_already_missed"])
        self.assertFalse(next_state["hazard"]["jump_must_start_this_decision"])
        self.assertEqual(next_state["reaction_timing"]["last_inference_delay_frames"], 6)
        self.assertEqual(next_state["hazard"]["projected_distance_after_reaction_pixels"], 0)

        deadline_ram = bytearray(0x0800)
        deadline_ram[0x000F] = 1
        deadline_ram[0x0016] = 0x06
        deadline_ram[0x006E] = 0
        deadline_ram[0x0087] = 236
        deadline_ram[0x00CF] = 80
        for column in range(16):
            deadline_ram[0x0500 + 5 * 16 + column] = 1
        deadline_parser = MarioStateParser()
        deadline_parser.parse(self.base_info(), deadline_ram)
        deadline_state = deadline_parser.parse(
            self.base_info(x_pos=108, progress=108),
            deadline_ram,
            previous_latency_ms=100,
            previous_response_delay_frames=8,
        ).to_state()
        self.assertEqual(deadline_state["hazard"]["estimated_contact_frames"], 16)
        self.assertEqual(deadline_state["hazard"]["takeoff_deadline_frames"], 0)
        self.assertEqual(deadline_state["reaction_timing"]["last_inference_delay_frames"], 8)
        self.assertTrue(deadline_state["hazard"]["jump_must_start_this_decision"])

    def test_pipe_geometry_is_exposed_as_navigation_state(self) -> None:
        ram = bytearray(0x0800)
        for column in range(16):
            ram[0x0500 + 5 * 16 + column] = 1
        pipe_column = 7
        ram[0x0500 + 3 * 16 + pipe_column] = 1
        ram[0x0500 + 4 * 16 + pipe_column] = 1

        snapshot = MarioStateParser().parse(self.base_info(), ram)
        terrain = snapshot.to_state()["terrain"]

        self.assertTrue(terrain["obstacle_ahead"])
        self.assertEqual(terrain["obstacle_distance_tiles"], 1)
        self.assertEqual(terrain["obstacle_height_tiles"], 2)
        self.assertNotIn("summary", terrain)

    # --- findings #21 (2026-09-21): floor profile — a drop is not a pit -------------------
    # Fixture geometry (Mario at x_pos=100 → tile column 6, y_pixel=80 → tile row 3):
    #   * a 2-wide "pipe" at columns 6–7 whose top is row 5 (two rows under Mario, the same
    #     support offset real play shows) and which is solid down to the bottom of the buffer
    #   * the real floor at rows 9–10, i.e. 4 tiles below the pipe top
    #   * a 2-wide bottomless pit at columns 11–12 (5 tiles ahead)
    def pipe_top_ram(self, *, hidden_block: bool = False) -> bytearray:
        ram = bytearray(0x0800)
        for column in range(16):
            if column in (11, 12):
                continue
            ram[0x0500 + 9 * 16 + column] = 0x54
            ram[0x0500 + 10 * 16 + column] = 0x54
        for column in (6, 7):
            for row in range(5, 13):
                ram[0x0500 + row * 16 + column] = 0x14
        if hidden_block:
            ram[0x0500 + 4 * 16 + 9] = 0x60  # the 1-1 hidden 1-UP metatile, 3 tiles ahead
        return ram

    def test_drop_off_a_pipe_top_is_not_a_gap(self) -> None:
        terrain = (
            MarioStateParser().parse(self.base_info(), self.pipe_top_ram()).to_state()["terrain"]
        )

        # The old parser read this state as "gap 2 tiles ahead, 7 wide" — the phantom gap that
        # made the teacher max-jump off pipe 4 into the real pit (x=1153 death).
        self.assertFalse(terrain["gap_ahead"])
        self.assertEqual(terrain["drop_distance_tiles"], 2)
        self.assertEqual(terrain["drop_depth_tiles"], 4)
        self.assertEqual(terrain["floor_below_tiles"], 0)
        self.assertIsNone(terrain["obstacle_distance_tiles"])
        self.assertEqual(terrain["floor_profile_tiles"], [0, 0, 4, 4, 4, None, None, 4, 4])

    def test_real_pit_is_reported_with_its_width(self) -> None:
        terrain = (
            MarioStateParser().parse(self.base_info(), self.pipe_top_ram()).to_state()["terrain"]
        )

        self.assertEqual(terrain["gap_distance_tiles"], 5)
        self.assertEqual(terrain["gap_width_tiles_visible"], 2)
        self.assertEqual(terrain["clear_forward_tiles"], 4)

    def test_hidden_blocks_are_invisible(self) -> None:
        # With the hidden 1-UP left in, obstacle_distance would read 3 — and that byte was the
        # only feature separating "top of pipe 3" from "top of pipe 4" in 1-1.
        terrain = (
            MarioStateParser()
            .parse(self.base_info(), self.pipe_top_ram(hidden_block=True))
            .to_state()["terrain"]
        )

        self.assertIsNone(terrain["obstacle_distance_tiles"])
        self.assertFalse(terrain["obstacle_ahead"])

    def test_enemy_horizon_is_the_visible_screen(self) -> None:
        ram = bytearray(0x0800)
        ram[0x03AD] = 128  # Mario at screen centre → 128 px visible ahead
        for slot, enemy_x in enumerate(
            (100 + 120, 100 + 200)
        ):  # 120 px ahead (on screen), 200 (off)
            ram[0x000F + slot] = 1
            ram[0x0016 + slot] = 0x06
            ram[0x006E + slot] = enemy_x // 256
            ram[0x0087 + slot] = enemy_x % 256
            ram[0x00CF + slot] = 80

        snapshot = MarioStateParser().parse(self.base_info(), ram)

        self.assertEqual([e.dx_pixels for e in snapshot.enemies], [120])

    def test_recent_control_tracks_macro_outcome(self) -> None:
        parser = MarioStateParser()
        parser.parse(self.base_info(), previous_action="right_run")
        state = parser.parse(
            self.base_info(x_pos=106, progress=106),
            previous_action="right_run",
        ).to_state()

        self.assertEqual(state["recent_control"]["action"], "right_run")
        self.assertEqual(state["recent_control"]["frames_observed"], 2)
        self.assertEqual(state["recent_control"]["progress_gained_pixels"], 6)

    def test_airborne_state_exposes_trajectory_and_low_reliability_geometry(self) -> None:
        ram = bytearray(0x0800)
        for column in range(16):
            ram[0x0500 + 5 * 16 + column] = 1
        parser = MarioStateParser()
        parser.parse(self.base_info(), ram)

        ram[0x001D] = 1  # the game's float state: airborne (grounded now reads this byte)
        state = parser.parse(self.base_info(x_pos=106, y_pos=90, progress=106), ram).to_state()

        self.assertEqual(state["trajectory"]["airborne_frames"], 1)
        self.assertEqual(state["trajectory"]["horizontal_distance_since_takeoff_pixels"], 6)
        self.assertEqual(state["terrain"]["observation_reliability"], "low_airborne")

    def test_reset_clears_episode_history(self) -> None:
        parser = MarioStateParser(goal="Finish safely")
        parser.parse(self.base_info(x_pos=300, progress=300), previous_action="right_run")

        parser.reset()
        snapshot = parser.parse(self.base_info(x_pos=40, progress=40))

        self.assertEqual(snapshot.goal, "Finish safely")
        self.assertEqual(snapshot.best_progress, 40)
        self.assertEqual(snapshot.stalled_steps, 0)
        self.assertIsNone(snapshot.previous_action)


if __name__ == "__main__":
    unittest.main()
