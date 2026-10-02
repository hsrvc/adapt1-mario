from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .actions import ACTION_DESCRIPTIONS, Action
from .policy import Decision
from .state import MarioSnapshot

Color = tuple[int, int, int]


class DashboardCommand(StrEnum):
    CONTINUE = "continue"
    RESTART = "restart"
    QUIT = "quit"


@dataclass(frozen=True)
class DashboardTheme:
    canvas: Color = (15, 17, 21)
    surface: Color = (22, 25, 31)
    raised: Color = (29, 33, 40)
    line: Color = (48, 54, 64)
    text: Color = (236, 239, 243)
    muted: Color = (141, 150, 163)
    accent: Color = (137, 166, 193)
    accent_soft: Color = (72, 89, 105)
    warning: Color = (210, 167, 100)
    danger: Color = (199, 112, 101)


@dataclass(frozen=True)
class DashboardConfig:
    width: int = 1440
    height: int = 810
    panel_width: int = 472
    fps: int = 60


def clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


class LiveDashboard:
    """One-window game feed and decision-model telemetry display.

    ``brain`` labels which model is driving (default "Adapt-1"); it sets the window
    title, header, and the "querying" status text.
    """

    def __init__(
        self,
        config: DashboardConfig | None = None,
        theme: DashboardTheme | None = None,
        brain: str = "Adapt-1",
    ) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "The dashboard needs pygame. Install the mario extras first."
            ) from exc

        self.pg = pygame
        self.config = config or DashboardConfig()
        self.theme = theme or DashboardTheme()
        self.brain = brain
        pygame.init()
        pygame.display.set_caption(f"{brain} plays Super Mario Bros.")
        size = (self.config.width, self.config.height)
        try:
            # High-DPI (Retina): SCALED lets SDL render at the display's native pixel density
            # instead of drawing at 1x and letting macOS upscale (the "blurry text" problem).
            self.screen = pygame.display.set_mode(size, pygame.SCALED, vsync=1)
        except (pygame.error, TypeError):
            self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        # Font stacks with macOS/Linux fallbacks: the Windows names alone fall back to
        # pygame's default face on a Mac, which is what looked blurry.
        sans = "Segoe UI,Helvetica Neue,Helvetica,Arial,DejaVu Sans"
        sans_bold = "Segoe UI Semibold,Helvetica Neue,Helvetica,Arial,DejaVu Sans"
        mono = "Cascadia Mono,Menlo,Monaco,DejaVu Sans Mono,Courier New"
        self.font_title = pygame.font.SysFont(sans_bold, 25, bold=True)
        self.font_action = pygame.font.SysFont(sans_bold, 31, bold=True)
        self.font_body = pygame.font.SysFont(sans, 17)
        self.font_small = pygame.font.SysFont(sans, 14)
        self.font_label = pygame.font.SysFont(sans_bold, 14, bold=True)
        self.font_mono = pygame.font.SysFont(mono, 14)

    def close(self) -> None:
        self.pg.quit()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pg.image.save(self.screen, str(path))

    def _text(
        self,
        value: str,
        font: Any,
        color: Color,
        x: int,
        y: int,
    ) -> None:
        self.screen.blit(font.render(value, True, color), (x, y))

    def _wrapped(
        self, value: str, font: Any, color: Color, x: int, y: int, width: int, line_height: int = 18
    ) -> int:
        """Draw text wrapped to `width` px; returns the y after the last line."""
        words = value.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if font.size(candidate)[0] <= width or not line:
                line = candidate
            else:
                self._text(line, font, color, x, y)
                y += line_height
                line = word
        if line:
            self._text(line, font, color, x, y)
            y += line_height
        return y

    def _rule(self, x: int, y: int, width: int) -> None:
        self.pg.draw.line(self.screen, self.theme.line, (x, y), (x + width, y), 1)

    def _bar(
        self,
        *,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        selected: bool = False,
        color: Color | None = None,
    ) -> None:
        value = clamp01(value)
        bar_color = color or (self.theme.accent if selected else self.theme.accent_soft)
        self._text(label, self.font_small, self.theme.text if selected else self.theme.muted, x, y)
        value_text = f"{value * 100:4.1f}%"
        text_width = self.font_small.size(value_text)[0]
        self._text(value_text, self.font_small, self.theme.muted, x + width - text_width, y)
        bar_y = y + 21
        self.pg.draw.rect(self.screen, self.theme.raised, (x, bar_y, width, 7), border_radius=3)
        fill = max(2, round(width * value)) if value > 0 else 0
        if fill:
            self.pg.draw.rect(self.screen, bar_color, (x, bar_y, fill, 7), border_radius=3)

    def _draw_game(self, frame: Any, x: int, y: int, width: int, height: int) -> None:
        self.pg.draw.rect(self.screen, self.theme.surface, (x, y, width, height), border_radius=4)
        if frame is None:
            self._text(
                "Waiting for emulator frame", self.font_body, self.theme.muted, x + 28, y + 28
            )
            return
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("The dashboard needs numpy.") from exc

        pixels = np.asarray(frame)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise ValueError(f"Expected an RGB frame, received shape {pixels.shape!r}")
        pixels = pixels[:, :, :3]
        source = self.pg.surfarray.make_surface(pixels.swapaxes(0, 1))
        source_ratio = source.get_width() / source.get_height()
        target_ratio = width / height
        if source_ratio > target_ratio:
            target_width = width
            target_height = round(width / source_ratio)
        else:
            target_height = height
            target_width = round(height * source_ratio)
        scaled = self.pg.transform.scale(source, (target_width, target_height))
        target_x = x + (width - target_width) // 2
        target_y = y + (height - target_height) // 2
        self.screen.blit(scaled, (target_x, target_y))
        self.pg.draw.rect(self.screen, self.theme.line, (x, y, width, height), 1, border_radius=4)

    @staticmethod
    def _action_order(names: Any) -> list[str]:
        """Stable row order for value/probability bars: the run's own action set order."""
        from .macros import Macro

        present = set(names)
        for enum in (Macro, Action):
            ordered = [m.value for m in enum if m.value in present]
            if ordered and len(ordered) == len(present):
                return ordered
        return sorted(present)

    def _restart_rect(self) -> Any:
        return self.pg.Rect(self.config.width - 144, 13, 120, 36)

    def draw(
        self,
        frame: Any,
        snapshot: MarioSnapshot,
        decision: Decision | None,
        *,
        decision_index: int,
        episode_reward: float,
        waiting: bool = False,
        run_ended: bool = False,
        ended_reason: str | None = None,
    ) -> DashboardCommand:
        restart_rect = self._restart_rect()
        command = DashboardCommand.CONTINUE
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return DashboardCommand.QUIT
            if event.type == self.pg.KEYDOWN and event.key in (self.pg.K_ESCAPE, self.pg.K_q):
                return DashboardCommand.QUIT
            if event.type == self.pg.KEYDOWN and event.key == self.pg.K_r:
                command = DashboardCommand.RESTART
            if (
                event.type == self.pg.MOUSEBUTTONDOWN
                and event.button == 1
                and restart_rect.collidepoint(event.pos)
            ):
                command = DashboardCommand.RESTART

        c = self.config
        t = self.theme
        self.screen.fill(t.canvas)
        margin = 24
        header_h = 58
        panel_x = c.width - c.panel_width
        game_x = margin
        game_y = header_h + 14
        game_w = panel_x - margin * 2
        game_h = c.height - game_y - margin

        self._text(f"{self.brain} plays Mario", self.font_title, t.text, margin, 18)
        if run_ended:
            status = f"Run ended · {ended_reason}" if ended_reason else "Run ended"
        else:
            status = f"Querying {self.brain}…" if waiting else "Live decision loop"
        status_width = self.font_small.size(status)[0]
        dot_x = panel_x - status_width - 24
        status_color = t.danger if run_ended else (t.warning if waiting else t.accent)
        self.pg.draw.circle(self.screen, status_color, (dot_x, 31), 4)
        self._text(status, self.font_small, t.muted, dot_x + 12, 21)
        hovered = restart_rect.collidepoint(self.pg.mouse.get_pos())
        button_color = t.accent_soft if hovered else t.raised
        self.pg.draw.rect(self.screen, button_color, restart_rect, border_radius=5)
        self.pg.draw.rect(self.screen, t.accent, restart_rect, 1, border_radius=5)
        label = "Restart"
        label_width, label_height = self.font_label.size(label)
        self._text(
            label,
            self.font_label,
            t.text,
            restart_rect.centerx - label_width // 2,
            restart_rect.centery - label_height // 2,
        )
        self._rule(margin, header_h, c.width - margin * 2)
        self._draw_game(frame, game_x, game_y, game_w, game_h)

        x = panel_x + 18
        width = c.panel_width - 42
        y = game_y
        self._text("World 1–1", self.font_label, t.muted, x, y)
        self._text(f"Decision {decision_index:04d}", self.font_small, t.muted, x + width - 100, y)
        y += 27

        action_name = (
            "Reading state…" if decision is None else decision.action.value.replace("_", " ")
        )
        self._text(action_name, self.font_action, t.text, x, y)
        y += 42
        if decision is None:
            description = f"{self.brain} is selecting the next controller macro."
        else:
            # Grounded-cadence Macros are not in ACTION_DESCRIPTIONS (findings #21).
            description = ACTION_DESCRIPTIONS.get(
                decision.action, "Committed move: buttons held until Mario lands."
            )
        self._text(description, self.font_small, t.muted, x, y)
        y += 32
        self._rule(x, y, width)
        y += 18

        confidence = decision.confidence if decision else 0.0
        latency = decision.latency_ms if decision else 0.0
        metric_width = width // 3
        tel = decision.telemetry if decision is not None else None
        if tel is not None:
            decided_by = (
                "explored"
                if tel.get("explored")
                else "server"
                if tel.get("status") == "selected"
                else "fallback (argmax)"
                if tel.get("status") == "abstained"
                else str(tel.get("status") or "—")
            )
            if tel.get("replay"):
                decided_by += " (replayed)"
            first_metric = ("Decided by", decided_by)
        elif decision is not None and decision.selection_status is not None:
            first_metric = ("Decided by", decision.selection_status)
        else:
            first_metric = ("Confidence", f"{confidence * 100:.0f}%")
        metrics = (
            first_metric,
            ("Latency", f"{latency:.0f} ms"),
            ("Episode reward", f"{episode_reward:+.3f}"),
        )
        for index, (label, value) in enumerate(metrics):
            metric_x = x + index * metric_width
            self._text(label, self.font_small, t.muted, metric_x, y)
            self._text(value, self.font_body, t.text, metric_x, y + 19)
        y += 58
        telemetry = decision.telemetry if decision is not None else None
        adapt1 = telemetry is not None or (
            decision is not None and decision.selection_status is not None
        )
        selected_action = decision.action.value if decision else None
        if telemetry:
            # --- Adapt-1 panel (findings #22): show what the SERVER said, not what we did with it.
            model = telemetry.get("model_type") or "no model"
            skill = telemetry.get("validation_skill")
            skill_text = f"skill {skill:.2f}" if isinstance(skill, (int, float)) else "skill —"
            samples = telemetry.get("sample_count")
            mode = "learning" if telemetry.get("learning") else "frozen"
            self._text(f"Adapt-1 · {telemetry.get('domain')}", self.font_label, t.text, x, y)
            y += 22
            self._text(
                f"{model} · {skill_text} · {samples} samples · {mode}",
                self.font_small,
                t.muted,
                x,
                y,
            )
            y += 20
            cost = (
                "Cost  replay · no API calls, nothing spent"
                if telemetry.get("replay")
                else "Cost  live · 1 Query per decision, 0 Records (frozen)"
                if not telemetry.get("learning")
                else "Cost  live · 1 Query + 1 Record per decision (learning)"
            )
            self._text(
                cost, self.font_mono, t.accent if telemetry.get("learning") else t.muted, x, y
            )
            y += 24
            status = telemetry.get("status") or "—"
            reason = telemetry.get("reason")
            tie = telemetry.get("tie_size")
            status_color = (
                t.warning if status == "abstained" else t.accent if status == "explored" else t.text
            )
            detail = f" ({reason}{f', tie {tie}' if tie else ''})" if reason else ""
            self._text(f"Selection  {status}{detail}", self.font_mono, status_color, x, y)
            y += 20
            counts = telemetry.get("counts") or {}
            self._text(
                f"This run   selected {counts.get('selected', 0)} · abstained "
                f"{counts.get('abstained', 0)} · explored {counts.get('explored', 0)}",
                self.font_mono,
                t.muted,
                x,
                y,
            )
            y += 26
            self._text("Learned value per macro", self.font_label, t.text, x, y)
            y += 22
            values = telemetry.get("values") or {}
            top = max(values.values()) if values else 0.0
            order = self._action_order(values.keys())
            for name in order[:8]:
                v = float(values.get(name, 0.0))
                self._bar(
                    label=f"{name.replace('_', ' ')}  {v:.3f}",
                    value=(v / top if top > 0 else 0.0),
                    x=x,
                    y=y,
                    width=width,
                    selected=name == selected_action,
                )
                y += 30
            if not values:
                y = (
                    self._wrapped(
                        "no per-macro values in the response (tie / no model)",
                        self.font_small,
                        t.warning,
                        x,
                        y,
                        width,
                    )
                    + 4
                )
            last = telemetry.get("last_feedback")
            if last:
                sent = "sent" if last.get("sent") else "not sent (frozen)"
                self._text(
                    f"Last feedback  r={last['reward']:+.4f} · {last['outcome']} · {sent}",
                    self.font_mono,
                    t.muted,
                    x,
                    y,
                )
                y += 20
            if telemetry.get("episode_id"):
                self._text(
                    f"TCP  {telemetry['episode_id']} · step {telemetry.get('step')}",
                    self.font_mono,
                    t.muted,
                    x,
                    y,
                )
                y += 20
            apex = telemetry.get("apex")
            if apex:
                vals = apex.get("values") or {}
                short = " ".join(
                    f"{k.removeprefix('apex_')[:4]}={v:.3f}" for k, v in sorted(vals.items())
                )
                y = (
                    self._wrapped(
                        f"Apex x={apex.get('x')}  {apex.get('choice')} · {apex.get('status')}"
                        + (f"  {short}" if short else ""),
                        self.font_mono,
                        t.accent,
                        x,
                        y,
                        width,
                    )
                    + 4
                )
            nav = snapshot.navigation_features()
            terrain = (
                f"Terrain  pit {nav.get('gap_distance_tiles')}/{nav.get('gap_width_tiles_visible')}"
                f" · drop {nav.get('drop_distance_tiles')}/{nav.get('drop_depth_tiles')}"
                f" · wall {nav.get('obstacle_distance_tiles')}"
            )
            self._text(terrain, self.font_mono, t.muted, x, y)
            y += 24
            self._rule(x, y + 2, width)
            y += 17
        else:
            probabilities = dict(decision.probabilities) if decision else {}
            names = (
                self._action_order(probabilities.keys())
                if probabilities
                else [a.value for a in Action]
            )
            self._text(
                "Controller probabilities" if not adapt1 else "Macro scores (context-free share)",
                self.font_label,
                t.text,
                x,
                y,
            )
            y += 24
            for name in names[:8]:
                self._bar(
                    label=name.replace("_", " "),
                    value=float(probabilities.get(name, 0.0)),
                    x=x,
                    y=y,
                    width=width,
                    selected=name == selected_action,
                )
                y += 36
            self._rule(x, y + 2, width)
            y += 17
        if adapt1 and not telemetry:
            note = (
                "Replay of a log recorded before decision telemetry was stored — "
                "moves only, no server values. Cost: nothing, no API calls."
                if decision
                and decision.selection_status
                and decision.selection_status.startswith("replay")
                else f"Selection   {decision.selection_status if decision else '—'}"
            )
            y = self._wrapped(note, self.font_small, t.muted, x, y, width) + 8
        elif not adapt1:
            self._text("Situation", self.font_label, t.text, x, y)
            y += 24
            jump = decision.jump_needed_probability if decision else None
            danger = clamp01((decision.danger_score or 0.0) / 2.0) if decision else 0.0
            self._bar(label="Jump useful now", value=clamp01(jump), x=x, y=y, width=width)
            y += 40
            danger_color = t.danger if danger >= 0.66 else t.warning
            self._bar(
                label="Immediate danger", value=danger, x=x, y=y, width=width, color=danger_color
            )
            y += 43

        enemies = ", ".join(enemy.kind for enemy in snapshot.enemies) or "none"
        state_lines = (
            f"Position  {snapshot.x}, {snapshot.y}",
            f"Motion    {snapshot.direction.replace('_', ' ')}",
            f"Nearby    {enemies}",
            f"Time      {snapshot.time_left}",
        )
        for line in state_lines:
            self._text(line, self.font_mono, t.muted, x, y)
            y += 20

        footer = (
            "Click Restart or press R to play again" if run_ended else "R restarts · Esc or Q quits"
        )
        self._text(footer, self.font_small, t.muted, x, c.height - 28)
        self.pg.display.flip()
        self.clock.tick(c.fps)
        return command
