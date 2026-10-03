"""Machina (Adapt-1 trajectory API) adapter for Mario — design in mario/machina-arm.md.

Variant A: one whole-attempt command sequence per proposal. The application owns everything the
docs say it owns: the state row layout, the decoder from continuous rows to NES buttons, the
8-frame command cadence, termination, and the outcome. The API sees numeric arrays only.

Measured 2026-09-22 (findings #25): create / configure / observe cost 1 Record each; state / propose
cost 1 Query each; delete is free. One attempt = 1 Record + 1 Query.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from .actions import Action
from .runner import _unwrap_ram
from .state import MarioSnapshot

LEVEL_LENGTH_PX = 3161.0
# x where the flag is reached, per level (2-1: every offline clear in w21-coverage-v5 ends at 3193). The goal coordinate
# and the outcome divide by the CURRENT level's length; `set_level_length` changes it for a run (`mach2`, 2026-10-03).
# Default 3161 keeps every 1-1 run (#26/#27/#52) and the z7r frozen 2-1 contrast (735) replaying bit for bit.
LEVEL_LENGTHS: dict[str, float] = {"SuperMarioBros-1-1-v0": 3161.0, "SuperMarioBros-2-1-v0": 3193.0}


# A non-clear attempt scores below a clear, always (`mach2`, 2026-10-03): on 2-1 Mario can fly past the pole's x without
# touching it (x 3206 > 3193), which scored 1.004 > the flag's 1.0 and made the overshoot the retained best for 140
# attempts. Changes nothing on 1-1: no non-flag attempt of #27/#52 got past x 2471 (< 0.995 × 3161).
NON_CLEAR_CAP = 0.995


def attempt_outcome(max_x: float, clear: bool) -> float:
    return 1.0 if clear else min(max_x / LEVEL_LENGTH_PX, NON_CLEAR_CAP)


def set_level_length(px: float) -> None:
    global LEVEL_LENGTH_PX
    LEVEL_LENGTH_PX = float(px)


def seed_rows_from_journal(path: Any) -> list[list[float]]:
    """The action rows of the first ``propose`` response in a Machina journal (plain or ``.gz``, e.g. a frozen run's) —
    the sequence a seeded attempt executes instead of the server's proposal."""
    import gzip as _gzip
    import json as _json

    p = Path(path)
    text = _gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
    for line in text.splitlines():
        d = _json.loads(line)
        rec = d.get("record") or {}
        if (
            d.get("kind") == "call"
            and rec.get("op") == "propose"
            and (rec.get("resp") or {}).get("actions")
        ):
            return [list(map(float, r)) for r in rec["resp"]["actions"]]
    raise ValueError(f"no propose response with actions in {path}")


FRAMES_PER_COMMAND = 8
STATE_DIMENSIONS = 8
ACTION_DIMENSIONS = 4
DEFAULT_HORIZON = 160

CONFIG_A: dict[str, Any] = {
    "mechanism": "episode_credit",
    "state_dimensions": STATE_DIMENSIONS,
    "action_dimensions": ACTION_DIMENSIONS,
    "goal_indices": [0],
    "horizon": DEFAULT_HORIZON,
    "capacity": 128,
    "max_pending": 1,
    "seed": 1,
    "outcome_mode": "reward",
}
GOAL = [1.0]  # the flag: x / 3161

# Decoder v4 (application-owned, versioned here). History: v1 thresholds swallowed the learner's
# 0.05-1.0 continuous edits (run 1); v2 duty cycles with a signed direction dithered because half of a
# random sequence walked left (run 2); v3 removed Left entirely (runs 3-6, incl. the seeded flag) —
# the continuous-control equivalent of gym's RIGHT_ONLY set, stated in every write-up. v4 restores
# Left as its OWN channel so the zero-start test carries no action-space asterisk:
#   row = (right, left, run, jump), each in [-1, 1] -> (v + 1) / 2 * 8 frames held.
#   Right has precedence on a frame where both are held (Left acts only on frames Right leaves free),
#   so random sequences still drift right on average (E[right frames] = 4, E[left-only frames] ~ 1.3)
#   while a high Left value genuinely walks left. Left never combines with B or A (no such index).
#   Jump: the button-up edge on frame 0 when grounded so a fresh press fires (findings #20 fix #2).
DECODER_VERSION = 4


def _frames(v: float) -> int:
    """[-1, 1] -> 0..8 frames (duty cycle of the 8-frame command)."""
    return max(
        0,
        min(
            FRAMES_PER_COMMAND,
            round((max(-1.0, min(1.0, v)) + 1.0) / 2.0 * FRAMES_PER_COMMAND),
        ),
    )


def decode_command(row: list[float]) -> list[Action]:
    """One continuous (right, left, run, jump) row -> the 8 per-frame SIMPLE_MOVEMENT actions."""
    r_, l_, run_, j_ = (float(v) for v in row)
    n_right, n_left, n_run, n_jump = _frames(r_), _frames(l_), _frames(run_), _frames(j_)
    frames = []
    for f in range(FRAMES_PER_COMMAND):
        right, left, run, jump = (
            f < n_right,
            (f < n_left) and not (f < n_right),
            f < n_run,
            f < n_jump,
        )
        if left:
            frames.append(Action.LEFT)
        elif right and run and jump:
            frames.append(Action.RIGHT_RUN_JUMP)
        elif right and run:
            frames.append(Action.RIGHT_RUN)
        elif right and jump:
            frames.append(Action.RIGHT_JUMP)
        elif right:
            frames.append(Action.RIGHT)
        elif jump:
            frames.append(Action.JUMP)
        else:
            frames.append(Action.NOOP)
    return frames


def encode_command(frames: list[Action]) -> list[float]:
    """What actually ran, back in native coordinates (duty in [0,1] -> [-1,1]), for the observation.
    Left frames are reported on the Left channel beyond the Right frames (Right precedence), so
    decode(encode(decode(row))) == decode(row)."""
    n = float(FRAMES_PER_COMMAND)
    right = sum(
        1
        for a in frames
        if a in (Action.RIGHT, Action.RIGHT_RUN, Action.RIGHT_JUMP, Action.RIGHT_RUN_JUMP)
    )
    left_frames = sum(1 for a in frames if a is Action.LEFT)
    left = right + left_frames if left_frames else 0
    run = sum(1 for a in frames if a in (Action.RIGHT_RUN, Action.RIGHT_RUN_JUMP))
    jump = sum(1 for a in frames if a in (Action.JUMP, Action.RIGHT_JUMP, Action.RIGHT_RUN_JUMP))
    return [right / n * 2 - 1, left / n * 2 - 1, run / n * 2 - 1, jump / n * 2 - 1]


def decode(row: list[float]) -> Action:
    """The dominant action of a row (first frame) — kept for readouts and tests."""
    return decode_command(row)[0]


def state_row(snap: MarioSnapshot) -> list[float]:
    """8 pre-action numbers in [0,1]-ish; index 0 is the goal coordinate (position fraction)."""
    st = snap.to_state()
    terr, haz = st["terrain"], st["hazard"]

    def cap(v: Any, m: float) -> float:
        return 1.0 if v is None else max(0.0, min(float(v), m)) / m

    return [
        snap.x / LEVEL_LENGTH_PX,
        snap.y / 240.0,
        snap.dx / 3.0,
        snap.dy / 5.0,
        1.0 if snap.grounded else 0.0,
        cap(terr.get("gap_distance_tiles"), 9.0),
        cap(terr.get("obstacle_distance_tiles"), 9.0),
        cap(haz.get("nearest_enemy_distance_pixels"), 128.0),
    ]


def execute_sequence(
    env: Any,
    parser: Any,
    snap: MarioSnapshot,
    rows: list[list[float]],
    *,
    stall_commands: int = 40,
    on_frame: Any = None,
) -> dict[str, Any]:
    """Run the proposal on the emulator with the #20 frame-cadence loop (release edge on a
    grounded jump, 8 frames per command). Returns the executed prefix only: T action rows
    (re-encoded, what actually ran), T+1 state rows, T per-command outcomes (progress fraction,
    0 on the command that died), and the whole-attempt outcome."""
    ram = _unwrap_ram(env)
    states = [state_row(snap)]
    actions: list[list[float]] = []
    step_outcomes: list[float] = []
    decoded: list[str] = []
    max_x = snap.x
    stalled = 0
    dead = clear = False
    from .actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION

    info: Any = None
    terminated = truncated = False
    for row in rows:
        frames = decode_command(row)
        total = 0.0
        for f, a in enumerate(frames):
            action = a
            if f == 0 and a in JUMP_ACTIONS and snap.grounded:
                action = JUMP_RELEASE_ACTION[a]  # button-up edge so a fresh press fires
            frame, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
            if on_frame is not None:
                on_frame(frame)
            total += float(reward)
            if terminated or truncated:
                break
        nxt = parser.parse(info, ram, previous_action=frames[0].value, previous_reward=total)
        dx = nxt.x - snap.x
        dead, clear = bool(nxt.dead or terminated), bool(nxt.clear)
        actions.append(encode_command(frames))
        decoded.append(frames[0].value)
        states.append(state_row(nxt))
        step_outcomes.append(0.0 if (dead and not clear) else max(0.0, dx) / LEVEL_LENGTH_PX)
        if nxt.x > max_x:
            max_x, stalled = nxt.x, 0
        else:
            stalled += 1
        snap = nxt
        if dead or clear or truncated or stalled >= stall_commands:
            break
    outcome = attempt_outcome(max_x, clear)
    assert len(states) == len(actions) + 1 == len(step_outcomes) + 1
    return {
        "states": states,
        "actions": actions,
        "step_outcomes": step_outcomes,
        "decoded": decoded,
        "outcome": outcome,
        "max_x": int(max_x),
        "dead": dead,
        "flag": clear,
        "executed": len(actions),
        "proposed": len(rows),
    }


class Journal:
    """Durable local record of every intent, execution and response — written BEFORE feedback."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, record: Any) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"t": time.time(), "kind": kind, "record": record},
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )


class MachinaClient:
    """`trajectory/{op}` over Adapt1Client (retries 502/503/504 incl. engine_session_loading)."""

    def __init__(self, client: Any, domain_id: str, journal: Journal | None = None) -> None:
        self._c, self.domain, self._j = client, domain_id, journal

    def trajectory(self, op: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        status, resp = self._c.call("POST", f"/domains/{self.domain}/trajectory/{op}", body or {})
        if self._j:
            self._j.write(
                "call",
                {
                    "op": op,
                    "status": status,
                    "ms": round((time.perf_counter() - t0) * 1000),
                    "body": body,
                    "resp": resp,
                },
            )
        return resp

    def create_domain(self, description: str) -> Any:
        return self._c.call(
            "POST", "/domains", {"domain_id": self.domain, "description": description}
        )[1]

    def delete_domain(self) -> Any:
        return self._c.call("DELETE", f"/domains/{self.domain}")[1]


class OfflineMachina:
    """Offline stand-in: random proposals in the declared shape, contract checks on observe. Free."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.config: dict[str, Any] | None = None
        self.pending: set[str] = set()
        self.retained = 0
        self.domain = "offline"

    def create_domain(self, description: str) -> Any:
        return {"domain_id": self.domain}

    def delete_domain(self) -> Any:
        return {}

    def trajectory(self, op: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        if op == "state":
            return {
                "status": "not_configured" if self.config is None else "ready",
                "retained_trajectories": self.retained,
                "pending_decisions": len(self.pending),
            }
        if op == "configure":
            self.config = dict(body["config"])
            return {"config": self.config, "status": "ready"}
        if op == "propose":
            assert self.config, "configure first"
            assert len(body["state"]) == self.config["state_dimensions"], "state length"
            assert len(body["goal"]) == len(self.config["goal_indices"]), "goal length"
            if len(self.pending) >= self.config["max_pending"]:
                raise AssertionError("max_pending exceeded: observe or cancel first")
            did = f"offline-{len(self.pending) + self.retained + 1}"
            self.pending.add(did)
            h, w = self.config["horizon"], self.config["action_dimensions"]
            return {
                "decision_id": did,
                "actions": [[self.rng.uniform(-1, 1) for _ in range(w)] for _ in range(h)],
                "status": "proposed",
                "source": "initial_trajectory_exploration",
            }
        if op == "observe":
            did = body["decision_id"]
            assert did in self.pending, "unknown / already observed decision_id"
            t = len(body["actions"])
            assert t >= 1, "empty executed prefix"
            assert len(body["states"]) == t + 1, "states must be T+1"
            assert all(len(r) == self.config["action_dimensions"] for r in body["actions"]), (
                "action width"
            )
            assert all(len(r) == self.config["state_dimensions"] for r in body["states"]), (
                "state width"
            )
            if "step_outcomes" in body:
                assert len(body["step_outcomes"]) == t, "step_outcomes must be T"
            assert isinstance(body["outcome"], float), "outcome must be a float"
            self.pending.discard(did)
            self.retained += 1
            return {
                "status": "learned",
                "retained_trajectories": self.retained,
                "improved_parent": False,
            }
        raise AssertionError(f"offline stub: unknown op {op}")


def execute_policy(
    env: Any,
    parser: Any,
    snap: MarioSnapshot,
    policy: Any,
    *,
    max_commands: int = DEFAULT_HORIZON,
    stall_commands: int = 40,
) -> dict[str, Any]:
    """Execution OVERRIDE for a seeded attempt: ignore the proposal and run a frame-cadence policy
    (one 8-frame Action per command, release edge on a grounded jump via macros.execute). Returns
    the same trace dict as execute_sequence, with the teacher's actions encoded as duty cycles —
    the observation then reports what actually ran, which is all the learner ever learns from.
    Label any run that uses this as *refined from a demonstration*, not acquired from zero."""
    from .macros import execute

    ram = _unwrap_ram(env)
    states = [state_row(snap)]
    actions: list[list[float]] = []
    step_outcomes: list[float] = []
    decoded: list[str] = []
    max_x = snap.x
    stalled = 0
    dead = clear = False
    for _ in range(max_commands):
        a = policy.choose(snap, tuple(Action)).action
        ex = execute(
            env, a, grounded=snap.grounded, frames_per_decision=FRAMES_PER_COMMAND, ram=ram
        )
        nxt = parser.parse(ex.info, ram, previous_action=a.value, previous_reward=ex.total_reward)
        dx = nxt.x - snap.x
        dead, clear = bool(nxt.dead or ex.terminated), bool(nxt.clear)
        actions.append(encode_command([a] * FRAMES_PER_COMMAND))
        decoded.append(a.value)
        states.append(state_row(nxt))
        step_outcomes.append(0.0 if (dead and not clear) else max(0.0, dx) / LEVEL_LENGTH_PX)
        if nxt.x > max_x:
            max_x, stalled = nxt.x, 0
        else:
            stalled += 1
        snap = nxt
        if dead or clear or ex.truncated or stalled >= stall_commands:
            break
    outcome = attempt_outcome(max_x, clear)
    return {
        "states": states,
        "actions": actions,
        "step_outcomes": step_outcomes,
        "decoded": decoded,
        "outcome": outcome,
        "max_x": int(max_x),
        "dead": dead,
        "flag": clear,
        "executed": len(actions),
        "proposed": max_commands,
        "seeded": True,
    }
