# HISTORICAL — apex_coverage.py as of commit 7631050 (2026-09-23 00:28), the version that wrote
# artifacts/demos/apex-coverage-v3.jsonl (the 4-macro coverage pool of the 1-1 clear, findings #28), patched on
# 2026-09-25 only to parse through CadenceParser with the executed frames and to emit the v2 feature superset.
# Regenerates that file row for row (865/865) with v2 features. Do not edit. Run from mario/agent:
#   PYTHONPATH=src ../.venv/bin/python scripts/historical/apex_coverage_7631050_v2.py --out artifacts/demos/apex-coverage-v3-full_v2.jsonl
#!/usr/bin/env python
"""Offline, FREE: branch rollouts at the states that matter — coverage rows for the apex arm.

Noisy demos never produce the arc that decides 1-1 (the full jump off the pipe-4 top: 0 of 452
apex rows in a 30-episode eps=0.10 set), because the teacher runs off that pipe and only the
*learner* jumps from it. This generator plays the deterministic grounded teacher through the
level and, at every grounded decision that matches ``--where`` (a drop, pit or obstacle inside
the window, plus every ``--stride``-th other state), saves the emulator and executes every
branch: each macro in ``--macros`` x each ``ApexChoice`` (jump macros only), then the teacher
for ``--k`` more decisions, then rewinds. Each branch yields its decision rows (takeoff, and
apex for jumps) with ``reward`` = the client-side k-step return of that branch
(``typesafe_mario.returns``) and ``immediate_reward`` kept — the same row shape as
``record_demos.py --apex``, ready for ``ingest_demos.py --bandit --kind takeoff|apex``.

This is reset-to-state sampling: the emulator is rewound, so one state is tried under every
action. It is the information a teacher with exploration would produce there, gathered
exhaustively where it matters. Say so in any write-up. ``explored`` marks branches that differ
from the teacher's own choice.

    PYTHONPATH=src python scripts/apex_coverage.py --out artifacts/demos/apex-coverage.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from typesafe_mario.adapt1_policy import APEX_EXTRA_NAMES, FEATURE_NAMES  # noqa: E402
from typesafe_mario.demos import apex_hook, build_rows  # noqa: E402
from typesafe_mario.macros import ApexChoice, Macro, action_set, execute, is_jump  # noqa: E402
from typesafe_mario.policy import Decision, DiversifiedHeuristicPolicy  # noqa: E402
from typesafe_mario.returns import kstep_return  # noqa: E402
from typesafe_mario.reward import PROGRESS_FRACTION  # noqa: E402
from typesafe_mario.runner import _unwrap_ram, create_mario_env  # noqa: E402
from typesafe_mario.cadence import CadenceParser as MarioStateParser  # noqa: E402

from typesafe_mario.adapt1_policy import FULL_V3_FEATURE_NAMES  # noqa: E402  (v3 superset: v2 arms project by name)
NAMES = tuple(dict.fromkeys((*FULL_V3_FEATURE_NAMES, *APEX_EXTRA_NAMES)))
DEFAULT_MACROS = "right_run,right_run_jump_short,right_run_jump_mid,right_run_jump_full"


class Forced:
    """A policy that answers one fixed apex choice for the branch, and records what the
    teacher's own apex rule would have chosen there (``teacher_choice``)."""

    def __init__(self, action, explored: bool, teacher) -> None:
        self.action, self.explored, self.teacher = action, explored, teacher
        self.teacher_choice = None

    def choose(self, snapshot, actions, extra=None) -> Decision:
        self.teacher_choice = self.teacher._apex_rule(snapshot, extra or {}, list(actions)).value
        return Decision(action=self.action, confidence=1.0,
                        probabilities={a.value: float(a == self.action) for a in actions},
                        latency_ms=0.0, selection_status="explored" if self.explored else None)


def interesting(nav: dict, hazard: dict | None = None) -> bool:
    """A state worth branching at: terrain inside the window, or (2026-09-22, 2-1 diagnosis) an enemy
    within 6 tiles ahead — the 1-1 policy died on 2-1 to a kicked shell at a state no terrain rule marks."""
    d, g, o = nav.get("drop_distance_tiles"), nav.get("gap_distance_tiles"), nav.get("obstacle_distance_tiles")
    e = (hazard or {}).get("nearest_enemy_distance_pixels")
    return ((d is not None and d <= 4) or (g is not None and g <= 6) or (o is not None and o <= 4)
            or (e is not None and 0 <= float(e) <= 96))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--macros", default=DEFAULT_MACROS, help="comma-separated macros to branch on")
    ap.add_argument("--stride", type=int, default=0,
                    help="also branch at every N-th uninteresting grounded state (0 = only interesting)")
    ap.add_argument("--k", type=int, default=6, help="teacher continuation decisions per branch")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    macros = [Macro(m.strip()) for m in args.macros.split(",") if m.strip()]
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0, rng_seed=args.seed)
    env = create_mario_env(args.env, render_mode="rgb_array")
    base = env.unwrapped
    parser = MarioStateParser(decision_horizon_frames=8)
    _f, info = env.reset(seed=args.seed)
    ram = _unwrap_ram(env)
    snap = parser.parse(info, ram, previous_action=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sink = args.out.open("w")
    n_rows = n_points = n_branches = 0
    summary: dict[str, list] = {}

    def run_decision(policy, decision, p, s):
        """Execute one decision with the apex look; return (rows-without-returns, next_snapshot)."""
        a_state: dict = {}
        ex = execute(env, decision.action, grounded=s.grounded, ram=ram,
                     apex=apex_hook(p, env, policy, decision, a_state))
        fired = ex.apex is not None and bool(a_state)
        nxt = p.parse(ex.info, ram,
                      previous_action=(a_state["decision"].action.value if fired else decision.action.value),
                      previous_reward=(ex.total_reward - ex.reward_to_apex if fired else ex.total_reward), frames=(ex.frames - ex.frames_to_apex if (ex.apex is not None and fired) else ex.frames))
        terminal = bool(ex.terminated or nxt.dead or nxt.clear)
        rows = build_rows(snapshot=s, decision=decision, execution=ex,
                          apex_state=a_state if fired else None, next_snapshot=nxt, terminal=terminal,
                          names=NAMES, reward_config=PROGRESS_FRACTION, episode_id="", first_step=0,
                          tag_kind=True)
        return rows, nxt, terminal

    for d in range(args.max_decisions):
        if snap.dead or snap.clear:
            break
        own = teacher.choose(snap, action_set("grounded"))
        nav = snap.navigation_features()
        branch_here = snap.grounded and (interesting(nav, snap.threat_features()) or (args.stride and d % args.stride == 0))
        if branch_here:
            n_points += 1
            base._backup()
            frozen_parser = copy.deepcopy(parser)
            x0 = snap.x
            for macro in macros:
                for choice in (list(ApexChoice) if is_jump(macro) else [None]):
                    base._restore()
                    base.done = False
                    p = copy.deepcopy(frozen_parser)
                    explored = macro is not own.action
                    branch_policy = Forced(choice, explored, teacher) if choice is not None else teacher
                    first_rows, nxt, terminal = run_decision(
                        branch_policy, Decision(action=macro, confidence=1.0, probabilities={},
                                                latency_ms=0.0,
                                                selection_status="explored" if explored else None),
                        p, snap)
                    rewards = [float(r["reward"]) for r in first_rows]
                    frames = [int(r["frames"]) for r in first_rows]
                    # Teacher continuation for the return (its own apex rule in the air).
                    s = nxt
                    cont = 0
                    while not terminal and cont < args.k:
                        dec = teacher.choose(s, action_set("grounded"))
                        rows, s, terminal = run_decision(teacher, dec, p, s)
                        rewards.extend(float(r["reward"]) for r in rows)
                        frames.extend(int(r["frames"]) for r in rows)
                        cont += 1
                    ep_id = f"cov-d{d:03d}-x{x0}-{macro.value}" + (f"-{choice.value}" if choice else "")
                    # What the teacher's own apex rule would have done at this branch's apex, so
                    # the takeoff domain can be fed only the rows played under that rule
                    # (ingest_demos.py --takeoff-under teacher) instead of pooling all three.
                    teacher_choice = branch_policy.teacher_choice if choice is not None else None
                    for i, r in enumerate(first_rows):
                        r["episode_id"] = ep_id
                        r["step"] = i
                        r["immediate_reward"] = r["reward"]
                        r["reward"] = kstep_return(rewards, i, args.k, args.gamma)
                        # Per-frame k-step return (findings #29): progress fraction per 8 frames over the
                        # same horizon, so a long macro is not paid more just for lasting longer.
                        w = rewards[i:i + args.k]; f = frames[i:i + args.k]
                        r["reward_rate"] = round(min(1.0, sum(w) / max(1, sum(f)) * 8.0), 6)
                        r["x_takeoff"] = x0
                        r["branch_return_rewards"] = len(rewards)
                        r["branch_apex"] = choice.value if choice is not None else None
                        r["teacher_apex"] = teacher_choice
                        sink.write(json.dumps(r, separators=(",", ":")) + "\n")
                        n_rows += 1
                    n_branches += 1
                    key = (x0, macro.value, choice.value if choice else "-")
                    summary[key] = [round(first_rows[0]["reward"], 4),
                                    first_rows[-1]["reward"] if len(first_rows) > 1 else None,
                                    "DEAD" if (terminal and s.dead) else f"x={s.x}"]
            base._restore()
            base.done = False
        # Main line: the teacher's own decision (apex rule in the air), real parser.
        _rows, snap, terminal = run_decision(teacher, own, parser, snap)
        if terminal:
            break
    sink.close()
    env.close()
    print(f"branch points {n_points}, branches {n_branches}, rows {n_rows} -> {args.out}")
    print("per branch: takeoff-row return / apex-row return / where the branch ended after the k continuation")
    last_x = None
    for (x0, macro, choice), (r0, r1, end) in summary.items():
        if x0 != last_x:
            print(f"  x={x0}")
            last_x = x0
        print(f"     {macro:<22} {choice:<15} {r0:.4f}  {'-' if r1 is None else f'{r1:.4f}'}  {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
