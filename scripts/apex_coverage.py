#!/usr/bin/env python
"""Offline, FREE: branch rollouts at the states that matter — coverage rows for the apex arm.

Noisy demos never produce the arc that decides 1-1 (the full jump off the pipe-4 top: 0 of 452
apex rows in a 30-episode eps=0.10 set), because the teacher runs off that pipe and only the
*learner* jumps from it. This generator plays the deterministic grounded teacher through the
level and, at every grounded decision that matches ``--where`` (a drop, pit or obstacle inside
the window, an enemy within 6 tiles, plus every ``--stride``-th other state), saves the emulator
and executes every branch: each macro in ``--macros`` x each ``ApexChoice`` (forward jump macros
only), then the teacher until ``--k`` more decisions AND ``--frames-horizon`` more frames have
passed, then rewinds. Each branch yields its decision rows (takeoff, and apex for jumps) with two
returns: ``reward`` = the client-side k-step (per-decision) return of that branch and
``reward_rate`` = the fixed frame-horizon return (``typesafe_mario.returns``) — the same row
shape as ``record_demos.py --apex``, ready for ``ingest_demos.py --bandit --kind takeoff|apex``.

``--follow best`` (2026-09-23) turns the generator into a lookahead teacher: at every branch
point the MAIN LINE takes the branch with the best frame-horizon return instead of the teacher's
own rule, so the run continues past the states where the rule dies (2-1: x=1651). When every
one-step branch is worthless (no progress inside the horizon — a wall), it searches deeper: a
breadth-first search over macro sequences (``--plan-depth``), de-duplicated on the emulator
state, until one advances ``--plan-goal`` px; the main line then follows that plan and branch
continuations that enter the plan's states follow it too. No level knowledge is used — the
planner only knows "progress". With ``--write-mainline`` the main line's own decisions are
written too (episode ``main-...``), which is the only demonstration of a level the rule cannot
finish.

This is reset-to-state sampling: the emulator is rewound, so one state is tried under every
action. It is the information a teacher with exploration would produce there, gathered
exhaustively where it matters — and with ``--follow best`` the demonstration itself is chosen
by rewinding. Say so in any write-up. ``explored`` marks branches that differ from the
teacher's own choice.

    PYTHONPATH=src python scripts/apex_coverage.py --out artifacts/demos/apex-coverage.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, "src")

from typesafe_mario.adapt1_policy import APEX_EXTRA_NAMES, FEATURE_NAMES, FULL_V3_FEATURE_NAMES  # noqa: E402
from typesafe_mario.demos import apex_hook, build_rows  # noqa: E402
from typesafe_mario.macros import (  # noqa: E402
    X_SPEED_ADDR, ApexChoice, Macro, _float_state, action_set, execute, has_apex,
)
from typesafe_mario.policy import Decision, DiversifiedHeuristicPolicy  # noqa: E402
from typesafe_mario.returns import frame_horizon_return, kstep_return  # noqa: E402
from typesafe_mario.reward import PROGRESS_FRACTION  # noqa: E402
from typesafe_mario.runner import _unwrap_ram, create_mario_env  # noqa: E402
from typesafe_mario.cadence import CadenceParser  # noqa: E402
from typesafe_mario.provenance import write_provenance  # noqa: E402

NAMES = tuple(dict.fromkeys((*FEATURE_NAMES, *APEX_EXTRA_NAMES)))
# --features-version 2: the same superset in per-frame units, no `reliability` (findings #33).
# The v2-family superset carries obstacle_height too (v3 arms select it; v2 arms ignore it — projection is by name).
NAMES_V2 = tuple(dict.fromkeys((*FULL_V3_FEATURE_NAMES, *APEX_EXTRA_NAMES)))
DEFAULT_MACROS = "right_run,right_run_jump_short,right_run_jump_mid,right_run_jump_full"
ALL_MACROS = ",".join(m.value for m in Macro)
STUCK_RATE = 3e-4  # below this, no branch made progress inside the horizon


def state_key(ram) -> tuple[int, int, int, int]:
    """The emulator state as the planner sees it: Mario's x, y, x-speed and float state."""
    return (int(ram[0x006D]) * 256 + int(ram[0x0086]), int(ram[0x00CE]), int(ram[X_SPEED_ADDR]),
            _float_state(ram))


def keep_hook(_event):
    return ApexChoice.KEEP


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


class PlanTeacher:
    """The continuation policy: the teacher's rule, except in states that belong to a found
    plan, where it plays the plan's next macro (apex: keep, as the plan was searched)."""

    def __init__(self, teacher, ram) -> None:
        self.teacher, self.ram = teacher, ram
        self.plan_map: dict[tuple, Macro] = {}
        self.in_plan = False

    def _apex_rule(self, snapshot, extra, allowed):
        return self.teacher._apex_rule(snapshot, extra, allowed)

    def choose(self, snapshot, actions, extra=None) -> Decision:
        allowed = list(actions)
        if allowed and isinstance(allowed[0], ApexChoice):
            if self.in_plan:
                return Decision(action=ApexChoice.KEEP, confidence=1.0,
                                probabilities={a.value: float(a is ApexChoice.KEEP) for a in allowed},
                                latency_ms=0.0)
            return self.teacher.choose(snapshot, actions, extra=extra)
        macro = self.plan_map.get(state_key(self.ram))
        self.in_plan = macro is not None
        if macro is None:
            return self.teacher.choose(snapshot, actions, extra=extra)
        return Decision(action=macro, confidence=1.0,
                        probabilities={a.value: float(a == macro) for a in allowed}, latency_ms=0.0)


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
    ap.add_argument("--macros", default=DEFAULT_MACROS,
                    help=f"comma-separated macros to branch on, or 'all' for {ALL_MACROS}")
    ap.add_argument("--stride", type=int, default=0,
                    help="also branch at every N-th uninteresting grounded state (0 = only interesting)")
    ap.add_argument("--k", type=int, default=6, help="teacher continuation decisions per branch (per-decision return)")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--frames-horizon", type=int, default=128,
                    help="fixed frame horizon of `reward_rate` (the continuation runs until both --k "
                         "decisions and this many frames after the branch's last decision row)")
    ap.add_argument("--follow", choices=("teacher", "best"), default="teacher",
                    help="the main line: the teacher's own rule, or the branch with the best "
                         "frame-horizon return at every branch point (lookahead teacher)")
    ap.add_argument("--plan-depth", type=int, default=7,
                    help="follow=best: when no one-step branch makes progress, search macro sequences "
                         "up to this depth (0 = off)")
    ap.add_argument("--plan-goal", type=int, default=24, help="px of progress that ends the plan search")
    ap.add_argument("--plan-beam", type=int, default=400, help="max sequences kept per depth")
    ap.add_argument("--write-mainline", action="store_true",
                    help="also write the main line's decisions as rows (episode id main-<env>-s<seed>)")
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--features-version", type=int, choices=(1, 2), default=1,
                    help="1 = the frozen v1 superset (the #28/#32 domains); 2 = per-frame units, "
                         "no reliability (full_v2 / apex_v2 domains, findings #33)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    global NAMES
    if args.features_version == 2:
        NAMES = NAMES_V2

    macro_spec = ALL_MACROS if args.macros == "all" else args.macros
    macros = [Macro(m.strip()) for m in macro_spec.split(",") if m.strip()]
    horizon = args.frames_horizon
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0, rng_seed=args.seed)
    env = create_mario_env(args.env, render_mode="rgb_array")
    base = env.unwrapped
    parser = CadenceParser(decision_horizon_frames=8)
    _f, info = env.reset(seed=args.seed)
    ram = _unwrap_ram(env)
    snap = parser.parse(info, ram, previous_action=None)
    plan_teacher = PlanTeacher(teacher, ram)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sink = args.out.open("w")
    n_rows = n_points = n_branches = 0
    summary: dict[str, list] = {}
    chosen: list[tuple[int, str, str | None, str]] = []  # follow=best: (x, macro, apex, teacher's macro)
    plans: list[tuple[int, list[str], int]] = []  # (x, plan, end x) — or end x = -1 when none found
    level_tag = args.env.replace("SuperMarioBros-", "").replace("-v0", "")
    main_rows: list[dict] = []
    main_rewards: list[float] = []
    main_frames: list[int] = []
    pending_plan: deque[Macro] = deque()

    def run_decision(policy, decision, p, s):
        """Execute one decision with the apex look; return (rows-without-returns, next_snapshot)."""
        a_state: dict = {}
        ex = execute(env, decision.action, grounded=s.grounded, ram=ram,
                     apex=apex_hook(p, env, policy, decision, a_state))
        fired = ex.apex is not None and bool(a_state)
        nxt = p.parse(ex.info, ram,
                      previous_action=(a_state["decision"].action.value if fired else decision.action.value),
                      previous_reward=(ex.total_reward - ex.reward_to_apex if fired else ex.total_reward),
                      frames=(ex.frames - ex.frames_to_apex if fired else ex.frames))
        terminal = bool(ex.terminated or nxt.dead or nxt.clear)
        rows = build_rows(snapshot=s, decision=decision, execution=ex,
                          apex_state=a_state if fired else None, next_snapshot=nxt, terminal=terminal,
                          names=NAMES, reward_config=PROGRESS_FRACTION, episode_id="", first_step=0,
                          tag_kind=True)
        return rows, nxt, terminal

    def plan_escape(frozen_parser, s0):
        """BFS over macro sequences from the backed-up state until one advances --plan-goal px.
        Returns (macros, keys-before-each-step) or None. Replays prefixes from the single backup."""
        x0 = s0.x
        root_key = None
        frontier: list[tuple[tuple[Macro, ...], tuple]] = [((), None)]
        seen: set[tuple] = set()
        for depth in range(1, args.plan_depth + 1):
            nxt_frontier = []
            for prefix, _ in frontier:
                for m in macros:
                    base._restore(); base.done = False
                    p = copy.deepcopy(frozen_parser); s = s0; keys = []; dead = False; x = x0
                    if root_key is None:
                        root_key = state_key(ram); seen.add(root_key)
                    for step_m in (*prefix, m):
                        keys.append(state_key(ram))
                        ex = execute(env, step_m, grounded=s.grounded, ram=ram,
                                     apex=keep_hook if has_apex(step_m) else None)
                        # keep_hook never parses at the apex, so the interval is the whole macro
                        s = p.parse(ex.info, ram, previous_action=step_m.value, previous_reward=ex.total_reward,
                                    frames=ex.frames)
                        x = s.x
                        if ex.terminated or s.dead or s.clear:
                            dead = s.dead; break
                    if dead:
                        continue
                    key = state_key(ram)
                    if x >= x0 + args.plan_goal or s.clear:
                        return list((*prefix, m)), keys
                    if key in seen:
                        continue
                    seen.add(key)
                    nxt_frontier.append(((*prefix, m), (x, int(ram[0x00CE]))))
            nxt_frontier.sort(key=lambda t: (-t[1][0], t[1][1]))
            frontier = nxt_frontier[: args.plan_beam]
            print(f"    plan search depth {depth}: {len(frontier)} states", flush=True)
            if not frontier:
                break
        return None

    for d in range(args.max_decisions):
        if snap.dead or snap.clear:
            break
        own = plan_teacher.choose(snap, action_set("grounded"))
        nav = snap.navigation_features()
        branch_here = snap.grounded and (interesting(nav, snap.threat_features()) or (args.stride and d % args.stride == 0))
        main_policy, main_decision = plan_teacher, own
        if branch_here:
            n_points += 1
            base._backup()
            frozen_parser = copy.deepcopy(parser)
            x0 = snap.x
            def do_branches():
                nonlocal n_branches
                best_key, best_rate, buffered = None, -1.0, []
                for macro in macros:
                    for choice in (list(ApexChoice) if has_apex(macro) else [None]):
                        base._restore()
                        base.done = False
                        p = copy.deepcopy(frozen_parser)
                        explored = macro is not own.action
                        branch_policy = Forced(choice, explored, teacher) if choice is not None else plan_teacher
                        first_rows, nxt, terminal = run_decision(
                            branch_policy, Decision(action=macro, confidence=1.0, probabilities={},
                                                    latency_ms=0.0,
                                                    selection_status="explored" if explored else None),
                            p, snap)
                        rewards = [float(r["reward"]) for r in first_rows]
                        frames = [int(r["frames"]) for r in first_rows]
                        lead = sum(frames[:-1])  # frames before the branch's LAST decision row
                        # Continuation for the returns (teacher rule, or the plan where one applies):
                        # until both k decisions and the frame horizon after the last first row are covered.
                        s = nxt
                        cont = 0
                        while not terminal and (cont < args.k or sum(frames) - lead < horizon):
                            dec = plan_teacher.choose(s, action_set("grounded"))
                            rows, s, terminal = run_decision(plan_teacher, dec, p, s)
                            rewards.extend(float(r["reward"]) for r in rows)
                            frames.extend(int(r["frames"]) for r in rows)
                            cont += 1
                        ep_id = f"cov-{level_tag}-d{d:03d}-x{x0}-{macro.value}" + (f"-{choice.value}" if choice else "")
                        teacher_choice = branch_policy.teacher_choice if choice is not None else None
                        dead, cleared = bool(terminal and s.dead), bool(s.clear)
                        for i, r in enumerate(first_rows):
                            r["episode_id"] = ep_id
                            r["step"] = i
                            r["immediate_reward"] = r["reward"]
                            r["reward"] = kstep_return(rewards, i, args.k, args.gamma)
                            r["reward_rate"] = frame_horizon_return(rewards, frames, i, horizon, cleared=cleared)
                            r["x_takeoff"] = x0
                            r["branch_return_rewards"] = len(rewards)
                            r["branch_frames"] = sum(frames)
                            r["branch_end_x"] = s.x
                            r["branch_dead"] = dead
                            r["branch_cleared"] = cleared
                            r["branch_apex"] = choice.value if choice is not None else None
                            r["teacher_apex"] = teacher_choice
                            buffered.append(r)
                        n_branches += 1
                        key = (x0, macro.value, choice.value if choice else "-")
                        rate0 = first_rows[0]["reward_rate"]
                        summary[key] = [round(first_rows[0]["reward"], 4), rate0,
                                        first_rows[-1]["reward"] if len(first_rows) > 1 else None,
                                        "DEAD" if dead else ("CLEAR" if cleared else f"x={s.x}")]
                        is_own = (macro is own.action) and (choice is None or choice.value == teacher_choice)
                        if rate0 > best_rate + 1e-9 or (abs(rate0 - best_rate) <= 1e-9 and is_own):
                            best_rate, best_key = rate0, (macro, choice)
                return best_key, best_rate, buffered

            best_key, best_rate, buffered = do_branches()
            base._restore()
            base.done = False
            if args.follow == "best":
                if best_rate < STUCK_RATE and args.plan_depth > 0 and not pending_plan:
                    print(f"  stuck at x={x0} (best one-step rate {best_rate:.5f}); planning...", flush=True)
                    found = plan_escape(frozen_parser, snap)
                    base._restore(); base.done = False
                    if found is None:
                        plans.append((x0, [], -1))
                        print(f"  no plan within depth {args.plan_depth}", flush=True)
                    else:
                        plan, keys = found
                        for k_, m_ in zip(keys, plan):
                            plan_teacher.plan_map[k_] = m_
                        pending_plan = deque(plan)
                        plans.append((x0, [m.value for m in plan], 0))
                        print(f"  plan: {[m.value for m in plan]}", flush=True)
                        # Re-branch this state now that continuations can follow the plan, so the
                        # branch rows carry the plan's credit (not the pre-plan zeros).
                        n_branches -= len(buffered)
                        best_key, best_rate, buffered = do_branches()
                        base._restore(); base.done = False
                if pending_plan:
                    macro = pending_plan.popleft()
                    choice = ApexChoice.KEEP if has_apex(macro) else None
                    chosen.append((x0, macro.value, choice.value if choice else None, own.action.value))
                    plan_teacher.in_plan = True
                    main_policy = Forced(choice, macro is not own.action, teacher) if choice is not None else plan_teacher
                    main_decision = Decision(action=macro, confidence=1.0, probabilities={}, latency_ms=0.0,
                                             selection_status=None)
                elif best_key is not None:
                    macro, choice = best_key
                    chosen.append((x0, macro.value, choice.value if choice else None, own.action.value))
                    main_policy = Forced(choice, macro is not own.action, teacher) if choice is not None else plan_teacher
                    main_decision = Decision(action=macro, confidence=1.0, probabilities={}, latency_ms=0.0,
                                             selection_status=None)
            for r in buffered:
                sink.write(json.dumps(r, separators=(",", ":")) + "\n")
                n_rows += 1
        elif pending_plan:
            macro = pending_plan.popleft()
            choice = ApexChoice.KEEP if has_apex(macro) else None
            plan_teacher.in_plan = True
            main_policy = Forced(choice, False, teacher) if choice is not None else plan_teacher
            main_decision = Decision(action=macro, confidence=1.0, probabilities={}, latency_ms=0.0,
                                     selection_status=None)
        # Main line: the teacher's own decision, the best branch, or the plan's next step.
        x_here = snap.x
        in_plan_step = bool(plans) and plans[-1][2] == 0 and (main_decision.action.value in plans[-1][1])
        rows, snap, terminal = run_decision(main_policy, main_decision, parser, snap)
        for r in rows:
            r["x_takeoff"] = x_here
            r["branch_apex"] = None
            r["teacher_apex"] = None
            r["mainline"] = True
            r["plan_step"] = in_plan_step
        main_rows.extend(rows)
        main_rewards.extend(float(r["reward"]) for r in rows)
        main_frames.extend(int(r["frames"]) for r in rows)
        if plans and plans[-1][2] == 0 and not pending_plan:
            plans[-1] = (plans[-1][0], plans[-1][1], snap.x)
        if terminal:
            break
    if args.write_mainline:
        cleared = bool(snap.clear)
        ep_id = f"main-{level_tag}-s{args.seed}-{args.follow}"
        for i, r in enumerate(main_rows):
            r["episode_id"] = ep_id
            r["step"] = i
            r["immediate_reward"] = r["reward"]
            r["reward"] = kstep_return(main_rewards, i, args.k, args.gamma)
            r["reward_rate"] = frame_horizon_return(main_rewards, main_frames, i, horizon, cleared=cleared)
            sink.write(json.dumps(r, separators=(",", ":")) + "\n")
            n_rows += 1
    sink.close()
    write_provenance(args.out, kind="rollouts", extra={"rows": n_rows, "branch_points": n_points, "branches": n_branches, "features_version": args.features_version})
    env.close()
    print(f"branch points {n_points}, branches {n_branches}, rows {n_rows} -> {args.out}")
    print(f"main line ({args.follow}): {len(main_rows)} rows, ended x={snap.x} dead={snap.dead} clear={snap.clear}"
          + (" (written)" if args.write_mainline else ""))
    for x0, plan, end in plans:
        print(f"plan at x={x0}: {plan or 'NONE FOUND'} -> x={end}")
    if chosen:
        print("lookahead choices at branch points (x: macro/apex, teacher's own macro):")
        for x0, m, c, own_m in chosen:
            mark = "" if m == own_m else "  <- differs"
            print(f"   x={x0:5d}: {m}{'/' + c if c else ''}  (teacher: {own_m}){mark}")
    print("per branch: takeoff-row k-step return / frame-horizon rate / apex-row k-step return / where the branch ended")
    last_x = None
    for (x0, macro, choice), (r0, rate0, r1, end) in summary.items():
        if x0 != last_x:
            print(f"  x={x0}")
            last_x = x0
        print(f"     {macro:<22} {choice:<15} {r0:.4f}  {rate0:.4f}  {'-' if r1 is None else f'{r1:.4f}'}  {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
