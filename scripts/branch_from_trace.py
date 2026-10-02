#!/usr/bin/env python
"""Branch rollouts from a frozen-eval TRACE: the policy's own states, not the teacher's (findings #38).

The coverage generator (apex_coverage.py) branches at hazard states of the teacher's mainline, which sees enemies
early. A jump-chaining policy lands next to enemies it never saw coming — the gate run first-sighted goombas at 42 and
50 px and died; the feed had one such row. This script replays a trace (`frozen_eval.py` writes one per seed: takeoff
rows with macro, apex rows with the choice), saves the emulator at every grounded decision of that run, and from each
tries every macro (jump macros × every apex choice, the teacher's own apex rule recorded as `teacher_apex`), then lets
the teacher continue for the returns — the same row shape and return math as apex_coverage.py, so `ingest_demos.py`
feeds the result unchanged (`--kind takeoff --takeoff-under teacher`, `--kind apex`). Rows carry the v3 superset
(v2 arms project by name). Offline, 0 Records.

    PYTHONPATH=src python scripts/branch_from_trace.py --trace artifacts/frozen-eval/gate2/1-1-seed777.trace.jsonl \\
        --macros noop,right,right_run,left,right_run_jump_short,right_run_jump_mid,right_run_jump_full,right_jump_full \\
        --out artifacts/demos/gate2-branches-1-1.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario.adapt1_policy import APEX_EXTRA_NAMES, FULL_V3_FEATURE_NAMES  # noqa: E402
from typesafe_mario.cadence import CadenceParser  # noqa: E402
from typesafe_mario.demos import apex_hook, build_rows  # noqa: E402
from typesafe_mario.macros import ApexChoice, Macro, execute, has_apex  # noqa: E402
from typesafe_mario.policy import Decision, DiversifiedHeuristicPolicy  # noqa: E402
from typesafe_mario.provenance import write_provenance  # noqa: E402
from typesafe_mario.returns import frame_horizon_return, kstep_return  # noqa: E402
from typesafe_mario.reward import PROGRESS_FRACTION  # noqa: E402
from typesafe_mario.runner import _unwrap_ram, create_mario_env  # noqa: E402

NAMES = tuple(dict.fromkeys((*FULL_V3_FEATURE_NAMES, *APEX_EXTRA_NAMES)))


class Forced:
    """One fixed apex choice for the branch; records what the teacher's rule would have chosen."""

    def __init__(self, action, explored: bool, teacher) -> None:
        self.action, self.explored, self.teacher = action, explored, teacher
        self.teacher_choice = None

    def choose(self, snapshot, actions, extra=None) -> Decision:
        self.teacher_choice = self.teacher._apex_rule(snapshot, extra or {}, list(actions)).value
        return Decision(action=self.action, confidence=1.0,
                        probabilities={a.value: float(a == self.action) for a in actions},
                        latency_ms=0.0, selection_status="explored" if self.explored else None)


def read_trace(path: Path) -> list[tuple[Macro, ApexChoice | None]]:
    """(macro, apex choice) per takeoff decision; an apex record precedes its takeoff record."""
    seq, pending = [], None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if t["kind"] == "apex":
            pending = ApexChoice(t["choice"])
        else:
            seq.append((Macro(t["macro"]), pending))
            pending = None
    return seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--seed", type=int, default=777, help="the seed the trace was played with")
    ap.add_argument("--macros", required=True, help="comma-separated macros to branch on")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--frames-horizon", type=int, default=128)
    ap.add_argument("--tag", default=None, help="episode-id tag (default: the trace file's stem)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    macros = [Macro(m.strip()) for m in args.macros.split(",") if m.strip()]
    seq = read_trace(args.trace)
    tag = args.tag or args.trace.stem.replace(".trace", "")
    teacher = DiversifiedHeuristicPolicy(epsilon=0.0, rng_seed=args.seed)
    env = create_mario_env(args.env, render_mode="rgb_array")
    base = env.unwrapped
    parser = CadenceParser(decision_horizon_frames=8)
    _f, info = env.reset(seed=args.seed)
    ram = _unwrap_ram(env)
    snap = parser.parse(info, ram, previous_action=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sink = args.out.open("w")
    n_rows = n_points = n_branches = 0

    def run_decision(policy, decision, p, s):
        a_state: dict = {}
        ex = execute(env, decision.action, grounded=s.grounded, ram=ram, apex=apex_hook(p, env, policy, decision, a_state))
        fired = ex.apex is not None and bool(a_state)
        nxt = p.parse(ex.info, ram,
                      previous_action=(a_state["decision"].action.value if fired else decision.action.value),
                      previous_reward=(ex.total_reward - ex.reward_to_apex if fired else ex.total_reward),
                      frames=(ex.frames - ex.frames_to_apex if fired else ex.frames))
        terminal = bool(ex.terminated or nxt.dead or nxt.clear)
        rows = build_rows(snapshot=s, decision=decision, execution=ex, apex_state=a_state if fired else None,
                          next_snapshot=nxt, terminal=terminal, names=NAMES, reward_config=PROGRESS_FRACTION,
                          episode_id="", first_step=0, tag_kind=True)
        return rows, nxt, terminal

    try:
        for d, (played, played_apex) in enumerate(seq):
            if snap.dead or snap.clear:
                break
            if snap.grounded:
                n_points += 1
                base._backup()
                frozen = copy.deepcopy(parser)
                x0 = snap.x
                for macro in macros:
                    for choice in (list(ApexChoice) if has_apex(macro) else [None]):
                        base._restore(); base.done = False
                        p = copy.deepcopy(frozen)
                        explored = macro is not played
                        policy = Forced(choice, explored, teacher) if choice is not None else teacher
                        first_rows, nxt, terminal = run_decision(
                            policy, Decision(action=macro, confidence=1.0, probabilities={}, latency_ms=0.0,
                                             selection_status="explored" if explored else None), p, snap)
                        rewards = [float(r["reward"]) for r in first_rows]
                        frames = [int(r["frames"]) for r in first_rows]
                        lead = sum(frames[:-1])
                        s, cont = nxt, 0
                        while not terminal and (cont < args.k or sum(frames) - lead < args.frames_horizon):
                            dec = teacher.choose(s, tuple(Macro(m) for m in [mm.value for mm in macros]))
                            rows, s, terminal = run_decision(teacher, dec, p, s)
                            rewards.extend(float(r["reward"]) for r in rows)
                            frames.extend(int(r["frames"]) for r in rows)
                            cont += 1
                        ep_id = f"tr-{tag}-d{d:03d}-x{x0}-{macro.value}" + (f"-{choice.value}" if choice else "")
                        dead, cleared = bool(terminal and s.dead), bool(s.clear)
                        for i, r in enumerate(first_rows):
                            r.update(episode_id=ep_id, step=i, immediate_reward=r["reward"],
                                     reward=kstep_return(rewards, i, args.k, args.gamma),
                                     reward_rate=frame_horizon_return(rewards, frames, i, args.frames_horizon, cleared=cleared),
                                     x_takeoff=x0, branch_return_rewards=len(rewards), branch_frames=sum(frames),
                                     branch_end_x=s.x, branch_dead=dead, branch_cleared=cleared,
                                     # a forced apex choice that never fired (the jump died before its apex)
                                     # did not shape the row: record it as unforced so the teacher-rule filter keeps it
                                     branch_apex=(choice.value if (choice is not None and len(first_rows) > 1) else None),
                                     teacher_apex=(policy.teacher_choice if (choice is not None and len(first_rows) > 1) else None),
                                     played_macro=played.value)
                            sink.write(json.dumps(r, separators=(",", ":")) + "\n"); n_rows += 1
                        n_branches += 1
                base._restore(); base.done = False
                parser = copy.deepcopy(frozen)
            # advance along the trace: the policy's own macro and apex choice
            hook = (lambda _e, _a=played_apex: _a) if played_apex is not None else None
            ex = execute(env, played, grounded=snap.grounded, ram=ram, apex=hook)
            snap = parser.parse(ex.info, ram, previous_action=played.value,
                                frames=(ex.frames - ex.frames_to_apex if (ex.apex is not None and played_apex is not None) else ex.frames))
            if ex.terminated:
                break
    finally:
        sink.close(); env.close()
    print(f"branched {n_points} states of the trace × {len(macros)} macros -> {n_branches} branches, {n_rows} rows -> {args.out}")
    write_provenance(args.out, kind="trace-branches", inputs={"trace": args.trace},
                     extra={"branch_points": n_points, "branches": n_branches, "rows": n_rows, "macros": [m.value for m in macros]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
