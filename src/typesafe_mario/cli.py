from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .policy import DiversifiedHeuristicPolicy, HeuristicPolicy, ReplayPolicy, TypeSafePolicy
from .runner import run_episode
from .state import MarioStateParser


def _build_policy(args: argparse.Namespace):
    if args.policy == "typesafe":
        return TypeSafePolicy()
    if args.policy == "heuristic":
        return HeuristicPolicy()
    if args.policy == "replay":
        if not args.replay_log:
            raise SystemExit("--policy replay needs --replay-log <run.jsonl>")
        return ReplayPolicy(args.replay_log)
    if args.policy == "adapt1":
        from .adapt1_client import Adapt1Client
        from .adapt1_policy import FEATURE_SETS, Adapt1Policy

        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.adapt1_trace or (args.artifacts_dir / f"adapt1-{args.domain_id}.jsonl")
        return Adapt1Policy(
            Adapt1Client(),
            args.domain_id,
            # --frozen = the docs' frozen evaluation: no feedback AND no exploration (sequential-learning
            # §16). Measured 2026-09-22: with exploration left on, a 'frozen' play of the apex arm hopped
            # everywhere and died at 717 while the true frozen eval cleared 1-1.
            allow_exploration=not (args.adapt1_no_explore or args.frozen),
            rng_seed=args.seed,
            run_id=args.run_id or args.domain_id,
            trace_path=trace_path,
            feature_names=FEATURE_SETS[args.features],
            sequential=args.learner == "sequential",
            epsilon=args.epsilon,
            temporal_context=args.tcp,
            learn=not args.frozen,
        )
    raise AssertionError(f"Unhandled policy: {args.policy}")


def _build_apex_policy(args: argparse.Namespace, policy: Any):
    """The in-air policy for `play` (design-notes §14a), or None."""
    if args.cadence != "grounded":
        if args.apex_domain or args.apex_teacher:
            raise SystemExit("--apex-domain / --apex-teacher need --cadence grounded")
        return None
    if args.policy == "replay":
        return policy if getattr(policy, "has_apex", False) else None  # replays its logged looks
    if args.apex_domain:
        from .adapt1_client import Adapt1Client
        from .adapt1_policy import FEATURE_SETS, Adapt1Policy

        return Adapt1Policy(
            Adapt1Client(),
            args.apex_domain,
            allow_exploration=not (args.adapt1_no_explore or args.frozen),
            rng_seed=args.seed,
            run_id=(args.run_id or args.domain_id or "apex") + "-apex",
            trace_path=args.artifacts_dir / f"adapt1-{args.apex_domain}.jsonl",
            feature_names=FEATURE_SETS["apex"],
            sequential=False,  # bandit with client-side credit; never a sequential block
            epsilon=args.epsilon,
            learn=not args.frozen,
        )
    if args.apex_teacher:
        return DiversifiedHeuristicPolicy(epsilon=0.0, rng_seed=args.seed)
    return None


def _reward_config(name: str):
    """Reward preset by name (None = raw gym reward)."""
    from .reward import DEFAULT, GROUNDED, PROGRESS_FRACTION, PROGRESS_RATE

    return {
        "shaped": DEFAULT,
        "grounded": GROUNDED,
        "progress": PROGRESS_FRACTION,
        "progress-rate": PROGRESS_RATE,
        "raw": None,
    }[name]


def create_domain(args: argparse.Namespace) -> int:
    """Create (or recreate) the Adapt-1 domain for a given learner + feature arm.

    Live call — needs REI_KEY. --recreate deletes an existing domain first so the config
    (feature set, sequential block) is applied cleanly.
    """
    from .adapt1_client import Adapt1Client, Adapt1Error
    from .adapt1_policy import FEATURE_SETS, build_domain_config

    client = Adapt1Client()
    sequential = args.learner == "sequential"
    config = build_domain_config(
        args.domain_id,
        feature_names=FEATURE_SETS[args.features],
        sequential=sequential,
    )
    if args.recreate:
        try:
            client.call("DELETE", f"/domains/{args.domain_id}")
            print(f"deleted existing domain {args.domain_id}")
        except Adapt1Error as exc:
            print(f"(no delete: {exc.status})")
    status, _ = client.create_domain(config)
    print(
        f"created domain {args.domain_id!r}  HTTP {status}  "
        f"learner={args.learner}  features={args.features} "
        f"({len(FEATURE_SETS[args.features])} paths)"
    )
    return 0


def _demo_ram() -> bytearray:
    ram = bytearray(0x0800)
    # Mario position and one goomba ahead.
    ram[0x006D] = 0
    ram[0x0086] = 172
    ram[0x000F] = 1
    ram[0x0016] = 0x06
    ram[0x006E] = 0
    ram[0x0087] = 214
    ram[0x00CF] = 79
    # Fill the tile-map row immediately below Mario with solid ground.
    ground_row = (79 + 16 - 32) // 16
    for column in range(16):
        ram[0x0500 + ground_row * 16 + column] = 1
    return ram


def state_demo() -> int:
    info = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 172,
        "y_pos": 79,
        "y_pixel": 79,
        "left_x_pos": 60,
        "progress": 172,
        "progress_max": 172,
        "status": "small",
        "player_state": 8,
        "life": 2,
        "coins": 0,
        "score": 200,
        "time": 387,
        "death": False,
        "clear": False,
    }
    snapshot = MarioStateParser().parse(info, _demo_ram(), previous_action="right")
    print(json.dumps(snapshot.to_state(), indent=2))
    print("\n--- TEXT VIEW ---\n")
    print(snapshot.to_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="typesafe-mario")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state-demo", help="Show structured and text state without API calls")

    play = subparsers.add_parser("play", help="Run a Mario episode")
    play.add_argument("--env", default="SuperMarioBros-1-1-v0")
    play.add_argument(
        "--frames-per-decision",
        type=int,
        default=8,
        help="Minimum macro duration in emulator frames",
    )
    play.add_argument("--max-decisions", type=int, default=2000)
    play.add_argument(
        "--stall-timeout",
        type=int,
        default=None,
        help="End the episode after this many decisions with no new max-x (default: off)",
    )
    play.add_argument("--seed", type=int, default=123)
    play.add_argument(
        "--policy", choices=("typesafe", "heuristic", "adapt1", "replay"), default="typesafe"
    )
    play.add_argument(
        "--replay-log",
        type=Path,
        default=None,
        help="with --policy replay: a run-*.jsonl to replay with NO queries (deterministic "
        "emulator); cadence is inferred from the log, the Adapt-1 panel redraws from the "
        "logged telemetry when present",
    )
    play.add_argument(
        "--domain-id",
        default="mario-1-1",
        help="Adapt-1 domain id (for --policy adapt1)",
    )
    play.add_argument(
        "--run-id",
        default=None,
        help="Stable run id recorded in Adapt-1 feedback metadata",
    )
    play.add_argument(
        "--adapt1-trace",
        type=Path,
        default=None,
        help="Where to write the Adapt-1 decision/feedback trace (default: artifacts/)",
    )
    play.add_argument(
        "--adapt1-no-explore",
        action="store_true",
        help="Disable exploration (allow_exploration=false) — for frozen evaluation runs",
    )
    play.add_argument(
        "--learner",
        choices=("bandit", "sequential"),
        default="bandit",
        help="Adapt-1 learner: bandit (direct feedback) or sequential (delayed credit). "
        "Must match how the domain was created.",
    )
    play.add_argument(
        "--features",
        choices=("full", "perception"),
        default="full",
        help="Feature arm: full (Jev-comparable, incl. parser verdicts) or perception "
        "(verdicts stripped). Must match how the domain was created.",
    )
    play.add_argument(
        "--display",
        choices=("dashboard", "game", "none"),
        default="dashboard",
        help="Combined telemetry dashboard, plain game window, or headless mode",
    )
    play.add_argument(
        "--cadence",
        choices=("frame", "grounded"),
        default="frame",
        help="frame = 7 actions × 8 frames; grounded = committed macros (findings #21)",
    )
    play.add_argument(
        "--reward",
        choices=("shaped", "grounded", "progress", "progress-rate", "raw"),
        default="progress",
        help="reward preset for online feedback (#22: progress)",
    )
    play.add_argument(
        "--frozen",
        action="store_true",
        help="Adapt-1 only: query but never send feedback (Queries only, 0 Records)",
    )
    play.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Adapt-1 only: client-side ε-greedy during learning (#22)",
    )
    play.add_argument(
        "--tcp", action="store_true", help="Adapt-1 only: the domain has learning.temporal_context"
    )
    play.add_argument(
        "--apex-domain",
        default=None,
        help="grounded cadence (design-notes §14a): a second Adapt-1 domain (features `apex`, "
        "hypotheses apex_keep/brake/pull_back) queried at the top of every jump; learns "
        "from the landing unless --frozen",
    )
    play.add_argument(
        "--apex-teacher",
        action="store_true",
        help="grounded cadence: use the heuristic teacher's apex rule for the in-air look "
        "(free; the demo-time behaviour, for watching or as a control)",
    )
    play.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    play.add_argument(
        "--screenshot",
        type=Path,
        help="Save the first populated dashboard frame as a PNG",
    )

    make = subparsers.add_parser(
        "create-domain", help="Create/recreate an Adapt-1 domain (live; needs REI_KEY)"
    )
    make.add_argument("--domain-id", required=True)
    make.add_argument("--learner", choices=("bandit", "sequential"), default="bandit")
    make.add_argument("--features", choices=("full", "perception"), default="full")
    make.add_argument(
        "--recreate",
        action="store_true",
        help="Delete an existing domain first so the new config applies cleanly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "state-demo":
        return state_demo()
    if args.command == "create-domain":
        return create_domain(args)
    if args.command == "play":
        policy = _build_policy(args)
        if args.policy == "replay":
            args.cadence = ReplayPolicy.infer_cadence(policy.rows)
            args.max_decisions = min(args.max_decisions, len(policy.rows))
            args.stall_timeout = None
        apex_policy = _build_apex_policy(args, policy)
        log_path = run_episode(
            env_id=args.env,
            policy=policy,
            apex_policy=apex_policy,
            frames_per_decision=args.frames_per_decision,
            max_decisions=args.max_decisions,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            display=args.display,
            screenshot_path=args.screenshot,
            stall_timeout=args.stall_timeout,
            cadence=args.cadence,
            reward_config=_reward_config(args.reward),
        )
        print(f"Run log: {log_path.resolve()}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
