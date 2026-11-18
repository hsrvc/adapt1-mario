#!/usr/bin/env python
"""A takeoff + apex pair, frozen, on a level (queries only): the check after a rebuild feed.

Waits until both learners have an installed model, records each learner's status (model type, version,
validation skill, sample counts), then plays frozen games (`exploit`, no exploration, no feedback) on the given
seeds with `online_duo.frozen_eval`, the evaluator of every zero-start checkpoint. Seeds default to 777, 1, 2, 3, 4. Writes <out>/check.json + one GIF and trace per seed.

    python scripts/rebuild_check.py --account mine --takeoff-domain mario-zero2r-takeoff \\
        --apex-domain mario-zero2r-apex --out artifacts/base-check
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typesafe_mario.adapt1_client import Adapt1Client  # noqa: E402
from typesafe_mario.adapt1_policy import FEATURE_SETS  # noqa: E402
from typesafe_mario.online_duo import MACROS_28, frozen_eval, status_query, wait_installed  # noqa: E402
from typesafe_mario.provenance import git_state  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--account", required=True, help="a label for the account, written to the log")
    ap.add_argument("--takeoff-domain", required=True)
    ap.add_argument("--apex-domain", required=True)
    ap.add_argument("--env", default="SuperMarioBros-1-1-v0")
    ap.add_argument("--seeds", default="777,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    client = Adapt1Client()
    args.out.mkdir(parents=True, exist_ok=True)
    doms = (args.takeoff_domain, args.apex_domain)
    version = client.version()
    before = wait_installed(client, doms, timeout_s=1800)
    seeds = [int(s) for s in args.seeds.split(",")]
    res = frozen_eval(client, takeoff_domain=args.takeoff_domain, apex_domain=args.apex_domain,
                      takeoff_features=FEATURE_SETS["full_v3"], apex_features=FEATURE_SETS["apex_v3"],
                      macros=MACROS_28, env_id=args.env, seeds=seeds, out_dir=args.out,
                      tag=args.env.split("-")[1] + "-" + args.env.split("-")[2])
    after = {d: status_query(client, d) for d in doms}
    out = {"kind": "rebuild-check", "account": args.account, "at": datetime.now(UTC).isoformat(),
           "git": git_state()["commit"], "server_version": version, "env": args.env, "seeds": seeds,
           "learner_before": before, "learner_after": after, "results": res,
           "flags": sum(1 for r in res if r.get("reached_flag")),
           "reach": [int(r["max_x"]) for r in res]}
    (args.out / "check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("env", "seeds", "flags", "reach")}))
    for d, st in after.items():
        print(d, {k: st.get(k) for k in ("model_type", "installed", "model_status", "version",
                                          "validation_skill", "sample_count", "live_sample_count")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
