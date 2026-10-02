"""Turn a zero-start run log (zero_start_duo.py JSONL) into the `curve.json` the learning-curve page reads.

    python scripts/zero_curve.py <run-log.jsonl> [-o curve.json]

Offline, 0 Records. Shape (the page's contract, mario/artifacts-src/zero-learning-curve.build.py):
  {"online": [{"episode", "reach", "records", "decisions", "looks"}, ...],
   "frozen": [{"tag", "episode", "reach", "records", "takeoff_skill", "takeoff_model", "takeoff_installed",
               "consumed", "sent"}, ...]}
`takeoff_installed` is False when the checkpoint was played while a retrain was running and no model was
installed: the page marks such a checkpoint as unreliable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_curve(run_log: Path) -> dict:
    rows = [json.loads(line) for line in run_log.read_text().splitlines() if line.strip()]
    online = [
        {
            "episode": r["episode"],
            "reach": r["max_x"],
            "records": r["records_spent"],
            "decisions": r["decisions"],
            "looks": r["apex_looks"],
        }
        for r in rows
        if r["kind"] == "episode"
    ]
    frozen = []
    for r in rows:
        if r["kind"] != "eval":
            continue
        t = r.get("takeoff_learner") or {}
        frozen.append(
            {
                "tag": r["tag"],
                "episode": r["episode"],
                "reach": r["reach"],
                "records": r["records_spent"],
                "takeoff_skill": t.get("validation_skill"),
                "takeoff_model": t.get("model_type") if t.get("installed") else None,
                "takeoff_installed": bool(t.get("installed")),
                "consumed": t.get("live_sample_count"),
                "sent": r.get("sent_by_domain"),
            }
        )
    return {"online": online, "frozen": frozen}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_log", type=Path)
    ap.add_argument(
        "-o", "--out", type=Path, default=None, help="default: curve.json beside the run log"
    )
    args = ap.parse_args()
    curve = build_curve(args.run_log)
    out = args.out or args.run_log.with_name("curve.json")
    out.write_text(json.dumps(curve, indent=1) + "\n")
    print(
        f"wrote {out}: {len(curve['online'])} episodes, frozen "
        + " -> ".join(
            f"{f['reach']}{'' if f['takeoff_installed'] else '(no model)'}" for f in curve["frozen"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
