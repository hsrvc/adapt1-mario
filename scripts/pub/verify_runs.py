"""Verify every replication bundle under mario/runs/ against its own hash index (offline, 0 Records).

    python scripts/pub/verify_runs.py [../runs]          # all bundles
    python scripts/pub/verify_runs.py ../runs/2026-10-01-gate-CLEAR-1-1

An `index.json` entry names a file and its hash in one of three shapes, all of which occur in the bundles:
  {"file": "x.jsonl", "sha256_uncompressed": ..}          -> x.jsonl.gz next to the index (or x.jsonl itself)
  {"file": "x.jsonl", "gz": "logs/x.jsonl.gz", "sha256_uncompressed": ..}
  {"file": "replays/x.gif", "sha256": ..}                  -> the raw file
Exit status 1 if anything is missing or mismatched; bundles without an index are listed as UNINDEXED (a warning,
not a failure — add one before the public cut).

    python scripts/pub/verify_runs.py ../runs --index   # add sha256 to unhashed entries and index unindexed bundles
                                                        # (raw files only; notes are left empty for you to fill)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_bundle(bundle: Path) -> tuple[list[str], list[str]]:
    """Return (ok, problems) for one bundle directory holding an index.json."""
    ok: list[str] = []
    problems: list[str] = []
    entries = json.loads((bundle / "index.json").read_text())
    for e in entries:
        name = e.get("file")
        if not name:
            problems.append(f"entry without 'file': {e}")
            continue
        if "sha256_uncompressed" in e:
            want = e["sha256_uncompressed"]
            gz = bundle / e["gz"] if e.get("gz") else bundle / (name + ".gz")
            raw = bundle / name
            if gz.exists():
                got = _sha(gzip.open(gz).read())
            elif raw.exists():
                got = _sha(raw.read_bytes())
            else:
                problems.append(f"MISSING {name} (neither {gz.name} nor the raw file)")
                continue
        elif "sha256" in e:
            want = e["sha256"]
            raw = bundle / name
            if not raw.exists():
                problems.append(f"MISSING {name}")
                continue
            got = _sha(raw.read_bytes())
        else:
            problems.append(f"{name}: no hash field")
            continue
        if got == want:
            ok.append(name)
        else:
            problems.append(f"MISMATCH {name}: index {want[:12]}… file {got[:12]}…")
    return ok, problems


def index_bundle(bundle: Path) -> int:
    """Add a sha256 to every unhashed entry and an entry for every file the index does not list.
    Returns the number of entries written. Never touches an entry that already carries a hash."""
    idx_path = bundle / "index.json"
    entries = json.loads(idx_path.read_text()) if idx_path.exists() else []
    listed = {e.get("file") for e in entries} | {e.get("gz") for e in entries if e.get("gz")}
    changed = 0
    for e in entries:
        if "sha256" not in e and "sha256_uncompressed" not in e and (bundle / e["file"]).exists():
            e["sha256"] = _sha((bundle / e["file"]).read_bytes())
            changed += 1
    for f in sorted(p for p in bundle.rglob("*") if p.is_file()):
        rel = str(f.relative_to(bundle))
        if (
            rel in ("index.json", "README.md")
            or rel in listed
            or rel.endswith(".gz")
            or "/__pycache__/" in rel
        ):
            continue
        if rel + ".gz" in listed or any(e.get("file") == rel for e in entries):
            continue
        entries.append(
            {"file": rel, "sha256": _sha(f.read_bytes()), "bytes": f.stat().st_size, "note": ""}
        )
        changed += 1
    if changed:
        idx_path.write_text(json.dumps(entries, indent=1) + "\n")
    return changed


def main(argv: list[str]) -> int:
    write = "--index" in argv
    argv = [a for a in argv if a != "--index"]
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[3] / "runs"
    # a bundle is a directory with an index.json, at any depth (runs/zero-start/stage-1/); a top-level
    # directory with no index anywhere beneath it is listed as UNINDEXED
    bundles = (
        [root]
        if (root / "index.json").exists()
        else sorted(
            {i.parent for i in root.rglob("index.json")}
            | {p for p in root.iterdir() if p.is_dir() and not any(p.rglob("index.json"))}
        )
    )
    failed = 0
    for b in bundles:
        if write:
            n = index_bundle(b)
            if n:
                print(f"INDEXED    {b.name}: {n} entr{'y' if n == 1 else 'ies'} written")
        if not (b / "index.json").exists():
            print(f"UNINDEXED  {b.relative_to(root)}")
            continue
        ok, problems = verify_bundle(b)
        status = "OK        " if not problems else "FAIL      "
        print(
            f"{status} {b.relative_to(root) if b != root else b.name}: {len(ok)} verified"
            + (f", {len(problems)} problem(s)" if problems else "")
        )
        for p in problems:
            print(f"    {p}")
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
