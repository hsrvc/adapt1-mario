#!/usr/bin/env python
"""Build the public tree from the allowlist: copy, then scan; non-zero exit on any hit.

    python scripts/pub/build_public_tree.py --manifest ../pub-manifest.txt --out /path/to/adapt1-mario

The output directory must not exist (a fresh copy every time; never edit the public tree by hand).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # scripts/pub -> scripts -> agent -> mario -> repo
# The proxy's per-account session (Machina journals carry it in every request). Not the key, but it identifies the account.
SESSION_ID = re.compile(rb'"session_id": ?"[^"]*"')


def redact_sessions(out: Path) -> int:
    """Blank every session_id in the copied run logs and re-hash those entries of each bundle's index.json.
    The private bundles keep the originals; the public index notes the redaction."""
    changed = 0
    for gz in sorted(out.glob("runs/**/*.gz")):
        raw = gzip.decompress(gz.read_bytes())
        red, n = SESSION_ID.subn(b'"session_id":"redacted"', raw)
        if not n:
            continue
        gz.write_bytes(gzip.compress(red, mtime=0))
        bundle = next(p for p in gz.parents if (p / "index.json").exists())
        index_path = bundle / "index.json"
        entries = json.loads(index_path.read_text())
        rel = gz.relative_to(bundle).as_posix()
        for e in entries:
            if e.get("gz") == rel or e.get("file") + ".gz" == rel:
                e["sha256_uncompressed"] = hashlib.sha256(red).hexdigest()
                e["bytes"] = len(red)
                e["note"] = (e.get("note") or "") + " [session_id values redacted for publication]"
        index_path.write_text(json.dumps(entries, indent=1) + "\n")
        changed += 1
    return changed


def strip_diffs(out: Path) -> int:
    """A run started from a dirty tree records the whole uncommitted diff in its provenance. The public copy keeps
    dirty / dirty_files / diff_sha256 and drops the diff text (it can carry private, unrelated work)."""
    n = 0
    for prov in sorted(out.glob("runs/**/*.provenance.json")):
        rec = json.loads(prov.read_text())
        git = rec.get("git") or {}
        if not git.get("diff"):
            continue
        git["diff"] = "[omitted in the public copy; diff_sha256 identifies it]"
        prov.write_text(json.dumps(rec, indent=1) + "\n")
        bundle = next(q for q in prov.parents if (q / "index.json").exists())
        idx = json.loads((bundle / "index.json").read_text())
        rel = prov.relative_to(bundle).as_posix()
        for e in idx:
            if e.get("file") == rel and "sha256" in e:
                e["sha256"] = hashlib.sha256(prov.read_bytes()).hexdigest()
        (bundle / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "mario" / "pub-manifest.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-scan", action="store_true")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists — the public tree is always a fresh copy")
    # manifest lines: `path` (copied, flattened), `path => dest` (copied to dest in the public tree),
    # `!dest` (removed from the public tree after copying, e.g. a sub-folder of a shipped bundle)
    lines = [
        l.strip()
        for l in args.manifest.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    excludes = [l[1:].strip() for l in lines if l.startswith("!")]
    entries = [l for l in lines if not l.startswith("!")]
    srcs = [e.split("=>")[0].strip() for e in entries]
    copied = 0
    for entry in entries:
        rel, _, dest = (x.strip() for x in entry.partition("=>"))
        # a path inside another listed directory is already copied with it (unless it is renamed)
        if not dest and any(rel != o and Path(rel).is_relative_to(o) for o in srcs):
            continue
        src = REPO_ROOT / rel
        if not src.exists():
            print(f"  missing: {rel}", file=sys.stderr)
            continue
        # flatten: mario/agent/<x> -> <x>; mario/pub-src/<x> -> <x> (the public root files); mario/<x> -> <x>
        parts = Path(rel).parts
        if dest:
            dst_rel = Path(dest)
        elif parts[:2] in (("mario", "agent"), ("mario", "pub-src")):
            dst_rel = Path(*parts[2:])
        elif parts[0] == "mario":
            dst_rel = Path(*parts[1:])
        else:
            dst_rel = Path(rel)
        dst = args.out / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            # dirs_exist_ok: pub-src/ overlays public READMEs onto copied bundles
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
        else:
            shutil.copy2(src, dst)
        copied += 1
    for ex in excludes:
        target = args.out / ex
        # drop the excluded path's entries from the enclosing bundle's hash index
        bundle = next((q for q in target.parents if (q / "index.json").exists()), None)
        if bundle is not None and bundle.is_relative_to(args.out):
            sub = target.relative_to(bundle).as_posix()
            idx = json.loads((bundle / "index.json").read_text())
            kept = [
                e
                for e in idx
                if not (e.get("file") == sub or e.get("file", "").startswith(sub + "/"))
            ]
            (bundle / "index.json").write_text(json.dumps(kept, indent=1) + "\n")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        else:
            print(f"  exclude matched nothing: {ex}", file=sys.stderr)
    print(f"copied {copied} of {len(entries)} manifest entries into {args.out}")
    print(f"redacted session_id in {redact_sessions(args.out)} run logs")
    print(f"dropped the uncommitted diff from {strip_diffs(args.out)} provenance file(s)")
    # the learning-curve figure is redrawn from the bundles that ship, so it never shows data the tree lacks
    plot = args.out / "scripts" / "plot_zero_curve.py"
    if plot.exists() and (args.out / "runs").is_dir():
        out_fig = args.out / "runs" / "zero-start" / "learning-curve"
        subprocess.check_call([sys.executable, str(plot), "-o", str(out_fig)], cwd=args.out)
    if args.no_scan:
        return 0
    return subprocess.call(
        [sys.executable, str(Path(__file__).with_name("check_public.py")), str(args.out)]
    )


if __name__ == "__main__":
    raise SystemExit(main())
