"""Provenance for every generated dataset and every live ingest (2026-09-25).

Why: findings #37 — one of the three data files behind the 1-1 clear (#28) was written from an
uncommitted working state and cannot be regenerated from the repo. A replication repo needs, for
every row that was ever fed to a domain: the exact code (commit + whether the tree was dirty, and
the diff if it was), the exact command, the environment, and a hash of the file that was fed.

``write_provenance(path, ...)`` writes ``<path>.provenance.json`` next to a dataset (or a run
log). ``check_clean()`` is what a live ingest calls: it refuses a dirty tree unless told not to.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = ("gym-super-mario-bros", "nes-py", "gymnasium", "numpy", "pillow")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_state() -> dict[str, Any]:
    """Commit, dirty flag and a hash of the uncommitted diff (the diff itself is saved when dirty)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") or ""
    diff = _git("diff", "HEAD") or "" if status else ""
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_files": [line[3:] for line in status.splitlines() if line.strip()],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "diff": diff if diff else None,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict[str, Any]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def write_provenance(
    target: Path,
    *,
    kind: str,
    inputs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<target>.provenance.json``: what produced ``target`` (a dataset, a run log, a domain).

    ``inputs`` maps a label to a file path; each gets its sha256 (and its own provenance file name
    if one exists), so a chain dataset → returns → curation → ingest is walkable backwards."""
    target = Path(target)
    script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    record: dict[str, Any] = {
        "kind": kind,
        "target": str(target),
        "target_sha256": file_sha256(target) if target.exists() and target.is_file() else None,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]] if sys.argv else None,
        "script_sha256": file_sha256(script) if script and script.exists() else None,
        "cwd": os.getcwd(),
        "git": git_state(),
        "environment": environment(),
        "inputs": {},
        "extra": extra or {},
    }
    for label, p in (inputs or {}).items():
        p = Path(p)
        prov = p.with_name(p.name + ".provenance.json")
        record["inputs"][label] = {
            "path": str(p),
            "sha256": file_sha256(p) if p.exists() else None,
            "provenance": str(prov) if prov.exists() else None,
        }
    out = target.with_name(target.name + ".provenance.json")
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return out


def check_clean(*, allow_dirty: bool = False, what: str = "this run") -> dict[str, Any]:
    """Refuse to spend Records from a dirty tree unless explicitly allowed (the state is still
    recorded either way — the diff goes into the provenance file)."""
    state = git_state()
    if state["dirty"] and not allow_dirty:
        raise SystemExit(
            f"{what}: the working tree has uncommitted changes in {state['dirty_files']} — commit first "
            f"so the run is reproducible from git, or pass --allow-dirty (the diff is then recorded)."
        )
    return state
