#!/usr/bin/env python
"""Scan a tree for anything the public repo must never contain. Exit 1 on any hit.

    python scripts/pub/check_public.py <dir> [--extra-pattern REGEX ...]

Forbidden: account names, personal e-mail, Keychain item names, Rei-draft files, the KB files, and any 64-hex
string that is not the value of a sha256 field (bundle indexes and provenance sidecars carry sha256 legitimately).
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from pathlib import Path

# Generic patterns only: this file ships. Personal names and addresses live in a private patterns file that never
# ships (one regex per line; default: PUB_PRIVATE_PATTERNS, else mario/pub-private-patterns.txt in the source repo).
PRIVATE_PATTERNS = Path(__file__).resolve().parents[3] / "pub-private-patterns.txt"
FORBIDDEN_TEXT = [
    r"REI_KEY_[A-Z]",
    r"find-generic-password",
    r"reilabs-dm-",
    r"Telegram",
    r"CLAUDE\.md",
    r"findings\.md",
    r'"session_id": ?"[0-9a-f]{16,}:',  # the proxy's per-account session (redacted by build_public_tree.py)
]
HEX64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
# Machina responses hash each proposal into "fingerprint" (server-side, not a secret); allowed per match, not per line.
FINGERPRINT = re.compile(r'"fingerprint": ?"$')
SHA_CONTEXT = re.compile(r"sha256|provenance|diff_sha256|script_sha256|target_sha256", re.I)
FORBIDDEN_NAMES = [
    r"^reilabs-dm-",
    r"^CLAUDE\.md$",
    r"^findings\.md$",
    r"^HANDOFF",
    r"^SESSION-",
    r"^\.claude",
]
SKIP_DIRS = {".git", ".venv", "__pycache__", "artifacts"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".jsonl",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".html",
}


def scan(root: Path, extra: list[str]) -> list[str]:
    hits: list[str] = []
    text_patterns = [re.compile(p, re.I) for p in FORBIDDEN_TEXT + extra]
    name_patterns = [re.compile(p) for p in FORBIDDEN_NAMES]
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in ("check_public.py", "test_pub_scanner.py"):
            continue  # the scanner and its test carry the patterns by necessity
        if any(p.search(path.name) for p in name_patterns):
            hits.append(f"{path}: forbidden file name")
        if not path.is_file():
            continue
        try:
            if path.suffix == ".gz" and Path(path.stem).suffix.lower() in TEXT_SUFFIXES:
                text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
            elif path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
            else:
                continue
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for p in text_patterns:
                if p.search(line):
                    hits.append(f"{path}:{n}: {p.pattern!r}")
            if not SHA_CONTEXT.search(line) and any(
                not FINGERPRINT.search(line[max(0, m.start() - 20) : m.start()])
                for m in HEX64.finditer(line)
            ):
                hits.append(f"{path}:{n}: 64-hex outside a sha256 field")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--extra-pattern", action="append", default=[])
    args = ap.parse_args()
    private = Path(os.environ.get("PUB_PRIVATE_PATTERNS", PRIVATE_PATTERNS))
    extra = list(args.extra_pattern)
    if private.is_file():
        extra += [
            line.strip()
            for line in private.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    hits = scan(args.root, extra)
    for h in hits:
        print(h)
    print(f"{len(hits)} hit(s) in {args.root}")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
