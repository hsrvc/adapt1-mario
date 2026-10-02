"""pub (2026-09-26): the public-tree scanner flags what must never ship and accepts sha256 fields."""

import subprocess
import sys
from pathlib import Path

SCANNER = Path(__file__).resolve().parents[1] / "scripts" / "pub" / "check_public.py"


def _run(root: Path, *extra: str) -> tuple[int, str]:
    args = [a for e in extra for a in ("--extra-pattern", e)]
    p = subprocess.run(
        [sys.executable, str(SCANNER), str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        env={"PUB_PRIVATE_PATTERNS": "/nonexistent", "PATH": ""},
    )
    return p.returncode, p.stdout


def test_clean_tree_passes(tmp_path):
    (tmp_path / "index.json").write_text('{"sha256_uncompressed": "' + "a" * 64 + '"}\n')
    (tmp_path / "notes.md").write_text(
        "The apex domain cleared 1-1 on hsc.\n"
    )  # 'hsc' alone is fine
    code, out = _run(tmp_path)
    assert code == 0 and out.strip().endswith("0 hit(s) in " + str(tmp_path))


def test_forbidden_strings_and_names_are_caught(tmp_path):
    (tmp_path / "a.md").write_text("domains on private-account-name\n")
    (tmp_path / "b.py").write_text('key = "' + "b" * 64 + '"\n')
    (tmp_path / ("reilabs-" + "dm-x.md")).write_text("hi\n")
    (tmp_path / "c.txt").write_text("security find-generic-" + "password -s REI_KEY_" + "HSC\n")
    code, out = _run(tmp_path, "private-account-name")
    assert code == 1
    assert (
        "a.md:1" in out
        and "64-hex outside" in out
        and "forbidden file name" in out
        and "c.txt:1" in out
    )


def test_verify_runs_checks_gz_and_raw_hashes(tmp_path):
    import gzip
    import hashlib
    import json
    import subprocess
    import sys

    b = tmp_path / "2026-01-01-demo"
    (b / "logs").mkdir(parents=True)
    raw = b"row1\nrow2\n"
    with gzip.open(b / "logs" / "a.jsonl.gz", "wb") as fh:
        fh.write(raw)
    (b / "clip.gif").write_bytes(b"GIF89a")
    json.dump(
        [
            {
                "file": "a.jsonl",
                "gz": "logs/a.jsonl.gz",
                "sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
            },
            {"file": "clip.gif", "sha256": hashlib.sha256(b"GIF89a").hexdigest()},
        ],
        (b / "index.json").open("w"),
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "pub" / "verify_runs.py"
    r = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0 and "2 verified" in r.stdout, r.stdout
    (b / "clip.gif").write_bytes(b"GIF89b")  # tamper
    r = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, check=False
    )
    assert r.returncode == 1 and "MISMATCH clip.gif" in r.stdout, r.stdout
