"""provenance (2026-09-25): every dataset and live ingest records the code, command, environment and
input hashes it came from — findings #37: a data file written from an uncommitted tree could not be rebuilt."""

import json

from typesafe_mario.provenance import check_clean, file_sha256, git_state, write_provenance


def test_write_provenance_records_git_env_and_inputs(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text('{"a":1}\n')
    out = tmp_path / "out.jsonl"
    out.write_text('{"a":2}\n')
    prov = write_provenance(out, kind="returns", inputs={"dataset": src}, extra={"k": 6})
    rec = json.loads(prov.read_text())
    assert prov.name == "out.jsonl.provenance.json"
    assert rec["kind"] == "returns" and rec["extra"] == {"k": 6}
    assert rec["target_sha256"] == file_sha256(out)
    assert rec["inputs"]["dataset"]["sha256"] == file_sha256(src)
    assert rec["git"]["commit"] and isinstance(rec["git"]["dirty"], bool)
    assert rec["environment"]["python"] and "gym-super-mario-bros" in rec["environment"]["packages"]
    assert rec["command"] and rec["written_at"]


def test_dirty_diff_is_recorded_when_present():
    state = git_state()
    if state["dirty"]:
        assert state["diff"] and state["diff_sha256"]
    else:
        assert state["diff"] is None and state["dirty_files"] == []


def test_check_clean_matches_git_state():
    state = git_state()
    if state["dirty"]:
        import pytest

        with pytest.raises(SystemExit, match="uncommitted"):
            check_clean(what="test")
        assert check_clean(allow_dirty=True)["dirty"] is True
    else:
        assert check_clean()["dirty"] is False
