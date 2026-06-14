"""
tests/test_session_design_sessions.py
--------------------------------------
session_state persistence for the Workbench DesignSession — and the BACKWARD-COMPAT
guard: an OLD session.json lacking `design_sessions` must restore clean (the
field can't break restoring a pre-Workbench session).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_state import SessionState


def test_old_json_without_design_sessions_loads_clean(tmp_path):
    p = tmp_path / "old_session.json"
    p.write_text(json.dumps({"session_start": "x", "working_dir": ".",
                             "structures": {"1": {"name": "1hsg"}}}))
    state, err = SessionState.try_load(str(p))
    assert err is None and state is not None
    assert state.design_sessions == {}           # backward-compat default
    assert state.structures == {"1": {"name": "1hsg"}}


def test_add_get_and_save_load_roundtrip(tmp_path):
    s = SessionState()
    payload = {"model_id": "1", "chains": {}, "next_id": 1}
    s.add_design_session("1", payload)
    assert s.get_design_session("1") == payload
    f = tmp_path / "s.json"
    s.save(str(f))
    s2 = SessionState.load(str(f))
    assert s2.get_design_session("1") == payload


def test_snapshot_restore_includes_design_sessions():
    s = SessionState()
    s.add_design_session("1", {"model_id": "1", "chains": {}, "next_id": 1})
    snap = s.snapshot()
    s.design_sessions.clear()
    s.restore(snap)
    assert s.get_design_session("1") is not None
