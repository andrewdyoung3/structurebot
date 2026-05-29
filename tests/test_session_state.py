"""
tests/test_session_state.py
---------------------------
Tests for session persistence and startup auto-restore:

  - SessionState.try_load / load robustness (corrupt JSON, missing file)
  - Restoring computed state (mutation scan results) across a save/load cycle
  - StructureBot 'clear session' command wiping session.json + in-memory state
  - Restore-time ChimeraX presence check offering to re-open a missing structure

All ChimeraX / network interactions are mocked; no real PDB fetches.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import main
from main import StructureBot
from session_state import SessionState


# ── Console suppression (Rich emits Unicode that cp1252 consoles reject) ──────

@pytest.fixture(autouse=True)
def _suppress_console(monkeypatch):
    monkeypatch.setattr("main.console", MagicMock())


def _make_mock_bot() -> StructureBot:
    """A StructureBot with heavyweight deps mocked, __init__ bypassed."""
    bot = object.__new__(StructureBot)
    bot.bridge             = MagicMock()
    bot.translator         = MagicMock()
    bot.session            = MagicMock()
    bot.router             = MagicMock()
    bot.auto_proceed       = False
    bot.auto_proceed_delay = 3
    bot.log_file           = Path("test_session.jsonl")
    bot._resume_flag       = False
    bot._interactive       = True
    return bot


# ══════════════════════════════════════════════════════════════════════════════
# SessionState load / restore
# ══════════════════════════════════════════════════════════════════════════════

def test_session_restore_loads_scan_results(tmp_path):
    """A saved session with scan results round-trips through save -> load."""
    path = tmp_path / "session.json"

    s = SessionState()
    # Pass metadata explicitly so add_structure does NOT hit the RCSB network.
    s.add_structure("1", "1HSG", metadata={"pdb_id": "1HSG", "chains": ["A", "B"]})
    s.add_scan_result("1", [
        {"position": 72, "from_aa": "I", "to_aa": "R", "ddg": -0.003,
         "ddg_source": "pyrosetta", "combined_score": 0.55},
    ])
    s.save(str(path))

    # try_load reports success and the scan data is intact
    state, err = SessionState.try_load(str(path))
    assert err is None
    assert state is not None
    data = state.get_scan_result("1")
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["position"] == 72
    assert data[0]["to_aa"] == "R"
    assert data[0]["ddg_source"] == "pyrosetta"
    assert "1" in state.structures

    # load() returns the same populated state
    loaded = SessionState.load(str(path))
    assert loaded.get_scan_result("1")[0]["ddg"] == -0.003


def test_session_restore_handles_corrupt_json(tmp_path):
    """Corrupt session.json is reported by try_load and never crashes load()."""
    path = tmp_path / "session.json"
    path.write_text("{ this is not valid json :: ", encoding="utf-8")

    state, err = SessionState.try_load(str(path))
    assert state is None
    assert err is not None and "corrupt" in err.lower()

    # load() must degrade gracefully to a fresh, empty session
    fresh = SessionState.load(str(path))
    assert isinstance(fresh, SessionState)
    assert fresh.scan_results == {}
    assert fresh.structures == {}


def test_try_load_missing_file(tmp_path):
    """A missing file is 'nothing to restore', not an error."""
    state, err = SessionState.try_load(str(tmp_path / "does_not_exist.json"))
    assert state is None
    assert err is None


def test_try_load_non_dict_json(tmp_path):
    """A JSON file that isn't an object is treated as incompatible."""
    path = tmp_path / "session.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    state, err = SessionState.try_load(str(path))
    assert state is None
    assert err is not None


# ══════════════════════════════════════════════════════════════════════════════
# StructureBot — clear session
# ══════════════════════════════════════════════════════════════════════════════

def test_clear_session_wipes_state(tmp_path, monkeypatch):
    """'clear session' deletes session.json and resets in-memory state."""
    sess_file = tmp_path / "session.json"
    monkeypatch.setattr(main, "SESSION_FILE", str(sess_file))

    # Seed a populated session file on disk.
    seed = SessionState()
    seed.add_structure("1", "1HSG", metadata={"pdb_id": "1HSG"})
    seed.add_scan_result("1", [{"position": 72, "to_aa": "R"}])
    seed.save(str(sess_file))
    assert sess_file.is_file()

    bot = _make_mock_bot()
    bot._cmd_clear_session()

    assert not sess_file.is_file(), "session.json should be deleted"
    assert isinstance(bot.session, SessionState), "in-memory state should be fresh"
    assert bot.session.scan_results == {}
    assert bot.session.structures == {}


# ══════════════════════════════════════════════════════════════════════════════
# StructureBot — restore-time ChimeraX presence check
# ══════════════════════════════════════════════════════════════════════════════

def test_restore_when_structure_not_in_chimerax(monkeypatch):
    """
    When a restored structure is no longer loaded in ChimeraX, the reconnect
    step offers to re-open it (a fast fetch), and issues 'open <name>' on accept.
    """
    bot = _make_mock_bot()

    sess = SessionState()
    sess.add_structure("1", "1HSG", metadata={"pdb_id": "1HSG"})  # reopenable PDB ID
    bot.session = sess

    calls: list[str] = []

    def _run_cmd(cmd, *a, **k):
        calls.append(cmd)
        if "info models" in cmd:
            return {"value": "no models currently open", "error": None}  # #1 absent
        return {"value": "", "error": None}

    bot.bridge.is_running.return_value = True
    bot.bridge.run_command.side_effect = _run_cmd

    # Accept the re-open prompt.
    monkeypatch.setattr("main.Prompt.ask", lambda *a, **k: "y")

    bot._reconnect_or_offer_reopen()

    assert any("info models" in c for c in calls), "should query open models"
    assert any(c.lower().startswith("open") and "1hsg" in c.lower() for c in calls), (
        f"should offer to re-open the missing structure; got calls={calls}"
    )


def test_restore_when_structure_present_no_reopen(monkeypatch):
    """When the structure IS still loaded in ChimeraX, no 'open' is issued."""
    bot = _make_mock_bot()

    sess = SessionState()
    sess.add_structure("1", "1HSG", metadata={"pdb_id": "1HSG"})
    bot.session = sess

    calls: list[str] = []

    def _run_cmd(cmd, *a, **k):
        calls.append(cmd)
        if "info models" in cmd:
            return {"value": "#1 1HSG, 1234 atoms", "error": None}  # #1 present
        return {"value": "", "error": None}

    bot.bridge.is_running.return_value = True
    bot.bridge.run_command.side_effect = _run_cmd

    # If asked, decline — but it should not be asked at all.
    monkeypatch.setattr("main.Prompt.ask", lambda *a, **k: "n")

    bot._reconnect_or_offer_reopen()

    assert not any(c.lower().startswith("open") for c in calls), (
        f"present structure must not trigger a re-open; got calls={calls}"
    )
