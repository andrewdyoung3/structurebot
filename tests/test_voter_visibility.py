"""
tests/test_voter_visibility.py
------------------------------
B2 — a silently-dropped EXPECTED voter must be VISIBLE. Three states, tested:
  - deliberately disabled            → SILENT (no note; don't nag a config choice)
  - capability-absent                → QUIET note (axis count stays visible)
  - available-then-empty {}          → LOUD, with the carried reason (the fix)
  - scores present                   → "ok" (normal; no user-facing noise)

The discriminator between quiet and loud is the now-reliable is_available()
capability flag (Unit B). Normal runs add nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation_scanner import MutationScanner
from tool_router import ToolRouter, ToolStepResult


def _scanner():
    s = MutationScanner(session=MagicMock(), model_id="1",
                        progress_callback=lambda *_: None)
    s.voter_notes = []
    return s


def _fake_bridge(status="available (x)", available=True, scores=None, reason=None):
    b = MagicMock()
    b.status.return_value = status
    b.is_available.return_value = available
    b.score_mutations.return_value = (scores or {}, {})
    b.last_skip_reason = reason
    return b


def _cands():
    return [{"position": 10, "resnum": 10, "from_aa": "A", "to_aa": "V"}]


# ── scanner classification (the 3-state LOGIC) ──────────────────────────────────

class TestScannerVoterNotes:

    def _run(self, tmp_path, bridge):
        pdb = tmp_path / "x.pdb"; pdb.write_text("ATOM\n")
        s = _scanner()
        with patch("thermompnn_bridge.ThermoMPNNBridge", return_value=bridge):
            s._run_thermompnn(str(pdb), "A", _cands(), enabled=True)
        return s.voter_notes

    def test_disabled_is_silent(self, tmp_path):
        notes = self._run(tmp_path, _fake_bridge(
            status="disabled (THERMOMPNN_ENABLE=false)", available=False))
        assert notes == []                                   # SILENT

    def test_capability_absent_is_quiet(self, tmp_path):
        notes = self._run(tmp_path, _fake_bridge(
            status="not available (THERMOMPNN_DIR=...)", available=False))
        assert len(notes) == 1
        assert notes[0]["voter"] == "ThermoMPNN"
        assert notes[0]["state"] == "unavailable"            # QUIET

    def test_available_but_empty_is_loud_with_reason(self, tmp_path):
        notes = self._run(tmp_path, _fake_bridge(
            available=True, scores={}, reason="no mappable chain residues — skipped"))
        loud = [n for n in notes if n["state"] == "empty"]
        assert len(loud) == 1
        assert loud[0]["reason"] == "no mappable chain residues — skipped"  # CARRIED

    def test_scores_present_is_ok(self, tmp_path):
        notes = self._run(tmp_path, _fake_bridge(available=True, scores={"A:A10V": -1.0}))
        assert any(n["state"] == "ok" for n in notes)

    def test_not_requested_is_silent(self, tmp_path):
        pdb = tmp_path / "x.pdb"; pdb.write_text("ATOM\n")
        s = _scanner()
        s._run_thermompnn(str(pdb), "A", _cands(), enabled=False)
        assert s.voter_notes == []                           # SILENT (not requested)


# ── router header strings + placement (loud prominent, quiet line, silent gone) ──

class TestRouterVoterHeader:

    def _router(self):
        sess = MagicMock()
        sess.get_protected_residues_for_chain.return_value = []
        r = ToolRouter(bridge=MagicMock(), session=sess)
        r._first_model_id = MagicMock(return_value="1")
        r._fetch_sequence = MagicMock(return_value="ACDEFGHIK")
        r._ensure_pdb_file = MagicMock(return_value="x.pdb")
        return r

    def _run_with_notes(self, notes):
        r = self._router()
        fake = MagicMock()
        fake.scan.return_value = [{
            "from_aa": "A", "position": 10, "to_aa": "V", "ddg": None,
            "combined_score": 1.0, "solubility_delta": 0.5, "ddg_source": "?",
        }]
        fake.voter_notes = notes
        fake.generate_chimerax_commands.return_value = ([], [])
        fake._generate_summary.return_value = "TOP: A10V"
        with patch("mutation_scanner.MutationScanner", return_value=fake):
            return r._run_mutation_scan({"model_id": "1", "chain": "A",
                                         "sequence": "ACDEFGHIK", "pdb_path": "x.pdb"})

    def test_loud_is_prominent_and_carries_reason(self):
        res = self._run_with_notes([
            {"voter": "ThermoMPNN", "state": "empty", "reason": "residue-mapping divergence"}])
        # loud line is at the TOP of the summary (before the TOP: result line)
        first = res.summary.splitlines()[0]
        assert first.startswith("⚠"), first
        assert "ThermoMPNN" in first and "residue-mapping divergence" in first
        assert res.summary.index("⚠") < res.summary.index("TOP:")

    def test_quiet_one_liner_keeps_axis_visible(self):
        res = self._run_with_notes([
            {"voter": "RaSP", "state": "unavailable", "reason": "not available"}])
        assert "RaSP not available this run" in res.summary
        assert "⚠" not in res.summary                        # not loud

    def test_disabled_silent_no_header(self):
        # an "ok" voter + nothing dropped → no header at all (normal run unchanged)
        res = self._run_with_notes([{"voter": "ThermoMPNN", "state": "ok", "reason": None}])
        assert "⚠" not in res.summary
        assert "not available this run" not in res.summary
        assert res.summary.startswith("TOP:")

    def test_voter_notes_in_data(self):
        notes = [{"voter": "ThermoMPNN", "state": "empty", "reason": "x"}]
        res = self._run_with_notes(notes)
        assert res.data["voter_notes"] == notes
