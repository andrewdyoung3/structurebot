"""
tests/test_foldseek_bridge.py
-----------------------------
foldseek_bridge.FoldseekBridge — the LOCAL-ONLY structural-neighbour search (Stage 2 template
auto-discovery). WSL mocked: parse/rank/dedup, the exact easy-search command surface, the
availability capability flag (fail-loud), the DB-scope label, and the eval delegation.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from foldseek_bridge import FoldseekBridge, foldseek_available


M8 = "\n".join([
    "1ABC_A\t0.950\t0.94\t0.94\t1e-9",
    "2XYZ_B\t0.400\t0.40\t0.40\t1e-3",
    "1ABC_A\t0.800\t0.80\t0.80\t1e-5",   # duplicate (1ABC,A) → dropped (keeps the first/higher)
    "3GHI_A\t0.200\t0.20\t0.20\t1e-1",   # below min_tm 0.3 → dropped
    "4JKL_C\t0.700\t0.70\t0.70\t1e-6",
    "notenoughcolumns",                  # <2 fields → skipped
])


def _bridge_with_mock_wsl():
    fb = FoldseekBridge()
    fb._wsl = MagicMock()
    return fb


class TestParse:
    def test_rank_filter_dedup(self):
        hits = FoldseekBridge._parse_hits(M8, min_tm=0.3, max_results=30)
        assert hits == [("1ABC", "A", 0.95), ("4JKL", "C", 0.7), ("2XYZ", "B", 0.4)]

    def test_max_results_caps(self):
        hits = FoldseekBridge._parse_hits(M8, min_tm=0.3, max_results=2)
        assert hits == [("1ABC", "A", 0.95), ("4JKL", "C", 0.7)]

    def test_min_tm_floor(self):
        # raise the floor → only the top survives
        hits = FoldseekBridge._parse_hits(M8, min_tm=0.5, max_results=30)
        assert [h[0] for h in hits] == ["1ABC", "4JKL"]

    def test_empty(self):
        assert FoldseekBridge._parse_hits("", 0.3, 30) == []


class TestCommand:
    def test_search_command_surface(self):
        fb = _bridge_with_mock_wsl()
        cmd = fb._search_command("/mnt/c/q.cif", "/tmp/o.m8", "/tmp/t")
        assert "easy-search" in cmd
        assert "/mnt/c/q.cif" in cmd
        assert fb._db in cmd                              # local DB, not a remote API
        assert "--alignment-type 1" in cmd
        assert "--format-output" in cmd and "alntmscore" in cmd
        assert "cat /tmp/o.m8" in cmd                     # results read back from the local out file
        # LOCAL-ONLY: no URL / remote-search flag anywhere in the command
        assert "http" not in cmd and "--remote" not in cmd


class TestAvailability:
    def test_available_true(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.is_available.return_value = True
        fb._wsl.run_command.return_value = {"ok": True, "stdout": "OK\n"}
        assert fb.is_available() is True

    def test_unavailable_no_marker(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.is_available.return_value = True
        fb._wsl.run_command.return_value = {"ok": True, "stdout": ""}   # binary/DB check failed
        assert fb.is_available() is False

    def test_unavailable_wsl_down(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.is_available.return_value = False
        assert fb.is_available() is False

    def test_module_flag_monkeypatched(self, monkeypatch):
        monkeypatch.setattr(FoldseekBridge, "is_available", lambda self: True)
        assert foldseek_available() is True
        monkeypatch.setattr(FoldseekBridge, "is_available",
                            lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        assert foldseek_available() is False              # never raises — fail-closed to False


class TestDbLabel:
    def test_parses_pdb_date(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.run_command.return_value = {
            "ok": True, "stdout": "abc  pdb100.tar.gz\n250101\tPDB_DATE\nsha1  FOLDSEEK_COMMIT"}
        lbl = fb.db_label()
        assert "2025-01" in lbl and "local" in lbl.lower()
        # cached: a second call doesn't re-shell
        fb._wsl.run_command.reset_mock()
        assert fb.db_label() == lbl
        fb._wsl.run_command.assert_not_called()

    def test_label_fallback(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.run_command.return_value = {"ok": False}
        assert "PDB" in fb.db_label() or "local" in fb.db_label().lower()


class TestSearch:
    def test_search_parses_stdout(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.translate_path.side_effect = lambda p: "/mnt" + str(p).replace("\\", "/")
        fb._wsl.run_command.return_value = {"ok": True, "stdout": M8}
        qf = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        qf.write(b"x"); qf.close()
        try:
            hits = fb.search_neighbors(qf.name, max_results=30, min_tm=0.3)
            assert hits[0] == ("1ABC", "A", 0.95)
            assert ("3GHI", "A", 0.2) not in hits         # below floor
        finally:
            os.unlink(qf.name)

    def test_missing_query_returns_empty(self):
        fb = _bridge_with_mock_wsl()
        assert fb.search_neighbors("/no/such/file.cif") == []

    def test_search_failure_returns_empty(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.translate_path.side_effect = lambda p: p
        fb._wsl.run_command.return_value = {"ok": False}
        qf = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        qf.write(b"x"); qf.close()
        try:
            assert fb.search_neighbors(qf.name) == []      # searched-but-failed → empty (caller checks is_available)
        finally:
            os.unlink(qf.name)


class TestTwoBucket:
    """The opt-in two-bucket return (the 'show lower-confidence hits' expander source). The DEFAULT
    (with_low_bucket=False) MUST stay byte-for-byte the old single-list contract — the eval and every
    existing caller depend on it (single source of truth)."""

    def _bridge(self):
        fb = _bridge_with_mock_wsl()
        fb._wsl.translate_path.side_effect = lambda p: "/mnt" + str(p).replace("\\", "/")
        fb._wsl.run_command.return_value = {"ok": True, "stdout": M8}
        return fb

    def _query(self):
        qf = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        qf.write(b"x"); qf.close()
        return qf.name

    def test_default_is_unchanged_primary_list(self):
        # Backward-compat PIN: the eval-shaped (primary-only) call returns exactly what it did before
        # — a plain list, TM≥0.3, NOT a tuple. (3GHI at 0.20 is dropped, NOT surfaced.)
        fb = self._bridge(); q = self._query()
        try:
            out = fb.search_neighbors(q, max_results=30, min_tm=0.3)   # eval-shaped call
            assert isinstance(out, list)
            assert out == [("1ABC", "A", 0.95), ("4JKL", "C", 0.7), ("2XYZ", "B", 0.4)]
            assert all(h[2] >= 0.3 for h in out) and ("3GHI", "A", 0.2) not in out
        finally:
            os.unlink(q)

    def test_opt_in_returns_primary_and_low_band(self):
        # with_low_bucket=True → (primary≥0.3, low in [low_bound, 0.3)). 3GHI (0.20) lands in low;
        # 2XYZ (0.40) stays primary. ONE search (run_command called once).
        fb = self._bridge(); q = self._query()
        try:
            primary, low = fb.search_neighbors(q, min_tm=0.3, with_low_bucket=True, low_bound=0.2)
            assert primary == [("1ABC", "A", 0.95), ("4JKL", "C", 0.7), ("2XYZ", "B", 0.4)]
            assert low == [("3GHI", "A", 0.2)]                         # [0.20, 0.30) band only
            assert all(0.2 <= h[2] < 0.3 for h in low)
            assert fb._wsl.run_command.call_count == 1                 # NO second foldseek search
        finally:
            os.unlink(q)

    def test_low_bound_excludes_below_floor(self):
        # raise low_bound to 0.25 → 3GHI (0.20) falls below it → low bucket empty.
        fb = self._bridge(); q = self._query()
        try:
            primary, low = fb.search_neighbors(q, min_tm=0.3, with_low_bucket=True, low_bound=0.25)
            assert low == []
        finally:
            os.unlink(q)

    def test_band_not_truncated_by_high_tm_hits(self):
        # REGRESSION (live-caught): when there are MORE than low_max_results hits above the floor,
        # the [low_bound, min_tm) band must NOT be crowded out — the band filter precedes the cap.
        # 20 hits at TM 0.90 + one genuine band hit at 0.22; low_max_results=15.
        rows = [f"1A{i:02d}_A\t0.900\t0.9\t0.9\t1e-9" for i in range(20)]
        rows.append("9LOW_A\t0.220\t0.22\t0.22\t1e-1")
        fb = _bridge_with_mock_wsl()
        fb._wsl.translate_path.side_effect = lambda p: p
        fb._wsl.run_command.return_value = {"ok": True, "stdout": "\n".join(rows)}
        q = self._query()
        try:
            primary, low = fb.search_neighbors(q, min_tm=0.3, with_low_bucket=True,
                                               low_bound=0.2, low_max_results=15)
            assert len(primary) == 30 or len(primary) == 20    # all high hits (cap 30)
            assert low == [("9LOW", "A", 0.22)]                # band hit SURVIVES the high-TM crowd
        finally:
            os.unlink(q)

    def test_parse_hits_band_filter_direct(self):
        # the max_tm band bound on the pure parser: [0.3, 0.8) excludes 0.95 and 0.20.
        band = FoldseekBridge._parse_hits(M8, min_tm=0.3, max_results=30, max_tm=0.8)
        assert band == [("4JKL", "C", 0.7), ("2XYZ", "B", 0.4)]   # 0.95 and 0.20 both excluded

    def test_opt_in_empty_on_missing_query(self):
        fb = _bridge_with_mock_wsl()
        assert fb.search_neighbors("/no/such.cif", with_low_bucket=True) == ([], [])


def test_eval_entrypoint_delegates(monkeypatch):
    """The eval's foldseek_neighbors is now a thin wrapper over the shipped bridge (single source)."""
    import importlib
    mod = importlib.import_module("scripts.eval_template_guided_calibration")
    called = {}
    def fake_search(self, query_path, max_results=60, min_tm=0.3):
        called["args"] = (query_path, max_results, min_tm)
        return [("9ZZZ", "A", 0.88)]
    monkeypatch.setattr(FoldseekBridge, "search_neighbors", fake_search)
    out = mod.foldseek_neighbors("/tmp/q.cif", max_results=42, min_tm=0.25)
    assert out == [("9ZZZ", "A", 0.88)]
    assert called["args"] == ("/tmp/q.cif", 42, 0.25)
