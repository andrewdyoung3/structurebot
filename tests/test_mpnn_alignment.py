"""
tests/test_mpnn_alignment.py
----------------------------
Tests for ProteinMPNN design persistence + retrieval (no re-run) + the WT-vs-
redesign alignment (console + ChimeraX Sequence Viewer command construction).
Mocks all subprocess / REST — no MPNN run, no live ChimeraX.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import proteinmpnn_bridge as pmb
from tool_router import ToolRouter
from session_state import SessionState


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def cache_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROTEINMPNN_CACHE_DIR", tmp_path / "pmcache")
    (tmp_path / "pmcache").mkdir()
    return tmp_path / "pmcache"


_WT  = "SAKELRCQCIKTYSKPFHPKF"            # 21 aa
_TOP = "SCKELRCQCAKTYSKPFHPKW"            # changes at 2 (A->C? actually S A->C), 10, 20
_DESIGNS = [
    {"sequence": _TOP, "score": -1.10, "recovery": 0.857,
     "mutations": ["A2C", "I10A", "F21W"]},
    {"sequence": "SAKELRCQCIKTYSKPFHPKW", "score": -0.90, "recovery": 0.952,
     "mutations": ["F21W"]},
]


# ── Persistence + retrieval (no subprocess) ─────────────────────────────────────

def test_write_read_designs_fasta_roundtrip(cache_tmp):
    fa = pmb.write_designs_fasta("1", _WT, _DESIGNS, "proteinmpnn")
    assert fa.is_file() and fa.parent == cache_tmp
    back = pmb.read_designs_fasta(fa)
    assert back["wildtype_sequence"] == _WT
    assert len(back["sequences"]) == 2
    # FULL sequence preserved (not truncated)
    assert back["sequences"][0]["sequence"] in (_TOP, "SAKELRCQCIKTYSKPFHPKW")
    # mutations recomputed vs WT, consistent
    top = next(s for s in back["sequences"] if s["sequence"] == _TOP)
    assert top["mutations"] == ["A2C", "I10A", "F21W"]


def test_latest_cached_fasta_picks_most_recent(cache_tmp):
    import time
    pmb.write_designs_fasta("1", _WT, _DESIGNS[:1], "proteinmpnn")
    time.sleep(1.05)
    f2 = pmb.write_designs_fasta("1", _WT, _DESIGNS, "proteinmpnn")
    assert pmb.latest_cached_fasta("1") == f2
    assert pmb.latest_cached_fasta() == f2          # model-agnostic too


def test_retrieval_reads_cache_without_subprocess(cache_tmp, monkeypatch):
    pmb.write_designs_fasta("1", _WT, _DESIGNS, "proteinmpnn")
    # Make ANY subprocess invocation fail loudly — retrieval must not spawn one.
    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("MPNN re-ran!")))
    r = ToolRouter(bridge=None, session=SessionState())   # empty session → cache fallback
    data, src = r._resolve_mpnn_data("1")
    assert data is not None and src.startswith("cache:")
    assert data["wildtype_sequence"] == _WT
    assert len(data["sequences"]) == 2


def test_run_writes_persistent_cache_fasta(cache_tmp, monkeypatch):
    """A run persists the FASTA even though the temp dir is deleted."""
    bridge = pmb.ProteinMPNNBridge.__new__(pmb.ProteinMPNNBridge)
    bridge._backend = "proteinmpnn"
    # Bypass the real subprocess: _run_proteinmpnn returns parsed result_data.
    monkeypatch.setattr(bridge, "_run_proteinmpnn",
                        lambda *a, **k: {"sequences": _DESIGNS, "wildtype_sequence": _WT,
                                         "fixed_positions": [], "backend": "proteinmpnn"})
    res = bridge._run_inference("x.pdb", "A", [], 2, 0.1, session=None, model_id="1")
    assert res.success
    assert res.data.get("fasta_path")
    assert Path(res.data["fasta_path"]).is_file()
    assert res.data["chain"] == "A"
    # the cache now holds it
    assert pmb.latest_cached_fasta("1") is not None


# ── Routing: display phrasings retrieve, never run ──────────────────────────────

def test_display_intent_classification():
    r = ToolRouter(bridge=None, session=SessionState())
    assert r._detect_mpnn_display_intent("output the full redesigned sequence for chain A") == "sequence"
    assert r._detect_mpnn_display_intent("output the amino acid sequence for the protein redesign") == "sequence"
    assert r._detect_mpnn_display_intent("show the alignment") == "alignment"
    assert r._detect_mpnn_display_intent("what changed in the design") == "alignment"
    # RUN / RE-RUN phrasings must NOT be treated as display
    assert r._detect_mpnn_display_intent("redesign chain A with proteinMPNN") is None
    assert r._detect_mpnn_display_intent("redesign the dimer interface to make it hydrophilic") is None
    assert r._detect_mpnn_display_intent("run proteinmpnn on chain B") is None


def test_singular_display_phrase_retrieves_from_cache(cache_tmp, monkeypatch):
    pmb.write_designs_fasta("1", _WT, _DESIGNS, "proteinmpnn")
    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("MPNN re-ran!")))
    r = ToolRouter(bridge=None, session=SessionState())
    out = r.handle_sequence_display_command("output the full redesigned sequence for chain A")
    assert out is not None and isinstance(out, str)
    assert _TOP in out                              # the full design sequence is shown


def test_run_phrase_is_not_intercepted():
    r = ToolRouter(bridge=None, session=SessionState())
    assert r.handle_sequence_display_command("redesign chain A with proteinMPNN") is None


# ── Console alignment ───────────────────────────────────────────────────────────

def test_console_alignment_marks_changed_positions():
    r = ToolRouter(bridge=None, session=SessionState())
    out = r._build_mpnn_alignment_console(_WT, _TOP, _DESIGNS[0], "1", "A", "session")
    # changed residues bold-red, with a ^ marker per change, full change list
    assert "[bold red]C[/bold red]" in out          # pos 2 S->C
    assert out.count("^") == 3                       # 3 changed columns
    assert "Changes (3):" in out and "A2C" in out and "I10A" in out and "F21W" in out
    assert "1-based residue position" in out
    assert "model #1 chain A" in out


def test_alignment_positions_helper():
    assert ToolRouter._alignment_rows("ABCDE", "AXCDE") == [2]
    assert ToolRouter._alignment_rows("ABCDE", "AXCYE") == [2, 4]
    assert ToolRouter._alignment_rows("ABC", "ABC") == []


# ── ChimeraX Sequence Viewer alignment FASTA + commands ─────────────────────────

def test_build_alignment_fasta_ungapped_1to1(tmp_path):
    out = pmb.build_alignment_fasta(_WT, _TOP, tmp_path / "aln.fa")
    lines = out.read_text().splitlines()
    assert lines[0] == ">WT" or lines[0].startswith(">")
    seqs = [lines[i] for i in (1, 3)]
    assert len(seqs[0]) == len(seqs[1]) == len(_WT)   # equal length, 1:1
    assert "-" not in seqs[0] and "-" not in seqs[1]  # ungapped (equal length here)
    assert seqs[0] == _WT and seqs[1] == _TOP


def test_sequence_viewer_commands_constructed(cache_tmp):
    calls = []
    class _FakeCx:
        def run_command(self, cmd, timeout=30):
            calls.append(cmd)
            return {"value": "", "error": None}
    r = ToolRouter(bridge=_FakeCx(), session=SessionState())
    note = r._open_mpnn_sequence_viewer(_WT, _TOP, "1", "A")
    # opens the alignment FASTA + force-associates the loaded chain
    assert any(c.startswith("open ") and ".fa" in c for c in calls)
    assert any(c == "sequence associate #1/A" for c in calls)
    assert "interactive alignment open" in note.lower()
    # the alignment FASTA was actually written to the cache
    assert (cache_tmp / "alignment_model1.fa").is_file()


def test_sequence_viewer_auto_decoration_targets_changed_columns(cache_tmp):
    """Auto-decoration: 3D structure coloured (tomato changed / cornflower blue
    conserved, the MPNN convention) and the changed columns SELECTED — targeting
    EXACTLY the changed positions (2, 10, 21 for _WT vs _TOP)."""
    calls = []
    class _FakeCx:
        def run_command(self, cmd, timeout=30):
            calls.append(cmd); return {"value": "", "error": None}
    r = ToolRouter(bridge=_FakeCx(), session=SessionState())
    r._open_mpnn_sequence_viewer(_WT, _TOP, "1", "A")
    changed_spec = "2,10,21"     # the diff positions
    assert f"color #1/A:{changed_spec} tomato" in calls        # changed → tomato
    assert f"select #1/A:{changed_spec}" in calls              # changed columns highlighted
    assert any(c.startswith("color #1/A:") and "cornflower blue" in c for c in calls)  # conserved
    # order: structure must be reset to white, then coloured, before select
    assert calls.index("color #1/A white") < calls.index(f"color #1/A:{changed_spec} tomato")
    assert calls.index(f"color #1/A:{changed_spec} tomato") < calls.index(f"select #1/A:{changed_spec}")


def test_compact_resspec():
    assert ToolRouter._compact_resspec([1, 2, 3, 5, 8, 9, 10]) == "1-3,5,8-10"
    assert ToolRouter._compact_resspec([2, 10, 21]) == "2,10,21"
    assert ToolRouter._compact_resspec([]) == ""
    assert ToolRouter._compact_resspec([7]) == "7"


def test_sequence_viewer_no_bridge_returns_hint():
    r = ToolRouter(bridge=None, session=SessionState())
    note = r._open_mpnn_sequence_viewer(_WT, _TOP, "1", "A")
    assert "chimerax not connected" in note.lower()
