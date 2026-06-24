"""
tests/test_structural_align.py
------------------------------
Stage 3: sequence-INDEPENDENT structural alignment (US-align) — router side.
All mocked: no real WSL/US-align, no live ChimeraX. Covers
  A. _parse_usalign_output  — the `-outfmt 2 -m -` tab line + 3×4 matrix (row-major).
  B. _view_matrix_command   — the ChimeraX view-matrix command form (option B).
  C. _run_structural_align  — captures both TM-scores/RMSD/Lali + transform, overlays the
     construct fold via `view matrix` (option B), honest shared-fold tier; error-first.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import wsl_bridge
from tool_router import ToolRouter

# A real US-align `-outfmt 2 -m -` capture (1LH1 leghemoglobin vs 4HHB chain-A hemoglobin).
USALIGN_OUT = (
    "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali\n"
    "1LH1.pdb:A\t4HHB_A.pdb:A\t0.7208\t0.7711\t2.46\t0.098\t0.106\t0.110\t153\t141\t136\n"
    "------ The rotation matrix to rotate Structure_1 to Structure_2 ------\n"
    "m               t[m]        u[m][0]        u[m][1]        u[m][2]\n"
    "0       4.5360591001   0.9063449333  -0.2267566096  -0.3565393412\n"
    "1      13.6309498105  -0.1267169519   0.6590983493  -0.7413043775\n"
    "2      10.7855668712   0.4030901585   0.7170570451   0.5686365431\n"
    "\nCode for rotating Structure 1 from (x,y,z) to (X,Y,Z):\n"
)
# An unrelated pair (1UBQ vs 4HHB/A): low TM → honest negative.
USALIGN_LOW = (
    "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali\n"
    "1UBQ.pdb:A\t4HHB_A.pdb:A\t0.32886\t0.22515\t4.46\t0.020\t0.011\t0.020\t76\t141\t51\n"
    "------ The rotation matrix to rotate Structure_1 to Structure_2 ------\n"
    "m               t[m]        u[m][0]        u[m][1]        u[m][2]\n"
    "0       1.0   1.0  0.0  0.0\n"
    "1       2.0   0.0  1.0  0.0\n"
    "2       3.0   0.0  0.0  1.0\n"
)


def _router() -> ToolRouter:
    return ToolRouter(bridge=MagicMock(), session=MagicMock())


def _fake_wsl():
    w = MagicMock()
    w.is_available.return_value = True
    w.translate_path.side_effect = lambda p: "/mnt/x" + str(p).replace("\\", "/")
    return w


# ── A. parser ──────────────────────────────────────────────────────────────────
def test_parse_scores_and_matrix_row_major():
    p = ToolRouter._parse_usalign_output(USALIGN_OUT)
    assert p["tm1"] == 0.7208 and p["tm2"] == 0.7711      # TM1=query-norm, TM2=ref-norm
    assert p["rmsd"] == 2.46 and p["lali"] == 136
    assert p["l1"] == 153 and p["l2"] == 141 and p["idali"] == 0.110
    # ROW-MAJOR [u00,u01,u02,t0, …] — the ChimeraX view-matrix order (live-verified convention)
    assert p["matrix"][0:4] == [0.9063449333, -0.2267566096, -0.3565393412, 4.5360591001]
    assert p["matrix"][4:8] == [-0.1267169519, 0.6590983493, -0.7413043775, 13.6309498105]
    assert p["matrix"][8:12] == [0.4030901585, 0.7170570451, 0.5686365431, 10.7855668712]


def test_parse_garbage_and_empty_return_none():
    assert ToolRouter._parse_usalign_output("no usable output here") is None
    assert ToolRouter._parse_usalign_output("") is None


def test_parse_scores_without_matrix_keeps_scores_matrix_none():
    only_scores = USALIGN_OUT.split("------")[0]          # tab line, no matrix block
    p = ToolRouter._parse_usalign_output(only_scores)
    assert p is not None and p["tm2"] == 0.7711 and p["matrix"] is None


# ── B. view-matrix command (option B) ────────────────────────────────────────────
def test_view_matrix_command_is_row_major_csv():
    cmd = ToolRouter._view_matrix_command("7", [1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4])
    assert cmd == "view matrix models #7,1,0,0,2,0,1,0,3,0,0,1,4"
    assert ToolRouter._view_matrix_command("#9", [0] * 12).startswith("view matrix models #9,")


# ── C. _run_structural_align ─────────────────────────────────────────────────────
def test_run_captures_scores_and_overlays_query_model(tmp_path, monkeypatch):
    r = _router()
    qf = tmp_path / "fold.cif"; qf.write_text("x")
    rf = tmp_path / "ref.pdb"; rf.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": True, "stdout": USALIGN_OUT, "stderr": "", "error": None}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    r.bridge.run_command.return_value = {"value": "Opened model #12"}
    r._parse_model_spec = MagicMock(return_value="#12")
    out = r._run_structural_align({
        "query_path": str(qf), "query_model_id": "7",
        "reference_path": str(rf), "ref_label": "1MBN"})
    assert out.success
    d = out.data
    # BOTH TM-scores captured; default-surfaced is the reference-normalized one (TM2)
    assert d["tm_ref"] == 0.7711 and d["tm_query"] == 0.7208
    assert d["rmsd"] == 2.46 and d["n_aligned"] == 136 and d["shared_fold"] is True
    assert len(d["matrix"]) == 12
    # option B: the construct fold model (#7) is MOVED via view matrix (no extra model)
    assert any(c.startswith("view matrix models #7,") for c in d["overlay_commands"])
    assert d["reference_model_id"] == "12"               # reference opened + id read back
    # US-align WAS invoked on the translated paths
    assert w.run_command.called and "USalign" in w.run_command.call_args[0][0]


def test_run_low_tm_reports_not_similar(tmp_path, monkeypatch):
    r = _router()
    qf = tmp_path / "f.cif"; qf.write_text("x"); rf = tmp_path / "r.pdb"; rf.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": True, "stdout": USALIGN_LOW, "stderr": "", "error": None}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    r.bridge.run_command.return_value = {"value": "#3"}
    r._parse_model_spec = MagicMock(return_value="#3")
    out = r._run_structural_align({
        "query_path": str(qf), "query_model_id": "2", "reference_path": str(rf), "ref_label": "4HHB"})
    assert out.success
    assert out.data["tm_ref"] == 0.2251 and out.data["shared_fold"] is False
    assert "NOT structurally similar" in out.summary


def test_run_loaded_model_reference_not_reopened(tmp_path, monkeypatch):
    # a loaded-model reference (panel saved its file) → router uses the OPEN model, no `open`
    r = _router()
    qf = tmp_path / "f.cif"; qf.write_text("x"); rf = tmp_path / "r.pdb"; rf.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": True, "stdout": USALIGN_OUT, "stderr": "", "error": None}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    r.bridge.run_command.return_value = {"value": ""}
    out = r._run_structural_align({
        "query_path": str(qf), "query_model_id": "7", "reference_path": str(rf),
        "reference_model_id": "5", "ref_label": "#5"})
    assert out.success and out.data["reference_model_id"] == "5"
    assert not any(c.startswith("open ") for c in out.data["overlay_commands"])


def test_run_missing_query_file_errors():
    out = _router()._run_structural_align({"query_path": "/nope/x.cif", "reference_pdb_id": "1MBN"})
    assert not out.success and "fold file" in out.error.lower()


def test_run_usalign_failure_errors(tmp_path, monkeypatch):
    r = _router()
    qf = tmp_path / "f.cif"; qf.write_text("x"); rf = tmp_path / "r.pdb"; rf.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": False, "stdout": "", "stderr": "boom", "error": "exit 1"}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    out = r._run_structural_align({"query_path": str(qf), "reference_path": str(rf)})
    assert not out.success and "US-align failed" in out.error


# ── D. _run_align_folds — compare two existing folds (US-align + per-residue), reuse only ──
def _ca_line(coords):
    # minimal {resnum: (x,y,z)} map for the per-residue deviation reuse
    return {i + 1: xyz for i, xyz in enumerate(coords)}


def test_align_folds_compares_two_folds_with_framing(tmp_path, monkeypatch):
    """Two existing fold files (local Boltz vs remote ColabFold) → US-align TM/RMSD + per-residue
    deviation, with the asymmetry stated up front. REUSE ONLY (no new alignment code)."""
    r = _router()
    fa = tmp_path / "boltz.cif"; fa.write_text("x")
    fb = tmp_path / "colabfold.pdb"; fb.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": True, "stdout": USALIGN_OUT, "stderr": "", "error": None}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    # both models "open" → identical Cα so the per-residue agreement path runs
    ca = _ca_line([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    r._read_fold_ca = MagicMock(return_value=ca)
    out = r._run_align_folds({
        "fold_a": {"label": "#1 boltz", "engine": "boltz", "model_id": "1",
                   "path": str(fa), "remote_msa": False},
        "fold_b": {"label": "#2 colabfold", "engine": "colabfold", "model_id": "2",
                   "path": str(fb), "remote_msa": True},
    })
    assert out.success
    d = out.data
    assert d["tm"] == 0.7711 and d["rmsd"] == 2.46 and d["n_aligned"] == 136
    assert d["mean_ddm_A"] == 0.0 and d["mean_lddt"] == 1.0 and d["n_common"] == 4  # identical Cα
    # asymmetry stated: local single-sequence vs MSA-informed (NOT a fair model-vs-model test)
    assert "local single-sequence" in d["framing"] and "MSA-informed" in d["framing"]
    assert "not a fair model-vs-model test" in d["framing"].lower()
    assert "boltz" in out.summary.lower() and "colabfold" in out.summary.lower()
    assert "USalign" in w.run_command.call_args[0][0]


def test_pdb_id_reference_registered_in_session(tmp_path, monkeypatch):
    # The just-opened PDB-id reference is tracked in session.structures (aligned_reference
    # metadata, name=PDB id) → under the visibility authority + reopen-by-id on reconnect.
    r = _router()
    qf = tmp_path / "fold.cif"; qf.write_text("x")
    rf = tmp_path / "1mbn.pdb"; rf.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": True, "stdout": USALIGN_OUT, "stderr": "", "error": None}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    r._download_pdb_by_id = MagicMock(return_value=str(rf))
    r.bridge.run_command.return_value = {"value": "Opened model #12"}
    r._parse_model_spec = MagicMock(return_value="#12")
    out = r._run_structural_align({
        "query_path": str(qf), "query_model_id": "7",
        "reference_pdb_id": "1MBN", "ref_label": "1MBN"})
    assert out.success
    r.session.add_structure.assert_called_once()
    args, kwargs = r.session.add_structure.call_args
    assert args[0] == "12"                                  # the opened reference's id
    assert kwargs["metadata"]["aligned_reference"] is True
    assert kwargs["metadata"]["ref_pdb_id"] == "1MBN"


def test_loaded_model_reference_not_reregistered(tmp_path, monkeypatch):
    # A loaded-model reference is ALREADY its own session structure → no double-registration.
    r = _router()
    qf = tmp_path / "f.cif"; qf.write_text("x"); rf = tmp_path / "r.pdb"; rf.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": True, "stdout": USALIGN_OUT, "stderr": "", "error": None}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    r.bridge.run_command.return_value = {"value": ""}
    r._run_structural_align({"query_path": str(qf), "query_model_id": "7",
                             "reference_path": str(rf), "reference_model_id": "5", "ref_label": "#5"})
    r.session.add_structure.assert_not_called()             # #5 was already open/registered


def test_align_folds_needs_both_files_on_disk(tmp_path):
    r = _router()
    fa = tmp_path / "a.cif"; fa.write_text("x")
    out = r._run_align_folds({"fold_a": {"path": str(fa)},
                              "fold_b": {"path": "/nope/b.pdb"}})
    assert not out.success and "on disk" in out.error.lower()


def test_align_folds_skips_per_residue_when_no_open_model(tmp_path, monkeypatch):
    r = _router()
    fa = tmp_path / "a.cif"; fa.write_text("x"); fb = tmp_path / "b.pdb"; fb.write_text("y")
    w = _fake_wsl()
    w.run_command.return_value = {"ok": True, "stdout": USALIGN_OUT, "stderr": "", "error": None}
    monkeypatch.setattr(wsl_bridge, "WSLBridge", lambda **k: w)
    out = r._run_align_folds({"fold_a": {"label": "A", "path": str(fa)},   # no model_id → no Cα
                              "fold_b": {"label": "B", "path": str(fb)}})
    assert out.success and out.data["mean_ddm_A"] is None       # TM/RMSD still captured
    assert out.data["tm"] == 0.7711
