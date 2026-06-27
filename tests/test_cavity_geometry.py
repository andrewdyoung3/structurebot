"""
tests/test_cavity_geometry.py
-----------------------------
The plain-Python cavity-filling core (cavity_geometry): grid+flood-fill internal-void detection
(enclosed vs surface-pocket discrimination, the volume floor), and the ROTAMER-AWARE fill half
(a larger side chain credited only when a rotamer reaches INTO the void clash-free; demoted/dropped
otherwise). No GPU, no ChimeraX. Real-structure checks skip silently if the cache CIF isn't present.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cavity_geometry as cg
from disulfide_geometry import ClashGrid

_CACHE = Path(__file__).parent.parent / "cache"


# ── synthetic structures ──────────────────────────────────────────────────────────────────────
def _sphere_shell(radius=6.0, n=400, z_cap=None, chain="S"):
    """A watertight Fibonacci-sphere shell of carbons around the origin (an interior void). With
    *z_cap* set, atoms with z > z_cap are REMOVED — a hole that opens the interior to bulk solvent
    (a surface pocket, no longer enclosed)."""
    atoms = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    idx = 0
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = ga * i
        x, z = math.cos(theta) * r, math.sin(theta) * r
        pos = (radius * x, radius * y, radius * z)
        if z_cap is not None and pos[2] > z_cap:
            continue
        idx += 1
        atoms.append((chain, idx, "C", pos))
    return atoms


# ── detection: enclosed cavity, surface-pocket discrimination, the volume floor ─────────────────
def test_detects_one_enclosed_cavity():
    atoms = _sphere_shell(radius=6.0, n=500)
    cavs = cg.detect_cavities(atoms)
    assert len(cavs) == 1
    c = cavs[0]
    assert c["volume"] >= cg.MIN_VOLUME
    # the void is centred on the origin (the hollow centre)
    assert all(abs(x) < 1.5 for x in c["centroid"])


def test_surface_pocket_not_flagged_internal():
    # the same shell with a cap removed: the interior connects to bulk solvent → NOT an internal void
    atoms = _sphere_shell(radius=6.0, n=500, z_cap=3.0)
    cavs = cg.detect_cavities(atoms)
    assert cavs == []


def test_volume_floor_skips_subthreshold():
    # a real enclosed void exists, but an absurd floor rejects everything → the gate works
    atoms = _sphere_shell(radius=6.0, n=500)
    assert cg.detect_cavities(atoms, min_volume=1.0e6) == []


def test_grid_safety_coarsens_not_hangs():
    # a huge bounding box must coarsen (effective spacing grows), never allocate > MAX_GRID_CELLS
    atoms = _sphere_shell(radius=6.0, n=300)
    grid, eff = cg._build_grid(atoms, spacing=0.001, probe=cg.PROBE_RADIUS)
    assert grid.nx * grid.ny * grid.nz <= cg.MAX_GRID_CELLS and eff > 0.001


# ── fill: rotamer-aware reach-into-void, clash-free credit, non-reaching → None ─────────────────
def _lining_ala(toward_center_from):
    """An ALA residue lining the origin void: CA near the shell, CB pointing inward toward (0,0,0)."""
    ca = toward_center_from
    # CB ~1.5 Å inward of CA along CA→origin
    d = math.sqrt(sum(v * v for v in ca))
    u = tuple(-v / d for v in ca)
    cb = tuple(ca[i] + 1.5 * u[i] for i in range(3))
    n = (ca[0], ca[1] + 1.0, ca[2])
    return {"N": n, "CA": ca, "CB": cb, "resname": "ALA"}


def test_fill_reaching_rotamer_credited_clash_free():
    atoms = _sphere_shell(radius=6.5, n=600)
    res = {"S": {10: _lining_ala((5.0, 0.0, 0.0))}}
    cavs = cg.detect_cavities(atoms, residues=res)
    assert cavs, "expected an enclosed cavity"
    grid = ClashGrid(atoms)
    cand = cg.fill_candidate(cavs[0], "S", 10, res["S"][10], "V", grid)
    assert cand is not None
    assert cand["from_aa"] == "A" and cand["to_aa"] == "V"
    assert cand["clash"] is False and cand["score"] > 0.0
    assert 0.0 < cand["fill_fraction"] <= 1.0


def test_fill_non_reaching_residue_returns_none():
    # a residue whose Cβ points AWAY from the void: no rotamer reaches → not a fill candidate
    atoms = _sphere_shell(radius=6.5, n=600)
    ca = (5.0, 0.0, 0.0)
    cb = (6.2, 0.0, 0.0)                       # Cβ OUTWARD (away from the origin void)
    res = {"N": (5.0, 1.0, 0.0), "CA": ca, "CB": cb, "resname": "ALA"}
    cavs = cg.detect_cavities(atoms)
    grid = ClashGrid(atoms)
    assert cg.fill_candidate(cavs[0], "S", 10, res, "V", grid) is None


def test_fill_rejects_disallowed_enlargement():
    atoms = _sphere_shell(radius=6.5, n=600)
    res = _lining_ala((5.0, 0.0, 0.0)); res["resname"] = "TRP"   # W has no allowed enlargement
    cavs = cg.detect_cavities(atoms, residues={"S": {10: res}})
    grid = ClashGrid(atoms)
    assert cg.fill_candidate(cavs[0], "S", 10, res, "V", grid) is None


def test_void_member_tolerates_boundary():
    atoms = _sphere_shell(radius=6.0, n=500)
    cav = cg.detect_cavities(atoms)[0]
    assert cg._void_member((0.0, 0.0, 0.0), cav["cells"], cav["grid"]) is True
    assert cg._void_member((30.0, 30.0, 30.0), cav["cells"], cav["grid"]) is False


# ── the scan end-to-end (synthetic): viable fills surfaced with honest metrics ──────────────────
def test_scan_surfaces_viable_fill_with_metrics():
    atoms = _sphere_shell(radius=6.5, n=600)
    res = {"S": {10: _lining_ala((5.0, 0.0, 0.0))}}
    cands, best, cavities = cg.scan_cavity_sites(atoms, res)
    assert cavities and cavities[0]["volume"] >= cg.MIN_VOLUME
    assert cands, "a lining ALA pointing into the void should yield a viable fill"
    top = cands[0]
    assert {"chain", "position", "from_aa", "to_aa", "cavity_id", "void_volume",
            "fill_fraction", "reach_score", "clash", "score"} <= set(top)
    # sorted by score desc; best_partner mirrors the lining residue's best score
    assert all(cands[i]["score"] >= cands[i + 1]["score"] for i in range(len(cands) - 1))
    assert best["S"][10] == max(c["score"] for c in cands if c["position"] == 10)


# ── real structure (skips silently without the cache CIF) ───────────────────────────────────────
def test_scan_on_real_structure_is_sane():
    cif = _CACHE / "1MBN.cif"
    if not cif.is_file():
        return
    heavy = cg.parse_heavy_atoms(str(cif))
    residues = cg.parse_residue_atoms(str(cif))
    cands, best, cavities = cg.scan_cavity_sites(heavy, residues)
    # myoglobin is a compact globin with documented internal packing voids — detection must find some
    assert cavities, "myoglobin should have internal cavities at the 1.4 Å probe"
    for cav in cavities:
        assert cav["volume"] >= cg.MIN_VOLUME
    # a curated handful, not a flood (the softened curation: viable fills above the noise floor)
    assert 0 < len(cands) < 40
    # every surfaced candidate is a conservative small→larger hydrophobic enlargement with a real score
    for c in cands:
        assert c["to_aa"] in cg._VOLUME_MUTATIONS.get(c["from_aa"], [])
        assert c["score"] > 0.0


# ── router: cavity_scan + cavity_ddg_estimate (parallel to the disulfide/proline tools) ─────────
from unittest.mock import MagicMock
from tool_router import ToolRouter
from session_state import SessionState


def _router():
    return ToolRouter(bridge=MagicMock(), session=SessionState())


def _write(text, suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    f.write(text); f.close()
    return f.name


def test_run_cavity_scan_context_dependent_caveat_and_named_step():
    cif = _CACHE / "1MBN.cif"
    if not cif.is_file():
        return
    r = _router()
    out = r._run_cavity_scan({"cif_path": str(cif)})
    assert out.success and out.data["candidates"]
    cav = out.data["caveat"]
    # the corrected, context-dependent framing — BOTH literatures named, NOT "least-reliable"
    assert "RSV" in cav and "conformational" in cav.lower() and "Matthews" in cav
    assert "least-reliable" not in cav.lower()
    assert "does not confirm" in cav.lower() and "validate" in cav.lower()
    assert "cavities" in out.data and isinstance(out.data["cavities"], list)
    # the pipeline strip shows the REAL name, not "Unknown tool"
    assert "Cavity-filling scan" in r._step_description("cavity_scan", {}, MagicMock())


def test_run_cavity_scan_needs_a_cif():
    out = _router()._run_cavity_scan({"cif_path": "/no/such/file.cif"})
    assert out.success is False and "cif" in out.error.lower()


def _mini_pdb_ala5():
    return ("ATOM      1  CA  ALA A   5       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n")


def test_cavity_ddg_from_aa_mismatch_aborts():
    pdb = _write(_mini_pdb_ala5(), ".pdb")
    out = _router()._run_cavity_ddg_estimate(
        {"pdb_path": pdb, "chain": "A", "resnum": 5, "from_aa": "V", "to_aa": "I", "source": "loaded"})
    assert out.success is False and "mismatch" in out.error.lower()   # PDB has ALA, scan said VAL → abort


def test_cavity_ddg_same_residue_is_noop():
    pdb = _write(_mini_pdb_ala5(), ".pdb")
    out = _router()._run_cavity_ddg_estimate(
        {"pdb_path": pdb, "chain": "A", "resnum": 5, "from_aa": "A", "to_aa": "A", "source": "loaded"})
    assert out.success is False and "no mutation" in out.error.lower()


def test_cavity_ddg_gate_blocks_denovo_web_upload():
    r = _router()
    ok, reason = r._ddg_escalation_gate("dynamut2", "denovo", wsl_ok=False)
    assert ok is False and "upload" in reason.lower()
    ok2, _ = r._ddg_escalation_gate("local", "loaded", wsl_ok=True)
    assert ok2 is True


def test_cavity_tools_have_icons_not_unknown():
    for tool in ("cavity_scan", "cavity_ddg_estimate"):
        assert ToolRouter._TOOL_ICONS.get(tool) and ToolRouter._TOOL_ICONS[tool] != "⚙️"
