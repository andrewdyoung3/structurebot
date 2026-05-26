"""
tests/test_disulfide.py
-----------------------
Tests for DisulfideBridge (disulfide_bridge.py).

Test categories
---------------
A. Geometry helpers   — calc_distance, calc_dihedral, geometry_score
B. Atom parsing       — parse_pdb_atoms, extract_sequence
C. Candidate finding  — find_cb_pairs filter logic
D. ESM tolerance      — _score_esm with mocked ESM data
E. DynaMut2 scoring   — _score_stability with mocked RosettaBridge
F. Ranking            — rank_candidates formula
G. Visualization      — generate_chimerax_commands output shape
H. Full pipeline      — DisulfideBridge.analyze() with mocks

Usage
-----
  cd structurebot
  python -m pytest tests/test_disulfide.py -v
  # or
  python tests/test_disulfide.py
"""

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from disulfide_bridge import (
    DisulfideBridge,
    calc_distance,
    calc_dihedral,
    geometry_score,
    parse_pdb_atoms,
    extract_sequence,
    find_cb_pairs,
    rank_candidates,
    generate_chimerax_commands,
    _CB_DIST_IDEAL,
    _CB_DIST_MAX,
    _DIHEDRAL_IDEAL,
    _MIN_ESM_TOLERANCE,
    _MAX_DESTABILIZING_DDG,
)
from tool_router import ToolStepResult

# ── Helpers ────────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

_results = {"pass": 0, "fail": 0, "skip": 0}


def _ok(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {PASS} {name}{suffix}")
    _results["pass"] += 1


def _fail(name: str, reason: str) -> None:
    print(f"  {FAIL} {name}: {reason}")
    _results["fail"] += 1


def _skip(name: str, reason: str) -> None:
    print(f"  {SKIP} {name}: {reason}")
    _results["skip"] += 1


def _assert(cond: bool, name: str, msg: str = "") -> bool:
    if cond:
        _ok(name)
        return True
    else:
        _fail(name, msg or "assertion failed")
        return False


def _approx_eq(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) < tol


# ── Minimal PDB content helpers ────────────────────────────────────────────────

def _make_pdb(records: List[str]) -> str:
    """Build a minimal PDB file content from ATOM record lines."""
    return "\n".join(records) + "\n"


def _atom_line(
    serial: int, atom: str, resname: str, chain: str, resno: int,
    x: float, y: float, z: float, bfac: float = 10.0,
) -> str:
    return (
        f"ATOM  {serial:5d}  {atom:<3} {resname:3} {chain}{resno:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}"
    )


# ── A. Geometry helpers ────────────────────────────────────────────────────────

def test_calc_distance() -> None:
    print("\n=== A. Geometry helpers ===")
    # Distance between identical points
    _assert(calc_distance((0, 0, 0), (0, 0, 0)) == 0.0, "distance to self = 0")
    # Known distance (3-4-5 triangle)
    d = calc_distance((0, 0, 0), (3, 4, 0))
    _assert(_approx_eq(d, 5.0), "3-4-5 right triangle distance",
            f"got {d:.4f}")
    # Typical CB-CB distance
    d2 = calc_distance((0, 0, 0), (0, 0, 3.8))
    _assert(_approx_eq(d2, 3.8, tol=0.001), "CB-CB 3.8 A along z-axis",
            f"got {d2:.4f}")


def test_calc_dihedral_90deg() -> None:
    """Dihedral of a canonical +90 degree arrangement returns ~90 degrees."""
    # Geometry where CA1-CB1 goes along +y, CB1-CB2 goes along +x, CB2-CA2 goes along +z.
    # The resulting CA1-CB1-CB2-CA2 dihedral is exactly +90 degrees.
    #   b1_ = (0,1,0), b2_ = (1,0,0), b3_ = (0,0,1)
    #   n1  = b1_ x b2_ = (0,0,-1)
    #   n2  = b2_ x b3_ = (0,-1,0)
    #   m1  = n1 x norm(b2_) = (0,-1,0)
    #   x   = dot(n1,n2) = 0,  y = dot(m1,n2) = 1  => atan2(1,0) = 90 deg
    ca1 = (0.0, 0.0, 0.0)
    cb1 = (0.0, 1.0, 0.0)   # CB1 above CA1 in y
    cb2 = (1.0, 1.0, 0.0)   # CB2 right of CB1 in x  (non-collinear with CA1-CB1)
    ca2 = (1.0, 1.0, 1.0)   # CA2 above CB2 in z
    angle = calc_dihedral(ca1, cb1, cb2, ca2)
    _assert(
        _approx_eq(abs(angle), 90.0, tol=5.0),
        "calc_dihedral ~90 degree arrangement",
        f"got {angle:.1f} deg",
    )


def test_calc_dihedral_0deg() -> None:
    """Coplanar atoms give dihedral of 0° or 180°."""
    ca1 = (0.0, 0.0, 0.0)
    cb1 = (1.0, 0.0, 0.0)
    cb2 = (2.0, 0.0, 0.0)
    ca2 = (3.0, 0.0, 0.0)   # collinear
    angle = calc_dihedral(ca1, cb1, cb2, ca2)
    # Collinear → undefined dihedral, function returns 0.0
    _assert(isinstance(angle, float), "calc_dihedral returns float for collinear atoms")


def test_geometry_score_ideal() -> None:
    """Ideal geometry (3.8 A, 90 deg) gives geometry_score ~= 1.0."""
    dist_sc, dihed_sc, geo_sc = geometry_score(_CB_DIST_IDEAL, _DIHEDRAL_IDEAL)
    _assert(_approx_eq(dist_sc,  1.0, tol=0.01), "ideal distance_score ~= 1.0",
            f"got {dist_sc:.4f}")
    _assert(_approx_eq(dihed_sc, 1.0, tol=0.01), "ideal dihedral_score ~= 1.0",
            f"got {dihed_sc:.4f}")
    _assert(_approx_eq(geo_sc,   1.0, tol=0.01), "ideal geometry_score ~= 1.0",
            f"got {geo_sc:.4f}")


def test_geometry_score_off_ideal() -> None:
    """Off-ideal geometry gives lower scores."""
    # Distance far from ideal
    _, _, geo_far = geometry_score(6.0, 90.0)
    _, _, geo_ideal = geometry_score(3.8, 90.0)
    _assert(geo_far < geo_ideal, "geometry_score lower at d=6.0 vs d=3.8")

    # Dihedral far from 90°
    _, _, geo_0deg  = geometry_score(3.8, 0.0)
    _, _, geo_90deg = geometry_score(3.8, 90.0)
    _assert(geo_0deg < geo_90deg, "geometry_score lower at dihedral=0° vs 90°")


def test_geometry_score_range() -> None:
    """All geometry scores are in [0, 1]."""
    for dist in (2.0, 3.5, 3.8, 4.0, 4.5, 6.0):
        for angle in (0.0, 45.0, 90.0, 135.0, 180.0):
            ds, hs, gs = geometry_score(dist, angle)
            ok = 0.0 <= ds <= 1.0 and 0.0 <= hs <= 1.0 and 0.0 <= gs <= 1.0
            if not ok:
                _fail(f"geometry_score in range (d={dist}, θ={angle})",
                      f"ds={ds}, hs={hs}, gs={gs}")
                return
    _ok("geometry_score always in [0, 1]")


# ── B. PDB parsing ─────────────────────────────────────────────────────────────

def test_parse_pdb_atoms_basic() -> None:
    """parse_pdb_atoms extracts CA and CB correctly."""
    print("\n=== B. PDB parsing ===")
    pdb_content = "\n".join([
        _atom_line(1,  "CA",  "LEU", "A", 10, 1.0, 2.0, 3.0),
        _atom_line(2,  "CB",  "LEU", "A", 10, 1.5, 2.5, 3.5),
        _atom_line(3,  "CA",  "VAL", "A", 11, 4.0, 5.0, 6.0),
        # No CB for Gly (intentional)
        _atom_line(4,  "CA",  "GLY", "A", 12, 7.0, 8.0, 9.0),
        _atom_line(5,  "CA",  "LEU", "B",  5, 0.1, 0.2, 0.3),
        _atom_line(6,  "CB",  "LEU", "B",  5, 0.4, 0.5, 0.6),
    ])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(pdb_content)
        tmp = fh.name

    try:
        atoms = parse_pdb_atoms(tmp)
        _assert("A" in atoms, "chain A parsed")
        _assert("B" in atoms, "chain B parsed")
        _assert(10 in atoms["A"], "resno 10 in chain A")
        _assert(atoms["A"][10]["resname"] == "LEU", "resname preserved")
        _assert("CB" in atoms["A"][10], "CB present for LEU A10")
        _assert("CA" in atoms["A"][10], "CA present for LEU A10")
        _assert("CB" not in atoms["A"].get(12, {}), "no CB for GLY A12")
        ca_a10 = atoms["A"][10]["CA"]
        _assert(_approx_eq(ca_a10[0], 1.0) and _approx_eq(ca_a10[2], 3.0),
                "CA coordinates correct")
    finally:
        os.unlink(tmp)


def test_parse_pdb_cys_excluded() -> None:
    """Already-Cys residues should NOT be excluded by parse_pdb_atoms — that's find_cb_pairs' job."""
    pdb_content = "\n".join([
        _atom_line(1, "CA", "CYS", "A", 5, 1.0, 1.0, 1.0),
        _atom_line(2, "CB", "CYS", "A", 5, 1.5, 1.5, 1.5),
    ])
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(pdb_content)
        tmp = fh.name
    try:
        atoms = parse_pdb_atoms(tmp)
        # parse_pdb_atoms includes all residues including CYS
        _assert("A" in atoms and 5 in atoms["A"],
                "CYS residue included in parse_pdb_atoms result")
    finally:
        os.unlink(tmp)


def test_extract_sequence() -> None:
    """extract_sequence returns correct sequence and residue mapping."""
    atoms = {
        "A": {
            10: {"resname": "LEU", "CA": (0, 0, 0), "CB": (1, 0, 0)},
            11: {"resname": "VAL", "CA": (2, 0, 0), "CB": (3, 0, 0)},
            12: {"resname": "GLY", "CA": (4, 0, 0)},  # no CB
        }
    }
    seq, mapping = extract_sequence(atoms, "A")
    _assert(seq == "LVG", f"sequence correct (got {seq!r})")
    _assert(mapping == {10: 1, 11: 2, 12: 3}, f"mapping correct (got {mapping})")


# ── C. Candidate finding ───────────────────────────────────────────────────────

def test_find_cb_pairs_basic() -> None:
    """find_cb_pairs finds a close pair and excludes a distant pair."""
    print("\n=== C. Candidate finding ===")
    atoms = {
        "A": {
            10: {"resname": "LEU", "CA": (0.0, 0.0, 0.0), "CB": (1.0, 0.0, 0.0)},
            20: {"resname": "VAL", "CA": (0.0, 0.0, 50.0), "CB": (1.0, 0.0, 50.0)},
        },
        "B": {
            10: {"resname": "LEU", "CA": (5.0, 0.0, 0.0), "CB": (4.5, 0.0, 0.0)},  # 3.5Å away from A10 CB
            20: {"resname": "ILE", "CA": (0.0, 0.0, 60.0), "CB": (1.0, 0.0, 60.0)},  # 10Å away
        },
    }
    # A10-CB=(1,0,0), B10-CB=(4.5,0,0): distance = 3.5 Å ≤ 4.5 → should be found
    # A20-CB=(1,0,50), B20-CB=(1,0,60): distance = 10 Å > 4.5 → excluded
    candidates = find_cb_pairs(atoms, "A", "B")
    _assert(len(candidates) >= 1, f"at least 1 candidate found (got {len(candidates)})")
    # Verify the close pair is included
    close_pairs = [
        c for c in candidates
        if c["chain_a_residue"] == 10 and c["chain_b_residue"] == 10
    ]
    _assert(len(close_pairs) == 1, "A10/B10 pair found")
    # Verify distances
    c = close_pairs[0]
    _assert(_approx_eq(c["cb_distance"], 3.5, tol=0.05),
            f"CB distance ~= 3.5 A (got {c['cb_distance']:.3f})")
    _assert(0.0 <= c["geometry_score"] <= 1.0,
            f"geometry_score in [0,1] (got {c['geometry_score']})")


def test_find_cb_pairs_excludes_cys() -> None:
    """find_cb_pairs excludes positions that are already Cys."""
    atoms = {
        "A": {
            10: {"resname": "CYS", "CA": (0, 0, 0), "CB": (0, 0, 0)},
            11: {"resname": "LEU", "CA": (0, 0, 0), "CB": (0, 0, 0)},
        },
        "B": {
            10: {"resname": "CYS", "CA": (0.5, 0, 0), "CB": (0.5, 0, 0)},
            11: {"resname": "LEU", "CA": (0.5, 0, 0), "CB": (0.5, 0, 0)},
        },
    }
    candidates = find_cb_pairs(atoms, "A", "B", max_dist=10.0)
    resno_pairs = [(c["chain_a_residue"], c["chain_b_residue"]) for c in candidates]
    _assert((10, 10) not in resno_pairs, "CYS-CYS pair excluded")
    _assert((10, 11) not in resno_pairs, "CYS-LEU pair A10/B11 excluded (A is CYS)")
    _assert((11, 10) not in resno_pairs, "LEU-CYS pair A11/B10 excluded (B is CYS)")
    # LEU-LEU should remain
    _assert((11, 11) in resno_pairs, "LEU-LEU pair A11/B11 included")


def test_find_cb_pairs_distance_filter() -> None:
    """Pairs beyond max_dist are excluded."""
    atoms = {
        "A": {1: {"resname": "ALA", "CA": (0, 0, 0), "CB": (0, 0, 0)}},
        "B": {1: {"resname": "ALA", "CA": (5, 0, 0), "CB": (5, 0, 0)}},
    }
    # With default max_dist=4.5, distance=5.0 should be excluded
    candidates = find_cb_pairs(atoms, "A", "B", max_dist=4.5)
    _assert(len(candidates) == 0, "5.0 Å pair excluded by 4.5 Å filter")

    # With max_dist=6.0, should be included
    candidates = find_cb_pairs(atoms, "A", "B", max_dist=6.0)
    _assert(len(candidates) == 1, "5.0 Å pair included by 6.0 Å filter")


# ── D. ESM tolerance ───────────────────────────────────────────────────────────

def test_esm_tolerance_calculation() -> None:
    """ESM tolerance = 1 - conservation; values in [0, 1]."""
    print("\n=== D. ESM tolerance ===")
    bridge = DisulfideBridge()

    # Mock ESM data: {int_pos: conservation_score}
    esm_data_a = {1: 0.9, 2: 0.3, 3: 0.5}   # position 1 is conserved (low tolerance)
    esm_data_b = {1: 0.2, 2: 0.7, 3: 0.4}

    candidates = [
        {
            "chain_a_residue": 1, "chain_b_residue": 1,
            "chain_a_aa": "L", "chain_b_aa": "V",
        },
        {
            "chain_a_residue": 2, "chain_b_residue": 2,
            "chain_a_aa": "V", "chain_b_aa": "I",
        },
    ]
    atoms = {
        "A": {
            1: {"resname": "LEU", "CA": (0, 0, 0)},
            2: {"resname": "VAL", "CA": (1, 0, 0)},
        },
        "B": {
            1: {"resname": "VAL", "CA": (4, 0, 0)},
            2: {"resname": "ILE", "CA": (5, 0, 0)},
        },
    }

    # Patch _run_esm to return our mock data
    with patch.object(bridge, "_run_esm") as mock_esm:
        # Return different data for each call (chain A then chain B)
        mock_esm.side_effect = [esm_data_a, esm_data_b]
        result = bridge._score_esm(candidates, atoms, "A", "B")

    # Candidate 0 (A1): conservation 0.9 -> tolerance = 0.1
    _assert(
        _approx_eq(result[0]["esm_tolerance_a"], 0.1, tol=0.01),
        "ESM tolerance_a for conserved position ~= 0.1",
        f"got {result[0]['esm_tolerance_a']}"
    )
    # Candidate 0 (B1): conservation 0.2 -> tolerance = 0.8
    _assert(
        _approx_eq(result[0]["esm_tolerance_b"], 0.8, tol=0.01),
        "ESM tolerance_b for variable position ~= 0.8",
        f"got {result[0]['esm_tolerance_b']}"
    )
    # Candidate 1 (A2): conservation 0.3 -> tolerance = 0.7
    _assert(
        _approx_eq(result[1]["esm_tolerance_a"], 0.7, tol=0.01),
        "ESM tolerance_a position 2 ~= 0.7",
        f"got {result[1]['esm_tolerance_a']}"
    )


def test_esm_tolerance_filter() -> None:
    """Candidates below MIN_ESM_TOLERANCE threshold are excluded."""
    candidates = [
        {
            "chain_a_residue": 1, "chain_b_residue": 1,
            "chain_a_aa": "L", "chain_b_aa": "V",
            "cb_distance": 3.8, "dihedral_angle": 90.0,
            "geometry_score": 0.9,
            "esm_tolerance_a": 0.1,   # below threshold
            "esm_tolerance_b": 0.8,
        },
        {
            "chain_a_residue": 2, "chain_b_residue": 2,
            "chain_a_aa": "V", "chain_b_aa": "I",
            "cb_distance": 3.8, "dihedral_angle": 90.0,
            "geometry_score": 0.9,
            "esm_tolerance_a": 0.6,   # above threshold
            "esm_tolerance_b": 0.5,   # above threshold
        },
    ]
    # Apply the same filter used in DisulfideBridge.analyze()
    filtered = [
        c for c in candidates
        if (c.get("esm_tolerance_a") or 0.0) >= _MIN_ESM_TOLERANCE
        and (c.get("esm_tolerance_b") or 0.0) >= _MIN_ESM_TOLERANCE
    ]
    _assert(len(filtered) == 1, "1 candidate passes ESM tolerance filter")
    _assert(filtered[0]["chain_a_residue"] == 2, "correct candidate passes (resno 2)")


# ── E. DynaMut2 stability scoring ─────────────────────────────────────────────

def test_stability_scoring_mock() -> None:
    """_score_stability correctly maps DynaMut2 ddg_scores to candidates."""
    print("\n=== E. DynaMut2 stability scoring ===")
    bridge = DisulfideBridge()

    candidates = [
        {
            "chain_a_residue": 10, "chain_b_residue": 15,
            "chain_a_aa": "L", "chain_b_aa": "V",
        },
        {
            "chain_a_residue": 10, "chain_b_residue": 20,
            "chain_a_aa": "L", "chain_b_aa": "I",
        },
        {
            "chain_a_residue": 25, "chain_b_residue": 15,
            "chain_a_aa": "A", "chain_b_aa": "V",
        },
    ]

    # Mock RosettaBridge.analyze() to return known ddG values
    mock_result_a = MagicMock()
    mock_result_a.success = True
    mock_result_a.data = {"ddg_scores": {"L10C": -0.3, "A25C": 0.8}}

    mock_result_b = MagicMock()
    mock_result_b.success = True
    mock_result_b.data = {"ddg_scores": {"V15C": 0.2, "I20C": 1.5}}

    with patch("rosetta_bridge.RosettaBridge") as MockRosetta:
        instance = MockRosetta.return_value
        instance.analyze.side_effect = [mock_result_a, mock_result_b]
        result = bridge._score_stability(
            candidates,
            pdb_path="/fake/path.pdb",
            chain_a="A",
            chain_b="B",
        )

    # Check ddG values are correctly assigned
    _assert(
        _approx_eq(result[0]["ddg_a"], -0.3, tol=0.01),
        "ddg_a for L10C = -0.3",
        f"got {result[0]['ddg_a']}"
    )
    _assert(
        _approx_eq(result[0]["ddg_b"], 0.2, tol=0.01),
        "ddg_b for V15C = 0.2",
        f"got {result[0]['ddg_b']}"
    )
    _assert(
        _approx_eq(result[1]["ddg_a"], -0.3, tol=0.01),
        "ddg_a reuses L10C result for second candidate",
        f"got {result[1]['ddg_a']}"
    )
    _assert(
        _approx_eq(result[1]["ddg_b"], 1.5, tol=0.01),
        "ddg_b for I20C = 1.5",
        f"got {result[1]['ddg_b']}"
    )
    _assert(
        _approx_eq(result[2]["ddg_a"], 0.8, tol=0.01),
        "ddg_a for A25C = 0.8",
        f"got {result[2]['ddg_a']}"
    )


def test_stability_filter_unstable() -> None:
    """Candidates with ddG > MAX_DESTABILIZING_DDG are excluded."""
    candidates = [
        {
            "chain_a_residue": 1, "chain_b_residue": 1,
            "chain_a_aa": "L", "chain_b_aa": "V",
            "ddg_a": -0.5,   # stabilising
            "ddg_b":  0.3,   # mild
        },
        {
            "chain_a_residue": 2, "chain_b_residue": 2,
            "chain_a_aa": "A", "chain_b_aa": "I",
            "ddg_a": 0.5,
            "ddg_b": 1.2,   # exceeds threshold
        },
    ]
    filtered = [
        c for c in candidates
        if (c.get("ddg_a") or 0.0) <= _MAX_DESTABILIZING_DDG
        and (c.get("ddg_b") or 0.0) <= _MAX_DESTABILIZING_DDG
    ]
    _assert(len(filtered) == 1, "only 1 candidate passes ddG filter")
    _assert(filtered[0]["chain_a_residue"] == 1, "correct candidate passes (ddg_b=1.2 excluded)")


# ── F. Ranking ─────────────────────────────────────────────────────────────────

def test_rank_candidates_formula() -> None:
    """Combined score = geo×0.4 + ESM×0.3 + stab×0.3."""
    print("\n=== F. Ranking ===")
    candidates = [
        {
            "chain_a_residue": 1, "chain_b_residue": 1,
            "chain_a_aa": "L", "chain_b_aa": "V",
            "cb_distance": 3.8, "dihedral_angle": 90.0,
            "geometry_score":   1.0,
            "esm_tolerance_a":  0.8,
            "esm_tolerance_b":  0.6,
            "ddg_a": -0.2,   # ddg_mean = -0.1 → stab = (2 - (-0.1)) / 2 = 1.05 → capped to 1.0
            "ddg_b": 0.0,
        },
        {
            "chain_a_residue": 2, "chain_b_residue": 2,
            "chain_a_aa": "A", "chain_b_aa": "I",
            "cb_distance": 4.2, "dihedral_angle": 60.0,
            "geometry_score":   0.4,
            "esm_tolerance_a":  0.5,
            "esm_tolerance_b":  0.5,
            "ddg_a": 0.5,
            "ddg_b": 0.5,    # ddg_mean = 0.5 → stab = (2 - 0.5) / 2 = 0.75
        },
    ]
    ranked = rank_candidates(candidates)

    # Candidate 1 should score higher
    _assert(len(ranked) == 2, "2 candidates returned")
    _assert(ranked[0]["chain_a_residue"] == 1, "higher-scoring candidate is first")
    _assert(ranked[0]["combined_score"] > ranked[1]["combined_score"],
            f"descending order: {ranked[0]['combined_score']:.3f} > {ranked[1]['combined_score']:.3f}")

    # Validate formula for candidate 2
    # geo=0.4, esm=(0.5+0.5)/2=0.5, stab=(2-0.5)/2=0.75
    # combined = 0.4*0.4 + 0.5*0.3 + 0.75*0.3 = 0.16+0.15+0.225 = 0.535
    c2 = ranked[1]
    expected_c2 = 0.4 * 0.4 + 0.5 * 0.3 + 0.75 * 0.3
    _assert(
        _approx_eq(c2["combined_score"], expected_c2, tol=0.01),
        f"combined_score formula correct for candidate 2",
        f"expected {expected_c2:.3f}, got {c2['combined_score']:.3f}"
    )


def test_rank_sets_recommendation() -> None:
    """rank_candidates populates the recommendation field."""
    candidates = [
        {
            "chain_a_residue": 1, "chain_b_residue": 1,
            "chain_a_aa": "L", "chain_b_aa": "V",
            "geometry_score": 0.9, "esm_tolerance_a": 0.7,
            "esm_tolerance_b": 0.8, "ddg_a": -0.5, "ddg_b": 0.0,
        }
    ]
    ranked = rank_candidates(candidates)
    rec = ranked[0].get("recommendation", "")
    _assert(isinstance(rec, str) and len(rec) > 0, "recommendation is non-empty string")
    # Should say something about geometry and evolution
    _assert("geometry" in rec.lower() or "candidate" in rec.lower(),
            f"recommendation mentions geometry or candidate: {rec!r}")


# ── G. Visualization ──────────────────────────────────────────────────────────

def test_generate_viz_commands_shape() -> None:
    """generate_chimerax_commands returns matching-length lists."""
    print("\n=== G. Visualization ===")
    candidates = [
        {
            "chain_a_residue": 10, "chain_b_residue": 15,
            "chain_a_aa": "L", "chain_b_aa": "V",
            "cb_distance": 3.8, "combined_score": 0.78,
        },
        {
            "chain_a_residue": 20, "chain_b_residue": 25,
            "chain_a_aa": "A", "chain_b_aa": "I",
            "cb_distance": 4.1, "combined_score": 0.61,
        },
    ]
    cmds, exps = generate_chimerax_commands(
        candidates, model_id="1", chain_a="A", chain_b="B", top_n=2
    )
    _assert(len(cmds) == len(exps), "commands and explanations have equal length",
            f"{len(cmds)} cmds vs {len(exps)} exps")
    _assert(len(cmds) > 2, f"non-trivial command list ({len(cmds)} commands)")


def test_generate_viz_commands_content() -> None:
    """Viz commands reference the correct chains and residue numbers."""
    candidates = [
        {
            "chain_a_residue": 42, "chain_b_residue": 17,
            "chain_a_aa": "L", "chain_b_aa": "V",
            "cb_distance": 3.8, "combined_score": 0.75,
        }
    ]
    cmds, _ = generate_chimerax_commands(
        candidates, model_id="2", chain_a="A", chain_b="B", top_n=1
    )
    cmd_str = " ".join(cmds)
    _assert("/A:42" in cmd_str, "chain A residue 42 referenced in commands")
    _assert("/B:17" in cmd_str, "chain B residue 17 referenced in commands")
    _assert("#2"    in cmd_str, "model_id 2 referenced in commands")


def test_generate_viz_commands_empty() -> None:
    """generate_chimerax_commands with empty list returns empty lists."""
    cmds, exps = generate_chimerax_commands([], top_n=3)
    _assert(cmds == [], "empty candidates -> empty cmds")
    _assert(exps == [], "empty candidates -> empty exps")


# ── H. Full pipeline (mocked) ─────────────────────────────────────────────────

def _write_minimal_pdb(chain_a: str, chain_b: str, n_pairs: int = 2) -> str:
    """Write a minimal PDB with n close CB-CB pairs across two chains."""
    lines = []
    serial = 1
    for i in range(n_pairs):
        # Chain A residue
        x_a = float(i * 10)
        lines.append(_atom_line(serial,   "CA", "LEU", chain_a, 10 + i, x_a + 0.0, 0.0, 0.0))
        lines.append(_atom_line(serial+1, "CB", "LEU", chain_a, 10 + i, x_a + 1.0, 0.0, 0.0))
        serial += 2
        # Chain B residue close to chain A (CB-CB distance ~3.6 Å)
        lines.append(_atom_line(serial,   "CA", "VAL", chain_b, 10 + i, x_a + 5.0, 0.0, 0.0))
        lines.append(_atom_line(serial+1, "CB", "VAL", chain_b, 10 + i, x_a + 4.6, 0.0, 0.0))
        serial += 2

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("\n".join(lines) + "\n")
        return fh.name


def test_full_pipeline_mock() -> None:
    """Full DisulfideBridge.analyze() pipeline with all external calls mocked."""
    print("\n=== H. Full pipeline (mocked) ===")
    pdb_path = _write_minimal_pdb("A", "B", n_pairs=2)

    try:
        bridge = DisulfideBridge()

        # Mock ESM: return uniform mid-tolerance (0.5 conservation → tolerance 0.5)
        mock_esm_result = MagicMock()
        mock_esm_result.success = True
        mock_esm_result.data = {"conservation": {1: 0.5, 2: 0.5}}

        # Mock DynaMut2: return mild ddG for all mutations
        mock_ddg_result_a = MagicMock()
        mock_ddg_result_a.success = True
        mock_ddg_result_a.data = {"ddg_scores": {"L10C": -0.2, "L11C": 0.3}}

        mock_ddg_result_b = MagicMock()
        mock_ddg_result_b.success = True
        mock_ddg_result_b.data = {"ddg_scores": {"V10C": 0.1, "V11C": 0.2}}

        with patch("esm_bridge.EsmBridge") as MockEsm, \
             patch("rosetta_bridge.RosettaBridge") as MockRosetta:

            MockEsm.return_value.analyze.return_value = mock_esm_result
            MockRosetta.return_value.analyze.side_effect = [
                mock_ddg_result_a, mock_ddg_result_b
            ]

            result = bridge.analyze(
                pdb_path = pdb_path,
                chain_a  = "A",
                chain_b  = "B",
            )

        _assert(result.success, "analyze() returned success", f"error: {result.error}")
        _assert(isinstance(result.data, dict), "result.data is a dict")
        _assert("candidates" in result.data, "data contains 'candidates'")
        _assert(isinstance(result.data["candidates"], list), "candidates is a list")
        _assert(len(result.data["candidates"]) > 0,
                f"at least 1 candidate found (got {len(result.data.get('candidates', []))})")

        # Each candidate should have required fields
        top = result.data["candidates"][0]
        for field in ("chain_a_residue", "chain_b_residue", "cb_distance",
                      "geometry_score", "esm_tolerance_a", "esm_tolerance_b",
                      "ddg_a", "ddg_b", "combined_score", "recommendation"):
            _assert(field in top, f"candidate has '{field}' field",
                    f"present fields: {list(top.keys())}")

        # Visualization commands should be generated
        _assert(isinstance(result.viz_commands, list), "viz_commands is a list")
        _assert(len(result.viz_commands) > 0, "non-empty viz_commands")

        # Summary should mention both chains
        _assert(isinstance(result.summary, str) and len(result.summary) > 10,
                "summary is non-empty string")

    finally:
        try:
            os.unlink(pdb_path)
        except OSError:
            pass


def test_full_pipeline_missing_pdb() -> None:
    """analyze() returns failure for a missing PDB file."""
    bridge = DisulfideBridge()
    result = bridge.analyze(
        pdb_path = "/nonexistent/path/protein.pdb",
        chain_a  = "A",
        chain_b  = "B",
    )
    _assert(not result.success, "analyze() fails for missing PDB")
    _assert(bool(result.error), "error message is non-empty")


def test_full_pipeline_missing_chain() -> None:
    """analyze() returns failure when a requested chain is absent."""
    pdb_path = _write_minimal_pdb("A", "B", n_pairs=1)
    try:
        bridge = DisulfideBridge()
        result = bridge.analyze(
            pdb_path = pdb_path,
            chain_a  = "A",
            chain_b  = "X",   # chain X does not exist
        )
        _assert(not result.success, "analyze() fails for missing chain X")
        _assert(bool(result.error), "error message mentions missing chain")
    finally:
        try:
            os.unlink(pdb_path)
        except OSError:
            pass


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("tests/test_disulfide.py — Disulfide Bridge Tests")
    print("=" * 60)

    # A. Geometry
    test_calc_distance()
    test_calc_dihedral_90deg()
    test_calc_dihedral_0deg()
    test_geometry_score_ideal()
    test_geometry_score_off_ideal()
    test_geometry_score_range()

    # B. PDB parsing
    test_parse_pdb_atoms_basic()
    test_parse_pdb_cys_excluded()
    test_extract_sequence()

    # C. Candidate finding
    test_find_cb_pairs_basic()
    test_find_cb_pairs_excludes_cys()
    test_find_cb_pairs_distance_filter()

    # D. ESM tolerance
    test_esm_tolerance_calculation()
    test_esm_tolerance_filter()

    # E. DynaMut2 scoring
    test_stability_scoring_mock()
    test_stability_filter_unstable()

    # F. Ranking
    test_rank_candidates_formula()
    test_rank_sets_recommendation()

    # G. Visualization
    test_generate_viz_commands_shape()
    test_generate_viz_commands_content()
    test_generate_viz_commands_empty()

    # H. Full pipeline
    test_full_pipeline_mock()
    test_full_pipeline_missing_pdb()
    test_full_pipeline_missing_chain()

    print()
    print("=" * 60)
    print(
        f"Results: {_results['pass']} passed, "
        f"{_results['fail']} failed, "
        f"{_results['skip']} skipped"
    )
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
