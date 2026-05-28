"""
structural_utils.py
-------------------
Shared structural geometry utilities for StructureBot bridges.

Centralises functions that were duplicated across proline_bridge.py and
glycan_bridge.py.

Used by
-------
  glycan_bridge.py  — SASA, backbone angles, projection scoring, sequon geometry
  proline_bridge.py — backbone angles, SASA-based active-site detection

Public API
----------
extract_backbone_angles(pdb_path, chain) -> dict[int, dict]
compute_sasa(pdb_path, chain)            -> dict[int, float]
compute_projection_score(pdb_path, chain) -> dict[int, dict]
classify_sequon_geometry(backbone, pos_i) -> str

Critical rules (same as all StructureBot bridges)
--------------------------------------------------
- Pure Python — no subprocess.
- All file-path args use Path(...).as_posix() internally.
- sys.stdout.reconfigure() MUST NOT be used.
- Every public function returns gracefully on failure; callers must handle
  empty-dict / "unknown" returns without crashing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ── Amino-acid code tables ────────────────────────────────────────────────────

_ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
_THREE_TO_ONE: Dict[str, str] = {v: k for k, v in _ONE_TO_THREE.items()}


# ── Private helpers ───────────────────────────────────────────────────────────

def _classify_ss_from_angles(phi: float, psi: float) -> str:
    """Classify φ/ψ into H (helix), E (sheet), or L (loop/coil)."""
    if (-90.0 < phi < -45.0) and (-60.0 < psi < -10.0):
        return "H"
    if (-160.0 < phi < -90.0) and (90.0 < abs(psi) < 180.0):
        return "E"
    return "L"


def _classify_single_pos(phi: float, psi: float) -> str:
    """
    Classify one backbone position for sequon-geometry scoring.

    Returns one of: H (helix), E (extended strand), T (turn-like), L (loop).

    Helix   : strict α-helix Ramachandran region
    Extended: β-strand region (φ ≤ -90, |ψ| ≥ 90)
    Turn    : a non-helix, non-extended region that is "turn-like"
              φ ∈ [-90, -30] × ψ ∈ [-90, +60]
    Loop    : everything else
    """
    # Strict α-helix — checked first; helix is highest-priority
    if (-90.0 <= phi <= -45.0) and (-60.0 <= psi <= -10.0):
        return "H"
    # β-strand / extended
    if (phi <= -90.0) and (abs(psi) >= 90.0):
        return "E"
    # Turn-like (not helix, not extended by the above checks)
    if (-90.0 <= phi <= -30.0) and (-90.0 <= psi <= 60.0):
        return "T"
    return "L"


# ── Public functions ──────────────────────────────────────────────────────────

def extract_backbone_angles(
    pdb_path: str,
    chain:    str,
) -> Dict[int, Dict[str, Any]]:
    """
    Extract per-residue φ/ψ angles, secondary structure, and Cα coordinates.

    Migrated from ProlineBridge.extract_backbone_angles() — identical behaviour.

    Parameters
    ----------
    pdb_path : local PDB file path (as_posix() used internally)
    chain    : chain identifier, e.g. "A"

    Returns
    -------
    {resno: {phi, psi, ss, resname, aa, ca_coords}} or {} on any failure.

    Notes
    -----
    - φ/ψ in degrees; None at chain termini where PPBuilder cannot compute them.
    - ss : "H" (helix), "E" (sheet), "L" (loop/coil).
    - ca_coords : (x, y, z) tuple or None when no CA atom is present.
    """
    try:
        from Bio.PDB import PDBParser, PPBuilder  # type: ignore

        pdb_str = Path(pdb_path).as_posix()
        parser  = PDBParser(QUIET=True)
        structure = parser.get_structure("p", pdb_str)
        model     = structure[0]

        builder = PPBuilder()
        raw: Dict[int, Dict[str, Any]] = {}

        for chain_obj in model:
            if chain_obj.id != chain:
                continue
            for pp in builder.build_peptides(chain_obj):
                for residue, (phi_rad, psi_rad) in zip(pp, pp.get_phi_psi_list()):
                    seq_num = residue.get_id()[1]
                    resname = residue.get_resname().strip()
                    phi_deg = math.degrees(phi_rad) if phi_rad is not None else None
                    psi_deg = math.degrees(psi_rad) if psi_rad is not None else None
                    ca_coords: Optional[Tuple[float, float, float]] = None
                    if "CA" in residue:
                        ca = residue["CA"].get_vector()
                        ca_coords = (float(ca[0]), float(ca[1]), float(ca[2]))
                    raw[seq_num] = {
                        "phi":      phi_deg,
                        "psi":      psi_deg,
                        "resname":  resname,
                        "ca_coords": ca_coords,
                    }

        if not raw:
            return {}

        # ── Secondary structure: DSSP → φ/ψ fallback ─────────────────────────
        ss_map: Dict[int, str] = {}
        try:
            from Bio.PDB.DSSP import DSSP  # type: ignore
            dssp = DSSP(model, pdb_str)
            for key in dssp:
                res_id  = key[1][1]
                ss_code = dssp[key][2]
                if ss_code in ("H", "G", "I"):
                    ss_map[res_id] = "H"
                elif ss_code in ("E", "B"):
                    ss_map[res_id] = "E"
                else:
                    ss_map[res_id] = "L"
        except Exception:
            for seq_num, ang in raw.items():
                phi = ang.get("phi")
                psi = ang.get("psi")
                if phi is not None and psi is not None:
                    ss_map[seq_num] = _classify_ss_from_angles(phi, psi)
                else:
                    ss_map[seq_num] = "L"

        return {
            seq_num: {
                "phi":      ang["phi"],
                "psi":      ang["psi"],
                "ss":       ss_map.get(seq_num, "L"),
                "resname":  ang["resname"],
                "aa":       _THREE_TO_ONE.get(ang["resname"], "X"),
                "ca_coords": ang.get("ca_coords"),
            }
            for seq_num, ang in raw.items()
        }

    except Exception:
        return {}


def compute_sasa(
    pdb_path: str,
    chain:    Optional[str] = None,
) -> Dict[int, float]:
    """
    Compute per-residue SASA in Å².

    Tries freesasa first; falls back to BioPython ShrakeRupley.

    Parameters
    ----------
    pdb_path : local PDB file path
    chain    : restrict to this chain identifier; None = all chains

    Returns
    -------
    {resno: sasa_Å²} or {} on failure.
    """
    path = Path(pdb_path)
    if not path.exists():
        return {}

    # ── freesasa ──────────────────────────────────────────────────────────────
    try:
        import freesasa  # type: ignore
        structure = freesasa.Structure(str(path))
        result    = freesasa.calc(structure)
        area_map: Dict[int, float] = {}
        for i in range(structure.nAtoms()):
            if chain is not None:
                try:
                    if structure.chainLabel(i) != chain:
                        continue
                except Exception:
                    pass
            raw = structure.residueNumber(i)
            try:
                res_n = int(str(raw).strip())
            except (ValueError, AttributeError):
                continue
            area_map[res_n] = area_map.get(res_n, 0.0) + result.atomArea(i)
        if area_map:
            return area_map
    except Exception:
        pass

    # ── BioPython ShrakeRupley ────────────────────────────────────────────────
    try:
        from Bio.PDB import PDBParser  # type: ignore
        from Bio.PDB.SASA import ShrakeRupley  # type: ignore
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", str(path))
        sr = ShrakeRupley()
        sr.compute(struct, level="R")
        sasa_map: Dict[int, float] = {}
        for model in struct:
            for ch in model:
                if chain is not None and ch.id != chain:
                    continue
                for res in ch:
                    rn = res.id[1]
                    sasa_map[rn] = float(getattr(res, "sasa", 0.0))
        return sasa_map
    except Exception:
        return {}


def compute_projection_score(
    pdb_path: str,
    chain:    str,
) -> Dict[int, Dict[str, Any]]:
    """
    For each residue, compute the cosine of the angle between the
    Cα→Cβ vector and the outward surface normal.

    Surface normal approximation
    ----------------------------
    The outward normal for residue *r* is the normalised vector from the
    Cα centroid of the whole chain to residue *r*'s Cα.  This approximation
    is most accurate for globular, convex proteins (e.g. HIV protease).

    Glycine handling
    ----------------
    Gly has no Cβ.  The Cα→N vector is used as a proxy and the result is
    flagged with ``gly_proxy=True``.

    Returns
    -------
    {resno: {"projection_score": float, "gly_proxy": bool}}
      Values near +1.0  → side chain pointing directly outward (ideal for glycosylation).
      Values near  0.0  → side chain lying flat along the surface.
      Negative values   → side chain pointing inward (buried).

    Returns {} on any failure — callers must handle gracefully.
    """
    try:
        from Bio.PDB import PDBParser  # type: ignore

        pdb_str   = Path(pdb_path).as_posix()
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("p", pdb_str)
        model     = structure[0]
        chain_obj = model[chain]

        # ── Centroid of all Cα atoms in the chain ────────────────────────────
        ca_list = []
        for res in chain_obj:
            if "CA" in res:
                ca = res["CA"].get_vector()
                ca_list.append((float(ca[0]), float(ca[1]), float(ca[2])))

        if not ca_list:
            return {}

        n_ca = len(ca_list)
        cx = sum(p[0] for p in ca_list) / n_ca
        cy = sum(p[1] for p in ca_list) / n_ca
        cz = sum(p[2] for p in ca_list) / n_ca

        result: Dict[int, Dict[str, Any]] = {}

        for res in chain_obj:
            resno = res.id[1]
            if "CA" not in res:
                continue

            ca      = res["CA"].get_vector()
            cax     = float(ca[0])
            cay     = float(ca[1])
            caz     = float(ca[2])

            # Outward normal: (Cα − centroid) normalised
            nx, ny, nz = cax - cx, cay - cy, caz - cz
            n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            if n_len < 1e-6:
                continue   # residue exactly at centroid — skip
            nx /= n_len
            ny /= n_len
            nz /= n_len

            # Direction vector: Cα → Cβ (or Cα → N for Gly)
            gly_proxy = False
            resname   = res.get_resname().strip()
            if resname == "GLY" or "CB" not in res:
                if "N" not in res:
                    continue
                dir_vec   = res["N"].get_vector()
                gly_proxy = True
            else:
                dir_vec = res["CB"].get_vector()

            dx = float(dir_vec[0]) - cax
            dy = float(dir_vec[1]) - cay
            dz = float(dir_vec[2]) - caz
            d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d_len < 1e-6:
                continue
            dx /= d_len
            dy /= d_len
            dz /= d_len

            proj = dx * nx + dy * ny + dz * nz

            result[resno] = {
                "projection_score": round(proj, 4),
                "gly_proxy":        gly_proxy,
            }

        return result

    except Exception:
        return {}


def classify_sequon_geometry(
    backbone: Dict[int, Dict[str, Any]],
    pos_i:   int,
) -> str:
    """
    Classify the local sequon backbone geometry at positions i, i+1, i+2.

    Parameters
    ----------
    backbone : output of extract_backbone_angles() — {resno: {phi, psi, ...}}
    pos_i    : residue number of the Asn at the N-X-S/T sequon (position i)

    Returns
    -------
    "beta_turn"  — at least one position is in a turn-like Ramachandran region.
                   Glycans are naturally enriched at β-turns (geometry factor 1.4).
    "loop"       — all positions in loop/coil; no distinct turn signature
                   (geometry factor 1.2).
    "extended"   — two or more positions in extended strand region
                   (geometry factor 1.0).
    "helix"      — one or more positions in helical region; glycans at helices
                   are sterically problematic (geometry factor 0.5).
    "unknown"    — insufficient angle data for at least one of the three positions.

    Classification priority: helix > beta_turn > extended > loop.
    """
    codes = []
    for p in (pos_i, pos_i + 1, pos_i + 2):
        entry = backbone.get(p)
        if entry is None:
            return "unknown"
        phi = entry.get("phi")
        psi = entry.get("psi")
        if phi is None or psi is None:
            return "unknown"
        codes.append(_classify_single_pos(float(phi), float(psi)))

    if "H" in codes:
        return "helix"
    if "T" in codes:
        return "beta_turn"
    if codes.count("E") >= 2:
        return "extended"
    return "loop"
