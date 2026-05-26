"""
disulfide_bridge.py
-------------------
Interchain disulfide bond candidate prediction for protein complex engineering.

Pipeline
--------
1. Parse PDB  — extract Cβ / Cα coordinates for both chains
2. Geometry   — find Cβ-Cβ pairs within 4.5 Å; score distance + dihedral
3. ESM        — estimate evolutionary tolerance to Cys substitution at each
                position via ESM-2 masked-language model scores
4. DynaMut2   — score stability of X→C mutations (both positions scored)
5. Rank       — combined score: geometry×0.4 + ESM×0.3 + stability×0.3
6. Visualise  — ChimeraX commands to show top-N candidates as gold spheres

Output schema (each candidate dict)
------------------------------------
{
  "chain_a_residue":  int,
  "chain_b_residue":  int,
  "chain_a_aa":       str,   # one-letter code of wild-type AA in chain A
  "chain_b_aa":       str,
  "cb_distance":      float,          # Å
  "dihedral_angle":   float | None,   # degrees (Cα1-Cβ1-Cβ2-Cα2); None for Gly
  "distance_score":   float, # 0–1 Gaussian centred at 3.8 Å
  "dihedral_score":   float, # 0–1 Gaussian centred at ±90°
  "geometry_score":   float, # 0.5×dist_score + 0.5×dihedral_score
  "esm_tolerance_a":  float, # 1 – conservation(A) in [0,1]
  "esm_tolerance_b":  float,
  "esm_score":        float, # mean of tolerance A and B
  "ddg_a":            float, # ΔΔG kcal/mol for chain_a_aa → C
  "ddg_b":            float,
  "stability_score":  float, # 0–1 derived from mean ddG
  "combined_score":   float,
  "recommendation":   str,
}
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tool_router import ToolStepResult


# ── Constants ─────────────────────────────────────────────────────────────────

# Ideal disulfide Cβ-Cβ distance range / centre
_CB_DIST_MIN    = 3.5    # Å  — lower bound for filter
_CB_DIST_MAX    = 4.5    # Å  — upper bound for filter (also search radius)
_CB_DIST_IDEAL  = 3.8    # Å  — Gaussian centre
_CB_DIST_SIGMA  = 0.4    # Å  — Gaussian sigma

# Ideal Cα-Cβ-Cβ-Cα dihedral (absolute value)
_DIHEDRAL_IDEAL  = 90.0   # degrees
_DIHEDRAL_CUTOFF = 135.0  # degrees — deviation beyond which score is hard-zero

# Hard filters
_MIN_ESM_TOLERANCE    = 0.3    # positions below this are excluded
_MAX_DESTABILIZING_DDG = 1.0   # kcal/mol — exclude if either mutation exceeds this


# ── Safe progress print ────────────────────────────────────────────────────────

def _pprint(msg: str) -> None:
    """Print with flush; falls back to ASCII for narrow terminal encodings."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ── Three-letter → one-letter amino acid map ──────────────────────────────────

_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "CSE": "C", "HYP": "P",
}


def _three_to_one(resname: str) -> str:
    return _THREE_TO_ONE.get(resname.strip().upper(), "X")


# ── PDB atom parsing ──────────────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path: str) -> Dict[str, Dict[int, Dict]]:
    """
    Parse CA and CB atom coordinates from a PDB file.

    Returns
    -------
    {chain_id: {resno: {"resname": str, "CA": (x,y,z), "CB": (x,y,z)}}}

    Glycine has no CB; its entry will have CA only.
    Only standard ATOM records are considered (not HETATM).
    """
    atoms: Dict[str, Dict[int, Dict]] = {}

    try:
        with open(pdb_path, "r", errors="replace") as fh:
            for line in fh:
                if line[:6].strip() not in ("ATOM",):
                    continue

                atom_name = line[12:16].strip()
                if atom_name not in ("CA", "CB"):
                    continue

                chain_id = line[21].strip() if len(line) > 21 else ""
                if not chain_id:
                    continue

                try:
                    resno = int(line[22:26].strip())
                except ValueError:
                    continue

                resname = line[17:20].strip()

                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue

                if chain_id not in atoms:
                    atoms[chain_id] = {}
                if resno not in atoms[chain_id]:
                    atoms[chain_id][resno] = {"resname": resname}

                # Only store CA / CB; don't overwrite with alternate conformations
                if atom_name not in atoms[chain_id][resno]:
                    atoms[chain_id][resno][atom_name] = (x, y, z)

    except OSError:
        pass

    return atoms


def extract_sequence(
    atoms: Dict[str, Dict[int, Dict]],
    chain_id: str,
) -> Tuple[str, Dict[int, int]]:
    """
    Build a 1-letter amino acid sequence and a mapping from PDB residue numbers
    to 1-based sequence indices.

    Returns
    -------
    (sequence_str, {pdb_resno: 1_based_index})
    """
    chain = atoms.get(chain_id, {})
    sorted_resnos = sorted(chain.keys())
    sequence = ""
    mapping: Dict[int, int] = {}
    idx = 1
    for resno in sorted_resnos:
        aa = _three_to_one(chain[resno].get("resname", "UNK"))
        if aa != "X":
            sequence += aa
            mapping[resno] = idx
            idx += 1
    return sequence, mapping


# ── Geometry calculations ─────────────────────────────────────────────────────

def calc_distance(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calc_dihedral(
    a1: Tuple[float, float, float],
    b1: Tuple[float, float, float],
    b2: Tuple[float, float, float],
    a2: Tuple[float, float, float],
) -> float:
    """
    Dihedral angle a1-b1-b2-a2 in degrees, range [-180, 180].

    Uses the atan2 formula:
        v1 = b1 - a1,  v2 = b2 - b1,  v3 = a2 - b2
        n1 = v1 x v2,  n2 = v2 x v3
        angle = atan2( (v2/|v2| x n1) . n2,  n1 . n2 )

    Returns 0.0 for degenerate geometries where any three atoms
    are collinear (cross-product is near-zero).  Callers that need to
    distinguish a genuine 0 degree dihedral from a degenerate case
    should check whether the residue has a real CB atom beforehand.
    """
    def _sub(u, v):
        return (u[0]-v[0], u[1]-v[1], u[2]-v[2])
    def _dot(u, v):
        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    def _cross(u, v):
        return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])

    v1 = _sub(b1, a1)   # CA1 → CB1
    v2 = _sub(b2, b1)   # CB1 → CB2
    v3 = _sub(a2, b2)   # CB2 → CA2

    n1 = _cross(v1, v2)
    n2 = _cross(v2, v3)

    if _dot(n1, n1) < 1e-10 or _dot(n2, n2) < 1e-10:
        return 0.0      # degenerate: atoms are collinear

    v2_mag = math.sqrt(_dot(v2, v2))
    if v2_mag < 1e-10:
        return 0.0

    inv_v2 = 1.0 / v2_mag
    v2_hat = (v2[0]*inv_v2, v2[1]*inv_v2, v2[2]*inv_v2)
    m1 = _cross(v2_hat, n1)     # (v2/|v2|) x n1

    x = _dot(n1, n2)
    y = _dot(m1, n2)
    return math.degrees(math.atan2(y, x))


def geometry_score(
    cb_distance:  float,
    dihedral_deg: Optional[float],
) -> Tuple[float, float, float]:
    """
    Score distance and dihedral independently; return (dist_score, dihed_score, geo_score).
    All scores are in [0, 1]; 1.0 = ideal disulfide geometry.

    Distance score
    --------------
    Gaussian centred at 3.8 Å, σ = 0.4 Å.

    Dihedral score
    --------------
    Raised-cosine centred at ±90°:

        dev = min(|theta - 90|, |theta + 90|)   in [0, 90]
        score = (1 + cos(dev * pi / 180)) / 2

    Property table:
        ±90°   → dev = 0   → score = 1.0  (ideal disulfide)
        0°/±180° → dev = 90  → score = 0.5  (eclipsed / anti — neutral)
        dev > 135° → score = 0.0  (hard floor; unreachable for [-180,180])

    If dihedral_deg is None (residue is Glycine — no real Cβ — so the
    Cα-Cβ-Cβ-Cα dihedral is undefined), dihed_score = 0.5 (neutral).

    geometry_score = 0.5 × dist_score + 0.5 × dihed_score
    """
    dist_score = math.exp(
        -((cb_distance - _CB_DIST_IDEAL) ** 2) / (2 * _CB_DIST_SIGMA ** 2)
    )

    if dihedral_deg is None:
        # Gly pseudo-CB: dihedral undefined; use neutral score
        dihed_score = 0.5
    else:
        dev = min(abs(dihedral_deg - _DIHEDRAL_IDEAL),
                  abs(dihedral_deg + _DIHEDRAL_IDEAL))
        if dev >= _DIHEDRAL_CUTOFF:
            dihed_score = 0.0
        else:
            # raised cosine: 1.0 at dev=0 (±90°), 0.5 at dev=90 (0°/180°)
            dihed_score = (1.0 + math.cos(dev * math.pi / 180.0)) / 2.0

    geo = 0.5 * dist_score + 0.5 * dihed_score
    return round(dist_score, 4), round(dihed_score, 4), round(geo, 4)


# ── Candidate finder ──────────────────────────────────────────────────────────

def find_cb_pairs(
    atoms:   Dict[str, Dict[int, Dict]],
    chain_a: str,
    chain_b: str,
    max_dist: float = _CB_DIST_MAX,
) -> List[Dict[str, Any]]:
    """
    Find all Cβ-Cβ pairs across chain_a × chain_b within max_dist Å.

    Excludes positions that are already Cys.
    Glycine uses Cα as its pseudo-Cβ position (standard disulfide design
    practice: Gly→Cys mutations are evaluated but scored by Cα distance).

    Returns a list of candidate dicts with geometry pre-scored.
    """
    data_a = atoms.get(chain_a, {})
    data_b = atoms.get(chain_b, {})
    candidates: List[Dict[str, Any]] = []

    for resno_a, info_a in data_a.items():
        aa_a = _three_to_one(info_a.get("resname", "UNK"))
        if aa_a in ("X", "C"):   # skip unknown or already-Cys
            continue
        cb_a = info_a.get("CB") or info_a.get("CA")
        ca_a = info_a.get("CA")
        has_real_cb_a = "CB" in info_a   # False for Gly
        if cb_a is None:
            continue

        for resno_b, info_b in data_b.items():
            aa_b = _three_to_one(info_b.get("resname", "UNK"))
            if aa_b in ("X", "C"):
                continue
            cb_b = info_b.get("CB") or info_b.get("CA")
            ca_b = info_b.get("CA")
            has_real_cb_b = "CB" in info_b   # False for Gly
            if cb_b is None:
                continue

            dist = calc_distance(cb_a, cb_b)
            if dist > max_dist:
                continue

            # Dihedral: None when either residue is Gly (no real Cβ).
            # Using CA as pseudo-Cβ makes the cross-product degenerate
            # (v1 = Cβ-Cα = 0), so we explicitly skip it and carry
            # None through to geometry_score() → neutral 0.5 score.
            dihedral: Optional[float] = None
            if ca_a and ca_b and has_real_cb_a and has_real_cb_b:
                try:
                    dihedral = calc_dihedral(ca_a, cb_a, cb_b, ca_b)
                except Exception:
                    dihedral = None

            dist_sc, dihed_sc, geo_sc = geometry_score(dist, dihedral)

            candidates.append({
                "chain_a_residue": resno_a,
                "chain_b_residue": resno_b,
                "chain_a_aa":      aa_a,
                "chain_b_aa":      aa_b,
                "cb_distance":     round(dist, 3),
                "dihedral_angle":  round(dihedral, 1) if dihedral is not None else None,
                "distance_score":  dist_sc,
                "dihedral_score":  dihed_sc,
                "geometry_score":  geo_sc,
                # Filled later
                "esm_tolerance_a": None,
                "esm_tolerance_b": None,
                "esm_score":       None,
                "ddg_a":           None,
                "ddg_b":           None,
                "stability_score": None,
                "combined_score":  None,
                "recommendation":  None,
            })

    return candidates


# ── Combined scoring ──────────────────────────────────────────────────────────

def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute combined_score = geometry_score×0.4 + esm_score×0.3 + stability_score×0.3
    and sort descending.
    """
    for c in candidates:
        geo_sc   = float(c.get("geometry_score",   0.0) or 0.0)
        esm_a    = float(c.get("esm_tolerance_a",  0.5) or 0.5)
        esm_b    = float(c.get("esm_tolerance_b",  0.5) or 0.5)
        esm_sc   = (esm_a + esm_b) / 2.0
        ddg_a    = float(c.get("ddg_a",            0.0) or 0.0)
        ddg_b    = float(c.get("ddg_b",            0.0) or 0.0)
        ddg_mean = (ddg_a + ddg_b) / 2.0
        # stability_score: 1.0 when ddg_mean=0, lower for destabilising
        # Clamp to [0, 1]; ddg_mean = -1 → 1.0, ddg_mean = +1 → 0.5
        stab_sc  = max(0.0, min(1.0, (2.0 - ddg_mean) / 2.0))

        combined = geo_sc * 0.4 + esm_sc * 0.3 + stab_sc * 0.3

        c["esm_score"]       = round(esm_sc,   4)
        c["stability_score"] = round(stab_sc,  4)
        c["combined_score"]  = round(combined, 4)
        c["recommendation"]  = _build_recommendation(c)

    return sorted(candidates, key=lambda x: x.get("combined_score") or 0.0, reverse=True)


def _build_recommendation(c: Dict[str, Any]) -> str:
    """Human-readable recommendation string for a disulfide candidate."""
    geo   = float(c.get("geometry_score",  0.0) or 0.0)
    esm   = float(c.get("esm_score",       0.5) or 0.5)
    ddg_a = float(c.get("ddg_a",           0.0) or 0.0)
    ddg_b = float(c.get("ddg_b",           0.0) or 0.0)
    score = float(c.get("combined_score",  0.0) or 0.0)

    parts = []

    if geo >= 0.8:
        parts.append("excellent geometry")
    elif geo >= 0.6:
        parts.append("good geometry")
    else:
        parts.append("marginal geometry")

    if esm >= 0.7:
        parts.append("well-tolerated by evolution")
    elif esm >= 0.5:
        parts.append("tolerated by evolution")
    else:
        parts.append("marginally tolerated by evolution")

    ddg_max = max(ddg_a, ddg_b)
    if ddg_max < 0:
        parts.append("stabilising mutations")
    elif ddg_max < 0.5:
        parts.append("approximately neutral stability")
    elif ddg_max < 1.0:
        parts.append("mildly destabilising mutations")
    else:
        parts.append("destabilising mutations")

    prefix = (
        "Strong candidate"  if score >= 0.7 else
        "Moderate candidate" if score >= 0.5 else
        "Weak candidate"
    )
    return f"{prefix} — {', '.join(parts)}"


# ── Visualization ─────────────────────────────────────────────────────────────

def generate_chimerax_commands(
    candidates: List[Dict[str, Any]],
    model_id:   str = "1",
    chain_a:    str = "A",
    chain_b:    str = "B",
    top_n:      int = 3,
) -> Tuple[List[str], List[str]]:
    """
    ChimeraX commands to visualise the top-N disulfide candidates.

    Each candidate is shown as:
    - Gold (or silver/light-blue) spheres at both Cβ atoms
    - A distance annotation between the Cβ atoms
    - Text labels identifying the pair and score

    Returns (commands, explanations).
    """
    if not candidates:
        return [], []

    colours = ["gold", "silver", "light blue", "light green", "orange"]

    cmds: List[str] = [
        "cartoon",
        "color white",
    ]
    exps: List[str] = [
        "Show cartoon representation",
        "Reset to white before applying candidate colours",
    ]

    for i, cand in enumerate(candidates[:top_n]):
        resno_a = cand["chain_a_residue"]
        resno_b = cand["chain_b_residue"]
        aa_a    = cand.get("chain_a_aa", "?")
        aa_b    = cand.get("chain_b_aa", "?")
        score   = cand.get("combined_score") or 0.0
        dist    = cand.get("cb_distance", 0.0)
        colour  = colours[i % len(colours)]

        spec_a = f"#{model_id}/{chain_a}:{resno_a}"
        spec_b = f"#{model_id}/{chain_b}:{resno_b}"

        cmds += [
            f"show {spec_a} atoms",
            f"style {spec_a}@CB sphere",
            f"color {spec_a} {colour}",
            f"show {spec_b} atoms",
            f"style {spec_b}@CB sphere",
            f"color {spec_b} {colour}",
            f"distance {spec_a}@CB {spec_b}@CB",
            f"label {spec_a} text \"SS#{i+1}: {chain_a}{resno_a}{aa_a}/C\" height 2",
            f"label {spec_b} text \"{chain_b}{resno_b}{aa_b}/C score={score:.2f}\" height 2",
        ]
        exps += [
            f"Show atoms of {chain_a}{resno_a} ({aa_a})",
            f"Style Cb of {chain_a}{resno_a} ({aa_a}) as sphere",
            f"Color {chain_a}{resno_a} {colour}",
            f"Show atoms of {chain_b}{resno_b} ({aa_b})",
            f"Style Cb of {chain_b}{resno_b} ({aa_b}) as sphere",
            f"Color {chain_b}{resno_b} {colour}",
            f"Measure Cb-Cb distance ({dist:.1f} Angstrom)",
            f"Label disulfide candidate #{i+1} at {chain_a}{resno_a}",
            f"Label {chain_b}{resno_b} with combined score={score:.2f}",
        ]

    cmds.append("view")
    exps.append("Fit structure in view")

    return cmds, exps


# ═══════════════════════════════════════════════════════════════════════════════
# Public bridge class
# ═══════════════════════════════════════════════════════════════════════════════

class DisulfideBridge:
    """
    Predicts viable interchain disulfide bond positions.

    Usage::

        bridge = DisulfideBridge()
        result = bridge.analyze(
            pdb_path = "cache/1HSG.pdb",
            chain_a  = "A",
            chain_b  = "B",
            session  = session_state,
            model_id = "1",
        )
        if result.success:
            for cand in result.data["candidates"]:
                print(cand["combined_score"], cand["recommendation"])
    """

    def __init__(self, chimerax_bridge: Any = None):
        self._cx_bridge = chimerax_bridge

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        pdb_path:              str,
        chain_a:               str,
        chain_b:               str,
        session:               Any = None,
        model_id:              str = "1",
        binding_site_residues: Optional[List[int]] = None,
        progress_callback:     Optional[Callable[[str], None]] = None,
    ) -> ToolStepResult:
        """
        Full disulfide candidate pipeline.

        Parameters
        ----------
        pdb_path               : local PDB file path
        chain_a / chain_b      : chains to search
        session                : SessionState (for ESM caching)
        model_id               : ChimeraX model ID used in viz commands
        binding_site_residues  : residue numbers to exclude from chain A
        progress_callback      : callable(str) for real-time progress
        """
        def _prog(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                _pprint(msg)

        if not Path(pdb_path).is_file():
            return ToolStepResult(
                tool="disulfide", success=False,
                error=f"PDB file not found: {pdb_path}",
            )

        _prog(f"🔗 [Disulfide] Parsing {Path(pdb_path).name} for chains {chain_a}/{chain_b}...")

        # ── 1. Parse PDB ───────────────────────────────────────────────────────
        atoms = parse_pdb_atoms(pdb_path)

        if chain_a not in atoms:
            return ToolStepResult(
                tool="disulfide", success=False,
                error=(
                    f"Chain {chain_a} not found in {pdb_path}. "
                    f"Available chains: {sorted(atoms.keys())}"
                ),
            )
        if chain_b not in atoms:
            return ToolStepResult(
                tool="disulfide", success=False,
                error=(
                    f"Chain {chain_b} not found in {pdb_path}. "
                    f"Available chains: {sorted(atoms.keys())}"
                ),
            )

        # ── 2. Find Cβ-Cβ pairs ────────────────────────────────────────────────
        _prog("🔗 [Disulfide] Searching for Cβ-Cβ pairs within 4.5 Å...")
        candidates = find_cb_pairs(atoms, chain_a, chain_b)
        _prog(f"🔗 [Disulfide] {len(candidates)} raw Cβ-Cβ pair(s) found.")

        if not candidates:
            return ToolStepResult(
                tool="disulfide", success=True,
                data={
                    "candidates": [], "count": 0,
                    "chain_a": chain_a, "chain_b": chain_b,
                },
                summary=(
                    f"Disulfide analysis {chain_a}/{chain_b}: "
                    "no residue pairs within 4.5 Å Cβ-Cβ distance."
                ),
            )

        # ── 3. Remove binding-site residues ────────────────────────────────────
        if binding_site_residues:
            bs = set(binding_site_residues)
            before = len(candidates)
            candidates = [
                c for c in candidates
                if c["chain_a_residue"] not in bs
                and c["chain_b_residue"] not in bs
            ]
            removed = before - len(candidates)
            if removed:
                _prog(
                    f"🔗 [Disulfide] {removed} pair(s) excluded (binding site). "
                    f"{len(candidates)} remaining."
                )

        if not candidates:
            return ToolStepResult(
                tool="disulfide", success=True,
                data={
                    "candidates": [], "count": 0,
                    "chain_a": chain_a, "chain_b": chain_b,
                },
                summary=(
                    f"Disulfide analysis {chain_a}/{chain_b}: "
                    "all candidates excluded after binding-site filter."
                ),
            )

        # ── 4. ESM tolerance scoring ───────────────────────────────────────────
        _prog(
            f"🔗 [Disulfide] Scoring evolutionary tolerance with ESM-2 "
            f"for chains {chain_a}/{chain_b}..."
        )
        candidates = self._score_esm(candidates, atoms, chain_a, chain_b)

        before = len(candidates)
        candidates = [
            c for c in candidates
            if (c.get("esm_tolerance_a") or 0.0) >= _MIN_ESM_TOLERANCE
            and (c.get("esm_tolerance_b") or 0.0) >= _MIN_ESM_TOLERANCE
        ]
        removed = before - len(candidates)
        if removed:
            _prog(
                f"🔗 [Disulfide] {removed} pair(s) excluded by ESM tolerance filter "
                f"(threshold {_MIN_ESM_TOLERANCE}). {len(candidates)} remaining."
            )

        if not candidates:
            return ToolStepResult(
                tool="disulfide", success=True,
                data={
                    "candidates": [], "count": 0,
                    "chain_a": chain_a, "chain_b": chain_b,
                },
                summary=(
                    f"Disulfide analysis {chain_a}/{chain_b}: "
                    "no candidates passed the ESM tolerance filter "
                    f"(minimum {_MIN_ESM_TOLERANCE})."
                ),
            )

        # ── 5. DynaMut2 stability scoring ──────────────────────────────────────
        n_muts_needed = (
            len({c["chain_a_residue"] for c in candidates})
            + len({c["chain_b_residue"] for c in candidates})
        )
        _prog(
            f"🔗 [Disulfide] Scoring X→C stability with DynaMut2 "
            f"({len(candidates)} pair(s), {n_muts_needed} unique mutations)..."
        )
        candidates = self._score_stability(
            candidates, pdb_path, chain_a, chain_b, _prog
        )

        before = len(candidates)
        candidates = [
            c for c in candidates
            if (c.get("ddg_a") or 0.0) <= _MAX_DESTABILIZING_DDG
            and (c.get("ddg_b") or 0.0) <= _MAX_DESTABILIZING_DDG
        ]
        removed = before - len(candidates)
        if removed:
            _prog(
                f"🔗 [Disulfide] {removed} pair(s) excluded (ddG > {_MAX_DESTABILIZING_DDG} kcal/mol). "
                f"{len(candidates)} remaining."
            )

        # ── 6. Rank ────────────────────────────────────────────────────────────
        candidates = rank_candidates(candidates)

        # ── 7. Visualization ───────────────────────────────────────────────────
        top_n = min(3, len(candidates))
        viz_cmds, viz_exps = generate_chimerax_commands(
            candidates[:top_n],
            model_id = model_id,
            chain_a  = chain_a,
            chain_b  = chain_b,
        )

        summary = (
            f"Disulfide analysis {chain_a}/{chain_b}: {len(candidates)} candidate(s)."
        )
        if candidates:
            top = candidates[0]
            summary += (
                f" Top: {chain_a}{top['chain_a_residue']}{top['chain_a_aa']}→C / "
                f"{chain_b}{top['chain_b_residue']}{top['chain_b_aa']}→C "
                f"(score={top['combined_score']:.2f}, "
                f"Cβ-Cβ={top['cb_distance']:.1f} Å, "
                f"dihedral={top['dihedral_angle']:.1f}°)"
            )
        _prog(f"🔗 {summary}")

        # ── Store in session state if provided ────────────────────────────────
        if session is not None:
            try:
                session.set_disulfide_candidates(
                    model_id, chain_a, chain_b, candidates
                )
            except AttributeError:
                pass  # session_state may be an older version

        return ToolStepResult(
            tool             = "disulfide",
            success          = True,
            data             = {
                "candidates": candidates,
                "count":      len(candidates),
                "chain_a":    chain_a,
                "chain_b":    chain_b,
                "top":        candidates[0] if candidates else None,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
        )

    # ── ESM tolerance scoring ─────────────────────────────────────────────────

    def _score_esm(
        self,
        candidates: List[Dict[str, Any]],
        atoms:      Dict[str, Dict[int, Dict]],
        chain_a:    str,
        chain_b:    str,
    ) -> List[Dict[str, Any]]:
        """
        Assign esm_tolerance_a and esm_tolerance_b to each candidate.

        ESM conservation (0–1, 1=conserved).
        ESM tolerance = 1 - conservation.  Variable positions tolerate mutation.
        Falls back to 0.5 (neutral) if ESM is unavailable.
        """
        # Extract sequences and residue→sequence-index mappings
        seq_a, map_a = extract_sequence(atoms, chain_a)
        seq_b, map_b = extract_sequence(atoms, chain_b)

        esm_a = self._run_esm(seq_a)   # {int_pos: conservation_score}
        esm_b = self._run_esm(seq_b)

        for c in candidates:
            idx_a = map_a.get(c["chain_a_residue"])
            idx_b = map_b.get(c["chain_b_residue"])

            c["esm_tolerance_a"] = round(
                1.0 - esm_a.get(idx_a, 0.5) if idx_a is not None else 0.5, 4
            )
            c["esm_tolerance_b"] = round(
                1.0 - esm_b.get(idx_b, 0.5) if idx_b is not None else 0.5, 4
            )

        return candidates

    def _run_esm(self, sequence: str) -> Dict[int, float]:
        """
        Run EsmBridge on *sequence*; return {1_based_pos: conservation_score}.
        Returns {} (→ fallback to 0.5) on any error.
        """
        if not sequence:
            return {}
        try:
            from esm_bridge import EsmBridge
            bridge = EsmBridge()
            result = bridge.analyze(sequence, model_id="disulfide_tmp", session=None)
            if result.success:
                # Keys are integers (start_resno + i, default start_resno=1)
                return {k: v for k, v in result.data.get("conservation", {}).items()}
        except Exception:
            pass
        return {}

    # ── DynaMut2 stability scoring ────────────────────────────────────────────

    def _score_stability(
        self,
        candidates:        List[Dict[str, Any]],
        pdb_path:          str,
        chain_a:           str,
        chain_b:           str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Assess stability of X→C mutations at each candidate position.

        Uses RosettaBridge (DynaMut2 by default) to batch-score unique mutations
        for each chain.  Falls back to 0.0 (neutral) on any error.
        """
        def _prog(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        try:
            from rosetta_bridge import RosettaBridge
            bridge = RosettaBridge()
        except ImportError:
            _prog("  [Disulfide] RosettaBridge unavailable — stability scores set to 0")
            for c in candidates:
                c["ddg_a"] = 0.0
                c["ddg_b"] = 0.0
            return candidates

        # ── Collect unique mutations per chain ────────────────────────────────
        muts_a: Dict[int, Dict[str, Any]] = {}
        muts_b: Dict[int, Dict[str, Any]] = {}

        for c in candidates:
            ra = c["chain_a_residue"]
            rb = c["chain_b_residue"]
            if ra not in muts_a:
                muts_a[ra] = {"chain": chain_a, "position": ra,
                              "from_aa": c["chain_a_aa"], "to_aa": "C"}
            if rb not in muts_b:
                muts_b[rb] = {"chain": chain_b, "position": rb,
                              "from_aa": c["chain_b_aa"], "to_aa": "C"}

        ddg_by_resno_a: Dict[int, float] = {}
        ddg_by_resno_b: Dict[int, float] = {}

        def _extract_ddg(result: ToolStepResult) -> Dict[int, float]:
            """Parse ddg_scores dict {mutkey: ddg} → {resno: ddg}."""
            out: Dict[int, float] = {}
            for key, ddg in result.data.get("ddg_scores", {}).items():
                m = re.match(r"[A-Z](\d+)[A-Z]", key)
                if m:
                    out[int(m.group(1))] = float(ddg)
            return out

        # Score chain A mutations
        if muts_a:
            r = bridge.analyze(
                pdb_path          = pdb_path,
                mutations         = list(muts_a.values()),
                progress_callback = lambda msg: _prog(f"  {msg}"),
            )
            if r.success:
                ddg_by_resno_a = _extract_ddg(r)

        # Score chain B mutations
        if muts_b:
            r = bridge.analyze(
                pdb_path          = pdb_path,
                mutations         = list(muts_b.values()),
                progress_callback = lambda msg: _prog(f"  {msg}"),
            )
            if r.success:
                ddg_by_resno_b = _extract_ddg(r)

        # Apply to candidates
        for c in candidates:
            c["ddg_a"] = ddg_by_resno_a.get(c["chain_a_residue"], 0.0)
            c["ddg_b"] = ddg_by_resno_b.get(c["chain_b_residue"], 0.0)

        return candidates

    def __repr__(self) -> str:
        return f"<DisulfideBridge cx_bridge={'set' if self._cx_bridge else 'None'}>"
