"""
disulfide_geometry.py
---------------------
Shared geometry core for the fold-based cysteine/disulfide analysis suite (Modes A discovery,
B geometry-readout, C declared-constraint — and later D engineering-scan). Plain-Python, NO
new dependency: the repo parses mmCIF via the `_atom_site` loop (see `boltz_bridge`), not gemmi,
so this module imports only stdlib and is loadable without the tool_router/bridge chain.

ONE helper, two callers: the pure primitives ``calc_distance`` / ``calc_dihedral`` are LIFTED
here (the existing interchain ``disulfide_bridge`` re-imports them, so its callers/tests keep
working) — both the existing tool and the new modes call a single source rather than reaching
into each other's internals. The SG-based disulfide geometry (SG–SG, Cβ–Cβ, Cα–Cα, and
χSS = Cβ–Sγ–Sγ–Cβ — the REAL disulfide dihedral) + canonical windows are new here.

Canonical disulfide geometry — the windows the readouts MEASURE against (measured, never
promised). Sources: standard disulfide stereochemistry (Cβ–Sγ–Sγ–Cβ ≈ ±90°; S–S ≈ 2.05 Å):
  SG–SG : 2.05 Å ideal   (bonding window 1.8–2.5 Å)
  Cβ–Cβ : 3.8 Å ideal    (window 3.0–4.5 Å)
  Cα–Cα : ~5.5 Å typical (window 4.5–7.5 Å)
  χSS   : |χ| ≈ 90° ideal (window 60–120°); the SIGN is the disulfide handedness.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


# ── primitives (LIFTED from disulfide_bridge; re-imported there for back-compat) ──────────

def calc_distance(p1: Vec3, p2: Vec3) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calc_dihedral(a1: Vec3, b1: Vec3, b2: Vec3, a2: Vec3) -> float:
    """Dihedral angle a1-b1-b2-a2 in degrees, range [-180, 180] (atan2 form). Returns 0.0 for
    a degenerate (collinear) geometry. Lifted verbatim from disulfide_bridge so there is ONE
    implementation; χSS reuses it with (Cβ, Sγ, Sγ, Cβ)."""
    def _sub(u, v): return (u[0]-v[0], u[1]-v[1], u[2]-v[2])
    def _dot(u, v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    def _cross(u, v):
        return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])

    v1 = _sub(b1, a1)
    v2 = _sub(b2, b1)
    v3 = _sub(a2, b2)
    n1 = _cross(v1, v2)
    n2 = _cross(v2, v3)
    if _dot(n1, n1) < 1e-10 or _dot(n2, n2) < 1e-10:
        return 0.0
    v2_mag = math.sqrt(_dot(v2, v2))
    if v2_mag < 1e-10:
        return 0.0
    inv = 1.0 / v2_mag
    v2_hat = (v2[0]*inv, v2[1]*inv, v2[2]*inv)
    m1 = _cross(v2_hat, n1)
    return math.degrees(math.atan2(_dot(m1, n2), _dot(n1, n2)))


# ── canonical disulfide windows ───────────────────────────────────────────────────────────

SG_SG_IDEAL, SG_SG_MIN, SG_SG_MAX = 2.05, 1.8, 2.5
CB_CB_IDEAL, CB_CB_MIN, CB_CB_MAX = 3.8, 3.0, 4.5
CA_CA_IDEAL, CA_CA_MIN, CA_CA_MAX = 5.5, 4.5, 7.5
CHI_SS_IDEAL, CHI_SS_MIN, CHI_SS_MAX = 90.0, 60.0, 120.0


def chi_ss(cb1: Vec3, sg1: Vec3, sg2: Vec3, cb2: Vec3) -> float:
    """The disulfide dihedral χSS = Cβ–Sγ–Sγ–Cβ (degrees, [-180,180]); |χ| ≈ 90° is ideal,
    the sign is the bond's handedness. Just ``calc_dihedral`` with the four disulfide atoms."""
    return calc_dihedral(cb1, sg1, sg2, cb2)


def _in_window(val: Optional[float], lo: float, hi: float) -> Optional[bool]:
    """True/False if *val* is within [lo, hi]; None when *val* is None (atom absent)."""
    if val is None:
        return None
    return lo <= val <= hi


# ── mmCIF cysteine-atom parser (the `_atom_site` loop; SG/CB/CA per CYS) ───────────────────

def parse_cys_atoms(cif_path: str) -> Dict[str, Dict[int, Dict[str, Vec3]]]:
    """Parse CYS SG/CB/CA coordinates from a fold's mmCIF, in ONE pass over the `_atom_site`
    loop (the repo's parsing convention — robust column lookup from the loop header; mmCIF
    column order is not fixed). Returns ``{auth_asym_id: {auth_seq_id: {"CA":xyz,"CB":xyz,
    "SG":xyz}}}`` for CYS residues only. A residue missing an atom simply lacks that key."""
    cols: List[str] = []
    in_loop = False
    out: Dict[str, Dict[int, Dict[str, Vec3]]] = {}
    try:
        with open(cif_path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("_atom_site."):
                    cols.append(s)
                    in_loop = True
                    continue
                if in_loop and s.startswith(("ATOM", "HETATM")):
                    parts = s.split()
                    if len(parts) < len(cols):
                        continue
                    col = {c.split(".")[1]: parts[i] for i, c in enumerate(cols)}
                    if (col.get("label_comp_id") or "").upper() != "CYS":
                        continue
                    atom = col.get("label_atom_id")
                    if atom not in ("CA", "CB", "SG"):
                        continue
                    ch = col.get("auth_asym_id") or col.get("label_asym_id")
                    rid = col.get("auth_seq_id") or col.get("label_seq_id")
                    if ch is None or rid is None:
                        continue
                    try:
                        xyz = (float(col["Cartn_x"]), float(col["Cartn_y"]), float(col["Cartn_z"]))
                        rid_i = int(rid)
                    except (KeyError, TypeError, ValueError):
                        continue
                    res = out.setdefault(ch, {}).setdefault(rid_i, {})
                    res.setdefault(atom, xyz)            # don't overwrite altlocs
                elif in_loop and s and not s.startswith(("ATOM", "HETATM", "_atom_site.")):
                    if out:
                        break                            # left the atom_site loop
    except OSError:
        return {}
    return out


# ── pair geometry + window membership (the shared measurement Modes A/B/C report) ─────────

def resnum_to_chain_index(ordered_resnums: List[int], resnum: int) -> Optional[int]:
    """1-based residue INDEX of *resnum* within a chain's ordered author resnums. Boltz `bond`
    constraints want a 1-based index, but the construct/variant carries AUTHOR resnums that need
    NOT start at 1 (crystal-seeded designs, custom numbering) — so this conversion is the real
    correctness hinge: an off-by-mapping silently points the bond at the wrong residues and the
    fold 'succeeds' looking wrong. None if *resnum* is not in the chain. PURE."""
    try:
        return list(ordered_resnums).index(int(resnum)) + 1
    except (ValueError, TypeError):
        return None


def bond_constraint(chain_id: str, idx_a: int, idx_b: int, atom: str = "SG") -> Dict[str, object]:
    """A Boltz `bond` constraint entry between two residues' *atom* (default SG): the
    ``{atom1:[chain,idx,atom], atom2:[chain,idx,atom]}`` shape `_build_yaml` emits. Indices are
    1-based chain indices (see `resnum_to_chain_index`)."""
    return {"atom1": [chain_id, int(idx_a), atom], "atom2": [chain_id, int(idx_b), atom]}


def pair_geometry(res_a: Dict[str, Vec3], res_b: Dict[str, Vec3]) -> Dict[str, object]:
    """Measured disulfide geometry between two CYS residues (each ``{"CA","CB","SG"}``): the
    SG–SG / Cβ–Cβ / Cα–Cα distances + χSS, each with its canonical-window membership, plus a
    single ``bonding_compatible`` (SG–SG within the bonding window — the core proximity signal
    Mode A tallies) and ``all_windows`` (every measured axis within window — the strict read).
    A missing atom yields None for the affected metric (never a fabricated 0). PURE."""
    ca1, cb1, sg1 = res_a.get("CA"), res_a.get("CB"), res_a.get("SG")
    ca2, cb2, sg2 = res_b.get("CA"), res_b.get("CB"), res_b.get("SG")
    sg = calc_distance(sg1, sg2) if sg1 and sg2 else None
    cb = calc_distance(cb1, cb2) if cb1 and cb2 else None
    ca = calc_distance(ca1, ca2) if ca1 and ca2 else None
    chi = chi_ss(cb1, sg1, sg2, cb2) if (cb1 and sg1 and sg2 and cb2) else None
    chi_ok = (None if chi is None else (CHI_SS_MIN <= abs(chi) <= CHI_SS_MAX))
    w = {
        "sg_sg":  _in_window(sg, SG_SG_MIN, SG_SG_MAX),
        "cb_cb":  _in_window(cb, CB_CB_MIN, CB_CB_MAX),
        "ca_ca":  _in_window(ca, CA_CA_MIN, CA_CA_MAX),
        "chi_ss": chi_ok,
    }
    measured = [v for v in w.values() if v is not None]
    return {
        "sg_sg":  None if sg is None else round(sg, 3),
        "cb_cb":  None if cb is None else round(cb, 3),
        "ca_ca":  None if ca is None else round(ca, 3),
        "chi_ss": None if chi is None else round(chi, 1),
        "windows": w,
        "bonding_compatible": bool(w["sg_sg"]),                    # SG–SG in the bonding window
        "all_windows": bool(measured) and all(measured),          # every measured axis in window
    }
