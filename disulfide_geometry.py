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

def _parse_atom_site(cif_path: str, want_atoms, comp_id=None) -> Dict[str, Dict[int, Dict[str, Vec3]]]:
    """ONE-pass `_atom_site` loop parse (the repo's mmCIF convention — robust column lookup from the
    loop header; column order is not fixed). Returns ``{auth_asym_id: {auth_seq_id: {atom: xyz}}}``
    for the atoms in *want_atoms*, restricted to residues of *comp_id* (None = ALL residue types).
    A residue missing an atom simply lacks that key. Shared by `parse_cys_atoms` (CYS, SG/CB/CA)
    and `parse_backbone_atoms` (all residues, CA/CB)."""
    want = set(want_atoms)
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
                    if comp_id is not None and (col.get("label_comp_id") or "").upper() != comp_id:
                        continue
                    atom = col.get("label_atom_id")
                    if atom not in want:
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


def parse_cys_atoms(cif_path: str) -> Dict[str, Dict[int, Dict[str, Vec3]]]:
    """CYS SG/CB/CA coordinates from a fold's mmCIF → ``{chain: {resnum: {"CA","CB","SG"}}}`` for
    CYS residues only (Modes A/B — existing cysteines)."""
    return _parse_atom_site(cif_path, ("CA", "CB", "SG"), comp_id="CYS")


def parse_backbone_atoms(cif_path: str) -> Dict[str, Dict[int, Dict[str, Vec3]]]:
    """N/CA/CB/C of ALL residues from a fold's mmCIF → ``{chain: {resnum: {"N","CA","CB","C"}}}``
    (Mode D — the residue-agnostic backbone scan). N and C are carried for the rotamer Sγ-reachability
    proxy: N anchors the Cys χ1 placement (N–CA–CB–Sγ), and C lets a Gly Cβ be reconstructed
    (`build_cb`) so Gly sites are scored too. Glycine's entry lacks CB; reachability builds one."""
    return _parse_atom_site(cif_path, ("N", "CA", "CB", "C"), comp_id=None)


def parse_heavy_atoms(cif_path: str) -> List[Tuple[str, int, str, Vec3]]:
    """ALL heavy atoms (element ≠ H) of every residue → a flat ``[(chain, resnum, element, xyz)]``
    list for the clash grid (tier b of the reachability proxy). Element from the mmCIF
    ``type_symbol`` column when present, else inferred from the atom-name's leading letters.
    ONE `_atom_site` pass; tolerant of column order. PURE-ish (file read)."""
    cols: List[str] = []
    in_loop = False
    out: List[Tuple[str, int, str, Vec3]] = []
    try:
        with open(cif_path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("_atom_site."):
                    cols.append(s); in_loop = True; continue
                if in_loop and s.startswith(("ATOM", "HETATM")):
                    parts = s.split()
                    if len(parts) < len(cols):
                        continue
                    col = {c.split(".")[1]: parts[i] for i, c in enumerate(cols)}
                    atom = col.get("label_atom_id") or ""
                    el = (col.get("type_symbol") or "").upper() or _element_from_atom_name(atom)
                    if el == "H":
                        continue                         # heavy atoms only
                    ch = col.get("auth_asym_id") or col.get("label_asym_id")
                    rid = col.get("auth_seq_id") or col.get("label_seq_id")
                    if ch is None or rid is None:
                        continue
                    try:
                        xyz = (float(col["Cartn_x"]), float(col["Cartn_y"]), float(col["Cartn_z"]))
                        out.append((ch, int(rid), el, xyz))
                    except (KeyError, TypeError, ValueError):
                        continue
                elif in_loop and s and not s.startswith(("ATOM", "HETATM", "_atom_site.")):
                    if out:
                        break
    except OSError:
        return []
    return out


def _element_from_atom_name(atom: str) -> str:
    """Element from a PDB atom name when `type_symbol` is absent: the leading non-digit letters,
    special-cased so CA/CB/CG… read as carbon (not calcium) and SG/SD as sulfur."""
    a = atom.strip().lstrip("0123456789")
    if not a:
        return "C"
    two = a[:2].upper()
    if two in ("SE", "FE", "ZN", "MG", "MN", "CL", "NA"):
        return two
    return a[0].upper()


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


def bond_constraint(chain_a: str, idx_a: int, chain_b: str, idx_b: int,
                    atom: str = "SG") -> Dict[str, object]:
    """A Boltz `bond` constraint entry between two residues' *atom* (default SG): the
    ``{atom1:[chain,idx,atom], atom2:[chain,idx,atom]}`` shape `_build_yaml` emits. A chain PER
    ATOM (``chain_a``/``chain_b``) so an INTER-chain bond is representable — ``atom1:[CHAIN_A, ia,
    SG], atom2:[CHAIN_B, ib, SG]``. A same-chain bond passes the SAME chain for both and emits the
    IDENTICAL constraint as before. Indices are 1-based chain indices (see `resnum_to_chain_index`),
    each within ITS OWN chain. PURE."""
    return {"atom1": [chain_a, int(idx_a), atom], "atom2": [chain_b, int(idx_b), atom]}


# ── pair-shape helpers (the two-chain container; intra-chain = chain_a == chain_b) ─────────
# A suite pair carries its chain PER MEMBER (`chain_a`/`chain_b`) so the same container holds an
# intra-chain pair (chain_a == chain_b) and — from step 4 — a cross-chain one (chain_a != chain_b),
# one shape not a fork. These two readers are the SINGLE source the router + workbench (display,
# highlight) + persistence rehydration go through, and they tolerate the LEGACY single-`chain`
# shape (pre-reshape pairs + old saved Mode-D scans) so a saved session still reads.

def pair_chains(pair: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    """The ``(chain_a, chain_b)`` of a pair dict. Reshaped pairs carry both keys; a LEGACY pair
    carrying only ``chain`` reads as ``chain_a == chain_b == chain`` (the back-compat hinge for
    old saved scans). None for a key genuinely absent. PURE."""
    ca, cb = pair.get("chain_a"), pair.get("chain_b")
    if ca is None and cb is None:
        ch = pair.get("chain")
        return ch, ch
    return ca, cb


def pair_label(pair: Dict[str, object], *, cys: bool = False) -> str:
    """Human pair label. SAME-chain → bare ``140-88`` (or ``Cys140-Cys88`` with *cys*) so the
    shipped intra-chain display is visually UNCHANGED; CROSS-chain (chain_a != chain_b) →
    ``A:140 <-> B:88`` (chain shown on BOTH members — unreadable otherwise). Reads the reshaped
    chains via `pair_chains` (legacy single-`chain` → same-chain branch). PURE."""
    ca, cb = pair_chains(pair)
    ra, rb = pair.get("resnum_a"), pair.get("resnum_b")
    pre = "Cys" if cys else ""
    if ca is not None and cb is not None and ca != cb:
        sep = " " if cys else ""
        return f"{pre}{sep}{ca}:{ra} ↔ {pre}{sep}{cb}:{rb}"
    return f"{pre}{ra}–{pre}{rb}"


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


# ── Mode D: backbone disulfide-ENGINEERING scan (find NOVEL installable sites) ──────────────
# Residue-agnostic geometry. The score is a PRODUCT of Cα–Cα × Cβ–Cβ × ROTAMER Sγ-REACHABILITY —
# the reachability term REPLACES the old backbone-orientation proxy (Cα-Cβ-Cβ-Cα dihedral) in the
# ranking: instead of surrogating "can the sulfurs reach a good χSS," we place idealized Cys Sγ at
# both backbones over a fine χ1 sweep and measure DIRECTLY whether any rotamer pair brings the two
# Sγ to ~2.05 Å at an acceptable χSS (the real disulfide determinant). The orientation dihedral is
# still MEASURED + displayed (transparency), it just no longer drives rank. Soft-graded, never a
# hard cutoff (a SUGGESTION surface). Cα–Cα stays the cheap, lossless PREFILTER.
CA_CA_SCORE_IDEAL, CA_CA_SCORE_SIGMA = 5.5, 1.0     # Cα–Cα soft window (Å)
CB_CB_SCORE_SIGMA = 0.6                              # Cβ–Cβ soft window σ (centre = CB_CB_IDEAL 3.8)
CA_CA_PREFILTER_GATE = 9.0     # Cα–Cα beyond this → ca-score < 0.0022 → sub-threshold; skip scoring
SCAN_MIN_SEQ_SEP = 2           # exclude sequence-adjacent residues (cannot form a disulfide)
SCAN_MIN_SCORE = 0.05          # pairs below this aren't engineerable-enough to surface (gate is
                               # conservative vs THIS default, so the prefilter is output-lossless)

# Cys sidechain placement (rotamer Sγ-reachability) — validated vs real CYS (n=24: CB–SG 1.818 Å,
# CA–CB–SG 114.0°) and Gly Cβ reconstruction vs real residues (n=112: CA–CB 1.530 Å, N–CA–CB 110.6°,
# C–N–CA–CB −122.3°). A Cys sidechain has ONE rotatable bond (χ1; Sγ is terminal pre-bond), so a
# single-dihedral placement + a fine χ1 sweep fully spans the reachable Sγ positions.
CYS_CB_SG_BOND, CYS_CA_CB_SG_ANGLE = 1.81, 114.0
CB_CA_BOND, CB_N_CA_ANGLE, CB_C_N_CA_DIHEDRAL = 1.53, 110.6, -122.3
CHI1_SWEEP_STEP = 15.0         # χ1 sweep granularity (deg) — FINE, so "unreachable" means NO rotamer
                               # reaches, not "none of 3 staggered did" (the honesty hinge)
SG_SG_REACH_SIGMA = 0.45       # σ of the soft Gaussian on best Sγ–Sγ vs SG_SG_IDEAL (the reach score)
CHI_OUT_FACTOR = 0.5           # multiplier when the best Sγ-pair's χSS is OUTSIDE [60,120]° (proximity
                               # without canonical handedness — softened, never hard-zeroed)
REACH_NEUTRAL = 0.5            # fallback when Sγ can't be placed (N absent) — neutral, never a veto

# Clash check (tier b) — vdW overlap of the placed Sγ (+ a built Gly Cβ) against surrounding heavy
# atoms. A clash flags + softly DEMOTES a site ("sulfurs could meet, but the Cys collides"), never
# hard-eliminates it (suite discipline). The check is ROTAMER-AWARE (2026-06-27): the χ1 sweep
# prefers a reachable CLASH-FREE rotamer and flags a clash only when NO reachable rotamer dodges the
# collision (genuinely unviable on the rigid backbone) — not when the reach-optimal rotamer happens
# to collide (a clash-unaware artifact that over-flagged ~10/17). Radii: standard vdW (Å).
HEAVY_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90,
             "FE": 1.80, "ZN": 1.39, "MG": 1.73, "MN": 1.80, "CL": 1.75, "NA": 2.27}
CLASH_TOLERANCE = 0.5          # allowed vdW overlap (Å) before a contact counts as a clash (kept at
                               # 0.5 — rotamer-awareness fixes the RATE; don't also loosen the
                               # threshold, or the genuinely-unavoidable cases get under-flagged)
CLASH_PENALTY = 0.6            # score multiplier for a residual-clash site (soft demotion; softened
                               # 0.4→0.6 — even "no rotamer dodges" is an uncertain rigid-backbone
                               # signal, neighbours still can't repack, so demote gently)
CLASH_GRID_CELL = 4.0          # spatial-hash cell size (≥ max vdW-sum − tol, so ±1 cell suffices)


def _gauss_score(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


def place_from_internal(a: Vec3, b: Vec3, c: Vec3,
                        bond: float, angle_deg: float, dihedral_deg: float) -> Vec3:
    """Place atom D in Cartesian space from three reference atoms A–B–C and internal coordinates:
    the C–D bond length, the B–C–D bond angle, and the A–B–C–D dihedral (the NeRF / Z-matrix
    construction). Used to put a Cys Sγ at (N, CA, CB) over χ1, and to reconstruct a Gly Cβ from
    (C, N, CA). PURE."""
    ang, dih = math.radians(angle_deg), math.radians(dihedral_deg)

    def _sub(u, v): return (u[0] - v[0], u[1] - v[1], u[2] - v[2])
    def _dot(u, v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    def _cross(u, v): return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])

    def _norm(u):
        m = math.sqrt(_dot(u, u)) or 1.0
        return (u[0]/m, u[1]/m, u[2]/m)

    bc = _norm(_sub(c, b))
    n  = _norm(_cross(_sub(b, a), bc))
    m  = _cross(n, bc)
    d2 = (-bond*math.cos(ang), bond*math.sin(ang)*math.cos(dih), bond*math.sin(ang)*math.sin(dih))
    return (c[0] + bc[0]*d2[0] + m[0]*d2[1] + n[0]*d2[2],
            c[1] + bc[1]*d2[0] + m[1]*d2[1] + n[1]*d2[2],
            c[2] + bc[2]*d2[0] + m[2]*d2[1] + n[2]*d2[2])


def build_cb(n: Vec3, ca: Vec3, c: Vec3) -> Vec3:
    """Reconstruct an idealized Cβ for a residue lacking one (Gly) from its N, CA, C — ideal
    L-amino-acid geometry (CA–CB 1.53 Å, N–CA–CB 110.6°, dihedral C–N–CA–CB −122.3°, validated
    against real residues). So a Gly→Cys engineering site is scored, not silently dropped. PURE."""
    return place_from_internal(c, n, ca, CB_CA_BOND, CB_N_CA_ANGLE, CB_C_N_CA_DIHEDRAL)


def _chi1_grid() -> List[float]:
    """The χ1 sweep samples over (−180, 180] at `CHI1_SWEEP_STEP`. FINE by design — the reachability
    answer must be 'no rotamer can reach', not 'none of a few staggered χ1 reached'."""
    n = int(round(360.0 / CHI1_SWEEP_STEP))
    return [-180.0 + CHI1_SWEEP_STEP * i for i in range(n)]


def _place_sg(n: Vec3, ca: Vec3, cb: Vec3, chi1: float) -> Vec3:
    """Place a Cys Sγ at this backbone for rotamer χ1 = N–CA–CB–Sγ (bond/angle from real CYS)."""
    return place_from_internal(n, ca, cb, CYS_CB_SG_BOND, CYS_CA_CB_SG_ANGLE, chi1)


def sg_rotamers(n: Vec3, ca: Vec3, cb: Vec3) -> List[Vec3]:
    """The Sγ position for every χ1 in the sweep — placed ONCE per residue, reused across all its
    candidate pairs in a scan (the cross-product of two residues' rotamer sets is the per-pair work)."""
    return [_place_sg(n, ca, cb, chi) for chi in _chi1_grid()]


def sg_reachability(cb_a: Vec3, sgs_a: List[Vec3], cb_b: Vec3, sgs_b: List[Vec3],
                    clash_checker=None) -> Dict[str, object]:
    """The DIRECT disulfide determinant the backbone-orientation term only proxied — ROTAMER-AWARE.
    Over both residues' Sγ rotamer sets (ONE sweep), score reach quality ``Gaussian(Sγ–Sγ, 2.05, σ)
    × (1 if |χSS|∈[60,120]° else CHI_OUT_FACTOR)`` for every χ1×χ1 pair, tracking TWO bests: the
    reach-optimal pair (any clash), and — when a *clash_checker(sg_a, sg_b)→bool* is given — the
    reach-optimal pair that is also CLASH-FREE. Reports the CLASH-FREE best if one exists (``clash``
    False, the geometry a real disulfide more likely adopts — the sidechain settles clash-free), else
    the reach-optimal best with ``clash`` True (genuinely unviable — NO reachable rotamer dodges the
    collision). Without a checker, ``clash`` is None (not evaluated). The clash probe runs only on a
    reach-IMPROVING candidate (q above the running clean best), so it stays cheap. Returns
    ``{reach_score, best_sg_sg, best_chi_ss, sg_a, sg_b, clash}``. Soft (never hard-zero). PURE-ish
    (clash_checker may read a grid)."""
    # Three bests, by quality q: best OVERALL (any rotamer — the fallback geometry/score), best
    # REACHING (a genuine disulfide: |χSS|∈[60,120]° AND Sγ–Sγ in the bonding window), and best
    # reaching AND CLASH-FREE. "Clash-free" is asked ONLY among REACHING rotamers — dodging a clash
    # by adopting a non-disulfide dihedral is NOT a viable disulfide, so it doesn't clear the flag.
    best_all = (-1.0, None, None, (sgs_a[0], sgs_b[0]))        # (q, d, chi, (sga, sgb))
    best_reach = (-1.0, None, None, None)
    best_clean = (-1.0, None, None, None)
    done = False
    for sga in sgs_a:
        for sgb in sgs_b:
            d = calc_distance(sga, sgb)
            chi = calc_dihedral(cb_a, sga, sgb, cb_b)
            in_win = CHI_SS_MIN <= abs(chi) <= CHI_SS_MAX
            q = _gauss_score(d, SG_SG_IDEAL, SG_SG_REACH_SIGMA) * (1.0 if in_win else CHI_OUT_FACTOR)
            if q > best_all[0]:
                best_all = (q, d, chi, (sga, sgb))
                if clash_checker is None and q >= 0.999:      # tier-a fast path: ideal, no clash to weigh
                    done = True; break
            if in_win and SG_SG_MIN <= d <= SG_SG_MAX:        # a REACHING rotamer (real disulfide geometry)
                if q > best_reach[0]:
                    best_reach = (q, d, chi, (sga, sgb))
                # clash-test only a reaching rotamer that could become the clash-free best (bounded)
                if clash_checker is not None and q > best_clean[0] and not clash_checker(sga, sgb):
                    best_clean = (q, d, chi, (sga, sgb))
                    if q >= 0.999:                            # ideal AND clash-free — can't do better
                        done = True; break
        if done:
            break
    if clash_checker is None:
        q, d, chi, sg, clash = (*best_all, None)
    elif best_clean[1] is not None:                           # a REACHING clash-free rotamer exists
        q, d, chi, sg, clash = (*best_clean, False)
    elif best_reach[1] is not None:                           # reaching rotamers exist but ALL collide
        q, d, chi, sg, clash = (*best_reach, True)            # → genuinely unviable on the rigid backbone
    else:                                                     # no reaching rotamer at all (a REACH miss,
        q, d, chi, sg, clash = (*best_all, False)             #   not a clash — the low reach score says so)
    return {"reach_score": round(q, 4),
            "best_sg_sg": (None if d is None else round(d, 3)),
            "best_chi_ss": (None if chi is None else round(chi, 1)),
            "sg_a": sg[0], "sg_b": sg[1], "clash": clash}


def backbone_pair_score(res_a: Dict[str, Vec3], res_b: Dict[str, Vec3], *,
                        min_surface_score: Optional[float] = None,
                        clash_grid: Optional["ClashGrid"] = None,
                        exclude: Optional[set] = None) -> Optional[Dict[str, object]]:
    """Soft-graded engineerability of INSTALLING a disulfide between two residues (residue-agnostic).
    ``score = Cα–Cα Gaussian × Cβ–Cβ Gaussian × ROTAMER Sγ-REACHABILITY`` — reachability takes the
    slot the weak backbone-orientation proxy held. Returns the measured backbone geometry (Cα–Cα,
    Cβ–Cβ, the Cα-Cβ-Cβ-Cα orientation dihedral — MEASURED + displayed, no longer scored) PLUS the
    reachability readout (best achievable Sγ–Sγ + χSS) + a ``clash`` flag. With *clash_grid* (+ the
    pair's *exclude* ``{(chain,resnum)}``) the χ1 sweep is ROTAMER-AWARE: it reports the best
    reachable CLASH-FREE rotamer's geometry (clash False) and flags a clash ONLY when no reachable
    rotamer dodges the collision (clash True) — the displayed Sγ–Sγ/χSS are then the clash-free
    conformation a real disulfide more likely adopts. Builds a Gly Cβ (`build_cb`) so Gly→Cys is
    scored. If N is absent (Sγ unplaceable) reachability falls back to REACH_NEUTRAL.
    *min_surface_score* (the scan passes its `min_score`) SKIPS the sweep when Cα×Cβ alone can't
    clear it (reach ≤ 1) — a LOSSLESS speed gate. None if either residue lacks a Cα."""
    ca1, ca2 = res_a.get("CA"), res_b.get("CA")
    if not (ca1 and ca2):
        return None
    # Cβ: real, or reconstructed for Gly (needs N+C); else CA pseudo-Cβ (last-resort, no reach).
    cb1, built1 = _resolve_cb(res_a, ca1)
    cb2, built2 = _resolve_cb(res_b, ca2)
    has_real_cb = bool(res_a.get("CB") and res_b.get("CB"))
    ca = calc_distance(ca1, ca2)
    cb = calc_distance(cb1, cb2)
    # Orientation dihedral — MEASURED for display only when BOTH Cβ are real (a built/pseudo Cβ is
    # not a measured backbone feature). No longer in the score.
    dih = calc_dihedral(ca1, cb1, cb2, ca2) if has_real_cb else None
    ca_sc = _gauss_score(ca, CA_CA_SCORE_IDEAL, CA_CA_SCORE_SIGMA)
    cb_sc = _gauss_score(cb, CB_CB_IDEAL, CB_CB_SCORE_SIGMA)

    # Rotamer Sγ-reachability (the ranking term). Needs N + a real-or-built Cβ at both positions to
    # place Sγ; if either Cβ is the last-resort CA pseudo-Cβ (no N/C to build), reach falls back.
    n1, n2 = res_a.get("N"), res_b.get("N")
    skip_sweep = (min_surface_score is not None and ca_sc * cb_sc < min_surface_score)
    if not skip_sweep and n1 and n2 and cb1 is not ca1 and cb2 is not ca2:
        sgs_a, sgs_b = sg_rotamers(n1, ca1, cb1), sg_rotamers(n2, ca2, cb2)
        checker = _make_clash_checker(clash_grid, exclude, cb1 if built1 is not None else None,
                                      cb2 if built2 is not None else None) if clash_grid else None
        reach = sg_reachability(cb1, sgs_a, cb2, sgs_b, clash_checker=checker)
        reach_sc, clash = float(reach["reach_score"]), reach["clash"]
    else:
        reach = {"reach_score": REACH_NEUTRAL, "best_sg_sg": None, "best_chi_ss": None}
        reach_sc, clash = REACH_NEUTRAL, None

    return {
        "ca_ca": round(ca, 3), "cb_cb": round(cb, 3),
        "orientation": (None if dih is None else round(dih, 1)),
        "ca_score": round(ca_sc, 4), "cb_score": round(cb_sc, 4),
        "reach_score": round(reach_sc, 4),
        "best_sg_sg": reach["best_sg_sg"], "best_chi_ss": reach["best_chi_ss"],
        "clash": clash,
        "score": round(ca_sc * cb_sc * reach_sc, 4),     # clash-FREE score; scan demotes a clash
    }


def _make_clash_checker(grid: "ClashGrid", exclude: Optional[set],
                        cb_a_built: Optional[Vec3], cb_b_built: Optional[Vec3]):
    """Build the per-pair clash predicate the rotamer sweep calls: ``checker(sg_a, sg_b) → bool``.
    The two placed Sγ vary per rotamer; any BUILT Gly Cβ is fixed, so it rides as a constant probe.
    Excludes the two mutated residues' own atoms (their sidechains become the Cys; the partner Sγ is
    the intended bond, not a clash)."""
    fixed = []
    if cb_a_built is not None:
        fixed.append((cb_a_built, "C"))
    if cb_b_built is not None:
        fixed.append((cb_b_built, "C"))
    exc = exclude or set()

    def checker(sg_a: Vec3, sg_b: Vec3) -> bool:
        return grid.any_clash([(sg_a, "S"), (sg_b, "S")] + fixed, exc)
    return checker


def _resolve_cb(res: Dict[str, Vec3], ca: Vec3) -> Tuple[Vec3, Optional[Vec3]]:
    """Return ``(cb, built)`` for a residue: the real Cβ (built=None), else a reconstructed Gly Cβ
    from N+CA+C (built=the same coord — flags it was built), else CA as a last-resort pseudo-Cβ
    (built=None, signalled by ``cb is ca``) when N or C is missing."""
    cb = res.get("CB")
    if cb is not None:
        return cb, None
    n, c = res.get("N"), res.get("C")
    if n and c:
        built = build_cb(n, ca, c)
        return built, built
    return ca, None                                      # pseudo-Cβ (can't build) — reach falls back


# ── clash grid (tier b — local vdW overlap of the placed rotamer) ──────────────────────────
class ClashGrid:
    """A spatial hash of a structure's heavy atoms (built ONCE per scan) so each candidate's placed
    Sγ/built-Cβ can be clash-tested in O(1) against only its ~27 neighbouring cells. `any_clash`
    excludes the two mutated residues' own atoms (their sidechains are replaced by the Cys, and the
    two Sγ are the intended bonding partners — not a clash with each other)."""

    def __init__(self, atoms: List[Tuple[str, int, str, Vec3]], cell: float = CLASH_GRID_CELL):
        self.cell = cell
        self.grid: Dict[Tuple[int, int, int], List[Tuple[str, int, str, Vec3]]] = {}
        for rec in atoms:
            self.grid.setdefault(self._key(rec[3]), []).append(rec)

    def _key(self, xyz: Vec3) -> Tuple[int, int, int]:
        c = self.cell
        return (int(math.floor(xyz[0]/c)), int(math.floor(xyz[1]/c)), int(math.floor(xyz[2]/c)))

    def any_clash(self, probes: List[Tuple[Vec3, str]], exclude: set) -> bool:
        """True if any probe ``(xyz, element)`` overlaps a heavy atom (vdW sum − tolerance) NOT in an
        excluded ``(chain, resnum)`` residue. Checks the probe's cell + its 26 neighbours only."""
        for xyz, el in probes:
            pr = HEAVY_VDW.get(el, 1.70)
            ci, cj, ck = self._key(xyz)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        for (ch, rn, ael, axyz) in self.grid.get((ci+di, cj+dj, ck+dk), ()):
                            if (ch, rn) in exclude:
                                continue
                            if calc_distance(xyz, axyz) < (pr + HEAVY_VDW.get(ael, 1.70) - CLASH_TOLERANCE):
                                return True
        return False


def _demote_if_clash(pg: Dict[str, object]) -> Dict[str, object]:
    """SOFTLY demote a residual-clash candidate (× CLASH_PENALTY), never hard-eliminate it. The
    ``clash`` flag is already set by `backbone_pair_score`'s ROTAMER-AWARE sweep (True only when NO
    reachable rotamer dodges the collision). The min_score gate upstream runs on the CLASH-FREE
    score, so a clash demotes rank but never hides a site."""
    if pg.get("clash") is True:
        pg["score"] = round(float(pg["score"]) * CLASH_PENALTY, 4)
    return pg


def scan_engineerable_sites(atoms_by_chain: Dict[str, Dict[int, Dict[str, Vec3]]], *,
                            min_seq_sep: int = SCAN_MIN_SEQ_SEP,
                            ca_gate: Optional[float] = CA_CA_PREFILTER_GATE,
                            min_score: float = SCAN_MIN_SCORE,
                            clash_grid: Optional["ClashGrid"] = None):
    """INTRACHAIN all-pairs backbone scan for NOVEL installable disulfide sites. Per chain, every
    residue pair separated by ≥ *min_seq_sep* in sequence is scored by `backbone_pair_score` (Cα–Cα
    × Cβ–Cβ × rotamer Sγ-reachability); a cheap conservative Cα–Cα PREFILTER (*ca_gate*) skips pairs
    too far to clear *min_score* BEFORE the (more expensive) reachability sweep — SPEED only, the
    OUTPUT is identical to a full scan (gated-out pairs are sub-threshold). With *clash_grid* (tier
    b) each surfaced candidate is clash-checked (flag + soft demotion). Returns ``(ranked_pairs,
    best_partner)``: ranked_pairs sorted by final score desc, each carrying ``chain_a``/``chain_b``
    (equal here — intrachain) + ``resnum_a``/``resnum_b``; best_partner ``{(chain, resnum): best
    score}`` for the heatmap. Pass ``ca_gate=None`` for an un-prefiltered full scan."""
    ranked: List[Dict[str, object]] = []
    best: Dict[tuple, float] = {}
    for ch, residues in (atoms_by_chain or {}).items():
        rns = sorted(residues)
        for ia in range(len(rns)):
            ra = rns[ia]
            ca_a = residues[ra].get("CA")
            if not ca_a:
                continue
            for ib in range(ia + 1, len(rns)):
                rb = rns[ib]
                if abs(rb - ra) < min_seq_sep:
                    continue
                ca_b = residues[rb].get("CA")
                if not ca_b:
                    continue
                if ca_gate is not None and calc_distance(ca_a, ca_b) > ca_gate:
                    continue                          # PREFILTER — too far; skip the full score
                pg = backbone_pair_score(residues[ra], residues[rb], min_surface_score=min_score,
                                         clash_grid=clash_grid, exclude={(ch, ra), (ch, rb)})
                if pg is None or pg["score"] < min_score:   # gate on the CLASH-FREE score
                    continue
                pg = _demote_if_clash(pg)
                # INTRACHAIN: both members are on `ch`, so the two-chain pair shape collapses to
                # chain_a == chain_b here (no cross-chain enumeration until step 4). best_partner is
                # keyed per MEMBER by ITS OWN chain — already the cross-chain-correct form.
                ranked.append({"chain_a": ch, "resnum_a": ra, "chain_b": ch, "resnum_b": rb, **pg})
                best[(ch, ra)] = max(best.get((ch, ra), 0.0), float(pg["score"]))
                best[(ch, rb)] = max(best.get((ch, rb), 0.0), float(pg["score"]))
    ranked.sort(key=lambda d: -d["score"])
    return ranked, best


def scan_interface_sites(atoms_by_chain: Dict[str, Dict[int, Dict[str, Vec3]]], *,
                         ca_gate: Optional[float] = CA_CA_PREFILTER_GATE,
                         min_score: float = SCAN_MIN_SCORE,
                         clash_grid: Optional["ClashGrid"] = None):
    """CROSS-chain (INTER-subunit) engineerable-disulfide scan: residue pairs whose two members are
    on DIFFERENT chains. The SAME `backbone_pair_score` Mode D uses scores each pair (NO new geometry
    loop — the §9 universal-disulfide-mode convergence point, the shared primitive both this and the
    intra `scan_engineerable_sites` call). The Cα–Cα PREFILTER (*ca_gate*) IS the INTERFACE bound: a
    pair close enough ACROSS chains to possibly form a disulfide is interface-proximal by definition;
    a buried-core residue is too far from the other chain to pass, so the output is interface-bounded
    (NOT the full A×B product). NO `min_seq_sep` — cross-chain residues have no sequence adjacency.
    With *clash_grid* (tier b) each surfaced candidate is clash-checked (flag + soft demotion).
    Returns ``(ranked_pairs, best_partner)``: each pair carries ``chain_a`` != ``chain_b`` (the
    two-chain reshape) + ``resnum_a``/``resnum_b``; best_partner ``{(chain, resnum): best score}``.
    DISTINCT from `scan_engineerable_sites` (intra-chain) — feeds the cross-chain Mode-C declare."""
    ranked: List[Dict[str, object]] = []
    best: Dict[tuple, float] = {}
    chains = sorted(atoms_by_chain or {})
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):               # each unordered chain pair once (A–B, not B–A)
            cha, chb = chains[i], chains[j]
            res_a, res_b = atoms_by_chain[cha], atoms_by_chain[chb]
            for ra in sorted(res_a):
                ca_a = res_a[ra].get("CA")
                if not ca_a:
                    continue
                for rb in sorted(res_b):
                    ca_b = res_b[rb].get("CA")
                    if not ca_b:
                        continue
                    if ca_gate is not None and calc_distance(ca_a, ca_b) > ca_gate:
                        continue                          # INTERFACE bound — too far across chains
                    pg = backbone_pair_score(res_a[ra], res_b[rb], min_surface_score=min_score,
                                             clash_grid=clash_grid, exclude={(cha, ra), (chb, rb)})
                    if pg is None or pg["score"] < min_score:   # gate on the CLASH-FREE score
                        continue
                    pg = _demote_if_clash(pg)
                    ranked.append({"chain_a": cha, "resnum_a": ra, "chain_b": chb, "resnum_b": rb, **pg})
                    best[(cha, ra)] = max(best.get((cha, ra), 0.0), float(pg["score"]))
                    best[(chb, rb)] = max(best.get((chb, rb), 0.0), float(pg["score"]))
    ranked.sort(key=lambda d: -d["score"])
    return ranked, best
