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
    """CA/CB of ALL residues from a fold's mmCIF → ``{chain: {resnum: {"CA","CB"}}}`` (Mode D —
    the residue-agnostic backbone scan). Glycine has no CB; its entry holds CA only (the scan
    falls back to CA as a pseudo-Cβ, standard disulfide-engineering practice)."""
    return _parse_atom_site(cif_path, ("CA", "CB"), comp_id=None)


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
# Residue-agnostic backbone geometry — NO χSS (no Sγ exists pre-mutation). Soft-graded windows,
# never hard cutoffs (a SUGGESTION surface, not a filter). The combine is a PRODUCT of the three
# axes, so the Cα–Cα distance dominates: a far-apart pair scores ~0 (physically — too far = no
# disulfide regardless of orientation). That same property makes the Cα PREFILTER lossless.
CA_CA_SCORE_IDEAL, CA_CA_SCORE_SIGMA = 5.5, 1.0     # Cα–Cα soft window (Å)
CB_CB_SCORE_SIGMA = 0.6                              # Cβ–Cβ soft window σ (centre = CB_CB_IDEAL 3.8)
ORIENT_IDEAL_D, ORIENT_CUTOFF_D = 90.0, 135.0       # |Cα-Cβ-Cβ-Cα| dihedral: ±90° ideal
CA_CA_PREFILTER_GATE = 9.0     # Cα–Cα beyond this → ca-score < 0.0022 → sub-threshold; skip scoring
SCAN_MIN_SEQ_SEP = 2           # exclude sequence-adjacent residues (cannot form a disulfide)
SCAN_MIN_SCORE = 0.05          # pairs below this aren't engineerable-enough to surface (gate is
                               # conservative vs THIS default, so the prefilter is output-lossless)


def _gauss_score(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


def _orient_score(dihedral_deg: Optional[float]) -> float:
    """Soft raised-cosine on |dev from ±90°|: 1.0 at ±90°, 0.5 at 0/180°, 0 beyond the cutoff.
    None (a Gly pseudo-Cβ → orientation undefined) → 0.5 neutral. Soft, never a hard cutoff
    except the far-deviation floor."""
    if dihedral_deg is None:
        return 0.5
    dev = min(abs(dihedral_deg - ORIENT_IDEAL_D), abs(dihedral_deg + ORIENT_IDEAL_D))
    if dev >= ORIENT_CUTOFF_D:
        return 0.0
    return (1.0 + math.cos(dev * math.pi / 180.0)) / 2.0


def backbone_pair_score(res_a: Dict[str, Vec3], res_b: Dict[str, Vec3]) -> Optional[Dict[str, object]]:
    """Soft-graded engineerability of INSTALLING a disulfide between two residues (residue-agnostic;
    CA + CB, Gly → CA as a pseudo-Cβ). NO χSS. Returns the measured backbone geometry (Cα–Cα,
    Cβ–Cβ, Cα-Cβ-Cβ-Cα orientation) + per-axis soft scores + a combined ``score`` in [0,1] (the
    PRODUCT — graded, Cα-distance-dominant; near-misses score lower but are NEVER hard-eliminated).
    None if either residue lacks a Cα. PURE."""
    ca1, ca2 = res_a.get("CA"), res_b.get("CA")
    if not (ca1 and ca2):
        return None
    cb1 = res_a.get("CB") or ca1                     # Gly: CA stands in as the pseudo-Cβ
    cb2 = res_b.get("CB") or ca2
    has_cb = bool(res_a.get("CB") and res_b.get("CB"))
    ca = calc_distance(ca1, ca2)
    cb = calc_distance(cb1, cb2)
    dih = calc_dihedral(ca1, cb1, cb2, ca2) if has_cb else None
    ca_sc, cb_sc = _gauss_score(ca, CA_CA_SCORE_IDEAL, CA_CA_SCORE_SIGMA), _gauss_score(cb, CB_CB_IDEAL, CB_CB_SCORE_SIGMA)
    or_sc = _orient_score(dih)
    return {
        "ca_ca": round(ca, 3), "cb_cb": round(cb, 3),
        "orientation": (None if dih is None else round(dih, 1)),
        "ca_score": round(ca_sc, 4), "cb_score": round(cb_sc, 4), "orient_score": round(or_sc, 4),
        "score": round(ca_sc * cb_sc * or_sc, 4),
    }


def scan_engineerable_sites(atoms_by_chain: Dict[str, Dict[int, Dict[str, Vec3]]], *,
                            min_seq_sep: int = SCAN_MIN_SEQ_SEP,
                            ca_gate: Optional[float] = CA_CA_PREFILTER_GATE,
                            min_score: float = SCAN_MIN_SCORE):
    """INTRACHAIN all-pairs backbone scan for NOVEL installable disulfide sites. Per chain, every
    residue pair separated by ≥ *min_seq_sep* in sequence is scored by `backbone_pair_score`; a
    cheap conservative Cα–Cα PREFILTER (*ca_gate*) skips pairs too far to clear *min_score* BEFORE
    the dihedral compute — SPEED only, the OUTPUT is identical to a full scan (gated-out pairs are
    sub-threshold). Returns ``(ranked_pairs, best_partner)``: ranked_pairs sorted by score desc,
    each carrying ``chain_a``/``chain_b`` (equal here — intrachain) + ``resnum_a``/``resnum_b``;
    best_partner ``{(chain, resnum): best score}`` for the heatmap (a residue's colour = its best
    available partner's score). Pass ``ca_gate=None`` for an un-prefiltered full scan."""
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
                pg = backbone_pair_score(residues[ra], residues[rb])
                if pg is None or pg["score"] < min_score:
                    continue
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
                         min_score: float = SCAN_MIN_SCORE):
    """CROSS-chain (INTER-subunit) engineerable-disulfide scan: residue pairs whose two members are
    on DIFFERENT chains. The SAME `backbone_pair_score` Mode D uses scores each pair (NO new geometry
    loop — the §9 universal-disulfide-mode convergence point, the shared primitive both this and the
    intra `scan_engineerable_sites` call). The Cα–Cα PREFILTER (*ca_gate*) IS the INTERFACE bound: a
    pair close enough ACROSS chains to possibly form a disulfide is interface-proximal by definition;
    a buried-core residue is too far from the other chain to pass, so the output is interface-bounded
    (NOT the full A×B product). NO `min_seq_sep` — cross-chain residues have no sequence adjacency.
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
                    pg = backbone_pair_score(res_a[ra], res_b[rb])
                    if pg is None or pg["score"] < min_score:
                        continue
                    ranked.append({"chain_a": cha, "resnum_a": ra, "chain_b": chb, "resnum_b": rb, **pg})
                    best[(cha, ra)] = max(best.get((cha, ra), 0.0), float(pg["score"]))
                    best[(chb, rb)] = max(best.get((chb, rb), 0.0), float(pg["score"]))
    ranked.sort(key=lambda d: -d["score"])
    return ranked, best
