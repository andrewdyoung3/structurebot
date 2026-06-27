"""
proline_geometry.py
-------------------
Plain-Python core for the PROLINE-stabilization suggestion scan — the sibling of
`disulfide_geometry.py`, built fresh in the same architecture (NO new dependency: it parses the
mmCIF `_atom_site` loop, reuses the lifted `calc_distance`/`calc_dihedral` primitives, and stays
loadable without the tool_router/bridge chain).

THE BIOCHEMISTRY (what the scan measures — grounded here, not hardcoded site lists):
Proline stabilizes a fold by REDUCING UNFOLDED-STATE conformational entropy — its pyrrolidine ring
locks the backbone φ ≈ −63° and removes N–Cα rotational freedom. A residue is a good X→Pro candidate
when (1) its existing backbone **φ is already proline-compatible** (~−63°) — the DOMINANT signal;
(2) its **ψ is not in a Pro-forbidden region** (proline tolerates a BROAD ψ, so this is a gentle
veto on the rare incompatible case, not a tight filter); (3) substituting Pro **would not break a
backbone H-bond** — proline has no amide H, so it cannot DONATE the N–H···O bond that helices/sheets
rely on (this is the real reason stabilizing prolines live in loops/turns, not helix interiors).

SECONDARY STRUCTURE FALLS OUT — there is NO explicit DSSP/SS-assignment term: a helix interior is
φ/ψ-in-the-helix-basin AND its amide N–H DONATES the i→i−4 backbone H-bond, so the H-bond penalty
fires there automatically and demotes it; a loop/turn donates no backbone H-bond, so it floats up.

This is the correct version built FRESH — NOT a port of the legacy `proline_bridge.py`, which (a)
HARD-excludes helix (here: a soft H-bond penalty, never exclusion) and (b) used a FAKE hbond_factor
(a cis-peptide proxy); here the H-bond signal is REAL geometric DSSP-style backbone H-bond detection.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from disulfide_geometry import Vec3, calc_distance, calc_dihedral

# ── tunable scoring constants (STARTING POINTS — tune from use, like the disulfide constants) ─────
PHI_IDEAL = -63.0          # proline's ring-locked backbone φ (deg)
PHI_SIGMA = 18.0           # Gaussian width on φ (deg) — φ is the DOMINANT discriminator
# ψ is BROADLY allowed for proline (the ring constrains φ, not ψ); only the positive-ψ "bridge"
# between the αR (ψ≈−35°) and PPII/β (ψ≈+150°) basins is genuinely depleted. The ψ term is ~1.0
# almost everywhere and dips (soft cosine) only inside that band → a gentle veto, not a tight filter.
PSI_FORBIDDEN_CENTER = 65.0    # middle of the depleted positive-ψ bridge (deg)
PSI_FORBIDDEN_HALFWIDTH = 45.0  # the dip spans ~+20°..+110°; flat (1.0) outside
PSI_PENALTY = 0.5          # ψ sub-score floor at the centre of the forbidden band
HBOND_PENALTY = 0.3        # HEAVY SOFT penalty (× score) when residue i's N–H donates a backbone
                           # H-bond → Pro can't donate it → destabilizing. FLAG + penalty, NEVER exclude.
SCAN_MIN_SCORE = 0.05      # surfacing floor — below this a site isn't proline-favourable enough to show

# DSSP-style backbone H-bond (the one new capability; plain-Python, CIF-based, no ChimeraX) ─────────
HB_N_H_BOND = 1.01         # inferred amide N–H bond length (Å)
HB_ENERGY_CUTOFF = -0.5    # DSSP H-bond electrostatic-energy threshold (kcal/mol); E below → bonded
HB_Q1Q2_F = 0.084 * 332.0  # DSSP electrostatic prefactor 0.084·332 = 27.888 (kcal·Å/mol)
HB_MIN_SEQ_SEP = 2         # ignore i and its sequence-bonded neighbours i±1 (DSSP convention)
HB_NEIGHBOR_CUTOFF = 5.5   # only test acceptor O within this of the donor N (a cheap N···O prefilter)

_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


# ── parse: per-residue backbone N/CA/C/O + residue name (ONE pass; the proline scan's only parse) ──
def parse_backbone_with_names(cif_path: str) -> Dict[str, Dict[int, Dict[str, object]]]:
    """``{chain: {resnum: {"N","CA","C","O": xyz, "resname": "ALA"}}}`` from a fold's mmCIF — backbone
    N/CA/C (for φ/ψ) + carbonyl O (for the H-bond acceptor) + the residue NAME (for from_aa / existing-
    Pro). ONE `_atom_site` pass (the repo convention; robust column lookup, tolerant of column order).
    A residue simply lacks a key for a missing atom. PURE-ish (file read)."""
    want = {"N", "CA", "C", "O"}
    cols: List[str] = []
    in_loop = False
    out: Dict[str, Dict[int, Dict[str, object]]] = {}
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
                    res.setdefault(atom, xyz)                # don't overwrite altlocs
                    res.setdefault("resname", (col.get("label_comp_id") or "").upper())
                elif in_loop and s and not s.startswith(("ATOM", "HETATM", "_atom_site.")):
                    if out:
                        break
    except OSError:
        return {}
    return out


# ── φ/ψ (pure reuse of calc_dihedral; neighbour C(i−1) / N(i+1); termini → None, never a crash) ────
def phi_psi(prev_res: Optional[dict], res: dict, next_res: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    """(φ, ψ) for *res* given its sequence neighbours. φ = C(i−1)–N(i)–CA(i)–C(i) (None without a
    preceding C — N-terminus / chain break); ψ = N(i)–CA(i)–C(i)–N(i+1) (None without a following N —
    C-terminus). PURE."""
    n, ca, c = res.get("N"), res.get("CA"), res.get("C")
    if not (n and ca and c):
        return None, None
    c_prev = prev_res.get("C") if prev_res else None
    n_next = next_res.get("N") if next_res else None
    phi = calc_dihedral(c_prev, n, ca, c) if c_prev else None
    psi = calc_dihedral(n, ca, c, n_next) if n_next else None
    return (None if phi is None else round(phi, 1)), (None if psi is None else round(psi, 1))


# ── soft scores ───────────────────────────────────────────────────────────────────────────────────
def _ang_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a−b in (−180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def phi_score(phi: Optional[float]) -> float:
    """Gaussian on φ centred at the proline ideal (−63°). None (no φ — N-terminus) → 0.0 (a residue
    with no defined φ can't host a stabilizing proline). The DOMINANT term."""
    if phi is None:
        return 0.0
    return math.exp(-(_ang_diff(phi, PHI_IDEAL) ** 2) / (2.0 * PHI_SIGMA * PHI_SIGMA))


def psi_score(psi: Optional[float]) -> float:
    """Soft, mostly-FLAT ψ term: 1.0 across proline's broad allowed ψ, dipping (cosine) to PSI_PENALTY
    only inside the depleted positive-ψ bridge (~+20°..+110°). None (no ψ — C-terminus) → 1.0 (neutral;
    don't penalize a terminus for a missing ψ). A gentle veto, not a tight filter."""
    if psi is None:
        return 1.0
    d = abs(_ang_diff(psi, PSI_FORBIDDEN_CENTER))
    if d >= PSI_FORBIDDEN_HALFWIDTH:
        return 1.0
    dip = 0.5 * (1.0 + math.cos(math.pi * d / PSI_FORBIDDEN_HALFWIDTH))   # 1 at centre → 0 at edge
    return 1.0 - dip * (1.0 - PSI_PENALTY)


# ── DSSP-style backbone H-bond detection (the load-bearing new capability) ──────────────────────────
def infer_amide_h(n: Vec3, ca: Vec3, c_prev: Vec3) -> Vec3:
    """The amide hydrogen on residue i's backbone N, inferred (no H in the CIF). The backbone N is
    planar sp2 with three substituents ~120° apart — C(i−1), CA(i), and H — so H lies in the peptide
    plane ANTI to the bisector of the N→C(i−1) and N→CA(i) directions, 1.01 Å from N. (This proper
    sp2 placement is validated against real helices to give the expected −1 to −4 kcal/mol i→i−4
    H-bond energies; the cruder 'extend the C(i−1)→N bond' approximation under-detects badly.) PURE."""
    def _u(a: Vec3, b: Vec3) -> Vec3:
        d = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        m = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) or 1.0
        return (d[0] / m, d[1] / m, d[2] / m)
    d1, d2 = _u(c_prev, n), _u(ca, n)                    # N→C(i−1), N→CA(i)
    bis = (-(d1[0] + d2[0]), -(d1[1] + d2[1]), -(d1[2] + d2[2]))
    m = math.sqrt(bis[0] ** 2 + bis[1] ** 2 + bis[2] ** 2) or 1.0
    return (n[0] + HB_N_H_BOND * bis[0] / m, n[1] + HB_N_H_BOND * bis[1] / m, n[2] + HB_N_H_BOND * bis[2] / m)


def hbond_energy(n: Vec3, h: Vec3, c: Vec3, o: Vec3) -> float:
    """DSSP backbone H-bond electrostatic energy (kcal/mol) for donor (N–H) → acceptor (C=O):
    ``E = 0.084·332·(1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN)``. More negative = stronger; < −0.5 = an
    H-bond. Returns a large positive number if any pair is degenerately close (guard). PURE."""
    r_on, r_ch, r_oh, r_cn = (calc_distance(o, n), calc_distance(c, h),
                              calc_distance(o, h), calc_distance(c, n))
    if min(r_on, r_ch, r_oh, r_cn) < 1e-3:
        return 1e9
    return HB_Q1Q2_F * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)


def donates_backbone_hbond(n: Vec3, h: Vec3, acceptors: List[Tuple[str, int, Vec3, Vec3]],
                           donor_key: Tuple[str, int]) -> Tuple[bool, Optional[float]]:
    """Does this donor amide (N, inferred H) donate a backbone H-bond to ANY acceptor carbonyl (C, O)?
    *acceptors* = ``[(chain, resnum, C, O)]`` (across ALL chains — an inter-chain backbone H-bond
    counts). Excludes the donor's own residue + its sequence neighbours i±1 (DSSP convention) and
    prefilters on N···O distance. Returns ``(donates, best_energy)`` — best_energy is the most negative
    E found (None if none qualify). PURE."""
    best: Optional[float] = None
    dch, drn = donor_key
    for ach, arn, c, o in acceptors:
        if ach == dch and abs(arn - drn) < HB_MIN_SEQ_SEP:        # self + i±1 (same chain)
            continue
        if calc_distance(n, o) > HB_NEIGHBOR_CUTOFF:              # cheap N···O prefilter
            continue
        e = hbond_energy(n, h, c, o)
        if e < HB_ENERGY_CUTOFF and (best is None or e < best):
            best = e
    return (best is not None), best


# ── the scan ────────────────────────────────────────────────────────────────────────────────────
def scan_proline_sites(atoms_by_chain: Dict[str, Dict[int, Dict[str, object]]], *,
                       min_score: float = SCAN_MIN_SCORE):
    """Per-residue X→Pro stabilization scan over a whole structure (multimers native — every chain,
    cross-chain H-bonds counted). For each residue with a defined φ that is not ALREADY proline, score
    ``phi_score × psi_score × hbond_factor`` (hbond_factor = HBOND_PENALTY if its amide N–H donates a
    backbone H-bond, else 1.0) and record the H-bond-donor FLAG. Returns ``(ranked, best_partner)``:
    ranked = per-residue dicts sorted by score desc, each ``{chain, position, from_aa, phi, psi,
    phi_score, psi_score, hbond_donates, hbond_energy, score}``; best_partner = ``{(chain, resnum):
    score}`` for the heatmap. SS context FALLS OUT of φ/ψ + H-bond (no explicit SS term). PURE-ish."""
    # one flat acceptor list (C, O) across ALL chains — an amide may H-bond across a subunit interface
    acceptors: List[Tuple[str, int, Vec3, Vec3]] = []
    for ch, residues in (atoms_by_chain or {}).items():
        for rn, r in residues.items():
            if r.get("C") and r.get("O"):
                acceptors.append((ch, rn, r["C"], r["O"]))

    ranked: List[Dict[str, object]] = []
    best: Dict[tuple, float] = {}
    for ch, residues in (atoms_by_chain or {}).items():
        rns = sorted(residues)
        for idx, rn in enumerate(rns):
            r = residues[rn]
            resname = str(r.get("resname") or "")
            aa = _THREE_TO_ONE.get(resname, "X")
            if aa == "P":                                        # X→Pro on an existing Pro is a no-op
                continue
            prev_r = residues.get(rns[idx - 1]) if idx > 0 else None
            next_r = residues.get(rns[idx + 1]) if idx + 1 < len(rns) else None
            phi, psi = phi_psi(prev_r, r, next_r)
            if phi is None:                                      # no defined φ (terminus / break) → skip
                continue
            ph, ps = phi_score(phi), psi_score(psi)
            # the H-bond donor check needs the inferred amide H, which needs the PRECEDING residue's C
            donates, energy = False, None
            n, ca, c_prev = r.get("N"), r.get("CA"), (prev_r.get("C") if prev_r else None)
            if n and ca and c_prev:
                h = infer_amide_h(n, ca, c_prev)
                donates, energy = donates_backbone_hbond(n, h, acceptors, (ch, rn))
            hbond_factor = HBOND_PENALTY if donates else 1.0
            score = ph * ps * hbond_factor
            if score < min_score:
                continue
            ranked.append({
                "chain": ch, "position": rn, "from_aa": aa,
                "phi": phi, "psi": psi,
                "phi_score": round(ph, 4), "psi_score": round(ps, 4),
                "hbond_donates": bool(donates),
                "hbond_energy": (None if energy is None else round(energy, 3)),
                "score": round(score, 4),
            })
            best[(ch, rn)] = max(best.get((ch, rn), 0.0), round(score, 4))
    ranked.sort(key=lambda d: -d["score"])
    return ranked, best


def existing_prolines(atoms_by_chain: Dict[str, Dict[int, Dict[str, object]]]) -> List[Tuple[str, int]]:
    """``[(chain, resnum)]`` of residues that ARE proline — the 'see what's already there' design-context
    half (a simple highlight, not a cis/trans/strain assessment). PURE."""
    out: List[Tuple[str, int]] = []
    for ch, residues in (atoms_by_chain or {}).items():
        for rn, r in residues.items():
            if str(r.get("resname") or "").upper() == "PRO":
                out.append((ch, rn))
    return sorted(out)
