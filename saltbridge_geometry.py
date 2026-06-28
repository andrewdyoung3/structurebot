"""
saltbridge_geometry.py
----------------------
Plain-Python core for the SALT-BRIDGE stabilization suite — the fourth peer of
`disulfide_geometry` / `proline_geometry` / `cavity_geometry`, built fresh in the same
architecture (parses the mmCIF `_atom_site` loop, reuses the lifted `calc_distance` +
`place_from_internal`/`build_cb` + `ClashGrid` primitives, stays loadable without the
tool_router/bridge chain). Salt-bridge is PAIRWISE (like disulfide, not per-residue like
proline), so it inherits the disulfide two-chain pair shape: every pair carries its chain
PER MEMBER (`chain_a`/`chain_b`), intra-chain being the chain_a == chain_b special case, and
the interface scan is the cross-chain analogue — exactly the §9 universal-disulfide-mode
convergence, now extended to electrostatics.

THE BIOCHEMISTRY (what the scan measures — grounded in the primary literature, not invented):
A salt bridge is the electrostatic interaction between an acidic carboxylate (Asp/Glu Oδ/Oε)
and a basic group (Arg guanidinium Nη/Nε, Lys Nζ) at close range. The established geometry
convention (Barlow & Thornton 1983, J Mol Biol 168:867; Kumar & Nussinov 1999/2002) is the
CLOSEST carboxyl-O ↔ basic-N heavy-atom distance ≤ ~4.0 Å — adopted here as the HARD gate,
with a soft shoulder to ~5.0 Å so a 4.x Å near-miss surfaces as OPTIMIZABLE rather than
vanishing. An O–N within H-bond range (~3.5 Å) means the bridge is ALSO an H-bond (the
strongest sub-class) — surfaced as a DISPLAYED-NOT-RANKED flag, not a ranking gate (no clean
single angle cutoff exists for salt bridges the way χSS≈90° exists for disulfides).

THE NUANCE (why this is context-dependent, not "any charged pair at 4 Å is stabilizing"):
salt bridges carry a DESOLVATION penalty (burying charges on folding costs energy). A
continuum-electrostatics analysis (Hendsch & Tidor 1994) found a majority of salt bridges
electrostatically DEstabilizing once desolvation is counted; Kumar & Nussinov found most
stabilizing but GEOMETRY + burial-dependent. So the score is geometry × a BURIAL factor
(buried bridges weighted up — stronger unscreened interaction; surface bridges down — the
desolvation cost dominates), and the caveat is honest that the geometric scan cannot resolve
the full electrostatic/desolvation balance: it surfaces candidates, the designer judges
context, the re-fold validates.

His is DEFERRED from v1 (its charge is pH-dependent/ambiguous): the atom data is kept
(`HIS_ATOMS`) but His is neither a candidate nor a partner in the v1 scan.

Two halves (like disulfide assess-vs-engineer, proline existing-vs-suggest):
  • EXISTING-PAIR ASSESSMENT (lead, lowest-risk — pure measurement on real atoms): measure
    every existing Asp/Glu ↔ Arg/Lys pair's closest O–N + H-bond flag + burial; flag the
    4–5 Å near-misses as optimizable.
  • NOVEL SUGGESTION (engineering): all-pairs over candidate positions; place complementary
    charged groups (acid + base) over a χ1 reach-ring (the disulfide Sγ-reachability analog),
    ROTAMER-AWARE-clash (credit a reaching clash-free placement; soft-demote only when none
    dodges). Arg/Lys are long/flexible (large reach) → more reach successes, fewer clash flags
    than Cys.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from disulfide_geometry import (Vec3, ClashGrid, build_cb, calc_distance,
                                place_from_internal)

# ── CONSTANTS LITERATURE-AUDIT (2026-06-28) — pinned vs tunable (the durable split; PROJECT_CONTEXT §9) ─
#   LITERATURE-PINNED (a sourced fact — do NOT change without contradicting the source):
#     SB_DIST_CUTOFF 4.0 Å  — closest carboxyl-O↔basic-N salt-bridge cutoff (Barlow & Thornton 1983,
#                             JMB 168:867; Kumar & Nussinov 1999/2002)
#     SB_DIST_IDEAL 2.8 Å   — ion-pair / H-bond O···N separation
#     SB_HBOND_DIST 3.5 Å   — donor–acceptor heavy-atom H-bond cutoff (standard)
#     NEG_ATOMS/POS_ATOMS   — charged groups: Asp Oδ, Glu Oε, Arg Nε/Nη, Lys Nζ (Barlow & Thornton / Kumar & Nussinov)
#   TUNABLE (literature range/note in parens — tune from use):
#     SB_DIST_SHOULDER 5.0 (long-range ion pair quoted to ~6 Å) ; SB_DIST_MIN 2.2 (O···N steric ~2.2–2.4) ;
#     SB_BURIED_SASA 20 / SB_SURFACE_SASA 40 (burial cutoffs, calibration) ; SB_BURIAL_SURFACE_WEIGHT 0.5
#     (DIRECTION pinned — surface bridges marginal/destabilizing via desolvation, Hendsch & Tidor 1994,
#     Protein Sci 3:211; magnitude tunable) ; SB_CB_REACH D2.5/E3.9/K4.5/R5.5 (representative Cβ→group reach
#     approximations, χ2.. folded in — NOT measured; K=4.5 short vs extended Lys Cβ–Nζ ≈6.4, longer=more
#     permissive) ; SB_CBCB_SIGMA 2.2 ; SB_GEOM_AT_CUTOFF 0.6 ; SB_NOVEL_MIN_SCORE 0.6 ;
#     SALTBRIDGE_CLASH_PENALTY 0.6 ; SB_MIN_SCORE 0.05
#
# ── charged-group atoms (salvaged from the legacy salt_bridge_bridge; His kept as data only) ──────
# Acidic carboxylate oxygens / basic side-chain nitrogens — the Barlow&Thornton / Kumar&Nussinov
# heavy atoms the closest-pair distance is measured between. Arg includes Nε (part of the
# guanidinium) as well as the two Nη.
NEG_ATOMS: Dict[str, Tuple[str, ...]] = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
POS_ATOMS: Dict[str, Tuple[str, ...]] = {"ARG": ("NE", "NH1", "NH2"), "LYS": ("NZ",)}
HIS_ATOMS: Dict[str, Tuple[str, ...]] = {"HIS": ("ND1", "NE2")}     # DEFERRED (pH-dependent) — data kept

_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}
_ONE_TO_THREE = {v: k for k, v in _THREE_TO_ONE.items()}

# ── distance convention (tunable STARTING POINTS — tune from use, like the disulfide constants) ────
SB_DIST_MIN = 2.2          # below this the charged ATOMS sterically overlap → NOT a bridge (score 0):
                           # an O···N van der Waals contact bottoms out ~2.2–2.4 Å (salt-bridge H-bonds
                           # are ~2.5–3.2); sub-2.2 is atom interpenetration (the reach model would
                           # otherwise credit two groups placed on top of each other)
SB_DIST_IDEAL = 2.8        # ideal H-bonded ion-pair O–N separation (geometry plateau = 1.0)
SB_DIST_CUTOFF = 4.0       # HARD gate: closest carboxyl-O ↔ basic-N ≤ this = a salt bridge (B&T 1983)
SB_DIST_SHOULDER = 5.0     # SOFT shoulder: 4–5 Å surfaces as an OPTIMIZABLE near-miss (lower score)
SB_GEOM_AT_CUTOFF = 0.6    # geometry score AT the 4 Å cutoff (1.0 @ideal → this @cutoff → 0 @shoulder)
SB_HBOND_DIST = 3.5        # O–N ≤ this → the bridge is ALSO an H-bond (DISPLAYED-not-ranked flag)
SB_CBCB_SIGMA = 2.2        # σ of the Cβ–Cβ plausibility Gaussian (novel only): a NOVEL pair is ranked
                           # also by how natural its Cβ–Cβ separation is for the chosen side chains
                           # (centre = reach_a + reach_b — groups meeting between), which sharpens the
                           # ranking AND cuts the reach model's permissive tail to a sensible shortlist

# ── burial (load-bearing; reuses the legacy SASA thresholds — desolvation is the salt-bridge nuance) ─
SB_BURIED_SASA = 20.0      # mean residue SASA ≤ this → buried (full weight; stronger unscreened pair)
SB_SURFACE_SASA = 40.0     # ≥ this → fully surface (down-weighted; the desolvation cost dominates)
SB_BURIAL_SURFACE_WEIGHT = 0.5   # surface-bridge weight (Hendsch&Tidor: surface bridges marginal)
# A buried bridge keeps weight 1.0; the factor ramps linearly down to SB_BURIAL_SURFACE_WEIGHT.

SB_MIN_SCORE = 0.05        # EXISTING-pair surfacing floor — low, so 4–5 Å optimizable near-misses show
SB_NOVEL_MIN_SCORE = 0.6   # NOVEL surfacing floor — HIGHER: geometrically MANY sites can host a clash-
                           # free ideal bridge (salt bridges are easy to introduce), so the engineering
                           # shortlist is gated to strong candidates. With burial weighting this is
                           # buried-LEANING by construction — which matches the honesty framing (surface
                           # bridges are the marginal/desolvation-dominated ones). Tune from use.

# ── novel-suggestion reach model (the disulfide Sγ-reachability analog, for a charged group) ───────
# Each installable charged residue's charged-group centroid is placed over a χ1 sweep at its
# characteristic Cβ→group distance (the dominant rotational freedom; the inner χ2.. flexibility of
# the long side chains is folded into the reach distance — an APPROXIMATION, flagged in the caveat,
# NOT a full rotamer library). This reuses `place_from_internal` exactly as the Cys Sγ placement does.
SB_CB_REACH: Dict[str, float] = {"D": 2.5, "E": 3.9, "K": 4.5, "R": 5.5}  # Cβ→charged-group centroid (Å)
SB_REACH_ANGLE = 113.0     # CA–CB–(group) placement angle (deg, representative sp3)
SB_CHI1_STEP = 20.0        # χ1 sweep granularity (deg)
SB_CA_CA_PREFILTER = 16.0  # Cα–Cα beyond this → the two groups can't reach (lossless speed prefilter:
                           # max reach Glu+Arg ≈ 3.9+5.5 + 4 Å cutoff ≈ 13.4, padded)
SB_SCAN_MIN_SEQ_SEP = 2    # exclude i, i±1 only (i,i+3 / i,i+4 HELICAL salt bridges are kept)
SALTBRIDGE_CLASH_PENALTY = 0.6   # soft demotion when NO reaching rotamer pair dodges a clash (the
                                 # disulfide rotamer-aware-clash lesson: lower = stricter; flag, never veto)
SB_ACID = ("D", "E")
SB_BASE = ("R", "K")
SKIP_NOVEL = {"PRO", "CYS"}      # never propose at Pro (backbone) / Cys (may host a disulfide); the
                                 # already-charged D/E/R/K are excluded in the scan (already a partner)

# the union of atoms the one-pass parser captures: backbone (placement) + every charged atom + names
_WANT_ATOMS = ({"N", "CA", "CB", "C"}
               | {a for atoms in (*NEG_ATOMS.values(), *POS_ATOMS.values(), *HIS_ATOMS.values())
                  for a in atoms})


# ── parse: per-residue backbone + charged atoms + residue name (ONE pass; the scan's only parse) ──
def parse_residues(cif_path: str) -> Dict[str, Dict[int, Dict[str, object]]]:
    """``{chain: {resnum: {"resname": "ASP", "N"/"CA"/"CB"/"C"/charged-atoms: xyz}}}`` from a fold's
    mmCIF — backbone N/CA/CB/C (for the novel reach placement) + every Asp/Glu/Arg/Lys/His charged
    atom (for the existing-pair O–N distance) + the residue NAME (from_aa / charge classification).
    ONE `_atom_site` pass (the repo convention; robust column lookup, tolerant of column order). A
    residue simply lacks a key for a missing atom. PURE-ish (file read)."""
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
                    if atom not in _WANT_ATOMS:
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


# ── soft scores (PURE) ────────────────────────────────────────────────────────────────────────────
def geometry_score(d: Optional[float]) -> float:
    """The salt-bridge GEOMETRY term from a closest carboxyl-O ↔ basic-N distance *d* (Å): a WINDOW,
    not a one-sided ramp — 0 below SB_DIST_MIN (2.4 Å — the charged atoms sterically overlap), ramping
    up to the 1.0 plateau across [2.4, 2.8], holding 1.0 to the ideal, then SB_GEOM_AT_CUTOFF at the
    4.0 Å hard cutoff and a soft shoulder to 0 at 5.0 Å (so a 4.x Å near-miss surfaces as OPTIMIZABLE).
    0.0 outside / None."""
    if d is None or d < SB_DIST_MIN:
        return 0.0
    if d <= SB_DIST_IDEAL:
        return (d - SB_DIST_MIN) / (SB_DIST_IDEAL - SB_DIST_MIN)    # ramp up out of the steric floor
    if d <= SB_DIST_CUTOFF:
        return 1.0 - (1.0 - SB_GEOM_AT_CUTOFF) * (d - SB_DIST_IDEAL) / (SB_DIST_CUTOFF - SB_DIST_IDEAL)
    if d <= SB_DIST_SHOULDER:
        return SB_GEOM_AT_CUTOFF * (SB_DIST_SHOULDER - d) / (SB_DIST_SHOULDER - SB_DIST_CUTOFF)
    return 0.0


def burial_factor(sasa: Optional[float]) -> float:
    """The BURIAL weight from a (mean) residue SASA: 1.0 for a buried pair (≤20 Å² — full, unscreened
    interaction), ramping down to SB_BURIAL_SURFACE_WEIGHT for a fully-surface pair (≥40 Å² — the
    desolvation cost dominates). None (SASA unavailable) → 1.0 NEUTRAL (burial unknown — never fabricate
    a burial signal; the score is then geometry-only and the caveat says so). PURE."""
    if sasa is None:
        return 1.0
    if sasa <= SB_BURIED_SASA:
        return 1.0
    if sasa >= SB_SURFACE_SASA:
        return SB_BURIAL_SURFACE_WEIGHT
    t = (sasa - SB_BURIED_SASA) / (SB_SURFACE_SASA - SB_BURIED_SASA)
    return 1.0 - t * (1.0 - SB_BURIAL_SURFACE_WEIGHT)


def _charged_atoms(res: Dict[str, object]) -> Tuple[str, List[Vec3]]:
    """``(sign, [charged-atom xyz])`` for a residue: sign ∈ {"+","-",""}. Only Asp/Glu (−) and Arg/Lys
    (+) — His returns "" (deferred). PURE."""
    rn = str(res.get("resname") or "").upper()
    if rn in NEG_ATOMS:
        return "-", [res[a] for a in NEG_ATOMS[rn] if a in res]
    if rn in POS_ATOMS:
        return "+", [res[a] for a in POS_ATOMS[rn] if a in res]
    return "", []


def _closest_on(atoms_a: List[Vec3], atoms_b: List[Vec3]) -> Optional[float]:
    """Closest atom-pair distance between two charged-atom sets (the carboxyl-O ↔ basic-N convention).
    None if either set is empty. PURE."""
    if not atoms_a or not atoms_b:
        return None
    return min(calc_distance(x, y) for x in atoms_a for y in atoms_b)


def _mean_sasa(sasa_map, key_a, key_b) -> Optional[float]:
    """Mean SASA of the two residues, or None if the map is absent or missing either (→ neutral burial)."""
    if not sasa_map:
        return None
    sa, sb = sasa_map.get(key_a), sasa_map.get(key_b)
    if sa is None or sb is None:
        return None
    return (float(sa) + float(sb)) / 2.0


# ── EXISTING-PAIR ASSESSMENT (lead mode — pure measurement on real atoms, no placement) ───────────
def scan_existing_pairs(residues_by_chain: Dict[str, Dict[int, Dict[str, object]]], *,
                        sasa_map: Optional[Dict[Tuple[str, int], float]] = None,
                        min_score: float = SB_MIN_SCORE):
    """Measure every existing Asp/Glu ↔ Arg/Lys pair (intra + inter-chain) whose closest carboxyl-O ↔
    basic-N distance is within the soft shoulder. Returns ``(ranked, best_partner)``: ranked pair dicts
    (sorted by score desc) carrying the two-chain shape (chain_a/resnum_a + chain_b/resnum_b),
    from_aa_a/from_aa_b, the measured `on_dist`, `within_cutoff` (≤4 Å = a salt bridge) / `optimizable`
    (4–5 Å near-miss), `hbond_like` (≤3.5 Å), burial, and ``score = geometry × burial``. PURE-ish."""
    # split into +/- residues across ALL chains (inter-chain pairs are first-class)
    pos: List[Tuple[str, int, str, List[Vec3]]] = []     # (chain, resnum, from_aa, atoms)
    neg: List[Tuple[str, int, str, List[Vec3]]] = []
    for ch, residues in (residues_by_chain or {}).items():
        for rn, res in residues.items():
            sign, atoms = _charged_atoms(res)
            if not atoms:
                continue
            aa = _THREE_TO_ONE.get(str(res.get("resname") or "").upper(), "X")
            (pos if sign == "+" else neg).append((ch, rn, aa, atoms))

    ranked: List[Dict[str, object]] = []
    best: Dict[tuple, float] = {}
    for (cp, rp, aap, ap) in pos:
        for (cn, rn_, aan, an) in neg:
            d = _closest_on(ap, an)
            if d is None or d > SB_DIST_SHOULDER:
                continue
            ms = _mean_sasa(sasa_map, (cp, rp), (cn, rn_))
            score = geometry_score(d) * burial_factor(ms)
            if score < min_score:
                continue
            # canonical member order: A-side = the acid (negative), B-side = the base (positive) — a
            # stable, readable convention (Asp/Glu ↔ Arg/Lys); the pair is symmetric for distance.
            pair = {
                "chain_a": cn, "resnum_a": rn_, "from_aa_a": aan,
                "chain_b": cp, "resnum_b": rp, "from_aa_b": aap,
                "type": f"{aan}-{aap}",
                "on_dist": round(d, 2),
                "within_cutoff": d <= SB_DIST_CUTOFF,
                "optimizable": SB_DIST_CUTOFF < d <= SB_DIST_SHOULDER,
                "hbond_like": d <= SB_HBOND_DIST,
                "mean_sasa": (None if ms is None else round(ms, 1)),
                "buried": (None if ms is None else ms <= SB_BURIED_SASA),
                "burial_factor": round(burial_factor(ms), 3),
                "interchain": cn != cp,
                "score": round(score, 4),
            }
            ranked.append(pair)
            best[(cn, rn_)] = max(best.get((cn, rn_), 0.0), pair["score"])
            best[(cp, rp)] = max(best.get((cp, rp), 0.0), pair["score"])
    ranked.sort(key=lambda p: -p["score"])
    return ranked, best


# ── NOVEL SUGGESTION (engineering mode — place complementary charged groups, rotamer-aware clash) ──
def _chi1_grid() -> List[float]:
    n = int(round(360.0 / SB_CHI1_STEP))
    return [-180.0 + SB_CHI1_STEP * i for i in range(n)]


def _cbcb_score(cb_cb: float, reach_a: float, reach_b: float) -> float:
    """Cβ–Cβ plausibility for a NOVEL pair (the disulfide Cβ–Cβ-window analog): a Gaussian centred at
    ``reach_a + reach_b`` (the two side chains meeting between the backbones) with σ = SB_CBCB_SIGMA.
    Rewards a geometrically-natural separation for the chosen charged residues and damps the reach
    model's permissive tail (rings that only ‘meet’ at extreme stretch). PURE."""
    mu = reach_a + reach_b
    return math.exp(-((cb_cb - mu) ** 2) / (2.0 * SB_CBCB_SIGMA * SB_CBCB_SIGMA))


def _resolve_cb(res: Dict[str, object]) -> Optional[Vec3]:
    """The residue's Cβ — real, or reconstructed for Gly from N/CA/C (`build_cb`), else None (no
    placement possible). Mirrors the disulfide Cβ resolution so a Gly site is scored, not dropped."""
    cb = res.get("CB")
    if cb is not None:
        return cb            # type: ignore[return-value]
    n, ca, c = res.get("N"), res.get("CA"), res.get("C")
    if n and ca and c:
        return build_cb(n, ca, c)    # type: ignore[arg-type]
    return None


def charged_reach_ring(res: Dict[str, object], reach: float) -> List[Vec3]:
    """The charged-group centroid positions over a χ1 sweep for an installed charged residue at this
    backbone — `place_from_internal(N, CA, CB, reach, angle, χ1)` for every χ1 (the disulfide Sγ-ring
    analog, with *reach* = the Cβ→group distance). [] if N/CA/Cβ can't be resolved. PURE-ish."""
    n, ca = res.get("N"), res.get("CA")
    cb = _resolve_cb(res)
    if not (n and ca and cb):
        return []
    return [place_from_internal(n, ca, cb, reach, SB_REACH_ANGLE, chi) for chi in _chi1_grid()]


def _reach_pair(ring_a: List[Vec3], ring_b: List[Vec3], clash_checker=None) -> Dict[str, object]:
    """Best reachable charged-group separation over the two χ1 rings (the disulfide `sg_reachability`
    analog). Tracks the geometry-best pair (any clash) and — with a *clash_checker(pa, pb)→bool* — the
    geometry-best REACHING (geometry_score > 0) pair that is also CLASH-FREE; reports the clash-free
    best if one exists (clash False), else, if reaching pairs all collide, the best reaching pair with
    clash True (genuinely unviable), else the geometry-best with clash False (a REACH miss, not a clash
    — the low score says so). ``{reach_score, best_on, clash}``. Soft (never hard-zero). PURE-ish."""
    best_all = (-1.0, None)          # (q, d)
    best_clean = (-1.0, None)
    any_reach = False
    for pa in ring_a:
        for pb in ring_b:
            d = calc_distance(pa, pb)
            q = geometry_score(d)
            if q > best_all[0]:
                best_all = (q, d)
            if q > 0.0:
                any_reach = True
                if clash_checker is not None and q > best_clean[0] and not clash_checker(pa, pb):
                    best_clean = (q, d)
    if clash_checker is None:
        q, d, clash = best_all[0], best_all[1], None
    elif best_clean[1] is not None:
        q, d, clash = best_clean[0], best_clean[1], False
    elif any_reach:
        q, d, clash = best_all[0], best_all[1], True          # reaching pairs exist but all collide
    else:
        q, d, clash = best_all[0], best_all[1], False         # never reached → a reach miss, not a clash
    return {"reach_score": round(max(q, 0.0), 4),
            "best_on": (None if d is None else round(d, 3)), "clash": clash}


def _make_clash_checker(grid: Optional[ClashGrid], exclude: set):
    """The per-pair clash predicate the reach sweep calls: ``checker(pa, pb) → bool`` — do EITHER placed
    charged groups overlap a heavy atom (treated as N — the charged-group terminus) outside the two
    mutated residues. None if no grid."""
    if grid is None:
        return None
    exc = exclude or set()

    def checker(pa: Vec3, pb: Vec3) -> bool:
        return grid.any_clash([(pa, "N"), (pb, "N")], exc)
    return checker


def _best_charge_assignment(res_a, res_b, key_a, key_b, *, sasa_map, clash_grid):
    """Over the acid×base residue choices AND the two A/B assignments, the best installable charged pair
    for two candidate positions. Returns the best candidate dict (to_aa_a/to_aa_b, reach, clash, score)
    or None (nothing reaches above the floor). The reach ring is built ONCE per (residue, reach) and
    reused across assignments. PURE-ish."""
    ms = _mean_sasa(sasa_map, key_a, key_b)
    bf = burial_factor(ms)
    cb_a, cb_b = _resolve_cb(res_a), _resolve_cb(res_b)
    if cb_a is None or cb_b is None:
        return None
    cb_cb = calc_distance(cb_a, cb_b)
    checker = _make_clash_checker(clash_grid, {key_a, key_b})
    rings_a: Dict[str, List[Vec3]] = {}
    rings_b: Dict[str, List[Vec3]] = {}

    def ring(res, cache, aa):
        if aa not in cache:
            cache[aa] = charged_reach_ring(res, SB_CB_REACH[aa])
        return cache[aa]

    best: Optional[Dict[str, object]] = None
    # assignment 1: A = acid, B = base ; assignment 2: A = base, B = acid
    for ta_set, tb_set in ((SB_ACID, SB_BASE), (SB_BASE, SB_ACID)):
        for ta in ta_set:
            ra = ring(res_a, rings_a, ta)
            if not ra:
                continue
            for tb in tb_set:
                rb = ring(res_b, rings_b, tb)
                if not rb:
                    continue
                reach = _reach_pair(ra, rb, clash_checker=checker)
                rscore = float(reach["reach_score"])
                if rscore <= 0.0:
                    continue
                cbcb = _cbcb_score(cb_cb, SB_CB_REACH[ta], SB_CB_REACH[tb])
                score = rscore * cbcb * bf
                if reach["clash"] is True:
                    score *= SALTBRIDGE_CLASH_PENALTY
                if best is None or score > float(best["score"]):
                    best = {"to_aa_a": ta, "to_aa_b": tb,
                            "best_on": reach["best_on"], "reach_score": reach["reach_score"],
                            "cb_cb": round(cb_cb, 2), "cbcb_score": round(cbcb, 4),
                            "clash": reach["clash"],
                            "hbond_like": (reach["best_on"] is not None and reach["best_on"] <= SB_HBOND_DIST),
                            "mean_sasa": (None if ms is None else round(ms, 1)),
                            "buried": (None if ms is None else ms <= SB_BURIED_SASA),
                            "burial_factor": round(bf, 3),
                            "score": round(score, 4)}
    return best


def _novel_candidate_residues(residues: Dict[int, Dict[str, object]]):
    """``[(resnum, from_aa)]`` of positions ELIGIBLE for a novel charged substitution in a chain:
    not already charged (D/E/R/K), not Pro/Cys, with a resolvable Cα. PURE."""
    out = []
    for rn, res in residues.items():
        name = str(res.get("resname") or "").upper()
        aa = _THREE_TO_ONE.get(name, "X")
        if aa in ("D", "E", "R", "K") or name in SKIP_NOVEL:
            continue
        if not res.get("CA"):
            continue
        out.append((rn, aa))
    return out


def scan_novel_sites(residues_by_chain: Dict[str, Dict[int, Dict[str, object]]], *,
                     clash_grid: Optional[ClashGrid] = None,
                     sasa_map: Optional[Dict[Tuple[str, int], float]] = None,
                     min_score: float = SB_NOVEL_MIN_SCORE):
    """INTRA-chain all-pairs scan for NOVEL installable salt-bridge sites: per chain, every eligible
    position pair separated by ≥ SB_SCAN_MIN_SEQ_SEP, the best complementary charged pair that reaches
    salt-bridge geometry clash-free (`_best_charge_assignment`). A cheap Cα–Cα PREFILTER skips pairs too
    far for the groups to reach (lossless). Returns ``(ranked, best_partner)``: ranked candidate dicts
    (chain_a == chain_b — intra) with to_aa_a/to_aa_b + from_aa + reach/clash/burial/score; best_partner
    ``{(chain, resnum): best score}`` for the heatmap. PURE-ish."""
    ranked: List[Dict[str, object]] = []
    best: Dict[tuple, float] = {}
    for ch, residues in (residues_by_chain or {}).items():
        cands = _novel_candidate_residues(residues)
        cands.sort(key=lambda t: t[0])
        for i in range(len(cands)):
            ra, aa_a = cands[i]
            ca_a = residues[ra].get("CA")
            for j in range(i + 1, len(cands)):
                rb, aa_b = cands[j]
                if abs(rb - ra) < SB_SCAN_MIN_SEQ_SEP:
                    continue
                ca_b = residues[rb].get("CA")
                if SB_CA_CA_PREFILTER is not None and calc_distance(ca_a, ca_b) > SB_CA_CA_PREFILTER:
                    continue
                cand = _best_charge_assignment(residues[ra], residues[rb], (ch, ra), (ch, rb),
                                               sasa_map=sasa_map, clash_grid=clash_grid)
                if cand is None or float(cand["score"]) < min_score:
                    continue
                row = {"chain_a": ch, "resnum_a": ra, "from_aa_a": aa_a,
                       "chain_b": ch, "resnum_b": rb, "from_aa_b": aa_b,
                       "ca_ca": round(calc_distance(ca_a, ca_b), 2), "interchain": False, **cand}
                ranked.append(row)
                best[(ch, ra)] = max(best.get((ch, ra), 0.0), float(cand["score"]))
                best[(ch, rb)] = max(best.get((ch, rb), 0.0), float(cand["score"]))
    ranked.sort(key=lambda d: -d["score"])
    return ranked, best


def scan_novel_interface(residues_by_chain: Dict[str, Dict[int, Dict[str, object]]], *,
                         clash_grid: Optional[ClashGrid] = None,
                         sasa_map: Optional[Dict[Tuple[str, int], float]] = None,
                         min_score: float = SB_NOVEL_MIN_SCORE):
    """CROSS-chain (inter-subunit) NOVEL salt-bridge scan — the interface analogue of
    `scan_novel_sites`, reusing the SAME `_best_charge_assignment` primitive (no new geometry loop). The
    Cα–Cα PREFILTER IS the interface bound (a pair close enough across chains to bridge is interface-
    proximal). No min_seq_sep (cross-chain residues have no sequence adjacency). Returns ``(ranked,
    best_partner)`` with chain_a != chain_b — feeds the (constraint-free) re-fold validation. PURE-ish."""
    ranked: List[Dict[str, object]] = []
    best: Dict[tuple, float] = {}
    chains = sorted(residues_by_chain or {})
    for ii in range(len(chains)):
        for jj in range(ii + 1, len(chains)):
            cha, chb = chains[ii], chains[jj]
            ca_list = _novel_candidate_residues(residues_by_chain[cha])
            cb_list = _novel_candidate_residues(residues_by_chain[chb])
            for ra, aa_a in ca_list:
                ca_a = residues_by_chain[cha][ra].get("CA")
                for rb, aa_b in cb_list:
                    ca_b = residues_by_chain[chb][rb].get("CA")
                    if SB_CA_CA_PREFILTER is not None and calc_distance(ca_a, ca_b) > SB_CA_CA_PREFILTER:
                        continue
                    cand = _best_charge_assignment(residues_by_chain[cha][ra], residues_by_chain[chb][rb],
                                                   (cha, ra), (chb, rb),
                                                   sasa_map=sasa_map, clash_grid=clash_grid)
                    if cand is None or float(cand["score"]) < min_score:
                        continue
                    row = {"chain_a": cha, "resnum_a": ra, "from_aa_a": aa_a,
                           "chain_b": chb, "resnum_b": rb, "from_aa_b": aa_b,
                           "ca_ca": round(calc_distance(ca_a, ca_b), 2), "interchain": True, **cand}
                    ranked.append(row)
                    best[(cha, ra)] = max(best.get((cha, ra), 0.0), float(cand["score"]))
                    best[(chb, rb)] = max(best.get((chb, rb), 0.0), float(cand["score"]))
    ranked.sort(key=lambda d: -d["score"])
    return ranked, best


# ── burial map (FreeSASA — a DECLARED project dependency; guarded so the core imports without it) ──
def compute_sasa_map(cif_path: str) -> Dict[Tuple[str, int], float]:
    """Per-residue total SASA ``{(chain, resnum): Å²}`` for the burial factor, via FreeSASA (declared:
    requirements.txt) on a BioPython-parsed mmCIF. GUARDED: returns {} on any failure (FreeSASA/
    BioPython absent, parse error) → the scan falls back to NEUTRAL burial (geometry-only, honestly
    flagged) rather than crashing. Kept here (not in the pure scan) so the geometry functions stay
    dependency-free + unit-testable with a synthetic sasa_map."""
    try:
        import freesasa
        from Bio.PDB import MMCIFParser
    except Exception:
        return {}
    try:
        structure = MMCIFParser(QUIET=True).get_structure("s", cif_path)
        fs = freesasa.structureFromBioPDB(structure)
        res = freesasa.calc(fs)
        areas = res.residueAreas()                          # {chain: {resnum_str: ResidueArea}}
        out: Dict[Tuple[str, int], float] = {}
        for ch, resmap in areas.items():
            for rn_str, area in resmap.items():
                try:
                    out[(ch, int(rn_str))] = float(area.total)
                except (ValueError, TypeError):
                    continue
        return out
    except Exception:
        return {}
