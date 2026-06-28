"""
cavity_geometry.py
-------------------
Plain-Python core for the CAVITY-FILLING stabilization suggestion scan — the third sibling of
`disulfide_geometry.py` / `proline_geometry.py`, built fresh in the same architecture (NO new
dependency: it reads the mmCIF `_atom_site` loop, reuses the lifted `calc_distance` /
`place_from_internal` / `build_cb` primitives + the `ClashGrid` spatial hash + `parse_heavy_atoms`
from `disulfide_geometry`, and stays loadable without the tool_router/bridge chain).

THE BIOPHYSICS (what the scan measures — grounded in the primary literature, not invented):
Buried internal cavities are packing defects — a residue with a smaller-than-optimal side chain
leaves a void that costs van-der-Waals/packing energy. Filling such a void with a larger hydrophobic
side chain can RECOVER that packing energy. The literature is two-sided and the tool says so:
  • GENERIC thermostability — modest, variable gains, often offset by STRAIN when the introduced
    side chain must adopt a non-optimal rotamer (T4 lysozyme: Karpusas/Baase/Matthews, PNAS 1989;
    apoflavodoxin: Machicado et al., JMB 2006 — measured 0.0–0.6 kcal/mol, sub-~20 Å³ voids
    destabilize REGARDLESS of the residue). This is the cautionary half.
  • CONFORMATIONAL stabilization — locking a SPECIFIC state against an alternative — is a proven,
    powerful technique: cavity-filling is central to the RSV prefusion-F vaccines (the McLellan/
    Graham DS-Cav1 lineage). The strategically-valuable cavities are the ones in conformationally-
    important regions, which a geometric scan CANNOT identify but the designer can.
So the tool SURFACES geometrically-viable fills with honest metrics (void volume, fill fraction,
clash); the human supplies the structural judgement about which cavity matters.

METHOD (the established probe-sphere / Richards–Connolly tradition, plain-Python):
  1. DETECT — a coarse 3D grid over the structure; a grid point is "occupied" if within (atom vdW +
     PROBE_RADIUS) of any heavy atom (the 1.4 Å water-probe convention, VOIDOO/CASTp). Flood the
     SOLVENT inward from the box exterior through unoccupied cells; unoccupied cells the solvent
     CANNOT reach are ENCLOSED (internal) voids. Connected components ≥ MIN_VOLUME are cavities.
  2. FILL — for each cavity-lining residue with an allowed small→larger enlargement
     (`_VOLUME_MUTATIONS`, mirroring the legacy `cavity_bridge` table), place the larger side chain's
     bulk over a χ1(×χ2) rotamer sweep (NeRF `place_from_internal`, `build_cb` for Gly) and credit it
     ONLY if a rotamer reaches INTO the void AND is clash-free against the walls (ROTAMER-AWARE, the
     disulfide reachability lesson). `score = fill_fraction × reach_score`, soft-demoted by
     OVERFILL_PENALTY when every reaching rotamer collides (no rigid-backbone-viable rotamer dodges).

DISTINCT from the legacy `cavity_bridge.py`, which stays PARALLEL (§9): that tool DETECTS by a SASA-
burial + Cα-cluster PROXY (no true void geometry) and SUGGESTS by a static volume-table lookup (no
rotamer placement / clash check). This core does true geometric void detection + rotamer-aware fill —
the proxy→geometry upgrade the disulfide mode made (orientation proxy → Sγ-reachability).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from disulfide_geometry import (
    Vec3, calc_distance, place_from_internal, build_cb, ClashGrid, HEAVY_VDW, parse_heavy_atoms,
)

# ── CONSTANTS LITERATURE-AUDIT (2026-06-28) — pinned vs tunable (the durable split; PROJECT_CONTEXT §9) ─
#   LITERATURE-PINNED (a sourced fact — do NOT change without contradicting the source):
#     PROBE_RADIUS 1.4 Å    — water-probe convention (VOIDOO: Kleywegt & Jones 1994, Acta Cryst D50:178; CASTp)
#     MIN_VOLUME 20 Å³      — sub-~20 Å³ voids destabilize regardless (Machicado et al. 2006, JMB) ≈ one methyl
#     _CC_BOND 1.52 Å / _CC_ANGLE 110.5° — standard sp³ C–C (Engh & Huber 1991)
#   TUNABLE (literature range/note in parens — tune from use):
#     GRID_SPACING 0.5 (VOIDOO ~0.33; coarser for plain-Python speed) ; LINING_RADIUS 4.5 ; OVERFILL_PENALTY
#     0.6 ; VOLUME_GAIN_CAP 50 ; CHI1_STEP 20° / CHI2_SET (sampling) ; _SC_VOLUME{} / _VOLUME_MUTATIONS{}
#     (Zamyatnin 1972 / Richards 1974 side-chain volumes — used RELATIVELY, absolute calibration NOT load-
#     bearing ‡ ; the enlargement map is a conservative curation policy)
#   ‡ the specific _SC_VOLUME values are standard but were not linked to a primary-source page (flagged).
#
# ── tunable scoring constants (STARTING POINTS — tune from use, like the disulfide/proline ones) ──
PROBE_RADIUS = 1.4          # rolling-probe radius (Å) — the classic water probe (VOIDOO/CASTp). A
                            # void must admit a probe CENTRE this size to seed (smaller interstitial
                            # packing slack is excluded). TUNE-FROM-USE: 1.4 is the literature convention
                            # and finds the discrete internal cavities of cavity-rich folds (e.g. the
                            # globins) while correctly showing none on very well-packed small domains;
                            # dropping toward 1.2 (within the cited range) surfaces more, smaller
                            # fillable packing defects if a structure looks under-sensitive.
GRID_SPACING = 0.5          # detection grid spacing (Å) — coarser than VOIDOO's 0.33 for plain-Python
                            # speed; the resolution/cost lever (smaller = finer voids, slower).
MIN_VOLUME = 20.0           # a void below ~one comfortable methyl (≈20 Å³) is sub-threshold NOISE —
                            # the literature says filling these destabilizes REGARDLESS (Machicado);
                            # HARD-skip them. This is the only volume gate (no large-void-only curation).
LINING_RADIUS = 4.5         # a residue is cavity-LINING if a side-chain/Cβ atom is within this of the void.
OVERFILL_PENALTY = 0.6      # soft score multiplier when a fill REACHES the void but EVERY reaching
                            # rotamer clashes the walls (no rigid-backbone-viable rotamer dodges) —
                            # the rigid backbone may still relax, so demote, never hard-eliminate
                            # (mirrors the disulfide CLASH_PENALTY; lower = stricter — do not invert).
VOLUME_GAIN_CAP = 50.0      # cap (Å³) on the absolute volume-gain reward, so a big enlargement can't
                            # run away purely on size (the void-relative fill_fraction carries the rest).
MAX_GRID_CELLS = 6_000_000  # safety valve: if the bounding box would exceed this many cells, COARSEN
                            # the spacing (recorded) rather than hang for minutes on a large assembly.

# rotamer sweep (place the larger side chain's bulk; FINE χ1 so "no rotamer reaches" is real, not
# "none of a few staggered did" — the same honesty hinge as the disulfide χ1 sweep)
CHI1_STEP = 20.0            # χ1 sweep granularity (deg)
CHI2_SET = (-60.0, 60.0, 180.0)   # coarse χ2 (only the γ-then-δ targets I/L use it)
_CC_BOND = 1.52            # idealized C–C side-chain bond (Å)
_CC_ANGLE = 110.5          # idealized tetrahedral C–C–C angle (deg)

_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Side-chain volumes (Å³, approximate) + allowed conservative one-step hydrophobic ENLARGEMENTS.
# These MIRROR the legacy `cavity_bridge._SC_VOLUME` / `_VOLUME_MUTATIONS` (the gate's "reuse
# _VOLUME_MUTATIONS"), lifted here so this core stays self-contained (the legacy tool's BioPython
# import chain is not pulled in just for two dicts) — the same lift-not-reach pattern as the
# `calc_distance` primitive. Conservative by design: only small→larger packing-compatible steps.
_SC_VOLUME: Dict[str, float] = {
    "G": 0.0, "A": 67.0, "S": 73.0, "T": 93.0, "V": 105.0, "L": 124.0, "I": 124.0, "M": 124.0,
    "P": 90.0, "F": 135.0, "Y": 141.0, "W": 163.0, "C": 86.0, "D": 91.0, "E": 109.0, "N": 96.0,
    "Q": 114.0, "K": 135.0, "R": 148.0, "H": 118.0,
}
_VOLUME_MUTATIONS: Dict[str, List[str]] = {
    "G": ["A"], "A": ["V", "I", "L"], "S": ["T", "V"], "T": ["V", "I"], "V": ["I", "L"],
}

# Per-target side-chain bulk builders — the EXTRA carbons a larger side chain adds beyond a small WT,
# placed by NeRF from the backbone (N–CA–CB) over the rotamer sweep. A geometric FILL PROBE, NOT a
# real rotamer library: a handful of representative γ/δ carbons whose job is to test "does the added
# bulk reach into the void without clashing the walls". Each step = (refs, bond, angle, χ-base, offset)
# where refs name earlier atoms ('N','CA','CB' or a built probe like 'p0'); χ-base ∈ {'chi1','chi2'}.
_PROBE_SPECS: Dict[str, List[tuple]] = {
    # A: G→A adds only Cβ — the built Cβ itself is the probe (handled specially, no steps here).
    "A": [],
    "V": [("N", "CA", "CB", _CC_BOND, _CC_ANGLE, "chi1", 0.0),       # CG1
          ("N", "CA", "CB", _CC_BOND, _CC_ANGLE, "chi1", 120.0)],    # CG2 (β-branch)
    "T": [("N", "CA", "CB", _CC_BOND, _CC_ANGLE, "chi1", 120.0)],    # CG2 (the added γ-methyl)
    "L": [("N", "CA", "CB", _CC_BOND, 116.0, "chi1", 0.0),           # CG
          ("CA", "CB", "p0", _CC_BOND, _CC_ANGLE, "chi2", 0.0),      # CD1
          ("CA", "CB", "p0", _CC_BOND, _CC_ANGLE, "chi2", 120.0)],   # CD2
    "I": [("N", "CA", "CB", _CC_BOND, _CC_ANGLE, "chi1", 0.0),       # CG1
          ("N", "CA", "CB", _CC_BOND, _CC_ANGLE, "chi1", -120.0),    # CG2 (β-branch)
          ("CA", "CB", "p0", _CC_BOND, 113.8, "chi2", 0.0)],         # CD1 (off CG1)
}


# ── mmCIF parse: per-residue backbone N/CA/C + Cβ + residue name (the lining/fill half's parse) ──
def parse_residue_atoms(cif_path: str) -> Dict[str, Dict[int, Dict[str, object]]]:
    """``{chain: {resnum: {"N","CA","C","CB": xyz, "resname": "ALA"}}}`` from a fold's mmCIF — the
    backbone (for rotamer placement) + Cβ (real, or absent for Gly → `build_cb` reconstructs it) +
    the residue NAME (for from_aa / the allowed-enlargement lookup). ONE `_atom_site` pass (the repo
    convention; robust column lookup, tolerant of column order). PURE-ish (file read)."""
    want = {"N", "CA", "C", "CB"}
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
                    res.setdefault(atom, xyz)
                    res.setdefault("resname", (col.get("label_comp_id") or "").upper())
                elif in_loop and s and not s.startswith(("ATOM", "HETATM", "_atom_site.")):
                    if out:
                        break
    except OSError:
        return {}
    return out


# ── cavity detection: grid + probe-sphere occupancy + exterior solvent flood (the §9 new shape) ──
class _Grid:
    """A coarse 3D occupancy grid over the structure's bounding box. Cell (i,j,k) center is
    ``origin + (i+0.5, j+0.5, k+0.5)·spacing``. Occupancy/flood are flat `bytearray`s indexed
    ``(i·ny + j)·nz + k`` (memory-lean; ≤ MAX_GRID_CELLS by the safety-coarsen guard)."""

    def __init__(self, origin: Vec3, spacing: float, nx: int, ny: int, nz: int):
        self.origin, self.spacing = origin, spacing
        self.nx, self.ny, self.nz = nx, ny, nz

    def idx(self, i: int, j: int, k: int) -> int:
        return (i * self.ny + j) * self.nz + k

    def cell_of(self, xyz: Vec3) -> Tuple[int, int, int]:
        return (int((xyz[0] - self.origin[0]) / self.spacing),
                int((xyz[1] - self.origin[1]) / self.spacing),
                int((xyz[2] - self.origin[2]) / self.spacing))

    def center(self, i: int, j: int, k: int) -> Vec3:
        return (self.origin[0] + (i + 0.5) * self.spacing,
                self.origin[1] + (j + 0.5) * self.spacing,
                self.origin[2] + (k + 0.5) * self.spacing)


def _build_grid(heavy_atoms, spacing: float, probe: float) -> Tuple[_Grid, float]:
    """Bounding-box grid sized to the structure + a margin (so the exterior shell is all-solvent and
    seeds the flood). Auto-COARSENS the spacing if the cell count would exceed MAX_GRID_CELLS (the
    big-assembly safety valve) and returns the effective spacing actually used."""
    xs = [a[3][0] for a in heavy_atoms]
    ys = [a[3][1] for a in heavy_atoms]
    zs = [a[3][2] for a in heavy_atoms]
    max_r = max(HEAVY_VDW.values()) + probe
    margin = max_r + 2.0 * spacing
    lo = (min(xs) - margin, min(ys) - margin, min(zs) - margin)
    hi = (max(xs) + margin, max(ys) + margin, max(zs) + margin)
    span = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    eff = spacing
    while True:
        nx = max(1, int(math.ceil(span[0] / eff)))
        ny = max(1, int(math.ceil(span[1] / eff)))
        nz = max(1, int(math.ceil(span[2] / eff)))
        if nx * ny * nz <= MAX_GRID_CELLS:
            break
        eff *= 1.26                                    # ≈ cube-root step; coarsen until it fits
    return _Grid(lo, eff, nx, ny, nz), eff


def detect_cavities(heavy_atoms, *, probe: float = PROBE_RADIUS, spacing: float = GRID_SPACING,
                    min_volume: float = MIN_VOLUME, lining_radius: float = LINING_RADIUS,
                    residues: Optional[Dict[str, Dict[int, Dict[str, object]]]] = None) -> List[dict]:
    """Detect INTERNAL (enclosed) cavities — the solvent-EXCLUDED void convention (VOIDOO/CASTp), so
    the reported volume is the vdW-empty cavity space, not just where a probe CENTRE fits. *heavy_atoms*
    = the flat ``[(chain, resnum, element, xyz)]`` from `parse_heavy_atoms`. Three cell classes:
      • SOLID   — within an atom's vdW radius (the protein; never void).
      • BLOCKED — within (vdW + *probe*): a probe CENTRE can't sit here.
      • the SOLVENT (exterior probe-accessible region) = flood(not BLOCKED) from the box boundary.
    A cell is CAVITY iff it is empty (not SOLID) AND the exterior probe surface can't reach it (no
    exterior-accessible probe centre within *probe* of it) — i.e. vdW-free space sealed off from bulk.
    This catches small sealed packing defects (no room for an enclosed probe centre) AND large voids.
    6-connected components with volume ≥ *min_volume* are cavities. Each cavity dict: ``{cavity_id,
    volume, centroid, cells:set[(i,j,k)], grid:_Grid, lining:[(chain,resnum,resname)], is_interface}``.
    *residues* (from `parse_residue_atoms`) supplies the lining identity. PURE-ish."""
    if not heavy_atoms:
        return []
    grid, eff = _build_grid(heavy_atoms, spacing, probe)
    nx, ny, nz, total = grid.nx, grid.ny, grid.nz, grid.nx * grid.ny * grid.nz
    solid = bytearray(total)        # within vdW — protein
    blocked = bytearray(total)      # within vdW+probe — no room for a probe centre

    # 1) rasterize — mark BLOCKED within (vdW+probe) and SOLID within vdW, in one atom pass
    for (_ch, _rn, el, xyz) in heavy_atoms:
        vdw = HEAVY_VDW.get(el, 1.70)
        r = vdw + probe
        ci, cj, ck = grid.cell_of(xyz)
        span = int(math.ceil(r / eff)) + 1
        r2, v2 = r * r, vdw * vdw
        for i in range(max(0, ci - span), min(nx, ci + span + 1)):
            cx = grid.origin[0] + (i + 0.5) * eff
            dx2 = (cx - xyz[0]) ** 2
            if dx2 > r2:
                continue
            for j in range(max(0, cj - span), min(ny, cj + span + 1)):
                cy = grid.origin[1] + (j + 0.5) * eff
                dxy2 = dx2 + (cy - xyz[1]) ** 2
                if dxy2 > r2:
                    continue
                base = (i * ny + j) * nz
                for k in range(max(0, ck - span), min(nz, ck + span + 1)):
                    cz = grid.origin[2] + (k + 0.5) * eff
                    d2 = dxy2 + (cz - xyz[2]) ** 2
                    if d2 <= r2:
                        blocked[base + k] = 1
                        if d2 <= v2:
                            solid[base + k] = 1

    # 2) flood the exterior probe-accessible region (not BLOCKED) inward from the boundary
    ext = bytearray(total)
    dq: deque = deque()

    def _seed(i: int, j: int, k: int) -> None:
        ix = (i * ny + j) * nz + k
        if not blocked[ix] and not ext[ix]:
            ext[ix] = 1
            dq.append((i, j, k))

    for i in range(nx):
        for j in range(ny):
            _seed(i, j, 0); _seed(i, j, nz - 1)
    for i in range(nx):
        for k in range(nz):
            _seed(i, 0, k); _seed(i, ny - 1, k)
    for j in range(ny):
        for k in range(nz):
            _seed(0, j, k); _seed(nx - 1, j, k)
    while dq:
        i, j, k = dq.popleft()
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                ix = (ni * ny + nj) * nz + nk
                if not blocked[ix] and not ext[ix]:
                    ext[ix] = 1
                    dq.append((ni, nj, nk))

    # 3) A discrete internal void admits an ENCLOSED probe centre: a cell that is NOT blocked (a probe
    #    fits) but is NOT exterior-connected. This excludes the diffuse INTERSTITIAL packing gaps
    #    (those are "blocked" — too tight for any probe centre — so they never seed a cavity, the very
    #    distinction that separates a real cavity from normal sub-vdW packing slack). Then GROW each
    #    enclosed-probe region by the probe radius into the vdW-empty space (not SOLID) to recover the
    #    full solvent-excluded cavity VOLUME + its lining shell. Growing the (few) enclosed cells is
    #    cheap; the result is realistic discrete cavities, not the whole interstitial network.
    pr_cells = int(math.ceil(probe / eff))
    pr2 = (probe / eff) ** 2
    offsets = [(di, dj, dk) for di in range(-pr_cells, pr_cells + 1)
               for dj in range(-pr_cells, pr_cells + 1)
               for dk in range(-pr_cells, pr_cells + 1)
               if di * di + dj * dj + dk * dk <= pr2]

    cavity = bytearray(total)
    for ci in range(nx):
        for cj in range(ny):
            base = (ci * ny + cj) * nz
            for ck in range(nz):
                ix = base + ck
                if blocked[ix] or ext[ix]:                  # an ENCLOSED probe centre (sealed void seed)
                    continue
                cavity[ix] = 1
                for di, dj, dk in offsets:                  # grow by the probe into vdW-empty space
                    ni, nj, nk = ci + di, cj + dj, ck + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        gx = (ni * ny + nj) * nz + nk
                        if not solid[gx]:
                            cavity[gx] = 1

    # 4) connected components of CAVITY cells (6-connectivity)
    cell_vol = eff ** 3
    seen = bytearray(total)
    cavities: List[dict] = []
    cid = 0
    for start in range(total):
        if not cavity[start] or seen[start]:
            continue
        # BFS one component
        comp: List[Tuple[int, int, int]] = []
        si = start // (ny * nz)
        sj = (start // nz) % ny
        sk = start % nz
        seen[start] = 1
        stack = [(si, sj, sk)]
        while stack:
            i, j, k = stack.pop()
            comp.append((i, j, k))
            for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                    ix = (ni * ny + nj) * nz + nk
                    if cavity[ix] and not seen[ix]:
                        seen[ix] = 1
                        stack.append((ni, nj, nk))
        volume = len(comp) * cell_vol
        if volume < min_volume:
            continue
        cid += 1
        cells = set(comp)
        cx = sum(c[0] for c in comp) / len(comp)
        cy = sum(c[1] for c in comp) / len(comp)
        cz = sum(c[2] for c in comp) / len(comp)
        centroid = (grid.origin[0] + (cx + 0.5) * eff,
                    grid.origin[1] + (cy + 0.5) * eff,
                    grid.origin[2] + (cz + 0.5) * eff)
        lining = _lining_residues(cells, grid, residues, lining_radius) if residues else []
        chains_in = sorted({ch for ch, _rn, _nm in lining})
        cavities.append({
            "cavity_id": cid, "volume": round(volume, 1), "centroid": centroid,
            "cells": cells, "grid": grid, "lining": lining,
            "is_interface": len(chains_in) > 1,
        })
    cavities.sort(key=lambda c: -c["volume"])
    for n, cav in enumerate(cavities, 1):
        cav["cavity_id"] = n                                # renumber by descending volume
    return cavities


def _lining_residues(cells: set, grid: _Grid, residues: Dict[str, Dict[int, Dict[str, object]]],
                     lining_radius: float) -> List[Tuple[str, int, str]]:
    """Residues with a Cβ (fallback Cα) within *lining_radius* of any void cell center — the residues
    whose side chain points into the cavity (the fill candidates come from here). Sorted, deduped."""
    cell_centers = [grid.center(i, j, k) for (i, j, k) in cells]
    out: List[Tuple[str, int, str]] = []
    lr2 = lining_radius * lining_radius
    for ch, residue_map in residues.items():
        for rn, r in residue_map.items():
            ref = r.get("CB") or r.get("CA")
            if not ref:
                continue
            for cc in cell_centers:
                if (ref[0] - cc[0]) ** 2 + (ref[1] - cc[1]) ** 2 + (ref[2] - cc[2]) ** 2 <= lr2:
                    out.append((ch, rn, str(r.get("resname") or "")))
                    break
    return sorted(out, key=lambda t: (t[0], t[1]))


# ── the fill half: place a larger side chain's bulk, rotamer-aware, crediting reach-into-void ──
def _chi1_grid() -> List[float]:
    n = int(round(360.0 / CHI1_STEP))
    return [-180.0 + CHI1_STEP * i for i in range(n)]


def _place_probes(to_aa: str, n: Vec3, ca: Vec3, cb: Vec3, chi1: float, chi2: float) -> List[Vec3]:
    """Place the larger side chain's bulk (γ/δ carbons) for ``to_aa`` at this backbone over (χ1, χ2).
    A→… (from Gly/Ala) with target A returns the Cβ itself (the only added atom). The geometric FILL
    PROBE — idealized carbons whose reach-into-void + clash are what the score weighs. PURE."""
    if to_aa == "A":
        return [cb]
    built: Dict[str, Vec3] = {"N": n, "CA": ca, "CB": cb}
    probes: List[Vec3] = []
    for idx, (ra, rb, rc, bond, angle, base, offset) in enumerate(_PROBE_SPECS.get(to_aa, [])):
        a, b, c = built[ra], built[rb], built[rc]
        dih = (chi1 if base == "chi1" else chi2) + offset
        p = place_from_internal(a, b, c, bond, angle, dih)
        built[f"p{idx}"] = p
        probes.append(p)
    return probes


def _void_member(xyz: Vec3, cells: set, grid: _Grid) -> bool:
    """Is *xyz* inside the cavity (its grid cell, or a 1-cell neighbour, is a void cell)? The ±1 slack
    forgives a probe atom landing just at the void boundary."""
    i0, j0, k0 = grid.cell_of(xyz)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if (i0 + di, j0 + dj, k0 + dk) in cells:
                    return True
    return False


def _resolve_cb(r: Dict[str, object]) -> Optional[Vec3]:
    """The residue's Cβ — real, else reconstructed for Gly from N/CA/C (`build_cb`); None if the
    backbone is too incomplete to place a side chain."""
    cb = r.get("CB")
    if cb is not None:
        return cb  # type: ignore[return-value]
    n, ca, c = r.get("N"), r.get("CA"), r.get("C")
    if n and ca and c:
        return build_cb(n, ca, c)  # type: ignore[arg-type]
    return None


def fill_candidate(cavity: dict, chain: str, resnum: int, r: Dict[str, object], to_aa: str,
                   clash_grid: ClashGrid) -> Optional[dict]:
    """Score ONE small→larger fill at (chain, resnum) for one cavity, ROTAMER-AWARE. Place ``to_aa``'s
    bulk over the χ1(×χ2) sweep; a rotamer REACHES if any probe atom is inside the void; track the best
    reach (fraction of probe atoms in-void) among CLASH-FREE rotamers, else among reaching-but-clashing
    ones. Returns None when no rotamer reaches the void at all (this residue's larger side chain does
    NOT point into THIS cavity — not a fill). Else ``{chain, position, from_aa, to_aa, cavity_id,
    void_volume, fill_fraction, reach_score, clash, score}``:
        fill_fraction = min(1, volume_gain / void_volume)        # how much of the void it fills
        reach_score   = best clash-free in-void fraction         # how well a viable rotamer reaches
        score         = fill_fraction × reach_score (× OVERFILL_PENALTY if every reaching rotamer clashes)
    PURE-ish (reads the clash grid)."""
    from_aa = _THREE_TO_ONE.get(str(r.get("resname") or ""), "X")
    if to_aa not in _VOLUME_MUTATIONS.get(from_aa, []):
        return None
    n, ca = r.get("N"), r.get("CA")
    cb = _resolve_cb(r)
    if not (n and ca and cb):
        return None
    volume_gain = _SC_VOLUME.get(to_aa, 0.0) - _SC_VOLUME.get(from_aa, 0.0)
    if volume_gain <= 0:
        return None
    cells, grid = cavity["cells"], cavity["grid"]
    exclude = {(chain, resnum)}
    chi2_opts = CHI2_SET if to_aa in ("L", "I") else (0.0,)
    best_clean = -1.0
    best_dirty = -1.0
    for chi1 in _chi1_grid():
        for chi2 in chi2_opts:
            probes = _place_probes(to_aa, n, ca, cb, chi1, chi2)  # type: ignore[arg-type]
            if not probes:
                continue
            in_void = sum(1 for p in probes if _void_member(p, cells, grid))
            if in_void == 0:
                continue                                          # this rotamer doesn't reach the void
            frac = in_void / len(probes)
            clash = clash_grid.any_clash([(p, "C") for p in probes], exclude)
            if clash:
                best_dirty = max(best_dirty, frac)
            else:
                best_clean = max(best_clean, frac)
    if best_clean < 0 and best_dirty < 0:
        return None                                               # no reaching rotamer → not a fill
    void_vol = float(cavity["volume"])
    fill_fraction = min(1.0, volume_gain / void_vol) if void_vol > 0 else 0.0
    if best_clean >= 0:
        reach_score, clash = best_clean, False
        score = fill_fraction * reach_score
    else:
        reach_score, clash = best_dirty, True                     # reaches, but every rotamer collides
        score = fill_fraction * reach_score * OVERFILL_PENALTY
    return {
        "chain": chain, "position": resnum, "from_aa": from_aa, "to_aa": to_aa,
        "cavity_id": cavity["cavity_id"], "void_volume": void_vol,
        "fill_fraction": round(fill_fraction, 3), "reach_score": round(reach_score, 3),
        "volume_gain": round(volume_gain, 1), "clash": clash, "score": round(score, 4),
    }


def scan_cavity_sites(heavy_atoms, residues: Dict[str, Dict[int, Dict[str, object]]], *,
                      probe: float = PROBE_RADIUS, spacing: float = GRID_SPACING,
                      min_volume: float = MIN_VOLUME, lining_radius: float = LINING_RADIUS):
    """The whole cavity-filling scan: DETECT internal voids (≥ *min_volume*, the sub-threshold-noise
    floor — the only volume gate, NO large-void-only curation) then, per cavity, score every lining
    residue's allowed small→larger enlargement (`fill_candidate`, rotamer-aware). SURFACES all
    geometrically-viable fills (a reaching rotamer exists) with honest metrics — clash-flagged ones are
    kept + soft-demoted, never hidden; the designer judges strategic relevance (which a geometric scan
    can't see). Multimers native (interface cavities fall out — `is_interface`). Returns
    ``(candidates, best_partner, cavities)``:
        candidates  = fill dicts sorted by score desc (the source-of-truth table)
        best_partner = {chain: {resnum: best score}} for the heatmap (lining residues by best fill)
        cavities    = per-void summary [{cavity_id, volume, centroid, n_lining, lining_labels,
                       is_interface}] (display/caveat; the heavy `cells`/`grid` are dropped).
    PURE-ish (reads the CIF-derived atoms)."""
    cavities = detect_cavities(heavy_atoms, probe=probe, spacing=spacing, min_volume=min_volume,
                               lining_radius=lining_radius, residues=residues)
    clash_grid = ClashGrid(heavy_atoms)
    candidates: List[dict] = []
    best: Dict[str, Dict[int, float]] = {}
    for cav in cavities:
        for (ch, rn, _nm) in cav["lining"]:
            r = (residues.get(ch) or {}).get(rn)
            if not r:
                continue
            from_aa = _THREE_TO_ONE.get(str(r.get("resname") or ""), "X")
            for to_aa in _VOLUME_MUTATIONS.get(from_aa, []):
                cand = fill_candidate(cav, ch, rn, r, to_aa, clash_grid)
                if cand is None:
                    continue
                candidates.append(cand)
                prev = best.setdefault(ch, {}).get(rn, 0.0)
                best[ch][rn] = max(prev, cand["score"])
    candidates.sort(key=lambda d: -d["score"])
    summary = [{
        "cavity_id": c["cavity_id"], "volume": c["volume"], "centroid": c["centroid"],
        "n_lining": len(c["lining"]),
        "lining_labels": [f"{ch}{rn}" for ch, rn, _nm in c["lining"]],
        "is_interface": c["is_interface"],
    } for c in cavities]
    return candidates, best, summary
