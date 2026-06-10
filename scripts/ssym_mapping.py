"""
scripts/ssym_mapping.py — Ssym forward/reverse anti-symmetry pairing (IDENTITY-CRITICAL).

The Ssym benchmark ships as two PARALLEL sets: Ssym_dir (forward X→A scored on the WT
structure) and Ssym_inv (reverse A→X scored on the MUTANT structure).  The fwd and rev
entries carry DIFFERENT pdbids (WT vs mutant) and SOMETIMES different residue numbering,
so there is no shared (pdbid, resnum) key — the data-gen harness deferred the pairing
for exactly this reason.

This module builds the fwd↔rev mapping and VERIFIES every pair truly corresponds, in
the same identity-first spirit as the residue-mapping work: it pairs by the parallel
row index, then GATES each pair on the two structure-independent identity signals —

  (1) INVERSE SUBSTITUTION   fwd.wt == rev.mut  AND  fwd.mut == rev.wt
  (2) ANTI-SYMMETRIC ddG     fwd.score == -rev.score   (Ssym is built this way)

Either failing → a true mis-alignment → HARD ERROR (raise), never a silent mis-pair.
Residue POSITION is checked too, but a difference is a benign WT-vs-mutant PDB
RENUMBERING (3 such pairs exist, all 4BVM↔5N4*), NOT a mis-pair — it is recorded as a
`position_offset` + `position_renumbered` flag and surfaced, not treated as fatal.

Builds the mapping only; runs NO voters and launches NO anti-symmetry sweep.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_DIR_SET = "Ssym_dir"
_INV_SET = "Ssym_inv"
_SCORE_TOL = 1e-6   # Ssym anti-symmetry is exact by construction

# Ssym provenance (matches the Task-1 manifest tags): S2648 family → DynaMut2 training.
_SSYM_PROVENANCE = {"thermompnn": "clean", "dynamut2": "training",
                    "rasp": "circular_vs_rosetta", "rosetta": "clean"}


class SsymPairError(ValueError):
    """Raised when a fwd/rev pair fails an identity gate — never mis-pair silently."""


def _parse_variant(v: str) -> Tuple[str, int, str]:
    v = v.strip()
    return v[0], int(v[1:-1]), v[-1]


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read(set_name: str) -> List[Dict[str, str]]:
    p = _ROOT / "RaSP_repo" / "data" / "test" / set_name / "ddG_experimental" / "ddg.csv"
    if not p.is_file():
        raise SsymPairError(f"missing Ssym CSV: {p}")
    return list(csv.DictReader(open(p)))


def _struct_path(set_name: str, pdbid: str) -> str:
    sd = _ROOT / "RaSP_repo" / "data" / "test" / set_name / "structure" / "raw"
    cand = sorted(sd.glob(f"{pdbid}*.pdb"))
    return str(cand[0]) if cand else str(sd / f"{pdbid}.pdb")


def build_ssym_pairs(dir_rows: Optional[List[Dict[str, str]]] = None,
                     inv_rows: Optional[List[Dict[str, str]]] = None,
                     score_tol: float = _SCORE_TOL) -> List[Dict[str, Any]]:
    """Pair Ssym_dir↔Ssym_inv by parallel row index, GATED on inverse-substitution +
    anti-symmetric ddG.  Raises SsymPairError on any identity violation (incl. unequal
    row counts) so a mis-aligned distribution fails loud rather than emitting a guessed
    mapping.  Returns a list of verified pair records."""
    d = dir_rows if dir_rows is not None else _read(_DIR_SET)
    i = inv_rows if inv_rows is not None else _read(_INV_SET)
    if len(d) != len(i):
        raise SsymPairError(
            f"Ssym_dir ({len(d)}) and Ssym_inv ({len(i)}) row counts differ — "
            f"parallel-row pairing assumption broken; refusing to mis-pair.")
    pairs: List[Dict[str, Any]] = []
    for k, (rd, ri) in enumerate(zip(d, i)):
        try:
            wd, pd_, md = _parse_variant(rd["variant"])
            wi, pi, mi = _parse_variant(ri["variant"])
        except (ValueError, IndexError) as exc:
            raise SsymPairError(f"row {k}: unparseable variant "
                                f"({rd.get('variant')!r}/{ri.get('variant')!r})") from exc
        # GATE 1 — inverse substitution (identity-critical)
        if not (wd == mi and md == wi):
            raise SsymPairError(
                f"row {k}: NOT inverse substitution — fwd {rd['pdbid']} {rd['variant']} "
                f"vs rev {ri['pdbid']} {ri['variant']} (expected fwd.wt==rev.mut & "
                f"fwd.mut==rev.wt); refusing to mis-pair.")
        # GATE 2 — anti-symmetric ddG (identity-critical)
        sd_, si_ = _f(rd["score"]), _f(ri["score"])
        if sd_ is None or si_ is None or abs(sd_ + si_) > score_tol:
            raise SsymPairError(
                f"row {k}: ddG not anti-symmetric — fwd {sd_} + rev {si_} "
                f"= {None if sd_ is None or si_ is None else sd_ + si_} (tol {score_tol}); "
                f"refusing to mis-pair.")
        # POSITION — benign renumbering allowed (flagged, not fatal)
        renumbered = (pd_ != pi)
        pair_id = f"ssym_{k:04d}"
        fwd = {
            "set": _DIR_SET, "pdbid": rd["pdbid"].strip(),
            "chain": (rd.get("chainid") or "A").strip() or "A",
            "wt": wd, "resnum": pd_, "mut": md, "variant": rd["variant"].strip(),
            "exp_ddg": sd_, "rosetta_ref_ddg": None,
            "pdb_path": _struct_path(_DIR_SET, rd["pdbid"].strip()),
            "pair_id": pair_id, "antisym_dir": "fwd",
            "provenance": dict(_SSYM_PROVENANCE), "role": "antisymmetry_fwd",
        }
        rev = {
            "set": _INV_SET, "pdbid": ri["pdbid"].strip(),
            "chain": (ri.get("chainid") or "A").strip() or "A",
            "wt": wi, "resnum": pi, "mut": mi, "variant": ri["variant"].strip(),
            "exp_ddg": si_, "rosetta_ref_ddg": None,
            "pdb_path": _struct_path(_INV_SET, ri["pdbid"].strip()),
            "pair_id": pair_id, "antisym_dir": "rev",
            "provenance": dict(_SSYM_PROVENANCE), "role": "antisymmetry_rev",
        }
        pairs.append({
            "pair_id": pair_id,
            "position_dir": pd_, "position_inv": pi,
            "position_offset": pi - pd_, "position_renumbered": renumbered,
            "fwd": fwd, "rev": rev,
        })
    return pairs


def ssym_pair_mutations(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten verified pairs into harness-ready mutation dicts (2 per pair, fwd+rev),
    each carrying the shared pair_id + antisym_dir so the data-gen schema links them."""
    muts: List[Dict[str, Any]] = []
    for p in pairs:
        muts.append(p["fwd"])
        muts.append(p["rev"])
    return muts


def summarize(pairs: List[Dict[str, Any]], log=print) -> None:
    renum = [p for p in pairs if p["position_renumbered"]]
    log(f"Ssym fwd/rev pairs VERIFIED: {len(pairs)} "
        f"(inverse-substitution + anti-symmetric ddG gated)")
    log(f"  position-renumbered pairs (fwd/mut PDB numbering differs): {len(renum)}")
    for p in renum:
        log(f"    {p['pair_id']}: fwd {p['fwd']['pdbid']} {p['fwd']['variant']} "
            f"(pos {p['position_dir']}) ↔ rev {p['rev']['pdbid']} {p['rev']['variant']} "
            f"(pos {p['position_inv']}); offset {p['position_offset']:+d}")


if __name__ == "__main__":
    ps = build_ssym_pairs()
    summarize(ps)
    print(f"flattened mutations (fwd+rev): {len(ssym_pair_mutations(ps))} "
          f"(NOT run — anti-symmetry sweep deferred)")
