"""
thermompnn_bridge.py
--------------------
ThermoMPNN (Kuhlman-Lab) as a FAST, LOCAL, per-mutation stability (ddG) voter for
the mutation-scan fast tier — a dense stability signal alongside CamSol + ESM,
reusing the existing ProteinMPNN encoder + the venv312 GPU env.

Design (mirrors the existing bridges):
  - EXECUTION shape: esm_bridge — a venv312 subprocess (THERMOMPNN_PYTHON) running
    the tool's own inference script (analysis/custom_inference.py), JSON/CSV I/O,
    CREATE_NO_WINDOW.
  - RESULT contract: rosetta_bridge / DynaMut2 — returns {candidate_key: ddg} +
    {candidate_key: source}; key is CHAIN-AWARE (f"{chain}:{wt}{resnum}{mut}") so a
    multimer scan never merges two chains onto one record.
  - ERROR-FIRST + GRACEFUL: unavailable / failed / unmapped → the mutation simply
    gets source='not_computed' (the caller leaves ddg=None).  NEVER a fake 0.0 and
    NEVER a silent empirical substitute.

Two correctness guards (the cross-tool gotchas, both caught in probe):
  1. SIGN — ThermoMPNN trains on NEGATED Megascale ddG (datasets.py:161), and
     Megascale positive = stabilising, so ThermoMPNN predicts negative = stabilising
     == the system convention (positive = destabilising).  The bridge applies
     THERMOMPNN_DDG_SIGN (default +1, no flip) so it ALWAYS returns ddg in the system
     convention; the live sign-guard confirms it on a known stabiliser.
  2. POSITION → RESNUM — ThermoMPNN's CSV `position` is author_resnum − min_author
     resnum (0-based over the author range), NOT a sequential present-residue index.
     The bridge maps author_resnum = position + min_author_resnum and VERIFIES the
     wildtype AA against the real PDB residue; on mismatch it drops that row (never
     mis-attributes a ddg to the wrong residue) and bails the batch if mismatches
     are widespread.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg

_CREATE_NO_WINDOW: int = (
    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]
)

# 3-letter → 1-letter for the PDB wildtype cross-check (standard AAs only).
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Fraction of CSV rows whose wildtype must verify against the PDB before the
# position→resnum map is trusted.  Below this the map is presumed wrong (e.g. a
# parser-ordering edge) and the whole batch is dropped to not_computed.
_MIN_MAP_VERIFY_RATE = 0.8


def candidate_key(chain: str, resnum: int, wt: str, mut: str) -> str:
    """Chain-aware candidate key — never collides across chains in a multimer."""
    return f"{chain}:{wt}{resnum}{mut}"


class ThermoMPNNBridge:
    """Per-mutation ThermoMPNN ddG, normalised to the system sign convention."""

    def __init__(self) -> None:
        self._dir    = Path(_cfg.THERMOMPNN_DIR) if _cfg.THERMOMPNN_DIR else None
        self._python = _cfg.THERMOMPNN_PYTHON
        self._model  = _cfg.THERMOMPNN_MODEL
        self._sign   = int(getattr(_cfg, "THERMOMPNN_DDG_SIGN", 1))

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True iff enabled AND a valid install (script + model + interpreter)."""
        enable = getattr(_cfg, "THERMOMPNN_ENABLE", "auto")
        if str(enable).lower() in ("false", "0", "off", "no"):
            return False
        if not self._dir:
            return False
        script = self._dir / "analysis" / "custom_inference.py"
        return (script.is_file() and Path(self._model).is_file()
                and Path(self._python).is_file())

    def status(self) -> str:
        if str(getattr(_cfg, "THERMOMPNN_ENABLE", "auto")).lower() in ("false", "0"):
            return "disabled (THERMOMPNN_ENABLE=false)"
        if self.is_available():
            return f"available ({self._dir})"
        return f"not available (THERMOMPNN_DIR={self._dir})"

    # ── Public: per-mutation ddG ──────────────────────────────────────────────

    def score_mutations(
        self,
        pdb_path:   str,
        chain:      str,
        candidates: List[Dict[str, Any]],
        progress:   Optional[Any] = None,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Return (ddg_by_key, source_by_key) for *candidates* on *chain*.

        candidates: [{"position": resnum, "from_aa": wt, "to_aa": mut}, ...]
        ddg is sign-normalised to the system convention (negative = stabilising).
        Unavailable / failed / unmapped → ({}, {}) or partial; callers mark the
        missing ones not_computed.  Never raises, never fakes a value.
        """
        def _log(m: str) -> None:
            if progress:
                try: progress(m)
                except Exception: pass

        if not candidates:
            return {}, {}
        if not self.is_available():
            _log(f"  ThermoMPNN: {self.status()} — skipped (fast tier renormalises without it).")
            return {}, {}

        # 1. PDB chain → {author_resnum: one_letter}, min author resnum.  Done
        # FIRST (cheap): if the chain has no mappable standard residues there is
        # nothing to attribute ddG to, so skip BEFORE spawning the GPU subprocess.
        res_by_num, min_resnum = self._chain_residues(pdb_path, chain)
        if not res_by_num:
            _log("  ThermoMPNN: no mappable chain residues — skipped (no inference run).")
            return {}, {}

        # 2. Run the tool's SSM inference (whole chain, one GPU pass) → CSV.
        try:
            csv_path = self._run_inference(pdb_path, chain, _log)
        except Exception as exc:                      # error-first: never propagate
            _log(f"  ThermoMPNN inference failed ({type(exc).__name__}: {str(exc)[:120]}) — skipped.")
            return {}, {}
        if not csv_path or not Path(csv_path).is_file():
            _log("  ThermoMPNN produced no output — skipped.")
            return {}, {}

        # 3. Parse CSV → (resnum, wt, mut) with wildtype VERIFICATION.
        ddg_all: Dict[str, float] = {}
        seen = verified = 0
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("chain") and row["chain"] != chain:
                        continue
                    try:
                        pos = int(float(row["position"]))
                        wt, mut = row["wildtype"].strip(), row["mutation"].strip()
                        ddg = float(row["ddG_pred"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    seen += 1
                    resnum = min_resnum + pos          # author = position + min author
                    if res_by_num.get(resnum) != wt:   # GUARD: never mis-attribute
                        continue
                    verified += 1
                    ddg_all[candidate_key(chain, resnum, wt, mut)] = self._sign * ddg
        except Exception as exc:
            _log(f"  ThermoMPNN: failed parsing output ({str(exc)[:120]}) — skipped.")
            return {}, {}

        # 4. Map-trust guard: if too few rows verified, the map is suspect → drop all.
        if seen and (verified / seen) < _MIN_MAP_VERIFY_RATE:
            _log(f"  ThermoMPNN: position→resnum map unverified "
                 f"({verified}/{seen} wildtypes matched) — dropping (not_computed) to "
                 "avoid mis-attribution.")
            return {}, {}

        # 5. Select the requested candidates.
        ddg_out: Dict[str, float] = {}
        src_out: Dict[str, str]   = {}
        for c in candidates:
            k = candidate_key(chain, int(c["position"]), c["from_aa"], c["to_aa"])
            if k in ddg_all:
                ddg_out[k] = round(ddg_all[k], 4)
                src_out[k] = "thermompnn"
        _log(f"  ThermoMPNN: {len(ddg_out)}/{len(candidates)} candidate ddG(s) "
             f"(map verified {verified}/{seen}).")
        return ddg_out, src_out

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_inference(self, pdb_path: str, chain: str, log) -> Optional[str]:
        """Subprocess custom_inference.py in venv312; return the output CSV path."""
        script  = str(self._dir / "analysis" / "custom_inference.py")
        out_dir = tempfile.mkdtemp(prefix="thermompnn_")
        cmd = [
            self._python, script,
            "--pdb", str(pdb_path),
            "--chain", chain,
            "--model_path", str(self._model),
            "--out_dir", out_dir,
        ]
        log("  ThermoMPNN: running site-saturation inference (GPU)…")
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            timeout=int(getattr(_cfg, "THERMOMPNN_TIMEOUT", 600)),
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW, env=env,
        )
        pdb_id = Path(pdb_path).stem
        out = Path(out_dir) / f"ThermoMPNN_inference_{pdb_id}.csv"
        return str(out) if out.is_file() else None

    @staticmethod
    def _chain_residues(pdb_path: str, chain: str) -> Tuple[Dict[int, str], int]:
        """{author_resnum: one_letter} for standard residues of *chain*, + min resnum.
        Hetero/water and insertion-coded residues are skipped (the latter are an
        alt_parse_PDB edge the wildtype guard will catch)."""
        res: Dict[int, str] = {}
        try:
            from Bio.PDB import PDBParser
            structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
            model = next(iter(structure))
            if chain not in [c.id for c in model]:
                return {}, 0
            for r in model[chain]:
                het, resseq, icode = r.id
                if het.strip() or (icode and icode.strip()):
                    continue                       # skip hetero + insertion codes
                one = _THREE_TO_ONE.get(r.resname.strip().upper())
                if one:
                    res[int(resseq)] = one
        except Exception:
            return {}, 0
        return res, (min(res) if res else 0)
