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

# Per-session cache for the venv312 import-chain capability probe, keyed by
# (interpreter, dir) so a config change re-probes. Spawned at most once per key.
_IMPORT_PROBE_CACHE: Dict[Tuple[str, str], bool] = {}


def _reset_import_probe_cache() -> None:
    """Clear the capability-probe cache (tests; or after a venv312 change)."""
    _IMPORT_PROBE_CACHE.clear()

# The residue-identity primitives now live in the shared `residue_mapping` module
# so every fast-tier voter (ThermoMPNN, RaSP, …) reuses the IDENTICAL spine +
# WT-anchored alignment.  Re-exported here for backward-compatible imports.
from residue_mapping import (          # noqa: F401  (re-export)
    _THREE_TO_ONE,
    candidate_key,
    ordered_chain_residues,
    align_predictions_to_resnums,
)


class ThermoMPNNBridge:
    """Per-mutation ThermoMPNN ddG, normalised to the system sign convention."""

    def __init__(self) -> None:
        self._dir    = Path(_cfg.THERMOMPNN_DIR) if _cfg.THERMOMPNN_DIR else None
        self._python = _cfg.THERMOMPNN_PYTHON
        self._model  = _cfg.THERMOMPNN_MODEL
        self._sign   = int(getattr(_cfg, "THERMOMPNN_DDG_SIGN", 1))

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True iff enabled AND a valid install AND the venv312 import chain RUNS.

        Two tiers — the second is the cavity-class fix (a True flag must mean
        "can run", not "files exist"):
          (1) cheap presence: script + model + interpreter files exist;
          (2) CAPABILITY: spawn the venv312 interpreter and confirm the import
              chain `custom_inference.py` needs actually resolves WHERE IT RUNS
              (torch/omegaconf/Bio.PDB + the tool's own modules). A venv312 whose
              torch/deps silently broke has the interpreter FILE present (tier 1
              passes) but cannot import — tier 2 catches that, where a same-process
              check (wrong environment) would not. Cached per session; graceful
              (probe failure → correctly False → clean skip, never a crash).
        """
        enable = getattr(_cfg, "THERMOMPNN_ENABLE", "auto")
        if str(enable).lower() in ("false", "0", "off", "no"):
            return False
        if not self._dir:
            return False
        script = self._dir / "analysis" / "custom_inference.py"
        if not (script.is_file() and Path(self._model).is_file()
                and Path(self._python).is_file()):
            return False
        return self._venv312_import_chain_ok()

    def _venv312_import_chain_ok(self) -> bool:
        """Tier-2 capability probe: does `custom_inference.py`'s import chain
        resolve in the venv312 SUBPROCESS (where inference runs)? Imports only —
        NOT a model load / inference. Cached per (interpreter, dir). A definitive
        result (probe ran to completion) is cached; a probe-infrastructure failure
        (spawn error / timeout) returns False WITHOUT caching, so a transient error
        never masquerades permanently as 'capability absent'."""
        key = (str(self._python), str(self._dir))
        if key in _IMPORT_PROBE_CACHE:
            return _IMPORT_PROBE_CACHE[key]
        analysis = self._dir / "analysis"
        # Replicate custom_inference.py's sys.path: its own dir (analysis/) + the
        # repo root (it does sys.path.append(dirname(ABPATH))).
        probe = (
            "import sys; sys.path[:0] = [r'{ana}', r'{root}']\n"
            "import torch, pandas, omegaconf\n"
            "from Bio.PDB import PDBParser\n"
            "import datasets, train_thermompnn, protein_mpnn_utils\n"
            "import thermompnn_benchmarking, SSM\n"
            "print('THERMOMPNN_IMPORT_OK')\n"
        ).format(ana=str(analysis), root=str(self._dir))
        try:
            r = subprocess.run(
                [self._python, "-c", probe],
                capture_output=True, text=True,
                timeout=int(getattr(_cfg, "THERMOMPNN_PROBE_TIMEOUT", 60)),
                stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
            )
        except Exception:
            # spawn failure / timeout — not a definitive capability verdict; do NOT
            # cache, skip this run gracefully (the voter renormalizes as today).
            return False
        ok = (r.returncode == 0) and ("THERMOMPNN_IMPORT_OK" in (r.stdout or ""))
        _IMPORT_PROBE_CACHE[key] = ok
        return ok

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

        # 1. The chain's ordered (resnum, icode, aa) list — the alignment spine
        # (shared with the scanner).  Done FIRST (cheap): no residues → nothing to
        # attribute → skip BEFORE the GPU subprocess.
        ordered = ordered_chain_residues(pdb_path, chain)
        if not ordered:
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

        # 3. Parse CSV → rows (position, wt, mut, ddg) + the per-position wildtype.
        rows: List[Tuple[int, str, str, float]] = []
        pos_wt: Dict[int, str] = {}
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
                    rows.append((pos, wt, mut, ddg))
                    pos_wt.setdefault(pos, wt)
        except Exception as exc:
            _log(f"  ThermoMPNN: failed parsing output ({str(exc)[:120]}) — skipped.")
            return {}, {}

        # 4. WT-ANCHORED ALIGNMENT (shared helper; exact across gaps AND insertion
        # codes; hard-error on divergence → not_computed, never a mis-attribution).
        pos_to_resnum = align_predictions_to_resnums(ordered, pos_wt, _log, tool="ThermoMPNN")
        if pos_to_resnum is None:
            return {}, {}

        ddg_all: Dict[str, float] = {}
        for pos, wt, mut, ddg in rows:
            ddg_all[candidate_key(chain, pos_to_resnum[pos], wt, mut)] = self._sign * ddg

        # 5. Select the requested candidates.
        ddg_out: Dict[str, float] = {}
        src_out: Dict[str, str]   = {}
        for c in candidates:
            k = candidate_key(chain, int(c["position"]), c["from_aa"], c["to_aa"])
            if k in ddg_all:
                ddg_out[k] = round(ddg_all[k], 4)
                src_out[k] = "thermompnn"
        _log(f"  ThermoMPNN: {len(ddg_out)}/{len(candidates)} candidate ddG(s) "
             f"(WT-anchored alignment over {len(ordered)} residues).")
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
