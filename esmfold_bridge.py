"""
esmfold_bridge.py
-----------------
Foldability prediction via ESMFold for validating mutant sequences.

Two prediction paths
--------------------
Primary  : ESM Atlas API  (https://esmatlas.com/api/fold)
           Free, no auth required.  POST the sequence as form data;
           response is a PDB-format string with B-factor = per-residue pLDDT.

Fallback : local ESMFold via the transformers library in venv312.
           Activated automatically when the Atlas API is unreachable and
           transformers + esm are installed in venv312.
           NOT implemented yet — flag for a future session.

Usage
-----
    bridge = ESMFoldBridge()

    # Fold a single sequence
    result = bridge.predict("MKTAYIAKQRQISFVKSHFSRQ...")
    # result["plddt"]       : {1: 87.3, 2: 91.2, ...}  (1-based residue index)
    # result["mean_plddt"]  : float
    # result["pdb_str"]     : PDB-format string

    # Compare wildtype vs mutant at specific positions
    cmp = bridge.compare_to_wildtype(wt_seq, mut_seq, mutation_positions=[64])
    # cmp["foldability_risk"]  : "low" | "medium" | "high"
    # cmp["plddt_drop"]        : float (positive = mutant is worse)
    # cmp["warning"]           : str or None

    # Quick foldability check for a disulfide Cys pair
    check = bridge.check_disulfide_foldability(pdb_path, chain_a_res=49, chain_b_res=112)

API details
-----------
  POST https://esmatlas.com/api/fold
  Content-Type: application/x-www-form-urlencoded
  Body: sequence=<AMINO_ACID_STRING>
  Response: PDB-format text  (B-factor column holds per-residue pLDDT)

  Note: an alternative endpoint that has been observed in the wild is
    POST https://api.esmatlas.com/foldSequence/v1/pdb/
    with the raw sequence as the request body (plain text).
  ESMFoldBridge tries the primary URL first, then the alternative.

  Timeout: 120 s (ESMFold can be slow for long sequences; atlas servers vary)
  Rate-limit: 1 request per 5 seconds (conservative; atlas enforces limits)

pLDDT interpretation
--------------------
  > 90    : very high confidence — structure almost certainly correct
  70–90   : high confidence
  50–70   : low confidence — treat as rough topology guide
  < 50    : very low confidence — likely disordered

Foldability risk thresholds (configurable via config.py)
  pLDDT drop >= ESMFOLD_PLDDT_WARNING_THRESHOLD → "high" risk
  drop in [5, threshold)                         → "medium"
  drop < 5                                       → "low"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg


# ── Constants ─────────────────────────────────────────────────────────────────

_ATLAS_PRIMARY_URL  = "https://esmatlas.com/api/fold"
_ATLAS_ALT_URL      = "https://api.esmatlas.com/foldSequence/v1/pdb/"
_DEFAULT_TIMEOUT    = 120    # seconds — ESMFold inference can be slow
_RATE_LIMIT_DELAY   = 5.0   # seconds between consecutive API calls
_MAX_SEQUENCE_LEN   = 400   # Atlas rejects very long sequences; warn above this


# ── Safe print ────────────────────────────────────────────────────────────────

def _pprint(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ── PDB pLDDT parser ──────────────────────────────────────────────────────────

def _parse_plddt_from_pdb(pdb_str: str) -> Dict[int, float]:
    """
    Extract per-residue pLDDT from the B-factor column of an ESMFold PDB string.

    ESMFold stores pLDDT (0–100) in the B-factor field (columns 61-66).
    Only CA atoms are used (one value per residue).

    Returns {residue_number: plddt_score} (1-based residue numbers from PDB).
    Returns {} on parse failure.
    """
    plddt: Dict[int, float] = {}
    seen_residues: set = set()
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        try:
            resno = int(line[22:26].strip())
            bfac  = float(line[60:66].strip())
        except (ValueError, IndexError):
            continue
        if resno not in seen_residues:
            plddt[resno] = round(bfac, 2)
            seen_residues.add(resno)
    return plddt


# ══════════════════════════════════════════════════════════════════════════════
# Public bridge class
# ══════════════════════════════════════════════════════════════════════════════

class ESMFoldBridge:
    """
    ESMFold foldability prediction bridge.

    Tries ESM Atlas API first; gracefully returns an error result if unavailable.
    No auth required for the Atlas API.
    """

    def __init__(self) -> None:
        self._last_request_time: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        sequence:  str,
        label:     str = "query",
        timeout:   int = _DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Predict structure of *sequence* using ESMFold.

        Returns
        -------
        {
          "success":     bool,
          "label":       str,
          "pdb_str":     str,     # PDB-format output from ESMFold
          "plddt":       {1: 87.3, 2: 91.2, ...},  # per-residue, 1-based
          "mean_plddt":  float,
          "length":      int,
          "error":       None or str,
          "source":      "atlas_api" | "error"
        }
        """
        if not sequence or not sequence.strip():
            return self._error_result(label, "empty sequence")

        sequence = sequence.strip().upper()
        if len(sequence) > _MAX_SEQUENCE_LEN:
            _pprint(
                f"  ESMFold: sequence length {len(sequence)} > {_MAX_SEQUENCE_LEN}; "
                "prediction may be slow or rejected by the Atlas API."
            )

        # Rate-limit: ensure at least _RATE_LIMIT_DELAY between calls
        elapsed = time.perf_counter() - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)

        pdb_str, error = self._call_atlas(sequence, timeout)
        self._last_request_time = time.perf_counter()

        if error or not pdb_str:
            return self._error_result(label, error or "empty response from Atlas API")

        plddt = _parse_plddt_from_pdb(pdb_str)
        mean_plddt = (
            round(sum(plddt.values()) / len(plddt), 2) if plddt else 0.0
        )

        return {
            "success":    True,
            "label":      label,
            "pdb_str":    pdb_str,
            "plddt":      plddt,
            "mean_plddt": mean_plddt,
            "length":     len(sequence),
            "error":      None,
            "source":     "atlas_api",
        }

    def compare_to_wildtype(
        self,
        wt_sequence:        str,
        mut_sequence:       str,
        mutation_positions: List[int],
    ) -> Dict[str, Any]:
        """
        Compare per-residue pLDDT at *mutation_positions* between WT and mutant.

        mutation_positions : 1-based residue numbers in the sequence.

        Returns
        -------
        {
          "success":         bool,
          "mean_plddt_wt":   float,
          "mean_plddt_mut":  float,
          "plddt_drop":      float,   # wt - mut; positive = mutant is worse
          "foldability_risk": "low" | "medium" | "high",
          "position_scores": {64: {"wt": 89.2, "mut": 86.1, "drop": 3.1}, ...},
          "warning":         None or str,
          "error":           None or str,
        }
        """
        _pprint(f"  ESMFold: predicting wildtype ({len(wt_sequence)} aa)...")
        wt_result = self.predict(wt_sequence, label="wildtype")
        if not wt_result["success"]:
            return {
                "success": False,
                "error": f"ESMFold WT prediction failed: {wt_result.get('error')}",
            }

        _pprint(f"  ESMFold: predicting mutant ({len(mut_sequence)} aa)...")
        mut_result = self.predict(mut_sequence, label="mutant")
        if not mut_result["success"]:
            return {
                "success": False,
                "error": f"ESMFold mutant prediction failed: {mut_result.get('error')}",
            }

        wt_plddt  = wt_result["plddt"]
        mut_plddt = mut_result["plddt"]

        # Position-level comparison
        pos_scores: Dict[int, Dict[str, float]] = {}
        local_drops: List[float] = []
        for pos in mutation_positions:
            wt_val  = wt_plddt.get(pos, 0.0)
            mut_val = mut_plddt.get(pos, 0.0)
            drop    = round(wt_val - mut_val, 2)
            pos_scores[pos] = {"wt": wt_val, "mut": mut_val, "drop": drop}
            local_drops.append(drop)

        mean_drop = round(sum(local_drops) / len(local_drops), 2) if local_drops else 0.0

        # Overall mean pLDDT (full sequence)
        mean_wt  = wt_result["mean_plddt"]
        mean_mut = mut_result["mean_plddt"]

        # Foldability risk classification
        threshold = getattr(_cfg, "ESMFOLD_PLDDT_WARNING_THRESHOLD", 10.0)
        if mean_drop >= threshold:
            risk    = "high"
            warning = (
                f"ESMFold: mean pLDDT drop of {mean_drop:.1f} at mutation positions "
                f"exceeds threshold ({threshold:.0f}). High foldability risk."
            )
        elif mean_drop >= 5.0:
            risk    = "medium"
            warning = (
                f"ESMFold: mean pLDDT drop of {mean_drop:.1f} at mutation positions. "
                "Moderate foldability concern — verify with full structure prediction."
            )
        else:
            risk    = "low"
            warning = None

        return {
            "success":          True,
            "mean_plddt_wt":    mean_wt,
            "mean_plddt_mut":   mean_mut,
            "plddt_drop":       mean_drop,
            "foldability_risk": risk,
            "position_scores":  pos_scores,
            "warning":          warning,
            "error":            None,
        }

    def check_disulfide_foldability(
        self,
        pdb_path:      str,
        chain_a_res:   int,
        chain_b_res:   int,
        chain_a:       str = "A",
        chain_b:       str = "B",
    ) -> Dict[str, Any]:
        """
        Check foldability impact of introducing Cys at both disulfide positions.

        Parses sequences from the PDB, introduces X→C at both positions,
        and compares pLDDT at those positions.

        Also checks for free Cys misparing risk in the mutant sequence.

        Returns a dict with "foldability_risk", "warning", and per-position scores.
        If ESMFold is unavailable, returns success=False with a descriptive error.
        """
        try:
            from disulfide_bridge import parse_pdb_atoms, extract_sequence
        except ImportError:
            return {
                "success": False,
                "error": "disulfide_bridge not available",
            }

        if not Path(pdb_path).is_file():
            return {"success": False, "error": f"PDB file not found: {pdb_path}"}

        atoms = parse_pdb_atoms(pdb_path)

        # Extract sequences for both chains
        seq_a, map_a = extract_sequence(atoms, chain_a)
        seq_b, map_b = extract_sequence(atoms, chain_b)

        if not seq_a or not seq_b:
            return {
                "success": False,
                "error": (
                    f"Could not extract sequence for chain {chain_a} and/or {chain_b}. "
                    "Check that the PDB file contains ATOM records for both chains."
                ),
            }

        # Build mutant sequences (X→C at target positions)
        idx_a = map_a.get(chain_a_res)
        idx_b = map_b.get(chain_b_res)

        if idx_a is None or idx_b is None:
            missing = []
            if idx_a is None: missing.append(f"{chain_a}{chain_a_res}")
            if idx_b is None: missing.append(f"{chain_b}{chain_b_res}")
            return {
                "success": False,
                "error": (
                    f"Position(s) not found in sequence map: {', '.join(missing)}. "
                    "Check residue numbering."
                ),
            }

        mut_seq_a = seq_a[:idx_a] + "C" + seq_a[idx_a + 1:]
        mut_seq_b = seq_b[:idx_b] + "C" + seq_b[idx_b + 1:]

        # Concatenate both chains for a combined foldability assessment
        # (positions are 1-based in the concatenated sequence)
        wt_combined  = seq_a + seq_b
        mut_combined = mut_seq_a + mut_seq_b
        pos_in_concat = [idx_a + 1, len(seq_a) + idx_b + 1]

        # Check existing free Cys in mutant (misparing risk)
        existing_cys_a = [
            i + 1 for i, aa in enumerate(seq_a) if aa == "C" and i != idx_a
        ]
        existing_cys_b = [
            i + 1 for i, aa in enumerate(seq_b) if aa == "C" and i != idx_b
        ]
        n_free = len(existing_cys_a) + len(existing_cys_b)
        misparing_warning = None
        if n_free:
            misparing_warning = (
                f"{n_free} existing Cys in the sequence (chain A: {len(existing_cys_a)}, "
                f"chain B: {len(existing_cys_b)}). "
                "Verify they are disulfide-bonded or buried before introducing a new Cys pair."
            )

        # Run the comparison
        result = self.compare_to_wildtype(
            wt_combined, mut_combined, pos_in_concat
        )
        if not result["success"]:
            return result

        # Append misparing note to warning
        if misparing_warning:
            existing_warn = result.get("warning") or ""
            result["warning"] = (
                (existing_warn + "  |  " if existing_warn else "")
                + misparing_warning
            )
        result["misparing_risk"] = n_free > 0
        result["existing_cys_count"] = n_free
        return result

    # ── Internal ───────────────────────────────────────────────────────────────

    def _call_atlas(
        self,
        sequence: str,
        timeout:  int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        POST sequence to ESM Atlas API.
        Tries primary URL, then alternative URL on failure.

        Returns (pdb_str, error_message).
        """
        try:
            import requests
        except ImportError:
            return None, "requests not installed (pip install requests)"

        # Primary URL
        pdb_str, err = self._post_atlas(
            _ATLAS_PRIMARY_URL,
            data={"sequence": sequence},
            content_type="application/x-www-form-urlencoded",
            timeout=timeout,
        )
        if pdb_str:
            return pdb_str, None

        _pprint(f"  ESMFold primary URL failed ({err}); trying alternative...")

        # Alternative URL (plain-text body)
        pdb_str, err2 = self._post_atlas(
            _ATLAS_ALT_URL,
            data=sequence,
            content_type="text/plain",
            timeout=timeout,
        )
        if pdb_str:
            return pdb_str, None

        return None, f"Both Atlas endpoints failed. Primary: {err}. Alt: {err2}"

    @staticmethod
    def _post_atlas(
        url:          str,
        data:         Any,
        content_type: str,
        timeout:      int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """POST to *url*; return (pdb_str, error). pdb_str is None on failure."""
        import requests

        headers = {"Content-Type": content_type}
        try:
            if content_type == "application/x-www-form-urlencoded" and isinstance(data, dict):
                resp = requests.post(url, data=data, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, data=data, headers=headers, timeout=timeout)

            if resp.status_code == 200:
                text = resp.text.strip()
                if text.startswith("ATOM") or text.startswith("HEADER") or "ATOM" in text:
                    return text, None
                return None, f"Unexpected response (not PDB): {text[:80]!r}"

            return None, f"HTTP {resp.status_code}: {resp.text[:120]}"

        except requests.exceptions.Timeout:
            return None, f"Request timed out after {timeout}s"
        except requests.exceptions.ConnectionError as exc:
            return None, f"Connection error: {exc}"
        except Exception as exc:
            return None, f"Unexpected error: {exc}"

    @staticmethod
    def _error_result(label: str, error: str) -> Dict[str, Any]:
        return {
            "success":    False,
            "label":      label,
            "pdb_str":    "",
            "plddt":      {},
            "mean_plddt": 0.0,
            "length":     0,
            "error":      error,
            "source":     "error",
        }

    def __repr__(self) -> str:
        return "<ESMFoldBridge atlas_api>"
