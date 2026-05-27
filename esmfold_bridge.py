"""
esmfold_bridge.py
-----------------
Foldability prediction via ESMFold for validating mutant sequences.

Two prediction paths
--------------------
Primary  : local GPU inference via venv312 subprocess calling esmfold_worker.py.
           Uses transformers (facebook/esmfold_v1) on the RTX 5070 Ti through
           the same venv312 delegation pattern as esm_bridge.py / esm_worker.py.
           Controlled by config.ESMFOLD_USE_LOCAL (default: True).

Fallback : ESM Atlas API  (https://esmatlas.com/api/fold)
           Free, no auth required.  POST the sequence as form data;
           response is a PDB-format string with B-factor = per-residue pLDDT.
           Used when ESMFOLD_USE_LOCAL=False, or when the local worker fails.

Usage
-----
    bridge = ESMFoldBridge()

    # Fold a single sequence
    result = bridge.predict("MKTAYIAKQRQISFVKSHFSRQ...")
    # result["plddt"]       : {1: 87.3, 2: 91.2, ...}  (1-based residue index)
    # result["mean_plddt"]  : float
    # result["pdb_str"]     : PDB-format string
    # result["source"]      : "local_venv312" | "atlas_api"

    # Compare wildtype vs mutant at specific positions
    cmp = bridge.compare_to_wildtype(wt_seq, mut_seq, mutation_positions=[64])
    # cmp["foldability_risk"]  : "low" | "medium" | "high"
    # cmp["plddt_drop"]        : float (positive = mutant is worse)
    # cmp["warning"]           : str or None

    # Quick foldability check for a disulfide Cys pair
    check = bridge.check_disulfide_foldability(pdb_path, chain_a_res=49, chain_b_res=112)

Local-path protocol (venv312 subprocess)
-----------------------------------------
1. Write {"sequence": ..., "label": ..., "model": ...} to a temp JSON file.
2. Run: venv312/Scripts/python.exe esmfold_worker.py --input … --output …
3. Read result JSON; relay worker stdout (progress lines).
4. Return structured result dict on success, None on any failure.

pLDDT interpretation
--------------------
  > 90    : very high confidence — structure almost certainly correct
  70–90   : high confidence
  50–70   : low confidence — treat as rough topology guide
  < 50    : very low confidence — likely disordered

pLDDT scale guard
-----------------
  If mean_plddt < 2.0, the worker returned values in 0-1 scale.
  All values are multiplied by 100 to normalise to the 0-100 range.

Foldability risk thresholds (configurable via config.py)
  pLDDT drop >= ESMFOLD_PLDDT_WARNING_THRESHOLD → "high" risk
  drop in [5, threshold)                         → "medium"
  drop < 5                                       → "low"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg

# Windows-only: prevent child console windows.
_CREATE_NO_WINDOW: int = (
    subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    if sys.platform == "win32"
    else 0
)

# ── Constants ─────────────────────────────────────────────────────────────────

_ATLAS_PRIMARY_URL  = "https://esmatlas.com/api/fold"
_ATLAS_ALT_URL      = "https://api.esmatlas.com/foldSequence/v1/pdb/"
_DEFAULT_TIMEOUT    = 120    # seconds — used as Atlas API timeout
_RATE_LIMIT_DELAY   = 5.0   # seconds between consecutive Atlas API calls
_MAX_SEQUENCE_LEN   = 400   # Atlas rejects very long sequences; warn above this
_PLDDT_SCALE_GUARD  = 2.0   # if mean_plddt < this, assume 0-1 scale; multiply ×100


# ── HuggingFace cache probe ───────────────────────────────────────────────────

def _get_model_cache_dir(model_name: str) -> Path:
    """
    Return the HuggingFace hub cache directory for *model_name*.

    Uses ``HF_HOME`` env var if set, otherwise ``USERPROFILE`` (Windows) or
    ``Path.home()`` as the home root.  Does NOT check whether the directory
    exists.
    """
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        cache_root = Path(hf_home) / "hub"
    else:
        home = Path(os.environ.get("USERPROFILE", "")) or Path.home()
        cache_root = home / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    return cache_root / slug


def _is_model_cached(model_name: str) -> bool:
    """
    Return True if HuggingFace has fully downloaded the model weights.

    Requires at least one non-empty ``.safetensors`` or ``.bin`` weight file
    under the ``snapshots/`` or ``blobs/`` subdirectory of the model cache dir.

    We deliberately avoid searching ``.no_exist/`` — HuggingFace uses that
    directory to cache 0-byte sentinel files marking files *absent* from a
    given revision.  Those sentinels match ``*.safetensors`` globs but are not
    weight files.  Restricting to ``snapshots/`` and ``blobs/`` and requiring
    ``st_size > 0`` prevents both false positives (sentinels) and false
    negatives (incomplete downloads leaving a 0-byte stub).

    Returns False on any error.
    """
    try:
        model_dir = _get_model_cache_dir(model_name)
        if not model_dir.exists():
            return False
        # Only search the subdirs that contain real weight data
        for subdir in ("snapshots", "blobs"):
            search_root = model_dir / subdir
            if not search_root.exists():
                continue
            for ext in ("*.safetensors", "*.bin"):
                if any(f for f in search_root.rglob(ext) if f.stat().st_size > 0):
                    return True
        return False
    except Exception:
        return False


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

    Primary:  venv312 subprocess (GPU inference via transformers).
    Fallback: ESM Atlas API (free, no auth).
    """

    def __init__(self) -> None:
        self._last_atlas_time: float = 0.0   # rate-limit tracker for Atlas API

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        sequence:  str,
        label:     str = "query",
        timeout:   int = _DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Predict structure of *sequence* using ESMFold.

        Tries local venv312 GPU inference first (if ESMFOLD_USE_LOCAL=True),
        then falls back to the ESM Atlas API.

        Returns
        -------
        {
          "success":     bool,
          "label":       str,
          "pdb_str":     str,
          "plddt":       {1: 87.3, 2: 91.2, ...},   # per-residue, 1-based
          "mean_plddt":  float,
          "length":      int,
          "error":       None or str,
          "source":      "local_venv312" | "atlas_api" | "error"
        }
        """
        if not sequence or not sequence.strip():
            return self._error_result(label, "empty sequence")

        sequence = sequence.strip().upper()

        # ── Primary: local venv312 GPU inference ──────────────────────────────
        use_local = getattr(_cfg, "ESMFOLD_USE_LOCAL", True)
        if use_local and self._local_available():
            model_name = getattr(_cfg, "ESMFOLD_MODEL_NAME", "facebook/esmfold_v1")
            is_cached  = _is_model_cached(model_name)
            model_dir  = _get_model_cache_dir(model_name)

            # Manual override: treat as cold (long timeout) even when weights appear cached.
            # Useful when _is_model_cached returns True for a partial/corrupt download.
            force_cold = getattr(_cfg, "ESMFOLD_FORCE_COLD_TIMEOUT", False)
            if force_cold and is_cached:
                is_cached = False   # honour the override

            _pprint(
                f"  ESMFold: cache path: {model_dir} — "
                f"{'cached' if is_cached else 'not cached, using cold timeout'}"
            )

            worker_timeout = (
                getattr(_cfg, "ESMFOLD_WORKER_TIMEOUT_WARM", 120)
                if is_cached
                else getattr(_cfg, "ESMFOLD_WORKER_TIMEOUT_COLD", 600)
            )
            if not is_cached:
                _pprint(
                    "  ESMFold: first run — downloading facebook/esmfold_v1 weights "
                    "(~2.5 GB).\n"
                    "  This will take several minutes. Subsequent runs will be fast."
                )
            result = self._run_local(sequence, label, worker_timeout)
            if result is not None:
                return result
            _pprint(
                "  ESMFold: local inference failed — trying ESM Atlas API "
                "(last resort — permanently unreliable, may return garbage results)..."
            )

        # ── Last resort: Atlas API ────────────────────────────────────────────
        # The Atlas API is permanently unreliable: it may return nonsense pLDDT
        # values (< 2.0) or reject long sequences without a useful error.
        # Always warn so the user knows to treat results with scepticism.
        _pprint(
            "  ESMFold: WARNING — ESM Atlas API is a last resort; results may be "
            "unreliable."
        )
        if len(sequence) > _MAX_SEQUENCE_LEN:
            _pprint(
                f"  ESMFold: sequence length {len(sequence)} > {_MAX_SEQUENCE_LEN}; "
                "prediction may be slow or rejected by the Atlas API."
            )

        # Rate-limit Atlas API calls
        elapsed = time.perf_counter() - self._last_atlas_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)

        pdb_str, error = self._call_atlas(sequence, timeout)
        self._last_atlas_time = time.perf_counter()

        if error or not pdb_str:
            return self._error_result(label, error or "empty response from Atlas API")

        plddt = _parse_plddt_from_pdb(pdb_str)
        mean_plddt = (
            round(sum(plddt.values()) / len(plddt), 2) if plddt else 0.0
        )

        # Hard check: Atlas sometimes returns values in 0-1 scale or pure noise.
        # A mean_plddt below the scale guard after all inference paths is garbage.
        if mean_plddt < _PLDDT_SCALE_GUARD:
            return self._error_result(
                label,
                f"pLDDT out of range after all inference paths — "
                f"Atlas API may be down or returning garbage "
                f"(mean pLDDT = {mean_plddt:.2f})"
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

    # ── Local venv312 path ─────────────────────────────────────────────────────

    def _local_available(self) -> bool:
        """True if venv312 python AND esmfold_worker.py both exist on disk."""
        venv312_python = _cfg.VENV312_PYTHON
        worker_path    = Path(__file__).parent / "esmfold_worker.py"
        return (
            Path(venv312_python).is_file()
            and worker_path.is_file()
        )

    def _run_local(
        self,
        sequence: str,
        label:    str,
        timeout:  int,
    ) -> Optional[Dict[str, Any]]:
        """
        Delegate inference to esmfold_worker.py running inside venv312.

        Protocol
        --------
        1. Write {"sequence": ..., "label": ..., "model": ...} to temp file.
        2. Run: venv312/Scripts/python.exe esmfold_worker.py --input … --output …
        3. Read result JSON; relay worker stdout.
        4. Return result dict on success, None on any failure (caller falls back).

        pLDDT scale guard
        -----------------
        If the returned mean_plddt < 2.0, the worker produced 0-1 scale values.
        All plddt values and mean_plddt are multiplied by 100.
        """
        worker_path    = Path(__file__).parent / "esmfold_worker.py"
        venv312_python = _cfg.VENV312_PYTHON
        model_name     = getattr(_cfg, "ESMFOLD_MODEL_NAME", "facebook/esmfold_v1")

        inp_path: Optional[str] = None
        out_path: Optional[str] = None
        try:
            # Write input JSON to temp file
            inp_fd, inp_path = tempfile.mkstemp(suffix="_esmfold_in.json")
            os.close(inp_fd)
            with open(inp_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"sequence": sequence, "label": label, "model": model_name},
                    fh,
                )

            # Temp file for worker output
            out_fd, out_path = tempfile.mkstemp(suffix="_esmfold_out.json")
            os.close(out_fd)

            _pprint(
                f"  ESMFold: launching venv312 worker "
                f"({len(sequence)} aa, {model_name})..."
            )

            proc = subprocess.run(
                [
                    venv312_python,
                    Path(worker_path).as_posix(),
                    "--input",  Path(inp_path).as_posix(),
                    "--output", Path(out_path).as_posix(),
                ],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=_CREATE_NO_WINDOW,
            )

            # Relay worker stdout so the user sees progress
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    if line.strip():
                        _pprint(line)

            # Relay stderr on failure
            if proc.returncode != 0 and proc.stderr:
                for line in proc.stderr.splitlines()[:10]:
                    if line.strip():
                        _pprint(f"  [ESMFold-worker] {line}")

            if proc.returncode != 0:
                _pprint(
                    f"  ESMFold: venv312 worker exited {proc.returncode}"
                )
                return None

            # Read result JSON
            if not Path(out_path).is_file():
                _pprint("  ESMFold: venv312 worker produced no output file")
                return None

            with open(out_path, "r", encoding="utf-8") as fh:
                result = json.load(fh)

            if not result.get("success"):
                _pprint(
                    f"  ESMFold: venv312 worker error: {result.get('error', '?')}"
                )
                return None

            # ── pLDDT scale guard ─────────────────────────────────────────────
            # Worker should output 0-100, but guard against 0-1 scale output.
            plddt      = result.get("plddt", {})
            mean_plddt = result.get("mean_plddt", 0.0)
            if mean_plddt < _PLDDT_SCALE_GUARD and mean_plddt > 0:
                _pprint(
                    f"  ESMFold: scale guard triggered (mean_plddt={mean_plddt:.4f}); "
                    "multiplying all values by 100."
                )
                plddt = {k: round(v * 100, 2) for k, v in plddt.items()}
                mean_plddt = round(mean_plddt * 100, 2)
                result["plddt"]      = plddt
                result["mean_plddt"] = mean_plddt

            # Convert plddt keys from str → int (worker uses str for JSON compat)
            result["plddt"] = {int(k): v for k, v in plddt.items()}
            result["source"] = "local_venv312"
            result.setdefault("error", None)

            _pprint(
                f"  ESMFold: local done — {len(sequence)} aa, "
                f"mean pLDDT {mean_plddt}, "
                f"device={result.get('device', '?')}"
            )
            return result

        except subprocess.TimeoutExpired:
            _pprint(f"  ESMFold: venv312 worker timed out after {timeout}s")
            return None
        except Exception as exc:
            _pprint(f"  ESMFold: venv312 worker exception: {exc}")
            return None
        finally:
            for p in (inp_path, out_path):
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass

    # ── Atlas API path ─────────────────────────────────────────────────────────

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
        return "<ESMFoldBridge local_venv312+atlas_api>"
