"""
double_mutant_bridge.py
-----------------------
Two-mode double mutant ΔΔG scoring bridge for StructureBot.

Builds candidate pairs from existing single-point scan results, scores them
via DynaMut2's multiple-mutation endpoint (prediction_mm) or PyRosetta, and
ranks them by a composite score that includes real epistasis measurement.

Modes
-----
stability : optimise global thermodynamic stability; excludes interface /
            active-site positions.
epitope   : engineer epitope-proximal pairs; targets interface-proximal
            positions while protecting catalytic machinery.

Backends
--------
dynamut2            : Cα-Cα > 10 Å — residues too far to interact directly;
                      DynaMut2 prediction is reliable.
dynamut2_warned     : 4–10 Å — possible direct interaction; DynaMut2 used
                      with an accuracy warning.
pyrosetta_required  : < 4 Å — strongly interacting; DynaMut2 unreliable;
                      PyRosetta used if run_pyrosetta=True, else pair skipped.

Epistasis convention
--------------------
epistasis = ddg_double − ddg_additive
  < 0  : synergistic  (double mutant more stable than additivity predicts)
  > 0  : antagonistic (mutations fight each other)
  ≈ 0  : purely additive

Sign convention for ΔΔG: positive = destabilising, negative = stabilising.

Critical rules (match all StructureBot bridges)
------------------------------------------------
- All subprocess calls: stdin=subprocess.DEVNULL + CREATE_NO_WINDOW
- All path args: Path(...).as_posix()
- Worker scripts: no project imports, JSON file handoff only
- PyRosetta worker: double-brace rule throughout (lives inside f-string)
- Do NOT import from other bridges directly — accept results as dicts
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import config as _cfg
from tool_router import ToolStepResult

# ── DynaMut2 multiple-mutations API ──────────────────────────────────────────

_MM_BASE        = "https://biosig.lab.uq.edu.au/dynamut2/api"
_MM_SUBMIT_URL  = f"{_MM_BASE}/prediction_mm"
_MM_RESULT_URL  = f"{_MM_BASE}/prediction_mm"

_TIMEOUT          = 15    # seconds per HTTP request
_POLL_INTERVAL    = 5     # seconds between polls
_MAX_POLLS        = 24    # 24 × 5 s = 120 s max per pair
_RETRY_DELAYS     = (3, 8)
_PER_PAIR_TIMEOUT = 120   # wall-clock budget per pair (double mutant takes longer)
_CIRCUIT_BREAKER  = 2     # consecutive failures → skip remaining pairs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _proximal(residues: Set[int], radius: int) -> Set[int]:
    """Return all positions within `radius` of any position in `residues`."""
    result: Set[int] = set()
    for r in residues:
        for off in range(-radius, radius + 1):
            result.add(r + off)
    return result


def _get_camsol(mut: Dict[str, Any]) -> float:
    """Extract camsol_delta accepting either field name convention."""
    return float(mut.get("camsol_delta") or mut.get("solubility_delta") or 0.0)


def _pair_key(m_a: Dict[str, Any], m_b: Dict[str, Any]) -> str:
    def _mk(m: Dict[str, Any]) -> str:
        return f"{m['from_aa']}{m['position']}{m['to_aa']}"
    return f"{_mk(m_a)}+{_mk(m_b)}"


# ══════════════════════════════════════════════════════════════════════════════
# Bridge class
# ══════════════════════════════════════════════════════════════════════════════

class DoubleMutantBridge:
    """
    Two-mode double mutant ΔΔG scoring bridge.

    Usage::

        bridge = DoubleMutantBridge()
        result = bridge.analyze(
            inputs={
                "pdb_path":   "cache/1HSG.pdb",
                "mutations":  scan_results,      # from MutationScanner
                "mode":       "stability",
                "top_n":      10,
            },
            session=session_state,
        )
        if result.success:
            for pair in result.data["top_pairs"]:
                print(pair["pair_key"], pair["ddg_double"], pair["epistasis"])
    """

    def __init__(self) -> None:
        pass

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        inputs:  Dict[str, Any],
        session: Any = None,
    ) -> ToolStepResult:
        """
        Score double mutant pairs and rank by composite score.

        Parameters
        ----------
        inputs keys:
          pdb_path           : str
          mutations          : list[dict] — single-point candidates with
                               {chain, position, from_aa, to_aa, camsol_delta,
                                esm_tolerance, ddg}
          mode               : "stability" | "epitope"
          interface_residues : set[int] | None
          functional_residues: set[int] | None
          esm_scores         : dict[int, float] | None
          top_n              : int  (default config.DOUBLE_MUTANT_TOP_N)
          run_pyrosetta      : bool (default False)
        """
        import time as _time
        t_start = _time.perf_counter()

        pdb_path     = inputs.get("pdb_path", "")
        mutations    = inputs.get("mutations", [])
        mode         = inputs.get("mode", "stability")
        iface_in     = inputs.get("interface_residues") or set()
        func_in      = inputs.get("functional_residues") or set()
        top_n        = int(inputs.get("top_n", _cfg.DOUBLE_MUTANT_TOP_N))
        run_pyrosetta = bool(inputs.get("run_pyrosetta", False))

        interface_residues  = set(iface_in)
        functional_residues = set(func_in)

        progress_cb: Callable[[str], None] = inputs.get(  # type: ignore[assignment]
            "progress_callback", _safe_print
        )

        if not mutations:
            return ToolStepResult(
                tool="double_mutant", success=False,
                error="No mutations supplied — run a mutation scan first.",
            )
        if not Path(pdb_path).is_file():
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=f"PDB file not found: {pdb_path}",
            )
        if mode not in ("stability", "epitope"):
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=f"Unknown mode {mode!r}; use 'stability' or 'epitope'.",
            )

        warnings: List[str] = []

        # ── Step 1: generate candidate pairs ──────────────────────────────────
        progress_cb(f"[DoubleMutant] Generating pairs from {len(mutations)} mutations ({mode} mode)...")
        raw_pairs = self.generate_pairs(
            mutations, mode, interface_residues, functional_residues
        )
        progress_cb(f"  {len(raw_pairs)} pairs after mode filtering.")

        if not raw_pairs:
            return ToolStepResult(
                tool="double_mutant", success=True,
                data={
                    "pairs": [], "top_pairs": [], "mode": mode,
                    "backend_summary": {}, "warnings": warnings,
                    "method_note": "No pairs passed the candidate filters.",
                },
                summary="No double-mutant pairs generated.",
            )

        # ── Step 2: distance routing ───────────────────────────────────────────
        progress_cb("  Computing Cα-Cα distances and routing pairs...")
        pair_dicts = self.route_pairs(raw_pairs, pdb_path)

        # Split by backend
        dynamut2_pairs = [p for p in pair_dicts
                          if p["backend"] in ("dynamut2", "dynamut2_warned")]
        pyrosetta_pairs = [p for p in pair_dicts
                           if p["backend"] == "pyrosetta_required"]

        # Close pairs with PyRosetta disabled → skip + warn
        if pyrosetta_pairs and not run_pyrosetta:
            n_skipped = len(pyrosetta_pairs)
            warnings.append(
                f"{n_skipped} pair(s) skipped — Cα-Cα < "
                f"{_cfg.DOUBLE_MUTANT_DISTANCE_THRESHOLD_CLOSE:.1f} Å requires "
                "PyRosetta (set run_pyrosetta=True to enable)."
            )
            pyrosetta_pairs = []

        # Mid-range accuracy note
        n_mid = sum(1 for p in dynamut2_pairs if p["backend"] == "dynamut2_warned")
        if n_mid:
            warnings.append(
                f"{n_mid} pair(s) have Cα-Cα distance "
                f"{_cfg.DOUBLE_MUTANT_DISTANCE_THRESHOLD_CLOSE:.1f}–"
                f"{_cfg.DOUBLE_MUTANT_DISTANCE_THRESHOLD_FAR:.1f} Å — "
                "DynaMut2 accuracy may be reduced; verify top pairs experimentally."
            )

        # ── Step 3: score via DynaMut2 ────────────────────────────────────────
        scored_pairs: List[Dict[str, Any]] = []

        if dynamut2_pairs:
            progress_cb(
                f"  Scoring {len(dynamut2_pairs)} pair(s) via DynaMut2 "
                f"(max_workers={_cfg.DYNAMUT2_MAX_WORKERS})..."
            )
            dm_results = self.score_pairs_dynamut2(
                dynamut2_pairs, pdb_path, progress_cb
            )
            scored_pairs.extend(dm_results)

        # ── Step 4: score via PyRosetta (optional) ─────────────────────────────
        if pyrosetta_pairs and run_pyrosetta:
            progress_cb(
                f"  Scoring {len(pyrosetta_pairs)} close pair(s) via PyRosetta WSL2..."
            )
            try:
                pr_results = self.score_pairs_pyrosetta(
                    pyrosetta_pairs, pdb_path, progress_cb
                )
                scored_pairs.extend(pr_results)
            except Exception as exc:
                warnings.append(f"PyRosetta scoring failed: {exc!s:.120}")

        if not scored_pairs:
            return ToolStepResult(
                tool="double_mutant", success=True,
                data={
                    "pairs": [], "top_pairs": [], "mode": mode,
                    "backend_summary": {"skipped": len(pair_dicts)},
                    "warnings": warnings,
                    "method_note": "All pairs were skipped (DynaMut2 failure or distance filter).",
                },
                summary="Double mutant scoring produced no results.",
            )

        # ── Step 5: composite scoring ──────────────────────────────────────────
        for pair in scored_pairs:
            pair["composite_score"] = self.compute_composite_score(pair, mode)
            cs = pair["composite_score"]
            pair["confidence"] = (
                "high"     if cs > 0.6 else
                "moderate" if cs >= 0.3 else
                "low"
            )
            # Per-pair warnings
            pair_warns: List[str] = []
            if pair.get("backend") == "dynamut2_warned":
                dist = pair.get("ca_distance")
                dist_str = f"{dist:.1f} Å" if dist is not None else "unknown"
                pair_warns.append(
                    f"Mid-range Cα-Cα distance ({dist_str}); "
                    "DynaMut2 accuracy reduced for interacting residues."
                )
            pair["warnings"] = pair_warns

        scored_pairs.sort(key=lambda p: p["composite_score"], reverse=True)
        top_pairs = scored_pairs[:top_n]

        # ── Backend summary ────────────────────────────────────────────────────
        backend_counts: Dict[str, int] = {}
        for p in pair_dicts:
            be = p.get("backend", "unknown")
            backend_counts[be] = backend_counts.get(be, 0) + 1
        scored_set = {p["pair_key"] for p in scored_pairs}
        backend_counts["skipped"] = sum(
            1 for p in pair_dicts if p["pair_key"] not in scored_set
        )

        elapsed_ms = (_time.perf_counter() - t_start) * 1000

        result_data: Dict[str, Any] = {
            "pairs":           scored_pairs,
            "top_pairs":       top_pairs,
            "mode":            mode,
            "backend_summary": backend_counts,
            "warnings":        warnings,
            "method_note": (
                f"DoubleMutantBridge ({mode} mode). "
                f"DynaMut2 prediction_mm endpoint for far/mid pairs; "
                f"PyRosetta WSL2 for close pairs. "
                f"Epistasis = ddG(double) − ddG(additive). "
                f"n_pairs_evaluated={len(pair_dicts)}, n_scored={len(scored_pairs)}."
            ),
        }

        summary = self.generate_summary(result_data, mode)

        return ToolStepResult(
            tool             = "double_mutant",
            success          = True,
            data             = result_data,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    # ── Step 1: pair generation ────────────────────────────────────────────────

    def generate_pairs(
        self,
        mutations:           List[Dict[str, Any]],
        mode:                str,
        interface_residues:  Optional[Set[int]] = None,
        functional_residues: Optional[Set[int]] = None,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generate filtered candidate pairs from single-point mutation list.

        Returns list of (mutation_a, mutation_b) tuples.
        """
        iface = set(interface_residues or [])
        func  = set(functional_residues or [])

        # Proximity zones
        iface_zone_3  = _proximal(iface, 3)   # interface + within 3 (stability exclusion)
        iface_zone_5  = _proximal(iface, 5)   # interface + within 5 (epitope inclusion)
        func_zone     = _proximal(func, 2)    # functional + within 2 (both modes)

        n = len(mutations)
        all_pairs: List[Tuple[Dict, Dict]] = []

        for i in range(n):
            for j in range(i + 1, n):
                m_a = mutations[i]
                m_b = mutations[j]
                # Both modes: exclude same-position pairs
                if m_a["position"] == m_b["position"]:
                    continue
                all_pairs.append((m_a, m_b))

        # Cap at DOUBLE_MUTANT_MAX_PAIRS before mode filtering
        if len(all_pairs) > _cfg.DOUBLE_MUTANT_MAX_PAIRS:
            all_pairs.sort(
                key=lambda p: (
                    abs(p[0].get("ddg", 0.0)) + abs(p[1].get("ddg", 0.0))
                ),
                reverse=True,
            )
            all_pairs = all_pairs[:_cfg.DOUBLE_MUTANT_MAX_PAIRS]

        filtered: List[Tuple[Dict, Dict]] = []

        if mode == "stability":
            for m_a, m_b in all_pairs:
                pos_a = m_a["position"]
                pos_b = m_b["position"]

                # Exclude interface / interface-proximal positions
                if iface and (pos_a in iface_zone_3 or pos_b in iface_zone_3):
                    continue

                # Exclude functional / functional-proximal positions
                if func and (pos_a in func_zone or pos_b in func_zone):
                    continue

                # Require at least one beneficial mutation
                ddg_a = m_a.get("ddg", 0.0)
                ddg_b = m_b.get("ddg", 0.0)
                sol_a = _get_camsol(m_a)
                sol_b = _get_camsol(m_b)
                has_benefit = (
                    ddg_a < 0 or ddg_b < 0
                    or sol_a > 1.0 or sol_b > 1.0
                )
                if not has_benefit:
                    continue

                filtered.append((m_a, m_b))

        elif mode == "epitope":
            for m_a, m_b in all_pairs:
                pos_a = m_a["position"]
                pos_b = m_b["position"]

                # Require at least one interface-proximal position
                if iface:
                    if pos_a not in iface_zone_5 and pos_b not in iface_zone_5:
                        continue

                # Exclude only if BOTH are in functional zone
                if func and (pos_a in func_zone and pos_b in func_zone):
                    continue

                # Exclude if either has very low ESM tolerance (misfolding risk)
                tol_a = m_a.get("esm_tolerance", 1.0)
                tol_b = m_b.get("esm_tolerance", 1.0)
                if tol_a < 0.3 or tol_b < 0.3:
                    continue

                filtered.append((m_a, m_b))

        return filtered

    # ── Step 2: distance routing ───────────────────────────────────────────────

    def compute_ca_distance(
        self,
        pdb_path: str,
        chain:    str,
        pos1:     int,
        pos2:     int,
        chain2:   Optional[str] = None,
    ) -> Optional[float]:
        """
        Euclidean distance between Cα atoms of two residues in Å.

        Parameters
        ----------
        chain2 : if None, both residues are looked up on `chain`;
                 if given, pos2 is looked up on chain2 (cross-chain pairs).

        Returns None if either residue is missing.
        """
        xyz1 = self._get_ca_xyz(pdb_path, chain, pos1)
        xyz2 = self._get_ca_xyz(pdb_path, chain2 or chain, pos2)
        if xyz1 is None or xyz2 is None:
            return None
        dx = xyz1[0] - xyz2[0]
        dy = xyz1[1] - xyz2[1]
        dz = xyz1[2] - xyz2[2]
        return round(math.sqrt(dx * dx + dy * dy + dz * dz), 2)

    def _get_ca_xyz(
        self,
        pdb_path: str,
        chain:    str,
        resno:    int,
    ) -> Optional[Tuple[float, float, float]]:
        """Extract Cα coordinates for a single residue. Returns None on failure."""
        try:
            from Bio.PDB import PDBParser  # type: ignore
            parser    = PDBParser(QUIET=True)
            structure = parser.get_structure("s", Path(pdb_path).as_posix())
            model     = structure[0]
            chain_obj = model[chain]
            for res in chain_obj:
                if res.id[1] == resno and "CA" in res:
                    v = res["CA"].get_vector()
                    return (float(v[0]), float(v[1]), float(v[2]))
            return None
        except Exception:
            return None

    def route_pairs(
        self,
        pairs:    List[Tuple[Dict[str, Any], Dict[str, Any]]],
        pdb_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert (m_a, m_b) tuples to fully annotated pair dicts with backend routing.

        Backend assignment:
          dist > FAR_THRESHOLD  → "dynamut2"
          CLOSE_THRESHOLD–FAR   → "dynamut2_warned"
          dist < CLOSE_THRESHOLD → "pyrosetta_required"
          dist is None           → "dynamut2" (fallback with warning)
        """
        FAR   = _cfg.DOUBLE_MUTANT_DISTANCE_THRESHOLD_FAR
        CLOSE = _cfg.DOUBLE_MUTANT_DISTANCE_THRESHOLD_CLOSE
        result: List[Dict[str, Any]] = []

        for m_a, m_b in pairs:
            chain_a = m_a.get("chain", "A")
            chain_b = m_b.get("chain", "A")
            pos_a   = m_a["position"]
            pos_b   = m_b["position"]

            dist = self.compute_ca_distance(
                pdb_path, chain_a, pos_a, pos_b,
                chain2=chain_b if chain_b != chain_a else None,
            )

            if dist is None:
                backend = "dynamut2"
                zone    = "far"
            elif dist > FAR:
                backend = "dynamut2"
                zone    = "far"
            elif dist >= CLOSE:
                backend = "dynamut2_warned"
                zone    = "mid"
            else:
                backend = "pyrosetta_required"
                zone    = "close"

            result.append({
                "mutation_a":     dict(m_a),
                "mutation_b":     dict(m_b),
                "pair_key":       _pair_key(m_a, m_b),
                "ca_distance":    dist,
                "distance_zone":  zone,
                "backend":        backend,
                # scored fields — populated later
                "ddg_double":     None,
                "ddg_additive":   None,
                "epistasis":      None,
                "ddg_A":          None,
                "ddg_B":          None,
                "avg_distance_api": None,
                "backend_used":   None,
                "composite_score": 0.0,
                "confidence":     "low",
                "scoring_components": {},
                "warnings":       [],
            })

        return result

    # ── Step 3: DynaMut2 double-mutant scoring ────────────────────────────────

    def score_pairs_dynamut2(
        self,
        pairs:    List[Dict[str, Any]],
        pdb_path: str,
        progress: Callable[[str], None] = _safe_print,
    ) -> List[Dict[str, Any]]:
        """
        Score double-mutant pairs via DynaMut2 prediction_mm endpoint.

        Runs pairs concurrently (DYNAMUT2_MAX_WORKERS). Each pair writes its
        result back into the pair dict in-place and returns it.
        """
        max_workers = max(1, getattr(_cfg, "DYNAMUT2_MAX_WORKERS", 4))

        _cb_lock        = __import__("threading").Lock()
        _consec_fail    = [0]
        _circuit_broken = __import__("threading").Event()

        def _score_one(pair: Dict[str, Any]) -> Dict[str, Any]:
            if _circuit_broken.is_set():
                return pair

            per_deadline = time.perf_counter() + _PER_PAIR_TIMEOUT
            try:
                res = self._query_dynamut2_mm(pdb_path, pair, per_deadline)
                pair.update(res)
                pair["backend_used"] = pair["backend"]  # dynamut2 or dynamut2_warned
                with _cb_lock:
                    _consec_fail[0] = 0
            except Exception as exc:
                with _cb_lock:
                    _consec_fail[0] += 1
                    if _consec_fail[0] >= _CIRCUIT_BREAKER:
                        _circuit_broken.set()
                pair["warnings"] = pair.get("warnings", []) + [
                    f"DynaMut2 failed for {pair['pair_key']}: {str(exc)[:100]}"
                ]

            return pair

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_score_one, p): p for p in pairs}
            done: List[Dict[str, Any]] = []
            for future in as_completed(future_map):
                result_pair = future.result()
                if result_pair.get("ddg_double") is not None:
                    done.append(result_pair)
                    progress(
                        f"  + {result_pair['pair_key']}: "
                        f"ddG={result_pair['ddg_double']:+.2f} kcal/mol "
                        f"(epistasis={result_pair['epistasis']:+.2f})"
                    )
                else:
                    progress(
                        f"  ! {result_pair['pair_key']}: scoring failed — skipped"
                    )

        if _circuit_broken.is_set():
            progress(
                f"  DynaMut2 circuit breaker tripped after "
                f"{_CIRCUIT_BREAKER} consecutive failures."
            )

        return done

    def _query_dynamut2_mm(
        self,
        pdb_path: str,
        pair:     Dict[str, Any],
        deadline: float,
    ) -> Dict[str, Any]:
        """
        Submit one pair to DynaMut2 prediction_mm and poll for result.

        Returns dict with ddg_double, ddg_additive, epistasis, avg_distance_api.
        Raises RuntimeError / TimeoutError on failure.
        """
        import requests

        m_a = pair["mutation_a"]
        m_b = pair["mutation_b"]

        def _mut_line(m: Dict[str, Any]) -> str:
            return (
                f"{m.get('chain', 'A')} "
                f"{m['from_aa']}{m['position']}{m['to_aa']}"
            )

        line_a = _mut_line(m_a)
        line_b = _mut_line(m_b)

        def _over() -> bool:
            return time.perf_counter() >= deadline

        # Build the mutations list temp file
        def _build_tmp_mutations() -> str:
            fd, path = tempfile.mkstemp(suffix=".txt")
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(f"{line_a}\n{line_b}\n")
            except Exception:
                os.close(fd)
                raise
            return path

        # ── Submit with retry ──────────────────────────────────────────────────
        job_id:   Optional[str]       = None
        last_exc: Optional[Exception] = None
        delays = list(_RETRY_DELAYS) + [None]

        for attempt, delay in enumerate(delays):
            if _over():
                raise TimeoutError(
                    f"Per-pair timeout exceeded during submit for {pair['pair_key']}"
                )
            tmp_path = _build_tmp_mutations()
            try:
                with open(pdb_path, "rb") as pdb_fh, \
                     open(tmp_path, "rb") as mut_fh:
                    resp = requests.post(
                        _MM_SUBMIT_URL,
                        files={
                            "pdb_file":      (Path(pdb_path).name, pdb_fh, "chemical/x-pdb"),
                            "mutations_list": ("mutations.txt", mut_fh, "text/plain"),
                        },
                        timeout=_TIMEOUT,
                    )

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", delay or 30))
                    if delay is not None:
                        time.sleep(retry_after)
                        last_exc = RuntimeError(f"HTTP 429 (attempt {attempt + 1})")
                        continue
                    raise RuntimeError("DynaMut2 rate limit exceeded — no more retries")

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DynaMut2 mm submit HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                body   = resp.json()
                job_id = str(body.get("job_id", ""))
                if not job_id:
                    raise RuntimeError(
                        f"DynaMut2 mm submit missing job_id. Body: {body}"
                    )
                break

            except requests.exceptions.Timeout:
                last_exc = TimeoutError(f"DynaMut2 mm submit timed out after {_TIMEOUT}s")
            except requests.exceptions.ConnectionError as exc:
                last_exc = ConnectionError(f"DynaMut2 unreachable: {exc}")
            except (RuntimeError, ValueError):
                raise
            except Exception as exc:
                last_exc = exc
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if delay is not None:
                time.sleep(delay)

        if job_id is None:
            raise last_exc or RuntimeError(
                f"DynaMut2 mm submit failed for {pair['pair_key']} after all retries"
            )

        # ── Poll for result ────────────────────────────────────────────────────
        for poll_n in range(_MAX_POLLS):
            if _over():
                raise TimeoutError(
                    f"Per-pair timeout during polling for {pair['pair_key']}"
                )

            time.sleep(_POLL_INTERVAL)

            try:
                r = requests.get(
                    _MM_RESULT_URL,
                    params={"job_id": job_id},
                    timeout=_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.Timeout:
                continue
            except Exception:
                continue

            # Running check
            if "message" in data:
                msg = str(data["message"]).upper()
                if msg in ("RUNNING", "PENDING", "QUEUED"):
                    continue
                if "error" in msg.lower():
                    raise RuntimeError(
                        f"DynaMut2 mm job error for {pair['pair_key']}: {data['message']}"
                    )

            # Find the result entry — key is "{chain} {mut1};{chain} {mut2}"
            parsed = self._parse_mm_result(data, line_a, line_b, pair["pair_key"])
            if parsed is not None:
                return parsed

            # Check for status / error fields
            status = str(data.get("status", "")).lower()
            if status in ("error", "failed", "failure"):
                raise RuntimeError(
                    f"DynaMut2 mm job failed for {pair['pair_key']}: {data}"
                )

            # Still waiting — no recognisable result key yet
            if poll_n % 6 == 0:
                elapsed = (poll_n + 1) * _POLL_INTERVAL
                _safe_print(
                    f"  DynaMut2 mm {pair['pair_key']}: waiting "
                    f"({elapsed}s elapsed)..."
                )

        raise TimeoutError(
            f"DynaMut2 mm job {job_id} timed out after "
            f"{_MAX_POLLS * _POLL_INTERVAL}s for {pair['pair_key']}"
        )

    def _parse_mm_result(
        self,
        data:     Dict[str, Any],
        line_a:   str,
        line_b:   str,
        pair_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract ddg_double, ddg_additive, epistasis from a DynaMut2 mm response.

        The result key is "{chain} {mut1};{chain} {mut2}" — tries both orderings.
        Values may be strings or floats; converts robustly.
        Returns None if no usable result found.
        """
        candidate_keys = [
            f"{line_a};{line_b}",
            f"{line_b};{line_a}",
        ]

        entry: Optional[Dict[str, Any]] = None
        for key in candidate_keys:
            if key in data:
                entry = data[key]
                break

        # Fallback: if data has exactly one non-metadata entry, use it
        if entry is None:
            skip = {"job_id", "status", "message", "error"}
            candidates = {k: v for k, v in data.items()
                          if k not in skip and isinstance(v, dict)}
            if len(candidates) == 1:
                entry = next(iter(candidates.values()))

        if entry is None:
            return None

        try:
            ddg_double  = round(float(entry["prediction"]), 3)
            ddg_additive = round(float(entry["sum_ddg"]), 3)
            epistasis   = round(ddg_double - ddg_additive, 3)
            avg_dist    = round(float(entry.get("avg_distance", 0) or 0), 2)
        except (KeyError, TypeError, ValueError):
            return None

        return {
            "ddg_double":       ddg_double,
            "ddg_additive":     ddg_additive,
            "epistasis":        epistasis,
            "avg_distance_api": avg_dist or None,
        }

    # ── Step 4: PyRosetta double-mutant scoring ────────────────────────────────

    def score_pairs_pyrosetta(
        self,
        pairs:    List[Dict[str, Any]],
        pdb_path: str,
        progress: Callable[[str], None] = _safe_print,
    ) -> List[Dict[str, Any]]:
        """
        Score close double-mutant pairs via PyRosetta in WSL2.

        Uses the same worker-script pattern as rosetta_bridge._run_rosetta_local().
        Scores each pair individually (sequential — each needs its own relax run).
        """
        try:
            from wsl_bridge import WSLBridge, PYROSETTA_PYTHON  # noqa: F401
        except ImportError:
            progress("  [DoubleMutant] wsl_bridge not found — skipping PyRosetta pairs.")
            return []

        wsl = WSLBridge()
        if not wsl.is_available():
            progress("  [DoubleMutant] WSL2 not available — skipping PyRosetta pairs.")
            return []

        scored: List[Dict[str, Any]] = []

        for pair in pairs:
            m_a = pair["mutation_a"]
            m_b = pair["mutation_b"]
            progress(f"  [PyRosetta] Scoring {pair['pair_key']}...")
            try:
                result_dict = self._run_pair_pyrosetta(wsl, pdb_path, m_a, m_b)
                pair.update(result_dict)
                pair["backend_used"] = "pyrosetta_wsl2"
                scored.append(pair)
                progress(
                    f"  + {pair['pair_key']}: "
                    f"ddG={pair['ddg_double']:+.2f} kcal/mol "
                    f"(epistasis={pair['epistasis']:+.2f})"
                )
            except Exception as exc:
                pair["warnings"] = pair.get("warnings", []) + [
                    f"PyRosetta failed for {pair['pair_key']}: {str(exc)[:100]}"
                ]
                progress(f"  ! {pair['pair_key']}: PyRosetta error — {exc!s:.80}")

        return scored

    def _run_pair_pyrosetta(
        self,
        wsl:      Any,
        pdb_path: str,
        m_a:      Dict[str, Any],
        m_b:      Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build and run a standalone PyRosetta worker for one pair."""
        wsl_pdb = wsl.copy_to_wsl(pdb_path)
        if not wsl_pdb:
            raise RuntimeError(f"Failed to copy {pdb_path} to WSL2")

        pdb_hash  = hashlib.md5(Path(pdb_path).read_bytes()).hexdigest()[:12]
        pair_id   = f"{m_a['from_aa']}{m_a['position']}{m_a['to_aa']}"
        pair_id  += f"_{m_b['from_aa']}{m_b['position']}{m_b['to_aa']}"
        wsl_out   = f"/tmp/dm_ddg_{pdb_hash}_{pair_id}.json"

        pair_data = json.dumps({
            "chain_a": m_a.get("chain", "A"),
            "pos_a":   m_a["position"],
            "from_a":  m_a["from_aa"],
            "to_a":    m_a["to_aa"],
            "chain_b": m_b.get("chain", "A"),
            "pos_b":   m_b["position"],
            "from_b":  m_b["from_aa"],
            "to_b":    m_b["to_aa"],
        })

        script = f"""
import json, sys, os

try:
    import pyrosetta
    from pyrosetta import init as rosetta_init, pose_from_file
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    from pyrosetta.toolbox import cleanATOM

    rosetta_init(options="-mute all -ex1 -ex2 -use_input_sc -ignore_unrecognized_res true")

    pair   = json.loads({pair_data!r})
    pdb_in = {wsl_pdb!r}
    out_f  = {wsl_out!r}

    # Clean PDB
    cleaned = pdb_in.replace('.pdb', '_clean.pdb')
    cleanATOM(pdb_in, cleaned)
    if os.path.isfile(cleaned) and os.path.getsize(cleaned) > 100:
        pdb_in = cleaned
    print(f"[DM] PDB cleaned => {{pdb_in}}", flush=True)

    pose     = pose_from_file(pdb_in)
    scorefxn = pyrosetta.create_score_function("ref2015")

    _aa1to3 = {{'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE',
                'G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU',
                'M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG',
                'S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR'}}

    def relax3(p):
        sfx = pyrosetta.create_score_function("ref2015")
        FastRelax(sfx, 3).apply(p)
        return sfx, p

    def wt_relax():
        wt = pose.clone()
        FastRelax(scorefxn, 5).apply(wt)
        print("[DM] WT relax done", flush=True)
        return wt

    def apply_mut(p, chain_id, pos, to_aa):
        rn = p.pdb_info().pdb2pose(chain_id, pos)
        if rn == 0:
            raise ValueError(f"Residue {{pos}}{{chain_id}} not found")
        MutateResidue(target=rn, new_res=_aa1to3.get(to_aa, to_aa)).apply(p)

    wt_pose = wt_relax()

    # Double mutant
    dm = wt_pose.clone()
    apply_mut(dm, pair["chain_a"], pair["pos_a"], pair["to_a"])
    apply_mut(dm, pair["chain_b"], pair["pos_b"], pair["to_b"])
    sfx_dm, dm = relax3(dm)
    wt_re_dm = wt_pose.clone()
    FastRelax(sfx_dm, 3).apply(wt_re_dm)
    ddg_double = sfx_dm(dm) - sfx_dm(wt_re_dm)
    print(f"[DM] double ddG={{ddg_double:+.3f}}", flush=True)

    # Single mutant A
    sm_a = wt_pose.clone()
    apply_mut(sm_a, pair["chain_a"], pair["pos_a"], pair["to_a"])
    sfx_a, sm_a = relax3(sm_a)
    wt_re_a = wt_pose.clone()
    FastRelax(sfx_a, 3).apply(wt_re_a)
    ddg_a = sfx_a(sm_a) - sfx_a(wt_re_a)
    print(f"[DM] ddG_A={{ddg_a:+.3f}}", flush=True)

    # Single mutant B
    sm_b = wt_pose.clone()
    apply_mut(sm_b, pair["chain_b"], pair["pos_b"], pair["to_b"])
    sfx_b, sm_b = relax3(sm_b)
    wt_re_b = wt_pose.clone()
    FastRelax(sfx_b, 3).apply(wt_re_b)
    ddg_b = sfx_b(sm_b) - sfx_b(wt_re_b)
    print(f"[DM] ddG_B={{ddg_b:+.3f}}", flush=True)

    ddg_additive = ddg_a + ddg_b
    epistasis    = ddg_double - ddg_additive

    result = {{
        "ddg_double":   round(float(ddg_double),   3),
        "ddg_additive": round(float(ddg_additive), 3),
        "epistasis":    round(float(epistasis),    3),
        "ddg_A":        round(float(ddg_a),        3),
        "ddg_B":        round(float(ddg_b),        3),
    }}

    with open(out_f, "w") as fh:
        json.dump(result, fh)
    print("[DM] done", flush=True)

except Exception as exc:
    import traceback
    traceback.print_exc()
    with open({wsl_out!r}, "w") as fh:
        json.dump({{"error": str(exc)}}, fh)
"""

        r = wsl.run_python_script(script, timeout=1800)
        if r["stdout"]:
            for line in r["stdout"].splitlines():
                if line.strip():
                    _safe_print(f"  {line.strip()}")

        win_out = str(Path(tempfile.gettempdir()) / f"dm_ddg_{pdb_hash}_{pair_id}.json")
        if not wsl.copy_from_wsl(wsl_out, win_out) or not Path(win_out).is_file():
            raise RuntimeError(
                f"No output file for {pair_id}.\n"
                f"stdout: {r['stdout'][-400:]}\nstderr: {r['stderr'][-200:]}"
            )

        with open(win_out, encoding="utf-8") as fh:
            data = json.load(fh)

        if "error" in data:
            raise RuntimeError(f"PyRosetta worker error: {data['error']}")

        return data

    # ── Step 5: composite scoring ──────────────────────────────────────────────

    def compute_composite_score(
        self,
        pair: Dict[str, Any],
        mode: str,
    ) -> float:
        """
        Compute composite score for a scored pair.

        Returns float in [0, 1] approximately; higher = better candidate.
        """
        ddg_double = pair.get("ddg_double")
        epistasis  = pair.get("epistasis", 0.0) or 0.0
        m_a = pair["mutation_a"]
        m_b = pair["mutation_b"]

        # Stability score: reward negative ddg_double, cap at -5 kcal/mol
        if ddg_double is not None:
            stability_score = min(1.0, max(0.0, -ddg_double) / 5.0)
        else:
            stability_score = 0.0

        # ESM score: mean tolerance (higher = safer to mutate)
        tol_a = m_a.get("esm_tolerance", 1.0) or 1.0
        tol_b = m_b.get("esm_tolerance", 1.0) or 1.0
        esm_score = min(1.0, max(0.0, (tol_a + tol_b) / 2.0))

        # CamSol score: mean delta, normalised to [0, 1]
        sol_a = _get_camsol(m_a)
        sol_b = _get_camsol(m_b)
        camsol_score = min(1.0, max(0.0, (sol_a + sol_b) / 2.0 / 3.0))

        # Synergy bonus: reward negative epistasis (synergistic)
        synergy_bonus = min(1.0, max(0.0, -epistasis) / 3.0)

        if mode == "stability":
            raw = (
                0.40 * stability_score
                + 0.25 * esm_score
                + 0.20 * camsol_score
                + 0.15 * synergy_bonus
            )
            components = {
                "stability_score": round(stability_score, 4),
                "esm_score":       round(esm_score, 4),
                "camsol_score":    round(camsol_score, 4),
                "synergy_bonus":   round(synergy_bonus, 4),
            }

        else:  # epitope
            # Interface score: reward if either mutation is interface-proximal
            iface_pos = set()
            # Pull from pair backend — proxy via distance_zone
            # Real interface proximity must be inferred from mutation flags
            is_proximal_a = m_a.get("interface_proximal", False)
            is_proximal_b = m_b.get("interface_proximal", False)
            interface_score = 1.0 if (is_proximal_a or is_proximal_b) else 0.5

            # Epitope conservation: penalise large surface property changes
            abs_sol = abs((sol_a + sol_b) / 2.0)
            epitope_conservation = min(1.0, max(0.0, 1.0 - abs_sol / 3.0))

            raw = (
                0.25 * stability_score
                + 0.35 * esm_score
                + 0.20 * interface_score
                + 0.10 * epitope_conservation
                + 0.10 * synergy_bonus
            )
            components = {
                "stability_score":      round(stability_score, 4),
                "esm_score":            round(esm_score, 4),
                "interface_score":      round(interface_score, 4),
                "epitope_conservation": round(epitope_conservation, 4),
                "synergy_bonus":        round(synergy_bonus, 4),
            }

        pair["scoring_components"] = components
        return round(raw, 3)

    # ── Step 6: summary ────────────────────────────────────────────────────────

    def generate_summary(
        self,
        result: Dict[str, Any],
        mode:   str,
    ) -> str:
        """Generate a plain-text summary for display in a Rich Panel."""
        pairs      = result.get("pairs", [])
        top_pairs  = result.get("top_pairs", [])
        backend_s  = result.get("backend_summary", {})
        warns      = result.get("warnings", [])

        n_total   = (
            backend_s.get("dynamut2", 0)
            + backend_s.get("dynamut2_warned", 0)
            + backend_s.get("pyrosetta_required", 0)
            + backend_s.get("skipped", 0)
        )
        n_scored  = len(pairs)
        n_skipped = backend_s.get("skipped", 0)
        n_dm      = backend_s.get("dynamut2", 0) + backend_s.get("dynamut2_warned", 0)
        n_pr      = backend_s.get("pyrosetta_required", 0)

        lines = [
            f"=== Double Mutant Analysis — {mode.capitalize()} Mode ===",
            f"Pairs evaluated: {n_total} | Scored: {n_scored} | Skipped: {n_skipped}",
            f"Backend: {n_dm} DynaMut2 / {n_pr} PyRosetta / {n_skipped} skipped",
            "",
        ]

        if top_pairs:
            lines.append(
                f"  {'Pair':<18} {'ddG(dbl)':>9} {'Additive':>9} "
                f"{'Epistasis':>10} {'Dist(Å)':>8} {'Conf':<8}"
            )
            lines.append("  " + "-" * 68)
            for p in top_pairs[:5]:
                dist = p.get("ca_distance")
                dist_str = f"{dist:.1f}" if dist is not None else "  N/A"
                ddg_d = p.get("ddg_double")
                ddg_a = p.get("ddg_additive")
                epi   = p.get("epistasis")
                lines.append(
                    f"  {p['pair_key']:<18} "
                    f"{(f'{ddg_d:+.2f}' if ddg_d is not None else '  N/A'):>9} "
                    f"{(f'{ddg_a:+.2f}' if ddg_a is not None else '  N/A'):>9} "
                    f"{(f'{epi:+.2f}' if epi is not None else '   N/A'):>10} "
                    f"{dist_str:>8} "
                    f"{p.get('confidence', '?'):<8}"
                )

        # Synergistic pairs
        synergistic = [p for p in pairs if (p.get("epistasis") or 0) < -0.5]
        if synergistic:
            lines.append("")
            lines.append(f"Synergistic pairs (epistasis < -0.5 kcal/mol):")
            for p in synergistic[:3]:
                lines.append(
                    f"  {p['pair_key']}: epistasis = {p['epistasis']:+.2f} kcal/mol"
                )

        # Antagonistic pairs
        antagonistic = [p for p in pairs if (p.get("epistasis") or 0) > 0.5]
        if antagonistic:
            lines.append("")
            lines.append(f"Antagonistic pairs (epistasis > +0.5 kcal/mol) — avoid:")
            for p in antagonistic[:3]:
                lines.append(
                    f"  {p['pair_key']}: epistasis = {p['epistasis']:+.2f} kcal/mol"
                )

        if warns:
            lines.append("")
            for w in warns:
                lines.append(f"  ⚠ {w}")

        lines.append("")
        lines.append(
            "Note: Epistasis values < -0.5 kcal/mol indicate genuine cooperativity "
            "beyond additive effects — prioritise these for experimental validation."
        )

        return "\n".join(lines)
