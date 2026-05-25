"""
rosetta_bridge.py
-----------------
Stability and ddG calculation bridge for StructureBot.

Two backends — identical public interface:

BACKEND A — PyRosetta (local)
  Activates when *both* conditions are true:
    1. PYROSETTA_AVAILABLE=true in .env.local
    2. `import pyrosetta` succeeds (i.e. the wheel is installed)

  Python 3.14 wheels are not yet available from pyrosetta.org.
  Windows-native support also requires manual wheel installation or WSL.
  This backend is therefore a documented stub in the current environment.

  To enable when a suitable Python / platform is available:
    Step 1 — Activate the venv: .\\venv\\Scripts\\Activate.ps1
    Step 2 — pip install pyrosetta-installer
    Step 3 — python -c "import pyrosetta_installer;
                         pyrosetta_installer.install_pyrosetta()"
    Step 4 — Set PYROSETTA_AVAILABLE=true in .env.local
    Step 5 — Restart StructureBot

  Reference: Park et al. (2016) Scientific Reports 6:10.1038/srep46918

BACKEND B — Robetta web API (always available)
  https://robetta.bakerlab.org — free academic registration required.
  Credentials: set ROBETTA_API_KEY in .env.local (get from profile page).
  Optionally set ROBETTA_EMAIL for documentation purposes.

  Jobs are submitted asynchronously; StructureBot polls every 30 s.
  The job_id is persisted in SessionState.rosetta_jobs so that jobs
  can be resumed if StructureBot is restarted mid-poll.

Backend selection
-----------------
  ROSETTA_BACKEND=auto      (default) — PyRosetta if available, else Robetta
  ROSETTA_BACKEND=pyrosetta — force PyRosetta (fails if not installed)
  ROSETTA_BACKEND=robetta   — always use Robetta

Output schema
-------------
ToolStepResult.data keys:
  mutations        : list of {chain, position, from_aa, to_aa}
  ddg_scores       : {mutation_key: float}  e.g. {"V82A": 1.47}
                     kcal/mol, positive = destabilising, negative = stabilising
  stability_change : float — mean ddG across all scored mutations
  confidence       : "high" | "medium" | "low"
  backend          : "pyrosetta" | "robetta"
  warnings         : list[str]
  job_id           : str | None  (Robetta job ID, for manual resume)

NOTE: Robetta API endpoints are documented below. They match the Robetta
REST API structure as of 2025 and should be verified against the live
documentation at https://robetta.bakerlab.org/docs/api/ after registering.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tool_router import ToolStepResult

# ── Robetta API constants ──────────────────────────────────────────────────────

_ROBETTA_BASE        = "https://robetta.bakerlab.org"
_ROBETTA_SUBMIT_URL  = f"{_ROBETTA_BASE}/api/queue/rosetta_ddg/"
_ROBETTA_STATUS_URL  = f"{_ROBETTA_BASE}/api/queue/{{job_id}}/"
_ROBETTA_RESULTS_URL = f"{_ROBETTA_BASE}/api/queue/{{job_id}}/results/"

_POLL_INTERVAL_S = 30    # seconds between Robetta status polls
_MAX_POLLS       = 200   # 200 × 30 s = 100-minute timeout per job

# ── Amino-acid helpers ─────────────────────────────────────────────────────────

_STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")


def _mutation_key(mut: Dict[str, Any]) -> str:
    """{"from_aa": "V", "position": 82, "to_aa": "A"} → "V82A"."""
    return f"{mut['from_aa']}{mut['position']}{mut['to_aa']}"


# ── Backend detection ──────────────────────────────────────────────────────────

def _pyrosetta_importable() -> bool:
    """True if the pyrosetta package can actually be imported."""
    try:
        import pyrosetta  # noqa: F401
        return True
    except ImportError:
        return False


def _select_backend() -> str:
    """Return "pyrosetta" or "robetta" based on env and availability."""
    forced = os.environ.get("ROSETTA_BACKEND", "auto").strip().lower()
    if forced == "pyrosetta":
        return "pyrosetta"
    if forced == "robetta":
        return "robetta"
    # "auto": prefer PyRosetta only if explicitly enabled AND importable
    flag = os.environ.get("PYROSETTA_AVAILABLE", "").strip().lower()
    if flag in ("1", "true", "yes") and _pyrosetta_importable():
        return "pyrosetta"
    return "robetta"


# ══════════════════════════════════════════════════════════════════════════════
# Public bridge class
# ══════════════════════════════════════════════════════════════════════════════

class RosettaBridge:
    """
    Unified stability / ddG calculation bridge.

    Usage::

        bridge = RosettaBridge()
        result = bridge.analyze(
            pdb_path  = "cache/1HSG.pdb",
            mutations = [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}],
            session   = session_state,
        )
        if result.success:
            ddg = result.data["ddg_scores"]["V82A"]   # kcal/mol
    """

    def __init__(self) -> None:
        self._backend  = _select_backend()
        self._api_key  = os.environ.get("ROBETTA_API_KEY", "").strip()
        self._email    = os.environ.get("ROBETTA_EMAIL",   "").strip()

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        pdb_path:          str,
        mutations:         List[Dict[str, Any]],
        mode:              str = "ddg",
        session:           Any = None,
        model_id:          str = "1",
        chain:             Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> ToolStepResult:
        """
        Calculate ddG for one or more mutations.

        Parameters
        ----------
        pdb_path          : local path to a PDB/CIF file
        mutations         : list of {chain, position, from_aa, to_aa}
        mode              : "ddg" (default) | "stability" (ignored by Robetta stub)
        session           : SessionState for job persistence
        model_id          : ChimeraX model number (used in viz commands)
        chain             : chain ID for viz colouring  (None = all chains)
        progress_callback : callable(str) for real-time progress messages
        """
        if not mutations:
            return ToolStepResult(
                tool="rosetta", success=False,
                error="No mutations specified.",
            )
        if not Path(pdb_path).is_file():
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"PDB file not found: {pdb_path}",
            )

        try:
            if self._backend == "pyrosetta":
                return self._run_pyrosetta(
                    pdb_path, mutations, mode, model_id, chain, progress_callback
                )
            else:
                return self._run_robetta(
                    pdb_path, mutations, model_id, chain, session, progress_callback
                )
        except Exception as exc:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"Rosetta [{self._backend}] unexpected error: {exc}",
            )

    def backend_status(self) -> str:
        """One-line status for display in StructureBot's `state` command."""
        if self._backend == "pyrosetta":
            return "PyRosetta (local) — ACTIVE"
        key_tag  = "✓ API key set" if self._api_key else "✗ ROBETTA_API_KEY not set"
        mail_tag = self._email or "set ROBETTA_EMAIL"
        return f"Robetta web API — {key_tag} | {mail_tag}"

    # ═══════════════════════════════════════════════════════════════════════════
    # Backend A: PyRosetta (documented stub)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_pyrosetta(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
        mode:      str,
        model_id:  str,
        chain:     Optional[str],
        progress_callback: Optional[Callable[[str], None]],
    ) -> ToolStepResult:
        """
        PyRosetta CartesianDDG protocol — documented stub.

        Full implementation outline (for when Python ≤ 3.13 + valid wheel):
        ─────────────────────────────────────────────────────────────────────
            import pyrosetta
            pyrosetta.init(flags="-mute all -ignore_unrecognized_res true")

            pose      = pyrosetta.io.pose_from_file(pdb_path)
            scorefxn  = pyrosetta.create_score_function("ref2015_cart")

            # 1. FastRelax to establish a clean energy baseline
            relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn, 5)
            relax.apply(pose)
            wt_energy = scorefxn(pose)

            # 2. Per-mutation CartesianDDG
            ddg_scores = {}
            for mut in mutations:
                mut_pose = pose.clone()
                mutate_residue(mut_pose, mut["position"], mut["to_aa"])
                repack_neighbors(mut_pose, scorefxn, mut["position"], radius=8.0)
                ddg = scorefxn(mut_pose) - wt_energy
                ddg_scores[_mutation_key(mut)] = round(ddg, 3)

        Reference: Park et al. (2016) Sci Rep 6; doi:10.1038/srep46918
        ─────────────────────────────────────────────────────────────────────
        To activate this backend:
          1. Install pyrosetta (Python ≤ 3.13 wheel from pyrosetta.org,
             or via conda: conda install -c rosettacommons pyrosetta)
          2. Set PYROSETTA_AVAILABLE=true in .env.local
          3. Restart StructureBot
        """
        return ToolStepResult(
            tool="rosetta", success=False,
            error=(
                "PyRosetta backend is not yet active.\n\n"
                "Python 3.14 wheels are not yet available from pyrosetta.org.\n"
                "To enable this backend:\n"
                "  1. Set up Python ≤ 3.13 (or configure WSL)\n"
                "  2. pip install pyrosetta-installer\n"
                "  3. python -c \"import pyrosetta_installer; "
                "pyrosetta_installer.install_pyrosetta()\"\n"
                "  4. Add PYROSETTA_AVAILABLE=true to .env.local\n"
                "  5. Restart StructureBot\n\n"
                "Robetta web API fallback is available — set ROBETTA_API_KEY "
                "in .env.local to enable it."
            ),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Backend B: Robetta web API
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_robetta(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
        model_id:  str,
        chain:     Optional[str],
        session:   Any,
        progress_callback: Optional[Callable[[str], None]],
    ) -> ToolStepResult:
        """Submit a ddG job to Robetta, poll to completion, return scores."""

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # ── Credential check ──────────────────────────────────────────────────
        if not self._api_key:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "ROBETTA_API_KEY is not set.\n"
                    "  1. Register at https://robetta.bakerlab.org (free academic)\n"
                    "  2. Copy the API key from your profile page\n"
                    "  3. Add to .env.local:  ROBETTA_API_KEY=<your-key>\n"
                    "  4. Restart StructureBot"
                ),
            )

        # ── Submit ────────────────────────────────────────────────────────────
        _progress(f"⚗️  Submitting {len(mutations)} mutation(s) to Robetta…")
        try:
            job_id, warnings = self._submit_ddg_job(pdb_path, mutations)
        except Exception as exc:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"Robetta submission failed: {exc}",
            )

        _progress(
            f"⚗️  Robetta job #{job_id} submitted. "
            f"Polling every {_POLL_INTERVAL_S} s…"
        )

        # Persist job so StructureBot can resume after a restart
        if session is not None:
            try:
                session.add_rosetta_job(job_id, {
                    "mutations":    mutations,
                    "pdb_path":     pdb_path,
                    "backend":      "robetta",
                    "submitted_at": datetime.now().isoformat(timespec="seconds"),
                    "status":       "submitted",
                })
            except AttributeError:
                pass   # older SessionState without rosetta_jobs

        # ── Poll ──────────────────────────────────────────────────────────────
        try:
            final_status = self._poll_job(job_id, _progress, session)
        except TimeoutError:
            timeout_min = _MAX_POLLS * _POLL_INTERVAL_S // 60
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    f"Robetta job #{job_id} timed out after {timeout_min} min.\n"
                    f"  The job is still running on Robetta's servers.\n"
                    f"  Resume with the 'jobs' command, or check "
                    f"https://robetta.bakerlab.org"
                ),
            )
        except Exception as exc:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"Robetta polling error (job #{job_id}): {exc}",
            )

        if final_status not in ("completed",):
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"Robetta job #{job_id} ended with status: {final_status!r}",
            )

        # ── Fetch results ─────────────────────────────────────────────────────
        try:
            ddg_scores = self._fetch_results(job_id)
        except Exception as exc:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"Robetta results fetch failed (job #{job_id}): {exc}",
            )

        # Update job record
        if session is not None:
            try:
                session.update_rosetta_job(job_id, {
                    "status":  "completed",
                    "results": ddg_scores,
                })
            except AttributeError:
                pass

        # ── Build result ──────────────────────────────────────────────────────
        stability_change = (
            sum(ddg_scores.values()) / len(ddg_scores)
            if ddg_scores else 0.0
        )
        coverage   = len(ddg_scores) / max(len(mutations), 1)
        confidence = "high" if coverage >= 0.9 else "medium" if coverage >= 0.5 else "low"

        viz_cmds, viz_exps = self._build_viz_commands(
            mutations, ddg_scores, model_id, chain
        )

        best_key = min(ddg_scores, key=ddg_scores.get) if ddg_scores else "?"
        best_ddg = ddg_scores.get(best_key, 0.0)
        summary = (
            f"Rosetta ddG (Robetta, job #{job_id}): "
            f"{len(ddg_scores)}/{len(mutations)} mutations scored. "
            f"Most stabilising: {best_key} ({best_ddg:+.2f} kcal/mol). "
            f"Mean ΔΔG: {stability_change:+.2f} kcal/mol."
        )
        _progress(f"✓ ⚗️  {summary}")

        return ToolStepResult(
            tool    = "rosetta",
            success = True,
            data    = {
                "mutations":        mutations,
                "ddg_scores":       ddg_scores,
                "stability_change": round(stability_change, 3),
                "confidence":       confidence,
                "backend":          "robetta",
                "warnings":         warnings,
                "job_id":           job_id,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
        )

    # ── Robetta API helpers ────────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self._api_key}",
            "Accept":        "application/json",
        }

    def _submit_ddg_job(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
    ) -> Tuple[str, List[str]]:
        """
        POST a ddG job to Robetta.

        API endpoint (verify at https://robetta.bakerlab.org/docs/api/):
          POST https://robetta.bakerlab.org/api/queue/rosetta_ddg/
          Content-Type: multipart/form-data
          Authorization: Token <key>

          Fields:
            pdb_file  (file)  — PDB bytes
            mutations (str)   — JSON array of mutation dicts

        Returns (job_id, warnings_list).
        """
        import requests

        with open(pdb_path, "rb") as fh:
            files   = {"pdb_file": (Path(pdb_path).name, fh, "chemical/x-pdb")}
            payload = {"mutations": json.dumps(mutations)}
            resp = requests.post(
                _ROBETTA_SUBMIT_URL,
                headers = self._auth_headers(),
                files   = files,
                data    = payload,
                timeout = 30,
            )

        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data     = resp.json()
        job_id   = str(
            data.get("job_id") or data.get("id") or data.get("jobId") or ""
        )
        warnings = data.get("warnings", [])

        if not job_id:
            raise RuntimeError(f"Robetta response missing job_id: {data}")

        return job_id, warnings

    def _poll_job(
        self,
        job_id:   str,
        progress: Callable[[str], None],
        session:  Any,
    ) -> str:
        """
        Poll Robetta status every _POLL_INTERVAL_S seconds.

        API endpoint:
          GET https://robetta.bakerlab.org/api/queue/{job_id}/
          Authorization: Token <key>

          Response JSON keys:
            status         : "queued" | "running" | "completed" | "failed"
            queue_position : int | null

        Returns the final status string.
        Raises TimeoutError after _MAX_POLLS attempts.
        """
        import requests

        url = _ROBETTA_STATUS_URL.format(job_id=job_id)

        for attempt in range(_MAX_POLLS):
            time.sleep(_POLL_INTERVAL_S)
            try:
                resp   = requests.get(url, headers=self._auth_headers(), timeout=15)
                resp.raise_for_status()
                data   = resp.json()
                status = data.get("status", "unknown")
                qpos   = data.get("queue_position")
            except Exception as exc:
                progress(f"⚗️  Robetta #{job_id}: poll error ({exc}) — retrying…")
                continue

            pos_str = f" (queue position {qpos})" if qpos is not None else ""
            progress(f"⚗️  Robetta job #{job_id}: {status}{pos_str}")

            # Update session status
            if session is not None:
                try:
                    session.update_rosetta_job(job_id, {"status": status})
                except AttributeError:
                    pass

            if status in ("completed", "failed", "error"):
                return status

        raise TimeoutError(f"Job #{job_id} exceeded {_MAX_POLLS} poll attempts")

    def _fetch_results(self, job_id: str) -> Dict[str, float]:
        """
        Download completed Robetta ddG results.

        API endpoint:
          GET https://robetta.bakerlab.org/api/queue/{job_id}/results/
          Authorization: Token <key>

          Expected response JSON:
            {"results": [{"mutation": "V82A", "ddg": 1.47, "chain": "A"}, ...]}

          Returns {mutation_key: ddg_kcal_mol}.
        """
        import requests

        url  = _ROBETTA_RESULTS_URL.format(job_id=job_id)
        resp = requests.get(url, headers=self._auth_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        scores: Dict[str, float] = {}
        for entry in data.get("results", []):
            # Accept "mutation": "V82A" or individual {from_aa, position, to_aa} fields
            key = (
                entry.get("mutation")
                or _mutation_key(entry)
                if all(k in entry for k in ("from_aa", "position", "to_aa"))
                else entry.get("mutation", "?")
            )
            val = entry.get("ddg") or entry.get("total_score") or 0.0
            scores[str(key)] = round(float(val), 3)

        return scores

    # ── Visualization ──────────────────────────────────────────────────────────

    def _build_viz_commands(
        self,
        mutations:  List[Dict[str, Any]],
        ddg_scores: Dict[str, float],
        model_id:   str,
        chain:      Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Colour mutated residues by ddG value on a 5-band scale:
          blue   ≤ −1.0  strongly stabilising
          cyan   −1 – 0  mildly stabilising
          white   ≈ 0    neutral
          yellow  0 – +1 mildly destabilising
          red    ≥ +1.0  strongly destabilising
        """
        if not mutations:
            return [], []

        chain_spec = f"/{chain}" if chain else ""
        cmds = [
            f"cartoon #{model_id}",
            f"color #{model_id}{chain_spec} white",
        ]
        exps = [
            "Switch to cartoon representation",
            "Reset all residues to white before applying Rosetta ddG colours",
        ]

        for mut in mutations:
            key  = _mutation_key(mut)
            ddg  = ddg_scores.get(key, 0.0)
            pos  = mut["position"]
            spec = f"#{model_id}{chain_spec}:{pos}"

            if ddg <= -1.0:
                colour, label = "blue",   "strongly stabilising"
            elif ddg < 0.0:
                colour, label = "cyan",   "mildly stabilising"
            elif ddg < 1.0:
                colour, label = "yellow", "mildly destabilising"
            else:
                colour, label = "red",    "strongly destabilising"

            cmds.append(f"color {spec} {colour}")
            exps.append(
                f"Residue {pos} ({key}): ΔΔG = {ddg:+.2f} kcal/mol — {label}"
            )
            cmds.append(f"show {spec} atoms")
            exps.append(f"Show residue {pos} as atoms for visibility")

        cmds.append(f"view #{model_id}")
        exps.append("Fit structure in view")

        return cmds, exps

    def __repr__(self) -> str:
        return (
            f"<RosettaBridge backend={self._backend!r} "
            f"api_key={'set' if self._api_key else 'not set'}>"
        )
