"""
rasp_bridge.py
--------------
RaSP (KULL-Centre/_2022_ML-ddG-Blaabjerg) as a FAST, LOCAL, per-mutation ddG
voter for the mutation-scan fast tier — the PHYSICS-AXIS PROXY.

RaSP is a trained SURROGATE FOR ROSETTA, so it fills the physics axis as a fast
proxy; it is NOT an independent voter.  The per-candidate handoff (Rosetta
supersedes RaSP, RaSP's score contribution → 0, value retained as proxy-QC) lives
in mutation_scanner.present_voters_score — this bridge only PRODUCES the ddG.

Design (mirrors the existing bridges):
  - EXECUTION: a WSL2 subprocess (RASP_PYTHON in the dedicated rasp_env venv)
    running rasp_worker.py — clean (reduce+pdbfixer+openmm) → extract envs →
    cavity+ds ensemble → inverse-Fermi → CSV.  Reuses WSLBridge for availability,
    path translation, and the subprocess (proper arg-list exec — no shell mangling).
  - MAPPING: the SHARED residue_mapping spine — ordered_chain_residues on the
    ORIGINAL pdb + the WT-anchored alignment.  RaSP NEVER does its own position
    mapping.  pdbfixer renumbers / strips insertion codes during cleaning, so the
    worker's resnums are re-anchored to the ORIGINAL author resnums here; any
    length/AA divergence (e.g. insertion codes pdbfixer dropped) is a HARD ERROR →
    not_computed for the whole chain (safe-but-lossy, never mis-attributed).
  - RESULT contract: {candidate_key: ddg} + {candidate_key: source='rasp'}, keyed
    chain-aware on (chain, AUTHOR resnum, wt, mut), sign-normalised (RASP_DDG_SIGN).
  - ERROR-FIRST + GRACEFUL: disabled / WSL or env absent / worker failure /
    alignment divergence → ({}, {}) so the fast tier renormalises without RaSP.
  - CACHE: keyed by (pdb content hash, chain, RASP_VERSION_TAG) so a re-scan is
    free AND a sign/model/port change busts the cache instead of serving stale ddG.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg
from residue_mapping import (
    candidate_key, ordered_chain_residues, align_predictions_to_resnums,
)

# Module-level availability cache: the WSL probe (test -x python && test -d repo)
# runs ONCE per process, not per scan, keyed by (distro, python, dir).
_AVAIL_CACHE: Dict[Tuple[str, str, str], bool] = {}


class RaSPBridge:
    """Per-mutation RaSP ddG (Rosetta surrogate), normalised to the system sign."""

    def __init__(self) -> None:
        self._dir    = str(getattr(_cfg, "RASP_DIR", "/home/andre/RaSP_repo"))
        self._python = str(getattr(_cfg, "RASP_PYTHON", "/home/andre/rasp_env/bin/python"))
        self._distro = str(getattr(_cfg, "RASP_WSL_DISTRO", "Ubuntu-24.04"))
        self._enable = str(getattr(_cfg, "RASP_ENABLE", "auto")).strip().lower()
        self._sign   = int(getattr(_cfg, "RASP_DDG_SIGN", 1))
        self._tag    = str(getattr(_cfg, "RASP_VERSION_TAG", "rasp-v1"))
        self._timeout = int(getattr(_cfg, "RASP_TIMEOUT", 600))
        self._cache  = Path(getattr(_cfg, "RASP_CACHE_DIR", "cache/rasp"))
        from wsl_bridge import WSLBridge
        self._wsl = WSLBridge(distribution=self._distro)

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._enable in ("false", "0", "no", "off"):
            return False
        ck = (self._distro, self._python, self._dir)
        if ck in _AVAIL_CACHE:
            return _AVAIL_CACHE[ck]
        ok = False
        if self._wsl.is_available():
            # rasp_env python + RaSP repo both present in WSL
            chk = (f"test -x {shlex.quote(self._python)} && "
                   f"test -d {shlex.quote(self._dir)} && echo RASP_OK")
            r = self._wsl.run_command(chk, timeout=30)
            ok = bool(r.get("ok")) and "RASP_OK" in r.get("stdout", "")
        _AVAIL_CACHE[ck] = ok
        return ok

    def status(self) -> str:
        if self._enable in ("false", "0", "no", "off"):
            return "disabled (RASP_ENABLE)"
        if not self._wsl.is_available():
            return "WSL2 unavailable"
        return "available" if self.is_available() else "rasp_env / RaSP_repo not found in WSL"

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_mutations(
        self,
        pdb_path:   str,
        chain:      str,
        candidates: List[Dict[str, Any]],
        progress=None,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Per-candidate RaSP ddG → ({key: ddg}, {key: 'rasp'}).  ({}, {}) when
        unavailable/failed/diverged (the caller leaves ddg=None / not_computed)."""
        _log = progress or (lambda *_: None)
        if not candidates:
            return {}, {}
        if not self.is_available():
            _log(f"  RaSP: {self.status()} — skipped (fast tier without it).")
            return {}, {}

        ordered = ordered_chain_residues(pdb_path, chain)
        if not ordered:
            _log("  RaSP: no mappable chain residues — skipped.")
            return {}, {}

        ddg_all = self._cached(pdb_path, chain)
        if ddg_all is None:
            ddg_all, ran_ok = self._run_and_align(pdb_path, chain, ordered, _log)
            if ran_ok:                       # cache deterministic outcomes only
                self._cache_store(pdb_path, chain, ddg_all)

        ddg_out: Dict[str, float] = {}
        src_out: Dict[str, str]   = {}
        for c in candidates:
            k = candidate_key(chain, int(c["position"]), c["from_aa"], c["to_aa"])
            if k in ddg_all:
                ddg_out[k] = round(ddg_all[k], 4)
                src_out[k] = "rasp"
        _log(f"  RaSP: {len(ddg_out)}/{len(candidates)} candidate ddG(s) "
             f"(physics proxy; WT-anchored over {len(ordered)} residues).")
        return ddg_out, src_out

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_and_align(
        self, pdb_path: str, chain: str,
        ordered: List[Tuple[int, str, str]], _log,
    ) -> Tuple[Dict[str, float], bool]:
        """Run the WSL worker, align its output to AUTHOR resnums.  Returns
        ({key: ddg}, worker_ran_ok).  worker_ran_ok distinguishes a deterministic
        outcome (cacheable — incl. a hard-error empty) from a transient WSL/worker
        failure (NOT cached, so a later run retries)."""
        try:
            csv_path = self._run_worker(pdb_path, chain, _log)
            if not csv_path:
                _log("  RaSP: worker failed — skipped (not cached, will retry).")
                return {}, False

            rows, pos_wt = self._parse_csv(csv_path, chain)
            if not rows:
                _log("  RaSP: worker produced no rows for this chain — not_computed.")
                return {}, True            # deterministic: cache the empty

            pos_to_resnum = align_predictions_to_resnums(ordered, pos_wt, _log, tool="RaSP")
            if pos_to_resnum is None:
                return {}, True            # hard error (e.g. insertion codes) — cacheable
            ddg_all = {
                candidate_key(chain, pos_to_resnum[rn], wt, mt): self._sign * ddg
                for (rn, wt, mt, ddg) in rows
            }
            _log(f"  RaSP: {len(ddg_all)} ddG(s) aligned to author resnums.")
            return ddg_all, True
        except Exception as exc:
            _log(f"  RaSP: unexpected error ({type(exc).__name__}: {str(exc)[:120]}) — skipped.")
            return {}, False

    def _run_worker(self, pdb_path: str, chain: str, _log) -> Optional[str]:
        """Run rasp_worker.py in the WSL rasp_env → output CSV path (or None on
        failure).  The single WSL seam — tests mock THIS to drive the real
        parse+align without a WSL roundtrip."""
        wsl = self._wsl
        pdb_wsl    = wsl.translate_path(os.path.abspath(pdb_path))
        worker_wsl = wsl.translate_path(os.path.join(os.path.dirname(__file__), "rasp_worker.py"))
        out_tmp = tempfile.NamedTemporaryFile(suffix="_rasp.csv", delete=False)
        out_tmp.close()
        out_wsl = wsl.translate_path(out_tmp.name)
        cmd = (f"{shlex.quote(self._python)} {shlex.quote(worker_wsl)} "
               f"--repo {shlex.quote(self._dir)} --pdb {shlex.quote(pdb_wsl)} "
               f"--chain {shlex.quote(chain)} --out_csv {shlex.quote(out_wsl)}")
        _log("  RaSP: clean + extract + ds-ensemble inference (WSL CPU)…")
        r = wsl.run_command(cmd, timeout=self._timeout)
        if not r.get("ok") or not (os.path.isfile(out_tmp.name) and os.path.getsize(out_tmp.name) > 0):
            _log(f"  RaSP: worker error ({r.get('error') or r.get('stderr','')[-160:]}).")
            try:
                os.unlink(out_tmp.name)
            except OSError:
                pass
            return None
        return out_tmp.name

    @staticmethod
    def _parse_csv(path: str, chain: str) -> Tuple[List[Tuple[int, str, str, float]], Dict[int, str]]:
        import csv
        rows: List[Tuple[int, str, str, float]] = []
        pos_wt: Dict[int, str] = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("chain") != chain:
                    continue
                try:
                    rn = int(row["resnum"]); wt = row["wt"].strip()
                    mt = row["mt"].strip(); ddg = float(row["ddg"])
                except (KeyError, ValueError, TypeError):
                    continue
                rows.append((rn, wt, mt, ddg))
                pos_wt.setdefault(rn, wt)
        return rows, pos_wt

    # ── Cache (pdb content hash + chain + version tag) ────────────────────────

    def _cache_key(self, pdb_path: str, chain: str) -> Optional[str]:
        try:
            with open(pdb_path, "rb") as fh:
                h = hashlib.sha1(fh.read()).hexdigest()[:16]
        except OSError:
            return None
        tag = hashlib.sha1(self._tag.encode()).hexdigest()[:8]
        return f"rasp_{h}_{chain}_{tag}"

    def _cached(self, pdb_path: str, chain: str) -> Optional[Dict[str, float]]:
        key = self._cache_key(pdb_path, chain)
        if not key:
            return None
        f = self._cache / f"{key}.json"
        if f.is_file():
            try:
                return {k: float(v) for k, v in json.loads(f.read_text()).items()}
            except Exception:
                return None
        return None

    def _cache_store(self, pdb_path: str, chain: str, ddg_all: Dict[str, float]) -> None:
        key = self._cache_key(pdb_path, chain)
        if not key:
            return
        try:
            self._cache.mkdir(parents=True, exist_ok=True)
            (self._cache / f"{key}.json").write_text(json.dumps(ddg_all))
        except Exception:
            pass
