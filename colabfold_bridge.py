"""
colabfold_bridge.py
-------------------
AF2-quality sequence→structure prediction via the isolated WSL2 ColabFold env.

This is the v1 STANDALONE bridge (Option A). The fused "validate design"
meta-tool (ColabFold + Rosetta energy + matchmaker), MPNN-result sequence
auto-pull, and batch top-N folding are DEFERRED (see PROJECT_CONTEXT.md §9).

How it works
------------
Mirrors ``rosetta_bridge._run_rosetta_local``: a standalone Python worker is
built as an f-string and run via ``COLABFOLD_PYTHON`` inside WSL2
(Ubuntu-24.04). The worker has ZERO project imports, communicates only through
a JSON file in ``/tmp``, and writes that file even on exception. It writes a
FASTA, runs ``colabfold_batch`` with the REMOTE MSA server (no local DBs),
parses the output directory, and returns structured confidence data. The ranked
PDB + the PAE/pLDDT/coverage PNGs are copied back to a Windows cache dir.

Result schema (``predict`` return value)
-----------------------------------------
    {
      "success":        bool,
      "error":          None | str,
      "oom_risk":       bool,            # blocked pre-launch OR runtime CUDA OOM
      "ranked_pdb":     "<windows path>" | "",
      "mean_plddt":     float,           # 0-100
      "plddt":          {1: 87.3, ...},  # per-residue, 1-based
      "pae":            [[...]] | None,  # predicted aligned error matrix
      "ptm":            float | None,
      "iptm":           float | None,    # oligomers only
      "length":         int,             # single-copy sequence length
      "copies":         int,
      "total_residues": int,             # length * copies
      "num_models":     int,
      "num_recycle":    int,
      "png_paths":      {"pae": "...", "plddt": "...", "coverage": "..."},
      "cached":         bool,
      "eta_s":          float,           # rough pre-run estimate (approximate)
      "elapsed_s":      float,
      "source":         "colabfold_wsl2" | "cache" | "error",
    }

pLDDT interpretation (same scale as ESMFold)
  > 90  very high · 70-90 high · 50-70 low · < 50 very low / disordered
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg
from wsl_bridge import WSLBridge, COLABFOLD_PYTHON


# ── Safe print (cp1252-safe, mirrors esmfold_bridge._pprint) ────────────────────

def _pprint(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ── Module constants ────────────────────────────────────────────────────────────

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Rough per-(model·recycle) seconds-per-residue used only for the ETA estimate.
# Deliberately coarse — the estimate is labelled approximate everywhere.
_ETA_SEC_PER_RES_PER_UNIT = 0.012
_ETA_COMPILE_COLD = 240.0   # one-time XLA compile + weight download, first run
_ETA_COMPILE_WARM = 30.0    # model already compiled this session


class ColabFoldBridge:
    """
    ColabFold structure-prediction bridge (WSL2 ColabFold env, remote MSA).

    Stateless apart from a per-process "have we compiled a model yet" flag used
    only to sharpen the ETA estimate. ``predict()`` never raises — on any
    failure it returns ``success=False`` with a descriptive ``error``.
    """

    def __init__(self) -> None:
        self._wsl = WSLBridge()
        self._compiled_this_session = False   # ETA hint only

    # ── Availability ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True if WSL2 is installed AND the ColabFold env interpreter exists."""
        return self._wsl.is_available() and self._wsl.check_colabfold()

    # ── ETA ──────────────────────────────────────────────────────────────────────

    def estimate_runtime_s(
        self,
        total_residues: int,
        num_models:     int,
        num_recycle:    int,
    ) -> float:
        """
        Rough pre-run wall-clock estimate (seconds). APPROXIMATE — folds vary
        widely with MSA depth and server queueing. Used only for display.
        """
        compile_cost = (
            _ETA_COMPILE_WARM if self._compiled_this_session else _ETA_COMPILE_COLD
        )
        units = max(1, num_models) * max(1, num_recycle)
        fold = _ETA_SEC_PER_RES_PER_UNIT * max(1, total_residues) * units
        return round(compile_cost + fold, 1)

    # ── Public API ────────────────────────────────────────────────────────────────

    def predict(
        self,
        sequence:    str,
        copies:      int = 1,
        template:    Optional[str] = None,
        num_models:  Optional[int] = None,
        num_recycle: Optional[int] = None,
        quick:       bool = False,
        label:       str = "colabfold",
    ) -> Dict[str, Any]:
        """
        Fold *sequence* (optionally as a homo-oligomer of *copies* chains).

        Parameters
        ----------
        sequence    : single-chain amino-acid sequence (one-letter).
        copies      : 1 = monomer; >1 = colon-joined homo-oligomer (multimer model).
        template    : optional PDB id or local .pdb path → custom-template mode.
                      De novo (no template) when None.
        num_models  : AF2 models to run (default config.COLABFOLD_NUM_MODELS).
        num_recycle : recycles per model (default config.COLABFOLD_NUM_RECYCLE).
        quick       : if True, force num_models=1, num_recycle=1 (plumbing preset).
        """
        import time as _time

        # ── Validate sequence ────────────────────────────────────────────────────
        if not sequence or not sequence.strip():
            return self._error("empty sequence")
        seq = "".join(sequence.split()).upper()
        bad = sorted(set(seq) - _VALID_AA)
        if bad:
            return self._error(
                f"sequence contains non-standard residue(s): {', '.join(bad)}. "
                "Use the 20 standard one-letter amino-acid codes."
            )

        try:
            copies = int(copies)
        except (TypeError, ValueError):
            copies = 1
        copies = max(1, copies)

        if quick:
            num_models, num_recycle = 1, 1
        n_models  = int(num_models  if num_models  is not None else _cfg.COLABFOLD_NUM_MODELS)
        n_recycle = int(num_recycle if num_recycle is not None else _cfg.COLABFOLD_NUM_RECYCLE)
        n_models  = max(1, min(5, n_models))
        n_recycle = max(1, n_recycle)

        length         = len(seq)
        total_residues = length * copies
        eta_s          = self.estimate_runtime_s(total_residues, n_models, n_recycle)

        # ── Total-residue guard (pre-launch OOM protection) ───────────────────────
        budget = int(getattr(_cfg, "COLABFOLD_MAX_TOTAL_RESIDUES", 1500))
        if total_residues > budget:
            return self._error(
                f"total residues {total_residues} (= {length} x {copies} copies) "
                f"exceeds the COLABFOLD_MAX_TOTAL_RESIDUES budget of {budget}. "
                "AlphaFold memory scales ~ (total residues)^2, so this would very "
                "likely OOM the laptop GPU. Try fewer copies, a shorter construct, "
                "raise COLABFOLD_MAX_TOTAL_RESIDUES if you have more VRAM, or run on "
                "a larger GPU.",
                oom_risk=True,
                extra={
                    "length": length, "copies": copies,
                    "total_residues": total_residues, "eta_s": eta_s,
                },
            )

        # ── Cache lookup ───────────────────────────────────────────────────────────
        cache_key = self._cache_key(seq, copies, template, n_models, n_recycle)
        cache_dir = Path(_cfg.COLABFOLD_CACHE_DIR) / f"colabfold_{cache_key}"
        cached = self._load_cache(cache_dir)
        if cached is not None:
            _pprint(f"  ColabFold: cache hit ({cache_dir.name}) — returning cached fold.")
            cached["cached"] = True
            cached["source"] = "cache"
            cached["eta_s"]  = 0.0
            return cached

        # ── Availability check (only when we actually need to fold) ────────────────
        if not self._wsl.is_available():
            return self._error(
                "WSL2 is not available. ColabFold runs in the WSL2 ~/colabfold_env. "
                "Install WSL2 (Ubuntu-24.04) first."
            )
        if not self._wsl.check_colabfold():
            return self._error(
                "ColabFold env not found in WSL2. Expected interpreter at "
                f"{COLABFOLD_PYTHON}. See PROJECT_CONTEXT.md §10 (ColabFold env setup)."
            )

        # ── Resolve template (optional) ────────────────────────────────────────────
        tmpl_wsl_dir = ""
        if template:
            tmpl_wsl_dir, terr = self._stage_template(template)
            if terr:
                return self._error(f"template error: {terr}")

        # ── Build worker + run ───────────────────────────────────────────────────────
        seq_line = ":".join([seq] * copies)   # colon-join → multimer when copies>1
        wsl_out  = f"/tmp/colabfold_{cache_key}"
        wsl_fasta = f"{wsl_out}/in.fasta"
        wsl_result = f"/tmp/colabfold_{cache_key}_result.json"
        script = self._build_worker(
            seq_line=seq_line, jobname=label, out_dir=wsl_out, fasta_path=wsl_fasta,
            result_path=wsl_result, n_models=n_models, n_recycle=n_recycle,
            msa_mode=str(getattr(_cfg, "COLABFOLD_MSA_MODE", "mmseqs2_uniref_env")),
            tmpl_dir=tmpl_wsl_dir,
            jax_compile_cache_dir=str(getattr(_cfg, "COLABFOLD_JAX_COMPILE_CACHE_DIR", "")),
        )

        # Scale the timeout with the workload; floor at COLABFOLD_TIMEOUT.
        timeout = max(
            int(getattr(_cfg, "COLABFOLD_TIMEOUT", 1800)),
            int(eta_s * 2 + 300),
        )
        _pprint(
            f"  ColabFold: folding {length} aa x {copies} "
            f"({'multimer' if copies > 1 else 'monomer'}), "
            f"{n_models} model(s) x {n_recycle} recycle(s). "
            f"Estimated ~{eta_s/60:.1f} min (approximate)."
        )

        t0 = _time.perf_counter()
        run = self._wsl.run_python_script(script, timeout=timeout, python_bin=COLABFOLD_PYTHON)
        elapsed_s = round(_time.perf_counter() - t0, 1)

        if run.get("stdout"):
            for line in run["stdout"].splitlines():
                if line.strip():
                    _pprint(f"  {line.strip()}")

        if not run["ok"]:
            why = (run.get("error") or "").strip() or str(run.get("stderr", ""))[:200]
            low = (why + str(run.get("stdout", ""))).lower()
            if any(k in low for k in ("out of memory", "resource_exhausted", "cuda_error_out_of_memory")):
                return self._error(
                    f"ColabFold ran out of GPU memory while folding {total_residues} "
                    "residues. Reduce copies/length or use a larger GPU.",
                    oom_risk=True, extra={"elapsed_s": elapsed_s, "eta_s": eta_s},
                )
            return self._error(f"ColabFold WSL2 run failed: {why}",
                               extra={"elapsed_s": elapsed_s, "eta_s": eta_s})

        # ── Copy results file back + parse ────────────────────────────────────────
        win_result = str(Path(tempfile.gettempdir()) / f"colabfold_{cache_key}_result.json")
        if not self._wsl.copy_from_wsl(wsl_result, win_result) or not Path(win_result).is_file():
            return self._error("worker produced no result file",
                               extra={"elapsed_s": elapsed_s})
        try:
            with open(win_result, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            return self._error(f"could not parse worker output ({exc})")

        if "error" in data and not data.get("success"):
            return self._error(
                f"ColabFold worker error: {str(data['error'])[:300]}",
                oom_risk=bool(data.get("oom")),
                extra={"elapsed_s": elapsed_s, "eta_s": eta_s},
            )

        # ── Copy ranked PDB + PNGs back to the Windows cache dir ────────────────────
        cache_dir.mkdir(parents=True, exist_ok=True)
        ranked_pdb = self._copy_back(data.get("ranked_pdb_wsl", ""), cache_dir, "ranked.pdb")
        if not ranked_pdb:
            return self._error("ranked PDB not returned by worker",
                               extra={"elapsed_s": elapsed_s})

        png_paths: Dict[str, str] = {}
        for kind in ("pae", "plddt", "coverage"):
            src = (data.get("pngs") or {}).get(kind, "")
            if src:
                dst = self._copy_back(src, cache_dir, f"{kind}.png")
                if dst:
                    png_paths[kind] = dst

        # Per-residue pLDDT → 1-based dict; guard 0-1 scale like ESMFold.
        plddt_list = data.get("plddt") or []
        plddt = {i + 1: round(float(v), 2) for i, v in enumerate(plddt_list)}
        mean_plddt = round(sum(plddt.values()) / len(plddt), 2) if plddt else 0.0
        if 0 < mean_plddt < 2.0:   # 0-1 scale guard
            plddt = {k: round(v * 100, 2) for k, v in plddt.items()}
            mean_plddt = round(mean_plddt * 100, 2)

        self._compiled_this_session = True

        result = {
            "success":        True,
            "error":          None,
            "oom_risk":       False,
            "ranked_pdb":     ranked_pdb,
            "mean_plddt":     mean_plddt,
            "plddt":          plddt,
            "pae":            data.get("pae"),
            "ptm":            data.get("ptm"),
            "iptm":           data.get("iptm"),
            "length":         length,
            "copies":         copies,
            "total_residues": total_residues,
            "num_models":     n_models,
            "num_recycle":    n_recycle,
            "png_paths":      png_paths,
            "cached":         False,
            "eta_s":          eta_s,
            "elapsed_s":      elapsed_s,
            "source":         "colabfold_wsl2",
        }
        self._save_cache(cache_dir, result)
        _pprint(
            f"  ColabFold: done — mean pLDDT {mean_plddt:.1f}, "
            f"pTM {result['ptm']}, {elapsed_s:.0f}s."
        )
        return result

    # ── Worker builder ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_worker(
        seq_line:   str,
        jobname:    str,
        out_dir:    str,
        fasta_path: str,
        result_path: str,
        n_models:   int,
        n_recycle:  int,
        msa_mode:   str,
        tmpl_dir:   str,
        jax_compile_cache_dir: str = "",
    ) -> str:
        """
        Build the standalone WSL2 worker (run via COLABFOLD_PYTHON).

        Rules (mirror rosetta_bridge): zero project imports, JSON-file I/O only,
        write the result file even on exception, literal braces doubled ``{{}}``.

        *jax_compile_cache_dir* (WSL2 path, '~' allowed) enables JAX's persistent
        compilation cache so XLA reuses compiled executables across the
        fresh-per-fold worker processes — a different sequence of the same length
        skips the ~10-min recompile. Empty string disables it.
        """
        return f"""
import json, os, sys, glob, subprocess

result_path = {result_path!r}

def _write(d):
    with open(result_path, "w") as fh:
        json.dump(d, fh)

try:
    out_dir    = {out_dir!r}
    fasta_path = {fasta_path!r}
    seq_line   = {seq_line!r}
    jobname    = {jobname!r}
    tmpl_dir   = {tmpl_dir!r}
    os.makedirs(out_dir, exist_ok=True)

    # JAX persistent compilation cache (shared across worker processes). AF2
    # compiles take many seconds, far above jax's min-compile-time threshold, so
    # they are cached; we set the threshold low explicitly to be version-robust.
    jax_cache = {jax_compile_cache_dir!r}
    if jax_cache:
        jax_cache = os.path.expanduser(jax_cache)
        os.makedirs(jax_cache, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = jax_cache
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"
        print("[colabfold] JAX compile cache: " + jax_cache, flush=True)

    with open(fasta_path, "w") as fh:
        fh.write(">" + jobname + "\\n" + seq_line + "\\n")

    # colabfold_batch console script lives alongside this interpreter.
    cf = os.path.join(os.path.dirname(sys.executable), "colabfold_batch")
    cmd = [cf,
           "--num-models",  {str(n_models)!r},
           "--num-recycle", {str(n_recycle)!r},
           "--msa-mode",    {msa_mode!r}]
    if tmpl_dir:
        cmd += ["--templates", "--custom-template-path", tmpl_dir]
    cmd += [fasta_path, out_dir]
    print("[colabfold] " + " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + "\\n" + (proc.stderr or "")
    if "Running on GPU" in log:
        print("[colabfold] GPU confirmed", flush=True)
    elif "Running on CPU" in log:
        print("[colabfold] WARNING: running on CPU", flush=True)

    if proc.returncode != 0:
        low = log.lower()
        oom = any(k in low for k in
                  ("out of memory", "resource_exhausted", "cuda_error_out_of_memory"))
        _write({{"success": False, "error": log[-3000:], "oom": oom}})
        sys.exit(0)

    # rank_001 = top-ranked model. Prefer relaxed if present, else unrelaxed.
    pdbs = (sorted(glob.glob(os.path.join(out_dir, "*_relaxed_rank_001_*.pdb")))
            or sorted(glob.glob(os.path.join(out_dir, "*_unrelaxed_rank_001_*.pdb")))
            or sorted(glob.glob(os.path.join(out_dir, "*_rank_001_*.pdb"))))
    scores = sorted(glob.glob(os.path.join(out_dir, "*_scores_rank_001_*.json")))
    if not pdbs or not scores:
        _write({{"success": False,
                 "error": "no rank_001 outputs found in " + out_dir + "; log:\\n" + log[-1500:],
                 "oom": False}})
        sys.exit(0)

    with open(scores[0]) as fh:
        sc = json.load(fh)
    plddt = sc.get("plddt") or []
    pae   = sc.get("pae") or sc.get("predicted_aligned_error")
    ptm   = sc.get("ptm")
    iptm  = sc.get("iptm")

    def _first(pat):
        hits = sorted(glob.glob(os.path.join(out_dir, pat)))
        return hits[0] if hits else ""

    pngs = {{"pae":      _first("*_pae.png"),
             "plddt":    _first("*_plddt.png"),
             "coverage": _first("*_coverage.png")}}

    _write({{"success": True,
             "ranked_pdb_wsl": pdbs[0],
             "plddt": plddt, "pae": pae, "ptm": ptm, "iptm": iptm,
             "pngs": pngs, "log_tail": log[-1500:]}})
    print("[colabfold] worker done", flush=True)

except Exception as exc:
    import traceback
    traceback.print_exc()
    try:
        _write({{"success": False, "error": str(exc), "oom": False}})
    except Exception:
        pass
"""

    # ── Template staging ──────────────────────────────────────────────────────────

    def _stage_template(self, template: str) -> Tuple[str, Optional[str]]:
        """
        Stage a custom template into a WSL2 dir for --custom-template-path.

        *template* may be a local .pdb path (copied into WSL2) or a 4-char PDB
        id (the caller is expected to have downloaded it; we accept a path). On
        success returns (wsl_dir, None). Returns ("", error) on failure.
        """
        p = Path(template)
        if not p.is_file():
            return "", (
                f"template '{template}' is not a local file. v1 accepts a local "
                ".pdb/.cif template path (download a PDB id first)."
            )
        wsl_path = self._wsl.copy_to_wsl(str(p.resolve()), dest_dir="/tmp/colabfold_templates")
        if not wsl_path:
            return "", f"failed to copy template {template} into WSL2"
        return "/tmp/colabfold_templates", None

    # ── Cache helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(
        seq: str, copies: int, template: Optional[str],
        n_models: int, n_recycle: int,
    ) -> str:
        tmpl_tag = ""
        if template:
            try:
                tmpl_tag = hashlib.md5(Path(template).read_bytes()).hexdigest()[:8]
            except Exception:
                tmpl_tag = hashlib.md5(str(template).encode()).hexdigest()[:8]
        raw = f"{seq}|c{copies}|t{tmpl_tag}|m{n_models}|r{n_recycle}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _load_cache(cache_dir: Path) -> Optional[Dict[str, Any]]:
        meta = cache_dir / "result.json"
        if not meta.is_file():
            return None
        try:
            with open(meta, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return None
        # Cache is only valid if the ranked PDB still exists on disk.
        if not data.get("ranked_pdb") or not Path(data["ranked_pdb"]).is_file():
            return None
        # JSON keys are strings — restore the 1-based int pLDDT keys.
        if isinstance(data.get("plddt"), dict):
            data["plddt"] = {int(k): v for k, v in data["plddt"].items()}
        return data

    @staticmethod
    def _save_cache(cache_dir: Path, result: Dict[str, Any]) -> None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / "result.json", "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
        except Exception as exc:
            _pprint(f"  ColabFold: warning — could not write cache ({exc})")

    def _copy_back(self, wsl_path: str, cache_dir: Path, dest_name: str) -> str:
        """Copy a WSL2 file into the Windows cache dir; return its Windows path or ''."""
        if not wsl_path:
            return ""
        dest = cache_dir / dest_name
        if self._wsl.copy_from_wsl(wsl_path, str(dest)) and dest.is_file():
            return str(dest)
        return ""

    # ── Error helper ──────────────────────────────────────────────────────────────

    @staticmethod
    def _error(
        message:  str,
        oom_risk: bool = False,
        extra:    Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = {
            "success":        False,
            "error":          message,
            "oom_risk":       oom_risk,
            "ranked_pdb":     "",
            "mean_plddt":     0.0,
            "plddt":          {},
            "pae":            None,
            "ptm":            None,
            "iptm":           None,
            "length":         0,
            "copies":         1,
            "total_residues": 0,
            "num_models":     0,
            "num_recycle":    0,
            "png_paths":      {},
            "cached":         False,
            "eta_s":          0.0,
            "elapsed_s":      0.0,
            "source":         "error",
        }
        if extra:
            base.update(extra)
        return base

    def __repr__(self) -> str:
        return f"<ColabFoldBridge python={COLABFOLD_PYTHON!r} available={self.is_available()}>"
