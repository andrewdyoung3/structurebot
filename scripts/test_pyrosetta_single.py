#!/usr/bin/env python3
"""
scripts/test_pyrosetta_single.py
--------------------------------
Standalone probe for the PyRosetta / WSL2 local ddG path.

Runs RosettaBridge._run_rosetta_local() on a SINGLE mutation (I72R, chain A
of 1HSG), bypassing the full mutation scan, so we can capture the exact
[Rosetta DEBUG] output (cleanATOM size-guard outcome, the pose_from_file
target path / size / first lines, and any worker traceback) in ~1 minute
instead of running the full ~30-minute scan.

Usage:
    python scripts/test_pyrosetta_single.py

Requires ROSETTA_BACKEND=local (PyRosetta in WSL2). The script forces it on
regardless of .env.local so the local path is exercised directly.

This script changes no other files. The [Rosetta DEBUG] diagnostics it relies
on live in rosetta_bridge.py and are marked TEMP DIAGNOSTIC.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

# ── Project path setup ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Force the local PyRosetta/WSL2 backend BEFORE importing the bridge
# (_select_backend reads ROSETTA_BACKEND at RosettaBridge() construction time).
import config  # noqa: E402

config.load_env_file()
os.environ["ROSETTA_BACKEND"] = "local"

from rosetta_bridge import RosettaBridge, _mutation_key  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _p(msg: str = "") -> None:
    """ASCII-safe print (Windows consoles choke on the bridge's emoji)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _rule(title: str = "") -> None:
    _p("=" * 78)
    if title:
        _p(title)
        _p("=" * 78)


def _ensure_1hsg() -> Path:
    """Return path to cache/1HSG.pdb, downloading from RCSB if missing."""
    cache_dir = _ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = cache_dir / "1HSG.pdb"
    if pdb.is_file() and pdb.stat().st_size > 1000:
        _p(f"Using cached PDB: {pdb}  ({pdb.stat().st_size} bytes)")
        return pdb

    _p(f"cache/1HSG.pdb missing — downloading from RCSB...")
    import requests
    url = "https://files.rcsb.org/download/1HSG.pdb"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    pdb.write_bytes(resp.content)
    _p(f"Downloaded {pdb}  ({pdb.stat().st_size} bytes)")
    return pdb


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    _rule("PyRosetta / WSL2 single-mutation probe — I72R (chain A, 1HSG)")

    pdb_path = _ensure_1hsg()

    mutation = {
        "chain":    "A",
        "position": 72,
        "from_aa":  "I",
        "to_aa":    "R",
    }
    mut_key = _mutation_key(mutation)
    _p(f"Mutation: {mut_key}  (chain {mutation['chain']}, pos {mutation['position']})")

    bridge = RosettaBridge()
    _p(f"Selected backend: {bridge._backend!r}")
    if bridge._backend != "local":
        _p("WARNING: backend is not 'local' — check ROSETTA_BACKEND / .env.local.")
    _p(f"Backend status: {bridge.backend_status()}")

    # Capture every progress line the bridge emits (includes the live
    # [Rosetta DEBUG] / [Rosetta] worker stdout lines).
    captured: list[str] = []

    def _progress(msg: str) -> None:
        captured.append(msg)
        _p(msg)

    _rule("LIVE WORKER OUTPUT")
    result = bridge._run_rosetta_local(
        pdb_path          = str(pdb_path),
        mutations         = [mutation],
        model_id          = "1",
        chain             = "A",
        progress_callback = _progress,
    )

    # ── Structured debug from the worker results JSON ─────────────────────────
    # _run_rosetta_local writes the worker output to %TEMP%/rosetta_ddg_<hash>.json
    # (hash = md5 of the original PDB bytes, first 12 hex chars). The "debug" key
    # survives there even though the Python side pops it out of its in-memory copy.
    _rule("STRUCTURED WORKER DEBUG (from results JSON)")
    pdb_hash = hashlib.md5(Path(pdb_path).read_bytes()).hexdigest()[:12]
    win_results = Path(tempfile.gettempdir()) / f"rosetta_ddg_{pdb_hash}.json"
    _p(f"Worker results JSON: {win_results}")
    if win_results.is_file():
        try:
            raw = json.loads(win_results.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            raw = None
            _p(f"  (could not parse results JSON: {exc})")
        if isinstance(raw, dict):
            dbg = raw.get("debug")
            if dbg is not None:
                _p(json.dumps(dbg, indent=2, default=str))
                # Highlight the fields we care about most
                _rule("KEY DIAGNOSTIC FIELDS")
                clean = dbg.get("cleanATOM", {}) if isinstance(dbg, dict) else {}
                load  = dbg.get("load", {})      if isinstance(dbg, dict) else {}
                _p(f"input_path           : {dbg.get('input_path')}")
                _p(f"cleanATOM.cleaned    : {clean.get('cleaned_path')}")
                _p(f"cleanATOM.exists     : {clean.get('cleaned_exists')}")
                _p(f"cleanATOM.size       : {clean.get('cleaned_size')} bytes")
                _p(f"cleanATOM.used_clean : {clean.get('used_cleaned')}  "
                   f"(guard kept: {clean.get('kept_path')})")
                _p(f"pose_from_file path  : {load.get('path')}")
                _p(f"pose_from_file exists: {load.get('exists')}")
                _p(f"pose_from_file size  : {load.get('size')} bytes")
                _p(f"first 5 lines        : {load.get('head')}")
                if dbg.get("renamed_to"):
                    _p(f"renamed_to .pdb      : {dbg.get('renamed_to')}")
                if dbg.get("pose_from_file_error"):
                    _p(f"pose_from_file error : {dbg.get('pose_from_file_error')}")
                if dbg.get("cleanATOM_error"):
                    _p(f"cleanATOM error      : {dbg.get('cleanATOM_error')}")
                if dbg.get("traceback"):
                    _rule("WORKER TRACEBACK")
                    _p(dbg["traceback"])
            else:
                _p("  (no 'debug' key in results JSON — older worker or write failed)")
            if "error" in raw:
                _p(f"\nWorker FATAL error string: {raw['error']}")
    else:
        _p("  (results JSON not found — worker may not have run; see live output above)")

    # ── Final result ──────────────────────────────────────────────────────────
    _rule("FINAL RESULT")
    _p(f"success      : {result.success}")
    if not result.success:
        _p(f"error        : {result.error}")
    data = result.data or {}
    _p(f"backend      : {data.get('backend')}")
    _p(f"confidence   : {data.get('confidence')}")
    ddg_scores = data.get("ddg_scores", {})
    ddg_source = data.get("ddg_source", {})
    _p(f"ddg_scores   : {ddg_scores}")
    _p(f"ddg_source   : {ddg_source}")
    if mut_key in ddg_scores:
        _p(f"\n>>> {mut_key}: ddg = {ddg_scores[mut_key]:+.3f} kcal/mol "
           f"(source = {ddg_source.get(mut_key, 'unknown')})")
    for w in data.get("warnings", []) or []:
        _p(f"warning      : {w}")

    _rule()
    if ddg_source.get(mut_key) == "pyrosetta":
        _p("RESULT: real PyRosetta ddG computed — local path is WORKING.")
    elif ddg_source.get(mut_key) == "empirical":
        _p("RESULT: PyRosetta failed; empirical fallback used. See worker debug above "
           "for the exact pose_from_file failure.")
    else:
        _p("RESULT: no ddG produced. See live output / error above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
