"""
session_io.py
-------------
Named-session save / load / list — the SINGLE source of truth shared by both front-ends
(`main.py` CLI + `gui_app.py` workbench). UI-agnostic: takes a ChimeraX bridge + a
SessionState and returns structured result dicts; each front-end renders its own way.

A named session is a DIRECTORY `config.SESSION_DIR/{name}/`:
  session.json   — StructureBot state (SessionState.save; the source of truth for listing)
  scene.cxs      — the ChimeraX scene (embeds coordinates → self-contained + durable, so a
                   load can re-display models NOT open in the current ChimeraX)
  folds/         — COPIES of the fold CIFs/PDBs referenced by template_fold/guided_fold/wt_refs.
                   Boltz writes folds to the system temp dir (subject to OS cleanup); copying
                   them here makes the saved session durable + self-contained, and the saved
                   session.json's path fields are rewritten to point at these copies.
  exports/       — CSVs, lDDT maps, figures, etc., written here as generated (kept across saves).

Fail-loud by contract: load validates session.json via SessionState.try_load FIRST and returns
its `(None, msg)` error rather than ever silently producing a fresh state.
"""
from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import config
from session_state import SessionState


def sanitize_session_name(name: str) -> str:
    """Filesystem-safe session name (mirrors the original CLI sanitiser). Empty → 'default'."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", (name or "").strip()) or "default"


def session_dir(name: str) -> Path:
    return config.SESSION_DIR / sanitize_session_name(name)


def session_paths(name: str) -> Dict[str, Path]:
    """The directory + its four well-known members for a (sanitised) name."""
    d = session_dir(name)
    return {"dir": d, "json": d / "session.json", "cxs": d / "scene.cxs",
            "folds": d / "folds", "exports": d / "exports"}


def list_saved_sessions() -> List[str]:
    """Saved session NAMES — every `{name}/session.json` directory under SESSION_DIR (state =
    source of truth), sorted. The picker shows these."""
    try:
        return sorted(p.name for p in config.SESSION_DIR.iterdir()
                      if p.is_dir() and (p / "session.json").is_file())
    except OSError:
        return []


# Path fields inside a persisted design that reference a fold artifact on disk.
_FOLD_SLOTS = ("template_fold", "guided_fold")
_FOLD_KEYS = ("cif_path", "pdb_path")


def _relocate_folds(design_sessions: Dict[str, Any], folds_dir: Path) -> Dict[str, Any]:
    """DEEP-COPY *design_sessions*, copy every referenced fold file into *folds_dir*, and rewrite
    the path fields to the durable copies. The LIVE session is never mutated (the caller swaps in
    this copy only for the on-disk write). Best-effort: a path whose file is already gone is left
    as-is (load surfaces a missing-file warning rather than this silently dropping it)."""
    out = copy.deepcopy(design_sessions or {})
    mapping: Dict[str, str] = {}                       # src abspath -> dst abspath (dedupe per save)

    def relocate(holder: Any, key: str) -> None:
        if not isinstance(holder, dict):
            return
        src = holder.get(key)
        if not isinstance(src, str) or not src:
            return
        if src in mapping:
            holder[key] = mapping[src]
            return
        p = Path(src)
        if not p.is_file():
            return
        dst = folds_dir / p.name
        n = 1
        while dst.exists():                            # distinct src wanting the same name → unique
            dst = folds_dir / f"{p.stem}__{n}{p.suffix}"
            n += 1
        try:
            shutil.copyfile(p, dst)
        except OSError:
            return
        mapping[src] = str(dst)
        holder[key] = str(dst)

    for ds in out.values():
        for cd in (ds.get("chains") or {}).values():
            for slot in _FOLD_SLOTS:
                for key in _FOLD_KEYS:
                    relocate(cd.get(slot), key)
            for ref in (cd.get("wt_refs") or {}).values():
                relocate(ref, "path")
    return out


def save_named_session(bridge, session: SessionState, name: str) -> Dict[str, Any]:
    """Save the session directory: scene.cxs + session.json (+ copied folds/). Never raises.
    Returns {name, dir, cxs_path, json_path, cxs_ok, cxs_error, json_error}. The state is always
    written even if the scene save fails (state is the source of truth). folds/ is rebuilt each
    save (stale copies dropped); exports/ is preserved."""
    paths = session_paths(name)
    info: Dict[str, Any] = {"name": sanitize_session_name(name), "dir": paths["dir"],
                            "cxs_path": paths["cxs"], "json_path": paths["json"],
                            "cxs_ok": False, "cxs_error": None, "json_error": None}
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["exports"].mkdir(parents=True, exist_ok=True)
    # Rebuild folds/ from scratch so it mirrors only the CURRENT save (no stale accumulation).
    shutil.rmtree(paths["folds"], ignore_errors=True)
    paths["folds"].mkdir(parents=True, exist_ok=True)

    if bridge is not None:
        try:
            res = bridge.run_command(f'save "{paths["cxs"].as_posix()}"')
            info["cxs_error"] = res.get("error")
            info["cxs_ok"] = not res.get("error")
        except Exception as exc:                       # bridge down / REST error — non-fatal
            info["cxs_error"] = f"{type(exc).__name__}: {exc}"
    else:
        info["cxs_error"] = "no ChimeraX bridge (scene not saved)"

    try:
        original = session.design_sessions
        session.design_sessions = _relocate_folds(original, paths["folds"])
        try:
            session.save(str(paths["json"]))
        finally:
            session.design_sessions = original         # never leave the live session rewritten
    except Exception as exc:
        info["json_error"] = f"{type(exc).__name__}: {exc}"
    return info


def load_named_session(bridge, name: str) -> Dict[str, Any]:
    """Load a named session. Validates session.json FIRST (fail-loud) and only then reopens
    scene.cxs. Returns {name, dir, cxs_path, json_path, state, error, cxs_ok, cxs_error}:
      - error set (state None) → caller must surface it and NOT swap session (never silent fresh).
      - state set → the loaded SessionState; cxs_ok/cxs_error report the scene reopen.
    Caller (GUI) does the model-id REMAP against the reopened scene after this returns."""
    paths = session_paths(name)
    info: Dict[str, Any] = {"name": sanitize_session_name(name), "dir": paths["dir"],
                            "cxs_path": paths["cxs"], "json_path": paths["json"], "state": None,
                            "error": None, "cxs_ok": False, "cxs_error": None}
    if not paths["json"].is_file():
        info["error"] = f"session not found: {sanitize_session_name(name)}"
        return info
    state, err = SessionState.try_load(str(paths["json"]))
    if err or state is None:
        info["error"] = err or "could not load session state"   # FAIL-LOUD
        return info
    info["state"] = state
    if paths["cxs"].is_file() and bridge is not None:
        try:
            res = bridge.run_command(f'open "{paths["cxs"].as_posix()}"')
            info["cxs_error"] = res.get("error")
            info["cxs_ok"] = not res.get("error")
        except Exception as exc:
            info["cxs_error"] = f"{type(exc).__name__}: {exc}"
    elif not paths["cxs"].is_file():
        info["cxs_error"] = "no scene.cxs (state restored; 3D not reopened)"
    return info
