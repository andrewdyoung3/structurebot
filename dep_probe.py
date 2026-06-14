"""
dep_probe.py
------------
Shared capability probes for external-dependency bridges — the generalization of
the cavity-class fix: an availability flag must mean "can actually run", not
"files exist". A bridge that runs as a SUBPROCESS can have its interpreter FILE
present while the import chain in that environment is silently broken (a bad/
missing dep, a torch ABI mismatch). These probes spawn the real interpreter WHERE
THE BRIDGE RUNS and confirm the import chain resolves — imports only, never a
model load / inference.

Design contract (proven on ThermoMPNN, mirrored here):
  - CHEAP: import the chain, do not load weights or run inference.
  - CACHED per key: spawned at most once/session.
  - GRACEFUL: a DEFINITIVE verdict (the probe ran to completion) is cached; a
    probe-INFRASTRUCTURE failure (spawn error / timeout) returns False WITHOUT
    caching, so a transient error never masquerades permanently as "absent".
  - This does NOT change the bridge's graceful-degrade contract — it only makes
    availability report absence CORRECTLY.

NOT for API bridges (e.g. DynaMut2): there is nothing to import-probe, a probe
burns an API call / rate-limit, and availability is transient — a cached flag
would be wrong. Those rely on graceful API handling + voter visibility.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Dict, List, Optional, Tuple

_CREATE_NO_WINDOW: int = (
    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]
)

_PROBE_CACHE: Dict[tuple, bool] = {}

_SENTINEL = "DEP_PROBE_IMPORT_OK"


def reset_probe_cache() -> None:
    """Clear all cached capability verdicts (tests; or after an env change)."""
    _PROBE_CACHE.clear()


def _build_code(path_dirs: List[str], imports: List[str]) -> str:
    pathins = ""
    if path_dirs:
        joined = ", ".join(repr(str(d)) for d in path_dirs)
        pathins = f"import sys; sys.path[:0] = [{joined}]\n"
    body = "\n".join(imports)
    return f"{pathins}{body}\nprint({_SENTINEL!r})\n"


def local_import_probe(
    interpreter: str,
    imports:     List[str],
    path_dirs:   Optional[List[str]] = None,
    timeout:     int = 60,
    cache_key:   Optional[tuple] = None,
) -> bool:
    """Spawn *interpreter* (a LOCAL python, e.g. venv312) and confirm *imports*
    resolve with *path_dirs* prepended to sys.path. See module contract."""
    key = cache_key or ("local", interpreter, tuple(path_dirs or ()), tuple(imports))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    code = _build_code(path_dirs or [], imports)
    try:
        r = subprocess.run(
            [interpreter, "-c", code],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
        )
    except Exception:
        return False   # probe infra failure → not definitive → not cached
    ok = (r.returncode == 0) and (_SENTINEL in (r.stdout or ""))
    _PROBE_CACHE[key] = ok
    return ok


def wsl_import_probe(
    wsl,
    interpreter: str,
    imports:     List[str],
    path_dirs:   Optional[List[str]] = None,
    timeout:     int = 60,
    cache_key:   Optional[tuple] = None,
) -> bool:
    """Run the import probe inside WSL via *wsl* (a WSLBridge), using its
    *interpreter* (a WSL path). Confirms the chain resolves in the WSL env where
    the bridge runs — stronger than a `test -x` presence check."""
    key = cache_key or ("wsl", interpreter, tuple(path_dirs or ()), tuple(imports))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    if not wsl.is_available():
        return False   # WSL itself down → not a definitive bridge verdict → no cache
    code = _build_code(path_dirs or [], imports)
    # single-quote the -c payload for bash; the code has no single quotes
    cmd = f"{interpreter} -c {_shq(code)}"
    try:
        r = wsl.run_command(cmd, timeout=timeout)
    except Exception:
        return False
    ok = bool(r.get("ok")) and (_SENTINEL in (r.get("stdout", "") or ""))
    _PROBE_CACHE[key] = ok
    return ok


def _shq(s: str) -> str:
    """POSIX single-quote a string for bash -c."""
    return "'" + s.replace("'", "'\"'\"'") + "'"
