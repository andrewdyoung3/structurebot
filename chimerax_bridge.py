"""
chimerax_bridge.py
------------------
Manages the connection between StructureBot and UCSF ChimeraX via its REST server.

ChimeraX REST server is started with:
    remotecontrol rest start port 60001

Commands are POSTed to:
    http://localhost:60001/run

Response format (JSON):
    {"value": "output text", "error": null}
    {"value": null,          "error": "error message"}
"""

import glob
import json
import os
import queue
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import requests

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PORT = 60001
REQUEST_TIMEOUT = 30  # seconds per command
STARTUP_TIMEOUT = 45  # seconds to wait for ChimeraX to start

# Common install locations on Windows, ordered by likelihood.
# Supports glob wildcards so version numbers don't need to be exact.
_WINDOWS_SEARCH_PATTERNS = [
    r"C:\Users\{user}\Documents\ChimeraX*\bin\ChimeraX.exe",
    r"C:\Program Files\ChimeraX*\bin\ChimeraX.exe",
    r"C:\Program Files (x86)\ChimeraX*\bin\ChimeraX.exe",
    r"C:\Users\{user}\AppData\Local\UCSF\ChimeraX\bin\ChimeraX.exe",
    r"C:\Users\{user}\AppData\Local\UCSF\ChimeraX*\bin\ChimeraX.exe",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_chimerax() -> Optional[str]:
    """
    Search common Windows install locations for the ChimeraX executable.
    Returns the path to the newest version found, or None.
    Respects the CHIMERAX_PATH environment variable if set.
    """
    env_override = os.environ.get("CHIMERAX_PATH")
    if env_override:
        if Path(env_override).is_file():
            return env_override
        print(f"[warn] CHIMERAX_PATH is set but '{env_override}' does not exist.")

    user = os.environ.get("USERNAME", os.environ.get("USER", "*"))
    matches: List[str] = []

    for pattern in _WINDOWS_SEARCH_PATTERNS:
        expanded = pattern.format(user=user)
        found = glob.glob(expanded)
        matches.extend(found)

    if not matches:
        return None

    # Prefer the lexicographically last entry (usually the highest version number)
    matches.sort(reverse=True)
    return matches[0]


# ── Main class ────────────────────────────────────────────────────────────────

class ChimeraXBridge:
    """
    Manages starting, connecting to, and communicating with a UCSF ChimeraX
    instance via its built-in REST server.

    Typical usage::

        bridge = ChimeraXBridge()
        bridge.start()               # launch ChimeraX + REST server
        result = bridge.run_command("open 1abc")
        bridge.stop()
    """

    def __init__(
        self,
        chimerax_path: Optional[str] = None,
        port: int = DEFAULT_PORT,
    ):
        self.chimerax_path: Optional[str] = chimerax_path or find_chimerax()
        self.port: int = port
        # Use 127.0.0.1, NOT "localhost": on Windows "localhost" resolves IPv6-first
        # (::1) and the REST server binds IPv4 only, so each connection eats a ~2 s
        # IPv6 connect-stall before falling back to IPv4 (measured: localhost 2054 ms
        # vs 127.0.0.1 29 ms per call). The hot REST path multiplies this per click.
        self.base_url: str = f"http://127.0.0.1:{port}"
        self.run_url: str = f"{self.base_url}/run"
        self._process: Optional[subprocess.Popen] = None
        self._command_queue: queue.Queue = queue.Queue()
        # Once-per-session guard for the lean window layout (Log/CLI/Toolbar hidden).
        self._lean_layout_applied: bool = False
        # Optional fire-and-forget post-open hook: called with the REAL opened model id
        # (digits, from _opened_model_id) after a successful structure open. The unified
        # GUI sets this to capture ground-truth ids for tab focus; left None elsewhere
        # (no-op). Never alters results, never aborts the open.
        self.on_structure_opened: Optional[Callable[[str], None]] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, timeout: int = STARTUP_TIMEOUT) -> bool:
        """
        Start ChimeraX with the REST server enabled on self.port.
        If a REST server is already reachable at that port, does nothing.

        Returns True on success; raises on failure.
        """
        if not self.chimerax_path:
            raise FileNotFoundError(
                "ChimeraX executable not found.\n"
                "  • Set the CHIMERAX_PATH environment variable to the full path of ChimeraX.exe, or\n"
                "  • Pass chimerax_path='...' when constructing ChimeraXBridge()."
            )

        if not Path(self.chimerax_path).is_file():
            raise FileNotFoundError(
                f"ChimeraX not found at: {self.chimerax_path}\n"
                "Please verify the installation path."
            )

        if self.is_running():
            return True  # already up — just reuse

        cmd = [
            self.chimerax_path,
            "--cmd",
            f"remotecontrol rest start port {self.port}",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Don't inherit the calling terminal so ChimeraX gets its own window
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        )

        # Poll until the REST server responds
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            # Check if ChimeraX crashed before the server came up
            if self._process.poll() is not None:
                stderr_out = self._process.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"ChimeraX exited prematurely (code {self._process.returncode}).\n"
                    f"stderr: {stderr_out[:500]}"
                )
            time.sleep(0.75)

        raise TimeoutError(
            f"ChimeraX REST server did not become reachable within {timeout}s.\n"
            "Try increasing the timeout or check that ChimeraX starts cleanly."
        )

    def ensure_visible_gui(self, timeout: int = STARTUP_TIMEOUT) -> str:
        """
        Guarantee a ChimeraX with a *visible* GUI window is up and reachable on
        self.port, then return how that was achieved:

            "connected"  — a visible ChimeraX was already running; reused as-is
            "started"    — nothing was running; launched a fresh ChimeraX
            "relaunched" — a leftover *windowless* ChimeraX (REST-reachable but with
                           no GUI window — e.g. a zombie from a prior session that
                           StructureBot deliberately left running) was squatting the
                           port; it is killed and a fresh visible instance launched

        This is stricter than start(), which reuses ANY reachable REST server.
        Reusing a windowless instance is exactly the failure the user hits: models
        open into an invisible viewer and "nothing appears". Raises on launch
        failure (same contract as start()).
        """
        if self.is_running():
            if self._visible_chimerax_window_exists():
                return "connected"
            # A windowless leftover is holding the port. Replace it: nothing visible
            # is being discarded (the precondition is "no visible ChimeraX window").
            self._kill_all_chimerax()
            deadline = time.time() + 10
            while time.time() < deadline and self.is_running():
                time.sleep(0.3)
            self.start(timeout=timeout)
            return "relaunched"
        self.start(timeout=timeout)
        return "started"

    @staticmethod
    def _visible_chimerax_window_exists() -> bool:
        """
        True if some ChimeraX process owns a visible top-level window (Windows).

        On non-Windows platforms, or if the probe fails for any reason, returns
        True (assume a usable GUI — never block startup on a best-effort check).
        """
        if sys.platform != "win32":
            return True
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

            found: List[int] = []

            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def _cb(hwnd, _lparam):
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                handle = kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
                if handle:
                    try:
                        buf = ctypes.create_unicode_buffer(32768)
                        size = wintypes.DWORD(len(buf))
                        if kernel32.QueryFullProcessImageNameW(
                                handle, 0, buf, ctypes.byref(size)):
                            if buf.value.lower().endswith("chimerax.exe"):
                                found.append(pid.value)
                    finally:
                        kernel32.CloseHandle(handle)
                return not found      # stop enumerating once a match is found

            user32.EnumWindows(_cb, 0)
            return bool(found)
        except Exception:
            return True

    def _kill_all_chimerax(self) -> None:
        """
        Force-kill every ChimeraX process (Windows). Only called once we have
        confirmed NO visible ChimeraX window exists, so nothing the user can see
        is discarded. Best-effort; never raises.
        """
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ChimeraX.exe"],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                pass
        self._process = None

    def stop(self, graceful: bool = True) -> None:
        """
        Stop the ChimeraX process.  Sends 'quit' via REST first if graceful=True.
        """
        if graceful and self.is_running():
            try:
                self.run_command("quit")
            except Exception:
                pass

        if self._process is not None:
            try:
                self._process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    # ── Connectivity ───────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """
        Return True if the ChimeraX REST server is currently reachable.
        Uses a short timeout so callers aren't blocked long.
        """
        try:
            resp = requests.get(self.base_url, timeout=2)
            # ChimeraX returns 200 or 404 for GET /  — either means it's alive
            return resp.status_code in (200, 404, 405)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    def ensure_connected(self, auto_start: bool = True) -> bool:
        """
        Verify the connection; optionally restart ChimeraX if it has gone away.
        Returns True if connected after the call.
        """
        if self.is_running():
            return True
        if auto_start:
            return self.start()
        return False

    # ── Command execution ──────────────────────────────────────────────────────

    def run_command(self, command: str, timeout: int = REQUEST_TIMEOUT) -> dict:
        """
        Execute a single ChimeraX command via the REST API.

        Resilient to a dropped REST connection (e.g. after a long-running
        operation): if the server is unreachable, attempt a single reconnect
        via ensure_connected() and retry the command once.  If reconnection
        succeeds the retry proceeds silently; if it fails, a ConnectionError is
        raised telling the user to check that ChimeraX is still open.

        Returns a dict with at least one of:
            {"value": "...", "error": None}   — success
            {"value": None,  "error": "..."}  — ChimeraX reported an error / timeout
        """
        try:
            return self._run_command_once(command, timeout)
        except ConnectionError as exc:
            # REST server unreachable (dropped connection). Try to reconnect once.
            if self._try_reconnect():
                return self._run_command_once(command, timeout)
            raise ConnectionError(
                f"ChimeraX REST server is not reachable and automatic "
                f"reconnection failed while running {command!r}.\n"
                "Check that ChimeraX is still open — the REST server stops when "
                "ChimeraX is closed.\n"
                f"(original error: {exc})"
            ) from exc

    def _try_reconnect(self) -> bool:
        """
        Attempt a single reconnect to the ChimeraX REST server after a dropped
        connection.  Returns True if the server is reachable afterward.  Never
        raises (failures are reported via the return value).
        """
        try:
            return self.ensure_connected(auto_start=True)
        except Exception:
            return False

    def _run_command_once(self, command: str, timeout: int = REQUEST_TIMEOUT) -> dict:
        """
        Execute a single ChimeraX command via the REST API (one attempt, no
        reconnect).  Raises ConnectionError if the REST server is unreachable
        (either at the pre-check or mid-request); run_command() handles the
        reconnect/retry around this method.
        """
        if not self.is_running():
            raise ConnectionError(
                "ChimeraX REST server is not reachable. "
                "Call start() or ensure_connected() first."
            )

        try:
            # ChimeraX REST server ONLY accepts GET with a URL query parameter:
            #   GET http://localhost:PORT/run?command=<urlencoded_command>
            # POST (form-encoded, JSON, or plain-text) returns
            # '"command" parameter missing' and does nothing.
            resp = requests.get(
                self.run_url,
                params={"command": command},   # → ?command=<urlencoded_command>
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            return {"value": None, "error": f"Request timed out after {timeout}s: {command!r}"}
        except requests.exceptions.ConnectionError as exc:
            # Raise so run_command() can attempt one reconnect + retry.
            raise ConnectionError(
                f"Connection lost during command {command!r}: {exc}"
            ) from exc

        # ChimeraX 1.x returns plain text (not JSON) for all responses.
        # Errors are returned as 200 OK with an error-prefixed string, e.g.:
        #   "Unknown command: foo\n"
        # Successful commands with output return the output as plain text.
        # Successful commands with no output (style/color/etc.) return "".
        body = resp.text.strip()

        # Build the result dict from the body (never early-return here so that
        # the post-save check below always runs for save commands).
        if not body:
            data: dict = {"value": "", "error": None}
        else:
            # Try JSON first in case a future ChimeraX version switches formats
            try:
                data = resp.json()
                if not isinstance(data, dict):
                    data = {"value": str(data), "error": None}
            except ValueError:
                data = {"value": body, "error": None}

        # Normalise: ensure both keys exist
        data.setdefault("value", None)
        data.setdefault("error", None)

        # HTTP-level errors
        if resp.status_code >= 400 and data.get("error") is None:
            data["error"] = f"HTTP {resp.status_code}: {body[:200]}"
            data["value"] = None

        # Detect ChimeraX error text returned as 200 OK with plain-text body.
        # These patterns come directly from ChimeraX's command dispatcher.
        if data.get("error") is None and isinstance(data.get("value"), str):
            val = data["value"].strip()
            _ERROR_PREFIXES = (
                "Unknown command",
                "Error ",
                "Traceback",
                "syntax error",
                "No such",
                "Command failed",
                "Expected",      # "Expected a collection of…", "Expected an atoms specifier…"
            )
            if any(val.startswith(p) for p in _ERROR_PREFIXES):
                data["error"] = val
                data["value"] = None

        # ── Post-save verification ────────────────────────────────────────────
        # Must run after all body parsing, including the empty-body path above.
        # Successful saves always return an empty body, so without this the
        # empty-body early-return would bypass the check.
        if command.strip().lower().startswith("save ") and data.get("error") is None:
            self._check_save_result(command, data)

        return data

    # ── Image-save sanity check ────────────────────────────────────────────────

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    # Minimum plausible bytes for a real rendered structure image.
    # An empty ChimeraX scene at 800×800 compresses to ~3 KB; any real
    # structure at the same size is ≥ 50 KB.
    _MIN_IMAGE_BYTES = 10_000

    def _check_save_result(self, command: str, result: dict) -> None:
        """
        After a 'save' command, verify the output image file exists and is not
        suspiciously small (which indicates a blank / empty-scene capture).
        Modifies *result* in-place to add an "error" or "warning" key.
        """
        # Extract the path — handles both quoted and unquoted forms:
        #   save "C:/path/file.png" width ...
        #   save file.png
        m = re.search(
            r'save\s+"([^"]+)"|save\s+\'([^\']+)\'|save\s+(\S+)',
            command,
            re.IGNORECASE,
        )
        if not m:
            return
        raw_path = next(g for g in m.groups() if g is not None)

        # Only check image-format saves (not session .cxs or .pdb exports)
        suffix = Path(raw_path).suffix.lower()
        if suffix not in self._IMAGE_SUFFIXES:
            return

        # Normalise separators for the local OS
        check_path = Path(raw_path.replace("/", os.sep))

        if not check_path.is_file():
            result["error"] = (
                f"Save returned success but file not found on disk: {check_path}\n"
                "Possible causes: invalid path, permission error, or no models loaded."
            )
            return

        size = check_path.stat().st_size
        if size < self._MIN_IMAGE_BYTES:
            result["warning"] = (
                f"Saved image is only {size:,} bytes — this is characteristic of a "
                "blank scene (no models rendered).\n"
                "Make sure a structure is open and visible before saving."
            )

    def run_commands(self, commands: List[str]) -> List[dict]:
        """
        Execute a list of ChimeraX commands sequentially.

        Each entry in the returned list is::

            {
                "command": "the command string",
                "result":  {"value": "...", "error": None | "..."},
            }

        Execution stops at the first error.
        """
        results = []
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd or cmd.startswith("#"):
                continue  # skip empty lines and comments
            result = self.run_command(cmd)
            results.append({"command": cmd, "result": result})
            if result.get("error"):
                break  # halt on first failure; let caller handle retry
            # Deterministic, config-driven post-open hooks. All fire-and-forget:
            # they do NOT add to `results`, so the one-entry-per-command contract
            # and the on-first-error halt above are both preserved. Order matters:
            # default presentation runs AFTER load and BEFORE any later analysis
            # colouring (which arrives in a separate request and overrides the
            # by-chain baseline); then the once-per-session lean layout; then the
            # Sequence Viewer is opened + associated.
            model_id = self._opened_model_id(cmd, result)
            if model_id is not None:
                self._maybe_apply_presentation_on_open()
                self._maybe_apply_lean_layout()
                self._maybe_show_sequence_on_open(model_id)
                # Fire-and-forget: hand the REAL opened model id to whoever registered
                # (the unified GUI, for ground-truth tab focus). None elsewhere → no-op.
                if self.on_structure_opened is not None:
                    try:
                        self.on_structure_opened(model_id)
                    except Exception:
                        pass
        return results

    # ── Post-open hooks ────────────────────────────────────────────────────────

    @staticmethod
    def _opened_model_id(command: str, result: dict) -> Optional[str]:
        """
        Return the model id (digits, as a string) if *command* was a successful
        structure ``open`` whose response named a ``#N`` model — else None. Skips
        session restores and image/data files. Shared by all post-open hooks.
        """
        low = command.strip().lower()
        if not low.startswith("open "):
            return None
        if "session" in low or low.endswith((".cxs", ".png", ".jpg", ".jpeg")):
            return None
        value = result.get("value")
        if not isinstance(value, str):
            return None
        m = re.search(r"#(\d+)", value)
        return m.group(1) if m else None

    def _model_chains(self, model_id: str) -> List[str]:
        """The chain ids of model #*model_id* (e.g. ``['A', 'B']``) via
        ``info chains``. Returns [] on any failure so callers fall back to the
        whole-model (grouped) viewer."""
        try:
            r = self.run_command(f"info chains #{model_id}")
            value = r.get("value") if isinstance(r, dict) else None
            if not isinstance(value, str):
                return []
            chains: List[str] = []
            for c in re.findall(r"chain_id\s+(\S+)", value):
                if c not in chains:
                    chains.append(c)
            return chains
        except Exception:
            return []

    def _model_chain_resnums(self, model_id: str, chain: str) -> List[int]:
        """The MACROMOLECULE residue numbers of chain *chain*, in sequence order,
        via ``info residues`` scoped to ``~solvent & ~ligand & ~ions`` (so waters /
        the MK1-style ligand sharing the chain id are excluded). Sorted ascending
        (the chain's residue order); [] on any failure. Insertion codes collapse to
        their base number (a rare edge; the common non-1-start / gap cases are kept)."""
        try:
            r = self.run_command(
                f"info residues #{model_id}/{chain} & ~solvent & ~ligand & ~ions")
            value = r.get("value") if isinstance(r, dict) else None
            if not isinstance(value, str):
                return []
            nums = {int(m) for m in re.findall(rf"/{re.escape(chain)}:(-?\d+)", value)}
            return sorted(nums)
        except Exception:
            return []

    def _maybe_show_sequence_on_open(self, model_id: str) -> None:
        """Open + associate the Sequence Viewer for the new model (config-gated by
        CHIMERAX_SHOW_SEQUENCE_ON_OPEN).

        Chain-count routing (all thresholds config-exposed):
          ≤ CONSOLIDATE_THRESHOLD chains  → per-chain viewers (existing behaviour)
          > CONSOLIDATE_THRESHOLD and     → consolidated: one alignment per unique
            ≤ PER_CHAIN_MAX chains           sequence group, one row per structure,
                                             rulers reused unchanged
          > PER_CHAIN_MAX chains          → single whole-model viewer (fallback)

        Best-effort; never disrupts the batch.
        """
        try:
            import config as _cfg
            if not getattr(_cfg, "CHIMERAX_SHOW_SEQUENCE_ON_OPEN", True):
                return
            from sequence_viewer import (
                ensure_sequence_viewer_commands,
                dock_sequences_bottom_command,
                numbering_header_command,
                consolidated_viewers_command,
                _run_error_first,
            )
            cap       = int(getattr(_cfg, "CHIMERAX_SEQUENCE_PER_CHAIN_MAX",          8))
            threshold = int(getattr(_cfg, "CHIMERAX_SEQUENCE_CONSOLIDATE_THRESHOLD",  3))
            interval  = int(getattr(_cfg, "CHIMERAX_SEQUENCE_NUMBER_INTERVAL",       10))
            numbering = bool(getattr(_cfg, "CHIMERAX_SEQUENCE_NUMBERING",           True))

            found: List[str] = []
            if getattr(_cfg, "CHIMERAX_SEQUENCE_PER_CHAIN", True):
                found = self._model_chains(model_id)

            if found and len(found) > cap:
                # Viral-capsid fallback: single whole-model viewer
                try:
                    self.run_command(ensure_sequence_viewer_commands(model_id, None)[0])
                except Exception:
                    pass

            elif found and len(found) > threshold:
                # CONSOLIDATED: reconsolidate all open multi-chain models.
                # Inner try/except so a failing consolidation runscript never
                # prevents the dock-bottom hook from running.
                try:
                    self.run_command(consolidated_viewers_command(
                        threshold=threshold, cap=cap,
                        interval=interval,   numbering=numbering,
                    ))
                except Exception:
                    pass

            else:
                # PER-CHAIN (≤ threshold) or per-chain disabled (found=[])
                chains = found if found else None
                for c in ensure_sequence_viewer_commands(model_id, chains):
                    try:
                        self.run_command(c)
                    except Exception:
                        pass
                # Residue-number ruler for per-chain viewers. Error-first so a
                # failure on one chain never disrupts the remaining chains or dock.
                if chains and numbering:
                    num_cmds: List[str] = []
                    for ch in chains:
                        cmd = numbering_header_command(
                            model_id, ch,
                            self._model_chain_resnums(model_id, ch),
                            interval)
                        if cmd:
                            num_cmds.append(cmd)
                    _run_error_first(self.run_command, num_cmds)

            if getattr(_cfg, "CHIMERAX_SEQUENCE_DOCK_BOTTOM", True):
                try:
                    self.run_command(dock_sequences_bottom_command())
                except Exception:
                    pass
        except Exception:
            pass  # sequence viewer is non-essential; never disrupt the batch

    def reconsolidate(self) -> None:
        """Re-run consolidated sequence-viewer grouping for all open multi-chain
        models. Call after a single-chain sequence edit to trigger dynamic regroup:
        the edited chain diverges from its former group → ChimeraX auto-destroys the
        old alignment and creates a fresh one that reflects the new grouping.
        Best-effort; never raises."""
        try:
            import config as _cfg
            from sequence_viewer import consolidated_viewers_command, dock_sequences_bottom_command
            cap       = int(getattr(_cfg, "CHIMERAX_SEQUENCE_PER_CHAIN_MAX",          8))
            threshold = int(getattr(_cfg, "CHIMERAX_SEQUENCE_CONSOLIDATE_THRESHOLD",  3))
            interval  = int(getattr(_cfg, "CHIMERAX_SEQUENCE_NUMBER_INTERVAL",       10))
            numbering = bool(getattr(_cfg, "CHIMERAX_SEQUENCE_NUMBERING",           True))
            self.run_command(consolidated_viewers_command(
                threshold=threshold, cap=cap,
                interval=interval,   numbering=numbering,
            ))
            if getattr(_cfg, "CHIMERAX_SEQUENCE_DOCK_BOTTOM", True):
                self.run_command(dock_sequences_bottom_command())
        except Exception:
            pass

    def _maybe_apply_presentation_on_open(self) -> None:
        """Apply the deterministic default presentation (config-gated) right after
        a structure loads, before any analysis colouring. Error-first."""
        try:
            from sequence_viewer import apply_default_presentation
            apply_default_presentation(self.run_command)
        except Exception:
            pass

    def _maybe_apply_lean_layout(self) -> None:
        """Apply the lean window layout ONCE per ChimeraX session (config-gated).
        The guard is set before running so a partial failure never causes a retry."""
        if self._lean_layout_applied:
            return
        self._lean_layout_applied = True
        try:
            from sequence_viewer import apply_lean_layout
            apply_lean_layout(self.run_command)
        except Exception:
            pass

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def ping(self) -> dict:
        """
        Send a lightweight command and return timing + status info.
        Uses 'version' because ChimeraX returns its version string over REST;
        commands like 'echo' output to the GUI log and return nothing via REST.
        """
        start = time.perf_counter()
        result = self.run_command("version")
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "ok": result.get("error") is None,
            "latency_ms": round(elapsed_ms, 1),
            "result": result,
        }

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return (
            f"<ChimeraXBridge exe={self.chimerax_path!r} "
            f"port={self.port} status={status}>"
        )
