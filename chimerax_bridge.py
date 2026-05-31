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
from typing import List, Optional

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
        self.base_url: str = f"http://localhost:{port}"
        self.run_url: str = f"{self.base_url}/run"
        self._process: Optional[subprocess.Popen] = None
        self._command_queue: queue.Queue = queue.Queue()

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
            # Auto-open the Sequence Viewer for newly opened structures so loaded
            # PDBs show their sequence by default (config-gated). Fire-and-forget:
            # it does NOT add to `results`, so the one-entry-per-command contract
            # and the on-first-error halt above are both preserved.
            self._maybe_show_sequence_on_open(cmd, result)
        return results

    # ── Sequence-on-open ──────────────────────────────────────────────────────

    def _maybe_show_sequence_on_open(self, command: str, result: dict) -> None:
        """
        After a successful structure ``open``, open the ChimeraX Sequence Viewer
        for the new model so its sequence is shown by default.

        Gated by config.CHIMERAX_SHOW_SEQUENCE_ON_OPEN (default True). Best-effort:
        only fires when the opened model id can be parsed from the open response
        (``#N``); never raises and never affects the run_commands result list.
        Skips session opens and image/data files.
        """
        try:
            import config as _cfg
            if not getattr(_cfg, "CHIMERAX_SHOW_SEQUENCE_ON_OPEN", True):
                return
        except Exception:
            return

        low = command.strip().lower()
        if not low.startswith("open "):
            return
        if "session" in low or low.endswith((".cxs", ".png", ".jpg", ".jpeg")):
            return

        # Parse the opened model id from the response (e.g. "...#1, ... model(s)").
        value = result.get("value")
        if not isinstance(value, str):
            return
        m = re.search(r"#(\d+)", value)
        if not m:
            return
        try:
            self.run_command(f"sequence chain #{m.group(1)}")
        except Exception:
            pass  # sequence viewer is non-essential; never disrupt the batch

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
