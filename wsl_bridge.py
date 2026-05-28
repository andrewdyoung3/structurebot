"""
wsl_bridge.py
-------------
Allows Windows Python to run commands inside WSL2 and transfer files between
Windows and the WSL2 filesystem.

WSL2 INSTALLATION STATUS (checked at import time)
--------------------------------------------------
  Current status: Not installed.

  To install (PowerShell as Administrator):
    wsl --install -d Ubuntu-24.04
  Then reboot.  After rebooting, set up Python 3.12 inside WSL2:
    wsl
    sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip -y
    pip install pyrosetta-installer
    python3 -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"

Path translation
----------------
  Windows  C:\\Users\\andre\\docs\\file.pdb
  WSL2     /mnt/c/Users/andre/docs/file.pdb

When WSL2 is not installed, is_available() returns False and all other
methods return error dicts rather than raising exceptions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional

PYROSETTA_PYTHON = "/home/andre/pyrosetta_env/bin/python"

# ── WSL availability cache ────────────────────────────────────────────────────

_WSL_AVAILABLE_CACHE: Optional[bool] = None   # populated on first check
_WSL_DISTRO:          Optional[str]  = None   # name of the running distribution


def _check_wsl_availability() -> bool:
    """
    Probe WSL2 by running ``wsl.exe --status``.

    Returns True only if wsl.exe is present, exits 0, and the output looks
    like at least one distribution is registered.
    Caches the result so the probe only runs once per process lifetime.
    """
    global _WSL_AVAILABLE_CACHE, _WSL_DISTRO

    if _WSL_AVAILABLE_CACHE is not None:
        return _WSL_AVAILABLE_CACHE

    try:
        result = subprocess.run(
            ["wsl.exe", "--list", "--verbose"],
            capture_output=True,
            stdin=subprocess.DEVNULL,          # do NOT inherit the parent console's
            creationflags=subprocess.CREATE_NO_WINDOW,  # stdin handle; wsl.exe is a
            timeout=10,                        # Windows console app that calls
            text=True,                         # SetConsoleMode() on inherited handles
            encoding="utf-16-le",              # and may not restore them, which
            errors="replace",                  # silently disables ReadConsole() input
        )                                      # for the rest of the process lifetime.
        if result.returncode != 0:
            _WSL_AVAILABLE_CACHE = False
            return False

        output = result.stdout or result.stderr or ""
        # Look for a line with a distribution name (not header lines)
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        distro_lines = [
            ln for ln in lines
            if not ln.startswith("NAME") and not ln.startswith("-") and len(ln) > 2
        ]
        if distro_lines:
            # Extract first distribution name (first non-whitespace token)
            _WSL_DISTRO = distro_lines[0].split()[0].lstrip("*").strip()
            _WSL_AVAILABLE_CACHE = True
        else:
            _WSL_AVAILABLE_CACHE = False

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _WSL_AVAILABLE_CACHE = False

    return _WSL_AVAILABLE_CACHE


# ── Public bridge class ───────────────────────────────────────────────────────

class WSLBridge:
    """
    Runs commands inside WSL2 from Windows Python.

    Usage::

        wsl = WSLBridge()
        if wsl.is_available():
            r = wsl.run_command("python3 --version")
            print(r["stdout"])   # Python 3.12.x

    All methods return dicts rather than raising, so callers can test
    ``result["ok"]`` without a try/except.
    """

    def __init__(self, distribution: Optional[str] = None):
        """
        Parameters
        ----------
        distribution : WSL2 distribution name, e.g. "Ubuntu-24.04".
                       If None, the first registered distribution is used.
        """
        self._distribution = distribution or os.environ.get(
            "WSL_DISTRIBUTION", "Ubuntu-24.04"
        )

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Return True if WSL2 is installed and at least one distribution is
        registered.  Result is cached after the first call.
        """
        return _check_wsl_availability()

    # ── Command execution ─────────────────────────────────────────────────────

    def run_command(
        self,
        cmd:     str,
        timeout: int = 300,
        cwd:     Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a shell command inside WSL2.

        Parameters
        ----------
        cmd     : bash command string, e.g. "python3 --version"
        timeout : seconds before the process is killed (default 300)
        cwd     : working directory inside WSL2 (optional)

        Returns
        -------
        {
          "ok":         bool,
          "returncode": int,
          "stdout":     str,
          "stderr":     str,
          "error":      str | None,
        }
        """
        if not self.is_available():
            return {
                "ok":         False,
                "returncode": -1,
                "stdout":     "",
                "stderr":     "",
                "error":      (
                    "WSL2 is not installed. "
                    "Run: wsl --install -d Ubuntu-24.04  (as Administrator)"
                ),
            }

        wsl_args = ["wsl.exe", "--distribution", self._distribution]
        if cwd:
            wsl_args += ["--cd", cwd]
        wsl_args += ["--exec", "bash", "-c", cmd]

        try:
            proc = subprocess.run(
                wsl_args,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            return {
                "ok":         proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout":     proc.stdout or "",
                "stderr":     proc.stderr or "",
                "error":      None if proc.returncode == 0
                              else f"Command exited {proc.returncode}: {proc.stderr[:200]}",
            }
        except subprocess.TimeoutExpired:
            return {
                "ok":         False,
                "returncode": -1,
                "stdout":     "",
                "stderr":     "",
                "error":      f"WSL2 command timed out after {timeout}s: {cmd[:80]}",
            }
        except Exception as exc:
            return {
                "ok":         False,
                "returncode": -1,
                "stdout":     "",
                "stderr":     "",
                "error":      f"WSL2 run_command error: {exc}",
            }

    # ── Path translation ──────────────────────────────────────────────────────

    def translate_path(self, windows_path: str) -> str:
        """
        Convert a Windows path to its WSL2 /mnt/… equivalent.

        Examples
        --------
        ``C:\\Users\\andre\\file.pdb``  →  ``/mnt/c/Users/andre/file.pdb``
        ``C:/Users/andre/file.pdb``    →  ``/mnt/c/Users/andre/file.pdb``
        ``/mnt/c/already/here``        →  ``/mnt/c/already/here``  (passthrough)
        """
        # Already a WSL path?
        if windows_path.startswith("/mnt/") or windows_path.startswith("/"):
            return windows_path

        # Normalise separators
        p = windows_path.replace("\\", "/")

        # Handle drive letter: C:/foo → /mnt/c/foo
        m = re.match(r"^([A-Za-z]):/?(.*)", p)
        if m:
            drive = m.group(1).lower()
            rest  = m.group(2).lstrip("/")
            return f"/mnt/{drive}/{rest}"

        # UNC path or relative path — return unchanged
        return windows_path

    # ── File copy ─────────────────────────────────────────────────────────────

    def copy_to_wsl(self, windows_path: str, dest_dir: str = "/tmp") -> str:
        """
        Copy a Windows file into WSL2's filesystem.

        Parameters
        ----------
        windows_path : Windows-side file path
        dest_dir     : target directory inside WSL2 (default /tmp)

        Returns
        -------
        WSL2 path to the copied file, or empty string on failure.
        """
        if not self.is_available():
            return ""

        src = Path(windows_path)
        if not src.is_file():
            return ""

        # WSL2 can see the Windows filesystem as /mnt/c/...
        # We translate the path and make WSL2 copy it locally.
        wsl_src  = self.translate_path(str(src.resolve()))
        wsl_dest = f"{dest_dir.rstrip('/')}/{src.name}"

        result = self.run_command(f"cp -f '{wsl_src}' '{wsl_dest}'")
        if result["ok"]:
            return wsl_dest
        return ""

    def copy_from_wsl(self, wsl_path: str, windows_dest: str) -> bool:
        """
        Copy a file from WSL2 back to a Windows path.

        Parameters
        ----------
        wsl_path     : path inside WSL2 filesystem
        windows_dest : Windows destination path

        Returns
        -------
        True on success, False on failure.
        """
        if not self.is_available():
            return False

        dest = Path(windows_dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Translate Windows destination to WSL2 path
        wsl_dest = self.translate_path(str(dest.resolve()))
        result   = self.run_command(f"cp -f '{wsl_path}' '{wsl_dest}'")
        return result["ok"]

    # ── Python / PyRosetta helpers ────────────────────────────────────────────

    def check_pyrosetta(self) -> bool:
        """Return True if PyRosetta can be imported inside WSL2."""
        if not self.is_available():
            return False
        result = self.run_command(
            f"{PYROSETTA_PYTHON} -c 'import pyrosetta; print(chr(79)+chr(75))'",
            timeout=30,
        )
        return result["ok"] and "OK" in result["stdout"]

    def run_python_script(
        self,
        script:  str,
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """
        Write *script* to a temp file and run it inside WSL2 with python3.

        Returns the same dict as run_command().
        """
        if not self.is_available():
            return {
                "ok": False, "returncode": -1,
                "stdout": "", "stderr": "",
                "error": "WSL2 not available",
            }

        # Write the script to a Windows temp file that WSL2 can read
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            win_path = fh.name

        try:
            wsl_path = self.translate_path(win_path)
            return self.run_command(f"{PYROSETTA_PYTHON} '{wsl_path}'", timeout=timeout)
        finally:
            try:
                os.unlink(win_path)
            except OSError:
                pass

    # ── Status ────────────────────────────────────────────────────────────────

    def status_string(self) -> str:
        """One-line status for the StructureBot startup display."""
        if not self.is_available():
            return (
                "WSL2: not installed — run `wsl --install -d Ubuntu-24.04` "
                "(PowerShell as Administrator) to enable local Rosetta"
            )
        distro = _WSL_DISTRO or self._distribution
        has_py = self.check_pyrosetta()
        if has_py:
            return f"WSL2: {distro} — PyRosetta available ✓"
        return f"WSL2: {distro} — PyRosetta not installed (run pyrosetta_installer)"

    def __repr__(self) -> str:
        avail = self.is_available()
        return f"<WSLBridge distribution={self._distribution!r} available={avail}>"
