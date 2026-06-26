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
import threading
import time
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional

# Bounded-wait knobs (the Mode-A "running forever" fix). A long fold's WSL subprocess can
# finish its compute (GPU idle, no Boltz process) yet leave the WINDOWS-side wait blocked:
# subprocess.run's post-timeout drain waits on stdout EOF that the LINUX-side process tree
# (unreapable from Windows — those processes live in the WSL VM under init) never delivers.
# We bound BOTH the tree-kill and the final drain so a timed-out OR hung fold becomes an
# honest terminal error that RETURNS, never an infinite spinner. See §9 Mode-A completion.
_WSL_DRAIN_TIMEOUT = 30    # seconds: final stdout drain AFTER the WSL tree is killed
_WSL_KILL_TIMEOUT  = 30    # seconds: the WSL-side tree-kill helper's own wall

PYROSETTA_PYTHON = "/home/andre/pyrosetta_env/bin/python"
# ColabFold env (WSL2, Python 3.12, hermetic JAX cuda12). Isolated from
# pyrosetta_env. The colabfold_bridge (future task) will run colabfold_batch via
# this interpreter; defined here now so that bridge can import it. See §10/§11.
COLABFOLD_PYTHON = "/home/andre/colabfold_env/bin/python"
# RFdiffusion env (WSL2, Python 3.9-3.11, Linux CUDA torch + SE3-Transformer).
# RFdiffusion has NO working Windows / Python-3.12 path, so the bridge ALWAYS
# runs run_inference.py through this interpreter — never VENV312. The env itself
# is NOT created here; this constant is defined so rfdiffusion_bridge can import
# it and so is_available() can probe for it (mirror of COLABFOLD_PYTHON). The
# attended GPU-activation session builds ~/rfdiffusion_env. See §9/§11.
RFDIFFUSION_PYTHON = "/home/andre/rfdiffusion_env/bin/python"

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
        # Instance-overridable so tests can shrink the final-drain wall (the bounded-wait
        # proof runs in ~1s, not 30s); production uses the module default.
        self._drain_timeout = _WSL_DRAIN_TIMEOUT

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

        # TAG the WSL-side tree so a timeout can reap the LINUX processes by marker. The
        # export rides in the bash -c string and is inherited by every child (boltz + its
        # workers), so _kill_wsl_tree can find the WHOLE tree, not just the bash parent.
        marker = uuid.uuid4().hex
        tagged = f"export SBWSL_TAG={marker}; {cmd}"
        wsl_args = ["wsl.exe", "--distribution", self._distribution]
        if cwd:
            wsl_args += ["--cd", cwd]
        wsl_args += ["--exec", "bash", "-c", tagged]

        try:
            proc = self._popen_wsl(wsl_args)
        except Exception as exc:
            return {
                "ok":         False,
                "returncode": -1,
                "stdout":     "",
                "stderr":     "",
                "error":      f"WSL2 run_command error: {exc}",
            }

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return {
                "ok":         proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout":     stdout or "",
                "stderr":     stderr or "",
                "error":      None if proc.returncode == 0
                              else f"Command exited {proc.returncode}: {(stderr or '')[:200]}",
            }
        except subprocess.TimeoutExpired:
            # THE Mode-A hang fix: reap the WSL-SIDE tree first (proc.kill() alone leaves the
            # Linux processes alive, holding stdout open → the post-timeout drain blocks
            # forever), THEN bound the final drain so run_command can NEVER hang. A finished-
            # but-not-draining OR genuinely-stuck fold returns an honest timeout error here,
            # which fires the worker's failed/done signal → the spinner becomes a real verdict.
            self._kill_wsl_tree(marker)
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr, drained = self._bounded_drain(proc, self._drain_timeout)
            tail = "drain completed" if drained else "drain abandoned"
            return {
                "ok":         False,
                "returncode": -1,
                "stdout":     stdout or "",
                "stderr":     stderr or "",
                "error":      f"WSL2 command timed out after {timeout}s "
                              f"(WSL tree killed; {tail}): {cmd[:80]}",
            }
        except Exception as exc:
            try:
                proc.kill()
            except Exception:
                pass
            return {
                "ok":         False,
                "returncode": -1,
                "stdout":     "",
                "stderr":     "",
                "error":      f"WSL2 run_command error: {exc}",
            }

    # ── bounded-wait helpers (the Mode-A completion-signal fix) ──────────────────
    def _popen_wsl(self, wsl_args: list) -> "subprocess.Popen":
        """Spawn the wsl.exe subprocess. A seam so the bounded-wait path is unit-testable
        (a test injects a fake Popen whose drain never closes)."""
        return subprocess.Popen(
            wsl_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _kill_wsl_tree(self, marker: str) -> None:
        """Kill the WSL-SIDE process tree of a timed-out command. Killing only the Windows
        wsl.exe handle does NOT reap the Linux processes (they live in the WSL VM under
        init), and they hold the stdout pipe open → the post-timeout drain hangs forever.
        Every process in the tree inherits ``SBWSL_TAG=<marker>`` in its environment, so we
        walk /proc and SIGKILL every process carrying the marker — the WHOLE tree, not just
        the bash parent (``pkill -f`` would match only the marker-bearing command line and
        MISS the children). Best-effort + its own wall: a failed reap must never itself hang."""
        walk = (
            "for e in /proc/[0-9]*/environ; do "
            f"tr '\\0' '\\n' < \"$e\" 2>/dev/null | grep -qx 'SBWSL_TAG={marker}' "
            "&& kill -9 \"$(basename \"$(dirname \"$e\")\")\" 2>/dev/null; "
            "done"
        )
        try:
            subprocess.run(
                ["wsl.exe", "--distribution", self._distribution, "--exec", "bash", "-c", walk],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True,
                timeout=_WSL_KILL_TIMEOUT,
            )
        except Exception:
            pass

    def _bounded_drain(self, proc, drain_timeout: float):
        """Collect any remaining output AFTER the tree is killed, but NEVER block forever:
        run communicate() in a daemon thread and ABANDON it if the pipe still won't close
        (a leaked daemon is bounded and harmless; an infinite wait is the bug). Returns
        ``(stdout, stderr, drained)`` — ``drained`` False means the pipe was abandoned."""
        box = {"out": "", "err": "", "done": False}

        def _drain():
            try:
                o, e = proc.communicate()
                box["out"], box["err"], box["done"] = o or "", e or "", True
            except Exception:
                pass

        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        t.join(drain_timeout)
        return box["out"], box["err"], box["done"]

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
        script:     str,
        timeout:    int = 600,
        python_bin: str = PYROSETTA_PYTHON,
    ) -> Dict[str, Any]:
        """
        Write *script* to a temp file and run it inside WSL2 with a Python
        interpreter.

        Parameters
        ----------
        python_bin : WSL2 path to the Python interpreter. Defaults to
                     ``PYROSETTA_PYTHON`` (backward compatible); pass
                     ``COLABFOLD_PYTHON`` to run inside the isolated ColabFold
                     env instead.

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
            return self.run_command(f"{python_bin} '{wsl_path}'", timeout=timeout)
        finally:
            try:
                os.unlink(win_path)
            except OSError:
                pass

    def check_colabfold(self) -> bool:
        """Return True if the ColabFold env interpreter + colabfold_batch exist in WSL2."""
        if not self.is_available():
            return False
        # -x on the interpreter; colabfold_batch lives alongside it in the env bin.
        _cf_batch = COLABFOLD_PYTHON.rsplit("/", 1)[0] + "/colabfold_batch"
        result = self.run_command(
            f"test -x {COLABFOLD_PYTHON} && test -x {_cf_batch} && echo {chr(79)+chr(75)}",
            timeout=30,
        )
        return result["ok"] and "OK" in result["stdout"]

    def check_rfdiffusion(self, rfd_dir: str = "/home/andre/RFdiffusion") -> bool:
        """
        Return True if the RFdiffusion env interpreter AND a run_inference.py
        (repo root or scripts/) exist in WSL2.  Mirror of check_colabfold.

        *rfd_dir* is the WSL2 clone path; run_inference.py lives at the repo root
        in some RFdiffusion versions and under scripts/ in others, so accept
        either.  The interpreter (~/rfdiffusion_env) is the hard requirement.
        """
        if not self.is_available():
            return False
        result = self.run_command(
            f"test -x {RFDIFFUSION_PYTHON} && "
            f"( test -f {rfd_dir}/run_inference.py || "
            f"test -f {rfd_dir}/scripts/run_inference.py ) && echo {chr(79)+chr(75)}",
            timeout=30,
        )
        return result["ok"] and "OK" in result["stdout"]

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
