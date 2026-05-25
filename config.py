"""
config.py
---------
Central configuration for StructureBot.
All path and tuning constants live here.
Override any value by setting the matching environment variable or by editing
this file — never hard-code paths in other modules.
"""

import os
from pathlib import Path

# ── ChimeraX ──────────────────────────────────────────────────────────────────

CHIMERAX_PATH: str = os.environ.get(
    "CHIMERAX_PATH",
    r"C:\Users\andre\documents\ChimeraX 1.11.1\bin\ChimeraX.exe",
)

REST_HOST: str = os.environ.get("CHIMERAX_HOST", "127.0.0.1")
REST_PORT: int = int(os.environ.get("CHIMERAX_PORT", "60001"))
REST_TIMEOUT: int = int(os.environ.get("CHIMERAX_TIMEOUT", "10"))

# ── Anthropic ─────────────────────────────────────────────────────────────────

# claude-sonnet-4-6 is the current recommended model.
# Upgrade to claude-opus-4-7 for the hardest multi-step reasoning tasks.
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Maximum number of user/assistant exchange *pairs* kept in rolling history.
# Each turn consumes input tokens; prompt caching absorbs the static block cost.
MAX_CONVERSATION_HISTORY: int = int(os.environ.get("MAX_CONVERSATION_HISTORY", "6"))

# ── UI behaviour ──────────────────────────────────────────────────────────────

# Seconds before auto-executing high/medium-confidence commands.
# Set to 0 to always require explicit confirmation.
AUTO_PROCEED_DELAY: int = int(os.environ.get("AUTO_PROCEED_DELAY", "2"))

# ── Directories ───────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent

LOG_DIR: Path     = Path(os.environ.get("STRUCTUREBOT_LOG_DIR",     str(_BASE / "logs")))
SESSION_DIR: Path = Path(os.environ.get("STRUCTUREBOT_SESSION_DIR", str(_BASE / "sessions")))

# Create at import time so nothing else has to mkdir guard
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ── Desktop path helper ───────────────────────────────────────────────────────

def desktop_path() -> str:
    """Return the current user's Desktop path, always with forward slashes."""
    username = os.environ.get("USERNAME", os.environ.get("USER", "user"))
    p = Path(f"C:/Users/{username}/Desktop")
    return p.as_posix()  # forward slashes — ChimeraX save prefers them

# ── Rosetta / Robetta ─────────────────────────────────────────────────────────

# Which Rosetta backend to prefer: "auto" | "pyrosetta" | "robetta"
# "auto" tries PyRosetta first, falls back to Robetta if unavailable.
ROSETTA_BACKEND: str = os.environ.get("ROSETTA_BACKEND", "auto").strip()

# Robetta web API (https://robetta.bakerlab.org — free academic registration)
# API key obtained from your profile page after registration.
ROBETTA_EMAIL:   str = os.environ.get("ROBETTA_EMAIL",   "").strip()
ROBETTA_API_KEY: str = os.environ.get("ROBETTA_API_KEY", "").strip()

# Set PYROSETTA_AVAILABLE=true in .env.local when PyRosetta is installed.
# Python 3.14 wheels are not yet released; this is False by default.
# See rosetta_bridge.py for installation instructions.
PYROSETTA_AVAILABLE: bool = (
    os.environ.get("PYROSETTA_AVAILABLE", "").strip().lower()
    in ("1", "true", "yes")
)

# ── .env.local loader ─────────────────────────────────────────────────────────
# Called by main.py at startup BEFORE any other imports that read env vars.

def load_env_file(path: str | Path | None = None) -> None:
    """Load key=value pairs from .env.local (or a given path) into os.environ."""
    env_file = Path(path) if path else Path(__file__).parent / ".env.local"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())
