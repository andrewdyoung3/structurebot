"""
log_analyser.py
---------------
Parse StructureBot session logs and generate a usage statistics report.

Log format (JSONL, one JSON object per line)
--------------------------------------------
Basic (all sessions):
    {
        "timestamp":  "2026-05-27T14:23:01",
        "user_input": "suggest mutations to improve solubility...",
        "commands":   ["open 1HSG", "cartoon #1", ...],
        "success":    true,
        "error":      null
    }

Enhanced (sessions since S8, if _log_exchange passes tool_steps):
    {
        ...,
        "tool_steps": [
            {
                "tool":          "mutation_scan",
                "elapsed_ms":    92340,
                "success":       true,
                "n_candidates":  5,
                "top_candidate": "I64E",
                "top_ddg":       -3.53,
                "backend":       "dynamut2"
            },
            ...
        ]
    }

Public API
----------
    data   = parse_logs(log_dir)
    report = generate_stats_report(data)   # returns Rich renderable

    # Or from main.py:
    from log_analyser import display_stats
    display_stats(console, log_dir)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as _cfg


# ── PDB ID extractor ──────────────────────────────────────────────────────────

_PDB_ID_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]{3})\b")   # 4-char alphanumeric


def _extract_pdb_ids(commands: List[str], user_input: str) -> List[str]:
    """
    Extract PDB IDs from `open XXXX` commands and the user's input.
    Returns a list of uppercase 4-letter PDB IDs.
    """
    ids: List[str] = []
    for cmd in commands:
        s = cmd.strip()
        if s.lower().startswith("open "):
            parts = s.split()
            if len(parts) >= 2:
                candidate = parts[1].strip("'\"")
                if re.match(r"^[A-Za-z][A-Za-z0-9]{3}$", candidate):
                    ids.append(candidate.upper())
    # Also scan user input for bare 4-char PDB-like tokens
    for m in _PDB_ID_RE.finditer(user_input):
        tok = m.group(1).upper()
        # Exclude common words that happen to be 4 chars (heuristic)
        if not tok.lower() in {"this", "that", "with", "from", "have", "open",
                                "show", "load", "make", "view", "find", "list"}:
            ids.append(tok)
    return ids


# ── Tool keyword detector ─────────────────────────────────────────────────────

_SCAN_KEYWORDS    = ("suggest mutations", "mutation scan", "engineering candidates",
                     "improve solubility", "what mutations", "scan chain")
_DISULFIDE_KW     = ("disulfide", "cross-link", "stabilise the interface")
_ESMFOLD_KW       = ("foldability", "will it fold", "check foldability", "validate design")
_PROTEINMPNN_KW   = ("proteinmpnn", "sequence redesign", "design sequences")


def _detect_tool(user_input: str) -> Optional[str]:
    """Return the primary tool detected in a user request, or None."""
    lower = user_input.lower()
    if any(kw in lower for kw in _SCAN_KEYWORDS):
        return "mutation_scan"
    if any(kw in lower for kw in _DISULFIDE_KW):
        return "disulfide"
    if any(kw in lower for kw in _ESMFOLD_KW):
        return "esmfold"
    if any(kw in lower for kw in _PROTEINMPNN_KW):
        return "proteinmpnn"
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Log parser
# ════════════════════════════════════════════════════════════════════════════════

def parse_logs(log_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Parse all session_*.jsonl files in *log_dir*.

    Returns a structured dict:
    {
        "n_sessions":       int,
        "n_requests":       int,
        "n_success":        int,
        "pdb_id_counts":    {pdb_id: count},
        "tool_counts":      {tool_name: count},
        "tool_timings":     {tool_name: [elapsed_s, ...]},
        "top_candidates":   {mutation_key: count},     # from enhanced logs
        "backends_used":    {backend: count},
        "log_files":        [str, ...],
    }
    """
    if log_dir is None:
        log_dir = _cfg.LOG_DIR

    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("session_*.jsonl"))

    n_sessions   = len(log_files)
    n_requests   = 0
    n_success    = 0
    pdb_counts:   Dict[str, int]        = defaultdict(int)
    tool_counts:  Dict[str, int]        = defaultdict(int)
    tool_timings: Dict[str, List[float]]= defaultdict(list)
    top_cands:    Dict[str, int]        = defaultdict(int)
    backends:     Dict[str, int]        = defaultdict(int)

    for path in log_files:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                n_requests += 1
                if entry.get("success"):
                    n_success += 1

                user_input = entry.get("user_input", "")
                commands   = entry.get("commands", [])

                # PDB IDs
                for pid in _extract_pdb_ids(commands, user_input):
                    pdb_counts[pid] += 1

                # Tool detection (heuristic from user input)
                tool = _detect_tool(user_input)
                if tool:
                    tool_counts[tool] += 1

                # Enhanced log fields (tool_steps)
                for step in entry.get("tool_steps", []):
                    t       = step.get("tool", "")
                    elapsed = step.get("elapsed_ms")
                    if t and elapsed is not None:
                        tool_timings[t].append(float(elapsed) / 1000.0)
                        tool_counts[t] = max(tool_counts.get(t, 0),
                                             tool_counts.get(t, 0))   # don't double-count
                    top = step.get("top_candidate")
                    if top:
                        top_cands[top] += 1
                    backend = step.get("backend")
                    if backend:
                        backends[backend] += 1

        except Exception:
            continue   # silently skip unreadable files

    return {
        "n_sessions":     n_sessions,
        "n_requests":     n_requests,
        "n_success":      n_success,
        "pdb_id_counts":  dict(pdb_counts),
        "tool_counts":    dict(tool_counts),
        "tool_timings":   {k: v for k, v in tool_timings.items()},
        "top_candidates": dict(top_cands),
        "backends_used":  dict(backends),
        "log_files":      [str(p) for p in log_files],
    }


# ════════════════════════════════════════════════════════════════════════════════
# Report generator
# ════════════════════════════════════════════════════════════════════════════════

def generate_stats_report(data: Dict[str, Any]) -> Any:
    """
    Generate a Rich renderable stats report from parsed log data.
    Returns a rich.panel.Panel object (or a plain string if rich unavailable).
    """
    lines: List[str] = []

    n_sessions = data.get("n_sessions", 0)
    n_requests = data.get("n_requests", 0)
    n_success  = data.get("n_success",  0)
    pct_ok     = f"{100*n_success/n_requests:.0f}%" if n_requests else "N/A"

    lines.append(f"Sessions:         {n_sessions}")
    lines.append(f"Total requests:   {n_requests}  ({pct_ok} success)")

    # Unique structures
    pdb_counts = data.get("pdb_id_counts", {})
    lines.append(f"Unique structures: {len(pdb_counts)}")

    # Tool usage
    tool_counts = data.get("tool_counts", {})
    if tool_counts:
        lines.append("")
        lines.append("Tool usage:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {tool:<18} {count} request(s)")

    # Tool timings (from enhanced logs)
    tool_timings = data.get("tool_timings", {})
    if tool_timings:
        lines.append("")
        lines.append("Tool performance (from enhanced logs):")
        for tool, times in sorted(tool_timings.items()):
            mean_s = sum(times) / len(times)
            if mean_s >= 60:
                t_str = f"{mean_s/60:.1f} min"
            else:
                t_str = f"{mean_s:.1f}s"
            lines.append(f"  {tool:<18} mean {t_str}  (n={len(times)})")

    # Top structures
    if pdb_counts:
        lines.append("")
        top_pdbs = sorted(pdb_counts.items(), key=lambda x: -x[1])[:5]
        top_str  = ", ".join(f"{pid} ({n})" for pid, n in top_pdbs)
        lines.append(f"Top structures: {top_str}")

    # Most common top candidate
    top_cands = data.get("top_candidates", {})
    if top_cands:
        best_cand, best_count = max(top_cands.items(), key=lambda x: x[1])
        lines.append(f"Most common top candidate: {best_cand} ({best_count}x)")

    # Backend breakdown
    backends = data.get("backends_used", {})
    if backends:
        lines.append("")
        lines.append("Stability backends used:")
        for b, n in sorted(backends.items(), key=lambda x: -x[1]):
            lines.append(f"  {b:<22} {n}x")

    if n_sessions == 0:
        lines.append("(No session logs found in " + str(_cfg.LOG_DIR) + ")")

    body = "\n".join(lines)

    try:
        from rich.panel import Panel
        return Panel(
            body,
            title="[bold cyan]StructureBot Usage Statistics[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    except ImportError:
        return body


# ── Entry point for standalone use ────────────────────────────────────────────

def display_stats(console: Any, log_dir: Optional[Path] = None) -> None:
    """Print stats report to *console* (a rich.console.Console object)."""
    data   = parse_logs(log_dir)
    report = generate_stats_report(data)
    console.print(report)
