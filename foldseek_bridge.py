"""
foldseek_bridge.py
------------------
LOCAL-ONLY structural-neighbour search (Stage 2 template auto-discovery).

`foldseek easy-search <query> <localDB> <out> <tmp>` against a PRE-DOWNLOADED local PDB database —
NO network at query time (the one-time `foldseek databases PDB` download is the only-ever remote
step; verified offline under `unshare -rn`). Same LOCAL-ONLY invariant as Boltz `msa:empty`.

This is the SHIPPED home of the search the monomer-calibration eval prototyped
(`scripts/eval_template_guided_calibration.py::foldseek_neighbors`, which now delegates here — single
source of truth). The binary + DB + tuning live in `config.FOLDSEEK_*`.

Mirrors the other WSL bridges (Boltz / US-align): an `is_available()` capability flag (binary AND DB
present → the feature is shown disabled-with-reason, never silently empty) and a `search_neighbors()`
that returns ranked `(pdb_id, chain, structTM-to-query)`. `db_label()` gives the DB-scope honesty
string (a miss is a FALSE NEGATIVE — a good template may simply not be in this finite snapshot —
never a false positive).
"""
from __future__ import annotations

import os
import re
import shlex
import itertools
from typing import List, Tuple, Optional

import config as _cfg

# Unique out/tmp suffixes per search within a process (avoid collisions on rapid re-search).
_COUNTER = itertools.count()


class FoldseekBridge:
    """LOCAL-ONLY foldseek structural-neighbour search against the local PDB DB."""

    def __init__(self) -> None:
        self._exe = str(getattr(_cfg, "FOLDSEEK_EXE", "/home/andre/foldseek/bin/foldseek"))
        self._db = str(getattr(_cfg, "FOLDSEEK_DB", "/home/andre/foldseek_db/pdb"))
        self._distro = str(getattr(_cfg, "FOLDSEEK_WSL_DISTRO", "Ubuntu-24.04"))
        self._timeout = int(getattr(_cfg, "FOLDSEEK_TIMEOUT", 600))
        from wsl_bridge import WSLBridge
        self._wsl = WSLBridge(distribution=self._distro)
        self._db_label_cache: Optional[str] = None

    # ── Availability (capability flag — fail-loud, never silently empty) ──────────
    def is_available(self) -> bool:
        """True iff WSL is up AND the foldseek binary is executable AND the DB exists (the
        `<db>.dbtype` index file is foldseek's own existence marker). A False here means the
        FEATURE is unavailable (disable it with a reason) — distinct from 'searched, 0 hits'."""
        try:
            if not self._wsl.is_available():
                return False
            chk = (f"test -x {shlex.quote(self._exe)} && "
                   f"test -f {shlex.quote(self._db + '.dbtype')} && echo OK")
            res = self._wsl.run_command(chk, timeout=20)
            return bool(res.get("ok")) and "OK" in (res.get("stdout") or "")
        except Exception:
            return False

    def status(self) -> str:
        if not self._wsl.is_available():
            return "WSL2 unavailable"
        return "available" if self.is_available() else "foldseek binary or local PDB DB not found"

    def db_label(self) -> str:
        """DB-scope honesty string for the UI, e.g. 'PDB snapshot 2025-01 (local foldseek DB)'.
        Parsed from `<db>.version` (PDB_DATE). Cached. Falls back to a generic label if unreadable.
        A miss against this finite snapshot is a FALSE NEGATIVE — never imply exhaustiveness."""
        if self._db_label_cache is not None:
            return self._db_label_cache
        label = "local PDB DB"
        try:
            res = self._wsl.run_command(f"cat {shlex.quote(self._db + '.version')}", timeout=15)
            if res.get("ok"):
                m = re.search(r"(\d{6})\s+PDB_DATE", res.get("stdout", "") or "")
                if m:
                    yy, mm = m.group(1)[:2], m.group(1)[2:4]
                    label = f"PDB snapshot 20{yy}-{mm} (local foldseek DB)"
        except Exception:
            pass
        self._db_label_cache = label
        return label

    # ── Search ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_hits(stdout: str, min_tm: float, max_results: int) -> List[Tuple[str, str, float]]:
        """Parse foldseek m8 (target<TAB>alntmscore<TAB>…) → [(pdb_id, chain, TM)] sorted desc,
        TM≥min_tm, deduped on (pdb_id, chain). Pure / unit-testable."""
        hits: List[Tuple[str, str, float]] = []
        seen = set()
        for ln in (stdout or "").splitlines():
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            tgt = parts[0]                                  # e.g. "1ABC_A" / "pdb_1abc.cif_A"
            try:
                tm = float(parts[1])                        # alntmscore (query-normalised structTM)
            except ValueError:
                continue
            mobj = re.search(r"([0-9][A-Za-z0-9]{3})[_\.-]?([A-Za-z0-9])?", tgt)
            if not mobj:
                continue
            pid = mobj.group(1).upper()
            ch = (mobj.group(2) or "A")
            key = (pid, ch)
            if key in seen or tm < min_tm:
                continue
            seen.add(key)
            hits.append((pid, ch, round(tm, 3)))
        hits.sort(key=lambda h: -h[2])
        return hits[:max_results]

    def _search_command(self, query_wsl: str, out_wsl: str, tmp_wsl: str) -> str:
        """The exact LOCAL-ONLY easy-search command (carried over from the eval spine)."""
        fmt = "target,alntmscore,qtmscore,ttmscore,evalue"
        return (f"{shlex.quote(self._exe)} easy-search {shlex.quote(query_wsl)} "
                f"{shlex.quote(self._db)} {shlex.quote(out_wsl)} {shlex.quote(tmp_wsl)} "
                f"--alignment-type 1 --format-output {shlex.quote(fmt)} --max-seqs 2000 -e 10 "
                f"&& cat {shlex.quote(out_wsl)}")

    def search_neighbors(self, query_path: str, max_results: int = 30,
                         min_tm: float = 0.3) -> List[Tuple[str, str, float]]:
        """foldseek easy-search of *query_path* (a structure file) against the LOCAL PDB DB.
        Returns [(pdb_id, chain, structTM-to-query)] ranked desc, TM≥*min_tm*. LOCAL-ONLY — no
        remote API at query time. Empty list = NO hits ≥ min_tm (a real answer; the caller must
        already have checked `is_available()` to distinguish unavailable from no-hits)."""
        if not (query_path and os.path.isfile(query_path)):
            return []
        q = self._wsl.translate_path(os.path.abspath(query_path))
        n = next(_COUNTER)
        out_wsl = f"/tmp/fs_out_{os.getpid()}_{n}.m8"
        tmp_wsl = f"/tmp/fs_tmp_{os.getpid()}_{n}"
        res = self._wsl.run_command(self._search_command(q, out_wsl, tmp_wsl), timeout=self._timeout)
        if not res.get("ok"):
            return []
        return self._parse_hits(res.get("stdout", "") or "", min_tm, max_results)


def foldseek_available() -> bool:
    """Module-level capability signal for the Workbench (mirrors `boltz_available`)."""
    try:
        return FoldseekBridge().is_available()
    except Exception:
        return False
