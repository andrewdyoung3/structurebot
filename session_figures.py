"""
session_figures.py
------------------
Phase 2 of the session EXPORTS layer: relevance-gated PROFILE FIGURES. The principle —
"not everything produces a figure": each result type declares its artifact, and a plot is
rendered ONLY where the data is a per-residue PROFILE; scalars (solubility Δ, ΣΔΔG, TM/RMSD)
get NO figure (they're already Summary rows + small tables). Consumes the SAME row tables
`session_export.build_tables` produces (no recompute) and renders headless (matplotlib Agg).

Rendered into `exports/figures/`:
  fold pLDDT vs resnum                    (one per fold: T / guided / each variant)
  deviation dRMSD + lDDT vs resnum (+floors)  (one per variant)
  template_assist Δflex vs resnum         (one per construct)

FAIL-LOUD-SKIP: a profile type with no rows renders nothing — no empty/placeholder images;
the figures dir is created only if at least one figure is written. Never raises.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

# Artifact registry: which result types are per-residue PROFILES (→ plot). Everything else is
# scalar/table-only (→ no figure).
PROFILE_TYPES = ("fold_plddt", "deviation", "template_assist_dflex")


def _safe(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(s))


def _num(x: Any):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _xy(rows: List[Dict[str, Any]], xk: str, yk: str):
    """(xs, ys) numeric pairs sorted by x, dropping non-numeric / missing."""
    pts = [(_num(r.get(xk)), _num(r.get(yk))) for r in rows]
    pts = sorted((x, y) for x, y in pts if x is not None and y is not None)
    return [p[0] for p in pts], [p[1] for p in pts]


def _group(rows: List[Dict[str, Any]], keys):
    out: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return out


def render_profile_figures(tables: Dict[str, List[Dict[str, Any]]], figures_dir) -> Dict[str, Any]:
    """Render the profile plots for the populated profile types into *figures_dir*. Returns
    {written:[filenames], skipped:[type titles with no profile data], error:str|None}. Never raises
    (a plotting failure is reported, never propagated — the data export already succeeded)."""
    figures_dir = Path(figures_dir)
    written: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")                              # headless, no display
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"written": [], "skipped": [], "error": f"matplotlib unavailable: {exc}"}

    def _new(name: str):
        figures_dir.mkdir(parents=True, exist_ok=True)
        return figures_dir / name

    try:
        # fold pLDDT vs resnum — one figure per (model, chain, row)
        for (m, c, row), rs in _group(tables.get("fold_plddt") or [], ("model", "design_chain", "row")).items():
            xs, ys = _xy(rs, "resnum", "plddt")
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(xs, ys, lw=1.3, color="#2E75B6")
            ax.set(xlabel="residue", ylabel="pLDDT", ylim=(0, 100),
                   title=f"pLDDT — {m}/{c} {row}")
            ax.grid(True, alpha=0.3)
            name = f"plddt_{_safe(m)}_{_safe(c)}_{_safe(row)}.png"
            fig.tight_layout(); fig.savefig(_new(name), dpi=110); plt.close(fig)
            written.append(name)

        # deviation dRMSD + lDDT vs resnum (with floors) — one figure per (model, chain, variant)
        for (m, c, v), rs in _group(tables.get("deviation") or [], ("model", "design_chain", "variant")).items():
            dx, dy = _xy(rs, "resnum", "dRMSD")
            lx, ly = _xy(rs, "resnum", "lDDT")
            if not dx and not lx:
                continue
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            if dx:
                a1.plot(dx, dy, lw=1.3, color="#C0392B", label="dRMSD")
                fx, fy = _xy(rs, "resnum", "dRMSD_floor")
                if fx:
                    a1.plot(fx, fy, lw=1.0, ls="--", color="#999999", label="floor")
                a1.legend(fontsize=7)
            a1.set(ylabel="dRMSD (Å)", title=f"deviation — {m}/{c} {v}"); a1.grid(True, alpha=0.3)
            if lx:
                a2.plot(lx, ly, lw=1.3, color="#27AE60", label="lDDT")
                fx, fy = _xy(rs, "resnum", "lDDT_floor")
                if fx:
                    a2.plot(fx, fy, lw=1.0, ls="--", color="#999999", label="floor")
                a2.legend(fontsize=7)
            a2.set(xlabel="residue", ylabel="Cα-lDDT", ylim=(0, 1)); a2.grid(True, alpha=0.3)
            name = f"deviation_{_safe(m)}_{_safe(c)}_{_safe(v)}.png"
            fig.tight_layout(); fig.savefig(_new(name), dpi=110); plt.close(fig)
            written.append(name)

        # template_assist Δflex vs resnum — one figure per (model, chain)
        for (m, c), rs in _group(tables.get("template_assist_dflex") or [], ("model", "design_chain")).items():
            xs, ys = _xy(rs, "resnum", "d_flex")
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(xs, ys, lw=1.3, color="#7F6000")
            ax.set(xlabel="residue", ylabel="Δflex (Å)", title=f"template-assist Δflex — {m}/{c}")
            ax.grid(True, alpha=0.3)
            name = f"dflex_{_safe(m)}_{_safe(c)}.png"
            fig.tight_layout(); fig.savefig(_new(name), dpi=110); plt.close(fig)
            written.append(name)
    except Exception as exc:
        return {"written": written, "skipped": [], "error": f"{type(exc).__name__}: {exc}"}

    skipped = [t for t in PROFILE_TYPES if not tables.get(t)]
    return {"written": written, "skipped": skipped, "error": None}
