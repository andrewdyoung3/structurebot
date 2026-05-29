#!/usr/bin/env python3
"""
scripts/test_pyrosetta_ddg_controls.py
--------------------------------------
Magnitude sanity check for the PyRosetta / WSL2 local ddG path.

Runs a panel of control mutations in 1HSG (chain A) chosen to span the ddG
range — buried-hydrophobic→charged (strongly destabilising), buried→smaller
(moderate), and surface/conservative (near-neutral) — one at a time through
the SAME RosettaBridge._run_rosetta_local entry point used by
test_pyrosetta_single.py.

Purpose: confirm the *magnitude* (not just the sign) is physically sensible.
If the buried/charged mutations come back near zero like the surface ones,
that indicates a real scoring problem (relax over-minimising / mutation not
applied). If they come back appropriately large (buried-charge >> surface),
the PyRosetta path — and the near-zero surface values we already measured —
are validated.

Usage:
    python scripts/test_pyrosetta_ddg_controls.py

Requires ROSETTA_BACKEND=local (PyRosetta in WSL2). Forced on regardless of
.env.local. Read-only: changes no project modules.

Residue identities are verified against the live 1HSG chain A sequence at
runtime; a mismatch corrects the displayed wild-type letter and prints a note
(all nine were confirmed correct against canonical HIV-1 protease at authoring
time: L90,I84,L24,V82,I50,L76,I72,R8,E35).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Project path setup ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

config.load_env_file()
os.environ["ROSETTA_BACKEND"] = "local"   # force the PyRosetta/WSL2 path

from rosetta_bridge import RosettaBridge, _mutation_key  # noqa: E402

_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# ── Control panel: (position, from_aa, to_aa, group, note) ───────────────────
# Groups: "buried_charged" (expect large +ddG), "moderate" (~+1..+4),
#         "surface" (near 0).  Wild-type letters verified vs 1HSG chain A.
CONTROLS = [
    # Strongly destabilising — buried hydrophobic core → buried charge
    (90, "L", "K", "buried_charged", "Leu90 buried core; buries a +charge"),
    (84, "I", "D", "buried_charged", "Ile84 buried; buries a -charge"),
    (24, "L", "K", "buried_charged", "Leu24 core packing -> Lys"),
    # Moderately destabilising — cavity-creating / small polar
    (82, "V", "A", "moderate",       "Val82: classic HIV-PR cavity mutation, exp ~+1.5..+2"),
    (50, "I", "V", "moderate",       "Ile50 -> Val, small cavity"),
    (76, "L", "A", "moderate",       "Leu76 -> Ala, cavity creation"),
    # Near-neutral — surface / conservative
    (72, "I", "R", "surface",        "Ile72 surface solubility candidate (measured ~-0.003)"),
    (8,  "R", "K", "surface",        "Arg8 surface conservative charge swap"),
    (35, "E", "D", "surface",        "Glu35 surface conservative"),
]

_BURIED = {"L90K", "I84D", "L24K"}
_SURFACE = {"I72R", "R8K", "E35D"}


def _p(msg: str = "") -> None:
    """ASCII-safe print (Windows consoles choke on the bridge's emoji)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _rule(title: str = "") -> None:
    _p("=" * 88)
    if title:
        _p(title)
        _p("=" * 88)


def _ensure_1hsg() -> Path:
    cache_dir = _ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = cache_dir / "1HSG.pdb"
    if pdb.is_file() and pdb.stat().st_size > 1000:
        return pdb
    _p("cache/1HSG.pdb missing — downloading from RCSB...")
    import requests
    resp = requests.get("https://files.rcsb.org/download/1HSG.pdb", timeout=30)
    resp.raise_for_status()
    pdb.write_bytes(resp.content)
    return pdb


def _parse_chain_a(pdb: Path) -> dict[int, str]:
    """{resseq: one_letter} for chain A CA atoms."""
    seq: dict[int, str] = {}
    for ln in pdb.read_text(errors="replace").splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21:22] == "A":
            try:
                seq[int(ln[22:26])] = _3TO1.get(ln[17:20].strip(), "X")
            except ValueError:
                pass
    return seq


def _bucket(ddg: float | None) -> str:
    """STRONG >+3 | MODERATE +1..+3 | NEUTRAL -1..+1 | STABILISING <-1."""
    if ddg is None:
        return "FAILED"
    if ddg > 3.0:
        return "STRONG"
    if ddg > 1.0:
        return "MODERATE"
    if ddg >= -1.0:
        return "NEUTRAL"
    return "STABILISING"


def main() -> int:
    _rule("PyRosetta / WSL2 ddG magnitude validation — 1HSG control panel")

    pdb_path = _ensure_1hsg()
    _p(f"PDB: {pdb_path} ({pdb_path.stat().st_size} bytes)")

    seq = _parse_chain_a(pdb_path)
    _p(f"Chain A: residues {min(seq)}..{max(seq)} ({len(seq)} residues)")

    bridge = RosettaBridge()
    _p(f"Backend: {bridge._backend!r} — {bridge.backend_status()}")
    if bridge._backend != "local":
        _p("WARNING: backend is not 'local'; results will not exercise PyRosetta.")

    # Verify / correct wild-type letters against the real structure.
    _rule("RESIDUE VERIFICATION (vs live 1HSG chain A)")
    panel = []
    for pos, claimed, to_aa, group, note in CONTROLS:
        actual = seq.get(pos, "?")
        from_aa = claimed
        if actual != claimed and actual in _3TO1.values():
            _p(f"  NOTE pos {pos}: claimed WT {claimed} but structure has {actual} "
               f"-> using {actual} (key relabelled).")
            from_aa = actual
        key = f"{from_aa}{pos}{to_aa}"
        flag = "OK" if actual == claimed else f"corrected->{actual}"
        _p(f"  {key:<6} pos {pos:<3} WT={actual}  group={group:<14} {flag}  ({note})")
        panel.append({"key": key, "chain": "A", "position": pos,
                      "from_aa": from_aa, "to_aa": to_aa, "group": group})

    # ── Score each mutation, one at a time ────────────────────────────────────
    def _quiet(msg: str) -> None:
        # Echo only the salient worker lines to keep output readable.
        if any(t in msg for t in ("ddG =", "FATAL", "failed", "Loading cached",
                                   "Relax complete", "falling back")):
            _p(f"      {msg.strip()}")

    rows = []
    _rule("SCORING (one mutation per _run_rosetta_local call)")
    for m in panel:
        mut = {"chain": m["chain"], "position": m["position"],
               "from_aa": m["from_aa"], "to_aa": m["to_aa"]}
        _p(f"  -> {m['key']} ({m['group']}) ...")
        t0 = time.perf_counter()
        result = bridge._run_rosetta_local(
            pdb_path          = str(pdb_path),
            mutations         = [mut],
            model_id          = "1",
            chain             = "A",
            progress_callback = _quiet,
        )
        dt = time.perf_counter() - t0
        data = result.data or {}
        ddg = (data.get("ddg_scores", {}) or {}).get(m["key"])
        src = (data.get("ddg_source", {}) or {}).get(m["key"], "none")
        conf = data.get("confidence", "?")
        if not result.success:
            _p(f"     [error] {result.error}")
        rows.append({**m, "ddg": ddg, "source": src,
                     "confidence": conf, "bucket": _bucket(ddg), "secs": dt})
        _p(f"     {m['key']}: ddG={('%.3f' % ddg) if ddg is not None else 'None':>8} "
           f"[{src}] {_bucket(ddg)}  ({dt:.0f}s)")

    # ── Results table ─────────────────────────────────────────────────────────
    _rule("RESULTS TABLE")
    _p(f"  {'Mutation':<8} {'Group':<15} {'ddG (kcal/mol)':>15} "
       f"{'source':<11} {'conf':<6} {'bucket':<12}")
    _p("  " + "-" * 74)
    for r in rows:
        ddg_s = f"{r['ddg']:+.3f}" if r["ddg"] is not None else "  None"
        _p(f"  {r['key']:<8} {r['group']:<15} {ddg_s:>15} "
           f"{r['source']:<11} {str(r['confidence']):<6} {r['bucket']:<12}")

    # ── Summary assessment ────────────────────────────────────────────────────
    _rule("SUMMARY ASSESSMENT")

    def _vals(keys):
        return {r["key"]: r["ddg"] for r in rows
                if r["key"] in keys and r["ddg"] is not None}

    buried = _vals(_BURIED)
    surface = _vals(_SURFACE)
    v82a = next((r["ddg"] for r in rows if r["key"] == "V82A"), None)

    if buried and surface:
        min_buried = min(buried.values())
        max_surface = max(surface.values())
        mean_buried = sum(buried.values()) / len(buried)
        mean_surface = sum(surface.values()) / len(surface)
        _p(f"Buried-charged ddG : {buried}")
        _p(f"Surface ddG        : {surface}")
        _p(f"mean(buried)={mean_buried:+.3f}   mean(surface)={mean_surface:+.3f}")
        sep_ok = min_buried > max_surface
        _p(f"\n1) Buried/charged clearly > surface?  "
           f"{'YES' if sep_ok else 'NO'}  "
           f"(min buried {min_buried:+.3f} vs max surface {max_surface:+.3f})")
    else:
        sep_ok = False
        _p("1) Could not compare buried vs surface (missing values).")

    if v82a is not None:
        in_range = 1.5 <= v82a <= 2.0
        near = -1.0 <= (v82a - 1.75) <= 1.0   # within ~1 kcal/mol of midpoint
        verdict = ("in range" if in_range else
                   "near range (+-1)" if near else "OUTSIDE expected range")
        _p(f"2) V82A = {v82a:+.3f} kcal/mol vs experimental ~+1.5..+2.0  ->  {verdict}")
    else:
        _p("2) V82A produced no value.")

    if buried and surface:
        spread_ok = mean_buried > mean_surface + 1.0
        _p(f"3) Spread physically sensible (buried-charge >> surface)?  "
           f"{'YES' if spread_ok else 'NO'}  "
           f"(delta mean = {mean_buried - mean_surface:+.3f} kcal/mol)")
    else:
        spread_ok = False
        _p("3) Could not assess spread.")

    _rule("VERDICT")
    all_empirical = rows and all(r["source"] == "empirical" for r in rows)
    if all_empirical:
        _p("All values came from the EMPIRICAL fallback — PyRosetta did not run. "
           "This is NOT a valid magnitude test; investigate the WSL2 path.")
    elif sep_ok and spread_ok:
        _p("PASS: buried-charge mutations are clearly more destabilising than "
           "surface mutations. Magnitudes are physically sensible — the "
           "PyRosetta path is validated and the near-zero surface ddG values "
           "are confirmed correct.")
    else:
        _p("CONCERN: buried mutations did NOT come back clearly larger than "
           "surface mutations. The relax may be over-minimising (washing out "
           "destabilisation) or mutations may not be applied. Inspect per-"
           "mutation output above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
