"""
Live-verify — ΔΔG-ESCALATION (analysis-side §9 convergence), with REAL PyRosetta-local ddG. The whole
point of THIS verify is the part that genuinely needs the ACTUAL scoring backend (not a mock): a REAL
ΔΔG comes back for the RIGHT mutation when a geometric interface hit is escalated into the legacy
bridge. Loads `.env.local` so ROSETTA_BACKEND=local (PyRosetta via WSL2) is active.

ChimeraX-FREE by design: the ddG path (PyRosetta-local) needs NO ChimeraX, and the production
ChimeraX step (save the LIVE model → PDB for the legacy bridge, which is PDB-only) is already
live-verified by `verify_disulfide_loaded_pdb_live.py` (the loaded-model save+highlight) and unit-tested
(`build_disulfide_ddg_spec` saves the PDB). So this verify feeds the router a REAL on-disk PDB directly
and exercises the genuine ΔΔG engine end-to-end.

Smallest cached dimer (1C9O CspB, 2×66 res → tractable real relax; the 2nd run reuses the 1st's cached
WT relax). Two REAL ddG runs:
  A) LOADED source — real interface scan (on the CIF) → escalate the top cross-chain pair → REAL ΔΔG
     via the NARROW primitive (`_score_stability`); confirm a real number for BOTH X→C mutations,
     attached to the RIGHT pair, from_aa verified against the scored PDB, analyze() NEVER called, base
     caveat present.
  B) DE-NOVO source — the SAME structure scored with source='denovo' → REAL ΔΔG comes back AND the
     Disulfides-tab detail adds the two-layer estimate-on-estimate caveat.

Run: venv/Scripts/python.exe scripts/verify_disulfide_ddg_escalation_live.py
     (real PyRosetta-local ddG — several minutes; needs WSL2+PyRosetta. No ChimeraX, no GPU.)
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config
config.load_env_file()                       # ROSETTA_BACKEND=local (PyRosetta via WSL2)

from unittest.mock import MagicMock
from session_state import SessionState
from tool_router import ToolRouter
from variant_workbench import DisulfidesResultsTab
from disulfide_bridge import parse_pdb_atoms, _three_to_one

ROOT = Path(__file__).resolve().parent.parent
CIF = ROOT / "cache" / "1C9O.cif"            # interface scan reads this (no ChimeraX)
PDB = ROOT / "cache" / "1C9O.pdb"            # the legacy bridge scores this (PyRosetta cleanATOM = PDB)

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def main():
    # 0) backend must be the LOCAL PyRosetta path (real ddG, no web upload) ──────────────────
    from rosetta_bridge import _select_backend
    backend = _select_backend()
    print(f"0) stability backend = {backend}", flush=True)
    check("backend is PyRosetta-local (real ddG, all local)", backend == "local",
          f"backend={backend} (need ROSETTA_BACKEND=local + WSL2/PyRosetta)")
    if backend != "local":
        print("  ABORT: a real ΔΔG needs the local PyRosetta backend; refusing to fake it.")
        return 1
    if not (CIF.is_file() and PDB.is_file()):
        print(f"  ABORT: need {CIF.name} + {PDB.name} in cache/."); return 1

    r = ToolRouter(bridge=MagicMock(), session=SessionState())

    # 1) REAL interface scan → pick the top cross-chain pair whose WT residues are in the PDB ──
    print("1) Interface scan (real, on the CIF)…", flush=True)
    scan = r._run_disulfide_interface_scan({"cif_path": str(CIF)})
    pairs = scan.data["pairs"] if scan.success else []
    check("interface scan found cross-chain candidate sites", bool(pairs),
          f"{len(pairs)} pairs" if pairs else (scan.error or ""))
    if not pairs:
        return 1
    atoms = parse_pdb_atoms(str(PDB))
    def _aa(ch, rn):
        res = (atoms.get(ch) or {}).get(int(rn))
        return _three_to_one(res.get("resname", "UNK")) if res else None
    top = next((p for p in pairs if _aa(p["chain_a"], p["resnum_a"]) and _aa(p["chain_b"], p["resnum_b"])), None)
    check("a top pair maps to real WT residues in the PDB", top is not None)
    if top is None:
        return 1
    aa_a, aa_b = _aa(top["chain_a"], top["resnum_a"]), _aa(top["chain_b"], top["resnum_b"])
    print(f"   top pair {top['chain_a']}:{top['resnum_a']}{aa_a} ↔ "
          f"{top['chain_b']}:{top['resnum_b']}{aa_b} — running REAL ddG (minutes)…", flush=True)

    # spy: the narrow primitive scores this pair; analyze() (the find/filter pipeline) must NOT run
    dbridge = r._get_disulfide_bridge()
    dbridge.analyze = MagicMock(side_effect=AssertionError("analyze() must NOT be called"))

    def _inputs(source):
        return {"pdb_path": str(PDB),
                "chain_a": top["chain_a"], "resnum_a": top["resnum_a"], "from_aa_a": aa_a,
                "chain_b": top["chain_b"], "resnum_b": top["resnum_b"], "from_aa_b": aa_b,
                "source": source}

    # A) LOADED — REAL ddG attached to the RIGHT pair, base caveat, analyze() never called ─────
    print("A) Loaded source — real ΔΔG escalation…", flush=True)
    out = r._run_disulfide_ddg_estimate(_inputs("loaded"))
    check("real ΔΔG returned successfully", out.success, out.error if not out.success else "")
    if not out.success:
        return 1
    d = out.data
    check("REAL ΔΔG numbers for BOTH X→C mutations (non-None)",
          d.get("ddg_a") is not None and d.get("ddg_b") is not None,
          f"{d['from_aa_a']}{d['resnum_a']}C {d['ddg_a']:+.2f}, "
          f"{d['from_aa_b']}{d['resnum_b']}C {d['ddg_b']:+.2f} kcal/mol (backend {d['backend']})")
    check("ΔΔG is for the RIGHT pair (the one escalated)",
          d["chain_a"] == top["chain_a"] and d["resnum_a"] == top["resnum_a"]
          and d["chain_b"] == top["chain_b"] and d["resnum_b"] == top["resnum_b"])
    check("backend used was PyRosetta-local", d["backend"] == "local")
    check("NEVER routed through disulfide_bridge.analyze (narrow primitive only)",
          not dbridge.analyze.called)
    check("base ΔΔG caveat present (uncalibrated; not confirmation)",
          "uncalibrated" in (out.summary or "").lower())

    # the tab detail renders the number + base caveat, NO de-novo layer for a loaded source
    loaded_pair = dict(top, ddg_a=d["ddg_a"], ddg_b=d["ddg_b"], ddg_backend=d["backend"],
                       ddg_source="loaded", from_aa_a=aa_a, from_aa_b=aa_b)
    det = DisulfidesResultsTab._pair_detail("I", loaded_pair)
    check("loaded detail shows ΔΔG + base caveat, NO de-novo layer",
          "uncalibrated" in det and "not confirmation" in det and "estimate on an estimate" not in det)

    # B) DE-NOVO — REAL ddG (reuses cached WT relax) + the two-layer estimate-on-estimate caveat ─
    print("B) De-novo source — real ΔΔG (reuses cached WT relax) + extra caveat…", flush=True)
    dn = r._run_disulfide_ddg_estimate(_inputs("denovo"))
    check("real ΔΔG returned under source='denovo'",
          dn.success and dn.data.get("ddg_a") is not None,
          (f"{dn.data['from_aa_a']}{dn.data['resnum_a']}C {dn.data['ddg_a']:+.2f} kcal/mol"
           if dn.success else (dn.error or "")))
    if dn.success:
        dn_pair = dict(loaded_pair, ddg_a=dn.data["ddg_a"], ddg_b=dn.data["ddg_b"], ddg_source="denovo")
        dn_det = DisulfidesResultsTab._pair_detail("I", dn_pair)
        check("de-novo detail ADDS the estimate-on-estimate two-layer caveat",
              "estimate on an estimate" in dn_det and "predicted structure" in dn_det.lower(),
              dn_det[-110:])

    ok = all(_checks) and bool(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
