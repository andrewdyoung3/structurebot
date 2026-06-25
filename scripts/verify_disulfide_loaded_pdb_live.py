"""
Live-verify — B / D / INTERFACE on a LOADED PDB (the structure-source abstraction). The reported
symptom: the fold-based disulfide suite was greyed on a loaded PDB because the preconditions required
a DE-NOVO construct. This verify opens REAL PDBs in REAL ChimeraX, drives the panel through the GENUINE
loaded-model path (`panel.load_model` → REST chain read, source='structure'), and confirms the cheap
read-modes now work on a loaded structure — A/C stay de-novo-only, and a pair-click highlights the
correct residues on the LIVE rendered model (not the temp save).

Two real loaded PDBs, both from the local cache (no network):
  • 7RSA  (RNase A — MONOMER, 4 disulfides): B measures the existing cys pairs, D finds engineerable
    intra sites, interface is GREYED (monomer), A + C are GREYED (de-novo-only). Pair-click highlights
    on the live model #mid.
  • 1F0V_AB (2 chains A/B, with cysteines): interface scan ENABLES + finds CROSS-chain sites, and a
    row-click highlights both members ACROSS chains on the live model.

Confirms:
  1. A loaded PDB seeds a source='structure' design (no template_fold) — the greyed-on-a-PDB case.
  2. GATE matrix on a real loaded PDB: B + D enabled; A + C greyed; interface greyed on the monomer,
     enabled on the multimer.
  3. B (geometry) on the loaded model produces REAL cys-pair geometry off a FRESH save of the live model.
  4. D (scan) + interface produce REAL candidate sites off the same fresh-saved structure.
  5. `_active_structure` model_id is the LIVE loaded id; a pair-click selects the right residues on
     #mid in REAL ChimeraX (the live model, not the temp copy).

Run: venv/Scripts/python.exe scripts/verify_disulfide_loaded_pdb_live.py   (no GPU; needs ChimeraX :60001)
"""
import os, sys, re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from unittest.mock import MagicMock
from PySide6 import QtWidgets
from chimerax_bridge import ChimeraXBridge
from session_state import SessionState
from tool_router import ToolRouter
from variant_workbench import VariantWorkbenchPanel
from seq_editor.controller import SequenceEditorController

CACHE = Path(__file__).resolve().parent.parent / "cache"
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _open(cif_name):
    """Open a cached PDB in REAL ChimeraX; return the new model id (str)."""
    before = set(_model_ids())
    run(f'open "{(CACHE / cif_name).as_posix()}"')
    new = sorted(set(_model_ids()) - before, key=int)
    return new[-1] if new else None


def _panel():
    """A panel wired to a REAL controller (run_command → ChimeraX) so the loaded-model SAVE path in
    `_active_structure` reaches ChimeraX for real, and the highlight pushes to the live scene."""
    ctrl = SequenceEditorController(run_command=run, fold_fn=MagicMock())
    p = VariantWorkbenchPanel(ctrl, session=SessionState(), pool=MagicMock())
    p._run_commands_bg = lambda cmds: [run(c) for c in cmds]    # synchronous → reaches ChimeraX
    return p


def verify_monomer(r):
    print("\n=== 7RSA (RNase A, MONOMER, 4 disulfides) — B + D, A/C/interface greyed ===")
    mid = _open("7RSA.cif")
    if not mid:
        print("  could not open 7RSA in ChimeraX"); return False
    p = _panel()
    p.load_model(mid)                                           # the GENUINE loaded-model path

    # 1) a loaded PDB → source='structure', NO template_fold (the greyed-on-a-PDB precondition)
    cd = next(iter(p._design.chains.values()))
    check("loaded PDB seeds a source='structure' design with no template_fold",
          p._design.source == "structure" and not (cd.template_fold or {}).get("model_id"),
          f"source={p._design.source}, rep_model={cd.rep_model}")

    # 2) GATE matrix on the real loaded monomer
    check("B (measure) + D (find) ENABLED on the loaded PDB",
          p._ss_geometry_btn.isEnabled() and p._ss_scan_btn.isEnabled())
    check("A (assess) + C (declare) GREYED (de-novo only)",
          not p._ss_discover_btn.isEnabled() and not p._ss_constrain_btn.isEnabled())
    check("interface GREYED on a monomer", not p._ss_interface_btn.isEnabled())

    # 3) _active_structure saves the LIVE model fresh; B reads REAL cys geometry off it
    src = p._active_structure()
    check("_active_structure yields the LIVE model id + a fresh saved CIF",
          src is not None and src["model_id"] == mid and Path(src["cif_path"]).is_file(),
          f"model_id={src and src['model_id']}, cif={src and Path(src['cif_path']).name}")
    geo = r._run_disulfide_geometry({"cif_path": src["cif_path"]})
    bonded = [g for g in geo.data["pairs"] if g.get("bonding_compatible")] if geo.success else []
    check("B (geometry) measures REAL cys pairs incl. ≥1 disulfide-compatible pair",
          geo.success and bool(geo.data["pairs"]) and bool(bonded),
          f"{len(geo.data['pairs']) if geo.success else 0} pairs, {len(bonded)} bonding-compatible")

    # 4) D (scan) finds engineerable intra sites off the same loaded structure
    scan = r._run_disulfide_scan({"cif_path": src["cif_path"]})
    check("D (scan) finds engineerable intra-chain sites on the loaded PDB",
          scan.success and bool(scan.data["pairs"]),
          f"{len(scan.data['pairs']) if scan.success else 0} candidate sites")

    # 5) a D-row click highlights the right residues on the LIVE model #mid
    p.apply_disulfide_scan_result(
        {"_align_ukey": p._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    QtWidgets.QApplication.instance().processEvents()
    top = scan.data["pairs"][0]
    run("~select")
    p.disulfides_tab._sec["D"]["table"].cellClicked.emit(0, 0)
    QtWidgets.QApplication.instance().processEvents()
    sel = run("info atoms sel").get("value") or ""
    check("D-row click SELECTS the pair on the LIVE model #" + mid + " in ChimeraX",
          f"#{mid}" in sel and str(top["resnum_a"]) in sel and str(top["resnum_b"]) in sel,
          f"pair {top['resnum_a']}–{top['resnum_b']} on #{mid}")

    run(f"close #{mid}")
    return True


def verify_multimer(r):
    print("\n=== 1F0V_AB (2 chains A/B) — interface ENABLES + cross-chain highlight ===")
    mid = _open("1F0V_AB.cif")
    if not mid:
        print("  could not open 1F0V_AB in ChimeraX"); return False
    p = _panel()
    p.load_model(mid)
    n_chains = sum(len(c.members) for c in p._design.chains.values())
    check("loaded multimer → ≥2 chains, source='structure'",
          p._design.source == "structure" and n_chains >= 2, f"{n_chains} chains")
    check("interface ENABLED on the loaded multimer", p._ss_interface_btn.isEnabled())

    src = p._active_structure()
    iface = r._run_disulfide_interface_scan({"cif_path": src["cif_path"]})
    pairs = iface.data["pairs"] if iface.success else []
    check("interface scan finds CROSS-chain sites on the loaded PDB",
          iface.success and bool(pairs) and all(pr["chain_a"] != pr["chain_b"] for pr in pairs),
          f"{len(pairs)} cross-chain candidate sites")
    if not pairs:
        run(f"close #{mid}"); return True

    p.apply_disulfide_interface_scan_result(
        {"_align_ukey": p._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_interface_scan", "success": True,
                                "data": iface.data, "summary": iface.summary}]})
    QtWidgets.QApplication.instance().processEvents()
    top = pairs[0]
    run("~select")
    p.disulfides_tab._sec["I"]["table"].cellClicked.emit(0, 0)
    QtWidgets.QApplication.instance().processEvents()
    sel = run("info atoms sel").get("value") or ""
    spans_both = (f"/{top['chain_a']}" in sel) and (f"/{top['chain_b']}" in sel)
    check("interface row-click SELECTS both members ACROSS chains on the LIVE model #" + mid,
          f"#{mid}" in sel and spans_both and str(top["resnum_a"]) in sel and str(top["resnum_b"]) in sel,
          f"{top['chain_a']}:{top['resnum_a']} ↔ {top['chain_b']}:{top['resnum_b']} on #{mid}")

    run(f"close #{mid}")
    return True


def main():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    r = ToolRouter(bridge=MagicMock(), session=SessionState())
    verify_monomer(r)
    verify_multimer(r)
    ok = all(_checks) and bool(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
