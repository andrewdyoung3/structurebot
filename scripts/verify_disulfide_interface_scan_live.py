"""
Live-verify — INTERFACE SCAN (step 4): find inter-subunit disulfide sites on a REAL two-chain fold,
show them chain-labeled in the Disulfides tab, highlight a pair ACROSS chains in REAL ChimeraX, and
confirm a Declare reuses the PROVEN step-3 cross-chain Mode-C path. The interface scan itself is
GPU-FREE (reads the fold's backbone); a small REAL Boltz dimer supplies the two-chain fold, and REAL
ChimeraX renders the cross-chain highlight (verify the real path, not a proxy).

Confirms:
  1. A real Boltz dimer fold lands (the two-chain construct).
  2. The interface scan finds INTER-CHAIN candidate sites (every pair spans chain A + chain B), and
     they are interface-bounded (the Cα prefilter).
  3. The Disulfides tab's I section lists them CHAIN-LABELED (A:x ↔ B:y — pair_label cross-chain).
  4. A row-click HIGHLIGHTS both members ACROSS chains in REAL ChimeraX (selection spans A and B).
  5. Declare on a found pair feeds the cross-chain Mode-C declare → a constrained fold spec with
     atom1:[A,…], atom2:[B,…] (the step-3 path, reused not reimplemented).

Run: venv/Scripts/python.exe scripts/verify_disulfide_interface_scan_live.py   (one small Boltz fold; needs ChimeraX :60001)
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

SEQ = "ACAYKQDGSACTWVGAA"     # 17 res; folds as a compact dimer with a chain–chain interface
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    panel._run_commands_bg = lambda cmds: [run(c) for c in cmds]   # synchronous → reaches ChimeraX
    r = ToolRouter(bridge=MagicMock(), session=SessionState())

    # 1) REAL Boltz dimer fold (the two-chain construct) ──────────────────────────────────
    print("1) Real Boltz dimer fold…")
    fold = r._get_boltz_bridge().predict(
        [{"id": "A", "sequence": SEQ}, {"id": "B", "sequence": SEQ}], seed=0, allow_remote=False)
    check("dimer fold succeeded", fold.get("success"), fold.get("error") or "")
    if not fold.get("success"):
        return 1
    cif = fold["cif_path"]

    # open it in REAL ChimeraX + seed the panel design to the folded-dimer state
    before = set(_model_ids())
    run(f'open "{Path(cif).as_posix()}"')
    mids = sorted(set(_model_ids()) - before, key=int)
    if not mids:
        print("  could not open the dimer in ChimeraX"); return 1
    mid = mids[-1]
    panel._add_sequence_construct("dimer", SEQ)
    cd = next(iter(panel._design.chains.values()))
    cd.members = [(mid, "A"), (mid, "B")]
    cd.rep_model, cd.rep_chain = mid, "A"
    cd.template_fold = {"engine": "boltz", "target": "assembly", "model_id": mid, "cif_path": cif}

    # 2) INTERFACE SCAN — inter-chain candidate sites (cheap, reads the CIF) ───────────────
    print("2) Interface scan — inter-chain candidate sites…")
    scan = r._run_disulfide_interface_scan({"cif_path": cif})
    check("interface scan found inter-chain candidate sites", scan.success and bool(scan.data["pairs"]),
          scan.summary if scan.success else scan.error)
    if not (scan.success and scan.data["pairs"]):
        return 1
    pairs = scan.data["pairs"]
    check("EVERY candidate spans two chains (chain_a ≠ chain_b)",
          all(p["chain_a"] != p["chain_b"] for p in pairs),
          f"{len(pairs)} pairs; top {pairs[0]['chain_a']}:{pairs[0]['resnum_a']}–"
          f"{pairs[0]['chain_b']}:{pairs[0]['resnum_b']} score {pairs[0]['score']:.2f}")

    # 3) the Disulfides tab's I section lists them CHAIN-LABELED ───────────────────────────
    print("3) The Disulfides tab's I section (chain-labeled)…")
    panel.apply_disulfide_interface_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_interface_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    app.processEvents()
    tbl = panel.disulfides_tab._sec["I"]["table"]
    top = pairs[0]
    want_label = f"{top['chain_a']}:{top['resnum_a']} ↔ {top['chain_b']}:{top['resnum_b']}"
    check("the I section lists candidates CHAIN-LABELED (A:x ↔ B:y)",
          tbl.rowCount() == len(pairs) and tbl.item(0, 0).text() == want_label,
          f"{tbl.rowCount()} rows; top '{tbl.item(0,0).text()}'")

    # 4) row-click → HIGHLIGHT both members ACROSS chains in REAL ChimeraX ─────────────────
    print("4) Row-click → highlight both members ACROSS chains in REAL ChimeraX…")
    run("~select")
    tbl.cellClicked.emit(0, 0)
    app.processEvents()
    sel = run("info atoms sel").get("value") or ""
    spans_both = (f"/{top['chain_a']} " in sel or f"/{top['chain_a']}:" in sel) and \
                 (f"/{top['chain_b']} " in sel or f"/{top['chain_b']}:" in sel)
    check("clicking the top pair SELECTS residues on BOTH chains in ChimeraX",
          str(top["resnum_a"]) in sel and str(top["resnum_b"]) in sel and spans_both,
          f"selected {top['chain_a']}:{top['resnum_a']} + {top['chain_b']}:{top['resnum_b']}")

    # 5) Declare on a found pair → the PROVEN step-3 cross-chain Mode-C path ────────────────
    print("5) Declare → cross-chain Mode-C (reuses step 3)…")
    emitted = []
    panel.launchRequested.connect(lambda s: emitted.append(s))
    panel.disulfides_tab._sec["I"]["table"].selectRow(0)
    panel.disulfides_tab._sec["I"]["declare_btn"].click()
    cons = (emitted[-1]["tool_inputs"]["disulfide_constraints"] if emitted else [])
    check("Declare feeds the cross-chain constrained fold (atom1:[A,…], atom2:[B,…])",
          bool(cons) and cons[0]["atom1"][0] == top["chain_a"] and cons[0]["atom2"][0] == top["chain_b"],
          str(cons))

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
