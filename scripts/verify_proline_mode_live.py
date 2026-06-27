"""
Live-verify — PROLINE MODE in the real panel + real ChimeraX (GPU-FREE). The scan's OUTPUT (does
the ranking look biophysically sensible — loop/turn sites on top, helix interiors demoted/flagged)
is the gate, not just "it ran". A real loaded crystal supplies the structure (no fold); the scan
reads the CIF; ChimeraX renders the heatmap + the row-click highlight + the existing-proline overlay.

Confirms:
  1. The scan on a real structure ranks SENSIBLY — top candidates are NOT backbone-H-bond donors
     (loop/turn-like, φ near the proline ideal), and a substantial fraction of residues ARE flagged
     as donors (helix/sheet interiors — demoted), with NO explicit SS-assignment term.
  2. The live Proline TABLE populates (Residue/φ/ψ/Score/H-bond-donor/ΔΔG) + the caveat shows.
  3. The 'Proline sites' heatmap paints the structure (per-residue magenta in ChimeraX).
  4. A row-click HIGHLIGHTS the right residue in real ChimeraX (the glow seam).
  5. 'Show existing prolines' highlights the existing prolines.

Run: venv/Scripts/python.exe scripts/verify_proline_mode_live.py   (needs ChimeraX :60001; NO GPU)
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

CIF = Path(__file__).resolve().parent.parent / "cache" / "1MBN.cif"   # myoglobin — helices + loops
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def main():
    if not CIF.is_file():
        print(f"cached CIF not found: {CIF}"); return 1
    if not bridge.is_running():
        print("ChimeraX REST not reachable on :60001 — open ChimeraX (or the app) first.")
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    sent = []
    panel._run_commands_bg = lambda cmds: (sent.extend(cmds), [run(c) for c in cmds])
    r = ToolRouter(bridge=MagicMock(), session=SessionState())

    # 1) scan on the real CIF — biophysical-sense gate ──────────────────────────────────────
    print("1) Proline scan on a real structure (biophysical-sense gate)…")
    scan = r._run_proline_scan({"cif_path": str(CIF)})
    ok1 = scan.success and bool(scan.data["candidates"])
    check("scan found proline-stabilization candidates", ok1, scan.summary[:90] if scan.success else scan.error)
    if not ok1:
        return 1
    cands = scan.data["candidates"]
    top5 = cands[:5]
    top_nondonor = sum(1 for c in top5 if not c["hbond_donates"])
    check("top sites are loop/turn-like (NOT backbone-H-bond donors)", top_nondonor >= 4,
          f"{top_nondonor}/5 top sites are non-donors; "
          f"top {top5[0]['from_aa']}{top5[0]['position']} φ={top5[0]['phi']} donor={top5[0]['hbond_donates']}")
    check("top sites are φ-compatible (near the proline ideal −63°)",
          all(abs(c["phi"] - (-63.0)) < 30 for c in top5),
          f"top φ values: {[c['phi'] for c in top5]}")
    flagged = sum(1 for c in cands if c["hbond_donates"])
    check("a substantial fraction ARE flagged as H-bond donors (helix/sheet → demoted, no SS term)",
          flagged > 0.2 * len(cands), f"{flagged}/{len(cands)} flagged as donors (demoted)")

    # 2) open in REAL ChimeraX + seed a loaded-structure construct ──────────────────────────
    print("2) Open in REAL ChimeraX + seed the loaded-structure panel…")
    before = set(_model_ids())
    run(f'open "{CIF.as_posix()}"')
    mids = sorted(set(_model_ids()) - before, key=int)
    if not mids:
        print("could not open the structure"); return 1
    mid = mids[-1]
    ch = (re.findall(r"/([A-Za-z0-9]+)", run(f"info chains #{mid}").get("value") or "") or ["A"])[0]
    panel._add_sequence_construct("mb", "A" * 30)
    cd = next(iter(panel._design.chains.values()))
    cd.members = [(mid, ch)]
    cd.rep_model, cd.rep_chain = mid, ch
    cd.template_fold = {"engine": "loaded", "target": "monomer", "model_id": mid, "cif_path": str(CIF)}

    # 3) the live TABLE populates + caveat ──────────────────────────────────────────────────
    print("3) The live Proline table populates + caveat…")
    panel.apply_proline_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "proline_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    app.processEvents()
    tab = panel.proline_tab
    headers = [tab._tbl.horizontalHeaderItem(i).text() for i in range(tab._tbl.columnCount())]
    check("the table populated with the proline columns",
          tab._tbl.rowCount() == len(cands) and "H-bond donor" in headers,
          " | ".join(headers))
    check("the caveat is shown (measured-not-promised)",
          tab._caveat.isVisibleTo(tab) and "does not confirm" in tab._caveat.text().lower())
    check("the heatmap auto-surfaced + painted the structure",
          panel._mode_key == "result:proline_scan" and len(panel._proline_scan_panel_hex(panel._cur_tab())) > 0,
          f"mode={panel._mode_key}, painted={len(panel._proline_scan_panel_hex(panel._cur_tab()))} residues")

    # 4) row-click → highlight the right residue in REAL ChimeraX ────────────────────────────
    print("4) Row-click → highlight the residue in REAL ChimeraX…")
    run("~select")
    sent.clear()
    tab._tbl.cellClicked.emit(0, 0)                          # the top candidate
    app.processEvents()
    sel = run("info atoms sel").get("value") or ""
    check("clicking the top row SELECTS that residue in ChimeraX",
          str(top5[0]["position"]) in sel, f"selected {ch}:{top5[0]['position']}")
    check("the glow is recorded (Clear control lit on both tabs)",
          panel._glow_state is not None and tab._clear_glow_btn.isEnabled())

    # 5) existing prolines highlight ────────────────────────────────────────────────────────
    print("5) Show existing prolines…")
    run("~select")
    sent.clear()
    panel._show_existing_prolines(cd)
    app.processEvents()
    existing = scan.data["existing"]
    check("existing prolines are highlighted", bool(existing) and any("purple" in c for c in sent),
          f"{len(existing)} existing prolines")

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
