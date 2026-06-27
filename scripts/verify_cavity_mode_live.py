"""
Live-verify — CAVITY-FILLING MODE in the real panel + real ChimeraX (GPU-FREE). The scan's OUTPUT
(does it surface a sensible, curated handful of large-void fills with honest metrics?) is the gate,
not just "it ran". A real loaded crystal supplies the structure (no fold); the scan reads the CIF;
ChimeraX renders the heatmap + the row-click highlight.

Confirms:
  1. The scan on a real structure detects internal cavities + surfaces a CURATED handful of viable
     fills (few, not a flood; each a conservative small→larger hydrophobic enlargement with honest
     metrics: void volume, fill fraction, clash flag).
  2. The live Cavity TABLE populates (Substitution/Cavity/Void/Fill/Clash/Score/ΔΔG) + the context-
     dependent caveat shows (BOTH literatures — RSV conformational + Matthews thermostability; NOT
     "least-reliable").
  3. The 'Cavity sites' heatmap paints the lining residues (teal→gold in ChimeraX).
  4. A row-click HIGHLIGHTS the right residue in real ChimeraX (the shared glow seam).
  5. Add-to-design stages the fill into the design basket (one substitution, the variable to_aa).

Run: venv/Scripts/python.exe scripts/verify_cavity_mode_live.py   (needs ChimeraX :60001; NO GPU)
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

CIF = Path(__file__).resolve().parent.parent / "cache" / "1MBN.cif"   # myoglobin — internal cavities
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

    # 1) scan on the real CIF — curated-handful gate ─────────────────────────────────────────
    print("1) Cavity scan on a real structure (curated-handful gate)…")
    scan = r._run_cavity_scan({"cif_path": str(CIF)})
    ok1 = scan.success and bool(scan.data["candidates"])
    check("scan detected cavities + surfaced viable fills", ok1,
          scan.summary[:100] if scan.success else scan.error)
    if not ok1:
        return 1
    cands = scan.data["candidates"]
    cavities = scan.data["cavities"]
    check("internal cavities were detected", bool(cavities),
          f"{len(cavities)} cavities; volumes {[round(c['volume']) for c in cavities[:6]]}")
    check("the fill list is a CURATED handful, not a flood", 0 < len(cands) < 40, f"{len(cands)} fills")
    check("every fill is a conservative small→larger hydrophobic enlargement",
          all(c["to_aa"] in __import__("cavity_geometry")._VOLUME_MUTATIONS.get(c["from_aa"], []) for c in cands),
          f"top: {cands[0]['from_aa']}{cands[0]['position']}{cands[0]['to_aa']} void={cands[0]['void_volume']:.0f}Å³")

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
    print("3) The live Cavity table populates + context-dependent caveat…")
    panel.apply_cavity_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "cavity_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    app.processEvents()
    tab = panel.cavity_tab
    headers = [tab._tbl.horizontalHeaderItem(i).text() for i in range(tab._tbl.columnCount())]
    check("the table populated with the cavity columns",
          tab._tbl.rowCount() == len(cands) and "Void (Å³)" in headers and "Clash" in headers,
          " | ".join(headers))
    cav_text = tab._caveat.text().lower()
    check("the context-dependent caveat shows (RSV + Matthews, NOT 'least-reliable')",
          tab._caveat.isVisibleTo(tab) and "rsv" in cav_text and "matthews" in cav_text
          and "least-reliable" not in cav_text)
    check("the heatmap auto-surfaced + painted the lining residues",
          panel._mode_key == "result:cavity_scan" and len(panel._cavity_scan_panel_hex(panel._cur_tab())) > 0,
          f"mode={panel._mode_key}, painted={len(panel._cavity_scan_panel_hex(panel._cur_tab()))} residues")

    # 4) row-click → highlight the right residue in REAL ChimeraX ────────────────────────────
    print("4) Row-click → highlight the residue in REAL ChimeraX…")
    run("~select")
    sent.clear()
    tab._tbl.cellClicked.emit(0, 0)                          # the top fill candidate
    app.processEvents()
    sel = run("info atoms sel").get("value") or ""
    check("clicking the top row SELECTS that residue in ChimeraX",
          str(cands[0]["position"]) in sel, f"selected {ch}:{cands[0]['position']}")
    check("the glow is recorded (Clear control lit)",
          panel._glow_state is not None and tab._clear_glow_btn.isEnabled())

    # 5) add-to-design → basket ──────────────────────────────────────────────────────────────
    print("5) Add-to-design stages the fill into the basket…")
    n0 = len(panel.design_basket.entries)
    panel._add_cavity_to_basket(cd, cands[0])
    app.processEvents()
    e = panel.design_basket.entries[-1] if panel.design_basket.entries else {}
    check("the fill was staged as a Cavity entry (one sub, the variable to_aa)",
          len(panel.design_basket.entries) == n0 + 1 and e.get("cls") == "Cavity"
          and e.get("subs", [{}])[0].get("to_aa") == cands[0]["to_aa"],
          f"entry subs={e.get('subs')}")

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
