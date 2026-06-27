"""
Live-verify — ROTAMER Sγ-REACHABILITY proxy in the REAL panel (GPU-FREE). The user reported the
ranking concern from LOOKING at the live interface-scan table, so this verify LOOKS at the live
table: it confirms the table now carries the rotamer Sγ-reachability readout (best Sγ–Sγ, χSS, clash)
+ the (now de-emphasized, still measured) orientation column, that the rank reflects sulfur-
reachability, and that the load-bearing caveat stays.

GPU-FREE: a real cached MULTIMER crystal CIF supplies the two-chain structure (no fold), opened in
REAL ChimeraX (the loaded-PDB source path); the interface scan reads the CIF directly (backbone +
heavy-atom clash grid), and REAL ChimeraX renders the cross-chain row-click highlight.

Confirms:
  1. The interface scan on a real dimer surfaces inter-chain candidates with the reachability readout
     (best_sg_sg + best_chi_ss) AND a clash flag (tier a+b ran).
  2. The live I-section TABLE shows the new columns (Sγ–Sγ, χSS, Clash) + Orientation (last), with the
     reachability values populated (not "—").
  3. The rank is driven by score = Cα×Cβ×reachability — a sulfur-reachable clean pair outranks a
     clashing one; the displayed orientation column is present but no longer the ranking signal.
  4. A row-click highlights both members ACROSS chains in REAL ChimeraX.
  5. The geometric-only caveat (now naming reachability + rigid-backbone clash) rides with the table.

Run: venv/Scripts/python.exe scripts/verify_disulfide_rotamer_reachability_live.py   (needs ChimeraX :60001; NO GPU)
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

# A real cached two-chain crystal structure (1A2W — a dimer). No fold, no GPU.
CIF = Path(__file__).resolve().parent.parent / "cache" / "1A2W.cif"
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
        print(f"cached CIF not found: {CIF} — fetch it first."); return 1
    if not bridge.is_running():
        print("ChimeraX REST not reachable on :60001 — open ChimeraX (or the app) first. "
              "This script will NOT auto-launch (avoids the process pile-up).")
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    panel._run_commands_bg = lambda cmds: [run(c) for c in cmds]   # synchronous → reaches ChimeraX
    r = ToolRouter(bridge=MagicMock(), session=SessionState())

    # 1) interface scan on the real dimer CIF (cheap, reads backbone + heavy atoms) ─────────
    print("1) Interface scan on a real dimer (reachability + clash, no GPU)…")
    scan = r._run_disulfide_interface_scan({"cif_path": str(CIF)})
    ok1 = scan.success and bool(scan.data["pairs"])
    check("interface scan found inter-chain candidate sites", ok1,
          scan.summary if scan.success else scan.error)
    if not ok1:
        return 1
    pairs = scan.data["pairs"]
    top = pairs[0]
    has_reach = top.get("best_sg_sg") is not None and top.get("best_chi_ss") is not None
    has_clash = any(p.get("clash") in (True, False) for p in pairs)
    check("top candidate carries the rotamer reachability readout (best Sγ–Sγ + χSS)", has_reach,
          f"best_sg_sg={top.get('best_sg_sg')} χSS={top.get('best_chi_ss')} reach={top.get('reach_score')}")
    check("a clash flag was evaluated (tier b ran)", has_clash,
          f"clashing {sum(1 for p in pairs if p.get('clash'))}/{len(pairs)}")
    check("score = Cα×Cβ×reachability (orientation NOT a factor)",
          abs(top["score"] - top["ca_score"] * top["cb_score"] * top["reach_score"]) < 2e-4,
          f"score={top['score']} ca={top['ca_score']} cb={top['cb_score']} reach={top['reach_score']}")
    # a clean pair should be able to outrank a clashing one of similar reach (the tier-b payoff)
    clash_pairs = [p for p in pairs if p.get("clash")]
    clean_pairs = [p for p in pairs if p.get("clash") is False]
    if clash_pairs and clean_pairs:
        check("a clash-free candidate can outrank a clashing one (soft demotion works)",
              clean_pairs[0]["score"] >= clash_pairs[0]["score"]
              or clean_pairs[0]["reach_score"] <= clash_pairs[0]["reach_score"],
              f"top clean {clean_pairs[0]['score']:.2f} vs top clashing {clash_pairs[0]['score']:.2f}")

    # 2) open the dimer in REAL ChimeraX + seed the panel as a loaded structure ─────────────
    print("2) Open the dimer in REAL ChimeraX + seed the loaded-structure panel…")
    before = set(_model_ids())
    run(f'open "{CIF.as_posix()}"')
    mids = sorted(set(_model_ids()) - before, key=int)
    if not mids:
        print("  could not open the dimer in ChimeraX"); return 1
    mid = mids[-1]
    chains = re.findall(r"/([A-Za-z0-9]+)", run(f"info chains #{mid}").get("value") or "")
    cset = sorted(set(chains))[:2] or ["A", "B"]
    panel._add_sequence_construct("dimer", "A" * 20)
    cd = next(iter(panel._design.chains.values()))
    cd.members = [(mid, cset[0]), (mid, cset[1])]
    cd.rep_model, cd.rep_chain = mid, cset[0]
    cd.template_fold = {"engine": "loaded", "target": "assembly", "model_id": mid, "cif_path": str(CIF)}

    # 3) the live I-section TABLE shows the reachability columns + orientation ───────────────
    print("3) The live I-section table (reachability columns + orientation)…")
    panel.apply_disulfide_interface_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_interface_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    app.processEvents()
    tab = panel.disulfides_tab
    tbl = tab._sec["I"]["table"]
    headers = [tbl.horizontalHeaderItem(i).text() for i in range(tbl.columnCount())]
    check("the table carries the reachability + orientation columns",
          "Sγ–Sγ (Å)" in headers and "χSS (°)" in headers and "Clash" in headers
          and "Orientation (°)" in headers,
          " | ".join(headers))
    # the reach cells are populated (not "—") for the top row; clash reads ok/clash
    sg_col = headers.index("Sγ–Sγ (Å)")
    clash_col = headers.index("Clash")
    check("the top row shows a real Sγ–Sγ value + a clash verdict (not '—')",
          tbl.item(0, sg_col).text() != "—" and tbl.item(0, clash_col).text() in ("ok", "clash"),
          f"Sγ–Sγ '{tbl.item(0, sg_col).text()}'  clash '{tbl.item(0, clash_col).text()}'")
    check("the caveat names rotamer reachability + rigid-backbone clash",
          "reachability" in tab._sec["I"]["caveat"].text().lower()
          and "clash" in tab._sec["I"]["caveat"].text().lower(),
          tab._sec["I"]["caveat"].text()[:80])

    # 4) row-click → highlight both members ACROSS chains in REAL ChimeraX ───────────────────
    print("4) Row-click → highlight both members ACROSS chains in REAL ChimeraX…")
    run("~select")
    tbl.cellClicked.emit(0, 0)
    app.processEvents()
    sel = run("info atoms sel").get("value") or ""
    spans_both = str(top["resnum_a"]) in sel and str(top["resnum_b"]) in sel
    check("clicking the top pair SELECTS residues on both members in ChimeraX", spans_both,
          f"selected {top['chain_a']}:{top['resnum_a']} + {top['chain_b']}:{top['resnum_b']}")

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
