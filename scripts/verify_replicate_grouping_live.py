"""
Live-verify — REPLICATE-ROW GROUPING in the REAL panel (GPU-FREE). The user reported (from LOOKING at
a trimer's Disulfides Mode-D table) that the SAME engineerable site shows up once per chain — three
near-identical rows — and asked to collapse them WITHOUT losing the small per-copy geometry differences
(real asymmetry in a predicted/crystal assembly). So this verify LOOKS at the live table on a real
homo-oligomer: the replicate rows collapse to one (×N badge), a varying geometry column shows its
min–max range, the row tooltip lists every copy, the header toggle un-groups back to per-chain rows,
and a click on a grouped row lights ALL copies in REAL ChimeraX (one union selection).

GPU-FREE: a real cached HOMO-oligomer crystal (1JS0 — a homotrimer with its three copies in the
asymmetric unit, so NCS gives each a slightly DIFFERENT backbone — the crystal analogue of a predicted
trimer's asymmetry) supplies the equivalent chains; the Mode-D engineering scan reads the CIF directly
(no fold), REAL ChimeraX renders the glow.

Confirms:
  1. The Mode-D scan on a homotrimer surfaces the SAME site on each equivalent chain (the redundancy).
  2. Grouped (default): the live D-table collapses those copies into one row with a ×N badge, and
     FEWER rows than raw candidates (the de-clutter the user asked for).
  3. The per-copy geometry is PRESERVED — a column that varies across copies shows a `lo–hi` range, and
     the row's tooltip lists each chain's exact value (real asymmetry kept, not discarded).
  4. Unchecking "Group equivalent chains" restores one row PER chain copy (each copy's own geometry);
     re-checking collapses again — the toggle the user asked for.
  5. A click on a grouped row emits ONE union selection spanning every equivalent copy → all copies
     glow together in REAL ChimeraX.

Run: venv/Scripts/python.exe scripts/verify_replicate_grouping_live.py [port]   (needs ChimeraX; NO GPU)
     (port defaults to 60001; the app usually serves ChimeraX there.)
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

# A real cached HOMO-oligomer with copies in the ASYMMETRIC UNIT → real per-copy (NCS) geometry spread:
# 1JS0 = a homotrimer (A/B/C, Δ Cα–Cα ~0.13 Å across copies). Fall back to 1C9O (a homodimer, Δ~0.12 Å).
CIF = Path(__file__).resolve().parent.parent / "cache" / "1JS0.cif"
if not CIF.is_file():
    CIF = Path(__file__).resolve().parent.parent / "cache" / "1C9O.cif"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 60001
bridge = ChimeraXBridge(port=PORT)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def main():
    if not CIF.is_file():
        print(f"cached CIF not found: {CIF} — fetch a homo-oligomer first."); return 1
    if not bridge.is_running():
        print(f"ChimeraX REST not reachable on :{PORT} — open ChimeraX (or the app) first, or pass the "
              f"right port as argv[1]. This script will NOT auto-launch (avoids the process pile-up).")
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pushed = []                                          # every command the panel would send to ChimeraX
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    panel._run_commands_bg = lambda cmds: (pushed.extend(cmds), [run(c) for c in cmds])[0]
    r = ToolRouter(bridge=MagicMock(), session=SessionState())

    # 1) Mode-D engineering scan on the real homotrimer CIF (per-chain, cheap) ────────────────
    print("1) Mode-D engineering scan on a real homotrimer (per-chain, no GPU)…")
    scan = r._run_disulfide_scan({"cif_path": str(CIF)})
    ok1 = scan.success and bool(scan.data["pairs"])
    check("engineering scan found engineerable sites", ok1, scan.summary if scan.success else scan.error)
    if not ok1:
        return 1
    pairs = scan.data["pairs"]
    scan_chains = sorted({str(p.get("chain_a")) for p in pairs} | {str(p.get("chain_b")) for p in pairs})
    check("the scan spans multiple equivalent chains (a homo-oligomer)", len(scan_chains) >= 2,
          f"chains {scan_chains}")
    # the SAME (resnum_a,resnum_b) appears on more than one chain → the redundancy the user saw
    from collections import Counter
    site_counts = Counter((p["resnum_a"], p["resnum_b"]) for p in pairs)
    replicated = [s for s, n in site_counts.items() if n >= 2]
    check("at least one site is scanned on 2+ chains (the replicate redundancy)", bool(replicated),
          f"{len(replicated)} replicated sites; e.g. {replicated[:2]}")
    if not replicated or len(scan_chains) < 2:
        return 1

    # 2) open the trimer in REAL ChimeraX + seed a HOMO-oligomer construct (one cd, 3 members) ─
    print("2) Open the trimer in REAL ChimeraX + seed the homo-oligomer panel…")
    before = set(_model_ids())
    run(f'open "{CIF.as_posix()}"')
    mids = sorted(set(_model_ids()) - before, key=int)
    if not mids:
        print("  could not open the trimer in ChimeraX"); return 1
    mid = mids[-1]
    panel._add_sequence_construct("pcna", "A" * 20)
    cd = next(iter(panel._design.chains.values()))
    cd.members = [(mid, ch) for ch in scan_chains]       # ONE cd, the equivalent copies → _chain_equiv
    cd.rep_model, cd.rep_chain = mid, scan_chains[0]
    cd.template_fold = {"engine": "loaded", "target": "assembly", "model_id": mid, "cif_path": str(CIF)}
    check("panel resolves the chains as equivalent copies (share one cd)",
          set(panel._chain_equiv(scan_chains[0])) == set(scan_chains),
          f"_chain_equiv({scan_chains[0]}) = {panel._chain_equiv(scan_chains[0])}")

    # 3) apply → the grouped D-table collapses the replicates (×N badge, fewer rows) ───────────
    print("3) The live D-table — grouped (default): replicates collapse, badge, fewer rows…")
    panel.apply_disulfide_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    app.processEvents()
    tab = panel.disulfides_tab
    tbl = tab._sec["D"]["table"]
    grouped_rows = tbl.rowCount()
    check("grouped table has FEWER rows than raw candidates (de-cluttered)", grouped_rows < len(pairs),
          f"{grouped_rows} grouped rows vs {len(pairs)} raw candidates")
    badge_rows = [i for i in range(grouped_rows) if "×" in (tbl.item(i, 0).text() if tbl.item(i, 0) else "")]
    check("a grouped row carries the ×N copy badge", bool(badge_rows),
          tbl.item(badge_rows[0], 0).text() if badge_rows else "no ×N row")
    check("the 'Group equivalent chains' toggle is offered", not tab._group_chk.isHidden())

    # 4) per-copy geometry PRESERVED — a range in a varying column + a per-copy tooltip ─────────
    print("4) Per-copy geometry preserved (range + tooltip)…")
    grow = badge_rows[0] if badge_rows else 0
    headers = [tbl.horizontalHeaderItem(j).text() for j in range(tbl.columnCount())]
    # scan EVERY grouped row for a lo–hi range cell (real per-copy asymmetry surfaced somewhere)
    ranged = sorted({headers[j] for i in (badge_rows or [grow]) for j in range(1, tbl.columnCount())
                     if "–" in (tbl.item(i, j).text() if tbl.item(i, j) else "")})
    check("a varying geometry column shows a lo–hi range (real per-copy asymmetry preserved)",
          bool(ranged),                                  # both fixtures have NCS spread → expect a range
          f"ranged columns: {ranged or 'none (copies identical to display precision — symmetric fixture)'}")
    tip = tbl.item(grow, 0).toolTip()
    n_chain_lines = tip.count("chain ")
    check("the grouped row's tooltip lists each equivalent copy", n_chain_lines >= 2,
          f"{n_chain_lines} per-copy lines in tooltip")

    # 5) a grouped row-click lights ALL copies in REAL ChimeraX (one union selection) ──────────
    print("5) A grouped row-click glows ALL copies (union selection) in REAL ChimeraX…")
    panel._clear_disulfide_glow()                        # reset any auto-glow so the click APPLIES (not toggle-off)
    pushed.clear()
    tbl.selectRow(grow)
    tab._on_row("D", grow)                                # the row-click path → panel highlight seam
    app.processEvents()
    sels = [c for c in pushed if c.startswith("select ") and "#" in c]
    union = next((c for c in sels if all(f"/{ch}:" in c for ch in scan_chains)), "")
    check("one selection spans EVERY equivalent copy (all chains glow together)", bool(union),
          union or f"selects seen: {sels[-1] if sels else 'none'}")

    # 6) the toggle un-groups back to per-chain rows, then re-groups ───────────────────────────
    print("6) The toggle un-groups to per-chain rows, then re-groups…")
    tab._group_chk.setChecked(False)
    app.processEvents()
    ungrouped_rows = tbl.rowCount()
    check("unchecking shows MORE rows (one per chain copy)", ungrouped_rows > grouped_rows,
          f"{ungrouped_rows} ungrouped vs {grouped_rows} grouped")
    check("no ×N badge when ungrouped", all("×" not in (tbl.item(i, 0).text() if tbl.item(i, 0) else "")
                                            for i in range(ungrouped_rows)))
    tab._group_chk.setChecked(True)
    app.processEvents()
    check("re-checking collapses the replicates again", tbl.rowCount() == grouped_rows,
          f"{tbl.rowCount()} rows (was {grouped_rows} grouped)")

    print()
    ok = all(_checks)
    print(f"{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
