"""
Live-verify — DESIGN BASKET cross-strategy curate→enact→variant flow in the real panel + real
ChimeraX (GPU-FREE). The gate is the CROSS-STRATEGY flow: browse Disulfides → add a pick → Proline →
add a pick → the basket shows BOTH with their class-specific metrics → remove one → a same-residue
conflict flags when a colliding pick is added → Enact → ONE variant appears in the Variant Workbench
ready to fold.

GPU-FREE: a real cached crystal supplies the structure (no fold); the disulfide + proline scans read
the CIF; the basket composes a variant via the existing variant path (no fold here — that's the
designer's opt-in next step).

Run: venv/Scripts/python.exe scripts/verify_design_basket_live.py   (needs ChimeraX :60001; NO GPU)
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

CIF = Path(__file__).resolve().parent.parent / "cache" / "1MBN.cif"   # myoglobin — proline + backbone sites
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
    panel._run_commands_bg = lambda cmds: [run(c) for c in cmds]
    r = ToolRouter(bridge=MagicMock(), session=SessionState())

    # open + seed a loaded-structure construct ──────────────────────────────────────────────
    before = set(_model_ids())
    run(f'open "{CIF.as_posix()}"')
    mids = sorted(set(_model_ids()) - before, key=int)
    if not mids:
        print("could not open the structure"); return 1
    mid = mids[-1]
    ch = (re.findall(r"/([A-Za-z0-9]+)", run(f"info chains #{mid}").get("value") or "") or ["A"])[0]
    # build the construct from the REAL chain sequence so the template axis matches the structure's
    # residue numbering (the basket Enact composes via _col_for_resnum/_from_aa_for against the axis).
    import proline_geometry as _pg
    atoms = _pg.parse_backbone_with_names(str(CIF))
    res = atoms[ch]
    rns = sorted(res)
    seq = "".join(_pg._THREE_TO_ONE.get(str(res[rn].get("resname") or ""), "A") for rn in rns)
    panel._add_sequence_construct("mb", seq)
    cd = next(iter(panel._design.chains.values()))
    # set the template axis resnums to the structure's real auth numbering
    for c, rn in zip(cd.template_cells, rns):
        c.resnum = rn
    cd.members = [(mid, ch)]
    cd.rep_model, cd.rep_chain = mid, ch
    cd.template_fold = {"engine": "loaded", "target": "monomer", "model_id": mid, "cif_path": str(CIF)}

    # 1) run BOTH scans, populate the tabs ───────────────────────────────────────────────────
    print("1) Run the Disulfide + Proline scans…")
    ds = r._run_disulfide_scan({"cif_path": str(CIF)})
    ps = r._run_proline_scan({"cif_path": str(CIF)})
    panel.apply_disulfide_scan_result({"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_scan", "success": True, "data": ds.data, "summary": ds.summary}]})
    panel.apply_proline_scan_result({"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "proline_scan", "success": True, "data": ps.data, "summary": ps.summary}]})
    app.processEvents()
    check("both scans populated their tabs",
          panel.disulfides_tab._sec["D"]["table"].rowCount() > 0 and panel.proline_tab._tbl.rowCount() > 0)

    # 2) add a Disulfide pick + a Proline pick via the "Add to design" buttons ────────────────
    print("2) Add a Disulfide pick + a Proline pick to the basket…")
    panel.disulfides_tab._sec["D"]["table"].selectRow(0)
    panel.disulfides_tab._add_to_basket("D")            # the real button path
    panel.proline_tab._tbl.selectRow(0)
    panel.proline_tab._add_to_basket()
    app.processEvents()
    basket = panel.design_basket
    classes = [e["cls"] for e in basket.entries]
    check("the basket holds BOTH picks with their class-specific metrics",
          "Disulfide" in classes and "Proline" in classes
          and any("Sγ–Sγ" in e["metrics_text"] for e in basket.entries if e["cls"] == "Disulfide")
          and any("φ" in e["metrics_text"] for e in basket.entries if e["cls"] == "Proline"),
          f"classes={classes}")
    n_before_remove = len(basket.entries)

    # 3) remove one ──────────────────────────────────────────────────────────────────────────
    print("3) Remove one pick…")
    basket._list.selectRow(0); basket._remove_selected()
    check("remove drops one entry", len(basket.entries) == n_before_remove - 1)

    # 4) same-residue conflict flags + blocks enact ──────────────────────────────────────────
    print("4) Add a colliding pick → same-residue conflict flags…")
    # re-add the proline pick, then add a disulfide that targets the SAME proline residue
    pro = ps.data["candidates"][0]
    panel._add_proline_to_basket(cd, pro)               # chain/pos → P
    panel._add_disulfide_to_basket(cd, {"chain_a": cd.rep_chain, "resnum_a": pro["position"],
                                        "chain_b": cd.rep_chain, "resnum_b": pro["position"] + 6,
                                        "best_sg_sg": 2.0, "best_chi_ss": -90, "clash": False, "score": 0.7})
    app.processEvents()
    check("a same-residue collision is FLAGGED and enact is BLOCKED",
          bool(basket._conflicts()) and basket._conflict.isVisibleTo(basket) and not basket._enact_btn.isEnabled(),
          f"conflicts={basket._conflicts()}")
    # remove the colliding disulfide (last row)
    basket._list.selectRow(len(basket.entries) - 1); basket._remove_selected()
    check("removing the collision clears the conflict + re-enables enact",
          not basket._conflicts() and basket._enact_btn.isEnabled())

    # 5) ENACT → one variant in the Variant Workbench ────────────────────────────────────────
    print("5) Enact → a variant appears in the Variant Workbench…")
    n_var = len(cd.variants)
    total_subs = sum(len(e["subs"]) for e in basket.entries)
    basket._enact()
    app.processEvents()
    check("enact composes ONE new variant from the basket's substitutions",
          len(cd.variants) == n_var + 1)
    v = cd.variants[-1]
    check("the variant carries all the basket substitutions (composed together)",
          len(v.mutations) == total_subs and all(m.to_aa in ("P", "C") for m in v.mutations),
          f"{[(m.resnum, m.to_aa) for m in v.mutations]}")
    check("the variant is the ACTIVE row (handed to the existing variant flow, ready to fold)",
          panel._cur_tab().active_row_id == v.id)
    check("the basket is consumed after enact", basket.entries == [])

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
