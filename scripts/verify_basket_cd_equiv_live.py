"""
Live-verify — DESIGN BASKET cd-EQUIVALENCE keying fix, in the REAL panel + REAL ChimeraX (GPU-FREE).

The gate is the homo-oligomer collapse that produced the original failure: chains A and B share ONE
ChainDesign (homo-dimer), so two picks at the SAME resnum on the two copies are the SAME residue. We
drive the REAL picker handlers (`_add_cavity_to_basket` — the "Add to design" button path) and the
REAL basket panel widgets / workbench grid, with 3D commands routed to the live ChimeraX :60001.

Three checks (relay):
  (a) DIVERGENT  A:r→W / B:r→Y  → the basket panel itself FLAGS the conflict with the new
      "equivalent copies" message, enact is REFUSED, and NO phantom variant appears in the grid.
  (b) IDENTICAL  A:r→W / B:r→W  → dedupes to ONE entry, enacts cleanly, ONE substitution (counted
      once), and the Detail column RENDERS "applies to A, B" live.
  (c) (covered by (a)) no phantom variant in the sequence viewer after a refused enact.

Real homodimer: 1C9O (cold-shock protein, 2 identical chains A,B) → `build_design_session` collapses
A,B into one cd with members A+B. Run: venv/Scripts/python.exe scripts/verify_basket_cd_equiv_live.py
(needs ChimeraX REST :60001; NO GPU).
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
from variant_workbench import VariantWorkbenchPanel

CIF = Path(__file__).resolve().parent.parent / "cache" / "1C9O.cif"   # homodimer A,B (identical)
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _cavity(chain, pos, from_aa, to_aa):
    return {"chain": chain, "position": pos, "from_aa": from_aa, "to_aa": to_aa,
            "void_volume": 40.0, "cavity_id": 1, "fill_fraction": 0.6, "clash": False, "score": 0.5}


def main():
    if not CIF.is_file():
        print(f"cached CIF not found: {CIF}"); return 1
    if not bridge.is_running():
        print("ChimeraX REST not reachable on :60001 — open ChimeraX (or the app) first.")
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    panel._run_commands_bg = lambda cmds: [run(c) for c in cmds]    # 3D pushes hit live ChimeraX

    # open the homodimer + build the design straight from its REAL chain sequences (so A,B collapse
    # into one cd with members A+B — the failing topology — and resnums match the structure) ───────
    before = set(_model_ids())
    run(f'open "{CIF.as_posix()}"')
    mid = sorted(set(_model_ids()) - before, key=int)[-1]
    import proline_geometry as _pg
    atoms = _pg.parse_backbone_with_names(str(CIF))
    chA = sorted(atoms["A"])
    seqA = "".join(_pg._THREE_TO_ONE.get(str(atoms["A"][rn].get("resname") or ""), "A") for rn in chA)
    panel._add_sequence_construct("csp", seqA)
    cd = next(iter(panel._design.chains.values()))
    for c, rn in zip(cd.template_cells, chA):                       # real author numbering
        c.resnum = rn
    cd.members = [(mid, "A"), (mid, "B")]                           # HOMODIMER: A+B share this cd
    cd.rep_model, cd.rep_chain = mid, "A"
    cd.template_fold = {"engine": "loaded", "target": "monomer", "model_id": mid, "cif_path": str(CIF)}
    panel._from_aa_for = lambda ch, rn: seqA[chA.index(rn)] if rn in chA else None
    panel._render()
    app.processEvents()
    check("homodimer built: ONE cd, members A+B share it (homo collapse)",
          len(panel._design.chains) == 1 and {c for _m, c in cd.members} == {"A", "B"},
          f"chains={len(panel._design.chains)} members={cd.members}")
    check("_chain_equiv resolves A and B to the same equivalence class {A,B}",
          set(panel._chain_equiv("A")) == {"A", "B"} == set(panel._chain_equiv("B")))

    pos = chA[len(chA) // 2]                                        # a real interior resnum
    wt = seqA[chA.index(pos)]
    print(f"\nColliding position: {wt}{pos} on equivalent copies A and B")

    # (a) DIVERGENT — A:pos→W and B:pos→Y ─────────────────────────────────────────────────────────
    print("\n(a) Stage A:%d→W and B:%d→Y (divergent on equivalent copies)…" % (pos, pos))
    panel._add_cavity_to_basket(cd, _cavity("A", pos, wt, "W"))
    panel._add_cavity_to_basket(cd, _cavity("B", pos, wt, "Y"))
    app.processEvents()
    basket = panel.design_basket
    tab = panel._cur_tab()
    grid_variant_rows = lambda t: len({rid for rid in t._row_ids if rid and rid != "T"})
    rows_before = grid_variant_rows(tab)
    nvar_before = len(cd.variants)
    check("the basket panel FLAGS the conflict (visible) with the 'equivalent copies' message",
          bool(basket._conflicts()) and basket._conflict.isVisibleTo(basket)
          and "equivalent copies" in basket._conflict.text(),
          f"text={basket._conflict.text()!r}")
    check("enact is REFUSED (button disabled)", not basket._enact_btn.isEnabled())
    basket._enact()                                                # the real button path — must no-op
    app.processEvents()
    rows_after = grid_variant_rows(panel._cur_tab())
    check("NO phantom variant in the sequence viewer (no new grid row, no new variant)",
          len(cd.variants) == nvar_before and rows_after == rows_before == 0,
          f"variants {nvar_before}->{len(cd.variants)}, grid variant-rows {rows_before}->{rows_after}")

    # (b) IDENTICAL — A:pos→W and B:pos→W ─────────────────────────────────────────────────────────
    print("\n(b) Clear, then stage A:%d→W and B:%d→W (identical on equivalent copies)…" % (pos, pos))
    basket.reset()
    panel._add_cavity_to_basket(cd, _cavity("A", pos, wt, "W"))
    panel._add_cavity_to_basket(cd, _cavity("B", pos, wt, "W"))
    app.processEvents()
    check("identical picks DEDUPE to ONE basket entry", len(basket.entries) == 1,
          f"entries={len(basket.entries)}")
    check("no conflict; enact ENABLED", not basket._conflicts() and basket._enact_btn.isEnabled())
    detail_text = basket._list.item(0, 3).text() if basket._list.item(0, 3) else ""
    check("the basket Detail column RENDERS 'applies to A, B' live",
          "applies to A, B" in detail_text, f"detail={detail_text!r}")
    nvar_before = len(cd.variants)
    basket._enact()
    app.processEvents()
    made = len(cd.variants) - nvar_before
    v = cd.variants[-1] if cd.variants else None
    check("enacts cleanly → ONE new variant", made == 1)
    check("the substitution is counted ONCE (not twice): one mutation at the resnum",
          v is not None and [(m.resnum, m.to_aa) for m in v.mutations] == [(pos, "W")],
          f"muts={None if v is None else [(m.resnum, m.to_aa) for m in v.mutations]}")
    check("the new variant RENDERS in the sequence viewer (active row)",
          panel._cur_tab().active_row_id == (v.id if v else None))
    check("basket consumed after enact", basket.entries == [])

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
