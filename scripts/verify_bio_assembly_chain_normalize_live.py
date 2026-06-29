"""
Live-verify — FIX A: biological-assembly chain-id normalization, end-to-end through StructureBot's
REAL assembly-build path (`ToolRouter._run_bio_assembly`), against real ChimeraX :60001 (GPU-free).

The original failure (2OMF, OmpF porin — single-chain ASU, C3 biological trimer): `sym … copies true`
makes submodel-per-copy with DUPLICATE chain ids (#2.1/A, #2.2/A, #2.3/A) → native `color bychain`
paints all copies the same AND StructureBot's ingestion sees zero/one chain. Fix A normalizes to one
flat integer-id model with unique chains A,B,C.

Checks:
  1. _run_bio_assembly records a FLAT model with unique chains [A,B,C].
  2. native `color #flat bychain` → 3 DISTINCT colors.
  3. the REAL controller load_model(flat) → 3 ChainSeqs A,B,C.
  4. build_design_session → ONE ChainDesign with members A,B,C (homo-trimer); _chain_equiv = {A,B,C}.
  5. interchain disulfide scan RUNS across copies (scan_interface_sites over A/B/C).
  6. cd-equivalence (basket fix): a same-resnum pick across two copies is FLAGGED.
  7. regression: an ordinary multi-chain load (no assembly) still ingests its chains.

Run: venv/Scripts/python.exe scripts/verify_bio_assembly_chain_normalize_live.py  (needs ChimeraX :60001)
"""
import os, sys, re, tempfile
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
from seq_editor.controller import SequenceEditorController
from variant_model import build_design_session
from variant_workbench import VariantWorkbenchPanel
import disulfide_geometry as dg

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

def _models():
    return run("info models").get("value") or ""

def _new_model(before_txt):
    before = set(re.findall(r"model id #(\d+)\b", before_txt))
    after = set(re.findall(r"model id #(\d+)\b", _models()))
    new = sorted(after - before, key=int)
    return new[-1] if new else None

def _color(spec):
    v = run(f"info atoms {spec} attribute color").get("value") or ""
    m = re.search(r"color\s+([0-9,]+)", v)
    return m.group(1) if m else None


def main():
    if not bridge.is_running():
        print("ChimeraX REST not reachable on :60001."); return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run("close session")

    # open 2OMF (single-chain ASU, C3 trimer assembly) ───────────────────────────────────────────
    before = _models()
    run("open 2omf")
    au = _new_model(before)
    sess = SessionState()
    sess.add_structure(au, "2omf", metadata={"name": "2omf"})   # metadata truthy → no network
    router = ToolRouter(bridge=bridge, session=sess)

    # 1) REAL assembly-build path ─────────────────────────────────────────────────────────────────
    print("1) Build assembly via the REAL _run_bio_assembly…")
    res = router._run_bio_assembly({"model_id": au, "assembly_id": 1})
    rec = sess.get_generated_assembly(au) or {}
    flat = str(rec.get("assembly_model_id") or "")
    check("assembly built + normalized to a flat model with unique chains [A,B,C]",
          res.success and rec.get("normalized") and rec.get("assembly_chains") == ["A", "B", "C"],
          f"success={res.success} normalized={rec.get('normalized')} chains={rec.get('assembly_chains')} flat=#{flat}")

    # 2) native color bychain → 3 distinct colors ─────────────────────────────────────────────────
    print("2) Native color bychain on the flat model…")
    run(f"color #{flat} bychain")
    cols = {ch: _color(f"#{flat}/{ch}:1@CA") or _color(f"#{flat}/{ch}@CA") for ch in "ABC"}
    check("color bychain gives 3 DISTINCT colors across copies",
          len(set(v for v in cols.values() if v)) == 3, f"{cols}")

    # 3) controller load_model → 3 ChainSeqs ──────────────────────────────────────────────────────
    print("3) StructureBot ingestion (real controller.load_model)…")
    ctrl = SequenceEditorController(run_command=run, fold_fn=lambda *a, **k: {})
    cs = ctrl.load_model(flat)
    chains = sorted(c.chain for c in cs)
    check("load_model enumerates all 3 copies as chains A,B,C", chains == ["A", "B", "C"],
          f"chains={chains} (n={len(cs)})")

    # 4) build_design_session → ONE cd, members A,B,C ─────────────────────────────────────────────
    print("4) build_design_session + _chain_equiv…")
    design = build_design_session(cs, flat)
    cds = list(design.chains.values())
    cd = cds[0] if cds else None
    members = sorted(c for _m, c in cd.members) if cd else []
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    panel._run_commands_bg = lambda cmds: None
    panel._design = design
    panel._render()
    equiv = sorted(panel._chain_equiv("A"))
    check("homo-trimer collapses to ONE ChainDesign with members A,B,C",
          len(cds) == 1 and members == ["A", "B", "C"], f"n_cd={len(cds)} members={members}")
    check("_chain_equiv treats A/B/C as ONE equivalence class", equiv == ["A", "B", "C"], f"{equiv}")

    # 5) interchain disulfide scan RUNS across copies ─────────────────────────────────────────────
    print("5) Interchain disulfide scan across copies (scan_interface_sites)…")
    cif = Path(tempfile.gettempdir()) / "asm_flat.cif"
    run(f'save "{cif.as_posix()}" #{flat}')
    ran = False; n_cand = 0; spanned = set()
    try:
        atoms = dg.parse_backbone_atoms(str(cif))
        ranked, _best = dg.scan_interface_sites(atoms)
        ran = True
        n_cand = len(ranked)
        spanned = {(p["chain_a"], p["chain_b"]) for p in ranked}
    except Exception as exc:
        print(f"     scan raised: {type(exc).__name__}: {exc}")
    check("interface disulfide scan RUNS over the distinct copies (enumerates cross-chain pairs)",
          ran and set(atoms.keys()) == {"A", "B", "C"},
          f"chains_scanned={sorted(atoms.keys()) if ran else '—'} candidates={n_cand} spanning={sorted(spanned)}")

    # 6) cd-equivalence: same-resnum pick across two copies is FLAGGED ─────────────────────────────
    print("6) Basket cd-equivalence on the assembly…")
    pos = cd.template_cells[len(cd.template_cells) // 2].resnum
    wt = cd.template_cells[len(cd.template_cells) // 2].aa
    cav = lambda ch, to: {"chain": ch, "position": pos, "from_aa": wt, "to_aa": to,
                          "void_volume": 40.0, "cavity_id": 1, "fill_fraction": 0.6,
                          "clash": False, "score": 0.5}
    panel._add_cavity_to_basket(cd, cav("A", "W"))
    panel._add_cavity_to_basket(cd, cav("B", "Y"))
    app.processEvents()
    check("same-resnum pick across two copies (A:%d→W / B:%d→Y) is FLAGGED" % (pos, pos),
          bool(panel.design_basket._conflicts())
          and "equivalent copies" in panel.design_basket._conflict.text(),
          f"conflicts={panel.design_basket._conflicts()}")

    # 7) regression: ordinary multi-chain load (no assembly) unaffected ───────────────────────────
    print("7) Regression — ordinary multi-chain load (1C9O, no assembly-build)…")
    before = _models()
    run("open 1c9o")
    m2 = _new_model(before)
    cs2 = ctrl.load_model(m2)
    check("non-assembly multi-chain load still ingests its chains",
          sorted(c.chain for c in cs2) == ["A", "B"], f"chains={sorted(c.chain for c in cs2)}")

    run("close session")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
