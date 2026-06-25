"""
Live-verify — CROSS-CHAIN Mode C (step 3): declare + fold an INTER-SUBUNIT disulfide against REAL
local Boltz (~/boltz_env, GPU). The correctness gate's whole point is "can we declare and fold a
cross-chain bond" — so this runs the REAL panel declare path (`build_disulfide_introduce_spec`, the
dangerous cross-chain mapping) and a REAL two-chain Boltz fold with the declared inter-chain bond.
SMALL homo-dimer (17 res × 2) — proves the path before any large dimer. No ChimeraX (the disulfide
modes read the Boltz CIF directly; viz is not under test here).

Confirms:
  1. A real Boltz dimer T-fold lands (the construct reference).
  2. The real panel declare emits a constraint with DISTINCT chains atom1:[A,…], atom2:[B,…] and an
     assembly carrying Cys on BOTH chains at the declared position (two-cd/two-member composition).
  3. Boltz ACCEPTS the cross-chain `constraints: bond` YAML (the constrained 2-chain fold succeeds).
  4. The folded assembly HAS cysteines on BOTH chains at the declared position, and the inter-chain
     SG–SG geometry is MEASURED honestly (bias, not enforcement — adopted / NOT realized).

Run: venv/Scripts/python.exe scripts/verify_disulfide_cross_chain_live.py   (two small Boltz folds — minutes)
"""
import os, sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from unittest.mock import MagicMock
from PySide6 import QtWidgets
from tool_router import ToolRouter
from session_state import SessionState
from variant_workbench import VariantWorkbenchPanel
import disulfide_geometry as dg

SEQ = "ACAYKQDGSACTWVGAA"     # 17 res; declare a SYMMETRIC inter-subunit bond A:5 ↔ B:5
POS = 5                        # author resnum 5 (K) → mutated to Cys on BOTH chains

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    r = ToolRouter(bridge=MagicMock(), session=SessionState())
    bridge = r._get_boltz_bridge()

    # 1) REAL Boltz dimer T-fold (the construct reference the variant folds against) ───────
    print("1) Real Boltz dimer fold (the construct T-fold)…")
    tf = bridge.predict([{"id": "A", "sequence": SEQ}, {"id": "B", "sequence": SEQ}],
                        seed=0, allow_remote=False)
    check("dimer T-fold succeeded", tf.get("success"), tf.get("error") or "")
    if not tf.get("success"):
        return 1

    # seed the panel design to this folded-dimer state: ONE cd across two member chains A,B with the
    # REAL T-fold cif (the construct-fold GUI flow is pre-existing/tested; step 3 is the declare+fold).
    panel._add_sequence_construct("dimer", SEQ)
    cd = next(iter(panel._design.chains.values()))
    cd.members = [("7", "A"), ("7", "B")]
    cd.rep_model, cd.rep_chain = "7", "A"
    cd.template_fold = {"engine": "boltz", "target": "assembly", "model_id": "7",
                        "cif_path": tf["cif_path"]}

    # 2) REAL cross-chain declare through the panel method (the dangerous mapping) ──────────
    print(f"2) Declare inter-chain bond A:{POS} ↔ B:{POS} (real panel declare path)…")
    spec = panel.build_disulfide_introduce_spec(
        {"chain_a": "A", "resnum_a": POS, "chain_b": "B", "resnum_b": POS})
    check("cross-chain spec built", spec is not None)
    if spec is None:
        return 1
    cons = spec["tool_inputs"]["disulfide_constraints"]
    check("constraint carries DISTINCT chains — atom1:[A,…], atom2:[B,…]",
          bool(cons) and cons[0]["atom1"][0] == "A" and cons[0]["atom2"][0] == "B", str(cons))
    chains = {c["id"]: c["sequence"] for c in spec["tool_inputs"]["chains"]}
    check("the assembly carries Cys on BOTH chains at the declared position (two-member compose)",
          chains.get("A", "")[POS - 1:POS] == "C" and chains.get("B", "")[POS - 1:POS] == "C",
          f"A[{POS}]={chains.get('A','')[POS-1:POS]} B[{POS}]={chains.get('B','')[POS-1:POS]}")

    # 3) REAL constrained 2-chain fold — Boltz must ACCEPT the cross-chain bond ─────────────
    print("3) Real Boltz fold WITH the inter-chain constraint…")
    fold = bridge.predict(
        [{"id": c["id"], "sequence": c["sequence"]} for c in spec["tool_inputs"]["chains"]],
        seed=0, allow_remote=False, constraints=cons)
    check("Boltz ACCEPTED the cross-chain constraints: bond (fold succeeded)",
          fold.get("success"), fold.get("error") or "")
    if not fold.get("success"):
        return 1

    # 4) parse the result — Cys on BOTH chains + honest inter-chain SG–SG readout ───────────
    print("4) Read the assembly — cysteines on both chains + honest geometry…")
    cys = dg.parse_cys_atoms(fold["cif_path"])
    has_both = POS in (cys.get("A") or {}) and POS in (cys.get("B") or {})
    check("the folded assembly HAS cysteines at the declared position on BOTH chains",
          has_both, f"A has {sorted(cys.get('A', {}))}, B has {sorted(cys.get('B', {}))}")
    if has_both:
        pg = dg.pair_geometry(cys["A"][POS], cys["B"][POS])
        verdict = "bond ADOPTED" if pg["bonding_compatible"] else "NOT realized in this fold"
        check("inter-chain bond geometry MEASURED honestly (bias, not enforcement)", True,
              f"inter-chain SG–SG {pg['sg_sg']} Å, χSS {pg['chi_ss']}° → {verdict}")

    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
