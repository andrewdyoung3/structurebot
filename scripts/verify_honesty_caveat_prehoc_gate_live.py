"""
verify_honesty_caveat_prehoc_gate_live.py — LIVE check of the §9 honesty-caveat polish.

The polish (tool_router._run_template_assist): the possible-COPYING caveat now fires on HIGH
adoption (≥0.8) ONLY when the template was NOT already close to the unguided fold — i.e. the
pre-hoc proxy structTM(template, unguided) is LOW (< 0.5). When the template was already
same-fold-close (prehoc ≥ 0.5) high adoption is the natural-success case and is NOT flagged.

This drives the REAL US-align binary (`_usalign_tm2`, LOCAL-ONLY WSL) on REAL crystal structures,
then runs the REAL `_run_template_assist` condition end-to-end. A fresh Boltz guided fold is NOT
run here: it is GPU-contended by the parallel hexamer eval AND this change consumes US-align TMs,
not fold internals — so real crystal structures stand in for the guided/unguided/template triple
(the proxy + condition are exercised for real; only the upstream folder is substituted).

Real triple (PCNA family share a fold; myoglobin 1MBN is unrelated):
  FIRE     guided=1PLQ(PCNA)  template=1AXC(PCNA)  unguided=1MBN(myoglobin)
           -> adoption HIGH (both PCNA), prehoc LOW (template not near the myoglobin "unguided") ->  fires
  SUPPRESS guided=1PLQ(PCNA)  template=1AXC(PCNA)  unguided=1VYM(PCNA)
           -> adoption HIGH,  prehoc HIGH (unguided already PCNA-like)                          -> no caveat

Floors are stubbed (no Boltz); `_usalign_tm2` and the condition run for real.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock
from tool_router import ToolRouter

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
def cif(pid): return os.path.join(CACHE, f"{pid}.cif")


def _run(guided, template, unguided):
    r = ToolRouter(bridge=MagicMock(), session=MagicMock())
    # Stub the two Boltz floors (we are not testing flexibility here, only the adoption/prehoc gate).
    r._fold_wt_reference = lambda inp: {"floor_ddm": {"1": 1.0}, "n_floor_seeds": 4}
    # Real template path (skip the download path); REAL US-align via _usalign_tm2 stays live.
    r._resolve_boltz_templates = lambda t: ([{"cif": cif(template), "chain_id": "A"}], None)
    inputs = {
        "engine": "boltz", "target": "monomer", "multichain": False, "variant_chain": "A",
        "wt_chains": [{"id": "A", "sequence": "M"}],
        "unguided_ref": {"model_id": "u", "path": cif(unguided), "seed": 0},
        "guided_ref":   {"model_id": "g", "path": cif(guided), "seed": 0},
        "templates": [{"pdb_id": template, "chain_id": "A"}],
        "guided_mean_plddt": 84.0, "unguided_mean_plddt": 62.0,
        "guided_plddt": {"1": 85.0}, "unguided_plddt": {"1": 60.0},
        "template_label": template, "force": False, "threshold": None,
    }
    res = r._run_template_assist(inputs)
    d = res.data
    pt = d["per_template"][0]
    return d, pt, res.summary


def main():
    print("== LIVE honesty-caveat pre-hoc gate (real US-align) ==\n")
    for pid in ("1PLQ", "1AXC", "1MBN", "1VYM"):
        print(f"  {pid}.cif present: {os.path.isfile(cif(pid))}")
    print()

    ok = True

    print("-- CASE 1: FIRE (template NOT already close to unguided) --")
    d, pt, summary = _run(guided="1PLQ", template="1AXC", unguided="1MBN")
    print(f"   adoption(guided 1PLQ vs template 1AXC) = {pt['adoption']}")
    print(f"   prehoc  (template 1AXC vs unguided 1MBN) = {pt['prehoc_structTM_to_unguided']}")
    print(f"   high_adoption_caveat = {d['high_adoption_caveat']}")
    fire_ok = (d["high_adoption_caveat"] is True
               and pt["adoption"] is not None and pt["adoption"] >= 0.8
               and pt["prehoc_structTM_to_unguided"] is not None
               and pt["prehoc_structTM_to_unguided"] < 0.5)
    print(f"   EXPECT fire -> {'PASS' if fire_ok else 'FAIL'}")
    print(f"   caveat in summary: {'imposing the template' in summary.lower()}\n")
    ok = ok and fire_ok

    print("-- CASE 2: SUPPRESS (template already close to unguided) --")
    d, pt, summary = _run(guided="1PLQ", template="1AXC", unguided="1VYM")
    print(f"   adoption(guided 1PLQ vs template 1AXC) = {pt['adoption']}")
    print(f"   prehoc  (template 1AXC vs unguided 1VYM) = {pt['prehoc_structTM_to_unguided']}")
    print(f"   high_adoption_caveat = {d['high_adoption_caveat']}")
    suppress_ok = (d["high_adoption_caveat"] is False
                   and pt["adoption"] is not None and pt["adoption"] >= 0.8
                   and pt["prehoc_structTM_to_unguided"] is not None
                   and pt["prehoc_structTM_to_unguided"] >= 0.5)
    print(f"   EXPECT suppress -> {'PASS' if suppress_ok else 'FAIL'}")
    print(f"   caveat absent from summary: {'imposing the template' not in summary.lower()}\n")
    ok = ok and suppress_ok

    print("== RESULT:", "ALL PASS" if ok else "FAILED", "==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
