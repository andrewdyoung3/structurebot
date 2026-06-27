"""
scripts/verify_saltbridge_live.py
---------------------------------
LIVE pipeline verification for the salt-bridge stabilization mode — runs the REAL scan on a REAL
cached structure (REAL FreeSASA burial, REAL rotamer-aware reach + clash), then exercises the
workbench surfaces that feed the panel + ChimeraX: the heatmap panel-hex map, the per-residue 3D
colour commands, and the pair-glow highlight commands (both members referenced). No GPU, no Boltz.

The GATE: sensible existing-pair measurements + a plausible (not zero / not everything) novel
shortlist + the honest desolvation framing present, and well-formed ChimeraX command strings.

Run:  python scripts/verify_saltbridge_live.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

_CACHE = Path(__file__).parent.parent / "cache"


def _pick_cif():
    for name in ("1C9O.cif", "1BOV.cif", "1A2W.cif", "1AXC.cif"):
        p = _CACHE / name
        if p.exists():
            return str(p)
    return None


def main() -> int:
    cif = _pick_cif()
    if cif is None:
        print("SKIP — no cached CIF in ./cache to verify against.")
        return 0
    print(f"=== Salt-bridge live verify on {Path(cif).name} ===\n")

    # 1) the REAL router scan (existing + novel, intra + inter, real FreeSASA burial)
    from tool_router import ToolRouter
    from unittest.mock import MagicMock
    from session_state import SessionState
    r = ToolRouter(bridge=MagicMock(), session=SessionState())
    out = r._run_saltbridge_scan({"cif_path": cif})
    assert out.success, "scan failed"
    d = out.data
    existing, novel = d["existing"], d["novel"]
    print(f"[A] existing salt bridges: {len(existing)}")
    for p in existing[:3]:
        print(f"    {p['type']} {p['chain_a']}:{p['resnum_a']}<->{p['chain_b']}:{p['resnum_b']} "
              f"O-N {p['on_dist']}A {'H-bond ' if p['hbond_like'] else ''}"
              f"{'buried' if p['buried'] else 'surface' if p['buried'] is not None else 'burial?'}"
              f"{' OPTIMIZABLE' if p['optimizable'] else ''} score {p['score']}")
    print(f"[B] novel candidates: {len(novel)} (of {d['n_novel_total']} total), burial_available={d['burial_available']}")
    for c in novel[:3]:
        print(f"    {c['from_aa_a']}{c['resnum_a']}{c['to_aa_a']}+{c['from_aa_b']}{c['resnum_b']}{c['to_aa_b']} "
              f"({c['chain_a']}:{c['resnum_a']}<->{c['chain_b']}:{c['resnum_b']}) O-N {c['best_on']}A "
              f"Cb-Cb {c['cb_cb']}A clash={c['clash']} score {c['score']}")

    # GATE: sensible counts (not zero, not everything)
    assert novel, "no novel candidates — the engineering mode surfaced nothing"
    assert len(novel) <= r.SB_NOVEL_TOP_N, "novel list not capped"
    for p in existing:
        assert p["on_dist"] <= 5.0, "an existing pair beyond the shoulder leaked"
    assert "desolvation" in d["caveat"].lower(), "the desolvation framing is missing from the caveat"
    print("\n[C] desolvation caveat present:", '"' + d["caveat"][:90] + '..."')

    # 2) the heatmap colour ramp (panel-hex map the workbench paints)
    from color_modes import saltbridge_compat_color
    best = d["best_partner"]
    rep = next(iter(best))
    hexmap = {int(rn): saltbridge_compat_color(sc) for rn, sc in best[rep].items()}
    hexmap = {k: v for k, v in hexmap.items() if v}
    assert hexmap, "heatmap produced no colours"
    print(f"\n[D] heatmap: {len(hexmap)} residues coloured on chain {rep}; sample "
          f"{list(hexmap.items())[:3]}")

    # 3) the pair-glow ChimeraX commands (both members referenced) — reuses the disulfide seam
    from variant_workbench import VariantWorkbenchPanel
    cand = novel[0]
    stub_cd = SimpleNamespace(template_fold={"model_id": "1"}, rep_model="1", rep_chain=rep)
    cmds = VariantWorkbenchPanel._disulfide_scan_highlight_commands(stub_cd, cand)
    assert cmds, "no glow commands generated"
    joined = " ".join(cmds)
    assert str(cand["resnum_a"]) in joined and str(cand["resnum_b"]) in joined, \
        "glow commands don't reference both pair members"
    print(f"\n[E] pair-glow ChimeraX commands ({len(cmds)}) reference both members "
          f"{cand['resnum_a']} & {cand['resnum_b']}:")
    for c in cmds[:4]:
        print("    " + c)

    print("\n=== PASS — salt-bridge live pipeline verified (real scan + burial + heatmap + glow) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
