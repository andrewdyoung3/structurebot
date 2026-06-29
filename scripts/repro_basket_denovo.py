"""
Thread-1 controlled repro — DE-NOVO basket enact, fully instrumented, NO FIX.

Chases the de-novo-vs-loaded-PDB asymmetry: the prior clean-room repros passed on a LOADED
crystal (1C9O); the two informal failures (zero subs landed / subs on only one chain) were on a
DE-NOVO construct. This drives the REAL code paths — build_design_session_from_sequence →
construct_fold_launch_spec → apply_construct_fold_result (the member RE-POINT) → the real
_add_*_to_basket for ALL FOUR modes → the real _enact_basket — and logs, per pick:

  • the target chain id, the target cd object identity (id()), the target resnum
  • for every edit_variant call inside enact: which cd it resolved to + the resnum at that col
  • post-enact: v.mutations / v.cells per chain, and column_tracks() per-chain sub counts

ORACLE: does the (chain,cd) a pick was STAGED against equal the (chain,cd) it was ENACTED onto,
for every pick, on every run? And do picks staged BEFORE chain B's cd exists differ from after?

Synthetic, deterministic, GPU-FREE: real geometry scans need a real Boltz fold (a 3D interface);
here the fold + scan CANDIDATES are synthesized so the chain→cd RESOLUTION logic (where the two
informal failures live) is isolated and reproducible. If results vary across the 3 identical runs
that itself is the finding (a construction-order race), per the relay.

Run: venv/Scripts/python.exe scripts/repro_basket_denovo.py   (no ChimeraX, no GPU)
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

import variant_model
from variant_model import column_tracks
from session_state import SessionState
from variant_workbench import VariantWorkbenchPanel

# Two distinct de-novo sequences (lengths differ so a cross-chain resnum mix-up is visible).
SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"   # 65 aa
SEQ_B = "GSHMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQK"   # 65 aa


# ── instrumentation: tap the model-level seams without changing behaviour ──────────────
_LOG = []   # list of dicts: {event, ...}
_orig_add = variant_model.ChainDesign.add_variant
_orig_edit = variant_model.ChainDesign.edit_variant


def _tap_add(self, variant_id, **kw):
    v = _orig_add(self, variant_id, **kw)
    _LOG.append({"event": "add_variant", "cd_id": id(self), "rep_chain": self.rep_chain,
                 "members": list(self.members), "vid": variant_id})
    return v


def _tap_edit(self, variant_id, col, new_aa, **kw):
    rn = self.template_cells[col].resnum if 0 <= col < len(self.template_cells) else None
    _LOG.append({"event": "edit_variant", "cd_id": id(self), "rep_chain": self.rep_chain,
                 "vid": variant_id, "col": col, "resnum_at_col": rn, "to_aa": new_aa})
    return _orig_edit(self, variant_id, col, new_aa, **kw)


variant_model.ChainDesign.add_variant = _tap_add
variant_model.ChainDesign.edit_variant = _tap_edit


def _mk_panel():
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    panel._run_commands_bg = lambda cmds: None          # no ChimeraX
    return panel


def _synth_fold_result(fold_mid, chain_ids):
    """A construct-fold result shaped like the real bridge output: one fold step whose data
    carries the OBSERVED chain ids + per-chain pLDDT (so the read-back guard passes and the
    member re-point runs exactly as in production)."""
    plddt_by_chain = {ch: {i: 90.0 for i in range(1, 6)} for ch in chain_ids}
    return {"tool_step_results": [{
        "tool": "boltz", "success": True,
        "data": {"engine": "boltz", "new_model_id": fold_mid, "chain_ids": list(chain_ids),
                 "plddt_by_chain": plddt_by_chain, "mean_plddt": 90.0,
                 "chains_ptm": {str(i): 0.8 for i in range(len(chain_ids))},
                 "target": f"{len(chain_ids)}-chain assembly"}}]}


def _fold_construct(panel, n_copies=1):
    """Run the REAL construct fold spec + re-point. Returns the fold model id."""
    spec = panel.construct_fold_launch_spec("boltz", n_copies=n_copies)
    assert spec is not None, "construct_fold_launch_spec returned None"
    blocks = spec["_denovo_chain_blocks"]
    chain_ids = [ch for blk in blocks.values() for ch in blk]
    fold_mid = "9001"
    panel.apply_construct_fold_result(spec, _synth_fold_result(fold_mid, chain_ids))
    return fold_mid, blocks


def _chain_to_cd(panel):
    return panel._chain_to_cd()


def _cd_tag(panel, cd):
    """Stable human tag for a cd: its ukey + object id."""
    ukey = next((k for k, c in panel._design.chains.items() if c is cd), "?")
    return f"{ukey}#{id(cd)}"


def _resolve(panel, chain):
    return _chain_to_cd(panel).get(str(chain))


# ── pick synthesis (what the real scans would emit for a folded de-novo construct) ─────
def _disulfide_cand(chain, ra, rb):
    return {"chain_a": chain, "resnum_a": ra, "chain_b": chain, "resnum_b": rb,
            "best_sg_sg": 2.05, "best_chi_ss": -87.0, "clash": False, "score": 0.82}


def _proline_cand(pos, from_aa):   # no "chain" key → forces cd.rep_chain fallback (the real intra case)
    return {"position": pos, "from_aa": from_aa, "phi": -63.0, "psi": 145.0,
            "hbond_donates": False, "score": 0.71}


def _cavity_cand(chain, pos, from_aa, to_aa):
    return {"chain": chain, "position": pos, "from_aa": from_aa, "to_aa": to_aa,
            "void_volume": 42.0, "cavity_id": 1, "fill_fraction": 0.6, "clash": False, "score": 0.55}


def _saltbridge_cand(ca, ra, aa_a, ta, cb, rb, aa_b, tb):
    return {"chain_a": ca, "resnum_a": ra, "from_aa_a": aa_a, "to_aa_a": ta,
            "chain_b": cb, "resnum_b": rb, "from_aa_b": aa_b, "to_aa_b": tb,
            "best_on": 2.8, "cb_cb": 4.5, "buried": True, "clash": False, "score": 0.66}


def _stage(panel, mode, cd, cand, picks_log):
    """Stage one pick via the REAL _add_*_to_basket; log the staged (chain→cd) identity for each sub."""
    before = len(panel.design_basket.entries)
    {"disulfide": panel._add_disulfide_to_basket,
     "proline":   panel._add_proline_to_basket,
     "cavity":    panel._add_cavity_to_basket,
     "saltbridge": panel._add_saltbridge_to_basket}[mode](cd, cand)
    added = panel.design_basket.entries[before:]
    for e in added:
        for s in e["subs"]:
            staged_cd = _resolve(panel, s["chain"])
            picks_log.append({
                "mode": mode, "passed_cd": _cd_tag(panel, cd),
                "sub_chain": s["chain"], "sub_pos": s["position"], "to_aa": s["to_aa"],
                "staged_resolved_cd": _cd_tag(panel, staged_cd) if staged_cd else "UNRESOLVED",
            })
    if not added:
        picks_log.append({"mode": mode, "passed_cd": _cd_tag(panel, cd),
                          "sub_chain": "?", "note": "NOT STAGED (add rejected)"})


def _dump_design(panel):
    out = []
    for ukey, cd in panel._design.chains.items():
        cons = []
        if len(cd.variants):
            _, conservation = column_tracks(cd)
            cons = [round(c, 2) for c in conservation if c < 1.0]
        for v in cd.variants:
            out.append({
                "cd": _cd_tag(panel, cd), "rep_chain": cd.rep_chain,
                "members": list(cd.members), "vid": v.id,
                "n_mutations": len(v.mutations),
                "mutations": [(m.resnum, m.from_aa, m.to_aa) for m in v.mutations],
                "n_nontrivial_conservation_cols": len(cons),
            })
        if not cd.variants:
            out.append({"cd": _cd_tag(panel, cd), "rep_chain": cd.rep_chain,
                        "members": list(cd.members), "vid": None, "n_mutations": 0})
    return out


# ── scenarios ──────────────────────────────────────────────────────────────────────────
def scenario_hetero(run_idx):
    """Two distinct sequences → two cds (chain A=cd0, chain B=cd1, both exist at design time).
    Stage: disulfide intra-A, proline-B, cavity-B, salt-bridge interface A↔B. Enact."""
    panel = _mk_panel()
    panel._add_sequence_construct("hetero", [(SEQ_A, 1), (SEQ_B, 1)])
    fold_mid, blocks = _fold_construct(panel, n_copies=1)
    cd_map = _chain_to_cd(panel)
    cdA, cdB = cd_map["A"], cd_map["B"]
    picks = []
    _stage(panel, "disulfide", cdA, _disulfide_cand("A", 10, 24), picks)
    _stage(panel, "proline",   cdB, _proline_cand(15, SEQ_B[14]), picks)
    _stage(panel, "cavity",    cdB, _cavity_cand("B", 30, SEQ_B[29], "F"), picks)
    _stage(panel, "saltbridge", cdA,
           _saltbridge_cand("A", 40, SEQ_A[39], "E", "B", 50, SEQ_B[49], "R"), picks)
    return _run_enact(panel, picks, fold_mid, blocks, f"HETERO run{run_idx}")


def scenario_homodimer(run_idx):
    """One sequence, copies=2 → ONE cd, members A,B (both exist at design time, same cd).
    Stage: disulfide intra-A, proline-A, salt-bridge interface A↔B. Enact."""
    panel = _mk_panel()
    panel._add_sequence_construct("homodimer", [(SEQ_A, 2)])
    fold_mid, blocks = _fold_construct(panel, n_copies=1)   # copies baked into members → folds 2-chain
    cd_map = _chain_to_cd(panel)
    cdA, cdB = cd_map["A"], cd_map["B"]
    picks = []
    _stage(panel, "disulfide", cdA, _disulfide_cand("A", 10, 24), picks)
    _stage(panel, "proline",   cdA, _proline_cand(15, SEQ_A[14]), picks)
    _stage(panel, "saltbridge", cdA,
           _saltbridge_cand("A", 40, SEQ_A[39], "E", "B", 50, SEQ_A[49], "R"), picks)
    return _run_enact(panel, picks, fold_mid, blocks, f"HOMODIMER run{run_idx}")


def scenario_homo_nmer(run_idx):
    """One sequence, copies=1 → ONE cd, member A ONLY at design time. Folded as a DIMER via the
    N-mer menu (n_copies=2): chain B is BORN at the re-point. Probes the relay's 'before any chain
    B content exists' axis. Stage: salt-bridge interface A↔B + proline-A. Enact."""
    panel = _mk_panel()
    panel._add_sequence_construct("homonmer", [(SEQ_A, 1)])
    pre_members = list(next(iter(panel._design.chains.values())).members)
    fold_mid, blocks = _fold_construct(panel, n_copies=2)   # N-mer menu → chain B appears now
    cd_map = _chain_to_cd(panel)
    post_chains = sorted(cd_map.keys())
    cdA = cd_map["A"]
    cdB = cd_map.get("B")
    picks = []
    _stage(panel, "proline", cdA, _proline_cand(15, SEQ_A[14]), picks)
    if cdB is not None:
        _stage(panel, "saltbridge", cdA,
               _saltbridge_cand("A", 40, SEQ_A[39], "E", "B", 50, SEQ_A[49], "R"), picks)
    else:
        picks.append({"mode": "saltbridge", "note": "chain B has NO cd post-fold — cannot stage B"})
    res = _run_enact(panel, picks, fold_mid, blocks, f"HOMO-NMER run{run_idx}")
    res["pre_fold_members"] = pre_members
    res["post_fold_chains"] = post_chains
    return res


def scenario_homodimer_collision(run_idx):
    """THE resnum-collision probe (relay step 2). Homodimer → ONE cd, chains A+B COLLAPSED onto a
    shared template axis. Stage a chain-A-only pick and a chain-B-only pick at the SAME resnum but
    DIFFERENT to_aa, plus an interchain disulfide A↔B. Oracle: is the A/B same-resnum collision
    flagged by _conflicts? does chain A's substitution survive enact, or does chain B's overwrite it?"""
    panel = _mk_panel()
    panel._add_sequence_construct("hdcoll", [(SEQ_A, 2)])
    fold_mid, blocks = _fold_construct(panel, n_copies=1)
    cd_map = _chain_to_cd(panel)
    cdA, cdB = cd_map["A"], cd_map["B"]
    picks = []
    # chain-A pick and chain-B pick at the SAME resnum (35), different residues:
    _stage(panel, "cavity", cdA, _cavity_cand("A", 35, SEQ_A[34], "W"), picks)
    _stage(panel, "cavity", cdB, _cavity_cand("B", 35, SEQ_A[34], "Y"), picks)
    # interchain disulfide A:10 <-> B:24
    _stage(panel, "disulfide", cdA,
           {"chain_a": "A", "resnum_a": 10, "chain_b": "B", "resnum_b": 24,
            "best_sg_sg": 2.05, "best_chi_ss": -87.0, "clash": False, "score": 0.8}, picks)
    flagged = bool(panel.design_basket._conflicts())
    res = _run_enact(panel, picks, fold_mid, blocks, f"HOMODIMER-COLLISION run{run_idx}")
    res["conflict_flagged"] = flagged
    res["conflicts"] = panel.design_basket._conflicts()
    return res


def scenario_hetero_collision(run_idx):
    """Control for the collision probe: hetero → SEPARATE cds, so A:35 and B:35 are different
    template axes. Both should survive (no shared-column overwrite). Confirms the mechanism is the
    homo COLLAPSE, not same-resnum per se."""
    panel = _mk_panel()
    panel._add_sequence_construct("htcoll", [(SEQ_A, 1), (SEQ_B, 1)])
    fold_mid, blocks = _fold_construct(panel, n_copies=1)
    cd_map = _chain_to_cd(panel)
    cdA, cdB = cd_map["A"], cd_map["B"]
    picks = []
    _stage(panel, "cavity", cdA, _cavity_cand("A", 35, SEQ_A[34], "W"), picks)
    _stage(panel, "cavity", cdB, _cavity_cand("B", 35, SEQ_B[34], "Y"), picks)
    flagged = bool(panel.design_basket._conflicts())
    res = _run_enact(panel, picks, fold_mid, blocks, f"HETERO-COLLISION run{run_idx}")
    res["conflict_flagged"] = flagged
    return res


def _member_chains(panel):
    return sorted({str(ch) for cd in panel._design.chains.values() for (_m, ch) in cd.members})


def scenario_zero_landed_chase(run_idx):
    """SEPARATE chase — the first relay's OTHER failure ('zero substitutions landed'). The relay's
    hypothesis: by_cd ends up empty, or _col_for_resnum is None for every pick — a candidate carrying
    a non-member chain id, or staging before the fold registers chain membership. This probes BOTH:
      (a) stage BEFORE the construct fold (members still synthetic), fold, then enact;
      (b) a pick whose candidate chain id is NOT in the member set (the by_cd-empty path).
    Diagnosis only — logs chain-in-member-set + _col_for_resnum(input→output) per pick."""
    out = {"label": f"ZERO-LANDED-CHASE run{run_idx}", "probes": []}

    # (a) pre-fold staging then fold then enact ----------------------------------------------------
    panel = _mk_panel()
    panel._add_sequence_construct("zchase", [(SEQ_A, 1)])
    cd_pre = next(iter(panel._design.chains.values()))
    members_pre = list(cd_pre.members)
    # stage a cavity pick on chain A BEFORE folding (members are synthetic denovo-… / A)
    pre_col = panel._col_for_resnum(cd_pre, 20)
    pre_in_members = "A" in _member_chains(panel)
    panel._add_cavity_to_basket(cd_pre, _cavity_cand("A", 20, SEQ_A[19], "F"))
    n_staged_pre = len(panel.design_basket.entries)
    fold_mid, _ = _fold_construct(panel, n_copies=2)     # chain B born; A re-pointed to fold mid
    cd_post = panel._chain_to_cd().get("A")
    post_col = panel._col_for_resnum(cd_post, 20)
    post_in_members = "A" in _member_chains(panel)
    panel.design_basket._enact()
    landed = sum(len(v.mutations) for cd in panel._design.chains.values() for v in cd.variants)
    out["probes"].append({
        "probe": "(a) stage BEFORE fold, then fold, then enact",
        "members_pre_fold": members_pre, "chainA_in_members_pre": pre_in_members,
        "col_for_resnum20_pre": pre_col, "staged_pre_fold": n_staged_pre,
        "members_post_fold": _member_chains(panel), "chainA_in_members_post": post_in_members,
        "col_for_resnum20_post": post_col, "mutations_landed": landed})

    # (b) candidate carrying a NON-member chain id -------------------------------------------------
    panel2 = _mk_panel()
    panel2._add_sequence_construct("zchase2", [(SEQ_A, 1)])
    _fold_construct(panel2, n_copies=1)
    cd2 = panel2._chain_to_cd()["A"]
    members2 = _member_chains(panel2)
    # a cavity candidate that (incorrectly) carries chain 'C' — NOT a member of this construct
    panel2._add_cavity_to_basket(cd2, _cavity_cand("C", 20, SEQ_A[19], "F"))
    staged2 = len(panel2.design_basket.entries)
    bad_chain_in_members = "C" in members2
    bad_resolves = panel2._chain_to_cd().get("C") is not None
    panel2.design_basket._enact()
    landed2 = sum(len(v.mutations) for cd in panel2._design.chains.values() for v in cd.variants)
    status2 = panel2._status.text()
    out["probes"].append({
        "probe": "(b) candidate carries NON-member chain 'C'",
        "members": members2, "staged_entries": staged2,
        "bad_chain_in_members": bad_chain_in_members, "bad_chain_resolves_to_cd": bad_resolves,
        "mutations_landed": landed2, "enact_status": status2})
    return out


def _print_zero(r):
    print(f"\n{'='*78}\n{r['label']}")
    for pr in r["probes"]:
        print(f"  {pr['probe']}")
        for k, v in pr.items():
            if k != "probe":
                print(f"      {k}: {v}")


def _run_enact(panel, picks, fold_mid, blocks, label):
    _LOG.clear()
    entries_at_enact = [dict(e) for e in panel.design_basket.entries]
    # ORACLE: re-resolve each sub's chain→cd AT ENACT TIME (just before _enact_basket runs).
    enact_resolution = []
    for e in entries_at_enact:
        for s in e["subs"]:
            rc = _resolve(panel, s["chain"])
            enact_resolution.append({"mode": e["cls"], "sub_chain": s["chain"],
                                     "enact_resolved_cd": _cd_tag(panel, rc) if rc else "UNRESOLVED"})
    panel.design_basket._enact()        # the real button path → _enact_basket
    edit_calls = [d for d in _LOG if d["event"] == "edit_variant"]
    add_calls = [d for d in _LOG if d["event"] == "add_variant"]
    return {"label": label, "fold_mid": fold_mid, "blocks": blocks,
            "n_entries": len(entries_at_enact), "staged": picks,
            "enact_resolution": enact_resolution,
            "add_calls": [{"cd_id": a["cd_id"], "rep_chain": a["rep_chain"],
                           "members": a["members"], "vid": a["vid"]} for a in add_calls],
            "edit_calls": [{"cd_id": c["cd_id"], "rep_chain": c["rep_chain"], "vid": c["vid"],
                            "col": c["col"], "resnum_at_col": c["resnum_at_col"],
                            "to_aa": c["to_aa"]} for c in edit_calls],
            "design": _dump_design(panel)}


def _print_run(r):
    print(f"\n{'='*78}\n{r['label']}   fold_mid={r['fold_mid']}  blocks={r['blocks']}")
    if "pre_fold_members" in r:
        print(f"  pre-fold members:  {r['pre_fold_members']}")
        print(f"  post-fold chains:  {r['post_fold_chains']}")
    if "conflict_flagged" in r:
        print(f"  conflict-check FLAGGED the A/B collision? {r['conflict_flagged']}  "
              f"conflicts={r.get('conflicts')}")
    print(f"  basket entries at enact: {r['n_entries']}")
    print("  STAGED picks (chain → cd resolved at staging time):")
    for p in r["staged"]:
        if p.get("note"):
            print(f"    - {p['mode']:10s} {p.get('note')}")
        else:
            print(f"    - {p['mode']:10s} sub {p['sub_chain']}:{p['sub_pos']}→{p['to_aa']:1s}"
                  f"  passed_cd={p['passed_cd']}  staged→{p['staged_resolved_cd']}")
    print("  ENACT-TIME resolution (chain → cd just before _enact_basket):")
    for er in r["enact_resolution"]:
        print(f"    - {er['mode']:10s} sub {er['sub_chain']} → {er['enact_resolved_cd']}")
    print("  edit_variant calls inside enact (what was actually written):")
    for c in r["edit_calls"]:
        print(f"    - cd#{c['cd_id']} rep={c['rep_chain']} {c['vid']} col={c['col']} "
              f"resnum={c['resnum_at_col']} →{c['to_aa']}")
    print("  POST-ENACT design (variants per cd):")
    for d in r["design"]:
        print(f"    - cd={d['cd']} rep={d['rep_chain']} members={d['members']} "
              f"vid={d['vid']} n_mut={d['n_mutations']} muts={d.get('mutations')}")


def _oracle(r):
    """Does each pick's STAGED cd == its ENACT-TIME-resolved cd, and did an edit actually land
    for every staged sub? Returns (ok, notes)."""
    notes = []
    staged = {(p.get("sub_chain"), p.get("sub_pos")): p for p in r["staged"] if "sub_pos" in p}
    # map enact resolution by chain
    enact_by_chain = {}
    for er in r["enact_resolution"]:
        enact_by_chain.setdefault(er["sub_chain"], er["enact_resolved_cd"])
    ok = True
    total_staged_subs = len([p for p in r["staged"] if "sub_pos" in p])
    total_edits = len(r["edit_calls"])
    for (ch, pos), p in staged.items():
        sc = p["staged_resolved_cd"]
        ec = enact_by_chain.get(ch, "MISSING")
        if sc != ec:
            ok = False
            notes.append(f"DIVERGENCE chain {ch}: staged→{sc} but enact→{ec}")
        if sc == "UNRESOLVED" or ec == "UNRESOLVED":
            ok = False
            notes.append(f"UNRESOLVED chain {ch}")
    if total_edits != total_staged_subs:
        notes.append(f"EDIT COUNT MISMATCH: {total_staged_subs} subs staged but "
                     f"{total_edits} edit_variant calls landed")
        if total_edits < total_staged_subs:
            ok = False
    return ok, notes


def main():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scenarios = [("hetero", scenario_hetero), ("homodimer", scenario_homodimer),
                 ("homo_nmer", scenario_homo_nmer),
                 ("homodimer_collision", scenario_homodimer_collision),
                 ("hetero_collision", scenario_hetero_collision)]
    summary = []
    for name, fn in scenarios:
        runs = []
        for i in range(1, 4):
            r = fn(i)
            _print_run(r)
            ok, notes = _oracle(r)
            for n in notes:
                print(f"    ⚠ {n}")
            print(f"  ORACLE: {'MATCH (no divergence)' if ok else 'DIVERGENCE / LOSS'}")
            runs.append((ok, tuple(notes),
                         tuple((d["rep_chain"], d["vid"], d["n_mutations"]) for d in r["design"])))
        # determinism across the 3 identical runs
        oks = {x[0] for x in runs}
        shapes = {x[2] for x in runs}
        deterministic = len(shapes) == 1
        summary.append((name, oks, deterministic, runs[0][2]))
        print(f"\n  >> {name}: deterministic across 3 runs = {deterministic}; "
              f"oracle results = {oks}")
    print(f"\n{'#'*78}\nZERO-LANDED CHASE (diagnosis only — no fix)")
    _print_zero(scenario_zero_landed_chase(1))

    print(f"\n{'#'*78}\nSUMMARY")
    for name, oks, det, shape0 in summary:
        verdict = "PASS" if oks == {True} else ("FAIL" if oks == {False} else "FLAKY")
        print(f"  {name:12s} oracle={verdict:6s} deterministic={det}  shape={shape0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
