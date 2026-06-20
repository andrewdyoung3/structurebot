"""
Live-verify Stage 2b — HETERO de-novo construct VARIANTS + DEVIATION — against REAL ChimeraX
(:60001) + the REAL local Boltz engine (~/boltz_env). Proves the cross-cd composition (a chain's
variant folds the WHOLE complex with its WT siblings), the variant-complex PARITY GUARD, and the
FLOOR-ONCE caching (the full-complex floor is folded once and SHARED by every cd — a sibling cd's
deviation does zero extra floor folds). 1AXC = PCNA×3 + p21×3 (the case the offline probe can't prove).

  0. Open 1AXC, harvest PCNA (chain A) + p21 (chain B) sequences, close it.
  1. Add sequence: a de-novo construct PCNA×3 + p21×3 (two ChainDesigns) → fold as ONE Boltz
     6-chain assembly (T-complex-fold = the displayed reference; chains A–F).
  2. PCNA substitution variant → fold:
     (a) the fold input is the FULL complex — PCNA-variant×3 + p21-WT×3 (NOT PCNA alone);
     (b) the PARITY GUARD passes — the variant complex returns the SAME chains as the T-fold.
  3. Deviation (PCNA variant):
     (c) REUSES the T-complex-fold as the reference (reference_model_id == T-fold id, NO net-new
         persistent model — no re-fold of T);
     (d-part) the floor folds the WHOLE complex N-1 times (instrumented) and is DISTRIBUTED to both
         cds (wt_refs cached on PCNA AND p21);
     (e) whole-complex deviation: the PCNA chains carry the variant's change, the WT p21 siblings
         read mostly within the floor.
  4. p21 substitution variant → fold (PCNA-WT×3 + p21-variant×3) → deviation:
     (d) does ZERO extra floor folds (reuses the shared floor) — floor folded exactly once total;
     (e) symmetric: the p21 chains carry the change, the PCNA siblings read within floor.
  5. Persist → restore: members + both cds' shared WT reference + the deviations survive.

Run: venv/Scripts/python.exe scripts/verify_workbench_denovo_hetero_deviation_live.py  (folds — many minutes)
"""
import os, sys, types, tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PySide6 import QtCore, QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from tool_router import ToolRouter
from request_engine import RequestEngine
from presenter import Presenter
from variant_workbench import VariantWorkbenchPanel, _RESULT_DEVIATION_MODE, _DDM_FLOOR_MIN_A
from config import DEVIATION_FLOOR_N

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command


def models():
    import re
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _runscript(script: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    try:
        r = run(f'runscript "{path}"')
    finally:
        try: os.unlink(path)
        except OSError: pass
    return r.get("value") or ""


def chain_sequences(model):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        "        for ch in sorted(set(r.chain_id for r in m.residues)):\n"
        "            rs = [r for r in m.residues if r.chain_id==ch and r.polymer_type==r.PT_AMINO]\n"
        "            rs.sort(key=lambda r: r.number)\n"
        "            seq = ''.join((r.one_letter_code or 'X') for r in rs)\n"
        "            if seq:\n"
        "                print('SEQ', ch, seq)\n"
        "        break\n")
    seqs = {}
    for line in out.splitlines():
        p = line.strip().split()
        if len(p) == 3 and p[0] == "SEQ":
            seqs[p[1]] = p[2]
    return seqs


def _model_attr(model, expr, tag):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        f"        print('{tag}:', {expr}); break\n")
    for line in out.splitlines():
        if line.strip().startswith(tag + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def chains_of(model):
    v = _model_attr(model, "','.join(sorted(set(r.chain_id for r in m.residues)))", "CH")
    return [c for c in v.split(",") if c]


def disrupted_by_chain(block):
    """{chain: # residues above the dRMSD floor} from a deviation block (keys are 'chain:resno')."""
    ddm = block.get("ddm") or {}
    floor = block.get("floor_ddm") or {}
    out = {}
    for k, val in ddm.items():
        ch = k.split(":")[0]
        if val > floor.get(k, _DDM_FLOOR_MIN_A):
            out[ch] = out.get(ch, 0) + 1
    return out


class ScriptedPresenter(Presenter):
    def __init__(self):
        self.warnings = []; self.confirm_calls = []; self.confirm_answer = "proceed"
    def info(self, t): pass
    def warn(self, t): self.warnings.append(t)
    def error(self, t): self.warnings.append("ERR:" + t)
    def success(self, t): pass
    def dim(self, t): pass
    def blank(self): pass
    def markup(self, t): pass
    def active_site_ok(self, m): pass
    def show_commands(self, c, e, conf): pass
    def show_tool_pipeline(self, r): pass
    def show_interface_summary(self, r): pass
    def tool_summary(self, *a, **k): pass
    def analysis_panel(self, s): pass
    def command_result(self, *a, **k): pass
    def blocked(self, cmd, error): pass
    def completed(self, n): pass
    def translation_declined(self, exc): pass
    def translation_error(self, exc): pass
    def status(self, label):
        class _CM:
            def __enter__(s): return s
            def __exit__(s, *a): return False
        return _CM()
    def running_tools(self, label, eta_s=0.0, needs_timer=False): return self.status(label)
    def tool_status(self, m): pass
    def ask_clarification(self, q): return ""
    def confirm(self, confidence):
        self.confirm_calls.append(confidence); return self.confirm_answer
    def ask_edit(self, original): return list(original)
    def ask_yes_no(self, q, default="y"): return False


# ── 0) harvest realistic PCNA + p21 sequences from 1AXC, then close it ────────────────
before = set(models())
run("open 1axc")
AXC = (sorted(set(models()) - before, key=int) or ["1"])[-1]
seqs = chain_sequences(AXC)
print(f"[harvest] 1AXC #{AXC} chains={list(seqs)}; lengths={{k: len(v) for k, v in seqs.items()}}")
run(f"close #{AXC}")
by_len = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)
pcna = seqs.get("A") or (by_len[0][1] if by_len else "")
p21 = seqs.get("B") or (by_len[-1][1] if by_len else "")
if not pcna or not p21 or pcna == p21:
    print(f"[abort] could not harvest two distinct sequences from 1AXC: {seqs}"); sys.exit(2)
print(f"[harvest] PCNA={len(pcna)} aa, p21={len(p21)} aa")

# ── build the panel + spine ───────────────────────────────────────────────────────────
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
router = ToolRouter(bridge, session)
host = types.SimpleNamespace(
    bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None, _log_exchange=lambda *a, **k: None)
engine = RequestEngine(host)
pres = ScriptedPresenter()
panel = VariantWorkbenchPanel(ctrl, session=session)

if not panel._fold_engine_availability().get("boltz"):
    print("[abort] Boltz env (~/boltz_env) unavailable — Stage 2b hetero deviation needs it.")
    sys.exit(2)


def drive(spec, on_apply, timeout=3_600_000):
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
    captured = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if captured:
        on_apply(captured[0])
    return captured


def deviate_counting(spec, on_apply):
    """Run a deviation while counting the Boltz floor folds it performs (predict seeds)."""
    bb = router._get_boltz_bridge()
    orig = bb.predict
    seeds = []
    def counting(*a, **k):
        seeds.append(k.get("seed"))
        return orig(*a, **k)
    bb.predict = counting
    pre = set(models())
    try:
        drive(spec, on_apply)
    finally:
        bb.predict = orig
    return seeds, (set(models()) - pre)


checks = []

# ── 1) Add PCNA×3 + p21×3 → fold the construct (6-chain Boltz assembly) ────────────────
panel._add_sequence_construct("1axc_denovo", [(pcna, 3), (p21, 3)])
cds = list(panel._design.chains.values())
print(f"[construct] {len(cds)} ChainDesigns; members={[[c for _m, c in cd.members] for cd in cds]}")
print("[construct] folding PCNA×3 + p21×3 as ONE Boltz 6-chain assembly — many minutes…")
cspec = panel.construct_fold_launch_spec("boltz", 1)
drive(cspec, lambda r: panel.apply_construct_fold_result(cspec, r))
t_mid = cds[0].template_fold.get("model_id")
t_chains = chains_of(t_mid) if t_mid else []
print(f"[construct] T-complex-fold #{t_mid} chains={t_chains}; "
      f"PCNA members={cds[0].members}; p21 members={cds[1].members}")
checks.append(("construct folded as a 6-chain assembly (A–F)",
               bool(t_mid) and len(t_chains) == 6))

# ── 2) PCNA substitution variant → fold the FULL complex + parity guard ───────────────
panel._tabs.setCurrentIndex(0)                                   # PCNA tab
panel._add_variant()
cd_pcna = panel._cur_tab().design
vp = cd_pcna.variants[-1]
panel._cur_tab().set_active_row(vp.id)
cd_pcna.edit_variant(vp.id, 0, "A" if vp.cells[0].aa != "A" else "S")
fspec = panel.fold_launch_spec("boltz")
fchains = fspec["tool_inputs"].get("chains") or []
pcna_ids = [c for _m, c in cd_pcna.members]
p21_ids = [c for _m, c in cds[1].members]
pcna_seqs = {c["sequence"] for c in fchains if c["id"] in pcna_ids}
p21_seqs = {c["sequence"] for c in fchains if c["id"] in p21_ids}
p21_t = "".join(x.aa for x in cds[1].template_cells)
print(f"[pcna-var] fold spec chains={[c['id'] for c in fchains]}; "
      f"PCNA seqs={[len(s) for s in pcna_seqs]} (variant), p21 seqs all WT={p21_seqs == {p21_t}}")
checks.append(("(a) PCNA variant fold input is the FULL complex (6 chains A–F)",
               [c["id"] for c in fchains] == t_chains))
checks.append(("(a) the PCNA chains carry the VARIANT seq, the p21 chains the WT seq",
               pcna_seqs == {vp.sequence} and p21_seqs == {p21_t} and vp.sequence != p21_t))

print("[pcna-var] folding the PCNA-variant complex (PCNA-variant×3 + p21-WT×3) — many minutes…")
drive(fspec, lambda r: panel.apply_fold_result(vp.id, r))
vp_fold = cd_pcna.get_variant(vp.id).results.fold or {}
vp_mid = vp_fold.get("model_id")
vp_chains = chains_of(vp_mid) if vp_mid else []
print(f"[pcna-var] variant fold #{vp_mid} chains={vp_chains}; status={panel._status.text()[:80]!r}")
checks.append(("(b) PARITY GUARD passed — variant complex stored (not refused)",
               bool(vp_mid) and "mismatch" not in panel._status.text().lower()))
checks.append(("(b) variant-fold chains == T-fold chains",
               sorted(vp_chains) == sorted(t_chains) and vp_fold.get("target") == "assembly"))

# ── 3) Deviation (PCNA variant): reuse the T-fold + floor-once (instrumented) ──────────
dspec = panel.deviation_launch_spec()
print(f"[pcna-dev] confidence={dspec['confidence']} wt_ref={dspec['tool_inputs']['wt_ref'].get('model_id')} "
      f"(T-fold #{t_mid}); wt_chains={len(dspec['tool_inputs']['wt_chains'])}")
checks.append(("deviation pre-seeds the T-complex-fold as the reference (reuse)",
               dspec["tool_inputs"]["wt_ref"].get("model_id") == t_mid))
checks.append(("deviation WT reference is the FULL complex (6 wt_chains)",
               len(dspec["tool_inputs"]["wt_chains"]) == 6))
print(f"[pcna-dev] deviation — reuse T-fold (seed-0) + fold the {DEVIATION_FLOOR_N-1} floor seeds "
      f"over the WHOLE complex — many minutes…")
seeds_pcna, new_models_pcna = deviate_counting(dspec, lambda r: panel.apply_deviation_result(vp.id, r))
bp = (cd_pcna.get_variant(vp.id).results.fold or {}).get("deviation") or {}
dbc_pcna = disrupted_by_chain(bp)
print(f"[pcna-dev] floor seeds folded={seeds_pcna}; ref={bp.get('reference_model_id')} (T #{t_mid}); "
      f"net-new persistent models={new_models_pcna}; floor_kind={bp.get('floor_kind')}; "
      f"disrupted/chain={dbc_pcna}")
checks.append(("(c) reference IS the reused T-complex-fold (no re-fold of T)",
               str(bp.get("reference_model_id")) == str(t_mid)))
checks.append(("(c) NO net-new persistent model from the deviation",
               len(new_models_pcna) == 0))
checks.append((f"(d) the floor folded the WHOLE complex N-1={DEVIATION_FLOOR_N-1} times",
               len([s for s in seeds_pcna if s is not None]) == DEVIATION_FLOOR_N - 1))
# floor-once cache distributed to BOTH cds
checks.append(("(d) the floor wt_ref is distributed to BOTH cds (PCNA + p21 cached)",
               bool(cds[0].wt_refs.get("boltz:assembly"))
               and bool(cds[1].wt_refs.get("boltz:assembly"))))
# (e) whole-complex: the PCNA chains carry the change, p21 siblings read mostly within floor
pcna_dis = sum(dbc_pcna.get(c, 0) for c in pcna_ids)
p21_dis = sum(dbc_pcna.get(c, 0) for c in p21_ids)
print(f"[pcna-dev] (e) disrupted: PCNA chains={pcna_dis}, p21 siblings={p21_dis}")
checks.append(("(e) the PCNA (varying) chains carry the change; p21 siblings carry less",
               pcna_dis > 0 and pcna_dis > p21_dis))

# ── 4) p21 substitution variant → fold + deviation (reuses the floor: ZERO floor folds) ─
panel._tabs.setCurrentIndex(1)                                   # p21 tab
panel._add_variant()
cd_p21 = panel._cur_tab().design
vq = cd_p21.variants[-1]
panel._cur_tab().set_active_row(vq.id)
cd_p21.edit_variant(vq.id, 0, "A" if vq.cells[0].aa != "A" else "S")
fspec2 = panel.fold_launch_spec("boltz")
fchains2 = fspec2["tool_inputs"].get("chains") or []
pcna_t = "".join(x.aa for x in cds[0].template_cells)
print(f"[p21-var] fold spec chains={[c['id'] for c in fchains2]}; "
      f"PCNA all WT={ {c['sequence'] for c in fchains2 if c['id'] in pcna_ids} == {pcna_t} }")
checks.append(("p21 variant fold input is the FULL complex (PCNA-WT×3 + p21-variant×3)",
               [c["id"] for c in fchains2] == t_chains
               and {c["sequence"] for c in fchains2 if c["id"] in pcna_ids} == {pcna_t}
               and {c["sequence"] for c in fchains2 if c["id"] in p21_ids} == {vq.sequence}))
print("[p21-var] folding the p21-variant complex — many minutes…")
drive(fspec2, lambda r: panel.apply_fold_result(vq.id, r))
vq_fold = cd_p21.get_variant(vq.id).results.fold or {}
vq_mid = vq_fold.get("model_id")
checks.append(("p21 variant parity guard passed",
               bool(vq_mid) and "mismatch" not in panel._status.text().lower()))

dspec2 = panel.deviation_launch_spec()
print(f"[p21-dev] confidence={dspec2['confidence']} wt_ref floor_present="
      f"{bool(dspec2['tool_inputs']['wt_ref'].get('floor_ddm'))}")
checks.append(("(d) p21 deviation reuses the SHARED floor (high-confidence, no gate)",
               dspec2["confidence"] == "high"
               and bool(dspec2["tool_inputs"]["wt_ref"].get("floor_ddm"))))
print("[p21-dev] deviation — must do ZERO floor folds (floor already established) …")
seeds_p21, new_models_p21 = deviate_counting(dspec2, lambda r: panel.apply_deviation_result(vq.id, r))
bq = (cd_p21.get_variant(vq.id).results.fold or {}).get("deviation") or {}
dbc_p21 = disrupted_by_chain(bq)
print(f"[p21-dev] floor seeds folded={seeds_p21} (expect none); disrupted/chain={dbc_p21}")
checks.append(("(d) the second cd's deviation did ZERO extra floor folds (floor folded once total)",
               len([s for s in seeds_p21 if s is not None]) == 0))
pcna_dis2 = sum(dbc_p21.get(c, 0) for c in pcna_ids)
p21_dis2 = sum(dbc_p21.get(c, 0) for c in p21_ids)
print(f"[p21-dev] (e) disrupted: p21 chains={p21_dis2}, PCNA siblings={pcna_dis2}")
checks.append(("(e) symmetric — the p21 (varying) chains carry the change; PCNA siblings carry less",
               p21_dis2 > 0 and p21_dis2 > pcna_dis2))

# ── 5) persist → restore ──────────────────────────────────────────────────────────────
dd = session.get_design_session(panel._design.model_id)
panel_r = VariantWorkbenchPanel(ctrl, session=session)
panel_r.rehydrate_denovo(dd)
cds_r = list(panel_r._design.chains.values())
vp_r = cds_r[0].get_variant(vp.id)
vq_r = cds_r[1].get_variant(vq.id)
print(f"[restore] PCNA members={cds_r[0].members}; wt_refs PCNA={list(cds_r[0].wt_refs)} "
      f"p21={list(cds_r[1].wt_refs)}")
checks.append(("restore: members still point at the T-fold chains",
               cds_r[0].rep_model == t_mid and len(cds_r[0].members) == 3))
checks.append(("restore: the SHARED WT reference + both deviations survive",
               bool(cds_r[0].wt_refs.get("boltz:assembly"))
               and bool(cds_r[1].wt_refs.get("boltz:assembly"))
               and bool((vp_r.results.fold or {}).get("deviation"))
               and bool((vq_r.results.fold or {}).get("deviation"))))

QtCore.QThreadPool.globalInstance().waitForDone(5000)
print("[cleanup] leaving models open for inspection")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
