"""
Live-verify the HETERO multi-chain de-novo construct against REAL ChimeraX (:60001) + the
REAL ~/boltz_env WSL GPU env. Proves the case the offline probe could not: a construct of
TWO distinct sequences (1AXC = PCNA + p21) folded as ONE 6-chain Boltz assembly, with each
ChainDesign re-pointed to its OWN three fold chains.

Flow (no crystal seeds the design — sequences are READ from 1AXC only to get realistic input):
  0. Open 1AXC just to harvest the PCNA (chain A) + p21 (chain B) sequences, then CLOSE it.
  1. Add sequence: a de-novo construct with PCNA×3 + p21×3 (two ChainDesigns, no structure).
  2. Fold construct → Boltz → ONE 6-chain assembly through the spine (confirm-gate at 'low').
  3. READ-BACK GUARD held: sent chain ids == observed CIF ids (the fold is trusted).
  4. PER-TAB RE-POINT: the PCNA tab's members are its three fold chains, p21's its three —
     a column-click on a PCNA position 3D-selects the three PCNA chains, a p21 position the p21.
  5. PER-CHAIN pLDDT distinct: each tab's pLDDT badge is ITS chain's value (PCNA ≠ p21).
  6. NO spurious matchmaker (de novo, no_reference → nothing to superpose onto).
  7. PERSIST → RESTORE intact (rehydrate_denovo; members + per-chain folds survive).

Run: venv/Scripts/python.exe scripts/verify_workbench_denovo_hetero_live.py   (Boltz folds — minutes)
"""
import os, sys, tempfile, types
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
from variant_workbench import VariantWorkbenchPanel

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
    """{chain_id: one-letter sequence} for the amino-acid residues of *model* (author order)."""
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


def residue_count(model):
    v = _model_attr(model, "m.num_residues", "NRES")
    return int(v) if v.lstrip("-").isdigit() else 0


def selected_chains(model):
    """Chain ids on *model* that currently have ≥1 selected atom (the column-click readback)."""
    v = _model_attr(model, "','.join(sorted(set(a.residue.chain_id for a in m.atoms if a.selected)))", "SEL")
    return [c for c in v.split(",") if c]


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
# PCNA = chain A (long); p21 = chain B (short peptide). Fall back to length sort if absent.
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

checks = []
checks.append(("engine.router IS the production ToolRouter", isinstance(engine.router, ToolRouter)))
checks.append(("Boltz picker enabled (capability flag)",
               panel._fold_engine_availability().get("boltz") is True))

# ── 1) Add sequence: PCNA×3 + p21×3 (two ChainDesigns, no structure) ──────────────────
panel._add_sequence_construct("1axc_denovo", [(pcna, 3), (p21, 3)])
cds = list(panel._design.chains.values())
syn = panel._design.model_id
print(f"[add] de-novo construct {syn}: {len(cds)} ChainDesigns; "
      f"members={[[c for _m, c in cd.members] for cd in cds]}")
checks.append(("two ChainDesigns (PCNA + p21)", len(cds) == 2))
checks.append(("grouped contiguous ids (A,B,C | D,E,F)",
               [c for _m, c in cds[0].members] == ["A", "B", "C"]
               and [c for _m, c in cds[1].members] == ["D", "E", "F"]))
checks.append(("nothing in ChimeraX yet (synthetic, inert)",
               all(m == syn for cd in cds for m, _c in cd.members)))

# ── 2) Fold construct → Boltz 6-chain assembly through the spine ──────────────────────
spec = panel.construct_fold_launch_spec("boltz", 1)
checks.append(("ESMFold refused for the hetero assembly",
               panel.construct_fold_launch_spec("esmfold", 1) is None))
checks.append(("spec is a 6-chain assembly, LOCAL-ONLY, no_reference",
               len(spec["tool_inputs"].get("chains") or []) == 6
               and spec["tool_inputs"].get("local_only") is True
               and spec["tool_inputs"].get("no_reference") is True))
print("[fold] Boltz 6-chain assembly fold (LOCAL-ONLY, no reference) — minutes…")
pres.confirm_calls.clear()
captured = []
bm = set(models())
engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                           pres, confidence=spec.get("confidence", "high"),
                           on_result=captured.append)
QtCore.QThreadPool.globalInstance().waitForDone(1800000)   # up to 30 min (6-chain ~800-res fold)
checks.append(("fold confidence='low' → confirm-gate asked", pres.confirm_calls == ["low"]))
checks.append(("fold ran through the spine (on_result fired)", bool(captured)))
if not captured:
    print(f"[abort] no fold result — warnings={pres.warnings}"); sys.exit(2)
panel.apply_construct_fold_result(spec, captured[0])
new_models = sorted(set(models()) - bm, key=int)
fold_mid = cds[0].rep_model
print(f"[fold] new models={new_models}; fold model #{fold_mid} "
      f"chains={chains_of(fold_mid)} residues={residue_count(fold_mid)}")

# ── 3) read-back guard held (no mismatch in the status) ───────────────────────────────
checks.append(("read-back guard passed (sent == observed; status not a mismatch)",
               "mismatch" not in panel._status.text().lower()))
data = panel._fold_from_result(captured[0])
sent = sorted({ch for blk in spec["_denovo_chain_blocks"].values() for ch in blk})
observed = sorted(map(str, data.get("chain_ids") or []))
print(f"[guard] sent={sent} observed={observed}")
checks.append(("observed CIF chain ids == sent ids", observed == sent))

# ── 4) ONE 6-chain model + per-tab re-point lands on the right chains ─────────────────
checks.append(("ONE fold model opened", len(new_models) == 1 and fold_mid in new_models))
checks.append(("fold model is a 6-chain assembly", len(chains_of(fold_mid)) == 6))
checks.append(("both tabs re-point to the SAME fold model",
               cds[0].rep_model == fold_mid and cds[1].rep_model == fold_mid))
checks.append(("PCNA tab → its 3 fold chains, p21 tab → its 3",
               [c for _m, c in cds[0].members] == ["A", "B", "C"]
               and [c for _m, c in cds[1].members] == ["D", "E", "F"]))

# column-click on a PCNA position selects the three PCNA chains; a p21 position the p21 three
specs_pcna = panel.select_specs_for_column(cds[0], 0)
specs_p21 = panel.select_specs_for_column(cds[1], 0)
print(f"[select] PCNA col0 specs={specs_pcna}")
print(f"[select] p21  col0 specs={specs_p21}")
checks.append(("PCNA column select targets A,B,C on the fold",
               sorted(c for _m, c, _r in specs_pcna) == ["A", "B", "C"]
               and all(m == fold_mid for m, _c, _r in specs_pcna)))
checks.append(("p21 column select targets D,E,F on the fold",
               sorted(c for _m, c, _r in specs_p21) == ["D", "E", "F"]
               and all(m == fold_mid for m, _c, _r in specs_p21)))
# prove it in the REAL 3D via the PRODUCTION select path (all copies in one go)
run("select clear")
ctrl.select_residues_multi(specs_pcna)
sel = selected_chains(fold_mid)
print(f"[select] live-selected chains after PCNA col0 = {sel}")
checks.append(("3D select on a PCNA position lit ONLY PCNA chains (A,B,C)",
               sel == ["A", "B", "C"]))
run("select clear")

# ── 5) per-chain pLDDT distinct (PCNA ≠ p21) ─────────────────────────────────────────
mp_pcna = cds[0].template_fold.get("mean_plddt")
mp_p21 = cds[1].template_fold.get("mean_plddt")
print(f"[plddt] PCNA mean pLDDT={mp_pcna}  p21 mean pLDDT={mp_p21}")
checks.append(("each tab has its OWN per-chain pLDDT",
               isinstance(mp_pcna, (int, float)) and isinstance(mp_p21, (int, float))))
checks.append(("per-chain pLDDT is DISTINCT (PCNA ≠ p21)", mp_pcna != mp_p21))

# ── 6) no spurious matchmaker (de novo) ───────────────────────────────────────────────
ref = cds[0].template_fold.get("reference_model_id")
print(f"[matchmaker] reference_model_id = {ref!r}; iptm = {cds[0].template_fold.get('iptm')}")
checks.append(("no reference fold target (de novo, no matchmaker)",
               ref in (None, "None", "")))

# ── 7) persist → restore intact ───────────────────────────────────────────────────────
dd = session.get_design_session(syn)
panel2 = VariantWorkbenchPanel(ctrl, session=session)
panel2.rehydrate_denovo(dd)
cds2 = list(panel2._design.chains.values())
print(f"[restore] members={[[c for _m, c in cd.members] for cd in cds2]}; "
      f"pLDDT={[cd.template_fold.get('mean_plddt') for cd in cds2]}")
checks.append(("restore: two ChainDesigns survive", len(cds2) == 2))
checks.append(("restore: members still point at the fold model chains",
               cds2[0].rep_model == fold_mid
               and [c for _m, c in cds2[0].members] == ["A", "B", "C"]
               and [c for _m, c in cds2[1].members] == ["D", "E", "F"]))
checks.append(("restore: per-chain pLDDT intact + distinct",
               cds2[0].template_fold.get("mean_plddt") == mp_pcna
               and cds2[1].template_fold.get("mean_plddt") == mp_p21))

# ── cleanup ───────────────────────────────────────────────────────────────────────────
QtCore.QThreadPool.globalInstance().waitForDone(5000)
for m in {fold_mid} | set(new_models):
    if m:
        run(f"close #{m}")
print("[cleanup] closed the fold model")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
