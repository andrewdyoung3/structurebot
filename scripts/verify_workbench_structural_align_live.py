"""
Live-verify Stage 3 — sequence-INDEPENDENT structural alignment (US-align) — against REAL
ChimeraX (:60001) + the REAL local Boltz fold + the REAL US-align binary (~/USalign/USalign,
LOCAL-ONLY). A de-novo construct's FOLD is structurally aligned onto a chosen PDB regardless of
sequence — the case ChimeraX matchmaker can't reach (it is sequence-guided, fails closed at zero
homology). Monomer.

  0. Harvest the myoglobin (1MBN) sequence, build a de-novo construct, fold it (Boltz monomer).
  1. HOMOLOG (4HHB, hemoglobin α, ~27% id): Align to PDB → US-align TM>0.5, RMSD/n_aligned
     surface; option-B overlay APPLIED (view matrix on the fold model) → the fold sits ON the
     reference (COM proximity confirms the transform reproduced the superposition).
  2. DISTANT, sequence-divergent (1LH1, leghemoglobin, ~15% id): meaningful TM>0.5 DESPITE low
     identity — the whole premise (matchmaker's sequence-guided path is unreliable here).
  3. UNRELATED (1UBQ, ubiquitin): low TM (<0.5) → honest "NOT structurally similar".
  4. HEAD-TO-HEAD: matchmaker vs US-align on the distant + unrelated references — where matchmaker
     STRUGGLES / fails closed and US-align is confident (or honestly low).

Run: venv/Scripts/python.exe scripts/verify_workbench_structural_align_live.py  (one Boltz fold — minutes)
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
        return run(f'runscript "{path}"').get("value") or ""
    finally:
        try: os.unlink(path)
        except OSError: pass


def chain_sequences(model):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        "        for ch in sorted(set(r.chain_id for r in m.residues)):\n"
        "            rs = [r for r in m.residues if r.chain_id==ch and r.polymer_type==r.PT_AMINO]\n"
        "            rs.sort(key=lambda r: r.number)\n"
        "            seq = ''.join((r.one_letter_code or 'X') for r in rs)\n"
        "            if seq: print('SEQ', ch, seq)\n"
        "        break\n")
    seqs = {}
    for line in out.splitlines():
        p = line.strip().split()
        if len(p) == 3 and p[0] == "SEQ":
            seqs[p[1]] = p[2]
    return seqs


def com_distance(m1, m2):
    """Distance between the CA centroids of two models (Å) — superposition proximity proxy."""
    script = (
        "from chimerax.atomic import all_atomic_structures\n"
        "import numpy as np\n"
        "c={}\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string in ('{m1}','{m2}'):\n"
        "        cas=[a.scene_coord for a in m.atoms if a.name=='CA']\n"
        "        if cas: c[m.id_string]=np.mean(np.array(cas),axis=0)\n"
        f"if '{m1}' in c and '{m2}' in c: print('COM:', float(np.linalg.norm(c['{m1}']-c['{m2}'])))\n")
    for line in _runscript(script).splitlines():
        if line.strip().startswith("COM:"):
            try: return float(line.split(":")[1])
            except ValueError: return None
    return None


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


# ── harvest the myoglobin sequence ────────────────────────────────────────────────────
run("close session")
before = set(models())
run("open 1mbn")
MBN = (sorted(set(models()) - before, key=int) or ["1"])[-1]
myo = chain_sequences(MBN).get("A", "")
run(f"close #{MBN}")
if not myo:
    print("[abort] could not harvest the myoglobin sequence from 1MBN"); sys.exit(2)
print(f"[harvest] myoglobin {len(myo)} aa")

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
    print("[abort] Boltz env unavailable — the construct fold needs it."); sys.exit(2)


def drive(spec, on_apply, timeout=1_800_000):
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
    captured = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if captured:
        on_apply(captured[0])
    return captured


checks = []

# ── 0) fold the myoglobin construct (Boltz monomer) ───────────────────────────────────
panel._add_sequence_construct("myoglobin", myo)
cd = next(iter(panel._design.chains.values()))
print("[fold] folding the myoglobin construct as a Boltz monomer — minutes…")
cspec = panel.construct_fold_launch_spec("boltz", 1)
drive(cspec, lambda r: panel.apply_construct_fold_result(cspec, r))
fold_mid = cd.template_fold.get("model_id")
cif = cd.template_fold.get("cif_path") or cd.template_fold.get("pdb_path")
print(f"[fold] construct fold #{fold_mid}; cif={cif}")
checks.append(("construct folded (model + cif on disk)", bool(fold_mid) and bool(cif)))


def do_align(pdb_id):
    spec = panel.structural_align_launch_spec(reference_pdb_id=pdb_id)
    drive(spec, lambda r: panel.apply_structural_align_result(spec, r))
    return dict(cd.structural_align)


def matchmaker_headtohead(ref_id, ref_model_id):
    """Run ChimeraX matchmaker: the construct fold → the reference. Capture its summary line."""
    out = run(f"matchmaker #{fold_mid} to #{ref_model_id}").get("value") or ""
    line = ""
    for ln in out.splitlines():
        low = ln.lower()
        if "rmsd between" in low or "cannot match" in low or "fewer than" in low \
                or "alignment score" in low:
            line += ln.strip() + " | "
    return line or out[:160]


# ── 1) HOMOLOG: 4HHB (hemoglobin α) ───────────────────────────────────────────────────
print("[align] HOMOLOG → 4HHB (hemoglobin α) …")
a1 = do_align("4HHB")
ref1 = a1.get("reference_model_id")
com1 = com_distance(fold_mid, ref1) if ref1 else None
print(f"[align] 4HHB: TM_ref={a1.get('tm_ref')} TM_query={a1.get('tm_query')} RMSD={a1.get('rmsd')} "
      f"n_aligned={a1.get('n_aligned')} shared={a1.get('shared_fold')} | overlay COM dist={com1}")
checks.append(("(1) homolog 4HHB → TM_ref > 0.5 (shared fold)", (a1.get("tm_ref") or 0) > 0.5))
checks.append(("(1) US-align RMSD + n_aligned captured", a1.get("rmsd") is not None and (a1.get("n_aligned") or 0) > 50))
checks.append(("(1) option-B overlay applied (view matrix on the fold model)",
               any(c.startswith(f"view matrix models #{fold_mid},") for c in a1.get("overlay_commands", []))))
checks.append(("(1) the fold OVERLAYS the reference (CA centroids within 8 Å)",
               com1 is not None and com1 < 8.0))

# ── 2) DISTANT, sequence-divergent: 1LH1 (leghemoglobin) ──────────────────────────────
print("[align] DISTANT → 1LH1 (leghemoglobin, low identity) …")
a2 = do_align("1LH1")
print(f"[align] 1LH1: TM_ref={a2.get('tm_ref')} seq_id_ali={a2.get('seq_id_ali')} "
      f"RMSD={a2.get('rmsd')} shared={a2.get('shared_fold')}")
checks.append(("(2) distant 1LH1 → meaningful TM_ref > 0.5 DESPITE low identity",
               (a2.get("tm_ref") or 0) > 0.5))
checks.append(("(2) and the structural Seq_ID is genuinely low (< 0.30)",
               (a2.get("seq_id_ali") if a2.get("seq_id_ali") is not None else 1.0) < 0.30))

# ── 3) UNRELATED: 1UBQ (ubiquitin) ────────────────────────────────────────────────────
print("[align] UNRELATED → 1UBQ (ubiquitin) …")
a3 = do_align("1UBQ")
print(f"[align] 1UBQ: TM_ref={a3.get('tm_ref')} shared={a3.get('shared_fold')}")
checks.append(("(3) unrelated 1UBQ → low TM_ref < 0.5, honest NOT-similar",
               (a3.get("tm_ref") or 1.0) < 0.5 and a3.get("shared_fold") is False))

# ── 4) HEAD-TO-HEAD: matchmaker vs US-align on the distant + unrelated refs ────────────
ref2 = a2.get("reference_model_id"); ref3 = a3.get("reference_model_id")
mm2 = matchmaker_headtohead("1LH1", ref2) if ref2 else "(no ref model)"
mm3 = matchmaker_headtohead("1UBQ", ref3) if ref3 else "(no ref model)"
print("\n── HEAD-TO-HEAD (matchmaker vs US-align) ──")
print(f"  1LH1 (distant):  US-align TM_ref={a2.get('tm_ref')} (confident)  |  matchmaker: {mm2}")
print(f"  1UBQ (unrelated): US-align TM_ref={a3.get('tm_ref')} (honest low) |  matchmaker: {mm3}")
# US-align gives a quantitative TM on BOTH; matchmaker fails-closed on the unrelated pair.
checks.append(("(4) matchmaker FAILS-CLOSED on the unrelated pair (cannot match / fewer than 3)",
               ("cannot match" in mm3.lower()) or ("fewer than" in mm3.lower())))
checks.append(("(4) US-align stayed quantitative on BOTH (a number where matchmaker gave none)",
               a3.get("tm_ref") is not None and a2.get("tm_ref") is not None))

QtCore.QThreadPool.globalInstance().waitForDone(5000)
print("[cleanup] leaving models open for inspection")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
