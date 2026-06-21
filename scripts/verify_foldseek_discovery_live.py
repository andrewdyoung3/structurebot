"""
LIVE verify — Stage 2 foldseek auto template discovery, end-to-end (the second gate).

Proves the SHIPPED seam on real artifacts: a de-novo construct → unguided monomer fold (real
`cd.template_fold`) → the productised `FoldseekBridge` search of that fold against the LOCAL PDB DB
→ real ranked structural neighbours → `_foldseek_refs` → `construct_fold_guided_spec` (the SAME
picker convergence point) → a guided re-fold by a discovered template → a real model + adoption.

Uses RNase A (bovine, 124 aa) as the construct — its unguided monomer folds well MSA-free and its
PDB neighbours are unambiguous ribonucleases, so the discovery is checkable.

Gate assertions:
  1. FoldseekBridge.is_available() is True (binary + local DB present).
  2. search_neighbors(real unguided monomer fold) returns real PDB neighbours, high-TM, ranked.
  3. refs build → construct_fold_guided_spec carries them as ti['templates'].
  4. a guided re-fold by the top discovered template produces a real model.

Run: venv/Scripts/python.exe scripts/verify_foldseek_discovery_live.py   (2 Boltz folds — minutes; background)
"""
import os, re, sys, types, tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import config as _cfg
from PySide6 import QtCore, QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from tool_router import ToolRouter
from request_engine import RequestEngine
from presenter import Presenter
from variant_workbench import VariantWorkbenchPanel
from foldseek_bridge import FoldseekBridge

FAILS = []
def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond: FAILS.append(msg)

bridge = ChimeraXBridge(port=60001)
bridge.start(timeout=120)
run = bridge.run_command
print("[chimerax] attached on :60001")


def models(): return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))
def _runscript(s):
    fd, p = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f: f.write(s)
    try: return run(f'runscript "{p}"').get("value") or ""
    finally:
        try: os.unlink(p)
        except OSError: pass


def harvest_seq(pdb, chain="A"):
    before = set(models()); run(f"open {pdb}")
    mid = (sorted(set(models()) - before, key=int) or ["1"])[-1]
    out = _runscript("from chimerax.atomic import all_atomic_structures\n"
                     "for m in all_atomic_structures(session):\n"
                     f"    if m.id_string=='{mid}':\n"
                     f"        rs=[r for r in m.residues if r.chain_id=='{chain}' and r.polymer_type==r.PT_AMINO]\n"
                     "        rs.sort(key=lambda r:r.number)\n"
                     "        print('SEQ',''.join((r.one_letter_code or 'X') for r in rs))\n")
    run(f"close #{mid}")
    return next((l.split()[1] for l in out.splitlines() if l.startswith("SEQ")), "")


class ScriptedPresenter(Presenter):
    def __init__(self): self.warnings = []
    def info(self, t): pass
    def warn(self, t): self.warnings.append("WARN:" + str(t))
    def error(self, t): self.warnings.append("ERR:" + str(t))
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
    def confirm(self, confidence): return "proceed"
    def ask_edit(self, original): return list(original)
    def ask_yes_no(self, q, default="y"): return False


run("close session")
seq = harvest_seq("7rsa", "A")
check(len(seq) > 100, f"harvested RNase A sequence ({len(seq)} aa)")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
router = ToolRouter(bridge, session)
host = types.SimpleNamespace(bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None, _log_exchange=lambda *a, **k: None)
engine = RequestEngine(host)
pres = ScriptedPresenter()
panel = VariantWorkbenchPanel(ctrl, session=session)
if not panel._fold_engine_availability().get("boltz"):
    print("[abort] Boltz env unavailable."); sys.exit(2)


def drive(spec, on_apply, timeout=3_600_000):
    cap = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"], pres,
                               confidence=spec.get("confidence", "high"), on_result=cap.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if cap: on_apply(cap[0])
    return cap[0] if cap else None


# ── 1. unguided MONOMER fold (the real query) ──────────────────────────────────────────
panel._add_sequence_construct("rnase", seq)
print("\n[unguided] folding RNase A monomer (the foldseek query) — minutes…")
uspec = panel.construct_fold_launch_spec("boltz", 1)
drive(uspec, lambda r: panel.apply_construct_fold_result(panel.construct_fold_launch_spec("boltz", 1), r))
cd = next(iter(panel._design.chains.values()))
tf = cd.template_fold or {}
qsrc = tf.get("cif_path") or tf.get("pdb_path")
check(bool(qsrc and os.path.isfile(qsrc)), f"unguided fold produced an on-disk query ({tf.get('mean_plddt')} pLDDT)")

# ── 2. foldseek discovery on the real fold ─────────────────────────────────────────────
fb = FoldseekBridge()
check(fb.is_available(), f"FoldseekBridge.is_available() — {fb.status()} · {fb.db_label()}")
qpath = panel._foldseek_query_path(qsrc)
hits = fb.search_neighbors(qpath, max_results=15, min_tm=0.3)
print(f"\n[foldseek] {len(hits)} neighbours (top 8):")
for pid, ch, tm in hits[:8]:
    print(f"    {pid}_{ch}  TM={tm:.3f}")
check(len(hits) >= 3, f"search returned ≥3 structural neighbours ({len(hits)})")
check(all(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", p) for p, _, _ in hits), "all hits are valid 4-char PDB ids")
check(hits and hits[0][2] > 0.8, f"top neighbour is high-TM ({hits[0][2] if hits else 'n/a'})")
# The neighbours must be the RNase A FOLD FAMILY. NOTE: foldseek's PDB DB is CLUSTERED, so the exact
# source entry (7RSA) need NOT appear — it is represented by cluster reps (e.g. 1A5P/1AFL/8UB2). The
# DB-agnostic, non-circular check: several hits share the RNase A fold (TM≥0.9 to the RNase A query).
top_ids = {p for p, _, _ in hits}
n_fold = sum(1 for _, _, tm in hits if tm >= 0.9)
check(n_fold >= 3, f"≥3 neighbours share the RNase A fold (TM≥0.9): {n_fold} of {len(hits)} "
                   f"({sorted(top_ids)[:6]}…)")

# ── 3. refs → guided spec (the picker convergence point) ───────────────────────────────
# pick the top discovered template that is NOT the trivial self-query source, to prove a real re-fold.
picked = [p for p, _, _ in hits if p != "7RSA"][:1] or [hits[0][0]]
refs = panel._foldseek_refs(picked)
gspec = panel.construct_fold_guided_spec("boltz", 1, refs)
check(gspec is not None, "construct_fold_guided_spec built from discovered refs")
tmpls = (gspec or {}).get("tool_inputs", {}).get("templates") if gspec else None
check(bool(tmpls) and any(t.get("pdb_id") == picked[0] for t in tmpls),
      f"guided spec carries the discovered template(s) as ti['templates'] ({picked})")

# ── 4. guided re-fold by the discovered template (closes the seam) ─────────────────────
print(f"\n[guided] re-folding RNase A monomer guided by the discovered template {picked[0]} — minutes…")
graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
gdata = panel._fold_from_result(graw) if graw else None
mid = (gdata or {}).get("new_model_id") or (gdata or {}).get("model_id")
check(bool(mid), f"guided re-fold by the discovered template produced a model (#{mid})")

print("\n══ STAGE-2 FOLDSEEK DISCOVERY — LIVE GATE ══")
print(f"  discovery: {len(hits)} real PDB neighbours of the unguided fold; top {hits[0][0] if hits else '-'}={hits[0][2] if hits else '-'}")
print("  RESULT:", "ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED → {FAILS}")
sys.exit(0 if not FAILS else 1)
