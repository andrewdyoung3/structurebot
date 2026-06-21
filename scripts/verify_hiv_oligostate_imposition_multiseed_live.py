"""
Step 1 — HIV-1 CA oligomeric-STATE imposition, MATCHED arrangement, multi-seed (lean).

Does a template impose the oligomeric STATE / ring arrangement (not the interface)? HIV CA forms BOTH
a pentamer (3P05) and a hexamer (3H47) from the SAME sequence — so this is a same-sequence two-state
test with NO sequence confound. The construct sets the chain count; what the template can impose is the
arrangement (ring closure/curvature) at that count.

  Fixed query = the CA sequence (harvested from 3P05 chain A). For each state, fold at the matching
  chain count, UNGUIDED (floor) vs GUIDED by the clean N-chain template, dual-decomposition vs BOTH
  crystals (native + the other state, to show which arrangement formed):
    • pentamer  N=5, template 3P05 (5 chains), native ref 3P05 / other ref 3H47
    • hexamer   N=6, template 3H47 (6 chains), native ref 3H47 / other ref 3P05
  Per fold: per-chain monomer-TM (subunits folded?) + whole-assembly ring-TM (US-align -mm) to native
  AND to the other state, + ipTM.

LEAN seed plan (the relay's weighting — the imposition signal is the guided-vs-unguided gap, and the
directional/fidelity claim lives in the GUIDED folds): GUIDED ≥3 seeds each state, UNGUIDED 1 seed
(the floor, not the variable). 2 + 6 = 8 folds.

Templates are the clean N-chain assemblies (3P05 ASU = 5 chains; 3H47 assembly-1 = 6 chains);
template_id OMITTED → Boltz search-maps the N query chains to the N template positions
(linear_sum_assignment, verified N-mer wiring). Multi-seed from the start (single-seed misled twice).

Run: venv/Scripts/python.exe scripts/verify_hiv_oligostate_imposition_multiseed_live.py   (8 Boltz folds, ~1000–1200 res — HOURS; background)
"""
import os, re, sys, types, tempfile, shlex, urllib.request
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
from boltz_bridge import BoltzBridge
from wsl_bridge import WSLBridge

GUIDED_SEEDS = [0, 1, 2]
# (state, N, template pdb, native ref pdb, other ref pdb)
STATES = [
    ("pentamer", 5, "3P05", "3P05", "3H47"),
    ("hexamer",  6, "3H47", "3H47", "3P05"),
]
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="hiv_state_")
_wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
_USEXE = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")

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


def fetch(pdb, assembly=False):
    name = f"{pdb}-assembly1.cif" if assembly else f"{pdb}.cif"
    dst = CACHE / name
    if dst.is_file() and dst.stat().st_size > 0: return str(dst)
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{name}", dst); return str(dst)


def chain_ca(path, col="auth_asym_id"):
    cols, counts = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("_atom_site."): cols.append(ln.strip())
            elif ln.startswith(("ATOM", "HETATM")):
                p = ln.split()
                ci = next((i for i, c in enumerate(cols) if c.endswith(col)), None)
                ai = next((i for i, c in enumerate(cols) if c.endswith("label_atom_id")), None)
                if ci is not None and ai is not None and ci < len(p) and ai < len(p) and p[ai].strip('"') == "CA":
                    counts[p[ci]] = counts.get(p[ci], 0) + 1
    return counts


def filter_chains(src, wanted, dst, col="auth_asym_id"):
    wanted = set(wanted); lines = open(src, encoding="utf-8", errors="replace").read().splitlines()
    hdr, data, i, n = [], [], 0, len(lines); ci = None
    while i < n:
        if lines[i].startswith("_atom_site."):
            while i < n and lines[i].startswith("_atom_site."): hdr.append(lines[i].strip()); i += 1
            ci = next((k for k, c in enumerate(hdr) if c.endswith(col)), None)
            while i < n and lines[i].startswith(("ATOM", "HETATM")):
                p = lines[i].split()
                if ci is not None and ci < len(p) and p[ci] in wanted: data.append(lines[i])
                i += 1
            break
        i += 1
    open(dst, "w", encoding="utf-8").write("data_x\nloop_\n" + "\n".join(hdr) + "\n" + "\n".join(data) + "\n")
    return dst


def _parse(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("#PDBchain1"): continue
        p = ln.split("\t")
        if len(p) >= 11:
            try: return {"tm2": float(p[3]), "rmsd": float(p[4]), "lali": int(p[10])}
            except ValueError: continue
    return None


def usalign(q, r, extra=""):
    if not (q and r and os.path.isfile(q) and os.path.isfile(r)): return None
    cmd = (f"{shlex.quote(_USEXE)} {shlex.quote(_wsl.translate_path(os.path.abspath(q)))} "
           f"{shlex.quote(_wsl.translate_path(os.path.abspath(r)))} -outfmt 2 {extra}").strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 300))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


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
ca = harvest_seq("3p05", "A")
print(f"[harvest] HIV-1 CA (3P05 chain A) {len(ca)} aa")
if len(ca) < 120:
    print("[abort] CA sequence too short"); sys.exit(2)

# clean N-chain templates + native/other refs; template_id OMITTED (search-map N query → N template)
REF, MONO_REF, TMPL = {}, {}, {}
for pdb, asm in (("3P05", False), ("3H47", True)):
    src = fetch(pdb, assembly=asm)
    prot = sorted([c for c, n in chain_ca(src).items() if n > 60])
    need = 5 if pdb == "3P05" else 6
    if len(prot) < need:
        print(f"[abort] {pdb} has {len(prot)} protein chains, need {need}: {prot}"); sys.exit(2)
    REF[pdb] = filter_chains(src, prot[:need], os.path.join(TD, f"{pdb}_ring.cif"))
    TMPL[pdb] = os.path.abspath(src)
    print(f"[{pdb}] {len(prot)} protein chains (using {need}) — template={Path(src).name}")
MONO_REF = filter_chains(fetch("3P05"), [sorted([c for c, n in chain_ca(fetch('3P05')).items() if n > 60])[0]],
                         os.path.join(TD, "mono_ref.cif"))

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

_cur = {"seed": 0}
_orig_predict = BoltzBridge.predict
def _seeded_predict(self, *a, **k):
    k["seed"] = _cur["seed"]; return _orig_predict(self, *a, **k)
BoltzBridge.predict = _seeded_predict


def drive(spec, on_apply, timeout=10_800_000):
    cap = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"], pres,
                               confidence=spec.get("confidence", "high"), on_result=cap.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if cap: on_apply(cap[0])
    return cap[0] if cap else None


def assess(data, native, other):
    cif = data.get("cif_path") or data.get("pdb_path"); iptm = data.get("iptm")
    pbc = data.get("plddt_by_chain") or {}
    if not (cif and os.path.isfile(cif)): return None
    chain_ids = data.get("chain_ids") or sorted(pbc.keys())
    mono = []
    for ch in chain_ids:
        cf = filter_chains(cif, [ch], os.path.join(TD, f"m_{ch}.cif"))
        a = usalign(cf, MONO_REF); mono.append(a["tm2"] if a else None)
    mono_mean = round(sum(x for x in mono if x)/len([x for x in mono if x]), 3) if any(mono) else None
    full = filter_chains(cif, list(chain_ids), os.path.join(TD, "asm.cif"))
    an = usalign(full, REF[native], extra="-mm 1"); ao = usalign(full, REF[other], extra="-mm 1")
    return {"iptm": round(iptm, 3) if iptm else None, "mono": mono_mean,
            "ring_native": round(an["tm2"], 3) if an else None,
            "ring_other": round(ao["tm2"], 3) if ao else None}


panel._add_sequence_construct("hivca", ca)
results = {}   # state -> {"unguided": m, "guided": [(seed, m), ...]}
for state, N, tmpl_pdb, native, other in STATES:
    results[state] = {"unguided": None, "guided": []}
    print(f"\n[{state}] UNGUIDED N={N} (floor, 1 seed, ~{N*len(ca)} res)…")
    _cur["seed"] = 0
    uspec = panel.construct_fold_launch_spec("boltz", N)
    uraw = drive(uspec, lambda r: panel.apply_construct_fold_result(panel.construct_fold_launch_spec("boltz", N), r))
    um = assess(panel._fold_from_result(uraw), native, other) if uraw else None
    results[state]["unguided"] = um
    print(f"   unguided: monoTM={um['mono'] if um else None} ringTM→{native}={um['ring_native'] if um else None} "
          f"→{other}={um['ring_other'] if um else None} ipTM={um['iptm'] if um else None}")
    for s in GUIDED_SEEDS:
        print(f"[{state}] GUIDED N={N} by {tmpl_pdb} seed={s}…")
        _cur["seed"] = s
        ref = {"path": TMPL[tmpl_pdb], "label": tmpl_pdb, "force": False}
        gspec = panel.construct_fold_guided_spec("boltz", N, dict(ref))
        graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
        gm = assess(panel._fold_from_result(graw), native, other) if graw else None
        results[state]["guided"].append((s, gm))
        print(f"   seed {s}: monoTM={gm['mono'] if gm else None} ringTM→{native}={gm['ring_native'] if gm else None} "
              f"→{other}={gm['ring_other'] if gm else None} ipTM={gm['iptm'] if gm else None}")


def spread(vals):
    v = [x for x in vals if x is not None]
    return (min(v), round(sum(v)/len(v), 3), max(v)) if v else (None, None, None)

print("\n══ HIV CA OLIGOMERIC-STATE IMPOSITION — matched, multi-seed (lean) ══")
print("  same CA sequence folded at each chain count; does the matching template impose the ring?\n")
for state, N, tmpl_pdb, native, other in STATES:
    um = results[state]["unguided"]; gm = results[state]["guided"]
    ufloor = um["ring_native"] if um else None
    gvals = [m["ring_native"] for _, m in gm if m]
    lo, mn, hi = spread(gvals)
    print(f"  {state.upper()} (N={N}, template {tmpl_pdb}, native vs {other}):")
    print(f"    unguided ring-TM→{native} (floor): {ufloor}  (monoTM {um['mono'] if um else None}, →{other} {um['ring_other'] if um else None})")
    for s, m in gm:
        if m:
            print(f"    guided seed {s}: ring-TM→{native}={m['ring_native']}  →{other}={m['ring_other']}  "
                  f"monoTM={m['mono']}  ipTM={m['iptm']}")
        else:
            print(f"    guided seed {s}: FAILED")
    gap = (mn - ufloor) if (mn is not None and ufloor is not None) else None
    print(f"    → guided ring-TM→{native} spread: min {lo} mean {mn} max {hi}  |  guided−unguided gap: {gap}")
    if mn is not None and ufloor is not None:
        verdict = ("IMPOSED (guided≫unguided, →native>→other)" if (gap and gap > 0.15
                   and all((m["ring_native"] - m["ring_other"]) > 0.05 for _, m in gm if m))
                   else "weak/ambiguous — inspect rows")
        print(f"    → {verdict}")
    print()

print("NOTES: imposition = guided ring-TM ≫ unguided floor AND guided→native > guided→other, every seed;")
print("the SAME CA sequence reaching BOTH states via matching templates = oligomeric-STATE imposition,")
print("NO sequence confound. Count-mismatch curvature-override DEFERRED (degenerate per the probe).")
print("\nDONE — HIV oligomeric-state matched multi-seed complete.")
