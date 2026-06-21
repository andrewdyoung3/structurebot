"""
PRESCREEN (Step 0) — HIV-1 CA pentamer headroom gate for the oligomeric-STATE imposition study.

The matched arrangement-imposition test (same CA sequence → pentamer via 3P05 template, hexamer via
3H47 template — no sequence confound) is only meaningful if the UNGUIDED ring is POOR (room for the
template to impose the arrangement). RISK: HIV CA hexamer/pentamer are iconic, PDB-trained structures
— Boltz may reproduce the ring unguided even MSA-free, killing the headroom. So before the multi-hour
matched run we spend ONE cheap unguided fold and read the ring-TM.

  Fold the CA sequence (harvested from 3P05 chain A) as a 5-mer UNGUIDED (single seed) → dual
  decomposition vs the 3P05 pentamer crystal:
    • per-chain monomer-TM (mean) vs the 3P05 CA monomer  — did the subunit fold?
    • whole-assembly ring-TM (US-align -mm) vs the 3P05 5-chain ring  — did the pentamer assemble?

  VERDICT BAND (the relay's gate):
    ring-TM > 0.85         → MEMORISED (Boltz reproduces the pentamer unguided) → NO headroom → HOLD.
    0.6 ≤ ring-TM ≤ 0.85   → ambiguous middle (room exists but a smaller signal) → HOLD for a call.
    ring-TM < 0.6          → CLEAR headroom (clearly below a competent ~0.9 guided ring) → auto-proceed
                             to the lean matched Step 1.

Run: venv/Scripts/python.exe scripts/verify_hiv_oligostate_prescreen_live.py   (1 Boltz fold, ~1030 res — minutes; background)
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
from wsl_bridge import WSLBridge

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="hiv_pre_")
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


def fetch(pdb):
    dst = CACHE / f"{pdb}.cif"
    if dst.is_file() and dst.stat().st_size > 0: return str(dst)
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.cif", dst); return str(dst)


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
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
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

pent_src = fetch("3P05")
prot = sorted([c for c, n in chain_ca(pent_src).items() if n > 60])
print(f"[3P05] pentamer protein chains: {prot}")
REF_RING = filter_chains(pent_src, prot[:5], os.path.join(TD, "ref_pent.cif"))
REF_MONO = filter_chains(pent_src, [prot[0]], os.path.join(TD, "ref_mono.cif"))

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


def drive(spec, on_apply, timeout=7_200_000):
    cap = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"], pres,
                               confidence=spec.get("confidence", "high"), on_result=cap.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if cap: on_apply(cap[0])
    return cap[0] if cap else None


N = 5
panel._add_sequence_construct("hivca", ca)
print(f"\n[unguided] folding HIV-1 CA as a {N}-mer UNGUIDED (single seed, ~{N*len(ca)} res) — minutes…")
uspec = panel.construct_fold_launch_spec("boltz", N)
uraw = drive(uspec, lambda r: panel.apply_construct_fold_result(panel.construct_fold_launch_spec("boltz", N), r))
data = panel._fold_from_result(uraw) if uraw else None
if not data:
    print(f"[abort] unguided fold produced no result. warnings={pres.warnings[-4:]}"); sys.exit(2)

cif = data.get("cif_path") or data.get("pdb_path")
iptm = data.get("iptm"); pbc = data.get("plddt_by_chain") or {}
chain_ids = data.get("chain_ids") or sorted(pbc.keys())
means = {ch: (round(sum(v.values())/len(v), 1) if v else None) for ch, v in pbc.items()}
mono = []
for ch in chain_ids:
    cf = filter_chains(cif, [ch], os.path.join(TD, f"u_{ch}.cif"))
    a = usalign(cf, REF_MONO); mono.append(a["tm2"] if a else None)
mono_mean = round(sum(x for x in mono if x)/len([x for x in mono if x]), 3) if any(mono) else None
full = filter_chains(cif, list(chain_ids), os.path.join(TD, "u_asm.cif"))
ra = usalign(full, REF_RING, extra="-mm 1")
ring_tm = round(ra["tm2"], 3) if ra else None
ring_rmsd = round(ra["rmsd"], 2) if ra else None

print(f"\n[unguided] #{data.get('new_model_id') or data.get('model_id')} ipTM={iptm} pLDDT={means}")
print(f"           monomer-TM(mean vs 3P05 CA)={mono_mean}  ring-TM(-mm vs 3P05 pentamer)={ring_tm}  RMSD={ring_rmsd}Å")

print("\n══ HIV CA PENTAMER PRESCREEN — headroom gate ══")
print(f"  unguided pentamer ring-TM vs 3P05 = {ring_tm}  (monomer-TM {mono_mean}, ipTM {iptm})")
if ring_tm is None:
    print("  VERDICT: ring-TM unmeasured — investigate before proceeding.")
elif ring_tm > 0.85:
    print("  VERDICT: MEMORISED (>0.85) — Boltz reproduces the pentamer unguided → NO headroom → HOLD.")
elif ring_tm >= 0.6:
    print("  VERDICT: AMBIGUOUS MIDDLE (0.6–0.85) — room exists but a smaller signal → HOLD for a call.")
else:
    print("  VERDICT: CLEAR HEADROOM (<0.6) — clearly below a competent guided ring → AUTO-PROCEED to lean Step 1.")
print("\nDONE — prescreen complete.")
