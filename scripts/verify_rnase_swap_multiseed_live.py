"""
Multi-seed hardening of the RNase A quaternary-IMPOSITION finding (both matched comparisons).

Single-seed gate (PASS) + decomposition (DECISIVE) established: guided-by-1A2W → N-swap (dimer-TM
0.823 vs 0.383), guided-by-1F0V → C-swap (0.773 vs 0.387), against the sequence's non-swap preference
(unguided makes neither: monoTM 0.978, TM~0.45 to both). This run does DOUBLE DUTY:

  (1) RULE OUT SEED-LUCK — fold each matched comparison at 3 seeds (0,1,2; seed 0 re-checks the
      pass-1 numbers, since Boltz is seed-pinned/deterministic). Imposition is robust iff EVERY seed
      keeps self-swap-TM ≫ other-swap-TM (the template's own assembly wins each time).
  (2) CHARACTERIZE REPRODUCTION FIDELITY — the matched swaps came back at 0.82/0.77, NOT PCNA's 0.99.
      Report the PER-SEED SPREAD (min/max/mean), not just the mean: is ~0.8 the reproducible CEILING
      for imposing a hard intertwined topology, or the center of a WIDE spread? Either is informative.

Both templates are CLEAN 2-chain dimers (1A2W ASU = N-swap; 1F0V_AB.cif = C-swap, built via ChimeraX
delete-C,D + save-whole-model — the 4-chain ASU + template_id route ERRORED, a harness artifact not a
capability limit). template_id omitted → Boltz default-searches the 2 template chains.

BOUND (record front-and-centre): ONE system, ONE structural class — DOMAIN-SWAPPED DIMERS. The claim
is "a template imposes its quaternary geometry, overriding sequence preference, FOR DOMAIN-SWAPPED
DIMERS" — NOT for all quaternary divergence. Non-swapped divergent interfaces / other symmetries /
larger assemblies = the §9 continuation, each needing fresh headroom-gated scoping before GPU.

Run: venv/Scripts/python.exe scripts/verify_rnase_swap_multiseed_live.py   (6 single-seed dimer folds — tens of minutes; background)
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

SEEDS = [0, 1, 2]
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="rnase_ms_")
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
            try: return {"tm1": float(p[2]), "tm2": float(p[3]), "rmsd": float(p[4]), "lali": int(p[10])}
            except ValueError: continue
    return None


def usalign(q, r, extra=""):
    if not (q and r and os.path.isfile(q) and os.path.isfile(r)): return None
    cmd = (f"{shlex.quote(_USEXE)} {shlex.quote(_wsl.translate_path(os.path.abspath(q)))} "
           f"{shlex.quote(_wsl.translate_path(os.path.abspath(r)))} -outfmt 2 {extra}").strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


# ── references + clean templates ────────────────────────────────────────────────────────
run("close session")
nswap_src = fetch("1A2W"); cswap_src = fetch("1F0V"); mono_src = fetch("7RSA")
REF_NSWAP = filter_chains(nswap_src, ["A", "B"], os.path.join(TD, "ref_nswap.cif"))
REF_CSWAP = filter_chains(cswap_src, ["A", "B"], os.path.join(TD, "ref_cswap.cif"))
REF_MONO  = filter_chains(mono_src, ["A"], os.path.join(TD, "ref_mono.cif"))

# N-swap template = 1A2W ASU (already a clean 2-chain dimer). C-swap template = clean 2-chain 1F0V_AB.
T_NSWAP_PATH = os.path.abspath(nswap_src)
clean_cswap = os.path.abspath(os.path.join(str(CACHE), "1F0V_AB.cif"))
before = set(models()); run("open 1F0V")
mid = (sorted(set(models()) - before, key=int) or ["1"])[-1]
run(f"delete #{mid}/C,D"); run(f'save "{Path(clean_cswap).as_posix()}" format mmcif models #{mid}'); run(f"close #{mid}")
cc = chain_ca(clean_cswap)
if len([c for c, n in cc.items() if n > 80]) != 2:
    print(f"[abort] clean C-swap template not 2 chains: {cc}"); sys.exit(2)
anc = usalign(REF_NSWAP, REF_CSWAP, extra="-mm 1")
print(f"[anchor] N-swap ⇄ C-swap = {anc['tm2']:.3f} (expect ~0.385)" if anc else "[anchor] failed")
print(f"[template] 1A2W (N-swap, {sorted(chain_ca(nswap_src))}) · clean 1F0V_AB (C-swap, {sorted(cc)})")

TEMPLATES = [("1A2W", "N", T_NSWAP_PATH), ("1F0V", "C", clean_cswap)]

# ── app plumbing ──────────────────────────────────────────────────────────────────────
seq = harvest_seq("7rsa", "A")
print(f"[harvest] RNase A {len(seq)} aa")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
router = ToolRouter(bridge, session)
host = types.SimpleNamespace(bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None, _log_exchange=lambda *a, **k: None)
engine = RequestEngine(host)


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
    def blocked(self, cmd, error): self.warnings.append(f"BLOCKED:{cmd}:{error}")
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


pres = ScriptedPresenter()
panel = VariantWorkbenchPanel(ctrl, session=session)
if not panel._fold_engine_availability().get("boltz"):
    print("[abort] Boltz env unavailable."); sys.exit(2)

# Seed injection: wrap predict to force the per-fold seed (seed is keyword-only).
_orig_predict = BoltzBridge.predict
_cur = {"seed": 0}
def _seeded_predict(self, *a, **k):
    k["seed"] = _cur["seed"]
    return _orig_predict(self, *a, **k)
BoltzBridge.predict = _seeded_predict


def drive(spec, on_apply, timeout=3_600_000):
    cap = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"], pres,
                               confidence=spec.get("confidence", "high"), on_result=cap.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if cap: on_apply(cap[0])
    return cap[0] if cap else None


def assess(data):
    cif = data.get("cif_path") or data.get("pdb_path"); iptm = data.get("iptm")
    pbc = data.get("plddt_by_chain") or {}
    if not (cif and os.path.isfile(cif)): return None
    chain_ids = data.get("chain_ids") or sorted(pbc.keys()) or ["A", "B"]
    mono = []
    for ch in chain_ids:
        cf = filter_chains(cif, [ch], os.path.join(TD, f"m_{ch}.cif"))
        a = usalign(cf, REF_MONO); mono.append(a["tm2"] if a else None)
    mono_mean = round(sum(x for x in mono if x)/len([x for x in mono if x]), 3) if any(mono) else None
    full = filter_chains(cif, list(chain_ids), os.path.join(TD, "asm.cif"))
    an = usalign(full, REF_NSWAP, extra="-mm 1"); ac = usalign(full, REF_CSWAP, extra="-mm 1")
    return {"iptm": round(iptm, 3) if iptm else None, "mono": mono_mean,
            "tm_n": round(an["tm2"], 3) if an else None, "tm_c": round(ac["tm2"], 3) if ac else None}


N = 2
panel._add_sequence_construct("rnase", seq)

results = {"N": [], "C": []}   # which -> list of (seed, metrics)
for pdb, which, path in TEMPLATES:
    for s in SEEDS:
        _cur["seed"] = s
        ref = {"path": path, "label": f"{pdb}", "force": False}
        gspec = panel.construct_fold_guided_spec("boltz", N, dict(ref))
        print(f"\n[{pdb}/{which}-swap seed={s}] folding…")
        graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
        gdata = panel._fold_from_result(graw) if graw else None
        m = assess(gdata) if gdata else None
        if m:
            print(f"   seed={s}: monoTM={m['mono']} TM→1A2W(N)={m['tm_n']} TM→1F0V(C)={m['tm_c']} ipTM={m['iptm']}")
        else:
            print(f"   seed={s}: FOLD FAILED  warnings={pres.warnings[-3:]}")
        results[which].append((s, m))

# ── REPORT ────────────────────────────────────────────────────────────────────────────
def spread(vals):
    v = [x for x in vals if x is not None]
    return (min(v), sum(v)/len(v), max(v)) if v else (None, None, None)

print("\n══ RNASE A QUATERNARY IMPOSITION — MULTI-SEED HARDENING ══")
print(f"  reference divergence N-swap ⇄ C-swap = {anc['tm2']:.3f}" if anc else "")
print(f"  unguided (pass1, ref): makes NEITHER swap — monoTM 0.978, TM→N 0.448 / →C 0.457\n")
robust = True
for which, pdb, self_key, other_key in [("N", "1A2W", "tm_n", "tm_c"), ("C", "1F0V", "tm_c", "tm_n")]:
    rows = results[which]
    print(f"  GUIDED-by-{pdb} ({which}-swap template) — does every seed reproduce its OWN swap?")
    selfs, others = [], []
    for s, m in rows:
        if not m:
            print(f"    seed {s}: FAILED"); robust = False; continue
        sv, ov = m[self_key], m[other_key]
        selfs.append(sv); others.append(ov)
        ok = (sv is not None and ov is not None and sv >= 0.55 and sv - ov > 0.10)
        robust = robust and ok
        print(f"    seed {s}: self({which})-TM={sv}  other-TM={ov}  ipTM={m['iptm']}  "
              f"{'✓ own swap wins' if ok else '✗ NOT clean'}")
    lo, mn, hi = spread(selfs)
    print(f"    → self-swap-TM spread: min {lo} · mean {round(mn,3) if mn else None} · max {hi}  "
          f"(reproduction fidelity for imposing a hard intertwined topology)\n")

print("── VERDICT ──")
if robust:
    print("  IMPOSITION ROBUST ACROSS SEEDS — every seed of each matched template reproduced its OWN")
    print("  divergent assembly (self-swap-TM ≫ other), against the sequence's non-swap preference.")
    print("  The single-seed decisive finding is NOT seed-luck.")
else:
    print("  NOT uniformly clean across seeds — inspect the per-seed rows above before claiming robust.")
print("\n  BOUND (front-and-centre): ONE system, ONE structural class — DOMAIN-SWAPPED DIMERS. The claim is")
print("  'a template imposes its quaternary geometry, overriding sequence preference, FOR DOMAIN-SWAPPED")
print("  DIMERS' — NOT all quaternary divergence. Non-swapped interfaces / other symmetries / larger")
print("  assemblies = §9 continuation, each headroom-gated before GPU.")
print("\nDONE — multi-seed hardening complete.")
