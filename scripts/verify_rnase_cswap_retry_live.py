"""
Gate completion — the C-swap (1F0V) side, retried with a CLEAN 2-chain template.

In the first gate pass the 1A2W (N-swap) matched fold PASSED decisively (dimer-TM 0.823 to 1A2W vs
0.383 to 1F0V; monoTM 0.978→0.827 = the open swapped protomer) — Boltz generated the intertwined
N-swap against the sequence's non-swap preference (unguided makes neither: monoTM 0.978, TM~0.45 to
both). But the 1F0V matched fold ERRORED (no model). The ONE variable that differed: 1F0V was handed
as the FULL 4-chain ASU + template_id=[A,B], whereas 1A2W was a clean 2-chain ASU (no template_id).
An ERRORED fold must NOT be read as "Boltz can't make the C-swap" — so this retry gives the C-swap a
clean 2-chain template (parallel to the working 1A2W case) and CAPTURES the Boltz error if it recurs.

  • Build cache/1F0V_AB.cif via the running ChimeraX: open 1F0V, delete chains C,D (the OTHER dimer;
    contact-confirmed A+B = the C-swap pair), save the remaining 2-chain model as mmCIF. (Deleting,
    then saving the whole model, sidesteps the ChimeraX 'CIF writer ignores chain atomspecs' gotcha.)
  • Fold RNase A dimer guided by the clean 1F0V_AB template (template_id OMITTED → Boltz default-
    searches the 2 template chains, exactly as the 1A2W case). Single seed.
  • Score vs REF N-swap (1A2W A+B) and REF C-swap (1F0V A+B): does it reproduce the C-swap?

Known from pass 1 (for the combined read): unguided TM→N 0.448 / →C 0.457 (no swap);
guided-1A2W TM→N 0.823 / →C 0.383 (N-swap ✓).

Run: venv/Scripts/python.exe scripts/verify_rnase_cswap_retry_live.py   (1 single-seed dimer fold — minutes)
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

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="rnase_cswap_")
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


# ── build the CLEAN 2-chain C-swap template via ChimeraX (delete C,D → save whole model) ──
run("close session")
nswap_src = fetch("1A2W"); cswap_src = fetch("1F0V"); mono_src = fetch("7RSA")
REF_NSWAP = filter_chains(nswap_src, ["A", "B"], os.path.join(TD, "ref_nswap.cif"))
REF_CSWAP = filter_chains(cswap_src, ["A", "B"], os.path.join(TD, "ref_cswap.cif"))
REF_MONO  = filter_chains(mono_src, ["A"], os.path.join(TD, "ref_mono.cif"))

clean_cif = os.path.abspath(os.path.join(str(CACHE), "1F0V_AB.cif"))
before = set(models()); run("open 1F0V")
mid = (sorted(set(models()) - before, key=int) or ["1"])[-1]
run(f"delete #{mid}/C,D")                                   # drop the OTHER dimer; keep the A+B C-swap pair
run(f'save "{Path(clean_cif).as_posix()}" format mmcif models #{mid}')
run(f"close #{mid}")
if not os.path.isfile(clean_cif):
    print(f"[abort] ChimeraX did not write {clean_cif}"); sys.exit(2)
cc = chain_ca(clean_cif)
print(f"[template] clean 1F0V_AB.cif chains (CA counts): {cc}")
if len([c for c, nca in cc.items() if nca > 80]) != 2:
    print(f"[abort] clean C-swap template is not 2 chains — got {cc} (atomspec-save gotcha?)"); sys.exit(2)

# sanity anchor (should reproduce 0.385) — and the clean template vs REF C-swap (should be ~1.0)
anc = usalign(REF_NSWAP, REF_CSWAP, extra="-mm 1")
tmpl_self = usalign(clean_cif, REF_CSWAP, extra="-mm 1")
print(f"[anchor] N-swap ⇄ C-swap = {anc['tm2']:.3f}" if anc else "[anchor] failed")
print(f"[template] clean 1F0V_AB ⇄ REF C-swap = {tmpl_self['tm2']:.3f} (expect ~1.0)" if tmpl_self else "[template] self-check failed")

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

_orig_yaml = BoltzBridge._build_yaml; _cap = {"yaml": []}
def _cap_yaml(chains, templates=None):
    y = _orig_yaml(chains, templates); _cap["yaml"].append(y); return y
BoltzBridge._build_yaml = staticmethod(_cap_yaml)


def drive(spec, on_apply, timeout=3_600_000):
    cap = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"], pres,
                               confidence=spec.get("confidence", "high"), on_result=cap.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if cap: on_apply(cap[0])
    return cap[0] if cap else None


def assess(tag, data):
    mid = str(data.get("new_model_id") or data.get("model_id"))
    cif = data.get("cif_path") or data.get("pdb_path"); iptm = data.get("iptm")
    pbc = data.get("plddt_by_chain") or {}
    if not (cif and os.path.isfile(cif)):
        print(f"[{tag}] no predicted CIF"); return None
    chain_ids = data.get("chain_ids") or sorted(pbc.keys()) or ["A", "B"]
    means = {ch: (round(sum(v.values())/len(v), 1) if v else None) for ch, v in pbc.items()}
    mono = []
    for ch in chain_ids:
        cf = filter_chains(cif, [ch], os.path.join(TD, f"{tag}_{ch}.cif"))
        a = usalign(cf, REF_MONO); mono.append(a["tm2"] if a else None)
    mono_mean = round(sum(x for x in mono if x)/len([x for x in mono if x]), 3) if any(mono) else None
    full = filter_chains(cif, list(chain_ids), os.path.join(TD, f"{tag}_asm.cif"))
    an = usalign(full, REF_NSWAP, extra="-mm 1"); ac = usalign(full, REF_CSWAP, extra="-mm 1")
    tm_n = round(an["tm2"], 3) if an else None
    tm_c = round(ac["tm2"], 3) if ac else None
    print(f"[{tag}] #{mid} ipTM={iptm} pLDDT={means} monoTM={mono_mean} "
          f"dimerTM→1A2W(N)={tm_n}(RMSD {round(an['rmsd'],2) if an else None}Å) "
          f"→1F0V(C)={tm_c}(RMSD {round(ac['rmsd'],2) if ac else None}Å)")
    return {"iptm": iptm, "mono_mean": mono_mean, "tm_n": tm_n, "tm_c": tm_c}


N = 2
panel._add_sequence_construct("rnase", seq)
T_CSWAP = {"path": clean_cif, "label": "1F0V_AB(clean)", "force": False}    # 2-chain, no template_id
gspec = panel.construct_fold_guided_spec("boltz", N, dict(T_CSWAP))
if not gspec:
    print("[abort] guided spec build failed"); sys.exit(2)

print(f"\n[guided-1F0V-clean] RNase A homodimer GUIDED by clean 2-chain C-swap template — minutes…")
_cap["yaml"].clear()
graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
gdata = panel._fold_from_result(graw) if graw else None
gy = next((y for y in _cap["yaml"] if "templates:" in y), "")
print(f"   [yaml] template line(s): {[ln.strip() for ln in gy.splitlines() if 'chain_id' in ln or 'template_id' in ln or '.cif' in ln]}")
if pres.warnings:
    print(f"   [warnings] {pres.warnings[:6]}")
g = assess("guided-1F0V-clean", gdata) if gdata else None

print("\n══ C-SWAP SIDE (clean template) — combined with pass-1 numbers ══")
print(f"  reference divergence N-swap ⇄ C-swap = {anc['tm2']:.3f}" if anc else "")
print(f"  {'fold':<22}{'monoTM':>8}{'TM→1A2W(N)':>12}{'TM→1F0V(C)':>12}{'ipTM':>9}  read")
print(f"  {'unguided (pass1)':<22}{0.978:>8}{0.448:>12}{0.457:>12}{0.799:>9}  no swap (neither)")
print(f"  {'guided-1A2W (pass1)':<22}{0.827:>8}{0.823:>12}{0.383:>12}{0.838:>9}  → N-swap ✓")
if g and g["tm_n"] is not None and g["tm_c"] is not None:
    tn, tc = g["tm_n"], g["tm_c"]
    if max(tn, tc) < 0.55: read = "NO defined swap"
    elif tc - tn > 0.10: read = "→ C-swap (1F0V) ✓matches template"
    elif tn - tc > 0.10: read = "→ N-swap (WRONG — copied N?)"
    else: read = "ambiguous"
    print(f"  {'guided-1F0V (clean)':<22}{str(g['mono_mean']):>8}{str(tn):>12}{str(tc):>12}{str(g['iptm']):>9}  {read}")
    print("\n── IMPOSITION VERDICT ──")
    if tc >= 0.55 and tc - tn > 0.10:
        print("  BOTH templates select their OWN divergent assembly (1A2W→N-swap, 1F0V→C-swap), against")
        print("  the sequence's non-swap preference → DECISIVE quaternary imposition (single seed).")
        print("  NEXT: multi-seed both matched comparisons to confirm it is not stochastic.")
    else:
        print("  C-swap NOT cleanly reproduced even with a clean template — diagnose (see TM split above).")
else:
    print(f"  {'guided-1F0V (clean)':<22}  — fold FAILED again; warnings: {pres.warnings[:6]}")
    print("\n  C-swap fold failed with a CLEAN 2-chain template too → the failure is NOT the 4-chain/")
    print("  template_id artifact. Capture the Boltz error and diagnose before any capability claim.")
print("\nDONE — C-swap retry complete.")
