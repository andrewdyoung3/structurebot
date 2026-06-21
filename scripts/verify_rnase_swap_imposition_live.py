"""
Step 2 — QUATERNARY IMPOSITION test on a genuinely quaternary-divergent system (RNase A), single-seed.

REFRAME (per the framing call): this is a quaternary IMPOSITION test, NOT copy-vs-unlock. RNase A
gives ONE sequence (100% id) with TWO crystal-solved divergent dimer assemblies (assembly-TM 0.385
apart, both X-ray, fixed chain count) AND an intrinsic preference for the CLOSED MONOMER (domain
swapping is metastable). So there is NO sequence-determined dimer truth — the TEMPLATE is the
comparison target. The question: does the choice of template CONTROL which of the two divergent
assemblies forms?

  Forms (bovine RNase A, 100% identical sequence, all X-ray):
    7RSA  closed monomer            — intrinsic ground state (subunit-fold reference)
    1A2W  N-terminal swapped DIMER  — assembly form A (ASU = the 2-chain dimer)
    1F0V  C-terminal swapped DIMER  — assembly form B (ASU = 4 chains / two dimers; A+B = C-swap,
                                       contact- AND divergence-confirmed: -mm to 1A2W = 0.385)

THE GATE (run FIRST — single seed): does Boltz make the swap AT ALL when matched-templated?
    fold RNase A as a dimer guided by 1A2W → dimer-TM to 1A2W ;  guided by 1F0V → dimer-TM to 1F0V.
  • matching-template fold does NOT reproduce its swap → Boltz cannot generate an intertwined
    domain-swapped topology even when templated. CLEAN CAPABILITY FINDING — record it; RNase A then
    cannot test imposition → STOP (do NOT pre-scope a fallback; the swap confound is self-gating).
  • matching-template fold DOES reproduce its swap → swap-generation works → imposition is testable.

IMPOSITION decomposition (computed in the same single-seed pass; multi-seed hardening is the NEXT
turn IF the gate passes): does guided-by-1A2W give the N-swap (TM to 1A2W ≫ TM to 1F0V) while
guided-by-1F0V gives the C-swap (TM to 1F0V ≫ TM to 1A2W)? If the template controls which of the
two 0.385-divergent assemblies forms — against the sequence's monomer preference — that is decisive
quaternary imposition. Plus an UNGUIDED dimer baseline (what does the sequence do alone?).

4-axis decomposition per fold: monomer-TM (vs closed 7RSA — subunits folded?), dimer-TM to 1A2W &
to 1F0V (-mm — which assembly?), ipTM, per-chain pLDDT.

Run: venv/Scripts/python.exe scripts/verify_rnase_swap_imposition_live.py   (3 single-seed dimer folds — many minutes; run backgrounded)
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
TD = tempfile.mkdtemp(prefix="rnase_swap_")
_wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
_USEXE = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")

bridge = ChimeraXBridge(port=60001)
try:
    bridge.start(timeout=120); print("[chimerax] REST server up on :60001")
except Exception as exc:
    print(f"[abort] could not start ChimeraX: {exc}"); sys.exit(2)
run = bridge.run_command


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
    try: urllib.request.urlretrieve(f"https://files.rcsb.org/download/{name}", dst); return str(dst)
    except Exception as exc: print(f"   [fetch] {name} failed: {exc}"); return None


def chain_ca(path, col):
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


def ca_coords(path, col):
    cols, out = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("_atom_site."): cols.append(ln.strip())
            elif ln.startswith(("ATOM", "HETATM")):
                p = ln.split()
                idx = {k: next((i for i, c in enumerate(cols) if c.endswith(k)), None)
                       for k in ("auth_asym_id", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z")}
                if None in idx.values() or max(idx.values()) >= len(p): continue
                if p[idx["label_atom_id"]].strip('"') == "CA":
                    try:
                        out.setdefault(p[idx["auth_asym_id"]], []).append(
                            (float(p[idx["Cartn_x"]]), float(p[idx["Cartn_y"]]), float(p[idx["Cartn_z"]])))
                    except ValueError: pass
    return out


def dimer_partner(path, anchor, cutoff=8.0):
    cc = ca_coords(path, "auth_asym_id")
    if anchor not in cc: return None, {}
    best, counts = None, {}
    for ch, pts in cc.items():
        if ch == anchor: continue
        n = 0
        for ax, ay, az in cc[anchor]:
            if any((ax-bx)**2 + (ay-by)**2 + (az-bz)**2 < cutoff*cutoff for bx, by, bz in pts):
                n += 1
        counts[ch] = n
        if best is None or n > counts[best]: best = ch
    return best, counts


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
            try: return {"tm1": float(p[2]), "tm2": float(p[3]), "rmsd": float(p[4]), "lali": int(p[10]), "l2": int(p[9])}
            except ValueError: continue
    return None


def usalign(q, r, extra=""):
    if not (q and r and os.path.isfile(q) and os.path.isfile(r)): return None
    cmd = (f"{shlex.quote(_USEXE)} {shlex.quote(_wsl.translate_path(os.path.abspath(q)))} "
           f"{shlex.quote(_wsl.translate_path(os.path.abspath(r)))} -outfmt 2 {extra}").strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


class ScriptedPresenter(Presenter):
    def __init__(self): self.warnings = []; self.confirm_calls = []; self.confirm_answer = "proceed"
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
    def confirm(self, confidence): self.confirm_calls.append(confidence); return self.confirm_answer
    def ask_edit(self, original): return list(original)
    def ask_yes_no(self, q, default="y"): return False


# ── sequence + references ───────────────────────────────────────────────────────────────
run("close session")
seq = harvest_seq("7rsa", "A")
if not seq or len(seq) < 100:
    print(f"[abort] could not harvest RNase A sequence (got {len(seq)} aa)"); sys.exit(2)
print(f"[harvest] RNase A {len(seq)} aa")

# REF dimers (US-align scoring targets; minimal CIFs are fine for US-align).
nswap_src = fetch("1A2W")                       # ASU = 2-chain N-swap dimer
cswap_src = fetch("1F0V")                       # ASU = 4 chains / two dimers
mono_src  = fetch("7RSA")
if not (nswap_src and cswap_src and mono_src):
    print("[abort] reference download failed"); sys.exit(2)

# 1F0V: isolate the intertwined dimer containing chain A (= the C-swap pair; contact-confirmed).
mate, contacts = dimer_partner(cswap_src, "A")
cswap_pair = ["A", mate or "B"]
print(f"[1F0V] C-swap dimer = chains {cswap_pair}  (CA-CA contacts to A: {contacts})")

REF_NSWAP = filter_chains(nswap_src, ["A", "B"], os.path.join(TD, "ref_nswap.cif"))
REF_CSWAP = filter_chains(cswap_src, cswap_pair, os.path.join(TD, "ref_cswap.cif"))
REF_MONO  = filter_chains(mono_src, ["A"], os.path.join(TD, "ref_mono.cif"))

# Sanity anchor: reproduce the scoping divergence (should be ~0.385).
anchor = usalign(REF_NSWAP, REF_CSWAP, extra="-mm 1")
print(f"[anchor] REF N-swap ⇄ REF C-swap assembly-TM (-mm) = "
      f"{anchor['tm2']:.3f} (expect ~0.385 — confirms divergent refs)" if anchor else "[anchor] FAILED")

# Boltz TEMPLATES = full RCSB mmCIFs (robust gemmi parse). 1A2W ASU = the 2-chain N-swap dimer
# (default chain mapping). 1F0V ASU has 4 chains → template_id picks the C-swap pair.
T_NSWAP = {"path": os.path.abspath(nswap_src), "label": "1A2W", "force": False}
T_CSWAP = {"path": os.path.abspath(cswap_src), "label": "1F0V", "force": False,
           "template_id": cswap_pair}

# ── app plumbing ──────────────────────────────────────────────────────────────────────
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

_orig_yaml = BoltzBridge._build_yaml; _cap = {"yaml": []}
def _cap_yaml(chains, templates=None):
    y = _orig_yaml(chains, templates); _cap["yaml"].append(y); return y
BoltzBridge._build_yaml = staticmethod(_cap_yaml)


def drive(spec, on_apply, timeout=3_600_000):
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"; cap = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"], pres,
                               confidence=spec.get("confidence", "high"), on_result=cap.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if cap: on_apply(cap[0])
    return cap[0] if cap else None


N = 2
panel._add_sequence_construct("rnase", seq)

# Build + sanity-check all specs BEFORE the (expensive) folds — fail fast on plumbing.
spec_u = panel.construct_fold_launch_spec("boltz", N)
spec_n = panel.construct_fold_guided_spec("boltz", N, dict(T_NSWAP))
spec_c = panel.construct_fold_guided_spec("boltz", N, dict(T_CSWAP))
if not (spec_u and spec_n and spec_c):
    print(f"[abort] spec build failed: unguided={bool(spec_u)} 1A2W={bool(spec_n)} 1F0V={bool(spec_c)}")
    sys.exit(2)
print(f"[plan] dimer N={N}; templates 1A2W (N-swap, full ASU) & 1F0V (C-swap, template_id={cswap_pair})")


def assess(tag, data):
    mid = str(data.get("new_model_id") or data.get("model_id"))
    cif = data.get("cif_path") or data.get("pdb_path")
    iptm = data.get("iptm"); pbc = data.get("plddt_by_chain") or {}
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
    rmsd_n = round(an["rmsd"], 2) if an else None
    rmsd_c = round(ac["rmsd"], 2) if ac else None
    print(f"[{tag}] #{mid} ipTM={iptm} pLDDT={means} monoTM={mono_mean} "
          f"dimerTM→1A2W(N)={tm_n}(RMSD {rmsd_n}Å) →1F0V(C)={tm_c}(RMSD {rmsd_c}Å)")
    return {"iptm": iptm, "mono_mean": mono_mean, "tm_n": tm_n, "tm_c": tm_c}


# ── UNGUIDED baseline (what does the sequence do alone?) ──────────────────────────────
print(f"\n[unguided] folding RNase A homodimer UNGUIDED (single seed) — minutes…")
uraw = drive(spec_u, lambda r: panel.apply_construct_fold_result(panel.construct_fold_launch_spec("boltz", N), r))
udata = panel._fold_from_result(uraw) if uraw else None
u = assess("unguided", udata) if udata else None

# ── GATE + IMPOSITION: guided by each swap template (single seed) ──────────────────────
def guided(tag, ref):
    print(f"\n[guided-{tag}] RNase A homodimer GUIDED by {ref['label']} — minutes…")
    gspec = panel.construct_fold_guided_spec("boltz", N, dict(ref))
    _cap["yaml"].clear()
    graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
    gdata = panel._fold_from_result(graw) if graw else None
    gy = next((y for y in _cap["yaml"] if "templates:" in y), "")
    print(f"   [yaml] template line(s): {[ln.strip() for ln in gy.splitlines() if 'chain_id' in ln or 'template_id' in ln or '.cif' in ln or '.pdb' in ln]}")
    return assess("guided-" + tag, gdata) if gdata else None

g_n = guided("1A2W", T_NSWAP)     # matched → should reproduce N-swap (gate)
g_c = guided("1F0V", T_CSWAP)     # matched → should reproduce C-swap (gate)

# ── REPORT ────────────────────────────────────────────────────────────────────────────
print("\n══ RNASE A QUATERNARY IMPOSITION — single-seed (GATE + decomposition) ══")
print(f"  reference divergence: N-swap ⇄ C-swap assembly-TM = {anchor['tm2']:.3f}" if anchor else "  (anchor failed)")
print(f"  {'fold':<18}{'monoTM':>8}{'TM→1A2W(N)':>12}{'TM→1F0V(C)':>12}{'ipTM':>7}  read")
def line(tag, m, expect):
    if not m: print(f"  {tag:<18}{'  — fold/metric failed'}"); return
    tn, tc = m["tm_n"], m["tm_c"]
    read = "?"
    if tn is not None and tc is not None:
        if max(tn, tc) < 0.55: read = "NO defined swap (neither assembly)"
        elif tn - tc > 0.10: read = "→ N-swap (1A2W)" + ("  ✓matches template" if expect == "N" else "")
        elif tc - tn > 0.10: read = "→ C-swap (1F0V)" + ("  ✓matches template" if expect == "C" else "")
        else: read = "ambiguous (both similar)"
    print(f"  {tag:<18}{str(m['mono_mean']):>8}{str(tn):>12}{str(tc):>12}{str(m['iptm']):>7}  {read}")
line("unguided", u, None)
line("guided-1A2W", g_n, "N")
line("guided-1F0V", g_c, "C")

print("\n── GATE VERDICT ──")
def reproduces(m, which):
    if not m: return False
    tn, tc = m["tm_n"], m["tm_c"]
    if tn is None or tc is None: return False
    return (tn if which == "N" else tc) >= 0.55 and (tn - tc if which == "N" else tc - tn) > 0.10
gate_n = reproduces(g_n, "N"); gate_c = reproduces(g_c, "C")
if gate_n or gate_c:
    print(f"  GATE PASS — Boltz reproduced a swap when matched-templated "
          f"(1A2W→N-swap: {gate_n}; 1F0V→C-swap: {gate_c}). Swap-generation works → imposition testable.")
    if gate_n and gate_c:
        print("  IMPOSITION (single-seed): BOTH templates selected their OWN divergent assembly →")
        print("  the template CONTROLS which of the two 0.385-divergent assemblies forms, against the")
        print("  sequence's monomer preference. DECISIVE imposition (single seed) → MULTI-SEED next to")
        print("  confirm it is not stochastic, then report the full decomposition.")
    else:
        print("  IMPOSITION PARTIAL — only one matched template reproduced its swap; the other did not.")
        print("  Interpret per the decomposition above; multi-seed the reproduced side, diagnose the other.")
else:
    print("  GATE FAIL — neither matched-template fold reproduced its swap. CLEAN CAPABILITY FINDING:")
    print("  Boltz cannot generate an intertwined domain-swapped topology even when templated, so")
    print("  RNase A cannot test quaternary imposition. STOP — do NOT pre-scope a fallback (the swap")
    print("  confound is self-gating; a closed-dimer alternative reintroduces a sequence confound).")
print("\nDONE — single-seed gate complete.")
