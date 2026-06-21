"""
NON-SWAP quaternary IMPOSITION — generality test on PERIOD PAS homolog dimers (single-seed gate + full
decomposition). Does the RNase A domain-swap imposition finding GENERALIZE to a normal docked
(non-intertwined) divergent interface, or is it swap-specific?

System (scoped no-GPU): mouse PERIOD PAS-AB homolog dimers — 3GDI ⇄ 4DJ3, dimer-TM 0.721 (US-align
-mm), monomer fold conserved (monoTM 0.866), seq-id 59%, BOTH crystal, non-swap, solution-validated.
Each sequence has a NATIVE dimer → this is a copy-vs-unlock-TOWARD-NATIVE test (sharper than RNase,
where neither swap was native):

  Fold each sequence as a dimer (N=2) under three conditions, score each vs its NATIVE dimer AND the
  OTHER (template) dimer interface:
    • unguided          — does the sequence make its NATIVE dimer alone? (the premise / baseline)
    • guided-by-OWN     — matched control: reproduces native? (gate: model+template can build it)
    • guided-by-OTHER   — OVERRIDE test: does the divergent template impose the OTHER interface
                          (TM→other ≫ TM→native) or does the sequence's native preference win?

  OVERRIDE (TM→other ≫ TM→native) = imposition GENERALIZES beyond swaps.
  NATIVE-WINS (TM→native ≫ TM→other) = the sequence's quaternary preference beats a misleading
  template → imposition was swap-SPECIFIC (RNase's swap was imposable precisely because it had no
  native competitor). Either is decisive on "general or swap-specific".

Templates = the full ASU CIFs (both 3GDI & 4DJ3 ASUs are clean 2-chain A+B dimers — parallel to the
RNase 1A2W clean-2-chain case; template_id omitted, chain_id defaults to the construct block [A,B]).
The 59% seq-id is a controllable homolog-template confound (monomer arc unlocked at 28% id, PCNA at
25% — well inside the transfer regime); one FIXED sequence per condition, only the template varies.

Single-seed here (gate + decomposition); multi-seed hardening (override AND matched, ~3 seeds, +the
ipTM-vs-fidelity recheck) is the NEXT run if the gate passes — same rhythm as the RNase arc.

Run: venv/Scripts/python.exe scripts/verify_period_interface_imposition_live.py   (6 single-seed dimer folds — tens of minutes; background)
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

SEED = int(os.environ.get("PERIOD_SEED", "0"))
PAIR = [("3GDI", "4DJ3"), ("4DJ3", "3GDI")]   # (sequence, the OTHER/override template)
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="period_imp_")
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
            try: return {"tm1": float(p[2]), "tm2": float(p[3]), "rmsd": float(p[4]),
                         "idali": float(p[7]), "lali": int(p[10])}
            except ValueError: continue
    return None


def usalign(q, r, extra=""):
    if not (q and r and os.path.isfile(q) and os.path.isfile(r)): return None
    cmd = (f"{shlex.quote(_USEXE)} {shlex.quote(_wsl.translate_path(os.path.abspath(q)))} "
           f"{shlex.quote(_wsl.translate_path(os.path.abspath(r)))} -outfmt 2 {extra}").strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


# ── sequences + references + templates ──────────────────────────────────────────────────
run("close session")
SEQ, REF_DIMER, REF_MONO, TMPL = {}, {}, {}, {}
for pdb in ("3GDI", "4DJ3"):
    src = fetch(pdb)
    cc = {c: n for c, n in chain_ca(src).items() if n > 60}
    chains = sorted(cc.keys())
    if len(chains) < 2:
        print(f"[abort] {pdb} ASU has <2 long chains: {cc}"); sys.exit(2)
    pair = chains[:2]
    REF_DIMER[pdb] = filter_chains(src, pair, os.path.join(TD, f"{pdb}_dim.cif"))
    REF_MONO[pdb]  = filter_chains(src, [pair[0]], os.path.join(TD, f"{pdb}_mono.cif"))
    TMPL[pdb] = os.path.abspath(src)                       # full ASU = clean 2-chain dimer
    SEQ[pdb] = harvest_seq(pdb, pair[0])
    print(f"[{pdb}] ASU chains={chains} dimer={pair} seq={len(SEQ[pdb])} aa")

anc = usalign(REF_DIMER["3GDI"], REF_DIMER["4DJ3"], extra="-mm 1")
mono = usalign(REF_MONO["3GDI"], REF_MONO["4DJ3"])
print(f"[anchor] 3GDI ⇄ 4DJ3 dimer-TM(-mm)={anc['tm2']:.3f} (expect ~0.721) · "
      f"monoTM={mono['tm2']:.3f} seq-id={mono['idali']*100:.0f}%" if (anc and mono) else "[anchor] FAILED")

# ── app plumbing ──────────────────────────────────────────────────────────────────────
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

# seed injection (keyword-only) + YAML capture
_orig_predict = BoltzBridge.predict
def _seeded_predict(self, *a, **k):
    k["seed"] = SEED
    return _orig_predict(self, *a, **k)
BoltzBridge.predict = _seeded_predict
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


def assess(data, native, other):
    cif = data.get("cif_path") or data.get("pdb_path"); iptm = data.get("iptm")
    pbc = data.get("plddt_by_chain") or {}
    if not (cif and os.path.isfile(cif)): return None
    chain_ids = data.get("chain_ids") or sorted(pbc.keys()) or ["A", "B"]
    means = {ch: (round(sum(v.values())/len(v), 1) if v else None) for ch, v in pbc.items()}
    mono = []
    for ch in chain_ids:
        cf = filter_chains(cif, [ch], os.path.join(TD, f"m_{ch}.cif"))
        a = usalign(cf, REF_MONO[native]); mono.append(a["tm2"] if a else None)
    mono_mean = round(sum(x for x in mono if x)/len([x for x in mono if x]), 3) if any(mono) else None
    full = filter_chains(cif, list(chain_ids), os.path.join(TD, "asm.cif"))
    an = usalign(full, REF_DIMER[native], extra="-mm 1")
    ao = usalign(full, REF_DIMER[other],  extra="-mm 1")
    return {"iptm": round(iptm, 3) if iptm else None, "mono": mono_mean,
            "tm_native": round(an["tm2"], 3) if an else None,
            "tm_other": round(ao["tm2"], 3) if ao else None,
            "rmsd_native": round(an["rmsd"], 2) if an else None}


N = 2
rows = []  # (seq, condition, template, metrics)
for seq_name, other_name in PAIR:
    panel._add_sequence_construct(f"period_{seq_name}", SEQ[seq_name])
    conditions = [("unguided", None), ("guided-own", seq_name), ("guided-other", other_name)]
    for cond, tmpl_name in conditions:
        print(f"\n[{seq_name}-seq / {cond}{'' if not tmpl_name else ' by '+tmpl_name}  seed={SEED}] folding…")
        _cap["yaml"].clear()
        if tmpl_name is None:
            spec = panel.construct_fold_launch_spec("boltz", N)
            graw = drive(spec, lambda r: panel.apply_construct_fold_result(panel.construct_fold_launch_spec("boltz", N), r))
        else:
            ref = {"path": TMPL[tmpl_name], "label": tmpl_name, "force": False}
            spec = panel.construct_fold_guided_spec("boltz", N, dict(ref))
            graw = drive(spec, lambda r: panel.apply_construct_fold_guided_result(spec, r))
        gdata = panel._fold_from_result(graw) if graw else None
        m = assess(gdata, seq_name, other_name) if gdata else None
        if m:
            print(f"   monoTM(native)={m['mono']} TM→NATIVE({seq_name})={m['tm_native']} "
                  f"TM→OTHER({other_name})={m['tm_other']} ipTM={m['iptm']}")
        else:
            print(f"   FOLD FAILED  warnings={pres.warnings[-3:]}")
        rows.append((seq_name, cond, tmpl_name, m))
    # fresh construct for the next sequence (panel holds one design)
    panel._design = None

# ── REPORT ────────────────────────────────────────────────────────────────────────────
print("\n══ PERIOD NON-SWAP INTERFACE IMPOSITION — single-seed gate + decomposition ══")
print(f"  reference: 3GDI ⇄ 4DJ3 dimer-TM = {anc['tm2']:.3f} (seq-id 59%, non-swap, both crystal)\n" if anc else "")
print(f"  {'seq':<6}{'condition':<14}{'template':<9}{'monoTM':>8}{'TM→native':>11}{'TM→other':>10}{'ipTM':>7}  read")
gate = {"native_pref": {}, "matched": {}}
for seq_name, cond, tmpl_name, m in rows:
    if not m:
        print(f"  {seq_name:<6}{cond:<14}{str(tmpl_name):<9}{'  — FAILED'}"); continue
    tn, to = m["tm_native"], m["tm_other"]
    read = "?"
    if tn is not None and to is not None:
        if cond == "unguided":
            read = "native ✓" if tn - to > 0.10 else ("→OTHER?!" if to - tn > 0.10 else "ambiguous")
            gate["native_pref"][seq_name] = (tn - to > 0.10)
        elif cond == "guided-own":
            read = "reproduces native ✓" if tn - to > 0.10 else "matched did NOT rebuild native ✗"
            gate["matched"][seq_name] = (tn - to > 0.10 and tn >= 0.55)
        else:  # guided-other = the override test
            if to - tn > 0.10 and to >= 0.55: read = "OVERRIDE → adopted OTHER interface"
            elif tn - to > 0.10: read = "NATIVE-WINS (template did NOT override)"
            else: read = "intermediate / ambiguous"
    print(f"  {seq_name:<6}{cond:<14}{str(tmpl_name):<9}{str(m['mono']):>8}{str(tn):>11}{str(to):>10}{str(m['iptm']):>7}  {read}")

print("\n── GATE ──")
np_ok = all(gate["native_pref"].get(s) for s in ("3GDI", "4DJ3"))
mt_ok = all(gate["matched"].get(s) for s in ("3GDI", "4DJ3"))
print(f"  native-preference (unguided→native both seqs): {np_ok}")
print(f"  matched-template rebuilds native (both seqs):   {mt_ok}")
if np_ok and mt_ok:
    print("  GATE PASS — premise holds (sequences prefer native; model+template can build each interface).")
    print("  → the OVERRIDE rows are interpretable. If override ADOPTS the other interface, imposition")
    print("    GENERALIZES to non-swap interfaces; if NATIVE-WINS, imposition was swap-specific.")
    print("  NEXT (if a direction shows): multi-seed override AND matched (~3 seeds) + ipTM-vs-fidelity recheck.")
else:
    print("  GATE NOT FULLY CLEAN — interpret the override rows with care; diagnose the failing leg first")
    print("  (premise/buildability must hold before an override claim).")
print("\n  BOUND: non-swap test carries a controllable 59%-seq-id homolog-template axis (one fixed sequence")
print("  per condition; only the template varies). Headroom 0.721 → override-vs-native separation ~0.28 TM.")
print("\nDONE — PERIOD single-seed gate + decomposition complete.")
