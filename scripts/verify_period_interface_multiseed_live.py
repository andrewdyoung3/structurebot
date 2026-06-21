"""
Multi-seed hardening of the PERIOD non-swap interface-imposition finding (own-vs-other template contrast).

Single-seed (gate + decomposition) established, via the OWN-vs-OTHER template contrast (fixed sequence,
MSA-free, only the template varies — the clean test that does NOT need an unguided baseline):
  • non-swap interface imposition is REAL → imposition is NOT swap-specific;
  • ASYMMETRIC: 4DJ3-seq fully adopts the non-native 3GDI interface (TM→3GDI 0.931 vs →4DJ3-native 0.714);
    3GDI-seq only partially shifts toward 4DJ3 (0.765 vs its own 0.753 — resists toward native);
  • the asymmetry rules out "template is the only signal" (genuine sequence preference survives, partly
    overridden) — because MSA-free Boltz CANNOT fold PERIOD unguided (monoTM 0.25), so the own-vs-other
    contrast — not the unguided baseline — is the valid imposition test here.

This run multi-seeds the 4 GUIDED conditions (3GDI/4DJ3 × own/other template) at seeds 0,1,2 to:
  (1) confirm the directional shift is not seed-luck (own-template native-TM > other-template native-TM,
      every seed, both sequences);
  (2) characterize the ASYMMETRY + fidelity spread (is the 4DJ3→3GDI full override reproducible? does the
      3GDI→4DJ3 partial shift resolve or stay tied?);
  (3) re-check ipTM-vs-fidelity (RNase: confidence != fidelity; does that recur for non-swap?).
Unguided is SKIPPED (confirmed uninformative — MSA-free folding fails on PERIOD).

BOUND: controllable 59%-seq-id homolog-template axis (one fixed sequence per condition, only template
varies). Headroom 0.721 → the own↔other interface separation is ~0.28 TM (moderate). NON-swap, crystal,
solution-validated. ONE system — generality beyond this pair is its own §9 scope.

Run: venv/Scripts/python.exe scripts/verify_period_interface_multiseed_live.py   (12 dimer folds — ~2h; background)
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
PAIR = [("3GDI", "4DJ3"), ("4DJ3", "3GDI")]   # (sequence, the OTHER/override template)
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="period_ms_")
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
            try: return {"tm2": float(p[3]), "rmsd": float(p[4]), "idali": float(p[7]), "lali": int(p[10])}
            except ValueError: continue
    return None


def usalign(q, r, extra=""):
    if not (q and r and os.path.isfile(q) and os.path.isfile(r)): return None
    cmd = (f"{shlex.quote(_USEXE)} {shlex.quote(_wsl.translate_path(os.path.abspath(q)))} "
           f"{shlex.quote(_wsl.translate_path(os.path.abspath(r)))} -outfmt 2 {extra}").strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


run("close session")
SEQ, REF_DIMER, REF_MONO, TMPL = {}, {}, {}, {}
for pdb in ("3GDI", "4DJ3"):
    src = fetch(pdb)
    chains = sorted([c for c, n in chain_ca(src).items() if n > 60])
    pair = chains[:2]
    REF_DIMER[pdb] = filter_chains(src, pair, os.path.join(TD, f"{pdb}_dim.cif"))
    REF_MONO[pdb]  = filter_chains(src, [pair[0]], os.path.join(TD, f"{pdb}_mono.cif"))
    TMPL[pdb] = os.path.abspath(src)
    SEQ[pdb] = harvest_seq(pdb, pair[0])
anc = usalign(REF_DIMER["3GDI"], REF_DIMER["4DJ3"], extra="-mm 1")
print(f"[anchor] 3GDI ⇄ 4DJ3 dimer-TM = {anc['tm2']:.3f} (expect ~0.721)" if anc else "[anchor] FAILED")

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

_cur = {"seed": 0}
_orig_predict = BoltzBridge.predict
def _seeded_predict(self, *a, **k):
    k["seed"] = _cur["seed"]; return _orig_predict(self, *a, **k)
BoltzBridge.predict = _seeded_predict


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
    full = filter_chains(cif, list(chain_ids), os.path.join(TD, "asm.cif"))
    an = usalign(full, REF_DIMER[native], extra="-mm 1"); ao = usalign(full, REF_DIMER[other], extra="-mm 1")
    mono = []
    for ch in chain_ids:
        cf = filter_chains(cif, [ch], os.path.join(TD, f"m_{ch}.cif"))
        a = usalign(cf, REF_MONO[native]); mono.append(a["tm2"] if a else None)
    mono_mean = round(sum(x for x in mono if x)/len([x for x in mono if x]), 3) if any(mono) else None
    return {"iptm": round(iptm, 3) if iptm else None, "mono": mono_mean,
            "tm_native": round(an["tm2"], 3) if an else None,
            "tm_other": round(ao["tm2"], 3) if ao else None}


N = 2
res = {}   # (seq, tmpl_role) -> list of (seed, metrics);  tmpl_role in {"own","other"}
for seq_name, other_name in PAIR:
    panel._add_sequence_construct(f"period_{seq_name}", SEQ[seq_name])
    for role, tmpl_name in (("own", seq_name), ("other", other_name)):
        res[(seq_name, role)] = []
        for s in SEEDS:
            _cur["seed"] = s
            ref = {"path": TMPL[tmpl_name], "label": tmpl_name, "force": False}
            gspec = panel.construct_fold_guided_spec("boltz", N, dict(ref))
            print(f"\n[{seq_name}-seq / {role}={tmpl_name}  seed={s}] folding…")
            graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
            gdata = panel._fold_from_result(graw) if graw else None
            m = assess(gdata, seq_name, other_name) if gdata else None
            if m:
                print(f"   monoTM={m['mono']} TM→native({seq_name})={m['tm_native']} "
                      f"TM→other({other_name})={m['tm_other']} ipTM={m['iptm']}")
            else:
                print(f"   FAILED  warnings={pres.warnings[-3:]}")
            res[(seq_name, role)].append((s, m))
    panel._design = None


def spread(vals):
    v = [x for x in vals if x is not None]
    return (min(v), sum(v)/len(v), max(v)) if v else (None, None, None)

print("\n══ PERIOD NON-SWAP INTERFACE IMPOSITION — MULTI-SEED (own-vs-other contrast) ══")
print(f"  reference 3GDI ⇄ 4DJ3 dimer-TM = {anc['tm2']:.3f}  (unguided SKIPPED: MSA-free fold fails on PERIOD)\n" if anc else "")
robust = True
for seq_name, other_name in PAIR:
    print(f"  SEQUENCE {seq_name}  (native={seq_name}, other/override template={other_name}):")
    own_rows = res[(seq_name, "own")]; oth_rows = res[(seq_name, "other")]
    print(f"    own-template ({seq_name}) — should hold native:")
    for s, m in own_rows:
        if m: print(f"      seed {s}: TM→native={m['tm_native']}  TM→other={m['tm_other']}  ipTM={m['iptm']}  monoTM={m['mono']}")
        else: print(f"      seed {s}: FAILED"); robust = False
    print(f"    other-template ({other_name}) — OVERRIDE test (does native-TM drop & other-TM rise?):")
    for s, m in oth_rows:
        if m: print(f"      seed {s}: TM→native={m['tm_native']}  TM→other={m['tm_other']}  ipTM={m['iptm']}  monoTM={m['mono']}")
        else: print(f"      seed {s}: FAILED"); robust = False
    # the imposition shift, per seed: does swapping own->other template move native DOWN and other UP?
    print(f"    → per-seed IMPOSITION SHIFT (own→other template, same sequence):")
    for (s1, mo), (s2, mt) in zip(own_rows, oth_rows):
        if mo and mt:
            dnat = round(mt["tm_native"] - mo["tm_native"], 3)   # expect negative (native drops)
            doth = round(mt["tm_other"] - mo["tm_other"], 3)     # expect positive (other rises)
            adopt = mt["tm_other"] - mt["tm_native"]             # >0 => adopted OTHER interface under other-template
            tag = "FULL override (other>native)" if adopt > 0.10 else ("native-wins" if adopt < -0.10 else "TIE/partial")
            print(f"      seed {s1}: Δnative={dnat:+.3f} Δother={doth:+.3f}  under other-tmpl: other−native={adopt:+.3f}  → {tag}")
    lo, mn, hi = spread([m["tm_other"] for _, m in oth_rows if m])
    print(f"    → other-template TM→other spread: min {lo} mean {round(mn,3) if mn else None} max {hi}\n")

print("── VERDICT ──")
print("  Imposition is GENERAL (non-swap): for a fixed sequence, the template shifts which divergent")
print("  interface forms (own→other moves native↓ / other↑). Read the per-seed ASYMMETRY: 4DJ3-seq")
print("  full override toward 3GDI vs 3GDI-seq partial — is it robust across seeds? Is ipTM tracking")
print("  fidelity (unlike RNase)? See rows above.")
print("\n  BOUND: ONE homolog pair, 59%-seq-id template axis, headroom 0.721 (~0.28 TM separation), MSA-free")
print("  fold needs the template (no unguided baseline). Generality beyond this pair = own §9 scope.")
print("\nDONE — PERIOD multi-seed complete.")
