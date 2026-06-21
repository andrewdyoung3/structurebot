"""
Step 2 — DIVERGENT-RING generalization (the quaternary unlock-vs-copy probe), single-seed.

Step 1 (the ladder) found PCNA quaternary structure is CONSERVED across all life (seq-id 25→100%
but ring-TM 0.90→0.99). So this system can ONLY test sequence-divergence-with-conserved-quaternary;
it CANNOT test quaternary divergence (logged as the real §9 continuation). Within that bound there
is still one genuinely uncertain question at the archaeal rung — so we run it, single-seed, and let
the data choose the claim under DUAL framing:

  Guide the HUMAN PCNA homotrimer by each ring template (the 3-CHAIN ring, via assembly-1 — NOT the
  1-chain ASU, else no quaternary info is conveyed), four-axis decomposition per rung:
    • monomer-TM (mean A,B,C vs 1AXC monomer)   — subunits folded?
    • ring-TM to 1AXC (-mm)                       — correct human ring assembled?
    • ipTM                                        — inter-chain confidence
    • per-chain pLDDT

  TRANSFER (clean; whole family supports it): does a sequence-divergent ring template trigger the
    assembly the UNGUIDED fold (ring-TM ~0.40) could not? ring-TM ≫ unguided = quaternary
    "structure without sequence" conveyed.
  UNLOCK vs COPY (only the archaeal 1GE8 rung has headroom to resolve): guided-ring-TM-to-1AXC vs
    the TEMPLATE's OWN ring-structTM-to-1AXC. ≫ template = the model CORRECTED the divergent
    template toward the human ring (unlock); ≈ template = it COPIED the template's arrangement.

  Rungs: 1AXC (human, identical control = ceiling) · 1PLQ (yeast, near-conserved 0.973) ·
         1GE8 (archaeal, the headroom rung 0.896, RMSD 2.54 Å — structurally distinguishable).
  Honest scope: a 1-rung probe, NOT a titration.

Run: venv/Scripts/python.exe scripts/verify_pcna_ring_divergent_template_live.py   (4 single-seed folds — many minutes)
"""
import os, re, sys, types, tempfile, shlex, urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DEVIATION_FLOOR_N", "2")
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
TD = tempfile.mkdtemp(prefix="pcna_div_")
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


def pcna_ring_file(pdb):
    """Path to a 3-chain PCNA ring (ASU if it has ≥3 long chains, else assembly-1). Used BOTH as the
    Boltz template AND filtered for US-align scoring. Returns (ring_full_path, [chain ids], col)."""
    asu = fetch(pdb)
    for src in (asu, None):
        path = src if src else fetch(pdb, assembly=True)
        if not path: continue
        for col in ("auth_asym_id", "label_asym_id"):
            ch = sorted([c for c, nca in chain_ca(path, col).items() if nca > 150])
            if len(ch) >= 3:
                return path, ch[:3], col
    return None, None, None


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


# ── references + human sequence ───────────────────────────────────────────────────────
run("close session")
pcna = harvest_seq("1axc", "A")
if not pcna: print("[abort] could not harvest human PCNA from 1AXC"); sys.exit(2)
print(f"[harvest] human PCNA {len(pcna)} aa")
axc_ring, axc_ch, axc_col = pcna_ring_file("1AXC")
REF_RING = filter_chains(axc_ring, axc_ch, os.path.join(TD, "ref_ring.cif"), axc_col)
REF_MONO = filter_chains(axc_ring, [axc_ch[0]], os.path.join(TD, "ref_mono.cif"), axc_col)
print(f"[ref] 1AXC ring chains {axc_ch} via {axc_col}")

# template rings (the 3-chain assemblies handed to Boltz) + their OWN ring-structTM to 1AXC
TEMPLATES = []   # (label, template_ref_for_spec, template_ring_TM_to_1AXC)
for pdb, label in [("1AXC", "1AXC human (identical ctrl)"),
                   ("1PLQ", "1PLQ yeast (near-conserved)"),
                   ("1GE8", "1GE8 archaeal (headroom rung)")]:
    rpath, rch, rcol = pcna_ring_file(pdb)
    if not rpath:
        print(f"[warn] {pdb}: no 3-chain ring — skipped as a template"); continue
    tfile = filter_chains(rpath, rch, os.path.join(TD, f"{pdb}_tmplring.cif"), rcol)
    a = usalign(tfile, REF_RING, extra="-mm 1")
    tmpl_tm = round(a["tm2"], 3) if a else None
    # Hand Boltz the 3-CHAIN ring. 1AXC ASU already carries the ring (+p21, harmless); 1PLQ/1GE8 use assembly-1.
    ref = {"path": os.path.abspath(rpath), "label": pdb}
    TEMPLATES.append((label, ref, tmpl_tm, len(rch), rpath))
    print(f"[tmpl] {pdb}: ring from {Path(rpath).name} ({len(rch)} chains), own ring-TM-to-1AXC={tmpl_tm}")

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


N = 3
panel._add_sequence_construct("pcna", pcna)
cd = next(iter(panel._design.chains.values()))


def assess(tag, data):
    mid = str(data.get("new_model_id") or data.get("model_id"))
    cif = data.get("cif_path") or data.get("pdb_path")
    iptm = data.get("iptm"); pbc = data.get("plddt_by_chain") or {}
    if not (cif and os.path.isfile(cif)):
        print(f"[{tag}] no predicted CIF"); return None
    chain_ids = data.get("chain_ids") or sorted(pbc.keys()) or ["A", "B", "C"]
    means = {ch: (round(sum(v.values())/len(v), 1) if v else None) for ch, v in pbc.items()}
    mono = []
    for ch in chain_ids:
        cf = filter_chains(cif, [ch], os.path.join(TD, f"{tag}_{ch}.cif"))
        a = usalign(cf, REF_MONO); mono.append(a["tm2"] if a else None)
    mono_mean = round(sum(x for x in mono if x)/len([x for x in mono if x]), 3) if any(mono) else None
    full = filter_chains(cif, list(chain_ids), os.path.join(TD, f"{tag}_asm.cif"))
    ra = usalign(full, REF_RING, extra="-mm 1")
    ring_tm = round(ra["tm2"], 3) if ra else None
    ring_rmsd = round(ra["rmsd"], 2) if ra else None
    print(f"[{tag}] #{mid} ipTM={iptm} pLDDT={means} monoTM={mono_mean} ringTM={ring_tm} RMSD={ring_rmsd}Å")
    return {"iptm": iptm, "mono_mean": mono_mean, "ring_tm": ring_tm, "ring_rmsd": ring_rmsd}


# ── UNGUIDED baseline (transfer denominator) ──────────────────────────────────────────
print(f"\n[unguided] folding human PCNA homotrimer UNGUIDED (single seed) — minutes…")
uraw = drive(panel.construct_fold_launch_spec("boltz", N),
             lambda r: panel.apply_construct_fold_result(panel.construct_fold_launch_spec("boltz", N), r))
udata = panel._fold_from_result(uraw) if uraw else None
u = assess("unguided", udata) if udata else None
ung_ring = (u or {}).get("ring_tm")

# ── guided by each ring template (single seed) ────────────────────────────────────────
results = []   # (label, tmpl_tm, metrics)
for label, ref, tmpl_tm, nch, rpath in TEMPLATES:
    print(f"\n[guided] human PCNA homotrimer GUIDED by {label} — {nch}-chain ring template — minutes…")
    ref = dict(ref); ref["force"] = False
    gspec = panel.construct_fold_guided_spec("boltz", N, ref)
    _cap["yaml"].clear()
    graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
    gdata = panel._fold_from_result(graw) if graw else None
    gy = next((y for y in _cap["yaml"] if "templates:" in y), "")
    tmpl_chain_lines = [ln for ln in gy.splitlines() if "chain_id" in ln]
    print(f"   [yaml] chain_id line(s): {tmpl_chain_lines}")
    g = assess("guided-" + label.split()[0], gdata) if gdata else None
    results.append((label, tmpl_tm, g))

# ── DUAL-FRAMING REPORT ───────────────────────────────────────────────────────────────
print("\n══ DIVERGENT-RING PROBE — human PCNA homotrimer, single-seed (dual framing) ══")
print(f"  unguided baseline ring-TM (transfer denominator): {ung_ring}")
print(f"  {'template':<32}{'tmpl ringTM':>12}{'monoTM':>8}{'guidedRingTM':>14}{'ipTM':>7}  verdict")
for label, tmpl_tm, g in results:
    if not g:
        print(f"  {label:<32}{str(tmpl_tm):>12}{'  — fold/metric failed'}"); continue
    gr = g["ring_tm"]; verdict = "?"
    if gr is not None and tmpl_tm is not None:
        transfer = (ung_ring is not None and gr - ung_ring > 0.3)
        if gr - tmpl_tm > 0.05:   quat = "UNLOCK (corrected toward human ring beyond template)"
        elif abs(gr - tmpl_tm) <= 0.03: quat = "COPY (≈ template arrangement)"
        else: quat = "between copy/unlock"
        verdict = ("TRANSFER + " if transfer else "no-transfer + ") + quat
    print(f"  {label:<32}{str(tmpl_tm):>12}{str(g['mono_mean']):>8}{str(gr):>14}{str(g['iptm']):>7}  {verdict}")

print("\nNOTES: (1) TRANSFER framing is clean (whole family); UNLOCK-vs-COPY is only resolvable at the")
print("archaeal 1GE8 rung (headroom ~0.10, RMSD-distinguishable). (2) PCNA quaternary structure is")
print("CONSERVED across life (ladder: seq-id 25→100% but ring-TM 0.90→0.99) — this probes")
print("sequence-divergence-with-conserved-quaternary ONLY; a genuinely quaternary-divergent system")
print("is the real continuation (§9). (3) Single-seed probe, NOT a titration.")
