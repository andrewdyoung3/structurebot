"""
HIV-1 CA hexamer — CLEAN-TEMPLATE re-run (root-causes the oligomeric-STATE hexamer ANOMALY).

Background (PROJECT_CONTEXT §13, 2026-06-22): in the matched oligomeric-state imposition study the
PENTAMER was DECISIVE (guided by the clean 5-chain 3P05 ASU → ring-TM→3P05 0.964-0.966, 3 seeds, vs a
0.296 unguided floor) but the HEXAMER was ANOMALOUS — guided by 3H47 it reverted to PENTAMER curvature
(ring-TM→3H47 only 0.62, →3P05 0.93-0.95, robust across seeds). DIAGNOSIS: the operational difference
was the TEMPLATE FILE, not the model — the pentamer used 3P05's clean deposited 5-chain ASU (ids A-E),
while the hexamer used the symmetry-EXPANDED assembly file (`3H47-assembly1.cif`, non-standard dashed
ids A,A-2..A-6, generated coords). An anomaly from a known-flawed setup can't be trusted: re-run CLEAN
before it means anything (same don't-conclude-from-an-artifact discipline as the RNase C-swap errored
fold).

THIS RE-RUN: the hexamer arm ONLY (pentamer already decisive — not re-spent), guided by a CLEAN
6-chain hexamer template built in ChimeraX (open 3H47 -> sym assembly 1 copies true -> combine ->
chains remapped A-F -> save; `cache/3H47_hexamer_clean.cif`, VERIFIED: 6 protein chains A-F, no dashes,
real non-overlapping 6-fold ring, 4.7 A ring contacts). Apples-to-apples with the pentamer setup (3P05
template also = protein chains + het chains, raw). Same harvested CA query (3P05 chain A) — NO sequence
confound. Soft (force:false), 3 seeds, dual-decomposition ring-TM vs BOTH 3H47 (native) and 3P05
(other), + unguided floor (1 seed).

VERDICTS:
  CLEAN -> hexamer forms (ring-TM->3H47 >> ->3P05, inverting the anomaly): ARTIFACT CONFIRMED — the
    dashed sym-expanded template was the cause; headline extends to "the SAME CA sequence reaches BOTH
    oligomeric states via matching templates", imposition both directions, no sequence confound.
  CLEAN -> still reverts to pentamer: artifact hypothesis FAILS — run the hard-steer arm separately
    (force:true + threshold) to separate "soft template insufficient" from "the model genuinely
    resists the hexamer at 6 chains even properly templated" (a real, bounded finding).

Run (BACKGROUND, hours; raise BOLTZ_TIMEOUT — a 6-mer ~1260 res blows the 1800s default):
  BOLTZ_TIMEOUT=9000 venv/Scripts/python.exe scripts/verify_hiv_hexamer_cleantemplate_multiseed_live.py
Requires a ChimeraX REST server already on :60001 (GUI desktop, or `--nogui` REST).
"""
import os, re, sys, json, types, tempfile, shlex, urllib.request
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

N = 6
GUIDED_SEEDS = [0, 1, 2]
CLEAN_HEX = os.path.abspath("cache/3H47_hexamer_clean.cif")   # the ChimeraX-built clean 6-chain ring
RESULTS_JSON = os.path.abspath("hiv_hexamer_clean_results.json")
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="hiv_hexclean_")
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


_3TO1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
         'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
         'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'MSE': 'M',
         'SEC': 'U', 'PYL': 'O'}


def harvest_seq(pdb, chain="A"):
    """Harvest the modelled chain-A sequence DIRECTLY from the deposited mmCIF (auth_asym_id ==
    chain, ATOM group, ordered by auth_seq_id+ins). NOTE: the committed imposition harness read this
    via a ChimeraX `runscript`+print, but a `--nogui` REST server does NOT return runscript print
    output (it goes to the process stdout) — so we parse the CIF, which is deterministic and
    GUI/nogui-independent (verified identical: 3P05 chain A = 210-aa HIV-1 CA)."""
    path = fetch(pdb.upper())
    cols, res = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("_atom_site."): cols.append(ln.strip().split(".")[1])
            elif ln.startswith("ATOM"):
                p = ln.split()
                def g(n):
                    i = next((k for k, c in enumerate(cols) if c == n), None)
                    return p[i] if i is not None and i < len(p) else None
                if g("auth_asym_id") != chain: continue
                comp = g("label_comp_id")
                if comp not in _3TO1: continue
                res[(int(g("auth_seq_id")), g("pdbx_PDB_ins_code") or "")] = _3TO1[comp]
    return "".join(res[k] for k in sorted(res))


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


# ── template + refs ──
if not os.path.isfile(CLEAN_HEX):
    print(f"[abort] clean hexamer template missing: {CLEAN_HEX}"); sys.exit(2)
hex_chains = sorted([c for c, n in chain_ca(CLEAN_HEX).items() if n > 60])
if len(hex_chains) != N:
    print(f"[abort] clean template has {len(hex_chains)} protein chains, need {N}: {hex_chains}"); sys.exit(2)
print(f"[template] CLEAN hexamer {Path(CLEAN_HEX).name} — protein chains {hex_chains}")

run("close session")
ca = harvest_seq("3p05", "A")
print(f"[harvest] HIV-1 CA (3P05 chain A) {len(ca)} aa  [SAME query as the pentamer arm — no seq confound]")
if len(ca) < 120:
    print("[abort] CA sequence too short"); sys.exit(2)

REF = {}
REF["3H47"] = filter_chains(CLEAN_HEX, hex_chains, os.path.join(TD, "3H47_ring.cif"))          # native ring
p05 = fetch("3P05")
p05_prot = sorted([c for c, n in chain_ca(p05).items() if n > 60])[:5]
REF["3P05"] = filter_chains(p05, p05_prot, os.path.join(TD, "3P05_ring.cif"))                  # other ring
MONO_REF = filter_chains(CLEAN_HEX, [hex_chains[0]], os.path.join(TD, "mono_ref.cif"))         # 3H47 protomer
print(f"[refs] native 3H47 ring={len(hex_chains)} chains | other 3P05 ring={len(p05_prot)} chains | mono=3H47 chain {hex_chains[0]}")

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


def drive(spec, on_apply, timeout=18_000_000):   # 5h Qt-wait — must OUTLAST BOLTZ_TIMEOUT (the 1260-res
    cap = []                                      # hexamer on the --no_kernels path runs >2.5h/fold)
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"], pres,
                               confidence=spec.get("confidence", "high"), on_result=cap.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if cap: on_apply(cap[0])
    return cap[0] if cap else None


def assess(data):
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
    an = usalign(full, REF["3H47"], extra="-mm 1"); ao = usalign(full, REF["3P05"], extra="-mm 1")
    return {"iptm": round(iptm, 3) if iptm else None, "mono": mono_mean,
            "ring_3H47": round(an["tm2"], 3) if an else None,
            "ring_3P05": round(ao["tm2"], 3) if ao else None,
            "n_chains": len(chain_ids)}


panel._add_sequence_construct("hivca", ca)
results = {"unguided": None, "guided": []}

def save_results():
    """Persist after EVERY fold so an interruption (these folds are >2.5h each) keeps what landed."""
    try:
        json.dump({"template": CLEAN_HEX, "query_len": len(ca), "results": results},
                  open(RESULTS_JSON, "w"), indent=2, default=str)
    except Exception as exc:
        print(f"[warn] results save failed: {exc}")
    sys.stdout.flush()


# GUIDED seeds FIRST (the decisive arm: does the CLEAN template form the hexamer? ->3H47 vs ->3P05),
# then the unguided floor LAST — so the result that answers the artifact lands first and survives a stop.
for s in GUIDED_SEEDS:
    print(f"[GUIDED] N={N} by CLEAN 3H47 hexamer (soft) seed={s} ...", flush=True)
    _cur["seed"] = s
    ref = {"path": CLEAN_HEX, "label": "3H47_clean", "force": False}
    gspec = panel.construct_fold_guided_spec("boltz", N, dict(ref))
    graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
    gm = assess(panel._fold_from_result(graw)) if graw else None
    results["guided"].append((s, gm))
    print(f"   seed {s}: monoTM={gm['mono'] if gm else None} ringTM->3H47={gm['ring_3H47'] if gm else None} "
          f"->3P05={gm['ring_3P05'] if gm else None} ipTM={gm['iptm'] if gm else None}", flush=True)
    save_results()

print(f"\n[UNGUIDED] N={N} hexamer (floor, 1 seed, ~{N*len(ca)} res) ...", flush=True)
_cur["seed"] = 0
uspec = panel.construct_fold_launch_spec("boltz", N)
uraw = drive(uspec, lambda r: panel.apply_construct_fold_result(panel.construct_fold_launch_spec("boltz", N), r))
um = assess(panel._fold_from_result(uraw)) if uraw else None
results["unguided"] = um
print(f"   unguided: monoTM={um['mono'] if um else None} ringTM->3H47={um['ring_3H47'] if um else None} "
      f"->3P05={um['ring_3P05'] if um else None} ipTM={um['iptm'] if um else None}", flush=True)
save_results()
print(f"\n[saved] {RESULTS_JSON}")


def spread(vals):
    v = [x for x in vals if x is not None]
    return (min(v), round(sum(v)/len(v), 3), max(v)) if v else (None, None, None)

print("\n══ HIV CA HEXAMER — CLEAN-TEMPLATE re-run (root-causing the anomaly) ══")
print("  same CA query, soft, guided by the CLEAN 6-chain 3H47 ring; does the hexamer now form?\n")
um = results["unguided"]; gm = results["guided"]
ufloor = um["ring_3H47"] if um else None
gvals = [m["ring_3H47"] for _, m in gm if m]
lo, mn, hi = spread(gvals)
print(f"  unguided floor: ring-TM->3H47={ufloor}  (->3P05 {um['ring_3P05'] if um else None}, mono {um['mono'] if um else None})")
for s, m in gm:
    if m:
        print(f"  guided seed {s}: ring-TM->3H47={m['ring_3H47']}  ->3P05={m['ring_3P05']}  "
              f"monoTM={m['mono']}  ipTM={m['iptm']}  (n_chains={m['n_chains']})")
    else:
        print(f"  guided seed {s}: FAILED")
gap = (mn - ufloor) if (mn is not None and ufloor is not None) else None
print(f"  -> guided ring-TM->3H47 spread: min {lo} mean {mn} max {hi}  |  guided-unguided gap: {gap}")

# Verdict: does the clean template now form the hexamer AND beat the pentamer arrangement?
formed = all(m and m["ring_3H47"] is not None and m["ring_3P05"] is not None and
             (m["ring_3H47"] - m["ring_3P05"]) > 0.05 for _, m in gm)
strong = mn is not None and mn > 0.85
if formed and strong:
    print("\n  VERDICT: ARTIFACT CONFIRMED — clean template forms the HEXAMER (ring-TM->3H47 >> ->3P05),")
    print("  inverting the dashed-template anomaly. The same CA reaches BOTH states via matching templates.")
elif gm and all(m and (m["ring_3P05"] - m["ring_3H47"]) > 0.1 for _, m in gm):
    print("\n  VERDICT: STILL REVERTS to pentamer curvature even with the CLEAN template — the artifact")
    print("  hypothesis FAILS. Run the HARD-STEER arm (force:true) to separate 'soft insufficient' from")
    print("  'the model genuinely resists the hexamer at 6 chains even properly templated'.")
else:
    print("\n  VERDICT: MIXED / partial — inspect the per-seed rows; consider hard-steer to disambiguate.")

print("\nDONE — HIV hexamer clean-template multi-seed complete.")
