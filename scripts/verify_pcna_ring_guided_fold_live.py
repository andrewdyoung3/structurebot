"""
Live-verify — PCNA-HOMOTRIMER RING: guided vs unguided, with a TWO-TIER structTM decomposition.

The wiring gate (verify_pcna_ring_yaml_gate.py) proved the template steers all 3 ring chains
(chain_id: [A, B, C]). This is the EXPERIMENT: fold the PCNA homotrimer UNGUIDED and GUIDED-by-1AXC
(both as the 3-chain assembly, apples-to-apples) and report each on THREE axes so the result can
separate the three distinct outcomes:

  • per-chain MONOMER structTM  — each predicted PCNA chain vs 1AXC's PCNA monomer
        → is each SUBUNIT folded correctly?
  • whole-trimer RING structTM  — the assembled predicted trimer vs 1AXC's PCNA C3 ring (US-align -mm)
        → did the three subunits ARRANGE into the ring?
  • ipTM                         — Boltz's inter-chain confidence proxy (unguided baseline was ~0.16)

Only BOTH structTM tiers can tell apart:
  - BAD MONOMERS                 (monomer-TM low)               — the fold itself failed
  - GOOD MONOMERS / NO RING      (monomer-TM high, ring-TM low) — the legitimate Boltz-attribution
                                   outcome we flagged: per-chain templating gives each PCNA its fold
                                   but does NOT impose the inter-chain C3 geometry (template_id omitted)
  - FULL RING                    (both high)                    — the ring reproduced

A single whole-trimer number cannot distinguish "bad monomers" from "good monomers that didn't
assemble" — and that distinction IS the question. References are 1AXC's PCNA chains ONLY (the p21
peptide chains are excluded), built live from the opened 1AXC.

Also LIVE-verifies (real run): the emitted Boltz YAML still carries `chain_id: [A, B, C]` AFTER
`_translate_template_paths` (the gate showed it pre-translation; this confirms it survives on the
real WSL path).

Run: venv/Scripts/python.exe scripts/verify_pcna_ring_guided_fold_live.py    (two homotrimer folds — many minutes)
"""
import os, re, sys, types, tempfile, shlex
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DEVIATION_FLOOR_N", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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

bridge = ChimeraXBridge(port=60001)
try:
    bridge.start(timeout=120)
    print("[chimerax] REST server up on :60001")
except Exception as exc:
    print(f"[abort] could not start ChimeraX: {exc}"); sys.exit(2)
run = bridge.run_command


def models():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _runscript(script: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    try:
        return run(f'runscript "{path}"').get("value") or ""
    finally:
        try: os.unlink(path)
        except OSError: pass


def chain_sequences(model):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        "        for ch in sorted(set(r.chain_id for r in m.residues)):\n"
        "            rs = [r for r in m.residues if r.chain_id==ch and r.polymer_type==r.PT_AMINO]\n"
        "            rs.sort(key=lambda r: r.number)\n"
        "            seq = ''.join((r.one_letter_code or 'X') for r in rs)\n"
        "            if seq: print('SEQ', ch, seq)\n"
        "        break\n")
    seqs = {}
    for line in out.splitlines():
        p = line.strip().split()
        if len(p) == 3 and p[0] == "SEQ":
            seqs[p[1]] = p[2]
    return seqs


def save_full(model_id, outpath):
    """Save the WHOLE model #model_id to a CIF (Windows path). Returns the path."""
    run(f'save "{Path(outpath).as_posix()}" models #{model_id}')
    return outpath


def filter_chains(src, wanted, dst):
    """Write a minimal CIF holding ONLY *wanted* chains of *src* — by copying the source's exact
    `_atom_site` loop header and keeping rows whose auth/label_asym_id matches. ChimeraX's CIF
    writer ignores chain atomspecs (saves the whole model), so this is the reliable subsetter.
    Returns (dst, n_atom_rows)."""
    wanted = set(wanted)
    lines = open(src).read().splitlines()
    hdr, data, i, n = [], [], 0, len(lines)
    while i < n:
        if lines[i].startswith("_atom_site."):
            while i < n and lines[i].startswith("_atom_site."):
                hdr.append(lines[i].strip()); i += 1
            cand = [k for k, c in enumerate(hdr) if c.endswith("auth_asym_id")] or \
                   [k for k, c in enumerate(hdr) if c.endswith("label_asym_id")]
            idx = cand[0] if cand else None
            while i < n and lines[i].startswith(("ATOM", "HETATM")):
                p = lines[i].split()
                if idx is not None and idx < len(p) and p[idx] in wanted:
                    data.append(lines[i])
                i += 1
            break
        i += 1
    with open(dst, "w") as f:
        f.write("data_filtered\nloop_\n" + "\n".join(hdr) + "\n" + "\n".join(data) + "\n")
    return dst, len(data)


# ── US-align (LOCAL-ONLY WSL binary) — returns ref-normalized TM (tm2) ─────────────────
_wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
_USEXE = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")


def _parse_us(stdout):
    """Robust parse of US-align `-outfmt 2`. Scans for ANY line whose tab-split has the 11 data
    columns with float-parseable TM/RMSD — NOT just header+1: in `-mm`/`-ter 0` mode US-align
    wraps the long chain-spec field with embedded NEWLINES, so the real data values land on a
    later fragment line (`,C<TAB>ref<TAB>0.99...`). Skips the header row itself."""
    for ln in stdout.splitlines():
        if ln.startswith("#PDBchain1"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 11:
            try:
                return {"tm1": float(parts[2]), "tm2": float(parts[3]),
                        "rmsd": float(parts[4]), "l1": int(parts[8]), "l2": int(parts[9]),
                        "lali": int(parts[10])}
            except ValueError:
                continue
    return None


def usalign(query_path, ref_path, extra=""):
    """US-align query→ref → parsed dict (tm1 query-norm, tm2 ref-norm). None on failure."""
    if not (query_path and ref_path and os.path.isfile(query_path) and os.path.isfile(ref_path)):
        return None
    q = _wsl.translate_path(os.path.abspath(query_path))
    r = _wsl.translate_path(os.path.abspath(ref_path))
    cmd = f"{shlex.quote(_USEXE)} {shlex.quote(q)} {shlex.quote(r)} -outfmt 2 {extra}".strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
    if not res.get("ok"):
        print(f"[usalign] FAILED: {res.get('error') or (res.get('stderr','') or '')[:160]}")
        return None
    return _parse_us(res.get("stdout", ""))


class ScriptedPresenter(Presenter):
    def __init__(self):
        self.warnings = []; self.confirm_calls = []; self.confirm_answer = "proceed"
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
    def confirm(self, confidence):
        self.confirm_calls.append(confidence); return self.confirm_answer
    def ask_edit(self, original): return list(original)
    def ask_yes_no(self, q, default="y"): return False


# ── harvest PCNA + build PCNA-ONLY references (monomer + ring) from 1AXC ───────────────
run("close session")
before = set(models())
run("open 1axc")
AXC = (sorted(set(models()) - before, key=int) or ["1"])[-1]
seqs = chain_sequences(AXC)
pcna_chains = sorted([c for c, s in seqs.items() if len(s) > 100])     # PCNA ≈ 249 aa; p21 peptide is short
print(f"[1axc] chains: {{ {', '.join(f'{c}:{len(s)}aa' for c, s in sorted(seqs.items()))} }}  → PCNA = {pcna_chains}")
if len(pcna_chains) < 3:
    print(f"[abort] expected ≥3 PCNA chains in 1AXC, found {pcna_chains}"); sys.exit(2)
pcna = seqs[pcna_chains[0]]
tmpdir = tempfile.mkdtemp(prefix="pcna_ring_")
_axc_full = save_full(AXC, os.path.join(tmpdir, "1axc_full.cif"))
run(f"close #{AXC}")
REF_MONO, _nm = filter_chains(_axc_full, [pcna_chains[0]], os.path.join(tmpdir, "1axc_pcna_monomer.cif"))
REF_RING, _nr = filter_chains(_axc_full, pcna_chains[:3], os.path.join(tmpdir, "1axc_pcna_ring.cif"))
print(f"[ref] PCNA {len(pcna)} aa | monomer ref={_nm} atoms, ring ref={_nr} atoms (p21 chains excluded)")

# ── app plumbing ──────────────────────────────────────────────────────────────────────
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
router = ToolRouter(bridge, session)
host = types.SimpleNamespace(
    bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None, _log_exchange=lambda *a, **k: None)
engine = RequestEngine(host)
pres = ScriptedPresenter()
panel = VariantWorkbenchPanel(ctrl, session=session)
if not panel._fold_engine_availability().get("boltz"):
    print("[abort] Boltz env unavailable — the ring fold needs it."); sys.exit(2)

# ── capture the REAL emitted YAML (post _translate_template_paths) ─────────────────────
_orig_build_yaml = BoltzBridge._build_yaml
_captured = {"yaml": []}
def _capturing_build_yaml(chains, templates=None):
    y = _orig_build_yaml(chains, templates)
    _captured["yaml"].append(y)
    return y
BoltzBridge._build_yaml = staticmethod(_capturing_build_yaml)


def drive(spec, on_apply, timeout=3_600_000):
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
    captured = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if captured:
        on_apply(captured[0])
    return captured[0] if captured else None


N = 3                      # PCNA homotrimer
TEMPLATE = "1AXC"
checks = []
panel._add_sequence_construct("pcna", pcna)
cd = next(iter(panel._design.chains.values()))


def assess(tag, data):
    """Two-tier structTM + ipTM for one fold result. Returns the metrics dict + prints them."""
    mid = str(data.get("new_model_id") or data.get("model_id"))
    cif = data.get("cif_path") or data.get("pdb_path")
    if not (cif and os.path.isfile(cif)):
        print(f"[{tag}] no predicted CIF on disk — structTM decomposition skipped")
        return {"model_id": mid, "iptm": data.get("iptm"), "chain_means": {},
                "mono_tms": {}, "mono_mean": None, "ring_tm": None}
    iptm = data.get("iptm")
    pbc = data.get("plddt_by_chain") or {}
    # per-chain mean pLDDT (monomer confidence)
    chain_means = {ch: (round(sum(v.values()) / len(v), 1) if v else None) for ch, v in pbc.items()}
    chain_ids = data.get("chain_ids") or sorted(pbc.keys()) or ["A", "B", "C"]
    # per-chain MONOMER structTM: filter each predicted chain, align to the PCNA monomer ref
    mono_tms = {}
    for ch in chain_ids:
        cf, _ = filter_chains(cif, [ch], os.path.join(tmpdir, f"{tag}_chain_{ch}.cif"))
        p = usalign(cf, REF_MONO)                       # default monomer alignment
        mono_tms[ch] = round(p["tm2"], 3) if p else None
    mono_vals = [t for t in mono_tms.values() if t is not None]
    mono_mean = round(sum(mono_vals) / len(mono_vals), 3) if mono_vals else None
    # whole-trimer RING structTM: the full predicted assembly vs the 3-chain PCNA ring (-mm complex)
    full, _ = filter_chains(cif, list(chain_ids), os.path.join(tmpdir, f"{tag}_assembly.cif"))
    rp = usalign(full, REF_RING, extra="-mm 1")          # multimer complex alignment
    ring_tm = round(rp["tm2"], 3) if rp else None
    ring_rmsd = round(rp["rmsd"], 2) if rp else None
    ring_lali = (rp["lali"], rp["l2"]) if rp else None
    print(f"[{tag}] model #{mid}  ipTM={iptm}  per-chain pLDDT={chain_means}")
    print(f"[{tag}]   MONOMER structTM (vs 1AXC PCNA monomer): per-chain={mono_tms} mean={mono_mean}")
    print(f"[{tag}]   RING    structTM (vs 1AXC PCNA C3 ring, -mm): {ring_tm}  RMSD={ring_rmsd}Å  aligned={ring_lali}")
    return {"model_id": mid, "iptm": iptm, "chain_means": chain_means,
            "mono_tms": mono_tms, "mono_mean": mono_mean, "ring_tm": ring_tm,
            "ring_rmsd": ring_rmsd, "ring_lali": ring_lali}


# ── UNGUIDED homotrimer baseline ──────────────────────────────────────────────────────
print(f"\n[unguided] folding PCNA homotrimer UNGUIDED (Boltz, MSA-free, {N} chains) — minutes…")
uspec = panel.construct_fold_launch_spec("boltz", N)
uraw = drive(uspec, lambda r: panel.apply_construct_fold_result(uspec, r))
udata = panel._fold_from_result(uraw) if uraw else None     # the FOLD STEP data (not the pipeline wrapper)
checks.append(("(0) unguided homotrimer folded (model + cif)", bool(udata and (udata.get("cif_path") or udata.get("pdb_path")))))
u = assess("unguided", udata) if udata else None

# ── GUIDED-SOFT homotrimer by 1AXC ────────────────────────────────────────────────────
print(f"\n[guided] folding PCNA homotrimer GUIDED-SOFT by {TEMPLATE} ({N} chains) — minutes…")
ref = {"pdb_id": TEMPLATE, "label": TEMPLATE, "force": False}
gspec = panel.construct_fold_guided_spec("boltz", N, ref)
_captured["yaml"].clear()
graw = drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
gdata = panel._fold_from_result(graw) if graw else None      # the FOLD STEP data (not the pipeline wrapper)
checks.append(("(a) guided homotrimer fold RAN (real model)", bool(gdata and (gdata.get("new_model_id") or gdata.get("model_id")))))

# ── LIVE YAML check on the REAL run (post translate) ──────────────────────────────────
guided_yaml = next((y for y in _captured["yaml"] if "templates:" in y), "")
print("\n--- REAL guided-run YAML templates block (post _translate_template_paths) ---")
for ln in guided_yaml.splitlines():
    if "template" in ln or "chain_id" in ln or ln.strip().startswith(("- cif", "- pdb")):
        print("   " + ln)
yaml_ok = "chain_id: [A, B, C]" in guided_yaml
path_translated = bool(re.search(r"(cif|pdb):\s*/mnt/", guided_yaml))      # WSL path survived
checks.append(("(b) real-run YAML still carries chain_id: [A, B, C] after path translation", yaml_ok))
checks.append(("(c) template path was translated to a WSL /mnt path on the real run", path_translated))

g = assess("guided", gdata) if gdata else None

# ── DECOMPOSED COMPARISON + outcome classification ────────────────────────────────────
def classify(m):
    if m is None or m.get("mono_mean") is None or m.get("ring_tm") is None:
        return "n/a (a metric failed)"
    mono, ring = m["mono_mean"], m["ring_tm"]
    if mono < 0.5:
        return "BAD MONOMERS (subunit fold failed)"
    if ring >= 0.7:
        return "FULL RING (subunits folded AND assembled)"
    if ring < 0.5:
        return "GOOD MONOMERS / NO RING (folded subunits, no C3 assembly — the flagged outcome)"
    return "PARTIAL (good monomers, partial assembly)"

print("\n══ DECOMPOSED RESULT — guided vs unguided, apples-to-apples (both PCNA homotrimer) ══")
print(f"  {'axis':<34}{'unguided':>14}{'guided':>14}")
def row(name, uk, gk):
    uv = (u or {}).get(uk); gv = (g or {}).get(gk)
    print(f"  {name:<34}{str(uv):>14}{str(gv):>14}")
row("ipTM (inter-chain confidence)", "iptm", "iptm")
row("MONOMER structTM (mean A,B,C)", "mono_mean", "mono_mean")
row("RING structTM (-mm, whole trimer)", "ring_tm", "ring_tm")
row("RING RMSD (Å)", "ring_rmsd", "ring_rmsd")
row("RING aligned (Lali/L)", "ring_lali", "ring_lali")
print(f"\n  unguided outcome: {classify(u)}")
print(f"  guided   outcome: {classify(g)}")

# the decomposition is the gate: BOTH folds must yield BOTH structTM tiers (else the readout is blind)
checks.append(("(d) BOTH tiers computed for BOTH folds (monomer-TM + ring-TM, guided & unguided)",
               all(x is not None for x in ((u or {}).get("mono_mean"), (u or {}).get("ring_tm"),
                                           (g or {}).get("mono_mean"), (g or {}).get("ring_tm")))))

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nNOTE: the gate (mechanism) is the structTM DECOMPOSITION running on both folds; the SCIENCE")
print("readout (did guiding move ring-TM?) is reported honestly — a 'good monomers / no ring' guided")
print("result is a legitimate Boltz multimer-assembly finding, NOT a wiring failure.")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
