"""
Live-verify (FAST, no GPU) — the PER-CHAIN TEMPLATE WIRING GATE for the PCNA-homotrimer ring test.

Settles the prerequisite the relay flagged: does a multi-member (homo-oligomer) guided fold emit a
template that steers ALL ring chains, or collapse to a single `chain_id: A`? Read of the code says
the default is the WHOLE cd block as a flow-list; this PROVES it end-to-end on the ACTUAL construct
by running the real spec→resolve→YAML path the Boltz bridge takes — WITHOUT a GPU fold.

  construct = PCNA (1AXC chain A) declared as a homotrimer (1 cd × copy_count 3 → chains A,B,C)
  template = 1AXC (the PCNA ring itself — the homologous guide)

  1. Build the GUIDED spec at n_copies=3 → its template entry's chain_id must be ["A","B","C"].
  2. Resolve templates via the REAL router (pdb_id → mmCIF), then emit the REAL bridge YAML.
  3. Assert the emitted YAML carries `chain_id: [A, B, C]` AND 3 protein sequence chains.

This is the gate that must pass BEFORE the (long, GPU) guided-vs-unguided ring fold is trustworthy:
if the YAML only templated one chain, "no ring" would be a wiring bug, not a Boltz finding.

Run: venv/Scripts/python.exe scripts/verify_pcna_ring_yaml_gate.py
"""
import os, re, sys, tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PySide6 import QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from tool_router import ToolRouter
from boltz_bridge import BoltzBridge
from variant_workbench import VariantWorkbenchPanel

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


# ── harvest the PCNA sequence from 1AXC ───────────────────────────────────────────────
run("close session")
before = set(models())
run("open 1axc")
AXC = (sorted(set(models()) - before, key=int) or ["1"])[-1]
pcna = chain_sequences(AXC).get("A", "")
run(f"close #{AXC}")
if not pcna:
    print("[abort] could not harvest the PCNA sequence from 1AXC"); sys.exit(2)
print(f"[harvest] PCNA {len(pcna)} aa")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
router = ToolRouter(bridge, session)
panel = VariantWorkbenchPanel(ctrl, session=session)

checks = []

# ── 1) GUIDED spec at n_copies=3 → template chain_id is the WHOLE block ───────────────
panel._add_sequence_construct("pcna", pcna)
N = 3
ref = {"pdb_id": "1AXC", "label": "1AXC", "force": False}
spec = panel.construct_fold_guided_spec("boltz", N, ref)
if spec is None:
    print("[abort] guided spec was None"); sys.exit(2)
ti = spec["tool_inputs"]
chains = ti.get("chains") or []
entry = (ti.get("templates") or [{}])[0]
print(f"[spec] fold chains = {[c['id'] for c in chains]}")
print(f"[spec] template entry chain_id = {entry.get('chain_id')!r}  template_id = {entry.get('template_id')!r}")
checks.append(("(1a) construct folds as a 3-chain homotrimer (A,B,C)",
               [c["id"] for c in chains] == ["A", "B", "C"]))
checks.append(("(1b) template chain_id is the WHOLE ring block ['A','B','C'] (not collapsed to 'A')",
               entry.get("chain_id") == ["A", "B", "C"]))

# ── 2) resolve templates via the REAL router, emit the REAL bridge YAML ────────────────
resolved, terr = router._resolve_boltz_templates(ti.get("templates"))
if terr:
    print(f"[abort] template resolve error: {terr}"); sys.exit(2)
print(f"[resolve] resolved entry keys = {sorted(resolved[0].keys())}")
yaml = BoltzBridge._build_yaml(chains, resolved)
print("\n--- EMITTED BOLTZ YAML (the actual bridge output) ---")
print(yaml)
print("--- end YAML ---\n")

# ── 3) assert the emitted YAML templates every ring chain ─────────────────────────────
n_seq_chains = yaml.count("- protein:")
has_flowlist = "chain_id: [A, B, C]" in yaml
checks.append(("(2) emitted YAML has 3 protein sequence chains", n_seq_chains == 3))
checks.append(("(3) emitted YAML carries `chain_id: [A, B, C]` (one template steers all 3 chains)",
               has_flowlist))

print("--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
sys.exit(0 if all(ok for _, ok in checks) else 1)
