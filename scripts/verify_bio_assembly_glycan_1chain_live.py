"""
Live-verify — glyco-assembly "1 chain" bug fix, end-to-end through the REAL shipped path
(`ToolRouter._run_bio_assembly` → `_normalize_assembly_to_flat_model` → controller.load_model →
build_design_session), against a DEDICATED ChimeraX on :60002 (launched directly so the user's
:60001 session is left untouched). GPU-free.

The bug (9drq, a glycosylated homotrimer): the AU's NAG glycans occupy their OWN chain ids B, C —
which `info chains` (polymer-only) does NOT report. So the copy-rename planner saw only chain A,
assigned copies → B, C (already taken by glycans) → `changechains` rejected → `combine retainIds`
refused the duplicate chain-A copies → fallback left the un-normalized submodel group → the
workbench showed "Active: #1 · 1 chain". Fix: enumerate chains from `info residues` (which reports
non-polymer chains too) so every colliding chain is relabelled to a globally-fresh id.

Checks:
  1. shipped _run_bio_assembly normalizes 9drq to a FLAT model (normalized=True, no normalize_error).
  2. the flat model has 3 DISTINCT polymer (protein) chains — NOT collapsed to 1.
  3. native `color #flat bychain` → 3 distinct colors across the protein copies.
  4. the REAL controller.load_model(flat) enumerates 3 ChainSeqs (the "1 chain" symptom is gone).
  5. build_design_session → ONE ChainDesign (homotrimer) whose members are the 3 copies.
  6. regression: the clean baseline 5hrz still normalizes to chains [A, B, C].

Run: venv/Scripts/python.exe scripts/verify_bio_assembly_glycan_1chain_live.py
"""
import os, sys, re, time, subprocess, urllib.request, urllib.parse
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from chimerax_bridge import find_chimerax, ChimeraXBridge
from session_state import SessionState
from tool_router import ToolRouter
from seq_editor.controller import SequenceEditorController
from variant_model import build_design_session

PORT = 60002
_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _reachable(port):
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/run?" + urllib.parse.urlencode({"command": "version"}), timeout=2)
        return True
    except Exception:
        return False


def main():
    bridge = ChimeraXBridge(port=PORT)
    proc = None
    if not bridge.is_running():
        print(f"[setup] launching dedicated ChimeraX on :{PORT} …")
        proc = subprocess.Popen(
            [find_chimerax(), "--cmd", f"remotecontrol rest start port {PORT}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
        deadline = time.time() + 90
        while time.time() < deadline and not bridge.is_running():
            time.sleep(1.0)
        if not bridge.is_running():
            print(f"[setup] FAILED — :{PORT} never became reachable")
            return 1
        time.sleep(1.0)   # settle
        print("[setup] reachable.")
    else:
        print(f"[setup] reusing ChimeraX already on :{PORT}")

    run = bridge.run_command

    def models_txt():
        return run("info models").get("value") or ""

    def new_model(before):
        b = set(re.findall(r"model id #(\d+)\b", before))
        a = set(re.findall(r"model id #(\d+)\b", models_txt()))
        n = sorted(a - b, key=int)
        return n[-1] if n else None

    def color(spec):
        v = run(f"info atoms {spec} attribute color").get("value") or ""
        m = re.search(r"color\s+([0-9,]+)", v)
        return m.group(1) if m else None

    try:
        # 9drq — the glycosylated homotrimer ("1 chain" bug case) ─────────────────────────
        print("\n=== 9drq (glyco homotrimer — the bug case) ===")
        run("close session")
        before = models_txt()
        run("open 9drq")
        au = new_model(before)
        sess = SessionState()
        sess.add_structure(au, "9drq", metadata={"name": "9drq"})
        router = ToolRouter(bridge=bridge, session=sess)

        res = router._run_bio_assembly({"model_id": au, "assembly_id": 1})
        rec = sess.get_generated_assembly(au) or {}
        flat = str(rec.get("assembly_model_id") or "")
        check("1) shipped _run_bio_assembly normalized 9drq to a flat model (no error)",
              res.success and rec.get("normalized") and not rec.get("normalize_error"),
              f"normalized={rec.get('normalized')} flat=#{flat} err={rec.get('normalize_error')!r}")

        # protein (polymer) chains of the flat model — must be 3 distinct, not collapsed to 1
        fc = run(f"info chains #{flat}").get("value") or ""
        prot_chains = sorted(set(re.findall(rf"#{flat}/(\S+)\s", fc)))
        check("2) flat model has 3 DISTINCT protein chains (not 1)", len(prot_chains) == 3,
              f"protein chains={prot_chains}")

        run(f"color #{flat} bychain")
        cols = {ch: color(f"#{flat}/{ch}:0@CA") or color(f"#{flat}/{ch}@CA") for ch in prot_chains}
        check("3) color bychain gives 3 distinct colors across copies",
              len(set(v for v in cols.values() if v)) == 3, f"{cols}")

        ctrl = SequenceEditorController(run_command=run, fold_fn=lambda *a, **k: {})
        cs = ctrl.load_model(flat)
        chains = sorted(c.chain for c in cs)
        check("4) controller.load_model enumerates 3 chains (the '1 chain' symptom is GONE)",
              len(cs) == 3, f"chains={chains} (n={len(cs)})")

        design = build_design_session(cs, flat)
        cds = list(design.chains.values())
        members = sorted(c for _m, c in cds[0].members) if cds else []
        check("5) build_design_session → ONE ChainDesign (homotrimer) with 3 copy members",
              len(cds) == 1 and len(members) == 3, f"n_cd={len(cds)} members={members}")

        # 5hrz — clean baseline regression ───────────────────────────────────────────────
        print("\n=== 5hrz (clean baseline regression) ===")
        run("close session")
        before = models_txt()
        run("open 5hrz")
        au2 = new_model(before)
        sess2 = SessionState()
        sess2.add_structure(au2, "5hrz", metadata={"name": "5hrz"})
        router2 = ToolRouter(bridge=bridge, session=sess2)
        router2._run_bio_assembly({"model_id": au2, "assembly_id": 1})
        rec2 = sess2.get_generated_assembly(au2) or {}
        check("6) baseline 5hrz still normalizes to chains [A, B, C]",
              rec2.get("normalized") and rec2.get("assembly_chains") == ["A", "B", "C"],
              f"chains={rec2.get('assembly_chains')}")

        run("close session")
    finally:
        if proc is not None:
            print("\n[teardown] closing dedicated ChimeraX …")
            try:
                run("exit")
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass

    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
