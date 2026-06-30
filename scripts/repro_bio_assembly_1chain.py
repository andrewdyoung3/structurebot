"""
Repro — bio-assembly "1 chain" bug. Reproduce 9drq (the user's glycosylated homotrimer "B")
and panel peers through the REAL shipped _run_bio_assembly, in a DEDICATED ChimeraX on :60002
(launched directly, NOT via bridge.start(), so the user's :60001 session is left untouched).

For each PDB we capture, end to end:
  - raw `info chains #<group>` BEFORE normalization (the addressing form `sym` actually emits)
  - what the shipped Fix A produced: normalized? assembly_chains? flat model id? note/error?
  - raw `info models` + `info chains #<flat>` AFTER
This tells us whether the shipped code normalizes 9drq (→ user's live session is just STALE)
or genuinely fails on it (→ real code bug, and exactly where).

Run: venv/Scripts/python.exe scripts/repro_bio_assembly_1chain.py
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
from tool_router import ToolRouter, _parse_submodel_chains

PORT = 60002
PANEL = [
    ("9drq", "user's PDB B — glyco homotrimer (NAG)"),
    ("5hrz", "designed homotrimer — KNOWN-GOOD baseline"),
    ("1ruz", "influenza HA — glyco trimer (HA1+HA2)"),
    ("4tvp", "HIV Env SOSIP — glyco trimer (NAG)"),
]


def _reachable(port):
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/run?" + urllib.parse.urlencode({"command": "version"}),
            timeout=2)
        return True
    except Exception:
        return False


def launch_dedicated():
    exe = find_chimerax()
    if _reachable(PORT):
        print(f"[setup] :{PORT} already reachable — reusing")
        return None
    print(f"[setup] launching dedicated ChimeraX on :{PORT} …")
    proc = subprocess.Popen(
        [exe, "--cmd", f"remotecontrol rest start port {PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
    deadline = time.time() + 60
    while time.time() < deadline:
        if _reachable(PORT):
            print("[setup] reachable.")
            return proc
        time.sleep(1.0)
    raise TimeoutError(f":{PORT} never came up")


def main():
    proc = launch_dedicated()
    bridge = ChimeraXBridge(port=PORT)
    run = bridge.run_command

    def models_txt():
        return run("info models").get("value") or ""

    def new_model(before):
        b = set(re.findall(r"model id #(\d+)\b", before))
        a = set(re.findall(r"model id #(\d+)\b", models_txt()))
        n = sorted(a - b, key=int)
        return n[-1] if n else None

    try:
        for pdb, note in PANEL:
            print("\n" + "=" * 78)
            print(f"### {pdb}  —  {note}")
            print("=" * 78)
            run("close session")
            before = models_txt()
            run(f"open {pdb}")
            au = new_model(before)
            if not au:
                print("  [SKIP] open produced no model"); continue

            # Pre-build: what does `sym ... copies true` emit as addressing? (no normalization yet)
            run(f"sym #{au} assembly 1 copies true")
            grp = new_model(before + f" model id #{au} ")
            grp_chains_raw = run(f"info chains #{grp}").get("value") or "" if grp else ""
            parsed = _parse_submodel_chains(grp_chains_raw, str(grp)) if grp else []
            print(f"  AU=#{au}  sym-group=#{grp}")
            print(f"  raw `info chains #{grp}`:")
            for ln in (grp_chains_raw.strip().splitlines() or ["<empty>"]):
                print(f"      {ln}")
            print(f"  _parse_submodel_chains -> {parsed}")
            run("close session")   # reset; rebuild cleanly through the shipped path

            # Now the REAL shipped path
            before = models_txt()
            run(f"open {pdb}")
            au = new_model(before)
            sess = SessionState()
            sess.add_structure(au, pdb, metadata={"name": pdb})
            router = ToolRouter(bridge=bridge, session=sess)
            res = router._run_bio_assembly({"model_id": au, "assembly_id": 1})
            rec = sess.get_generated_assembly(au) or {}
            flat = str(rec.get("assembly_model_id") or "")
            print(f"  shipped _run_bio_assembly: success={res.success}"
                  + (f" error={res.error!r}" if res.error else ""))
            print(f"    normalized={rec.get('normalized')} "
                  f"assembly_chains={rec.get('assembly_chains')} "
                  f"flat=#{flat} group=#{rec.get('group_model_id')}")
            if rec.get("note") or rec.get("normalize_error"):
                print(f"    note={rec.get('note')!r} normalize_error={rec.get('normalize_error')!r}")
            print(f"  AFTER `info models`:")
            for ln in (models_txt().strip().splitlines()):
                print(f"      {ln}")
            if flat:
                fc = run(f"info chains #{flat}").get("value") or ""
                print(f"  `info chains #{flat}` (the flat model):")
                for ln in (fc.strip().splitlines() or ["<empty>"]):
                    print(f"      {ln}")
    finally:
        run("close session")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
