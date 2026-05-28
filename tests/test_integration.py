"""
tests/test_integration.py
--------------------------
End-to-end integration test: full stack against 1HSG (HIV protease + MK1).

What it tests
-------------
Phase A — Translation only (no ChimeraX needed):
  Verify that each of the 5 prompts produces reasonable ChimeraX commands
  without the model hallucinating incorrect syntax.

Phase B — Execution (ChimeraX must be running with REST server):
  Execute each set of commands and verify no errors are returned.

Prerequisites
-------------
  1. venv activated:  .\\venv\\Scripts\\Activate.ps1
  2. Internet access (to fetch 1HSG from RCSB and RCSB metadata)
  3. For Phase B: ChimeraX running with:
       remotecontrol rest start port 60001

Run:
    python tests/test_integration.py               # translation only
    python tests/test_integration.py --execute     # translation + execution
    python tests/test_integration.py --start       # start ChimeraX then execute
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from the project root even when run from tests/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # ensure relative paths (chimerax_commands.md etc.) resolve

# ── Load env ──────────────────────────────────────────────────────────────────
import config
config.load_env_file()

# ── UTF-8 on Windows ──────────────────────────────────────────────────────────
# Guard: only replace sys.stdout when running as a standalone script, never
# during pytest collection (replacing stdout at module level breaks pytest's
# capture plugin and causes "ValueError: I/O operation on closed file").
if sys.platform == "win32" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from chimerax_bridge import ChimeraXBridge
from translator import CommandTranslator
from session_state import SessionState

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg: str) -> None: print(f"  {RED}✗{RESET}  {msg}")
def info(msg: str) -> None: print(f"  {CYAN}·{RESET}  {msg}")
def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * 60)


# ════════════════════════════════════════════════════════════════════════════════
# Test definitions
# Each entry: (id, nl_prompt, assertion_fn, description)
# assertion_fn(commands) → (passed: bool, detail: str)
# ════════════════════════════════════════════════════════════════════════════════

def _any_contains(commands: List[str], *substrings: str) -> bool:
    return any(all(s.lower() in c.lower() for s in substrings) for c in commands)

def assert_open_ribbon(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_open = _any_contains(commands, "open", "1HSG") or _any_contains(commands, "open", "1hsg")
    has_cartoon = _any_contains(commands, "cartoon")
    if not has_open:
        return False, f"Expected 'open 1HSG' — got: {commands}"
    if not has_cartoon:
        return False, f"Expected 'cartoon' — got: {commands}"
    return True, f"open + cartoon ✓  ({len(commands)} command(s))"

def assert_color_bychain(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    if _any_contains(commands, "bychain"):
        return True, "'color bychain' found ✓"
    return False, f"Expected 'bychain' in commands — got: {commands}"

def assert_ligand_sphere(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_sphere = _any_contains(commands, "sphere")
    has_mk1    = _any_contains(commands, "MK1") or _any_contains(commands, "mk1")
    has_byelement = _any_contains(commands, "byelement")
    issues = []
    if not has_sphere:
        issues.append("'sphere' style not found")
    if not has_mk1:
        issues.append("'MK1' residue name not found (ligand not resolved from session state)")
    if not has_byelement:
        issues.append("'byelement' coloring not found")
    if issues:
        return False, "; ".join(issues) + f" — commands: {commands}"
    return True, "sphere + MK1 + byelement ✓"

def assert_zone_selection(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_zone = _any_contains(commands, "zone")
    has_show = _any_contains(commands, "show") or _any_contains(commands, "atoms")
    if not has_zone:
        return False, f"Expected 'zone' selection — got: {commands}"
    if not has_show:
        return False, f"Expected 'show' after zone selection — got: {commands}"
    return True, "zone selection + show ✓"

def assert_save_image(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_save = _any_contains(commands, "save")
    has_png  = _any_contains(commands, ".png")
    has_path = any("desktop" in c.lower() or "Users" in c for c in commands)
    if not has_save:
        return False, f"Expected 'save' command — got: {commands}"
    if not has_png:
        return False, f"Expected '.png' in save command — got: {commands}"
    if not has_path:
        warn(f"Desktop path not detected in save command (may still be correct): {commands}")
    return True, "save + .png ✓"


TEST_SUITE = [
    (
        1,
        "open 1HSG and show it as a ribbon diagram",
        assert_open_ribbon,
        "Open 1HSG from RCSB and display as cartoon",
    ),
    (
        2,
        "color each chain a different color",
        assert_color_bychain,
        "Per-chain coloring → must use 'color bychain'",
    ),
    (
        3,
        "show the ligand as spheres and color it by element",
        assert_ligand_sphere,
        "Ligand sphere + byelement → must resolve 'MK1' from session state",
    ),
    (
        4,
        "find all residues within 4 angstroms of the ligand",
        assert_zone_selection,
        "Zone selection within 4 Å of ligand",
    ),
    (
        5,
        "save an image called test_output.png to my desktop",
        assert_save_image,
        "Save image to Desktop with Windows path",
    ),
]


# ════════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════════

class IntegrationRunner:
    def __init__(self, execute: bool = False, start_cx: bool = False):
        self.execute   = execute
        self.start_cx  = start_cx
        self.translator = CommandTranslator()
        self.session    = SessionState()
        self.bridge: Optional[ChimeraXBridge] = None

        self.passed  = 0
        self.failed  = 0
        self.skipped = 0
        self.results: List[Dict[str, Any]] = []

    def setup_bridge(self) -> bool:
        """Connect to / start ChimeraX.  Returns True if ready."""
        self.bridge = ChimeraXBridge(port=config.REST_PORT)
        self.bridge.base_url = f"http://{config.REST_HOST}:{config.REST_PORT}"
        self.bridge.run_url  = f"{self.bridge.base_url}/run"

        if self.bridge.is_running():
            info(f"ChimeraX REST server reachable at {self.bridge.run_url}")
            return True

        if self.start_cx:
            info("Launching ChimeraX (this may take 30 s)…")
            try:
                self.bridge.start(timeout=60)
                ok("ChimeraX started.")
                return True
            except Exception as exc:
                fail(f"Could not start ChimeraX: {exc}")
                return False

        warn("ChimeraX not running — execution tests will be skipped.")
        warn("Run with --start to launch ChimeraX, or enable REST manually:")
        warn("  remotecontrol rest start port 60001")
        return False

    def run(self) -> None:
        print(f"\n{BOLD}{CYAN}StructureBot — Integration Test Suite (1HSG){RESET}")
        print("=" * 60)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            fail("ANTHROPIC_API_KEY is not set — cannot run translation tests.")
            sys.exit(1)

        cx_ready = False
        if self.execute or self.start_cx:
            section("Setup — ChimeraX connection")
            cx_ready = self.setup_bridge()

        # Run each test
        for test_id, prompt, assert_fn, description in TEST_SUITE:
            self._run_one(test_id, prompt, assert_fn, description, cx_ready)

        # Cleanup
        if cx_ready and self.bridge:
            section("Cleanup")
            res = self.bridge.run_command("close all")
            if not res.get("error"):
                ok("Closed all models — session is clean.")

        self._print_summary()

    def _run_one(
        self,
        test_id:     int,
        prompt:      str,
        assert_fn,
        description: str,
        cx_ready:    bool,
    ) -> None:
        section(f"Test {test_id} — {description}")
        info(f"Prompt: {prompt!r}")

        # ── Phase A: Translation ───────────────────────────────────────────────
        try:
            result = self.translator.translate(prompt, self.session)
        except Exception as exc:
            fail(f"Translation error: {exc}")
            self.failed += 1
            return

        commands     = result.get("commands", [])
        explanations = result.get("explanations", [])
        confidence   = result.get("confidence", "?")
        clarify      = result.get("clarification_needed")
        warnings_    = result.get("warnings", [])

        info(f"Confidence: {confidence}")

        if clarify:
            warn(f"Clarification requested: {clarify}")
            warn("This means the LLM did not have enough context to translate.")
            # For tests 3-5, a clarification about ligand name is a soft failure
            if test_id in (3, 4):
                warn("Hint: RCSB metadata fetch may have failed — check internet access.")
            fail("FAIL (translation): clarification_needed is set")
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "reason": f"clarification: {clarify}"})
            return

        if not commands:
            fail("FAIL (translation): no commands generated")
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "reason": "no commands"})
            return

        for w in warnings_:
            warn(f"Model warning: {w}")

        # Print commands
        for i, (cmd, exp) in enumerate(zip(commands, explanations), 1):
            info(f"  {i}. {cmd!r}  →  {exp}")

        # Assert
        passed, detail = assert_fn(commands)
        if passed:
            ok(f"PASS (translation): {detail}")
        else:
            fail(f"FAIL (translation): {detail}")
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "reason": detail})
            return

        # Update session state (simulates what main.py does after execution)
        self._simulate_state_update(test_id, commands, prompt)

        # ── Phase B: Execution ─────────────────────────────────────────────────
        if not cx_ready:
            warn("Skipping execution (ChimeraX not running).")
            self.passed  += 1
            self.skipped += 1
            self.results.append({"id": test_id, "passed": True, "executed": False})
            return

        # Wait briefly between tests so ChimeraX can finish rendering
        if test_id > 1:
            time.sleep(1.5)

        exec_results = self.bridge.run_commands(commands)
        exec_ok = True
        for r in exec_results:
            cmd = r["command"]
            err = r["result"].get("error")
            val = (r["result"].get("value") or "").strip()[:50]
            if err:
                fail(f"FAIL (execution): {cmd!r} → {err}")
                exec_ok = False
            else:
                suf = f" → {val!r}" if val else ""
                ok(f"Executed: {cmd!r}{suf}")

        if exec_ok:
            ok(f"PASS (translation + execution): all {len(commands)} command(s) succeeded")
            self.passed += 1
            self.results.append({"id": test_id, "passed": True, "executed": True})
        else:
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "executed": True,
                                  "reason": "execution error"})

    def _simulate_state_update(self, test_id: int, commands: List[str], prompt: str) -> None:
        """Mimic what main.py's _maybe_update_structure_state does."""
        import re
        for cmd in commands:
            s = cmd.strip().lower()
            if s.startswith("open ") and "session" not in s:
                parts = cmd.split()
                if len(parts) >= 2:
                    name = parts[1].strip("'\"")
                    mid  = self.session.next_model_id()
                    # This triggers RCSB metadata fetch for 4-char PDB IDs
                    self.session.add_structure(mid, name)
            elif s.startswith("close"):
                if "all" in s:
                    self.session.clear_all_structures()
                else:
                    m = re.search(r"#(\d+)", cmd)
                    if m:
                        self.session.remove_structure(m.group(1))

        self.session.add_to_history(prompt, commands, success=True)

    def _print_summary(self) -> None:
        section("Summary")
        total = len(TEST_SUITE)
        exec_count = sum(1 for r in self.results if r.get("executed"))
        print(
            f"  {GREEN}{self.passed} passed{RESET}  "
            f"{RED}{self.failed} failed{RESET}  "
            f"({self.skipped} skipped execution)  "
            f"of {total} tests"
        )
        if exec_count:
            print(f"  {exec_count} test(s) were fully executed against ChimeraX.")
        print()

        if self.failed == 0:
            print(f"{GREEN}All tests passed!{RESET}")
            if self.skipped:
                print(f"  Re-run with --execute to also test ChimeraX execution.")
            sys.exit(0)
        else:
            print(f"{RED}Some tests failed. See details above.{RESET}")
            for r in self.results:
                if not r["passed"]:
                    print(f"  Test {r['id']}: {r.get('reason', 'unknown')}")
            sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StructureBot integration test — 1HSG full workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute commands against ChimeraX (must be running)",
    )
    parser.add_argument(
        "--start", action="store_true",
        help="Launch ChimeraX automatically before executing",
    )
    args = parser.parse_args()

    runner = IntegrationRunner(
        execute  = args.execute or args.start,
        start_cx = args.start,
    )
    runner.run()


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════════════
# Pytest integration tests — glycan + proline bridges
# (self-contained; no ChimeraX instance required)
# ════════════════════════════════════════════════════════════════════════════════

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_glycan_sequon_detection():
    """
    GlycanBridge.full_glycan_scan() should detect NXS/T sequons
    and return results with the correct field structure.
    """
    from glycan_bridge import GlycanBridge
    gb = GlycanBridge()
    # MNASVNATL: contains NAS (pos 2) and NAT (pos 6 = N-A-T)
    seq = "MNASVNATL"
    result = gb.full_glycan_scan(sequence=seq, chain="A")
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert "native_sequons"        in result
    assert "engineered_candidates" in result
    assert "all_ranked"            in result
    # At least one NXS/T sequon should be found (NAS at pos 2, NAT at pos 7)
    assert len(result["all_ranked"]) >= 1
    for site in result["all_ranked"]:
        assert "position"          in site
        assert "sequon"            in site
        assert "composite_score"   in site
        assert "confidence"        in site


def test_glycan_chimerax_commands():
    """
    GlycanBridge.generate_chimerax_commands() should produce color + label
    commands with green/cyan/yellow palette (not magenta).
    """
    from glycan_bridge import GlycanBridge
    gb = GlycanBridge()
    seq = "MNASVNATL"
    result = gb.full_glycan_scan(sequence=seq, chain="A")
    # Use all_ranked so we always have candidates to pass
    candidates = result["all_ranked"]
    cmds, exps = gb.generate_chimerax_commands(
        candidates, model_id="1", chain="A"
    )
    assert isinstance(cmds, list)
    assert len(cmds) > 0
    # Colors must be green/cyan/yellow — not magenta
    color_cmds = [c for c in cmds if c.startswith("color")]
    for cmd in color_cmds:
        assert "#cc00cc" not in cmd, "Glycan used proline's magenta color"
    # Exactly one explanation per candidate (3 cmds each)
    assert len(exps) == len(candidates)


def test_proline_scan_no_overlap_with_glycan_colors():
    """
    ProlineBridge.generate_chimerax_commands() uses magenta/orange/yellow.
    These must not overlap with the green/cyan used by GlycanBridge.
    """
    from proline_bridge import ProlineBridge
    pb = ProlineBridge()
    fake_cands = [
        {
            "position":        10,
            "from_aa":         "L",
            "to_aa":           "P",
            "phi":             -60.0,
            "psi":             -40.0,
            "ss":              "L",
            "phi_score":       1.0,
            "loop_bonus":      1.3,
            "esm_factor":      1.0,
            "iface_factor":    1.0,
            "hbond_factor":    1.0,
            "composite_score": 0.9,
            "confidence":      "high",
            "near_interface":  False,
        }
    ]
    cmds, _ = pb.generate_chimerax_commands(fake_cands, model_id="1", chain="A")
    color_cmds = [c for c in cmds if c.startswith("color")]
    for cmd in color_cmds:
        # Proline uses magenta (#cc00cc) for high — not green (#00cc00) or cyan (#00cccc)
        assert "#00cc00" not in cmd, "Proline used glycan's green"
        assert "#00cccc" not in cmd, "Proline used glycan's cyan"
        assert "#cc00cc" in cmd     # magenta for high confidence


def test_proline_full_scan_mocked_backbone():
    """
    ProlineBridge.full_proline_scan() orchestrator should integrate
    backbone extraction → candidate scan → result dict correctly
    when backbone extraction is mocked.
    """
    from proline_bridge import ProlineBridge
    import tempfile, os

    pb = ProlineBridge()

    # 12-residue backbone: positions 1-12, all LEU in loop with ideal φ=-60
    fake_backbone = {
        pos: {
            "phi":     -60.0,
            "psi":     -40.0,
            "ss":      "L",
            "resname": "LEU",
            "aa":      "L",
        }
        for pos in range(1, 13)
    }

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh:
        fh.write("REMARK fake\nEND\n")
        pdb_path = fh.name

    try:
        with patch.object(pb, "extract_backbone_angles", return_value=fake_backbone):
            result = pb.full_proline_scan(
                pdb_path = pdb_path,
                chain    = "A",
            )
    finally:
        os.unlink(pdb_path)

    assert result["count"] > 0
    assert result["top"] is not None
    assert result["chain"] == "A"
    assert result["n_residues_scanned"] == 12
    # All candidates should be → P
    for c in result["candidates"]:
        assert c["to_aa"] == "P"
        assert c["from_aa"] == "L"


# ================================================================================
# Pytest integration tests -- salt_bridge + cavity session state coexistence
# ================================================================================

def test_salt_bridge_cavity_coexist_in_session_state():
    """
    set_salt_bridge_results and set_cavity_results must both persist without
    overwriting each other in the same SessionState.
    """
    from session_state import SessionState
    s = SessionState()

    sb_result = {
        'success': True,
        'chain': 'A',
        'existing_salt_bridges': [{'res1': 'A10', 'res2': 'A20', 'distance': 3.5}],
        'candidates': [],
        'total_existing': 1,
        'total_candidates': 0,
        'error': None,
    }
    cav_result = {
        'success': True,
        'chain': 'A',
        'cavities': [{'cavity_id': 1, 'lining_residues': ['A5'], 'estimated_volume_A3': 30.0}],
        'candidates': [],
        'total_cavities': 1,
        'total_candidates': 0,
        'error': None,
    }

    s.set_salt_bridge_results('1', sb_result)
    s.set_cavity_results('1', cav_result)

    retrieved_sb  = s.get_salt_bridge_results('1')
    retrieved_cav = s.get_cavity_results('1')

    assert retrieved_sb  is not None, 'salt_bridge_results should be retrievable'
    assert retrieved_cav is not None, 'cavity_results should be retrievable'
    assert retrieved_sb['total_existing'] == 1
    assert retrieved_cav['total_cavities'] == 1


def test_all_five_results_coexist():
    """
    Five result types -- proline, glycan, salt_bridge, cavity, and ProteinMPNN --
    must all coexist in the same session without cross-contamination.
    """
    from session_state import SessionState
    s = SessionState()

    # Store each result type
    s.set_proline_results('1', 'A', {'candidates': [], 'count': 0, 'chain': 'A'})
    s.set_glycan_results('1', 'A', {'native_sequons': [], 'engineered_candidates': []})
    s.set_salt_bridge_results('1', {'chain': 'A', 'existing_salt_bridges': [], 'candidates': []})
    s.set_cavity_results('1', {'chain': 'A', 'cavities': [], 'candidates': []})
    s.add_proteinmpnn_result('1', {'sequences': [], 'wildtype_sequence': 'AKLDE'})

    # All should be independently retrievable
    assert s.get_proline_results('1', 'A') is not None, 'Proline results missing'
    assert s.get_glycan_results('1', 'A')  is not None, 'Glycan results missing'
    assert s.get_salt_bridge_results('1')  is not None, 'Salt bridge results missing'
    assert s.get_cavity_results('1')       is not None, 'Cavity results missing'
    assert s.get_proteinmpnn_result('1')   is not None, 'ProteinMPNN results missing'

    # No cross-contamination
    assert s.get_salt_bridge_results('2') is None, 'model 2 should have no salt bridge data'
    assert s.get_cavity_results('2')      is None, 'model 2 should have no cavity data'
