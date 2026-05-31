"""
tests/test_validate_design.py
-----------------------------
Unit tests for the validate-design meta-tool (thin orchestrator: ColabFold
confidence + matchmaker RMSD + Rosetta folding-energy). All three sub-steps are
mocked — NO live fold / WSL2 / ChimeraX. Emphasis on the honesty-critical energy
logic: sanity vs relative (topologies match) vs DECLINE (cross-topology).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter
from session_state import SessionState


# ── Helpers: tiny PDBs + fakes ──────────────────────────────────────────────────

def _write_pdb(path: Path, chains: dict) -> str:
    """chains = {'A': n_residues, ...} → a minimal CA-only PDB. Returns the path."""
    lines, serial = [], 1
    for ch, n in chains.items():
        for i in range(1, n + 1):
            lines.append(
                f"ATOM  {serial:>5}  CA  ALA {ch}{i:>4}      "
                f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 50.00           C"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class _FakeColabFold:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, **kw):
        self.predict_calls += 1
        return {"success": True, "cached": False, "ranked_pdb": "X",
                "mean_plddt": 88.0, "plddt": {1: 88.0}, "pae": [[0]],
                "ptm": 0.8, "iptm": None, "length": len(kw.get("sequence", "")),
                "copies": kw.get("copies", 1), "png_paths": {}}


class _FakeRosetta:
    def __init__(self, total=-120.0, clash_ok=True, succeed=True):
        self.calls = []
        self._total = total
        self._clash_ok = clash_ok
        self._succeed = succeed

    def relax_and_score(self, pdb_path, relax_cycles=3, progress_callback=None):
        self.calls.append(pdb_path)
        if not self._succeed:
            return {"success": False, "error": "WSL2 not available"}
        # Slightly different total per distinct path so deltas are meaningful.
        total = self._total - (1.5 * (len(self.calls) - 1))
        nres = 36
        return {"success": True, "total_reu": total, "n_residues": nres,
                "per_residue_density": round(total / nres, 4), "per_residue": [],
                "fa_rep": 10.0 if self._clash_ok else 900.0,
                "clash_ok": self._clash_ok, "converged": True,
                "relaxed_pdb": "relaxed.pdb", "backend": "pyrosetta_wsl2", "error": None}


class _FakeChimerax:
    """run_command stub: open returns an id, matchmaker returns parseable RMSD."""
    def __init__(self, rmsd=0.85):
        self.commands = []
        self._rmsd = rmsd
        self._next = 1

    def run_command(self, command, timeout=30):
        self.commands.append(command)
        if command.startswith("open "):
            mid = self._next
            self._next += 1
            return {"value": f"Opened as #{mid}, 1 model(s)", "error": None}
        if command.startswith("matchmaker"):
            return {"value": f"RMSD between 36 pruned atom pairs is {self._rmsd} angstroms",
                    "error": None}
        return {"value": "", "error": None}


def _router(rosetta, chimerax=None, colabfold=None):
    r = ToolRouter(bridge=chimerax, session=SessionState())
    r._get_rosetta_bridge = lambda: rosetta            # type: ignore
    if colabfold is not None:
        r._get_colabfold_bridge = lambda: colabfold    # type: ignore
    return r


def _fold(copies=1, length=36, ranked="X"):
    return {"success": True, "ranked_pdb": ranked, "mean_plddt": 88.0,
            "plddt": {i: 88.0 for i in range(1, length + 1)}, "pae": [[0]],
            "ptm": 0.8, "iptm": None, "length": length, "copies": copies,
            "png_paths": {}, "fold_source": "provided"}


# ══════════════════════════════════════════════════════════════════════════════
# PURE honesty logic — the energy decision (no I/O)
# ══════════════════════════════════════════════════════════════════════════════

def test_energy_decision_sanity_when_no_relative_requested():
    d = ToolRouter._design_energy_decision((1, (36,)), (1, (36,)), requested_relative=False)
    assert d["mode"] == "sanity"


def test_energy_decision_sanity_when_no_reference():
    d = ToolRouter._design_energy_decision((1, (36,)), None, requested_relative=True)
    assert d["mode"] == "sanity"


def test_energy_decision_relative_when_topologies_match():
    d = ToolRouter._design_energy_decision((1, (36,)), (1, (36,)), requested_relative=True)
    assert d["mode"] == "relative"


def test_energy_decision_declines_cross_topology():
    d = ToolRouter._design_energy_decision(
        (1, (36,)), (2, (36, 36)), requested_relative=True, ref_name="WT-dimer"
    )
    assert d["mode"] == "declined"
    assert "WT-dimer" in d["reason"]
    assert "2-mer" in d["reason"] and "1-mer" in d["reason"]


def test_energy_decision_declines_same_count_different_length():
    # same chain count but different per-chain length → still declined
    d = ToolRouter._design_energy_decision((1, (36,)), (1, (58,)), requested_relative=True)
    assert d["mode"] == "declined"


# ══════════════════════════════════════════════════════════════════════════════
# Topology extraction
# ══════════════════════════════════════════════════════════════════════════════

def test_topology_from_fold_homo_oligomer():
    assert ToolRouter._topology_from_fold({"copies": 3, "length": 40}) == (3, (40, 40, 40))


def test_topology_from_pdb_counts_chains(tmp_path):
    p = _write_pdb(tmp_path / "dimer.pdb", {"A": 36, "B": 36})
    assert ToolRouter._topology_from_pdb(p) == (2, (36, 36))
    p2 = _write_pdb(tmp_path / "mono.pdb", {"A": 36})
    assert ToolRouter._topology_from_pdb(p2) == (1, (36,))
    assert ToolRouter._topology_from_pdb("/no/such.pdb") is None


# ══════════════════════════════════════════════════════════════════════════════
# Fold acquisition — REUSE vs fold
# ══════════════════════════════════════════════════════════════════════════════

def test_fold_reused_from_session_no_refold(tmp_path):
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    cf = _FakeColabFold()
    r = _router(_FakeRosetta(), chimerax=_FakeChimerax(), colabfold=cf)
    r.session.set_colabfold_results("1", {"ranked_pdb": ranked, "mean_plddt": 90.0,
                                          "ptm": 0.7, "length": 36, "copies": 1})
    res = r._run_validate_design({"model_id": "1"}, user_input="validate this design")
    assert res.success
    assert res.data["design"]["fold_source"] == "reused (session)"
    assert cf.predict_calls == 0          # MUST NOT re-fold


def test_fold_folds_when_sequence_given_and_no_session(tmp_path):
    cf = _FakeColabFold()
    r = _router(_FakeRosetta(), chimerax=None, colabfold=cf)
    res = r._run_validate_design(
        {"model_id": "9", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
        user_input="validate this design",
    )
    assert res.success
    assert cf.predict_calls == 1
    assert res.data["design"]["fold_source"] == "folded"


def test_no_sequence_no_session_errors():
    r = _router(_FakeRosetta(), chimerax=None, colabfold=_FakeColabFold())
    res = r._run_validate_design({"model_id": "5"}, user_input="validate this design")
    assert res.success is False
    assert "no in-session" in res.error.lower() or "no sequence" in res.error.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Matchmaker RMSD parse
# ══════════════════════════════════════════════════════════════════════════════

def test_matchmaker_rmsd_parsed_with_named_reference(tmp_path, monkeypatch):
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    cx = _FakeChimerax(rmsd=1.23)
    r = _router(_FakeRosetta(), chimerax=cx)
    monkeypatch.setattr(r, "_download_pdb_by_id",
                        lambda pid: _write_pdb(tmp_path / f"{pid}.pdb", {"A": 36}))
    out = r._matchmaker_rmsd_live(_fold(ranked=ranked),
                                  {"rmsd_ref": {"pdb": "1HSG", "chain": "A"}})
    assert out["rmsd_ca"] == 1.23
    assert "1HSG" in out["reference"]
    assert any(c.startswith("matchmaker") for c in out["commands"])


def test_matchmaker_skips_without_bridge():
    r = _router(_FakeRosetta(), chimerax=None)
    out = r._matchmaker_rmsd_live(_fold(ranked="X"), {})
    assert out["rmsd_ca"] is None
    assert "unavailable" in out["note"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Full report assembly + session storage + honesty end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def test_report_sanity_default(tmp_path):
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    ros = _FakeRosetta()
    r = _router(ros, chimerax=_FakeChimerax())
    res = r._run_validate_design(
        {"model_id": "1", "colabfold_result": _fold(ranked=ranked)},
        user_input="validate this design",
    )
    assert res.success
    e = res.data["folding_energy"]
    assert e["mode"] == "sanity"
    assert "relative_delta_reu" not in e          # no relative number by default
    assert e["total_reu"] is not None             # sanity signal present
    assert len(ros.calls) == 1                     # only the design scored
    # stored in session
    assert r.session.get_validate_design_results("1") is not None
    # evidence-rich: all three axes present, no verdict key
    assert set(["fold_confidence", "fold_preservation", "folding_energy"]) <= set(res.data)
    assert "verdict" not in res.data and "pass" not in res.data


def test_report_relative_when_topologies_match(tmp_path, monkeypatch):
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    eref   = _write_pdb(tmp_path / "ref_mono.pdb", {"A": 36})       # same topology
    ros = _FakeRosetta()
    r = _router(ros, chimerax=_FakeChimerax())
    res = r._run_validate_design(
        {"model_id": "1", "colabfold_result": _fold(copies=1, length=36, ranked=ranked),
         "requested_relative": True, "energy_ref": eref},
        user_input="validate this design, relative stability",
    )
    e = res.data["folding_energy"]
    assert e["mode"] == "relative"
    assert "relative_delta_reu" in e               # relative number emitted
    assert e["energy_reference"] == eref
    assert len(ros.calls) == 2                     # design + reference both scored


def test_report_declines_cross_topology_no_number(tmp_path, monkeypatch):
    """THE honesty-critical path: relative requested but design (monomer) vs
    reference (dimer) → NO relative number, a reason given, sanity still reported."""
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    eref   = _write_pdb(tmp_path / "ref_dimer.pdb", {"A": 36, "B": 36})   # cross-topology
    ros = _FakeRosetta()
    r = _router(ros, chimerax=_FakeChimerax())
    res = r._run_validate_design(
        {"model_id": "1", "colabfold_result": _fold(copies=1, length=36, ranked=ranked),
         "requested_relative": True, "energy_ref": eref},
        user_input="validate this design, relative stability vs the WT dimer",
    )
    e = res.data["folding_energy"]
    assert e["mode"] == "declined"
    assert "relative_delta_reu" not in e           # MUST NOT emit a relative number
    assert "reference_total_reu" not in e
    assert "isn't meaningful" in e["reason"]
    assert e["total_reu"] is not None              # sanity still surfaced
    assert len(ros.calls) == 1                      # reference NOT scored (no delta computed)


def test_report_energy_unavailable_still_succeeds(tmp_path):
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    r = _router(_FakeRosetta(succeed=False), chimerax=_FakeChimerax())
    res = r._run_validate_design(
        {"model_id": "1", "colabfold_result": _fold(ranked=ranked)},
        user_input="validate this design",
    )
    assert res.success                              # other axes still produce a report
    assert res.data["folding_energy"]["available"] is False
    assert res.data["fold_confidence"]["mean_plddt"] is not None


def test_clash_flag_surfaced(tmp_path):
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    r = _router(_FakeRosetta(clash_ok=False), chimerax=_FakeChimerax())
    res = r._run_validate_design(
        {"model_id": "1", "colabfold_result": _fold(ranked=ranked)},
        user_input="validate this design",
    )
    flags = res.data["folding_energy"]["flags"]
    assert any("clash" in f.lower() for f in flags)


def test_low_plddt_flag_surfaced(tmp_path):
    ranked = _write_pdb(tmp_path / "ranked.pdb", {"A": 36})
    low = _fold(ranked=ranked); low["mean_plddt"] = 55.0
    r = _router(_FakeRosetta(), chimerax=_FakeChimerax())
    res = r._run_validate_design(
        {"model_id": "1", "colabfold_result": low}, user_input="validate this design")
    assert any("plddt" in f.lower() for f in res.data["fold_confidence"]["flags"])


# ══════════════════════════════════════════════════════════════════════════════
# Intent + routing
# ══════════════════════════════════════════════════════════════════════════════

def test_intent_positive_negative():
    r = ToolRouter(bridge=None, session=SessionState())
    assert r._detect_validate_design_intent("validate this design") is True
    assert r._detect_validate_design_intent("fully validate the design") is True
    assert r._detect_validate_design_intent("validate ddg of V82A") is False
    assert r._detect_validate_design_intent("fold MKT as a dimer") is False


def test_route_wins_over_mpnn_esmfold_keyword_collision():
    # 'validate design' is also an mpnn_esmfold keyword; validate_design must win.
    r = ToolRouter(bridge=None, session=SessionState())
    routed = r.route({"tools_needed": ["esmfold"], "tool_inputs": {"esmfold": {}}},
                     user_input="validate this design vs 1HSG chain A")
    assert routed["tools_needed"] == ["validate_design"]
    assert routed["tool_inputs"]["validate_design"]["rmsd_ref"] == {"pdb": "1HSG", "chain": "A"}


def test_route_colabfold_not_grabbed_by_validate_design():
    r = ToolRouter(bridge=None, session=SessionState())
    routed = r.route({"tools_needed": ["chimerax"], "tool_inputs": {}},
                     user_input="fold MKTAYIAK as a dimer with colabfold")
    assert routed["tools_needed"] == ["colabfold"]


def test_parse_relative_and_energy_ref():
    r = ToolRouter(bridge=None, session=SessionState())
    opts = r._parse_validate_design_options(
        "validate design, relative stability vs energy reference 2LZM")
    assert opts.get("requested_relative") is True
    assert opts.get("energy_ref") == "2LZM"


# ── Optional live e2e (opt-in; never runs in CI) ────────────────────────────────

@pytest.mark.skipif(
    os.environ.get("STRUCTUREBOT_RUN_LIVE_COLABFOLD") != "1",
    reason="live validate-design e2e; set STRUCTUREBOT_RUN_LIVE_COLABFOLD=1 to run",
)
def test_live_validate_design_reuses_cached_fold():
    """Tiny monomer e2e REUSING a cached HP36 fold — exercises the two real
    parsers mocks can't: matchmaker RMSD (if ChimeraX up) and rosetta relax-score."""
    from colabfold_bridge import ColabFoldBridge
    from rosetta_bridge import RosettaBridge
    cf = ColabFoldBridge()
    if not cf.is_available():
        pytest.skip("ColabFold env / WSL2 not available")
    fold = cf.predict("MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", quick=True)  # cached from prior runs
    assert fold["success"], fold.get("error")
    score = RosettaBridge().relax_and_score(fold["ranked_pdb"])
    assert score["success"], score.get("error")
    assert score["total_reu"] is not None and score["n_residues"] > 0
