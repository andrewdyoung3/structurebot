"""
tests/test_disulfide_modes.py
-----------------------------
Router side of the fold-based disulfide suite: Mode A discovery (multi-seed bonding FREQUENCY,
reusing the shared `_fold_n_seeds`), Mode B geometry readout (cheap, reads ONE fold — must NEVER
trigger Mode A's multi-seed run), and the Mode C provenance threading. All mocked: no Boltz, no GPU.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter
from session_state import SessionState


def _cif(sg_sg_pair: float) -> str:
    """A minimal fold CIF with two cysteines whose SG–SG distance is *sg_sg_pair* Å (along x)."""
    return (
        "data_model\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM CA CYS A 12 0.0 0.0 0.0\n"
        "ATOM CB CYS A 12 0.0 1.5 0.0\n"
        "ATOM SG CYS A 12 0.0 2.5 0.0\n"
        "ATOM CA CYS A 45 5.5 0.0 0.0\n"
        "ATOM CB CYS A 45 3.8 1.5 0.0\n"
        f"ATOM SG CYS A 45 {sg_sg_pair:.3f} 2.5 0.0\n#\n"
    )


def _write(text: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.write(text); f.close()
    return f.name


def _router():
    return ToolRouter(bridge=MagicMock(), session=SessionState())


# ── Mode A — discovery frequency tally ───────────────────────────────────────────────
def test_discovery_tallies_per_pair_frequency():
    r = _router()
    compat = _write(_cif(2.0))      # SG–SG 2.0 (in 1.8–2.5 window) → bonding-compatible
    incompat = _write(_cif(4.0))    # SG–SG 4.0 (outside) → not compatible
    # 4 folds: 3 compatible, 1 not → frequency 3/4
    r._fold_n_seeds = MagicMock(return_value=[compat, compat, compat, incompat])
    out = r._run_disulfide_discovery({"sequence": "MKVC" + "A" * 40 + "C" + "A" * 4})
    assert out.success
    top = out.data["pairs"][0]
    assert (top["resnum_a"], top["resnum_b"]) == (12, 45)
    assert top["n_compatible"] == 3 and top["n_folds"] == 4 and top["frequency"] == 0.75
    assert "3/4 folds" in out.summary and "unconstrained" in out.summary
    r._fold_n_seeds.assert_called_once()           # discovery DID fold


def test_discovery_unconstrained_invariant():
    # discovery must fold UNCONSTRAINED — _fold_n_seeds called with no constraints
    r = _router()
    captured = {}
    def _fake(chains, n, **kw):
        captured.update(kw); return [_write(_cif(2.0))] * 3
    r._fold_n_seeds = _fake
    r._run_disulfide_discovery({"sequence": "MKVCAAACAA"})
    assert captured.get("constraints") in (None, [])   # never constrained


def test_discovery_fails_loud_when_no_folds():
    r = _router()
    r._fold_n_seeds = MagicMock(return_value=[])
    out = r._run_disulfide_discovery({"sequence": "MKVCAAACAA"})
    assert out.success is False and "no unconstrained folds" in out.error.lower()


# ── done-vs-stuck legibility (Mode-A completion-signal fix, observability half) ────────
def test_fold_n_seeds_emits_per_seed_progress():
    """A long N-seed fold must ADVANCE visibly: _fold_n_seeds emits a per-seed heartbeat via
    the status_callback (start + completion per seed) so a healthy long run is never silent —
    the done-vs-stuck legibility the §9 bug demanded. REAL _fold_n_seeds, fake boltz bridge."""
    r = _router()
    fake_boltz = MagicMock()
    fake_boltz.predict.return_value = {"success": True, "cif_path": _write(_cif(2.0))}
    r._boltz_bridge = fake_boltz                     # bypass the lazy real-bridge import

    msgs = []
    r._status_callback = msgs.append               # the seam execute() sets per-run

    paths = r._fold_n_seeds([{"id": "A", "sequence": "MKVCAAACAA"}], 3)
    assert len(paths) == 3
    assert fake_boltz.predict.call_count == 3
    # one "Folding seed k/3" + one "Seed k/3 folded" per seed → 2×N advancing messages
    assert len(msgs) == 2 * 3, msgs
    assert any("seed 1/3" in m.lower() for m in msgs)
    assert any("seed 3/3" in m.lower() for m in msgs)
    assert any("folded" in m.lower() for m in msgs)   # a completion signal, not just a start


def test_fold_n_seeds_no_callback_is_safe():
    """No status_callback set (e.g. a non-GUI caller) → _fold_n_seeds still folds, no crash."""
    r = _router()
    fake_boltz = MagicMock()
    fake_boltz.predict.return_value = {"success": True, "cif_path": _write(_cif(2.0))}
    r._boltz_bridge = fake_boltz
    # _status_callback never assigned → getattr(...None) path
    paths = r._fold_n_seeds([{"id": "A", "sequence": "MKVCAAACAA"}], 2)
    assert len(paths) == 2


def test_discovery_busy_label_is_named_not_bare():
    """_long_tools must include disulfide_discovery so the busy label is the NAMED tool, not
    a bare 'Running ' — the GUI shows this label immediately, so a bare one is the user's
    'I can't tell what's happening' complaint. Drives the real engine completion path."""
    from request_engine import RequestEngine

    r = _router()
    r._fold_n_seeds = MagicMock(return_value=[_write(_cif(2.0))] * 2)

    class _Host:
        bridge = None                              # → verb-probe skipped (no ChimeraX)
        def __init__(self, router):
            self.router = router
            self.translator = MagicMock()
            self.session = router.session
        def _maybe_update_structure_state(self, cmds): pass
        def _log_exchange(self, *a, **k): pass

    labels = []

    class _Presenter:
        cancelled = False
        def running_tools(self, label, eta_s=0.0, needs_timer=False):
            labels.append(label)
            class _CM:
                def __enter__(s): return s
                def __exit__(s, *a): return False
            return _CM()
        def confirm(self, c): return "proceed"
        def __getattr__(self, name):
            return lambda *a, **k: None

    engine = RequestEngine(_Host(r))
    engine.handle_tool_request(
        "disulfide_discovery", {"sequence": "MKVCAAACAA", "n_seeds": 2},
        "[Workbench] disulfide discovery", _Presenter(), confidence="low",
    )
    assert labels, "running_tools was never entered"
    assert "disulfide_discovery" in labels[0], labels
    assert labels[0].strip() != "Running", "busy label is the bare 'Running ' (the bug)"


# ── Mode B — geometry readout is CHEAP and never triggers Mode A ──────────────────────
def test_geometry_readout_reads_one_fold_without_folding():
    r = _router()
    r._fold_n_seeds = MagicMock()                  # if Mode B ever folds, this fires → fail
    cif = _write(_cif(2.05))
    out = r._run_disulfide_geometry({"cif_path": cif})
    assert out.success
    p = out.data["pairs"][0]
    assert p["sg_sg"] == 2.05 and p["bonding_compatible"] is True
    assert p["chi_ss"] is not None                 # χSS measured from SG atoms
    assert "measured" in out.summary
    r._fold_n_seeds.assert_not_called()            # THE decoupling: B must not trigger A


def test_geometry_readout_incompatible_pair_reported_honestly():
    r = _router()
    out = r._run_disulfide_geometry({"cif_path": _write(_cif(9.0))})
    assert out.success
    assert out.data["pairs"][0]["bonding_compatible"] is False
    assert "incompatible" in out.summary


def test_geometry_readout_needs_a_cif():
    out = _router()._run_disulfide_geometry({"cif_path": "/nope/x.cif"})
    assert out.success is False and "fold cif" in out.error.lower()


# ── Mode C — constraints + provenance threaded through _run_boltz ─────────────────────
def test_run_boltz_threads_constraints_and_tags_provenance(monkeypatch):
    r = _router()
    fake_bridge = MagicMock()
    fake_bridge.predict.return_value = {
        "success": True, "cif_path": _write(_cif(2.05)), "mean_plddt": 85.0,
        "plddt": {1: 85.0}, "plddt_by_chain": {}, "chain_ids": ["A"], "seed": 0,
        "source": "local_boltz_env"}
    r._get_boltz_bridge = lambda: fake_bridge
    r._open_and_viz_fold_live = lambda *a, **k: ("9", [], [])
    cons = [{"atom1": ["A", 2, "SG"], "atom2": ["A", 4, "SG"]}]
    out = r._run_boltz({"chains": [{"id": "A", "sequence": "MKVC"}], "no_reference": True,
                        "disulfide_constraints": cons, "disulfide_bonds": [(2, 4)]})
    assert out.success
    # constraints reached the bridge
    assert fake_bridge.predict.call_args.kwargs["constraints"] == cons
    # provenance tagged on the result data
    assert out.data["constrained"] is True and out.data["disulfide_bonds"] == [(2, 4)]
    assert "SS-constrained" in out.summary


def _backbone_cif():
    # two residues in engineerable backbone geometry (1 & 5) + a far one (9) + adjacent (2)
    return (
        "data_model\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM CA ALA A 1 0.0 0.0 0.0\nATOM CB ALA A 1 0.0 1.5 0.0\n"
        "ATOM CA GLY A 2 2.0 0.0 0.0\n"
        "ATOM CA VAL A 5 5.5 0.0 0.0\nATOM CB VAL A 5 3.8 1.5 0.0\n"
        "ATOM CA LEU A 9 40.0 0.0 0.0\nATOM CB LEU A 9 40.0 1.5 0.0\n#\n"
    )


# ── Mode D — engineering scan (reads one fold's backbone; ranked + best-partner + caveat) ──
def test_engineering_scan_finds_sites_and_carries_caveat():
    r = _router()
    r._fold_n_seeds = MagicMock()                  # the scan must NEVER fold (cheap geometry only)
    out = r._run_disulfide_scan({"cif_path": _write(_backbone_cif())})
    assert out.success
    pairs = {(p["resnum_a"], p["resnum_b"]) for p in out.data["pairs"]}
    assert (1, 5) in pairs                         # the engineerable pair surfaces
    assert all(9 not in (a, b) for a, b in pairs)  # far residue excluded
    assert out.data["best_partner"].get("A", {}).get(1) is not None   # heatmap map present
    # the load-bearing caveat rides with the readout (data AND summary)
    assert "starting point" in out.data["caveat"] and "does not imply" in out.data["caveat"].lower()
    assert "starting point" in out.summary.lower()
    r._fold_n_seeds.assert_not_called()


def test_engineering_scan_needs_a_cif():
    out = _router()._run_disulfide_scan({"cif_path": "/nope/x.cif"})
    assert out.success is False and "fold cif" in out.error.lower()


def test_engineering_scan_empty_when_no_viable_sites():
    # a chain whose only residues are far apart → no sites, but still a success + the caveat
    cif = ("data_model\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n"
           "_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
           "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
           "ATOM CA ALA A 1 0.0 0.0 0.0\nATOM CA ALA A 9 50.0 0.0 0.0\n#\n")
    out = _router()._run_disulfide_scan({"cif_path": _write(cif)})
    assert out.success and out.data["pairs"] == [] and "starting point" in out.summary.lower()


# ── interface scan (cross-chain analogue of Mode D — inter-subunit sites at the interface) ──
def _interface_cif():
    # chain A and chain B with ONE engineerable interface pair (A:5 ↔ B:5, Cα 5.5 / Cβ 3.8); the
    # other residues are far from the OTHER chain (buried/non-interface) → must not surface.
    return (
        "data_model\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM CA VAL A 5 0.0 0.0 0.0\nATOM CB VAL A 5 0.0 1.5 0.0\n"
        "ATOM CA LEU A 9 40.0 0.0 0.0\nATOM CB LEU A 9 40.0 1.5 0.0\n"
        "ATOM CA VAL B 5 5.5 0.0 0.0\nATOM CB VAL B 5 3.8 1.5 0.0\n"
        "ATOM CA LEU B 9 80.0 0.0 0.0\nATOM CB LEU B 9 80.0 1.5 0.0\n#\n"
    )


def test_interface_scan_finds_cross_chain_sites_and_carries_caveat():
    r = _router()
    r._fold_n_seeds = MagicMock()                  # the scan must NEVER fold (cheap geometry only)
    out = r._run_disulfide_interface_scan({"cif_path": _write(_interface_cif())})
    assert out.success and out.data["pairs"]
    top = out.data["pairs"][0]
    assert top["chain_a"] == "A" and top["chain_b"] == "B"      # CROSS-chain
    assert {top["resnum_a"], top["resnum_b"]} == {5}
    surfaced = {(p["chain_a"], p["resnum_a"]) for p in out.data["pairs"]} \
        | {(p["chain_b"], p["resnum_b"]) for p in out.data["pairs"]}
    assert ("A", 9) not in surfaced and ("B", 9) not in surfaced  # non-interface → interface-bounded
    assert out.data["caveat"] and "does not imply" in out.data["caveat"].lower()
    r._fold_n_seeds.assert_not_called()


def test_interface_scan_needs_a_multimer():
    # one chain → no interface; explicit error, never a silent empty
    mono = ("data_model\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n"
            "_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
            "ATOM CA ALA A 1 0.0 0.0 0.0\nATOM CA ALA A 5 5.5 0.0 0.0\n#\n")
    out = _router()._run_disulfide_interface_scan({"cif_path": _write(mono)})
    assert out.success is False and "multi-chain" in out.error.lower()


def test_interface_scan_needs_a_cif():
    out = _router()._run_disulfide_interface_scan({"cif_path": "/nope/x.cif"})
    assert out.success is False and "fold cif" in out.error.lower()


# ── cosmetic: the pipeline strip shows real tool names (no "Unknown tool") ────────────
def test_step_description_real_names_for_disulfide_tools():
    r = _router()
    for tool, key in [("disulfide_discovery", "discovery"),
                      ("disulfide_geometry", "geometry"),
                      ("disulfide_scan", "engineering scan"),
                      ("disulfide_interface_scan", "interface")]:
        desc = r._step_description(tool, {tool: {}}, "")
        assert "Unknown tool" not in desc
        assert "Disulfide" in desc and key.split()[0].lower() in desc.lower()


def test_disulfide_tools_have_pipeline_icons():
    from tool_router import ToolRouter
    for tool in ("disulfide_discovery", "disulfide_geometry", "disulfide_scan",
                 "disulfide_interface_scan", "disulfide_ddg_estimate"):
        assert ToolRouter._TOOL_ICONS.get(tool) and ToolRouter._TOOL_ICONS[tool] != "⚙️"


# ── ΔΔG-escalation (route a flagged interface pair into the legacy ΔΔG bridge) ────────────
def _pdb_atom(serial, atom, resname, chain, resnum, x, y, z):
    """One fixed-column PDB ATOM record (the legacy `parse_pdb_atoms` reads PDB, not mmCIF)."""
    aname = f"{(' ' + atom) if len(atom) < 4 else atom:<4}"
    return (f"ATOM  {serial:>5} {aname}{' '}{resname:>3} {chain}{resnum:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}\n")


def _two_chain_pdb() -> str:
    """A:12 = VAL, B:30 = LEU (each CA+CB) — a cross-chain pair with DISTINCT WT residues per chain."""
    return (
        _pdb_atom(1, "CA", "VAL", "A", 12, 0.0, 0.0, 0.0)
        + _pdb_atom(2, "CB", "VAL", "A", 12, 0.0, 1.5, 0.0)
        + _pdb_atom(3, "CA", "LEU", "B", 30, 5.5, 0.0, 0.0)
        + _pdb_atom(4, "CB", "LEU", "B", 30, 3.8, 1.5, 0.0)
        + "END\n"
    )


def _pdb(text: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    f.write(text); f.close()
    return f.name


def _ddg_inputs(pdb, *, aa_a="V", aa_b="L", source="loaded"):
    return {"pdb_path": pdb, "chain_a": "A", "resnum_a": 12, "from_aa_a": aa_a,
            "chain_b": "B", "resnum_b": 30, "from_aa_b": aa_b, "source": source}


def _mock_stability(r, ddg_a=1.2, ddg_b=0.4):
    """Stand in for the legacy ddG bridge: fill ddg on the one-element candidate, record the call."""
    bridge = MagicMock()
    def _score(cands, pdb_path, ca, cb, *a, **k):
        cands[0]["ddg_a"], cands[0]["ddg_b"] = ddg_a, ddg_b
        return cands
    bridge._score_stability.side_effect = _score
    r._disulfide_bridge = bridge                    # pre-seed so _get_disulfide_bridge returns it
    return bridge


def test_ddg_escalation_gate_matrix():
    from tool_router import ToolRouter as T
    assert T._ddg_escalation_gate("dynamut2", "denovo", False)[0] is False   # no de-novo web upload
    assert T._ddg_escalation_gate("dynamut2", "loaded", False)[0] is True    # loaded PDB is public
    assert T._ddg_escalation_gate("local", "loaded", False)[0] is False      # local needs WSL
    assert T._ddg_escalation_gate("local", "denovo", True)[0] is True
    assert T._ddg_escalation_gate("pyrosetta", "denovo", False)[0] is True   # local import path


def test_ddg_estimate_scores_the_clicked_pair(monkeypatch):
    import rosetta_bridge
    monkeypatch.setattr(rosetta_bridge, "_select_backend", lambda: "pyrosetta")
    r = _router()
    bridge = _mock_stability(r, ddg_a=1.2, ddg_b=0.4)
    out = r._run_disulfide_ddg_estimate(_ddg_inputs(_pdb(_two_chain_pdb())))
    assert out.success
    d = out.data
    assert d["ddg_a"] == 1.2 and d["ddg_b"] == 0.4 and d["ddg_mean"] == 0.8
    assert d["chain_a"] == "A" and d["resnum_a"] == 12 and d["chain_b"] == "B" and d["resnum_b"] == 30
    # the NARROW primitive scored EXACTLY this pair — never the full analyze() find/filter pipeline
    bridge._score_stability.assert_called_once()
    bridge.analyze.assert_not_called()
    cand = bridge._score_stability.call_args.args[0][0]
    assert cand["chain_a_residue"] == 12 and cand["chain_a_aa"] == "V"      # own-chain from_aa threaded
    assert cand["chain_b_residue"] == 30 and cand["chain_b_aa"] == "L"


def test_ddg_estimate_verifies_from_aa_fail_closed(monkeypatch):
    # claim G at A:12 (the structure has V) → ABORT, never score the wrong mutation
    import rosetta_bridge
    monkeypatch.setattr(rosetta_bridge, "_select_backend", lambda: "pyrosetta")
    r = _router()
    bridge = _mock_stability(r)
    out = r._run_disulfide_ddg_estimate(_ddg_inputs(_pdb(_two_chain_pdb()), aa_a="G"))
    assert out.success is False and "mismatch" in out.error.lower()
    bridge._score_stability.assert_not_called()                            # no scoring of a wrong target


def test_ddg_estimate_blocks_denovo_web_upload(monkeypatch):
    import rosetta_bridge
    monkeypatch.setattr(rosetta_bridge, "_select_backend", lambda: "dynamut2")
    r = _router()
    bridge = _mock_stability(r)
    out = r._run_disulfide_ddg_estimate(_ddg_inputs(_pdb(_two_chain_pdb()), source="denovo"))
    assert out.success is False and ("upload" in out.error.lower() or "dynamut2" in out.error.lower())
    bridge._score_stability.assert_not_called()                            # nothing left the machine


def test_ddg_estimate_allows_loaded_over_web(monkeypatch):
    import rosetta_bridge
    monkeypatch.setattr(rosetta_bridge, "_select_backend", lambda: "dynamut2")
    r = _router()
    _mock_stability(r)
    out = r._run_disulfide_ddg_estimate(_ddg_inputs(_pdb(_two_chain_pdb()), source="loaded"))
    assert out.success                                                      # a public loaded PDB may use the web


def test_ddg_estimate_local_needs_wsl(monkeypatch):
    import rosetta_bridge, wsl_bridge
    monkeypatch.setattr(rosetta_bridge, "_select_backend", lambda: "local")
    wb = MagicMock(); wb.return_value.is_available.return_value = False
    monkeypatch.setattr(wsl_bridge, "WSLBridge", wb)
    r = _router()
    bridge = _mock_stability(r)
    out = r._run_disulfide_ddg_estimate(_ddg_inputs(_pdb(_two_chain_pdb())))
    assert out.success is False and "wsl" in out.error.lower()
    bridge._score_stability.assert_not_called()


def test_ddg_estimate_needs_a_pdb():
    out = _router()._run_disulfide_ddg_estimate(_ddg_inputs("/nope/x.pdb"))
    assert out.success is False and "pdb" in out.error.lower()
