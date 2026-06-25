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
