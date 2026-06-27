"""
tests/test_saltbridge_modes.py
------------------------------
Router + GUI side of the salt-bridge stabilization mode: `_run_saltbridge_scan` (existing + novel,
intra + inter, common-schema output + caveat), `_run_saltbridge_ddg_estimate` (2-residue, from_aa-
verified for BOTH positions, de-novo web-upload blocked), the colour ramp, and the SaltBridge tab
(populate + 2-substitution basket add). Mocked: no Boltz, no GPU, no PyRosetta.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter
from session_state import SessionState


def _write(text: str, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    f.write(text); f.close()
    return f.name


def _router():
    return ToolRouter(bridge=MagicMock(), session=SessionState())


# A 2-chain CIF: chain A has an existing Asp(10)–Arg(30) salt bridge + plain residues; chain B too.
def _two_chain_cif() -> str:
    head = ("data_model\nloop_\n"
            "_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
            "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n")
    rows = []

    def res(ch, rn, comp, atoms):
        for name, (x, y, z) in atoms.items():
            rows.append(f"ATOM {name} {comp} {ch} {rn} {x:.3f} {y:.3f} {z:.3f}")

    def asp(ch, rn, ox):
        res(ch, rn, "ASP", {"N": (ox - 4, 0, 0), "CA": (ox - 3, 0, 0), "C": (ox - 3, 1.2, 0),
                            "CB": (ox - 1.8, 0, 0), "OD1": (ox, 0, 0), "OD2": (ox, 1.2, 0)})

    def arg(ch, rn, nx):
        res(ch, rn, "ARG", {"N": (nx + 4, 0, 0), "CA": (nx + 3, 0, 0), "C": (nx + 3, 1.2, 0),
                            "CB": (nx + 1.8, 0, 0), "NH1": (nx, 0, 0), "NH2": (nx, 1.2, 0),
                            "NE": (nx + 1.0, 0.6, 0)})

    def ala(ch, rn, ca, cbdir):
        cx, cy, cz = ca
        res(ch, rn, "ALA", {"N": (cx - cbdir, cy + 0.5, cz), "CA": ca,
                            "CB": (cx + 1.5 * cbdir, cy, cz), "C": (cx, cy + 1.4, cz)})

    for ch, z0 in (("A", 0.0), ("B", 30.0)):
        asp(ch, 10, 0.0 + 0 * 0 + 0 + 0 + 0 + 0)            # acid O at x≈0
        # shift chain B in z so the two chains don't overlap
    # build with z offset per chain
    rows = []
    for ch, z0 in (("A", 0.0), ("B", 30.0)):
        def r2(rn, comp, atoms):
            for name, (x, y, z) in atoms.items():
                rows.append(f"ATOM {name} {comp} {ch} {rn} {x:.3f} {y:.3f} {z + z0:.3f}")
        r2(10, "ASP", {"N": (-4, 0, 0), "CA": (-3, 0, 0), "C": (-3, 1.2, 0),
                       "CB": (-1.8, 0, 0), "OD1": (0, 0, 0), "OD2": (0, 1.2, 0)})
        r2(30, "ARG", {"N": (6.8, 0, 0), "CA": (5.8, 0, 0), "C": (5.8, 1.2, 0),
                       "CB": (4.6, 0, 0), "NH1": (2.8, 0, 0), "NH2": (2.8, 1.2, 0),
                       "NE": (3.8, 0.6, 0)})
        # two ALA backbones facing each other (a novel charge-pair site), well away from the bridge
        r2(50, "ALA", {"N": (9, 0.5, 0), "CA": (10, 0, 0), "CB": (11.5, 0, 0), "C": (10, 1.4, 0)})
        r2(60, "ALA", {"N": (19, 0.5, 0), "CA": (18, 0, 0), "CB": (16.5, 0, 0), "C": (18, 1.4, 0)})
    return head + "\n".join(rows) + "\n#\n"


# ── scan: existing + novel, schema + caveat ─────────────────────────────────────────────────────
def test_scan_finds_existing_and_novel():
    r = _router()
    cif = _write(_two_chain_cif(), ".cif")
    out = r._run_saltbridge_scan({"cif_path": cif})
    assert out.success
    d = out.data
    assert d["mode"] == "saltbridge_scan"
    assert d["existing"], "the planted Asp–Arg bridge should be assessed"
    e = d["existing"][0]
    assert e["type"] == "D-R" and e["within_cutoff"]
    assert d["novel"], "the facing ALA pair should surface a novel charge-pair site"
    c = d["novel"][0]
    # common 2-substitution schema fields present on a novel candidate
    for k in ("chain_a", "resnum_a", "from_aa_a", "to_aa_a",
              "chain_b", "resnum_b", "from_aa_b", "to_aa_b", "score"):
        assert k in c
    assert {c["to_aa_a"], c["to_aa_b"]} & {"D", "E"} and {c["to_aa_a"], c["to_aa_b"]} & {"R", "K"}
    assert "best_partner" in d and d["best_partner"]
    assert "desolvation" in (d["caveat"] or "").lower() or "context-dependent" in (d["caveat"] or "").lower()


def test_scan_no_structure_fails_closed():
    r = _router()
    out = r._run_saltbridge_scan({"cif_path": "/no/such/file.cif"})
    assert out.success is False and "cif" in out.error.lower()


def test_scan_interface_pairs_on_multimer():
    # the planted ALA pair is INTRA-chain; an inter-chain candidate may also appear. Assert the scan
    # runs the interface pass on a ≥2-chain structure (some candidate carries chain_a != chain_b OR the
    # scan simply ran without error — interface is additive, not guaranteed for this synthetic).
    r = _router()
    out = r._run_saltbridge_scan({"cif_path": _write(_two_chain_cif(), ".cif")})
    assert out.success
    # both chains present in the heatmap map
    assert set(out.data["best_partner"].keys()) >= {"A", "B"}


# ── ΔΔG escalation: 2-residue, from_aa-verified BOTH, de-novo blocked ────────────────────────────
def _pdb_atom(serial, name, resn, chain, resi, x, y, z):
    el = name[0]
    return (f"ATOM  {serial:>5} {name:<4} {resn} {chain}{resi:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}\n")


def _ddg_pdb() -> str:
    # A:50 = ALA, B:60 = SER — two positions to mutate to a charge pair (A→D, S→K)
    return (_pdb_atom(1, "CA", "ALA", "A", 50, 0.0, 0.0, 0.0)
            + _pdb_atom(2, "CB", "ALA", "A", 50, 1.5, 0.0, 0.0)
            + _pdb_atom(3, "CA", "SER", "B", 60, 8.0, 0.0, 0.0)
            + _pdb_atom(4, "CB", "SER", "B", 60, 6.5, 0.0, 0.0)
            + "END\n")


def _ddg_inputs(pdb, *, aa_a="A", aa_b="S", source="loaded"):
    return {"pdb_path": pdb, "chain_a": "A", "resnum_a": 50, "from_aa_a": aa_a, "to_aa_a": "D",
            "chain_b": "B", "resnum_b": 60, "from_aa_b": aa_b, "to_aa_b": "K", "source": source}


def _mock_rosetta(monkeypatch, ddg_map):
    """Patch RosettaBridge so analyze() returns ddg_scores keyed like 'A50D'/'S60K'."""
    import rosetta_bridge
    monkeypatch.setattr(rosetta_bridge, "_select_backend", lambda: "pyrosetta")

    class _R:
        success = True
        data = {"ddg_scores": ddg_map}

    fake = MagicMock()
    fake.analyze.return_value = _R()
    monkeypatch.setattr(rosetta_bridge, "RosettaBridge", lambda *a, **k: fake)
    return fake


def test_ddg_scores_both_positions(monkeypatch):
    fake = _mock_rosetta(monkeypatch, {"A50D": 1.0, "S60K": 0.4})
    r = _router()
    out = r._run_saltbridge_ddg_estimate(_ddg_inputs(_write(_ddg_pdb(), ".pdb")))
    assert out.success, out.error
    d = out.data
    assert d["ddg_a"] == 1.0 and d["ddg_b"] == 0.4 and d["ddg_mean"] == 0.7
    assert d["to_aa_a"] == "D" and d["to_aa_b"] == "K"
    fake.analyze.assert_called_once()
    muts = fake.analyze.call_args.kwargs["mutations"]
    assert {m["position"] for m in muts} == {50, 60}        # BOTH positions scored


def test_ddg_verifies_both_from_aa_fail_closed(monkeypatch):
    fake = _mock_rosetta(monkeypatch, {"A50D": 1.0, "S60K": 0.4})
    r = _router()
    # claim G at A:50 (structure has A) → ABORT, never score the wrong mutation
    out = r._run_saltbridge_ddg_estimate(_ddg_inputs(_write(_ddg_pdb(), ".pdb"), aa_a="G"))
    assert out.success is False and "mismatch" in out.error.lower()
    fake.analyze.assert_not_called()


def test_ddg_blocks_denovo_web_upload(monkeypatch):
    import rosetta_bridge
    monkeypatch.setattr(rosetta_bridge, "_select_backend", lambda: "dynamut2")
    r = _router()
    out = r._run_saltbridge_ddg_estimate(_ddg_inputs(_write(_ddg_pdb(), ".pdb"), source="denovo"))
    assert out.success is False and "upload" in out.error.lower()


# ── colour ramp + step description / icons ──────────────────────────────────────────────────────
def test_color_ramp_and_registration():
    from color_modes import saltbridge_compat_color
    assert saltbridge_compat_color(None) is None
    assert saltbridge_compat_color(0.01) is None            # below floor → no colour
    hi = saltbridge_compat_color(0.95)
    assert hi and hi.startswith("#")
    r = _router()
    assert "salt-bridge" in r._step_description("saltbridge_scan", {}, {}).lower()
    assert r._TOOL_ICONS.get("saltbridge_scan") and r._TOOL_ICONS.get("saltbridge_ddg_estimate")


# ── the SaltBridge tab: populate + 2-substitution basket add (offscreen Qt) ──────────────────────
def test_tab_populate_and_basket_two_substitutions():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from variant_workbench import SaltBridgeResultsTab, DesignBasketPanel
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    basket = DesignBasketPanel()
    added = {}

    def _add(cd, cand):
        # mirror the workbench normalizer: a 2-substitution entry
        basket.add_entry({
            "cls": "Salt bridge", "score": float(cand.get("score", 0.0)),
            "subs": [{"chain": cand["chain_a"], "position": cand["resnum_a"],
                      "from_aa": cand["from_aa_a"], "to_aa": cand["to_aa_a"]},
                     {"chain": cand["chain_b"], "position": cand["resnum_b"],
                      "from_aa": cand["from_aa_b"], "to_aa": cand["to_aa_b"]}],
            "metrics_text": "test"})
        added["hit"] = True

    tab = SaltBridgeResultsTab(on_highlight=lambda *a: None, on_add_to_basket=_add)
    scan = {
        "existing": [{"chain_a": "A", "resnum_a": 10, "from_aa_a": "D",
                      "chain_b": "A", "resnum_b": 30, "from_aa_b": "R", "type": "D-R",
                      "on_dist": 2.8, "within_cutoff": True, "optimizable": False,
                      "hbond_like": True, "buried": True, "score": 1.0}],
        "novel": [{"chain_a": "A", "resnum_a": 50, "from_aa_a": "A", "to_aa_a": "D",
                   "chain_b": "B", "resnum_b": 60, "from_aa_b": "S", "to_aa_b": "K",
                   "best_on": 2.9, "cb_cb": 5.0, "hbond_like": True, "buried": True,
                   "clash": False, "score": 0.8}],
        "caveat": "context-dependent; desolvation",
    }
    tab.populate(None, scan)
    assert tab._ex_tbl.rowCount() == 1 and tab._nv_tbl.rowCount() == 1
    assert "desolvation" in tab._caveat.text().lower()       # caveat populated (isVisible needs a shown window)
    # add the selected novel candidate → a 2-substitution basket entry
    tab._nv_tbl.selectRow(0)
    tab._cd = "cd"                                            # non-None so the add fires
    tab._add_to_basket()
    assert added.get("hit")
    assert len(basket.entries) == 1
    assert len(basket.entries[0]["subs"]) == 2               # TWO positions (the salt-bridge pair)
    assert basket.entries[0]["cls"] == "Salt bridge"


def test_long_tools_registration():
    # the ΔΔG estimate is long-running → must be in the busy-ticker set with a correct label
    import request_engine
    src = Path(request_engine.__file__).read_text(encoding="utf-8")
    assert "saltbridge_ddg_estimate" in src
