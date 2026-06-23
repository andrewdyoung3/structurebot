"""
tests/test_session_figures.py
-----------------------------
Phase 2 — relevance-gated profile figures. A plot ONLY for per-residue profiles (pLDDT,
deviation, d_flex); scalars get none; a profile type with no data renders nothing (no empty
images); figures dir created only when something is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import session_figures


def test_renders_only_profile_types(tmp_path):
    tables = {
        "fold_plddt": [{"model": "m", "design_chain": "A", "row": "T", "resnum": i, "plddt": 90 - i}
                       for i in range(1, 6)],
        "deviation": [{"model": "m", "design_chain": "A", "variant": "V1", "resnum": i,
                       "dRMSD": 0.2 * i, "dRMSD_floor": 0.3, "lDDT": 0.99 - 0.02 * i, "lDDT_floor": 0.9}
                      for i in range(1, 6)],
        "template_assist_dflex": [{"model": "m", "design_chain": "A", "resnum": i, "d_flex": 0.1 * i}
                                  for i in range(1, 6)],
        # scalar types — must NOT yield figures
        "solubility": [{"model": "m", "design_chain": "A", "variant": "V1", "delta": 0.1}],
        "structural_align": [{"model": "m", "design_chain": "A", "tm_ref": 0.9}],
    }
    figs = tmp_path / "figures"
    rep = session_figures.render_profile_figures(tables, figs)
    assert rep["error"] is None
    names = set(rep["written"])
    assert any(n.startswith("plddt_") for n in names)
    assert any(n.startswith("deviation_") for n in names)
    assert any(n.startswith("dflex_") for n in names)
    # exactly the three profiles, no scalar figure
    assert len(names) == 3
    for n in names:
        assert (figs / n).is_file() and (figs / n).stat().st_size > 1000   # a real PNG, not empty


def test_no_profile_data_writes_no_images(tmp_path):
    tables = {"solubility": [{"model": "m", "design_chain": "A", "variant": "V1", "delta": 0.1}],
              "structural_align": [{"model": "m", "design_chain": "A", "tm_ref": 0.9}]}
    figs = tmp_path / "figures"
    rep = session_figures.render_profile_figures(tables, figs)
    assert rep["written"] == []
    assert not figs.exists()                       # no empty figures dir / placeholder images
    assert set(rep["skipped"]) == set(session_figures.PROFILE_TYPES)


def test_export_session_emits_figures(tmp_path):
    import session_export
    ds = {"m": {"source": "sequence", "chains": {"c": {"rep_chain": "A",
        "template_cells": [{"col": i, "resnum": i + 1, "aa": "A"} for i in range(3)],
        "template_fold": {"engine": "boltz", "target": "monomer",
                          "plddt": {1: 90.0, 2: 88.0, 3: 91.0}},
        "variants": []}}}}
    rep = session_export.export_session(ds, tmp_path)
    assert rep["any"] is True
    assert rep["figures"]["written"]               # the fold pLDDT profile produced a plot
    assert (tmp_path / "figures").is_dir()
    assert any((tmp_path / "figures").glob("plddt_*.png"))
