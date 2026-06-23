"""
tests/test_session_export.py
----------------------------
The session EXPORTS core (session_export) — one row-building pass feeding BOTH the multi-tab
workbook and per-type CSVs, the partial-data Summary roll-up (blank cells, never 0/crash), and
FAIL-LOUD skip-empty (no header-only files; nothing-has-data → no files).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import session_export
from session_export import build_tables, export_session, _COLUMNS, _TITLE


def _mixed_sessions():
    """A MIXED session: a construct with template-assist + structural-align; V1 folded + deviation
    + stability; V2 solubility-only (partial); V3 no results at all."""
    return {"denovo-1": {"model_id": "denovo-1", "source": "sequence", "chains": {"c1": {
        "rep_chain": "A", "members": [["3", "A"]],
        "template_fold": {"engine": "boltz", "target": "monomer", "mean_plddt": 90.0,
                          "plddt": {1: 90.0, 2: 88.0}},
        "guided_fold": {},
        "template_assist": {"template_label": "8UB2", "unguided_mean_plddt": 92.0,
                            "guided_mean_plddt": 95.0, "d_plddt": 3.0, "n_stabilized": 2,
                            "n_residues": 2, "mean_d_flex": 0.1, "max_adoption": 0.93,
                            "tm_adopt": 0.9, "force": False, "threshold": None,
                            "d_flex": {1: 0.4, 2: 0.1}},
        "structural_align": {"reference": "1AXC", "ref_label": "1AXC", "tm_ref": 0.90,
                             "tm_query": 0.88, "rmsd": 1.1, "n_aligned": 2, "norm": "ref"},
        "variants": [
            {"id": "V1", "mutations": [{"resnum": 2, "from_aa": "C", "to_aa": "W", "source": "manual"}],
             "results": {"fold": {"engine": "boltz", "target": "monomer", "mean_plddt": 89.0,
                                  "plddt": {1: 89.0, 2: 87.0},
                                  "deviation": {"ddm": {"1": 0.5, "2": 2.0}, "lddt": {"1": 0.95, "2": 0.7},
                                                "floor_ddm": {"1": 0.3, "2": 0.3},
                                                "floor_lddt": {"1": 0.9, "2": 0.9}, "multichain": False}},
                         "stability": {"rows": [{"resnum": 2, "from_aa": "C", "to_aa": "W", "ddg": 1.2,
                                                 "ddg_source": "rosetta", "combined_score": 0.5,
                                                 "recommendation": "ok"}], "sum_ddg": 1.2, "tier": "deep"},
                         "solubility": None}},
            {"id": "V2", "mutations": [{"resnum": 1, "from_aa": "A", "to_aa": "S", "source": "manual"}],
             "results": {"fold": None, "stability": None,
                         "solubility": {"variant": 1.1, "wt": 1.0, "delta": 0.1}}},
            {"id": "V3", "mutations": [], "results": {"fold": None, "stability": None, "solubility": None}},
        ]}}}}


def _read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_export_writes_workbook_and_csvs_with_matching_columns(tmp_path):
    rep = export_session(_mixed_sessions(), tmp_path)
    assert rep["any"] is True
    xlsx = tmp_path / "results.xlsx"
    assert xlsx.is_file()

    import openpyxl
    wb = openpyxl.load_workbook(str(xlsx))
    # every populated result type present as a sheet (+ Summary); none empty
    expected = ["Summary", "Fold pLDDT", "Deviation", "Stability ddG", "Solubility",
                "Template assist", "Template assist dflex", "Structural align"]
    assert wb.sheetnames == expected
    # workbook sheet headers == CSV headers == _COLUMNS for each type
    for key, cols in _COLUMNS.items():
        ws = wb[_TITLE[key]]
        assert [c.value for c in ws[1]] == cols
        csv_rows = _read_csv(tmp_path / "csv" / f"{key}.csv")
        assert csv_rows[0] == cols                      # same columns in the CSV


def test_summary_partial_data_blank_never_zero(tmp_path):
    export_session(_mixed_sessions(), tmp_path)
    rows = _read_csv(tmp_path / "csv" / "summary.csv")
    hdr, data = rows[0], rows[1:]
    by_row = {r[hdr.index("row")]: dict(zip(hdr, r)) for r in data}
    assert set(by_row) == {"T", "V1", "V2"}             # V3 (no results) absent

    v1 = by_row["V1"]
    assert v1["mean_plddt"] == "89.0" and v1["sum_ddg"] == "1.2" and v1["max_dRMSD"] == "2.0"
    assert v1["solubility_delta"] == ""                 # missing → BLANK, not 0
    assert v1["adoption"] == "0.93" and v1["tm_align"] == "0.9"

    v2 = by_row["V2"]                                    # solubility-only variant
    assert v2["solubility_delta"] == "0.1"
    assert v2["mean_plddt"] == "" and v2["sum_ddg"] == "" and v2["max_dRMSD"] == ""

    t = by_row["T"]                                      # construct row: assist/align present
    assert t["mean_plddt"] == "90.0" and t["adoption"] == "0.93" and t["tm_align"] == "0.9"
    assert t["sum_ddg"] == ""


def test_deviation_keeps_both_metrics_and_floors(tmp_path):
    export_session(_mixed_sessions(), tmp_path)
    rows = _read_csv(tmp_path / "csv" / "deviation.csv")
    hdr, data = rows[0], rows[1:]
    assert hdr == _COLUMNS["deviation"]
    d = {r[hdr.index("resnum")]: dict(zip(hdr, r)) for r in data}
    assert d["2"]["dRMSD"] == "2.0" and d["2"]["dRMSD_floor"] == "0.3"
    assert d["2"]["lDDT"] == "0.7" and d["2"]["lDDT_floor"] == "0.9"


def test_failloud_skips_empty_types(tmp_path):
    # a session with ONLY solubility data → deviation/stability/fold/assist/align all skipped
    ds = {"m": {"source": "sequence", "chains": {"c": {"rep_chain": "A", "variants": [
        {"id": "V1", "mutations": [], "results": {"solubility": {"variant": 1.0, "wt": 0.9, "delta": 0.1}}}]}}}}
    rep = export_session(ds, tmp_path)
    assert rep["any"] is True
    assert "Solubility" in rep["written"] and "Deviation" in rep["skipped"]
    assert not (tmp_path / "csv" / "deviation.csv").exists()      # no empty file
    import openpyxl
    wb = openpyxl.load_workbook(str(tmp_path / "results.xlsx"))
    assert wb.sheetnames == ["Summary", "Solubility"]


def test_nothing_to_export_writes_no_files(tmp_path):
    ds = {"m": {"source": "sequence", "chains": {"c": {"rep_chain": "A", "variants": [
        {"id": "V1", "mutations": [], "results": {}}]}}}}
    rep = export_session(ds, tmp_path)
    assert rep["any"] is False and rep["files"] == []
    assert not (tmp_path / "results.xlsx").exists()
    assert not (tmp_path / "csv").exists()


def test_build_tables_no_recompute_is_pure():
    ds = _mixed_sessions()
    import copy
    snap = copy.deepcopy(ds)
    build_tables(ds)
    assert ds == snap                                    # input not mutated
