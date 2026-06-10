"""
Tests for the dual-output (Raw + Calibrated) candidate exporter
(scripts/stability_export.py).

Verify the lossless invariant (Raw columns copied verbatim), the identity-passthrough
calibration (Calibrated == Raw, tagged uncalibrated), property/ddG column separation,
and a real round-trip through both CSV (dependency-free) and XLSX (openpyxl).
"""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stability_export as sx  # noqa: E402


def _sample_records():
    return [
        {"key": "1ABC:A:A10G", "set": "S669", "role": "diversity_core",
         "pdbid": "1ABC", "chain": "A", "resnum": 10, "wt": "A", "mut": "G",
         "variant": "A10G", "exp_ddg": 1.0, "rosetta_ref_ddg": 0.8,
         "rosetta_ddg": 3.3, "rasp_ddg": 0.9, "thermompnn_ddg": -0.4,
         "dynamut2_ddg": None, "camsol_score": 0.3, "esm_tolerance": 0.5,
         "prov_rosetta": "clean", "prov_rasp": "circular_vs_rosetta",
         "prov_thermompnn": "clean", "prov_dynamut2": "clean",
         "rosetta_in_subset": True, "dynamut2_in_subset": False},
        # a legacy-shaped record: no provenance/role keys → must degrade to None
        {"key": "1PGA:A:M1A", "set": "Protein_G", "pdbid": "1PGA", "chain": "A",
         "resnum": 1, "wt": "M", "mut": "A", "variant": "M1A", "exp_ddg": -0.14,
         "rosetta_ddg": 4.2, "rasp_ddg": 0.95, "thermompnn_ddg": -0.45,
         "dynamut2_ddg": None, "camsol_score": -0.9, "esm_tolerance": 0.08},
    ]


def _write_jsonl(tmp_path, records):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


# ── calibration contract ─────────────────────────────────────────────────────────

def test_calibrate_is_identity_and_lossless():
    for v in sx.DDG_VOTERS:
        val, meta = sx.calibrate(v, 2.5)
        assert val == 2.5                          # identity passthrough
        assert meta["version"] == sx.CAL_VERSION
        assert meta["provenance"] == "uncalibrated"
        assert meta["offset"] is None and meta["slope"] is None
    none_val, _ = sx.calibrate("rosetta", None)
    assert none_val is None                         # not_computed stays None, never 0.0


def test_property_and_ddg_columns_are_separated():
    names = [c for _g, c in sx._columns()]
    raw_cols = {f"{v}_ddg_raw" for v in sx.DDG_VOTERS}
    cal_cols = {f"{v}_ddg_cal" for v in sx.DDG_VOTERS}
    prop_cols = {"property_camsol_solubility", "property_esm_fitness_tolerance"}
    assert raw_cols <= set(names) and cal_cols <= set(names)
    assert prop_cols <= set(names)
    # property columns are NOT inside the ddG raw/cal groups
    assert not (prop_cols & raw_cols) and not (prop_cols & cal_cols)


# ── CSV round-trip (dependency-free) ─────────────────────────────────────────────

def test_csv_roundtrip_raw_lossless_and_cal_identity(tmp_path):
    recs = _sample_records()
    jp = _write_jsonl(tmp_path, recs)
    out = tmp_path / "candidates.csv"
    n = sx.export(str(jp), str(out), log=lambda *_: None)
    assert n == 2
    with open(out, newline="", encoding="utf-8") as fh:
        rdr = list(csv.reader(fh))
    group_band, header, data = rdr[0], rdr[1], rdr[2:]
    assert "ddG RAW (lossless — never altered)" in group_band
    idx = {nm: i for i, nm in enumerate(header)}
    by_key = {row[idx["key"]]: row for row in data}
    r0 = by_key["1ABC:A:A10G"]
    # RAW copied verbatim; CALIBRATED equals RAW (identity)
    assert r0[idx["rosetta_ddg_raw"]] == "3.3" and r0[idx["rosetta_ddg_cal"]] == "3.3"
    assert r0[idx["thermompnn_ddg_raw"]] == "-0.4"
    # not_computed (None) stays empty, never 0.0
    assert r0[idx["dynamut2_ddg_raw"]] == "" and r0[idx["dynamut2_ddg_cal"]] == ""
    # property axes carried into their own columns
    assert r0[idx["property_camsol_solubility"]] == "0.3"
    assert r0[idx["property_esm_fitness_tolerance"]] == "0.5"
    # calibration metadata present + tagged uncalibrated
    assert r0[idx["cal_version"]] == sx.CAL_VERSION
    assert "UNCALIBRATED" in r0[idx["cal_status"]]
    # legacy record without provenance → empty prov cell, not a crash
    r1 = by_key["1PGA:A:M1A"]
    assert r1[idx["prov_rosetta"]] == ""


# ── XLSX round-trip (openpyxl) ───────────────────────────────────────────────────

def test_xlsx_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    recs = _sample_records()
    jp = _write_jsonl(tmp_path, recs)
    out = tmp_path / "candidates.xlsx"
    sx.export(str(jp), str(out), log=lambda *_: None)
    wb = openpyxl.load_workbook(out, read_only=True)
    ws = wb.active
    vals = list(ws.values)
    header, data = vals[1], vals[2:]
    idx = {nm: i for i, nm in enumerate(header)}
    by_key = {row[idx["key"]]: row for row in data}
    r0 = by_key["1ABC:A:A10G"]
    assert r0[idx["rosetta_ddg_raw"]] == 3.3 and r0[idx["rosetta_ddg_cal"]] == 3.3
    assert r0[idx["dynamut2_ddg_raw"]] is None        # not_computed preserved
    assert r0[idx["property_esm_fitness_tolerance"]] == 0.5
