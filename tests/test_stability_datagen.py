"""
Machinery tests for the multi-protein data-gen harness (scripts/stability_datagen.py).

These verify the GENERALIZED harness plumbing — manifest-driven multi-protein
assembly, provenance threading, crash-safe resume (full + partial), Rosetta/DynaMut2
backend-distinctness, and provenance-aware DynaMut2 subsetting — WITHOUT running any
real voter (GPU/WSL/remote API are all stubbed).  No large run is launched; this is
the "machinery only" verification the benchmark-prep brief asks for.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import stability_datagen as sdg  # noqa: E402


# ── fixtures: a tiny 2-protein manifest + stubbed voters ─────────────────────────

def _write_manifest(tmp_path: Path) -> Path:
    struct = tmp_path / "structure" / "raw"
    struct.mkdir(parents=True)
    (struct / "AAAA.pdb").write_text("dummy", encoding="utf-8")
    (struct / "BBBB.pdb").write_text("dummy", encoding="utf-8")
    exp = tmp_path / "exp.csv"
    exp.write_text("pdbid,chainid,variant,score\n"
                   "AAAA,A,A10G,1.0\nAAAA,A,L20P,-0.5\nBBBB,A,V5A,2.0\nCCCC,A,M1K,0.3\n",
                   encoding="utf-8")
    ref = tmp_path / "ref.csv"
    ref.write_text("pdbid,chainid,variant,score\nAAAA,A,A10G,0.8\n", encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "entries": [
            {"set": "TST", "pdbid": "AAAA", "chain": "A",
             "exp_csv": str(exp), "rosetta_ref_csv": str(ref), "struct_dir": str(struct),
             "provenance": {"thermompnn": "clean", "dynamut2": "clean",
                            "rasp": "circular_vs_rosetta", "rosetta": "clean"},
             "proposed_include": True, "role": "diversity_core"},
            {"set": "TST", "pdbid": "BBBB", "chain": "A",
             "exp_csv": str(exp), "rosetta_ref_csv": str(ref), "struct_dir": str(struct),
             "provenance": {"thermompnn": "clean", "dynamut2": "training",
                            "rasp": "circular_vs_rosetta", "rosetta": "clean"},
             "proposed_include": True, "role": "antisymmetry_fwd"},
            {"set": "TST", "pdbid": "CCCC", "chain": "A",
             "exp_csv": str(exp), "rosetta_ref_csv": str(ref), "struct_dir": str(struct),
             "provenance": {"thermompnn": "training", "dynamut2": "unknown",
                            "rasp": "circular_vs_rosetta", "rosetta": "clean"},
             "proposed_include": False, "role": "excluded_demo"},
        ],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return mpath


@pytest.fixture
def stub_voters(monkeypatch):
    """Replace the heavy voters with deterministic canned values; record Rosetta/
    DynaMut2 backends so the test can assert they stay distinct."""
    backends = []

    def fake_wc(pdb_path, chain, group, log):
        ck = sdg.candidate_key
        return {
            "thermo": {ck(chain, m["resnum"], m["wt"], m["mut"]): 0.11 for m in group},
            "rasp": {ck(chain, m["resnum"], m["wt"], m["mut"]): 0.22 for m in group},
            "camsol_pos": {m["resnum"]: 0.30 for m in group},
            "esm_pos": {m["resnum"]: 0.50 for m in group},
            "resnum2seq": {m["resnum"]: m["resnum"] for m in group},
        }

    def fake_ros(pdb_path, chain, muts, backend, log):
        backends.append(backend)
        val = 3.3 if backend == "local" else 0.44
        return {f"{m['wt']}{m['resnum']}{m['mut']}": val for m in muts}

    monkeypatch.setattr(sdg, "_whole_chain_voters", fake_wc)
    monkeypatch.setattr(sdg, "_rosetta_batch", fake_ros)
    return backends


# ── assembly ─────────────────────────────────────────────────────────────────────

def test_assemble_multi_protein_carries_provenance(tmp_path):
    mpath = _write_manifest(tmp_path)
    muts = sdg.assemble_from_manifest(str(mpath), proposed_only=True, log=lambda *_: None)
    # proposed_only drops CCCC (proposed_include False)
    assert {m["pdbid"] for m in muts} == {"AAAA", "BBBB"}
    assert sum(1 for m in muts if m["pdbid"] == "AAAA") == 2
    assert sum(1 for m in muts if m["pdbid"] == "BBBB") == 1
    a10g = next(m for m in muts if m["variant"] == "A10G")
    assert a10g["exp_ddg"] == 1.0
    assert a10g["rosetta_ref_ddg"] == 0.8                 # ref attached by variant
    assert a10g["provenance"]["dynamut2"] == "clean"
    l20p = next(m for m in muts if m["variant"] == "L20P")
    assert l20p["rosetta_ref_ddg"] is None                # no ref row → None, not faked


def test_assemble_all_entries_includes_excluded(tmp_path):
    mpath = _write_manifest(tmp_path)
    muts = sdg.assemble_from_manifest(str(mpath), proposed_only=False, log=lambda *_: None)
    assert "CCCC" in {m["pdbid"] for m in muts}


# ── run + provenance threading + backend distinctness ────────────────────────────

def test_run_writes_rows_with_provenance_and_distinct_backends(tmp_path, stub_voters):
    mpath = _write_manifest(tmp_path)
    muts = sdg.assemble_from_manifest(str(mpath), log=lambda *_: None)
    keys = {sdg._key(m) for m in muts}
    out = tmp_path / "rows.jsonl"
    n = sdg.run(muts, str(out), rosetta_keys=keys, dynamut2_keys=keys, log=lambda *_: None)
    assert n == 3
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert {r["pdbid"] for r in rows} == {"AAAA", "BBBB"}
    r = next(r for r in rows if r["variant"] == "A10G")
    assert r["rosetta_ddg"] == 3.3 and r["dynamut2_ddg"] == 0.44   # local vs dynamut2
    assert r["thermompnn_ddg"] == 0.11 and r["rasp_ddg"] == 0.22
    assert r["camsol_score"] == 0.30 and r["esm_tolerance"] == 0.5
    assert r["prov_rosetta"] == "clean" and r["role"] == "diversity_core"
    # Rosetta ran on the "local" backend, DynaMut2 on "dynamut2" — never conflated.
    assert "local" in stub_voters and "dynamut2" in stub_voters


# ── resume (crash-safe) ──────────────────────────────────────────────────────────

def test_resume_full_skips_all(tmp_path, stub_voters):
    mpath = _write_manifest(tmp_path)
    muts = sdg.assemble_from_manifest(str(mpath), log=lambda *_: None)
    keys = {sdg._key(m) for m in muts}
    out = tmp_path / "rows.jsonl"
    sdg.run(muts, str(out), keys, keys, log=lambda *_: None)
    again = sdg.run(muts, str(out), keys, keys, log=lambda *_: None)
    assert again == 0                                      # nothing recomputed
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_resume_partial_recomputes_only_missing(tmp_path, stub_voters):
    mpath = _write_manifest(tmp_path)
    muts = sdg.assemble_from_manifest(str(mpath), log=lambda *_: None)
    keys = {sdg._key(m) for m in muts}
    out = tmp_path / "rows.jsonl"
    sdg.run(muts, str(out), keys, keys, log=lambda *_: None)
    lines = out.read_text(encoding="utf-8").splitlines()
    out.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")   # drop 1 row
    n = sdg.run(muts, str(out), keys, keys, log=lambda *_: None)
    assert n == 1                                          # exactly the missing one
    final = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(final) == 3
    assert len({r["key"] for r in final}) == 3            # no duplicates


# ── provenance-aware DynaMut2 subsetting ─────────────────────────────────────────

def test_dynamut2_subset_prefers_clean_proteins(tmp_path):
    mpath = _write_manifest(tmp_path)
    muts = sdg.assemble_from_manifest(str(mpath), log=lambda *_: None)
    ros_keys, dyn_keys = sdg._select_subsets(muts, rosetta_cap=0, dynamut2_cap=0)
    bbbb = next(m for m in muts if m["pdbid"] == "BBBB")   # dynamut2='training'
    aaaa = next(m for m in muts if m["pdbid"] == "AAAA")   # dynamut2='clean'
    assert sdg._key(bbbb) in ros_keys                      # Rosetta = anchor on all
    assert sdg._key(bbbb) not in dyn_keys                  # DynaMut2 skips training protein
    assert sdg._key(aaaa) in dyn_keys                      # DynaMut2 kept on clean protein
