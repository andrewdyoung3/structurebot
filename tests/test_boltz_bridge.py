"""
tests/test_boltz_bridge.py
--------------------------
Boltz-2 bridge — the LOCAL-ONLY multimer fold engine. The WSL subprocess is mocked; these
test the pure surfaces: the fail-closed LOCAL-ONLY guard (refinement 1 — the breach backstop),
the MSA-free YAML build, the confidence/CIF parse, the capability flag, and seed plumbing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from boltz_bridge import BoltzBridge, boltz_available, _RemoteBreach, _SOURCE


# ── A minimal mmCIF with an _atom_site loop + B-factor (pLDDT) ────────────────────────
_FAKE_CIF = """data_model
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 CA MET A A 1 0.0 0.0 0.0 90.0
ATOM 2 CA LYS A A 2 1.0 0.0 0.0 80.0
ATOM 3 CA VAL A A 3 2.0 0.0 0.0 40.0
ATOM 4 CA MET B B 1 0.0 5.0 0.0 70.0
#
"""

_FAKE_CONF = {
    "confidence_score": 0.95, "ptm": 0.96, "iptm": 0.959,
    "complex_plddt": 0.80, "chains_ptm": {"0": 0.97, "1": 0.97},
}


def _bridge():
    b = BoltzBridge()
    b._wsl = MagicMock()
    b._wsl.translate_path = lambda p: p          # identity (we run "WSL" on local paths)
    return b


# ── Fail-closed LOCAL-ONLY guard (refinement 1) ──────────────────────────────────────

class TestLocalOnlyGuardFailClosed:
    def test_refuses_use_msa_server_flag(self):
        with pytest.raises(_RemoteBreach):
            BoltzBridge._assert_local_only(
                "python -m boltz predict x.yaml --use_msa_server", "msa: empty\n", False)

    def test_refuses_msa_server_url_flag(self):
        with pytest.raises(_RemoteBreach):
            BoltzBridge._assert_local_only(
                "python -m boltz predict x.yaml --msa_server_url http://h", "msa: empty\n", False)

    def test_refuses_non_empty_msa_in_yaml(self):
        with pytest.raises(_RemoteBreach):
            BoltzBridge._assert_local_only(
                "python -m boltz predict x.yaml --seed 0", "msa: somefile.a3m\n", False)

    def test_refuses_yaml_with_no_msa_declaration(self):
        # no `msa:` at all → Boltz would auto-MSA via the remote server → refuse
        with pytest.raises(_RemoteBreach):
            BoltzBridge._assert_local_only(
                "python -m boltz predict x.yaml --seed 0", "version: 1\nsequences: []\n", False)

    def test_passes_clean_msa_empty(self):
        # no raise on a proper MSA-free command + YAML
        BoltzBridge._assert_local_only(
            "python -m boltz predict x.yaml --accelerator gpu --seed 0",
            "msa: empty\nmsa: empty\n", False)

    def test_predict_refuses_breach_with_NO_subprocess(self, monkeypatch):
        # a breach must short-circuit BEFORE any WSL exec — no subprocess spawned.
        b = _bridge()
        # force a breach: YAML build emits a non-empty msa
        monkeypatch.setattr(b, "_build_yaml",
                            lambda chains: "version: 1\nsequences:\n  - protein:\n      msa: remote\n")
        res = b.predict([{"id": "A", "sequence": "MK"}])
        assert res["success"] is False
        assert "LOCAL-ONLY breach refused" in res["error"]
        b._wsl.run_command.assert_not_called()       # the critical fail-closed assertion


# ── MSA-free YAML build ───────────────────────────────────────────────────────────────

class TestYaml:
    def test_one_protein_block_per_chain_all_msa_empty(self):
        y = BoltzBridge._build_yaml([{"id": "A", "sequence": "MK"}, {"id": "B", "sequence": "MK"}])
        assert y.count("- protein:") == 2
        assert y.count("msa: empty") == 2            # every chain MSA-free
        assert "id: A" in y and "id: B" in y and "sequence: MK" in y


# ── confidence/CIF parse ──────────────────────────────────────────────────────────────

class TestParse:
    def _outdir(self, tmp_path):
        d = tmp_path / "out" / "predictions" / "boltz_in"
        d.mkdir(parents=True)
        (d / "boltz_in_model_0.cif").write_text(_FAKE_CIF)
        (d / "confidence_boltz_in_model_0.json").write_text(json.dumps(_FAKE_CONF))
        return str(tmp_path / "out")

    def test_parse_mean_plddt_iptm_perchain(self, tmp_path):
        b = _bridge()
        out = b._parse_outputs(self._outdir(tmp_path), [{"id": "A", "sequence": "MKV"}])
        assert out["success"] is True
        assert out["mean_plddt"] == 80.0             # complex_plddt 0.80 × 100
        assert out["iptm"] == 0.959
        assert out["chains_ptm"] == {"0": 0.97, "1": 0.97}
        assert os.path.isfile(out["cif_path"]) and out["cif_path"].endswith(".cif")

    def test_cif_bfactor_rep_chain_1based(self, tmp_path):
        b = _bridge()
        out = b._parse_outputs(self._outdir(tmp_path), [{"id": "A", "sequence": "MKV"}])
        # rep chain A only, keyed 1..N over its CA residues; chain B excluded
        assert out["plddt"] == {1: 90.0, 2: 80.0, 3: 40.0}

    def test_parse_per_chain_plddt_and_observed_chain_ids(self, tmp_path):
        b = _bridge()
        out = b._parse_outputs(self._outdir(tmp_path), [{"id": "A", "sequence": "MKV"}])
        # EACH chain keyed 1..N over its OWN CA residues (hetero re-point needs this)
        assert out["plddt_by_chain"] == {"A": {1: 90.0, 2: 80.0, 3: 40.0}, "B": {1: 70.0}}
        # observed chain ids in CIF first-appearance order (the read-back / ptm-alignment key)
        assert out["chain_ids"] == ["A", "B"]

    def test_missing_cif_is_error(self, tmp_path):
        b = _bridge()
        (tmp_path / "out").mkdir()
        out = b._parse_outputs(str(tmp_path / "out"), [{"id": "A", "sequence": "M"}])
        assert out["success"] is False


# ── Capability flag (Unit-B) ──────────────────────────────────────────────────────────

class TestCapability:
    def test_disabled_via_env(self, monkeypatch):
        b = BoltzBridge()
        b._enable = "false"
        assert b.is_available() is False

    def test_available_when_import_chain_ok(self, monkeypatch):
        import boltz_bridge
        b = _bridge()
        b._wsl.is_available.return_value = True
        monkeypatch.setattr(boltz_bridge, "_AVAIL_CACHE", {}, raising=False)
        import dep_probe
        monkeypatch.setattr(dep_probe, "wsl_import_probe", lambda *a, **k: True)
        assert b.is_available() is True

    def test_unavailable_when_import_chain_fails(self, monkeypatch):
        import boltz_bridge
        b = _bridge()
        b._wsl.is_available.return_value = True
        monkeypatch.setattr(boltz_bridge, "_AVAIL_CACHE", {}, raising=False)
        import dep_probe
        monkeypatch.setattr(dep_probe, "wsl_import_probe", lambda *a, **k: False)
        assert b.is_available() is False


# ── End-to-end predict flow (WSL mocked to write the fake outputs) ────────────────────

class TestPredictFlow:
    def test_predict_success_source_and_seed(self, monkeypatch):
        b = _bridge()

        def _fake_run(cmd, timeout=None):
            # parse --out_dir from the command and write the fake CIF + confidence there
            toks = cmd.split()
            out = toks[toks.index("--out_dir") + 1].strip("'\"")
            d = Path(out) / "predictions" / "boltz_in"
            d.mkdir(parents=True, exist_ok=True)
            (d / "boltz_in_model_0.cif").write_text(_FAKE_CIF)
            (d / "confidence_boltz_in_model_0.json").write_text(json.dumps(_FAKE_CONF))
            return {"ok": True, "stdout": "", "stderr": ""}

        b._wsl.run_command.side_effect = _fake_run
        res = b.predict([{"id": "A", "sequence": "MKV"}, {"id": "B", "sequence": "MKV"}], seed=7)
        assert res["success"] is True
        assert res["source"] == _SOURCE
        assert res["seed"] == 7
        assert res["iptm"] == 0.959 and res["mean_plddt"] == 80.0
        # the command that ran is MSA-free (no remote flag) and seed-pinned
        ran_cmd = b._wsl.run_command.call_args.args[0]
        assert "--use_msa_server" not in ran_cmd and "--seed 7" in ran_cmd


def test_boltz_available_swallows_errors(monkeypatch):
    import boltz_bridge
    monkeypatch.setattr(boltz_bridge.BoltzBridge, "is_available",
                        lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    assert boltz_available() is False
