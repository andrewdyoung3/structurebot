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
                            lambda chains, templates=None:
                            "version: 1\nsequences:\n  - protein:\n      msa: remote\n")
        res = b.predict([{"id": "A", "sequence": "MK"}])
        assert res["success"] is False
        assert "LOCAL-ONLY breach refused" in res["error"]
        b._wsl.run_command.assert_not_called()       # the critical fail-closed assertion


# ── Swallowed-error surfacing (Boltz exits 0 but writes no model) ─────────────────────

class TestSwallowedError:
    def test_extract_picks_final_exception_line(self):
        stderr = ("  0%|          | 0/1 [00:00<?, ?it/s]\n"
                  "Traceback (most recent call last):\n"
                  '  File ".../schema.py", line 1691, in parse_boltz_schema\n'
                  "    raise ValueError(msg)\n"
                  "ValueError: Template chain A assigned for template4HHB is not one of the protein chains!\n"
                  "100%|██████████| 1/1 [00:00<00:00,  6.70it/s]\n")
        got = BoltzBridge._extract_boltz_error("", stderr)
        assert "Template chain A" in got and got.startswith("ValueError")

    def test_extract_keyerror(self):
        got = BoltzBridge._extract_boltz_error("", "KeyError: 'Axp'\n")
        assert got.startswith("KeyError")

    def test_extract_empty_when_no_error(self):
        assert BoltzBridge._extract_boltz_error("all good\n", "100%|████| 1/1 [00:00<00:00]\n") in (
            "all good", "")          # no exception-shaped line → last meaningful or empty

    def test_predict_surfaces_swallowed_error_on_no_cif(self, monkeypatch, tmp_path):
        # Boltz exits 0 (ok=True) but writes no CIF + a swallowed ValueError in stderr → the
        # returned error must include the REAL cause, not just "no predicted CIF".
        b = _bridge()
        b._wsl.run_command = MagicMock(return_value={
            "ok": True, "stdout": "",
            "stderr": "ValueError: Template chain A is not one of the protein chains!\n"})
        monkeypatch.setattr(b, "_parse_outputs",
                            lambda out_dir, chains: {"success": False,
                                                     "error": "no predicted CIF in the Boltz output"})
        res = b.predict([{"id": "A", "sequence": "MK"}])
        assert res["success"] is False
        assert "no predicted CIF" in res["error"]
        assert "Boltz error:" in res["error"] and "Template chain A" in res["error"]


# ── MSA-free YAML build ───────────────────────────────────────────────────────────────

class TestYaml:
    def test_one_protein_block_per_chain_all_msa_empty(self):
        y = BoltzBridge._build_yaml([{"id": "A", "sequence": "MK"}, {"id": "B", "sequence": "MK"}])
        assert y.count("- protein:") == 2
        assert y.count("msa: empty") == 2            # every chain MSA-free
        assert "id: A" in y and "id: B" in y and "sequence: MK" in y

    def test_no_templates_block_when_absent(self):
        y = BoltzBridge._build_yaml([{"id": "A", "sequence": "MK"}])
        assert "templates:" not in y                 # plain de-novo fold unchanged


# ── TEMPLATE-GUIDED YAML injection ────────────────────────────────────────────────────

class TestTemplateYaml:
    def test_soft_template_block_fields(self):
        y = BoltzBridge._build_yaml(
            [{"id": "A", "sequence": "MK"}],
            [{"cif": "/wsl/path/t.cif", "chain_id": "A"}])
        assert "templates:" in y
        assert "- cif: /wsl/path/t.cif" in y
        assert "chain_id: A" in y
        assert "force:" not in y                      # soft = no force/threshold
        assert "threshold:" not in y
        # every chain still MSA-free → the LOCAL-ONLY guard is unaffected
        assert y.count("msa: empty") == 1

    def test_hard_template_emits_force_and_threshold(self):
        y = BoltzBridge._build_yaml(
            [{"id": "A", "sequence": "MK"}],
            [{"pdb": "/wsl/t.pdb", "chain_id": "A", "force": True, "threshold": 10.0}])
        assert "- pdb: /wsl/t.pdb" in y
        assert "force: true" in y
        assert "threshold: 10.0" in y

    def test_force_false_omits_threshold(self):
        y = BoltzBridge._build_yaml(
            [{"id": "A", "sequence": "MK"}],
            [{"cif": "/x.cif", "force": False, "threshold": 8.0}])
        assert "force:" not in y                      # force False → no force/threshold lines
        assert "threshold:" not in y

    def test_template_id_and_list_chain_ids_flow_list(self):
        y = BoltzBridge._build_yaml(
            [{"id": "A", "sequence": "MK"}, {"id": "B", "sequence": "MK"}],
            [{"cif": "/x.cif", "chain_id": ["A", "B"], "template_id": ["A", "B"]}])
        assert "chain_id: [A, B]" in y                # list → YAML flow list
        assert "template_id: [A, B]" in y

    def test_predict_forwards_translated_template_paths(self, monkeypatch):
        # predict() → _translate_template_paths → _build_yaml; the cif path is translate_path'd.
        b = _bridge()
        b._wsl.translate_path = lambda p: "/wsl" + str(p).replace("\\", "/")
        captured = {}
        def fake_build(chains, templates=None):
            captured["templates"] = templates
            return "version: 1\nsequences:\n  - protein:\n      msa: empty\n"
        monkeypatch.setattr(b, "_build_yaml", fake_build)
        # short-circuit the run so we only exercise the build path
        b._wsl.run_command = MagicMock(return_value={"ok": False, "error": "stop"})
        b.predict([{"id": "A", "sequence": "MK"}],
                  templates=[{"cif": "C:/local/t.cif", "chain_id": "A", "force": True,
                              "threshold": 10.0}])
        t = captured["templates"][0]
        assert t["cif"].startswith("/wsl")            # path translated into WSL
        assert t["chain_id"] == "A" and t["force"] is True and t["threshold"] == 10.0

    def test_template_field_indentation_is_4_not_6_spaces(self):
        # THE live-verify regression guard. A template's keys are SIBLINGS of cif/pdb in a YAML
        # sequence item → 4-space indent. The original bug emitted 6 (wrongly mirroring the
        # `protein:` block, whose children nest UNDER a key): that made chain_id a child of pdb,
        # so Boltz got a dict where it expected a path string and produced NO CIF. Pin the exact
        # form. (PyYAML isn't a main-venv dep, so assert the indentation directly.)
        y = BoltzBridge._build_yaml(
            [{"id": "A", "sequence": "MK"}],
            [{"pdb": "/x/4hhb.pdb", "chain_id": "A", "force": True, "threshold": 10.0}])
        lines = y.splitlines()
        assert "  - pdb: /x/4hhb.pdb" in lines            # item key at the sequence-item indent
        assert "    chain_id: A" in lines                 # SIBLING key → exactly 4 spaces
        assert "    force: true" in lines
        assert "    threshold: 10.0" in lines
        # the bug, explicitly forbidden: NO template field may be indented 6 spaces (= nested
        # under pdb). 6-space lines belong ONLY to the protein block (id/sequence/msa).
        for fld in ("chain_id", "template_id", "force", "threshold"):
            assert f"      {fld}:" not in y               # never 6 spaces

    def test_assert_local_only_passes_templates_yaml(self):
        # a templates-carrying YAML (still msa: empty everywhere) must PASS the fail-closed guard —
        # a template is not an MSA.
        y = BoltzBridge._build_yaml(
            [{"id": "A", "sequence": "MK"}],
            [{"cif": "/x.cif", "chain_id": "A", "force": True, "threshold": 10.0}])
        BoltzBridge._assert_local_only(
            "boltz predict in.yaml --accelerator gpu --seed 0", y, False)   # no raise


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
