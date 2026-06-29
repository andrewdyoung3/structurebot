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
                            lambda chains, templates=None, constraints=None:
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

    def test_disulfide_bond_constraints_block(self):
        # Mode C: a declared SG–SG bond → a top-level `constraints: - bond:` block, DECLARATIVE
        # (no path, no msa, no remote) so the fail-closed LOCAL-ONLY guard is unaffected.
        y = BoltzBridge._build_yaml(
            [{"id": "A", "sequence": "MKVC"}], None,
            [{"atom1": ["A", 2, "SG"], "atom2": ["A", 4, "SG"]}])
        assert "constraints:" in y
        assert "- bond:" in y
        assert "atom1: [A, 2, SG]" in y and "atom2: [A, 4, SG]" in y
        assert y.count("msa: empty") == 1            # still MSA-free
        BoltzBridge._assert_local_only("boltz predict x.yaml --seed 0", y, False)  # no breach

    def test_no_constraints_block_when_absent(self):
        y = BoltzBridge._build_yaml([{"id": "A", "sequence": "MK"}])
        assert "constraints:" not in y               # plain fold unchanged


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
        def fake_build(chains, templates=None, constraints=None):
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


# ── Size-scaled wall-clock budget (_resolve_timeout) ─────────────────────────────────

class TestSizeScaledTimeout:
    def _b(self):
        # default knobs (scale 0.012, floor 1800, cap 21600, no explicit env override)
        b = _bridge()
        b._timeout_explicit = False
        b._timeout_scale = 0.012
        b._timeout_floor = 1800
        b._timeout_cap = 21600
        b._timeout = 1800
        return b

    def test_small_fold_lands_on_floor(self):
        b = self._b()
        # 100 res → 0.012·100² = 120s, below the 1800 floor → floored
        assert b._resolve_timeout([{"id": "A", "sequence": "M" * 100}], None) == 1800

    def test_large_dimer_scales_quadratically(self):
        b = self._b()
        # 550+550 = 1100 res → ceil(0.012·1100²) = 14521s (≈4 h), well above the old flat 1800
        budget = b._resolve_timeout(
            [{"id": "A", "sequence": "M" * 550}, {"id": "B", "sequence": "M" * 550}], None)
        assert budget == 14521

    def test_huge_fold_clamped_to_cap(self):
        b = self._b()
        # 2000 res → 0.012·2000² = 48000s, clamped to the 21600 cap
        assert b._resolve_timeout([{"id": "A", "sequence": "M" * 2000}], None) == 21600

    def test_explicit_caller_timeout_wins(self):
        b = self._b()
        assert b._resolve_timeout([{"id": "A", "sequence": "M" * 1100}], 999) == 999

    def test_explicit_env_override_beats_scaling(self):
        b = self._b()
        b._timeout_explicit = True          # operator set BOLTZ_TIMEOUT — authoritative
        b._timeout = 7200
        assert b._resolve_timeout([{"id": "A", "sequence": "M" * 1100}], None) == 7200

    def test_predict_passes_scaled_budget_to_run_command(self, monkeypatch):
        # the budget computed from sequence length is what actually reaches wsl.run_command.
        b = self._b()
        b._wsl.run_command = MagicMock(return_value={"ok": False, "error": "stop"})
        monkeypatch.setattr(b, "_parse_outputs",
                            lambda out_dir, chains: {"success": False, "error": "x"})
        b.predict([{"id": "A", "sequence": "M" * 550}, {"id": "B", "sequence": "M" * 550}])
        assert b._wsl.run_command.call_args.kwargs["timeout"] == 14521


# ── Informative timeout / failure messages (cause + size + next step) ────────────────

class TestInformativeFailureMessage:
    def _b(self):
        b = _bridge()
        b._timeout_explicit = False
        b._timeout_scale = 0.012
        b._timeout_floor = 1800
        b._timeout_cap = 21600
        b._timeout = 1800
        return b

    @staticmethod
    def _timeout_result(secs):
        # the shape WSLBridge.run_command returns on subprocess.TimeoutExpired
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": "",
                "error": f"WSL2 command timed out after {secs}s: boltz predict ..."}

    def test_cap_timeout_message_names_cap_size_and_override(self):
        b = self._b()
        b._wsl.run_command = MagicMock(return_value=self._timeout_result(21600))
        # 2000 res → scaled 48000s ≥ 21600 cap
        res = b.predict([{"id": "A", "sequence": "M" * 2000}])
        assert res["success"] is False
        err = res["error"]
        assert "6h cap" in err and "2000 residues" in err and "BOLTZ_TIMEOUT" in err

    def test_subcap_timeout_message_reports_computed_value_and_size(self):
        b = self._b()
        b._wsl.run_command = MagicMock(return_value=self._timeout_result(14521))
        # 550+550 = 1100 res → computed budget 14521s (below the cap)
        res = b.predict([{"id": "A", "sequence": "M" * 550}, {"id": "B", "sequence": "M" * 550}])
        assert res["success"] is False
        err = res["error"]
        assert "timed out after 14521s" in err and "1100 residues" in err and "BOLTZ_TIMEOUT" in err

    def test_real_error_is_not_mislabelled_as_timeout(self):
        # Boltz crashed (non-zero exit, a swallowed ValueError) — NOT a wall. The message must
        # surface the real cause and must NOT call it a timeout.
        b = self._b()
        b._wsl.run_command = MagicMock(return_value={
            "ok": False, "returncode": 1, "stdout": "",
            "stderr": "ValueError: Template chain A is not one of the protein chains!\n",
            "error": "Command exited 1: ValueError: Template chain A is not one of the protein chains!"})
        res = b.predict([{"id": "A", "sequence": "M" * 1100}])
        assert res["success"] is False
        err = res["error"]
        assert "failed" in err.lower() and "Template chain A" in err
        assert "timed out" not in err.lower() and "cap" not in err.lower()


def test_boltz_available_swallows_errors(monkeypatch):
    import boltz_bridge
    monkeypatch.setattr(boltz_bridge.BoltzBridge, "is_available",
                        lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    assert boltz_available() is False


# ── Pre-flight GPU guards: VRAM/size + concurrency (the fail-loud-before-thrash gate) ────
#
# Measured on the RTX 5070 Ti Laptop (12 GB): peak VRAM ≈ 3447 + 0.010336·N² MiB. Anchors
# (N→peak MiB): 80→3411, 300→4554, 500→5942, 700→8526; 900→11693 (at the ceiling — thrashed);
# 1084→~15 GB predicted (thrashed). The guard reads ACTUAL free VRAM and refuses BEFORE launch.

class TestVramCurve:
    def test_curve_matches_swept_anchors(self):
        b = _bridge()
        # the fitted curve reproduces the measured fitting points within the fit residual (~200 MiB)
        for n, measured in [(80, 3411), (300, 4554), (500, 5942), (700, 8526)]:
            assert abs(b.estimated_vram_mib(n) - measured) <= 250, (n, b.estimated_vram_mib(n))

    def test_curve_is_quadratic_and_monotonic(self):
        b = _bridge()
        assert b.estimated_vram_mib(1084) > b.estimated_vram_mib(900) > b.estimated_vram_mib(700)
        # 1084 res predicts well past a 12 GB GPU (≈15 GB) — the doomed fold
        assert b.estimated_vram_mib(1084) > 14000

    def test_total_residues_sums_all_chains(self):
        b = _bridge()
        assert b._total_residues([{"sequence": "M" * 503}, {"sequence": "M" * 581}]) == 1084


class TestSizeGuard:
    def _b(self, *, free, running=()):
        b = _bridge()
        b._gpu_guard = True
        b._free_vram_mib = lambda: free            # inject GPU state (no real nvidia-smi)
        b._running_fold_pids = lambda: list(running)
        return b

    def test_refuses_1084_on_12gb_before_any_subprocess(self):
        # the doomed fold: 503+581 needs ~15 GB; 11.4 GB free → refuse, NO fold subprocess
        b = self._b(free=11400)
        res = b.predict([{"id": "A", "sequence": "M" * 503}, {"id": "B", "sequence": "M" * 581}])
        assert res["success"] is False
        err = res["error"]
        assert "1084-residue" in err and "GB VRAM" in err and "too large to fold safely" in err
        b._wsl.run_command.assert_not_called()      # THE gate: fail loud BEFORE launch (no thrash)

    def test_approves_monomer_on_free_gpu(self):
        # 80-res monomer needs ~3.4 GB; 11.4 GB free → cleared, the fold subprocess IS launched
        b = self._b(free=11400)
        b._wsl.run_command = MagicMock(return_value={"ok": False, "error": "stop after launch"})
        res = b.predict([{"id": "A", "sequence": "M" * 80}])
        assert res["success"] is False             # our mock stops it, but the point is it LAUNCHED
        assert b._wsl.run_command.called
        assert "too large" not in (res["error"] or "")

    def test_margin_refuses_a_fold_that_would_fit_bare_but_not_with_buffer(self):
        # 700 res needs ~8.5 GB. With only 9 GB free, 8.5 + 2 GB margin > 9 → refuse (the noisy-
        # free-VRAM lesson: a fold that fits at 11 GB free thrashes at 9 GB).
        b = self._b(free=9000)
        res = b.predict([{"id": "A", "sequence": "M" * 700}])
        assert res["success"] is False and "too large to fold safely" in res["error"]
        b._wsl.run_command.assert_not_called()

    def test_unknown_free_vram_does_not_block(self):
        # can't probe VRAM → size guard SKIPPED (don't block on a probe failure)
        b = self._b(free=None)
        b._wsl.run_command = MagicMock(return_value={"ok": False, "error": "stop after launch"})
        res = b.predict([{"id": "A", "sequence": "M" * 1084}])
        assert b._wsl.run_command.called           # launched despite size (couldn't assess)

    def test_guard_disabled_via_config_allows_oversize(self):
        b = self._b(free=11400)
        b._gpu_guard = False                       # BOLTZ_GPU_GUARD=off escape hatch
        b._wsl.run_command = MagicMock(return_value={"ok": False, "error": "stop after launch"})
        res = b.predict([{"id": "A", "sequence": "M" * 1084}])
        assert b._wsl.run_command.called           # guard off → oversize fold launches


class TestConcurrencyGuard:
    def _b(self, *, free, running):
        b = _bridge()
        b._gpu_guard = True
        b._free_vram_mib = lambda: free
        b._running_fold_pids = lambda: list(running)
        return b

    def test_refuses_second_fold_while_one_runs_with_specific_message(self):
        # another fold holds the GPU → refuse with the BUSY message (distinct from the size one),
        # NO second subprocess — even for a small fold that would otherwise fit
        b = self._b(free=400, running=["12345"])
        res = b.predict([{"id": "A", "sequence": "M" * 80}])
        assert res["success"] is False
        err = res["error"]
        assert "Another Boltz fold is already using the GPU" in err
        assert "too large" not in err              # specific 'busy', not misattributed to size
        b._wsl.run_command.assert_not_called()

    def test_concurrency_checked_before_size(self):
        # a busy GPU leaves low free VRAM; the message must say BUSY, not "too big"
        b = self._b(free=300, running=["999"])
        res = b.predict([{"id": "A", "sequence": "M" * 1084}])
        assert "Another Boltz fold" in res["error"] and "too large" not in res["error"]


class TestGpuFaultMessage:
    def test_detects_illegal_access_and_oom(self):
        b = _bridge()
        assert b._is_gpu_memory_fault("torch.AcceleratorError: CUDA error: an illegal memory access")
        assert b._is_gpu_memory_fault("RuntimeError: CUDA out of memory. Tried to allocate ...")
        assert not b._is_gpu_memory_fault("ValueError: Template chain A is not one of the chains")

    def test_run_failure_translates_cuda_fault(self):
        # a CUDA illegal-access crash → the informative cause+size message, NOT the raw torch line
        b = _bridge()
        b._free_vram_mib = lambda: 510
        r = {"ok": False, "returncode": 1, "stdout": "",
             "stderr": "torch.AcceleratorError: CUDA error: an illegal memory access was encountered\n",
             "error": "Command exited 1"}
        msg = b._run_failure_message(r, [{"id": "A", "sequence": "M" * 1084}], budget=9000)
        assert "GPU memory fault on the 1084-residue fold" in msg
        assert "0.5 GB free" in msg and "another fold" in msg.lower()
        assert "AcceleratorError" not in msg        # the raw torch line is NOT leaked

    def test_predict_surfaces_gpu_fault_message_on_crash(self):
        # end-to-end through predict: a CUDA fault from the (guard-cleared) fold is translated
        b = _bridge()
        b._gpu_guard = False                        # let it reach the fold to simulate the crash
        b._free_vram_mib = lambda: 480
        b._wsl.run_command = MagicMock(return_value={
            "ok": False, "returncode": 1, "stdout": "",
            "stderr": "torch.AcceleratorError: CUDA error: an illegal memory access was encountered",
            "error": "Command exited 1"})
        res = b.predict([{"id": "A", "sequence": "M" * 1084}])
        assert res["success"] is False
        assert "GPU memory fault" in res["error"] and "1084-residue" in res["error"]


class TestVramProbeParsing:
    def test_free_vram_parses_nvidia_smi(self):
        b = _bridge()
        b._wsl.run_command = MagicMock(return_value={"ok": True, "stdout": "11385\n", "stderr": ""})
        assert b._free_vram_mib() == 11385

    def test_free_vram_none_on_probe_failure(self):
        b = _bridge()
        b._wsl.run_command = MagicMock(return_value={"ok": False, "stdout": "", "stderr": "no smi"})
        assert b._free_vram_mib() is None

    def test_free_vram_none_on_exception(self):
        b = _bridge()
        b._wsl.run_command = MagicMock(side_effect=RuntimeError("wsl down"))
        assert b._free_vram_mib() is None           # degrades gracefully (never raises)

    def test_running_fold_pids_parses_pgrep(self):
        b = _bridge()
        b._wsl.run_command = MagicMock(return_value={"ok": True, "stdout": "321\n494\n", "stderr": ""})
        assert b._running_fold_pids() == ["321", "494"]

    def test_running_fold_pids_empty_when_none(self):
        b = _bridge()
        b._wsl.run_command = MagicMock(return_value={"ok": True, "stdout": "", "stderr": ""})
        assert b._running_fold_pids() == []
