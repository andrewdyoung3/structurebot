"""
tests/test_dep_probe.py
-----------------------
The shared capability-probe helper (dep_probe) — the generalization of the
cavity-class fix. Hermetic: subprocess / WSL are mocked. Locks the contract the
four bridges (ThermoMPNN/proteinmpnn/RaSP/rfdiffusion) depend on:
  - import-chain OK  → True
  - import fails (returncode != 0) → False           (the capability catch)
  - probe-infra failure (spawn error / timeout)      → False AND NOT cached
  - definitive verdict cached (no re-spawn)
"""
from __future__ import annotations

import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import dep_probe


def _ok():
    return CompletedProcess(args=[], returncode=0, stdout="DEP_PROBE_IMPORT_OK\n", stderr="")


def _fail():
    return CompletedProcess(args=[], returncode=1, stdout="", stderr="ModuleNotFoundError")


class TestLocalImportProbe:
    def setup_method(self):
        dep_probe.reset_probe_cache()

    def test_true_when_imports_ok(self):
        with patch.object(dep_probe.subprocess, "run", return_value=_ok()):
            assert dep_probe.local_import_probe("py", ["import torch"]) is True

    def test_false_when_import_fails(self):
        with patch.object(dep_probe.subprocess, "run", return_value=_fail()):
            assert dep_probe.local_import_probe("py", ["import torch"]) is False

    def test_infra_failure_false_and_not_cached(self):
        with patch.object(dep_probe.subprocess, "run", side_effect=TimeoutError("boom")):
            assert dep_probe.local_import_probe("py", ["import x"],
                                                cache_key=("k",)) is False
        assert ("k",) not in dep_probe._PROBE_CACHE   # transient → re-probe later

    def test_definitive_result_cached_no_respawn(self):
        with patch.object(dep_probe.subprocess, "run", return_value=_ok()) as m:
            assert dep_probe.local_import_probe("py", ["import x"], cache_key=("k",)) is True
            assert dep_probe.local_import_probe("py", ["import x"], cache_key=("k",)) is True
            assert m.call_count == 1

    def test_sentinel_required_not_just_returncode(self):
        # returncode 0 but missing sentinel (e.g. truncated/odd output) → False
        odd = CompletedProcess(args=[], returncode=0, stdout="(no sentinel)", stderr="")
        with patch.object(dep_probe.subprocess, "run", return_value=odd):
            assert dep_probe.local_import_probe("py", ["import x"]) is False


class TestWslImportProbe:
    def setup_method(self):
        dep_probe.reset_probe_cache()

    def _wsl(self, ok=True, stdout="DEP_PROBE_IMPORT_OK", avail=True):
        w = MagicMock()
        w.is_available.return_value = avail
        w.run_command.return_value = {"ok": ok, "stdout": stdout, "stderr": ""}
        return w

    def test_true_when_wsl_imports_ok(self):
        assert dep_probe.wsl_import_probe(self._wsl(), "/wsl/py", ["import torch"]) is True

    def test_false_when_wsl_import_fails(self):
        w = self._wsl(ok=False, stdout="")
        assert dep_probe.wsl_import_probe(w, "/wsl/py", ["import torch"]) is False

    def test_wsl_down_false_not_cached(self):
        w = self._wsl(avail=False)
        assert dep_probe.wsl_import_probe(w, "/wsl/py", ["import x"],
                                          cache_key=("wk",)) is False
        assert ("wk",) not in dep_probe._PROBE_CACHE


class TestBridgeGating:
    """Each bridge gates tier-1 (presence) THEN tier-2 (dep_probe capability)."""

    def setup_method(self):
        dep_probe.reset_probe_cache()

    def test_proteinmpnn_tier1_blocks_probe(self):
        import proteinmpnn_bridge as pm
        b = pm.ProteinMPNNBridge(); b._available = False
        with patch.object(dep_probe, "local_import_probe",
                          side_effect=AssertionError("tier-1 must block the probe")):
            assert b.is_available() is False

    def test_proteinmpnn_tier2_capability(self):
        import proteinmpnn_bridge as pm
        b = pm.ProteinMPNNBridge()
        b._available, b._dir, b._backend = True, Path("x"), "proteinmpnn"
        with patch.object(dep_probe, "local_import_probe", return_value=False):
            assert b.is_available() is False
        with patch.object(dep_probe, "local_import_probe", return_value=True):
            assert b.is_available() is True

    def test_rasp_tier2_capability(self):
        import rasp_bridge
        rasp_bridge._AVAIL_CACHE.clear()
        b = rasp_bridge.RaSPBridge(); b._enable = "auto"
        b._wsl = MagicMock(); b._wsl.is_available.return_value = True
        b._wsl.run_command.return_value = {"ok": True, "stdout": "RASP_DIR_OK"}
        with patch.object(dep_probe, "wsl_import_probe", return_value=False):
            assert b.is_available() is False
        rasp_bridge._AVAIL_CACHE.clear()
        with patch.object(dep_probe, "wsl_import_probe", return_value=True):
            assert b.is_available() is True

    def test_rfdiffusion_tier1_blocks_probe(self):
        import rfdiffusion_bridge as rf
        b = rf.RFdiffusionBridge(); b._available = False
        with patch.object(dep_probe, "wsl_import_probe",
                          side_effect=AssertionError("tier-1 must block the probe")):
            assert b.is_available() is False

    def test_rfdiffusion_tier2_capability(self):
        import rfdiffusion_bridge as rf
        b = rf.RFdiffusionBridge(); b._available = True; b._wsl = MagicMock()
        with patch.object(dep_probe, "wsl_import_probe", return_value=False):
            assert b.is_available() is False
        with patch.object(dep_probe, "wsl_import_probe", return_value=True):
            assert b.is_available() is True
