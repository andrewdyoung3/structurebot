"""
pytest conftest — UTF-8 stdout/stderr for all test runs.

Python 3.x on Windows defaults stdout to cp1252, which cannot represent
→ (U+2192) or Greek letters (α/β).  Reconfiguring here before any test
imports produce output is the canonical fix (avoids UnicodeEncodeError in
test helper print() calls that contain Unicode in assertion names).
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ── Real-GPU serialization (Boltz-stage hardening; §9 follow-up #2) ──────────────────
# The DEFAULT suite is mock-only — real GPU/WSL inference lives behind the gated smokes
# (STRUCTUREBOT_RUN_LIVE_DEPS / STRUCTUREBOT_RUN_LIVE_COLABFOLD). When those DO run, a real
# fold must never contend with another real fold for the 12 GB VRAM — the contention class
# that flaked the suite while the multimer probe's GPU folds were still running. A test that
# does real GPU/WSL work marks itself `@pytest.mark.gpu`; this autouse fixture then holds a
# CROSS-PROCESS file lock for that test's duration, so gated smokes + live-verify scripts +
# any concurrent invocation serialize GPU access. Unmarked (mock-only) tests are untouched.
_GPU_LOCK_PATH = str(Path(tempfile.gettempdir()) / "structurebot_gpu.lock")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: real GPU/WSL inference — serialized across processes via a file lock "
        "(only meaningful under the gated live-deps env vars; default suite is mock-only)")


@pytest.fixture(autouse=True)
def _gpu_serialize(request):
    """Hold the cross-process GPU lock for the duration of a `@pytest.mark.gpu` test."""
    if request.node.get_closest_marker("gpu") is None:
        yield
        return
    import filelock
    with filelock.FileLock(_GPU_LOCK_PATH).acquire(timeout=3600):
        yield


@pytest.fixture(autouse=True)
def _disable_dynamut2_api(monkeypatch):
    """DynaMut2 is a REMOTE API — no test may hit it implicitly (slow + flaky +
    perturbs scores).  Default it OFF for the whole suite; the DynaMut2 voter tests
    enable it explicitly while MOCKING the bridge.  A deep-tier scan in any other
    test therefore leaves the dynamics axis not_computed (graceful), unperturbed."""
    monkeypatch.setenv("DYNAMUT2_ENABLE", "false")
    try:
        import config
        monkeypatch.setattr(config, "DYNAMUT2_ENABLE", "false", raising=False)
    except Exception:
        pass
