"""
pytest conftest — UTF-8 stdout/stderr for all test runs.

Python 3.x on Windows defaults stdout to cp1252, which cannot represent
→ (U+2192) or Greek letters (α/β).  Reconfiguring here before any test
imports produce output is the canonical fix (avoids UnicodeEncodeError in
test helper print() calls that contain Unicode in assertion names).
"""
import os
import sys

import pytest

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


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
