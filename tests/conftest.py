"""
pytest conftest — UTF-8 stdout/stderr for all test runs.

Python 3.x on Windows defaults stdout to cp1252, which cannot represent
→ (U+2192) or Greek letters (α/β).  Reconfiguring here before any test
imports produce output is the canonical fix (avoids UnicodeEncodeError in
test helper print() calls that contain Unicode in assertion names).
"""
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
