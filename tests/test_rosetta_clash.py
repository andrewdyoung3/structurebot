"""
Tests for the Rosetta clash-artifact flag (scripts/rosetta_clash.py).

The flag is metadata only — it must NEVER imply the raw value was altered.  These
tests pin the conservative threshold, the not_computed handling, and the boundary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rosetta_clash as rc  # noqa: E402


def test_threshold_is_conservative_default():
    assert rc.ROSETTA_CLASH_REU_THRESHOLD == 10.0


def test_clash_tail_flagged():
    assert rc.is_clash_artifact(97.18) is True          # 1PGA K50P
    assert rc.rosetta_confidence(97.18) == rc.CONFIDENCE_CLASH


def test_normal_value_ok():
    assert rc.is_clash_artifact(4.2) is False
    assert rc.rosetta_confidence(4.2) == rc.CONFIDENCE_OK
    assert rc.rosetta_confidence(-7.0) == rc.CONFIDENCE_OK   # large negative ≠ clash


def test_not_computed_is_not_an_artifact():
    assert rc.is_clash_artifact(None) is False
    assert rc.rosetta_confidence(None) is None              # absent, not "ok"


def test_boundary_is_strict_greater_than():
    assert rc.is_clash_artifact(10.0) is False             # exactly threshold → ok
    assert rc.is_clash_artifact(10.01) is True


def test_custom_threshold():
    assert rc.is_clash_artifact(12.0, threshold=15.0) is False
    assert rc.rosetta_confidence(20.0, threshold=15.0) == rc.CONFIDENCE_CLASH
