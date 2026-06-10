"""
scripts/rosetta_clash.py — Rosetta ddG clash-artifact FLAGGING (flag, never delete).

The deployed Rosetta (WSL PyRosetta FastRelax) ddG carries a clash tail: a handful of
mutations — overwhelmingly proline substitutions and bulky insertions — relax into a
steric clash and score implausibly high (e.g. 1PGA K50P ≈ +97 REU, G38P ≈ +95, vs an
experimental destabilising ddG that almost never exceeds ~+7 kcal/mol).  Treating those
as real ddG would poison any correlation / calibration.

This module surfaces a LOW-CONFIDENCE FLAG for that range.  It is metadata only: the
raw Rosetta value is NEVER deleted or altered — the flag rides alongside it so the
exporter / future aggregate can down-weight (not discard) the value, and a human can
still see the raw number.

Threshold is CONSERVATIVE and documented: +10 REU on the destabilising (positive) tail.
Experimental destabilising ddG rarely clears ~+7-8 kcal/mol, so > +10 REU is far more
likely a FastRelax clash artifact than a true measurement.  Tunable via
ROSETTA_CLASH_REU_THRESHOLD; only the positive tail is flagged (clashes score high, not
low — large negatives are not this artifact).
"""
from __future__ import annotations

from typing import Optional

# Conservative destabilising-tail cutoff (REU).  See module docstring.
ROSETTA_CLASH_REU_THRESHOLD = 10.0

CONFIDENCE_OK = "ok"
CONFIDENCE_CLASH = "low_confidence_clash_artifact"


def is_clash_artifact(raw_ddg: Optional[float],
                      threshold: float = ROSETTA_CLASH_REU_THRESHOLD) -> bool:
    """True iff a raw Rosetta ddG is in the clash-artifact tail (> threshold REU).
    None (not_computed) is NOT an artifact — it is simply absent."""
    return raw_ddg is not None and raw_ddg > threshold


def rosetta_confidence(raw_ddg: Optional[float],
                       threshold: float = ROSETTA_CLASH_REU_THRESHOLD) -> Optional[str]:
    """Confidence label for a raw Rosetta ddG, WITHOUT touching the value:
      None  → None            (not_computed)
      >thr  → CONFIDENCE_CLASH (low-confidence clash artifact)
      else  → CONFIDENCE_OK
    """
    if raw_ddg is None:
        return None
    return CONFIDENCE_CLASH if raw_ddg > threshold else CONFIDENCE_OK
