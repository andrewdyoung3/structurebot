"""
netnglyc_bridge.py
------------------
Integrates with the NetNGlyc 1.0 REST API (DTU Health Tech) to predict
OST (oligosaccharyltransferase) recognition of N-linked glycosylation sequons.

NetNGlyc predicts, for each N-X-S/T sequon in a protein sequence, whether
it will be recognised and glycosylated by OST in the endoplasmic reticulum.
This complements structural scores (projection, SASA) with biochemical
validation: a geometrically ideal site that scores poorly on OST recognition
is unlikely to be glycosylated in vivo.

Public API
----------
predict_glycosylation(sequence, name, timeout)
    Submit a sequence, parse per-sequon OST scores.

score_engineered_sequon(sequon_position, engineered_sequence, wildtype_sequence)
    Validate one engineered sequon vs the wildtype at that position.

integrate_with_glycan_candidates(candidates, engineered_sequence)
    Bulk-annotate candidate list with OST recognition scores.

Error contract
--------------
All public methods return a dict — never raise.  On API failure the dict
has {"success": False, "error": "..."}.  Callers degrade gracefully.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional


# ── OST score thresholds ──────────────────────────────────────────────────────

_SCORE_HIGH:   float = 0.7    # "high" — glycosylated with high probability
_SCORE_MEDIUM: float = 0.5    # "medium" — ambiguous / threshold-level
# below 0.5 → "low"

# ── Confidence-level numeric mapping ──────────────────────────────────────────

_CONF_MAP: Dict[str, float] = {"high": 0.9, "moderate": 0.6, "low": 0.3}


def _harmonic_mean(a: float, b: float) -> float:
    """Two-term harmonic mean.  Returns 0.0 if either term is 0."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _classify_ost_score(score: float) -> str:
    if score >= _SCORE_HIGH:
        return "high"
    if score >= _SCORE_MEDIUM:
        return "medium"
    return "low"


def _strip_html(text: str) -> str:
    """Remove HTML tags from *text*.  Very lightweight — no html.parser needed."""
    return re.sub(r"<[^>]+>", "", text)


# ── Parser ────────────────────────────────────────────────────────────────────

# Matches lines like:
#   query    5   N   NIT   0.5766   YES   9/9
#   MySeq   35   N   NAS   0.7200    +    +++
#   name1   10   N   NGT   0.3100   NO    ---
#
# Groups: (name)(position)(aa=N)(sequon3)(score)(pred)
_LINE_RE = re.compile(
    r"^\s*(\S+)\s+"               # sequence name
    r"(\d+)\s+"                   # position (integer)
    r"N\s+"                       # residue letter (N)
    r"([A-Z]{3})\s+"              # 3-letter sequon
    r"([\d.]+)"                   # score (float)
    r".*?(\+|-|YES|NO|yes|no)",   # prediction
    re.IGNORECASE,
)

# Simpler fallback: just find integer + 3-mer starting with N + float
_FALLBACK_RE = re.compile(
    r"\b(\d+)\b"                  # position
    r".{0,20}?"                   # anything (non-greedy)
    r"\b(N[A-Z]{2})\b"            # N-sequon
    r".{0,20}?"
    r"\b(0\.\d+)\b",              # score 0.xxx
)


def _parse_netnglyc_output(text: str, sequence_name: str = "query") -> List[Dict[str, Any]]:
    """
    Parse NetNGlyc output text (HTML-stripped or plain text).

    Tries the primary regex first; falls back to the simpler pattern.
    Returns list of dicts with keys: position, sequon, score, prediction.
    """
    clean  = _strip_html(text)
    rows:  List[Dict[str, Any]] = []
    seen_pos: set = set()

    for line in clean.splitlines():
        m = _LINE_RE.match(line)
        if m:
            pos  = int(m.group(2))
            seq3 = m.group(3).upper()
            if not seq3.startswith("N"):
                continue
            score = float(m.group(4))
            pred_raw = m.group(5).upper()
            prediction = "Glycosylated" if pred_raw in ("+", "YES") else "Not glycosylated"

            if pos in seen_pos:
                continue
            seen_pos.add(pos)
            rows.append({
                "position":   pos,
                "sequon":     seq3,
                "score":      score,
                "prediction": prediction,
            })

    if not rows:
        # Fallback: looser pattern
        for m in _FALLBACK_RE.finditer(clean):
            pos   = int(m.group(1))
            seq3  = m.group(2).upper()
            score = float(m.group(3))
            if pos in seen_pos:
                continue
            seen_pos.add(pos)
            rows.append({
                "position":   pos,
                "sequon":     seq3,
                "score":      score,
                "prediction": "Glycosylated" if score >= _SCORE_MEDIUM else "Not glycosylated",
            })

    rows.sort(key=lambda r: r["position"])
    return rows


# ── NetNGlycBridge class ──────────────────────────────────────────────────────

class NetNGlycBridge:
    """
    Client for the NetNGlyc 1.0 OST recognition prediction service.

    All public methods are stateless and safe to call concurrently.
    Construct once and reuse.
    """

    # Default API URL — overridden by config.NETNGLYC_API_URL if set
    _DEFAULT_URL: str = (
        "https://services.healthtech.dtu.dk/service.php?NetNGlyc-1.0"
    )
    _HEADERS: Dict[str, str] = {
        "User-Agent": "StructureBot/1.0 (research; structurebot@example.com)",
        "Accept":     "text/html,application/xhtml+xml,text/plain",
    }

    def __init__(self, api_url: Optional[str] = None, timeout: int = 30):
        try:
            import config as _cfg
            self._url     = api_url or getattr(_cfg, "NETNGLYC_API_URL", self._DEFAULT_URL)
            self._timeout = timeout or getattr(_cfg, "NETNGLYC_TIMEOUT", 30)
            self._enabled = getattr(_cfg, "NETNGLYC_ENABLED", True)
        except ImportError:
            self._url     = api_url or self._DEFAULT_URL
            self._timeout = timeout
            self._enabled = True

    # ── predict_glycosylation ──────────────────────────────────────────────────

    def predict_glycosylation(
        self,
        sequence: str,
        name:     str = "query",
        timeout:  Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Submit *sequence* to NetNGlyc 1.0 and return per-sequon OST scores.

        Parameters
        ----------
        sequence : single-letter amino-acid sequence
        name     : identifier for the server (truncated to 40 chars)
        timeout  : HTTP timeout in seconds; falls back to instance default

        Returns
        -------
        {
          "success"            : True,
          "sequence"           : str,
          "n_residues"         : int,
          "api_version"        : "NetNGlyc-1.0",
          "predictions"        : [{position, residue, sequon, ost_score,
                                   ost_category, prediction, confidence}, ...],
          "n_sequons_found"    : int,
          "n_sequons_high_score": int,    # ost_score > 0.7
          "error"              : None,
        }

        On failure:
        {
          "success": False,
          "error"  : "reason…",
        }
        """
        if not sequence:
            return {"success": False, "error": "Empty sequence provided to NetNGlyc"}

        sequence = sequence.upper().strip()
        name     = (name or "query")[:40].replace(" ", "_")
        t0       = time.perf_counter()

        fasta = f">{name}\n{sequence}\n"
        data  = {"SEQENCE": fasta}     # "SEQENCE" — server-side spelling

        try:
            import requests
            resp = requests.post(
                self._url,
                data    = data,
                headers = self._HEADERS,
                timeout = timeout or self._timeout,
            )
            resp.raise_for_status()
            raw_text = resp.text
        except Exception as exc:
            return self._error_result(exc, timeout or self._timeout)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_rows = _parse_netnglyc_output(raw_text, name)
        predictions = self._enrich_rows(raw_rows, sequence)

        n_high = sum(1 for p in predictions if p["ost_score"] >= _SCORE_HIGH)

        return {
            "success":              True,
            "sequence":             sequence,
            "n_residues":           len(sequence),
            "api_version":          "NetNGlyc-1.0",
            "predictions":          predictions,
            "n_sequons_found":      len(predictions),
            "n_sequons_high_score": n_high,
            "elapsed_ms":           round(elapsed_ms, 1),
            "error":                None,
        }

    # ── score_engineered_sequon ────────────────────────────────────────────────

    def score_engineered_sequon(
        self,
        sequon_position:     int,
        engineered_sequence: str,
        wildtype_sequence:   str,
        timeout:             Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict OST recognition for a single engineered sequon.

        Calls predict_glycosylation() on both *engineered_sequence* and
        *wildtype_sequence* (if the WT position was a sequon) and returns a
        comparison dict.

        Parameters
        ----------
        sequon_position    : 1-indexed Asn position in *engineered_sequence*
        engineered_sequence: full sequence with the sequon mutation applied
        wildtype_sequence  : unmodified sequence for reference comparison

        Returns
        -------
        {
          "position"         : int,
          "sequon"           : str,       # engineered 3-mer at sequon_position
          "ost_score"        : float,
          "ost_category"     : str,
          "prediction"       : str,
          "confidence"       : float,
          "vs_wildtype_sequon": {
            "wt_sequon"    : str,
            "wt_ost_score" : float | None,
            "delta"        : float | None,
          },
          "notes"            : str,
          "success"          : True,
          "error"            : None,
        }
        """
        _to = timeout or self._timeout

        # Engineered-sequence prediction
        eng_result = self.predict_glycosylation(
            engineered_sequence,
            name    = "engineered",
            timeout = _to,
        )
        if not eng_result["success"]:
            return {**eng_result, "position": sequon_position}

        # Look up the score at sequon_position
        eng_pred = _find_prediction_at(eng_result["predictions"], sequon_position)

        # Extract engineered sequon from the sequence directly as fallback
        i = sequon_position - 1   # 0-indexed
        seq = engineered_sequence.upper()
        raw_sequon = seq[i : i + 3] if i + 3 <= len(seq) else "???"

        if eng_pred:
            ost_score  = eng_pred["ost_score"]
            ost_cat    = eng_pred["ost_category"]
            prediction = eng_pred["prediction"]
            confidence = eng_pred["confidence"]
            seq3       = eng_pred["sequon"]
        else:
            # NetNGlyc did not report this position (score < threshold or not a sequon)
            ost_score  = 0.0
            ost_cat    = "low"
            prediction = "Not glycosylated"
            confidence = 0.0
            seq3       = raw_sequon

        # Wildtype comparison
        wt_seq = (wildtype_sequence or "").upper()
        wt_idx = i
        wt3    = wt_seq[wt_idx : wt_idx + 3] if wt_idx + 3 <= len(wt_seq) else "???"
        wt_is_sequon = (
            len(wt3) == 3
            and wt3[0] == "N"
            and wt3[1] != "P"
            and wt3[2] in "ST"
        )

        wt_ost_score: Optional[float] = None
        if wt_is_sequon:
            wt_result = self.predict_glycosylation(
                wildtype_sequence,
                name    = "wildtype",
                timeout = _to,
            )
            if wt_result["success"]:
                wt_pred = _find_prediction_at(wt_result["predictions"], sequon_position)
                if wt_pred:
                    wt_ost_score = wt_pred["ost_score"]

        delta: Optional[float] = None
        if wt_ost_score is not None:
            delta = round(ost_score - wt_ost_score, 4)
        elif wt_is_sequon:
            delta = None   # WT had a sequon but score wasn't found
        else:
            delta = ost_score   # WT had no sequon → full gain

        notes = self._sequon_notes(
            seq3, ost_score, ost_cat, wt3, wt_ost_score, wt_is_sequon, delta
        )

        return {
            "success":          True,
            "position":         sequon_position,
            "sequon":           seq3,
            "ost_score":        ost_score,
            "ost_category":     ost_cat,
            "prediction":       prediction,
            "confidence":       confidence,
            "vs_wildtype_sequon": {
                "wt_sequon":    wt3,
                "wt_ost_score": wt_ost_score,
                "delta":        delta,
            },
            "notes": notes,
            "error": None,
        }

    # ── integrate_with_glycan_candidates ──────────────────────────────────────

    def integrate_with_glycan_candidates(
        self,
        candidates:          List[Dict[str, Any]],
        engineered_sequence: str,
        timeout:             Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Bulk-annotate a list of glycan engineering candidates with OST scores.

        Submits *engineered_sequence* (the sequence with all proposed mutations
        applied, creating NxS/T sequons at each candidate position) to NetNGlyc
        once, then looks up the score for each candidate position.

        Parameters
        ----------
        candidates           : list of dicts from glycan_bridge methods;
                               must each have a "position" key
        engineered_sequence  : sequence with NxS/T sequons at candidate positions

        Returns
        -------
        Annotated copy of *candidates* — original dicts are NOT mutated.
        On API failure, returns candidates unchanged (no ost_score field added).
        """
        if not candidates or not engineered_sequence:
            return candidates

        result = self.predict_glycosylation(
            engineered_sequence,
            name    = "candidates",
            timeout = timeout or self._timeout,
        )

        if not result["success"]:
            # Graceful degradation — return candidates unchanged
            return [{**c, "ost_score": None, "ost_category": None,
                     "ost_error": result.get("error")}
                    for c in candidates]

        predictions = result["predictions"]
        annotated: List[Dict[str, Any]] = []

        for cand in candidates:
            pos  = cand.get("position")
            pred = _find_prediction_at(predictions, pos) if pos is not None else None

            if pred:
                ost_score  = pred["ost_score"]
                ost_cat    = pred["ost_category"]
                ost_pred   = pred["prediction"]
                ost_conf   = pred["confidence"]
            else:
                ost_score  = 0.0
                ost_cat    = "low"
                ost_pred   = "Not glycosylated"
                ost_conf   = 0.0

            # Combine structural confidence (from composite_score or confidence label)
            struct_conf_str = cand.get("confidence", "moderate")
            struct_conf     = _CONF_MAP.get(struct_conf_str, 0.5)
            combined        = round(_harmonic_mean(struct_conf, max(ost_conf, 1e-6)), 4)

            annotated.append({
                **cand,
                "ost_score":          ost_score,
                "ost_category":       ost_cat,
                "ost_prediction":     ost_pred,
                "ost_confidence":     ost_conf,
                "combined_confidence": combined,
                "ost_error":          None,
            })

        return annotated

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _enrich_rows(
        rows:     List[Dict[str, Any]],
        sequence: str,
    ) -> List[Dict[str, Any]]:
        """
        Add ost_category, confidence, and residue fields to parsed rows.
        Also validates that each position is a real N in *sequence*.
        """
        enriched: List[Dict[str, Any]] = []
        for row in rows:
            pos  = row["position"]
            score = row["score"]
            # Validate position against the actual sequence
            if 1 <= pos <= len(sequence) and sequence[pos - 1] != "N":
                continue   # parser artefact — not a real N position

            ost_cat    = _classify_ost_score(score)
            confidence = round(score, 4)

            enriched.append({
                "position":    pos,
                "residue":     "N",
                "sequon":      row["sequon"],
                "ost_score":   round(score, 4),
                "ost_category": ost_cat,
                "prediction":  row["prediction"],
                "confidence":  confidence,
            })
        return enriched

    @staticmethod
    def _error_result(
        exc:     Exception,
        timeout: int,
    ) -> Dict[str, Any]:
        """Map an exception to a structured error dict."""
        exc_str = str(exc)
        try:
            import requests
            if isinstance(exc, requests.exceptions.Timeout):
                return {
                    "success": False,
                    "error":   f"NetNGlyc API timeout after {timeout}s",
                }
            if isinstance(exc, requests.exceptions.ConnectionError):
                return {
                    "success": False,
                    "error":   "NetNGlyc API unreachable — check network or server status",
                }
        except ImportError:
            pass
        return {
            "success": False,
            "error":   f"NetNGlyc API error: {exc_str}",
        }

    @staticmethod
    def _sequon_notes(
        eng_seq:      str,
        ost_score:    float,
        ost_cat:      str,
        wt_seq:       str,
        wt_score:     Optional[float],
        wt_is_sequon: bool,
        delta:        Optional[float],
    ) -> str:
        parts: List[str] = []

        if ost_cat == "high":
            parts.append(
                f"Engineered {eng_seq} sequon scores {ost_score:.3f} (high) "
                f"— will be glycosylated with high probability in ER"
            )
        elif ost_cat == "medium":
            parts.append(
                f"Engineered {eng_seq} sequon scores {ost_score:.3f} (medium) "
                f"— borderline OST recognition; experimental validation required"
            )
        else:
            parts.append(
                f"Engineered {eng_seq} sequon scores {ost_score:.3f} (low) "
                f"— poor OST recognition; alternative sequon design recommended"
            )

        if wt_is_sequon and wt_score is not None:
            parts.append(
                f"WT sequon {wt_seq} already scored {wt_score:.3f}; "
                f"delta = {delta:+.3f}"
            )
        elif wt_is_sequon:
            parts.append(f"WT {wt_seq} is a sequon (score not found in WT prediction)")
        else:
            parts.append(
                f"WT position ({wt_seq}) was not a sequon — full gain from engineering"
            )

        return "; ".join(parts)


# ── Convenience function ──────────────────────────────────────────────────────

def _find_prediction_at(
    predictions: List[Dict[str, Any]],
    position:    Optional[int],
) -> Optional[Dict[str, Any]]:
    """Return the prediction dict for a given 1-indexed position, or None."""
    if position is None:
        return None
    for p in predictions:
        if p.get("position") == position:
            return p
    return None
