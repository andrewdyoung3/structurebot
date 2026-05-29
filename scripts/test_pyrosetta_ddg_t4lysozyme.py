#!/usr/bin/env python3
"""
scripts/test_pyrosetta_ddg_t4lysozyme.py
----------------------------------------
Independent experimental cross-check of the PyRosetta / WSL2 local ddG path
against T4 lysozyme (PDB 2LZM) — the most extensively characterised ddG
benchmark protein in the literature (Matthews lab hydrophobic-core mutations).

This complements scripts/test_pyrosetta_ddg_controls.py (1HSG) with a
DIFFERENT fold, different residues, and REAL published experimental ddG
values to compare predicted magnitudes against. Same entry point
(RosettaBridge._run_rosetta_local), one mutation at a time.

Per mutation prints: mutation, predicted ddG (3 dp), experimental ddG,
signed error (pred - exp), ddg_source, confidence.

At the end: sign accuracy (|ddG|<0.5 treated as neutral, not penalised),
MAE, RMSE, Pearson r (computed manually), per-mutation bucket comparison,
an explicit L99A / large-cavity UNDERESTIMATION check (the known single-
trajectory FastRelax washout failure mode), and a verdict comparing this
run's MAE / sign accuracy / r to Benchmark Run 1 in
scripts/rosetta_validation_notes.md.

Usage:
    python scripts/test_pyrosetta_ddg_t4lysozyme.py

Requires ROSETTA_BACKEND=local (PyRosetta in WSL2). Forced on regardless of
.env.local. Read-only: changes no project modules.

Wild-type residue identities are verified against the live 2LZM chain A
sequence at runtime (all ten confirmed correct against canonical T4 lysozyme
at authoring time: L99,L133,L121,F153,V149,L118,V87,S117,T152,N116).

Experimental ddG values (kcal/mol, +ve = destabilising) are well-established
literature values from Matthews and coworkers; they are approximate
consensus figures used here for magnitude comparison, not exact citations.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# ── Project path setup ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

config.load_env_file()
os.environ["ROSETTA_BACKEND"] = "local"   # force the PyRosetta/WSL2 path

from rosetta_bridge import RosettaBridge, _mutation_key  # noqa: E402

_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# ── Control panel: (position, from_aa, to_aa, exp_ddg, group, note) ──────────
# exp_ddg: published experimental ddG (kcal/mol), +ve = destabilising.
CONTROLS = [
    # Large cavity-creating (buried Leu/Ile/Val/Phe -> Ala)
    (99,  "L", "A", 5.0, "large_cavity", "canonical large-cavity mutation"),
    (133, "L", "A", 2.7, "large_cavity", "buried core Leu -> Ala"),
    (121, "L", "A", 2.7, "large_cavity", "buried core Leu -> Ala"),
    (153, "F", "A", 3.0, "large_cavity", "buried Phe -> Ala, large cavity"),
    (149, "V", "A", 2.0, "large_cavity", "buried Val -> Ala"),
    # Moderate
    (118, "L", "A", 2.5, "moderate",     "Leu -> Ala, moderate cavity"),
    (87,  "V", "M", 1.5, "moderate",     "cavity-FILLING (opposite direction)"),
    (117, "S", "V", 0.9, "moderate",     "Ser -> Val"),
    # Near-neutral / surface
    (152, "T", "S", 0.5, "surface",      "surface conservative Thr -> Ser"),
    (116, "N", "D", 0.1, "surface",      "surface Asn -> Asp, small effect"),
]

_LARGE_CAVITY = {"L99A", "L133A", "L121A", "F153A", "V149A"}
_NEUTRAL_BAND = 0.5   # |ddG| below this is treated as "neutral"

# Benchmark Run 1 reference (scripts/rosetta_validation_notes.md, 2026-05-29)
_BR1 = {"sign": 0.60, "mae": 3.823, "rmse": 5.492, "r": -0.059}
# Revised (realistic) thresholds for the current single-trajectory protocol
_THRESH = {"r": 0.30, "rmse": 4.00, "sign": 0.60}


def _p(msg: str = "") -> None:
    """ASCII-safe print (Windows consoles choke on the bridge's emoji)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _rule(title: str = "") -> None:
    _p("=" * 92)
    if title:
        _p(title)
        _p("=" * 92)


def _ensure_2lzm() -> Path:
    cache_dir = _ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = cache_dir / "2LZM.pdb"
    if pdb.is_file() and pdb.stat().st_size > 1000:
        return pdb
    _p("cache/2LZM.pdb missing — downloading from RCSB...")
    import requests
    resp = requests.get("https://files.rcsb.org/download/2LZM.pdb", timeout=30)
    resp.raise_for_status()
    pdb.write_bytes(resp.content)
    return pdb


def _parse_chain_a(pdb: Path) -> dict[int, str]:
    """{resseq: one_letter} for chain A CA atoms."""
    seq: dict[int, str] = {}
    for ln in pdb.read_text(errors="replace").splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21:22] == "A":
            try:
                seq[int(ln[22:26])] = _3TO1.get(ln[17:20].strip(), "X")
            except ValueError:
                pass
    return seq


def _bucket(ddg: float | None) -> str:
    """STRONG >+3 | MODERATE +1..+3 | NEUTRAL -1..+1 | STABILISING <-1."""
    if ddg is None:
        return "FAILED"
    if ddg > 3.0:
        return "STRONG"
    if ddg > 1.0:
        return "MODERATE"
    if ddg >= -1.0:
        return "NEUTRAL"
    return "STABILISING"


def _sign_ok(pred: float, exp: float) -> bool:
    """Sign agreement, not penalising when either value is within ±0.5 (neutral)."""
    if abs(pred) < _NEUTRAL_BAND or abs(exp) < _NEUTRAL_BAND:
        return True
    return (pred > 0) == (exp > 0)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, computed manually. None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def main() -> int:
    _rule("PyRosetta / WSL2 ddG validation — T4 lysozyme (2LZM) vs experiment")

    pdb_path = _ensure_2lzm()
    _p(f"PDB: {pdb_path} ({pdb_path.stat().st_size} bytes)")

    seq = _parse_chain_a(pdb_path)
    _p(f"Chain A: residues {min(seq)}..{max(seq)} ({len(seq)} residues)")

    bridge = RosettaBridge()
    _p(f"Backend: {bridge._backend!r} — {bridge.backend_status()}")
    if bridge._backend != "local":
        _p("WARNING: backend is not 'local'; results will not exercise PyRosetta.")

    # Verify / correct wild-type letters against the real structure.
    _rule("RESIDUE VERIFICATION (vs live 2LZM chain A)")
    panel = []
    for pos, claimed, to_aa, exp, group, note in CONTROLS:
        actual = seq.get(pos, "?")
        from_aa = claimed
        if actual != claimed and actual in _3TO1.values():
            _p(f"  NOTE pos {pos}: claimed WT {claimed} but structure has {actual} "
               f"-> using {actual} (key relabelled).")
            from_aa = actual
        key = f"{from_aa}{pos}{to_aa}"
        flag = "OK" if actual == claimed else f"corrected->{actual}"
        _p(f"  {key:<7} pos {pos:<4} WT={actual}  exp={exp:+.1f}  "
           f"group={group:<13} {flag}  ({note})")
        panel.append({"key": key, "chain": "A", "position": pos,
                      "from_aa": from_aa, "to_aa": to_aa,
                      "exp": exp, "group": group})

    # ── Score each mutation, one at a time ────────────────────────────────────
    def _quiet(msg: str) -> None:
        if any(t in msg for t in ("ddG =", "FATAL", "failed", "Loading cached",
                                   "Relax complete", "falling back")):
            _p(f"      {msg.strip()}")

    rows = []
    _rule("SCORING (one mutation per _run_rosetta_local call)")
    for m in panel:
        mut = {"chain": m["chain"], "position": m["position"],
               "from_aa": m["from_aa"], "to_aa": m["to_aa"]}
        _p(f"  -> {m['key']} ({m['group']}, exp={m['exp']:+.1f}) ...")
        t0 = time.perf_counter()
        result = bridge._run_rosetta_local(
            pdb_path          = str(pdb_path),
            mutations         = [mut],
            model_id          = "1",
            chain             = "A",
            progress_callback = _quiet,
        )
        dt = time.perf_counter() - t0
        data = result.data or {}
        ddg = (data.get("ddg_scores", {}) or {}).get(m["key"])
        src = (data.get("ddg_source", {}) or {}).get(m["key"], "none")
        conf = data.get("confidence", "?")
        if not result.success:
            _p(f"     [error] {result.error}")
        err = (ddg - m["exp"]) if ddg is not None else None
        rows.append({**m, "ddg": ddg, "err": err, "source": src,
                     "confidence": conf, "secs": dt})
        ddg_s = f"{ddg:+.3f}" if ddg is not None else "None"
        err_s = f"{err:+.3f}" if err is not None else "  N/A"
        _p(f"     {m['key']}: pred={ddg_s} exp={m['exp']:+.2f} err={err_s} "
           f"[{src}] {conf}  ({dt:.0f}s)")

    # ── Results table ─────────────────────────────────────────────────────────
    _rule("RESULTS TABLE")
    _p(f"  {'Mutation':<8} {'Group':<13} {'pred':>8} {'exp':>7} {'err':>8} "
       f"{'source':<11} {'conf':<6}")
    _p("  " + "-" * 70)
    for r in rows:
        pred_s = f"{r['ddg']:+.3f}" if r["ddg"] is not None else "  None"
        err_s = f"{r['err']:+.3f}" if r["err"] is not None else "   N/A"
        _p(f"  {r['key']:<8} {r['group']:<13} {pred_s:>8} {r['exp']:>+7.2f} "
           f"{err_s:>8} {r['source']:<11} {str(r['confidence']):<6}")

    scored = [r for r in rows if r["ddg"] is not None]
    if not scored:
        _rule("VERDICT")
        _p("No mutations produced a ddG value — cannot assess. Check the WSL2 path.")
        return 0

    all_empirical = all(r["source"] == "empirical" for r in scored)

    # ── Metrics ───────────────────────────────────────────────────────────────
    _rule("METRICS (predicted vs experimental)")
    preds = [r["ddg"] for r in scored]
    exps = [r["exp"] for r in scored]

    n_sign_ok = sum(1 for r in scored if _sign_ok(r["ddg"], r["exp"]))
    sign_acc = n_sign_ok / len(scored)
    mae = sum(abs(r["err"]) for r in scored) / len(scored)
    rmse = math.sqrt(sum(r["err"] ** 2 for r in scored) / len(scored))
    r_p = _pearson(preds, exps)

    _p(f"  N scored               : {len(scored)}/{len(rows)}")
    _p(f"  Sign accuracy          : {n_sign_ok}/{len(scored)} = {sign_acc:.0%}  "
       f"(|ddG|<{_NEUTRAL_BAND} treated as neutral, not penalised)")
    _p(f"  Mean absolute error    : {mae:.3f} kcal/mol")
    _p(f"  RMSE                   : {rmse:.3f} kcal/mol")
    _p(f"  Pearson r              : "
       f"{('%.3f' % r_p) if r_p is not None else 'undefined'}")

    # ── Bucket comparison ─────────────────────────────────────────────────────
    _rule("PER-MUTATION BUCKET COMPARISON (predicted vs experimental range)")
    _p(f"  {'Mutation':<8} {'pred bucket':<13} {'exp bucket':<13} {'match':<6} {'sign':<6}")
    _p("  " + "-" * 52)
    n_bucket_match = 0
    for r in scored:
        pb = _bucket(r["ddg"])
        eb = _bucket(r["exp"])
        match = "yes" if pb == eb else "no"
        if pb == eb:
            n_bucket_match += 1
        sgn = "ok" if _sign_ok(r["ddg"], r["exp"]) else "WRONG"
        _p(f"  {r['key']:<8} {pb:<13} {eb:<13} {match:<6} {sgn:<6}")
    _p(f"  bucket matches: {n_bucket_match}/{len(scored)}")

    # ── Large-cavity underestimation check (known failure mode) ───────────────
    _rule("LARGE-CAVITY UNDERESTIMATION CHECK (single-trajectory washout)")
    lc = [r for r in scored if r["key"] in _LARGE_CAVITY]
    underestimated = []
    for r in lc:
        gap = r["exp"] - r["ddg"]   # +ve => predicted below experiment
        tag = "UNDERESTIMATED" if gap > 1.0 else "ok"
        if gap > 1.0:
            underestimated.append(r["key"])
        _p(f"  {r['key']:<7} pred={r['ddg']:+.3f}  exp={r['exp']:+.2f}  "
           f"gap(exp-pred)={gap:+.3f}  {tag}")
    l99a = next((r for r in scored if r["key"] == "L99A"), None)
    if l99a is not None:
        gap99 = l99a["exp"] - l99a["ddg"]
        washed = gap99 > 2.0
        _p(f"\n  L99A: predicted {l99a['ddg']:+.3f} vs experimental {l99a['exp']:+.2f} "
           f"(gap {gap99:+.3f})")
        _p(f"  -> {'CONFIRMS washout' if washed else 'no major washout'}: "
           f"the canonical +5.0 cavity penalty "
           f"{'is substantially underestimated' if washed else 'is reasonably recovered'}.")
    if lc:
        mean_lc_pred = sum(r["ddg"] for r in lc) / len(lc)
        mean_lc_exp = sum(r["exp"] for r in lc) / len(lc)
        _p(f"  large-cavity mean: predicted {mean_lc_pred:+.3f} vs "
           f"experimental {mean_lc_exp:+.3f} "
           f"(predicted is {'LOWER' if mean_lc_pred < mean_lc_exp else 'higher'})")

    # ── Verdict vs Benchmark Run 1 ────────────────────────────────────────────
    _rule("VERDICT — this run vs Benchmark Run 1 (rosetta_validation_notes.md)")
    if all_empirical:
        _p("All values came from the EMPIRICAL fallback — PyRosetta did not run. "
           "This is NOT a valid experimental comparison; investigate the WSL2 path.")
        return 0

    def _cmp(label, val, br1, better_higher, unit=""):
        arrow = "better" if ((val > br1) == better_higher) else "worse"
        if abs(val - br1) < 1e-9:
            arrow = "same"
        _p(f"  {label:<18}: this={val:.3f}{unit}   run1={br1:.3f}{unit}   ({arrow} than run 1)")

    _cmp("Sign accuracy", sign_acc, _BR1["sign"], better_higher=True)
    _cmp("MAE", mae, _BR1["mae"], better_higher=False, unit=" kcal/mol")
    _cmp("RMSE", rmse, _BR1["rmse"], better_higher=False, unit=" kcal/mol")
    if r_p is not None:
        _cmp("Pearson r", r_p, _BR1["r"], better_higher=True)

    _p("")
    _p("  vs revised thresholds (r>0.30, RMSE<4.00, sign>=0.60):")
    _p(f"    Pearson r > 0.30   : "
       f"{'PASS' if (r_p is not None and r_p > _THRESH['r']) else 'FAIL'} "
       f"(r={('%.3f' % r_p) if r_p is not None else 'NA'})")
    _p(f"    RMSE < 4.00        : {'PASS' if rmse < _THRESH['rmse'] else 'FAIL'} "
       f"(RMSE={rmse:.3f})")
    _p(f"    Sign acc >= 0.60   : {'PASS' if sign_acc >= _THRESH['sign'] else 'FAIL'} "
       f"(sign={sign_acc:.0%})")

    _rule("INTERPRETATION")
    _p("This 2LZM panel is intentionally cavity-heavy, so it stresses the known")
    _p("single-trajectory FastRelax failure mode. Expect MAE/RMSE dominated by")
    _p("underestimated large-cavity mutations (L99A et al.); a positive Pearson r")
    _p("on an independent protein would still confirm the protocol ranks")
    _p("destabilising mutations correctly even if it compresses their magnitude.")
    _p("Multi-trajectory averaging (build-queue item 4) is the documented fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
