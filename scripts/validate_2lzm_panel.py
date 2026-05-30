#!/usr/bin/env python3
"""
scripts/validate_2lzm_panel.py
------------------------------
2LZM confirmation panel at the HIGH-ACCURACY VALIDATION TIER
(ROSETTA_VALIDATION_TRAJECTORIES trajectories x ROSETTA_VALIDATION_CYCLES relax
cycles, MEDIAN aggregation) to measure RMSE vs experiment before deciding whether
to keep or soften the "magnitudes approximate" disclosure.

This is a multi-hour PyRosetta/WSL2 job, so it is built RESTARTABLE:
  * one JSON line is appended to scripts/validate_2lzm_results.jsonl the moment
    each mutation finishes -- a crash never loses completed work;
  * on startup it reads that file and SKIPS mutations already done with a real
    PyRosetta result (empirical-fallback lines are retried, not skipped);
  * the per-mutation FastRelax WT baseline is cached by PDB hash, so resume is
    cheap (only the unfinished mutations re-run).

Ground truth (mutations + experimental ddG) and the metric helpers are REUSED,
not retyped, from scripts/test_pyrosetta_ddg_t4lysozyme.py (CONTROLS, _pearson,
_sign_ok, _ensure_2lzm, _parse_chain_a, _3TO1, _BR1, _THRESH). The validation-tier
entry point is the SAME one the chat tool uses:
RosettaBridge.validate_ddg(), which calls _run_rosetta_local with
num_trajectories=config.ROSETTA_VALIDATION_TRAJECTORIES and
relax_cycles=config.ROSETTA_VALIDATION_CYCLES.

Usage:
    python scripts/validate_2lzm_panel.py             # full 10-mutation panel
    python scripts/validate_2lzm_panel.py --limit 1   # smoke-test ONE mutation
    python scripts/validate_2lzm_panel.py --only L99A # one named mutation

Requires ROSETTA_BACKEND=local (PyRosetta in WSL2); exits with a clear message
otherwise. Read-only with respect to project modules; writes only its own
results files under scripts/.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup: repo root (for config/rosetta_bridge) + scripts dir (sibling) ─
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

config.load_env_file()

RESULTS_JSONL = _ROOT / "scripts" / "validate_2lzm_results.jsonl"
RESULTS_JSON = _ROOT / "scripts" / "validate_2lzm_results.json"


def _say(msg: str = "") -> None:
    """ASCII-safe print -- the bridge's progress messages contain emoji that
    crash the Windows cp1252 console/redirect; replace rather than raise."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


def _inhibit_system_sleep() -> None:
    """Keep Windows awake for the duration of this (multi-hour) run.

    Requests ES_CONTINUOUS | ES_SYSTEM_REQUIRED so the OS will not sleep or
    hibernate while the panel is scoring. ES_DISPLAY_REQUIRED is deliberately
    omitted -- there is no need to keep the monitor on. An atexit handler
    clears the request (ES_CONTINUOUS alone) when the script finishes or is
    Ctrl+C'd; the OS also releases it on process exit, so this is belt-and-
    suspenders. Silent no-op off Windows, and wrapped so it can never crash
    the run."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        atexit.register(
            ctypes.windll.kernel32.SetThreadExecutionState, ES_CONTINUOUS
        )
        _say("Sleep inhibited: system will stay awake until this run exits "
             "(display may still sleep).")
    except Exception:
        pass


# Single-trajectory baselines for the side-by-side (SOURCED, not invented):
#   BASELINE_CROSS (cross-protein Benchmark Run 1) is read from the T4 test
#   module's _BR1 constant at run time.
#   BASELINE_T4_SINGLE: the 2LZM-only single-trajectory cross-check, as recorded
#   in scripts/rosetta_validation_notes.md (Conclusion: r=+0.487, sign 100%) and
#   PROJECT_CONTEXT.md (MAE 3.92, RMSE 5.23). Display-only reference.
BASELINE_T4_SINGLE = {"sign": 1.00, "r": 0.487, "mae": 3.92, "rmse": 5.23}

# Heavy / backend-forcing modules are imported lazily in _load_deps() so that
# --help and argument validation work regardless of ROSETTA_BACKEND.
t4 = None              # scripts/test_pyrosetta_ddg_t4lysozyme module
RosettaBridge = None   # rosetta_bridge.RosettaBridge
BASELINE_CROSS: dict = {}


def _require_local_backend() -> None:
    """Exit unless ROSETTA_BACKEND=local -- the only backend that runs the tier."""
    backend = os.environ.get("ROSETTA_BACKEND", "auto").strip().lower()
    if backend != "local":
        print(
            "ERROR: ROSETTA_BACKEND must be 'local' for the 2LZM validation panel.\n"
            f"  Current value: {backend!r}\n"
            "  This job measures the PyRosetta/WSL2 validation tier; the other\n"
            "  backends (dynamut2/empirical) would not exercise it. Set it and re-run:\n"
            "      PowerShell:  $env:ROSETTA_BACKEND='local'\n"
            "      bash/WSL:    export ROSETTA_BACKEND=local\n"
            "  (or add a line  ROSETTA_BACKEND=local  to .env.local).",
            flush=True,
        )
        raise SystemExit(2)


def _load_deps() -> None:
    """Import the dataset module + bridge once the backend guard has passed."""
    global t4, RosettaBridge, BASELINE_CROSS
    import test_pyrosetta_ddg_t4lysozyme as _t4   # forces backend=local; harmless
    from rosetta_bridge import RosettaBridge as _RB
    t4 = _t4
    RosettaBridge = _RB
    BASELINE_CROSS = dict(_t4._BR1)


def _build_panel() -> tuple[str, list[dict]]:
    """Return (pdb_path, panel) reusing the T4 dataset, WT-verified vs structure."""
    pdb_path = t4._ensure_2lzm()
    seq = t4._parse_chain_a(pdb_path)
    panel: list[dict] = []
    for pos, claimed, to_aa, exp, group, _note in t4.CONTROLS:
        actual = seq.get(pos, "?")
        from_aa = claimed
        if actual != claimed and actual in t4._3TO1.values():
            # Relabel the key to the real WT letter so it matches what the bridge
            # computes (mutation key = from_aa + pos + to_aa).
            from_aa = actual
        panel.append({
            "key": f"{from_aa}{pos}{to_aa}",
            "chain": "A", "position": pos,
            "from_aa": from_aa, "to_aa": to_aa,
            "exp": float(exp), "group": group,
        })
    return str(pdb_path), panel


def _load_done() -> dict[str, dict]:
    """Read the JSONL log; last line per mutation key wins (so retries supersede)."""
    done: dict[str, dict] = {}
    if RESULTS_JSONL.is_file():
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key"):
                done[rec["key"]] = rec
    return done


def _is_complete(rec: dict | None) -> bool:
    """A mutation counts as done only with a finite PyRosetta median (not empirical)."""
    return (
        rec is not None
        and rec.get("source") == "pyrosetta"
        and isinstance(rec.get("predicted_median"), (int, float))
        and math.isfinite(rec["predicted_median"])
    )


def _append(rec: dict) -> None:
    """Append one result line and flush+fsync immediately (crash-safe)."""
    with RESULTS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _score_one(bridge, pdb_path: str, m: dict) -> dict:
    """Run the validation tier for ONE mutation and return a result record."""
    t0 = time.perf_counter()
    result = bridge.validate_ddg(
        pdb_path=pdb_path,
        mutations=[{
            "chain": m["chain"], "position": m["position"],
            "from_aa": m["from_aa"], "to_aa": m["to_aa"],
        }],
        model_id="1",
        chain="A",
        progress_callback=lambda msg: _say(f"      {msg}"),
    )
    secs = time.perf_counter() - t0
    data = result.data or {}
    scores = data.get("ddg_scores", {}) or {}
    # Prefer our computed key; fall back to whatever single key the bridge returned.
    key = m["key"] if m["key"] in scores else next(iter(scores), m["key"])
    return {
        "key": key,
        "chain": m["chain"], "position": m["position"],
        "from_aa": m["from_aa"], "to_aa": m["to_aa"],
        "exp": m["exp"], "group": m["group"],
        "predicted_median": scores.get(key),
        "mad_spread": (data.get("ddg_spread", {}) or {}).get(key),
        "ddg_confidence": (data.get("ddg_confidence", {}) or {}).get(key),
        "source": (data.get("ddg_source", {}) or {}).get(key),
        "n_trajectories": data.get("n_trajectories"),
        "tier": data.get("tier", "validation"),
        "secs": round(secs, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "success": bool(result.success),
        "error": result.error,
    }


def _print_summary(panel: list[dict], done: dict[str, dict]) -> None:
    """When all 10 are done, compute metrics + side-by-side vs single-trajectory."""
    recs = [done.get(m["key"]) for m in panel]
    n_done = sum(1 for r in recs if _is_complete(r))
    if n_done < len(panel):
        print(
            f"\n[partial] {n_done}/{len(panel)} mutations complete "
            "-- full metrics deferred until the whole panel is done. "
            "Re-run (no flags) to continue.",
            flush=True,
        )
        return

    preds = [r["predicted_median"] for r in recs]
    exps = [r["exp"] for r in recs]
    errs = [p - e for p, e in zip(preds, exps)]
    n = len(recs)
    sign_acc = sum(1 for p, e in zip(preds, exps) if t4._sign_ok(p, e)) / n
    mae = sum(abs(x) for x in errs) / n
    rmse = math.sqrt(sum(x * x for x in errs) / n)
    r_p = t4._pearson(preds, exps)

    print("\n" + "=" * 78)
    print("2LZM VALIDATION-TIER PANEL -- COMPLETE (median-aggregated)")
    print("=" * 78)
    print(f"  {'Mutation':<8} {'group':<13} {'pred(med)':>9} {'spread':>7} "
          f"{'exp':>6} {'err':>8} {'conf':<8}")
    print("  " + "-" * 64)
    for r in recs:
        print(f"  {r['key']:<8} {r['group']:<13} {r['predicted_median']:>+9.3f} "
              f"{(r['mad_spread'] if r['mad_spread'] is not None else 0.0):>7.2f} "
              f"{r['exp']:>+6.1f} {r['predicted_median'] - r['exp']:>+8.3f} "
              f"{str(r['ddg_confidence']):<8}")

    n_traj = recs[0].get("n_trajectories", config.ROSETTA_VALIDATION_TRAJECTORIES)
    print("\n  Validation tier: "
          f"N={n_traj} trajectories x {config.ROSETTA_VALIDATION_CYCLES} relax cycles, median.")
    print(f"  {'metric':<16} {'this (val tier)':>16} {'1-traj Run1*':>14} {'1-traj 2LZM**':>14}")
    print("  " + "-" * 62)
    r_p_s = f"{r_p:+.3f}" if r_p is not None else "undefined"
    print(f"  {'Sign accuracy':<16} {sign_acc:>15.0%} "
          f"{BASELINE_CROSS['sign']:>13.0%} {BASELINE_T4_SINGLE['sign']:>13.0%}")
    print(f"  {'MAE (kcal/mol)':<16} {mae:>16.3f} "
          f"{BASELINE_CROSS['mae']:>14.3f} {BASELINE_T4_SINGLE['mae']:>14.3f}")
    print(f"  {'RMSE (kcal/mol)':<16} {rmse:>16.3f} "
          f"{BASELINE_CROSS['rmse']:>14.3f} {BASELINE_T4_SINGLE['rmse']:>14.3f}")
    print(f"  {'Pearson r':<16} {r_p_s:>16} "
          f"{BASELINE_CROSS['r']:>+14.3f} {BASELINE_T4_SINGLE['r']:>+14.3f}")
    print("\n   * Run1  = cross-protein Benchmark Run 1 (rosetta_validation_notes.md).")
    print("  ** 2LZM = single-trajectory 2LZM cross-check (notes Conclusion + PROJECT_CONTEXT).")

    th = t4._THRESH
    print("\n  vs revised thresholds "
          f"(r>{th['r']}, RMSE<{th['rmse']}, sign>={th['sign']:.0%}):")
    print(f"    Pearson r  : {'PASS' if (r_p is not None and r_p > th['r']) else 'FAIL'} "
          f"(r={r_p_s})")
    print(f"    RMSE       : {'PASS' if rmse < th['rmse'] else 'FAIL'} (RMSE={rmse:.3f})")
    print(f"    Sign acc   : {'PASS' if sign_acc >= th['sign'] else 'FAIL'} (sign={sign_acc:.0%})")

    summary = {
        "panel": "2LZM",
        "tier": "validation",
        "n_trajectories": n_traj,
        "relax_cycles": config.ROSETTA_VALIDATION_CYCLES,
        "n_mutations": n,
        "sign_accuracy": round(sign_acc, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": (round(r_p, 4) if r_p is not None else None),
        "baseline_cross_protein_single_traj": BASELINE_CROSS,
        "baseline_2lzm_single_traj": BASELINE_T4_SINGLE,
        "thresholds": dict(th),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "per_mutation": recs,
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Wrote summary -> {RESULTS_JSON}")


def main() -> int:
    # Force UTF-8 output so the bridge's emoji progress messages never crash the
    # Windows cp1252 console or a redirected log file.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Keep the machine awake for the duration of this multi-hour run / resume.
    _inhibit_system_sleep()

    ap = argparse.ArgumentParser(description="2LZM validation-tier confirmation panel.")
    ap.add_argument("--limit", type=int, default=None,
                    help="consider only the first N panel mutations this run "
                         "(already-done ones within that window are skipped). "
                         "--limit 1 is the single-mutation smoke test.")
    ap.add_argument("--only", type=str, default=None,
                    help="process only the named mutation, e.g. L99A.")
    args = ap.parse_args()

    # Guard first (clear exit if the backend is wrong), THEN import heavy deps.
    _require_local_backend()
    _load_deps()

    pdb_path, panel = _build_panel()
    done = _load_done()

    if args.only and not any(m["key"].upper() == args.only.upper() for m in panel):
        print(f"ERROR: --only {args.only!r} matches no mutation in the panel. "
              f"Available: {', '.join(m['key'] for m in panel)}", flush=True)
        return 2

    # Candidate window FIRST, then skip already-complete within it. --limit
    # selects the first N panel mutations to *consider* this run (NOT the first
    # N still-undone) -- so `--limit 1` always means "only L99A", and a resume
    # re-run of `--limit 1` is a clean no-op skip rather than advancing to the
    # next mutation and silently kicking off a fresh multi-hour job.
    candidates = [m for m in panel
                  if not args.only or m["key"].upper() == args.only.upper()]
    if args.limit is not None:
        candidates = candidates[:max(0, args.limit)]

    todo: list[dict] = []
    for m in candidates:
        prev = done.get(m["key"])
        if _is_complete(prev):
            print(f"[skip] {m['key']} already complete "
                  f"(pred={prev['predicted_median']:+.3f}, "
                  f"spread={prev.get('mad_spread')}, resume).", flush=True)
            continue
        todo.append(m)

    bridge = RosettaBridge()
    backend = os.environ.get("ROSETTA_BACKEND", "auto").strip().lower()
    print(f"Backend: ROSETTA_BACKEND={backend!r} | bridge={bridge._backend!r}")
    print(f"  {bridge.backend_status()}")
    print(f"Validation tier: N={config.ROSETTA_VALIDATION_TRAJECTORIES} trajectories "
          f"x {config.ROSETTA_VALIDATION_CYCLES}+{config.ROSETTA_VALIDATION_CYCLES} "
          f"relax cycles, median aggregation.")
    print(f"Panel: {len(panel)} mutations | {len(todo)} to run this invocation | "
          f"results -> {RESULTS_JSONL.name}\n")

    for m in todo:
        idx = next(j for j, mm in enumerate(panel, 1) if mm["key"] == m["key"])
        print(f"[{idx}/{len(panel)}] {m['key']} ({m['group']}, "
              f"exp={m['exp']:+.1f}) -- scoring (this can take many minutes)...",
              flush=True)
        rec = _score_one(bridge, pdb_path, m)
        _append(rec)
        done[rec["key"]] = rec

        pm = rec["predicted_median"]
        sp = rec["mad_spread"]
        pm_s = f"{pm:+.3f}" if isinstance(pm, (int, float)) else "None"
        sp_s = f"{sp:.3f}" if isinstance(sp, (int, float)) else "—"
        print(f"[{idx}/{len(panel)}] {m['key']}  pred={pm_s}  spread={sp_s}  "
              f"exp={m['exp']:+.2f}  [{rec['source']}] {rec['ddg_confidence']}  "
              f"({rec['secs']:.0f}s)", flush=True)
        if rec["source"] != "pyrosetta":
            print(f"   [WARN] {m['key']} did not produce a PyRosetta result "
                  f"(source={rec['source']!r}); it will be RETRIED on the next run. "
                  f"error={rec.get('error')!r}", flush=True)

    if not todo:
        print("Nothing to run (all requested mutations already complete).", flush=True)

    _print_summary(panel, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
