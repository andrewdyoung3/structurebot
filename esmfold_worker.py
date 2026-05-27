"""
esmfold_worker.py
-----------------
Standalone ESMFold inference worker.  Designed to run inside venv312
(Python 3.12 + torch 2.11.0+cu128) so the main Python 3.14 process can
delegate GPU-accelerated structure prediction without needing a working
CUDA build in the main venv.

Usage
-----
python esmfold_worker.py --input input.json --output output.json

Input JSON
----------
{
  "sequence": "MKTLL...",
  "label":    "wildtype",          # optional, passed through to output
  "model":    "facebook/esmfold_v1"  # optional, this is the default
}

Output JSON (success)
---------------------
{
  "success":    true,
  "label":      "wildtype",
  "plddt":      {"1": 87.3, "2": 91.2, ...},   # 1-based, 0-100 scale
  "mean_plddt": 88.1,
  "pdb_str":    "ATOM ...",
  "length":     99,
  "error":      null,
  "elapsed_s":  12.4,
  "device":     "cuda"
}

Output JSON (failure)
---------------------
{
  "success": false,
  "error":   "ImportError: No module named 'transformers'"
}

pLDDT extraction
----------------
outputs.plddt has shape (1, seq_len, 37) — 37 atom types per residue.
We take the mean over the 37 atom channels to get per-residue confidence,
then multiply from 0-1 → 0-100 scale (ESMFold natively outputs 0-1).

NO imports from the main StructureBot project — this file is fully standalone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback as _traceback


def _select_device() -> str:
    """Return 'cuda' if a minimal CUDA kernel launch succeeds, else 'cpu'."""
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return "cpu"
        _t = torch.zeros(4, device="cuda")
        _ = (_t + 1).sum().item()   # force a kernel launch
        return "cuda"
    except Exception:
        return "cpu"


def _run_inference(sequence: str, label: str, model_name: str) -> dict:
    """
    Core inference: run ESMFold on *sequence* and return structured output.

    Returns a dict suitable for JSON serialisation (see module docstring).
    """
    t0 = time.perf_counter()

    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, EsmForProteinFolding  # type: ignore
    except ImportError as exc:
        return {"success": False, "error": f"Import failed: {exc}"}

    # ── Device ────────────────────────────────────────────────────────────────
    device = _select_device()
    print(f"ESMFold: device={device}  model={model_name}", flush=True)

    # ── Model load ────────────────────────────────────────────────────────────
    print("ESMFold: loading model...", flush=True)
    t_load = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the full model in float32 first.
        # We deliberately do NOT pass torch_dtype=float16 here because
        # the ESMFold folding head (trunk / structure_module / distogram /
        # compute_tm) has a known float16 precision bug: torch.max on a
        # float16 logit tensor can produce NaN, causing nonzero() to return
        # an empty tensor and crashing with:
        #   IndexError: index 0 is out of bounds for dimension 0 with size 0
        # The fix is mixed precision — ESM-2 stem in float16 for speed,
        # folding head kept in float32 for numerical stability.
        model = EsmForProteinFolding.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        return {"success": False, "error": f"Model load failed: {exc}"}

    model = model.to(device)
    model.eval()
    # Cast ONLY the ESM-2 language-model stem to float16.
    # The folding head (trunk, structure_module, distogram, compute_tm)
    # stays in float32 so pTM computation remains numerically stable.
    # The float16→float32 boundary is handled internally by the model.
    if device == "cuda":
        model.esm = model.esm.half()
    dtype_label = "mixed_fp16" if device == "cuda" else "float32"
    print(
        f"ESMFold: model loaded ({time.perf_counter() - t_load:.1f}s, dtype={dtype_label})",
        flush=True,
    )

    # ── Tokenise + Inference + pLDDT extraction ───────────────────────────────
    # Any exception here (dtype mismatch, CUDA OOM, shape error, etc.) is
    # caught and returned as a structured failure dict.  The caller (main)
    # writes this to the output JSON and exits 0 so the bridge always has
    # a parseable result — it never sees a silent exit-code-1 failure.
    try:
        tokenized = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        print("ESMFold: running inference...", flush=True)
        t_inf = time.perf_counter()
        with torch.no_grad():
            outputs = model(**tokenized)
        print(
            f"ESMFold: inference done ({time.perf_counter() - t_inf:.1f}s)",
            flush=True,
        )

        # outputs.plddt: (1, seq_len, 37) — 37 atom types per residue, 0-1 scale
        plddt_tensor  = outputs.plddt                               # (1, seq_len, 37)
        plddt_per_res = plddt_tensor[0].mean(dim=-1)                # (seq_len,)
        plddt_per_res = plddt_per_res.cpu().float().numpy() * 100   # 0-100 scale

        # 1-based residue dict (string keys for JSON compatibility)
        plddt_dict = {
            str(i + 1): round(float(v), 2)
            for i, v in enumerate(plddt_per_res)
        }
        mean_plddt = round(float(plddt_per_res.mean()), 2) if len(plddt_per_res) else 0.0

        # PDB conversion (best-effort — non-fatal if it fails)
        pdb_str = ""
        try:
            pdb_strings = model.output_to_pdb(outputs)
            pdb_str = pdb_strings[0] if pdb_strings else ""
        except Exception as exc:
            print(f"ESMFold: PDB conversion failed ({exc}); pdb_str will be empty.", flush=True)

        elapsed = time.perf_counter() - t0
        print("ESMFold: done.", flush=True)

        return {
            "success":    True,
            "label":      label,
            "plddt":      plddt_dict,
            "mean_plddt": mean_plddt,
            "pdb_str":    pdb_str,
            "length":     len(sequence),
            "error":      None,
            "elapsed_s":  round(elapsed, 3),
            "device":     device,
            "dtype":      dtype_label,
        }

    except Exception as exc:
        print(
            f"ESMFold: inference exception: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "success": False,
            "label":   label,
            "error":   f"{type(exc).__name__}: {exc}",
            "trace":   _traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESMFold structure-prediction worker (GPU-accelerated)"
    )
    parser.add_argument("--input",  required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    # ── Read input ────────────────────────────────────────────────────────────
    try:
        with open(args.input, "r", encoding="utf-8") as fh:
            inp = json.load(fh)
    except Exception as exc:
        result: dict = {"success": False, "error": f"Failed to read input: {exc}"}
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        print(f"ESMFold worker FAILED: {result['error']}", flush=True)
        sys.exit(0)  # exit cleanly — bridge reads the error JSON

    sequence   = inp.get("sequence", "")
    label      = inp.get("label",    "query")
    model_name = inp.get("model",    "facebook/esmfold_v1")

    if not sequence:
        result = {"success": False, "error": "Input JSON missing 'sequence' key"}
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        print(f"ESMFold worker FAILED: {result['error']}", flush=True)
        sys.exit(0)  # exit cleanly — bridge reads the error JSON

    # ── Run inference ─────────────────────────────────────────────────────────
    # _run_inference catches all internal exceptions and returns a failure dict;
    # this outer try is a last-resort guard for anything that slips through.
    try:
        result = _run_inference(sequence, label, model_name)
    except Exception as exc:
        result = {
            "success": False,
            "error":   f"{type(exc).__name__}: {exc}",
            "trace":   _traceback.format_exc(),
            "label":   label,
        }

    # ── Write output ──────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    if not result.get("success"):
        # Print to stdout (captured by bridge via capture_output=True)
        # so the error is always surfaced regardless of exit code.
        print(
            f"ESMFold worker FAILED: {result.get('error', '?')}",
            flush=True,
        )
        sys.exit(0)  # exit cleanly — bridge reads success=False from the JSON

    print(
        f"ESMFold: {result.get('length', '?')} residues, "
        f"mean pLDDT {result.get('mean_plddt', '?')}, "
        f"device={result.get('device', '?')}, "
        f"elapsed={result.get('elapsed_s', '?')}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
