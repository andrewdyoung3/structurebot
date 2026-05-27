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
        model = EsmForProteinFolding.from_pretrained(
            model_name, low_cpu_mem_usage=True
        )
    except Exception as exc:
        return {"success": False, "error": f"Model load failed: {exc}"}

    model = model.to(device)
    model.eval()
    print(
        f"ESMFold: model loaded ({time.perf_counter() - t_load:.1f}s)",
        flush=True,
    )

    # ── Tokenise ──────────────────────────────────────────────────────────────
    tokenized = tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    # ── Inference ─────────────────────────────────────────────────────────────
    print("ESMFold: running inference...", flush=True)
    t_inf = time.perf_counter()
    with torch.no_grad():
        outputs = model(**tokenized)
    print(
        f"ESMFold: inference done ({time.perf_counter() - t_inf:.1f}s)",
        flush=True,
    )

    # ── Extract pLDDT ─────────────────────────────────────────────────────────
    # outputs.plddt: (1, seq_len, 37) — 37 atom types per residue, 0-1 scale
    plddt_tensor = outputs.plddt                               # (1, seq_len, 37)
    plddt_per_res = plddt_tensor[0].mean(dim=-1)               # (seq_len,)
    plddt_per_res = plddt_per_res.cpu().float().numpy() * 100  # 0-100 scale

    # 1-based residue dict  (keys are strings for JSON compatibility)
    plddt_dict = {
        str(i + 1): round(float(v), 2)
        for i, v in enumerate(plddt_per_res)
    }
    mean_plddt = round(float(plddt_per_res.mean()), 2) if len(plddt_per_res) else 0.0

    # ── PDB conversion ────────────────────────────────────────────────────────
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
        sys.exit(1)

    sequence   = inp.get("sequence", "")
    label      = inp.get("label",    "query")
    model_name = inp.get("model",    "facebook/esmfold_v1")

    if not sequence:
        result = {"success": False, "error": "Input JSON missing 'sequence' key"}
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        result = _run_inference(sequence, label, model_name)
    except Exception as exc:
        import traceback
        result = {
            "success": False,
            "error":   f"{type(exc).__name__}: {exc}",
            "trace":   traceback.format_exc(),
        }

    # ── Write output ──────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    if not result.get("success"):
        print(
            f"ESMFold worker FAILED: {result.get('error', '?')}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(
        f"ESMFold: {result.get('length', '?')} residues, "
        f"mean pLDDT {result.get('mean_plddt', '?')}, "
        f"device={result.get('device', '?')}, "
        f"elapsed={result.get('elapsed_s', '?')}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
