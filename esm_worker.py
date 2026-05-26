"""
esm_worker.py
-------------
Standalone ESM-2 inference worker.  Designed to run inside venv312
(Python 3.12 + torch 2.11.0+cu128) so the main Python 3.14 process can
delegate GPU-accelerated inference without needing a working CUDA build.

Usage
-----
python esm_worker.py --input input.json --output output.json

Input JSON
----------
{
  "sequence": "MKTLL...",
  "model":    "esm2_t6_8M_UR50D"   # optional, this is the default
}

Output JSON (success)
---------------------
{
  "ok":       true,
  "probs":    [[p_A, p_C, ...], ...],   # shape (L, 20) — 20 standard AAs
  "device":   "cuda",
  "elapsed_s": 2.14
}

Output JSON (failure)
---------------------
{
  "ok":    false,
  "error": "ImportError: No module named 'esm'"
}

Performance
-----------
Uses batched masking: all L masked variants of the sequence are stacked into
a single (L, L+2) tensor and processed in mini-batches of BATCH_SIZE=32.
This reduces the number of forward passes from L to ceil(L/32), giving a
~30x raw throughput gain over the position-by-position loop in the CPU path.

On RTX 5070 Ti (16 GB VRAM), esm2_t6_8M_UR50D (8 M params) can handle
full batches of BATCH_SIZE=32 at any practical sequence length.  Sequences
longer than ~2000 residues may require reducing BATCH_SIZE to avoid OOM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

# 20 canonical amino acids in the order fair-esm uses for AA-prob extraction
_STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
_BATCH_SIZE   = 32          # residues per GPU mini-batch


def _load_model(model_name: str):
    """Load ESM-2 model + alphabet via fair-esm.  Returns (model, alphabet)."""
    import esm as esm_lib                                    # type: ignore

    loader = getattr(esm_lib.pretrained, model_name, None)
    if loader is None:
        raise AttributeError(
            f"Unknown ESM-2 model '{model_name}'. "
            f"Valid examples: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, "
            f"esm2_t30_150M_UR50D"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, alphabet = loader()
    model.eval()
    return model, alphabet


def _select_device() -> str:
    """Return 'cuda' if a minimal CUDA kernel launch succeeds, else 'cpu'."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _t = torch.zeros(4, device="cuda")
        _ = (_t + 1).sum().item()          # force a kernel launch
        return "cuda"
    except Exception:
        return "cpu"


def _run_inference(sequence: str, model_name: str) -> dict:
    """
    Core inference: compute per-position masked-LM probability distributions.

    Returns a dict suitable for JSON serialisation:
      {"ok": True, "probs": [...], "device": "cuda", "elapsed_s": 2.14}
    or
      {"ok": False, "error": "..."}
    """
    t0 = time.perf_counter()

    import torch

    # ── Device selection ──────────────────────────────────────────────────────
    device = _select_device()
    print(f"[ESM-worker] device={device}  model={model_name}", flush=True)

    # ── Model load ────────────────────────────────────────────────────────────
    t_load = time.perf_counter()
    model, alphabet = _load_model(model_name)
    model = model.to(device)
    print(
        f"[ESM-worker] Model loaded ({time.perf_counter() - t_load:.1f}s)",
        flush=True,
    )

    # ── Tokenise ──────────────────────────────────────────────────────────────
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("query", sequence)])   # (1, L+2)
    L = len(sequence)

    # ── Build all L masked copies in one shot ─────────────────────────────────
    # batch_tokens shape: (L, L+2)  — row i has position i+1 masked
    batch_tokens = tokens.repeat(L, 1)             # (L, L+2)
    for i in range(L):
        batch_tokens[i, i + 1] = alphabet.mask_idx  # +1 skips <cls>

    # ── Batched forward passes ────────────────────────────────────────────────
    all_probs: list[list[float]] = []
    t_inf = time.perf_counter()

    with torch.no_grad():
        for batch_start in range(0, L, _BATCH_SIZE):
            batch_end   = min(batch_start + _BATCH_SIZE, L)
            mini_batch  = batch_tokens[batch_start:batch_end].to(device)
            logits      = model(mini_batch)["logits"]          # (B, L+2, vocab)

            for local_j, global_i in enumerate(range(batch_start, batch_end)):
                pos_logits = logits[local_j, global_i + 1]    # logit at masked pos
                probs_full = (
                    torch.softmax(pos_logits, dim=-1).cpu().numpy()
                )
                aa_probs = [
                    float(probs_full[alphabet.get_idx(aa)])
                    for aa in _STANDARD_AA
                ]
                total = sum(aa_probs)
                all_probs.append(
                    [p / total for p in aa_probs] if total > 0 else aa_probs
                )

            elapsed = time.perf_counter() - t_inf
            print(
                f"[ESM-worker] {batch_end}/{L} residues "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    return {
        "ok":        True,
        "probs":     all_probs,
        "device":    device,
        "elapsed_s": round(elapsed_total, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESM-2 masked-LM inference worker (GPU-accelerated)"
    )
    parser.add_argument("--input",  required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    # ── Read input ────────────────────────────────────────────────────────────
    try:
        with open(args.input, "r", encoding="utf-8") as fh:
            inp = json.load(fh)
    except Exception as exc:
        result = {"ok": False, "error": f"Failed to read input: {exc}"}
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        sys.exit(1)

    sequence   = inp.get("sequence", "")
    model_name = inp.get("model", "esm2_t6_8M_UR50D")

    if not sequence:
        result = {"ok": False, "error": "Input JSON missing 'sequence' key"}
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        result = _run_inference(sequence, model_name)
    except Exception as exc:
        import traceback
        result = {
            "ok":    False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(),
        }

    # ── Write output ──────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    if not result["ok"]:
        print(f"[ESM-worker] FAILED: {result['error']}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(
        f"[ESM-worker] Done: {len(result['probs'])} positions "
        f"on {result['device']} in {result['elapsed_s']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
