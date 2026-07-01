"""
download_esmfold_weights.py
---------------------------
Download + cache (and verify) the ESMFold weights the StructureBot ESMFold path expects, using the
EXACT same load call as esmfold_worker.py so they land in the right HuggingFace hub cache location.

The worker loads with HuggingFace transformers:
    AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
so weights cache under  $HF_HOME/hub/models--facebook--esmfold_v1  (or
~/.cache/huggingface/hub/models--facebook--esmfold_v1). This is NOT a torch.hub download — the
esm2_t6_8M_UR50D.pt under ~/.cache/torch/hub is a DIFFERENT (small ESM-2) model, unrelated to ESMFold.

`from_pretrained` is idempotent: it downloads only what's missing/corrupt (HuggingFace shows a
per-file progress bar), then LOADS the model — so a successful run both fetches AND verifies the
weights are complete and loadable. NOTE: ESMFold is ~3.5B params (~8.4 GB on disk), NOT ~2.7 GB.

Run inside venv312 (the ESMFold env — transformers + torch):
    venv312/Scripts/python.exe scripts/download_esmfold_weights.py

Standalone — imports NOTHING from the StructureBot project (mirrors esmfold_worker.py).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Keep this in sync with config.ESMFOLD_MODEL_NAME / esmfold_worker.py's default.
MODEL_NAME = os.environ.get("ESMFOLD_MODEL_NAME", "facebook/esmfold_v1")


def _hf_cache_dir(model_name: str) -> Path:
    """The HuggingFace hub cache dir for *model_name* — the SAME resolution esmfold_bridge uses."""
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        root = Path(hf_home) / "hub"
    else:
        home = Path(os.environ.get("USERPROFILE", "")) or Path.home()
        root = home / ".cache" / "huggingface" / "hub"
    return root / ("models--" + model_name.replace("/", "--"))


def _weights_present(cache_dir: Path) -> bool:
    """True iff a non-empty *.safetensors / *.bin weight file is cached (bridge's _is_model_cached
    check). Guards against sentinels + 0-byte stubs."""
    for subdir in ("snapshots", "blobs"):
        root = cache_dir / subdir
        if not root.exists():
            continue
        for ext in ("*.safetensors", "*.bin"):
            try:
                if any(f.stat().st_size > 0 for f in root.rglob(ext)):
                    return True
            except OSError:
                continue
    return False


def main() -> int:
    print(f"ESMFold weights — model: {MODEL_NAME}")
    cache_dir = _hf_cache_dir(MODEL_NAME)
    print(f"HuggingFace cache dir: {cache_dir}")
    print(f"Pre-check: weights {'already cached' if _weights_present(cache_dir) else 'NOT cached yet'}")

    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, EsmForProteinFolding
    except Exception as exc:  # ImportError etc. — must run inside venv312
        print("FAILED: could not import transformers/torch — run this INSIDE venv312.",
              file=sys.stderr)
        traceback.print_exc()
        return 1

    t0 = time.perf_counter()
    try:
        print("\n[1/2] Tokenizer (downloads if missing)…", flush=True)
        AutoTokenizer.from_pretrained(MODEL_NAME)

        print("[2/2] Model weights — download if missing + LOAD to verify "
              "(~8.4 GB; HuggingFace shows per-file progress below)…", flush=True)
        # EXACT same call as esmfold_worker.py — so a load success proves the worker will load too.
        model = EsmForProteinFolding.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        print("\nFAILED: could not download/load the ESMFold weights — real traceback below.",
              file=sys.stderr)
        traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - t0
    ok = _weights_present(cache_dir)
    print(f"\nVerification:")
    print(f"  model loaded : {type(model).__name__}  ({n_params / 1e6:.0f}M params) in {elapsed:.0f}s")
    print(f"  weights cached: {'PRESENT' if ok else 'NOT FOUND'} at {cache_dir}")
    if ok:
        print("\nSUCCESS — ESMFold weights are present and loadable "
              "(the esmfold_worker.py load path will succeed).")
        return 0
    print("\nWARNING — model loaded but the cache probe found no weight file; check the path above.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
