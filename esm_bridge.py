"""
esm_bridge.py
-------------
Per-residue evolutionary conservation scoring via ESM-2 (Meta AI).

Model used by default: esm2_t6_8M_UR50D  (~30 MB download on first use)
  — 6 transformer layers, 8 M parameters; fast enough for interactive use.
  — For higher accuracy at the cost of speed, set ESM_MODEL env var to
    "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", etc.

CUDA acceleration
-----------------
The model is automatically placed on the GPU when torch.cuda.is_available()
(e.g. RTX 5070 Ti).  CPU is used as fallback.  Force a device with:
  ESM_DEVICE=cpu   — always use CPU (for reproducibility / debugging)
  ESM_DEVICE=cuda  — always use CUDA (fails if no GPU is present)
GPU inference is typically 10–50× faster than CPU for ESM-2, depending on
sequence length and GPU generation.  The speedup is most noticeable on long
sequences (>200 residues) where the per-position masking loop dominates.

Library priority
----------------
1. fair-esm   (pip install fair-esm)        — Meta's official package
2. transformers (pip install transformers)  — HuggingFace fallback

Conservation score
------------------
For each position i, we compute the marginal-entropy H(i) of the model's
masked-token probability distribution over the 20 standard amino acids:

    H(i) = − Σ_a p(a|context) · log₂ p(a|context)

  H = 0   → perfectly conserved (one amino acid has probability ≈ 1)
  H = 4.3 → maximally variable  (all 20 equally likely, log₂(20) ≈ 4.3)

The returned conservation score is 1 − H/H_max, so that:
  1.0 = perfectly conserved
  0.0 = maximally variable

Visualization
-------------
Residues coloured by conservation on a gradient from:
  deep blue  (conservation ≥ 0.8)   — highly conserved
  cyan/white (conservation 0.4–0.8) — moderate
  red        (conservation ≤ 0.2)   — rapidly evolving

Caching
-------
Results are cached on disk as JSON in ./cache/esm_{hash}.json
(hash of sequence + model name) so the model only runs once per sequence.
Disable with ESM_NO_CACHE=1 env var.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config as _cfg
from tool_router import ToolStepResult

# Windows-only flag: prevent child processes from opening a console window.
# Using it on all subprocess calls is belt-and-suspenders — it stops a Python
# subprocess from inheriting and mutating the parent's console handle.
_CREATE_NO_WINDOW: int = (
    subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    if sys.platform == "win32"
    else 0
)


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL   = "esm2_t6_8M_UR50D"
_H_MAX           = math.log2(20)          # ≈ 4.322 bits
_CACHE_DIR       = Path("cache")
_STANDARD_AA     = list("ACDEFGHIKLMNPQRSTVWY")

# 5-band colour scale (conservation score 0→1 → ChimeraX colour)
_CONS_COLOUR_BANDS: List[Tuple[float, float, str]] = [
    (0.8, 1.01, "blue"),
    (0.6, 0.8,  "dodger blue"),
    (0.4, 0.6,  "white"),
    (0.2, 0.4,  "tomato"),
    (0.0, 0.2,  "red"),
]


# ── venv312 CUDA availability (cached) ───────────────────────────────────────

_VENV312_CUDA_AVAILABLE: Optional[bool] = None   # None = not yet checked


def _check_venv312_cuda() -> bool:
    """
    Test whether venv312 (Python 3.12 + torch 2.11.0+cu128) exists and can
    run a CUDA tensor operation.

    The check is performed exactly once per process lifetime; subsequent calls
    return the cached result immediately.

    Decision matrix (ESM_USE_VENV312 env var):
      "false"/"0" → always returns False (CPU path forced)
      "auto"      → True only if VENV312_PYTHON exists + CUDA smoke-test passes
      "true"/"1"  → same as "auto" but an explicit opt-in for clarity

    The smoke-test runs::

        venv312/Scripts/python.exe -c
            "import torch; torch.tensor([1.0]).cuda(); print('CUDA_OK')"

    This loads Python + torch and performs one CUDA kernel launch
    (~5–15 s on first run, mostly JIT/shader compilation).  The result is then
    cached, so later ``analyze()`` calls pay no detection cost.
    """
    global _VENV312_CUDA_AVAILABLE
    if _VENV312_CUDA_AVAILABLE is not None:
        return _VENV312_CUDA_AVAILABLE

    use_setting = _cfg.ESM_USE_VENV312.lower()
    if use_setting in ("false", "0", "no"):
        _VENV312_CUDA_AVAILABLE = False
        return False

    venv312_python = _cfg.VENV312_PYTHON
    if not Path(venv312_python).is_file():
        _VENV312_CUDA_AVAILABLE = False
        return False

    # fair-esm must also be importable inside venv312
    try:
        probe = subprocess.run(
            [
                venv312_python, "-c",
                (
                    "import torch; "
                    "t = torch.tensor([1.0]).cuda(); "
                    "_ = (t + 1).sum().item(); "
                    "import esm; "          # verify fair-esm present
                    "print('CUDA_OK')"
                ),
            ],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,                   # first-run JIT can be slow
            creationflags=_CREATE_NO_WINDOW,
        )
        _VENV312_CUDA_AVAILABLE = (
            probe.returncode == 0 and "CUDA_OK" in probe.stdout
        )
        if not _VENV312_CUDA_AVAILABLE:
            _stderr_snippet = (probe.stderr or "")[:300]
            print(
                f"[ESM] venv312 CUDA check failed "
                f"(rc={probe.returncode}): {_stderr_snippet}",
                flush=True,
            )
    except subprocess.TimeoutExpired:
        print("[ESM] venv312 CUDA check timed out — using CPU backend.", flush=True)
        _VENV312_CUDA_AVAILABLE = False
    except Exception as exc:
        print(f"[ESM] venv312 CUDA check error: {exc}", flush=True)
        _VENV312_CUDA_AVAILABLE = False

    return _VENV312_CUDA_AVAILABLE


# ── CUDA probe ───────────────────────────────────────────────────────────────

def _probe_cuda_device(preferred: str = "cuda") -> str:
    """
    Test whether CUDA is actually usable for tensor operations.

    Returns "cuda" if a minimal tensor computation succeeds, "cpu" otherwise.
    This catches cases where torch.cuda.is_available() is True but the GPU
    kernel image is not available for the device's compute capability (e.g.
    Blackwell sm_120 with a cu126 torch build).
    """
    if preferred != "cuda":
        return preferred
    try:
        import torch
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _t = torch.zeros(4, device="cuda")
            _ = (_t + 1).sum().item()   # forces a kernel launch
        return "cuda"
    except Exception:
        print(
            "[ESM] CUDA device detected but kernel execution failed "
            "(compute capability mismatch). Falling back to CPU.\n"
            "      RTX 5070 Ti (sm_120 Blackwell) requires PyTorch cu130 "
            "with CUDA 13 support.",
            flush=True,
        )
        return "cpu"


# ── Entropy helpers ───────────────────────────────────────────────────────────

def _entropy(probs: List[float]) -> float:
    """Shannon entropy in bits for a probability distribution."""
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def _conservation_from_entropy(h: float) -> float:
    """Map entropy (bits) to a conservation score in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - h / _H_MAX))


# ── Visualization helpers ─────────────────────────────────────────────────────

def _assign_cons_colour(score: float) -> str:
    for lo, hi, colour in _CONS_COLOUR_BANDS:
        if lo <= score < hi:
            return colour
    return "white"


def _build_viz_commands(
    scores:      List[float],
    model_id:    str,
    chain:       Optional[str],
    start_resno: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    Generate compact ChimeraX colour commands for conservation scores.
    Consecutive same-colour runs are merged into one command.
    """
    if not scores:
        return [], []

    chain_spec = f"/{chain}" if chain else ""
    coloured   = [(start_resno + i, _assign_cons_colour(s)) for i, s in enumerate(scores)]

    runs: List[Tuple[str, List[int]]] = []
    for resno, colour in coloured:
        if runs and runs[-1][0] == colour:
            runs[-1][1].append(resno)
        else:
            runs.append((colour, [resno]))

    cmds = [
        f"cartoon #{model_id}",
        f"color #{model_id}{chain_spec} white",
    ]
    exps = [
        "Switch to cartoon representation",
        "Reset all residues to white before applying ESM-2 conservation colours",
    ]

    for colour, resnos in runs:
        if colour == "white":
            continue
        if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
            spec = f":{resnos[0]}-{resnos[-1]}"
        else:
            spec = ":" + ",".join(str(r) for r in resnos)
        full_spec = f"#{model_id}{chain_spec}{spec}"
        cmds.append(f"color {full_spec} {colour}")
        conserved = colour in ("blue", "dodger blue")
        exps.append(
            f"Color residues {spec} {colour} "
            f"({'highly conserved' if conserved else 'variable/rapidly evolving'})"
        )

    cmds.append(f"view #{model_id}")
    exps.append("Fit structure in view")

    return cmds, exps


# ── ESM backend (fair-esm or transformers) ────────────────────────────────────

class _EsmBackend:
    """
    Thin wrapper that tries fair-esm first, then transformers.
    Loads the model lazily and caches the loaded model instance.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model     = None
        self._alphabet  = None
        self._backend   = None    # "fair_esm" | "transformers"
        self._tokenizer = None
        self._device: Optional[str] = None

    def _load(self) -> None:
        if self._model is not None:
            return

        _load_t0 = time.perf_counter()

        import importlib.util
        import sys
        from pathlib import Path

        # ── Ensure venv site-packages takes priority over any global install ─
        #
        # On Windows, pip install --user drops packages into
        #   %APPDATA%\Python\PythonXYZ\site-packages
        # which may appear on sys.path *before* the venv's site-packages,
        # causing the wrong (torch-less) copy of fair-esm to be loaded.
        #
        # We locate the venv relative to this file, and if any AppData path
        # precedes it we move the venv path to position 0 so the venv's
        # packages always win.  We also evict any already-cached 'esm.*'
        # modules so the re-import isn't short-circuited by sys.modules.

        _project_root   = Path(__file__).resolve().parent
        _venv_site_pkgs = _project_root / "venv" / "Lib" / "site-packages"
        _appdata_marker = str(Path.home() / "AppData" / "Roaming" / "Python")

        if _venv_site_pkgs.is_dir():
            _venv_str = str(_venv_site_pkgs)

            # Case-insensitive path lookup (Windows paths are not case-sensitive)
            _venv_idx = next(
                (i for i, p in enumerate(sys.path)
                 if Path(p).resolve() == _venv_site_pkgs),
                None,
            )
            _appdata_idxs = [
                i for i, p in enumerate(sys.path)
                if _appdata_marker.lower() in p.lower()
            ]

            _needs_fix = (
                _venv_idx is None                                       # not on path at all
                or (_appdata_idxs and min(_appdata_idxs) < _venv_idx)  # AppData wins the race
            )

            if _needs_fix:
                # Remove the duplicate entry first (avoids double entries)
                if _venv_idx is not None:
                    sys.path.pop(_venv_idx)
                sys.path.insert(0, _venv_str)

                # Evict every cached esm.* module so that the subsequent
                # `import esm` re-resolves against the corrected sys.path
                # rather than returning the already-loaded global copy.
                for _mod in [m for m in list(sys.modules)
                             if m == "esm" or m.startswith("esm.")]:
                    del sys.modules[_mod]

        # ── Try fair-esm ──────────────────────────────────────────────────────
        if importlib.util.find_spec("esm") is not None:
            try:
                import esm as esm_lib
                print(
                    f"\n[ESM] Loading {self.model_name} via fair-esm "
                    f"(~30 MB download on first use)…",
                    flush=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loader = getattr(esm_lib.pretrained, self.model_name, None)
                    if loader is None:
                        raise AttributeError(f"Unknown model: {self.model_name}")
                    self._model, self._alphabet = loader()
                self._model.eval()
                import torch as _torch
                _env_dev = os.environ.get("ESM_DEVICE", "").strip().lower()
                if _env_dev in ("cuda", "cpu"):
                    preferred = _env_dev
                else:
                    preferred = "cuda" if _torch.cuda.is_available() else "cpu"
                self._device = _probe_cuda_device(preferred)
                self._model = self._model.to(self._device)
                self._backend = "fair_esm"
                print(f"[ESM] Model loaded (fair-esm) on {self._device} ({time.perf_counter() - _load_t0:.1f}s).", flush=True)
                return
            except ImportError:
                pass
            except Exception as exc:
                print(f"[ESM] fair-esm load failed: {exc}. Trying transformers…", flush=True)

        # ── Try transformers (HuggingFace) ────────────────────────────────────
        if importlib.util.find_spec("transformers") is not None:
            try:
                from transformers import EsmForMaskedLM, EsmTokenizer  # type: ignore
                hf_name = f"facebook/{self.model_name}"
                print(
                    f"\n[ESM] Loading {hf_name} via transformers "
                    f"(~30 MB download on first use)…",
                    flush=True,
                )
                self._tokenizer = EsmTokenizer.from_pretrained(hf_name)
                self._model     = EsmForMaskedLM.from_pretrained(hf_name)
                self._model.eval()
                import torch as _torch
                _env_dev = os.environ.get("ESM_DEVICE", "").strip().lower()
                if _env_dev in ("cuda", "cpu"):
                    preferred = _env_dev
                else:
                    preferred = "cuda" if _torch.cuda.is_available() else "cpu"
                self._device = _probe_cuda_device(preferred)
                self._model = self._model.to(self._device)
                self._backend   = "transformers"
                print(f"[ESM] Model loaded (transformers) on {self._device} ({time.perf_counter() - _load_t0:.1f}s).", flush=True)
                return
            except ImportError:
                pass
            except Exception as exc:
                print(f"[ESM] transformers load failed: {exc}.", flush=True)

        raise ImportError(
            "ESM-2 requires either 'fair-esm' or 'transformers'.\n"
            "  Install with: pip install fair-esm\n"
            "  Or:           pip install transformers torch"
        )

    def masked_probabilities(
        self,
        sequence: str,
        deadline: Optional[float] = None,
    ) -> List[List[float]]:
        """
        Compute per-position amino-acid probability distributions.

        For each position i, masks that position and runs a forward pass,
        returning the softmax probabilities over the 20 standard amino acids.

        Parameters
        ----------
        sequence : single-letter AA string
        deadline : optional ``time.perf_counter()`` value; if the wall clock
                   passes this value mid-loop a ``TimeoutError`` is raised.

        Returns: list[list[float]], shape (seq_len, 20)
        """
        self._load()

        if self._backend == "fair_esm":
            return self._masked_probs_fair_esm(sequence, deadline=deadline)
        elif self._backend == "transformers":
            return self._masked_probs_transformers(sequence, deadline=deadline)
        else:
            raise RuntimeError("No ESM backend available.")

    def _masked_probs_fair_esm(
        self,
        sequence: str,
        deadline: Optional[float] = None,
    ) -> List[List[float]]:
        """Per-position masked probabilities using the fair-esm library."""
        import torch

        batch_converter = self._alphabet.get_batch_converter()
        data = [("query", sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self._device)

        _inf_start = time.perf_counter()
        all_probs: List[List[float]] = []
        with torch.no_grad():
            for i in range(len(sequence)):
                if deadline is not None and time.perf_counter() >= deadline:
                    elapsed = time.perf_counter() - _inf_start
                    raise TimeoutError(
                        f"ESM-2 inference timed out after {elapsed:.1f}s "
                        f"at position {i + 1}/{len(sequence)}"
                    )
                masked = tokens.clone()
                masked[0, i + 1] = self._alphabet.mask_idx  # +1 for <cls>
                logits = self._model(masked)["logits"]
                # Shape: (1, seq_len+2, vocab_size)
                pos_logits = logits[0, i + 1]
                # Softmax over vocab; .cpu() ensures numpy() works on CUDA tensors
                probs_full = torch.softmax(pos_logits, dim=-1).cpu().numpy()
                # Extract only the 20 standard amino acids
                aa_probs = [
                    float(probs_full[self._alphabet.get_idx(aa)])
                    for aa in _STANDARD_AA
                ]
                total = sum(aa_probs)
                all_probs.append([p / total for p in aa_probs] if total > 0 else aa_probs)

                if (i + 1) % 10 == 0 or (i + 1) == len(sequence):
                    elapsed = time.perf_counter() - _inf_start
                    print(f"[ESM] {i + 1}/{len(sequence)} residues ({elapsed:.1f}s)…", flush=True)

        return all_probs

    def _masked_probs_transformers(
        self,
        sequence: str,
        deadline: Optional[float] = None,
    ) -> List[List[float]]:
        """Per-position masked probabilities using the transformers library."""
        import torch

        _inf_start = time.perf_counter()
        all_probs: List[List[float]] = []
        for i in range(len(sequence)):
            if deadline is not None and time.perf_counter() >= deadline:
                elapsed = time.perf_counter() - _inf_start
                raise TimeoutError(
                    f"ESM-2 inference timed out after {elapsed:.1f}s "
                    f"at position {i + 1}/{len(sequence)}"
                )
            masked_seq = sequence[:i] + "<mask>" + sequence[i+1:]
            inputs = self._tokenizer(
                masked_seq,
                return_tensors="pt",
                add_special_tokens=True,
            )
            mask_pos = (inputs["input_ids"][0] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                all_probs.append([1.0 / 20] * 20)
                if (i + 1) % 10 == 0 or (i + 1) == len(sequence):
                    elapsed = time.perf_counter() - _inf_start
                    print(f"[ESM] {i + 1}/{len(sequence)} residues ({elapsed:.1f}s)…", flush=True)
                continue
            mask_idx = mask_pos[0].item()

            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            logits   = outputs.logits[0, mask_idx]
            # .cpu() ensures numpy() works on CUDA tensors
            probs_full = torch.softmax(logits, dim=-1).cpu().numpy()
            aa_probs = [
                float(probs_full[self._tokenizer.convert_tokens_to_ids(aa)])
                for aa in _STANDARD_AA
            ]
            total = sum(aa_probs)
            all_probs.append([p / total for p in aa_probs] if total > 0 else aa_probs)

            if (i + 1) % 10 == 0 or (i + 1) == len(sequence):
                elapsed = time.perf_counter() - _inf_start
                print(f"[ESM] {i + 1}/{len(sequence)} residues ({elapsed:.1f}s)…", flush=True)

        return all_probs


# ── Bridge class ───────────────────────────────────────────────────────────────

class EsmBridge:
    """
    Computes per-residue evolutionary conservation via ESM-2 masked-language
    modelling and generates ChimeraX visualization commands.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name    = os.environ.get("ESM_MODEL", model_name)
        self._backend      = _EsmBackend(self.model_name)
        # Populated on first analyze() call that needs inference (not cached).
        # "gpu"  → venv312 detected OK, will use GPU subprocess
        # "cpu"  → venv312 absent/broken, using in-process CPU path
        # None   → not yet checked
        self._venv312_status: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        sequence:          str,
        model_id:          str  = "1",
        chain:             Optional[str] = None,
        session:           Any  = None,
        start_resno:       int  = 1,
        inference_timeout: int  = 120,
    ) -> ToolStepResult:
        """
        Compute ESM-2 conservation scores for *sequence*.

        Parameters
        ----------
        sequence          : single-letter AA string (standard letters only)
        model_id          : ChimeraX model number
        chain             : chain ID or None
        session           : SessionState (unused here, reserved for caching)
        start_resno       : first residue number (default 1)
        inference_timeout : seconds before inference is aborted and uniform
                            0.5 fallback scores are used (default 120)

        Returns
        -------
        ToolStepResult with data["conservation"] = {resno: score (0–1)}
        and viz_commands for ChimeraX colouring.
        """
        _t0 = time.perf_counter()

        sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence.upper())
        if len(sequence) < 5:
            return ToolStepResult(
                tool="esm", success=False,
                error=f"Sequence too short ({len(sequence)} residues). ESM-2 requires ≥ 5.",
            )

        # ── Disk cache ─────────────────────────────────────────────────────────
        cache_key    = self._cache_key(sequence)
        cached_probs = self._load_cache(cache_key)
        timed_out    = False
        device_used  = "cpu"       # overwritten below

        if cached_probs is not None:
            all_probs   = cached_probs
            device_used = "cached"
        else:
            # ── One-time backend detection + announcement ───────────────────
            if self._venv312_status is None:
                venv312_ok = _check_venv312_cuda()
                if venv312_ok:
                    self._venv312_status = "gpu"
                    print(
                        "[ESM] Using venv312 GPU backend (RTX 5070 Ti)",
                        flush=True,
                    )
                else:
                    self._venv312_status = "cpu"
                    print(
                        "[ESM] Using CPU backend (venv312 not available)",
                        flush=True,
                    )

            # ── GPU path via venv312 subprocess ────────────────────────────
            all_probs = None
            if self._venv312_status == "gpu":
                all_probs = self._run_esm_in_venv312(sequence, timeout=inference_timeout)
                if all_probs is not None:
                    device_used = "cuda"
                else:
                    print(
                        "[ESM] venv312 inference failed — falling back to CPU.",
                        flush=True,
                    )

            # ── CPU fallback (in-process _EsmBackend) ──────────────────────
            if all_probs is None:
                deadline = time.perf_counter() + inference_timeout
                try:
                    all_probs = self._backend.masked_probabilities(
                        sequence, deadline=deadline
                    )
                    device_used = self._backend._device or "cpu"
                except TimeoutError as exc:
                    elapsed = time.perf_counter() - _t0
                    warnings.warn(
                        f"[ESM] Inference timed out after {elapsed:.1f}s "
                        f"(limit {inference_timeout}s) — "
                        f"falling back to uniform conservation scores (0.5).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    print(
                        f"[ESM] Timeout after {elapsed:.1f}s at {exc}. "
                        f"Using uniform 0.5 fallback.",
                        flush=True,
                    )
                    uniform    = [1.0 / 20] * 20
                    all_probs  = [uniform[:] for _ in sequence]
                    timed_out  = True
                    device_used = "cpu (timeout)"
                except ImportError as exc:
                    return ToolStepResult(
                        tool="esm", success=False, error=str(exc),
                    )
                except Exception as exc:
                    return ToolStepResult(
                        tool="esm", success=False,
                        error=f"ESM-2 inference failed: {exc}",
                    )

            if not timed_out:
                self._save_cache(cache_key, all_probs)

        # ── Conservation scores ─────────────────────────────────────────────
        conservation = [
            _conservation_from_entropy(_entropy(probs))
            for probs in all_probs
        ]

        scores_dict = {
            start_resno + i: round(s, 4)
            for i, s in enumerate(conservation)
        }

        highly_conserved = [r for r, s in scores_dict.items() if s >= 0.8]
        rapidly_evolving = [r for r, s in scores_dict.items() if s <= 0.2]

        viz_cmds, viz_exps = _build_viz_commands(
            conservation, model_id, chain, start_resno
        )

        mean_cons = sum(conservation) / len(conservation) if conservation else 0
        elapsed   = time.perf_counter() - _t0

        if timed_out:
            timing_note = f" (timed out {elapsed:.1f}s, uniform fallback)"
        elif cached_probs is not None:
            timing_note = " (cached)"
        else:
            timing_note = f" ({elapsed:.1f}s, {device_used})"

        summary = (
            f"ESM-2{timing_note}: {len(sequence)} residues. "
            f"Mean conservation {mean_cons:.2f}. "
            f"{len(highly_conserved)} highly conserved, "
            f"{len(rapidly_evolving)} rapidly evolving."
        )

        return ToolStepResult(
            tool    = "esm",
            success = True,
            data    = {
                "conservation":      scores_dict,
                "highly_conserved":  highly_conserved,
                "rapidly_evolving":  rapidly_evolving,
                "mean_conservation": round(mean_cons, 4),
                "model_used":        self.model_name,
                "device":            device_used,
                "cached":            cached_probs is not None,
                "timed_out":         timed_out,
                "elapsed_s":         round(elapsed, 2),
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
        )

    # ── venv312 GPU inference ──────────────────────────────────────────────────

    def _run_esm_in_venv312(
        self,
        sequence: str,
        timeout:  int = 300,
    ) -> Optional[List[List[float]]]:
        """
        Delegate ESM-2 inference to esm_worker.py running inside venv312.

        The worker uses batched masking (all L masked variants processed in
        mini-batches of 32), so the number of forward passes is
        ceil(L / 32) instead of L — roughly a 30× reduction in compute.
        Combined with GPU acceleration this is typically 100–500× faster than
        the CPU position-by-position path for sequences longer than ~50 AA.

        Protocol
        --------
        1. Write ``{"sequence": ..., "model": ...}`` to a temp JSON file.
        2. Run: venv312/Scripts/python.exe esm_worker.py --input … --output …
        3. Read the result JSON; relay worker stdout (progress lines).
        4. Return probs list on success, None on any failure.

        The caller falls back to the CPU _EsmBackend if None is returned.
        """
        worker_path    = str(Path(__file__).parent / "esm_worker.py")
        venv312_python = _cfg.VENV312_PYTHON

        if not Path(worker_path).is_file():
            print(f"[ESM] esm_worker.py not found at {worker_path}", flush=True)
            return None

        inp_path: Optional[str] = None
        out_path: Optional[str] = None
        try:
            # Write sequence + model to temp input file
            inp_fd, inp_path = tempfile.mkstemp(suffix="_esm_in.json")
            os.close(inp_fd)
            with open(inp_path, "w", encoding="utf-8") as fh:
                json.dump({"sequence": sequence, "model": self.model_name}, fh)

            # Temp file for worker output
            out_fd, out_path = tempfile.mkstemp(suffix="_esm_out.json")
            os.close(out_fd)

            print(
                f"[ESM] Launching venv312 worker "
                f"({len(sequence)} residues, batched GPU)…",
                flush=True,
            )

            proc = subprocess.run(
                [
                    venv312_python,
                    worker_path,
                    "--input",  inp_path,
                    "--output", out_path,
                ],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=_CREATE_NO_WINDOW,
            )

            # Relay worker progress output so the user sees it
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    if line.strip():
                        print(line, flush=True)
            if proc.returncode != 0 and proc.stderr:
                for line in proc.stderr.splitlines()[:10]:
                    if line.strip():
                        print(f"[ESM-worker] {line}", flush=True)

            if proc.returncode != 0:
                print(
                    f"[ESM] venv312 worker exited {proc.returncode}",
                    flush=True,
                )
                return None

            # Read results JSON
            if not Path(out_path).is_file():
                print("[ESM] venv312 worker produced no output file", flush=True)
                return None

            with open(out_path, "r", encoding="utf-8") as fh:
                result = json.load(fh)

            if not result.get("ok"):
                print(
                    f"[ESM] venv312 worker error: {result.get('error', '?')}",
                    flush=True,
                )
                return None

            probs = result.get("probs")
            if not probs or len(probs) != len(sequence):
                print(
                    f"[ESM] venv312 worker returned {len(probs or [])} probs "
                    f"for {len(sequence)}-residue sequence",
                    flush=True,
                )
                return None

            print(
                f"[ESM] venv312 GPU done: {len(sequence)} residues in "
                f"{result.get('elapsed_s', '?')}s on {result.get('device', '?')}",
                flush=True,
            )
            return probs

        except subprocess.TimeoutExpired:
            print(
                f"[ESM] venv312 worker timed out after {timeout}s",
                flush=True,
            )
            return None
        except Exception as exc:
            print(f"[ESM] venv312 worker exception: {exc}", flush=True)
            return None
        finally:
            for p in (inp_path, out_path):
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_key(self, sequence: str) -> str:
        tag   = f"{self.model_name}:{sequence}"
        digest = hashlib.sha256(tag.encode()).hexdigest()[:16]
        return digest

    def _cache_path(self, key: str) -> Path:
        return _CACHE_DIR / f"esm_{key}.json"

    def _load_cache(self, key: str) -> Optional[List[List[float]]]:
        if os.environ.get("ESM_NO_CACHE", "").strip() in ("1", "true", "yes"):
            return None
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _save_cache(self, key: str, probs: List[List[float]]) -> None:
        if os.environ.get("ESM_NO_CACHE", "").strip() in ("1", "true", "yes"):
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(probs, fh)
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<EsmBridge model={self.model_name!r}>"
