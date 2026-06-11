# RFdiffusion Activation Plan (deferred — attended, GPU-healthy session)

This is the **written** activation recipe + reachability investigation produced by
the GPU-independent scaffold task. **Nothing here was installed, downloaded, or
built.** The bridge (`rfdiffusion_bridge.py`) is wired and tested with inference
mocked; `is_available()` is **False** on this machine until the WSL2 env below
exists. Activation itself (the SE3/torch env build on Blackwell `sm_120`) is the
attended, machine-free, GPU-healthy follow-up — explicitly **not** done here.

Date of investigation: 2026-06-11. **Verified facts vs verify-at-build:** the
probe results in §3–§4 (weights reachable HTTP 200; WSL has only py3.12; weights
host/domains) are measured and settled. Everything in §2 and the §5 version pins
is **forward-looking → verify at build time** — the `sm_120` / dgl toolchain is
moving fast and the dgl long-pole is unresolved.

---

## 0. What the bridge already expects

- Interpreter constant: `wsl_bridge.RFDIFFUSION_PYTHON = /home/andre/rfdiffusion_env/bin/python`
  (the bridge runs `run_inference.py` **only** through this — never VENV312;
  RFdiffusion has no Windows / Py-3.12 path).
- WSL clone path probed: `RFDIFFUSION_WSL_DIR` (default `/home/andre/RFdiffusion`),
  `run_inference.py` accepted at repo root **or** `scripts/`.
- **Single execution path:** `run_inference.py` is ALWAYS dispatched through
  `wsl.exe` → `RFDIFFUSION_PYTHON` (there is no native-Windows / Py-3.12 path).
- Availability only records WHERE the clone lives (for path translation): a
  Windows-visible clone (`RFDIFFUSION_DIR` with `run_inference.py` + `models/`,
  → `win_clone`, fs-checked) **or** a WSL probe of `RFDIFFUSION_PYTHON` + the WSL
  clone (`WSLBridge.check_rfdiffusion`, mirror of `check_colabfold`, → `wsl_clone`).
- Output goes to `config.RFDIFFUSION_CACHE_DIR` (`cache/rfdiffusion/`), a Windows
  dir that is **also** WSL-visible via `/mnt/c/...`; the WSL run writes straight
  into it and the Windows side collects the PDBs (no copy-back).

> **Either clone location is legitimate.** A `C:\`-placed clone (`win_clone`) is
> invoked via the same `wsl.exe` → `RFDIFFUSION_PYTHON` dispatch, with its `C:\`
> script + path overrides translated to `/mnt/c`; a WSL-native clone (`wsl_clone`)
> is already addressable. **Recommended:** the WSL-native layout
> (`/home/andre/RFdiffusion`) avoids the `/mnt/c` boundary I/O penalty on the
> large weights. Either way, execution requires WSL; if WSL/the env is missing at
> dispatch, `_dispatch` returns an error-first result (never a fabricated run).

---

## 1. Verified RFdiffusion v1 CLI (run_inference.py, Hydra overrides)

Verified against the RosettaCommons/RFdiffusion README (not guessed). The bridge
emits exactly these keys:

| Purpose | Override |
|---|---|
| input pdb | `inference.input_pdb=<path>` |
| output prefix | `inference.output_prefix=<path>` |
| num designs | `inference.num_designs=N` |
| diffusion steps | `diffuser.T=50` |
| binder hotspots | `'ppi.hotspot_res=[A30,A33,A34]'` (chain-prefixed) |
| binder contig | `'contigmap.contigs=[B1-100/0 100-100]'` (target range `/0` break + binder length) |
| motif contig | `'contigmap.contigs=[5-15/A10-25/30-40]'` |
| symmetry | **`--config-name symmetry`** + `inference.symmetry=c4` / `d2` / `tetrahedral` (lowercase) |
| partial diffusion | `diffuser.partial_T=20` |

Note: every override containing `[`, `]`, or a space **must be single-quoted** for
bash — the bridge's WSL path does this via `shlex.quote` on each token.

---

## 2. torch / CUDA path for Blackwell `sm_120` (the hard reconciliation)

**The blocker is not RFdiffusion itself — it is its SE3-Transformer dependency
stack, which is pinned to an ancient toolchain incompatible with Blackwell.**

- **Stock `env/SE3nv.yml`** pins roughly `python=3.9`, `pytorch=1.9`,
  `dgl-cuda11.1`. That torch/dgl/CUDA-11.1 combination **cannot** target
  `sm_120` (RTX 5070 Ti / Blackwell). Using it as-is = "no kernel image
  available" at runtime. This is the analogue of the ColabFold `jax-0.5.3`
  reconciliation.
- **Current `sm_120` torch state (2026-06, verify at build):** CUDA 12.8/12.9
  supports Blackwell; stable PyTorch wheels were *still* catching up (2.9.0 stable
  did **not** ship `sm_120`; nightlies `2.10.0.dev+cu129/cu128` detect it).
  **Substantiated datum:** the Windows `venv312` has **`torch 2.11.0+cu128` /
  CUDA 12.8 installed** (import-verified this session: `torch.__version__ ==
  2.11.0+cu128`, `torch.version.cuda == 12.8`). Runtime `sm_120` could **not** be
  re-confirmed right now because the GPU is down post-reboot (`cuda.is_available()
  == False`); prior project work (memory `venv312-cuda`) reports this build ran
  `sm_120` on the RTX 5070 Ti. So the **Linux** target is the `cu128` wheel of an
  `sm_120`-capable torch (try **`torch==2.11.0+cu128`** from the `cu128` index
  first, else the closest nightly exposing `sm_120`) — **not** torch 1.9 — and the
  capability must be re-verified on the env that gets built.
- **dgl is the real risk.** SE3-Transformer uses `dgl` for graph ops, and dgl's
  CUDA wheels historically lag (cu118/cu121/cu124), with no `cu128`/`sm_120`
  build. Two viable routes to evaluate in the attended session:
  1. **dgl on CPU** (graph ops on CPU, diffusion on GPU) — slowest path but
     unblocks Blackwell immediately; verify SE3Transformer tolerates a CPU dgl.
  2. **dgl built/levelled against torch 2.x + cu12x**, then rebuild the in-repo
     `env/SE3Transformer` (`pip install -r requirements.txt && python setup.py
     install`) against the upgraded torch.
- **SE3-Transformer** must be installed from the in-repo copy
  (`env/SE3Transformer/`), plus `hydra-core` and `pyrsistent` — rebuilt against
  whatever torch is chosen above.

**Deliverable for the attended session:** pin a concrete, mutually-compatible
`(python 3.10, torch 2.11.0+cu128, dgl <route>, e3nn, hydra-core, pyrsistent)`
set, confirm `torch.cuda.get_device_capability() == (12, 0)` inside
`~/rfdiffusion_env`, and run the README's `Base_ckpt.pt` smoke design.

---

## 3. Light environment probes (read-only — run this session)

| Probe | Result |
|---|---|
| `wsl -- python3.9/3.10/3.11 --version` | **absent** — WSL (Ubuntu-24.04) ships **python3.12 only** (`Python 3.12.3`). RFdiffusion needs 3.9–3.11 ⇒ must add an interpreter (deadsnakes PPA `python3.10` or a conda/mamba env). |
| `curl -I http://files.ipd.uw.edu/.../Base_ckpt.pt` | **HTTP/1.1 200 OK**, `Content-Length: 483616107` (~461 MB). Weights host **reachable**. |

---

## 4. Weights reachability

- **Host:** `files.ipd.uw.edu` (UW Institute for Protein Design), plain **http**
  (port 80). URLs of the form
  `http://files.ipd.uw.edu/pub/RFdiffusion/<md5>/<Name>_ckpt.pt`.
- **Checkpoints** (from the README download script): `Base_ckpt.pt`,
  `Complex_base_ckpt.pt`, `Complex_Fold_base_ckpt.pt`, `ActiveSite_ckpt.pt`,
  `Base_epoch8_ckpt.pt`, `Complex_beta_ckpt.pt`, and the auxiliary
  `RF_structure_prediction_weights.pt`. **Measured:** `Base_ckpt.pt` is
  **~461 MB** (`Content-Length: 483616107`). Total set size is **to-confirm** —
  count the download script's URLs × their `Content-Length` at build (the
  oft-quoted "~20 GB" is unverified here; a binder run needs only a subset).
- **Reachability:** the HEAD probe **succeeded (200)** from this environment, so
  `files.ipd.uw.edu` is **fetchable**, not blocked. WSL2 NATs through the Windows
  network stack, so if Windows can reach it WSL can too.
- **Exact domains the activation needs on the allow-list:**
  - `files.ipd.uw.edu` (http) — model weights. **← the only non-standard host;
    confirmed reachable here, but verify on the user's network; if blocked, the
    user allow-lists it or supplies the `.pt` files manually.**
  - `github.com` / `codeload.github.com` — `git clone` the repo (standard).
  - `pypi.org` / `files.pythonhosted.org` — pip deps (standard).
  - conda channels (`conda.anaconda.org`: `dglteam`, `nvidia`, `pytorch`) **iff**
    the conda/dgl route in §2 is taken.

---

## 5. Install recipe (WSL2, deferred)

```bash
# 1. Interpreter (WSL has only 3.12; RFdiffusion needs 3.9-3.11)
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# 2. Clone + weights (files.ipd.uw.edu — reachable; total size to-confirm, §4)
git clone https://github.com/RosettaCommons/RFdiffusion ~/RFdiffusion
cd ~/RFdiffusion && bash scripts/download_models.sh models/

# 3. Env (NOT the stock SE3nv.yml torch — see §2; ALL pins here are verify-at-build)
python3.10 -m venv ~/rfdiffusion_env
~/rfdiffusion_env/bin/pip install torch==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128   # confirm sm_120 on the built env
# dgl: choose the §2 route (CPU dgl, or a cu12x build), then:
cd env/SE3Transformer
~/rfdiffusion_env/bin/pip install --no-cache-dir -r requirements.txt
~/rfdiffusion_env/bin/python setup.py install
~/rfdiffusion_env/bin/pip install hydra-core pyrsistent e3nn
cd ~/RFdiffusion && ~/rfdiffusion_env/bin/pip install -e .

# 4. Smoke (proves sm_120 + weights):
~/rfdiffusion_env/bin/python scripts/run_inference.py \
    'contigmap.contigs=[100-100]' inference.output_prefix=test/sm120 \
    inference.num_designs=1
```

After this, the bridge's WSL probe (`check_rfdiffusion`) flips `is_available()` to
True automatically — no bridge code change required.

---

## Sources

- RFdiffusion v1 README & download script — https://github.com/RosettaCommons/RFdiffusion
- PyTorch `sm_120` / Blackwell status — https://github.com/pytorch/pytorch/issues/164342
- Blackwell build guide (torch 2.10 + CUDA 12.8 + cuDNN 9) — https://github.com/bajegani/pytorch-build-blackwell-sm120
- SE3-Transformer install (dgl/e3nn/hydra) — https://github.com/RosettaCommons/RFdiffusion/issues/1 ,
  https://github.com/RosettaCommons/RFdiffusion/issues/260
