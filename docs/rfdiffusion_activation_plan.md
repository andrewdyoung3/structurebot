# RFdiffusion Activation Plan (✅ EXECUTED 2026-06-12)

> **✅ EXECUTED 2026-06-12 — RFdiffusion is ACTIVATED on the RTX 5070 Ti (Blackwell `sm_120`).**
> `~/rfdiffusion_env` is built and the live smoke passes; `is_available()` is now True.
> **Resolved stack (native WSL2 venv, NOT Docker):** deadsnakes **py3.11** · **torch 2.11.0+cu128**
> (`get_device_capability()==(12,0)` confirmed) · **dgl 2.5 built from source** (pinned `dmlc/dgl@3d16000`)
> with **real sm_120 kernels** · **SE3Transformer UNMODIFIED**. Only two adaptations: e3nn
> `o3/_wigner.py` → `torch.load(..., weights_only=False)`, and `numpy<2`.
>
> **The dgl long-pole, resolved (the crux this plan flagged as unresolved):** dgl does **not** read
> `TORCH_CUDA_ARCH_LIST` or `CMAKE_CUDA_ARCHITECTURES`. Its arch lever is the legacy
> `dgl_select_nvcc_arch_flags` path: **`-DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=120 -DCUDA_ARCH_PTX=120`**
> (drives libdgl `-gencode` *and* graphbolt `CUDAARCHS`). sm_120 is absent from dgl's
> `dgl_known_gpu_archs` default, so omitting this flag silently builds ≤sm_90 → runtime "no kernel
> image." The §2 routes (CPU dgl / conda dgl) were both wrong: the working route is a **CUDA source
> build of dgl 2.5 against torch 2.11** with that flag — the SE3Transformer↔dgl "API chasm" feared
> below did **not** materialize (it installs unmodified; the only patches are the e3nn + numpy ones).
>
> **Recipe basis:** the `JMB-Scripts/RFdiffusion-dockerfile-nvidia-RTX5090` dependency closure,
> replayed as a native WSL venv so the existing `RFDIFFUSION_PYTHON` bridge contract holds with **zero
> bridge change** — *with the arch flag JMB omitted* (which is why JMB reports no working smoke).
> **Build gotchas:** the dgl CUDA compile OOMs at high `-j` → WSL `.wslconfig` 22 GB + 8 GB swap and
> cap **all** parallelism at `-j3` (main ninja + the bare `make -j` in `graphbolt`/`dgl_sparse`).
> **Verified:** arch pre-check → dgl GPU micro-smoke (`dgl.ops.copy_e_sum`/`edge_softmax` on CUDA) →
> Block-E smoke (`contigmap.contigs=[100-100]` → valid 100-res backbone, 50 timesteps) → full
> RFdiffusion→ProteinMPNN→ColabFold handoff. See PROJECT_CONTEXT §8/§9/§13. The forward-looking
> §2/§5 below is retained as the historical investigation; the resolved facts above supersede it.

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

## 5. Install recipe (AS-BUILT — executed 2026-06-12)

This is the recipe that actually built the working env (py3.11, not the original 3.10 guess).
**Machine prep (durable):** the dgl CUDA compile OOMs on WSL's default ~15.5 GB — set
`C:\Users\andre\.wslconfig` to `[wsl2]\nmemory=22GB\nswap=8GB` then `wsl --shutdown` first.

```bash
# 1. Toolchain (sudo). py3.11 + FULL CUDA 12.8 toolkit (nvcc, to compile dgl for sm_120) + gcc/g++-11.
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
    build-essential gcc-11 g++-11 git wget curl ca-certificates gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8

# 2. venv + torch (confirm sm_120: get_device_capability()==(12,0))
python3.11 -m venv ~/rfdiffusion_env && source ~/rfdiffusion_env/bin/activate
pip install -U pip wheel setuptools "cmake>=3.29,<4" ninja   # cmake <4: dgl uses pre-4.0 CMake policies
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 3. dgl 2.5 FROM SOURCE with sm_120 kernels (the long-pole; >1h compile, OOM-prone -> -j3 everywhere)
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64 CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 DGL_HOME=$HOME/dgl
git clone --recursive https://github.com/dmlc/dgl.git ~/dgl && cd ~/dgl
git checkout 3d16000b4170fa741ed9e9667f22ba84d3493026 && git submodule update --init --recursive
sed -i -E 's/make -j([^0-9]|$)/make -j3\1/g' graphbolt/build.sh dgl_sparse/build.sh  # else 32 jobs -> OOM
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
    -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=120 -DCUDA_ARCH_PTX=120 ..   # <-- THE FIX (see top banner)
ninja -j3 && cd ../python && pip install .
# verify sm_120 kernels run (no "no kernel image"):
python -c "import torch,dgl; dgl.ops.copy_e_sum(dgl.graph(([0],[1])).to('cuda'), torch.ones(1,4,device='cuda')); print('dgl sm_120 OK')"

# 4. SE3Transformer (UNMODIFIED) + RFdiffusion + the runtime deps its setup.py does NOT declare
cd ~/RFdiffusion/env/SE3Transformer && pip install -r requirements.txt && python setup.py install
pip install git+https://github.com/NVIDIA/dllogger.git
cd ~/RFdiffusion && pip install -e .
pip install "hydra-core==1.3.2" "pyrsistent>=0.19.3" pandas "pydantic>=2.0" wandb pynvml torchdata decorator gitpython  # NOT e3nn (keep SE3's 0.3.3)

# 5. The two (and only two) adaptations
WIG=$(python -c "import e3nn,os;print(os.path.join(os.path.dirname(e3nn.__file__),'o3','_wigner.py'))")
sed -i "s/'constants.pt'))/'constants.pt'), weights_only=False)/" "$WIG"   # torch>=2.6 weights_only default
pip install "numpy<2"

# 6. Weights (Base_ckpt only suffices for the smoke + monomer/unconditional; binder/complex need the full
#    7-ckpt set via scripts/download_models.sh — deferred, see PROJECT_CONTEXT §9). NOTE: the URL path
#    segment is NOT the file md5 (no upstream checksum exists); verify by byte-size = 483616107.
cd ~/RFdiffusion && mkdir -p models && wget -O models/Base_ckpt.pt \
    http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt

# 7. Smoke (proves sm_120 + weights -> a real backbone)
python scripts/run_inference.py 'contigmap.contigs=[100-100]' \
    inference.output_prefix=$HOME/rfd_smoke/mono inference.num_designs=1
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
