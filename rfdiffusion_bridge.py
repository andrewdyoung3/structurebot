"""
rfdiffusion_bridge.py
---------------------
RFdiffusion integration for StructureBot.

RFdiffusion (Watson et al., 2023) is a deep-learning backbone diffusion model
for *de novo* protein design.  It can:
  - Design binders to a target protein (hotspot-guided)
  - Scaffold functional motifs (e.g. enzyme active sites)
  - Generate symmetric oligomers (cyclic, dihedral, tetrahedral)
  - Partial-diffusion (diversify) an existing structure

Unlike ProteinMPNN (sequence design given a fixed backbone), RFdiffusion
generates *new backbones* and is therefore a substantially heavier compute step.

Execution model (Windows <-> WSL2 boundary)
-------------------------------------------
RFdiffusion needs Linux Python 3.9-3.11 + a CUDA torch + SE3-Transformer stack.
There is NO working Windows / Python-3.12 path, so the real backbone run ALWAYS
executes through ``wsl_bridge.RFDIFFUSION_PYTHON`` (the isolated WSL2
``~/rfdiffusion_env``) — never VENV312.  The single subtle point is the
filesystem boundary: ``inference.input_pdb`` / ``inference.output_prefix`` must be
paths WSL can see, and the generated PDBs must be collected from where WSL
actually wrote them.  We resolve this by pointing the output at a Windows cache
dir (``config.RFDIFFUSION_CACHE_DIR``) which is *also* WSL-visible via
``/mnt/c/...``: the bash command gets the ``/mnt/c`` form, and the Windows side
collects the PDBs straight out of the same directory (no copy-back).

Availability is reconciled the same way as ColabFold: either a Windows-visible
clone (``RFDIFFUSION_DIR`` containing ``run_inference.py`` + ``models/``) OR a WSL
probe for ``RFDIFFUSION_PYTHON`` + a WSL clone (mirror of
``WSLBridge.check_colabfold``).  On a machine with neither, ``is_available()`` is
False and ``analyze()`` returns the honest "not configured" error — it NEVER
fabricates a backbone or claims a run.

Installation (deferred — attended GPU-activation session)
---------------------------------------------------------
  # In WSL2 (Ubuntu), Python 3.9-3.11:
  git clone https://github.com/RosettaCommons/RFdiffusion ~/RFdiffusion
  cd ~/RFdiffusion && bash scripts/download_models.sh models/   # ~weights
  python -m venv ~/rfdiffusion_env && ~/rfdiffusion_env/bin/pip install -e .
  # + the SE3-Transformer dependency (see docs/rfdiffusion_activation_plan.md)

Interface
---------
analyze(inputs, session) -> ToolStepResult
  inputs keys:
    mode              : "binder" | "motif_scaffold" | "symmetric" | "partial_diffusion"
    pdb_path          : target structure (for binder/motif/partial modes)
    chain_id          : target chain (binder mode)
    hotspot_residues  : list of residue numbers on target to bind near
                        e.g. [82, 83, 84, 119, 120]
    binder_length     : length of the binder to grow (binder mode, default 100)
    num_designs       : number of backbone samples to generate (default 4)
    num_steps         : diffusion steps (default 50; more = better, slower)
    symmetry          : "C3" / "c3" / "D2" / "tetrahedral" ... (symmetric mode)
    partial_T         : noise level for partial-diffusion (0.0-1.0, default 0.2)
    contigs           : explicit contig string (overrides the NL-derived one)
                        e.g. "A1-10/20-30/A50-80"
    motif_residues    : residue numbers of the motif to keep (motif mode)

  Returns (when active):
    data["pdb_paths"]    : list of generated .pdb file paths
    data["mode"]         : the design mode used
    data["num_designs"]  : number of structures generated
    viz_commands         : ChimeraX commands to open all designs
    summary              : e.g. "RFdiffusion: 4 binder backbones generated"

Handoff
-------
``run_handoff`` feeds a generated backbone into the EXISTING ProteinMPNN ->
ColabFold path (it does not reimplement either): backbone PDB ->
``ProteinMPNNBridge.analyze`` -> ``ColabFoldBridge.predict`` on the top sequence.

References
----------
Watson JL, Juergens D, Bennett NR, et al. (2023).
"De novo design of protein structure and function with RFdiffusion."
Nature 620:1089-1100.  https://doi.org/10.1038/s41586-023-06415-8
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg
from tool_router import ToolStepResult
from wsl_bridge import WSLBridge, RFDIFFUSION_PYTHON

# ── Configuration ─────────────────────────────────────────────────────────────

_RFDIFFUSION_DIR: str = os.environ.get(
    "RFDIFFUSION_DIR",
    str(Path(_cfg.__file__).parent / "RFdiffusion"),
).strip()

# WSL2 clone path probed when there is no Windows-visible clone (mirror of the
# ColabFold env layout). The attended activation session clones it here.
_RFDIFFUSION_WSL_DIR: str = os.environ.get(
    "RFDIFFUSION_WSL_DIR", "/home/andre/RFdiffusion"
).strip()

_INSTALL_INSTRUCTIONS = """\
RFdiffusion is not yet configured.

RFdiffusion runs in an isolated WSL2 env (Linux Python 3.9-3.11 + CUDA torch +
SE3-Transformer) — there is no Windows path. To enable it:
  1. In WSL2 (Ubuntu), clone the repo:
       git clone https://github.com/RosettaCommons/RFdiffusion ~/RFdiffusion

  2. Download model weights (~20 GB, from files.ipd.uw.edu):
       cd ~/RFdiffusion && bash scripts/download_models.sh models/

  3. Create the env and install (+ SE3-Transformer):
       python3.10 -m venv ~/rfdiffusion_env
       ~/rfdiffusion_env/bin/pip install -e .

  4. (Optional) point RFDIFFUSION_DIR at a Windows-visible clone, or leave it
     unset to auto-probe the WSL2 env.

  5. Restart StructureBot.

See docs/rfdiffusion_activation_plan.md.
Reference: Watson et al. (2023) Nature 620:1089-1100
"""

# Named (non-cyclic/dihedral) RFdiffusion symmetries — passed through lowercased.
_NAMED_SYMMETRIES = {"tetrahedral", "octahedral", "icosahedral"}


# ── NL -> contig / hotspot / symmetry parsing (the substantive new logic) ────────
# RFdiffusion v1 (run_inference.py) is Hydra-configured. The design intent the
# translator/router hands us ("design an 80-residue binder to chain A hotspots
# 82,119"; "scaffold this motif"; "make a C3 trimer") must become the exact v1
# override strings, VERIFIED against the v1 README:
#   binder    : 'contigmap.contigs=[B1-100/0 100-100]'  + 'ppi.hotspot_res=[A30,A33]'
#   motif     : 'contigmap.contigs=[5-15/A10-25/30-40]'
#   symmetric : --config-name symmetry  inference.symmetry=c3|d2|tetrahedral ...
# We reuse the resnum/chain spine (proteinmpnn_bridge.chain_resnum_to_seqpos /
# _chain_positions_and_cys, selection.parse_selection_text) rather than
# re-parsing residues here.


def _chain_extent(pdb_path: str, chain: str) -> Optional[Tuple[int, int]]:
    """
    (min_resnum, max_resnum) of *chain* in *pdb_path*, or None if it can't be
    determined.  Reuses proteinmpnn_bridge's BioPython residue spine.
    """
    try:
        from proteinmpnn_bridge import _chain_positions_and_cys
        positions, _ = _chain_positions_and_cys(pdb_path, chain)
        if positions:
            return positions[0], positions[-1]
    except Exception:
        pass
    return None


def build_binder_contig(
    target_chain:   str,
    target_extent:  Optional[Tuple[int, int]],
    binder_length:  int,
) -> str:
    """
    Binder contig: keep the whole target chain, chain-break (``/0``), then grow a
    fresh binder of *binder_length* residues.  v1 form: ``B1-100/0 100-100``
    (verified against the README binder example).

    Returns "" when the target extent is unknown (caller then omits the contig
    override rather than emitting a wrong one — never guess the backbone).
    """
    if not target_extent:
        return ""
    lo, hi = target_extent
    L = max(1, int(binder_length))
    return f"{target_chain}{lo}-{hi}/0 {L}-{L}"


def build_motif_contig(
    motif_chain:    str,
    motif_residues: List[int],
    flank:          int = 10,
) -> str:
    """
    Motif-scaffolding contig: a flexible flank, the kept motif segment(s), and a
    trailing flank.  v1 form: ``5-15/A10-25/30-40`` (verified against the README).

    *flank* is the (min) number of scaffold residues padded on each side; v1
    accepts a fixed length here.  Contiguous motif residues collapse into one
    ``<chain><start>-<end>`` segment; gaps insert a scaffold spacer.
    """
    nums = sorted({int(r) for r in motif_residues})
    if not nums:
        return ""
    segments: List[str] = [f"{flank}-{flank}"]
    seg_start = nums[0]
    prev = nums[0]
    for r in nums[1:]:
        if r == prev + 1:
            prev = r
            continue
        segments.append(f"{motif_chain}{seg_start}-{prev}")
        segments.append(f"{flank}-{flank}")   # scaffold spacer across the gap
        seg_start = r
        prev = r
    segments.append(f"{motif_chain}{seg_start}-{prev}")
    segments.append(f"{flank}-{flank}")
    return "/".join(segments)


def normalize_symmetry(symmetry: str) -> str:
    """
    Canonicalise a symmetry spec to the exact token v1 expects (verified):
      cyclic    -> 'c<N>'   (e.g. "C3", "3-fold", "trimer" -> "c3")
      dihedral  -> 'd<N>'   (e.g. "D2" -> "d2")
      named     -> 'tetrahedral' | 'octahedral' | 'icosahedral' (lowercased)

    Returns "" when nothing recognisable is found (caller omits the override).
    """
    if not symmetry:
        return ""
    s = str(symmetry).strip().lower()
    if s in _NAMED_SYMMETRIES:
        return s
    m = re.match(r"^([cd])\s*(\d+)$", s)            # c3 / d2 / "c 3"
    if m:
        return f"{m.group(1)}{m.group(2)}"
    m = re.search(r"(\d+)\s*-?\s*fold", s)          # "3-fold" -> cyclic c3
    if m:
        return f"c{m.group(1)}"
    if s in _NAMED_SYMMETRIES:
        return s
    return ""


# ── Main class ────────────────────────────────────────────────────────────────

class RFdiffusionBridge:
    """
    RFdiffusion backbone diffusion bridge.

    There is ONE execution path: ``run_inference.py`` is ALWAYS dispatched through
    ``wsl.exe`` -> ``RFDIFFUSION_PYTHON`` (RFdiffusion has no native-Windows
    interpreter, and VENV312 is not it).  Availability only records WHERE the
    clone lives, which determines the script/path translation — not how it runs:
      * ``win_clone`` — a Windows-visible clone (RFDIFFUSION_DIR with
        run_inference.py + models/).  Its ``C:\\`` script + path overrides are
        translated to ``/mnt/c`` and run via WSL2 like any other.  (fs-based
        availability check; the WSL env is validated at dispatch, error-first.)
      * ``wsl_clone`` — a WSL-native clone, discovered by probing
        ``RFDIFFUSION_PYTHON`` + the clone via WSLBridge (mirror of
        ``check_colabfold``).

    When neither is present (true on this machine): ``is_available()`` is False
    and ``analyze()`` returns the honest "not configured" error.  All compute is
    delegated to the isolated RFdiffusion env — never the main runtime.
    """

    def __init__(self) -> None:
        self._dir:       Optional[Path] = Path(_RFDIFFUSION_DIR) if _RFDIFFUSION_DIR else None
        self._script:    Optional[Path] = None
        self._backend:   Optional[str]  = None      # "win_clone" | "wsl_clone" | None
        self._wsl_dir:   str            = _RFDIFFUSION_WSL_DIR
        self._wsl:       WSLBridge      = WSLBridge()
        self._available: bool = self._check_available()

    def _check_available(self) -> bool:
        """
        Return True if RFdiffusion is runnable, by EITHER route. Sets
        ``self._backend`` (+ ``self._script`` for the Windows route) on success.

        Execution is identical for both (always WSL2); this only records the
        clone location for path translation at dispatch.  Resolution order keeps
        the unit tests deterministic and hermetic while letting the real WSL env
        be discovered:
          1. RFDIFFUSION_DIR unset/empty            -> not configured (no probe).
          2. RFDIFFUSION_DIR is an existing dir     -> fs check only (no WSL
             probe): run_inference.py (root or scripts/) + models/ -> win_clone.
          3. RFDIFFUSION_DIR set but missing on the Windows fs -> probe the WSL2
             env (RFDIFFUSION_PYTHON + a WSL clone), mirror of check_colabfold.
        """
        # (1) unset/empty -> not configured.
        if not self._dir:
            return False

        # (2) Windows-visible clone present -> fs check (it runs via WSL anyway,
        #     with C:\ paths translated to /mnt/c; the WSL env is validated at
        #     dispatch and surfaced error-first if missing).
        if self._dir.is_dir():
            script = self._find_script(self._dir)
            if script and (self._dir / "models").is_dir():
                self._script = script
                self._backend = "win_clone"
                return True
            return False

        # (3) No Windows clone -> probe the isolated WSL2 env.
        if self._wsl.is_available() and self._wsl.check_rfdiffusion(self._wsl_dir):
            self._backend = "wsl_clone"
            return True
        return False

    @staticmethod
    def _find_script(root: Path) -> Optional[Path]:
        """run_inference.py at the repo root OR under scripts/ (v1 varies)."""
        for cand in (root / "run_inference.py", root / "scripts" / "run_inference.py"):
            if cand.is_file():
                return cand
        return None

    # ── Availability (public, ColabFold parity) ─────────────────────────────────

    def is_available(self) -> bool:
        """True iff installed (route detected) AND the WSL rfdiffusion_env import
        chain RUNS.

        Tier 1 = `self._available` (route detection: Windows clone fs-check OR WSL
        env probe). Tier 2 = a cached WSL import probe (torch + dgl in
        RFDIFFUSION_PYTHON) — the cavity-class capability check: the Windows-clone
        route's fs-check confirms FILES exist but not that the WSL env can import
        (dgl/sm_120 is the real break risk). dep_probe caches a DEFINITIVE verdict;
        WSL-down / spawn error → False without caching. Run path is unaffected (it
        does not gate on this).
        """
        if not self._available:
            return False
        from dep_probe import wsl_import_probe
        return wsl_import_probe(
            self._wsl, RFDIFFUSION_PYTHON, ["import torch", "import dgl"],
            timeout=int(getattr(_cfg, "RFDIFFUSION_PROBE_TIMEOUT", 90)),
            cache_key=("rfdiffusion", RFDIFFUSION_PYTHON, self._wsl_dir))

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        inputs:  Dict[str, Any],
        session: Any = None,
    ) -> ToolStepResult:
        """
        Run RFdiffusion backbone generation.

        If not yet configured, returns a helpful "not configured" error (it never
        fabricates a backbone or claims a run).  When active, delegates to the
        appropriate backend.
        """
        if not self._available:
            return ToolStepResult(
                tool    = "rfdiffusion",
                success = False,
                error   = _INSTALL_INSTRUCTIONS,
            )

        mode = inputs.get("mode", "binder").lower()
        pdb_path = inputs.get("pdb_path", "")

        if mode in ("binder", "motif_scaffold", "partial_diffusion") and (
            not pdb_path or not Path(pdb_path).is_file()
        ):
            return ToolStepResult(
                tool    = "rfdiffusion",
                success = False,
                error   = (
                    f"RFdiffusion mode '{mode}' requires a local PDB file.\n"
                    "  Provide pdb_path in tool_inputs, or load the structure first."
                ),
            )

        try:
            return self._run_inference(inputs, session)
        except Exception as exc:
            return ToolStepResult(
                tool    = "rfdiffusion",
                success = False,
                error   = f"RFdiffusion inference failed: {exc}",
            )

    def status(self) -> str:
        """Return a one-line status string for display."""
        if self._available and self._backend == "win_clone":
            return f"rfdiffusion — WSL2 {RFDIFFUSION_PYTHON} (win clone {self._dir})"
        if self._available and self._backend == "wsl_clone":
            return f"rfdiffusion — WSL2 {RFDIFFUSION_PYTHON} ({self._wsl_dir})"
        if self._dir and self._dir.is_dir():
            return f"directory found ({self._dir}) but run_inference.py/models/ missing"
        return "not configured (set RFDIFFUSION_DIR in .env.local, or build ~/rfdiffusion_env)"

    # ── Intent resolution (NL/inputs -> concrete v1 overrides) ───────────────────

    def _resolve_spec(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Turn the routed inputs into concrete contig/hotspot/symmetry strings.

        Honours an explicit ``contigs`` override; otherwise derives one per mode
        from the design intent, reusing the residue/chain spine. Never raises —
        an underivable contig resolves to "" and the override is simply omitted.
        """
        mode        = inputs.get("mode", "binder").lower()
        pdb_path    = inputs.get("pdb_path", "")
        chain_id    = inputs.get("chain_id", "A")
        hotspots    = [int(r) for r in (inputs.get("hotspot_residues") or [])]
        contigs     = str(inputs.get("contigs", "")).strip()
        symmetry    = ""

        if mode == "binder" and not contigs:
            extent = _chain_extent(pdb_path, chain_id) if pdb_path else None
            contigs = build_binder_contig(
                chain_id, extent, int(inputs.get("binder_length", 100))
            )
        elif mode == "motif_scaffold" and not contigs:
            contigs = build_motif_contig(
                inputs.get("motif_chain", chain_id),
                [int(r) for r in (inputs.get("motif_residues") or [])],
                int(inputs.get("flank", 10)),
            )
        elif mode == "symmetric":
            symmetry = normalize_symmetry(inputs.get("symmetry", ""))

        return {
            "mode":        mode,
            "pdb_path":    pdb_path,
            "chain_id":    chain_id,
            "hotspots":    hotspots,
            "num_designs": int(inputs.get("num_designs", 4)),
            "num_steps":   int(inputs.get("num_steps", 50)),
            "symmetry":    symmetry or str(inputs.get("symmetry", "")),
            "partial_T":   float(inputs.get("partial_T", 0.2)),
            "contigs":     contigs,
        }

    # ── Internal inference ─────────────────────────────────────────────────────

    def _run_inference(
        self,
        inputs:  Dict[str, Any],
        session: Any,
    ) -> ToolStepResult:
        """Resolve intent, check the cache, run the configured backend, collect PDBs."""
        import time
        t0 = time.perf_counter()

        spec = self._resolve_spec(inputs)

        # Content-hash cache (mirror of the ColabFold fold cache). A re-run of an
        # identical request returns the cached PDB set without recomputing.
        cache_key = self._cache_key(spec)
        run_dir   = Path(_cfg.RFDIFFUSION_CACHE_DIR) / f"rfd_{cache_key}"
        cached = self._load_cache(run_dir, spec["mode"])
        if cached is not None:
            return cached

        run_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = run_dir / spec["mode"]

        cmd = self._build_cmd(
            mode=spec["mode"], pdb_path=spec["pdb_path"], chain_id=spec["chain_id"],
            hotspots=spec["hotspots"], num_designs=spec["num_designs"],
            num_steps=spec["num_steps"], symmetry=spec["symmetry"],
            partial_T=spec["partial_T"], contigs=spec["contigs"],
            out_path=out_prefix,
        )

        # Single execution path: run_inference.py ALWAYS runs through wsl.exe ->
        # RFDIFFUSION_PYTHON. A win_clone is dispatched identically, with its C:\
        # script + path overrides translated to /mnt/c. _dispatch raises on a
        # failed run; analyze() turns that into an error-first result.
        self._dispatch(cmd, run_dir)

        pdb_files = sorted(run_dir.rglob("*.pdb"))
        pdb_paths = [str(p) for p in pdb_files]
        if not pdb_paths:
            return ToolStepResult(
                tool    = "rfdiffusion",
                success = False,
                error   = (
                    "RFdiffusion produced no .pdb output in "
                    f"{run_dir} (no backbone was generated)."
                ),
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = self._build_result(spec, pdb_paths, elapsed_ms, cached=False)
        self._save_cache(run_dir, spec, pdb_paths)
        return result

    def _dispatch(self, cmd: List[str], run_dir: Path) -> None:
        """
        Execute run_inference.py inside WSL2 via RFDIFFUSION_PYTHON — the single
        execution path for BOTH clone locations.  A win_clone has its ``C:\\``
        script + path overrides translated to ``/mnt/c``; a wsl_clone is already
        WSL-addressable.  Output lands in *run_dir* (a ``/mnt/c``-visible Windows
        cache dir), so the Windows side collects the PDBs with no copy-back.

        Single-quotes every Hydra override for bash (contig strings contain '['
        and a space).  Raises RuntimeError on a failed / non-zero run — surfaced
        by analyze() as an error-first result, never a fabricated success.
        """
        # cmd == [RFDIFFUSION_PYTHON, <script>, *hydra_tokens]
        wsl_script = self._wsl.translate_path(cmd[1])
        repo_dir = (
            self._wsl.translate_path(str(self._dir))
            if self._backend == "win_clone" and self._dir else self._wsl_dir
        )
        parts: List[str] = [RFDIFFUSION_PYTHON, shlex.quote(wsl_script)]
        for tok in cmd[2:]:
            parts.append(shlex.quote(self._wslify_token(tok)))
        bash_cmd = f"cd {shlex.quote(repo_dir)} && " + " ".join(parts)

        timeout = int(getattr(_cfg, "RFDIFFUSION_TIMEOUT", 3600))
        run = self._wsl.run_command(bash_cmd, timeout=timeout)
        if not run.get("ok"):
            why = (run.get("error") or "").strip() or str(run.get("stderr", ""))[:400]
            raise RuntimeError(f"RFdiffusion WSL2 run failed: {why}")

    def _wslify_token(self, tok: str) -> str:
        """
        Translate a Windows path inside a ``key=value`` Hydra override to /mnt/c.
        Non-path tokens (and value parts that aren't Windows paths) pass through
        unchanged.
        """
        if "=" not in tok:
            return tok
        key, value = tok.split("=", 1)
        if re.match(r"^[A-Za-z]:[\\/]", value) or "\\" in value:
            value = self._wsl.translate_path(value)
        return f"{key}={value}"

    # ── Result + viz ─────────────────────────────────────────────────────────────

    def _build_result(
        self,
        spec:       Dict[str, Any],
        pdb_paths:  List[str],
        elapsed_ms: float,
        cached:     bool,
    ) -> ToolStepResult:
        viz_cmds: List[str] = []
        viz_exps: List[str] = []
        for i, p in enumerate(pdb_paths, 1):
            viz_cmds.append(f"open {p!r}")
            viz_exps.append(f"Open RFdiffusion design {i}: {Path(p).stem}")
        if pdb_paths:
            viz_cmds.append("rainbow models")
            viz_exps.append("Rainbow-color each design for visual distinction")

        src = " (cached)" if cached else ""
        summary = (
            f"RFdiffusion ({spec['mode']}): {len(pdb_paths)} backbone(s) generated"
            f"{src} in {elapsed_ms/1000:.1f}s."
        )
        return ToolStepResult(
            tool             = "rfdiffusion",
            success          = True,
            data             = {
                "pdb_paths":   pdb_paths,
                "mode":        spec["mode"],
                "num_designs": len(pdb_paths),
                "cached":      cached,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    # ── Cache helpers (mirror ColabFoldBridge) ──────────────────────────────────

    def _cache_key(self, spec: Dict[str, Any]) -> str:
        """md5 of the resolved request (+ input-PDB content hash) -> 16 hex chars."""
        pdb_tag = ""
        p = spec.get("pdb_path") or ""
        if p and Path(p).is_file():
            try:
                pdb_tag = hashlib.md5(Path(p).read_bytes()).hexdigest()[:8]
            except Exception:
                pdb_tag = ""
        raw = (
            f"{spec['mode']}|{spec['chain_id']}|{pdb_tag}|"
            f"hs{','.join(map(str, spec['hotspots']))}|"
            f"n{spec['num_designs']}|t{spec['num_steps']}|"
            f"sym{spec['symmetry']}|pt{spec['partial_T']}|c{spec['contigs']}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _load_cache(self, run_dir: Path, mode: str) -> Optional[ToolStepResult]:
        """Return a cached ToolStepResult if a previous run's PDBs still exist."""
        if not run_dir.is_dir():
            return None
        pdb_files = sorted(run_dir.rglob("*.pdb"))
        if not pdb_files:
            return None
        spec = {"mode": mode}
        return self._build_result(spec, [str(p) for p in pdb_files], 0.0, cached=True)

    @staticmethod
    def _save_cache(run_dir: Path, spec: Dict[str, Any], pdb_paths: List[str]) -> None:
        try:
            meta = {
                "mode": spec["mode"], "num_designs": len(pdb_paths),
                "pdb_paths": pdb_paths,
            }
            (run_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── Handoff: backbone -> EXISTING ProteinMPNN -> EXISTING ColabFold ──────────

    def run_handoff(
        self,
        backbone_pdb:  str,
        chain_id:      str = "A",
        num_sequences: int = 8,
        session:       Any = None,
        fold:          bool = True,
    ) -> Dict[str, Any]:
        """
        Feed a generated backbone into the EXISTING design->validate path.

        backbone PDB -> ProteinMPNNBridge.analyze (sequence design on the new
        backbone) -> ColabFoldBridge.predict (fold the top sequence).  Reuses
        both bridges as-is; this method only wires inputs and is error-first
        (never raises, never fabricates).
        """
        if not backbone_pdb or not Path(backbone_pdb).is_file():
            return {"success": False, "stage": "input",
                    "error": f"backbone PDB not found: {backbone_pdb!r}"}

        try:
            from proteinmpnn_bridge import ProteinMPNNBridge
        except Exception as exc:
            return {"success": False, "stage": "proteinmpnn",
                    "error": f"could not import ProteinMPNN bridge: {exc}"}

        mpnn = ProteinMPNNBridge()
        mpnn_res = mpnn.analyze(
            {"pdb_path": backbone_pdb, "chain_id": chain_id,
             "num_sequences": int(num_sequences)},
            session=session,
        )
        if not getattr(mpnn_res, "success", False):
            return {"success": False, "stage": "proteinmpnn",
                    "error": getattr(mpnn_res, "error", "ProteinMPNN failed")}

        sequences = (mpnn_res.data or {}).get("sequences", [])
        if not sequences:
            return {"success": False, "stage": "proteinmpnn",
                    "error": "ProteinMPNN returned no sequences", "mpnn": mpnn_res.data}
        top_seq = sequences[0].get("sequence", "")

        out: Dict[str, Any] = {"success": True, "stage": "proteinmpnn",
                               "mpnn": mpnn_res.data, "top_sequence": top_seq}
        if not fold:
            return out

        try:
            from colabfold_bridge import ColabFoldBridge
        except Exception as exc:
            out.update({"stage": "colabfold_import",
                        "error": f"could not import ColabFold bridge: {exc}"})
            return out

        fold_res = ColabFoldBridge().predict(top_seq, label="rfd_handoff")
        out["fold"] = fold_res
        out["stage"] = "colabfold"
        out["success"] = bool(fold_res.get("success"))
        if not fold_res.get("success"):
            out["error"] = fold_res.get("error", "ColabFold fold failed")
        return out

    # ── Command construction (Hydra overrides; flags verified vs v1 README) ──────

    def _build_cmd(
        self,
        mode: str,
        pdb_path: str,
        chain_id: str,
        hotspots: List[int],
        num_designs: int,
        num_steps: int,
        symmetry: str,
        partial_T: float,
        contigs: str,
        out_path: Path,
    ) -> List[str]:
        """
        Build the run_inference.py command list for the requested mode.

        The interpreter is ALWAYS RFDIFFUSION_PYTHON (the WSL env) — never
        VENV312; RFdiffusion has no Windows build.  The path-bearing tokens are
        translated to /mnt/c at WSL-invocation time (_dispatch); here they carry
        the (possibly Windows) form so a win_clone / the mocked tests see real
        paths.  Override keys are the exact v1 names:
          inference.input_pdb / inference.output_prefix / inference.num_designs
          diffuser.T / ppi.hotspot_res / contigmap.contigs / diffuser.partial_T
          inference.symmetry (with --config-name symmetry)
        """
        script = str(self._script) if self._script else f"{self._wsl_dir}/run_inference.py"
        cmd = [RFDIFFUSION_PYTHON, script]

        if mode == "binder":
            hs_str = ",".join(f"{chain_id}{r}" for r in hotspots)
            cmd += [
                f"inference.input_pdb={pdb_path}",
                f"inference.num_designs={num_designs}",
                f"diffuser.T={num_steps}",
                f"inference.output_prefix={out_path}",
                f"ppi.hotspot_res=[{hs_str}]",
            ]
            if contigs:
                # v1 binder design REQUIRES a contig (target range /0 binder len).
                cmd.append(f"contigmap.contigs=[{contigs}]")

        elif mode == "motif_scaffold":
            cmd += [
                f"inference.input_pdb={pdb_path}",
                f"inference.num_designs={num_designs}",
                f"diffuser.T={num_steps}",
                f"inference.output_prefix={out_path}",
                f"contigmap.contigs=[{contigs}]",
            ]

        elif mode == "symmetric":
            # v1 symmetric oligomers require the dedicated config group.
            cmd += [
                "--config-name", "symmetry",
                f"inference.symmetry={symmetry}",
                f"inference.num_designs={num_designs}",
                f"diffuser.T={num_steps}",
                f"inference.output_prefix={out_path}",
            ]
            if contigs:
                cmd.append(f"contigmap.contigs=[{contigs}]")

        elif mode == "partial_diffusion":
            cmd += [
                f"inference.input_pdb={pdb_path}",
                f"inference.num_designs={num_designs}",
                f"diffuser.partial_T={int(partial_T * num_steps)}",
                f"inference.output_prefix={out_path}",
            ]

        else:
            raise ValueError(
                f"Unknown RFdiffusion mode '{mode}'. "
                "Valid modes: binder, motif_scaffold, symmetric, partial_diffusion"
            )

        return cmd

    def __repr__(self) -> str:
        return f"<RFdiffusionBridge available={self._available} backend={self._backend}>"
