"""
boltz_bridge.py
---------------
Boltz-2 as a LOCAL-ONLY multimer/assembly (and high-quality monomer) fold engine for the
Variant Workbench — the second engine on the S4b engine-agnostic fold seam (ESMFold is the
first). Plugged into that seam, NOT a parallel path: `tool_router._run_boltz` consumes this
bridge's result and feeds the SAME `_fold_viz_commands` + `variant_model.fold_summary`.

Design (mirrors rasp_bridge's WSL-subprocess pattern + esmfold_bridge's LOCAL-ONLY gate):
  - EXECUTION: a WSL2 subprocess (BOLTZ_PYTHON in the dedicated ~/boltz_env — NOT venv312,
    whose cu128 torch ESM/ESMFold/ThermoMPNN need; Boltz ships torch cu13). Reuses WSLBridge
    for availability, path translation, and the subprocess. The probe's proven verbatim CLI:
      boltz predict <yaml> --out_dir <out> --accelerator gpu --no_kernels --override --seed <s>
  - LOCAL-ONLY, FAIL-CLOSED: Boltz DEFAULTS to the REMOTE ColabFold MSA server (the breach,
    the analog of esmfold's Atlas fallback / the §0 invariant). We fold MSA-free (`msa: empty`
    per chain) and NEVER pass --use_msa_server. `_assert_local_only` runs right before exec and
    REFUSES (no subprocess) if the built command carries --use_msa_server/--msa_server_url OR any
    chain's YAML msa is not exactly `empty`. Correct construction is not trusted on its own.
  - SEED-PINNED: Boltz is diffusion-stochastic; a fixed --seed makes a re-fold reproducible
    (the S4b CA-drift≈0 bar) so S4c's variant-vs-WT deviation isn't confounded by sampling noise.
  - CAPABILITY FLAG (Unit-B): is_available() probes the ~/boltz_env import chain WHERE IT RUNS
    (dep_probe.wsl_import_probe), B2 3-state, definitive-only cache — enables the picker's Boltz.
  - RESULT contract: {success, cif_path, mean_plddt, iptm, chains_ptm, plddt(rep-chain per
    author-index), source:"local_boltz_env", seed, error}. ERROR-FIRST + GRACEFUL throughout.
"""
from __future__ import annotations

import glob
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg

_SOURCE = "local_boltz_env"

# Module-level availability cache (definitive verdict only), keyed by (distro, python).
_AVAIL_CACHE: Dict[Tuple[str, str], bool] = {}


class BoltzBridge:
    """LOCAL-ONLY Boltz-2 fold (monomer or assembly), seed-pinned, MSA-free."""

    def __init__(self) -> None:
        self._python  = str(getattr(_cfg, "BOLTZ_PYTHON", "/home/andre/boltz_env/bin/python"))
        # The `boltz` console entry-point lives beside the interpreter; `python -m boltz`
        # does NOT work (boltz has no __main__). POSIX string op (this is a WSL path).
        self._exe     = self._python.rsplit("/", 1)[0] + "/boltz"
        self._distro  = str(getattr(_cfg, "BOLTZ_WSL_DISTRO", "Ubuntu-24.04"))
        self._enable  = str(getattr(_cfg, "BOLTZ_ENABLE", "auto")).strip().lower()
        self._seed    = int(getattr(_cfg, "BOLTZ_SEED", 0))
        self._recycle = int(getattr(_cfg, "BOLTZ_RECYCLING_STEPS", 3))
        self._sampling = int(getattr(_cfg, "BOLTZ_SAMPLING_STEPS", 200))
        self._samples = int(getattr(_cfg, "BOLTZ_DIFFUSION_SAMPLES", 1))
        self._timeout = int(getattr(_cfg, "BOLTZ_TIMEOUT", 1800))
        from wsl_bridge import WSLBridge
        self._wsl = WSLBridge(distribution=self._distro)

    # ── Availability (Unit-B capability flag) ───────────────────────────────────
    def is_available(self) -> bool:
        if self._enable in ("false", "0", "no", "off"):
            return False
        ck = (self._distro, self._python)
        if ck in _AVAIL_CACHE:
            return _AVAIL_CACHE[ck]
        if not self._wsl.is_available():
            return False                       # WSL down — transient, do NOT cache
        from dep_probe import wsl_import_probe
        ok = wsl_import_probe(
            self._wsl, self._python, ["import boltz", "import torch"],
            timeout=int(getattr(_cfg, "BOLTZ_PROBE_TIMEOUT", 90)),
            cache_key=("boltz", self._distro, self._python))
        if ok:
            _AVAIL_CACHE[ck] = True             # cache only definitive success
        return ok

    def status(self) -> str:
        if self._enable in ("false", "0", "no", "off"):
            return "disabled (BOLTZ_ENABLE)"
        if not self._wsl.is_available():
            return "WSL2 unavailable"
        return "available" if self.is_available() else "boltz_env not found / import chain failed in WSL"

    # ── Fold ────────────────────────────────────────────────────────────────────
    def predict(
        self,
        chains:       List[Dict[str, str]],     # [{"id": "A", "sequence": "MK..."}, ...]
        *,
        seed:         Optional[int] = None,
        allow_remote: bool = False,
        label:        str = "boltz",
    ) -> Dict[str, Any]:
        """Fold *chains* (one `protein` block each, MSA-free) with Boltz via WSL. Returns the
        result contract (success/cif_path/mean_plddt/iptm/chains_ptm/plddt/source/seed/error).
        Never raises for an expected failure — returns {success: False, error: ...}."""
        if not chains:
            return self._err(label, "no chains given to fold")
        seed = self._seed if seed is None else int(seed)

        yaml_text = self._build_yaml(chains)
        # Workspace on the WINDOWS side so outputs are readable back without a copy roundtrip.
        work = tempfile.mkdtemp(prefix="boltz_")
        yaml_path = os.path.join(work, "boltz_in.yaml")
        out_dir   = os.path.join(work, "out")
        os.makedirs(out_dir, exist_ok=True)
        with open(yaml_path, "w") as fh:
            fh.write(yaml_text)

        wsl = self._wsl
        yaml_wsl = wsl.translate_path(os.path.abspath(yaml_path))
        out_wsl  = wsl.translate_path(os.path.abspath(out_dir))
        cmd = (f"{shlex.quote(self._exe)} predict {shlex.quote(yaml_wsl)} "
               f"--out_dir {shlex.quote(out_wsl)} --accelerator gpu --no_kernels --override "
               f"--seed {int(seed)} --recycling_steps {self._recycle} "
               f"--sampling_steps {self._sampling} --diffusion_samples {self._samples}")

        # FAIL-CLOSED LOCAL-ONLY guard — refuse to run if the breach surface is present.
        try:
            self._assert_local_only(cmd, yaml_text, allow_remote)
        except _RemoteBreach as exc:
            return self._err(label, f"LOCAL-ONLY breach refused: {exc}")

        r = wsl.run_command(cmd, timeout=self._timeout)
        if not r.get("ok"):
            return self._err(label, f"Boltz run failed: {r.get('error') or (r.get('stderr','') or '')[-200:]}")

        parsed = self._parse_outputs(out_dir, chains)
        if not parsed.get("success"):
            return self._err(label, parsed.get("error", "Boltz produced no parsable output"))
        parsed.update(source=_SOURCE, seed=int(seed), label=label)
        return parsed

    # ── LOCAL-ONLY guard (fail-closed) ──────────────────────────────────────────
    @staticmethod
    def _assert_local_only(cmd: str, yaml_text: str, allow_remote: bool) -> None:
        """Refuse (raise) unless the fold is provably MSA-free + offline. Boltz defaults to
        the remote MSA server, so this is the breach backstop, not a courtesy check."""
        if allow_remote:
            return                              # never used by the workbench; explicit opt-in only
        low = cmd.lower()
        if "--use_msa_server" in low or "--msa_server_url" in low:
            raise _RemoteBreach("command requests the remote MSA server")
        # Every protein chain must declare `msa: empty` (no remote MSA, no local a3m path).
        msa_decls = [ln.strip() for ln in yaml_text.splitlines() if ln.strip().startswith("msa:")]
        if not msa_decls:
            raise _RemoteBreach("YAML declares no `msa:` — Boltz would auto-MSA via the remote server")
        for d in msa_decls:
            val = d.split(":", 1)[1].strip()
            if val != "empty":
                raise _RemoteBreach(f"YAML msa is '{val}', not 'empty' (would hit the remote server)")

    # ── YAML + output parsing ───────────────────────────────────────────────────
    @staticmethod
    def _build_yaml(chains: List[Dict[str, str]]) -> str:
        lines = ["version: 1", "sequences:"]
        for c in chains:
            lines += [
                "  - protein:",
                f"      id: {c['id']}",
                f"      sequence: {c['sequence']}",
                "      msa: empty",
            ]
        return "\n".join(lines) + "\n"

    def _parse_outputs(self, out_dir: str, chains: List[Dict[str, str]]) -> Dict[str, Any]:
        cifs = sorted(glob.glob(os.path.join(out_dir, "**", "*_model_0.cif"), recursive=True)) \
            or sorted(glob.glob(os.path.join(out_dir, "**", "*.cif"), recursive=True))
        confs = sorted(glob.glob(os.path.join(out_dir, "**", "confidence_*_model_0.json"), recursive=True)) \
            or sorted(glob.glob(os.path.join(out_dir, "**", "confidence_*.json"), recursive=True))
        if not cifs:
            return {"success": False, "error": "no predicted CIF in the Boltz output"}
        cif_src = cifs[0]
        # Copy the CIF to a stable temp so the workspace can be cleaned and ChimeraX can open it.
        cif_dst = tempfile.NamedTemporaryFile(suffix=".cif", prefix="boltz_pred_", delete=False)
        cif_dst.close()
        shutil.copyfile(cif_src, cif_dst.name)

        conf: Dict[str, Any] = {}
        if confs:
            try:
                with open(confs[0]) as fh:
                    conf = json.load(fh)
            except Exception:
                conf = {}
        # complex_plddt is 0–1 in the JSON; the CIF B-factor is already 0–100.
        cplx = conf.get("complex_plddt")
        rep_chain = chains[0]["id"]
        plddt = self._cif_bfactor_by_index(cif_src, rep_chain)   # {1-based author index: pLDDT}
        mean_plddt = (float(cplx) * 100.0 if isinstance(cplx, (int, float))
                      else (round(sum(plddt.values()) / len(plddt), 2) if plddt else 0.0))
        return {
            "success":    True,
            "cif_path":   cif_dst.name,
            "mean_plddt": round(mean_plddt, 2),
            "iptm":       conf.get("iptm"),
            "ptm":        conf.get("ptm"),
            "chains_ptm": conf.get("chains_ptm"),
            "plddt":      plddt,
            "length":     len(plddt),
        }

    @staticmethod
    def _cif_bfactor_by_index(cif_path: str, chain: str) -> Dict[int, float]:
        """Per-residue pLDDT (CIF B-factor, 0–100) for *chain*'s CA atoms, keyed 1..N over the
        chain's residue order — matches `fold_summary`'s 1-based→author-resnum remap. Parses the
        `_atom_site` loop header for robust column lookup (mmCIF column order is not fixed)."""
        cols: List[str] = []
        in_loop = False
        out: Dict[int, float] = {}
        seen: set = set()
        idx = 0
        try:
            with open(cif_path) as fh:
                for line in fh:
                    s = line.strip()
                    if s.startswith("_atom_site."):
                        cols.append(s)
                        in_loop = True
                        continue
                    if in_loop and s.startswith("ATOM"):
                        parts = s.split()
                        if len(parts) < len(cols):
                            continue
                        col = {c.split(".")[1]: parts[i] for i, c in enumerate(cols)}
                        if col.get("label_atom_id") not in ("CA",):
                            continue
                        ch = col.get("auth_asym_id") or col.get("label_asym_id")
                        if ch != chain:
                            continue
                        rid = col.get("auth_seq_id") or col.get("label_seq_id")
                        if rid in seen:
                            continue
                        seen.add(rid)
                        idx += 1
                        try:
                            out[idx] = float(col.get("B_iso_or_equiv"))
                        except (TypeError, ValueError):
                            pass
                    elif in_loop and s and not s.startswith(("ATOM", "HETATM", "_atom_site.")):
                        if out:                 # left the atom_site loop after collecting atoms
                            break
        except OSError:
            return {}
        return out

    @staticmethod
    def _err(label: str, msg: str) -> Dict[str, Any]:
        return {"success": False, "label": label, "error": msg, "source": "error"}


class _RemoteBreach(Exception):
    """Raised by the fail-closed LOCAL-ONLY guard when a remote-MSA breach is detected."""


def boltz_available() -> bool:
    """Module-level capability signal for the Workbench engine picker (B2 3-state)."""
    try:
        return BoltzBridge().is_available()
    except Exception:
        return False
