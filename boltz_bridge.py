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
    author-index), plddt_by_chain({chain: {idx: pLDDT}}), chain_ids(observed CIF order),
    source:"local_boltz_env", seed, error}. ERROR-FIRST + GRACEFUL throughout.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg

_SOURCE = "local_boltz_env"


def _yaml_id(val: Any) -> str:
    """Emit a chain id (or list of ids) as a YAML scalar or flow-list. A scalar `"A"` → `A`;
    a list `["A", "B"]` → `[A, B]` (Boltz accepts either for chain_id/template_id)."""
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(str(v) for v in val) + "]"
    return str(val)

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
        templates:    Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Fold *chains* (one `protein` block each, MSA-free) with Boltz via WSL. Returns the
        result contract (success/cif_path/mean_plddt/iptm/chains_ptm/plddt/source/seed/error).
        Never raises for an expected failure — returns {success: False, error: ...}.

        TEMPLATE-GUIDED (optional): *templates* is a per-template list of dicts — each
        ``{cif|pdb: <WINDOWS path>, chain_id, template_id, force, threshold}`` (see `_build_yaml`).
        The cif/pdb path is a LOCAL on-disk structure; it is `translate_path`'d into WSL and
        emitted as Boltz's top-level `templates:` block. A template is NOT an MSA — it adds no
        `msa:` line and no remote flag, so the fail-closed LOCAL-ONLY guard is unaffected
        (chains still declare `msa: empty`). The list is per-template from day one so
        multi-template / multimer (per-chain `chain_id`↔`template_id`) fold in with no schema
        change; the first build passes a single monomer entry."""
        if not chains:
            return self._err(label, "no chains given to fold")
        seed = self._seed if seed is None else int(seed)

        # Translate each template's on-disk structure path into WSL before it enters the YAML
        # (Boltz reads the cif/pdb from inside WSL, like the YAML/out paths below).
        tmpl_yaml = self._translate_template_paths(templates) if templates else None
        yaml_text = self._build_yaml(chains, tmpl_yaml)
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
            # Boltz can raise a per-input exception (e.g. a template parse KeyError/ValueError) yet
            # still EXIT 0 — so r.ok is True but no model is written. Surface the SWALLOWED error
            # from the logs instead of the opaque "no predicted CIF", so the real cause is visible.
            swallowed = self._extract_boltz_error(r.get("stdout", ""), r.get("stderr", ""))
            base = parsed.get("error", "Boltz produced no parsable output")
            return self._err(label, f"{base}{(' — Boltz error: ' + swallowed) if swallowed else ''}")
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
    def _translate_template_paths(
        self, templates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return *templates* with each entry's `cif`/`pdb` value translated to a WSL path
        (the only field that must change crossing the Windows→WSL seam). Other fields
        (chain_id/template_id/force/threshold) pass through verbatim. Pure-ish (path xlate)."""
        out: List[Dict[str, Any]] = []
        for t in templates:
            t2 = dict(t)
            for key in ("cif", "pdb"):
                if t2.get(key):
                    t2[key] = self._wsl.translate_path(os.path.abspath(str(t2[key])))
            out.append(t2)
        return out

    @staticmethod
    def _build_yaml(
        chains:    List[Dict[str, str]],
        templates: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Boltz input YAML. Chains fold MSA-free (`msa: empty`). When *templates* is given,
        a top-level `templates:` block is appended — one entry per dict, emitting only the
        fields present. Boltz 2.2.1 per-template schema (verified against the installed
        source): `cif`|`pdb` (path), `chain_id` (query chain(s) the template steers; default
        ALL protein chains), `template_id` (template chain(s); if BOTH given the counts must
        match), `force` (bool, default False), `threshold` (REQUIRED when force is True — the
        distance the template is forced toward; omitted otherwise). chain_id/template_id may be
        a scalar or a list — emitted as a YAML flow list when a list. Paths are expected to be
        WSL paths already (see `_translate_template_paths`)."""
        lines = ["version: 1", "sequences:"]
        for c in chains:
            lines += [
                "  - protein:",
                f"      id: {c['id']}",
                f"      sequence: {c['sequence']}",
                "      msa: empty",
            ]
        if templates:
            lines.append("templates:")
            for t in templates:
                path_key = "cif" if t.get("cif") else ("pdb" if t.get("pdb") else None)
                if not path_key:
                    continue                       # skip a malformed entry (no structure path)
                # In a YAML sequence item the mapping keys are SIBLINGS — chain_id/force/etc.
                # align with the path key at 4 spaces (the `- ` is the item indent), NOT 6.
                # `protein:` nests its children at 6 because they sit UNDER a key; a template's
                # fields are flat, so 6 is invalid YAML (it made chain_id a child of `pdb`).
                lines.append(f"  - {path_key}: {t[path_key]}")
                for k in ("chain_id", "template_id"):
                    if t.get(k) is not None:
                        lines.append(f"    {k}: {_yaml_id(t[k])}")
                if t.get("force"):
                    lines.append("    force: true")
                    if t.get("threshold") is not None:
                        lines.append(f"    threshold: {t['threshold']}")
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
        # PER-CHAIN pLDDT (each auth_asym_id keyed 1..N) + the OBSERVED chain ids in CIF order.
        # Hetero assemblies need each ChainDesign's OWN chain pLDDT (not a shared rep), and the
        # observed-id list is the read-back the consumer guards sent==got + aligns index-keyed
        # chains_ptm against. `plddt` (rep chain) is retained for the monomer/back-compat path.
        plddt_by_chain, chain_ids = self._per_chain_bfactor(cif_src)
        rep_chain = chains[0]["id"]
        plddt = plddt_by_chain.get(rep_chain, {})                # rep chain (1-based author index)
        all_vals = [v for d in plddt_by_chain.values() for v in d.values()]
        mean_plddt = (float(cplx) * 100.0 if isinstance(cplx, (int, float))
                      else (round(sum(all_vals) / len(all_vals), 2) if all_vals else 0.0))
        return {
            "success":        True,
            "cif_path":       cif_dst.name,
            "mean_plddt":     round(mean_plddt, 2),
            "iptm":           conf.get("iptm"),
            "ptm":            conf.get("ptm"),
            "chains_ptm":     conf.get("chains_ptm"),
            "plddt":          plddt,                             # rep chain (back-compat)
            "plddt_by_chain": plddt_by_chain,                    # {chain: {1-based idx: pLDDT}}
            "chain_ids":      chain_ids,                         # observed, CIF order
            "length":         len(plddt),
        }

    @staticmethod
    def _per_chain_bfactor(cif_path: str) -> Tuple[Dict[str, Dict[int, float]], List[str]]:
        """Per-CHAIN per-residue pLDDT in ONE pass: returns ({auth_asym_id: {1-based idx: pLDDT}},
        [chain ids in CIF first-appearance order]). Each chain's CA B-factors (0–100) are keyed
        1..N over that chain's own residue order — matches `fold_summary`'s 1-based→author-resnum
        remap, now per chain. The id list is the read-back the consumer aligns the index-keyed
        `chains_ptm` against and guards sent==observed. Parses the `_atom_site` loop header for
        robust column lookup (mmCIF column order is not fixed)."""
        cols: List[str] = []
        in_loop = False
        by_chain: Dict[str, Dict[int, float]] = {}
        order: List[str] = []
        seen: Dict[str, set] = {}
        idx: Dict[str, int] = {}
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
                        if ch is None:
                            continue
                        if ch not in by_chain:
                            by_chain[ch] = {}
                            order.append(ch)
                            seen[ch] = set()
                            idx[ch] = 0
                        rid = col.get("auth_seq_id") or col.get("label_seq_id")
                        if rid in seen[ch]:
                            continue
                        seen[ch].add(rid)
                        idx[ch] += 1
                        try:
                            by_chain[ch][idx[ch]] = float(col.get("B_iso_or_equiv"))
                        except (TypeError, ValueError):
                            pass
                    elif in_loop and s and not s.startswith(("ATOM", "HETATM", "_atom_site.")):
                        if by_chain:            # left the atom_site loop after collecting atoms
                            break
        except OSError:
            return {}, []
        return by_chain, order

    @staticmethod
    def _extract_boltz_error(stdout: str, stderr: str) -> str:
        """Pull the real cause out of a Boltz run that exited 0 but wrote no model. Boltz catches a
        per-input exception inside `process_input` and continues, so the traceback lands in the
        logs while the process returns 0. Prefer the final exception line (e.g.
        ``ValueError: Template chain A is not one of the protein chains`` / ``KeyError: 'Axp'``);
        fall back to the last non-progress-bar stderr line. Empty if nothing error-shaped is found."""
        text = ((stderr or "") + "\n" + (stdout or "")).replace("\r", "\n")
        exc_lines = [
            ln.strip() for ln in text.splitlines()
            if re.match(r"^\s*[A-Za-z_][\w.]*(Error|Exception|Warning)\b\s*:", ln)
            or ln.strip().startswith(("ValueError", "KeyError", "TypeError", "RuntimeError",
                                      "FileNotFoundError", "AssertionError"))
        ]
        if exc_lines:
            return exc_lines[-1][:300]
        # else the last meaningful (non-tqdm/non-blank) line
        for ln in reversed(text.splitlines()):
            s = ln.strip()
            if s and "%|" not in s and "it/s" not in s:
                return s[:300]
        return ""

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
