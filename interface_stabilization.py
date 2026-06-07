"""
interface_stabilization.py
---------------------------
Inter-subunit interface detection, characterization, and disulfide candidate
prediction for oligomeric assemblies.

Dual-state principle
--------------------
Every disulfide candidate is evaluated in context of BOTH states:
  Assembled   : interface contacts from zone-select on the assembly model
  Dissociated : monomer stability from DisulfideBridge (DynaMut2 / ESM)

A disulfide at the A-B intra-subunit interface increases interface burial
(good) but exposes the engineered Cys on dissociation only at the interface
face — DiSulfideBridge's stability filter already guards against this.

Sub-model addressing
--------------------
Assemblies from ``sym #N assembly M copies true`` produce a group model
(e.g. ``#2``) with sub-models ``#2.1``, ``#2.2``, … each carrying their
own chain IDs.  Zone-selection uses sub-model specs::

    select #2.1/A@CA & (#2.2/B :< 5.0); info selection

NOT the flat spec ``#2/A``, which addresses all sub-models simultaneously.

Interface types
---------------
``intra_subunit``  Two chains within the SAME sub-model (e.g. A-B in #2.1).
                   Present in the AU PDB → disulfide scan runs immediately.
``inter_subunit``  Same or different chain ID between DIFFERENT sub-models
                   (e.g. #2.1/A vs #2.2/A).  Requires assembly coordinates;
                   disulfide scan is deferred to Phase 2.
``flat``           Model has no sub-models (raw AU); plain chain-chain interface.
                   Treated identically to intra_subunit.

Output schema (per interface, stored in data["interfaces"])
-------------------------------------------------------------
{
    "type":               "intra_subunit" | "inter_subunit" | "flat",
    "spec_a":             "#2.1/A",
    "spec_b":             "#2.2/B",
    "chain_a":            "A",
    "chain_b":            "B",
    "submodel_a":         "2.1",
    "submodel_b":         "2.2",
    "contact_residues_a": [int, ...],
    "contact_residues_b": [int, ...],
    "contact_count":      int,
    "buried_area_ang2":   float | None,
    "disulfide_top":      {...} | None,
    "disulfide_count":    int,
    "disulfide_note":     str | None,
}
"""

from __future__ import annotations

import re
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

from assembly_analyser import AssemblyAnalyser
from disulfide_bridge import DisulfideBridge
from tool_router import ToolStepResult

_CONTACT_DISTANCE: float = 5.0
_COLOURS = ["orange", "magenta", "lime", "gold", "cyan"]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pprint(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _parse_buried_area(text: str) -> Optional[float]:
    """Parse ChimeraX ``measure buriedarea`` output → Å² float or None."""
    m = re.search(r"=\s*([\d.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class InterfaceStabilization:
    """
    Detect and characterize inter-chain interfaces in oligomeric assemblies,
    then rank inter-chain disulfide candidates at the most promising interface.

    Usage::

        stab = InterfaceStabilization(bridge=cx_bridge, session=session_state)
        result = stab.analyze(
            model_id         = "2",   # assembly group model
            pdb_path         = "cache/2VNC.pdb",
            pdb_id           = "2VNC",
        )
    """

    def __init__(self, bridge: Any, session: Any):
        self.bridge  = bridge
        self.session = session

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        model_id:          str,
        pdb_path:          str,
        pdb_id:            Optional[str]          = None,
        contact_distance:  float                  = _CONTACT_DISTANCE,
        top_n_disulfides:  int                    = 3,
        progress_callback: Optional[Callable]     = None,
    ) -> ToolStepResult:
        """
        Full interface stabilization pipeline.

        Parameters
        ----------
        model_id          : ChimeraX model ID (assembly group or plain AU model)
        pdb_path          : AU PDB file path for disulfide geometry scoring
        pdb_id            : 4-letter PDB ID (informational, used in output)
        contact_distance  : Å CA–CA cutoff for interface detection (default 5.0)
        top_n_disulfides  : number of disulfide candidates to visualize
        progress_callback : callable(str) for real-time progress
        """
        def _prog(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                _pprint(msg)

        t0 = _time.perf_counter()

        if not self.bridge.is_running():
            return ToolStepResult(
                tool="interface_stabilization", success=False,
                error="ChimeraX bridge unavailable.",
            )

        label = pdb_id or f"#{model_id}"
        _prog(f"🔗 [InterfaceStabilization] Detecting interfaces for {label}…")

        # 1. Enumerate interfaces (sub-model-aware)
        submodels = self._get_submodels(model_id)
        is_assembly = bool(submodels)

        if is_assembly:
            interfaces = self._detect_submodel_interfaces(
                model_id, submodels, contact_distance, _prog
            )
        else:
            # Plain AU model — flat chain detection
            interfaces = self._detect_flat_interfaces(model_id, contact_distance, _prog)

        if not interfaces:
            return ToolStepResult(
                tool="interface_stabilization", success=False,
                error=(
                    f"No inter-chain contacts detected in model #{model_id} "
                    f"at {contact_distance:.1f} Å cutoff.  Ensure multiple chains "
                    "are loaded and the distance cutoff is appropriate."
                ),
            )

        _prog(f"🔗 [InterfaceStabilization] {len(interfaces)} interface(s) detected.")

        # 2. Characterize each interface (buried area)
        for iface in interfaces:
            _prog(
                f"🔗 [InterfaceStabilization] Measuring buried area: "
                f"{iface['spec_a']} ↔ {iface['spec_b']}…"
            )
            iface["buried_area_ang2"] = self._measure_buried_area(
                iface["spec_a"], iface["spec_b"]
            )

        # 3. Disulfide scan (intra-subunit and flat only)
        ds_bridge = DisulfideBridge(chimerax_bridge=self.bridge)

        for iface in interfaces:
            if iface["type"] in ("intra_subunit", "flat"):
                _prog(
                    f"🔗 [InterfaceStabilization] Scanning inter-chain disulfides: "
                    f"chain {iface['chain_a']} ↔ chain {iface['chain_b']}…"
                )
                cands = self._scan_disulfides(
                    ds_bridge, pdb_path,
                    iface["chain_a"], iface["chain_b"],
                    iface.get("contact_residues_a", []),
                    _prog,
                )
                iface["disulfide_candidates"] = cands
                iface["disulfide_count"]      = len(cands)
                iface["disulfide_top"]        = cands[0] if cands else None
                iface["disulfide_note"]       = None
            else:
                iface["disulfide_candidates"] = None
                iface["disulfide_count"]      = 0
                iface["disulfide_top"]        = None
                iface["disulfide_note"] = (
                    "Inter-subunit disulfide scan deferred — requires biological "
                    "assembly PDB coordinates (Phase 2)."
                )

        # 4. Generate viz commands
        viz_cmds, viz_exps = self._build_viz_commands(
            model_id, interfaces, top_n_disulfides
        )

        # 5. Build summary
        summary = self._build_summary(interfaces, label)
        _prog(f"🔗 [InterfaceStabilization] {summary.splitlines()[0]}")

        # 6. Persist in session
        if self.session is not None:
            try:
                key = f"{model_id}:{pdb_id or 'unknown'}"
                self.session.set_interface_stabilization_result(
                    model_id, {
                        "model_id":    model_id,
                        "pdb_id":      pdb_id,
                        "interfaces":  interfaces,
                        "is_assembly": is_assembly,
                        "submodels":   submodels,
                    }
                )
            except AttributeError:
                pass

        elapsed_ms = (_time.perf_counter() - t0) * 1000
        return ToolStepResult(
            tool             = "interface_stabilization",
            success          = True,
            data             = {
                "interfaces":   interfaces,
                "is_assembly":  is_assembly,
                "submodels":    submodels,
                "model_id":     model_id,
                "pdb_id":       pdb_id,
                "n_interfaces": len(interfaces),
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    # ── Sub-model discovery ────────────────────────────────────────────────────

    def _get_submodels(self, group_model_id: str) -> List[str]:
        """
        Return sub-model IDs for an assembly group (e.g. '2' → ['2.1', '2.2']).
        Returns [] if the model has no sub-models (plain AU).
        """
        res = self.bridge.run_command(f"info models #{group_model_id}")
        if res.get("error"):
            return []
        text = res.get("value") or ""
        pat = re.compile(
            rf"model id #{re.escape(group_model_id)}\.(\d+)\b"
        )
        sub_ids = pat.findall(text)
        return [f"{group_model_id}.{sid}" for sid in sorted(set(sub_ids), key=int)]

    def _get_chains_for_submodel(self, submodel_id: str) -> List[str]:
        """
        Return single-letter chain IDs present in a sub-model.
        E.g. '2.1' → ['A', 'B'].
        """
        res = self.bridge.run_command(f"info chains #{submodel_id}")
        if res.get("error"):
            return []
        text = res.get("value") or ""
        found = re.findall(r"chain_id\s+([A-Za-z])\b", text)
        return sorted(set(found))

    # ── Interface detection ────────────────────────────────────────────────────

    def _detect_submodel_interfaces(
        self,
        group_model_id:   str,
        submodels:        List[str],
        contact_distance: float,
        prog:             Callable,
    ) -> List[Dict[str, Any]]:
        """
        Enumerate interfaces between all (submodel, chain) combinations.
        Produces intra_subunit and inter_subunit entries.
        """
        # Collect chains per sub-model
        submodel_chains: Dict[str, List[str]] = {}
        for sm in submodels:
            chains = self._get_chains_for_submodel(sm)
            if chains:
                submodel_chains[sm] = chains
            _time.sleep(0.05)  # small yield between ChimeraX queries

        interfaces: List[Dict[str, Any]] = []
        seen: set = set()

        # All (sm, chain) pairs
        all_specs: List[Tuple[str, str]] = [
            (sm, ch)
            for sm in submodels
            for ch in submodel_chains.get(sm, [])
        ]

        for i, (sm1, ch1) in enumerate(all_specs):
            for sm2, ch2 in all_specs[i + 1:]:
                if sm1 == sm2 and ch1 == ch2:
                    continue
                # Canonical key (sorted)
                key = tuple(sorted([f"{sm1}/{ch1}", f"{sm2}/{ch2}"]))
                if key in seen:
                    continue
                seen.add(key)

                iface_type = "intra_subunit" if sm1 == sm2 else "inter_subunit"
                spec_a = f"#{sm1}/{ch1}"
                spec_b = f"#{sm2}/{ch2}"

                prog(
                    f"🔗 [InterfaceStabilization] Checking {spec_a} ↔ {spec_b}…"
                )
                res_a, res_b = self._zone_select(spec_a, spec_b, contact_distance)

                if not res_a and not res_b:
                    continue

                interfaces.append({
                    "type":               iface_type,
                    "spec_a":             spec_a,
                    "spec_b":             spec_b,
                    "chain_a":            ch1,
                    "chain_b":            ch2,
                    "submodel_a":         sm1,
                    "submodel_b":         sm2,
                    "contact_residues_a": sorted(res_a),
                    "contact_residues_b": sorted(res_b),
                    "contact_count":      len(res_a) + len(res_b),
                    "buried_area_ang2":   None,
                    "disulfide_candidates": None,
                    "disulfide_count":    0,
                    "disulfide_top":      None,
                    "disulfide_note":     None,
                })

        self.bridge.run_command("select clear")
        return interfaces

    def _detect_flat_interfaces(
        self,
        model_id:         str,
        contact_distance: float,
        prog:             Callable,
    ) -> List[Dict[str, Any]]:
        """
        Interface detection for a plain AU model (no sub-models).
        Wraps AssemblyAnalyser.detect_interfaces() and converts to standard schema.
        """
        analyser = AssemblyAnalyser(bridge=self.bridge, session=self.session)
        raw = analyser.detect_interfaces(model_id, contact_distance)
        interfaces: List[Dict[str, Any]] = []
        for (c1, c2), resnos in raw.items():
            interfaces.append({
                "type":               "flat",
                "spec_a":             f"#{model_id}/{c1}",
                "spec_b":             f"#{model_id}/{c2}",
                "chain_a":            c1,
                "chain_b":            c2,
                "submodel_a":         model_id,
                "submodel_b":         model_id,
                "contact_residues_a": sorted(resnos),
                "contact_residues_b": [],
                "contact_count":      len(resnos),
                "buried_area_ang2":   None,
                "disulfide_candidates": None,
                "disulfide_count":    0,
                "disulfide_top":      None,
                "disulfide_note":     None,
            })
        return interfaces

    # ── Zone-select helper ─────────────────────────────────────────────────────

    def _zone_select(
        self,
        spec_a:           str,
        spec_b:           str,
        contact_distance: float,
    ) -> Tuple[set, set]:
        """
        Run bidirectional CA zone-select between spec_a and spec_b.

        Returns (resnos_from_a, resnos_from_b) — residue numbers of CA atoms
        from each side that are within contact_distance of the other chain.
        """
        pat = re.compile(r"atom id (?:#[^/]+)?/[A-Za-z]+:(\d+)@CA")
        res_a: set = set()
        res_b: set = set()

        # spec_a atoms near spec_b
        for attempt in range(3):
            cmd = f"select {spec_a}@CA & ({spec_b} :< {contact_distance}); info selection"
            r   = self.bridge.run_command(cmd)
            if r.get("error"):
                break
            text = r.get("value") or ""
            found = {int(m.group(1)) for m in pat.finditer(text)}
            if found:
                res_a = found
                break
            if attempt < 2:
                _time.sleep(0.5)

        # spec_b atoms near spec_a
        for attempt in range(3):
            cmd = f"select {spec_b}@CA & ({spec_a} :< {contact_distance}); info selection"
            r   = self.bridge.run_command(cmd)
            if r.get("error"):
                break
            text = r.get("value") or ""
            found = {int(m.group(1)) for m in pat.finditer(text)}
            if found:
                res_b = found
                break
            if attempt < 2:
                _time.sleep(0.5)

        return res_a, res_b

    # ── Buried area ───────────────────────────────────────────────────────────

    def _measure_buried_area(self, spec_a: str, spec_b: str) -> Optional[float]:
        """
        Run ``measure buriedarea spec_a withAtoms2 spec_b`` and return Å² or None.

        ChimeraX output: "Buried area between X and Y = 1518.8"
        """
        cmd = f"measure buriedarea {spec_a} withAtoms2 {spec_b}"
        r   = self.bridge.run_command(cmd)
        if r.get("error"):
            return None
        return _parse_buried_area(r.get("value") or "")

    # ── Disulfide scan ────────────────────────────────────────────────────────

    def _scan_disulfides(
        self,
        ds_bridge:        Any,
        pdb_path:         str,
        chain_a:          str,
        chain_b:          str,
        interface_resnos: List[int],
        prog:             Callable,
    ) -> List[Dict[str, Any]]:
        """
        Call DisulfideBridge for chain_a × chain_b from the AU PDB.
        Passes interface residues as binding_site_residues so they are excluded
        (we WANT residues AT the interface, not within the binding-site core).

        Returns sorted candidates list; [] on any error.
        """
        try:
            result = ds_bridge.analyze(
                pdb_path              = pdb_path,
                chain_a               = chain_a,
                chain_b               = chain_b,
                session               = self.session,
                progress_callback     = prog,
            )
            if result.success:
                return result.data.get("candidates") or []
        except Exception as exc:
            prog(f"  [InterfaceStabilization] Disulfide scan error: {exc}")
        return []

    # ── Visualization ─────────────────────────────────────────────────────────

    def _build_viz_commands(
        self,
        model_id:        str,
        interfaces:      List[Dict[str, Any]],
        top_n_disulfides: int,
    ) -> Tuple[List[str], List[str]]:
        cmds: List[str] = []
        exps: List[str] = []

        if not interfaces:
            return cmds, exps

        cmds += [f"cartoon #{model_id}", f"color #{model_id} white"]
        exps += ["Cartoon representation", f"Reset #{model_id} to white"]

        # Colour each interface
        for i, iface in enumerate(interfaces):
            colour  = _COLOURS[i % len(_COLOURS)]
            spec_a  = iface["spec_a"]
            spec_b  = iface["spec_b"]
            resnos  = iface.get("contact_residues_a", [])

            if resnos:
                res_spec = ",".join(str(r) for r in resnos[:30])
                # Build selector using spec_a but restrict to residue numbers
                # ChimeraX residue spec: #{submodel}/{chain}:{resnos}
                base = spec_a.rstrip("/")
                iface_sel = f"{base}:{res_spec}"
                cmds += [
                    f"color {iface_sel} {colour}",
                    f"show {iface_sel} atoms",
                    f"style {iface_sel} ball",
                ]
                exps += [
                    f"Colour interface {spec_a}↔{spec_b} ({len(resnos)} res) {colour}",
                    f"Show interface atoms {spec_a}",
                    f"Ball style interface residues",
                ]

        # Disulfide spheres for top intra-subunit interface
        for iface in interfaces:
            if iface["type"] not in ("intra_subunit", "flat"):
                continue
            cands = iface.get("disulfide_candidates") or []
            if not cands:
                continue

            sm_a = iface.get("submodel_a", model_id)
            sm_b = iface.get("submodel_b", model_id)
            ch_a = iface["chain_a"]
            ch_b = iface["chain_b"]
            colours_ss = ["gold", "silver", "light blue"]

            for j, cand in enumerate(cands[:top_n_disulfides]):
                ra      = cand.get("chain_a_residue")
                rb      = cand.get("chain_b_residue")
                score   = cand.get("combined_score", 0.0)
                col_ss  = colours_ss[j % len(colours_ss)]

                if ra is None or rb is None:
                    continue

                spec_ca = f"#{sm_a}/{ch_a}:{ra}"
                spec_cb = f"#{sm_b}/{ch_b}:{rb}"

                cmds += [
                    f"show {spec_ca} atoms",
                    f"style {spec_ca}@CB sphere",
                    f"color {spec_ca} {col_ss}",
                    f"show {spec_cb} atoms",
                    f"style {spec_cb}@CB sphere",
                    f"color {spec_cb} {col_ss}",
                    f"distance {spec_ca}@CB {spec_cb}@CB",
                ]
                exps += [
                    f"SS#{j+1} candidate at {ch_a}{ra}",
                    f"Sphere Cb {ch_a}{ra}",
                    f"Colour {col_ss}",
                    f"SS#{j+1} at {ch_b}{rb}",
                    f"Sphere Cb {ch_b}{rb}",
                    f"Colour {col_ss}",
                    f"Cb-Cb distance (score={score:.2f})",
                ]
            break  # only visualise disulfides for the first intra interface

        cmds.append(f"view #{model_id}")
        exps.append("Fit assembly in view")

        return cmds, exps

    # ── Summary ───────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        interfaces: List[Dict[str, Any]],
        label:      str,
    ) -> str:
        lines: List[str] = []
        lines.append(f"Interface stabilization — {label}  ({len(interfaces)} interface(s))")
        lines.append("-" * 56)

        for i, iface in enumerate(interfaces):
            buried = iface.get("buried_area_ang2")
            buried_s = f"{buried:.0f} Å²" if buried is not None else "N/A"
            n_contact = iface.get("contact_count", 0)
            itype = iface["type"].replace("_", "-")

            lines.append(
                f"  [{i+1}] {iface['spec_a']} ↔ {iface['spec_b']}  "
                f"({itype})  contacts={n_contact}  buried={buried_s}"
            )

            # Disulfide candidates
            n_ss = iface.get("disulfide_count", 0)
            top  = iface.get("disulfide_top")
            note = iface.get("disulfide_note")

            if note:
                lines.append(f"       SS scan: {note}")
            elif top:
                ra   = top.get("chain_a_residue", "?")
                rb   = top.get("chain_b_residue", "?")
                aa_a = top.get("chain_a_aa", "?")
                aa_b = top.get("chain_b_aa", "?")
                sc   = top.get("combined_score", 0.0)
                dist = top.get("cb_distance", 0.0)
                lines.append(
                    f"       SS scan: {n_ss} candidate(s).  Top: "
                    f"{iface['chain_a']}{ra}({aa_a})→C / "
                    f"{iface['chain_b']}{rb}({aa_b})→C  "
                    f"score={sc:.2f}  Cβ-Cβ={dist:.1f} Å"
                )
            else:
                lines.append(f"       SS scan: 0 candidates passed filters.")

        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Review disulfide candidates above (gold spheres in ChimeraX)")
        lines.append("  2. Order double-Cys gene synthesis for top candidate")
        lines.append("  3. Validate by DSP crosslinker / non-reducing SDS-PAGE")

        # Truncate at 20 lines to match other bridges
        return "\n".join(lines[:20])

    def __repr__(self) -> str:
        return (
            f"<InterfaceStabilization bridge={'set' if self.bridge else 'None'} "
            f"session={'set' if self.session else 'None'}>"
        )
