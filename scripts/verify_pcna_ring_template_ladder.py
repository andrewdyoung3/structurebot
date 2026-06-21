"""
Step 1 (NO GPU) of the DIVERGENT-RING generalization study — scope + measure the crystal
PCNA-family ring-template divergence ladder, then STOP.

The question (quaternary analog of the monomer calibration arc): does a DIVERGENT ring template
convey the C3 assembly geometry the way a divergent monomer template unlocked the monomer fold?
Before any GPU fold we must confirm the ladder spans a REAL divergence range AND that the rungs
diverge in QUATERNARY structure (ring-TM), not just sequence — else there is no "divergent ring"
to test within the family.

DISCIPLINE (monomer-arc lessons applied upfront):
  • CRYSTAL ring templates ONLY (drop NMR / EM — the NMR ceiling compressed the monomer arc).
  • HOMOtrimer rings only (Sulfolobus/Aeropyrum PCNA are HETEROtrimers → excluded by design).
  • 1AXC (human) is the identical control = the ceiling anchor (ring-TM 1.0 by construction).

For each candidate: download mmCIF (ASU; fall back to biological assembly 1 if the ASU has <3 PCNA
chains, i.e. a crystallographic trimer), confirm X-ray, isolate the 3 PCNA chains (length filter
drops bound p21 peptides), then via US-align report:
  • seq-id to human PCNA (monomer structural alignment, identity-in-alignment)
  • monomer-TM to 1AXC PCNA monomer  (is it the PCNA fold?)
  • ring-TM to the 1AXC PCNA C3 ring  (-mm complex alignment — the QUATERNARY divergence axis)

Run: venv/Scripts/python.exe scripts/verify_pcna_ring_template_ladder.py     (no GPU; downloads + US-align)
"""
import os, sys, shlex, tempfile, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import config as _cfg
from wsl_bridge import WSLBridge

# Ladder: human (control) → plant → yeast → archaeal. All crystal homotrimer PCNA rings.
LADDER = [
    ("1AXC", "Homo sapiens (human) + p21  [CONTROL]"),
    ("1VYM", "Homo sapiens (human), apo"),
    ("2ZVV", "Arabidopsis thaliana PCNA1 + p21 (plant)"),
    ("2ZVW", "Arabidopsis thaliana PCNA2 + p21 (plant)"),
    ("1PLQ", "Saccharomyces cerevisiae (yeast)"),
    ("1GE8", "Pyrococcus furiosus (archaeal)"),
]
REF = "1AXC"
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="pcna_ladder_")
_wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
_USEXE = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")


def fetch(pdb, assembly=False):
    """Download an mmCIF (ASU or biological assembly 1) from RCSB to cache/. Returns path or None."""
    name = f"{pdb}-assembly1.cif" if assembly else f"{pdb}.cif"
    dst = CACHE / name
    if dst.is_file() and dst.stat().st_size > 0:
        return str(dst)
    url = f"https://files.rcsb.org/download/{name}"
    try:
        urllib.request.urlretrieve(url, dst)
        return str(dst)
    except Exception as exc:
        print(f"   [fetch] {name} failed: {exc}")
        return None


def exptl_method(path):
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("_exptl.method"):
            return ln.split("_exptl.method", 1)[1].strip().strip("'\"")
        if ln.strip().upper().startswith(("X-RAY", "SOLUTION NMR", "ELECTRON")):
            return ln.strip().strip("'\"")
    return "?"


def chain_residue_counts(path, col):
    """{chain_id: n_CA} under the chosen _atom_site column (auth_asym_id or label_asym_id)."""
    cols, counts = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("_atom_site."):
                cols.append(ln.strip())
            elif ln.startswith(("ATOM", "HETATM")):
                p = ln.split()
                ci = next((i for i, c in enumerate(cols) if c.endswith(col)), None)
                ai = next((i for i, c in enumerate(cols) if c.endswith("label_atom_id")), None)
                if ci is not None and ai is not None and ci < len(p) and ai < len(p) and p[ai].strip('"') == "CA":
                    counts[p[ci]] = counts.get(p[ci], 0) + 1
    return counts


def filter_chains(src, wanted, dst, col):
    wanted = set(wanted); lines = open(src, encoding="utf-8", errors="replace").read().splitlines()
    hdr, data, i, n = [], [], 0, len(lines)
    while i < n:
        if lines[i].startswith("_atom_site."):
            while i < n and lines[i].startswith("_atom_site."):
                hdr.append(lines[i].strip()); i += 1
            ci = next((k for k, c in enumerate(hdr) if c.endswith(col)), None)
            while i < n and lines[i].startswith(("ATOM", "HETATM")):
                p = lines[i].split()
                if ci is not None and ci < len(p) and p[ci] in wanted:
                    data.append(lines[i])
                i += 1
            break
        i += 1
    open(dst, "w", encoding="utf-8").write("data_x\nloop_\n" + "\n".join(hdr) + "\n" + "\n".join(data) + "\n")
    return dst


def _parse(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("#PDBchain1"):
            continue
        p = ln.split("\t")
        if len(p) >= 11:
            try:
                return {"tm1": float(p[2]), "tm2": float(p[3]), "rmsd": float(p[4]),
                        "id1": float(p[5]), "id2": float(p[6]), "idali": float(p[7]),
                        "l1": int(p[8]), "l2": int(p[9]), "lali": int(p[10])}
            except ValueError:
                continue
    return None


def usalign(q, r, extra=""):
    if not (os.path.isfile(q) and os.path.isfile(r)):
        return None
    cmd = (f"{shlex.quote(_USEXE)} {shlex.quote(_wsl.translate_path(os.path.abspath(q)))} "
           f"{shlex.quote(_wsl.translate_path(os.path.abspath(r)))} -outfmt 2 {extra}").strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


def pcna_ring(pdb):
    """Return (ring_file, monomer_file, method, note) for a PDB's 3-chain PCNA ring, or (None,...)."""
    path = fetch(pdb)
    if not path:
        return None, None, "?", "download failed"
    method = exptl_method(path)
    # PCNA chains = long protein chains (>150 CA); drops short bound peptides (p21 ≈ 18 aa)
    for col, src in (("auth_asym_id", path), ("auth_asym_id", "ASM"), ("label_asym_id", "ASM")):
        if src == "ASM":
            apath = fetch(pdb, assembly=True)
            if not apath:
                continue
            src_path = apath
        else:
            src_path = path
        counts = chain_residue_counts(src_path, col)
        pcna = sorted([c for c, n in counts.items() if n > 150])
        if len(pcna) >= 3:
            ring = filter_chains(src_path, pcna[:3], os.path.join(TD, f"{pdb}_ring.cif"), col)
            mono = filter_chains(src_path, [pcna[0]], os.path.join(TD, f"{pdb}_mono.cif"), col)
            note = f"{len(pcna)} PCNA chains via {col}" + (" [assembly1]" if src_path != path else " [ASU]")
            return ring, mono, method, note
    return None, None, method, "no 3-chain PCNA ring found (ASU or assembly1)"


# ── build the 1AXC reference ring + monomer ──────────────────────────────────────────
print("[ref] building human 1AXC PCNA ring + monomer reference…")
ref_ring, ref_mono, ref_method, ref_note = pcna_ring(REF)
if not ref_ring:
    print(f"[abort] could not build 1AXC reference ({ref_note})"); sys.exit(2)
print(f"[ref] 1AXC method={ref_method} ({ref_note})")

# ── measure the ladder ────────────────────────────────────────────────────────────────
print("\n══ PCNA-FAMILY RING-TEMPLATE DIVERGENCE LADDER (vs human 1AXC) ══")
print(f"  {'PDB':<6}{'organism':<42}{'method':<14}{'seq-id%':>8}{'monoTM':>8}{'ringTM':>8}{'ringRMSD':>9}")
rows = []
for pdb, org in LADDER:
    ring, mono, method, note = pcna_ring(pdb)
    crystal = "X-RAY" in (method or "").upper()
    if not ring:
        print(f"  {pdb:<6}{org:<42}{(method or '?'):<14}{'— DROPPED: '+note}")
        rows.append((pdb, org, method, None, None, None, None, "DROPPED:"+note)); continue
    if not crystal:
        print(f"  {pdb:<6}{org:<42}{(method or '?'):<14}{'— DROPPED: not crystal (discipline)'}")
        rows.append((pdb, org, method, None, None, None, None, "DROPPED: not crystal")); continue
    mono_a = usalign(mono, ref_mono)                       # monomer structural alignment → seq-id + TM
    ring_a = usalign(ring, ref_ring, extra="-mm 1")        # -mm complex alignment → ring-TM
    seqid  = round(mono_a["idali"] * 100, 1) if mono_a else None     # identity within the structural alignment
    monoTM = round(mono_a["tm2"], 3) if mono_a else None             # normalized to 1AXC monomer
    ringTM = round(ring_a["tm2"], 3) if ring_a else None             # normalized to 1AXC ring
    rRMSD  = round(ring_a["rmsd"], 2) if ring_a else None
    print(f"  {pdb:<6}{org:<42}{method[:13]:<14}{str(seqid):>8}{str(monoTM):>8}{str(ringTM):>8}{str(rRMSD):>9}  ({note})")
    rows.append((pdb, org, method, seqid, monoTM, ringTM, rRMSD, note))

# ── ladder verdict: does it span REAL quaternary (ring-TM) divergence? ────────────────
ok = [r for r in rows if r[5] is not None and r[0] != REF]
ringtms = sorted(r[5] for r in ok)
seqids  = sorted(r[3] for r in ok if r[3] is not None)
print("\n── LADDER ASSESSMENT ──")
if ringtms:
    print(f"  non-control rungs measured: {len(ok)}")
    print(f"  seq-id span : {min(seqids):.1f}% → {max(seqids):.1f}%" if seqids else "  seq-id span : n/a")
    print(f"  ring-TM span: {min(ringtms):.3f} → {max(ringtms):.3f}  (spread {max(ringtms)-min(ringtms):.3f})")
    if max(ringtms) - min(ringtms) < 0.10 and min(ringtms) > 0.80:
        print("  ⚠ QUATERNARY structure is HIGHLY CONSERVED across the family — the rungs diverge in")
        print("    SEQUENCE but NOT in ring geometry. A within-PCNA 'divergent ring' test may have NO")
        print("    real divergent rung (ring-TM near-constant). Step 2 framing must account for this:")
        print("    a near-1.0 ring-TM template is still effectively the answer (copy/unlock indistinct).")
    else:
        print("  ✓ ring-TM spans a real range — there are genuinely divergent-quaternary rungs to titrate.")
else:
    print("  no rungs measured — fix downloads before Step 2.")

print("\nSTOP — Step 1 (ladder scoping) complete. Report before any GPU fold (Step 2 titration).")
