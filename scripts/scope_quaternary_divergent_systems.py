"""
SCOPING (NO GPU) — find a genuinely QUATERNARY-DIVERGENT oligomer system, the thing PCNA lacked.

Context: We proved Boltz multi-chain templates convey quaternary geometry (PCNA ring, ring-TM 0.99),
then asked whether a DIVERGENT ring template conveys it — but PCNA quaternary structure is CONSERVED
across all life (ring-TM 0.896→0.985 spread 0.089), so it could only test sequence-divergence, not
quaternary divergence. THE question: does a quaternary-divergent template IMPOSE its assembly on the
fold, or does the sequence's intrinsic assembly preference win? Decisive here (unlike PCNA) because a
quaternary-divergent template that gets COPIED produces a measurably WRONG assembly.

This step (the whole task this turn): scope candidate systems, harvest assemblies, measure pairwise
assembly-structTM with US-align -mm, report the ladder. GATE: does the ladder span REAL quaternary
divergence headroom (assembly-TM ~0.5–0.9, not PCNA's compressed 0.90–0.99) at FIXED chain count?
STOP and report for the framing call — NO GPU until a divergent ladder with headroom is confirmed.

DISCIPLINE (carried from the PCNA arc):
  • CRYSTAL only (no NMR — the ceiling burned us once).
  • Fixed oligomeric state preferred (templatable chain-for-chain).
  • Report crystal-vs-NMR, seq-id, oligomeric state, chain counts for every rung — no assuming.

CANDIDATE SYSTEMS:
  (A) DIRECTION 2 — RNase A: ONE sequence (100% id), TWO divergent crystal dimer assemblies.
      The sharp bistable test (guide A→A? guide B→B?). Same monomer fold, divergent quaternary.
        7RSA  closed monomer (intrinsic ground state — the unguided assembly preference proxy)
        1A2W  N-terminal domain-swapped dimer (swaps the N-term alpha-helix)
        1F0V  C-terminal domain-swapped dimer (swaps the C-term beta-strand) [+ minor N-swap dimer]
        1JS0  domain-swapped TRIMER (different chain count — reported, not in the dimer ladder)
  (B) DIRECTION 1 — Cro family (REPORTED w/ caveat): "completely different homodimer interfaces"
      (N15 vs lambda Cro) BUT a fold-switching confound (lambda alpha+beta vs P22/N15 all-alpha) —
      conflates fold divergence with quaternary divergence, so NOT clean chain-for-chain. Probed
      lightly for the record.

Run: venv/Scripts/python.exe scripts/scope_quaternary_divergent_systems.py   (no GPU; downloads + US-align)
"""
import os, sys, shlex, tempfile, urllib.request, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import config as _cfg
from wsl_bridge import WSLBridge

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="quatdiv_")
_wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
_USEXE = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")


def fetch(pdb, assembly=False):
    name = f"{pdb}-assembly1.cif" if assembly else f"{pdb}.cif"
    dst = CACHE / name
    if dst.is_file() and dst.stat().st_size > 0:
        return str(dst)
    url = f"https://files.rcsb.org/download/{name}"
    try:
        urllib.request.urlretrieve(url, dst); return str(dst)
    except Exception as exc:
        print(f"   [fetch] {name} failed: {exc}"); return None


def exptl_method(path):
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("_exptl.method"):
            return ln.split("_exptl.method", 1)[1].strip().strip("'\"")
        if ln.strip().upper().startswith(("X-RAY", "SOLUTION NMR", "ELECTRON")):
            return ln.strip().strip("'\"")
    return "?"


def chain_residue_counts(path, col):
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
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 120))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


def ca_coords(path, col):
    """{chain_id: [(x,y,z), ...]} CA-only, for contact-based dimer-partner detection."""
    cols, out = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("_atom_site."):
                cols.append(ln.strip())
            elif ln.startswith(("ATOM", "HETATM")):
                p = ln.split()
                ci = next((i for i, c in enumerate(cols) if c.endswith(col)), None)
                ai = next((i for i, c in enumerate(cols) if c.endswith("label_atom_id")), None)
                xi = next((i for i, c in enumerate(cols) if c.endswith("Cartn_x")), None)
                yi = next((i for i, c in enumerate(cols) if c.endswith("Cartn_y")), None)
                zi = next((i for i, c in enumerate(cols) if c.endswith("Cartn_z")), None)
                if None in (ci, ai, xi, yi, zi):
                    continue
                if max(ci, ai, xi, yi, zi) < len(p) and p[ai].strip('"') == "CA":
                    try:
                        out.setdefault(p[ci], []).append((float(p[xi]), float(p[yi]), float(p[zi])))
                    except ValueError:
                        pass
    return out


def dimer_partner(path, col, anchor, cutoff=8.0):
    """Chain (besides `anchor`) with the most CA-CA contacts < cutoff to `anchor` = its dimer mate."""
    cc = ca_coords(path, col)
    if anchor not in cc:
        return None, {}
    best, counts = None, {}
    for ch, pts in cc.items():
        if ch == anchor:
            continue
        n = 0
        for ax, ay, az in cc[anchor]:
            for bx, by, bz in pts:
                if (ax-bx)**2 + (ay-by)**2 + (az-bz)**2 < cutoff*cutoff:
                    n += 1; break
        counts[ch] = n
        if best is None or n > counts[best]:
            best = ch
    return best, counts


def harvest(pdb, min_ca=80):
    """Return dict: method, asu_chains{ch:nCA}, asm_chains{ch:nCA}, and a best 'assembly' file.
    Picks the file (ASU or assembly1) and the protein chains (>min_ca CA) that form the assembly."""
    path = fetch(pdb)
    if not path:
        return None
    method = exptl_method(path)
    out = {"pdb": pdb, "method": method, "files": {}}
    for tag, getter in (("ASU", lambda: path), ("ASM", lambda: fetch(pdb, assembly=True))):
        src = getter()
        if not src:
            out["files"][tag] = None; continue
        col = "auth_asym_id"
        counts = chain_residue_counts(src, col)
        prot = {c: n for c, n in counts.items() if n > min_ca}
        out["files"][tag] = {"src": src, "col": col, "prot_chains": prot}
    return out


print("══ SCOPING: QUATERNARY-DIVERGENT OLIGOMER CANDIDATES (no GPU) ══\n")

# ── DIRECTION 2: RNase A — one sequence, multiple crystal assemblies ──────────────────
RNASE = ["7RSA", "1A2W", "1F0V", "1JS0"]
LABEL = {
    "7RSA": "RNase A closed MONOMER (intrinsic ground state)",
    "1A2W": "RNase A N-terminal domain-swapped DIMER",
    "1F0V": "RNase A C-terminal domain-swapped DIMER (+minor N-swap)",
    "1JS0": "RNase A domain-swapped TRIMER",
}
print("── DIRECTION 2 — RNase A (bovine, 100% identical sequence across forms) ──")
info = {}
for pdb in RNASE:
    h = harvest(pdb)
    info[pdb] = h
    if not h:
        print(f"  {pdb}: download FAILED"); continue
    asu = h["files"].get("ASU"); asm = h["files"].get("ASM")
    asu_s = ",".join(f"{c}:{n}" for c, n in sorted(asu["prot_chains"].items())) if asu else "—"
    asm_s = ",".join(f"{c}:{n}" for c, n in sorted(asm["prot_chains"].items())) if asm else "—"
    print(f"  {pdb}  {h['method'][:12]:<12}  {LABEL[pdb]}")
    print(f"        ASU prot-chains  : {asu_s}")
    print(f"        assembly1 chains : {asm_s}")

print()
print("── PAIRWISE QUATERNARY (assembly-TM via -mm) vs MONOMER-TM (single chain) ──")
print("   assembly-TM LOW + monomer-TM HIGH  ⇒  pure quaternary divergence (the thing PCNA lacked)\n")

# Build per-form dimer/monomer files for the dimer forms (1A2W, 1F0V); monomer for 7RSA.
def two_chain_assembly(pdb, prefer="ASU"):
    """Pick a file with >=2 long chains; isolate the INTERTWINED dimer (anchor chain + its closest-
    contact mate, so a 4-chain ASU of two independent dimers yields ONE real dimer, not a random pair).
    Return (dimer_file, mono_file, [anchor, mate], tag, contacts)."""
    h = info.get(pdb)
    if not h: return None, None, [], "?", {}
    for tag in ([prefer] + [t for t in ("ASU", "ASM") if t != prefer]):
        f = h["files"].get(tag)
        if not (f and f["prot_chains"]):
            continue
        chains = sorted(f["prot_chains"].keys())
        anchor = chains[0]
        if len(chains) >= 2:
            mate, contacts = dimer_partner(f["src"], f["col"], anchor)
            if mate is None:
                mate = chains[1]; contacts = {}
            pair = [anchor, mate]
            ring = filter_chains(f["src"], pair, os.path.join(TD, f"{pdb}_dimer.cif"), f["col"])
            mono = filter_chains(f["src"], [anchor], os.path.join(TD, f"{pdb}_mono.cif"), f["col"])
            return ring, mono, pair, tag, contacts
        # single chain (monomer)
        mono = filter_chains(f["src"], [anchor], os.path.join(TD, f"{pdb}_mono.cif"), f["col"])
        return None, mono, [anchor], tag, {}
    return None, None, [], "?", {}

forms = {}
for pdb in RNASE:
    ring, mono, chains, tag, contacts = two_chain_assembly(pdb)
    forms[pdb] = {"ring": ring, "mono": mono, "chains": chains, "tag": tag, "contacts": contacts}
    if len(chains) >= 2:
        print(f"  [{pdb}] dimer isolated from {tag}: chains {chains[0]}+{chains[1]}  "
              f"(CA-CA contacts to {chains[0]}: {contacts})")

# monomer-TM: each form's protomer vs the closed 7RSA monomer
print("  monomer-TM (protomer vs closed 7RSA monomer) — fold sanity:")
ref_mono = forms["7RSA"]["mono"]
for pdb in RNASE:
    if pdb == "7RSA": continue
    a = usalign(forms[pdb]["mono"], ref_mono)
    if a:
        print(f"    {pdb} protomer vs 7RSA : monoTM(norm-7RSA)={a['tm2']:.3f}  seqid-in-ali={a['idali']*100:.1f}%  RMSD={a['rmsd']:.2f}  Lali={a['lali']}")
    else:
        print(f"    {pdb} protomer vs 7RSA : US-align failed")

# assembly-TM: pairwise -mm between the dimer forms (the QUATERNARY divergence axis)
print("\n  assembly-TM (-mm, pairwise between dimer forms) — the QUATERNARY axis:")
dimers = [p for p in ("1A2W", "1F0V") if forms[p]["ring"] and len(forms[p]["chains"]) >= 2]
for a_pdb, b_pdb in itertools.combinations(dimers, 2):
    a = usalign(forms[a_pdb]["ring"], forms[b_pdb]["ring"], extra="-mm 1")
    if a:
        print(f"    {a_pdb} ⇄ {b_pdb} : assemblyTM={a['tm2']:.3f} / {a['tm1']:.3f}  RMSD={a['rmsd']:.2f}  Lali={a['lali']}  "
              f"(N-swap ⇄ C-swap)")
    else:
        print(f"    {a_pdb} ⇄ {b_pdb} : US-align -mm failed")

print("\n── ASSESSMENT ──")
print("  Target intrinsic preference proxy = the closed 7RSA monomer (the form the sequence 'wants').")
print("  If assembly-TM(N-swap ⇄ C-swap) is LOW (<~0.7) while monomer-TM stays HIGH, RNase A spans")
print("  REAL quaternary divergence at (near) fixed chain count — the headroom PCNA lacked — and the")
print("  bistable guide-A→A? / guide-B→B? test is decisive. Report the ladder; STOP before any fold.")
print("\nSTOP — scoping complete. Report the ladder for the framing call (no GPU until headroom confirmed).")
