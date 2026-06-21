"""
SCOPING (NO GPU) — a NON-SWAPPED quaternary-divergent system, to test whether the RNase A
imposition finding is GENERAL or specific to intertwined domain swaps.

RNase A (§13) showed a template imposes WHICH domain-swapped dimer forms (directional 6/6 robust,
fidelity topology-dependent). Open question: does that hold for NORMAL docked assemblies — divergent
NON-intertwined interfaces — or is it a property of the swap topology? This step scopes candidates
with genuine quaternary divergence WITHOUT the swap, measures pairwise assembly-structTM (US-align
-mm), and reports the ladder + each candidate's confound. GATE (same as all session): the divergence
must span REAL headroom (like RNase A's 0.385, NOT PCNA's compressed 0.90–0.99), all crystal, ground
truth. STOP before any GPU.

Priority tiers (relay):
  T1  same-sequence, two NON-swap assemblies (interface polymorphism) — cleanest, but rare.
      Clean same-seq two-dimer case did NOT surface (only proposed/ambiguous: IRE1, SmAP) → likely
      rare (informative null). HIV-1 capsid is a genuine same-sequence non-swap case (hexamer vs
      pentamer build the fullerene core) — count differs (6 vs 5), so dimer-level is the fixed-count
      comparison.
  T2  morpheein — same sequence, two assemblies, DIFFERENT chain counts (messier to template).
      Human PBGS octamer 1E51 ⇄ hexamer 1PV8 (F12L; the prototype morpheein, two distinct dimers).
  T3  tight homolog family, divergent non-swap interface — reintroduces a sequence axis, but PCNA
      showed sequence-divergence transfers, so it's a controllable confound. Mouse PERIOD 3GDI/4DJ3
      ("dimerize in genuinely different ways", SOLUTION-VALIDATED); Ig light-chain Rei 1REI backup.

Per system: method (crystal?), per-chain CA counts (ASU + assembly1), monomer-TM + seq-id (single-
chain align), and the QUATERNARY axis = dimer assembly-TM (-mm, anchor chain + its closest-contact
partner = a fixed-count-2 unit) + (for different-count assemblies) full-assembly -mm.

Run: venv/Scripts/python.exe scripts/scope_nonswap_quaternary_divergent.py   (no GPU; downloads + US-align)
"""
import os, sys, shlex, tempfile, urllib.request, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import config as _cfg
from wsl_bridge import WSLBridge

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
TD = tempfile.mkdtemp(prefix="nonswap_")
_wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
_USEXE = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")

SYSTEMS = [
    ("PBGS morpheein  [T2: same-seq, diff chain count]", "morpheein (octamer⇄hexamer); 1PV8 is F12L",
     [("1E51", "human PBGS octamer"), ("1PV8", "human PBGS hexamer (F12L)")]),
    ("PERIOD PAS dimers [T3: homolog, seq axis]", "homolog isoforms, divergent dimer (solution-validated)",
     [("3GDI", "mouse PERIOD PAS-AB"), ("4DJ3", "mouse PERIOD PAS-AB")]),
    ("HIV-1 capsid     [T1-ish: same-seq non-swap, diff count]", "hexamer vs pentamer (fullerene core)",
     [("3H47", "HIV-1 CA hexamer"), ("3P05", "HIV-1 CA pentamer")]),
]


def fetch(pdb, assembly=False):
    name = f"{pdb}-assembly1.cif" if assembly else f"{pdb}.cif"
    dst = CACHE / name
    if dst.is_file() and dst.stat().st_size > 0: return str(dst)
    try: urllib.request.urlretrieve(f"https://files.rcsb.org/download/{name}", dst); return str(dst)
    except Exception as exc: print(f"   [fetch] {name} failed: {exc}"); return None


def exptl_method(path):
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("_exptl.method"):
            return ln.split("_exptl.method", 1)[1].strip().strip("'\"")
    return "?"


def chain_ca(path, col="auth_asym_id"):
    cols, counts = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("_atom_site."): cols.append(ln.strip())
            elif ln.startswith(("ATOM", "HETATM")):
                p = ln.split()
                ci = next((i for i, c in enumerate(cols) if c.endswith(col)), None)
                ai = next((i for i, c in enumerate(cols) if c.endswith("label_atom_id")), None)
                if ci is not None and ai is not None and ci < len(p) and ai < len(p) and p[ai].strip('"') == "CA":
                    counts[p[ci]] = counts.get(p[ci], 0) + 1
    return counts


def ca_coords(path, col="auth_asym_id"):
    cols, out = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("_atom_site."): cols.append(ln.strip())
            elif ln.startswith(("ATOM", "HETATM")):
                p = ln.split()
                idx = {k: next((i for i, c in enumerate(cols) if c.endswith(k)), None)
                       for k in (col, "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z")}
                if None in idx.values() or max(idx.values()) >= len(p): continue
                if p[idx["label_atom_id"]].strip('"') == "CA":
                    try: out.setdefault(p[idx[col]], []).append(
                        (float(p[idx["Cartn_x"]]), float(p[idx["Cartn_y"]]), float(p[idx["Cartn_z"]])))
                    except ValueError: pass
    return out


def dimer_partner(path, anchor, col="auth_asym_id", cutoff=8.0):
    cc = ca_coords(path, col)
    if anchor not in cc: return None, {}
    best, counts = None, {}
    for ch, pts in cc.items():
        if ch == anchor: continue
        n = sum(1 for ax, ay, az in cc[anchor]
                if any((ax-bx)**2+(ay-by)**2+(az-bz)**2 < cutoff*cutoff for bx, by, bz in pts))
        counts[ch] = n
        if best is None or n > counts[best]: best = ch
    return best, counts


def filter_chains(src, wanted, dst, col="auth_asym_id"):
    wanted = set(wanted); lines = open(src, encoding="utf-8", errors="replace").read().splitlines()
    hdr, data, i, n = [], [], 0, len(lines); ci = None
    while i < n:
        if lines[i].startswith("_atom_site."):
            while i < n and lines[i].startswith("_atom_site."): hdr.append(lines[i].strip()); i += 1
            ci = next((k for k, c in enumerate(hdr) if c.endswith(col)), None)
            while i < n and lines[i].startswith(("ATOM", "HETATM")):
                p = lines[i].split()
                if ci is not None and ci < len(p) and p[ci] in wanted: data.append(lines[i])
                i += 1
            break
        i += 1
    open(dst, "w", encoding="utf-8").write("data_x\nloop_\n" + "\n".join(hdr) + "\n" + "\n".join(data) + "\n")
    return dst


def _parse(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("#PDBchain1"): continue
        p = ln.split("\t")
        if len(p) >= 11:
            try: return {"tm1": float(p[2]), "tm2": float(p[3]), "rmsd": float(p[4]),
                         "id1": float(p[5]), "id2": float(p[6]), "idali": float(p[7]), "lali": int(p[10])}
            except ValueError: continue
    return None


def usalign(q, r, extra=""):
    if not (q and r and os.path.isfile(q) and os.path.isfile(r)): return None
    cmd = (f"{shlex.quote(_USEXE)} {shlex.quote(_wsl.translate_path(os.path.abspath(q)))} "
           f"{shlex.quote(_wsl.translate_path(os.path.abspath(r)))} -outfmt 2 {extra}").strip()
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 240))
    return _parse(res.get("stdout", "")) if res.get("ok") else None


def best_src(pdb, min_ca=60):
    """Prefer the ASU if it has >=2 long chains; else assembly1. Return (path, col, prot_chains, where)."""
    asu = fetch(pdb)
    if not asu: return None
    for where, path in (("ASU", asu), ("ASM", fetch(pdb, assembly=True))):
        if not path: continue
        prot = {c: n for c, n in chain_ca(path, "auth_asym_id").items() if n > min_ca}
        if len(prot) >= 2:
            return path, "auth_asym_id", prot, where
    # single-chain fallback (assembly may collapse to one auth id)
    return asu, "auth_asym_id", {c: n for c, n in chain_ca(asu).items() if n > min_ca}, "ASU"


print("══ SCOPING: NON-SWAPPED QUATERNARY-DIVERGENT CANDIDATES (no GPU) ══\n")
print("   GATE: dimer assembly-TM LOW (real headroom, like RNase A 0.385, not PCNA 0.90–0.99) + crystal.\n")

for title, confound, members in SYSTEMS:
    print(f"── {title} ──   confound: {confound}")
    info = {}
    for pdb, label in members:
        bs = best_src(pdb)
        if not bs:
            print(f"  {pdb}: download FAILED"); info[pdb] = None; continue
        path, col, prot, where = bs
        method = exptl_method(path)
        crystal = "X-RAY" in (method or "").upper()
        chains = sorted(prot.keys())
        anchor = chains[0]
        mate, contacts = (dimer_partner(path, anchor, col) if len(chains) >= 2 else (None, {}))
        pair = [anchor, mate] if mate else chains[:1]
        dimer = filter_chains(path, pair, os.path.join(TD, f"{pdb}_dim.cif"), col) if len(pair) == 2 else None
        mono = filter_chains(path, [anchor], os.path.join(TD, f"{pdb}_mono.cif"), col)
        full = filter_chains(path, chains, os.path.join(TD, f"{pdb}_full.cif"), col)
        info[pdb] = {"label": label, "method": method, "crystal": crystal, "chains": chains,
                     "pair": pair, "dimer": dimer, "mono": mono, "full": full, "where": where,
                     "contacts": contacts}
        print(f"  {pdb}  {method[:18]:<18} chains={len(chains)} ({where}) dimer={pair} "
              f"contacts={contacts if len(contacts)<=6 else '…'}  [{label}]")
    # pairwise divergence
    ok = [p for p, _ in members if info.get(p)]
    for a, b in itertools.combinations(ok, 2):
        ia, ib = info[a], info[b]
        ma = usalign(ia["mono"], ib["mono"])                       # monomer fold + seq-id
        da = usalign(ia["dimer"], ib["dimer"], extra="-mm 1") if (ia["dimer"] and ib["dimer"]) else None
        fa = usalign(ia["full"], ib["full"], extra="-mm 1")        # full assembly (handles diff counts)
        seqid = f"{ma['idali']*100:.0f}%" if ma else "?"
        monoTM = f"{ma['tm2']:.3f}" if ma else "?"
        dimTM = f"{da['tm2']:.3f}" if da else "—"
        fullTM = f"{fa['tm2']:.3f}" if fa else "—"
        crystal = ia["crystal"] and ib["crystal"]
        print(f"    {a} ⇄ {b}: seq-id={seqid} monoTM={monoTM} | DIMER-TM(-mm)={dimTM} "
              f"fullAsmTM(-mm)={fullTM}  chains {len(ia['chains'])}v{len(ib['chains'])}  "
              f"{'CRYSTAL' if crystal else '⚠NOT-ALL-CRYSTAL'}")
    print()

print("── ASSESSMENT (headroom gate) ──")
print("  Want: a NON-swap system whose DIMER-TM(-mm) sits in real headroom (~0.4–0.8) at FIXED chain")
print("  count, crystal, with ground truth — the cleanest carries the least confound. Report the ladder")
print("  + each confound; STOP for the framing call (no GPU until a divergent non-swap rung is confirmed).")
print("\nSTOP — non-swap scoping complete.")
