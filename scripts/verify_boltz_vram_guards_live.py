"""
Live-verify the Boltz PRE-FLIGHT GPU GUARDS against the REAL boltz_env + real GPU (RTX 5070 Ti
Laptop, 12 GB) over WSL. The gate: a fold too large for this GPU's free VRAM now FAILS IN SECONDS
with an honest message instead of thrashing for hours (the silent VRAM-overcommit death that left
PID 321 spinning 2 h with zero output).

Checks (real nvidia-smi / real boltz):
  1. SIZE GUARD: a 1084-residue fold (503+581, the assembly that crashed) is REFUSED in <15 s with
     the informative "needs ~X GB … too large to fold safely" message — and NO boltz subprocess is
     spawned (no thrash).
  2. MONOMER still folds: an 80-residue fold passes the guard and actually completes (real GPU fold).
  3. CONCURRENCY GUARD: with a real fold in flight, a second fold is REFUSED with the specific
     "Another Boltz fold is already using the GPU" message (distinct from the size message).

Run: venv/Scripts/python.exe scripts/verify_boltz_vram_guards_live.py   (needs WSL boltz_env + GPU)
"""
import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from boltz_bridge import BoltzBridge

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    b = BoltzBridge()
    if not b._wsl.is_available():
        print("WSL not available — cannot live-verify."); return 1
    free0 = b._free_vram_mib()
    print(f"GPU free VRAM at start: {free0} MiB  (running folds: {b._running_fold_pids()})")
    if free0 is None:
        print("nvidia-smi not readable — cannot live-verify the VRAM guard."); return 1
    if free0 < 6000:
        print(f"Only {free0} MiB free — another process holds the GPU; free it first."); return 1

    # 1) SIZE GUARD: the doomed 1084-res fold fails FAST, no subprocess ───────────────────
    print("\n1) Size guard refuses the 1084-residue assembly (fast, no thrash)…")
    chains_big = [{"id": "A", "sequence": "M" * 503}, {"id": "B", "sequence": "M" * 581}]
    print(f"   estimated need: {b.estimated_vram_mib(1084)} MiB ({b.estimated_vram_mib(1084)/1024:.1f} GB) "
          f"vs {free0} MiB free")
    t0 = time.time()
    res = b.predict(chains_big, label="big")
    dt = time.time() - t0
    check("1084-res fold refused in <15 s (no thrash)", (not res["success"]) and dt < 15,
          f"{dt:.1f}s")
    check("message is the informative size guard",
          "1084-residue" in (res["error"] or "") and "too large to fold safely" in (res["error"] or ""),
          res["error"])
    check("GPU stayed free (no fold subprocess spawned)",
          (b._free_vram_mib() or 0) >= free0 - 500 and not b._running_fold_pids())

    # 2) MONOMER still folds (real GPU fold) ──────────────────────────────────────────────
    print("\n2) Monomer (80 res) passes the guard and actually folds…")
    t0 = time.time()
    res_m = b.predict([{"id": "A", "sequence": "MKVLWAAGTDERCFHINPQSYKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTV"}],
                      label="mono")
    dt = time.time() - t0
    check("monomer folded successfully", res_m.get("success") is True,
          f"{dt:.0f}s, plddt={res_m.get('mean_plddt')}, err={res_m.get('error')}")

    # 3) CONCURRENCY GUARD: refuse a 2nd fold while one runs ───────────────────────────────
    print("\n3) Concurrency guard refuses a 2nd fold while one is in flight…")
    # launch a real (mid-size) fold in the background via WSL, wait until pgrep sees it
    import subprocess, tempfile
    work = tempfile.mkdtemp(prefix="boltz_concbg_")
    (Path(work) / "in.yaml").write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: "
        + "M" * 300 + "\n      msa: empty\n")
    out_wsl = b._wsl.translate_path(work)
    # Hold the WSL session open with a NON-WAITING Popen (how the real app launches a fold via
    # wsl_bridge) so the boltz child stays alive while we attempt the second fold.
    bg_inner = (f"source ~/boltz_env/bin/activate; {b._exe} predict {out_wsl}/in.yaml "
                f"--out_dir {out_wsl}/out --accelerator gpu --no_kernels --override --seed 0 "
                f"--recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 > {out_wsl}/bg.log 2>&1")
    bg = subprocess.Popen(["wsl.exe", "--distribution", b._distro, "--exec", "bash", "-lc", bg_inner])
    try:
        seen = False
        for _ in range(30):                        # up to ~60 s for model load + GPU grab
            time.sleep(2)
            if b._running_fold_pids():
                seen = True; break
        print(f"   background fold detected: {seen} (pids={b._running_fold_pids()})")
        if seen:
            res_c = b.predict([{"id": "A", "sequence": "M" * 80}], label="second")
            check("2nd fold refused with the BUSY message (not 'too large')",
                  (not res_c["success"]) and "Another Boltz fold is already using the GPU" in (res_c["error"] or "")
                  and "too large" not in (res_c["error"] or ""),
                  res_c["error"])
        else:
            check("background fold detected for concurrency test", False, "bg fold never appeared")
    finally:
        bg.terminate()
        b._wsl.run_command("pkill -9 -f '[b]oltz predict' || true", timeout=20)
    import shutil; shutil.rmtree(work, ignore_errors=True)
    time.sleep(2)
    print(f"   GPU free after teardown: {b._free_vram_mib()} MiB")

    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
