"""
verify_foldseek_lowbucket_live.py — LIVE (GPU-free) check of the foldseek-picker refinements.

Runs a REAL foldseek search (LOCAL-ONLY WSL binary, CPU) on real cache structures, then builds the
REAL Qt picker dialog offscreen (only the modal exec() loop is stubbed — the widgets are really
constructed and really inspected, so the actual Qt wiring is exercised, not render-mocked). Confirms:
  1. the assembly-variant caveat ALWAYS shows (both with and without a low bucket);
  2. the "show lower-confidence hits" expander appears ONLY when the low bucket is non-empty
     (present for a query with [0.20,0.30) hits; OMITTED for one without);
  3. a low-bucket pick flows through _foldseek_refs -> construct_fold_guided_spec (real seam).

LOCAL-ONLY: foldseek easy-search vs the pre-downloaded DB; no network, no GPU.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock

from PySide6 import QtWidgets, QtCore
from foldseek_bridge import FoldseekBridge
from variant_workbench import VariantWorkbenchPanel

SEQ = ("KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYST"
       "MSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def cif(name): return os.path.join(ROOT, "cache", name)


def _panel():
    return VariantWorkbenchPanel(MagicMock(), session=None, pool=MagicMock())


def _render_dialog(panel, hits, low, tick_pid=None):
    """Build the REAL dialog; stub only QDialog.exec to inspect widgets, optionally tick a pid, and
    accept (or reject if tick_pid is None). Returns (picked, labels, toggles)."""
    cap = {}
    def fake_exec(self_dlg):
        cap["labels"] = [w.text() for w in self_dlg.findChildren(QtWidgets.QLabel)]
        cap["toggles"] = [w.text() for w in self_dlg.findChildren(QtWidgets.QToolButton)]
        if tick_pid is not None:
            for lw in self_dlg.findChildren(QtWidgets.QListWidget):
                for i in range(lw.count()):
                    if lw.item(i).data(QtCore.Qt.UserRole) == tick_pid:
                        lw.item(i).setCheckState(QtCore.Qt.Checked)
            return QtWidgets.QDialog.Accepted
        return QtWidgets.QDialog.Rejected
    orig = QtWidgets.QDialog.exec
    QtWidgets.QDialog.exec = fake_exec
    try:
        picked = panel._foldseek_pick_dialog(hits, low, 90.0, "PDB snapshot 2025-01 (local foldseek DB)")
    finally:
        QtWidgets.QDialog.exec = orig
    return picked, cap.get("labels", []), cap.get("toggles", [])


def main():
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    fb = FoldseekBridge()
    print("== LIVE foldseek-picker low-bucket (real foldseek, GPU-free) ==")
    print("  foldseek available:", fb.is_available(), "|", fb.db_label(), "\n")
    if not fb.is_available():
        print("FOLDSEEK UNAVAILABLE — cannot live-verify."); sys.exit(2)

    ok = True

    # --- Real searches: one query WITH a low bucket, one WITHOUT --------------------------------
    q_with = cif("1A2W.cif")     # RNase — many ribonuclease neighbours + a populated [0.20,0.30) band
    q_without = cif("2ACY.cif")  # acylphosphatase — neighbours all >= 0.30 (empty low bucket)
    primary_w, low_w = fb.search_neighbors(q_with, with_low_bucket=True, low_bound=0.2)
    primary_o, low_o = fb.search_neighbors(q_without, with_low_bucket=True, low_bound=0.2)
    print(f"  1A2W: primary={len(primary_w)} low={len(low_w)}  low_sample={low_w[:2]}")
    print(f"  2ACY: primary={len(primary_o)} low={len(low_o)}\n")
    band_ok = bool(low_w) and all(0.2 <= h[2] < 0.3 for h in low_w) and not low_o
    print(f"  [bucket] real low band populated for 1A2W, empty for 2ACY -> {'PASS' if band_ok else 'FAIL'}")
    ok = ok and band_ok

    p = _panel()

    # --- CASE A: low bucket present -> caveat shown, expander present ----------------------------
    _, labels_a, toggles_a = _render_dialog(p, primary_w, low_w, tick_pid=None)
    j_a = " ".join(labels_a).lower()
    caveat_a = "single-chain fold homologs" in j_a and "fold family" in j_a
    expander_a = any("lower-confidence hits" in t.lower() for t in toggles_a)
    note_a = "lower-similarity neighbours" in j_a and "not a recommendation" in j_a
    print(f"  [A:low present ] assembly caveat shown -> {'PASS' if caveat_a else 'FAIL'}")
    print(f"  [A:low present ] expander toggle present -> {'PASS' if expander_a else 'FAIL'}")
    print(f"  [A:low present ] low-bucket honesty note shown -> {'PASS' if note_a else 'FAIL'}")
    ok = ok and caveat_a and expander_a and note_a

    # --- CASE B: low bucket empty -> caveat STILL shown, expander OMITTED -----------------------
    _, labels_b, toggles_b = _render_dialog(p, primary_o, low_o, tick_pid=None)
    j_b = " ".join(labels_b).lower()
    caveat_b = "single-chain fold homologs" in j_b and "fold family" in j_b
    no_expander_b = (toggles_b == []) and ("lower-similarity neighbours" not in j_b)
    print(f"  [B:low empty   ] assembly caveat STILL shown -> {'PASS' if caveat_b else 'FAIL'}")
    print(f"  [B:low empty   ] expander OMITTED -> {'PASS' if no_expander_b else 'FAIL'}")
    ok = ok and caveat_b and no_expander_b

    # --- CASE C: a low-bucket pick flows through the full seam to a guided spec ------------------
    p.launchRequested.disconnect() if False else None
    p._add_sequence_construct("x", SEQ)
    cd = next(iter(p._design.chains.values()))
    low_pid = low_w[0][0]
    emitted = []
    p.launchRequested.connect(lambda spec: emitted.append(spec))

    # Drive the real callback (_on_foldseek_hits) with REAL hits; tick the low-bucket pid in the
    # real dialog via the exec stub, then let the seam build construct_fold_guided_spec.
    def fake_exec_pick(self_dlg):
        for lw in self_dlg.findChildren(QtWidgets.QListWidget):
            for i in range(lw.count()):
                if lw.item(i).data(QtCore.Qt.UserRole) == low_pid:
                    lw.item(i).setCheckState(QtCore.Qt.Checked)
        return QtWidgets.QDialog.Accepted
    orig = QtWidgets.QDialog.exec
    QtWidgets.QDialog.exec = fake_exec_pick
    try:
        p._on_foldseek_hits(primary_w, low_w, cd, "boltz", 1, 90.0, fb.db_label())
    finally:
        QtWidgets.QDialog.exec = orig

    seam_ok = (len(emitted) == 1
               and any(t.get("pdb_id") == low_pid
                       for t in emitted[0]["tool_inputs"].get("templates", [])))
    print(f"  [C:seam        ] low-bucket pick {low_pid} -> guided spec templates -> "
          f"{'PASS' if seam_ok else 'FAIL'}")
    ok = ok and seam_ok

    print("\n== RESULT:", "ALL PASS" if ok else "FAILED", "==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
