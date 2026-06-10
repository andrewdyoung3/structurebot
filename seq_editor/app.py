"""
seq_editor.app — entry point. Wires the REAL ChimeraX REST + ColabFold bridges into
the controller and launches the PySide6 window in its own QApplication event loop
(separate process — no conflict with the REPL's loop).

Launch:  python -m seq_editor            (uses config.REST_PORT)
         python -m seq_editor --port N
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_controller(port: int):
    """Build the controller backed by the real bridges. Importing here (not at module
    top) keeps the controller unit tests free of Qt/bridge import cost."""
    import config
    from chimerax_bridge import ChimeraXBridge
    from colabfold_bridge import ColabFoldBridge
    from seq_editor.controller import SequenceEditorController

    bridge = ChimeraXBridge(port=port or config.REST_PORT)
    fold = ColabFoldBridge()
    return SequenceEditorController(bridge.run_command, fold.predict)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="StructureBot standalone Sequence Editor (MVP)")
    ap.add_argument("--port", type=int, default=0,
                    help="ChimeraX REST port (default: config.REST_PORT / 60001)")
    a = ap.parse_args(argv)

    from PySide6 import QtWidgets
    from seq_editor.view import SequenceEditorWindow

    controller = _make_controller(a.port)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = SequenceEditorWindow(controller)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
