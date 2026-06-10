"""Enable `python -m seq_editor` to launch the standalone Sequence Editor."""
from seq_editor.app import main

if __name__ == "__main__":
    raise SystemExit(main())
