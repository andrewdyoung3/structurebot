"""
make_icon.py
------------
Generate StructureBot.ico (project root) — a simple, distinct app icon for the
desktop shortcut: a dark rounded tile with a small 3-atom "molecule" motif so it
reads as a structure viewer and is visually distinct from the ChimeraX icon.

Idempotent: re-run any time to regenerate. Needs Pillow (already a venv dep).
    python scripts/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

_OUT = Path(__file__).resolve().parent.parent / "StructureBot.ico"
_BG      = (22, 32, 44)       # deep slate (matches the GUI's dark pane)
_TILE    = (30, 46, 64)       # slightly lighter tile
_BOND    = (120, 150, 170)
_ATOMS   = [((0.34, 0.34), (79, 209, 255)),    # cyan
            ((0.66, 0.40), (255, 120, 180)),   # magenta
            ((0.48, 0.68), (120, 220, 140))]   # green


def _render(px: int) -> Image.Image:
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = px * 0.06
    d.rounded_rectangle([m, m, px - m, px - m], radius=px * 0.22, fill=_TILE)
    # bonds first (under the atoms)
    pts = [(cx * px, cy * px) for (cx, cy), _ in _ATOMS]
    w = max(2, int(px * 0.045))
    for i in range(len(pts)):
        d.line([pts[i], pts[(i + 1) % len(pts)]], fill=_BOND, width=w)
    # atoms
    r = px * 0.13
    for (cx, cy), col in _ATOMS:
        x, y = cx * px, cy * px
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return img


def main() -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    base = _render(256)
    base.save(_OUT, format="ICO",
              sizes=[(s, s) for s in sizes])
    print(f"wrote {_OUT}  ({', '.join(str(s) for s in sizes)} px)")


if __name__ == "__main__":
    main()
