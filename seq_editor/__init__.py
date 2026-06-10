"""
seq_editor — StructureBot's standalone Sequence Viewer/Editor (PySide6).

The project's FIRST standalone GUI surface (see PROJECT_CONTEXT.md §9 PLAN).
A separate-process window linked to the SAME live ChimeraX over REST; an EXTERNAL
editor (distinct from the built-in ChimeraX Sequence Viewer that `sequence_viewer.py`
drives). MVP slice: read+display sequences, click→3D select, on-command reverse sync,
substitution→variant, fold the variant via ColabFold.

Logic/view split: `controller.SequenceEditorController` is pure Python (unit-tested
with ChimeraX/ColabFold mocked); `view`/`app` are the Qt layer.
"""
