# ChimeraX Command Reference — StructureBot

Injected verbatim into every translation request. Optimised for accuracy.

---

## ⚠ CRITICAL TRANSLATION RULES

These are the most common mistakes. Always apply these first.

| Natural language | CORRECT command | WRONG (do not use) |
|---|---|---|
| "color each chain a different color" | `color bychain` or `color #1 bychain` | `color bychain #1`, `color #1 rainbow`, `color bychain palette` |
| "show as ribbon/cartoon" | `cartoon #N` | `show ribbon`, `display cartoon` |
| "show the ligand as spheres" | `style :LIGNAME sphere` | `sphere :LIGNAME`, `display :LIGNAME sphere` |
| "show residues near the ligand" | `select zone :LIGNAME DIST` then `show sel atoms` | `nearby :LIGNAME` |
| "color by element" | `color #N byelement` | `color byelement #N`, `cpk` |
| "white/black background" | `set bgColor white` / `set bgColor black` | `background color white`, `background white` |
| "publication lighting" | `lighting soft` | `lighting preset publication` (preset doesn't exist) |
| "save image to Desktop" | `save "C:/Users/USERNAME/Desktop/file.png"` | backslash paths |
| "remove/delete a structure" | `close #N` | `delete #N`, `remove #N` |
| "reset the view" | `view` | `reset`, `center` |
| "show the surface" | `surface #N` | `show surface`, `solvent surface` |

**`color by*` UNIVERSAL RULE**: The selector ALWAYS comes before the `by*` keyword.
`color <selector> by<keyword>` ✓ — `color by<keyword> <selector>` ✗ (gives "Expected a collection" error).
This applies to ALL `by*` color keywords: `bychain`, `byelement`, `bypolymer`, `byhetero`, `bymodel`, etc.
When no model specifier is needed, omit it entirely: `color bychain` ✓

**LIGAND NAMES**: Always use the exact 3-letter residue code from session state.
If session state shows `Ligands: MK1`, use `:MK1` — never `:LIG` or `:ligand` as a residue name.

**WINDOWS PATHS**: ChimeraX save commands need forward slashes:
```
save "C:/Users/andre/Desktop/figure.png" width 1920 height 1080
```

---

## MODEL SPECIFIERS

```
#1          entire model 1
#1,2        models 1 and 2
#1-3        models 1 through 3
#1/A        chain A of model 1
#1/A,B      chains A and B of model 1
```

---

## ATOM SPECIFIER (selection) SYNTAX

Atom specs are used in most commands: `command atomspec options`

```
:HIS            all histidine residues in all models
:HIS/A          histidines in chain A
:100            residue 100
:100-150        residues 100 to 150
:MK1            residue named MK1 (exact 3-letter code required)
@@name=CA       alpha carbons
#1 & protein    protein atoms in model 1
#1 & ligand     ligand atoms in model 1 (keyword, not residue name)
#1 & solvent    water molecules
#1 & ~protein   everything in model 1 that is NOT protein
```

### Zone selection (residues within a distance)

```
# Select all atoms within 4 Å of MK1:
select zone :MK1 4.0

# Select and show protein atoms within 5 Å of ligand:
select zone :MK1 5.0 & protein
show sel atoms
style sel stick
color sel byelement

# Zone with residues true — selects whole residues (not just nearby atoms):
select zone :MK1 4.0
```

### Boolean operators

```
#1 & :HIS           AND  — model 1 AND histidine
#1 | #2             OR   — model 1 OR model 2
~#1                 NOT  — everything except model 1
sel                 current selection
```

---

## FILE I/O

```
open 1HSG                       # fetch from RCSB by PDB ID (4-letter code)
open 1HSG from alphafold        # fetch from AlphaFold DB
open myprotein.pdb              # open local PDB file
open myligand.mol2              # open local mol2/SDF ligand
close #1                        # close model 1
close all                       # close everything
save session.cxs                # save full ChimeraX session
open session.cxs                # restore session
```

---

## SHOW / HIDE / STYLE

```
# Show/hide atoms or ribbons
show #1 atoms           # show all atoms in model 1
show #1 ribbon          # show ribbon only
hide #1 atoms           # hide atoms
hide #1 ribbon          # hide ribbon
show :MK1 atoms         # show ligand atoms (use actual residue name)

# Representation styles (apply to atom selections)
cartoon #1              # ribbon/cartoon for whole model
style #1 stick          # stick bonds
style #1 sphere         # space-filling spheres
style #1 ball           # ball-and-stick
style :MK1 sphere       # ligand as space-filling (use actual residue name)
style :MK1 stick        # ligand as sticks
nucleotides slab #1     # nucleotide representation

# Surfaces
surface #1              # solvent-accessible surface for model 1
surface #1 enclose #1   # van der Waals surface
transparency #1 50      # 50% transparent surface (0=opaque, 100=invisible)
surface style #1 mesh   # mesh instead of solid surface
```

---

## COLORS

```
color #1 red                    # named color
color #1 #3399ff                # hex color
color bychain                   # DIFFERENT COLOR PER CHAIN (all models)
color #1 bychain                # per-chain coloring for model 1 only  ← selector BEFORE keyword
color #1 byelement              # CPK element colors (C=gray, N=blue, O=red…)
color :MK1 byelement            # element colors for ligand (use actual name)
color #1 bfactor palette blue:white:red   # B-factor heatmap
color #1 rainbow                # N→C rainbow (single chain)
rainbow chain #1                # rainbow per chain N→C

# Secondary structure coloring
color #1 :helix dodger blue
color #1 :strand goldenrod
color #1 :coil gray60

# Surface electrostatics / hydrophobicity
coulombic #1 surfaces True      # electrostatic potential mapped to surface
mlp #1                          # molecular lipophilicity (hydrophobicity)
```

---

## ANALYSIS

```
# Distances and geometry
distance #1:50@CA #1:100@CA     # distance between two atoms
angle #1:50@N #1:50@CA #1:50@C
dihedral #1:50@N #1:50@CA #1:50@C #1:50@O

# Contacts and hydrogen bonds
hbonds #1 reveal true           # detect & display H-bonds
hbonds :MK1 reveal true         # H-bonds to/from ligand
contacts #1 #2 restrict any     # contacts between two models
clashes #1                      # steric clashes within model 1

# Surfaces
area #1 protein                 # SASA of protein
volume #1                       # enclosed volume (requires surface)
```

---

## STRUCTURAL ALIGNMENT

```
matchmaker #1 to #2             # sequence-aware alignment (#1 moves to #2)
matchmaker #1 to #2 alwaysSucceed true showAlignment true
align #1 to #2                  # structure-based (no sequence needed)
rmsd #1 #2 pairedOnly true      # RMSD after alignment
```

---

## CAMERA & VIEW

```
view                            # fit all models in window (always add after big changes)
view #1                         # focus on model 1
zoom 1.5                        # zoom in
cofr :MK1                       # center rotation on ligand (use actual name)
set bgColor white               # white background  (NOT "background color white")
set bgColor black               # black background
set bgColor transparent         # transparent background (for compositing)
lighting soft                   # soft shadows — good default
lighting gentle                 # gentler soft shadows
lighting full                   # full ambient lighting
# Valid lighting preset names: default, flat, full, gentle, simple, soft
# INVALID: "lighting preset publication" — that preset does not exist
camera ortho                    # orthographic projection
camera perspective              # perspective projection
turn y 45                       # rotate 45° around Y axis
```

---

## IMAGE & SESSION EXPORT

### Saving images — ALWAYS use forward slashes on Windows

```
# Desktop
save "C:/Users/andre/Desktop/figure.png" width 1920 height 1080

# Current directory
save figure.png width 2400 height 2400 supersample 3

# Transparent background
save figure.png transparentBackground true width 3000 height 3000

# Full path with spaces — use double quotes
save "C:/Users/andre/Desktop/my figure.png" width 1920 height 1080
```

### Publication preset sequence (always in this order)

```
preset publication
graphics silhouettes true width 2
set bgColor white
lighting soft
save "C:/Users/USERNAME/Desktop/pub_figure.png" width 3000 height 3000 supersample 3
```

### Sessions

```
save "sessions/my_session.cxs"
open "sessions/my_session.cxs"
```

---

## WORKING EXAMPLES BY TASK

### Open and display

```
open 1HSG
cartoon #1
color bychain
view
```

### Ligand visualization (MK1 example — use ACTUAL residue name from session)

```
show :MK1 atoms
style :MK1 sphere
color :MK1 byelement
view :MK1
```

### Binding pocket within 4 Å

```
select zone :MK1 4.0
show sel atoms
style sel stick
color sel byelement
label sel residues text {1-letter}{number}
view :MK1
```

### Hydrogen bond network to ligand

```
show :MK1 atoms
style :MK1 stick
hbonds :MK1 reveal true
view :MK1
```

### Color by secondary structure

```
cartoon #1
color #1 :helix dodger blue
color #1 :strand goldenrod
color #1 :coil gray60
view
```

### Compare two structures

```
open 1AKE
open 4AKE
matchmaker #1 to #2 alwaysSucceed true
rmsd #1 #2 pairedOnly true
rainbow chain #1
rainbow chain #2
view
```

### Electrostatic surface

```
surface #1
coulombic #1 surfaces True
view
```

### Publication image

```
cartoon #1
color bychain
preset publication
graphics silhouettes true width 2
set bgColor white
lighting soft
view
save "C:/Users/andre/Desktop/figure.png" width 3000 height 3000 supersample 3
```
