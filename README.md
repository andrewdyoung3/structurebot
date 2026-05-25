# StructureBot

**Natural Language Interface for UCSF ChimeraX**

Type requests in plain English; StructureBot translates them into ChimeraX
commands using Claude, previews them, and executes them automatically.

---

## Quick start

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 2. Verify ChimeraX connectivity (no app launch required)
python test_bridge.py

# 3. If ChimeraX is not already open, launch it and enable the REST server
python test_bridge.py --start

# 4. Run the app
python main.py
python main.py --resume          # restore last session
python main.py --no-auto-proceed # always confirm before executing
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ (venv at `.\venv`) | `pip install -r requirements.txt` |
| UCSF ChimeraX 1.x | Install at default path or set `CHIMERAX_PATH` |
| Anthropic API key | Already in `.env.local` |
| Port 60001 free | REST server default |

---

## Configuration

Copy `.env.example` to `.env.local` and fill in your values.
`.env.local` is loaded automatically — no shell export needed.

Key settings (all optional — defaults work out of the box):

| Variable | Default | Description |
|---|---|---|
| `CHIMERAX_PATH` | `C:\Users\andre\documents\ChimeraX 1.11.1\bin\ChimeraX.exe` | ChimeraX executable |
| `CHIMERAX_HOST` | `127.0.0.1` | REST server host |
| `CHIMERAX_PORT` | `60001` | REST server port |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model for translation |
| `AUTO_PROCEED_DELAY` | `2` | Seconds before auto-executing (0 = always confirm) |
| `MAX_CONVERSATION_HISTORY` | `6` | Rolling turns kept per session |

---

## Manual ChimeraX REST server setup

If you prefer to open ChimeraX yourself rather than having StructureBot launch it:

1. Open ChimeraX
2. In the ChimeraX command bar, type:
   ```
   remotecontrol rest start port 60001
   ```
3. Leave ChimeraX open, then run `python main.py`

---

## Example prompts

### Loading & display

```
Open 1HSG and show it as a ribbon diagram
Load myprotein.pdb and display as ball-and-stick
Open 1AKE and 4AKE
```

### Coloring

```
Color each chain a different color
Color by secondary structure — helices blue, strands gold, loops gray
Color by B-factor using a blue-white-red heatmap
Show the electrostatic surface
Color the protein by hydrophobicity
```

### Ligand analysis

```
Show the ligand as spheres and color it by element
Find all residues within 4 angstroms of the ligand
Show the hydrogen bonds between the ligand and protein
Label the binding pocket residues
Color the binding site surface by hydrophobicity
```

### Structural comparison

```
Align 1AKE onto 4AKE and show the RMSD
Compare the active sites of 1AKE and 4AKE
Show both structures colored differently after alignment
```

### Image export

```
Save a publication-quality image to my desktop as figure1.png
Save the current view as figure.png at 3000x3000 with transparent background
```

---

## Built-in commands (no LLM)

| Command | Action |
|---|---|
| `history` | Show last 15 commands with natural language origin |
| `state` | Dump current session state (loaded models, ligand names, etc.) |
| `undo` | Send undo to ChimeraX + remove last history entry |
| `clear` | Close all models, reset session state |
| `reset` | Clear conversation context (keeps loaded structures) |
| `save session NAME` | Save ChimeraX .cxs + state JSON to `sessions/NAME` |
| `load session NAME` | Restore a saved session |
| `help` | Show categorised example prompts |
| `quit` / `exit` | Save session and exit |

---

## Architecture

```
structurebot/
├── main.py               # REPL loop, Rich UI, startup, special commands, logging
├── config.py             # All constants; loads .env.local overrides
├── chimerax_bridge.py    # REST client: GET /run?command= to ChimeraX
├── translator.py         # Anthropic API: NL → ChimeraX commands (JSON output)
│                         #   • Two-block system prompt with prompt caching
│                         #   • JSON retry on parse failure
│                         #   • Auto-fix on command execution error
├── session_state.py      # Loaded structures, selections, history
│                         #   • parse_pdb_header() for local .pdb files
│                         #   • fetch_rcsb_metadata() for PDB IDs (gets ligand codes)
├── chimerax_commands.md  # Curated reference injected into every API call
├── test_bridge.py        # Standalone ChimeraX connectivity test (run first)
├── tests/
│   └── test_integration.py  # End-to-end 1HSG workflow test
├── logs/                 # session_YYYYMMDD_HHMMSS.jsonl (auto-created)
├── sessions/             # Saved sessions: NAME.cxs + NAME.json (auto-created)
├── requirements.txt
├── .env.local            # Your API key and overrides (not committed)
└── .env.example          # Template for .env.local
```

### Request flow

```
User types request
    ↓
translator.py
  • Assembles two-block system prompt:
      Block 1 (CACHED)   — role + rules + chimerax_commands.md
      Block 2 (UNCACHED) — current session state (models, ligand names, history)
  • Calls Anthropic API (claude-sonnet-4-6)
  • Returns structured JSON: {commands, explanations, warnings, confidence}
    ↓
main.py
  • Shows preview table with confidence level
  • High/medium: 2 s countdown then auto-execute
  • Low: always prompts for confirmation
    ↓
chimerax_bridge.py
  • GET http://127.0.0.1:60001/run?command=<urlencoded>
  • Detects errors via response text prefixes (ChimeraX returns 200 for errors)
    ↓
ChimeraX executes; results shown in terminal
    ↓
session_state.py updated (structure list, history)
logs/session_*.jsonl appended
```

---

## Testing

```powershell
# 1. Verify ChimeraX connection (safe, read-only)
python test_bridge.py
python test_bridge.py --start   # also launch ChimeraX

# 2. Full connectivity test (opens/closes 1AON from RCSB)
python test_bridge.py --full

# 3. Translation integration test (no ChimeraX needed)
python tests/test_integration.py

# 4. Full end-to-end with execution (ChimeraX must be running)
python tests/test_integration.py --execute

# 5. Full end-to-end, auto-launching ChimeraX
python tests/test_integration.py --start
```

---

## Troubleshooting

**"ChimeraX not found"**
```powershell
$env:CHIMERAX_PATH = "C:\full\path\to\ChimeraX.exe"
python main.py
```

**"ChimeraX REST server not reachable"**
- In ChimeraX command bar: `remotecontrol rest start port 60001`
- Or: `python test_bridge.py --start`

**Ligand name wrong in generated commands**
- Run `state` to see what session knows about the loaded structure
- RCSB lookup runs automatically when a PDB ID is opened (needs internet)
- Specify the residue name explicitly: "show the MK1 ligand as spheres"

**"ANTHROPIC_API_KEY not set"**
- Check `.env.local` contains `ANTHROPIC_API_KEY=sk-ant-...`
- Or: `$env:ANTHROPIC_API_KEY = "sk-ant-..."`

**Long sessions / token cost**
- Prompt caching is active: the large static system block (rules + command reference)
  is cached after the first API call in each session, dramatically reducing input cost.
- Rolling history is trimmed to `MAX_CONVERSATION_HISTORY` (default 6 pairs).
- Use `reset` to clear context when switching to an unrelated task.
