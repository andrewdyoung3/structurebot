#!/bin/bash
# Run from project root inside the main venv
# Regenerates PROJECT_CONTEXT.md from the live codebase
echo "Regenerating PROJECT_CONTEXT.md..."
claude "Read PROJECT_CONTEXT.md for regeneration instructions, then regenerate it in full by reading the entire codebase. Preserve the Changelog section and append a new entry."
