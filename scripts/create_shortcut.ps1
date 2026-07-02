# create_shortcut.ps1
# --------------------
# Create a "StructureBot" shortcut (.lnk) on the Desktop (and optionally the Start
# Menu) that launches the app via StructureBot.bat - no terminal, just double-click.
# The shortcut runs MINIMIZED (the log console tucks into the taskbar; it pops back
# and pauses only if startup fails). Re-run any time to refresh it.
#
#   powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
#   ... -StartMenu     # also add it to the Start Menu
param([switch]$StartMenu)

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot        # ..\ (the repo root)
$batPath    = Join-Path $projectDir "StructureBot.bat"
$iconPath   = Join-Path $projectDir "StructureBot.ico"

if (-not (Test-Path $batPath))  { throw "Launcher not found: $batPath" }
if (-not (Test-Path $iconPath)) { Write-Warning "Icon not found: run python scripts\make_icon.py first" }

function New-SBShortcut([string]$linkPath) {
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($linkPath)
    $sc.TargetPath       = $batPath
    $sc.WorkingDirectory = $projectDir
    $sc.WindowStyle      = 7                            # 7 = minimized
    $sc.Description       = "StructureBot - NL interface for UCSF ChimeraX"
    if (Test-Path $iconPath) { $sc.IconLocation = "$iconPath,0" }
    $sc.Save()
    Write-Host "Created shortcut: $linkPath"
}

$desktop = [Environment]::GetFolderPath("Desktop")
New-SBShortcut (Join-Path $desktop "StructureBot.lnk")

if ($StartMenu) {
    $programs = [Environment]::GetFolderPath("Programs")
    New-SBShortcut (Join-Path $programs "StructureBot.lnk")
}
