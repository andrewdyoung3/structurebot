@echo off
REM StructureBot launcher — double-click (or the desktop shortcut) instead of
REM `python main.py`. Runs the GUI (which manages ChimeraX) from the project venv,
REM no matter what the current directory is. Any args are passed through, e.g.
REM   StructureBot.bat --resume
setlocal
cd /d "%~dp0"

set "PYEXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo Could not find the project virtualenv at:
  echo   "%PYEXE%"
  echo Create it first ^(python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt^),
  echo or edit this file to point PYEXE at the right interpreter.
  pause
  exit /b 1
)

"%PYEXE%" main.py %*
set "EXITCODE=%errorlevel%"

if not "%EXITCODE%"=="0" (
  echo.
  echo ================================================================
  echo StructureBot exited with error code %EXITCODE%.
  echo Read the messages above, then close this window.
  echo ================================================================
  pause
)
exit /b %EXITCODE%
