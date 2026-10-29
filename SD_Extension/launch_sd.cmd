@echo off
cd /d "%~dp0"
if exist "%~dp0python.local.cmd" call "%~dp0python.local.cmd"
if defined DEHAZE_PYTHON (
  start "" "%DEHAZE_PYTHON%" -B "%~dp0open_sd_demo.py"
  exit /b
)
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" -B "%~dp0open_sd_demo.py"
  exit /b
)
where pythonw >nul 2>nul
if not errorlevel 1 (
  start "" pythonw -B "%~dp0open_sd_demo.py"
) else (
  python -B "%~dp0open_sd_demo.py"
  if errorlevel 1 pause
)
