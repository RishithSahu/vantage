@echo off
rem Starts VehicleGPT and opens it in your browser. Uses the private environment made by Install-VehicleGPT.bat when it exists.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set PY=".venv\Scripts\python.exe") else (set PY=python)
%PY% vehiclegpt.py run %*
if errorlevel 1 (
  echo.
  echo VehicleGPT did not start. Run  Install-VehicleGPT.bat  once, or see TROUBLESHOOTING.md
  pause
)
