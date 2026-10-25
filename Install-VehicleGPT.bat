@echo off
rem One-time setup: makes a private Python environment in this folder and installs what VehicleGPT needs (needs internet once).
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is not installed. Get it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this again.
  pause
  exit /b 1
)
python -m venv .venv
if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo Installing the requirements failed - check your internet connection and run this again. & pause & exit /b 1 )
echo.
".venv\Scripts\python.exe" vehiclegpt.py doctor
echo.
echo If it says READY, start VehicleGPT with Start-VehicleGPT.bat
pause
