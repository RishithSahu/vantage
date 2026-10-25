@echo off
rem VehicleGPT. Opens it in your browser (local only, no network use). Same as Start-VehicleGPT.bat.
rem   start_ui.bat                 recorded dashcam clip + language model if installed
rem   start_ui.bat --camera webcam start on your webcam
rem   start_ui.bat --model none    rules only
cd /d "%~dp0"
python vehiclegpt.py run %*
