@echo off
REM Starts the student portal at http://localhost:8000. Close this window to stop it.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
set HF_HUB_OFFLINE=1
python server.py
