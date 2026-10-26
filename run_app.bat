@echo off
REM Starts the testing UI in your browser (http://localhost:8501). Close this window to stop it.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
set HF_HUB_OFFLINE=1
streamlit run app.py
