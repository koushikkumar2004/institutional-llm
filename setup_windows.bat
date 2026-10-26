@echo off
REM One-time setup on Windows: creates .venv, installs packages, downloads models, builds the index.
cd /d "%~dp0"

echo.
echo === Creating virtual environment (.venv)
py -3.12 -m venv .venv 2>nul
if errorlevel 1 (
    echo Python 3.12 not found through the "py" launcher, trying the default python...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment.
        echo Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo.
echo === Installing packages (first time takes a few minutes)
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package installation failed. See "Troubleshooting" in README.md.
    pause
    exit /b 1
)

echo.
echo === Downloading models, building the index, checking Ollama
python scripts\prepare.py

echo.
echo Setup finished. Start the app any time with run_app.bat
pause
