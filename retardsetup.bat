@echo off
title bonziPONY Setup
color 0d
echo.
echo  ============================================
echo     bonziPONY v1.69 - One-Click Setup
echo  ============================================
echo.

:: ── Find the script's own directory and cd into it ───────────
cd /d "%~dp0"

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Download it from https://www.python.org/downloads/
    echo  MAKE SURE you check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found.

:: ── Install uv if not available ──────────────────────────────
python -m uv --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Installing uv package manager...
    python -m pip install uv --quiet
)
python -m uv --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Failed to install uv. Run: pip install uv
    pause
    exit /b 1
)
echo  [OK] uv ready.

:: ── Create venv if it doesn't exist ──────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo.
    echo  Creating virtual environment...
    python -m uv venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)

set PY=venv\Scripts\python.exe

:: ── Install dependencies (hashes enforced) ───────────────────
echo.
echo  Installing dependencies with hash verification...
echo.
python -m uv pip install --require-hashes -r requirements-lock.txt --python venv\Scripts\python.exe
if errorlevel 1 goto :lockfail
goto :installdone

:lockfail
echo.
echo  [WARN] Lockfile install failed, trying loose requirements...
python -m uv pip install -r requirements.txt --python venv\Scripts\python.exe
if errorlevel 1 goto :installerror
goto :installdone

:installerror
echo.
echo  [ERROR] Dependency install failed. Check the errors above.
echo.
pause
exit /b 1

:installdone
echo.
echo  [OK] Dependencies installed.

:: ── Quick sanity check ───────────────────────────────────────
%PY% -c "import yaml" >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Core dependency missing. Install may have failed.
    pause
    exit /b 1
)

:: ── Copy config if needed ────────────────────────────────────
if not exist "config.yaml" (
    echo.
    echo  No config.yaml found, copying from example...
    copy config.yaml.example config.yaml >nul
    echo  [OK] config.yaml created.
    echo.
    echo  IMPORTANT: Edit config.yaml with your API keys
    echo  Or use the right-click menu after launch to set them.
    echo.
)

:: ── Create empty dirs ────────────────────────────────────────
if not exist "memory" mkdir memory
if not exist "diary" mkdir diary
if not exist "logs" mkdir logs

:: ── Launch ───────────────────────────────────────────────────
echo.
echo  ============================================
echo     Setup complete! Launching bonziPONY...
echo  ============================================
echo.
echo  Right-click the pony for settings.
echo  Double-click the pony to type a message.
echo  Close this window to kill the pony.
echo.

%PY% main.py

if errorlevel 1 (
    echo.
    echo  [ERROR] bonziPONY crashed. Check the error above.
    echo.
    pause
)
